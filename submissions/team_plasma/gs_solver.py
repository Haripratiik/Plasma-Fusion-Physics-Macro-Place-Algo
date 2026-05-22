"""
Grad-Shafranov equation solver.

Solves

    Delta-star(psi) = -mu0_eff * R^2 * p'(psi) - F(psi) * F'(psi) + coil_source

with Dirichlet `psi = 0` on the canvas boundary, via damped Picard iteration
on the nonlinear right-hand side and red-black Gauss-Seidel sweeps on each
inner linear solve.

The Grad-Shafranov operator under axisymmetric cylindrical geometry is

    Delta-star(psi) = d^2 psi / dR^2 - (1/R) * d psi / dR + d^2 psi / dZ^2.

The `-(1/R) d/dR` term is what makes Delta-star differ from the planar
Laplacian; it comes from the cylindrical Jacobian and the `1/R` in
`B_pol = grad(psi) x e_phi / R`.

See docs/DERIVATION.md sections 1.3-1.4 and 3 for the math.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, List, Optional, Tuple

import torch

from geometry import CylindricalGeometry


@dataclass
class GSSolverConfig:
    """Numerical settings for the Grad-Shafranov solver."""
    picard_outer: int = 30
    gs_inner_sweeps: int = 40
    omega_sor: float = 1.2          # SOR relaxation parameter in [1, 2)
    omega_picard: float = 0.7       # Outer-Picard damping
    convergence_tol: float = 1e-4
    mu0_eff: float = 1.0
    use_newton_solver: bool = False
    newton_outer_max: int = 6
    newton_tol_residual: float = 1e-4
    newton_tol_step: float = 1e-3
    newton_armijo_c1: float = 0.5
    newton_min_alpha: float = 1.0 / 64.0
    # Defensive numerical clamps
    psi_clamp: float = 1e6
    rhs_clamp: float = 1e6


@dataclass
class ConvergenceTrace:
    """Records per-iteration residuals for diagnostic and debugging."""
    residuals: List[float]
    converged: bool

    def final_residual(self) -> float:
        return float(self.residuals[-1]) if self.residuals else float("inf")


def delta_star_apply(psi: torch.Tensor, geom: CylindricalGeometry) -> torch.Tensor:
    """Apply the Grad-Shafranov operator to `psi`.

        (Delta-star psi)_{i,j} =
              (psi_{i,j+1} - 2 psi_{i,j} + psi_{i,j-1}) / dR^2
            - (psi_{i,j+1} - psi_{i,j-1}) / (2 * R_j * dR)
            + (psi_{i+1,j} - 2 psi_{i,j} + psi_{i-1,j}) / dZ^2

    Boundary cells are treated as zero (Dirichlet).
    """
    rows, cols = psi.shape
    R = geom.R_grid()
    dR, dZ = geom.dR, geom.dZ

    out = torch.zeros_like(psi)

    # R-derivative terms (interior columns 1..cols-2)
    psi_E = psi[:, 2:]      # j+1
    psi_W = psi[:, :-2]     # j-1
    psi_C = psi[:, 1:-1]    # j
    R_C = R[:, 1:-1]

    d2_dR2 = (psi_E - 2.0 * psi_C + psi_W) / (dR * dR)
    d_dR_over_R = (psi_E - psi_W) / (2.0 * R_C * dR)
    out[:, 1:-1] = d2_dR2 - d_dR_over_R

    # Z-derivative term (interior rows 1..rows-2)
    out[1:-1, :] = out[1:-1, :] + (psi[2:, :] - 2.0 * psi[1:-1, :] + psi[:-2, :]) / (dZ * dZ)

    return out


def taylor_beltrami_mode(
    geom: CylindricalGeometry,
    reference: Optional[torch.Tensor] = None,
    source: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lowest Dirichlet Beltrami/Taylor mode on the placement canvas.

    Taylor relaxation says the fixed-helicity minimum-energy force-free
    state satisfies curl(B) = lambda B. In the 2D flux-function reduction
    used here, the lowest rectangular Dirichlet eigenfunction is the
    corresponding global relaxation target:

        laplacian(psi_T) = -lambda psi_T.

    We use the analytic mode instead of an iterative Helmholtz solve because
    the canvas boundary is rectangular and fixed. The mode is signed by the
    current source and scaled to the current GS flux amplitude so it nudges
    topology without swamping the local Grad-Shafranov equilibrium.
    """
    x = (torch.arange(geom.grid_cols, device=geom.device, dtype=torch.float32) + 0.5) * geom.dR
    y = (torch.arange(geom.grid_rows, device=geom.device, dtype=torch.float32) + 0.5) * geom.dZ
    mode = torch.sin(math.pi * x / max(float(geom.canvas_width), 1e-9)).unsqueeze(0)
    mode = mode * torch.sin(math.pi * y / max(float(geom.canvas_height), 1e-9)).unsqueeze(1)

    # Match the solver's Dirichlet convention exactly on boundary cells.
    mode[0, :] = 0.0
    mode[-1, :] = 0.0
    mode[:, 0] = 0.0
    mode[:, -1] = 0.0
    mode = mode / torch.clamp(mode.abs().max(), min=1e-12)

    sign = 1.0
    if source is not None and source.numel() == mode.numel():
        helicity_proxy = float((mode * source.to(device=geom.device, dtype=mode.dtype)).sum().item())
        if helicity_proxy < 0.0:
            sign = -1.0

    amp = 1.0
    if reference is not None and reference.numel() == mode.numel():
        ref_abs = reference.to(device=geom.device, dtype=mode.dtype).abs().reshape(-1)
        positive = ref_abs[ref_abs > 1e-12]
        if positive.numel() > 0:
            amp = float(torch.quantile(positive, 0.90).item())
    return mode * (sign * max(amp, 1e-12))


def _gs_stencil_coefficients(geom: CylindricalGeometry) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Compute the 5-point stencil coefficients for Delta-star at each cell.

    Returns (a_E, a_W, a_N, a_S, a_inv_sum) where a_inv_sum = 1 / (a_E + a_W + a_N + a_S).

    The stencil identity is
        (Delta-star psi)_C  =  a_E psi_E + a_W psi_W + a_N psi_N + a_S psi_S
                              - (a_E + a_W + a_N + a_S) psi_C
    so the Gauss-Seidel update at C is
        psi_C  <-  (a_E psi_E + a_W psi_W + a_N psi_N + a_S psi_S - rhs) / (a_E + a_W + a_N + a_S).
    """
    R = geom.R_grid()
    dR, dZ = geom.dR, geom.dZ
    inv_dR2 = 1.0 / (dR * dR)
    inv_2RdR = 1.0 / (2.0 * R * dR)
    inv_dZ2 = 1.0 / (dZ * dZ)

    a_E = inv_dR2 - inv_2RdR       # east (j+1)
    a_W = inv_dR2 + inv_2RdR       # west (j-1)
    a_N = torch.full_like(R, inv_dZ2)
    a_S = torch.full_like(R, inv_dZ2)

    a_sum = a_E + a_W + a_N + a_S
    a_inv_sum = 1.0 / torch.clamp(a_sum, min=1e-12)
    return a_E, a_W, a_N, a_S, a_inv_sum


def gs_residual(
    psi: torch.Tensor,
    rhs_fn: Callable[[torch.Tensor], torch.Tensor],
    geom: CylindricalGeometry,
    rhs_clamp: float = 1e6,
) -> torch.Tensor:
    """Residual F(psi) = Delta-star(psi) - rhs(psi), with Dirichlet rows zeroed."""
    rhs = rhs_fn(psi)
    rhs = torch.clamp(rhs, min=-float(rhs_clamp), max=float(rhs_clamp))
    out = delta_star_apply(psi, geom) - rhs
    out[0, :] = 0.0
    out[-1, :] = 0.0
    out[:, 0] = 0.0
    out[:, -1] = 0.0
    return out


def newton_diagonal_correction(
    psi: torch.Tensor,
    profiles,
    geom: CylindricalGeometry,
    mu0_eff: float,
) -> torch.Tensor:
    """Diagonal derivative of the nonlinear GS residual."""
    R_sq = geom.R_grid() * geom.R_grid()
    p_pp = profiles.p_double_prime_fn(psi)
    F_val = profiles.F_fn(psi)
    F_p = profiles.F_prime_fn(psi)
    F_pp = profiles.F_double_prime_fn(psi)
    d = float(mu0_eff) * R_sq * p_pp + F_p * F_p + F_val * F_pp
    d[0, :] = 0.0
    d[-1, :] = 0.0
    d[:, 0] = 0.0
    d[:, -1] = 0.0
    return d


def rb_gauss_seidel_solve_modified(
    delta_psi: torch.Tensor,
    rhs: torch.Tensor,
    diag_correction: torch.Tensor,
    geom: CylindricalGeometry,
    n_sweeps: int = 1,
    omega: float = 1.0,
    coef_cache: Optional[Tuple] = None,
) -> torch.Tensor:
    """RB-GS for (Delta-star + diag_correction) delta_psi = rhs."""
    rows, cols = delta_psi.shape
    if coef_cache is None:
        coef_cache = _gs_stencil_coefficients(geom)
    a_E, a_W, a_N, a_S, a_inv_sum = coef_cache
    a_sum = 1.0 / torch.clamp(a_inv_sum, min=1e-12)
    d_safe = torch.minimum(diag_correction, 0.9 * a_sum)
    inv_center = 1.0 / torch.clamp(a_sum - d_safe, min=1e-12)

    i_idx = torch.arange(rows, device=delta_psi.device).unsqueeze(1)
    j_idx = torch.arange(cols, device=delta_psi.device).unsqueeze(0)
    parity = (i_idx + j_idx) % 2
    interior = (
        (i_idx >= 1) & (i_idx <= rows - 2) &
        (j_idx >= 1) & (j_idx <= cols - 2)
    )
    red_mask = (parity == 0) & interior
    black_mask = (parity == 1) & interior

    for _ in range(int(n_sweeps)):
        for color_mask in (red_mask, black_mask):
            psi_E = torch.zeros_like(delta_psi)
            psi_W = torch.zeros_like(delta_psi)
            psi_N = torch.zeros_like(delta_psi)
            psi_S = torch.zeros_like(delta_psi)
            psi_E[:, :-1] = delta_psi[:, 1:]
            psi_W[:, 1:] = delta_psi[:, :-1]
            psi_N[:-1, :] = delta_psi[1:, :]
            psi_S[1:, :] = delta_psi[:-1, :]

            numerator = a_E * psi_E + a_W * psi_W + a_N * psi_N + a_S * psi_S - rhs
            candidate = numerator * inv_center
            updated = (1.0 - omega) * delta_psi + omega * candidate
            delta_psi = torch.where(color_mask, updated, delta_psi)

    return delta_psi


def rb_gauss_seidel_sweep(
    psi: torch.Tensor,
    rhs: torch.Tensor,
    geom: CylindricalGeometry,
    n_sweeps: int = 1,
    omega: float = 1.0,
    coef_cache: Optional[Tuple] = None,
) -> torch.Tensor:
    """One outer call of RB-GS, performing `n_sweeps` red-black sweeps.

    SOR relaxation parameter `omega` in (0, 2). omega = 1 is plain GS;
    omega > 1 is over-relaxation (faster for smooth problems); omega < 1
    is under-relaxation (more stable for ill-conditioned).

    Updates `psi` in place on the interior; boundary cells are not touched.
    """
    rows, cols = psi.shape
    if coef_cache is None:
        coef_cache = _gs_stencil_coefficients(geom)
    a_E, a_W, a_N, a_S, a_inv_sum = coef_cache

    # Precompute parity masks for the interior cells once.
    i_idx = torch.arange(rows, device=psi.device).unsqueeze(1)
    j_idx = torch.arange(cols, device=psi.device).unsqueeze(0)
    parity = (i_idx + j_idx) % 2
    interior = (
        (i_idx >= 1) & (i_idx <= rows - 2) &
        (j_idx >= 1) & (j_idx <= cols - 2)
    )
    red_mask = (parity == 0) & interior
    black_mask = (parity == 1) & interior

    for _ in range(int(n_sweeps)):
        for color_mask in (red_mask, black_mask):
            # Compute candidate update using full-array neighbor reads.
            # Use roll-like indexing via slicing.
            psi_E = torch.zeros_like(psi)
            psi_W = torch.zeros_like(psi)
            psi_N = torch.zeros_like(psi)
            psi_S = torch.zeros_like(psi)

            psi_E[:, :-1] = psi[:, 1:]
            psi_W[:, 1:] = psi[:, :-1]
            psi_N[:-1, :] = psi[1:, :]
            psi_S[1:, :] = psi[:-1, :]

            numerator = a_E * psi_E + a_W * psi_W + a_N * psi_N + a_S * psi_S - rhs
            psi_candidate = numerator * a_inv_sum

            # SOR-blended update only on the cells of this color.
            updated = (1.0 - omega) * psi + omega * psi_candidate
            psi = torch.where(color_mask, updated, psi)

    return psi


def solve_grad_shafranov(
    rhs_fn: Callable[[torch.Tensor], torch.Tensor],
    geom: CylindricalGeometry,
    psi_init: Optional[torch.Tensor] = None,
    cfg: Optional[GSSolverConfig] = None,
) -> Tuple[torch.Tensor, ConvergenceTrace]:
    """Solve the nonlinear Grad-Shafranov equation by damped Picard + RB-GS.

    Parameters
    ----------
    rhs_fn : callable psi -> rhs
        Builds the right-hand side from the current psi iterate. Encapsulates
        `-mu0_eff R^2 p'(psi) - F F'(psi) + coil_sources`.
    geom : CylindricalGeometry
    psi_init : optional Tensor [grid_rows, grid_cols]
        Initial guess. Zero by default.
    cfg : GSSolverConfig

    Returns
    -------
    psi : Tensor [grid_rows, grid_cols] with Dirichlet psi=0 on boundary.
    trace : ConvergenceTrace
    """
    if cfg is None:
        cfg = GSSolverConfig()

    if psi_init is None:
        psi = torch.zeros(geom.grid_rows, geom.grid_cols, device=geom.device, dtype=torch.float32)
    else:
        psi = psi_init.clone().to(device=geom.device, dtype=torch.float32)
        psi[0, :] = 0.0
        psi[-1, :] = 0.0
        psi[:, 0] = 0.0
        psi[:, -1] = 0.0

    coef_cache = _gs_stencil_coefficients(geom)
    residuals: List[float] = []
    converged = False

    for k in range(int(cfg.picard_outer)):
        prev = psi.clone()

        rhs = rhs_fn(psi)
        rhs = torch.clamp(rhs, min=-cfg.rhs_clamp, max=cfg.rhs_clamp)
        # Enforce Dirichlet by zeroing rhs on boundary (so GS sweep keeps psi=0 there).
        rhs[0, :] = 0.0
        rhs[-1, :] = 0.0
        rhs[:, 0] = 0.0
        rhs[:, -1] = 0.0

        psi_new = rb_gauss_seidel_sweep(
            psi,
            rhs,
            geom,
            n_sweeps=int(cfg.gs_inner_sweeps),
            omega=float(cfg.omega_sor),
            coef_cache=coef_cache,
        )

        # Damped Picard blend.
        omega_p = float(cfg.omega_picard)
        psi = (1.0 - omega_p) * prev + omega_p * psi_new
        psi = torch.clamp(psi, min=-cfg.psi_clamp, max=cfg.psi_clamp)
        # Restore Dirichlet boundary.
        psi[0, :] = 0.0
        psi[-1, :] = 0.0
        psi[:, 0] = 0.0
        psi[:, -1] = 0.0

        # Residual: L2 norm of the change.
        diff = (psi - prev).reshape(-1)
        ref = max(float(prev.abs().mean().item()), 1e-9)
        residual = float((diff.pow(2).mean().sqrt().item()) / ref)
        residuals.append(residual)

        if residual <= cfg.convergence_tol:
            converged = True
            break

    return psi, ConvergenceTrace(residuals=residuals, converged=converged)


def solve_grad_shafranov_newton(
    rhs_fn: Callable[[torch.Tensor], torch.Tensor],
    profiles,
    geom: CylindricalGeometry,
    psi_init: Optional[torch.Tensor] = None,
    cfg: Optional[GSSolverConfig] = None,
) -> Tuple[torch.Tensor, ConvergenceTrace]:
    """Solve the nonlinear GS residual with damped Newton-Raphson.

    This is intentionally side-by-side with the Picard solver while it is
    validated. The placer can enable it via `use_newton_solver` without
    changing the macro-update, Taylor overlay, or soft-TILOS stages.
    """
    if cfg is None:
        cfg = GSSolverConfig()

    if psi_init is None:
        psi = torch.zeros(geom.grid_rows, geom.grid_cols, device=geom.device, dtype=torch.float32)
    else:
        psi = psi_init.clone().to(device=geom.device, dtype=torch.float32)
    psi[0, :] = 0.0
    psi[-1, :] = 0.0
    psi[:, 0] = 0.0
    psi[:, -1] = 0.0

    coef_cache = _gs_stencil_coefficients(geom)
    residuals: List[float] = []
    converged = False

    for _ in range(int(cfg.newton_outer_max)):
        res = gs_residual(psi, rhs_fn, geom, rhs_clamp=cfg.rhs_clamp)
        res_norm = float(res.pow(2).mean().sqrt().item())
        residuals.append(res_norm)
        if res_norm <= float(cfg.newton_tol_residual):
            converged = True
            break

        diag = newton_diagonal_correction(psi, profiles, geom, cfg.mu0_eff)
        diag = torch.clamp(diag, min=-cfg.rhs_clamp, max=cfg.rhs_clamp)
        delta = rb_gauss_seidel_solve_modified(
            torch.zeros_like(psi),
            -res,
            diag,
            geom,
            n_sweeps=int(cfg.gs_inner_sweeps),
            omega=float(cfg.omega_sor),
            coef_cache=coef_cache,
        )
        delta[0, :] = 0.0
        delta[-1, :] = 0.0
        delta[:, 0] = 0.0
        delta[:, -1] = 0.0

        step_norm = float(delta.abs().max().item())
        psi_norm = max(float(psi.abs().max().item()), 1e-9)
        alpha = 1.0
        armijo_c1 = float(cfg.newton_armijo_c1)
        min_alpha = float(cfg.newton_min_alpha)
        best_trial = psi
        best_norm = res_norm

        while alpha >= min_alpha:
            trial = psi + alpha * delta
            trial = torch.clamp(trial, min=-cfg.psi_clamp, max=cfg.psi_clamp)
            trial[0, :] = 0.0
            trial[-1, :] = 0.0
            trial[:, 0] = 0.0
            trial[:, -1] = 0.0
            trial_res = gs_residual(trial, rhs_fn, geom, rhs_clamp=cfg.rhs_clamp)
            trial_norm = float(trial_res.pow(2).mean().sqrt().item())
            if trial_norm < best_norm:
                best_trial = trial
                best_norm = trial_norm
            if trial_norm <= (1.0 - armijo_c1 * alpha) * res_norm:
                best_trial = trial
                best_norm = trial_norm
                break
            alpha *= 0.5

        psi = best_trial
        if step_norm <= float(cfg.newton_tol_step) * psi_norm or best_norm <= float(cfg.newton_tol_residual):
            converged = best_norm <= max(float(cfg.newton_tol_residual), res_norm)
            residuals.append(best_norm)
            break

    return psi, ConvergenceTrace(residuals=residuals, converged=converged)


def gradient_of_psi(
    psi: torch.Tensor, geom: CylindricalGeometry
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Central-difference gradient of psi on the grid.

    Returns (d psi / dR, d psi / dZ), each [grid_rows, grid_cols]. Boundary
    gradients use one-sided differences.
    """
    dR, dZ = geom.dR, geom.dZ
    gR = torch.zeros_like(psi)
    gZ = torch.zeros_like(psi)

    gR[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2.0 * dR)
    gR[:, 0] = (psi[:, 1] - psi[:, 0]) / dR
    gR[:, -1] = (psi[:, -1] - psi[:, -2]) / dR

    gZ[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2.0 * dZ)
    gZ[0, :] = (psi[1, :] - psi[0, :]) / dZ
    gZ[-1, :] = (psi[-1, :] - psi[-2, :]) / dZ

    return gR, gZ
