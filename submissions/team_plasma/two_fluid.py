"""
Two-fluid soft-macro evolution (adiabatic-electron approximation).

In the Hall-MHD two-fluid model, electrons are light and follow the
ion-set potential adiabatically:

    n_e(r) = n_0 * exp(-V(psi(r)) / T_e)     [Boltzmann relation]

In macro placement, soft macros (standard-cell-cluster abstractions) are
the electron fluid: they are dynamically fast relative to hard macros,
and they primarily react to the equilibrium set by hard-macro positions.

Implementation: at each Picard step, soft macros drift along `-grad(psi)`
scaled by a fast timestep. This replaces the slow `plc.optimize_stdcells`
call (multi-minute Python force-directed) with one vectorized GPU pass.

See docs/DERIVATION.md section 2.8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from geometry import CylindricalGeometry, sample_vector
from gs_solver import gradient_of_psi
from profiles import build_density_field


@dataclass
class TwoFluidConfig:
    """Soft-macro adiabatic step settings."""
    enabled: bool = True
    drift_step_frac: float = 0.05   # step size as fraction of max(W, H)
    trust_radius_frac: float = 0.02
    T_e: float = 1.0                # electron temperature scale
    normalize_force: bool = True
    pde_enabled: bool = False
    D_parallel: float = 0.20
    D_perp: float = 0.05
    dt: float = 0.20
    max_iters: int = 80
    conv_tol: float = 1e-4
    density_weight: float = 1.0
    psi_weight: float = 0.15
    trust_radius_pde_frac: float = 0.01
    psi_aligned_anisotropy_enabled: bool = False
    min_anisotropy_ratio: float = 100.0
    q_weight: float = 0.0           # Ware-pinch coupling to routing congestion (q-field)


def soft_macro_drift_step(
    soft_pos: torch.Tensor,
    psi: torch.Tensor,
    geom: CylindricalGeometry,
    cfg: TwoFluidConfig,
    soft_fixed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One adiabatic-electron drift step for soft macros.

    Each soft macro moves along `-grad(psi)` at its current position,
    rescaled by a per-iteration trust radius.

    Parameters
    ----------
    soft_pos : Tensor [N_soft, 2]
    psi : Tensor [grid_rows, grid_cols]
    geom : CylindricalGeometry
    cfg : TwoFluidConfig
    soft_fixed : optional Tensor [N_soft] bool. Fixed soft macros do not move.

    Returns
    -------
    new_soft_pos : Tensor [N_soft, 2]
    """
    if not cfg.enabled or soft_pos.numel() == 0:
        return soft_pos.clone()

    gR, gZ = gradient_of_psi(psi, geom)

    # Soft macros drift along -grad psi.
    force_at = sample_vector(-gR, -gZ, soft_pos, geom)

    if cfg.normalize_force:
        norms = torch.linalg.norm(force_at, dim=1, keepdim=True)
        denom = torch.quantile(norms.reshape(-1), 0.90) if norms.numel() >= 10 else norms.max()
        denom_val = float(denom.item()) if norms.numel() > 0 else 0.0
        if denom_val > 1e-9:
            force_at = force_at / denom_val

    span = max(geom.canvas_width, geom.canvas_height)
    step_size = float(cfg.drift_step_frac) * span * float(cfg.T_e)
    trust = float(cfg.trust_radius_frac) * span

    step = step_size * force_at
    step_norm = torch.linalg.norm(step, dim=1, keepdim=True)
    scale = torch.clamp(trust / torch.clamp(step_norm, min=1e-9), max=1.0)
    step = step * scale

    new_pos = soft_pos + step

    if soft_fixed is not None and bool(soft_fixed.any()):
        new_pos[soft_fixed] = soft_pos[soft_fixed]

    return new_pos


def relax_soft_density_field(
    hard_pos: torch.Tensor,
    hard_sizes: torch.Tensor,
    soft_pos: torch.Tensor,
    soft_sizes: torch.Tensor,
    psi: torch.Tensor,
    geom: CylindricalGeometry,
    cfg: TwoFluidConfig,
    soft_fixed: Optional[torch.Tensor] = None,
    q_field: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Braginskii-style two-fluid drift-diffusion relaxation for soft macros.

    This is the R4 pure-plasma replacement for TILOS `optimize_stdcells`.
    Soft macros are treated as adiabatic electrons moving down a chemical
    potential built from density pressure plus a weak equilibrium potential.
    Transport is anisotropic: fast along flux surfaces and slower across them.
    """
    if not bool(cfg.pde_enabled) or soft_pos.numel() == 0:
        return soft_pos.clone()

    device = psi.device
    soft = soft_pos.to(device=device, dtype=torch.float32).clone()
    hard = hard_pos.to(device=device, dtype=torch.float32)
    hard_sz = hard_sizes.to(device=device, dtype=torch.float32)
    soft_sz = soft_sizes.to(device=device, dtype=torch.float32)
    fixed = soft_fixed.to(device=device).bool() if soft_fixed is not None else None

    gpsi_x, gpsi_y = gradient_of_psi(psi, geom)
    span = max(float(geom.canvas_width), float(geom.canvas_height))
    trust = max(1e-6, float(cfg.trust_radius_pde_frac) * span)
    dt = max(1e-6, float(cfg.dt))
    dpar = max(0.0, float(cfg.D_parallel))
    dperp = max(0.0, float(cfg.D_perp))
    if bool(getattr(cfg, "psi_aligned_anisotropy_enabled", False)):
        # Boozer-style magnetic-coordinate transport: parallel motion along
        # flux surfaces must dominate cross-flux diffusion. This keeps the R4
        # mechanism plasma-unique instead of collapsing toward isotropic MD.
        ratio = max(1.0, float(getattr(cfg, "min_anisotropy_ratio", 100.0)))
        if dpar <= 0.0 and dperp > 0.0:
            dpar = dperp * ratio
        elif dperp <= 0.0 and dpar > 0.0:
            dperp = dpar / ratio
        elif dpar > 0.0 and dperp > 0.0 and dpar / max(dperp, 1e-12) < ratio:
            dperp = dpar / ratio
    max_iters = max(1, int(cfg.max_iters))

    q_weight = float(getattr(cfg, "q_weight", 0.0))
    q_grad_x = q_grad_y = None
    if q_weight != 0.0 and q_field is not None and q_field.numel() > 0:
        q_dev = q_field.to(device=device, dtype=torch.float32)
        q_scale = torch.clamp(torch.quantile(q_dev.reshape(-1), 0.90), min=1e-6)
        q_norm_grid = q_dev / q_scale
        # Pre-compute q-gradient ONCE per relax call. The Ware pinch is a
        # cross-field drift -- ISOTROPIC, not subordinate to D_par/D_perp
        # anisotropy. So we evaluate grad(q) outside the anisotropy
        # decomposition and add it as a separate velocity term below.
        q_grad_x = torch.zeros_like(q_norm_grid)
        q_grad_y = torch.zeros_like(q_norm_grid)
        q_grad_x[:, 1:-1] = (q_norm_grid[:, 2:] - q_norm_grid[:, :-2]) / max(2.0 * float(geom.dR), 1e-9)
        q_grad_y[1:-1, :] = (q_norm_grid[2:, :] - q_norm_grid[:-2, :]) / max(2.0 * float(geom.dZ), 1e-9)

    prev_mean = None
    for _ in range(max_iters):
        all_pos = torch.cat([hard, soft], dim=0)
        all_sizes = torch.cat([hard_sz, soft_sz], dim=0)
        density = build_density_field(all_pos, all_sizes, geom)
        density = density / torch.clamp(torch.quantile(density.reshape(-1), 0.90), min=1e-6)
        mu = float(cfg.density_weight) * density + float(cfg.psi_weight) * psi

        # Central differences on the chemical potential.
        mu_x = torch.zeros_like(mu)
        mu_y = torch.zeros_like(mu)
        mu_x[:, 1:-1] = (mu[:, 2:] - mu[:, :-2]) / max(2.0 * float(geom.dR), 1e-9)
        mu_y[1:-1, :] = (mu[2:, :] - mu[:-2, :]) / max(2.0 * float(geom.dZ), 1e-9)

        grad_mu = sample_vector(mu_x, mu_y, soft, geom)
        grad_psi = sample_vector(gpsi_x, gpsi_y, soft, geom)
        b = torch.stack([-grad_psi[:, 1], grad_psi[:, 0]], dim=1)
        b_norm = torch.linalg.norm(b, dim=1, keepdim=True)
        b = torch.where(b_norm > 1e-9, b / torch.clamp(b_norm, min=1e-9), torch.zeros_like(b))
        grad_parallel = (grad_mu * b).sum(dim=1, keepdim=True) * b
        grad_perp = grad_mu - grad_parallel

        velocity = -(dpar * grad_parallel + dperp * grad_perp)
        if q_grad_x is not None:
            # Ware-pinch cross-field drift: macros flow DOWN q-gradient
            # (away from RUDY peaks). Bypass the anisotropy decomposition --
            # in plasma physics the Ware pinch is itself a cross-field
            # neoclassical drift, not a parallel transport.
            grad_q_at_soft = sample_vector(q_grad_x, q_grad_y, soft, geom)
            velocity = velocity - q_weight * grad_q_at_soft
        if bool(cfg.normalize_force):
            norms = torch.linalg.norm(velocity, dim=1, keepdim=True)
            scale = torch.quantile(norms.reshape(-1), 0.90) if norms.numel() >= 10 else norms.max()
            if float(scale.item()) > 1e-9:
                velocity = velocity / scale

        step = dt * trust * velocity
        step_norm = torch.linalg.norm(step, dim=1, keepdim=True)
        step = step * torch.clamp(trust / torch.clamp(step_norm, min=1e-9), max=1.0)
        if fixed is not None and bool(fixed.any()):
            step[fixed] = 0.0
        soft = soft + step
        soft[:, 0] = torch.clamp(soft[:, 0], min=0.0, max=float(geom.canvas_width))
        soft[:, 1] = torch.clamp(soft[:, 1], min=0.0, max=float(geom.canvas_height))

        mean_step = float(torch.linalg.norm(step, dim=1).mean().item()) if step.numel() else 0.0
        if prev_mean is not None and abs(prev_mean - mean_step) < float(cfg.conv_tol) * span:
            break
        prev_mean = mean_step

    return soft
