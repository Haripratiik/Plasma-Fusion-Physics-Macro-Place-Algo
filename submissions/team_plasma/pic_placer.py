"""R5 hybrid Particle-In-Cell soft-macro relaxation.

This module is an opt-in plasma-kinetic replacement/competitor for the R4
Braginskii PDE relaxer. It uses canonical PIC ingredients: particle-to-grid
charge/current deposition, self-consistent Poisson field solves, grid-to-
particle interpolation, Debye-shielded short-range repulsion, sheath boundary
forces, and a Boris pusher for magnetized particle motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch

from geometry import CylindricalGeometry, bilinear_sample, sample_vector


@dataclass
class PICConfig:
    enabled: bool = False
    max_iters: int = 32
    dt_frac: float = 0.08
    debye_length_frac: float = 0.025
    sheath_width_frac: float = 0.035
    ion_to_electron_mass_ratio: float = 100.0
    net_current_scale: float = 0.30
    coulomb_strength: float = 0.15
    electric_strength: float = 0.45
    magnetic_strength: float = 0.20
    sheath_strength: float = 0.40
    collision_freq: float = 0.12
    trust_radius_frac: float = 0.006
    field_solve_iters: int = 42
    field_solver: str = "jacobi"
    conv_tol: float = 1e-4
    current_line_samples: int = 12
    b0_strength: float = 0.0
    b_ripple_strength: float = 0.0
    grad_b_drift_strength: float = 0.0
    magnetic_mirror_strength: float = 0.0
    pair_attraction_strength: float = 0.0
    pair_attraction_range_frac: float = 1.0
    pair_attraction_softening_frac: float = 0.01
    bootstrap_current_strength: float = 0.0
    diamagnetic_drift_strength: float = 0.0
    annealed_schedule_enabled: bool = False


def _cic_deposit(points: torch.Tensor, values: torch.Tensor, geom: CylindricalGeometry) -> torch.Tensor:
    """Cloud-in-cell scalar deposition onto the geometry grid."""
    rows, cols = geom.grid_rows, geom.grid_cols
    grid = torch.zeros(rows, cols, device=geom.device, dtype=torch.float32)
    if points.numel() == 0:
        return grid
    pts = points.to(device=geom.device, dtype=torch.float32)
    val = values.to(device=geom.device, dtype=torch.float32).reshape(-1)
    cx = torch.clamp(pts[:, 0] / max(float(geom.dR), 1e-9) - 0.5, 0.0, float(cols - 1))
    cy = torch.clamp(pts[:, 1] / max(float(geom.dZ), 1e-9) - 0.5, 0.0, float(rows - 1))
    x0 = torch.floor(cx).long()
    y0 = torch.floor(cy).long()
    x1 = torch.clamp(x0 + 1, max=cols - 1)
    y1 = torch.clamp(y0 + 1, max=rows - 1)
    tx = cx - x0.float()
    ty = cy - y0.float()
    flat = grid.reshape(-1)
    stride = cols
    flat.scatter_add_(0, y0 * stride + x0, val * (1.0 - tx) * (1.0 - ty))
    flat.scatter_add_(0, y0 * stride + x1, val * tx * (1.0 - ty))
    flat.scatter_add_(0, y1 * stride + x0, val * (1.0 - tx) * ty)
    flat.scatter_add_(0, y1 * stride + x1, val * tx * ty)
    return grid


def deposit_charge_density(
    positions: torch.Tensor,
    charge: torch.Tensor,
    geom: CylindricalGeometry,
) -> torch.Tensor:
    rho = _cic_deposit(positions, charge, geom)
    scale = torch.clamp(torch.quantile(torch.abs(rho).reshape(-1), 0.90), min=1e-6)
    return (rho - rho.mean()) / scale


def deposit_current_density(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    geom: CylindricalGeometry,
    current_scale: float,
    max_samples: int,
) -> torch.Tensor:
    """Deposit quasi-static net current along edge chords."""
    if edge_index.numel() == 0 or edge_weight.numel() == 0 or positions.numel() == 0:
        return torch.zeros(geom.grid_rows, geom.grid_cols, device=geom.device, dtype=torch.float32)
    pos = positions.to(device=geom.device, dtype=torch.float32)
    edges = edge_index.to(device=geom.device, dtype=torch.long)
    weights = edge_weight.to(device=geom.device, dtype=torch.float32)
    median_w = torch.clamp(torch.median(torch.abs(weights)), min=1e-6)
    n = int(pos.shape[0])
    samples_cap = max(2, int(max_samples))
    valid = (
        (edges[:, 0] >= 0)
        & (edges[:, 0] < n)
        & (edges[:, 1] >= 0)
        & (edges[:, 1] < n)
        & (edges[:, 0] != edges[:, 1])
    )
    if not bool(valid.any()):
        return torch.zeros(geom.grid_rows, geom.grid_cols, device=geom.device, dtype=torch.float32)
    edges = edges[valid]
    weights = weights[valid]
    pa = pos[edges[:, 0]]
    pb = pos[edges[:, 1]]
    t = torch.linspace(0.0, 1.0, samples_cap, device=geom.device, dtype=torch.float32)
    line = pa.unsqueeze(1) * (1.0 - t.view(1, -1, 1)) + pb.unsqueeze(1) * t.view(1, -1, 1)
    vals = float(current_scale) * (weights / median_w).unsqueeze(1).expand(-1, samples_cap) / float(samples_cap)
    J = _cic_deposit(line.reshape(-1, 2), vals.reshape(-1), geom)
    scale = torch.clamp(torch.quantile(torch.abs(J).reshape(-1), 0.90), min=1e-6)
    return J / scale


def _fft_poisson_periodic(source: torch.Tensor, geom: CylindricalGeometry) -> torch.Tensor:
    """Fast spectral Poisson solve with zero-mean periodic closure."""
    rhs = source.to(dtype=torch.float32) - source.to(dtype=torch.float32).mean()
    rows, cols = rhs.shape
    ky = 2.0 * math.pi * torch.fft.fftfreq(rows, d=float(geom.dZ), device=rhs.device)
    kx = 2.0 * math.pi * torch.fft.fftfreq(cols, d=float(geom.dR), device=rhs.device)
    k2 = ky[:, None] * ky[:, None] + kx[None, :] * kx[None, :]
    rhs_hat = torch.fft.fft2(rhs)
    phi_hat = torch.zeros_like(rhs_hat)
    mask = k2 > 1e-12
    phi_hat[mask] = -rhs_hat[mask] / k2[mask]
    phi = torch.real(torch.fft.ifft2(phi_hat))
    phi[0, :] = 0.0
    phi[-1, :] = 0.0
    phi[:, 0] = 0.0
    phi[:, -1] = 0.0
    return phi


def solve_poisson_2d(source: torch.Tensor, geom: CylindricalGeometry, iters: int, solver: str = "jacobi") -> torch.Tensor:
    """Jacobi solve of planar Poisson with zero-Dirichlet limiter boundary."""
    if str(solver).lower() == "fft":
        return _fft_poisson_periodic(source, geom)
    phi = torch.zeros_like(source)
    rhs = source.to(dtype=torch.float32)
    h2 = float(geom.dR) * float(geom.dZ)
    for _ in range(max(1, int(iters))):
        new_phi = phi.clone()
        new_phi[1:-1, 1:-1] = 0.25 * (
            phi[1:-1, 2:]
            + phi[1:-1, :-2]
            + phi[2:, 1:-1]
            + phi[:-2, 1:-1]
            + h2 * rhs[1:-1, 1:-1]
        )
        phi = new_phi
    return phi


def _gradient(field: torch.Tensor, geom: CylindricalGeometry) -> tuple[torch.Tensor, torch.Tensor]:
    gx = torch.zeros_like(field)
    gy = torch.zeros_like(field)
    gx[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / max(2.0 * float(geom.dR), 1e-9)
    gy[1:-1, :] = (field[2:, :] - field[:-2, :]) / max(2.0 * float(geom.dZ), 1e-9)
    return gx, gy


def solve_vector_poisson_2d(
    current_density: torch.Tensor,
    geom: CylindricalGeometry,
    iters: int,
    solver: str = "jacobi",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve for A_z and return in-plane B. Bz comes from real background B0."""
    A = solve_poisson_2d(current_density, geom, iters, solver=solver)
    dA_dx, dA_dy = _gradient(A, geom)
    Bx = dA_dy
    By = -dA_dx
    return Bx, By


def debye_coulomb_pair(
    soft: torch.Tensor,
    all_pos: torch.Tensor,
    soft_charge: torch.Tensor,
    all_charge: torch.Tensor,
    lambda_d: float,
    strength: float,
) -> torch.Tensor:
    if soft.numel() == 0 or strength <= 0.0:
        return torch.zeros_like(soft)
    delta = soft.unsqueeze(1) - all_pos.unsqueeze(0)
    dist = torch.linalg.norm(delta, dim=2).clamp_min(1e-6)
    cutoff = max(float(lambda_d) * 4.0, 1e-6)
    mask = dist < cutoff
    qprod = soft_charge.unsqueeze(1) * all_charge.unsqueeze(0)
    screened = torch.exp(-dist / max(float(lambda_d), 1e-6)) / torch.clamp(dist * dist, min=1e-6)
    force = delta / dist.unsqueeze(2) * (qprod * screened * mask).unsqueeze(2)
    return float(strength) * force.sum(dim=1)


def biot_savart_pair_attraction(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    current_scale: float,
    range_limit: float,
    softening: float,
) -> torch.Tensor:
    """P3M short-range magnetic attraction between connected current filaments.

    The mesh current in `deposit_current_density` captures long-range field
    structure. This pair term restores the per-net pair information that a
    single grid current loop loses: connected pins behave like parallel
    current filaments and attract by Ampere/Biot-Savart, |F| ~ I^2 / d.
    """
    if positions.numel() == 0 or edge_index.numel() == 0 or edge_weight.numel() == 0 or current_scale <= 0.0:
        return torch.zeros_like(positions)
    pos = positions
    n = int(pos.shape[0])
    edges = edge_index.to(device=pos.device, dtype=torch.long)
    weights = edge_weight.to(device=pos.device, dtype=torch.float32)
    valid = (
        (edges[:, 0] >= 0)
        & (edges[:, 0] < n)
        & (edges[:, 1] >= 0)
        & (edges[:, 1] < n)
        & (edges[:, 0] != edges[:, 1])
    )
    if not bool(valid.any()):
        return torch.zeros_like(pos)
    edges = edges[valid]
    weights = weights[valid]
    pa = pos[edges[:, 0]]
    pb = pos[edges[:, 1]]
    delta = pb - pa
    dist = torch.linalg.norm(delta, dim=1).clamp_min(max(float(softening), 1e-6))
    if range_limit > 0.0:
        active = dist <= float(range_limit)
    else:
        active = torch.ones_like(dist, dtype=torch.bool)
    if not bool(active.any()):
        return torch.zeros_like(pos)
    median_w = torch.clamp(torch.median(torch.abs(weights)), min=1e-6)
    current = float(current_scale) * weights / median_w
    # Dimensionless mu0/(2*pi) absorbed into strength; normalization happens
    # downstream with the other PIC forces.
    force_mag = (current * current) / dist
    unit = delta / dist.unsqueeze(1)
    force_vec = force_mag.unsqueeze(1) * unit * active.to(dtype=torch.float32).unsqueeze(1)
    forces = torch.zeros_like(pos)
    idx_a = edges[:, 0].view(-1, 1).expand_as(force_vec)
    idx_b = edges[:, 1].view(-1, 1).expand_as(force_vec)
    forces.scatter_add_(0, idx_a, force_vec)
    forces.scatter_add_(0, idx_b, -force_vec)
    return forces


def sheath_wall_force(
    soft: torch.Tensor,
    soft_sizes: torch.Tensor,
    geom: CylindricalGeometry,
    width: float,
    strength: float,
) -> torch.Tensor:
    if soft.numel() == 0 or strength <= 0.0:
        return torch.zeros_like(soft)
    w = max(float(width), 1e-6)
    half = 0.5 * soft_sizes
    left = torch.exp(-torch.clamp(soft[:, 0] - half[:, 0], min=0.0) / w)
    right = torch.exp(-torch.clamp(float(geom.canvas_width) - (soft[:, 0] + half[:, 0]), min=0.0) / w)
    bottom = torch.exp(-torch.clamp(soft[:, 1] - half[:, 1], min=0.0) / w)
    top = torch.exp(-torch.clamp(float(geom.canvas_height) - (soft[:, 1] + half[:, 1]), min=0.0) / w)
    return float(strength) * torch.stack([left - right, bottom - top], dim=1)


def boris_push(
    pos: torch.Tensor,
    vel: torch.Tensor,
    charge: torch.Tensor,
    mass: torch.Tensor,
    electric_force: torch.Tensor,
    Bz: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """2D Boris push with scalar out-of-plane magnetic field."""
    qm = (charge / mass.clamp_min(1e-6)).unsqueeze(1)
    eacc = qm * electric_force
    v_minus = vel + 0.5 * float(dt) * eacc
    t = 0.5 * float(dt) * (charge / mass.clamp_min(1e-6)) * Bz
    s = 2.0 * t / (1.0 + t * t)
    v_prime = torch.stack([v_minus[:, 0] + v_minus[:, 1] * t, v_minus[:, 1] - v_minus[:, 0] * t], dim=1)
    v_plus = torch.stack([v_minus[:, 0] + v_prime[:, 1] * s, v_minus[:, 1] - v_prime[:, 0] * s], dim=1)
    v_new = v_plus + 0.5 * float(dt) * eacc
    return pos + float(dt) * v_new, v_new


def _degree_charge(n: int, edge_index: torch.Tensor, edge_weight: torch.Tensor, device: torch.device) -> torch.Tensor:
    charge = torch.ones(n, device=device, dtype=torch.float32)
    if edge_index.numel() == 0 or edge_weight.numel() == 0:
        return charge
    edges = edge_index.to(device=device, dtype=torch.long)
    weights = torch.abs(edge_weight.to(device=device, dtype=torch.float32))
    valid = (
        (edges[:, 0] >= 0)
        & (edges[:, 0] < n)
        & (edges[:, 1] >= 0)
        & (edges[:, 1] < n)
    )
    if not bool(valid.any()):
        return charge
    charge.zero_()
    ends = edges[valid].reshape(-1)
    vals = weights[valid].repeat_interleave(2)
    charge.scatter_add_(0, ends, vals)
    return charge / charge.mean().clamp_min(1e-6)


def pic_relax_soft_macros(
    hard_pos: torch.Tensor,
    hard_sizes: torch.Tensor,
    soft_pos: torch.Tensor,
    soft_sizes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    geom: CylindricalGeometry,
    cfg: PICConfig,
    soft_fixed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Hybrid-PIC relaxation: fixed ion hard macros, kinetic soft electrons."""
    if not bool(cfg.enabled) or soft_pos.numel() == 0:
        return soft_pos.clone()
    device = geom.device
    hard = hard_pos.to(device=device, dtype=torch.float32)
    soft = soft_pos.to(device=device, dtype=torch.float32).clone()
    hard_sz = hard_sizes.to(device=device, dtype=torch.float32)
    soft_sz = soft_sizes.to(device=device, dtype=torch.float32)
    n_hard = int(hard.shape[0])
    n_soft = int(soft.shape[0])
    n_total = n_hard + n_soft
    fixed = soft_fixed.to(device=device).bool() if soft_fixed is not None else torch.zeros(n_soft, device=device, dtype=torch.bool)

    charge = _degree_charge(n_total, edge_index, edge_weight, device)
    mass = torch.ones(n_soft, device=device, dtype=torch.float32)
    vel = torch.zeros(n_soft, 2, device=device, dtype=torch.float32)
    span = max(float(geom.canvas_width), float(geom.canvas_height), 1e-6)
    trust = max(1e-6, float(cfg.trust_radius_frac) * span)
    dt = max(1e-4, float(cfg.dt_frac))
    lambda_d = max(1e-6, float(cfg.debye_length_frac) * span)
    sheath_w = max(1e-6, float(cfg.sheath_width_frac) * span)
    prev_step = None

    for _ in range(max(1, int(cfg.max_iters))):
        all_pos = torch.cat([hard, soft], dim=0)
        if bool(cfg.annealed_schedule_enabled):
            frac = float(_) / float(max(int(cfg.max_iters) - 1, 1))
            attract_ramp = math.exp(-2.0 * frac)
            repel_ramp = math.exp(2.0 * frac)
        else:
            attract_ramp = 1.0
            repel_ramp = 1.0

        rho = deposit_charge_density(all_pos, charge, geom)
        rho_x, rho_y = _gradient(rho, geom)
        phi = solve_poisson_2d(-rho, geom, int(cfg.field_solve_iters), solver=str(cfg.field_solver))
        phi_x, phi_y = _gradient(phi, geom)
        Ex = -phi_x
        Ey = -phi_y

        J = deposit_current_density(
            all_pos,
            edge_index,
            edge_weight,
            geom,
            float(cfg.net_current_scale) * attract_ramp,
            int(cfg.current_line_samples),
        )
        bs = float(getattr(cfg, "bootstrap_current_strength", 0.0))
        if bs != 0.0:
            # Hirshman-style bootstrap closure in 2D: pressure gradients
            # self-drive a current source that feeds back into magnetic
            # confinement. Normalize to keep it benchmark-scale invariant.
            J_bootstrap = rho_x - rho_y
            J_bootstrap = J_bootstrap / torch.clamp(torch.quantile(torch.abs(J_bootstrap).reshape(-1), 0.90), min=1e-6)
            J = J + bs * J_bootstrap
        Bx, By = solve_vector_poisson_2d(J, geom, int(cfg.field_solve_iters), solver=str(cfg.field_solver))
        mag_pressure = 0.5 * (Bx * Bx + By * By)
        mp_x, mp_y = _gradient(mag_pressure, geom)

        E_at = sample_vector(Ex, Ey, soft, geom)
        Bz_grid = torch.full_like(rho, float(getattr(cfg, "b0_strength", 0.0)))
        ripple = float(getattr(cfg, "b_ripple_strength", 0.0))
        if ripple != 0.0:
            phi_norm = phi / torch.clamp(torch.quantile(torch.abs(phi).reshape(-1), 0.90), min=1e-6)
            Bz_grid = Bz_grid + ripple * phi_norm
        Bz_x, Bz_y = _gradient(Bz_grid, geom)
        Bz_at = bilinear_sample(Bz_grid, soft, geom)
        mag_pull = sample_vector(mp_x, mp_y, soft, geom)
        rho_grad_at = sample_vector(rho_x, rho_y, soft, geom)
        b_grad_at = sample_vector(Bz_x, Bz_y, soft, geom)
        dia = torch.zeros_like(soft)
        dia_strength = float(getattr(cfg, "diamagnetic_drift_strength", 0.0))
        if dia_strength != 0.0:
            b2 = torch.clamp(Bz_at * Bz_at, min=1e-6).unsqueeze(1)
            dia = dia_strength * torch.stack([-Bz_at * rho_grad_at[:, 1], Bz_at * rho_grad_at[:, 0]], dim=1) / b2
        grad_b = torch.zeros_like(soft)
        grad_b_strength = float(getattr(cfg, "grad_b_drift_strength", 0.0))
        if grad_b_strength != 0.0:
            b2 = torch.clamp(Bz_at * Bz_at, min=1e-6).unsqueeze(1)
            grad_b = grad_b_strength * torch.stack([-Bz_at * b_grad_at[:, 1], Bz_at * b_grad_at[:, 0]], dim=1) / b2
        mirror = -float(getattr(cfg, "magnetic_mirror_strength", 0.0)) * b_grad_at
        q_soft = charge[n_hard:]
        pair_force = torch.zeros_like(soft)
        pair_strength = float(getattr(cfg, "pair_attraction_strength", 0.0))
        if pair_strength != 0.0:
            all_pair_force = biot_savart_pair_attraction(
                all_pos,
                edge_index,
                edge_weight,
                pair_strength * attract_ramp,
                max(1e-6, float(getattr(cfg, "pair_attraction_range_frac", 1.0)) * span),
                max(1e-6, float(getattr(cfg, "pair_attraction_softening_frac", 0.01)) * span),
            )
            pair_force = all_pair_force[n_hard:]
        extra = (
            float(cfg.coulomb_strength)
            * repel_ramp
            * debye_coulomb_pair(soft, all_pos, q_soft, charge, lambda_d, 1.0)
            + sheath_wall_force(soft, soft_sz, geom, sheath_w, float(cfg.sheath_strength))
            + float(cfg.magnetic_strength) * mag_pull
            + pair_force
            + mirror
        )
        total_electric = float(cfg.electric_strength) * E_at + dia + grad_b + extra / q_soft.unsqueeze(1).clamp_min(1e-6)
        total_electric = total_electric - float(cfg.collision_freq) * vel
        norms = torch.linalg.norm(total_electric, dim=1, keepdim=True)
        denom = torch.quantile(norms.reshape(-1), 0.90) if n_soft >= 10 else norms.max()
        if float(denom.item()) > 1e-9:
            total_electric = total_electric / denom

        new_soft, vel = boris_push(soft, vel, q_soft, mass, total_electric, Bz_at, dt)
        step = new_soft - soft
        step_norm = torch.linalg.norm(step, dim=1, keepdim=True)
        step = step * torch.clamp(trust / step_norm.clamp_min(1e-9), max=1.0)
        step[fixed] = 0.0
        vel[fixed] = 0.0
        soft = soft + step
        soft[:, 0] = torch.clamp(soft[:, 0], min=0.5 * soft_sz[:, 0], max=float(geom.canvas_width) - 0.5 * soft_sz[:, 0])
        soft[:, 1] = torch.clamp(soft[:, 1], min=0.5 * soft_sz[:, 1], max=float(geom.canvas_height) - 0.5 * soft_sz[:, 1])

        mean_step = float(torch.linalg.norm(step, dim=1).mean().item()) if step.numel() else 0.0
        if prev_step is not None and abs(prev_step - mean_step) < float(cfg.conv_tol) * span:
            break
        prev_step = mean_step
    return soft
