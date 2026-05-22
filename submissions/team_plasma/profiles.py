"""
Online profile fitting for the Grad-Shafranov equation.

The GS equation has two free profile functions of psi:

    p(psi)  := plasma pressure on the flux surface
    F(psi)  := poloidal current function (F = R B_phi)

In tokamak equilibrium reconstruction (e.g. EFIT) these are fit from
magnetic diagnostics. Here we fit them at every Picard step from the
current placement state:

    p(psi) = flux-surface average of (q + alpha * rho)
    F^2(psi) = F_vac^2 + beta_F * I_net_enclosed(psi)

where q is RUDY routing demand, rho is macro density, and I_net_enclosed
is the cumulative net-attraction "current" enclosed inside the flux surface.

See docs/DERIVATION.md sections 2.4-2.5 for the derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch

from geometry import CylindricalGeometry, bilinear_sample


@dataclass
class ProfileConfig:
    """Settings for fitting p(psi) and F(psi)."""
    n_bins: int = 32
    poly_degree: int = 3
    alpha_q_rho: float = 0.5     # weight of rho in p = <q + alpha rho>
    F_vac: float = 1.0           # vacuum baseline for F
    beta_F: float = 0.1          # net-current coupling for F
    smoothing_eps: float = 1e-6
    tail_enabled: bool = False
    tail_quantile: float = 0.90
    tail_gamma: float = 1.0
    tail_power: float = 2.0


@dataclass
class Profiles:
    """A fitted (p, F) pair with analytic derivatives.

    Each `_fn` is a callable mapping `psi: Tensor [...]` -> `Tensor [...]`.
    """
    p_fn: Callable[[torch.Tensor], torch.Tensor]
    p_prime_fn: Callable[[torch.Tensor], torch.Tensor]
    p_double_prime_fn: Callable[[torch.Tensor], torch.Tensor]
    F_fn: Callable[[torch.Tensor], torch.Tensor]
    F_prime_fn: Callable[[torch.Tensor], torch.Tensor]
    F_double_prime_fn: Callable[[torch.Tensor], torch.Tensor]
    psi_axis: float
    psi_separatrix: float
    p_coeffs: torch.Tensor       # polynomial coefficients, monomial basis
    F_sq_coeffs: torch.Tensor    # coefficients of F^2(psi)
    bin_centers: torch.Tensor
    p_bin_values: torch.Tensor
    I_enc_bin_values: torch.Tensor


def _polyfit_torch(x: torch.Tensor, y: torch.Tensor, degree: int) -> torch.Tensor:
    """Least-squares polynomial fit in pure torch.

    Returns coefficients in increasing order, shape [degree+1], so
    p(x) = sum_{k=0..degree} c[k] * x^k.

    Numerically stable for small degrees (<= 6).
    """
    if x.numel() == 0:
        return torch.zeros(degree + 1, dtype=torch.float32, device=x.device)
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    # Vandermonde matrix [N, degree+1] with columns 1, x, x^2, ...
    cols = [torch.ones_like(x)]
    for k in range(1, degree + 1):
        cols.append(cols[-1] * x)
    V = torch.stack(cols, dim=1)
    # Solve V c = y in least squares sense.
    try:
        coeffs = torch.linalg.lstsq(V, y.unsqueeze(1)).solution.squeeze(1)
    except Exception:
        # Fallback: normal equations with ridge for stability.
        VtV = V.T @ V + 1e-9 * torch.eye(V.shape[1], device=V.device, dtype=V.dtype)
        Vty = V.T @ y
        coeffs = torch.linalg.solve(VtV, Vty)
    return coeffs


def _poly_eval(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Horner evaluation. coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ..."""
    if coeffs.numel() == 0:
        return torch.zeros_like(x)
    out = torch.zeros_like(x)
    for k in range(int(coeffs.shape[0]) - 1, -1, -1):
        out = out * x + coeffs[k]
    return out


def _poly_derivative_coeffs(coeffs: torch.Tensor) -> torch.Tensor:
    """Coefficients of d/dx of a polynomial in monomial form."""
    if coeffs.numel() <= 1:
        return torch.zeros(1, dtype=coeffs.dtype, device=coeffs.device)
    k = torch.arange(1, coeffs.shape[0], device=coeffs.device, dtype=coeffs.dtype)
    return coeffs[1:] * k


def _make_poly_callable(coeffs: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a callable that evaluates the polynomial."""
    coeffs = coeffs.clone()
    def _fn(x: torch.Tensor) -> torch.Tensor:
        return _poly_eval(coeffs.to(x.device, x.dtype), x)
    return _fn


def fit_profiles(
    psi: torch.Tensor,
    q_field: torch.Tensor,
    rho_field: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    positions: torch.Tensor,
    geom: CylindricalGeometry,
    cfg: ProfileConfig,
) -> Profiles:
    """Fit p(psi) and F(psi) from the current placement state.

    Algorithm
    ---------
    1. Bin grid cells by their psi value into `n_bins` equal-psi bins.
    2. For each bin: compute p_b = mean(q + alpha*rho) over its cells.
    3. Compute I_net_enclosed by sampling psi at edge midpoints and taking
       the cumulative distribution of edge-weight vs midpoint-psi.
    4. Fit polynomials p(psi) and F^2(psi) of degree `cfg.poly_degree`.

    Parameters
    ----------
    psi : [grid_rows, grid_cols]
    q_field, rho_field : [grid_rows, grid_cols]  RUDY congestion and density.
    edge_index : [E, 2] long, hard-macro endpoints of each net edge.
    edge_weight : [E]
    positions : [N_macros, 2]  macro positions (only first N_hard used as edge endpoints).
    geom : CylindricalGeometry
    cfg : ProfileConfig
    """
    psi_flat = psi.reshape(-1)
    psi_min = float(psi_flat.min().item())
    psi_max = float(psi_flat.max().item())

    # Degenerate case: nearly-constant psi -> default profile.
    if psi_max - psi_min < cfg.smoothing_eps:
        return _default_profiles(psi_min, psi_max, cfg, device=psi.device)

    # Bin cells by psi.
    n_bins = int(cfg.n_bins)
    edges = torch.linspace(psi_min, psi_max, n_bins + 1, device=psi.device)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    field_for_p = (q_field + cfg.alpha_q_rho * rho_field).reshape(-1)
    if bool(cfg.tail_enabled) and field_for_p.numel() > 0:
        q = max(0.50, min(0.99, float(cfg.tail_quantile)))
        threshold = torch.quantile(field_for_p, q)
        top = torch.clamp(field_for_p - threshold, min=0.0)
        top_ref = torch.clamp(field_for_p.max() - threshold, min=float(cfg.smoothing_eps))
        tail_t = top / top_ref
        tail_boost = 1.0 + float(cfg.tail_gamma) * torch.pow(tail_t, float(cfg.tail_power))
        field_for_p = field_for_p * tail_boost

    # Vectorized binning: bucketize then scatter_mean.
    bin_idx = torch.bucketize(psi_flat, edges[1:-1])  # in [0, n_bins-1]
    bin_idx = torch.clamp(bin_idx, 0, n_bins - 1)
    p_sum = torch.zeros(n_bins, device=psi.device, dtype=torch.float32)
    p_cnt = torch.zeros(n_bins, device=psi.device, dtype=torch.float32)
    p_sum.scatter_add_(0, bin_idx, field_for_p)
    p_cnt.scatter_add_(0, bin_idx, torch.ones_like(field_for_p))
    p_bin = p_sum / torch.clamp(p_cnt, min=1.0)

    # Fit p(psi) polynomial. Only fit on bins that have cells.
    populated = p_cnt > 0.5
    if int(populated.sum().item()) <= cfg.poly_degree:
        # Not enough data points; fall back to a parabolic profile.
        p_coeffs = torch.zeros(cfg.poly_degree + 1, device=psi.device)
        # Default: p(psi) = mean value of all q+rho, constant
        p_coeffs[0] = float(field_for_p.mean().item())
    else:
        p_coeffs = _polyfit_torch(bin_centers[populated], p_bin[populated], cfg.poly_degree)
    p_prime_coeffs = _poly_derivative_coeffs(p_coeffs)
    p_double_prime_coeffs = _poly_derivative_coeffs(p_prime_coeffs)

    # Enclosed net current I_net(psi).
    # Sample psi at edge midpoints; sort edges by midpoint psi; cumulative sum of weights.
    I_enc_bin = torch.zeros(n_bins, device=psi.device, dtype=torch.float32)
    if edge_index.numel() > 0:
        n_macros = int(positions.shape[0])
        e_src = edge_index[:, 0].clamp(0, n_macros - 1)
        e_dst = edge_index[:, 1].clamp(0, n_macros - 1)
        mid = 0.5 * (positions[e_src] + positions[e_dst])
        mid_psi = bilinear_sample(psi, mid, geom)
        # I_net(psi) = sum of weights for edges whose midpoint psi <= psi.
        # For each bin center, count weight contributions of edges with mid_psi <= center.
        # Vectorized: for each edge, find which bin its midpoint falls in; cumulative sum.
        mid_bin = torch.bucketize(mid_psi, edges[1:-1])
        mid_bin = torch.clamp(mid_bin, 0, n_bins - 1)
        weight_per_bin = torch.zeros(n_bins, device=psi.device, dtype=torch.float32)
        weight_per_bin.scatter_add_(0, mid_bin, edge_weight.to(torch.float32))
        I_enc_bin = torch.cumsum(weight_per_bin, dim=0)

    # Fit F^2(psi) = F_vac^2 + beta_F * I_net_enclosed(psi).
    F_sq_target = float(cfg.F_vac) ** 2 + float(cfg.beta_F) * I_enc_bin
    F_sq_coeffs = _polyfit_torch(bin_centers, F_sq_target, cfg.poly_degree)
    F_sq_prime_coeffs = _poly_derivative_coeffs(F_sq_coeffs)
    F_sq_double_prime_coeffs = _poly_derivative_coeffs(F_sq_prime_coeffs)

    # F(psi) = sqrt(F^2(psi)), so F'(psi) = (F^2)'(psi) / (2 F(psi)).
    # We build callables that handle the sqrt/division carefully.
    def F_fn(x: torch.Tensor) -> torch.Tensor:
        F_sq = _poly_eval(F_sq_coeffs.to(x.device, x.dtype), x)
        return torch.sqrt(torch.clamp(F_sq, min=cfg.smoothing_eps))

    def F_prime_fn(x: torch.Tensor) -> torch.Tensor:
        F_val = F_fn(x)
        F_sq_p = _poly_eval(F_sq_prime_coeffs.to(x.device, x.dtype), x)
        return F_sq_p / (2.0 * torch.clamp(F_val, min=cfg.smoothing_eps))

    def F_double_prime_fn(x: torch.Tensor) -> torch.Tensor:
        F_val = F_fn(x)
        F_safe = torch.clamp(F_val, min=cfg.smoothing_eps)
        F_sq_p = _poly_eval(F_sq_prime_coeffs.to(x.device, x.dtype), x)
        F_sq_pp = _poly_eval(F_sq_double_prime_coeffs.to(x.device, x.dtype), x)
        return F_sq_pp / (2.0 * F_safe) - (F_sq_p * F_sq_p) / (4.0 * F_safe * F_safe * F_safe)

    p_fn = _make_poly_callable(p_coeffs)
    p_prime_fn = _make_poly_callable(p_prime_coeffs)
    p_double_prime_fn = _make_poly_callable(p_double_prime_coeffs)

    # Magnetic axis = location of min(psi); separatrix = max(psi) (we use 0 at boundary,
    # so the "axis" is typically the most-negative or most-positive interior value).
    psi_axis = psi_min
    psi_separatrix = psi_max

    return Profiles(
        p_fn=p_fn,
        p_prime_fn=p_prime_fn,
        p_double_prime_fn=p_double_prime_fn,
        F_fn=F_fn,
        F_prime_fn=F_prime_fn,
        F_double_prime_fn=F_double_prime_fn,
        psi_axis=psi_axis,
        psi_separatrix=psi_separatrix,
        p_coeffs=p_coeffs,
        F_sq_coeffs=F_sq_coeffs,
        bin_centers=bin_centers,
        p_bin_values=p_bin,
        I_enc_bin_values=I_enc_bin,
    )


def _default_profiles(
    psi_min: float, psi_max: float, cfg: ProfileConfig, device: torch.device
) -> Profiles:
    """Default profile when psi is too uniform to bin meaningfully."""
    zero = torch.zeros(cfg.poly_degree + 1, device=device, dtype=torch.float32)
    F_sq = zero.clone()
    F_sq[0] = float(cfg.F_vac) ** 2

    def zero_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def F_const(x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, float(cfg.F_vac))

    def F_prime_zero(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    return Profiles(
        p_fn=zero_fn,
        p_prime_fn=zero_fn,
        p_double_prime_fn=zero_fn,
        F_fn=F_const,
        F_prime_fn=F_prime_zero,
        F_double_prime_fn=zero_fn,
        psi_axis=psi_min,
        psi_separatrix=psi_max,
        p_coeffs=zero,
        F_sq_coeffs=F_sq,
        bin_centers=torch.zeros(1, device=device),
        p_bin_values=torch.zeros(1, device=device),
        I_enc_bin_values=torch.zeros(1, device=device),
    )


def build_rudy_field(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    geom: CylindricalGeometry,
) -> torch.Tensor:
    """Vectorized RUDY-style routing demand field on the grid.

    For each net edge (i, j), deposit a uniform rectangular demand over the
    bounding box of the two endpoints. Vectorized via a summed-area-table
    (integral-image) trick: increment corner-sentinels in a `(rows+1, cols+1)`
    `diff` buffer, then take a 2D cumulative sum to recover the rectangle.
    """
    rows, cols = geom.grid_rows, geom.grid_cols
    device = geom.device
    if edge_index.numel() == 0:
        return torch.zeros(rows, cols, device=device, dtype=torch.float32)

    n_macros = int(positions.shape[0])
    e_src = edge_index[:, 0].clamp(0, n_macros - 1).long()
    e_dst = edge_index[:, 1].clamp(0, n_macros - 1).long()
    pi = positions[e_src]  # [E, 2]
    pj = positions[e_dst]

    # Bounding box in (R, Z) coordinates - here R == x for our embedding.
    x_lo = torch.minimum(pi[:, 0], pj[:, 0])
    x_hi = torch.maximum(pi[:, 0], pj[:, 0])
    y_lo = torch.minimum(pi[:, 1], pj[:, 1])
    y_hi = torch.maximum(pi[:, 1], pj[:, 1])

    dx = x_hi - x_lo
    dy = y_hi - y_lo
    cw = geom.dR
    ch = geom.dZ
    bbox_w = dx + cw
    bbox_h = dy + ch
    demand = edge_weight.to(torch.float32) * (dx + dy + 1e-6) / torch.clamp(bbox_w * bbox_h, min=1e-12)

    c0 = torch.clamp((x_lo / cw).long(), 0, cols - 1)
    c1 = torch.clamp((x_hi / cw).long(), 0, cols - 1)
    r0 = torch.clamp((y_lo / ch).long(), 0, rows - 1)
    r1 = torch.clamp((y_hi / ch).long(), 0, rows - 1)

    diff = torch.zeros(rows + 1, cols + 1, device=device, dtype=torch.float32)
    flat = diff.view(-1)
    cols_buf = cols + 1
    # diff[r0, c0] += d
    flat.scatter_add_(0, r0 * cols_buf + c0, demand)
    # diff[r1+1, c0] -= d
    flat.scatter_add_(0, (r1 + 1) * cols_buf + c0, -demand)
    # diff[r0, c1+1] -= d
    flat.scatter_add_(0, r0 * cols_buf + (c1 + 1), -demand)
    # diff[r1+1, c1+1] += d
    flat.scatter_add_(0, (r1 + 1) * cols_buf + (c1 + 1), demand)

    q = torch.cumsum(torch.cumsum(diff, dim=0), dim=1)[:rows, :cols]
    mean_q = float(q.mean().item())
    if mean_q > 1e-9:
        q = q / mean_q
    return q


def _smooth_min_pair(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    """Differentiable approximation of min(a, b)."""
    sharp = max(1e-6, float(alpha))
    return -torch.logsumexp(torch.stack([-sharp * a, -sharp * b], dim=0), dim=0) / sharp


def _smooth_max_pair(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    """Differentiable approximation of max(a, b)."""
    sharp = max(1e-6, float(alpha))
    return torch.logsumexp(torch.stack([sharp * a, sharp * b], dim=0), dim=0) / sharp


def build_soft_rudy_field(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    geom: CylindricalGeometry,
    bbox_alpha: float = 16.0,
    rasterize_sharpness: float = 8.0,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Smooth RUDY routing-demand diagnostic.

    The hard RUDY field uses min/max endpoint boxes and integer grid
    rasterization. That is fast, but it gives a noisy diagnostic for
    gradient-alignment calibration. This EFIT-congestion diagnostic replaces
    both discontinuities with LogSumExp bbox edges and sigmoid raster masks.
    It is intentionally gated because the hard summed-area version is much
    cheaper and remains a good non-gradient diagnostic.
    """
    rows, cols = geom.grid_rows, geom.grid_cols
    device = geom.device
    if edge_index.numel() == 0:
        return torch.zeros(rows, cols, device=device, dtype=torch.float32)

    n_macros = int(positions.shape[0])
    e_src = edge_index[:, 0].clamp(0, n_macros - 1).long()
    e_dst = edge_index[:, 1].clamp(0, n_macros - 1).long()
    pos = positions.to(device=device, dtype=torch.float32)
    pi = pos[e_src]
    pj = pos[e_dst]

    x_lo = _smooth_min_pair(pi[:, 0], pj[:, 0], bbox_alpha)
    x_hi = _smooth_max_pair(pi[:, 0], pj[:, 0], bbox_alpha)
    y_lo = _smooth_min_pair(pi[:, 1], pj[:, 1], bbox_alpha)
    y_hi = _smooth_max_pair(pi[:, 1], pj[:, 1], bbox_alpha)

    dx = torch.clamp(x_hi - x_lo, min=0.0)
    dy = torch.clamp(y_hi - y_lo, min=0.0)
    cw = float(geom.dR)
    ch = float(geom.dZ)
    bbox_w = dx + cw
    bbox_h = dy + ch
    demand = edge_weight.to(device=device, dtype=torch.float32) * (dx + dy + 1e-6)
    demand = demand / torch.clamp(bbox_w * bbox_h, min=1e-12)

    x_centers = (torch.arange(cols, device=device, dtype=torch.float32) + 0.5) * cw
    y_centers = (torch.arange(rows, device=device, dtype=torch.float32) + 0.5) * ch
    sharp = max(1e-3, float(rasterize_sharpness))
    q = torch.zeros(rows, cols, device=device, dtype=torch.float32)
    chunk = max(1, int(chunk_size))

    for start in range(0, int(edge_index.shape[0]), chunk):
        end = min(start + chunk, int(edge_index.shape[0]))
        xl = x_lo[start:end].view(-1, 1)
        xh = x_hi[start:end].view(-1, 1)
        yl = y_lo[start:end].view(-1, 1)
        yh = y_hi[start:end].view(-1, 1)
        mx = torch.sigmoid(sharp * (x_centers.view(1, -1) - xl) / max(cw, 1e-9))
        mx = mx * torch.sigmoid(sharp * (xh - x_centers.view(1, -1)) / max(cw, 1e-9))
        my = torch.sigmoid(sharp * (y_centers.view(1, -1) - yl) / max(ch, 1e-9))
        my = my * torch.sigmoid(sharp * (yh - y_centers.view(1, -1)) / max(ch, 1e-9))
        weighted = demand[start:end].view(-1, 1, 1) * my[:, :, None] * mx[:, None, :]
        q = q + weighted.sum(dim=0)

    mean_q = float(q.mean().item())
    if mean_q > 1e-9:
        q = q / mean_q
    return q


def build_density_field(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    geom: CylindricalGeometry,
) -> torch.Tensor:
    """Vectorized macro density field (area-overlap per cell).

    Returns a [grid_rows, grid_cols] field of macro footprint coverage,
    normalized so that a fully-covered cell has value 1.
    """
    rows, cols = geom.grid_rows, geom.grid_cols
    device = geom.device
    n = int(positions.shape[0])
    if n == 0:
        return torch.zeros(rows, cols, device=device, dtype=torch.float32)

    cw, ch = geom.dR, geom.dZ
    cell_area = cw * ch

    x = positions[:, 0]
    y = positions[:, 1]
    w = sizes[:, 0]
    h = sizes[:, 1]

    x_lo = x - 0.5 * w
    x_hi = x + 0.5 * w
    y_lo = y - 0.5 * h
    y_hi = y + 0.5 * h

    # Use integral-image trick with a uniform per-macro density = w*h/cell_area
    # spread over the bounding box. For first cut, treat as constant in the box.
    demand = (w * h) / max(cell_area, 1e-12)
    # We deposit demand / bbox_area_in_cells to give correct mean density.
    bbox_area_cells = torch.clamp(((x_hi - x_lo) / cw + 1.0) * ((y_hi - y_lo) / ch + 1.0), min=1.0)
    demand = demand / bbox_area_cells

    c0 = torch.clamp((x_lo / cw).long(), 0, cols - 1)
    c1 = torch.clamp((x_hi / cw).long(), 0, cols - 1)
    r0 = torch.clamp((y_lo / ch).long(), 0, rows - 1)
    r1 = torch.clamp((y_hi / ch).long(), 0, rows - 1)

    diff = torch.zeros(rows + 1, cols + 1, device=device, dtype=torch.float32)
    flat = diff.view(-1)
    cols_buf = cols + 1
    flat.scatter_add_(0, r0 * cols_buf + c0, demand)
    flat.scatter_add_(0, (r1 + 1) * cols_buf + c0, -demand)
    flat.scatter_add_(0, r0 * cols_buf + (c1 + 1), -demand)
    flat.scatter_add_(0, (r1 + 1) * cols_buf + (c1 + 1), demand)

    rho = torch.cumsum(torch.cumsum(diff, dim=0), dim=1)[:rows, :cols]
    return rho
