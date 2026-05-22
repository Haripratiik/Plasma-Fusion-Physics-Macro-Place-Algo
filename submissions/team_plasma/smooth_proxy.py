"""
Gyroaveraged smooth-hotspot surrogate for exact-score-gated refinement.

This is the minimum-viable HAAMP component from the research log: use a
gyrokinetic-style spatial average to turn sharp density/congestion tails into
a smooth field, then use that field only to propose candidate directions. The
official proxy still accepts or rejects every move.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from geometry import CylindricalGeometry, bilinear_sample
from gs_solver import gradient_of_psi
from profiles import build_density_field, build_rudy_field, build_soft_rudy_field


@dataclass
class SmoothProxyConfig:
    """Settings for the smooth-hotspot proposal field."""

    enabled: bool = False
    density_weight: float = 0.5
    rudy_weight: float = 0.5
    top_frac: float = 0.10
    lse_tau: float = 0.10
    gyro_radius_frac: float = 0.025
    hpwl_weight: float = 0.0
    smooth_rudy_enabled: bool = False
    smooth_rudy_bbox_alpha: float = 16.0
    smooth_rudy_rasterize_sharpness: float = 8.0
    bohm_sheath_boundary_enabled: bool = False
    bohm_sheath_boundary_weight: float = 0.25
    bohm_sheath_boundary_width_frac: float = 0.15
    bohm_sheath_limiter_enabled: bool = False
    bohm_sheath_limiter_quantile: float = 0.75
    large_macro_boundary_bias_enabled: bool = False
    large_macro_boundary_weight: float = 0.5
    large_macro_boundary_quantile: float = 0.75
    priority_beta: float = 1.0
    direction_beta: float = 1.0


def _bessel_j0(x: torch.Tensor) -> torch.Tensor:
    """Bessel J0 with a Gaussian fallback for older torch builds."""
    special = getattr(torch, "special", None)
    fn = getattr(special, "bessel_j0", None) if special is not None else None
    if fn is not None:
        return fn(x)
    # For proposal generation, a stable low-pass fallback is better than
    # failing at import time on older environments.
    return torch.exp(-0.25 * x * x)


def gyroaverage_isotropic(field: torch.Tensor, geom: CylindricalGeometry, rho: float) -> torch.Tensor:
    """Apply a J0^2 gyroaverage low-pass filter in Fourier space."""
    if rho <= 0.0 or field.numel() == 0:
        return field

    field_k = torch.fft.rfft2(field)
    ky = torch.fft.fftfreq(
        geom.grid_rows,
        d=float(geom.canvas_height) / float(geom.grid_rows),
        device=field.device,
    )
    kx = torch.fft.rfftfreq(
        geom.grid_cols,
        d=float(geom.canvas_width) / float(geom.grid_cols),
        device=field.device,
    )
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    k_mag = torch.sqrt(KX * KX + KY * KY) * (2.0 * torch.pi)
    kernel = _bessel_j0(k_mag * float(rho))
    kernel = kernel * kernel
    out = torch.fft.irfft2(field_k * kernel, s=field.shape)
    return torch.clamp(out.real, min=0.0)


def smooth_top_tail_field(field: torch.Tensor, cfg: SmoothProxyConfig) -> torch.Tensor:
    """Return a smooth emphasis field for top-tail cells.

    We keep this intentionally simple and robust: normalize the field, compute
    a quantile threshold, then use a softplus tail above that threshold. This
    approximates the top-k tail while preserving useful spatial gradients.
    """
    if field.numel() == 0:
        return field
    x = field.to(torch.float32)
    mean = torch.clamp(x.mean(), min=1e-6)
    x = x / mean
    frac = min(0.50, max(0.01, float(cfg.top_frac)))
    threshold = torch.quantile(x.reshape(-1), max(0.0, min(0.99, 1.0 - frac)))
    tau = max(1e-4, float(cfg.lse_tau))
    return tau * torch.nn.functional.softplus((x - threshold) / tau)


def bohm_sheath_boundary_field(
    geom: CylindricalGeometry,
    device: torch.device,
    positions: Optional[torch.Tensor] = None,
    sizes: Optional[torch.Tensor] = None,
    fixed_mask: Optional[torch.Tensor] = None,
    cfg: Optional[SmoothProxyConfig] = None,
) -> torch.Tensor:
    """Return an interior pressure field whose descent points toward edges.

    In the Bohm-presheath analogy, the canvas edge acts as an open limiter
    surface. We add a smooth interior pressure that is low at the sheath edge
    and high in the plasma bulk; the proposal direction uses ``-grad(field)``,
    so macros on this component drift outward toward the limiter.
    """
    rows = torch.arange(geom.grid_rows, device=device, dtype=torch.float32)
    cols = torch.arange(geom.grid_cols, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    denom_x = max(float(geom.grid_cols - 1), 1.0)
    denom_y = max(float(geom.grid_rows - 1), 1.0)
    d_left = xx / denom_x
    d_right = (denom_x - xx) / denom_x
    d_top = yy / denom_y
    d_bottom = (denom_y - yy) / denom_y
    d_edge = torch.minimum(torch.minimum(d_left, d_right), torch.minimum(d_top, d_bottom))
    if (
        cfg is not None
        and bool(getattr(cfg, "bohm_sheath_limiter_enabled", False))
        and positions is not None
        and sizes is not None
        and fixed_mask is not None
        and positions.numel() > 0
    ):
        pos = positions.detach().to(device=device, dtype=torch.float32)
        sz = sizes.detach().to(device=device, dtype=torch.float32)
        fixed = fixed_mask.detach().to(device=device).bool()
        areas = sz[:, 0] * sz[:, 1]
        fixed_areas = areas[fixed]
        if fixed_areas.numel() > 0:
            q = min(0.99, max(0.0, float(getattr(cfg, "bohm_sheath_limiter_quantile", 0.75))))
            threshold = torch.quantile(fixed_areas, q)
            selected = fixed & (areas >= threshold)
            x_grid = (xx + 0.5) * float(geom.canvas_width) / float(geom.grid_cols)
            y_grid = (yy + 0.5) * float(geom.canvas_height) / float(geom.grid_rows)
            span = max(float(geom.canvas_width), float(geom.canvas_height), 1e-9)
            for idx in torch.nonzero(selected, as_tuple=False).flatten().tolist():
                cx = pos[int(idx), 0]
                cy = pos[int(idx), 1]
                half_w = 0.5 * sz[int(idx), 0]
                half_h = 0.5 * sz[int(idx), 1]
                dx = torch.clamp(torch.abs(x_grid - cx) - half_w, min=0.0)
                dy = torch.clamp(torch.abs(y_grid - cy) - half_h, min=0.0)
                dist = torch.sqrt(dx * dx + dy * dy) / span
                d_edge = torch.minimum(d_edge, dist)
    return d_edge


def build_smooth_hotspot_field(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    geom: CylindricalGeometry,
    cfg: SmoothProxyConfig,
    fixed_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build a gyroaveraged density/RUDY tail field on the GS grid."""
    rho = build_density_field(positions, sizes, geom)
    if bool(getattr(cfg, "smooth_rudy_enabled", False)):
        q = build_soft_rudy_field(
            positions,
            edge_index,
            edge_weight,
            geom,
            bbox_alpha=float(getattr(cfg, "smooth_rudy_bbox_alpha", 16.0)),
            rasterize_sharpness=float(getattr(cfg, "smooth_rudy_rasterize_sharpness", 8.0)),
        )
    else:
        q = build_rudy_field(positions, edge_index, edge_weight, geom)
    raw = float(cfg.density_weight) * smooth_top_tail_field(rho, cfg)
    raw = raw + float(cfg.rudy_weight) * smooth_top_tail_field(q, cfg)
    if bool(getattr(cfg, "bohm_sheath_boundary_enabled", False)):
        width = max(1e-4, float(getattr(cfg, "bohm_sheath_boundary_width_frac", 0.15)))
        sheath = torch.tanh(bohm_sheath_boundary_field(geom, raw.device, positions, sizes, fixed_mask, cfg) / width)
        raw = raw + max(0.0, float(getattr(cfg, "bohm_sheath_boundary_weight", 0.25))) * sheath
    if bool(getattr(cfg, "large_macro_boundary_bias_enabled", False)):
        width = max(1e-4, float(getattr(cfg, "bohm_sheath_boundary_width_frac", 0.15)))
        limiter = torch.tanh(bohm_sheath_boundary_field(geom, raw.device, positions, sizes, fixed_mask, cfg) / width)
        raw = raw + max(0.0, float(getattr(cfg, "large_macro_boundary_weight", 0.5))) * limiter
    span = max(float(geom.canvas_width), float(geom.canvas_height), 1e-9)
    radius = max(0.0, float(cfg.gyro_radius_frac)) * span
    return gyroaverage_isotropic(raw, geom, radius)


def _hpwl_transport_descent(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    """Return graph-tension descent directions for the hard-macro net graph.

    In the Proxy-EFIT interpretation this is the field-line-tension term:
    gyroaveraged pressure marks where the plasma wants to decompress, while
    graph tension keeps connected current channels from stretching apart.
    """
    n = int(positions.shape[0])
    out = torch.zeros(n, 2, dtype=torch.float32, device=positions.device)
    if edge_index.numel() == 0 or n == 0:
        return out
    pos = positions.detach().to(dtype=torch.float32)
    edges = edge_index.detach().to(device=positions.device, dtype=torch.long)
    weights = edge_weight.detach().to(device=positions.device, dtype=torch.float32)
    for e in range(int(edges.shape[0])):
        a = int(edges[e, 0].item())
        b = int(edges[e, 1].item())
        if not (0 <= a < n and 0 <= b < n and a != b):
            continue
        diff = pos[a] - pos[b]
        norm = torch.linalg.norm(diff).clamp_min(1e-9)
        grad = weights[e] * diff / norm
        out[a] -= grad
        out[b] += grad
    return out


def _unit_rows(v: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(1e-12)
    return v / norms


def sample_smooth_hotspot_guidance(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    geom: CylindricalGeometry,
    cfg: SmoothProxyConfig,
    fixed_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-macro hotspot priority and descent directions.

    Priority is the smoothed hotspot value at each macro. Direction is
    negative gradient of the hotspot field, i.e. move away from the local
    tail-pressure ridge. Both are proposals only; exact proxy gates all moves.
    """
    field = build_smooth_hotspot_field(positions, sizes, edge_index, edge_weight, geom, cfg, fixed_mask=fixed_mask)
    gx, gy = gradient_of_psi(field, geom)
    priority = bilinear_sample(field, positions, geom)
    dx = -bilinear_sample(gx, positions, geom)
    dy = -bilinear_sample(gy, positions, geom)
    directions = torch.stack([dx, dy], dim=1)
    if bool(getattr(cfg, "large_macro_boundary_bias_enabled", False)) and sizes.numel() > 0:
        areas = (sizes[:, 0] * sizes[:, 1]).detach().to(device=positions.device, dtype=torch.float32)
        q = min(0.99, max(0.0, float(getattr(cfg, "large_macro_boundary_quantile", 0.75))))
        threshold = torch.quantile(areas, q) if areas.numel() > 0 else torch.tensor(0.0, device=positions.device)
        scale = torch.clamp(areas / torch.clamp(threshold, min=1e-9), min=0.0, max=4.0)
        priority = priority * (1.0 + max(0.0, float(getattr(cfg, "large_macro_boundary_weight", 0.5))) * scale)
    hpwl_weight = max(0.0, float(getattr(cfg, "hpwl_weight", 0.0)))
    if hpwl_weight > 0.0:
        pressure_dirs = _unit_rows(directions)
        tension_dirs = _unit_rows(_hpwl_transport_descent(positions, edge_index, edge_weight))
        directions = _unit_rows(pressure_dirs + hpwl_weight * tension_dirs)
    return priority, directions
