"""
GS-inspired plasma field core for macro placement.

This module adapts equilibrium-style PDE ideas to placement:
  div(kappa * grad(psi)) - eta * psi = rhs

where rhs blends density overflow pressure, RUDY-like routing pressure,
net-attraction source, and wall confinement field.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from . import jax_accel
except Exception:  # pragma: no cover - standalone import fallback
    try:
        import jax_accel  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        jax_accel = None


def _normalize_vector_force(force: torch.Tensor) -> torch.Tensor:
    n = int(force.shape[0])
    if n <= 0:
        return force
    norm = torch.linalg.norm(force, dim=1)
    denom = torch.quantile(norm, 0.90) if n >= 10 else torch.max(norm)
    d = float(denom.item()) if norm.numel() > 0 else 0.0
    if d > 1e-6:
        force = force / d
    return force


def _normalize_scalar_field(field: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    if field.numel() == 0:
        return field
    norm_mode = str(mode or "mean").strip().lower()
    flat = field.reshape(-1)
    if norm_mode == "sum":
        ref = float(flat.sum().item())
    elif norm_mode == "p90":
        ref = float(torch.quantile(flat, 0.90).item()) if flat.numel() > 0 else 0.0
    elif norm_mode == "p95":
        ref = float(torch.quantile(flat, 0.95).item()) if flat.numel() > 0 else 0.0
    else:
        ref = float(flat.mean().item())
    if ref > 1e-6:
        field = field / ref
    return field


def _hotspot_gain(
    samples: torch.Tensor,
    alpha: float,
    quantile: float = 0.85,
    power: float = 1.0,
    cap: float = 3.0,
) -> torch.Tensor:
    """Amplify forces for macros sitting inside the hottest pressure regions."""
    if samples.numel() == 0 or float(alpha) <= 0.0:
        return torch.ones(samples.shape[0], 1, dtype=samples.dtype, device=samples.device)

    flat = samples.reshape(-1).float()
    q = max(0.50, min(0.99, float(quantile)))
    ref = torch.quantile(flat, q)
    ref_val = max(float(ref.item()), 1e-6)

    scaled = torch.clamp(samples.float() / ref_val - 1.0, min=0.0)
    p = max(float(power), 1e-6)
    if p != 1.0:
        scaled = torch.pow(scaled, p)

    gain = 1.0 + float(alpha) * scaled
    max_cap = max(float(cap), 1.0)
    gain = torch.clamp(gain, min=1.0, max=max_cap)
    return gain.unsqueeze(1)


def _grid_geometry(
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
    device: torch.device,
) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    cw = float(canvas_width) / float(max(cols, 1))
    ch = float(canvas_height) / float(max(rows, 1))
    xs = (torch.arange(cols, device=device, dtype=torch.float32) + 0.5) * cw
    ys = (torch.arange(rows, device=device, dtype=torch.float32) + 0.5) * ch
    return cw, ch, xs, ys


def _grid_centers(
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
    device: torch.device,
) -> torch.Tensor:
    _, _, xs, ys = _grid_geometry(canvas_width, canvas_height, rows, cols, device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


def deposit_macro_density(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Area-overlap density map on a rows x cols grid (utilization ratio per cell)."""
    device = positions.device
    density = torch.zeros(rows, cols, dtype=torch.float32, device=device)

    if positions.numel() == 0 or rows <= 0 or cols <= 0:
        return density

    cw, ch, xs, ys = _grid_geometry(canvas_width, canvas_height, rows, cols, device)
    half_cw = 0.5 * cw
    half_ch = 0.5 * ch
    cell_area = max(cw * ch, 1e-12)

    for i in range(int(positions.shape[0])):
        x = float(positions[i, 0].item())
        y = float(positions[i, 1].item())
        w = float(sizes[i, 0].item())
        h = float(sizes[i, 1].item())

        ovx = torch.relu((0.5 * w + half_cw) - torch.abs(xs - x))
        if float(ovx.max().item()) <= 0.0:
            continue
        ovy = torch.relu((0.5 * h + half_ch) - torch.abs(ys - y))
        if float(ovy.max().item()) <= 0.0:
            continue

        density += torch.outer(ovy, ovx) / cell_area

    return density


def _deposit_points_bilinear(
    points: torch.Tensor,
    weights: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Scatter points to a grid with bilinear weights."""
    device = points.device
    out = torch.zeros(rows, cols, dtype=torch.float32, device=device)
    if points.numel() == 0:
        return out

    cw = float(canvas_width) / float(max(cols, 1))
    ch = float(canvas_height) / float(max(rows, 1))

    cx = torch.clamp(points[:, 0] / max(cw, 1e-12) - 0.5, 0.0, float(cols - 1))
    cy = torch.clamp(points[:, 1] / max(ch, 1e-12) - 0.5, 0.0, float(rows - 1))

    x0 = torch.floor(cx).long()
    y0 = torch.floor(cy).long()
    x1 = torch.clamp(x0 + 1, max=cols - 1)
    y1 = torch.clamp(y0 + 1, max=rows - 1)

    tx = cx - x0.float()
    ty = cy - y0.float()

    w00 = (1.0 - tx) * (1.0 - ty) * weights
    w10 = tx * (1.0 - ty) * weights
    w01 = (1.0 - tx) * ty * weights
    w11 = tx * ty * weights

    # Accumulate in a fixed point order rather than relying on scatter-style
    # kernel accumulation, which can introduce process-to-process jitter.
    count = int(points.shape[0])
    for idx in range(count):
        iy0 = int(y0[idx].item())
        ix0 = int(x0[idx].item())
        iy1 = int(y1[idx].item())
        ix1 = int(x1[idx].item())
        out[iy0, ix0] += w00[idx]
        out[iy0, ix1] += w10[idx]
        out[iy1, ix0] += w01[idx]
        out[iy1, ix1] += w11[idx]
    return out


def build_pin_density_pressure(
    positions: torch.Tensor,
    pin_index: torch.Tensor,
    pin_offsets: torch.Tensor,
    pin_weight: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
    smoothing_passes: int = 1,
    norm_mode: str = "mean",
) -> torch.Tensor:
    """Build a pin-access pressure map from hard-macro pin locations."""
    if (
        positions.numel() == 0
        or pin_index.numel() == 0
        or pin_weight.numel() == 0
        or pin_offsets.numel() == 0
    ):
        return torch.zeros(rows, cols, dtype=torch.float32, device=positions.device)

    device = positions.device
    pin_pts = positions[pin_index.long()].float() + pin_offsets.to(device=device, dtype=torch.float32)
    pressure = _deposit_points_bilinear(
        pin_pts,
        pin_weight.to(device=device, dtype=torch.float32),
        canvas_width,
        canvas_height,
        rows,
        cols,
    )

    if smoothing_passes > 0:
        pv = pressure.view(1, 1, rows, cols)
        for _ in range(int(smoothing_passes)):
            pv = F.avg_pool2d(pv, kernel_size=3, stride=1, padding=1)
        pressure = pv.view(rows, cols)

    return _normalize_scalar_field(pressure, norm_mode)


def build_porosity_pressure(
    hard_density: torch.Tensor,
    bg_density: Optional[torch.Tensor],
    cfg: Dict[str, object],
) -> torch.Tensor:
    """
    Build a low-porosity pressure field from local occupancy.

    The field rises when a neighborhood's combined hard-macro occupancy and
    soft-background occupancy exceed a target threshold, which helps the PDE
    preserve usable whitespace and avoid collapsing narrow routing corridors.
    """
    if hard_density.numel() == 0 or not bool(cfg.get("enabled", False)):
        return torch.zeros_like(hard_density)

    rows, cols = hard_density.shape
    hard_scale = max(float(cfg.get("hard_scale", 1.0)), 0.0)
    bg_scale = max(float(cfg.get("bg_scale", 0.0)), 0.0)
    occ_cap = float(cfg.get("occupancy_cap", 0.0))
    pre_smoothing = max(int(cfg.get("pre_smoothing", 0)), 0)
    post_smoothing = max(int(cfg.get("post_smoothing", 0)), 0)
    target = max(float(cfg.get("target_occupancy", 0.45)), 0.0)
    overflow_power = max(float(cfg.get("overflow_power", 1.0)), 1e-6)
    norm_mode = str(cfg.get("norm_mode", "mean"))

    occupancy = hard_scale * torch.clamp(hard_density, min=0.0)
    if bg_scale > 0.0 and bg_density is not None and bg_density.numel() == hard_density.numel():
        occupancy = occupancy + bg_scale * torch.clamp(bg_density, min=0.0)
    if occ_cap > 0.0:
        occupancy = torch.clamp(occupancy, max=occ_cap)

    occv = occupancy.view(1, 1, rows, cols)
    for _ in range(pre_smoothing):
        occv = F.avg_pool2d(occv, kernel_size=3, stride=1, padding=1)

    window_cells = int(cfg.get("window_cells", 0))
    if window_cells <= 0:
        frac = max(float(cfg.get("window_frac", 0.18)), 0.0)
        window_cells = max(1, int(round(frac * max(rows, cols))))
    if window_cells % 2 == 0:
        window_cells += 1
    window_cells = min(window_cells, max(1, 2 * min(rows, cols) - 1))

    if window_cells > 1:
        occv = F.avg_pool2d(occv, kernel_size=window_cells, stride=1, padding=window_cells // 2)
    local_occ = occv.view(rows, cols)

    pressure = torch.relu(local_occ - target)
    if overflow_power != 1.0:
        pressure = torch.pow(pressure + 1e-8, overflow_power)

    pv = pressure.view(1, 1, rows, cols)
    for _ in range(post_smoothing):
        pv = F.avg_pool2d(pv, kernel_size=3, stride=1, padding=1)
    pressure = pv.view(rows, cols)
    return _normalize_scalar_field(pressure, norm_mode)


def build_density_pressure(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
    target_density: float,
    overflow_power: float,
) -> torch.Tensor:
    density = deposit_macro_density(positions, sizes, canvas_width, canvas_height, rows, cols)
    overflow = torch.relu(density - float(target_density))
    if overflow_power != 1.0:
        overflow = torch.pow(overflow + 1e-8, float(overflow_power))
    return overflow


def build_rudy_pressure(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
    edge_offsets: Optional[torch.Tensor] = None,
    smoothing_passes: int = 1,
) -> torch.Tensor:
    """RUDY-like rectangular demand accumulation, normalized by mean."""
    device = positions.device
    q = torch.zeros(rows, cols, dtype=torch.float32, device=device)
    if edge_index.numel() == 0:
        return q

    cw = float(canvas_width) / float(max(cols, 1))
    ch = float(canvas_height) / float(max(rows, 1))

    diff = torch.zeros(rows + 1, cols + 1, dtype=torch.float32, device=device)

    for e in range(int(edge_index.shape[0])):
        i = int(edge_index[e, 0].item())
        j = int(edge_index[e, 1].item())
        w = float(edge_weight[e].item()) if edge_weight.numel() else 1.0

        xi = float(positions[i, 0].item())
        yi = float(positions[i, 1].item())
        xj = float(positions[j, 0].item())
        yj = float(positions[j, 1].item())
        if edge_offsets is not None and edge_offsets.numel() > 0:
            xi += float(edge_offsets[e, 0, 0].item())
            yi += float(edge_offsets[e, 0, 1].item())
            xj += float(edge_offsets[e, 1, 0].item())
            yj += float(edge_offsets[e, 1, 1].item())

        dx = abs(xi - xj)
        dy = abs(yi - yj)

        bbox_w = dx + cw
        bbox_h = dy + ch
        demand = w * (dx + dy + 1e-6) / max(bbox_w * bbox_h, 1e-12)

        c0 = int(min(xi, xj) / max(cw, 1e-12))
        c1 = int(max(xi, xj) / max(cw, 1e-12))
        r0 = int(min(yi, yj) / max(ch, 1e-12))
        r1 = int(max(yi, yj) / max(ch, 1e-12))

        c0 = max(0, min(c0, cols - 1))
        c1 = max(0, min(c1, cols - 1))
        r0 = max(0, min(r0, rows - 1))
        r1 = max(0, min(r1, rows - 1))

        diff[r0, c0] += demand
        diff[r1 + 1, c0] -= demand
        diff[r0, c1 + 1] -= demand
        diff[r1 + 1, c1 + 1] += demand

    q = torch.cumsum(torch.cumsum(diff, dim=0), dim=1)[:rows, :cols]

    if smoothing_passes > 0:
        qv = q.view(1, 1, rows, cols)
        for _ in range(int(smoothing_passes)):
            qv = F.avg_pool2d(qv, kernel_size=3, stride=1, padding=1)
        q = qv.view(rows, cols)

    mean_val = float(q.mean().item())
    if mean_val > 1e-12:
        q = q / mean_val
    return q


def build_bundle_rudy_pressure(
    positions: torch.Tensor,
    bundle_ptr: torch.Tensor,
    bundle_index: torch.Tensor,
    bundle_offsets: torch.Tensor,
    bundle_weight: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
    smoothing_passes: int = 1,
) -> torch.Tensor:
    """
    RUDY-like demand accumulation over whole multi-pin net bundles.

    Each bundle contributes a single rectangular demand over the bounding box of
    all connected hard macros, which better matches the original RUDY intuition
    than decomposing every multi-pin net solely into pairwise rectangles.
    """
    device = positions.device
    q = torch.zeros(rows, cols, dtype=torch.float32, device=device)
    num_bundles = int(bundle_weight.shape[0])
    if num_bundles <= 0 or bundle_ptr.numel() < 2 or bundle_index.numel() == 0:
        return q

    cw = float(canvas_width) / float(max(cols, 1))
    ch = float(canvas_height) / float(max(rows, 1))
    diff = torch.zeros(rows + 1, cols + 1, dtype=torch.float32, device=device)

    for b in range(num_bundles):
        start = int(bundle_ptr[b].item())
        stop = int(bundle_ptr[b + 1].item())
        if stop - start < 2:
            continue

        members = bundle_index[start:stop].long()
        pts = positions[members]
        if bundle_offsets.numel() > 0:
            pts = pts + bundle_offsets[start:stop].to(device=device, dtype=positions.dtype)

        xs = pts[:, 0]
        ys = pts[:, 1]
        x_lo = float(xs.min().item())
        x_hi = float(xs.max().item())
        y_lo = float(ys.min().item())
        y_hi = float(ys.max().item())

        dx = x_hi - x_lo
        dy = y_hi - y_lo
        w = float(bundle_weight[b].item()) if bundle_weight.numel() else 1.0
        bbox_w = dx + cw
        bbox_h = dy + ch
        demand = w * (dx + dy + 1e-6) / max(bbox_w * bbox_h, 1e-12)

        c0 = int(x_lo / max(cw, 1e-12))
        c1 = int(x_hi / max(cw, 1e-12))
        r0 = int(y_lo / max(ch, 1e-12))
        r1 = int(y_hi / max(ch, 1e-12))

        c0 = max(0, min(c0, cols - 1))
        c1 = max(0, min(c1, cols - 1))
        r0 = max(0, min(r0, rows - 1))
        r1 = max(0, min(r1, rows - 1))

        diff[r0, c0] += demand
        diff[r1 + 1, c0] -= demand
        diff[r0, c1 + 1] -= demand
        diff[r1 + 1, c1 + 1] += demand

    q = torch.cumsum(torch.cumsum(diff, dim=0), dim=1)[:rows, :cols]

    if smoothing_passes > 0:
        qv = q.view(1, 1, rows, cols)
        for _ in range(int(smoothing_passes)):
            qv = F.avg_pool2d(qv, kernel_size=3, stride=1, padding=1)
        q = qv.view(rows, cols)

    mean_val = float(q.mean().item())
    if mean_val > 1e-12:
        q = q / mean_val
    return q


def build_bundle_net_source(
    positions: torch.Tensor,
    bundle_ptr: torch.Tensor,
    bundle_index: torch.Tensor,
    bundle_offsets: torch.Tensor,
    bundle_weight: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
    degree_boost: float = 0.0,
) -> torch.Tensor:
    """
    Whole-net source term using a star / centroid-style bundle model.

    Each multi-pin bundle pulls its member pins toward the bundle centroid,
    preserving hyperedge structure more directly than pairwise decomposition.
    """
    device = positions.device
    num_bundles = int(bundle_weight.shape[0])
    if num_bundles <= 0 or bundle_ptr.numel() < 2 or bundle_index.numel() == 0:
        return torch.zeros(rows, cols, dtype=torch.float32, device=device)

    current_pts: List[torch.Tensor] = []
    target_pts: List[torch.Tensor] = []
    point_weights: List[float] = []

    for b in range(num_bundles):
        start = int(bundle_ptr[b].item())
        stop = int(bundle_ptr[b + 1].item())
        degree = stop - start
        if degree < 2:
            continue

        members = bundle_index[start:stop].long()
        pts = positions[members]
        if bundle_offsets.numel() > 0:
            pts = pts + bundle_offsets[start:stop].to(device=device, dtype=positions.dtype)
        centroid = pts.mean(dim=0, keepdim=True)

        w = float(bundle_weight[b].item()) if bundle_weight.numel() else 1.0
        if degree_boost != 0.0:
            w *= float(max(degree, 1) ** degree_boost)
        for i in range(degree):
            current_pts.append(pts[i])
            target_pts.append(centroid[0])
            point_weights.append(w)

    if not current_pts:
        return torch.zeros(rows, cols, dtype=torch.float32, device=device)

    current_tensor = torch.stack(current_pts, dim=0)
    target_tensor = torch.stack(target_pts, dim=0)
    weight_tensor = torch.tensor(point_weights, dtype=torch.float32, device=device)

    src_from = _deposit_points_bilinear(
        current_tensor,
        weight_tensor,
        canvas_width,
        canvas_height,
        rows,
        cols,
    )
    src_to = _deposit_points_bilinear(
        target_tensor,
        weight_tensor,
        canvas_width,
        canvas_height,
        rows,
        cols,
    )

    source = src_from - src_to
    scale = float(source.abs().mean().item())
    if scale > 1e-12:
        source = source / scale
    return source


def build_net_source(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
    edge_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Net source term for PDE.

    Positive values indicate "mass should move away", negative values indicate
    "mass should gather", built from current-vs-neighbor-centroid mismatch.
    """
    n = int(positions.shape[0])
    device = positions.device
    if n == 0 or edge_index.numel() == 0:
        return torch.zeros(rows, cols, dtype=torch.float32, device=device)

    if edge_offsets is None or edge_offsets.numel() == 0:
        sum_w = torch.zeros(n, dtype=torch.float32, device=device)
        centroid = torch.zeros(n, 2, dtype=torch.float32, device=device)
        degree = torch.zeros(n, dtype=torch.float32, device=device)

        for e in range(int(edge_index.shape[0])):
            i = int(edge_index[e, 0].item())
            j = int(edge_index[e, 1].item())
            w = float(edge_weight[e].item()) if edge_weight.numel() else 1.0

            wi = float(w)
            sum_w[i] += wi
            sum_w[j] += wi
            centroid[i] += wi * positions[j]
            centroid[j] += wi * positions[i]
            degree[i] += wi
            degree[j] += wi

        target = positions.clone()
        mask = sum_w > 1e-12
        if bool(mask.any()):
            target[mask] = centroid[mask] / sum_w[mask].unsqueeze(1)

        point_weight = torch.clamp(degree, min=1e-3)
        src_from = _deposit_points_bilinear(
            positions,
            point_weight,
            canvas_width,
            canvas_height,
            rows,
            cols,
        )
        src_to = _deposit_points_bilinear(
            target,
            point_weight,
            canvas_width,
            canvas_height,
            rows,
            cols,
        )

        source = src_from - src_to
        scale = float(source.abs().mean().item())
        if scale > 1e-12:
            source = source / scale
        return source

    current_pts = []
    target_pts = []
    point_weights = []

    for e in range(int(edge_index.shape[0])):
        i = int(edge_index[e, 0].item())
        j = int(edge_index[e, 1].item())
        w = float(edge_weight[e].item()) if edge_weight.numel() else 1.0

        pi = positions[i].clone()
        pj = positions[j].clone()
        if edge_offsets is not None and edge_offsets.numel() > 0:
            pi = pi + edge_offsets[e, 0].to(device=device, dtype=positions.dtype)
            pj = pj + edge_offsets[e, 1].to(device=device, dtype=positions.dtype)

        current_pts.extend((pi, pj))
        target_pts.extend((pj, pi))
        point_weights.extend((w, w))

    if not current_pts:
        return torch.zeros(rows, cols, dtype=torch.float32, device=device)

    current_tensor = torch.stack(current_pts, dim=0)
    target_tensor = torch.stack(target_pts, dim=0)
    weight_tensor = torch.tensor(point_weights, dtype=torch.float32, device=device)

    src_from = _deposit_points_bilinear(
        current_tensor,
        weight_tensor,
        canvas_width,
        canvas_height,
        rows,
        cols,
    )
    src_to = _deposit_points_bilinear(
        target_tensor,
        weight_tensor,
        canvas_width,
        canvas_height,
        rows,
        cols,
    )

    source = src_from - src_to
    scale = float(source.abs().mean().item())
    if scale > 1e-12:
        source = source / scale
    return source


def build_anchor_pressure(
    positions: torch.Tensor,
    anchor_targets: torch.Tensor,
    anchor_weights: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
    smoothing_passes: int = 1,
) -> torch.Tensor:
    """RUDY-style demand induced by hard-to-anchor target connections."""
    device = positions.device
    q = torch.zeros(rows, cols, dtype=torch.float32, device=device)
    if positions.numel() == 0 or anchor_targets.numel() == 0 or anchor_weights.numel() == 0:
        return q

    weights = torch.clamp(anchor_weights.float().to(device), min=0.0)
    mask = weights > 1e-8
    if not bool(mask.any()):
        return q

    ref = torch.quantile(weights[mask], 0.90)
    ref_val = max(float(ref.item()), 1e-6)
    scaled_w = torch.clamp(torch.sqrt(weights / ref_val), min=0.0, max=2.5)

    cw = float(canvas_width) / float(max(cols, 1))
    ch = float(canvas_height) / float(max(rows, 1))
    diff = torch.zeros(rows + 1, cols + 1, dtype=torch.float32, device=device)

    for i in range(int(positions.shape[0])):
        w = float(scaled_w[i].item())
        if w <= 1e-8:
            continue

        xi = float(positions[i, 0].item())
        yi = float(positions[i, 1].item())
        xj = float(anchor_targets[i, 0].item())
        yj = float(anchor_targets[i, 1].item())

        dx = abs(xi - xj)
        dy = abs(yi - yj)
        bbox_w = dx + cw
        bbox_h = dy + ch
        demand = w * (dx + dy + 1e-6) / max(bbox_w * bbox_h, 1e-12)

        c0 = int(min(xi, xj) / max(cw, 1e-12))
        c1 = int(max(xi, xj) / max(cw, 1e-12))
        r0 = int(min(yi, yj) / max(ch, 1e-12))
        r1 = int(max(yi, yj) / max(ch, 1e-12))

        c0 = max(0, min(c0, cols - 1))
        c1 = max(0, min(c1, cols - 1))
        r0 = max(0, min(r0, rows - 1))
        r1 = max(0, min(r1, rows - 1))

        diff[r0, c0] += demand
        diff[r1 + 1, c0] -= demand
        diff[r0, c1 + 1] -= demand
        diff[r1 + 1, c1 + 1] += demand

    q = torch.cumsum(torch.cumsum(diff, dim=0), dim=1)[:rows, :cols]

    if smoothing_passes > 0:
        qv = q.view(1, 1, rows, cols)
        for _ in range(int(smoothing_passes)):
            qv = F.avg_pool2d(qv, kernel_size=3, stride=1, padding=1)
        q = qv.view(rows, cols)

    mean_val = float(q.mean().item())
    if mean_val > 1e-12:
        q = q / mean_val
    return q


def build_anchor_source(
    positions: torch.Tensor,
    anchor_targets: torch.Tensor,
    anchor_weights: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Net-source analogue induced by hard-to-anchor target motion."""
    device = positions.device
    if positions.numel() == 0 or anchor_targets.numel() == 0 or anchor_weights.numel() == 0:
        return torch.zeros(rows, cols, dtype=torch.float32, device=device)

    weights = torch.clamp(anchor_weights.float().to(device), min=0.0)
    mask = weights > 1e-8
    if not bool(mask.any()):
        return torch.zeros(rows, cols, dtype=torch.float32, device=device)

    ref = torch.quantile(weights[mask], 0.90)
    ref_val = max(float(ref.item()), 1e-6)
    point_weight = torch.clamp(torch.sqrt(weights / ref_val), min=0.0, max=2.5)

    src_from = _deposit_points_bilinear(
        positions[mask],
        point_weight[mask],
        canvas_width,
        canvas_height,
        rows,
        cols,
    )
    src_to = _deposit_points_bilinear(
        anchor_targets[mask],
        point_weight[mask],
        canvas_width,
        canvas_height,
        rows,
        cols,
    )

    source = src_from - src_to
    scale = float(source.abs().mean().item())
    if scale > 1e-12:
        source = source / scale
    return source


def build_wall_field(
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
    decay_frac: float,
    device: torch.device,
) -> torch.Tensor:
    """Higher near boundaries, lower in the interior."""
    _, _, xs, ys = _grid_geometry(canvas_width, canvas_height, rows, cols, device)
    xx = xs.unsqueeze(0).repeat(rows, 1)
    yy = ys.unsqueeze(1).repeat(1, cols)

    d_left = xx
    d_right = float(canvas_width) - xx
    d_bottom = yy
    d_top = float(canvas_height) - yy
    d = torch.minimum(torch.minimum(d_left, d_right), torch.minimum(d_bottom, d_top))

    span = max(float(canvas_width), float(canvas_height))
    decay = max(float(decay_frac) * span, 1e-6)
    wall = torch.exp(-d / decay)

    mean_val = float(wall.mean().item())
    if mean_val > 1e-12:
        wall = wall / mean_val
    return wall


def build_channel_permeability_maps(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    edge_profile: Optional[torch.Tensor],
    cfg: Dict[str, object],
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build directional channel barrier maps.

    barrier_x is large where transport across x should be suppressed
    (left-right macro corridors). barrier_y is analogous for top-bottom
    corridors. These maps can then bias the PDE toward tangential escape.
    """
    device = positions.device
    barrier_x = torch.zeros(rows, cols, dtype=torch.float32, device=device)
    barrier_y = torch.zeros(rows, cols, dtype=torch.float32, device=device)
    n = int(positions.shape[0])
    if n <= 1 or rows <= 0 or cols <= 0:
        return barrier_x, barrier_y

    if edge_profile is None or edge_profile.numel() == 0:
        profile = torch.zeros((n, 4), dtype=torch.float32, device=device)
    else:
        profile = edge_profile.to(device=device, dtype=torch.float32)

    span = max(float(cfg.get("channel_span_frac", 0.06)), 1e-4) * max(float(canvas_width), float(canvas_height))
    sigma = max(float(cfg.get("sigma_frac", 0.03)), 1e-4) * max(float(canvas_width), float(canvas_height))
    min_overlap_frac = max(float(cfg.get("min_overlap_frac", 0.15)), 0.0)
    inv_gap_power = max(float(cfg.get("inv_gap_power", 1.0)), 0.0)
    edge_power = max(float(cfg.get("edge_power", 1.0)), 1e-6)
    uniform_edge_bias = max(float(cfg.get("uniform_edge_bias", 0.0)), 0.0)
    min_gap = max(float(cfg.get("min_gap_frac", 0.002)), 1e-5) * max(float(canvas_width), float(canvas_height))
    cap = max(float(cfg.get("max_pair_pressure", 3.0)), 0.0)
    bleed = max(float(cfg.get("rect_bleed_frac", 0.10)), 0.0) * max(float(canvas_width), float(canvas_height))
    _, _, xs, ys = _grid_geometry(canvas_width, canvas_height, rows, cols, device)
    xx = xs.unsqueeze(0).repeat(rows, 1)
    yy = ys.unsqueeze(1).repeat(1, cols)

    def _normalize_map(field: torch.Tensor) -> torch.Tensor:
        if field.numel() == 0:
            return field
        q = float(cfg.get("normalize_quantile", 0.90))
        denom = torch.quantile(field, q) if field.numel() >= 10 else torch.max(field)
        d = float(denom.item())
        if d > 1e-6:
            field = field / d
        return field

    for i in range(n):
        xi = float(positions[i, 0].item())
        yi = float(positions[i, 1].item())
        wi = float(sizes[i, 0].item())
        hi = float(sizes[i, 1].item())
        li, ri, bi, ti = [float(v) for v in profile[i].tolist()]
        if uniform_edge_bias > 0.0:
            li += uniform_edge_bias
            ri += uniform_edge_bias
            bi += uniform_edge_bias
            ti += uniform_edge_bias

        for j in range(i + 1, n):
            xj = float(positions[j, 0].item())
            yj = float(positions[j, 1].item())
            wj = float(sizes[j, 0].item())
            hj = float(sizes[j, 1].item())
            lj, rj, bj, tj = [float(v) for v in profile[j].tolist()]
            if uniform_edge_bias > 0.0:
                lj += uniform_edge_bias
                rj += uniform_edge_bias
                bj += uniform_edge_bias
                tj += uniform_edge_bias

            dx = xj - xi
            dy = yj - yi
            overlap_y = max(0.0, 0.5 * (hi + hj) - abs(dy))
            overlap_x = max(0.0, 0.5 * (wi + wj) - abs(dx))

            gap_x = abs(dx) - 0.5 * (wi + wj)
            if gap_x >= 0.0 and gap_x < span and overlap_y > min_overlap_frac * max(min(hi, hj), 1e-6):
                facing = (ri * lj) if dx >= 0.0 else (li * rj)
                if facing > 0.0:
                    overlap_scale = overlap_y / max(min(hi, hj), 1e-6)
                    gap_eff = max(gap_x, min_gap)
                    pressure = (facing ** edge_power) * overlap_scale * math.exp(-gap_x / sigma) / (gap_eff ** inv_gap_power)
                    pressure = min(cap, pressure)
                    x_lo = min(xi + 0.5 * wi, xj + 0.5 * wj) if dx < 0.0 else xi + 0.5 * wi
                    x_hi = max(xi - 0.5 * wi, xj - 0.5 * wj) if dx < 0.0 else xj - 0.5 * wj
                    y_lo = max(yi - 0.5 * hi, yj - 0.5 * hj) - bleed
                    y_hi = min(yi + 0.5 * hi, yj + 0.5 * hj) + bleed
                    if x_hi > x_lo:
                        mask = (xx >= x_lo) & (xx <= x_hi) & (yy >= y_lo) & (yy <= y_hi)
                        barrier_x = barrier_x + pressure * mask.to(dtype=torch.float32)

            gap_y = abs(dy) - 0.5 * (hi + hj)
            if gap_y >= 0.0 and gap_y < span and overlap_x > min_overlap_frac * max(min(wi, wj), 1e-6):
                facing = (ti * bj) if dy >= 0.0 else (bi * tj)
                if facing > 0.0:
                    overlap_scale = overlap_x / max(min(wi, wj), 1e-6)
                    gap_eff = max(gap_y, min_gap)
                    pressure = (facing ** edge_power) * overlap_scale * math.exp(-gap_y / sigma) / (gap_eff ** inv_gap_power)
                    pressure = min(cap, pressure)
                    x_lo = max(xi - 0.5 * wi, xj - 0.5 * wj) - bleed
                    x_hi = min(xi + 0.5 * wi, xj + 0.5 * wj) + bleed
                    y_lo = min(yi + 0.5 * hi, yj + 0.5 * hj) if dy < 0.0 else yi + 0.5 * hi
                    y_hi = max(yi - 0.5 * hi, yj - 0.5 * hj) if dy < 0.0 else yj - 0.5 * hj
                    if y_hi > y_lo:
                        mask = (xx >= x_lo) & (xx <= x_hi) & (yy >= y_lo) & (yy <= y_hi)
                        barrier_y = barrier_y + pressure * mask.to(dtype=torch.float32)

    return _normalize_map(barrier_x), _normalize_map(barrier_y)


def apply_source_quench(
    n_src: torch.Tensor,
    q_over: torch.Tensor,
    bg_density: Optional[torch.Tensor],
    pin_over: Optional[torch.Tensor],
    porosity_over: Optional[torch.Tensor],
    channel_barrier_x: Optional[torch.Tensor],
    channel_barrier_y: Optional[torch.Tensor],
    cfg: Dict[str, object],
) -> torch.Tensor:
    """
    Attenuate net-attraction source in regions that already exhibit strong
    congestion, soft-background crowding, pin pressure, or corridor blockage.
    """
    if n_src.numel() == 0 or not bool(cfg.get("enabled", False)):
        return n_src

    q_alpha = max(float(cfg.get("q_alpha", 0.0)), 0.0)
    bg_alpha = max(float(cfg.get("bg_alpha", 0.0)), 0.0)
    pin_alpha = max(float(cfg.get("pin_alpha", 0.0)), 0.0)
    porosity_alpha = max(float(cfg.get("porosity_alpha", 0.0)), 0.0)
    channel_alpha = max(float(cfg.get("channel_alpha", 0.0)), 0.0)
    q_power = max(float(cfg.get("q_power", 1.0)), 1e-6)
    bg_power = max(float(cfg.get("bg_power", 1.0)), 1e-6)
    pin_power = max(float(cfg.get("pin_power", 1.0)), 1e-6)
    porosity_power = max(float(cfg.get("porosity_power", 1.0)), 1e-6)
    channel_power = max(float(cfg.get("channel_power", 1.0)), 1e-6)
    min_scale = max(min(float(cfg.get("min_scale", 0.15)), 1.0), 0.0)

    suppress = torch.ones_like(n_src)
    if q_alpha > 0.0:
        suppress = suppress + q_alpha * torch.pow(torch.clamp(q_over, min=0.0), q_power)
    if bg_alpha > 0.0 and bg_density is not None and bg_density.numel() == n_src.numel():
        suppress = suppress + bg_alpha * torch.pow(torch.clamp(bg_density, min=0.0), bg_power)
    if pin_alpha > 0.0 and pin_over is not None and pin_over.numel() == n_src.numel():
        suppress = suppress + pin_alpha * torch.pow(torch.clamp(pin_over, min=0.0), pin_power)
    if porosity_alpha > 0.0 and porosity_over is not None and porosity_over.numel() == n_src.numel():
        suppress = suppress + porosity_alpha * torch.pow(torch.clamp(porosity_over, min=0.0), porosity_power)
    if channel_alpha > 0.0 and channel_barrier_x is not None and channel_barrier_y is not None:
        barrier = torch.maximum(channel_barrier_x, channel_barrier_y)
        if barrier.numel() == n_src.numel():
            suppress = suppress + channel_alpha * torch.pow(torch.clamp(barrier, min=0.0), channel_power)

    scale = 1.0 / torch.clamp(suppress, min=1.0)
    scale = torch.clamp(scale, min=min_scale, max=1.0)
    return n_src * scale


def _compute_residual(
    psi: torch.Tensor,
    rhs: torch.Tensor,
    kappa: torch.Tensor,
    eta: float,
    dx: float,
    dy: float,
    kappa_y: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if psi.shape[0] < 3 or psi.shape[1] < 3:
        return torch.zeros((), dtype=torch.float32, device=psi.device)

    dx2 = max(dx * dx, 1e-12)
    dy2 = max(dy * dy, 1e-12)

    c = psi[1:-1, 1:-1]
    e = psi[1:-1, 2:]
    w = psi[1:-1, :-2]
    n = psi[:-2, 1:-1]
    s = psi[2:, 1:-1]

    kx = kappa
    ky = kappa if kappa_y is None else kappa_y
    kx_c = kx[1:-1, 1:-1]
    ky_c = ky[1:-1, 1:-1]
    ke = 0.5 * (kx_c + kx[1:-1, 2:])
    kw = 0.5 * (kx_c + kx[1:-1, :-2])
    kn = 0.5 * (ky_c + ky[:-2, 1:-1])
    ks = 0.5 * (ky_c + ky[2:, 1:-1])

    div_term = (
        ke * (e - c) / dx2
        - kw * (c - w) / dx2
        + ks * (s - c) / dy2
        - kn * (c - n) / dy2
    )
    lhs = div_term - float(eta) * c
    res = lhs - rhs[1:-1, 1:-1]
    return torch.mean(torch.abs(res))


def _apply_boundary_condition(
    psi: torch.Tensor,
    mode: str,
) -> None:
    rows, cols = psi.shape
    if rows == 0 or cols == 0:
        return

    bc_mode = str(mode).lower()
    if bc_mode == "neumann":
        if rows >= 2:
            psi[0, :] = psi[1, :]
            psi[-1, :] = psi[-2, :]
        if cols >= 2:
            psi[:, 0] = psi[:, 1]
            psi[:, -1] = psi[:, -2]
        return

    psi[0, :] = 0.0
    psi[-1, :] = 0.0
    psi[:, 0] = 0.0
    psi[:, -1] = 0.0


def _red_black_gauss_seidel(
    psi_init: torch.Tensor,
    rhs: torch.Tensor,
    kappa: torch.Tensor,
    eta: float,
    dx: float,
    dy: float,
    gs_iters: int,
    omega: float,
    boundary_mode: str = "dirichlet",
    kappa_y: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    psi = psi_init.clone()
    rows, cols = psi.shape
    if rows < 3 or cols < 3:
        return psi

    device = psi.device
    dx2 = max(dx * dx, 1e-12)
    dy2 = max(dy * dy, 1e-12)

    yy = torch.arange(rows - 2, device=device).unsqueeze(1)
    xx = torch.arange(cols - 2, device=device).unsqueeze(0)
    red_mask = ((yy + xx) % 2) == 0
    black_mask = ~red_mask

    for _ in range(max(int(gs_iters), 1)):
        for mask in (red_mask, black_mask):
            center = psi[1:-1, 1:-1]
            kx = kappa
            ky = kappa if kappa_y is None else kappa_y
            kx_c = kx[1:-1, 1:-1]
            ky_c = ky[1:-1, 1:-1]
            ke = 0.5 * (kx_c + kx[1:-1, 2:])
            kw = 0.5 * (kx_c + kx[1:-1, :-2])
            kn = 0.5 * (ky_c + ky[:-2, 1:-1])
            ks = 0.5 * (ky_c + ky[2:, 1:-1])

            coeff = (ke + kw) / dx2 + (kn + ks) / dy2 + float(eta)
            neighbors = (
                ke * psi[1:-1, 2:] / dx2
                + kw * psi[1:-1, :-2] / dx2
                + kn * psi[:-2, 1:-1] / dy2
                + ks * psi[2:, 1:-1] / dy2
            )
            update = (neighbors - rhs[1:-1, 1:-1]) / torch.clamp(coeff, min=1e-12)
            relaxed = (1.0 - float(omega)) * center + float(omega) * update
            psi[1:-1, 1:-1] = torch.where(mask, relaxed, center)

        _apply_boundary_condition(psi, boundary_mode)

    return psi


def solve_gs_plasma_pde(
    rhs: torch.Tensor,
    kappa_base: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    pde_cfg: Dict,
    psi_init: Optional[torch.Tensor] = None,
    kappa_y_base: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[float]]:
    """
    Solve GS-inspired nonlinear elliptic PDE with damped Picard + RB-GS.

    Returns:
        (psi, residual_history)
    """
    rows, cols = int(rhs.shape[0]), int(rhs.shape[1])
    device = rhs.device

    dx = float(canvas_width) / float(max(cols, 1))
    dy = float(canvas_height) / float(max(rows, 1))

    picard_outer = int(pde_cfg.get("picard_outer", 3))
    gs_iters = int(pde_cfg.get("gs_iters", 24))
    omega = float(pde_cfg.get("omega", 1.05))
    damping = float(pde_cfg.get("damping", 0.65))
    eta = float(pde_cfg.get("eta", 0.2))
    kappa_psi = float(pde_cfg.get("kappa_psi", 0.35))
    psi_scale = float(pde_cfg.get("psi_nonlinear_scale", 1.0))
    tol = float(pde_cfg.get("convergence_tol", 1e-4))
    boundary_mode = str(pde_cfg.get("boundary_mode", "dirichlet"))

    if psi_init is None:
        psi = torch.zeros_like(rhs, device=device)
    else:
        psi = psi_init.clone().to(device=device)
    _apply_boundary_condition(psi, boundary_mode)

    residual_history: List[float] = []

    for _ in range(max(picard_outer, 1)):
        prev = psi.clone()
        kappa_nl = kappa_psi * torch.sigmoid(psi * psi_scale)
        kappa = torch.clamp(
            kappa_base + kappa_nl,
            min=0.05,
            max=10.0,
        )
        ky = None
        if kappa_y_base is not None:
            ky = torch.clamp(
                kappa_y_base + kappa_nl,
                min=0.05,
                max=10.0,
            )
        linear_sol = _red_black_gauss_seidel(
            psi_init=prev,
            rhs=rhs,
            kappa=kappa,
            eta=eta,
            dx=dx,
            dy=dy,
            gs_iters=gs_iters,
            omega=omega,
            boundary_mode=boundary_mode,
            kappa_y=ky,
        )
        psi_new = (1.0 - damping) * prev + damping * linear_sol
        _apply_boundary_condition(psi_new, boundary_mode)

        res_new = float(_compute_residual(psi_new, rhs, kappa, eta, dx, dy, kappa_y=ky).item())
        if residual_history and res_new > residual_history[-1]:
            # Fallback: reduce update magnitude until residual does not increase.
            accepted = False
            blend = 0.5
            while blend >= 1.0 / 64.0:
                cand = prev + blend * (psi_new - prev)
                cand_res = float(_compute_residual(cand, rhs, kappa, eta, dx, dy, kappa_y=ky).item())
                if cand_res <= residual_history[-1]:
                    psi_new = cand
                    res_new = cand_res
                    accepted = True
                    break
                blend *= 0.5
            if not accepted:
                psi_new = prev
                res_new = residual_history[-1]

        psi = psi_new
        residual_history.append(res_new)
        if res_new <= tol:
            break

    return psi, residual_history


def gradient_from_potential(
    psi: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute dpsi/dx and dpsi/dy on grid cells."""
    rows, cols = int(psi.shape[0]), int(psi.shape[1])
    dx = float(canvas_width) / float(max(cols, 1))
    dy = float(canvas_height) / float(max(rows, 1))
    inv_2dx = 1.0 / max(2.0 * dx, 1e-12)
    inv_2dy = 1.0 / max(2.0 * dy, 1e-12)

    gx = torch.zeros_like(psi)
    gy = torch.zeros_like(psi)

    gx[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) * inv_2dx
    gx[:, 0] = (psi[:, 1] - psi[:, 0]) / max(dx, 1e-12)
    gx[:, -1] = (psi[:, -1] - psi[:, -2]) / max(dx, 1e-12)

    gy[1:-1, :] = (psi[2:, :] - psi[:-2, :]) * inv_2dy
    gy[0, :] = (psi[1, :] - psi[0, :]) / max(dy, 1e-12)
    gy[-1, :] = (psi[-1, :] - psi[-2, :]) / max(dy, 1e-12)

    return gx, gy


def bilinear_sample(
    field: torch.Tensor,
    points_xy: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
) -> torch.Tensor:
    """Bilinear sampling on a [rows, cols] field at N point coordinates."""
    rows, cols = int(field.shape[0]), int(field.shape[1])
    if points_xy.numel() == 0:
        return torch.zeros(0, dtype=field.dtype, device=field.device)

    cw = float(canvas_width) / float(max(cols, 1))
    ch = float(canvas_height) / float(max(rows, 1))

    cx = torch.clamp(points_xy[:, 0] / max(cw, 1e-12) - 0.5, 0.0, float(cols - 1))
    cy = torch.clamp(points_xy[:, 1] / max(ch, 1e-12) - 0.5, 0.0, float(rows - 1))

    x0 = torch.floor(cx).long()
    y0 = torch.floor(cy).long()
    x1 = torch.clamp(x0 + 1, max=cols - 1)
    y1 = torch.clamp(y0 + 1, max=rows - 1)

    tx = cx - x0.float()
    ty = cy - y0.float()

    v00 = field[y0, x0]
    v10 = field[y0, x1]
    v01 = field[y1, x0]
    v11 = field[y1, x1]

    return (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )


def sample_vector_field(
    fx_grid: torch.Tensor,
    fy_grid: torch.Tensor,
    points_xy: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
) -> torch.Tensor:
    fx = bilinear_sample(fx_grid, points_xy, canvas_width, canvas_height)
    fy = bilinear_sample(fy_grid, points_xy, canvas_width, canvas_height)
    return torch.stack([fx, fy], dim=1)


def compute_net_force(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    force = torch.zeros_like(positions)
    if edge_index.numel() == 0:
        return force

    eps = 1e-6
    for e in range(int(edge_index.shape[0])):
        i = int(edge_index[e, 0].item())
        j = int(edge_index[e, 1].item())
        w = float(edge_weight[e].item()) if edge_weight.numel() else 1.0

        pi = positions[i]
        pj = positions[j]
        if edge_offsets is not None and edge_offsets.numel() > 0:
            pi = pi + edge_offsets[e, 0].to(device=positions.device, dtype=positions.dtype)
            pj = pj + edge_offsets[e, 1].to(device=positions.device, dtype=positions.dtype)

        d = pj - pi
        dist = torch.sqrt(torch.clamp(torch.sum(d * d), min=eps))
        f = (float(w) * d) / dist
        force[i] += f
        force[j] -= f

    # Normalize for stable blending across benchmark scales.
    norm = torch.linalg.norm(force, dim=1)
    denom = torch.quantile(norm, 0.90) if force.shape[0] >= 10 else torch.max(norm)
    d = float(denom.item()) if norm.numel() > 0 else 0.0
    if d > 1e-6:
        force = force / d
    return force


def compute_overlap_repulsion(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    gap: float = 1e-4,
    size_inflation: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    n = int(positions.shape[0])
    if n <= 1:
        return torch.zeros_like(positions)

    effective_sizes = sizes
    if size_inflation is not None and size_inflation.numel() == n:
        inflation = torch.clamp(
            size_inflation.reshape(n, 1).to(device=positions.device, dtype=positions.dtype),
            min=1.0,
        )
        effective_sizes = sizes * inflation

    x = positions[:, 0]
    y = positions[:, 1]
    w = effective_sizes[:, 0]
    h = effective_sizes[:, 1]

    dx = x.unsqueeze(1) - x.unsqueeze(0)
    dy = y.unsqueeze(1) - y.unsqueeze(0)

    sep_x = 0.5 * (w.unsqueeze(1) + w.unsqueeze(0)) + float(gap)
    sep_y = 0.5 * (h.unsqueeze(1) + h.unsqueeze(0)) + float(gap)

    ovx = sep_x - torch.abs(dx)
    ovy = sep_y - torch.abs(dy)
    mask = (ovx > 0.0) & (ovy > 0.0)
    eye = torch.eye(n, dtype=torch.bool, device=positions.device)
    mask = mask & (~eye)

    push = torch.minimum(torch.relu(ovx), torch.relu(ovy))
    fx = torch.where(mask, torch.sign(dx) * push, torch.zeros_like(dx))
    fy = torch.where(mask, torch.sign(dy) * push, torch.zeros_like(dy))

    force = torch.stack([fx.sum(dim=1), fy.sum(dim=1)], dim=1)

    norm = torch.linalg.norm(force, dim=1)
    denom = torch.quantile(norm, 0.90) if n >= 10 else torch.max(norm)
    d = float(denom.item()) if norm.numel() > 0 else 0.0
    if d > 1e-6:
        force = force / d
    return force


def _jax_backend_enabled(
    pde_cfg: Dict,
    device: torch.device,
) -> bool:
    if device.type != "cpu":
        return False
    if str(pde_cfg.get("kernel_backend", "torch")).lower() != "jax":
        return False
    return bool(jax_accel is not None and jax_accel.is_available())


def _torch_from_numpy_copy(
    arr: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    return torch.from_numpy(np.array(arr, copy=True)).to(device=device)


def build_ballooning_force(
    positions: torch.Tensor,
    q_field: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    quantile: float = 0.92,
    topk: int = 12,
    sigma_frac: float = 0.18,
    power: float = 1.25,
) -> torch.Tensor:
    """
    Repel macros from the hottest congestion pockets.

    This complements the local q-gradient with a longer-range expulsion force,
    which helps macros escape broad, flat congestion wells.
    """
    if positions.numel() == 0 or q_field.numel() == 0:
        return torch.zeros_like(positions)

    flat_q = q_field.reshape(-1).float()
    q_thresh = torch.quantile(flat_q, float(quantile))
    excess = torch.clamp(flat_q - q_thresh, min=0.0)
    mask = excess > 1e-9
    if not bool(mask.any()):
        return torch.zeros_like(positions)

    weights = excess[mask].pow(float(power))
    rows = int(q_field.shape[0])
    cols = int(q_field.shape[1]) if q_field.ndim >= 2 else rows
    centers = _grid_centers(canvas_width, canvas_height, rows, cols, positions.device)[mask]

    topk_eff = max(1, int(topk))
    if weights.numel() > topk_eff:
        keep = torch.topk(weights, topk_eff).indices
        weights = weights[keep]
        centers = centers[keep]

    dx = positions[:, 0:1] - centers[:, 0].unsqueeze(0)
    dy = positions[:, 1:2] - centers[:, 1].unsqueeze(0)
    dist2 = dx * dx + dy * dy

    sigma = max(float(sigma_frac) * max(float(canvas_width), float(canvas_height)), 1e-3)
    kernel = torch.exp(-0.5 * dist2 / (sigma * sigma))
    inv_r = torch.rsqrt(torch.clamp(dist2, min=(0.05 * sigma) ** 2))
    weighted = weights.unsqueeze(0) * kernel * inv_r

    fx = (weighted * dx).sum(dim=1)
    fy = (weighted * dy).sum(dim=1)
    return _normalize_vector_force(torch.stack([fx, fy], dim=1))


def compute_plasma_forces(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    grid_size: int,
    pde_cfg: Dict,
    rhs_weights: Dict[str, float],
    background_positions: Optional[torch.Tensor] = None,
    background_sizes: Optional[torch.Tensor] = None,
    anchor_targets: Optional[torch.Tensor] = None,
    anchor_weights: Optional[torch.Tensor] = None,
    edge_offsets: Optional[torch.Tensor] = None,
    previous_psi: Optional[torch.Tensor] = None,
    bundle_terms: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    pin_terms: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    pin_edge_profile: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """
    Build plasma fields, solve PDE, and return sampled plasma/net/repulsion forces.
    """
    n = int(positions.shape[0])
    device = positions.device
    rows = max(8, int(grid_size))
    cols = max(8, int(grid_size))

    beta = float(pde_cfg.get("rhs_softplus_beta", 8.0))
    density_target = float(pde_cfg.get("density_target", 0.90))
    overflow_power = float(pde_cfg.get("overflow_power", 1.2))
    flux_tube_enabled = bool(pde_cfg.get("pin_flux_tubes", True))
    edge_offsets = edge_offsets if flux_tube_enabled else None
    bundle_enabled = bool(pde_cfg.get("bundle_q_enabled", 0.0))
    pin_pressure_enabled = bool(pde_cfg.get("pin_pressure_enabled", 0.0))
    channel_kappa_cfg = dict(pde_cfg.get("channel_kappa", {}))
    channel_kappa_enabled = bool(channel_kappa_cfg.get("enabled", False))
    has_pin_terms = (
        pin_terms is not None
        and int(pin_terms[0].numel()) > 0
        and int(pin_terms[1].numel()) > 0
        and int(pin_terms[2].numel()) > 0
    )
    use_jax = (
        _jax_backend_enabled(pde_cfg, device)
        and edge_offsets is None
        and not bundle_enabled
        and not (pin_pressure_enabled and has_pin_terms)
        and not channel_kappa_enabled
    )

    pos_np = None
    size_np = None
    edge_np = None
    weight_np = None
    if use_jax:
        pos_np = np.asarray(positions.detach().cpu(), dtype=np.float32)
        size_np = np.asarray(sizes.detach().cpu(), dtype=np.float32)
        edge_np = np.asarray(edge_index.detach().cpu(), dtype=np.int32)
        weight_np = (
            np.asarray(edge_weight.detach().cpu(), dtype=np.float32)
            if edge_weight.numel()
            else np.ones((int(edge_index.shape[0]),), dtype=np.float32)
        )

    if use_jax:
        rho = _torch_from_numpy_copy(
            jax_accel.build_density_pressure(
                pos_np,
                size_np,
                canvas_width,
                canvas_height,
                rows,
                cols,
                target_density=density_target,
                overflow_power=overflow_power,
            ),
            device=device,
        )
    else:
        rho = build_density_pressure(
            positions,
            sizes,
            canvas_width,
            canvas_height,
            rows,
            cols,
            target_density=density_target,
            overflow_power=overflow_power,
        )
    hard_density = deposit_macro_density(
        positions,
        sizes,
        canvas_width,
        canvas_height,
        rows,
        cols,
    )
    bg_density = torch.zeros_like(rho)
    if background_positions is not None and background_sizes is not None and background_positions.numel() > 0:
        bg_mode = str(pde_cfg.get("background_deposit", "point"))
        if bg_mode == "overlap":
            if use_jax:
                bg_density = _torch_from_numpy_copy(
                    jax_accel.deposit_macro_density(
                        np.asarray(background_positions.detach().cpu(), dtype=np.float32),
                        np.asarray(background_sizes.detach().cpu(), dtype=np.float32),
                        canvas_width,
                        canvas_height,
                        rows,
                        cols,
                    ),
                    device=device,
                )
            else:
                bg_density = deposit_macro_density(
                    background_positions,
                    background_sizes,
                    canvas_width,
                    canvas_height,
                    rows,
                    cols,
                )
        else:
            cell_area = max((float(canvas_width) / max(cols, 1)) * (float(canvas_height) / max(rows, 1)), 1e-12)
            bg_weights = (background_sizes[:, 0] * background_sizes[:, 1]) / cell_area
            bg_density = _deposit_points_bilinear(
                background_positions,
                bg_weights.to(dtype=torch.float32),
                canvas_width,
                canvas_height,
                rows,
                cols,
            )
    if use_jax:
        q = _torch_from_numpy_copy(
            jax_accel.build_rudy_pressure(
                pos_np,
                edge_np,
                weight_np,
                canvas_width,
                canvas_height,
                rows,
                cols,
                smoothing_passes=int(pde_cfg.get("rudy_smoothing", 1)),
            ),
            device=device,
        )
        n_src = _torch_from_numpy_copy(
            jax_accel.build_net_source(
                pos_np,
                edge_np,
                weight_np,
                canvas_width,
                canvas_height,
                rows,
                cols,
            ),
            device=device,
        )
    else:
        q = build_rudy_pressure(
            positions,
            edge_index,
            edge_weight,
            canvas_width,
            canvas_height,
            rows,
            cols,
            edge_offsets=edge_offsets,
            smoothing_passes=int(pde_cfg.get("rudy_smoothing", 1)),
        )
        n_src = build_net_source(
            positions,
            edge_index,
            edge_weight,
            canvas_width,
            canvas_height,
            rows,
            cols,
            edge_offsets=edge_offsets,
        )
    q_bundle = torch.zeros_like(q)
    n_bundle = torch.zeros_like(n_src)
    if bundle_enabled and bundle_terms is not None:
        bundle_ptr, bundle_index, bundle_offsets, bundle_weight = bundle_terms
        if bundle_weight.numel() > 0:
            q_bundle = build_bundle_rudy_pressure(
                positions,
                bundle_ptr.to(device=device),
                bundle_index.to(device=device),
                bundle_offsets.to(device=device),
                bundle_weight.to(device=device),
                canvas_width,
                canvas_height,
                rows,
                cols,
                smoothing_passes=int(pde_cfg.get("bundle_q_smoothing", pde_cfg.get("rudy_smoothing", 1))),
            )
            q = q + float(pde_cfg.get("bundle_q_scale", 0.25)) * q_bundle
            if bool(pde_cfg.get("bundle_n_enabled", 0.0)):
                n_bundle = build_bundle_net_source(
                    positions,
                    bundle_ptr.to(device=device),
                    bundle_index.to(device=device),
                    bundle_offsets.to(device=device),
                    bundle_weight.to(device=device),
                    canvas_width,
                    canvas_height,
                    rows,
                    cols,
                    degree_boost=float(pde_cfg.get("bundle_n_degree_boost", 0.0)),
                )
                bundle_n_mix = max(0.0, min(1.0, float(pde_cfg.get("bundle_n_mix", 0.0))))
                bundle_n_scale = float(pde_cfg.get("bundle_n_scale", 0.20))
                if bundle_n_mix > 0.0:
                    n_src = (1.0 - bundle_n_mix) * n_src + bundle_n_mix * n_bundle
                else:
                    n_src = n_src + bundle_n_scale * n_bundle
    pin_pressure = torch.zeros_like(rho)
    pin_over = torch.zeros_like(rho)
    if pin_pressure_enabled and has_pin_terms:
        pin_index, pin_offsets, pin_weight = pin_terms
        pin_pressure = build_pin_density_pressure(
            positions,
            pin_index.to(device=device),
            pin_offsets.to(device=device),
            pin_weight.to(device=device),
            canvas_width,
            canvas_height,
            rows,
            cols,
            smoothing_passes=int(pde_cfg.get("pin_smoothing", 1)),
            norm_mode=str(pde_cfg.get("pin_norm_mode", "mean")),
        )
        pin_beta = float(pde_cfg.get("pin_softplus_beta", beta))
        pin_target = float(pde_cfg.get("pin_target", 1.0))
        pin_over = F.softplus(pin_beta * (pin_pressure - pin_target)) / max(pin_beta, 1e-12)
    porosity_cfg = dict(pde_cfg.get("porosity", {}))
    porosity_over = build_porosity_pressure(
        hard_density,
        bg_density,
        porosity_cfg,
    )
    anchor_q_weight = float(pde_cfg.get("anchor_q_weight", 0.0))
    anchor_n_weight = float(pde_cfg.get("anchor_n_weight", 0.0))
    if (
        anchor_targets is not None
        and anchor_weights is not None
        and anchor_targets.numel() > 0
        and anchor_weights.numel() > 0
    ):
        if anchor_q_weight > 0.0:
            if use_jax:
                anchor_q_np = jax_accel.build_anchor_pressure(
                    pos_np,
                    np.asarray(anchor_targets.detach().cpu(), dtype=np.float32),
                    np.asarray(anchor_weights.detach().cpu(), dtype=np.float32),
                    canvas_width,
                    canvas_height,
                    rows,
                    cols,
                    smoothing_passes=int(pde_cfg.get("anchor_rudy_smoothing", pde_cfg.get("rudy_smoothing", 1))),
                )
                anchor_q = (
                    _torch_from_numpy_copy(anchor_q_np, device=device)
                    if anchor_q_np is not None
                    else torch.zeros_like(q)
                )
            else:
                anchor_q = build_anchor_pressure(
                    positions,
                    anchor_targets,
                    anchor_weights,
                    canvas_width,
                    canvas_height,
                    rows,
                    cols,
                    smoothing_passes=int(pde_cfg.get("anchor_rudy_smoothing", pde_cfg.get("rudy_smoothing", 1))),
                )
            q = q + anchor_q_weight * anchor_q
        if anchor_n_weight > 0.0:
            if use_jax:
                anchor_src_np = jax_accel.build_anchor_source(
                    pos_np,
                    np.asarray(anchor_targets.detach().cpu(), dtype=np.float32),
                    np.asarray(anchor_weights.detach().cpu(), dtype=np.float32),
                    canvas_width,
                    canvas_height,
                    rows,
                    cols,
                )
                anchor_src = (
                    _torch_from_numpy_copy(anchor_src_np, device=device)
                    if anchor_src_np is not None
                    else torch.zeros_like(n_src)
                )
            else:
                anchor_src = build_anchor_source(
                    positions,
                    anchor_targets,
                    anchor_weights,
                    canvas_width,
                    canvas_height,
                    rows,
                    cols,
                )
            n_src = n_src + anchor_n_weight * anchor_src
    wall = build_wall_field(
        canvas_width,
        canvas_height,
        rows,
        cols,
        decay_frac=float(pde_cfg.get("wall_decay_frac", 0.12)),
        device=device,
    )

    w_rho = float(rhs_weights.get("rho", 1.0))
    w_q = float(rhs_weights.get("q", 0.5))
    w_n = float(rhs_weights.get("n", 0.2))
    w_wall = float(rhs_weights.get("wall", 0.7))
    w_bg = float(rhs_weights.get("bg", 0.0))
    w_pin = float(rhs_weights.get("pin", 0.0))
    w_porosity = float(rhs_weights.get("porosity", 0.0))

    q_over = F.softplus(beta * (q - 1.0)) / max(beta, 1e-12)
    # Optional congestion focusing: boost highest-demand regions to
    # encourage stronger dispersal on congested benchmarks.
    q_focus_q = float(pde_cfg.get("q_focus_quantile", 0.0))
    q_focus_alpha = float(pde_cfg.get("q_focus_alpha", 0.0))
    if q_focus_alpha > 0.0 and q_focus_q > 0.0 and q_over.numel() > 0:
        q_thresh = torch.quantile(q_over, q_focus_q)
        scale = max(float(q_thresh.item()) * 0.25, 1e-6)
        focus = torch.sigmoid((q_over - q_thresh) / scale)
        q_over = q_over * (1.0 + q_focus_alpha * focus)
    q_dyn_alpha = float(pde_cfg.get("q_dynamic_alpha", 0.0))
    if q_dyn_alpha > 0.0 and q_over.numel() > 0:
        q_peak = torch.quantile(q_over, 0.90)
        boost = torch.clamp(q_peak - 1.0, min=0.0, max=2.0)
        q_over = q_over * (1.0 + q_dyn_alpha * boost)
    channel_barrier_x = torch.zeros_like(rho)
    channel_barrier_y = torch.zeros_like(rho)
    if channel_kappa_enabled:
        # Build channel barriers before source quench so the source model can
        # suppress attraction into narrow corridors, not just the kappa field.
        channel_barrier_x, channel_barrier_y = build_channel_permeability_maps(
            positions,
            sizes,
            pin_edge_profile,
            channel_kappa_cfg,
            canvas_width,
            canvas_height,
            rows,
            cols,
        )
    source_quench_cfg = dict(pde_cfg.get("source_quench", {}))
    if bool(source_quench_cfg.get("enabled", False)):
        n_src = apply_source_quench(
            n_src,
            q_over,
            bg_density,
            pin_over if (pin_pressure_enabled and has_pin_terms) else None,
            porosity_over if bool(porosity_cfg.get("enabled", False)) else None,
            channel_barrier_x if channel_kappa_enabled else None,
            channel_barrier_y if channel_kappa_enabled else None,
            source_quench_cfg,
        )
    rhs = (
        w_rho * rho
        + w_q * q_over
        + w_n * n_src
        + w_bg * bg_density
        + w_pin * pin_over
        + w_porosity * porosity_over
        - w_wall * wall
    )

    kappa_base = torch.clamp(
        float(pde_cfg.get("kappa_base", 1.0))
        + float(pde_cfg.get("kappa_rho", 1.2)) * rho
        + float(pde_cfg.get("kappa_q", 0.7)) * q_over,
        min=0.05,
        max=10.0,
    )
    if pin_pressure_enabled and has_pin_terms:
        kappa_base = torch.clamp(
            kappa_base + float(pde_cfg.get("kappa_pin", 0.0)) * pin_over,
            min=0.05,
            max=10.0,
        )
    if bool(porosity_cfg.get("enabled", False)):
        kappa_base = torch.clamp(
            kappa_base + float(pde_cfg.get("kappa_porosity", 0.0)) * porosity_over,
            min=0.05,
            max=10.0,
        )
    kappa_x_base = kappa_base
    kappa_y_base = None
    if channel_kappa_enabled:
        suppress_alpha = max(float(channel_kappa_cfg.get("cross_suppress_alpha", 0.0)), 0.0)
        tangential_alpha = max(float(channel_kappa_cfg.get("tangential_boost_alpha", 0.0)), 0.0)
        floor_scale = max(float(channel_kappa_cfg.get("min_scale", 0.25)), 1e-3)
        ceil_scale = max(float(channel_kappa_cfg.get("max_scale", 4.0)), floor_scale)
        kappa_x_scale = (1.0 + tangential_alpha * channel_barrier_y) / torch.clamp(
            1.0 + suppress_alpha * channel_barrier_x,
            min=1e-6,
        )
        kappa_y_scale = (1.0 + tangential_alpha * channel_barrier_x) / torch.clamp(
            1.0 + suppress_alpha * channel_barrier_y,
            min=1e-6,
        )
        kappa_x_base = torch.clamp(kappa_base * kappa_x_scale, min=0.05 * floor_scale, max=10.0 * ceil_scale)
        kappa_y_base = torch.clamp(kappa_base * kappa_y_scale, min=0.05 * floor_scale, max=10.0 * ceil_scale)

    psi, residual_history = solve_gs_plasma_pde(
        rhs=rhs,
        kappa_base=kappa_x_base,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        pde_cfg=pde_cfg,
        psi_init=previous_psi,
        kappa_y_base=kappa_y_base,
    )

    gx, gy = gradient_from_potential(psi, canvas_width, canvas_height)
    plasma_force = sample_vector_field(-gx, -gy, positions, canvas_width, canvas_height)
    plasma_force = _normalize_vector_force(plasma_force)
    # Hall-like drift term: perpendicular to grad(psi), can help escape
    # congestion wells without collapsing clusters.
    hall_force = sample_vector_field(-gy, gx, positions, canvas_width, canvas_height)
    hall_force = _normalize_vector_force(hall_force)

    # Dual-temperature plasma: hot/cold variants of psi.
    hot_force = torch.zeros_like(plasma_force)
    cold_force = torch.zeros_like(plasma_force)
    if float(pde_cfg.get("dual_enabled", 0.0)) > 0.0:
        hot_cfg = dict(pde_cfg)
        hot_cfg["eta"] = float(pde_cfg.get("hot_eta", pde_cfg.get("eta", 0.22)))
        hot_cfg["kappa_q"] = float(pde_cfg.get("hot_kappa_q", pde_cfg.get("kappa_q", 0.7)))
        hot_cfg["kappa_rho"] = float(pde_cfg.get("hot_kappa_rho", pde_cfg.get("kappa_rho", 1.2)))

        cold_cfg = dict(pde_cfg)
        cold_cfg["eta"] = float(pde_cfg.get("cold_eta", pde_cfg.get("eta", 0.22)))
        cold_cfg["kappa_q"] = float(pde_cfg.get("cold_kappa_q", pde_cfg.get("kappa_q", 0.7)))
        cold_cfg["kappa_rho"] = float(pde_cfg.get("cold_kappa_rho", pde_cfg.get("kappa_rho", 1.2)))

        hot_psi, _ = solve_gs_plasma_pde(
            rhs=rhs,
            kappa_base=kappa_x_base,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            pde_cfg=hot_cfg,
            psi_init=None,
            kappa_y_base=kappa_y_base,
        )
        cold_psi, _ = solve_gs_plasma_pde(
            rhs=rhs,
            kappa_base=kappa_x_base,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            pde_cfg=cold_cfg,
            psi_init=None,
            kappa_y_base=kappa_y_base,
        )
        hot_gx, hot_gy = gradient_from_potential(hot_psi, canvas_width, canvas_height)
        cold_gx, cold_gy = gradient_from_potential(cold_psi, canvas_width, canvas_height)
        hot_force = sample_vector_field(-hot_gx, -hot_gy, positions, canvas_width, canvas_height)
        cold_force = sample_vector_field(-cold_gx, -cold_gy, positions, canvas_width, canvas_height)
        hot_force = _normalize_vector_force(hot_force)
        cold_force = _normalize_vector_force(cold_force)

    # Secondary congestion-only potential chi (optional, for unique plasma behavior).
    chi_force = torch.zeros_like(plasma_force)
    if float(pde_cfg.get("chi_enabled", 0.0)) > 0.0:
        chi_rhs = float(pde_cfg.get("chi_q_weight", 1.0)) * q_over
        chi_kappa = torch.clamp(
            float(pde_cfg.get("chi_kappa_base", 1.0))
            + float(pde_cfg.get("chi_kappa_q", 0.7)) * q_over,
            min=0.05,
            max=10.0,
        )
        chi, _ = solve_gs_plasma_pde(
            rhs=chi_rhs,
            kappa_base=chi_kappa,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            pde_cfg=pde_cfg,
            psi_init=None,
            kappa_y_base=chi_kappa if kappa_y_base is None else torch.clamp(
                chi_kappa * torch.clamp(kappa_y_base / torch.clamp(kappa_x_base, min=1e-6), min=0.2, max=5.0),
                min=0.05,
                max=10.0 * max(float(channel_kappa_cfg.get("max_scale", 4.0)), 1.0),
            ),
        )
        chi_gx, chi_gy = gradient_from_potential(chi, canvas_width, canvas_height)
        chi_force = sample_vector_field(-chi_gx, -chi_gy, positions, canvas_width, canvas_height)
        chi_force = _normalize_vector_force(chi_force)

    balloon_force = torch.zeros_like(plasma_force)
    if float(pde_cfg.get("balloon_enabled", 0.0)) > 0.0:
        balloon_force = build_ballooning_force(
            positions,
            q_over,
            canvas_width,
            canvas_height,
            quantile=float(pde_cfg.get("balloon_quantile", 0.92)),
            topk=int(pde_cfg.get("balloon_topk", 12)),
            sigma_frac=float(pde_cfg.get("balloon_sigma_frac", 0.18)),
            power=float(pde_cfg.get("balloon_power", 1.25)),
        )

    # Explicit pressure-gradient channels (plasma-inspired):
    # push macros away from high density/congestion "pressure" regions.
    rho_gx, rho_gy = gradient_from_potential(rho, canvas_width, canvas_height)
    q_over_gx, q_over_gy = gradient_from_potential(q_over, canvas_width, canvas_height)
    bg_gx, bg_gy = gradient_from_potential(bg_density, canvas_width, canvas_height)
    pin_gx, pin_gy = gradient_from_potential(pin_over, canvas_width, canvas_height)
    porosity_gx, porosity_gy = gradient_from_potential(porosity_over, canvas_width, canvas_height)
    rho_force = sample_vector_field(-rho_gx, -rho_gy, positions, canvas_width, canvas_height)
    q_force = sample_vector_field(-q_over_gx, -q_over_gy, positions, canvas_width, canvas_height)
    bg_force = sample_vector_field(-bg_gx, -bg_gy, positions, canvas_width, canvas_height)
    pin_force = sample_vector_field(-pin_gx, -pin_gy, positions, canvas_width, canvas_height)
    porosity_force = sample_vector_field(-porosity_gx, -porosity_gy, positions, canvas_width, canvas_height)
    rho_force = _normalize_vector_force(rho_force)
    q_force = _normalize_vector_force(q_force)
    bg_force = _normalize_vector_force(bg_force)
    pin_force = _normalize_vector_force(pin_force)
    porosity_force = _normalize_vector_force(porosity_force)

    q_samples = bilinear_sample(q_over, positions, canvas_width, canvas_height)
    rho_samples = bilinear_sample(rho, positions, canvas_width, canvas_height)
    bg_samples = bilinear_sample(bg_density, positions, canvas_width, canvas_height)
    pin_samples = bilinear_sample(pin_over, positions, canvas_width, canvas_height)
    porosity_samples = bilinear_sample(porosity_over, positions, canvas_width, canvas_height)

    q_gain = _hotspot_gain(
        q_samples,
        alpha=float(pde_cfg.get("beta_q_alpha", 0.0)),
        quantile=float(pde_cfg.get("beta_q_quantile", 0.80)),
        power=float(pde_cfg.get("beta_q_power", 1.0)),
        cap=float(pde_cfg.get("beta_q_cap", 3.0)),
    )
    rho_gain = _hotspot_gain(
        rho_samples,
        alpha=float(pde_cfg.get("beta_rho_alpha", 0.0)),
        quantile=float(pde_cfg.get("beta_rho_quantile", 0.85)),
        power=float(pde_cfg.get("beta_rho_power", 1.0)),
        cap=float(pde_cfg.get("beta_rho_cap", 2.5)),
    )
    bg_gain = _hotspot_gain(
        bg_samples,
        alpha=float(pde_cfg.get("beta_bg_alpha", 0.0)),
        quantile=float(pde_cfg.get("beta_bg_quantile", 0.85)),
        power=float(pde_cfg.get("beta_bg_power", 1.0)),
        cap=float(pde_cfg.get("beta_bg_cap", 2.5)),
    )
    pin_gain = _hotspot_gain(
        pin_samples,
        alpha=float(pde_cfg.get("beta_pin_alpha", 0.0)),
        quantile=float(pde_cfg.get("beta_pin_quantile", 0.85)),
        power=float(pde_cfg.get("beta_pin_power", 1.0)),
        cap=float(pde_cfg.get("beta_pin_cap", 2.5)),
    )
    porosity_gain = _hotspot_gain(
        porosity_samples,
        alpha=float(pde_cfg.get("beta_porosity_alpha", 0.0)),
        quantile=float(pde_cfg.get("beta_porosity_quantile", 0.85)),
        power=float(pde_cfg.get("beta_porosity_power", 1.0)),
        cap=float(pde_cfg.get("beta_porosity_cap", 2.5)),
    )

    plasma_force = plasma_force * (1.0 + float(pde_cfg.get("beta_plasma_alpha", 0.0)) * (q_gain - 1.0))
    hall_force = hall_force * (1.0 + float(pde_cfg.get("beta_hall_alpha", 0.0)) * (q_gain - 1.0))
    balloon_force = balloon_force * (1.0 + float(pde_cfg.get("beta_balloon_alpha", 0.0)) * (q_gain - 1.0))
    chi_force = chi_force * (1.0 + float(pde_cfg.get("beta_chi_alpha", 0.0)) * (q_gain - 1.0))
    q_force = q_force * q_gain
    rho_force = rho_force * rho_gain
    bg_force = bg_force * bg_gain
    pin_force = pin_force * pin_gain
    porosity_force = porosity_force * porosity_gain

    dia_force = torch.zeros_like(plasma_force)
    if float(pde_cfg.get("dia_enabled", 0.0)) > 0.0:
        dia_force = sample_vector_field(-q_over_gy, q_over_gx, positions, canvas_width, canvas_height)
        dia_force = _normalize_vector_force(dia_force)
        dia_force = dia_force * (1.0 + float(pde_cfg.get("beta_dia_alpha", 1.0)) * (q_gain - 1.0))

    repulsion_inflation = torch.ones((n, 1), dtype=torch.float32, device=device)
    rep_q_alpha = float(pde_cfg.get("beta_repulsion_q_alpha", 0.0))
    rep_rho_alpha = float(pde_cfg.get("beta_repulsion_rho_alpha", 0.0))
    rep_porosity_alpha = float(pde_cfg.get("beta_repulsion_porosity_alpha", 0.0))
    if rep_q_alpha > 0.0 and q_samples.numel() > 0:
        rep_q_gain = _hotspot_gain(
            q_samples,
            alpha=1.0,
            quantile=float(pde_cfg.get("beta_repulsion_q_quantile", 0.80)),
            power=float(pde_cfg.get("beta_repulsion_q_power", 1.0)),
            cap=float(pde_cfg.get("beta_repulsion_cap", 2.0)),
        )
        repulsion_inflation = repulsion_inflation + rep_q_alpha * (rep_q_gain - 1.0)
    if rep_rho_alpha > 0.0 and rho_samples.numel() > 0:
        rep_rho_gain = _hotspot_gain(
            rho_samples,
            alpha=1.0,
            quantile=float(pde_cfg.get("beta_repulsion_rho_quantile", 0.85)),
            power=float(pde_cfg.get("beta_repulsion_rho_power", 1.0)),
            cap=float(pde_cfg.get("beta_repulsion_cap", 2.0)),
        )
        repulsion_inflation = repulsion_inflation + rep_rho_alpha * (rep_rho_gain - 1.0)
    if rep_porosity_alpha > 0.0 and porosity_samples.numel() > 0:
        rep_porosity_gain = _hotspot_gain(
            porosity_samples,
            alpha=1.0,
            quantile=float(pde_cfg.get("beta_repulsion_porosity_quantile", 0.85)),
            power=float(pde_cfg.get("beta_repulsion_porosity_power", 1.0)),
            cap=float(pde_cfg.get("beta_repulsion_cap", 2.0)),
        )
        repulsion_inflation = repulsion_inflation + rep_porosity_alpha * (rep_porosity_gain - 1.0)
    repulsion_cap = max(float(pde_cfg.get("beta_repulsion_cap", 2.0)), 1.0)
    repulsion_inflation = torch.clamp(repulsion_inflation, min=1.0, max=repulsion_cap)

    if use_jax:
        net_force = _torch_from_numpy_copy(
            jax_accel.compute_net_force(
                pos_np,
                edge_np,
                weight_np,
            ),
            device=device,
        )
    else:
        net_force = compute_net_force(positions, edge_index, edge_weight, edge_offsets=edge_offsets)
    rep_force = compute_overlap_repulsion(
        positions,
        sizes,
        gap=1e-4,
        size_inflation=repulsion_inflation.squeeze(1),
    )

    return {
        "plasma_force": plasma_force,
        "hall_force": hall_force,
        "hot_force": hot_force,
        "cold_force": cold_force,
        "chi_force": chi_force,
        "dia_force": dia_force,
        "balloon_force": balloon_force,
        "rho_force": rho_force,
        "q_force": q_force,
        "bg_force": bg_force,
        "pin_force": pin_force,
        "porosity_force": porosity_force,
        "net_force": net_force,
        "repulsion_force": rep_force,
        "psi": psi,
        "rho": rho,
        "hard_density": hard_density,
        "bg_density": bg_density,
        "pin_pressure": pin_pressure,
        "pin_over": pin_over,
        "porosity_over": porosity_over,
        "q": q,
        "q_bundle": q_bundle,
        "n_bundle": n_bundle,
        "q_over": q_over,
        "channel_barrier_x": channel_barrier_x,
        "channel_barrier_y": channel_barrier_y,
        "q_samples": q_samples,
        "rho_samples": rho_samples,
        "pin_samples": pin_samples,
        "porosity_samples": porosity_samples,
        "repulsion_inflation": repulsion_inflation.squeeze(1),
        "n_src": n_src,
        "wall": wall,
        "residual_history": residual_history,
    }
