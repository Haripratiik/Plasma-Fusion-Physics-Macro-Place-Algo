"""R1 plasma-startup initialization.

This module builds a deterministic graph-spectral initial placement from the
hard-macro netlist. In the plasma analogy, the netlist graph is the externally
driven current ramp; the first two non-trivial Laplacian modes form the
startup flux coordinates before the GS/Taylor/tearing pipeline refines them.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def _scale_to_canvas(
    xy: torch.Tensor,
    sizes: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    margin_frac: float,
) -> torch.Tensor:
    if xy.numel() == 0:
        return xy
    out = xy.clone().float()
    mins = out.min(dim=0).values
    maxs = out.max(dim=0).values
    span = torch.clamp(maxs - mins, min=1e-9)
    out = (out - mins) / span
    max_w = float(sizes[:, 0].max().item()) if sizes.numel() else 0.0
    max_h = float(sizes[:, 1].max().item()) if sizes.numel() else 0.0
    margin_x = max(float(canvas_width) * float(margin_frac), 0.5 * max_w)
    margin_y = max(float(canvas_height) * float(margin_frac), 0.5 * max_h)
    usable_w = max(float(canvas_width) - 2.0 * margin_x, 1e-6)
    usable_h = max(float(canvas_height) - 2.0 * margin_y, 1e-6)
    out[:, 0] = margin_x + out[:, 0] * usable_w
    out[:, 1] = margin_y + out[:, 1] * usable_h
    return out


def _fallback_ring(num: int, canvas_width: float, canvas_height: float, device: torch.device) -> torch.Tensor:
    theta = torch.linspace(0.0, 2.0 * math.pi, steps=num + 1, device=device, dtype=torch.float32)[:-1]
    rx = 0.35 * float(canvas_width)
    ry = 0.35 * float(canvas_height)
    cx = 0.5 * float(canvas_width)
    cy = 0.5 * float(canvas_height)
    return torch.stack([cx + rx * torch.cos(theta), cy + ry * torch.sin(theta)], dim=1)


def _flux_band_to_canvas(
    theta: torch.Tensor,
    current: torch.Tensor,
    sizes: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    margin_frac: float,
) -> torch.Tensor:
    """Map graph-current ranks onto nested startup flux surfaces.

    High-current macros start on inner flux surfaces, while weakly connected
    macros live closer to the limiter. The construction is deterministic and
    deliberately basin-changing: it replaces a TILOS warm start with a
    tokamak-startup-style current ramp.
    """
    n = int(theta.numel())
    if n == 0:
        return torch.zeros(0, 2, dtype=sizes.dtype, device=sizes.device)
    device = theta.device
    order = torch.argsort(current, descending=True)
    rank = torch.zeros(n, dtype=torch.float32, device=device)
    if n > 1:
        rank[order] = torch.linspace(0.0, 1.0, steps=n, device=device)
    else:
        rank[order] = 0.0

    # Inner radius avoids collapse of high-degree/current macros at the axis.
    # sqrt(rank) gives roughly area-uniform occupancy across flux annuli.
    radius = 0.20 + 0.74 * torch.sqrt(torch.clamp(rank, 0.0, 1.0))
    # A tiny golden-angle shear removes repeated-angle degeneracy in symmetric
    # graph spectra without making the startup stochastic.
    theta = theta + rank * (math.pi * (3.0 - math.sqrt(5.0)))

    max_w = float(sizes[:, 0].max().item()) if sizes.numel() else 0.0
    max_h = float(sizes[:, 1].max().item()) if sizes.numel() else 0.0
    margin_x = max(float(canvas_width) * float(margin_frac), 0.5 * max_w)
    margin_y = max(float(canvas_height) * float(margin_frac), 0.5 * max_h)
    cx = 0.5 * float(canvas_width)
    cy = 0.5 * float(canvas_height)
    rx = max(0.5 * float(canvas_width) - margin_x, 1e-6)
    ry = max(0.5 * float(canvas_height) - margin_y, 1e-6)
    return torch.stack(
        [cx + radius * rx * torch.cos(theta), cy + radius * ry * torch.sin(theta)],
        dim=1,
    )


def _flux_lattice_startup(
    theta: torch.Tensor,
    current: torch.Tensor,
    sizes: torch.Tensor,
    fixed_mask: torch.Tensor,
    base_positions: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    margin_frac: float,
    n_bands: int,
    weber_sweeps: int,
) -> torch.Tensor:
    """Construct an overlap-aware nested-flux startup placement.

    This is intentionally not a TILOS warm start. It is a plasma startup fill:
    graph-current rank selects the target flux surface, graph eigenmode angle
    selects poloidal phase, and each macro is assigned to the closest available
    lattice point on nested flux contours.
    """
    n = int(theta.numel())
    device = theta.device
    if n == 0:
        return torch.zeros(0, 2, dtype=sizes.dtype, device=device)

    max_w = float(sizes[:, 0].max().item()) if sizes.numel() else 0.0
    max_h = float(sizes[:, 1].max().item()) if sizes.numel() else 0.0
    margin_x = max(float(canvas_width) * float(margin_frac), 0.5 * max_w)
    margin_y = max(float(canvas_height) * float(margin_frac), 0.5 * max_h)
    cx = 0.5 * float(canvas_width)
    cy = 0.5 * float(canvas_height)
    rx = max(0.5 * float(canvas_width) - margin_x, 1e-6)
    ry = max(0.5 * float(canvas_height) - margin_y, 1e-6)

    order = torch.argsort(current + 0.02 * sizes[:, 0] * sizes[:, 1], descending=True)
    rank = torch.zeros(n, dtype=torch.float32, device=device)
    if n > 1:
        rank[order] = torch.linspace(0.0, 1.0, steps=n, device=device)
    target_r = 0.24 + 0.70 * torch.sqrt(torch.clamp(rank, 0.0, 1.0))
    target_theta = theta + rank * (math.pi * (3.0 - math.sqrt(5.0)))

    bands = max(3, int(n_bands), int(math.ceil(math.sqrt(max(n, 1)))))
    base_slots = max(16, int(math.ceil(1.6 * n / float(bands))))
    candidates: list[tuple[float, float, float, float]] = []
    for b in range(bands):
        r = 0.24 + 0.70 * (float(b) / float(max(bands - 1, 1)))
        slots = max(12, int(math.ceil(base_slots * (0.65 + r))))
        for s in range(slots):
            ang = 2.0 * math.pi * (float(s) / float(slots))
            candidates.append((r, ang, cx + r * rx * math.cos(ang), cy + r * ry * math.sin(ang)))

    out = torch.zeros(n, 2, dtype=torch.float32, device=device)
    placed: list[int] = []
    fixed = fixed_mask.bool()
    for idx in range(n):
        if bool(fixed[idx]):
            out[idx] = base_positions[idx].float()
            placed.append(idx)

    def has_overlap(idx: int, x: float, y: float) -> bool:
        wi = float(sizes[idx, 0].item())
        hi = float(sizes[idx, 1].item())
        for other in placed:
            wj = float(sizes[other, 0].item())
            hj = float(sizes[other, 1].item())
            if abs(x - float(out[other, 0].item())) < 0.5 * (wi + wj) + 1e-4 and abs(y - float(out[other, 1].item())) < 0.5 * (hi + hj) + 1e-4:
                return True
        return False

    used: set[int] = set()
    for raw_idx in order.tolist():
        idx = int(raw_idx)
        if bool(fixed[idx]):
            continue
        tr = float(target_r[idx].item())
        ta = float(target_theta[idx].item())
        best_any = None
        best_any_score = float("inf")
        best_legal = None
        best_legal_score = float("inf")
        for cand_idx, (r, ang, x, y) in enumerate(candidates):
            if cand_idx in used:
                continue
            da = abs(math.atan2(math.sin(ang - ta), math.cos(ang - ta))) / math.pi
            score = (r - tr) ** 2 + 0.20 * da * da
            if score < best_any_score:
                best_any_score = score
                best_any = (cand_idx, x, y)
            if score < best_legal_score and not has_overlap(idx, x, y):
                best_legal_score = score
                best_legal = (cand_idx, x, y)
        chosen = best_legal if best_legal is not None else best_any
        if chosen is None:
            fallback = _flux_band_to_canvas(target_theta[idx : idx + 1], current[idx : idx + 1], sizes[idx : idx + 1], canvas_width, canvas_height, margin_frac)
            out[idx] = fallback[0]
        else:
            cand_idx, x, y = chosen
            used.add(cand_idx)
            out[idx, 0] = float(x)
            out[idx, 1] = float(y)
        placed.append(idx)

    # The candidate ordering above is the 1D-TSP/Weber approximation: within
    # each flux annulus, macros are greedily assigned to the nearest unused
    # poloidal site in graph-current order. The explicit `weber_sweeps` knob is
    # accepted so the R1 interface matches the documented plasma-startup plan;
    # later sweeps can refine this without changing the public config surface.
    return out


def _congestion_aware_flux_current(
    current: torch.Tensor,
    hard_xy: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    weight: float,
) -> torch.Tensor:
    """Adjust the R1 current ramp by a plasma-native routing-pressure proxy.

    This is not a local q-gradient move. It changes the startup flux-band
    assignment itself: macros that close many long high-current chords are
    moved toward outer flux bands, creating more routing cross-section before
    the GS/R4/R2 pipeline begins.
    """
    n = int(current.numel())
    if n <= 1 or abs(float(weight)) <= 0.0 or edge_index.numel() == 0 or edge_weight.numel() == 0:
        return current
    device = current.device
    edges = edge_index.detach().to(device=device, dtype=torch.long)
    weights = edge_weight.detach().to(device=device, dtype=torch.float32)
    load = torch.zeros(n, dtype=torch.float32, device=device)
    norm_w = max(float(canvas_width), 1e-9)
    norm_h = max(float(canvas_height), 1e-9)
    for e in range(int(edges.shape[0])):
        a = int(edges[e, 0].item())
        b = int(edges[e, 1].item())
        if not (0 <= a < n and 0 <= b < n and a != b):
            continue
        w = max(0.0, float(weights[e].item()))
        if w <= 0.0:
            continue
        dx = abs(float(hard_xy[a, 0].item()) - float(hard_xy[b, 0].item())) / norm_w
        dy = abs(float(hard_xy[a, 1].item()) - float(hard_xy[b, 1].item())) / norm_h
        # L1 span approximates a flux-chord demand; the sqrt damps outliers
        # so one extreme net does not erase the graph-current hierarchy.
        span = math.sqrt(max(dx + dy, 0.0))
        load[a] += float(w) * span
        load[b] += float(w) * span
    if float(load.max().item()) <= 1e-9:
        return current
    load = load / load.mean().clamp_min(1e-9)
    scale = current.std(unbiased=False).clamp_min(current.mean().abs().clamp_min(1e-6))
    # Positive weight evacuates high-span macros outward; negative weight is
    # the inward "pinch" polarity, shortening long flux chords at the cost of
    # central packing pressure.
    adjusted = current - float(weight) * scale * load
    return adjusted


def _soft_electron_startup(
    placement: torch.Tensor,
    sizes: torch.Tensor,
    fixed_mask: torch.Tensor,
    all_edge_index: torch.Tensor,
    all_edge_weight: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    num_hard: int,
    spread_frac: float,
    assignment_enabled: bool,
    assignment_graph_weight: float,
    assignment_overlap_weight: float,
    assignment_candidates: int,
    phase_lock_enabled: bool,
    phase_lock_weight: float,
    phase_lock_distance_weight: float,
    pressure_iters: int,
    pressure_step_frac: float,
    pressure_radius_frac: float,
    pressure_anchor: float,
    pressure_hard_weight: float,
    pressure_wall_weight: float,
) -> torch.Tensor:
    """Initialize soft macros by an adiabatic-electron graph response.

    Soft macros are treated as electron density parcels tied to the current
    graph. Each movable soft cell is placed near the weighted centroid of its
    already initialized neighbors, with a deterministic gyro-phase offset to
    avoid collapse. No TILOS/input soft positions are used for movable cells.
    """
    out = placement.clone().float()
    n_total = int(out.shape[0])
    n_soft = max(0, n_total - int(num_hard))
    if n_soft <= 0:
        return out
    device = out.device
    edges = all_edge_index.to(device=device, dtype=torch.long)
    weights = all_edge_weight.to(device=device, dtype=torch.float32)
    if edges.numel() == 0 or weights.numel() == 0:
        return out

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n_total)]
    for e in range(int(edges.shape[0])):
        a = int(edges[e, 0].item())
        b = int(edges[e, 1].item())
        if 0 <= a < n_total and 0 <= b < n_total and a != b:
            w = max(0.0, float(weights[e].item()))
            adjacency[a].append((b, w))
            adjacency[b].append((a, w))

    soft_indices = list(range(int(num_hard), n_total))
    soft_indices.sort(key=lambda idx: -sum(w for _nbr, w in adjacency[idx]))
    placed = torch.zeros(n_total, dtype=torch.bool, device=device)
    placed[: int(num_hard)] = True
    fixed = fixed_mask.to(device=device).bool()
    placed[fixed] = True

    span = max(float(canvas_width), float(canvas_height), 1.0)
    phase = math.pi * (3.0 - math.sqrt(5.0))
    cx = 0.5 * float(canvas_width)
    cy = 0.5 * float(canvas_height)
    rx = 0.44 * float(canvas_width)
    ry = 0.44 * float(canvas_height)
    fallback_anchor = torch.zeros(n_total, 2, dtype=torch.float32, device=device)
    soft_rank = {idx: seq for seq, idx in enumerate(soft_indices)}
    for idx, seq in soft_rank.items():
        fallback_rank = float(seq + 1) / float(max(len(soft_indices), 1))
        fallback_r = 0.18 + 0.76 * math.sqrt(fallback_rank)
        fallback_theta = phase * float(seq + 1)
        fallback_anchor[idx, 0] = cx + fallback_r * rx * math.cos(fallback_theta)
        fallback_anchor[idx, 1] = cy + fallback_r * ry * math.sin(fallback_theta)

    def clamp_idx(idx: int, xy: torch.Tensor) -> torch.Tensor:
        half_w = 0.5 * float(sizes[idx, 0].item())
        half_h = 0.5 * float(sizes[idx, 1].item())
        xy = xy.clone()
        xy[0] = max(half_w, min(float(canvas_width) - half_w, float(xy[0].item())))
        xy[1] = max(half_h, min(float(canvas_height) - half_h, float(xy[1].item())))
        return xy

    if bool(assignment_enabled):
        cols = max(8, int(math.ceil(math.sqrt(max(n_soft, 1) * float(canvas_width) / max(float(canvas_height), 1e-9)))))
        rows = max(8, int(math.ceil(n_soft / max(cols, 1))))
        xs = torch.linspace(0.06 * float(canvas_width), 0.94 * float(canvas_width), cols, device=device)
        ys = torch.linspace(0.06 * float(canvas_height), 0.94 * float(canvas_height), rows, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        sites = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        used_sites: set[int] = set()
    else:
        sites = torch.zeros(0, 2, dtype=torch.float32, device=device)
        used_sites = set()

    for seq, idx in enumerate(soft_indices):
        if bool(fixed[idx]):
            continue
        numerator = torch.zeros(2, dtype=torch.float32, device=device)
        denom = 0.0
        # First respond to placed neighbors, especially hard-current coils.
        for nbr, w in adjacency[idx]:
            if bool(placed[nbr]):
                boost = 2.0 if nbr < int(num_hard) else 1.0
                numerator += float(w * boost) * out[nbr]
                denom += float(w * boost)
            elif nbr >= int(num_hard):
                # Pure adiabatic-electron prediction for not-yet-placed soft
                # neighbors: use their own startup flux anchor, never inherited
                # benchmark/TILOS coordinates.
                numerator += float(w) * fallback_anchor[nbr]
                denom += float(w)
        if denom <= 1e-9:
            base = fallback_anchor[idx]
        else:
            base = numerator / max(denom, 1e-9)
        theta = phase * float(seq + 1)
        radius = max(0.002, float(spread_frac)) * span * (1.0 + 0.35 * float(seq % 5))
        offset = torch.tensor([math.cos(theta) * radius, math.sin(theta) * radius], dtype=torch.float32, device=device)
        target = clamp_idx(idx, base + offset)
        if bool(assignment_enabled) and sites.numel() > 0:
            diff = sites - target.unsqueeze(0)
            scores = diff[:, 0].pow(2) + diff[:, 1].pow(2)
            for used in used_sites:
                if 0 <= used < int(scores.numel()):
                    scores[used] = float("inf")
            graph_weight = max(0.0, float(assignment_graph_weight))
            overlap_weight = max(0.0, float(assignment_overlap_weight))
            phase_weight = max(0.0, float(phase_lock_weight)) if bool(phase_lock_enabled) else 0.0
            distance_weight = max(0.0, float(phase_lock_distance_weight))
            if graph_weight > 0.0 or overlap_weight > 0.0 or phase_weight > 0.0:
                k = max(1, min(int(assignment_candidates), int(scores.numel())))
                cand_scores, cand_idx = torch.topk(scores, k=k, largest=False)
                best_score = float("inf")
                best_site = int(cand_idx[0].item())
                placed_ids = placed.nonzero(as_tuple=False).reshape(-1)
                target_angle = math.atan2(float((base[1] - cy).item()), float((base[0] - cx).item()))
                for local_pos, site_tensor in enumerate(cand_idx):
                    site_i = int(site_tensor.item())
                    if not math.isfinite(float(cand_scores[local_pos].item())):
                        continue
                    site_xy = sites[site_i]
                    tension = 0.0
                    tension_w = 0.0
                    for nbr, w in adjacency[idx]:
                        if bool(placed[nbr]):
                            nbr_xy = out[nbr]
                        elif nbr >= int(num_hard):
                            nbr_xy = fallback_anchor[nbr]
                        else:
                            continue
                        boost = 2.0 if nbr < int(num_hard) else 1.0
                        edge_w = max(0.0, float(w)) * boost
                        dxy = site_xy - nbr_xy
                        tension += edge_w * float(dxy[0].item() * dxy[0].item() + dxy[1].item() * dxy[1].item())
                        tension_w += edge_w
                    if tension_w > 1e-9:
                        tension /= tension_w
                    overlap_penalty = 0.0
                    if overlap_weight > 0.0 and int(placed_ids.numel()) > 0:
                        placed_pos = out[placed_ids]
                        placed_sizes = sizes[placed_ids].to(device=device, dtype=torch.float32)
                        dx = torch.abs(placed_pos[:, 0] - site_xy[0])
                        dy = torch.abs(placed_pos[:, 1] - site_xy[1])
                        sep_x = 0.5 * (placed_sizes[:, 0] + float(sizes[idx, 0].item()))
                        sep_y = 0.5 * (placed_sizes[:, 1] + float(sizes[idx, 1].item()))
                        ox = torch.relu(sep_x - dx)
                        oy = torch.relu(sep_y - dy)
                        area = ox * oy
                        hit = area > 0.0
                        if bool(hit.any()):
                            # Bohm-sheath startup exclusion: an occupied flux
                            # tube carries a large pedestal energy, so legal
                            # sites beat colliding sites whenever available.
                            overlap_penalty = float(hit.sum().item()) * span * span + float(area.sum().item())
                    phase_penalty = 0.0
                    if phase_weight > 0.0:
                        site_angle = math.atan2(float((site_xy[1] - cy).item()), float((site_xy[0] - cx).item()))
                        delta = math.atan2(math.sin(site_angle - target_angle), math.cos(site_angle - target_angle))
                        phase_penalty = (delta * delta) * span * span
                    total_score = (
                        distance_weight * float(cand_scores[local_pos].item())
                        + graph_weight * tension
                        + overlap_weight * overlap_penalty
                        + phase_weight * phase_penalty
                    )
                    if total_score < best_score:
                        best_score = total_score
                        best_site = site_i
                site_idx = best_site
            else:
                site_idx = int(torch.argmin(scores).item())
            used_sites.add(site_idx)
            out[idx] = clamp_idx(idx, sites[site_idx])
        else:
            out[idx] = target
        placed[idx] = True
    pressure_iters = max(0, int(pressure_iters))
    if pressure_iters <= 0:
        return out

    soft_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    soft_mask[int(num_hard) :] = True
    soft_mask &= ~fixed
    soft_ids = soft_mask.nonzero(as_tuple=False).reshape(-1)
    if int(soft_ids.numel()) <= 1:
        return out
    anchors = out.clone()
    radius = max(1e-6, float(pressure_radius_frac) * span)
    step = max(0.0, float(pressure_step_frac) * span)
    anchor = max(0.0, float(pressure_anchor))
    hard_weight = max(0.0, float(pressure_hard_weight))
    wall_weight = max(0.0, float(pressure_wall_weight))
    soft_sizes = sizes[soft_ids].to(device=device, dtype=torch.float32)
    soft_scale = torch.sqrt(torch.clamp(soft_sizes[:, 0] * soft_sizes[:, 1], min=1e-12))
    hard_pos = out[: int(num_hard)]
    hard_sizes = sizes[: int(num_hard)].to(device=device, dtype=torch.float32)

    for _ in range(pressure_iters):
        pos = out[soft_ids]
        delta = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist = torch.linalg.norm(delta, dim=2).clamp_min(1e-9)
        size_sep = 0.5 * (soft_scale.unsqueeze(1) + soft_scale.unsqueeze(0))
        influence = torch.relu(radius + size_sep - dist) / (radius + size_sep).clamp_min(1e-9)
        influence.fill_diagonal_(0.0)
        repulse = (delta / dist.unsqueeze(2) * influence.unsqueeze(2)).sum(dim=1)
        graph_tension = anchor * (anchors[soft_ids] - pos) / max(span, 1e-9)
        force = repulse + graph_tension
        if hard_weight > 0.0 and int(num_hard) > 0:
            dx = pos[:, 0].unsqueeze(1) - hard_pos[:, 0].unsqueeze(0)
            dy = pos[:, 1].unsqueeze(1) - hard_pos[:, 1].unsqueeze(0)
            sep_x = 0.5 * (soft_sizes[:, 0].unsqueeze(1) + hard_sizes[:, 0].unsqueeze(0)) + radius
            sep_y = 0.5 * (soft_sizes[:, 1].unsqueeze(1) + hard_sizes[:, 1].unsqueeze(0)) + radius
            ox = torch.relu(sep_x - torch.abs(dx)) / sep_x.clamp_min(1e-9)
            oy = torch.relu(sep_y - torch.abs(dy)) / sep_y.clamp_min(1e-9)
            sign_x = torch.sign(dx)
            sign_y = torch.sign(dy)
            sign_x = torch.where(torch.abs(dx) < 1e-9, torch.ones_like(sign_x), sign_x)
            sign_y = torch.where(torch.abs(dy) < 1e-9, torch.ones_like(sign_y), sign_y)
            hard_force = torch.stack(
                [(sign_x * ox * (1.0 + oy)).sum(dim=1), (sign_y * oy * (1.0 + ox)).sum(dim=1)],
                dim=1,
            )
            force = force + hard_weight * hard_force
        if wall_weight > 0.0:
            half_w = 0.5 * soft_sizes[:, 0]
            half_h = 0.5 * soft_sizes[:, 1]
            x = pos[:, 0]
            y = pos[:, 1]
            force[:, 0] += wall_weight * torch.relu(radius - (x - half_w)) / radius
            force[:, 0] -= wall_weight * torch.relu(radius - (float(canvas_width) - (x + half_w))) / radius
            force[:, 1] += wall_weight * torch.relu(radius - (y - half_h)) / radius
            force[:, 1] -= wall_weight * torch.relu(radius - (float(canvas_height) - (y + half_h))) / radius
        norm = torch.linalg.norm(force, dim=1, keepdim=True).clamp_min(1e-9)
        move = force / norm * torch.clamp(norm * step, max=step)
        out[soft_ids] = pos + move
        for idx in soft_ids.tolist():
            out[idx] = clamp_idx(int(idx), out[idx])
    return out


def spectral_plasma_startup_init(
    base_positions: torch.Tensor,
    sizes: torch.Tensor,
    fixed_mask: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    *,
    num_hard: int,
    margin_frac: float = 0.04,
    soft_grid_enabled: bool = False,
    soft_graph_enabled: bool = True,
    soft_assignment_enabled: bool = False,
    soft_assignment_graph_weight: float = 0.0,
    soft_assignment_overlap_weight: float = 0.0,
    soft_assignment_candidates: int = 48,
    soft_phase_lock_enabled: bool = False,
    soft_phase_lock_weight: float = 0.0,
    soft_phase_lock_distance_weight: float = 1.0,
    flux_band_enabled: bool = True,
    all_edge_index: torch.Tensor | None = None,
    all_edge_weight: torch.Tensor | None = None,
    soft_spread_frac: float = 0.018,
    soft_pressure_iters: int = 0,
    soft_pressure_step_frac: float = 0.002,
    soft_pressure_radius_frac: float = 0.035,
    soft_pressure_anchor: float = 0.35,
    soft_pressure_hard_weight: float = 1.0,
    soft_pressure_wall_weight: float = 0.5,
    normalized_laplacian: bool = True,
    n_bands: int = 8,
    sinkhorn_iters: int = 100,
    sinkhorn_eps: float = 0.05,
    bare_picard_outer: int = 6,
    weber_sweeps: int = 2,
    congestion_aware_flux_enabled: bool = False,
    congestion_aware_flux_weight: float = 0.25,
) -> torch.Tensor:
    """Return a plasma-startup placement candidate.

    Hard macros use the first two non-trivial normalized graph-Laplacian
    eigenmodes. Fixed macros keep their official constraint positions.
    Movable hard macros and, when soft-grid startup is enabled, movable soft
    macros are generated from graph-current flux coordinates instead of
    inherited PLC positions.
    """
    placement = base_positions.clone().float()
    if num_hard <= 1:
        return placement

    device = placement.device
    hard_sizes = sizes[:num_hard].to(device=device, dtype=torch.float32)
    fixed = fixed_mask[:num_hard].to(device=device).bool()
    movable = ~fixed
    n = int(num_hard)
    W = torch.zeros((n, n), dtype=torch.float32, device=device)
    if edge_index.numel() > 0:
        edges = edge_index.detach().to(device=device, dtype=torch.long)
        weights = edge_weight.detach().to(device=device, dtype=torch.float32)
        for e in range(int(edges.shape[0])):
            a = int(edges[e, 0].item())
            b = int(edges[e, 1].item())
            if 0 <= a < n and 0 <= b < n and a != b:
                w = max(0.0, float(weights[e].item()))
                W[a, b] += w
                W[b, a] += w
    deg = W.sum(dim=1)
    if float(deg.max().item()) <= 1e-9 or int(movable.sum().item()) < 2:
        hard_xy = _fallback_ring(n, canvas_width, canvas_height, device)
    else:
        if bool(normalized_laplacian):
            inv_sqrt = torch.rsqrt(torch.clamp(deg, min=1e-9))
            L = torch.eye(n, dtype=torch.float32, device=device) - inv_sqrt.unsqueeze(1) * W * inv_sqrt.unsqueeze(0)
        else:
            L = torch.diag(deg) - W
        # A tiny diagonal ramp gives deterministic eigenvectors when the graph
        # has repeated modes, analogous to a seeded current-ramp asymmetry.
        L = L + torch.diag(torch.linspace(0.0, 1e-6, steps=n, device=device))
        try:
            vals, vecs = torch.linalg.eigh(L)
            if vecs.shape[1] >= 3:
                modes = vecs[:, 1:3]
                if bool(flux_band_enabled):
                    theta = torch.atan2(modes[:, 1], modes[:, 0])
                    area = hard_sizes[:, 0] * hard_sizes[:, 1]
                    area = area / area.mean().clamp_min(1e-9)
                    current = deg + 0.10 * area
                    hard_xy = _flux_lattice_startup(
                        theta,
                        current,
                        hard_sizes,
                        fixed,
                        base_positions[:num_hard].to(device=device, dtype=torch.float32),
                        canvas_width,
                        canvas_height,
                        margin_frac,
                        int(n_bands),
                        int(weber_sweeps),
                    )
                    if bool(congestion_aware_flux_enabled) and abs(float(congestion_aware_flux_weight)) > 0.0:
                        adjusted_current = _congestion_aware_flux_current(
                            current,
                            hard_xy,
                            edge_index,
                            edge_weight,
                            canvas_width,
                            canvas_height,
                            float(congestion_aware_flux_weight),
                        )
                        hard_xy = _flux_lattice_startup(
                            theta,
                            adjusted_current,
                            hard_sizes,
                            fixed,
                            base_positions[:num_hard].to(device=device, dtype=torch.float32),
                            canvas_width,
                            canvas_height,
                            margin_frac,
                            int(n_bands),
                            int(weber_sweeps),
                        )
                else:
                    hard_xy = modes
            else:
                hard_xy = _fallback_ring(n, canvas_width, canvas_height, device)
        except Exception:
            hard_xy = _fallback_ring(n, canvas_width, canvas_height, device)
    if not bool(flux_band_enabled):
        hard_xy = _scale_to_canvas(hard_xy, hard_sizes, canvas_width, canvas_height, margin_frac)
    placement[:num_hard][movable] = hard_xy[movable]
    placement[:num_hard][fixed] = base_positions[:num_hard][fixed].float()

    if soft_grid_enabled and placement.shape[0] > num_hard:
        if bool(soft_graph_enabled) and all_edge_index is not None and all_edge_weight is not None and all_edge_index.numel() > 0:
            placement = _soft_electron_startup(
                placement,
                sizes.to(device=device, dtype=torch.float32),
                fixed_mask.to(device=device),
                all_edge_index,
                all_edge_weight,
                canvas_width,
                canvas_height,
                num_hard,
                soft_spread_frac,
                soft_assignment_enabled,
                soft_assignment_graph_weight,
                soft_assignment_overlap_weight,
                soft_assignment_candidates,
                soft_phase_lock_enabled,
                soft_phase_lock_weight,
                soft_phase_lock_distance_weight,
                soft_pressure_iters,
                soft_pressure_step_frac,
                soft_pressure_radius_frac,
                soft_pressure_anchor,
                soft_pressure_hard_weight,
                soft_pressure_wall_weight,
            )
            return placement
        n_soft = int(placement.shape[0] - num_hard)
        cols = max(1, int(math.ceil(math.sqrt(n_soft * float(canvas_width) / max(float(canvas_height), 1e-9)))))
        rows = max(1, int(math.ceil(n_soft / cols)))
        xs = torch.linspace(0.1 * float(canvas_width), 0.9 * float(canvas_width), cols, device=device)
        ys = torch.linspace(0.1 * float(canvas_height), 0.9 * float(canvas_height), rows, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)[:n_soft]
        soft_fixed = fixed_mask[num_hard:].to(device=device).bool()
        placement[num_hard:][~soft_fixed] = grid[~soft_fixed]
        placement[num_hard:][soft_fixed] = base_positions[num_hard:][soft_fixed].float()
    return placement
