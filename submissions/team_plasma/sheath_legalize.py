"""
Continuous Bohm-sheath / Coulomb hard-macro legalization.

This is the R2 plasma-native replacement path from the research log. Instead
of snapping overlaps apart with a geometric rule, macros evolve under a
short-range repulsive core force plus an inward wall-sheath force. The fixed
point is a legal on-canvas placement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from legalization import count_hard_overlaps, overlap_pairs, project_to_canvas


@dataclass
class SheathLegalizeConfig:
    enabled: bool = False
    fallback_strict: bool = True
    max_iters: int = 160
    dt: float = 0.35
    damping: float = 0.60
    coulomb_strength: float = 1.0
    sheath_strength: float = 1.0
    debye_frac: float = 0.02
    max_step_frac: float = 0.010
    continuation_enabled: bool = False
    beta_overlap_start: float = 0.10
    beta_overlap_max: float = 1.0
    beta_overlap_ramp_stages: int = 8
    beta_wall: float = 1.0
    inner_iters: int = 40
    temp_start: float = 0.0
    temp_decay: float = 0.70
    force_tol: float = 1e-5
    seed: int = 7
    hardening_enabled: bool = True
    hardening_iters: int = 320
    hardening_max_pairs_per_iter: int = 12000
    relocation_enabled: bool = True
    relocation_iters: int = 0
    relocation_rings: int = 6
    relocation_density_weight: float = 0.0
    flux_lattice_relocation_enabled: bool = True
    flux_lattice_cols: int = 16
    local_flux_refine_enabled: bool = False
    local_flux_refine_steps: int = 5
    local_flux_refine_pair_threshold: int = 4
    anchor_strength: float = 0.0
    anchor_release_frac: float = 0.5


def bohm_sheath_legalize(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    fixed_mask: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    num_hard: int,
    gap: float = 1e-4,
    cfg: SheathLegalizeConfig | None = None,
) -> torch.Tensor:
    """Relax hard macros under overlap-core repulsion and wall sheath forces."""
    if cfg is None:
        cfg = SheathLegalizeConfig(enabled=True)

    out = positions.clone().float()
    n = int(num_hard)
    if n <= 1:
        return out

    canvas_w = float(canvas_width)
    canvas_h = float(canvas_height)
    span = max(canvas_w, canvas_h, 1.0)
    hard_sizes = sizes[:n].float()
    hard_fixed = fixed_mask[:n].bool()
    fixed_pos = out[:n].clone()

    half_w = 0.5 * hard_sizes[:, 0]
    half_h = 0.5 * hard_sizes[:, 1]
    debye = max(1e-6, float(cfg.debye_frac) * span)
    max_step = max(1e-6, float(cfg.max_step_frac) * span)
    dt = max(1e-5, float(cfg.dt))
    damping = min(0.99, max(0.0, float(cfg.damping)))
    coulomb = max(0.0, float(cfg.coulomb_strength))
    sheath = max(0.0, float(cfg.sheath_strength))

    velocity = torch.zeros_like(out[:n])
    out[:n] = project_to_canvas(
        out[:n],
        hard_sizes,
        canvas_w,
        canvas_h,
        fixed_mask=hard_fixed,
        fixed_positions=fixed_pos,
    )
    anchor_pos = out[:n].clone()
    anchor_strength = max(0.0, float(getattr(cfg, "anchor_strength", 0.0)))

    # Deterministic tiny angle table for exactly coincident macro centers.
    angles = torch.arange(n, dtype=out.dtype, device=out.device) * (math.pi * (3.0 - math.sqrt(5.0)))
    jitter_dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

    if bool(cfg.continuation_enabled):
        stages = max(1, int(cfg.beta_overlap_ramp_stages))
        inner_iters = max(1, int(cfg.inner_iters))
        beta_start = max(0.0, float(cfg.beta_overlap_start))
        beta_max = max(beta_start, float(cfg.beta_overlap_max))
    else:
        stages = 1
        inner_iters = max(1, int(cfg.max_iters))
        beta_start = max(0.0, float(cfg.coulomb_strength))
        beta_max = beta_start

    gen = torch.Generator(device=out.device)
    gen.manual_seed(int(cfg.seed))
    temp = max(0.0, float(cfg.temp_start))
    total_iters = 0
    for stage in range(stages):
        if stages <= 1:
            beta = beta_max
            stage_frac = 1.0
        else:
            stage_frac = float(stage) / float(max(stages - 1, 1))
            beta = beta_start * ((beta_max / max(beta_start, 1e-9)) ** stage_frac) if beta_start > 0 else beta_max * stage_frac
        release_frac = max(1e-6, float(getattr(cfg, "anchor_release_frac", 0.5)))
        stage_anchor = anchor_strength * max(0.0, 1.0 - stage_frac / release_frac)
        wall_beta = float(cfg.beta_wall) if bool(cfg.continuation_enabled) else sheath
        for _inner in range(inner_iters):
            total_iters += 1
            if total_iters > max(1, int(cfg.max_iters)) and bool(cfg.continuation_enabled):
                break
            pos = out[:n]
            force = torch.zeros_like(pos)

            dx = pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)
            dy = pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)
            abs_dx = torch.abs(dx)
            abs_dy = torch.abs(dy)
            sep_x = half_w.unsqueeze(1) + half_w.unsqueeze(0) + float(gap)
            sep_y = half_h.unsqueeze(1) + half_h.unsqueeze(0) + float(gap)
            overlap_x = sep_x - abs_dx
            overlap_y = sep_y - abs_dy
            mask = (overlap_x > 0.0) & (overlap_y > 0.0)
            mask = torch.triu(mask, diagonal=1)
            pairs = mask.nonzero(as_tuple=False)

            if pairs.numel() > 0:
                i = pairs[:, 0]
                j = pairs[:, 1]
                vx = dx[i, j]
                vy = dy[i, j]
                sign_x = torch.sign(vx)
                sign_y = torch.sign(vy)
                zero_x = torch.abs(vx) < 1e-9
                zero_y = torch.abs(vy) < 1e-9
                if bool(zero_x.any()):
                    sign_x[zero_x] = torch.sign(jitter_dirs[i[zero_x], 0])
                if bool(zero_y.any()):
                    sign_y[zero_y] = torch.sign(jitter_dirs[i[zero_y], 1])

                # Smooth rectangular Coulomb-core force. This is the gradient of
                # a soft overlap energy, so it respects macro geometry while still
                # behaving like short-range charged-core repulsion.
                ox = overlap_x[i, j] / sep_x[i, j].clamp_min(1e-9)
                oy = overlap_y[i, j] / sep_y[i, j].clamp_min(1e-9)
                fx = sign_x * ox * (1.0 + oy)
                fy = sign_y * oy * (1.0 + ox)
                pair_force = beta * torch.stack([fx, fy], dim=1)
                force.index_add_(0, i, pair_force)
                force.index_add_(0, j, -pair_force)

            # Bohm/Debye wall sheath: macros entering the sheath are reflected
            # inward by a smooth field whose scale is the Debye length.
            x = pos[:, 0]
            y = pos[:, 1]
            left = x - half_w
            right = canvas_w - (x + half_w)
            bottom = y - half_h
            top = canvas_h - (y + half_h)
            force[:, 0] += wall_beta * torch.relu(debye - left) / debye
            force[:, 0] -= wall_beta * torch.relu(debye - right) / debye
            force[:, 1] += wall_beta * torch.relu(debye - bottom) / debye
            force[:, 1] -= wall_beta * torch.relu(debye - top) / debye

            if stage_anchor > 0.0:
                # Cost-aware magnetic shape control: keep the plasma close to
                # the incoming low-proxy basin while overlap pedestals discharge.
                force = force - stage_anchor * (pos - anchor_pos) / span

            if temp > 0.0:
                force = force + temp * torch.randn(force.shape, dtype=force.dtype, device=force.device, generator=gen)
            force[hard_fixed] = 0.0
            velocity = damping * velocity + dt * force
            step_norm = torch.linalg.norm(velocity, dim=1, keepdim=True).clamp_min(1e-12)
            velocity = velocity * torch.clamp(max_step / step_norm, max=1.0)
            velocity[hard_fixed] = 0.0
            out[:n] = out[:n] + velocity
            out[:n] = project_to_canvas(
                out[:n],
                hard_sizes,
                canvas_w,
                canvas_h,
                fixed_mask=hard_fixed,
                fixed_positions=fixed_pos,
            )

            if pairs.numel() == 0 and float(torch.linalg.norm(velocity, dim=1).max().item()) < float(cfg.force_tol):
                break
        temp *= max(0.0, min(1.0, float(cfg.temp_decay)))

    if bool(cfg.continuation_enabled) and bool(cfg.hardening_enabled):
        eps = max(1e-6, 1e-6 * span)
        relocation_iters = int(getattr(cfg, "relocation_iters", 0))
        if relocation_iters <= 0:
            relocation_iters = int(cfg.hardening_iters)
        for _ in range(max(1, relocation_iters)):
            pairs = overlap_pairs(out, sizes, n, gap=0.0)
            if pairs.numel() == 0:
                break
            pair_list = pairs.tolist()
            if len(pair_list) > int(cfg.hardening_max_pairs_per_iter):
                pair_list = pair_list[: int(cfg.hardening_max_pairs_per_iter)]
            moved = False
            for i, j in pair_list:
                if bool(hard_fixed[i]) and bool(hard_fixed[j]):
                    continue
                xi = float(out[i, 0].item())
                yi = float(out[i, 1].item())
                xj = float(out[j, 0].item())
                yj = float(out[j, 1].item())
                wi = float(hard_sizes[i, 0].item())
                hi = float(hard_sizes[i, 1].item())
                wj = float(hard_sizes[j, 0].item())
                hj = float(hard_sizes[j, 1].item())
                overlap_x = 0.5 * (wi + wj) - abs(xj - xi)
                overlap_y = 0.5 * (hi + hj) - abs(yj - yi)
                if overlap_x <= 0.0 or overlap_y <= 0.0:
                    continue
                # ELM-pacing hardening: discharge the smallest overlap pedestal.
                if overlap_x / max(wi + wj, 1e-9) <= overlap_y / max(hi + hj, 1e-9):
                    sign = 1.0 if xj >= xi else -1.0
                    delta = overlap_x + eps
                    if not bool(hard_fixed[i]) and not bool(hard_fixed[j]):
                        out[i, 0] -= 0.5 * sign * delta
                        out[j, 0] += 0.5 * sign * delta
                    elif bool(hard_fixed[i]) and not bool(hard_fixed[j]):
                        out[j, 0] += sign * delta
                    elif not bool(hard_fixed[i]) and bool(hard_fixed[j]):
                        out[i, 0] -= sign * delta
                else:
                    sign = 1.0 if yj >= yi else -1.0
                    delta = overlap_y + eps
                    if not bool(hard_fixed[i]) and not bool(hard_fixed[j]):
                        out[i, 1] -= 0.5 * sign * delta
                        out[j, 1] += 0.5 * sign * delta
                    elif bool(hard_fixed[i]) and not bool(hard_fixed[j]):
                        out[j, 1] += sign * delta
                    elif not bool(hard_fixed[i]) and bool(hard_fixed[j]):
                        out[i, 1] -= sign * delta
                moved = True
            out[:n] = project_to_canvas(
                out[:n],
                hard_sizes,
                canvas_w,
                canvas_h,
                fixed_mask=hard_fixed,
                fixed_positions=fixed_pos,
            )
            if not moved:
                break

    if bool(cfg.continuation_enabled) and bool(cfg.relocation_enabled) and count_hard_overlaps(out, sizes, n, gap=0.0) > 0:
        eps = max(1e-6, 1e-6 * span)

        def clamp_center(idx: int, x: float, y: float) -> tuple[float, float]:
            lo_x = float(half_w[idx].item())
            hi_x = canvas_w - float(half_w[idx].item())
            lo_y = float(half_h[idx].item())
            hi_y = canvas_h - float(half_h[idx].item())
            return max(lo_x, min(hi_x, x)), max(lo_y, min(hi_y, y))

        def overlaps_idx(idx: int) -> bool:
            pos = out[:n]
            dx = torch.abs(pos[:, 0] - pos[idx, 0])
            dy = torch.abs(pos[:, 1] - pos[idx, 1])
            sep_x = 0.5 * (hard_sizes[:, 0] + hard_sizes[idx, 0])
            sep_y = 0.5 * (hard_sizes[:, 1] + hard_sizes[idx, 1])
            mask = (dx < sep_x) & (dy < sep_y)
            mask[idx] = False
            return bool(mask.any())

        for _ in range(max(1, int(cfg.hardening_iters))):
            pairs = overlap_pairs(out, sizes, n, gap=0.0)
            if pairs.numel() == 0:
                break
            pair_count = int(pairs.shape[0])
            offenders = sorted(set(int(x) for x in pairs.reshape(-1).tolist()), key=lambda idx: float((hard_sizes[idx, 0] * hard_sizes[idx, 1]).item()))
            moved_any = False
            for idx in offenders:
                if bool(hard_fixed[idx]):
                    continue
                cur = out[idx].clone()
                width = float(hard_sizes[idx, 0].item())
                height = float(hard_sizes[idx, 1].item())
                radius0 = max(width, height, 0.01 * span)
                candidates: list[tuple[float, float]] = []
                for ring in range(1, max(1, int(cfg.relocation_rings)) + 1):
                    radius = radius0 * float(ring)
                    for ax, ay in (
                        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                        (0.707, 0.707), (0.707, -0.707), (-0.707, 0.707), (-0.707, -0.707),
                    ):
                        candidates.append((float(cur[0].item()) + ax * radius, float(cur[1].item()) + ay * radius))
                pair_threshold = max(0, int(getattr(cfg, "local_flux_refine_pair_threshold", 4)))
                if bool(getattr(cfg, "local_flux_refine_enabled", False)) and pair_count <= pair_threshold:
                    # Local Debye-scale flux scan: before jumping to the
                    # global lattice, sample a small rectangular flux tube
                    # around the offender. This is a pure R2 repair that tries
                    # to preserve the incoming basin while discharging the
                    # last overlap island.
                    steps = max(1, int(getattr(cfg, "local_flux_refine_steps", 5)))
                    max_radius = radius0 * float(max(1, int(cfg.relocation_rings)))
                    xs_local = torch.linspace(-max_radius, max_radius, 2 * steps + 1, dtype=out.dtype, device=out.device)
                    ys_local = torch.linspace(-max_radius, max_radius, 2 * steps + 1, dtype=out.dtype, device=out.device)
                    cx = float(cur[0].item())
                    cy = float(cur[1].item())
                    local: list[tuple[float, float, float]] = []
                    for dy in ys_local.tolist():
                        for dx in xs_local.tolist():
                            if abs(float(dx)) <= 1e-12 and abs(float(dy)) <= 1e-12:
                                continue
                            dist = float(dx) ** 2 + float(dy) ** 2
                            local.append((dist, cx + float(dx), cy + float(dy)))
                    local.sort(key=lambda item: item[0])
                    candidates.extend((x, y) for _dist, x, y in local)
                if bool(getattr(cfg, "flux_lattice_relocation_enabled", True)):
                    # Last-resort flux-tube relocation: sample a deterministic
                    # lattice of Debye-scale flux surfaces and pick the closest
                    # non-overlapping equilibrium point. This keeps R2 pure
                    # plasma-derived instead of falling back to strict geometry.
                    cols = max(2, int(getattr(cfg, "flux_lattice_cols", 16)))
                    rows = max(2, int(math.ceil(cols * canvas_h / max(canvas_w, 1e-9))))
                    lo_x = float(half_w[idx].item())
                    hi_x = canvas_w - float(half_w[idx].item())
                    lo_y = float(half_h[idx].item())
                    hi_y = canvas_h - float(half_h[idx].item())
                    if hi_x >= lo_x and hi_y >= lo_y:
                        xs = torch.linspace(lo_x, hi_x, cols, dtype=out.dtype, device=out.device)
                        ys = torch.linspace(lo_y, hi_y, rows, dtype=out.dtype, device=out.device)
                        # Sort by distance from the current magnetic surface so
                        # the scan remains basin-preserving when several holes
                        # are legal.
                        lattice: list[tuple[float, float, float]] = []
                        cx = float(cur[0].item())
                        cy = float(cur[1].item())
                        for yy in ys.tolist():
                            for xx in xs.tolist():
                                dist = (cx - float(xx)) ** 2 + (cy - float(yy)) ** 2
                                lattice.append((dist, float(xx), float(yy)))
                        lattice.sort(key=lambda item: item[0])
                        candidates.extend((x, y) for _dist, x, y in lattice)
                best = None
                best_dist = float("inf")
                density_weight = max(0.0, float(getattr(cfg, "relocation_density_weight", 0.0)))
                for raw_x, raw_y in candidates:
                    x, y = clamp_center(idx, raw_x, raw_y)
                    out[idx, 0] = x
                    out[idx, 1] = y
                    if not overlaps_idx(idx):
                        dist = (float(cur[0].item()) - x) ** 2 + (float(cur[1].item()) - y) ** 2
                        if density_weight > 0.0:
                            other = torch.cat([out[:idx], out[idx + 1:n]], dim=0)
                            delta = other - torch.tensor([x, y], dtype=out.dtype, device=out.device)
                            d2 = (delta[:, 0] / span).pow(2) + (delta[:, 1] / span).pow(2)
                            pressure = float(torch.exp(-d2 / max(2.0 * (0.08 ** 2), 1e-9)).sum().item())
                        else:
                            pressure = 0.0
                        score = dist + density_weight * (span * span) * pressure
                        if score < best_dist:
                            best_dist = score
                            best = (x, y)
                out[idx] = cur
                if best is not None:
                    out[idx, 0] = best[0]
                    out[idx, 1] = best[1]
                    moved_any = True
                    break
            out[:n] = project_to_canvas(
                out[:n],
                hard_sizes,
                canvas_w,
                canvas_h,
                fixed_mask=hard_fixed,
                fixed_positions=fixed_pos,
            )
            if not moved_any:
                break

    return out
