"""
Hard-macro legalization: remove overlaps after the GS solve.

The GS equilibrium discourages overlaps because hard macros are
current-bearing coils with finite extent (they repel each other through
the equilibrium itself). But the FD grid is coarse and the equilibrium
is approximate, so a final pairwise cleanup pass is still needed to
guarantee zero overlaps.

Ported from `team-plasma-release/submissions/team_plasma/placer.py` and
simplified: pairwise push apart along the cheaper axis, optionally
weighted by an anchor-restoration term.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch


def project_to_canvas(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    fixed_mask: Optional[torch.Tensor] = None,
    fixed_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Clamp macro centers so that the full macro stays inside the canvas."""
    out = positions.clone()
    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5

    x_min = half_w
    x_max = torch.full_like(half_w, float(canvas_width)) - half_w
    y_min = half_h
    y_max = torch.full_like(half_h, float(canvas_height)) - half_h

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)

    out_x = torch.max(torch.min(out[:, 0], x_max), x_min)
    out_y = torch.max(torch.min(out[:, 1], y_max), y_min)

    out[:, 0] = torch.where(x_max >= x_min, out_x, x_mid)
    out[:, 1] = torch.where(y_max >= y_min, out_y, y_mid)

    if fixed_mask is not None and fixed_positions is not None and bool(fixed_mask.any()):
        out[fixed_mask] = fixed_positions[fixed_mask]

    return out


def overlap_pairs(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    num_hard: int,
    gap: float = 1e-4,
) -> torch.Tensor:
    """Vectorized overlap-pair detection on hard macros. Returns [P, 2] indices."""
    n = int(num_hard)
    if n <= 1:
        return torch.zeros(0, 2, dtype=torch.long, device=positions.device)

    pos = positions[:n]
    hard_sizes = sizes[:n]
    widths = hard_sizes[:, 0]
    heights = hard_sizes[:, 1]

    dx = torch.abs(pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0))
    dy = torch.abs(pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0))

    sep_x = (widths.unsqueeze(1) + widths.unsqueeze(0)) * 0.5 + gap
    sep_y = (heights.unsqueeze(1) + heights.unsqueeze(0)) * 0.5 + gap

    mask = (dx < sep_x) & (dy < sep_y)
    mask = torch.triu(mask, diagonal=1)
    return mask.nonzero(as_tuple=False)


def count_hard_overlaps(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    num_hard: int,
    gap: float = 1e-4,
) -> int:
    return int(overlap_pairs(positions, sizes, num_hard, gap=gap).shape[0])


def legalize_hard_macros(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    fixed_mask: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    num_hard: int,
    gap: float = 1e-4,
    max_iters: int = 80,
    max_pairs_per_iter: int = 8000,
) -> torch.Tensor:
    """Pairwise push-apart legalizer.

    Iteratively finds overlapping hard-macro pairs and pushes them apart
    along the cheaper axis. Up to `max_iters` outer iterations. Fixed
    macros never move; other macros take half the push.
    """
    out = positions.clone()
    n = int(num_hard)
    if n <= 1:
        return out

    hard_fixed = fixed_mask[:n].clone()
    hard_fixed_pos = out[:n].clone()

    out[:n] = project_to_canvas(
        out[:n],
        sizes[:n],
        canvas_width,
        canvas_height,
        fixed_mask=hard_fixed,
        fixed_positions=hard_fixed_pos,
    )

    for _ in range(int(max_iters)):
        pairs = overlap_pairs(out, sizes, n, gap=gap)
        if pairs.numel() == 0:
            return out

        moved = False
        pair_list = pairs.tolist()
        if len(pair_list) > max_pairs_per_iter:
            pair_list = pair_list[:max_pairs_per_iter]

        for i, j in pair_list:
            if hard_fixed[i] and hard_fixed[j]:
                continue

            xi = float(out[i, 0].item())
            yi = float(out[i, 1].item())
            xj = float(out[j, 0].item())
            yj = float(out[j, 1].item())
            wi = float(sizes[i, 0].item())
            hi = float(sizes[i, 1].item())
            wj = float(sizes[j, 0].item())
            hj = float(sizes[j, 1].item())

            overlap_x = (wi + wj) * 0.5 + gap - abs(xj - xi)
            overlap_y = (hi + hj) * 0.5 + gap - abs(yj - yi)
            if overlap_x <= 0.0 or overlap_y <= 0.0:
                continue

            # Choose the smaller-overlap axis and push along it.
            if overlap_x <= overlap_y:
                sign = 1.0 if xj >= xi else -1.0
                delta = overlap_x + 1e-6
                if not hard_fixed[i] and not hard_fixed[j]:
                    out[i, 0] -= 0.5 * sign * delta
                    out[j, 0] += 0.5 * sign * delta
                elif hard_fixed[i] and not hard_fixed[j]:
                    out[j, 0] += sign * delta
                elif not hard_fixed[i] and hard_fixed[j]:
                    out[i, 0] -= sign * delta
            else:
                sign = 1.0 if yj >= yi else -1.0
                delta = overlap_y + 1e-6
                if not hard_fixed[i] and not hard_fixed[j]:
                    out[i, 1] -= 0.5 * sign * delta
                    out[j, 1] += 0.5 * sign * delta
                elif hard_fixed[i] and not hard_fixed[j]:
                    out[j, 1] += sign * delta
                elif not hard_fixed[i] and hard_fixed[j]:
                    out[i, 1] -= sign * delta

            moved = True

        out[:n] = project_to_canvas(
            out[:n],
            sizes[:n],
            canvas_width,
            canvas_height,
            fixed_mask=hard_fixed,
            fixed_positions=hard_fixed_pos,
        )

        if not moved:
            break

    return out


def shelf_legalize_hard_macros(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    fixed_mask: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    num_hard: int,
    gap: float = 1e-4,
) -> torch.Tensor:
    """Deterministic fallback legalizer using shelf packing.

    The pairwise push-apart pass preserves the incoming placement better,
    but dense cases can get trapped against canvas boundaries. When that
    happens, this routine trades some wirelength for guaranteed legality by
    packing movable hard macros into non-overlapping horizontal shelves.
    """
    out = positions.clone()
    n = int(num_hard)
    if n <= 1:
        return out

    hard_fixed = fixed_mask[:n].clone()
    if bool(hard_fixed.any()):
        # Obstacle-aware shelf packing is a larger feature. Keep fixed-macro
        # cases on the local legalizer path rather than moving fixed cells.
        return out

    hard_sizes = sizes[:n]
    canvas_w = float(canvas_width)
    canvas_h = float(canvas_height)
    pack_gap = max(0.0, float(gap))

    def try_order(order: List[int]) -> Optional[torch.Tensor]:
        candidate = out.clone()
        x_cursor = 0.0
        y_cursor = 0.0
        row_height = 0.0

        for idx in order:
            width = float(hard_sizes[idx, 0].item())
            height = float(hard_sizes[idx, 1].item())
            if width > canvas_w + 1e-9 or height > canvas_h + 1e-9:
                return None

            if x_cursor > 0.0 and x_cursor + width > canvas_w + 1e-9:
                y_cursor += row_height + pack_gap
                x_cursor = 0.0
                row_height = 0.0

            if y_cursor + height > canvas_h + 1e-9:
                return None

            candidate[idx, 0] = x_cursor + 0.5 * width
            candidate[idx, 1] = y_cursor + 0.5 * height
            x_cursor += width + pack_gap
            row_height = max(row_height, height)

        return project_to_canvas(
            candidate,
            sizes,
            canvas_w,
            canvas_h,
            fixed_mask=fixed_mask,
            fixed_positions=positions,
        )

    indices = list(range(n))
    orders = [
        sorted(indices, key=lambda i: (-float(hard_sizes[i, 1].item()), -float(hard_sizes[i, 0].item()))),
        sorted(indices, key=lambda i: -float((hard_sizes[i, 0] * hard_sizes[i, 1]).item())),
        sorted(indices, key=lambda i: -float(hard_sizes[i, 0].item())),
        sorted(indices, key=lambda i: (float(positions[i, 1].item()), float(positions[i, 0].item()))),
    ]

    # Official validation treats touching edges as legal. Use gap=0.0 for
    # fallback acceptance so a tiny numerical clearance target does not
    # reject an otherwise valid emergency placement.
    eval_gap = 0.0
    best = out
    best_overlaps = count_hard_overlaps(best, sizes, n, gap=eval_gap)
    for order in orders:
        candidate = try_order(order)
        if candidate is None:
            continue
        overlaps = count_hard_overlaps(candidate, sizes, n, gap=eval_gap)
        if overlaps < best_overlaps:
            best = candidate
            best_overlaps = overlaps
        if overlaps == 0:
            return candidate

    return best


def _macro_overlaps_any(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    idx: int,
    num_hard: int,
    gap: float = 0.0,
) -> bool:
    """Return whether one hard macro overlaps any other hard macro."""
    n = int(num_hard)
    if n <= 1:
        return False
    pos = positions[:n]
    mpos = pos[idx]
    ms = sizes[idx]
    dx = torch.abs(pos[:, 0] - mpos[0])
    dy = torch.abs(pos[:, 1] - mpos[1])
    sep_x = (sizes[:n, 0] + ms[0]) * 0.5 + gap
    sep_y = (sizes[:n, 1] + ms[1]) * 0.5 + gap
    mask = (dx < sep_x) & (dy < sep_y)
    mask[idx] = False
    return bool(mask.any())


def repair_true_overlaps_by_relocation(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    fixed_mask: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    num_hard: int,
    max_iters: int = 12,
) -> torch.Tensor:
    """Repair remaining true overlaps by moving only the offending macros.

    This is a middle path between local pairwise push-apart and full shelf
    repacking. It preserves the nearly-good placement when only a handful of
    true overlaps remain after pairwise legalization.
    """
    out = positions.clone()
    n = int(num_hard)
    if n <= 1:
        return out

    hard_fixed = fixed_mask[:n].clone()
    canvas_w = float(canvas_width)
    canvas_h = float(canvas_height)
    span = max(canvas_w, canvas_h)
    eps = 1e-5 * max(span, 1.0)

    def clamp_center(idx: int, x: float, y: float) -> Tuple[float, float]:
        half_w = 0.5 * float(sizes[idx, 0].item())
        half_h = 0.5 * float(sizes[idx, 1].item())
        lo_x = half_w
        hi_x = canvas_w - half_w
        lo_y = half_h
        hi_y = canvas_h - half_h
        if hi_x < lo_x:
            x = 0.5 * (lo_x + hi_x)
        else:
            x = max(lo_x, min(hi_x, x))
        if hi_y < lo_y:
            y = 0.5 * (lo_y + hi_y)
        else:
            y = max(lo_y, min(hi_y, y))
        return x, y

    def try_place(idx: int, centers: Sequence[Tuple[float, float]]) -> bool:
        if bool(hard_fixed[idx]):
            return False
        cur = out[idx].clone()
        best_center: Optional[Tuple[float, float]] = None
        best_dist = float("inf")
        for x_raw, y_raw in centers:
            x, y = clamp_center(idx, float(x_raw), float(y_raw))
            out[idx, 0] = x
            out[idx, 1] = y
            if not _macro_overlaps_any(out, sizes, idx, n, gap=0.0):
                dist = float((cur[0].item() - x) ** 2 + (cur[1].item() - y) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_center = (x, y)
        out[idx] = cur
        if best_center is None:
            return False
        out[idx, 0] = best_center[0]
        out[idx, 1] = best_center[1]
        return True

    for _ in range(int(max_iters)):
        pairs = overlap_pairs(out, sizes, n, gap=0.0)
        if pairs.numel() == 0:
            return out

        moved = False
        for i, j in pairs.tolist():
            xi, yi = float(out[i, 0].item()), float(out[i, 1].item())
            xj, yj = float(out[j, 0].item()), float(out[j, 1].item())
            wi, hi = float(sizes[i, 0].item()), float(sizes[i, 1].item())
            wj, hj = float(sizes[j, 0].item()), float(sizes[j, 1].item())

            sign_x = 1.0 if xj >= xi else -1.0
            sign_y = 1.0 if yj >= yi else -1.0
            sep_x = 0.5 * (wi + wj) + eps
            sep_y = 0.5 * (hi + hj) + eps
            radius = max(wi, hi, wj, hj, 0.02 * span)

            candidates_i: List[Tuple[float, float]] = [
                (xj - sign_x * sep_x, yi),
                (xi, yj - sign_y * sep_y),
                (xj - sign_x * sep_x, yj - sign_y * sep_y),
            ]
            candidates_j: List[Tuple[float, float]] = [
                (xi + sign_x * sep_x, yj),
                (xj, yi + sign_y * sep_y),
                (xi + sign_x * sep_x, yi + sign_y * sep_y),
            ]

            for scale in (0.5, 1.0, 1.5, 2.5, 4.0):
                r = radius * scale
                for ax, ay in (
                    (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                    (0.707, 0.707), (0.707, -0.707), (-0.707, 0.707), (-0.707, -0.707),
                ):
                    candidates_i.append((xi + ax * r, yi + ay * r))
                    candidates_j.append((xj + ax * r, yj + ay * r))

            # Try the smaller/displaceable macro first to preserve large blocks.
            area_i = wi * hi
            area_j = wj * hj
            order = [(i, candidates_i), (j, candidates_j)]
            if area_j < area_i:
                order = [(j, candidates_j), (i, candidates_i)]
            for idx, centers in order:
                if try_place(idx, centers):
                    moved = True
                    break

        if not moved:
            break

    return out


def strict_legalize(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    fixed_mask: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    num_hard: int,
    gap: float = 1e-4,
) -> torch.Tensor:
    """Robust legalization: try increasing iteration budgets until clean."""
    out = positions.clone()
    n = int(num_hard)
    if n <= 1:
        return out

    for max_iters in (80, 160, 320):
        out = legalize_hard_macros(
            out,
            sizes,
            fixed_mask,
            canvas_width,
            canvas_height,
            n,
            gap=gap,
            max_iters=max_iters,
        )
        if count_hard_overlaps(out, sizes, n, gap=gap) == 0:
            return out

    # If we only missed the tiny optional clearance gap, the placement is
    # already legal by the official "touching edges are OK" rule.
    if count_hard_overlaps(out, sizes, n, gap=0.0) == 0:
        return out

    repaired = repair_true_overlaps_by_relocation(
        out,
        sizes,
        fixed_mask,
        canvas_width,
        canvas_height,
        n,
    )
    if count_hard_overlaps(repaired, sizes, n, gap=0.0) == 0:
        return repaired
    if count_hard_overlaps(repaired, sizes, n, gap=0.0) < count_hard_overlaps(out, sizes, n, gap=0.0):
        out = repaired

    packed = shelf_legalize_hard_macros(
        out,
        sizes,
        fixed_mask,
        canvas_width,
        canvas_height,
        n,
        gap=gap,
    )
    packed_overlaps = count_hard_overlaps(packed, sizes, n, gap=0.0)
    out_overlaps = count_hard_overlaps(out, sizes, n, gap=0.0)
    if out_overlaps > 0 and (packed_overlaps == 0 or packed_overlaps <= out_overlaps):
        return packed
    return out
