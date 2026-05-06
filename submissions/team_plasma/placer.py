"""
Team plasma placer submission.

Pipeline:
1. GS-inspired plasma global stage.
2. Deterministic hard-macro legalization.
3. Portfolio SA with synchronized elite replication (GWTW style).
4. Periodic/final soft-macro optimization via PlacementCost.
"""

from __future__ import annotations

import math
import random
import sys
import time
import importlib.util
import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import load_team_plasma_config
from plasma_core import compute_plasma_forces, gradient_from_potential, sample_vector_field

try:
    from macro_place.benchmark import Benchmark
except Exception:
    _bench_path = (Path(__file__).resolve().parents[1] / ".." / "macro_place" / "benchmark.py").resolve()
    _bench_spec = importlib.util.spec_from_file_location("macro_place_benchmark_fallback", str(_bench_path))
    if _bench_spec is None or _bench_spec.loader is None:
        raise
    _bench_mod = importlib.util.module_from_spec(_bench_spec)
    _bench_spec.loader.exec_module(_bench_mod)
    Benchmark = _bench_mod.Benchmark

try:
    from macro_place.loader import load_benchmark, load_benchmark_from_dir
except Exception:
    load_benchmark = None
    load_benchmark_from_dir = None

try:
    from macro_place.objective import compute_proxy_cost
except Exception:
    compute_proxy_cost = None

try:
    from macro_place.utils import validate_placement
except Exception:
    validate_placement = None


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def benchmark_soft_fill_ratio(benchmark: Benchmark) -> float:
    if int(benchmark.num_soft_macros) <= 0:
        return 0.0
    sizes = benchmark.macro_sizes.float()
    num_hard = int(benchmark.num_hard_macros)
    soft = sizes[num_hard:]
    soft_area = float((soft[:, 0] * soft[:, 1]).sum().item()) if int(soft.shape[0]) > 0 else 0.0
    canvas_area = max(float(benchmark.canvas_width) * float(benchmark.canvas_height), 1e-9)
    return soft_area / canvas_area


def _flat_quantile(field: torch.Tensor, q: float) -> float:
    if not isinstance(field, torch.Tensor) or field.numel() == 0:
        return 0.0
    flat = field.reshape(-1).float()
    return float(torch.quantile(flat, float(q)).item())


def compute_plasma_control_scales(fields: Dict[str, torch.Tensor], cfg: Dict[str, object]) -> Dict[str, float]:
    if not bool(cfg.get("enabled", False)):
        return {
            "q_rhs_scale": 1.0,
            "rho_rhs_scale": 1.0,
            "q_force_scale": 1.0,
            "rho_force_scale": 1.0,
            "hall_scale": 1.0,
            "dia_scale": 1.0,
            "balloon_scale": 1.0,
            "hot_scale": 1.0,
            "cold_scale": 1.0,
            "repulsion_scale": 1.0,
            "net_scale": 1.0,
            "anchor_scale": 1.0,
            "trust_scale": 1.0,
            "q_metric": 0.0,
            "rho_metric": 0.0,
            "q_rho_ratio": 0.0,
            "temp_split": 0.0,
            "q_dom": 0.0,
            "rho_dom": 0.0,
            "q_abs_dom": 0.0,
            "rho_abs_dom": 0.0,
        }

    q_field = fields.get("q_over", fields.get("q", torch.zeros(0)))
    rho_field = fields.get("rho", torch.zeros(0))
    q_metric = _flat_quantile(q_field, float(cfg.get("q_quantile", 0.90)))
    rho_metric = _flat_quantile(rho_field, float(cfg.get("rho_quantile", 0.90)))
    ratio = q_metric / max(rho_metric, 1e-6)
    ratio_ref = max(float(cfg.get("ratio_ref", 1.35)), 1e-6)
    q_dom = max(0.0, ratio / ratio_ref - 1.0)
    rho_dom = max(0.0, ratio_ref / max(ratio, 1e-6) - 1.0)
    q_abs_ref = max(float(cfg.get("q_abs_ref", q_metric if q_metric > 0.0 else 1.0)), 1e-6)
    rho_abs_ref = max(float(cfg.get("rho_abs_ref", rho_metric if rho_metric > 0.0 else 1.0)), 1e-6)
    q_abs_dom = max(0.0, q_metric / q_abs_ref - 1.0)
    rho_abs_dom = max(0.0, rho_metric / rho_abs_ref - 1.0)

    hot_force = fields.get("hot_force", torch.zeros(0))
    cold_force = fields.get("cold_force", torch.zeros(0))
    temp_split = 0.0
    if isinstance(hot_force, torch.Tensor) and isinstance(cold_force, torch.Tensor):
        if hot_force.numel() > 0 and hot_force.shape == cold_force.shape:
            temp_split = float((hot_force.float() - cold_force.float()).norm(dim=1).mean().item())
    temp_ref = max(float(cfg.get("temp_split_ref", 0.25)), 1e-6)
    temp_dom = max(0.0, temp_split / temp_ref - 1.0)
    cap = max(1.0, float(cfg.get("cap", 2.5)))

    def _grow(alpha: float, dom: float) -> float:
        return min(cap, 1.0 + max(0.0, float(alpha)) * max(0.0, dom))

    def _shrink(beta: float, dom: float) -> float:
        return 1.0 / min(cap, 1.0 + max(0.0, float(beta)) * max(0.0, dom))

    q_dom_total = (
        q_dom
        + float(cfg.get("q_abs_mix", 0.0)) * q_abs_dom
        + float(cfg.get("temp_q_mix", 0.0)) * temp_dom
    )
    rho_dom_total = (
        rho_dom
        + float(cfg.get("rho_abs_mix", 0.0)) * rho_abs_dom
        + float(cfg.get("temp_rho_mix", 0.0)) * temp_dom
    )
    return {
        "q_rhs_scale": _grow(float(cfg.get("q_rhs_alpha", 0.0)), q_dom + float(cfg.get("q_abs_mix", 0.0)) * q_abs_dom),
        "rho_rhs_scale": _grow(float(cfg.get("rho_rhs_alpha", 0.0)), rho_dom + float(cfg.get("rho_abs_mix", 0.0)) * rho_abs_dom),
        "q_force_scale": _grow(float(cfg.get("q_force_alpha", 0.0)), q_dom_total),
        "rho_force_scale": _grow(float(cfg.get("rho_force_alpha", 0.0)), rho_dom_total),
        "hall_scale": _grow(float(cfg.get("hall_alpha", 0.0)), q_dom_total),
        "dia_scale": _grow(float(cfg.get("dia_alpha", 0.0)), q_dom_total),
        "balloon_scale": _grow(float(cfg.get("balloon_alpha", 0.0)), q_dom_total),
        "hot_scale": _grow(float(cfg.get("hot_alpha", 0.0)), q_dom + temp_dom),
        "cold_scale": _grow(float(cfg.get("cold_alpha", 0.0)), rho_dom_total + temp_dom),
        "repulsion_scale": _grow(float(cfg.get("repulsion_alpha", 0.0)), max(q_dom_total, rho_dom_total)),
        "net_scale": _shrink(float(cfg.get("net_down_beta", 0.0)), q_dom_total),
        "anchor_scale": _shrink(float(cfg.get("anchor_down_beta", 0.0)), q_dom_total),
        "trust_scale": _grow(float(cfg.get("trust_up_gamma", 0.0)), q_dom_total),
        "q_metric": q_metric,
        "rho_metric": rho_metric,
        "q_rho_ratio": ratio,
        "temp_split": temp_split,
        "q_dom": q_dom,
        "rho_dom": rho_dom,
        "q_abs_dom": q_abs_dom,
        "rho_abs_dom": rho_abs_dom,
    }


def compute_soft_fill_scale(
    soft_fill_ratio: float,
    cfg: Dict,
    disabled_value: float = 0.0,
) -> float:
    if not bool(cfg.get("enabled", False)):
        return float(disabled_value)

    min_soft_fill = float(cfg.get("min_soft_fill", 0.45))
    ref_soft_fill = max(float(cfg.get("ref_soft_fill", 0.75)), min_soft_fill + 1e-6)
    power = max(float(cfg.get("power", 1.0)), 1e-6)
    floor = float(cfg.get("floor", 0.0))

    if soft_fill_ratio <= min_soft_fill:
        return max(0.0, min(1.0, floor))

    scale = (soft_fill_ratio - min_soft_fill) / (ref_soft_fill - min_soft_fill)
    scale = max(0.0, min(1.0, scale))
    scale = scale ** power
    scale = floor + (1.0 - floor) * scale
    return max(0.0, min(1.0, scale))


def compute_anchor_confidence_scale(anchor_weights: torch.Tensor, cfg: Dict) -> torch.Tensor:
    if anchor_weights.numel() == 0:
        return anchor_weights.float()

    weights = torch.clamp(anchor_weights.float(), min=0.0)
    mask = weights > 1e-8
    if not bool(mask.any()):
        return torch.zeros_like(weights)

    q = float(cfg.get("confidence_quantile", 0.90))
    q = max(0.50, min(0.99, q))
    ref = torch.quantile(weights[mask], q)
    ref_val = max(float(ref.item()), 1e-6)

    scaled = weights / ref_val
    power = float(cfg.get("confidence_power", 0.50))
    if power != 1.0:
        scaled = torch.pow(torch.clamp(scaled, min=0.0), power)

    min_scale = float(cfg.get("confidence_min", 0.35))
    max_scale = float(cfg.get("confidence_max", 1.75))
    scaled = torch.clamp(scaled, min=min_scale, max=max_scale)
    return torch.where(mask, scaled, torch.zeros_like(scaled))


def compute_anchor_alignment_gate(
    anchor_force: torch.Tensor,
    plasma_force: torch.Tensor,
    q_force: torch.Tensor,
    rho_force: torch.Tensor,
    repulsion_force: torch.Tensor,
    cfg: Dict,
) -> torch.Tensor:
    align_cfg = dict(cfg.get("alignment", {}))
    n = int(anchor_force.shape[0])
    if n <= 0:
        return torch.zeros(0, dtype=torch.float32, device=anchor_force.device)
    if not bool(align_cfg.get("enabled", False)):
        return torch.ones(n, dtype=torch.float32, device=anchor_force.device)

    guide_force = (
        float(align_cfg.get("plasma_weight", 1.0)) * plasma_force
        + float(align_cfg.get("q_weight", 0.8)) * q_force
        + float(align_cfg.get("rho_weight", 0.5)) * rho_force
        + float(align_cfg.get("repulsion_weight", 0.7)) * repulsion_force
    )
    anum = torch.linalg.norm(anchor_force, dim=1)
    gnum = torch.linalg.norm(guide_force, dim=1)
    denom = torch.clamp(anum * gnum, min=1e-9)
    cosine = torch.sum(anchor_force * guide_force, dim=1) / denom
    cosine = torch.clamp(cosine, min=-1.0, max=1.0)

    threshold = float(align_cfg.get("threshold", 0.0))
    threshold = max(-0.95, min(0.95, threshold))
    gate = (cosine - threshold) / max(1.0 - threshold, 1e-6)
    gate = torch.clamp(gate, min=0.0, max=1.0)

    power = max(float(align_cfg.get("power", 1.0)), 1e-6)
    if power != 1.0:
        gate = torch.pow(gate, power)

    min_gate = float(align_cfg.get("min_gate", 0.0))
    max_gate = float(align_cfg.get("max_gate", 1.0))
    gate = torch.clamp(gate, min=min_gate, max=max_gate)
    return gate


def resample_scalar_field(field: Optional[torch.Tensor], target_size: int) -> Optional[torch.Tensor]:
    if field is None:
        return None
    if int(field.shape[0]) == target_size and int(field.shape[1]) == target_size:
        return field
    src = field.unsqueeze(0).unsqueeze(0)
    out = F.interpolate(src, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return out[0, 0]


def _smooth_l1(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(x * x + eps)


def _normalize_move_force(force: torch.Tensor) -> torch.Tensor:
    n = int(force.shape[0])
    if n <= 0:
        return force
    norm = torch.linalg.norm(force, dim=1)
    denom = torch.quantile(norm, 0.90) if n >= 10 else torch.max(norm)
    d = float(denom.item()) if norm.numel() > 0 else 0.0
    if d > 1e-6:
        force = force / d
    return force


def compute_transport_preconditioner(
    sizes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    cfg: Dict,
) -> torch.Tensor:
    """
    Mixed-size mobility preconditioner inspired by ePlace-MS.

    Larger macros and higher-degree macros get reduced mobility so they do not
    dominate the optimization with oscillatory boundary-to-boundary moves.
    """
    n = int(sizes.shape[0])
    device = sizes.device
    if n <= 0:
        return torch.zeros((0, 1), dtype=torch.float32, device=device)

    if not bool(cfg.get("enabled", False)):
        return torch.ones((n, 1), dtype=torch.float32, device=device)

    areas = torch.clamp(sizes[:, 0] * sizes[:, 1], min=1e-9)
    aq = max(0.50, min(0.99, float(cfg.get("area_quantile", 0.60))))
    area_ref = torch.quantile(areas, aq) if n > 1 else areas[0]
    area_ref_val = max(float(area_ref.item()), 1e-9)
    area_power = max(float(cfg.get("area_power", 0.5)), 1e-6)
    area_term = torch.pow(areas / area_ref_val, area_power)

    degree = torch.zeros(n, dtype=torch.float32, device=device)
    if edge_index.numel() > 0 and int(edge_index.shape[0]) > 0:
        src = edge_index[:, 0].long()
        dst = edge_index[:, 1].long()
        weight = edge_weight.float() if edge_weight.numel() else torch.ones(int(edge_index.shape[0]), dtype=torch.float32, device=device)
        degree.scatter_add_(0, src, weight)
        degree.scatter_add_(0, dst, weight)
    degree = torch.clamp(degree, min=float(cfg.get("degree_floor", 1e-3)))
    if n > 1 and bool((degree > 0).any()):
        dq = max(0.50, min(0.99, float(cfg.get("degree_quantile", 0.70))))
        degree_ref = torch.quantile(degree, dq)
    else:
        degree_ref = degree[0]
    degree_ref_val = max(float(degree_ref.item()), 1e-6)
    degree_power = max(float(cfg.get("degree_power", 0.5)), 1e-6)
    degree_term = torch.pow(degree / degree_ref_val, degree_power)

    base = float(cfg.get("base", 1.0))
    area_alpha = float(cfg.get("area_alpha", 0.0))
    degree_alpha = float(cfg.get("degree_alpha", 0.0))
    precond = base + area_alpha * area_term + degree_alpha * degree_term

    min_val = max(float(cfg.get("min", 0.25)), 1e-6)
    max_val = max(float(cfg.get("max", 8.0)), min_val)
    precond = torch.clamp(precond, min=min_val, max=max_val)
    return precond.unsqueeze(1)


def apply_transport_barrier(
    transport_force: torch.Tensor,
    plasma_force: torch.Tensor,
    q_samples: torch.Tensor,
    rho_samples: torch.Tensor,
    cfg: Dict,
) -> torch.Tensor:
    """
    Plasma-native transport barrier inspired by E x B shear suppression.

    When macros sit in strong congestion/density pressure regions, suppress
    cross-field drift (normal to psi contours) more than tangential motion.
    Repulsion and anchor terms are intentionally handled outside this helper
    so legality repair and equilibrium preservation still work.
    """
    if not bool(cfg.get("enabled", False)):
        return transport_force
    n = int(transport_force.shape[0])
    if n <= 0:
        return transport_force

    normal = plasma_force.float()
    normal_norm = torch.linalg.norm(normal, dim=1, keepdim=True)
    valid = normal_norm > 1e-6
    safe_norm = torch.where(valid, normal_norm, torch.ones_like(normal_norm))
    n_hat = normal / safe_norm

    q_gate = torch.zeros((n, 1), dtype=torch.float32, device=transport_force.device)
    rho_gate = torch.zeros((n, 1), dtype=torch.float32, device=transport_force.device)

    if q_samples.numel() == n:
        q_q = max(0.50, min(0.99, float(cfg.get("q_quantile", 0.80))))
        q_ref = torch.quantile(q_samples.float(), q_q)
        q_scale = max(float(q_ref.item()) * float(cfg.get("q_scale_frac", 0.20)), 1e-6)
        q_gate = torch.sigmoid((q_samples.float().unsqueeze(1) - q_ref) / q_scale)

    if rho_samples.numel() == n:
        rho_q = max(0.50, min(0.99, float(cfg.get("rho_quantile", 0.85))))
        rho_ref = torch.quantile(rho_samples.float(), rho_q)
        rho_scale = max(float(rho_ref.item()) * float(cfg.get("rho_scale_frac", 0.20)), 1e-6)
        rho_gate = torch.sigmoid((rho_samples.float().unsqueeze(1) - rho_ref) / rho_scale)

    activity = (
        float(cfg.get("q_weight", 0.75)) * q_gate
        + float(cfg.get("rho_weight", 0.25)) * rho_gate
    )

    if n > 1:
        grad_norm = normal_norm.squeeze(1)
        g_q = max(0.50, min(0.99, float(cfg.get("grad_quantile", 0.65))))
        grad_ref = torch.quantile(grad_norm, g_q)
        grad_scale = max(float(grad_ref.item()) * float(cfg.get("grad_scale_frac", 0.25)), 1e-6)
        grad_gate = torch.sigmoid((grad_norm.unsqueeze(1) - grad_ref) / grad_scale)
        activity = activity * grad_gate

    activity = torch.clamp(activity, min=0.0, max=float(cfg.get("activity_cap", 1.5)))

    tangential = transport_force - torch.sum(transport_force * n_hat, dim=1, keepdim=True) * n_hat
    normal_comp = transport_force - tangential

    normal_scale = 1.0 / (
        1.0
        + max(float(cfg.get("normal_alpha", 0.0)), 0.0) * activity
    )
    tangent_scale = 1.0 + max(float(cfg.get("tangent_beta", 0.0)), 0.0) * activity
    min_normal = max(0.0, float(cfg.get("min_normal_scale", 0.15)))
    max_tangent = max(1.0, float(cfg.get("max_tangent_scale", 2.0)))
    normal_scale = torch.clamp(normal_scale, min=min_normal, max=1.0)
    tangent_scale = torch.clamp(tangent_scale, min=1.0, max=max_tangent)

    out = normal_scale * normal_comp + tangent_scale * tangential
    out = torch.where(valid.expand_as(out), out, transport_force)
    return out


def project_to_canvas(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    fixed_mask: Optional[torch.Tensor] = None,
    fixed_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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

    if fixed_mask is not None and fixed_positions is not None and fixed_mask.any():
        out[fixed_mask] = fixed_positions[fixed_mask]

    return out


def sanitize_canvas_bounds(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    fixed_mask: Optional[torch.Tensor] = None,
    fixed_positions: Optional[torch.Tensor] = None,
    safety_eps: float = 1e-6,
) -> torch.Tensor:
    """Clamp macro centers to valid box with a tiny inward margin when available."""
    out = positions.clone()
    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5

    base_x_min = half_w
    base_x_max = torch.full_like(half_w, float(canvas_width)) - half_w
    base_y_min = half_h
    base_y_max = torch.full_like(half_h, float(canvas_height)) - half_h

    eps = max(float(safety_eps), 0.0)
    if eps > 0.0:
        x_slack = base_x_max - base_x_min
        y_slack = base_y_max - base_y_min
        x_use_margin = x_slack > (2.0 * eps)
        y_use_margin = y_slack > (2.0 * eps)

        x_min = torch.where(x_use_margin, base_x_min + eps, base_x_min)
        x_max = torch.where(x_use_margin, base_x_max - eps, base_x_max)
        y_min = torch.where(y_use_margin, base_y_min + eps, base_y_min)
        y_max = torch.where(y_use_margin, base_y_max - eps, base_y_max)
    else:
        x_min = base_x_min
        x_max = base_x_max
        y_min = base_y_min
        y_max = base_y_max

    x_mid = 0.5 * (base_x_min + base_x_max)
    y_mid = 0.5 * (base_y_min + base_y_max)

    out_x = torch.max(torch.min(out[:, 0], x_max), x_min)
    out_y = torch.max(torch.min(out[:, 1], y_max), y_min)

    out[:, 0] = torch.where(base_x_max >= base_x_min, out_x, x_mid)
    out[:, 1] = torch.where(base_y_max >= base_y_min, out_y, y_mid)

    if fixed_mask is not None and fixed_positions is not None and fixed_mask.any():
        out[fixed_mask] = fixed_positions[fixed_mask]

    return out


def overlap_pairs(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    num_hard_macros: int,
    gap: float = 1e-4,
) -> torch.Tensor:
    n = int(num_hard_macros)
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
    num_hard_macros: int,
    gap: float = 1e-4,
) -> int:
    return int(overlap_pairs(positions, sizes, num_hard_macros, gap=gap).shape[0])


def _clamp_single_center_to_canvas(
    xy: torch.Tensor,
    size: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
) -> torch.Tensor:
    out = xy.clone()
    half_w = float(size[0].item()) * 0.5
    half_h = float(size[1].item()) * 0.5
    out[0] = min(max(float(out[0].item()), half_w), float(canvas_width) - half_w)
    out[1] = min(max(float(out[1].item()), half_h), float(canvas_height) - half_h)
    return out


def _has_overlap_for_indices(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    num_hard_macros: int,
    moved_indices: Sequence[int],
    gap: float = 1e-4,
) -> bool:
    if not moved_indices:
        return False

    n = int(num_hard_macros)
    if n <= 1:
        return False

    pos = positions[:n]
    hard_sizes = sizes[:n]
    widths = hard_sizes[:, 0]
    heights = hard_sizes[:, 1]

    for idx in moved_indices:
        i = int(idx)
        if i < 0 or i >= n:
            continue
        dx = torch.abs(pos[i, 0] - pos[:, 0])
        dy = torch.abs(pos[i, 1] - pos[:, 1])
        sep_x = (widths[i] + widths) * 0.5 + gap
        sep_y = (heights[i] + heights) * 0.5 + gap
        ov = (dx < sep_x) & (dy < sep_y)
        ov[i] = False
        if bool(ov.any()):
            return True

    return False


def _build_overlap_components(pairs: torch.Tensor, n: int) -> List[List[int]]:
    graph: List[List[int]] = [[] for _ in range(n)]
    for i, j in pairs.tolist():
        graph[i].append(j)
        graph[j].append(i)

    seen = [False] * n
    comps: List[List[int]] = []

    for start in range(n):
        if seen[start] or not graph[start]:
            continue
        stack = [start]
        seen[start] = True
        comp: List[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in graph[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        if len(comp) > 1:
            comps.append(sorted(comp))

    return comps


def _repack_component(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    fixed_mask: torch.Tensor,
    component: Sequence[int],
    canvas_width: float,
    canvas_height: float,
    anchor_positions: Optional[torch.Tensor] = None,
) -> None:
    comp = list(component)
    movable = [i for i in comp if not bool(fixed_mask[i])]
    if len(movable) <= 1:
        return

    anchor_x = float(positions[comp, 0].mean().item())
    anchor_y = float(positions[comp, 1].mean().item())

    if anchor_positions is not None:
        anchor_xy = anchor_positions[movable]
        span_x = float(anchor_xy[:, 0].max().item() - anchor_xy[:, 0].min().item())
        span_y = float(anchor_xy[:, 1].max().item() - anchor_xy[:, 1].min().item())
        major_axis = 0 if span_x >= span_y else 1
        ordered = sorted(
            movable,
            key=lambda i: (
                float(anchor_positions[i, major_axis].item()),
                float(anchor_positions[i, 1 - major_axis].item()),
                i,
            ),
        )
    else:
        areas = [(i, float((sizes[i, 0] * sizes[i, 1]).item())) for i in movable]
        areas.sort(key=lambda x: (-x[1], x[0]))
        ordered = [i for i, _ in areas]

    total_area = sum(float((sizes[i, 0] * sizes[i, 1]).item()) for i in ordered)
    row_limit = max(
        max(float(sizes[i, 0].item()) for i in ordered),
        math.sqrt(max(total_area, 1e-6)) * 1.25,
    )

    cursor_x = 0.0
    cursor_y = 0.0
    row_h = 0.0

    local_xy: Dict[int, Tuple[float, float]] = {}
    for idx in ordered:
        w = float(sizes[idx, 0].item())
        h = float(sizes[idx, 1].item())

        if cursor_x + w > row_limit and cursor_x > 0.0:
            cursor_x = 0.0
            cursor_y += row_h + 1e-4
            row_h = 0.0

        local_xy[idx] = (cursor_x + 0.5 * w, cursor_y + 0.5 * h)
        cursor_x += w + 1e-4
        row_h = max(row_h, h)

    block_w = max((x + float(sizes[i, 0].item()) * 0.5) for i, (x, _) in local_xy.items())
    block_h = max((y + float(sizes[i, 1].item()) * 0.5) for i, (_, y) in local_xy.items())

    base_x = anchor_x - 0.5 * block_w
    base_y = anchor_y - 0.5 * block_h

    for idx, (lx, ly) in local_xy.items():
        positions[idx, 0] = float(base_x + lx)
        positions[idx, 1] = float(base_y + ly)

    positions[:] = project_to_canvas(
        positions,
        sizes,
        canvas_width,
        canvas_height,
        fixed_mask=fixed_mask,
        fixed_positions=positions.clone(),
    )

def legalize_hard_macros(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    fixed_mask: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    num_hard_macros: int,
    gap: float = 1e-4,
    max_iters: int = 80,
    fallback_iters: int = 60,
    max_pairs_per_iter: int = 8000,
    anchor_positions: Optional[torch.Tensor] = None,
    anchor_strength: float = 0.0,
    restore_iters: int = 0,
) -> torch.Tensor:
    out = positions.clone()
    n = int(num_hard_macros)
    if n <= 1:
        return out

    hard_fixed = fixed_mask[:n].clone()
    hard_fixed_pos = out[:n].clone()
    hard_anchors = None
    use_anchor = (anchor_positions is not None) and (float(anchor_strength) > 0.0 or int(restore_iters) > 0)
    if use_anchor:
        hard_anchors = anchor_positions[:n].clone().float()

    out[:n] = project_to_canvas(
        out[:n],
        sizes[:n],
        canvas_width,
        canvas_height,
        fixed_mask=hard_fixed,
        fixed_positions=hard_fixed_pos,
    )

    for _ in range(max_iters):
        pairs = overlap_pairs(out, sizes, n, gap=gap)
        if pairs.numel() == 0:
            return out

        moved_any = False
        pair_list = pairs.tolist()
        if len(pair_list) > max_pairs_per_iter:
            pair_list = pair_list[:max_pairs_per_iter]
        for i, j in pair_list:
            if hard_fixed[i] and hard_fixed[j]:
                continue

            xi, yi = float(out[i, 0].item()), float(out[i, 1].item())
            xj, yj = float(out[j, 0].item()), float(out[j, 1].item())

            wi, hi = float(sizes[i, 0].item()), float(sizes[i, 1].item())
            wj, hj = float(sizes[j, 0].item()), float(sizes[j, 1].item())

            dx = xj - xi
            dy = yj - yi
            overlap_x = (wi + wj) * 0.5 + gap - abs(dx)
            overlap_y = (hi + hj) * 0.5 + gap - abs(dy)

            if overlap_x <= 0.0 or overlap_y <= 0.0:
                continue

            best_choice = None
            if hard_anchors is not None and float(anchor_strength) > 0.0:
                old_i = out[i].clone()
                old_j = out[j].clone()
                area_i = max(1e-6, float((sizes[i, 0] * sizes[i, 1]).item()))
                area_j = max(1e-6, float((sizes[j, 0] * sizes[j, 1]).item()))
                old_anchor_cost = 0.0
                if not hard_fixed[i]:
                    old_anchor_cost += area_i * float(((old_i - hard_anchors[i]) ** 2).sum().item())
                if not hard_fixed[j]:
                    old_anchor_cost += area_j * float(((old_j - hard_anchors[j]) ** 2).sum().item())

                for axis, overlap_amt, raw_sign in (
                    (0, overlap_x, 1.0 if dx >= 0.0 else -1.0),
                    (1, overlap_y, 1.0 if dy >= 0.0 else -1.0),
                ):
                    delta = overlap_amt + 1e-6
                    if not hard_fixed[i] and not hard_fixed[j]:
                        move_i = -0.5 * raw_sign * delta
                        move_j = 0.5 * raw_sign * delta
                    elif hard_fixed[i] and not hard_fixed[j]:
                        move_i = 0.0
                        move_j = raw_sign * delta
                    elif not hard_fixed[i] and hard_fixed[j]:
                        move_i = -raw_sign * delta
                        move_j = 0.0
                    else:
                        move_i = 0.0
                        move_j = 0.0

                    new_i = old_i.clone()
                    new_j = old_j.clone()
                    new_i[axis] += float(move_i)
                    new_j[axis] += float(move_j)
                    new_i = _clamp_single_center_to_canvas(new_i, sizes[i], canvas_width, canvas_height)
                    new_j = _clamp_single_center_to_canvas(new_j, sizes[j], canvas_width, canvas_height)

                    new_anchor_cost = 0.0
                    if not hard_fixed[i]:
                        new_anchor_cost += area_i * float(((new_i - hard_anchors[i]) ** 2).sum().item())
                    if not hard_fixed[j]:
                        new_anchor_cost += area_j * float(((new_j - hard_anchors[j]) ** 2).sum().item())
                    score = float(overlap_amt) + float(anchor_strength) * max(0.0, new_anchor_cost - old_anchor_cost)
                    choice = (score, axis, raw_sign, delta, move_i, move_j)
                    if best_choice is None or choice < best_choice:
                        best_choice = choice

            if best_choice is None:
                if overlap_x <= overlap_y:
                    axis = 0
                    sign = 1.0 if dx >= 0.0 else -1.0
                    delta = overlap_x + 1e-6
                else:
                    axis = 1
                    sign = 1.0 if dy >= 0.0 else -1.0
                    delta = overlap_y + 1e-6

                if not hard_fixed[i] and not hard_fixed[j]:
                    move_i = -0.5 * sign * delta
                    move_j = 0.5 * sign * delta
                elif hard_fixed[i] and not hard_fixed[j]:
                    move_i = 0.0
                    move_j = sign * delta
                elif not hard_fixed[i] and hard_fixed[j]:
                    move_i = -sign * delta
                    move_j = 0.0
                else:
                    move_i = 0.0
                    move_j = 0.0
            else:
                _, axis, sign, delta, move_i, move_j = best_choice

            if axis == 0:
                out[i, 0] += float(move_i)
                out[j, 0] += float(move_j)
            else:
                out[i, 1] += float(move_i)
                out[j, 1] += float(move_j)

            moved_any = True

        out[:n] = project_to_canvas(
            out[:n],
            sizes[:n],
            canvas_width,
            canvas_height,
            fixed_mask=hard_fixed,
            fixed_positions=hard_fixed_pos,
        )

        if not moved_any:
            break

    pairs = overlap_pairs(out, sizes, n, gap=gap)
    if pairs.numel() > 0:
        for comp in _build_overlap_components(pairs, n):
            _repack_component(
                out[:n],
                sizes[:n],
                hard_fixed,
                comp,
                canvas_width,
                canvas_height,
                anchor_positions=hard_anchors,
            )

        for _ in range(fallback_iters):
            pairs = overlap_pairs(out, sizes, n, gap=gap)
            if pairs.numel() == 0:
                break

            pair_list = pairs.tolist()
            if len(pair_list) > max_pairs_per_iter:
                pair_list = pair_list[:max_pairs_per_iter]

            for i, j in pair_list:
                if hard_fixed[i] and hard_fixed[j]:
                    continue

                xi, yi = float(out[i, 0].item()), float(out[i, 1].item())
                xj, yj = float(out[j, 0].item()), float(out[j, 1].item())
                wi, hi = float(sizes[i, 0].item()), float(sizes[i, 1].item())
                wj, hj = float(sizes[j, 0].item()), float(sizes[j, 1].item())

                overlap_x = (wi + wj) * 0.5 + gap - abs(xj - xi)
                overlap_y = (hi + hj) * 0.5 + gap - abs(yj - yi)
                if overlap_x <= 0.0 or overlap_y <= 0.0:
                    continue

                if overlap_x <= overlap_y:
                    sign = 1.0 if xj >= xi else -1.0
                    if not hard_fixed[i]:
                        out[i, 0] -= sign * overlap_x
                    if not hard_fixed[j]:
                        out[j, 0] += sign * overlap_x
                else:
                    sign = 1.0 if yj >= yi else -1.0
                    if not hard_fixed[i]:
                        out[i, 1] -= sign * overlap_y
                    if not hard_fixed[j]:
                        out[j, 1] += sign * overlap_y

            out[:n] = project_to_canvas(
                out[:n],
                sizes[:n],
                canvas_width,
                canvas_height,
                fixed_mask=hard_fixed,
                fixed_positions=hard_fixed_pos,
            )

    if hard_anchors is not None and int(restore_iters) > 0:
        moved = []
        for i in range(n):
            if bool(hard_fixed[i]):
                continue
            dist = float(torch.norm(out[i] - hard_anchors[i], p=2).item())
            if dist > 1e-5:
                moved.append((dist, i))

        moved.sort(reverse=True)
        line_steps = (1.0, 0.75, 0.5, 0.25)
        axis_steps = (1.0, 0.5, 0.25)

        for _ in range(int(restore_iters)):
            restored_any = False
            for _, i in moved:
                current = out[i].clone()
                anchor = hard_anchors[i].clone()
                axis_candidates: List[torch.Tensor] = []
                for frac in line_steps:
                    axis_candidates.append(current + float(frac) * (anchor - current))
                for frac in axis_steps:
                    cx = current.clone()
                    cx[0] = current[0] + float(frac) * (anchor[0] - current[0])
                    axis_candidates.append(cx)
                    cy = current.clone()
                    cy[1] = current[1] + float(frac) * (anchor[1] - current[1])
                    axis_candidates.append(cy)

                best_pos = current
                best_dist = float(torch.norm(current - anchor, p=2).item())
                seen = set()
                for cand in axis_candidates:
                    cand = _clamp_single_center_to_canvas(cand, sizes[i], canvas_width, canvas_height)
                    key = (round(float(cand[0].item()), 6), round(float(cand[1].item()), 6))
                    if key in seen:
                        continue
                    seen.add(key)
                    out[i] = cand
                    if not _has_overlap_for_indices(out, sizes, n, [i], gap=gap):
                        cand_dist = float(torch.norm(out[i] - anchor, p=2).item())
                        if cand_dist + 1e-9 < best_dist:
                            best_pos = out[i].clone()
                            best_dist = cand_dist
                    out[i] = current

                if best_dist + 1e-9 < float(torch.norm(current - anchor, p=2).item()):
                    out[i] = best_pos
                    restored_any = True

            if not restored_any:
                break

        out[:n] = sanitize_canvas_bounds(
            out[:n],
            sizes[:n],
            canvas_width,
            canvas_height,
            fixed_mask=hard_fixed,
            fixed_positions=hard_fixed_pos,
            safety_eps=1e-6,
        )

    return out


def strict_legalize_hard_macros(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    fixed_mask: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    num_hard_macros: int,
    gap: float = 1e-4,
    max_iters: int = 80,
    fallback_iters: int = 60,
    max_pairs_per_iter: int = 8000,
    anchor_positions: Optional[torch.Tensor] = None,
    anchor_strength: float = 0.0,
    restore_iters: int = 0,
) -> torch.Tensor:
    """Robust hard-macro legalization with extra safety passes."""
    out = positions.clone()
    n = int(num_hard_macros)
    if n <= 1:
        return out

    hard_fixed = fixed_mask[:n].clone()
    hard_fixed_pos = out[:n].clone()

    for scale in (1.0, 1.6):
        out[:n] = sanitize_canvas_bounds(
            out[:n],
            sizes[:n],
            canvas_width,
            canvas_height,
            fixed_mask=hard_fixed,
            fixed_positions=hard_fixed_pos,
            safety_eps=1e-6,
        )

        out = legalize_hard_macros(
            out,
            sizes,
            fixed_mask,
            canvas_width,
            canvas_height,
            n,
            gap=gap,
            max_iters=max(20, int(round(max_iters * scale))),
            fallback_iters=max(20, int(round(fallback_iters * scale))),
            anchor_positions=anchor_positions,
            anchor_strength=anchor_strength,
            restore_iters=restore_iters,
        )

        out[:n] = sanitize_canvas_bounds(
            out[:n],
            sizes[:n],
            canvas_width,
            canvas_height,
            fixed_mask=hard_fixed,
            fixed_positions=hard_fixed_pos,
            safety_eps=1e-6,
        )

        if count_hard_overlaps(out, sizes, n, gap=gap) == 0:
            return out

    for _ in range(80):
        pairs = overlap_pairs(out, sizes, n, gap=gap)
        if pairs.numel() == 0:
            break

        moved_any = False
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

            moved_any = True

        out[:n] = sanitize_canvas_bounds(
            out[:n],
            sizes[:n],
            canvas_width,
            canvas_height,
            fixed_mask=hard_fixed,
            fixed_positions=hard_fixed_pos,
            safety_eps=1e-6,
        )

        if not moved_any:
            break

    return out


def legalize_hard_vs_all(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    fixed_mask: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    num_hard_macros: int,
    gap: float = 1e-4,
    max_iters: int = 80,
) -> torch.Tensor:
    """
    Resolve overlaps between movable hard macros and all other macros
    (including soft/fixed macros) by moving only hard macros.
    """
    out = positions.clone()
    n_h = int(num_hard_macros)
    n_all = int(out.shape[0])
    if n_h <= 0 or n_all <= 1:
        return out

    hard_fixed = fixed_mask[:n_h].clone()
    hard_fixed_pos = out[:n_h].clone()

    for _ in range(max_iters):
        moved_any = False
        for i in range(n_h):
            if bool(hard_fixed[i]):
                continue

            xi = float(out[i, 0].item())
            yi = float(out[i, 1].item())
            wi = float(sizes[i, 0].item())
            hi = float(sizes[i, 1].item())

            dx = out[:, 0] - xi
            dy = out[:, 1] - yi
            sep_x = 0.5 * (wi + sizes[:, 0]) + float(gap)
            sep_y = 0.5 * (hi + sizes[:, 1]) + float(gap)

            ovx = sep_x - torch.abs(dx)
            ovy = sep_y - torch.abs(dy)
            mask = (ovx > 0.0) & (ovy > 0.0)
            mask[i] = False

            if not bool(mask.any()):
                continue

            # Prefer resolving the largest penetration first.
            pen = torch.minimum(torch.clamp(ovx, min=0.0), torch.clamp(ovy, min=0.0))
            pen = torch.where(mask, pen, torch.zeros_like(pen))
            j = int(torch.argmax(pen).item())

            xj = float(out[j, 0].item())
            yj = float(out[j, 1].item())
            wj = float(sizes[j, 0].item())
            hj = float(sizes[j, 1].item())

            overlap_x = (wi + wj) * 0.5 + gap - abs(xj - xi)
            overlap_y = (hi + hj) * 0.5 + gap - abs(yj - yi)
            if overlap_x <= 0.0 or overlap_y <= 0.0:
                continue

            if overlap_x <= overlap_y:
                sign = -1.0 if xj >= xi else 1.0
                out[i, 0] += sign * (overlap_x + 1e-6)
            else:
                sign = -1.0 if yj >= yi else 1.0
                out[i, 1] += sign * (overlap_y + 1e-6)

            moved_any = True

        out[:n_h] = project_to_canvas(
            out[:n_h],
            sizes[:n_h],
            canvas_width,
            canvas_height,
            fixed_mask=hard_fixed,
            fixed_positions=hard_fixed_pos,
        )

        if not moved_any:
            break

    return out


def legalize_soft_vs_all(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    fixed_mask: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    num_hard_macros: int,
    gap: float = 1e-4,
    max_iters: int = 80,
) -> torch.Tensor:
    """
    Resolve overlaps involving movable soft macros by moving only soft macros.
    """
    out = positions.clone()
    n_h = int(num_hard_macros)
    n_all = int(out.shape[0])
    if n_all - n_h <= 0:
        return out

    soft_fixed = fixed_mask[n_h:].clone()
    soft_fixed_pos = out[n_h:].clone()

    for _ in range(max_iters):
        moved_any = False
        for local_i in range(n_all - n_h):
            i = n_h + local_i
            if bool(soft_fixed[local_i]):
                continue

            xi = float(out[i, 0].item())
            yi = float(out[i, 1].item())
            wi = float(sizes[i, 0].item())
            hi = float(sizes[i, 1].item())

            dx = out[:, 0] - xi
            dy = out[:, 1] - yi
            sep_x = 0.5 * (wi + sizes[:, 0]) + float(gap)
            sep_y = 0.5 * (hi + sizes[:, 1]) + float(gap)

            ovx = sep_x - torch.abs(dx)
            ovy = sep_y - torch.abs(dy)
            mask = (ovx > 0.0) & (ovy > 0.0)
            mask[i] = False

            if not bool(mask.any()):
                continue

            pen = torch.minimum(torch.clamp(ovx, min=0.0), torch.clamp(ovy, min=0.0))
            pen = torch.where(mask, pen, torch.zeros_like(pen))
            j = int(torch.argmax(pen).item())

            xj = float(out[j, 0].item())
            yj = float(out[j, 1].item())
            wj = float(sizes[j, 0].item())
            hj = float(sizes[j, 1].item())

            overlap_x = (wi + wj) * 0.5 + gap - abs(xj - xi)
            overlap_y = (hi + hj) * 0.5 + gap - abs(yj - yi)
            if overlap_x <= 0.0 or overlap_y <= 0.0:
                continue

            if overlap_x <= overlap_y:
                sign = -1.0 if xj >= xi else 1.0
                out[i, 0] += sign * (overlap_x + 1e-6)
            else:
                sign = -1.0 if yj >= yi else 1.0
                out[i, 1] += sign * (overlap_y + 1e-6)

            moved_any = True

        out[n_h:] = project_to_canvas(
            out[n_h:],
            sizes[n_h:],
            canvas_width,
            canvas_height,
            fixed_mask=soft_fixed,
            fixed_positions=soft_fixed_pos,
        )

        if not moved_any:
            break

    return out


def extract_hard_edges_from_plc(
    benchmark: Benchmark,
    plc,
    max_edges: int = 12000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if plc is None:
        return (
            torch.zeros(0, 2, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
        )

    name_to_hidx: Dict[str, int] = {}
    for hard_i, module_idx in enumerate(benchmark.hard_macro_indices):
        if 0 <= module_idx < len(plc.modules_w_pins):
            name = plc.modules_w_pins[module_idx].get_name()
            name_to_hidx[name] = hard_i

    edge_dict: Dict[Tuple[int, int], float] = {}
    nets = getattr(plc, "nets", {})

    for driver, sinks in sorted(nets.items(), key=lambda kv: str(kv[0])):
        macros = set()
        ordered_sinks = sorted(list(sinks), key=str)
        for pin_name in [driver] + ordered_sinks:
            parent = pin_name.split("/")[0]
            if parent in name_to_hidx:
                macros.add(name_to_hidx[parent])

        if len(macros) < 2:
            continue

        macro_list = sorted(macros)
        w = 1.0 / max(1, len(macro_list) - 1)
        for i in range(len(macro_list)):
            for j in range(i + 1, len(macro_list)):
                a, b = macro_list[i], macro_list[j]
                pair = (a, b) if a < b else (b, a)
                edge_dict[pair] = edge_dict.get(pair, 0.0) + w

    if not edge_dict:
        return (
            torch.zeros(0, 2, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
        )

    items = sorted(edge_dict.items(), key=lambda kv: (-float(kv[1]), kv[0]))
    if max_edges > 0 and len(items) > max_edges:
        items = items[:max_edges]

    edge_index = torch.tensor([k for k, _ in items], dtype=torch.long)
    edge_weight = torch.tensor([v for _, v in items], dtype=torch.float32)
    return edge_index, edge_weight


def _get_plc_pin_maps(plc) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    cached = getattr(plc, "_team_plasma_pin_maps", None)
    if cached is not None:
        return cached

    pin_offset_by_name: Dict[str, torch.Tensor] = {}
    pin_pos_by_name: Dict[str, torch.Tensor] = {}
    for mod in getattr(plc, "modules_w_pins", []):
        if not hasattr(mod, "get_name"):
            continue
        pin_name = str(mod.get_name())
        pin_type = None
        if hasattr(mod, "get_type"):
            try:
                pin_type = mod.get_type()
            except Exception:
                pin_type = None
        macro_name = None
        if hasattr(mod, "get_macro_name"):
            try:
                macro_name = mod.get_macro_name()
            except Exception:
                macro_name = None

        is_macro_pin = pin_type == "MACRO_PIN" or ("/" in pin_name and macro_name is not None)
        if not is_macro_pin:
            continue

        pin_offset_by_name[pin_name] = torch.tensor(
            [
                float(getattr(mod, "x_offset", 0.0)),
                float(getattr(mod, "y_offset", 0.0)),
            ],
            dtype=torch.float32,
        )
        if hasattr(mod, "get_pos"):
            try:
                x, y = mod.get_pos()
                pin_pos_by_name[pin_name] = torch.tensor([float(x), float(y)], dtype=torch.float32)
            except Exception:
                pass

    cached = (pin_offset_by_name, pin_pos_by_name)
    setattr(plc, "_team_plasma_pin_maps", cached)
    return cached


def extract_pin_flux_tubes_from_plc(
    benchmark: Benchmark,
    plc,
    max_edges: int = 12000,
    min_degree: int = 2,
    max_degree: int = 0,
    weight_mode: str = "unit_net",
    degree_power: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if plc is None:
        return (
            torch.zeros(0, 2, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            torch.zeros(0, 2, 2, dtype=torch.float32),
        )

    name_to_hidx: Dict[str, int] = {}
    for hard_i, module_idx in enumerate(benchmark.hard_macro_indices):
        if 0 <= module_idx < len(plc.modules_w_pins):
            name = plc.modules_w_pins[module_idx].get_name()
            name_to_hidx[name] = hard_i

    pin_offset_by_name, _ = _get_plc_pin_maps(plc)
    edge_dict: Dict[Tuple[int, int], List[float]] = {}
    nets = getattr(plc, "nets", {})

    def _degree_scale(degree: int) -> float:
        denom = max(1.0, float(max(degree - 1, 1)))
        mode = str(weight_mode or "unit_net").strip().lower()
        power = max(float(degree_power), 0.0)
        if mode in {"unit", "unit_net", "uniform"}:
            return 1.0
        if mode == "inverse_sqrt_degree":
            return denom ** -0.5
        if mode == "inverse_power":
            return denom ** (-power)
        return 1.0 / denom

    for driver, sinks in sorted(nets.items(), key=lambda kv: str(kv[0])):
        hard_pins: List[Tuple[int, float, float]] = []
        ordered_sinks = sorted(list(sinks), key=str)
        for pin_name in [driver] + ordered_sinks:
            parent = pin_name.split("/")[0]
            if parent not in name_to_hidx:
                continue
            pin_offset = pin_offset_by_name.get(pin_name)
            if pin_offset is None:
                ox = 0.0
                oy = 0.0
            else:
                ox = float(pin_offset[0].item())
                oy = float(pin_offset[1].item())
            hard_pins.append((name_to_hidx[parent], ox, oy))

        m = len(hard_pins)
        if m < 2:
            continue
        macro_degree = len({hard_i for hard_i, _, _ in hard_pins})
        if macro_degree < max(2, int(min_degree)):
            continue
        if int(max_degree) > 0 and macro_degree > int(max_degree):
            continue

        pair_weight = _degree_scale(macro_degree) * (2.0 / float(m * max(m - 1, 1)))
        for i in range(m):
            hi, oxi, oyi = hard_pins[i]
            for j in range(i + 1, m):
                hj, oxj, oyj = hard_pins[j]
                if hi == hj:
                    continue
                if hi < hj:
                    key = (hi, hj)
                    ax, ay, bx, by = oxi, oyi, oxj, oyj
                else:
                    key = (hj, hi)
                    ax, ay, bx, by = oxj, oyj, oxi, oyi
                stats = edge_dict.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0])
                stats[0] += pair_weight
                stats[1] += pair_weight * ax
                stats[2] += pair_weight * ay
                stats[3] += pair_weight * bx
                stats[4] += pair_weight * by

    if not edge_dict:
        edge_index, edge_weight = extract_hard_edges_from_plc(benchmark, plc, max_edges=max_edges)
        edge_offsets = torch.zeros((int(edge_index.shape[0]), 2, 2), dtype=torch.float32)
        return edge_index, edge_weight, edge_offsets

    items = sorted(edge_dict.items(), key=lambda kv: (-float(kv[1][0]), kv[0]))
    if max_edges > 0 and len(items) > max_edges:
        items = items[:max_edges]

    edge_index = torch.tensor([k for k, _ in items], dtype=torch.long)
    edge_weight = torch.tensor([v[0] for _, v in items], dtype=torch.float32)
    edge_offsets = torch.zeros((len(items), 2, 2), dtype=torch.float32)
    for idx, (_, stats) in enumerate(items):
        w = max(float(stats[0]), 1e-9)
        edge_offsets[idx, 0, 0] = float(stats[1]) / w
        edge_offsets[idx, 0, 1] = float(stats[2]) / w
        edge_offsets[idx, 1, 0] = float(stats[3]) / w
        edge_offsets[idx, 1, 1] = float(stats[4]) / w
    return edge_index, edge_weight, edge_offsets


def extract_hard_net_bundles_from_plc(
    benchmark: Benchmark,
    plc,
    max_nets: int = 6000,
    min_degree: int = 3,
    use_pin_offsets: bool = False,
    weight_mode: str = "inverse_degree",
    degree_power: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract multi-pin hard-macro net bundles for whole-net congestion modeling.

    Returns a compressed representation:
      bundle_ptr:    [B + 1] offsets into member arrays
      bundle_index:  [M] hard-macro indices
      bundle_offsets:[M, 2] average pin offsets per hard macro in each bundle
      bundle_weight: [B] per-bundle demand weight
    """
    if plc is None or int(benchmark.num_hard_macros) <= 1:
        return (
            torch.zeros(1, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, 2, dtype=torch.float32),
            torch.zeros(0, dtype=torch.float32),
        )

    name_to_hidx: Dict[str, int] = {}
    for hard_i, module_idx in enumerate(benchmark.hard_macro_indices):
        if 0 <= module_idx < len(plc.modules_w_pins):
            name_to_hidx[plc.modules_w_pins[module_idx].get_name()] = hard_i

    pin_offset_by_name, _ = _get_plc_pin_maps(plc)
    nets = getattr(plc, "nets", {})
    bundles: List[Tuple[Tuple[int, ...], torch.Tensor, float, str]] = []

    for driver, sinks in sorted(nets.items(), key=lambda kv: str(kv[0])):
        macro_offsets: Dict[int, List[torch.Tensor]] = {}
        ordered_sinks = sorted(list(sinks), key=str)
        for pin_name in [driver] + ordered_sinks:
            parent = pin_name.split("/")[0]
            if parent not in name_to_hidx:
                continue
            hard_i = name_to_hidx[parent]
            if use_pin_offsets:
                pin_offset = pin_offset_by_name.get(pin_name, torch.zeros(2, dtype=torch.float32))
            else:
                pin_offset = torch.zeros(2, dtype=torch.float32)
            macro_offsets.setdefault(hard_i, []).append(pin_offset.float())

        degree = len(macro_offsets)
        if degree < max(2, int(min_degree)):
            continue

        members = tuple(sorted(macro_offsets))
        avg_offsets = []
        for hard_i in members:
            vals = macro_offsets[hard_i]
            if len(vals) == 1:
                avg_offsets.append(vals[0])
            else:
                avg_offsets.append(torch.stack(vals, dim=0).mean(dim=0))
        denom = max(1.0, float(degree - 1))
        mode = str(weight_mode or "inverse_degree").strip().lower()
        power = max(float(degree_power), 0.0)
        if mode == "unit":
            bundle_w = 1.0
        elif mode == "inverse_sqrt_degree":
            bundle_w = denom ** -0.5
        elif mode == "inverse_power":
            bundle_w = denom ** (-power)
        else:
            bundle_w = 1.0 / denom

        bundles.append(
            (
                members,
                torch.stack(avg_offsets, dim=0),
                bundle_w,
                str(driver),
            )
        )

    if not bundles:
        return (
            torch.zeros(1, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, 2, dtype=torch.float32),
            torch.zeros(0, dtype=torch.float32),
        )

    bundles.sort(key=lambda item: (-len(item[0]), item[3], item[0]))
    if max_nets > 0 and len(bundles) > max_nets:
        bundles = bundles[:max_nets]

    bundle_ptr = [0]
    bundle_index: List[int] = []
    bundle_offsets: List[torch.Tensor] = []
    bundle_weight: List[float] = []
    for members, offsets, weight, _ in bundles:
        bundle_index.extend(int(i) for i in members)
        bundle_offsets.extend(offsets)
        bundle_weight.append(float(weight))
        bundle_ptr.append(len(bundle_index))

    return (
        torch.tensor(bundle_ptr, dtype=torch.long),
        torch.tensor(bundle_index, dtype=torch.long),
        torch.stack(bundle_offsets, dim=0) if bundle_offsets else torch.zeros(0, 2, dtype=torch.float32),
        torch.tensor(bundle_weight, dtype=torch.float32),
    )


def extract_soft_cluster_bundles_from_plc(
    benchmark: Benchmark,
    plc,
    max_clusters: int = 24,
    min_hard_degree: int = 2,
    use_pin_offsets: bool = False,
    weight_mode: str = "soft_mass_inverse_degree",
    degree_power: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build virtual hard-macro bundles induced by high-signal soft clusters.

    Each soft macro acts like a compressed cluster surrogate: if several hard
    macros repeatedly connect through the same soft node, we create one virtual
    bundle over those hard macros. This captures hidden macro<->soft<->macro
    structure without moving soft macros directly.
    """
    if plc is None or int(benchmark.num_hard_macros) <= 1 or int(benchmark.num_soft_macros) <= 0:
        return (
            torch.zeros(1, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, 2, dtype=torch.float32),
            torch.zeros(0, dtype=torch.float32),
        )

    name_to_hidx: Dict[str, int] = {}
    for hard_i, module_idx in enumerate(benchmark.hard_macro_indices):
        if 0 <= module_idx < len(plc.modules_w_pins):
            name_to_hidx[plc.modules_w_pins[module_idx].get_name()] = hard_i

    soft_names: Dict[str, int] = {}
    for soft_i, module_idx in enumerate(benchmark.soft_macro_indices):
        if 0 <= module_idx < len(plc.modules_w_pins):
            soft_names[plc.modules_w_pins[module_idx].get_name()] = soft_i

    pin_offset_by_name, _ = _get_plc_pin_maps(plc)
    nets = getattr(plc, "nets", {})

    cluster_hard_offsets: Dict[str, Dict[int, List[torch.Tensor]]] = {}
    cluster_mass: Dict[str, float] = {}

    for driver, sinks in sorted(nets.items(), key=lambda kv: str(kv[0])):
        soft_nodes = set()
        hard_offsets: Dict[int, List[torch.Tensor]] = {}
        ordered_sinks = sorted(list(sinks), key=str)
        for pin_name in [driver] + ordered_sinks:
            parent = pin_name.split("/")[0]
            if parent in soft_names:
                soft_nodes.add(parent)
            if parent not in name_to_hidx:
                continue
            hard_i = name_to_hidx[parent]
            if use_pin_offsets:
                pin_offset = pin_offset_by_name.get(pin_name, torch.zeros(2, dtype=torch.float32))
            else:
                pin_offset = torch.zeros(2, dtype=torch.float32)
            hard_offsets.setdefault(hard_i, []).append(pin_offset.float())

        if len(hard_offsets) < max(2, int(min_hard_degree)) or not soft_nodes:
            continue

        soft_scale = 1.0 / max(1, len(soft_nodes))
        for soft_name in sorted(soft_nodes):
            entry = cluster_hard_offsets.setdefault(soft_name, {})
            cluster_mass[soft_name] = cluster_mass.get(soft_name, 0.0) + soft_scale
            for hard_i, vals in hard_offsets.items():
                entry.setdefault(hard_i, []).extend(vals)

    bundles: List[Tuple[Tuple[int, ...], torch.Tensor, float, str]] = []
    mode = str(weight_mode or "soft_mass_inverse_degree").strip().lower()
    power = max(float(degree_power), 0.0)

    for soft_name, macro_offsets in sorted(cluster_hard_offsets.items(), key=lambda kv: str(kv[0])):
        degree = len(macro_offsets)
        if degree < max(2, int(min_hard_degree)):
            continue
        members = tuple(sorted(macro_offsets))
        avg_offsets = []
        for hard_i in members:
            vals = macro_offsets[hard_i]
            if len(vals) == 1:
                avg_offsets.append(vals[0])
            else:
                avg_offsets.append(torch.stack(vals, dim=0).mean(dim=0))

        base_mass = float(cluster_mass.get(soft_name, 1.0))
        denom = max(1.0, float(degree - 1))
        if mode == "unit":
            bundle_w = 1.0
        elif mode == "soft_mass":
            bundle_w = base_mass
        elif mode == "soft_mass_inverse_sqrt_degree":
            bundle_w = base_mass * (denom ** -0.5)
        elif mode == "soft_mass_inverse_power":
            bundle_w = base_mass * (denom ** (-power))
        else:
            bundle_w = base_mass / denom

        bundles.append(
            (
                members,
                torch.stack(avg_offsets, dim=0),
                bundle_w,
                str(soft_name),
            )
        )

    if not bundles:
        return (
            torch.zeros(1, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, 2, dtype=torch.float32),
            torch.zeros(0, dtype=torch.float32),
        )

    bundles.sort(key=lambda item: (-item[2], -len(item[0]), item[3], item[0]))
    if max_clusters > 0 and len(bundles) > max_clusters:
        bundles = bundles[:max_clusters]

    bundle_ptr = [0]
    bundle_index: List[int] = []
    bundle_offsets: List[torch.Tensor] = []
    bundle_weight: List[float] = []
    for members, offsets, weight, _ in bundles:
        bundle_index.extend(int(i) for i in members)
        bundle_offsets.extend(offsets)
        bundle_weight.append(float(weight))
        bundle_ptr.append(len(bundle_index))

    return (
        torch.tensor(bundle_ptr, dtype=torch.long),
        torch.tensor(bundle_index, dtype=torch.long),
        torch.stack(bundle_offsets, dim=0) if bundle_offsets else torch.zeros(0, 2, dtype=torch.float32),
        torch.tensor(bundle_weight, dtype=torch.float32),
    )


def extract_soft_cluster_edges_from_plc(
    benchmark: Benchmark,
    plc,
    max_edges: int = 12000,
    min_hard_degree: int = 2,
    min_shared_clusters: int = 1,
    use_pin_offsets: bool = False,
    weight_mode: str = "soft_mass_inverse_degree",
    degree_power: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build pairwise hard-macro affinities induced by shared soft clusters.

    Compared with whole-net bundle forcing, this is a lighter hidden-cluster
    surrogate: soft-connected hard macros gain extra affinity edges, but keep
    the existing pairwise transport model instead of adding a new global source.
    """
    if plc is None or int(benchmark.num_hard_macros) <= 1 or int(benchmark.num_soft_macros) <= 0:
        return (
            torch.zeros(0, 2, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            torch.zeros(0, 2, 2, dtype=torch.float32),
        )

    name_to_hidx: Dict[str, int] = {}
    for hard_i, module_idx in enumerate(benchmark.hard_macro_indices):
        if 0 <= module_idx < len(plc.modules_w_pins):
            name_to_hidx[plc.modules_w_pins[module_idx].get_name()] = hard_i

    soft_names: Dict[str, int] = {}
    for soft_i, module_idx in enumerate(benchmark.soft_macro_indices):
        if 0 <= module_idx < len(plc.modules_w_pins):
            soft_names[plc.modules_w_pins[module_idx].get_name()] = soft_i

    pin_offset_by_name, _ = _get_plc_pin_maps(plc)
    nets = getattr(plc, "nets", {})

    cluster_hard_offsets: Dict[str, Dict[int, List[torch.Tensor]]] = {}
    cluster_hard_mass: Dict[str, Dict[int, float]] = {}
    cluster_mass: Dict[str, float] = {}

    for driver, sinks in sorted(nets.items(), key=lambda kv: str(kv[0])):
        soft_nodes = set()
        hard_offsets: Dict[int, List[torch.Tensor]] = {}
        ordered_sinks = sorted(list(sinks), key=str)
        for pin_name in [driver] + ordered_sinks:
            parent = pin_name.split("/")[0]
            if parent in soft_names:
                soft_nodes.add(parent)
            if parent not in name_to_hidx:
                continue
            hard_i = name_to_hidx[parent]
            if use_pin_offsets:
                pin_offset = pin_offset_by_name.get(pin_name, torch.zeros(2, dtype=torch.float32))
            else:
                pin_offset = torch.zeros(2, dtype=torch.float32)
            hard_offsets.setdefault(hard_i, []).append(pin_offset.float())

        if len(hard_offsets) < max(2, int(min_hard_degree)) or not soft_nodes:
            continue

        soft_scale = 1.0 / max(1, len(soft_nodes))
        for soft_name in sorted(soft_nodes):
            off_entry = cluster_hard_offsets.setdefault(soft_name, {})
            mass_entry = cluster_hard_mass.setdefault(soft_name, {})
            cluster_mass[soft_name] = cluster_mass.get(soft_name, 0.0) + soft_scale
            for hard_i, vals in hard_offsets.items():
                off_entry.setdefault(hard_i, []).extend(vals)
                mass_entry[hard_i] = mass_entry.get(hard_i, 0.0) + soft_scale * float(len(vals))

    edge_dict: Dict[Tuple[int, int], List[float]] = {}
    edge_cluster_count: Dict[Tuple[int, int], int] = {}
    mode = str(weight_mode or "soft_mass_inverse_degree").strip().lower()
    power = max(float(degree_power), 0.0)

    for soft_name, macro_offsets in sorted(cluster_hard_offsets.items(), key=lambda kv: str(kv[0])):
        degree = len(macro_offsets)
        if degree < max(2, int(min_hard_degree)):
            continue

        members = tuple(sorted(macro_offsets))
        avg_offsets: Dict[int, torch.Tensor] = {}
        for hard_i in members:
            vals = macro_offsets[hard_i]
            if len(vals) == 1:
                avg_offsets[hard_i] = vals[0]
            else:
                avg_offsets[hard_i] = torch.stack(vals, dim=0).mean(dim=0)

        base_mass = float(cluster_mass.get(soft_name, 1.0))
        per_hard_mass = cluster_hard_mass.get(soft_name, {})
        denom = max(1.0, float(degree - 1))
        if mode == "unit":
            cluster_scale = 1.0
        elif mode == "soft_mass":
            cluster_scale = base_mass
        elif mode == "soft_mass_inverse_sqrt_degree":
            cluster_scale = base_mass * (denom ** -0.5)
        elif mode == "soft_mass_inverse_power":
            cluster_scale = base_mass * (denom ** (-power))
        else:
            cluster_scale = base_mass / denom

        pair_count = max(1.0, float(degree * max(degree - 1, 1) / 2.0))
        for i in range(len(members)):
            a = members[i]
            ma = max(float(per_hard_mass.get(a, 0.0)), 0.0)
            if ma <= 0.0:
                continue
            for j in range(i + 1, len(members)):
                b = members[j]
                mb = max(float(per_hard_mass.get(b, 0.0)), 0.0)
                if mb <= 0.0:
                    continue

                affinity = math.sqrt(ma * mb) / max(base_mass, 1e-9)
                pair_w = (cluster_scale * affinity) / pair_count
                if pair_w <= 0.0:
                    continue

                key = (a, b)
                edge_cluster_count[key] = edge_cluster_count.get(key, 0) + 1
                stats = edge_dict.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0])
                stats[0] += pair_w
                stats[1] += pair_w * float(avg_offsets[a][0].item())
                stats[2] += pair_w * float(avg_offsets[a][1].item())
                stats[3] += pair_w * float(avg_offsets[b][0].item())
                stats[4] += pair_w * float(avg_offsets[b][1].item())

    min_shared = max(1, int(min_shared_clusters))
    if min_shared > 1 and edge_dict:
        edge_dict = {
            key: stats
            for key, stats in edge_dict.items()
            if int(edge_cluster_count.get(key, 0)) >= min_shared
        }

    if not edge_dict:
        return (
            torch.zeros(0, 2, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            torch.zeros(0, 2, 2, dtype=torch.float32),
        )

    items = sorted(
        edge_dict.items(),
        key=lambda kv: (-int(edge_cluster_count.get(kv[0], 0)), -float(kv[1][0]), kv[0]),
    )
    if max_edges > 0 and len(items) > max_edges:
        items = items[:max_edges]

    edge_index = torch.tensor([k for k, _ in items], dtype=torch.long)
    edge_weight = torch.tensor([v[0] for _, v in items], dtype=torch.float32)
    edge_offsets = torch.zeros((len(items), 2, 2), dtype=torch.float32)
    for idx, (_, stats) in enumerate(items):
        w = max(float(stats[0]), 1e-9)
        edge_offsets[idx, 0, 0] = float(stats[1]) / w
        edge_offsets[idx, 0, 1] = float(stats[2]) / w
        edge_offsets[idx, 1, 0] = float(stats[3]) / w
        edge_offsets[idx, 1, 1] = float(stats[4]) / w

    return edge_index, edge_weight, edge_offsets


def extract_soft_cluster_macro_group_info_from_plc(
    benchmark: Benchmark,
    plc,
    max_edges: int = 64,
    min_hard_degree: int = 2,
    min_shared_clusters: int = 2,
    use_pin_offsets: bool = False,
    weight_mode: str = "unit",
    degree_power: float = 1.0,
    max_group_size: int = 6,
    max_groups: int = 8,
) -> List[Tuple[List[int], float]]:
    """
    Build small hard-macro communities induced by repeated shared soft clusters.

    These groups are meant for exact local refine proposals, not for global
    transport. We keep only compact connected components from the strongest
    repeated-support pair graph.
    """
    edge_index, edge_weight, _ = extract_soft_cluster_edges_from_plc(
        benchmark,
        plc,
        max_edges=max_edges,
        min_hard_degree=min_hard_degree,
        min_shared_clusters=min_shared_clusters,
        use_pin_offsets=use_pin_offsets,
        weight_mode=weight_mode,
        degree_power=degree_power,
    )
    if edge_index.numel() == 0:
        return []

    n = int(benchmark.num_hard_macros)
    adj: Dict[int, set] = {}
    edge_score: Dict[Tuple[int, int], float] = {}
    for idx, (a, b) in enumerate(edge_index.tolist()):
        if idx >= max_edges:
            break
        if a == b:
            continue
        aa = int(a)
        bb = int(b)
        if aa > bb:
            aa, bb = bb, aa
        adj.setdefault(aa, set()).add(bb)
        adj.setdefault(bb, set()).add(aa)
        edge_score[(aa, bb)] = float(edge_weight[idx].item())

    seen = set()
    groups: List[Tuple[List[int], float]] = []
    for start in sorted(adj):
        if start in seen:
            continue
        stack = [start]
        comp = []
        seen.add(start)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in sorted(adj.get(cur, ())):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comp = sorted(set(int(i) for i in comp if 0 <= int(i) < n))
        if len(comp) < 2:
            continue
        if max_group_size > 0 and len(comp) > max_group_size:
            deg = {node: len(adj.get(node, ())) for node in comp}
            comp = sorted(comp, key=lambda node: (-deg[node], node))[:max_group_size]
            comp = sorted(comp)
        score = 0.0
        for i in range(len(comp)):
            for j in range(i + 1, len(comp)):
                a = int(comp[i])
                b = int(comp[j])
                if a > b:
                    a, b = b, a
                score += float(edge_score.get((a, b), 0.0))
        groups.append((comp, score))

    groups.sort(key=lambda item: (-float(item[1]), -len(item[0]), item[0]))
    if max_groups > 0 and len(groups) > max_groups:
        groups = groups[:max_groups]
    return groups


def extract_soft_cluster_macro_groups_from_plc(
    benchmark: Benchmark,
    plc,
    max_edges: int = 64,
    min_hard_degree: int = 2,
    min_shared_clusters: int = 2,
    use_pin_offsets: bool = False,
    weight_mode: str = "unit",
    degree_power: float = 1.0,
    max_group_size: int = 6,
    max_groups: int = 8,
) -> List[List[int]]:
    return [
        group
        for group, _score in extract_soft_cluster_macro_group_info_from_plc(
            benchmark,
            plc,
            max_edges=max_edges,
            min_hard_degree=min_hard_degree,
            min_shared_clusters=min_shared_clusters,
            use_pin_offsets=use_pin_offsets,
            weight_mode=weight_mode,
            degree_power=degree_power,
            max_group_size=max_group_size,
            max_groups=max_groups,
        )
    ]


def merge_bundle_terms(
    primary: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    secondary: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    p_ptr, p_idx, p_off, p_w = primary
    s_ptr, s_idx, s_off, s_w = secondary
    if int(p_w.numel()) == 0:
        return secondary
    if int(s_w.numel()) == 0:
        return primary

    out_ptr = [0]
    out_idx: List[int] = []
    out_off: List[torch.Tensor] = []
    out_w: List[float] = []

    for ptr, idx, off, weight in ((p_ptr, p_idx, p_off, p_w), (s_ptr, s_idx, s_off, s_w)):
        for b in range(int(weight.shape[0])):
            start = int(ptr[b].item())
            stop = int(ptr[b + 1].item())
            if stop <= start:
                continue
            out_idx.extend(int(v) for v in idx[start:stop].tolist())
            if off.numel() > 0:
                out_off.extend(off[start:stop])
            else:
                out_off.extend(torch.zeros((stop - start, 2), dtype=torch.float32))
            out_w.append(float(weight[b].item()))
            out_ptr.append(len(out_idx))

    return (
        torch.tensor(out_ptr, dtype=torch.long),
        torch.tensor(out_idx, dtype=torch.long),
        torch.stack(out_off, dim=0) if out_off else torch.zeros(0, 2, dtype=torch.float32),
        torch.tensor(out_w, dtype=torch.float32),
    )


def extract_hard_pin_pressure_terms_from_plc(
    benchmark: Benchmark,
    plc,
    min_degree: int = 2,
    max_degree: int = 0,
    weight_mode: str = "inverse_sqrt_degree",
    degree_power: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract hard-macro pin point terms for local pin-access pressure modeling.

    Returns:
      pin_index:   [P] hard-macro indices
      pin_offsets: [P, 2] pin offsets relative to macro center
      pin_weight:  [P] per-pin occupancy weight
    """
    if plc is None or int(benchmark.num_hard_macros) <= 0:
        return (
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, 2, dtype=torch.float32),
            torch.zeros(0, dtype=torch.float32),
        )

    name_to_hidx: Dict[str, int] = {}
    for hard_i, module_idx in enumerate(benchmark.hard_macro_indices):
        if 0 <= module_idx < len(plc.modules_w_pins):
            name_to_hidx[plc.modules_w_pins[module_idx].get_name()] = hard_i

    pin_offset_by_name, _ = _get_plc_pin_maps(plc)
    nets = getattr(plc, "nets", {})
    pin_index: List[int] = []
    pin_offsets: List[torch.Tensor] = []
    pin_weight: List[float] = []

    def _degree_scale(degree: int) -> float:
        denom = max(1.0, float(max(degree - 1, 1)))
        mode = str(weight_mode or "inverse_sqrt_degree").strip().lower()
        power = max(float(degree_power), 0.0)
        if mode in {"unit", "unit_net", "uniform"}:
            return 1.0
        if mode == "inverse_degree":
            return 1.0 / denom
        if mode == "inverse_power":
            return denom ** (-power)
        return denom ** -0.5

    for driver, sinks in sorted(nets.items(), key=lambda kv: str(kv[0])):
        hard_pin_names: List[str] = []
        hard_nodes: List[int] = []
        ordered_sinks = sorted(list(sinks), key=str)
        for pin_name in [driver] + ordered_sinks:
            parent = pin_name.split("/")[0]
            if parent not in name_to_hidx:
                continue
            hard_pin_names.append(pin_name)
            hard_nodes.append(name_to_hidx[parent])

        degree = len(set(hard_nodes))
        if degree < max(2, int(min_degree)):
            continue
        if int(max_degree) > 0 and degree > int(max_degree):
            continue

        w = _degree_scale(degree)
        for pin_name, hard_i in zip(hard_pin_names, hard_nodes):
            pin_index.append(int(hard_i))
            pin_offsets.append(pin_offset_by_name.get(pin_name, torch.zeros(2, dtype=torch.float32)).float())
            pin_weight.append(float(w))

    if not pin_index:
        return (
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, 2, dtype=torch.float32),
            torch.zeros(0, dtype=torch.float32),
        )

    return (
        torch.tensor(pin_index, dtype=torch.long),
        torch.stack(pin_offsets, dim=0),
        torch.tensor(pin_weight, dtype=torch.float32),
    )


def extract_hard_pin_edge_profile_from_plc(
    benchmark: Benchmark,
    plc,
    min_degree: int = 2,
    max_degree: int = 0,
    weight_mode: str = "inverse_sqrt_degree",
    degree_power: float = 1.0,
) -> torch.Tensor:
    """
    Build per-hard-macro pin weights on each edge: [left, right, bottom, top].
    """
    n = int(benchmark.num_hard_macros)
    profile = torch.zeros((n, 4), dtype=torch.float32)
    if plc is None or n <= 0:
        return profile

    name_to_hidx: Dict[str, int] = {}
    for hard_i, module_idx in enumerate(benchmark.hard_macro_indices):
        if 0 <= module_idx < len(plc.modules_w_pins):
            name_to_hidx[plc.modules_w_pins[module_idx].get_name()] = hard_i

    hard_sizes = benchmark.macro_sizes[:n].float()
    pin_offset_by_name, _ = _get_plc_pin_maps(plc)
    nets = getattr(plc, "nets", {})

    def _degree_scale(degree: int) -> float:
        denom = max(1.0, float(max(degree - 1, 1)))
        mode = str(weight_mode or "inverse_sqrt_degree").strip().lower()
        power = max(float(degree_power), 0.0)
        if mode in {"unit", "unit_net", "uniform"}:
            return 1.0
        if mode == "inverse_degree":
            return 1.0 / denom
        if mode == "inverse_power":
            return denom ** (-power)
        return denom ** -0.5

    for driver, sinks in sorted(nets.items(), key=lambda kv: str(kv[0])):
        hard_pins: List[Tuple[int, torch.Tensor]] = []
        ordered_sinks = sorted(list(sinks), key=str)
        for pin_name in [driver] + ordered_sinks:
            parent = pin_name.split("/")[0]
            if parent not in name_to_hidx:
                continue
            hard_i = name_to_hidx[parent]
            hard_pins.append((hard_i, pin_offset_by_name.get(pin_name, torch.zeros(2, dtype=torch.float32)).float()))

        degree = len({hard_i for hard_i, _ in hard_pins})
        if degree < max(2, int(min_degree)):
            continue
        if int(max_degree) > 0 and degree > int(max_degree):
            continue

        per_pin_w = _degree_scale(degree) / max(1, len(hard_pins))
        for hard_i, offset in hard_pins:
            half_w = max(float(hard_sizes[hard_i, 0].item()) * 0.5, 1e-6)
            half_h = max(float(hard_sizes[hard_i, 1].item()) * 0.5, 1e-6)
            nx = abs(float(offset[0].item())) / half_w
            ny = abs(float(offset[1].item())) / half_h
            if nx >= ny:
                side = 1 if float(offset[0].item()) >= 0.0 else 0
            else:
                side = 3 if float(offset[1].item()) >= 0.0 else 2
            profile[hard_i, side] += float(per_pin_w)
    return profile


def compute_pin_edge_sheath_force(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    edge_profile: torch.Tensor,
    cfg: Dict[str, object],
    canvas_width: float,
    canvas_height: float,
) -> torch.Tensor:
    """
    Plasma-sheath-inspired local force for pin-heavy macro edges facing across
    narrow horizontal/vertical routing channels.
    """
    n = int(positions.shape[0])
    out = torch.zeros((n, 2), dtype=torch.float32, device=positions.device)
    if n <= 1 or edge_profile.numel() == 0 or not bool(cfg.get("enabled", False)):
        return out

    profile = edge_profile.to(device=positions.device, dtype=torch.float32)
    span = max(float(cfg.get("channel_span_frac", 0.06)), 1e-4) * max(float(canvas_width), float(canvas_height))
    sigma = max(float(cfg.get("sigma_frac", 0.03)), 1e-4) * max(float(canvas_width), float(canvas_height))
    min_overlap_frac = max(float(cfg.get("min_overlap_frac", 0.15)), 0.0)
    inv_gap_power = max(float(cfg.get("inv_gap_power", 1.0)), 0.0)
    edge_power = max(float(cfg.get("edge_power", 1.0)), 1e-6)
    uniform_edge_bias = max(float(cfg.get("uniform_edge_bias", 0.0)), 0.0)
    min_gap = max(float(cfg.get("min_gap_frac", 0.002)), 1e-5) * max(float(canvas_width), float(canvas_height))
    cap = max(float(cfg.get("max_pair_pressure", 3.0)), 0.0)

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
                    sx = 1.0 if dx >= 0.0 else -1.0
                    out[i, 0] -= sx * pressure
                    out[j, 0] += sx * pressure

            gap_y = abs(dy) - 0.5 * (hi + hj)
            if gap_y >= 0.0 and gap_y < span and overlap_x > min_overlap_frac * max(min(wi, wj), 1e-6):
                facing = (ti * bj) if dy >= 0.0 else (bi * tj)
                if facing > 0.0:
                    overlap_scale = overlap_x / max(min(wi, wj), 1e-6)
                    gap_eff = max(gap_y, min_gap)
                    pressure = (facing ** edge_power) * overlap_scale * math.exp(-gap_y / sigma) / (gap_eff ** inv_gap_power)
                    pressure = min(cap, pressure)
                    sy = 1.0 if dy >= 0.0 else -1.0
                    out[i, 1] -= sy * pressure
                    out[j, 1] += sy * pressure

    norm = torch.linalg.norm(out, dim=1)
    if norm.numel() > 0:
        denom = torch.quantile(norm, float(cfg.get("normalize_quantile", 0.90))) if n >= 10 else torch.max(norm)
        d = float(denom.item())
        if d > 1e-6:
            out = out / d
    return out


def build_hard_anchor_targets_from_plc(
    benchmark: Benchmark,
    plc,
    use_pin_positions: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-hard-macro anchor targets from nets touching soft macros and ports.

    Returns:
      targets: [num_hard, 2] anchor centroid targets (defaults to current pos)
      weights: [num_hard] confidence weights (0 means no anchor info)
    """
    n = int(benchmark.num_hard_macros)
    if n <= 0:
        return torch.zeros(0, 2, dtype=torch.float32), torch.zeros(0, dtype=torch.float32)

    hard_current = benchmark.macro_positions[:n].clone().float()
    if plc is None:
        return hard_current, torch.zeros(n, dtype=torch.float32)

    _, pin_pos_by_name = _get_plc_pin_maps(plc)
    name_to_hidx: Dict[str, int] = {}
    for hard_i, module_idx in enumerate(benchmark.hard_macro_indices):
        if 0 <= module_idx < len(plc.modules_w_pins):
            name_to_hidx[plc.modules_w_pins[module_idx].get_name()] = hard_i

    anchor_pos_by_name: Dict[str, torch.Tensor] = {}
    for module_idx in benchmark.soft_macro_indices:
        if 0 <= module_idx < len(plc.modules_w_pins):
            mod = plc.modules_w_pins[module_idx]
            x, y = mod.get_pos()
            anchor_pos_by_name[mod.get_name()] = torch.tensor([float(x), float(y)], dtype=torch.float32)
    for module_idx in getattr(plc, "port_indices", []):
        if 0 <= module_idx < len(plc.modules_w_pins):
            mod = plc.modules_w_pins[module_idx]
            x, y = mod.get_pos()
            anchor_pos_by_name[mod.get_name()] = torch.tensor([float(x), float(y)], dtype=torch.float32)

    target_sum = torch.zeros(n, 2, dtype=torch.float32)
    weight_sum = torch.zeros(n, dtype=torch.float32)

    nets = getattr(plc, "nets", {})
    for driver, sinks in sorted(nets.items(), key=lambda kv: str(kv[0])):
        hard_nodes = set()
        anchors: List[torch.Tensor] = []
        ordered_sinks = sorted(list(sinks), key=str)
        for pin_name in [driver] + ordered_sinks:
            parent = pin_name.split("/")[0]
            if parent in name_to_hidx:
                hard_nodes.add(name_to_hidx[parent])
            elif use_pin_positions and pin_name in pin_pos_by_name:
                anchors.append(pin_pos_by_name[pin_name])
            elif parent in anchor_pos_by_name:
                anchors.append(anchor_pos_by_name[parent])

        if not hard_nodes or not anchors:
            continue

        anchor_center = torch.stack(anchors, dim=0).mean(dim=0)
        # Scale by fanout so large nets do not dominate.
        w = 1.0 / max(1, len(hard_nodes))
        for hi in sorted(hard_nodes):
            target_sum[hi] += float(w) * anchor_center
            weight_sum[hi] += float(w)

    targets = hard_current.clone()
    mask = weight_sum > 1e-12
    if bool(mask.any()):
        targets[mask] = target_sum[mask] / weight_sum[mask].unsqueeze(1)
    return targets, weight_sum


def filter_hard_anchor_targets(
    targets: torch.Tensor,
    weights: torch.Tensor,
    cfg: Dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Keep only the strongest PLC-derived hard anchor signals.

    This lets route-local regimes use anchor information selectively instead of
    pulling every macro toward a broad soft/port centroid field.
    """
    if not bool(cfg.get("enabled", False)):
        return targets, weights
    if weights.numel() == 0:
        return targets, weights

    out_w = weights.clone().float()
    keep = out_w > 1e-12
    if not bool(keep.any()):
        return targets, out_w

    positive = out_w[keep]

    min_weight = float(cfg.get("min_weight", 0.0))
    if min_weight > 0.0:
        keep &= out_w >= min_weight

    weight_quantile = cfg.get("min_weight_quantile", None)
    if weight_quantile is not None and positive.numel() > 0:
        q = float(weight_quantile)
        q = min(1.0, max(0.0, q))
        thresh = float(torch.quantile(positive, torch.tensor(q, dtype=torch.float32)).item())
        keep &= out_w >= thresh

    topk = int(cfg.get("top_hard_nodes", 0))
    if topk > 0 and positive.numel() > 0:
        active = torch.nonzero(out_w > 1e-12, as_tuple=False).flatten()
        if active.numel() > topk:
            active_scores = out_w[active]
            order = torch.argsort(active_scores, descending=True, stable=True)
            top_active = active[order[:topk]]
            top_keep = torch.zeros_like(keep)
            top_keep[top_active] = True
            keep &= top_keep

    if bool(torch.all(keep)):
        return targets, out_w

    out_w[~keep] = 0.0
    return targets, out_w


def build_soft_transport_model_from_plc(
    benchmark: Benchmark,
    plc,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build sparse soft-to-hard anchor edges plus fixed (port) anchor sums.

    Returns:
      soft_hard_index: [E, 2] with columns (soft_idx, hard_idx)
      soft_hard_weight: [E]
      soft_fixed_sum: [num_soft, 2]
      soft_fixed_weight: [num_soft]
    """
    m = int(benchmark.num_soft_macros)
    if m <= 0 or plc is None:
        return (
            torch.zeros(0, 2, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            torch.zeros(m, 2, dtype=torch.float32),
            torch.zeros(m, dtype=torch.float32),
        )

    name_to_soft: Dict[str, int] = {}
    for soft_i, module_idx in enumerate(benchmark.soft_macro_indices):
        if 0 <= module_idx < len(plc.modules_w_pins):
            name_to_soft[plc.modules_w_pins[module_idx].get_name()] = soft_i

    name_to_hidx: Dict[str, int] = {}
    for hard_i, module_idx in enumerate(benchmark.hard_macro_indices):
        if 0 <= module_idx < len(plc.modules_w_pins):
            name_to_hidx[plc.modules_w_pins[module_idx].get_name()] = hard_i

    port_pos_by_name: Dict[str, torch.Tensor] = {}
    for module_idx in getattr(plc, "port_indices", []):
        if 0 <= module_idx < len(plc.modules_w_pins):
            mod = plc.modules_w_pins[module_idx]
            x, y = mod.get_pos()
            port_pos_by_name[mod.get_name()] = torch.tensor([float(x), float(y)], dtype=torch.float32)

    edge_dict: Dict[Tuple[int, int], float] = {}
    fixed_sum = torch.zeros(m, 2, dtype=torch.float32)
    fixed_weight = torch.zeros(m, dtype=torch.float32)

    nets = getattr(plc, "nets", {})
    for driver, sinks in sorted(nets.items(), key=lambda kv: str(kv[0])):
        soft_nodes = set()
        hard_nodes = set()
        port_anchors: List[torch.Tensor] = []

        ordered_sinks = sorted(list(sinks), key=str)
        for pin_name in [driver] + ordered_sinks:
            parent = pin_name.split("/")[0]
            if parent in name_to_soft:
                soft_nodes.add(name_to_soft[parent])
            elif parent in name_to_hidx:
                hard_nodes.add(name_to_hidx[parent])
            elif parent in port_pos_by_name:
                port_anchors.append(port_pos_by_name[parent])

        if not soft_nodes or (not hard_nodes and not port_anchors):
            continue

        soft_scale = 1.0 / max(1, len(soft_nodes))
        if hard_nodes:
            hard_scale = soft_scale / max(1, len(hard_nodes))
            for s in sorted(soft_nodes):
                for h in sorted(hard_nodes):
                    edge_dict[(s, h)] = edge_dict.get((s, h), 0.0) + hard_scale

        if port_anchors:
            port_center = torch.stack(port_anchors, dim=0).mean(dim=0)
            for s in sorted(soft_nodes):
                fixed_sum[s] += float(soft_scale) * port_center
                fixed_weight[s] += float(soft_scale)

    if not edge_dict:
        return (
            torch.zeros(0, 2, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            fixed_sum,
            fixed_weight,
        )

    items = sorted(edge_dict.items())
    edge_index = torch.tensor([k for k, _ in items], dtype=torch.long)
    edge_weight = torch.tensor([v for _, v in items], dtype=torch.float32)
    return edge_index, edge_weight, fixed_sum, fixed_weight


def compute_soft_anchor_targets(
    soft_pos: torch.Tensor,
    hard_pos: torch.Tensor,
    soft_hard_index: torch.Tensor,
    soft_hard_weight: torch.Tensor,
    soft_fixed_sum: torch.Tensor,
    soft_fixed_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m = int(soft_pos.shape[0])
    if m <= 0:
        return soft_pos, torch.zeros(0, dtype=torch.float32, device=soft_pos.device)

    target_sum = soft_fixed_sum.clone()
    weight_sum = soft_fixed_weight.clone()

    if soft_hard_index.numel() > 0:
        sidx = soft_hard_index[:, 0].long()
        hidx = soft_hard_index[:, 1].long()
        contrib = hard_pos[hidx] * soft_hard_weight.unsqueeze(1)
        target_sum.index_add_(0, sidx, contrib)
        weight_sum.index_add_(0, sidx, soft_hard_weight)

    targets = soft_pos.clone()
    mask = weight_sum > 1e-9
    if bool(mask.any()):
        targets[mask] = target_sum[mask] / weight_sum[mask].unsqueeze(1)
    return targets, weight_sum


def filter_soft_transport_model(
    soft_transport_model: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    cfg: Dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Keep only the highest-signal soft transport anchors.

    This is intentionally selective: for soft-heavy regimes we only want the
    strongest PLC-derived soft<->hard couplings, not the full diffuse graph.
    """
    if not bool(cfg.get("enabled", False)):
        return soft_transport_model

    soft_hard_index, soft_hard_weight, soft_fixed_sum, soft_fixed_weight = soft_transport_model
    num_soft = int(soft_fixed_weight.shape[0]) if soft_fixed_weight.ndim > 0 else int(soft_fixed_sum.shape[0])
    if num_soft <= 0:
        return soft_transport_model

    soft_mass = torch.zeros(num_soft, dtype=torch.float32)
    soft_degree = torch.zeros(num_soft, dtype=torch.long)
    if soft_hard_index.numel() > 0:
        sidx = soft_hard_index[:, 0].long()
        soft_mass.index_add_(0, sidx, soft_hard_weight.float())
        soft_degree = torch.bincount(sidx, minlength=num_soft)
    if soft_fixed_weight.numel() > 0:
        soft_mass = soft_mass + soft_fixed_weight.float()

    keep = torch.ones(num_soft, dtype=torch.bool)

    min_degree = int(cfg.get("min_soft_degree", 0))
    if min_degree > 0:
        keep &= soft_degree >= min_degree

    positive = soft_mass[soft_mass > 1e-12]
    mass_quantile = cfg.get("min_soft_mass_quantile", None)
    if positive.numel() > 0 and mass_quantile is not None:
        q = float(mass_quantile)
        q = min(1.0, max(0.0, q))
        thresh = float(torch.quantile(positive, torch.tensor(q, dtype=torch.float32)).item())
        keep &= soft_mass >= thresh

    topk = int(cfg.get("top_soft_nodes", 0))
    if topk > 0 and positive.numel() > 0:
        active = torch.nonzero(soft_mass > 1e-12, as_tuple=False).flatten()
        if active.numel() > topk:
            active_scores = soft_mass[active]
            order = torch.argsort(active_scores, descending=True, stable=True)
            top_active = active[order[:topk]]
            top_keep = torch.zeros(num_soft, dtype=torch.bool)
            top_keep[top_active] = True
            keep &= top_keep

    if not bool(keep.any()):
        return (
            torch.zeros(0, 2, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            torch.zeros_like(soft_fixed_sum),
            torch.zeros_like(soft_fixed_weight),
        )

    if bool(torch.all(keep)):
        return soft_transport_model

    if soft_hard_index.numel() > 0:
        edge_mask = keep[soft_hard_index[:, 0].long()]
        soft_hard_index = soft_hard_index[edge_mask]
        soft_hard_weight = soft_hard_weight[edge_mask]

    soft_fixed_sum = soft_fixed_sum.clone()
    soft_fixed_weight = soft_fixed_weight.clone()
    drop = ~keep
    if bool(drop.any()):
        soft_fixed_sum[drop] = 0.0
        soft_fixed_weight[drop] = 0.0

    return soft_hard_index, soft_hard_weight, soft_fixed_sum, soft_fixed_weight


def build_knn_edges(
    hard_positions: torch.Tensor,
    k: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(hard_positions.shape[0])
    if n <= 1:
        return (
            torch.zeros(0, 2, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
        )

    dist = torch.cdist(hard_positions.float(), hard_positions.float(), p=2)
    dist.fill_diagonal_(float("inf"))

    k_eff = min(k, n - 1)
    knn = torch.topk(dist, k=k_eff, largest=False).indices

    pair_to_w: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j in knn[i].tolist():
            a, b = (i, j) if i < j else (j, i)
            d = float(dist[i, j].item())
            w = 1.0 / max(d, 1e-3)
            pair_to_w[(a, b)] = max(pair_to_w.get((a, b), 0.0), w)

    pairs = sorted(pair_to_w)
    edge_index = torch.tensor(pairs, dtype=torch.long)
    edge_weight = torch.tensor([pair_to_w[p] for p in pairs], dtype=torch.float32)
    return edge_index, edge_weight


def merge_edge_sets(
    edge_index_a: torch.Tensor,
    edge_weight_a: torch.Tensor,
    edge_index_b: torch.Tensor,
    edge_weight_b: torch.Tensor,
    scale_b: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge two undirected edge sets by summing duplicate pair weights."""
    edge_map: Dict[Tuple[int, int], float] = {}

    if edge_index_a.numel() > 0:
        for idx, (i, j) in enumerate(edge_index_a.tolist()):
            a, b = (i, j) if i < j else (j, i)
            edge_map[(a, b)] = edge_map.get((a, b), 0.0) + float(edge_weight_a[idx].item())

    if edge_index_b.numel() > 0:
        s = float(scale_b)
        for idx, (i, j) in enumerate(edge_index_b.tolist()):
            a, b = (i, j) if i < j else (j, i)
            edge_map[(a, b)] = edge_map.get((a, b), 0.0) + s * float(edge_weight_b[idx].item())

    if not edge_map:
        return (
            torch.zeros(0, 2, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
        )

    items = list(edge_map.items())
    edge_index = torch.tensor([k for k, _ in items], dtype=torch.long)
    edge_weight = torch.tensor([v for _, v in items], dtype=torch.float32)
    return edge_index, edge_weight


def merge_edge_sets_with_offsets(
    edge_index_a: torch.Tensor,
    edge_weight_a: torch.Tensor,
    edge_offsets_a: Optional[torch.Tensor],
    edge_index_b: torch.Tensor,
    edge_weight_b: torch.Tensor,
    edge_offsets_b: Optional[torch.Tensor] = None,
    scale_b: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_map: Dict[Tuple[int, int], List[float]] = {}

    def _add(
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_offsets: Optional[torch.Tensor],
        scale: float,
    ) -> None:
        if edge_index.numel() == 0:
            return
        s = float(scale)
        has_offsets = edge_offsets is not None and edge_offsets.numel() > 0
        for idx, (i, j) in enumerate(edge_index.tolist()):
            w = s * float(edge_weight[idx].item())
            if i <= j:
                key = (i, j)
                if has_offsets:
                    ax = float(edge_offsets[idx, 0, 0].item())
                    ay = float(edge_offsets[idx, 0, 1].item())
                    bx = float(edge_offsets[idx, 1, 0].item())
                    by = float(edge_offsets[idx, 1, 1].item())
                else:
                    ax = ay = bx = by = 0.0
            else:
                key = (j, i)
                if has_offsets:
                    ax = float(edge_offsets[idx, 1, 0].item())
                    ay = float(edge_offsets[idx, 1, 1].item())
                    bx = float(edge_offsets[idx, 0, 0].item())
                    by = float(edge_offsets[idx, 0, 1].item())
                else:
                    ax = ay = bx = by = 0.0
            stats = edge_map.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0])
            stats[0] += w
            stats[1] += w * ax
            stats[2] += w * ay
            stats[3] += w * bx
            stats[4] += w * by

    _add(edge_index_a, edge_weight_a, edge_offsets_a, 1.0)
    _add(edge_index_b, edge_weight_b, edge_offsets_b, scale_b)

    if not edge_map:
        return (
            torch.zeros(0, 2, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            torch.zeros(0, 2, 2, dtype=torch.float32),
        )

    items = list(edge_map.items())
    edge_index = torch.tensor([k for k, _ in items], dtype=torch.long)
    edge_weight = torch.tensor([v[0] for _, v in items], dtype=torch.float32)
    edge_offsets = torch.zeros((len(items), 2, 2), dtype=torch.float32)
    for idx, (_, stats) in enumerate(items):
        w = max(float(stats[0]), 1e-9)
        edge_offsets[idx, 0, 0] = float(stats[1]) / w
        edge_offsets[idx, 0, 1] = float(stats[2]) / w
        edge_offsets[idx, 1, 0] = float(stats[3]) / w
        edge_offsets[idx, 1, 1] = float(stats[4]) / w
    return edge_index, edge_weight, edge_offsets


def build_flux_surface_seed(
    base_positions: torch.Tensor,
    sizes: torch.Tensor,
    fixed_mask: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    anchor_targets: Optional[torch.Tensor] = None,
    anchor_weights: Optional[torch.Tensor] = None,
    cfg: Optional[Dict] = None,
) -> torch.Tensor:
    cfg = dict(cfg or {})
    n = int(base_positions.shape[0])
    if n <= 3 or edge_index.numel() == 0:
        return base_positions.clone()

    work = base_positions.detach().cpu().double()
    edge_idx = edge_index.detach().cpu().long()
    edge_w = edge_weight.detach().cpu().double() if edge_weight.numel() > 0 else torch.ones(int(edge_idx.shape[0]), dtype=torch.double)

    affinity_power = max(float(cfg.get("affinity_power", 1.0)), 1e-6)
    edge_w = torch.pow(torch.clamp(edge_w, min=1e-9), affinity_power)

    wmat = torch.zeros((n, n), dtype=torch.double)
    for idx, (i, j) in enumerate(edge_idx.tolist()):
        wij = float(edge_w[idx].item())
        wmat[i, j] += wij
        wmat[j, i] += wij

    degree = wmat.sum(dim=1)
    if float(torch.max(degree).item()) <= 1e-9:
        return base_positions.clone()

    inv_sqrt_deg = torch.rsqrt(torch.clamp(degree, min=1e-9))
    lap = torch.eye(n, dtype=torch.double) - inv_sqrt_deg.unsqueeze(1) * wmat * inv_sqrt_deg.unsqueeze(0)
    evals, evecs = torch.linalg.eigh(lap)

    valid = torch.nonzero(evals > 1e-7).flatten()
    if int(valid.numel()) < 2:
        return base_positions.clone()

    v1 = evecs[:, int(valid[0].item())]
    v2 = evecs[:, int(valid[1].item())]
    embed = torch.stack([v1, v2], dim=1)
    embed = embed - embed.mean(dim=0, keepdim=True)

    mode = str(cfg.get("mode", "shell")).lower()
    scale_frac = float(cfg.get("scale_frac", 0.34))
    scale_frac = max(0.10, min(0.48, scale_frac))
    center = torch.tensor([0.5 * float(canvas_width), 0.5 * float(canvas_height)], dtype=torch.double)

    if mode == "direct":
        denom = torch.quantile(embed.abs(), 0.90, dim=0)
        denom = torch.clamp(denom, min=1e-6)
        norm = embed / denom
        coords = torch.empty_like(norm)
        coords[:, 0] = center[0] + norm[:, 0] * (float(canvas_width) * scale_frac)
        coords[:, 1] = center[1] + norm[:, 1] * (float(canvas_height) * scale_frac)
    else:
        ang = torch.atan2(embed[:, 1], embed[:, 0] + 1e-12)
        rad = torch.linalg.norm(embed, dim=1)
        r0 = float(torch.quantile(rad, 0.10).item())
        r1 = float(torch.quantile(rad, 0.90).item())
        rden = max(r1 - r0, 1e-6)
        rad_norm = torch.clamp((rad - r0) / rden, min=0.0, max=1.0)

        degree_norm = (degree - degree.mean()) / max(float(degree.std().item()), 1e-6)
        degree_norm = torch.clamp(0.5 + 0.2 * degree_norm, min=0.0, max=1.0)
        radial_mix = float(cfg.get("degree_radial_mix", 0.25))
        radius = torch.clamp((1.0 - radial_mix) * rad_norm + radial_mix * degree_norm, min=0.12, max=1.0)

        a = float(canvas_width) * scale_frac
        b = float(canvas_height) * scale_frac
        coords = torch.empty((n, 2), dtype=torch.double)
        coords[:, 0] = center[0] + torch.cos(ang) * a * radius
        coords[:, 1] = center[1] + torch.sin(ang) * b * radius

    if anchor_targets is not None and anchor_weights is not None and anchor_targets.numel() > 0 and anchor_weights.numel() > 0:
        targets = anchor_targets.detach().cpu().double()
        weights = torch.clamp(anchor_weights.detach().cpu().double(), min=0.0)
        mask = weights > 1e-8
        if bool(mask.any()):
            ref = max(float(torch.quantile(weights[mask], 0.90).item()), 1e-6)
            blend = float(cfg.get("anchor_blend", 0.25))
            scaled = torch.clamp(weights / ref, min=0.0, max=1.0).unsqueeze(1) * blend
            coords[mask] = (1.0 - scaled[mask]) * coords[mask] + scaled[mask] * targets[mask]

    mix_with_initial = float(cfg.get("mix_with_initial", 0.10))
    mix_with_initial = max(0.0, min(0.95, mix_with_initial))
    if mix_with_initial > 0.0:
        coords = (1.0 - mix_with_initial) * coords + mix_with_initial * work

    out = base_positions.clone()
    out[:] = coords.float()
    out = project_to_canvas(
        out,
        sizes,
        canvas_width,
        canvas_height,
        fixed_mask=fixed_mask,
        fixed_positions=base_positions,
    )
    return out


def _edge_endpoint_positions(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    src_pos = positions[src]
    dst_pos = positions[dst]
    if edge_offsets is not None and edge_offsets.numel() > 0:
        offs = edge_offsets.to(device=positions.device, dtype=positions.dtype)
        src_pos = src_pos + offs[:, 0]
        dst_pos = dst_pos + offs[:, 1]
    return src_pos, dst_pos


def approximate_wirelength_cost(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_offsets: Optional[torch.Tensor] = None,
) -> float:
    if edge_index.numel() == 0:
        return 0.0

    src_pos, dst_pos = _edge_endpoint_positions(positions, edge_index, edge_offsets=edge_offsets)
    dx = torch.abs(src_pos[:, 0] - dst_pos[:, 0])
    dy = torch.abs(src_pos[:, 1] - dst_pos[:, 1])
    wl = (edge_weight * (dx + dy)).sum()
    norm = float(edge_weight.sum().item()) + 1e-9
    return float(wl.item() / norm)


def approximate_wirelength_delta_for_shift(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    macro_idx: int,
    new_xy: torch.Tensor,
    edge_offsets: Optional[torch.Tensor] = None,
) -> float:
    if edge_index.numel() == 0:
        return 0.0

    idx = int(macro_idx)
    touch = (edge_index[:, 0] == idx) | (edge_index[:, 1] == idx)
    if not bool(touch.any()):
        return 0.0

    sel_edges = edge_index[touch]
    sel_weights = edge_weight[touch]

    old_pos = positions
    new_pos = positions.clone()
    new_pos[idx] = new_xy

    sel_offsets = edge_offsets[touch] if edge_offsets is not None and edge_offsets.numel() > 0 else None
    old_src_pos, old_dst_pos = _edge_endpoint_positions(old_pos, sel_edges, edge_offsets=sel_offsets)
    new_src_pos, new_dst_pos = _edge_endpoint_positions(new_pos, sel_edges, edge_offsets=sel_offsets)

    old_cost = sel_weights * (
        torch.abs(old_src_pos[:, 0] - old_dst_pos[:, 0])
        + torch.abs(old_src_pos[:, 1] - old_dst_pos[:, 1])
    )
    new_cost = sel_weights * (
        torch.abs(new_src_pos[:, 0] - new_dst_pos[:, 0])
        + torch.abs(new_src_pos[:, 1] - new_dst_pos[:, 1])
    )

    norm = float(edge_weight.sum().item()) + 1e-9
    return float((new_cost.sum() - old_cost.sum()).item() / norm)

def _coarse_density_penalty(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    grid_size: int = 12,
) -> float:
    if positions.numel() == 0:
        return 0.0

    areas = sizes[:, 0] * sizes[:, 1]
    gx = torch.clamp((positions[:, 0] / max(canvas_w, 1e-6) * grid_size).long(), 0, grid_size - 1)
    gy = torch.clamp((positions[:, 1] / max(canvas_h, 1e-6) * grid_size).long(), 0, grid_size - 1)

    idx = gy * grid_size + gx
    occ = torch.zeros(grid_size * grid_size, dtype=positions.dtype, device=positions.device)
    occ.scatter_add_(0, idx, areas)

    cell_area = (canvas_w * canvas_h) / max(grid_size * grid_size, 1)
    overflow = torch.relu(occ / max(cell_area, 1e-9) - 1.0)
    return float((overflow * overflow).mean().item())


def _coarse_congestion_penalty(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    edge_offsets: Optional[torch.Tensor] = None,
    grid_size: int = 14,
) -> float:
    if edge_index.numel() == 0:
        return 0.0

    src_pos, dst_pos = _edge_endpoint_positions(positions, edge_index, edge_offsets=edge_offsets)
    mid = 0.5 * (src_pos + dst_pos)
    span = torch.abs(src_pos[:, 0] - dst_pos[:, 0]) + torch.abs(src_pos[:, 1] - dst_pos[:, 1])
    demand = edge_weight * span

    gx = torch.clamp((mid[:, 0] / max(canvas_w, 1e-6) * grid_size).long(), 0, grid_size - 1)
    gy = torch.clamp((mid[:, 1] / max(canvas_h, 1e-6) * grid_size).long(), 0, grid_size - 1)

    idx = gy * grid_size + gx
    occ = torch.zeros(grid_size * grid_size, dtype=positions.dtype, device=positions.device)
    occ.scatter_add_(0, idx, demand)

    k = max(1, int(0.05 * occ.numel()))
    top = torch.topk(occ, k).values
    return float((top.mean() / (occ.mean() + 1e-9)).item())


def surrogate_wirelength(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros((), device=positions.device)

    src_pos, dst_pos = _edge_endpoint_positions(positions, edge_index, edge_offsets=edge_offsets)
    dx = _smooth_l1(src_pos[:, 0] - dst_pos[:, 0])
    dy = _smooth_l1(src_pos[:, 1] - dst_pos[:, 1])
    return (edge_weight * (dx + dy)).mean()


def _grid_centers(
    canvas_w: float,
    canvas_h: float,
    grid_size: int,
    device: torch.device,
) -> torch.Tensor:
    xs = torch.linspace(0.0, float(canvas_w), grid_size, device=device)
    ys = torch.linspace(0.0, float(canvas_h), grid_size, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


def surrogate_density(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    grid_size: int,
) -> torch.Tensor:
    if positions.numel() == 0:
        return torch.zeros((), device=positions.device)

    centers = _grid_centers(canvas_w, canvas_h, grid_size, positions.device)
    areas = sizes[:, 0] * sizes[:, 1]

    sigma_x = torch.clamp(sizes[:, 0] * 0.75, min=max(canvas_w / (grid_size * 4.0), 1e-3))
    sigma_y = torch.clamp(sizes[:, 1] * 0.75, min=max(canvas_h / (grid_size * 4.0), 1e-3))

    dx = (positions[:, 0:1] - centers[:, 0].unsqueeze(0)) / sigma_x.unsqueeze(1)
    dy = (positions[:, 1:2] - centers[:, 1].unsqueeze(0)) / sigma_y.unsqueeze(1)
    influence = torch.exp(-0.5 * (dx * dx + dy * dy))

    cell_area = (canvas_w * canvas_h) / max(1, grid_size * grid_size)
    density = (influence * areas.unsqueeze(1)).sum(dim=0) / max(cell_area, 1e-9)

    overflow = torch.relu(density - 1.0)
    tail = torch.quantile(density, 0.90)
    return overflow.pow(2).mean() + 0.2 * tail


def surrogate_congestion(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    grid_size: int,
    edge_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros((), device=positions.device)

    src_pos, dst_pos = _edge_endpoint_positions(positions, edge_index, edge_offsets=edge_offsets)
    mid = 0.5 * (src_pos + dst_pos)
    span = _smooth_l1(src_pos[:, 0] - dst_pos[:, 0]) + _smooth_l1(src_pos[:, 1] - dst_pos[:, 1])
    demand = edge_weight * span

    centers = _grid_centers(canvas_w, canvas_h, grid_size, positions.device)
    sigma = max(max(canvas_w, canvas_h) / max(grid_size * 3.0, 1.0), 1e-3)

    dx = (mid[:, 0:1] - centers[:, 0].unsqueeze(0)) / sigma
    dy = (mid[:, 1:2] - centers[:, 1].unsqueeze(0)) / sigma
    kernel = torch.exp(-0.5 * (dx * dx + dy * dy))

    cong_map = (kernel * demand.unsqueeze(1)).sum(dim=0)
    k = max(1, int(0.05 * cong_map.numel()))
    top = torch.topk(cong_map, k).values
    return top.mean() / (cong_map.mean() + 1e-9)


def surrogate_overlap(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    gap: float = 1e-4,
) -> torch.Tensor:
    n = int(positions.shape[0])
    if n <= 1:
        return torch.zeros((), device=positions.device)

    widths = sizes[:, 0]
    heights = sizes[:, 1]

    dx = _smooth_l1(positions[:, 0].unsqueeze(1) - positions[:, 0].unsqueeze(0))
    dy = _smooth_l1(positions[:, 1].unsqueeze(1) - positions[:, 1].unsqueeze(0))

    sep_x = (widths.unsqueeze(1) + widths.unsqueeze(0)) * 0.5 + gap
    sep_y = (heights.unsqueeze(1) + heights.unsqueeze(0)) * 0.5 + gap

    ovx = torch.relu(sep_x - dx)
    ovy = torch.relu(sep_y - dy)

    mask = torch.triu(torch.ones_like(ovx, dtype=torch.bool), diagonal=1)
    if not bool(mask.any()):
        return torch.zeros((), device=positions.device)

    ov = ovx * ovy
    return ov[mask].mean()


def _load_plc_for_benchmark(benchmark_name: str):
    if load_benchmark is None or load_benchmark_from_dir is None:
        return None
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark_name
    if root.exists() and (root / "netlist.pb.txt").exists():
        try:
            _, plc = load_benchmark_from_dir(str(root))
            return plc
        except Exception:
            return None

    aliases = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
    }

    base = aliases.get(benchmark_name, benchmark_name)
    ng45_dir = Path("external/MacroPlacement/Flows/NanGate45") / base / "netlist" / "output_CT_Grouping"
    if (ng45_dir / "netlist.pb.txt").exists():
        try:
            _, plc = load_benchmark(
                str(ng45_dir / "netlist.pb.txt"),
                str(ng45_dir / "initial.plc"),
            )
            return plc
        except Exception:
            return None

    return None


class WorkerState:
    def __init__(
        self,
        hard_pos: torch.Tensor,
        approx_score: float,
        exact_score: float = float("inf"),
        full_placement: Optional[torch.Tensor] = None,
    ) -> None:
        self.hard_pos = hard_pos
        self.approx_score = float(approx_score)
        self.exact_score = float(exact_score)
        self.full_placement = full_placement


def fallback_shelfpack_hard_macros(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    fixed_mask: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    num_hard_macros: int,
    gap: float = 1e-4,
) -> torch.Tensor:
    """Emergency legal fallback when iterative legalizer fails."""
    out = positions.clone()
    n = int(num_hard_macros)
    if n <= 1:
        return out

    hard_fixed = fixed_mask[:n]
    if bool(hard_fixed.any()):
        return out

    movable = list(range(n))
    movable.sort(key=lambda i: -float(sizes[i, 1].item()))

    cursor_x = 0.0
    cursor_y = 0.0
    row_h = 0.0

    for idx in movable:
        w = float(sizes[idx, 0].item())
        h = float(sizes[idx, 1].item())

        if cursor_x + w > canvas_width and cursor_x > 0.0:
            cursor_x = 0.0
            cursor_y += row_h + gap
            row_h = 0.0

        if cursor_y + h > canvas_height:
            out[idx, 0] = w * 0.5
            out[idx, 1] = h * 0.5
            continue

        out[idx, 0] = cursor_x + w * 0.5
        out[idx, 1] = cursor_y + h * 0.5

        cursor_x += w + gap
        row_h = max(row_h, h)

    out[:n] = project_to_canvas(out[:n], sizes[:n], canvas_width, canvas_height)
    return out


def apply_rmp_kick(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    fixed_mask: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    num_hard_macros: int,
    kick_cfg: Optional[Dict[str, object]] = None,
    fixed_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply a resonant-magnetic-perturbation style initialization kick.

    The kick is a smooth radial/tangential displacement pattern centered on the
    canvas, used to break symmetry and explore alternate plasma equilibria.
    """
    out = positions.clone()
    n = int(num_hard_macros)
    if n <= 0 or kick_cfg is None:
        return out

    amp_frac = float(kick_cfg.get("amplitude_frac", 0.0))
    if amp_frac <= 0.0:
        return out

    device = out.device
    hard = out[:n].clone()
    hard_fixed = fixed_mask[:n].clone()
    movable = ~hard_fixed
    if not bool(movable.any()):
        return out

    center = torch.tensor(
        [0.5 * float(canvas_width), 0.5 * float(canvas_height)],
        dtype=hard.dtype,
        device=device,
    )
    half = torch.tensor(
        [max(0.5 * float(canvas_width), 1e-6), max(0.5 * float(canvas_height), 1e-6)],
        dtype=hard.dtype,
        device=device,
    )
    rel = (hard - center.unsqueeze(0)) / half.unsqueeze(0)
    r = torch.linalg.norm(rel, dim=1)
    theta = torch.atan2(rel[:, 1], rel[:, 0] + 1e-12)

    radial_mode = float(kick_cfg.get("radial_mode", 1.0))
    poloidal_mode = float(kick_cfg.get("poloidal_mode", 2.0))
    phase = float(kick_cfg.get("phase", 0.0))
    resonance_radius = float(kick_cfg.get("resonance_radius", 0.62))
    resonance_width = max(float(kick_cfg.get("resonance_width", 0.28)), 1e-3)
    tangential_ratio = float(kick_cfg.get("tangential_ratio", 0.35))
    envelope_power = max(float(kick_cfg.get("envelope_power", 1.0)), 0.25)
    span = max(float(canvas_width), float(canvas_height))
    amplitude = amp_frac * span

    env = torch.exp(-0.5 * ((r - resonance_radius) / resonance_width) ** 2)
    env = torch.clamp(env, min=0.0, max=1.0).pow(envelope_power)

    radial_u = rel / torch.clamp(r.unsqueeze(1), min=1e-6)
    tangential_u = torch.stack([-radial_u[:, 1], radial_u[:, 0]], dim=1)
    pattern = radial_mode * r * math.pi + poloidal_mode * theta + phase

    radial_amp = amplitude * env * torch.sin(pattern)
    tangential_amp = amplitude * tangential_ratio * env * torch.cos(pattern)
    delta = radial_u * radial_amp.unsqueeze(1) + tangential_u * tangential_amp.unsqueeze(1)
    delta[~movable] = 0.0

    hard = hard + delta
    hard = project_to_canvas(
        hard,
        sizes[:n],
        canvas_width,
        canvas_height,
        fixed_mask=hard_fixed,
        fixed_positions=fixed_positions[:n] if fixed_positions is not None else positions[:n].clone(),
    )
    out[:n] = hard
    return out


class TeamPlasmaPlacer:
    def __init__(
        self,
        config_path: Optional[str] = None,
        config_overrides: Optional[Dict] = None,
        seed: Optional[int] = None,
    ):
        self.cfg = load_team_plasma_config(config_path)

        if config_overrides:
            self.cfg = self._deep_update(self.cfg, config_overrides)

        if seed is not None:
            self.cfg["seed"] = int(seed)

        self.mode = self.cfg.get("mode", "local")
        self.seed = int(self.cfg.get("seed", 42))
        self._plc_cache: Dict[str, object] = {}
        self._exact_used_calls: int = 0
        self._exact_max_calls: int = 0
        self._exact_allow_midrun: bool = False
        self._active_bundle_terms: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = (
            torch.zeros(1, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, 2, dtype=torch.float32),
            torch.zeros(0, dtype=torch.float32),
        )
        self._active_pin_terms: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, 2, dtype=torch.float32),
            torch.zeros(0, dtype=torch.float32),
        )
        self._active_pin_edge_profile: torch.Tensor = torch.zeros(0, 4, dtype=torch.float32)

    def _fresh_plc_cache_entry(self, benchmark_name: str) -> Dict[str, object]:
        plc = self._plc_cache.get(benchmark_name)
        if plc is not None:
            try:
                return {benchmark_name: copy.deepcopy(plc)}
            except Exception:
                pass

        fresh = _load_plc_for_benchmark(benchmark_name)
        if fresh is not None:
            return {benchmark_name: fresh}

        if plc is not None:
            return {benchmark_name: plc}
        return {}

    def _prepare_surrogate_terms(
        self,
        benchmark: Benchmark,
        placement: torch.Tensor,
    ) -> Tuple[object, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        plc = self._plc_cache.get(benchmark.name)
        if plc is None:
            plc = _load_plc_for_benchmark(benchmark.name)
            if plc is not None:
                self._plc_cache[benchmark.name] = plc

        max_surrogate_edges = int(self.cfg["global_stage"].get("max_surrogate_edges", 12000))
        pin_flux_enabled = bool(self.cfg.get("global_stage", {}).get("pde", {}).get("pin_flux_tubes", False))
        pin_flux_filter = dict(self.cfg.get("global_stage", {}).get("pin_flux_edge_filter", {}))
        if pin_flux_enabled:
            edge_index, edge_weight, edge_offsets = extract_pin_flux_tubes_from_plc(
                benchmark,
                plc,
                max_edges=max_surrogate_edges,
                min_degree=int(pin_flux_filter.get("min_degree", 2)),
                max_degree=int(pin_flux_filter.get("max_degree", 0)),
                weight_mode=str(pin_flux_filter.get("weight_mode", "unit_net")),
                degree_power=float(pin_flux_filter.get("degree_power", 1.0)),
            )
        else:
            edge_index, edge_weight = extract_hard_edges_from_plc(
                benchmark,
                plc,
                max_edges=max_surrogate_edges,
            )
            edge_offsets = torch.zeros((int(edge_index.shape[0]), 2, 2), dtype=torch.float32)
        knn_cfg = self.cfg.get("global_stage", {}).get("knn_edges", {})
        knn_enabled = bool(knn_cfg.get("enabled", True))
        knn_k = int(knn_cfg.get("k", 6))
        knn_weight = float(knn_cfg.get("weight", 0.25))
        num_hard = int(benchmark.num_hard_macros)
        hard_pos = placement[:num_hard]
        if edge_index.numel() == 0:
            edge_index, edge_weight = build_knn_edges(hard_pos, k=max(1, knn_k))
            edge_offsets = torch.zeros((int(edge_index.shape[0]), 2, 2), dtype=torch.float32)
        elif knn_enabled:
            knn_idx, knn_w = build_knn_edges(hard_pos, k=max(1, knn_k))
            knn_offsets = torch.zeros((int(knn_idx.shape[0]), 2, 2), dtype=torch.float32)
            edge_index, edge_weight, edge_offsets = merge_edge_sets_with_offsets(
                edge_index,
                edge_weight,
                edge_offsets,
                knn_idx,
                knn_w,
                knn_offsets,
                scale_b=knn_weight,
            )
        anchor_targets, anchor_weights = build_hard_anchor_targets_from_plc(
            benchmark,
            plc,
            use_pin_positions=pin_flux_enabled,
        )
        anchor_targets, anchor_weights = filter_hard_anchor_targets(
            anchor_targets,
            anchor_weights,
            dict(self.cfg.get("global_stage", {}).get("anchor_filter", {})),
        )
        bundle_cfg = dict(self.cfg.get("global_stage", {}).get("bundle_nets", {}))
        if bool(bundle_cfg.get("enabled", False)):
            bundle_terms = extract_hard_net_bundles_from_plc(
                benchmark,
                plc,
                max_nets=int(bundle_cfg.get("max_nets", 6000)),
                min_degree=int(bundle_cfg.get("min_degree", 3)),
                use_pin_offsets=bool(bundle_cfg.get("use_pin_offsets", pin_flux_enabled)),
                weight_mode=str(bundle_cfg.get("weight_mode", "inverse_degree")),
                degree_power=float(bundle_cfg.get("degree_power", 1.0)),
            )
        else:
            bundle_terms = (
                torch.zeros(1, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, 2, dtype=torch.float32),
                torch.zeros(0, dtype=torch.float32),
            )
        soft_cluster_bundle_cfg = dict(self.cfg.get("global_stage", {}).get("soft_cluster_bundles", {}))
        if bool(soft_cluster_bundle_cfg.get("enabled", False)):
            cluster_terms = extract_soft_cluster_bundles_from_plc(
                benchmark,
                plc,
                max_clusters=int(soft_cluster_bundle_cfg.get("max_clusters", 24)),
                min_hard_degree=int(soft_cluster_bundle_cfg.get("min_hard_degree", 2)),
                use_pin_offsets=bool(soft_cluster_bundle_cfg.get("use_pin_offsets", pin_flux_enabled)),
                weight_mode=str(soft_cluster_bundle_cfg.get("weight_mode", "soft_mass_inverse_degree")),
                degree_power=float(soft_cluster_bundle_cfg.get("degree_power", 1.0)),
            )
            bundle_terms = merge_bundle_terms(bundle_terms, cluster_terms)
        soft_cluster_edge_cfg = dict(self.cfg.get("global_stage", {}).get("soft_cluster_edges", {}))
        if bool(soft_cluster_edge_cfg.get("enabled", False)):
            cluster_edge_index, cluster_edge_weight, cluster_edge_offsets = extract_soft_cluster_edges_from_plc(
                benchmark,
                plc,
                max_edges=int(soft_cluster_edge_cfg.get("max_edges", max_surrogate_edges)),
                min_hard_degree=int(soft_cluster_edge_cfg.get("min_hard_degree", 2)),
                min_shared_clusters=int(soft_cluster_edge_cfg.get("min_shared_clusters", 1)),
                use_pin_offsets=bool(soft_cluster_edge_cfg.get("use_pin_offsets", pin_flux_enabled)),
                weight_mode=str(soft_cluster_edge_cfg.get("weight_mode", "soft_mass_inverse_degree")),
                degree_power=float(soft_cluster_edge_cfg.get("degree_power", 1.0)),
            )
            edge_index, edge_weight, edge_offsets = merge_edge_sets_with_offsets(
                edge_index,
                edge_weight,
                edge_offsets,
                cluster_edge_index,
                cluster_edge_weight,
                cluster_edge_offsets,
                scale_b=float(soft_cluster_edge_cfg.get("scale", 1.0)),
            )
        pin_pressure_cfg = dict(self.cfg.get("global_stage", {}).get("pin_pressure", {}))
        if bool(pin_pressure_cfg.get("enabled", False)):
            pin_terms = extract_hard_pin_pressure_terms_from_plc(
                benchmark,
                plc,
                min_degree=int(pin_pressure_cfg.get("min_degree", 2)),
                max_degree=int(pin_pressure_cfg.get("max_degree", 0)),
                weight_mode=str(pin_pressure_cfg.get("weight_mode", "inverse_sqrt_degree")),
                degree_power=float(pin_pressure_cfg.get("degree_power", 1.0)),
            )
        else:
            pin_terms = (
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, 2, dtype=torch.float32),
                torch.zeros(0, dtype=torch.float32),
            )
        pin_sheath_cfg = dict(self.cfg.get("global_stage", {}).get("pin_sheath", {}))
        if bool(pin_sheath_cfg.get("enabled", False)):
            pin_edge_profile = extract_hard_pin_edge_profile_from_plc(
                benchmark,
                plc,
                min_degree=int(pin_sheath_cfg.get("min_degree", 2)),
                max_degree=int(pin_sheath_cfg.get("max_degree", 0)),
                weight_mode=str(pin_sheath_cfg.get("weight_mode", "inverse_sqrt_degree")),
                degree_power=float(pin_sheath_cfg.get("degree_power", 1.0)),
            )
        else:
            pin_edge_profile = torch.zeros((int(benchmark.num_hard_macros), 4), dtype=torch.float32)
        self._active_bundle_terms = bundle_terms
        self._active_pin_terms = pin_terms
        self._active_pin_edge_profile = pin_edge_profile
        return plc, edge_index, edge_weight, edge_offsets, anchor_targets, anchor_weights, bundle_terms

    def _build_portfolio_child(
        self,
        benchmark_name: str,
        entry: Dict[str, object],
        seed_offset_default: int = 0,
    ) -> Tuple["TeamPlasmaPlacer", Dict[str, object]]:
        overrides = dict(entry.get("overrides", {})) if isinstance(entry, dict) else {}
        seed_offset = int(entry.get("seed_offset", seed_offset_default)) if isinstance(entry, dict) else seed_offset_default

        if isinstance(entry, dict) and entry.get("config_path"):
            base_cfg = load_team_plasma_config(str(entry["config_path"]))
        else:
            base_cfg = copy.deepcopy(self.cfg)

        merged_cfg = self._deep_update(copy.deepcopy(base_cfg), overrides)
        merged_cfg["router"] = {"enabled": False, "prototypes": [], "rules": []}
        merged_cfg["portfolio"] = {"enabled": False, "entries": []}
        merged_cfg["benchmark_isolation"] = {"enabled": False}
        merged_cfg["_portfolio_child"] = True

        child = TeamPlasmaPlacer(config_overrides=merged_cfg, seed=self.seed + seed_offset)
        child._plc_cache = self._fresh_plc_cache_entry(benchmark_name)
        return child, merged_cfg

    def _resolve_multi_start_bases(
        self,
        benchmark: Benchmark,
        placement: torch.Tensor,
        multi_cfg: Dict[str, object],
    ) -> List[torch.Tensor]:
        source_mode = str(multi_cfg.get("source", "initial")).lower()
        candidates: List[torch.Tensor] = []

        def _append_unique(candidate: torch.Tensor) -> None:
            cand = candidate.clone().float()
            for existing in candidates:
                if cand.shape == existing.shape and torch.allclose(cand, existing, atol=1e-9, rtol=0.0):
                    return
            candidates.append(cand)

        if source_mode in ("current", "both"):
            _append_unique(placement)
        if source_mode in ("initial", "both"):
            _append_unique(benchmark.macro_positions)

        if not candidates:
            _append_unique(benchmark.macro_positions)

        return candidates

    def _run_isolated_place(self, benchmark: Benchmark) -> torch.Tensor:
        bench_name = str(getattr(benchmark, "name", "") or "").strip()
        if not bench_name:
            raise RuntimeError("benchmark isolation requires a benchmark with a stable name")

        root_dir = _THIS_DIR.parent.parent
        iso_cfg = dict(self.cfg.get("benchmark_isolation", {}))
        work_dir = Path(str(iso_cfg.get("work_dir", "output/team_plasma/_isolation_tmp")))
        if not work_dir.is_absolute():
            work_dir = (root_dir / work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{bench_name}_{os.getpid()}_{time.time_ns()}"
        cfg_path = work_dir / f"{stem}_config.json"
        out_path = work_dir / f"{stem}_placement.pt"

        child_cfg = copy.deepcopy(self.cfg)
        child_cfg["_subprocess_child"] = True
        child_cfg["benchmark_isolation"] = {"enabled": False}
        cfg_path.write_text(json.dumps(child_cfg, indent=2), encoding="utf-8")

        worker = _THIS_DIR / "isolation_worker.py"
        cmd = [
            sys.executable,
            str(worker),
            "--benchmark",
            bench_name,
            "--config",
            str(cfg_path),
            "--seed",
            str(self.seed),
            "--output",
            str(out_path),
        ]

        try:
            subprocess.run(cmd, check=True, cwd=str(root_dir))
            placement = torch.load(out_path, map_location="cpu")
            if not isinstance(placement, torch.Tensor):
                raise RuntimeError(f"isolation worker produced non-tensor output for {bench_name}")
            return placement.float()
        finally:
            for path in (cfg_path, out_path):
                try:
                    if path.exists():
                        path.unlink()
                except Exception:
                    pass

    @staticmethod
    def _deep_update(base: Dict, updates: Dict) -> Dict:
        merged = dict(base)
        for k, v in updates.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = TeamPlasmaPlacer._deep_update(merged[k], v)
            else:
                merged[k] = v
        return merged

    @staticmethod
    def _resolve_legal_anchor_positions(
        benchmark: Benchmark,
        reference_placement: Optional[torch.Tensor],
        anchor_mode: str,
    ) -> Optional[torch.Tensor]:
        mode = str(anchor_mode).lower()
        if mode == "initial":
            return benchmark.macro_positions.clone().float()
        if mode == "current":
            if reference_placement is None:
                return None
            return reference_placement.clone().float()
        return None

    def _repair_snapshot_for_exact(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
    ) -> torch.Tensor:
        n = int(benchmark.num_hard_macros)
        if n <= 1:
            return placement

        legal_cfg = self.cfg["legalizer"]
        snap_cfg = dict(self.cfg.get("global_stage", {}).get("exact_snapshot_repair", {}))
        if not bool(snap_cfg.get("enabled", True)):
            return placement

        gap = float(snap_cfg.get("gap", legal_cfg.get("gap", 1e-4)))
        anchor_mode = str(snap_cfg.get("anchor_mode", legal_cfg.get("anchor_mode", "current"))).lower()
        anchor_strength = float(
            snap_cfg.get("anchor_strength", max(0.08, float(legal_cfg.get("anchor_strength", 0.0))))
        )
        restore_iters = int(
            snap_cfg.get("restore_iters", max(2, int(legal_cfg.get("restore_iters", 0))))
        )
        anchor_positions = self._resolve_legal_anchor_positions(benchmark, placement, anchor_mode)

        repaired = legalize_hard_macros(
            placement,
            benchmark.macro_sizes,
            benchmark.macro_fixed,
            benchmark.canvas_width,
            benchmark.canvas_height,
            n,
            gap=gap,
            max_iters=int(snap_cfg.get("max_iters", min(50, int(legal_cfg.get("max_iters", 80))))),
            fallback_iters=int(
                snap_cfg.get("fallback_iters", min(30, int(legal_cfg.get("fallback_iters", 60))))
            ),
            max_pairs_per_iter=int(snap_cfg.get("max_pairs_per_iter", 10000)),
            anchor_positions=anchor_positions,
            anchor_strength=anchor_strength,
            restore_iters=restore_iters,
        )
        repaired = sanitize_canvas_bounds(
            repaired,
            benchmark.macro_sizes,
            benchmark.canvas_width,
            benchmark.canvas_height,
            fixed_mask=benchmark.macro_fixed,
            fixed_positions=benchmark.macro_positions,
            safety_eps=1e-6,
        )

        if bool(snap_cfg.get("strict_if_needed", True)) and count_hard_overlaps(repaired, benchmark.macro_sizes, n, gap=gap) > 0:
            repaired = strict_legalize_hard_macros(
                repaired,
                benchmark.macro_sizes,
                benchmark.macro_fixed,
                benchmark.canvas_width,
                benchmark.canvas_height,
                n,
                gap=gap,
                max_iters=int(
                    snap_cfg.get("strict_max_iters", min(80, int(legal_cfg.get("max_iters", 80))))
                ),
                fallback_iters=int(
                    snap_cfg.get(
                        "strict_fallback_iters",
                        min(60, int(legal_cfg.get("fallback_iters", 60))),
                    )
                ),
                max_pairs_per_iter=int(snap_cfg.get("strict_max_pairs_per_iter", 12000)),
                anchor_positions=anchor_positions,
                anchor_strength=anchor_strength,
                restore_iters=max(restore_iters, int(snap_cfg.get("strict_restore_iters", restore_iters))),
            )
            repaired = sanitize_canvas_bounds(
                repaired,
                benchmark.macro_sizes,
                benchmark.canvas_width,
                benchmark.canvas_height,
                fixed_mask=benchmark.macro_fixed,
                fixed_positions=benchmark.macro_positions,
                safety_eps=1e-6,
            )

        return repaired

    def _pilot_plasma_candidate(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_offsets: torch.Tensor,
        anchor_targets: torch.Tensor,
        anchor_weights: torch.Tensor,
        pilot_cfg: Dict[str, object],
    ) -> Tuple[torch.Tensor, float]:
        n = int(benchmark.num_hard_macros)
        if n <= 1:
            return placement, 0.0

        steps = max(0, int(pilot_cfg.get("steps", 0)))
        if steps <= 0:
            return placement, 0.0

        device = self._resolve_device(str(self.cfg.get("device", "auto")))
        hard_pos = placement[:n].to(device)
        hard_sizes = benchmark.macro_sizes[:n].to(device)
        hard_fixed = benchmark.macro_fixed[:n].to(device)
        hard_fixed_pos = hard_pos.clone()
        soft_pos = placement[n:].to(device) if int(placement.shape[0]) > n else torch.zeros(0, 2, device=device)
        soft_sizes = benchmark.macro_sizes[n:].to(device) if int(benchmark.macro_sizes.shape[0]) > n else torch.zeros(0, 2, device=device)
        edge_index_d = edge_index.to(device)
        edge_weight_d = edge_weight.to(device)
        edge_offsets_d = edge_offsets.to(device) if edge_offsets.numel() > 0 else torch.zeros(0, 2, 2, dtype=torch.float32, device=device)
        anchor_targets_d = anchor_targets.to(device)
        anchor_weights_d = anchor_weights.to(device)
        precond_cfg = dict(self.cfg.get("global_stage", {}).get("preconditioner", {}))
        move_precond = compute_transport_preconditioner(
            hard_sizes,
            edge_index_d,
            edge_weight_d,
            precond_cfg,
        )

        stages = self.cfg.get("global_stage", {}).get("stages", [])
        stage0 = dict(stages[0]) if stages else {}
        rhs_weights = dict(stage0.get("rhs_weights", {}))
        rhs_weights.update(dict(pilot_cfg.get("rhs_weights", {})))
        force_weights = dict(stage0.get("force_weights", {}))
        force_weights.update(dict(pilot_cfg.get("force_weights", {})))
        grid_size = int(pilot_cfg.get("grid_size", stage0.get("grid_size", 16)))
        lr = float(pilot_cfg.get("lr", stage0.get("lr", 0.05)))
        trust_radius = float(pilot_cfg.get("trust_radius_frac", stage0.get("trust_radius_frac", 0.04))) * max(
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
        )
        pde_cfg = dict(self.cfg.get("global_stage", {}).get("pde", {}))
        pde_cfg.update(dict(pilot_cfg.get("pde", {})))
        sheath_cfg = dict(self.cfg.get("global_stage", {}).get("pin_sheath", {}))
        sheath_cfg.update(dict(pilot_cfg.get("pin_sheath", {})))
        psi_cache: Optional[torch.Tensor] = None

        for _ in range(steps):
            fields = compute_plasma_forces(
                positions=hard_pos,
                sizes=hard_sizes,
                edge_index=edge_index_d,
                edge_weight=edge_weight_d,
                canvas_width=benchmark.canvas_width,
                canvas_height=benchmark.canvas_height,
                grid_size=grid_size,
                pde_cfg=pde_cfg,
                rhs_weights=rhs_weights,
                background_positions=soft_pos,
                background_sizes=soft_sizes,
                anchor_targets=anchor_targets_d,
                anchor_weights=anchor_weights_d,
                edge_offsets=edge_offsets_d,
                previous_psi=psi_cache,
                bundle_terms=self._active_bundle_terms,
                pin_terms=self._active_pin_terms,
                pin_edge_profile=self._active_pin_edge_profile,
            )
            channel_force = compute_pin_edge_sheath_force(
                hard_pos,
                hard_sizes,
                self._active_pin_edge_profile,
                sheath_cfg,
                benchmark.canvas_width,
                benchmark.canvas_height,
            )
            psi_cache = fields["psi"]
            total_force = (
                float(force_weights.get("plasma", 1.0)) * fields["plasma_force"]
                + float(force_weights.get("hall", 0.0)) * fields.get("hall_force", torch.zeros_like(hard_pos))
                + float(force_weights.get("chi", 0.0)) * fields.get("chi_force", torch.zeros_like(hard_pos))
                + float(force_weights.get("dia", 0.0)) * fields.get("dia_force", torch.zeros_like(hard_pos))
                + float(force_weights.get("balloon", 0.0)) * fields.get("balloon_force", torch.zeros_like(hard_pos))
                + float(force_weights.get("hot", 0.0)) * fields.get("hot_force", torch.zeros_like(hard_pos))
                + float(force_weights.get("cold", 0.0)) * fields.get("cold_force", torch.zeros_like(hard_pos))
                + float(force_weights.get("rho_pressure", 0.0)) * fields.get("rho_force", torch.zeros_like(hard_pos))
                + float(force_weights.get("q_pressure", 0.0)) * fields.get("q_force", torch.zeros_like(hard_pos))
                + float(force_weights.get("pin_pressure", 0.0)) * fields.get("pin_force", torch.zeros_like(hard_pos))
                + float(force_weights.get("pin_channel", 0.0)) * channel_force
                + float(force_weights.get("bg_pressure", 0.0)) * fields.get("bg_force", torch.zeros_like(hard_pos))
                + float(force_weights.get("porosity_pressure", 0.0)) * fields.get("porosity_force", torch.zeros_like(hard_pos))
                + float(force_weights.get("net", 0.3)) * fields["net_force"]
                + float(force_weights.get("repulsion", 0.2)) * fields["repulsion_force"]
            )
            if hard_fixed.any():
                total_force[hard_fixed] = 0.0

            step_vec = lr * total_force / torch.clamp(move_precond, min=1e-6)
            norms = torch.linalg.norm(step_vec, dim=1, keepdim=True)
            if trust_radius > 0.0:
                step_vec = step_vec * torch.clamp(trust_radius / torch.clamp(norms, min=1e-9), max=1.0)

            hard_pos = hard_pos + step_vec
            hard_pos = project_to_canvas(
                hard_pos,
                hard_sizes,
                benchmark.canvas_width,
                benchmark.canvas_height,
                fixed_mask=hard_fixed,
                fixed_positions=hard_fixed_pos,
            )

        out = placement.clone()
        out[:n] = hard_pos.detach().cpu()
        score = self._approximate_total_cost(
            out[:n],
            benchmark.macro_sizes[:n].float(),
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
            edge_index,
            edge_weight,
            edge_offsets=edge_offsets,
            anchor_targets=anchor_targets,
            anchor_weights=anchor_weights,
        )
        return out, score

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        set_random_seed(self.seed)

        isolation_cfg = self.cfg.get("benchmark_isolation", {})
        if (
            bool(isolation_cfg.get("enabled", False))
            and not bool(self.cfg.get("_subprocess_child", False))
            # Routed/portfolio children already run inside the parent placement call.
            and not bool(self.cfg.get("_router_child", False))
            and not bool(self.cfg.get("_portfolio_child", False))
        ):
            return self._run_isolated_place(benchmark)

        router_cfg = self.cfg.get("router", {})
        if bool(router_cfg.get("enabled", False)) and not bool(self.cfg.get("_router_child", False)):
            return self._run_feature_router(benchmark)

        portfolio_cfg = self.cfg.get("portfolio", {})
        if bool(portfolio_cfg.get("enabled", False)) and not bool(self.cfg.get("_portfolio_child", False)):
            return self._run_profile_portfolio(benchmark)

        placement = benchmark.macro_positions.clone().float()
        fixed_mask = benchmark.macro_fixed.clone()

        num_hard = int(benchmark.num_hard_macros)
        if num_hard <= 1:
            placement[fixed_mask] = benchmark.macro_positions[fixed_mask]
            return placement

        sizes = benchmark.macro_sizes.float()
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)

        initial_guard_cfg = self.cfg.get("initial_guard", {})
        if bool(initial_guard_cfg.get("enabled", False)):
            guard_valid = False
            if validate_placement is not None:
                try:
                    guard_valid, _ = validate_placement(placement, benchmark, check_overlaps=True)
                except TypeError:
                    guard_valid, _ = validate_placement(placement, benchmark)
            else:
                legal_gap = float(self.cfg.get("legalizer", {}).get("gap", 1e-4))
                guard_valid = count_hard_overlaps(placement, sizes, num_hard, gap=legal_gap) == 0
            if guard_valid and bool(initial_guard_cfg.get("return_initial_if_valid", True)):
                if fixed_mask.any():
                    placement[fixed_mask] = benchmark.macro_positions[fixed_mask]
                return placement

        init_cfg = self.cfg.get("initialization", {})
        jitter_frac = float(init_cfg.get("jitter_frac", 0.0))
        if jitter_frac > 0.0:
            span = max(canvas_w, canvas_h)
            sigma = jitter_frac * span
            hard_fixed = fixed_mask[:num_hard].clone()
            movable = torch.where(~hard_fixed)[0]
            if movable.numel() > 0:
                noise = torch.randn(int(movable.numel()), 2, dtype=placement.dtype) * float(sigma)
                placement[movable] = placement[movable] + noise
                placement[:num_hard] = project_to_canvas(
                    placement[:num_hard],
                    sizes[:num_hard],
                    canvas_w,
                    canvas_h,
                    fixed_mask=hard_fixed,
                    fixed_positions=benchmark.macro_positions[:num_hard],
                )
        if isinstance(init_cfg.get("rmp_kick", None), dict):
            placement = apply_rmp_kick(
                placement,
                sizes,
                fixed_mask,
                canvas_w,
                canvas_h,
                num_hard,
                kick_cfg=dict(init_cfg.get("rmp_kick", {})),
                fixed_positions=benchmark.macro_positions,
            )

        start_time = time.time()
        total_budget = self._time_budget_sec()
        exact_cfg = self.cfg.get("exact_eval", {})
        self._exact_max_calls = int(exact_cfg.get("max_calls", {}).get(self.mode, 1))
        self._exact_allow_midrun = bool(exact_cfg.get("allow_midrun", {}).get(self.mode, False))
        self._exact_used_calls = 0

        plc, edge_index, edge_weight, edge_offsets, anchor_targets, anchor_weights, _bundle_terms = self._prepare_surrogate_terms(
            benchmark,
            placement,
        )
        soft_transport_model = build_soft_transport_model_from_plc(benchmark, plc)
        soft_transport_cfg = dict(self.cfg.get("global_stage", {}).get("soft_transport", {}))
        soft_transport_model = filter_soft_transport_model(
            soft_transport_model,
            dict(soft_transport_cfg.get("filter", {})),
        )
        soft_transport_active = (
            bool(soft_transport_cfg.get("enabled", False))
            and int(benchmark.num_soft_macros) > 0
            and benchmark_soft_fill_ratio(benchmark) >= float(soft_transport_cfg.get("min_soft_fill", 0.45))
        )

        flux_cfg = dict(init_cfg.get("flux_seed", {}))
        if bool(flux_cfg.get("enabled", False)):
            flux_candidate = placement.clone()
            flux_candidate[:num_hard] = build_flux_surface_seed(
                placement[:num_hard],
                sizes[:num_hard],
                fixed_mask[:num_hard],
                canvas_w,
                canvas_h,
                edge_index,
                edge_weight,
                anchor_targets=anchor_targets,
                anchor_weights=anchor_weights,
                cfg=flux_cfg,
            )
            flux_candidate = sanitize_canvas_bounds(
                flux_candidate,
                sizes,
                canvas_w,
                canvas_h,
                fixed_mask=fixed_mask,
                fixed_positions=benchmark.macro_positions,
                safety_eps=1e-6,
            )

            selection = str(flux_cfg.get("selection", "force")).lower()
            if selection == "pilot_best":
                compare_cfg = dict(flux_cfg.get("pilot", {}))
                if bool(compare_cfg.get("enabled", False)):
                    base_eval, base_score = self._pilot_plasma_candidate(
                        placement.clone(),
                        benchmark,
                        edge_index,
                        edge_weight,
                        edge_offsets,
                        anchor_targets,
                        anchor_weights,
                        compare_cfg,
                    )
                    flux_eval, flux_score = self._pilot_plasma_candidate(
                        flux_candidate,
                        benchmark,
                        edge_index,
                        edge_weight,
                        edge_offsets,
                        anchor_targets,
                        anchor_weights,
                        compare_cfg,
                    )
                else:
                    base_eval = placement.clone()
                    flux_eval = flux_candidate
                    base_score = self._approximate_total_cost(
                        base_eval[:num_hard],
                        sizes[:num_hard],
                        canvas_w,
                        canvas_h,
                        edge_index,
                        edge_weight,
                        edge_offsets=edge_offsets,
                        anchor_targets=anchor_targets,
                        anchor_weights=anchor_weights,
                    )
                    flux_score = self._approximate_total_cost(
                        flux_eval[:num_hard],
                        sizes[:num_hard],
                        canvas_w,
                        canvas_h,
                        edge_index,
                        edge_weight,
                        edge_offsets=edge_offsets,
                        anchor_targets=anchor_targets,
                        anchor_weights=anchor_weights,
                    )

                min_gain = float(flux_cfg.get("min_improvement_frac", 0.0))
                if flux_score <= base_score * (1.0 - min_gain):
                    placement = flux_eval
                else:
                    placement = base_eval
            else:
                placement = flux_candidate

        multi_cfg = init_cfg.get("multi_start", {})
        if bool(multi_cfg.get("enabled", False)):
            hard_fixed = fixed_mask[:num_hard].clone()
            movable = torch.where(~hard_fixed)[0]
            span = max(canvas_w, canvas_h)
            pilot_cfg = dict(multi_cfg.get("pilot", {}))
            pilot_enabled = bool(pilot_cfg.get("enabled", False))
            base_candidates = self._resolve_multi_start_bases(benchmark, placement, multi_cfg)

            jitter_fracs = list(multi_cfg.get("jitter_fracs", []))
            trials = int(multi_cfg.get("trials", 0))
            rmp_kicks = list(multi_cfg.get("rmp_kicks", []))
            if not jitter_fracs and trials <= 0:
                jitter_fracs = [0.0, jitter_frac] if jitter_frac > 0.0 else [0.0]
            if trials > 0 and not jitter_fracs:
                jitter_fracs = [jitter_frac] * trials
            if not rmp_kicks:
                rmp_kicks = [None]

            best_score = float("inf")
            best_start = placement.clone()

            for base in base_candidates:
                for frac in jitter_fracs:
                    for kick in rmp_kicks:
                        sigma = float(frac) * span
                        candidate = base.clone()
                        if movable.numel() > 0 and sigma > 0.0:
                            noise = torch.randn(int(movable.numel()), 2, dtype=candidate.dtype) * sigma
                            candidate[movable] = candidate[movable] + noise
                        candidate[:num_hard] = project_to_canvas(
                            candidate[:num_hard],
                            sizes[:num_hard],
                            canvas_w,
                            canvas_h,
                            fixed_mask=hard_fixed,
                            fixed_positions=benchmark.macro_positions[:num_hard],
                        )
                        if isinstance(kick, dict):
                            candidate = apply_rmp_kick(
                                candidate,
                                sizes,
                                fixed_mask,
                                canvas_w,
                                canvas_h,
                                num_hard,
                                kick_cfg=kick,
                                fixed_positions=benchmark.macro_positions,
                            )

                        if pilot_enabled:
                            candidate, approx = self._pilot_plasma_candidate(
                                candidate,
                                benchmark,
                                edge_index,
                                edge_weight,
                                edge_offsets,
                                anchor_targets,
                                anchor_weights,
                                pilot_cfg,
                            )
                        else:
                            approx = self._approximate_total_cost(
                                candidate[:num_hard],
                                sizes[:num_hard],
                                canvas_w,
                                canvas_h,
                                edge_index,
                                edge_weight,
                                edge_offsets=edge_offsets,
                                anchor_targets=anchor_targets,
                                anchor_weights=anchor_weights,
                            )
                        if approx < best_score:
                            best_score = approx
                            best_start = candidate

            placement = best_start

        best_exact_cost = float("inf")
        best_exact_placement: Optional[torch.Tensor] = None

        if bool(self.cfg["global_stage"].get("enabled", True)):
            placement, maybe_cost = self._run_plasma_global_stage(
                placement,
                benchmark,
                plc,
                edge_index,
                edge_weight,
                edge_offsets,
                anchor_targets,
                anchor_weights,
                soft_transport_model,
                start_time,
                total_budget,
                allow_exact_snapshots=self._exact_allow_midrun,
            )
            placement = sanitize_canvas_bounds(
                placement,
                sizes,
                canvas_w,
                canvas_h,
                fixed_mask=fixed_mask,
                fixed_positions=benchmark.macro_positions,
                safety_eps=1e-6,
            )
            if maybe_cost is not None:
                best_exact_cost = maybe_cost
                best_exact_placement = placement.clone()

        legal_cfg = self.cfg["legalizer"]
        legal_anchor_mode = str(legal_cfg.get("anchor_mode", "current")).lower()
        legal_anchor_strength = float(legal_cfg.get("anchor_strength", 0.0))
        legal_restore_iters = int(legal_cfg.get("restore_iters", 0))
        placement = legalize_hard_macros(
            placement,
            sizes,
            fixed_mask,
            canvas_w,
            canvas_h,
            num_hard,
            gap=float(legal_cfg.get("gap", 1e-4)),
            max_iters=int(legal_cfg.get("max_iters", 80)),
            fallback_iters=int(legal_cfg.get("fallback_iters", 60)),
            anchor_positions=self._resolve_legal_anchor_positions(benchmark, placement, legal_anchor_mode),
            anchor_strength=legal_anchor_strength,
            restore_iters=legal_restore_iters,
        )

        placement = sanitize_canvas_bounds(
            placement,
            sizes,
            canvas_w,
            canvas_h,
            fixed_mask=fixed_mask,
            fixed_positions=benchmark.macro_positions,
            safety_eps=1e-6,
        )

        if soft_transport_active:
            placement = legalize_soft_vs_all(
                placement,
                sizes,
                fixed_mask,
                canvas_w,
                canvas_h,
                num_hard,
                gap=float(legal_cfg.get("gap", 1e-4)),
                max_iters=int(soft_transport_cfg.get("final_legalize_iters", 30)),
            )
            placement = sanitize_canvas_bounds(
                placement,
                sizes,
                canvas_w,
                canvas_h,
                fixed_mask=fixed_mask,
                fixed_positions=benchmark.macro_positions,
                safety_eps=1e-6,
            )

        if bool(self.cfg.get("post_legal_global", {}).get("enabled", False)):
            placement, post_cost = self._run_post_legal_plasma_stage(
                placement,
                benchmark,
                plc,
                edge_index,
                edge_weight,
                edge_offsets,
                anchor_targets,
                anchor_weights,
                start_time,
                total_budget,
            )
            placement = sanitize_canvas_bounds(
                placement,
                sizes,
                canvas_w,
                canvas_h,
                fixed_mask=fixed_mask,
                fixed_positions=benchmark.macro_positions,
                safety_eps=1e-6,
            )
            if post_cost is not None and post_cost < best_exact_cost:
                best_exact_cost = post_cost
                best_exact_placement = placement.clone()

        if fixed_mask.any():
            placement[fixed_mask] = benchmark.macro_positions[fixed_mask]

        exact_cost = self._consume_exact_eval(
            placement,
            benchmark,
            plc,
            force_final=False,
        )
        if exact_cost is not None and exact_cost < best_exact_cost:
            best_exact_cost = exact_cost
            best_exact_placement = placement.clone()

        if bool(self.cfg.get("exact_local_refine", {}).get("enabled", False)):
            placement, refine_best = self._run_exact_local_refine(
                placement,
                benchmark,
                plc,
                edge_index,
                edge_weight,
                anchor_targets,
                anchor_weights,
                start_time,
                total_budget,
                edge_offsets=edge_offsets,
            )
            placement = sanitize_canvas_bounds(
                placement,
                sizes,
                canvas_w,
                canvas_h,
                fixed_mask=fixed_mask,
                fixed_positions=benchmark.macro_positions,
                safety_eps=1e-6,
            )
            if refine_best is not None and refine_best < best_exact_cost:
                best_exact_cost = refine_best
                best_exact_placement = placement.clone()

        if bool(self.cfg["sa"].get("enabled", True)):
            placement, sa_best = self._run_plasma_guided_sa(
                placement,
                benchmark,
                plc,
                edge_index,
                edge_weight,
                anchor_targets,
                anchor_weights,
                start_time,
                total_budget,
                allow_exact_sync=self._exact_allow_midrun,
                edge_offsets=edge_offsets,
            )
            placement = sanitize_canvas_bounds(
                placement,
                sizes,
                canvas_w,
                canvas_h,
                fixed_mask=fixed_mask,
                fixed_positions=benchmark.macro_positions,
                safety_eps=1e-6,
            )
            if sa_best is not None and sa_best < best_exact_cost:
                best_exact_cost = sa_best
                best_exact_placement = placement.clone()

        run_soft_macro = bool(self.cfg["soft_macro"].get("enabled", True))
        soft_adaptive_cfg = self.cfg["soft_macro"].get("adaptive", {})
        if run_soft_macro and bool(soft_adaptive_cfg.get("enabled", False)):
            soft_fill = benchmark_soft_fill_ratio(benchmark)
            run_soft_macro = soft_fill >= float(soft_adaptive_cfg.get("min_soft_fill", 0.5))

        if run_soft_macro:
            soft_cfg = dict(self.cfg.get("soft_macro", {}))
            portfolio_cfg = dict(soft_cfg.get("portfolio", {}))
            if bool(portfolio_cfg.get("enabled", False)):
                soft_candidates: List[Tuple[str, torch.Tensor]] = []
                if bool(portfolio_cfg.get("include_noop", True)):
                    soft_candidates.append(("noop", placement.clone()))

                base_candidate = self._optimize_soft_macros(
                    placement,
                    benchmark,
                    plc,
                    num_steps=soft_cfg.get("final_num_steps", [40, 40, 40]),
                    use_current_loc=bool(soft_cfg.get("final_use_current_loc", True)),
                )
                soft_candidates.append(("base", base_candidate))

                for idx, strategy in enumerate(portfolio_cfg.get("strategies", [])):
                    if not isinstance(strategy, dict):
                        continue
                    name = str(strategy.get("name", f"variant_{idx}"))
                    cand = self._optimize_soft_macros(
                        placement,
                        benchmark,
                        plc,
                        num_steps=soft_cfg.get("final_num_steps", [40, 40, 40]),
                        use_current_loc=bool(soft_cfg.get("final_use_current_loc", True)),
                        strategy=strategy,
                    )
                    soft_candidates.append((name, cand))

                best_soft_name = None
                best_soft_score = None
                best_soft_place = placement.clone()
                for name, cand in soft_candidates:
                    prepared = self._prepare_soft_candidate_for_exact(
                        cand,
                        benchmark,
                        legal_cfg,
                        legal_anchor_mode,
                        legal_anchor_strength,
                        legal_restore_iters,
                    )
                    score = self._evaluate_exact_proxy(prepared, benchmark, plc)
                    if score is None:
                        continue
                    if best_soft_score is None or score < best_soft_score:
                        best_soft_score = score
                        best_soft_name = name
                        best_soft_place = prepared
                if best_soft_score is not None:
                    placement = best_soft_place
                    if best_soft_score < best_exact_cost:
                        best_exact_cost = best_soft_score
                        best_exact_placement = placement.clone()
                else:
                    placement = base_candidate
            else:
                placement = self._optimize_soft_macros(
                    placement,
                    benchmark,
                    plc,
                    num_steps=soft_cfg.get("final_num_steps", [40, 40, 40]),
                    use_current_loc=bool(soft_cfg.get("final_use_current_loc", True)),
                )
            if bool(self.cfg.get("post_soft_exact_local_refine", {}).get("enabled", False)):
                placement, post_soft_refine_best = self._run_exact_local_refine(
                    placement,
                    benchmark,
                    plc,
                    edge_index,
                    edge_weight,
                    anchor_targets,
                    anchor_weights,
                    start_time,
                    total_budget,
                    edge_offsets=edge_offsets,
                    cfg_section="post_soft_exact_local_refine",
                )
                placement = sanitize_canvas_bounds(
                    placement,
                    sizes,
                    canvas_w,
                    canvas_h,
                    fixed_mask=fixed_mask,
                    fixed_positions=benchmark.macro_positions,
                    safety_eps=1e-6,
                )
                if post_soft_refine_best is not None and post_soft_refine_best < best_exact_cost:
                    best_exact_cost = post_soft_refine_best
                    best_exact_placement = placement.clone()

        placement = project_to_canvas(
            placement,
            sizes,
            canvas_w,
            canvas_h,
            fixed_mask=fixed_mask,
            fixed_positions=benchmark.macro_positions,
        )

        placement = legalize_hard_macros(
            placement,
            sizes,
            fixed_mask,
            canvas_w,
            canvas_h,
            num_hard,
                gap=float(legal_cfg.get("gap", 1e-4)),
                max_iters=min(50, int(legal_cfg.get("max_iters", 80))),
                fallback_iters=min(45, int(legal_cfg.get("fallback_iters", 60))),
                max_pairs_per_iter=6000,
                anchor_positions=self._resolve_legal_anchor_positions(benchmark, placement, legal_anchor_mode),
                anchor_strength=legal_anchor_strength,
                restore_iters=legal_restore_iters,
            )

        placement = sanitize_canvas_bounds(
            placement,
            sizes,
            canvas_w,
            canvas_h,
            fixed_mask=fixed_mask,
            fixed_positions=benchmark.macro_positions,
            safety_eps=1e-6,
        )

        if count_hard_overlaps(placement, sizes, num_hard, gap=float(legal_cfg.get("gap", 1e-4))) > 0:
            placement = strict_legalize_hard_macros(
                placement,
                sizes,
                fixed_mask,
                canvas_w,
                canvas_h,
                num_hard,
                gap=float(legal_cfg.get("gap", 1e-4)),
                max_iters=min(40, int(legal_cfg.get("max_iters", 80))),
                fallback_iters=min(40, int(legal_cfg.get("fallback_iters", 60))),
                max_pairs_per_iter=5000,
                anchor_positions=self._resolve_legal_anchor_positions(benchmark, placement, legal_anchor_mode),
                anchor_strength=legal_anchor_strength,
                restore_iters=legal_restore_iters,
            )
            placement = sanitize_canvas_bounds(
                placement,
                sizes,
                canvas_w,
                canvas_h,
                fixed_mask=fixed_mask,
                fixed_positions=benchmark.macro_positions,
                safety_eps=1e-6,
            )

        if fixed_mask.any():
            placement[fixed_mask] = benchmark.macro_positions[fixed_mask]

        final_cost = self._consume_exact_eval(
            placement,
            benchmark,
            plc,
            force_final=True,
        )
        if final_cost is not None and final_cost < best_exact_cost:
            best_exact_cost = final_cost
            best_exact_placement = placement.clone()

        if best_exact_placement is not None:
            placement = best_exact_placement

        placement = sanitize_canvas_bounds(
            placement,
            sizes,
            canvas_w,
            canvas_h,
            fixed_mask=fixed_mask,
            fixed_positions=benchmark.macro_positions,
            safety_eps=1e-6,
        )

        legal_max_iters = int(legal_cfg.get("max_iters", 80))
        legal_fallback_iters = int(legal_cfg.get("fallback_iters", 60))
        repair_pairs = 6000 if num_hard >= 700 else 8000
        final_gap = max(float(legal_cfg.get("gap", 1e-4)), 4e-3)

        # Heavy final cleanup is only needed when hard overlaps still exist.
        if (
            count_hard_overlaps(placement, sizes, num_hard, gap=float(legal_cfg.get("gap", 1e-4))) > 0
            and (time.time() - start_time) <= 0.95 * total_budget
        ):
            placement = legalize_hard_macros(
                placement,
                sizes,
                fixed_mask,
                canvas_w,
                canvas_h,
                num_hard,
                gap=float(legal_cfg.get("gap", 1e-4)),
                max_iters=legal_max_iters,
                fallback_iters=legal_fallback_iters,
                max_pairs_per_iter=repair_pairs,
                anchor_positions=self._resolve_legal_anchor_positions(benchmark, placement, legal_anchor_mode),
                anchor_strength=legal_anchor_strength,
                restore_iters=legal_restore_iters,
            )

            placement = sanitize_canvas_bounds(
                placement,
                sizes,
                canvas_w,
                canvas_h,
                fixed_mask=fixed_mask,
                fixed_positions=benchmark.macro_positions,
                safety_eps=1e-6,
            )

        # Resolve hard-vs-soft/fixed overlaps to satisfy official overlap checks.
        if bool(legal_cfg.get("resolve_hard_soft", False)) and (time.time() - start_time) <= 0.97 * total_budget:
            placement = legalize_hard_vs_all(
                placement,
                sizes,
                fixed_mask,
                canvas_w,
                canvas_h,
                num_hard,
                gap=final_gap,
                max_iters=60 if num_hard >= 700 else 40,
            )
            placement = legalize_hard_macros(
                placement,
                sizes,
                fixed_mask,
                canvas_w,
                canvas_h,
                num_hard,
                gap=final_gap,
                max_iters=min(30, legal_max_iters),
                fallback_iters=min(25, legal_fallback_iters),
                max_pairs_per_iter=repair_pairs,
                anchor_positions=self._resolve_legal_anchor_positions(benchmark, placement, legal_anchor_mode),
                anchor_strength=legal_anchor_strength,
                restore_iters=legal_restore_iters,
            )
            placement = sanitize_canvas_bounds(
                placement,
                sizes,
                canvas_w,
                canvas_h,
                fixed_mask=fixed_mask,
                fixed_positions=benchmark.macro_positions,
                safety_eps=1e-6,
            )

        if fixed_mask.any():
            placement[fixed_mask] = benchmark.macro_positions[fixed_mask]

        # Final guardrail: if the official validator still sees violations,
        # run a bounded repair loop and re-pin fixed macros.
        if validate_placement is not None:
            for _ in range(1):
                is_valid, _ = validate_placement(placement, benchmark, check_overlaps=True)
                if is_valid:
                    break
                if (time.time() - start_time) > 0.98 * total_budget:
                    break
                placement = strict_legalize_hard_macros(
                    placement,
                    sizes,
                    fixed_mask,
                    canvas_w,
                    canvas_h,
                    num_hard,
                    gap=final_gap,
                    max_iters=legal_max_iters,
                    fallback_iters=legal_fallback_iters,
                    max_pairs_per_iter=max(5000, repair_pairs),
                    anchor_positions=self._resolve_legal_anchor_positions(benchmark, placement, legal_anchor_mode),
                    anchor_strength=legal_anchor_strength,
                    restore_iters=legal_restore_iters,
                )
                if bool(legal_cfg.get("resolve_hard_soft", False)):
                    placement = legalize_hard_vs_all(
                        placement,
                        sizes,
                        fixed_mask,
                        canvas_w,
                        canvas_h,
                        num_hard,
                        gap=final_gap,
                        max_iters=30,
                    )
                if soft_transport_active:
                    placement = legalize_soft_vs_all(
                        placement,
                        sizes,
                        fixed_mask,
                        canvas_w,
                        canvas_h,
                        num_hard,
                        gap=final_gap,
                        max_iters=max(12, int(soft_transport_cfg.get("final_legalize_iters", 30)) // 2),
                    )
                placement = sanitize_canvas_bounds(
                    placement,
                    sizes,
                    canvas_w,
                    canvas_h,
                    fixed_mask=fixed_mask,
                    fixed_positions=benchmark.macro_positions,
                    safety_eps=1e-6,
                )
                if fixed_mask.any():
                    placement[fixed_mask] = benchmark.macro_positions[fixed_mask]

        return placement

    def _run_profile_portfolio(self, benchmark: Benchmark) -> torch.Tensor:
        portfolio_cfg = self.cfg.get("portfolio", {})
        entries = list(portfolio_cfg.get("entries", []))
        debug = bool(portfolio_cfg.get("debug", False))
        if not entries:
            fallback_cfg = copy.deepcopy(self.cfg)
            fallback_cfg["portfolio"] = {"enabled": False, "entries": []}
            fallback_cfg["benchmark_isolation"] = {"enabled": False}
            fallback_cfg["_portfolio_child"] = True
            child = TeamPlasmaPlacer(config_overrides=fallback_cfg, seed=self.seed)
            child._plc_cache = self._fresh_plc_cache_entry(benchmark.name)
            return child.place(benchmark)

        plc = self._plc_cache.get(benchmark.name)
        if plc is None:
            plc = _load_plc_for_benchmark(benchmark.name)
            if plc is not None:
                self._plc_cache[benchmark.name] = plc

        best_score = float("inf")
        best_place: Optional[torch.Tensor] = None
        best_valid = False

        for idx, entry in enumerate(entries):
            seed_offset = int(entry.get("seed_offset", idx)) if isinstance(entry, dict) else idx
            trial_seed = self.seed + seed_offset
            child, _ = self._build_portfolio_child(
                benchmark.name,
                dict(entry) if isinstance(entry, dict) else {},
                idx,
            )
            cand = child.place(benchmark)
            cand = sanitize_canvas_bounds(
                cand,
                benchmark.macro_sizes,
                benchmark.canvas_width,
                benchmark.canvas_height,
                fixed_mask=benchmark.macro_fixed,
                fixed_positions=benchmark.macro_positions,
                safety_eps=1e-6,
            )

            is_valid = True
            if validate_placement is not None:
                is_valid, _ = validate_placement(cand, benchmark, check_overlaps=True)

            score = self._evaluate_exact_proxy(cand, benchmark, plc)
            score = float(score) if score is not None else float("inf")
            if debug:
                print(
                    f"[portfolio] {benchmark.name} entry={entry.get('name', idx) if isinstance(entry, dict) else idx} "
                    f"seed={trial_seed} valid={bool(is_valid)} score={score:.6f}"
                )

            if best_place is None:
                best_place = cand.clone()
                best_score = score
                best_valid = bool(is_valid)
                continue

            if bool(is_valid) and not best_valid:
                best_place = cand.clone()
                best_score = score
                best_valid = True
                if debug:
                    print(f"[portfolio] {benchmark.name} new_best={score:.6f} (valid upgrade)")
                continue

            if bool(is_valid) == best_valid and score < best_score:
                best_place = cand.clone()
                best_score = score
                if debug:
                    print(f"[portfolio] {benchmark.name} new_best={score:.6f}")

        if best_place is not None:
            return best_place

        return benchmark.macro_positions.clone().float()

    def _router_plasma_probe_features(
        self,
        benchmark: Benchmark,
        probe_cfg: Dict[str, object],
    ) -> Dict[str, float]:
        n = int(benchmark.num_hard_macros)
        if n <= 1:
            return {}

        device = self._resolve_device(str(probe_cfg.get("device", self.cfg.get("device", "auto"))))
        hard_pos = benchmark.macro_positions[:n].clone().float().to(device)
        hard_sizes = benchmark.macro_sizes[:n].float().to(device)

        plc = None
        if bool(probe_cfg.get("use_plc", True)):
            plc = self._plc_cache.get(benchmark.name)
            if plc is None:
                plc = _load_plc_for_benchmark(benchmark.name)
                if plc is not None:
                    self._plc_cache[benchmark.name] = plc

        edge_index = torch.empty((0, 2), dtype=torch.long, device=device)
        edge_weight = torch.empty((0,), dtype=torch.float32, device=device)
        edge_offsets = torch.empty((0, 2, 2), dtype=torch.float32, device=device)
        max_edges = int(
            probe_cfg.get(
                "max_surrogate_edges",
                self.cfg.get("global_stage", {}).get("max_surrogate_edges", 8000),
            )
        )
        probe_pde = copy.deepcopy(self.cfg.get("global_stage", {}).get("pde", {}))
        probe_pde = self._deep_update(probe_pde, dict(probe_cfg.get("pde", {})))
        probe_pin_flux = bool(probe_pde.get("pin_flux_tubes", False))
        probe_pin_flux_filter = dict(self.cfg.get("global_stage", {}).get("pin_flux_edge_filter", {}))
        probe_pin_flux_filter.update(dict(probe_cfg.get("pin_flux_edge_filter", {})))
        if plc is not None:
            if probe_pin_flux:
                plc_edge_index, plc_edge_weight, plc_edge_offsets = extract_pin_flux_tubes_from_plc(
                    benchmark,
                    plc,
                    max_edges=max_edges,
                    min_degree=int(probe_pin_flux_filter.get("min_degree", 2)),
                    max_degree=int(probe_pin_flux_filter.get("max_degree", 0)),
                    weight_mode=str(probe_pin_flux_filter.get("weight_mode", "unit_net")),
                    degree_power=float(probe_pin_flux_filter.get("degree_power", 1.0)),
                )
            else:
                plc_edge_index, plc_edge_weight = extract_hard_edges_from_plc(
                    benchmark,
                    plc,
                    max_edges=max_edges,
                )
                plc_edge_offsets = torch.zeros((int(plc_edge_index.shape[0]), 2, 2), dtype=torch.float32)
            edge_index = plc_edge_index.to(device)
            edge_weight = plc_edge_weight.to(device)
            edge_offsets = plc_edge_offsets.to(device)

        knn_cfg = dict(probe_cfg.get("knn_edges", {}))
        knn_enabled = bool(knn_cfg.get("enabled", edge_index.numel() == 0))
        if knn_enabled:
            knn_k = max(1, int(knn_cfg.get("k", 6)))
            knn_scale = float(knn_cfg.get("weight", 0.2 if edge_index.numel() > 0 else 1.0))
            knn_index, knn_weight = build_knn_edges(benchmark.macro_positions[:n].float(), k=knn_k)
            knn_index = knn_index.to(device)
            knn_weight = knn_weight.to(device)
            knn_offsets = torch.zeros((int(knn_index.shape[0]), 2, 2), dtype=torch.float32, device=device)
            if edge_index.numel() == 0:
                edge_index = knn_index
                edge_weight = knn_weight * knn_scale
                edge_offsets = knn_offsets
            else:
                edge_index, edge_weight, edge_offsets = merge_edge_sets_with_offsets(
                    edge_index,
                    edge_weight,
                    edge_offsets,
                    knn_index,
                    knn_weight,
                    knn_offsets,
                    scale_b=knn_scale,
                )

        anchor_targets = None
        anchor_weights = None
        if plc is not None and bool(probe_cfg.get("use_anchors", True)):
            plc_anchor_targets, plc_anchor_weights = build_hard_anchor_targets_from_plc(
                benchmark,
                plc,
                use_pin_positions=probe_pin_flux,
            )
            if plc_anchor_targets.numel() > 0 and plc_anchor_weights.numel() > 0:
                anchor_targets = plc_anchor_targets.to(device)
                anchor_weights = plc_anchor_weights.to(device)

        bg_positions = None
        bg_sizes = None
        if bool(probe_cfg.get("include_soft_background", True)) and int(benchmark.num_soft_macros) > 0:
            bg_positions = benchmark.macro_positions[n:].clone().float().to(device)
            bg_sizes = benchmark.macro_sizes[n:].float().to(device)

        probe_rhs = {
            "rho": 1.1,
            "q": 0.8,
            "n": 0.25,
            "wall": 0.7,
        }
        probe_rhs.update(dict(probe_cfg.get("rhs_weights", {})))

        fields = compute_plasma_forces(
            positions=hard_pos,
            sizes=hard_sizes,
            edge_index=edge_index,
            edge_weight=edge_weight,
            canvas_width=float(benchmark.canvas_width),
            canvas_height=float(benchmark.canvas_height),
            grid_size=max(8, int(probe_cfg.get("grid_size", 16))),
            pde_cfg=probe_pde,
            rhs_weights=probe_rhs,
            background_positions=bg_positions,
            background_sizes=bg_sizes,
            anchor_targets=anchor_targets,
            anchor_weights=anchor_weights,
            edge_offsets=edge_offsets,
            previous_psi=None,
            bundle_terms=self._active_bundle_terms,
            pin_terms=self._active_pin_terms,
            pin_edge_profile=self._active_pin_edge_profile,
        )

        def _flat_quantile(field: torch.Tensor, q: float) -> float:
            if field.numel() == 0:
                return 0.0
            return float(torch.quantile(field.reshape(-1), q).item())

        rho = fields.get("rho", torch.zeros((), dtype=torch.float32, device=device)).detach().float()
        q_raw = fields.get("q", torch.zeros_like(rho)).detach().float()
        psi = fields.get("psi", torch.zeros_like(rho)).detach().float()
        beta = float(probe_pde.get("rhs_softplus_beta", 8.0))
        q_over = F.softplus(beta * (q_raw - 1.0)) / max(beta, 1e-12)

        q_p90 = _flat_quantile(q_over, 0.90)
        rho_p90 = _flat_quantile(rho, 0.90)

        hot_force = fields.get("hot_force", torch.zeros((n, 2), dtype=torch.float32, device=device)).detach().float()
        cold_force = fields.get("cold_force", torch.zeros_like(hot_force)).detach().float()
        temp_split = 0.0
        if hot_force.numel() > 0 and cold_force.numel() == hot_force.numel():
            temp_split = float((hot_force - cold_force).norm(dim=1).mean().item())

        residual_history = list(fields.get("residual_history", []))
        return {
            "probe_q_p90": q_p90,
            "probe_q_p99": _flat_quantile(q_over, 0.99),
            "probe_q_max": float(q_over.max().item()) if q_over.numel() > 0 else 0.0,
            "probe_rho_p90": rho_p90,
            "probe_rho_p99": _flat_quantile(rho, 0.99),
            "probe_rho_max": float(rho.max().item()) if rho.numel() > 0 else 0.0,
            "probe_q_rho_ratio": q_p90 / max(rho_p90, 1e-6),
            "probe_psi_abs_mean": float(psi.abs().mean().item()) if psi.numel() > 0 else 0.0,
            "probe_psi_peak": float(psi.abs().max().item()) if psi.numel() > 0 else 0.0,
            "probe_temp_split": temp_split,
            "probe_residual_last": float(residual_history[-1]) if residual_history else 0.0,
        }

    def _benchmark_router_features(self, benchmark: Benchmark) -> Dict[str, float]:
        n = max(1, int(benchmark.num_hard_macros))
        overlap_rate = float(
            count_hard_overlaps(
                benchmark.macro_positions,
                benchmark.macro_sizes,
                int(benchmark.num_hard_macros),
                gap=float(self.cfg.get("legalizer", {}).get("gap", 1e-4)),
            )
        ) / float(n)
        canvas_area = max(float(benchmark.canvas_width) * float(benchmark.canvas_height), 1e-6)
        hard_area = float((benchmark.macro_sizes[: int(benchmark.num_hard_macros), 0] * benchmark.macro_sizes[: int(benchmark.num_hard_macros), 1]).sum().item())
        features = {
            "num_hard_norm": float(benchmark.num_hard_macros) / 800.0,
            "soft_fill": float(benchmark_soft_fill_ratio(benchmark)),
            "init_overlap_rate": overlap_rate,
            "hard_fill": hard_area / canvas_area,
        }
        router_cfg = self.cfg.get("router", {})
        probe_cfg = dict(router_cfg.get("plasma_probe", {}))
        if bool(probe_cfg.get("enabled", False)):
            try:
                features.update(self._router_plasma_probe_features(benchmark, probe_cfg))
            except Exception as exc:
                if bool(router_cfg.get("debug", False)):
                    print(f"[router-probe] {benchmark.name} failed={exc!r}")
        return features

    def _build_router_child(
        self,
        benchmark_name: str,
        entry: Dict[str, object],
    ) -> Tuple["TeamPlasmaPlacer", Dict[str, float]]:
        router_cfg = self.cfg.get("router", {})
        if entry.get("config_path"):
            base_cfg = load_team_plasma_config(str(entry["config_path"]))
        else:
            base_cfg = copy.deepcopy(self.cfg)

        merged_cfg = self._deep_update(
            copy.deepcopy(base_cfg),
            dict(router_cfg.get("child_overrides", {})),
        )
        merged_cfg = self._deep_update(merged_cfg, dict(entry.get("overrides", {})))
        merged_cfg["router"] = {"enabled": False, "prototypes": [], "rules": []}
        merged_cfg["benchmark_isolation"] = {"enabled": False}
        merged_cfg["_router_child"] = True

        seed_offset = int(entry.get("seed_offset", 0))
        child = TeamPlasmaPlacer(config_overrides=merged_cfg, seed=self.seed + seed_offset)
        child._plc_cache = self._fresh_plc_cache_entry(benchmark_name)
        return child, merged_cfg

    @staticmethod
    def _router_rule_matches(features: Dict[str, float], rule: Dict[str, object]) -> bool:
        conditions = dict(rule.get("conditions", {}))
        for key, spec in conditions.items():
            if key not in features:
                return False
            value = float(features[key])
            if isinstance(spec, dict):
                if "min" in spec and value < float(spec["min"]):
                    return False
                if "max" in spec and value > float(spec["max"]):
                    return False
                if "eq" in spec and abs(value - float(spec["eq"])) > 1e-12:
                    return False
            else:
                if abs(value - float(spec)) > 1e-12:
                    return False
        return True

    def _run_rule_router(self, benchmark: Benchmark) -> torch.Tensor:
        router_cfg = self.cfg.get("router", {})
        rules = list(router_cfg.get("rules", []))
        debug = bool(router_cfg.get("debug", False))

        feat = self._benchmark_router_features(benchmark)
        selected: Optional[Dict[str, object]] = None
        for idx, rule in enumerate(rules):
            if self._router_rule_matches(feat, rule):
                selected = dict(rule)
                selected.setdefault("name", f"rule_{idx}")
                break

        if selected is None:
            default_entry = dict(router_cfg.get("default", {}))
            if not default_entry:
                fallback_cfg = copy.deepcopy(self.cfg)
                fallback_cfg["router"] = {"enabled": False, "prototypes": [], "rules": []}
                fallback_cfg["benchmark_isolation"] = {"enabled": False}
                fallback_cfg["_router_child"] = True
                child = TeamPlasmaPlacer(config_overrides=fallback_cfg, seed=self.seed)
                child._plc_cache = self._fresh_plc_cache_entry(benchmark.name)
                return child.place(benchmark)
            selected = default_entry
            selected.setdefault("name", "default")

        child, _ = self._build_router_child(benchmark.name, selected)
        if debug:
            print(
                f"[router-rule] {benchmark.name} features={feat} "
                f"choice={selected.get('name', 'unnamed')} config={selected.get('config_path', '<self>')}"
            )
        return child.place(benchmark)

    def _run_feature_router(self, benchmark: Benchmark) -> torch.Tensor:
        router_cfg = self.cfg.get("router", {})
        rules = list(router_cfg.get("rules", []))
        prototypes = list(router_cfg.get("prototypes", []))
        debug = bool(router_cfg.get("debug", False))
        feat = self._benchmark_router_features(benchmark)
        weights = dict(router_cfg.get("weights", {}))

        if rules:
            selected: Optional[Dict[str, object]] = None
            for idx, rule in enumerate(rules):
                if self._router_rule_matches(feat, rule):
                    selected = dict(rule)
                    selected.setdefault("name", f"rule_{idx}")
                    break
            if selected is not None:
                child, _ = self._build_router_child(benchmark.name, selected)
                if debug:
                    print(
                        f"[router-rule] {benchmark.name} features={feat} "
                        f"choice={selected.get('name', 'unnamed')} config={selected.get('config_path', '<self>')}"
                    )
                return child.place(benchmark)

        if not prototypes:
            default_entry = dict(router_cfg.get("default", {}))
            if default_entry:
                default_entry.setdefault("name", "default")
                child, _ = self._build_router_child(benchmark.name, default_entry)
                if debug:
                    print(
                        f"[router-default] {benchmark.name} features={feat} "
                        f"choice={default_entry.get('name', 'unnamed')} config={default_entry.get('config_path', '<self>')}"
                    )
                return child.place(benchmark)
            fallback_cfg = copy.deepcopy(self.cfg)
            fallback_cfg["router"] = {"enabled": False, "prototypes": [], "rules": []}
            fallback_cfg["benchmark_isolation"] = {"enabled": False}
            fallback_cfg["_router_child"] = True
            child = TeamPlasmaPlacer(config_overrides=fallback_cfg, seed=self.seed)
            child._plc_cache = self._fresh_plc_cache_entry(benchmark.name)
            return child.place(benchmark)

        def _proto_distance(proto: Dict[str, object]) -> float:
            proto_feat = dict(proto.get("features", {}))
            dist = 0.0
            for key, value in proto_feat.items():
                w = float(weights.get(key, 1.0))
                dist += (w * (float(feat.get(key, 0.0)) - float(value))) ** 2
            return dist

        ranked = sorted(
            ((float(_proto_distance(proto)), idx, proto) for idx, proto in enumerate(prototypes)),
            key=lambda item: (item[0], item[1]),
        )
        top_k = max(1, int(router_cfg.get("top_k", 1)))
        diversify = bool(router_cfg.get("diversify_by_config", True))
        shortlist: List[Tuple[float, int, Dict[str, object]]] = []
        seen_keys = set()
        for dist, idx, proto in ranked:
            key = str(proto.get("config_path", "")) or f"idx:{idx}"
            if diversify and key in seen_keys:
                continue
            shortlist.append((dist, idx, proto))
            seen_keys.add(key)
            if len(shortlist) >= top_k:
                break

        pilot_cfg = dict(router_cfg.get("pilot_rerank", {}))
        min_best_distance = float(pilot_cfg.get("skip_if_best_distance_below", -1.0))
        if (
            bool(pilot_cfg.get("enabled", False))
            and len(shortlist) > 1
            and not (min_best_distance >= 0.0 and shortlist[0][0] <= min_best_distance)
        ):
            base_placement = benchmark.macro_positions.clone().float()
            dist_weight = float(pilot_cfg.get("distance_weight", 0.0))
            best_choice: Optional[Tuple[float, float, int, Dict[str, object], TeamPlasmaPlacer]] = None
            for dist, idx, proto in shortlist:
                child, _ = self._build_router_child(benchmark.name, proto)
                try:
                    surrogate_terms = child._prepare_surrogate_terms(benchmark, base_placement)
                    if len(surrogate_terms) == 5:
                        _, edge_index, edge_weight, anchor_targets, anchor_weights = surrogate_terms
                        edge_offsets = torch.zeros((int(edge_index.shape[0]), 2, 2), dtype=torch.float32)
                    elif len(surrogate_terms) == 6:
                        _, edge_index, edge_weight, edge_offsets, anchor_targets, anchor_weights = surrogate_terms
                    else:
                        _, edge_index, edge_weight, edge_offsets, anchor_targets, anchor_weights, _bundle_terms = surrogate_terms
                    try:
                        _, pilot_score = child._pilot_plasma_candidate(
                            base_placement.clone(),
                            benchmark,
                            edge_index,
                            edge_weight,
                            anchor_targets,
                            anchor_weights,
                            pilot_cfg,
                            edge_offsets=edge_offsets,
                        )
                    except TypeError:
                        _, pilot_score = child._pilot_plasma_candidate(
                            base_placement.clone(),
                            benchmark,
                            edge_index,
                            edge_weight,
                            anchor_targets,
                            anchor_weights,
                            pilot_cfg,
                        )
                    combined = float(pilot_score) + dist_weight * float(dist)
                except Exception:
                    combined = float("inf")
                if best_choice is None or combined < best_choice[0]:
                    best_choice = (combined, dist, idx, proto, child)

            if best_choice is not None and math.isfinite(best_choice[0]):
                _, dist, _, best_proto, child = best_choice
                if debug:
                    print(
                        f"[router-pilot] {benchmark.name} features={feat} "
                        f"choice={best_proto.get('name', 'unnamed')} "
                        f"config={best_proto.get('config_path', '<self>')} dist={dist:.5f}"
                    )
                return child.place(benchmark)

        _, _, best_proto = shortlist[0]
        child, _ = self._build_router_child(benchmark.name, best_proto)
        if debug:
            print(
                f"[router] {benchmark.name} features={feat} "
                f"choice={best_proto.get('name', 'unnamed')} config={best_proto.get('config_path', '<self>')}"
            )
        return child.place(benchmark)

    def _time_budget_sec(self) -> float:
        budget = self.cfg.get("time_budget_sec", {})
        mode_budget = budget.get(self.mode, budget.get("local", 60.0))
        return float(mode_budget)

    @staticmethod
    def _resolve_device(device_cfg: str) -> torch.device:
        if device_cfg == "cpu":
            return torch.device("cpu")
        if device_cfg == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _run_plasma_global_stage(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_offsets: torch.Tensor,
        anchor_targets: torch.Tensor,
        anchor_weights: torch.Tensor,
        soft_transport_model: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        start_time: float,
        total_budget: float,
        allow_exact_snapshots: bool,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        n = int(benchmark.num_hard_macros)
        if n <= 1:
            return placement, None

        device = self._resolve_device(str(self.cfg.get("device", "auto")))

        hard_pos = placement[:n].to(device)
        hard_sizes = benchmark.macro_sizes[:n].to(device)
        hard_fixed = benchmark.macro_fixed[:n].to(device)
        hard_fixed_pos = hard_pos.clone()
        soft_pos = placement[n:].to(device) if int(placement.shape[0]) > n else torch.zeros(0, 2, device=device)
        soft_sizes = benchmark.macro_sizes[n:].to(device) if int(benchmark.macro_sizes.shape[0]) > n else torch.zeros(0, 2, device=device)

        edge_index_d = edge_index.to(device)
        edge_weight_d = edge_weight.to(device)
        edge_offsets_d = edge_offsets.to(device) if edge_offsets.numel() > 0 else torch.zeros(0, 2, 2, dtype=torch.float32, device=device)
        anchor_targets_d = anchor_targets.to(device)
        anchor_weights_d = anchor_weights.to(device)

        best_exact_cost: Optional[float] = None
        best_exact_pos: Optional[torch.Tensor] = None

        snapshot_every = int(self.cfg["global_stage"].get("snapshot_every", 10))
        legal_cfg = self.cfg["legalizer"]
        pde_cfg_base = dict(self.cfg["global_stage"].get("pde", {}))
        dyn_cfg = dict(self.cfg.get("global_stage", {}).get("dynamic_congestion", {}))
        dyn_enabled = bool(dyn_cfg.get("enabled", False))
        dyn_q_alpha = float(dyn_cfg.get("q_boost_alpha", 0.0))
        dyn_net_beta = float(dyn_cfg.get("net_down_beta", 0.0))
        dyn_trust_gamma = float(dyn_cfg.get("trust_up_gamma", 0.0))
        dyn_q_ref = float(dyn_cfg.get("q_peak_ref", 1.0))
        dyn_stage_names = set(str(s) for s in dyn_cfg.get("stage_names", []))
        control_cfg = dict(self.cfg.get("global_stage", {}).get("plasma_control", {}))
        control_enabled = bool(control_cfg.get("enabled", False))
        control_stage_names = set(str(s) for s in control_cfg.get("stage_names", []))
        soft_bg_cfg = dict(self.cfg.get("global_stage", {}).get("soft_background", {}))
        soft_fill_ratio = benchmark_soft_fill_ratio(benchmark)
        bg_scale = compute_soft_fill_scale(soft_fill_ratio, soft_bg_cfg, disabled_value=0.0)
        soft_transport_cfg = dict(self.cfg.get("global_stage", {}).get("soft_transport", {}))
        soft_transport_enabled = (
            bool(soft_transport_cfg.get("enabled", False))
            and soft_pos.numel() > 0
            and soft_fill_ratio >= float(soft_transport_cfg.get("min_soft_fill", 0.45))
        )
        soft_anchor_cfg = dict(self.cfg.get("global_stage", {}).get("soft_anchor", {}))
        soft_sync_cfg = dict(self.cfg.get("global_stage", {}).get("soft_macro_sync", {}))
        barrier_cfg_base = dict(self.cfg.get("global_stage", {}).get("transport_barrier", {}))
        sheath_cfg_base = dict(self.cfg.get("global_stage", {}).get("pin_sheath", {}))
        anchor_stage_scale = compute_soft_fill_scale(soft_fill_ratio, soft_anchor_cfg, disabled_value=1.0)
        anchor_conf = compute_anchor_confidence_scale(anchor_weights_d, soft_anchor_cfg).unsqueeze(1)
        precond_cfg = dict(self.cfg.get("global_stage", {}).get("preconditioner", {}))
        move_precond = compute_transport_preconditioner(
            hard_sizes,
            edge_index_d,
            edge_weight_d,
            precond_cfg,
        )
        soft_hard_index, soft_hard_weight, soft_fixed_sum, soft_fixed_weight = soft_transport_model
        soft_hard_index_d = soft_hard_index.to(device) if soft_hard_index.numel() > 0 else torch.zeros(0, 2, dtype=torch.long, device=device)
        soft_hard_weight_d = soft_hard_weight.to(device) if soft_hard_weight.numel() > 0 else torch.zeros(0, dtype=torch.float32, device=device)
        soft_fixed_sum_d = soft_fixed_sum.to(device) if soft_fixed_sum.numel() > 0 else torch.zeros(int(benchmark.num_soft_macros), 2, dtype=torch.float32, device=device)
        soft_fixed_weight_d = soft_fixed_weight.to(device) if soft_fixed_weight.numel() > 0 else torch.zeros(int(benchmark.num_soft_macros), dtype=torch.float32, device=device)
        soft_sizes = benchmark.macro_sizes[n:].to(device) if int(benchmark.macro_sizes.shape[0]) > n else torch.zeros(0, 2, device=device)
        soft_fixed = benchmark.macro_fixed[n:].to(device) if int(benchmark.macro_fixed.shape[0]) > n else torch.zeros(0, dtype=torch.bool, device=device)
        soft_fixed_pos = soft_pos.clone()
        max_span = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        soft_force_weights = dict(soft_transport_cfg.get("force_weights", {}))
        soft_lr_scale = float(soft_transport_cfg.get("lr_scale", 0.55))
        soft_update_every = max(1, int(soft_transport_cfg.get("update_every", 2)))
        soft_trust_radius = float(soft_transport_cfg.get("trust_radius_frac", 0.010)) * max_span
        soft_precond_cfg = dict(soft_transport_cfg.get("preconditioner", {}))
        soft_precond = compute_transport_preconditioner(
            soft_sizes,
            torch.zeros(0, 2, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.float32, device=device),
            soft_precond_cfg,
        ) if soft_transport_enabled else torch.ones((int(soft_pos.shape[0]), 1), dtype=torch.float32, device=device)
        psi_cache: Optional[torch.Tensor] = None
        control_scales = compute_plasma_control_scales({}, {"enabled": False})

        for stage in self.cfg["global_stage"].get("stages", []):
            if time.time() - start_time > 0.45 * total_budget:
                break

            steps = int(stage.get("steps", 30))
            adaptive_cfg = self.cfg.get("global_stage", {}).get("adaptive_scale", {})
            if bool(adaptive_cfg.get("enabled", True)):
                ref_n = float(adaptive_cfg.get("reference_num_hard", 320.0))
                min_s = float(adaptive_cfg.get("min_scale", 0.85))
                max_s = float(adaptive_cfg.get("max_scale", 1.50))
                scale = max(min_s, min(max_s, n / max(ref_n, 1.0)))
                steps = max(6, int(round(steps * scale)))
            lr = float(stage.get("lr", 0.05))
            grid_size = int(stage.get("grid_size", 24))
            trust_radius = float(stage.get("trust_radius_frac", 0.03)) * max_span
            rhs_weights = dict(stage.get("rhs_weights", {}))
            force_weights = dict(stage.get("force_weights", {}))
            rhs_weights["bg"] = float(rhs_weights.get("bg", 0.0)) * bg_scale
            force_weights["bg_pressure"] = float(force_weights.get("bg_pressure", 0.0)) * bg_scale
            psi_cache = resample_scalar_field(psi_cache, grid_size)
            q_boost = 1.0

            stage_name = str(stage.get("name", ""))
            stage_dyn_enabled = dyn_enabled and (not dyn_stage_names or stage_name in dyn_stage_names)
            stage_ctrl_enabled = control_enabled and (not control_stage_names or stage_name in control_stage_names)
            stage_barrier_cfg = dict(barrier_cfg_base)
            stage_barrier_cfg.update(dict(stage.get("transport_barrier", {})))
            stage_sheath_cfg = dict(sheath_cfg_base)
            stage_sheath_cfg.update(dict(stage.get("pin_sheath", {})))

            for it in range(steps):
                if time.time() - start_time > 0.50 * total_budget:
                    break

                fields = compute_plasma_forces(
                    positions=hard_pos,
                    sizes=hard_sizes,
                    edge_index=edge_index_d,
                    edge_weight=edge_weight_d,
                    canvas_width=benchmark.canvas_width,
                    canvas_height=benchmark.canvas_height,
                    grid_size=grid_size,
                    pde_cfg=pde_cfg_base,
                    rhs_weights={
                        **rhs_weights,
                        "rho": float(rhs_weights.get("rho", 1.0)) * float(control_scales.get("rho_rhs_scale", 1.0)),
                        "q": float(rhs_weights.get("q", 0.5)) * q_boost * float(control_scales.get("q_rhs_scale", 1.0)),
                    },
                    background_positions=soft_pos,
                    background_sizes=soft_sizes,
                    anchor_targets=anchor_targets_d,
                    anchor_weights=anchor_weights_d,
                    edge_offsets=edge_offsets_d,
                    previous_psi=psi_cache,
                    bundle_terms=self._active_bundle_terms,
                    pin_terms=self._active_pin_terms,
                    pin_edge_profile=self._active_pin_edge_profile,
                )

                plasma_force = fields["plasma_force"]
                hall_force = fields.get("hall_force", torch.zeros_like(hard_pos))
                chi_force = fields.get("chi_force", torch.zeros_like(hard_pos))
                dia_force = fields.get("dia_force", torch.zeros_like(hard_pos))
                balloon_force = fields.get("balloon_force", torch.zeros_like(hard_pos))
                hot_force = fields.get("hot_force", torch.zeros_like(hard_pos))
                cold_force = fields.get("cold_force", torch.zeros_like(hard_pos))
                rho_force = fields.get("rho_force", torch.zeros_like(hard_pos))
                q_force = fields.get("q_force", torch.zeros_like(hard_pos))
                bg_force = fields.get("bg_force", torch.zeros_like(hard_pos))
                channel_force = compute_pin_edge_sheath_force(
                    hard_pos,
                    hard_sizes,
                    self._active_pin_edge_profile,
                    stage_sheath_cfg,
                    benchmark.canvas_width,
                    benchmark.canvas_height,
                )
                net_force = fields["net_force"]
                repulsion_force = fields["repulsion_force"]
                psi_cache = fields["psi"]
                q_peak = 1.0
                if stage_dyn_enabled and isinstance(fields.get("q", None), torch.Tensor):
                    q_field = fields["q"].float()
                    if q_field.numel() > 0:
                        q_peak = float(torch.quantile(q_field, 0.90).item())
                        excess = max(0.0, q_peak - dyn_q_ref)
                        q_boost = 1.0 + dyn_q_alpha * excess
                if stage_ctrl_enabled:
                    control_scales = compute_plasma_control_scales(fields, control_cfg)
                else:
                    control_scales = compute_plasma_control_scales({}, {"enabled": False})

                anchor_force = torch.zeros_like(hard_pos)
                anchor_mask = anchor_weights_d > 1e-8
                if bool(anchor_mask.any()):
                    anchor_force[anchor_mask] = anchor_targets_d[anchor_mask] - hard_pos[anchor_mask]
                    anorm = torch.linalg.norm(anchor_force, dim=1)
                    aden = torch.quantile(anorm, 0.90) if n >= 10 else torch.max(anorm)
                    av = float(aden.item()) if anorm.numel() > 0 else 0.0
                    if av > 1e-6:
                        anchor_force = anchor_force / av
                    anchor_gate = compute_anchor_alignment_gate(
                        anchor_force,
                        plasma_force,
                        q_force,
                        rho_force,
                        repulsion_force,
                        soft_anchor_cfg,
                    ).unsqueeze(1)
                    anchor_force = anchor_force * anchor_conf * anchor_gate

                net_scale = 1.0
                if stage_dyn_enabled:
                    net_scale = 1.0 / (1.0 + dyn_net_beta * max(0.0, q_peak - dyn_q_ref))
                net_scale *= float(control_scales.get("net_scale", 1.0))
                anchor_scale = float(control_scales.get("anchor_scale", 1.0))
                transport_force = (
                    float(force_weights.get("plasma", 1.0)) * plasma_force
                    + float(force_weights.get("hall", 0.0)) * float(control_scales.get("hall_scale", 1.0)) * hall_force
                    + float(force_weights.get("chi", 0.0)) * chi_force
                    + float(force_weights.get("dia", 0.0)) * float(control_scales.get("dia_scale", 1.0)) * dia_force
                    + float(force_weights.get("balloon", 0.0)) * float(control_scales.get("balloon_scale", 1.0)) * balloon_force
                    + float(force_weights.get("hot", 0.0)) * float(control_scales.get("hot_scale", 1.0)) * hot_force
                    + float(force_weights.get("cold", 0.0)) * float(control_scales.get("cold_scale", 1.0)) * cold_force
                    + float(force_weights.get("rho_pressure", 0.0)) * float(control_scales.get("rho_force_scale", 1.0)) * rho_force
                    + float(force_weights.get("q_pressure", 0.0)) * float(control_scales.get("q_force_scale", 1.0)) * q_force
                    + float(force_weights.get("pin_pressure", 0.0)) * fields.get("pin_force", torch.zeros_like(hard_pos))
                    + float(force_weights.get("pin_channel", 0.0)) * channel_force
                    + float(force_weights.get("bg_pressure", 0.0)) * bg_force
                    + float(force_weights.get("porosity_pressure", 0.0)) * fields.get("porosity_force", torch.zeros_like(hard_pos))
                    + float(force_weights.get("net", 0.3)) * net_force * net_scale
                )
                transport_force = apply_transport_barrier(
                    transport_force,
                    plasma_force,
                    fields.get("q_samples", torch.zeros(0, device=device)),
                    fields.get("rho_samples", torch.zeros(0, device=device)),
                    stage_barrier_cfg,
                )
                total_force = (
                    transport_force
                    + float(force_weights.get("repulsion", 0.2)) * float(control_scales.get("repulsion_scale", 1.0)) * repulsion_force
                    + float(force_weights.get("anchor", 0.24)) * anchor_stage_scale * anchor_scale * anchor_force
                )
                if hard_fixed.any():
                    total_force[hard_fixed] = 0.0

                step_vec = lr * total_force / torch.clamp(move_precond, min=1e-6)
                norms = torch.linalg.norm(step_vec, dim=1, keepdim=True)
                dynamic_trust = trust_radius
                if stage_dyn_enabled:
                    dynamic_trust = trust_radius * (1.0 + dyn_trust_gamma * max(0.0, q_peak - dyn_q_ref))
                dynamic_trust = dynamic_trust * float(control_scales.get("trust_scale", 1.0))
                if dynamic_trust > 0.0:
                    scale = torch.clamp(
                        dynamic_trust / torch.clamp(norms, min=1e-9),
                        max=1.0,
                    )
                    step_vec = step_vec * scale

                hard_pos = hard_pos + step_vec
                hard_pos = project_to_canvas(
                    hard_pos,
                    hard_sizes,
                    benchmark.canvas_width,
                    benchmark.canvas_height,
                    fixed_mask=hard_fixed,
                    fixed_positions=hard_fixed_pos,
                )

                if soft_transport_enabled and ((it + 1) % soft_update_every == 0):
                    psi_gx, psi_gy = gradient_from_potential(fields["psi"], benchmark.canvas_width, benchmark.canvas_height)
                    q_field = fields.get("q_over", fields.get("q"))
                    q_gx, q_gy = gradient_from_potential(q_field, benchmark.canvas_width, benchmark.canvas_height)
                    rho_gx, rho_gy = gradient_from_potential(fields["rho"], benchmark.canvas_width, benchmark.canvas_height)
                    bg_gx, bg_gy = gradient_from_potential(fields["bg_density"], benchmark.canvas_width, benchmark.canvas_height)

                    soft_plasma = _normalize_move_force(
                        sample_vector_field(-psi_gx, -psi_gy, soft_pos, benchmark.canvas_width, benchmark.canvas_height)
                    )
                    soft_q = _normalize_move_force(
                        sample_vector_field(-q_gx, -q_gy, soft_pos, benchmark.canvas_width, benchmark.canvas_height)
                    )
                    soft_rho = _normalize_move_force(
                        sample_vector_field(-rho_gx, -rho_gy, soft_pos, benchmark.canvas_width, benchmark.canvas_height)
                    )
                    soft_bg = _normalize_move_force(
                        sample_vector_field(-bg_gx, -bg_gy, soft_pos, benchmark.canvas_width, benchmark.canvas_height)
                    )
                    soft_targets, soft_target_weight = compute_soft_anchor_targets(
                        soft_pos,
                        hard_pos,
                        soft_hard_index_d,
                        soft_hard_weight_d,
                        soft_fixed_sum_d,
                        soft_fixed_weight_d,
                    )
                    soft_anchor = soft_targets - soft_pos
                    soft_anchor = _normalize_move_force(soft_anchor)
                    soft_anchor_gate = torch.clamp(soft_target_weight.unsqueeze(1), min=0.0, max=1.5)
                    soft_anchor = soft_anchor * soft_anchor_gate

                    soft_total = (
                        float(soft_force_weights.get("plasma", 0.20)) * soft_plasma
                        + float(soft_force_weights.get("q_pressure", 0.10)) * soft_q
                        + float(soft_force_weights.get("rho_pressure", 0.0)) * soft_rho
                        + float(soft_force_weights.get("bg_pressure", 0.25)) * soft_bg
                        + float(soft_force_weights.get("anchor", 0.75)) * soft_anchor
                    )
                    if soft_fixed.numel() > 0 and bool(soft_fixed.any()):
                        soft_total[soft_fixed] = 0.0

                    soft_step = (soft_lr_scale * lr) * soft_total / torch.clamp(soft_precond, min=1e-6)
                    soft_norm = torch.linalg.norm(soft_step, dim=1, keepdim=True)
                    if soft_trust_radius > 0.0:
                        soft_step = soft_step * torch.clamp(soft_trust_radius / torch.clamp(soft_norm, min=1e-9), max=1.0)
                    soft_pos = soft_pos + soft_step
                    soft_pos = project_to_canvas(
                        soft_pos,
                        soft_sizes,
                        benchmark.canvas_width,
                        benchmark.canvas_height,
                        fixed_mask=soft_fixed,
                        fixed_positions=soft_fixed_pos,
                    )

                if allow_exact_snapshots and snapshot_every > 0 and (it + 1) % snapshot_every == 0:
                    cand = placement.clone()
                    cand[:n] = hard_pos.detach().cpu()
                    if int(cand.shape[0]) > n and soft_pos.numel() > 0:
                        cand[n:] = soft_pos.detach().cpu()
                    cand = self._repair_snapshot_for_exact(cand, benchmark)
                    exact = self._consume_exact_eval(
                        cand,
                        benchmark,
                        plc,
                        force_final=False,
                    )
                    if exact is not None and (best_exact_cost is None or exact < best_exact_cost):
                        best_exact_cost = exact
                        best_exact_pos = cand.clone()

            run_soft_sync = (
                bool(soft_sync_cfg.get("enabled", False))
                and plc is not None
                and int(benchmark.num_soft_macros) > 0
                and soft_fill_ratio >= float(soft_sync_cfg.get("min_soft_fill", 0.45))
                and (time.time() - start_time) <= float(soft_sync_cfg.get("max_budget_frac", 0.70)) * total_budget
            )
            sync_stage_names = set(str(s) for s in soft_sync_cfg.get("stage_names", []))
            if run_soft_sync and sync_stage_names and stage_name not in sync_stage_names:
                run_soft_sync = False

            if run_soft_sync:
                sync_num_steps = soft_sync_cfg.get("num_steps", [4, 4, 4])
                sync_use_current = bool(soft_sync_cfg.get("use_current_loc", True))
                stage_place = placement.clone()
                stage_place[:n] = hard_pos.detach().cpu()
                if int(stage_place.shape[0]) > n and soft_pos.numel() > 0:
                    stage_place[n:] = soft_pos.detach().cpu()
                refreshed = self._optimize_soft_macros(
                    stage_place,
                    benchmark,
                    plc,
                    num_steps=sync_num_steps,
                    use_current_loc=sync_use_current,
                )
                if int(refreshed.shape[0]) > n:
                    soft_pos = refreshed[n:].to(device)
                    placement[n:] = refreshed[n:]
                if bool(soft_sync_cfg.get("reset_psi", False)):
                    psi_cache = None

        placement[:n] = hard_pos.detach().cpu()

        if best_exact_pos is not None:
            return best_exact_pos, best_exact_cost

        return placement, None

    def _run_post_legal_plasma_stage(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_offsets: torch.Tensor,
        anchor_targets: torch.Tensor,
        anchor_weights: torch.Tensor,
        start_time: float,
        total_budget: float,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        n = int(benchmark.num_hard_macros)
        if n <= 1:
            return placement, None

        cfg = dict(self.cfg.get("post_legal_global", {}))
        if not bool(cfg.get("enabled", False)):
            return placement, None

        max_budget_frac = float(cfg.get("max_budget_frac", 0.82))
        if time.time() - start_time > max_budget_frac * total_budget:
            return placement, None

        device = self._resolve_device(str(self.cfg.get("device", "auto")))
        hard_pos = placement[:n].to(device)
        hard_sizes = benchmark.macro_sizes[:n].to(device)
        hard_fixed = benchmark.macro_fixed[:n].to(device)
        hard_fixed_pos = hard_pos.clone()
        soft_pos = placement[n:].to(device) if int(placement.shape[0]) > n else torch.zeros(0, 2, device=device)
        soft_sizes = benchmark.macro_sizes[n:].to(device) if int(benchmark.macro_sizes.shape[0]) > n else torch.zeros(0, 2, device=device)

        edge_index_d = edge_index.to(device)
        edge_weight_d = edge_weight.to(device)
        edge_offsets_d = edge_offsets.to(device) if edge_offsets.numel() > 0 else torch.zeros(0, 2, 2, dtype=torch.float32, device=device)
        anchor_targets_d = anchor_targets.to(device)
        anchor_weights_d = anchor_weights.to(device)

        steps = max(1, int(cfg.get("steps", 6)))
        lr = float(cfg.get("lr", 0.02))
        grid_size = max(8, int(cfg.get("grid_size", 40)))
        trust_radius = float(cfg.get("trust_radius_frac", 0.012)) * max(
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
        )
        relegalize_every = max(0, int(cfg.get("relegalize_every", 4)))
        rhs_weights = dict(cfg.get("rhs_weights", {}))
        force_weights = dict(cfg.get("force_weights", {}))
        pde_cfg = dict(self.cfg.get("global_stage", {}).get("pde", {}))
        control_cfg = dict(cfg.get("plasma_control", self.cfg.get("global_stage", {}).get("plasma_control", {})))
        barrier_cfg = dict(cfg.get("transport_barrier", self.cfg.get("global_stage", {}).get("transport_barrier", {})))
        sheath_cfg = dict(cfg.get("pin_sheath", self.cfg.get("global_stage", {}).get("pin_sheath", {})))
        soft_fill_ratio = benchmark_soft_fill_ratio(benchmark)
        anchor_cfg = dict(cfg.get("soft_anchor", self.cfg.get("global_stage", {}).get("soft_anchor", {})))
        anchor_stage_scale = compute_soft_fill_scale(soft_fill_ratio, anchor_cfg, disabled_value=1.0)
        anchor_conf = compute_anchor_confidence_scale(anchor_weights_d, anchor_cfg).unsqueeze(1)
        precond_cfg = dict(cfg.get("preconditioner", self.cfg.get("global_stage", {}).get("preconditioner", {})))
        move_precond = compute_transport_preconditioner(
            hard_sizes,
            edge_index_d,
            edge_weight_d,
            precond_cfg,
        )
        psi_cache: Optional[torch.Tensor] = None
        control_scales = compute_plasma_control_scales({}, {"enabled": False})

        legal_cfg = self.cfg["legalizer"]
        for it in range(steps):
            if time.time() - start_time > max_budget_frac * total_budget:
                break

            fields = compute_plasma_forces(
                positions=hard_pos,
                sizes=hard_sizes,
                edge_index=edge_index_d,
                edge_weight=edge_weight_d,
                canvas_width=benchmark.canvas_width,
                canvas_height=benchmark.canvas_height,
                grid_size=grid_size,
                pde_cfg=pde_cfg,
                rhs_weights={
                    **rhs_weights,
                    "rho": float(rhs_weights.get("rho", 1.0)) * float(control_scales.get("rho_rhs_scale", 1.0)),
                    "q": float(rhs_weights.get("q", 0.5)) * float(control_scales.get("q_rhs_scale", 1.0)),
                },
                background_positions=soft_pos,
                background_sizes=soft_sizes,
                anchor_targets=anchor_targets_d,
                anchor_weights=anchor_weights_d,
                edge_offsets=edge_offsets_d,
                previous_psi=psi_cache,
                bundle_terms=self._active_bundle_terms,
                pin_terms=self._active_pin_terms,
                pin_edge_profile=self._active_pin_edge_profile,
            )

            plasma_force = fields["plasma_force"]
            hall_force = fields.get("hall_force", torch.zeros_like(hard_pos))
            chi_force = fields.get("chi_force", torch.zeros_like(hard_pos))
            dia_force = fields.get("dia_force", torch.zeros_like(hard_pos))
            balloon_force = fields.get("balloon_force", torch.zeros_like(hard_pos))
            hot_force = fields.get("hot_force", torch.zeros_like(hard_pos))
            cold_force = fields.get("cold_force", torch.zeros_like(hard_pos))
            rho_force = fields.get("rho_force", torch.zeros_like(hard_pos))
            q_force = fields.get("q_force", torch.zeros_like(hard_pos))
            bg_force = fields.get("bg_force", torch.zeros_like(hard_pos))
            channel_force = compute_pin_edge_sheath_force(
                hard_pos,
                hard_sizes,
                self._active_pin_edge_profile,
                sheath_cfg,
                benchmark.canvas_width,
                benchmark.canvas_height,
            )
            net_force = fields["net_force"]
            repulsion_force = fields["repulsion_force"]
            psi_cache = fields["psi"]
            control_scales = compute_plasma_control_scales(fields, control_cfg)

            anchor_force = torch.zeros_like(hard_pos)
            anchor_mask = anchor_weights_d > 1e-8
            if bool(anchor_mask.any()):
                anchor_force[anchor_mask] = anchor_targets_d[anchor_mask] - hard_pos[anchor_mask]
                anorm = torch.linalg.norm(anchor_force, dim=1)
                aden = torch.quantile(anorm, 0.90) if n >= 10 else torch.max(anorm)
                av = float(aden.item()) if anorm.numel() > 0 else 0.0
                if av > 1e-6:
                    anchor_force = anchor_force / av
                anchor_gate = compute_anchor_alignment_gate(
                    anchor_force,
                    plasma_force,
                    q_force,
                    rho_force,
                    repulsion_force,
                    anchor_cfg,
                ).unsqueeze(1)
                anchor_force = anchor_force * anchor_conf * anchor_gate

            transport_force = (
                float(force_weights.get("plasma", 1.0)) * plasma_force
                + float(force_weights.get("hall", 0.0)) * float(control_scales.get("hall_scale", 1.0)) * hall_force
                + float(force_weights.get("chi", 0.0)) * chi_force
                + float(force_weights.get("dia", 0.0)) * float(control_scales.get("dia_scale", 1.0)) * dia_force
                + float(force_weights.get("balloon", 0.0)) * float(control_scales.get("balloon_scale", 1.0)) * balloon_force
                + float(force_weights.get("hot", 0.0)) * float(control_scales.get("hot_scale", 1.0)) * hot_force
                + float(force_weights.get("cold", 0.0)) * float(control_scales.get("cold_scale", 1.0)) * cold_force
                + float(force_weights.get("rho_pressure", 0.0)) * float(control_scales.get("rho_force_scale", 1.0)) * rho_force
                + float(force_weights.get("q_pressure", 0.5)) * float(control_scales.get("q_force_scale", 1.0)) * q_force
                + float(force_weights.get("pin_pressure", 0.0)) * fields.get("pin_force", torch.zeros_like(hard_pos))
                + float(force_weights.get("pin_channel", 0.0)) * channel_force
                + float(force_weights.get("bg_pressure", 0.0)) * bg_force
                + float(force_weights.get("porosity_pressure", 0.0)) * fields.get("porosity_force", torch.zeros_like(hard_pos))
                + float(force_weights.get("net", 0.12)) * float(control_scales.get("net_scale", 1.0)) * net_force
            )
            transport_force = apply_transport_barrier(
                transport_force,
                plasma_force,
                fields.get("q_samples", torch.zeros(0, device=device)),
                fields.get("rho_samples", torch.zeros(0, device=device)),
                barrier_cfg,
            )
            total_force = (
                transport_force
                + float(force_weights.get("repulsion", 0.35)) * float(control_scales.get("repulsion_scale", 1.0)) * repulsion_force
                + float(force_weights.get("anchor", 0.0)) * anchor_stage_scale * float(control_scales.get("anchor_scale", 1.0)) * anchor_force
            )
            if hard_fixed.any():
                total_force[hard_fixed] = 0.0

            step_vec = lr * total_force / torch.clamp(move_precond, min=1e-6)
            norms = torch.linalg.norm(step_vec, dim=1, keepdim=True)
            dynamic_trust = trust_radius * float(control_scales.get("trust_scale", 1.0))
            if dynamic_trust > 0.0:
                scale = torch.clamp(
                    dynamic_trust / torch.clamp(norms, min=1e-9),
                    max=1.0,
                )
                step_vec = step_vec * scale

            hard_pos = hard_pos + step_vec
            hard_pos = project_to_canvas(
                hard_pos,
                hard_sizes,
                benchmark.canvas_width,
                benchmark.canvas_height,
                fixed_mask=hard_fixed,
                fixed_positions=hard_fixed_pos,
            )

            if relegalize_every > 0 and (it + 1) % relegalize_every == 0:
                tmp = placement.clone()
                tmp[:n] = hard_pos.detach().cpu()
                tmp = legalize_hard_macros(
                    tmp,
                    benchmark.macro_sizes,
                    benchmark.macro_fixed,
                    benchmark.canvas_width,
                    benchmark.canvas_height,
                    n,
                    gap=float(legal_cfg.get("gap", 1e-4)),
                    max_iters=min(35, int(legal_cfg.get("max_iters", 80))),
                    fallback_iters=min(25, int(legal_cfg.get("fallback_iters", 60))),
                )
                tmp = sanitize_canvas_bounds(
                    tmp,
                    benchmark.macro_sizes,
                    benchmark.canvas_width,
                    benchmark.canvas_height,
                    fixed_mask=benchmark.macro_fixed,
                    fixed_positions=benchmark.macro_positions,
                    safety_eps=1e-6,
                )
                hard_pos = tmp[:n].to(device)

        placement[:n] = hard_pos.detach().cpu()
        if int(placement.shape[0]) > n and soft_pos.numel() > 0:
            placement[n:] = soft_pos.detach().cpu()
        placement = legalize_hard_macros(
            placement,
            benchmark.macro_sizes,
            benchmark.macro_fixed,
            benchmark.canvas_width,
            benchmark.canvas_height,
            n,
            gap=float(legal_cfg.get("gap", 1e-4)),
            max_iters=min(45, int(legal_cfg.get("max_iters", 80))),
            fallback_iters=min(30, int(legal_cfg.get("fallback_iters", 60))),
        )
        placement = sanitize_canvas_bounds(
            placement,
            benchmark.macro_sizes,
            benchmark.canvas_width,
            benchmark.canvas_height,
            fixed_mask=benchmark.macro_fixed,
            fixed_positions=benchmark.macro_positions,
            safety_eps=1e-6,
        )
        return placement, None

    def _run_exact_local_refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        anchor_targets: torch.Tensor,
        anchor_weights: torch.Tensor,
        start_time: float,
        total_budget: float,
        edge_offsets: Optional[torch.Tensor] = None,
        cfg_section: str = "exact_local_refine",
    ) -> Tuple[torch.Tensor, Optional[float]]:
        cfg = dict(self.cfg.get(cfg_section, {}))
        if not bool(cfg.get("enabled", False)):
            return placement, None

        n = int(benchmark.num_hard_macros)
        if n <= 1:
            return placement, None

        trials = int(cfg.get("trials", {}).get(self.mode, cfg.get("trials", {}).get("local", 0)))
        if trials <= 0:
            return placement, None

        span = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        sigma0 = float(cfg.get("sigma_frac", 0.01)) * span
        cluster_radius = float(cfg.get("cluster_radius_frac", 0.06)) * span
        cluster_prob = float(cfg.get("cluster_prob", 0.35))
        proposal_beam = max(1, int(cfg.get("proposal_beam", 1)))
        plasma_bias_scale = float(cfg.get("plasma_bias", 0.25))
        hotspot_prob = float(cfg.get("hotspot_prob", 0.0))
        hotspot_topk = max(1, int(cfg.get("hotspot_topk", 8)))
        hotspot_step_frac = float(cfg.get("hotspot_step_frac", 0.012))
        hotspot_cluster_radius = float(cfg.get("hotspot_cluster_radius_frac", 0.08)) * span
        hotspot_q_weight = float(cfg.get("hotspot_q_weight", 1.0))
        hotspot_dia_weight = float(cfg.get("hotspot_dia_weight", 0.0))
        hotspot_hall_weight = float(cfg.get("hotspot_hall_weight", 0.0))
        hotspot_balloon_weight = float(cfg.get("hotspot_balloon_weight", 0.0))
        hotspot_plasma_weight = float(cfg.get("hotspot_plasma_weight", 0.0))
        hotspot_channel_weight = float(cfg.get("hotspot_channel_weight", 0.0))
        hotspot_channel_score_weight = float(cfg.get("hotspot_channel_score_weight", 0.0))
        hotspot_step_boost = float(cfg.get("hotspot_step_boost", 0.0))
        hotspot_score_power = float(cfg.get("hotspot_score_power", 1.0))
        hotspot_cluster_taper = float(cfg.get("hotspot_cluster_taper", 1.0))
        recompute_every = max(1, int(cfg.get("recompute_every", 4)))
        guide_grid_size = max(8, int(cfg.get("guide_grid_size", 24)))
        legalize_iters = max(8, int(cfg.get("legalize_iters", 25)))
        raw_hotspot_mode_weights = cfg.get("hotspot_mode_weights", {})
        hotspot_mode_items: List[Tuple[str, float]] = []
        if isinstance(raw_hotspot_mode_weights, dict):
            for name, weight in raw_hotspot_mode_weights.items():
                try:
                    w = float(weight)
                except (TypeError, ValueError):
                    continue
                if w > 0.0:
                    hotspot_mode_items.append((str(name).lower(), w))
        if not hotspot_mode_items:
            hotspot_mode_items = [("forward", 1.0)]
        hotspot_mode_total = sum(weight for _, weight in hotspot_mode_items)
        hotspot_mode_items = [(name, weight / max(hotspot_mode_total, 1e-9)) for name, weight in hotspot_mode_items]
        sheath_cfg = dict(self.cfg.get("global_stage", {}).get("pin_sheath", {}))
        sheath_cfg.update(dict(cfg.get("pin_sheath", {})))

        legal_cfg = self.cfg["legalizer"]
        legal_anchor_mode = str(legal_cfg.get("anchor_mode", "current")).lower()
        legal_anchor_strength = float(legal_cfg.get("anchor_strength", 0.0))
        legal_restore_iters = int(legal_cfg.get("restore_iters", 0))
        hard_sizes = benchmark.macro_sizes[:n].float()
        hard_fixed = benchmark.macro_fixed[:n].clone()
        movable = torch.where(~hard_fixed)[0].tolist()
        if not movable:
            return placement, None

        soft_cluster_group_cfg = dict(cfg.get("soft_cluster_groups", {}))
        soft_cluster_groups_enabled = bool(soft_cluster_group_cfg.get("enabled", False))
        soft_cluster_groups: List[List[int]] = []
        soft_cluster_group_by_macro: Dict[int, List[int]] = {}
        if soft_cluster_groups_enabled:
            soft_cluster_group_info = extract_soft_cluster_macro_group_info_from_plc(
                benchmark,
                plc,
                max_edges=int(soft_cluster_group_cfg.get("max_edges", 64)),
                min_hard_degree=int(soft_cluster_group_cfg.get("min_hard_degree", 2)),
                min_shared_clusters=int(soft_cluster_group_cfg.get("min_shared_clusters", 2)),
                use_pin_offsets=bool(soft_cluster_group_cfg.get("use_pin_offsets", False)),
                weight_mode=str(soft_cluster_group_cfg.get("weight_mode", "unit")),
                degree_power=float(soft_cluster_group_cfg.get("degree_power", 1.0)),
                max_group_size=int(soft_cluster_group_cfg.get("max_group_size", 6)),
                max_groups=int(soft_cluster_group_cfg.get("max_groups", 8)),
            )
            soft_cluster_groups = [group for group, _score in soft_cluster_group_info]
            soft_cluster_group_scores = [float(score) for _group, score in soft_cluster_group_info]
            for group, score in zip(soft_cluster_groups, soft_cluster_group_scores):
                for macro_idx in group:
                    prev = soft_cluster_group_by_macro.get(int(macro_idx))
                    if prev is None or len(group) < len(prev):
                        soft_cluster_group_by_macro[int(macro_idx)] = group
            soft_cluster_macro_score: Dict[int, float] = {}
            for group, score in zip(soft_cluster_groups, soft_cluster_group_scores):
                for macro_idx in group:
                    soft_cluster_macro_score[int(macro_idx)] = max(
                        float(soft_cluster_macro_score.get(int(macro_idx), 0.0)),
                        float(score),
                    )
        else:
            soft_cluster_group_scores = []
            soft_cluster_macro_score = {}
        soft_cluster_group_move_prob = float(soft_cluster_group_cfg.get("group_move_prob", 0.0))
        soft_cluster_group_step_scale = max(float(soft_cluster_group_cfg.get("group_step_scale", 1.0)), 0.0)
        soft_cluster_group_score_power = max(float(soft_cluster_group_cfg.get("score_power", 1.0)), 1e-6)
        soft_cluster_group_hotspot_bonus = float(soft_cluster_group_cfg.get("hotspot_bonus_weight", 0.0))
        soft_cluster_group_weighted_sampling = bool(soft_cluster_group_cfg.get("weighted_sampling", False))
        movable_soft_cluster_groups = [
            [int(i) for i in group if int(i) in movable]
            for group in soft_cluster_groups
        ]
        movable_soft_cluster_groups = [group for group in movable_soft_cluster_groups if len(group) >= 2]
        movable_soft_cluster_group_scores: List[float] = []
        if movable_soft_cluster_groups:
            for group in movable_soft_cluster_groups:
                score = 0.0
                for macro_idx in group:
                    score = max(score, float(soft_cluster_macro_score.get(int(macro_idx), 0.0)))
                movable_soft_cluster_group_scores.append(max(score, 0.0))

        best = placement.clone()
        best_cost = self._evaluate_exact_proxy(best, benchmark, plc)
        if best_cost is None:
            return placement, None

        rng = random.Random(self.seed + 7919)
        plasma_bias: Optional[torch.Tensor] = None
        hotspot_indices: List[int] = []
        hotspot_dir: Optional[torch.Tensor] = None
        hotspot_strengths: Dict[int, float] = {}
        guide_pde_cfg = dict(self.cfg.get("global_stage", {}).get("pde", {}))
        guide_pde_cfg["picard_outer"] = max(1, int(guide_pde_cfg.get("picard_outer", 4)) // 2)
        guide_pde_cfg["gs_iters"] = max(8, int(guide_pde_cfg.get("gs_iters", 28)) // 2)
        guide_rhs = dict(self.cfg.get("global_stage", {}).get("stages", [{}])[-1].get("rhs_weights", {}))

        def choose_hotspot_mode() -> str:
            pick = rng.random()
            accum = 0.0
            for name, weight in hotspot_mode_items:
                accum += weight
                if pick <= accum:
                    return name
            return hotspot_mode_items[-1][0]

        def _normalize_dir(vec: torch.Tensor) -> torch.Tensor:
            norm = torch.linalg.norm(vec)
            if float(norm.item()) <= 1e-9:
                return torch.tensor([1.0, 0.0], dtype=vec.dtype, device=vec.device)
            return vec / norm

        def _apply_hotspot_move(
            cand_hard: torch.Tensor,
            base: torch.Tensor,
            center: int,
            direction: torch.Tensor,
            step_scale: float,
            mode: str,
            score_scale: float,
        ) -> bool:
            dir_vec = _normalize_dir(direction)
            tangential = torch.stack((-dir_vec[1], dir_vec[0]))

            if mode == "reverse":
                basis = -dir_vec
                use_cluster = False
            elif mode == "shear_left":
                basis = tangential
                use_cluster = False
            elif mode == "shear_right":
                basis = -tangential
                use_cluster = False
            elif mode == "diag_left":
                basis = _normalize_dir(dir_vec + 0.55 * tangential)
                use_cluster = False
            elif mode == "diag_right":
                basis = _normalize_dir(dir_vec - 0.55 * tangential)
                use_cluster = False
            elif mode == "cluster_forward":
                basis = dir_vec
                use_cluster = True
            elif mode == "cluster_shear_left":
                basis = tangential
                use_cluster = True
            elif mode == "cluster_shear_right":
                basis = -tangential
                use_cluster = True
            else:
                basis = dir_vec
                use_cluster = False

            scaled_step = step_scale * (1.0 + hotspot_step_boost * max(0.0, score_scale))
            if use_cluster and len(movable) >= 3:
                group = soft_cluster_group_by_macro.get(int(center))
                if group is not None and len(group) >= 2:
                    moved_any = False
                    for i in group:
                        if bool(hard_fixed[i]):
                            continue
                        cand_hard[i, 0] += float(basis[0].item()) * scaled_step
                        cand_hard[i, 1] += float(basis[1].item()) * scaled_step
                        if plasma_bias is not None:
                            cand_hard[i, 0] += plasma_bias_scale * float(plasma_bias[i, 0].item()) * span * 0.004
                            cand_hard[i, 1] += plasma_bias_scale * float(plasma_bias[i, 1].item()) * span * 0.004
                        moved_any = True
                    if moved_any:
                        return True
                cx = float(base[center, 0].item())
                cy = float(base[center, 1].item())
                moved_any = False
                for i in movable:
                    px = float(base[i, 0].item())
                    py = float(base[i, 1].item())
                    dist = math.sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy))
                    if dist <= hotspot_cluster_radius:
                        radial = 1.0 - min(1.0, dist / max(hotspot_cluster_radius, 1e-9))
                        taper = radial ** max(hotspot_cluster_taper, 1e-6)
                        cand_hard[i, 0] += float(basis[0].item()) * scaled_step * taper
                        cand_hard[i, 1] += float(basis[1].item()) * scaled_step * taper
                        if plasma_bias is not None:
                            cand_hard[i, 0] += plasma_bias_scale * float(plasma_bias[i, 0].item()) * span * 0.004 * taper
                            cand_hard[i, 1] += plasma_bias_scale * float(plasma_bias[i, 1].item()) * span * 0.004 * taper
                        moved_any = True
                return moved_any

            cand_hard[center, 0] += float(basis[0].item()) * scaled_step
            cand_hard[center, 1] += float(basis[1].item()) * scaled_step
            if plasma_bias is not None:
                cand_hard[center, 0] += plasma_bias_scale * float(plasma_bias[center, 0].item()) * span * 0.006
                cand_hard[center, 1] += plasma_bias_scale * float(plasma_bias[center, 1].item()) * span * 0.006
            return True

        def propose_full_candidate(base: torch.Tensor, sigma: float) -> Optional[torch.Tensor]:
            cand_hard = base.clone()
            use_group = (
                soft_cluster_groups_enabled
                and soft_cluster_group_move_prob > 0.0
                and movable_soft_cluster_groups
                and rng.random() < soft_cluster_group_move_prob
            )
            use_hotspot = hotspot_prob > 0.0 and hotspot_dir is not None and hotspot_indices and rng.random() < hotspot_prob
            if use_group:
                if soft_cluster_group_weighted_sampling and movable_soft_cluster_group_scores:
                    total = sum(max(score, 0.0) ** soft_cluster_group_score_power for score in movable_soft_cluster_group_scores)
                    if total > 1e-12:
                        pick = rng.random() * total
                        accum = 0.0
                        chosen = movable_soft_cluster_groups[-1]
                        for group, score in zip(movable_soft_cluster_groups, movable_soft_cluster_group_scores):
                            accum += max(score, 0.0) ** soft_cluster_group_score_power
                            if pick <= accum:
                                chosen = group
                                break
                        group = list(chosen)
                    else:
                        group = list(rng.choice(movable_soft_cluster_groups))
                else:
                    group = list(rng.choice(movable_soft_cluster_groups))
                dir_vec = None
                if hotspot_dir is not None:
                    dir_vec = hotspot_dir[group].mean(dim=0)
                    if float(torch.linalg.norm(dir_vec).item()) <= 1e-9:
                        dir_vec = None
                if dir_vec is not None:
                    basis = _normalize_dir(dir_vec)
                    step_scale = sigma * soft_cluster_group_step_scale * rng.uniform(0.75, 1.25)
                    for i in group:
                        cand_hard[i, 0] += float(basis[0].item()) * step_scale
                        cand_hard[i, 1] += float(basis[1].item()) * step_scale
                        if plasma_bias is not None:
                            cand_hard[i, 0] += plasma_bias_scale * float(plasma_bias[i, 0].item()) * span * 0.004
                            cand_hard[i, 1] += plasma_bias_scale * float(plasma_bias[i, 1].item()) * span * 0.004
                else:
                    dx = float(rng.gauss(0.0, sigma * soft_cluster_group_step_scale))
                    dy = float(rng.gauss(0.0, sigma * soft_cluster_group_step_scale))
                    for i in group:
                        cand_hard[i, 0] += dx
                        cand_hard[i, 1] += dy
                        if plasma_bias is not None:
                            cand_hard[i, 0] += plasma_bias_scale * float(plasma_bias[i, 0].item()) * span * 0.004
                            cand_hard[i, 1] += plasma_bias_scale * float(plasma_bias[i, 1].item()) * span * 0.004
            elif use_hotspot:
                center = int(rng.choice(hotspot_indices))
                dir_vec = hotspot_dir[center]
                step_scale = hotspot_step_frac * span * rng.uniform(0.75, 1.25)
                score_scale = float(hotspot_strengths.get(center, 0.0))
                mode = choose_hotspot_mode()
                if mode == "cluster_random":
                    mode = "cluster_forward" if rng.random() < 0.5 else "cluster_shear_left"
                if mode == "forward" and rng.random() < cluster_prob and len(movable) >= 3:
                    mode = "cluster_forward"
                moved_any = _apply_hotspot_move(cand_hard, base, center, dir_vec, step_scale, mode, score_scale)
                if not moved_any:
                    return None
            elif rng.random() < cluster_prob and len(movable) >= 3:
                center = int(rng.choice(movable))
                group = soft_cluster_group_by_macro.get(int(center))
                if group is not None and len(group) >= 2:
                    dx = float(rng.gauss(0.0, sigma))
                    dy = float(rng.gauss(0.0, sigma))
                    moved_any = False
                    for i in group:
                        if bool(hard_fixed[i]):
                            continue
                        cand_hard[i, 0] += dx
                        cand_hard[i, 1] += dy
                        if plasma_bias is not None:
                            cand_hard[i, 0] += plasma_bias_scale * float(plasma_bias[i, 0].item()) * span * 0.006
                            cand_hard[i, 1] += plasma_bias_scale * float(plasma_bias[i, 1].item()) * span * 0.006
                        moved_any = True
                    if not moved_any:
                        return None
                else:
                    cx = float(base[center, 0].item())
                    cy = float(base[center, 1].item())
                    dx = float(rng.gauss(0.0, sigma))
                    dy = float(rng.gauss(0.0, sigma))
                    moved_any = False
                    for i in movable:
                        px = float(base[i, 0].item())
                        py = float(base[i, 1].item())
                        if (px - cx) * (px - cx) + (py - cy) * (py - cy) <= cluster_radius * cluster_radius:
                            cand_hard[i, 0] += dx
                            cand_hard[i, 1] += dy
                            if plasma_bias is not None:
                                cand_hard[i, 0] += plasma_bias_scale * float(plasma_bias[i, 0].item()) * span * 0.006
                                cand_hard[i, 1] += plasma_bias_scale * float(plasma_bias[i, 1].item()) * span * 0.006
                            moved_any = True
                    if not moved_any:
                        return None
            else:
                i = int(rng.choice(movable))
                cand_hard[i, 0] += float(rng.gauss(0.0, sigma))
                cand_hard[i, 1] += float(rng.gauss(0.0, sigma))
                if plasma_bias is not None:
                    cand_hard[i, 0] += plasma_bias_scale * float(plasma_bias[i, 0].item()) * span * 0.010
                    cand_hard[i, 1] += plasma_bias_scale * float(plasma_bias[i, 1].item()) * span * 0.010

            cand_hard = project_to_canvas(
                cand_hard,
                hard_sizes,
                benchmark.canvas_width,
                benchmark.canvas_height,
                fixed_mask=hard_fixed,
                fixed_positions=base,
            )

            full = best.clone()
            full[:n] = cand_hard
            full = legalize_hard_macros(
                full,
                benchmark.macro_sizes,
                benchmark.macro_fixed,
                benchmark.canvas_width,
                benchmark.canvas_height,
                n,
                gap=float(legal_cfg.get("gap", 1e-4)),
                max_iters=legalize_iters,
                fallback_iters=max(6, legalize_iters // 2),
                anchor_positions=self._resolve_legal_anchor_positions(benchmark, full, legal_anchor_mode),
                anchor_strength=legal_anchor_strength,
                restore_iters=legal_restore_iters,
            )
            full = sanitize_canvas_bounds(
                full,
                benchmark.macro_sizes,
                benchmark.canvas_width,
                benchmark.canvas_height,
                fixed_mask=benchmark.macro_fixed,
                fixed_positions=benchmark.macro_positions,
                safety_eps=1e-6,
            )
            return full

        for t in range(trials):
            if (time.time() - start_time) > 0.94 * total_budget:
                break

            alpha = float(t) / max(1, trials - 1)
            sigma = sigma0 * (0.35 + 0.65 * (1.0 - alpha))

            hard_pos = best[:n].clone()
            if plasma_bias_scale > 0.0 and (plasma_bias is None or (t % recompute_every == 0)):
                fields = compute_plasma_forces(
                    positions=hard_pos,
                    sizes=hard_sizes,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    canvas_width=benchmark.canvas_width,
                    canvas_height=benchmark.canvas_height,
                    grid_size=guide_grid_size,
                    pde_cfg=guide_pde_cfg,
                    rhs_weights=guide_rhs,
                    anchor_targets=anchor_targets,
                    anchor_weights=anchor_weights,
                    edge_offsets=edge_offsets,
                    previous_psi=None,
                    bundle_terms=self._active_bundle_terms,
                    pin_terms=self._active_pin_terms,
                    pin_edge_profile=self._active_pin_edge_profile,
                )
                channel_force = compute_pin_edge_sheath_force(
                    hard_pos,
                    hard_sizes,
                    self._active_pin_edge_profile,
                    sheath_cfg,
                    benchmark.canvas_width,
                    benchmark.canvas_height,
                )
                plasma_bias = fields["plasma_force"]
                hotspot_dir = (
                    hotspot_q_weight * fields.get("q_force", torch.zeros_like(hard_pos))
                    + hotspot_dia_weight * fields.get("dia_force", torch.zeros_like(hard_pos))
                    + hotspot_hall_weight * fields.get("hall_force", torch.zeros_like(hard_pos))
                    + hotspot_balloon_weight * fields.get("balloon_force", torch.zeros_like(hard_pos))
                    + hotspot_plasma_weight * fields.get("plasma_force", torch.zeros_like(hard_pos))
                    + hotspot_channel_weight * channel_force
                )
                dir_norm = torch.linalg.norm(hotspot_dir, dim=1, keepdim=True)
                hotspot_dir = hotspot_dir / torch.clamp(dir_norm, min=1e-9)
                q_samples = fields.get("q_samples", torch.zeros(n, dtype=torch.float32, device=hard_pos.device)).float()
                channel_norm = torch.linalg.norm(channel_force, dim=1)
                channel_ref = torch.quantile(channel_norm, 0.90) if channel_norm.numel() >= 10 else torch.max(channel_norm)
                channel_denom = max(float(channel_ref.item()), 1e-6) if channel_norm.numel() > 0 else 1.0
                movable_q = []
                cluster_score_ref = max([float(v) for v in soft_cluster_macro_score.values()], default=0.0)
                for i in movable:
                    score = float(q_samples[i].item())
                    if hotspot_channel_score_weight > 0.0:
                        score += hotspot_channel_score_weight * float(channel_norm[i].item()) / channel_denom
                    if soft_cluster_group_hotspot_bonus > 0.0 and cluster_score_ref > 1e-12:
                        score += soft_cluster_group_hotspot_bonus * float(soft_cluster_macro_score.get(int(i), 0.0)) / cluster_score_ref
                    movable_q.append((score, int(i)))
                movable_q.sort(reverse=True)
                hotspot_indices = [idx for _, idx in movable_q[:hotspot_topk]]
                hotspot_strengths = {}
                if hotspot_indices:
                    hot_vals = [score for score, idx in movable_q[:hotspot_topk] if idx in hotspot_indices]
                    if hot_vals:
                        q_lo = min(hot_vals)
                        q_hi = max(hot_vals)
                        q_span = max(q_hi - q_lo, 1e-6)
                        for score, idx in movable_q[:hotspot_topk]:
                            if idx not in hotspot_indices:
                                continue
                            norm_score = max(0.0, (score - q_lo) / q_span)
                            hotspot_strengths[idx] = norm_score ** max(hotspot_score_power, 1e-6)
                if not hotspot_indices:
                    hotspot_indices = movable[:]

            best_full: Optional[torch.Tensor] = None
            best_approx = float("inf")
            for _ in range(proposal_beam):
                full = propose_full_candidate(hard_pos, sigma)
                if full is None:
                    continue
                approx = self._approximate_total_cost(
                    full[:n].float(),
                    hard_sizes,
                    benchmark.canvas_width,
                    benchmark.canvas_height,
                    edge_index,
                    edge_weight,
                    edge_offsets=edge_offsets,
                    anchor_targets=anchor_targets,
                    anchor_weights=anchor_weights,
                )
                if approx < best_approx:
                    best_approx = approx
                    best_full = full

            if best_full is None:
                continue

            exact = self._evaluate_exact_proxy(best_full, benchmark, plc)
            if exact is not None and exact < best_cost:
                best = best_full
                best_cost = exact

        return best, best_cost

    def _run_plasma_guided_sa(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        anchor_targets: torch.Tensor,
        anchor_weights: torch.Tensor,
        start_time: float,
        total_budget: float,
        allow_exact_sync: bool,
        edge_offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        sa_cfg = self.cfg["sa"]
        legal_cfg = self.cfg["legalizer"]
        guide_cfg = sa_cfg.get("plasma_guidance", {})
        strict_legality_filter = bool(sa_cfg.get("strict_legality_filter", False))

        n = int(benchmark.num_hard_macros)
        sizes = benchmark.macro_sizes[:n].float()
        fixed_mask = benchmark.macro_fixed[:n].clone()

        if n <= 1:
            return placement, None

        workers = int(sa_cfg["workers"].get(self.mode, sa_cfg["workers"].get("local", 8)))
        epochs = int(sa_cfg["epochs"].get(self.mode, sa_cfg["epochs"].get("local", 10)))
        proposals = int(
            sa_cfg["proposals_per_worker"].get(
                self.mode, sa_cfg["proposals_per_worker"].get("local", 80)
            )
        )

        adaptive_cfg = sa_cfg.get("adaptive_scale", {})
        if bool(adaptive_cfg.get("enabled", True)):
            ref_n = float(adaptive_cfg.get("reference_num_hard", 320.0))
            min_s = float(adaptive_cfg.get("min_scale", 0.85))
            max_s = float(adaptive_cfg.get("max_scale", 2.25))
            scale = max(min_s, min(max_s, n / max(ref_n, 1.0)))
            epochs = max(1, int(round(epochs * scale)))
            proposals = max(8, int(round(proposals * scale)))

        workers = max(1, workers)
        epochs = max(1, epochs)
        proposals = max(1, proposals)

        guidance_enabled = bool(guide_cfg.get("enabled", True))
        guide_grid_size = int(guide_cfg.get("grid_size", 24))
        guide_recompute_every = max(1, int(guide_cfg.get("recompute_every", 6)))
        guide_bias_scale = float(guide_cfg.get("bias_scale", 0.25))
        guide_q_bias_weight = float(guide_cfg.get("q_bias_weight", 0.0))
        guide_rho_bias_weight = float(guide_cfg.get("rho_bias_weight", 0.0))
        anchor_move_scale = float(sa_cfg.get("anchor_move_scale", 0.0))
        guide_rhs_weights = dict(guide_cfg.get("rhs_weights", {}))
        if not guide_rhs_weights:
            stages = self.cfg.get("global_stage", {}).get("stages", [])
            if stages:
                guide_rhs_weights = dict(stages[-1].get("rhs_weights", {}))

        guide_pde_cfg = dict(self.cfg.get("global_stage", {}).get("pde", {}))
        guide_pde_cfg["picard_outer"] = max(1, int(guide_pde_cfg.get("picard_outer", 4)) // 2)
        guide_pde_cfg["gs_iters"] = max(8, int(guide_pde_cfg.get("gs_iters", 28)) // 2)

        movable_indices = torch.where(~fixed_mask)[0].tolist()
        if not movable_indices:
            return placement, self._evaluate_exact_proxy(placement, benchmark, plc)

        adjacency = self._build_adjacency(n, edge_index)
        rng = random.Random(self.seed)

        hard_base = placement[:n].clone()
        hard_base = project_to_canvas(
            hard_base,
            sizes,
            benchmark.canvas_width,
            benchmark.canvas_height,
            fixed_mask=fixed_mask,
            fixed_positions=hard_base.clone(),
        )
        baseline_approx = self._approximate_total_cost(
            hard_base,
            sizes,
            benchmark.canvas_width,
            benchmark.canvas_height,
            edge_index,
            edge_weight,
            edge_offsets=edge_offsets,
            anchor_targets=anchor_targets,
            anchor_weights=anchor_weights,
        )

        states: List[WorkerState] = []
        max_span = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        jitter_scale = float(sa_cfg.get("shift_sigma_scale", 0.08)) * max_span

        for w in range(workers):
            if w == 0:
                pos = hard_base.clone()
            else:
                pos = hard_base.clone()
                for i in movable_indices:
                    pos[i, 0] += float(rng.gauss(0.0, jitter_scale * 0.15))
                    pos[i, 1] += float(rng.gauss(0.0, jitter_scale * 0.15))
                pos = project_to_canvas(
                    pos,
                    sizes,
                    benchmark.canvas_width,
                    benchmark.canvas_height,
                    fixed_mask=fixed_mask,
                    fixed_positions=hard_base,
                )

            full = placement.clone()
            full[:n] = pos
            full = legalize_hard_macros(
                full,
                benchmark.macro_sizes,
                benchmark.macro_fixed,
                benchmark.canvas_width,
                benchmark.canvas_height,
                n,
                gap=float(legal_cfg.get("gap", 1e-4)),
                max_iters=30,
                fallback_iters=20,
            )
            full = sanitize_canvas_bounds(
                full,
                benchmark.macro_sizes,
                benchmark.canvas_width,
                benchmark.canvas_height,
                fixed_mask=benchmark.macro_fixed,
                fixed_positions=benchmark.macro_positions,
                safety_eps=1e-6,
            )
            pos = full[:n].clone()

            approx = self._approximate_total_cost(
                pos,
                sizes,
                benchmark.canvas_width,
                benchmark.canvas_height,
                edge_index,
                edge_weight,
                edge_offsets=edge_offsets,
                anchor_targets=anchor_targets,
                anchor_weights=anchor_weights,
            )
            states.append(WorkerState(hard_pos=pos, approx_score=approx))

        elite_count = min(int(sa_cfg.get("elite_count", 2)), workers)
        sync_fraction = float(sa_cfg.get("sync_fraction", 0.10))
        sync_every = max(1, int(math.ceil(epochs * sync_fraction)))

        init_temp = float(sa_cfg.get("init_temperature", 1.0))
        min_temp = float(sa_cfg.get("min_temperature", 0.02))

        best_exact_cost: Optional[float] = None
        best_exact_full: Optional[torch.Tensor] = None

        for epoch in range(epochs):
            if time.time() - start_time > 0.92 * total_budget:
                break

            alpha = epoch / max(1, epochs - 1)
            temp = init_temp * ((min_temp / init_temp) ** alpha)

            for state in states:
                plasma_bias = getattr(state, "plasma_bias", None)
                if guidance_enabled and (plasma_bias is None or (epoch % guide_recompute_every == 0)):
                    p_fields = compute_plasma_forces(
                        positions=state.hard_pos,
                        sizes=sizes,
                        edge_index=edge_index,
                        edge_weight=edge_weight,
                        canvas_width=benchmark.canvas_width,
                        canvas_height=benchmark.canvas_height,
                        grid_size=guide_grid_size,
                        pde_cfg=guide_pde_cfg,
                        rhs_weights=guide_rhs_weights,
                        anchor_targets=anchor_targets,
                        anchor_weights=anchor_weights,
                        edge_offsets=edge_offsets,
                        previous_psi=None,
                        bundle_terms=self._active_bundle_terms,
                        pin_terms=self._active_pin_terms,
                        pin_edge_profile=self._active_pin_edge_profile,
                    )
                    plasma_bias = (
                        p_fields["plasma_force"]
                        + float(guide_cfg.get("dia_bias_weight", 0.0))
                        * p_fields.get("dia_force", torch.zeros_like(state.hard_pos))
                        + float(guide_cfg.get("balloon_bias_weight", 0.0))
                        * p_fields.get("balloon_force", torch.zeros_like(state.hard_pos))
                        + guide_q_bias_weight * p_fields.get("q_force", torch.zeros_like(state.hard_pos))
                        + guide_rho_bias_weight * p_fields.get("rho_force", torch.zeros_like(state.hard_pos))
                    )
                    state.plasma_bias = plasma_bias

                for _ in range(proposals):
                    if time.time() - start_time > 0.92 * total_budget:
                        break

                    candidate = state.hard_pos.clone()
                    moved = self._apply_move(
                        candidate,
                        state.hard_pos,
                        movable_indices,
                        adjacency,
                        sizes,
                        benchmark.canvas_width,
                        benchmark.canvas_height,
                        temp,
                        rng,
                        plasma_bias=plasma_bias,
                        plasma_bias_scale=guide_bias_scale,
                        anchor_targets=anchor_targets,
                        anchor_weights=anchor_weights,
                        anchor_move_scale=anchor_move_scale,
                    )
                    if not moved:
                        continue

                    candidate = project_to_canvas(
                        candidate,
                        sizes,
                        benchmark.canvas_width,
                        benchmark.canvas_height,
                        fixed_mask=fixed_mask,
                        fixed_positions=state.hard_pos,
                    )

                    if strict_legality_filter:
                        if _has_overlap_for_indices(
                            candidate,
                            sizes,
                            n,
                            moved,
                            gap=float(legal_cfg.get("gap", 1e-4)),
                        ):
                            continue

                    if len(moved) == 1:
                        delta_wl = approximate_wirelength_delta_for_shift(
                            state.hard_pos,
                            edge_index,
                            edge_weight,
                            moved[0],
                            candidate[moved[0]],
                            edge_offsets=edge_offsets,
                        )
                    else:
                        delta_wl = 0.0

                    new_score = self._approximate_total_cost(
                        candidate,
                        sizes,
                        benchmark.canvas_width,
                        benchmark.canvas_height,
                        edge_index,
                        edge_weight,
                        edge_offsets=edge_offsets,
                        anchor_targets=anchor_targets,
                        anchor_weights=anchor_weights,
                    )

                    if len(moved) == 1:
                        old_wl = approximate_wirelength_cost(
                            state.hard_pos,
                            edge_index,
                            edge_weight,
                            edge_offsets=edge_offsets,
                        )
                        new_wl_est = old_wl + delta_wl
                        new_wl_full = approximate_wirelength_cost(
                            candidate,
                            edge_index,
                            edge_weight,
                            edge_offsets=edge_offsets,
                        )
                        new_score = 0.8 * new_score + 0.2 * (new_score - new_wl_full + new_wl_est)

                    delta = new_score - state.approx_score
                    accept = delta <= 0.0 or rng.random() < math.exp(-delta / max(temp, 1e-8))

                    if accept:
                        state.hard_pos = candidate
                        state.approx_score = float(new_score)

            do_sync = ((epoch + 1) % sync_every == 0) or (epoch == epochs - 1)
            if not do_sync:
                continue

            for state in states:
                full = placement.clone()
                full[:n] = state.hard_pos

                full = legalize_hard_macros(
                    full,
                    benchmark.macro_sizes,
                    benchmark.macro_fixed,
                    benchmark.canvas_width,
                    benchmark.canvas_height,
                    n,
                    gap=float(legal_cfg.get("gap", 1e-4)),
                    max_iters=40,
                    fallback_iters=30,
                )

                if bool(sa_cfg.get("soft_opt_every_sync", True)):
                    full = self._optimize_soft_macros(
                        full,
                        benchmark,
                        plc,
                        num_steps=self.cfg["soft_macro"].get("sync_num_steps", [12, 12, 12]),
                        use_current_loc=False,
                    )

                full = sanitize_canvas_bounds(
                    full,
                    benchmark.macro_sizes,
                    benchmark.canvas_width,
                    benchmark.canvas_height,
                    fixed_mask=benchmark.macro_fixed,
                    fixed_positions=benchmark.macro_positions,
                    safety_eps=1e-6,
                )

                if allow_exact_sync:
                    exact = self._consume_exact_eval(
                        full,
                        benchmark,
                        plc,
                        force_final=False,
                    )
                    if exact is None:
                        state.exact_score = float("inf")
                        state.full_placement = full
                    else:
                        state.exact_score = exact
                        state.full_placement = full
                        if best_exact_cost is None or exact < best_exact_cost:
                            best_exact_cost = exact
                            best_exact_full = full.clone()
                else:
                    approx = self._approximate_total_cost(
                        full[:n],
                        sizes,
                        benchmark.canvas_width,
                        benchmark.canvas_height,
                        edge_index,
                        edge_weight,
                        edge_offsets=edge_offsets,
                        anchor_targets=anchor_targets,
                        anchor_weights=anchor_weights,
                    )
                    state.exact_score = float(approx)
                    state.full_placement = full
                    if best_exact_cost is None or state.exact_score < best_exact_cost:
                        best_exact_cost = state.exact_score
                        best_exact_full = full.clone()

            states.sort(key=lambda s: s.exact_score)
            elites = states[:elite_count]

            for wi in range(elite_count, workers):
                src = elites[wi % elite_count]
                dst = states[wi]
                dst.hard_pos = src.hard_pos.clone()

                for i in movable_indices:
                    dst.hard_pos[i, 0] += float(rng.gauss(0.0, jitter_scale * 0.03))
                    dst.hard_pos[i, 1] += float(rng.gauss(0.0, jitter_scale * 0.03))

                dst.hard_pos = project_to_canvas(
                    dst.hard_pos,
                    sizes,
                    benchmark.canvas_width,
                    benchmark.canvas_height,
                    fixed_mask=fixed_mask,
                    fixed_positions=src.hard_pos,
                )
                dst.approx_score = self._approximate_total_cost(
                    dst.hard_pos,
                    sizes,
                    benchmark.canvas_width,
                    benchmark.canvas_height,
                    edge_index,
                    edge_weight,
                    edge_offsets=edge_offsets,
                    anchor_targets=anchor_targets,
                    anchor_weights=anchor_weights,
                )
                dst.exact_score = float("inf")
                dst.full_placement = None
                dst.plasma_bias = None

        if best_exact_full is not None:
            post_approx = self._approximate_total_cost(
                best_exact_full[:n],
                sizes,
                benchmark.canvas_width,
                benchmark.canvas_height,
                edge_index,
                edge_weight,
                edge_offsets=edge_offsets,
                anchor_targets=anchor_targets,
                anchor_weights=anchor_weights,
            )
            min_improve = float(sa_cfg.get("min_improvement_frac", 0.0))
            if post_approx <= baseline_approx * (1.0 - min_improve):
                return best_exact_full, best_exact_cost
            return placement, None

        return placement, None

    @staticmethod
    def _build_adjacency(num_hard: int, edge_index: torch.Tensor) -> List[List[int]]:
        adj = [[] for _ in range(num_hard)]
        for i, j in edge_index.tolist():
            adj[i].append(j)
            adj[j].append(i)
        for i in range(num_hard):
            adj[i] = sorted(set(adj[i]))
        return adj

    def _sample_move(self, rng: random.Random) -> str:
        probs = self.cfg["sa"]["move_probs"]
        items = [
            ("shift", probs["shift"]),
            ("swap", probs["swap"]),
            ("attract", probs["attract"]),
            ("cluster", probs["cluster"]),
        ]
        flux_prob = float(probs.get("flux", 0.0))
        if flux_prob > 0.0:
            items.append(("flux", flux_prob))
        total = sum(p for _, p in items)
        r = rng.random() * total
        acc = 0.0
        for name, p in items:
            acc += p
            if r <= acc:
                return name
        return "shift"

    def _apply_move(
        self,
        candidate: torch.Tensor,
        current: torch.Tensor,
        movable_indices: Sequence[int],
        adjacency: Sequence[Sequence[int]],
        sizes: torch.Tensor,
        canvas_w: float,
        canvas_h: float,
        temp: float,
        rng: random.Random,
        plasma_bias: Optional[torch.Tensor] = None,
        plasma_bias_scale: float = 0.0,
        anchor_targets: Optional[torch.Tensor] = None,
        anchor_weights: Optional[torch.Tensor] = None,
        anchor_move_scale: float = 0.0,
    ) -> List[int]:
        move = self._sample_move(rng)

        if not movable_indices:
            return []

        span = max(canvas_w, canvas_h)
        sigma = float(self.cfg["sa"].get("shift_sigma_scale", 0.08)) * span * max(temp, 0.05)

        def _bias_for(idx: int) -> torch.Tensor:
            if plasma_bias is None or idx < 0 or idx >= int(plasma_bias.shape[0]):
                return torch.zeros(2, dtype=torch.float32, device=candidate.device)
            return plasma_bias[idx]

        def _anchor_for(idx: int) -> torch.Tensor:
            if (
                anchor_targets is None
                or anchor_weights is None
                or idx < 0
                or idx >= int(anchor_targets.shape[0])
            ):
                return torch.zeros(2, dtype=torch.float32, device=candidate.device)
            w = float(anchor_weights[idx].item())
            if w <= 1e-8:
                return torch.zeros(2, dtype=torch.float32, device=candidate.device)
            return anchor_targets[idx] - current[idx]

        if move == "flux":
            i = int(rng.choice(movable_indices))
            b = _bias_for(i)
            bx = float(b[0].item())
            by = float(b[1].item())
            bnorm = math.hypot(bx, by)
            if bnorm <= 1e-9:
                move = "shift"
            else:
                sign = -1.0 if rng.random() < 0.5 else 1.0
                perp_x = sign * (-by / bnorm)
                perp_y = sign * (bx / bnorm)
                flux_scale = float(self.cfg["sa"].get("flux_step_scale", 0.018))
                parallel_scale = float(self.cfg["sa"].get("flux_parallel_scale", 0.003))
                candidate[i, 0] += perp_x * span * flux_scale * max(temp, 0.08)
                candidate[i, 1] += perp_y * span * flux_scale * max(temp, 0.08)
                candidate[i, 0] += float(plasma_bias_scale) * bx * span * parallel_scale
                candidate[i, 1] += float(plasma_bias_scale) * by * span * parallel_scale
                a = _anchor_for(i)
                candidate[i, 0] += float(anchor_move_scale) * 0.004 * float(a[0].item())
                candidate[i, 1] += float(anchor_move_scale) * 0.004 * float(a[1].item())
                return [i]

        if move == "shift":
            i = int(rng.choice(movable_indices))
            candidate[i, 0] += float(rng.gauss(0.0, sigma))
            candidate[i, 1] += float(rng.gauss(0.0, sigma))
            b = _bias_for(i)
            a = _anchor_for(i)
            candidate[i, 0] += float(plasma_bias_scale) * float(temp) * float(b[0].item()) * span * 0.02
            candidate[i, 1] += float(plasma_bias_scale) * float(temp) * float(b[1].item()) * span * 0.02
            candidate[i, 0] += float(anchor_move_scale) * 0.010 * float(a[0].item())
            candidate[i, 1] += float(anchor_move_scale) * 0.010 * float(a[1].item())
            return [i]

        if move == "swap":
            if len(movable_indices) < 2:
                return []
            i = int(rng.choice(movable_indices))
            j = int(rng.choice(movable_indices))
            if i == j:
                return []
            tmp = candidate[i].clone()
            candidate[i] = candidate[j]
            candidate[j] = tmp
            return [i, j]

        if move == "attract":
            i = int(rng.choice(movable_indices))
            neigh = adjacency[i]
            if not neigh:
                return []
            j = int(rng.choice(neigh))
            alpha = rng.uniform(0.08, 0.30)
            candidate[i, 0] = candidate[i, 0] + float(alpha) * (candidate[j, 0] - candidate[i, 0])
            candidate[i, 1] = candidate[i, 1] + float(alpha) * (candidate[j, 1] - candidate[i, 1])
            b = _bias_for(i)
            a = _anchor_for(i)
            candidate[i, 0] += float(plasma_bias_scale) * float(b[0].item()) * span * 0.012
            candidate[i, 1] += float(plasma_bias_scale) * float(b[1].item()) * span * 0.012
            candidate[i, 0] += float(anchor_move_scale) * 0.006 * float(a[0].item())
            candidate[i, 1] += float(anchor_move_scale) * 0.006 * float(a[1].item())
            return [i]

        center = int(rng.choice(movable_indices))
        radius = float(self.cfg["sa"].get("cluster_radius_frac", 0.08)) * span
        dx = float(rng.gauss(0.0, sigma * 0.6))
        dy = float(rng.gauss(0.0, sigma * 0.6))

        moved: List[int] = []
        cx = float(current[center, 0].item())
        cy = float(current[center, 1].item())
        for i in movable_indices:
            px = float(current[i, 0].item())
            py = float(current[i, 1].item())
            if (px - cx) * (px - cx) + (py - cy) * (py - cy) <= radius * radius:
                candidate[i, 0] += dx
                candidate[i, 1] += dy
                b = _bias_for(i)
                a = _anchor_for(i)
                candidate[i, 0] += float(plasma_bias_scale) * float(temp) * float(b[0].item()) * span * 0.008
                candidate[i, 1] += float(plasma_bias_scale) * float(temp) * float(b[1].item()) * span * 0.008
                candidate[i, 0] += float(anchor_move_scale) * 0.004 * float(a[0].item())
                candidate[i, 1] += float(anchor_move_scale) * 0.004 * float(a[1].item())
                moved.append(int(i))

        return moved

    def _approximate_total_cost(
        self,
        hard_pos: torch.Tensor,
        hard_sizes: torch.Tensor,
        canvas_w: float,
        canvas_h: float,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_offsets: Optional[torch.Tensor] = None,
        anchor_targets: Optional[torch.Tensor] = None,
        anchor_weights: Optional[torch.Tensor] = None,
    ) -> float:
        sa_cfg = self.cfg["sa"]

        wl = approximate_wirelength_cost(hard_pos, edge_index, edge_weight, edge_offsets=edge_offsets)
        overlap = float(surrogate_overlap(hard_pos, hard_sizes, gap=1e-4).item())
        density = _coarse_density_penalty(hard_pos, hard_sizes, canvas_w, canvas_h, grid_size=12)
        cong = _coarse_congestion_penalty(
            hard_pos,
            edge_index,
            edge_weight,
            canvas_w,
            canvas_h,
            edge_offsets=edge_offsets,
            grid_size=14,
        )
        anchor_term = 0.0
        if anchor_targets is not None and anchor_weights is not None and int(anchor_targets.shape[0]) == int(hard_pos.shape[0]):
            w = torch.clamp(anchor_weights.float(), min=0.0)
            if float(w.sum().item()) > 1e-8:
                d = torch.linalg.norm(anchor_targets.float() - hard_pos, dim=1)
                anchor_term = float((w * d).sum().item() / (w.sum().item() + 1e-9))

        total = (
            wl
            + float(sa_cfg.get("overlap_penalty", 8.0)) * overlap
            + float(sa_cfg.get("density_penalty", 0.2)) * density
            + float(sa_cfg.get("congestion_penalty", 0.1)) * cong
            + float(sa_cfg.get("anchor_penalty", 0.05)) * anchor_term
        )
        return float(total)

    def _optimize_soft_macros(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        num_steps: Sequence[int],
        use_current_loc: bool,
        strategy: Optional[Dict[str, object]] = None,
    ) -> torch.Tensor:
        if plc is None or compute_proxy_cost is None:
            return placement
        if benchmark.num_soft_macros <= 0:
            return placement

        updated = placement.clone()
        try:
            compute_proxy_cost(updated, benchmark, plc)

            strategy = dict(strategy or {})
            canvas_size = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
            steps = [int(s) for s in strategy.get("num_steps", num_steps)]
            if not steps:
                return updated

            attract = strategy.get("attract_factor")
            if attract is None:
                attract = [100.0] + [1.0e-3] * max(0, len(steps) - 2) + [1.0e-5]
            else:
                attract = [float(v) for v in attract]

            repel = strategy.get("repel_factor")
            if repel is None:
                repel = [0.0] + [1.0e6] * max(0, len(steps) - 2) + [1.0e7]
            else:
                repel = [float(v) for v in repel]

            if len(attract) != len(steps) or len(repel) != len(steps):
                return placement

            max_move_distance = strategy.get("max_move_distance")
            if max_move_distance is None:
                move_scale = float(strategy.get("max_move_distance_scale", 100.0))
                max_move_distance = [canvas_size / max(move_scale, 1e-6)] * len(steps)
            else:
                max_move_distance = [float(v) for v in max_move_distance]
                if len(max_move_distance) != len(steps):
                    return placement

            plc.optimize_stdcells(
                use_current_loc=bool(strategy.get("use_current_loc", use_current_loc)),
                move_stdcells=bool(strategy.get("move_stdcells", True)),
                move_macros=bool(strategy.get("move_macros", False)),
                log_scale_conns=bool(strategy.get("log_scale_conns", False)),
                use_sizes=bool(strategy.get("use_sizes", False)),
                io_factor=float(strategy.get("io_factor", 1.0)),
                num_steps=steps,
                max_move_distance=max_move_distance,
                attract_factor=attract,
                repel_factor=repel,
            )

            num_hard = int(benchmark.num_hard_macros)
            for i, module_idx in enumerate(benchmark.soft_macro_indices):
                x, y = plc.modules_w_pins[module_idx].get_pos()
                updated[num_hard + i, 0] = float(x)
                updated[num_hard + i, 1] = float(y)

        except Exception:
            return placement

        return updated

    def _prepare_soft_candidate_for_exact(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        legal_cfg: Dict[str, object],
        legal_anchor_mode: str,
        legal_anchor_strength: float,
        legal_restore_iters: int,
    ) -> torch.Tensor:
        candidate = placement.clone()
        num_hard = int(benchmark.num_hard_macros)
        candidate = project_to_canvas(
            candidate,
            benchmark.macro_sizes,
            benchmark.canvas_width,
            benchmark.canvas_height,
            fixed_mask=benchmark.macro_fixed,
            fixed_positions=benchmark.macro_positions,
        )
        candidate = legalize_hard_macros(
            candidate,
            benchmark.macro_sizes,
            benchmark.macro_fixed,
            benchmark.canvas_width,
            benchmark.canvas_height,
            num_hard,
            gap=float(legal_cfg.get("gap", 1e-4)),
            max_iters=min(50, int(legal_cfg.get("max_iters", 80))),
            fallback_iters=min(45, int(legal_cfg.get("fallback_iters", 60))),
            max_pairs_per_iter=6000,
            anchor_positions=self._resolve_legal_anchor_positions(benchmark, candidate, legal_anchor_mode),
            anchor_strength=legal_anchor_strength,
            restore_iters=legal_restore_iters,
        )
        candidate = sanitize_canvas_bounds(
            candidate,
            benchmark.macro_sizes,
            benchmark.canvas_width,
            benchmark.canvas_height,
            fixed_mask=benchmark.macro_fixed,
            fixed_positions=benchmark.macro_positions,
            safety_eps=1e-6,
        )
        return candidate

    def _consume_exact_eval(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        force_final: bool = False,
    ) -> Optional[float]:
        if self._exact_max_calls <= 0:
            return None

        if not force_final:
            if not self._exact_allow_midrun:
                return None
            # Reserve one call for final end-of-pipeline selection.
            if self._exact_used_calls >= max(0, self._exact_max_calls - 1):
                return None
        else:
            if self._exact_used_calls >= self._exact_max_calls:
                return None

        score = self._evaluate_exact_proxy(placement, benchmark, plc)
        if score is not None:
            self._exact_used_calls += 1
        return score

    @staticmethod
    def _evaluate_exact_proxy(placement: torch.Tensor, benchmark: Benchmark, plc) -> Optional[float]:
        if plc is None or compute_proxy_cost is None:
            return None

        probe = sanitize_canvas_bounds(
            placement,
            benchmark.macro_sizes,
            benchmark.canvas_width,
            benchmark.canvas_height,
            fixed_mask=benchmark.macro_fixed,
            fixed_positions=benchmark.macro_positions,
            safety_eps=1e-6,
        )

        if validate_placement is not None:
            is_valid, _ = validate_placement(probe, benchmark, check_overlaps=False)
            if not is_valid:
                return None

        try:
            costs = compute_proxy_cost(probe, benchmark, plc)
        except Exception:
            return None

        if int(costs.get("overlap_count", 1)) > 0:
            return None
        return float(costs["proxy_cost"])


if __name__ == "__main__":
    pass



















































