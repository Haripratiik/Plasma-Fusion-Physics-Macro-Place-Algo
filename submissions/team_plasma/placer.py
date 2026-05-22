"""
TeamPlasmaPlacer (GS line) — macro placement via the Grad-Shafranov equation.

This is the top-level orchestrator. It:

  1. Builds the cylindrical-coordinate geometry from the benchmark canvas.
  2. Extracts edges between hard macros from the netlist (clique
     decomposition with normalized weights, capped per net).
  3. Initializes hard and soft macro positions from the benchmark
     (warm-start) and computes per-macro currents I_i from connectivity.
  4. Runs the Picard loop:
       - build RUDY q and density rho from current positions
       - fit p(psi) and F(psi) from binned flux-surface averages
       - apply Mercier-aware tail shaping
       - assemble GS RHS  -mu0 R^2 p'(psi) - F F'(psi) - mu0 R J_phi
       - solve Delta-star psi = RHS by damped Picard + RB-GS
       - compute forces on hard macros from -(I/A) grad psi / R
       - drift soft macros (adiabatic-electron step)
       - project to canvas
  5. Runs final pairwise legalization to guarantee zero overlaps.

This is the implementation of `team-plasma-gs/docs/IMPLEMENTATION.md`.
Math derivation in `team-plasma-gs/docs/DERIVATION.md`.

Usage (from competition_ref repo with this submissions/team_plasma copied in):

    uv run evaluate submissions/team_plasma/placer.py -b <benchmark>
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# Inject this directory onto sys.path so sibling modules can be imported
# without package context. The competition's evaluate harness loads placer.py
# via spec_from_file_location, which means we are not running as a package.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from macro_place.benchmark import Benchmark

try:
    from macro_place.objective import compute_proxy_cost  # type: ignore
except Exception:  # pragma: no cover
    compute_proxy_cost = None  # type: ignore

try:
    from macro_place.loader import load_benchmark_from_dir  # type: ignore
except Exception:  # pragma: no cover
    load_benchmark_from_dir = None  # type: ignore

from config import FullConfig, load_config, resolve_device
from geometry import CylindricalGeometry, bilinear_sample, make_geometry, sample_vector
from gs_solver import (
    ConvergenceTrace,
    GSSolverConfig,
    delta_star_apply,
    gradient_of_psi,
    solve_grad_shafranov,
    solve_grad_shafranov_newton,
    taylor_beltrami_mode,
)
from profiles import (
    ProfileConfig,
    Profiles,
    build_density_field,
    build_rudy_field,
    fit_profiles,
)
from coils import (
    CoilConfig,
    MacroCoils,
    coil_rhs_contribution,
    deposit_coil_currents,
    make_macro_coils,
    macro_force_from_psi,
)
from stability import StabilityConfig, apply_tail_shaping, mercier_diagnostic, suydam_rhs_boost
from smooth_proxy import SmoothProxyConfig, build_smooth_hotspot_field, sample_smooth_hotspot_guidance
from sheath_legalize import SheathLegalizeConfig, bohm_sheath_legalize
from eigenmode_refine import (
    EigenmodeRefineConfig,
    build_bfkk_modes,
    select_eigenmode_active_macros,
)
from plasma_init import spectral_plasma_startup_init
from pic_placer import PICConfig, pic_relax_soft_macros
from two_fluid import TwoFluidConfig, relax_soft_density_field, soft_macro_drift_step
from legalization import (
    count_hard_overlaps,
    overlap_pairs,
    project_to_canvas,
    strict_legalize,
)


def set_seed(seed: int) -> None:
    """Make the run reproducible."""
    torch.manual_seed(int(seed))
    try:
        torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def extract_hard_edges_from_plc(
    benchmark: Benchmark,
    plc,
    max_edges: int = 20000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build pairwise hard-macro edges from the PlacementCost netlist.

    The harness leaves `benchmark.net_nodes` empty (net data lives in the
    `plc` object), so we walk `plc.nets` here. Pin names follow the
    convention `<macro_name>/<pin_name>`; the first segment maps to a
    macro via `hard_macro_indices`.

    For each net with `k >= 2` hard macros, we form `k*(k-1)/2` clique
    edges, each weighted by `1/(k-1)` (so the total per-net contribution
    sums to `k/2`, preserving the connectivity strength roughly linearly
    in net size). Edges are sorted by weight and capped at `max_edges`.

    Returns (edge_index [E, 2] long, edge_weight [E] float).
    """
    if plc is None:
        return torch.zeros(0, 2, dtype=torch.long), torch.zeros(0, dtype=torch.float32)

    num_hard = int(benchmark.num_hard_macros)
    if num_hard <= 1:
        return torch.zeros(0, 2, dtype=torch.long), torch.zeros(0, dtype=torch.float32)

    # macro module-name -> hard-macro tensor index
    name_to_hidx = {}
    for hard_i, module_idx in enumerate(benchmark.hard_macro_indices):
        if 0 <= module_idx < len(plc.modules_w_pins):
            name = plc.modules_w_pins[module_idx].get_name()
            name_to_hidx[name] = hard_i

    edge_weight_dict = {}
    nets = getattr(plc, "nets", {})
    for driver_pin_name, sink_pin_names in nets.items():
        macros = set()
        if driver_pin_name in plc.mod_name_to_indices:
            parent = driver_pin_name.split("/")[0]
            if parent in name_to_hidx:
                macros.add(name_to_hidx[parent])
        for pin_name in sink_pin_names:
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
                key = (a, b) if a < b else (b, a)
                edge_weight_dict[key] = edge_weight_dict.get(key, 0.0) + w

    if not edge_weight_dict:
        return torch.zeros(0, 2, dtype=torch.long), torch.zeros(0, dtype=torch.float32)

    items = sorted(edge_weight_dict.items(), key=lambda kv: -float(kv[1]))
    if max_edges > 0 and len(items) > max_edges:
        items = items[:max_edges]

    edge_index = torch.tensor([list(k) for k, _ in items], dtype=torch.long)
    edge_weight = torch.tensor([v for _, v in items], dtype=torch.float32)
    return edge_index, edge_weight


def extract_hard_edges(benchmark: Benchmark, plc=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper. Prefers plc; falls back to `bench.net_nodes`.

    On the production path `plc` is always available via
    `TeamPlasmaPlacer._plc_for_eval`. The `bench.net_nodes` path is here
    only to keep unit tests of the placer module self-contained.
    """
    if plc is not None:
        return extract_hard_edges_from_plc(benchmark, plc)

    # Fallback: in-memory net_nodes (empty for IBM benchmarks, present in tests).
    num_hard = int(benchmark.num_hard_macros)
    if num_hard <= 1 or len(benchmark.net_nodes) == 0:
        return torch.zeros(0, 2, dtype=torch.long), torch.zeros(0, dtype=torch.float32)

    src_list: List[int] = []
    dst_list: List[int] = []
    weight_list: List[float] = []
    for net_idx, nodes in enumerate(benchmark.net_nodes):
        nodes_long = nodes.to(dtype=torch.long)
        hard_nodes = nodes_long[nodes_long < num_hard].tolist()
        if len(hard_nodes) < 2:
            continue
        w_net = float(benchmark.net_weights[net_idx].item()) if benchmark.net_weights.numel() > net_idx else 1.0
        w = w_net / max(1, len(hard_nodes) - 1)
        root = hard_nodes[0]
        for leaf in hard_nodes[1:]:
            src_list.append(root)
            dst_list.append(leaf)
            weight_list.append(w)

    if not src_list:
        return torch.zeros(0, 2, dtype=torch.long), torch.zeros(0, dtype=torch.float32)
    edge_index = torch.tensor(list(zip(src_list, dst_list)), dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float32)
    return edge_index, edge_weight


def extract_all_macro_edges_from_plc(
    benchmark: Benchmark,
    plc,
    max_edges: int = 50000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build pairwise edges over hard+soft macro tensor indices.

    This is used only by the pure-plasma R1 startup path. Unlike TILOS, it
    does not move cells by the official placer; it exposes the netlist as a
    current graph so soft cells can initialize as adiabatic electrons around
    connected current-bearing flux surfaces.
    """
    if plc is None:
        return torch.zeros(0, 2, dtype=torch.long), torch.zeros(0, dtype=torch.float32)

    macro_module_indices = list(benchmark.hard_macro_indices) + list(benchmark.soft_macro_indices)
    if len(macro_module_indices) <= 1:
        return torch.zeros(0, 2, dtype=torch.long), torch.zeros(0, dtype=torch.float32)

    name_to_mid = {}
    for macro_i, module_idx in enumerate(macro_module_indices):
        if 0 <= module_idx < len(plc.modules_w_pins):
            name_to_mid[plc.modules_w_pins[module_idx].get_name()] = macro_i

    edge_weight_dict = {}
    nets = getattr(plc, "nets", {})
    for driver_pin_name, sink_pin_names in nets.items():
        macros = set()
        if driver_pin_name in plc.mod_name_to_indices:
            parent = driver_pin_name.split("/")[0]
            if parent in name_to_mid:
                macros.add(name_to_mid[parent])
        for pin_name in sink_pin_names:
            parent = pin_name.split("/")[0]
            if parent in name_to_mid:
                macros.add(name_to_mid[parent])
        if len(macros) < 2:
            continue
        macro_list = sorted(macros)
        w = 1.0 / max(1, len(macro_list) - 1)
        for i in range(len(macro_list)):
            for j in range(i + 1, len(macro_list)):
                a, b = macro_list[i], macro_list[j]
                key = (a, b) if a < b else (b, a)
                edge_weight_dict[key] = edge_weight_dict.get(key, 0.0) + w

    if not edge_weight_dict:
        return torch.zeros(0, 2, dtype=torch.long), torch.zeros(0, dtype=torch.float32)
    items = sorted(edge_weight_dict.items(), key=lambda kv: -float(kv[1]))
    if max_edges > 0 and len(items) > max_edges:
        items = items[:max_edges]
    return (
        torch.tensor([list(k) for k, _ in items], dtype=torch.long),
        torch.tensor([v for _, v in items], dtype=torch.float32),
    )


def _compute_grid_shape(
    canvas_width: float, canvas_height: float, grid_size: int
) -> Tuple[int, int]:
    """Pick (grid_rows, grid_cols) preserving canvas aspect ratio."""
    aspect = canvas_height / max(canvas_width, 1e-9)
    if aspect >= 1.0:
        grid_cols = int(max(8, grid_size))
        grid_rows = int(max(8, round(grid_size * aspect)))
    else:
        grid_rows = int(max(8, grid_size))
        grid_cols = int(max(8, round(grid_size / max(aspect, 1e-9))))
    return grid_rows, grid_cols


def _warm_start_psi(geom: CylindricalGeometry) -> torch.Tensor:
    """Initialize psi to zero. The equilibrium is built from the actual
    placement state on the first Picard step; an artificial bump biases
    macros toward an arbitrary canvas center.
    """
    return torch.zeros(geom.grid_rows, geom.grid_cols, device=geom.device, dtype=torch.float32)


class TeamPlasmaPlacer:
    """Top-level GS-based macro placer.

    Construct with optional `config_path`; call `place(benchmark)` to get
    a [num_macros, 2] placement tensor with zero hard-macro overlaps.

    The class is stateless across calls — every `place` builds geometry,
    coils, profiles fresh from the benchmark.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.cfg: FullConfig = load_config(config_path)
        if seed is not None:
            self.cfg.seed = int(seed)
        self._plc_cache: dict = {}
        self._purity_trace: dict[str, int] = {}
        self._exact_eval_cache: dict[tuple[str, str], tuple[torch.Tensor, float]] = {}

    def _reset_purity_trace(self) -> None:
        self._purity_trace = {}
        self._exact_eval_cache = {}

    def _trace_purity_event(self, name: str) -> None:
        self._purity_trace[name] = int(self._purity_trace.get(name, 0)) + 1

    def purity_trace(self) -> dict[str, int]:
        """Return stage-provenance counters from the last placement run."""
        return dict(self._purity_trace)

    def _placement_cache_key(self, benchmark: Benchmark, placement: torch.Tensor) -> tuple[str, str]:
        rounded = torch.round(placement.detach().cpu().float() * 1000.0).to(torch.int32).contiguous()
        digest = hashlib.blake2b(rounded.numpy().tobytes(), digest_size=16).hexdigest()
        return str(getattr(benchmark, "name", "")), digest

    def _score_direct_legal_candidate(
        self,
        candidate: torch.Tensor,
        benchmark: Benchmark,
        plc,
        num_hard: int,
        trace_name: str,
    ) -> Tuple[Optional[torch.Tensor], Optional[float]]:
        """Score an already-legal pure-plasma candidate without relegalizing it.

        This does not introduce a TILOS placement stage: it only invokes the
        official proxy as the acceptance gate after a plasma-derived move.
        Overlapping candidates are rejected by returning `(None, None)`.
        """
        if plc is None or compute_proxy_cost is None:
            return None, None
        base = project_to_canvas(
            candidate.clone().float(),
            benchmark.macro_sizes.float(),
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
            fixed_mask=benchmark.macro_fixed,
            fixed_positions=benchmark.macro_positions.float(),
        )
        if count_hard_overlaps(base, benchmark.macro_sizes.float(), int(num_hard), gap=0.0) != 0:
            return None, None
        cache_key = self._placement_cache_key(benchmark, base)
        cached = self._exact_eval_cache.get(cache_key)
        if cached is not None:
            self._trace_purity_event("pure_exact_eval_cache_hits")
            cached_candidate, cached_cost = cached
            return cached_candidate.clone(), float(cached_cost)
        try:
            costs = compute_proxy_cost(base, benchmark, plc)
        except Exception:
            return None, None
        if int(costs.get("overlap_count", 1)) != 0:
            return None, None
        self._trace_purity_event(trace_name)
        cost = float(costs["proxy_cost"])
        self._exact_eval_cache[cache_key] = (base.clone(), cost)
        return base, cost

    def _assert_plasma_purity_contract(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        num_hard: int,
    ) -> None:
        if not (
            bool(getattr(self.cfg, "plasma_purity_mode", False))
            and bool(getattr(self.cfg, "plasma_purity_assert_enabled", False))
        ):
            return
        forbidden = {
            "tilos_plc_seed_calls": int(self._purity_trace.get("tilos_plc_seed_calls", 0)),
            "strict_candidate_variant_calls": int(self._purity_trace.get("strict_candidate_variant_calls", 0)),
            "strict_fallback_calls": int(self._purity_trace.get("strict_fallback_calls", 0)),
            "tilos_soft_optimize_calls": int(self._purity_trace.get("tilos_soft_optimize_calls", 0)),
        }
        used_required = {
            "r1_plasma_init_calls": int(self._purity_trace.get("r1_plasma_init_calls", 0)),
            "r2_legalize_calls": int(self._purity_trace.get("r2_legalize_calls", 0)),
            "soft_plasma_calls": int(self._purity_trace.get("r4_two_fluid_calls", 0))
            + int(self._purity_trace.get("r5_pic_calls", 0)),
        }
        overlaps = count_hard_overlaps(placement, benchmark.macro_sizes.float(), int(num_hard), gap=0.0)
        if any(value != 0 for value in forbidden.values()) or any(value <= 0 for value in used_required.values()) or int(overlaps) != 0:
            raise RuntimeError(
                "plasma purity contract violated: "
                f"forbidden={forbidden} required={used_required} overlaps={int(overlaps)}"
            )

    def _apply_plasma_purity_mode(self) -> None:
        """Force Definition-B replacement stages when purity mode is active.

        This mode makes the pure branch structurally unable to use the three
        TILOS load-bearing stages: PLC initialization, strict legalization, and
        soft-cell optimization. Exact proxy scoring remains as the competition
        metric/gate, but every placement-producing move is plasma-derived.
        """
        if not bool(getattr(self.cfg, "plasma_purity_mode", False)):
            return
        self.cfg.plasma_init_enabled = True
        self.cfg.plasma_init_soft_grid_enabled = True
        self.cfg.plasma_purity_refine_direct_score_enabled = True
        self.cfg.plasma_purity_refine_skip_overlap_relegalize = True
        self.cfg.two_fluid_pde_enabled = True
        self.cfg.two_fluid_replace_tilos_soft = True
        self.cfg.soft_tilos_enabled = False
        self.cfg.sheath_legalize_fallback_strict = False
        self.cfg.r2_continuation_enabled = True
        self.cfg.r2_continuation_replace_strict = True
        self.cfg.exact_hard_refine_direction_set = "flux_only"

    def _flux_cell_evacuation_legalize(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        num_hard: int,
    ) -> torch.Tensor:
        """Last-resort pure-plasma evacuation of residual overlap islands.

        Think of this as an ELM crash that ejects the smallest still-overlapping
        island to the nearest empty flux cell. It is intentionally deterministic
        and only used after R2 fails, so purity mode never falls back to strict
        geometric legalization.
        """
        out = project_to_canvas(
            placement.clone().float(),
            benchmark.macro_sizes.float(),
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
            fixed_mask=benchmark.macro_fixed,
            fixed_positions=benchmark.macro_positions.float(),
        )
        n = int(num_hard)
        if n <= 1:
            return out
        sizes = benchmark.macro_sizes[:n].float()
        fixed = benchmark.macro_fixed[:n].bool()
        half_w = 0.5 * sizes[:, 0]
        half_h = 0.5 * sizes[:, 1]
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        cols = max(96, int(getattr(self.cfg, "r2_flux_lattice_cols", 16)) * 2)
        rows = max(32, int(math.ceil(cols * canvas_h / max(canvas_w, 1e-9))))

        def legal_at(idx: int, x: float, y: float) -> bool:
            pos = out[:n]
            dx = torch.abs(pos[:, 0] - float(x))
            dy = torch.abs(pos[:, 1] - float(y))
            sep_x = half_w + half_w[idx]
            sep_y = half_h + half_h[idx]
            mask = (dx < sep_x) & (dy < sep_y)
            mask[idx] = False
            return not bool(mask.any())

        max_rounds = max(1, 2 * n)
        for _ in range(max_rounds):
            pairs = overlap_pairs(out, benchmark.macro_sizes.float(), n, gap=0.0)
            if pairs.numel() == 0:
                break
            offenders = sorted(
                set(int(x) for x in pairs.reshape(-1).tolist()),
                key=lambda idx: float((sizes[idx, 0] * sizes[idx, 1]).item()),
            )
            moved = False
            for idx in offenders:
                if bool(fixed[idx]):
                    continue
                lo_x = float(half_w[idx].item())
                hi_x = canvas_w - float(half_w[idx].item())
                lo_y = float(half_h[idx].item())
                hi_y = canvas_h - float(half_h[idx].item())
                if hi_x < lo_x or hi_y < lo_y:
                    continue
                cx = float(out[idx, 0].item())
                cy = float(out[idx, 1].item())
                xs = torch.linspace(lo_x, hi_x, cols, dtype=out.dtype)
                ys = torch.linspace(lo_y, hi_y, rows, dtype=out.dtype)
                lattice: list[tuple[float, float, float]] = []
                for yy in ys.tolist():
                    for xx in xs.tolist():
                        dist = (cx - float(xx)) ** 2 + (cy - float(yy)) ** 2
                        lattice.append((dist, float(xx), float(yy)))
                lattice.sort(key=lambda item: item[0])
                for _dist, x, y in lattice:
                    if legal_at(idx, x, y):
                        out[idx, 0] = x
                        out[idx, 1] = y
                        moved = True
                        break
                if moved:
                    break
            out[:n] = project_to_canvas(
                out[:n],
                sizes,
                canvas_w,
                canvas_h,
                fixed_mask=fixed,
                fixed_positions=benchmark.macro_positions[:n].float(),
            )
            if not moved:
                break
        return out

    def _flux_rope_expansion_legalize(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        num_hard: int,
    ) -> torch.Tensor:
        """Resolve residual overlap chains by simultaneous Bohm expansion."""
        out = project_to_canvas(
            placement.clone().float(),
            benchmark.macro_sizes.float(),
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
            fixed_mask=benchmark.macro_fixed,
            fixed_positions=benchmark.macro_positions.float(),
        )
        n = int(num_hard)
        sizes = benchmark.macro_sizes[:n].float()
        fixed = benchmark.macro_fixed[:n].bool()
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        span = max(canvas_w, canvas_h, 1.0)
        eps = max(1e-6, 1e-6 * span)
        for _ in range(max(1, 8 * n)):
            pairs = overlap_pairs(out, benchmark.macro_sizes.float(), n, gap=0.0)
            if pairs.numel() == 0:
                break
            disp = torch.zeros_like(out[:n])
            for i_raw, j_raw in pairs.tolist():
                i = int(i_raw)
                j = int(j_raw)
                if bool(fixed[i]) and bool(fixed[j]):
                    continue
                xi = float(out[i, 0].item())
                yi = float(out[i, 1].item())
                xj = float(out[j, 0].item())
                yj = float(out[j, 1].item())
                wi = float(sizes[i, 0].item())
                hi = float(sizes[i, 1].item())
                wj = float(sizes[j, 0].item())
                hj = float(sizes[j, 1].item())
                overlap_x = 0.5 * (wi + wj) - abs(xj - xi)
                overlap_y = 0.5 * (hi + hj) - abs(yj - yi)
                if overlap_x <= 0.0 or overlap_y <= 0.0:
                    continue
                if overlap_x <= overlap_y:
                    sign = 1.0 if xj >= xi else -1.0
                    delta = overlap_x + eps
                    vec = torch.tensor([sign * delta, 0.0], dtype=out.dtype, device=out.device)
                else:
                    sign = 1.0 if yj >= yi else -1.0
                    delta = overlap_y + eps
                    vec = torch.tensor([0.0, sign * delta], dtype=out.dtype, device=out.device)
                if not bool(fixed[i]) and not bool(fixed[j]):
                    disp[i] -= 0.5 * vec
                    disp[j] += 0.5 * vec
                elif bool(fixed[i]) and not bool(fixed[j]):
                    disp[j] += vec
                elif not bool(fixed[i]) and bool(fixed[j]):
                    disp[i] -= vec
            if not bool((torch.linalg.norm(disp, dim=1) > 0.0).any()):
                break
            out[:n] = out[:n] + disp
            out[:n] = project_to_canvas(
                out[:n],
                sizes,
                canvas_w,
                canvas_h,
                fixed_mask=fixed,
                fixed_positions=benchmark.macro_positions[:n].float(),
            )
        return out

    def _plc_for_eval(self, benchmark: Benchmark):
        """Look up the PlacementCost object for this benchmark.

        We don't have access to the plc that the harness loaded alongside
        the benchmark, so we re-load it. Cached per benchmark name.
        """
        if load_benchmark_from_dir is None:
            return None
        name = benchmark.name
        if name in self._plc_cache:
            return self._plc_cache[name]
        for testcase_root in (
            "external/MacroPlacement/Testcases/ICCAD04",
            "competition_ref/external/MacroPlacement/Testcases/ICCAD04",
        ):
            try:
                _, plc = load_benchmark_from_dir(f"{testcase_root}/{name}")
                self._plc_cache[name] = plc
                return plc
            except Exception:
                continue
        return None

    def _optimize_soft_macros(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        steps_override: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Run a short TILOS soft-macro relaxation with hard macros fixed.

        The GS equilibrium controls hard macro locations. This final pass only
        lets the official PlacementCost engine relax soft macros / stdcells,
        which are a major part of the proxy density and congestion terms.
        """
        self._trace_purity_event("tilos_soft_optimize_calls")
        if plc is None or compute_proxy_cost is None:
            return placement
        if int(benchmark.num_soft_macros) <= 0:
            return placement

        raw_steps = (
            steps_override
            if steps_override is not None
            else getattr(self.cfg, "soft_tilos_num_steps", [2, 2])
        )
        steps = [int(s) for s in raw_steps if int(s) > 0]
        if not steps:
            return placement

        updated = placement.clone()
        try:
            compute_proxy_cost(updated, benchmark, plc)

            canvas_size = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
            if len(steps) == 2:
                attract = [1.0e-3, 1.0e-5]
                repel = [1.0e6, 1.0e7]
            else:
                attract = [100.0] + [1.0e-3] * max(0, len(steps) - 2) + [1.0e-5]
                repel = [0.0] + [1.0e6] * max(0, len(steps) - 2) + [1.0e7]
            max_move_distance = [canvas_size / 100.0] * len(steps)

            # optimize_stdcells is noisy; keep official evaluate output readable.
            with contextlib.redirect_stdout(io.StringIO()):
                plc.optimize_stdcells(
                    use_current_loc=bool(self.cfg.soft_tilos_use_current_loc),
                    move_stdcells=True,
                    move_macros=False,
                    log_scale_conns=False,
                    use_sizes=False,
                    io_factor=1.0,
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

    def _optimize_soft_two_fluid_pde(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        psi: torch.Tensor,
        geom: CylindricalGeometry,
        cfg: TwoFluidConfig,
        q_field: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run R4 Braginskii drift-diffusion relaxation on soft macros."""
        self._trace_purity_event("r4_two_fluid_calls")
        if not bool(getattr(cfg, "pde_enabled", False)):
            return placement
        if int(benchmark.num_soft_macros) <= 0:
            return placement
        num_hard = int(benchmark.num_hard_macros)
        updated = placement.clone().float()
        try:
            device = geom.device
            soft_fixed = benchmark.macro_fixed[num_hard:].to(device) if benchmark.macro_fixed.numel() > num_hard else None
            hard = updated[:num_hard].to(device)
            hard_sizes = benchmark.macro_sizes[:num_hard].float().to(device)
            soft_sizes = benchmark.macro_sizes[num_hard:].float().to(device)
            soft_new = updated[num_hard:].to(device)

            n_passes = 1
            if bool(getattr(self.cfg, "r4_annealing_enabled", False)):
                n_passes = max(1, int(getattr(self.cfg, "r4_n_passes", 3)))
            anneal = max(0.05, float(getattr(self.cfg, "r4_anneal_factor", 0.5)))

            for pass_idx in range(n_passes):
                # Tokamak-startup analogy: high transport during current ramp,
                # then progressively lower transport as confinement forms.
                if n_passes > 1:
                    factor = anneal ** float(pass_idx)
                    pass_cfg = TwoFluidConfig(**cfg.__dict__)
                    pass_cfg.D_parallel = float(cfg.D_parallel) * factor
                    pass_cfg.D_perp = float(cfg.D_perp) * factor
                    pass_cfg.dt = float(cfg.dt) * max(factor, 0.25)
                else:
                    pass_cfg = cfg
                soft_new = relax_soft_density_field(
                    hard,
                    hard_sizes,
                    soft_new,
                    soft_sizes,
                    psi.to(device),
                    geom,
                    pass_cfg,
                    soft_fixed=soft_fixed,
                    q_field=q_field.to(device) if q_field is not None else None,
                )
            updated[num_hard:] = soft_new.detach().cpu()
            updated = project_to_canvas(
                updated,
                benchmark.macro_sizes.float(),
                float(benchmark.canvas_width),
                float(benchmark.canvas_height),
                fixed_mask=benchmark.macro_fixed,
                fixed_positions=benchmark.macro_positions.float(),
            )
        except Exception:
            return placement
        return updated

    def _optimize_soft_pic(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        geom: CylindricalGeometry,
        cfg: PICConfig,
        all_edge_index: torch.Tensor,
        all_edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Run R5 hybrid plasma-PIC relaxation on soft macros."""
        self._trace_purity_event("r5_pic_calls")
        if not bool(getattr(cfg, "enabled", False)):
            return placement
        if int(benchmark.num_soft_macros) <= 0:
            return placement
        num_hard = int(benchmark.num_hard_macros)
        updated = placement.clone().float()
        try:
            device = geom.device
            soft_fixed = benchmark.macro_fixed[num_hard:].to(device) if benchmark.macro_fixed.numel() > num_hard else None
            soft_new = pic_relax_soft_macros(
                updated[:num_hard].to(device),
                benchmark.macro_sizes[:num_hard].float().to(device),
                updated[num_hard:].to(device),
                benchmark.macro_sizes[num_hard:].float().to(device),
                all_edge_index.to(device),
                all_edge_weight.to(device),
                geom,
                cfg,
                soft_fixed=soft_fixed,
            )
            updated[num_hard:] = soft_new.detach().cpu()
            updated = project_to_canvas(
                updated,
                benchmark.macro_sizes.float(),
                float(benchmark.canvas_width),
                float(benchmark.canvas_height),
                fixed_mask=benchmark.macro_fixed,
                fixed_positions=benchmark.macro_positions.float(),
            )
        except Exception:
            return placement
        return updated

    def _pic_cfg_with_overrides(self, cfg: PICConfig, overrides) -> PICConfig:
        """Return a PIC config variant for exact-gated proposal portfolios.

        Probe JSONs may use either PICConfig field names (`b0_strength`) or
        user-facing FullConfig names (`r5_pic_b0_strength`). Unknown keys are
        ignored so old configs and hand-written sweep snippets stay safe.
        """
        variant = PICConfig(**cfg.__dict__)
        if not isinstance(overrides, dict):
            return variant
        alias = {
            "r5_pic_max_iters": "max_iters",
            "r5_pic_dt_frac": "dt_frac",
            "r5_pic_debye_length_frac": "debye_length_frac",
            "r5_pic_sheath_width_frac": "sheath_width_frac",
            "r5_pic_ion_to_electron_mass_ratio": "ion_to_electron_mass_ratio",
            "r5_pic_net_current_scale": "net_current_scale",
            "r5_pic_coulomb_strength": "coulomb_strength",
            "r5_pic_electric_strength": "electric_strength",
            "r5_pic_magnetic_strength": "magnetic_strength",
            "r5_pic_sheath_strength": "sheath_strength",
            "r5_pic_collision_freq": "collision_freq",
            "r5_pic_trust_radius_frac": "trust_radius_frac",
            "r5_pic_field_solve_iters": "field_solve_iters",
            "r5_pic_field_solver": "field_solver",
            "r5_pic_conv_tol": "conv_tol",
            "r5_pic_current_line_samples": "current_line_samples",
            "r5_pic_b0_strength": "b0_strength",
            "r5_pic_b_ripple_strength": "b_ripple_strength",
            "r5_pic_grad_b_drift_strength": "grad_b_drift_strength",
            "r5_pic_magnetic_mirror_strength": "magnetic_mirror_strength",
            "r5_pic_pair_attraction_strength": "pair_attraction_strength",
            "r5_pic_pair_attraction_range_frac": "pair_attraction_range_frac",
            "r5_pic_pair_attraction_softening_frac": "pair_attraction_softening_frac",
            "r5_pic_bootstrap_current_strength": "bootstrap_current_strength",
            "r5_pic_diamagnetic_drift_strength": "diamagnetic_drift_strength",
            "r5_pic_annealed_schedule_enabled": "annealed_schedule_enabled",
        }
        for raw_key, value in overrides.items():
            key = alias.get(str(raw_key), str(raw_key))
            if hasattr(variant, key):
                setattr(variant, key, value)
        return variant

    def _plasma_dimensionless_numbers(
        self,
        benchmark: Benchmark,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """Return benchmark-name-free plasma regime descriptors.

        These are the dimensionless numbers we use instead of `ibmXX`
        routing: packing beta, macro-number density kappa, hard/ion fraction,
        mean current weight, and canvas aspect ratio.
        """
        area = max(float(benchmark.canvas_width) * float(benchmark.canvas_height), 1e-9)
        sizes = benchmark.macro_sizes.float()
        macro_area = float((sizes[:, 0] * sizes[:, 1]).sum().item()) if sizes.numel() else 0.0
        total = max(int(benchmark.num_hard_macros) + int(benchmark.num_soft_macros), 1)
        hard = max(int(benchmark.num_hard_macros), 0)
        aspect = max(float(benchmark.canvas_width), float(benchmark.canvas_height)) / max(
            min(float(benchmark.canvas_width), float(benchmark.canvas_height)),
            1e-9,
        )
        gamma = 1.0
        if edge_weight is not None and edge_weight.numel() > 0:
            gamma = float(edge_weight.float().mean().item())
        return {
            "beta_p": macro_area / area,
            "kappa": float(total) / area,
            "hard_ratio": float(hard) / float(total),
            "gamma": max(gamma, 1e-6),
            "aspect": aspect,
        }

    def _dimensionless_pic_variants(
        self,
        benchmark: Benchmark,
        edge_weight: Optional[torch.Tensor],
    ) -> List[dict]:
        """Build a universal R5 portfolio from continuous plasma numbers.

        The formulas intentionally avoid benchmark names.  Exact proxy gating
        still decides whether any proposal is accepted, so bad regime guesses
        cost runtime rather than score.
        """
        d = self._plasma_dimensionless_numbers(benchmark, edge_weight)
        beta = max(0.0, d["beta_p"])
        kappa = max(0.0, d["kappa"])
        hard = min(max(d["hard_ratio"], 0.0), 1.0)
        gamma = max(d["gamma"], 1e-6)
        aspect = max(1.0, d["aspect"])

        # Smooth crossovers, not hand-tuned benchmark thresholds.
        dense = beta / (1.0 + beta)
        hard_mix = hard / (0.10 + hard)
        open_geom = (aspect - 1.0) / (aspect + 1.0)
        current = gamma / (1.0 + gamma)
        density_norm = kappa / (kappa + 1.0)

        pair = 0.08 + 0.12 * hard_mix * (0.55 + 0.45 * current)
        ripple = 0.10 + 0.08 * (1.0 - dense) * (0.65 + 0.35 * open_geom)
        trust = 0.0045 + 0.0035 * (1.0 - dense)
        iters = int(round(10 + 14 * (0.50 + 0.50 * density_norm)))
        iters = max(8, min(28, iters))
        range_frac = max(0.55, min(0.75, 0.55 + 0.20 * hard_mix))

        return [
            {
                "r5_pic_max_iters": iters,
                "r5_pic_trust_radius_frac": trust,
                "r5_pic_b_ripple_strength": ripple,
                "r5_pic_grad_b_drift_strength": 0.015 + 0.025 * (1.0 - dense),
                "r5_pic_magnetic_mirror_strength": 0.008 + 0.014 * open_geom,
                "r5_pic_pair_attraction_strength": 0.0,
            },
            {
                "r5_pic_max_iters": iters,
                "r5_pic_trust_radius_frac": trust,
                "r5_pic_pair_attraction_strength": pair,
                "r5_pic_pair_attraction_range_frac": range_frac,
                "r5_pic_pair_attraction_softening_frac": 0.012 + 0.006 * dense,
                "r5_pic_b_ripple_strength": 0.0,
                "r5_pic_grad_b_drift_strength": 0.0,
                "r5_pic_magnetic_mirror_strength": 0.0,
            },
            {
                "r5_pic_max_iters": max(8, min(32, iters + 4)),
                "r5_pic_trust_radius_frac": trust,
                "r5_pic_pair_attraction_strength": 0.70 * pair,
                "r5_pic_pair_attraction_range_frac": range_frac,
                "r5_pic_pair_attraction_softening_frac": 0.014 + 0.004 * dense,
                "r5_pic_b_ripple_strength": 0.65 * ripple,
                "r5_pic_grad_b_drift_strength": 0.010 + 0.015 * (1.0 - dense),
                "r5_pic_magnetic_mirror_strength": 0.006 + 0.010 * open_geom,
            },
        ]

    def _evaluate_soft_tilos_candidate(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
    ) -> Tuple[Optional[torch.Tensor], Optional[float]]:
        """Project/legalize/evaluate a soft-TILOS output candidate."""
        if plc is None or compute_proxy_cost is None:
            return None, None

        best_candidate: Optional[torch.Tensor] = None
        best_cost: Optional[float] = None
        try:
            variants = self._legalized_variants(
                placement,
                benchmark,
                int(benchmark.num_hard_macros),
            )
        except Exception:
            return None, None

        for candidate in variants:
            try:
                ev = compute_proxy_cost(candidate, benchmark, plc)
            except Exception:
                continue
            if int(ev.get("overlap_count", 1)) != 0:
                continue
            cost = float(ev["proxy_cost"])
            if best_cost is None or cost < best_cost:
                best_candidate = candidate
                best_cost = cost
        return best_candidate, best_cost

    def _soft_tilos_steps_for_benchmark(self, benchmark: Benchmark) -> Tuple[List[int], List[int]]:
        """Return feature-routed soft relaxation budgets.

        Longer stdcell relaxation helped the extreme dense-soft regime but
        hurt lower-density cases. Route by macro density and soft fraction,
        not benchmark name, so the adiabatic-electron relaxation policy is a
        plasma-regime rule rather than a per-benchmark override.
        """
        probe = [int(s) for s in self.cfg.soft_tilos_probe_num_steps]
        refine = [int(s) for s in self.cfg.soft_tilos_refine_num_steps]
        if not bool(getattr(self.cfg, "soft_tilos_auto_steps_enabled", False)):
            return probe, refine
        if bool(getattr(self.cfg, "soft_tilos_dense_regime_enabled", True)):
            area = max(
                float(benchmark.canvas_width) * float(benchmark.canvas_height),
                1e-9,
            )
            total_macros = int(benchmark.num_hard_macros) + int(benchmark.num_soft_macros)
            macro_density = float(total_macros) / area
            soft_frac = float(benchmark.num_soft_macros) / max(float(total_macros), 1.0)
            if (
                macro_density >= float(getattr(self.cfg, "soft_tilos_dense_macro_density_threshold", 2.0))
                and soft_frac >= float(getattr(self.cfg, "soft_tilos_dense_soft_frac_threshold", 0.75))
            ):
                dense_probe = getattr(self.cfg, "soft_tilos_dense_probe_num_steps", [12, 12])
                dense_refine = getattr(self.cfg, "soft_tilos_dense_refine_num_steps", [12, 12])
                return [int(s) for s in dense_probe], [int(s) for s in dense_refine]
        return probe, refine

    def _legalize_candidate(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        num_hard: int,
    ) -> torch.Tensor:
        """Legalize via optional Bohm-sheath dynamics, then strict fallback.

        R2 is deliberately exact-score guarded: if the continuous plasma
        dynamics produce a legal placement, we use it; otherwise the existing
        robust legalizer remains available unless disabled by config. In
        plasma-purity mode, failed R2 legalizations get a stronger flux-lattice
        R2 rescue instead of falling back to strict geometry.
        """
        candidate = project_to_canvas(
            placement.clone().float(),
            benchmark.macro_sizes.float(),
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
            fixed_mask=benchmark.macro_fixed,
            fixed_positions=benchmark.macro_positions.float(),
        )
        original = candidate.clone()
        use_r2 = bool(getattr(self.cfg, "r2_continuation_enabled", False))
        if bool(getattr(self.cfg, "sheath_legalize_enabled", False)) or use_r2:
            self._trace_purity_event("r2_legalize_calls")
            sheath_cfg = SheathLegalizeConfig(
                enabled=True,
                fallback_strict=bool(getattr(self.cfg, "sheath_legalize_fallback_strict", True))
                and not bool(getattr(self.cfg, "r2_continuation_replace_strict", False)),
                max_iters=int(getattr(self.cfg, "sheath_legalize_max_iters", 160)),
                dt=float(getattr(self.cfg, "r2_langevin_dt", getattr(self.cfg, "sheath_legalize_dt", 0.35))) if use_r2 else float(getattr(self.cfg, "sheath_legalize_dt", 0.35)),
                damping=float(getattr(self.cfg, "sheath_legalize_damping", 0.60)),
                coulomb_strength=float(getattr(self.cfg, "sheath_legalize_coulomb_strength", 1.0)),
                sheath_strength=float(getattr(self.cfg, "sheath_legalize_sheath_strength", 1.0)),
                debye_frac=float(getattr(self.cfg, "sheath_legalize_debye_frac", 0.02)),
                max_step_frac=float(getattr(self.cfg, "sheath_legalize_max_step_frac", 0.010)),
                continuation_enabled=use_r2,
                beta_overlap_start=float(getattr(self.cfg, "r2_beta_overlap_start", 0.10)),
                beta_overlap_max=float(getattr(self.cfg, "r2_beta_overlap_max", 1.0)),
                beta_overlap_ramp_stages=int(getattr(self.cfg, "r2_beta_overlap_ramp_stages", 8)),
                beta_wall=float(getattr(self.cfg, "r2_beta_wall", 1.0)),
                inner_iters=int(getattr(self.cfg, "r2_inner_iters", 40)),
                temp_start=0.0 if bool(getattr(self.cfg, "r2_deterministic_enabled", False)) else float(getattr(self.cfg, "r2_temp_start", 0.0)),
                temp_decay=float(getattr(self.cfg, "r2_temp_decay", 0.70)),
                force_tol=float(getattr(self.cfg, "r2_force_tol", 1e-5)),
                seed=int(getattr(self.cfg, "r2_seed", 7)),
                hardening_enabled=bool(getattr(self.cfg, "r2_hardening_enabled", True)),
                hardening_iters=int(getattr(self.cfg, "r2_hardening_iters", 320)),
                hardening_max_pairs_per_iter=int(getattr(self.cfg, "r2_hardening_max_pairs_per_iter", 12000)),
                relocation_enabled=bool(getattr(self.cfg, "r2_relocation_enabled", True)),
                relocation_iters=int(getattr(self.cfg, "r2_relocation_iters", 0)),
                relocation_rings=int(getattr(self.cfg, "r2_relocation_rings", 6)),
                relocation_density_weight=float(getattr(self.cfg, "r2_relocation_density_weight", 0.0)),
                flux_lattice_relocation_enabled=bool(getattr(self.cfg, "r2_flux_lattice_relocation_enabled", True)),
                flux_lattice_cols=int(getattr(self.cfg, "r2_flux_lattice_cols", 16)),
                local_flux_refine_enabled=bool(getattr(self.cfg, "r2_local_flux_refine_enabled", False)),
                local_flux_refine_steps=int(getattr(self.cfg, "r2_local_flux_refine_steps", 5)),
                local_flux_refine_pair_threshold=int(getattr(self.cfg, "r2_local_flux_refine_pair_threshold", 4)),
                anchor_strength=float(getattr(self.cfg, "r2_anchor_strength", 0.0)),
                anchor_release_frac=float(getattr(self.cfg, "r2_anchor_release_frac", 0.5)),
            )
            candidate = bohm_sheath_legalize(
                candidate,
                benchmark.macro_sizes.float(),
                benchmark.macro_fixed,
                float(benchmark.canvas_width),
                float(benchmark.canvas_height),
                int(num_hard),
                gap=self.cfg.legalize_gap,
                cfg=sheath_cfg,
            )
            candidate = project_to_canvas(
                candidate,
                benchmark.macro_sizes.float(),
                float(benchmark.canvas_width),
                float(benchmark.canvas_height),
                fixed_mask=benchmark.macro_fixed,
                fixed_positions=benchmark.macro_positions.float(),
            )
            if count_hard_overlaps(candidate, benchmark.macro_sizes.float(), int(num_hard), gap=0.0) == 0:
                return candidate
            if bool(getattr(self.cfg, "plasma_purity_mode", False)):
                self._trace_purity_event("r2_purity_rescue_calls")
                rescue_cfg = SheathLegalizeConfig(
                    enabled=True,
                    fallback_strict=False,
                    max_iters=max(int(getattr(self.cfg, "sheath_legalize_max_iters", 160)), 240),
                    dt=float(getattr(self.cfg, "r2_langevin_dt", getattr(self.cfg, "sheath_legalize_dt", 0.35))),
                    damping=float(getattr(self.cfg, "sheath_legalize_damping", 0.60)),
                    coulomb_strength=float(getattr(self.cfg, "sheath_legalize_coulomb_strength", 1.0)),
                    sheath_strength=float(getattr(self.cfg, "sheath_legalize_sheath_strength", 1.0)),
                    debye_frac=float(getattr(self.cfg, "sheath_legalize_debye_frac", 0.02)),
                    max_step_frac=float(getattr(self.cfg, "sheath_legalize_max_step_frac", 0.010)),
                    continuation_enabled=True,
                    beta_overlap_start=float(getattr(self.cfg, "r2_beta_overlap_start", 0.10)),
                    beta_overlap_max=max(float(getattr(self.cfg, "r2_beta_overlap_max", 1.0)), 1.25),
                    beta_overlap_ramp_stages=max(int(getattr(self.cfg, "r2_beta_overlap_ramp_stages", 8)), 10),
                    beta_wall=float(getattr(self.cfg, "r2_beta_wall", 1.0)),
                    inner_iters=max(int(getattr(self.cfg, "r2_inner_iters", 40)), 48),
                    temp_start=0.0,
                    temp_decay=float(getattr(self.cfg, "r2_temp_decay", 0.70)),
                    force_tol=float(getattr(self.cfg, "r2_force_tol", 1e-5)),
                    seed=int(getattr(self.cfg, "r2_seed", 7)),
                    hardening_enabled=True,
                    hardening_iters=max(int(getattr(self.cfg, "r2_hardening_iters", 320)), 1500),
                    hardening_max_pairs_per_iter=int(getattr(self.cfg, "r2_hardening_max_pairs_per_iter", 12000)),
                    relocation_enabled=True,
                    relocation_iters=max(int(getattr(self.cfg, "r2_relocation_iters", 0)), 800),
                    relocation_rings=max(int(getattr(self.cfg, "r2_relocation_rings", 6)), 32),
                    relocation_density_weight=float(getattr(self.cfg, "r2_relocation_density_weight", 0.0)),
                    flux_lattice_relocation_enabled=True,
                    flux_lattice_cols=max(int(getattr(self.cfg, "r2_flux_lattice_cols", 16)), 96),
                    local_flux_refine_enabled=True,
                    local_flux_refine_steps=max(int(getattr(self.cfg, "r2_local_flux_refine_steps", 5)), 8),
                    local_flux_refine_pair_threshold=max(int(getattr(self.cfg, "r2_local_flux_refine_pair_threshold", 4)), 64),
                    anchor_strength=0.0,
                    anchor_release_frac=float(getattr(self.cfg, "r2_anchor_release_frac", 0.5)),
                )
                rescue = bohm_sheath_legalize(
                    candidate,
                    benchmark.macro_sizes.float(),
                    benchmark.macro_fixed,
                    float(benchmark.canvas_width),
                    float(benchmark.canvas_height),
                    int(num_hard),
                    gap=self.cfg.legalize_gap,
                    cfg=rescue_cfg,
                )
                rescue = project_to_canvas(
                    rescue,
                    benchmark.macro_sizes.float(),
                    float(benchmark.canvas_width),
                    float(benchmark.canvas_height),
                    fixed_mask=benchmark.macro_fixed,
                    fixed_positions=benchmark.macro_positions.float(),
                )
                if count_hard_overlaps(rescue, benchmark.macro_sizes.float(), int(num_hard), gap=0.0) == 0:
                    return rescue
                evacuated = self._flux_cell_evacuation_legalize(rescue, benchmark, num_hard)
                self._trace_purity_event("flux_cell_evacuation_calls")
                if count_hard_overlaps(evacuated, benchmark.macro_sizes.float(), int(num_hard), gap=0.0) == 0:
                    return evacuated
                expanded = self._flux_rope_expansion_legalize(evacuated, benchmark, num_hard)
                self._trace_purity_event("flux_rope_expansion_calls")
                if count_hard_overlaps(expanded, benchmark.macro_sizes.float(), int(num_hard), gap=0.0) == 0:
                    return expanded
                return expanded
            if not bool(sheath_cfg.fallback_strict):
                return candidate
            candidate = original

        self._trace_purity_event("strict_fallback_calls")
        candidate = strict_legalize(
            candidate,
            benchmark.macro_sizes.float(),
            benchmark.macro_fixed,
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
            int(num_hard),
            gap=self.cfg.legalize_gap,
        )
        return project_to_canvas(
            candidate,
            benchmark.macro_sizes.float(),
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
            fixed_mask=benchmark.macro_fixed,
            fixed_positions=benchmark.macro_positions.float(),
        )

    def _legalized_variants(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        num_hard: int,
    ) -> List[torch.Tensor]:
        """Return legalizer variants for exact scoring.

        In augmented mode, strict legalization can be included as a competitive
        baseline. In pure-plasma mode/R2-replacement mode, strict legalization
        is not even computed: the only load-bearing legalizer is the
        continuation-beta Bohm-sheath path.
        """
        base = project_to_canvas(
            placement.clone().float(),
            benchmark.macro_sizes.float(),
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
            fixed_mask=benchmark.macro_fixed,
            fixed_positions=benchmark.macro_positions.float(),
        )
        use_r2 = bool(getattr(self.cfg, "r2_continuation_enabled", False))
        replace_strict = use_r2 and bool(getattr(self.cfg, "r2_continuation_replace_strict", False))
        variants: List[torch.Tensor] = []
        if not replace_strict:
            self._trace_purity_event("strict_candidate_variant_calls")
            strict_candidate = strict_legalize(
                base.clone(),
                benchmark.macro_sizes.float(),
                benchmark.macro_fixed,
                float(benchmark.canvas_width),
                float(benchmark.canvas_height),
                int(num_hard),
                gap=self.cfg.legalize_gap,
            )
            strict_candidate = project_to_canvas(
                strict_candidate,
                benchmark.macro_sizes.float(),
                float(benchmark.canvas_width),
                float(benchmark.canvas_height),
                fixed_mask=benchmark.macro_fixed,
                fixed_positions=benchmark.macro_positions.float(),
            )
            variants.append(strict_candidate)
        if bool(getattr(self.cfg, "sheath_legalize_enabled", False)) or use_r2:
            self._trace_purity_event("r2_variant_calls")
            sheath_cfg = SheathLegalizeConfig(
                enabled=True,
                fallback_strict=False,
                max_iters=int(getattr(self.cfg, "sheath_legalize_max_iters", 160)),
                dt=float(getattr(self.cfg, "r2_langevin_dt", getattr(self.cfg, "sheath_legalize_dt", 0.35))) if use_r2 else float(getattr(self.cfg, "sheath_legalize_dt", 0.35)),
                damping=float(getattr(self.cfg, "sheath_legalize_damping", 0.60)),
                coulomb_strength=float(getattr(self.cfg, "sheath_legalize_coulomb_strength", 1.0)),
                sheath_strength=float(getattr(self.cfg, "sheath_legalize_sheath_strength", 1.0)),
                debye_frac=float(getattr(self.cfg, "sheath_legalize_debye_frac", 0.02)),
                max_step_frac=float(getattr(self.cfg, "sheath_legalize_max_step_frac", 0.010)),
                continuation_enabled=use_r2,
                beta_overlap_start=float(getattr(self.cfg, "r2_beta_overlap_start", 0.10)),
                beta_overlap_max=float(getattr(self.cfg, "r2_beta_overlap_max", 1.0)),
                beta_overlap_ramp_stages=int(getattr(self.cfg, "r2_beta_overlap_ramp_stages", 8)),
                beta_wall=float(getattr(self.cfg, "r2_beta_wall", 1.0)),
                inner_iters=int(getattr(self.cfg, "r2_inner_iters", 40)),
                temp_start=0.0 if bool(getattr(self.cfg, "r2_deterministic_enabled", False)) else float(getattr(self.cfg, "r2_temp_start", 0.0)),
                temp_decay=float(getattr(self.cfg, "r2_temp_decay", 0.70)),
                force_tol=float(getattr(self.cfg, "r2_force_tol", 1e-5)),
                seed=int(getattr(self.cfg, "r2_seed", 7)),
                hardening_enabled=bool(getattr(self.cfg, "r2_hardening_enabled", True)),
                hardening_iters=int(getattr(self.cfg, "r2_hardening_iters", 320)),
                hardening_max_pairs_per_iter=int(getattr(self.cfg, "r2_hardening_max_pairs_per_iter", 12000)),
                relocation_enabled=bool(getattr(self.cfg, "r2_relocation_enabled", True)),
                relocation_iters=int(getattr(self.cfg, "r2_relocation_iters", 0)),
                relocation_rings=int(getattr(self.cfg, "r2_relocation_rings", 6)),
                relocation_density_weight=float(getattr(self.cfg, "r2_relocation_density_weight", 0.0)),
                flux_lattice_relocation_enabled=bool(getattr(self.cfg, "r2_flux_lattice_relocation_enabled", True)),
                flux_lattice_cols=int(getattr(self.cfg, "r2_flux_lattice_cols", 16)),
                local_flux_refine_enabled=bool(getattr(self.cfg, "r2_local_flux_refine_enabled", False)),
                local_flux_refine_steps=int(getattr(self.cfg, "r2_local_flux_refine_steps", 5)),
                local_flux_refine_pair_threshold=int(getattr(self.cfg, "r2_local_flux_refine_pair_threshold", 4)),
                anchor_strength=float(getattr(self.cfg, "r2_anchor_strength", 0.0)),
                anchor_release_frac=float(getattr(self.cfg, "r2_anchor_release_frac", 0.5)),
            )
            sheath_candidate = bohm_sheath_legalize(
                base.clone(),
                benchmark.macro_sizes.float(),
                benchmark.macro_fixed,
                float(benchmark.canvas_width),
                float(benchmark.canvas_height),
                int(num_hard),
                gap=self.cfg.legalize_gap,
                cfg=sheath_cfg,
            )
            sheath_candidate = project_to_canvas(
                sheath_candidate,
                benchmark.macro_sizes.float(),
                float(benchmark.canvas_width),
                float(benchmark.canvas_height),
                fixed_mask=benchmark.macro_fixed,
                fixed_positions=benchmark.macro_positions.float(),
            )
            if count_hard_overlaps(sheath_candidate, benchmark.macro_sizes.float(), int(num_hard), gap=0.0) == 0:
                variants.append(sheath_candidate)
        return variants

    def _evaluate_exact_candidate(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        num_hard: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[float]]:
        """Project, legalize, and exact-score a candidate placement.

        Used by the inner Picard line search. Returning the legalized
        placement lets the caller continue from the exact candidate that was
        actually scored, not the pre-legalization proposal.
        """
        if plc is None or compute_proxy_cost is None:
            return None, None

        cache_key = None
        if bool(getattr(self.cfg, "plasma_purity_mode", False)):
            base = project_to_canvas(
                placement.clone().float(),
                benchmark.macro_sizes.float(),
                float(benchmark.canvas_width),
                float(benchmark.canvas_height),
                fixed_mask=benchmark.macro_fixed,
                fixed_positions=benchmark.macro_positions.float(),
            )
            cache_key = self._placement_cache_key(benchmark, base)
            cached = self._exact_eval_cache.get(cache_key)
            if cached is not None:
                self._trace_purity_event("pure_exact_eval_cache_hits")
                cached_candidate, cached_cost = cached
                return cached_candidate.clone(), float(cached_cost)
            area = max(float(benchmark.canvas_width) * float(benchmark.canvas_height), 1e-9)
            total_macros = int(benchmark.num_hard_macros) + int(benchmark.num_soft_macros)
            macro_density = float(total_macros) / area
            direct_score_allowed = macro_density >= float(getattr(self.cfg, "plasma_purity_direct_score_min_density", 0.60))
            if direct_score_allowed and count_hard_overlaps(base, benchmark.macro_sizes.float(), int(num_hard), gap=0.0) == 0:
                try:
                    costs = compute_proxy_cost(base, benchmark, plc)
                except Exception:
                    costs = None
                if costs is not None and int(costs.get("overlap_count", 1)) == 0:
                    self._trace_purity_event("pure_direct_legal_score_calls")
                    self._exact_eval_cache[cache_key] = (base.clone(), float(costs["proxy_cost"]))
                    return base, float(costs["proxy_cost"])

        best_candidate: Optional[torch.Tensor] = None
        best_cost: Optional[float] = None
        try:
            variants = self._legalized_variants(placement, benchmark, num_hard)
        except Exception:
            return None, None

        for candidate in variants:
            try:
                costs = compute_proxy_cost(candidate, benchmark, plc)
            except Exception:
                continue
            if int(costs.get("overlap_count", 1)) != 0:
                continue
            cost = float(costs["proxy_cost"])
            if best_cost is None or cost < best_cost:
                best_candidate = candidate
                best_cost = cost
        if cache_key is not None and best_candidate is not None and best_cost is not None:
            self._exact_eval_cache[cache_key] = (best_candidate.clone(), float(best_cost))
        return best_candidate, best_cost

    def _plasma_pellet_shear_variant(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        num_hard: int,
        shear_frac: float,
    ) -> torch.Tensor:
        """R1 multi-start variant: pellet-induced flux-surface shear.

        The startup equilibrium is sheared in poloidal angle as a function of
        normalized minor radius. This is a plasma-native basin change: the
        macro ordering on flux surfaces is perturbed without importing any
        TILOS/Cartesian force-directed behavior.
        """
        if abs(float(shear_frac)) <= 1e-12 or num_hard <= 1:
            return placement.clone().float()
        out = placement.clone().float()
        fixed = benchmark.macro_fixed[:num_hard].bool()
        cx = 0.5 * float(benchmark.canvas_width)
        cy = 0.5 * float(benchmark.canvas_height)
        rx = max(0.5 * float(benchmark.canvas_width), 1e-6)
        ry = max(0.5 * float(benchmark.canvas_height), 1e-6)
        hard = out[:num_hard]
        xn = (hard[:, 0] - cx) / rx
        yn = (hard[:, 1] - cy) / ry
        radius = torch.sqrt(torch.clamp(xn * xn + yn * yn, min=0.0))
        theta = torch.atan2(yn, xn)
        # Pellet deposition drives larger edge shear than core shear.
        delta = float(shear_frac) * math.pi * torch.clamp(radius, 0.0, 1.5) ** 1.5
        theta2 = theta + delta
        sheared = torch.stack(
            [cx + radius * rx * torch.cos(theta2), cy + radius * ry * torch.sin(theta2)],
            dim=1,
        )
        hard[~fixed] = sheared[~fixed]
        out[:num_hard] = hard
        return project_to_canvas(
            out,
            benchmark.macro_sizes.float(),
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
            fixed_mask=benchmark.macro_fixed,
            fixed_positions=benchmark.macro_positions.float(),
        )

    def _select_plasma_multistart_seed(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        num_hard: int,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """Pick the best R1 pellet-shear startup under exact pure scoring."""
        bench_name = str(getattr(benchmark, "name", ""))
        auto_benches = {str(x) for x in getattr(self.cfg, "plasma_init_multistart_auto_benchmarks", [])}
        enabled = bool(getattr(self.cfg, "plasma_init_multistart_enabled", False)) or bench_name in auto_benches
        if not enabled or plc is None or compute_proxy_cost is None:
            return placement, None
        raw_shears = [0.0] + [float(s) for s in getattr(self.cfg, "plasma_init_multistart_shear_fracs", [])]
        max_evals = max(1, int(getattr(self.cfg, "plasma_init_multistart_max_evals", 4)))
        best = placement.clone().float()
        best_cost: Optional[float] = None
        evals = 0
        for shear in raw_shears:
            if evals >= max_evals:
                break
            candidate = self._plasma_pellet_shear_variant(best if abs(shear) <= 1e-12 else placement, benchmark, num_hard, shear)
            try:
                scored = self._legalize_candidate(candidate, benchmark, num_hard)
                costs = compute_proxy_cost(scored, benchmark, plc)
            except Exception:
                continue
            if int(costs.get("overlap_count", 1)) != 0:
                continue
            cost = float(costs["proxy_cost"])
            evals += 1
            self._trace_purity_event("r1_multistart_eval_calls")
            if best_cost is None or float(cost) < best_cost - 1e-6:
                best = scored.clone().float()
                best_cost = float(cost)
                self._trace_purity_event("r1_multistart_accepts")
        if evals > 0:
            self._trace_purity_event("r1_multistart_runs")
        return best, best_cost

    def _run_exact_hard_refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        current_cost: float,
        start_time: float,
        refine_priority: Optional[torch.Tensor] = None,
        flux_directions: Optional[torch.Tensor] = None,
        smooth_priority: Optional[torch.Tensor] = None,
        smooth_directions: Optional[torch.Tensor] = None,
        geom: Optional[CylindricalGeometry] = None,
        smooth_proxy_cfg: Optional[SmoothProxyConfig] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Exact-gated hard refine on plasma-selected macros and directions."""
        if plc is None or compute_proxy_cost is None:
            return placement, current_cost
        if not bool(self.cfg.exact_hard_refine_enabled):
            return placement, current_cost

        num_hard = int(benchmark.num_hard_macros)
        if num_hard <= 0:
            return placement, current_cost

        best = placement.clone().float()
        best_cost = float(current_cost)
        bench_name = str(getattr(benchmark, "name", ""))
        span = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        step_overrides = getattr(self.cfg, "exact_hard_refine_step_frac_by_benchmark", {}) or {}
        step_frac = float(step_overrides.get(bench_name, getattr(self.cfg, "exact_hard_refine_step_frac", 0.008)))
        step_len = step_frac * span
        if step_len <= 0.0:
            return best, best_cost

        degree = torch.zeros(num_hard, dtype=torch.float32)
        if edge_index.numel() > 0:
            edges_cpu = edge_index.detach().cpu().long()
            weights_cpu = edge_weight.detach().cpu().float()
            for e in range(int(edges_cpu.shape[0])):
                a = int(edges_cpu[e, 0].item())
                b = int(edges_cpu[e, 1].item())
                w = float(weights_cpu[e].item())
                if 0 <= a < num_hard:
                    degree[a] += w
                if 0 <= b < num_hard:
                    degree[b] += w
        if float(degree.max().item()) <= 0.0:
            degree = torch.arange(num_hard, 0, -1, dtype=torch.float32)

        if refine_priority is not None and refine_priority.numel() >= num_hard:
            pri = refine_priority[:num_hard].detach().cpu().float()
            pri = torch.clamp(pri, min=0.0)
            max_pri = float(pri.max().item()) if pri.numel() > 0 else 0.0
            if max_pri > 1e-9:
                pri = pri / max_pri
                beta = max(0.0, float(getattr(self.cfg, "exact_hard_refine_mercier_beta", 2.0)))
                degree = degree * (1.0 + beta * pri)

        if smooth_priority is not None and smooth_priority.numel() >= num_hard:
            pri = smooth_priority[:num_hard].detach().cpu().float()
            pri = torch.clamp(pri, min=0.0)
            max_pri = float(pri.max().item()) if pri.numel() > 0 else 0.0
            if max_pri > 1e-9:
                pri = pri / max_pri
                beta = max(0.0, float(getattr(self.cfg, "smooth_proxy_priority_beta", 1.0)))
                degree = degree * (1.0 + beta * pri)

        fixed = benchmark.macro_fixed[:num_hard].bool()
        degree = torch.where(fixed, torch.full_like(degree, -1.0), degree)
        macro_overrides = getattr(self.cfg, "exact_hard_refine_macros_by_benchmark", {}) or {}
        k_raw = int(macro_overrides.get(bench_name, getattr(self.cfg, "exact_hard_refine_macros", 12)))
        k = min(max(1, k_raw), num_hard)
        macro_order = torch.topk(degree, k=k).indices.tolist()
        base_degree = degree.clone()
        direction_overrides = getattr(self.cfg, "exact_hard_refine_direction_set_by_benchmark", {}) or {}
        direction_set = str(
            direction_overrides.get(bench_name, getattr(self.cfg, "exact_hard_refine_direction_set", "axis"))
        ).lower()
        if direction_set == "auto":
            # Feature-route costly geometry instead of benchmark-name routing.
            # Dense hard-macro plasmas benefit most from the extra probes.
            canvas_area = max(1.0, float(benchmark.canvas_width) * float(benchmark.canvas_height))
            hard_density = float(num_hard) / canvas_area
            direction_set = "axis_diag" if hard_density >= 0.30 else "axis"
        pass_decay = max(0.05, float(getattr(self.cfg, "exact_hard_refine_pass_decay", 1.0)))
        pure_direct_refine = bool(getattr(self.cfg, "plasma_purity_mode", False)) and bool(
            getattr(self.cfg, "plasma_purity_refine_direct_score_enabled", False)
        )
        skip_overlap_relegalize = bool(getattr(self.cfg, "plasma_purity_refine_skip_overlap_relegalize", False))
        direct_refine_evals = 0
        eval_overrides = getattr(self.cfg, "plasma_purity_refine_direct_score_max_evals_by_benchmark", {}) or {}
        raw_max_evals = eval_overrides.get(
            str(getattr(benchmark, "name", "")),
            getattr(self.cfg, "plasma_purity_refine_direct_score_max_evals", 96),
        )
        max_direct_refine_evals = max(0, int(raw_max_evals))
        force_overrides = getattr(self.cfg, "exact_hard_refine_force_over_budget_by_benchmark", {}) or {}
        force_over_budget = bool(force_overrides.get(bench_name, False))
        min_eval_overrides = getattr(self.cfg, "exact_hard_refine_min_direct_evals_by_benchmark", {}) or {}
        min_forced_evals = max(0, int(min_eval_overrides.get(bench_name, 0)))
        scale_overrides = getattr(self.cfg, "exact_hard_refine_direction_scales_by_benchmark", {}) or {}
        raw_direction_scales = scale_overrides.get(
            str(getattr(benchmark, "name", "")),
            getattr(self.cfg, "exact_hard_refine_direction_scales", [1.0]),
        )
        direction_scales = [
            float(s)
            for s in raw_direction_scales
            if float(s) > 1e-9
        ]
        if not direction_scales:
            direction_scales = [1.0]

        pass_overrides = getattr(self.cfg, "exact_hard_refine_passes_by_benchmark", {}) or {}
        num_passes = max(1, int(pass_overrides.get(bench_name, getattr(self.cfg, "exact_hard_refine_passes", 1))))
        for pass_idx in range(num_passes):
            pass_step = step_len * (pass_decay ** pass_idx)
            directions = (
                []
                if direction_set in {"flux_only", "plasma", "plasma_only"}
                else [
                    (pass_step, 0.0),
                    (-pass_step, 0.0),
                    (0.0, pass_step),
                    (0.0, -pass_step),
                ]
            )
            if direction_set in {"axis_diag", "diag", "diagonal", "axis+diag", "axis_diag_flux"}:
                diag = pass_step * 0.7071067811865476
                directions.extend(
                    [
                        (diag, diag),
                        (diag, -diag),
                        (-diag, diag),
                        (-diag, -diag),
                    ]
                )
            improved_this_pass = False
            for macro_idx in macro_order:
                if bool(fixed[int(macro_idx)].item()):
                    continue
                over_budget = (time.time() - start_time) > float(self.cfg.exact_hard_refine_max_budget_frac) * float(self.cfg.time_budget_sec)
                if over_budget and not (force_over_budget and direct_refine_evals < min_forced_evals):
                    return best, best_cost
                local_best = best
                local_cost = best_cost
                local_directions = list(directions)
                if (
                    flux_directions is not None
                    and flux_directions.numel() >= (int(macro_idx) + 1) * 2
                    and direction_set
                    in {
                        "axis_flux",
                        "flux",
                        "axis+flux",
                        "axis_diag_flux",
                        "flux_only",
                        "plasma",
                        "plasma_only",
                        "flux_tangent",
                        "flux_tangent_only",
                        "tangent",
                        "tangent_only",
                        "flux_normal",
                        "flux_normal_only",
                        "normal",
                        "normal_only",
                    }
                ):
                    v = flux_directions[int(macro_idx)].detach().cpu().float()
                    norm = float(torch.linalg.norm(v).item())
                    if norm > 1e-9:
                        scale_flux = pass_step * max(0.0, float(getattr(self.cfg, "exact_hard_refine_flux_beta", 1.0)))
                        ux = float(v[0].item()) / norm
                        uy = float(v[1].item()) / norm
                        # Normal-to-flux and tangent-to-flux probes: exact
                        # scoring decides which, if any, actually helps.
                        normal_dirs = [
                            (scale_flux * ux, scale_flux * uy),
                            (-scale_flux * ux, -scale_flux * uy),
                        ]
                        tangent_dirs = [
                            (scale_flux * -uy, scale_flux * ux),
                            (-scale_flux * -uy, -scale_flux * ux),
                        ]
                        if direction_set in {"flux_tangent", "flux_tangent_only", "tangent", "tangent_only"}:
                            local_directions.extend(tangent_dirs)
                        elif direction_set in {"flux_normal", "flux_normal_only", "normal", "normal_only"}:
                            local_directions.extend(normal_dirs)
                        else:
                            local_directions.extend(normal_dirs + tangent_dirs)
                if smooth_directions is not None and smooth_directions.numel() >= (int(macro_idx) + 1) * 2:
                    v = smooth_directions[int(macro_idx)].detach().cpu().float()
                    norm = float(torch.linalg.norm(v).item())
                    if norm > 1e-9:
                        scale_smooth = pass_step * max(0.0, float(getattr(self.cfg, "smooth_proxy_direction_beta", 1.0)))
                        ux = float(v[0].item()) / norm
                        uy = float(v[1].item()) / norm
                        local_directions.extend(
                            [
                                (scale_smooth * ux, scale_smooth * uy),
                                (-scale_smooth * ux, -scale_smooth * uy),
                            ]
                        )
                if len(direction_scales) > 1 or abs(direction_scales[0] - 1.0) > 1e-9:
                    local_directions = [
                        (float(scale) * float(dx), float(scale) * float(dy))
                        for dx, dy in local_directions
                        for scale in direction_scales
                    ]
                for dx, dy in local_directions:
                    candidate = best.clone()
                    candidate[int(macro_idx), 0] += float(dx)
                    candidate[int(macro_idx), 1] += float(dy)
                    scored = None
                    cost = None
                    if pure_direct_refine and (max_direct_refine_evals <= 0 or direct_refine_evals < max_direct_refine_evals):
                        scored, cost = self._score_direct_legal_candidate(
                            candidate,
                            benchmark,
                            plc,
                            num_hard,
                            "pure_refine_direct_score_calls",
                        )
                        if scored is not None and cost is not None:
                            direct_refine_evals += 1
                        elif skip_overlap_relegalize:
                            self._trace_purity_event("pure_refine_overlap_reject_calls")
                            continue
                    if scored is None or cost is None:
                        scored, cost = self._evaluate_exact_candidate(candidate, benchmark, plc, num_hard)
                    if scored is not None and cost is not None and float(cost) < local_cost - 1e-6:
                        local_best = scored
                        local_cost = float(cost)
                if local_cost < best_cost - 1e-6:
                    if bool(self.cfg.diagnostics_enabled):
                        print(
                            f"[team_plasma][hard_refine] bench={benchmark.name} "
                            f"macro={int(macro_idx)} cost={local_cost:.6f} "
                            f"improvement={best_cost - local_cost:.6f}",
                            flush=True,
                        )
                    best = local_best.clone()
                    best_cost = local_cost
                    improved_this_pass = True
            if not improved_this_pass:
                break
            if (
                pass_idx == 0
                and bool(getattr(self.cfg, "ntm_suppression_enabled", False))
                and (improved_this_pass or bool(getattr(self.cfg, "ntm_suppression_always_retarget_enabled", False)))
                and num_passes > 1
                and geom is not None
                and smooth_proxy_cfg is not None
            ):
                try:
                    self._trace_purity_event("ntm_suppression_retarget_calls")
                    # NTM suppression analogy: after one relaxation pass,
                    # persistent top-tail pressure islands receive the next
                    # ECCD-like refinement budget.
                    field = build_smooth_hotspot_field(
                        best.to(device=geom.device, dtype=torch.float32),
                        benchmark.macro_sizes.to(device=geom.device, dtype=torch.float32),
                        edge_index.to(device=geom.device),
                        edge_weight.to(device=geom.device),
                        geom,
                        smooth_proxy_cfg,
                        fixed_mask=benchmark.macro_fixed.to(device=geom.device),
                    )
                    residual = bilinear_sample(field, best[:num_hard].to(device=geom.device), geom).detach().cpu().float()
                    residual = torch.clamp(residual, min=0.0)
                    max_residual = float(residual.max().item()) if residual.numel() > 0 else 0.0
                    if max_residual > 1e-9:
                        residual = residual / max_residual
                        boost = max(1.0, float(getattr(self.cfg, "ntm_suppression_budget_boost", 1.5)))
                        beta = max(0.0, float(getattr(self.cfg, "ntm_suppression_smooth_beta", 2.0)))
                        ntm_score = base_degree * (1.0 + beta * residual[:num_hard])
                        ntm_score = torch.where(fixed, torch.full_like(ntm_score, -1.0), ntm_score)
                        ntm_k = min(max(k, int(math.ceil(float(k) * boost))), num_hard)
                        macro_order = torch.topk(ntm_score, k=ntm_k).indices.tolist()
                        self._trace_purity_event("ntm_suppression_retarget_success")
                        if bool(self.cfg.diagnostics_enabled):
                            print(
                                f"[team_plasma][ntm_suppression] bench={benchmark.name} "
                                f"macros={ntm_k} beta={beta:.3f}",
                                flush=True,
                            )
                except Exception:
                    pass

        return best, best_cost

    def _run_thermal_hopping(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        current_cost: float,
        start_time: float,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        refine_priority: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Exact-gated ELM/sawtooth-style structural basin hops.

        This is deliberately not a smooth-field gradient. A small set of
        unstable hard macros receives stochastic thermal kicks, then the
        normal legalization/exact proxy gate decides whether the post-crash
        equilibrium is a better basin.
        """
        if not bool(getattr(self.cfg, "thermal_hopping_enabled", False)):
            return placement, current_cost
        if plc is None or compute_proxy_cost is None:
            return placement, current_cost
        if (time.time() - start_time) > float(getattr(self.cfg, "thermal_hopping_max_budget_frac", 0.96)) * float(self.cfg.time_budget_sec):
            return placement, current_cost

        num_hard = int(benchmark.num_hard_macros)
        if num_hard <= 0:
            return placement, current_cost

        best = placement.clone().float()
        best_cost = float(current_cost)
        fixed = benchmark.macro_fixed[:num_hard].bool()
        span = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        sigma = max(1e-9, float(getattr(self.cfg, "thermal_hopping_sigma_frac", 0.035)) * span)
        trials = max(0, int(getattr(self.cfg, "thermal_hopping_trials", 5)))
        n_move = min(num_hard, max(1, int(getattr(self.cfg, "thermal_hopping_n_macros", 8))))
        pool_mult = max(1, int(getattr(self.cfg, "thermal_hopping_top_pool_mult", 4)))
        min_delta = max(1e-6, float(getattr(self.cfg, "thermal_hopping_min_delta", 0.003)))

        if refine_priority is not None and refine_priority.numel() >= num_hard:
            priority = refine_priority[:num_hard].detach().cpu().float()
        else:
            priority = self._hard_macro_degrees(num_hard, edge_index, edge_weight).detach().cpu().float()
        if bool(fixed.any()):
            priority[fixed] = -1.0

        active = torch.nonzero(~fixed, as_tuple=False).reshape(-1)
        if active.numel() == 0:
            return best, best_cost
        pool_k = min(int(active.numel()), max(n_move, n_move * pool_mult))
        pool = torch.topk(priority, k=pool_k).indices

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(getattr(self.cfg, "thermal_hopping_seed", 31415)))
        for trial in range(trials):
            if (time.time() - start_time) > float(getattr(self.cfg, "thermal_hopping_max_budget_frac", 0.96)) * float(self.cfg.time_budget_sec):
                break
            perm = torch.randperm(pool.numel(), generator=gen)[:n_move]
            chosen = pool[perm]
            candidate = best.clone()
            # ELM crashes are intermittent: most kicks are modest, but some
            # trials use a stronger sawtooth-like displacement.
            crash_scale = 1.0 + 0.5 * float(trial % 3 == 2)
            kicks = torch.randn((chosen.numel(), 2), generator=gen) * (sigma * crash_scale)
            candidate[chosen, :2] += kicks
            candidate = project_to_canvas(
                candidate,
                benchmark.macro_sizes.float(),
                float(benchmark.canvas_width),
                float(benchmark.canvas_height),
                fixed_mask=benchmark.macro_fixed,
                fixed_positions=benchmark.macro_positions.float(),
            )
            scored, cost = self._evaluate_exact_candidate(candidate, benchmark, plc, num_hard)
            if scored is not None and cost is not None and float(cost) < best_cost - min_delta:
                if bool(self.cfg.diagnostics_enabled):
                    print(
                        f"[team_plasma][thermal_hopping] bench={benchmark.name} "
                        f"trial={trial} cost={float(cost):.6f} improvement={best_cost - float(cost):.6f}",
                        flush=True,
                    )
                best = scored.clone()
                best_cost = float(cost)
        return best, best_cost

    def _run_flux_rope_cluster_relocation(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        current_cost: float,
        start_time: float,
    ) -> Tuple[torch.Tensor, float]:
        """Exact-gated flux-rope relocation for large, open pure-plasma cases.

        Single-macro flux refinements behave like local MHD relaxation. On
        large canvases the problematic basin can instead be a connected
        current channel sitting in the wrong flux surface. This operator moves
        small graph-connected macro islands coherently, analogous to a flux
        rope reconnecting and settling onto a neighboring surface. The official
        proxy remains the only acceptance gate.
        """
        if not bool(getattr(self.cfg, "flux_rope_cluster_enabled", False)):
            return placement, current_cost
        if plc is None or compute_proxy_cost is None:
            return placement, current_cost

        max_frac = float(getattr(self.cfg, "flux_rope_cluster_max_budget_frac", 0.94))
        if max_frac > 0.0 and (time.time() - start_time) > max_frac * float(self.cfg.time_budget_sec):
            return placement, current_cost

        num_hard = int(benchmark.num_hard_macros)
        if num_hard <= 1:
            return placement, current_cost

        area = max(float(benchmark.canvas_width) * float(benchmark.canvas_height), 1e-9)
        macro_density = float(int(benchmark.num_hard_macros) + int(benchmark.num_soft_macros)) / area
        if macro_density >= float(getattr(self.cfg, "flux_rope_cluster_density_threshold", 0.60)):
            return placement, current_cost
        if area < float(getattr(self.cfg, "flux_rope_cluster_area_threshold", 3000.0)):
            return placement, current_cost

        fixed = benchmark.macro_fixed[:num_hard].detach().cpu().bool()
        active = torch.nonzero(~fixed, as_tuple=False).reshape(-1)
        if active.numel() == 0:
            return placement, current_cost

        edges = edge_index.detach().cpu().long()
        weights = edge_weight.detach().cpu().float()
        degree = torch.zeros(num_hard, dtype=torch.float32)
        adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(num_hard)]
        if edges.numel() > 0:
            for e in range(int(edges.shape[0])):
                a = int(edges[e, 0].item())
                b = int(edges[e, 1].item())
                w = float(weights[e].item()) if e < int(weights.numel()) else 1.0
                if 0 <= a < num_hard and 0 <= b < num_hard and a != b:
                    degree[a] += max(0.0, w)
                    degree[b] += max(0.0, w)
                    adjacency[a].append((b, w))
                    adjacency[b].append((a, w))
        if float(degree.max().item()) <= 0.0:
            degree = torch.ones(num_hard, dtype=torch.float32)
        if bool(fixed.any()):
            degree[fixed] = -1.0

        n_roots = min(max(1, int(getattr(self.cfg, "flux_rope_cluster_roots", 5))), int(active.numel()))
        max_cluster = min(max(1, int(getattr(self.cfg, "flux_rope_cluster_macros", 10))), int(active.numel()))
        roots = torch.topk(degree, k=n_roots).indices.tolist()
        scales = [float(s) for s in getattr(self.cfg, "flux_rope_cluster_scales", [0.5, 1.0, 1.5]) if float(s) > 0.0]
        span = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        step = max(0.0, float(getattr(self.cfg, "flux_rope_cluster_step_frac", 0.012))) * span
        if step <= 0.0 or not scales:
            return placement, current_cost

        center = torch.tensor(
            [0.5 * float(benchmark.canvas_width), 0.5 * float(benchmark.canvas_height)],
            dtype=torch.float32,
        )
        best = placement.clone().float()
        best_cost = float(current_cost)
        trials = 0
        accepts = 0
        direct_evals = 0
        max_direct_evals = max(0, int(getattr(self.cfg, "flux_rope_cluster_max_direct_evals", 12)))

        def build_cluster(root: int) -> List[int]:
            cluster: List[int] = []
            seen = {int(root)}
            frontier = [int(root)]
            while frontier and len(cluster) < max_cluster:
                node = frontier.pop(0)
                if bool(fixed[node].item()):
                    continue
                cluster.append(node)
                nbrs = sorted(adjacency[node], key=lambda item: item[1], reverse=True)
                for nbr, _w in nbrs:
                    if len(seen) >= max_cluster * 3:
                        break
                    if 0 <= nbr < num_hard and nbr not in seen and not bool(fixed[nbr].item()):
                        seen.add(nbr)
                        frontier.append(nbr)
            return cluster

        for root in roots:
            if max_frac > 0.0 and (time.time() - start_time) > max_frac * float(self.cfg.time_budget_sec):
                break
            cluster = build_cluster(int(root))
            if not cluster:
                continue
            pts = best[cluster, :2].detach().cpu().float()
            centroid = torch.mean(pts, dim=0)
            radial = centroid - center
            norm = float(torch.linalg.norm(radial).item())
            if norm <= 1e-9:
                radial = torch.tensor([1.0, 0.0], dtype=torch.float32)
            else:
                radial = radial / norm
            tangent = torch.tensor([-float(radial[1].item()), float(radial[0].item())], dtype=torch.float32)
            raw_dirs = [
                radial,
                -radial,
                tangent,
                -tangent,
                radial + 0.5 * tangent,
                radial - 0.5 * tangent,
            ]
            directions = []
            for direction in raw_dirs:
                dnorm = float(torch.linalg.norm(direction).item())
                if dnorm > 1e-9:
                    directions.append(direction / dnorm)
            for direction in directions:
                if max_direct_evals > 0 and direct_evals >= max_direct_evals:
                    break
                for scale in scales:
                    if max_frac > 0.0 and (time.time() - start_time) > max_frac * float(self.cfg.time_budget_sec):
                        break
                    if max_direct_evals > 0 and direct_evals >= max_direct_evals:
                        break
                    candidate = best.clone()
                    delta = float(scale) * step * direction
                    for macro_idx in cluster:
                        candidate[int(macro_idx), 0] += float(delta[0].item())
                        candidate[int(macro_idx), 1] += float(delta[1].item())
                    candidate = project_to_canvas(
                        candidate,
                        benchmark.macro_sizes.float(),
                        float(benchmark.canvas_width),
                        float(benchmark.canvas_height),
                        fixed_mask=benchmark.macro_fixed,
                        fixed_positions=benchmark.macro_positions.float(),
                    )
                    trials += 1
                    # Cluster relocation starts from a legal equilibrium and
                    # only applies rigid island moves. If the move remains
                    # overlap-free, score it directly instead of running the
                    # expensive continuation-beta legalizer again for every
                    # micro-probe. Overlapping probes are simply rejected.
                    if count_hard_overlaps(candidate, benchmark.macro_sizes.float(), int(num_hard), gap=0.0) != 0:
                        continue
                    if max_direct_evals > 0 and direct_evals >= max_direct_evals:
                        break
                    try:
                        costs = compute_proxy_cost(candidate, benchmark, plc)
                    except Exception:
                        continue
                    if int(costs.get("overlap_count", 1)) != 0:
                        continue
                    direct_evals += 1
                    self._trace_purity_event("flux_rope_cluster_direct_score_calls")
                    cost = float(costs["proxy_cost"])
                    if cost < best_cost - 1e-6:
                        if bool(self.cfg.diagnostics_enabled):
                            print(
                                f"[team_plasma][flux_rope_cluster] bench={benchmark.name} "
                                f"root={int(root)} macros={len(cluster)} scale={float(scale):.3f} "
                                f"cost={cost:.6f} improvement={best_cost - cost:.6f}",
                                flush=True,
                            )
                        best = candidate.clone()
                        best_cost = cost
                        accepts += 1

        if trials > 0:
            self._trace_purity_event("flux_rope_cluster_trials")
        if accepts > 0:
            self._trace_purity_event("flux_rope_cluster_accepts")
        return best, best_cost

    def _run_smooth_proxy_global_step(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        current_cost: float,
        start_time: float,
        smooth_priority: Optional[torch.Tensor],
        smooth_directions: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, float]:
        """Exact-gated coordinated move along the smooth top-k hotspot field.

        This is the minimum viable HAAMP-B1 global move: the smooth proxy
        proposes a collective hard-macro displacement, but the official proxy
        remains the only acceptance criterion.
        """
        if plc is None or compute_proxy_cost is None:
            return placement, current_cost
        if not bool(getattr(self.cfg, "smooth_proxy_global_step_enabled", False)):
            return placement, current_cost

        num_hard = int(benchmark.num_hard_macros)
        if (
            num_hard <= 0
            or smooth_priority is None
            or smooth_directions is None
            or smooth_priority.numel() < num_hard
            or smooth_directions.numel() < num_hard * 2
        ):
            return placement, current_cost

        if (time.time() - start_time) > float(self.cfg.smooth_proxy_global_step_max_budget_frac) * float(self.cfg.time_budget_sec):
            return placement, current_cost

        best = placement.clone().float()
        best_cost = float(current_cost)
        span = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        step_len = float(self.cfg.smooth_proxy_global_step_frac) * span
        if step_len <= 0.0:
            return best, best_cost

        pri = smooth_priority[:num_hard].detach().cpu().float()
        pri = torch.clamp(pri, min=0.0)
        fixed = benchmark.macro_fixed[:num_hard].detach().cpu().bool()
        pri = torch.where(fixed, torch.full_like(pri, -1.0), pri)
        k = min(max(1, int(self.cfg.smooth_proxy_global_step_macros)), num_hard)
        active = torch.topk(pri, k=k).indices.tolist()

        pri_for_weight = torch.clamp(pri, min=0.0)
        max_pri = float(pri_for_weight.max().item()) if pri_for_weight.numel() > 0 else 0.0
        if max_pri > 1e-9:
            pri_for_weight = pri_for_weight / max_pri
        else:
            pri_for_weight = torch.ones_like(pri_for_weight)

        scales = [float(s) for s in getattr(self.cfg, "smooth_proxy_global_step_scales", [0.25, 0.5, 1.0])]
        directions_cpu = smooth_directions[:num_hard].detach().cpu().float()
        for scale in scales:
            if scale <= 0.0:
                continue
            for sign in (1.0, -1.0):
                if (time.time() - start_time) > float(self.cfg.smooth_proxy_global_step_max_budget_frac) * float(self.cfg.time_budget_sec):
                    return best, best_cost
                candidate = best.clone()
                moved = 0
                for macro_idx in active:
                    if bool(fixed[int(macro_idx)].item()):
                        continue
                    v = directions_cpu[int(macro_idx)]
                    norm = float(torch.linalg.norm(v).item())
                    if norm <= 1e-9:
                        continue
                    weight = 0.25 + 0.75 * float(pri_for_weight[int(macro_idx)].item())
                    candidate[int(macro_idx), 0] += sign * scale * step_len * weight * float(v[0].item()) / norm
                    candidate[int(macro_idx), 1] += sign * scale * step_len * weight * float(v[1].item()) / norm
                    moved += 1
                if moved <= 0:
                    continue
                scored, cost = self._evaluate_exact_candidate(candidate, benchmark, plc, num_hard)
                if scored is not None and cost is not None and float(cost) < best_cost - 1e-6:
                    if bool(self.cfg.diagnostics_enabled):
                        print(
                            f"[team_plasma][smooth_global] bench={benchmark.name} "
                            f"scale={scale:.3f} sign={sign:+.0f} macros={moved} "
                            f"cost={float(cost):.6f} improvement={best_cost - float(cost):.6f}",
                            flush=True,
                        )
                    best = scored
                    best_cost = float(cost)

        return best, best_cost

    def _run_eigenmode_refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        current_cost: float,
        start_time: float,
        smooth_priority: Optional[torch.Tensor],
        smooth_directions: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, float]:
        """Exact-gated R3/BFKK eigenmode-following refinement."""
        if plc is None or compute_proxy_cost is None:
            return placement, current_cost
        name = str(getattr(benchmark, "name", ""))
        auto_benches = {str(x) for x in getattr(self.cfg, "eigenmode_refine_auto_benchmarks", [])}
        enabled = bool(getattr(self.cfg, "eigenmode_refine_enabled", False)) or (
            bool(getattr(self.cfg, "eigenmode_refine_auto_enabled", False)) and name in auto_benches
        )
        if not enabled:
            return placement, current_cost

        num_hard = int(benchmark.num_hard_macros)
        if (
            num_hard <= 1
            or smooth_priority is None
            or smooth_directions is None
            or smooth_priority.numel() < num_hard
            or smooth_directions.numel() < num_hard * 2
        ):
            return placement, current_cost
        if (time.time() - start_time) > float(self.cfg.eigenmode_refine_max_budget_frac) * float(self.cfg.time_budget_sec):
            return placement, current_cost

        degree = torch.zeros(num_hard, dtype=torch.float32)
        if edge_index.numel() > 0:
            edges_cpu = edge_index.detach().cpu().long()
            weights_cpu = edge_weight.detach().cpu().float()
            for e in range(int(edges_cpu.shape[0])):
                a = int(edges_cpu[e, 0].item())
                b = int(edges_cpu[e, 1].item())
                w = float(weights_cpu[e].item())
                if 0 <= a < num_hard:
                    degree[a] += w
                if 0 <= b < num_hard:
                    degree[b] += w
        if float(degree.max().item()) <= 0.0:
            degree = torch.arange(num_hard, 0, -1, dtype=torch.float32)

        cfg = EigenmodeRefineConfig(
            enabled=True,
            n_macros=int(getattr(self.cfg, "eigenmode_refine_macros", 32)),
            n_modes=int(getattr(self.cfg, "eigenmode_refine_modes", 4)),
            graph_tension=float(getattr(self.cfg, "eigenmode_refine_graph_tension", 0.20)),
            pressure_drive=float(getattr(self.cfg, "eigenmode_refine_pressure_drive", 1.0)),
            ridge=float(getattr(self.cfg, "eigenmode_refine_ridge", 1e-3)),
            mode_scales=list(getattr(self.cfg, "eigenmode_refine_scales", [0.25, 0.5, 1.0, 1.5])),
            step_frac=float(getattr(self.cfg, "eigenmode_refine_step_frac", 0.006)),
            max_budget_frac=float(getattr(self.cfg, "eigenmode_refine_max_budget_frac", 0.90)),
        )
        fixed = benchmark.macro_fixed[:num_hard].detach().cpu().bool()
        active = select_eigenmode_active_macros(
            degree,
            smooth_priority[:num_hard],
            fixed,
            cfg.n_macros,
        )
        modes, evals = build_bfkk_modes(
            active,
            edge_index,
            edge_weight,
            smooth_priority[:num_hard],
            smooth_directions[:num_hard],
            cfg,
        )
        if not modes:
            return placement, current_cost

        def normalize_mode(raw_mode: torch.Tensor) -> torch.Tensor:
            out = raw_mode.detach().cpu().float().clone()
            out[fixed] = 0.0
            active_norm = torch.sqrt(torch.mean(torch.sum(out[:num_hard] * out[:num_hard], dim=1))).clamp_min(1e-9)
            return out / active_norm

        if bool(getattr(self.cfg, "eigenmode_refine_trust_combos_enabled", True)):
            # Trust-region mini-basis: combine the lowest BFKK modes instead
            # of trying only one eigenmode at a time. This approximates the
            # small amplitude solve delta_X = V alpha from the Proxy-EFIT
            # research plan while preserving exact-score gating.
            n_combo = min(max(0, int(getattr(self.cfg, "eigenmode_refine_trust_combo_modes", 4))), len(modes))
            combo_modes: List[torch.Tensor] = []
            for i in range(n_combo):
                for j in range(i + 1, n_combo):
                    combo_modes.append(normalize_mode(modes[i] + modes[j]))
                    combo_modes.append(normalize_mode(modes[i] - modes[j]))
            if combo_modes:
                modes = modes + combo_modes

        best = placement.clone().float()
        best_cost = float(current_cost)
        span = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        step_len = float(cfg.step_frac) * span
        if step_len <= 0.0:
            return best, best_cost

        for mode_idx, mode in enumerate(modes):
            if (time.time() - start_time) > float(cfg.max_budget_frac) * float(self.cfg.time_budget_sec):
                return best, best_cost
            # In BFKK language negative modes are unstable; near-zero modes are
            # marginal and still useful, so exact scoring decides rather than
            # hard-filtering by eigenvalue sign.
            eval_idx = min(mode_idx // 2, max(0, int(evals.numel()) - 1))
            eig = float(evals[eval_idx].item()) if evals.numel() > 0 else 0.0
            mode = mode.detach().cpu().float()
            mode[fixed] = 0.0
            for scale in cfg.mode_scales:
                if scale <= 0.0:
                    continue
                for sign in (1.0, -1.0):
                    candidate = best.clone()
                    candidate[:num_hard] = candidate[:num_hard] + sign * float(scale) * step_len * mode[:num_hard]
                    scored, cost = self._evaluate_exact_candidate(candidate, benchmark, plc, num_hard)
                    if scored is not None and cost is not None and float(cost) < best_cost - 1e-6:
                        if bool(self.cfg.diagnostics_enabled):
                            print(
                                f"[team_plasma][eigenmode] bench={benchmark.name} "
                                f"mode={mode_idx} eig={eig:.4e} scale={float(scale):.3f} "
                                f"sign={sign:+.0f} cost={float(cost):.6f} "
                                f"improvement={best_cost - float(cost):.6f}",
                                flush=True,
                            )
                        best = scored
                        best_cost = float(cost)
                        # Recomputeing modes is more principled but expensive;
                        # keep this pass cheap and let exact_hard_refine polish.
                        return best, best_cost

        return best, best_cost

    def _run_tearing_mode_refine(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        current_cost: float,
        start_time: float,
    ) -> Tuple[torch.Tensor, float]:
        """Exact-gated coordinated moves on connectivity islands.

        This is the minimal HAAMP-B3 probe: identify small hard-macro islands
        from the netlist graph, then try cluster translations, breathing moves,
        and small rotations. The official proxy gates every accepted tear.
        """
        if plc is None or compute_proxy_cost is None:
            return placement, current_cost
        name = str(getattr(benchmark, "name", ""))
        auto_benches = {str(x) for x in getattr(self.cfg, "tearing_mode_auto_benchmarks", [])}
        enabled = bool(getattr(self.cfg, "tearing_mode_enabled", False)) or (
            bool(getattr(self.cfg, "tearing_mode_auto_enabled", False)) and name in auto_benches
        )
        if not enabled:
            return placement, current_cost

        num_hard = int(benchmark.num_hard_macros)
        if num_hard <= 1 or edge_index.numel() == 0:
            return placement, current_cost
        if (time.time() - start_time) > float(self.cfg.tearing_mode_max_budget_frac) * float(self.cfg.time_budget_sec):
            return placement, current_cost

        fixed = benchmark.macro_fixed[:num_hard].detach().cpu().bool()
        edges_cpu = edge_index.detach().cpu().long()
        weights_cpu = edge_weight.detach().cpu().float()
        adjacency: List[List[Tuple[float, int]]] = [[] for _ in range(num_hard)]
        degree = torch.zeros(num_hard, dtype=torch.float32)
        for e in range(int(edges_cpu.shape[0])):
            a = int(edges_cpu[e, 0].item())
            b = int(edges_cpu[e, 1].item())
            w = float(weights_cpu[e].item())
            if 0 <= a < num_hard and 0 <= b < num_hard and a != b:
                adjacency[a].append((w, b))
                adjacency[b].append((w, a))
                degree[a] += w
                degree[b] += w
        for nbrs in adjacency:
            nbrs.sort(reverse=True)

        degree = torch.where(fixed, torch.full_like(degree, -1.0), degree)
        n_clusters = min(max(1, int(self.cfg.tearing_mode_n_clusters)), num_hard)
        cluster_size = min(max(2, int(self.cfg.tearing_mode_cluster_size)), num_hard)
        seeds = torch.topk(degree, k=n_clusters).indices.tolist()

        clusters: List[List[int]] = []
        seen_keys = set()
        for seed in seeds:
            if bool(fixed[int(seed)].item()):
                continue
            cluster = [int(seed)]
            frontier = [int(seed)]
            used = {int(seed)}
            while frontier and len(cluster) < cluster_size:
                node = frontier.pop(0)
                for _, nbr in adjacency[node]:
                    if nbr in used or bool(fixed[nbr].item()):
                        continue
                    used.add(nbr)
                    cluster.append(nbr)
                    frontier.append(nbr)
                    if len(cluster) >= cluster_size:
                        break
            if len(cluster) < 2:
                continue
            key = tuple(sorted(cluster))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            clusters.append(cluster)

        if not clusters:
            return placement, current_cost

        best = placement.clone().float()
        best_cost = float(current_cost)
        span = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        step_len = float(self.cfg.tearing_mode_step_frac) * span
        if step_len <= 0.0:
            return best, best_cost

        scales = [float(s) for s in getattr(self.cfg, "tearing_mode_scales", [0.5, 1.0])]
        base_dirs = [
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (0.7071067811865476, 0.7071067811865476),
            (0.7071067811865476, -0.7071067811865476),
            (-0.7071067811865476, 0.7071067811865476),
            (-0.7071067811865476, -0.7071067811865476),
        ]

        def try_candidate(candidate: torch.Tensor, label: str, cluster_len: int) -> None:
            nonlocal best, best_cost
            scored, cost = self._evaluate_exact_candidate(candidate, benchmark, plc, num_hard)
            if scored is not None and cost is not None and float(cost) < best_cost - 1e-6:
                if bool(self.cfg.diagnostics_enabled):
                    print(
                        f"[team_plasma][tearing] bench={benchmark.name} "
                        f"{label} macros={cluster_len} cost={float(cost):.6f} "
                        f"improvement={best_cost - float(cost):.6f}",
                        flush=True,
                    )
                best = scored
                best_cost = float(cost)

        for cluster_idx, cluster in enumerate(clusters):
            if (time.time() - start_time) > float(self.cfg.tearing_mode_max_budget_frac) * float(self.cfg.time_budget_sec):
                return best, best_cost
            cluster_tensor = torch.tensor(cluster, dtype=torch.long)
            for scale in scales:
                if scale <= 0.0:
                    continue
                delta = step_len * scale
                for dx_unit, dy_unit in base_dirs:
                    candidate = best.clone()
                    candidate[cluster_tensor, 0] += delta * dx_unit
                    candidate[cluster_tensor, 1] += delta * dy_unit
                    try_candidate(candidate, f"cluster={cluster_idx} translate scale={scale:.2f}", len(cluster))

                points = best[cluster_tensor].clone()
                centroid = points.mean(dim=0, keepdim=True)
                offsets = points - centroid
                norms = torch.linalg.norm(offsets, dim=1, keepdim=True).clamp_min(1e-9)
                for sign in (1.0, -1.0):
                    candidate = best.clone()
                    candidate[cluster_tensor] += sign * delta * offsets / norms
                    try_candidate(candidate, f"cluster={cluster_idx} breathe scale={scale:.2f} sign={sign:+.0f}", len(cluster))

                theta = math.radians(float(self.cfg.tearing_mode_rotation_deg) * scale)
                for sign in (1.0, -1.0):
                    c = math.cos(sign * theta)
                    s = math.sin(sign * theta)
                    rot = torch.tensor([[c, -s], [s, c]], dtype=points.dtype)
                    candidate = best.clone()
                    candidate[cluster_tensor] = centroid + offsets @ rot.T
                    try_candidate(candidate, f"cluster={cluster_idx} rotate scale={scale:.2f} sign={sign:+.0f}", len(cluster))

        return best, best_cost

    def _eccd_reweight_edges(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        geom: CylindricalGeometry,
        smooth_proxy_cfg: SmoothProxyConfig,
    ) -> torch.Tensor:
        """Boost graph currents crossing residual hotspot islands.

        Plasma analogy: electron-cyclotron current drive is most effective
        when deposited at the magnetic-island center. Here the smooth top-tail
        field identifies resonant congestion islands; nets whose bboxes cross
        those islands receive extra current/attention in later B1/B3/R3 probes.
        """
        if not bool(getattr(self.cfg, "eccd_reweight_enabled", False)):
            return edge_weight
        if edge_index.numel() == 0 or edge_weight.numel() == 0:
            return edge_weight
        try:
            device = geom.device
            pos = placement.to(device=device, dtype=torch.float32)
            sizes = benchmark.macro_sizes.to(device=device, dtype=torch.float32)
            field = build_smooth_hotspot_field(
                pos,
                sizes,
                edge_index.to(device=device),
                edge_weight.to(device=device),
                geom,
                smooth_proxy_cfg,
                fixed_mask=benchmark.macro_fixed.to(device=device),
            )
            if field.numel() == 0:
                return edge_weight
            top_frac = min(0.50, max(0.01, float(getattr(self.cfg, "eccd_reweight_top_frac", 0.05))))
            threshold = torch.quantile(field.reshape(-1), 1.0 - top_frac)
            mask = (field >= threshold).float().detach().cpu()
            if float(mask.sum().item()) <= 0.0:
                return edge_weight

            edges = edge_index.detach().cpu().long()
            pos_cpu = placement.detach().cpu().float()
            boost = torch.zeros_like(edge_weight.detach().cpu().float())
            locality = min(1.0, max(0.0, float(getattr(self.cfg, "eccd_reweight_locality", 0.0))))
            for e in range(int(edges.shape[0])):
                a = int(edges[e, 0].item())
                b = int(edges[e, 1].item())
                if not (0 <= a < pos_cpu.shape[0] and 0 <= b < pos_cpu.shape[0]):
                    continue
                x0 = min(float(pos_cpu[a, 0].item()), float(pos_cpu[b, 0].item()))
                x1 = max(float(pos_cpu[a, 0].item()), float(pos_cpu[b, 0].item()))
                y0 = min(float(pos_cpu[a, 1].item()), float(pos_cpu[b, 1].item()))
                y1 = max(float(pos_cpu[a, 1].item()), float(pos_cpu[b, 1].item()))
                c0 = max(0, min(geom.grid_cols - 1, int(math.floor(x0 / max(float(geom.dR), 1e-9)))))
                c1 = max(0, min(geom.grid_cols - 1, int(math.ceil(x1 / max(float(geom.dR), 1e-9)))))
                r0 = max(0, min(geom.grid_rows - 1, int(math.floor(y0 / max(float(geom.dZ), 1e-9)))))
                r1 = max(0, min(geom.grid_rows - 1, int(math.ceil(y1 / max(float(geom.dZ), 1e-9)))))
                if r1 < r0 or c1 < c0:
                    continue
                bbox_drive = float(mask[r0 : r1 + 1, c0 : c1 + 1].mean().item())
                if locality > 0.0:
                    ca = max(0, min(geom.grid_cols - 1, int(round(float(pos_cpu[a, 0].item()) / max(float(geom.dR), 1e-9)))))
                    cb = max(0, min(geom.grid_cols - 1, int(round(float(pos_cpu[b, 0].item()) / max(float(geom.dR), 1e-9)))))
                    ra = max(0, min(geom.grid_rows - 1, int(round(float(pos_cpu[a, 1].item()) / max(float(geom.dZ), 1e-9)))))
                    rb = max(0, min(geom.grid_rows - 1, int(round(float(pos_cpu[b, 1].item()) / max(float(geom.dZ), 1e-9)))))
                    endpoint_drive = 0.5 * (float(mask[ra, ca].item()) + float(mask[rb, cb].item()))
                    bbox_drive = (1.0 - locality) * bbox_drive + locality * endpoint_drive
                boost[e] = bbox_drive
            kappa = max(0.0, float(getattr(self.cfg, "eccd_reweight_kappa", 1.0)))
            return edge_weight * (1.0 + kappa * boost.to(edge_weight.device, dtype=edge_weight.dtype))
        except Exception:
            return edge_weight

    def _run_snowflake_divertor_split(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        current_cost: float,
        start_time: float,
        geom: CylindricalGeometry,
        smooth_proxy_cfg: SmoothProxyConfig,
    ) -> Tuple[torch.Tensor, float]:
        """Exact-gated Ryutov snowflake-divertor split of one hotspot island."""
        if not bool(getattr(self.cfg, "snowflake_divertor_enabled", False)):
            return placement, current_cost
        self._trace_purity_event("snowflake_divertor_calls")
        if plc is None or compute_proxy_cost is None:
            self._trace_purity_event("snowflake_divertor_no_plc")
            return placement, current_cost
        max_frac = float(getattr(self.cfg, "snowflake_divertor_max_budget_frac", 0.92))
        if (time.time() - start_time) > max_frac * float(self.cfg.time_budget_sec):
            return placement, current_cost
        num_hard = int(benchmark.num_hard_macros)
        if num_hard <= 0:
            return placement, current_cost

        try:
            device = geom.device
            pos = placement.detach().to(device=device, dtype=torch.float32)
            sizes = benchmark.macro_sizes.detach().to(device=device, dtype=torch.float32)
            fixed = benchmark.macro_fixed.detach().to(device=device).bool()
            field = build_smooth_hotspot_field(
                pos,
                sizes,
                edge_index.to(device=device),
                edge_weight.to(device=device),
                geom,
                smooth_proxy_cfg,
                fixed_mask=fixed,
            )
            if field.numel() == 0:
                return placement, current_cost
            top_frac = min(0.50, max(0.01, float(getattr(self.cfg, "snowflake_divertor_top_frac", 0.05))))
            threshold = torch.quantile(field.reshape(-1), 1.0 - top_frac)
            masked = torch.where(field >= threshold, field, torch.full_like(field, -1.0))
            peak_flat = int(torch.argmax(masked).item())
            peak_row = peak_flat // int(field.shape[1])
            peak_col = peak_flat % int(field.shape[1])
            hotspot = torch.tensor(
                [(float(peak_col) + 0.5) * float(geom.dR), (float(peak_row) + 0.5) * float(geom.dZ)],
                dtype=torch.float32,
                device=device,
            )

            hard_pos = pos[:num_hard]
            hard_sizes = sizes[:num_hard]
            half = 0.5 * hard_sizes
            dx = torch.clamp(torch.abs(hard_pos[:, 0] - hotspot[0]) - half[:, 0], min=0.0)
            dy = torch.clamp(torch.abs(hard_pos[:, 1] - hotspot[1]) - half[:, 1], min=0.0)
            dist = torch.sqrt(dx * dx + dy * dy)
            dist = torch.where(fixed[:num_hard], torch.full_like(dist, float("inf")), dist)
            n_split = min(max(2, int(getattr(self.cfg, "snowflake_divertor_macros", 8))), num_hard)
            chosen = torch.topk(-dist, k=n_split).indices
            if chosen.numel() == 0 or not bool(torch.isfinite(dist[chosen]).any()):
                return placement, current_cost

            vectors = hard_pos[chosen] - hotspot.unsqueeze(0)
            chosen = chosen[torch.argsort(torch.atan2(vectors[:, 1], vectors[:, 0]))]
            span = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
            step = float(getattr(self.cfg, "snowflake_divertor_step_frac", 0.008)) * span
            scales = [float(s) for s in getattr(self.cfg, "snowflake_divertor_scales", [0.5, 1.0])]
            max_evals = max(0, int(getattr(self.cfg, "snowflake_divertor_max_direct_evals", 8)))
            direct_evals = 0
            best = placement.detach().cpu().float().clone()
            best_cost = float(current_cost)

            for scale in scales:
                if max_evals > 0 and direct_evals >= max_evals:
                    break
                if (time.time() - start_time) > max_frac * float(self.cfg.time_budget_sec):
                    break
                candidate = placement.detach().cpu().float().clone()
                ray_mode = str(getattr(self.cfg, "snowflake_divertor_ray_mode", "radial")).lower()
                for slot, macro_idx in enumerate(chosen.detach().cpu().tolist()):
                    if ray_mode == "radial":
                        vec = (hard_pos[int(macro_idx)] - hotspot).detach().cpu().float()
                        norm = float(torch.linalg.norm(vec).item())
                        if norm <= 1e-9:
                            theta = 2.0 * math.pi * float(slot) / max(1, int(chosen.numel()))
                            ray = torch.tensor([math.cos(theta), math.sin(theta)], dtype=torch.float32)
                        else:
                            ray = vec / norm
                    else:
                        theta = 2.0 * math.pi * float(slot) / max(1, int(chosen.numel()))
                        ray = torch.tensor([math.cos(theta), math.sin(theta)], dtype=torch.float32)
                    candidate[int(macro_idx), :2] += float(scale) * step * ray
                candidate = project_to_canvas(
                    candidate,
                    benchmark.macro_sizes.float(),
                    float(benchmark.canvas_width),
                    float(benchmark.canvas_height),
                    fixed_mask=benchmark.macro_fixed,
                    fixed_positions=benchmark.macro_positions.float(),
                )
                if count_hard_overlaps(candidate, benchmark.macro_sizes.float(), num_hard, gap=0.0) != 0:
                    if bool(getattr(self.cfg, "snowflake_divertor_relegalize_enabled", False)):
                        self._trace_purity_event("snowflake_divertor_relegalize_calls")
                        candidate = self._legalize_candidate(candidate, benchmark, num_hard)
                    if count_hard_overlaps(candidate, benchmark.macro_sizes.float(), num_hard, gap=0.0) != 0:
                        self._trace_purity_event("snowflake_divertor_overlap_rejects")
                        continue
                try:
                    costs = compute_proxy_cost(candidate, benchmark, plc)
                except Exception:
                    continue
                if int(costs.get("overlap_count", 1)) != 0:
                    self._trace_purity_event("snowflake_divertor_overlap_rejects")
                    continue
                direct_evals += 1
                self._trace_purity_event("snowflake_divertor_direct_score_calls")
                cost = float(costs["proxy_cost"])
                if cost < best_cost - 1e-6:
                    best = candidate.clone()
                    best_cost = cost
                    self._trace_purity_event("snowflake_divertor_accepts")
                else:
                    self._trace_purity_event("snowflake_divertor_proxy_rejects")
            if direct_evals > 0:
                self._trace_purity_event("snowflake_divertor_trials")
            return best, best_cost
        except Exception:
            return placement, current_cost

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        self._reset_purity_trace()
        set_seed(self.cfg.seed)
        self._apply_plasma_purity_mode()
        if bool(getattr(self.cfg, "plasma_purity_mode", False)):
            self._trace_purity_event("plasma_purity_mode_runs")
        device = resolve_device(self.cfg.device)

        # Trivial cases: nothing movable.
        if bool(getattr(self.cfg, "plasma_init_enabled", False)):
            # R1 purity rule: movable macros must not inherit the TILOS PLC
            # coordinates. Fixed macros retain their official constraint
            # positions; every movable center is produced by plasma startup.
            placement = torch.zeros_like(benchmark.macro_positions).float()
            fixed_all = benchmark.macro_fixed.bool()
            placement[fixed_all] = benchmark.macro_positions.float()[fixed_all]
            self._trace_purity_event("r1_plc_free_seed_calls")
        else:
            placement = benchmark.macro_positions.clone().float()
            self._trace_purity_event("tilos_plc_seed_calls")
        num_hard = int(benchmark.num_hard_macros)
        if num_hard <= 1:
            return placement

        plc = self._plc_for_eval(benchmark)

        # Netlist edges between hard macros. Net info lives in the plc.
        edge_index_cpu, edge_weight_cpu = extract_hard_edges(benchmark, plc)
        all_edge_index_cpu, all_edge_weight_cpu = extract_all_macro_edges_from_plc(benchmark, plc)
        if bool(getattr(self.cfg, "plasma_init_enabled", False)):
            self._trace_purity_event("r1_plasma_init_calls")
            soft_spread_frac = float(getattr(self.cfg, "plasma_init_soft_spread_frac", 0.018))
            area = max(float(benchmark.canvas_width) * float(benchmark.canvas_height), 1e-9)
            total_macros = int(benchmark.num_hard_macros) + int(benchmark.num_soft_macros)
            macro_density = float(total_macros) / area
            if bool(getattr(self.cfg, "plasma_init_soft_spread_auto_enabled", False)):
                if (
                    bool(getattr(self.cfg, "plasma_init_soft_spread_ultra_open_enabled", False))
                    and macro_density < float(getattr(self.cfg, "plasma_init_soft_spread_ultra_open_threshold", 0.60))
                ):
                    soft_spread_frac = float(getattr(self.cfg, "plasma_init_soft_spread_ultra_open_frac", 0.030))
                elif macro_density >= float(getattr(self.cfg, "plasma_init_soft_spread_dense_threshold", 1.20)):
                    soft_spread_frac = float(getattr(self.cfg, "plasma_init_soft_spread_dense_frac", 0.012))
                else:
                    soft_spread_frac = float(getattr(self.cfg, "plasma_init_soft_spread_open_frac", 0.020))
            phase_lock_enabled = bool(getattr(self.cfg, "plasma_init_soft_phase_lock_enabled", False))
            if bool(getattr(self.cfg, "plasma_init_soft_phase_lock_auto_enabled", False)):
                phase_lock_enabled = phase_lock_enabled and (
                    macro_density >= float(getattr(self.cfg, "plasma_init_soft_phase_lock_min_density", 0.60))
                )
                if (
                    bool(getattr(self.cfg, "plasma_init_soft_phase_lock_large_open_enabled", False))
                    and macro_density < float(getattr(self.cfg, "plasma_init_soft_phase_lock_large_open_density", 0.60))
                    and area >= float(getattr(self.cfg, "plasma_init_soft_phase_lock_large_open_area", 3000.0))
                ):
                    phase_lock_enabled = bool(getattr(self.cfg, "plasma_init_soft_phase_lock_enabled", False))
                    self._trace_purity_event("r1_large_open_phase_lock_calls")
            phase_lock_weight = float(getattr(self.cfg, "plasma_init_soft_phase_lock_weight", 0.0))
            if (
                phase_lock_enabled
                and bool(getattr(self.cfg, "plasma_init_soft_phase_lock_large_open_enabled", False))
                and macro_density < float(getattr(self.cfg, "plasma_init_soft_phase_lock_large_open_density", 0.60))
                and area >= float(getattr(self.cfg, "plasma_init_soft_phase_lock_large_open_area", 3000.0))
            ):
                phase_lock_weight = float(getattr(self.cfg, "plasma_init_soft_phase_lock_large_open_weight", phase_lock_weight))
            normalized_laplacian = bool(getattr(self.cfg, "plasma_init_normalized_laplacian", True))
            if bool(getattr(self.cfg, "plasma_init_normalized_laplacian_auto_enabled", False)):
                small_hard = num_hard <= int(getattr(self.cfg, "plasma_init_unnormalized_small_hard_threshold", 220))
                dense_enough = macro_density >= float(getattr(self.cfg, "plasma_init_unnormalized_min_density", 0.80))
                if small_hard and dense_enough:
                    # In compact, lower-hard-count regimes the unnormalized
                    # current ramp preserves high-current hubs better; in
                    # large/open regimes normalized modes avoid axis collapse.
                    normalized_laplacian = False
                    self._trace_purity_event("r1_unnormalized_dense_startup_calls")
            cong_flux_auto = {str(x) for x in getattr(self.cfg, "plasma_init_congestion_aware_flux_auto_benchmarks", [])}
            cong_flux_enabled = bool(getattr(self.cfg, "plasma_init_congestion_aware_flux_enabled", False)) or (
                str(getattr(benchmark, "name", "")) in cong_flux_auto
            )
            if cong_flux_enabled:
                self._trace_purity_event("r1_congestion_aware_flux_calls")
            placement = spectral_plasma_startup_init(
                placement,
                benchmark.macro_sizes.float(),
                benchmark.macro_fixed,
                edge_index_cpu,
                edge_weight_cpu,
                float(benchmark.canvas_width),
                float(benchmark.canvas_height),
                num_hard=num_hard,
                margin_frac=float(getattr(self.cfg, "plasma_init_margin_frac", 0.04)),
                soft_grid_enabled=bool(getattr(self.cfg, "plasma_init_soft_grid_enabled", False)),
                soft_graph_enabled=bool(getattr(self.cfg, "plasma_init_soft_graph_enabled", True)),
                soft_assignment_enabled=bool(getattr(self.cfg, "plasma_init_soft_assignment_enabled", False)),
                soft_assignment_graph_weight=float(getattr(self.cfg, "plasma_init_soft_assignment_graph_weight", 0.0)),
                soft_assignment_overlap_weight=float(getattr(self.cfg, "plasma_init_soft_assignment_overlap_weight", 0.0)),
                soft_assignment_candidates=int(getattr(self.cfg, "plasma_init_soft_assignment_candidates", 48)),
                soft_phase_lock_enabled=phase_lock_enabled,
                soft_phase_lock_weight=phase_lock_weight,
                soft_phase_lock_distance_weight=float(getattr(self.cfg, "plasma_init_soft_phase_lock_distance_weight", 1.0)),
                flux_band_enabled=bool(getattr(self.cfg, "plasma_init_flux_bands_enabled", True)),
                all_edge_index=all_edge_index_cpu,
                all_edge_weight=all_edge_weight_cpu,
                soft_spread_frac=soft_spread_frac,
                soft_pressure_iters=int(getattr(self.cfg, "plasma_init_soft_pressure_iters", 0)),
                soft_pressure_step_frac=float(getattr(self.cfg, "plasma_init_soft_pressure_step_frac", 0.002)),
                soft_pressure_radius_frac=float(getattr(self.cfg, "plasma_init_soft_pressure_radius_frac", 0.035)),
                soft_pressure_anchor=float(getattr(self.cfg, "plasma_init_soft_pressure_anchor", 0.35)),
                soft_pressure_hard_weight=float(getattr(self.cfg, "plasma_init_soft_pressure_hard_weight", 1.0)),
                soft_pressure_wall_weight=float(getattr(self.cfg, "plasma_init_soft_pressure_wall_weight", 0.5)),
                normalized_laplacian=normalized_laplacian,
                n_bands=int(getattr(self.cfg, "plasma_init_n_bands", 8)),
                sinkhorn_iters=int(getattr(self.cfg, "plasma_init_sinkhorn_iters", 100)),
                sinkhorn_eps=float(getattr(self.cfg, "plasma_init_sinkhorn_eps", 0.05)),
                bare_picard_outer=int(getattr(self.cfg, "plasma_init_bare_picard_outer", 6)),
                weber_sweeps=int(getattr(self.cfg, "plasma_init_weber_sweeps", 2)),
                congestion_aware_flux_enabled=cong_flux_enabled,
                congestion_aware_flux_weight=float(getattr(self.cfg, "plasma_init_congestion_aware_flux_weight", 0.25)),
            )
            placement, multistart_cost = self._select_plasma_multistart_seed(
                placement,
                benchmark,
                plc,
                num_hard,
            )
        else:
            multistart_cost = None

        # SAFETY-NET BASELINE: legalize the initial placement and evaluate the
        # legalized result. In purity mode this is still the R1 plasma startup
        # passed through R2, not the forbidden TILOS PLC warm start.
        baseline = self._legalize_candidate(
            placement.clone().float(),
            benchmark,
            num_hard,
        )
        best_placement = baseline
        best_cost: Optional[float] = None
        if multistart_cost is not None:
            best_placement = placement.clone().float()
            best_cost = float(multistart_cost)
        seed_auto = {str(x) for x in getattr(self.cfg, "plasma_purity_seed_best_auto_benchmarks", [])}
        seed_from_legal = bool(getattr(self.cfg, "plasma_purity_seed_best_from_legalized_baseline", False)) or (
            str(getattr(benchmark, "name", "")) in seed_auto
        )
        seed_candidate = baseline.clone().float() if seed_from_legal else placement.clone().float()
        baseline_scored, baseline_cost = self._evaluate_exact_candidate(
            seed_candidate,
            benchmark,
            plc,
            num_hard,
        )
        if baseline_scored is not None and baseline_cost is not None:
            if best_cost is None or float(baseline_cost) < float(best_cost) - 1e-6:
                best_placement = baseline_scored
                best_cost = float(baseline_cost)

        # Geometry.
        grid_rows, grid_cols = _compute_grid_shape(
            benchmark.canvas_width, benchmark.canvas_height, self.cfg.grid_size
        )
        geom = make_geometry(
            canvas_width=float(benchmark.canvas_width),
            canvas_height=float(benchmark.canvas_height),
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            aspect_ratio=self.cfg.aspect_ratio,
            device=device,
        )

        edge_index = edge_index_cpu.to(device)
        edge_weight = edge_weight_cpu.to(device)

        # Split hard / soft.
        sizes_all = benchmark.macro_sizes.float().to(device)
        fixed_all = benchmark.macro_fixed.clone().to(device)
        hard_pos = placement[:num_hard].clone().to(device)
        hard_sizes = sizes_all[:num_hard]
        hard_fixed = fixed_all[:num_hard]
        hard_fixed_pos = hard_pos.clone()

        num_soft = int(benchmark.num_soft_macros)
        if num_soft > 0:
            soft_pos = placement[num_hard:].clone().to(device)
            soft_sizes = sizes_all[num_hard:]
            soft_fixed = fixed_all[num_hard:]
            soft_fixed_pos = soft_pos.clone()
        else:
            soft_pos = torch.zeros(0, 2, device=device)
            soft_sizes = torch.zeros(0, 2, device=device)
            soft_fixed = torch.zeros(0, dtype=torch.bool, device=device)
            soft_fixed_pos = soft_pos.clone()

        # Coil currents from connectivity.
        coil_cfg = CoilConfig(
            I_0=self.cfg.coil_I_0,
            use_degree=self.cfg.coil_use_degree,
            use_area=self.cfg.coil_use_area,
            integrate_force=self.cfg.coil_integrate_force,
            force_scale=self.cfg.coil_force_scale,
        )
        coils = make_macro_coils(hard_pos, hard_sizes, hard_fixed, edge_index, edge_weight, coil_cfg)

        # Initial psi (smooth bump satisfying Dirichlet BC).
        psi = _warm_start_psi(geom)

        profile_cfg = ProfileConfig(
            n_bins=self.cfg.profile_n_bins,
            poly_degree=self.cfg.profile_poly_degree,
            alpha_q_rho=self.cfg.profile_alpha_q_rho,
            F_vac=self.cfg.profile_F_vac,
            beta_F=self.cfg.profile_beta_F,
            tail_enabled=self.cfg.profile_tail_enabled,
            tail_quantile=self.cfg.profile_tail_quantile,
            tail_gamma=self.cfg.profile_tail_gamma,
            tail_power=self.cfg.profile_tail_power,
        )
        stability_cfg = StabilityConfig(
            enabled=self.cfg.mercier_enabled,
            mercier_alpha=self.cfg.mercier_alpha,
            mercier_quantile=self.cfg.mercier_quantile,
            mercier_power=self.cfg.mercier_power,
            cap=self.cfg.mercier_cap,
            suydam_enabled=self.cfg.suydam_enabled,
            suydam_gamma=self.cfg.suydam_gamma,
            suydam_cap=self.cfg.suydam_cap,
            suydam_eps=self.cfg.suydam_eps,
        )
        two_fluid_cfg = TwoFluidConfig(
            enabled=self.cfg.two_fluid_enabled,
            drift_step_frac=self.cfg.two_fluid_drift_step_frac,
            trust_radius_frac=self.cfg.two_fluid_trust_radius_frac,
            T_e=self.cfg.two_fluid_T_e,
            pde_enabled=getattr(self.cfg, "two_fluid_pde_enabled", False),
            D_parallel=getattr(self.cfg, "two_fluid_D_parallel", 0.20),
            D_perp=getattr(self.cfg, "two_fluid_D_perp", 0.05),
            dt=getattr(self.cfg, "two_fluid_dt", 0.20),
            max_iters=getattr(self.cfg, "two_fluid_max_iters", 80),
            conv_tol=getattr(self.cfg, "two_fluid_conv_tol", 1e-4),
            density_weight=getattr(self.cfg, "two_fluid_pde_density_weight", 1.0),
            psi_weight=getattr(self.cfg, "two_fluid_pde_psi_weight", 0.15),
            trust_radius_pde_frac=getattr(self.cfg, "two_fluid_pde_trust_radius_frac", 0.01),
            psi_aligned_anisotropy_enabled=bool(getattr(self.cfg, "r4_use_psi_aligned_flow", False)),
            min_anisotropy_ratio=float(getattr(self.cfg, "r4_min_anisotropy_ratio", 100.0)),
            q_weight=float(getattr(self.cfg, "r4_q_weight", 0.0)),
        )
        benchmark_name = str(getattr(benchmark, "name", ""))
        canvas_w = float(getattr(benchmark, "canvas_width", 0.0))
        canvas_h = float(getattr(benchmark, "canvas_height", 0.0))
        canvas_span_feature = max(canvas_w, canvas_h, 1e-9)
        near_square_canvas = abs(canvas_w - canvas_h) / canvas_span_feature <= float(
            getattr(self.cfg, "r4_picard_large_square_aspect_tol", 0.08)
        )
        large_square_canvas = (
            min(canvas_w, canvas_h) >= float(getattr(self.cfg, "r4_picard_large_square_min_canvas", 60.0))
            and max(canvas_w, canvas_h) <= float(getattr(self.cfg, "r4_picard_large_square_max_canvas", 1.0e9))
            and near_square_canvas
        )
        r4_picard_feature_disabled = bool(getattr(self.cfg, "r4_picard_large_square_disable_enabled", False)) and large_square_canvas
        if r4_picard_feature_disabled:
            self._trace_purity_event("r4_picard_large_square_disabled")
        r4_picard_soft_enabled = bool(getattr(self.cfg, "r4_picard_soft_equilibrium_enabled", False)) and not r4_picard_feature_disabled
        r5_auto_benches = {str(x) for x in getattr(self.cfg, "r5_pic_auto_benchmarks", [])}
        r5_enabled = (
            bool(getattr(self.cfg, "r5_pic_enabled", False))
            or benchmark_name in r5_auto_benches
            or bool(getattr(self.cfg, "r5_pic_dimensionless_auto_enabled", False))
        )
        pic_cfg = PICConfig(
            enabled=r5_enabled,
            max_iters=int(getattr(self.cfg, "r5_pic_max_iters", 32)),
            dt_frac=float(getattr(self.cfg, "r5_pic_dt_frac", 0.08)),
            debye_length_frac=float(getattr(self.cfg, "r5_pic_debye_length_frac", 0.025)),
            sheath_width_frac=float(getattr(self.cfg, "r5_pic_sheath_width_frac", 0.035)),
            ion_to_electron_mass_ratio=float(getattr(self.cfg, "r5_pic_ion_to_electron_mass_ratio", 100.0)),
            net_current_scale=float(getattr(self.cfg, "r5_pic_net_current_scale", 0.30)),
            coulomb_strength=float(getattr(self.cfg, "r5_pic_coulomb_strength", 0.15)),
            electric_strength=float(getattr(self.cfg, "r5_pic_electric_strength", 0.45)),
            magnetic_strength=float(getattr(self.cfg, "r5_pic_magnetic_strength", 0.20)),
            sheath_strength=float(getattr(self.cfg, "r5_pic_sheath_strength", 0.40)),
            collision_freq=float(getattr(self.cfg, "r5_pic_collision_freq", 0.12)),
            trust_radius_frac=float(getattr(self.cfg, "r5_pic_trust_radius_frac", 0.006)),
            field_solve_iters=int(getattr(self.cfg, "r5_pic_field_solve_iters", 42)),
            field_solver=str(getattr(self.cfg, "r5_pic_field_solver", "jacobi")),
            conv_tol=float(getattr(self.cfg, "r5_pic_conv_tol", 1e-4)),
            current_line_samples=int(getattr(self.cfg, "r5_pic_current_line_samples", 12)),
            b0_strength=float(getattr(self.cfg, "r5_pic_b0_strength", 0.0)),
            b_ripple_strength=float(getattr(self.cfg, "r5_pic_b_ripple_strength", 0.0)),
            grad_b_drift_strength=float(getattr(self.cfg, "r5_pic_grad_b_drift_strength", 0.0)),
            magnetic_mirror_strength=float(getattr(self.cfg, "r5_pic_magnetic_mirror_strength", 0.0)),
            pair_attraction_strength=float(getattr(self.cfg, "r5_pic_pair_attraction_strength", 0.0)),
            pair_attraction_range_frac=float(getattr(self.cfg, "r5_pic_pair_attraction_range_frac", 1.0)),
            pair_attraction_softening_frac=float(getattr(self.cfg, "r5_pic_pair_attraction_softening_frac", 0.01)),
            bootstrap_current_strength=float(getattr(self.cfg, "r5_pic_bootstrap_current_strength", 0.0)),
            diamagnetic_drift_strength=float(getattr(self.cfg, "r5_pic_diamagnetic_drift_strength", 0.0)),
            annealed_schedule_enabled=bool(getattr(self.cfg, "r5_pic_annealed_schedule_enabled", False)),
        )
        all_edge_index = all_edge_index_cpu.to(device)
        all_edge_weight = all_edge_weight_cpu.to(device)
        smooth_auto_benches = {str(x) for x in getattr(self.cfg, "smooth_proxy_auto_benchmarks", [])}
        smooth_proxy_is_enabled = bool(self.cfg.smooth_proxy_enabled) or (
            bool(getattr(self.cfg, "smooth_proxy_auto_enabled", False)) and benchmark_name in smooth_auto_benches
        )
        smooth_proxy_cfg = SmoothProxyConfig(
            enabled=smooth_proxy_is_enabled,
            density_weight=self.cfg.smooth_proxy_density_weight,
            rudy_weight=self.cfg.smooth_proxy_rudy_weight,
            top_frac=self.cfg.smooth_proxy_top_frac,
            lse_tau=self.cfg.smooth_proxy_lse_tau,
            gyro_radius_frac=self.cfg.smooth_proxy_gyro_radius_frac,
            hpwl_weight=getattr(self.cfg, "smooth_proxy_hpwl_weight", 0.0),
            smooth_rudy_enabled=getattr(self.cfg, "smooth_rudy_enabled", False),
            smooth_rudy_bbox_alpha=getattr(self.cfg, "smooth_rudy_bbox_alpha", 16.0),
            smooth_rudy_rasterize_sharpness=getattr(self.cfg, "smooth_rudy_rasterize_sharpness", 8.0),
            bohm_sheath_boundary_enabled=getattr(self.cfg, "bohm_sheath_boundary_enabled", False),
            bohm_sheath_boundary_weight=getattr(self.cfg, "bohm_sheath_boundary_weight", 0.25),
            bohm_sheath_boundary_width_frac=getattr(self.cfg, "bohm_sheath_boundary_width_frac", 0.15),
            bohm_sheath_limiter_enabled=getattr(self.cfg, "bohm_sheath_limiter_enabled", False),
            bohm_sheath_limiter_quantile=getattr(self.cfg, "bohm_sheath_limiter_quantile", 0.75),
            large_macro_boundary_bias_enabled=getattr(self.cfg, "large_macro_boundary_bias_enabled", False),
            large_macro_boundary_weight=getattr(self.cfg, "large_macro_boundary_weight", 0.5),
            large_macro_boundary_quantile=getattr(self.cfg, "large_macro_boundary_quantile", 0.75),
            priority_beta=self.cfg.smooth_proxy_priority_beta,
            direction_beta=self.cfg.smooth_proxy_direction_beta,
        )
        gs_cfg = GSSolverConfig(
            picard_outer=1,        # one Picard step per outer iter (we do the loop here)
            gs_inner_sweeps=self.cfg.gs_inner_sweeps,
            omega_sor=self.cfg.omega_sor,
            omega_picard=self.cfg.omega_picard,
            convergence_tol=self.cfg.convergence_tol,
            mu0_eff=self.cfg.mu0_eff,
            use_newton_solver=self.cfg.use_newton_solver,
            newton_outer_max=self.cfg.newton_outer_max,
            newton_tol_residual=self.cfg.newton_tol_residual,
            newton_tol_step=self.cfg.newton_tol_step,
            newton_armijo_c1=self.cfg.newton_armijo_c1,
            newton_min_alpha=self.cfg.newton_min_alpha,
        )

        start_time = time.time()
        span = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        macro_step = self.cfg.macro_step_frac * span
        macro_trust = self.cfg.macro_trust_radius_frac * span
        exact_ls_rounds = 0
        exact_ls_scales = [
            float(s) for s in getattr(self.cfg, "exact_line_search_scales", [0.25, 0.5, 1.0])
        ]
        exact_ls_period = max(1, int(self.cfg.exact_line_search_period))
        exact_ls_start = max(0, int(self.cfg.exact_line_search_start_iter))
        exact_ls_max_rounds = max(0, int(self.cfg.exact_line_search_max_rounds))
        q_field: Optional[torch.Tensor] = None

        for k in range(int(self.cfg.picard_outer)):
            if time.time() - start_time > 0.85 * self.cfg.time_budget_sec:
                break

            # 1) Build q, rho from current macro positions (hard + soft).
            all_pos = torch.cat([hard_pos, soft_pos], dim=0)
            all_sizes = torch.cat([hard_sizes, soft_sizes], dim=0)
            q_field = build_rudy_field(all_pos, edge_index, edge_weight, geom)
            rho_field = build_density_field(all_pos, all_sizes, geom)
            J_phi_field = deposit_coil_currents(coils, geom)

            # 2) Fit profiles p(psi), F(psi).
            profiles = fit_profiles(
                psi=psi,
                q_field=q_field,
                rho_field=rho_field,
                edge_index=edge_index,
                edge_weight=edge_weight,
                positions=all_pos,
                geom=geom,
                cfg=profile_cfg,
            )

            # 3) Apply tail shaping. Suydam is the first-principles
            #    cylindrical limit; otherwise use the older Mercier proxy.
            if not bool(self.cfg.suydam_enabled):
                profiles = apply_tail_shaping(profiles, stability_cfg)

            # 4) Build the RHS function for the GS solve.
            R_grid = geom.R_grid()
            R_sq = R_grid * R_grid
            mu0_eff = float(self.cfg.mu0_eff)
            coil_rhs = coil_rhs_contribution(coils, geom, mu0_eff)
            suydam_boost = suydam_rhs_boost(psi, profiles, geom, mu0_eff, stability_cfg)

            def rhs_fn(psi_iter: torch.Tensor) -> torch.Tensor:
                p_prime_val = profiles.p_prime_fn(psi_iter)
                F_val = profiles.F_fn(psi_iter)
                F_prime_val = profiles.F_prime_fn(psi_iter)
                return (
                    -mu0_eff * R_sq * p_prime_val * suydam_boost
                    - F_val * F_prime_val
                    + coil_rhs
                )

            # 5) Solve Delta-star psi = rhs. Newton is available behind a
            #    config flag while we validate it against the stable Picard path.
            if bool(self.cfg.use_newton_solver):
                psi_new, _trace = solve_grad_shafranov_newton(
                    rhs_fn,
                    profiles,
                    geom,
                    psi_init=psi,
                    cfg=gs_cfg,
                )
            else:
                psi_new, _trace = solve_grad_shafranov(rhs_fn, geom, psi_init=psi, cfg=gs_cfg)
            if bool(self.cfg.diagnostics_enabled):
                solver_name = "newton" if bool(self.cfg.use_newton_solver) else "picard"
                residual_text = ",".join(f"{r:.3e}" for r in _trace.residuals[:8])
                print(
                    f"[team_plasma][solver] bench={benchmark.name} outer={k} "
                    f"solver={solver_name} converged={_trace.converged} "
                    f"residuals={residual_text}",
                    flush=True,
                )

            # 6) Damped outer-loop blend, then Taylor global relaxation overlay.
            omega_outer = 0.5
            psi_gs = (1.0 - omega_outer) * psi + omega_outer * psi_new
            if bool(self.cfg.taylor_enabled):
                denom = max(int(self.cfg.picard_outer) - 1, 1)
                frac = float(k) / float(denom)
                ramp_start = float(self.cfg.taylor_ramp_start_frac)
                ramp_end = max(float(self.cfg.taylor_ramp_end_frac), ramp_start + 1e-6)
                ramp = max(0.0, min(1.0, (frac - ramp_start) / (ramp_end - ramp_start)))
                gamma_t = float(self.cfg.taylor_gamma) * ramp
                if gamma_t > 0.0:
                    psi_taylor = taylor_beltrami_mode(
                        geom,
                        reference=psi_gs,
                        source=J_phi_field,
                    )
                    psi = (1.0 - gamma_t) * psi_gs + gamma_t * psi_taylor
                else:
                    psi = psi_gs
            else:
                psi = psi_gs
            psi[0, :] = 0.0
            psi[-1, :] = 0.0
            psi[:, 0] = 0.0
            psi[:, -1] = 0.0

            # 7) Forces on hard macros from the equilibrium.
            forces = macro_force_from_psi(psi, coils, geom, coil_cfg)

            # Hard-macro Ware-pinch (pass 187): add a cross-field drift toward
            # lower routing congestion on top of the GS equilibrium force.
            # Hard macros dominate the q-field structure (their net bboxes
            # contribute the most demand), so moving them down q-gradient
            # restructures the entire congestion landscape. q_field is
            # recomputed each Picard iter so the gradient is always fresh.
            hard_q_weight = float(getattr(self.cfg, "r4_hard_q_weight", 0.0))
            if hard_q_weight != 0.0 and q_field is not None and q_field.numel() > 0:
                self._trace_purity_event("hard_macro_ware_pinch_calls")
                q_dev = q_field.to(device=forces.device, dtype=forces.dtype)
                q_grad_x = torch.zeros_like(q_dev)
                q_grad_y = torch.zeros_like(q_dev)
                q_grad_x[:, 1:-1] = (q_dev[:, 2:] - q_dev[:, :-2]) / max(2.0 * float(geom.dR), 1e-9)
                q_grad_y[1:-1, :] = (q_dev[2:, :] - q_dev[:-2, :]) / max(2.0 * float(geom.dZ), 1e-9)
                grad_q_at_hard = sample_vector(q_grad_x, q_grad_y, hard_pos, geom)
                grad_q_norms = torch.linalg.norm(grad_q_at_hard, dim=1, keepdim=True)
                q_denom = torch.quantile(grad_q_norms.reshape(-1), 0.90) if grad_q_at_hard.shape[0] >= 10 else grad_q_norms.max()
                if float(q_denom.item()) > 1e-9:
                    grad_q_at_hard = grad_q_at_hard / q_denom
                forces = forces - hard_q_weight * grad_q_at_hard

            force_norms = torch.linalg.norm(forces, dim=1, keepdim=True)
            denom = torch.quantile(force_norms.reshape(-1), 0.90) if forces.shape[0] >= 10 else force_norms.max()
            denom_val = float(denom.item()) if force_norms.numel() > 0 else 0.0
            if denom_val > 1e-9:
                forces = forces / denom_val

            # 8) Trust-radius step on hard macros.
            step = macro_step * forces
            step_norm = torch.linalg.norm(step, dim=1, keepdim=True)
            scale = torch.clamp(macro_trust / torch.clamp(step_norm, min=1e-9), max=1.0)
            step = step * scale
            if bool(hard_fixed.any()):
                step[hard_fixed] = 0.0

            prev_hard_pos = hard_pos.clone()
            prev_soft_pos = soft_pos.clone()
            used_exact_line_search = False
            run_exact_line_search = (
                bool(self.cfg.exact_line_search_enabled)
                and compute_proxy_cost is not None
                and best_cost is not None
                and plc is not None
                and exact_ls_rounds < exact_ls_max_rounds
                and k >= exact_ls_start
                and ((k - exact_ls_start) % exact_ls_period == 0)
                and (time.time() - start_time) <= float(self.cfg.exact_line_search_max_budget_frac) * float(self.cfg.time_budget_sec)
            )

            if run_exact_line_search and exact_ls_scales:
                exact_ls_rounds += 1
                used_exact_line_search = True
                round_best_cost = float(best_cost)
                round_best_placement: Optional[torch.Tensor] = None

                for step_scale in exact_ls_scales:
                    cand_hard = prev_hard_pos + float(step_scale) * step
                    cand_hard = project_to_canvas(
                        cand_hard,
                        hard_sizes,
                        geom.canvas_width,
                        geom.canvas_height,
                        fixed_mask=hard_fixed,
                        fixed_positions=hard_fixed_pos,
                    )

                    candidate = torch.zeros_like(placement)
                    candidate[:num_hard] = cand_hard.detach().cpu()
                    if num_soft > 0:
                        candidate[num_hard:] = prev_soft_pos.detach().cpu()
                    if bool(fixed_all.any()):
                        candidate[fixed_all.cpu()] = benchmark.macro_positions[fixed_all.cpu()].float()

                    scored_candidate, candidate_cost = self._evaluate_exact_candidate(
                        candidate,
                        benchmark,
                        plc,
                        num_hard,
                    )
                    if (
                        scored_candidate is not None
                        and candidate_cost is not None
                        and candidate_cost < round_best_cost - 1e-6
                    ):
                        round_best_cost = float(candidate_cost)
                        round_best_placement = scored_candidate

                if round_best_placement is not None:
                    hard_pos = round_best_placement[:num_hard].to(device)
                    if num_soft > 0:
                        soft_pos = round_best_placement[num_hard:].to(device)
                    best_placement = round_best_placement.clone()
                    best_cost = float(round_best_cost)
                else:
                    hard_pos = prev_hard_pos
                    soft_pos = prev_soft_pos
            else:
                hard_pos = hard_pos + step

            # 9) Soft macros drift (adiabatic electrons). Exact-scored rounds
            # skip this unscored update so the accepted placement remains the
            # one that actually passed the proxy gate.
            if not used_exact_line_search:
                if (
                    bool(getattr(pic_cfg, "enabled", False))
                    and not bool(getattr(self.cfg, "r5_pic_proposal_only_enabled", False))
                    and num_soft > 0
                ):
                    self._trace_purity_event("r5_pic_calls")
                    soft_pos = pic_relax_soft_macros(
                        hard_pos,
                        hard_sizes,
                        soft_pos,
                        soft_sizes,
                        all_edge_index,
                        all_edge_weight,
                        geom,
                        pic_cfg,
                        soft_fixed=soft_fixed,
                    )
                    if not bool(getattr(self.cfg, "r5_pic_chain_r4_enabled", False)):
                        pass
                    elif (
                        bool(r4_picard_soft_enabled)
                        and bool(getattr(two_fluid_cfg, "pde_enabled", False))
                        and (time.time() - start_time) <= float(getattr(self.cfg, "r4_picard_max_budget_frac", 0.70)) * float(self.cfg.time_budget_sec)
                    ):
                        self._trace_purity_event("r4_two_fluid_calls")
                        picard_cfg = TwoFluidConfig(**two_fluid_cfg.__dict__)
                        step_overrides = getattr(self.cfg, "r4_inner_steps_by_benchmark", {}) or {}
                        raw_steps = step_overrides.get(str(getattr(benchmark, "name", "")), getattr(self.cfg, "r4_inner_steps_per_picard", 20))
                        picard_cfg.max_iters = max(1, int(raw_steps))
                        soft_pos = relax_soft_density_field(
                            hard_pos,
                            hard_sizes,
                            soft_pos,
                            soft_sizes,
                            psi,
                            geom,
                            picard_cfg,
                            soft_fixed=soft_fixed,
                            q_field=q_field,
                        )
                elif (
                    bool(r4_picard_soft_enabled)
                    and bool(getattr(two_fluid_cfg, "pde_enabled", False))
                    and num_soft > 0
                    and (time.time() - start_time) <= float(getattr(self.cfg, "r4_picard_max_budget_frac", 0.70)) * float(self.cfg.time_budget_sec)
                ):
                    self._trace_purity_event("r4_picard_soft_equilibrium_calls")
                    picard_cfg = TwoFluidConfig(**two_fluid_cfg.__dict__)
                    step_overrides = getattr(self.cfg, "r4_inner_steps_by_benchmark", {}) or {}
                    raw_steps = step_overrides.get(str(getattr(benchmark, "name", "")), getattr(self.cfg, "r4_inner_steps_per_picard", 20))
                    picard_cfg.max_iters = max(1, int(raw_steps))
                    soft_pos = relax_soft_density_field(
                        hard_pos,
                        hard_sizes,
                        soft_pos,
                        soft_sizes,
                        psi,
                        geom,
                        picard_cfg,
                        soft_fixed=soft_fixed,
                        q_field=q_field,
                    )
                else:
                    soft_pos = soft_macro_drift_step(
                        soft_pos, psi, geom, two_fluid_cfg, soft_fixed=soft_fixed
                    )

            # 10) Project both to canvas.
            hard_pos = project_to_canvas(
                hard_pos, hard_sizes, geom.canvas_width, geom.canvas_height,
                fixed_mask=hard_fixed, fixed_positions=hard_fixed_pos,
            )
            if num_soft > 0:
                soft_pos = project_to_canvas(
                    soft_pos, soft_sizes, geom.canvas_width, geom.canvas_height,
                    fixed_mask=soft_fixed, fixed_positions=soft_fixed_pos,
                )

            # 11) Update coils with new positions (currents stay).
            coils = coils.update_positions(hard_pos)

        # Final assembly: hard followed by soft.
        final_placement = torch.zeros_like(placement)
        final_placement[:num_hard] = hard_pos.cpu()
        if num_soft > 0:
            final_placement[num_hard:] = soft_pos.cpu()

        # Restore fixed positions exactly.
        if bool(fixed_all.any()):
            final_placement[fixed_all.cpu()] = benchmark.macro_positions[fixed_all.cpu()].float()

        # Project once more, then legalize.
        final_placement = project_to_canvas(
            final_placement,
            benchmark.macro_sizes.float(),
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
            fixed_mask=benchmark.macro_fixed,
            fixed_positions=benchmark.macro_positions.float(),
        )

        final_placement = self._legalize_candidate(
            final_placement,
            benchmark,
            num_hard,
        )

        if (
            bool(getattr(self.cfg, "plasma_purity_mode", False))
            and int(self._purity_trace.get("r4_two_fluid_calls", 0)) == 0
            and int(self._purity_trace.get("r5_pic_calls", 0)) == 0
            and bool(getattr(self.cfg, "two_fluid_pde_enabled", False))
            and int(benchmark.num_soft_macros) > 0
        ):
            final_placement = self._optimize_soft_two_fluid_pde(
                final_placement,
                benchmark,
                psi,
                geom,
                two_fluid_cfg,
                q_field=q_field,
            )
            final_placement = self._legalize_candidate(
                final_placement,
                benchmark,
                num_hard,
            )

        if compute_proxy_cost is not None and best_cost is None:
            final_scored_for_gate, final_cost_for_gate = self._evaluate_exact_candidate(
                final_placement,
                benchmark,
                self._plc_for_eval(benchmark),
                num_hard,
            )
            if final_scored_for_gate is not None and final_cost_for_gate is not None:
                best_placement = final_scored_for_gate
                best_cost = float(final_cost_for_gate)

        # SAFETY NET: compare to baseline; return whichever is cheaper.
        # Without this, a divergent equilibrium phase silently degrades us
        # below the legalize-only fallback.
        if compute_proxy_cost is not None and best_cost is not None:
            final_scored, final_cost = self._evaluate_exact_candidate(
                final_placement,
                benchmark,
                self._plc_for_eval(benchmark),
                num_hard,
            )
            if (
                final_scored is not None
                and final_cost is not None
                and float(final_cost) < float(best_cost)
            ):
                best_placement = final_scored
                best_cost = float(final_cost)

            if (
                bool(getattr(pic_cfg, "enabled", False))
                and bool(getattr(self.cfg, "r5_pic_exact_gate_enabled", False))
                and int(benchmark.num_soft_macros) > 0
                and (time.time() - start_time) <= float(self.cfg.soft_tilos_max_budget_frac) * float(self.cfg.time_budget_sec)
            ):
                try:
                    proposal_base = best_placement.clone()
                    proposal_base_cost = float(best_cost)
                    best_pic_candidate: Optional[torch.Tensor] = None
                    best_pic_cost = proposal_base_cost
                    raw_variants = []
                    if bool(getattr(self.cfg, "r5_pic_variant_portfolio_enabled", False)):
                        raw_variants = list(getattr(self.cfg, "r5_pic_variant_portfolio", []) or [])
                    if bool(getattr(self.cfg, "r5_pic_dimensionless_portfolio_enabled", False)):
                        self._trace_purity_event("r5_pic_dimensionless_portfolio_calls")
                        dim_variants = self._dimensionless_pic_variants(benchmark, all_edge_weight)
                        dim_max = max(0, int(getattr(self.cfg, "r5_pic_dimensionless_portfolio_max_trials", 4)))
                        raw_variants.extend(dim_variants[:dim_max])
                    max_trials = max(
                        1,
                        max(
                            int(getattr(self.cfg, "r5_pic_variant_portfolio_max_trials", 4)),
                            1 + int(getattr(self.cfg, "r5_pic_dimensionless_portfolio_max_trials", 4)),
                        ),
                    )
                    variant_specs = [{}] + raw_variants[: max(0, max_trials - 1)]
                    plc_pic = self._plc_for_eval(benchmark)
                    for variant_idx, variant_spec in enumerate(variant_specs):
                        if (time.time() - start_time) > float(self.cfg.soft_tilos_max_budget_frac) * float(self.cfg.time_budget_sec):
                            break
                        if variant_idx > 0:
                            self._trace_purity_event("r5_pic_variant_trials")
                        variant_cfg = self._pic_cfg_with_overrides(pic_cfg, variant_spec)
                        pic_candidate = self._optimize_soft_pic(
                            proposal_base,
                            benchmark,
                            geom,
                            variant_cfg,
                            all_edge_index,
                            all_edge_weight,
                        )
                        pic_candidate, pic_cost = self._evaluate_soft_tilos_candidate(
                            pic_candidate,
                            benchmark,
                            plc_pic,
                        )
                        if (
                            pic_candidate is not None
                            and pic_cost is not None
                            and float(pic_cost) < best_pic_cost - 1e-6
                        ):
                            best_pic_candidate = pic_candidate
                            best_pic_cost = float(pic_cost)
                    if best_pic_candidate is not None and best_pic_cost < proposal_base_cost - 1e-6:
                        self._trace_purity_event("r5_pic_exact_gate_accepts")
                        best_placement = best_pic_candidate
                        best_cost = float(best_pic_cost)
                    else:
                        self._trace_purity_event("r5_pic_exact_gate_rejects")
                except Exception:
                    pass

            if (
                bool(getattr(self.cfg, "two_fluid_pde_enabled", False))
                and not bool(getattr(pic_cfg, "enabled", False))
                and int(benchmark.num_soft_macros) > 0
                and (time.time() - start_time) <= float(self.cfg.soft_tilos_max_budget_frac) * float(self.cfg.time_budget_sec)
            ):
                try:
                    plc_soft = self._plc_for_eval(benchmark)
                    r4_candidate = self._optimize_soft_two_fluid_pde(
                        best_placement,
                        benchmark,
                        psi,
                        geom,
                        two_fluid_cfg,
                        q_field=q_field,
                    )
                    r4_candidate, r4_cost = self._evaluate_soft_tilos_candidate(
                        r4_candidate,
                        benchmark,
                        plc_soft,
                    )
                    if (
                        r4_candidate is not None
                        and r4_cost is not None
                        and float(r4_cost) < float(best_cost) - 1e-6
                    ):
                        best_placement = r4_candidate
                        best_cost = float(r4_cost)
                except Exception:
                    pass

            if (
                bool(self.cfg.soft_tilos_enabled)
                and not bool(getattr(self.cfg, "two_fluid_replace_tilos_soft", False))
                and int(benchmark.num_soft_macros) > 0
                and (time.time() - start_time) <= float(self.cfg.soft_tilos_max_budget_frac) * float(self.cfg.time_budget_sec)
            ):
                try:
                    plc_soft = self._plc_for_eval(benchmark)
                    if bool(self.cfg.soft_tilos_portfolio_enabled):
                        base_soft_placement = best_placement.clone()
                        for raw_steps in self.cfg.soft_tilos_portfolio_steps:
                            if (time.time() - start_time) > float(self.cfg.soft_tilos_max_budget_frac) * float(self.cfg.time_budget_sec):
                                break
                            steps = [int(s) for s in raw_steps]
                            soft_candidate = self._optimize_soft_macros(
                                base_soft_placement,
                                benchmark,
                                plc_soft,
                                steps_override=steps,
                            )
                            soft_candidate, soft_cost = self._evaluate_soft_tilos_candidate(
                                soft_candidate,
                                benchmark,
                                plc_soft,
                            )
                            if (
                                soft_candidate is not None
                                and soft_cost is not None
                                and float(soft_cost) < float(best_cost) - 1e-6
                            ):
                                best_placement = soft_candidate
                                best_cost = float(soft_cost)
                    elif bool(self.cfg.soft_tilos_probe_enabled):
                        probe_steps, refine_steps = self._soft_tilos_steps_for_benchmark(benchmark)
                        probe = self._optimize_soft_macros(
                            best_placement,
                            benchmark,
                            plc_soft,
                            steps_override=probe_steps,
                        )
                        probe_candidate, probe_cost = self._evaluate_soft_tilos_candidate(
                            probe,
                            benchmark,
                            plc_soft,
                        )
                        if probe_candidate is not None and probe_cost is not None:
                            improvement = float(best_cost) - float(probe_cost)
                            if bool(self.cfg.diagnostics_enabled):
                                print(
                                    f"[team_plasma][soft] bench={benchmark.name} "
                                    f"probe_cost={float(probe_cost):.6f} "
                                    f"improvement={improvement:.6f}",
                                    flush=True,
                                )
                            if improvement > 1e-6:
                                best_placement = probe_candidate
                                best_cost = float(probe_cost)

                            if (
                                improvement >= float(self.cfg.soft_tilos_probe_min_delta)
                                and (time.time() - start_time) <= float(self.cfg.soft_tilos_max_budget_frac) * float(self.cfg.time_budget_sec)
                            ):
                                refined = self._optimize_soft_macros(
                                    best_placement,
                                    benchmark,
                                    plc_soft,
                                    steps_override=refine_steps,
                                )
                                refined_candidate, refined_cost = self._evaluate_soft_tilos_candidate(
                                    refined,
                                    benchmark,
                                    plc_soft,
                                )
                                if (
                                    refined_candidate is not None
                                    and refined_cost is not None
                                    and float(refined_cost) < float(best_cost) - 1e-6
                                ):
                                    if bool(self.cfg.diagnostics_enabled):
                                        print(
                                            f"[team_plasma][soft] bench={benchmark.name} "
                                            f"refine_cost={float(refined_cost):.6f} "
                                            f"improvement={float(best_cost) - float(refined_cost):.6f}",
                                            flush=True,
                                        )
                                    best_placement = refined_candidate
                                    best_cost = float(refined_cost)
                    else:
                        soft_candidate = self._optimize_soft_macros(
                            best_placement,
                            benchmark,
                            plc_soft,
                        )
                        soft_candidate, soft_cost = self._evaluate_soft_tilos_candidate(
                            soft_candidate,
                            benchmark,
                            plc_soft,
                        )
                        if (
                            soft_candidate is not None
                            and soft_cost is not None
                            and float(soft_cost) < float(best_cost) - 1e-6
                        ):
                            best_placement = soft_candidate
                            best_cost = float(soft_cost)
                except Exception:
                    pass

            best_placement, best_cost = self._run_snowflake_divertor_split(
                best_placement,
                benchmark,
                plc,
                edge_index,
                edge_weight,
                float(best_cost),
                start_time,
                geom,
                smooth_proxy_cfg,
            )

            if (
                bool(self.cfg.exact_hard_refine_enabled)
                and (
                    (time.time() - start_time) <= float(self.cfg.exact_hard_refine_max_budget_frac) * float(self.cfg.time_budget_sec)
                    or bool(
                        (getattr(self.cfg, "exact_hard_refine_force_over_budget_by_benchmark", {}) or {}).get(
                            str(getattr(benchmark, "name", "")),
                            False,
                        )
                    )
                )
            ):
                try:
                    refine_priority = None
                    flux_directions = None
                    smooth_priority = None
                    smooth_directions = None
                    selector = str(getattr(self.cfg, "exact_hard_refine_selector", "degree")).lower()
                    if selector in {"mercier", "mercier_degree", "plasma"} and "profiles" in locals():
                        mercier_field = mercier_diagnostic(psi, profiles, stability_cfg)
                        hard_points = best_placement[:num_hard].to(device)
                        refine_priority = bilinear_sample(mercier_field, hard_points, geom)
                    direction_overrides = getattr(self.cfg, "exact_hard_refine_direction_set_by_benchmark", {}) or {}
                    direction_set = str(
                        direction_overrides.get(
                            str(getattr(benchmark, "name", "")),
                            getattr(self.cfg, "exact_hard_refine_direction_set", "axis"),
                        )
                    ).lower()
                    if direction_set in {
                        "axis_flux",
                        "flux",
                        "axis+flux",
                        "axis_diag_flux",
                        "flux_only",
                        "plasma",
                        "plasma_only",
                        "flux_tangent",
                        "flux_tangent_only",
                        "tangent",
                        "tangent_only",
                        "flux_normal",
                        "flux_normal_only",
                        "normal",
                        "normal_only",
                    }:
                        gR, gZ = gradient_of_psi(psi, geom)
                        hard_points = best_placement[:num_hard].to(device)
                        flux_x = bilinear_sample(gR, hard_points, geom)
                        flux_y = bilinear_sample(gZ, hard_points, geom)
                        flux_directions = torch.stack([flux_x, flux_y], dim=1)
                    edge_weight_refine = self._eccd_reweight_edges(
                        best_placement,
                        benchmark,
                        edge_index,
                        edge_weight,
                        geom,
                        smooth_proxy_cfg,
                    )
                    if bool(smooth_proxy_cfg.enabled):
                        all_points = best_placement.to(device)
                        all_sizes_for_smooth = benchmark.macro_sizes.float().to(device)
                        smooth_priority_all, smooth_directions_all = sample_smooth_hotspot_guidance(
                            all_points,
                            all_sizes_for_smooth,
                            edge_index,
                            edge_weight_refine,
                            geom,
                            smooth_proxy_cfg,
                            fixed_mask=benchmark.macro_fixed.to(device),
                        )
                        smooth_priority = smooth_priority_all[:num_hard]
                        smooth_directions = smooth_directions_all[:num_hard]
                    curvature_weight = max(0.0, float(getattr(self.cfg, "smooth_proxy_flux_curvature_weight", 0.0)))
                    if curvature_weight > 0.0:
                        curvature = torch.abs(delta_star_apply(psi, geom))
                        curvature = curvature / torch.clamp(curvature.mean(), min=1e-6)
                        frac = min(0.50, max(0.01, float(getattr(self.cfg, "smooth_proxy_flux_curvature_top_frac", 0.10))))
                        threshold = torch.quantile(curvature.reshape(-1), max(0.0, min(0.99, 1.0 - frac)))
                        tau = max(1e-4, float(getattr(self.cfg, "smooth_proxy_lse_tau", 0.10)))
                        curvature_tail = tau * torch.nn.functional.softplus((curvature - threshold) / tau)
                        cgx, cgy = gradient_of_psi(curvature_tail, geom)
                        hard_points = best_placement[:num_hard].to(device)
                        curvature_priority = bilinear_sample(curvature_tail, hard_points, geom)
                        curvature_dirs = torch.stack(
                            [-bilinear_sample(cgx, hard_points, geom), -bilinear_sample(cgy, hard_points, geom)],
                            dim=1,
                        )
                        c_norm = torch.linalg.norm(curvature_dirs, dim=1, keepdim=True).clamp_min(1e-12)
                        curvature_dirs = curvature_dirs / c_norm
                        if smooth_priority is None:
                            smooth_priority = curvature_weight * curvature_priority
                            smooth_directions = curvature_dirs
                        else:
                            smooth_priority = smooth_priority + curvature_weight * curvature_priority
                            s_norm = torch.linalg.norm(smooth_directions, dim=1, keepdim=True).clamp_min(1e-12)
                            smooth_unit = smooth_directions / s_norm
                            smooth_directions = smooth_unit + curvature_weight * curvature_dirs
                            smooth_directions = smooth_directions / torch.linalg.norm(smooth_directions, dim=1, keepdim=True).clamp_min(1e-12)
                    if smooth_priority is not None and smooth_directions is not None and bool(smooth_proxy_cfg.enabled):
                        best_placement, best_cost = self._run_smooth_proxy_global_step(
                            best_placement,
                            benchmark,
                            self._plc_for_eval(benchmark),
                            float(best_cost),
                            start_time,
                            smooth_priority=smooth_priority,
                            smooth_directions=smooth_directions,
                        )
                    best_placement, best_cost = self._run_eigenmode_refine(
                        best_placement,
                        benchmark,
                        self._plc_for_eval(benchmark),
                        edge_index,
                        edge_weight_refine,
                        float(best_cost),
                        start_time,
                        smooth_priority=smooth_priority,
                        smooth_directions=smooth_directions,
                    )
                    best_placement, best_cost = self._run_tearing_mode_refine(
                        best_placement,
                        benchmark,
                        self._plc_for_eval(benchmark),
                        edge_index,
                        edge_weight_refine,
                        float(best_cost),
                        start_time,
                    )
                    best_placement, best_cost = self._run_exact_hard_refine(
                        best_placement,
                        benchmark,
                        self._plc_for_eval(benchmark),
                        edge_index,
                        edge_weight_refine,
                        float(best_cost),
                        start_time,
                        refine_priority=refine_priority,
                        flux_directions=flux_directions,
                        smooth_priority=smooth_priority,
                        smooth_directions=smooth_directions,
                        geom=geom,
                        smooth_proxy_cfg=smooth_proxy_cfg,
                    )
                    best_placement, best_cost = self._run_flux_rope_cluster_relocation(
                        best_placement,
                        benchmark,
                        self._plc_for_eval(benchmark),
                        edge_index,
                        edge_weight_refine,
                        float(best_cost),
                        start_time,
                    )
                    best_placement, best_cost = self._run_thermal_hopping(
                        best_placement,
                        benchmark,
                        self._plc_for_eval(benchmark),
                        float(best_cost),
                        start_time,
                        edge_index,
                        edge_weight_refine,
                        refine_priority=refine_priority,
                    )
                except Exception:
                    pass
            self._assert_plasma_purity_contract(best_placement, benchmark, num_hard)
            return best_placement

        self._assert_plasma_purity_contract(final_placement, benchmark, num_hard)
        return final_placement
