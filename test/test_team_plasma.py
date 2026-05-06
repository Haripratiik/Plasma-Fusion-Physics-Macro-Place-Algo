"""Tests for TeamPlasmaPlacer core invariants."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


def _load_team_module():
    path = Path("submissions/team_plasma/placer.py").resolve()
    spec = importlib.util.spec_from_file_location("team_plasma_placer_test", str(path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_core_module():
    path = Path("submissions/team_plasma/plasma_core.py").resolve()
    spec = importlib.util.spec_from_file_location("team_plasma_core_test", str(path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _dummy_benchmark(mod):
    Benchmark = mod.Benchmark
    macro_positions = torch.tensor(
        [
            [1.0, 1.0],
            [1.2, 1.1],
            [2.8, 2.8],
            [3.0, 3.1],
        ],
        dtype=torch.float32,
    )
    macro_sizes = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    macro_fixed = torch.tensor([False, False, False, False], dtype=torch.bool)
    return Benchmark(
        name="dummy",
        canvas_width=5.0,
        canvas_height=5.0,
        num_macros=4,
        num_hard_macros=4,
        num_soft_macros=0,
        macro_positions=macro_positions,
        macro_sizes=macro_sizes,
        macro_fixed=macro_fixed,
        macro_names=["m0", "m1", "m2", "m3"],
        num_nets=0,
        net_nodes=[],
        net_weights=torch.zeros(0),
        grid_rows=8,
        grid_cols=8,
        hard_macro_indices=[],
        soft_macro_indices=[],
    )


class _MockPlcModule:
    def __init__(
        self,
        name: str,
        pos=(0.0, 0.0),
        module_type: str = "MACRO",
        macro_name: str | None = None,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
    ):
        self._name = name
        self._pos = pos
        self._type = module_type
        self._macro_name = macro_name
        self.x_offset = float(x_offset)
        self.y_offset = float(y_offset)

    def get_name(self):
        return self._name

    def get_pos(self):
        return self._pos

    def get_type(self):
        return self._type

    def get_macro_name(self):
        return self._macro_name


class _MockPlc:
    def __init__(self, module_names, nets, extra_modules=None):
        self.modules_w_pins = [_MockPlcModule(name) for name in module_names]
        if extra_modules:
            self.modules_w_pins.extend(extra_modules)
        self.nets = nets
        self.port_indices = []


def test_projection_clamps_bounds():
    mod = _load_team_module()

    pos = torch.tensor([[0.1, 0.1], [9.0, 9.0]], dtype=torch.float32)
    sizes = torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float32)
    projected = mod.project_to_canvas(pos, sizes, canvas_width=6.0, canvas_height=6.0)

    assert torch.all(projected[:, 0] >= 1.0)
    assert torch.all(projected[:, 0] <= 5.0)
    assert torch.all(projected[:, 1] >= 1.0)
    assert torch.all(projected[:, 1] <= 5.0)


def test_channel_permeability_map_prefers_cross_channel_suppression():
    core = _load_core_module()

    positions = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 2.0],
        ],
        dtype=torch.float32,
    )
    sizes = torch.tensor(
        [
            [1.0, 1.2],
            [1.0, 1.2],
        ],
        dtype=torch.float32,
    )
    edge_profile = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    bx, by = core.build_channel_permeability_maps(
        positions,
        sizes,
        edge_profile,
        {
            "channel_span_frac": 0.5,
            "sigma_frac": 0.2,
            "min_overlap_frac": 0.1,
            "inv_gap_power": 0.5,
            "uniform_edge_bias": 0.0,
            "normalize_quantile": 0.9,
        },
        canvas_width=5.0,
        canvas_height=5.0,
        rows=16,
        cols=16,
    )

    assert float(torch.max(bx).item()) > 0.0
    assert float(torch.max(by).item()) == 0.0


def test_source_quench_attenuates_hot_regions_more_than_cool_regions():
    core = _load_core_module()

    n_src = torch.tensor([[1.0, -1.0], [0.5, -0.5]], dtype=torch.float32)
    q_over = torch.tensor([[2.0, 0.2], [0.1, 0.1]], dtype=torch.float32)
    bg = torch.tensor([[1.5, 0.1], [0.0, 0.0]], dtype=torch.float32)
    pin = torch.zeros_like(q_over)
    bx = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    by = torch.zeros_like(bx)

    quenched = core.apply_source_quench(
        n_src,
        q_over,
        bg,
        pin,
        None,
        bx,
        by,
        {
            "enabled": True,
            "q_alpha": 0.5,
            "bg_alpha": 0.3,
            "channel_alpha": 0.4,
            "min_scale": 0.1,
        },
    )

    assert float(torch.abs(quenched[0, 0]).item()) < float(torch.abs(n_src[0, 0]).item())
    assert float(torch.abs(quenched[1, 1]).item()) <= float(torch.abs(n_src[1, 1]).item())
    assert float(torch.abs(quenched[0, 0]).item()) < float(torch.abs(quenched[1, 1]).item())


def test_porosity_pressure_rises_in_low_whitespace_neighborhoods():
    core = _load_core_module()

    hard_density = torch.zeros((9, 9), dtype=torch.float32)
    hard_density[3:6, 3:6] = 0.9
    bg_density = torch.zeros_like(hard_density)
    bg_density[2:7, 2:7] = 0.35

    pressure = core.build_porosity_pressure(
        hard_density,
        bg_density,
        {
            "enabled": True,
            "hard_scale": 1.0,
            "bg_scale": 0.5,
            "window_cells": 5,
            "target_occupancy": 0.45,
            "overflow_power": 1.0,
            "norm_mode": "mean",
        },
    )

    assert float(pressure[4, 4].item()) > float(pressure[0, 0].item())
    assert float(torch.max(pressure).item()) > 0.0


def test_bundle_net_source_degree_boost_emphasizes_high_degree_bundles():
    core = _load_core_module()

    positions = torch.tensor(
        [
            [1.0, 2.0],
            [2.0, 2.0],
            [7.0, 2.0],
            [8.0, 2.0],
            [7.5, 3.0],
            [8.5, 3.0],
        ],
        dtype=torch.float32,
    )
    bundle_ptr = torch.tensor([0, 2, 6], dtype=torch.long)
    bundle_index = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    bundle_offsets = torch.zeros((6, 2), dtype=torch.float32)
    bundle_weight = torch.ones(2, dtype=torch.float32)

    src_base = core.build_bundle_net_source(
        positions,
        bundle_ptr,
        bundle_index,
        bundle_offsets,
        bundle_weight,
        canvas_width=10.0,
        canvas_height=5.0,
        rows=20,
        cols=20,
        degree_boost=0.0,
    )
    src_boost = core.build_bundle_net_source(
        positions,
        bundle_ptr,
        bundle_index,
        bundle_offsets,
        bundle_weight,
        canvas_width=10.0,
        canvas_height=5.0,
        rows=20,
        cols=20,
        degree_boost=1.0,
    )

    left_ratio_base = float(src_base[:, 0:8].abs().sum().item()) / max(float(src_base[:, 12:20].abs().sum().item()), 1e-6)
    left_ratio_boost = float(src_boost[:, 0:8].abs().sum().item()) / max(float(src_boost[:, 12:20].abs().sum().item()), 1e-6)

    assert left_ratio_boost < left_ratio_base


def test_overlap_detector_adversarial():
    mod = _load_team_module()
    b = _dummy_benchmark(mod)

    overlaps = mod.count_hard_overlaps(
        b.macro_positions,
        b.macro_sizes,
        num_hard_macros=b.num_hard_macros,
    )
    assert overlaps > 0


def test_legalizer_terminates_with_zero_overlap():
    mod = _load_team_module()
    b = _dummy_benchmark(mod)

    legalized = mod.legalize_hard_macros(
        b.macro_positions,
        b.macro_sizes,
        b.macro_fixed,
        b.canvas_width,
        b.canvas_height,
        b.num_hard_macros,
        max_iters=100,
        fallback_iters=80,
    )

    overlaps = mod.count_hard_overlaps(
        legalized,
        b.macro_sizes,
        num_hard_macros=b.num_hard_macros,
    )
    assert overlaps == 0


def test_pde_residual_nonincreasing_or_fallback():
    core = _load_core_module()

    rhs = torch.zeros(18, 18, dtype=torch.float32)
    rhs[6:12, 6:12] = 1.0
    kappa = torch.ones_like(rhs)

    pde_cfg = {
        "picard_outer": 6,
        "gs_iters": 24,
        "omega": 1.1,
        "damping": 0.7,
        "eta": 0.2,
        "kappa_psi": 0.35,
        "psi_nonlinear_scale": 0.8,
    }
    _, residuals = core.solve_gs_plasma_pde(
        rhs=rhs,
        kappa_base=kappa,
        canvas_width=10.0,
        canvas_height=10.0,
        pde_cfg=pde_cfg,
        psi_init=None,
    )

    assert len(residuals) >= 1
    for i in range(1, len(residuals)):
        assert residuals[i] <= residuals[i - 1] + 1e-9


def test_pde_neumann_boundary_matches_adjacent_cells():
    core = _load_core_module()

    rhs = torch.zeros(18, 18, dtype=torch.float32)
    rhs[5:13, 5:13] = 1.0
    kappa = torch.ones_like(rhs)

    pde_cfg = {
        "picard_outer": 4,
        "gs_iters": 20,
        "omega": 1.1,
        "damping": 0.7,
        "eta": 0.2,
        "kappa_psi": 0.35,
        "psi_nonlinear_scale": 0.8,
        "boundary_mode": "neumann",
    }
    psi, _ = core.solve_gs_plasma_pde(
        rhs=rhs,
        kappa_base=kappa,
        canvas_width=10.0,
        canvas_height=10.0,
        pde_cfg=pde_cfg,
        psi_init=None,
    )

    assert torch.allclose(psi[0, :], psi[1, :], atol=1e-5)
    assert torch.allclose(psi[-1, :], psi[-2, :], atol=1e-5)
    assert torch.allclose(psi[:, 0], psi[:, 1], atol=1e-5)
    assert torch.allclose(psi[:, -1], psi[:, -2], atol=1e-5)


def test_overlap_repulsion_supports_pressure_inflation_buffer():
    core = _load_core_module()

    positions = torch.tensor(
        [
            [1.0, 1.0],
            [2.30, 1.0],
        ],
        dtype=torch.float32,
    )
    sizes = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    base_force = core.compute_overlap_repulsion(positions, sizes, gap=1e-4)
    assert torch.allclose(base_force, torch.zeros_like(base_force))

    inflated_force = core.compute_overlap_repulsion(
        positions,
        sizes,
        gap=1e-4,
        size_inflation=torch.tensor([1.4, 1.4], dtype=torch.float32),
    )
    assert torch.linalg.norm(inflated_force, dim=1).max().item() > 0.0


def test_bundle_extractor_supports_degree_weight_normalization():
    mod = _load_team_module()
    Benchmark = mod.Benchmark
    benchmark = Benchmark(
        name="bundle_dummy",
        canvas_width=10.0,
        canvas_height=10.0,
        num_macros=3,
        num_hard_macros=3,
        num_soft_macros=0,
        macro_positions=torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]], dtype=torch.float32),
        macro_sizes=torch.ones((3, 2), dtype=torch.float32),
        macro_fixed=torch.zeros(3, dtype=torch.bool),
        macro_names=["m0", "m1", "m2"],
        num_nets=1,
        net_nodes=[],
        net_weights=torch.ones(1, dtype=torch.float32),
        grid_rows=8,
        grid_cols=8,
        hard_macro_indices=[0, 1, 2],
        soft_macro_indices=[],
    )
    plc = _MockPlc(
        ["m0", "m1", "m2"],
        {"m0/p0": {"m1/p0", "m2/p0"}},
    )

    _, _, _, inverse_w = mod.extract_hard_net_bundles_from_plc(
        benchmark,
        plc,
        max_nets=8,
        min_degree=3,
        weight_mode="inverse_degree",
    )
    _, _, _, unit_w = mod.extract_hard_net_bundles_from_plc(
        benchmark,
        plc,
        max_nets=8,
        min_degree=3,
        weight_mode="unit",
    )

    assert inverse_w.numel() == 1
    assert abs(float(inverse_w[0].item()) - 0.5) < 1e-6
    assert abs(float(unit_w[0].item()) - 1.0) < 1e-6


def test_transport_barrier_suppresses_normal_more_than_tangential():
    mod = _load_team_module()

    transport_force = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    plasma_force = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
    q_samples = torch.tensor([3.0], dtype=torch.float32)
    rho_samples = torch.tensor([2.0], dtype=torch.float32)
    cfg = {
        "enabled": True,
        "q_quantile": 0.5,
        "rho_quantile": 0.5,
        "q_scale_frac": 0.1,
        "rho_scale_frac": 0.1,
        "q_weight": 1.0,
        "rho_weight": 0.0,
        "grad_quantile": 0.5,
        "grad_scale_frac": 0.1,
        "normal_alpha": 4.0,
        "tangent_beta": 0.5,
        "min_normal_scale": 0.1,
        "max_tangent_scale": 2.0,
    }

    out = mod.apply_transport_barrier(
        transport_force,
        plasma_force,
        q_samples,
        rho_samples,
        cfg,
    )

    assert float(out[0, 0].item()) < 1.0
    assert float(out[0, 1].item()) >= 1.0


def test_bundle_net_source_is_nonzero_for_multiterm_bundle():
    mod = _load_team_module()
    core = _load_core_module()
    Benchmark = mod.Benchmark
    benchmark = Benchmark(
        name="bundle_source_dummy",
        canvas_width=10.0,
        canvas_height=10.0,
        num_macros=3,
        num_hard_macros=3,
        num_soft_macros=0,
        macro_positions=torch.tensor([[1.0, 1.0], [4.0, 4.0], [7.0, 1.5]], dtype=torch.float32),
        macro_sizes=torch.ones((3, 2), dtype=torch.float32),
        macro_fixed=torch.zeros(3, dtype=torch.bool),
        macro_names=["m0", "m1", "m2"],
        num_nets=1,
        net_nodes=[],
        net_weights=torch.ones(1, dtype=torch.float32),
        grid_rows=8,
        grid_cols=8,
        hard_macro_indices=[0, 1, 2],
        soft_macro_indices=[],
    )
    plc = _MockPlc(
        ["m0", "m1", "m2"],
        {"m0/p0": {"m1/p0", "m2/p0"}},
    )
    bundle_ptr, bundle_index, bundle_offsets, bundle_weight = mod.extract_hard_net_bundles_from_plc(
        benchmark,
        plc,
        max_nets=8,
        min_degree=3,
        weight_mode="inverse_degree",
    )

    source = core.build_bundle_net_source(
        benchmark.macro_positions.float(),
        bundle_ptr,
        bundle_index,
        bundle_offsets,
        bundle_weight,
        canvas_width=benchmark.canvas_width,
        canvas_height=benchmark.canvas_height,
        rows=8,
        cols=8,
    )
    assert source.shape == (8, 8)
    assert float(source.abs().sum().item()) > 0.0


def test_extract_hard_pin_pressure_terms_respects_degree_filter():
    mod = _load_team_module()
    Benchmark = mod.Benchmark
    benchmark = Benchmark(
        name="pin_pressure_terms",
        canvas_width=10.0,
        canvas_height=10.0,
        num_macros=4,
        num_hard_macros=4,
        num_soft_macros=0,
        macro_positions=torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [7.0, 7.0]], dtype=torch.float32),
        macro_sizes=torch.ones((4, 2), dtype=torch.float32),
        macro_fixed=torch.zeros(4, dtype=torch.bool),
        macro_names=["m0", "m1", "m2", "m3"],
        num_nets=0,
        net_nodes=[],
        net_weights=torch.zeros(0),
        grid_rows=8,
        grid_cols=8,
        hard_macro_indices=[0, 1, 2, 3],
        soft_macro_indices=[],
    )
    extra_modules = [
        _MockPlcModule("m0/p0", module_type="MACRO_PIN", macro_name="m0", x_offset=-0.4, y_offset=0.0),
        _MockPlcModule("m1/p0", module_type="MACRO_PIN", macro_name="m1", x_offset=0.4, y_offset=0.0),
        _MockPlcModule("m2/p0", module_type="MACRO_PIN", macro_name="m2", x_offset=0.0, y_offset=-0.4),
        _MockPlcModule("m3/p0", module_type="MACRO_PIN", macro_name="m3", x_offset=0.0, y_offset=0.4),
    ]
    plc = _MockPlc(
        ["m0", "m1", "m2", "m3"],
        {
            "m0/p0": {"m1/p0", "m2/p0"},
            "m2/p0": {"m3/p0"},
        },
        extra_modules=extra_modules,
    )

    idx_all, _, w_all = mod.extract_hard_pin_pressure_terms_from_plc(benchmark, plc)
    idx_filtered, _, w_filtered = mod.extract_hard_pin_pressure_terms_from_plc(benchmark, plc, max_degree=2)

    assert idx_all.numel() == 5
    assert idx_filtered.numel() == 2
    assert torch.equal(idx_filtered, torch.tensor([2, 3], dtype=torch.long))
    assert float(w_all[0].item()) < float(w_filtered[0].item())


def test_extract_hard_pin_edge_profile_tracks_pin_sides():
    mod = _load_team_module()
    Benchmark = mod.Benchmark
    benchmark = Benchmark(
        name="pin_edge_profile",
        canvas_width=10.0,
        canvas_height=10.0,
        num_macros=2,
        num_hard_macros=2,
        num_soft_macros=0,
        macro_positions=torch.tensor([[2.0, 2.0], [8.0, 2.0]], dtype=torch.float32),
        macro_sizes=torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float32),
        macro_fixed=torch.zeros(2, dtype=torch.bool),
        macro_names=["m0", "m1"],
        num_nets=0,
        net_nodes=[],
        net_weights=torch.zeros(0),
        grid_rows=8,
        grid_cols=8,
        hard_macro_indices=[0, 1],
        soft_macro_indices=[],
    )
    extra_modules = [
        _MockPlcModule("m0/p0", module_type="MACRO_PIN", macro_name="m0", x_offset=0.6, y_offset=0.0),
        _MockPlcModule("m1/p0", module_type="MACRO_PIN", macro_name="m1", x_offset=-0.6, y_offset=0.0),
    ]
    plc = _MockPlc(["m0", "m1"], {"m0/p0": {"m1/p0"}}, extra_modules=extra_modules)

    profile = mod.extract_hard_pin_edge_profile_from_plc(benchmark, plc)
    assert profile.shape == (2, 4)
    assert float(profile[0, 1].item()) > 0.0
    assert float(profile[1, 0].item()) > 0.0


def test_pin_edge_sheath_force_pushes_macros_apart_across_narrow_channel():
    mod = _load_team_module()
    positions = torch.tensor([[4.0, 5.0], [6.1, 5.0]], dtype=torch.float32)
    sizes = torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float32)
    edge_profile = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    force = mod.compute_pin_edge_sheath_force(
        positions,
        sizes,
        edge_profile,
        {
            "enabled": True,
            "channel_span_frac": 0.08,
            "sigma_frac": 0.03,
            "min_overlap_frac": 0.2,
            "inv_gap_power": 1.0,
            "edge_power": 1.0,
            "max_pair_pressure": 4.0,
        },
        canvas_width=10.0,
        canvas_height=10.0,
    )
    assert float(force[0, 0].item()) < 0.0
    assert float(force[1, 0].item()) > 0.0


def test_pin_edge_sheath_force_uniform_edge_bias_works_without_pin_profile():
    mod = _load_team_module()
    positions = torch.tensor([[4.0, 5.0], [6.1, 5.0]], dtype=torch.float32)
    sizes = torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float32)
    edge_profile = torch.zeros((2, 4), dtype=torch.float32)
    force = mod.compute_pin_edge_sheath_force(
        positions,
        sizes,
        edge_profile,
        {
            "enabled": True,
            "channel_span_frac": 0.08,
            "sigma_frac": 0.03,
            "min_overlap_frac": 0.2,
            "inv_gap_power": 1.0,
            "edge_power": 1.0,
            "uniform_edge_bias": 1.0,
            "max_pair_pressure": 4.0,
        },
        canvas_width=10.0,
        canvas_height=10.0,
    )
    assert float(force[0, 0].item()) < 0.0
    assert float(force[1, 0].item()) > 0.0


def test_pin_pressure_field_produces_nonzero_force():
    core = _load_core_module()

    positions = torch.tensor([[2.0, 2.0], [8.0, 8.0]], dtype=torch.float32)
    sizes = torch.ones((2, 2), dtype=torch.float32)
    edge_index = torch.zeros((0, 2), dtype=torch.long)
    edge_weight = torch.zeros(0, dtype=torch.float32)
    pin_terms = (
        torch.tensor([0, 0, 0], dtype=torch.long),
        torch.tensor([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]], dtype=torch.float32),
        torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
    )

    fields = core.compute_plasma_forces(
        positions=positions,
        sizes=sizes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        canvas_width=10.0,
        canvas_height=10.0,
        grid_size=12,
        pde_cfg={
            "pin_pressure_enabled": True,
            "pin_target": 0.4,
            "pin_softplus_beta": 6.0,
            "pin_smoothing": 1,
            "pin_norm_mode": "mean",
            "kappa_pin": 0.2,
        },
        rhs_weights={"rho": 0.0, "q": 0.0, "n": 0.0, "wall": 0.0, "pin": 1.0},
        pin_terms=pin_terms,
    )

    assert float(fields["pin_over"].max().item()) > 0.0
    assert float(torch.linalg.norm(fields["pin_force"], dim=1).max().item()) > 0.0


def test_porosity_pressure_field_produces_nonzero_force():
    core = _load_core_module()

    positions = torch.tensor([[3.0, 5.0], [5.0, 5.0], [7.0, 5.0]], dtype=torch.float32)
    sizes = torch.tensor([[1.8, 4.0], [1.8, 4.0], [1.8, 4.0]], dtype=torch.float32)
    edge_index = torch.zeros((0, 2), dtype=torch.long)
    edge_weight = torch.zeros(0, dtype=torch.float32)
    bg_pos = torch.tensor([[5.0, 5.0]], dtype=torch.float32)
    bg_sizes = torch.tensor([[2.5, 2.5]], dtype=torch.float32)

    fields = core.compute_plasma_forces(
        positions=positions,
        sizes=sizes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        canvas_width=10.0,
        canvas_height=10.0,
        grid_size=16,
        pde_cfg={
            "porosity": {
                "enabled": True,
                "hard_scale": 1.0,
                "bg_scale": 0.35,
                "window_cells": 5,
                "target_occupancy": 0.40,
                "overflow_power": 1.0,
                "norm_mode": "mean",
            },
            "kappa_porosity": 0.2,
        },
        rhs_weights={"rho": 0.0, "q": 0.0, "n": 0.0, "wall": 0.0, "porosity": 1.0},
        background_positions=bg_pos,
        background_sizes=bg_sizes,
    )

    assert float(fields["porosity_over"].max().item()) > 0.0
    assert float(torch.linalg.norm(fields["porosity_force"], dim=1).max().item()) > 0.0


def test_compute_plasma_forces_applies_channel_quench_to_net_source():
    core = _load_core_module()

    positions = torch.tensor([[1.0, 2.0], [3.0, 2.0]], dtype=torch.float32)
    sizes = torch.tensor([[1.0, 1.2], [1.0, 1.2]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1]], dtype=torch.long)
    edge_weight = torch.ones(1, dtype=torch.float32)
    edge_profile = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    common_pde = {
        "rhs_softplus_beta": 8.0,
        "wall_decay_frac": 0.12,
        "density_target": 0.9,
        "overflow_power": 1.2,
        "source_quench": {
            "enabled": True,
            "channel_alpha": 0.5,
            "channel_power": 1.0,
            "min_scale": 0.1,
        },
    }
    channel_cfg = {
        "enabled": True,
        "channel_span_frac": 0.5,
        "sigma_frac": 0.2,
        "min_overlap_frac": 0.1,
        "inv_gap_power": 0.5,
        "edge_power": 1.0,
        "uniform_edge_bias": 0.0,
        "max_pair_pressure": 4.0,
        "normalize_quantile": 0.9,
    }

    captured = {}
    original = core.apply_source_quench

    def _capture(n_src, q_over, bg_density, pin_over, porosity_over, bx, by, cfg):
        captured["bx"] = None if bx is None else bx.clone()
        captured["by"] = None if by is None else by.clone()
        return original(n_src, q_over, bg_density, pin_over, porosity_over, bx, by, cfg)

    core.apply_source_quench = _capture
    try:
        core.compute_plasma_forces(
            positions=positions,
            sizes=sizes,
            edge_index=edge_index,
            edge_weight=edge_weight,
            canvas_width=5.0,
            canvas_height=5.0,
            grid_size=16,
            pde_cfg={**common_pde, "channel_kappa": channel_cfg},
            rhs_weights={"rho": 0.0, "q": 0.0, "n": 1.0, "wall": 0.0},
            pin_edge_profile=edge_profile,
        )
    finally:
        core.apply_source_quench = original

    assert captured["bx"] is not None
    assert captured["by"] is not None
    assert float(torch.max(captured["bx"]).item()) > 0.0
    assert float(torch.max(captured["by"]).item()) == 0.0


def test_compute_plasma_forces_without_channel_quench_passes_no_barriers():
    core = _load_core_module()

    positions = torch.tensor([[1.0, 2.0], [3.0, 2.0]], dtype=torch.float32)
    sizes = torch.tensor([[1.0, 1.2], [1.0, 1.2]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1]], dtype=torch.long)
    edge_weight = torch.ones(1, dtype=torch.float32)

    common_pde = {
        "rhs_softplus_beta": 8.0,
        "wall_decay_frac": 0.12,
        "density_target": 0.9,
        "overflow_power": 1.2,
        "source_quench": {
            "enabled": True,
            "channel_alpha": 0.5,
            "channel_power": 1.0,
            "min_scale": 0.1,
        },
    }

    captured = {}
    original = core.apply_source_quench

    def _capture(n_src, q_over, bg_density, pin_over, porosity_over, bx, by, cfg):
        captured["bx"] = bx
        captured["by"] = by
        return original(n_src, q_over, bg_density, pin_over, porosity_over, bx, by, cfg)

    core.apply_source_quench = _capture
    try:
        core.compute_plasma_forces(
            positions=positions,
            sizes=sizes,
            edge_index=edge_index,
            edge_weight=edge_weight,
            canvas_width=5.0,
            canvas_height=5.0,
            grid_size=16,
            pde_cfg=dict(common_pde),
            rhs_weights={"rho": 0.0, "q": 0.0, "n": 1.0, "wall": 0.0},
        )
    finally:
        core.apply_source_quench = original

    assert captured["bx"] is None
    assert captured["by"] is None


def test_router_plasma_probe_emits_pressure_features():
    mod = _load_team_module()
    b = _dummy_benchmark(mod)

    overrides = {
        "router": {
            "plasma_probe": {
                "enabled": True,
                "use_plc": False,
                "use_anchors": False,
                "include_soft_background": False,
                "grid_size": 12,
                "knn_edges": {"enabled": True, "k": 2, "weight": 1.0},
            }
        }
    }
    placer = mod.TeamPlasmaPlacer(config_overrides=overrides, seed=123)
    feat = placer._benchmark_router_features(b)

    for key in (
        "probe_q_p90",
        "probe_q_p99",
        "probe_rho_p90",
        "probe_q_rho_ratio",
        "probe_psi_abs_mean",
    ):
        assert key in feat
        assert float(feat[key]) >= 0.0


def test_multi_start_source_modes_preserve_current_candidate():
    mod = _load_team_module()
    b = _dummy_benchmark(mod)
    placer = mod.TeamPlasmaPlacer(seed=123)

    current = b.macro_positions.clone()
    current[0] = torch.tensor([4.0, 4.0], dtype=torch.float32)

    current_only = placer._resolve_multi_start_bases(b, current, {"source": "current"})
    assert len(current_only) == 1
    assert torch.allclose(current_only[0], current)

    both = placer._resolve_multi_start_bases(b, current, {"source": "both"})
    assert len(both) == 2
    assert any(torch.allclose(candidate, current) for candidate in both)
    assert any(torch.allclose(candidate, b.macro_positions) for candidate in both)

    defaulted = placer._resolve_multi_start_bases(b, current, {"source": "unknown"})
    assert len(defaulted) == 1
    assert torch.allclose(defaulted[0], b.macro_positions)


def test_exact_local_refine_can_read_alternate_config_section():
    mod = _load_team_module()
    b = _dummy_benchmark(mod)
    placer = mod.TeamPlasmaPlacer(
        config_overrides={
            "exact_local_refine": {"enabled": False},
            "post_soft_exact_local_refine": {"enabled": False},
        },
        seed=123,
    )

    placement, best = placer._run_exact_local_refine(
        b.macro_positions.clone(),
        b,
        plc=None,
        edge_index=torch.zeros((0, 2), dtype=torch.long),
        edge_weight=torch.zeros(0, dtype=torch.float32),
        anchor_targets=torch.zeros((b.num_hard_macros, 2), dtype=torch.float32),
        anchor_weights=torch.zeros(b.num_hard_macros, dtype=torch.float32),
        start_time=0.0,
        total_budget=1.0,
        edge_offsets=torch.zeros((0, 2, 2), dtype=torch.float32),
        cfg_section="post_soft_exact_local_refine",
    )

    assert torch.allclose(placement, b.macro_positions)
    assert best is None


def test_prepare_soft_candidate_for_exact_legalizes_overlap():
    mod = _load_team_module()
    b = _dummy_benchmark(mod)
    placer = mod.TeamPlasmaPlacer(seed=123)

    candidate = b.macro_positions.clone()
    candidate[0] = torch.tensor([1.0, 1.0], dtype=torch.float32)
    candidate[1] = torch.tensor([1.1, 1.0], dtype=torch.float32)

    out = placer._prepare_soft_candidate_for_exact(
        candidate,
        b,
        legal_cfg={"gap": 1e-4, "max_iters": 80, "fallback_iters": 60},
        legal_anchor_mode="current",
        legal_anchor_strength=0.0,
        legal_restore_iters=0,
    )

    assert mod.count_hard_overlaps(out, b.macro_sizes, b.num_hard_macros) == 0


def test_extract_hard_edges_from_plc_is_order_stable():
    mod = _load_team_module()
    Benchmark = mod.Benchmark
    benchmark = Benchmark(
        name="order_stable",
        canvas_width=10.0,
        canvas_height=10.0,
        num_macros=4,
        num_hard_macros=4,
        num_soft_macros=0,
        macro_positions=torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
            dtype=torch.float32,
        ),
        macro_sizes=torch.ones((4, 2), dtype=torch.float32),
        macro_fixed=torch.zeros(4, dtype=torch.bool),
        macro_names=["m0", "m1", "m2", "m3"],
        num_nets=0,
        net_nodes=[],
        net_weights=torch.zeros(0),
        grid_rows=8,
        grid_cols=8,
        hard_macro_indices=[0, 1, 2, 3],
        soft_macro_indices=[],
    )

    plc_a = _MockPlc(
        ["m0", "m1", "m2", "m3"],
        {
            "m0/p0": {"m1/p0", "m2/p0"},
            "m2/p1": {"m3/p0", "m1/p1"},
        },
    )
    plc_b = _MockPlc(
        ["m0", "m1", "m2", "m3"],
        {
            "m2/p1": {"m1/p1", "m3/p0"},
            "m0/p0": {"m2/p0", "m1/p0"},
        },
    )

    edge_index_a, edge_weight_a = mod.extract_hard_edges_from_plc(benchmark, plc_a, max_edges=64)
    edge_index_b, edge_weight_b = mod.extract_hard_edges_from_plc(benchmark, plc_b, max_edges=64)

    assert torch.equal(edge_index_a, edge_index_b)
    assert torch.allclose(edge_weight_a, edge_weight_b)


def test_extract_pin_flux_tubes_from_plc_is_order_stable_and_pin_aware():
    mod = _load_team_module()
    Benchmark = mod.Benchmark
    benchmark = Benchmark(
        name="pin_flux",
        canvas_width=10.0,
        canvas_height=10.0,
        num_macros=4,
        num_hard_macros=4,
        num_soft_macros=0,
        macro_positions=torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
            dtype=torch.float32,
        ),
        macro_sizes=torch.ones((4, 2), dtype=torch.float32),
        macro_fixed=torch.zeros(4, dtype=torch.bool),
        macro_names=["m0", "m1", "m2", "m3"],
        num_nets=0,
        net_nodes=[],
        net_weights=torch.zeros(0),
        grid_rows=8,
        grid_cols=8,
        hard_macro_indices=[0, 1, 2, 3],
        soft_macro_indices=[],
    )

    extra_modules = [
        _MockPlcModule("m0/p0", module_type="MACRO_PIN", macro_name="m0", x_offset=-0.5, y_offset=0.0),
        _MockPlcModule("m1/p0", module_type="MACRO_PIN", macro_name="m1", x_offset=0.5, y_offset=0.0),
        _MockPlcModule("m2/p0", module_type="MACRO_PIN", macro_name="m2", x_offset=0.0, y_offset=-0.5),
        _MockPlcModule("m3/p0", module_type="MACRO_PIN", macro_name="m3", x_offset=0.0, y_offset=0.5),
    ]
    nets_a = {
        "m0/p0": {"m1/p0", "m2/p0"},
        "m2/p0": {"m3/p0"},
    }
    nets_b = {
        "m2/p0": {"m3/p0"},
        "m0/p0": {"m2/p0", "m1/p0"},
    }
    plc_a = _MockPlc(["m0", "m1", "m2", "m3"], nets_a, extra_modules=extra_modules)
    plc_b = _MockPlc(["m0", "m1", "m2", "m3"], nets_b, extra_modules=extra_modules)

    edge_index_a, edge_weight_a, edge_offsets_a = mod.extract_pin_flux_tubes_from_plc(benchmark, plc_a, max_edges=64)
    edge_index_b, edge_weight_b, edge_offsets_b = mod.extract_pin_flux_tubes_from_plc(benchmark, plc_b, max_edges=64)

    assert torch.equal(edge_index_a, edge_index_b)
    assert torch.allclose(edge_weight_a, edge_weight_b)
    assert torch.allclose(edge_offsets_a, edge_offsets_b)
    assert float(edge_offsets_a.abs().max().item()) > 0.0


def test_extract_pin_flux_tubes_from_plc_supports_degree_filter():
    mod = _load_team_module()
    Benchmark = mod.Benchmark
    benchmark = Benchmark(
        name="pin_flux_degree_filter",
        canvas_width=10.0,
        canvas_height=10.0,
        num_macros=4,
        num_hard_macros=4,
        num_soft_macros=0,
        macro_positions=torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
            dtype=torch.float32,
        ),
        macro_sizes=torch.ones((4, 2), dtype=torch.float32),
        macro_fixed=torch.zeros(4, dtype=torch.bool),
        macro_names=["m0", "m1", "m2", "m3"],
        num_nets=0,
        net_nodes=[],
        net_weights=torch.zeros(0),
        grid_rows=8,
        grid_cols=8,
        hard_macro_indices=[0, 1, 2, 3],
        soft_macro_indices=[],
    )

    extra_modules = [
        _MockPlcModule("m0/p0", module_type="MACRO_PIN", macro_name="m0", x_offset=-0.5, y_offset=0.0),
        _MockPlcModule("m1/p0", module_type="MACRO_PIN", macro_name="m1", x_offset=0.5, y_offset=0.0),
        _MockPlcModule("m2/p0", module_type="MACRO_PIN", macro_name="m2", x_offset=0.0, y_offset=-0.5),
        _MockPlcModule("m3/p0", module_type="MACRO_PIN", macro_name="m3", x_offset=0.0, y_offset=0.5),
    ]
    plc = _MockPlc(
        ["m0", "m1", "m2", "m3"],
        {
            "m0/p0": {"m1/p0", "m2/p0"},
            "m2/p0": {"m3/p0"},
        },
        extra_modules=extra_modules,
    )

    edge_index_all, _, _ = mod.extract_pin_flux_tubes_from_plc(
        benchmark,
        plc,
        max_edges=64,
    )
    edge_index_filtered, _, _ = mod.extract_pin_flux_tubes_from_plc(
        benchmark,
        plc,
        max_edges=64,
        max_degree=2,
    )

    assert edge_index_all.shape[0] == 4
    assert edge_index_filtered.shape[0] == 1
    assert torch.equal(edge_index_filtered[0], torch.tensor([2, 3], dtype=torch.long))


def test_filter_soft_transport_model_keeps_high_mass_soft_nodes():
    mod = _load_team_module()

    soft_hard_index = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [2, 1],
            [2, 2],
            [3, 2],
        ],
        dtype=torch.long,
    )
    soft_hard_weight = torch.tensor([4.0, 2.0, 1.0, 3.0, 3.0, 0.5], dtype=torch.float32)
    soft_fixed_sum = torch.zeros(4, 2, dtype=torch.float32)
    soft_fixed_weight = torch.zeros(4, dtype=torch.float32)

    filt_index, filt_weight, _, filt_fixed_weight = mod.filter_soft_transport_model(
        (soft_hard_index, soft_hard_weight, soft_fixed_sum, soft_fixed_weight),
        {
            "enabled": True,
            "top_soft_nodes": 2,
            "min_soft_degree": 2,
        },
    )

    assert torch.equal(
        filt_index,
        torch.tensor(
            [
                [0, 0],
                [0, 1],
                [2, 1],
                [2, 2],
            ],
            dtype=torch.long,
        ),
    )
    assert torch.allclose(filt_weight, torch.tensor([4.0, 2.0, 3.0, 3.0], dtype=torch.float32))
    assert torch.allclose(filt_fixed_weight, torch.zeros(4, dtype=torch.float32))


def test_filter_hard_anchor_targets_keeps_top_weights():
    mod = _load_team_module()

    targets = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=torch.float32,
    )
    weights = torch.tensor([0.5, 4.0, 2.0, 6.0], dtype=torch.float32)

    out_targets, out_weights = mod.filter_hard_anchor_targets(
        targets,
        weights,
        {
            "enabled": True,
            "top_hard_nodes": 2,
        },
    )

    assert torch.equal(out_targets, targets)
    assert torch.allclose(out_weights, torch.tensor([0.0, 4.0, 0.0, 6.0], dtype=torch.float32))


def test_extract_soft_cluster_bundles_from_plc_keeps_top_soft_clusters():
    mod = _load_team_module()
    Benchmark = mod.Benchmark
    benchmark = Benchmark(
        name="soft_cluster_bundle",
        canvas_width=10.0,
        canvas_height=10.0,
        num_macros=5,
        num_hard_macros=3,
        num_soft_macros=2,
        macro_positions=torch.tensor(
            [[1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [2.0, 6.0], [6.0, 2.0]],
            dtype=torch.float32,
        ),
        macro_sizes=torch.ones((5, 2), dtype=torch.float32),
        macro_fixed=torch.zeros(5, dtype=torch.bool),
        macro_names=["h0", "h1", "h2", "s0", "s1"],
        num_nets=0,
        net_nodes=[],
        net_weights=torch.zeros(0),
        grid_rows=8,
        grid_cols=8,
        hard_macro_indices=[0, 1, 2],
        soft_macro_indices=[3, 4],
    )

    plc = _MockPlc(
        ["h0", "h1", "h2", "s0", "s1"],
        {
            "s0/p0": {"h0/p0", "h1/p0"},
            "s0/p1": {"h0/p1", "h1/p1"},
            "s1/p0": {"h1/p0", "h2/p0"},
        },
    )

    ptr, idx, off, weight = mod.extract_soft_cluster_bundles_from_plc(
        benchmark,
        plc,
        max_clusters=1,
        min_hard_degree=2,
        use_pin_offsets=False,
    )

    assert torch.equal(ptr, torch.tensor([0, 2], dtype=torch.long))
    assert torch.equal(idx, torch.tensor([0, 1], dtype=torch.long))
    assert off.shape == (2, 2)
    assert torch.allclose(weight, torch.tensor([2.0], dtype=torch.float32))


def test_extract_soft_cluster_edges_from_plc_prefers_stronger_shared_cluster():
    mod = _load_team_module()
    Benchmark = mod.Benchmark
    benchmark = Benchmark(
        name="soft_cluster_edges",
        canvas_width=10.0,
        canvas_height=10.0,
        num_macros=5,
        num_hard_macros=3,
        num_soft_macros=2,
        macro_positions=torch.tensor(
            [[1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [2.0, 6.0], [6.0, 2.0]],
            dtype=torch.float32,
        ),
        macro_sizes=torch.ones((5, 2), dtype=torch.float32),
        macro_fixed=torch.zeros(5, dtype=torch.bool),
        macro_names=["h0", "h1", "h2", "s0", "s1"],
        num_nets=0,
        net_nodes=[],
        net_weights=torch.zeros(0),
        grid_rows=8,
        grid_cols=8,
        hard_macro_indices=[0, 1, 2],
        soft_macro_indices=[3, 4],
    )

    plc = _MockPlc(
        ["h0", "h1", "h2", "s0", "s1"],
        {
            "s0/p0": {"h0/p0", "h1/p0"},
            "s0/p1": {"h0/p1", "h1/p1"},
            "s1/p0": {"h1/p0", "h2/p0"},
        },
    )

    edge_index, edge_weight, edge_offsets = mod.extract_soft_cluster_edges_from_plc(
        benchmark,
        plc,
        max_edges=1,
        min_hard_degree=2,
        use_pin_offsets=False,
    )

    assert torch.equal(edge_index, torch.tensor([[0, 1]], dtype=torch.long))
    assert edge_offsets.shape == (1, 2, 2)
    assert edge_weight.shape == (1,)
    assert float(edge_weight[0].item()) > 0.0


def test_extract_soft_cluster_edges_can_require_repeated_soft_support():
    mod = _load_team_module()
    Benchmark = mod.Benchmark
    benchmark = Benchmark(
        name="soft_cluster_edges_repeat",
        canvas_width=10.0,
        canvas_height=10.0,
        num_macros=6,
        num_hard_macros=3,
        num_soft_macros=3,
        macro_positions=torch.tensor(
            [[1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [2.0, 6.0], [6.0, 2.0], [7.0, 7.0]],
            dtype=torch.float32,
        ),
        macro_sizes=torch.ones((6, 2), dtype=torch.float32),
        macro_fixed=torch.zeros(6, dtype=torch.bool),
        macro_names=["h0", "h1", "h2", "s0", "s1", "s2"],
        num_nets=0,
        net_nodes=[],
        net_weights=torch.zeros(0),
        grid_rows=8,
        grid_cols=8,
        hard_macro_indices=[0, 1, 2],
        soft_macro_indices=[3, 4, 5],
    )

    plc = _MockPlc(
        ["h0", "h1", "h2", "s0", "s1", "s2"],
        {
            "s0/p0": {"h0/p0", "h1/p0"},
            "s1/p0": {"h0/p1", "h1/p1"},
            "s2/p0": {"h1/p2", "h2/p0"},
        },
    )

    edge_index, edge_weight, _ = mod.extract_soft_cluster_edges_from_plc(
        benchmark,
        plc,
        max_edges=8,
        min_hard_degree=2,
        min_shared_clusters=2,
        use_pin_offsets=False,
        weight_mode="unit",
    )

    assert torch.equal(edge_index, torch.tensor([[0, 1]], dtype=torch.long))
    assert edge_weight.shape == (1,)
    assert float(edge_weight[0].item()) > 0.0


def test_extract_soft_cluster_macro_groups_builds_small_components():
    mod = _load_team_module()
    Benchmark = mod.Benchmark
    benchmark = Benchmark(
        name="soft_cluster_groups",
        canvas_width=10.0,
        canvas_height=10.0,
        num_macros=8,
        num_hard_macros=4,
        num_soft_macros=4,
        macro_positions=torch.tensor(
            [
                [1.0, 1.0],
                [3.0, 3.0],
                [5.0, 5.0],
                [7.0, 7.0],
                [2.0, 6.0],
                [6.0, 2.0],
                [2.5, 6.5],
                [6.5, 2.5],
            ],
            dtype=torch.float32,
        ),
        macro_sizes=torch.ones((8, 2), dtype=torch.float32),
        macro_fixed=torch.zeros(8, dtype=torch.bool),
        macro_names=["h0", "h1", "h2", "h3", "s0", "s1", "s2", "s3"],
        num_nets=0,
        net_nodes=[],
        net_weights=torch.zeros(0),
        grid_rows=8,
        grid_cols=8,
        hard_macro_indices=[0, 1, 2, 3],
        soft_macro_indices=[4, 5, 6, 7],
    )

    plc = _MockPlc(
        ["h0", "h1", "h2", "h3", "s0", "s1", "s2", "s3"],
        {
            "s0/p0": {"h0/p0", "h1/p0"},
            "s2/p0": {"h0/p1", "h1/p1"},
            "s1/p0": {"h2/p0", "h3/p0"},
            "s3/p0": {"h2/p1", "h3/p1"},
        },
    )

    groups = mod.extract_soft_cluster_macro_groups_from_plc(
        benchmark,
        plc,
        max_edges=8,
        min_hard_degree=2,
        min_shared_clusters=2,
        weight_mode="unit",
        max_group_size=4,
        max_groups=4,
    )

    assert groups == [[0, 1], [2, 3]]

    info = mod.extract_soft_cluster_macro_group_info_from_plc(
        benchmark,
        plc,
        max_edges=8,
        min_hard_degree=2,
        min_shared_clusters=2,
        weight_mode="unit",
        max_group_size=4,
        max_groups=4,
    )
    assert info[0][0] == [0, 1]
    assert info[1][0] == [2, 3]
    assert float(info[0][1]) > 0.0
    assert float(info[1][1]) > 0.0


def test_plasma_control_scales_react_to_q_dominance():
    mod = _load_team_module()

    fields = {
        "q_over": torch.full((8, 8), 3.0, dtype=torch.float32),
        "rho": torch.full((8, 8), 0.5, dtype=torch.float32),
        "hot_force": torch.ones((4, 2), dtype=torch.float32),
        "cold_force": torch.zeros((4, 2), dtype=torch.float32),
    }
    cfg = {
        "enabled": True,
        "ratio_ref": 1.5,
        "q_force_alpha": 0.5,
        "hall_alpha": 0.4,
        "dia_alpha": 0.8,
        "balloon_alpha": 0.3,
        "hot_alpha": 0.2,
        "repulsion_alpha": 0.25,
        "net_down_beta": 0.5,
        "trust_up_gamma": 0.3,
        "temp_split_ref": 0.25,
    }

    scales = mod.compute_plasma_control_scales(fields, cfg)

    assert scales["q_rho_ratio"] > cfg["ratio_ref"]
    assert scales["q_force_scale"] > 1.0
    assert scales["hall_scale"] > 1.0
    assert scales["dia_scale"] > 1.0
    assert scales["balloon_scale"] > 1.0
    assert scales["hot_scale"] > 1.0
    assert scales["repulsion_scale"] > 1.0
    assert scales["net_scale"] < 1.0
    assert scales["trust_scale"] > 1.0


def test_plasma_control_scales_react_to_absolute_q_pressure():
    mod = _load_team_module()

    fields = {
        "q_over": torch.full((8, 8), 2.2, dtype=torch.float32),
        "rho": torch.full((8, 8), 2.0, dtype=torch.float32),
    }
    cfg = {
        "enabled": True,
        "ratio_ref": 1.5,
        "q_abs_ref": 1.2,
        "q_abs_mix": 1.0,
        "q_rhs_alpha": 0.3,
        "q_force_alpha": 0.5,
        "net_down_beta": 0.4,
        "trust_up_gamma": 0.25,
    }

    scales = mod.compute_plasma_control_scales(fields, cfg)

    assert scales["q_rho_ratio"] < cfg["ratio_ref"]
    assert scales["q_abs_dom"] > 0.0
    assert scales["q_rhs_scale"] > 1.0
    assert scales["q_force_scale"] > 1.0
    assert scales["net_scale"] < 1.0
    assert scales["trust_scale"] > 1.0


def test_snapshot_repair_uses_current_equilibrium_anchor():
    mod = _load_team_module()
    b = _dummy_benchmark(mod)

    placer = mod.TeamPlasmaPlacer(
        config_overrides={
            "global_stage": {
                "exact_snapshot_repair": {
                    "enabled": True,
                    "anchor_mode": "current",
                    "anchor_strength": 0.2,
                    "restore_iters": 2,
                    "strict_if_needed": True,
                }
            },
            "legalizer": {
                "gap": 1e-4,
                "max_iters": 60,
                "fallback_iters": 40,
                "anchor_mode": "current",
                "anchor_strength": 0.1,
                "restore_iters": 1,
            },
            "router": {"enabled": False},
            "portfolio": {"enabled": False},
        }
    )

    snapshot = b.macro_positions.clone()
    anchors = placer._resolve_legal_anchor_positions(b, snapshot, "current")
    assert anchors is not None
    assert torch.allclose(anchors, snapshot)
    assert anchors.data_ptr() != snapshot.data_ptr()

    repaired = placer._repair_snapshot_for_exact(snapshot, b)
    overlaps = mod.count_hard_overlaps(
        repaired,
        b.macro_sizes,
        num_hard_macros=b.num_hard_macros,
    )
    assert overlaps == 0


def test_router_child_disables_recursive_isolation():
    mod = _load_team_module()

    placer = mod.TeamPlasmaPlacer(
        config_overrides={
            "benchmark_isolation": {"enabled": True},
            "router": {"enabled": True},
        },
        seed=123,
    )
    child, merged = placer._build_router_child("dummy", {})

    assert bool(child.cfg.get("_router_child", False))
    assert bool(merged.get("_router_child", False))
    assert not bool(child.cfg.get("benchmark_isolation", {}).get("enabled", True))
    assert not bool(merged.get("benchmark_isolation", {}).get("enabled", True))


def test_config_local_enables_benchmark_isolation():
    mod = _load_team_module()
    placer = mod.TeamPlasmaPlacer(config_path="submissions/team_plasma/config_local.json", seed=123)
    assert bool(placer.cfg.get("benchmark_isolation", {}).get("enabled", False))


def test_config_local_uses_rule_only_plasma_router():
    cfg_path = Path("submissions/team_plasma/config_local.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))

    router = cfg.get("router", {})
    assert bool(router.get("enabled", False))
    assert not router.get("prototypes")

    targets = [entry["config_path"] for entry in router.get("rules", [])]
    default_entry = router.get("default")
    if default_entry:
        targets.append(default_entry["config_path"])

    assert targets
    for target in targets:
        target_path = Path(target)
        target_cfg = json.loads(target_path.read_text(encoding="utf-8-sig"))
        assert bool(target_cfg.get("global_stage", {}).get("enabled", False))
        assert not bool(target_cfg.get("initial_guard", {}).get("enabled", False))


def test_config_local_prioritizes_softsea_before_softanchor():
    cfg_path = Path("submissions/team_plasma/config_local.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    rules = list(cfg.get("router", {}).get("rules", []))

    name_to_index = {rule.get("name"): idx for idx, rule in enumerate(rules)}
    assert "at_small_softsea" in name_to_index
    assert "u3_large_soft_ultraratio" in name_to_index
    assert "dv_mid_qhot_pinflux" in name_to_index
    assert "dv_mid_softwarm_preserve" in name_to_index
    assert "av_softheavy_mid_cool" in name_to_index
    assert "dv_mid_preserve_equilibrium" in name_to_index
    assert "ao_small_qdom_cool" in name_to_index
    assert "am_small_lowsoft_pinflux" in name_to_index
    assert "am_small_lowsoft" in name_to_index
    assert "an_small_dense_pinflux" in name_to_index
    assert "dv_small_dense_rhohot_preserve" in name_to_index
    assert "u3_mid_softwarm_hot" in name_to_index
    assert "ao_midlarge_qdom" in name_to_index
    assert "s3_large_soft_hot" in name_to_index
    assert "av_mid_softwarm" in name_to_index
    assert name_to_index["at_small_softsea"] < name_to_index["av_softheavy_mid_cool"]
    assert name_to_index["u3_large_soft_ultraratio"] < name_to_index["s3_large_soft_hot"]
    assert name_to_index["dv_mid_qhot_pinflux"] < name_to_index["dv_mid_softwarm_preserve"]
    assert name_to_index["dv_mid_softwarm_preserve"] < name_to_index["dv_mid_preserve_equilibrium"]
    assert name_to_index["am_small_lowsoft_pinflux"] < name_to_index["am_small_lowsoft"]
    assert name_to_index["an_small_dense_pinflux"] < name_to_index["an_small_hot_balanced"]
    assert name_to_index["dv_mid_preserve_equilibrium"] < name_to_index["ao_small_qdom_cool"]
    assert name_to_index["dv_small_dense_rhohot_preserve"] < name_to_index["ao_midlarge_qdom"]
    assert name_to_index["u3_mid_softwarm_hot"] < name_to_index["av_mid_softwarm"]


def test_config_local_targets_regime_scoped_soft_final_overrides():
    cfg_path = Path("submissions/team_plasma/config_local.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    rules = list(cfg.get("router", {}).get("rules", []))
    by_name = {rule.get("name"): rule for rule in rules}

    expected = {
        "u3_large_soft_ultraratio": {"min_soft_fill": 0.50, "steps": [4, 4, 4]},
        "u3_mid_softwarm_hot": {"min_soft_fill": 0.50, "steps": [2, 2, 2]},
        "s3_large_soft_hot": {"min_soft_fill": 0.50, "steps": [2, 2, 2]},
        "s3_large_dense_balanced": {"min_soft_fill": 0.28, "steps": [2, 2, 2]},
        "ao_midlarge_qdom": {"min_soft_fill": 0.40, "steps": [2, 2, 2]},
        "av_softheavy_mid_cool": {"min_soft_fill": 0.40, "steps": [2, 2, 2]},
        "av_mid_softwarm": {"min_soft_fill": 0.40, "steps": [2, 2, 2]},
        "dv_mid_preserve_equilibrium": {"min_soft_fill": 0.34, "steps": [2, 2, 2]},
        "dv_mid_softwarm_preserve": {"min_soft_fill": 0.34, "steps": [2, 2, 2]},
        "dv_mid_qhot_pinflux": {"min_soft_fill": 0.34, "steps": [2, 2, 2]},
        "dv_small_dense_rhohot_preserve": {"min_soft_fill": 0.34, "steps": [2, 2, 2]},
        "am_small_hot_softmid": {"min_soft_fill": 0.36, "steps": [2, 2, 2]},
        "an_small_hot_balanced": {"min_soft_fill": 0.36, "steps": [2, 2, 2]},
        "an_small_dense_pinflux": {"min_soft_fill": 0.36, "steps": [2, 2, 2]},
        "am_small_lowsoft": {"min_soft_fill": 0.24, "steps": [2, 2, 2]},
        "am_small_lowsoft_pinflux": {"min_soft_fill": 0.24, "steps": [2, 2, 2]},
    }

    for name, expected_cfg in expected.items():
        rule = by_name[name]
        soft_cfg = (((rule.get("overrides") or {}).get("soft_macro")) or {})
        assert bool(soft_cfg.get("enabled", False))
        assert list(soft_cfg.get("final_num_steps", [])) == expected_cfg["steps"]
        assert bool((soft_cfg.get("adaptive") or {}).get("enabled", False))
        assert float((soft_cfg.get("adaptive") or {}).get("min_soft_fill", 0.0)) >= expected_cfg["min_soft_fill"]

    for name in ("at_small_softsea", "dg_large_harddom", "ao_small_qdom_cool"):
        rule = by_name[name]
        assert "soft_macro" not in (rule.get("overrides") or {})

    assert (
        by_name["at_small_softsea"]["config_path"]
        == "submissions/team_plasma/candidates/candidate_FO_at_sourcequench.json"
    )
    u3_init = (((by_name["u3_large_soft_ultraratio"].get("overrides") or {}).get("initialization")) or {})
    u3_multi = u3_init.get("multi_start", {})
    assert bool(u3_multi.get("enabled", False))
    assert str(u3_multi.get("source", "")).lower() == "both"
    assert list(u3_multi.get("jitter_fracs", [])) == [0.0, 0.0015, 0.003]
    assert bool((u3_multi.get("pilot") or {}).get("enabled", False))

    dv_mid_init = (((by_name["dv_mid_preserve_equilibrium"].get("overrides") or {}).get("initialization")) or {})
    dv_mid_multi = dv_mid_init.get("multi_start", {})
    assert bool(dv_mid_multi.get("enabled", False))
    assert str(dv_mid_multi.get("source", "")).lower() == "both"
    assert list(dv_mid_multi.get("jitter_fracs", [])) == [0.0, 0.0015, 0.003]
    assert bool((dv_mid_multi.get("pilot") or {}).get("enabled", False))
    assert "initialization" not in (by_name["dv_mid_softwarm_preserve"].get("overrides") or {})

    am_pinflux = by_name["am_small_lowsoft_pinflux"]
    assert (
        am_pinflux["config_path"]
        == "submissions/team_plasma/candidates/candidate_GE_am_pinflux_sourcequench.json"
    )
    am_soft_portfolio = (((am_pinflux.get("overrides") or {}).get("soft_macro")) or {}).get("portfolio", {})
    assert bool(am_soft_portfolio.get("enabled", False))
    assert bool(am_soft_portfolio.get("include_noop", False))
    strategy_names = [str(entry.get("name", "")) for entry in am_soft_portfolio.get("strategies", [])]
    assert strategy_names == ["soft1", "gentle2", "gentle1"]
    assert int(((dv_mid_multi.get("pilot") or {}).get("steps", 0))) == 1

    dv_small_legal = (((by_name["dv_small_dense_rhohot_preserve"].get("overrides") or {}).get("legalizer")) or {})
    assert str(dv_small_legal.get("anchor_mode", "")).lower() == "current"
    assert float(dv_small_legal.get("anchor_strength", 0.0)) == 0.1
    assert int(dv_small_legal.get("restore_iters", 0)) == 2


def test_probe_router_candidate_targets_plasma_global_configs():
    cfg_path = Path("submissions/team_plasma/candidates/candidate_DQ_plasma_probe_router_pure.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))

    router = cfg.get("router", {})
    assert bool(router.get("enabled", False))
    assert bool(router.get("plasma_probe", {}).get("enabled", False))

    targets = []
    for rule in router.get("rules", []):
        cp = rule.get("config_path")
        if cp:
            targets.append(cp)
    default_cp = router.get("default", {}).get("config_path")
    if default_cp:
        targets.append(default_cp)

    assert targets
    for target in targets:
        target_cfg = json.loads(Path(target).read_text(encoding="utf-8-sig"))
        if bool(target_cfg.get("portfolio", {}).get("enabled", False)):
            entries = target_cfg.get("portfolio", {}).get("entries", [])
            assert entries
            for entry in entries:
                entry_cfg = json.loads(Path(entry["config_path"]).read_text(encoding="utf-8-sig"))
                assert bool(entry_cfg.get("global_stage", {}).get("enabled", False))
        else:
            assert bool(target_cfg.get("global_stage", {}).get("enabled", False))


def test_router_pilot_rerank_can_override_nearest_prototype():
    mod = _load_team_module()
    b = _dummy_benchmark(mod)

    placer = mod.TeamPlasmaPlacer(
        config_overrides={
            "benchmark_isolation": {"enabled": False},
            "router": {
                "enabled": True,
                "rules": [],
                "weights": {"probe": 1.0},
                "top_k": 2,
                "diversify_by_config": True,
                "pilot_rerank": {
                    "enabled": True,
                    "steps": 1,
                    "distance_weight": 0.0,
                    "skip_if_best_distance_below": -1.0,
                },
                "prototypes": [
                    {
                        "name": "near",
                        "config_path": "near.json",
                        "features": {"probe": 0.0},
                    },
                    {
                        "name": "far_better",
                        "config_path": "far.json",
                        "features": {"probe": 1.0},
                    },
                ],
            }
        },
        seed=123,
    )

    placer._benchmark_router_features = lambda benchmark: {"probe": 0.0}

    class _FakeChild:
        def __init__(self, marker: float, score: float):
            self.marker = marker
            self.score = score

        def _prepare_surrogate_terms(self, benchmark, placement):
            n = benchmark.num_hard_macros
            return (
                None,
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0,), dtype=torch.float32),
                torch.zeros((n, 2), dtype=torch.float32),
                torch.ones((n,), dtype=torch.float32),
            )

        def _pilot_plasma_candidate(
            self,
            placement,
            benchmark,
            edge_index,
            edge_weight,
            anchor_targets,
            anchor_weights,
            pilot_cfg,
        ):
            return placement, self.score

        def place(self, benchmark):
            return torch.full_like(benchmark.macro_positions, self.marker)

    def _fake_build_router_child(benchmark_name, entry):
        if entry.get("name") == "near":
            return _FakeChild(marker=1.0, score=10.0), {}
        return _FakeChild(marker=2.0, score=1.0), {}

    placer._build_router_child = _fake_build_router_child
    routed = placer._run_feature_router(b)

    assert torch.allclose(routed, torch.full_like(b.macro_positions, 2.0))


def test_router_pilot_rerank_skips_exact_prototype_match_when_guarded():
    mod = _load_team_module()
    b = _dummy_benchmark(mod)

    placer = mod.TeamPlasmaPlacer(
        config_overrides={
            "benchmark_isolation": {"enabled": False},
            "router": {
                "enabled": True,
                "rules": [],
                "weights": {"probe": 1.0},
                "top_k": 2,
                "diversify_by_config": True,
                "pilot_rerank": {
                    "enabled": True,
                    "steps": 1,
                    "distance_weight": 0.0,
                    "skip_if_best_distance_below": 1.0e-9,
                },
                "prototypes": [
                    {
                        "name": "near",
                        "config_path": "near.json",
                        "features": {"probe": 0.0},
                    },
                    {
                        "name": "far_better",
                        "config_path": "far.json",
                        "features": {"probe": 1.0},
                    },
                ],
            }
        },
        seed=123,
    )

    placer._benchmark_router_features = lambda benchmark: {"probe": 0.0}

    class _FakeChild:
        def __init__(self, marker: float, score: float):
            self.marker = marker
            self.score = score

        def _prepare_surrogate_terms(self, benchmark, placement):
            n = benchmark.num_hard_macros
            return (
                None,
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0,), dtype=torch.float32),
                torch.zeros((n, 2), dtype=torch.float32),
                torch.ones((n,), dtype=torch.float32),
            )

        def _pilot_plasma_candidate(
            self,
            placement,
            benchmark,
            edge_index,
            edge_weight,
            anchor_targets,
            anchor_weights,
            pilot_cfg,
        ):
            return placement, self.score

        def place(self, benchmark):
            return torch.full_like(benchmark.macro_positions, self.marker)

    def _fake_build_router_child(benchmark_name, entry):
        if entry.get("name") == "near":
            return _FakeChild(marker=1.0, score=10.0), {}
        return _FakeChild(marker=2.0, score=1.0), {}

    placer._build_router_child = _fake_build_router_child
    routed = placer._run_feature_router(b)

    assert torch.allclose(routed, torch.full_like(b.macro_positions, 1.0))


def test_portfolio_child_disables_router_and_portfolio():
    mod = _load_team_module()

    placer = mod.TeamPlasmaPlacer(
        config_overrides={
            "benchmark_isolation": {"enabled": True},
            "router": {"enabled": True, "prototypes": [{"name": "self", "features": {}}]},
            "portfolio": {"enabled": True, "entries": [{"name": "base"}]},
        },
        seed=123,
    )
    child, merged = placer._build_portfolio_child("dummy", {"name": "base"})

    assert bool(child.cfg.get("_portfolio_child", False))
    assert bool(merged.get("_portfolio_child", False))
    assert not bool(child.cfg.get("router", {}).get("enabled", True))
    assert not bool(merged.get("router", {}).get("enabled", True))
    assert not bool(child.cfg.get("portfolio", {}).get("enabled", True))
    assert not bool(merged.get("portfolio", {}).get("enabled", True))
    assert not bool(child.cfg.get("benchmark_isolation", {}).get("enabled", True))
    assert not bool(merged.get("benchmark_isolation", {}).get("enabled", True))


def test_exact_local_refine_proposal_beam_runs_and_preserves_legality():
    mod = _load_team_module()
    b = _dummy_benchmark(mod)
    legal_start = mod.legalize_hard_macros(
        b.macro_positions.clone(),
        b.macro_sizes,
        b.macro_fixed,
        b.canvas_width,
        b.canvas_height,
        b.num_hard_macros,
        max_iters=100,
        fallback_iters=80,
    )

    placer = mod.TeamPlasmaPlacer(
        config_overrides={
            "router": {"enabled": False},
            "portfolio": {"enabled": False},
            "exact_eval": {"max_calls": {"local": 8}, "allow_midrun": {"local": True}},
            "exact_local_refine": {
                "enabled": True,
                "trials": {"local": 2},
                "proposal_beam": 3,
                "sigma_frac": 0.01,
                "cluster_radius_frac": 0.08,
                "cluster_prob": 0.4,
                "plasma_bias": 0.2,
                "recompute_every": 1,
                "guide_grid_size": 12,
                "legalize_iters": 16,
                "hotspot_prob": 0.8,
                "hotspot_topk": 2,
                "hotspot_step_frac": 0.02,
                "hotspot_cluster_radius_frac": 0.10,
                "hotspot_q_weight": 1.0,
                "hotspot_dia_weight": 0.2,
                "hotspot_hall_weight": 0.1,
                "hotspot_balloon_weight": 0.1,
                "hotspot_plasma_weight": 0.1,
                "hotspot_channel_weight": 0.2,
                "hotspot_channel_score_weight": 0.4,
                "pin_sheath": {
                    "enabled": True,
                    "uniform_edge_bias": 0.6,
                },
            },
        },
        seed=123,
    )

    placer._evaluate_exact_proxy = lambda placement, benchmark, plc: float(placement[: benchmark.num_hard_macros].pow(2).sum().item())
    placer._active_pin_edge_profile = torch.tensor(
        [
            [1.0, 0.2, 0.0, 0.0],
            [0.0, 1.2, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.1],
            [0.0, 0.0, 0.1, 0.9],
        ],
        dtype=torch.float32,
    )
    anchor_targets = b.macro_positions.clone()
    anchor_weights = torch.ones(b.num_hard_macros, dtype=torch.float32)
    edge_index, edge_weight = mod.build_knn_edges(b.macro_positions, k=2)

    refined, score = placer._run_exact_local_refine(
        legal_start,
        b,
        None,
        edge_index,
        edge_weight,
        anchor_targets,
        anchor_weights,
        start_time=0.0,
        total_budget=1.0e9,
    )

    overlaps = mod.count_hard_overlaps(
        refined,
        b.macro_sizes,
        num_hard_macros=b.num_hard_macros,
    )
    assert overlaps == 0
    assert score is not None
    assert float(score) >= 0.0


def test_transport_preconditioner_slows_large_high_degree_macros():
    mod = _load_team_module()

    sizes = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [4.0, 4.0],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [
            [0, 2],
            [1, 2],
            [0, 2],
        ],
        dtype=torch.long,
    )
    edge_weight = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32)

    pre = mod.compute_transport_preconditioner(
        sizes,
        edge_index,
        edge_weight,
        {
            "enabled": True,
            "base": 0.9,
            "area_alpha": 0.4,
            "degree_alpha": 0.3,
            "area_power": 0.5,
            "degree_power": 0.5,
            "min": 0.35,
            "max": 4.0,
        },
    ).squeeze(1)

    assert pre.shape[0] == 3
    assert float(pre[2].item()) > float(pre[1].item()) > float(pre[0].item())


def test_jax_kernel_backend_matches_torch_on_small_case():
    mod = _load_team_module()
    core = _load_core_module()

    if core.jax_accel is None or not core.jax_accel.is_available():
        return

    b = _dummy_benchmark(mod)
    positions = b.macro_positions.float()
    sizes = b.macro_sizes.float()
    edge_index = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    edge_weight = torch.ones(3, dtype=torch.float32)
    pde_cfg = {
        "picard_outer": 3,
        "gs_iters": 12,
        "omega": 1.1,
        "damping": 0.7,
        "eta": 0.2,
        "kappa_psi": 0.35,
        "psi_nonlinear_scale": 0.8,
        "rhs_softplus_beta": 8.0,
        "wall_decay_frac": 0.12,
        "density_target": 0.9,
        "overflow_power": 1.2,
    }
    rhs_weights = {"rho": 1.1, "q": 0.6, "n": 0.25, "wall": 0.7}

    a = core.compute_plasma_forces(
        positions,
        sizes,
        edge_index,
        edge_weight,
        b.canvas_width,
        b.canvas_height,
        12,
        dict(pde_cfg),
        rhs_weights,
    )
    b_jax = core.compute_plasma_forces(
        positions,
        sizes,
        edge_index,
        edge_weight,
        b.canvas_width,
        b.canvas_height,
        12,
        {**pde_cfg, "kernel_backend": "jax"},
        rhs_weights,
    )

    for key in ("rho", "q", "n_src", "net_force", "plasma_force", "psi"):
        assert torch.allclose(a[key], b_jax[key], atol=1e-4, rtol=1e-4)


def test_pin_flux_tubes_shift_connectivity_force_geometry():
    core = _load_core_module()

    positions = torch.tensor(
        [
            [4.0, 2.0],
            [4.0, 8.0],
        ],
        dtype=torch.float32,
    )
    sizes = torch.ones((2, 2), dtype=torch.float32)
    edge_index = torch.tensor([[0, 1]], dtype=torch.long)
    edge_weight = torch.ones(1, dtype=torch.float32)
    edge_offsets = torch.tensor(
        [[[-2.0, 0.0], [2.0, 0.0]]],
        dtype=torch.float32,
    )
    pde_cfg = {
        "picard_outer": 2,
        "gs_iters": 8,
        "omega": 1.1,
        "damping": 0.7,
        "eta": 0.2,
        "kappa_psi": 0.35,
        "psi_nonlinear_scale": 0.8,
        "rhs_softplus_beta": 8.0,
        "wall_decay_frac": 0.12,
        "density_target": 0.9,
        "overflow_power": 1.2,
        "pin_flux_tubes": True,
    }
    rhs_weights = {"rho": 1.0, "q": 0.8, "n": 0.3, "wall": 0.6}

    base = core.compute_plasma_forces(
        positions,
        sizes,
        edge_index,
        edge_weight,
        canvas_width=10.0,
        canvas_height=10.0,
        grid_size=12,
        pde_cfg=dict(pde_cfg),
        rhs_weights=rhs_weights,
    )
    pin_flux = core.compute_plasma_forces(
        positions,
        sizes,
        edge_index,
        edge_weight,
        canvas_width=10.0,
        canvas_height=10.0,
        grid_size=12,
        pde_cfg=dict(pde_cfg),
        rhs_weights=rhs_weights,
        edge_offsets=edge_offsets,
    )

    assert abs(float(base["net_force"][0, 0].item())) < 1e-6
    assert abs(float(pin_flux["net_force"][0, 0].item())) > 1e-3


def test_determinism_fixed_seed():
    mod = _load_team_module()
    b = _dummy_benchmark(mod)

    overrides = {
        "mode": "local",
        "benchmark_isolation": {"enabled": False},
        "router": {"enabled": False},
        "time_budget_sec": {"local": 25.0},
        "global_stage": {
            "stages": [
                {
                    "steps": 8,
                    "lr": 0.05,
                    "grid_size": 12,
                    "trust_radius_frac": 0.04,
                    "rhs_weights": {"rho": 1.0, "q": 0.5, "n": 0.2, "wall": 0.6},
                    "force_weights": {"plasma": 1.0, "net": 0.3, "repulsion": 0.2},
                }
            ]
        },
        "sa": {
            "workers": {"local": 2},
            "epochs": {"local": 2},
            "proposals_per_worker": {"local": 12},
            "plasma_guidance": {"enabled": False},
        },
    }

    p1 = mod.TeamPlasmaPlacer(config_overrides=overrides, seed=123)
    p2 = mod.TeamPlasmaPlacer(config_overrides=overrides, seed=123)

    out1 = p1.place(b)
    out2 = p2.place(b)

    assert torch.allclose(out1, out2, atol=1e-6)


def test_balloon_force_activates_on_hotspots():
    core = _load_core_module()

    positions = torch.tensor(
        [
            [2.0, 2.0],
            [2.4, 2.1],
            [8.0, 8.0],
            [8.3, 7.9],
        ],
        dtype=torch.float32,
    )
    sizes = torch.full((4, 2), 1.0, dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]], dtype=torch.long)
    edge_weight = torch.ones(4, dtype=torch.float32)

    fields = core.compute_plasma_forces(
        positions,
        sizes,
        edge_index,
        edge_weight,
        canvas_width=10.0,
        canvas_height=10.0,
        grid_size=12,
        pde_cfg={
            "picard_outer": 2,
            "gs_iters": 8,
            "omega": 1.1,
            "damping": 0.7,
            "eta": 0.2,
            "kappa_psi": 0.35,
            "psi_nonlinear_scale": 0.8,
            "rhs_softplus_beta": 8.0,
            "wall_decay_frac": 0.12,
            "density_target": 0.9,
            "overflow_power": 1.2,
            "balloon_enabled": 1.0,
            "balloon_quantile": 0.75,
            "balloon_topk": 6,
            "balloon_sigma_frac": 0.25,
            "balloon_power": 1.1,
        },
        rhs_weights={"rho": 1.0, "q": 1.0, "n": 0.2, "wall": 0.6},
    )

    balloon_force = fields["balloon_force"]
    assert balloon_force.shape == positions.shape
    assert float(balloon_force.norm(dim=1).max().item()) > 0.0


def test_rmp_kick_preserves_fixed_and_moves_movable():
    mod = _load_team_module()
    b = _dummy_benchmark(mod)
    fixed_mask = b.macro_fixed.clone()
    fixed_mask[0] = True

    kicked = mod.apply_rmp_kick(
        b.macro_positions,
        b.macro_sizes,
        fixed_mask,
        b.canvas_width,
        b.canvas_height,
        b.num_hard_macros,
        kick_cfg={
            "amplitude_frac": 0.05,
            "radial_mode": 1.0,
            "poloidal_mode": 2.0,
            "phase": 0.3,
        },
        fixed_positions=b.macro_positions,
    )

    assert torch.allclose(kicked[0], b.macro_positions[0], atol=1e-6)
    assert not torch.allclose(kicked[1:], b.macro_positions[1:], atol=1e-6)
