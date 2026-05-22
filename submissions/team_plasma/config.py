"""
Configuration loading for the GS Team Plasma placer.

The placer is parametrized by a small set of numerical and physical knobs.
Config can be supplied as a JSON file or built from defaults.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class FullConfig:
    """All settings for one run of the GS placer."""
    # Numerical
    grid_size: int = 64
    aspect_ratio: float = 3.0
    seed: int = 42
    device: str = "auto"           # 'cpu', 'cuda', or 'auto'
    diagnostics_enabled: bool = False
    plasma_purity_mode: bool = False
    plasma_purity_assert_enabled: bool = False
    plasma_purity_direct_score_min_density: float = 0.60
    plasma_purity_refine_direct_score_enabled: bool = False
    plasma_purity_refine_skip_overlap_relegalize: bool = False
    plasma_purity_refine_direct_score_max_evals: int = 96
    plasma_purity_refine_direct_score_max_evals_by_benchmark: dict = field(default_factory=dict)
    plasma_purity_seed_best_from_legalized_baseline: bool = False
    plasma_purity_seed_best_auto_benchmarks: list = field(default_factory=list)
    plasma_init_multistart_enabled: bool = False
    plasma_init_multistart_auto_benchmarks: list = field(default_factory=list)
    plasma_init_multistart_shear_fracs: list = field(default_factory=lambda: [-0.08, 0.08])
    plasma_init_multistart_max_evals: int = 4
    plasma_init_congestion_aware_flux_enabled: bool = False
    plasma_init_congestion_aware_flux_auto_benchmarks: list = field(default_factory=list)
    plasma_init_congestion_aware_flux_weight: float = 0.25

    # R1 plasma-startup initialization. Disabled by default because TILOS PLC
    # is a strong empirical warm start; this branch lets us test whether a
    # graph-spectral current-ramp state reaches a better basin.
    plasma_init_enabled: bool = False
    plasma_init_margin_frac: float = 0.04
    plasma_init_soft_grid_enabled: bool = False
    plasma_init_soft_graph_enabled: bool = True
    plasma_init_soft_assignment_enabled: bool = False
    plasma_init_soft_assignment_graph_weight: float = 0.0
    plasma_init_soft_assignment_overlap_weight: float = 0.0
    plasma_init_soft_assignment_candidates: int = 48
    plasma_init_soft_phase_lock_enabled: bool = False
    plasma_init_soft_phase_lock_weight: float = 0.0
    plasma_init_soft_phase_lock_distance_weight: float = 1.0
    plasma_init_soft_phase_lock_auto_enabled: bool = False
    plasma_init_soft_phase_lock_min_density: float = 0.60
    plasma_init_soft_phase_lock_large_open_enabled: bool = False
    plasma_init_soft_phase_lock_large_open_density: float = 0.60
    plasma_init_soft_phase_lock_large_open_area: float = 3000.0
    plasma_init_soft_phase_lock_large_open_weight: float = 0.18
    plasma_init_flux_bands_enabled: bool = True
    plasma_init_soft_spread_frac: float = 0.012
    plasma_init_soft_spread_auto_enabled: bool = False
    plasma_init_soft_spread_dense_threshold: float = 1.20
    plasma_init_soft_spread_dense_frac: float = 0.012
    plasma_init_soft_spread_open_frac: float = 0.020
    plasma_init_soft_spread_ultra_open_enabled: bool = False
    plasma_init_soft_spread_ultra_open_threshold: float = 0.60
    plasma_init_soft_spread_ultra_open_frac: float = 0.030
    plasma_init_soft_pressure_iters: int = 0
    plasma_init_soft_pressure_step_frac: float = 0.002
    plasma_init_soft_pressure_radius_frac: float = 0.035
    plasma_init_soft_pressure_anchor: float = 0.35
    plasma_init_soft_pressure_hard_weight: float = 1.0
    plasma_init_soft_pressure_wall_weight: float = 0.5
    plasma_init_normalized_laplacian: bool = True
    plasma_init_normalized_laplacian_auto_enabled: bool = False
    plasma_init_unnormalized_small_hard_threshold: int = 220
    plasma_init_unnormalized_min_density: float = 0.80
    plasma_init_n_bands: int = 8
    plasma_init_sinkhorn_iters: int = 100
    plasma_init_sinkhorn_eps: float = 0.05
    plasma_init_bare_picard_outer: int = 6
    plasma_init_weber_sweeps: int = 2

    # Picard / GS solver
    picard_outer: int = 12
    gs_inner_sweeps: int = 30
    omega_sor: float = 1.2
    omega_picard: float = 0.7
    convergence_tol: float = 1e-4
    mu0_eff: float = 1.0
    use_newton_solver: bool = False
    newton_outer_max: int = 6
    newton_tol_residual: float = 1e-4
    newton_tol_step: float = 1e-3
    newton_armijo_c1: float = 0.5
    newton_min_alpha: float = 0.015625

    # Profile fitting
    profile_n_bins: int = 32
    profile_poly_degree: int = 3
    profile_alpha_q_rho: float = 0.5
    profile_F_vac: float = 1.0
    profile_beta_F: float = 0.1
    profile_tail_enabled: bool = False
    profile_tail_quantile: float = 0.90
    profile_tail_gamma: float = 1.0
    profile_tail_power: float = 2.0

    # Coils
    coil_I_0: float = 0.05
    coil_use_degree: bool = True
    coil_use_area: bool = False
    coil_integrate_force: bool = False
    coil_force_scale: float = 1.0

    # Stability shaping (Mercier)
    mercier_enabled: bool = True
    mercier_alpha: float = 1.0
    mercier_quantile: float = 0.85
    mercier_power: float = 2.0
    mercier_cap: float = 5.0
    suydam_enabled: bool = False
    suydam_gamma: float = 0.5
    suydam_cap: float = 3.0
    suydam_eps: float = 1e-6

    # Taylor / Beltrami global relaxation overlay.
    # This is the coherent global-topology primitive from PHYSICS_APPROACH:
    # blend the local GS equilibrium toward the lowest-energy force-free mode.
    taylor_enabled: bool = True
    taylor_gamma: float = 0.10
    taylor_ramp_start_frac: float = 0.20
    taylor_ramp_end_frac: float = 0.80

    # Two-fluid soft macros
    two_fluid_enabled: bool = True
    two_fluid_drift_step_frac: float = 0.05
    two_fluid_trust_radius_frac: float = 0.02
    two_fluid_T_e: float = 1.0
    two_fluid_pde_enabled: bool = True
    two_fluid_replace_tilos_soft: bool = False
    two_fluid_D_parallel: float = 0.20
    two_fluid_D_perp: float = 0.05
    two_fluid_dt: float = 0.20
    two_fluid_max_iters: int = 80
    two_fluid_conv_tol: float = 1e-4
    two_fluid_pde_density_weight: float = 1.0
    two_fluid_pde_psi_weight: float = 0.15
    two_fluid_pde_trust_radius_frac: float = 0.01
    r4_use_psi_aligned_flow: bool = False
    r4_min_anisotropy_ratio: float = 100.0
    r4_picard_soft_equilibrium_enabled: bool = False
    r4_picard_large_square_disable_enabled: bool = False
    r4_picard_large_square_min_canvas: float = 60.0
    r4_picard_large_square_max_canvas: float = 1.0e9
    r4_picard_large_square_aspect_tol: float = 0.08
    r4_inner_steps_per_picard: int = 20
    r4_inner_steps_by_benchmark: dict = field(default_factory=dict)
    r4_picard_max_budget_frac: float = 0.70
    r4_annealing_enabled: bool = False
    r4_n_passes: int = 3
    r4_anneal_factor: float = 0.5
    r4_q_weight: float = 0.0           # soft-macro Ware-pinch (pass 185, default off)
    r4_hard_q_weight: float = 0.0      # hard-macro Ware-pinch (pass 187, default off)
    r5_pic_enabled: bool = False
    r5_pic_auto_benchmarks: list = field(default_factory=list)
    r5_pic_chain_r4_enabled: bool = False
    r5_pic_proposal_only_enabled: bool = False
    r5_pic_exact_gate_enabled: bool = False
    r5_pic_max_iters: int = 32
    r5_pic_dt_frac: float = 0.08
    r5_pic_debye_length_frac: float = 0.025
    r5_pic_sheath_width_frac: float = 0.035
    r5_pic_ion_to_electron_mass_ratio: float = 100.0
    r5_pic_net_current_scale: float = 0.30
    r5_pic_coulomb_strength: float = 0.15
    r5_pic_electric_strength: float = 0.45
    r5_pic_magnetic_strength: float = 0.20
    r5_pic_sheath_strength: float = 0.40
    r5_pic_collision_freq: float = 0.12
    r5_pic_trust_radius_frac: float = 0.006
    r5_pic_field_solve_iters: int = 42
    r5_pic_field_solver: str = "jacobi"
    r5_pic_conv_tol: float = 1e-4
    r5_pic_current_line_samples: int = 12
    r5_pic_b0_strength: float = 0.0
    r5_pic_b_ripple_strength: float = 0.0
    r5_pic_grad_b_drift_strength: float = 0.0
    r5_pic_magnetic_mirror_strength: float = 0.0
    r5_pic_pair_attraction_strength: float = 0.0
    r5_pic_pair_attraction_range_frac: float = 1.0
    r5_pic_pair_attraction_softening_frac: float = 0.01
    r5_pic_bootstrap_current_strength: float = 0.0
    r5_pic_diamagnetic_drift_strength: float = 0.0
    r5_pic_annealed_schedule_enabled: bool = False
    r5_pic_variant_portfolio_enabled: bool = False
    r5_pic_variant_portfolio: list = field(default_factory=list)
    r5_pic_variant_portfolio_max_trials: int = 4
    r5_pic_dimensionless_auto_enabled: bool = False
    r5_pic_dimensionless_portfolio_enabled: bool = False
    r5_pic_dimensionless_portfolio_max_trials: int = 4

    # Final soft-macro relaxation through the official PlacementCost engine.
    # This keeps the GS hard-macro equilibrium, but lets TILOS relax the
    # standard-cell / soft-macro field that dominates density and congestion.
    soft_tilos_enabled: bool = True
    soft_tilos_num_steps: list = field(default_factory=lambda: [2, 2])
    soft_tilos_use_current_loc: bool = True
    soft_tilos_max_budget_frac: float = 0.92
    soft_tilos_probe_enabled: bool = True
    soft_tilos_probe_num_steps: list = field(default_factory=lambda: [1, 1])
    soft_tilos_refine_num_steps: list = field(default_factory=lambda: [1, 1])
    soft_tilos_auto_steps_enabled: bool = True
    soft_tilos_probe_min_delta: float = 1e-4
    soft_tilos_portfolio_enabled: bool = False
    soft_tilos_portfolio_steps: list = field(default_factory=lambda: [[1], [1, 1], [2, 2], [3, 3]])
    soft_tilos_dense_regime_enabled: bool = True
    soft_tilos_dense_macro_density_threshold: float = 2.0
    soft_tilos_dense_soft_frac_threshold: float = 0.75
    soft_tilos_dense_probe_num_steps: list = field(default_factory=lambda: [12, 12])
    soft_tilos_dense_refine_num_steps: list = field(default_factory=lambda: [12, 12])

    # Macro trust-radius and step
    macro_step_frac: float = 0.005
    macro_trust_radius_frac: float = 0.005

    # Exact-score-guided hard-macro line search inside the Picard loop.
    # This spends a small number of TILOS proxy calls to accept only GS
    # hard-macro moves that beat the current best exact score.
    exact_line_search_enabled: bool = False
    exact_line_search_max_rounds: int = 0
    exact_line_search_period: int = 5
    exact_line_search_start_iter: int = 0
    exact_line_search_scales: list = field(default_factory=lambda: [0.25, 0.5, 1.0])
    exact_line_search_max_budget_frac: float = 0.70
    exact_hard_refine_enabled: bool = True
    exact_hard_refine_macros: int = 12
    exact_hard_refine_macros_by_benchmark: dict = field(default_factory=dict)
    exact_hard_refine_passes: int = 2
    exact_hard_refine_passes_by_benchmark: dict = field(default_factory=dict)
    exact_hard_refine_step_frac: float = 0.008
    exact_hard_refine_step_frac_by_benchmark: dict = field(default_factory=dict)
    exact_hard_refine_direction_scales: list = field(default_factory=lambda: [1.0])
    exact_hard_refine_direction_scales_by_benchmark: dict = field(default_factory=dict)
    exact_hard_refine_direction_set: str = "flux_only"
    exact_hard_refine_direction_set_by_benchmark: dict = field(default_factory=dict)
    exact_hard_refine_pass_decay: float = 1.0
    exact_hard_refine_selector: str = "mercier"
    exact_hard_refine_mercier_beta: float = 2.0
    exact_hard_refine_flux_beta: float = 1.0
    exact_hard_refine_max_budget_frac: float = 0.95
    exact_hard_refine_force_over_budget_by_benchmark: dict = field(default_factory=dict)
    exact_hard_refine_min_direct_evals_by_benchmark: dict = field(default_factory=dict)
    ntm_suppression_enabled: bool = True
    ntm_suppression_budget_boost: float = 1.5
    ntm_suppression_smooth_beta: float = 2.0
    ntm_suppression_always_retarget_enabled: bool = False
    snowflake_divertor_enabled: bool = False
    snowflake_divertor_macros: int = 8
    snowflake_divertor_step_frac: float = 0.008
    snowflake_divertor_scales: list = field(default_factory=lambda: [0.5, 1.0])
    snowflake_divertor_ray_mode: str = "radial"
    snowflake_divertor_top_frac: float = 0.05
    snowflake_divertor_max_direct_evals: int = 8
    snowflake_divertor_max_budget_frac: float = 0.92
    snowflake_divertor_relegalize_enabled: bool = False
    thermal_hopping_enabled: bool = False
    thermal_hopping_trials: int = 5
    thermal_hopping_n_macros: int = 8
    thermal_hopping_sigma_frac: float = 0.035
    thermal_hopping_top_pool_mult: int = 4
    thermal_hopping_max_budget_frac: float = 0.96
    thermal_hopping_min_delta: float = 0.003
    thermal_hopping_seed: int = 31415
    flux_rope_cluster_enabled: bool = False
    flux_rope_cluster_density_threshold: float = 0.60
    flux_rope_cluster_area_threshold: float = 3000.0
    flux_rope_cluster_roots: int = 5
    flux_rope_cluster_macros: int = 10
    flux_rope_cluster_step_frac: float = 0.012
    flux_rope_cluster_scales: list = field(default_factory=lambda: [0.5, 1.0, 1.5])
    flux_rope_cluster_max_direct_evals: int = 12
    flux_rope_cluster_max_budget_frac: float = 0.94
    eccd_reweight_enabled: bool = False
    eccd_reweight_kappa: float = 1.0
    eccd_reweight_top_frac: float = 0.05
    eccd_reweight_locality: float = 0.0

    # HAAMP minimum-viable smooth top-k surrogate. Current gradient-alignment
    # tests show the fixed density/RUDY gyroaverage is not safe as a standalone
    # global force, but keeping the field available as exact-gated refine
    # guidance recovered small-benchmark score. The global B1 step remains disabled.
    smooth_proxy_enabled: bool = True
    smooth_proxy_auto_enabled: bool = False
    smooth_proxy_auto_benchmarks: list = field(default_factory=list)
    smooth_proxy_density_weight: float = 1.0
    smooth_proxy_rudy_weight: float = 0.0
    smooth_proxy_top_frac: float = 0.10
    smooth_proxy_lse_tau: float = 0.10
    smooth_proxy_gyro_radius_frac: float = 0.025
    smooth_proxy_hpwl_weight: float = 1.0
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
    smooth_proxy_priority_beta: float = 1.0
    smooth_proxy_direction_beta: float = 1.0
    smooth_proxy_flux_curvature_weight: float = 0.0
    smooth_proxy_flux_curvature_top_frac: float = 0.10
    smooth_proxy_global_step_enabled: bool = True
    smooth_proxy_global_step_macros: int = 12
    smooth_proxy_global_step_frac: float = 0.002
    smooth_proxy_global_step_scales: list = field(default_factory=lambda: [0.25, 0.5])
    smooth_proxy_global_step_max_budget_frac: float = 0.35

    # HAAMP tearing-mode probe: exact-gated coordinated moves on small
    # connectivity islands. Enabled globally to avoid benchmark-name routing.
    tearing_mode_enabled: bool = True
    tearing_mode_auto_enabled: bool = False
    tearing_mode_auto_benchmarks: list = field(default_factory=list)
    tearing_mode_n_clusters: int = 4
    tearing_mode_cluster_size: int = 5
    tearing_mode_step_frac: float = 0.006
    tearing_mode_scales: list = field(default_factory=lambda: [0.5, 1.0])
    tearing_mode_rotation_deg: float = 5.0
    tearing_mode_max_budget_frac: float = 0.90

    # R3 / BFKK eigenmode-following refinement. Standalone A5 evidence was
    # weak, but integrated cleanup runs showed small-benchmark regressions when old R3 was
    # disabled. Keep it exact-gated while B1 global motion stays off.
    eigenmode_refine_enabled: bool = True
    eigenmode_refine_auto_enabled: bool = False
    eigenmode_refine_auto_benchmarks: list = field(default_factory=list)
    eigenmode_refine_macros: int = 24
    eigenmode_refine_modes: int = 3
    eigenmode_refine_graph_tension: float = 0.20
    eigenmode_refine_pressure_drive: float = 1.0
    eigenmode_refine_ridge: float = 1e-3
    eigenmode_refine_step_frac: float = 0.006
    eigenmode_refine_scales: list = field(default_factory=lambda: [0.25, 0.5, 1.0, 1.5])
    eigenmode_refine_trust_combos_enabled: bool = False
    eigenmode_refine_trust_combo_modes: int = 4
    eigenmode_refine_max_budget_frac: float = 0.90

    # Legalization
    legalize_gap: float = 1e-4
    legalize_max_iters: int = 120
    sheath_legalize_enabled: bool = False
    sheath_legalize_fallback_strict: bool = True
    sheath_legalize_max_iters: int = 160
    sheath_legalize_dt: float = 0.35
    sheath_legalize_damping: float = 0.60
    sheath_legalize_coulomb_strength: float = 1.0
    sheath_legalize_sheath_strength: float = 1.0
    sheath_legalize_debye_frac: float = 0.02
    sheath_legalize_max_step_frac: float = 0.010
    r2_continuation_enabled: bool = False
    r2_continuation_replace_strict: bool = False
    r2_beta_overlap_start: float = 0.10
    r2_beta_overlap_max: float = 1.0
    r2_beta_overlap_ramp_stages: int = 8
    r2_beta_wall: float = 1.0
    r2_inner_iters: int = 40
    r2_langevin_dt: float = 0.35
    r2_temp_start: float = 0.0
    r2_temp_decay: float = 0.70
    r2_deterministic_enabled: bool = False
    r2_force_tol: float = 1e-5
    r2_seed: int = 7
    r2_hardening_enabled: bool = True
    r2_hardening_iters: int = 320
    r2_hardening_max_pairs_per_iter: int = 12000
    r2_relocation_enabled: bool = True
    r2_relocation_iters: int = 0
    r2_relocation_rings: int = 6
    r2_relocation_density_weight: float = 0.0
    r2_flux_lattice_relocation_enabled: bool = True
    r2_flux_lattice_cols: int = 16
    r2_local_flux_refine_enabled: bool = False
    r2_local_flux_refine_steps: int = 5
    r2_local_flux_refine_pair_threshold: int = 4
    r2_anchor_strength: float = 0.0
    r2_anchor_release_frac: float = 0.5

    # Time budget
    time_budget_sec: float = 1500.0


_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config(path: Optional[str] = None) -> FullConfig:
    """Load configuration with fallback chain.

    Priority order:
      1. Explicit `path` argument.
      2. `TEAM_PLASMA_GS_CONFIG` environment variable.
      3. `submissions/team_plasma/config.json` next to this module.
      4. Built-in defaults.
    """
    cfg = FullConfig()

    candidates = []
    if path:
        candidates.append(path)
    env = os.environ.get("TEAM_PLASMA_GS_CONFIG")
    if env:
        candidates.append(env)
    if _DEFAULT_CONFIG_PATH.exists():
        candidates.append(str(_DEFAULT_CONFIG_PATH))

    for candidate in candidates:
        try:
            data = _load_config_json(Path(candidate))
            cfg = _apply_overrides(cfg, data)
            return cfg
        except (FileNotFoundError, json.JSONDecodeError):
            continue

    return cfg


def _load_config_json(path: Path) -> Dict[str, Any]:
    """Load a JSON config, resolving optional parent configs.

    Probe configs are often small variants over a known baseline. The
    `extends` field is intentionally local and shallow: parent keys load first,
    child keys override them, and nested dict-valued config fields remain
    intact through `_apply_overrides`.
    """
    with open(path, "r") as f:
        data = json.load(f)
    parent = data.pop("extends", None)
    if not parent:
        return data
    parent_path = Path(parent)
    candidates = []
    if parent_path.is_absolute():
        candidates.append(parent_path)
    else:
        candidates.append(Path.cwd() / parent_path)
        candidates.append(path.parent / parent_path)
    for candidate in candidates:
        if candidate.exists():
            merged = _load_config_json(candidate)
            merged.update(data)
            return merged
    raise FileNotFoundError(parent)


def _apply_overrides(cfg: FullConfig, overrides: Dict[str, Any]) -> FullConfig:
    """Update FullConfig fields from a flat or nested dict."""
    cfg_dict = asdict(cfg)
    flat: Dict[str, Any] = {}
    for k, v in overrides.items():
        if k in cfg_dict:
            cfg_dict[k] = v
        elif isinstance(v, dict):
            flat.update(_flatten_dict(v, k))
        else:
            flat[k] = v
    for k, v in flat.items():
        if k in cfg_dict:
            cfg_dict[k] = v
    return FullConfig(**cfg_dict)


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a one-level-nested dict by joining keys with underscore.

    e.g. {'mercier': {'alpha': 1.0}} -> {'mercier_alpha': 1.0}.
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


def resolve_device(device_cfg: str) -> torch.device:
    """Resolve 'auto' / 'cuda' / 'cpu' to a torch.device."""
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
