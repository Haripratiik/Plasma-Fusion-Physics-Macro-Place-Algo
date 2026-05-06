"""
Configuration utilities for TeamPlasmaPlacer.

This keeps all solver knobs in JSON-serializable dictionaries so local/cloud
profiles are easy to switch without code edits.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONFIG: Dict[str, Any] = {
    "mode": "local",
    "seed": 42,
    "device": "auto",
    "initialization": {
        "jitter_frac": 0.0,
        "rmp_kick": {},
        "multi_start": {
            "enabled": False,
            "jitter_fracs": [],
            "trials": 0,
            "rmp_kicks": [],
            "pilot": {
                "enabled": False,
                "steps": 0,
            },
        },
    },
    "time_budget_sec": {
        "local": 90.0,
        "cloud": 1800.0,
    },
    "benchmark_isolation": {
        "enabled": True,
        "work_dir": "output/team_plasma/_isolation_tmp",
    },
    "portfolio": {
        "enabled": False,
        "entries": [],
    },
    "exact_eval": {
        "max_calls": {
            "local": 1,
            "cloud": 12,
        },
        "allow_midrun": {
            "local": False,
            "cloud": True,
        },
    },
    "global_stage": {
        "enabled": True,
        "snapshot_every": 6,
        "max_surrogate_edges": 14000,
        "dynamic_congestion": {
            "enabled": False,
            "q_boost_alpha": 0.0,
            "net_down_beta": 0.0,
            "trust_up_gamma": 0.0,
            "q_peak_ref": 1.0,
            "stage_names": [],
        },
        "plasma_control": {
            "enabled": False,
            "ratio_ref": 1.35,
            "q_quantile": 0.90,
            "rho_quantile": 0.90,
            "q_abs_ref": 1.0,
            "rho_abs_ref": 1.0,
            "q_abs_mix": 0.0,
            "rho_abs_mix": 0.0,
            "q_rhs_alpha": 0.0,
            "rho_rhs_alpha": 0.0,
            "q_force_alpha": 0.0,
            "rho_force_alpha": 0.0,
            "hall_alpha": 0.0,
            "dia_alpha": 0.0,
            "balloon_alpha": 0.0,
            "hot_alpha": 0.0,
            "cold_alpha": 0.0,
            "repulsion_alpha": 0.0,
            "net_down_beta": 0.0,
            "anchor_down_beta": 0.0,
            "trust_up_gamma": 0.0,
            "temp_split_ref": 0.25,
            "temp_q_mix": 0.0,
            "temp_rho_mix": 0.0,
            "cap": 2.5,
            "stage_names": [],
        },
        "soft_background": {
            "enabled": False,
            "min_soft_fill": 0.45,
            "ref_soft_fill": 0.75,
            "power": 1.0,
        },
        "soft_transport": {
            "enabled": False,
            "min_soft_fill": 0.45,
            "update_every": 2,
            "lr_scale": 0.55,
            "trust_radius_frac": 0.010,
            "preconditioner": {
                "enabled": False,
                "base": 0.95,
                "area_alpha": 0.20,
                "degree_alpha": 0.0,
                "area_quantile": 0.60,
                "degree_quantile": 0.70,
                "area_power": 0.5,
                "degree_power": 0.5,
                "min": 0.35,
                "max": 2.5,
            },
            "force_weights": {
                "plasma": 0.20,
                "q_pressure": 0.10,
                "rho_pressure": 0.0,
                "bg_pressure": 0.25,
                "anchor": 0.75,
            },
            "final_legalize_iters": 30,
        },
        "soft_anchor": {
            "enabled": False,
            "min_soft_fill": 0.35,
            "ref_soft_fill": 0.75,
            "power": 1.0,
            "floor": 0.0,
            "confidence_quantile": 0.90,
            "confidence_power": 0.50,
            "confidence_min": 0.35,
            "confidence_max": 1.75,
            "alignment": {
                "enabled": False,
                "plasma_weight": 1.0,
                "q_weight": 0.8,
                "rho_weight": 0.5,
                "repulsion_weight": 0.7,
                "threshold": 0.0,
                "power": 1.0,
                "min_gate": 0.0,
                "max_gate": 1.0,
            },
        },
        "knn_edges": {
            "enabled": True,
            "k": 6,
            "weight": 0.25,
        },
        "adaptive_scale": {
            "enabled": True,
            "reference_num_hard": 320.0,
            "min_scale": 0.85,
            "max_scale": 1.65,
        },
        "preconditioner": {
            "enabled": False,
            "base": 1.0,
            "area_alpha": 0.0,
            "degree_alpha": 0.0,
            "area_quantile": 0.60,
            "degree_quantile": 0.70,
            "area_power": 0.5,
            "degree_power": 0.5,
            "degree_floor": 1.0e-3,
            "min": 0.25,
            "max": 8.0,
        },
        "pde": {
            "picard_outer": 4,
            "gs_iters": 28,
            "omega": 1.10,
            "damping": 0.65,
            "eta": 0.22,
            "kappa_base": 1.00,
            "kappa_rho": 1.20,
            "kappa_q": 0.70,
            "kappa_psi": 0.35,
            "psi_nonlinear_scale": 0.85,
            "convergence_tol": 1e-4,
            "rhs_softplus_beta": 8.0,
            "wall_decay_frac": 0.12,
            "density_target": 0.90,
            "overflow_power": 1.2,
            "anchor_q_weight": 0.0,
            "anchor_n_weight": 0.0,
            "pin_flux_tubes": False,
        },
        "stages": [
            {
                "name": "P1",
                "steps": 22,
                "lr": 0.085,
                "grid_size": 16,
                "trust_radius_frac": 0.050,
                "rhs_weights": {
                    "rho": 1.10,
                    "q": 0.35,
                    "n": 0.25,
                    "wall": 0.90,
                },
                "force_weights": {
                    "plasma": 1.00,
                    "rho_pressure": 0.00,
                    "q_pressure": 0.00,
                    "net": 0.42,
                    "repulsion": 0.18,
                    "anchor": 0.22,
                },
            },
            {
                "name": "P2",
                "steps": 22,
                "lr": 0.060,
                "grid_size": 32,
                "trust_radius_frac": 0.032,
                "rhs_weights": {
                    "rho": 1.25,
                    "q": 0.65,
                    "n": 0.32,
                    "wall": 0.70,
                },
                "force_weights": {
                    "plasma": 1.00,
                    "rho_pressure": 0.00,
                    "q_pressure": 0.00,
                    "net": 0.34,
                    "repulsion": 0.24,
                    "anchor": 0.24,
                },
            },
            {
                "name": "P3",
                "steps": 18,
                "lr": 0.040,
                "grid_size": 64,
                "trust_radius_frac": 0.020,
                "rhs_weights": {
                    "rho": 1.50,
                    "q": 0.95,
                    "n": 0.40,
                    "wall": 0.60,
                },
                "force_weights": {
                    "plasma": 1.00,
                    "rho_pressure": 0.00,
                    "q_pressure": 0.00,
                    "net": 0.28,
                    "repulsion": 0.30,
                    "anchor": 0.25,
                },
            },
        ],
    },
    "post_legal_global": {
        "enabled": True,
        "steps": 6,
        "lr": 0.020,
        "grid_size": 40,
        "trust_radius_frac": 0.012,
        "relegalize_every": 3,
        "max_budget_frac": 0.82,
        "rhs_weights": {
            "rho": 1.35,
            "q": 1.15,
            "n": 0.14,
            "wall": 0.45,
        },
        "force_weights": {
            "plasma": 1.00,
            "rho_pressure": 0.25,
            "q_pressure": 0.60,
            "net": 0.12,
            "repulsion": 0.38,
            "anchor": 0.00,
        },
    },
    "legalizer": {
        "gap": 1e-4,
        "max_iters": 90,
        "fallback_iters": 70,
        "resolve_hard_soft": False,
    },
    "sa": {
        "enabled": True,
        "workers": {
            "local": 4,
            "cloud": 32,
        },
        "elite_count": 2,
        "sync_fraction": 0.25,
        "epochs": {
            "local": 6,
            "cloud": 60,
        },
        "proposals_per_worker": {
            "local": 50,
            "cloud": 300,
        },
        "adaptive_scale": {
            "enabled": True,
            "reference_num_hard": 320.0,
            "min_scale": 0.85,
            "max_scale": 2.25,
        },
        "init_temperature": 1.0,
        "min_temperature": 0.02,
        "move_probs": {
            "shift": 0.44,
            "swap": 0.22,
            "attract": 0.18,
            "cluster": 0.16,
        },
        "shift_sigma_scale": 0.08,
        "cluster_radius_frac": 0.09,
        "overlap_penalty": 10.0,
        "density_penalty": 0.18,
        "congestion_penalty": 0.12,
        "anchor_penalty": 0.05,
        "anchor_move_scale": 0.0,
        "min_improvement_frac": 0.015,
        "strict_legality_filter": False,
        "soft_opt_every_sync": False,
        "plasma_guidance": {
            "enabled": True,
            "grid_size": 24,
            "recompute_every": 6,
            "bias_scale": 0.28,
            "q_bias_weight": 0.15,
            "rho_bias_weight": 0.05,
            "rhs_weights": {
                "rho": 1.35,
                "q": 0.85,
                "n": 0.30,
                "wall": 0.55,
            },
        },
    },
    "exact_local_refine": {
        "enabled": False,
        "trials": {
            "local": 0,
            "cloud": 32,
        },
        "proposal_beam": 1,
        "sigma_frac": 0.010,
        "cluster_radius_frac": 0.06,
        "cluster_prob": 0.35,
        "plasma_bias": 0.25,
        "recompute_every": 4,
        "guide_grid_size": 24,
        "legalize_iters": 25,
    },
    "soft_macro": {
        "enabled": True,
        "adaptive": {
            "enabled": False,
            "min_soft_fill": 0.5,
        },
        "sync_num_steps": [8, 8, 8],
        "final_num_steps": [24, 24, 24],
        "final_use_current_loc": True,
    },
}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *updates* into *base* and return a new dictionary."""
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_default_config_path() -> Optional[Path]:
    env_path = os.environ.get("TEAM_PLASMA_CONFIG")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        return path if path.exists() else None

    this_dir = Path(__file__).resolve().parent
    local_cfg = this_dir / "config_local.json"
    return local_cfg if local_cfg.exists() else None


def load_team_plasma_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load TeamPlasma configuration from JSON and merge it with defaults.

    Priority:
    1. explicit path argument
    2. TEAM_PLASMA_CONFIG environment variable
    3. submissions/team_plasma/config_local.json
    4. built-in defaults
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    resolved_path: Optional[Path] = None
    if path:
        candidate = Path(path).expanduser().resolve()
        resolved_path = candidate if candidate.exists() else None
    else:
        resolved_path = _resolve_default_config_path()

    if resolved_path:
        loaded = json.loads(resolved_path.read_text(encoding="utf-8-sig"))
        cfg = _deep_update(cfg, loaded)

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: Dict[str, Any]) -> None:
    mode = cfg.get("mode", "local")
    if mode not in ("local", "cloud"):
        raise ValueError(f"Unsupported mode={mode!r}. Expected 'local' or 'cloud'.")

    pde = cfg["global_stage"].get("pde", {})
    if int(pde.get("gs_iters", 0)) <= 0:
        raise ValueError("global_stage.pde.gs_iters must be > 0")
    if int(pde.get("picard_outer", 0)) <= 0:
        raise ValueError("global_stage.pde.picard_outer must be > 0")

    stages = cfg["global_stage"].get("stages", [])
    if not stages:
        raise ValueError("global_stage.stages must contain at least one stage")
    for idx, stage in enumerate(stages):
        if int(stage.get("steps", 0)) <= 0:
            raise ValueError(f"global_stage.stages[{idx}].steps must be > 0")
        if int(stage.get("grid_size", 0)) < 8:
            raise ValueError(f"global_stage.stages[{idx}].grid_size must be >= 8")
        if float(stage.get("lr", 0.0)) <= 0:
            raise ValueError(f"global_stage.stages[{idx}].lr must be > 0")

    if int(cfg["sa"].get("elite_count", 0)) < 1:
        raise ValueError("sa.elite_count must be >= 1")

    move_probs = cfg["sa"]["move_probs"]
    required = {"shift", "swap", "attract", "cluster"}
    missing = required - set(move_probs)
    if missing:
        raise ValueError(f"sa.move_probs is missing keys: {sorted(missing)}")
    prob_sum = sum(float(move_probs[k]) for k in required)
    if prob_sum <= 0:
        raise ValueError("sa.move_probs must have positive total probability")
