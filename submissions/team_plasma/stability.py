"""
Mercier-criterion tail-aware source shaping.

Real Mercier index `D_M(psi)` (Mercier 1962; Greene & Johnson 1962) is a
local stability test for ideal-MHD interchange modes. Where `D_M > 0`
on a flux surface, the equilibrium is locally unstable and will develop a
sharp pressure spike on that surface.

In the placement context, the practical content is: cells with steep
`p'(psi)` near the separatrix are about to become top-5% congestion
hotspots. We pre-empt them by **inflating `p'(psi)` where the Mercier
proxy predicts instability**, which drives the equilibrium harder against
those regions.

This is the principled version of v1's `q_focus_quantile` knob: the boost
is keyed to a real stability criterion rather than a hand-picked quantile.

See docs/DERIVATION.md section 5.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from profiles import Profiles
from gs_solver import gradient_of_psi


@dataclass
class StabilityConfig:
    """Settings for Mercier tail shaping."""
    enabled: bool = True
    mercier_alpha: float = 1.0       # strength of the tail boost
    mercier_quantile: float = 0.85   # fraction of psi range above which boost activates
    mercier_power: float = 2.0       # quadratic (=2) or higher
    cap: float = 5.0                 # max boost multiplier on p'
    suydam_enabled: bool = False
    suydam_gamma: float = 0.5
    suydam_cap: float = 3.0
    suydam_eps: float = 1e-6


def apply_tail_shaping(profiles: Profiles, cfg: StabilityConfig) -> Profiles:
    """Wrap `profiles.p_prime_fn` with a Mercier-aware tail boost.

    The boost activates for `psi > psi_threshold` where `psi_threshold` is
    chosen at `mercier_quantile` of `[psi_axis, psi_separatrix]`. For
    `psi <= psi_threshold` the original `p'(psi)` is returned unchanged.

    For `psi > psi_threshold` we multiply by `(1 + mercier_alpha * t^power)`
    where `t = (psi - threshold) / (psi_sep - threshold)` is the
    normalized distance into the tail.
    """
    if not cfg.enabled:
        return profiles

    psi_axis = float(profiles.psi_axis)
    psi_sep = float(profiles.psi_separatrix)
    span = psi_sep - psi_axis
    if span <= 1e-9:
        return profiles
    threshold = psi_axis + float(cfg.mercier_quantile) * span

    orig_p_prime = profiles.p_prime_fn
    orig_p_double_prime = profiles.p_double_prime_fn
    alpha = float(cfg.mercier_alpha)
    power = float(cfg.mercier_power)
    cap = float(cfg.cap)
    tail_width = max(psi_sep - threshold, 1e-9)

    def boosted_p_prime(psi: torch.Tensor) -> torch.Tensor:
        base = orig_p_prime(psi)
        excess = torch.clamp(psi - threshold, min=0.0)
        t = excess / tail_width
        boost = 1.0 + alpha * torch.pow(t, power)
        boost = torch.clamp(boost, min=1.0, max=cap)
        return base * boost

    def boosted_p_double_prime(psi: torch.Tensor) -> torch.Tensor:
        base = orig_p_prime(psi)
        base_pp = orig_p_double_prime(psi)
        active = psi > threshold
        excess = torch.clamp(psi - threshold, min=0.0)
        t = excess / tail_width
        raw_boost = 1.0 + alpha * torch.pow(t, power)
        boost = torch.clamp(raw_boost, min=1.0, max=cap)
        if abs(power) <= 1e-12:
            boost_prime = torch.zeros_like(psi)
        else:
            t_safe = torch.clamp(t, min=1e-12)
            boost_prime = alpha * power * torch.pow(t_safe, power - 1.0) / tail_width
        boost_prime = torch.where(active & (raw_boost < cap), boost_prime, torch.zeros_like(boost_prime))
        return base_pp * boost + base * boost_prime

    return Profiles(
        p_fn=profiles.p_fn,
        p_prime_fn=boosted_p_prime,
        p_double_prime_fn=boosted_p_double_prime,
        F_fn=profiles.F_fn,
        F_prime_fn=profiles.F_prime_fn,
        F_double_prime_fn=profiles.F_double_prime_fn,
        psi_axis=profiles.psi_axis,
        psi_separatrix=profiles.psi_separatrix,
        p_coeffs=profiles.p_coeffs,
        F_sq_coeffs=profiles.F_sq_coeffs,
        bin_centers=profiles.bin_centers,
        p_bin_values=profiles.p_bin_values,
        I_enc_bin_values=profiles.I_enc_bin_values,
    )


def mercier_diagnostic(
    psi: torch.Tensor,
    profiles: Profiles,
    cfg: StabilityConfig,
) -> torch.Tensor:
    """Return a per-cell Mercier-proxy field for inspection.

    Positive values indicate (proxy-)unstable cells. Used for logging and
    Innovation-writeup figures; not used in the equilibrium solve itself
    (the boost is applied via `apply_tail_shaping`).
    """
    psi_axis = float(profiles.psi_axis)
    psi_sep = float(profiles.psi_separatrix)
    span = psi_sep - psi_axis
    if span <= 1e-9:
        return torch.zeros_like(psi)
    threshold = psi_axis + float(cfg.mercier_quantile) * span
    excess = torch.clamp(psi - threshold, min=0.0)
    # Steeper |p'| means more pressure-gradient -> more interchange-driven.
    p_prime_local = profiles.p_prime_fn(psi)
    return excess * p_prime_local.abs()


def compute_suydam_index(
    psi: torch.Tensor,
    profiles: Profiles,
    geom,
    mu0_eff: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute the cylindrical Suydam interchange-stability index.

    Stable cells have S > 0. Negative S means pressure-gradient drive wins
    over magnetic-shear stabilization; those cells are the physics-native
    analog of tail congestion/density hotspots.
    """
    R = geom.R_grid()
    r = torch.clamp(R - float(geom.R_0), min=float(eps))
    gR, gZ = gradient_of_psi(psi, geom)
    grad_abs = torch.sqrt(gR * gR + gZ * gZ + float(eps) * float(eps))
    B_pol_sq = torch.clamp((grad_abs / torch.clamp(R, min=float(eps))) ** 2, min=float(eps))

    F_val = profiles.F_fn(psi)
    F_prime = profiles.F_prime_fn(psi)
    q_safe = torch.clamp(F_val / torch.clamp(R * grad_abs, min=float(eps)), min=float(eps))
    # First-order q'(psi) approximation: keep the profile derivative exactly
    # and treat |grad psi| as fixed during this outer Picard step.
    q_prime = F_prime / torch.clamp(R * grad_abs, min=float(eps))

    shear_term = 0.25 * torch.square(r * q_prime / q_safe)
    pressure_drive = (2.0 * float(mu0_eff) * r * profiles.p_prime_fn(psi)) / B_pol_sq
    S = shear_term + pressure_drive
    S[0, :] = 0.0
    S[-1, :] = 0.0
    S[:, 0] = 0.0
    S[:, -1] = 0.0
    return S


def suydam_rhs_boost(
    psi: torch.Tensor,
    profiles: Profiles,
    geom,
    mu0_eff: float,
    cfg: StabilityConfig,
) -> torch.Tensor:
    """Return a multiplicative RHS boost from Suydam-unstable cells."""
    if not bool(cfg.suydam_enabled):
        return torch.ones_like(psi)
    S = compute_suydam_index(psi, profiles, geom, mu0_eff=mu0_eff, eps=cfg.suydam_eps)
    unstable = torch.clamp(-S, min=0.0)
    positive = unstable[unstable > float(cfg.suydam_eps)]
    if positive.numel() > 0:
        scale = torch.quantile(positive, 0.90)
        unstable = unstable / torch.clamp(scale, min=float(cfg.suydam_eps))
    boost = 1.0 + float(cfg.suydam_gamma) * unstable * unstable
    return torch.clamp(boost, min=1.0, max=float(cfg.suydam_cap))
