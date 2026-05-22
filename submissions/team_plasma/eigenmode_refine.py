"""
BFKK-style eigenmode proposals for hard-macro refinement.

This module implements the minimum viable R3 path from the research log:
construct a small linearized stability operator on a selected hard-macro
subset, find its lowest modes, and use those modes only as exact-score-gated
placement proposals.

Plasma analogy: the graph Laplacian is field-line tension, the smooth proxy
hotspot pressure is the destabilizing pressure-gradient drive, and the lowest
eigenvectors are the easiest-to-excite MHD displacement modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch


@dataclass
class EigenmodeRefineConfig:
    enabled: bool = False
    n_macros: int = 32
    n_modes: int = 4
    graph_tension: float = 0.20
    pressure_drive: float = 1.0
    ridge: float = 1e-3
    mode_scales: list = field(default_factory=lambda: [0.25, 0.5, 1.0, 1.5])
    step_frac: float = 0.006
    max_budget_frac: float = 0.90


def select_eigenmode_active_macros(
    degree: torch.Tensor,
    smooth_priority: torch.Tensor,
    fixed: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Select the macro subset where BFKK modes should be computed."""
    n = int(degree.numel())
    if n == 0:
        return torch.zeros(0, dtype=torch.long)
    k = min(max(1, int(k)), n)
    deg = degree.detach().cpu().float()
    pri = smooth_priority[:n].detach().cpu().float() if smooth_priority.numel() >= n else torch.zeros(n)
    pri = torch.clamp(pri, min=0.0)
    if float(deg.max().item()) > 1e-9:
        deg = deg / float(deg.max().item())
    if float(pri.max().item()) > 1e-9:
        pri = pri / float(pri.max().item())
    score = deg * (1.0 + pri)
    score = torch.where(fixed.detach().cpu().bool(), torch.full_like(score, -1.0), score)
    return torch.topk(score, k=k).indices.long()


def build_bfkk_modes(
    active: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    smooth_priority: torch.Tensor,
    smooth_directions: torch.Tensor,
    cfg: EigenmodeRefineConfig,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Return vector displacement modes and their scalar eigenvalues."""
    active = active.detach().cpu().long()
    m = int(active.numel())
    if m <= 1:
        return [], torch.zeros(0)

    local_of = {int(idx.item()): p for p, idx in enumerate(active)}
    L = torch.zeros(m, m, dtype=torch.float32)
    edges_cpu = edge_index.detach().cpu().long()
    weights_cpu = edge_weight.detach().cpu().float()
    for e in range(int(edges_cpu.shape[0])):
        a = int(edges_cpu[e, 0].item())
        b = int(edges_cpu[e, 1].item())
        if a not in local_of or b not in local_of or a == b:
            continue
        ia = local_of[a]
        ib = local_of[b]
        w = max(0.0, float(weights_cpu[e].item()))
        L[ia, ia] += w
        L[ib, ib] += w
        L[ia, ib] -= w
        L[ib, ia] -= w
    if float(L.abs().max().item()) > 1e-9:
        L = L / float(torch.diag(L).clamp_min(1e-9).max().item())

    pri = smooth_priority[active].detach().cpu().float()
    pri = torch.clamp(pri, min=0.0)
    if float(pri.max().item()) > 1e-9:
        pri = pri / float(pri.max().item())

    H = float(cfg.graph_tension) * L
    H = H - float(cfg.pressure_drive) * torch.diag(pri)
    H = H + float(cfg.ridge) * torch.eye(m, dtype=torch.float32)
    evals, evecs = torch.linalg.eigh(H)

    dirs = smooth_directions[active].detach().cpu().float()
    norms = torch.linalg.norm(dirs, dim=1, keepdim=True)
    fallback = torch.zeros_like(dirs)
    fallback[:, 0] = 1.0
    dirs = torch.where(norms > 1e-9, dirs / norms.clamp_min(1e-9), fallback)
    tangent = torch.stack([-dirs[:, 1], dirs[:, 0]], dim=1)

    modes: List[torch.Tensor] = []
    max_modes = min(max(1, int(cfg.n_modes)), m)
    for mode_idx in range(max_modes):
        amp = evecs[:, mode_idx].unsqueeze(1)
        for polarization in (dirs, tangent):
            disp = torch.zeros(int(smooth_directions.shape[0]), 2, dtype=torch.float32)
            disp[active] = amp * polarization
            rms = torch.sqrt(torch.mean(torch.sum(disp[active] * disp[active], dim=1))).clamp_min(1e-9)
            modes.append(disp / rms)

    return modes, evals[:max_modes]

