"""
Hard macros as internal current-bearing coils in the GS equilibrium.

Each hard macro contributes a toroidal current `I_i` proportional to its
connectivity. The current density `J_phi_i = I_i / area_i` is deposited over
the macro's footprint, and enters the GS right-hand side as `-mu0_eff R J_phi`.

In real tokamak equilibrium codes (VMEC, EFIT) external poloidal-field coils
enter exactly this way; we are just treating macros as the internal version
of the same object.

See docs/DERIVATION.md sections 2.7 and 3.

Macro forces from the equilibrium are computed as

    F_macro_i  =  -(I_i / area_i) * integral over macro_i of (1/R) grad(psi) dA.

Vectorized as a bilinear sample at the macro center scaled by `1/R(center)`
for the first cut; full integral version provided as `_macro_force_integral`
for higher accuracy on large macros.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from geometry import CylindricalGeometry, sample_vector
from gs_solver import gradient_of_psi


@dataclass
class CoilConfig:
    """Settings for macro coil currents and force evaluation."""
    I_0: float = 0.05            # per-net current quantum
    use_degree: bool = True      # I_i ~ degree (vs constant)
    use_area: bool = False       # I_i also scales with area
    integrate_force: bool = False  # full integral vs point sample
    force_scale: float = 1.0     # overall force scaling


@dataclass
class MacroCoils:
    """Current-bearing macros: positions, sizes, currents, fixity."""
    positions: torch.Tensor    # [N, 2]
    sizes: torch.Tensor        # [N, 2]
    currents: torch.Tensor     # [N]
    fixed: torch.Tensor        # [N] bool

    @property
    def n(self) -> int:
        return int(self.positions.shape[0])

    def update_positions(self, new_positions: torch.Tensor) -> "MacroCoils":
        """Return a new MacroCoils with updated positions; other fields shared."""
        return MacroCoils(
            positions=new_positions,
            sizes=self.sizes,
            currents=self.currents,
            fixed=self.fixed,
        )


def compute_macro_currents(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    cfg: CoilConfig,
) -> torch.Tensor:
    """Compute per-macro effective current `I_i` from connectivity.

    Default: I_i = I_0 * sum of net weights at macro i.
    Optionally also scales with macro area.
    """
    n = int(positions.shape[0])
    device = positions.device
    if n == 0:
        return torch.zeros(0, dtype=torch.float32, device=device)

    degree = torch.zeros(n, dtype=torch.float32, device=device)
    if edge_index.numel() > 0 and cfg.use_degree:
        # Sum edge weights at each endpoint.
        e_src = edge_index[:, 0].long().clamp(0, n - 1)
        e_dst = edge_index[:, 1].long().clamp(0, n - 1)
        ew = edge_weight.to(torch.float32)
        degree.scatter_add_(0, e_src, ew)
        degree.scatter_add_(0, e_dst, ew)

    if not cfg.use_degree:
        degree = torch.ones(n, dtype=torch.float32, device=device)

    I_i = float(cfg.I_0) * degree

    if cfg.use_area:
        areas = (sizes[:, 0] * sizes[:, 1]).to(torch.float32)
        mean_area = float(areas.mean().clamp(min=1e-9).item())
        I_i = I_i * (areas / max(mean_area, 1e-9))

    return I_i


def make_macro_coils(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    fixed_mask: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    cfg: CoilConfig,
) -> MacroCoils:
    """Convenience factory: positions + sizes + connectivity -> MacroCoils."""
    currents = compute_macro_currents(positions, sizes, edge_index, edge_weight, cfg)
    return MacroCoils(
        positions=positions.clone(),
        sizes=sizes.clone(),
        currents=currents,
        fixed=fixed_mask.clone(),
    )


def deposit_coil_currents(coils: MacroCoils, geom: CylindricalGeometry) -> torch.Tensor:
    """Deposit per-macro `J_phi = I/area` onto the FD grid.

    Returns a `[grid_rows, grid_cols]` tensor of the toroidal current density.
    Uses an integral-image (summed-area-table) scatter for vectorized deposition.
    """
    rows, cols = geom.grid_rows, geom.grid_cols
    device = geom.device
    n = coils.n
    if n == 0:
        return torch.zeros(rows, cols, device=device, dtype=torch.float32)

    cw, ch = geom.dR, geom.dZ
    cell_area = cw * ch

    x = coils.positions[:, 0]
    y = coils.positions[:, 1]
    w = coils.sizes[:, 0]
    h = coils.sizes[:, 1]
    I = coils.currents

    # Macro footprint as a rectangle aligned with the grid.
    x_lo = x - 0.5 * w
    x_hi = x + 0.5 * w
    y_lo = y - 0.5 * h
    y_hi = y + 0.5 * h

    # Cells the macro overlaps.
    c0 = torch.clamp((x_lo / cw).long(), 0, cols - 1)
    c1 = torch.clamp((x_hi / cw).long(), 0, cols - 1)
    r0 = torch.clamp((y_lo / ch).long(), 0, rows - 1)
    r1 = torch.clamp((y_hi / ch).long(), 0, rows - 1)

    # Current per cell within the footprint = I / area_in_cells. We approximate
    # area_in_cells by the bbox area in cell units (each cell counts as 1).
    bbox_cells = torch.clamp(((x_hi - x_lo) / cw + 1.0) * ((y_hi - y_lo) / ch + 1.0), min=1.0)
    j_per_cell = I / bbox_cells / max(cell_area, 1e-12)

    diff = torch.zeros(rows + 1, cols + 1, device=device, dtype=torch.float32)
    flat = diff.view(-1)
    cols_buf = cols + 1
    flat.scatter_add_(0, r0 * cols_buf + c0, j_per_cell)
    flat.scatter_add_(0, (r1 + 1) * cols_buf + c0, -j_per_cell)
    flat.scatter_add_(0, r0 * cols_buf + (c1 + 1), -j_per_cell)
    flat.scatter_add_(0, (r1 + 1) * cols_buf + (c1 + 1), j_per_cell)

    J_phi = torch.cumsum(torch.cumsum(diff, dim=0), dim=1)[:rows, :cols]
    return J_phi


def coil_rhs_contribution(
    coils: MacroCoils,
    geom: CylindricalGeometry,
    mu0_eff: float,
) -> torch.Tensor:
    """RHS contribution from coil currents.

    From Ampere's law in axisymmetric coordinates: `Delta-star psi = -mu0 R J_phi`.
    So coil currents add `-mu0_eff * R * J_phi` to the GS right-hand side.
    """
    J_phi = deposit_coil_currents(coils, geom)
    R = geom.R_grid()
    return -float(mu0_eff) * R * J_phi


def macro_force_from_psi(
    psi: torch.Tensor,
    coils: MacroCoils,
    geom: CylindricalGeometry,
    cfg: CoilConfig,
) -> torch.Tensor:
    """Forces on hard macros from the GS equilibrium.

    Force on macro i:
        F_i  =  -(I_i / area_i) * integral_{macro_i} (1/R) grad(psi) dA.

    Point-sample variant (fast, default): integrate at macro center.
    """
    n = coils.n
    if n == 0:
        return torch.zeros(0, 2, device=geom.device, dtype=torch.float32)

    gR, gZ = gradient_of_psi(psi, geom)
    R_grid = geom.R_grid()
    # Pre-divide gradients by R so the sampler returns (1/R) grad psi.
    gR_over_R = gR / torch.clamp(R_grid, min=1e-9)
    gZ_over_R = gZ / torch.clamp(R_grid, min=1e-9)

    if not cfg.integrate_force:
        # Point sample at macro centers.
        grad_at_macros = sample_vector(gR_over_R, gZ_over_R, coils.positions, geom)
        # F_i = -(I_i / area_i) * grad_psi_over_R(center). Per-area gives correct
        # scaling: force is proportional to current density * gradient.
        areas = (coils.sizes[:, 0] * coils.sizes[:, 1]).clamp(min=1e-9)
        scale = -coils.currents / areas
        forces = scale.unsqueeze(1) * grad_at_macros
    else:
        # Full integral by averaging over cells in the macro footprint.
        # For macros covering only a few cells, this is essentially the
        # point sample. For large macros it is more accurate.
        forces = _macro_force_integral(gR_over_R, gZ_over_R, coils, geom)

    forces = forces * float(cfg.force_scale)
    if bool(coils.fixed.any()):
        forces[coils.fixed] = 0.0
    return forces


def _macro_force_integral(
    field_x: torch.Tensor,
    field_y: torch.Tensor,
    coils: MacroCoils,
    geom: CylindricalGeometry,
) -> torch.Tensor:
    """Integrate field over each macro footprint via simple grid binning.

    For each macro, sum field values at the cells its bounding box overlaps,
    times the cell area, and divide by macro area to get the per-area force
    contribution. Then multiply by -I_i.
    """
    n = coils.n
    device = geom.device
    rows, cols = geom.grid_rows, geom.grid_cols
    cw, ch = geom.dR, geom.dZ
    cell_area = cw * ch

    forces = torch.zeros(n, 2, device=device, dtype=torch.float32)

    # Simple per-macro loop (still vectorized internally). Only used when
    # cfg.integrate_force is True; not on the hot path.
    for i in range(n):
        x = float(coils.positions[i, 0].item())
        y = float(coils.positions[i, 1].item())
        w = float(coils.sizes[i, 0].item())
        h = float(coils.sizes[i, 1].item())
        I = float(coils.currents[i].item())
        area = max(w * h, 1e-9)

        c0 = max(0, min(cols - 1, int((x - 0.5 * w) / cw)))
        c1 = max(0, min(cols - 1, int((x + 0.5 * w) / cw)))
        r0 = max(0, min(rows - 1, int((y - 0.5 * h) / ch)))
        r1 = max(0, min(rows - 1, int((y + 0.5 * h) / ch)))

        sub_x = field_x[r0 : r1 + 1, c0 : c1 + 1]
        sub_y = field_y[r0 : r1 + 1, c0 : c1 + 1]
        ix = float(sub_x.sum().item()) * cell_area
        iy = float(sub_y.sum().item()) * cell_area
        # Force = -(I / area) * integral
        forces[i, 0] = -I / area * ix
        forces[i, 1] = -I / area * iy

    return forces
