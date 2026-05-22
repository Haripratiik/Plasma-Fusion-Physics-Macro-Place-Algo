"""
Cylindrical-coordinate embedding of the chip canvas.

The macro placement canvas is interpreted as the poloidal (R, Z) cross-section
of an axisymmetric torus. We embed canvas coordinates (x, y) via

    R(x) = R_0 + x,    Z(y) = y

where R_0 is the major-radius offset. This makes the Grad-Shafranov operator
literal: the 1/R term in Delta-star comes from cylindrical geometry under
axisymmetry.

See docs/DERIVATION.md sections 1.3 and 2.2 for the math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class CylindricalGeometry:
    """Geometry for the cylindrical embedding of the canvas.

    All quantities derived once and cached. Cells are zero-indexed; cell
    (row, col) covers `R in [R_0 + col*dR, R_0 + (col+1)*dR]` and
    `Z in [row*dZ, (row+1)*dZ]`. Cell centers are at the half-cell offset.

    Attributes
    ----------
    canvas_width, canvas_height : float
        Original canvas extent in microns.
    grid_rows, grid_cols : int
        FD grid resolution.
    R_0 : float
        Major-radius offset chosen so that R = R_0 + x > 0 everywhere.
    device : torch.device
        Device that grid tensors live on.
    """

    canvas_width: float
    canvas_height: float
    grid_rows: int
    grid_cols: int
    R_0: float
    device: torch.device

    def __post_init__(self) -> None:
        if self.canvas_width <= 0 or self.canvas_height <= 0:
            raise ValueError("Canvas dimensions must be positive")
        if self.grid_rows < 8 or self.grid_cols < 8:
            raise ValueError("Grid must be at least 8x8 for stable FD")
        if self.R_0 <= 0:
            raise ValueError("R_0 must be positive (cylindrical singularity at R=0)")

    @property
    def dR(self) -> float:
        return float(self.canvas_width) / float(self.grid_cols)

    @property
    def dZ(self) -> float:
        return float(self.canvas_height) / float(self.grid_rows)

    def R_axis(self) -> torch.Tensor:
        """1D tensor of R values at cell centers, shape [grid_cols]."""
        cols = torch.arange(self.grid_cols, device=self.device, dtype=torch.float32)
        return self.R_0 + (cols + 0.5) * self.dR

    def Z_axis(self) -> torch.Tensor:
        """1D tensor of Z values at cell centers, shape [grid_rows]."""
        rows = torch.arange(self.grid_rows, device=self.device, dtype=torch.float32)
        return (rows + 0.5) * self.dZ

    def R_grid(self) -> torch.Tensor:
        """2D tensor of R values at cell centers, shape [grid_rows, grid_cols].

        Broadcasts R_axis() across rows.
        """
        return self.R_axis().unsqueeze(0).expand(self.grid_rows, self.grid_cols).contiguous()

    def cell_volume(self) -> torch.Tensor:
        """Axisymmetric cell volume element 2*pi*R*dR*dZ on the grid.

        Returns shape [grid_rows, grid_cols]. Used for flux-surface-aware
        integrations that respect the toroidal geometry.
        """
        import math
        return 2.0 * math.pi * self.R_grid() * (self.dR * self.dZ)

    def x_to_col_float(self, x: torch.Tensor) -> torch.Tensor:
        """Map canvas-x (microns) to fractional column index (0 .. grid_cols)."""
        return x / self.dR - 0.5

    def y_to_row_float(self, y: torch.Tensor) -> torch.Tensor:
        """Map canvas-y (microns) to fractional row index."""
        return y / self.dZ - 0.5

    def aspect_ratio(self) -> float:
        """A := R_0 / a, the tokamak aspect ratio with a = max(W, H) / 2."""
        return self.R_0 / max(0.5 * max(self.canvas_width, self.canvas_height), 1e-9)

    def inverse_aspect_ratio(self) -> float:
        """epsilon := 1/A. Default config gives epsilon ~ 0.17 (ITER-like)."""
        return 1.0 / max(self.aspect_ratio(), 1e-9)


def make_geometry(
    canvas_width: float,
    canvas_height: float,
    grid_rows: int,
    grid_cols: int,
    aspect_ratio: float = 3.0,
    device: torch.device | None = None,
) -> CylindricalGeometry:
    """Construct a CylindricalGeometry with a chosen aspect ratio.

    Parameters
    ----------
    aspect_ratio : float
        The tokamak aspect ratio A = R_0 / a, where a = max(W, H) / 2.
        Default A=3 is ITER-like; the 1/R correction is noticeable but
        not dominant. A->inf recovers the planar Laplacian (no GS effect).

    The minor radius `a` is set so the canvas inscribes the plasma cross
    section: `a = max(W, H) / 2`. Then `R_0 = A * a`.
    """
    a = 0.5 * max(canvas_width, canvas_height)
    R_0 = max(aspect_ratio * a, 0.5 * canvas_width + 1.0)
    if device is None:
        device = torch.device("cpu")
    return CylindricalGeometry(
        canvas_width=float(canvas_width),
        canvas_height=float(canvas_height),
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
        R_0=float(R_0),
        device=device,
    )


def bilinear_sample(
    field: torch.Tensor,
    points_xy: torch.Tensor,
    geom: CylindricalGeometry,
) -> torch.Tensor:
    """Bilinear sample a scalar field at canvas-coordinate points.

    Parameters
    ----------
    field : Tensor [grid_rows, grid_cols]
    points_xy : Tensor [N, 2] of canvas-coordinate (x, y) points
    geom : CylindricalGeometry

    Returns
    -------
    Tensor [N] of field values sampled bilinearly. Points outside the canvas
    are clamped to the boundary.
    """
    if points_xy.numel() == 0:
        return torch.zeros(0, dtype=field.dtype, device=field.device)

    cx = torch.clamp(geom.x_to_col_float(points_xy[:, 0]), 0.0, float(geom.grid_cols - 1))
    cy = torch.clamp(geom.y_to_row_float(points_xy[:, 1]), 0.0, float(geom.grid_rows - 1))

    x0 = torch.floor(cx).long()
    y0 = torch.floor(cy).long()
    x1 = torch.clamp(x0 + 1, max=geom.grid_cols - 1)
    y1 = torch.clamp(y0 + 1, max=geom.grid_rows - 1)

    tx = cx - x0.float()
    ty = cy - y0.float()

    v00 = field[y0, x0]
    v10 = field[y0, x1]
    v01 = field[y1, x0]
    v11 = field[y1, x1]

    return (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )


def sample_vector(
    field_x: torch.Tensor,
    field_y: torch.Tensor,
    points_xy: torch.Tensor,
    geom: CylindricalGeometry,
) -> torch.Tensor:
    """Bilinear sample (field_x, field_y) at points, returns [N, 2]."""
    fx = bilinear_sample(field_x, points_xy, geom)
    fy = bilinear_sample(field_y, points_xy, geom)
    return torch.stack([fx, fy], dim=1)
