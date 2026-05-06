from __future__ import annotations

from functools import partial
from typing import Optional

import numpy as np

try:
    import jax
    import jax.lax as lax
    import jax.numpy as jnp

    HAS_JAX = True
except Exception:  # pragma: no cover - optional dependency
    jax = None
    lax = None
    jnp = None
    HAS_JAX = False


def is_available() -> bool:
    return bool(HAS_JAX)


def _as_float32(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


def _as_int32(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.int32)


def _avg_pool3_same(x: jnp.ndarray) -> jnp.ndarray:
    kernel = jnp.ones((1, 1, 3, 3), dtype=x.dtype) / 9.0
    x4 = x[jnp.newaxis, jnp.newaxis, :, :]
    y4 = lax.conv_general_dilated(
        x4,
        kernel,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return y4[0, 0]


@partial(jax.jit, static_argnames=("rows", "cols"))
def _deposit_points_bilinear_kernel(
    points: jnp.ndarray,
    weights: jnp.ndarray,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
) -> jnp.ndarray:
    out = jnp.zeros((rows, cols), dtype=jnp.float32)
    cw = jnp.asarray(canvas_width, dtype=jnp.float32) / float(max(cols, 1))
    ch = jnp.asarray(canvas_height, dtype=jnp.float32) / float(max(rows, 1))

    cx = jnp.clip(points[:, 0] / jnp.maximum(cw, 1e-12) - 0.5, 0.0, float(cols - 1))
    cy = jnp.clip(points[:, 1] / jnp.maximum(ch, 1e-12) - 0.5, 0.0, float(rows - 1))

    x0 = jnp.floor(cx).astype(jnp.int32)
    y0 = jnp.floor(cy).astype(jnp.int32)
    x1 = jnp.clip(x0 + 1, 0, cols - 1)
    y1 = jnp.clip(y0 + 1, 0, rows - 1)

    tx = cx - x0.astype(jnp.float32)
    ty = cy - y0.astype(jnp.float32)

    w00 = (1.0 - tx) * (1.0 - ty) * weights
    w10 = tx * (1.0 - ty) * weights
    w01 = (1.0 - tx) * ty * weights
    w11 = tx * ty * weights

    out = out.at[y0, x0].add(w00)
    out = out.at[y0, x1].add(w10)
    out = out.at[y1, x0].add(w01)
    out = out.at[y1, x1].add(w11)
    return out


@partial(jax.jit, static_argnames=("rows", "cols"))
def _deposit_macro_density_kernel(
    positions: jnp.ndarray,
    sizes: jnp.ndarray,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
) -> jnp.ndarray:
    cw = jnp.asarray(canvas_width, dtype=jnp.float32) / float(max(cols, 1))
    ch = jnp.asarray(canvas_height, dtype=jnp.float32) / float(max(rows, 1))
    xs = (jnp.arange(cols, dtype=jnp.float32) + 0.5) * cw
    ys = (jnp.arange(rows, dtype=jnp.float32) + 0.5) * ch
    half_cw = 0.5 * cw
    half_ch = 0.5 * ch
    cell_area = jnp.maximum(cw * ch, 1e-12)

    x = positions[:, 0:1]
    y = positions[:, 1:2]
    w = sizes[:, 0:1]
    h = sizes[:, 1:2]

    ovx = jax.nn.relu((0.5 * w + half_cw) - jnp.abs(xs[jnp.newaxis, :] - x))
    ovy = jax.nn.relu((0.5 * h + half_ch) - jnp.abs(ys[jnp.newaxis, :] - y))
    density = jnp.einsum("nr,nc->rc", ovy, ovx) / cell_area
    return density.astype(jnp.float32)


@partial(jax.jit, static_argnames=("rows", "cols"))
def _build_density_pressure_kernel(
    positions: jnp.ndarray,
    sizes: jnp.ndarray,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
    target_density: float,
    overflow_power: float,
) -> jnp.ndarray:
    density = _deposit_macro_density_kernel(positions, sizes, canvas_width, canvas_height, rows, cols)
    overflow = jax.nn.relu(density - jnp.asarray(target_density, dtype=jnp.float32))
    overflow = jnp.power(overflow + 1e-8, jnp.asarray(overflow_power, dtype=jnp.float32))
    return overflow.astype(jnp.float32)


@partial(jax.jit, static_argnames=("rows", "cols", "smoothing_passes"))
def _build_rudy_pressure_kernel(
    positions: jnp.ndarray,
    edge_index: jnp.ndarray,
    edge_weight: jnp.ndarray,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
    smoothing_passes: int,
) -> jnp.ndarray:
    cw = jnp.asarray(canvas_width, dtype=jnp.float32) / float(max(cols, 1))
    ch = jnp.asarray(canvas_height, dtype=jnp.float32) / float(max(rows, 1))

    i = edge_index[:, 0]
    j = edge_index[:, 1]
    xi = positions[i, 0]
    yi = positions[i, 1]
    xj = positions[j, 0]
    yj = positions[j, 1]

    dx = jnp.abs(xi - xj)
    dy = jnp.abs(yi - yj)
    bbox_w = dx + cw
    bbox_h = dy + ch
    demand = edge_weight * (dx + dy + 1e-6) / jnp.maximum(bbox_w * bbox_h, 1e-12)

    c0 = jnp.clip(jnp.floor(jnp.minimum(xi, xj) / jnp.maximum(cw, 1e-12)).astype(jnp.int32), 0, cols - 1)
    c1 = jnp.clip(jnp.floor(jnp.maximum(xi, xj) / jnp.maximum(cw, 1e-12)).astype(jnp.int32), 0, cols - 1)
    r0 = jnp.clip(jnp.floor(jnp.minimum(yi, yj) / jnp.maximum(ch, 1e-12)).astype(jnp.int32), 0, rows - 1)
    r1 = jnp.clip(jnp.floor(jnp.maximum(yi, yj) / jnp.maximum(ch, 1e-12)).astype(jnp.int32), 0, rows - 1)

    diff = jnp.zeros((rows + 1, cols + 1), dtype=jnp.float32)
    diff = diff.at[r0, c0].add(demand)
    diff = diff.at[r1 + 1, c0].add(-demand)
    diff = diff.at[r0, c1 + 1].add(-demand)
    diff = diff.at[r1 + 1, c1 + 1].add(demand)

    q = jnp.cumsum(jnp.cumsum(diff, axis=0), axis=1)[:rows, :cols]

    def _smooth(_, val):
        return _avg_pool3_same(val)

    q = lax.fori_loop(0, int(smoothing_passes), _smooth, q)
    mean_val = jnp.mean(q)
    q = jnp.where(mean_val > 1e-12, q / mean_val, q)
    return q.astype(jnp.float32)


@partial(jax.jit, static_argnames=("rows", "cols"))
def _build_net_source_kernel(
    positions: jnp.ndarray,
    edge_index: jnp.ndarray,
    edge_weight: jnp.ndarray,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
) -> jnp.ndarray:
    n = positions.shape[0]
    i = edge_index[:, 0]
    j = edge_index[:, 1]

    sum_w = jnp.zeros((n,), dtype=jnp.float32)
    sum_w = sum_w.at[i].add(edge_weight)
    sum_w = sum_w.at[j].add(edge_weight)

    centroid = jnp.zeros((n, 2), dtype=jnp.float32)
    centroid = centroid.at[i].add(edge_weight[:, None] * positions[j])
    centroid = centroid.at[j].add(edge_weight[:, None] * positions[i])

    target = jnp.where(
        (sum_w > 1e-12)[:, None],
        centroid / jnp.maximum(sum_w[:, None], 1e-12),
        positions,
    )
    point_weight = jnp.maximum(sum_w, 1e-3)
    src_from = _deposit_points_bilinear_kernel(positions, point_weight, canvas_width, canvas_height, rows, cols)
    src_to = _deposit_points_bilinear_kernel(target, point_weight, canvas_width, canvas_height, rows, cols)
    source = src_from - src_to
    scale = jnp.mean(jnp.abs(source))
    source = jnp.where(scale > 1e-12, source / scale, source)
    return source.astype(jnp.float32)


@partial(jax.jit, static_argnames=("rows", "cols", "smoothing_passes"))
def _build_anchor_pressure_kernel(
    positions: jnp.ndarray,
    anchor_targets: jnp.ndarray,
    scaled_w: jnp.ndarray,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
    smoothing_passes: int,
) -> jnp.ndarray:
    cw = jnp.asarray(canvas_width, dtype=jnp.float32) / float(max(cols, 1))
    ch = jnp.asarray(canvas_height, dtype=jnp.float32) / float(max(rows, 1))

    xi = positions[:, 0]
    yi = positions[:, 1]
    xj = anchor_targets[:, 0]
    yj = anchor_targets[:, 1]

    dx = jnp.abs(xi - xj)
    dy = jnp.abs(yi - yj)
    bbox_w = dx + cw
    bbox_h = dy + ch
    demand = scaled_w * (dx + dy + 1e-6) / jnp.maximum(bbox_w * bbox_h, 1e-12)

    c0 = jnp.clip(jnp.floor(jnp.minimum(xi, xj) / jnp.maximum(cw, 1e-12)).astype(jnp.int32), 0, cols - 1)
    c1 = jnp.clip(jnp.floor(jnp.maximum(xi, xj) / jnp.maximum(cw, 1e-12)).astype(jnp.int32), 0, cols - 1)
    r0 = jnp.clip(jnp.floor(jnp.minimum(yi, yj) / jnp.maximum(ch, 1e-12)).astype(jnp.int32), 0, rows - 1)
    r1 = jnp.clip(jnp.floor(jnp.maximum(yi, yj) / jnp.maximum(ch, 1e-12)).astype(jnp.int32), 0, rows - 1)

    diff = jnp.zeros((rows + 1, cols + 1), dtype=jnp.float32)
    diff = diff.at[r0, c0].add(demand)
    diff = diff.at[r1 + 1, c0].add(-demand)
    diff = diff.at[r0, c1 + 1].add(-demand)
    diff = diff.at[r1 + 1, c1 + 1].add(demand)

    q = jnp.cumsum(jnp.cumsum(diff, axis=0), axis=1)[:rows, :cols]

    def _smooth(_, val):
        return _avg_pool3_same(val)

    q = lax.fori_loop(0, int(smoothing_passes), _smooth, q)
    mean_val = jnp.mean(q)
    q = jnp.where(mean_val > 1e-12, q / mean_val, q)
    return q.astype(jnp.float32)


@partial(jax.jit, static_argnames=("rows", "cols"))
def _build_anchor_source_kernel(
    positions: jnp.ndarray,
    anchor_targets: jnp.ndarray,
    point_weight: jnp.ndarray,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
) -> jnp.ndarray:
    src_from = _deposit_points_bilinear_kernel(positions, point_weight, canvas_width, canvas_height, rows, cols)
    src_to = _deposit_points_bilinear_kernel(anchor_targets, point_weight, canvas_width, canvas_height, rows, cols)
    source = src_from - src_to
    scale = jnp.mean(jnp.abs(source))
    source = jnp.where(scale > 1e-12, source / scale, source)
    return source.astype(jnp.float32)


@jax.jit
def _compute_net_force_kernel(
    positions: jnp.ndarray,
    edge_index: jnp.ndarray,
    edge_weight: jnp.ndarray,
) -> jnp.ndarray:
    i = edge_index[:, 0]
    j = edge_index[:, 1]
    d = positions[j] - positions[i]
    dist = jnp.sqrt(jnp.maximum(jnp.sum(d * d, axis=1), 1e-6))
    f = edge_weight[:, None] * d / dist[:, None]

    force = jnp.zeros_like(positions)
    force = force.at[i].add(f)
    force = force.at[j].add(-f)

    norm = jnp.linalg.norm(force, axis=1)
    denom = jnp.where(force.shape[0] >= 10, jnp.quantile(norm, 0.90), jnp.max(norm))
    force = jnp.where(denom > 1e-6, force / denom, force)
    return force.astype(jnp.float32)


def build_density_pressure(
    positions: np.ndarray,
    sizes: np.ndarray,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
    target_density: float,
    overflow_power: float,
) -> np.ndarray:
    return np.asarray(
        _build_density_pressure_kernel(
            _as_float32(positions),
            _as_float32(sizes),
            float(canvas_width),
            float(canvas_height),
            int(rows),
            int(cols),
            float(target_density),
            float(overflow_power),
        )
    )


def deposit_macro_density(
    positions: np.ndarray,
    sizes: np.ndarray,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
) -> np.ndarray:
    return np.asarray(
        _deposit_macro_density_kernel(
            _as_float32(positions),
            _as_float32(sizes),
            float(canvas_width),
            float(canvas_height),
            int(rows),
            int(cols),
        )
    )


def build_rudy_pressure(
    positions: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
    smoothing_passes: int = 1,
) -> np.ndarray:
    return np.asarray(
        _build_rudy_pressure_kernel(
            _as_float32(positions),
            _as_int32(edge_index),
            _as_float32(edge_weight),
            float(canvas_width),
            float(canvas_height),
            int(rows),
            int(cols),
            int(smoothing_passes),
        )
    )


def build_net_source(
    positions: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
) -> np.ndarray:
    return np.asarray(
        _build_net_source_kernel(
            _as_float32(positions),
            _as_int32(edge_index),
            _as_float32(edge_weight),
            float(canvas_width),
            float(canvas_height),
            int(rows),
            int(cols),
        )
    )


def build_anchor_pressure(
    positions: np.ndarray,
    anchor_targets: np.ndarray,
    anchor_weights: np.ndarray,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
    smoothing_passes: int = 1,
) -> Optional[np.ndarray]:
    weights = _as_float32(anchor_weights)
    mask = weights > 1e-8
    if not np.any(mask):
        return None

    ref_val = max(float(np.quantile(weights[mask], 0.90)), 1e-6)
    scaled_w = np.clip(np.sqrt(weights[mask] / ref_val), 0.0, 2.5).astype(np.float32)
    return np.asarray(
        _build_anchor_pressure_kernel(
            _as_float32(positions)[mask],
            _as_float32(anchor_targets)[mask],
            scaled_w,
            float(canvas_width),
            float(canvas_height),
            int(rows),
            int(cols),
            int(smoothing_passes),
        )
    )


def build_anchor_source(
    positions: np.ndarray,
    anchor_targets: np.ndarray,
    anchor_weights: np.ndarray,
    canvas_width: float,
    canvas_height: float,
    rows: int,
    cols: int,
) -> Optional[np.ndarray]:
    weights = _as_float32(anchor_weights)
    mask = weights > 1e-8
    if not np.any(mask):
        return None

    ref_val = max(float(np.quantile(weights[mask], 0.90)), 1e-6)
    point_weight = np.clip(np.sqrt(weights[mask] / ref_val), 0.0, 2.5).astype(np.float32)
    return np.asarray(
        _build_anchor_source_kernel(
            _as_float32(positions)[mask],
            _as_float32(anchor_targets)[mask],
            point_weight,
            float(canvas_width),
            float(canvas_height),
            int(rows),
            int(cols),
        )
    )


def compute_net_force(
    positions: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        _compute_net_force_kernel(
            _as_float32(positions),
            _as_int32(edge_index),
            _as_float32(edge_weight),
        )
    )
