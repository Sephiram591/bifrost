"""Exact 2D/2.5D Green's-theorem grid-line voxelizer for :mod:`bifrost.stack`.

Eliminates corner-column capacities, max-corner/edge enumerations, split-line
caps, and sloped-wall seam passes by integrating divergence-theorem fluxes
directly across simulation grid lines in physical box-centered micrometers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from fdtdx.materials import compute_ordered_names

from .geometry import _boundary_links, _edge_topology, has_tracers, run_recipe, split_t_junctions, to_numpy
from .subpixel import (
    Array,
    JUNCTION_HIGH,
    JUNCTION_LOW,
    from_six,
    homogenize_diagonal,
    homogenize_full,
    sparse_voxel_map,
    to_six,
)

_HIGHEST = jax.lax.Precision.HIGHEST
_BIG_UM = 1e6
_TINY = 1e-12
_FACE_THICKNESS = 0.1
_N_ARC_CHORDS = 48
_GL3_2 = (0.5 - 0.5 / math.sqrt(3.0), 0.5 + 0.5 / math.sqrt(3.0))
_NO_LAYER = -(2**30)
# Batched z-slice evaluation (``vmap`` over slabs of slices instead of a sequential ``lax.map`` with traced
# ``cond``s, which left the GPU launch-bound: ~70k sub-10 us kernels per forward write). The element budget
# bounds the per-slab working set (~4 bytes each).
_BATCHED_SLICES = True
_SLICE_BATCH_ELEMS = 1 << 28
# Batched slices split each offset boundary into pieces spanning a few cells and intersect each piece with
# only this many grid lines per axis (0 disables: every piece meets all nx + ny lines).
_RASTER_WINDOW = 8


def _st_clip(x: Array, lo: Any, hi: Any) -> Array:
    """Clamp primal values to ``[lo, hi]`` while preserving unclipped Leibniz AD derivatives."""
    return x + jax.lax.stop_gradient(jnp.clip(x, lo, hi) - x)


def _st_snap01(f: Array, eps: float) -> Array:
    """Snap primal floating-point residuals within ``eps`` of 0 and 1 without changing AD derivatives."""
    snapped = jnp.where(f <= eps, 0.0, jnp.where(f >= 1.0 - eps, 1.0, f))
    return f + jax.lax.stop_gradient(snapped - f)


# ---------------------------------------------------------------- grid coordinates (um)


def _sample_axis_bounds_um(device: Any, axis: int, shift: float) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper sample interval edges in um along ``axis`` in box-relative lateral coordinates."""
    res_um = float(device._res_um())
    frame = device.grid_frame
    n = int(device.grid_shape[axis])
    center = float(device.z_center) if axis == 2 else 0.0
    if frame is None:
        sa_shift = float(shift)
        lo = np.arange(n, dtype=np.float64)
        hi = lo + 1.0
        box_f = float(n)
    else:
        sa = frame.sample_axis(axis, shift)
        sa_shift = float(sa.shift)
        lo = np.asarray(sa.lo, dtype=np.float64)
        hi = np.asarray(sa.hi, dtype=np.float64)
        box_f = float(frame.edges[axis][-1])
    lo_um = center + (lo + sa_shift - 0.5 * box_f) * res_um
    hi_um = center + (hi + sa_shift - 0.5 * box_f) * res_um
    return lo_um, hi_um


def sample_grid_edges_um(
    device: Any,
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """Physical um sample intervals and grid metadata for ``device`` at Yee ``shift``.

    When ``edge_width == 1.0``, lateral sample intervals are contiguous and can be
    subdivided by ``xy_subsamples``. When ``edge_width != 1.0``, explicit non-contiguous
    ``[x_lo, x_hi] x [y_lo, y_hi]`` intervals are returned for exact box clipping.
    """
    res_um = float(device._res_um())
    x_lo, x_hi = _sample_axis_bounds_um(device, 0, shift[0])
    y_lo, y_hi = _sample_axis_bounds_um(device, 1, shift[1])
    z_lo, z_hi = _sample_axis_bounds_um(device, 2, shift[2])
    z_edges = np.concatenate([z_lo[:1], z_hi])
    n_sub = int(device.xy_subsamples)
    edge_width = float(device.edge_width)

    if edge_width == 1.0:
        x_base = np.concatenate([x_lo[:1], x_hi])
        y_base = np.concatenate([y_lo[:1], y_hi])
        if n_sub > 1:
            t = np.arange(n_sub, dtype=np.float64) / n_sub
            x_fine = np.concatenate([(x_lo[:, None] + (x_hi - x_lo)[:, None] * t[None, :]).ravel(), x_hi[-1:]])
            y_fine = np.concatenate([(y_lo[:, None] + (y_hi - y_lo)[:, None] * t[None, :]).ravel(), y_hi[-1:]])
        else:
            x_fine, y_fine = x_base, y_base
        return {
            "contiguous": True,
            "res_um": res_um,
            "x_edges": x_base,
            "y_edges": y_base,
            "x_fine": x_fine,
            "y_fine": y_fine,
            "z_edges": z_edges,
            "n_sub": n_sub,
            "dx": x_hi - x_lo,
            "dy": y_hi - y_lo,
            "dz": z_hi - z_lo,
        }

    def _expand_boxes(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mid = 0.5 * (lo + hi)
        w = edge_width * (hi - lo)
        if n_sub == 1:
            return mid - 0.5 * w, mid + 0.5 * w
        sub_w = w / n_sub
        idx = np.arange(n_sub, dtype=np.float64)
        s_lo = (mid - 0.5 * w)[:, None] + idx[None, :] * sub_w[:, None]
        s_hi = s_lo + sub_w[:, None]
        return s_lo.ravel(), s_hi.ravel()

    xf_lo, xf_hi = _expand_boxes(x_lo, x_hi)
    yf_lo, yf_hi = _expand_boxes(y_lo, y_hi)
    return {
        "contiguous": False,
        "non_overlapping_y": bool(edge_width <= 1.0 + 1e-12),
        "res_um": res_um,
        "x_lo": x_lo - 0.5 * (edge_width - 1.0) * (x_hi - x_lo),
        "x_hi": x_hi + 0.5 * (edge_width - 1.0) * (x_hi - x_lo),
        "y_lo": y_lo - 0.5 * (edge_width - 1.0) * (y_hi - y_lo),
        "y_hi": y_hi + 0.5 * (edge_width - 1.0) * (y_hi - y_lo),
        "xf_lo": xf_lo,
        "xf_hi": xf_hi,
        "yf_lo": yf_lo,
        "yf_hi": yf_hi,
        "z_edges": z_edges,
        "n_sub": n_sub,
        "dx": edge_width * (x_hi - x_lo),
        "dy": edge_width * (y_hi - y_lo),
        "dz": z_hi - z_lo,
    }


# ---------------------------------------------------------------- 2D Green's rasterizer


def _pool_subsamples(
    fill: Array,
    std_mx: Array,
    std_my: Array,
    b_stats: Array,
    x_edges_parent: Array,
    y_edges_parent: Array,
    n_sub: int,
) -> tuple[Array, Array, Array, Array]:
    """Pool `(n_sub*nx, n_sub*ny)` sub-square moments and boundary stats onto `(nx, ny)` parent cells."""
    if n_sub == 1:
        return fill, std_mx, std_my, b_stats
    nx = x_edges_parent.shape[0] - 1
    ny = y_edges_parent.shape[0] - 1
    dx_sub = (x_edges_parent[1:] - x_edges_parent[:-1]) / n_sub
    dy_sub = (y_edges_parent[1:] - y_edges_parent[:-1]) / n_sub
    s = n_sub
    offs = (jnp.arange(s, dtype=fill.dtype) + 0.5 - 0.5 * s) / s  # centered offset fraction in (-0.5, 0.5)
    f4 = fill.reshape(nx, s, ny, s)
    mx4 = std_mx.reshape(nx, s, ny, s)
    my4 = std_my.reshape(nx, s, ny, s)
    b4 = b_stats.reshape(nx, s, ny, s, 6)

    fill_out = jnp.mean(f4, axis=(1, 3))
    # int (x - xc_p) dA = sum_{a,b} [int (x - xc_{ab}) dA + (xc_{ab} - xc_p) * area_{ab}]
    # standardized moment in parent = mean_{a,b}(mx_{ab} / s + sqrt(12) * offs_x * f_{ab})
    sqrt12 = math.sqrt(12.0)
    dx_p = x_edges_parent[1:] - x_edges_parent[:-1]
    dy_p = y_edges_parent[1:] - y_edges_parent[:-1]
    mx_parent = jnp.mean(
        mx4 * (dx_sub[:, None, None, None] / dx_p[:, None, None, None])
        + sqrt12 * offs[None, :, None, None] * f4,
        axis=(1, 3),
    )
    my_parent = jnp.mean(
        my4 * (dy_sub[None, None, :, None] / dy_p[None, None, :, None])
        + sqrt12 * offs[None, None, None, :] * f4,
        axis=(1, 3),
    )
    b_out = jnp.sum(b4, axis=(1, 3))
    return fill_out, mx_parent, my_parent, b_out


def rasterize_segments_2d(
    p0: Array,
    p1: Array,
    valid: Array,
    x_edges: Any,
    y_edges: Any,
    res_um: float,
    *,
    need_moments: bool = True,
    seg_chunk: int = 128,
    window: tuple[int, int] | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Exact 2D Green's-theorem grid-line rasterizer on closed CCW boundary segments.

    Computes exact cell volume fraction, standardized first moments ``(std_mx, std_my)``
    matching :func:`bifrost.stack._cell_standard`, and boundary integral channels
    ``[ell, ell*nx, ell*ny, ell*nx^2, ell*ny^2, ell*nx*ny]`` in frame units (``um / res_um``).

    Args:
        p0: ``(E, 2)`` segment start points in ``um``.
        p1: ``(E, 2)`` segment end points in ``um`` (CCW: interior on left).
        valid: ``(E,)`` boolean mask of active segments.
        x_edges: ``(nx + 1,)`` ascending grid lines in ``um``.
        y_edges: ``(ny + 1,)`` ascending grid lines in ``um``.
        res_um: frame reference resolution in ``um`` for dimensionless boundary lengths.
        need_moments: whether to compute standardized moments and boundary dyad channels.
        seg_chunk: segment batch size bounding intermediate sort memory.
        window: ``(wx, wy)``: intersect each segment only with the ``wx`` x-lines and ``wy`` y-lines
            following its lower end, instead of every grid line (``O(E (wx + wy))`` instead of
            ``O(E (nx + ny))`` work). Exact only if no segment strictly contains more grid lines than
            that (see :func:`_max_line_crossings`); the caller must guarantee it.

    Returns:
        ``(fill, std_mx, std_my, b_stats)`` where ``fill, std_mx, std_my`` have shape
        ``(nx, ny)`` and ``b_stats`` has shape ``(nx, ny, 6)``.
    """
    dtype = p0.dtype
    calc = jnp.float64 if jax.config.jax_enable_x64 else dtype
    snap_eps = 1e-12 if calc == jnp.float64 else 1e-7
    if not has_tracers(x_edges) and not has_tracers(y_edges):
        xe_np = np.asarray(x_edges, dtype=np.float64)
        ye_np = np.asarray(y_edges, dtype=np.float64)
        nx = len(xe_np) - 1
        ny = len(ye_np) - 1
        ox = 0.5 * (xe_np[0] + xe_np[-1])
        oy = 0.5 * (ye_np[0] + ye_np[-1])
        xe = jnp.asarray(xe_np - ox, dtype=calc)
        ye = jnp.asarray(ye_np - oy, dtype=calc)
        shift_origin = jnp.asarray([ox, oy], dtype=calc)
    else:
        xe_j = jnp.asarray(x_edges, dtype=calc)
        ye_j = jnp.asarray(y_edges, dtype=calc)
        nx = xe_j.shape[0] - 1
        ny = ye_j.shape[0] - 1
        ox = 0.5 * (xe_j[0] + xe_j[-1])
        oy = 0.5 * (ye_j[0] + ye_j[-1])
        xe = xe_j - ox
        ye = ye_j - oy
        shift_origin = jnp.stack([ox, oy])
    q0 = jnp.asarray(p0, dtype=calc) - shift_origin
    q1 = jnp.asarray(p1, dtype=calc) - shift_origin
    act = jnp.asarray(valid, dtype=jnp.bool_)

    dx_grid = xe[1:] - xe[:-1]
    dy_grid = ye[1:] - ye[:-1]
    cell_area = dx_grid[:, None] * dy_grid[None, :]

    if q0.shape[0] == 0:
        z = jnp.zeros((nx, ny), dtype=dtype)
        return z, z, z, jnp.zeros((nx, ny, 6), dtype=dtype)

    x_max = xe[-1]
    y_min, y_max = ye[0], ye[-1]
    res_c = jnp.asarray(res_um, dtype=calc)
    eps30 = jnp.asarray(1e-30, dtype=calc)

    def _accumulate_chunk(carry: tuple[Array, ...], segs: tuple[Array, Array, Array]) -> tuple[tuple[Array, ...], None]:
        sum_dv, sum_udv, sum_v_dv, sum_u2_dv, sum_uv_dv, sum_b = carry
        c_p0, c_p1, c_act = segs
        d = c_p1 - c_p0
        dx_e, dy_e = d[:, 0], d[:, 1]
        len2 = dx_e * dx_e + dy_e * dy_e
        len_e = jnp.sqrt(jnp.maximum(len2, eps30))

        # Quick geometric cull: segments wholly right of x_max, below y_min, or >= y_max contribute 0
        seg_x_min = jnp.minimum(c_p0[:, 0], c_p1[:, 0])
        seg_y_min = jnp.minimum(c_p0[:, 1], c_p1[:, 1])
        seg_y_max = jnp.maximum(c_p0[:, 1], c_p1[:, 1])
        live = c_act & (len_e > _TINY) & (seg_x_min < x_max) & (seg_y_max > y_min) & (seg_y_min < y_max)

        def _do_chunk() -> tuple[Array, ...]:
            n_e = jnp.where(live[:, None], jnp.stack([dy_e, -dx_e], axis=-1) / len_e[:, None], 0.0)
            vert = jnp.abs(dx_e) <= _TINY
            horiz = jnp.abs(dy_e) <= _TINY
            safe_dx = jnp.where(vert, 1.0, dx_e)
            safe_dy = jnp.where(horiz, 1.0, dy_e)
            if window is None:
                x_lines = xe[None, :]
                y_lines = ye[None, :]
            else:
                # Only the lines following each segment's lower end; lines past its upper end clip to t = 1
                # and give zero-length pieces, so the window just has to cover the lines it crosses
                wx, wy = window
                ix0 = jnp.searchsorted(xe, jax.lax.stop_gradient(seg_x_min), side="right")
                iy0 = jnp.searchsorted(ye, jax.lax.stop_gradient(seg_y_min), side="right")
                x_lines = xe[jnp.clip(ix0[:, None] + jnp.arange(wx), 0, nx)]
                y_lines = ye[jnp.clip(iy0[:, None] + jnp.arange(wy), 0, ny)]
            tx = jnp.where(vert[:, None], 0.0, jnp.clip((x_lines - c_p0[:, 0:1]) / safe_dx[:, None], 0.0, 1.0))
            ty = jnp.where(horiz[:, None], 0.0, jnp.clip((y_lines - c_p0[:, 1:2]) / safe_dy[:, None], 0.0, 1.0))
            b_size = c_p0.shape[0]
            taus = jnp.concatenate(
                [jnp.zeros((b_size, 1), dtype=calc), jnp.ones((b_size, 1), dtype=calc), tx, ty],
                axis=-1,
            )
            tau = jnp.sort(taus, axis=-1)
            t0, t1 = tau[:, :-1], tau[:, 1:]
            dt = jnp.where(live[:, None], t1 - t0, 0.0)
            t_mid = 0.5 * (t0 + t1)
            mid_x = c_p0[:, 0:1] + t_mid * dx_e[:, None]
            mid_y = c_p0[:, 1:2] + t_mid * dy_e[:, None]

            ix = jnp.searchsorted(xe, jax.lax.stop_gradient(mid_x), side="right")  # 0..nx+1
            iy = jnp.searchsorted(ye, jax.lax.stop_gradient(mid_y), side="right") - 1  # -1..ny
            in_y = (iy >= 0) & (iy < ny) & (dt > 0)
            in_slab = in_y & (ix <= nx)
            in_cell = in_y & (ix >= 1) & (ix <= nx)

            slab_i = jnp.where(in_slab, ix, nx + 2)
            cell_i = jnp.where(in_cell, ix - 1, nx + 2)
            gather_i = jnp.clip(ix - 1, 0, nx - 1)
            cell_j = jnp.clip(iy, 0, ny - 1)

            x0_seg = c_p0[:, 0:1] + t0 * dx_e[:, None]
            x1_seg = c_p0[:, 0:1] + t1 * dx_e[:, None]
            y0_seg = c_p0[:, 1:2] + t0 * dy_e[:, None]
            y1_seg = c_p0[:, 1:2] + t1 * dy_e[:, None]

            x_base = xe[gather_i]
            y_base = ye[cell_j]
            u0 = x0_seg - x_base
            u1 = x1_seg - x_base
            v0 = y0_seg - y_base
            v1 = y1_seg - y_base
            dv = jnp.where(in_slab, v1 - v0, 0.0)
            udv = jnp.where(in_cell, 0.5 * (u0 + u1) * dv, 0.0)

            next_dv = sum_dv.at[slab_i, cell_j].add(dv.astype(calc), mode="drop")
            next_udv = sum_udv.at[cell_i, cell_j].add(udv.astype(calc), mode="drop")

            if need_moments:
                v_dv = jnp.where(in_slab, 0.5 * (v0 + v1) * dv, 0.0)
                u2_dv = jnp.where(in_cell, ((u0 * u0 + u0 * u1 + u1 * u1) / 6.0) * dv, 0.0)
                uv_dv = jnp.where(in_cell, ((u0 * v0 + u1 * v1) / 3.0 + (u0 * v1 + u1 * v0) / 6.0) * dv, 0.0)
                ell_f = jnp.where(in_cell, dt * len_e[:, None] / res_c, 0.0)
                n_ch = jnp.stack(
                    [
                        jnp.ones_like(n_e[:, 0]),
                        n_e[:, 0],
                        n_e[:, 1],
                        n_e[:, 0] ** 2,
                        n_e[:, 1] ** 2,
                        n_e[:, 0] * n_e[:, 1],
                    ],
                    axis=-1,
                )
                b_seg = ell_f[..., None] * n_ch[:, None, :]
                next_v_dv = sum_v_dv.at[slab_i, cell_j].add(v_dv.astype(calc), mode="drop")
                next_u2_dv = sum_u2_dv.at[cell_i, cell_j].add(u2_dv.astype(calc), mode="drop")
                next_uv_dv = sum_uv_dv.at[cell_i, cell_j].add(uv_dv.astype(calc), mode="drop")
                next_b = sum_b.at[cell_i, cell_j].add(b_seg.astype(calc), mode="drop")
                return (next_dv, next_udv, next_v_dv, next_u2_dv, next_uv_dv, next_b)
            return (next_dv, next_udv, sum_v_dv, sum_u2_dv, sum_uv_dv, sum_b)

        out = jax.lax.cond(jnp.any(live), _do_chunk, lambda: carry)
        return out, None

    init_carry = (
        jnp.zeros((nx + 1, ny), dtype=calc),
        jnp.zeros((nx, ny), dtype=calc),
        jnp.zeros((nx + 1, ny), dtype=calc),
        jnp.zeros((nx, ny), dtype=calc),
        jnp.zeros((nx, ny), dtype=calc),
        jnp.zeros((nx, ny, 6), dtype=calc),
    )

    n_segs = q0.shape[0]
    # Keep segment*line work per step roughly constant (~64k pairs), so windowed slices and small local
    # sub-grids (2x2 junction cells, 4x4 overlap pixels) run in a single flat pass without an inner scan
    eff_lines = (nx + ny + 4) if window is None else (window[0] + window[1] + 2)
    seg_chunk = int(max(seg_chunk, min(4096, 65536 // max(eff_lines, 1))))
    if n_segs <= seg_chunk:
        carry, _ = _accumulate_chunk(init_carry, (q0, q1, act))
    else:
        n_batches = math.ceil(n_segs / seg_chunk)
        pad = n_batches * seg_chunk - n_segs
        if pad > 0:
            q0_p = jnp.pad(q0, ((0, pad), (0, 0)))
            q1_p = jnp.pad(q1, ((0, pad), (0, 0)))
            act_p = jnp.pad(act, (0, pad), constant_values=False)
        else:
            q0_p, q1_p, act_p = q0, q1, act
        carry, _ = jax.lax.scan(
            jax.checkpoint(_accumulate_chunk),
            init_carry,
            (
                q0_p.reshape(n_batches, seg_chunk, 2),
                q1_p.reshape(n_batches, seg_chunk, 2),
                act_p.reshape(n_batches, seg_chunk),
            ),
        )

    sum_dv, sum_udv, sum_v_dv, sum_u2_dv, sum_uv_dv, sum_b = carry
    tau_right = (-jnp.cumsum(sum_dv, axis=0))[1:, :]
    area_raw = dx_grid[:, None] * tau_right + sum_udv
    if calc != jnp.float64:
        big = jnp.asarray(_BIG_UM, dtype=calc)
        s_x_min = jnp.min(jnp.where(act, jnp.minimum(q0[:, 0], q1[:, 0]), big))
        s_x_max = jnp.max(jnp.where(act, jnp.maximum(q0[:, 0], q1[:, 0]), -big))
        s_y_min = jnp.min(jnp.where(act, jnp.minimum(q0[:, 1], q1[:, 1]), big))
        s_y_max = jnp.max(jnp.where(act, jnp.maximum(q0[:, 1], q1[:, 1]), -big))
        out_bb = (xe[1:, None] <= s_x_min) | (xe[:-1, None] >= s_x_max) | (ye[None, 1:] <= s_y_min) | (ye[None, :-1] >= s_y_max)
        area_raw = area_raw + jax.lax.stop_gradient(jnp.where(out_bb, -area_raw, 0.0))
    area = _st_clip(area_raw, 0.0, cell_area)
    fill = _st_snap01(area / cell_area, snap_eps)

    if not need_moments:
        z = jnp.zeros((nx, ny), dtype=dtype)
        return fill.astype(dtype), z, z, jnp.zeros((nx, ny, 6), dtype=dtype)

    sqrt12 = math.sqrt(12.0)
    mx_raw = 0.5 * (dx_grid[:, None] ** 2) * tau_right + sum_u2_dv - 0.5 * dx_grid[:, None] * area_raw
    mu_right = (-jnp.cumsum(sum_v_dv, axis=0))[1:, :]
    my_raw = dx_grid[:, None] * mu_right + sum_uv_dv - 0.5 * dy_grid[None, :] * area_raw
    std_mx = (sqrt12 / dx_grid[:, None]) * (mx_raw / cell_area)
    std_my = (sqrt12 / dy_grid[None, :]) * (my_raw / cell_area)
    # Clamp tiny floating-point overshoot on nearly full/empty voxels without zeroing AD
    lim = 0.5 * sqrt12
    std_mx = _st_clip(std_mx, -lim, lim)
    std_my = _st_clip(std_my, -lim, lim)
    return fill.astype(dtype), std_mx.astype(dtype), std_my.astype(dtype), sum_b.astype(dtype)


def _split_segments(p0: Array, p1: Array, valid: Array, counts: np.ndarray) -> tuple[Array, Array, Array]:
    """Split segment ``e`` into ``counts[e]`` equal collinear pieces (static counts; Green's integrals unchanged)."""
    counts = np.asarray(counts, dtype=np.int64)
    if counts.shape[0] > p0.shape[0]:
        counts = counts[: p0.shape[0]]  # a zero (concrete) offset returns the edges without their arc chords
    if counts.shape[0] != p0.shape[0]:
        raise ValueError(f"split counts {counts.shape[0]} != segments {p0.shape[0]}")
    if np.all(counts == 1):
        return p0, p1, valid
    e_idx = np.repeat(np.arange(len(counts)), counts)
    k = np.concatenate([np.arange(c) for c in counts])
    c = counts[e_idx].astype(np.float64)
    t0 = jnp.asarray(k / c, dtype=p0.dtype)[:, None]
    t1 = jnp.asarray((k + 1) / c, dtype=p0.dtype)[:, None]
    a, b = p0[e_idx], p1[e_idx]
    d = b - a
    q0 = a + t0 * d
    q1 = jnp.where((k + 1 == counts[e_idx])[:, None], b, a + t1 * d)  # last piece ends exactly on b
    return q0, q1, valid[e_idx]


def _max_line_crossings(p0: Array, p1: Array, valid: Array, x_edges: Array, y_edges: Array) -> tuple[Array, Array]:
    """Largest number of x- and y-grid lines strictly inside any valid segment's extent."""
    xe = jax.lax.stop_gradient(jnp.asarray(x_edges))
    ye = jax.lax.stop_gradient(jnp.asarray(y_edges))
    a = jax.lax.stop_gradient(p0).astype(xe.dtype)
    b = jax.lax.stop_gradient(p1).astype(xe.dtype)
    cx = jnp.searchsorted(xe, jnp.maximum(a[:, 0], b[:, 0]), side="left") - jnp.searchsorted(
        xe, jnp.minimum(a[:, 0], b[:, 0]), side="right"
    )
    cy = jnp.searchsorted(ye, jnp.maximum(a[:, 1], b[:, 1]), side="left") - jnp.searchsorted(
        ye, jnp.minimum(a[:, 1], b[:, 1]), side="right"
    )
    return jnp.max(jnp.where(valid, cx, 0)), jnp.max(jnp.where(valid, cy, 0))


def _clip_segments_to_box_x(
    p0: Array,
    p1: Array,
    valid: Array,
    x_lo: Array,
    x_hi: Array,
    y_edges: Array,
    res_um: float,
    need_moments: bool,
) -> tuple[Array, Array, Array, Array]:
    """Exact Green's integral for a single column strip ``[x_lo, x_hi]`` over all ``y`` boxes."""
    dtype = p0.dtype
    calc = jnp.float64 if jax.config.jax_enable_x64 else dtype
    snap_eps = 1e-12 if calc == jnp.float64 else 1e-7
    q0 = jnp.asarray(p0, dtype=calc)
    q1 = jnp.asarray(p1, dtype=calc)
    xl = jnp.asarray(x_lo, dtype=calc)
    xh = jnp.asarray(x_hi, dtype=calc)
    ye = jnp.asarray(y_edges, dtype=calc)
    res_c = jnp.asarray(res_um, dtype=calc)
    ny = ye.shape[0] - 1
    dx = xh - xl
    dy_grid = ye[1:] - ye[:-1]
    cell_area = dx * dy_grid[None, :]

    d = q1 - q0
    dx_e, dy_e = d[:, 0], d[:, 1]
    len_e = jnp.sqrt(jnp.maximum(dx_e * dx_e + dy_e * dy_e, jnp.asarray(1e-30, dtype=calc)))
    live = valid & (len_e > _TINY)
    n_e = jnp.where(live[:, None], jnp.stack([dy_e, -dx_e], axis=-1) / len_e[:, None], 0.0)

    vert = jnp.abs(dx_e) <= _TINY
    horiz = jnp.abs(dy_e) <= _TINY
    safe_dx = jnp.where(vert, 1.0, dx_e)
    safe_dy = jnp.where(horiz, 1.0, dy_e)
    tx = jnp.where(
        vert[:, None],
        0.0,
        jnp.clip((jnp.stack([xl, xh])[None, :] - q0[:, 0:1]) / safe_dx[:, None], 0.0, 1.0),
    )
    ty = jnp.where(horiz[:, None], 0.0, jnp.clip((ye[None, :] - q0[:, 1:2]) / safe_dy[:, None], 0.0, 1.0))
    taus = jnp.concatenate([jnp.zeros((q0.shape[0], 1), dtype=calc), jnp.ones((q0.shape[0], 1), dtype=calc), tx, ty], -1)
    tau = jnp.sort(taus, axis=-1)
    t0, t1 = tau[:, :-1], tau[:, 1:]
    dt = jnp.where(live[:, None], t1 - t0, 0.0)
    t_mid = 0.5 * (t0 + t1)
    mid_x = q0[:, 0:1] + t_mid * dx_e[:, None]
    mid_y = q0[:, 1:2] + t_mid * dy_e[:, None]
    iy = jnp.searchsorted(ye, jax.lax.stop_gradient(mid_y), side="right") - 1
    sg_x = jax.lax.stop_gradient(mid_x)
    in_y = (iy >= 0) & (iy < ny) & (dt > 0)
    left_or_in = in_y & (sg_x < xh)
    in_box = in_y & (sg_x >= xl) & (sg_x < xh)
    gather_j = jnp.clip(iy, 0, ny - 1)
    slab_j = jnp.where(left_or_in, iy, ny + 2)
    box_j = jnp.where(in_box, iy, ny + 2)

    u0 = (q0[:, 0:1] + t0 * dx_e[:, None]) - xl
    u1 = (q0[:, 0:1] + t1 * dx_e[:, None]) - xl
    y_base = ye[gather_j]
    v0 = (q0[:, 1:2] + t0 * dy_e[:, None]) - y_base
    v1 = (q0[:, 1:2] + t1 * dy_e[:, None]) - y_base

    dv_right = jnp.where(left_or_in, v1 - v0, 0.0)
    tau_right = -jnp.zeros((ny,), dtype=calc).at[slab_j].add(dv_right.astype(calc), mode="drop")
    dv_in = jnp.where(in_box, v1 - v0, 0.0)
    udv = jnp.zeros((ny,), dtype=calc).at[box_j].add((0.5 * (u0 + u1) * dv_in).astype(calc), mode="drop")
    area_raw = dx * tau_right + udv
    area = _st_clip(area_raw, 0.0, cell_area[0])
    safe_dy = jnp.where(dy_grid > 0, dy_grid, 1.0)
    safe_area = jnp.where(cell_area[0] > 0, cell_area[0], 1.0)
    fill = _st_snap01(jnp.where(cell_area[0] > 0, area / safe_area, 0.0), snap_eps)

    if not need_moments:
        z = jnp.zeros((ny,), dtype=dtype)
        return fill.astype(dtype), z, z, jnp.zeros((ny, 6), dtype=dtype)

    sqrt12 = math.sqrt(12.0)
    v_dv_right = jnp.where(left_or_in, 0.5 * (v0 + v1) * (v1 - v0), 0.0)
    mu_right = -jnp.zeros((ny,), dtype=calc).at[slab_j].add(v_dv_right.astype(calc), mode="drop")
    u2_dv = jnp.zeros((ny,), dtype=calc).at[box_j].add((((u0 * u0 + u0 * u1 + u1 * u1) / 6.0) * dv_in).astype(calc), mode="drop")
    uv_dv = jnp.zeros((ny,), dtype=calc).at[box_j].add(
        (((u0 * v0 + u1 * v1) / 3.0 + (u0 * v1 + u1 * v0) / 6.0) * dv_in).astype(calc), mode="drop"
    )
    mx_raw = 0.5 * dx**2 * tau_right + u2_dv - 0.5 * dx * area_raw
    my_raw = dx * mu_right + uv_dv - 0.5 * dy_grid * area_raw
    std_mx = jnp.where(cell_area[0] > 0, _st_clip((sqrt12 / dx) * (mx_raw / safe_area), -0.5 * sqrt12, 0.5 * sqrt12), 0.0)
    std_my = jnp.where(cell_area[0] > 0, _st_clip((sqrt12 / safe_dy) * (my_raw / safe_area), -0.5 * sqrt12, 0.5 * sqrt12), 0.0)
    ell_f = jnp.where(in_box, dt * len_e[:, None] / res_c, 0.0)
    n_ch = jnp.stack(
        [
            jnp.ones_like(n_e[:, 0]),
            n_e[:, 0],
            n_e[:, 1],
            n_e[:, 0] ** 2,
            n_e[:, 1] ** 2,
            n_e[:, 0] * n_e[:, 1],
        ],
        axis=-1,
    )
    b_stats = jnp.zeros((ny, 6), dtype=calc).at[box_j].add((ell_f[..., None] * n_ch[:, None, :]).astype(calc), mode="drop")
    return fill.astype(dtype), std_mx.astype(dtype), std_my.astype(dtype), b_stats.astype(dtype)


def rasterize_noncontiguous_boxes_2d(
    p0: Array,
    p1: Array,
    valid: Array,
    x_lo: Any,
    x_hi: Any,
    y_lo: Any,
    y_hi: Any,
    res_um: float,
    *,
    need_moments: bool = True,
    non_overlapping_y: bool | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Exact Green's rasterizer for possibly overlapping/non-contiguous box grids (``edge_width != 1.0``)."""
    dtype = p0.dtype
    calc = jnp.float64 if jax.config.jax_enable_x64 else dtype
    if not any(has_tracers(a) for a in (x_lo, x_hi, y_lo, y_hi)):
        xl = np.asarray(x_lo, dtype=np.float64)
        xh = np.asarray(x_hi, dtype=np.float64)
        yl = np.asarray(y_lo, dtype=np.float64)
        yh = np.asarray(y_hi, dtype=np.float64)
        ox = 0.5 * (xl[0] + xh[-1])
        oy = 0.5 * (yl[0] + yh[-1])
        q0 = jnp.asarray(p0, dtype=calc) - jnp.asarray([ox, oy], dtype=calc)
        q1 = jnp.asarray(p1, dtype=calc) - jnp.asarray([ox, oy], dtype=calc)
        xl_j = jnp.asarray(xl - ox, dtype=calc)
        xh_j = jnp.asarray(xh - ox, dtype=calc)
        yl_j = jnp.asarray(yl - oy, dtype=calc)
        yh_j = jnp.asarray(yh - oy, dtype=calc)
        non_ov_y = bool(np.all(yh[:-1] <= yl[1:] + 1e-12)) if non_overlapping_y is None else bool(non_overlapping_y)
    else:
        xl_r = jnp.asarray(x_lo, dtype=calc)
        xh_r = jnp.asarray(x_hi, dtype=calc)
        yl_r = jnp.asarray(y_lo, dtype=calc)
        yh_r = jnp.asarray(y_hi, dtype=calc)
        ox = 0.5 * (xl_r[0] + xh_r[-1])
        oy = 0.5 * (yl_r[0] + yh_r[-1])
        shift_o = jnp.stack([ox, oy])
        q0 = jnp.asarray(p0, dtype=calc) - shift_o
        q1 = jnp.asarray(p1, dtype=calc) - shift_o
        xl_j = xl_r - ox
        xh_j = xh_r - ox
        yl_j = yl_r - oy
        yh_j = yh_r - oy
        non_ov_y = bool(non_overlapping_y) if non_overlapping_y is not None else False
    act = jnp.asarray(valid, dtype=jnp.bool_)
    ny = yl_j.shape[0]
    if non_ov_y:
        # Interleave [y_lo, y_hi] into a 2*ny non-decreasing edge array, then keep even slices
        y_inter = jnp.stack([yl_j, yh_j], axis=1).reshape(-1)
        ye_j = (
            jnp.concatenate([y_inter[:1], jax.lax.cummax(y_inter[1:])])
            if ny > 0 and y_inter.shape[0] > 1
            else y_inter
        )

        def _col_step(xy_i: tuple[Array, Array]) -> tuple[Array, Array, Array, Array]:
            f, mx, my, bs = _clip_segments_to_box_x(q0, q1, act, xy_i[0], xy_i[1], ye_j, res_um, need_moments)
            return f[0::2], mx[0::2], my[0::2], bs[0::2]

        out = jax.lax.map(_col_step, (xl_j, xh_j))
        return jax.tree.map(lambda a: a.astype(dtype), out)

    # General overlapping y boxes (edge_width > 1): process per y_j strip via lax.map
    def _row_step(yy_j: tuple[Array, Array]) -> tuple[Array, Array, Array, Array]:
        y2 = jnp.stack([yy_j[0], yy_j[1]])

        def _col_inner(xx_i: tuple[Array, Array]) -> tuple[Array, Array, Array, Array]:
            f, mx, my, bs = _clip_segments_to_box_x(q0, q1, act, xx_i[0], xx_i[1], y2, res_um, need_moments)
            return f[0], mx[0], my[0], bs[0]

        return jax.lax.map(_col_inner, (xl_j, xh_j))

    out = jax.lax.map(_row_step, (yl_j, yh_j))  # each (ny, nx, ...)
    return jax.tree.map(lambda a: jnp.swapaxes(a, 0, 1).astype(dtype), out)


# ---------------------------------------------------------------- boundary offsetting & SDF fallback


def _halfplane_box_moments(
    signed_d: Array,
    nx_u: Array,
    ny_u: Array,
    hx: Array,
    hy: Array,
    res_um: Array,
) -> tuple[Array, Array, Array, Array]:
    """Exact clip of half-plane ``nx*u + ny*v + signed_d <= 0`` against box ``[-hx, hx] x [-hy, hy]``.

    Returns ``(fill, std_mx, std_my, ell_frame)``.
    """
    dtype = signed_d.dtype
    hx, hy = jnp.broadcast_arrays(jnp.asarray(hx, dtype=dtype), jnp.asarray(hy, dtype=dtype))
    # Clip unit box [-hx, hx] x [-hy, hy] CCW vertices in um by linear inequality
    rect = jnp.stack(
        [
            jnp.stack([-hx, -hy], axis=-1),
            jnp.stack([hx, -hy], axis=-1),
            jnp.stack([hx, hy], axis=-1),
            jnp.stack([-hx, hy], axis=-1),
        ],
        axis=-2,
    )  # (..., 4, 2)
    n_v = jnp.stack([nx_u, ny_u], axis=-1)
    phi = jnp.einsum("...vi,...i->...v", rect, n_v, precision=_HIGHEST) + signed_d[..., None]
    phi_next = jnp.roll(phi, -1, axis=-1)
    p_cur = rect
    p_nxt = jnp.roll(rect, -1, axis=-2)
    in_cur = phi <= 0
    in_nxt = phi_next <= 0
    cross_edge = in_cur ^ in_nxt
    denom = phi - phi_next
    safe_d = jnp.where(jnp.abs(denom) > _TINY, denom, 1.0)
    t_cross = jnp.where(cross_edge, jnp.clip(phi / safe_d, 0.0, 1.0), 0.0)
    p_cross = p_cur + t_cross[..., None] * (p_nxt - p_cur)

    # Exact 5-segment Green's boundary integral over rect intersect {phi <= 0}
    ta = jnp.where(in_cur, 0.0, jnp.where(in_nxt, t_cross, 0.0))
    tb = jnp.where(in_nxt, 1.0, jnp.where(in_cur, t_cross, 0.0))
    seg_a_box = p_cur + ta[..., None] * (p_nxt - p_cur)
    seg_b_box = p_cur + tb[..., None] * (p_nxt - p_cur)
    exit_mask = in_cur & ~in_nxt
    enter_mask = (~in_cur) & in_nxt
    q_exit = jnp.sum(jnp.where(exit_mask[..., None], p_cross, 0.0), axis=-2)
    q_enter = jnp.sum(jnp.where(enter_mask[..., None], p_cross, 0.0), axis=-2)
    seg_a = jnp.concatenate([seg_a_box, q_exit[..., None, :]], axis=-2)
    seg_b = jnp.concatenate([seg_b_box, q_enter[..., None, :]], axis=-2)

    x0, y0 = seg_a[..., 0], seg_a[..., 1]
    x1, y1 = seg_b[..., 0], seg_b[..., 1]
    cross2 = x0 * y1 - x1 * y0
    area_box = 4.0 * hx * hy
    area = _st_clip(0.5 * jnp.sum(cross2, axis=-1), 0.0, area_box)
    fill = area / area_box
    ix = jnp.sum(cross2 * (x0 + x1), axis=-1) / 6.0
    iy = jnp.sum(cross2 * (y0 + y1), axis=-1) / 6.0
    sqrt12 = math.sqrt(12.0)
    std_mx = _st_clip((sqrt12 / (2.0 * hx)) * (ix / area_box), -0.5 * sqrt12, 0.5 * sqrt12)
    std_my = _st_clip((sqrt12 / (2.0 * hy)) * (iy / area_box), -0.5 * sqrt12, 0.5 * sqrt12)

    has_cut = jnp.any(in_cur, axis=-1) & jnp.any(~in_cur, axis=-1)
    cut_d2 = jnp.sum((q_enter - q_exit) ** 2, axis=-1)
    ell = jnp.where(has_cut & (cut_d2 > 1e-24), jnp.sqrt(jnp.maximum(cut_d2, jnp.asarray(1e-24, dtype=dtype))), 0.0)
    return fill, std_mx, std_my, ell / res_um


def _eval_expr_sdf_um(
    expr: tuple,
    sdf_map: dict[int, tuple[Array, Array, Array]],
    shape: tuple[int, ...],
) -> tuple[Array, Array, Array]:
    """Evaluate boolean ``expr`` tree in physical ``um`` given per-source ``(d_um, nx, ny)``."""

    def where(cond: Array, u: tuple[Array, Array, Array], v: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        return tuple(jnp.where(cond, ui, vi) for ui, vi in zip(u, v, strict=True))  # type: ignore[return-value]

    dtype = next(iter(sdf_map.values()))[0].dtype if sdf_map else jnp.float32
    zero = jnp.zeros(shape, dtype=dtype)
    kind = expr[0]
    if kind == "fill":
        out, grow = (jnp.full(shape, -_BIG_UM, dtype=dtype), zero, zero), float(expr[1])
    elif kind == "empty":
        out, grow = (jnp.full(shape, _BIG_UM, dtype=dtype), zero, zero), 0.0
    elif kind == "layer":
        out, grow = sdf_map.get(expr[1], (jnp.full(shape, _BIG_UM, dtype=dtype), zero, zero)), float(expr[2])
    else:
        u = _eval_expr_sdf_um(expr[1], sdf_map, shape)
        v = _eval_expr_sdf_um(expr[2], sdf_map, shape)
        if kind == "or":
            out = where(u[0] <= v[0], u, v)
        elif kind == "and":
            out = where(u[0] >= v[0], u, v)
        elif kind == "not":
            out = where(u[0] >= -v[0], u, (-v[0], -v[1], -v[2]))
        elif kind == "xor":
            or_uv = where(u[0] <= v[0], u, v)
            and_uv = where(u[0] >= v[0], u, v)
            out = where(or_uv[0] >= -and_uv[0], or_uv, (-and_uv[0], -and_uv[1], -and_uv[2]))
        else:
            raise ValueError(f"Unknown layer operation {kind!r}")
        grow = float(expr[3])
    return (out[0] - jnp.asarray(grow, dtype=dtype), out[1], out[2]) if grow else out


def _sdf_and_grad(
    pts: Array,
    v_um: Array,
    nxt: np.ndarray,
    ids: np.ndarray,
    boundary: np.ndarray,
) -> tuple[Array, Array, Array]:
    """Analytical signed distance and spatial outward gradient wrt ``pts`` without nested AD."""
    dtype = pts.dtype
    p_cnt = pts.shape[0]
    e_cnt = v_um.shape[0]
    if e_cnt == 0:
        z = jnp.zeros((p_cnt,), dtype=dtype)
        return jnp.full((p_cnt,), _BIG_UM, dtype=dtype), z, z

    a = jnp.asarray(v_um, dtype=dtype)
    b = a[nxt]
    ab = b - a
    len2 = jnp.sum(ab * ab, axis=-1)
    len_e = jnp.sqrt(jnp.maximum(len2, jnp.asarray(1e-24, dtype=dtype)))
    ids_arr = np.asarray(ids, dtype=np.int32)
    num_polys = int(ids_arr.max()) + 1
    ids_j = jnp.asarray(ids_arr)
    cross = jax.lax.stop_gradient(a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1])
    poly_area2 = jax.ops.segment_sum(cross, ids_j, num_segments=num_polys)
    orient = jnp.where(poly_area2[ids_j] >= 0, jnp.asarray(1.0, dtype=dtype), jnp.asarray(-1.0, dtype=dtype))
    n_out = orient[:, None] * jnp.stack([ab[:, 1], -ab[:, 0]], axis=-1) / len_e[:, None]
    bnd_j = jnp.asarray(boundary, dtype=jnp.bool_)

    px = pts[None, :, 0]
    py = pts[None, :, 1]
    inf = jnp.asarray(jnp.inf, dtype=dtype)

    def _chunk_scan(chunk_sz: int):
        n_batches = math.ceil(e_cnt / chunk_sz)
        pad = n_batches * chunk_sz - e_cnt
        a_p = jnp.pad(a, ((0, pad), (0, 0))) if pad else a
        b_p = jnp.pad(b, ((0, pad), (0, 0))) if pad else b
        ab_p = jnp.pad(ab, ((0, pad), (0, 0))) if pad else ab
        len2_p = jnp.pad(len2, (0, pad), constant_values=1.0) if pad else len2
        n_p = jnp.pad(n_out, ((0, pad), (0, 0))) if pad else n_out
        bnd_p = jnp.pad(bnd_j, (0, pad), constant_values=False) if pad else bnd_j
        ids_p = jnp.pad(ids_j, (0, pad), constant_values=0) if pad else ids_j

        def step(carry, batch):
            d2_best, r_best, n_best, wind_acc = carry
            ca, cb, cab, clen2, cn, cbnd, cids = batch
            ap = pts[None, :, :] - ca[:, None, :]
            t = jnp.clip(jnp.sum(ap * cab[:, None, :], axis=-1) / jnp.maximum(clen2[:, None], jnp.asarray(1e-24, dtype=dtype)), 0.0, 1.0)
            r = ap - t[..., None] * cab[:, None, :]
            d2 = jnp.sum(r * r, axis=-1)
            d2_m = jnp.where(cbnd[:, None], d2, inf)
            idx_min = jnp.argmin(d2_m, axis=0)
            p_rng = jnp.arange(p_cnt)
            d2_c = d2_m[idx_min, p_rng]
            r_c = r[idx_min, p_rng]
            n_c = cn[idx_min]
            take = d2_c < d2_best
            d2_best = jnp.where(take, d2_c, d2_best)
            r_best = jnp.where(take[:, None], r_c, r_best)
            n_best = jnp.where(take[:, None], n_c, n_best)

            ay = ca[:, 1:2]
            by = cb[:, 1:2]
            up = (ay <= py) & (by > py)
            dn = (by <= py) & (ay > py)
            cx = (cb[:, 0:1] - ca[:, 0:1]) * (py - ay) - (px - ca[:, 0:1]) * (by - ay)
            dw = jnp.where(
                up & (cx > 0),
                jnp.ones((), dtype=jnp.int32),
                jnp.where(dn & (cx < 0), -jnp.ones((), dtype=jnp.int32), jnp.zeros((), dtype=jnp.int32)),
            )
            wind_c = jax.ops.segment_sum(dw, cids, num_segments=num_polys).astype(jnp.int32)
            return (d2_best, r_best, n_best, wind_acc + wind_c), None

        init = (
            jnp.full((p_cnt,), inf, dtype=dtype),
            jnp.zeros((p_cnt, 2), dtype=dtype),
            jnp.zeros((p_cnt, 2), dtype=dtype),
            jnp.zeros((num_polys, p_cnt), dtype=jnp.int32),
        )
        batches = (
            a_p.reshape(n_batches, chunk_sz, 2),
            b_p.reshape(n_batches, chunk_sz, 2),
            ab_p.reshape(n_batches, chunk_sz, 2),
            len2_p.reshape(n_batches, chunk_sz),
            n_p.reshape(n_batches, chunk_sz, 2),
            bnd_p.reshape(n_batches, chunk_sz),
            ids_p.reshape(n_batches, chunk_sz),
        )
        if n_batches == 1:
            (d2_min, r_min, n_min, wind_poly), _ = step(init, jax.tree.map(lambda x: x[0], batches))
        else:
            (d2_min, r_min, n_min, wind_poly), _ = jax.lax.scan(step, init, batches)
        return d2_min, r_min, n_min, wind_poly

    d2_min, r_min, n_min, wind_poly = _chunk_scan(min(32, e_cnt))
    inside = jnp.any(wind_poly != 0, axis=0)
    sign = jnp.where(inside, jnp.asarray(-1.0, dtype=dtype), jnp.asarray(1.0, dtype=dtype))
    dist = jnp.sqrt(jnp.maximum(d2_min, jnp.asarray(1e-24, dtype=dtype)))
    sd = sign * dist
    on_edge = d2_min <= jnp.asarray(1e-16, dtype=dtype)
    grad = jnp.where(on_edge[:, None], n_min, (sign[:, None] * r_min) / dist[:, None])
    return sd, grad[:, 0], grad[:, 1]


def rasterize_sdf_subgrid(
    expr: tuple,
    source_edges_um: dict[int, tuple[Array, np.ndarray, np.ndarray, np.ndarray]],
    grow_um: Any,
    x_lo_um: Any,
    x_hi_um: Any,
    y_lo_um: Any,
    y_hi_um: Any,
    res_um: float,
    *,
    sub_n: int = 4,
    dtype: Any = jnp.float32,
) -> tuple[Array, Array, Array, Array]:
    """Smooth SDF + ``sub_n x sub_n`` half-plane box clip fallback for collapsed shrinks or composites."""
    nx = x_lo_um.shape[0] if hasattr(x_lo_um, "shape") else len(x_lo_um)
    ny = y_lo_um.shape[0] if hasattr(y_lo_um, "shape") else len(y_lo_um)
    if not any(has_tracers(a) for a in (x_lo_um, x_hi_um, y_lo_um, y_hi_um)):
        xl_np = np.asarray(x_lo_um, dtype=np.float64)
        xh_np = np.asarray(x_hi_um, dtype=np.float64)
        yl_np = np.asarray(y_lo_um, dtype=np.float64)
        yh_np = np.asarray(y_hi_um, dtype=np.float64)
        ox = 0.5 * (xl_np[0] + xh_np[-1])
        oy = 0.5 * (yl_np[0] + yh_np[-1])
        origin = jnp.asarray([ox, oy], dtype=jnp.float64 if jax.config.jax_enable_x64 else dtype)
        cx_j = jnp.asarray(0.5 * (xl_np + xh_np) - ox, dtype=dtype)
        cy_j = jnp.asarray(0.5 * (yl_np + yh_np) - oy, dtype=dtype)
        dx_j = jnp.asarray(xh_np - xl_np, dtype=dtype)
        dy_j = jnp.asarray(yh_np - yl_np, dtype=dtype)
    else:
        c_dt = jnp.asarray(x_lo_um).dtype
        xl_j = jnp.asarray(x_lo_um, dtype=c_dt)
        xh_j = jnp.asarray(x_hi_um, dtype=c_dt)
        yl_j = jnp.asarray(y_lo_um, dtype=c_dt)
        yh_j = jnp.asarray(y_hi_um, dtype=c_dt)
        ox = 0.5 * (xl_j[0] + xh_j[-1])
        oy = 0.5 * (yl_j[0] + yh_j[-1])
        origin = jnp.stack([ox, oy])
        cx_j = (0.5 * (xl_j + xh_j) - ox).astype(dtype)
        cy_j = (0.5 * (yl_j + yh_j) - oy).astype(dtype)
        dx_j = (xh_j - xl_j).astype(dtype)
        dy_j = (yh_j - yl_j).astype(dtype)
    grow_arr = jnp.asarray(grow_um, dtype=dtype)

    shifted_sources = {
        idx: ((jnp.asarray(v_um, dtype=origin.dtype) - origin).astype(dtype), nxt, ids, boundary)
        for idx, (v_um, nxt, ids, boundary) in source_edges_um.items()
    }
    xx, yy = jnp.meshgrid(cx_j, cy_j, indexing="ij")
    pts = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

    def _eval_points(p_xy: Array) -> tuple[Array, Array, Array]:
        s_map: dict[int, tuple[Array, Array, Array]] = {}
        for idx, (v_s, nxt, ids, bnd) in shifted_sources.items():
            s_map[idx] = _sdf_and_grad(p_xy, v_s, nxt, ids, bnd)
        d0, gx0, gy0 = _eval_expr_sdf_um(expr, s_map, (p_xy.shape[0],))
        return d0 - grow_arr, gx0, gy0

    d_center, gx_center, gy_center = _eval_points(pts)
    d_grid = d_center.reshape(nx, ny)
    res_j = jnp.asarray(res_um, dtype=dtype)
    if sub_n == 1:
        gx_g = gx_center.reshape(nx, ny)
        gy_g = gy_center.reshape(nx, ny)
        norm = jnp.sqrt(jnp.maximum(gx_g * gx_g + gy_g * gy_g, jnp.asarray(1e-24, dtype=dtype)))
        nx_u = jnp.where(norm > 1e-9, gx_g / norm, jnp.ones((), dtype=dtype))
        ny_u = jnp.where(norm > 1e-9, gy_g / norm, jnp.zeros((), dtype=dtype))
        f_c, mx_c, my_c, ell_c = _halfplane_box_moments(
            d_grid, nx_u, ny_u, 0.5 * dx_j[:, None], 0.5 * dy_j[None, :], res_j
        )
        ch_c = ell_c[..., None] * jnp.stack(
            [jnp.ones_like(nx_u), nx_u, ny_u, nx_u * nx_u, ny_u * ny_u, nx_u * ny_u],
            axis=-1,
        )
        return f_c, mx_c, my_c, ch_c

    diag = jnp.hypot(dx_j[:, None], dy_j[None, :])
    near = jnp.abs(d_grid) <= 0.65 * diag + jnp.asarray(1e-4, dtype=dtype)
    inside = (d_grid < 0).astype(dtype)

    offs = (jnp.arange(sub_n, dtype=dtype) + 0.5 - 0.5 * sub_n) / sub_n
    ox_s, oy_s = jnp.meshgrid(offs, offs, indexing="ij")
    sqrt12 = math.sqrt(12.0)

    def _row_step(row_in: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array]:
        cx_r, dx_r, near_r, ins_r = row_in

        def _do_row() -> tuple[Array, Array, Array, Array]:
            sub_x = cx_r + dx_r * ox_s[None, :, :]
            sub_y = cy_j[:, None, None] + dy_j[:, None, None] * oy_s[None, :, :]
            sub_x = jnp.broadcast_to(sub_x, (ny, sub_n, sub_n))
            sub_pts = jnp.stack([sub_x.ravel(), sub_y.ravel()], axis=-1)
            sd, gx, gy = _eval_points(sub_pts)
            sd = sd.reshape(ny, sub_n, sub_n)
            gx = gx.reshape(ny, sub_n, sub_n)
            gy = gy.reshape(ny, sub_n, sub_n)
            norm = jnp.sqrt(jnp.maximum(gx * gx + gy * gy, jnp.asarray(1e-24, dtype=dtype)))
            nx_u = jnp.where(norm > 1e-9, gx / norm, jnp.ones((), dtype=dtype))
            ny_u = jnp.where(norm > 1e-9, gy / norm, jnp.zeros((), dtype=dtype))
            hx_s = (0.5 / sub_n) * dx_r
            hy_s = (0.5 / sub_n) * dy_j[:, None, None]
            f_s, mx_s, my_s, ell_s = _halfplane_box_moments(sd, nx_u, ny_u, hx_s, hy_s, res_j)
            f_r = jnp.where(near_r, jnp.mean(f_s, axis=(-2, -1)), ins_r)
            mx_r = jnp.where(near_r, jnp.mean(mx_s / sub_n + sqrt12 * ox_s[None] * f_s, axis=(-2, -1)), 0.0)
            my_r = jnp.where(near_r, jnp.mean(my_s / sub_n + sqrt12 * oy_s[None] * f_s, axis=(-2, -1)), 0.0)
            ch_raw = jnp.stack(
                [
                    jnp.sum(ell_s, axis=(-2, -1)),
                    jnp.sum(ell_s * nx_u, axis=(-2, -1)),
                    jnp.sum(ell_s * ny_u, axis=(-2, -1)),
                    jnp.sum(ell_s * nx_u * nx_u, axis=(-2, -1)),
                    jnp.sum(ell_s * ny_u * ny_u, axis=(-2, -1)),
                    jnp.sum(ell_s * nx_u * ny_u, axis=(-2, -1)),
                ],
                axis=-1,
            )
            ch_r = jnp.where(near_r[:, None], ch_raw, jnp.zeros((ny, 6), dtype=dtype))
            return f_r, mx_r, my_r, ch_r

        def _skip_row() -> tuple[Array, Array, Array, Array]:
            z_r = jnp.zeros((ny,), dtype=dtype)
            return ins_r, z_r, z_r, jnp.zeros((ny, 6), dtype=dtype)

        return jax.lax.cond(jnp.any(near_r), _do_row, _skip_row)

    return jax.lax.map(_row_step, (cx_j, dx_j, near, inside))


def _sdf_rasterize_local_grid(
    expr: tuple,
    source_edges_um: dict[int, tuple[Array, np.ndarray, np.ndarray, np.ndarray]],
    grow_um: Any,
    xe_um: Array,
    ye_um: Array,
    res_um: float,
    dtype: Any,
) -> tuple[Array, Array, Array, Array]:
    """Evaluate boolean ``expr`` via signed distance + half-plane clip on local sub-cell edges ``(xe_um, ye_um)``."""
    sx = xe_um.shape[0] - 1
    sy = ye_um.shape[0] - 1
    xe_a = jnp.asarray(xe_um)
    ye_a = jnp.asarray(ye_um)
    ox = 0.5 * (xe_a[0] + xe_a[-1])
    oy = 0.5 * (ye_a[0] + ye_a[-1])
    origin = jnp.stack([ox, oy])
    cx = (0.5 * (xe_a[:-1] + xe_a[1:]) - ox).astype(dtype)
    cy = (0.5 * (ye_a[:-1] + ye_a[1:]) - oy).astype(dtype)
    hx = (0.5 * (xe_a[1:] - xe_a[:-1])).astype(dtype)
    hy = (0.5 * (ye_a[1:] - ye_a[:-1])).astype(dtype)
    xx, yy = jnp.meshgrid(cx, cy, indexing="ij")
    pts = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
    grow_arr = jnp.asarray(grow_um, dtype=dtype)

    s_map: dict[int, tuple[Array, Array, Array]] = {}
    for idx, (v_um, nxt, ids, bnd) in source_edges_um.items():
        v_s = (jnp.asarray(v_um, dtype=origin.dtype) - origin).astype(dtype)
        s_map[idx] = _sdf_and_grad(pts, v_s, nxt, ids, bnd)
    sd, gx, gy = _eval_expr_sdf_um(expr, s_map, (pts.shape[0],))
    sd = (sd - grow_arr).reshape(sx, sy)
    gx = gx.reshape(sx, sy)
    gy = gy.reshape(sx, sy)
    norm = jnp.sqrt(jnp.maximum(gx * gx + gy * gy, jnp.asarray(1e-24, dtype=dtype)))
    nx_u = jnp.where(norm > 1e-9, gx / norm, 1.0)
    ny_u = jnp.where(norm > 1e-9, gy / norm, 0.0)
    f_s, mx_s, my_s, ell_s = _halfplane_box_moments(
        sd, nx_u, ny_u, hx[:, None], hy[None, :], jnp.asarray(res_um, dtype=dtype)
    )
    b_s = ell_s[..., None] * jnp.stack(
        [jnp.ones_like(nx_u), nx_u, ny_u, nx_u * nx_u, ny_u * ny_u, nx_u * ny_u],
        axis=-1,
    )
    return f_s, mx_s, my_s, b_s


def _nominal_arc_counts(
    topo_um: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    succ: np.ndarray,
    boundary: np.ndarray,
    max_arc: int = _N_ARC_CHORDS,
) -> np.ndarray:
    """Static per-boundary-edge chord counts proportional to nominal corner turn angle."""
    bnd = np.flatnonzero(boundary)
    if len(bnd) == 0:
        return np.zeros((0,), dtype=np.int32)
    pos = np.full(len(boundary), -1, dtype=np.int64)
    pos[bnd] = np.arange(len(bnd))
    next_b = np.where(succ[bnd] >= 0, pos[np.maximum(succ[bnd], 0)], -1)
    has_s = next_b >= 0
    safe_n = np.where(has_s, next_b, np.arange(len(bnd)))
    d = topo_um[dst[bnd]] - topo_um[src[bnd]]
    ln = np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-30)
    t = d / ln
    n = np.stack([t[:, 1], -t[:, 0]], axis=-1)
    nf = n[safe_n]
    dot = np.clip(np.sum(n * nf, axis=-1), -1.0, 1.0)
    theta = np.arccos(dot)
    counts = np.clip(np.ceil(max_arc * theta / (0.5 * math.pi)).astype(np.int32), 2, max_arc)
    return np.where(has_s, counts, 0).astype(np.int32)


def offset_boundary_segments(
    v_um: Array,
    src: np.ndarray,
    dst: np.ndarray,
    succ: np.ndarray,
    boundary: np.ndarray,
    grow_um: Any,
    *,
    n_arc: int = _N_ARC_CHORDS,
    arc_counts: np.ndarray | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Offset closed CCW boundary edges by signed ``grow_um`` (um) with round/mitred corners.

    Convex corners grown (or reflex corners shrunk) emit ``n_arc`` circular arc chords;
    opposite-sign corners meet at exact bounded mitres. Also returns ``has_inversion``
    indicating whether any boundary edge reversed direction past collapse.
    """
    dtype = v_um.dtype
    bnd = np.flatnonzero(boundary)
    if len(bnd) == 0:
        z2 = jnp.zeros((0, 2), dtype=dtype)
        zf = jnp.zeros((0,), dtype=bool)
        return z2, z2, zf, jnp.asarray(False)

    e_src = src[bnd]
    e_dst = dst[bnd]
    a = v_um[e_src]
    b = v_um[e_dst]
    if has_tracers(grow_um):
        is_zero = False
    else:
        try:
            is_zero = float(np.asarray(grow_um)) == 0.0
        except Exception:
            is_zero = False

    if is_zero:
        d = b - a
        lens = jnp.sum(d * d, axis=-1)
        return a, b, lens > 1e-24, jnp.asarray(False)

    g = jnp.asarray(grow_um, dtype=dtype)
    edge_pos = np.full(len(boundary), -1, dtype=np.int64)
    edge_pos[bnd] = np.arange(len(bnd))
    next_bnd = np.where(succ[bnd] >= 0, edge_pos[np.maximum(succ[bnd], 0)], -1)
    has_succ = jnp.asarray(next_bnd >= 0)
    safe_next = np.where(next_bnd >= 0, next_bnd, np.arange(len(bnd)))

    d_e = b - a
    l_e = jnp.sqrt(jnp.maximum(jnp.sum(d_e * d_e, axis=-1), jnp.asarray(1e-30, dtype=dtype)))
    t_e = d_e / l_e[:, None]
    n_e = jnp.stack([t_e[:, 1], -t_e[:, 0]], axis=-1)  # outward normal (CCW interior on left)

    t_f = t_e[safe_next]
    n_f = n_e[safe_next]
    turn = t_e[:, 0] * t_f[:, 1] - t_e[:, 1] * t_f[:, 0]  # >0 convex, <0 reflex
    dot = jnp.sum(n_e * n_f, axis=-1)
    # Mitre corner direction: (n_e + n_f) / max(1 + dot, 0.05)
    denom = jnp.maximum(1.0 + dot, 0.05)
    mitre_dir = (n_e + n_f) / denom[:, None]
    mitre_dir = jnp.where(has_succ[:, None], mitre_dir, n_e)

    # Check if this end corner rounds: convex grown OR reflex shrunk => g * turn > 0
    is_round = has_succ & ((g * turn) > 0.0) & (jnp.abs(turn) > 1e-6)
    end_dir = jnp.where(is_round[:, None], n_e, mitre_dir)

    # Previous edge entering e determines start corner of edge e
    prev_of = np.full(len(bnd), -1, dtype=np.int64)
    valid_succ = np.flatnonzero(next_bnd >= 0)
    prev_of[next_bnd[valid_succ]] = valid_succ
    has_prev = jnp.asarray(prev_of >= 0)
    safe_prev = np.where(prev_of >= 0, prev_of, np.arange(len(bnd)))

    start_dir = jnp.where(
        has_prev[:, None],
        jnp.where(is_round[safe_prev, None], n_e, mitre_dir[safe_prev]),
        n_e,
    )

    # Main shifted edge segments
    q0 = a + g * start_dir
    q1 = b + g * end_dir
    edge_valid = l_e > 1e-12
    # Detect if any active shifted edge inverted against its tangent
    edge_dot = jnp.sum((q1 - q0) * t_e, axis=-1)
    has_inversion = jnp.any(edge_valid & (edge_dot <= 1e-9))

    # Build n_arc chords around each vertex b where is_round is True
    n_perp = n_f - dot[:, None] * n_e
    sin_th2 = jnp.sum(n_perp * n_perp, axis=-1)
    sin_th = jnp.sqrt(jnp.maximum(sin_th2, jnp.asarray(1e-24, dtype=dtype)))
    theta = jnp.arctan2(sin_th, dot)  # [0, pi]
    small_th = sin_th <= 1e-4

    if arc_counts is None:
        s_vals = jnp.linspace(0.0, 1.0, n_arc + 1, dtype=dtype)[None, :]  # (1, n_arc + 1)
        th_s = s_vals * theta[:, None]
        # Exact sin(t*theta)/sin(theta) or smooth Taylor limit when corner nearly straight
        alpha = jnp.where(
            small_th[:, None],
            s_vals * (1.0 + (1.0 - s_vals * s_vals) * (theta[:, None] ** 2) / 6.0),
            jnp.sin(th_s) / jnp.where(small_th[:, None], 1.0, sin_th[:, None]),
        )
        u_arc = jnp.cos(th_s)[..., None] * n_e[:, None, :] + alpha[..., None] * n_perp[:, None, :]
        arc_pts = b[:, None, :] + g * u_arc  # (E_b, n_arc + 1, 2)
        a0 = arc_pts[:, :-1, :].reshape(-1, 2)
        a1 = arc_pts[:, 1:, :].reshape(-1, 2)
        arc_valid = jnp.repeat(is_round, n_arc)
    else:
        counts_arr = np.asarray(arc_counts, dtype=np.int32)
        tot_arcs = int(counts_arr.sum())
        if tot_arcs == 0:
            return q0, q1, edge_valid, has_inversion
        e_idx = np.repeat(np.arange(len(bnd)), counts_arr)
        s0_np = np.concatenate([np.arange(k, dtype=np.float64) / k for k in counts_arr if k > 0])
        s1_np = np.concatenate([np.arange(1, k + 1, dtype=np.float64) / k for k in counts_arr if k > 0])

        def _arc_point(s_arr: Array, idx_arr: np.ndarray) -> Array:
            th_e = theta[idx_arr]
            sm_e = small_th[idx_arr]
            sin_e = sin_th[idx_arr]
            th_val = s_arr * th_e
            alp = jnp.where(
                sm_e,
                s_arr * (1.0 + (1.0 - s_arr * s_arr) * (th_e * th_e) / 6.0),
                jnp.sin(th_val) / jnp.where(sm_e, 1.0, sin_e),
            )
            u = jnp.cos(th_val)[:, None] * n_e[idx_arr] + alp[:, None] * n_perp[idx_arr]
            return b[idx_arr] + g * u

        a0 = _arc_point(jnp.asarray(s0_np, dtype=dtype), e_idx)
        a1 = _arc_point(jnp.asarray(s1_np, dtype=dtype), e_idx)
        arc_valid = is_round[e_idx]

    p0_out = jnp.concatenate([q0, a0], axis=0)
    p1_out = jnp.concatenate([q1, a1], axis=0)
    v_out = jnp.concatenate([edge_valid, arc_valid], axis=0)
    return p0_out, p1_out, v_out, has_inversion


# ---------------------------------------------------------------- 3D & multi-level interface rasterization


def _extract_geometries_um(
    device: Any,
    params: dict[str, Array],
    dtype: Any,
) -> tuple[
    dict[int, tuple[Array, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    dict[str, list[tuple[int, bool]]],
]:
    """Box-centered physical um vertices & static boundary links for source layers and exact derived shape parts."""
    from .stack._layers import _part_index

    center_np = np.asarray(device.center, dtype=np.float64)
    sources: dict[
        int, tuple[Array, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    keys = {index: k for k, index in device.source_layers}

    def _to_box_um(coords: Array | np.ndarray) -> Array:
        if has_tracers(coords):
            calc = jnp.float64 if jax.config.jax_enable_x64 else dtype
            c = jnp.asarray(center_np, dtype=calc)
            return (jnp.asarray(coords, dtype=calc) - c).astype(dtype)
        return jnp.asarray(np.asarray(to_numpy(coords), dtype=np.float64) - center_np, dtype=dtype)

    for (k, index), v0, sizes in zip(device.source_layers, device.source_vertices, device.source_sizes, strict=True):
        topo_um = np.asarray(v0, dtype=np.float64) - center_np
        tol_um = float(device._topology_tol(v0))
        split, split_sizes = split_t_junctions(topo_um, sizes, tol_um)
        v_um = _to_box_um(params[k])[split]
        split_topo = topo_um[split]
        nxt, ids, boundary = _edge_topology(split_topo, split_sizes, tol_um)
        src, dst, succ = _boundary_links(split_topo, split_sizes, nxt, ids, boundary, tol_um)
        arc_c = _nominal_arc_counts(split_topo, src, dst, succ, boundary)
        sources[index] = (v_um, src, dst, succ, boundary, nxt, ids, arc_c)

    for k_d, d in enumerate(device.derived_layers):
        pool_um = (
            jnp.concatenate([_to_box_um(params[keys[idx]]) for idx in d.layers])
            if d.layers
            else jnp.zeros((0, 2), dtype=dtype)
        )
        for j_r, r in enumerate(d.recipes):
            if not len(r.kind):
                continue
            part_idx = _part_index(k_d, j_r)
            topo_um = np.asarray(r.positions, dtype=np.float64) - center_np
            v_um = run_recipe(r.kind, r.index, topo_um, pool_um)
            sizes = (len(topo_um),)
            tol_um = float(d.tol)
            nxt, ids, boundary = _edge_topology(topo_um, sizes, tol_um)
            src, dst, succ = _boundary_links(topo_um, sizes, nxt, ids, boundary, tol_um)
            arc_c = _nominal_arc_counts(topo_um, src, dst, succ, boundary)
            sources[part_idx] = (v_um, src, dst, succ, boundary, nxt, ids, arc_c)

    derived_meta: dict[str, list[tuple[int, bool]]] = {}
    for k_d, d in enumerate(device.derived_layers):
        parts = [(_part_index(k_d, j_r), bool(r.negate)) for j_r, r in enumerate(d.recipes) if len(r.kind)]
        for lv_name in d.levels:
            derived_meta[lv_name] = parts
    return sources, derived_meta


def _static_source_topology(device: Any) -> dict[int, np.ndarray]:
    """Nominal (drawn) box-centered vertices in um, as numpy, per source index of :func:`_extract_geometries_um`.

    Same split/ordering as the traced ``v_um`` there, but concrete even under ``jit`` (for static sizing).
    """
    from .stack._layers import _part_index

    center_np = np.asarray(device.center, dtype=np.float64)
    out: dict[int, np.ndarray] = {}
    for (_, index), v0, sizes in zip(device.source_layers, device.source_vertices, device.source_sizes, strict=True):
        topo_um = np.asarray(v0, dtype=np.float64) - center_np
        split, _ = split_t_junctions(topo_um, sizes, float(device._topology_tol(v0)))
        out[index] = topo_um[split]
    for k_d, d in enumerate(device.derived_layers):
        for j_r, r in enumerate(d.recipes):
            if len(r.kind):
                out[_part_index(k_d, j_r)] = np.asarray(r.positions, dtype=np.float64) - center_np
    return out


def _level_kind_and_parts(
    device: Any,
    lv: Any,
    derived_meta: dict[str, list[tuple[int, bool]]],
    sources: dict[int, Any] | None = None,
) -> tuple[str, Any]:
    """Classify a level as ('fill', 0), ('empty', 0), ('boundary', (slots, negate, root_grow)), or ('sdf', expr)."""
    from .stack._layers import _inner_grows, _root_grow, _simplify_expr

    if lv.name in derived_meta:
        parts = derived_meta[lv.name]
        grow = float(_root_grow(lv.expr))
        if not parts:
            # All derived recipes empty -> negated means full fill, non-negated means empty
            expr = device._drawn_expr(lv)
            return ("fill", 0.0) if expr[0] == "fill" else ("empty", 0.0)
        return ("boundary", ([p[0] for p in parts], parts[0][1], grow))

    expr = device._drawn_expr(lv)
    if sources is not None:
        expr = _simplify_expr(expr, sources)
    kind = expr[0]
    if kind == "fill":
        return ("fill", 0.0)
    if kind == "empty":
        return ("empty", 0.0)
    if kind == "layer":
        idx = int(expr[1])
        if idx == _NO_LAYER:
            return ("empty", 0.0)
        return ("boundary", ([idx], False, float(expr[2])))
    # Composite or inner-grow boolean -> use smooth SDF sub-grid fallback
    if kind == "not" and expr[1][0] == "fill" and not _inner_grows(expr) and expr[2][0] == "layer":
        return ("boundary", ([int(expr[2][1])], True, float(expr[3])))
    return ("sdf", expr)


def _grid_bounds(grid: dict[str, Any], dtype: Any) -> tuple[Array, Array, Array, Array]:
    """Return ``(x_lo, x_hi, y_lo, y_hi)`` voxel boundary arrays for contiguous or box grids."""
    if grid["contiguous"]:
        return (
            jnp.asarray(grid["x_edges"][:-1], dtype=dtype),
            jnp.asarray(grid["x_edges"][1:], dtype=dtype),
            jnp.asarray(grid["y_edges"][:-1], dtype=dtype),
            jnp.asarray(grid["y_edges"][1:], dtype=dtype),
        )
    return (
        jnp.asarray(grid["x_lo"], dtype=dtype),
        jnp.asarray(grid["x_hi"], dtype=dtype),
        jnp.asarray(grid["y_lo"], dtype=dtype),
        jnp.asarray(grid["y_hi"], dtype=dtype),
    )


def _union_expr(slot_indices: Sequence[int], valid_sources: dict[int, Any]) -> tuple:
    """Build SDF union expression over active boundary slots."""
    kept = [idx for idx in slot_indices if idx in valid_sources]
    if not kept:
        return ("empty", 0.0)
    expr: tuple = ("layer", kept[0], 0.0)
    for idx in kept[1:]:
        expr = ("or", expr, ("layer", idx, 0.0), 0.0)
    return expr


def _rasterize_slots_local(
    slot_segs: Sequence[tuple[Array, Array, Array]],
    xe_p: Any,
    ye_p: Any,
    res_um: float,
    need_moments: bool,
    negate: bool,
    dtype: Any,
) -> tuple[Array, Array, Array, Array]:
    """Rasterize 1 or more CCW boundary-segment slots onto local sub-grid ``(xe_p, ye_p)`` with union merge."""
    if len(slot_segs) == 1:
        q0, q1, qv = slot_segs[0]
        fj, mxj, myj, bj = rasterize_segments_2d(q0, q1, qv, xe_p, ye_p, res_um, need_moments=need_moments)
        fj = jnp.clip(fj, 0.0, 1.0)
    else:
        outs = [rasterize_segments_2d(q0, q1, qv, xe_p, ye_p, res_um, need_moments=need_moments) for q0, q1, qv in slot_segs]
        fs = jnp.stack([jnp.clip(o[0], 0.0, 1.0) for o in outs], axis=0)
        fj = jnp.max(fs, axis=0)
        if need_moments:
            idx_max = jnp.argmax(fs, axis=0)
            mxs = jnp.stack([o[1] for o in outs], axis=0)
            mys = jnp.stack([o[2] for o in outs], axis=0)
            bs = jnp.stack([o[3] for o in outs], axis=0)
            mxj = jnp.take_along_axis(mxs, idx_max[None], axis=0)[0]
            myj = jnp.take_along_axis(mys, idx_max[None], axis=0)[0]
            m_cnt = len(outs)
            other_max = jnp.stack(
                [jnp.max(jnp.stack([fs[k] for k in range(m_cnt) if k != m], axis=0), axis=0) for m in range(m_cnt)],
                axis=0,
            )
            bj = jnp.sum((1.0 - other_max)[..., None] * bs, axis=0)
        else:
            mxj = outs[0][1]
            myj = outs[0][2]
            bj = outs[0][3]
    if negate:
        bj = bj * jnp.asarray([1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=dtype)
        return 1.0 - fj, -mxj, -myj, bj
    return fj, mxj, myj, bj


def _rasterize_level_slice(
    kind: str,
    info: Any,
    grow_um: Any,
    sources: dict[int, tuple[Array, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    grid: dict[str, Any],
    *,
    need_moments: bool,
    dtype: Any,
    assume_no_inversion: bool = False,
    window: tuple[int, int] | None = None,
    piece_counts: dict[int, np.ndarray] | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Rasterize one level at scalar lateral offset ``grow_um`` across the 2D sample grid.

    ``assume_no_inversion=True`` skips the traced offset-inversion ``lax.cond`` (SDF fallback); the caller
    must have checked :func:`offset_boundary_segments`' inversion flag beforehand. Used by the batched
    (``vmap``) slice path, where a traced ``cond`` would degrade into running both branches.
    ``piece_counts`` (per source index) splits the offset segments into short pieces and ``window`` then
    limits each piece to that many grid lines (contiguous grids only; see :func:`rasterize_segments_2d`).
    """
    nx = grid["dx"].shape[0]
    ny = grid["dy"].shape[0]
    res_um = grid["res_um"]
    n_sub = grid["n_sub"]
    coord_dtype = jnp.float64 if jax.config.jax_enable_x64 else dtype

    def _fill_const(val: float) -> tuple[Array, Array, Array, Array]:
        f = jnp.full((nx, ny), val, dtype=dtype)
        z = jnp.zeros((nx, ny), dtype=dtype)
        return f, z, z, jnp.zeros((nx, ny, 6), dtype=dtype)

    if kind == "fill":
        return _fill_const(1.0)
    if kind == "empty":
        return _fill_const(0.0)

    def _run_raster(p0: Array, p1: Array, valid: Array) -> tuple[Array, Array, Array, Array]:
        if grid["contiguous"]:
            f_f, mx_f, my_f, b_f = rasterize_segments_2d(
                p0,
                p1,
                valid,
                grid["x_fine"],
                grid["y_fine"],
                res_um,
                need_moments=need_moments,
                window=window,
            )
            return _pool_subsamples(
                f_f,
                mx_f,
                my_f,
                b_f,
                jnp.asarray(grid["x_edges"], dtype=dtype),
                jnp.asarray(grid["y_edges"], dtype=dtype),
                n_sub,
            )
        f_f, mx_f, my_f, b_f = rasterize_noncontiguous_boxes_2d(
            p0,
            p1,
            valid,
            grid["xf_lo"],
            grid["xf_hi"],
            grid["yf_lo"],
            grid["yf_hi"],
            res_um,
            need_moments=need_moments,
            non_overlapping_y=grid.get("non_overlapping_y"),
        )
        if n_sub == 1:
            return f_f, mx_f, my_f, b_f
        offs = (jnp.arange(n_sub, dtype=dtype) + 0.5 - 0.5 * n_sub) / n_sub
        sqrt12 = math.sqrt(12.0)
        f4 = f_f.reshape(nx, n_sub, ny, n_sub)
        mx4 = mx_f.reshape(nx, n_sub, ny, n_sub)
        my4 = my_f.reshape(nx, n_sub, ny, n_sub)
        b4 = b_f.reshape(nx, n_sub, ny, n_sub, 6)
        return (
            jnp.mean(f4, axis=(1, 3)),
            jnp.mean(mx4 / n_sub + sqrt12 * offs[None, :, None, None] * f4, axis=(1, 3)),
            jnp.mean(my4 / n_sub + sqrt12 * offs[None, None, None, :] * f4, axis=(1, 3)),
            jnp.sum(b4, axis=(1, 3)),
        )

    if kind == "sdf":
        src_sdf = {idx: (sources[idx][0], sources[idx][5], sources[idx][6], sources[idx][4]) for idx in sources if idx >= 0}
        x_lo, x_hi, y_lo, y_hi = _grid_bounds(grid, coord_dtype)
        return rasterize_sdf_subgrid(info, src_sdf, grow_um, x_lo, x_hi, y_lo, y_hi, res_um, dtype=dtype)

    # Boundary representation: (slot_indices, negate, root_grow)
    slot_indices, negate, root_grow = info
    # Sign: grow_um here is outward grow of the level BEFORE negation.
    # For negated level ("not", fill, union, root_grow): outward grow on union is (-eff_grow_level)
    eff_grow = jnp.asarray(grow_um, dtype=dtype)
    union_grow = (-(eff_grow + jnp.asarray(root_grow, dtype=dtype))) if negate else (
        eff_grow + jnp.asarray(root_grow, dtype=dtype)
    )

    slot_segs: list[tuple[Array, Array, Array]] = []
    inv_flags: list[Array] = []
    for idx in slot_indices:
        if idx not in sources:
            continue
        v_um, src, dst, succ, bnd, _, _, arc_c = sources[idx]
        p0_i, p1_i, v_i, inv_i = offset_boundary_segments(
            v_um, src, dst, succ, bnd, union_grow, arc_counts=arc_c
        )
        if piece_counts is not None and idx in piece_counts:
            p0_i, p1_i, v_i = _split_segments(p0_i, p1_i, v_i, piece_counts[idx])
        slot_segs.append((p0_i, p1_i, v_i))
        inv_flags.append(inv_i)

    if not slot_segs:
        return _fill_const(1.0 if negate else 0.0)

    def _run_boundary() -> tuple[Array, Array, Array, Array]:
        if len(slot_segs) == 1:
            return _run_raster(*slot_segs[0])
        slot_outs = [_run_raster(p0_i, p1_i, v_i) for p0_i, p1_i, v_i in slot_segs]
        fs = jnp.stack([jnp.clip(o[0], 0.0, 1.0) for o in slot_outs], axis=0)
        f_sum = _st_clip(jnp.sum(fs, axis=0), 0.0, 1.0)
        mx_sum = jnp.sum(jnp.stack([o[1] for o in slot_outs], axis=0), axis=0)
        my_sum = jnp.sum(jnp.stack([o[2] for o in slot_outs], axis=0), axis=0)
        b_sum = jnp.sum(jnp.stack([o[3] for o in slot_outs], axis=0), axis=0)
        multi_slot = jnp.sum((fs > 1e-6).astype(jnp.int32), axis=0) >= 2
        if has_tracers(multi_slot) or bool(jnp.any(multi_slot)):
            sub_m = 32
            x_lo_arr, x_hi_arr, y_lo_arr, y_hi_arr = _grid_bounds(grid, coord_dtype)

            def _refine_slots(c_idx: Array, c_mask: Array, _: Any) -> tuple[Array, Array, Array, Array]:
                ix = c_idx // ny
                iy = c_idx % ny
                t_lin = jnp.linspace(0.0, 1.0, sub_m + 1, dtype=coord_dtype)
                sx = x_lo_arr[ix, None] + (x_hi_arr[ix] - x_lo_arr[ix])[:, None] * t_lin[None, :]
                sy = y_lo_arr[iy, None] + (y_hi_arr[iy] - y_lo_arr[iy])[:, None] * t_lin[None, :]

                def _one_px(p_i: Array) -> tuple[Array, Array, Array, Array]:
                    xe_p, ye_p = sx[p_i], sy[p_i]
                    f_s, mx_s, my_s, b_s = _rasterize_slots_local(
                        slot_segs, xe_p, ye_p, res_um, need_moments, False, dtype
                    )
                    xp = jnp.stack([xe_p[0], xe_p[-1]]).astype(dtype)
                    yp = jnp.stack([ye_p[0], ye_p[-1]]).astype(dtype)
                    f_p, mx_p, my_p, b_p = _pool_subsamples(f_s, mx_s, my_s, b_s, xp, yp, sub_m)
                    return f_p[0, 0], mx_p[0, 0], my_p[0, 0], b_p[0, 0]

                return jax.vmap(_one_px)(jnp.arange(c_idx.shape[0]))

            rf, rmx, rmy, rb = sparse_voxel_map(_refine_slots, multi_slot, None, chunk_size=8, group_size=64)
            f_sum = jnp.where(multi_slot, rf, f_sum)
            if need_moments:
                mx_sum = jnp.where(multi_slot, rmx, mx_sum)
                my_sum = jnp.where(multi_slot, rmy, my_sum)
                b_sum = jnp.where(multi_slot[..., None], rb, b_sum)
        return f_sum, mx_sum, my_sum, b_sum

    def _run_sdf_fallback() -> tuple[Array, Array, Array, Array]:
        src_sdf = {idx: (sources[idx][0], sources[idx][5], sources[idx][6], sources[idx][4]) for idx in slot_indices if idx in sources}
        expr = _union_expr(slot_indices, sources)
        x_lo, x_hi, y_lo, y_hi = _grid_bounds(grid, coord_dtype)
        return rasterize_sdf_subgrid(expr, src_sdf, union_grow, x_lo, x_hi, y_lo, y_hi, res_um, sub_n=1, dtype=dtype)

    any_inv = jax.tree.reduce(lambda a, b: a | b, inv_flags)
    if assume_no_inversion:
        f_u, mx_u, my_u, b_u = _run_boundary()
    elif not has_tracers(any_inv):
        f_u, mx_u, my_u, b_u = _run_sdf_fallback() if bool(any_inv) else _run_boundary()
    else:
        f_u, mx_u, my_u, b_u = jax.lax.cond(any_inv, _run_sdf_fallback, _run_boundary)

    f_u = jnp.clip(f_u, 0.0, 1.0)
    if negate:
        b_neg = b_u * jnp.asarray([1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=dtype)
        return 1.0 - f_u, -mx_u, -my_u, b_neg
    return f_u, mx_u, my_u, b_u


def _unit_square_offset(f: Array, nx_u: Array, ny_u: Array) -> Array:
    """Signed offset ``o`` so ``nx_u * u + ny_u * v + o <= 0`` on ``[-0.5, 0.5]^2`` has area ``f``."""
    dtype = f.dtype
    a = jnp.minimum(jnp.abs(nx_u), jnp.abs(ny_u))
    b = jnp.maximum(jnp.abs(nx_u), jnp.abs(ny_u))
    safe_b = jnp.maximum(b, jnp.asarray(1e-12, dtype=dtype))
    f_lo = jnp.minimum(f, 1.0 - f)
    f_tri = a / (2.0 * safe_b)
    in_tri = (f_lo < f_tri) & (f_lo > 1e-12)
    rad = jnp.where(in_tri, 2.0 * a * safe_b * f_lo, jnp.ones((), dtype=dtype))
    d_tri = -0.5 * (a + safe_b) + jnp.sqrt(rad)
    d_trap = safe_b * (f_lo - 0.5)
    d_lo = jnp.where(
        f_lo <= 1e-12,
        -0.5 * (a + safe_b) - jnp.asarray(1e-3, dtype=dtype),
        jnp.where(in_tri, d_tri, d_trap),
    )
    d = jnp.where(f <= 0.5, d_lo, -d_lo)
    return -d


def _poly_area_and_moments(poly: Array, count: Array) -> tuple[Array, Array, Array]:
    """Shoelace area and standardized first moments ``(area, std_mx, std_my)`` on ``[-0.5, 0.5]^2``."""
    idx = jnp.arange(poly.shape[-2])
    valid = idx < count[..., None]
    nxt = jnp.where(idx + 1 < count[..., None], idx + 1, 0)
    q = jnp.take_along_axis(poly, nxt[..., None], axis=-2)
    cross2 = jnp.where(valid, poly[..., 0] * q[..., 1] - q[..., 0] * poly[..., 1], 0.0)
    area = jnp.clip(0.5 * jnp.sum(cross2, axis=-1), 0.0, 1.0)
    sqrt12 = math.sqrt(12.0)
    mx = (sqrt12 / 6.0) * jnp.sum(cross2 * (poly[..., 0] + q[..., 0]), axis=-1)
    my = (sqrt12 / 6.0) * jnp.sum(cross2 * (poly[..., 1] + q[..., 1]), axis=-1)
    lim = 0.5 * sqrt12
    return area, jnp.clip(mx, -lim, lim), jnp.clip(my, -lim, lim)


def _combine_levels_subinterval(
    f_lvl: Array,
    mx_lvl: Array,
    my_lvl: Array,
    b_lvl: Array,
    present_lvl: Array,
    b_slopes: Array,
    vectors: Array,
    contrast2: Array,
    area_frame: Array,
    cell_scale_frame: Array,
    *,
    need_direction: bool,
    polygon_clip: bool = False,
    sub_dx: Any = 1.0,
    sub_dy: Any = 1.0,
) -> tuple[Array, Array, Array]:
    """Analytically combine priority-ordered levels ``0..K-1`` (plus background ``K``) on one z sub-interval.

    Returns ``(visible_K1, lateral_dir_6, moments_6_3)``.
    Exact whenever at most 1 patterned level is strictly inside ``(0, 1)``; pixels with
    2+ partially filled patterned levels are refined separately via :func:`_eval_overlap_pixels`
    using ``polygon_clip=True`` for sequential half-plane clipping on sub-cells.
    """
    num_k = f_lvl.shape[-1]
    dtype = f_lvl.dtype
    batch_shape = f_lvl.shape[:-1]

    vis_list = []
    mx_vis_list = []
    my_vis_list = []
    uncov_before_list = []

    if not polygon_clip:
        uncovered = jnp.ones(batch_shape, dtype=dtype)
        rem_mx = jnp.zeros(batch_shape, dtype=dtype)
        rem_my = jnp.zeros(batch_shape, dtype=dtype)

        for j in range(num_k):
            pres = present_lvl[..., j]
            fj = jnp.clip(f_lvl[..., j], 0.0, 1.0)
            vj = jnp.where(pres, fj * uncovered, 0.0)
            uncov_before = uncovered
            uncov_before_list.append(uncov_before)
            uncovered = jnp.maximum(uncovered - vj, 0.0)
            vis_list.append(vj)
            is_full = fj >= 1.0 - 1e-6
            mv_x = jnp.where(pres, jnp.where(is_full, rem_mx, uncov_before * mx_lvl[..., j]), 0.0)
            mv_y = jnp.where(pres, jnp.where(is_full, rem_my, uncov_before * my_lvl[..., j]), 0.0)
            rem_mx = rem_mx - mv_x
            rem_my = rem_my - mv_y
            mx_vis_list.append(mv_x)
            my_vis_list.append(mv_y)

        # Background gets remainder
        vis_list.append(uncovered)
        mx_vis_list.append(rem_mx)
        my_vis_list.append(rem_my)
    else:
        from .stack._corner_model import _clip

        max_v = 4 + num_k
        sq_verts = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]] + [[-0.5, 0.5]] * num_k
        poly = jnp.broadcast_to(jnp.asarray(sq_verts, dtype=dtype), (*batch_shape, max_v, 2))
        count = jnp.full(batch_shape, 4, dtype=jnp.int32)
        rem_area = jnp.ones(batch_shape, dtype=dtype)
        rem_mx = jnp.zeros(batch_shape, dtype=dtype)
        rem_my = jnp.zeros(batch_shape, dtype=dtype)
        dx_s = jnp.asarray(sub_dx, dtype=dtype)
        dy_s = jnp.asarray(sub_dy, dtype=dtype)

        for j in range(num_k):
            pres = present_lvl[..., j]
            fj = jnp.clip(f_lvl[..., j], 0.0, 1.0)
            uncov_before_list.append(rem_area)

            is_empty = (~pres) | (fj <= 1e-6) | (rem_area <= 1e-7)
            is_full = pres & (fj >= 1.0 - 1e-6) & (~is_empty)
            is_part = pres & (~is_empty) & (~is_full)

            gx = b_lvl[..., j, 1] * dx_s
            gy = b_lvl[..., j, 2] * dy_s
            gnorm2 = gx * gx + gy * gy
            has_dir = is_part & (gnorm2 > 1e-16)
            inv_gn = jax.lax.rsqrt(jnp.where(has_dir, gnorm2, jnp.ones((), dtype=dtype)))
            nx_u = jnp.where(has_dir, gx * inv_gn, jnp.ones((), dtype=dtype))
            ny_u = jnp.where(has_dir, gy * inv_gn, jnp.zeros((), dtype=dtype))
            o_j = _unit_square_offset(jnp.where(has_dir, fj, jnp.full_like(fj, 0.5)), nx_u, ny_u)

            clipped, c_count = _clip(poly, count, jnp.stack([nx_u, ny_u], axis=-1), o_j)
            clip_area, clip_mx, clip_my = _poly_area_and_moments(clipped, c_count)
            clip_area = jnp.minimum(clip_area, rem_area)

            fb_area = rem_area * (1.0 - fj)
            fb_mx = rem_mx * (1.0 - fj)
            fb_my = rem_my * (1.0 - fj)

            next_area = jnp.where(
                is_empty,
                rem_area,
                jnp.where(is_full, jnp.zeros_like(rem_area), jnp.where(has_dir, clip_area, fb_area)),
            )
            next_mx = jnp.where(
                is_empty,
                rem_mx,
                jnp.where(is_full, jnp.zeros_like(rem_mx), jnp.where(has_dir, clip_mx, fb_mx)),
            )
            next_my = jnp.where(
                is_empty,
                rem_my,
                jnp.where(is_full, jnp.zeros_like(rem_my), jnp.where(has_dir, clip_my, fb_my)),
            )
            poly = jnp.where(has_dir[..., None, None], clipped, poly)
            count = jnp.where(is_full, jnp.zeros_like(count), jnp.where(has_dir, c_count, count))

            vis_list.append(jnp.maximum(rem_area - next_area, 0.0))
            mx_vis_list.append(rem_mx - next_mx)
            my_vis_list.append(rem_my - next_my)
            rem_area, rem_mx, rem_my = next_area, next_mx, next_my

        vis_list.append(rem_area)
        mx_vis_list.append(rem_mx)
        my_vis_list.append(rem_my)

    visible = jnp.stack(vis_list, axis=-1)  # (..., K + 1)
    if not need_direction:
        return visible, jnp.zeros((*batch_shape, 6), dtype=dtype), jnp.zeros((*batch_shape, 6, 3), dtype=dtype)

    mx_vis = jnp.stack(mx_vis_list, axis=-1)
    my_vis = jnp.stack(my_vis_list, axis=-1)
    mom = jnp.stack([visible, mx_vis, my_vis], axis=-1)  # (..., K + 1, 3)
    moments = jnp.einsum("...lc,lv->...vc", mom, vectors, precision=_HIGHEST)

    # Lateral direction dyad per level j: weight against visible later levels k > j (plus background K)
    dir_6 = jnp.zeros((*batch_shape, 6), dtype=dtype)
    for j in range(num_k):
        pres = present_lvl[..., j]
        # Total visible fraction strictly outside j (among k > j + background)
        v_after = visible[..., j + 1 :]  # (..., K + 1 - (j + 1))
        out_tot = jnp.sum(v_after, axis=-1)
        vj = visible[..., j]
        ell_f = b_lvl[..., j, 0] * uncov_before_list[j]  # weighted by occlusion before j
        some = pres & (ell_f > _TINY) & (vj > _TINY) & (out_tot > _TINY)
        sum_vo = jnp.where(some, vj + out_tot, 1.0)
        thin = jnp.where(some, (vj * out_tot) / sum_vo, 0.0) * area_frame
        safe_ell = jnp.where(some, ell_f, 1.0)
        t_frame = thin / (safe_ell * cell_scale_frame)  # width ratio matching stack.py:_face_weights
        psi = jnp.where(some, t_frame / (t_frame + _FACE_THICKNESS), 0.0)
        c2_row = contrast2[j, j + 1 :]
        safe_out = jnp.where(out_tot > _TINY, out_tot, 1.0)
        mean_c2 = jnp.sum(v_after * c2_row, axis=-1) / safe_out
        w_j = psi * mean_c2  # scalar weight multiplying b_lvl[..., j, :]

        # Tilt 3D normal dyad from b_lvl channels:
        # [ell, ell*nx, ell*ny, ell*nx^2, ell*ny^2, ell*nx*ny] / sqrt(1 + bj^2) * [nx^2, ny^2, bj^2, nx*ny, bj*nx, bj*ny]
        bj = b_lvl[..., j, :] * uncov_before_list[j][..., None]
        slope_j = b_slopes[..., j]
        inv_tilt = jax.lax.rsqrt(1.0 + slope_j * slope_j)
        d6 = jnp.stack(
            [
                bj[..., 3],
                bj[..., 4],
                (slope_j * slope_j) * bj[..., 0],
                bj[..., 5],
                slope_j * bj[..., 1],
                slope_j * bj[..., 2],
            ],
            axis=-1,
        ) * (w_j * inv_tilt)[..., None]
        dir_6 = dir_6 + d6

    return visible, dir_6, moments


# ---------------------------------------------------------------- public StackDevice integration


def _is_constant_zero(val: Any) -> bool:
    if has_tracers(val):
        return False
    try:
        return float(np.asarray(val)) == 0.0
    except Exception:
        return False


def _layer_split_1d(lows: Array, highs: Array, z_lo: Array, z_hi: Array) -> tuple[Array, Array, Array]:
    """Split layer interval ``[z_lo, z_hi]`` by uniform ``(lows, highs)`` (1D, shape ``(K,)``)."""
    from .stack._profiles import _split_layer

    perm, bp, length, present = _split_layer(lows, highs, z_lo, top=z_hi)
    return bp, length, present


def _level_grow_and_slope(
    lv: Any,
    wall: tuple[Array, Array, Array] | None,
    z_f: Array,
    res_um: float,
    dtype: Any,
) -> tuple[Array, Array]:
    """Outward lateral grow (um) and wall slope ``d(shift)/dz`` for one level at height ``z_f``."""
    from .stack._corner_model import _sidewall_shift

    z_f = jnp.asarray(z_f, dtype=dtype)
    if wall is None:
        z0_arr = jnp.zeros_like(z_f)
        return z0_arr, z0_arr
    zb_r, t_r, ang_r = wall
    if lv.z_to_bias is None and _is_constant_zero(ang_r):
        g0 = jnp.broadcast_to(jnp.asarray(lv.bias, dtype=dtype), z_f.shape)
        return g0, jnp.zeros_like(z_f)
    zb = jnp.asarray(zb_r, dtype=dtype)
    t = jnp.asarray(t_r, dtype=dtype)
    angle = jnp.asarray(ang_r, dtype=dtype)
    shift_z = _sidewall_shift(lv, z_f, zb, t, angle, res_um)
    grow_um_j = (-shift_z * jnp.asarray(res_um, dtype=dtype)).astype(dtype)
    b_j = (jnp.tan(jnp.deg2rad(angle)) + jnp.zeros_like(z_f)).astype(dtype)
    if lv.z_to_bias is not None:
        zs, bs = lv.z_to_bias
        zs_j = jnp.asarray(zs, dtype=dtype)
        bs_j = jnp.asarray(bs, dtype=dtype)
        nonzero_t = jnp.abs(t) > _TINY
        safe_t = jnp.where(nonzero_t, t, 1.0)
        frac = jnp.where(nonzero_t, (z_f - zb) / safe_t, 0.0)
        seg_i = jnp.clip(jnp.searchsorted(zs_j, jax.lax.stop_gradient(frac), side="right") - 1, 0, len(zs) - 2)
        dz_p = jnp.where(jnp.abs(zs_j[seg_i + 1] - zs_j[seg_i]) > _TINY, zs_j[seg_i + 1] - zs_j[seg_i], 1.0)
        db_dz = jnp.where(nonzero_t, ((bs_j[seg_i + 1] - bs_j[seg_i]) / res_um) / (safe_t * dz_p), 0.0)
        in_tab = nonzero_t & (frac >= zs_j[0]) & (frac <= zs_j[-1])
        b_j = (b_j - jnp.where(in_tab, db_dz, 0.0)).astype(dtype)
    return grow_um_j, b_j


def _prepare_shift_grid(device: Any, shift: tuple[float, float, float], dtype: Any) -> tuple[dict[str, Array], dict[str, Any]]:
    """Convert one Yee-shifted sample grid into a pure-Array dict PyTree plus static metadata."""
    grid = sample_grid_edges_um(device, shift)
    zaxis = device._z_axis(shift[2])
    coord_dtype = jnp.float64 if jax.config.jax_enable_x64 else dtype
    arrays: dict[str, Array] = {
        "dx": jnp.asarray(grid["dx"], dtype=coord_dtype),
        "dy": jnp.asarray(grid["dy"], dtype=coord_dtype),
        "z_frame_lo": jnp.asarray(zaxis.lo, dtype=dtype),
        "z_frame_hi": jnp.asarray(zaxis.hi, dtype=dtype),
        "z_shift": jnp.asarray(zaxis.shift, dtype=dtype),
    }
    if grid["contiguous"]:
        arrays.update(
            {
                "x_edges": jnp.asarray(grid["x_edges"], dtype=coord_dtype),
                "y_edges": jnp.asarray(grid["y_edges"], dtype=coord_dtype),
                "x_fine": jnp.asarray(grid["x_fine"], dtype=coord_dtype),
                "y_fine": jnp.asarray(grid["y_fine"], dtype=coord_dtype),
            }
        )
    else:
        arrays.update(
            {
                "x_lo": jnp.asarray(grid["x_lo"], dtype=coord_dtype),
                "x_hi": jnp.asarray(grid["x_hi"], dtype=coord_dtype),
                "y_lo": jnp.asarray(grid["y_lo"], dtype=coord_dtype),
                "y_hi": jnp.asarray(grid["y_hi"], dtype=coord_dtype),
                "xf_lo": jnp.asarray(grid["xf_lo"], dtype=coord_dtype),
                "xf_hi": jnp.asarray(grid["xf_hi"], dtype=coord_dtype),
                "yf_lo": jnp.asarray(grid["yf_lo"], dtype=coord_dtype),
                "yf_hi": jnp.asarray(grid["yf_hi"], dtype=coord_dtype),
            }
        )
    meta = {
        "contiguous": bool(grid["contiguous"]),
        "non_overlapping_y": bool(grid.get("non_overlapping_y", False)),
        "res_um": float(grid["res_um"]),
        "n_sub": int(grid["n_sub"]),
        "uniform_z": bool(zaxis.uniform),
    }
    return arrays, meta


def _build_level_surfaces_greens(
    device: Any,
    p: dict[str, Array],
    offsets_z: Sequence[Array],
    z_shift: Any,
    dtype: Any,
) -> tuple[list[tuple[Any, tuple[Array, Array, Array] | None]], list[Array], list[Array]]:
    """Priority-ordered ``(level, wall_tuple)`` and ``(lows, highs)`` cell arrays for scalar/traced ``z_shift``."""
    nx, ny, nz = (int(x) for x in device.grid_shape)
    res_um = float(device._res_um())
    frame = device.grid_frame
    if frame is None or frame.uniform[2]:
        z0 = float(device.z_center) - 0.5 * nz * res_um
    else:
        z0 = float(device.z_center) - 0.5 * float(frame.edges[2][-1]) * res_um
    calc = jnp.float64 if jax.config.jax_enable_x64 else dtype
    z0_c = jnp.asarray(z0, dtype=calc)
    res_c = jnp.asarray(res_um, dtype=calc)
    z_sh_c = jnp.asarray(z_shift, dtype=calc)
    order = sorted(range(len(device.levels)), key=lambda i: (device.levels[i].mesh_order, i))
    tops = {lv.name: max(lv.zmin, lv.zmin + lv.thickness) for lv in device.levels}
    active: list[tuple[Any, tuple[Array, Array, Array] | None]] = []
    lows, highs = [], []
    for i in order:
        lv = device.levels[i]
        zb = ((jnp.asarray(p[f"zmin/{lv.name}"], dtype=calc) - z0_c) / res_c - z_sh_c).astype(dtype)
        t = (jnp.asarray(p[f"thickness/{lv.name}"], dtype=calc) / res_c).astype(dtype)
        angle = jnp.asarray(p[f"sidewall_angle/{lv.name}"], dtype=dtype)
        z_lo, z_hi = (zb, zb + t) if lv.thickness > 0 else (zb + t, zb)
        lo_nominal = lv.zmin if lv.thickness > 0 else lv.zmin + lv.thickness
        z_lo = jnp.broadcast_to(z_lo, (nx, ny))
        z_hi = jnp.broadcast_to(z_hi, (nx, ny))
        for r, dz in zip(device.roughness if offsets_z else (), offsets_z, strict=True):
            if r.level == lv.name:
                z_hi = z_hi + jnp.asarray(dz, dtype=dtype)
            elif r.fill_above and r.level in tops and abs(lo_nominal - tops[r.level]) < 1e-6:
                z_lo = z_lo + jnp.asarray(dz, dtype=dtype)
        wall = None if device._drawn_expr(lv)[0] == "fill" else (zb, t, angle)
        active.append((lv, wall))
        lows.append(z_lo.astype(dtype))
        highs.append(z_hi.astype(dtype))
    return active, lows, highs


def _build_material_tables(
    device: Any,
    ordered_levels: Sequence[Any],
    dtype: Any,
) -> tuple[list[str], np.ndarray, list[int], Array, Array, Array]:
    """Static material tables shared across all Yee shifts of a StackDevice."""
    names, eps_table = device._material_table()
    level_materials = [lv.material for lv in ordered_levels]
    mat_indices = [names.index(m) for m in (*level_materials, device.background)]
    level_eps = eps_table[mat_indices]
    scale = float(max(np.linalg.norm(e) for e in level_eps))
    vectors_np = np.stack(
        [
            np.array(
                [e[0, 0], e[1, 1], e[2, 2], math.sqrt(2.0) * e[0, 1], math.sqrt(2.0) * e[0, 2], math.sqrt(2.0) * e[1, 2]]
            )
            for e in level_eps
        ]
    ) / scale
    contrast2_np = np.sum((level_eps[:, None] - level_eps[None, :]) ** 2, axis=(-2, -1)) / (scale * scale)
    vectors = jnp.asarray(vectors_np, dtype=dtype)
    contrast2 = jnp.asarray(contrast2_np, dtype=dtype)
    onehot_np = np.zeros((len(ordered_levels) + 1, len(names)), dtype=float)
    for j, m_idx in enumerate(mat_indices):
        onehot_np[j, m_idx] = 1.0
    onehot = jnp.asarray(onehot_np, dtype=dtype)
    return names, eps_table, mat_indices, vectors, contrast2, onehot


def _greens_single_shift(
    device: Any,
    p: dict[str, Array],
    sources: dict[int, tuple[Array, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    derived_meta: dict[str, list[tuple[int, bool]]],
    mat_tables: tuple[list[str], np.ndarray, list[int], Array, Array, Array],
    grid: dict[str, Any],
    z_frame_lo: Array,
    z_frame_hi: Array,
    graded_z: bool,
    active: Sequence[tuple[Any, tuple[Array, Array, Array] | None]],
    lows_cells: Sequence[Array],
    highs_cells: Sequence[Array],
    apply_roughness: bool,
    res_um: float,
    dtype: Any,
    *,
    need_direction: bool,
    need_junction: bool,
) -> tuple[list[str], Array, Array, Any]:
    """Rasterize one Yee-shifted sample grid from pre-extracted geometries and material tables."""
    from .stack._corner_model import _vertical_tensor
    from .stack._layers import _KOTTKE_TINY
    from .stack._profiles import _point_slopes, _split_layer

    names, _, mat_indices, vectors, contrast2, onehot = mat_tables
    nx, ny, nz = (int(x) for x in device.grid_shape)
    coord_dtype = jnp.float64 if jax.config.jax_enable_x64 else dtype
    area_frame = jnp.asarray((grid["dx"][:, None] * grid["dy"][None, :]) / (res_um * res_um), dtype=dtype)
    cell_scale_frame = jnp.sqrt(area_frame)

    level_specs = [_level_kind_and_parts(device, lv, derived_meta, sources) for lv, _ in active]
    num_k = len(active)
    has_rough = bool(apply_roughness and device.roughness)

    def _level_gs(j: int, z_f: Array) -> tuple[Array, Array]:
        lv, wall = active[j]
        return _level_grow_and_slope(lv, wall, z_f, res_um, dtype)

    def _is_vertical(j: int) -> bool:
        lv, wall = active[j]
        if wall is None:
            return True
        if lv.z_to_bias is not None:
            return False
        _, _, angle = wall
        return _is_constant_zero(angle)

    is_vert = [_is_vertical(j) for j in range(num_k)]
    sloped_mask = jnp.asarray([not v for v in is_vert])
    z0_xy = jnp.zeros((nx, ny), dtype=dtype)
    zb0_xy = jnp.zeros((nx, ny, 6), dtype=dtype)
    zero_slope = jnp.zeros((), dtype=dtype)
    bg_vis = jnp.zeros((nx, ny, num_k + 1), dtype=dtype).at[..., num_k].set(jnp.ones((), dtype=dtype))
    bg_dir = jnp.zeros((nx, ny, 6), dtype=dtype)
    bg_mom = (
        jnp.broadcast_to(
            vectors[num_k][None, None, :, None] * jnp.asarray([1.0, 0.0, 0.0], dtype=dtype),
            (nx, ny, 6, 3),
        )
        if need_direction
        else jnp.zeros((nx, ny, 6, 3), dtype=dtype)
    )

    # Pre-rasterize all vertical levels ONCE across all z!
    vert_cache: dict[int, tuple[Array, Array, Array, Array]] = {}
    for j in range(num_k):
        if is_vert[j]:
            kind, info = level_specs[j]
            grow_static, _ = _level_gs(j, zero_slope)
            vert_cache[j] = _rasterize_level_slice(
                kind, info, grow_static, sources, grid, need_moments=need_direction, dtype=dtype
            )

    def _eval_level_at_z(j: int, z_f: Array, p_j: Array | None = None) -> tuple[Array, Array, Array, Array, Array]:
        if j in vert_cache:
            return *vert_cache[j], zero_slope

        def _do_lvl() -> tuple[Array, Array, Array, Array, Array]:
            kind, info = level_specs[j]
            grow_j, slope_j = _level_gs(j, z_f)
            f, mx, my, bs = _rasterize_level_slice(
                kind, info, grow_j, sources, grid, need_moments=need_direction, dtype=dtype
            )
            return f, mx, my, bs, slope_j

        def _zero_lvl() -> tuple[Array, Array, Array, Array, Array]:
            return z0_xy, z0_xy, z0_xy, zb0_xy, zero_slope

        if p_j is None:
            return _do_lvl()
        return jax.lax.cond(p_j, _do_lvl, _zero_lvl)

    # Multi-level overlap refinement helper (runs only where 2+ patterned levels have 0 < f < 1 simultaneously)
    def _eval_overlap_pixels(
        overlap_mask: Array,
        z_gauss: Array,
        present_s: Array,
        f_base: Array,
        mx_base: Array,
        my_base: Array,
        b_base: Array,
        slope_s: Array,
    ) -> tuple[Array, Array, Array]:
        # Standard analytic result everywhere
        vis, dir6, mom = _combine_levels_subinterval(
            f_base, mx_base, my_base, b_base, present_s, slope_s, vectors, contrast2, area_frame, cell_scale_frame,
            need_direction=need_direction,
        )
        pattern_flags = [
            level_specs[j][0] not in ("fill", "empty")
            and not (j == num_k - 1 and mat_indices[j] == mat_indices[num_k])
            for j in range(num_k)
        ]
        if sum(pattern_flags) < 2:
            return vis, dir6, mom
        # Check whether 2+ non-fill levels have 0 < f < 1 simultaneously
        is_pattern = jnp.asarray(pattern_flags)
        partial = present_s & is_pattern & (f_base > 1e-5) & (f_base < 1.0 - 1e-5)
        multi = overlap_mask | (jnp.sum(partial.astype(jnp.int32), axis=-1) >= 2)
        if not has_tracers(multi) and not bool(jnp.any(multi)):
            return vis, dir6, mom

        # Local 4x4 sub-cell Green's refinement on multi-level crossing pixels
        sub_m = 4
        x_lo_arr, x_hi_arr, y_lo_arr, y_hi_arr = (jax.lax.stop_gradient(a) for a in _grid_bounds(grid, coord_dtype))

        # Precompute offset segments per level at this z height
        seg_data: list[tuple[list[tuple[Array, Array, Array]], bool] | None] = []
        for j in range(num_k):
            kind, info = level_specs[j]
            if kind != "boundary":
                seg_data.append(None)
                continue
            slot_indices, negate, root_grow = info
            grow_j, _ = _level_gs(j, z_gauss)
            ug = -(grow_j + jnp.asarray(root_grow, dtype=dtype)) if negate else (grow_j + jnp.asarray(root_grow, dtype=dtype))
            slot_segs = []
            for idx in slot_indices:
                if idx in sources:
                    v_um, src, dst, succ, bnd, _, _, arc_c = sources[idx]
                    p0i, p1i, vi, _ = offset_boundary_segments(v_um, src, dst, succ, bnd, ug, arc_counts=arc_c)
                    slot_segs.append((p0i, p1i, vi))
            seg_data.append((slot_segs, negate) if slot_segs else None)

        src_sdf = {
            id0: (sources[id0][0], sources[id0][5], sources[id0][6], sources[id0][4])
            for id0 in sources
            if id0 >= 0
        }

        def _refine_chunk(c_idx: Array, c_mask: Array, c_xs: tuple[Array, Array]) -> tuple[Array, Array, Array]:
            c_pres, c_slope = c_xs
            ix = c_idx // ny
            iy = c_idx % ny
            xl, xh = x_lo_arr[ix], x_hi_arr[ix]
            yl, yh = y_lo_arr[iy], y_hi_arr[iy]
            t_lin = jnp.linspace(0.0, 1.0, sub_m + 1, dtype=coord_dtype)
            sx = xl[:, None] + (xh - xl)[:, None] * t_lin[None, :]  # (B, 5)
            sy = yl[:, None] + (yh - yl)[:, None] * t_lin[None, :]  # (B, 5)
            b_size = c_idx.shape[0]
            offs = (jnp.arange(sub_m, dtype=dtype) + 0.5 - 0.5 * sub_m) / sub_m
            sqrt12 = math.sqrt(12.0)

            def _one_pixel(p_idx: Array) -> tuple[Array, Array, Array]:
                xe_p, ye_p = sx[p_idx], sy[p_idx]
                f_sub_list, mx_sub_list, my_sub_list, b_sub_list = [], [], [], []
                for j in range(num_k):
                    kind, _ = level_specs[j]
                    if kind == "fill":
                        f_sub_list.append(jnp.ones((sub_m, sub_m), dtype=dtype))
                        mx_sub_list.append(jnp.zeros((sub_m, sub_m), dtype=dtype))
                        my_sub_list.append(jnp.zeros((sub_m, sub_m), dtype=dtype))
                        b_sub_list.append(jnp.zeros((sub_m, sub_m, 6), dtype=dtype))
                    elif kind == "sdf":
                        grow_j, _ = _level_gs(j, z_gauss)
                        fj, mxj, myj, bj = _sdf_rasterize_local_grid(
                            level_specs[j][1], src_sdf, grow_j, xe_p, ye_p, res_um, dtype
                        )
                        f_sub_list.append(fj)
                        mx_sub_list.append(mxj)
                        my_sub_list.append(myj)
                        b_sub_list.append(bj)
                    elif kind == "empty" or seg_data[j] is None:
                        v0 = 1.0 if (kind == "boundary" and level_specs[j][1][1]) else 0.0
                        f_sub_list.append(jnp.full((sub_m, sub_m), v0, dtype=dtype))
                        mx_sub_list.append(jnp.zeros((sub_m, sub_m), dtype=dtype))
                        my_sub_list.append(jnp.zeros((sub_m, sub_m), dtype=dtype))
                        b_sub_list.append(jnp.zeros((sub_m, sub_m, 6), dtype=dtype))
                    else:
                        slot_segs, neg_j = seg_data[j]
                        fj, mxj, myj, bj = _rasterize_slots_local(
                            slot_segs, xe_p, ye_p, res_um, True, neg_j, dtype
                        )
                        f_sub_list.append(fj)
                        mx_sub_list.append(mxj)
                        my_sub_list.append(myj)
                        b_sub_list.append(bj)

                f_s = jnp.stack(f_sub_list, axis=-1)
                mx_s = jnp.stack(mx_sub_list, axis=-1)
                my_s = jnp.stack(my_sub_list, axis=-1)
                b_s = jnp.stack(b_sub_list, axis=-2)
                pres_p = c_pres[p_idx]
                pres_sub = jnp.broadcast_to(pres_p, (sub_m, sub_m, num_k))
                sl_p = jnp.broadcast_to(c_slope[p_idx], (sub_m, sub_m, num_k))
                dx_p = ((xh[p_idx] - xl[p_idx]) / sub_m).astype(dtype)
                dy_p = ((yh[p_idx] - yl[p_idx]) / sub_m).astype(dtype)
                sub_area = ((dx_p * dy_p) / (res_um * res_um)).astype(dtype)
                sub_cs = jnp.sqrt(sub_area)
                vis_s, dir_s, mom_s = _combine_levels_subinterval(
                    f_s,
                    mx_s,
                    my_s,
                    b_s,
                    pres_sub,
                    sl_p,
                    vectors,
                    contrast2,
                    sub_area,
                    sub_cs,
                    need_direction=need_direction,
                    polygon_clip=True,
                    sub_dx=dx_p,
                    sub_dy=dy_p,
                )
                vis_p = jnp.mean(vis_s, axis=(0, 1))
                dir_p = jnp.sum(dir_s, axis=(0, 1))
                # First moment in parent pixel = mean over subcells(m_sub / s + sqrt(12) * offs * vis_sub)
                mom_0 = jnp.mean(mom_s[..., 0], axis=(0, 1))
                v_par = jnp.einsum("...l,lv->...v", vis_s, vectors, precision=_HIGHEST)
                mom_x = jnp.mean(mom_s[..., 1] / sub_m + sqrt12 * offs[:, None, None] * v_par, axis=(0, 1))
                mom_y = jnp.mean(mom_s[..., 2] / sub_m + sqrt12 * offs[None, :, None] * v_par, axis=(0, 1))
                return vis_p, dir_p, jnp.stack([mom_0, mom_x, mom_y], axis=-1)

            return jax.vmap(_one_pixel)(jnp.arange(b_size))

        ref_vis, ref_dir, ref_mom = sparse_voxel_map(
            _refine_chunk, multi, (present_s, slope_s), chunk_size=16, group_size=256
        )
        vis = jnp.where(multi[..., None], ref_vis, vis)
        if need_direction:
            dir6 = jnp.where(multi[..., None], ref_dir, dir6)
            mom = jnp.where(multi[..., None, None], ref_mom, mom)
        return vis, dir6, mom

    # Column-uniform scalar lows/highs per level (K,), used when no rough surface hits a slice
    lows_1d = jnp.stack([lo[0, 0] for lo in lows_cells], axis=-1)
    highs_1d = jnp.stack([hi[0, 0] for hi in highs_cells], axis=-1)
    knot_vals = []
    for lv, wall in active:
        if wall is not None and lv.z_to_bias is not None and len(lv.z_to_bias[0]) > 2:
            zb_w, t_w, _ = wall
            knot_vals.append(zb_w + jnp.asarray(lv.z_to_bias[0][1:-1], dtype=dtype) * t_w)
    knots_1d = jnp.sort(jnp.concatenate(knot_vals)) if knot_vals else None

    @jax.checkpoint
    def _slice_uniform(z_bounds: tuple[Array, Array]) -> tuple[Array, Array]:
        z_lo, z_hi = z_bounds
        bp, length, present = _layer_split_1d(lows_1d, highs_1d, z_lo, z_hi)  # (S+1,), (S,), (S, K)
        num_s = length.shape[0]
        gl_pts = jnp.asarray(_GL3_2, dtype=dtype)

        def _combine_stack(res: list[tuple[Array, Array, Array, Array, Array]], zg: Array, pres_g: Array) -> tuple[Array, Array, Array]:
            f_g = jnp.stack([r[0] for r in res], axis=-1)
            mx_g = jnp.stack([r[1] for r in res], axis=-1)
            my_g = jnp.stack([r[2] for r in res], axis=-1)
            b_g = jnp.stack([r[3] for r in res], axis=-2)
            sl_g = jnp.broadcast_to(jnp.stack([r[4] for r in res], axis=-1), (nx, ny, num_k))
            return _eval_overlap_pixels(
                jnp.zeros((nx, ny), dtype=bool), zg, pres_g, f_g, mx_g, my_g, b_g, sl_g
            )

        def _gauss_piece(pa: Array, pb: Array, p_s: Array) -> tuple[Array, Array, Array]:
            pres_g = jnp.broadcast_to(p_s, (nx, ny, num_k))

            def _one_gp(g_pt: Array) -> tuple[Array, Array, Array]:
                zg = pa + g_pt * (pb - pa)
                res = [_eval_level_at_z(j, zg, p_s[j]) for j in range(num_k)]
                return _combine_stack(res, zg, pres_g)

            vg, dg, mg = jax.lax.map(_one_gp, gl_pts)
            return jnp.mean(vg, axis=0), jnp.mean(dg, axis=0), jnp.mean(mg, axis=0)

        def _eval_subinterval(sub_in: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array]:
            za, zb, l_s, p_s = sub_in

            def _eval_cached(mid_z: Array) -> tuple[Array, Array, Array]:
                res = [
                    (*vert_cache[j], zero_slope) if j in vert_cache else (z0_xy, z0_xy, z0_xy, zb0_xy, zero_slope)
                    for j in range(num_k)
                ]
                return _combine_stack(res, mid_z, jnp.broadcast_to(p_s, (nx, ny, num_k)))

            def _do_sub() -> tuple[Array, Array, Array]:
                if all(is_vert):
                    return _eval_cached(0.5 * (za + zb))

                def _eval_sloped() -> tuple[Array, Array, Array]:
                    if knots_1d is None:
                        return _gauss_piece(za, zb, p_s)

                    def _with_knots() -> tuple[Array, Array, Array]:
                        sub_pts = jnp.concatenate([za[None], jnp.clip(knots_1d, za, zb), zb[None]])
                        span = jnp.maximum(zb - za, _TINY)

                        def _knot_piece(pair: tuple[Array, Array]) -> tuple[Array, Array, Array]:
                            pa, pb = pair
                            wm = (pb - pa) / span
                            vm, dm, mm = _gauss_piece(pa, pb, p_s)
                            return wm * vm, wm * dm, wm * mm

                        v_parts, d_parts, m_parts = jax.lax.map(_knot_piece, (sub_pts[:-1], sub_pts[1:]))
                        return jnp.sum(v_parts, axis=0), jnp.sum(d_parts, axis=0), jnp.sum(m_parts, axis=0)

                    return jax.lax.cond(zb - za > _KOTTKE_TINY, _with_knots, lambda: _gauss_piece(za, zb, p_s))

                return jax.lax.cond(jnp.any(p_s & sloped_mask), _eval_sloped, lambda: _eval_cached(0.5 * (za + zb)))

            # Never skip zero-length subintervals with zeros: tie surfaces rely on exact visible & background fractions
            return jax.lax.cond(jnp.any(p_s), _do_sub, lambda: (bg_vis, bg_dir, bg_mom))

        vis_subs, dir_subs, mom_subs = jax.lax.map(_eval_subinterval, (bp[:-1], bp[1:], length, present))
        visible = jnp.moveaxis(vis_subs, 0, -2)  # (nx, ny, S, K + 1)
        lat_dir = jnp.moveaxis(dir_subs, 0, -2)  # (nx, ny, S, 6)
        moments = jnp.moveaxis(mom_subs, 0, -3)  # (nx, ny, S, 6, 3)
        dz_k = z_hi - z_lo
        frac_lvl = jnp.sum(length[:, None] * visible, axis=-2) / dz_k
        w_mat = jnp.einsum("xyl,lm->mxy", frac_lvl, onehot, precision=_HIGHEST)
        if not need_direction:
            return w_mat, jnp.zeros((nx, ny, 6), dtype=dtype)
        d_lat = jnp.sum(length[:, None] * lat_dir, axis=-2)
        sl_z = jnp.zeros((nx, ny, num_s - 1, 2), dtype=dtype)
        d_vert = _vertical_tensor(length, bp, moments, sl_z, graded=graded_z) * area_frame[..., None]
        return w_mat, d_lat + d_vert

    # ---- Batched (vmap) slice path: no traced conds, so XLA runs whole slabs of z slices per kernel ----
    pattern_flags_all = [
        level_specs[j][0] not in ("fill", "empty")
        and not (j == num_k - 1 and mat_indices[j] == mat_indices[num_k])
        for j in range(num_k)
    ]

    def _vec_level_ok(j: int) -> bool:
        kind_j, info_j = level_specs[j]
        if kind_j in ("fill", "empty"):
            return True
        if kind_j != "boundary":
            return False
        return sum(1 for idx in info_j[0] if idx in sources) <= 1

    vec_ok = (
        _BATCHED_SLICES
        and not has_rough
        and knots_1d is None
        and sum(pattern_flags_all) < 2
        and all(_vec_level_ok(j) for j in range(num_k))
        and (need_direction or has_tracers(p) or has_tracers(z_frame_lo))
    )
    sloped_js = [j for j in range(num_k) if not is_vert[j]]

    def _sloped_source(j: int) -> int | None:
        kind_j, info_j = level_specs[j]
        if kind_j != "boundary":
            return None
        idxs = [idx for idx in info_j[0] if idx in sources]
        return idxs[0] if idxs else None

    def _static_pieces() -> tuple[dict[int, np.ndarray] | None, tuple[int, int] | None]:
        """Static split counts so each offset-boundary piece spans a few cells, and the line window for them.

        Sized from the nominal (``init_params``) geometry with a 2x margin; :func:`_needs_fallback` checks the
        actual pieces at every sample and falls back to the full-line path if any piece outgrows the window.
        """
        if not (_RASTER_WINDOW and vec_ok and grid["contiguous"] and sloped_js):
            return None, None
        g0 = sample_grid_edges_um(device, (0.0, 0.0, 0.0))
        xf = np.asarray(g0["x_fine"], dtype=np.float64)
        yf = np.asarray(g0["y_fine"], dtype=np.float64)
        dmin = float(min(np.min(np.diff(xf)), np.min(np.diff(yf))))
        target = 0.5 * (_RASTER_WINDOW - 1) * dmin
        topo0 = _static_source_topology(device)
        counts: dict[int, np.ndarray] = {}
        for j in sloped_js:
            idx = _sloped_source(j)
            if idx is None or idx in counts:
                continue
            if idx not in topo0:
                return None, None
            _, src, dst, _, bnd, _, _, arc_c = sources[idx]
            v_np = topo0[idx]
            b_idx = np.flatnonzero(bnd)
            lens = np.linalg.norm(v_np[dst[b_idx]] - v_np[src[b_idx]], axis=-1)
            ce = np.maximum(1, np.ceil(lens / target)).astype(np.int64)
            counts[idx] = np.concatenate([ce, np.ones(int(np.sum(arc_c)), dtype=np.int64)])
        win = (int(min(_RASTER_WINDOW, len(xf))), int(min(_RASTER_WINDOW, len(yf))))
        return counts, win

    piece_counts, raster_window = _static_pieces()

    def _sloped_segments(j: int, grow_j: Array) -> tuple[Array, Array, Array, Array] | None:
        """Offset (and split) boundary segments (single slot) of sloped level ``j`` at outward grow ``grow_j``."""
        idx = _sloped_source(j)
        if idx is None:
            return None
        _, (_, negate, root_grow) = level_specs[j]
        ug = -(grow_j + jnp.asarray(root_grow, dtype=dtype)) if negate else (grow_j + jnp.asarray(root_grow, dtype=dtype))
        v_um, src, dst, succ, bnd, _, _, arc_c = sources[idx]
        p0, p1, val, inv = offset_boundary_segments(v_um, src, dst, succ, bnd, ug, arc_counts=arc_c)
        if piece_counts is not None and idx in piece_counts:
            p0, p1, val = _split_segments(p0, p1, val, piece_counts[idx])
        return p0, p1, val, inv

    def _gauss_z(bp: Array) -> Array:
        gl = jnp.asarray(_GL3_2, dtype=dtype)
        return bp[..., :-1, None] + gl * (bp[..., 1:] - bp[..., :-1])[..., None]  # (..., S, G)

    def _slice_batched(z_bounds: tuple[Array, Array], with_sloped: bool = True) -> tuple[Array, Array]:
        """One z slice; ``with_sloped=False`` is the cheap form for slices no sloped level reaches (one
        sample per sub-interval: every present level is vertical there, so nothing varies along z)."""
        z_lo, z_hi = z_bounds
        bp, length, present = _layer_split_1d(lows_1d, highs_1d, z_lo, z_hi)  # (S+1,), (S,), (S, K)
        num_s = length.shape[0]
        zg = _gauss_z(bp) if with_sloped else 0.5 * (bp[:-1] + bp[1:])[:, None]  # (S, G)
        n_g = zg.shape[-1]
        zg_flat = zg.reshape(-1)
        pres_flat = jnp.repeat(present, n_g, axis=0)  # (S*G, K)

        sloped_out: dict[int, tuple[Array, Array, Array, Array, Array]] = {}
        for j in sloped_js if with_sloped else ():
            kind_j, info_j = level_specs[j]
            # Clamp samples into the level's own z-range: samples where it is absent are masked out by
            # ``present`` below, and clamping keeps their (unused) geometry physical so no NaN reaches AD.
            zc = jnp.clip(zg_flat, lows_1d[j], highs_1d[j])

            def _one(zv: Array, j_: int = j, kind_: str = kind_j, info_: Any = info_j) -> tuple[Array, ...]:
                grow_v, slope_v = _level_gs(j_, zv)
                f, mx, my, bs = _rasterize_level_slice(
                    kind_, info_, grow_v, sources, grid, need_moments=need_direction, dtype=dtype,
                    assume_no_inversion=True, window=raster_window, piece_counts=piece_counts,
                )
                return f, mx, my, bs, slope_v

            sloped_out[j] = jax.vmap(_one)(zc)

        def _combine_sample(n_idx: Array) -> tuple[Array, Array, Array]:
            fs, mxs, mys, bss, sls = [], [], [], [], []
            for j in range(num_k):
                if j in sloped_out:
                    f, mx, my, bs, sl = (a[n_idx] for a in sloped_out[j])
                elif j in vert_cache:
                    f, mx, my, bs = vert_cache[j]
                    sl = zero_slope
                else:
                    f, mx, my, bs, sl = z0_xy, z0_xy, z0_xy, zb0_xy, zero_slope
                fs.append(f)
                mxs.append(mx)
                mys.append(my)
                bss.append(bs)
                sls.append(sl)
            p_n = pres_flat[n_idx]
            vis, dir6, mom = _combine_levels_subinterval(
                jnp.stack(fs, axis=-1), jnp.stack(mxs, axis=-1), jnp.stack(mys, axis=-1), jnp.stack(bss, axis=-2),
                jnp.broadcast_to(p_n, (nx, ny, num_k)), jnp.broadcast_to(jnp.stack(sls, axis=-1), (nx, ny, num_k)),
                vectors, contrast2, area_frame, cell_scale_frame, need_direction=need_direction,
            )
            any_p = jnp.any(p_n)
            return (
                jnp.where(any_p, vis, bg_vis),
                jnp.where(any_p, dir6, bg_dir),
                jnp.where(any_p, mom, bg_mom),
            )

        vg, dg, mg = jax.vmap(_combine_sample)(jnp.arange(num_s * n_g))
        vis_subs = jnp.mean(vg.reshape(num_s, n_g, *vg.shape[1:]), axis=1)
        dir_subs = jnp.mean(dg.reshape(num_s, n_g, *dg.shape[1:]), axis=1)
        mom_subs = jnp.mean(mg.reshape(num_s, n_g, *mg.shape[1:]), axis=1)
        visible = jnp.moveaxis(vis_subs, 0, -2)  # (nx, ny, S, K + 1)
        lat_dir = jnp.moveaxis(dir_subs, 0, -2)  # (nx, ny, S, 6)
        moments = jnp.moveaxis(mom_subs, 0, -3)  # (nx, ny, S, 6, 3)
        dz_k = z_hi - z_lo
        frac_lvl = jnp.sum(length[:, None] * visible, axis=-2) / dz_k
        w_mat = jnp.einsum("xyl,lm->mxy", frac_lvl, onehot, precision=_HIGHEST)
        if not need_direction:
            return w_mat, jnp.zeros((nx, ny, 6), dtype=dtype)
        d_lat = jnp.sum(length[:, None] * lat_dir, axis=-2)
        sl_z = jnp.zeros((nx, ny, num_s - 1, 2), dtype=dtype)
        d_vert = _vertical_tensor(length, bp, moments, sl_z, graded=graded_z) * area_frame[..., None]
        return w_mat, d_lat + d_vert

    def _static_band() -> int:
        """Static number of z slices the sloped levels can reach: nominal extent plus one cell each side
        (Yee half-cell z shifts, small thickness/zmin changes); :func:`_needs_fallback` checks the actual one."""
        if not sloped_js:
            return 0
        frame = device.grid_frame
        res = float(device._res_um())
        if frame is None or frame.uniform[2]:
            z0 = float(device.z_center) - 0.5 * nz * res
        else:
            z0 = float(device.z_center) - 0.5 * float(frame.edges[2][-1]) * res
        zax = device._z_axis(0.0)
        zl = np.asarray(zax.lo, dtype=np.float64)
        zh = np.asarray(zax.hi, dtype=np.float64)
        lo = min(min(lv.zmin, lv.zmin + lv.thickness) for lv, _ in (active[j] for j in sloped_js))
        hi = max(max(lv.zmin, lv.zmin + lv.thickness) for lv, _ in (active[j] for j in sloped_js))
        lo_f, hi_f = (lo - z0) / res, (hi - z0) / res
        band = int(np.sum((zh >= lo_f - 1.0) & (zl <= hi_f + 1.0)))
        return int(min(nz, max(1, band)))

    band_len = _static_band()

    def _band_start_and_count() -> tuple[Array, Array]:
        lo = jnp.min(jnp.stack([lows_1d[j] for j in sloped_js]))
        hi = jnp.max(jnp.stack([highs_1d[j] for j in sloped_js]))
        inter = (z_frame_hi >= lo) & (z_frame_lo <= hi)
        k0 = jnp.where(jnp.any(inter), jnp.argmax(inter), 0)
        return jnp.clip(k0, 0, nz - band_len).astype(jnp.int32), jnp.sum(inter.astype(jnp.int32))

    def _batch_for(per_slice: int, n: int) -> int:
        return int(max(1, min(n, _SLICE_BATCH_ELEMS // max(per_slice, 1))))

    def _run_batched() -> tuple[Array, Array]:
        n_sub_g = int(grid.get("n_sub", 1) or 1)
        nxf, nyf = nx * n_sub_g, ny * n_sub_g
        n_seg = 0
        for j in sloped_js:
            segs = _sloped_segments(j, zero_slope)
            if segs is not None:
                n_seg = max(n_seg, int(segs[0].shape[0]))
        num_s_static = 2 * num_k + 1
        lines = (nxf + nyf + 4) if raster_window is None else (raster_window[0] + raster_window[1] + 2)
        per_sample = nxf * nyf * (12 * max(1, len(sloped_js))) + nx * ny * 48 + n_seg * lines * 24
        per_slice = num_s_static * len(_GL3_2) * per_sample
        # Every slice in the cheap vertical-only form ...
        w_all, d_all = jax.lax.map(
            jax.checkpoint(lambda zb: _slice_batched(zb, with_sloped=False)),
            (z_frame_lo, z_frame_hi),
            batch_size=_batch_for(num_s_static * nx * ny * 48, nz),
        )
        if not sloped_js:
            return w_all, d_all
        # ... then the band of slices the sloped levels reach, in full, written over them
        k0, _ = _band_start_and_count()
        zl_b = jax.lax.dynamic_slice_in_dim(z_frame_lo, k0, band_len)
        zh_b = jax.lax.dynamic_slice_in_dim(z_frame_hi, k0, band_len)
        w_b, d_b = jax.lax.map(jax.checkpoint(_slice_batched), (zl_b, zh_b), batch_size=_batch_for(per_slice, band_len))
        w_all = jax.lax.dynamic_update_slice_in_dim(w_all, w_b.astype(w_all.dtype), k0, axis=0)
        d_all = jax.lax.dynamic_update_slice_in_dim(d_all, d_b.astype(d_all.dtype), k0, axis=0)
        return w_all, d_all

    def _needs_fallback() -> Any:
        """Whether any present Gauss sample of a sloped level inverts its offset boundary (needs the SDF path),
        has a split piece crossing more grid lines than the raster window (needs the full-line path), or the
        sloped levels reach more z slices than the static band."""
        flags = []
        if sloped_js:
            flags.append(_band_start_and_count()[1] > band_len)
        bp_all, _, pres_all = jax.vmap(lambda a, b: _layer_split_1d(lows_1d, highs_1d, a, b))(z_frame_lo, z_frame_hi)
        zg_all = _gauss_z(bp_all)  # (nz, S, G)
        for j in sloped_js:
            zc = jnp.clip(zg_all, lows_1d[j], highs_1d[j]).reshape(-1)

            def _bad(zv: Array, j_: int = j) -> Array:
                grow_v, _ = _level_gs(j_, zv)
                segs = _sloped_segments(j_, grow_v)
                if segs is None:
                    return jnp.asarray(False)
                bad = segs[3]
                if raster_window is not None:
                    cx, cy = _max_line_crossings(segs[0], segs[1], segs[2], grid["x_fine"], grid["y_fine"])
                    bad = bad | (cx > raster_window[0]) | (cy > raster_window[1])
                return bad

            bad = jax.vmap(_bad)(zc).reshape(zg_all.shape)
            flags.append(jnp.any(bad & pres_all[..., j][..., None]))
        if not flags:
            return False
        return jax.tree.reduce(lambda a, b: a | b, flags)

    # ---- Fast path when all active surfaces are laterally flat (no roughness) ----
    if not has_rough:
        if vec_ok:
            any_inv_all = _needs_fallback()
            if not has_tracers(any_inv_all) and not bool(any_inv_all):
                w_slices, dir_slices = _run_batched()
            else:
                w_slices, dir_slices = jax.lax.cond(
                    any_inv_all,
                    lambda: jax.lax.map(_slice_uniform, (z_frame_lo, z_frame_hi)),
                    _run_batched,
                )
        else:
            w_slices, dir_slices = jax.lax.map(_slice_uniform, (z_frame_lo, z_frame_hi))
        weights = jnp.transpose(w_slices, (1, 2, 3, 0))
        direction = jnp.moveaxis(dir_slices, 0, 2)  # (nx, ny, nz, 6)
    else:
        # ---- Roughness path: column-dependent (nx, ny, K) surfaces ----
        tops_nom = {lv_r.name: max(lv_r.zmin, lv_r.zmin + lv_r.thickness) for lv_r, _ in active}
        rough_surfaces: list[Array] = []
        for j_r, (lv_r, _) in enumerate(active):
            lo_nom_r = lv_r.zmin if lv_r.thickness > 0 else lv_r.zmin + lv_r.thickness
            for r in device.roughness:
                if r.level == lv_r.name:
                    rough_surfaces.append(highs_cells[j_r])
                elif r.fill_above and r.level in tops_nom and abs(lo_nom_r - tops_nom[r.level]) < 1e-6:
                    rough_surfaces.append(lows_cells[j_r])

        lows_xy = jnp.stack(lows_cells, axis=-1)
        highs_xy = jnp.stack(highs_cells, axis=-1)
        surface_slopes = jnp.stack(
            [device._surface_slopes(x) for x in (*lows_cells, *highs_cells)], axis=-2
        )
        point_slopes = _point_slopes(lows_xy, surface_slopes)

        @jax.checkpoint
        def _slice_rough(z_bounds: tuple[Array, Array]) -> tuple[Array, Array]:
            z_lo, z_hi = z_bounds
            perm, bp, length, present = _split_layer(lows_xy, highs_xy, z_lo, top=z_hi)
            slopes_z = jnp.take_along_axis(point_slopes, perm[..., 1:-1, None], axis=-2)
            slice_has_lvl = jnp.any(present, axis=(0, 1, 2))
            # Evaluate at 3 Lobatto points (-1, 0, +1) -> quadratic interpolation on each column's Gauss points
            z_nodes = jnp.stack([z_lo, 0.5 * (z_lo + z_hi), z_hi])

            def _eval_node(zn: Array) -> tuple[Array, Array, Array, Array, Array]:
                res = [_eval_level_at_z(j, zn, slice_has_lvl[j]) for j in range(num_k)]
                return (
                    jnp.stack([r[0] for r in res], axis=-1),
                    jnp.stack([r[1] for r in res], axis=-1),
                    jnp.stack([r[2] for r in res], axis=-1),
                    jnp.stack([r[3] for r in res], axis=-2),
                    jnp.stack([r[4] for r in res], axis=-1),
                )

            f_nodes, mx_nodes, my_nodes, b_nodes, sl_nodes = jax.lax.map(_eval_node, z_nodes)
            dz_k = z_hi - z_lo
            gl_pts = jnp.asarray(_GL3_2, dtype=dtype)

            def _interp3(arr: Array, xi: Array) -> Array:
                c0 = 0.5 * xi * (xi - 1.0)
                c1 = 1.0 - xi * xi
                c2 = 0.5 * xi * (xi + 1.0)
                while c0.ndim < arr.ndim - 1:
                    c0 = c0[..., None]
                    c1 = c1[..., None]
                    c2 = c2[..., None]
                return c0 * arr[0] + c1 * arr[1] + c2 * arr[2]

            def _eval_sub_rough(sub_in: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array]:
                ba, bb, l_s, p_s = sub_in

                def _do_sub() -> tuple[Array, Array, Array]:
                    def _one_gp(g_pt: Array) -> tuple[Array, Array, Array]:
                        zg = ba + g_pt * (bb - ba)
                        xi = 2.0 * (zg - z_lo) / dz_k - 1.0
                        f_g = jnp.clip(_interp3(f_nodes, xi), 0.0, 1.0)
                        mx_g = _interp3(mx_nodes, xi)
                        my_g = _interp3(my_nodes, xi)
                        b_g = _interp3(b_nodes, xi)
                        sl_g = jnp.broadcast_to(_interp3(sl_nodes[:, None, None, :], xi), (nx, ny, num_k))
                        return _combine_levels_subinterval(
                            f_g, mx_g, my_g, b_g, p_s, sl_g, vectors, contrast2, area_frame, cell_scale_frame,
                            need_direction=need_direction,
                        )

                    vg, dg, mg = jax.lax.map(_one_gp, gl_pts)
                    return jnp.mean(vg, axis=0), jnp.mean(dg, axis=0), jnp.mean(mg, axis=0)

                return jax.lax.cond(jnp.any(p_s), _do_sub, lambda: (bg_vis, bg_dir, bg_mom))

            vis_subs, dir_subs, mom_subs = jax.lax.map(
                _eval_sub_rough,
                (
                    jnp.moveaxis(bp[..., :-1], 2, 0),
                    jnp.moveaxis(bp[..., 1:], 2, 0),
                    jnp.moveaxis(length, 2, 0),
                    jnp.moveaxis(present, 2, 0),
                ),
            )
            visible = jnp.moveaxis(vis_subs, 0, -2)
            lat_dir = jnp.moveaxis(dir_subs, 0, -2)
            moments = jnp.moveaxis(mom_subs, 0, -3)
            frac_lvl = jnp.sum(length[..., None] * visible, axis=-2) / dz_k
            w_mat = jnp.einsum("xyl,lm->mxy", frac_lvl, onehot, precision=_HIGHEST)
            if not need_direction:
                return w_mat, jnp.zeros((nx, ny, 6), dtype=dtype)
            d_lat = jnp.sum(length[..., None] * lat_dir, axis=-2)
            d_vert = _vertical_tensor(length, bp, moments, slopes_z, graded=graded_z) * area_frame[..., None]
            return w_mat, d_lat + d_vert

        def _one_slice(z_bounds: tuple[Array, Array]) -> tuple[Array, Array]:
            if not rough_surfaces:
                return _slice_uniform(z_bounds)
            z_lo, z_hi = z_bounds
            slice_is_rough = jnp.any(
                jnp.stack(
                    [(jnp.max(jax.lax.stop_gradient(s)) >= z_lo) & (jnp.min(jax.lax.stop_gradient(s)) <= z_hi) for s in rough_surfaces]
                )
            )
            return jax.lax.cond(slice_is_rough, lambda: _slice_rough(z_bounds), lambda: _slice_uniform(z_bounds))

        w_slices, dir_slices = jax.lax.map(_one_slice, (z_frame_lo, z_frame_hi))
        weights = jnp.transpose(w_slices, (1, 2, 3, 0))
        direction = jnp.moveaxis(dir_slices, 0, 2)

    # Axes of one cell have no interface normal along them
    keep = np.asarray([n > 1 for n in device.grid_shape], dtype=float)
    keep6 = jnp.asarray([keep[0] ** 2, keep[1] ** 2, keep[2] ** 2, keep[0] * keep[1], keep[0] * keep[2], keep[1] * keep[2]], dtype=dtype)
    direction = direction * keep6

    junction_ctx = None
    if need_direction and need_junction:
        junction_ctx = greens_junction_cells(
            device, p, sources, derived_meta, grid, active, lows_cells, highs_cells,
            level_specs, mat_indices, vectors, contrast2, z_frame_lo, z_frame_hi, apply_roughness, dtype
        )
    return names, weights, direction, junction_ctx


def greens_device_interfaces(
    device: Any,
    params: dict[str, Array] | None,
    noise_key: Array | None,
    apply_roughness: bool,
    shift: tuple[float, float, float],
    *,
    need_direction: bool = True,
    need_junction: bool = True,
) -> tuple[list[str], Array, Array, Any]:
    """Compute material volume fractions, 6-component interface dyads, and junction context via Green's theorem."""
    p = {**device.init_params(), **(params or {})}
    nx, ny, nz = (int(x) for x in device.grid_shape)
    dtype = device._config.dtype
    res_um = float(device._res_um())
    names = compute_ordered_names(device.materials)
    zaxis = device._z_axis(shift[2])
    offsets_z = device.interface_offsets(noise_key) if apply_roughness and device.roughness else []
    active, lows_cells, highs_cells = _build_level_surfaces_greens(device, p, offsets_z, zaxis.shift, dtype)
    if not active:
        w = jnp.zeros((len(names), nx, ny, nz), dtype=dtype).at[names.index(device.background)].set(jnp.ones((), dtype=dtype))
        return names, w, jnp.zeros((nx, ny, nz, 6), dtype=dtype), None

    grid = sample_grid_edges_um(device, shift)
    sources, derived_meta = _extract_geometries_um(device, p, dtype)
    mat_tables = _build_material_tables(device, [lv for lv, _ in active], dtype)
    z_frame_lo = jnp.asarray(zaxis.lo, dtype=dtype)
    z_frame_hi = jnp.asarray(zaxis.hi, dtype=dtype)
    return _greens_single_shift(
        device, p, sources, derived_meta, mat_tables, grid, z_frame_lo, z_frame_hi, not zaxis.uniform,
        active, lows_cells, highs_cells, apply_roughness, res_um, dtype,
        need_direction=need_direction, need_junction=need_junction,
    )


def _prepare_yee_batch_context(
    device: Any,
    params: dict[str, Array] | None,
    noise_key: Array | None,
    apply_roughness: bool,
    shifts: Sequence[tuple[float, float, float]],
) -> dict[str, Any]:
    """Build hoisted geometry, material, and stacked per-shift grid state for batched Yee evaluation."""
    p = {**device.init_params(), **(params or {})}
    dtype = device._config.dtype
    names, eps_table = device._material_table()
    offsets_z = device.interface_offsets(noise_key) if apply_roughness and device.roughness else []
    order = sorted(range(len(device.levels)), key=lambda i: (device.levels[i].mesh_order, i))
    ordered_levels = [device.levels[i] for i in order]
    sources, derived_meta = _extract_geometries_um(device, p, dtype) if ordered_levels else ({}, {})
    mat_tables = _build_material_tables(device, ordered_levels, dtype) if ordered_levels else (names, eps_table, [], jnp.zeros((0,), dtype=dtype), jnp.zeros((0,), dtype=dtype), jnp.zeros((0,), dtype=dtype))
    prep = [_prepare_shift_grid(device, sh, dtype) for sh in shifts]
    shift_arrays = jax.tree.map(lambda *xs: jnp.stack(xs), *[a for a, _ in prep])
    keep = np.asarray([n > 1 for n in device.grid_shape], dtype=float)
    keep_33 = jnp.asarray(np.outer(keep, keep), dtype=dtype)
    return {
        "p": p,
        "dtype": dtype,
        "names": names,
        "eps_table": eps_table,
        "offsets_z": offsets_z,
        "empty": not ordered_levels,
        "sources": sources,
        "derived_meta": derived_meta,
        "mat_tables": mat_tables,
        "shift_arrays": shift_arrays,
        "meta0": prep[0][1],
        "graded_z": any(not m["uniform_z"] for _, m in prep),
        "keep_33": keep_33,
        "res_um": float(device._res_um()),
    }


def greens_yee_diagonal(
    device: Any,
    params: dict[str, Array] | None,
    noise_key: Array | None,
    apply_roughness: bool,
    shifts: Sequence[tuple[float, float, float]],
) -> tuple[list[str], Array, Array]:
    """Compute diagonal inverse permittivities across all Yee ``shifts`` inside one ``jax.lax.map``."""
    ctx = _prepare_yee_batch_context(device, params, noise_key, apply_roughness, shifts)
    names = ctx["names"]
    eps_table = ctx["eps_table"]
    dtype = ctx["dtype"]
    nx, ny, nz = (int(x) for x in device.grid_shape)
    s_cnt = len(shifts)
    if ctx["empty"]:
        w = jnp.zeros((len(names), nx, ny, nz), dtype=dtype).at[names.index(device.background)].set(jnp.ones((), dtype=dtype))
        inv_bg = np.linalg.inv(eps_table[names.index(device.background)])
        diag_bg = jnp.asarray(np.diag(inv_bg), dtype=dtype)
        return names, jnp.broadcast_to(w, (s_cnt, len(names), nx, ny, nz)), jnp.broadcast_to(diag_bg, (s_cnt, nx, ny, nz, 3))

    meta0 = ctx["meta0"]

    @jax.checkpoint
    def _eval_shift(grid_arr: dict[str, Array]) -> tuple[Array, Array]:
        grid_s = {
            **grid_arr,
            "contiguous": meta0["contiguous"],
            "non_overlapping_y": meta0["non_overlapping_y"],
            "res_um": meta0["res_um"],
            "n_sub": meta0["n_sub"],
        }
        active_s, lows_s, highs_s = _build_level_surfaces_greens(device, ctx["p"], ctx["offsets_z"], grid_arr["z_shift"], dtype)
        _, w_s, d6_s, _ = _greens_single_shift(
            device, ctx["p"], ctx["sources"], ctx["derived_meta"], ctx["mat_tables"],
            grid_s, grid_arr["z_frame_lo"], grid_arr["z_frame_hi"], ctx["graded_z"],
            active_s, lows_s, highs_s, apply_roughness, ctx["res_um"], dtype,
            need_direction=True, need_junction=False,
        )
        d_diag = d6_s[..., :3].astype(dtype) * jnp.diagonal(ctx["keep_33"])
        inv_diag_s = homogenize_diagonal(eps_table, jnp.moveaxis(w_s, 0, -1), d_diag).astype(dtype)
        return w_s.astype(dtype), inv_diag_s

    w_all, inv_diag_all = jax.lax.map(_eval_shift, ctx["shift_arrays"])
    return names, w_all, inv_diag_all


def greens_yee_full(
    device: Any,
    params: dict[str, Array] | None,
    noise_key: Array | None,
    apply_roughness: bool,
    shifts: Sequence[tuple[float, float, float]],
    capacity: int | None,
) -> tuple[list[str], Array, Array, dict[str, Array]]:
    """Compute full symmetric 6-entry inverse permittivities across all Yee ``shifts`` inside one ``jax.lax.map``."""
    ctx = _prepare_yee_batch_context(device, params, noise_key, apply_roughness, shifts)
    names = ctx["names"]
    eps_table = ctx["eps_table"]
    dtype = ctx["dtype"]
    nx, ny, nz = (int(x) for x in device.grid_shape)
    s_cnt = len(shifts)
    if ctx["empty"]:
        w = jnp.zeros((len(names), nx, ny, nz), dtype=dtype).at[names.index(device.background)].set(jnp.ones((), dtype=dtype))
        inv_bg = np.linalg.inv(eps_table[names.index(device.background)])
        inv6_bg = to_six(jnp.asarray(inv_bg, dtype=dtype))
        st_empty = {
            "junctions": jnp.zeros((), dtype=jnp.int32),
            "junction_overflow": jnp.zeros((), dtype=jnp.int32),
            "corner_dropped": jnp.zeros((), dtype=jnp.int32),
            "corner_uncertified": jnp.zeros((), dtype=jnp.int32),
            "closure_residual": jnp.zeros((), dtype=dtype),
        }
        return names, jnp.broadcast_to(w, (s_cnt, len(names), nx, ny, nz)), jnp.broadcast_to(inv6_bg, (s_cnt, nx, ny, nz, 6)), st_empty

    meta0 = ctx["meta0"]

    @jax.checkpoint
    def _eval_shift(grid_arr: dict[str, Array]) -> tuple[Array, Array, dict[str, Array]]:
        grid_s = {
            **grid_arr,
            "contiguous": meta0["contiguous"],
            "non_overlapping_y": meta0["non_overlapping_y"],
            "res_um": meta0["res_um"],
            "n_sub": meta0["n_sub"],
        }
        active_s, lows_s, highs_s = _build_level_surfaces_greens(device, ctx["p"], ctx["offsets_z"], grid_arr["z_shift"], dtype)
        _, w_s, d6_s, jctx_s = _greens_single_shift(
            device, ctx["p"], ctx["sources"], ctx["derived_meta"], ctx["mat_tables"],
            grid_s, grid_arr["z_frame_lo"], grid_arr["z_frame_hi"], ctx["graded_z"],
            active_s, lows_s, highs_s, apply_roughness, ctx["res_um"], dtype,
            need_direction=True, need_junction=True,
        )
        d_mat = from_six(d6_s).astype(dtype) * ctx["keep_33"]
        jfn = jctx_s["junction_cell_fn"] if isinstance(jctx_s, dict) else None
        jxs = jctx_s["junction_xs"] if isinstance(jctx_s, dict) else None
        inv6_s, st_s = homogenize_full(
            eps_table,
            jnp.moveaxis(w_s, 0, -1),
            d_mat,
            jfn,
            junction_xs=jxs,
            junction_low=JUNCTION_LOW,
            junction_high=JUNCTION_HIGH,
            max_items=capacity,
        )
        return w_s.astype(dtype), inv6_s.astype(dtype), st_s

    w_all, inv6_all, st_all = jax.lax.map(_eval_shift, ctx["shift_arrays"])
    status = {
        "junctions": jnp.sum(st_all["junctions"], dtype=jnp.int32),
        "junction_overflow": jnp.sum(st_all["junction_overflow"], dtype=jnp.int32),
        "corner_dropped": jnp.zeros((), dtype=jnp.int32),
        "corner_uncertified": jnp.zeros((), dtype=jnp.int32),
        "closure_residual": jnp.max(st_all["closure_residual"]),
    }
    return names, w_all, inv6_all, status


def greens_junction_cells(
    device: Any,
    params: dict[str, Array],
    sources: dict[int, tuple[Array, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    derived_meta: dict[str, list[tuple[int, bool]]],
    grid: dict[str, Any],
    active: Sequence[tuple[Any, Any]],
    lows_cells: Sequence[Array],
    highs_cells: Sequence[Array],
    level_specs: Sequence[tuple[str, Any]],
    mat_indices: Sequence[int],
    vectors: Array,
    contrast2: Array,
    z_frame_lo: Array,
    z_frame_hi: Array,
    apply_roughness: bool,
    dtype: Any,
) -> dict[str, Any]:
    """Build the chunk-evaluated sub-cell geometry closure for ``homogenize_full``."""
    from .stack._corner_model import _vertical_tensor
    from .stack._junctions import _match_fractions
    from .stack._profiles import _point_slopes, _split_layer

    nx, ny, nz = (int(x) for x in device.grid_shape)
    jx, jy, jz = (int(x) for x in device.junction_cells)
    sx = 1 if nx == 1 else jx
    sy = 1 if ny == 1 else jy
    sz = 1 if nz == 1 else jz
    res_um = grid["res_um"]
    num_k = len(active)
    num_mats = len(compute_ordered_names(device.materials))
    onehot_np = np.zeros((num_k + 1, num_mats), dtype=float)
    for j, m_idx in enumerate(mat_indices):
        onehot_np[j, m_idx] = 1.0
    onehot = jnp.asarray(onehot_np, dtype=dtype)

    coord_dtype = jnp.float64 if jax.config.jax_enable_x64 else dtype
    x_lo_arr, x_hi_arr, y_lo_arr, y_hi_arr = (jax.lax.stop_gradient(a) for a in _grid_bounds(grid, coord_dtype))
    z_lo_arr = jax.lax.stop_gradient(jnp.asarray(z_frame_lo, dtype=dtype))
    z_hi_arr = jax.lax.stop_gradient(jnp.asarray(z_frame_hi, dtype=dtype))

    has_rough = bool(apply_roughness and device.roughness)
    if has_rough:
        lows_xy = jnp.stack(lows_cells, axis=-1)
        highs_xy = jnp.stack(highs_cells, axis=-1)
        surface_slopes = jnp.stack(
            [device._surface_slopes(x) for x in (*lows_cells, *highs_cells)], axis=-2
        )
        point_slopes = _point_slopes(lows_xy, surface_slopes)
        lows_1d = highs_1d = None
        junction_xs: Any = (
            jnp.broadcast_to(lows_xy[:, :, None, :], (nx, ny, nz, num_k)),
            jnp.broadcast_to(highs_xy[:, :, None, :], (nx, ny, nz, num_k)),
            jnp.broadcast_to(point_slopes[:, :, None, :, :], (nx, ny, nz, *point_slopes.shape[2:])),
        )
    else:
        lows_1d = jnp.stack([lo[0, 0] for lo in lows_cells], axis=-1)
        highs_1d = jnp.stack([hi[0, 0] for hi in highs_cells], axis=-1)
        junction_xs = None

    z0_sub = jnp.zeros((sx, sy), dtype=dtype)
    o0_sub = jnp.ones((sx, sy), dtype=dtype)
    zb0_sub = jnp.zeros((sx, sy, 6), dtype=dtype)
    zero_slope = jnp.zeros((), dtype=dtype)
    z3_sub = jnp.zeros((3, sx, sy), dtype=dtype)
    zb3_sub = jnp.zeros((3, sx, sy, 6), dtype=dtype)
    zs3 = jnp.zeros((3,), dtype=dtype)
    bg_vis_sub = jnp.zeros((sx, sy, num_k + 1), dtype=dtype).at[..., num_k].set(jnp.ones((), dtype=dtype))
    bg_dir_sub = jnp.zeros((sx, sy, 6), dtype=dtype)
    bg_mom_sub = jnp.broadcast_to(
        vectors[num_k][None, None, :, None] * jnp.asarray([1.0, 0.0, 0.0], dtype=dtype),
        (sx, sy, 6, 3),
    )
    bg_frac_vox = jnp.zeros((sx, sy, sz, num_k + 1), dtype=dtype).at[..., num_k].set(jnp.ones((), dtype=dtype))
    zero_dir_vox = jnp.zeros((sx, sy, sz, 6), dtype=dtype)

    def _level_gs_j(j: int, z_f: Array) -> tuple[Array, Array]:
        lv, wall = active[j]
        return _level_grow_and_slope(lv, wall, z_f, res_um, dtype)

    def _is_vert_lvl(j: int) -> bool:
        lv, wall = active[j]
        if wall is None:
            return True
        if lv.z_to_bias is not None:
            return False
        return _is_constant_zero(wall[2])

    is_vert_j = [_is_vert_lvl(j) for j in range(num_k)]
    g_pts_arr = jnp.asarray((0.5,) if all(is_vert_j) else _GL3_2, dtype=dtype)
    src_sdf_all = {
        id0: (sources[id0][0], sources[id0][5], sources[id0][6], sources[id0][4])
        for id0 in sources
        if id0 >= 0
    }
    vert_slot_segs: dict[int, tuple[list[tuple[Array, Array, Array]], bool]] = {}
    for j in range(num_k):
        kind_j, info_j = level_specs[j]
        if is_vert_j[j] and kind_j == "boundary":
            slot_indices, negate, root_grow = info_j
            grow_j, _ = _level_gs_j(j, zero_slope)
            ug = -(grow_j + jnp.asarray(root_grow, dtype=dtype)) if negate else (
                grow_j + jnp.asarray(root_grow, dtype=dtype)
            )
            segs_j = []
            for idx_s in slot_indices:
                if idx_s in sources:
                    v_um, src_s, dst_s, succ_s, bnd_s, _, _, arc_s = sources[idx_s]
                    p0i, p1i, vi, _ = offset_boundary_segments(
                        v_um, src_s, dst_s, succ_s, bnd_s, ug, arc_counts=arc_s
                    )
                    segs_j.append((p0i, p1i, vi))
            if segs_j:
                vert_slot_segs[j] = (segs_j, negate)

    def junction_cell_fn(chunk_indices: Array, chunk_mask: Array, chunk_xs: Any) -> tuple[Array, Array, Array]:
        if has_rough:
            chunk_target, (chunk_lo, chunk_hi, chunk_ps) = chunk_xs
        else:
            chunk_target = chunk_xs
        b_size = chunk_indices.shape[0]
        iz = chunk_indices % nz
        col = chunk_indices // nz
        iy = col % ny
        ix = col // ny

        xl, xh = x_lo_arr[ix], x_hi_arr[ix]
        yl, yh = y_lo_arr[iy], y_hi_arr[iy]
        zl, zh = z_lo_arr[iz], z_hi_arr[iz]
        sub_dz = (zh - zl) / sz
        sub_dx = (xh - xl) / sx
        sub_dy = (yh - yl) / sy
        sub_area_frame = ((sub_dx * sub_dy) / (res_um * res_um)).astype(dtype)
        sub_cs_frame = jnp.sqrt(sub_area_frame)

        sides = jnp.stack(
            [((xh - xl) / res_um).astype(dtype), ((yh - yl) / res_um).astype(dtype), zh - zl],
            axis=-1,
        ).astype(dtype)  # (B, 3)

        sub_z_edges = zl[:, None] + jnp.arange(sz + 1, dtype=dtype)[None, :] * sub_dz[:, None]
        t_x = jnp.linspace(0.0, 1.0, sx + 1, dtype=coord_dtype)[None, :]
        t_y = jnp.linspace(0.0, 1.0, sy + 1, dtype=coord_dtype)[None, :]
        sub_x_edges = xl[:, None] + (xh - xl)[:, None] * t_x  # (B, sx + 1)
        sub_y_edges = yl[:, None] + (yh - yl)[:, None] * t_y  # (B, sy + 1)

        def _one_voxel(v_idx: Array) -> tuple[Array, Array]:
            if has_rough:
                lo_v = chunk_lo[v_idx]
                hi_v = chunk_hi[v_idx]
                ps_v = chunk_ps[v_idx]
            else:
                assert lows_1d is not None and highs_1d is not None
                lo_v = lows_1d
                hi_v = highs_1d
                ps_v = None
            zl_v, zh_v = zl[v_idx], zh[v_idx]
            dz_v = jnp.maximum(zh_v - zl_v, jnp.asarray(_TINY, dtype=dtype))
            vox_has_lvl = (hi_v > zl_v) & (lo_v < zh_v)
            z_nodes_v = jnp.stack([zl_v, 0.5 * (zl_v + zh_v), zh_v])
            xe_v = sub_x_edges[v_idx]
            ye_v = sub_y_edges[v_idx]
            dx_sub_v = (xe_v[1:] - xe_v[:-1])[:, None].astype(dtype)
            dy_sub_v = (ye_v[1:] - ye_v[:-1])[None, :].astype(dtype)
            ar_f = sub_area_frame[v_idx]
            cs_f = sub_cs_frame[v_idx]

            # Hoist vertical level rasterizations (1x/voxel) and sloped levels (3 Lobatto z-nodes/voxel)
            lvl_cache: list[tuple[str, Any]] = []
            for j in range(num_k):
                kind, info = level_specs[j]
                if kind == "fill":
                    lvl_cache.append(("const", (o0_sub, z0_sub, z0_sub, zb0_sub, zero_slope)))
                    continue
                if kind == "empty":
                    lvl_cache.append(("const", (z0_sub, z0_sub, z0_sub, zb0_sub, zero_slope)))
                    continue

                def _eval_at_z(
                    zn: Array,
                    j_idx: int = j,
                    kind_j: str = kind,
                    info_j: Any = info,
                ) -> tuple[Array, Array, Array, Array, Array]:
                    if kind_j == "sdf":
                        grow_j, slope_j = _level_gs_j(j_idx, zn)
                        fj, mxj, myj, bj = _sdf_rasterize_local_grid(
                            info_j, src_sdf_all, grow_j, xe_v, ye_v, res_um, dtype
                        )
                        return fj, mxj, myj, bj, slope_j
                    slot_indices, negate, root_grow = info_j
                    if j_idx in vert_slot_segs:
                        slot_segs, neg_j = vert_slot_segs[j_idx]
                        slope_j = zero_slope
                    else:
                        grow_j, slope_j = _level_gs_j(j_idx, zn)
                        ug = -(grow_j + jnp.asarray(root_grow, dtype=dtype)) if negate else (
                            grow_j + jnp.asarray(root_grow, dtype=dtype)
                        )
                        slot_segs = []
                        for idx_s in slot_indices:
                            if idx_s in sources:
                                v_um, src_s, dst_s, succ_s, bnd_s, _, _, arc_s = sources[idx_s]
                                p0i, p1i, vi, _ = offset_boundary_segments(
                                    v_um, src_s, dst_s, succ_s, bnd_s, ug, arc_counts=arc_s
                                )
                                slot_segs.append((p0i, p1i, vi))
                        neg_j = negate
                        if not slot_segs:
                            v_def = 1.0 if negate else 0.0
                            return jnp.full((sx, sy), v_def, dtype=dtype), z0_sub, z0_sub, zb0_sub, slope_j
                    fj, mxj, myj, bj = _rasterize_slots_local(
                        slot_segs, xe_v, ye_v, res_um, True, neg_j, dtype
                    )
                    return fj, mxj, myj, bj, slope_j

                if is_vert_j[j]:
                    val = jax.lax.cond(
                        vox_has_lvl[j],
                        lambda fn=_eval_at_z: fn(zero_slope),
                        lambda: (z0_sub, z0_sub, z0_sub, zb0_sub, zero_slope),
                    )
                    lvl_cache.append(("vert", val))
                else:
                    nodes = jax.lax.cond(
                        vox_has_lvl[j],
                        lambda fn=_eval_at_z: jax.vmap(fn)(z_nodes_v),
                        lambda: (z3_sub, z3_sub, z3_sub, zb3_sub, zs3),
                    )
                    lvl_cache.append(("sloped", nodes))

            def _one_z_sub(k_z: Array) -> tuple[Array, Array]:
                za, zb = sub_z_edges[v_idx, k_z], sub_z_edges[v_idx, k_z + 1]
                perm, bp, length, present = _split_layer(lo_v, hi_v, za, top=zb)
                num_s = length.shape[0]
                sl_z = (
                    jnp.zeros((sx, sy, num_s - 1, 2), dtype=dtype)
                    if ps_v is None
                    else jnp.broadcast_to(jnp.take_along_axis(ps_v, perm[1:-1, None], axis=0), (sx, sy, num_s - 1, 2))
                )

                def _one_subinterval(s_in: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array]:
                    za_s, zb_s, l_s, pres_vec = s_in

                    def _do_zsub() -> tuple[Array, Array, Array]:
                        pres_s = jnp.broadcast_to(pres_vec, (sx, sy, num_k))

                        def _one_gp(g_pt: Array) -> tuple[Array, Array, Array]:
                            zm = za_s + g_pt * (zb_s - za_s)
                            xi_z = 2.0 * (zm - zl_v) / dz_v - 1.0
                            c0 = 0.5 * xi_z * (xi_z - 1.0)
                            c1 = 1.0 - xi_z * xi_z
                            c2 = 0.5 * xi_z * (xi_z + 1.0)
                            f_k, mx_k, my_k, b_k, sl_k = [], [], [], [], []
                            for j in range(num_k):
                                mode_j, payload = lvl_cache[j]
                                if mode_j == "sloped":
                                    fn, mxn, myn, bn, sln = payload
                                    fj = jnp.clip(c0 * fn[0] + c1 * fn[1] + c2 * fn[2], 0.0, 1.0)
                                    mxj = c0 * mxn[0] + c1 * mxn[1] + c2 * mxn[2]
                                    myj = c0 * myn[0] + c1 * myn[1] + c2 * myn[2]
                                    bj = c0 * bn[0] + c1 * bn[1] + c2 * bn[2]
                                    slj = c0 * sln[0] + c1 * sln[1] + c2 * sln[2]
                                else:
                                    fj, mxj, myj, bj, slj = payload
                                f_k.append(fj)
                                mx_k.append(mxj)
                                my_k.append(myj)
                                b_k.append(bj)
                                sl_k.append(slj)

                            f_mat = jnp.stack(f_k, axis=-1)
                            mx_mat = jnp.stack(mx_k, axis=-1)
                            my_mat = jnp.stack(my_k, axis=-1)
                            b_mat = jnp.stack(b_k, axis=-2)
                            sl_mat = jnp.broadcast_to(jnp.stack(sl_k, axis=-1), (sx, sy, num_k))
                            return _combine_levels_subinterval(
                                f_mat,
                                mx_mat,
                                my_mat,
                                b_mat,
                                pres_s,
                                sl_mat,
                                vectors,
                                contrast2,
                                ar_f,
                                cs_f,
                                need_direction=True,
                                polygon_clip=True,
                                sub_dx=dx_sub_v,
                                sub_dy=dy_sub_v,
                            )

                        vg_s, dg_s, mg_s = jax.vmap(_one_gp)(g_pts_arr)
                        return jnp.mean(vg_s, axis=0), jnp.mean(dg_s, axis=0), jnp.mean(mg_s, axis=0)

                    return jax.lax.cond(
                        jnp.any(pres_vec),
                        _do_zsub,
                        lambda: (bg_vis_sub, bg_dir_sub, bg_mom_sub),
                    )

                vis_subs, dir_subs, mom_subs = jax.vmap(_one_subinterval)((bp[:-1], bp[1:], length, present))
                visible = jnp.moveaxis(vis_subs, 0, -2)  # (sx, sy, S, K + 1)
                lat_dir = jnp.moveaxis(dir_subs, 0, -2)
                moments = jnp.moveaxis(mom_subs, 0, -3)
                dz_s = zb - za
                frac_sub = jnp.sum(length[:, None] * visible, axis=-2) / dz_s
                d_lat = jnp.sum(length[:, None] * lat_dir, axis=-2)
                d_vert = _vertical_tensor(length, jnp.broadcast_to(bp, (sx, sy, num_s + 1)), moments, sl_z, graded=True) * ar_f
                return frac_sub, d_lat + d_vert

            frac_z, dir_z = jax.vmap(_one_z_sub)(jnp.arange(sz))
            # (sz, sx, sy, ...) -> (sx, sy, sz, ...)
            return jnp.moveaxis(frac_z, 0, 2), jnp.moveaxis(dir_z, 0, 2)

        raw_fracs, raw_dir6 = jax.vmap(_one_voxel)(jnp.arange(b_size))
        act_mask = chunk_mask[:, None, None, None, None]
        sub_lvl_fracs = jnp.where(act_mask, raw_fracs, bg_frac_vox)
        sub_dir6 = jnp.where(act_mask, raw_dir6, zero_dir_vox)
        # Match sub-cell level fractions so their mean strictly equals the voxel's target fractions
        sub_lvl_flat = sub_lvl_fracs.reshape(b_size, sx * sy * sz, num_k + 1)
        matched = _match_fractions(sub_lvl_flat, chunk_target, mat_indices).reshape(
            b_size, sx, sy, sz, num_k + 1
        )
        sub_mat_fracs = jnp.einsum("...l,lm->...m", matched, onehot, precision=_HIGHEST)
        keep = np.asarray([n > 1 for n in device.grid_shape], dtype=float)
        keep33 = jnp.asarray(np.outer(keep, keep), dtype=dtype)
        sub_dir33 = from_six(sub_dir6) * keep33
        return sub_mat_fracs, sub_dir33, sides

    return {
        "junction_cell_fn": junction_cell_fn,
        "junction_xs": junction_xs,
        "make_junction_fn": lambda _w: junction_cell_fn,
    }
