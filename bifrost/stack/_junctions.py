"""Periodic-cell junction homogenization and safe SPD tensor inversion."""

import math
import jax
import jax.numpy as jnp
import numpy as np
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple
from ..subpixel import from_six, laminate, periodic_cell_tensor, principal_direction, symmetric_inverse
from ._layers import Array, _HIGHEST, _KOTTKE_TINY, _NO_FACES, _SUBCELL_GAP, _SampleAxis
from ._corners import (
    _Cells,
    _CornerInputs,
    _cell_area,
    _cell_corners,
    _cells_expand,
    _corner_interfaces,
    _dyad,
    _interface_parts,
    _mean_shift,
    _paint_levels,
)
from ._profiles import _gauss_heights, _split_layer


def _is_finite_spd(a: Array) -> Array:
    """Certifies that (..., 3, 3) symmetric matrices are finite and strictly positive definite."""
    finite = jnp.all(jnp.isfinite(a), axis=(-2, -1))
    safe = jnp.where(finite[..., None, None], a, jnp.eye(3, dtype=a.dtype))
    sym = 0.5 * (safe + jnp.swapaxes(safe, -1, -2))
    a00 = sym[..., 0, 0]
    a01, a02 = sym[..., 0, 1], sym[..., 0, 2]
    a11, a12, a22 = sym[..., 1, 1], sym[..., 1, 2], sym[..., 2, 2]
    c00 = a11 * a22 - a12 * a12
    c01 = a02 * a12 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    m2 = a00 * a11 - a01 * a01
    det = a00 * c00 + a01 * c01 + a02 * c02
    # Keep det^2 above underflow in quotient-rule reverse-mode derivatives.
    min_det = 2.0 * math.sqrt(float(jnp.finfo(a.dtype).tiny))
    return finite & (a00 > 0.0) & (m2 > 0.0) & (det > min_det)


def _safe_symmetric_inverse(a: Array) -> tuple[Array, Array]:
    """Symmetric 3x3 inverse with finite-SPD certification and double-where gradient guard."""
    ok = _is_finite_spd(a)
    eye = jnp.eye(3, dtype=a.dtype)
    a_safe = jnp.where(ok[..., None, None], a, eye)
    sym = 0.5 * (a_safe + jnp.swapaxes(a_safe, -1, -2))
    inv_safe = symmetric_inverse(sym)
    return jnp.where(ok[..., None, None], inv_safe, eye), ok


def _junction_cells(
    offsets: Array,
    normals: Array,
    centers: Array,
    is_fill: Sequence[bool],
    lows: Array,
    highs: Array,
    layer: Array,
    lateral: Callable[[Array], Array],
    sub: float,
    knots: Array | None,
    cell: float,
    vectors: np.ndarray,
    contrast2: np.ndarray,
    level_eps: np.ndarray,
    cells: tuple[int, int, int],
    point_slopes: Array | None = None,
    corner: "_CornerInputs | None" = None,
    planes: tuple[Array, Array] | None = None,
    target: Array | None = None,
    level_material: Sequence[int] = (),
    top: Array | None = None,
    half: Any = 0.5,
) -> tuple[Array, Array]:
    """The sub-cells of junction voxels for their closure: the voxel's own geometry, cell by cell.

    The voxel is cut into ``cells`` sub-cells (lateral sub-squares times equal parts of its height). Each
    sub-square is painted with the half-planes of the voxel's sample square it lies in
    (:func:`_interface_parts`), or, in a corner column, with the voxel's corner model
    (:func:`_corner_interfaces`: its cells, tips, seams and certificate), the same model as the voxel's
    fractions, so the sub-cells' volumes add up to the voxel's; and every sub-cell gets the laminate of
    its materials
    (:func:`bifrost.subpixel.laminate`) along the leading direction of its own faces (its z
    sub-intervals' lateral faces and the variance of its material moments along z), which is exact where a
    sub-cell holds one interface or parallel ones. The voxel's tensor is then the energy form of the
    periodic cell problem on these sub-cells (:func:`bifrost.subpixel.periodic_cell_tensor`).

    Args:
        offsets: (N, Q, K) for every voxel and sub-square (Q = n n, x-major), the signed distances of the
            levels in the frame of the sample square holding it (sides of that square).
        normals: (N, Q, K, 2) the unit normals.
        centers: (Q, 2) every sub-square's center in its sample square's frame (sides of that square).
        is_fill: (K,) static.
        lows: (N, K) bottom surfaces of the levels at the voxels' columns (cells).
        highs: (N, K) top surfaces (cells).
        layer: (N,) the voxels' layers.
        lateral: see :func:`_composite` (sides of the sample square).
        sub: half side of the sub-squares (sides of the sample square).
        knots: see :func:`_composite`.
        cell: cells per side of the sample square.
        vectors: (K + 1, 6) static normalized material tensors of the levels and the background.
        contrast2: (K + 1, K + 1) static squared contrasts.
        level_eps: (K + 1, 3, 3) static permittivity of every level's material, then the background's.
        cells: (n, n, m) static sub-cells along x, y, z.
        point_slopes: (N, 2K + 2, 2) or None, see :func:`_point_slopes`.
        corner: the corner model of the voxels' columns (no indices; ``taken`` False where a voxel's
            column is not a corner column, which keeps its half-planes), or None. One sample square.
        planes: with ``corner``, (N, K) offsets and (N, K, 2) normals of the voxels' half-planes at their
            centers (sides of the square).
        target: (N, M) the voxels' material fractions, or None. Given, the sub-cells' fractions are made to
            add up to them exactly (:func:`_match_fractions`): with several sample squares per voxel the
            sub-squares are painted with their samples' half-planes, while the voxel's fractions blend the
            corner model in.
        level_material: (K + 1,) with ``target``, the material of every level, then the background's.
        top: (N,) the voxels' layer tops on a graded grid (``layer`` is then their bottoms, frame units),
            or None: layer k is [k, k + 1]. The sub-cells are then equal parts of the layer's own height,
            and their lateral faces as high as they are.
        half: the voxel's lateral cell (sides of a sample square; ``(hx, hy)`` where x and y differ), whose
            corner model and half-planes the sub-cells blend like the voxel. On graded x or y every voxel's
            own (:class:`_Cells`, (N,)); ``sub`` is then the voxels' sub-cells, ``centers`` (N, Q, 2) and
            ``cell`` unused (the cells' own).

    Returns:
        fractions: (N, nx, ny, nz, K + 1) every sub-cell's fractions of the levels, then the background.
        coefficients: (N, nx, ny, nz, 3, 3) every sub-cell's laminate.
    """
    nxs, nys, nzs = cells
    dtype = offsets.dtype
    if nxs < 1 or nys < 1 or nzs < 1:
        raise ValueError(f"junction_cells must be positive, got {cells}")
    per_column = isinstance(half, _Cells)
    if per_column:
        cell = half.cell
    if top is None:
        split = jax.vmap(_split_layer)(lows, highs, layer)
    else:
        split = jax.vmap(_split_layer)(lows, highs, layer, top)
    count = nxs * nys

    def per_square(x: Array) -> Array:
        """(N, ...) -> (N, Q, ...)."""
        return jnp.broadcast_to(x[:, None], (x.shape[0], count, *x.shape[1:]))

    visible, tensor, moments, slopes, _ = _interface_parts(
        offsets,
        normals,
        is_fill,
        tuple(per_square(x) for x in split),
        lateral,
        _cells_expand(sub, 1) if per_column else sub,  # per column over the sub-squares
        knots,
        cell,
        vectors,
        contrast2,
        None if point_slopes is None else per_square(point_slopes),
        center=jnp.broadcast_to(centers, (offsets.shape[0], count, 2)),
    )  # (N, Q, S, ...)
    perm, bp, length, present = split
    if corner is not None:
        # the voxel's own half-planes (its certificate and seams), the sub-squares' at the Gauss heights,
        # then the corner model on the sub-squares, blended like the voxel
        num_levels = len(is_fill)
        t_visible, t_tensor, t_moments, _, b = _interface_parts(
            planes[0], planes[1], is_fill, split, lateral, half, knots, cell, vectors, contrast2, point_slopes
        )
        gauss = _gauss_heights(split, lateral, num_levels)  # (N, S, K, 2)
        if per_column:  # frame units -> sides of each voxel's cell
            gauss = gauss / cell[:, None, None, None]
        num_v = 4 + sum(not f for f in is_fill)
        square = _cell_corners(sub, dtype)
        if per_column:  # (N, 1, 1, V, 2): over the sub-intervals and sub-squares
            square = jnp.concatenate([square, jnp.repeat(square[..., -1:, :], num_v - 4, axis=-2)], axis=-2)
            square = square[:, None, None]
        else:
            square = jnp.concatenate([square, jnp.repeat(square[-1:], num_v - 4, axis=0)])
        shape = (*length.shape, count)  # (N, S, Q)
        q_normals = jnp.broadcast_to(planes[1][:, None, None], (*shape, num_levels, 2))
        def q_centers() -> Array:
            """The sub-squares' centers over the voxels (per column), sub-intervals and levels (traced where used,
            so uniform grids trace as before)."""
            return centers[:, None, :, None, :] if per_column else centers[None, None, :, None, :]

        tq_gauss = 0.0
        for g in (0, 1):
            o_g = planes[0][:, None, None, :] + gauss[:, :, None, :, g] + jnp.sum(
                planes[1][:, None, None] * q_centers(), axis=-1
            )
            tq_gauss = tq_gauss + 0.5 * _paint_levels(
                jnp.broadcast_to(square, (*shape, num_v, 2)), jnp.full(shape, 4, dtype=jnp.int32),
                jnp.ones(shape, dtype=dtype), q_normals, o_g, jnp.broadcast_to(present[:, :, None], (*shape, num_levels)),
                is_fill, _cells_expand(sub, 2) if per_column else sub,
            )[0]
        b = jnp.broadcast_to(b[:, None, :], (*length.shape, num_levels))
        out = _corner_interfaces(
            corner, t_visible, t_tensor, t_moments, length, present, gauss, b, planes, is_fill, half, cell,
            vectors, contrast2,
            squares=(
                centers, sub, jnp.swapaxes(visible, 1, 2), jnp.swapaxes(tensor, 1, 2), jnp.swapaxes(moments, 1, 2),
                tq_gauss,
            ),
        )
        visible, tensor, moments = (jnp.swapaxes(x, 1, 2) for x in out[4:])  # (N, Q, S, ...)
    # the sub-cells along z: equal parts of the voxel layer, overlapping the sub-intervals
    if top is None:
        z0 = layer[:, None] + jnp.arange(nzs, dtype=dtype)[None, :] / nzs  # (N, m)
        lo = jnp.maximum(bp[:, :-1, None], z0[:, None, :])
        hi = jnp.minimum(bp[:, 1:, None], z0[:, None, :] + 1.0 / nzs)
        overlap = jnp.maximum(hi - lo, 0.0) * nzs  # (N, S, m): share of the sub-cell's height
    else:
        part = (top - layer)[:, None] / nzs  # (N, 1) the sub-cells' height
        z0 = layer[:, None] + part * jnp.arange(nzs, dtype=dtype)[None, :]
        z1 = z0 + part
        lo = jnp.maximum(bp[:, :-1, None], z0[:, None, :])
        hi = jnp.minimum(bp[:, 1:, None], z1[:, None, :])
        overlap = jnp.maximum(hi - lo, 0.0) / part[:, :, None]
    fractions = jnp.einsum("nsm,nqsl->nqml", overlap, visible, precision=_HIGHEST)  # (N, Q, m, K + 1)
    if target is not None:
        fractions = _match_fractions(fractions.reshape(fractions.shape[0], -1, fractions.shape[-1]), target,
                                     level_material).reshape(fractions.shape)
    if top is None:
        lateral_d = jnp.einsum("nsm,nqsv->nqmv", overlap, tensor, precision=_HIGHEST) / nzs
    else:  # the faces per unit height times the sub-cell's height (frame units)
        lateral_d = jnp.einsum("nsm,nqsv->nqmv", overlap, tensor, precision=_HIGHEST) * part[:, :, None, None]
    # each sub-cell's own vertical variance: its sub-intervals' moments weighted by their share of it
    mean = jnp.einsum("nsm,nqsvc->nqmvc", overlap, moments, precision=_HIGHEST)
    spread = jnp.einsum(
        "nsm,nqsm->nqm", overlap,
        jnp.sum((moments[:, :, :, None] - mean[:, :, None]) ** 2, axis=(-2, -1)),
        precision=_HIGHEST,
    )  # (N, Q, m): sum_s share_s |m_s - <m>|^2
    # times the sub-cell's area (cells^2), like the faces
    if per_column:
        vertical = 4.0 * spread * (_cell_area(sub) * cell**2)[:, None, None]
    else:
        vertical = 4.0 * spread * ((2 * sub * cell) ** 2 if not isinstance(sub, tuple) else _cell_area(sub) * cell**2)
    if point_slopes is not None:  # tilted by the surfaces in the sub-cell (their jump-weighted mix)
        ends = bp[:, 1:-1]  # (N, S - 1)
        if top is None:
            inside = (ends[:, :, None] > z0[:, None, :]) & (ends[:, :, None] < z0[:, None, :] + 1.0 / nzs)
        else:
            inside = (ends[:, :, None] > z0[:, None, :]) & (ends[:, :, None] < z1[:, None, :])
        jump = jnp.sum((moments[:, :, 1:] - moments[:, :, :-1]) ** 2, axis=(-2, -1))  # (N, n n, S - 1)
        wj = jump[:, :, :, None] * inside[:, None]  # (N, n n, S - 1, m)
        normal = jnp.concatenate([-slopes, jnp.ones_like(slopes[..., :1])], axis=-1)
        normal = normal / jnp.sqrt(jnp.sum(normal**2, axis=-1, keepdims=True))  # (N, n n, S - 1, 3)
        total = jnp.sum(wj, axis=2)
        some = total > _KOTTKE_TINY
        mix = jnp.einsum("nqsm,nqsv->nqmv", wj, _dyad(normal), precision=_HIGHEST) / jnp.where(some, total, 1.0)[..., None]
        up = jnp.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=dtype)
        vertical_d = vertical[..., None] * jnp.where(some[..., None], mix, up)
    else:
        vertical_d = vertical[..., None] * jnp.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=dtype)
    direction = from_six(lateral_d + vertical_d)  # (N, nx ny, m, 3, 3)
    normal, lam1 = principal_direction(direction)
    coefficients = laminate(level_eps, fractions, normal)  # (N, nx ny, m, 3, 3)
    # Where a sub-cell's two largest face directions weigh nearly the same (a corner in it with two walls of like
    # weight), its leading direction is not defined: it would switch between the walls' normals at round-off as
    # the corner moves, and the laminate with it. There the laminate blends, by a smooth step in the gap between
    # the two largest eigenvalues, toward one that needs no direction: the laminates along x, y and z weighted by
    # the direction tensor's diagonal (continuous in it). Elsewhere it is the leading direction's, exactly.
    # Compute lam1 - lam2 from the 2x2 tangent-plane invariants of (I - n n^T) D (I - n n^T) rather than eigvalsh
    # so repeated eigenvalues (e.g. isotropic D or degenerate corners) keep finite derivatives under jax.grad.
    trace = jnp.trace(direction, axis1=-2, axis2=-1)
    some = trace > _NO_FACES
    safe_trace = jnp.where(some, trace, 1.0)
    proj = jnp.eye(3, dtype=dtype) - normal[..., :, None] * normal[..., None, :]
    orth = jnp.matmul(jnp.matmul(proj, direction, precision=_HIGHEST), proj, precision=_HIGHEST)
    orth = 0.5 * (orth + jnp.swapaxes(orth, -1, -2))
    tr2 = jnp.maximum(jnp.trace(orth, axis1=-2, axis2=-1), 0.0)
    s2 = jnp.sum(orth * orth, axis=(-2, -1))
    disc = 2.0 * s2 - tr2 * tr2
    disc_floor = jnp.maximum(
        jnp.asarray(1e-12, dtype=dtype) * safe_trace * safe_trace,
        jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype) ** (2.0 / 3.0),
    )
    disc_ok = disc > disc_floor
    half_diff = 0.5 * jnp.where(disc_ok, jnp.sqrt(jnp.where(disc_ok, disc, 1.0)), 0.0)
    lam2 = 0.5 * tr2 + half_diff
    gap = jnp.where(some, jnp.maximum(lam1 - lam2, 0.0) / safe_trace, 1.0)
    x = jnp.clip((gap - _SUBCELL_GAP[0]) / (_SUBCELL_GAP[1] - _SUBCELL_GAP[0]), 0.0, 1.0)
    keep = x * x * (3.0 - 2.0 * x)  # 1: the leading direction's laminate
    share = jnp.diagonal(direction, axis1=-2, axis2=-1) / safe_trace[..., None]
    axes = sum(
        share[..., a, None, None]
        * laminate(level_eps, fractions, jnp.broadcast_to(jnp.asarray(np.eye(3)[a], dtype=dtype), normal.shape))
        for a in range(3)
    )
    coefficients = jnp.where((keep < 1.0)[..., None, None], axes + keep[..., None, None] * (coefficients - axes),
                             coefficients)
    shape = (offsets.shape[0], nxs, nys, nzs)
    return fractions.reshape(*shape, -1), coefficients.reshape(*shape, 3, 3)


def _match_fractions(fractions: Array, target: Array, level_material: Sequence[int]) -> Array:
    """Sub-cell fractions of the levels made to average to the voxel's material fractions exactly.

    Each material's fraction of the voxel is split among its levels by their mean over the sub-cells (all to
    its first level where they hold none of it); then, level by level in priority order, every sub-cell's
    union of the levels up to it gets one shift, clipped to [its union below, 1] (:func:`_mean_shift`), so
    that the unions' mean is the target's: the sub-cells stay fractions, keep their arrangement, and hold the
    voxel's materials. Where they already add up to the voxel (one sample square per voxel) the shifts vanish.

    Args:
        fractions: (N, C, K + 1) the C sub-cells' fractions of the levels, then the background.
        target: (N, M) the voxels' material fractions.
        level_material: (K + 1,) static material of every level, then the background's.

    Returns:
        (N, C, K + 1) the matched fractions.
    """
    num = fractions.shape[-1]
    onehot = np.zeros((num, target.shape[-1]))
    onehot[np.arange(num), np.asarray(level_material)] = 1.0
    first = np.zeros(num)
    for m in set(level_material):
        first[list(level_material).index(m)] = 1.0
    onehot_j = jnp.asarray(onehot, dtype=fractions.dtype)
    mean = jnp.mean(fractions, axis=1)  # (N, K + 1)
    # full float32 sums (a default-precision GPU matmul rounds the fractions to TF32: 1e-3)
    per_material = jnp.einsum("nl,lm->nm", mean, onehot_j, precision=_HIGHEST)
    of_material = jnp.einsum("nm,lm->nl", per_material, onehot_j, precision=_HIGHEST)  # each level's material's
    share = jnp.where(of_material > 0, mean / jnp.where(of_material > 0, of_material, 1.0), jnp.asarray(first, mean.dtype))
    goal = jnp.einsum("nm,lm->nl", target, onehot_j, precision=_HIGHEST) * share  # (N, K + 1) the levels' targets
    out = []
    below = jnp.zeros_like(fractions[..., 0])
    union, union_goal = 0.0, 0.0
    for j in range(num - 1):
        union = union + fractions[..., j]
        union_goal = union_goal + goal[:, j : j + 1]
        c = union + _mean_shift(union, below, union_goal)
        c = jnp.where(c > 1.0, 1.0, c)
        c = jnp.where(c < below, below, c)
        out.append(c - below)
        below = c
    out.append(1.0 - below)
    return jnp.stack(out, axis=-1)


def _junction_tensor(*args: Any, **kwargs: Any) -> tuple[Array, Array]:
    """The closure of junction voxels: the periodic cell problem on the sub-cells of :func:`_junction_cells`.

    Returns:
        tensor: (N, 3, 3) the effective permittivity
            (:func:`bifrost.subpixel.periodic_cell_tensor`).
        residual: (N,) the solves' normwise backward errors.
    """
    coefficients = _junction_cells(*args, **kwargs)[1]
    if kwargs.get("top") is None:
        return periodic_cell_tensor(coefficients)
    height = kwargs["top"] - (args[6] if len(args) > 6 else kwargs["layer"])  # a graded layer: a box
    ones = jnp.ones_like(height)
    return periodic_cell_tensor(coefficients, jnp.stack([ones, ones, height], axis=-1))


class _LayerGeometry(NamedTuple):
    """A source layer's polygons on the grid, after T-junction splitting."""

    #: (E, 2) vertices in cells (traced)
    vertices: Array
    #: (E,) end vertex of every edge
    nxt: np.ndarray
    #: (E,) polygon index of every edge
    ids: np.ndarray
    #: (E,) the edges that bound the union of the polygons
    boundary: np.ndarray
    #: (E,) start vertex of every edge with the filled side on its left (None
    #: without corner_lines)
    src: np.ndarray | None
    #: (E,) end vertex of every edge with the filled side on its left
    dst: np.ndarray | None
    #: (E,) the next boundary edge, -1 if none or ambiguous
    succ: np.ndarray | None


class _JunctionInputs(NamedTuple):
    """What the junction closure needs to paint any voxel of a sample grid again (see :func:`_junction_tensor`)."""

    #: (P, nx, ny, K) the levels' signed distances of every sample square (sides of a sample square)
    offsets: Array
    #: (P, nx, ny, K, 2) their unit normals
    normals: Array
    #: (nx, ny, K) bottom surfaces (cells)
    lows: Array
    #: (nx, ny, K) top surfaces (cells)
    highs: Array
    #: (nx, ny, 2K + 2, 2) or None, see :func:`_point_slopes`
    point_slopes: Array | None
    #: the levels' sidewall offsets at heights (see :func:`_composite`)
    lateral: Callable[[Array], Array]
    #: (K, M) z_to_bias knots, or None
    knots: Array | None
    #: (K,) whether a level fills the whole plane
    is_fill: tuple[bool, ...]
    #: (K + 1, 3, 3) the levels' materials' permittivity, then the background's
    level_eps: np.ndarray
    #: (K + 1, 6) the same over the largest Frobenius norm, as 6-vectors of the same norm
    vectors: np.ndarray
    #: (K + 1, K + 1) squared contrasts between them
    contrast2: np.ndarray
    #: cells per side of a sample square
    cell: float
    #: the corner columns of the grid (one sample square per voxel only), or None
    corner: "_CornerInputs | None" = None
    #: (K + 1,) the material (index in the grid's material names) of every level, then the background's
    level_material: tuple[int, ...] = ()
    #: the grid's z axis when it is graded (its layers' bounds), or None: layer k is [k, k + 1]
    zaxis: _SampleAxis | None = None
    #: the lateral cell a sample is painted on: 0.5 (a square), or (hx, hy) (see StackDevice._lateral_cell)
    half: Any = 0.5
    #: (wx, wy) the voxels' lateral widths (frame units); on graded x or y per column, (nx,) and (ny,)
    widths: tuple[Any, Any] = (1.0, 1.0)


class _Interfaces(NamedTuple):
    """The voxels of one sample grid with their interfaces (see :meth:`StackDevice._interfaces`)."""

    #: material names in fdtdx order
    names: list[str]
    #: (M, nx, ny, nz) material fractions
    weights: Array
    #: (nx, ny, nz, 6) interface direction tensor (xx, yy, zz, xy, xz, yz)
    direction: Array
    #: the junction closure's inputs, or None (no level)
    voxel: _JunctionInputs | None
    #: () corner candidate columns that found no slot (summed over the sample squares)
    corner_dropped: Array | int = 0
    #: () voxels of corner columns whose corner model is not certified, so that their fractions and faces are
    #: partly the half-plane model's (one plane per level; summed over the sample squares)
    corner_uncertified: Array | int = 0
