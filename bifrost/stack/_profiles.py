"""Vertical z-splitting, top roughness, composite voxelization, and interface profiles."""

import math
import jax
import jax.numpy as jnp
import numpy as np
from collections.abc import Callable, Sequence
from fdtdx.core.jax.pytrees import TreeClass, autoinit, field, frozen_field
from ..timing import timed
from ..geometry import gaussian_random_field, has_tracers
from ._layers import Array, _BIG, _OFFDIAG_PAIRS, _SampleAxis, _primal
from ._corners import (
    _Cells,
    _CornerInputs,
    _cell_area,
    _cell_corners,
    _cells_expand,
    _cells_take,
    _corner_pass,
    _interface_corner_pass,
    _interface_parts,
    _kernel_mass,
    _level_walls,
    _normal_tensor,
    _paint_levels,
    _vertical_tensor,
    _weigh_levels,
)


def _point_slopes(lows: Array, surface_slopes: Array | None) -> Array | None:
    """(nx, ny, 2K + 2, 2) surface slopes per point of :func:`_split_layer`: none at the layer's ends."""
    if surface_slopes is None:
        return None
    ends = jnp.zeros((*lows.shape[:-1], 2, 2), dtype=lows.dtype)
    return jnp.concatenate([ends, surface_slopes.astype(lows.dtype)], axis=-2)


def _split_layer(
    lows: Array, highs: Array, k: Array, top: Array | None = None
) -> tuple[Array, Array, Array, Array]:
    """The z sub-intervals of voxel layer k and the levels present on each.

    Args:
        lows: (..., K) bottom surface per level (cells).
        highs: (..., K) top surface per level (cells).
        k: the voxel layer's bottom; it spans [k, k + 1] (a uniform grid's layer k).
        top: the layer's top instead of k + 1 (a graded grid's layer spans [k, top],
            in frame units, see :class:`_Frame`).

    Returns:
        perm: (..., 2K + 2) the sort of the points (the layer's ends, the
            bottoms, the tops), no gradient.
        bp: (..., S + 1) sorted sub-interval ends, S = 2K + 1 (cells).
        length: (..., S) sub-interval lengths, summing to the layer's height.
        present: (..., S, K) whether each level is in each sub-interval.
    """
    num_levels = lows.shape[-1]
    # k + 1 is recomputed at every use, so a uniform grid's trace stays op for op what it was
    up = (lambda: k + 1.0) if top is None else (lambda: top)
    lo = jnp.clip(lows, k, up())
    hi = jnp.clip(highs, k, up())
    ends = jnp.broadcast_to(jnp.asarray([k, up()], dtype=lows.dtype), (*lows.shape[:-1], 2))
    points = jnp.concatenate([ends, lo, hi], axis=-1)  # (..., 2K+2)
    # membership of sub-interval m (between sorted points m and m+1) comes from
    # the ranks of each level's surfaces in a stable sort, not from midpoints:
    # with tied surfaces every level keeps the sub-interval on its own side,
    # so moving a surface always changes that level's coverage. The layer's
    # ends sort outside any surface tied with them (bottom first, top last), so
    # the lengths always add up to the constant layer height: a surface clipped
    # onto the top end (its derivative half there) moves the sub-interval
    # above it, not the layer's end
    tie = jnp.asarray([0, 2] + [1] * (2 * num_levels), dtype=jnp.int32)
    perm = jnp.lexsort((jnp.broadcast_to(tie, points.shape), jax.lax.stop_gradient(points)), axis=-1)
    bp = jnp.take_along_axis(points, perm, axis=-1)
    rank = jnp.argsort(perm, axis=-1, stable=True)
    r_lo = rank[..., 2 : 2 + num_levels]
    r_hi = rank[..., 2 + num_levels :]
    length = bp[..., 1:] - bp[..., :-1]  # (..., S)
    m = jnp.arange(length.shape[-1])
    inside = (m >= r_lo[..., None]) & (m < r_hi[..., None])  # (..., K, S)
    # a level that ends strictly before this voxel layer (or starts strictly
    # after it) is clamped to a zero-length interval at one of its ends; it
    # is not there, and must not take the sub-intervals next to a surface
    # that moves onto that end (it would claim that surface's growth)
    reach = (jax.lax.stop_gradient(highs) >= k) & (jax.lax.stop_gradient(lows) <= up())
    inside = inside & reach[..., None]
    return perm, bp, length, jnp.moveaxis(inside, -2, -1)


def _paint_layer(
    offsets: Array,
    normals: Array,
    is_fill: Sequence[bool],
    split: tuple[Array, Array, Array, Array],
    lateral: Callable[[Array], Array],
    half: float,
    level_eps: Sequence[Sequence[float]] | None = None,
    kernel: tuple[float, float, float] = (0.0, 0.0, 1.0),
    cell: float = 1.0,
    point_slopes: Array | None = None,
    knots: Array | None = None,
) -> tuple[Array, ...]:
    """Half-plane painting of columns of one voxel layer, per z sub-interval (see :func:`_composite`).

    Args:
        offsets: (B1, B2, K) signed distance of each level (sides of the square).
        normals: (B1, B2, K, 2) unit in-plane normals.
        is_fill: (K,) static, whether a level fills the whole plane.
        split: the layer's sub-intervals (:func:`_split_layer`).
        lateral: the levels' sidewall offsets at heights (see :func:`_composite`).
        half: half side of the lateral square.
        level_eps: (C, K) static contrasts, or None: fractions only.
        kernel: the flux kernel (see :func:`_composite`).
        cell: cells per side of the lateral square.
        point_slopes: (B1, B2, 2K + 2, 2) :func:`_point_slopes`, or None (flat).
        knots: (K, M) heights where the levels' offsets bend (see
            :func:`_level_walls`), or None.

    Returns:
        visible (B1, B2, S, K + 1) fractions per sub-interval (not length
        weighted), length (B1, B2, S), present (B1, B2, S, K), shift (B1, B2, S,
        K) the levels' sidewall offsets; with ``level_eps`` also their rates
        (B1, B2, S, K, 2): the z slope b (cells per cell, 0 for fill levels)
        and ``b * lag``, with lag the height of the sub-interval's middle above
        the middle of the level's part (where its wall is painted); the
        contrast flux (B1, B2, S, C, 3) and rates (B1, B2, S, C, 2; see
        :func:`_normal_tensor`), the sub-interval ends (B1, B2, S + 1) and,
        with ``point_slopes``, the inner ends' surface slopes (B1, B2, S - 1,
        2).
    """
    perm, bp, length, present = split
    num_levels = len(is_fill)
    num_vertices = 4 + sum(not f for f in is_fill)
    per_column = isinstance(half, _Cells)  # graded x or y: each column's own cell, lateral() in frame units
    if per_column:
        half = _cells_expand(half, 1)  # (B1, B2, 1): over the sub-intervals
        square = _cell_corners(half, bp.dtype)  # (B1, B2, 1, 4, 2)
        square = jnp.concatenate([square, jnp.repeat(square[..., -1:, :], num_vertices - 4, axis=-2)], axis=-2)
    else:
        square = _cell_corners(half, bp.dtype)
        square = jnp.concatenate([square, jnp.repeat(square[-1:], num_vertices - 4, axis=0)])
    with_n = level_eps is not None
    # each level's sidewall over its own part of the layer (the same in every
    # sub-interval), which follows its (traced) surfaces; with level_eps also
    # its z slope (cells per cell; 0 for fill levels)
    batch = length.shape
    shift, slope, middle = _level_walls(split, lateral, num_levels, knots, with_n)
    if per_column:  # frame units -> sides of each column's cell
        shift = shift / half.cell[..., 0, None]
    shift = jnp.broadcast_to(shift[..., None, :], (*batch, num_levels))  # (B1, B2, S, K)
    if with_n:
        # and how far the sub-interval's middle is from where the wall is
        # painted: the normal's drift is that of walls at the middles
        lag = 0.5 * (bp[..., 1:] + bp[..., :-1])[..., None] - middle[..., None, :]  # (B1, B2, S, K)
        # per column the slope is already frame units per frame unit
        slope = slope[..., None, :] if per_column else cell * slope[..., None, :]
        slope = jnp.stack([jnp.broadcast_to(slope, (*batch, num_levels)), slope * lag], axis=-1)

    painted = _paint_levels(
        jnp.broadcast_to(square, (*batch, num_vertices, 2)),
        jnp.full(batch, 4, dtype=jnp.int32),
        jnp.ones(batch, dtype=bp.dtype),
        jnp.broadcast_to(normals[:, :, None], (*batch, num_levels, 2)),
        offsets[:, :, None, :] + shift,
        present,
        is_fill,
        half,
        # the square's sides are not interfaces
        *((jnp.zeros((*batch, num_vertices, 4), dtype=bp.dtype), slope, kernel) if with_n else ()),
    )
    parts = (painted[0], length, present, shift)
    if not with_n:
        return parts
    parts += (
        slope,
        _weigh_levels(painted[2], level_eps, axis=-2),
        _weigh_levels(painted[3], level_eps, axis=-2),
        bp,
    )
    if point_slopes is not None:
        parts += (jnp.take_along_axis(point_slopes, perm[..., 1:-1, None], axis=-2),)
    return parts


def _gauss_heights(split: tuple[Array, ...], lateral: Callable[[Array], Array], num_levels: int) -> Array:
    """Each level's sidewall offset at the two Gauss-Legendre heights of its own part of a voxel layer.

    The corner model paints there (:func:`_corner_fill`): near a corner of a
    sloped wall the area is quadratic in z, which the 2-point rule integrates
    exactly over the level's part (where no z_to_bias knot is inside it). As
    for :func:`_level_walls`, only the level's own surfaces enter: another
    level's surface that cuts the part changes nothing.

    Args:
        split: (B1, B2, ...) sub-intervals of :func:`_split_layer`.
        lateral: the levels' sidewall offsets at heights (see :func:`_composite`).
        num_levels: K.

    Returns:
        (B1, B2, S, K, 2) the offsets at the lower, then the upper height (the
        same in every sub-interval).
    """
    perm, bp = split[:2]
    points = jnp.take_along_axis(bp, jnp.argsort(perm, axis=-1), axis=-1)  # the layer's ends, lows, highs
    lo, hi = points[..., 2 : 2 + num_levels], points[..., 2 + num_levels :]
    z = 0.5 * (lo + hi)[..., None] + jnp.asarray([-0.5, 0.5], dtype=bp.dtype) / math.sqrt(3.0) * (hi - lo)[..., None]
    every = lateral(z.reshape(*z.shape[:-2], -1)).reshape(*z.shape, num_levels)  # (B1, B2, K, 2, K)
    own = jnp.moveaxis(jnp.diagonal(every, axis1=-3, axis2=-1), -1, -2)  # (B1, B2, K, 2)
    return jnp.broadcast_to(own[..., None, :, :], (*bp.shape[:-1], bp.shape[-1] - 1, num_levels, 2))


@timed("composite")
def _composite(
    offsets: Array,
    normals: Array,
    is_fill: Sequence[bool],
    lows: Array,
    highs: Array,
    lateral: Callable[[Array], Array],
    nz: int,
    half: float = 0.5,
    corner: "_CornerInputs | None" = None,
    level_eps: Sequence[Sequence[float]] | None = None,
    kernel: tuple[float, float, float] = (0.0, 0.0, 1.0),
    cell: float = 1.0,
    surface_slopes: Array | None = None,
    knots: Array | None = None,
    offdiag: bool = False,
    zaxis: _SampleAxis | None = None,
) -> Array | tuple[Array, Array]:
    """Volume fraction of each level (and the background) per voxel.

    Along z the voxel is split at the heights of all level surfaces (surfaces
    are clamped to the voxel, so a surface on a voxel boundary contributes half
    its derivative to each neighbor). On each sub-interval, the levels present
    there are painted in priority order. Laterally each level is a half-plane
    ``{p: n . p + offset + s_j < 0}`` in voxel coordinates, from its signed
    distance and gradient at the voxel center, moved by the mean s_j of its
    sidewall ``lateral`` over its own part of the voxel along z
    (:func:`_level_walls`; exact between the ``knots``, and independent of the
    other levels' surfaces). So a surface cutting the voxel gets the boundary
    at its own height, and a straight edge gets its exact volume summed along
    the edge, also with z_to_bias (per voxel, the area at s_j differs from its
    mean over z where the edge crosses a corner of the square within the
    voxel, by up to about 5e-3 of the voxel for axis-aligned edges and 2e-2
    for edges at 17 deg to the grid, at a 20 deg wall on cubic voxels (as
    measured), but these errors cancel along the edge; corner columns use two
    heights, see below): level j gets
    the area of the voxel square
    inside it but outside every earlier present level, computed exactly by
    clipping the square with the complements in priority order. Levels that
    meet at a seam (complementary half-planes) therefore tile the voxel, nested
    levels nest, and slanted straight edges get exact areas. Fill levels cover
    the whole square. With ``corner``, the columns where a polygon corner or a
    crossing of layer edges is within reach are repainted with the exact local
    geometry (see :func:`_corner_fill`), in the voxel layers where a polygon
    level is present in one of them, with each level at the two Gauss heights
    of its own part of the voxel (:func:`_gauss_heights`), which integrate a
    sloped corner's quadratic area exactly, and with the half-planes' result
    where the edge is straight in the square (so a straight edge has no seam
    between corner and ordinary columns, see :func:`_corner_fill`).

    With ``level_eps``, the clips also carry each level's plane: its in-plane
    normal and its z slope ``b = d lateral / dz`` (so a sidewall angle tilts it
    out of the plane), and every voxel gets the squared-normal tensor of its
    own model (:func:`_normal_tensor`): its eps variance split into each
    sub-interval's lateral part, along the interface flux weighted by the
    kernel ``(1 - 4 xi^2) (1 - 4 eta^2)`` over the lateral square (which
    vanishes on the square's sides, so a line entering it changes nothing
    abruptly), and the vertical part along z (tilted by ``surface_slopes``).
    For several sub-squares per voxel, :func:`_composite_samples` adds up their
    moments before the tensor.

    Args:
        offsets: (nx, ny, K) signed distance of each level at the voxel
            centers (cells, negative inside); +-BIG for absent / fill.
        normals: (nx, ny, K, 2) unit in-plane gradient of each level's distance.
        is_fill: (K,) static, whether a level fills the whole plane.
        lows: (nx, ny, K) bottom surface per level (cells).
        highs: (nx, ny, K) top surface per level (cells).
        lateral: maps heights (..., ) (cells, voxel k spans [k, k + 1]) to the
            (..., K) offset of each level's boundary there, added to
            ``offsets`` (ignored for fill levels).
        nz: number of voxel layers.
        half: half side of the lateral square (cells); < 0.5 for sub-samples.
        corner: the corner columns of this sample grid, or None (one
            half-plane per level everywhere).
        level_eps: (C, K) static contrast of each level to the background per
            E component, normalized by the eps range, or None: fractions only.
        kernel: (ax, ay, beta): the flux kernel's coordinates ``ax + beta u``
            at the lateral square's points u (its side is 1).
        cell: cells per side of the lateral square.
        surface_slopes: (nx, ny, 2K, 2) lateral gradient (cells per cell) of
            each level's bottom then top surface, or None (flat).
        knots: (K, M) heights (cells) where the levels' offsets bend (their
            z_to_bias knots), or None: every offset is linear in z.
        offdiag: the tensor also has the off-diagonal entries (C, 7; see
            :func:`_normal_tensor`).
        zaxis: the z axis of the sample grid (:class:`_Frame`); a graded one
            gives the voxel layers [lo, hi] (frame units, the surfaces' frame)
            instead of [k, k + 1]. None: [k, k + 1].

    Returns:
        (nx, ny, nz, K + 1): K levels then the background, summing to one;
        with ``level_eps`` also the (nx, ny, nz, C, 4) tensor of
        :func:`_normal_tensor` ((nx, ny, nz, C, 7) with ``offdiag``).
    """
    num_levels = lows.shape[-1]
    with_n = level_eps is not None
    point_slopes = _point_slopes(lows, surface_slopes) if with_n else None
    if with_n:
        kmass = _kernel_mass(kernel)
    graded = zaxis is not None and not zaxis.uniform
    # per column (graded x or y): each column's frame units per side, over the sub-intervals and components
    per_column = isinstance(half, _Cells)
    cell_n = half.cell[..., None, None] if per_column else cell

    @jax.checkpoint
    def one_slice(k: Array) -> Array:
        """Fractions (nx, ny, K + 1) of the voxel layer k (graded: k is the layer's (bottom, top))."""
        split = _split_layer(lows, highs, *k) if graded else _split_layer(lows, highs, k)
        parts = _paint_layer(
            offsets, normals, is_fill, split, lateral, half, level_eps, kernel, cell, point_slopes, knots
        )
        visible, length = parts[:2]  # (nx, ny, S, K + 1), (nx, ny, S)
        out = jnp.stack([jnp.sum(length * visible[..., j], axis=-1) for j in range(num_levels + 1)], axis=-1)
        if graded:
            out = out / (k[1] - k[0])
        if with_n:
            flux, rate, bp = parts[5:8]
            tensor = _normal_tensor(
                length,
                bp,
                flux,
                _weigh_levels(visible[..., :-1], level_eps),
                _weigh_levels(visible[..., :-1], level_eps, power=2),
                rate,
                cell_n,
                kmass,
                parts[8] if len(parts) > 8 else None,
                offdiag=offdiag,
                graded=graded,
            )
            out = (out, tensor)  # (nx, ny, K + 1), (nx, ny, C, 4)
        if corner is None:
            return out  # (nx, ny, K + 1)
        # the corner columns' sub-intervals, for the corner pass, with the
        # offsets at their Gauss heights
        cols = jnp.minimum(corner.cols, length.shape[0] * length.shape[1] - 1)
        gathered = [x.reshape(-1, *x.shape[2:])[cols][:, None] for x in split]
        gauss = _gauss_heights(gathered, lateral, num_levels)[:, 0]
        if per_column:  # frame units -> sides of each corner column's cell
            gauss = gauss / corner.cells.cell[:, None, None, None]
        parts = (*parts[:3], gauss, *parts[4:])
        return out, tuple(x if i == 3 else x.reshape(-1, *x.shape[2:])[cols] for i, x in enumerate(parts))

    if graded:
        ks = (jnp.asarray(zaxis.lo, dtype=lows.dtype), jnp.asarray(zaxis.hi, dtype=lows.dtype))
    else:
        ks = jnp.arange(nz, dtype=lows.dtype)
    if corner is None:
        out = jax.lax.map(one_slice, ks)  # (nz, nx, ny, K+1)
        if with_n:
            return jnp.moveaxis(out[0], 0, 2), jnp.moveaxis(out[1], 0, 2)
        return jnp.moveaxis(out, 0, 2)
    out, columns = jax.lax.map(one_slice, ks)
    if with_n:
        out, tensor = out
    cols = jnp.minimum(corner.cols, offsets.shape[0] * offsets.shape[1] - 1)
    fill = _corner_pass(
        corner.layers,
        corner.taken,
        *columns[:4],
        columns[4:] if with_n else None,
        # the columns' half-planes, for the corner model's straight-edge part
        (offsets.reshape(-1, num_levels)[cols], normals.reshape(-1, num_levels, 2)[cols]),
        trees=corner.trees,
        literals=corner.literals,
        is_fill=tuple(is_fill),
        lines=corner.lines,
        sc=None if per_column else corner.sc,
        half=half.reach if per_column else half,  # per column: the cells' static reach, the cells traced
        level_eps=None if level_eps is None else tuple(tuple(float(e) for e in row) for row in level_eps),
        kernel=tuple(float(x) for x in kernel),
        cell=1.0 if per_column else float(cell),
        offdiag=offdiag,
        graded=graded,
        **({"cells": (corner.cells.hx, corner.cells.hy, corner.cells.cell)} if per_column else {}),
    )  # (nz, N, K + 1)
    if with_n:
        fill, corner_tensor = fill
    if graded:
        fill = fill / (ks[1] - ks[0])[:, None, None]
    out = out.reshape(nz, -1, num_levels + 1).at[:, corner.cols].set(fill, mode="drop").reshape(out.shape)
    if not with_n:
        return jnp.moveaxis(out, 0, 2)
    shape = tensor.shape
    tensor = tensor.reshape(nz, -1, *shape[3:]).at[:, corner.cols].set(corner_tensor, mode="drop").reshape(shape)
    return jnp.moveaxis(out, 0, 2), jnp.moveaxis(tensor, 0, 2)


def _lateral_samples(n_sub: int, edge_width: float) -> tuple[tuple[float, float, tuple[float, float, float]], ...]:
    """The lateral sub-samples of a voxel and their flux kernels.

    Sample (a, b) is the square ``edge_width / n_sub`` cells wide around the
    point (a, b) cells from the voxel center. Where these squares tile the
    voxel (``edge_width`` 1, or one sample), their kernels are the parts of one
    kernel over the voxel, which vanishes on its sides only. Otherwise (gaps or
    overlaps between the squares) each square gets its own kernel, which
    vanishes on its own sides: a line crossing a side that no other square
    continues would bring or take its flux at once.

    Args:
        n_sub: sub-samples per axis.
        edge_width: lateral width (cells) of the voxel square.

    Returns:
        (a, b, kernel) per sample, a-major; kernel is the (ax, ay, beta) of
        :func:`_moments` on the sample's square (side 1).
    """
    sub = [(k + 0.5) / n_sub - 0.5 for k in range(n_sub)]
    cell = edge_width / n_sub
    if n_sub > 1 and not math.isclose(edge_width, 1.0):
        return tuple((a, b, (0.0, 0.0, 1.0)) for a in sub for b in sub)
    span = (1 - 1 / n_sub) + cell  # side (cells) of the union of the squares
    return tuple((a, b, (a / span, b / span, cell / span)) for a in sub for b in sub)


@timed("composite_samples")
def _composite_samples(
    offsets: Array,
    normals: Array,
    is_fill: Sequence[bool],
    lows: Array,
    highs: Array,
    lateral: Callable[[Array], Array],
    nz: int,
    level_eps: Sequence[Sequence[float]],
    kernels: Sequence[tuple[float, float, float]],
    cell: float,
    corners: Sequence["_CornerInputs"] | None = None,
    surface_slopes: Array | None = None,
    half: float = 0.5,
    knots: Array | None = None,
    offdiag: bool = False,
    zaxis: _SampleAxis | None = None,
) -> tuple[Array, Array | None]:
    """Fractions and normal tensor of voxels painted on several lateral sub-squares.

    Every sample (a square in the voxel, see :func:`_lateral_samples`) is
    painted as in :func:`_composite`, with its own distances, corner columns
    and kernel, on the same z sub-intervals (the surfaces are shared). The
    samples' moments per sub-interval are added up before
    :func:`_normal_tensor`: the voxel's lateral variance is the one over all its
    squares, also between them, and its flux is the sum of theirs. With one
    kernel over the voxel (squares that tile it), a line crossing the face
    between two squares takes its flux from one to the other, so the tensor is
    continuous there, and a plane on that face keeps its exact normal. (A
    tensor per square would miss the variance between the squares, and so give
    that plane none, and a vanishing sliver would bring its whole edge's flux
    into its square and turn that square's whole variance.) This continuity
    needs the squares' models to agree on the line: straight edges, and corner
    columns where the corner model is certified. Near a polygon corner painted
    with half-planes (``corner_lines`` 0, or beyond the corner model's
    certified reach), a square's half-plane can be another edge than the one
    that crosses into it from its neighbour, whose flux then comes at once:
    the normal jumps there, like the half-plane model's normal (and fill) where
    a voxel center crosses a corner's bisector.

    The corner columns of every sample are painted first (their half-planes,
    then :func:`_corner_pass` per sub-interval), one sample at a time with the
    kernel's center traced, so that all samples share one compiled pass; the
    voxel layers take them over before adding up. Every sample's corner model
    E is blended with its half-planes T by one weight per voxel and
    sub-interval: the least of the samples' certificates (``raw`` of
    :func:`_corner_fill`; 1 where a column is not a corner column of a sample,
    since its half-plane is exact there, and 0 where a sample's grid had no
    slot for it). Where it is > 0, every sample's E is exact, so the voxel's
    average ``W mean(E) + (1 - W) mean(T)`` is never farther from the exact
    fill than the half-planes' (with a weight per sample, the samples' T errors
    could cancel where some of them are made exact). The certificate ramps to
    1 within ``_VOXEL_RAMP`` of where the corner columns end (their radius is
    that much larger with sub-samples), so the weight stays continuous.

    Args:
        offsets: (P, nx, ny, K) each sample's signed distances (sides of its
            square; see :func:`_composite`).
        normals: (P, nx, ny, K, 2) each sample's unit normals.
        is_fill: (K,) static, whether a level fills the whole plane.
        lows: (nx, ny, K) bottom surface per level (cells).
        highs: (nx, ny, K) top surface per level (cells).
        lateral: see :func:`_composite` (sides of a sample's square).
        nz: number of voxel layers.
        level_eps: (C, K) static contrasts (see :func:`_composite`), or None:
            fractions only.
        kernels: (ax, ay, beta) of every sample (:func:`_lateral_samples`),
            all with the same beta.
        cell: cells per side of a sample's square.
        corners: every sample's corner columns (all with the same number of
            columns and the same static structure), or None: none.
        surface_slopes: see :func:`_composite`.
        half: half side of a sample's square; on graded x or y each column's own cell (dense :class:`_Cells`,
            see :func:`_composite`), ``cell`` then unused.
        knots: see :func:`_composite`.
        offdiag: see :func:`_composite`.
        zaxis: see :func:`_composite`.

    Returns:
        (nx, ny, nz, K + 1) the samples' fractions, summed (not averaged), and
        the voxels' (nx, ny, nz, C, 4) tensor of :func:`_normal_tensor` (None
        without ``level_eps``).

    Raises:
        ValueError: kernels of different scales.
    """
    num_samples = offsets.shape[0]
    nx, ny, num_levels = lows.shape
    num = nx * ny
    with_n = level_eps is not None
    scales = {kernel[2] for kernel in kernels}
    if len(scales) != 1:
        raise ValueError(f"The samples' kernels have different scales {sorted(scales)}")
    # lengths and areas on a sample's square are beta and beta^2 of those on the
    # kernel's square: flux / kmass is then per side of the kernel's square
    scale = scales.pop()
    kmass = scale * sum(_kernel_mass(kernel) for kernel in kernels)
    point_slopes = _point_slopes(lows, surface_slopes) if with_n else None
    if with_n:
        level_eps = tuple(tuple(float(e) for e in row) for row in level_eps)
    graded = zaxis is not None and not zaxis.uniform
    if graded:  # the voxel layers' (bottom, top)
        ks = (jnp.asarray(zaxis.lo, dtype=lows.dtype), jnp.asarray(zaxis.hi, dtype=lows.dtype))
    else:
        ks = jnp.arange(nz, dtype=lows.dtype)
    dtype = lows.dtype
    per_column = isinstance(half, _Cells)  # graded x or y (see _composite)
    cell_n = half.cell[..., None, None] if per_column else cell

    def split_at(lo: Array, hi: Array, k: Array | tuple[Array, Array]) -> tuple[Array, ...]:
        """:func:`_split_layer` of the voxel layer k (graded: its (bottom, top))."""
        return _split_layer(lo, hi, *k) if graded else _split_layer(lo, hi, k)

    def take(x: Array | None, cols: Array) -> Array | None:
        """(nx, ny, ...) -> (N, 1, ...) at the columns cols."""
        return None if x is None else x.reshape(num, *x.shape[2:])[cols][:, None]

    centers = jnp.asarray([kernel[:2] for kernel in kernels], dtype=dtype)  # (P, 2): ax, ay

    def sample_columns(inputs: tuple[Array, ...]) -> tuple[Array, ...]:
        """The corner columns of one sample per sub-interval: their half-planes, then the corner model."""
        o, n, center, cols, taken, layers = inputs
        gathered = tuple(take(x, cols) for x in (o, n, lows, highs, point_slopes))
        kernel = (center[0], center[1], scale)
        # per column: the corner columns' own cells, (N,) and (N, 1) over the gathered columns' batch
        col_cells = _cells_take(half, cols) if per_column else None
        col_half = _cells_expand(col_cells, 1) if per_column else half

        @jax.checkpoint
        def columns(k: Array) -> tuple[Array, ...]:
            """(N, ...) half-plane painting of the corner columns in voxel layer k."""
            go, gn, lo, hi, slopes = gathered
            split = split_at(lo, hi, k)
            parts = _paint_layer(go, gn, is_fill, split, lateral, col_half, level_eps, kernel, cell, slopes, knots)
            gauss = _gauss_heights(split, lateral, num_levels)
            if per_column:  # frame units -> sides of each column's cell
                gauss = gauss / col_cells.cell[:, None, None, None, None]
            parts = (*parts[:3], gauss, *parts[4:7])
            return tuple(x[:, 0] for x in parts)

        dense = jax.lax.map(columns, ks)
        first = corners[0]  # the static structure is the same in every sample
        return _corner_pass(
            layers,
            taken,
            *dense[:4],
            dense[4:] if with_n else None,
            (gathered[0][:, 0], gathered[1][:, 0]),  # the columns' half-planes
            center=center,
            trees=first.trees,
            literals=first.literals,
            is_fill=tuple(is_fill),
            lines=first.lines,
            sc=None if per_column else first.sc,
            half=half.reach if per_column else half,
            level_eps=level_eps,
            kernel=(0.0, 0.0, float(scale)),
            cell=1.0 if per_column else float(cell),
            per_interval=True,
            **({"cells": (col_cells.hx, col_cells.hy, col_cells.cell)} if per_column else {}),
        )  # (nz, N, S, K + 1), (nz, N, S) and (nz, N, S, C, 3), (nz, N, S, C, 2)

    repainted, corner_cols, dropped = (), None, None
    if corners:  # one sample at a time, so that their corner passes compile once
        corner_cols = jnp.stack([corner.cols for corner in corners])  # (P, N)
        repainted = jax.lax.map(
            sample_columns,
            (
                offsets,
                normals,
                centers,
                jnp.minimum(corner_cols, num - 1),
                jnp.stack([corner.taken for corner in corners]),
                jax.tree.map(lambda *x: jnp.stack(x), *(corner.layers for corner in corners)),
            ),
        )
        # per voxel layer, stacked over the samples: (nz, P, N, ...)
        repainted = tuple(jnp.moveaxis(x, 0, 1) for x in repainted)
        # columns that some sample needed but had no room for: half-planes only
        if all(corner.dropped is not None for corner in corners):
            dropped = jnp.any(jnp.stack([corner.dropped for corner in corners]), axis=0)  # (nx * ny,)
    num_s = 2 * num_levels + 1

    def put(x: Array, cols: Array, value: Array) -> Array:
        """x (nx, ny, ...) with the columns cols set to value (N, ...); unused slots (nx * ny) are dropped."""
        return x.reshape(num, *x.shape[2:]).at[cols].set(value, mode="drop").reshape(x.shape)

    @jax.checkpoint
    def one_slice(inputs: tuple[Array, tuple]) -> tuple[Array, Array | None]:
        """Summed fractions (nx, ny, K + 1) and tensor (nx, ny, C, 4) of the voxel layer k."""
        k, corner_layers = inputs
        split = split_at(lows, highs, k)
        perm, bp, length, _ = split
        if corners:
            # the voxel's certificate per sub-interval: the least of its samples'
            # (1 where a column is not a corner column of a sample: its
            # half-plane is exact there), 0 where a sample had no room for it
            certificate = jax.vmap(lambda c, v: jnp.ones((num, num_s), dtype=dtype).at[c].set(v, mode="drop"))(
                corner_cols, corner_layers[1]
            )
            voxel = jnp.min(certificate, axis=0)  # (nx * ny, S)
            if dropped is not None:
                voxel = jnp.where(dropped[:, None], 0.0, voxel)

        @jax.checkpoint
        def add(total: tuple[Array, ...], sample: tuple[Array, ...]) -> tuple[tuple[Array, ...], None]:
            """Adds one sample's fractions and moments (per sub-interval) to the voxels'."""
            o, n, center, *fix = sample
            kernel = (center[0], center[1], scale)
            parts = _paint_layer(o, n, is_fill, split, lateral, half, level_eps, kernel, cell, knots=knots)
            moments = (parts[0], parts[5], parts[6]) if with_n else parts[:1]  # fractions, flux, rate
            if fix:  # the corner model in this sample's corner columns, blended by the voxel's certificate
                cols, exact, _, *exact_moments = fix
                cs = jnp.minimum(cols, num - 1)
                w = voxel[cs]  # (N, S)
                blended = []
                for x, e in zip(moments, (exact, *exact_moments), strict=True):
                    t = x.reshape(num, *x.shape[2:])[cs]
                    ww = w.reshape(*w.shape, *([1] * (t.ndim - 2)))
                    blended.append(put(x, cols, ww * e + (1 - ww) * t))
                moments = tuple(blended)
            fill = jnp.stack([jnp.sum(length * moments[0][..., j], axis=-1) for j in range(num_levels + 1)], axis=-1)
            return tuple(x + y for x, y in zip(total, (fill, *moments), strict=True)), None

        # one sample at a time (as many small steps as separate samples would take)
        zeros = (
            jnp.zeros((nx, ny, num_levels + 1), dtype=dtype),
            jnp.zeros((nx, ny, num_s, num_levels + 1), dtype=dtype),
        )
        if with_n:
            num_c = len(level_eps)
            zeros += (
                jnp.zeros((nx, ny, num_s, num_c, 3), dtype=dtype),
                jnp.zeros((nx, ny, num_s, num_c, 2), dtype=dtype),
            )
        samples = (offsets, normals, centers) + ((corner_cols, *corner_layers) if corners else ())
        total, _ = jax.lax.scan(add, zeros, samples)
        if graded:  # the layer's fractions, not its volumes (frame units)
            total = (total[0] / (k[1] - k[0]), *total[1:])
        if not with_n:
            return total[0], None
        fractions, visible, flux, rate = total
        slopes = None
        if point_slopes is not None:
            slopes = jnp.take_along_axis(point_slopes, perm[..., 1:-1, None], axis=-2)
        tensor = _normal_tensor(
            length,
            bp,
            flux,
            _weigh_levels(visible[..., :-1], level_eps) / num_samples,
            _weigh_levels(visible[..., :-1], level_eps, power=2) / num_samples,
            rate / num_samples,
            cell_n,
            kmass,
            slopes,
            offdiag=offdiag,
            graded=graded,
        )
        return fractions, tensor

    fractions, tensor = jax.lax.map(one_slice, (ks, repainted))
    fractions = jnp.moveaxis(fractions, 0, 2)
    return fractions, (None if tensor is None else jnp.moveaxis(tensor, 0, 2))


@timed("interface_composite")
def _interface_composite(
    offsets: Array,
    normals: Array,
    is_fill: Sequence[bool],
    lows: Array,
    highs: Array,
    lateral: Callable[[Array], Array],
    nz: int,
    half: float,
    knots: Array | None,
    cell: float,
    vectors: np.ndarray,
    contrast2: np.ndarray,
    surface_slopes: Array | None = None,
    corner: "_CornerInputs | None" = None,
    zaxis: _SampleAxis | None = None,
) -> tuple[Array, Array]:
    """Volume fractions and the interface direction tensor of every voxel.

    Every voxel's lateral faces (:func:`_interface_parts`) and the variance of its material moments along
    z (:func:`_vertical_tensor`) give ``D = sum_faces w N N^T``: its interfaces' normals N (3D, a sloped
    wall tilted out of the plane), weighted by area, squared contrast and the fade-in of thin parts. Its
    spread off the leading direction decides whether the voxel is planar (:func:`junction_weight`). The
    fractions are those of :func:`_composite` (the same painting).

    Args:
        offsets: (nx, ny, K) signed distance of each level at the sample points (sides of the square).
        normals: (nx, ny, K, 2) unit normals.
        is_fill: (K,) static.
        lows: (nx, ny, K) bottom surfaces (cells).
        highs: (nx, ny, K) top surfaces (cells).
        lateral: see :func:`_composite`.
        nz: number of voxel layers.
        half: half side of the lateral square.
        knots: see :func:`_composite`.
        cell: cells per side of the lateral square.
        vectors: (K + 1, 6) static normalized material tensors (see :func:`_interface_parts`).
        contrast2: (K + 1, K + 1) static squared contrasts.
        surface_slopes: (nx, ny, 2K, 2) or None, see :func:`_composite`.
        corner: the corner columns of this sample grid, or None: in them the corner model
            (:func:`_corner_interfaces`) replaces the half-planes, fractions and faces alike.
        zaxis: see :func:`_composite`. On a graded axis the lateral faces are as high as the layer
            (frame units), so they weigh against the horizontal ones as in the physical voxel.

    On graded x or y ``half`` is the columns' own cells (dense :class:`_Cells`, see :func:`_composite`).

    Returns:
        fractions: (nx, ny, nz, K + 1).
        direction: (nx, ny, nz, 6) the direction tensor (xx, yy, zz, xy, xz, yz).
        uncertified: () the corner columns' voxels whose corner model is not certified (0 without one).
    """
    point_slopes = _point_slopes(lows, surface_slopes)
    num_levels = lows.shape[-1]
    per_column = isinstance(half, _Cells)
    if per_column:  # each column's (frame units^2), over the tensor's entries
        area = half.cell[..., None] ** 2
    else:
        area = (2 * half * cell) ** 2 if not isinstance(half, tuple) else _cell_area(half) * cell**2  # as the faces
    graded = zaxis is not None and not zaxis.uniform

    @jax.checkpoint
    def one_slice(k: Array) -> tuple:
        split = _split_layer(lows, highs, *k) if graded else _split_layer(lows, highs, k)
        visible, lateral_tensor, moments, slopes, b = _interface_parts(
            offsets, normals, is_fill, split, lateral, half, knots, cell, vectors, contrast2, point_slopes
        )
        length = split[2]
        fractions = jnp.sum(length[..., None] * visible, axis=-2)
        if graded:
            fractions = fractions / (k[1] - k[0])
        direction = jnp.sum(length[..., None] * lateral_tensor, axis=-2)
        if graded:
            direction = direction + _vertical_tensor(length, split[1], moments, slopes, graded=True) * area
        else:
            direction = direction + _vertical_tensor(length, split[1], moments, slopes) * area
        if corner is None:
            return fractions, direction
        # the corner columns' sub-intervals, for the corner pass, with the offsets at their Gauss heights
        cols = jnp.minimum(corner.cols, length.shape[0] * length.shape[1] - 1)

        def take(x: Array) -> Array:
            return x.reshape(-1, *x.shape[2:])[cols]

        gauss = _gauss_heights([x.reshape(-1, *x.shape[2:])[cols][:, None] for x in split], lateral, num_levels)
        if per_column:  # frame units -> sides of each corner column's cell
            gauss = gauss / corner.cells.cell[:, None, None, None, None]
        b_s = jnp.broadcast_to(b[..., None, :], (*length.shape, num_levels))
        columns = (
            take(visible), take(lateral_tensor), take(moments), take(length), take(split[1]), take(split[3]),
            gauss[:, 0], take(b_s), take(slopes),
        )
        return (fractions, direction), columns

    if graded:
        ks = (jnp.asarray(zaxis.lo, dtype=lows.dtype), jnp.asarray(zaxis.hi, dtype=lows.dtype))
    else:
        ks = jnp.arange(nz, dtype=lows.dtype)
    if corner is None:
        fractions, direction = jax.lax.map(one_slice, ks)
        return jnp.moveaxis(fractions, 0, 2), jnp.moveaxis(direction, 0, 2), jnp.zeros((), dtype=jnp.int32)
    (fractions, direction), columns = jax.lax.map(one_slice, ks)
    cols = jnp.minimum(corner.cols, offsets.shape[0] * offsets.shape[1] - 1)
    c_fractions, c_direction, uncertified = _interface_corner_pass(
        corner.layers,
        corner.taken,
        columns,
        (jnp.clip(offsets.reshape(-1, num_levels)[cols], -_BIG, _BIG), normals.reshape(-1, num_levels, 2)[cols]),
        trees=corner.trees,
        literals=corner.literals,
        is_fill=tuple(is_fill),
        lines=corner.lines,
        sc=None if per_column else corner.sc,
        half=half.reach if per_column else half,  # per column: the cells' static reach, the cells traced
        cell=1.0 if per_column else float(cell),
        vectors=tuple(tuple(float(x) for x in row) for row in vectors),
        contrast2=tuple(tuple(float(x) for x in row) for row in contrast2),
        **({"graded": True} if graded else {}),
        **({"cells": (corner.cells.hx, corner.cells.hy, corner.cells.cell)} if per_column else {}),
    )  # (nz, N, K + 1), (nz, N, 6), (nz,)
    fractions = fractions.reshape(nz, -1, num_levels + 1).at[:, corner.cols].set(c_fractions, mode="drop")
    direction = direction.reshape(nz, -1, 6).at[:, corner.cols].set(c_direction, mode="drop")
    shape = offsets.shape[:2]
    return (
        jnp.moveaxis(fractions.reshape(nz, *shape, num_levels + 1), 0, 2),
        jnp.moveaxis(direction.reshape(nz, *shape, 6), 0, 2),
        jnp.sum(uncertified),
    )


@autoinit
class TopRoughness(TreeClass):
    """Gaussian roughness of the top surface of one level.

    Only the top surface of ``level`` is displaced. Where it dips below its
    nominal height the gap is filled by the surrounding layers: levels deposited
    on that surface (nominal bottom == the level's nominal top) follow it down
    when ``fill_above`` is set, and any other level covering that height (e.g. a
    cladding spanning it) fills it by ``mesh_order``. Where it rises, it
    displaces the material above.

    Attributes:
        level: name of the level whose top surface is rough.
        rms: standard deviation of the height offset (um).
        correlation_length: correlation length (um) of the Gaussian autocorrelation.
        fill_above: levels whose bottom sits on this surface follow it (no voids,
            no overlaps).
    """

    #: name of the level whose top surface is rough
    level: str = frozen_field()
    #: standard deviation of the height offset (um)
    rms: float = field()
    #: correlation length (um) of the Gaussian autocorrelation
    correlation_length: float = frozen_field(default=0.0)
    #: levels whose bottom sits on this surface follow it (no voids, no overlaps)
    fill_above: bool = frozen_field(default=True)

    def __post_init__(self) -> None:
        if not isinstance(self.level, str) or not self.level:
            raise ValueError(f"TopRoughness level must be a non-empty string, got {self.level!r}")
        if not isinstance(self.fill_above, bool):
            raise TypeError(f"TopRoughness fill_above must be a bool, got {self.fill_above!r}")
        if not has_tracers(self.rms):
            if isinstance(self.rms, bool):
                raise TypeError(f"TopRoughness rms must be numeric, got {self.rms!r}")
            rms_val = float(self.rms)
            if not math.isfinite(rms_val) or rms_val < 0.0:
                raise ValueError(f"TopRoughness rms must be finite and >= 0, got {self.rms!r}")
        if not has_tracers(self.correlation_length):
            if isinstance(self.correlation_length, bool):
                raise TypeError(f"TopRoughness correlation_length must be numeric, got {self.correlation_length!r}")
            cl_val = float(self.correlation_length)
            if not math.isfinite(cl_val) or cl_val < 0.0:
                raise ValueError(
                    f"TopRoughness correlation_length must be finite and >= 0, got {self.correlation_length!r}"
                )

    def sample(
        self, shape: tuple[int, int], res_um: float, key: Array, widths: tuple[float, float] | None = None
    ) -> Array:
        """Samples one rough surface.

        Args:
            shape: (nx, ny) of the device grid.
            res_um: grid resolution in um (the frame unit).
            key: PRNG key.
            widths: (wx, wy) the cells' widths in frame units where they differ along x and y, or None.

        Returns:
            (nx, ny) height offsets in cells (frame units).
        """
        if widths is None:
            f = gaussian_random_field(shape, self.correlation_length / res_um, key)
        else:
            length = self.correlation_length / res_um
            f = gaussian_random_field(shape, (length / widths[0], length / widths[1]), key)
        return (self.rms / res_um) * f


def _roughness_reach(
    items: Sequence[tuple[str, float, float]],
    roughness: Sequence[TopRoughness],
    bottom: float,
    top: float,
    sigma: float = 5.0,
) -> tuple[set[str], dict[str, tuple[float, float]]]:
    """Level names whose vertical span (padded by ``5 * rms`` or coupled via ``fill_above``) overlaps ``(bottom, top)``."""
    spans = {name: (float(lo), float(hi)) for name, lo, hi in items}
    padded = dict(spans)
    deps: dict[str, set[str]] = {}
    traced: set[str] = set()
    for r in roughness:
        if r.level not in spans:
            continue
        rms_val = _primal(r.rms)
        if isinstance(rms_val, jax.core.Tracer):
            traced.add(r.level)
            pad = 0.0
        else:
            pad = sigma * float(rms_val)
        top_r = spans[r.level][1]
        lo_r, hi_r = padded[r.level]
        padded[r.level] = (min(lo_r, top_r - pad), max(hi_r, top_r + pad))
        if not r.fill_above:
            continue
        for name, (lo, hi) in spans.items():
            if name != r.level and abs(lo - top_r) < 1e-6:
                lo_c, hi_c = padded[name]
                padded[name] = (min(lo_c, lo - pad), max(hi_c, hi))
                deps.setdefault(name, set()).add(r.level)
                if r.level in traced:
                    traced.add(name)
    keep = {name for name, (lo, hi) in padded.items() if (hi > bottom and lo < top) or name in traced}
    # Retain fill_above roughness bases used by kept levels so _level_surfaces finds tops[r.level].
    frontier = list(keep)
    while frontier:
        cur = frontier.pop()
        for base in deps.get(cur, ()):
            if base not in keep:
                keep.add(base)
                frontier.append(base)
    return keep, padded


def _anisotropy(w: Array, e: np.ndarray, ignore: np.ndarray) -> Array:
    """``<eps> <1/eps> - 1`` of the voxels, as a sum over pairs of materials.

    ``a = 1/2 sum_lm w_l w_m (e_l - e_m)^2 / (e_l e_m)``, which equals
    ``<eps> <1/eps> - 1`` where the weights sum to one, and is exactly 0 (with
    a zero derivative) wherever only one material, or materials of one eps, are
    present: the Kottke average ``(1 + n2 a) / <eps>`` then does not depend on
    the normal.

    Args:
        w: (M, nx, ny, nz) material fractions.
        e: (M,) static permittivities.
        ignore: (M,) static, materials left out (never painted).

    Returns:
        (nx, ny, nz) the anisotropy a.
    """
    a = jnp.zeros(w.shape[1:], dtype=w.dtype)
    for i in range(len(e)):
        for j in range(i + 1, len(e)):
            g = (e[i] - e[j]) ** 2 / (e[i] * e[j])
            if g != 0 and not (ignore[i] or ignore[j]):
                a = a + float(g) * (w[i] * w[j])
    return a


def _center_distances(axis: _SampleAxis) -> np.ndarray | None:
    """(n + 1,) distances between the centers of neighbouring sample voxels, for :func:`_normal_squared`.

    None on a uniform axis (a distance of one cell). The two outer entries face the edge-padded
    neighbours, whose difference is zero, and only need to be nonzero.
    """
    if axis.uniform:
        return None
    centers = 0.5 * (axis.lo + axis.hi)
    inner = np.diff(centers)
    edge = [float(axis.hi[0] - axis.lo[0])] if len(inner) == 0 else [float(inner[0])]
    return np.concatenate([edge, inner, [float(inner[-1]) if len(inner) else edge[0]]])


def _normal_squared(q: Array, spacing: Sequence[np.ndarray | None] | None = None) -> Array:
    """Squared components (3, nx, ny, nz) of the interface normal of ``q``.

    Uses the mean of the squared forward and backward differences, which (unlike
    central differences) does not vanish where a film thinner than a voxel makes
    ``q`` extremal, and is zero along axes of a single cell (2D simulations).
    The subpixel average uses it only for the variance whose interface flux
    cancels inside the voxel (see :func:`_normal_tensor`), where the voxel's
    own model gives no direction.

    Args:
        q: (nx, ny, nz) mean permittivity.
        spacing: per axis, None (neighbours one cell apart) or the distances of
            :func:`_center_distances` (frame units), which divide the differences.

    Returns:
        (3, nx, ny, nz) squared normal components, summing to one (or zero
        where ``q`` is uniform).
    """
    g2 = []
    for axis in range(3):
        pad = [(1, 1) if a == axis else (0, 0) for a in range(3)]
        d = jnp.diff(jnp.pad(q, pad, mode="edge"), axis=axis)  # n + 1 differences
        if spacing is not None and spacing[axis] is not None:
            shape = [1, 1, 1]
            shape[axis] = -1
            d = d / jnp.asarray(spacing[axis], dtype=d.dtype).reshape(shape)
        n = q.shape[axis]
        forward = jax.lax.slice_in_dim(d, 1, n + 1, axis=axis)
        backward = jax.lax.slice_in_dim(d, 0, n, axis=axis)
        g2.append(0.5 * (forward**2 + backward**2))
    g2 = jnp.stack(g2)
    return g2 / (jnp.sum(g2, axis=0, keepdims=True) + 1e-12)


def _normal_dyad(q: Array) -> Array:
    """Off-diagonal entries (xy, xz, yz; (3, nx, ny, nz)) of the finite-difference normal tensor of ``q``.

    Products of the central differences (the mean of the forward and backward
    ones), normalized like :func:`_normal_squared`, so that with its diagonal
    they form a positive semidefinite tensor of unit trace (zero where ``q`` is
    uniform, and along axes of a single cell).
    """
    g, g2 = [], []
    for axis in range(3):
        pad = [(1, 1) if a == axis else (0, 0) for a in range(3)]
        d = jnp.diff(jnp.pad(q, pad, mode="edge"), axis=axis)  # n + 1 differences
        n = q.shape[axis]
        forward = jax.lax.slice_in_dim(d, 1, n + 1, axis=axis)
        backward = jax.lax.slice_in_dim(d, 0, n, axis=axis)
        g.append(0.5 * (forward + backward))
        g2.append(0.5 * (forward**2 + backward**2))
    total = jnp.sum(jnp.stack(g2), axis=0) + 1e-12
    return jnp.stack([g[i] * g[j] for i, j in _OFFDIAG_PAIRS]) / total
