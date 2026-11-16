"""Voxel square clipping primitives, moments, and analytic 2D corner geometry."""

import math
import jax
import jax.numpy as jnp
import numpy as np
from collections.abc import Callable, Iterator, Sequence
from functools import partial
from typing import Any, NamedTuple
from ..timing import timed
from ._layers import (
    Array,
    _BIG,
    _CERTIFIED,
    _CORNER_RAMP,
    _CORNER_TOL,
    _FACE_THICKNESS,
    _GL3,
    _HIGHEST,
    _KOTTKE_ETA,
    _KOTTKE_TINY,
    _Level,
    _MITRE_MIN,
    _OFFDIAG_PAIRS,
    _ON_LINE,
    _PARALLEL,
    _PART_TINY,
    _PLANE_TOL,
    _STRAIGHT_TOL,
    _STRAIGHT_WIDTH,
    _TIP_AREA,
    _T_FAR,
    _VOXEL_RAMP,
    _eval_bool,
)


def _clip(
    poly: Array, count: Array, n: Array, s: Array, attr: Array | None = None, new_attr: Array | None = None
) -> tuple[Array, ...]:
    """Clips convex polygons to the half-planes ``n . p + s >= 0``.

    With ``attr``, every vertex carries the attributes of its outgoing edge: a
    kept vertex keeps its own, a crossing that leaves the kept side starts an
    edge on the clipping line (``new_attr``), and a crossing that enters it
    continues the edge it lies on.

    Args:
        poly: (..., V, 2) vertices; the first ``count`` are valid, the rest repeat
            the last valid one.
        count: (...,) number of valid vertices.
        n: (..., 2) normals.
        s: (...,) offsets.
        attr: (..., V, A) attributes of the vertices' outgoing edges, or None.
        new_attr: (..., A) attributes of the edges on the clipping line.

    Returns:
        The clipped polygons (..., V, 2) in the same layout and their vertex
        counts; with ``attr`` also the clipped polygons' attributes (..., V, A).
    """
    num = poly.shape[-2]
    i = jnp.arange(num)
    valid = i < count[..., None]
    nxt = jnp.where(i + 1 < count[..., None], i + 1, 0)
    q = jnp.take_along_axis(poly, nxt[..., None], axis=-2)
    fp = jnp.sum(poly * n[..., None, :], axis=-1) + s[..., None]
    fq = jnp.sum(q * n[..., None, :], axis=-1) + s[..., None]
    keep = valid & (fp >= 0)
    cross = valid & ((fp >= 0) != (fq >= 0))
    # an edge (nearly) on the line has no well-defined crossing: take its kept
    # end, which changes no area and keeps the derivatives finite
    denom = fp - fq
    ok = cross & (jnp.abs(denom) > 16 * jnp.finfo(poly.dtype).eps)
    t = jnp.where(ok, fp / jnp.where(ok, denom, 1.0), jnp.where(fp >= 0, 0.0, 1.0))
    x = poly + t[..., None] * (q - poly)
    cand = jnp.stack([poly, x], axis=-2).reshape(*poly.shape[:-2], 2 * num, 2)
    if attr is not None:
        x_attr = jnp.where((fp >= 0)[..., None], new_attr[..., None, :], attr)
        cand_attr = jnp.stack([attr, x_attr], axis=-2).reshape(*poly.shape[:-2], 2 * num, attr.shape[-1])
        cand = jnp.concatenate([cand, cand_attr], axis=-1)  # one gather for both
    mask = jnp.stack([keep, cross], axis=-1).reshape(*poly.shape[:-2], 2 * num)
    cum = jnp.cumsum(mask, axis=-1)
    new_count = cum[..., -1]
    # slot j takes the (j+1)-th kept candidate; later slots repeat the last one
    # (an empty result still gathers in bounds: out-of-bounds gathers fill NaN,
    # which the masked area ignores but its derivative does not)
    j = jnp.minimum(i, jnp.maximum(new_count[..., None] - 1, 0))
    src = jnp.minimum(jnp.sum(cum[..., None, :] <= j[..., :, None], axis=-1), 2 * num - 1)
    out = jnp.take_along_axis(cand, src[..., None], axis=-2)
    if attr is None:
        return out, new_count
    return out[..., :2], new_count, out[..., 2:]


def _area(poly: Array, count: Array) -> Array:
    """Area of counter-clockwise polygons (shoelace).

    Args:
        poly: (..., V, 2) vertices, the first ``count`` valid.
        count: (...,) number of valid vertices.

    Returns:
        (...,) areas.
    """
    i = jnp.arange(poly.shape[-2])
    q = jnp.take_along_axis(poly, jnp.where(i + 1 < count[..., None], i + 1, 0)[..., None], axis=-2)
    cross = poly[..., 0] * q[..., 1] - q[..., 0] * poly[..., 1]
    return 0.5 * jnp.sum(jnp.where(i < count[..., None], cross, 0.0), axis=-1)


class _Cells(NamedTuple):
    """The lateral cells of a grid whose x or y cells are graded: one per column.

    Each is the rectangle of :func:`_cell_corners` (area 1) for its own widths, in its own frame of ``cell``
    frame units per side (see StackDevice._sample_cells). The arrays broadcast against the leading (column) axes
    of the call that paints them: the caller expands them (:func:`_cells_expand`) to that call's batch.
    """

    hx: Array | float  # half sides along x (sides of the cell's frame)
    hy: Array | float
    cell: Array | float  # frame units per side of the cell's frame
    reach: float  # static: 2 max(hx, hy) over every cell (the painted offsets' clip)
    area: float = 1.0  # static: 4 hx hy (1; 1 / per^2 for the junction closure's sub-rectangles)


def _cells_expand(cells: _Cells, extra: int) -> _Cells:
    """The cells with ``extra`` trailing singleton axes, to broadcast against a batch with more axes."""
    if extra == 0:
        return cells
    index = (...,) + (None,) * extra
    return cells._replace(hx=cells.hx[index], hy=cells.hy[index], cell=cells.cell[index])


def _cells_twice(cells: _Cells) -> _Cells:
    """The cells of (N,) columns twice, (2N,): the corner model's two Gauss heights."""
    return cells._replace(**{f: jnp.concatenate([getattr(cells, f)] * 2) for f in ("hx", "hy", "cell")})


def _cells_take(cells: _Cells, cols: Array) -> _Cells:
    """The cells of the columns ``cols`` (flat, i-major) of dense (nx, ny) cells: (N,) arrays."""
    return cells._replace(
        hx=jnp.reshape(cells.hx, (-1,))[cols], hy=jnp.reshape(cells.hy, (-1,))[cols],
        cell=jnp.reshape(cells.cell, (-1,))[cols],
    )


def _half_sides(half: Any) -> tuple[Any, Any]:
    """(hx, hy) of a lateral cell."""
    if isinstance(half, _Cells):
        return half.hx, half.hy
    return half if isinstance(half, tuple) else (half, half)


def _cell_corners(half: Any, dtype: Any) -> Array:
    """(4, 2) counter-clockwise corners of a lateral cell, centered on it ((..., 4, 2) per column for _Cells)."""
    if isinstance(half, _Cells):
        hx, hy = (jnp.asarray(x, dtype=dtype) for x in (half.hx, half.hy))
        return jnp.stack([jnp.stack(c, axis=-1) for c in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))], axis=-2)
    if not isinstance(half, tuple):
        return jnp.asarray([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=dtype)
    hx, hy = half
    return jnp.asarray([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=dtype)


def _cell_area(half: Any) -> float:
    """Area of a lateral cell (the same for every rectangle of a _Cells: 1, or a sub-rectangle's)."""
    if isinstance(half, _Cells):
        return half.area
    return 4.0 * half[0] * half[1] if isinstance(half, tuple) else (2 * half) ** 2


def _cell_reach(half: Any) -> float:
    """An offset beyond which a line keeps or removes a lateral cell whole (its painted offsets are clipped to it)."""
    if isinstance(half, _Cells):
        return half.reach
    return 2.0 * max(half) if isinstance(half, tuple) else 2 * half


def _cell_standard(half: Any, side: float, dtype: Any) -> Any:
    """Scale of first moments about a cell's center: ``sqrt(12) / side`` per axis (the sides' standard deviation)."""
    if isinstance(half, _Cells):
        return jnp.stack([math.sqrt(12.0) / (2.0 * jnp.asarray(h, dtype=dtype)) for h in (half.hx, half.hy)], axis=-1)
    if not isinstance(half, tuple):
        return math.sqrt(12.0) / side
    return jnp.asarray([math.sqrt(12.0) / (2.0 * half[0]), math.sqrt(12.0) / (2.0 * half[1])], dtype=dtype)


def _cell_diagonal(half: Any) -> Any:
    """Half-diagonal of a lateral cell."""
    if isinstance(half, _Cells):
        return jnp.hypot(half.hx, half.hy)
    return math.hypot(*half) if isinstance(half, tuple) else math.sqrt(2.0) * half


def _kernel_scales(kernel: tuple, half: Any) -> tuple:
    """The kernel (ax, ay, beta) of :func:`_moments` on a lateral cell: on a rectangle, per-axis scales.

    The kernel is ``psi(ax + beta u / side_x) psi(ay + beta v / side_y)`` over the cell's sides; a square of side
    1 (half 0.5) keeps the 3-entry kernel. For _Cells the scales get a trailing axis for :func:`_moments`'s
    vertices.
    """
    if isinstance(half, _Cells):
        ax, ay, beta = kernel
        return (ax, ay, (beta / (2.0 * half.hx))[..., None], (beta / (2.0 * half.hy))[..., None])
    if not isinstance(half, tuple):
        return kernel
    ax, ay, beta = kernel
    return (ax, ay, beta / (2.0 * half[0]), beta / (2.0 * half[1]))


def _moments(
    poly: Array, count: Array, attr: Array, kernel: tuple[float, float, float]
) -> tuple[Array, Array, Array]:
    """Area and kernel-weighted interface flux of counter-clockwise polygons.

    Every edge carries the attributes ``(n_x, n_y, b, ...)`` of the interface it
    lies on (:func:`_clip`), the plane ``n . p + b z + c = 0`` with a unit
    in-plane normal n, then further rates (``b`` times a lag, see
    :func:`_paint_layer`), or 0 on edges that are not interfaces. ``ell = du n_y - dv n_x``
    of an edge (du, dv) is then +-its length on its line (0 elsewhere, and no
    square root), and ``ell * (n, b)`` is the edge's area vector per unit
    height, pointing into the polygon (for either sign of the attributes).
    The kernel ``w = psi(ax + beta u) psi(ay + beta v)``, ``psi(x) = 1 - 4 x^2``,
    is integrated exactly along each edge (3-point Gauss-Legendre, degree 4).

    Args:
        poly: (..., V, 2) vertices, the first ``count`` valid.
        count: (...,) number of valid vertices.
        attr: (..., V, A) attributes of the vertices' outgoing edges, A >= 3.
        kernel: (ax, ay, beta): the kernel's center (floats or traced
            scalars) and its scale; or (ax, ay, beta_x, beta_y) on a rectangular
            cell (:func:`_kernel_scales`).

    Returns:
        area: (...,) areas (as :func:`_area`).
        flux: (..., 3) sum over the edges of ``ell * mean(w) * (n, b)``.
        rate: (..., A - 2) sum over the edges of ``ell * attr[2:]``: first the
            rate of change of the area with height when the planes move (p and
            z in one unit), then the further rates.
    """
    if len(kernel) == 3:
        ax, ay, beta = kernel
        bx = by = beta
    else:
        ax, ay, bx, by = kernel
    area = _area(poly, count)
    i = jnp.arange(poly.shape[-2])
    q = jnp.take_along_axis(poly, jnp.where(i + 1 < count[..., None], i + 1, 0)[..., None], axis=-2)
    du, dv = q[..., 0] - poly[..., 0], q[..., 1] - poly[..., 1]
    ell = jnp.where(i < count[..., None], du * attr[..., 1] - dv * attr[..., 0], 0.0)
    omega = 0.0
    for t, weight in _GL3:
        xi = ax + bx * (poly[..., 0] + t * du)
        eta = ay + by * (poly[..., 1] + t * dv)
        omega = omega + weight * (1 - 4 * xi**2) * (1 - 4 * eta**2)
    flux = jnp.sum((ell * omega)[..., None] * attr[..., :3], axis=-2)
    return area, flux, jnp.sum(ell[..., None] * attr[..., 2:], axis=-2)


def _cross(x: Array, y: Array) -> Array:
    """z component of the cross product of (..., 2) vectors."""
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]


def _angle(x: Array, y: Array) -> Array:
    """Signed angle from x to y, (..., 2) vectors; 0 where |x| |y| < 1e-12."""
    ok = jnp.sum(x * x, axis=-1) * jnp.sum(y * y, axis=-1) >= 1e-24
    cr = jnp.where(ok, _cross(x, y), 0.0)
    dot = jnp.where(ok, jnp.sum(x * y, axis=-1), 1.0)
    return jnp.where(ok, jnp.arctan2(cr, dot), 0.0)


def _disk_area(poly: Array, count: Array, center: Array, r: Array) -> Array:
    """Area of counter-clockwise convex polygons inside disks.

    Sums, over the edges p -> q taken relative to the center, the signed area
    of the triangle (center, p, q) inside the disk: a circular sector where the
    edge is outside the circle, the triangle where it is inside.

    Args:
        poly: (..., V, 2) vertices, the first ``count`` valid.
        count: (...,) number of valid vertices.
        center: (..., 2) disk centers.
        r: (...,) radii.

    Returns:
        (...,) areas.
    """
    i = jnp.arange(poly.shape[-2])
    p = poly - center[..., None, :]
    q = jnp.take_along_axis(p, jnp.where(i + 1 < count[..., None], i + 1, 0)[..., None], axis=-2)
    dd = q - p
    r2 = (r * r)[..., None]
    a = jnp.sum(dd * dd, axis=-1)
    b = jnp.sum(p * dd, axis=-1)
    disc = b * b - a * (jnp.sum(p * p, axis=-1) - r2)
    edge = a >= 1e-12  # padding repeats the last vertex
    hit = edge & (disc > 1e-12)  # the edge's line crosses the circle
    root = jnp.sqrt(jnp.where(hit, disc, 1.0))
    a = jnp.where(hit, a, 1.0)
    # the part of the edge inside the circle; without one, the whole edge is a sector
    t1 = jnp.where(hit, jnp.clip((-b - root) / a, 0.0, 1.0), 1.0)
    t2 = jnp.where(hit, jnp.clip((-b + root) / a, 0.0, 1.0), 1.0)
    p1 = p + t1[..., None] * dd
    p2 = p + t2[..., None] * dd
    area = 0.5 * r2 * (_angle(p, p1) + _angle(p2, q)) + 0.5 * _cross(p1, p2)
    keep = (i < count[..., None]) & edge & (r[..., None] >= 1e-9)
    return jnp.sum(jnp.where(keep, area, 0.0), axis=-1)


def _sidewall_shift(level: _Level, z: Array, zb: Array, t: Array, angle: Array, res_um: float) -> Array:
    """Lateral offset of a level's boundary at heights z.

    ``(z - z_ref) tan(angle) - bias - z_to_bias(z)`` with ``z_ref = zb +
    width_to_z * t``; ``z_to_bias`` is interpolated at the fraction ``(z - zb) / t``
    of the thickness (0 at zmin, 1 at zmin + thickness, also for negative
    thicknesses). It is added to the level's signed distance: positive shrinks
    the level.

    Args:
        level: the level (``width_to_z``, ``bias``, ``z_to_bias``).
        z: heights (cells).
        zb: the level's zmin (cells, same frame as z).
        t: the level's thickness (cells, negative grows down).
        angle: sidewall angle (degrees).
        res_um: grid resolution in um (converts the biases to cells).

    Returns:
        Offsets (cells) shaped like z, in z's dtype.
    """
    z_ref = zb + level.width_to_z * t
    grow = level.bias / res_um
    if level.z_to_bias is not None:
        zs, bs = level.z_to_bias
        safe_t = jnp.where(t == 0, 1.0, t)
        frac = jnp.where(t == 0, 0.0, (z - zb) / safe_t)
        grow = grow + jnp.interp(frac, jnp.asarray(zs, dtype=z.dtype), jnp.asarray(bs, dtype=z.dtype)) / res_um
    return ((z - z_ref) * jnp.tan(jnp.deg2rad(angle)) - grow).astype(z.dtype)


def _paint_levels(
    poly: Array,
    count: Array,
    area0: Array,
    n: Array,
    o: Array,
    present: Array,
    is_fill: Sequence[bool],
    half: float,
    attr: Array | None = None,
    slope: Array | None = None,
    kernel: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[Array, ...]:
    """Paints levels in priority order into convex cells of the lateral square.

    Level j gets the part of the cell inside its half-plane ``n . p + o < 0``
    but outside every earlier present level, computed exactly by clipping the
    cell with the complements in priority order; fill levels take the whole
    rest of the cell; the background gets what is left. With ``attr``, every
    edge a clip creates carries its level's plane ``(n, slope)`` and each
    level's part also gets its interface moments (:func:`_moments`).

    Args:
        poly: (..., V, 2) cells (convex, counter-clockwise), the first
            ``count`` vertices valid; V must hold the vertices the clips add.
        count: (...,) number of valid vertices.
        area0: (...,) area of the cells over the square's.
        n: (..., K, 2) unit normals of the levels' half-planes.
        o: (..., K) their offsets (sides of the square).
        present: (..., K) whether each level is there.
        is_fill: (K,) static, whether a level fills the whole plane.
        half: half side of the lateral square.
        attr: (..., V, 2 + R) the cells' edge attributes (see
            :func:`_moments`; 0 on edges that are not interfaces), or None:
            fractions only.
        slope: (..., K, R) z slope of each level's half-plane, then its
            further rates (with ``attr``).
        kernel: (ax, ay, beta) of :func:`_moments` (with ``attr``).

    Returns:
        visible: (..., K + 1) fractions of the square: the K levels, then the
            background.
        overflow: (...,) a clip produced more vertices than V.
        flux: (..., K, 3) with ``attr``: the kernel flux of each level's part.
        rate: (..., K, R) with ``attr``: its rates (see :func:`_moments`).
    """
    full_area = _cell_area(half)
    area = area0  # uncovered fraction so far
    overflow = jnp.zeros(count.shape, dtype=bool)
    visible = []
    moments = attr is not None
    if moments:
        kernel = _kernel_scales(kernel, half)
        _, flux, rate = _moments(poly, count, attr, kernel)  # of the uncovered part so far
        fluxes, rates = [], []
    for j, fill in enumerate(is_fill):
        here = present[..., j]
        if fill:
            visible.append(jnp.where(here, area, 0.0))
            area = jnp.where(here, 0.0, area)
            count = jnp.where(here, 0, count)
            if moments:
                fluxes.append(jnp.where(here[..., None], flux, 0.0))
                rates.append(jnp.where(here[..., None], rate, 0.0))
                flux = jnp.where(here[..., None], 0.0, flux)
                rate = jnp.where(here[..., None], 0.0, rate)
            continue
        # beyond the square's half-diagonal a line keeps or removes it whole
        s = jnp.clip(o[..., j], -_cell_reach(half), _cell_reach(half))
        if moments:
            new_attr = jnp.concatenate([n[..., j, :], slope[..., j, :]], axis=-1)
            clipped, c_count, c_attr = _clip(poly, count, n[..., j, :], s, attr, new_attr)
            c_area, c_flux, c_rate = _moments(clipped, c_count, c_attr, kernel)
            c_area = c_area / full_area  # the part outside level j
            fluxes.append(jnp.where(here[..., None], flux - c_flux, 0.0))
            rates.append(jnp.where(here[..., None], rate - c_rate, 0.0))
            flux = jnp.where(here[..., None], c_flux, flux)
            rate = jnp.where(here[..., None], c_rate, rate)
            attr = jnp.where(here[..., None, None], c_attr, attr)
        else:
            clipped, c_count = _clip(poly, count, n[..., j, :], s)
            c_area = _area(clipped, c_count) / full_area  # the part outside level j
        visible.append(jnp.where(here, area - c_area, 0.0))
        overflow = overflow | (here & (c_count > poly.shape[-2]))
        poly = jnp.where(here[..., None, None], clipped, poly)
        count = jnp.where(here, c_count, count)
        area = jnp.where(here, c_area, area)
    visible.append(area)
    if moments:
        return jnp.stack(visible, axis=-1), overflow, jnp.stack(fluxes, axis=-2), jnp.stack(rates, axis=-2)
    return jnp.stack(visible, axis=-1), overflow


def _weigh_levels(x: Array, level_eps: Sequence[Sequence[float]], power: int = 1, axis: int = -1) -> Array:
    """Sums per-level values weighted by the levels' contrasts, per E component.

    Args:
        x: per-level values, levels along ``axis`` (K).
        level_eps: (C, K) static contrast of each level per E component.
        power: power of the contrasts.
        axis: the level axis of x; it becomes the component axis (C).

    Returns:
        ``sum_j level_eps[c, j] ** power * x_j`` per component c.
    """
    x = jnp.moveaxis(x, axis, -1)
    out = []
    for row in level_eps:
        terms = [float(e) ** power * x[..., j] for j, e in enumerate(row) if e != 0]
        out.append(sum(terms[1:], terms[0]) if terms else jnp.zeros_like(x[..., 0]))
    return jnp.moveaxis(jnp.stack(out, axis=-1), -1, axis)


@timed("normal_tensor")
def _normal_tensor(
    length: Array,
    bp: Array,
    flux: Array,
    mean: Array,
    square: Array,
    rate: Array,
    cell: float,
    kmass: float,
    slopes: Array | None = None,
    offdiag: bool = False,
    graded: bool = False,
) -> Array:
    """Squared-normal tensor of the voxels of one layer from their own model.

    The eps variance of a voxel (contrasts normalized by the eps range) splits
    exactly over its z sub-intervals s into a lateral part, ``sum_s len_s
    Var_s``, and a vertical one, the variance along z of the sub-intervals'
    means. The lateral variance of sub-interval s goes along its kernel-weighted
    interface flux ``F_s = int w grad(eps)`` (whose z component carries the
    sidewall tilt), with weight ``1 - beta``, where beta ramps from 0 to 1 as
    ``|F_s| / Var_s`` (both in sides of the lateral square) falls below
    ``_KOTTKE_ETA``: flux that cancels inside the sub-interval (a strip of one
    level between two of another, corner slivers) is unexplained variance,
    returned separately. The vertical variance is taken after removing the
    drift of the means between sub-intervals that tapered sidewalls cause (so
    only the surfaces' jumps count) and goes along z, or, with ``slopes``,
    along the jump-weighted mean of the rough surfaces' normals ``(-g, 1)``.
    The drift is that of sidewalls placed at each sub-interval's middle: a
    level painted at the middle of its own part of the voxel (see
    :func:`_level_walls`) gets its lag to the sub-interval's middle added to
    the means first (the rate's second channel), so a tilted wall whose part
    another level's surface cuts keeps its exact normal.
    The ratios are not differentiated below ``_KOTTKE_TINY`` (a sub-interval's
    variance there is unexplained, a vertical part goes along z), so the
    gradients stay finite in float32 at any sliver of any contrast.

    Args:
        length: (..., S) sub-interval lengths (cells; summing to one, or to the
            layer's height with ``graded``).
        bp: (..., S + 1) sorted sub-interval ends (cells).
        flux: (..., S, C, 3) kernel-weighted contrast flux per sub-interval
            (see :func:`_moments`).
        mean: (..., S, C) mean contrast per sub-interval.
        square: (..., S, C) mean squared contrast per sub-interval.
        rate: (..., S, C, 2) cell times the rate of change of the mean with
            height, and cell times the mean's change from moving each wall
            from its painted height to the sub-interval's middle (see
            :func:`_paint_layer`).
        cell: cells per side of the lateral square.
        kmass: kernel integral over the area the flux comes from, in units in
            which ``flux / kmass`` is per side of the kernel's square.
        slopes: (..., S - 1, 2) lateral gradient (cells per cell) of the surface
            at each inner end, or None (flat surfaces).
        offdiag: also the off-diagonal entries (signed products of normal
            components) of the variance-weighted normal tensor.
        graded: the layer is ``bp[0]`` to ``bp[-1]`` high (a graded grid's, in
            frame units, see :class:`_Frame`): lengths and heights in the
            variances become shares of it, while the drift stays in frame
            units, the units of the rates.

    Returns:
        (..., C, 4): the variance along x, y, z (variance times squared normal
        components, summed), then the unexplained variance; with ``offdiag``
        (..., C, 7), followed by the xy, xz and yz entries (variance times the
        products of normal components, summed).
    """
    if graded:
        height = bp[..., -1:] - bp[..., :1]
        length = length / height
    v_lat = jnp.maximum(square - mean**2, 0.0)
    f = flux / kmass
    f2 = jnp.sum(f**2, axis=-1)
    low = f2 < (_KOTTKE_ETA * v_lat) ** 2
    # (|f| / (eta v))^2 from f / (eta v), whose derivative has (eta v)^-2 but no
    # higher power (f2 / lam2 would have lam2^-2, which float32 flushes to inf);
    # a tiny variance has no ratio: all unexplained (it adds at most its size)
    ramp = low & (v_lat > _KOTTKE_TINY)
    ratio = jnp.sum((f / (_KOTTKE_ETA * jnp.where(ramp, v_lat, 1.0))[..., None]) ** 2, axis=-1)
    beta = jnp.where(low, (1 - jnp.where(ramp, ratio, 0.0)) ** 2, 0.0)
    ok = f2 > 1e-12
    direction = jnp.where(ok[..., None], f**2 / jnp.where(ok, f2, 1.0)[..., None], 0.0)
    weight = length[..., None] * v_lat
    lateral = jnp.sum((weight * (1 - beta))[..., None] * direction, axis=-3)
    unexplained = jnp.sum(weight * beta, axis=-2)
    if offdiag:  # the signed products of the flux's components, normalized like the squares
        products = jnp.stack([f[..., a] * f[..., b] for a, b in _OFFDIAG_PAIRS], axis=-1)
        cross = jnp.where(ok[..., None], products / jnp.where(ok, f2, 1.0)[..., None], 0.0)
        lateral_off = jnp.sum((weight * (1 - beta))[..., None] * cross, axis=-3)

    # the means without the drift of tapered lines between the middles of
    # neighbouring sub-intervals: only the surfaces' jumps are left
    c = bp[..., 1:-1]
    mid = 0.5 * (bp[..., 1:] + bp[..., :-1])
    u = rate[..., 0] / cell
    step = u[..., :-1, :] * (c - mid[..., :-1])[..., None] + u[..., 1:, :] * (mid[..., 1:] - c)[..., None]
    drift = jnp.concatenate([jnp.zeros_like(step[..., :1, :]), jnp.cumsum(step, axis=-2)], axis=-2)
    flat = mean + rate[..., 1] / cell - drift
    centered = flat - jnp.sum(length[..., None] * flat, axis=-2, keepdims=True)
    v_z = jnp.sum(length[..., None] * centered**2, axis=-2)
    vertical_off = jnp.zeros((*v_z.shape, 3), dtype=v_z.dtype)
    if slopes is None:
        vertical = v_z[..., None] * jnp.asarray([0.0, 0.0, 1.0], dtype=v_z.dtype)
    else:
        # each surface's normal (-g, 1), weighted by its squared jump and by
        # 4 t (1 - t) at its height t in the voxel
        t = c - bp[..., :1]
        if graded:
            t = t / height
        wj = (flat[..., 1:, :] - flat[..., :-1, :]) ** 2 * (4 * t * (1 - t))[..., None]  # (..., S - 1, C)
        g2 = slopes**2
        dirs = jnp.concatenate([g2, jnp.ones_like(g2[..., :1])], axis=-1) / (1 + jnp.sum(g2, axis=-1))[..., None]
        total = jnp.sum(wj, axis=-2)
        some = total > _KOTTKE_TINY  # (v_z is of the order of total: below, z is as good)
        tilt = jnp.sum(wj[..., None] * dirs[..., None, :], axis=-3) / jnp.where(some, total, 1.0)[..., None]
        tilt = jnp.where(some[..., None], tilt, jnp.asarray([0.0, 0.0, 1.0], dtype=tilt.dtype))
        vertical = v_z[..., None] * tilt
        if offdiag:  # the normal (-g, 1) / |(-g, 1)|: products g_x g_y, -g_x, -g_y over 1 + |g|^2
            gx, gy = slopes[..., 0], slopes[..., 1]
            pair_dirs = jnp.stack([gx * gy, -gx, -gy], axis=-1) / (1 + jnp.sum(g2, axis=-1))[..., None]
            tilt_off = jnp.sum(wj[..., None] * pair_dirs[..., None, :], axis=-3) / jnp.where(some, total, 1.0)[..., None]
            vertical_off = v_z[..., None] * jnp.where(some[..., None], tilt_off, 0.0)
    out = [lateral + vertical, unexplained[..., None]]
    if offdiag:
        out.append(lateral_off + vertical_off)
    return jnp.concatenate(out, axis=-1)


def _kernel_mass(kernel: tuple[float, float, float]) -> float:
    """Integral of the flux kernel (see :func:`_moments`) over the unit lateral square."""
    ax, ay, beta = kernel
    return (1 - 4 * (ax**2 + beta**2 / 12)) * (1 - 4 * (ay**2 + beta**2 / 12))


def _level_walls(
    split: tuple[Array, Array, Array, Array],
    lateral: Callable[[Array], Array],
    num_levels: int,
    knots: Array | None = None,
    with_slope: bool = False,
) -> tuple[Array, Array | None, Array]:
    """Each level's sidewall offset over its own part of a voxel layer, and its z slope.

    Level j fills [lo_j, hi_j] of the layer (its surfaces clamped to it). Its
    offset there is the exact mean of ``lateral`` over that part: the offset is
    linear between the level's ``knots`` (its z_to_bias knots), so the mean is
    the length-weighted midpoint values of the pieces between them. Only the
    level's own surfaces and knots enter: another level's surface cutting the
    part changes nothing. A straight edge (summed along its length) then gets
    its exact volume, also with z_to_bias. Parts shorter than ``_PART_TINY``
    take the offset at their middle.

    Args:
        split: the layer's sub-intervals (:func:`_split_layer`).
        lateral: maps heights (...,) to the (..., K) offsets of the levels (see
            :func:`_composite`).
        num_levels: K.
        knots: (K, M) heights (cells, the frame of the surfaces) where a
            level's offset may bend, or None: every offset is linear in z.
        with_slope: also return the mean z slope of each offset.

    Returns:
        (..., K) mean offsets; with ``with_slope``, (..., K) mean slopes
        (offset per cell of height), else None; and (..., K) the middles of
        the parts (cells).
    """
    perm, bp = split[:2]
    points = jnp.take_along_axis(bp, jnp.argsort(perm, axis=-1), axis=-1)  # the layer's ends, lows, highs
    lo = points[..., 2 : 2 + num_levels]
    hi = points[..., 2 + num_levels :]
    if knots is None:
        pieces = 0.5 * (lo + hi)[..., None]  # (..., K, 1)
        weights = jnp.ones_like(pieces)
    else:
        bends = jnp.clip(jnp.asarray(knots, dtype=lo.dtype), lo[..., None], hi[..., None])  # (..., K, M)
        ends = jnp.concatenate([lo[..., None], bends, hi[..., None]], axis=-1)
        ends = jnp.take_along_axis(ends, jnp.argsort(jax.lax.stop_gradient(ends), axis=-1), axis=-1)
        pieces = 0.5 * (ends[..., 1:] + ends[..., :-1])  # (..., K, M + 1) middles of the linear pieces
        length = hi - lo
        long = length > _PART_TINY
        weights = jnp.where(
            long[..., None],
            (ends[..., 1:] - ends[..., :-1]) / jnp.where(long, length, 1.0)[..., None],
            # a tiny part: its middle (the pieces all lie within _PART_TINY of it)
            jnp.full_like(pieces, 1.0 / pieces.shape[-1]),
        )
        pieces = jnp.where(long[..., None], pieces, 0.5 * (lo + hi)[..., None])

    def own(z: Array) -> Array:
        """(..., K, P) offset of each level at its own heights z (..., K, P)."""
        # lateral sees one axis of heights per column, as for the sub-intervals
        every = lateral(z.reshape(*z.shape[:-2], -1)).reshape(*z.shape, num_levels)  # (..., K, P, K)
        return jnp.moveaxis(jnp.diagonal(every, axis1=-3, axis2=-1), -1, -2)

    middle = 0.5 * (lo + hi)
    if not with_slope:
        return jnp.sum(weights * own(pieces), axis=-1), None, middle
    value, dvalue = jax.jvp(own, (pieces,), (jnp.ones_like(pieces),))
    return jnp.sum(weights * value, axis=-1), jnp.sum(weights * dvalue, axis=-1), middle


def _first_moment(poly: Array, count: Array) -> Array:
    """First moments ``int (x, y) dA`` (..., 2) of counter-clockwise polygons (shoelace), the first ``count`` vertices valid."""
    i = jnp.arange(poly.shape[-2])
    q = jnp.take_along_axis(poly, jnp.where(i + 1 < count[..., None], i + 1, 0)[..., None], axis=-2)
    cross = jnp.where(i < count[..., None], poly[..., 0] * q[..., 1] - q[..., 0] * poly[..., 1], 0.0)
    return jnp.stack([jnp.sum(cross * (poly[..., a] + q[..., a]), axis=-1) for a in (0, 1)], axis=-1) / 6.0


def _paint_moments(
    poly: Array, count: Array, n: Array, o: Array, present: Array, is_fill: Sequence[bool], half: float
) -> tuple[Array, Array, Array]:
    """Fractions and first moments of the levels painted in priority order into convex cells of a square.

    The painting of :func:`_paint_levels` (without attributes: the same clips, so the same fractions), with
    each part's first moment ``int p dA`` as well.

    Args:
        poly: (..., V, 2) cells (convex, counter-clockwise) in the square's frame, the first ``count``
            vertices valid; V must hold the vertices the clips add.
        count: (...,) number of valid vertices.
        n: (..., K, 2) unit normals of the levels' half-planes ``n . p + o < 0``.
        o: (..., K) their offsets (sides of the square).
        present: (..., K) whether each level is there.
        is_fill: (K,) static, whether a level fills the whole plane.
        half: half side of the square.

    Returns:
        visible: (..., K + 1) fractions of the square: the K levels, then the background.
        first: (..., K + 1, 2) their first moments about the square's center, over the square's area.
        overflow: (...,) a clip produced more vertices than V.
    """
    full_area = _cell_area(half)
    area = _area(poly, count) / full_area
    first = _first_moment(poly, count) / full_area
    overflow = jnp.zeros(count.shape, dtype=bool)
    visible, moments = [], []
    for j, fill in enumerate(is_fill):
        here = present[..., j]
        if fill:
            visible.append(jnp.where(here, area, 0.0))
            moments.append(jnp.where(here[..., None], first, 0.0))
            area = jnp.where(here, 0.0, area)
            first = jnp.where(here[..., None], 0.0, first)
            count = jnp.where(here, 0, count)
            continue
        s = jnp.clip(o[..., j], -_cell_reach(half), _cell_reach(half))
        clipped, c_count = _clip(poly, count, n[..., j, :], s)
        c_area = _area(clipped, c_count) / full_area  # the part outside level j
        c_first = _first_moment(clipped, c_count) / full_area
        visible.append(jnp.where(here, area - c_area, 0.0))
        moments.append(jnp.where(here[..., None], first - c_first, 0.0))
        overflow = overflow | (here & (c_count > poly.shape[-2]))
        poly = jnp.where(here[..., None, None], clipped, poly)
        count = jnp.where(here, c_count, count)
        area = jnp.where(here, c_area, area)
        first = jnp.where(here[..., None], c_first, first)
    visible.append(area)
    moments.append(first)
    return jnp.stack(visible, axis=-1), jnp.stack(moments, axis=-2), overflow


def _line_range(p0: Array, d: Array, center: Array, half: Any) -> tuple[Array, Array]:
    """Parameter range (lo, hi) of the points ``p0 + t d`` inside the cell ``|p_a - center_a| <= half_a``.

    Empty where lo > hi. ``center`` is (..., 2) or (2,); ``half`` a square's half side or a rectangle's (hx, hy).
    """
    lo = jnp.full(p0.shape[:-1], -_T_FAR, dtype=p0.dtype)
    hi = jnp.full(p0.shape[:-1], _T_FAR, dtype=p0.dtype)
    for a in (0, 1):
        ha = _half_sides(half)[a]
        da = d[..., a]
        ok = jnp.abs(da) > _PARALLEL
        inv = 1.0 / jnp.where(ok, da, 1.0)
        t1 = (center[..., a] - ha - p0[..., a]) * inv
        t2 = (center[..., a] + ha - p0[..., a]) * inv
        inside = jnp.abs(jax.lax.stop_gradient(p0[..., a] - center[..., a])) <= ha
        lo = jnp.where(ok, jnp.maximum(lo, jnp.minimum(t1, t2)), jnp.where(inside, lo, _T_FAR))
        hi = jnp.where(ok, jnp.minimum(hi, jnp.maximum(t1, t2)), jnp.where(inside, hi, -_T_FAR))
    return lo, hi


def _keep_side(lo: Array, hi: Array, a: Array, c: Array, tie: Array) -> tuple[Array, Array]:
    """The part of [lo, hi] where ``c + a t >= 0`` (a line's value along another one).

    ``tie`` is added to c where the lines are parallel only (which side a coincident line is on).
    """
    ok = jnp.abs(a) > _PARALLEL
    tau = -c / jnp.where(ok, a, 1.0)
    lo = jnp.where(ok & (a > 0), jnp.maximum(lo, tau), lo)
    hi = jnp.where(ok & (a < 0), jnp.minimum(hi, tau), hi)
    return lo, jnp.where(~ok & (c + tie < 0), -_T_FAR, hi)


def _take_side(lo: Array, hi: Array, a: Array, c: Array, tie: Array) -> tuple[Array, Array, Array]:
    """Splits [lo, hi] at ``c + a t = 0``: the length where it is < 0, and the interval where it is >= 0.

    ``tie`` is added to c where the lines are parallel only.
    """
    ok = jnp.abs(a) > _PARALLEL
    tau = -c / jnp.where(ok, a, 1.0)
    up = ok & (a > 0)  # < 0 below tau
    down = ok & (a < 0)  # < 0 above tau
    inside = c + tie < 0  # parallel: all or nothing
    taken = jnp.where(
        up,
        jnp.minimum(hi, tau) - lo,
        jnp.where(down, hi - jnp.maximum(lo, tau), jnp.where(inside, hi - lo, 0.0)),
    )
    rlo = jnp.where(up, jnp.maximum(lo, tau), lo)
    rhi = jnp.where(down, jnp.minimum(hi, tau), jnp.where(~ok & inside, -_T_FAR, hi))
    return jnp.maximum(taken, 0.0), rlo, rhi


def _take_interval(lo: Array, hi: Array, a: Array, c: Array, tie: Array) -> tuple[Array, Array, Array, Array]:
    """Splits [lo, hi] at ``c + a t = 0``: the interval where it is < 0, then the one where it is >= 0.

    ``tie`` is added to c where the lines are parallel only. Empty intervals have lo > hi.
    """
    ok = jnp.abs(a) > _PARALLEL
    tau = -c / jnp.where(ok, a, 1.0)
    up = ok & (a > 0)  # < 0 below tau
    down = ok & (a < 0)  # < 0 above tau
    inside = c + tie < 0  # parallel: all or nothing
    tlo = jnp.where(down, jnp.maximum(lo, tau), lo)
    thi = jnp.where(up, jnp.minimum(hi, tau), jnp.where(down | inside, hi, -_T_FAR))
    rlo = jnp.where(up, jnp.maximum(lo, tau), lo)
    rhi = jnp.where(down, jnp.minimum(hi, tau), jnp.where(~ok & inside, -_T_FAR, hi))
    return tlo, thi, rlo, rhi


def _line_partition(
    n: Array, o: Array, present: Array, is_fill: Sequence[bool], p0: Array, d: Array, lo: Array, hi: Array,
    side: Array,
) -> list[tuple[Array, Array]]:
    """Priority painting of the levels' half-planes along one segment of a line, on one side of it.

    Args:
        n: (..., K, 2) unit normals of the half-planes ``n . p + o < 0``.
        o: (..., K) offsets.
        present: (..., K) whether each level is there.
        is_fill: (K,) static.
        p0: (..., 2) a point of the line.
        d: (..., 2) its unit direction.
        lo, hi: (...,) the segment ``p0 + t d``, t in [lo, hi].
        side: (..., 2) the side looked at: the line's normal (or its negative), for coincident lines.

    Returns:
        (lo, hi) per level and then the background: the interval (one per level, possibly empty) it holds.
    """
    out = []
    for k, fill in enumerate(is_fill):
        here = present[..., k]
        if fill:
            out.append((lo, jnp.where(here, hi, -_T_FAR)))
            hi = jnp.where(here, -_T_FAR, hi)
            continue
        nk = n[..., k, :]
        a = jnp.sum(nk * d, axis=-1)
        c = jnp.sum(nk * p0, axis=-1) + o[..., k]
        tlo, thi, rlo, rhi = _take_interval(lo, hi, a, c, _ON_LINE * jnp.sum(nk * side, axis=-1))
        out.append((tlo, jnp.where(here, thi, -_T_FAR)))
        lo, hi = jnp.where(here, rlo, lo), jnp.where(here, rhi, hi)
    out.append((lo, hi))
    return out


def _line_faces(
    n: Array, o: Array, present: Array, is_fill: Sequence[bool], center: Array, half: float,
    bounds: Sequence[tuple[Array, Array]] = (),
) -> Iterator[tuple[int, int, Array]]:
    """The interfaces on the levels' lines in a square, with the parts on both sides.

    In the half-plane model, level j's visible part is bounded on its own line by the piece of that line
    that no earlier level covers (points just inside level j); just outside it, each later level takes, in
    priority order, the piece inside its own half-plane (a half-line: one interval at one end of what is
    left), and the background the rest. So every interface of the painting is found exactly once: the
    piece of line j between level j and level k > j (or the background), by 1D painting along the line.
    Coincident lines are told apart by looking ``_ON_LINE`` to each side.

    Args:
        n: (..., K, 2) unit normals of the half-planes ``n . p + o < 0``.
        o: (..., K) offsets (sides of the square; as painted, clipped to +-2 half).
        present: (..., K) whether each level is there.
        is_fill: (K,) static, whether a level fills the whole plane (fill levels have no line).
        center: (..., 2) or (2,) center of the square, in the frame of n and o.
        half: half side of the square.
        bounds: (normal (..., 2), offset (...,)) pairs: only the part of the square where every
            ``normal . p + offset >= 0`` (a cell of the corner model); a line on a bound is not followed.

    Yields:
        (j, k, length): level j's line, the level k (K: the background) on its other side, and the
        (...,) length of their interface in the square (sides).
    """
    num = len(is_fill)
    d = jnp.stack([-n[..., 1], n[..., 0]], axis=-1)  # along each line
    p0 = -o[..., None] * n  # each line's point nearest the origin
    for j, fill in enumerate(is_fill):
        if fill:
            continue
        dj, pj, nj = d[..., j, :], p0[..., j, :], n[..., j, :]
        lo, hi = _line_range(pj, dj, center, half)
        for nb, ob in bounds:
            lo, hi = _keep_side(
                lo, hi, jnp.sum(nb * dj, axis=-1), jnp.sum(nb * pj, axis=-1) + ob,
                -_ON_LINE * jnp.abs(jnp.sum(nb * nj, axis=-1)),
            )
        hi = jnp.where(present[..., j], hi, -_T_FAR)
        for i in range(j):  # not covered by an earlier level on either side of the line
            here = present[..., i]
            if is_fill[i]:
                hi = jnp.where(here, -_T_FAR, hi)
                continue
            ni = n[..., i, :]
            a = jnp.sum(ni * dj, axis=-1)
            c = jnp.sum(ni * pj, axis=-1) + o[..., i]
            # a crossing line covers both sides alike; a coincident one covers one side (the same half-plane:
            # level j is hidden; the complementary one: their seam is level i's face)
            klo, khi = _keep_side(lo, hi, a, c, -_ON_LINE * jnp.abs(jnp.sum(ni * nj, axis=-1)))
            lo, hi = jnp.where(here, klo, lo), jnp.where(here, khi, hi)
        for k in range(j + 1, num):  # the later levels take pieces of the outside (points just outside)
            here = present[..., k]
            if is_fill[k]:
                taken = jnp.where(here, jnp.maximum(hi - lo, 0.0), 0.0)
                hi = jnp.where(here, -_T_FAR, hi)
            else:
                nk = n[..., k, :]
                a = jnp.sum(nk * dj, axis=-1)
                c = jnp.sum(nk * pj, axis=-1) + o[..., k]
                taken, rlo, rhi = _take_side(lo, hi, a, c, _ON_LINE * jnp.sum(nk * nj, axis=-1))
                taken = jnp.where(here, taken, 0.0)
                lo, hi = jnp.where(here, rlo, lo), jnp.where(here, rhi, hi)
            yield j, k, taken
        yield j, num, jnp.maximum(hi - lo, 0.0)


def _face_lengths(
    n: Array, o: Array, present: Array, is_fill: Sequence[bool], center: Array, half: float,
    bounds: Sequence[tuple[Array, Array]] = (),
) -> Array:
    """The faces of :func:`_line_faces` as one array (..., K, K + 1): [j, k] on level j's line, against level k
    (K: the background); zero where there is none (fill levels have no line)."""
    num = len(is_fill)
    found = {(j, k): v for j, k, v in _line_faces(n, o, present, is_fill, center, half, bounds)}
    zero = jnp.zeros(o.shape[:-1], dtype=o.dtype)
    return jnp.stack([jnp.stack([found.get((j, k), zero) for k in range(num + 1)], axis=-1) for j in range(num)], axis=-2)


def _face_weights(ell: Array, a: Array, b: Array, contrast2: np.ndarray, cell: float, side: float) -> Array:
    """``cell l psi c^2`` of faces of lengths ell between parts of fractions a and b (all broadcast).

    The faces' area per unit height (cells), their squared contrast (static, broadcast like ell's last axes)
    and the fade-in of thin parts ``psi = t / (t + _FACE_THICKNESS)``, t the thinner part's area (fraction
    times the square's area, ``side^2``) over the face's length.
    """
    thin = jnp.minimum(a, b) * side**2
    some = ell > 0
    t = jnp.where(some, thin / jnp.where(some, ell, 1.0), 0.0)
    return cell * ell * (t / (t + _FACE_THICKNESS)) * jnp.asarray(contrast2, dtype=ell.dtype)


def _dyad(v: Array) -> Array:
    """(..., 3) vectors -> (..., 6) xx, yy, zz, xy, xz, yz of v v^T."""
    return jnp.stack(
        [v[..., 0] ** 2, v[..., 1] ** 2, v[..., 2] ** 2, v[..., 0] * v[..., 1], v[..., 0] * v[..., 2], v[..., 1] * v[..., 2]],
        axis=-1,
    )


def _interface_parts(
    offsets: Array,
    normals: Array,
    is_fill: Sequence[bool],
    split: tuple[Array, Array, Array, Array],
    lateral: Callable[[Array], Array],
    half: float,
    knots: Array | None,
    cell: float,
    vectors: np.ndarray,
    contrast2: np.ndarray,
    point_slopes: Array | None = None,
    center: Array | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Half-plane painting of one voxel layer's columns per z sub-interval, with its interfaces.

    Every level is the half-plane of :func:`_paint_layer`, moved by its sidewall over its own part of the
    voxel (:func:`_level_walls`), painted in priority order; with it, each sub-interval's interfaces
    between parts of different materials (:func:`_line_faces`), weighted for :func:`_interface_layer`.

    Args:
        offsets: (..., K) signed distance of each level (sides of the square).
        normals: (..., K, 2) unit in-plane normals.
        is_fill: (K,) static.
        split: the layer's sub-intervals (:func:`_split_layer`), (..., S) each.
        lateral: the levels' sidewall offsets at heights (see :func:`_composite`).
        half: half side of the painted square.
        knots: (K, M) heights where the offsets bend, or None.
        cell: cells per side of the voxel's lateral sample square (face lengths are converted to cells).
        vectors: (K + 1, 6) static normalized material tensor of every level, then the background (see
            :func:`_interface_layer`).
        contrast2: (K + 1, K + 1) static squared contrasts of the levels' materials.
        point_slopes: (..., 2K + 2, 2) see :func:`_point_slopes`, or None.
        center: (..., 2) center of the painted square in the frame of the half-planes (a sub-square), or
            None: the frame's origin.

    On graded x or y ``half`` is the columns' own cells (:class:`_Cells`, broadcasting against the leading
    axes), ``lateral`` gives frame units (scaled here per column) and ``cell`` is unused (the cells' own).

    Returns:
        visible: (..., S, K + 1) fractions of the painted square per sub-interval (not length weighted).
        lateral_tensor: (..., S, 6) per sub-interval: sum over its faces of ``cell l w`` times the faces'
            normal dyad (see :func:`_interface_layer`), not length weighted.
        moments: (..., S, 6, 3) the sub-interval's material moments: zeroth and standardized first moments
            (``sqrt 12 int p dA / side^3``) of every level's part, times its material's ``vectors``.
        slopes: (..., S - 1, 2) the surfaces' slopes at the inner ends (zeros without ``point_slopes``).
        b: (..., K) each level's wall slope (cells per cell; its planes lean out of the plane by it).
    """
    perm, bp, length, present = split
    num = len(is_fill)
    per_column = isinstance(half, _Cells)
    if per_column:
        half = _cells_expand(half, 1)  # over the sub-intervals
    shift, slope, _ = _level_walls(split, lateral, num, knots, with_slope=True)
    if per_column:  # frame units -> sides of each column's cell
        shift = shift / half.cell[..., 0, None]
    o = offsets + shift  # (..., K): the same in every sub-interval
    if center is not None:
        o = o + jnp.sum(normals * center[..., None, :], axis=-1)  # the frame moved to the square's center
    o = jnp.clip(o, -_cell_reach(half), _cell_reach(half))
    # z slope of each level's plane (cells per cell; per column already frame units per frame unit)
    b = slope if per_column else cell * slope
    batch = length.shape
    n_s = jnp.broadcast_to(normals[..., None, :, :], (*batch, num, 2))
    o_s = jnp.broadcast_to(o[..., None, :], (*batch, num))
    num_vertices = 4 + sum(not f for f in is_fill)
    square = _cell_corners(half, bp.dtype)
    if per_column:
        square = jnp.concatenate([square, jnp.repeat(square[..., -1:, :], num_vertices - 4, axis=-2)], axis=-2)
    else:
        square = jnp.concatenate([square, jnp.repeat(square[-1:], num_vertices - 4, axis=0)])
    visible, first, _ = _paint_moments(
        jnp.broadcast_to(square, (*batch, num_vertices, 2)),
        jnp.full(batch, 4, dtype=jnp.int32),
        n_s,
        o_s,
        present,
        is_fill,
        half,
    )
    # the faces (:func:`_face_weights`): area weight cell * l (per unit height), sqrt(1 + b^2) for a tilted
    # wall, c^2 (none where the same tensor is on both sides) and the fade-in of thin parts
    tilt = jnp.sqrt(1.0 + b**2)
    dyads = _dyad(jnp.concatenate([normals, b[..., None]], axis=-1) / tilt[..., None])  # (..., K, 6)
    side = 2 * half if not isinstance(half, tuple) else math.sqrt(_cell_area(half))  # side**2: the area
    lengths = _face_lengths(n_s, o_s, present, is_fill, jnp.zeros(2, dtype=bp.dtype), half)  # (..., S, K, K + 1)
    cell_f = half.cell[..., None, None] if per_column else cell  # over the faces
    weight = _face_weights(lengths, visible[..., :-1, None], visible[..., None, :], contrast2[:-1], cell_f, side)
    weight = jnp.sum(weight, axis=-1) * tilt[..., None, :]  # (..., S, K): per line
    tensor = jnp.einsum("...sk,...kv->...sv", weight, dyads, precision=_HIGHEST)
    # the materials' zeroth and (standardized) first moments per sub-interval
    standard = _cell_standard(half, side, bp.dtype)
    if per_column:  # over the levels
        standard = standard[..., None, :]
    mom = jnp.concatenate([visible[..., None], standard * first], axis=-1)  # (..., S, K + 1, 3)
    moments = jnp.einsum("...lc,lv->...vc", mom, jnp.asarray(vectors, dtype=bp.dtype), precision=_HIGHEST)
    slopes = jnp.zeros((*batch[:-1], batch[-1] - 1, 2), dtype=bp.dtype)
    if point_slopes is not None:
        slopes = jnp.take_along_axis(point_slopes, perm[..., 1:-1, None], axis=-2)
    return visible, tensor, moments, slopes, b


def _vertical_tensor(length: Array, bp: Array, moments: Array, slopes: Array, graded: bool = False) -> Array:
    """The vertical part of a voxel's interface direction tensor (..., 6) from its sub-intervals.

    ``4 sum_s len_s |m_s - <m>|^2``, the variance along z of the sub-intervals' material moments (zeroth
    and standardized first moments, so materials that only swap sides count), times the normal of the
    surfaces: z, or the jump-weighted mix of the rough surfaces' normals ``(-g, 1)``. For one surface
    between two materials at height t in the voxel it is ``4 t (1 - t) c^2``, like a lateral face of the
    same contrast; sub-intervals of no length count nothing (tied surfaces).

    Args:
        length: (..., S) sub-interval lengths (summing to one).
        bp: (..., S + 1) their ends.
        moments: (..., S, 6, 3) material moments per sub-interval.
        slopes: (..., S - 1, 2) the surfaces' slopes at the inner ends.
        graded: the layer is ``bp[0]`` to ``bp[-1]`` high (a graded grid's, frame units): its lengths and
            heights become shares of it.

    Returns:
        (..., 6) the tensor.
    """
    if graded:
        height = bp[..., -1:] - bp[..., :1]
        length = length / height
    mean = jnp.sum(length[..., None, None] * moments, axis=-3)
    weight = 4.0 * jnp.sum(length * jnp.sum((moments - mean[..., None, :, :]) ** 2, axis=(-2, -1)), axis=-1)
    jump = jnp.sum((moments[..., 1:, :, :] - moments[..., :-1, :, :]) ** 2, axis=(-2, -1))  # (..., S - 1)
    t = bp[..., 1:-1] - bp[..., :1]
    if graded:
        t = t / height
    wj = jump * 4 * t * (1 - t)
    normal = jnp.concatenate([-slopes, jnp.ones_like(slopes[..., :1])], axis=-1)
    normal = normal / jnp.sqrt(jnp.sum(normal**2, axis=-1, keepdims=True))
    total = jnp.sum(wj, axis=-1)
    some = total > _KOTTKE_TINY
    mix = jnp.sum(wj[..., None] * _dyad(normal), axis=-2) / jnp.where(some, total, 1.0)[..., None]
    up = jnp.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=length.dtype)
    return weight[..., None] * jnp.where(some[..., None], mix, up)


class _CornerPlan(NamedTuple):
    """Static literals of the levels for the corner model (see :func:`_literals`)."""

    #: per level (priority order): its literal tree, None for fill levels
    trees: tuple
    #: per level: (layer slot, a, b in cells) of each literal
    literals: tuple
    #: per layer slot: its position in ``StackDevice._slots()`` (source layers, then derived shapes)
    layers: tuple[int, ...]


class _LayerCorner(NamedTuple):
    """Two-edge local model of a source layer at the corner columns (see :func:`_layer_corner`).

    Lengths are in sides of the lateral square, relative to each column's
    sample point; the arrays are (N, ...), one row per column.
    """

    #: (N, 2, 3) half-planes ``n . p + lam < 0`` (outward n) of the two nearest
    #: edges; at a vertex the edge into it first
    planes: Array
    #: (N,) the layer is the intersection of the two half-planes, else their union
    op_and: Array
    #: (N,) the two edges meet at a vertex
    linked: Array
    #: (N,) that vertex is convex
    convex: Array
    #: (N, 2) that vertex
    vertex: Array
    #: (N,) distance to the second nearest edge (no gradient)
    d2s: Array
    #: (N,) radius within which the two edges are the whole boundary; -BIG
    #: where the model is invalid
    rpair: Array
    #: (N,) the layer's signed distance at the point
    sd: Array
    #: (N, 2) its unit gradient there
    normal: Array


class _CornerInputs(NamedTuple):
    """The corner columns of one sample grid (see :func:`_corner_fill`)."""

    #: (N,) flat column index (i * ny + j); nx * ny in unused slots
    cols: Array
    #: (N,) whether a slot holds a column
    taken: Array
    #: local model of every source layer the literals read, stacked (L, N, ...)
    layers: _LayerCorner
    #: per level (priority order): its literal tree, None for fill levels
    trees: tuple
    #: per level: (layer in ``layers``, a, b in cells) of each literal
    literals: tuple
    #: number of split lines per column
    lines: int
    #: cells -> sides of the lateral square (static); on graded x or y (N, 1) per column (see ``cells``)
    sc: Any
    #: (nx * ny,) the candidate columns that found no slot (they keep the
    #: half-planes), or None
    dropped: Array | None = None
    #: on graded x or y: the columns' own lateral cells, (N,) arrays (:class:`_Cells`); else None
    cells: "_Cells | None" = None


class _Slot(NamedTuple):
    """One literal of one level in the corner columns; arrays (N, S) per column and sub-interval."""

    #: the level (priority order)
    level: int
    #: the literal's source layer (index into the stacked models)
    layer: int
    #: offset of the literal's boundary from its layer's (sides of the square)
    off: Array
    #: the literal's signed distance at the sample point
    val: Array
    #: the square is on one side of the literal's boundary: the value is ``val < 0``
    const: Array
    #: one edge in reach: the half-plane at the sample point, exact there
    single: Array
    #: a pair is the intersection of its planes, else their union
    op_and: Array
    #: (N, S, 3) planes [n, offset]: the single plane or the pair's first, then the pair's second
    planes: tuple[Array, Array]
    #: the planes are used and cut the square
    live: tuple[Array, Array]
    #: a pair at a vertex that the offset rounds (convex grown, reflex shrunk)
    rounded: Array


def _layer_corner(
    v: Array,
    src: np.ndarray,
    dst: np.ndarray,
    succ: np.ndarray,
    pid: np.ndarray,
    ids: Array,
    dist: Array,
    c: Array,
    sd: Array,
    normal: Array,
    sc: float,
) -> _LayerCorner:
    """Local model of a source layer from its two nearest boundary edges.

    Two edges that meet at a vertex bound the intersection (convex vertex) or
    the union (reflex vertex) of their half-planes; two edges that do not meet
    bound a strip (intersection) or a gap (union) of one polygon, or belong to
    two polygons of the union (union). The model is the layer up to ``rpair``
    from the point: the third nearest edge and the far ends of the two edges.
    It is invalid for degenerate edges and self-intersections, and where it
    disagrees with the layer's sign at the point. The classification has no
    gradient; the half-planes, the vertex and the radius follow the vertices.

    Args:
        v: (E, 2) the layer's vertices (cells, after T-junction splitting).
        src: (E,) static start vertex of every edge (filled side on its left).
        dst: (E,) static end vertex of every edge.
        succ: (E,) static next boundary edge (-1 if none).
        pid: (E,) static polygon index of every edge.
        ids: (N, 3) the three nearest boundary edges (:func:`nearest_edges`).
        dist: (N, 3) their distances (cells).
        c: (N, 2) the sample points (cells).
        sd: (N,) the layer's signed distance at the points (cells).
        normal: (N, 2) its unit gradient.
        sc: cells -> sides of the lateral square; (N,) per column on graded x or y.

    Returns:
        The model, in sides of the lateral square relative to the points.
    """
    sg = jax.lax.stop_gradient
    src, dst, succ, pid = (jnp.asarray(x) for x in (src, dst, succ, pid))
    ids, dist = sg(ids), sg(dist)
    e1, e2, e3 = ids[:, 0], ids[:, 1], ids[:, 2]
    link12, link21 = succ[e1] == e2, succ[e2] == e1
    linked = link12 ^ link21  # both: a polygon of two edges
    ea = jnp.where(link12, e1, e2)  # at a vertex: the edge into it
    eb = jnp.where(link12, e2, e1)  # and the edge out of it

    def line(e: Array) -> tuple[Array, ...]:
        """Ends, unit direction, outward normal and scaled signed distance of the edges e."""
        p, q = v[src[e]], v[dst[e]]
        d = q - p
        l2 = jnp.sum(d * d, axis=-1)
        ok = sg(l2) > 1e-12
        u = d * jax.lax.rsqrt(jnp.where(ok, l2, 1.0))[:, None]
        n = jnp.stack([u[:, 1], -u[:, 0]], axis=-1)
        return p, q, u, n, sc * jnp.sum(n * (c - p), axis=-1), ok

    pa, qa, ua, na, lam_a, ok_a = line(ea)
    pb, qb, ub, nb, lam_b, ok_b = line(eb)
    convex = sg(_cross(ua, ub)) >= 0  # 180 deg (split T-junctions): coincident planes

    # two edges that do not meet: one polygon (a strip where the part of e1
    # nearest to the point is inside e2's half-plane, else a gap), or two
    s1, t1, s2, t2 = (sg(v[i[e]]) for i, e in ((src, e1), (dst, e1), (src, e2), (dst, e2)))
    d1, d2 = t1 - s1, t2 - s2
    f = jnp.sum((sg(c) - s1) * d1, axis=-1) / jnp.maximum(jnp.sum(d1 * d1, axis=-1), 1e-30)
    foot = s1 + jnp.clip(f, 0.0, 1.0)[:, None] * d1
    strip = jnp.sum(sg(jnp.where(link12[:, None], nb, na)) * (foot - s2), axis=-1) < 0
    same = pid[e1] == pid[e2]
    op_and = jnp.where(linked, convex, same & strip)
    crossing = (_cross(d1, s2 - s1) * _cross(d1, t2 - s1) < 0) & (_cross(d2, s1 - s2) * _cross(d2, t1 - s2) < 0)
    valid = ok_a & ok_b & (dist[:, 1] < jnp.inf) & ~(link12 & link21) & ~(~linked & same & crossing)
    # the model must give the layer's sign at the point
    la, lb, sdv = sg(lam_a), sg(lam_b), sg(sd)
    inside = jnp.where(op_and, (la < 0) & (lb < 0), (la < 0) | (lb < 0))
    valid = valid & ((inside == (sdv < 0)) | (jnp.abs(sdv) < 1e-4))

    def norm(x: Array) -> Array:
        """|x| of (N, 2) vectors, with a finite derivative at 0."""
        r2 = jnp.sum(x * x, axis=-1)
        ok = sg(r2) > 1e-24
        return jnp.where(ok, jnp.sqrt(jnp.where(ok, r2, 1.0)), 0.0)

    # the far ends of the two edges (a vertex they share is part of the model)
    far = jnp.minimum(norm(c - pa), norm(c - qb))
    far = jnp.where(linked, far, jnp.minimum(far, jnp.minimum(norm(c - qa), norm(c - pb))))
    p3, q3 = v[src[e3]], v[dst[e3]]
    d3 = q3 - p3
    l3 = jnp.sum(d3 * d3, axis=-1)
    long3 = sg(l3) > 1e-12  # a point-like edge: its start
    f3 = jnp.where(long3, jnp.sum((c - p3) * d3, axis=-1) / jnp.where(long3, l3, 1.0), 0.0)
    third = norm(c - p3 - jnp.clip(f3, 0.0, 1.0)[:, None] * d3)
    third = jnp.where(dist[:, 2] < jnp.inf, third, _BIG)
    return _LayerCorner(
        planes=jnp.stack([jnp.concatenate([na, lam_a[:, None]], -1), jnp.concatenate([nb, lam_b[:, None]], -1)], 1),
        op_and=op_and,
        linked=linked,
        convex=convex,
        vertex=(sc[:, None] if jnp.ndim(sc) else sc) * (qa - c),
        d2s=sc * jnp.minimum(dist[:, 1], _BIG),
        rpair=jnp.where(valid, sc * jnp.minimum(third, far) - 2 * _CORNER_TOL, -_BIG),
        sd=sc * sd,
        normal=normal,
    )


def _same_plane(p: Array, q: Array) -> Array:
    """+1 where half-planes (..., 3) are the same, -1 where complementary, else 0 (no gradient)."""
    p, q = jax.lax.stop_gradient(p), jax.lax.stop_gradient(q)
    same = jnp.max(jnp.abs(p - q), axis=-1) < _PLANE_TOL
    flipped = jnp.max(jnp.abs(p + q), axis=-1) < _PLANE_TOL
    return jnp.where(same, 1, jnp.where(flipped, -1, 0))


def _literal_value(slot: _Slot, first: Array, second: Array) -> Array:
    """A literal from whether points are inside its planes.

    Args:
        slot: the literal, (N, S) per column and sub-interval.
        first: (N, S, ...) whether the points are inside its first plane.
        second: (N, S, ...) the same for its second plane.

    Returns:
        (N, S, ...) whether the points are in the literal's region.
    """
    extra = (1,) * (first.ndim - slot.const.ndim)
    op_and, const, single = (x.reshape(x.shape + extra) for x in (slot.op_and, slot.const, slot.single))
    negative = (jax.lax.stop_gradient(slot.val) < 0).reshape(slot.val.shape + extra)
    return jnp.where(const, negative, jnp.where(single, first, jnp.where(op_and, first & second, first | second)))


def _assign(
    slots: Sequence[_Slot],
    trees: tuple,
    present: Array,
    is_fill: Sequence[bool],
    point: Array,
    flips: Array | None,
) -> Array:
    """First present level (priority order) whose literals' planes contain points (no gradient).

    Args:
        slots: the literals.
        trees: per level, its literal tree (None for fill levels).
        present: (N, S, K) whether each level is there.
        is_fill: (K,) static, whether a level fills the whole plane.
        point: (N, S, T, 2) T points per column and sub-interval.
        flips: (N, S, T, num slots) where to negate each literal (or None).

    Returns:
        (N, S, T) level index, K (the background) where no level contains the point.
    """
    sg = jax.lax.stop_gradient
    shape = point.shape[:-1]
    level = jnp.full(shape, len(is_fill), dtype=jnp.int32)
    done = jnp.zeros(shape, dtype=bool)
    for j, fill in enumerate(is_fill):
        hit = jnp.broadcast_to(present[..., j, None], shape)
        if not fill:
            values = []
            for i, slot in enumerate(slots):
                if slot.level != j:
                    continue
                first, second = (
                    jnp.sum(sg(p[..., None, :2]) * point, axis=-1) + sg(p[..., None, 2]) < 0 for p in slot.planes
                )
                value = _literal_value(slot, first, second)
                values.append(value ^ flips[..., i] if flips is not None else value)
            hit = hit & jnp.broadcast_to(_eval_bool(trees[j], values), shape)
        level = jnp.where(~done & hit, j, level)
        done = done | hit
    return level


def _round_tips(
    slots: Sequence[_Slot],
    corner: _CornerInputs,
    exact: Array,
    present: Array,
    is_fill: Sequence[bool],
    half: float,
    center: Array | None = None,
) -> tuple[Array, Array]:
    """Corrects the cells' mitred corners where the level set is round.

    A convex corner grown by g (or a reflex corner shrunk by g) is a circular
    arc of radius g around the vertex v, while the two offset half-planes meet
    at a mitre. They differ exactly in the tip ``Z = kite - disk``, with the kite
    ``v, v + g n_a, v + g (n_a + n_b) / (1 + n_a . n_b), v + g n_b`` (outward
    edge normals n_a, n_b; -g for a reflex corner): there the literal is the
    opposite of the mitred one. Where every other plane is on one side of the
    tip in the square, the mitred painting gave all of it to one level, and
    the literal negated gives it to another: the tip's area moves between the
    two. Literals of the same layer with the same offset share their tip.
    Tips of other literals at the same vertex (derived shapes of several
    levels sharing a crossing) all move when their kites lie in disjoint cones
    of their outward normals, and no other plane cuts them.

    All literals are processed together, along a slot axis T.

    Args:
        slots: the literals.
        corner: the corner columns.
        exact: (N, S, K + 1) the cells' (mitred) fractions per sub-interval.
        present: (N, S, K) whether each level is there.
        is_fill: (K,) static, whether a level fills the whole plane.
        half: half side of the painted square (per column: the cells, over the tips).
        center: (2,) center of the painted square in the lateral square's frame (a sub-square of it), or
            None: the lateral square; per column (..., 2) over the tips.

    Returns:
        exact: the fractions with the tips moved.
        gate: (N, S) False where the correction does not apply: tips of two
            vertices, tips of one vertex whose cones overlap, a tip cut by
            another plane, or a mitre sharper than ``_MITRE_MIN``.
    """
    sg = jax.lax.stop_gradient
    n, s = exact.shape[:2]
    num = len(slots)
    dtype = exact.dtype
    full_area = _cell_area(half)
    layer = np.array([slot.layer for slot in slots])

    def per_slot(x: Array) -> Array:
        """(L, N, ...) per layer -> (N, S, T, ...) per literal."""
        x = jnp.moveaxis(x[layer], 0, 1)[:, None]
        return jnp.broadcast_to(x, (n, s, num, *x.shape[3:]))

    m = jax.tree.map(per_slot, corner.layers)
    off = jnp.stack([slot.off for slot in slots], axis=-1)  # (N, S, T)
    rounded = jnp.stack([slot.rounded for slot in slots], axis=-1)
    na, nb = m.planes[..., 0, :2], m.planes[..., 1, :2]
    den = 1.0 + jnp.sum(na * nb, axis=-1)
    sharp = sg(den) <= _MITRE_MIN
    mitre = (na + nb) / jnp.where(sharp, 1.0, den)[..., None]
    g = jnp.abs(off)  # rounded literals have off != 0
    step = (jnp.where(m.convex, 1.0, -1.0) * g)[..., None]
    kite = jnp.stack([m.vertex, m.vertex + step * na, m.vertex + step * mitre, m.vertex + step * nb], axis=-2)
    count = jnp.full((n, s, num), 4, dtype=jnp.int32)
    ccw = sg(_area(kite, count)) >= 0
    kite = jnp.where(ccw[..., None, None], kite, kite[..., ::-1, :])
    poly = jnp.concatenate([kite, jnp.repeat(kite[..., -1:, :], 4, axis=-2)], axis=-2)
    for side in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # the kite in the cell
        h_side = _half_sides(half)[0 if side[0] else 1]
        if center is None:
            bound = h_side
        elif isinstance(half, _Cells):
            bound = h_side - (side[0] * center[..., 0] + side[1] * center[..., 1])
        else:
            bound = h_side - (side[0] * center[0] + side[1] * center[1])
        side = jnp.broadcast_to(jnp.asarray(side, dtype=dtype), (n, s, num, 2))
        poly, count = _clip(poly, count, side, jnp.broadcast_to(jnp.asarray(bound, dtype=dtype), (n, s, num)))
    zone = (_area(poly, count) - _disk_area(poly, count, m.vertex, g)) / full_area
    zone = jnp.where(rounded & ~sharp, jnp.maximum(zone, 0.0), 0.0)
    gate = ~jnp.any(rounded & sharp, axis=-1)

    # literals of one layer with one offset share their tip (same layer, so same
    # vertex); the first of them moves it
    tip = sg(zone) > _TIP_AREA
    member = (layer[:, None] == layer[None, :]) & (jnp.abs(sg(off)[..., :, None] - sg(off)[..., None, :]) < _PLANE_TOL)
    earlier = np.tril(np.ones((num, num), dtype=bool), -1)  # [t, u]: u before t
    first = tip & ~jnp.any(tip[..., None, :] & member & earlier, axis=-1)
    # several tips may move if they round one vertex (two derived shapes sharing a crossing) and lie in
    # disjoint cones of their outward normals (each kite lies in its own cone, so they never overlap)
    def cross(x: Array, y: Array) -> Array:
        return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

    ga, gb = sg(step * na), sg(step * nb)  # (N, S, T, 2) the generators of each tip's cone
    turn = cross(ga, gb) >= 0
    lo, hi = jnp.where(turn[..., None], ga, gb), jnp.where(turn[..., None], gb, ga)  # counterclockwise

    def strictly_in(v: Array, a: Array, b: Array) -> Array:
        """Whether the directions v [..., u, :] are strictly inside the cones (a, b) [..., t, :]."""
        v, a, b = v[..., None, :, :], a[..., :, None, :], b[..., :, None, :]
        eps = 1e-6 * jnp.sum(v * v, axis=-1)
        return (cross(a, v) > eps) & (cross(v, b) > eps)

    def unit(x: Array) -> Array:
        return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-30)

    same_cone = (jnp.sum(unit(lo)[..., :, None, :] * unit(lo)[..., None, :, :], axis=-1) > 1 - 1e-6) & (
        jnp.sum(unit(hi)[..., :, None, :] * unit(hi)[..., None, :, :], axis=-1) > 1 - 1e-6
    )
    overlap = strictly_in(lo, lo, hi) | strictly_in(hi, lo, hi) | same_cone
    overlap = overlap | jnp.swapaxes(overlap, -1, -2)
    vertex = sg(m.vertex)
    one_vertex = jnp.max(jnp.abs(vertex[..., :, None, :] - vertex[..., None, :, :]), axis=-1) < _PLANE_TOL
    others = ~np.eye(num, dtype=bool)
    clash = first[..., :, None] & first[..., None, :] & others & (~one_vertex | overlap)
    gate = gate & ~jnp.any(clash, axis=(-2, -1))

    # every other live plane must leave the tip whole
    valid = jnp.arange(poly.shape[-2]) < count[..., None]  # (N, S, T, 8)
    planes = sg(jnp.stack([p for slot in slots for p in slot.planes], axis=-2))  # (N, S, 2T, 3)
    live = jnp.stack([x for slot in slots for x in slot.live], axis=-1)  # (N, S, 2T)
    f = jnp.sum(planes[..., None, :, None, :2] * sg(poly)[..., None, :, :], axis=-1) + planes[..., None, :, None, 2]
    inside = valid[..., None, :]
    both = jnp.any(inside & (f > 1e-5), axis=-1) & jnp.any(inside & (f < -1e-5), axis=-1)  # (N, S, T, 2T)
    cut = jnp.any(live[..., None, :] & both & ~member[..., np.repeat(np.arange(num), 2)], axis=-1)
    gate = gate & ~jnp.any(first & cut, axis=-1)

    probe = jnp.sum(jnp.where(valid[..., None], sg(poly), 0.0), axis=-2) / jnp.maximum(count, 1)[..., None]
    mitred = _assign(slots, corner.trees, present, is_fill, probe, None)
    exact_level = _assign(slots, corner.trees, present, is_fill, probe, member)
    move = jnp.where(first, zone, 0.0)[..., None]
    num_out = len(is_fill) + 1
    moved = jax.nn.one_hot(exact_level, num_out, dtype=dtype) - jax.nn.one_hot(mitred, num_out, dtype=dtype)
    return exact + jnp.sum(move * moved, axis=-2), gate


@timed("corner_pass")
@partial(
    jax.jit,
    static_argnames=(
        "trees", "literals", "is_fill", "lines", "sc", "half", "level_eps", "kernel", "cell", "per_interval",
        "offdiag", "graded",
    ),
)
def _corner_pass(
    layers: _LayerCorner,
    taken: Array,
    dense: Array,
    length: Array,
    present: Array,
    shift: Array,
    moments: tuple[Array, ...] | None = None,
    planes: tuple[Array, Array] | None = None,
    *,
    center: Array | None = None,
    trees: tuple,
    literals: tuple,
    is_fill: tuple[bool, ...],
    lines: int,
    sc: float,
    half: float,
    level_eps: tuple[tuple[float, ...], ...] | None = None,
    kernel: tuple[float, float, float] = (0.0, 0.0, 1.0),
    cell: float = 1.0,
    per_interval: bool = False,
    offdiag: bool = False,
    graded: bool = False,
    cells: tuple[Array, Array, Array] | None = None,
) -> Array | tuple[Array, ...]:
    """The corner columns of every voxel layer, painted with :func:`_corner_fill`.

    A separate jitted pass with a static structure (not part of the dense scan),
    so its compiled code is shared by every call with the same shapes: sample
    grids, Yee positions, devices of the same stack. Voxel layers where no
    polygon level is in a corner column keep the half-plane result (exact
    there). Each voxel layer is rematerialized in the backward pass, so memory
    stays per layer. With ``per_interval``, the moments per sub-interval are
    returned instead of the tensor, for :func:`_composite_samples` to add up
    with the other sub-samples of the voxels.

    Args:
        layers: local model of every source layer the literals read, (L, N, ...).
        taken: (N,) whether a slot holds a column.
        dense: (nz, N, S, K + 1) half-plane fractions per sub-interval (not
            length weighted).
        length: (nz, N, S) sub-interval lengths.
        present: (nz, N, S, K) whether each level is in each sub-interval.
        shift: (nz, N, S, K, 2) lateral offset of each level at the two Gauss
            heights of its part of the voxel (sides of the square).
        moments: with ``level_eps``, the columns' rates of each level (nz,
            N, S, K, 2), the half-plane contrast flux (nz, N, S, C, 3) and rates
            (nz, N, S, C, 2) and, without ``per_interval``, the sub-interval ends
            (nz, N, S + 1) and, with rough surfaces, their slopes (nz, N, S - 1,
            2); see :func:`_paint_layer`.
        planes: (N, K) offsets and (N, K, 2) normals of the columns'
            half-planes (see :func:`_corner_fill`), or None.
        center: (2,) the kernel's (ax, ay), traced (so that the sub-samples of
            :func:`_composite_samples` share one compiled pass), or None: those
            of ``kernel``.
        trees: per level, its literal tree (None for fill levels).
        literals: per level, (layer in ``layers``, a, b in cells) of each literal.
        is_fill: (K,) whether a level fills the whole plane.
        lines: number of split lines.
        sc: cells -> sides of the lateral square.
        half: half side of the lateral square.
        level_eps: (C, K) contrasts of the levels (see :func:`_composite`), or
            None: fractions only.
        kernel: the flux kernel (see :func:`_composite`).
        cell: cells per side of the lateral square.
        per_interval: return the corner model per sub-interval, unblended,
            with each sample's certificate for its voxel (see
            :func:`_composite_samples`).
        offdiag: the tensor also has the off-diagonal entries (see
            :func:`_normal_tensor`).
        graded: the layers are a graded grid's (see :func:`_normal_tensor`);
            the fractions then sum to each layer's height (frame units).
        cells: on graded x or y, the columns' own lateral cells (hx, hy, cell), (N,) each (traced: one compiled
            pass for every sample grid of a shape; see :class:`_Cells`); ``half`` is then their static reach
            and ``sc`` and ``cell`` are unused. None: the static lateral cell.

    Returns:
        (nz, N, K + 1) volume fractions of the levels, then the background;
        with ``level_eps`` also the (nz, N, C, 4) tensor of
        :func:`_normal_tensor`. With ``per_interval`` instead the corner
        model's fractions (nz, N, S, K + 1; not length weighted), the
        certificate for the voxel (nz, N, S; ``raw`` of :func:`_corner_fill`)
        and, with ``level_eps``, its contrast flux (nz, N, S, C, 3) and rates
        (nz, N, S, C, 2) per sub-interval.
    """
    if cells is not None:  # each column's own cell, over the sub-intervals (and the components for the tensor)
        half = _Cells(*cells, reach=half)
        sc, cell = (1.0 / cells[2])[:, None], cells[2][:, None, None]
    corner = _CornerInputs(  # the columns are already gathered: no indices
        cols=None, taken=taken, layers=layers, trees=trees, literals=literals, lines=lines, sc=sc
    )
    polygons = np.logical_not(is_fill)
    with_n = level_eps is not None
    if center is not None:
        kernel = (center[0], center[1], kernel[2])
    if with_n:
        kmass = _kernel_mass(kernel)

    @jax.checkpoint
    def repaint(inputs: tuple[Array, ...]) -> Array:
        """(N, K + 1) fractions of one voxel layer from the corner model (and its moments)."""
        if with_n:
            return _corner_fill(
                corner, *inputs[:4], is_fill, half, (*inputs[4:7], level_eps, kernel), per_interval, planes
            )
        return _corner_fill(corner, *inputs[:4], is_fill, half, raw=per_interval, planes=planes)

    def half_planes(inputs: tuple[Array, ...]) -> Array:
        """(N, K + 1) fractions of one voxel layer from the half-planes (and their moments)."""
        dense, length = inputs[:2]
        if per_interval:  # exact: E = T, certified
            return (dense, jnp.ones(length.shape, dtype=length.dtype), *inputs[5:7])
        fill = jnp.stack([jnp.sum(length * dense[..., j], axis=-1) for j in range(dense.shape[-1])], axis=-1)
        return (fill, dense, *inputs[5:7]) if with_n else fill

    def one_slice(inputs: tuple[Array, ...]) -> Array:
        """(N, K + 1) fractions of one voxel layer: the corner model where a polygon level is (and the tensor)."""
        busy = jnp.any(inputs[2][..., polygons] & taken[:, None, None])
        out = jax.lax.cond(busy, repaint, half_planes, inputs)
        if not with_n or per_interval:
            return out
        fill, visible, flux, rate = out
        mean = _weigh_levels(visible[..., :-1], level_eps)
        square = _weigh_levels(visible[..., :-1], level_eps, power=2)
        slopes = inputs[8] if len(inputs) > 8 else None
        return fill, _normal_tensor(
            inputs[1], inputs[7], flux, mean, square, rate, cell, kmass, slopes, offdiag, graded=graded
        )

    return jax.lax.map(one_slice, (dense, length, present, shift, *(moments or ())))


def _corner_fill(
    corner: _CornerInputs,
    dense: Array,
    length: Array,
    present: Array,
    shift: Array,
    is_fill: Sequence[bool],
    half: float,
    moments: tuple | None = None,
    raw: bool = False,
    planes: tuple[Array, Array] | None = None,
) -> Array | tuple[Array, ...]:
    """Paints the corner columns of one voxel layer with the exact local geometry.

    In each corner column and z sub-interval, every literal of every present
    level (:func:`_literals`) is constant (its boundary is farther than the
    square's half-diagonal: exact, since distances are 1-Lipschitz), single
    (one edge within reach: the half-plane at the sample point, exact there)
    or a pair (the two nearest edges of :func:`_layer_corner`, exact within
    ``rpair``). The distinct half-planes of the levels become up to
    ``corner.lines`` split lines plus one leftover plane per level; in each of
    the 2^M cells the lines cut, every level is then one half-plane, full or
    empty, and the cells are painted like the half-plane path. Grown convex
    and shrunk reflex corners are round (see :func:`_round_tips`).

    Every level is painted with its offsets at the two Gauss-Legendre heights
    of its own part of the voxel (``shift``, :func:`_gauss_heights`), and E is
    the mean of the two paintings: near a corner of a sloped wall the area is
    quadratic in z, so E is the exact volume of the 3D shape, as long as no
    vertex or edge crosses a corner or side of the square within the part and
    no z_to_bias knot is inside it. Where an edge is straight in the square,
    the two heights alone differ from the ordinary columns' T at the mean
    offset (by up to ``tan(angle) / (4 sqrt 3)`` of the voxel where the wall
    crosses a side of the square within the voxel), which would show as a
    seam along the edge. So with ``planes`` the columns' half-planes are also
    painted at the two heights (T_g), and the seam ``T - T_g`` is added to E
    level by level in priority order, on the union of the levels painted up
    to each (its fraction c_j, the sum of levels 0..j, which the later levels
    do not change): c_j becomes ``c_j(E) + v_j (c_j(T) - c_j(T_g))``, with v_j
    1 where ``c_j(E) = c_j(T_g)`` (that union's geometry in the square is the
    half-planes: a straight edge, a junction, a collinear vertex), so such an
    edge gets T exactly, and v_j ramping smoothly to 0 (``(1 - gap^2 /
    width^2)^2``, differentiated) as the gap ``|c_j(E) - c_j(T_g)|`` reaches
    a width of ``_STRAIGHT_WIDTH`` times the seam ``|c_j(T) - c_j(T_g)|`` (at
    least ``_STRAIGHT_TOL``). The unions are kept nested and in [0, 1] (c_j
    is clipped to [c_(j-1), 1]: a level keeps what the levels before it
    hold), and level j gets ``c_j - c_(j-1)``, the background the rest. So as
    a vertex bends off collinear or a corner enters the square, the fill
    moves from T to E at a rate of the order of the gap's own (no jump, and
    the gradient sees it), and only where the seam is nonzero (sloped walls)
    does v_j reach into real corners, by at most the seam (5e-4 of a voxel at
    the corners of ``test_offset_corners_get_exact_areas``). Where v_j = 0, E
    stays the two heights' mean, since T's half-plane is not the geometry
    there. A level's fraction thus depends on its own and the earlier levels'
    geometry only: a later level's corner or surface in the square changes
    nothing of it. The moments are mixed as ``v T + (1 - v) E``, with v the
    least v_j. Levels whose parts differ are paired height for height (lower
    with lower), as the half-planes pair the levels' mean offsets; another
    level's surface cutting a level's part changes nothing.

    The exact fractions E are blended with the half-plane ones T by a weight w
    that is 1 where every literal's model is certified (at both heights),
    ramps to 0 as a third feature comes within reach (and with the mode
    switches, so the fill stays continuous), and is 0 where a model is
    invalid, the planes need more lines or a tip cannot be corrected. Since E
    is exact wherever w > 0, the result ``w E + (1 - w) T`` is never farther
    from the level-set model than T, per sub-interval. With several
    sub-squares per voxel, :func:`_composite_samples` blends them all by the
    voxel's certificate instead (``raw``), so that this holds for their
    average. Limit: a hole's cut line or a T-junction that splits an outer
    edge near a corner makes the short piece a third feature there (w = 0 at
    that corner: the half-plane result).

    With ``moments``, the clips carry the planes as :func:`_composite` does;
    the edges that split cells carry their line's plane too, so where one
    material is on both sides their fluxes cancel and the cells' fluxes add up
    to the interfaces' (the round tips' area moves no flux). The flux and rate
    are blended like the fractions.

    Args:
        corner: the corner columns.
        dense: (N, S, K + 1) the columns' half-plane fractions per sub-interval
            (not length weighted).
        length: (N, S) sub-interval lengths.
        present: (N, S, K) whether each level is in each sub-interval.
        shift: (N, S, K, 2) lateral offset of each level at the lower and the
            upper Gauss height of its part of the voxel (sides of the square).
        is_fill: (K,) static, whether a level fills the whole plane.
        half: half side of the lateral square.
        moments: for the normal tensor, the rates of each level (N, S, K, 2;
            its z slope in cells per cell, then the slope times its lag, see
            :func:`_paint_layer`), the half-plane contrast flux (N, S, C, 3)
            and rates (N, S, C, 2), the static (C, K) contrasts and the kernel
            (see :func:`_composite`), or None.
        raw: return E and the certificate for the voxel instead of the blend:
            ``max(w, v)``, with v 1 where every literal of a present level is
            constant, or all but one, which is a single edge, each with
            ``_VOXEL_RAMP`` to spare, ramping to 0 within that margin (so
            ``v > 0`` only where ``w > 0``, and 1 where a column stops being a
            corner column of a sample grid with sub-samples).
        planes: the columns' half-planes: (N, K) offsets (sides of the square,
            without ``shift``) and (N, K, 2) unit normals, as painted into
            ``dense``; or None: E is the two heights' mean alone.

    Returns:
        (N, K + 1) the columns' fractions of the levels, then the background;
        with ``moments`` also the blended fractions per sub-interval (N, S,
        K + 1), contrast flux (N, S, C, 3) and rates (N, S, C, 2). With
        ``raw`` instead E's fractions per sub-interval (N, S, K + 1), the
        certificate (N, S) and, with ``moments``, E's flux and rates.
    """
    with_n = moments is not None
    if with_n:
        slope, flux_t, rate_t, level_eps, kernel = moments
    n0 = length.shape[0]
    dtype = length.dtype
    # every level at the two Gauss heights of its part: the columns twice (the
    # second copy at the upper heights), painted together
    corner = corner._replace(
        taken=jnp.concatenate([corner.taken] * 2),
        layers=jax.tree.map(lambda x: jnp.concatenate([x, x], axis=1), corner.layers),
    )
    per_column = isinstance(half, _Cells)  # graded x or y: (N,) cells, (N, 1) sc
    if per_column:  # doubled with the columns, over the sub-intervals
        corner = corner._replace(sc=jnp.concatenate([corner.sc] * 2))
        half = _cells_expand(_cells_twice(half), 1)
    voxel_length = length
    if with_n:
        # the walls' rates there: the slope, and its lag to the sub-interval's
        # middle, from the middle of the level's part (slope[..., 1]) less the
        # height's distance from it (its part is where it is present)
        part = jnp.sum(jnp.where(present, length[..., None], 0.0), axis=-2)[:, None]  # (N, 1, K)
        b = slope[..., 0]
        slope = jnp.concatenate(
            [jnp.stack([b, slope[..., 1] + g * b * part / (2 * math.sqrt(3.0))], axis=-1) for g in (1.0, -1.0)]
        )
    length, present = (jnp.concatenate([x, x]) for x in (length, present))
    shift = jnp.concatenate([shift[..., 0], shift[..., 1]])
    n, s = length.shape
    full_area = _cell_area(half)
    gauss_t = None
    if planes is not None:
        # the columns' half-planes at the same two heights: where E equals
        # them, the edge is straight in the square and E becomes T
        po, pn = (jnp.concatenate([x, x]) for x in planes)  # (2N, K), (2N, K, 2)
        num_t = 4 + sum(not f for f in is_fill)
        square_t = _cell_corners(half, dtype)  # (4, 2); per column (2N, 1, 4, 2)
        if per_column:
            square_t = jnp.concatenate([square_t, jnp.repeat(square_t[..., -1:, :], num_t - 4, axis=-2)], axis=-2)
        else:
            square_t = jnp.concatenate([square_t, jnp.repeat(square_t[-1:], num_t - 4, axis=0)])
        gauss_t = _paint_levels(
            jnp.broadcast_to(square_t, (n, s, num_t, 2)),
            jnp.full((n, s), 4, dtype=jnp.int32),
            jnp.ones((n, s), dtype=dtype),
            jnp.broadcast_to(pn[:, None], (n, s, *pn.shape[1:])),
            po[:, None, :] + shift,
            present,
            is_fill,
            half,
        )

    cc = _corner_cells(corner, present, shift, is_fill, half, slope if with_n else None)
    slots, weight, voxel = cc.slots, cc.weight, cc.voxel
    poly, count, attr = _corner_polygons(cc, is_fill, half, with_n)
    num_cells = poly.shape[2]
    ok_clip = jnp.all(count <= poly.shape[-2], axis=-1)
    normals, offsets, slopes = cc.normals, cc.offsets, cc.slopes
    # per column: the cells over the cut cells too
    half_cells = _cells_expand(half, 1) if per_column else half
    painted = _paint_levels(
        poly,
        count,
        _area(poly, count) / full_area,
        jnp.stack(normals, axis=-2),
        jnp.stack(offsets, axis=-1),
        jnp.broadcast_to(present[:, :, None, :], (n, s, num_cells, len(is_fill))),
        is_fill,
        half_cells,
        *((attr, jnp.stack(slopes, axis=-2), kernel) if with_n else ()),
    )
    visible, overflow = painted[:2]
    exact = jnp.sum(visible, axis=2)  # (N, S, K + 1)
    # round joins only where some literal of this voxel layer is rounded
    exact, gate = jax.lax.cond(
        jnp.any(jnp.stack([slot.rounded for slot in slots])),
        lambda e: _round_tips(slots, corner, e, present, is_fill, half_cells),
        lambda e: (e, jnp.ones((n, s), dtype=bool)),
        exact,
    )
    ok = cc.ok_lines & ok_clip & ~jnp.any(overflow, axis=-1) & gate
    w = jnp.where(ok, weight, 0.0)
    if raw:  # the certificate for the voxel instead (> 0 only where w is)
        w = jnp.maximum(w, jnp.where(ok, voxel, 0.0))
    # the two heights: E is their mean, certified where both are
    def mean(x: Array) -> Array:
        """(2N, ...) -> (N, ...): the mean over the two heights."""
        return 0.5 * (x[:n0] + x[n0:])

    exact = mean(exact)  # (N, S, K + 1)
    w = jnp.minimum(w[:n0], w[n0:])  # (N, S)
    straight = None
    if gauss_t is not None:
        exact, straight = _straight_seams(exact, dense, mean(gauss_t[0]), is_fill)
    if with_n:
        flux = _weigh_levels(jnp.sum(painted[2], axis=2), level_eps, axis=-2)  # (2N, S, C, 3)
        rate = _weigh_levels(jnp.sum(painted[3], axis=2), level_eps, axis=-2)  # (2N, S, C, 2)
        flux, rate = mean(flux), mean(rate)
        if gauss_t is not None:
            # one mix for all levels (the moments are summed over them): the
            # least of the levels' v, so T's only where every union is straight
            mix = jnp.min(straight, axis=-1)[..., None, None]  # (N, S, 1, 1)
            flux = mix * flux_t + (1 - mix) * flux
            rate = mix * rate_t + (1 - mix) * rate
    if raw:
        return (exact, w, flux, rate) if with_n else (exact, w)
    return _blend(w, exact, dense, voxel_length, *((flux, flux_t, rate, rate_t) if with_n else ()))


def _mean_shift(raw: Array, lower: Array, target: Array, iterations: int | None = None) -> Array:
    """The shift s (..., 1) with ``mean(clip(raw + s, lower, 1), axis=-1) = target`` (..., 1).

    The mean is monotone in s: s is found by bisection (no derivative), then one Newton step from it gives
    s the implicit derivative (``-d(mean - target) / d mean / ds``, zero where every value is clipped).
    """
    steps = (52 if jnp.dtype(raw.dtype) == jnp.float64 else 24) if iterations is None else iterations
    r, low_b, t = (jax.lax.stop_gradient(x) for x in (raw, lower, target))

    def mean_at(s: Array, values: Array, bound: Array) -> Array:
        return jnp.mean(jnp.clip(values + s, bound, 1.0), axis=-1, keepdims=True)

    def step(_: int, interval: tuple[Array, Array]) -> tuple[Array, Array]:
        a, b = interval
        m = 0.5 * (a + b)
        short = mean_at(m, r, low_b) < t
        return jnp.where(short, m, a), jnp.where(short, b, m)

    start = (jnp.min(low_b - r, axis=-1, keepdims=True), jnp.max(1.0 - r, axis=-1, keepdims=True))
    a, b = jax.lax.fori_loop(0, steps, step, start)
    s = jax.lax.stop_gradient(0.5 * (a + b))
    free = ((raw + s > lower) & (raw + s < 1.0)).astype(raw.dtype)
    slope = jnp.mean(free, axis=-1, keepdims=True)
    miss = mean_at(s, raw, lower) - target
    return s - jnp.where(slope > 0, miss / jnp.where(slope > 0, slope, 1.0), 0.0)


def _straight_seams(
    exact: Array,
    dense: Array,
    gauss_t: Array,
    is_fill: Sequence[bool],
    given: Array | None = None,
    target: Array | None = None,
) -> tuple[Array, Array]:
    """E with the straight-edge seam T - T_g added level by level (see :func:`_corner_fill`).

    Level by level in priority order, on the union of the levels painted up to j (its fraction c_j: the
    later levels change nothing of it, so neither does their corner or surface in the square): where the
    corner model's c_j is the half-planes' at both heights (the union's geometry is straight in the
    square), c_j(E) + (c_j(T) - c_j(T_g)) is T's, the mean offset of the ordinary columns along the edge.
    That seam fades out smoothly (v_j = (1 - gap^2 / width^2)^2, differentiated) as c_j(E) leaves
    c_j(T_g), over a width of _STRAIGHT_WIDTH times the seam's size: the fill moves by at most v' |seam|
    <= 1.54 / _STRAIGHT_WIDTH times the gap's own rate, and a real corner (gap >= width) keeps E exactly.
    Vertical walls have no seam: E there is untouched.

    Args:
        exact: (N, S, K + 1) E, the mean of the two heights' corner model.
        dense: (N, S, K + 1) T, the half-planes at the mean offsets.
        gauss_t: (N, S, K + 1) T_g, the half-planes at the two heights (their mean).
        is_fill: (K,) static.
        given: (..., K) v to use (another square's, e.g. the voxel's for its sub-squares), or None.
        target: with ``given``, (..., 1, K) the unions c_j of the square the last axis of ``exact``'s
            batch tiles (the voxel's, for its sub-squares): every sub-square's c_j gets one shift before
            it is clipped, so that their mean is the voxel's exactly (clipped one by one, their own seams
            would not add up where some sub-squares clip and the voxel does not).

    Returns:
        The fractions (N, S, K + 1), and v (N, S, K) per level (1 where the union is straight).
    """
    zero = jnp.zeros_like(exact[..., 0])
    sums = (zero, zero, zero)  # c_j of E, T and T_g (in order, so an empty level adds exactly 0)
    below, moved = zero, zero  # c_(j-1) as painted, and what its seam added to it
    fractions, straight = [], []
    for j in range(len(is_fill)):
        sums = tuple(c + x[..., j] for c, x in zip(sums, (exact, dense, gauss_t), strict=True))
        gap, seam = sums[0] - sums[2], sums[1] - sums[2]
        if given is None:
            v = (1.0 - jnp.minimum(gap**2 / (_STRAIGHT_WIDTH**2 * seam**2 + _STRAIGHT_TOL**2), 1.0)) ** 2
        else:
            v = given[..., j]
        # the unions stay nested and in [0, 1]: c_j in [c_(j-1), 1], so a
        # level keeps what the levels before it hold (where clipped, the
        # gradient is the bound's; at a tie, the union's own)
        c = sums[0] + v * seam
        if target is not None:
            c = c + _mean_shift(c, below, target[..., j])
        c = jnp.where(c > 1.0, 1.0, c)
        c = jnp.where(c < below, below, c)
        # level j: E_j plus what the seam added to c_j less what it added to
        # c_(j-1) (E_j itself where neither has a seam)
        add = c - sums[0]
        fractions.append(exact[..., j] + add - moved)
        straight.append(v)
        below, moved = c, add
    return jnp.stack([*fractions, exact[..., -1] - moved], axis=-1), jnp.stack(straight, axis=-1)


class _CornerCells(NamedTuple):
    """The corner model of the corner columns of one voxel layer as cells (see :func:`_corner_fill`).

    Arrays are (2N, S, ...): the columns at the lower, then the upper Gauss heights, per sub-interval.
    """

    #: every literal (constant, single edge or pair)
    slots: list
    #: certificate of the literals' models (1 where all hold over the whole reach)
    weight: Array
    #: certificate for the voxel (see ``raw`` of :func:`_corner_fill`)
    voxel: Array
    #: the M split lines, (2N, S, 3) each: [n, offset] (``n . p + offset < 0`` inside)
    lines: list
    #: whether each split line is used (an unused one is far outside the square)
    used: list
    #: the split lines' rates (2N, S, R), or zeros
    lines_b: list
    #: every live plane found a split line or its level's leftover
    ok_lines: Array
    #: per level, its unit normal (2N, S, C, 2) in each of the C cells (full / empty levels: far offsets)
    normals: list
    #: per level, its offset (2N, S, C) in each cell
    offsets: list
    #: per level, its rates (2N, S, C, R) in each cell, or None
    slopes: list | None


def _corner_cells(
    corner: _CornerInputs, present: Array, shift: Array, is_fill: Sequence[bool], half: float, rates: Array | None
) -> _CornerCells:
    """The corner columns' literals, split lines and each level's half-plane in every cell.

    Args:
        corner: the corner columns, already doubled for the two Gauss heights.
        present: (2N, S, K) whether each level is in each sub-interval.
        shift: (2N, S, K) each level's lateral offset at the height (sides of the square).
        is_fill: (K,) static.
        half: half side of the lateral square.
        rates: (2N, S, K, R) each level's rates (its planes' z slope, ...), or None.

    Returns:
        The cells (see :class:`_CornerCells`).
    """
    sg = jax.lax.stop_gradient
    n, s = present.shape[:2]
    dtype = shift.dtype
    rq = _cell_diagonal(half)  # the cell's half-diagonal
    sc = corner.sc
    with_rates = rates is not None

    def ramp(x: Array) -> Array:
        return jnp.clip(x / _CORNER_RAMP, 0.0, 1.0)

    def per_column(x: Array) -> Array:
        """(N, ...) -> (N, S, ...)."""
        return jnp.broadcast_to(x[:, None], (n, s, *x.shape[1:]))

    def cuts(p: Array) -> Array:
        """Whether half-planes (..., 3) cut the cell (no gradient): their offset within its support."""
        p = sg(p)
        if isinstance(half, tuple):
            return jnp.abs(p[..., 2]) < half[0] * jnp.abs(p[..., 0]) + half[1] * jnp.abs(p[..., 1])
        return jnp.abs(p[..., 2]) < half * (jnp.abs(p[..., 0]) + jnp.abs(p[..., 1]))

    # every literal: constant, single edge or pair, and how far it is certified
    slots: list[_Slot] = []
    slot_slopes = []  # z slope of each literal's planes (they move with their level)
    weight = jnp.where(corner.taken[:, None], jnp.ones((n, s), dtype=dtype), 0.0)
    # and the certificate for the voxel: 1 where every literal is constant, or
    # all but one of each level, which is single, with _VOXEL_RAMP to spare;
    # it is > 0 only where the weight is
    voxel = weight
    for j, fill in enumerate(is_fill):
        if fill:
            continue
        here = present[..., j]
        constant, alone = [], []  # per literal: how surely constant, single
        for layer, a, b in corner.literals[j]:
            m = jax.tree.map(lambda x, layer=layer: x[layer], corner.layers)
            off = a * shift[..., j] + b * sc
            val = m.sd[:, None] + off
            reach = rq + jnp.abs(off) + _CORNER_TOL
            d2s = m.d2s[:, None]
            const = sg(jnp.abs(val)) >= rq
            single = ~const & (d2s > sg(reach))
            pair = ~const & ~single
            today = jnp.concatenate([per_column(m.normal), val[..., None]], axis=-1)
            pa, pb = (
                jnp.concatenate([per_column(m.planes[:, i, :2]), (m.planes[:, i, 2, None] + off)[..., None]], axis=-1)
                for i in (0, 1)
            )
            first = jnp.where(single[..., None], today, pa)
            # 1 where the model holds over the whole reach, ramping to 0 with the
            # radius; the switches between modes happen where both are exact
            cert = ramp(m.rpair[:, None] - reach)
            certified = jnp.where(d2s > reach, jnp.maximum(cert, ramp(d2s - reach)), cert)
            certified = jnp.where(jnp.abs(val) >= rq, jnp.maximum(certified, ramp(jnp.abs(val) - rq)), certified)
            weight = weight * jnp.where(here, certified, 1.0)
            constant.append(jnp.clip((jnp.abs(val) - rq) / _VOXEL_RAMP, 0.0, 1.0))
            alone.append(jnp.where(d2s > reach, jnp.clip((d2s - reach) / _VOXEL_RAMP, 0.0, 1.0), 0.0))
            if with_rates:
                slot_slopes.append(a * rates[..., j, :])
            o = sg(off)
            slots.append(
                _Slot(
                    level=j,
                    layer=layer,
                    off=off,
                    val=val,
                    const=const,
                    single=single,
                    op_and=per_column(m.op_and),
                    planes=(first, pb),
                    live=(here & ~const & cuts(first), here & pair & cuts(pb)),
                    rounded=here & pair & m.linked[:, None] & jnp.where(m.convex[:, None], o < 0, o > 0),
                )
            )
        sure = math.prod(constant) + sum(
            g * math.prod(c for k, c in enumerate(constant) if k != i) for i, g in enumerate(alone)
        )
        voxel = voxel * jnp.where(here, jnp.minimum(sure, 1.0), 1.0)

    # distinct live planes: the first of each level is its leftover, the others
    # take the split lines (unused lines are far: their inside is empty)
    num_lines = corner.lines
    far = jnp.broadcast_to(jnp.asarray([1.0, 0.0, _cell_reach(half)], dtype=dtype), (n, s, 3))
    lines, used = [far] * num_lines, [jnp.zeros((n, s), dtype=bool)] * num_lines
    left, has = [far] * len(is_fill), [jnp.zeros((n, s), dtype=bool)] * len(is_fill)
    width = rates.shape[-1] if with_rates else 2
    lines_b = [jnp.zeros((n, s, width), dtype=dtype)] * num_lines  # rates of the lines and leftovers
    left_b = [jnp.zeros((n, s, width), dtype=dtype)] * len(is_fill)
    ok_lines = jnp.ones((n, s), dtype=bool)
    for i, slot in enumerate(slots):
        j = slot.level
        for plane, live in zip(slot.planes, slot.live, strict=True):
            seen = has[j] & (_same_plane(plane, left[j]) != 0)
            for k in range(num_lines):
                seen = seen | (used[k] & (_same_plane(plane, lines[k]) != 0))
            new = live & ~seen
            leftover = new & ~has[j]
            left[j] = jnp.where(leftover[..., None], plane, left[j])
            if with_rates:
                left_b[j] = jnp.where(leftover[..., None], slot_slopes[i], left_b[j])
            has[j] = has[j] | leftover
            rest = new & ~leftover
            for k in range(num_lines):
                put = rest & ~used[k]
                lines[k] = jnp.where(put[..., None], plane, lines[k])
                if with_rates:
                    lines_b[k] = jnp.where(put[..., None], slot_slopes[i], lines_b[k])
                used[k] = used[k] | put
                rest = rest & ~put
            ok_lines = ok_lines & ~rest

    # cell c is on the inside of line k if bit k of c is set (see _corner_polygons)
    num_cells = 2**num_lines

    # every plane's value on each side of its level's leftover (outside,
    # inside) in each cell: the side of the line or leftover it is, else its
    # constant value over the square
    planes = jnp.stack([p for slot in slots for p in slot.planes], axis=-2)  # (N, S, 2T, 3)
    live = jnp.stack([x for slot in slots for x in slot.live], axis=-1)  # (N, S, 2T)
    level_of = [slot.level for slot in slots for _ in slot.planes]
    own = jnp.stack([left[j] for j in level_of], axis=-2)
    own_has = jnp.stack([has[j] for j in level_of], axis=-1)
    on_left = jnp.where(own_has, _same_plane(planes, own), 0)[..., None, None]  # (N, S, 2T, 1, 1)
    on_line = _same_plane(planes[..., None, :], jnp.stack(lines, axis=-2)[..., None, :, :])  # (N, S, 2T, M)
    on_line = jnp.where(jnp.stack(used, axis=-1)[..., None, :], on_line, 0)
    negative = (sg(planes[..., 2]) < 0)[..., None, None]  # the value over the square if it does not cut it
    sides = jnp.asarray([[False], [True]])  # (2, 1)
    value = jnp.where(on_left == 1, sides, jnp.where(on_left == -1, ~sides, negative))
    for k in range(num_lines):
        bits = jnp.asarray([bool((c >> k) & 1) for c in range(num_cells)])  # (C,)
        on = on_line[..., k, None, None]
        value = jnp.where(on == 1, bits, jnp.where(on == -1, ~bits, value))
    value = jnp.where(live[..., None, None], value, negative)
    value = jnp.broadcast_to(value, (n, s, len(level_of), 2, num_cells))  # (N, S, 2T, 2, C)

    # each level in each cell: full, empty, or inside / outside its leftover
    unit = jnp.asarray([1.0, 0.0], dtype=dtype)
    normals, offsets, slopes = [], [], []
    for j, fill in enumerate(is_fill):
        if fill:
            normals.append(jnp.broadcast_to(unit, (n, s, num_cells, 2)))
            offsets.append(jnp.zeros((n, s, num_cells), dtype=dtype))
            slopes.append(jnp.zeros((n, s, num_cells, width), dtype=dtype) if with_rates else None)
            continue
        values = [
            _literal_value(slot, value[..., 2 * i, :, :], value[..., 2 * i + 1, :, :])
            for i, slot in enumerate(slots)
            if slot.level == j
        ]
        t = jnp.broadcast_to(_eval_bool(corner.trees[j], values), (n, s, 2, num_cells))
        outside, inside = t[..., 0, :], t[..., 1, :]
        lf = left[j][:, :, None, :]
        only_in, only_out = inside & ~outside, outside & ~inside
        normals.append(jnp.where(only_in[..., None], lf[..., :2], jnp.where(only_out[..., None], -lf[..., :2], unit)))
        offsets.append(
            jnp.where(only_in, lf[..., 2], jnp.where(only_out, -lf[..., 2], jnp.where(inside, -_cell_reach(half), _cell_reach(half))))
        )
        if with_rates:
            lb = left_b[j][:, :, None]
            slopes.append(jnp.where(only_in[..., None], lb, jnp.where(only_out[..., None], -lb, 0.0)))
    return _CornerCells(
        slots=slots,
        weight=weight,
        voxel=voxel,
        lines=lines,
        used=used,
        lines_b=lines_b,
        ok_lines=ok_lines,
        normals=normals,
        offsets=offsets,
        slopes=slopes if with_rates else None,
    )


def _corner_polygons(
    cc: _CornerCells, is_fill: Sequence[bool], half: float, with_attr: bool, center: Array | None = None,
    sub: float | None = None,
) -> tuple[Array, Array, Array | None]:
    """The 2^M cells the split lines cut a square into (cell c inside line k if bit k of c is set).

    Args:
        cc: the corner model (:func:`_corner_cells`).
        is_fill: (K,) static.
        half: half side of the lateral square (the lines' offsets are clipped to +-2 half).
        with_attr: carry the edge attributes of :func:`_moments` (both sides of a line carry its plane).
        center: (..., 2) center of the square to cut, in the lateral square's frame (a sub-square), or None.
        sub: half side of that square (default: ``half``).

    Returns:
        poly (2N, S, C, V, 2), count (2N, S, C) and, with ``with_attr``, the attributes (2N, S, C, V, 4); the
        polygons are in the cut square's own frame (centered on it).
    """
    lines = cc.lines
    n, s = lines[0].shape[:2]
    dtype = lines[0].dtype
    num_lines = len(lines)
    num_poly = sum(not f for f in is_fill)
    num_vertices = 4 + num_lines + num_poly
    h = half if sub is None else sub
    square = _cell_corners(h, dtype)
    if isinstance(h, _Cells):  # per column (2N, 1, 4, 2): over the cells (2N, 1, 1, V, 2)
        square = jnp.concatenate([square, jnp.repeat(square[..., -1:, :], num_vertices - 4, axis=-2)], axis=-2)
        square = square[..., None, :, :]
    else:
        square = jnp.concatenate([square, jnp.repeat(square[-1:], num_vertices - 4, axis=0)])
    batch = (n, s) if center is None else center.shape[:-1]
    poly = jnp.broadcast_to(square, (*batch, 1, num_vertices, 2))  # (..., C, V, 2)
    count = jnp.full((*batch, 1), 4, dtype=jnp.int32)
    attr = jnp.zeros((*batch, 1, num_vertices, 4), dtype=dtype) if with_attr else None
    for k in range(num_lines):  # the cells outside line k, then those inside
        cells = poly.shape[-3]
        offset = jnp.clip(lines[k][..., 2], -_cell_reach(half), _cell_reach(half))
        if center is not None:  # in the cut square's frame
            offset = offset + jnp.sum(lines[k][..., :2] * center, axis=-1)
        nk = jnp.broadcast_to(lines[k][..., None, :2], (*batch, cells, 2))
        ok = jnp.broadcast_to(offset[..., None], (*batch, cells))
        args = (
            jnp.concatenate([poly, poly], axis=-3),
            jnp.concatenate([count, count], axis=-1),
            jnp.concatenate([nk, -nk], axis=-2),
            jnp.concatenate([ok, -ok], axis=-1),
        )
        if with_attr:  # both sides' new edges carry the line's plane
            line = jnp.concatenate([lines[k][..., :2], cc.lines_b[k]], axis=-1)
            line = jnp.broadcast_to(line[..., None, :], (*batch, 2 * cells, 4))
            poly, count, attr = _clip(*args, jnp.concatenate([attr, attr], axis=-3), line)
        else:
            poly, count = _clip(*args)
    return poly, count, attr


def _corner_square(
    cc: _CornerCells,
    corner: _CornerInputs,
    present: Array,
    is_fill: Sequence[bool],
    half: float,
    cell: float,
    vectors: np.ndarray,
    contrast2: np.ndarray,
    center: Array | None = None,
    sub: float | None = None,
) -> tuple[Array, Array, Array, Array]:
    """The corner model's cells painted on one square, with their faces (both Gauss heights, no blending).

    In every cell, each level's half-plane is bounded by its leftover line (:func:`_line_faces`, the chord
    restricted to the cell), and across every split line the two cells' 1D paintings along it are overlaid
    (:func:`_line_partition`), so faces between cells are found with the parts on both sides, and one
    material on both sides of a split line is no face. Faces carry the planes' tilt (their wall slope).
    Round tips move area, not faces (the faces are those of the mitred corner).

    Args:
        cc: the corner model (:func:`_corner_cells`), (2N, S, ...).
        corner: the corner columns, doubled for the two heights.
        present: (2N, S, K) whether each level is in each sub-interval.
        is_fill: (K,) static.
        half: half side of the lateral square.
        cell: cells per side of the lateral square.
        vectors: (K + 1, 6) static normalized material tensors.
        contrast2: (K + 1, K + 1) static squared contrasts.
        center: (2,) center of the painted square in the lateral square's frame, or None: the lateral
            square itself.
        sub: half side of the painted square (default: ``half``).

    On graded x or y ``half`` (and ``sub``) are the columns' own cells, (2N, 1) over the sub-intervals,
    ``cell`` (2N, 1) and ``center`` (2N, 2).

    Returns:
        exact: (2N, S, K + 1) fractions of the painted square (round tips moved).
        moments: (2N, S, 6, 3) material moments (zeroth and standardized first, about its center).
        tensor: (2N, S, 6) the lateral face tensor (weights as :func:`_interface_parts`).
        ok: (2N, S) no clip overflowed and the tips could be corrected.
    """
    n, s = present.shape[:2]
    dtype = cc.weight.dtype
    num = len(is_fill)
    h = half if sub is None else sub
    side = 2 * h if not isinstance(h, tuple) else math.sqrt(_cell_area(h))  # side**2: the area
    per_column = isinstance(h, _Cells)
    # per column: the painted cell over the cut cells (and the tips), cell over the faces, center over the
    # sub-intervals (and the cells and levels)
    h_c = _cells_expand(h, 1) if per_column else h
    center_s = center[:, None, :] if per_column and center is not None else center
    normals = jnp.stack(cc.normals, axis=-2)  # (2N, S, C, K, 2)
    offsets = jnp.stack(cc.offsets, axis=-1)  # (2N, S, C, K)
    if center is None:
        poly, count, _ = _corner_polygons(cc, is_fill, half, False)
    else:  # the painted square's own frame
        poly, count, _ = _corner_polygons(cc, is_fill, half, False, jnp.broadcast_to(center_s, (n, s, 2)), h)
        if per_column:
            offsets = offsets + jnp.sum(normals * center[:, None, None, None, :], axis=-1)
        else:
            offsets = offsets + jnp.sum(normals * center, axis=-1)
    num_cells = poly.shape[2]
    ok_clip = jnp.all(count <= poly.shape[-2], axis=-1)
    tilts = jnp.stack([x[..., 0] for x in cc.slopes], axis=-1)  # (2N, S, C, K)
    present_c = jnp.broadcast_to(present[:, :, None, :], (n, s, num_cells, num))
    visible, first, overflow = _paint_moments(poly, count, normals, offsets, present_c, is_fill, h)

    def moved(line: Array) -> Array:
        """A split line's offset (clipped as painted) in the painted square's frame."""
        o = jnp.clip(line[..., 2], -_cell_reach(half), _cell_reach(half))
        return o if center is None else o + jnp.sum(line[..., :2] * center_s, axis=-1)

    # the cells: inside split line k where bit k is set (keep -(n . p + o) >= 0), else outside
    bounds = []
    for k, line in enumerate(cc.lines):
        sign = jnp.asarray([-1.0 if (c >> k) & 1 else 1.0 for c in range(num_cells)], dtype=dtype)
        bounds.append((sign[:, None] * line[..., None, :2], sign * moved(line)[..., None]))
    zero = jnp.zeros(2, dtype=dtype)
    # faces on the levels' half-planes inside each cell: lengths (2N, S, C, K, K + 1), weighed together
    lengths = _face_lengths(normals, offsets, present_c, is_fill, zero, h_c, bounds)
    cell_f = cell[..., None, None, None] if per_column else cell  # (2N, 1, 1, 1, 1) over the cells' faces
    weight = _face_weights(lengths, visible[..., :-1, None], visible[..., None, :], contrast2[:-1], cell_f, side)
    weight = jnp.sum(weight, axis=-1) * jnp.sqrt(1.0 + tilts**2)  # (2N, S, C, K): per line
    dyads = _dyad(jnp.concatenate([normals, tilts[..., None]], axis=-1) / jnp.sqrt(1.0 + tilts**2)[..., None])
    tensor = jnp.einsum("...ck,...ckv->...v", weight, dyads, precision=_HIGHEST)
    # faces on the split lines: the paintings of the two cells along it, overlaid
    for m, line in enumerate(cc.lines):
        nm = line[..., :2]
        dm = jnp.stack([-nm[..., 1], nm[..., 0]], axis=-1)
        pm = -moved(line)[..., None] * nm
        tilt = cc.lines_b[m][..., 0]
        dyad = _dyad(jnp.concatenate([nm, tilt[..., None]], axis=-1) / jnp.sqrt(1.0 + tilt**2)[..., None])
        lo0, hi0 = _line_range(pm, dm, zero, h)
        hi0 = jnp.where(cc.used[m], hi0, -_T_FAR)
        for c0 in range(num_cells):
            if (c0 >> m) & 1:
                continue
            c1 = c0 | (1 << m)
            lo, hi = lo0, hi0
            for k, (nb, ob) in enumerate(bounds):
                if k != m:
                    nb0, ob0 = nb[..., c0, :], ob[..., c0]
                    lo, hi = _keep_side(
                        lo, hi, jnp.sum(nb0 * dm, axis=-1), jnp.sum(nb0 * pm, axis=-1) + ob0,
                        -_ON_LINE * jnp.abs(jnp.sum(nb0 * nm, axis=-1)),
                    )
            outside = _line_partition(normals[:, :, c0], offsets[:, :, c0], present, is_fill, pm, dm, lo, hi, nm)
            inside = _line_partition(normals[:, :, c1], offsets[:, :, c1], present, is_fill, pm, dm, lo, hi, -nm)
            # every pair of the two sides' parts: their overlap along the line (2N, S, K + 1, K + 1)
            a0, b0 = (jnp.stack(x, axis=-1)[..., :, None] for x in zip(*outside, strict=True))
            a1, b1 = (jnp.stack(x, axis=-1)[..., None, :] for x in zip(*inside, strict=True))
            overlap = jnp.maximum(jnp.minimum(b0, b1) - jnp.maximum(a0, a1), 0.0)
            weight = _face_weights(
                overlap, visible[:, :, c0, :, None], visible[:, :, c1, None, :], contrast2,
                cell[..., None, None] if per_column else cell, side,
            )
            tensor = tensor + (jnp.sum(weight, axis=(-2, -1)) * jnp.sqrt(1.0 + tilt**2))[..., None] * dyad
    exact = jnp.sum(visible, axis=2)  # (2N, S, K + 1)
    # round joins only where some literal of this voxel layer is rounded (area only)
    center_t = center[:, None, None, :] if per_column and center is not None else center  # over the tips
    exact, gate = jax.lax.cond(
        jnp.any(jnp.stack([slot.rounded for slot in cc.slots])),
        lambda e: _round_tips(cc.slots, corner, e, present, is_fill, h_c, center_t),
        lambda e: (e, jnp.ones((n, s), dtype=bool)),
        exact,
    )
    ok = ok_clip & ~jnp.any(overflow, axis=-1) & gate
    standard = _cell_standard(h, side, dtype)
    if per_column:  # over the levels
        standard = standard[..., None, :]
    mom = jnp.concatenate([exact[..., None], standard * jnp.sum(first, axis=2)], axis=-1)  # (2N, S, K + 1, 3)
    moments = jnp.einsum("...lc,lv->...vc", mom, jnp.asarray(vectors, dtype=dtype), precision=_HIGHEST)
    return exact, moments, tensor, ok


def _corner_interfaces(
    corner: _CornerInputs,
    t_visible: Array,
    t_tensor: Array,
    t_moments: Array,
    length: Array,
    present: Array,
    shift: Array,
    rates: Array,
    planes: tuple[Array, Array],
    is_fill: Sequence[bool],
    half: float,
    cell: float,
    vectors: np.ndarray,
    contrast2: np.ndarray,
    squares: tuple | None = None,
) -> tuple[Array, ...]:
    """The corner columns of one voxel layer with their interfaces, for averaging="subpixel_full".

    The corner model of :func:`_corner_fill` (the same cells, at the two Gauss heights of every level's
    part, with round tips, the straight-edge seams and the blend ``w E + (1 - w) T`` by the certificate),
    with the cells also giving its faces (:func:`_corner_square`). E's faces and material moments are
    blended with T's exactly as its fractions: the seams' mix (T's where the corner model is straight)
    and then w.

    With ``squares``, sub-squares of the voxel (the junction closure's) are painted with the same model,
    each with its own tips, and blended with their own half-plane paintings by the voxel's certificate and
    seams, their unions shifted together so that the sub-squares add up to the voxel exactly
    (:func:`_straight_seams`).

    Args:
        corner: the corner columns (no indices).
        t_visible: (N, S, K + 1) the half-plane model's fractions per sub-interval (T).
        t_tensor: (N, S, 6) T's lateral face tensor per sub-interval (:func:`_interface_parts`).
        t_moments: (N, S, 6, 3) T's material moments.
        length: (N, S) sub-interval lengths.
        present: (N, S, K) whether each level is in each sub-interval.
        shift: (N, S, K, 2) each level's offset at the lower and upper Gauss heights of its part.
        rates: (N, S, K) each level's wall slope (cells per cell).
        planes: (N, K) offsets and (N, K, 2) normals of the columns' half-planes (T's).
        is_fill: (K,) static.
        half: half side of the lateral square.
        cell: cells per side of the lateral square.
        vectors: (K + 1, 6) static normalized material tensors.
        contrast2: (K + 1, K + 1) static squared contrasts.
        squares: None, or (centers (Q, 2), their half side, and their T: fractions (N, S, Q, K + 1), face
            tensors (N, S, Q, 6), moments (N, S, Q, 6, 3), and the fractions at the Gauss heights' mean
            (N, S, Q, K + 1)).

    On graded x or y ``half`` is the columns' own cells and ``cell`` their (N,) frame units per side
    (:class:`_Cells`); the squares' centers are then (N, Q, 2) and their half sides the columns' sub-cells.

    Returns:
        The blended fractions (N, S, K + 1), lateral face tensor (N, S, 6) and material moments (N, S, 6, 3)
        per sub-interval, and the certificate w (N, S) they were blended with (below 1: partly the half-plane
        model); with ``squares`` also the sub-squares' fractions, face tensors and moments (N, S, Q, ...).
    """
    n0 = length.shape[0]
    dtype = length.dtype
    corner = corner._replace(
        taken=jnp.concatenate([corner.taken] * 2),
        layers=jax.tree.map(lambda x: jnp.concatenate([x, x], axis=1), corner.layers),
    )
    per_column = isinstance(half, _Cells)
    if per_column:  # doubled with the columns, over the sub-intervals: (2N, 1)
        corner = corner._replace(sc=jnp.concatenate([corner.sc] * 2))
        half = _cells_expand(_cells_twice(half), 1)
        cell = jnp.concatenate([cell, cell])[:, None]
    length2, present2 = (jnp.concatenate([x, x]) for x in (length, present))
    shift2 = jnp.concatenate([shift[..., 0], shift[..., 1]])
    n, s = length2.shape
    # the half-planes at the two heights (T_g, for the straight seams)
    po, pn = (jnp.concatenate([x, x]) for x in planes)
    num_t = 4 + sum(not f for f in is_fill)
    square_t = _cell_corners(half, dtype)
    if per_column:
        square_t = jnp.concatenate([square_t, jnp.repeat(square_t[..., -1:, :], num_t - 4, axis=-2)], axis=-2)
    else:
        square_t = jnp.concatenate([square_t, jnp.repeat(square_t[-1:], num_t - 4, axis=0)])
    gauss_t = _paint_levels(
        jnp.broadcast_to(square_t, (n, s, num_t, 2)),
        jnp.full((n, s), 4, dtype=jnp.int32),
        jnp.ones((n, s), dtype=dtype),
        jnp.broadcast_to(pn[:, None], (n, s, *pn.shape[1:])),
        po[:, None, :] + shift2,
        present2,
        is_fill,
        half,
    )[0]

    cc = _corner_cells(corner, present2, shift2, is_fill, half, jnp.concatenate([rates, rates])[..., None])
    exact, moments, tensor, ok = _corner_square(cc, corner, present2, is_fill, half, cell, vectors, contrast2)
    w = jnp.where(cc.ok_lines & ok, cc.weight, 0.0)

    def mean(x: Array) -> Array:
        """(2N, ...) -> (N, ...): the mean over the two heights."""
        return 0.5 * (x[:n0] + x[n0:])

    exact, tensor, moments = mean(exact), mean(tensor), mean(moments)
    w = jnp.minimum(w[:n0], w[n0:])  # (N, S)
    exact, straight = _straight_seams(exact, t_visible, mean(gauss_t), is_fill)
    mix = jnp.min(straight, axis=-1)  # (N, S): T's faces where every union is straight
    tensor = mix[..., None] * t_tensor + (1 - mix[..., None]) * tensor
    moments = mix[..., None, None] * t_moments + (1 - mix[..., None, None]) * moments
    wv = w[..., None]
    out = (
        wv * exact + (1 - wv) * t_visible,
        wv * tensor + (1 - wv) * t_tensor,
        wv[..., None] * moments + (1 - wv[..., None]) * t_moments,
        w,
    )
    if squares is None:
        return out
    centers, sub, tq_visible, tq_tensor, tq_moments, tq_gauss = squares
    if per_column:  # each column's own: centers (Q, 2N, 2), sub-cells (2N, 1)
        centers = jnp.moveaxis(jnp.concatenate([centers, centers]), 1, 0)
        sub = _cells_expand(_cells_twice(sub), 1)
    # one sub-square at a time (a vmap over them made XLA fuse the whole painting into one kernel that took
    # minutes to compile)
    q_exact, q_moments, q_tensor, _ = jax.lax.map(
        jax.checkpoint(lambda c: _corner_square(cc, corner, present2, is_fill, half, cell, vectors, contrast2, c, sub)),
        centers,
    )  # (Q, 2N, S, ...), each sub-square recomputed in the backward pass
    q_exact, q_moments, q_tensor = (jnp.moveaxis(mean(jnp.moveaxis(x, 0, 1)), 1, 2) for x in (q_exact, q_moments, q_tensor))
    # (N, S, Q, ...): the voxel's seams (its v per level, and its unions as their mean) and its certificate
    unions = jnp.cumsum(exact[..., :-1], axis=-1)[:, :, None, :]  # (N, S, 1, K)
    q_exact = _straight_seams(q_exact, tq_visible, tq_gauss, is_fill, straight[:, :, None], unions)[0]
    mq = mix[:, :, None]
    q_tensor = mq[..., None] * tq_tensor + (1 - mq[..., None]) * q_tensor
    q_moments = mq[..., None, None] * tq_moments + (1 - mq[..., None, None]) * q_moments
    wq = w[:, :, None, None]
    return (
        *out,
        wq * q_exact + (1 - wq) * tq_visible,
        wq * q_tensor + (1 - wq) * tq_tensor,
        wq[..., None] * q_moments + (1 - wq[..., None]) * tq_moments,
    )


@timed("interface_corner_pass")
@partial(
    jax.jit,
    static_argnames=("trees", "literals", "is_fill", "lines", "sc", "half", "cell", "vectors", "contrast2", "graded"),
)
def _interface_corner_pass(
    layers: _LayerCorner,
    taken: Array,
    columns: tuple[Array, ...],
    planes: tuple[Array, Array],
    *,
    trees: tuple,
    literals: tuple,
    is_fill: tuple[bool, ...],
    lines: int,
    sc: float,
    half: float,
    cell: float,
    vectors: tuple,
    contrast2: tuple,
    graded: bool = False,
    cells: tuple[Array, Array, Array] | None = None,
) -> tuple[Array, Array, Array]:
    """The corner columns of every voxel layer with :func:`_corner_interfaces` (a separate jitted pass).

    Voxel layers where no polygon level is in a corner column keep the half-plane result.

    Args:
        layers: local model of every source layer the literals read, (L, N, ...).
        taken: (N,) whether a slot holds a column.
        columns: per voxel layer (nz, N, ...): T's fractions, lateral face tensor and moments per
            sub-interval, the sub-interval lengths and ends, presence, the levels' offsets at the Gauss
            heights, their wall slopes and the surfaces' slopes at the inner ends (see
            :func:`_interface_composite`).
        planes: (N, K) offsets and (N, K, 2) normals of the columns' half-planes.
        trees, literals, lines, sc: see :class:`_CornerInputs`.
        is_fill: (K,) static.
        half: half side of the lateral square.
        cell: cells per side of the lateral square.
        vectors: (K + 1, 6) static normalized material tensors (nested tuples).
        contrast2: (K + 1, K + 1) static squared contrasts (nested tuples).
        graded: the layers are a graded grid's (see :func:`_interface_composite`).
        cells: on graded x or y, the columns' own lateral cells (see :func:`_corner_pass`).

    Returns:
        (nz, N, K + 1) the columns' fractions, (nz, N, 6) their direction tensors and (nz,) the number of
        the columns' voxels per layer that the corner model does not certify (blended toward half-planes).
    """
    if cells is not None:  # each column's own cell (see _corner_pass)
        half = _Cells(*cells, reach=half)
        sc, cell = (1.0 / cells[2])[:, None], cells[2]
    corner = _CornerInputs(cols=None, taken=taken, layers=layers, trees=trees, literals=literals, lines=lines, sc=sc)
    polygons = np.logical_not(is_fill)
    vec, con = np.asarray(vectors), np.asarray(contrast2)
    if cells is not None:  # (N, 1): over the tensor's entries
        area = (cell**2)[:, None]
    else:
        area = (2 * half * cell) ** 2 if not isinstance(half, tuple) else _cell_area(half) * cell**2

    @jax.checkpoint  # recomputed in the backward pass: one voxel layer's corner model in memory at a time
    def corner_model(*args: Array) -> tuple[Array, Array, Array, Array]:
        return _corner_interfaces(corner, *args, planes, is_fill, half, cell, vec, con)

    def one_slice(inputs: tuple[Array, ...]) -> tuple[Array, Array, Array]:
        t_visible, t_tensor, t_moments, length, bp, present, shift, rates, slopes = inputs
        busy = jnp.any(present[..., polygons] & taken[:, None, None])
        f, d, m, w = jax.lax.cond(
            busy,
            lambda _: corner_model(t_visible, t_tensor, t_moments, length, present, shift, rates),
            lambda _: (t_visible, t_tensor, t_moments, jnp.ones(length.shape, dtype=length.dtype)),
            None,
        )
        fractions = jnp.sum(length[..., None] * f, axis=-2)
        if graded:  # the layer's fractions; its lateral faces as high as it is
            fractions = fractions / (bp[..., -1:] - bp[..., :1])
            direction = jnp.sum(length[..., None] * d, axis=-2) + _vertical_tensor(length, bp, m, slopes, True) * area
        else:
            direction = jnp.sum(length[..., None] * d, axis=-2) + _vertical_tensor(length, bp, m, slopes) * area
        # the columns' voxels whose polygon levels the corner model does not certify (partly half-planes)
        drawn = jnp.any(present[..., polygons], axis=-1) & (length > 0)
        uncertified = jnp.sum(taken & jnp.any(drawn & (w < 1.0 - _CERTIFIED), axis=-1), dtype=jnp.int32)
        return fractions, direction, uncertified

    return jax.lax.map(one_slice, columns)


def _blend(w: Array, exact: Array, dense: Array, length: Array, *moments: Array) -> Array | tuple[Array, ...]:
    """``w E + (1 - w) T`` of the corner model E and the half-planes T (see :func:`_corner_fill`).

    Args:
        w: (N, S) the weight.
        exact: (N, S, K + 1) E's fractions per sub-interval.
        dense: (N, S, K + 1) T's.
        length: (N, S) sub-interval lengths.
        moments: optionally E's flux (N, S, C, 3), T's, E's rates (N, S, C,
            2) and T's.

    Returns:
        (N, K + 1) the blended fractions (length weighted); with ``moments``
        also the blended fractions, flux and rates per sub-interval.
    """
    vis = w[..., None] * exact + (1 - w[..., None]) * dense
    fill = jnp.stack([jnp.sum(length * vis[..., j], axis=-1) for j in range(vis.shape[-1])], axis=-1)
    if not moments:
        return fill
    flux, flux_t, rate, rate_t = moments
    w = w[..., None, None]
    return fill, vis, w * flux + (1 - w) * flux_t, w * rate + (1 - w) * rate_t


def _lateral_range(level: _Level, wall: tuple[Array, Array, Array], lo: Array, hi: Array,
                   z_span: tuple[Any, Any], res_um: float) -> tuple[Array, Array]:
    """Smallest and largest sidewall offset (cells) of a level in the box (no gradient).

    The offset is piecewise linear in z (kinks at the z_to_bias knots), so its
    range over the heights the level covers is reached at their ends or a knot.

    Args:
        level: the level.
        wall: its (zmin, thickness, sidewall angle) as in :func:`_sidewall_shift`.
        lo: (nx, ny) its bottom surface (cells).
        hi: (nx, ny) its top surface (cells).
        z_span: (bottom, top) of the sampled voxel layers, in the surfaces' frame
            (0 and the number of layers on a uniform grid).
        res_um: grid resolution in um.

    Returns:
        (min, max) of the offset.
    """
    sg = jax.lax.stop_gradient
    zb, t, angle = (sg(jnp.asarray(x, dtype=lo.dtype)) for x in wall)
    z_lo = jnp.clip(jnp.min(sg(lo)), *z_span)
    z_hi = jnp.clip(jnp.max(sg(hi)), *z_span)
    z = [z_lo, z_hi]
    if level.z_to_bias is not None:
        z += [jnp.clip(zb + f * t, z_lo, z_hi) for f in level.z_to_bias[0]]
    offsets = _sidewall_shift(level, jnp.stack(z), zb, t, angle, res_um)
    return jnp.min(offsets), jnp.max(offsets)


def _corner_candidates(plan: _CornerPlan, radii: Sequence[Sequence[Array]], dist: Sequence[Array]) -> Array:
    """The columns where the half-plane of some level may not be exact (no gradient).

    A literal is near a column if its layer's nearest boundary edge is within
    its radius (the lateral square's half-diagonal plus the largest offset of
    the literal in the box). Elsewhere a literal is constant over the square.
    A level whose only near literal has a single edge within the radius is
    exactly today's half-plane: the point lies in the interior Voronoi region
    of that edge (if its nearest point were a vertex, the edge continuing there
    would be as near), so the boundary there is that edge's line. The other
    columns are the candidates: two near literals of one level (a crossing of
    layer edges), or a near literal with a second edge within its radius (a
    corner, a strip or a gap).

    Args:
        plan: the levels' literals.
        radii: per level, per literal: the radius (cells).
        dist: per layer slot of the plan, (P, 3) distances (cells) to the
            nearest boundary edges (:func:`bifrost.geometry.nearest_edges`).

    Returns:
        (P,) whether each column is a candidate.
    """
    candidate = jnp.zeros(dist[0].shape[0], dtype=bool)
    for literals, rad in zip(plan.literals, radii, strict=True):
        near_count = jnp.zeros(candidate.shape, dtype=jnp.int32)
        for (layer, *_), r in zip(literals, rad, strict=True):
            near = dist[layer][:, 0] <= r
            near_count = near_count + near.astype(jnp.int32)
            candidate = candidate | (near & (dist[layer][:, 1] <= r))
        candidate = candidate | (near_count >= 2)
    return candidate
