"""KLayout-based width, spacing, and overlap DRC diagnostics."""

import math
import numpy as np
from collections.abc import Callable, Sequence
from typing import Any, Literal, NamedTuple
from ._layers import _CHECK_HEIGHTS, _Level, _NARROW_ANGLE, _kdb_region


class StackDiagnostic(NamedTuple):
    """One kind of problem in one level's geometry, found by `StackDevice.diagnose` (lengths in um).

    Attributes:
        level: the level's name.
        kind: ``"width"``: parts of the level's shape narrower than
            ``threshold``; ``"gap"``: gaps in it narrower than ``threshold``;
            ``"overlap"``: overlapping polygons on layers the level shrinks.
        size: the narrowest width or gap found in the box (of places within
            1% of the threshold of the narrowest, the longest one); for
            ``"overlap"``, the largest shrink of the overlapping layers.
        cells: ``size`` in grid cells.
        threshold: the width or gap below which parts are reported (0 for
            ``"overlap"``).
        location: (x, y) gds point in the middle of that width or gap, in the
            box, or in the largest overlap.
        z: stack height of the cross-section with the narrowest width or gap,
            or where the shrink is largest.
        count: number of separate places found in the box.
        layers: for ``"overlap"``, the parameter keys (``vertices/<layer>``)
            of the layers whose polygons overlap; empty otherwise.
        message: the warning text.
    """

    level: str
    kind: Literal["width", "gap", "overlap"]
    size: float
    cells: float
    threshold: float
    location: tuple[float, float]
    z: float
    count: int
    layers: tuple[str, ...]
    message: str


def _wall_offset(level: _Level, z: np.ndarray) -> np.ndarray:
    """:func:`_sidewall_shift` in um at stack heights z (um), for the level's nominal zmin, thickness and angle."""
    z = np.asarray(z, dtype=float)
    z_ref = level.zmin + level.width_to_z * level.thickness
    grow = level.bias
    if level.z_to_bias is not None:
        zs, bs = level.z_to_bias
        frac = np.zeros_like(z) if level.thickness == 0 else (z - level.zmin) / level.thickness
        grow = grow + np.interp(frac, zs, bs)
    return (z - z_ref) * math.tan(math.radians(level.sidewall_angle)) - grow


def _check_heights(level: _Level, lo: float, hi: float, step: float) -> np.ndarray:
    """Stack heights (um) of the cross-sections of a level that are checked.

    The level's offset is piecewise linear in z (kinks at the z_to_bias knots).
    The heights include ``lo``, ``hi`` and the knots between them, and are
    spaced so that the offset changes by at most ``step`` from one to the next
    (at most about ``_CHECK_HEIGHTS`` heights; all knots are kept).

    Args:
        level: the level.
        lo: lowest height (um).
        hi: highest height (um), above ``lo``.
        step: largest change of the offset (um) between neighbouring heights.

    Returns:
        The increasing heights.
    """
    knots = [lo, hi]
    if level.z_to_bias is not None:
        knots += [level.zmin + f * level.thickness for f in level.z_to_bias[0]]
    knots = np.unique(np.clip(knots, lo, hi))
    steps = np.maximum(1, np.ceil(np.abs(np.diff(_wall_offset(level, knots))) / step)).astype(int)
    if steps.sum() >= _CHECK_HEIGHTS:
        steps = np.maximum(1, steps * (_CHECK_HEIGHTS - 1) // steps.sum())
    pieces = [np.linspace(a, b, n + 1)[1:] for a, b, n in zip(knots[:-1], knots[1:], steps, strict=True)]
    return np.concatenate([knots[:1], *pieces])


def _corners(region: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convex and reflex vertices of a merged KLayout region, with the arcs of their round joins.

    KLayout's hulls run clockwise and its holes counterclockwise, so the region
    is on the right of every contour and its convex vertices turn right. The
    round join of radius r at a convex vertex (where the region grows) or a
    reflex one (where it shrinks) is the arc of radius r around the vertex from
    the normal of its incoming edge to that of its outgoing edge, turning with
    the contour; it bulges up to ``r (1 - cos(turn / 2))`` beyond the chord
    between its ends, which is where KLayout's sizing in mode 0 cuts the corner.

    Args:
        region: the merged region.

    Returns:
        (convex, reflex): (n, 5) arrays of x, y (dbu), the angle of the join's
        first normal (outward at convex vertices, inward at reflex ones), the
        signed turn (radians) and ``1 - cos(turn / 2)``.
    """
    convex, reflex = [np.zeros((0, 5))], [np.zeros((0, 5))]
    for poly in region.each_merged():
        for contour in (poly.each_point_hull(), *(poly.each_point_hole(h) for h in range(poly.holes()))):
            c = np.array([(p.x, p.y) for p in contour], dtype=float)
            if len(c) < 3:
                continue
            e_in, e_out = c - np.roll(c, 1, axis=0), np.roll(c, -1, axis=0) - c
            cross = e_in[:, 0] * e_out[:, 1] - e_in[:, 1] * e_out[:, 0]
            turn = np.arctan2(cross, np.sum(e_in * e_out, axis=1))
            first = np.arctan2(e_in[:, 0], -e_in[:, 1]) + np.where(cross < 0, 0.0, math.pi)  # left or right normal
            rows = np.column_stack([c, first, turn, 1.0 - np.cos(0.5 * turn)])
            convex.append(rows[cross < 0])
            reflex.append(rows[cross > 0])
    return np.concatenate(convex), np.concatenate(reflex)


def _seeds(points: np.ndarray) -> Any:
    """A raw (unmerged) KLayout region of 2 x 2 dbu boxes centered on points ((n, >= 2) array, dbu)."""
    import klayout.db as kdb

    region = kdb.Region()
    region.merged_semantics = False
    for x, y in np.rint(points[:, :2]).astype(np.int64):
        region.insert(kdb.Box(int(x) - 1, int(y) - 1, int(x) + 1, int(y) + 1))
    return region


def _fans(points: np.ndarray, radius: float, tol: float) -> Any:
    """Round joins of a radius (dbu) at corners (rows of :func:`_corners`), as a KLayout region.

    Each join is the polygon from the corner to the ends of its arc and along
    the arc's circumscribed polygon, which is tangent to the two offset edges
    at their ends (it continues them without a step) and at most ``tol`` (dbu)
    outside the circle.
    """
    import klayout.db as kdb

    most = 2 * math.acos(1 / (1 + tol / radius))  # largest turn of one tangent piece
    fans = kdb.Region()
    templates: dict[float, Any] = {}  # the fan of a turn at the origin, its first normal along x
    for x, y, first, turn, _ in points:
        if turn not in templates:
            k = max(1, math.ceil(abs(turn) / most))
            step = turn / k
            angles = np.concatenate([[0.0], (np.arange(k) + 0.5) * step, [turn]])
            radii = np.full(k + 2, radius / math.cos(0.5 * step))
            radii[[0, -1]] = radius
            ends = zip(radii * np.cos(angles), radii * np.sin(angles), strict=True)
            templates[turn] = kdb.DPolygon([kdb.DPoint(0, 0), *(kdb.DPoint(float(u), float(w)) for u, w in ends)])
        fan = templates[turn].transformed(kdb.DCplxTrans(1.0, math.degrees(first), False, float(x), float(y)))
        fans.insert(fan.to_itype(1.0))
    return fans


def _round_offset(
    layer: Any,
    grow: int,
    corners: tuple[tuple[np.ndarray, Any], tuple[np.ndarray, Any]],
    near: Any,
    reach: int,
    tol: float,
) -> tuple[Any, np.ndarray]:
    """A layer's region offset the way the device draws it: by its signed distance, with round joins.

    KLayout's sizing in mode 0 cuts every corner between the ends of its two
    offset edges. That is the exact offset except at the round joins (convex
    corners where the layer grows, reflex ones where it shrinks): there the cut
    lies inside the join, and the join's fan (see :func:`_fans`) completes it.
    A corner keeps its cut where the join bulges at most ``tol`` beyond it, or
    where no edge but the corner's own two comes within ``reach + |grow|`` of
    the corner: then no width or gap involves the join, and the cut, like the
    join, meets the corner's two sides where they end (so their pair measures
    the same, see :func:`_narrow_places`).

    Args:
        layer: the layer's merged region (dbu).
        grow: offset (dbu, negative shrinks).
        corners: (vertices, seeds) of the layer's convex and of its reflex
            vertices (see :func:`_corners` and :func:`_seeds`).
        near: edges of every layer the checked shape is made of (raw).
        reach: the checks' distance plus the largest offset of those layers in
            the shape (dbu).
        tol: largest deviation (dbu) of the drawn joins from round ones.

    Returns:
        The offset region, and (m, 3) x, y and radius (dbu) of the joins drawn.
    """
    none = np.zeros((0, 3))
    if not grow:
        return layer, none
    cut = layer.sized(grow, grow, 0)
    points, seeds = corners[0] if grow > 0 else corners[1]
    r = abs(grow)
    big = points[:, 4] * r > tol
    if not big.any():
        return cut, none
    boxes = seeds.sized(reach + r)
    boxes.merged_semantics = False  # one box per corner, also where they overlap
    hits = [complex(c.x, c.y) for c in (p.bbox().center() for p in boxes.interacting(near, 3).each())]
    chosen = big & np.isin(np.rint(points[:, 0]) + 1j * np.rint(points[:, 1]), np.array(hits, dtype=complex))
    if not chosen.any():
        return cut, none
    fans = _fans(points[chosen], r, tol)
    joins = np.column_stack([points[chosen, :2], np.full(int(chosen.sum()), float(r))])
    return (cut | fans if grow > 0 else cut - fans), joins


def _tree_region(
    tree: tuple,
    literals: Sequence[tuple[int, int, float]],
    literal: Callable[[int, int], Any],
    s: float,
    universe: Any,
    dbu: float,
) -> Any:
    """A level's shape at lateral offset s, as a KLayout region (see :func:`_literals`).

    Args:
        tree: the level's literal tree.
        literals: (layer index, a, b) per literal, b in um.
        literal: the region of a layer (index) grown by an offset (dbu,
            negative shrinks).
        s: the level's offset (um, positive shrinks).
        universe: region of the plane the checks see (the complement of a
            literal is taken in it).
        dbu: grid of the regions (um).

    Returns:
        The region.
    """
    import klayout.db as kdb

    kind = tree[0]
    if kind == "lit":
        index, a, b = literals[tree[1]]
        return literal(index, round(-(a * s + b) / dbu))
    if kind in ("true", "false"):
        return universe if kind == "true" else kdb.Region()
    if kind == "neg":
        return universe - _tree_region(tree[1], literals, literal, s, universe, dbu)
    x = _tree_region(tree[1], literals, literal, s, universe, dbu)
    y = _tree_region(tree[2], literals, literal, s, universe, dbu)
    return x & y if kind == "and" else x | y


def _clip_segments(segments: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The parts of segments ((n, 2, 2) array) in the box ``|x| <= half`` (Liang-Barsky).

    Returns:
        The parts ((n, 2, 2)), and (n,) whether each one is in the open box (a
        part only on the box's boundary, along a face or at a point, is not).
    """
    p, d = segments[:, 0], segments[:, 1] - segments[:, 0]
    t0, t1 = np.zeros(len(p)), np.ones(len(p))
    inside = np.ones(len(p), dtype=bool)
    for k in range(2):
        for sign in (-1.0, 1.0):  # sign * (p + t d) <= half
            room, rate = half[k] - sign * p[:, k], sign * d[:, k]
            inside &= (rate != 0) | (room >= 0)
            t = room / np.where(rate == 0, 1.0, rate)
            t1 = np.where(rate > 0, np.minimum(t1, t), t1)
            t0 = np.where(rate < 0, np.maximum(t0, t), t0)
    parts = p[:, None] + np.stack([t0, t1], axis=1)[:, :, None] * d[:, None]
    inside &= (t0 <= t1) & np.all(np.abs(parts.mean(axis=1)) < half, axis=1)
    return parts, inside


def _closest_points(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distances between pairs of segments that do not cross ((n, 2, 2) arrays) and a closest point on each.

    Parallel segments are closest along the overlap of their projections: the
    points are then in its middle.

    Returns:
        (n,) distances, and (n, 2) points on ``a`` and on ``b``.
    """

    def foot(p: np.ndarray, s: np.ndarray) -> np.ndarray:
        """The points of segments s closest to points p."""
        d = s[:, 1] - s[:, 0]
        n = np.einsum("ij,ij->i", d, d)
        t = np.einsum("ij,ij->i", p - s[:, 0], d) / np.where(n > 0, n, 1.0)
        return s[:, 0] + np.clip(t, 0.0, 1.0)[:, None] * d

    ps = np.stack([a[:, 0], a[:, 1], foot(b[:, 0], a), foot(b[:, 1], a)], axis=1)
    qs = np.stack([foot(a[:, 0], b), foot(a[:, 1], b), b[:, 0], b[:, 1]], axis=1)
    rows = np.arange(len(a))
    k = np.argmin(np.linalg.norm(ps - qs, axis=2), axis=1)
    p, q = ps[rows, k], qs[rows, k]
    ua, ub = a[:, 1] - a[:, 0], b[:, 1] - b[:, 0]
    ua2 = np.einsum("ij,ij->i", ua, ua)
    lengths = np.sqrt(ua2 * np.einsum("ij,ij->i", ub, ub))
    t = np.einsum("nij,nj->ni", b - a[:, :1], ua) / np.where(ua2 > 0, ua2, 1.0)[:, None]  # b's ends along a
    lo, hi = np.maximum(t.min(axis=1), 0.0), np.minimum(t.max(axis=1), 1.0)
    parallel = (lengths > 0) & (np.abs(ua[:, 0] * ub[:, 1] - ua[:, 1] * ub[:, 0]) <= 1e-12 * lengths) & (lo <= hi)
    if parallel.any():
        middle = a[:, 0] + (0.5 * (lo + hi))[:, None] * ua
        p = np.where(parallel[:, None], middle, p)
        q = np.where(parallel[:, None], foot(middle, b), q)
    return np.linalg.norm(p - q, axis=1), p, q


def _on_joins(segments: np.ndarray, joins: np.ndarray, tol: float) -> np.ndarray:
    """Whether segments ((n, 2, 2), dbu) lie on round joins: their ends and middle on the arc of one of ``joins``.

    ``joins`` are (m, 3) x, y and radius (dbu); the arcs are drawn up to
    ``tol`` outside their circle.
    """
    out = np.zeros(len(segments), dtype=bool)
    if not len(joins) or not len(segments):
        return out
    slack = tol + 2.0
    largest = float(joins[:, 2].max())
    # a piece of a join is at most this long: only short segments can be on one
    short = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) <= 2 * math.sqrt((2 * largest + slack) * slack) + 4
    if not short.any():
        return out
    by_x = joins[np.argsort(joins[:, 0])]
    which = np.flatnonzero(short)
    points = np.concatenate([segments[which], segments[which].mean(axis=1, keepdims=True)], axis=1)  # ends, middle
    lo = np.searchsorted(by_x[:, 0], points[:, 2, 0] - largest - slack, side="left")
    hi = np.searchsorted(by_x[:, 0], points[:, 2, 0] + largest + slack, side="right")
    for i, start, stop, p in zip(which, lo, hi, points, strict=True):
        if stop > start:
            j = by_x[start:stop]
            off = np.hypot(j[None, :, 0] - p[:, None, 0], j[None, :, 1] - p[:, None, 1]) - j[None, :, 2]
            out[i] = bool(np.any(np.all((off > -2.0) & (off < slack), axis=0)))
    return out


def _hull_key(poly: Any) -> tuple:
    """The points of a KLayout polygon's hull, sorted (identifies a polygon whatever its first point)."""
    return tuple(sorted((p.x, p.y) for p in poly.each_point_hull()))


def _joined(a: np.ndarray, b: np.ndarray, shape: Any) -> np.ndarray:
    """Whether the shape's boundary joins the two edges of each pair where they converge.

    The two sides of a corner are joined there: along the segment between
    their closest points (a cut corner, or a blunt tip), or by a chain bulging
    beyond it toward the point where their lines meet (a round corner; sides
    that touch, as at a sharp corner, are not narrow anyway). So are a part's
    side and the round join that ends the part (the end of a strip grown with
    round corners). A pair is therefore joined where the boundary passes
    through the middle of its closest points, or crosses the triangle they span
    with the point where the lines of the two edges meet. A corner pointing at
    an edge, two corners facing each other, or the side of a V-notch facing the
    edge it nearly cuts through, are not joined there.

    Args:
        a: the first edge of each pair ((n, 2, 2), dbu).
        b: the second edge of each pair.
        shape: the checked region.

    Returns:
        (n,) whether each pair is joined.
    """
    import klayout.db as kdb

    d, p, q = _closest_points(a, b)
    mid = np.rint(0.5 * (p + q)).astype(np.int64)
    half = np.clip(d // 4, 1, 2).astype(np.int64)  # the edges of the pair stay d / 2 away from the middle
    ua, ub = a[:, 1] - a[:, 0], b[:, 1] - b[:, 0]
    den = ua[:, 0] * ub[:, 1] - ua[:, 1] * ub[:, 0]
    w = b[:, 0] - a[:, 0]
    x = a[:, 0] + ((w[:, 0] * ub[:, 1] - w[:, 1] * ub[:, 0]) / np.where(den != 0, den, 1.0))[:, None] * ua
    sides = np.linalg.norm(np.stack([q - x, x - p, p - q], axis=1), axis=2)  # opposite p, q and x
    perimeter = sides.sum(axis=1)
    area = 0.5 * np.abs((q - p)[:, 0] * (x - p)[:, 1] - (q - p)[:, 1] * (x - p)[:, 0])
    # where the lines meet (within 32-bit coordinates), and the triangle is more than a dbu thick (its
    # inradius is 2 area / perimeter): moved in by one dbu toward its incenter
    thick = np.flatnonzero((den != 0) & (sides[:, 1] < 2**28) & (2 * area > 1.5 * perimeter))
    tri = np.stack([p[thick], q[thick], x[thick]], axis=1)
    center = np.einsum("ni,nij->nj", sides[thick], tri) / perimeter[thick, None]
    scale = 1.0 - perimeter[thick] / (2 * area[thick])
    tri = np.rint(center[:, None] + scale[:, None, None] * (tri - center[:, None]))
    mids, triangles = kdb.Region(), kdb.Region()
    mids.merged_semantics = triangles.merged_semantics = False
    keys = [[] for _ in range(len(a))]
    for i, ((u, v), h) in enumerate(zip(mid.tolist(), half.tolist(), strict=True)):
        box = kdb.Polygon(kdb.Box(u - h, v - h, u + h, v + h))
        mids.insert(box)
        keys[i].append(_hull_key(box))
    for i, corners in zip(thick.tolist(), tri.astype(np.int64).tolist(), strict=True):
        triangle = kdb.Polygon([kdb.Point(u, v) for u, v in corners])
        triangles.insert(triangle)
        keys[i].append(_hull_key(triangle))
    edges = shape.edges()
    hit = {_hull_key(poly) for poly in mids.interacting(edges).each()}
    hit |= {_hull_key(poly) for poly in triangles.interacting(edges).each()}
    return np.array([any(k in hit for k in key) for key in keys], dtype=bool)


def _narrow_places(
    pairs: Any, inside: tuple[int, int], limit: int, shape: Any, joins: np.ndarray, tol: float
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """The places of a width or space check that are narrow in the box.

    The box is ``|x| < inside[0]``, ``|y| < inside[1]`` (dbu). A pair of edges
    counts where at least one of them enters the box. It is measured between
    the parts of its edges in the box (or the whole edge that stays outside),
    so a gap that narrows outside the box gets its width at the box, and it is
    left out where that is ``limit`` or wider. Pairs of ``tol`` or less
    (touching edges, as at a sharp corner, or round-off of the grid and of the
    joins) are left out, and so are pairs whose edges the boundary joins where
    they converge (see :func:`_joined`) if they face each other at
    ``_NARROW_ANGLE`` or more (the two sides of a corner) or one of them is on
    a round join (the corner's own round shape). Sides joined at a smaller
    angle (a needle, a taper, a strip) are measured.

    Args:
        pairs: the KLayout edge pairs of the check.
        inside: half-sizes of the box (dbu).
        limit: the distance of the check (dbu).
        shape: the checked region.
        joins: (m, 3) x, y and radius (dbu) of the round joins in the shape.
        tol: largest deviation (dbu) of the joins from round ones (at least
            one dbu).

    Returns:
        Per narrow place: its width, a point in the middle of its narrowest
        part, and the lower-left and upper-right corners ((2, 2)) of its part in
        the box, all in dbu.
    """
    rows = [
        (e.x1, e.y1, e.x2, e.y2, f.x1, f.y1, f.x2, f.y2)
        for e, f in ((pair.first, pair.second) for pair in pairs.each() if pair.distance() > tol)
    ]
    if not rows:
        return []
    a, b = np.moveaxis(np.array(rows, dtype=float).reshape(-1, 2, 2, 2), 1, 0)
    half = np.asarray(inside, dtype=float)
    ca, in_a = _clip_segments(a, half)
    cb, in_b = _clip_segments(b, half)
    d, p, q = _closest_points(np.where(in_a[:, None, None], ca, a), np.where(in_b[:, None, None], cb, b))
    keep = (in_a | in_b) & (d < limit)
    middle = np.clip(0.5 * (p + q), -half, half)
    lo = np.minimum(middle, np.minimum(np.where(in_a[:, None], ca.min(axis=1), np.inf),
                                       np.where(in_b[:, None], cb.min(axis=1), np.inf)))
    hi = np.maximum(middle, np.maximum(np.where(in_a[:, None], ca.max(axis=1), -np.inf),
                                       np.where(in_b[:, None], cb.max(axis=1), -np.inf)))
    ua, ub = a[:, 1] - a[:, 0], b[:, 1] - b[:, 0]
    lengths = np.linalg.norm(ua, axis=1) * np.linalg.norm(ub, axis=1)
    # facing each other at _NARROW_ANGLE or more, or one of them on a round join
    test = keep & (lengths > 0) & (-np.einsum("ij,ij->i", ua, ub) <= math.cos(math.radians(_NARROW_ANGLE)) * lengths)
    rest = np.flatnonzero(keep & ~test)
    if len(rest) and len(joins):
        test[rest[_on_joins(a[rest], joins, tol) | _on_joins(b[rest], joins, tol)]] = True
    if test.any():
        keep[np.flatnonzero(test)[_joined(a[test], b[test], shape)]] = False
    return [(float(d[i]), middle[i], np.array([lo[i], hi[i]])) for i in np.flatnonzero(keep)]


def _overlaps(polygons: Sequence[np.ndarray], dbu: float, box: Any) -> tuple[int, float, np.ndarray] | None:
    """Where polygons (um) overlap with positive area in a box (KLayout region).

    Overlaps of two dbu or thinner (round-off of abutting polygons) are left out.

    Returns:
        (number of separate overlaps, their area in um^2, (2,) center (um) of
        the largest one's bounding box), or None if they do not overlap.
    """
    overlap = (_kdb_region(polygons, dbu, merged=False).merged(False, 2) & box).sized(-1).sized(1)
    if overlap.is_empty():
        return None
    largest = max(overlap.each(), key=lambda p: p.area())
    c = largest.bbox().center()
    return overlap.count(), float(overlap.area()) * dbu**2, np.array([c.x, c.y], dtype=float) * dbu
