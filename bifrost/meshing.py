"""Automatic non-uniform meshing of a :class:`~bifrost.StackDevice`'s box.

:func:`mesh` builds the cell edges of all three axes once, before a run or an optimization, from the
device's geometry, the wavelength band and a statement of where the geometry may move
(:class:`Refine`). The grid is then fixed, so an optimization on it compiles once.

Every place a material is, or may come to be, asks for cells of at most ``lambda / (n ppw)`` there (its
target); so do the evanescent tails past each boundary (a global dilation and z padding in units of the
boundary's decay length ``l = lambda / (2 pi sqrt(n_high^2 - n_low^2))``), narrow parts and gaps of the
starting geometry, and explicit regions. The grid is a tensor product, so an axis takes, at each
coordinate, the smallest target anywhere in the plane through it; :func:`~bifrost.axis_edges` then
places continuous real-valued edges along that axis (or :func:`~bifrost.lattice_edges` when an explicit
``quantum`` is given).

The mesher only learns where to refine: it never limits how the parameters may change. A vertex that
moves past its dilation lands in cells coarser than its target, which
:meth:`~bifrost.StackDevice.coarse_vertices` counts.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import numpy as np

from .geometry import recipe_rings, run_recipe
from .grids import AxisEdges, axis_edges, lattice_edges
from .stack import (
    _ANISOTROPY_SENTINEL,
    _CHECK_ANGLE,
    _CHECK_DBU,
    _kdb_region,
    _literals,
    _part_index,
    _ring_area,
    _span,
    _wall_offset,
)
from .subpixel import material_tensor

__all__ = ["CoarseVertices", "Mesh", "Refine", "coarse_vertices", "mesh"]

_C0 = 299792458.0  # m/s
# lateral offsets: KLayout's square corners (sizing mode 2) bound a round offset from outside when growing
# and from inside when shrinking; sizing by 1 / sqrt(2) of the offset bounds it from the other side
_MODE = 2
_INNER = 1.0 / math.sqrt(2.0)
_BAND_SAMPLES = 33  # wavelengths sampled over the band, evenly in frequency
_RING = 0.005  # um: width of the ring outside a level in which its lateral neighbours are found
_OCTAGON = 1.0 / math.cos(math.pi / 8)  # circumradius of the octagon around a unit circle


@dataclass(frozen=True)
class Refine:
    """Where :func:`mesh` refines beyond the starting geometry: the geometry's motion and the tails.

    Only where to refine: nothing here limits how the parameters may change.

    Attributes:
        dilation: per source layer (its name, e.g. ``"WG"``, or its parameter key ``"vertices/WG"``), how
            far (um) its boundary may move: a scalar, or one value per vertex in the layout of
            ``stack.init_params()["vertices/<layer>"]`` (0 for vertices that never move). Along an edge the
            band tapers between its two vertices' values.
        z_padding: per level, how far (um) its faces may move: a scalar, or (below, above).
        global_dilation: lateral evanescent tail past every level's boundary, in units of that boundary's
            decay length ``l = lambda_max / (2 pi sqrt(n_high^2 - n_low^2))`` (n_high the level's index,
            n_low the highest index below it among its neighbours); none where no neighbour is below.
        global_z_padding: the same tail above and below every level's faces.
        no_tail: levels without the global tails (their own dilation and z padding still apply).
        regions: explicit refinements ``((x0, y0, z0), (x1, y1, z1), size_um)`` (stack coordinates, um).
        material_cells: explicit target cell sizes (um) per material; required for materials that are
            conductive or whose permittivity is not positive over the band (metals), for which no
            automatic size is offered.
    """

    dilation: Mapping[str, Any] = field(default_factory=dict)
    z_padding: Mapping[str, float | tuple[float, float]] = field(default_factory=dict)
    global_dilation: float = 3.0
    global_z_padding: float = 3.0
    no_tail: Sequence[str] = ()
    regions: Sequence[tuple[tuple[float, float, float], tuple[float, float, float], float]] = ()
    material_cells: Mapping[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------- indices


def _wavelengths(wavelength: float | tuple[float, float], materials: Mapping[str, Any]) -> np.ndarray:
    """Sampled band (um): evenly in frequency, plus the pole resonances inside it."""
    if np.ndim(wavelength) == 0:
        lo = hi = float(wavelength)
    else:
        vals = tuple(wavelength)
        if len(vals) != 2:
            raise ValueError(f"wavelength must be a value or (min, max), got {wavelength!r}")
        lo, hi = float(vals[0]), float(vals[1])
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
        raise ValueError(f"wavelength must be finite and positive (um), as a value or (min, max); got {wavelength}")
    if lo == hi:
        return np.array([lo])
    samples = list(1.0 / np.linspace(1.0 / hi, 1.0 / lo, _BAND_SAMPLES))
    for m in materials.values():
        for pole in m.dispersion.poles if m.dispersion is not None else ():
            for w0 in set(pole.omega_0_axes):
                if w0 > 0 and lo < (lam := 2 * math.pi * _C0 / w0 * 1e6) < hi:
                    samples += [lam * (1 - 1e-3), lam, lam * (1 + 1e-3)]
    return np.unique(np.clip(samples, lo, hi))


class _Index(NamedTuple):
    """A material's refractive index over the sampled band."""

    #: (samples,) the largest index (over polarizations) per sampled wavelength
    n: np.ndarray
    #: the permittivity is not positive definite somewhere in the band, or the material conducts
    unsupported: bool


def _index(material: Any, lambdas: np.ndarray) -> _Index:
    """The refractive index of an fdtdx Material at the sampled wavelengths (um)."""
    eps_inf = material_tensor(material)
    n, bad = [], False
    for lam in lambdas:
        if material.dispersion is None:
            eps = eps_inf
        else:
            eps = np.asarray(
                material.dispersion.permittivity_tensor(2 * math.pi * _C0 / (lam * 1e-6), material.permittivity)
            )
        if not np.all(np.isfinite(eps)):
            bad = True
            n.append(0.0)
            continue
        if np.isrealobj(eps) and np.allclose(eps, eps.T):
            eig = np.linalg.eigvalsh(0.5 * (eps + eps.T))
        else:
            eig = np.linalg.eigvals(eps)
        roots = np.sqrt(eig.astype(np.complex128))
        n.append(max(float(np.max(np.real(roots))), 0.0))
        bad |= bool(np.any(np.real(eig) <= 0))
    cond = np.asarray(material.electric_conductivity, dtype=float)
    bad |= bool(not np.all(np.isfinite(cond)) or np.any(cond != 0))
    return _Index(np.asarray(n), bad)


class _Materials(NamedTuple):
    """Per material: its target cell size (um) and its index at the longest wavelength (for the tails)."""

    cell: dict[str, float]
    n_tail: dict[str, float]


def _materials(stack: Any, wavelength: Any, ppw: float, refine: Refine) -> _Materials:
    """Target cells and tail indices of the device's materials."""
    materials = {k: m for k, m in stack.materials.items() if k != _ANISOTROPY_SENTINEL}
    lambdas = _wavelengths(wavelength, materials)
    for name, size in refine.material_cells.items():
        if name not in materials:
            raise ValueError(f"mesh: material_cells names {name!r}, not a material of the device ({sorted(materials)})")
        sf = float(size)
        if not (math.isfinite(sf) and sf > 0):
            raise ValueError(f"mesh: material_cells[{name!r}] must be finite and positive, got {size}")
    cell, n_tail = {}, {}
    for name, m in materials.items():
        idx = _index(m, lambdas)
        n_tail[name] = float(idx.n[-1])
        if name in refine.material_cells:
            cell[name] = float(refine.material_cells[name])
        elif idx.unsupported:
            raise ValueError(
                f"mesh: material {name!r} conducts or has a permittivity that is not positive over the band; "
                f"give its cell size in Refine(material_cells={{{name!r}: ...}})"
            )
        else:
            cell[name] = float(np.min(lambdas / (np.maximum(idx.n, 1e-12) * ppw)))
    return _Materials(cell, n_tail)


def _decay_length(lam_max: float, n_high: float, n_low: float) -> float:
    """The shortest decay length (um) of an evanescent field from n_high into n_low < n_high."""
    return lam_max / (2 * math.pi * math.sqrt(n_high**2 - n_low**2))


# ---------------------------------------------------------------- the lateral plane


class _Plane(NamedTuple):
    """KLayout regions of the plane, on a grid of ``dbu`` um, relative to the box's lateral center."""

    dbu: float
    center: np.ndarray  # (2,) um
    half: np.ndarray  # (2,) um: half the box
    box: Any  # the box
    universe: Any  # the box grown by the reach of every refinement

    def region_box(self, lo: Sequence[float], hi: Sequence[float]) -> Any:
        """The region of an axis-aligned box (stack coordinates, um)."""
        import klayout.db as kdb

        (x0, y0), (x1, y1) = (np.asarray(p, dtype=float) - self.center for p in (lo, hi))
        return kdb.Region(kdb.Box(*(math.floor(v / self.dbu) for v in (x0, y0)), *(math.ceil(v / self.dbu)
                                                                                    for v in (x1, y1))))

    def extent(self, region: Any, axis: int) -> list[tuple[float, float]]:
        """The (lo, hi) stack coordinates (um) along x (0) or y (1) of each polygon of a region in the box."""
        out = []
        for poly in (region & self.box).each():
            bb = poly.bbox()
            lo, hi = (bb.left, bb.right) if axis == 0 else (bb.bottom, bb.top)
            out.append((float(self.center[axis] + lo * self.dbu), float(self.center[axis] + hi * self.dbu)))
        return out


def _box(stack: Any) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """(lo, hi) of the device's box along x, y, z (stack coordinates, um)."""
    if stack.partial_real_shape is None or None in tuple(stack.partial_real_shape):
        raise ValueError(f"StackDevice {stack.name!r} has no box (partial_real_shape)")
    middle = (float(stack.center[0]), float(stack.center[1]), float(stack.z_center))
    size = [float(s) * 1e6 for s in stack.partial_real_shape]
    return tuple((m - s / 2, m + s / 2) for m, s in zip(middle, size, strict=True))  # type: ignore[return-value]


def _plane(stack: Any, reach: float) -> _Plane:
    """The plane of the device's box, its regions reaching ``reach`` um past the box."""
    import klayout.db as kdb

    (x0, x1), (y0, y1), _ = _box(stack)
    center, half = np.array([(x0 + x1) / 2, (y0 + y1) / 2]), np.array([(x1 - x0) / 2, (y1 - y0) / 2])
    extent = max(float(np.abs(np.concatenate([np.asarray(v, dtype=float) - center for v in stack.source_vertices]
                                             or [np.zeros((1, 2))])).max()), *(half + reach)) + reach
    dbu = max(_CHECK_DBU, 4 * extent / 2**31)

    def rect(h: np.ndarray) -> Any:
        x, y = (math.ceil(float(c) / dbu) for c in h)
        return kdb.Region(kdb.Box(-x, -y, x, y))

    return _Plane(dbu, center, half, rect(half), rect(half + reach))


def _sized(region: Any, grow: float, dbu: float, upper: bool) -> Any:
    """A region grown by ``grow`` um (negative shrinks), bounding the round offset from outside (``upper``)
    or from inside."""
    outer = (grow > 0) == upper
    g = (grow if outer else grow * _INNER) / dbu
    d = math.ceil(g) if upper else math.floor(g)
    return region.sized(d, d, _MODE) if d else region.dup()


def _layer_regions(stack: Any, plane: _Plane, reach: float) -> tuple[dict[int, Any], dict[int, list[np.ndarray]]]:
    """KLayout regions of the source layers and the derived shapes' parts, and the source polygons (um,
    relative to the box's center). Port extensions that end near a box face continue past it."""
    extensions = stack.source_extensions or tuple(() for _ in stack.source_layers)
    polygons: dict[int, list[np.ndarray]] = {}
    for (_, index), v, sizes, exts in zip(
        stack.source_layers, stack.source_vertices, stack.source_sizes, extensions, strict=True
    ):
        polys = np.split(np.asarray(v, dtype=float) - plane.center, np.cumsum(sizes)[:-1])
        for rect in polys[len(polys) - len(exts) :]:
            along = rect[1] - rect[0]  # corners 0, 3 on the port face, 1, 2 at the end
            length = float(np.linalg.norm(along))
            if not length > 0:
                continue
            n, far = along / length, 0.5 * (rect[1] + rect[2])
            axes = np.abs(n) > 1e-12
            exit_ = np.min((np.sign(n[axes]) * plane.half[axes] - far[axes]) / n[axes])  # to the face ahead
            if exit_ < reach:
                rect[1:3] += (2 * float(plane.half.sum()) + 2 * reach) * n
        polygons[index] = polys
    import klayout.db as kdb

    regions = {index: _kdb_region(polys, plane.dbu) & plane.universe for index, polys in polygons.items()}
    for k, d in enumerate(stack.derived_layers):
        for j, r in enumerate(d.recipes):
            if len(r.kind):
                pool = np.concatenate([np.concatenate(polygons[index]) for index in d.layers])
                rings = recipe_rings(r, run_recipe(r.kind, r.index, r.positions, pool, np))
                sized_rings = [(abs(a), a, q) for q in rings if len(q) >= 3 and (a := _ring_area(q)) != 0.0]
                region = kdb.Region()
                for _, area, q in sorted(sized_rings, key=lambda item: item[0], reverse=True):
                    poly_reg = _kdb_region([q], plane.dbu)
                    region = region + poly_reg if area > 0 else region - poly_reg
                regions[_part_index(k, j)] = region & plane.universe
    return regions, polygons


def _bound(tree: tuple, literals: Sequence[tuple[int, int, float]], regions: dict[int, Any], s: float,
           plane: _Plane, upper: bool) -> Any:
    """A level's shape at lateral offset s (um, positive shrinks), bounded from outside (``upper``) or inside
    (see :func:`bifrost.stack._literals`)."""
    import klayout.db as kdb

    kind = tree[0]
    if kind == "lit":
        index, a, b = literals[tree[1]]
        return _sized(regions.get(index, kdb.Region()), -(a * s + b), plane.dbu, upper)
    if kind in ("true", "false"):
        return plane.universe.dup() if kind == "true" else kdb.Region()
    if kind == "neg":
        return plane.universe - _bound(tree[1], literals, regions, s, plane, not upper)
    x = _bound(tree[1], literals, regions, s, plane, upper)
    y = _bound(tree[2], literals, regions, s, plane, upper)
    return x & y if kind == "and" else x | y


class _LevelGeom(NamedTuple):
    """A level's lateral extent over its whole height."""

    name: str
    material: str
    rank: int  # painting order: lower wins
    span: tuple[float, float]  # nominal (bottom, top), um
    reads: frozenset[int]  # source layer indices it is drawn from
    upper: Any  # region containing its shape at every height
    lower: Any  # region inside its shape at every height
    grow: float  # largest lateral offset (um) of its layers over its height
    thickness: float  # nominal |thickness| (um)


def _level_geoms(stack: Any, plane: _Plane, regions: dict[int, Any]) -> list[_LevelGeom]:
    """Every level's widest and narrowest cross-sections: its offset is piecewise linear in z, so they are at
    its bottom, its top or a z_to_bias knot."""
    source = {index: i for i, (_, index) in enumerate(stack._slots())}
    rank = {i: r for r, i in enumerate(sorted(range(len(stack.levels)), key=lambda i: (stack.levels[i].mesh_order, i)))}
    out = []
    for i, lv in enumerate(stack.levels):
        expr = stack._drawn_expr(lv)
        tree, literals = _literals(expr, 1.0, source)
        lo, hi = _span(lv)
        if expr[0] == "fill":
            offsets = np.zeros(1)
        else:
            knots = [lo, hi]
            if lv.z_to_bias is not None:
                knots += [lv.zmin + f * lv.thickness for f in lv.z_to_bias[0]]
            offsets = _wall_offset(lv, np.clip(knots, lo, hi))
        s_lo, s_hi = float(offsets.min()), float(offsets.max())
        reads: set[int] = set()
        for index, _, _ in literals:
            if index >= 0:
                reads.add(index)
            else:
                reads.update(stack.derived_layers[(-1 - index) // 1024].layers)
        grow = max([0.0, *(abs(a * s + b) for _, a, b in literals for s in (s_lo, s_hi))])
        out.append(_LevelGeom(
            lv.name, lv.material, rank[i], (lo, hi), frozenset(reads),
            _bound(tree, literals, regions, s_lo, plane, True), _bound(tree, literals, regions, s_hi, plane, False),
            grow, hi - lo,
        ))
    return out


# ---------------------------------------------------------------- where the geometry may move


def _layer_dilations(stack: Any, refine: Refine) -> dict[int, np.ndarray]:
    """Per moving source layer (index), the dilation (um) of each of its vertices."""
    by_name = {}
    for (key, index), v in zip(stack.source_layers, stack.source_vertices, strict=True):
        by_name[key] = by_name[key.split("/", 1)[-1]] = (index, len(v))
    out = {}
    for name, d in refine.dilation.items():
        if name not in by_name:
            names = sorted(k for k in by_name if not k.startswith("vertices/"))
            raise ValueError(f"Refine.dilation names {name!r}, not a source layer of the device ({names})")
        index, n = by_name[name]
        d = np.asarray(d, dtype=float)
        if d.ndim == 0:
            d = np.full(n, float(d))
        if d.shape != (n,):
            raise ValueError(f"Refine.dilation[{name!r}] must be a scalar or {n} values (one per vertex), got {d.shape}")
        if not np.all(np.isfinite(d)) or np.any(d < 0):
            raise ValueError(f"Refine.dilation[{name!r}] must be finite and nonnegative")
        if np.any(d > 0):
            out[index] = d
    return out


def _level_padding(stack: Any, refine: Refine) -> dict[str, tuple[float, float]]:
    """Per level, how far (um) below and above its nominal span its faces may move."""
    names = [lv.name for lv in stack.levels]
    out = {}
    for name, pad in refine.z_padding.items():
        if name not in names:
            raise ValueError(f"Refine.z_padding names {name!r}, not a level of the device ({names})")
        if np.ndim(pad) == 0:
            below = above = float(pad)
        else:
            vals = tuple(pad)
            if len(vals) != 2:
                raise ValueError(f"Refine.z_padding[{name!r}] must be a float or (below, above), got {pad!r}")
            below, above = float(vals[0]), float(vals[1])
        if not (math.isfinite(below) and math.isfinite(above) and below >= 0 and above >= 0):
            raise ValueError(f"Refine.z_padding[{name!r}] must be finite and nonnegative, got {pad}")
        out[name] = (below, above)
    return out


def _roughness_padding(stack: Any) -> dict[str, tuple[float, float]]:
    """Per level, conservative z margin (+/- 5 * rms, um) from TopRoughness interfaces."""
    tops = {lv.name: max(lv.zmin, lv.zmin + lv.thickness) for lv in stack.levels}
    out: dict[str, list[float]] = {}
    for r in getattr(stack, "roughness", ()):
        rms = float(r.rms)
        if not (math.isfinite(rms) and rms >= 0):
            raise ValueError(f"TopRoughness.rms must be finite and nonnegative, got {r.rms}")
        pad = 5.0 * rms
        if pad <= 0 or r.level not in tops:
            continue
        cur = out.setdefault(r.level, [0.0, 0.0])
        cur[1] = max(cur[1], pad)
        if r.fill_above:
            top_z = tops[r.level]
            for lv in stack.levels:
                if abs(min(lv.zmin, lv.zmin + lv.thickness) - top_z) < 1e-6:
                    cur_a = out.setdefault(lv.name, [0.0, 0.0])
                    cur_a[0] = max(cur_a[0], pad)
    return {k: (v[0], v[1]) for k, v in out.items()}


def _capsules(polys: list[np.ndarray], radii: np.ndarray, dbu: float) -> Any:
    """The region within ``radii`` (per vertex, linear along each edge) of the polygons' edges: the union of
    the convex hulls of octagons around each edge's two ends (an outside bound of the round band)."""
    import klayout.db as kdb

    angles = (np.arange(8) + 0.5) * (math.pi / 4)
    unit = np.column_stack([np.cos(angles), np.sin(angles)]) * _OCTAGON
    region = kdb.Region()
    start = 0
    for p in polys:
        r = radii[start : start + len(p)]
        start += len(p)
        for i in range(len(p)):
            j = (i + 1) % len(p)
            if r[i] <= 0 and r[j] <= 0:
                continue
            pts = np.concatenate([p[i] + unit * r[i], p[j] + unit * r[j]]) / dbu
            hull = _convex_hull(pts)
            region.insert(kdb.Polygon([kdb.Point(math.floor(x) if x < 0 else math.ceil(x),
                                                 math.floor(y) if y < 0 else math.ceil(y)) for x, y in hull]))
    return region.merged()


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Convex hull (counterclockwise) of 2D points (Andrew's monotone chain)."""
    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) <= 2:
        return np.asarray(pts)

    def cross(o: tuple, a: tuple, b: tuple) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


# ---------------------------------------------------------------- refinement contributions


class _Contribution(NamedTuple):
    """Cells of at most ``size`` um over a lateral region (plane frame) times a z interval (um)."""

    region: Any
    z: tuple[float, float]
    size: float
    what: str  # e.g. "material core", "tail core", "feature core", "region", "background"


class _Geometry(NamedTuple):
    """What :func:`mesh` refines, before the axes are placed."""

    plane: _Plane
    box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    contributions: list[_Contribution]
    levels: list[_LevelGeom]
    possible: dict[str, Any]  # per level: where its material may be (plane frame), over its padded span
    padded: dict[str, tuple[float, float]]  # per level: its span with its z padding
    tails: dict[str, tuple[float, float, float]]  # per level: decay lengths (lateral, below, above), 0: none
    cell: dict[str, float]  # target cell per material
    moving: dict[int, np.ndarray]  # per moving source layer: dilation per vertex
    neighbours: dict[str, tuple[set[str], set[str], set[str]]]  # per level: materials beside, below, above it
    index: dict[str, float]  # per material: its index at the longest wavelength
    rough_faces: tuple[float, ...] = ()  # z (um) of TopRoughness-perturbed interfaces


def _geometry(
    stack: Any,
    wavelength: Any,
    ppw: float,
    refine: Refine,
    feature_cells: float = 1.5,
    min_cell: float | None = None,
) -> _Geometry:
    """Every refinement contribution of a device (see the module docstring)."""
    import klayout.db as kdb

    ppw_f = float(ppw)
    if not (math.isfinite(ppw_f) and ppw_f > 0):
        raise ValueError(f"ppw must be finite and positive, got {ppw}")
    fc_f = float(feature_cells)
    if not (math.isfinite(fc_f) and fc_f >= 0):
        raise ValueError(f"feature_cells must be finite and nonnegative, got {feature_cells}")
    if min_cell is not None:
        mc_f = float(min_cell)
        if not (math.isfinite(mc_f) and mc_f > 0):
            raise ValueError(f"min_cell must be finite and positive, got {min_cell}")
    gd = float(refine.global_dilation)
    gz = float(refine.global_z_padding)
    if not (math.isfinite(gd) and gd >= 0):
        raise ValueError(f"Refine.global_dilation must be finite and nonnegative, got {refine.global_dilation}")
    if not (math.isfinite(gz) and gz >= 0):
        raise ValueError(f"Refine.global_z_padding must be finite and nonnegative, got {refine.global_z_padding}")
    mats = _materials(stack, wavelength, ppw_f, refine)
    lam_max = float(np.max(np.atleast_1d(wavelength)))
    moving = _layer_dilations(stack, refine)
    padding = _level_padding(stack, refine)
    rough = _roughness_padding(stack)
    tops = {lv.name: max(lv.zmin, lv.zmin + lv.thickness) for lv in stack.levels}
    rough_faces = tuple(tops[r.level] for r in getattr(stack, "roughness", ()) if r.level in tops and float(r.rms) > 0)
    no_tail = set(refine.no_tail)
    unknown = no_tail - {lv.name for lv in stack.levels}
    if unknown:
        raise ValueError(f"Refine.no_tail names {sorted(unknown)}, not levels of the device")
    # how far refinements reach past the box: dilations, offsets and the longest possible tail
    n_values = sorted(set(mats.n_tail.values()))
    longest = max([0.0, *(_decay_length(lam_max, b, a) for a in n_values for b in n_values if b > a)])
    tail_reach = max(gd, gz, 0.0) * longest
    reach = max([0.0, *(float(d.max()) for d in moving.values())]) + tail_reach + 1.0
    plane = _plane(stack, reach)
    regions, polygons = _layer_regions(stack, plane, reach)
    levels = _level_geoms(stack, plane, regions)
    box = _box(stack)
    finest = min(mats.cell.values())

    # moving bands: within the dilation (plus the level's lateral offsets) of a moving layer's edges
    capsules = {index: _capsules(polygons[index], d, plane.dbu) for index, d in moving.items()}
    padded = {
        lv.name: (
            lv.span[0] - padding.get(lv.name, (0.0, 0.0))[0] - rough.get(lv.name, (0.0, 0.0))[0],
            lv.span[1] + padding.get(lv.name, (0.0, 0.0))[1] + rough.get(lv.name, (0.0, 0.0))[1],
        )
        for lv in levels
    }
    static = {
        lv.name: not (lv.reads & moving.keys()) and lv.name not in padding and lv.name not in rough
        for lv in levels
    }
    possible: dict[str, Any] = {}
    bands: dict[str, Any] = {}
    for lv in levels:
        band = kdb.Region()
        for index in lv.reads & moving.keys():
            band |= _sized(capsules[index], lv.grow, plane.dbu, True)
        bands[lv.name] = band & plane.universe
        possible[lv.name] = (lv.upper | bands[lv.name]) & plane.universe

    # occlusion, per z interval: only by winners that never move (laterally or in z), at their narrowest,
    # eroded by the finest cell
    breaks = np.unique(np.concatenate([[box[2][0], box[2][1]], *(np.asarray(padded[lv.name]) for lv in levels)]))
    contributions: list[_Contribution] = []
    visible: dict[str, Any] = {}
    for lv in levels:
        lo, hi = padded[lv.name]
        here = kdb.Region()
        cuts = [z for z in breaks if lo < z < hi]
        for a, b in zip([lo, *cuts], [*cuts, hi], strict=True):
            covered = kdb.Region()
            for w in levels:
                if w.rank < lv.rank and static[w.name] and w.span[0] <= a and b <= w.span[1]:
                    covered |= _sized(w.lower, -finest, plane.dbu, False)
            region = possible[lv.name] - covered
            if not region.is_empty():
                contributions.append(_Contribution(region, (a, b), mats.cell[lv.material], f"material {lv.name}"))
                here |= region
        visible[lv.name] = here

    # tails: past each boundary toward neighbours of a lower index, in units of the boundary's decay length
    tails: dict[str, tuple[float, float, float]] = {}
    neighbours: dict[str, tuple[set[str], set[str], set[str]]] = {}
    for lv in levels:
        tails[lv.name] = (0.0, 0.0, 0.0)
        neighbours[lv.name] = (set(), set(), set())
        if visible[lv.name].is_empty():
            continue
        n_high = mats.n_tail[lv.material]
        mine, (lo, hi) = visible[lv.name], padded[lv.name]
        # neighbours in the box only: past it a level's edge is where the plane ends, not a boundary
        ring = (_sized(mine, _RING, plane.dbu, True) - mine) & plane.box
        inside = mine & plane.box
        lateral = _neighbours(levels, visible, padded, lv, ring, (lo, hi), stack.background)
        below = _neighbours(levels, visible, padded, lv, inside, (lo - 1e-6, lo - 1e-6), stack.background)
        above = _neighbours(levels, visible, padded, lv, inside, (hi + 1e-6, hi + 1e-6), stack.background)
        neighbours[lv.name] = (lateral, below, above)
        if lv.name in no_tail:
            continue
        lengths = []
        for names in (lateral, below, above):
            lows = [mats.n_tail[m] for m in names if mats.n_tail[m] < n_high]
            lengths.append(_decay_length(lam_max, n_high, max(lows)) if lows else 0.0)
        tails[lv.name] = tuple(lengths)  # type: ignore[assignment]
        size = mats.cell[lv.material]
        if lengths[0] > 0 and refine.global_dilation > 0:
            grown = _sized(mine, refine.global_dilation * lengths[0], plane.dbu, True) & plane.universe
            contributions.append(_Contribution(grown, (lo, hi), size, f"tail {lv.name}"))
        z_lo = lo - refine.global_z_padding * lengths[1]
        z_hi = hi + refine.global_z_padding * lengths[2]
        if z_lo < lo or z_hi > hi:
            contributions.append(_Contribution(mine, (z_lo, z_hi), size, f"tail {lv.name}"))

    # narrow parts and gaps of the starting geometry, and thin levels
    if feature_cells > 0:
        for lv in levels:
            if visible[lv.name].is_empty() or lv.upper.is_empty():
                continue
            limit = feature_cells * mats.cell[lv.material]
            d = max([0.0, *(float(moving[i].max()) for i in lv.reads & moving.keys())])
            for shape, check in ((lv.lower, "width_check"), (lv.upper, "space_check")):
                # on the shapes past the box (no artificial parts where the box cuts them), kept where in it
                pairs = getattr(shape, check)(math.ceil(limit / plane.dbu), ignore_angle=_CHECK_ANGLE)
                for ep in pairs.each():
                    w = ep.distance() * plane.dbu
                    if 0 < w < limit and not (kdb.Region(ep.bbox()) & plane.box).is_empty():
                        spot = _sized(kdb.Region(ep.bbox()), d, plane.dbu, True)
                        contributions.append(_Contribution(spot, padded[lv.name], w / feature_cells,
                                                           f"feature {lv.name}"))
            if 0 < lv.thickness < limit:
                contributions.append(_Contribution(visible[lv.name], padded[lv.name], lv.thickness / feature_cells,
                                                   f"thin {lv.name}"))

    for region in refine.regions:
        if len(region) != 3:
            raise ValueError(f"Refine.regions: each region must be (lo3, hi3, size), got {region!r}")
        lo3, hi3, size = region
        if len(lo3) != 3 or len(hi3) != 3:
            raise ValueError(f"Refine.regions: bounds must each have 3 coordinates, got {(lo3, hi3)!r}")
        lo_f = tuple(float(v) for v in lo3)
        hi_f = tuple(float(v) for v in hi3)
        sz = float(size)
        if not (all(math.isfinite(v) for v in (*lo_f, *hi_f)) and math.isfinite(sz) and sz > 0):
            raise ValueError(f"Refine.regions: bounds and positive size must be finite, got {region!r}")
        if not all(a <= b for a, b in zip(lo_f, hi_f, strict=True)):
            raise ValueError(f"Refine.regions: expected lo <= hi along x, y, z, got {lo_f} and {hi_f}")
        contributions.append(
            _Contribution(plane.region_box(lo_f[:2], hi_f[:2]), (lo_f[2], hi_f[2]), sz, "region")
        )
    contributions.append(_Contribution(plane.box.dup(), box[2], mats.cell[stack.background], "background"))
    if min_cell is not None:
        contributions = [c._replace(size=max(c.size, float(min_cell))) for c in contributions]
    return _Geometry(
        plane, box, contributions, levels, visible, padded, tails, mats.cell, moving, neighbours, mats.n_tail,
        rough_faces,
    )


def _neighbours(levels: list[_LevelGeom], visible: dict[str, Any], padded: dict[str, tuple[float, float]],
                lv: _LevelGeom, where: Any, z: tuple[float, float], background: str) -> set[str]:
    """Materials of the other levels that may be in ``where`` within the z range, and the background if
    some of ``where`` may be bare."""
    names, bare = set(), where.dup()
    if where.is_empty():
        return names
    for other in levels:
        if other.name == lv.name:
            continue
        lo, hi = padded[other.name]
        # spans that only touch (a core on a substrate) are not lateral neighbours; a height is in a closed span
        meets = (min(hi, z[1]) > max(lo, z[0])) if z[1] > z[0] else lo <= z[0] <= hi
        if meets and not (visible[other.name] & where).is_empty():
            names.add(other.material)
            bare -= visible[other.name]
    if not bare.is_empty():
        names.add(background)
    return names


def _axis_targets(geom: _Geometry) -> tuple[list[tuple[float, float, float]], ...]:
    """Per axis, the (a, b, size) intervals of every contribution within the box: the tensor product takes, at
    each coordinate, the smallest target anywhere in the plane through it."""
    (_, _), (_, _), (z0, z1) = geom.box
    out: tuple[list, list, list] = ([], [], [])
    for c in geom.contributions:
        a, b = max(c.z[0], z0), min(c.z[1], z1)
        if b < a or (c.region & geom.plane.box).is_empty():
            continue
        for axis in (0, 1):
            out[axis].extend((lo, hi, c.size) for lo, hi in geom.plane.extent(c.region, axis))
        out[2].append((a, b, c.size))
    return out


# ---------------------------------------------------------------- the mesh

#: the six faces in fdtdx's BoundaryConfig order
FACES = ("minx", "maxx", "miny", "maxy", "minz", "maxz")
_KINDS = ("pml", "periodic", "pec", "pmc", "bloch")
# relative slack of the realized (float) grid's checks
_REALIZED_RTOL = 1e-3


@dataclass(frozen=True, eq=False)
class Mesh:
    """The cell edges :func:`mesh` built for a device's box.

    Attributes:
        edges: per axis, the (n + 1,) cell edges (um, stack coordinates) from the box's lower face to its
            upper one.
        pml_cells: PML cells at each face, in :data:`FACES` order (0 where the face has none).
        boundaries: the boundary kind of each face, in :data:`FACES` order.
        axes: per axis, the :class:`~bifrost.AxisEdges` (smallest cell, alignment, symmetry).
        growth: the largest ratio of neighbouring cells.
        stack: the device it was built for.
        geometry: the refinements it was built from (for plots and the coarse-vertex report).
    """

    edges: tuple[np.ndarray, np.ndarray, np.ndarray]
    pml_cells: tuple[int, int, int, int, int, int]
    boundaries: tuple[str, str, str, str, str, str]
    axes: tuple[AxisEdges, AxisEdges, AxisEdges]
    growth: float
    stack: Any = field(repr=False)
    geometry: _Geometry = field(repr=False)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Cells along x, y and z."""
        return tuple(len(e) - 1 for e in self.edges)  # type: ignore[return-value]

    @property
    def grid(self) -> Any:
        """The fdtdx ``RectilinearGrid`` over the device's box (in the device's dtype), checked once more in
        that precision: increasing edges and neighbouring cells within ``growth``."""
        grid = self.stack.rectilinear_grid(min(a.min_cell for a in self.axes), *self.edges)
        for name, e in zip("xyz", (grid.x_edges, grid.y_edges, grid.z_edges), strict=True):
            w = np.diff(np.asarray(e, dtype=np.float64))
            ratio = float(np.max(np.maximum(w[1:] / w[:-1], w[:-1] / w[1:]))) if len(w) > 1 else 1.0
            if not (np.all(w > 0) and ratio <= self.growth * (1 + _REALIZED_RTOL)):
                raise ValueError(
                    f"mesh: the {name} edges do not survive the device's dtype (largest ratio {ratio:.6g}); use a "
                    "float64 device or a larger min_cell"
                )
        return grid

    @property
    def boundary(self) -> Any:
        """The fdtdx ``BoundaryConfig``: the PML cells of each PML face and the other faces' kinds."""
        import fdtdx

        kwargs: dict[str, Any] = {}
        for face, kind, cells in zip(FACES, self.boundaries, self.pml_cells, strict=True):
            kwargs[f"boundary_type_{face}"] = kind
            kwargs[f"thickness_grid_{face}"] = cells if kind == "pml" else 1
        return fdtdx.BoundaryConfig(**kwargs)

    def summary(self, time: float | None = None, courant_factor: float = 0.99) -> str:
        """Cells, cell sizes, the time step and the field memory, as text.

        Args:
            time: simulated time (s), for the number of time steps.
            courant_factor: fdtdx's ``SimulationConfig.courant_factor``.
        """
        grid = self.grid
        dt = float(grid.cfl_time_step(courant_factor))
        cells = int(np.prod(self.shape))
        nbytes = np.dtype(self.stack.dtype).itemsize
        lines = [f"mesh of {self.stack.name!r}: {self.shape[0]} x {self.shape[1]} x {self.shape[2]} = {cells:,} cells"]
        for name, e, ax in zip("xyz", self.edges, self.axes, strict=True):
            w = np.diff(e) * 1e3
            ratio = float(np.max(np.maximum(w[1:] / w[:-1], w[:-1] / w[1:]))) if len(w) > 1 else 1.0
            flags = ", ".join(f for f, on in (("aligned", ax.aligned), ("mirror-symmetric", ax.symmetric),
                                              (f"faces centered at {", ".join(f"{c:g}" for c in ax.centered)} um",
                                               bool(ax.centered))) if on)
            lines.append(
                f"  {name}: {len(w)} cells, {w.min():.4g} to {w.max():.4g} nm, largest ratio {ratio:.3f}, PML cells "
                f"{ax.pml_cells[0]} / {ax.pml_cells[1]}" + (f" ({flags})" if flags else "")
            )
        steps = f", {math.ceil(time / dt):,} steps" if time is not None else ""
        lines.append(f"  time step {dt * 1e18:.4g} as{steps}")
        lines.append(f"  fields (E and H, {nbytes * 8}-bit): {6 * cells * nbytes / 1e9:.3g} GB")
        report = coarse_vertices(self.stack, self.stack.init_params(), self)
        outside = int(np.sum(report.outside))
        outside_note = f", {outside} outside box" if outside else ""
        lines.append(
            f"  starting geometry: {int(np.sum(report.coarse))} of {int(np.sum(report.total))} vertices in cells "
            f"wider than their target (largest cell / target {float(report.worst):.3f}{outside_note})"
        )
        if outside:
            lines.append(f"  warning: {outside} vertices lie outside the mesh box")
        return "\n".join(lines)


def mesh(
    stack: Any,
    wavelength: float | tuple[float, float],
    *,
    ppw: float = 20.0,
    refine: Refine | None = None,
    boundaries: str | Sequence[str] = "pml",
    pml: float | Sequence[float] = 1.0,
    growth: float = 1.25,
    quantum: float | None = None,
    min_cell: float | None = None,
    align_fine: bool = True,
    center_faces: bool = True,
    feature_cells: float = 1.5,
    max_states: float = 2e7,
) -> Mesh:
    """Cell edges over a device's box, fine wherever its materials are or may come to be.

    Every place a level's material is, or may come to be (its shape, the bands of its moving layers, its
    z padding), asks for cells of at most ``lambda / (n ppw)``; so do the evanescent tails past its
    boundaries (``refine.global_dilation`` and ``global_z_padding``, in units of each boundary's decay
    length), narrow parts and gaps of the starting geometry (``feature_cells`` cells across), and
    ``refine.regions``. Each axis takes, at each coordinate, the smallest target anywhere in the plane
    through it, and :func:`~bifrost.axis_edges` places continuous real-valued edges along it: maximizing
    the smallest cell, keeping cell counts minimal, smoothing transitions, and covering each PML with
    uniform cells (or :func:`~bifrost.lattice_edges` on a discrete lattice when ``quantum`` is given). The
    result is fixed: build it once, then run or optimize on ``mesh.grid``::

        m = bf.mesh(stack, (1.50, 1.60), refine=bf.Refine(dilation={"WG": 0.45}))
        config = stack.get_simulation_config(time=..., grid=m.grid)
        boundaries, constraints = fdtdx.boundary_objects_from_config(m.boundary, volume)

    Args:
        stack: the device (placed or not); its box is meshed.
        wavelength: wavelength (um), or the band (min, max).
        ppw: cells per wavelength in each material. 20 (the default) gives the inverse-design bend's
            transmission within 4e-4 of a uniform grid of the core's cells (benchmarks/rectilinear); a cladding or
            a substrate that spans every plane caps the lateral cells everywhere, so coarsen it with
            ``Refine.material_cells`` (13 to 14 cells per wavelength suffice there). 14 everywhere is about a
            uniform grid of half again the core's cells: for fast optimizations.
        refine: where the geometry may move, and the tails (:class:`Refine`); default ``Refine()``.
        boundaries: the kind of each face ("pml", "periodic", "pec", "pmc" or "bloch"), one for all or six
            in :data:`FACES` order.
        pml: PML thickness (um), one for all or six; uniform cells cover it at each PML face.
        growth: largest ratio of neighbouring cells.
        quantum: default ``None`` places continuous real-valued edges via :func:`~bifrost.axis_edges` (any
            box size is accepted). Passing a positive step (um, e.g. ``0.001``) places edges on whole
            multiples of ``quantum`` via :func:`~bifrost.lattice_edges` and requires the box to divide it.
        min_cell: no target finer than this (um): protects the time step from thin films or narrow gaps.
        align_fine: put each axis's finest cells on the lattice of their own width from the box's face when
            compatible with maximizing the smallest cell and mirror symmetry.
        center_faces: put every static level face where the material changes at the exact middle of a z
            cell (not on an edge): a markedly smaller error with subpixel averaging (a third, on the
            coupler of benchmarks/rectilinear). The smallest z cell may come out up to 20 % below the largest
            possible (two faces a fractional number of cells apart need some narrower cells); a face that still
            finds no room is left off with a warning, and faces under a PML are left out. Faces moved by
            ``Refine.z_padding`` are left uncentered so vertical derivatives do not start on a dual-cell
            boundary.
        feature_cells: cells across the narrowest part or gap of each level (0: no such targets).
        max_states: state budget forwarded to :func:`~bifrost.lattice_edges` when ``quantum`` is not ``None``.

    Returns:
        The :class:`Mesh`.

    Raises:
        ValueError: on an explicit ``quantum`` that does not divide the box, an unknown boundary kind, a
            level that may move into the box (z padding) but that placing the device would drop
            (``level_margin``), a material without an automatic size (see ``Refine.material_cells``), or
            targets an axis cannot meet.
    """
    refine = Refine() if refine is None else refine
    kinds = (boundaries,) * 6 if isinstance(boundaries, str) else tuple(boundaries)
    if len(kinds) != 6 or any(k not in _KINDS for k in kinds):
        raise ValueError(f"mesh: boundaries must be one or six of {_KINDS}, got {boundaries!r}")
    for a in range(3):
        k0, k1 = kinds[2 * a], kinds[2 * a + 1]
        if (k0 in ("periodic", "bloch") or k1 in ("periodic", "bloch")) and k0 != k1:
            raise ValueError(
                f"mesh: periodic/bloch boundaries must match on both faces of {'xyz'[a]}, got {(k0, k1)!r}"
            )
    if np.ndim(pml) == 0:
        thick = (float(pml),) * 6
    else:
        thick = tuple(float(t) for t in pml)  # type: ignore[arg-type]
    if len(thick) != 6 or any(not math.isfinite(t) or t < 0 for t in thick):
        raise ValueError(f"mesh: pml must be one or six finite nonnegative thicknesses (um), got {pml!r}")
    growth_f = float(growth)
    if not (math.isfinite(growth_f) and growth_f > 1):
        raise ValueError(f"mesh: growth must be finite and > 1, got {growth!r}")
    max_states_f = float(max_states)
    if not (math.isfinite(max_states_f) and max_states_f > 0):
        raise ValueError(f"mesh: max_states must be finite and positive, got {max_states!r}")
    geom = _geometry(stack, wavelength, ppw, refine, feature_cells, min_cell)
    _check_levels_kept(stack, geom)
    targets = _axis_targets(geom)
    faces = _interface_faces(geom, kinds, thick) if center_faces else []
    axes = []
    for a in range(3):
        lo, hi = geom.box[a]
        sides = tuple(thick[2 * a + e] if kinds[2 * a + e] == "pml" else 0.0 for e in (0, 1))
        is_periodic = kinds[2 * a] in ("periodic", "bloch")
        if quantum is not None:
            quantum = float(quantum)
            if not (math.isfinite(quantum) and quantum > 0):
                raise ValueError(f"mesh: quantum must be positive (um) or None, got {quantum!r}")
            n = round((hi - lo) / quantum)
            if abs(n * quantum - (hi - lo)) > 1e-6 * quantum:
                raise ValueError(
                    f"mesh: the box is {hi - lo:.9g} um along {'xyz'[a]}, not a whole number of {quantum:g} um quanta; "
                    f"move a face by {(math.ceil((hi - lo) / quantum) * quantum - (hi - lo)):.6g} um (boundary_offset) "
                    "or pass quantum=None"
                )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                if quantum is None:
                    axis = axis_edges(
                        lo, hi, max(s for *_, s in targets[a]), targets[a], growth=growth_f, pml=sides,
                        align=align_fine, centers=faces if a == 2 else (), periodic=is_periodic, dtype=stack.dtype,
                    )
                else:
                    axis = lattice_edges(
                        lo, hi, max(s for *_, s in targets[a]), targets[a], quantum=quantum, growth=growth_f, pml=sides,
                        align=align_fine, centers=faces if a == 2 else (), max_states=max_states_f,
                    )
            except ValueError as e:
                raise ValueError(f"mesh: the {'xyz'[a]} axis: {e}") from e
        for w in caught:
            if "do not all start" in str(w.message):
                continue  # alignment is a preference (summary() reports it): ppw rarely gives a width the box divides
            warnings.warn(f"mesh: the {'xyz'[a]} axis: {w.message}", w.category, stacklevel=2)
        if is_periodic:
            w = np.diff(axis.edges)
            if max(w[0] / w[-1], w[-1] / w[0]) > growth * (1 + 1e-9):
                warnings.warn(
                    f"mesh: the {'xyz'[a]} axis's end cells ({w[0] * 1e3:.4g} and {w[-1] * 1e3:.4g} nm) differ by more "
                    f"than growth across its {kinds[2 * a]} boundary",
                    stacklevel=2,
                )
        axes.append(axis)
    pml_cells = tuple(axes[a].pml_cells[e] for a in range(3) for e in (0, 1))
    return Mesh(tuple(ax.edges for ax in axes), pml_cells, kinds, tuple(axes), float(growth), stack, geom)  # type: ignore[arg-type]


def _interface_faces(geom: _Geometry, kinds: Sequence[str], thick: Sequence[float]) -> list[float]:
    """The z (um) of every visible level's static bottom and top face where the material changes, inside the
    box and clear of the PMLs."""
    (_, _), (_, _), (z0, z1) = geom.box
    lo = z0 + (thick[4] if kinds[4] == "pml" else 0.0)
    hi = z1 - (thick[5] if kinds[5] == "pml" else 0.0)
    moving: list[float] = [float(z) for z in geom.rough_faces]
    raw: list[float] = []
    for lv in geom.levels:
        if geom.possible[lv.name].is_empty():
            continue
        plo, phi = geom.padded[lv.name]
        if plo < lv.span[0] - 1e-12:
            moving.append(float(lv.span[0]))
        if phi > lv.span[1] + 1e-12:
            moving.append(float(lv.span[1]))
        _, below, above = geom.neighbours[lv.name]
        n = geom.index.get(lv.material)
        for face, across in ((lv.span[0], below), (lv.span[1], above)):
            if any(abs(geom.index.get(m, -1.0) - n) > 1e-9 for m in across) and lo < face < hi:
                raw.append(float(face))
    tol = 1e-12 * max(1.0, abs(z0), abs(z1))
    faces: list[float] = []
    for face in sorted(raw):
        if any(abs(face - m) <= 1e-9 for m in moving):
            continue
        if not faces or face - faces[-1] > tol:
            faces.append(face)
    return faces


def _check_levels_kept(stack: Any, geom: _Geometry) -> None:
    """Refuses a level that may move into the box but that placing the device would drop."""
    (_, _), (_, _), (z0, z1) = geom.box
    margin = float(stack.level_margin)
    for lv in geom.levels:
        lo, hi = lv.span
        plo, phi = geom.padded[lv.name]
        if phi > z0 and plo < z1 and not (hi > z0 - margin and lo < z1 + margin):
            need = max(z0 - hi, lo - z1)
            raise ValueError(
                f"mesh: level {lv.name!r} may move into the box (z padding) but its nominal span is {need:.4g} um "
                f"outside it, beyond level_margin={margin:g}: placing the device drops it. Build the device with "
                f"level_margin >= {need:.4g}"
            )


# ---------------------------------------------------------------- the coarse-vertex report

# a cell counts as coarse when wider than its target by more than this (relative): float32 edges
_COARSE_RTOL = 1e-4


@jax.tree_util.register_pytree_node_class
class CoarseVertices:
    """Vertices in cells wider than their target, per source layer (see :meth:`StackDevice.coarse_vertices`).

    A pytree (the layer names are static), so it can leave a jitted function as auxiliary output; ``str()``
    of concrete values prints a table.

    Attributes:
        layers: the source layers' parameter keys.
        total: (layers,) vertices per layer.
        x: (layers,) vertices whose x cell is wider than their target.
        y: (layers,) the same along y.
        z: (layers,) vertices of a layer that a level reading it puts in a z cell wider than that level's
            target (all its vertices in the box, if any).
        coarse: (layers,) vertices coarse along x, y or z.
        outside: (layers,) vertices outside the box (not counted above).
        worst: () the largest ratio of a cell to its target, over every vertex in the box and every level's z
            cells.
    """

    def __init__(self, layers: tuple[str, ...], total: Any, x: Any, y: Any, z: Any, coarse: Any, outside: Any,
                 worst: Any) -> None:
        self.layers, self.total, self.x, self.y, self.z = layers, total, x, y, z
        self.coarse, self.outside, self.worst = coarse, outside, worst

    def tree_flatten(self) -> tuple[tuple, tuple[str, ...]]:
        return (self.total, self.x, self.y, self.z, self.coarse, self.outside, self.worst), self.layers

    @classmethod
    def tree_unflatten(cls, layers: tuple[str, ...], leaves: tuple) -> CoarseVertices:
        return cls(layers, *leaves)

    def __str__(self) -> str:
        rows = [f"{'layer':<20} {'vertices':>8} {'x':>6} {'y':>6} {'z':>6} {'coarse':>7} {'outside':>8}"]
        for i, name in enumerate(self.layers):
            cells = (int(np.asarray(f)[i]) for f in (self.total, self.x, self.y, self.z, self.coarse, self.outside))
            rows.append("{:<20} {:>8} {:>6} {:>6} {:>6} {:>7} {:>8}".format(name, *cells))
        rows.append(f"largest cell / target: {float(np.asarray(self.worst)):.3f}")
        return "\n".join(rows)


def _report_targets(stack: Any, geom: _Geometry) -> tuple[dict[int, float], dict[str, float]]:
    """Per source layer, the lateral target of its vertices; per level, its z target: the smallest target of the
    materials that meet there (the level's own and its neighbours')."""
    by_level = {lv.name: lv for lv in geom.levels}
    level_z = {}
    for name, lv in by_level.items():
        _, below, above = geom.neighbours[name]
        level_z[name] = min([geom.cell[lv.material], *(geom.cell[m] for m in below | above)])
    layer_xy = {}
    for _, index in stack.source_layers:
        sizes = [
            min([geom.cell[lv.material], *(geom.cell[m] for m in geom.neighbours[lv.name][0])])
            for lv in geom.levels if index in lv.reads
        ]
        layer_xy[index] = min(sizes) if sizes else math.inf
    return layer_xy, level_z


def _bound_tol(edges: np.ndarray, dtype: Any) -> float:
    """Float precision margin for box boundary comparisons against ``edges``."""
    dt = np.dtype(dtype)
    scale = max(1.0, abs(float(edges[0])), abs(float(edges[-1])))
    if np.issubdtype(dt, np.floating) and dt.itemsize < 8:
        return 2.0 * float(np.finfo(dt).eps) * scale
    return 1e-12 * scale


def _cell_width(edges: np.ndarray, widths: np.ndarray, x: Any) -> Any:
    """The width of the cell holding each x (the wider of the two on an edge)."""
    import jax.numpy as jnp

    n = len(widths)
    mid = 0.5 * (float(edges[0]) + float(edges[-1]))
    e = jnp.asarray(edges - mid, dtype=jnp.float64)
    x_rel = jnp.asarray(x, dtype=jnp.float64) - mid
    left = jnp.clip(jnp.searchsorted(e, x_rel, side="left") - 1, 0, n - 1)
    right = jnp.clip(jnp.searchsorted(e, x_rel, side="right") - 1, 0, n - 1)
    w = jnp.asarray(widths, dtype=jnp.float64)
    return jnp.maximum(w[left], w[right])


def coarse_vertices(stack: Any, params: Mapping[str, Any], mesh: Mesh) -> CoarseVertices:
    """Per source layer, how many vertices sit in cells wider than their target (see
    :meth:`bifrost.StackDevice.coarse_vertices`)."""
    import jax.numpy as jnp

    geom = mesh.geometry
    if stack.grid_frame is not None:  # a placed device: on the mesh's grid (including box origin)
        frame = stack.grid_frame
        mids = (float(stack.center[0]), float(stack.center[1]), float(stack.z_center))
        for name, mid, e, fe in zip("xyz", mids, mesh.edges, frame.edges, strict=True):
            fe_um = np.asarray(fe, dtype=np.float64) * frame.h_um
            placed = mid + (fe_um - 0.5 * (fe_um[0] + fe_um[-1]))
            if placed.shape != e.shape or not np.allclose(placed, e, atol=1e-4):
                raise ValueError(f"coarse_vertices: the device is placed on another grid than the mesh (along {name})")
    layer_xy, level_z = _report_targets(stack, geom)
    (ex, ey, ez) = (np.asarray(e, dtype=np.float64) for e in mesh.edges)
    wx, wy, wz = np.diff(ex), np.diff(ey), np.diff(ez)
    rows = []
    worst = jnp.zeros(())
    for key, index in stack.source_layers:
        v = jnp.asarray(params[key])
        target = layer_xy[index]
        tol_x = _bound_tol(ex, v.dtype)
        tol_y = _bound_tol(ey, v.dtype)
        inside = (
            (v[:, 0] >= ex[0] - tol_x)
            & (v[:, 0] <= ex[-1] + tol_x)
            & (v[:, 1] >= ey[0] - tol_y)
            & (v[:, 1] <= ey[-1] + tol_y)
        )
        cx, cy = _cell_width(ex, wx, v[:, 0]), _cell_width(ey, wy, v[:, 1])
        over_x = inside & (cx > target * (1 + _COARSE_RTOL))
        over_y = inside & (cy > target * (1 + _COARSE_RTOL))
        worst = jnp.maximum(worst, jnp.max(jnp.where(inside, jnp.maximum(cx, cy) / target, 0.0), initial=0.0))
        over_z = jnp.zeros((), dtype=bool)
        for lv in geom.levels:
            if index not in lv.reads:
                continue
            zb = jnp.asarray(params[f"zmin/{lv.name}"])
            top = zb + jnp.asarray(params[f"thickness/{lv.name}"])  # signed: a negative thickness grows down
            lo, hi = jnp.minimum(zb, top), jnp.maximum(zb, top)
            hit = (ez[:-1] < hi) & (ez[1:] > lo)
            over_z = over_z | jnp.any(hit & (wz > level_z[lv.name] * (1 + _COARSE_RTOL)))
            worst = jnp.maximum(worst, jnp.max(jnp.where(hit, wz / level_z[lv.name], 0.0), initial=0.0))
        z = inside & over_z
        rows.append((len(v), over_x.sum(), over_y.sum(), z.sum(), (over_x | over_y | z).sum(), (~inside).sum()))
    columns = [jnp.asarray([r[i] for r in rows], dtype=jnp.int32) for i in range(6)]
    return CoarseVertices(tuple(k for k, _ in stack.source_layers), *columns, worst)
