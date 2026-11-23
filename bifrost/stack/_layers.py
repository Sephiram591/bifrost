"""Shared constants, layer expressions, grid frame, port extensions, and derived shapes."""

import math
import warnings
import fdtdx
import jax
import jax.numpy as jnp
import numpy as np
from collections.abc import Callable, Container, Sequence
from typing import Any, Literal, NamedTuple
from fdtdx.config import SimulationConfig
from ..geometry import DerivedRecipe, derive_polygons, recipe_rings, run_recipe, to_numpy


Array = jax.Array


Averaging = Literal["inverse", "subpixel", "subpixel_full"]


PortDirection = Literal["in", "out"]  # into the component through the port, or out of it


_BIG = 1e6  # "infinitely" far, in cells


# um: StackDevice keeps the levels this close to the box (beyond
# level_margin), to cover fdtdx's rounding of the box to whole cells and the
# half cell sampled below it on grids up to ~130 nm; the levels that do not
# reach the placed box are dropped in place_on_grid
_SELECT_SLACK = 0.1


_ANISOTROPY_SENTINEL = "__diagonal_anisotropy__"


# offsets (cells) from a voxel center to fdtdx's Ex, Ey, Ez Yee positions:
# Ex at (i + 1/2, j, k), Ey at (i, j + 1/2, k), Ez at (i, j, k + 1/2)
_YEE_E_SHIFTS = ((0.0, -0.5, -0.5), (-0.5, 0.0, -0.5), (-0.5, -0.5, 0.0))


_OFFDIAG_PAIRS = ((0, 1), (0, 2), (1, 2))  # the xy, xz, yz entries of a symmetric 3 x 3 tensor


# corner model (see _corner_fill), lengths in sides of the lateral square:
_CORNER_TOL = 1e-3  # slack of the certification radii and of the corner columns


_PLANE_TOL = 1e-4  # half-planes closer than this are the same line


_CORNER_RAMP = 0.25  # width of the ramps that blend the corner model in


# width of the ramps of a sample's certificate for its voxel (several sub-samples per voxel): every
# literal constant or single over this much more than the reach; the corner columns reach this far
_VOXEL_RAMP = 0.05


_TIP_AREA = 1e-9  # round-join tips smaller than this (fraction of the square) are ignored


_MITRE_MIN = 0.05  # no tip correction where 1 + cos(angle between the edge normals) is below


# corner columns whose corner model is near the half-planes (both at the Gauss heights) take the seam
# between the half-planes at the mean offset (the ordinary columns along the edge) and at the Gauss
# heights (see _corner_fill), per level on the union of the levels painted up to it: it fades out
# smoothly as their gap reaches _STRAIGHT_WIDTH times the seam's own size, but at least _STRAIGHT_TOL
# (fractions)
_STRAIGHT_WIDTH = 0.5


_STRAIGHT_TOL = 1e-4


# cells: a level's part of a voxel layer shorter than this takes its sidewall at the part's middle
# (the exact mean over its pieces divides by the part's length)
_PART_TINY = 1e-6


# 3-point Gauss-Legendre (t, weight) on [0, 1]
_GL3 = ((0.5 - 0.5 * math.sqrt(0.6), 5 / 18), (0.5, 8 / 18), (0.5 + 0.5 * math.sqrt(0.6), 5 / 18))


# averaging="subpixel_full" (see _interface_composite and bifrost.subpixel.junction_weight): a
# voxel's interface direction tensor D sets how far its interfaces are from one plane, r = 1 - lambda_1 /
# tr D; up to _JUNCTION_LOW the voxel is planar (the laminate along its leading direction), from
# _JUNCTION_HIGH a junction (the closure), with a smooth step between
_JUNCTION_LOW = 0.002


_JUNCTION_HIGH = 0.02


# the junction closure's sub-cells: the gap between the two largest eigenvalues of a sub-cell's direction tensor
# (over its trace) below which its laminate blends away from the leading direction (fully below the first; see
# _junction_cells), and the trace below which it has no faces
_SUBCELL_GAP = (0.02, 0.1)


_NO_FACES = 1e-15


# sides of the square: a face next to a part thinner than this (the part's area over the face's length)
# counts less, t / (t + _FACE_THICKNESS), so the faces of a sliver fade in with it
_FACE_THICKNESS = 0.1


_ON_LINE = 1e-6  # sides of the square: points on a line are looked at this far to each side (coincident lines)


_PARALLEL = 1e-6  # |sin| of the angle below which two lines are parallel


_T_FAR = 8.0  # sides of the square: beyond every point of the squares a line is followed in


_JUNCTION_CHUNK = 256  # junction voxels whose cell problems are solved together (bounds the memory)


_CERTIFIED = 1e-6  # a corner column's voxel whose corner model's certificate is below 1 - this is uncertified


# normwise backward error of a cell problem's solve above which it failed (a Cholesky solve of a few hundred
# unknowns stays near 1e-6 in float32); NaN (a failed factorization) fails too
_CLOSURE_RESIDUAL = 1e-4


_HIGHEST = jax.lax.Precision.HIGHEST  # full float32 contractions (GPUs default to TF32)


# kernel flux per unit lateral eps variance below which the variance falls back to the FD normal
_KOTTKE_ETA = 0.25


# normalized eps variances (and squared jumps) below this are too small to divide by: they count as
# unexplained (and the vertical part goes along z) with no derivative of the ratios, which in float32
# would square the tiny denominator below the smallest normal number (an inf, then NaN gradients)
_KOTTKE_TINY = 1e-12


# um: a vertex this close to a port face is on it (default of _face_intervals;
# the device uses its topology tolerance, see _topology_tol)
_FACE_TOL = 1e-3


# polygon topology tolerance (see StackDevice.topology_tol): database units of
# the layout (snapping to its grid moves each vertex by up to half a unit per
# axis, so a vertex on a slanted edge can end up sqrt(2) units off it), plus
# float32 epsilons times the largest coordinate (the round-off of float32
# parameters moves a vertex off an edge by at most sqrt(2) of them)
_TOPOLOGY_DBU = 2.0


_TOPOLOGY_EPS = 2.0


# geometry diagnostics (see StackDevice.diagnose):
_MIN_FEATURE = 1.5  # cells (divided by xy_subsamples): narrower parts and gaps are reported


_CHECK_ANGLE = 90.0  # degrees: KLayout's ignore_angle; edges facing each other at a larger angle are never close


_NARROW_ANGLE = 45.0  # degrees: the two sides of one corner meeting at a larger angle are the corner, not a narrow part


_CHECK_DBU = 1e-4  # um: grid of the KLayout regions (coarser only where 32-bit coordinates need it)


_CHECK_HEIGHTS = 64  # most cross-sections checked per level


_ROUND_TOL = 0.01  # of the threshold: round joins are drawn to within it, and narrower places are round-off


class _Level(NamedTuple):
    """A LayerLevel compiled to static data (lengths in um, angles in degrees)."""

    name: str
    expr: tuple  # layer expression, see _compile_layer
    zmin: float
    thickness: float
    sidewall_angle: float
    width_to_z: float
    bias: float
    z_to_bias: tuple[tuple[float, ...], tuple[float, ...]] | None
    mesh_order: int
    material: str
    declared_layers: tuple[int, ...] = ()  # positive declared layers before inversion and empty-layer folding


class _Port(NamedTuple):
    """A component port as static data (um, degrees; orientation None if unset)."""

    name: str
    x: float
    y: float
    orientation: float | None
    width: float
    layer: int


def _topology_tol(tol: float | None, vertices: Any) -> float:
    """Distance (um) within which the polygon topology of a layer's vertices is found.

    Vertices this close to an edge lie on it (T-junctions), this close to each
    other coincide (seams), and this close to a port face are on it. The same
    rule serves the device (built from ``source_vertices``), the topology check
    and :meth:`StackDevice.retopologize`, so what the check keeps a rebuild keeps.

    Args:
        tol: the device's ``topology_tol`` (um), or None for ``_TOPOLOGY_DBU``
            database units of the layout.
        vertices: (E, 2) vertices (um) of the layer.

    Returns:
        ``tol`` plus the float32 round-off of the vertices: ``_TOPOLOGY_EPS``
        float32 epsilons times their largest finite coordinate.
    """
    if tol is None:
        from gdsfactory import kcl

        tol = _TOPOLOGY_DBU * float(kcl.dbu)
    v = np.abs(np.asarray(vertices, dtype=float))
    largest = float(v[np.isfinite(v)].max(initial=0.0))
    return float(tol) + _TOPOLOGY_EPS * float(np.finfo(np.float32).eps) * largest


def _read_port(port: Any) -> _Port:
    """A component's port as static data (gradients stopped)."""
    from gdsfactory.pdk import get_layer

    orientation = port.orientation
    return _Port(
        name=str(port.name),
        x=float(to_numpy(port.x)),
        y=float(to_numpy(port.y)),
        orientation=None if orientation is None else float(to_numpy(orientation)),
        width=0.0 if port.width is None else float(to_numpy(port.width)),
        layer=int(get_layer(port.layer)),
    )


# A corner of a port extension on the port face, in a layer's packed vertices:
# (i, i) is vertex i, (i, j) is where edge i -> j crosses the face, (-1, -1) is
# the port's own corner (center -+ width / 2 along the face)
_FaceEnd = tuple[int, int]


_PORT_CORNER: _FaceEnd = (-1, -1)


def _face_frame(orientation: Any, xp: Any) -> tuple[Any, Any]:
    """Outward normal (along the port's orientation) and face tangent of a port."""
    angle = xp.deg2rad(orientation)
    return xp.stack([xp.cos(angle), xp.sin(angle)]), xp.stack([-xp.sin(angle), xp.cos(angle)])


def _face_point(end: _FaceEnd, side: float, v: Any, center: Any, width: Any, orientation: Any, xp: Any) -> Any:
    """A corner of an extension on the port face (see ``_FaceEnd``).

    Args:
        end: the corner.
        side: -1 for the first corner along the face tangent, +1 for the second
            (only used by the port's own corners).
        v: (E, 2) packed vertices of the layer.
        center: (2,) port center.
        width: port width.
        orientation: port orientation (degrees).
        xp: numpy or jax.numpy.

    Returns:
        The (2,) point.
    """
    n, t = _face_frame(orientation, xp)
    i, j = end
    if i < 0:
        return xp.asarray(center) + side * 0.5 * width * t
    if i == j:
        return v[i]
    a, b = v[i], v[j]
    return a + (xp.dot(n, xp.asarray(center) - a) / xp.dot(n, b - a)) * (b - a)


def _extension_rect(
    ends: tuple[_FaceEnd, _FaceEnd], v: Any, center: Any, width: Any, orientation: Any, length: float, xp: Any
) -> Any:
    """Corners (4, 2) of one extension, counter-clockwise.

    Corners 0 and 3 are on the port face (ordered along the face tangent), 1 and
    2 at the far end, ``length`` along the port's orientation.
    """
    n, _ = _face_frame(orientation, xp)
    a = _face_point(ends[0], -1.0, v, center, width, orientation, xp)
    b = _face_point(ends[1], 1.0, v, center, width, orientation, xp)
    return xp.stack([a, a + length * n, b + length * n, b])


def _face_intervals(
    port: _Port,
    vertices: np.ndarray,
    sizes: Sequence[int],
    tol: float = _FACE_TOL,
    *,
    inverted: bool = False,
) -> list[tuple[_FaceEnd, _FaceEnd]]:
    """Where a layer meets a port's face: the cross-section the port extension continues.

    The union of the layer's polygons (nonzero winding per polygon, so abutting
    and overlapping polygons merge) is sliced along a line parallel to the port
    face, just inside the component. Each covered interval that overlaps the port
    (its width along the face; for an ``inverted`` layer, also intervals whose
    edge bounds the port's core at ``+-width / 2``) is returned by its two ends
    on the face itself: a polygon vertex there, else the crossing of the polygon
    edge with the face, so the extension follows the vertices when they move.

    Args:
        port: the port (with an orientation).
        vertices: (E, 2) packed vertices (um) of the layer.
        sizes: vertices per polygon.
        tol: distance (um) within which a vertex is on the face and two crossings
            coincide.
        inverted: whether the layer is inverted (keep boundary-touching intervals).

    Returns:
        ``(first end, second end)`` per interval, ordered along the face tangent.
    """
    if not len(vertices):
        return []
    n, t = _face_frame(port.orientation, np)
    rel = np.asarray(vertices, dtype=float) - np.array([port.x, port.y])
    dn, dt = rel @ n, rel @ t  # dn < 0: inside the component
    starts = np.cumsum([0, *sizes[:-1]])
    nxt = np.arange(1, len(vertices) + 1)
    ids = np.repeat(np.arange(len(sizes)), sizes)
    for start, size in zip(starts, sizes, strict=True):
        nxt[start + size - 1] = start

    # crossings of the slice line dn = -2 tol, with their winding direction
    a = dn + 2 * tol
    b = a[nxt]
    edges = np.flatnonzero((a < 0) != (b < 0))
    if not edges.size:
        return []
    s = dt[edges] + (dt[nxt[edges]] - dt[edges]) * a[edges] / (a[edges] - b[edges])
    sign = np.where(b[edges] >= 0, 1, -1)

    def end(e: int) -> _FaceEnd:
        i, j = int(e), int(nxt[e])
        if abs(dn[i]) <= tol:
            return (i, i)
        if abs(dn[j]) <= tol:
            return (j, j)
        return (i, j)

    # sweep along the face; crossings at one position are applied together, so
    # seams between abutting polygons do not end an interval
    order = np.argsort(s, kind="stable")
    winding = np.zeros(len(sizes), dtype=int)
    intervals, opened = [], None
    k = 0
    while k < len(order):
        group = [order[k]]
        k += 1
        while k < len(order) and s[order[k]] - s[group[0]] <= tol:
            group.append(order[k])
            k += 1
        before = winding.copy()
        for g in group:
            winding[ids[edges[g]]] += sign[g]
        was, now = bool(before.any()), bool(winding.any())
        if was == now:
            continue
        # an edge of a polygon that entered (or left) the union here
        g = next(g for g in group if bool(before[ids[edges[g]]]) == was and bool(winding[ids[edges[g]]]) == now)
        if now:
            opened = (s[g], end(edges[g]))
        else:
            intervals.append((opened[0], s[g], opened[1], end(edges[g])))
    half = 0.5 * port.width
    if inverted:
        return [(ea, eb) for s0, s1, ea, eb in intervals if s1 >= -half - tol and s0 <= half + tol]
    return [(ea, eb) for s0, s1, ea, eb in intervals if s1 > -half + tol and s0 < half - tol]


def _port_extensions(
    polygons: dict[int, list[np.ndarray]],
    ports: dict[str, _Port],
    extended: Sequence[tuple[str, float, tuple[int, ...]]],
    tol: float | None = None,
    inverted: Container[int] = (),
) -> tuple[dict[int, list[tuple[str, float, _FaceEnd, _FaceEnd]]], dict[int, list[np.ndarray]]]:
    """The port extensions of every layer (see ``extend_ports`` of :class:`StackDevice`).

    Args:
        polygons: static polygons (um) per layer index.
        ports: the component's ports by name.
        extended: (port, length, layers it may continue) per extended port.
        tol: the device's ``topology_tol``: where a layer meets a port face is
            found within the layer's topology tolerance (:func:`_topology_tol`).
        inverted: layer indices whose material is applied to the inverse of
            their geometry (never synthesized across an empty port face).

    Returns:
        extensions: per layer, (port, length, face end a, face end b) of each
            extension, in the order of ``extended``, then along the face.
        rectangles: per layer, the (4, 2) corners (um) of each extension.
    """
    extensions: dict[int, list[tuple[str, float, _FaceEnd, _FaceEnd]]] = {}
    rectangles: dict[int, list[np.ndarray]] = {}
    for name, length, layers in extended:
        p = ports[name]
        for index in layers:
            polys = polygons.get(index, [])
            own = np.concatenate(polys) if polys else np.zeros((0, 2))
            inv = index in inverted
            ends = _face_intervals(p, own, [len(q) for q in polys], _topology_tol(tol, own), inverted=inv)
            if index == p.layer and not inv and not ends:
                ends = [(_PORT_CORNER, _PORT_CORNER)]  # nothing drawn at the face: the port's width
            for pair in ends:
                extensions.setdefault(index, []).append((name, float(length), *pair))
                rectangles.setdefault(index, []).append(
                    _extension_rect(pair, own, (p.x, p.y), p.width, p.orientation, float(length), np)
                )
    return extensions, rectangles


def _face_spans(ends: Sequence[tuple[_FaceEnd, _FaceEnd]], v: np.ndarray, port: _Port) -> np.ndarray:
    """The (m, 2, 2) corners on the port face of extensions ``ends`` of the layer with own vertices ``v``.

    Non-finite where a face end is an edge that no longer crosses the face.
    """
    if not ends:
        return np.zeros((0, 2, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray([
            [_face_point(end, side, v, (port.x, port.y), port.width, port.orientation, np)
             for end, side in ((a, -1.0), (b, 1.0))]
            for a, b in ends
        ], dtype=float)


def _span(level: Any) -> tuple[float, float]:
    """Nominal (bottom, top) of a level in um.

    Args:
        level: anything with ``zmin`` and ``thickness`` (negative thickness grows down).

    Returns:
        (bottom, top) stack heights.
    """
    z0, z1 = float(level.zmin), float(level.zmin) + float(level.thickness)
    return min(z0, z1), max(z0, z1)


# relative cell-width variation up to which an axis of a non-uniform grid counts as uniform (fdtdx's
# own test, fdtdx.core.grid.RectilinearGrid), plus a round-off floor that grows with the coordinates
_FRAME_RTOL = 1e-4


# lateral cells longer than this in one direction than in the other get a placement warning
_MAX_ASPECT = 16.0


class _SampleAxis(NamedTuple):
    """One axis of a sample grid: the voxel centers', or one E component's Yee positions.

    Sample voxel k spans ``[lo[k], hi[k]]`` in the frame (:class:`_Frame`) moved by ``shift``.
    On a frame-uniform axis that is the lattice the device always used: ``shift`` is the Yee
    shift (0 or -0.5 cells), ``lo = 0, 1, ..., n - 1`` and ``hi = lo + 1``. On a graded axis
    ``shift`` is 0 and the sample voxels are the cells (unshifted) or the dual cells between
    neighbouring cell centers (shifted by half a cell): a Yee point lies in its dual cell, but
    not at its center where the two cells differ.
    """

    shift: float
    lo: np.ndarray  # (n,) frame units
    hi: np.ndarray  # (n,) frame units
    uniform: bool


class _Frame(NamedTuple):
    """The static grid frame of a placed device, from the resolved simulation grid.

    Frame coordinates are lengths over ``h_um`` from the lower corner of the device's box. On a
    uniform grid ``h_um`` is the grid spacing and every axis has the edges 0, 1, ..., n: the cell
    frame of a uniform grid. On a non-uniform grid ``h_um`` is the lateral spacing when x and y
    are uniform with one spacing, else the smallest lateral cell, and an axis whose cells all
    have that width keeps the integer edges.
    """

    h_um: float
    #: per axis, the (n + 1,) cell edges in frame units, from 0
    edges: tuple[np.ndarray, np.ndarray, np.ndarray]
    #: per axis, the width (frame units) of the grid cell just below the box: the box's first
    #: cell where the box starts at the domain's lower face (mirrored, as fdtdx's dual widths)
    below: tuple[float, float, float]
    #: per axis, whether the edges are exactly 0, 1, ..., n
    uniform: tuple[bool, bool, bool]
    #: per axis, the width (frame units) of its cells where they are all one size (also the cell below the
    #: box), the edges then exactly 0, w, 2 w, ...; None on a graded axis
    widths: tuple[float | None, float | None, float | None] = (1.0, 1.0, 1.0)

    def sample_axis(self, axis: int, shift: float) -> _SampleAxis:
        """The sample voxels along ``axis`` of the grid whose points are moved by ``shift`` cells.

        Args:
            axis: 0, 1 or 2.
            shift: 0 (cell centers) or -0.5 (Yee positions on the lower cell faces).

        Returns:
            The sample axis.

        Raises:
            ValueError: if a graded axis gets a shift other than 0 or -0.5.
        """
        edges = self.edges[axis]
        n = len(edges) - 1
        if self.uniform[axis]:
            lo = np.arange(n, dtype=float)
            return _SampleAxis(float(shift), lo, lo + 1.0, True)
        if shift == 0:
            return _SampleAxis(0.0, edges[:-1], edges[1:], False)
        if shift != -0.5:
            raise ValueError(f"a graded axis samples at cell centers (0) or Yee positions (-0.5), not {shift}")
        centers = 0.5 * (edges[:-1] + edges[1:])
        lo = np.concatenate([[-0.5 * self.below[axis]], centers[:-1]])
        return _SampleAxis(0.0, lo, centers, False)


def _grid_frame(config: SimulationConfig, slices: Any) -> _Frame:
    """The frame of a device placed on ``slices`` (its grid slice tuple) of the config's grid.

    Args:
        config: the simulation config of the placement (its grid resolved, as fdtdx's
            ``place_objects`` makes it).
        slices: ((x0, x1), (y0, y1), (z0, z1)) cell ranges of the device.

    Returns:
        The frame; on a uniform grid ``h_um`` is ``config.uniform_spacing() * 1e6``.
    """
    counts = tuple(int(b) - int(a) for a, b in slices)
    grid = config.resolved_grid
    if grid is None or grid.is_uniform:
        if grid is not None and np.asarray(grid.edges(0)).dtype != np.float64:
            a0, b0 = int(slices[0][0]), int(slices[0][1])
            e0 = np.asarray(grid.edges(0), dtype=np.float64)[a0 : b0 + 1] * 1e6
            h_um = round(float(e0[-1] - e0[0]) / (b0 - a0), 8)
        else:
            h_um = config.uniform_spacing() * 1e6
        return _Frame(
            h_um,
            tuple(np.arange(n + 1, dtype=float) for n in counts),
            (1.0, 1.0, 1.0),
            (True, True, True),
        )
    edges, below, widths, uniform, floors = [], [], [], [], []
    for axis, (a, b) in enumerate(slices):
        raw = np.asarray(grid.edges(axis))
        full = raw.astype(np.float64) * 1e6
        e = full[int(a) : int(b) + 1]
        w = np.diff(e)
        eps = float(np.finfo(raw.dtype).eps) if np.issubdtype(raw.dtype, np.floating) else 0.0
        floor = 8.0 * eps * float(np.max(np.abs(full)))
        edges.append(e)
        widths.append(w)
        floors.append(floor)
        below.append(float(full[a] - full[a - 1]) if a > 0 else float(w[0]))
        uniform.append(bool(np.max(np.abs(w - w[0])) <= _FRAME_RTOL * w[0] + floor))
    # an evenly spaced axis' spacing over its whole extent, rounded like fdtdx rounds a uniform grid's (1e-14 m):
    # one cell's width carries the float32 round-off of the edges (~1e-6 of it), which would move the frame
    spacing = [
        round(float(e[-1] - e[0]) / len(w), 8) if u else float(e[-1] - e[0]) / len(w)
        for e, w, u in zip(edges, widths, uniform, strict=True)
    ]
    h = min(spacing[i] if uniform[i] else float(np.min(widths[i])) for i in (0, 1))
    out_edges, out_uniform, out_widths = [], [], []
    for e, w, u, s, b, fl in zip(edges, widths, uniform, spacing, below, floors, strict=True):
        # the cell below the box too: the Yee-shifted samples of the first layer reach into it
        even = u and abs(b - s) <= _FRAME_RTOL * s + fl
        same = even and abs(s - h) <= _FRAME_RTOL * h + fl
        if same:
            out_edges.append(np.arange(len(w) + 1, dtype=float))
        elif even:  # one other size: exact multiples of it
            out_edges.append(np.arange(len(w) + 1, dtype=float) * (s / h))
        else:
            out_edges.append((e - e[0]) / h)
        out_uniform.append(same)
        out_widths.append(1.0 if same else (float(s / h) if even else None))
    return _Frame(h, tuple(out_edges), tuple(b / h for b in below), tuple(out_uniform), tuple(out_widths))


def _conductivity_scale(config: SimulationConfig) -> float:
    """The factor fdtdx stores conductivities with (its update's ``c * dt / courant_number``).

    On a uniform grid this is the grid spacing, as bifrost always wrote it; on a non-uniform one
    it is fdtdx's own expression (``fdtdx.fdtd.initialization``), the same for every cell.
    """
    if not config.has_nonuniform_grid:
        return config.uniform_spacing()
    return fdtdx.constants.c * config.time_step_duration / config.courant_number


def _sizing(layer: Any) -> float:
    """Uniform grow of a Logical/DerivedLayer from its sizings.

    gdsfactory applies the sizings one after another (``region.sized(x, y,
    mode)``, zipped like gdsfactory zips them). A signed distance field can only
    shift by a constant, so the steps are summed. That equals the sequence only
    when every step grows or every step shrinks with one mode: a shrink then a
    grow (opening) removes features narrower than twice the shrink, which a sum
    cannot, so such sequences are rejected rather than drawn differently.

    Args:
        layer: the layer.

    Returns:
        The grow in um.

    Raises:
        NotImplementedError: for different x and y sizings, steps that both grow
            and shrink, or steps with different modes.
    """
    from gdsfactory import kcl

    steps = [
        (int(x), int(y), int(m))
        for x, y, m in zip(layer.sizings_xoffsets, layer.sizings_yoffsets, layer.sizings_modes, strict=False)
        if x or y
    ]
    if any(x != y for x, y, _ in steps):
        raise NotImplementedError(f"Anisotropic layer sizing {steps} (x != y) is not supported in {layer}")
    if any(x > 0 for x, *_ in steps) and any(x < 0 for x, *_ in steps):
        raise NotImplementedError(
            f"Layer sizing {[x for x, *_ in steps]} dbu of {layer} both grows and shrinks (an opening "
            "or closing), which is not a single offset; only all-grow or all-shrink sequences are supported"
        )
    if len({m for *_, m in steps}) > 1:
        raise NotImplementedError(f"Layer sizing of {layer} mixes modes {[m for *_, m in steps]}")
    return float(sum(x for x, *_ in steps)) * float(kcl.dbu)


def _compile_layer(layer: Any, inverted_layers: set[int]) -> tuple:
    """Turns a LogicalLayer / DerivedLayer into a static expression tree.

    Nodes are ``("fill", grow)``, ``("layer", index, grow)`` and
    ``(op, left, right, grow)`` with op in and / or / xor / not.

    Args:
        layer: LogicalLayer, DerivedLayer or anything ``get_layer`` accepts.
        inverted_layers: layer indices whose material is applied to the inverse
            of their geometry.

    Returns:
        The expression tree.
    """
    from gdsfactory.pdk import get_layer
    from gdsfactory.technology.layer_stack import DerivedLayer, LogicalLayer

    if isinstance(layer, LogicalLayer):
        index = int(get_layer(layer.layer))
        if index in inverted_layers:
            return ("not", ("fill", 0.0), ("layer", index, 0.0), _sizing(layer))
        return ("layer", index, _sizing(layer))
    if isinstance(layer, DerivedLayer):
        op = {"&": "and", "|": "or", "^": "xor", "-": "not"}.get(layer.operation, layer.operation)
        return (
            op,
            _compile_layer(layer.layer1, inverted_layers),
            _compile_layer(layer.layer2, inverted_layers),
            _sizing(layer),
        )
    return _compile_layer(LogicalLayer(layer=layer), inverted_layers)


def _compile_level(level: Any, inverted_layers: set[int]) -> tuple:
    """Expression tree of a LayerLevel, including background levels.

    Args:
        level: the LayerLevel.
        inverted_layers: layer indices whose material is applied to the inverse
            of their geometry.

    Returns:
        The expression tree (see :func:`_compile_layer`).
    """
    if not level.background:
        return _compile_layer(level.layer, inverted_layers)
    expr: tuple = ("fill", 0.0)
    for lay in level.background_exclude_layers:
        expr = ("not", expr, _compile_layer(lay, inverted_layers), 0.0)
    return expr


def _add_grow(expr: tuple, grow: float) -> tuple:
    """Fold an additional scalar grow ``grow`` (um) into a boolean geometry expression."""
    if not grow:
        return expr
    kind = expr[0]
    if kind in ("fill", "empty"):
        return expr
    if kind == "layer":
        return ("layer", int(expr[1]), float(expr[2]) + grow)
    return (kind, expr[1], expr[2], float(expr[3]) + grow)


def _simplify_expr(expr: tuple, valid_sources: Container[int]) -> tuple:
    """Propagate ``empty`` and ``fill`` identities through a LayerStack expression tree."""
    kind = expr[0]
    if kind == "fill":
        return ("fill", 0.0)
    if kind == "empty":
        return ("empty", 0.0)
    if kind == "layer":
        idx = int(expr[1])
        return expr if idx in valid_sources else ("empty", 0.0)
    left = _simplify_expr(expr[1], valid_sources)
    right = _simplify_expr(expr[2], valid_sources)
    grow = float(expr[3])
    lk, rk = left[0], right[0]
    if kind == "or":
        if lk == "fill" or rk == "fill":
            return ("fill", 0.0)
        if lk == "empty":
            return _add_grow(right, grow)
        if rk == "empty":
            return _add_grow(left, grow)
    elif kind == "and":
        if lk == "empty" or rk == "empty":
            return ("empty", 0.0)
        if lk == "fill":
            return _add_grow(right, grow)
        if rk == "fill":
            return _add_grow(left, grow)
    elif kind == "not":
        if lk == "empty" or rk == "fill":
            return ("empty", 0.0)
        if rk == "empty":
            return _add_grow(left, grow)
        if lk == "fill" and rk == "not" and right[1][0] == "fill":
            return _add_grow(right[2], grow - float(right[3]))
    elif kind == "xor":
        if lk == "empty":
            return _add_grow(right, grow)
        if rk == "empty":
            return _add_grow(left, grow)
        if lk == "fill" and rk == "fill":
            return ("empty", 0.0)
        if lk == "fill":
            return ("not", ("fill", 0.0), right, grow)
        if rk == "fill":
            return ("not", ("fill", 0.0), left, grow)
    return (kind, left, right, grow)


def _expr_layers(expr: tuple) -> set[int]:
    """Indices of all polygon layers an expression reads."""
    if expr[0] in ("fill", "empty"):
        return set()
    if expr[0] == "layer":
        return {expr[1]}
    return _expr_layers(expr[1]) | _expr_layers(expr[2])


def _expr_can_draw(expr: tuple, nonempty: set[int]) -> bool:
    """Whether an expression can be nonempty, given the layers that have polygons."""
    return _simplify_expr(expr, nonempty)[0] != "empty"


def _expr_positive_layers(expr: tuple) -> set[int]:
    """Indices of the polygon layers that add to an expression (not the subtracted ones)."""
    if expr[0] in ("fill", "empty"):
        return set()
    if expr[0] == "layer":
        return {expr[1]}
    if expr[0] == "not":
        return _expr_positive_layers(expr[1])
    return _expr_positive_layers(expr[1]) | _expr_positive_layers(expr[2])


Field = tuple[Array, Array, Array]  # signed distance (cells) and its in-plane gradient


def _eval_expr(expr: tuple, sdf: dict[int, Field], shape: tuple[int, int], res_um: float) -> Field:
    """Signed distance of a layer expression and its in-plane gradient.

    Args:
        expr: expression tree (see :func:`_compile_layer`).
        sdf: (distance, gx, gy) per source layer index, (nx, ny) each, in cells.
        shape: (nx, ny).
        res_um: grid resolution in um (converts grows to cells).

    Returns:
        (distance, gx, gy), negative inside; layers without polygons are far outside.
    """

    def where(cond: Array, u: Field, v: Field) -> Field:
        return tuple(jnp.where(cond, ui, vi) for ui, vi in zip(u, v, strict=True))  # type: ignore[return-value]

    def neg(u: Field) -> Field:
        return (-u[0], -u[1], -u[2])

    def lo(u: Field, v: Field) -> Field:  # union
        return where(u[0] <= v[0], u, v)

    def hi(u: Field, v: Field) -> Field:  # intersection
        return where(u[0] >= v[0], u, v)

    zero = jnp.zeros(shape)
    kind = expr[0]
    if kind == "fill":
        out, grow = (jnp.full(shape, -_BIG), zero, zero), expr[1]
    elif kind == "layer":
        out, grow = sdf.get(expr[1], (jnp.full(shape, _BIG), zero, zero)), expr[2]
    else:
        u = _eval_expr(expr[1], sdf, shape, res_um)
        v = _eval_expr(expr[2], sdf, shape, res_um)
        if kind == "or":
            out = lo(u, v)
        elif kind == "and":
            out = hi(u, v)
        elif kind == "not":
            out = hi(u, neg(v))
        elif kind == "xor":
            out = hi(lo(u, v), neg(hi(u, v)))
        else:
            raise ValueError(f"Unknown layer operation {kind!r}")
        grow = expr[3]
    return (out[0] - grow / res_um, out[1], out[2]) if grow else out


_TRUE: tuple = ("true",)


_FALSE: tuple = ("false",)


def _literals(
    expr: tuple, res_um: float, layers: Container[int]
) -> tuple[tuple, tuple[tuple[int, int, float], ...]]:
    """A level's region as a boolean tree of offset source-layer literals.

    The region of a level with lateral offset s (cells) is ``{F + s < 0}`` for
    the distance F of :func:`_eval_expr`. ``compile(node, a, b)`` is the region
    ``{F_node + a * s + b < 0}``: ``or`` / ``and`` are the union / intersection
    of their operands, the subtracted operands of ``not`` and ``xor`` are
    negated with the offset's sign flipped, and grows move b. A literal
    ``(i, a, b)`` is ``{sd_i + a * s + b < 0}`` for the distance sd_i of source
    layer i. Fill layers are true and layers without polygons false; both are
    folded away. Equal regions up to their boundaries.

    Args:
        expr: expression tree (see :func:`_compile_layer`).
        res_um: grid resolution in um (converts grows to cells).
        layers: indices of the source layers that have polygons.

    Returns:
        tree: ``("lit", k)``, ``("and", x, y)``, ``("or", x, y)``,
            ``("neg", x)``, ``("true",)`` or ``("false",)``; negations only
            wrap literals' subtrees, never a constant.
        literals: ``(layer index, a, b)`` per literal k, a = +-1, b in cells;
            equal literals share one k.
    """
    literals: list[tuple[int, int, float]] = []

    def lit(i: int, a: int, b: float) -> tuple:
        if (i, a, b) not in literals:
            literals.append((i, a, b))
        return ("lit", literals.index((i, a, b)))

    def both(x: tuple, y: tuple) -> tuple:
        if _FALSE in (x, y):
            return _FALSE
        return y if x == _TRUE else x if y == _TRUE else ("and", x, y)

    def either(x: tuple, y: tuple) -> tuple:
        if _TRUE in (x, y):
            return _TRUE
        return y if x == _FALSE else x if y == _FALSE else ("or", x, y)

    def neg(x: tuple) -> tuple:
        return _FALSE if x == _TRUE else _TRUE if x == _FALSE else ("neg", x)

    def compile_(node: tuple, a: int, b: float) -> tuple:
        kind = node[0]
        if kind == "fill":
            return _TRUE
        if kind == "layer":
            return lit(node[1], a, b - node[2] / res_um) if node[1] in layers else _FALSE
        u, v, grow = node[1], node[2], node[3]
        b = b - grow / res_um
        if kind == "or":
            return either(compile_(u, a, b), compile_(v, a, b))
        if kind == "and":
            return both(compile_(u, a, b), compile_(v, a, b))
        if kind == "not":
            return both(compile_(u, a, b), neg(compile_(v, -a, -b)))
        if kind == "xor":  # hi(lo(u, v), neg(hi(u, v)))
            inner = both(compile_(u, -a, -b), compile_(v, -a, -b))
            return both(either(compile_(u, a, b), compile_(v, a, b)), neg(inner))
        raise ValueError(f"Unknown layer operation {kind!r}")

    tree = compile_(expr, 1, 0.0)
    return tree, tuple(literals)


def _eval_bool(tree: tuple, values: Sequence[Any]) -> Any:
    """Evaluates a literal tree (see :func:`_literals`).

    Args:
        tree: the tree.
        values: boolean arrays, one per literal.

    Returns:
        The tree's value (a Python bool for a constant tree).
    """
    kind = tree[0]
    if kind == "lit":
        return values[tree[1]]
    if kind in ("true", "false"):
        return kind == "true"
    if kind == "neg":
        return ~_eval_bool(tree[1], values)
    x, y = _eval_bool(tree[1], values), _eval_bool(tree[2], values)
    return x & y if kind == "and" else x | y


def _unit_normal(gx: Array, gy: Array) -> Array:
    """Unit vectors (..., 2) along (gx, gy); (1, 0) where the gradient vanishes (no layer near)."""
    norm2 = gx**2 + gy**2
    ok = norm2 > 1e-12
    scale = jax.lax.rsqrt(jnp.where(ok, norm2, 1.0))
    return jnp.stack([jnp.where(ok, gx * scale, 1.0), jnp.where(ok, gy * scale, 0.0)], axis=-1)


def _layer_key(index: int) -> str:
    """Parameter name ``vertices/<layer name>`` (or the index if it has no name)."""
    from gdsfactory.pdk import get_layer_name

    try:
        return f"vertices/{get_layer_name(index)}"
    except Exception:
        return f"vertices/{index}"


class StackDiagnosticWarning(UserWarning):
    """A `StackDiagnostic` reported when a `StackDevice` is placed (see its `diagnostics`)."""


def _kdb_region(polygons: Sequence[np.ndarray], dbu: float, merged: bool = True) -> Any:
    """A KLayout region of polygons (um) on a grid of ``dbu`` um; ``merged=False`` keeps their overlaps."""
    import klayout.db as kdb

    region = kdb.Region()
    region.merged_semantics = merged
    for p in polygons:
        q = np.rint(np.asarray(p, dtype=float) / dbu).astype(np.int64)
        region.insert(kdb.Polygon([kdb.Point(int(x), int(y)) for x, y in q]))
    return region


_NO_LAYER = -(2**30)  # a layer index no slot has: read as empty


class _DerivedShape(NamedTuple):
    """The exact shape of a level's layer expression (see :func:`derive_polygons`), drawn as one layer slot."""

    #: the expression without the root's sizing (levels with equal ones share the shape)
    expr: tuple
    #: names of the levels that draw it
    levels: tuple[str, ...]
    #: layer indices of the pool (their vertices concatenated, in this order)
    layers: tuple[int, ...]
    #: the recipes of its parts (rings that touch at a point are in different
    #: parts): vertices are copies of pool vertices and crossings of pool edges
    recipes: tuple[DerivedRecipe, ...]
    #: distance (um) within which the recipe's nodes were found
    tol: float


def _root_grow(expr: tuple) -> float:
    """The sizing of an expression's root node (um)."""
    return float(expr[1] if expr[0] == "fill" else expr[2] if expr[0] == "layer" else expr[3])


def _with_root_grow(expr: tuple, grow: float) -> tuple:
    """The expression with its root node's sizing replaced."""
    if expr[0] == "fill":
        return ("fill", grow)
    if expr[0] == "layer":
        return ("layer", expr[1], grow)
    return (expr[0], expr[1], expr[2], grow)


def _inner_grows(expr: tuple) -> bool:
    """Whether any node below the root is sized."""

    def any_grow(node: tuple) -> bool:
        if node[0] in ("fill", "layer"):
            return bool(_root_grow(node))
        return bool(node[3]) or any_grow(node[1]) or any_grow(node[2])

    return expr[0] not in ("fill", "layer") and (any_grow(expr[1]) or any_grow(expr[2]))


def _slot_expr(expr: tuple, slots: dict[int, int]) -> tuple:
    """The (unsized) expression over pool slots for :func:`derive_polygons`; missing layers are empty."""
    kind = expr[0]
    if kind == "fill":
        return ("fill",)
    if kind == "layer":
        return ("layer", slots[expr[1]]) if expr[1] in slots else ("not", ("fill",), ("fill",))
    return (kind, _slot_expr(expr[1], slots), _slot_expr(expr[2], slots))


def _shapely_polygons(v: np.ndarray, sizes: Sequence[int]) -> Any:
    """The union (shapely, nonzero rule) of a layer's polygons."""
    import shapely

    polys = [shapely.make_valid(shapely.Polygon(p)) for p in np.split(np.asarray(v, dtype=float), np.cumsum(sizes)[:-1])
             if len(p) >= 3]
    return shapely.union_all(polys) if polys else shapely.Polygon()


def _union_overlaps(v: np.ndarray, sizes: Sequence[int], tol: float) -> bool:
    """Whether a layer's polygons overlap with positive area (more than ``tol`` wide)."""
    import shapely

    polys = [shapely.make_valid(shapely.Polygon(p)) for p in np.split(np.asarray(v, dtype=float), np.cumsum(sizes)[:-1])
             if len(p) >= 3]
    if len(polys) < 2:
        return False
    tree = shapely.STRtree(polys)
    ii, jj = tree.query(polys, predicate="intersects")
    for i, j in zip(ii, jj, strict=True):
        if i < j and not shapely.intersection(polys[i], polys[j]).buffer(-tol).is_empty:
            return True
    return False


#: a derived shape may differ from the boolean by parts narrower than twice this many tolerances (one
#: bound for the build's validation and for check_topology, so a device that builds also passes its check)
_DERIVED_ERROR_WIDTH = 2.0


def _part_index(shape: int, part: int) -> int:
    """Layer slot index of a part of a derived shape (negative, unlike gds layer indices)."""
    return -1 - 1024 * shape - part


def _shape_region(recipes: Sequence[DerivedRecipe], pool: np.ndarray) -> Any:
    """The bounded region (shapely) a derived shape's parts draw from pool vertices (before ``negate``)."""
    import shapely

    parts = [_recipe_region(r, run_recipe(r.kind, r.index, r.positions, pool, np)) for r in recipes if len(r.kind)]
    return shapely.union_all(parts) if parts else shapely.Polygon()


def _recipe_region(recipe: DerivedRecipe, points: np.ndarray) -> Any:
    """The bounded region (shapely) a recipe's evaluated vertices draw (before ``negate``)."""
    import shapely

    signed = [
        (area, ring)
        for ring in recipe_rings(recipe, points)
        if len(ring) >= 3 and (area := float(_ring_area(ring))) != 0.0
    ]
    signed.sort(key=lambda item: abs(item[0]), reverse=True)
    region = shapely.Polygon()
    for area, ring in signed:
        poly = shapely.make_valid(shapely.Polygon(ring))
        region = region.union(poly) if area > 0.0 else region.difference(poly)
    return region


def _kdb_fold_rings(rings: Sequence[np.ndarray], dbu: float) -> Any:
    """Folds outer (+area) and hole (-area) rings in descending |area| order into a kdb.Region."""
    import klayout.db as kdb

    signed = [
        (area, q)
        for q in rings
        if len(q) >= 3 and (area := float(_ring_area(q))) != 0.0
    ]
    signed.sort(key=lambda item: abs(item[0]), reverse=True)
    region = kdb.Region()
    i = 0
    while i < len(signed):
        is_outer = signed[i][0] > 0.0
        j = i + 1
        while j < len(signed) and (signed[j][0] > 0.0) == is_outer:
            j += 1
        batch = _kdb_region([q for _, q in signed[i:j]], dbu)
        region = region + batch if is_outer else region - batch
        i = j
    return region


def _expr_region(expr: tuple, layers: dict[int, Any], universe: Any) -> Any:
    """The region (shapely) of an unsized expression on the layers' unions; missing layers are empty."""
    import shapely

    kind = expr[0]
    if kind == "fill":
        return universe
    if kind == "layer":
        return layers.get(expr[1], shapely.Polygon())
    x, y = _expr_region(expr[1], layers, universe), _expr_region(expr[2], layers, universe)
    return {"and": x.intersection, "or": x.union, "not": x.difference, "xor": x.symmetric_difference}[kind](y)


def _ring_area(ring: np.ndarray) -> float:
    """Signed area of a ring (positive counterclockwise)."""
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _derived_shapes(
    levels: Sequence[_Level],
    sources: Sequence[tuple[str, int]],
    vertices: Sequence[np.ndarray],
    sizes: Sequence[tuple[int, ...]],
    tol: Callable[[np.ndarray], float],
    name: str,
) -> tuple[_DerivedShape, ...]:
    """The exact shapes of the derived levels (see ``StackDevice.derived_shapes``).

    A level gets one when it is not a fill and its expression, with empty
    layers folded away, is more than one layer (or its complement), or one
    layer whose polygons overlap. Levels whose shape cannot be built (sizing
    below the expression's root, a corner without a formula, a recipe that
    does not reproduce the boolean) keep the boolean of offset layers, with a
    :class:`StackDiagnosticWarning`.

    Args:
        levels: the compiled levels.
        sources: (key, layer index) per source layer.
        vertices: their vertices (um).
        sizes: their vertices per polygon.
        tol: the topology tolerance (um) of a layer's vertices.
        name: the device's name (for warnings).

    Returns:
        The shapes; levels with equal unsized expressions share one.
    """
    import shapely

    by_index = {index: i for i, (_, index) in enumerate(sources)}
    nonempty = set(by_index)
    overlapping: dict[int, bool] = {}
    shapes: list[_DerivedShape] = []
    for lv in levels:
        if lv.expr[0] == "fill":
            continue
        key = _with_root_grow(lv.expr, 0.0)
        tree, literals = _literals(key, 1.0, nonempty)
        if tree in (_TRUE, _FALSE):
            continue
        single = tree[0] == "lit" or (tree[0] == "neg" and tree[1][0] == "lit")
        if single:
            index = literals[tree[1] if tree[0] == "lit" else tree[1][1]][0]  # not literals[0]: folded subtrees leave literals
            if index not in overlapping:
                i = by_index[index]
                overlapping[index] = _union_overlaps(vertices[i], sizes[i], tol(vertices[i]))
            if not overlapping[index]:
                continue  # one layer: its own distance is exact
        done = [k for k, s in enumerate(shapes) if s.expr == key]
        if done:
            shapes[done[0]] = shapes[done[0]]._replace(levels=(*shapes[done[0]].levels, lv.name))
            continue
        layers = tuple(sorted(_expr_layers(key) & nonempty))
        reason = ""
        if _inner_grows(lv.expr):
            reason = "it sizes a layer or an operand below the expression's root"
        else:
            slots = {index: s for s, index in enumerate(layers)}
            pool = [(vertices[by_index[index]], sizes[by_index[index]]) for index in layers]
            eps = max(tol(v) for v, _ in pool)
            try:
                recipes = derive_polygons(pool, _slot_expr(key, slots), eps)
            except ValueError as error:
                reason = str(error)
            else:
                # the recipe must reproduce the boolean (up to slivers of the tolerance)
                regions = {index: _shapely_polygons(*p) for index, p in zip(layers, pool, strict=True)}
                points = np.concatenate([v for v, _ in pool])
                lo, hi = points.min(axis=0) - 1.0, points.max(axis=0) + 1.0
                universe = shapely.box(*lo, *hi)
                drawn = _shape_region(recipes, points)
                if recipes[0].negate:
                    drawn = universe.difference(drawn)
                error = drawn.symmetric_difference(_expr_region(key, regions, universe))
                if not error.buffer(-_DERIVED_ERROR_WIDTH * eps).is_empty:
                    reason = f"its polygons differ from the boolean by {error.area:.3g} um^2"
        if reason:
            warnings.warn(
                f"StackDevice {name!r}: level {lv.name!r} is drawn as the boolean of offset layers ({reason}); "
                "it may draw slivers where edges of its layers coincide",
                StackDiagnosticWarning,
                stacklevel=3,
            )
            continue
        shapes.append(_DerivedShape(expr=key, levels=(lv.name,), layers=layers, recipes=recipes, tol=eps))
    return tuple(shapes)


def _primal(x: Any) -> Any:
    """The concrete value under eager-autodiff tracers (x itself otherwise)."""
    while isinstance(x, jax.core.Tracer) and hasattr(x, "primal"):
        x = x.primal
    return x
