"""Cell edges for non-uniform rectilinear grids.

`axis_edges` places the cell edges of one axis in continuous coordinates: every cell and dual cell
within its target, neighbouring cells within `growth`, uniform cells under each PML, and planar
interfaces in the exact middle of their cells. `graded_edges` places smoothly graded edges with
given coordinates on cell edges, and `lattice_edges` places edges on an explicit discrete lattice of
whole quanta. `bifrost.StackDevice.rectilinear_grid` turns such edges into the fdtdx grid of a
device's box.

The smallest cell sets the time step of the whole simulation, so refining saves cells and memory, not
time steps.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

__all__ = ["AxisEdges", "axis_edges", "graded_edges", "lattice_edges"]

# coordinates closer than this share (of the local cell size) are one edge
_MERGE = 0.5


def graded_edges(
    lo: float,
    hi: float,
    spacing: float,
    fine: Sequence[tuple[float, float, float]] = (),
    growth: float = 1.25,
    snap: Sequence[float] = (),
    uniform_ends: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Cell edges from ``lo`` to ``hi``: at most ``spacing`` wide, finer where asked, smoothly graded.

    The cell size at x is ``min(spacing, d_i + (growth - 1) dist(x, [a_i, b_i]))`` over the ``fine``
    regions ``(a_i, b_i, d_i)``: ``d_i`` inside a region, growing by about ``growth`` per cell away
    from it. Between consecutive ``snap`` coordinates (and the ends) the edges spread that size
    evenly, so every snap coordinate is an edge. A snap coordinate closer than half the local cell
    size to another one (or to an end) is dropped, with a warning, rather than making a sliver of a
    cell, which would shorten the time step of the whole simulation.

    Args:
        lo: first edge.
        hi: last edge.
        spacing: largest cell.
        fine: (a, b, d) regions [a, b] with cells of at most d.
        growth: largest ratio of neighbouring cells away from a fine region (> 1).
        snap: coordinates that become edges (e.g. layer interfaces).
        uniform_ends: the first and last this many cells are exactly ``spacing`` (e.g. under a PML,
            whose thickness fdtdx counts in cells); fine regions and snap coordinates must lie between
            them.

    Returns:
        (n + 1,) strictly increasing edges, the first ``lo`` and the last ``hi``.

    Raises:
        ValueError: on an empty range, a nonpositive spacing or fine size, ``growth <= 1``, or a fine
            region or snap coordinate outside the range between the uniform ends.
    """
    lo, hi, spacing, growth = float(lo), float(hi), float(spacing), float(growth)
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        raise ValueError(f"graded_edges needs finite lo < hi, got {lo} and {hi}")
    if not (math.isfinite(spacing) and spacing > 0):
        raise ValueError(f"spacing must be finite and positive, got {spacing}")
    if not (math.isfinite(growth) and growth > 1):
        raise ValueError(f"growth must be finite and > 1, got {growth}")
    regions: list[tuple[float, float, float]] = []
    for region in fine:
        if len(region) != 3:
            raise ValueError(f"a fine region is (a, b, d), got {tuple(region)}")
        a, b, d = (float(x) for x in region)
        if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(d) and b > a and d > 0):
            raise ValueError(f"a fine region is (a, b, d) with finite a < b and d > 0, got {(a, b, d)}")
        regions.append((a, b, d))
    if len(uniform_ends) != 2:
        raise ValueError(f"uniform_ends must be two nonnegative integers, got {uniform_ends}")
    for n in uniform_ends:
        if isinstance(n, bool) or not math.isfinite(float(n)):
            raise ValueError(f"uniform_ends must be finite nonnegative integers, got {uniform_ends}")
    n_lo, n_hi = (int(n) for n in uniform_ends)
    if n_lo < 0 or n_hi < 0:
        raise ValueError(f"uniform_ends must be nonnegative, got {uniform_ends}")
    inner_lo, inner_hi = lo + n_lo * spacing, hi - n_hi * spacing
    if not inner_hi > inner_lo:
        raise ValueError(f"uniform_ends {uniform_ends} of {spacing} leave no room in [{lo}, {hi}]")
    tol = 1e-9 * (hi - lo)
    for a, b, _d in regions:
        if a < inner_lo - tol or b > inner_hi + tol:
            raise ValueError(f"fine region [{a}, {b}] is outside [{inner_lo}, {inner_hi}] (the uniform ends)")

    def size(x: np.ndarray) -> np.ndarray:
        """The cell size at x."""
        s = np.full_like(x, spacing)
        for a, b, d in regions:
            s = np.minimum(s, d + (growth - 1.0) * np.maximum(np.maximum(a - x, x - b), 0.0))
        return s

    snap_vals: list[float] = []
    for z in snap:
        zf = float(z)
        if not math.isfinite(zf):
            raise ValueError(f"snap coordinates must be finite, got {z}")
        snap_vals.append(zf)

    points = [inner_lo]
    dropped = []
    for z in sorted(snap_vals):
        if z < inner_lo - tol or z > inner_hi + tol:
            raise ValueError(f"snap coordinate {z} is outside [{inner_lo}, {inner_hi}] (the uniform ends)")
        near = _MERGE * float(size(np.asarray([z]))[0])
        if z - points[-1] < near or inner_hi - z < near:
            if abs(z - points[-1]) > tol and abs(inner_hi - z) > tol:
                dropped.append(z)
            continue
        points.append(z)
    points.append(inner_hi)
    if dropped:
        warnings.warn(
            f"graded_edges: snap coordinates {dropped} are within half a cell of another edge and are not "
            "edges (a sliver of a cell would shorten the time step of the whole simulation)",
            stacklevel=2,
        )

    edges = [lo + spacing * np.arange(n_lo)]
    for p, q in itertools.pairwise(points):
        x = np.linspace(p, q, max(65, math.ceil(64 * (q - p) / float(np.min(size(np.linspace(p, q, 65)))))))
        density = 1.0 / size(x)
        cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(x))])
        n = max(1, math.ceil(cumulative[-1] - 1e-6))
        e = np.interp(np.linspace(0.0, cumulative[-1], n + 1), cumulative, x)
        e[0], e[-1] = p, q
        edges.append(e[:-1])
    edges.append(inner_hi + spacing * np.arange(n_hi))
    edges.append(np.asarray([hi]))
    return np.concatenate(edges)


class AxisEdges(NamedTuple):
    """The cell edges of one axis, from `axis_edges` or `lattice_edges`."""

    #: (n + 1,) strictly increasing edges (um), the first `lo` and the last `hi`
    edges: np.ndarray
    #: uniform cells under the PML at the lower and at the upper end
    pml_cells: tuple[int, int]
    #: the smallest cell (um): this axis's share of the time step
    min_cell: float
    #: the cells of the finest target's width (outside the PMLs) start at whole multiples of it from `lo`
    aligned: bool
    #: the edges are mirror-symmetric about the middle of the axis
    symmetric: bool
    #: the `centers` (um) that lie in the middle of a cell
    centered: tuple[float, ...] = ()


# tolerance of the ratio comparisons (in quanta)
_TOL = 1e-9
# PML widths tried per end when the widest admissible one does not lead to a solution
_PML_TRIES = 6
# centers may bring the smallest cell down to this share of the largest possible one
_CENTER_FLOOR = 0.8


def lattice_edges(
    lo: float,
    hi: float,
    spacing: float,
    fine: Sequence[tuple[float, float, float]] = (),
    *,
    quantum: float = 0.001,
    growth: float = 1.25,
    pml: tuple[float, float] = (0.0, 0.0),
    align: bool = True,
    symmetric: bool = True,
    centers: Sequence[float] = (),
    max_states: float = 2e7,
) -> AxisEdges:
    """Cell edges from ``lo`` to ``hi`` on a lattice of whole quanta, every cell within its target.

    Every cell is a whole number of ``quantum`` wide, counted from ``lo``. The target at x is ``spacing``,
    or ``d_i`` in a ``fine`` region ``[a_i, b_i]`` (the smallest that applies). A cell is no wider than the
    smallest target within half its own width on either side of it, so that the dual cells between
    neighbouring cell centers (where fdtdx samples the E components transverse to the axis) are within
    their targets too. Neighbouring cells differ by at most ``growth``. The PML at each end gets uniform
    cells covering its thickness, as wide as their targets allow.

    Among all such edges the result has the largest smallest cell (which sets this axis's share of the time
    step), then the fewest cells, then the cells closest to their targets (the least sum of squared log
    ratios). Each of `centers` (e.g. a planar interface) lies in the middle of a cell, within half a
    quantum, no edge on it: with subpixel averaging an interface there avoids asymmetric dual-cell
    averaging error and cuts discretization error substantially compared to placing it on a cell edge.
    For them the smallest cell may come out up to 20 % below the largest possible one (two faces a fractional
    number of cells apart need some cells narrower than the rest: a layer 4 cells thick or more fits); the
    most centers that fit are met, the rest dropped with a warning, as are those under a PML. With ``symmetric``, mirror-symmetric targets, PMLs and centers give
    mirror-symmetric edges. With ``align``, cells as wide as the finest target start at whole multiples of it
    from ``lo``, so that they coincide with a uniform grid of that spacing; where that cannot be met it is
    dropped with a warning.

    Args:
        lo: first edge (um).
        hi: last edge (um); ``hi - lo`` must be a whole number of quanta.
        spacing: largest cell (um).
        fine: (a, b, d) regions [a, b] (um) with cells of at most d. A region refines every quantum of the
            lattice it touches.
        quantum: the lattice step (um).
        growth: largest ratio of neighbouring cells (> 1).
        pml: PML thickness (um) at the lower and at the upper end (0: none).
        align: put the cells of the finest target's width on the lattice of that width from ``lo``.
        symmetric: give mirror-symmetric targets and PMLs mirror-symmetric edges.
        centers: coordinates (um) to put in the middle of a cell (within half a quantum).
        max_states: refuse problems with more states (lattice positions times cell widths) than this.

    Returns:
        The edges, the PML cells at both ends, the smallest cell, whether the cells are aligned and
        mirror-symmetric, and the centers met.

    Raises:
        ValueError: on an empty range, a range that is not whole quanta, a target smaller than a quantum, a
            nonpositive spacing or quantum, ``growth <= 1``, PMLs that do not fit, a problem larger than
            ``max_states``, or targets the growth limit cannot meet (the message names where).
    """
    lo, hi, q = float(lo), float(hi), float(quantum)
    spacing, growth = float(spacing), float(growth)
    if not (math.isfinite(q) and q > 0):
        raise ValueError(f"quantum must be finite and positive, got {quantum}")
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        raise ValueError(f"lattice_edges needs finite lo < hi, got {lo} and {hi}")
    if not (math.isfinite(spacing) and spacing > 0):
        raise ValueError(f"spacing must be finite and positive, got {spacing}")
    if not (math.isfinite(growth) and growth > 1):
        raise ValueError(f"growth must be finite and > 1, got {growth}")
    if not (math.isfinite(max_states) and max_states > 0):
        raise ValueError(f"max_states must be finite and positive, got {max_states}")
    n = round((hi - lo) / q)
    if n < 1:
        raise ValueError(f"lattice_edges: hi - lo = {hi - lo:.9g} um must span at least one {q:g} um quantum")
    if abs(n * q - (hi - lo)) > 1e-6 * q:
        raise ValueError(f"lattice_edges: hi - lo = {hi - lo:.9g} um is not a whole number of {q:g} um quanta")
    if len(pml) != 2:
        raise ValueError(f"pml must be two nonnegative thicknesses, got {pml}")
    pml_vals = tuple(float(t) for t in pml)
    if not all(math.isfinite(t) and t >= 0 for t in pml_vals):
        raise ValueError(f"pml must be two finite nonnegative thicknesses, got {pml}")
    thick = tuple(t / q for t in pml_vals)
    targets = _raster(lo, q, n, spacing, fine)
    states = (n + 1) * (int(targets.max()) + 1)
    if states > max_states:
        raise ValueError(
            f"lattice_edges: {n} quanta by {int(targets.max())} cell widths is {states:.3g} states, more than "
            f"max_states={max_states:.3g}; use a coarser quantum"
        )
    # centers in half quanta from lo, inside the range (an edge is there at the ends anyway)
    center_vals: list[float] = []
    for c in centers:
        cf = float(c)
        if not math.isfinite(cf):
            raise ValueError(f"centers must be finite, got {c}")
        center_vals.append(cf)
    half = sorted({round(2 * (c - lo) / q) for c in center_vals})
    half = np.asarray([c for c in half if 0 < c < 2 * n], dtype=np.int64)
    widths, ends, mirror = _Lattice(targets, growth).solve(thick, align, symmetric, lo, q, half)
    # aligned: every cell of the finest target's width outside the PMLs starts on that width's lattice
    fine_k, starts = int(targets.min()), np.concatenate([[0], np.cumsum(widths)[:-1]])
    w_arr = np.asarray(widths)
    inner = slice(ends[0][0], len(widths) - ends[1][0])
    w_in, s_in = w_arr[inner], starts[inner]
    aligned = bool(np.all(s_in[w_in == fine_k] % fine_k == 0))
    starts2 = 2 * starts
    ends2 = starts2 + 2 * w_arr
    middles = starts2 + w_arr  # cell middles in half quanta
    hit = [
        bool(np.any((w_arr >= 2) & (starts2 < c) & (c < ends2) & (np.abs(middles - c) <= 1)))
        for c in half
    ]
    met = [int(c) for c, h in zip(half, hit, strict=True) if h]
    missed = [lo + c * q / 2 for c, h in zip(half, hit, strict=True) if not h]
    if missed:
        warnings.warn(
            f"lattice_edges: {', '.join(f'{c:.6g}' for c in missed)} um not in the middle of a cell (under a PML, or "
            "no room within the targets and the growth limit at the largest smallest cell)",
            stacklevel=2,
        )
    if align and not aligned:
        warnings.warn(
            f"lattice_edges: the {int(targets.min()) * q:g} um cells do not all start at whole multiples of "
            f"their width from {lo:g} um (the largest smallest cell and mirror symmetry come first)",
            stacklevel=2,
        )
    edges = lo + q * np.concatenate([[0], np.cumsum(widths)]).astype(np.float64)
    edges[0], edges[-1] = lo, hi
    centered = tuple(round(lo + c * q / 2, 12) for c in met)
    return AxisEdges(edges, (ends[0][0], ends[1][0]), q * min(widths), aligned, mirror, centered)


def _raster(lo: float, q: float, n: int, spacing: float, fine: Sequence[tuple[float, float, float]]) -> np.ndarray:
    """(n,) targets in whole quanta of the lattice's cells: the smallest region touching each one."""
    size = np.full(n, spacing)
    for region in fine:
        if len(region) != 3:
            raise ValueError(f"a fine region is (a, b, d), got {tuple(region)}")
        a, b, d = (float(x) for x in region)
        if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(d) and b >= a and d > 0):
            raise ValueError(f"a fine region is (a, b, d) with finite a <= b and d > 0, got {tuple(region)}")
        ja = max(0, math.ceil((a - lo) / q - 1e-9) - 1)
        jb = min(n - 1, math.floor((b - lo) / q + 1e-9))
        if ja <= jb:
            size[ja : jb + 1] = np.minimum(size[ja : jb + 1], d)
    targets = np.floor(size / q + 1e-9).astype(np.int64)
    if targets.min() < 1:
        raise ValueError(f"lattice_edges: a target of {size.min():g} um is smaller than the quantum {q:g} um")
    return targets


def _sparse_table(x: np.ndarray, fill: float) -> np.ndarray:
    """(levels, len(x)) table of minima: row j holds min(x[i : i + 2**j]) (``fill`` past the end)."""
    m = len(x)
    rows = [np.asarray(x)]
    j = 1
    while (1 << j) <= m:
        prev, half = rows[-1], 1 << (j - 1)
        row = np.full(m, fill, dtype=np.result_type(x, type(fill)))
        row[: m - (1 << j) + 1] = np.minimum(prev[: m - (1 << j) + 1], prev[half : half + m - (1 << j) + 1])
        rows.append(row)
        j += 1
    return np.stack(rows)


def _range_min(table: np.ndarray, start: np.ndarray, stop: np.ndarray) -> np.ndarray:
    """min(x[start_i : stop_i]) from a :func:`_sparse_table`, for stop_i > start_i."""
    level = np.frexp((stop - start).astype(np.float64))[1] - 1
    return np.minimum(table[level, start], table[level, stop - (1 << level)])


def _window_min(w: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """min(w[a_i : b_i + 1]) for each i, inf where a_i > b_i."""
    out = np.full(len(a), np.inf)
    ok = a <= b
    if ok.any():
        out[ok] = _range_min(_sparse_table(w, np.inf), a[ok], b[ok] + 1)
    return out


def _centered(centers: np.ndarray, start2: int, k: np.ndarray) -> np.ndarray:
    """Whether the cells of k quanta from start2 / 2 hold no center (half quanta, sorted) but one strictly inside
    and within half a quantum of their middle (requiring width >= 2 quanta so edges cannot masquerade as centers)."""
    i = int(np.searchsorted(centers, start2, side="left"))
    first = centers[i] if i < len(centers) else np.iinfo(np.int64).max
    second = centers[i + 1] if i + 1 < len(centers) else np.iinfo(np.int64).max
    end2 = start2 + 2 * k
    return (first > end2) | (
        (k >= 2) & (first > start2) & (first < end2) & (np.abs(first - (start2 + k)) <= 1) & (second > end2)
    )


def _pick(options: list[tuple[int, int]], i: int, growth: float) -> tuple[int, int]:
    """The i-th PML choice: the widest admissible cells at most growth**-i of the widest."""
    widest = options[0][1]
    for cells, width in options:
        if width <= widest * growth**-i + _TOL:
            return cells, width
    return options[-1]


class _Lattice:
    """The admissible cells of one axis: targets in quanta, cells checked over their half-width halo."""

    def __init__(self, targets: np.ndarray, growth: float) -> None:
        self.targets = targets
        self.n = len(targets)
        self.k_max = int(targets.max())
        self.k_fine = int(targets.min())
        self.growth = growth
        self.mirrored = bool(np.array_equal(targets, targets[::-1]))
        # window[p, k]: the smallest target within half a cell of the cell of k quanta from p (0: does not fit)
        self.window = np.zeros((self.n, self.k_max + 1), dtype=np.int32)
        table = _sparse_table(targets, np.iinfo(np.int64).max)
        p = np.arange(self.n)
        for k in range(1, self.k_max + 1):
            pk = p[p + k <= self.n]
            half = (k + 1) // 2
            start, stop = np.maximum(pk - half, 0), np.minimum(pk + k + half, self.n)
            self.window[pk, k] = _range_min(table, start, stop)
        ok = self.window[:, 1:] >= np.arange(1, self.k_max + 1)[None, :]
        # the widest admissible cell from each position (admissible widths are a prefix: the window only grows)
        self.widest = np.cumprod(ok, axis=1).sum(axis=1)

    def cost(self, p: int, k: np.ndarray | int) -> np.ndarray | float:
        """Squared log ratio of a cell's target to its width."""
        return np.log(self.window[p, k] / k) ** 2

    def pml_options(self, end: int, thickness: float, k_lo: int) -> list[tuple[int, int]]:
        """(cells, width) of the PML at one end, for every admissible uniform width >= k_lo, widest first."""
        if thickness <= 0:
            return [(0, 0)]
        out = []
        for width in range(self.k_max, k_lo - 1, -1):
            cells = max(1, math.ceil(thickness / width - _TOL))
            if cells * width > self.n:
                continue
            starts = np.arange(cells) * width if end == 0 else self.n - (np.arange(cells) + 1) * width
            if np.all(self.widest[starts] >= width):
                out.append((cells, width))
        return out

    def ends(self, thickness: tuple[float, float], k_lo: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """The PML choices to try at both ends together: the widest first."""
        lo, hi = (self.pml_options(e, thickness[e], k_lo) for e in (0, 1))
        if not lo or not hi:
            return []
        out = []
        for i in range(_PML_TRIES):
            pair = (_pick(lo, i, self.growth), _pick(hi, i, self.growth))
            if pair not in out:
                out.append(pair)
        return out

    def dp(
        self,
        k_lo: int,
        ends: tuple[tuple[int, int], tuple[int, int]],
        align: int | None,
        mirror: bool,
        eps: float,
        values_only: bool = False,
        centers: np.ndarray | None = None,
    ) -> list[int] | None:
        """The best cell widths (quanta) with every cell >= k_lo, or None if there are none.

        Backward over lattice positions, the state being the width of the previous cell; ``mirror``
        solves the lower half (each cell counting twice) and ends on the middle edge or a middle cell.
        ``centers`` (half quanta, sorted) must each lie in the middle of an interior cell.
        """
        centers = np.zeros(0, dtype=np.int64) if centers is None else centers
        (n_lo, w_lo), (n_hi, w_hi) = ends
        n, g, k_max = self.n, self.growth, self.k_max
        s0, e0 = n_lo * w_lo, n - n_hi * w_hi
        top = n // 2 if mirror else e0
        if s0 > top or s0 > e0 or (n_lo and not k_lo <= w_lo <= k_max):
            return None
        ks = np.arange(k_lo, k_max + 1)
        nk = len(ks)
        band_lo = np.maximum(np.ceil(ks / g - _TOL).astype(np.int64), k_lo) - k_lo
        band_hi = np.floor(ks * g + _TOL).astype(np.int64) - k_lo
        weight = 2.0 if mirror else 1.0
        value = np.full((top - s0 + 1, nk + 1), np.inf)  # column nk: no previous cell

        def joins(c: int) -> np.ndarray:
            return (c <= g * ks + _TOL) & (ks <= g * c + _TOL)

        def options(p: int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
            """(value of ending here per state, candidate widths, value of each candidate)."""
            term = np.full(nk + 1, np.inf)
            if mirror:
                if 2 * p == n and not np.any(np.abs(centers - n) <= 1):  # the middle edge, unless on a center
                    term[:] = 0.0
                c = n - 2 * p
                if (k_lo <= c <= self.widest[p] and not (align and c == align and p % align)
                        and (not len(centers) or _centered(centers, 2 * p, np.array([c]))[0])):
                    v = 1.0 + eps * float(self.cost(p, c))
                    term[:nk] = np.where(joins(c), np.minimum(term[:nk], v), term[:nk])
                    term[nk] = min(term[nk], v)
            elif p == e0:
                term[:nk] = np.where(joins(w_hi), 0.0, np.inf) if n_hi else 0.0
                term[nk] = 0.0
            widest = min(int(self.widest[p]), top - p) if p < n else 0
            if widest < k_lo:
                return term, None, None
            cand = np.arange(k_lo, widest + 1)
            w = value[p - s0 + cand, cand - k_lo] + weight * (1.0 + eps * self.cost(p, cand))
            if align and p % align and k_lo <= align <= widest:
                w[align - k_lo] = np.inf
            if len(centers):
                w = np.where(_centered(centers, 2 * p, cand), w, np.inf)
            return term, cand, w

        for p in range(top, s0 - 1, -1):
            term, cand, w = options(p)
            if cand is not None:
                term[:nk] = np.minimum(term[:nk], _window_min(w, band_lo, np.minimum(band_hi, len(cand) - 1)))
                term[nk] = min(term[nk], float(w.min()))
            value[p - s0] = term
        col = w_lo - k_lo if n_lo else nk
        if not np.isfinite(value[0, col]):
            return None
        if values_only:
            return []
        widths: list[int] = []
        p = s0
        while True:
            term, cand, w = options(p)
            best, pick = np.inf, 0
            if cand is not None:
                a = 0 if col == nk else int(band_lo[col])
                b = len(cand) - 1 if col == nk else min(int(band_hi[col]), len(cand) - 1)
                if a <= b:
                    j = a + int(np.argmin(w[a : b + 1]))
                    best, pick = float(w[j]), int(cand[j])
            if term[col] <= best:
                break
            widths.append(pick)
            p += pick
            col = pick - k_lo
        if mirror:
            middle = [] if 2 * p == n else [n - 2 * p]
            widths = widths + middle + widths[::-1]
        return [w_lo] * n_lo + widths + [w_hi] * n_hi

    def furthest(self, ends: tuple[tuple[int, int], tuple[int, int]]) -> int:
        """The furthest lattice position any admissible cell sequence reaches (for error messages)."""
        (n_lo, w_lo), (n_hi, w_hi) = ends
        s0, e0, g = n_lo * w_lo, self.n - n_hi * w_hi, self.growth
        if s0 > e0:
            return s0
        reached = np.zeros((e0 - s0 + 1, self.k_max + 1), dtype=bool)  # column 0: no previous cell
        reached[0, w_lo if n_lo else 0] = True
        far = s0
        for p in range(s0, e0 + 1):
            prev = np.nonzero(reached[p - s0])[0]
            if not len(prev):
                continue
            far = p
            widest = min(int(self.widest[p]) if p < self.n else 0, e0 - p)
            for kp in prev:
                a = 1 if kp == 0 else max(1, math.ceil(kp / g - _TOL))
                b = widest if kp == 0 else min(widest, math.floor(kp * g + _TOL))
                for k in range(a, b + 1):
                    reached[p + k - s0, k] = True
        return far

    def solve(
        self, thickness: tuple[float, float], align: bool, symmetric: bool, lo: float, q: float,
        centers: np.ndarray | None = None,
    ) -> tuple[list[int], tuple[tuple[int, int], tuple[int, int]], bool]:
        """(widths, PML ends, mirrored): the largest smallest cell first (centers may take it down to
        _CENTER_FLOOR of that), then as many centers as fit, then the best widths."""
        centers = np.zeros(0, dtype=np.int64) if centers is None else centers
        eps = 0.25 / (2.0 * self.n * (math.log(self.k_max) ** 2 + 1.0) + 1.0)

        def feasible(k_lo: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
            for ends in self.ends(thickness, k_lo):
                if self.dp(k_lo, ends, None, False, eps, values_only=True) is not None:
                    return ends
            return None

        found = None
        for k in range(self.k_fine, max(0, self.k_fine - 3), -1):
            ends = feasible(k)
            if ends is not None:
                found = (k, ends)
                break
        if found is None:
            if feasible(1) is None:
                self._raise_infeasible(thickness, lo, q)
            a, b = 1, max(1, self.k_fine - 3)
            while a < b:
                mid = (a + b + 1) // 2
                if feasible(mid) is not None:
                    a = mid
                else:
                    b = mid - 1
            found = (a, feasible(a))
        k_star, ends_star = found

        def choices(k_lo: int, ends: tuple) -> list:
            """The PML choices to try: the one found first, then narrower ones (the fine cells' lattice or a
            center may need narrower PML cells to be reached within the growth limit)."""
            return [ends, *(e for e in self.ends(thickness, k_lo) if e != ends)]

        def attempt(k_lo: int, options: list, cs: np.ndarray) -> tuple | None:
            """The first mode that meets every center in cs: mirrored before not, aligned before not."""
            mirrored_centers = np.array_equal(np.sort(2 * self.n - cs), cs)
            modes: list[tuple[bool, int | None, tuple]] = []
            if symmetric and self.mirrored and thickness[0] == thickness[1] and mirrored_centers:
                if align and self.n % self.k_fine == 0:
                    modes += [(True, self.k_fine, e) for e in options if e[0] == e[1]]
                modes += [(True, None, e) for e in options if e[0] == e[1]][: 1 if not len(cs) else None]
            if align:
                modes += [(False, self.k_fine, e) for e in options]
            modes += [(False, None, e) for e in (options if len(cs) else options[:1])]
            for mirror, aligned, pml in modes:
                widths = self.dp(k_lo, pml, aligned, mirror, eps, centers=cs)
                if widths is not None:
                    return widths, pml, mirror
            return None

        best = choices(k_star, ends_star)
        if len(centers):
            # the most centers, then the largest smallest cell down to _CENTER_FLOOR of the largest possible
            kept_best: tuple[int, int, list, list[int]] = (0, k_star, best, [])
            for k in range(k_star, max(1, math.ceil(_CENTER_FLOOR * k_star)) - 1, -1):
                ends_k = ends_star if k == k_star else feasible(k)
                if ends_k is None:
                    continue
                opts = choices(k, ends_k)

                def fits(cs: list[int], k: int = k, opts: list = opts) -> bool:
                    trial = np.array(sorted(cs), dtype=np.int64)
                    return any(self.dp(k, e, None, False, eps, values_only=True, centers=trial) is not None
                               for e in opts)

                kept = centers.tolist() if fits(centers.tolist()) else []
                if not kept:  # not all at once: keep each center that still fits with those kept before it
                    for c in centers.tolist():
                        if fits([*kept, c]):
                            kept.append(c)
                if len(kept) > kept_best[0]:
                    kept_best = (len(kept), k, opts, kept)
                if len(kept) == len(centers):
                    break
            if kept_best[0]:
                _, k, opts, kept = kept_best
                got = attempt(k, opts, np.array(sorted(kept), dtype=np.int64))
                if got is not None:
                    return got
        got = attempt(k_star, best, np.zeros(0, dtype=np.int64))
        if got is not None:
            return got
        raise AssertionError("lattice_edges: a feasible problem lost its solution")  # pragma: no cover

    def _raise_infeasible(self, thickness: tuple[float, float], lo: float, q: float) -> None:
        tries = self.ends(thickness, 1)
        if not tries:
            raise ValueError(
                f"lattice_edges: PMLs of {thickness[0] * q:g} and {thickness[1] * q:g} um do not fit the "
                f"{self.n * q:g} um axis on whole cells within their targets"
            )
        ends = tries[0]
        if ends[0][0] * ends[0][1] > self.n - ends[1][0] * ends[1][1]:
            raise ValueError(f"lattice_edges: the PMLs at both ends overlap on the {self.n * q:g} um axis")
        far = self.furthest(ends)
        raise ValueError(
            f"lattice_edges: no cells within their targets and the growth limit {self.growth:g} reach past "
            f"{lo + far * q:.6g} um (PML cells {ends[0][1] * q:g} um x {ends[0][0]} and "
            f"{ends[1][1] * q:g} um x {ends[1][0]}); the targets change too fast for the growth limit there"
        )



# ---------------------------------------------------------------- continuous axis solver


class _TargetMap:
    """Piecewise-constant target cap ``T(x)`` on ``[lo, hi]`` with ``O(log M)`` closed-interval queries."""

    def __init__(
        self,
        lo: float,
        hi: float,
        spacing: float,
        fine: Sequence[tuple[float, float, float]],
    ) -> None:
        self.lo = lo
        self.hi = hi
        self.length = hi - lo
        self.mid = 0.5 * (lo + hi)
        self.spacing = spacing
        coord_tol = 1e-12 * max(1.0, self.length, abs(lo), abs(hi))

        clipped: list[tuple[float, float, float]] = []
        pts = [lo, hi]
        for region in fine:
            if len(region) != 3:
                raise ValueError(f"a fine region is (a, b, d), got {tuple(region)}")
            a, b, d = (float(x) for x in region)
            if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(d) and b >= a and d > 0):
                raise ValueError(f"a fine region is (a, b, d) with a <= b and d > 0, got {(a, b, d)}")
            if b < lo - coord_tol or a > hi + coord_tol:
                continue
            ac = min(max(a, lo), hi)
            bc = min(max(b, lo), hi)
            dc = min(d, spacing)
            clipped.append((ac, bc, dc))
            pts.extend((ac, bc))

        pts_sorted = sorted(pts)
        xp_list = [pts_sorted[0]]
        for x in pts_sorted[1:]:
            if x - xp_list[-1] > coord_tol:
                xp_list.append(x)
        xp_list[0], xp_list[-1] = lo, hi
        if len(xp_list) == 1:
            xp_list.append(hi)
        xp = np.asarray(xp_list, dtype=np.float64)
        m = len(xp) - 1

        node_d = np.full(m + 1, spacing, dtype=np.float64)
        seg_d = np.full(m, spacing, dtype=np.float64)
        seg_mid = 0.5 * (xp[:-1] + xp[1:])
        for ac, bc, dc in clipped:
            hit_node = (xp >= ac - coord_tol) & (xp <= bc + coord_tol)
            node_d[hit_node] = np.minimum(node_d[hit_node], dc)
            if bc > ac + coord_tol:
                hit_seg = (seg_mid >= ac - coord_tol) & (seg_mid <= bc + coord_tol)
                seg_d[hit_seg] = np.minimum(seg_d[hit_seg], dc)
        node_d[:-1] = np.minimum(node_d[:-1], seg_d)
        node_d[1:] = np.minimum(node_d[1:], seg_d)

        interleaved = np.empty(2 * m + 1, dtype=np.float64)
        interleaved[0::2] = node_d
        interleaved[1::2] = seg_d

        self.xp = xp
        self.node_d = node_d
        self.seg_d = seg_d
        self.interleaved = interleaved
        self._table = _sparse_table(interleaved, np.inf)
        self.d_fine = float(interleaved.min())
        self.is_constant = bool(float(interleaved.max()) <= self.d_fine * (1.0 + 1e-13))

        merged: list[tuple[float, float, float]] = []
        i = 0
        while i < m:
            d_cur = float(seg_d[i])
            j = i + 1
            while j < m and abs(float(seg_d[j]) - d_cur) <= 1e-13 * max(1.0, d_cur):
                j += 1
            if d_cur < spacing * (1.0 - 1e-13):
                merged.append((float(xp[i]), float(xp[j]), d_cur))
            i = j
        for j, xj in enumerate(xp):
            nd = float(node_d[j])
            adj = min(
                float(seg_d[j - 1]) if j > 0 else spacing,
                float(seg_d[j]) if j < m else spacing,
            )
            if nd < adj * (1.0 - 1e-13):
                merged.append((float(xj), float(xj), nd))
        merged.sort(key=lambda r: (r[0], r[1], r[2]))
        self.merged_fine = merged

    def is_symmetric(self, tol: float) -> bool:
        if not np.allclose(self.xp - self.lo, (self.hi - self.xp)[::-1], atol=tol, rtol=0.0):
            return False
        return bool(
            np.allclose(self.seg_d, self.seg_d[::-1], rtol=1e-9, atol=tol)
            and np.allclose(self.node_d, self.node_d[::-1], rtol=1e-9, atol=tol)
        )

    def min_on_vec(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        uc = np.clip(np.asarray(u, dtype=np.float64), self.lo, self.hi)
        vc = np.clip(np.asarray(v, dtype=np.float64), self.lo, self.hi)
        uc, vc = np.minimum(uc, vc), np.maximum(uc, vc)
        n_int = len(self.interleaved) - 1
        ia = np.clip(
            np.searchsorted(self.xp, uc, side="left") + np.searchsorted(self.xp, uc, side="right") - 1,
            0,
            n_int,
        )
        ib = np.clip(
            np.searchsorted(self.xp, vc, side="left") + np.searchsorted(self.xp, vc, side="right") - 1,
            ia,
            n_int,
        )
        return _range_min(self._table, ia, ib + 1)

    def min_on(self, u: float, v: float) -> float:
        return float(self.min_on_vec(np.array([u]), np.array([v]))[0])


class _SizingEnvelope:
    """Analytic piecewise-linear upper-bound sizing field ``s(x)`` and cumulative density ``F(x)``."""

    def __init__(
        self,
        tmap: _TargetMap,
        spacing_eff: float,
        g_eff: float,
        cap_margin: float,
        *,
        periodic: bool = False,
        extra_caps: Sequence[tuple[float, float, float]] = (),
    ) -> None:
        lo, hi, L = tmap.lo, tmap.hi, tmap.length
        alpha = min(1.0, 2.0 * (g_eff - 1.0) / (g_eff + 1.0))
        has_alpha = alpha < 1.0 - 1e-12

        items: list[tuple[float, float, float]] = []
        knots = [lo, hi]
        shifts = (-L, 0.0, L) if periodic else (0.0,)
        regions = [(a0, b0, min(spacing_eff, max(d - cap_margin, 0.5 * d))) for a0, b0, d in tmap.merged_fine] + [
            (a0, b0, min(spacing_eff, de_cap)) for a0, b0, de_cap in extra_caps
        ]
        for a0, b0, de in regions:
            if de >= spacing_eff * (1.0 - 1e-13):
                continue
            d_star = 0.25 * (g_eff + 3.0) * de if has_alpha else spacing_eff
            d_max = (0.5 * de + (spacing_eff - de) / alpha) if has_alpha else spacing_eff
            for sh in shifts:
                a, b = a0 + sh, b0 + sh
                if a - d_max >= hi or b + d_max <= lo:
                    continue
                items.append((a, b, de))
                for off in (0.0, de, min(d_star, d_max), d_max):
                    if lo < a - off < hi:
                        knots.append(a - off)
                    if lo < b + off < hi:
                        knots.append(b + off)

        coord_tol = 1e-12 * max(1.0, hi - lo, abs(lo), abs(hi))
        knots.sort()
        base = [knots[0]]
        for z in knots[1:]:
            if z - base[-1] > coord_tol:
                base.append(z)
        base[0], base[-1] = lo, hi

        # Refine each interval between critical points with exact lower-envelope intersections of active lines.
        refined = [base[0]]
        for za, zb in itertools.pairwise(base):
            zm = 0.5 * (za + zb)
            # Group active lines y = m * x + c by slope m in {0, -1, +1, -alpha, +alpha}.
            best_lines: dict[float, tuple[float, float]] = {0.0: (0.0, spacing_eff)}
            for a, b, de in items:
                if zm < a:
                    dist = a - zm
                    if dist <= de:
                        m_s, c_s = 0.0, de
                    elif not has_alpha or dist <= 0.25 * (g_eff + 3.0) * de:
                        m_s, c_s = -1.0, a
                    else:
                        m_s, c_s = -alpha, de + alpha * (a - 0.5 * de)
                elif zm > b:
                    dist = zm - b
                    if dist <= de:
                        m_s, c_s = 0.0, de
                    elif not has_alpha or dist <= 0.25 * (g_eff + 3.0) * de:
                        m_s, c_s = 1.0, -b
                    else:
                        m_s, c_s = alpha, de - alpha * (b + 0.5 * de)
                else:
                    m_s, c_s = 0.0, de
                val = m_s * zm + c_s
                if val < spacing_eff + 1e-14:
                    prev = best_lines.get(m_s)
                    if prev is None or val < prev[0] * zm + prev[1]:
                        best_lines[m_s] = (m_s, c_s)
            lines = list(best_lines.values())
            splits = []
            for idx1 in range(len(lines)):
                m1, c1 = lines[idx1]
                for idx2 in range(idx1 + 1, len(lines)):
                    m2, c2 = lines[idx2]
                    if abs(m1 - m2) > 1e-14:
                        xi = (c2 - c1) / (m1 - m2)
                        if za + coord_tol < xi < zb - coord_tol:
                            splits.append(xi)
            for xi in sorted(splits):
                if xi - refined[-1] > coord_tol:
                    refined.append(xi)
            if zb - refined[-1] > coord_tol:
                refined.append(zb)
        refined[-1] = hi
        z_arr = np.asarray(refined, dtype=np.float64)

        s_arr = np.full_like(z_arr, spacing_eff)
        for a, b, de in items:
            dist = np.maximum(0.0, np.maximum(a - z_arr, z_arr - b))
            ramp = np.minimum(dist, de + alpha * (dist - 0.5 * de)) if has_alpha else dist
            s_arr = np.minimum(s_arr, np.maximum(de, ramp))

        dz = np.diff(z_arr)
        sa, sb = s_arr[:-1], s_arr[1:]
        rel = (sb - sa) / sa
        small = np.abs(rel) < 1e-7
        df = np.empty_like(dz)
        if np.any(small):
            r = rel[small]
            df[small] = (dz[small] / sa[small]) * (1.0 - 0.5 * r + (r * r) / 3.0 - (r * r * r) / 4.0)
        if np.any(~small):
            df[~small] = dz[~small] * np.log1p(rel[~small]) / (sb[~small] - sa[~small])

        self.z = z_arr
        self.s = s_arr
        self.F = np.concatenate([[0.0], np.cumsum(df)])
        self.total_F = float(self.F[-1])

    def eval_s(self, x: np.ndarray | float) -> np.ndarray:
        return np.interp(x, self.z, self.s)

    def eval_F(self, x: np.ndarray | float) -> np.ndarray:
        xv = np.clip(np.asarray(x, dtype=np.float64), self.z[0], self.z[-1])
        idx = np.clip(np.searchsorted(self.z, xv, side="right") - 1, 0, len(self.z) - 2)
        dx = xv - self.z[idx]
        sa = self.s[idx]
        slope = (self.s[idx + 1] - sa) / (self.z[idx + 1] - self.z[idx])
        rel = slope * dx / sa
        small = np.abs(rel) < 1e-7
        part = np.empty_like(xv)
        if np.any(small):
            r = rel[small]
            part[small] = (dx[small] / sa[small]) * (1.0 - 0.5 * r + (r * r) / 3.0)
        if np.any(~small):
            part[~small] = np.log1p(rel[~small]) / slope[~small]
        return self.F[idx] + part

    def inv_F(self, u: np.ndarray) -> np.ndarray:
        uv = np.clip(np.asarray(u, dtype=np.float64), 0.0, self.total_F)
        idx = np.clip(np.searchsorted(self.F, uv, side="right") - 1, 0, len(self.z) - 2)
        du = uv - self.F[idx]
        dz = self.z[idx + 1] - self.z[idx]
        sa = self.s[idx]
        sb = self.s[idx + 1]
        slope = (sb - sa) / dz
        rel = (sb - sa) / sa
        small = np.abs(rel) < 1e-7
        dx = np.empty_like(uv)
        if np.any(small):
            df_seg = np.maximum(self.F[idx + 1] - self.F[idx], 1e-30)
            f = du[small] / df_seg[small]
            r = rel[small]
            dx[small] = dz[small] * f * (1.0 + 0.5 * r * (f - 1.0))
        if np.any(~small):
            dx[~small] = dz[~small] * np.expm1(slope[~small] * du[~small]) / rel[~small]
        return np.clip(self.z[idx] + dx, self.z[0], self.z[-1])


def _active_set_qp(
    Q: np.ndarray,
    q: np.ndarray,
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    x0: np.ndarray,
) -> np.ndarray:
    """Minimize ``0.5 x^T Q x - q^T x`` subject to ``A_eq x = b_eq`` and ``A_ub x <= b_ub`` from feasible ``x0``."""
    from scipy import linalg

    v = len(x0)
    if v == 0:
        return x0.copy()
    x = x0.astype(np.float64, copy=True)
    n_eq = len(b_eq)
    if n_eq:
        res_eq = b_eq - A_eq @ x
        if np.max(np.abs(res_eq)) > 1e-13:
            dx, *_ = np.linalg.lstsq(A_eq, res_eq, rcond=1e-12)
            if len(b_ub) == 0 or np.max(A_ub @ (x + dx) - b_ub) <= 1e-10:
                x = x + dx

    def obj(z: np.ndarray) -> float:
        return float(0.5 * (z @ (Q @ z)) - q @ z)

    # Seed working set with linearly independent tight inequalities using incremental QR.
    work: list[int] = []
    if len(b_ub):
        slacks = b_ub - A_ub @ x
        cand_rows = A_eq.copy() if n_eq else np.zeros((0, v), dtype=np.float64)
        cur_rank = 0
        if n_eq:
            _, r0 = np.linalg.qr(cand_rows.T, mode="reduced")
            cur_rank = int(np.sum(np.abs(np.diag(r0)) > 1e-10))
        for i in np.argsort(slacks):
            if slacks[i] > 1e-11 or cur_rank >= v:
                break
            nxt = np.vstack([cand_rows, A_ub[i : i + 1]])
            _, rn = np.linalg.qr(nxt.T, mode="reduced")
            rk = int(np.sum(np.abs(np.diag(rn)) > 1e-10))
            if rk > cur_rank:
                work.append(int(i))
                cand_rows = nxt
                cur_rank = rk

    max_iter = min(350, 4 * v + 25)
    zero_steps = 0
    for _ in range(max_iter):
        grad = Q @ x - q  # gradient of 0.5 x^T Q x - q^T x
        act = np.vstack([A_eq, A_ub[work]]) if work else A_eq
        m_act = len(act)
        if m_act:
            # Null-space projection: Q_null^T Q Q_null is strictly SPD (no saddle-point ill-conditioning).
            q_mat, r_mat, piv = linalg.qr(act.T, mode="full", pivoting=True)
            diag_r = np.abs(np.diag(r_mat)) if min(r_mat.shape) else np.zeros(0)
            rk = int(np.sum(diag_r > 1e-10))
            if rk < v:
                q_null = q_mat[:, rk:]
                q_red = q_null.T @ (Q @ q_null)
                rhs_red = -(q_null.T @ grad)
                try:
                    z_null = np.linalg.solve(q_red, rhs_red)
                except np.linalg.LinAlgError:
                    z_null, *_ = np.linalg.lstsq(q_red, rhs_red, rcond=1e-12)
                p = q_null @ z_null
            else:
                p = np.zeros(v, dtype=np.float64)
            lam_full = np.zeros(m_act, dtype=np.float64)
            if rk > 0 and float(np.linalg.norm(p)) <= 1e-11:
                # KKT stationarity: Q x - q + act^T lam = 0  =>  R[:rk, :rk] lam_piv[:rk] = -Q_range^T grad
                lam_piv = linalg.solve_triangular(r_mat[:rk, :rk], -(q_mat[:, :rk].T @ grad))
                lam_full[piv[:rk]] = lam_piv
            lam = lam_full[n_eq:]
        else:
            try:
                p = np.linalg.solve(Q, -grad)
            except np.linalg.LinAlgError:
                p, *_ = np.linalg.lstsq(Q, -grad, rcond=1e-12)
            lam = np.zeros(0, dtype=np.float64)

        if float(np.linalg.norm(p)) <= 1e-11:
            if len(lam) == 0 or float(np.min(lam)) >= -1e-9:
                break
            drop_pos = int(np.argmin(lam))
            work.pop(drop_pos)
            zero_steps = 0
            continue

        alpha = 1.0
        blocker = -1
        if len(b_ub):
            in_work = set(work)
            ap = A_ub @ p
            slack = b_ub - A_ub @ x
            for i in range(len(b_ub)):
                if i in in_work or ap[i] <= 1e-13:
                    continue
                step_i = max(0.0, float(slack[i] / ap[i]))
                if step_i < alpha - 1e-14 or (abs(step_i - alpha) <= 1e-14 and (blocker < 0 or i < blocker)):
                    alpha = step_i
                    blocker = i
        x = x + alpha * p
        if alpha < 1.0 - 1e-12 and blocker >= 0:
            zero_steps = zero_steps + 1 if alpha <= 1e-13 else 0
            if zero_steps > 6:
                break
            work.append(blocker)

    eq_ok = not n_eq or float(np.max(np.abs(A_eq @ x - b_eq))) <= 1e-10
    ub_ok = not len(b_ub) or float(np.max(A_ub @ x - b_ub)) <= 1e-10
    if eq_ok and ub_ok and obj(x) <= obj(x0) + 1e-12:
        return x
    return x0


def _pml_candidates(
    tmap: _TargetMap,
    env: _SizingEnvelope,
    end: int,
    thickness: float,
    spacing_eff: float,
    cap_margin: float,
    pml_pad: float,
    g_eff: float,
) -> list[tuple[int, float, float]]:
    """Admissible ``(n_pml, p_min, p_cap)`` tuples at ``end`` (0: lo, 1: hi), widest first."""
    if thickness <= 0:
        return [(0, 0.0, 0.0)]
    t_req = thickness + pml_pad
    if t_req > tmap.length * (1.0 + 1e-12):
        return []

    d_fine_eff = min(spacing_eff, max(tmap.d_fine - cap_margin, 0.5 * tmap.d_fine))

    def p_cap_of(n: int) -> float:
        hi_p = min(spacing_eff, tmap.length / n)

        def ok(p: float) -> bool:
            reach = min(tmap.length, (n + 0.5) * p)
            raw = tmap.min_on(tmap.lo, tmap.lo + reach) if end == 0 else tmap.min_on(tmap.hi - reach, tmap.hi)
            ce = min(spacing_eff, max(raw - cap_margin, 0.5 * raw))
            return p <= ce + 1e-15

        if ok(hi_p):
            return hi_p
        lo_p = 0.0
        for _ in range(54):
            mid_p = 0.5 * (lo_p + hi_p)
            if ok(mid_p):
                lo_p = mid_p
            else:
                hi_p = mid_p
        return max(0.0, lo_p - 2e-11 * max(1.0, tmap.length) / (n + 0.5))

    outer = (
        tmap.min_on(tmap.lo, min(tmap.hi, tmap.lo + t_req))
        if end == 0
        else tmap.min_on(max(tmap.lo, tmap.hi - t_req), tmap.hi)
    )
    outer_eff = min(spacing_eff, max(outer - cap_margin, 0.5 * outer))
    n0 = max(1, math.ceil(t_req / outer_eff - 1e-12))
    while n0 <= max(100_000, int(4 * t_req / max(tmap.d_fine, 1e-12)) + 10):
        pc = p_cap_of(n0)
        if pc >= t_req / n0 - 1e-14:
            break
        n_jump = math.ceil(t_req / max(pc, 1e-15) - 1e-12)
        n0 = max(n0 + 1, n_jump)
    else:
        return []

    s_inner = float(env.eval_s(tmap.lo + min(tmap.length, t_req) if end == 0 else tmap.hi - min(tmap.length, t_req)))
    n_env = max(n0, math.ceil(t_req / max(s_inner, 1e-15) - 1e-12))
    n_fine = max(n0, math.ceil(t_req / max(d_fine_eff, 1e-15) - 1e-12))

    # Try integer PML cell counts from coarsest n0 through n_env + 3, plus n_fine as a last-resort fallback.
    raw_counts = set(range(n0, min(n0 + 10, max(n0, n_env) + 4)))
    raw_counts.update({max(n0, n_env - 1), n_env, n_env + 1, n_env + 2, n_env + 3})
    raw_counts.update({max(n0, math.ceil(n0 * g_eff - 1e-12)), n_fine, n_fine + 1})

    out: list[tuple[int, float, float]] = []
    for n in sorted(raw_counts):
        pc = p_cap_of(n)
        pm = t_req / n
        if pm <= pc + 1e-14 and n * pm <= tmap.length + 1e-12:
            out.append((n, min(pm, pc), pc))
    return out


def _solve_blocks(
    tmap: _TargetMap,
    env: _SizingEnvelope,
    pml_lo_opt: tuple[int, float, float],
    pml_hi_opt: tuple[int, float, float],
    centers: tuple[float, ...],
    span_counts: tuple[int, ...],
    *,
    mirror: bool,
    periodic: bool,
    spacing_eff: float,
    g_eff: float,
    cap_margin: float,
    w_floor: float,
    pinned_runs: Sequence[tuple[int, int, float, float]] = (),
    stage2: bool = True,
) -> tuple[np.ndarray, float, float] | None:
    """Solve Stage 1 LP (+ optional Stage 2 QP); return ``(edges, min_cell, m_stage1)``."""
    from scipy import optimize, sparse

    lo, hi, L = tmap.lo, tmap.hi, tmap.length
    n_lo, p_lo_min, p_lo_cap = pml_lo_opt
    n_hi, p_hi_min, p_hi_cap = pml_hi_opt
    if n_lo * p_lo_min + n_hi * p_hi_min > L * (1.0 + 1e-12):
        return None

    k_c = len(centers)
    n_in = sum(span_counts) + (1 if k_c else 0)
    n_total = n_lo + n_in + n_hi
    if n_total < 1:
        return None
    if mirror:
        if n_lo != n_hi or span_counts != span_counts[::-1]:
            return None
        # A center at the axis midpoint requires an odd total cell count (middle cell, not middle edge).
        if k_c % 2 == 1 and n_total % 2 == 0:
            return None

    # 1. Initial edge proposal e_bar from the sizing envelope on each anchor span.
    p_lo_est = p_lo_min if n_lo else 0.0
    p_hi_est = p_hi_min if n_hi else 0.0
    if mirror:
        p_lo_est = p_hi_est = 0.5 * (p_lo_est + p_hi_est)
    if n_lo * p_lo_est + n_hi * p_hi_est >= 0.98 * L:
        scale = min(1.0, 0.95 * L / max(n_lo * p_lo_est + n_hi * p_hi_est, 1e-30))
        p_lo_est = max(p_lo_min, p_lo_est * scale) if n_lo else 0.0
        p_hi_est = max(p_hi_min, p_hi_est * scale) if n_hi else 0.0
        if n_lo * p_lo_est + n_hi * p_hi_est > L:
            return None

    x_in_lo = lo + n_lo * p_lo_est
    x_in_hi = hi - n_hi * p_hi_est
    if k_c and (centers[0] <= lo + n_lo * p_lo_min or centers[-1] >= hi - n_hi * p_hi_min):
        return None
    if x_in_lo > x_in_hi + 1e-12:
        return None
    if k_c:
        x_in_lo = min(x_in_lo, centers[0] - 0.25 * min(p_lo_cap if n_lo else spacing_eff, tmap.d_fine))
        x_in_hi = max(x_in_hi, centers[-1] + 0.25 * min(p_hi_cap if n_hi else spacing_eff, tmap.d_fine))
        x_in_lo = max(lo + n_lo * p_lo_min, x_in_lo)
        x_in_hi = min(hi - n_hi * p_hi_min, x_in_hi)

    e_bar = np.empty(n_total + 1, dtype=np.float64)
    if n_lo:
        e_bar[: n_lo + 1] = np.linspace(lo, x_in_lo, n_lo + 1)
    else:
        e_bar[0] = lo
    if n_hi:
        e_bar[n_total - n_hi :] = np.linspace(x_in_hi, hi, n_hi + 1)
    else:
        e_bar[n_total] = hi

    center_cells: list[int] = []
    if k_c == 0:
        if n_in > 0:
            f0, f1 = float(env.eval_F(x_in_lo)), float(env.eval_F(x_in_hi))
            if f1 > f0 + 1e-14:
                e_bar[n_lo : n_lo + n_in + 1] = env.inv_F(np.linspace(f0, f1, n_in + 1))
            else:
                e_bar[n_lo : n_lo + n_in + 1] = np.linspace(x_in_lo, x_in_hi, n_in + 1)
            e_bar[n_lo], e_bar[n_lo + n_in] = x_in_lo, x_in_hi
    else:
        f_lo = float(env.eval_F(x_in_lo))
        f_hi = float(env.eval_F(x_in_hi))
        f_c = [float(env.eval_F(c)) for c in centers]
        cur = n_lo
        # Span before first center: span_counts[0] full cells + half of center cell
        nb = span_counts[0]
        if nb > 0:
            u_grid = f_lo + (f_c[0] - f_lo) * (np.arange(nb + 1, dtype=np.float64) / (nb + 0.5))
            vals = env.inv_F(u_grid) if f_c[0] > f_lo + 1e-14 else x_in_lo + (centers[0] - x_in_lo) * (
                np.arange(nb + 1, dtype=np.float64) / (nb + 0.5)
            )
            vals[0] = x_in_lo
            e_bar[cur : cur + nb + 1] = vals
        cur += nb
        center_cells.append(cur)
        for m_idx in range(1, k_c):
            dc = span_counts[m_idx]
            if dc < 1:
                return None
            u_grid = f_c[m_idx - 1] + (f_c[m_idx] - f_c[m_idx - 1]) * (
                (np.arange(dc, dtype=np.float64) + 0.5) / dc
            )
            vals = (
                env.inv_F(u_grid)
                if f_c[m_idx] > f_c[m_idx - 1] + 1e-14
                else centers[m_idx - 1] + (centers[m_idx] - centers[m_idx - 1]) * (
                    (np.arange(dc, dtype=np.float64) + 0.5) / dc
                )
            )
            e_bar[cur + 1 : cur + dc + 1] = vals
            cur += dc
            center_cells.append(cur)
        na = span_counts[-1]
        u_grid = f_c[-1] + (f_hi - f_c[-1]) * ((np.arange(na + 1, dtype=np.float64) + 0.5) / (na + 0.5))
        vals = (
            env.inv_F(u_grid)
            if f_hi > f_c[-1] + 1e-14
            else centers[-1] + (x_in_hi - centers[-1]) * ((np.arange(na + 1, dtype=np.float64) + 0.5) / (na + 0.5))
        )
        vals[-1] = x_in_hi
        e_bar[cur + 1 : cur + na + 2] = vals

    for idx_start, idx_end, x_start, x_end in pinned_runs:
        if 0 <= idx_start < idx_end <= n_total:
            e_bar[idx_start : idx_end + 1] = np.linspace(x_start, x_end, idx_end - idx_start + 1)

    if mirror:
        e_bar = 0.5 * (e_bar + (lo + hi - e_bar[::-1]))
        e_bar[0], e_bar[-1] = lo, hi

    # Ensure strictly increasing proposal edges (repair any degenerate initial collapse).
    w_bar_cells = np.diff(e_bar)
    if np.any(w_bar_cells <= 0):
        e_bar = np.linspace(lo, hi, n_total + 1)
        w_bar_cells = np.diff(e_bar)

    # 2. Collapse flat interior plateaus (clear of transitions, centers, and breakpoints) into uniform blocks.
    single = np.zeros(n_total, dtype=bool)
    for ci in center_cells:
        single[max(0, ci - 1) : min(n_total, ci + 2)] = True
    if periodic and n_lo == 0 and n_hi == 0:
        seam_k = min(n_total, 5)
        single[:seam_k] = True
        single[n_total - seam_k :] = True
    pinned_mask = np.zeros(n_total, dtype=bool)
    for idx_start, idx_end, _, _ in pinned_runs:
        pinned_mask[idx_start:idx_end] = True
        if idx_start > 0:
            single[idx_start - 1] = True
        if idx_end < n_total:
            single[idx_end] = True

    if n_in > 0:
        inner_sl = slice(n_lo, n_total - n_hi)
        wi = w_bar_cells[inner_sl]
        jump = np.zeros(len(wi), dtype=bool)
        if len(wi) > 1:
            diff_rel = np.abs(np.diff(wi)) / np.maximum(np.minimum(wi[:-1], wi[1:]), 1e-15)
            jump[:-1] |= diff_rel > 1e-4
            jump[1:] |= diff_rel > 1e-4
        # Mark cells near any target breakpoint.
        ei_lo = e_bar[n_lo : n_total - n_hi] - 2.0 * wi
        ei_hi = e_bar[n_lo + 1 : n_total - n_hi + 1] + 2.0 * wi
        for bp in tmap.xp[1:-1]:
            jump |= (ei_lo <= bp) & (ei_hi >= bp)
        single[inner_sl] |= jump
        if n_lo > 0:
            w_lo_ref = min(p_lo_cap, float(w_bar_cells[0]))
            k_lo = min(
                n_in,
                max(2, math.ceil(math.log(max(spacing_eff / max(w_lo_ref, 1e-15), 1.0)) / math.log(max(g_eff, 1.001))) + 2),
            )
            single[n_lo : n_lo + k_lo] = True
        if n_hi > 0:
            w_hi_ref = min(p_hi_cap, float(w_bar_cells[-1]))
            k_hi = min(
                n_in,
                max(2, math.ceil(math.log(max(spacing_eff / max(w_hi_ref, 1e-15), 1.0)) / math.log(max(g_eff, 1.001))) + 2),
            )
            single[n_total - n_hi - k_hi : n_total - n_hi] = True

    # Expand single-cell cushions by 4 cells on each side inside the non-PML, non-pinned interior.
    base_single = single.copy()
    for shift in range(1, 5):
        single[shift:] |= base_single[:-shift]
        single[:-shift] |= base_single[shift:]
    if n_lo:
        single[:n_lo] = False
    if n_hi:
        single[n_total - n_hi :] = False
    for idx_start, idx_end, _, _ in pinned_runs:
        single[idx_start:idx_end] = False
    if mirror:
        single |= single[::-1]

    block_n: list[int] = []
    block_starts: list[int] = []
    pinned_by_start = {r[0]: r for r in pinned_runs}
    pos = 0
    if n_lo:
        block_starts.append(0)
        block_n.append(n_lo)
        pos = n_lo
    stop_in = n_total - n_hi
    while pos < stop_in:
        if pos in pinned_by_start:
            _, idx_end, _, _ = pinned_by_start[pos]
            block_starts.append(pos)
            block_n.append(idx_end - pos)
            pos = idx_end
            continue
        if single[pos]:
            block_starts.append(pos)
            block_n.append(1)
            pos += 1
            continue
        end_run = pos + 1
        while end_run < stop_in and not single[end_run] and end_run not in pinned_by_start:
            if abs(w_bar_cells[end_run] - w_bar_cells[pos]) > 1e-4 * w_bar_cells[pos]:
                break
            end_run += 1
        block_starts.append(pos)
        block_n.append(end_run - pos)
        pos = end_run
    if n_hi:
        block_starts.append(stop_in)
        block_n.append(n_hi)

    B = len(block_n)
    bn = np.asarray(block_n, dtype=np.float64)
    b_edges_idx = np.asarray([*block_starts, n_total], dtype=np.int64)
    E_bar = e_bar[b_edges_idx]
    y_bar = (E_bar - lo) / L
    y_bar[0], y_bar[-1] = 0.0, 1.0
    w_bar_norm = np.maximum(np.diff(y_bar) / bn, 1e-12)

    if mirror:
        if not np.array_equal(block_n, block_n[::-1]):
            return None
        H = B // 2
        V = H - 1 if B % 2 == 0 else H
    else:
        V = B - 1

    # Map free variable index r in [0, V) to block edge y_b = sign_b * x[var_b] + const_b.
    var_idx = np.full(B + 1, -1, dtype=np.int64)
    var_sign = np.zeros(B + 1, dtype=np.float64)
    var_const = np.zeros(B + 1, dtype=np.float64)
    var_const[B] = 1.0
    if mirror:
        for b in range(1, V + 1):
            var_idx[b] = b - 1
            var_sign[b] = 1.0
            var_idx[B - b] = b - 1
            var_sign[B - b] = -1.0
            var_const[B - b] = 1.0
        if B % 2 == 0:
            var_const[B // 2] = 0.5
    else:
        for b in range(1, B):
            var_idx[b] = b - 1
            var_sign[b] = 1.0

    cell_to_block = {s: b for b, s in enumerate(block_starts) if block_n[b] == 1}
    center_blocks: list[tuple[int, float]] = []
    for ci, c_val in zip(center_cells, centers, strict=True):
        if ci not in cell_to_block:
            return None
        center_blocks.append((cell_to_block[ci], (c_val - lo) / L))

    pinned_blocks: list[tuple[int, float, float]] = []
    start_to_block = {s: b for b, s in enumerate(block_starts)}
    for idx_start, idx_end, x_start, x_end in pinned_runs:
        b = start_to_block.get(idx_start)
        if b is None or block_starts[b] + block_n[b] != idx_end:
            return None
        pinned_blocks.append((b, (x_start - lo) / L, (x_end - lo) / L))

    # Branch state per (interval_index, block_index): 0 = in_halo (cap w_b <= d_m), -1 = left, +1 = right.
    raw_fine_norm = [
        ((a - lo) / L, (b_end - lo) / L, min(spacing_eff, max(d - cap_margin, 0.5 * d)) / L)
        for a, b_end, d in tmap.merged_fine
    ]
    fine_norm = list(raw_fine_norm)
    if periodic and n_lo == 0 and n_hi == 0:
        s_half = 0.5 * (spacing_eff / L)
        for a_n, b_n, d_n in raw_fine_norm:
            if a_n <= s_half:
                fine_norm.append((a_n + 1.0, b_n + 1.0, d_n))
            if b_n >= 1.0 - s_half:
                fine_norm.append((a_n - 1.0, b_n - 1.0, d_n))
    tau_norm = 1e-11
    branches: list[np.ndarray] = []
    for a_n, b_n, d_n in fine_norm:
        h_lo = y_bar[:-1] - 0.5 * w_bar_norm
        h_hi = y_bar[1:] + 0.5 * w_bar_norm
        br = np.zeros(B, dtype=np.int8)
        br[h_hi < a_n - 1e-12] = -1
        br[h_lo > b_n + 1e-12] = 1
        branches.append(br)
    if mirror:
        # Enforce symmetric branch caps across mirrored intervals by pairing m with its mirror interval.
        for m1, (a1, b1, d1) in enumerate(fine_norm):
            for m2, (a2, b2, d2) in enumerate(fine_norm):
                if abs(a1 - (1.0 - b2)) <= 1e-9 and abs(b1 - (1.0 - a2)) <= 1e-9 and abs(d1 - d2) <= 1e-9:
                    in_mask = (branches[m1] == 0) | (branches[m2][::-1] == 0)
                    branches[m1][in_mask] = 0
                    branches[m2][in_mask[::-1]] = 0

    spacing_norm = spacing_eff / L
    d_fine_norm = min(spacing_eff, max(tmap.d_fine - cap_margin, 0.5 * tmap.d_fine)) / L
    w_floor_norm = w_floor / L
    reach_blocks = max(3, math.ceil(math.log(max(g_eff, 1.01)) / math.log(3.0)) + 2)

    def build_system(with_m: bool, m_fixed: float = 0.0) -> tuple:
        """Assemble sparse (A_eq, b_eq, A_ub, b_ub) in variable vector z (x of length V, plus m if with_m)."""
        n_vars = V + (1 if with_m else 0)
        eq_r: list[int] = []
        eq_c: list[int] = []
        eq_v: list[float] = []
        eq_b: list[float] = []

        def add_y_comb(
            rows: list[int],
            cols: list[int],
            vals: list[float],
            rhs_list: list[float],
            terms: Sequence[tuple[int, float]],
            rhs: float,
            extra_m: float = 0.0,
        ) -> bool:
            r_id = len(rhs_list)
            acc: dict[int, float] = {}
            r_val = rhs
            for b_edge, coef in terms:
                r_val -= coef * var_const[b_edge]
                vi = int(var_idx[b_edge])
                if vi >= 0:
                    acc[vi] = acc.get(vi, 0.0) + coef * var_sign[b_edge]
            if with_m and abs(extra_m) > 0:
                acc[V] = acc.get(V, 0.0) + extra_m
            nz_items = [(k, val) for k, val in acc.items() if abs(val) > 1e-15]
            if not nz_items:
                return abs(r_val) <= 1e-10
            for k, val in nz_items:
                rows.append(r_id)
                cols.append(k)
                vals.append(val)
            rhs_list.append(r_val)
            return True

        # Equalities: centers (y_b + y_{b+1} = 2 c_n) and pinned aligned runs.
        b_limit_eq = (B + 1) // 2 if mirror else B
        for b_c, c_n in center_blocks:
            if mirror and b_c >= b_limit_eq:
                continue
            if mirror and B % 2 == 1 and b_c == B // 2:
                if abs(c_n - 0.5) > 1e-9:
                    return None
                continue
            if not add_y_comb(eq_r, eq_c, eq_v, eq_b, [(b_c, 1.0), (b_c + 1, 1.0)], 2.0 * c_n):
                return None

        for b_p, y_s, y_e in pinned_blocks:
            if not (mirror and b_p > B // 2):
                if not add_y_comb(eq_r, eq_c, eq_v, eq_b, [(b_p, 1.0)], y_s):
                    return None
            if not (mirror and b_p + 1 > (B + 1) // 2):
                if not add_y_comb(eq_r, eq_c, eq_v, eq_b, [(b_p + 1, 1.0)], y_e):
                    return None

        ub_r: list[int] = []
        ub_c: list[int] = []
        ub_v: list[float] = []
        ub_b: list[float] = []
        halo_meta: list[tuple[int, int, int, int]] = []  # (row_idx, m_idx, b_idx, side)

        def add_ub(terms: Sequence[tuple[int, float]], rhs: float, extra_m: float = 0.0) -> int | None:
            r_id = len(ub_b)
            acc: dict[int, float] = {}
            r_val = rhs
            for b_edge, coef in terms:
                r_val -= coef * var_const[b_edge]
                vi = int(var_idx[b_edge])
                if vi >= 0:
                    acc[vi] = acc.get(vi, 0.0) + coef * var_sign[b_edge]
            if with_m and abs(extra_m) > 0:
                acc[V] = acc.get(V, 0.0) + extra_m
            nz_items = [(k, val) for k, val in acc.items() if abs(val) > 1e-15]
            if not nz_items:
                return -1 if r_val >= -1e-11 else None
            for k, val in nz_items:
                ub_r.append(r_id)
                ub_c.append(k)
                ub_v.append(val)
            ub_b.append(r_val)
            return r_id

        # Per-block width floor and merged upper cap (from spacing + all "in" branches + PML caps).
        cap_b = np.full(B, spacing_norm, dtype=np.float64)
        if n_lo:
            cap_b[0] = min(cap_b[0], p_lo_cap / L)
        if n_hi:
            cap_b[B - 1] = min(cap_b[B - 1], p_hi_cap / L)
        for m_idx, (_, _, d_n) in enumerate(fine_norm):
            cap_b[branches[m_idx] == 0] = np.minimum(cap_b[branches[m_idx] == 0], d_n)

        b_max_w = (B // 2 + 1) if mirror else B
        for b in range(min(B, b_max_w)):
            inv_n = 1.0 / bn[b]
            # -w_b + m <= 0  (or -w_b <= -m_fixed)
            if with_m:
                if add_ub([(b, inv_n), (b + 1, -inv_n)], 0.0, extra_m=1.0) is None:
                    return None
            else:
                if add_ub([(b, inv_n), (b + 1, -inv_n)], -m_fixed) is None:
                    return None
            # w_b <= cap_b[b]
            if add_ub([(b, -inv_n), (b + 1, inv_n)], float(cap_b[b])) is None:
                return None

        # PML minimum thickness coverage: y_1 >= n_lo * p_lo_min / L, 1 - y_{B-1} >= n_hi * p_hi_min / L.
        if n_lo:
            if add_ub([(1, -1.0)], -(n_lo * p_lo_min / L)) is None:
                return None
        if n_hi and not mirror:
            if add_ub([(B - 1, 1.0)], 1.0 - (n_hi * p_hi_min / L)) is None:
                return None

        # Adjacent growth inequalities: w_{b+1} - g_eff * w_b <= 0 and w_b - g_eff * w_{b+1} <= 0.
        g_pairs = list(range(B // 2 if (mirror and B % 2 == 1) else (B // 2 - 1 if mirror else B - 1)))
        for b in g_pairs:
            in0, in1 = 1.0 / bn[b], 1.0 / bn[b + 1]
            # w_{b+1} - g_eff * w_b <= 0
            if add_ub([(b, g_eff * in0), (b + 1, -in1 - g_eff * in0), (b + 2, in1)], 0.0) is None:
                return None
            # w_b - g_eff * w_{b+1} <= 0
            if add_ub([(b, -in0), (b + 1, in0 + g_eff * in1), (b + 2, -g_eff * in1)], 0.0) is None:
                return None
        if periodic and B > 1 and not (mirror):
            in0, in1 = 1.0 / bn[B - 1], 1.0 / bn[0]
            if add_ub([(B - 1, g_eff * in0), (B, -g_eff * in0), (0, -in1), (1, in1)], 0.0) is None:
                return None
            if add_ub([(B - 1, -in0), (B, in0), (0, g_eff * in1), (1, -g_eff * in1)], 0.0) is None:
                return None

        # Positional halo bounds for nearest blocks assigned left (-1) or right (+1) of each fine interval.
        for m_idx, (a_n, b_n, _) in enumerate(fine_norm):
            br = branches[m_idx]
            left_idxs = np.nonzero(br == -1)[0]
            if len(left_idxs):
                for b in left_idxs[-reach_blocks:]:
                    if mirror and b >= (B + 1) // 2:
                        continue
                    c_hi = 1.0 + 0.5 / bn[b]
                    c_lo = -0.5 / bn[b]
                    rid = add_ub([(int(b), c_lo), (int(b) + 1, c_hi)], a_n - tau_norm)
                    if rid is None:
                        return None
                    if rid >= 0:
                        halo_meta.append((rid, m_idx, int(b), -1))
            right_idxs = np.nonzero(br == 1)[0]
            if len(right_idxs):
                for b in right_idxs[:reach_blocks]:
                    if mirror and b >= (B + 1) // 2:
                        continue
                    # -(y_b - 0.5 * w_b) <= -(b_n + tau_norm)  <=>  -(1 + 0.5/n)*y_b + (0.5/n)*y_{b+1} <= -(b_n + tau)
                    c_lo = -(1.0 + 0.5 / bn[b])
                    c_hi = 0.5 / bn[b]
                    rid = add_ub([(int(b), c_lo), (int(b) + 1, c_hi)], -(b_n + tau_norm))
                    if rid is None:
                        return None
                    if rid >= 0:
                        halo_meta.append((rid, m_idx, int(b), 1))

        A_eq_sp = (
            sparse.csc_matrix((eq_v, (eq_r, eq_c)), shape=(len(eq_b), n_vars))
            if eq_b
            else sparse.csc_matrix((0, n_vars))
        )
        A_ub_sp = (
            sparse.csc_matrix((ub_v, (ub_r, ub_c)), shape=(len(ub_b), n_vars))
            if ub_b
            else sparse.csc_matrix((0, n_vars))
        )
        return A_eq_sp, np.asarray(eq_b, dtype=np.float64), A_ub_sp, np.asarray(ub_b, dtype=np.float64), halo_meta

    # Stage 1: maximize m with up to 3 automatic boundary halo-branch relaxations.
    best_lp: tuple[float, np.ndarray] | None = None
    best_branches = [b.copy() for b in branches]
    for _attempt in range(4):
        built = build_system(with_m=True)
        if built is None:
            # Flip outermost boundary blocks to "in" in case positional halo walls collided with centers/PMLs.
            changed = False
            for m_idx in range(len(fine_norm)):
                br = branches[m_idx]
                left_i = np.nonzero(br == -1)[0]
                right_i = np.nonzero(br == 1)[0]
                if len(left_i):
                    br[left_i[-1]] = 0
                    changed = True
                if len(right_i):
                    br[right_i[0]] = 0
                    changed = True
            if not changed:
                break
            continue

        A_eq_sp, b_eq_v, A_ub_sp, b_ub_v, halo_meta = built
        c_obj = np.zeros(V + 1, dtype=np.float64)
        c_obj[V] = -1.0
        bounds = [*((0.0, 0.5 if mirror else 1.0) for _ in range(V)), (w_floor_norm, d_fine_norm)]
        res = optimize.linprog(
            c_obj, A_ub=A_ub_sp, b_ub=b_ub_v, A_eq=A_eq_sp, b_eq=b_eq_v, bounds=bounds, method="highs"
        )
        if res.success and res.x is not None and res.x[V] >= w_floor_norm - 1e-11:
            m_val = float(res.x[V])
            if best_lp is None or m_val > best_lp[0] + 1e-11:
                best_lp = (m_val, res.x[:V].copy())
                best_branches = [b.copy() for b in branches]
            if m_val >= d_fine_norm * (1.0 - 1e-6) or not halo_meta:
                break
            # Check if any positional halo wall is active; relax it into "in" to see if m_val rises.
            slacks = b_ub_v - A_ub_sp.dot(res.x)
            flipped = False
            for rid, m_idx, b_idx, _side in halo_meta:
                if slacks[rid] <= 1e-8 and branches[m_idx][b_idx] != 0:
                    branches[m_idx][b_idx] = 0
                    if mirror:
                        # Mirror counterpart
                        a1, b1, d1 = fine_norm[m_idx]
                        for m2, (a2, b2, d2) in enumerate(fine_norm):
                            if abs(a1 - (1.0 - b2)) <= 1e-9 and abs(b1 - (1.0 - a2)) <= 1e-9 and abs(d1 - d2) <= 1e-9:
                                branches[m2][B - 1 - b_idx] = 0
                    flipped = True
            if not flipped:
                break
        else:
            changed = False
            for m_idx in range(len(fine_norm)):
                br = branches[m_idx]
                left_i = np.nonzero(br == -1)[0]
                right_i = np.nonzero(br == 1)[0]
                if len(left_i):
                    br[left_i[-1]] = 0
                    changed = True
                if len(right_i):
                    br[right_i[0]] = 0
                    changed = True
            if not changed:
                break

    if best_lp is None:
        return None
    m_star, x_star = best_lp
    branches = best_branches

    # Stage 2: strictly convex regularization at w_b >= (1 - 1e-7) * m_star pulling blocks toward their target caps.
    if stage2 and V > 0:
        m_fix = max(w_floor_norm, m_star * (1.0 - 1e-7))
        built2 = build_system(with_m=False, m_fixed=m_fix)
        if built2 is not None:
            A_eq2, b_eq2, A_ub2, b_ub2, _ = built2
            cap_s2 = np.full(B, spacing_norm, dtype=np.float64)
            if n_lo:
                cap_s2[0] = min(cap_s2[0], p_lo_cap / L)
            if n_hi:
                cap_s2[B - 1] = min(cap_s2[B - 1], p_hi_cap / L)
            for m_idx, (_, _, d_n) in enumerate(fine_norm):
                cap_s2[branches[m_idx] == 0] = np.minimum(cap_s2[branches[m_idx] == 0], d_n)
            if V <= 250:
                # Build dense V x V quadratic objective in x:
                # 0.5 * sum_b n_b ((w_b - cap_s2_b) / cap_s2_b)^2 + 0.5 * beta * sum_b ((w_{b+1} - w_b) / w_avg_b)^2.
                Q = np.zeros((V, V), dtype=np.float64)
                q_vec = np.zeros(V, dtype=np.float64)
                beta = 0.25

                def accum_sq(terms: Sequence[tuple[int, float]], target_val: float, weight: float) -> None:
                    # Adds 0.5 * weight * (sum(coef * y_edge) - target_val)^2
                    c0 = -target_val
                    lin: dict[int, float] = {}
                    for b_edge, coef in terms:
                        c0 += coef * var_const[b_edge]
                        vi = int(var_idx[b_edge])
                        if vi >= 0:
                            lin[vi] = lin.get(vi, 0.0) + coef * var_sign[b_edge]
                    for i1, a1 in lin.items():
                        q_vec[i1] -= weight * c0 * a1
                        for i2, a2 in lin.items():
                            Q[i1, i2] += weight * a1 * a2

                for b in range(B):
                    wb0 = float(cap_s2[b])
                    inb = 1.0 / bn[b]
                    accum_sq([(b, -inb), (b + 1, inb)], wb0, float(bn[b]) / (wb0 * wb0))
                for b in range(B - 1):
                    w_avg = 0.5 * float(cap_s2[b] + cap_s2[b + 1])
                    in0, in1 = 1.0 / bn[b], 1.0 / bn[b + 1]
                    accum_sq(
                        [(b, in0), (b + 1, -in1 - in0), (b + 2, in1)],
                        0.0,
                        beta / (w_avg * w_avg),
                    )
                if periodic and B > 1 and not mirror:
                    w_avg = 0.5 * float(cap_s2[B - 1] + cap_s2[0])
                    in0, in1 = 1.0 / bn[B - 1], 1.0 / bn[0]
                    accum_sq([(B - 1, in0), (B, -in0), (0, -in1), (1, in1)], 0.0, beta / (w_avg * w_avg))
                Q += 1e-12 * np.eye(V)
                x_star = _active_set_qp(Q, q_vec, A_eq2.toarray(), b_eq2, A_ub2.toarray(), b_ub2, x_star)
            else:
                # Sparse L1 smoothing fallback for very high uncollapsed block counts (V > 250).
                # Variables: [x (V), t_dev (B), t_jump (B - 1)]
                n_aux = B + (B - 1)
                tot_v = V + n_aux
                r_l, c_l, v_l, rhs_l = [], [], [], []
                A_ub_coo = A_ub2.tocoo()
                r_l.extend(A_ub_coo.row.tolist())
                c_l.extend(A_ub_coo.col.tolist())
                v_l.extend(A_ub_coo.data.tolist())
                rhs_l.extend(b_ub2.tolist())

                def add_abs_bound(terms: Sequence[tuple[int, float]], target_val: float, aux_col: int) -> None:
                    for sgn in (1.0, -1.0):
                        rid = len(rhs_l)
                        r_val = sgn * target_val
                        acc: dict[int, float] = {}
                        for b_edge, coef in terms:
                            r_val -= sgn * coef * var_const[b_edge]
                            vi = int(var_idx[b_edge])
                            if vi >= 0:
                                acc[vi] = acc.get(vi, 0.0) + sgn * coef * var_sign[b_edge]
                        for k, val in acc.items():
                            if abs(val) > 1e-15:
                                r_l.append(rid)
                                c_l.append(k)
                                v_l.append(val)
                        r_l.append(rid)
                        c_l.append(aux_col)
                        v_l.append(-1.0)
                        rhs_l.append(r_val)

                c2 = np.zeros(tot_v, dtype=np.float64)
                for b in range(B):
                    wb0 = float(cap_s2[b])
                    inb = 1.0 / bn[b]
                    add_abs_bound([(b, -inb), (b + 1, inb)], wb0, V + b)
                    c2[V + b] = float(bn[b]) / wb0
                for b in range(B - 1):
                    w_avg = 0.5 * float(cap_s2[b] + cap_s2[b + 1])
                    in0, in1 = 1.0 / bn[b], 1.0 / bn[b + 1]
                    add_abs_bound([(b, in0), (b + 1, -in1 - in0), (b + 2, in1)], 0.0, V + B + b)
                    c2[V + B + b] = 0.25 / w_avg
                A_ub_s2 = sparse.csc_matrix((v_l, (r_l, c_l)), shape=(len(rhs_l), tot_v))
                A_eq_s2 = sparse.hstack([A_eq2, sparse.csc_matrix((len(b_eq2), n_aux))], format="csc")
                bounds2 = [*((0.0, 0.5 if mirror else 1.0) for _ in range(V)), *((0.0, None) for _ in range(n_aux))]
                res2 = optimize.linprog(
                    c2, A_ub=A_ub_s2, b_ub=np.asarray(rhs_l), A_eq=A_eq_s2, b_eq=b_eq2, bounds=bounds2, method="highs"
                )
                if res2.success and res2.x is not None:
                    x_star = res2.x[:V].copy()

    # Reconstruct block edges Y and expand to cell edges.
    y_sol = var_const.copy()
    valid = var_idx >= 0
    y_sol[valid] += var_sign[valid] * x_star[var_idx[valid]]
    E_sol = lo + L * y_sol
    E_sol[0], E_sol[-1] = lo, hi

    parts = [
        np.linspace(E_sol[b], E_sol[b + 1], block_n[b] + 1)[:-1] if block_n[b] > 1 else E_sol[b : b + 1]
        for b in range(B)
    ]
    edges = np.concatenate([*parts, np.array([hi], dtype=np.float64)])
    edges[0], edges[-1] = lo, hi
    if mirror:
        half_n = n_total // 2
        edges[n_total - half_n :] = (lo + hi) - edges[: half_n + 1][::-1]
        if n_total % 2 == 0:
            edges[half_n] = tmap.mid
        edges[0], edges[-1] = lo, hi

    w_full = np.diff(edges)
    if np.any(w_full <= 0):
        return None
    return edges, float(w_full.min()), float(m_star * L)


def _certify_edges(
    edges: np.ndarray,
    tmap: _TargetMap,
    pml_cells: tuple[int, int],
    pml: tuple[float, float],
    growth: float,
    centers: tuple[float, ...],
    periodic: bool,
    mirror: bool,
    dtype: Any,
) -> bool:
    """Independently check all target, halo, dual-cell, growth, PML, center, symmetry, and post-cast invariants."""
    if len(edges) < 2 or not np.all(np.isfinite(edges)):
        return False
    if abs(float(edges[0]) - tmap.lo) > 1e-12 or abs(float(edges[-1]) - tmap.hi) > 1e-12:
        return False
    w = np.diff(edges)
    if np.any(w <= 0):
        return False
    rtol = 1e-6
    if len(w) > 1:
        ratios = np.maximum(w[1:] / w[:-1], w[:-1] / w[1:])
        if float(np.max(ratios)) > growth * (1.0 + rtol):
            return False
    if periodic and len(w) > 1:
        if max(float(w[0] / w[-1]), float(w[-1] / w[0])) > growth * (1.0 + rtol):
            return False

    h_lo = np.maximum(tmap.lo, edges[:-1] - 0.5 * w)
    h_hi = np.minimum(tmap.hi, edges[1:] + 0.5 * w)
    caps = tmap.min_on_vec(h_lo, h_hi)
    if np.any(w > caps * (1.0 + rtol)):
        return False
    if periodic:
        if float(w[0]) > tmap.min_on(max(tmap.lo, tmap.hi - 0.5 * float(w[0])), tmap.hi) * (1.0 + rtol):
            return False
        if float(w[-1]) > tmap.min_on(tmap.lo, min(tmap.hi, tmap.lo + 0.5 * float(w[-1]))) * (1.0 + rtol):
            return False
    if len(w) > 1:
        c = 0.5 * (edges[:-1] + edges[1:])
        dcaps = tmap.min_on_vec(c[:-1], c[1:])
        if np.any(0.5 * (w[:-1] + w[1:]) > dcaps * (1.0 + rtol)):
            return False

    n_lo, n_hi = pml_cells
    if n_lo < 0 or n_hi < 0 or n_lo + n_hi > len(w):
        return False
    if pml[0] > 0:
        if n_lo < 1 or not np.allclose(w[:n_lo], w[0], rtol=1e-7, atol=1e-12):
            return False
        if float(edges[n_lo] - tmap.lo) < pml[0] * (1.0 - 1e-7):
            return False
    if pml[1] > 0:
        if n_hi < 1 or not np.allclose(w[-n_hi:], w[-1], rtol=1e-7, atol=1e-12):
            return False
        if float(tmap.hi - edges[-n_hi - 1]) < pml[1] * (1.0 - 1e-7):
            return False

    if mirror and not np.allclose(edges - tmap.lo, (tmap.hi - edges)[::-1], atol=1e-9 * max(1.0, tmap.length)):
        return False

    if centers:
        mids = 0.5 * (edges[:-1] + edges[1:])
        inner_mids = mids[n_lo : len(w) - n_hi]
        if len(inner_mids) == 0:
            return False
        tol_c = 1e-8 * max(1.0, tmap.length)
        for c_val in centers:
            if float(np.min(np.abs(inner_mids - c_val))) > tol_c:
                return False

    # Check survival in target simulation dtype (centered metres, as StackDevice.rectilinear_grid constructs it).
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.floating) and dt != np.float64:
        raw = ((edges - tmap.mid) * 1e-6).astype(dt)
        wc = np.diff(raw.astype(np.float64))
        if np.any(wc <= 0):
            return False
        if len(wc) > 1:
            rc = float(np.max(np.maximum(wc[1:] / wc[:-1], wc[:-1] / wc[1:])))
            if rc > growth * (1.0 + 8e-4):
                return False
    return True


def axis_edges(
    lo: float,
    hi: float,
    spacing: float,
    fine: Sequence[tuple[float, float, float]] = (),
    *,
    growth: float = 1.25,
    pml: tuple[float, float] = (0.0, 0.0),
    align: bool = True,
    symmetric: bool = True,
    centers: Sequence[float] = (),
    periodic: bool = False,
    dtype: Any = np.float64,
) -> AxisEdges:
    """Continuous cell edges from ``lo`` to ``hi``, every cell and dual cell within its target.

    Places real-valued cell edges without quantizing cell widths or box lengths onto a discrete lattice.
    Every cell is no wider than the smallest target within half its own width on either side of it (certifying
    both primal cells and Yee dual cells), neighbouring cells (and wrap-around end cells when ``periodic``)
    differ by at most ``growth``, and each PML end receives uniform cells covering its requested thickness.

    Maximizes the smallest cell width (which sets this axis's share of the CFL time step), keeps cell counts
    minimal, and regularizes cell widths smoothly across plateaus and ramps. Each coordinate in ``centers``
    (e.g. a static planar material interface) is placed at the exact midpoint of an interior cell where it
    costs at most 20 % of the uncentered smallest cell; conflicting or PML-covered centers are dropped with a
    warning.

    Args:
        lo: first edge (um).
        hi: last edge (um); any ``hi > lo`` is accepted.
        spacing: largest cell (um).
        fine: ``(a, b, d)`` regions ``[a, b]`` (um, ``a <= b``) with cells of at most ``d``.
        growth: largest ratio of neighbouring cells (> 1).
        pml: PML thickness (um) at the lower and upper ends (0: none).
        align: prefer placing cells of the finest target's width on the ``lo + k * d_fine`` lattice when
            compatible with the largest smallest cell, mirror symmetry, and centered interfaces.
        symmetric: give mirror-symmetric targets, PMLs, and centers mirror-symmetric edges.
        centers: coordinates (um) to place at the exact middle of an interior cell.
        periodic: also bound the ratio of the first and last cell by ``growth`` (when ``pml == (0, 0)``).
        dtype: floating-point precision the realized grid will be cast to (e.g. ``jnp.float32`` or
            ``np.float64``), so growth and PML margins survive conversion to centered metres.

    Returns:
        `AxisEdges` `(edges, pml_cells, min_cell, aligned, symmetric, centered)`.
    """
    lo, hi, spacing, growth = float(lo), float(hi), float(spacing), float(growth)
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        raise ValueError(f"axis_edges needs finite lo < hi, got {lo} and {hi}")
    if not (math.isfinite(spacing) and spacing > 0):
        raise ValueError(f"spacing must be positive, got {spacing}")
    if not (math.isfinite(growth) and growth > 1):
        raise ValueError(f"growth must be > 1, got {growth}")
    if len(pml) != 2:
        raise ValueError(f"pml must be two nonnegative thicknesses, got {pml}")
    pml_lo, pml_hi = float(pml[0]), float(pml[1])
    if not (math.isfinite(pml_lo) and math.isfinite(pml_hi) and min(pml_lo, pml_hi) >= 0):
        raise ValueError(f"pml must be two nonnegative thicknesses, got {pml}")
    pml_pair = (pml_lo, pml_hi)
    L = hi - lo
    if pml_lo + pml_hi > L * (1.0 + 1e-12):
        raise ValueError(f"axis_edges: PMLs of {pml_lo:g} and {pml_hi:g} um exceed the {L:g} um axis")

    tmap = _TargetMap(lo, hi, spacing, fine)
    dt = np.dtype(dtype)
    ulp_um = (
        float(np.finfo(dt).eps) * 0.5 * L
        if np.issubdtype(dt, np.floating) and dt != np.float64
        else 0.0
    )
    if ulp_um > 0 and 4.0 * ulp_um >= tmap.d_fine:
        raise ValueError(
            f"axis_edges: finest target {tmap.d_fine:g} um does not survive {dt} precision over a {L:g} um axis; "
            "use a float64 device"
        )

    sym_tol = 1e-9 * max(1.0, L, abs(lo), abs(hi))
    center_vals: list[float] = []
    for c in centers:
        cf = float(c)
        if not math.isfinite(cf):
            raise ValueError(f"centers must be finite, got {c}")
        center_vals.append(cf)
    raw_centers = sorted(center_vals)
    dedup_centers: list[float] = []
    for c_val in raw_centers:
        if lo + sym_tol < c_val < hi - sym_tol:
            if not dedup_centers or c_val - dedup_centers[-1] > sym_tol:
                dedup_centers.append(c_val)

    can_mirror_base = bool(symmetric and abs(pml_lo - pml_hi) <= sym_tol and tmap.is_symmetric(sym_tol))

    def is_aligned(edges_arr: np.ndarray, pml_c: tuple[int, int]) -> bool:
        w = np.diff(edges_arr)
        inner = slice(pml_c[0], len(w) - pml_c[1])
        w_in = w[inner]
        if len(w_in) == 0:
            return True
        d_f = tmap.d_fine
        if tmap.is_constant:
            if not np.allclose(w_in, d_f, rtol=1e-5, atol=1e-9):
                return False
            s_in = edges_arr[:-1][inner] - lo
            return bool(np.all(np.abs(s_in / d_f - np.round(s_in / d_f)) <= 1e-4))
        fine_cells = np.abs(w_in - d_f) <= 2e-4 * d_f
        if not np.any(fine_cells):
            return True
        s_in = (edges_arr[:-1][inner][fine_cells] - lo) / d_f
        return bool(np.all(np.abs(s_in - np.round(s_in)) <= 1e-4))

    # Retry loop only triggers if tight float32 rounding asks for a slightly larger ULP margin.
    for margin_scale in (1.0, 3.0):
        cap_margin = margin_scale * (2e-7 * tmap.d_fine + 1e-12 * L)
        pml_pad = margin_scale * (2e-8 * max(pml_lo, pml_hi) + 1e-12 * L)
        ratio_guard = 1.0 - margin_scale * (1e-5 + (6.0 * ulp_um / max(tmap.d_fine, 1e-12)))
        if ratio_guard <= 0.0 or 1.0 + (growth - 1.0) * ratio_guard <= 1.0:
            continue
        g_eff = 1.0 + (growth - 1.0) * ratio_guard
        spacing_eff = max(spacing - cap_margin, 0.5 * spacing)
        d_fine_eff = min(spacing_eff, max(tmap.d_fine - cap_margin, 0.5 * tmap.d_fine))

        # O(1) exact uniform fast path when target is constant and centers are absent (or already mid-cell).
        if tmap.is_constant:
            n_u = max(1, math.ceil(L / tmap.d_fine - 1e-12))
            for n_try in range(n_u, n_u + 8):
                h_u = L / n_try
                n0_u = math.ceil(pml_lo / h_u - 1e-9) if pml_lo > 0 else 0
                n1_u = math.ceil(pml_hi / h_u - 1e-9) if pml_hi > 0 else 0
                if n0_u + n1_u <= n_try:
                    e_u = tmap.mid + np.linspace(-0.5 * L, 0.5 * L, n_try + 1)
                    e_u[0], e_u[-1] = lo, hi
                    mids_in = 0.5 * (e_u[:-1] + e_u[1:])[n0_u : n_try - n1_u]
                    centers_ok = not dedup_centers or (
                        len(mids_in) > 0
                        and all(float(np.min(np.abs(mids_in - c))) <= sym_tol for c in dedup_centers)
                    )
                    if centers_ok and _certify_edges(
                        e_u,
                        tmap,
                        (n0_u, n1_u),
                        pml_pair,
                        growth,
                        tuple(dedup_centers),
                        periodic and pml_lo == 0 and pml_hi == 0,
                        can_mirror_base,
                        dt,
                    ):
                        al_u = is_aligned(e_u, (n0_u, n1_u))
                        if align and not al_u:
                            warnings.warn(
                                f"axis_edges: the {tmap.d_fine:g} um cells do not all start at whole multiples of "
                                f"their width from {lo:g} um (the largest smallest cell and mirror symmetry come first)",
                                stacklevel=2,
                            )
                        return AxisEdges(
                            e_u, (n0_u, n1_u), h_u, al_u, can_mirror_base, tuple(dedup_centers)
                        )
                    break

        periodic_eff = bool(periodic and pml_lo == 0 and pml_hi == 0)
        env = _SizingEnvelope(tmap, spacing_eff, g_eff, cap_margin, periodic=periodic_eff)
        lo_opts = _pml_candidates(tmap, env, 0, pml_lo, spacing_eff, cap_margin, pml_pad, g_eff)
        hi_opts = _pml_candidates(tmap, env, 1, pml_hi, spacing_eff, cap_margin, pml_pad, g_eff)
        if not lo_opts or not hi_opts:
            raise ValueError(
                f"axis_edges: PMLs of {pml_lo:g} and {pml_hi:g} um do not fit the {L:g} um axis within their targets"
            )
        pml_caps: list[tuple[float, float, float]] = []
        if pml_lo > 0 and lo_opts[0][2] < spacing_eff * (1.0 - 1e-12):
            pml_caps.append((lo, min(hi, lo + max(pml_lo, lo_opts[0][0] * lo_opts[0][1])), lo_opts[0][2]))
        if pml_hi > 0 and hi_opts[0][2] < spacing_eff * (1.0 - 1e-12):
            pml_caps.append((max(lo, hi - max(pml_hi, hi_opts[0][0] * hi_opts[0][1])), hi, hi_opts[0][2]))
        if pml_caps:
            env = _SizingEnvelope(tmap, spacing_eff, g_eff, cap_margin, periodic=periodic_eff, extra_caps=pml_caps)

        pml_pairs: list[tuple[tuple[int, float, float], tuple[int, float, float]]] = []
        if can_mirror_base:
            lo_by_n = {o[0]: o for o in lo_opts}
            hi_by_n = {o[0]: o for o in hi_opts}
            for n_shared in sorted(set(lo_by_n) & set(hi_by_n)):
                o0, o1 = lo_by_n[n_shared], hi_by_n[n_shared]
                pm = max(o0[1], o1[1])
                pc = min(o0[2], o1[2])
                if pm <= pc + 1e-14 and 2 * n_shared * pm <= L + 1e-12:
                    shared = (n_shared, pm, pc)
                    pml_pairs.append((shared, shared))
        asym_set: set[tuple[tuple[int, float, float], tuple[int, float, float]]] = set()
        for o0 in lo_opts[:5]:
            for o1 in hi_opts[:5]:
                if o0[0] * o0[1] + o1[0] * o1[1] <= L + 1e-12:
                    asym_set.add((o0, o1))
        for o0 in lo_opts:
            for o1 in (hi_opts[0], hi_opts[min(2, len(hi_opts) - 1)], hi_opts[-1]):
                if o0[0] * o0[1] + o1[0] * o1[1] <= L + 1e-12:
                    asym_set.add((o0, o1))
        for o1 in hi_opts:
            for o0 in (lo_opts[0], lo_opts[min(2, len(lo_opts) - 1)], lo_opts[-1]):
                if o0[0] * o0[1] + o1[0] * o1[1] <= L + 1e-12:
                    asym_set.add((o0, o1))
        for pair in sorted(asym_set, key=lambda p: (p[0][0] + p[1][0], abs(p[0][0] - p[1][0]), p[0][0])):
            if pair not in pml_pairs:
                pml_pairs.append(pair)
        if not pml_pairs:
            raise ValueError(f"axis_edges: the PMLs at both ends overlap on the {L:g} um axis")

        def estimate_spans(
            o0: tuple[int, float, float],
            o1: tuple[int, float, float],
            cs: tuple[float, ...],
            mir: bool,
        ) -> list[tuple[int, ...]]:
            x0 = lo + o0[0] * min(o0[2], max(o0[1], float(env.eval_s(lo + o0[0] * o0[1])))) if o0[0] else lo
            x1 = hi - o1[0] * min(o1[2], max(o1[1], float(env.eval_s(hi - o1[0] * o1[1])))) if o1[0] else hi
            if x0 > x1:
                x0, x1 = lo + o0[0] * o0[1], hi - o1[0] * o1[1]
            if not cs:
                u_in = max(0.0, float(env.eval_F(x1) - env.eval_F(x0)))
                base_n = max(1 if (o0[0] + o1[0] == 0) else 0, round(u_in))
                cand_set = {max(0, base_n + d) for d in (-1, 0, 1, 2, 3, 4) if base_n + d >= 0}
                extra_ramp = 0
                if o0[0]:
                    w0 = max(o0[1], min(o0[2], (x0 - lo) / o0[0]))
                    s0 = float(env.eval_s(x0))
                    if w0 < s0 * (1.0 - 1e-6):
                        extra_ramp += math.ceil(math.log(s0 / max(w0, 1e-15)) / math.log(max(g_eff, 1.001)))
                if o1[0]:
                    w1 = max(o1[1], min(o1[2], (hi - x1) / o1[0]))
                    s1 = float(env.eval_s(x1))
                    if w1 < s1 * (1.0 - 1e-6):
                        extra_ramp += math.ceil(math.log(s1 / max(w1, 1e-15)) / math.log(max(g_eff, 1.001)))
                if extra_ramp > 0:
                    cand_set.update(max(0, base_n + extra_ramp + d) for d in (-1, 0, 1, 2, 3) if base_n + extra_ramp + d >= 0)
                cands = sorted(cand_set)
                return [(n_c,) for n_c in cands if not (o0[0] + o1[0] + n_c < 1)]
            if cs[0] <= lo + o0[0] * o0[1] or cs[-1] >= hi - o1[0] * o1[1]:
                return []
            x0 = min(x0, cs[0])
            x1 = max(x1, cs[-1])
            f0 = float(env.eval_F(x0))
            f1 = float(env.eval_F(x1))
            fc = [float(env.eval_F(c)) for c in cs]
            nb0 = max(0, math.ceil((fc[0] - f0) - 0.5 - 1e-6))
            na0 = max(0, math.ceil((f1 - fc[-1]) - 0.5 - 1e-6))
            if mir:
                nb0 = na0 = max(nb0, na0)
            deltas = [max(1, math.ceil((fc[i + 1] - fc[i]) - 1e-6)) for i in range(len(cs) - 1)]
            if mir:
                deltas = [max(deltas[i], deltas[-1 - i]) for i in range(len(deltas))]
            base = (nb0, *deltas, na0)
            variants = [base]
            # Add +1/-1 outer and tight-span perturbations (symmetric if mir).
            for d_end in (-1, 1, 2):
                if nb0 + d_end >= 0 and na0 + d_end >= 0:
                    if mir:
                        v_end = (nb0 + d_end, *deltas, na0 + d_end)
                        if v_end not in variants:
                            variants.append(v_end)
                    else:
                        for v_end in ((nb0 + d_end, *deltas, na0), (nb0, *deltas, na0 + d_end),
                                      (nb0 + d_end, *deltas, na0 + d_end)):
                            if v_end not in variants:
                                variants.append(v_end)
            for idx_d in range((len(deltas) + 1) // 2 if mir else len(deltas)):
                for step_d in (1, -1):
                    d_copy = list(deltas)
                    if d_copy[idx_d] + step_d < 1:
                        continue
                    d_copy[idx_d] += step_d
                    if mir and len(deltas) - 1 - idx_d != idx_d:
                        d_copy[len(deltas) - 1 - idx_d] += step_d
                    v_mid = (nb0, *d_copy, na0)
                    if v_mid not in variants:
                        variants.append(v_mid)
            if len(deltas) >= 2:
                v_all_up = (nb0, *(d + 1 for d in deltas), na0)
                if v_all_up not in variants:
                    variants.append(v_all_up)
            return variants

        def solve_for_centers(
            cs: tuple[float, ...],
            floor_w: float,
            stage2_on: bool,
        ) -> tuple[np.ndarray, tuple[int, int], float, bool] | None:
            cs_arr = np.asarray(cs, dtype=np.float64)
            mir_cs = bool(
                can_mirror_base
                and (len(cs) == 0 or np.allclose(cs_arr - lo, (hi - cs_arr)[::-1], atol=sym_tol, rtol=0.0))
            )
            m_ceil_cs = d_fine_eff
            for i_c in range(len(cs) - 1):
                span_w = cs[i_c + 1] - cs[i_c]
                ia = max(0, int(np.searchsorted(tmap.xp, cs[i_c], side="right")) - 1)
                ib = min(len(tmap.seg_d) - 1, int(np.searchsorted(tmap.xp, cs[i_c + 1], side="left")) - 1)
                if ib >= ia:
                    t_max_raw = float(np.max(tmap.seg_d[ia : ib + 1]))
                    t_max_eff = min(spacing_eff, max(t_max_raw - cap_margin, 0.5 * t_max_raw))
                    n_min_span = max(1, math.ceil(span_w / max(t_max_eff, 1e-15) - 1e-12))
                    m_ceil_cs = min(m_ceil_cs, span_w / n_min_span)

            tie_rel = 1e-2 if cs else 2e-3
            best_sol: tuple[float, float, int, bool, np.ndarray, tuple[int, int]] | None = None
            modes = [True, False] if mir_cs else [False]
            for mir_mode in modes:
                for o0, o1 in pml_pairs:
                    if mir_mode and o0 != o1:
                        continue
                    if (
                        not mir_mode
                        and best_sol is not None
                        and best_sol[3] == mir_mode
                        and best_sol[0] >= m_ceil_cs * (1.0 - tie_rel)
                        and o0[0] + o1[0] > best_sol[5][0] + best_sol[5][1]
                    ):
                        break
                    for sp in estimate_spans(o0, o1, cs, mir_mode):
                        got = _solve_blocks(
                            tmap,
                            env,
                            o0,
                            o1,
                            cs,
                            sp,
                            mirror=mir_mode,
                            periodic=periodic_eff,
                            spacing_eff=spacing_eff,
                            g_eff=g_eff,
                            cap_margin=cap_margin,
                            w_floor=floor_w,
                            stage2=stage2_on,
                        )
                        if got is None:
                            continue
                        e_sol, w_min_sol, m_lp = got
                        if stage2_on and not _certify_edges(
                            e_sol, tmap, (o0[0], o1[0]), pml_pair, growth, cs, periodic_eff, mir_mode, dt
                        ):
                            continue
                        n_cells_sol = len(e_sol) - 1
                        if best_sol is None:
                            best_sol = (m_lp, w_min_sol, n_cells_sol, mir_mode, e_sol, (o0[0], o1[0]))
                        else:
                            bm, _, bn_c, bmir, _, _ = best_sol
                            better_w = m_lp > bm * (1.0 + tie_rel)
                            same_w = abs(m_lp - bm) <= tie_rel * max(bm, m_lp)
                            if better_w or (
                                same_w
                                and (
                                    mir_mode > bmir
                                    or (mir_mode == bmir and n_cells_sol < bn_c)
                                    or (mir_mode == bmir and n_cells_sol == bn_c and m_lp > bm * (1.0 + 1e-7))
                                )
                            ):
                                best_sol = (m_lp, w_min_sol, n_cells_sol, mir_mode, e_sol, (o0[0], o1[0]))
                    # Early exit once symmetric mode already hits m_ceil_cs with fewest PML cells.
                    if mir_mode and best_sol is not None and best_sol[3] == mir_mode and best_sol[0] >= m_ceil_cs * (1.0 - tie_rel):
                        break
                if best_sol is not None and mir_mode and best_sol[0] >= max(floor_w, m_ceil_cs * (1.0 - tie_rel)):
                    break
            if best_sol is None:
                return None
            return best_sol[4], best_sol[5], best_sol[1], best_sol[3]

        # 1. Baseline uncentered optimum w_min_base.
        base_res = solve_for_centers((), 0.0, stage2_on=not bool(dedup_centers))
        if base_res is None:
            continue
        e_chosen, pml_chosen, w_min_base, mir_chosen = base_res
        kept_centers: tuple[float, ...] = ()

        # 2. Retain maximal subset of centers keeping min_cell >= _CENTER_FLOOR * w_min_base.
        if dedup_centers:
            c_floor = _CENTER_FLOOR * w_min_base
            # Filter out centers strictly inside the narrowest possible PML coverage right away.
            min_lo_cov = min(o[0] * o[1] for o in lo_opts)
            min_hi_cov = min(o[0] * o[1] for o in hi_opts)
            cand_cs = [c for c in dedup_centers if lo + min_lo_cov < c < hi - min_hi_cov]
            tried_all = solve_for_centers(tuple(cand_cs), c_floor, stage2_on=True) if cand_cs else None
            if tried_all is not None:
                e_chosen, pml_chosen, _, mir_chosen = tried_all
                kept_centers = tuple(cand_cs)
            elif cand_cs:
                # Group symmetric pairs when problem and requested centers are symmetric so mirror is preserved.
                cs_np = np.asarray(cand_cs, dtype=np.float64)
                sym_req = can_mirror_base and np.allclose(cs_np - lo, (hi - cs_np)[::-1], atol=sym_tol, rtol=0.0)
                groups: list[list[float]] = []
                if sym_req:
                    i_l, i_r = 0, len(cand_cs) - 1
                    while i_l <= i_r:
                        groups.append([cand_cs[i_l]] if i_l == i_r else [cand_cs[i_l], cand_cs[i_r]])
                        i_l += 1
                        i_r -= 1
                else:
                    groups = [[c] for c in cand_cs]
                acc_cs: list[float] = []
                for grp in groups:
                    trial = tuple(sorted([*acc_cs, *grp]))
                    if solve_for_centers(trial, c_floor, stage2_on=False) is not None:
                        acc_cs = list(trial)
                if acc_cs:
                    final_c = solve_for_centers(tuple(acc_cs), c_floor, stage2_on=True)
                    if final_c is not None:
                        e_chosen, pml_chosen, _, mir_chosen = final_c
                        kept_centers = tuple(acc_cs)
            if not kept_centers:
                # Re-run baseline with Stage 2 smoothing enabled.
                base_smooth = solve_for_centers((), 0.0, stage2_on=True)
                if base_smooth is not None:
                    e_chosen, pml_chosen, _, mir_chosen = base_smooth

        # 3. Optional alignment of finest-target runs onto the lo + k * d_fine lattice when compatible.
        if align and not tmap.is_constant and not is_aligned(e_chosen, pml_chosen):
            d_f = tmap.d_fine
            # Mirror + align requires L / d_fine to be an integer.
            mirror_align_ok = (not mir_chosen) or abs(L / d_f - round(L / d_f)) <= 1e-5
            centers_align_ok = all(
                abs(((c - lo) / d_f - 0.5) - round((c - lo) / d_f - 0.5)) <= 1e-5
                or tmap.min_on(c - d_f, c + d_f) > d_f * (1.0 + 1e-4)
                for c in kept_centers
            )
            if mirror_align_ok and centers_align_ok:
                cur_w_min = float(np.min(np.diff(e_chosen)))
                for o0, o1 in pml_pairs:
                    if mir_chosen and o0 != o1:
                        continue
                    lo_in_min = lo + o0[0] * o0[1]
                    hi_in_max = hi - o1[0] * o1[1]
                    runs_k: list[tuple[int, int]] = []
                    for a_f, b_f, d_val in tmap.merged_fine:
                        if abs(d_val - d_f) > 1e-9 * d_f:
                            continue
                        af_c = max(a_f, lo_in_min)
                        bf_c = min(b_f, hi_in_max)
                        if bf_c < af_c:
                            continue
                        k0 = max(0, math.ceil((af_c - lo) / d_f - 1.5 - 1e-9))
                        k1 = math.floor((bf_c - lo) / d_f + 1.5 + 1e-9)
                        while k0 < k1 and lo + k0 * d_f < lo_in_min - 1e-12:
                            k0 += 1
                        while k1 > k0 and lo + k1 * d_f > hi_in_max + 1e-12:
                            k1 -= 1
                        if k1 > k0:
                            runs_k.append((k0, k1))
                    if not runs_k:
                        continue
                    # Merge overlapping or adjacent lattice runs.
                    runs_k.sort()
                    merged_k = [runs_k[0]]
                    for k0, k1 in runs_k[1:]:
                        if k0 <= merged_k[-1][1]:
                            merged_k[-1] = (merged_k[-1][0], max(merged_k[-1][1], k1))
                        else:
                            merged_k.append((k0, k1))
                    if kept_centers:
                        # Skip complex multi-anchor index interleaving when centers are already active;
                        # centers take priority over alignment.
                        break
                    # Compute interior span counts before, between, and after the aligned lattice runs.
                    anchors = [lo + o0[0] * o0[1]]
                    for k0, k1 in merged_k:
                        anchors.extend([lo + k0 * d_f, lo + k1 * d_f])
                    anchors.append(hi - o1[0] * o1[1])
                    if any(anchors[j + 1] < anchors[j] - 1e-12 for j in range(0, len(anchors), 2)):
                        continue
                    gap_counts = []
                    for j in range(0, len(anchors), 2):
                        xa, xb = anchors[j], anchors[j + 1]
                        if xb <= xa + 1e-12:
                            gap_counts.append(0)
                        else:
                            gap_counts.append(max(1, round(float(env.eval_F(xb) - env.eval_F(xa)))))
                    if mir_chosen:
                        gap_counts = [max(gap_counts[j], gap_counts[-1 - j]) for j in range(len(gap_counts))]
                    for pad_gap in (0, 1, -1, 2):
                        gc = [g_c + pad_gap if g_c + pad_gap >= (0 if idx_g in (0, len(gap_counts) - 1) else 1) else g_c
                              for idx_g, g_c in enumerate(gap_counts)]
                        if mir_chosen and gc != gc[::-1]:
                            continue
                        pinned_spec: list[tuple[int, int, float, float]] = []
                        idx_cur = o0[0] + gc[0]
                        for r_i, (k0, k1) in enumerate(merged_k):
                            nk = k1 - k0
                            pinned_spec.append((idx_cur, idx_cur + nk, lo + k0 * d_f, lo + k1 * d_f))
                            idx_cur += nk + gc[r_i + 1]
                        tot_in = sum(gc) + sum(k1 - k0 for k0, k1 in merged_k)
                        got_al = _solve_blocks(
                            tmap,
                            env,
                            o0,
                            o1,
                            (),
                            (tot_in,),
                            mirror=mir_chosen,
                            periodic=periodic_eff,
                            spacing_eff=min(spacing_eff, max(spacing, d_f)),
                            g_eff=g_eff,
                            cap_margin=0.0,
                            w_floor=cur_w_min * (1.0 - 1e-4),
                            pinned_runs=pinned_spec,
                            stage2=True,
                        )
                        if got_al is not None:
                            e_al, _, _ = got_al
                            if (
                                len(e_al) <= len(e_chosen) + 4
                                and is_aligned(e_al, (o0[0], o1[0]))
                                and _certify_edges(
                                    e_al, tmap, (o0[0], o1[0]), pml_pair, growth, (), periodic_eff, mir_chosen, dt
                                )
                            ):
                                e_chosen, pml_chosen = e_al, (o0[0], o1[0])
                                break
                    if is_aligned(e_chosen, pml_chosen):
                        break

        if not _certify_edges(
            e_chosen, tmap, pml_chosen, pml_pair, growth, kept_centers, periodic_eff, mir_chosen, dt
        ):
            continue

        missed = [c for c in dedup_centers if all(abs(c - kc) > sym_tol for kc in kept_centers)]
        if missed:
            warnings.warn(
                f"axis_edges: {', '.join(f'{c:.6g}' for c in missed)} um not in the middle of a cell (under a PML, or "
                "no room within the targets and the growth limit at the largest smallest cell)",
                stacklevel=2,
            )
        al_final = is_aligned(e_chosen, pml_chosen)
        if align and not al_final:
            warnings.warn(
                f"axis_edges: the {tmap.d_fine:g} um cells do not all start at whole multiples of "
                f"their width from {lo:g} um (the largest smallest cell and mirror symmetry come first)",
                stacklevel=2,
            )
        w_final = np.diff(e_chosen)
        return AxisEdges(
            e_chosen,
            pml_chosen,
            float(w_final.min()),
            al_final,
            mir_chosen,
            tuple(kept_centers),
        )

    raise ValueError(
        f"axis_edges: no cells within their targets and growth={growth:g} survive {dt} precision on [{lo:g}, {hi:g}] um"
    )
