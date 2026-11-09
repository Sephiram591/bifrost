"""Cell edges for non-uniform rectilinear grids.

:func:`graded_edges` places the cell edges of one axis: cells no larger than a spacing, finer in given
regions, growing by at most a fixed ratio from one cell to the next, with given coordinates (layer
interfaces, for example) exactly on edges. :func:`lattice_edges` places them exactly on a lattice of
whole quanta: the largest smallest cell, then the fewest cells, under the same kind of targets, with
uniform cells under the PMLs. :meth:`bifrost.StackDevice.rectilinear_grid` turns such edges into the
fdtdx grid of a device's box.

The smallest cell sets the time step of the whole simulation, so refining saves cells and memory, not
time steps.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

__all__ = ["AxisEdges", "graded_edges", "lattice_edges"]

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
    if not hi > lo:
        raise ValueError(f"graded_edges needs lo < hi, got {lo} and {hi}")
    if not spacing > 0:
        raise ValueError(f"spacing must be positive, got {spacing}")
    if not growth > 1:
        raise ValueError(f"growth must be > 1, got {growth}")
    regions = [(float(a), float(b), float(d)) for a, b, d in fine]
    n_lo, n_hi = (int(n) for n in uniform_ends)
    if n_lo < 0 or n_hi < 0:
        raise ValueError(f"uniform_ends must be nonnegative, got {uniform_ends}")
    inner_lo, inner_hi = lo + n_lo * spacing, hi - n_hi * spacing
    if not inner_hi > inner_lo:
        raise ValueError(f"uniform_ends {uniform_ends} of {spacing} leave no room in [{lo}, {hi}]")
    tol = 1e-9 * (hi - lo)
    for a, b, d in regions:
        if not (b > a and d > 0):
            raise ValueError(f"a fine region is (a, b, d) with a < b and d > 0, got {(a, b, d)}")
        if a < inner_lo - tol or b > inner_hi + tol:
            raise ValueError(f"fine region [{a}, {b}] is outside [{inner_lo}, {inner_hi}] (the uniform ends)")

    def size(x: np.ndarray) -> np.ndarray:
        """The cell size at x."""
        s = np.full_like(x, spacing)
        for a, b, d in regions:
            s = np.minimum(s, d + (growth - 1.0) * np.maximum(np.maximum(a - x, x - b), 0.0))
        return s

    points = [inner_lo]
    dropped = []
    for z in sorted(float(z) for z in snap):
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
    """The cell edges of one axis, from :func:`lattice_edges`."""

    #: (n + 1,) strictly increasing edges (um), the first ``lo`` and the last ``hi``
    edges: np.ndarray
    #: uniform cells under the PML at the lower and at the upper end
    pml_cells: tuple[int, int]
    #: the smallest cell (um): this axis's share of the time step
    min_cell: float
    #: the cells of the finest target's width (outside the PMLs) start at whole multiples of it from ``lo``
    aligned: bool
    #: the edges are mirror-symmetric about the middle of the axis
    symmetric: bool


# tolerance of the ratio comparisons (in quanta)
_TOL = 1e-9
# PML widths tried per end when the widest admissible one does not lead to a solution
_PML_TRIES = 6


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
    ratios). With ``symmetric``, mirror-symmetric targets and PMLs give mirror-symmetric edges. With
    ``align``, cells as wide as the finest target start at whole multiples of it from ``lo``, so that they
    coincide with a uniform grid of that spacing; where that cannot be met it is dropped with a warning.

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
        max_states: refuse problems with more states (lattice positions times cell widths) than this.

    Returns:
        The edges, the PML cells at both ends, the smallest cell, and whether the cells are aligned and
        mirror-symmetric.

    Raises:
        ValueError: on an empty range, a range that is not whole quanta, a target smaller than a quantum, a
            nonpositive spacing or quantum, ``growth <= 1``, PMLs that do not fit, a problem larger than
            ``max_states``, or targets the growth limit cannot meet (the message names where).
    """
    lo, hi, q = float(lo), float(hi), float(quantum)
    spacing, growth = float(spacing), float(growth)
    if not q > 0:
        raise ValueError(f"quantum must be positive, got {quantum}")
    if not hi > lo:
        raise ValueError(f"lattice_edges needs lo < hi, got {lo} and {hi}")
    if not spacing > 0:
        raise ValueError(f"spacing must be positive, got {spacing}")
    if not growth > 1:
        raise ValueError(f"growth must be > 1, got {growth}")
    n = round((hi - lo) / q)
    if abs(n * q - (hi - lo)) > 1e-6 * q:
        raise ValueError(f"lattice_edges: hi - lo = {hi - lo:.9g} um is not a whole number of {q:g} um quanta")
    thick = tuple(float(t) / q for t in pml)
    if len(thick) != 2 or min(thick) < 0:
        raise ValueError(f"pml must be two nonnegative thicknesses, got {pml}")
    targets = _raster(lo, q, n, spacing, fine)
    states = (n + 1) * (int(targets.max()) + 1)
    if states > max_states:
        raise ValueError(
            f"lattice_edges: {n} quanta by {int(targets.max())} cell widths is {states:.3g} states, more than "
            f"max_states={max_states:.3g}; use a coarser quantum"
        )
    widths, ends, mirror = _Lattice(targets, growth).solve(thick, align, symmetric, lo, q)
    # aligned: every cell of the finest target's width outside the PMLs starts on that width's lattice
    fine_k, starts = int(targets.min()), np.concatenate([[0], np.cumsum(widths)[:-1]])
    inner = slice(ends[0][0], len(widths) - ends[1][0])
    w_in, s_in = np.asarray(widths[inner]), starts[inner]
    aligned = bool(np.all(s_in[w_in == fine_k] % fine_k == 0))
    if align and not aligned:
        warnings.warn(
            f"lattice_edges: the {int(targets.min()) * q:g} um cells do not all start at whole multiples of "
            f"their width from {lo:g} um (the largest smallest cell and mirror symmetry come first)",
            stacklevel=2,
        )
    edges = lo + q * np.concatenate([[0], np.cumsum(widths)]).astype(np.float64)
    edges[0], edges[-1] = lo, hi
    return AxisEdges(edges, (ends[0][0], ends[1][0]), q * min(widths), aligned, mirror)


def _raster(lo: float, q: float, n: int, spacing: float, fine: Sequence[tuple[float, float, float]]) -> np.ndarray:
    """(n,) targets in whole quanta of the lattice's cells: the smallest region touching each one."""
    size = np.full(n, spacing)
    for region in fine:
        a, b, d = (float(x) for x in region)
        if not (b >= a and d > 0):
            raise ValueError(f"a fine region is (a, b, d) with a <= b and d > 0, got {tuple(region)}")
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
    ) -> list[int] | None:
        """The best cell widths (quanta) with every cell >= k_lo, or None if there are none.

        Backward over lattice positions, the state being the width of the previous cell; ``mirror``
        solves the lower half (each cell counting twice) and ends on the middle edge or a middle cell.
        """
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
                if 2 * p == n:
                    term[:] = 0.0
                c = n - 2 * p
                if k_lo <= c <= self.widest[p] and not (align and c == align and p % align):
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
        self, thickness: tuple[float, float], align: bool, symmetric: bool, lo: float, q: float
    ) -> tuple[list[int], tuple[tuple[int, int], tuple[int, int]], bool]:
        """(widths, PML ends, mirrored): the largest smallest cell first, then the best widths."""
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
        k_lo, ends = found
        # the PML choices to try for alignment: the one found first, then narrower ones (the fine cells' lattice
        # may need narrower PML cells to be reached within the growth limit)
        choices = [ends, *(e for e in self.ends(thickness, k_lo) if e != ends)]
        modes: list[tuple[bool, int | None, tuple]] = []
        if symmetric and self.mirrored and thickness[0] == thickness[1]:
            if align and self.n % self.k_fine == 0:
                modes += [(True, self.k_fine, e) for e in choices if e[0] == e[1]]
            modes += [(True, None, e) for e in choices[:1] if e[0] == e[1]]
        if align:
            modes += [(False, self.k_fine, e) for e in choices]
        modes.append((False, None, ends))
        for mirror, aligned, pml in modes:
            widths = self.dp(k_lo, pml, aligned, mirror, eps)
            if widths is not None:
                return widths, pml, mirror
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
