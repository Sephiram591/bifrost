"""Cell edges for non-uniform rectilinear grids.

:func:`graded_edges` places the cell edges of one axis: cells no larger than a spacing, finer in given
regions, growing by at most a fixed ratio from one cell to the next, with given coordinates (layer
interfaces, for example) exactly on edges. :meth:`bifrost.StackDevice.rectilinear_grid` turns such edges
into the fdtdx grid of a device's box.

The smallest cell sets the time step of the whole simulation, so refining saves cells and memory, not
time steps.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Sequence

import numpy as np

__all__ = ["graded_edges"]

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
