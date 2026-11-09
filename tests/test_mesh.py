"""Tests for the automatic mesher: the lattice edges of one axis (:func:`bifrost.lattice_edges`)."""

from __future__ import annotations

import functools
import itertools
import math

import numpy as np
import pytest

import bifrost as bf
from bifrost.grids import _raster

Q = 0.001  # um: the default quantum


def widths_q(axis: bf.AxisEdges, q: float = Q) -> np.ndarray:
    """Cell widths in whole quanta."""
    w = np.diff(axis.edges) / q
    assert np.allclose(w, np.round(w), atol=1e-6)
    return np.round(w).astype(int)


def check_axis(axis: bf.AxisEdges, lo: float, hi: float, spacing: float, fine, growth: float, q: float = Q) -> None:
    """Every cell and every dual cell within its target, neighbours within growth, edges on the lattice."""
    e = axis.edges
    assert e[0] == lo and e[-1] == hi and np.all(np.diff(e) > 0)
    w = widths_q(axis, q)
    assert np.all(np.max([w[1:] / w[:-1], w[:-1] / w[1:]], axis=0) <= growth + 1e-9)
    t = _raster(lo, q, round((hi - lo) / q), spacing, fine)
    pos = np.concatenate([[0], np.cumsum(w)])
    for i, k in enumerate(w):
        half = (k + 1) // 2
        window = t[max(0, pos[i] - half) : pos[i] + k + half]
        assert k <= window.min(), (i, k, window.min())
    for i in range(len(w) - 1):  # dual cell between the centers of cells i and i + 1
        a, b = pos[i] + w[i] / 2, pos[i + 1] + w[i + 1] / 2
        assert (w[i] + w[i + 1]) / 2 <= t[math.floor(a) : math.ceil(b)].min() + 1e-9
    assert axis.min_cell == pytest.approx(w.min() * q)


def test_lattice_uniform_target_that_divides_the_axis():
    axis = bf.lattice_edges(0.0, 1.0, 0.05)
    assert np.allclose(np.diff(axis.edges), 0.05)
    assert axis.symmetric and axis.aligned and axis.pml_cells == (0, 0)


def test_lattice_uniform_target_that_does_not_divide_the_axis():
    # 1010 nm with cells of at most 50 nm: 21 cells, the smallest as large as possible (48 nm)
    axis = bf.lattice_edges(0.0, 1.01, 0.05)
    w = widths_q(axis)
    assert len(w) == 21 and w.min() == 48 and w.max() <= 50 and w.sum() == 1010
    assert np.array_equal(w, w[::-1])  # mirror-symmetric
    check_axis(axis, 0.0, 1.01, 0.05, (), 1.25)


def test_lattice_fine_region_is_graded_symmetric_and_aligned():
    fine = [(-0.25, 0.25, 0.02)]
    axis = bf.lattice_edges(-1.0, 1.0, 0.06, fine, growth=1.25)
    check_axis(axis, -1.0, 1.0, 0.06, fine, 1.25)
    w = widths_q(axis)
    assert w.min() == 20 and 55 <= w.max() <= 60  # the coarse cells tile their span with the fewest cells
    assert axis.symmetric and np.array_equal(w, w[::-1])
    assert axis.aligned
    starts = np.concatenate([[0], np.cumsum(w)])[:-1]
    assert np.all(starts[w == 20] % 20 == 0)
    # the fine region is covered by 20 nm cells, with the half-cell halo
    centers = -1.0 + Q * (starts + w / 2)
    assert np.all(w[np.abs(centers) <= 0.25] == 20)


def test_lattice_pml_cells_are_uniform_and_cover_the_thickness():
    axis = bf.lattice_edges(0.0, 3.0, 0.08, [(1.2, 1.8, 0.03)], pml=(0.5, 0.7))
    check_axis(axis, 0.0, 3.0, 0.08, [(1.2, 1.8, 0.03)], 1.25)
    w = widths_q(axis)
    n_lo, n_hi = axis.pml_cells
    assert np.all(w[:n_lo] == 80) and n_lo * 80 >= 500 > (n_lo - 1) * 80
    assert np.all(w[-n_hi:] == 80) and n_hi * 80 >= 700 > (n_hi - 1) * 80
    assert not axis.symmetric  # the PMLs differ


def test_lattice_fine_lead_through_a_face_keeps_fine_pml_cells():
    fine = [(0.0, 0.8, 0.02)]  # a guide running out through the lower face
    axis = bf.lattice_edges(0.0, 3.0, 0.08, fine, pml=(0.5, 0.5))
    check_axis(axis, 0.0, 3.0, 0.08, fine, 1.25)
    w = widths_q(axis)
    n_lo, n_hi = axis.pml_cells
    assert np.all(w[:n_lo] == 20) and n_lo == 25
    assert np.all(w[-n_hi:] == 80)


def test_lattice_alignment_dropped_with_a_warning_when_it_cannot_be_met():
    # 20 nm cells everywhere but the axis is not a whole number of them: alignment needs a narrower cell
    with pytest.warns(UserWarning, match="do not all start"):
        axis = bf.lattice_edges(0.0, 0.31, 0.06, [(0.0, 0.31, 0.02)])
    assert not axis.aligned
    assert widths_q(axis).max() <= 20


def test_lattice_is_deterministic():
    fine = [(0.3, 0.45, 0.017), (1.1, 1.13, 0.009)]
    a = bf.lattice_edges(0.0, 2.0, 0.07, fine, pml=(0.3, 0.2))
    b = bf.lattice_edges(0.0, 2.0, 0.07, fine, pml=(0.3, 0.2))
    assert np.array_equal(a.edges, b.edges)
    check_axis(a, 0.0, 2.0, 0.07, fine, 1.25)


def brute_force(targets: np.ndarray, growth: float) -> tuple[int, int]:
    """(largest smallest cell, fewest cells) by plain memoized recursion over every admissible sequence."""
    n = len(targets)

    def admissible(p: int, k: int) -> bool:
        half = (k + 1) // 2
        return p + k <= n and k <= targets[max(0, p - half) : p + k + half].min()

    @functools.cache
    def fewest(p: int, prev: int, floor: int) -> float:
        """Fewest cells from p to the end, every cell >= floor, after a cell of width prev (0: none)."""
        if p == n:
            return 0
        best = math.inf
        for k in range(floor, n - p + 1):
            if not admissible(p, k):
                continue
            if prev and (k > growth * prev + 1e-9 or prev > growth * k + 1e-9):
                continue
            best = min(best, 1 + fewest(p + k, k, floor))
        return best

    for floor in range(int(targets.max()), 0, -1):
        count = fewest(0, 0, floor)
        if count < math.inf:
            return floor, int(count)
    raise AssertionError("no admissible sequence")


@pytest.mark.parametrize("seed", range(25))
def test_lattice_is_optimal_against_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(14, 26))
    spacing = int(rng.integers(5, 9))
    fine = []
    for _ in range(int(rng.integers(1, 3))):
        a = int(rng.integers(0, n - 2))
        fine.append((a, a + int(rng.integers(1, 5)), int(rng.integers(2, spacing))))
    growth = float(rng.choice([1.25, 1.5, 2.0]))
    axis = bf.lattice_edges(0.0, float(n), spacing, fine, quantum=1.0, growth=growth, align=False, symmetric=False)
    w = widths_q(axis, 1.0)
    targets = _raster(0.0, 1.0, n, spacing, fine)
    smallest, count = brute_force(targets, growth)
    assert w.min() == smallest and len(w) == count
    check_axis(axis, 0.0, float(n), spacing, fine, growth, q=1.0)


def test_lattice_mirror_symmetric_targets_give_mirror_symmetric_edges():
    for half_width, d in itertools.product((0.13, 0.2501), (0.011, 0.023)):
        fine = [(-half_width, half_width, d)]
        axis = bf.lattice_edges(-1.2, 1.2, 0.09, fine, pml=(0.4, 0.4), align=False)
        e = axis.edges
        assert axis.symmetric
        assert np.allclose(e - e[0], (e[-1] - e[::-1]), atol=1e-12)
        check_axis(axis, -1.2, 1.2, 0.09, fine, 1.25)


def test_lattice_errors():
    with pytest.raises(ValueError, match="whole number"):
        bf.lattice_edges(0.0, 1.0005, 0.05)
    with pytest.raises(ValueError, match="smaller than the quantum"):
        bf.lattice_edges(0.0, 1.0, 0.05, [(0.4, 0.6, 0.0005)])
    with pytest.raises(ValueError, match="PML"):
        bf.lattice_edges(0.0, 1.0, 0.05, pml=(0.6, 0.6))
    with pytest.raises(ValueError, match="max_states"):
        bf.lattice_edges(0.0, 100.0, 0.5, quantum=0.001)
    with pytest.raises(ValueError, match="growth"):
        bf.lattice_edges(0.0, 1.0, 0.05, growth=1.0)
