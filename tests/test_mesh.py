"""Tests for the automatic mesher: continuous (:func:`bifrost.axis_edges`) and discrete
(:func:`bifrost.lattice_edges`) 1D axis solvers, and 3D device refinements (:mod:`bifrost.meshing`)."""

from __future__ import annotations

import functools
import itertools
import math
import warnings
from types import SimpleNamespace

import fdtdx
import gdsfactory as gf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gdsfactory.technology import LayerLevel, LayerStack

import bifrost as bf
from bifrost import meshing
from bifrost.grids import _TargetMap, _centered, _raster

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


def test_lattice_alignment_takes_narrower_pml_cells_when_it_must():
    # the fine block ends close to the PML: wide PML cells cannot reach the 20 nm lattice within the growth limit
    fine = [(-0.39, 0.61, 0.02)]
    axis = bf.lattice_edges(-1.48, 1.72, 0.03, fine, pml=(1.0, 1.0))
    assert axis.aligned
    check_axis(axis, -1.48, 1.72, 0.03, fine, 1.25)
    w = widths_q(axis)
    starts = np.concatenate([[0], np.cumsum(w)])[:-1]
    inner = slice(axis.pml_cells[0], len(w) - axis.pml_cells[1])
    assert np.all(starts[inner][w[inner] == 20] % 20 == 0)


def middles(axis: bf.AxisEdges) -> np.ndarray:
    """The middle of every cell (um)."""
    return 0.5 * (axis.edges[1:] + axis.edges[:-1])


def test_lattice_centers_lie_in_the_middle_of_a_cell():
    fine = [(0.2, 0.6, 0.02)]
    centers = (0.3, 0.513)
    axis = bf.lattice_edges(0.0, 1.0, 0.05, fine, centers=centers, align=False)
    check_axis(axis, 0.0, 1.0, 0.05, fine, 1.25)
    assert axis.centered == centers
    for c in centers:
        assert np.min(np.abs(middles(axis) - c)) <= 0.0005 + 1e-12  # within half a quantum
        assert np.min(np.abs(axis.edges - c)) > 1e-3  # no edge on it
    plain = bf.lattice_edges(0.0, 1.0, 0.05, fine, align=False)
    assert axis.min_cell >= 0.8 * plain.min_cell - 1e-12  # centering costs at most a fifth of the smallest cell


def test_lattice_symmetric_centers_keep_the_mirror():
    fine = [(-0.3, 0.3, 0.02)]
    axis = bf.lattice_edges(-1.0, 1.0, 0.06, fine, centers=(-0.11, 0.11), pml=(0.3, 0.3), align=False)
    assert axis.symmetric and axis.centered == (-0.11, 0.11)
    e = axis.edges
    assert np.allclose(e - e[0], e[-1] - e[::-1], atol=1e-12)


def test_lattice_drops_centers_without_room_with_a_warning():
    # two faces 0.004 um apart in 0.02 um cells: one fits, the other cannot also be centered
    with pytest.warns(UserWarning, match="not in the middle of a cell"):
        axis = bf.lattice_edges(0.0, 1.0, 0.02, centers=(0.5, 0.504), align=False)
    assert len(axis.centered) == 1
    check_axis(axis, 0.0, 1.0, 0.02, (), 1.25)


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


def test_graded_and_lattice_reject_non_finite_and_sub_quantum():
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            bf.graded_edges(bad, 1.0, 0.05)
        with pytest.raises(ValueError):
            bf.graded_edges(0.0, bad, 0.05)
        with pytest.raises(ValueError):
            bf.graded_edges(0.0, 1.0, bad)
        with pytest.raises(ValueError):
            bf.graded_edges(0.0, 1.0, 0.05, growth=bad)
        with pytest.raises(ValueError):
            bf.graded_edges(0.0, 1.0, 0.05, fine=[(0.2, 0.5, bad)])
        with pytest.raises(ValueError):
            bf.graded_edges(0.0, 1.0, 0.05, snap=(bad,))

        with pytest.raises(ValueError):
            bf.lattice_edges(bad, 1.0, 0.05)
        with pytest.raises(ValueError):
            bf.lattice_edges(0.0, bad, 0.05)
        with pytest.raises(ValueError):
            bf.lattice_edges(0.0, 1.0, 0.05, quantum=bad)
        with pytest.raises(ValueError):
            bf.lattice_edges(0.0, 1.0, bad)
        with pytest.raises(ValueError):
            bf.lattice_edges(0.0, 1.0, 0.05, growth=bad)
        with pytest.raises(ValueError):
            bf.lattice_edges(0.0, 1.0, 0.05, pml=(0.1, bad))
        with pytest.raises(ValueError):
            bf.lattice_edges(0.0, 1.0, 0.05, centers=(bad,))
        with pytest.raises(ValueError):
            bf.lattice_edges(0.0, 1.0, 0.05, fine=[(0.2, 0.5, bad)])
        with pytest.raises(ValueError):
            bf.lattice_edges(0.0, 1.0, 0.05, max_states=bad)

        with pytest.raises(ValueError):
            bf.axis_edges(0.0, 1.0, 0.05, centers=(bad,))

    with pytest.raises(ValueError):
        bf.lattice_edges(0.0, 0.0004, 0.05, quantum=0.001)


def test_centered_and_empty_fine_alignment():
    # Edges on half-quantum 2 must never pass as centered even for width-1 cells
    assert _centered(np.array([2]), 0, np.array([1, 2, 3])).tolist() == [False, True, True]
    assert _centered(np.array([2]), 2, np.array([1, 2])).tolist() == [False, False]
    # Width-1 cells cannot be centered even at their midpoint
    assert _centered(np.array([1]), 0, np.array([1, 2])).tolist() == [False, True]

    # When target caps cell width at 1 quantum everywhere, the boundary center must be dropped with warning
    with pytest.warns(UserWarning, match="not in the middle of a cell"):
        axis = bf.lattice_edges(0.0, 0.01, 0.001, centers=(0.005,), quantum=0.001, align=False)
    assert axis.centered == ()

    # Fine region entirely inside a PML leaves zero d_fine cells in the interior: no spurious alignment warning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        axis_c = bf.axis_edges(0.0, 2.0, 0.08, [(0.0, 0.2, 0.02)], pml=(0.5, 0.5), align=True)
    assert axis_c.aligned
    assert not any("do not all start" in str(w.message) for w in caught)


# ---------------------------------------------------------------- continuous axis_edges


def check_continuous_axis(
    axis: bf.AxisEdges,
    lo: float,
    hi: float,
    spacing: float,
    fine=(),
    growth: float = 1.25,
    pml: tuple[float, float] = (0.0, 0.0),
    periodic: bool = False,
) -> None:
    """Every primal cell, half-cell halo, and Yee dual cell within its continuous target, neighbours within growth."""
    e = axis.edges
    assert e[0] == pytest.approx(lo, abs=1e-12) and e[-1] == pytest.approx(hi, abs=1e-12)
    w = np.diff(e)
    assert np.all(w > 0)
    if len(w) > 1:
        assert np.all(np.maximum(w[1:] / w[:-1], w[:-1] / w[1:]) <= growth * (1 + 1e-6))
    if periodic and pml == (0.0, 0.0) and len(w) > 1:
        assert max(w[0] / w[-1], w[-1] / w[0]) <= growth * (1 + 1e-6)
    tmap = _TargetMap(lo, hi, spacing, fine)
    caps = tmap.min_on_vec(np.maximum(lo, e[:-1] - 0.5 * w), np.minimum(hi, e[1:] + 0.5 * w))
    assert np.all(w <= caps * (1 + 1e-6))
    if len(w) > 1:
        c = 0.5 * (e[:-1] + e[1:])
        dcaps = tmap.min_on_vec(c[:-1], c[1:])
        assert np.all(0.5 * (w[:-1] + w[1:]) <= dcaps * (1 + 1e-6))
    n_lo, n_hi = axis.pml_cells
    if pml[0] > 0:
        assert n_lo >= 1 and np.allclose(w[:n_lo], w[0], rtol=1e-7, atol=1e-12)
        assert e[n_lo] - lo >= pml[0] * (1 - 1e-7)
    if pml[1] > 0:
        assert n_hi >= 1 and np.allclose(w[-n_hi:], w[-1], rtol=1e-7, atol=1e-12)
        assert hi - e[-n_hi - 1] >= pml[1] * (1 - 1e-7)
    assert axis.min_cell == pytest.approx(float(w.min()), rel=1e-12)


def test_axis_edges_uniform_exact_for_arbitrary_boxes():
    for lo, hi, spacing, pml in (
        (0.0, 1.00037, 0.05, (0.0, 0.0)),
        (-0.5 * math.pi, 0.5 * math.pi, 0.073, (0.3, 0.3)),
        (0.0, 43.0, 1.55 / (3.48 * 14.0), (1.0, 1.0)),
    ):
        axis = bf.axis_edges(lo, hi, spacing, pml=pml, align=False)
        check_continuous_axis(axis, lo, hi, spacing, (), 1.25, pml=pml)
        w = np.diff(axis.edges)
        assert np.ptp(w) <= 1e-12  # single uniform cell width across the whole axis
        assert axis.symmetric

    # Exact uniform lattices stay aligned and do not add +1 cell or +1 PML cell under float32 precision
    for lo, hi, spacing, pml, n_exp, pml_exp in (
        (-21.5, 21.5, 0.02, (0.5, 0.5), 2150, (25, 25)),
        (-3.0, 3.0, 0.02, (1.0, 1.0), 300, (50, 50)),
    ):
        axis_f32 = bf.axis_edges(lo, hi, spacing, pml=pml, align=True, dtype=jnp.float32)
        check_continuous_axis(axis_f32, lo, hi, spacing, (), 1.25, pml=pml)
        assert len(axis_f32.edges) - 1 == n_exp
        assert axis_f32.pml_cells == pml_exp
        assert axis_f32.aligned and axis_f32.symmetric


def test_axis_edges_smooth_variation_without_lattice_plateaus():
    targets = np.linspace(0.036, 0.044, 17)
    mins = []
    for d in targets:
        axis = bf.axis_edges(-1.5, 1.5, 0.09, [(-0.3, 0.3, float(d))], pml=(0.4, 0.4), align=False)
        check_continuous_axis(axis, -1.5, 1.5, 0.09, [(-0.3, 0.3, float(d))], 1.25, pml=(0.4, 0.4))
        assert axis.symmetric
        mins.append(axis.min_cell)
    # Continuous solver tracks every target to < 0.05 nm instead of collapsing onto 1 nm integer steps
    assert len(np.unique(np.round(mins, 6))) == len(targets)
    np.testing.assert_allclose(mins, targets, rtol=2e-3)


def test_axis_edges_exact_centers_symmetry_and_conflict_drop():
    fine = [(0.2, 0.6, 0.02)]
    centers = (0.3, 0.5137)
    axis = bf.axis_edges(0.0, 1.0003, 0.05, fine, centers=centers, align=False)
    check_continuous_axis(axis, 0.0, 1.0003, 0.05, fine, 1.25)
    assert axis.centered == centers
    for c in centers:
        assert np.min(np.abs(middles(axis) - c)) <= 1e-10
    plain = bf.axis_edges(0.0, 1.0003, 0.05, fine, align=False)
    assert axis.min_cell >= 0.8 * plain.min_cell - 1e-12

    # Symmetric pair of centers and odd-cell midpoint center both preserve exact mirror symmetry
    for sym_cs in ((-0.113, 0.113), (0.0,), (-0.21, 0.0, 0.21)):
        sym_ax = bf.axis_edges(-1.0, 1.0, 0.06, [(-0.3, 0.3, 0.02)], centers=sym_cs, pml=(0.3, 0.3), align=False)
        check_continuous_axis(sym_ax, -1.0, 1.0, 0.06, [(-0.3, 0.3, 0.02)], 1.25, pml=(0.3, 0.3))
        assert sym_ax.symmetric and sym_ax.centered == sym_cs
        for c in sym_cs:
            assert np.min(np.abs(middles(sym_ax) - c)) <= 1e-10
        np.testing.assert_allclose(sym_ax.edges - sym_ax.edges[0], sym_ax.edges[-1] - sym_ax.edges[::-1], atol=1e-12)

    with pytest.warns(UserWarning, match="not in the middle of a cell"):
        dropped = bf.axis_edges(0.0, 1.0, 0.02, centers=(0.5, 0.504), align=False)
    assert len(dropped.centered) == 1
    check_continuous_axis(dropped, 0.0, 1.0, 0.02)


def test_axis_edges_pml_leads_periodic_and_millimeter_scale():
    # Fine waveguide lead running out through the lower PML
    lead_fine = [(0.0, 0.8, 0.02)]
    lead = bf.axis_edges(0.0, 3.0, 0.08, lead_fine, pml=(0.5, 0.5), align=False)
    check_continuous_axis(lead, 0.0, 3.0, 0.08, lead_fine, 1.25, pml=(0.5, 0.5))
    w_lead = np.diff(lead.edges)
    n_lo, n_hi = lead.pml_cells
    assert w_lead[:n_lo].max() <= 0.02 + 1e-9 and w_lead[-n_hi:].min() >= 0.05

    # Periodic wrap-around growth enforcement across lo/hi when asymmetric fine region touches lo
    per_fine = [(0.0, 0.12, 0.015)]
    per = bf.axis_edges(0.0, 1.5, 0.08, per_fine, periodic=True, symmetric=False, align=False)
    check_continuous_axis(per, 0.0, 1.5, 0.08, per_fine, 1.25, periodic=True)

    # Millimeter-scale axis (1000 um with 20 nm fine region) solves cleanly without lattice state blow-up
    mm_fine = [(499.5, 500.5, 0.02)]
    mm = bf.axis_edges(0.0, 1000.0, 0.5, mm_fine, pml=(1.0, 1.0), align=False, dtype=jnp.float32)
    check_continuous_axis(mm, 0.0, 1000.0, 0.5, mm_fine, 1.25, pml=(1.0, 1.0))
    assert mm.symmetric and len(mm.edges) < 2500


def test_axis_edges_errors():
    with pytest.raises(ValueError, match="lo < hi"):
        bf.axis_edges(1.0, 1.0, 0.05)
    with pytest.raises(ValueError, match="spacing"):
        bf.axis_edges(0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="growth"):
        bf.axis_edges(0.0, 1.0, 0.05, growth=1.0)
    with pytest.raises(ValueError, match="PML"):
        bf.axis_edges(0.0, 1.0, 0.05, pml=(0.6, 0.6))
    with pytest.raises(ValueError, match="float64"):
        bf.axis_edges(0.0, 1e5, 0.5, [(5e4, 5e4 + 0.1, 0.005)], dtype=jnp.float32)


# ---------------------------------------------------------------- geometry of the refinements

LAYER = gf.gpdk.LAYER
N_SI, N_OX = 3.48, 1.444
MATS = {
    "si": fdtdx.Material(permittivity=N_SI**2),
    "sio2": fdtdx.Material(permittivity=N_OX**2),
    "air": fdtdx.Material(),
}


@pytest.fixture
def pdk():
    gf.gpdk.PDK.activate()
    gf.snap.SNAP_ENABLED = False
    yield
    gf.snap.SNAP_ENABLED = True


def device(rects=(), levels: dict | None = None, mats: dict | None = None, angle: float = 0.0,
           **kwargs) -> bf.StackDevice:
    """Silicon rectangles (x0, y0, x1, y1) on WG (0 to 0.22 um) and the given extra levels, in silica.

    A SLAB150 rectangle sets the box laterally (2 x 2 um); its level ("frame", silica, -0.6 to 1.0 um) sets
    it in z. Extra levels may draw on SLAB90 (rectangles in ``levels["_slab90"]``).
    """
    levels = dict(levels or {})
    slab90 = levels.pop("_slab90", ())
    c = gf.Component()
    c.add_polygon([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)], layer=LAYER.SLAB150)
    for layer, shapes in ((LAYER.WG, rects or [(-1.0, -0.25, 1.0, 0.25)]), (LAYER.SLAB90, slab90)):
        for x0, y0, x1, y1 in shapes:
            c.add_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer=layer)
    stack = {
        "core": LayerLevel(layer=LAYER.WG, zmin=0.0, thickness=0.22, material="si", mesh_order=2,
                           sidewall_angle=angle),
        "frame": LayerLevel(layer=LAYER.SLAB150, zmin=-0.6, thickness=1.6, material="sio2", mesh_order=9),
        **levels,
    }
    return bf.StackDevice(
        c, LayerStack(layers=stack), mats or MATS, background="sio2", name="s", zrange_layers=["frame"],
        diagnostics=False, **kwargs,
    )


def pocket_device() -> bf.StackDevice:
    """A 1 um wide silicon strip with an air pocket (SLAB90, painted over it) in its middle."""
    pocket = LayerLevel(layer=LAYER.SLAB90, zmin=0.0, thickness=0.22, material="air", mesh_order=1)
    return device([(-1.0, -0.5, 1.0, 0.5)], levels={"pocket": pocket, "_slab90": [(-0.3, -0.2, 0.3, 0.2)]})


def contributions(geom, what: str) -> list:
    return [c for c in geom.contributions if c.what == what]


def y_extent(geom, region) -> tuple[float, float]:
    spans = geom.plane.extent(region, 1)
    return min(a for a, _ in spans), max(b for _, b in spans)


def test_indices_and_targets():
    lam = meshing._wavelengths(1.55, MATS)
    assert lam.tolist() == [1.55]
    idx = meshing._index(MATS["si"], lam)
    assert idx.n[0] == pytest.approx(N_SI) and not idx.unsupported
    aniso = meshing._index(fdtdx.Material(permittivity=(9.0, 12.0, 14.0)), lam)
    assert aniso.n[0] == pytest.approx(math.sqrt(14.0))
    band = meshing._wavelengths((1.5, 1.6), MATS)
    assert band[0] == 1.5 and band[-1] == 1.6 and len(band) == 33
    w0 = 2 * math.pi * meshing._C0 / (1.55e-6 / 1.3)  # a Lorentz resonance inside the band is sampled
    lorentz = fdtdx.Material(permittivity=2.0, dispersion=fdtdx.DispersionModel(
        poles=(fdtdx.LorentzPole(resonance_frequency=w0, damping=1e12, delta_epsilon=0.5),)))
    lam_l = meshing._wavelengths((1.1, 1.6), {"l": lorentz})
    assert np.any(np.isclose(lam_l, 1.55 / 1.3))
    got = meshing._index(lorentz, np.array([1.5]))
    eps = np.real(lorentz.dispersion.permittivity_tensor(2 * math.pi * meshing._C0 / 1.5e-6, lorentz.permittivity))
    assert got.n[0] == pytest.approx(math.sqrt(eps[0, 0]))
    assert meshing._index(fdtdx.Material(permittivity=2.0, electric_conductivity=1e3), lam).unsupported
    drude = fdtdx.Material(dispersion=fdtdx.DispersionModel(
        poles=(fdtdx.DrudePole(plasma_frequency=1.4e16, damping=1e14),)))
    assert meshing._index(drude, lam).unsupported


def test_strip_material_and_tail_targets(pdk):
    geom = meshing._geometry(device(), 1.55, 14.0, meshing.Refine())
    t_si, t_ox = 1.55 / (N_SI * 14), 1.55 / (N_OX * 14)
    assert geom.cell["si"] == pytest.approx(t_si) and geom.cell["sio2"] == pytest.approx(t_ox)
    (core,) = contributions(geom, "material core")
    assert core.size == pytest.approx(t_si) and core.z == pytest.approx((0.0, 0.22))
    assert y_extent(geom, core.region) == pytest.approx((-0.25, 0.25), abs=1e-3)
    ell = 1.55 / (2 * math.pi * math.sqrt(N_SI**2 - N_OX**2))
    assert geom.tails["core"] == pytest.approx((ell, ell, ell))
    lateral, vertical = contributions(geom, "tail core")
    assert y_extent(geom, lateral.region) == pytest.approx((-0.25 - 3 * ell, 0.25 + 3 * ell), abs=1e-3)
    assert vertical.z == pytest.approx((-3 * ell, 0.22 + 3 * ell))
    assert geom.tails["frame"] == (0.0, 0.0, 0.0)  # silica: nothing of a lower index next to it
    _, y, z = meshing._axis_targets(geom)
    assert min(s for *_, s in y) == pytest.approx(t_si)
    assert any(a == pytest.approx(-0.25 - 3 * ell, abs=1e-3) and s == pytest.approx(t_si) for a, _, s in y)
    assert any(a == pytest.approx(-3 * ell) and s == pytest.approx(t_si) for a, _, s in z)


def test_sloped_strip_bounds(pdk):
    refine = meshing.Refine(global_dilation=0, global_z_padding=0)
    geom = meshing._geometry(device(angle=10.0), 1.55, 14.0, refine)
    (core,) = [lv for lv in geom.levels if lv.name == "core"]
    shrink = 0.22 * math.tan(math.radians(10.0))  # narrower toward the top (z_ref at the bottom)
    assert y_extent(geom, core.upper) == pytest.approx((-0.25, 0.25), abs=1e-3)
    assert y_extent(geom, core.lower) == pytest.approx((-0.25 + shrink, 0.25 - shrink), abs=1e-3)


def test_dilation_band_and_per_vertex_taper(pdk):
    refine = meshing.Refine(dilation={"WG": 0.1}, global_dilation=0, global_z_padding=0)
    geom = meshing._geometry(device(), 1.55, 14.0, refine)
    assert y_extent(geom, geom.possible["core"]) == pytest.approx((-0.35, 0.35), abs=1e-3)
    d = device()
    v = d.source_vertices[[k for k, _ in d.source_layers].index("vertices/WG")]
    taper = np.zeros(len(v))
    taper[np.argmin(v[:, 0] + v[:, 1])] = 0.2  # only the vertex at (-1, -0.25) moves
    geom = meshing._geometry(d, 1.55, 14.0, meshing.Refine(dilation={"vertices/WG": taper}, global_dilation=0,
                                                           global_z_padding=0))
    lo, hi = y_extent(geom, geom.possible["core"])
    assert lo == pytest.approx(-0.45, abs=1e-3) and hi == pytest.approx(0.25, abs=1e-3)


def test_z_padding_extends_the_span(pdk):
    refine = meshing.Refine(z_padding={"core": (0.05, 0.1)}, global_dilation=0, global_z_padding=0)
    geom = meshing._geometry(device(), 1.55, 14.0, refine)
    assert geom.padded["core"] == pytest.approx((-0.05, 0.32))
    spans = [c.z for c in contributions(geom, "material core")]
    assert min(a for a, _ in spans) == pytest.approx(-0.05) and max(b for _, b in spans) == pytest.approx(0.32)


def test_narrow_gap_is_a_feature(pdk):
    rects = [(-1.0, -0.25, 1.0, 0.25), (-1.0, 0.29, 1.0, 0.6)]
    geom = meshing._geometry(device(rects), 1.55, 14.0, meshing.Refine(global_dilation=0, global_z_padding=0))
    feats = contributions(geom, "feature core")
    assert feats and min(c.size for c in feats) == pytest.approx(0.04 / 1.5, rel=1e-3)
    lo, hi = y_extent(geom, min(feats, key=lambda c: c.size).region)
    assert lo == pytest.approx(0.25, abs=1e-3) and hi == pytest.approx(0.29, abs=1e-3)


def test_static_pocket_hides_the_core_and_gets_no_tail(pdk):
    geom = meshing._geometry(pocket_device(), 1.55, 14.0, meshing.Refine())
    assert geom.tails["pocket"] == (0.0, 0.0, 0.0)  # air next to silicon only: no tail of its own
    center = geom.plane.region_box((-0.1, -0.05), (0.1, 0.05))
    assert (geom.possible["core"] & center).is_empty()  # silicon cannot be under the static pocket
    assert not (geom.possible["pocket"] & center).is_empty()


def test_moving_pocket_does_not_hide_the_core(pdk):
    geom = meshing._geometry(pocket_device(), 1.55, 14.0, meshing.Refine(dilation={"SLAB90": 0.05}))
    center = geom.plane.region_box((-0.1, -0.05), (0.1, 0.05))
    assert not (geom.possible["core"] & center).is_empty()


def test_pocket_inner_boundary_gets_the_core_tail(pdk):
    geom = meshing._geometry(pocket_device(), 1.55, 14.0, meshing.Refine())
    ell = 1.55 / (2 * math.pi * math.sqrt(N_SI**2 - N_OX**2))  # silica is the highest index below silicon
    assert geom.tails["core"][0] == pytest.approx(ell)
    lateral = contributions(geom, "tail core")[0]
    center = geom.plane.region_box((-0.1, -0.05), (0.1, 0.05))  # 3 l ~ 0.23 um into the pocket from its edges
    assert not (lateral.region & center).is_empty()


def test_fill_slab_gets_z_tails_only(pdk):
    slab = LayerLevel(layer=LAYER.WAFER, zmin=-0.4, thickness=0.1, material="si", mesh_order=5)
    geom = meshing._geometry(device(levels={"slab": slab}), (1.5, 1.6), 14.0, meshing.Refine())
    lateral, below, above = geom.tails["slab"]
    ell = 1.6 / (2 * math.pi * math.sqrt(N_SI**2 - N_OX**2))  # the longest wavelength
    assert lateral == 0.0 and below == pytest.approx(ell) and above == pytest.approx(ell)


def test_no_tail_keeps_the_padding(pdk):
    refine = meshing.Refine(no_tail=("core",), z_padding={"core": 0.05})
    geom = meshing._geometry(device(), 1.55, 14.0, refine)
    assert not contributions(geom, "tail core")
    assert geom.padded["core"] == pytest.approx((-0.05, 0.27))
    spans = [c.z for c in contributions(geom, "material core")]
    assert min(a for a, _ in spans) == pytest.approx(-0.05)


def cells_where(m, axis: int, size: float) -> tuple[float, float]:
    """(first, last) coordinate covered by cells of at most ``size`` um along an axis."""
    e = m.edges[axis]
    w = np.diff(e)
    fine = np.nonzero(w <= size + 1e-6)[0]
    return float(e[fine.min()]), float(e[fine.max() + 1])


def test_mesh_one_index_is_uniform(pdk):
    same = {**MATS, "si": MATS["sio2"]}
    m = bf.mesh(device(mats=same), 1.55, ppw=14, pml=0.3)
    t = 1.55 / (N_OX * 14)
    for axis in m.axes:
        w = np.diff(axis.edges)
        assert np.ptp(w) <= 1e-12 and w.max() <= t + 1e-9  # exact single cell size along each axis
    mq = bf.mesh(device(mats=same), 1.55, ppw=14, pml=0.3, quantum=Q)
    for axis in mq.axes:
        wq = widths_q(axis)
        assert wq.max() <= math.floor(t * 1e3 + 1e-9) and wq.min() >= wq.max() - 4


def test_mesh_strip_end_to_end(pdk):
    d = device()
    m = bf.mesh(d, 1.55, ppw=14, refine=bf.Refine(dilation={"WG": 0.1}), pml=0.3)
    t_si = 1.55 / (N_SI * 14)
    ell = 1.55 / (2 * math.pi * math.sqrt(N_SI**2 - N_OX**2))
    # x: the strip runs through every x plane -> uniform continuous cells
    wx = np.diff(m.edges[0])
    assert np.ptp(wx) <= 1e-12 and wx.max() <= t_si + 1e-6
    # the fine cells reach past the region by half the next cell (its halo) and up to one fine cell of tiling
    lo, hi = cells_where(m, 1, t_si)
    assert lo == pytest.approx(-0.35 - 3 * ell, abs=0.06) and hi == pytest.approx(0.35 + 3 * ell, abs=0.06)
    assert lo <= -0.35 - 3 * ell and hi >= 0.35 + 3 * ell
    lo, hi = cells_where(m, 2, t_si)
    assert lo == pytest.approx(-3 * ell, abs=0.06) and hi == pytest.approx(0.22 + 3 * ell, abs=0.06)
    assert lo <= -3 * ell and hi >= 0.22 + 3 * ell
    assert (m.edges[1][0], m.edges[1][-1], m.edges[2][0], m.edges[2][-1]) == pytest.approx((-1.0, 1.0, -0.6, 1.0))
    for axis in m.axes:
        w = np.diff(axis.edges)
        assert np.all(w > 0) and np.all(np.maximum(w[1:] / w[:-1], w[:-1] / w[1:]) <= 1.25 * (1 + 1e-6))
    assert m.pml_cells == tuple(ax.pml_cells[e] for ax in m.axes for e in (0, 1))
    bc = m.boundary
    assert bc.boundary_type_minx == "pml" and bc.thickness_grid_miny == m.pml_cells[2]
    grid = m.grid
    np.testing.assert_allclose(np.asarray(grid.y_edges) * 1e6, m.edges[1], atol=1e-5)
    text = m.summary(time=1e-13)
    assert "cells" in text and "steps" in text and "time step" in text
    assert "starting geometry: 0 of" in text


def test_mesh_centers_the_core_faces_by_default(pdk):
    d = device()
    m = bf.mesh(d, 1.55, ppw=14, pml=0.3)
    z = m.edges[2]
    mid = 0.5 * (z[1:] + z[:-1])
    for face in (0.0, 0.22):  # the core's static faces against silica sit at exact cell midpoints
        assert np.min(np.abs(mid - face)) <= 1e-9 and np.min(np.abs(z - face)) > 1e-3
    assert m.axes[2].centered == (0.0, 0.22)
    assert "faces centered at 0, 0.22 um" in m.summary()
    off = bf.mesh(d, 1.55, ppw=14, pml=0.3, center_faces=False)
    assert off.axes[2].centered == ()
    assert m.axes[2].min_cell >= 0.8 * off.axes[2].min_cell - 1e-12


def test_mesh_skips_centering_faces_moved_by_z_padding(pdk):
    d = device()
    top_moves = bf.mesh(d, 1.55, ppw=14, refine=bf.Refine(z_padding={"core": (0.0, 0.05)}), pml=0.3)
    assert top_moves.axes[2].centered == (0.0,)
    both_move = bf.mesh(d, 1.55, ppw=14, refine=bf.Refine(z_padding={"core": 0.05}), pml=0.3)
    assert both_move.axes[2].centered == ()


def test_mesh_periodic_faces(pdk):
    m = bf.mesh(device(), 1.55, boundaries=("periodic", "periodic", "pml", "pml", "pml", "pml"), pml=0.3)
    assert m.pml_cells[:2] == (0, 0) and m.boundaries[0] == "periodic"
    assert m.boundary.boundary_type_maxx == "periodic" and m.boundary.thickness_grid_minx == 1
    wx = np.diff(m.edges[0])
    assert max(wx[0] / wx[-1], wx[-1] / wx[0]) <= 1.25 * (1 + 1e-6)


def test_mesh_arbitrary_box_succeeds_and_quantum_requires_whole_quanta(pdk):
    arb = device(boundary_offset=(0.0005, 0.00037, -0.00019, 0.0, 0.00041, 0.0))
    m = bf.mesh(arb, 1.55, ppw=14, pml=0.3)
    assert m.grid.shape == m.shape
    with pytest.raises(ValueError, match="whole number of 0.001 um quanta"):
        bf.mesh(device(boundary_offset=(0.0005, 0, 0, 0, 0, 0)), 1.55, pml=0.3, quantum=0.001)


def test_mesh_refuses_a_level_placement_drops(pdk):
    lid = LayerLevel(layer=LAYER.SLAB150, zmin=1.05, thickness=0.1, material="si", mesh_order=3)
    d = device(levels={"lid": lid})
    assert "lid" in [lv.name for lv in d.levels]  # within the construction slack, beyond level_margin
    with pytest.raises(ValueError, match="level_margin"):
        bf.mesh(d, 1.55, refine=bf.Refine(z_padding={"lid": (0.1, 0.0)}), pml=0.3)


def test_mesh_reproduces_the_hand_set_bend_grid(pdk):
    """The inverse-design bend: dilation 0.45 um (half-widths up to 1.25 um) and a 0.45 um tail give the fine
    regions benchmarks/rectilinear/graded_bend.py sets by hand (40 nm cells; 80 nm in air and silica) without
    needing a 0.0400001 um quantum-floor dodge."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "guide"))
    import inverse_design as ex

    z_margin = (ex.BOX_HEIGHT - ex.THICKNESS) / 2
    stack = bf.StackDevice(
        ex.tight_bend(ex.UNIFORM), ex.layer_stack, ex.materials, background="air", zrange_layers=["core"],
        extend_ports=(("o1", ex.EXTENSION), ("o2", ex.EXTENSION)),
        boundary_offset=(ex.MARGIN, 0, 0, -ex.MARGIN, z_margin, -z_margin), name="stack",
    )
    lam, n_core, n_box = ex.WAVELENGTH, ex.CORE_INDEX, ex.BOX_INDEX
    ell_air = lam / (2 * math.pi * math.sqrt(n_core**2 - 1.0))
    ell_box = lam / (2 * math.pi * math.sqrt(n_core**2 - n_box**2))
    refine = bf.Refine(dilation={"WG": 0.45}, global_dilation=0.45 / ell_air, global_z_padding=0.45 / ell_box,
                       material_cells={"silica": 0.08})
    m = bf.mesh(stack, lam, ppw=lam / (n_core * 0.04), refine=refine, pml=1.0)
    coarse = 0.08
    x_lo, x_hi = cells_where(m, 0, 0.04)
    assert x_lo == m.edges[0][0] and x_hi == pytest.approx(8.7, abs=coarse)
    y_lo, y_hi = cells_where(m, 1, 0.04)
    assert y_lo == pytest.approx(-1.7, abs=coarse) and y_hi == pytest.approx(m.edges[1][-1], abs=1e-9)
    z_lo, z_hi = cells_where(m, 2, 0.04)
    assert z_lo == pytest.approx(-0.45, abs=coarse)
    assert z_hi == pytest.approx(ex.THICKNESS + 0.45 * ell_air / ell_box, abs=coarse)  # air above: shorter tail
    assert np.diff(m.edges[0]).max() <= coarse + 1e-6


# ---------------------------------------------------------------- the coarse-vertex report


def strip_mesh() -> tuple[bf.StackDevice, bf.Mesh]:
    d = device()
    return d, bf.mesh(d, 1.55, ppw=14, refine=bf.Refine(dilation={"WG": 0.1}), pml=0.3)


def moved(params: dict, key: str, i: int, xy: tuple[float, float]) -> dict:
    v = np.array(params[key])
    v[i] = xy
    return {**params, key: jnp.asarray(v, dtype=params[key].dtype)}


def corner(params: dict) -> int:
    v = np.asarray(params["vertices/WG"])
    return int(np.argmin(v[:, 0] + v[:, 1]))  # the strip's vertex at (-1, -0.25)


def test_coarse_vertices_counts(pdk):
    d, m = strip_mesh()
    p = d.init_params()
    r = d.coarse_vertices(p, m)
    i = list(r.layers).index("vertices/WG")
    assert int(r.total[i]) == 4 and int(np.sum(r.coarse)) == 0 and float(r.worst) <= 1 + 1e-3
    far = d.coarse_vertices(moved(p, "vertices/WG", corner(p), (-0.5, -0.9)), m)  # in the silica's coarse cells
    assert (int(far.x[i]), int(far.y[i]), int(far.coarse[i])) == (0, 1, 1) and float(far.worst) > 1
    gone = d.coarse_vertices(moved(p, "vertices/WG", corner(p), (-0.5, -1.5)), m)
    assert int(gone.outside[i]) == 1 and int(gone.coarse[i]) == 0
    thick = d.coarse_vertices({**p, "thickness/core": jnp.asarray(0.7, dtype=p["thickness/core"].dtype)}, m)
    assert int(thick.z[i]) == 4 and int(thick.coarse[i]) == 4
    assert "vertices/WG" in str(thick) and "largest cell / target" in str(thick)


def test_coarse_vertices_traced_and_compiled_once(pdk):
    d, m = strip_mesh()
    p = d.init_params()
    f = jax.jit(lambda q: d.coarse_vertices(q, m))
    far = moved(p, "vertices/WG", corner(p), (-0.5, -0.9))
    f(p)
    traced, concrete = f(far), d.coarse_vertices(far, m)
    assert traced.layers == concrete.layers
    for a, b in zip(jax.tree_util.tree_leaves(traced), jax.tree_util.tree_leaves(concrete), strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)
    assert f._cache_size() == 1


def test_vertices_at_their_dilation_paint_and_are_not_coarse(pdk):
    d, m = strip_mesh()
    p = d.init_params()
    v = np.asarray(p["vertices/WG"])
    grown = v.copy()
    grown[:, 1] = np.sign(v[:, 1]) * 0.35  # both long edges moved out by the whole dilation
    q = {**p, "vertices/WG": jnp.asarray(grown, dtype=p["vertices/WG"].dtype)}
    assert int(np.sum(d.coarse_vertices(q, m).coarse)) == 0
    config = d.get_simulation_config(time=1e-15, grid=m.grid)
    volume, constraints = d.get_simulation_volume()
    bounds, placing = fdtdx.boundary_objects_from_config(m.boundary, volume)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        [volume, d, *bounds.values()], config, constraints + placing, jax.random.PRNGKey(0)
    )
    params = {**params, **bf.init_stack_params(objects)}
    params["s"] = {**params["s"], "vertices/WG": q["vertices/WG"]}
    arrays, objects, _ = bf.apply_stack(arrays, objects, params, jax.random.PRNGKey(0))
    assert np.all(np.isfinite(np.asarray(arrays.inv_permittivities)))
    placed = objects["s"]
    assert tuple(len(e) - 1 for e in placed.grid_frame.edges) == m.shape
    assert int(np.sum(placed.coarse_vertices(params["s"], m).coarse)) == 0


def test_refine_errors(pdk):
    d = device()
    with pytest.raises(ValueError, match="not a source layer"):
        meshing._geometry(d, 1.55, 14.0, meshing.Refine(dilation={"NOPE": 0.1}))
    with pytest.raises(ValueError, match="one per vertex"):
        meshing._geometry(d, 1.55, 14.0, meshing.Refine(dilation={"WG": np.zeros(3)}))
    with pytest.raises(ValueError, match="not a level"):
        meshing._geometry(d, 1.55, 14.0, meshing.Refine(z_padding={"nope": 0.1}))
    with pytest.raises(ValueError, match="not levels"):
        meshing._geometry(d, 1.55, 14.0, meshing.Refine(no_tail=("nope",)))
    metal = {**MATS, "si": fdtdx.Material(permittivity=2.0, electric_conductivity=1e4)}
    with pytest.raises(ValueError, match="material_cells"):
        meshing._geometry(device(mats=metal), 1.55, 14.0, meshing.Refine())
    geom = meshing._geometry(device(mats=metal), 1.55, 14.0, meshing.Refine(material_cells={"si": 0.01}))
    assert geom.cell["si"] == 0.01


def test_layer_regions_preserves_nested_island_in_hole(pdk):
    def square(h: float) -> np.ndarray:
        return np.array([[-h, -h], [h, -h], [h, h], [-h, h]], dtype=float)

    outer, hole, island = square(0.8), square(0.5), square(0.15)
    recipes = bf.geometry.derive_polygons(
        [(outer, [4]), (hole, [4]), (island, [4])],
        ("or", ("not", ("layer", 0), ("layer", 1)), ("layer", 2)),
        tol=1e-4,
    )
    geom = meshing._geometry(device(), 1.55, 14.0, meshing.Refine(global_dilation=0, global_z_padding=0))
    fake_stack = SimpleNamespace(
        source_layers=[("vertices/a", 0), ("vertices/b", 1), ("vertices/c", 2)],
        source_vertices=[outer, hole, island],
        source_sizes=[(4,), (4,), (4,)],
        source_extensions=(),
        derived_layers=[SimpleNamespace(layers=(0, 1, 2), recipes=recipes)],
    )
    regions, _ = meshing._layer_regions(fake_stack, geom.plane, 0.1)
    combined = regions[-1]
    for j in range(1, len(recipes)):
        combined = combined | regions[-1 - j]
    assert not (combined & geom.plane.region_box((-0.05, -0.05), (0.05, 0.05))).is_empty()  # inner island preserved
    assert (combined & geom.plane.region_box((0.25, -0.05), (0.35, 0.05))).is_empty()  # moat stayed carved out
    assert not (combined & geom.plane.region_box((0.6, -0.05), (0.7, 0.05))).is_empty()  # outer ring present


def test_index_negative_lossy_and_gyrotropic():
    neg = fdtdx.Material(permittivity=-12.0)
    idx_neg = meshing._index(neg, np.array([1.55]))
    assert idx_neg.unsupported and idx_neg.n[0] == 0.0

    # Complex lossy + gyrotropic off-diagonal permittivity (non-real symmetric) uses eigvals and Re(sqrt(eig))
    tensor = np.array([[2.0 + 0.4j, 0.25j, 0.0], [-0.25j, 2.0 + 0.4j, 0.0], [0.0, 0.0, 2.0]])

    class _FakeDispersion:
        def permittivity_tensor(self, omega: float, perm: object) -> np.ndarray:
            return tensor

    gyro = SimpleNamespace(
        permittivity=(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0),
        dispersion=_FakeDispersion(),
        electric_conductivity=0.0,
    )
    idx_gyro = meshing._index(gyro, np.array([1.55]))
    expected = float(np.max(np.real(np.sqrt(np.linalg.eigvals(tensor)))))
    assert not idx_gyro.unsupported
    assert idx_gyro.n[0] == pytest.approx(expected)
    assert idx_gyro.n[0] > math.sqrt(2.0)


def test_mesh_validation_periodic_and_top_roughness(pdk):
    d = device()
    for bad_bc in (
        ("periodic", "pml", "pml", "pml", "pml", "pml"),
        ("pml", "periodic", "pml", "pml", "pml", "pml"),
        ("periodic", "bloch", "pml", "pml", "pml", "pml"),
        ("bloch", "pec", "pml", "pml", "pml", "pml"),
    ):
        with pytest.raises(ValueError, match="periodic/bloch"):
            bf.mesh(d, 1.55, boundaries=bad_bc, pml=0.3)

    m_bloch = bf.mesh(d, 1.55, boundaries=("bloch", "bloch", "pml", "pml", "pml", "pml"), pml=0.3)
    assert m_bloch.pml_cells[:2] == (0, 0) and m_bloch.boundaries[:2] == ("bloch", "bloch")

    for kwargs in (
        {"wavelength": (1.5,)},
        {"wavelength": (1.5, 1.6, 1.7)},
        {"wavelength": (math.nan, 1.6)},
        {"wavelength": 1.55, "ppw": 0.0},
        {"wavelength": 1.55, "ppw": math.nan},
        {"wavelength": 1.55, "pml": -0.1},
        {"wavelength": 1.55, "pml": math.inf},
        {"wavelength": 1.55, "growth": math.nan},
        {"wavelength": 1.55, "min_cell": 0.0},
        {"wavelength": 1.55, "min_cell": math.nan},
        {"wavelength": 1.55, "feature_cells": -0.5},
        {"wavelength": 1.55, "feature_cells": math.nan},
        {"wavelength": 1.55, "max_states": math.nan},
        {"wavelength": 1.55, "refine": bf.Refine(global_dilation=-1.0)},
        {"wavelength": 1.55, "refine": bf.Refine(global_z_padding=math.inf)},
        {"wavelength": 1.55, "refine": bf.Refine(dilation={"WG": math.nan})},
        {"wavelength": 1.55, "refine": bf.Refine(z_padding={"core": (0.1,)})},
        {"wavelength": 1.55, "refine": bf.Refine(z_padding={"core": (0.1, math.nan)})},
        {"wavelength": 1.55, "refine": bf.Refine(material_cells={"si": math.nan})},
        {"wavelength": 1.55, "refine": bf.Refine(regions=[((0.0, 1.0, 0.0), (1.0, 0.0, 1.0), 0.05)])},
        {"wavelength": 1.55, "refine": bf.Refine(regions=[((0.0, 0.0, 0.0), (1.0, 1.0, math.inf), 0.05)])},
    ):
        with pytest.raises(ValueError):
            bf.mesh(d, **kwargs)  # type: ignore[arg-type]

    m_no_feat = bf.mesh(d, 1.55, ppw=14, pml=0.3, feature_cells=0)
    assert m_no_feat.shape[0] > 0

    d_rough = device(roughness=(bf.TopRoughness(level="core", rms=0.008, correlation_length=0.05),))
    geom_rough = meshing._geometry(d_rough, 1.55, 14.0, bf.Refine(global_dilation=0, global_z_padding=0))
    assert geom_rough.padded["core"][1] == pytest.approx(0.22 + 5 * 0.008)
    m_rough = bf.mesh(d_rough, 1.55, ppw=14, pml=0.3)
    assert m_rough.axes[2].centered == pytest.approx((0.0,))


def test_coarse_vertices_empty_layer_boundary_float32_and_summary_outside(pdk):
    d, m = strip_mesh()
    p = d.init_params()
    empty = {**p, "vertices/WG": jnp.zeros((0, 2), dtype=p["vertices/WG"].dtype)}
    rep_empty = d.coarse_vertices(empty, m)
    i = list(rep_empty.layers).index("vertices/WG")
    assert (int(rep_empty.total[i]), int(rep_empty.coarse[i]), int(rep_empty.outside[i])) == (0, 0, 0)
    assert float(rep_empty.worst) >= 0.0

    # Flush non-dyadic boundary 0.3 in float32 (rounds above float64 0.3) stays counted inside
    c_flush = gf.Component()
    for layer in (LAYER.SLAB150, LAYER.WG):
        c_flush.add_polygon([(-0.3, -0.3), (0.3, -0.3), (0.3, 0.3), (-0.3, 0.3)], layer=layer)
    d_flush = bf.StackDevice(
        c_flush,
        LayerStack(layers={
            "core": LayerLevel(layer=LAYER.WG, zmin=0.0, thickness=0.22, material="si", mesh_order=2),
            "frame": LayerLevel(layer=LAYER.SLAB150, zmin=-0.6, thickness=1.6, material="sio2", mesh_order=9),
        }),
        MATS,
        background="sio2",
        name="s",
        zrange_layers=["frame"],
        diagnostics=False,
    )
    m_flush = bf.mesh(d_flush, 1.55, ppw=14, pml=0.2)
    rep_flush = d_flush.coarse_vertices(d_flush.init_params(), m_flush)
    assert int(np.sum(rep_flush.outside)) == 0

    # Summary surfaces report.outside count/warning when starting vertices sit outside the mesh box
    d_clip = device(boundary_offset=(-0.3, 0.0, 0.0, 0.0, 0.0, 0.0))
    m_clip = bf.mesh(d_clip, 1.55, ppw=14, pml=0.3)
    outside_count = int(np.sum(d_clip.coarse_vertices(d_clip.init_params(), m_clip).outside))
    assert outside_count == 4
    summary = m_clip.summary()
    assert f"{outside_count} outside" in summary and "warning:" in summary

    # Placed device checks box origin alignment against mesh edges
    config = d.get_simulation_config(time=1e-15, grid=m.grid)
    volume, constraints = d.get_simulation_volume()
    bounds, placing = fdtdx.boundary_objects_from_config(m.boundary, volume)
    objects, _, params, _, _ = fdtdx.place_objects(
        [volume, d, *bounds.values()], config, constraints + placing, jax.random.PRNGKey(0)
    )
    params = {**params, **bf.init_stack_params(objects)}
    shifted = objects["s"].aset("center", (objects["s"].center[0] + 0.5, objects["s"].center[1]))
    with pytest.raises(ValueError, match="another grid"):
        shifted.coarse_vertices(params["s"], m)
