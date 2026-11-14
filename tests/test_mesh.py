"""Tests for the automatic mesher: the lattice edges of one axis (:func:`bifrost.lattice_edges`) and the
refinements of a device (:mod:`bifrost.meshing`)."""

from __future__ import annotations

import functools
import itertools
import math

import fdtdx
import gdsfactory as gf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gdsfactory.technology import LayerLevel, LayerStack

import bifrost as bf
from bifrost import meshing
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
    return bf.StackDevice.from_layer_stack(
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
        w = widths_q(axis)
        assert w.max() <= math.floor(t * 1e3 + 1e-9) and w.min() >= w.max() - 4  # one size up to the PML rounding


def test_mesh_strip_end_to_end(pdk):
    d = device()
    m = bf.mesh(d, 1.55, ppw=14, refine=bf.Refine(dilation={"WG": 0.1}), pml=0.3)
    t_si = 1.55 / (N_SI * 14)
    ell = 1.55 / (2 * math.pi * math.sqrt(N_SI**2 - N_OX**2))
    # x: the strip runs through every x plane
    assert np.diff(m.edges[0]).max() <= t_si + 1e-6
    # the fine cells reach past the region by half the next cell (its halo) and up to one fine cell of tiling
    lo, hi = cells_where(m, 1, t_si)
    assert lo == pytest.approx(-0.35 - 3 * ell, abs=0.06) and hi == pytest.approx(0.35 + 3 * ell, abs=0.06)
    assert lo <= -0.35 - 3 * ell and hi >= 0.35 + 3 * ell
    lo, hi = cells_where(m, 2, t_si)
    assert lo == pytest.approx(-3 * ell, abs=0.06) and hi == pytest.approx(0.22 + 3 * ell, abs=0.06)
    assert lo <= -3 * ell and hi >= 0.22 + 3 * ell
    assert (m.edges[1][0], m.edges[1][-1], m.edges[2][0], m.edges[2][-1]) == pytest.approx((-1.0, 1.0, -0.6, 1.0))
    for axis in m.axes:
        w = widths_q(axis)
        assert np.all(np.maximum(w[1:] / w[:-1], w[:-1] / w[1:]) <= 1.25 + 1e-9)
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
    for face in (0.0, 0.22):  # the core's faces against silica
        assert np.min(np.abs(mid - face)) <= 0.0005 + 1e-12 and np.min(np.abs(z - face)) > 1e-3
    assert m.axes[2].centered == (0.0, 0.22)
    assert "faces centered at 0, 0.22 um" in m.summary()
    off = bf.mesh(d, 1.55, ppw=14, pml=0.3, center_faces=False)
    assert off.axes[2].centered == ()
    assert m.axes[2].min_cell >= 0.8 * off.axes[2].min_cell - 1e-12


def test_mesh_periodic_faces(pdk):
    m = bf.mesh(device(), 1.55, boundaries=("periodic", "periodic", "pml", "pml", "pml", "pml"), pml=0.3)
    assert m.pml_cells[:2] == (0, 0) and m.boundaries[0] == "periodic"
    assert m.boundary.boundary_type_maxx == "periodic" and m.boundary.thickness_grid_minx == 1


def test_mesh_box_must_be_whole_quanta(pdk):
    with pytest.raises(ValueError, match="whole number of 0.001 um quanta"):
        bf.mesh(device(boundary_offset=(0.0005, 0, 0, 0, 0, 0)), 1.55, pml=0.3)


def test_mesh_refuses_a_level_placement_drops(pdk):
    lid = LayerLevel(layer=LAYER.SLAB150, zmin=1.05, thickness=0.1, material="si", mesh_order=3)
    d = device(levels={"lid": lid})
    assert "lid" in [lv.name for lv in d.levels]  # within the construction slack, beyond level_margin
    with pytest.raises(ValueError, match="level_margin"):
        bf.mesh(d, 1.55, refine=bf.Refine(z_padding={"lid": (0.1, 0.0)}), pml=0.3)


def test_mesh_reproduces_the_hand_set_bend_grid(pdk):
    """The inverse-design bend: dilation 0.45 um (half-widths up to 1.25 um) and a 0.45 um tail give the fine
    regions benchmarks/rectilinear/graded_bend.py sets by hand (40 nm cells; 80 nm in air and silica)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "guide"))
    import inverse_design as ex

    z_margin = (ex.BOX_HEIGHT - ex.THICKNESS) / 2
    stack = bf.StackDevice.from_layer_stack(
        ex.tight_bend(ex.UNIFORM), ex.layer_stack, ex.materials, background="air", zrange_layers=["core"],
        extend_ports=(("o1", ex.EXTENSION), ("o2", ex.EXTENSION)),
        boundary_offset=(ex.MARGIN, 0, 0, -ex.MARGIN, z_margin, -z_margin), name="stack",
    )
    lam, n_core, n_box = ex.WAVELENGTH, ex.CORE_INDEX, ex.BOX_INDEX
    ell_air = lam / (2 * math.pi * math.sqrt(n_core**2 - 1.0))
    ell_box = lam / (2 * math.pi * math.sqrt(n_core**2 - n_box**2))
    refine = bf.Refine(dilation={"WG": 0.45}, global_dilation=0.45 / ell_air, global_z_padding=0.45 / ell_box,
                       material_cells={"silica": 0.08})
    m = bf.mesh(stack, lam, ppw=lam / (n_core * 0.0400001), refine=refine, pml=1.0)
    coarse = 0.08
    x_lo, x_hi = cells_where(m, 0, 0.04)
    assert x_lo == m.edges[0][0] and x_hi == pytest.approx(8.7, abs=coarse)
    y_lo, y_hi = cells_where(m, 1, 0.04)
    assert y_lo == pytest.approx(-1.7, abs=coarse) and y_hi == m.edges[1][-1]
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
