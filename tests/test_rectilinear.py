"""Tests for the StackDevice's grid frame (the groundwork for non-uniform rectilinear grids)."""

from __future__ import annotations

import math
from collections.abc import Iterator

import fdtdx
import gdsfactory as gf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fdtdx.core.grid import RectilinearGrid, UniformGrid
from gdsfactory.technology import LayerLevel, LayerStack

import bifrost as bf
from bifrost.geometry import gaussian_random_field
from bifrost.stack import (
    StackDiagnosticWarning,
    _conductivity_scale,
    _Frame,
    _grid_frame,
    _normal_squared,
    _split_layer,
)

L = gf.gpdk.LAYER
MATERIALS = {"air": fdtdx.Material(), "si": fdtdx.Material(permittivity=12.0), "sio2": fdtdx.Material(permittivity=2.1)}


@pytest.fixture(autouse=True)
def pdk() -> Iterator[None]:
    gf.gpdk.PDK.activate()
    gf.snap.SNAP_ENABLED = False
    yield
    gf.snap.SNAP_ENABLED = True


def grid(x_um, y_um, z_um) -> RectilinearGrid:
    """A rectilinear grid from edges in um."""
    x, y, z = (jnp.asarray(e, dtype=jnp.float32) * 1e-6 for e in (x_um, y_um, z_um))
    return RectilinearGrid(x_edges=x, y_edges=y, z_edges=z)


def config(g) -> fdtdx.SimulationConfig:
    return fdtdx.SimulationConfig(time=1e-15, grid=g, dtype=jnp.float32)


X = np.linspace(-0.7, 0.7, 15)  # 14 cells of 0.1 um
Y = np.linspace(-0.45, 0.45, 10)  # 9 cells of 0.1 um
Z = np.array([0.0, 0.1, 0.15, 0.2, 0.25, 0.35, 0.45, 0.6]) - 0.3  # 7 graded cells


def test_uniform_frame_is_the_cell_frame():
    cfg = config(UniformGrid(spacing=0.02e-6))
    frame = _grid_frame(cfg, ((0, 7), (2, 5), (1, 4)))
    assert frame.h_um == cfg.uniform_spacing() * 1e6
    assert [e.tolist() for e in frame.edges] == [list(range(8)), list(range(4)), list(range(4))]
    assert frame.below == (1.0, 1.0, 1.0)
    assert frame.uniform == (True, True, True)
    # after place_objects the grid is a resolved (uniform) RectilinearGrid
    resolved = config(UniformGrid(spacing=0.02e-6).resolve((10, 10, 10)))
    other = _grid_frame(resolved, ((0, 7), (2, 5), (1, 4)))
    assert other.h_um == resolved.uniform_spacing() * 1e6 == pytest.approx(0.02, rel=1e-12)
    assert [e.tolist() for e in other.edges] == [e.tolist() for e in frame.edges]
    assert other.below == frame.below and other.uniform == frame.uniform


def test_graded_z_frame():
    cfg = config(grid(X, Y, Z))
    frame = _grid_frame(cfg, ((0, 14), (0, 9), (0, 7)))
    assert frame.h_um == pytest.approx(0.1, rel=1e-6)
    assert frame.uniform == (True, True, False)
    assert frame.edges[0].tolist() == list(range(15))
    np.testing.assert_allclose(frame.edges[2], [0, 1, 1.5, 2, 2.5, 3.5, 4.5, 6], rtol=1e-5, atol=1e-5)
    assert frame.below[2] == pytest.approx(1.0, rel=1e-5)  # the first cell, mirrored
    inner = _grid_frame(cfg, ((0, 14), (0, 9), (2, 7)))
    np.testing.assert_allclose(inner.edges[2], [0, 0.5, 1, 2, 3, 4.5], rtol=1e-5, atol=1e-5)
    assert inner.below[2] == pytest.approx(0.5, rel=1e-5)  # the cell under the device


def test_per_axis_uniform_spacings():
    cfg = config(grid(X, np.linspace(-0.45, 0.45, 19), np.linspace(-0.3, 0.3, 13)))  # dy 0.05, dz 0.05
    frame = _grid_frame(cfg, ((0, 14), (0, 18), (0, 12)))
    assert frame.h_um == pytest.approx(0.05, rel=1e-6)  # the smaller lateral width
    assert frame.uniform == (False, True, True)
    np.testing.assert_allclose(frame.edges[0], 2.0 * np.arange(15), rtol=1e-5)


def test_sample_axes():
    frame = _Frame(0.1, (np.arange(4.0), np.arange(3.0), np.array([0.0, 1.0, 1.5, 3.5])), (1.0, 1.0, 2.0),
                   (True, True, False))
    u = frame.sample_axis(0, -0.5)
    assert u.uniform and u.shift == -0.5
    assert u.lo.tolist() == [0, 1, 2] and u.hi.tolist() == [1, 2, 3]
    primal = frame.sample_axis(2, 0.0)
    assert not primal.uniform and primal.shift == 0.0
    assert primal.lo.tolist() == [0, 1, 1.5] and primal.hi.tolist() == [1, 1.5, 3.5]
    dual = frame.sample_axis(2, -0.5)  # [C_{k-1}, C_k], the cell below the box 2 wide
    assert dual.lo.tolist() == [-1.0, 0.5, 1.25] and dual.hi.tolist() == [0.5, 1.25, 2.5]
    with pytest.raises(ValueError):
        frame.sample_axis(2, -0.25)


def test_split_layer_top():
    key = jax.random.PRNGKey(3)
    lows = jax.random.uniform(key, (5, 4, 3), minval=-1.0, maxval=3.0)
    highs = lows + jax.random.uniform(jax.random.fold_in(key, 1), (5, 4, 3), maxval=2.0)
    k = jnp.asarray(1.0)
    for a, b in zip(_split_layer(lows, highs, k), _split_layer(lows, highs, k, k + 1.0), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    _, bp, length, present = _split_layer(lows, highs, jnp.asarray(0.3), jnp.asarray(1.7))
    np.testing.assert_allclose(np.asarray(length.sum(-1)), 1.4, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(bp[..., 0]), 0.3)
    np.testing.assert_allclose(np.asarray(bp[..., -1]), 1.7)
    # each level covers its overlap with the layer
    covered = np.sum(np.asarray(length)[..., None] * np.asarray(present), axis=-2)
    expect = np.clip(np.minimum(highs, 1.7) - np.maximum(lows, 0.3), 0.0, None)
    np.testing.assert_allclose(covered, expect, atol=1e-6)


def test_conductivity_scale():
    uniform = config(UniformGrid(spacing=0.02e-6))
    assert _conductivity_scale(uniform) == uniform.uniform_spacing()
    graded = config(grid(X, Y, Z))
    expect = fdtdx.constants.c * graded.time_step_duration / graded.courant_number
    assert _conductivity_scale(graded) == expect


def device(**kwargs) -> bf.StackDevice:
    c = gf.Component()
    c.add_polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)], layer=L.WG)
    c.add_polygon([(-0.2, -0.2), (1.2, -0.2), (1.2, 0.7), (-0.2, 0.7)], layer=L.WAFER)
    ls = LayerStack(layers={
        "core": LayerLevel(layer=L.WG, zmin=0.0, thickness=0.22, material="si", mesh_order=1),
        "under": LayerLevel(layer=L.WAFER, zmin=-1.0, thickness=0.85, material="sio2", mesh_order=9),
    })
    # the box: 1.6 x 1.2 x 0.8 um (x -0.2 to 1.4, y -0.2 to 1.0, z 0 to 0.8), whole 0.02 and 0.4 um cells
    return bf.StackDevice.from_layer_stack(
        c, ls, MATERIALS, background="air", name="s", zrange_layers=["core"],
        boundary_offset=(0.4, -0.2, 0.5, -0.2, 0.58, 0.0), diagnostics=False, **kwargs,
    )


def place(d: bf.StackDevice, g) -> dict:
    volume, constraints = d.get_simulation_volume()
    objects, *_ = fdtdx.place_objects(
        object_list=[volume, d], config=config(g), constraints=constraints, key=jax.random.PRNGKey(0)
    )
    return objects


def test_placed_device_has_its_frame():
    d = device()
    ((name, lo, hi),) = d.dropped_levels  # 0.15 um under the box: beyond the slack
    assert name == "under" and (lo, hi) == pytest.approx((-1.0, -0.15))
    placed = place(d, UniformGrid(spacing=0.02e-6))["s"]
    assert placed.grid_frame is not None and placed.grid_frame.uniform == (True, True, True)
    assert placed._res_um() == placed._config.uniform_spacing() * 1e6
    assert tuple(len(e) - 1 for e in placed.grid_frame.edges) == tuple(placed.grid_shape)


def test_coarse_grid_warns_about_a_dropped_level():
    d = device()
    with pytest.warns(StackDiagnosticWarning, match="'under'.*left out"):
        place(d, UniformGrid(spacing=0.4e-6))  # half a 0.4 um cell under the box reaches it


# ---------------------------------------------------------------- graded z

H = 0.05  # um, the lateral cells
SLAB = (0.0, 0.13)  # um, the slab level's span
Z_EDGES = np.array([0.0, 0.1, 0.15, 0.18, 0.2, 0.25, 0.32, 0.36, 0.42, 0.5])  # um from the box's bottom


def slab_device(angle: float = 0.0, width: float | None = None, **kwargs) -> bf.StackDevice:
    """A silicon slab (0 to 0.13 um) in air; a rectangle past the box, or a ``width`` um wide strip.

    The box is 1.6 x 1.1 um (whole 0.05 um cells) and 0.5 um high, from -0.2 um.
    """
    c = gf.Component()
    c.add_polygon([(-0.5, -0.5), (1.5, -0.5), (1.5, 1.0), (-0.5, 1.0)], layer=L.SLAB150)  # sets the box (WAFER would be a fill: no box)
    if width is None:
        c.add_polygon([(-0.5, -0.5), (1.5, -0.5), (1.5, 1.0), (-0.5, 1.0)], layer=L.WG)
    else:
        c.add_polygon([(0.3, 0.25 - width / 2), (1.2, 0.25 - width / 2), (1.2, 0.25 + width / 2),
                       (0.3, 0.25 + width / 2)], layer=L.WG)
        c.add_port(name="o1", center=(0.3, 0.25), width=width, orientation=180, layer=L.WG)
    ls = LayerStack(layers={
        "slab": LayerLevel(layer=L.WG, zmin=SLAB[0], thickness=SLAB[1] - SLAB[0], material="si", mesh_order=1,
                           sidewall_angle=angle),
        # air over the whole box: sets the box, paints nothing
        "frame": LayerLevel(layer=L.SLAB150, zmin=-0.2, thickness=0.5, material="air", mesh_order=9),
    })
    return bf.StackDevice.from_layer_stack(
        c, ls, MATERIALS, background="air", name="s", zrange_layers=["slab"],
        boundary_offset=(-0.2, 0.2, -0.2, 0.2, 0.17, -0.2), diagnostics=False, **kwargs,
    )


def z_graded(d: bf.StackDevice, z_um: np.ndarray) -> RectilinearGrid:
    """x, y uniform (H) over the device's box, z edges ``z_um`` from the box's bottom."""
    sx, sy, sz = (s * 1e6 for s in d.partial_real_shape)
    assert abs(z_um[-1] - sz) < 1e-9
    nx, ny = round(sx / H), round(sy / H)
    return grid(np.linspace(-sx / 2, sx / 2, nx + 1), np.linspace(-sy / 2, sy / 2, ny + 1), z_um - sz / 2)


def overlap(a: tuple[float, float], lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Fraction of each interval [lo, hi] inside a."""
    return np.clip(np.minimum(hi, a[1]) - np.maximum(lo, a[0]), 0.0, None) / (hi - lo)


def test_graded_z_fractions_are_exact():
    d = slab_device()
    placed = place(d, z_graded(d, Z_EDGES))["s"]
    assert placed.grid_frame.uniform == (True, True, False)
    names, w = placed.material_weights()
    si = np.asarray(w[names.index("si")])
    bottom = -0.2  # the box's bottom (stack z)
    e = Z_EDGES + bottom
    np.testing.assert_allclose(si[16, 11], overlap(SLAB, e[:-1], e[1:]), atol=2e-6)
    # Ex, Ey: the dual cells between the cell centers (the first one mirrored below the box)
    c = 0.5 * (e[:-1] + e[1:])
    lo = np.concatenate([[e[0] - 0.5 * (e[1] - e[0])], c[:-1]])
    for shift in ((0.0, -0.5, -0.5), (-0.5, 0.0, -0.5)):
        names, w = placed.material_weights(shift=shift)
        np.testing.assert_allclose(np.asarray(w[names.index("si")])[16, 11], overlap(SLAB, lo, c), atol=2e-6)


def test_graded_z_split_cell_keeps_the_volumes():
    """A uniform z grid and the same grid with one cell split in two: the halves add up to the cell.

    Exact (to float32 round-off) for flat walls. Sloped walls are painted at each level's mean offset over its
    part of the voxel (half-planes, see _composite: off the mean area by up to ~5e-3 where the edge crosses a
    corner of the square inside the voxel) or at two Gauss heights (the corner model: exact where the corner's
    area is quadratic in z, which near a corner of sloped walls it is only piecewise); either way splitting the
    cell moves where it is painted, and the split cell changes by up to a few 1e-3 there.
    """
    cases = (  # (device, tolerance of the split cell)
        ({"angle": 0.0, "width": 0.4}, 2e-5),
        ({"angle": 25.0, "width": 0.4, "corner_lines": 0}, 5e-3),  # half-planes only
        ({"angle": 25.0, "width": 0.4}, 1e-2),  # sloped walls with the corner model
        ({"angle": 25.0, "width": 0.4, "xy_subsamples": 2}, 1e-2),
    )
    for kwargs, tol in cases:
        d = slab_device(**kwargs)
        uniform = np.linspace(0.0, 0.5, 11)  # 0.05 um cells: frame-uniform
        # the slab's top (0.33 um above the box's bottom) cuts cell 6, which is split in two
        split = np.insert(uniform, 7, 0.5 * (uniform[6] + uniform[7]))
        a = place(d, z_graded(d, uniform))["s"]
        b = place(d, z_graded(d, split))["s"]
        assert a.grid_frame.uniform[2] and not b.grid_frame.uniform[2]
        _, wa = a.material_weights()
        _, wb = b.material_weights()
        wa, wb = np.asarray(wa), np.asarray(wb)
        merged = 0.5 * (wb[..., 6] + wb[..., 7])
        np.testing.assert_allclose(merged, wa[..., 6], atol=tol, err_msg=str(kwargs))
        # the other cells are the same cells, up to float32 round-off (the graded path splits the layers in frame
        # units, up to 10 here, the uniform one in unit layers: ~6e-6 where an edge is on a voxel face)
        np.testing.assert_allclose(wb[..., :6], wa[..., :6], atol=2e-5, err_msg=str(kwargs))
        np.testing.assert_allclose(wb[..., 8:], wa[..., 7:], atol=2e-5, err_msg=str(kwargs))


def test_graded_z_averaging_takes_the_horizontal_interface():
    """Across a horizontal interface Ex, Ey see the mean eps and Ez the harmonic mean, on graded cells."""
    d = slab_device()
    objects, arrays, params = place_all(d, z_graded(d, Z_EDGES))
    out, _, _ = bf.apply_stack(arrays, objects, params, jax.random.PRNGKey(0))
    inv = np.asarray(out.inv_permittivities[:, 16, 11, :])  # (3, nz)
    e = Z_EDGES - 0.2
    c = 0.5 * (e[:-1] + e[1:])
    dual = (np.concatenate([[e[0] - 0.5 * (e[1] - e[0])], c[:-1]]), c)
    f_dual, f_primal = overlap(SLAB, *dual), overlap(SLAB, e[:-1], e[1:])
    np.testing.assert_allclose(inv[0], 1.0 / (12.0 * f_dual + (1.0 - f_dual)), rtol=2e-5)
    np.testing.assert_allclose(inv[2], f_primal / 12.0 + (1.0 - f_primal), rtol=2e-5)


def test_graded_z_gradient():
    d = slab_device(angle=20.0, width=0.4)
    placed = place(d, z_graded(d, Z_EDGES))["s"]
    p0 = placed.init_params()

    def volume(t: jax.Array) -> jax.Array:
        names, w = placed.material_weights({**p0, "thickness/slab": t})
        return jnp.sum(w[names.index("si")] * jnp.arange(w.shape[-1], dtype=w.dtype))  # z-weighted

    t0 = jnp.asarray(0.13, dtype=jnp.float32)
    g = float(jax.grad(volume)(t0))
    step = 2e-3
    fd = (float(volume(t0 + step)) - float(volume(t0 - step))) / (2 * step)
    assert np.isfinite(g) and g == pytest.approx(fd, rel=2e-2)


def test_graded_z_traces_once():
    """The grid is static: new parameters reuse the compiled program."""
    d = slab_device(angle=20.0, width=0.4)
    placed = place(d, z_graded(d, Z_EDGES))["s"]
    p0 = placed.init_params()
    traces = []

    def weights(t: jax.Array) -> jax.Array:
        traces.append(t)
        return placed.material_weights({**p0, "thickness/slab": t})[1]

    f = jax.jit(weights)
    for t in (0.13, 0.14, 0.11):
        f(jnp.asarray(t, dtype=jnp.float32)).block_until_ready()
    assert len(traces) == 1


def test_graded_z_full_averaging_takes_the_horizontal_interface():
    """averaging="subpixel_full": the laminate along z on graded cells, no off-diagonal entries."""
    d = slab_device(averaging="subpixel_full")
    objects, arrays, params = place_all(d, z_graded(d, Z_EDGES))
    out, _, _ = bf.apply_stack(arrays, objects, params, jax.random.PRNGKey(0))
    inv = np.asarray(out.inv_permittivities[:, 16, 11, :])  # (9, nz): xx, xy, xz, yx, yy, yz, zx, zy, zz
    e = Z_EDGES - 0.2
    c = 0.5 * (e[:-1] + e[1:])
    f_dual = overlap(SLAB, np.concatenate([[e[0] - 0.5 * (e[1] - e[0])], c[:-1]]), c)
    f_primal = overlap(SLAB, e[:-1], e[1:])
    np.testing.assert_allclose(inv[0], 1.0 / (12.0 * f_dual + (1.0 - f_dual)), rtol=2e-5)
    np.testing.assert_allclose(inv[8], f_primal / 12.0 + (1.0 - f_primal), rtol=2e-5)
    np.testing.assert_allclose(inv[[1, 2, 5]], 0.0, atol=1e-6)


def test_graded_z_full_averaging_junctions_are_bounded():
    """Junction voxels (the corners of sloped walls) on graded cells: their closure, a box per voxel, solves
    and stays between the materials (each row of the stored inverse comes from its own E component's Yee
    position, so the 3 x 3 is not symmetric pointwise)."""
    d = slab_device(angle=25.0, width=0.4, averaging="subpixel_full")
    objects, arrays, params = place_all(d, z_graded(d, Z_EDGES))
    placed = objects["s"]
    out, status = placed.write_with_status(arrays, params["s"])
    assert int(status["junctions"]) > 0 and int(status["junction_overflow"]) == 0
    assert float(status["closure_residual"]) < 1e-3
    inv = np.asarray(out.inv_permittivities[:, *placed.grid_slice]).reshape(3, 3, -1)
    assert np.all(np.isfinite(inv))
    diagonal = inv[[0, 1, 2], [0, 1, 2]]
    assert diagonal.min() > 1.0 / 12.0 - 1e-5 and diagonal.max() < 1.0 + 1e-5
    off = inv[[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]
    assert np.abs(off).max() < 0.5 * (1.0 - 1.0 / 12.0)
    assert np.abs(off).max() > 1e-4  # the sloped corners do couple the components


def test_rectilinear_grid_spans_the_box():
    d = slab_device()
    g = d.rectilinear_grid(H, z_um=Z_EDGES - 0.2)  # stack heights of the edges
    ref = z_graded(d, Z_EDGES)
    for a in range(3):
        np.testing.assert_allclose(np.asarray(g.edges(a)), np.asarray(ref.edges(a)), atol=1e-13)
    assert place(d, g)["s"].grid_frame.uniform == (True, True, False)
    with pytest.raises(ValueError, match="box's face"):
        d.rectilinear_grid(H, z_um=Z_EDGES)  # not stack heights
    with pytest.raises(ValueError, match="whole number"):
        d.rectilinear_grid(0.07)


def test_graded_edges():
    e = bf.graded_edges(-1.0, 2.0, 0.1, fine=[(0.0, 0.22, 0.01)], growth=1.2, snap=[0.0, 0.22, 0.5],
                        uniform_ends=(5, 3))
    w = np.diff(e)
    assert e[0] == -1.0 and e[-1] == 2.0 and np.all(w > 0)
    for z in (0.0, 0.22, 0.5):
        assert np.min(np.abs(e - z)) < 1e-12
    np.testing.assert_allclose(w[:5], 0.1, rtol=1e-12)  # the uniform ends
    np.testing.assert_allclose(w[-3:], 0.1, rtol=1e-12)
    assert np.max(w) <= 0.1 * (1 + 1e-9)
    inside = (e[:-1] >= 0.0 - 1e-12) & (e[1:] <= 0.22 + 1e-12)
    assert np.max(w[inside]) <= 0.01 * (1 + 1e-6)
    assert np.max(np.maximum(w[1:] / w[:-1], w[:-1] / w[1:])) < 1.2 * 1.15  # graded
    with pytest.warns(UserWarning, match="not edges"):  # a sliver of a cell is not made
        e2 = bf.graded_edges(0.0, 1.0, 0.1, snap=[0.5, 0.501])
    assert np.min(np.diff(e2)) > 0.04
    with pytest.raises(ValueError):
        bf.graded_edges(0.0, 1.0, 0.1, fine=[(0.0, 0.2, 0.01)], uniform_ends=(1, 0))  # under the uniform end


def test_port_planes_take_whole_graded_cells():
    d = slab_device(width=0.4)
    g = d.rectilinear_grid(H, z_um=Z_EDGES - 0.2)
    det, constraints = d.create_field_detector_at("o1", (0.0, 0.0), port_offset=-0.1, grid=g)
    assert det.partial_real_shape[2] is None and det.partial_grid_shape[2] == 2  # z: graded, whole cells
    assert det.partial_real_shape[1] == pytest.approx(0.4e-6)  # y: uniform, exact
    volume, placing = d.get_simulation_volume()
    objects, *_ = fdtdx.place_objects(
        object_list=[volume, d, det], config=config(g), constraints=placing + constraints, key=jax.random.PRNGKey(0)
    )
    k0, k1 = objects["o1_field_detector"].grid_slice_tuple[2]
    e = Z_EDGES - 0.2  # stack heights
    assert (e[k0], e[k1]) == pytest.approx((0.0, 0.12))  # the slab, 0 to 0.13 um, on the nearest edges


def test_graded_grids_follow_the_device_dtype():
    """fdtdx scales the fields by the cell widths in the edges' dtype: a float32 run on a graded grid needs float32
    edges, also under JAX's 64-bit mode. rectilinear_grid and get_simulation_config give them; a config built by
    hand with float64 edges is refused at placement instead of failing in the time loop."""
    assert jax.config.jax_enable_x64  # importing bifrost turns it on
    d = slab_device(width=0.4)
    g = d.rectilinear_grid(H, z_um=Z_EDGES - 0.2)
    assert all(g.edges(a).dtype == jnp.float32 for a in range(3))
    g64 = RectilinearGrid(**{f"{n}_edges": jnp.asarray(np.asarray(g.edges(a)), dtype=jnp.float64)
                             for a, n in enumerate("xyz")})
    cfg = d.get_simulation_config(time=3e-15, grid=g64)
    assert all(cfg.grid.edges(a).dtype == jnp.float32 for a in range(3))
    volume, placing = d.get_simulation_volume()
    with pytest.raises(ValueError, match="grid's edges"):
        fdtdx.place_objects(object_list=[volume, d], config=fdtdx.SimulationConfig(time=3e-15, grid=g64,
                            dtype=jnp.float32), constraints=placing, key=jax.random.PRNGKey(0))
    wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
    dipole, at = d.create_point_dipole_source_at(
        "o1", wave_character=wave, polarization=1, port_offset=0.3,
        temporal_profile=fdtdx.GaussianPulseProfile(center_wave=wave, spectral_width=fdtdx.WaveCharacter(
            frequency=0.5 * wave.get_frequency())),
    )
    objects, arrays, params, cfg, _ = fdtdx.place_objects(
        object_list=[volume, d, dipole], config=cfg, constraints=placing + at, key=jax.random.PRNGKey(0)
    )
    arrays, objects, _ = bf.apply_stack(arrays, objects, {**params, **bf.init_stack_params(objects)},
                                        jax.random.PRNGKey(0))
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=cfg, key=jax.random.PRNGKey(0),
                               show_progress=False)
    assert arrays.fields.E.dtype == jnp.float32 and bool(jnp.all(jnp.isfinite(arrays.fields.E)))
    assert float(jnp.max(jnp.abs(arrays.fields.E))) > 0.0


def test_port_planes_snap_on_an_asymmetric_box():
    """A box far from symmetric about the port (boundary_offset 0.55 / 0.45 um in y around the port, 0.2 / 0.37 um
    in z around the slab), graded along y and z: the plane's faces land on the edges nearest the port's own
    extent; a grid that does not span the box (a larger volume) is refused."""
    width = 0.4
    c = gf.Component()
    c.add_polygon([(-0.5, -0.5), (1.5, -0.5), (1.5, 1.0), (-0.5, 1.0)], layer=L.SLAB150)
    c.add_polygon([(0.3, 0.05), (1.2, 0.05), (1.2, 0.45), (0.3, 0.45)], layer=L.WG)
    c.add_port(name="o1", center=(0.3, 0.25), width=width, orientation=180, layer=L.WG)
    ls = LayerStack(layers={
        "slab": LayerLevel(layer=L.WG, zmin=SLAB[0], thickness=SLAB[1] - SLAB[0], material="si", mesh_order=1),
        "frame": LayerLevel(layer=L.SLAB150, zmin=-0.2, thickness=0.7, material="air", mesh_order=9),
    })
    # box: x [-0.3, 1.3], y [-0.3, 0.7] (port y 0.05 to 0.45), z [-0.2, 0.5] (slab 0 to 0.13)
    d = bf.StackDevice.from_layer_stack(
        c, ls, MATERIALS, background="air", name="s", zrange_layers=["slab"],
        boundary_offset=(-0.2, 0.2, -0.3, 0.2, 0.37, -0.2), diagnostics=False, corner_lines=0,
    )
    assert d.center[1] == pytest.approx(0.2) and d.z_center == pytest.approx(0.15)  # neither the port's nor the slab's
    ey = bf.graded_edges(-0.3, 0.7, 0.05, fine=[(0.0, 0.5, 0.0175)], growth=1.25)
    ez = bf.graded_edges(-0.2, 0.5, 0.05, fine=[(-0.03, 0.16, 0.02)], growth=1.25)
    g = d.rectilinear_grid(0.05, y_um=ey, z_um=ez)
    det, constraints = d.create_field_detector_at("o1", (0.0, 0.0), port_offset=-0.1, grid=g)
    assert det.partial_real_shape[1:] == (None, None)  # y and z: graded, whole cells
    volume, placing = d.get_simulation_volume()
    objects, *_ = fdtdx.place_objects(
        object_list=[volume, d, det], config=config(g), constraints=placing + constraints, key=jax.random.PRNGKey(0)
    )
    (j0, j1), (k0, k1) = objects["o1_field_detector"].grid_slice_tuple[1:]

    def nearest(e: np.ndarray, x: float) -> float:
        return float(e[np.argmin(np.abs(e - x))])

    assert (ey[j0], ey[j1]) == pytest.approx((nearest(ey, 0.05), nearest(ey, 0.45)), abs=1e-6)
    assert (ez[k0], ez[k1]) == pytest.approx((nearest(ez, 0.0), nearest(ez, 0.13)), abs=1e-6)
    wider = grid(np.asarray(g.edges(0)) * 1e6, np.concatenate([np.asarray(g.edges(1)) * 1e6, [0.6]]),
                 np.asarray(g.edges(2)) * 1e6)  # y 0.1 um past the box: a larger volume
    with pytest.raises(ValueError, match="own simulation volume"):
        d.create_field_detector_at("o1", (0.0, 0.0), grid=wider)


def place_all(d: bf.StackDevice, g) -> tuple:
    volume, constraints = d.get_simulation_volume()
    objects, arrays, params, _, _ = fdtdx.place_objects(
        object_list=[volume, d], config=config(g), constraints=constraints, key=jax.random.PRNGKey(0)
    )
    return objects, arrays, {**params, **bf.init_stack_params(objects)}


# ---------------------------------------------------------------- x and y cells of different sizes

BOX_XY = (-0.2, 1.4, -0.2, 1.1)  # um: 1.6 x 1.3, whole cells of 0.02 and 0.04 um (also transposed)
CORE_LAYER = 10  # the voxel layer 0.10 to 0.12 um, inside the core (0 to 0.22 um) of lateral_device


def rotated(points: list, degrees: float, about: tuple[float, float]) -> np.ndarray:
    a = math.radians(degrees)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = np.asarray(about, dtype=float)
    return (np.asarray(points, dtype=float) - c) @ r.T + c


# non-convex, 17 deg off the grid, inside BOX_XY
L_SHAPE = rotated(
    [(0.0, 0.0), (1.0, 0.0), (1.0, 0.6), (0.5, 0.6), (0.5, 0.3), (0.2, 0.3), (0.2, 0.75), (0.0, 0.75)], 17.0,
    (0.5, 0.4),
)


def lateral_device(polygon, angle: float = 0.0, transpose: bool = False, shrink=(0.0, 0.0), **kwargs) -> bf.StackDevice:
    """A silicon core (0 to 0.22 um) of ``polygon`` in air; the box BOX_XY (x and y swapped with ``transpose``)
    and -0.1 to 0.32 um. ``shrink`` (um per x, y face) takes the box back in from polygons that run past it."""
    x0, x1, y0, y1 = BOX_XY
    p = np.asarray(polygon, dtype=float)
    if transpose:  # x and y swapped, still counter-clockwise
        x0, x1, y0, y1 = y0, y1, x0, x1
        p = p[::-1, ::-1]
    c = gf.Component()
    c.add_polygon([tuple(q) for q in p], layer=L.WG)
    c.add_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer=L.SLAB150)  # the box
    ls = LayerStack(layers={
        "core": LayerLevel(layer=L.WG, zmin=0.0, thickness=0.22, material="si", mesh_order=1, sidewall_angle=angle),
        "frame": LayerLevel(layer=L.SLAB150, zmin=-0.1, thickness=0.42, material="air", mesh_order=9),
    })
    sx, sy = shrink
    return bf.StackDevice.from_layer_stack(
        c, ls, MATERIALS, background="air", name="s", zrange_layers=["core"],
        boundary_offset=(-sx, sx, -sy, sy, 0.1, -0.1), diagnostics=False, **kwargs,
    )


def lateral_grid(d: bf.StackDevice, dx: float, dy: float, dz: float = 0.02) -> RectilinearGrid:
    """Cells of dx x dy x dz um over the device's box."""
    sx, sy, sz = (s * 1e6 for s in d.partial_real_shape)
    return grid(*(np.linspace(-s / 2, s / 2, round(s / w) + 1) for s, w in ((sx, dx), (sy, dy), (sz, dz))))


def exact_fractions(polygon, dx: float, dy: float) -> np.ndarray:
    """(nx, ny) fraction of every dx x dy cell of BOX_XY inside the polygon."""
    shapely = pytest.importorskip("shapely")
    x0, x1, y0, y1 = BOX_XY
    nx, ny = round((x1 - x0) / dx), round((y1 - y0) / dy)
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cells = shapely.box(x0 + i * dx, y0 + j * dy, x0 + (i + 1) * dx, y0 + (j + 1) * dy)
    return shapely.area(shapely.intersection(shapely.Polygon(polygon), cells)) / (dx * dy)


def test_lateral_frame_widths():
    cfg = config(grid(np.linspace(-0.8, 0.8, 41), np.linspace(-0.65, 0.65, 66), np.linspace(-0.21, 0.21, 22)))
    frame = _grid_frame(cfg, ((0, 40), (0, 65), (0, 21)))
    assert frame.h_um == 0.02  # rounded like fdtdx's uniform spacing: no float32 round-off of the edges
    assert frame.widths[0] == 2.0 and frame.widths[1] == 1.0
    assert frame.uniform == (False, True, True)
    np.testing.assert_allclose(frame.edges[0], frame.widths[0] * np.arange(41), rtol=1e-12)


@pytest.mark.parametrize("subsamples", [1, 2])
def test_rectangular_cells_get_exact_areas(subsamples: int):
    """dx = 2 dy: every voxel of a rotated non-convex polygon, at its corners too, gets its exact area."""
    d = lateral_device(L_SHAPE, xy_subsamples=subsamples)
    placed = place(d, lateral_grid(d, 0.04, 0.02))["s"]
    assert placed.grid_frame.widths[:2] == (2.0, 1.0)
    names, w = placed.material_weights()
    exact = exact_fractions(L_SHAPE, 0.04, 0.02)
    assert ((exact > 0) & (exact < 1)).sum() > 100
    tol = 1e-4 if subsamples == 1 else 1e-3  # sub-samples blend the corner model by the voxel's certificate
    np.testing.assert_allclose(np.asarray(w[names.index("si")])[:, :, CORE_LAYER], exact, atol=tol)


@pytest.mark.parametrize(("along", "angle"), [("x", 0.0), ("x", 20.0), ("y", 0.0), ("y", 20.0)])
def test_rectangular_cells_add_up_to_their_squares(along: str, angle: float):
    """dx = 2 dy against dx = dy on a strip that runs through the box (no corner inside): along x the two grids
    share the y cells, so every voxel is the same, sloped walls too; along y a rectangle is its two squares
    (with sloped walls only summed along the edge, as the half-planes paint each voxel at its wall's mean)."""
    x0, x1, y0, y1 = BOX_XY
    if along == "x":
        strip, shrink = [(x0 - 0.5, 0.13), (x1 + 0.5, 0.13), (x1 + 0.5, 0.42), (x0 - 0.5, 0.42)], (0.5, 0.0)
    else:
        strip, shrink = [(0.33, y0 - 0.5), (0.87, y0 - 0.5), (0.87, y1 + 0.5), (0.33, y1 + 0.5)], (0.0, 0.5)
    d = lateral_device(strip, angle=angle, shrink=shrink)
    _, wa = place(d, lateral_grid(d, 0.04, 0.02))["s"].material_weights()
    _, wb = place(d, lateral_grid(d, 0.02, 0.02))["s"].material_weights()
    wa, wb = np.asarray(wa), np.asarray(wb)
    pairs = 0.5 * (wb[:, 0::2] + wb[:, 1::2])
    if along == "x" or angle == 0.0:
        np.testing.assert_allclose(wa, pairs, atol=2e-5)
    else:
        np.testing.assert_allclose(wa.sum(axis=1), pairs.sum(axis=1), atol=1e-4)


def test_swapping_x_and_y_transposes_the_weights():
    d = lateral_device(L_SHAPE, angle=15.0)
    _, w = place(d, lateral_grid(d, 0.04, 0.02))["s"].material_weights()
    t = lateral_device(L_SHAPE, angle=15.0, transpose=True)
    _, wt = place(t, lateral_grid(t, 0.02, 0.04))["s"].material_weights()
    np.testing.assert_allclose(np.asarray(wt), np.asarray(w).transpose(0, 2, 1, 3), atol=1e-4)


def test_rectangular_cells_full_averaging_is_bounded():
    """averaging="subpixel_full" on dx = 2 dy: the closure of the junction voxels on their boxes solves and stays
    between the materials."""
    d = lateral_device(L_SHAPE, angle=15.0, averaging="subpixel_full")
    objects, arrays, params = place_all(d, lateral_grid(d, 0.04, 0.02))
    placed = objects["s"]
    out, status = placed.write_with_status(arrays, params["s"])
    assert int(status["junctions"]) > 0 and int(status["junction_overflow"]) == 0
    assert float(status["closure_residual"]) < 1e-3
    inv = np.asarray(out.inv_permittivities[:, *placed.grid_slice]).reshape(3, 3, -1)
    assert np.all(np.isfinite(inv))
    diagonal = inv[[0, 1, 2], [0, 1, 2]]
    assert diagonal.min() > 1.0 / 12.0 - 1e-5 and diagonal.max() < 1.0 + 1e-5
    assert np.abs(inv[[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]).max() < 0.5 * (1.0 - 1.0 / 12.0)


def test_normal_squared_uses_the_cell_sizes():
    """A 45 degree planar field on cells twice as wide along x: the normal is at 45 degrees with the distances."""
    i, j = np.meshgrid(np.arange(6), np.arange(8), indexing="ij")
    q = jnp.asarray(np.broadcast_to(((i + 0.5) * 2.0 + (j + 0.5))[:, :, None], (6, 8, 3)))
    n2 = np.asarray(_normal_squared(q, (np.full(7, 2.0), np.full(9, 1.0), None)))
    np.testing.assert_allclose(n2[:2, 1:-1, 1:-1], 0.5, atol=1e-6)
    np.testing.assert_allclose(n2[2], 0.0, atol=1e-6)
    assert float(np.asarray(_normal_squared(q))[0, 2, 3, 1]) == pytest.approx(0.8)  # undivided: biased


def test_viewer_adopts_and_draws_a_graded_result():
    """A result on a graded z grid: its detector is adopted at its physical extent, the result is accepted for
    its stack, and an xz plane's field reaches the browser on an even z grid at the finest cell."""
    from types import SimpleNamespace

    pytest.importorskip("anywidget")
    d = slab_device(width=0.4)
    g = d.rectilinear_grid(H, z_um=Z_EDGES - 0.2)
    det = fdtdx.PhasorDetector(
        name="xz", wave_characters=(fdtdx.WaveCharacter(wavelength=1.55e-6),), components=("Ex",),
        partial_grid_shape=(None, 1, None),
    )
    volume, placing = d.get_simulation_volume()
    objects, _, _, cfg, _ = fdtdx.place_objects(
        object_list=[volume, d, det], config=config(g),
        constraints=[*placing, det.same_size(volume, axes=(0, 2)), det.place_at_center(volume)],
        key=jax.random.PRNGKey(0),
    )
    state = bf.viewer.state.ViewerState(d)
    specs = {s.params["name"]: s for s in bf.viewer.adopt.adopt_result(state, objects, cfg)}
    glyph = specs["xz"].glyph
    assert glyph is not None and glyph["axis"] == 1
    assert glyph["v_half"] == pytest.approx(0.25, abs=1e-6)  # z: the whole 0.5 um box, from the graded edges
    # a field linear in z, at the Ex samples (cell centers along z)
    nx, _, nz = objects["xz"].grid_shape
    zc = 0.5 * (Z_EDGES[:-1] + Z_EDGES[1:])  # um from the box's bottom
    phasor = np.broadcast_to(zc[None, None, :], (nx, 1, nz)).astype(np.complex64)[None, None, None]
    result = SimpleNamespace(
        objects=objects, arrays=SimpleNamespace(detector_states={"xz": {"phasor": phasor}}), config=cfg
    )
    bf.viewer.fields.validate_foreign_result(objects, result.arrays, cfg, d)
    entry = bf.viewer.fields.field_payload(result, "xz", "id")
    nv, nu = entry["grid"]  # (v, u) = (z, x)
    assert nu == nx and nv == 25  # z on an even grid at its finest cell, 0.02 um
    real = np.frombuffer(entry["real"], dtype=np.float32).reshape(1, 1, nv, nu)
    even = (np.arange(nv) + 0.5) * (0.5 / nv)  # the even grid's sample heights (um from the bottom)
    inside = (even >= zc[0]) & (even <= zc[-1])
    np.testing.assert_allclose(real[0, 0, inside, 0], even[inside], atol=1e-5)


def test_random_field_with_two_lengths():
    key = jax.random.PRNGKey(1)
    a = gaussian_random_field((20, 30), 3.0, key)
    b = gaussian_random_field((20, 30), (3.0, 3.0), key)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5)


# ---------------------------------------------------------------- graded x and y

# x edges (um, stack x) over BOX_XY: 20 nm cells from 0.2 to 0.9 um, growing to 60 nm outside
X_GRADED = bf.graded_edges(BOX_XY[0], BOX_XY[1], 0.06, fine=[(0.2, 0.9, 0.02)], growth=1.25)


def x_graded_grid(d: bf.StackDevice, dy: float = 0.02, dz: float = 0.02) -> RectilinearGrid:
    """X_GRADED along x, dy and dz cells along y and z, over the device's box (BOX_XY)."""
    _, sy, sz = (s * 1e6 for s in d.partial_real_shape)
    middle = 0.5 * (BOX_XY[0] + BOX_XY[1])
    return grid(X_GRADED - middle, np.linspace(-sy / 2, sy / 2, round(sy / dy) + 1),
                np.linspace(-sz / 2, sz / 2, round(sz / dz) + 1))


def dual_cells(e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[C_{i-1}, C_i] between neighbouring cell centers (the first cell mirrored below the first edge)."""
    c = 0.5 * (e[:-1] + e[1:])
    return np.concatenate([[e[0] - 0.5 * (e[1] - e[0])], c[:-1]]), c


Y_EDGES = BOX_XY[2] + 0.02 * np.arange(66)  # um: the y edges of x_graded_grid over BOX_XY


def sample_boxes(e: np.ndarray, shifted: bool) -> tuple[np.ndarray, np.ndarray]:
    """The sample boxes along an axis of edges e: the cells, or on a Yee-shifted axis the dual cells."""
    return dual_cells(e) if shifted else (e[:-1], e[1:])


def exact_areas(polygon, x: tuple, y: tuple, erode: float = 0.0) -> np.ndarray:
    """(nx, ny) fraction of every box [x_lo, x_hi] x [y_lo, y_hi] (um) inside the polygon eroded by ``erode`` um
    (its signed distance's level set: round at reflex corners)."""
    shapely = pytest.importorskip("shapely")
    shape = shapely.Polygon(polygon)
    if erode:
        shape = shape.buffer(-erode, join_style="round", quad_segs=256)
    xl, yl = np.meshgrid(x[0], y[0], indexing="ij")
    xh, yh = np.meshgrid(x[1], y[1], indexing="ij")
    return shapely.area(shapely.intersection(shape, shapely.box(xl, yl, xh, yh))) / ((xh - xl) * (yh - yl))


def y_graded_grid(d: bf.StackDevice, dx: float = 0.02, dz: float = 0.02) -> RectilinearGrid:
    """X_GRADED along y over a transposed lateral_device (its y spans BOX_XY's x), dx and dz cells along x and z."""
    sx, _, sz = (s * 1e6 for s in d.partial_real_shape)
    middle = 0.5 * (BOX_XY[0] + BOX_XY[1])
    return grid(np.linspace(-sx / 2, sx / 2, round(sx / dx) + 1), X_GRADED - middle,
                np.linspace(-sz / 2, sz / 2, round(sz / dz) + 1))


def test_graded_x_strip_fractions_are_exact():
    """corner_lines=0 on graded x: a strip with edges along y fills each column (its own cell, or for Ey, which
    fdtdx samples on the x faces, the dual cell) exactly by its overlap."""
    _, _, y0, y1 = BOX_XY
    strip = [(0.33, y0 - 0.5), (0.87, y0 - 0.5), (0.87, y1 + 0.5), (0.33, y1 + 0.5)]
    d = lateral_device(strip, shrink=(0.0, 0.5), corner_lines=0)
    placed = place(d, x_graded_grid(d))["s"]
    frame = placed.grid_frame  # the frame unit: the smallest lateral width (the fine x cells, about 0.02 um)
    assert frame.widths[0] is None and frame.widths[1] == pytest.approx(0.02 / frame.h_um)
    span = (0.33, 0.87)
    names, w = placed.material_weights()
    np.testing.assert_allclose(np.asarray(w[names.index("si")])[:, 20, CORE_LAYER],
                               overlap(span, X_GRADED[:-1], X_GRADED[1:]), atol=2e-5)
    names, w = placed.material_weights(shift=(-0.5, 0.0, -0.5))  # Ey
    np.testing.assert_allclose(np.asarray(w[names.index("si")])[:, 20, CORE_LAYER],
                               overlap(span, *dual_cells(X_GRADED)), atol=2e-5)


def test_graded_x_strip_along_x_matches_the_uniform_grid():
    """A strip with edges along x (sloped walls): every graded column is the uniform grid's column."""
    x0, x1, _, _ = BOX_XY
    strip = [(x0 - 0.5, 0.13), (x1 + 0.5, 0.13), (x1 + 0.5, 0.42), (x0 - 0.5, 0.42)]
    d = lateral_device(strip, angle=20.0, shrink=(0.5, 0.0), corner_lines=0)
    _, wg = place(d, x_graded_grid(d))["s"].material_weights()
    _, wu = place(d, lateral_grid(d, 0.02, 0.02))["s"].material_weights()
    wg, wu = np.asarray(wg), np.asarray(wu)
    np.testing.assert_allclose(wg, np.broadcast_to(wu[:, :1], wg.shape), atol=2e-5)


def test_graded_x_averaging_takes_the_horizontal_interface():
    """Per-column normal tensors on graded x: the core (everywhere laterally) ends at 0.22 um, a voxel face; Ex
    and Ey sample the dual cells 0.21 to 0.23 um, half silicon, and see the mean eps (the normal along z) in every
    graded column, while Ez sees silicon below the face and air above it."""
    d = lateral_device([(-0.3, -0.3), (1.5, -0.3), (1.5, 1.2), (-0.3, 1.2)], shrink=(0.1, 0.1), corner_lines=0)
    objects, arrays, params = place_all(d, lateral_grid_x(d))
    out, _, _ = bf.apply_stack(arrays, objects, params, jax.random.PRNGKey(0))
    inv = np.asarray(out.inv_permittivities[:, :, 30, :])  # (3, nx, nz); voxel layer k is -0.1 + 0.02 k um
    np.testing.assert_allclose(inv[2, :, 15], 1.0 / 12.0, rtol=1e-5)
    np.testing.assert_allclose(inv[2, :, 16], 1.0, rtol=1e-5)
    np.testing.assert_allclose(inv[0, :, 16], 2.0 / 13.0, rtol=1e-5)
    np.testing.assert_allclose(inv[1, :, 16], 2.0 / 13.0, rtol=1e-5)


def lateral_grid_x(d: bf.StackDevice) -> RectilinearGrid:
    """X_GRADED along x over a device whose box is BOX_XY shrunk by 0.1 um: its own graded edges."""
    sx, sy, sz = (s * 1e6 for s in d.partial_real_shape)
    x = bf.graded_edges(-sx / 2, sx / 2, 0.06, fine=[(-0.3, 0.3, 0.02)], growth=1.25)
    return grid(x, np.linspace(-sy / 2, sy / 2, round(sy / 0.02) + 1), np.linspace(-sz / 2, sz / 2, round(sz / 0.02) + 1))


def test_graded_x_traces_once_and_roughness_runs():
    rough = bf.TopRoughness(level="core", rms=0.004, correlation_length=0.1)
    d = lateral_device(L_SHAPE, angle=10.0, roughness=(rough,))  # with the corner model (corner_lines=2)
    placed = place(d, x_graded_grid(d))["s"]
    offsets = placed.interface_offsets(jax.random.PRNGKey(2))
    assert offsets[0].shape == tuple(placed.grid_shape[:2]) and bool(jnp.all(jnp.isfinite(offsets[0])))
    p0 = placed.init_params()
    traces = []

    def weights(t: jax.Array) -> jax.Array:
        traces.append(t)
        return placed.material_weights({**p0, "thickness/core": t})[1]

    f = jax.jit(weights)
    for t in (0.22, 0.21):
        assert bool(jnp.all(jnp.isfinite(f(jnp.asarray(t, dtype=jnp.float32)))))
    assert len(traces) == 1


@pytest.mark.parametrize(
    ("shift", "subsamples"),
    [((0.0, 0.0, 0.0), 1), ((-0.5, 0.0, -0.5), 1), ((0.0, -0.5, -0.5), 1), ((0.0, 0.0, 0.0), 2), ((-0.5, 0.0, -0.5), 2)],
)
def test_graded_x_corners_get_exact_areas(shift: tuple[float, float, float], subsamples: int):
    """The corner model on graded x: every voxel of a rotated non-convex polygon, at its corners too, gets its exact
    area, on the cells and on the Yee positions' dual cells, also painted on sub-samples."""
    d = lateral_device(L_SHAPE, xy_subsamples=subsamples)
    placed = place(d, x_graded_grid(d))["s"]
    names, w = placed.material_weights(shift=shift)
    exact = exact_areas(L_SHAPE, sample_boxes(X_GRADED, shift[0] != 0), sample_boxes(Y_EDGES, shift[1] != 0))
    assert ((exact > 0) & (exact < 1)).sum() > 50
    tol = 1e-4 if subsamples == 1 else 1e-3  # sub-samples blend the corner model by the voxel's certificate
    np.testing.assert_allclose(np.asarray(w[names.index("si")])[:, :, CORE_LAYER], exact, atol=tol)


def test_graded_x_sloped_corners_follow_the_eroded_polygon():
    """15 degree walls on graded x: the layer 0.10 to 0.12 um holds the polygon eroded by z tan(15 deg) (round at
    its reflex corner), averaged over the layer's heights. Its volume, summed over cells of different areas, is
    exact; per voxel the half-planes paint a wall at its mean offset over the layer, which differs from the mean
    area where the wall crosses a cell's corner within the layer (by up to 1.1e-2 of the voxel here, on a uniform
    0.02 um grid too), and these errors cancel along the edge."""
    angle = 15.0
    d = lateral_device(L_SHAPE, angle=angle)
    names, w = place(d, x_graded_grid(d))["s"].material_weights()
    nodes, gauss = np.polynomial.legendre.leggauss(8)
    boxes = (sample_boxes(X_GRADED, False), sample_boxes(Y_EDGES, False))
    exact = sum(0.5 * g * exact_areas(L_SHAPE, *boxes, (0.11 + 0.01 * x) * math.tan(math.radians(angle)))
                for x, g in zip(nodes, gauss, strict=True))
    got = np.asarray(w[names.index("si")])[:, :, CORE_LAYER]
    edge = (exact > 0) & (exact < 1)
    assert edge.sum() > 50
    areas = np.outer(np.diff(X_GRADED), np.diff(Y_EDGES))
    assert np.sum(got * areas) == pytest.approx(np.sum(exact * areas), rel=1e-4)
    assert np.abs(got - exact)[edge].mean() < 1.5e-3
    assert np.abs(got - exact).max() < 1.5e-2


def test_graded_y_is_graded_x_transposed():
    d = lateral_device(L_SHAPE, angle=15.0)
    _, w = place(d, x_graded_grid(d))["s"].material_weights()
    t = lateral_device(L_SHAPE, angle=15.0, transpose=True)
    _, wt = place(t, y_graded_grid(t))["s"].material_weights()
    np.testing.assert_allclose(np.asarray(wt), np.asarray(w).transpose(0, 2, 1, 3), atol=1e-4)


def test_graded_x_corner_tensors_keep_ez_along_vertical_walls():
    """The corner model's normal tensors on graded x: vertical walls are parallel to Ez, which sees the mean eps of
    its exact (dual x and y cell) fractions in every voxel inside the core, at corners too."""
    d = lateral_device(L_SHAPE)
    objects, arrays, params = place_all(d, x_graded_grid(d))
    out, _, _ = bf.apply_stack(arrays, objects, params, jax.random.PRNGKey(0))
    inv = np.asarray(out.inv_permittivities[2, :, :, CORE_LAYER])
    exact = exact_areas(L_SHAPE, dual_cells(X_GRADED), dual_cells(Y_EDGES))
    np.testing.assert_allclose((1.0 / inv - 1.0) / 11.0, exact, atol=1e-4)  # eps = 1 + 11 f


@pytest.mark.parametrize("angle", [0.0, 15.0])
def test_graded_x_full_averaging_closes_corners(angle: float):
    """averaging="subpixel_full" on graded x: the junction voxels' closure solves on every voxel's own box and stays
    between the materials; with vertical walls (z-invariant: the cell problem's zz entry is the mean eps) Ez keeps
    the mean eps of its exact dual-cell fractions in every voxel inside the core, junctions at corners too."""
    d = lateral_device(L_SHAPE, angle=angle, averaging="subpixel_full")
    objects, arrays, params = place_all(d, x_graded_grid(d))
    out, status = objects["s"].write_with_status(arrays, params["s"])
    assert int(status["junctions"]) > 0 and int(status["junction_overflow"]) == 0
    assert float(status["closure_residual"]) < 1e-3
    inv = np.asarray(out.inv_permittivities)
    inv = inv.reshape(3, 3, *inv.shape[1:])
    assert np.all(np.isfinite(inv))
    diagonal = inv[[0, 1, 2], [0, 1, 2]]
    assert diagonal.min() > 1.0 / 12.0 - 1e-5 and diagonal.max() < 1.0 + 1e-5
    assert np.abs(inv[[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]).max() < 0.5 * (1.0 - 1.0 / 12.0)
    if angle == 0.0:
        exact = exact_areas(L_SHAPE, dual_cells(X_GRADED), dual_cells(Y_EDGES))
        np.testing.assert_allclose((1.0 / inv[2, 2, :, :, CORE_LAYER] - 1.0) / 11.0, exact, atol=1e-4)


def test_mode_source_and_overlap_are_port_area_invariant_on_graded_grids():
    """ModePlaneSource and ModeOverlapDetector normalize against min_spacing**2 on graded grids:
    uniform grids stay bit-for-bit identical to fdtdx, while on graded grids two port planes with
    different paddings (or orthogonal x/y slices on a 90-degree bend with dx != dy) agree on modal
    power without fdtdx's mean(A_in)/mean(A_out) bias."""
    c = gf.Component()
    bend = c << gf.components.bend_euler(radius=1.8, width=0.45, layer=L.WG, allow_min_radius_violation=True)
    c.add_port("o1", port=bend.ports["o1"])
    c.add_port("o2", port=bend.ports["o2"])
    b = c.bbox()
    c.add_polygon(
        [(b.left - 2.0, b.bottom - 2.0), (b.right + 2.0, b.bottom - 2.0),
         (b.right + 2.0, b.top + 2.0), (b.left - 2.0, b.top + 2.0)],
        layer=L.WAFER,
    )
    ls = LayerStack(layers={
        "core": LayerLevel(layer=L.WG, zmin=0.0, thickness=0.22, material="si", mesh_order=1),
        "box": LayerLevel(layer=L.WAFER, zmin=-1.5, thickness=1.5, material="sio2", mesh_order=2),
    })
    d = bf.StackDevice.from_layer_stack(
        c, ls, MATERIALS, background="air", name="s", zrange_layers=["core"],
        boundary_offset=(1.4, -1.2, 1.8, -1.2, 1.0, -1.2), extend_ports=(("o1", 0.6), ("o2", 0.6)),
        diagnostics=False,
    )
    wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
    volume, placing = d.get_simulation_volume()

    # 1. UniformGrid: bit-for-bit identical to fdtdx's base classes
    u_cfg = d.get_simulation_config(time=2e-15, grid=UniformGrid(spacing=0.05e-6))
    src_u, sc_u = d.create_mode_plane_source_at("o1", (0.4, 0.3), wave_character=wave, port_offset=-0.2)
    det_u, dc_u = d.create_mode_overlap_detector_at("o1", (0.4, 0.3), wave_characters=(wave,), direction="in")
    assert isinstance(src_u, bf.ModePlaneSource) and isinstance(det_u, bf.ModeOverlapDetector)
    raw_src, rsc = d._at_port(
        fdtdx.ModePlaneSource, "o1", (0.4, 0.3), "mode_source", wavelengths_um=[1.55],
        direction="in", level=None, port_offset=-0.2, name="raw_src", wave_character=wave, mode_index=0,
    )
    raw_det, rdc = d._at_port(
        fdtdx.ModeOverlapDetector, "o1", (0.4, 0.3), "mode_detector", wavelengths_um=[1.55],
        direction="in", level=None, port_offset=0.0, name="raw_det", wave_characters=(wave,), mode_index=0,
    )
    objs_u, arr_u, par_u, u_cfg, _ = fdtdx.place_objects(
        object_list=[volume, d, src_u, det_u, raw_src, raw_det],
        config=u_cfg, constraints=placing + sc_u + dc_u + rsc + rdc, key=jax.random.PRNGKey(0),
    )
    arr_u, objs_u, _ = bf.apply_stack(arr_u, objs_u, {**par_u, **bf.init_stack_params(objs_u)}, jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(objs_u[src_u.name]._E), np.asarray(objs_u["raw_src"]._E))
    np.testing.assert_array_equal(
        np.asarray(objs_u[det_u.name]._cached_face_area_weights),
        np.asarray(objs_u["raw_det"]._cached_face_area_weights),
    )

    # 2. Graded RectilinearGrid with unequal padding and 90-deg x/y ports:
    #    launching amplitude=1 on either port gives physical flux == min_spacing**2 and |alpha| == 1
    m = bf.mesh(d, 1.55, ppw=16, pml=0.6)
    r_cfg = d.get_simulation_config(time=2e-15, grid=m.grid)
    s1, s1c = d.create_mode_plane_source_at("o1", (0.35, 0.30), wave_character=wave, port_offset=-0.2, grid=m.grid)
    s2, s2c = d.create_mode_plane_source_at("o2", (0.65, 0.50), wave_character=wave, port_offset=-0.2, grid=m.grid)
    d1, d1c = d.create_mode_overlap_detector_at(
        "o1", (0.35, 0.30), wave_characters=(wave,), direction="in", port_offset=-0.2, name="d1", grid=m.grid
    )
    d2, d2c = d.create_mode_overlap_detector_at(
        "o2", (0.65, 0.50), wave_characters=(wave,), direction="in", port_offset=-0.2, name="d2", grid=m.grid
    )
    objs_r, arr_r, par_r, r_cfg, _ = fdtdx.place_objects(
        object_list=[volume, d, s1, s2, d1, d2],
        config=r_cfg, constraints=placing + s1c + s2c + d1c + d2c, key=jax.random.PRNGKey(0),
    )
    arr_r, objs_r, _ = bf.apply_stack(arr_r, objs_r, {**par_r, **bf.init_stack_params(objs_r)}, jax.random.PRNGKey(0))

    a1 = np.asarray(bf.stack._plane_face_areas(objs_r["d1"]))
    a2 = np.asarray(bf.stack._plane_face_areas(objs_r["d2"]))
    assert a2.mean() / a1.mean() > 1.2  # genuinely different mean cell areas on the two port planes
    ref_area = float(r_cfg.resolved_grid.min_spacing) ** 2

    for s_name, d_name in ((s1.name, "d1"), (s2.name, "d2")):
        src_obj, det_obj = objs_r[s_name], objs_r[d_name]
        areas = np.asarray(bf.stack._plane_face_areas(src_obj))
        # Physical Poynting flux of the source's mode fields equals min_spacing**2 on both ports
        s_vec = 0.5 * np.real(np.cross(np.conj(np.asarray(src_obj._E)), np.asarray(src_obj._H), axis=0))
        flux = abs(float(np.sum(s_vec[src_obj.propagation_axis] * areas)))
        assert flux == pytest.approx(ref_area, rel=2e-5)
        # Feeding a mode of physical power 2 * ref_area (2 * |src|) into compute_overlap gives |alpha|^2 == 2.0
        phasor = jnp.concatenate([src_obj._E, src_obj._H], axis=0)[None, None] * math.sqrt(2.0)
        alpha = complex(np.asarray(det_obj.compute_overlap({"phasor": phasor.astype(det_obj.dtype)}))[0])
        assert abs(alpha) ** 2 == pytest.approx(2.0, rel=2e-5)
