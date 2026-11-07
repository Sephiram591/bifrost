"""Tests for the StackDevice's grid frame (the groundwork for non-uniform rectilinear grids)."""

from __future__ import annotations

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
from bifrost.stack import StackDiagnosticWarning, _conductivity_scale, _Frame, _grid_frame, _split_layer

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
    return RectilinearGrid(*(jnp.asarray(e, dtype=jnp.float32) * 1e-6 for e in (x_um, y_um, z_um)))


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
    return bf.StackDevice.from_layer_stack(
        c, ls, MATERIALS, background="air", name="s", zrange_layers=["core"],
        boundary_offset=(0.2, -0.2, 0.2, -0.2, 0.38, 0.0), diagnostics=False, **kwargs,
    )


def place(d: bf.StackDevice, g) -> dict:
    volume, constraints = d.get_simulation_volume()
    objects, *_ = fdtdx.place_objects(
        object_list=[volume, d], config=config(g), constraints=constraints, key=jax.random.PRNGKey(0)
    )
    return objects


def test_placed_device_has_its_frame():
    d = device()
    assert d.dropped_levels == (("under", -1.0, -0.15),)  # 0.15 um under the box: beyond the slack
    placed = place(d, UniformGrid(spacing=0.02e-6))["s"]
    assert placed.grid_frame is not None and placed.grid_frame.uniform == (True, True, True)
    assert placed._res_um() == placed._config.uniform_spacing() * 1e6
    assert tuple(len(e) - 1 for e in placed.grid_frame.edges) == tuple(placed.grid_shape)


def test_coarse_grid_warns_about_a_dropped_level():
    d = device()
    with pytest.warns(StackDiagnosticWarning, match="'under'.*left out"):
        place(d, UniformGrid(spacing=0.4e-6))  # half a 0.4 um cell under the box reaches it


def test_non_uniform_grid_is_not_supported_yet():
    d = device()
    sx, sy, sz = (s * 1e6 for s in d.partial_real_shape)
    z = np.array([0.0, 0.1, 0.15, 0.2, 0.25, 0.35, 0.45, 0.6]) * sz / 0.6 - sz / 2
    g = grid(np.linspace(-sx / 2, sx / 2, 15), np.linspace(-sy / 2, sy / 2, 10), z)
    with pytest.raises(NotImplementedError, match="non-uniform"):
        place(d, g)
