"""Regression tests for Wave 3 review fixes in ``bifrost``."""

from __future__ import annotations

import ast
import math
import warnings
from collections.abc import Iterator
from pathlib import Path

import fdtdx
import gdsfactory as gf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fdtdx.core.grid import RectilinearGrid, UniformGrid
from fdtdx.dispersion import DispersionModel
from fdtdx.materials import compute_allowed_dispersive_coefficients, compute_allowed_electric_conductivities
from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer

import bifrost as bf
from bifrost.geometry import SlotPole, has_tracers, to_numpy
from bifrost.stack import (
    _PENDING_REAPPLY,
    _YEE_E_SHIFTS,
    StackDiagnosticWarning,
    StackDevice,
    StackOptions,
    TopRoughness,
    _conductivity_scale,
    _kdb_fold_rings,
    _reapply_cache,
    _roughness_reach,
    apply_stack,
    init_stack_params,
    pole_slots,
)

RES = 0.04  # um
L = gf.gpdk.LAYER
WG, SLAB, ETCH, WAFER = L.WG, L.SLAB150, L.DEEP_ETCH, L.WAFER

MATERIALS = {
    "air": fdtdx.Material(),
    "sio2": fdtdx.Material(permittivity=2.1),
    "si": fdtdx.Material(permittivity=12.0),
}


@pytest.fixture(autouse=True)
def _pdk_env() -> Iterator[None]:
    gf.gpdk.PDK.activate()
    gf.snap.SNAP_ENABLED = False
    yield
    gf.snap.SNAP_ENABLED = True


def _rect(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def _poly_comp(layers: dict[object, list[np.ndarray]]) -> gf.Component:
    comp = gf.Component()
    for layer, polys in layers.items():
        for poly in polys:
            comp.add_polygon(poly, layer=layer)
    return comp


def _place(
    device: StackDevice,
    *,
    spacing: float = RES,
    grid: object | None = None,
    extra: tuple[object, ...] = (),
    extra_constraints: tuple[object, ...] = (),
) -> tuple[object, object, dict, object]:
    if not extra_constraints and (device.name is None or device.name.startswith("Object_")):
        device = device.aset("name", "stack")
    use_grid = UniformGrid(spacing=spacing * 1e-6) if grid is None else grid
    config = device.get_simulation_config(time=10e-15, grid=use_grid, backend=jax.default_backend())
    volume, vol_constraints = device.get_simulation_volume()
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=[volume, device, *extra],
        config=config,
        constraints=[*vol_constraints, *extra_constraints],
        key=jax.random.PRNGKey(0),
    )
    params = {**params, **init_stack_params(objects)}
    return objects, arrays, params, config


# --------------------------------------------------------------------------- I-BND-05


def test_bifrost_imports_zero_gdsfactory_jax_symbols() -> None:
    """No Python module inside bifrost may import gdsfactory._jax or its symbols."""
    pkg_dir = Path(bf.__file__).resolve().parent
    violations: list[str] = []
    for path in sorted(pkg_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "gdsfactory._jax" or alias.name.startswith("gdsfactory._jax."):
                        violations.append(f"{path.name}:{node.lineno} import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "gdsfactory._jax" or mod.startswith("gdsfactory._jax."):
                    violations.append(f"{path.name}:{node.lineno} from {mod}")
                elif mod == "gdsfactory" and any(alias.name == "_jax" for alias in node.names):
                    violations.append(f"{path.name}:{node.lineno} from gdsfactory import _jax")
    assert violations == []


def test_geometry_has_tracers_and_to_numpy_behavior() -> None:
    """has_tracers and to_numpy handle concrete arrays, eager JVP tracers, and jit tracers."""
    obj_arr = np.empty(2, dtype=object)
    obj_arr[0] = None
    obj_arr[1] = {"pts": [np.array([1.0, 2.0])]}
    assert not has_tracers({"nested": (jnp.ones(2), obj_arr)})
    np.testing.assert_allclose(to_numpy(jnp.array([1.25, 2.5])), [1.25, 2.5])

    seen: dict[str, object] = {}

    def eager_fn(x: jax.Array) -> jax.Array:
        seen["has"] = has_tracers({"k": [x]})
        seen["primal"] = to_numpy(x)
        return x * x

    jax.jvp(eager_fn, (jnp.asarray(2.5),), (jnp.asarray(1.0),))
    assert seen["has"] is True
    np.testing.assert_allclose(seen["primal"], 2.5)  # type: ignore[arg-type]

    @jax.jit
    def jit_fn(x: jax.Array) -> jax.Array:
        assert has_tracers({"leaf": (x,)})
        with pytest.raises(TypeError, match="concrete"):
            to_numpy(x)
        return x

    jit_fn(jnp.asarray(1.0))


# --------------------------------------------------------------------------- E-BF-01


def test_periodic_cell_tensor_lossless_plasmonic_finite_primal_and_grad() -> None:
    """Lossless plasmonic cell (eps1 + eps2 == 0) keeps finite SPD output and finite gradient."""
    eye = jnp.eye(3, dtype=jnp.float32)
    base = jnp.zeros((1, 2, 2, 2, 3, 3), dtype=jnp.float32)
    base = base.at[:, :1, :, :, :, :].set(4.0 * eye)
    base = base.at[:, 1:, :, :, :, :].set(-4.0 * eye)

    def obj(scale: jax.Array) -> jax.Array:
        tensor, _ = bf.subpixel.periodic_cell_tensor(scale * base)
        return jnp.sum(tensor)

    val, grad = jax.value_and_grad(obj)(jnp.asarray(1.0, dtype=jnp.float32))
    assert bool(jnp.isfinite(val)) and bool(jnp.isfinite(grad))
    tensor, res = bf.subpixel.periodic_cell_tensor(base)
    assert bool(jnp.all(jnp.isfinite(tensor)))
    assert float(np.linalg.eigvalsh(np.asarray(tensor[0])).min()) > 0.0
    assert bool(jnp.isnan(res[0]))


def test_complete_tensor_warn_fallback_and_subpixel_full_grad_finite() -> None:
    """subpixel_full closure stays finite, SPD, and differentiable even under capacity overflow."""
    ls = LayerStack(layers={"core": LayerLevel(layer=WG, zmin=0.0, thickness=0.16, material="si", mesh_order=1)})
    comp = _poly_comp({WG: [_rect(-0.26, -0.22, 0.26, 0.22)]})
    dev = StackDevice.from_layer_stack(
        comp,
        ls,
        {"air": MATERIALS["air"], "si": MATERIALS["si"]},
        background="air",
        zrange_layers=["core"],
        boundary_offset=(0.16, -0.16, 0.16, -0.16, 0.08, -0.08),
        averaging="subpixel_full",
        interface_failure="warn",
    )
    objects, arrays, params, _ = _place(dev)
    placed = objects["stack"]

    p0 = params["stack"]
    g = placed._interfaces(p0, None, True, (0.0, 0.0, 0.0))
    inv6, status = placed._complete_tensor(g, placed.junction_voxels)
    inv_mats = np.asarray(bf.subpixel.from_six(inv6), dtype=np.float64).reshape(-1, 3, 3)
    assert np.all(np.isfinite(inv_mats))
    assert float(np.linalg.eigvalsh(inv_mats).min()) > 0.0
    assert math.isfinite(float(status["closure_residual"]))

    # Overflow triggers warn fallback cleanly while keeping SPD/finite arrays and finite gradient.
    small = placed.aset("junction_voxels", 2)
    with pytest.warns(StackDiagnosticWarning, match="junction_voxels"):
        out_arr, out_status = small.write_with_status(arrays, p0)
    assert bool(out_status["failed"])
    assert bool(jnp.all(jnp.isfinite(out_arr.inv_permittivities)))

    def loss(verts: jax.Array) -> jax.Array:
        return jnp.sum(placed.write(arrays, {**p0, "vertices/WG": verts}).inv_permittivities)

    val, grad = jax.value_and_grad(loss)(p0["vertices/WG"])
    assert bool(jnp.isfinite(val)) and bool(jnp.all(jnp.isfinite(grad)))


# --------------------------------------------------------------------------- E-BF-03


def test_nested_island_in_hole_preserved_by_fold_rings_recipe_and_diagnose() -> None:
    """Islands inside holes survive ring folding in _kdb_fold_rings, diagnose(), and material_weights()."""
    pytest.importorskip("klayout.db")

    outer = _rect(-1.0, -1.0, 1.0, 1.0)
    hole = _rect(-0.6, -0.6, 0.6, 0.6)[::-1]
    island = _rect(-0.2, -0.2, 0.2, 0.2)
    reg_ordered = _kdb_fold_rings([outer, hole, island], 0.001)
    reg_scrambled = _kdb_fold_rings([island, outer, hole], 0.001)
    expected_area = 4.0 - 1.44 + 0.16
    assert reg_ordered.area() * 1e-6 == pytest.approx(expected_area, rel=1e-4)
    assert reg_scrambled.area() * 1e-6 == pytest.approx(expected_area, rel=1e-4)

    comp = _poly_comp(
        {
            WG: [_rect(-0.8, -0.8, 0.8, 0.8)],
            ETCH: [_rect(-0.5, -0.5, 0.5, 0.5)],
            SLAB: [_rect(-0.2, -0.2, 0.2, 0.2)],
        }
    )
    ls = LayerStack(
        layers={
            "core": LayerLevel(
                layer=(LogicalLayer(layer=WG) - LogicalLayer(layer=ETCH)) | LogicalLayer(layer=SLAB),
                derived_layer=LogicalLayer(layer=WG),
                zmin=0.0,
                thickness=0.20,
                material="si",
                sidewall_angle=15.0,
                width_to_z=0.5,
                mesh_order=1,
            )
        }
    )
    dev = StackDevice.from_layer_stack(
        comp,
        ls,
        {"air": MATERIALS["air"], "si": MATERIALS["si"]},
        background="air",
        zrange_layers=["core"],
        boundary_offset=(0.2, -0.2, 0.2, -0.2, 0.12, -0.12),
        derived_shapes="exact",
    )
    assert len(dev.derived_layers) == 1
    objects, _, params, _ = _place(dev)
    placed = objects["stack"]
    findings = placed.diagnose()
    assert isinstance(findings, list)

    names, w = placed.material_weights(params["stack"], apply_roughness=False)
    si_w = np.asarray(w[names.index("si")])
    nx, ny, nz = placed.grid_shape
    xs = (np.arange(nx) + 0.5 - nx / 2) * RES + placed.center[0]
    iy, iz = ny // 2, nz // 2
    ix_island = int(np.argmin(np.abs(xs - 0.0)))
    ix_trench = int(np.argmin(np.abs(xs - 0.35)))
    ix_outer = int(np.argmin(np.abs(xs - 0.65)))
    assert si_w[ix_island, iy, iz] > 0.9
    assert si_w[ix_trench, iy, iz] < 0.1
    assert si_w[ix_outer, iy, iz] > 0.9


# --------------------------------------------------------------------------- E-BF-04


def test_stack_options_and_device_validation() -> None:
    """Invalid StackOptions and StackDevice field values are rejected with TypeError/ValueError."""
    with pytest.raises(ValueError, match="averaging"):
        StackOptions(averaging="bogus")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="yee_staggered"):
        StackOptions(yee_staggered=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="corner_lines"):
        StackOptions(corner_lines=-1)
    with pytest.raises(TypeError, match="corner_lines"):
        StackOptions(corner_lines=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="xy_subsamples"):
        StackOptions(xy_subsamples=0)
    with pytest.raises(ValueError, match="derived_shapes"):
        StackOptions(derived_shapes="approx")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="diagnostics"):
        StackOptions(diagnostics=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="junction_cells"):
        StackOptions(junction_cells=(0, 4, 4))
    with pytest.raises(ValueError, match="xy_subsamples"):
        StackOptions(averaging="subpixel_full", xy_subsamples=2, junction_cells=(3, 4, 4))
    with pytest.raises(ValueError, match="interface_failure"):
        StackOptions(interface_failure="ignore")  # type: ignore[arg-type]

    comp = _poly_comp({WG: [_rect(-0.3, -0.3, 0.3, 0.3)]})
    ls = LayerStack(layers={"core": LayerLevel(layer=WG, zmin=0.0, thickness=0.2, material="si", mesh_order=1)})
    base_kw = dict(materials={"air": MATERIALS["air"], "si": MATERIALS["si"]}, background="air", zrange_layers=["core"])
    with pytest.raises(ValueError, match="edge_width"):
        StackDevice.from_layer_stack(comp, ls, edge_width=0.0, **base_kw)
    with pytest.raises(TypeError, match="edge_width"):
        StackDevice.from_layer_stack(comp, ls, edge_width=True, **base_kw)
    with pytest.raises(ValueError, match="edge_chunk"):
        StackDevice.from_layer_stack(comp, ls, edge_chunk=0, **base_kw)
    with pytest.raises(ValueError, match="corner_columns"):
        StackDevice.from_layer_stack(comp, ls, corner_columns=-1, **base_kw)
    with pytest.raises(ValueError, match="junction_voxels"):
        StackDevice.from_layer_stack(comp, ls, junction_voxels=-1, **base_kw)
    with pytest.raises(ValueError, match="level_margin"):
        StackDevice.from_layer_stack(comp, ls, level_margin=-0.1, **base_kw)
    with pytest.raises(TypeError, match="noise_seed"):
        StackDevice.from_layer_stack(comp, ls, noise_seed=False, **base_kw)


def test_junction_inputs_asymmetric_per_x_and_per_y() -> None:
    """Asymmetric junction_cells uses distinct per_x and per_y subdivision along x and y under legacy and greens."""
    comp = _poly_comp({WG: [_rect(-0.24, -0.24, 0.24, 0.24)]})
    ls = LayerStack(layers={"core": LayerLevel(layer=WG, zmin=0.0, thickness=0.16, material="si", mesh_order=1)})
    for jcells in ((2, 4, 4), (4, 2, 4)):
        dev = StackDevice.from_layer_stack(
            comp,
            ls,
            {"air": MATERIALS["air"], "si": MATERIALS["si"]},
            background="air",
            zrange_layers=["core"],
            boundary_offset=(0.12, -0.12, 0.12, -0.12, 0.08, -0.08),
            averaging="subpixel_full",
            xy_subsamples=2,
            junction_cells=jcells,
            voxelizer="legacy",
        )
        objects, arrays, params, _ = _place(dev)
        placed = objects["stack"]
        g = placed._interfaces(params["stack"], None, True, (0.0, 0.0, 0.0))
        inputs, cells_of = placed._junction_inputs(g, jnp.arange(4, dtype=jnp.int32))
        fracs, tensors = cells_of(inputs)
        assert fracs.shape[:4] == (4, *jcells)
        assert tensors.shape[:4] == (4, *jcells)
        out_arr, status = placed.write_with_status(arrays, params["stack"])
        assert not bool(status["failed"])
        assert bool(jnp.all(jnp.isfinite(out_arr.inv_permittivities)))

        dev_g = StackDevice.from_layer_stack(
            comp,
            ls,
            {"air": MATERIALS["air"], "si": MATERIALS["si"]},
            background="air",
            zrange_layers=["core"],
            boundary_offset=(0.12, -0.12, 0.12, -0.12, 0.08, -0.08),
            averaging="subpixel_full",
            xy_subsamples=2,
            junction_cells=jcells,
        )
        objs_g, arrs_g, params_g, _ = _place(dev_g)
        out_g, st_g = objs_g["stack"].write_with_status(arrs_g, params_g["stack"])
        assert not bool(st_g["failed"])
        assert bool(jnp.all(jnp.isfinite(out_g.inv_permittivities)))


def test_degenerate_90deg_corner_isotropic_subpixel_full_grad_finite() -> None:
    """Symmetric 90-degree corner centered on cell boundaries yields finite primal and grad."""
    comp = _poly_comp({WG: [_rect(-0.20, -0.20, 0.20, 0.20)]})
    ls = LayerStack(layers={"core": LayerLevel(layer=WG, zmin=0.0, thickness=0.16, material="si", mesh_order=1)})
    dev = StackDevice.from_layer_stack(
        comp,
        ls,
        {"air": MATERIALS["air"], "si": MATERIALS["si"]},
        background="air",
        zrange_layers=["core"],
        boundary_offset=(0.16, -0.16, 0.16, -0.16, 0.08, -0.08),
        averaging="subpixel_full",
    )
    objects, arrays, params, _ = _place(dev)
    placed = objects["stack"]
    p0 = params["stack"]

    val, grad = jax.value_and_grad(
        lambda v: jnp.sum(placed.write(arrays, {**p0, "vertices/WG": v}).inv_permittivities)
    )(p0["vertices/WG"])
    assert bool(jnp.isfinite(val))
    assert bool(jnp.all(jnp.isfinite(grad)))


# --------------------------------------------------------------------------- E-BF-05 (M1 Option A)


def test_init_params_and_pack_match_device_dtype() -> None:
    """init_params() and pack() both return arrays in the device's dtype (float32 and float64)."""
    comp = _poly_comp({WG: [_rect(-0.3, -0.2, 0.3, 0.2)]})
    ls = LayerStack(layers={"core": LayerLevel(layer=WG, zmin=0.0, thickness=0.2, material="si", mesh_order=1)})
    for dt in (jnp.float32, jnp.float64):
        dev = StackDevice.from_layer_stack(
            comp,
            ls,
            {"air": MATERIALS["air"], "si": MATERIALS["si"]},
            background="air",
            zrange_layers=["core"],
            dtype=dt,
        )
        for tree in (dev.init_params(), dev.pack(comp)):
            for leaf in jax.tree.leaves(tree):
                assert np.dtype(leaf.dtype) == np.dtype(dt)


def test_float32_large_reticle_offset_warns_and_preserves_local_geometry() -> None:
    """M1 Option A: float32 warns when ULP rounding exceeds 1% spacing while _layer_geometry shifts in float64."""
    shift = 20_000.0
    sub_cell = 0.0005  # below float32 ULP (~0.0024 um) at 20_000 um
    verts_local = np.array([[-0.3, -0.2], [0.3 + sub_cell, -0.2], [0.3, 0.2], [-0.3, 0.2]], dtype=float)
    comp0 = _poly_comp({WG: [verts_local]})
    comp1 = _poly_comp({WG: [verts_local + shift]})
    ls = LayerStack(layers={"core": LayerLevel(layer=WG, zmin=0.0, thickness=0.2, material="si", mesh_order=1)})
    kw = dict(
        materials={"air": MATERIALS["air"], "si": MATERIALS["si"]},
        background="air",
        zrange_layers=["core"],
        boundary_offset=(0.16, -0.16, 0.16, -0.16, 0.08, -0.08),
    )
    d0 = StackDevice.from_layer_stack(comp0, ls, dtype=jnp.float32, **kw)
    d1_f32 = StackDevice.from_layer_stack(comp1, ls, dtype=jnp.float32, **kw)
    d1_f64 = StackDevice.from_layer_stack(comp1, ls, dtype=jnp.float64, **kw)

    with warnings.catch_warnings(record=True) as rec0:
        warnings.simplefilter("always")
        objs0, _, _, _ = _place(d0, spacing=0.02)
    assert not any("ULP rounding" in str(w.message) for w in rec0)

    with pytest.warns(UserWarning, match="ULP rounding"):
        objs1, _, _, _ = _place(d1_f32, spacing=0.02)

    with warnings.catch_warnings(record=True) as rec64:
        warnings.simplefilter("always")
        _place(d1_f64, spacing=0.02)
    assert not any("ULP rounding" in str(w.message) for w in rec64)

    p0_64 = {"vertices/WG": np.asarray(objs0["stack"].source_vertices[0], dtype=np.float64)}
    p1_64 = {"vertices/WG": np.asarray(objs1["stack"].source_vertices[0], dtype=np.float64)}
    v0 = np.asarray(objs0["stack"]._layer_geometry(p0_64)[0].vertices)
    v1 = np.asarray(objs1["stack"]._layer_geometry(p1_64)[0].vertices)
    assert v1.dtype == np.float32
    np.testing.assert_allclose(v1, v0, atol=1e-5)

    # Traced x64 path also runs without assuming x64 is always on.
    v1_jit = jax.jit(lambda x: objs1["stack"]._layer_geometry({"vertices/WG": x})[0].vertices)(
        jnp.asarray(d1_f64.source_vertices[0], dtype=jnp.float64)
    )
    np.testing.assert_allclose(np.asarray(v1_jit), v0, atol=1e-5)
    with jax.enable_x64(False):
        v0_no64 = jax.jit(lambda x: objs0["stack"]._layer_geometry({"vertices/WG": x})[0].vertices)(
            jnp.asarray(d0.source_vertices[0], dtype=jnp.float32)
        )
        assert bool(jnp.all(jnp.isfinite(v0_no64)))


# --------------------------------------------------------------------------- E-BF-07


def test_zero_thickness_nonzero_sidewall_angle_finite_primal_and_grad() -> None:
    """Setting level thickness=0 with nonzero sidewall_angle and z_to_bias keeps primal and grad finite."""
    comp = _poly_comp({WG: [_rect(-0.3, -0.2, 0.3, 0.2)]})
    for z2b in (None, ([0.0, 1.0], [0.0, 0.02])):
        ls = LayerStack(
            layers={
                "core": LayerLevel(
                    layer=WG,
                    zmin=0.0,
                    thickness=0.12,
                    material="si",
                    sidewall_angle=20.0,
                    width_to_z=0.5,
                    z_to_bias=z2b,
                    mesh_order=1,
                )
            }
        )
        dev = StackDevice.from_layer_stack(
            comp,
            ls,
            {"air": MATERIALS["air"], "si": MATERIALS["si"]},
            background="air",
            zrange_layers=["core"],
            boundary_offset=(0.16, -0.16, 0.16, -0.16, 0.08, -0.08),
        )
        objects, arrays, params, _ = _place(dev)
        placed = objects["stack"]
        p_zero = {**params["stack"], "thickness/core": jnp.asarray(0.0, dtype=placed.dtype)}
        arr_out = placed.write(arrays, p_zero)
        assert bool(jnp.all(jnp.isfinite(arr_out.inv_permittivities)))

        def loss(verts: jax.Array, thick: jax.Array, _placed=placed, _p=p_zero, _arr=arrays) -> jax.Array:
            p_cur = {**_p, "vertices/WG": verts, "thickness/core": thick}
            return jnp.sum(_placed.write(_arr, p_cur).inv_permittivities)

        val, (gv, gt) = jax.value_and_grad(loss, argnums=(0, 1))(p_zero["vertices/WG"], p_zero["thickness/core"])
        assert bool(jnp.isfinite(val))
        assert bool(jnp.all(jnp.isfinite(gv))) and bool(jnp.isfinite(gt))


# --------------------------------------------------------------------------- E-BF-08


def test_retopologize_grows_junction_voxels_and_check_topology_fill_only() -> None:
    """retopologize grows junction_voxels under legacy and handles denser topologies dynamically under greens."""
    comp_simple = _poly_comp({WG: [_rect(-0.8, -0.1, 0.8, 0.1)]})
    ls_core = LayerStack(layers={"core": LayerLevel(layer=WG, zmin=0.0, thickness=0.16, material="si", mesh_order=1)})
    dev = StackDevice.from_layer_stack(
        comp_simple,
        ls_core,
        {"air": MATERIALS["air"], "si": MATERIALS["si"]},
        background="air",
        zrange_layers=["core"],
        boundary_offset=(0.4, -0.4, 0.6, -0.6, 0.08, -0.08),
        averaging="subpixel_full",
        interface_failure="raise",
        voxelizer="legacy",
    )
    objects, arrays, _, _ = _place(dev)
    placed = objects["stack"]

    islands = [
        _rect(-0.75 + 0.2 * i, -0.45 + 0.4 * j, -0.63 + 0.2 * i, -0.33 + 0.4 * j)
        for i in range(8)
        for j in range(3)
    ]
    comp_dense = _poly_comp({WG: islands})
    retop = placed.retopologize(comp_dense)
    assert retop.junction_voxels is not None and placed.junction_voxels is not None
    assert retop.junction_voxels > placed.junction_voxels
    assert (retop.corner_columns or 0) >= (placed.corner_columns or 0)
    _, status = retop.write_with_status(arrays, retop.pack(comp_dense))
    assert int(status["junction_overflow"]) == 0

    dev_g = StackDevice.from_layer_stack(
        comp_simple,
        ls_core,
        {"air": MATERIALS["air"], "si": MATERIALS["si"]},
        background="air",
        zrange_layers=["core"],
        boundary_offset=(0.4, -0.4, 0.6, -0.6, 0.08, -0.08),
        averaging="subpixel_full",
        interface_failure="raise",
    )
    objs_g, arrs_g, _, _ = _place(dev_g)
    retop_g = objs_g["stack"].retopologize(comp_dense)
    assert retop_g.junction_voxels is None
    _, st_g = retop_g.write_with_status(arrs_g, retop_g.pack(comp_dense))
    assert int(st_g["junction_overflow"]) == 0

    comp_fill = _poly_comp({WAFER: [_rect(-0.3, -0.3, 0.3, 0.3)]})
    ls_fill = LayerStack(layers={"sub": LayerLevel(layer=WAFER, zmin=0.0, thickness=0.16, material="si", mesh_order=1)})
    fill_dev = StackDevice.from_layer_stack(
        comp_fill,
        ls_fill,
        {"air": MATERIALS["air"], "si": MATERIALS["si"]},
        background="air",
        zrange_layers=["sub"],
        fill_layers=(WAFER,),
    )
    assert fill_dev.source_layers == ()
    assert fill_dev.check_topology({}) is None


# --------------------------------------------------------------------------- E-BF-09


def test_top_roughness_validation_and_finite_grad() -> None:
    """TopRoughness rejects invalid inputs and differentiates cleanly wrt rms."""
    with pytest.raises(ValueError, match="level"):
        TopRoughness(level="", rms=0.01)
    with pytest.raises(TypeError, match="fill_above"):
        TopRoughness(level="core", rms=0.01, fill_above=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="rms"):
        TopRoughness(level="core", rms=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rms"):
        TopRoughness(level="core", rms=-0.01)
    with pytest.raises(ValueError, match="rms"):
        TopRoughness(level="core", rms=float("nan"))
    with pytest.raises(ValueError, match="correlation_length"):
        TopRoughness(level="core", rms=0.01, correlation_length=-0.01)

    comp = _poly_comp({WG: [_rect(-0.24, -0.20, 0.24, 0.20)]})
    ls = LayerStack(layers={"core": LayerLevel(layer=WG, zmin=0.0, thickness=0.16, material="si", mesh_order=1)})
    dev = StackDevice.from_layer_stack(
        comp,
        ls,
        {"air": MATERIALS["air"], "si": MATERIALS["si"]},
        background="air",
        zrange_layers=["core"],
        boundary_offset=(0.12, -0.12, 0.12, -0.12, 0.12, -0.12),
        roughness=(TopRoughness(level="core", rms=0.02, correlation_length=0.08),),
    )
    objects, arrays, params, _ = _place(dev)
    placed = objects["stack"]
    nkey = jax.random.PRNGKey(3)

    def loss(rms: jax.Array) -> jax.Array:
        rough_dev = placed.aset("roughness", (TopRoughness(level="core", rms=rms, correlation_length=0.08),))
        return jnp.sum(rough_dev.write(arrays, params["stack"], noise_key=nkey).inv_permittivities)

    val, grad = jax.value_and_grad(loss)(jnp.asarray(0.02, dtype=placed.dtype))
    assert bool(jnp.isfinite(val))
    assert bool(jnp.isfinite(grad)) and float(jnp.abs(grad)) > 0.0


def test_roughness_reach_5rms_and_fill_above_keep_reachable_levels() -> None:
    """_roughness_reach and _keep_reachable_levels pad by 5 * rms and retain fill_above coupled bases."""
    items = [("base", -0.5, -0.1), ("cap", -0.1, 0.2), ("far", 0.5, 0.8)]
    keep_plain, _ = _roughness_reach(items, (), 0.0, 0.3)
    assert keep_plain == {"cap"}

    keep_5rms, _ = _roughness_reach(items, (TopRoughness(level="base", rms=0.03, fill_above=False),), 0.0, 0.3)
    assert keep_5rms == {"base", "cap"}

    keep_coupled, _ = _roughness_reach(items, (TopRoughness(level="base", rms=0.005, fill_above=True),), 0.0, 0.3)
    assert keep_coupled == {"base", "cap"}

    keep_uncoupled, _ = _roughness_reach(items, (TopRoughness(level="base", rms=0.005, fill_above=False),), 0.0, 0.3)
    assert keep_uncoupled == {"cap"}

    # Placed integration: base below box is kept in from_layer_stack, then retained iff 5 * rms reaches box bottom.
    comp = _poly_comp({SLAB: [_rect(-0.24, -0.24, 0.24, 0.24)], WG: [_rect(-0.24, -0.24, 0.24, 0.24)]})
    ls = LayerStack(
        layers={
            "base": LayerLevel(layer=SLAB, zmin=-0.18, thickness=0.14, material="sio2", mesh_order=2),
            "core": LayerLevel(layer=WG, zmin=0.0, thickness=0.20, material="si", mesh_order=1),
        }
    )
    dev_keep = StackDevice.from_layer_stack(
        comp,
        ls,
        MATERIALS,
        background="air",
        zrange_layers=["core"],
        boundary_offset=(0.12, -0.12, 0.12, -0.12, 0.04, 0.04),
        roughness=(TopRoughness(level="base", rms=0.02, fill_above=False),),
    )
    assert {lv.name for lv in dev_keep.levels} == {"base", "core"}
    objs_keep, _, _, _ = _place(dev_keep)
    assert {lv.name for lv in objs_keep["stack"].levels} == {"base", "core"}
    assert [r.level for r in objs_keep["stack"].roughness] == ["base"]

    dev_drop = StackDevice.from_layer_stack(
        comp,
        ls,
        MATERIALS,
        background="air",
        zrange_layers=["core"],
        boundary_offset=(0.12, -0.12, 0.12, -0.12, 0.04, 0.04),
        roughness=(TopRoughness(level="base", rms=0.005, fill_above=False),),
    )
    assert {lv.name for lv in dev_drop.levels} == {"base", "core"}
    objs_drop, _, _, _ = _place(dev_drop)
    assert {lv.name for lv in objs_drop["stack"].levels} == {"core"}
    assert objs_drop["stack"].roughness == ()


# --------------------------------------------------------------------------- E-BF-11


def test_sigma_9_component_yee_weighting() -> None:
    """9-component electric_conductivity writes per-axis Yee-staggered weights across each tensor row."""
    mats = {
        "air": MATERIALS["air"],
        "cond": fdtdx.Material(permittivity=2.0, electric_conductivity=1e4),
    }
    comp = _poly_comp({WG: [_rect(-0.24, -0.24, 0.02, 0.24)]})
    ls = LayerStack(layers={"core": LayerLevel(layer=WG, zmin=0.0, thickness=0.16, material="cond", mesh_order=1)})
    dev = StackDevice.from_layer_stack(
        comp,
        ls,
        mats,
        background="air",
        zrange_layers=["core"],
        boundary_offset=(0.24, -0.14, 0.12, -0.12, 0.08, -0.08),
        averaging="subpixel",
        yee_staggered=True,
    )
    objects, arrays, params, config = _place(dev)
    placed = objects["stack"]
    dtype = arrays.inv_permittivities.dtype
    arrays9 = arrays.at["electric_conductivity"].set(
        jnp.zeros((9, *arrays.inv_permittivities.shape[1:]), dtype=dtype)
    )
    out = placed.write(arrays9, params["stack"])
    written = np.asarray(out.electric_conductivity[:, *placed.grid_slice])
    assert written.shape[0] == 9
    assert np.all(np.isfinite(written))

    names = fdtdx.materials.compute_ordered_names(placed.materials)
    idx_cond = names.index("cond")
    sigma_table = np.asarray(
        compute_allowed_electric_conductivities(placed.materials, isotropic=False, diagonally_anisotropic=False),
        dtype=float,
    )
    scale = float(_conductivity_scale(config))
    w_shifts = [np.asarray(placed.material_weights(params["stack"], shift=s)[1][idx_cond]) for s in _YEE_E_SHIFTS]
    for c in range(3):
        for d in range(3):
            want = scale * sigma_table[idx_cond, 3 * c + d] * w_shifts[c]
            np.testing.assert_allclose(written[3 * c + d], want, rtol=1e-5, atol=1e-6)
    assert float(np.abs(w_shifts[0] - w_shifts[1]).max()) > 1e-2


def test_rectilinear_grid_validation_and_float32_edge_collapse() -> None:
    """rectilinear_grid validates spacing_um and raises ValueError if float32 casting collapses edges."""
    comp = _poly_comp({WG: [_rect(-0.2, -0.2, 0.2, 0.2)]})
    ls = LayerStack(layers={"core": LayerLevel(layer=WG, zmin=0.0, thickness=0.2, material="si", mesh_order=1)})
    base_kw = dict(
        materials={"air": MATERIALS["air"], "si": MATERIALS["si"]},
        background="air",
        zrange_layers=["core"],
        boundary_offset=(0.2, -0.2, 0.2, -0.2, 0.1, -0.1),
    )
    dev32 = StackDevice.from_layer_stack(comp, ls, dtype=jnp.float32, **base_kw)
    dev64 = StackDevice.from_layer_stack(comp, ls, dtype=jnp.float64, **base_kw)

    for bad in (0.0, -0.05, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="spacing_um"):
            dev32.rectilinear_grid(bad)
    with pytest.raises(TypeError, match="spacing_um"):
        dev32.rectilinear_grid(True)  # type: ignore[arg-type]

    # Distinct in float64 (~1e-9 um gap at ~0.2 um) but identical in float32 metres.
    collapsed_x = np.array([-0.4, 0.2, 0.2 + 1e-9, 0.4], dtype=np.float64)
    y_edges = np.array([-0.4, 0.0, 0.4], dtype=np.float64)
    z_edges = np.array([-0.1, 0.1, 0.3], dtype=np.float64)
    with pytest.raises(ValueError, match="collapses adjacent edges"):
        dev32.rectilinear_grid(0.2, x_um=collapsed_x, y_um=y_edges, z_um=z_edges)
    grid64 = dev64.rectilinear_grid(0.2, x_um=collapsed_x, y_um=y_edges, z_um=z_edges)
    assert isinstance(grid64, RectilinearGrid)


def test_write_with_status_checks_materials_and_rejects_magnetic() -> None:
    """write_with_status runs _check_materials and rejects magnetic / non-SPD materials."""
    comp = _poly_comp({WG: [_rect(-0.2, -0.2, 0.2, 0.2)]})
    ls = LayerStack(layers={"core": LayerLevel(layer=WG, zmin=0.0, thickness=0.16, material="si", mesh_order=1)})

    with pytest.raises(ValueError, match="magnetic"):
        StackDevice.from_layer_stack(
            comp,
            ls,
            {"air": MATERIALS["air"], "si": fdtdx.Material(permittivity=2.0, permeability=2.0)},
            background="air",
            zrange_layers=["core"],
        )

    dev = StackDevice.from_layer_stack(
        comp,
        ls,
        {"air": MATERIALS["air"], "si": MATERIALS["si"]},
        background="air",
        zrange_layers=["core"],
        boundary_offset=(0.12, -0.12, 0.12, -0.12, 0.08, -0.08),
        averaging="subpixel_full",
    )
    objects, arrays, params, _ = _place(dev)
    placed = objects["stack"]

    for bad_mat in (
        fdtdx.Material(permittivity=2.0, permeability=2.0),
        fdtdx.Material(permittivity=2.0, magnetic_conductivity=1e3),
        fdtdx.Material(permittivity=-1.0),
    ):
        tampered = placed.aset("materials", {**placed.materials, "si": bad_mat})
        with pytest.raises(ValueError):
            tampered.write_with_status(arrays, params["stack"])


def test_apply_stack_omits_fixed_keys_checks_overlap_and_dtype() -> None:
    """apply_stack accepts omitted StackDevice keys and rejects overlapping boxes or dtype mismatches."""
    comp1 = _poly_comp({WG: [_rect(-0.2, -0.2, 0.2, 0.2)]})
    comp2 = _poly_comp({WG: [_rect(0.8, -0.2, 1.2, 0.2)]})
    comp_overlap = _poly_comp({WG: [_rect(-0.08, -0.2, 0.32, 0.2)]})
    ls = LayerStack(layers={"core": LayerLevel(layer=WG, zmin=0.0, thickness=0.16, material="si", mesh_order=1)})
    kw = dict(
        materials={"air": MATERIALS["air"], "si": MATERIALS["si"]},
        background="air",
        zrange_layers=["core"],
        boundary_offset=(0.12, -0.12, 0.12, -0.12, 0.08, -0.08),
    )
    s1 = StackDevice.from_layer_stack(comp1, ls, name="s1", dtype=jnp.float32, **kw)
    objects, arrays, _, _ = _place(s1)
    arrays_out, _, info = apply_stack(arrays, objects, {}, jax.random.PRNGKey(0))
    assert bool(jnp.all(jnp.isfinite(arrays_out.inv_permittivities)))
    assert "stack_status" in info

    # Mismatched arrays dtype.
    arrays64 = arrays.at["inv_permittivities"].set(arrays.inv_permittivities.astype(jnp.float64))
    with pytest.raises(ValueError, match="dtype"):
        apply_stack(arrays64, objects, {}, jax.random.PRNGKey(0))

    # Disjoint vs overlapping / mixed-dtype stacks inside one volume.
    s2 = StackDevice.from_layer_stack(comp2, ls, name="s2", dtype=jnp.float32, **kw)
    s_ov = StackDevice.from_layer_stack(comp_overlap, ls, name="s_ov", dtype=jnp.float32, **kw)
    vol = fdtdx.SimulationVolume(partial_real_shape=(2.4e-6, 0.8e-6, 0.4e-6))
    cfg32 = s1.get_simulation_config(time=10e-15, grid=UniformGrid(spacing=RES * 1e-6), backend=jax.default_backend())

    def place_two(other: StackDevice):
        return fdtdx.place_objects(
            object_list=[vol, s1, other],
            config=cfg32,
            constraints=[
                s1.place_relative_to(
                    vol,
                    axes=(0, 1, 2),
                    own_positions=(0, 0, 0),
                    other_positions=(0, 0, 0),
                    margins=((s1.center[0] - 0.48) * 1e-6, 0.0, 0.0),
                ),
                other.place_relative_to(
                    vol,
                    axes=(0, 1, 2),
                    own_positions=(0, 0, 0),
                    other_positions=(0, 0, 0),
                    margins=((other.center[0] - 0.48) * 1e-6, 0.0, 0.0),
                ),
            ],
            key=jax.random.PRNGKey(0),
        )

    objs_disjoint, arr_disjoint, _, _, _ = place_two(s2)
    res_disjoint, _, _ = apply_stack(arr_disjoint, objs_disjoint, {}, jax.random.PRNGKey(0))
    assert bool(jnp.all(jnp.isfinite(res_disjoint.inv_permittivities)))

    mixed_list = [o.aset("dtype", jnp.float64) if o.name == "s2" else o for o in objs_disjoint.object_list]
    objs_mixed = type(objs_disjoint)(object_list=mixed_list, volume_idx=objs_disjoint.volume_idx)
    with pytest.raises(ValueError, match="one dtype"):
        apply_stack(arr_disjoint, objs_mixed, {}, jax.random.PRNGKey(0))

    objs_ov, arr_ov, _, _, _ = place_two(s_ov)
    with pytest.raises(ValueError, match="overlapping grid boxes"):
        apply_stack(arr_ov, objs_ov, {}, jax.random.PRNGKey(0))


# --------------------------------------------------------------------------- E-BF-12


def test_from_layer_stack_validation_and_list_bias() -> None:
    """from_layer_stack validates levels, extend_ports, sidewall_angle, z_to_bias, and list bias."""
    comp = gf.components.straight(length=1.0, width=0.5)
    ls = LayerStack(layers={"core": LayerLevel(layer=WG, zmin=0.0, thickness=0.2, material="si", mesh_order=1)})
    kw = dict(materials={"air": MATERIALS["air"], "si": MATERIALS["si"]}, background="air", zrange_layers=["core"])

    with pytest.raises(ValueError, match="Duplicate level names"):
        StackDevice.from_layer_stack(comp, ls, levels=["core", "core"], **kw)
    with pytest.raises(TypeError, match="levels must be a sequence"):
        StackDevice.from_layer_stack(comp, ls, levels="core", **kw)
    with pytest.raises(KeyError, match="Unknown levels"):
        StackDevice.from_layer_stack(comp, ls, levels=["missing"], **kw)

    with pytest.raises(ValueError, match="Duplicate extend_ports"):
        StackDevice.from_layer_stack(comp, ls, extend_ports=(("o1", 0.3), ("o1", 0.4)), **kw)
    with pytest.raises(ValueError, match="extend_ports"):
        StackDevice.from_layer_stack(comp, ls, extend_ports=(("o1",),), **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="length > 0"):
        StackDevice.from_layer_stack(comp, ls, extend_ports=(("o1", -0.2),), **kw)

    for bad_angle in (90.0, -90.0, float("nan")):
        ls_bad_ang = LayerStack(
            layers={"core": LayerLevel(layer=WG, zmin=0.0, thickness=0.2, sidewall_angle=bad_angle, material="si")}
        )
        with pytest.raises(ValueError, match="sidewall_angle"):
            StackDevice.from_layer_stack(comp, ls_bad_ang, **kw)

    for bad_z2b in (([], []), ([0.0, 0.5], [0.0]), ([0.5, 0.5], [0.0, 0.1])):
        lvl = LayerLevel(layer=WG, zmin=0.0, thickness=0.2, z_to_bias=bad_z2b, material="si")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="z_to_bias"):
            StackDevice.from_layer_stack(comp, LayerStack(layers={"core": lvl}), **kw)
    lvl_bad_len = LayerLevel.model_construct(
        layer=WG, zmin=0.0, thickness=0.2, z_to_bias=([0.0],), material="si"
    )
    with pytest.raises(ValueError, match="z_to_bias"):
        StackDevice.from_layer_stack(
            comp, LayerStack.model_construct(layers={"core": lvl_bad_len}), **kw
        )

    lvl_ok = LayerLevel(layer=WG, zmin=0.0, thickness=0.2, bias=[0.01, 0.01], material="si")  # type: ignore[arg-type]
    dev_ok = StackDevice.from_layer_stack(comp, LayerStack(layers={"core": lvl_ok}), **kw)
    assert dev_ok.levels[0].bias == pytest.approx(0.01)

    lvl_aniso = LayerLevel(layer=WG, zmin=0.0, thickness=0.2, bias=[0.01, 0.02], material="si")  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="Anisotropic bias"):
        StackDevice.from_layer_stack(comp, LayerStack(layers={"core": lvl_aniso}), **kw)


# --------------------------------------------------------------------------- E-BF-13 (M2 Option A)


def test_gaussian_source_default_radius_uses_snapped_span() -> None:
    """M2 Option A: default GaussianPlaneSource radius comes from grid-snapped spans on non-uniform grids."""
    comp = gf.components.straight(length=2.0, width=0.4)
    ls = LayerStack(layers={"core": LayerLevel(layer=WG, zmin=0.0, thickness=0.22, material="si", mesh_order=1)})
    dev = StackDevice.from_layer_stack(
        comp,
        ls,
        {"air": MATERIALS["air"], "si": MATERIALS["si"]},
        background="air",
        zrange_layers=["core"],
        boundary_offset=(0.0, 0.0, 0.5, -0.5, 0.4, -0.4),
    )
    # Unsnapped extents with padding=(0.1, 0.1) at lambda=1.55:
    # y: 0.4 + 0.31 = 0.71 um -> faces +-0.355 snap to +-0.45 -> span 0.90 um
    # z: 0.22 + 0.31 = 0.53 um around 0.11 -> faces -0.155..0.375 snap to -0.35..0.57 -> span 0.92 um
    x_um = np.linspace(0.0, 2.0, 11)
    y_um = np.array([-0.70, -0.45, -0.05, 0.05, 0.45, 0.70])
    z_um = np.array([-0.40, -0.35, 0.11, 0.57, 0.62])
    rgrid = dev.rectilinear_grid(0.2, x_um=x_um, y_um=y_um, z_um=z_um)
    wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
    src, _ = dev.create_gaussian_plane_source_at("o1", (0.1, 0.1), wave_character=wave, grid=rgrid)
    assert float(src.radius) == pytest.approx(0.5 * 0.90e-6, rel=1e-6)


def test_point_dipole_polarization_validation() -> None:
    """create_point_dipole_source_at rejects bool/non-int types and out-of-range axes."""
    comp = gf.components.straight(length=1.0, width=0.4)
    ls = LayerStack(layers={"core": LayerLevel(layer=WG, zmin=0.0, thickness=0.2, material="si", mesh_order=1)})
    dev = StackDevice.from_layer_stack(
        comp, ls, {"air": MATERIALS["air"], "si": MATERIALS["si"]}, background="air", zrange_layers=["core"]
    )
    wave = fdtdx.WaveCharacter(wavelength=1.55e-6)

    for bad_type in (True, False, 1.5, [0]):
        with pytest.raises(TypeError, match="polarization"):
            dev.create_point_dipole_source_at(
                "o1", wave_character=wave, polarization=bad_type  # type: ignore[arg-type]
            )
    for bad_val in ("w", "xy", -1, 3):
        with pytest.raises(ValueError, match="polarization"):
            dev.create_point_dipole_source_at(
                "o1", wave_character=wave, polarization=bad_val  # type: ignore[arg-type]
            )

    for pol, want in ((0, 0), (1, 1), (2, 2), ("x", 0), ("y", 1), ("z", 2)):
        src, _ = dev.create_point_dipole_source_at("o1", wave_character=wave, polarization=pol)
        assert src.polarization == want


def test_plane_frame_port_offset_moves_along_in_and_outward_positive() -> None:
    """M2 Option A: plane port_offset > 0 shifts along 'in' (-axis) with direction='-'."""
    comp = _poly_comp({WG: [_rect(-0.5, -0.5, 0.5, 0.5)]})
    ls = LayerStack(layers={"core": LayerLevel(layer=WG, zmin=0.0, thickness=0.2, material="si", mesh_order=1)})
    dev = StackDevice.from_layer_stack(
        comp,
        ls,
        {"air": MATERIALS["air"], "si": MATERIALS["si"]},
        background="air",
        zrange_layers=["core"],
        boundary_offset=(0.2, -0.2, 0.2, -0.2, 0.2, -0.2),
    )
    wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
    for plane, axis in (("yz", 0), ("xz", 1), ("xy", 2)):
        s0, c0 = dev.create_mode_plane_source_at(
            plane, (0.0, 0.0), wave_character=wave, port_offset=0.0, direction="in"
        )
        sp, cp = dev.create_mode_plane_source_at(
            plane, (0.0, 0.0), wave_character=wave, port_offset=0.25, direction="in"
        )
        sn, cn = dev.create_mode_plane_source_at(
            plane, (0.0, 0.0), wave_character=wave, port_offset=-0.15, direction="out"
        )
        assert s0.direction == "-" and sp.direction == "-" and sn.direction == "+"
        assert float(cp[0].margins[axis]) - float(c0[0].margins[axis]) == pytest.approx(-0.25e-6, abs=1e-12)
        assert float(cn[0].margins[axis]) - float(c0[0].margins[axis]) == pytest.approx(0.15e-6, abs=1e-12)


# --------------------------------------------------------------------------- I-BF-02..05


def test_interface_status_returns_python_scalars_and_matches_write() -> None:
    """interface_status uses JIT core and returns Python int/float scalars matching write_with_status."""
    comp = _poly_comp({WG: [_rect(-0.24, -0.22, 0.24, 0.22)]})
    ls = LayerStack(layers={"core": LayerLevel(layer=WG, zmin=0.0, thickness=0.16, material="si", mesh_order=1)})
    dev = StackDevice.from_layer_stack(
        comp,
        ls,
        {"air": MATERIALS["air"], "si": MATERIALS["si"]},
        background="air",
        zrange_layers=["core"],
        boundary_offset=(0.12, -0.12, 0.12, -0.12, 0.08, -0.08),
        averaging="subpixel_full",
    )
    objects, arrays, params, _ = _place(dev)
    placed = objects["stack"]
    st = placed.interface_status(params["stack"])
    _, wst = placed.write_with_status(arrays, params["stack"])
    for key in ("junctions", "junction_overflow", "corner_dropped", "corner_uncertified", "junction_voxels"):
        assert type(st[key]) is int
    assert type(st["closure_residual"]) is float
    for key in ("junctions", "junction_overflow", "corner_dropped", "corner_uncertified"):
        assert st[key] == int(wst[key])
    assert st["closure_residual"] == pytest.approx(float(wst["closure_residual"]), rel=1e-6, abs=1e-7)


def test_write_dispersion_yee_weights_and_slot_consistency() -> None:
    """Dispersive pole alignment and _write_dispersion handle 1/3/9 coupling components with Yee weights."""
    pole = SlotPole(resonance=2.0e15, damping_rate=1.0e13, strength=1.0e30)
    mats = {
        "air": MATERIALS["air"],
        "disp": fdtdx.Material(permittivity=2.1, dispersion=DispersionModel(poles=(pole,))),
    }
    comp = _poly_comp({WG: [_rect(-0.20, -0.20, 0.02, 0.20)]})
    ls = LayerStack(layers={"core": LayerLevel(layer=WG, zmin=0.0, thickness=0.16, material="disp", mesh_order=1)})
    dev = StackDevice.from_layer_stack(
        comp,
        ls,
        mats,
        background="air",
        zrange_layers=["core"],
        boundary_offset=(0.20, -0.14, 0.12, -0.12, 0.08, -0.08),
        averaging="subpixel",
        yee_staggered=True,
    )
    objects, arrays, params, config = _place(dev)
    placed = objects["stack"]
    assert len(pole_slots(placed.materials)) == 1
    out = placed.write(arrays, params["stack"])
    assert out.dispersive_c1 is not None and bool(jnp.all(jnp.isfinite(out.dispersive_c3)))

    w_yee = [placed.material_weights(params["stack"], shift=s)[1] for s in _YEE_E_SHIFTS]
    with pytest.raises(ValueError, match="yee_weights"):
        placed._write_dispersion(arrays, w_yee[0], yee_weights=(w_yee[0],))

    dtype = out.dispersive_c1.dtype
    n_slots, n_comp = out.dispersive_c1.shape[:2]
    dt = config.time_step_duration
    for nc in (3, 9):
        arr_nc = out.at["dispersive_c3"].set(jnp.zeros((n_slots, nc, *out.inv_permittivities.shape[1:]), dtype=dtype))
        written = placed._write_dispersion(arr_nc, w_yee[0], yee_weights=w_yee)
        c3_box = np.asarray(written.dispersive_c3[:, :, *placed.grid_slice])
        _, _, a3 = compute_allowed_dispersive_coefficients(
            placed.materials, dt, max_num_poles=1, num_components=n_comp, coupling_components=nc
        )
        assert c3_box.shape[1] == nc
        assert np.all(np.isfinite(c3_box))
        assert float(np.abs(c3_box).max()) > 0.0
        if nc == 3:
            want_0 = np.einsum("ms,mxyz->sxyz", np.asarray(a3)[:, :, 0], np.asarray(w_yee[0]))
            want_1 = np.einsum("ms,mxyz->sxyz", np.asarray(a3)[:, :, 1], np.asarray(w_yee[1]))
            np.testing.assert_allclose(c3_box[:, 0], want_0, rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(c3_box[:, 1], want_1, rtol=1e-5, atol=1e-7)


def test_reapply_cache_bounded_and_avoids_stack_materials_retention() -> None:
    """apply_stack reuses mode solves without mutating objects with _stack_materials and bounds cache at 32."""
    _reapply_cache.cache_clear()
    try:
        comp = gf.components.straight(length=1.2, width=0.5)
        ls = LayerStack(layers={"core": LayerLevel(layer=WG, zmin=0.0, thickness=0.22, material="si", mesh_order=1)})
        dev = StackDevice.from_layer_stack(
            comp,
            ls,
            {"air": MATERIALS["air"], "si": MATERIALS["si"]},
            background="air",
            zrange_layers=["core"],
            extend_ports=(("o1", 0.3), ("o2", 0.3)),
            boundary_offset=(0.0, 0.0, 0.5, -0.5, 0.39, -0.39),
            dtype=jnp.float64,
        )
        wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
        src, src_c = dev.create_mode_plane_source_at("o1", (0.15, 0.15), wave_character=wave, port_offset=-0.1)
        det, det_c = dev.create_mode_overlap_detector_at("o2", (0.15, 0.15), wave_characters=(wave,))
        assert det.dtype == jnp.complex128
        objects, arrays, params, _ = _place(dev, spacing=0.05, extra=(src, det), extra_constraints=(*src_c, *det_c))

        key = jax.random.PRNGKey(0)
        _, objs1, info1 = apply_stack(arrays, objects, params, key)
        assert set(info1["stack_reapplied"]) == {src.name, det.name}
        assert info1["stack_reused"] == []
        assert not hasattr(objs1[src.name], "_stack_materials")
        assert not hasattr(objs1[det.name], "_stack_materials")

        _, _, info2 = apply_stack(arrays, objects, params, key)
        assert info2["stack_reapplied"] == []
        assert set(info2["stack_reused"]) == {src.name, det.name}

        assert _reapply_cache.cache_info().maxsize == 32
        for i in range(40):
            tok = _PENDING_REAPPLY.set(lambda i=i: i)
            try:
                _reapply_cache(("cfg", i), ("digest", i))
            finally:
                _PENDING_REAPPLY.reset(tok)
        assert _reapply_cache.cache_info().currsize == 32
    finally:
        _reapply_cache.cache_clear()


def test_sloped_sidewall_float32_matches_float64_precision() -> None:
    """_mean_shift keeps sloped sidewall fractions accurate in float32 relative to float64."""
    comp = _poly_comp({WG: [_rect(-0.3, -0.2, 0.3, 0.2)]})
    ls = LayerStack(
        layers={
            "core": LayerLevel(
                layer=WG, zmin=0.01, thickness=0.20, sidewall_angle=20.0, width_to_z=0.5, material="si", mesh_order=1
            )
        }
    )
    kw = dict(
        materials={"air": MATERIALS["air"], "si": MATERIALS["si"]},
        background="air",
        zrange_layers=["core"],
        boundary_offset=(0.16, -0.16, 0.16, -0.16, 0.08, -0.08),
    )
    objs32, _, params32, _ = _place(StackDevice.from_layer_stack(comp, ls, dtype=jnp.float32, **kw), spacing=0.02)
    objs64, _, params64, _ = _place(StackDevice.from_layer_stack(comp, ls, dtype=jnp.float64, **kw), spacing=0.02)
    w32 = np.asarray(objs32["stack"].material_weights(params32["stack"], apply_roughness=False)[1], dtype=float)
    w64 = np.asarray(objs64["stack"].material_weights(params64["stack"], apply_roughness=False)[1], dtype=float)
    np.testing.assert_allclose(w32, w64, atol=2e-5)
