"""Tests for the exact Green's-theorem voxelizer and unified subpixel pipeline."""

from __future__ import annotations

import math
from typing import Any

import fdtdx
import gdsfactory as gf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import shapely
from shapely import Polygon
from shapely.ops import unary_union

import bifrost as bf
from bifrost import greens, subpixel
from bifrost.geometry import _boundary_links, _edge_topology, pack_polygons, split_t_junctions


# ---------------------------------------------------------------- helpers


def _shoelace_moments_single(ring_xy: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> tuple[float, float, float]:
    """Exact area, x-moment about cell center, y-moment about cell center of one polygon ring."""
    x = ring_xy[:, 0]
    y = ring_xy[:, 1]
    xn = np.roll(x, -1)
    yn = np.roll(y, -1)
    cross = x * yn - xn * y
    area = 0.5 * float(np.sum(cross))
    if abs(area) < 1e-18:
        return 0.0, 0.0, 0.0
    ix = float(np.sum(cross * (x + xn))) / 6.0
    iy = float(np.sum(cross * (y + yn))) / 6.0
    xc = 0.5 * (x0 + x1)
    yc = 0.5 * (y0 + y1)
    return area, ix - xc * area, iy - yc * area


def _shapely_cell_moments(
    geom: Any,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference area fraction and standardized first moments per cell via Shapely + shoelace."""
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    fill = np.zeros((nx, ny), dtype=np.float64)
    mx = np.zeros((nx, ny), dtype=np.float64)
    my = np.zeros((nx, ny), dtype=np.float64)
    sqrt12 = math.sqrt(12.0)
    for i in range(nx):
        x0, x1 = float(x_edges[i]), float(x_edges[i + 1])
        dx = x1 - x0
        for j in range(ny):
            y0, y1 = float(y_edges[j]), float(y_edges[j + 1])
            dy = y1 - y0
            inter = geom.intersection(shapely.box(x0, y0, x1, y1))
            if inter.is_empty:
                continue
            polys = list(inter.geoms) if hasattr(inter, "geoms") else [inter]
            a_sum, ix_sum, iy_sum = 0.0, 0.0, 0.0
            for p in polys:
                if p.geom_type != "Polygon" or p.area <= 0:
                    continue
                ext = shapely.geometry.polygon.orient(p, sign=1.0)
                a0, ix0, iy0 = _shoelace_moments_single(np.asarray(ext.exterior.coords[:-1]), x0, x1, y0, y1)
                a_sum += a0
                ix_sum += ix0
                iy_sum += iy0
                for hole in ext.interiors:
                    ah, ixh, iyh = _shoelace_moments_single(np.asarray(hole.coords[:-1]), x0, x1, y0, y1)
                    a_sum += ah
                    ix_sum += ixh
                    iy_sum += iyh
            c_area = dx * dy
            fill[i, j] = np.clip(a_sum / c_area, 0.0, 1.0)
            mx[i, j] = (sqrt12 / dx) * (ix_sum / c_area)
            my[i, j] = (sqrt12 / dy) * (iy_sum / c_area)
    return fill, mx, my


def _polys_to_segments(polys: list[np.ndarray], tol: float = 1e-9, dtype: Any = jnp.float64) -> tuple[jax.Array, jax.Array, jax.Array]:
    v, sizes = pack_polygons(polys)
    split, split_sizes = split_t_junctions(v, sizes, tol)
    v_s = v[split]
    nxt, ids, boundary = _edge_topology(v_s, split_sizes, tol)
    src, dst, _ = _boundary_links(v_s, split_sizes, nxt, ids, boundary, tol)
    bnd = np.flatnonzero(boundary)
    p0 = jnp.asarray(v_s[src[bnd]], dtype=dtype)
    p1 = jnp.asarray(v_s[dst[bnd]], dtype=dtype)
    valid = jnp.ones((len(bnd),), dtype=bool)
    return p0, p1, valid


def _place_test_stack(
    polys: dict[int, list[np.ndarray]],
    levels: Any,
    mats: dict[str, fdtdx.Material],
    box_size: tuple[float, float, float],
    res_um: float,
    *,
    dtype: Any = jnp.float32,
    boundary_offset: tuple[float, float, float, float, float, float] | None = None,
    **stack_kwargs: Any,
) -> tuple[bf.StackDevice, Any, Any, dict[str, jax.Array]]:
    c = gf.Component()
    for lay, plist in polys.items():
        for pts in plist:
            c.add_polygon(pts, layer=(lay, 0))
    znames = list(levels.layers.keys())
    if boundary_offset is None:
        all_pts = np.concatenate([pts for plist in polys.values() for pts in plist])
        x0, y0 = all_pts.min(axis=0)
        x1, y1 = all_pts.max(axis=0)
        dx_extra = max(0.0, 0.5 * (box_size[0] - (x1 - x0)))
        dy_extra = max(0.0, 0.5 * (box_size[1] - (y1 - y0)))
        z_lo = min(lv.zmin for lv in levels.layers.values())
        z_hi = max(lv.zmin + lv.thickness for lv in levels.layers.values())
        dz_extra = max(0.0, 0.5 * (box_size[2] - (z_hi - z_lo)))
        boundary_offset = (dx_extra, -dx_extra, dy_extra, -dy_extra, dz_extra, -dz_extra)

    stack = bf.StackDevice(
        c,
        levels,
        mats,
        background="clad",
        zrange_layers=znames,
        boundary_offset=boundary_offset,
        name="stack",
        dtype=dtype,
        **stack_kwargs,
    )
    cfg = stack.get_simulation_config(
        time=10e-15,
        grid=fdtdx.core.grid.UniformGrid(spacing=res_um * 1e-6),
    )
    vol = fdtdx.SimulationVolume(
        partial_real_shape=stack.partial_real_shape,
        material=mats["clad"],
    )
    objects, arrays, params, cfg, _ = fdtdx.place_objects(
        object_list=[vol, stack],
        config=cfg,
        constraints=[
            stack.place_relative_to(vol, axes=(0, 1, 2), own_positions=(0, 0, 0), other_positions=(0, 0, 0))
        ],
        key=jax.random.PRNGKey(0),
    )
    return objects["stack"], arrays, cfg, {**params, **bf.init_stack_params(objects)}["stack"]


# ---------------------------------------------------------------- 2D Green's vs Shapely


@pytest.mark.parametrize("use_f32", [False, True])
def test_rasterize_segments_exact_vs_shapely(use_f32: bool) -> None:
    """Exact areas & moments vs Shapely across multi-vertex single cells, acute tips, holes, T-junctions & outliers."""
    dtype = jnp.float32 if use_f32 else jnp.float64
    atol = 5e-6 if use_f32 else 1e-11
    x_edges = np.linspace(-0.2, 1.0, 7)  # dx = 0.2
    y_edges = np.linspace(-0.3, 0.9, 7)  # dy = 0.2
    res_um = 0.2

    # 1. Rotated rectangle extending slightly outside box
    rect = np.array([[-0.5, -0.1], [0.75, -0.05], [0.65, 0.55], [-0.4, 0.45]])
    # 2. 32-vertex circle completely inside single voxel cell [0.6, 0.8] x [0.5, 0.7]
    th = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)
    circle = np.stack([0.70 + 0.07 * np.cos(th), 0.60 + 0.07 * np.sin(th)], axis=1)
    # 3. 5-degree acute needle tip
    ang = math.radians(5.0)
    needle = np.array([[-0.05, 0.78], [0.95, 0.78 + 1.0 * math.tan(0.5 * ang)], [-0.05, 0.78 + 2.0 * math.tan(0.5 * ang)]])

    for polys_case, name in [
        ([rect], "rotated_outlier"),
        ([circle], "32v_single_cell"),
        ([needle], "5deg_needle"),
    ]:
        p0, p1, valid = _polys_to_segments(polys_case, dtype=dtype)
        f, mx, my, _ = greens.rasterize_segments_2d(p0, p1, valid, x_edges, y_edges, res_um)
        ref_f, ref_mx, ref_my = _shapely_cell_moments(Polygon(polys_case[0]), x_edges, y_edges)
        np.testing.assert_allclose(np.asarray(f), ref_f, atol=atol, err_msg=f"{name} fill")
        np.testing.assert_allclose(np.asarray(mx), ref_mx, atol=atol, err_msg=f"{name} mx")
        np.testing.assert_allclose(np.asarray(my), ref_my, atol=atol, err_msg=f"{name} my")

    # 4. Polygon with hole + KLayout keyhole cut-line + edges on exact grid lines
    outer_box = np.array([[0.0, 0.1], [0.8, 0.1], [0.8, 0.7], [0.0, 0.7], [0.0, 0.3], [0.2, 0.3], [0.2, 0.5], [0.6, 0.5], [0.6, 0.3], [0.0, 0.3]])
    # Note: KLayout keyhole self-touching seam from (0.0, 0.3) <-> (0.2, 0.3) cancels cleanly
    p0, p1, valid = _polys_to_segments([outer_box], dtype=dtype)
    f, mx, my, _ = greens.rasterize_segments_2d(p0, p1, valid, x_edges, y_edges, res_um)
    hole_poly = Polygon([(0.0, 0.1), (0.8, 0.1), (0.8, 0.7), (0.0, 0.7)], holes=[[(0.2, 0.3), (0.6, 0.3), (0.6, 0.5), (0.2, 0.5)]])
    ref_f, ref_mx, ref_my = _shapely_cell_moments(hole_poly, x_edges, y_edges)
    np.testing.assert_allclose(np.asarray(f), ref_f, atol=atol, err_msg="keyhole fill")
    np.testing.assert_allclose(np.asarray(mx), ref_mx, atol=atol, err_msg="keyhole mx")
    np.testing.assert_allclose(np.asarray(my), ref_my, atol=atol, err_msg="keyhole my")

    # 5. Unsplit abutting T-junction polygons (one vertical trunk, one horizontal bar touching middle)
    bar = np.array([[-0.1, 0.0], [0.4, 0.0], [0.4, 0.5], [-0.1, 0.5]])
    arm = np.array([[0.4, 0.15], [0.85, 0.15], [0.85, 0.35], [0.4, 0.35]])
    p0, p1, valid = _polys_to_segments([bar, arm], dtype=dtype)
    f, mx, my, _ = greens.rasterize_segments_2d(p0, p1, valid, x_edges, y_edges, res_um)
    union_geom = unary_union([Polygon(bar), Polygon(arm)])
    ref_f, ref_mx, ref_my = _shapely_cell_moments(union_geom, x_edges, y_edges)
    np.testing.assert_allclose(np.asarray(f), ref_f, atol=atol, err_msg="t_junction fill")
    np.testing.assert_allclose(np.asarray(mx), ref_mx, atol=atol, err_msg="t_junction mx")
    np.testing.assert_allclose(np.asarray(my), ref_my, atol=atol, err_msg="t_junction my")


# ---------------------------------------------------------------- offsets & collapsed shrinks


def test_offset_boundary_segments_round_mitre_and_collapse() -> None:
    """Convex grown round arcs, shrunk exact mitres, reflex fillets, and collapsed shrink SDF fallback."""
    sq = np.array([[-0.25, -0.25], [0.25, -0.25], [0.25, 0.25], [-0.25, 0.25]])
    v, sizes = pack_polygons([sq])
    nxt, ids, boundary = _edge_topology(v, sizes, 1e-9)
    src, dst, succ = _boundary_links(v, sizes, nxt, ids, boundary, 1e-9)
    v_j = jnp.asarray(v, dtype=jnp.float64)
    edges = np.linspace(-0.6, 0.6, 13)  # 0.1 um cells

    # Grow convex square by +0.1 um -> area = 0.5^2 + 4*0.5*0.1 + pi*0.1^2 (12 chords/corner)
    p0, p1, val, inv = greens.offset_boundary_segments(v_j, src, dst, succ, boundary, 0.1, n_arc=12)
    assert not bool(inv)
    f, _, _, b = greens.rasterize_segments_2d(p0, p1, val, edges, edges, 0.1)
    area = float(jnp.sum(f)) * (0.1 * 0.1)
    exact_round_area = 0.25 + 4 * 0.05 + math.pi * 0.01
    # 12 chords per 90-deg corner has area accuracy ~1e-4
    np.testing.assert_allclose(area, exact_round_area, atol=2e-4)

    # Shrink convex square by -0.1 um -> exact 0.3 x 0.3 mitre square
    p0s, p1s, vals, invs = greens.offset_boundary_segments(v_j, src, dst, succ, boundary, -0.1, n_arc=12)
    assert not bool(invs)
    fs, _, _, _ = greens.rasterize_segments_2d(p0s, p1s, vals, edges, edges, 0.1)
    np.testing.assert_allclose(float(jnp.sum(fs)) * 0.01, 0.09, atol=1e-11)

    # Collapsed shrink (shrink square of half-width 0.25 by -0.30) triggers has_inversion
    _, _, _, inv_col = greens.offset_boundary_segments(v_j, src, dst, succ, boundary, -0.30)
    assert bool(inv_col)

    # Verify collapsed shrink on StackDevice stays in [0, 1] with finite AD gradients via SDF fallback
    lstack = gf.technology.LayerStack(
        layers={
            "core": gf.technology.LayerLevel(
                layer=(1, 0),
                thickness=0.22,
                zmin=0.0,
                material="si",
                sidewall_angle=70.0,  # large positive sidewall angle shrinks top beyond narrow base
                width_to_z=0.0,
            )
        }
    )
    mats = {"si": fdtdx.Material(permittivity=12.0), "clad": fdtdx.Material(permittivity=1.0)}
    narrow = np.array([[-0.08, -0.4], [0.08, -0.4], [0.08, 0.4], [-0.08, 0.4]])
    d, _, _, p = _place_test_stack(
        {1: [narrow]}, lstack, mats, (1.0, 1.0, 0.24), 0.04, dtype=jnp.float64, averaging="subpixel"
    )
    _, w = d.material_weights(p)
    assert np.all(np.isfinite(w)) and float(jnp.min(w)) >= -1e-12 and float(jnp.max(w)) <= 1.0 + 1e-12
    vkey = d.source_layers[0][0]
    g = jax.grad(lambda params: jnp.sum(d.material_weights(params)[1][0]))(p)
    assert np.all(np.isfinite(g[vkey]))


# ---------------------------------------------------------------- multi-level, sloped walls, roughness, rectilinear


def test_multilevel_mesh_order_and_overlap() -> None:
    """Priority order (lower mesh_order wins) + 4x4 crossing refinement inside a single voxel."""
    lstack = gf.technology.LayerStack(
        layers={
            "high_prio": gf.technology.LayerLevel(layer=(1, 0), thickness=0.2, zmin=0.0, material="si", mesh_order=1),
            "low_prio": gf.technology.LayerLevel(layer=(2, 0), thickness=0.2, zmin=0.0, material="sin", mesh_order=2),
        }
    )
    mats = {
        "si": fdtdx.Material(permittivity=12.0),
        "sin": fdtdx.Material(permittivity=4.0),
        "clad": fdtdx.Material(permittivity=1.0),
    }
    p1 = np.array([[-0.3, -0.05], [0.3, -0.05], [0.3, 0.3], [-0.3, 0.3]])
    p2 = np.array([[-0.05, -0.3], [0.3, -0.3], [0.3, 0.3], [-0.05, 0.3]])
    d, _, _, p = _place_test_stack({1: [p1], 2: [p2]}, lstack, mats, (0.8, 0.8, 0.2), 0.1, dtype=jnp.float64)
    names, w = d.material_weights(p)
    # Volume totals across the 2D slab must match Shapely union breakdown: si = |p1|, sin = |p2 \ p1|
    g1, g2 = Polygon(p1), Polygon(p2)
    voxel_vol = 0.1**3
    tot_si = float(jnp.sum(w[names.index("si")])) * voxel_vol
    tot_sin = float(jnp.sum(w[names.index("sin")])) * voxel_vol
    np.testing.assert_allclose(tot_si, g1.area * 0.2, atol=1e-4)
    np.testing.assert_allclose(tot_sin, g2.difference(g1).area * 0.2, atol=5e-4)


def test_sidewall_z_to_bias_and_roughness() -> None:
    """Sloped walls with z_to_bias knots and TopRoughness keep valid weights, dyads & AD."""
    lstack = gf.technology.LayerStack(
        layers={
            "core": gf.technology.LayerLevel(
                layer=(1, 0),
                thickness=0.22,
                zmin=0.0,
                material="si",
                sidewall_angle=10.0,
                width_to_z=0.5,
                z_to_bias=((0.0, 0.4, 1.0), (0.02, -0.01, 0.0)),
            )
        }
    )
    mats = {"si": fdtdx.Material(permittivity=12.0), "clad": fdtdx.Material(permittivity=1.0)}
    wg = np.array([[-0.6, -0.25], [0.6, -0.25], [0.6, 0.25], [-0.6, 0.25]])
    rough = (bf.TopRoughness(level="core", rms=0.01, correlation_length=0.1),)
    d, arrays, _, p = _place_test_stack(
        {1: [wg]},
        lstack,
        mats,
        (1.2, 0.8, 0.3),
        0.05,
        dtype=jnp.float32,
        averaging="subpixel_full",
        roughness=rough,
    )
    out, status = d.write_with_status(arrays, p)
    assert not bool(status["failed"]) and int(status["junction_overflow"]) == 0
    assert bool(jnp.all(jnp.isfinite(out.inv_permittivities[:, *d.grid_slice])))


def test_rectilinear_edge_width_and_subsamples() -> None:
    """Graded RectilinearGrid, xy_subsamples=2, and edge_width=1.2 work with greens."""
    from fdtdx.core.grid import RectilinearGrid

    mats = {"si": fdtdx.Material(permittivity=12.0), "clad": fdtdx.Material(permittivity=1.0)}
    lstack = gf.technology.LayerStack(
        layers={"core": gf.technology.LayerLevel(layer=(1, 0), thickness=0.2, zmin=0.0, material="si")}
    )
    poly = np.array([[-0.6, -0.22], [0.6, -0.22], [0.6, 0.22], [-0.6, 0.22]])
    c = gf.Component()
    c.add_polygon(poly, layer=(1, 0))
    stack = bf.StackDevice(
        c,
        lstack,
        mats,
        background="clad",
        zrange_layers=["core"],
        boundary_offset=(0.2, -0.2, 0.28, -0.28, 0.1, -0.1),
        name="stack",
        averaging="subpixel_full",
        xy_subsamples=2,
        junction_cells=(4, 4, 4),
    )
    xe = np.array([-0.8, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.8])
    ye = np.array([-0.5, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5])
    ze = np.array([-0.1, -0.02, 0.05, 0.15, 0.22, 0.3])
    rgrid = RectilinearGrid(
        x_edges=jnp.asarray(xe * 1e-6, dtype=jnp.float32),
        y_edges=jnp.asarray(ye * 1e-6, dtype=jnp.float32),
        z_edges=jnp.asarray(ze * 1e-6, dtype=jnp.float32),
    )
    cfg = stack.get_simulation_config(time=10e-15, grid=rgrid)
    vol = fdtdx.SimulationVolume(partial_real_shape=stack.partial_real_shape, material=mats["clad"])
    objs, arrays, _, _, _ = fdtdx.place_objects(
        object_list=[vol, stack],
        config=cfg,
        constraints=[stack.place_relative_to(vol, axes=(0, 1, 2), own_positions=(0, 0, 0), other_positions=(0, 0, 0))],
        key=jax.random.PRNGKey(0),
    )
    d_rect = objs["stack"]
    out, status = d_rect.write_with_status(arrays)
    assert not bool(status["failed"]) and bool(jnp.all(jnp.isfinite(out.inv_permittivities[:, *d_rect.grid_slice])))
    _, w = d_rect.material_weights()
    np.testing.assert_allclose(np.asarray(jnp.sum(w, axis=0)), 1.0, atol=1e-6)

    # Also check edge_width=1.2 + xy_subsamples=2 non-contiguous Green's clipping
    d, arrays_ew, _, p = _place_test_stack(
        {1: [poly]}, lstack, mats, (1.6, 1.0, 0.4), 0.05, xy_subsamples=2, edge_width=1.2, averaging="subpixel"
    )
    out_ew = d.write(arrays_ew, p)
    assert bool(jnp.all(jnp.isfinite(out_ew.inv_permittivities[:, *d.grid_slice])))
    _, w_ew = d.material_weights(p)
    np.testing.assert_allclose(np.asarray(jnp.sum(w_ew, axis=0)), 1.0, atol=1e-6)


# ---------------------------------------------------------------- unified subpixel & dynamic junctions


def test_unified_subpixel_agreement_wiener_bounds_and_dynamic_junctions() -> None:
    """Greens subpixel == subpixel_full on axis-aligned walls, Wiener SPD bounds everywhere, and no junction overflow when vertices grow."""
    mats = {"si": fdtdx.Material(permittivity=12.0), "clad": fdtdx.Material(permittivity=1.44**2)}
    lstack = gf.technology.LayerStack(
        layers={"core": gf.technology.LayerLevel(layer=(1, 0), thickness=0.20, zmin=0.0, material="si")}
    )
    # 1. Axis-aligned rectangle -> subpixel vs subpixel_full diagonal entries match on planar walls
    rect = np.array([[-0.6, -0.23], [0.6, -0.23], [0.6, 0.23], [-0.6, 0.23]])
    d_diag, a_diag, _, p_diag = _place_test_stack(
        {1: [rect]}, lstack, mats, (1.6, 1.0, 0.4), 0.05, averaging="subpixel", dtype=jnp.float64
    )
    d_full, a_full, _, p_full = _place_test_stack(
        {1: [rect]}, lstack, mats, (1.6, 1.0, 0.4), 0.05, averaging="subpixel_full", dtype=jnp.float64
    )
    inv_diag = d_diag.write(a_diag, p_diag).inv_permittivities[:, *d_diag.grid_slice]
    inv_full = d_full.write(a_full, p_full).inv_permittivities[:, *d_full.grid_slice]
    # Away from 2D top/bottom corner edges (z in {2, 6}, y in {5, 15}), planar Kottke agrees identically
    np.testing.assert_allclose(inv_diag[:, 8:-8, :, 3:6], inv_full[[0, 4, 8], 8:-8, :, 3:6], atol=1e-6)
    np.testing.assert_allclose(inv_diag[:, 8:-8, 7:13, :], inv_full[[0, 4, 8], 8:-8, 7:13, :], atol=1e-6)

    # 2. Rotated octagon -> check SPD, Wiener bounds, and dynamic scaling when polygon perimeter triples
    th = np.linspace(0, 2 * math.pi, 17)[:-1]
    oct_small = 0.25 * np.stack([np.cos(th), np.sin(th)], axis=1)
    d_oct, a_oct, _, p_oct = _place_test_stack(
        {1: [oct_small]}, lstack, mats, (1.8, 1.8, 0.3), 0.05, averaging="subpixel_full", dtype=jnp.float64
    )
    assert d_oct.junction_voxels is None
    st_small = d_oct.interface_status(p_oct)
    # Triple perimeter without re-placing
    vkey = d_oct.source_layers[0][0]
    p_big = {**p_oct, vkey: p_oct[vkey] * 3.0}
    out_big, st_big = d_oct.write_with_status(a_oct, p_big)
    assert int(st_big["junctions"]) > int(st_small["junctions"]) * 2
    assert int(st_big["junction_overflow"]) == 0
    assert float(st_big["closure_residual"]) < 1e-6

    # Check SPD and Wiener bounds at every Yee position for greens _complete_tensor
    _, eps = d_oct._material_table()
    inv_eps = np.linalg.inv(eps)
    for sh in bf.stack._YEE_E_SHIFTS:
        g = d_oct._interfaces(p_big, None, True, sh)
        inv6, _ = d_oct._complete_tensor(g, None)
        eff = np.linalg.inv(np.asarray(subpixel.from_six(inv6)).reshape(-1, 3, 3))
        w = np.moveaxis(np.asarray(g.weights), 0, -1).reshape(-1, len(eps))
        lo_bnd = np.linalg.inv(np.einsum("nm,mij->nij", w, inv_eps))
        hi_bnd = np.einsum("nm,mij->nij", w, eps)
        eig_lo = np.linalg.eigvalsh(eff - lo_bnd).min()
        eig_hi = np.linalg.eigvalsh(hi_bnd - eff).min()
        assert eig_lo >= -1e-9 and eig_hi >= -1e-9, (eig_lo, eig_hi)

    # 3. Explicit junction_voxels cap raises/warns when exceeded
    capped = d_oct.aset("junction_voxels", 4)
    with pytest.raises(ValueError, match="junction_voxels"):
        capped.write(a_oct, p_big)
    warn_dev = capped.aset("interface_failure", "warn")
    with pytest.warns(bf.StackDiagnosticWarning, match="junction_voxels"):
        warn_dev.write(a_oct, p_big)


# ---------------------------------------------------------------- grad & jvp vs finite differences


@pytest.mark.parametrize("averaging", ["subpixel", "subpixel_full"])
@pytest.mark.parametrize("angle0", [0.0, 12.0, -12.0])
def test_greens_grad_and_jvp_match_fd(averaging: str, angle0: float) -> None:
    """Both jax.grad and jax.jvp wrt vertices, zmin, thickness, and sidewall_angle match FD in float64."""
    mats = {"si": fdtdx.Material(permittivity=9.0), "clad": fdtdx.Material(permittivity=1.0)}
    lstack = gf.technology.LayerStack(
        layers={
            "core": gf.technology.LayerLevel(
                layer=(1, 0), thickness=0.18, zmin=0.03, material="si", sidewall_angle=angle0
            )
        }
    )
    tri = np.array([[-0.22, -0.17], [0.27, -0.11], [0.04, 0.29]])
    d, arrays, _, p0 = _place_test_stack(
        {1: [tri]}, lstack, mats, (0.8, 0.8, 0.24), 0.08, averaging=averaging, dtype=jnp.float64
    )

    def loss(p_dict: dict[str, jax.Array]) -> jax.Array:
        out = d.write(arrays, p_dict).inv_permittivities[:, *d.grid_slice]
        # Asymmetric spatially weighted loss so all components (vertices, z, angle) have nonzero gradient
        nx, ny, nz = d.grid_shape
        wx = jnp.linspace(0.5, 1.5, nx)[:, None, None]
        wy = jnp.linspace(1.2, 0.7, ny)[None, :, None]
        wz = jnp.linspace(0.8, 1.6, nz)[None, None, :]
        wt = (wx * wy * wz)[None]
        return jnp.sum(out * wt)

    g = jax.jit(jax.grad(loss))(p0)
    rng = np.random.default_rng(7)
    vkey = d.source_layers[0][0]
    tangent = {
        vkey: jnp.asarray(rng.normal(scale=0.1, size=p0[vkey].shape), dtype=jnp.float64),
        "zmin/core": jnp.asarray(0.25, dtype=jnp.float64),
        "thickness/core": jnp.asarray(-0.3, dtype=jnp.float64),
        "sidewall_angle/core": jnp.asarray(1.0, dtype=jnp.float64),
    }
    _, jvp_val = jax.jit(lambda tang: jax.jvp(loss, (p0,), (tang,)))(tangent)
    grad_dot = sum(float(jnp.sum(g[k] * tangent[k])) for k in tangent)
    np.testing.assert_allclose(float(jvp_val), grad_dot, rtol=1e-8, atol=1e-10)

    eps_fd = 1e-6
    loss_0 = float(loss(p0))
    for key in [vkey, "zmin/core", "thickness/core", "sidewall_angle/core"]:
        if key == vkey:
            # Check inner vertex movement along x
            d_k = jnp.zeros_like(p0[key]).at[1, 0].set(1.0)
            g_k = float(g[key][1, 0])
        else:
            d_k = jnp.ones_like(p0[key])
            g_k = float(g[key])
        p_plus = {**p0, key: p0[key] + eps_fd * d_k}
        p_minus = {**p0, key: p0[key] - eps_fd * d_k}
        fd_k = float((loss(p_plus) - loss(p_minus)) / (2 * eps_fd))
        if angle0 == 0.0 and key == "sidewall_angle/core":
            fd_right = (float(loss(p_plus)) - loss_0) / eps_fd
            np.testing.assert_allclose(g_k, fd_right, rtol=1e-4, atol=1e-5, err_msg=f"{averaging} angle=0 right FD")
            # angle = 0 is a genuine kink (convex corners round for one sign of the angle, mitre for the
            # other), so the central FD averages two one-sided slopes; the bound is on the kink's size
            # (0.13% with the eps-space sub-cell blend, 0.21% with the inverse-space voxel blend)
            np.testing.assert_allclose(g_k, fd_k, rtol=3e-3, atol=1e-5, err_msg=f"{averaging} angle=0 central kink")
        else:
            np.testing.assert_allclose(g_k, fd_k, rtol=1e-4, atol=1e-5, err_msg=f"{averaging} angle={angle0} key={key}")


@pytest.mark.parametrize("derived_shapes", ["exact", "composite"])
def test_colocated_derived_levels_have_no_phantom_halo(derived_shapes: str) -> None:
    """Co-located boundaries (one = WG - ETCH, two = WG | ETCH) yield zero phantom second-level material."""
    from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer

    mats = {
        "si": fdtdx.Material(permittivity=12.0),
        "sio2": fdtdx.Material(permittivity=2.1),
        "clad": fdtdx.Material(permittivity=1.0),
    }
    derived = {"derived_layer": LogicalLayer(layer=(1, 0)), "zmin": 0.0, "thickness": 0.22}
    lstack = LayerStack(
        layers={
            "one": LayerLevel(
                layer=LogicalLayer(layer=(1, 0)) - LogicalLayer(layer=(2, 0)),
                material="si",
                mesh_order=1,
                **derived,
            ),
            "two": LayerLevel(
                layer=LogicalLayer(layer=(1, 0)) | LogicalLayer(layer=(2, 0)),
                material="sio2",
                mesh_order=2,
                **derived,
            ),
        }
    )
    # Angled strip on layer 1 (not grid-aligned) with ETCH on layer 2 cutting only the upper right end
    wg = np.array([[-0.35, -0.19], [0.45, -0.05], [0.41, 0.23], [-0.39, 0.09]])
    etch = np.array([[0.22, -0.25], [0.55, -0.25], [0.55, 0.35], [0.22, 0.35]])
    d, _, _, p0 = _place_test_stack(
        {1: [wg], 2: [etch]},
        lstack,
        mats,
        (1.2, 0.88, 0.28),
        0.04,
        averaging="subpixel",
        derived_shapes=derived_shapes,
        yee_staggered=False,
    )
    names, w = d.material_weights(p0)
    si_idx = names.index("si")
    sio2_idx = names.index("sio2")
    res_um = float(d._res_um())
    nx, ny, nz = d.grid_shape
    x_centers = (np.arange(nx) + 0.5 - 0.5 * nx) * res_um + float(d.center[0])
    k_mid = nz // 2
    away_x = x_centers < 0.10  # far left of ETCH (>= 0.22 um)
    w_si = np.asarray(w[si_idx, away_x, :, k_mid])
    w_sio2 = np.asarray(w[sio2_idx, away_x, :, k_mid])
    cut = (w_si > 0.05) & (w_si < 0.95)
    assert int(cut.sum()) > 5
    assert float(np.max(w_sio2[cut])) < 1e-5

    # AD gradient wrt WG vertices remains finite under polygon clipping
    vkey = next(k for k in p0 if k.startswith("vertices/"))

    def loss(p_dict: dict[str, jax.Array]) -> jax.Array:
        return jnp.sum(d.material_weights(p_dict)[1][si_idx])

    g = jax.jit(jax.grad(loss))(p0)
    assert bool(jnp.all(jnp.isfinite(g[vkey])))
    assert float(jnp.sum(jnp.abs(g[vkey]))) > 0.0


@pytest.mark.parametrize("averaging", ["subpixel", "subpixel_full"])
def test_large_polygon_jit_with_x64_enabled(averaging: str) -> None:
    """Polygons with >32 vertices compile and differentiate under jax_enable_x64=True multi-chunk SDF scan."""
    mats = {"si": fdtdx.Material(permittivity=12.0), "clad": fdtdx.Material(permittivity=1.0)}
    lstack = gf.technology.LayerStack(
        layers={
            "core": gf.technology.LayerLevel(layer=(1, 0), thickness=0.22, zmin=0.0, material="si", sidewall_angle=10.0)
        }
    )
    th = np.linspace(0.0, 2.0 * math.pi, 201)[:-1]
    disk = np.stack([0.35 * np.cos(th), 0.35 * np.sin(th)], axis=1)
    with jax.enable_x64(True):
        d, arrays, _, p0 = _place_test_stack(
            {1: [disk]}, lstack, mats, (1.2, 1.2, 0.3), 0.1, averaging=averaging, dtype=jnp.float32
        )
        vkey = next(k for k in p0 if k.startswith("vertices/"))

        def loss(p_dict: dict[str, jax.Array]) -> jax.Array:
            return jnp.sum(d.write(arrays, p_dict).inv_permittivities[:, *d.grid_slice])

        val, grad = jax.jit(jax.value_and_grad(loss))(p0)
        assert bool(jnp.isfinite(val)) and bool(jnp.all(jnp.isfinite(grad[vkey])))
        assert float(jnp.sum(jnp.abs(grad[vkey]))) > 0.0
