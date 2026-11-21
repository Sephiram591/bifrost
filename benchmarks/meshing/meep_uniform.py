"""Uniform-grid MEEP reference benchmarks using native ``mp.Prism`` and ``mp.Block`` geometry.

Exports only exact polygons, ``LayerLevel`` geometry metadata, material permittivities, and port
planes from each benchmark stack—never voxelized permittivity arrays from ``bifrost``—and runs
MEEP with its own subpixel smoothing (``eps_averaging=True``) on MPI CPUs.

Usage:
    python benchmarks/meshing/meep_uniform.py export --geom-dir <dir>
    mpirun -np 48 python benchmarks/meshing/meep_uniform.py run --geom-dir <dir> --bench sin_bend --dx-nm 30 --out shard.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

WT_ROOT = Path(__file__).resolve().parents[2]
C_UM_PER_S = 299792458.0 * 1e6

BENCHES = (
    "aniso_slab",
    "sloped_guide",
    "sin_bend",
    "elevator_coupler",
    "directional_coupler",
    "linbo3_bend",
)


def _sys_paths() -> None:
    for p in (
        WT_ROOT,
        WT_ROOT / "examples",
        WT_ROOT / "examples" / "guide",
        WT_ROOT / "benchmarks" / "subpixel",
        WT_ROOT / "benchmarks" / "meshing",
    ):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _extract_stack_geometry(
    stack: Any,
    layer_stack: Any,
    materials: dict[str, Any],
    background: str,
    *,
    wafer_layer: Any,
    extend_ends_um: float = 2.0,
) -> dict[str, Any]:
    """Extract pure polygon, layer-level, and bounding-box metadata without voxelizing."""
    import gdsfactory as gf

    cx, cy = float(stack.center[0]), float(stack.center[1])
    cz = float(stack.z_center)
    lx, ly, lz = (float(v) * 1e6 for v in stack.partial_real_shape)
    x0, x1 = cx - lx / 2.0, cx + lx / 2.0
    y0, y1 = cy - ly / 2.0, cy + ly / 2.0
    z0, z1 = cz - lz / 2.0, cz + lz / 2.0

    polygons = {
        index: np.split(np.asarray(v, dtype=float), np.cumsum(sizes)[:-1])
        for (_, index), v, sizes in zip(stack.source_layers, stack.source_vertices, stack.source_sizes, strict=True)
    }

    def _union_level_polygons(raw_polys: list[np.ndarray]) -> list[np.ndarray]:
        from shapely.geometry import Polygon
        from shapely.geometry.polygon import orient
        from shapely.ops import unary_union

        geoms = [Polygon(p) for p in raw_polys if len(p) >= 3]
        if not geoms:
            return []
        if len(geoms) == 1:
            merged = geoms[0]
        else:
            try:
                merged = unary_union(geoms, grid_size=1e-9)
            except TypeError:
                merged = unary_union(geoms)
        parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
        out: list[np.ndarray] = []
        for part in parts:
            if not isinstance(part, Polygon) or part.is_empty:
                continue
            if part.interiors:
                raise ValueError("Polygons with interior holes are not supported in MEEP export")
            out.append(np.asarray(orient(part, sign=1.0).exterior.coords[:-1], dtype=float))
        return out

    def _extend_boundary_caps(pts: np.ndarray) -> list[list[float]]:
        out = np.asarray(pts, dtype=float).copy()
        if extend_ends_um > 0:
            tol = 1e-4
            # Extend vertices lying strictly on open port-extended box boundaries outward so
            # MEEP mp.Prism(sidewall_angle=...) does not bevel waveguide ends inside the PML.
            on_x0 = np.abs(out[:, 0] - x0) < tol
            on_x1 = np.abs(out[:, 0] - x1) < tol
            on_y0 = np.abs(out[:, 1] - y0) < tol
            on_y1 = np.abs(out[:, 1] - y1) < tol
            out[on_x0, 0] -= extend_ends_um
            out[on_x1, 0] += extend_ends_um
            out[on_y0, 1] -= extend_ends_um
            out[on_y1, 1] += extend_ends_um
        return out.tolist()

    mats_out: dict[str, list[float]] = {}
    for mname, mat in materials.items():
        perm = np.asarray(mat.permittivity, dtype=float).reshape(-1)
        if perm.size == 1:
            eps9 = np.diag([perm[0], perm[0], perm[0]]).reshape(-1).tolist()
        elif perm.size == 3:
            eps9 = np.diag(perm).reshape(-1).tolist()
        elif perm.size == 9:
            eps9 = perm.tolist()
        else:
            raise ValueError(f"Unsupported permittivity shape for {mname!r}: {perm.shape}")
        mats_out[mname] = eps9

    levels_out: list[dict[str, Any]] = []
    for lv_name, lv in layer_stack.layers.items():
        raw_layer = lv.layer.layer if hasattr(lv.layer, "layer") else lv.layer
        idx = gf.get_layer(raw_layer)
        is_wafer = idx == wafer_layer or bool(lv.background)
        polys = None if is_wafer else [_extend_boundary_caps(p) for p in _union_level_polygons(polygons.get(idx, []))]
        levels_out.append(
            {
                "name": lv_name,
                "zmin": float(lv.zmin),
                "thickness": float(lv.thickness),
                "sidewall_angle": float(lv.sidewall_angle or 0.0),
                "width_to_z": float(lv.width_to_z or 0.0),
                "mesh_order": int(lv.mesh_order),
                "material": lv.material,
                "polygons": polys,
            }
        )

    return {
        "domain": [[x0, x1], [y0, y1], [z0, z1]],
        "background": background,
        "materials": mats_out,
        "levels": levels_out,
    }


def _port_plane(
    stack: Any,
    port: str,
    padding: tuple[float, float],
    wavelengths_um: list[float],
    port_offset: float,
) -> dict[str, Any]:
    from bifrost.stack._device import plane_extents

    p, axis, _, center, height = stack._port_frame(port, None, port_offset)
    width_um, height_um = plane_extents(p.width, height, padding, wavelengths_um)
    size = [width_um, height_um]
    size.insert(axis, 0.0)
    return {
        "axis": int(axis),
        "center": [float(c) for c in center],
        "size": [float(s) for s in size],
    }


def export_all(geom_dir: Path) -> None:
    """Export analytical geometry and benchmark specs for all 6 benchmarks (CPU, no voxelization)."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    _sys_paths()

    import fdtdx
    import fdtdx_directional_coupler as dc
    import fdtdx_elevator_coupler as ec
    import gdsfactory as gf
    import inverse_design as invd
    import linbo3_bend as lb
    from berreman import jones
    from gdsfactory.technology import LayerLevel, LayerStack

    import accuracy_per_cell as apc
    import bifrost as bf

    gf.gpdk.PDK.activate()
    gf.snap.SNAP_ENABLED = False
    geom_dir.mkdir(parents=True, exist_ok=True)
    wafer = gf.get_layer("WAFER")

    # 1. aniso_slab
    eps_slab = apc._aniso_slab_tensor()
    _, t_ref = jones([(eps_slab, 0.137)], 1.0, 1.0, 1.55)
    t_ref = t_ref * np.exp(-2j * np.pi / 1.55 * 0.137)
    aniso_spec = {
        "bench": "aniso_slab",
        "wavelength": 1.55,
        "thickness": 0.137,
        "slab_z": 0.013,
        "half": 1.8,
        "pml": 0.6,
        "time": 120e-15,
        "spectral_width_frac": 0.1,
        "z_src": -0.7,
        "z_trans": 0.75,
        "eps_tensor": eps_slab.tolist(),
        "t_ref_real": t_ref.real.tolist(),
        "t_ref_imag": t_ref.imag.tolist(),
    }
    (geom_dir / "geometry_aniso_slab.json").write_text(json.dumps(aniso_spec, indent=2))

    # 2. sloped_guide
    sg_box = ((-3.20, 3.20), (-1.60, 1.60), (-1.49, 1.71))
    (x0, x1), (y0, y1), (z0, z1) = sg_box
    sg_c = gf.Component()
    h = 0.25
    sg_c.add_polygon([(x0, -h), (x1, -h), (x1, h), (x0, h)], layer=gf.gpdk.LAYER.WG)
    sg_c.add_port(name="o1", center=(x0, 0.0), width=0.5, orientation=180, layer=gf.gpdk.LAYER.WG)
    sg_c.add_port(name="o2", center=(x1, 0.0), width=0.5, orientation=0, layer=gf.gpdk.LAYER.WG)
    sg_ls = LayerStack(
        layers={
            "core": LayerLevel(
                layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=0.22, material="si", mesh_order=1, sidewall_angle=10.0
            )
        }
    )
    sg_mats = {"si": fdtdx.Material(permittivity=3.48**2), "clad": fdtdx.Material(permittivity=1.444**2)}
    sg_stack = bf.StackDevice.from_layer_stack(
        sg_c, sg_ls, sg_mats, background="clad", zrange_layers=["core"], name="stack",
        diagnostics=False, boundary_offset=(0.0, 0.0, y1 - h, y0 + h, z1 - 0.22, z0),
    )
    sg_spec = {
        "bench": "sloped_guide",
        **_extract_stack_geometry(sg_stack, sg_ls, sg_mats, "clad", wafer_layer=wafer),
        "wavelength": 1.55,
        "pml": 1.0,
        "time": 200e-15,
        "spectral_width_frac": 0.1,
        "source": _port_plane(sg_stack, "o1", (0.6, 0.6), [1.55], 1.3),
        "line_x": [-1.4, 2.0],
        "line_y": 0.0,
        "line_z": 0.11,
        "trim": 0.3,
    }
    (geom_dir / "geometry_sloped_guide.json").write_text(json.dumps(sg_spec, indent=2))

    # 3. sin_bend
    z_margin_sb = (invd.BOX_HEIGHT - invd.THICKNESS) / 2
    sb_stack = bf.StackDevice.from_layer_stack(
        invd.tight_bend(invd.UNIFORM), invd.layer_stack, invd.materials, background="air",
        zrange_layers=["core"], extend_ports=(("o1", invd.EXTENSION), ("o2", invd.EXTENSION)),
        boundary_offset=(invd.MARGIN, 0.0, 0.0, -invd.MARGIN, z_margin_sb, -z_margin_sb),
        name="stack", diagnostics=False,
    )
    sb_spec = {
        "bench": "sin_bend",
        **_extract_stack_geometry(sb_stack, invd.layer_stack, invd.materials, "air", wafer_layer=wafer),
        "wavelength": float(invd.WAVELENGTH),
        "pml": float(invd.PML),
        "time": float(invd.DURATION),
        "spectral_width_frac": 0.05,
        "source": _port_plane(sb_stack, "o1", invd.PADDING, [invd.WAVELENGTH], -1.0),
        "in": _port_plane(sb_stack, "o1", invd.PADDING, [invd.WAVELENGTH], -0.5),
        "out": _port_plane(sb_stack, "o2", invd.PADDING, [invd.WAVELENGTH], -0.5),
    }
    (geom_dir / "geometry_sin_bend.json").write_text(json.dumps(sb_spec, indent=2))

    # 4. elevator_coupler
    ec_stack = bf.StackDevice.from_layer_stack(
        ec.elevator_coupler(3.80), ec.LAYER_STACK, ec.MATERIALS, background="air", name="stack",
        zrange_layers=["lower", "upper"], extend_ports=(("o1", ec.EXTENSION), ("o2", ec.EXTENSION)),
        boundary_offset=(0.0, 0.0, 2.0 - ec.W_MAX / 2, -2.0 + ec.W_MAX / 2, 1.60 - 0.19, -1.60 + 0.21),
        diagnostics=False,
    )
    ec_spec = {
        "bench": "elevator_coupler",
        **_extract_stack_geometry(ec_stack, ec.LAYER_STACK, ec.MATERIALS, "air", wafer_layer=wafer),
        "wavelength": float(ec.WAVELENGTH),
        "pml": float(ec.PML),
        "time": 400e-15,
        "spectral_width_frac": 0.05,
        "source": _port_plane(ec_stack, "o1", (1.838, 1.074), [ec.WAVELENGTH], -0.3),
        "in": _port_plane(ec_stack, "o1", (1.838, 1.074), [ec.WAVELENGTH], 0.0),
        "out": _port_plane(ec_stack, "o2", (1.838, 1.511), [ec.WAVELENGTH], 0.0),
    }
    (geom_dir / "geometry_elevator_coupler.json").write_text(json.dumps(ec_spec, indent=2))

    # 5. directional_coupler
    dc_stack = dc.build_stack()
    wls_dc = (1.0 / dc.frequencies()).tolist()
    dc_spec = {
        "bench": "directional_coupler",
        **_extract_stack_geometry(dc_stack, dc.LAYER_STACK, dc.MATERIALS, "clad", wafer_layer=wafer),
        "wavelength": float(dc.WAVELENGTH),
        "frequencies": dc.frequencies().tolist(),
        "pml": float(dc.PML),
        "time": float(dc.TIME),
        "spectral_width_frac": float(dc.PULSE_WIDTH),
        "source": _port_plane(dc_stack, "o1", dc.padding([fdtdx.WaveCharacter(wavelength=dc.WAVELENGTH * 1e-6)]), [dc.WAVELENGTH], -0.3),
        "in": _port_plane(dc_stack, "o1", ((dc.PLANE[0] - dc.WIDTH) / 2 / np.mean(wls_dc), (dc.PLANE[1] - dc.HEIGHT) / 2 / np.mean(wls_dc)), wls_dc, 0.0),
        "through": _port_plane(dc_stack, "o4", ((dc.PLANE[0] - dc.WIDTH) / 2 / np.mean(wls_dc), (dc.PLANE[1] - dc.HEIGHT) / 2 / np.mean(wls_dc)), wls_dc, 0.0),
        "cross": _port_plane(dc_stack, "o3", ((dc.PLANE[0] - dc.WIDTH) / 2 / np.mean(wls_dc), (dc.PLANE[1] - dc.HEIGHT) / 2 / np.mean(wls_dc)), wls_dc, 0.0),
    }
    (geom_dir / "geometry_directional_coupler.json").write_text(json.dumps(dc_spec, indent=2))

    # 6. linbo3_bend
    lb_stack = lb.build_stack()
    lb_spec = {
        "bench": "linbo3_bend",
        **_extract_stack_geometry(lb_stack, lb.LAYER_STACK, lb.MATERIALS, "air", wafer_layer=wafer),
        "wavelength": float(lb.WAVELENGTH),
        "pml": float(lb.PML),
        "time": float(lb.DURATION),
        "spectral_width_frac": 0.05,
        "source": _port_plane(lb_stack, "o1", lb.PLANE_PADDING, [lb.WAVELENGTH], -1.0),
        "in": _port_plane(lb_stack, "o1", lb.PLANE_PADDING, [lb.WAVELENGTH], -0.5),
        "out": _port_plane(lb_stack, "o2", lb.PLANE_PADDING, [lb.WAVELENGTH], -0.5),
    }
    (geom_dir / "geometry_linbo3_bend.json").write_text(json.dumps(lb_spec, indent=2))
    print(f"Exported 6 benchmark geometry JSONs into {geom_dir}", flush=True)


# ------------------------------------------------------------------ MEEP runners


def _meep_medium(mp: Any, eps9: list[float]) -> Any:
    e = np.asarray(eps9, dtype=float).reshape(3, 3)
    if np.allclose(e, np.diag(np.diag(e)), atol=1e-12):
        d = np.diag(e)
        if np.allclose(d, d[0], atol=1e-12):
            return mp.Medium(epsilon=float(d[0]))
        return mp.Medium(epsilon_diag=mp.Vector3(float(d[0]), float(d[1]), float(d[2])))
    return mp.Medium(
        epsilon_diag=mp.Vector3(float(e[0, 0]), float(e[1, 1]), float(e[2, 2])),
        epsilon_offdiag=mp.Vector3(float(e[0, 1]), float(e[0, 2]), float(e[1, 2])),
    )


def _prism_signed_angle_rad(verts: list[list[float]], deg: float) -> float:
    """Sign ``sidewall_angle`` for MEEP ``mp.Prism`` so positive ``LayerLevel.sidewall_angle`` narrows upward.

    ``libctlgeom`` shifts vertices along the edge normal ``(dx, dy) x z``; CCW polygons shrink as
    height increases with positive ``sidewall_angle`` whereas CW polygons expand unless negated.
    """
    if abs(deg) < 1e-12:
        return 0.0
    xs = np.array([p[0] for p in verts], dtype=float)
    ys = np.array([p[1] for p in verts], dtype=float)
    signed_area = 0.5 * float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
    sign = 1.0 if signed_area >= 0.0 else -1.0
    return sign * math.radians(deg)


def _split_ribbon_polygon(
    poly: list[list[float]],
    domain: list[list[float]],
    angle_deg: float,
    thickness: float = 0.5,
) -> list[list[list[float]]]:
    """Split long waveguide ribbon polygons into small native mp.Prism chunks with tight AABBs.

    For simple ``<= 12``-vertex waveguides (e.g. ``sloped_guide``, ``elevator_coupler``), slicing
    into ``<= 0.4–0.5 um`` segments prevents ``libctlgeom``'s proportional bounding-box edge
    tolerance from widening narrow tapers and sidewalls. When ``angle_deg != 0``, adjacent chunks
    overlap so internal transverse prism bevels remain strictly inside the neighboring chunk.
    """
    pts = np.asarray(poly, dtype=float)
    if len(pts) <= 12:
        if abs(angle_deg) > 1e-12:
            xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
            ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
            if xmax - xmin <= 0.6:
                return [pts.tolist()]
            n_seg = max(2, int(math.ceil((xmax - xmin) / 0.4)))
            xs = np.linspace(xmin, xmax, n_seg + 1)
            ov_x = 2.5 * abs(thickness) * abs(math.tan(math.radians(angle_deg)))
            segs: list[list[list[float]]] = []
            for k in range(n_seg):
                xa = float(xs[k]) - (ov_x if k > 0 else 0.0)
                xb = float(xs[k + 1]) + (ov_x if k < n_seg - 1 else 0.0)
                segs.append([[xa, ymin], [xb, ymin], [xb, ymax], [xa, ymax]])
            return segs

        from shapely.geometry import Polygon
        from shapely.geometry.polygon import orient
        from shapely.ops import clip_by_rect

        p_geom = Polygon(pts)
        xmin, ymin, xmax, ymax = p_geom.bounds
        if xmax - xmin <= 0.6:
            return [pts.tolist()]
        xs = np.linspace(xmin, xmax, max(2, int(math.ceil((xmax - xmin) / 0.5))) + 1)
        clipped_out: list[list[list[float]]] = []
        for k in range(len(xs) - 1):
            c = clip_by_rect(p_geom, float(xs[k]), ymin - 1.0, float(xs[k + 1]), ymax + 1.0)
            parts = list(c.geoms) if hasattr(c, "geoms") else [c]
            for part in parts:
                if isinstance(part, Polygon) and not part.is_empty:
                    clipped_out.append(np.asarray(orient(part, sign=1.0).exterior.coords[:-1], dtype=float).tolist())
        return clipped_out

    (x0, x1), (y0, y1), _ = domain
    is_cap = (pts[:, 0] < x0 - 0.5) | (pts[:, 0] > x1 + 0.5) | (pts[:, 1] < y0 - 0.5) | (pts[:, 1] > y1 + 0.5)
    cap_edges = [i for i in range(len(pts)) if is_cap[i] and is_cap[(i + 1) % len(pts)]]
    if len(cap_edges) != 2:
        return [pts.tolist()]
    pts = np.roll(pts, -((cap_edges[0] + 1) % len(pts)), axis=0)
    is_cap = (pts[:, 0] < x0 - 0.5) | (pts[:, 0] > x1 + 0.5) | (pts[:, 1] < y0 - 0.5) | (pts[:, 1] > y1 + 0.5)
    m = next(i for i in range(len(pts) - 1) if is_cap[i] and is_cap[i + 1])
    sa_pts, sb_pts = pts[: m + 1], pts[m + 1 :][::-1]
    da = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(sa_pts, axis=0), axis=1))])
    db = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(sb_pts, axis=0), axis=1))])
    sa, sb = da / da[-1], db / db[-1]
    n_ch = max(16, min(32, max(len(sa_pts), len(sb_pts)) // 3))
    ca = np.round(np.linspace(0, len(sa_pts) - 1, n_ch + 1)).astype(int)
    cb = np.round(np.interp(sa[ca], sb, np.arange(len(sb_pts)))).astype(int)
    cb[0], cb[-1] = 0, len(sb_pts) - 1
    ov = 3 if abs(angle_deg) > 1e-12 else 0
    out: list[list[list[float]]] = []
    for k in range(n_ch):
        a0 = max(0, ca[k] - ov) if k > 0 else 0
        a1 = min(len(sa_pts) - 1, ca[k + 1] + ov) if k < n_ch - 1 else len(sa_pts) - 1
        b0 = max(0, cb[k] - ov) if k > 0 else 0
        b1 = min(len(sb_pts) - 1, cb[k + 1] + ov) if k < n_ch - 1 else len(sb_pts) - 1
        if a1 > a0 and b1 > b0:
            out.append(np.vstack([sa_pts[a0 : a1 + 1], sb_pts[b0 : b1 + 1][::-1]]).tolist())
    return out


def _build_meep_geometry(mp: Any, g: dict[str, Any], center: Any) -> tuple[list[Any], Any]:
    media = {name: _meep_medium(mp, eps9) for name, eps9 in g["materials"].items()}
    default_mat = media[g["background"]]
    geometry: list[Any] = []
    # MEEP: later objects take priority; descending mesh_order means lowest mesh_order wins.
    for lv in sorted(g["levels"], key=lambda item: -item["mesh_order"]):
        mat = media[lv["material"]]
        if lv["polygons"] is None:
            zc = lv["zmin"] + 0.5 * lv["thickness"]
            geometry.append(
                mp.Block(
                    center=mp.Vector3(center.x, center.y, zc),
                    size=mp.Vector3(mp.inf, mp.inf, abs(lv["thickness"])),
                    material=mat,
                )
            )
            continue
        for raw_poly in lv["polygons"]:
            for poly in _split_ribbon_polygon(raw_poly, g["domain"], lv["sidewall_angle"], lv["thickness"]):
                verts = [mp.Vector3(float(x), float(y), lv["zmin"]) for x, y in poly]
                geometry.append(
                    mp.Prism(
                        verts,
                        height=lv["thickness"],
                        sidewall_angle=_prism_signed_angle_rad(poly, lv["sidewall_angle"]),
                        material=mat,
                    )
                )
    return geometry, default_mat


def _cell_dims(box: list[list[float]], dx_nm: float) -> tuple[list[int], int]:
    dx_um = dx_nm * 1e-3
    cells = [round((hi - lo) / dx_um) for lo, hi in box]
    return cells, int(math.prod(cells))


def _run_aniso_slab(g: dict[str, Any], dx_nm: float) -> dict[str, Any]:
    import meep as mp

    dz_um = dx_nm * 1e-3
    res = 1.0 / dz_um
    f0 = 1.0 / g["wavelength"]
    half = g["half"]
    lz = 2.0 * half
    n_z = round(lz / dz_um)
    cell = mp.Vector3(0, 0, lz)
    e = np.asarray(g["eps_tensor"], dtype=float)
    slab_med = _meep_medium(mp, e.reshape(-1).tolist())
    slab_geom = [
        mp.Block(
            center=mp.Vector3(0, 0, g["slab_z"] + 0.5 * g["thickness"]),
            size=mp.Vector3(mp.inf, mp.inf, g["thickness"]),
            material=slab_med,
        )
    ]
    pulse = mp.GaussianSource(f0, fwidth=2.0 * np.pi * g["spectral_width_frac"] * f0, cutoff=6)
    t_until = g["time"] * C_UM_PER_S

    def _one_pol(with_slab: bool, comp: int) -> np.ndarray:
        sim = mp.Simulation(
            cell_size=cell,
            geometry=slab_geom if with_slab else [],
            default_material=mp.Medium(epsilon=1.0),
            sources=[mp.Source(pulse, component=comp, center=mp.Vector3(0, 0, g["z_src"]))],
            resolution=res,
            boundary_layers=[mp.PML(g["pml"], direction=mp.Z)],
            force_all_components=True,
            subpixel_tol=1e-3,
            subpixel_maxeval=2000,
        )
        dft = sim.add_dft_fields([mp.Ex, mp.Ey], f0, 0, 1, center=mp.Vector3(0, 0, g["z_trans"]), size=mp.Vector3())
        sim.run(until=t_until)
        return np.array([complex(np.asarray(sim.get_dft_array(dft, c, 0)).reshape(-1)[0]) for c in (mp.Ex, mp.Ey)])

    t0 = time.perf_counter()
    t_mat = np.zeros((2, 2), dtype=complex)
    for j, comp in enumerate((mp.Ex, mp.Ey)):
        w_air = _one_pol(False, comp)
        w_slab = _one_pol(True, comp)
        t_mat[:, j] = w_slab / w_air[j]
    run_s = time.perf_counter() - t0
    t_ref = np.asarray(g["t_ref_real"], dtype=float) + 1j * np.asarray(g["t_ref_imag"], dtype=float)
    return {
        "bench": "aniso_slab",
        "kind": f"uniform-{dx_nm:g}",
        "solver": "meep",
        "cells": [4, 4, n_z],
        "cells_z": n_z,
        "total_cells": 16 * n_z,
        "dx_nm": dx_nm,
        "t_error": float(np.max(np.abs(t_mat - t_ref))),
        "setup_s": 0.0,
        "run_s": round(run_s, 2),
        "mpi_procs": mp.count_processors(),
    }


def _classify_band_pol(sim: Any, f0: float, axis: Any, vol: Any, band: int) -> str:
    import meep as mp

    normal = axis
    transverse = (mp.Ey, mp.Ez) if normal == mp.X else (mp.Ex, mp.Ez)
    em = sim.get_eigenmode(f0, normal, vol, band, mp.Vector3(), match_frequency=True)
    # Sample field intensity across a coarse cross-section grid inside the core region
    xs = np.linspace(vol.center.x - 0.35 * vol.size.x, vol.center.x + 0.35 * vol.size.x, 7 if vol.size.x > 0 else 1)
    ys = np.linspace(vol.center.y - 0.35 * vol.size.y, vol.center.y + 0.35 * vol.size.y, 7 if vol.size.y > 0 else 1)
    zs = np.linspace(vol.center.z - 0.35 * vol.size.z, vol.center.z + 0.35 * vol.size.z, 7 if vol.size.z > 0 else 1)
    e_horiz = 0.0
    e_vert = 0.0
    for xx in xs:
        for yy in ys:
            for zz in zs:
                pt = mp.Vector3(float(xx), float(yy), float(zz))
                e_horiz += abs(em.amplitude(pt, transverse[0])) ** 2
                e_vert += abs(em.amplitude(pt, transverse[1])) ** 2
    return "te" if e_horiz >= e_vert else "tm"


def _run_3d(g: dict[str, Any], dx_nm: float) -> dict[str, Any]:
    import meep as mp

    bench = g["bench"]
    box = g["domain"]
    (x0, x1), (y0, y1), (z0, z1) = box
    cell = mp.Vector3(x1 - x0, y1 - y0, z1 - z0)
    center = mp.Vector3(0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1))
    res = 1.0 / (dx_nm * 1e-3)
    geometry, default_mat = _build_meep_geometry(mp, g, center)
    cells, total_cells = _cell_dims(box, dx_nm)

    f0 = 1.0 / g["wavelength"]
    src_spec = g["source"]
    src_vol = mp.Volume(center=mp.Vector3(*src_spec["center"]), size=mp.Vector3(*src_spec["size"]))

    use_y_sym = bench in ("sloped_guide", "elevator_coupler")
    parity = mp.ODD_Y if use_y_sym else mp.NO_PARITY
    pulse = mp.GaussianSource(f0, fwidth=2.0 * np.pi * g["spectral_width_frac"] * f0, cutoff=6)
    source = mp.EigenModeSource(
        pulse,
        center=src_vol.center,
        size=src_vol.size,
        direction=mp.X,
        eig_band=1,
        eig_parity=parity,
        eig_match_freq=True,
    )
    sim = mp.Simulation(
        cell_size=cell,
        geometry_center=center,
        geometry=geometry,
        default_material=default_mat,
        sources=[source],
        resolution=res,
        boundary_layers=[mp.PML(g["pml"])],
        symmetries=[mp.Mirror(mp.Y, phase=-1)] if use_y_sym else [],
        subpixel_tol=1e-3,
        subpixel_maxeval=2000,
    )

    t_until = g["time"] * C_UM_PER_S

    if bench == "sloped_guide":
        lx0, lx1 = g["line_x"]
        dx_um = dx_nm * 1e-3
        dft_line = sim.add_dft_fields(
            [mp.Ey],
            f0,
            0,
            1,
            center=mp.Vector3(0.5 * (lx0 + lx1), g["line_y"] + 0.25 * dx_um, g["line_z"] + 0.25 * dx_um),
            size=mp.Vector3(lx1 - lx0, 0.0, 0.0),
        )
        t_setup0 = time.perf_counter()
        sim.init_sim()
        setup_s = round(time.perf_counter() - t_setup0, 2)
        t0 = time.perf_counter()
        sim.run(until=t_until)
        run_s = time.perf_counter() - t0
        em = sim.get_eigenmode(f0, mp.X, src_vol, 1, mp.Vector3(f0 * 2.4, 0.0, 0.0), parity=mp.ODD_Y, match_frequency=True)
        neff_mode = float(abs(em.k.x) / f0)
        ey = np.asarray(sim.get_dft_array(dft_line, mp.Ey, 0)).reshape(-1)
        x_coords = np.asarray(sim.get_array_metadata(dft_cell=dft_line)[0], dtype=float)
        trim = g["trim"]
        keep = (x_coords > lx0 + trim) & (x_coords < lx1 - trim)
        phase = np.unwrap(np.angle(ey[keep]))
        slope = np.polyfit(x_coords[keep], phase, 1)[0]
        neff_fdtd = float(abs(slope) / (2.0 * np.pi / g["wavelength"]))
        return {
            "bench": bench,
            "kind": f"uniform-{dx_nm:g}",
            "solver": "meep",
            "cells": cells,
            "total_cells": total_cells,
            "dx_nm": dx_nm,
            "neff_mode": neff_mode,
            "neff_fdtd": neff_fdtd,
            "setup_s": setup_s,
            "run_s": round(run_s, 2),
            "mpi_procs": mp.count_processors(),
        }

    if bench == "directional_coupler":
        freqs = g["frequencies"]
        nfreq, df = len(freqs), freqs[-1] - freqs[0]
        mons = {
            name: sim.add_mode_monitor(
                f0, df, nfreq, mp.ModeRegion(center=mp.Vector3(*g[name]["center"]), size=mp.Vector3(*g[name]["size"]))
            )
            for name in ("in", "through", "cross")
        }
        t_setup0 = time.perf_counter()
        sim.init_sim()
        setup_s = round(time.perf_counter() - t_setup0, 2)
        t0 = time.perf_counter()
        sim.run(until=t_until)
        run_s = time.perf_counter() - t0
        alpha = {
            name: sim.get_eigenmode_coefficients(m, [1], direction=mp.X, eig_parity=mp.NO_PARITY).alpha[0, :, 0]
            for name, m in mons.items()
        }
        through = (np.abs(alpha["through"] / alpha["in"]) ** 2).tolist()
        cross = (np.abs(alpha["cross"] / alpha["in"]) ** 2).tolist()
        mid = nfreq // 2
        s_mid = float(cross[mid] / (cross[mid] + through[mid]))
        return {
            "bench": bench,
            "kind": f"uniform-{dx_nm:g}",
            "solver": "meep",
            "cells": cells,
            "total_cells": total_cells,
            "dx_nm": dx_nm,
            "through": through,
            "cross": cross,
            "split_mid": s_mid,
            "total_mid": float(through[mid] + cross[mid]),
            "setup_s": setup_s,
            "run_s": round(run_s, 2),
            "mpi_procs": mp.count_processors(),
        }

    in_spec, out_spec = g["in"], g["out"]
    mon_in = sim.add_mode_monitor(
        f0, 0, 1, mp.ModeRegion(center=mp.Vector3(*in_spec["center"]), size=mp.Vector3(*in_spec["size"]))
    )
    mon_out = sim.add_mode_monitor(
        f0, 0, 1, mp.ModeRegion(center=mp.Vector3(*out_spec["center"]), size=mp.Vector3(*out_spec["size"]))
    )
    t_setup0 = time.perf_counter()
    sim.init_sim()
    setup_s = round(time.perf_counter() - t_setup0, 2)
    t0 = time.perf_counter()
    sim.run(until=t_until)
    run_s = time.perf_counter() - t0

    out_dir = mp.X if out_spec["axis"] == 0 else mp.Y
    in_res = sim.get_eigenmode_coefficients(mon_in, [1], direction=mp.X, eig_parity=parity)
    a_in = in_res.alpha[0, 0, 0]

    if bench == "linbo3_bend":
        # Inspect low-order bands at o2 (+y) to pick fundamental TE0 and TM0 reliably.
        out_res = sim.get_eigenmode_coefficients(mon_out, [1, 2, 3], direction=mp.Y, eig_parity=mp.NO_PARITY)
        out_vol = mp.Volume(center=mp.Vector3(*out_spec["center"]), size=mp.Vector3(*out_spec["size"]))
        pols = [_classify_band_pol(sim, f0, mp.Y, out_vol, b) for b in (1, 2, 3)]
        te_idx = pols.index("te")
        tm_idx = pols.index("tm")
        a_out_te = out_res.alpha[te_idx, 0, 0]
        a_out_tm = out_res.alpha[tm_idx, 0, 0]
        em_in = sim.get_eigenmode(f0, mp.X, mp.Volume(center=mp.Vector3(*in_spec["center"]), size=mp.Vector3(*in_spec["size"])), 1, mp.Vector3(), match_frequency=True)
        em_out = sim.get_eigenmode(f0, mp.Y, out_vol, te_idx + 1, mp.Vector3(), match_frequency=True)
        return {
            "bench": bench,
            "kind": f"uniform-{dx_nm:g}",
            "solver": "meep",
            "cells": cells,
            "total_cells": total_cells,
            "dx_nm": dx_nm,
            "T": float(abs(a_out_te / a_in) ** 2),
            "T_tm0": float(abs(a_out_tm / a_in) ** 2),
            "neff_in": float(abs(em_in.k.x) / f0),
            "neff_out": float(abs(em_out.k.y) / f0),
            "out_te_band": te_idx + 1,
            "out_tm_band": tm_idx + 1,
            "setup_s": setup_s,
            "run_s": round(run_s, 2),
            "mpi_procs": mp.count_processors(),
        }

    out_res = sim.get_eigenmode_coefficients(mon_out, [1], direction=out_dir, eig_parity=parity)
    a_out = out_res.alpha[0, 0, 0]
    return {
        "bench": bench,
        "kind": f"uniform-{dx_nm:g}",
        "solver": "meep",
        "cells": cells,
        "total_cells": total_cells,
        "dx_nm": dx_nm,
        "T": float(abs(a_out / a_in) ** 2),
        "setup_s": setup_s,
        "run_s": round(run_s, 2),
        "mpi_procs": mp.count_processors(),
    }


def run_one(geom_dir: Path, bench: str, dx_nm: float, out_path: Path) -> None:
    import meep as mp

    mp.verbosity(0)
    g = json.loads((geom_dir / f"geometry_{bench}.json").read_text())
    row = _run_aniso_slab(g, dx_nm) if bench == "aniso_slab" else _run_3d(g, dx_nm)
    if mp.am_master():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(row))
        print(
            f"MEEP {bench} uniform-{dx_nm:g}: cells={row['total_cells']:,d} run={row['run_s']:.2f}s "
            f"procs={row['mpi_procs']}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("export", help="Export exact polygons and benchmark metadata to JSON.")
    p_exp.add_argument("--geom-dir", type=Path, required=True)

    p_run = sub.add_parser("run", help="Run one MEEP uniform-grid benchmark simulation.")
    p_run.add_argument("--geom-dir", type=Path, required=True)
    p_run.add_argument("--bench", choices=BENCHES, required=True)
    p_run.add_argument("--dx-nm", type=float, required=True)
    p_run.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "export":
        export_all(args.geom_dir)
    else:
        run_one(args.geom_dir, args.bench, args.dx_nm, args.out)


if __name__ == "__main__":
    main()
