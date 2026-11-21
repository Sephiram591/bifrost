"""Accuracy per grid cell across six photonic simulations: UniformGrid vs bf.mesh.

All simulations use bifrost's default Green's-theorem voxelizer (``voxelizer="greens"``)
and run under both Kottke subpixel homogenization modes, selected with ``--averaging``:
``"subpixel"`` (the diagonal of the averaged tensor, bifrost's default) and
``"subpixel_full"`` (with its off-diagonal entries). Each configuration runs in an isolated worker
subprocess so XLA compilation caches do not accumulate VRAM and GPU peak memory stats
are clean.

Benchmarks:
    1. ``aniso_slab``        1D rotated anisotropic crystal slab vs exact Berreman transfer matrix
    2. ``sloped_guide``      10-degree sloped SOI strip waveguide effective index
    3. ``soi_bragg``         Broadband 10-period SOI sidewall Bragg grating (1.48-1.60 um)
    4. ``elevator_coupler``  Multi-level SiN adiabatic elevator coupler (100 nm gap, 50 nm tips)
    5. ``directional_coupler`` Broadband SiN directional coupler (300 nm gap, 538-728 nm)
    6. ``sin_bend``          Curved SiN bend (R=7 um) + Refine.material_cells substrate study
    7. ``linbo3_bend``       X-cut LiNbO3 (TFLN) 90-degree trapezoidal bend on silica substrate
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

WT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT_ROOT))
sys.path.insert(0, str(WT_ROOT / "examples"))
sys.path.insert(0, str(WT_ROOT / "examples" / "guide"))
sys.path.insert(0, str(WT_ROOT / "benchmarks" / "subpixel"))
sys.path.insert(0, str(WT_ROOT / "benchmarks" / "greens"))

AVERAGINGS = ("subpixel", "subpixel_full")


def _apply_stack(arrays: Any, objects: Any, params: Any, key: Any) -> Any:
    """``bf.apply_stack``, except that under ``"subpixel_full"`` without ``tidy3d_extras`` the
    mode sources and detectors are solved on the diagonal of the averaged tensor (tidy3d's mode
    solver needs ``tidy3d_extras`` for fully anisotropic media) while the FDTD run keeps it whole."""
    from dc_convergence_compare import _apply_stack_safe

    return _apply_stack_safe(arrays, objects, params, key)


# ---------------------------------------------------------------- helpers


def _peak_gb(jax_mod: Any) -> float | None:
    stats = jax_mod.devices()[0].memory_stats() or {}
    if "peak_bytes_in_use" in stats:
        return round(stats["peak_bytes_in_use"] / 1e9, 3)
    return None


def _grid_stats(config: Any) -> tuple[list[int], int, list[float], list[float]]:
    edges = [np.asarray(config.resolved_grid.edges(a), dtype=np.float64) * 1e6 for a in range(3)]
    shape = [len(e) - 1 for e in edges]
    min_nm = [round(float(np.diff(e).min()) * 1e3, 2) for e in edges]
    max_nm = [round(float(np.diff(e).max()) * 1e3, 2) for e in edges]
    return shape, int(np.prod(shape)), min_nm, max_nm


# ---------------------------------------------------------------- 1. aniso_slab


def _aniso_slab_tensor() -> np.ndarray:
    eps = np.diag([9.0, 12.0, 14.0])
    a = np.radians(25.0)
    rot = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    out = rot @ eps @ rot.T
    out[0, 2] = out[2, 0] = 0.8
    return out


def _run_aniso_slab_one(
    z_edges_um: np.ndarray, pml_min: int, pml_max: int, eps_tensor: np.ndarray, with_slab: bool, pol: int,
    averaging: str,
) -> tuple[np.ndarray, int, float, float | None]:
    import fdtdx
    import gdsfactory as gf
    import jax
    import jax.numpy as jnp
    from gdsfactory.technology import LayerLevel, LayerStack

    import bifrost as bf

    assert Path(bf.__file__).resolve().is_relative_to(WT_ROOT), bf.__file__
    gf.gpdk.PDK.activate()
    gf.snap.SNAP_ENABLED = False

    wavelength = 1.55
    thickness = 0.137
    slab_z = 0.013
    half = 1.8
    lateral = 4 * 0.04
    h = lateral / 2

    eps = eps_tensor if with_slab else np.eye(3)
    mats = {
        "air": fdtdx.Material(),
        "slab": fdtdx.Material(permittivity=tuple(tuple(float(x) for x in row) for row in eps)),
    }
    c = gf.Component()
    c.add_polygon([(-h, -h), (h, -h), (h, h), (-h, h)], layer=gf.gpdk.LAYER.WAFER)
    ls = LayerStack(
        layers={
            "slab": LayerLevel(
                layer=gf.gpdk.LAYER.WAFER, zmin=slab_z, thickness=thickness, material="slab", mesh_order=1
            )
        }
    )
    device = bf.StackDevice.from_layer_stack(
        c,
        ls,
        mats,
        background="air",
        name="stack",
        zrange_layers=["slab"],
        boundary_offset=(0.0, 0.0, 0.0, 0.0, half - slab_z - thickness, -half - slab_z),
        diagnostics=False,
        averaging=averaging,
        voxelizer="greens",
    )
    grid = device.rectilinear_grid(0.04, z_um=z_edges_um)
    config = device.get_simulation_config(time=120e-15, grid=grid, backend=jax.default_backend())
    volume, constraints = device.get_simulation_volume(material=mats["air"])
    bound = fdtdx.BoundaryConfig(
        boundary_type_minx="periodic",
        boundary_type_maxx="periodic",
        boundary_type_miny="periodic",
        boundary_type_maxy="periodic",
        thickness_grid_minz=pml_min,
        thickness_grid_maxz=pml_max,
    )
    boundaries, b_constraints = fdtdx.boundary_objects_from_config(bound, volume)
    wave = fdtdx.WaveCharacter(wavelength=wavelength * 1e-6)
    profile = fdtdx.GaussianPulseProfile(
        spectral_width=fdtdx.WaveCharacter(wavelength=10 * wavelength * 1e-6), center_wave=wave
    )
    source = fdtdx.UniformPlaneSource(
        name="source",
        partial_grid_shape=(None, None, 1),
        direction="+",
        wave_character=wave,
        temporal_profile=profile,
        fixed_E_polarization_vector=(1.0, 0.0, 0.0) if pol == 0 else (0.0, 1.0, 0.0),
    )
    planes = {"trans": 0.75, "trans2": 0.95, "refl": -1.0}
    detectors = [
        fdtdx.PhasorDetector(
            name=n,
            partial_grid_shape=(None, None, 1),
            wave_characters=(wave,),
            components=("Ex", "Ey"),
            reduce_volume=True,
            scaling_mode="pulse",
        )
        for n in planes
    ]

    def at(obj: Any, zz: float) -> Any:
        return obj.place_relative_to(
            volume, axes=(0, 1, 2), own_positions=(0, 0, 0), other_positions=(0, 0, 0), margins=(0, 0, zz * 1e-6)
        )

    constraints += [*b_constraints, at(source, -0.7), *(at(d, planes[d.name]) for d in detectors)]
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=[volume, device, source, *detectors, *boundaries.values()],
        config=config,
        constraints=constraints,
        key=key,
    )
    params = {**params, **bf.init_stack_params(objects)}
    arrays, objects, _ = _apply_stack(arrays, objects, params, key)
    jax.block_until_ready(arrays.inv_permittivities)

    t0 = time.perf_counter()
    with jax.enable_x64(False):
        _, state = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key, show_progress=False)
        jax.block_until_ready(state.detector_states)
    run_s = time.perf_counter() - t0
    rows = [np.asarray(jnp.reshape(state.detector_states[n]["phasor"], (-1,))[:2]) for n in ("trans", "trans2", "refl")]
    return np.stack(rows), int(config.time_steps_total), run_s, _peak_gb(jax)


def run_aniso_slab(kind: str, averaging: str) -> dict[str, Any]:
    import fdtdx
    import gdsfactory as gf
    from gdsfactory.technology import LayerLevel, LayerStack

    import bifrost as bf
    from berreman import jones

    gf.gpdk.PDK.activate()
    gf.snap.SNAP_ENABLED = False

    wavelength, thickness, slab_z, pml, half = 1.55, 0.137, 0.013, 0.6, 1.8
    eps_tensor = _aniso_slab_tensor()
    r_ref, t_ref = jones([(eps_tensor, thickness)], 1.0, 1.0, wavelength)
    t_ref = t_ref * np.exp(-2j * np.pi / wavelength * thickness)

    if kind.startswith("uniform-"):
        dz_um = float(kind.split("-", 1)[1]) * 1e-3
        n = round(2.0 * half / dz_um)
        z = np.linspace(-half, half, n + 1)
        pml_min = pml_max = round(pml / dz_um)
        summary = None
    elif kind.startswith("mesh-"):
        ppw = float(kind.split("-", 1)[1])
        h = 2 * 0.04
        c = gf.Component()
        c.add_polygon([(-h, -h), (h, -h), (h, h), (-h, h)], layer=gf.gpdk.LAYER.WAFER)
        ls = LayerStack(
            layers={
                "slab": LayerLevel(
                    layer=gf.gpdk.LAYER.WAFER, zmin=slab_z, thickness=thickness, material="slab", mesh_order=1
                )
            }
        )
        mats = {
            "air": fdtdx.Material(),
            "slab": fdtdx.Material(permittivity=tuple(tuple(float(x) for x in row) for row in eps_tensor)),
        }
        device = bf.StackDevice.from_layer_stack(
            c,
            ls,
            mats,
            background="air",
            name="stack",
            zrange_layers=["slab"],
            boundary_offset=(0.0, 0.0, 0.0, 0.0, half - slab_z - thickness, -half - slab_z),
            diagnostics=False,
            averaging=averaging,
            voxelizer="greens",
        )
        m = bf.mesh(
            device,
            wavelength,
            ppw=ppw,
            boundaries=("periodic", "periodic", "periodic", "periodic", "pml", "pml"),
            pml=pml,
        )
        z = m.edges[2]
        pml_min, pml_max = m.pml_cells[4], m.pml_cells[5]
        summary = m.summary(time=120e-15)
    else:
        raise ValueError(f"Unknown aniso_slab kind {kind!r}")

    t_mat = np.zeros((2, 2), dtype=complex)
    r_mat = np.zeros((2, 2))
    run_s_total = 0.0
    steps = 0
    peak_gb = None
    for pol in (0, 1):
        with_slab, steps, rs1, pk1 = _run_aniso_slab_one(z, pml_min, pml_max, eps_tensor, True, pol, averaging)
        without, _, rs2, pk2 = _run_aniso_slab_one(z, pml_min, pml_max, eps_tensor, False, pol, averaging)
        run_s_total += rs1 + rs2
        if pk1 is not None or pk2 is not None:
            peak_gb = max(pk1 or 0.0, pk2 or 0.0, peak_gb or 0.0)
        inc = without[0, pol]
        t_mat[:, pol] = with_slab[0] / inc
        r_mat[:, pol] = np.abs(with_slab[2] / inc)

    dz = np.diff(z) * 1e3
    cells_z = len(z) - 1
    return {
        "bench": "aniso_slab",
        "kind": kind,
        "voxelizer": "greens",
        "averaging": averaging,
        "cells": [4, 4, cells_z],
        "cells_z": cells_z,
        "total_cells": 16 * cells_z,
        "min_dz_nm": round(float(dz.min()), 2),
        "max_dz_nm": round(float(dz.max()), 2),
        "pml_cells": [pml_min, pml_max],
        "steps": steps,
        "t_error": float(np.max(np.abs(t_mat - t_ref))),
        "r_error": float(np.max(np.abs(r_mat - np.abs(r_ref)))),
        "energy_err": float(np.max(np.abs(np.sum(np.abs(t_mat) ** 2 + r_mat**2, axis=0) - 1.0))),
        "run_s": round(run_s_total, 2),
        "peak_gb": peak_gb,
        "summary": summary,
    }


# ---------------------------------------------------------------- 2. sloped_guide


def run_sloped_guide(kind: str, averaging: str) -> dict[str, Any]:
    import fdtdx
    import gdsfactory as gf
    import jax
    import jax.numpy as jnp
    from gdsfactory.technology import LayerLevel, LayerStack

    import bifrost as bf

    assert Path(bf.__file__).resolve().is_relative_to(WT_ROOT), bf.__file__
    gf.gpdk.PDK.activate()
    gf.snap.SNAP_ENABLED = False

    wavelength = 1.55
    width, thickness, angle = 0.5, 0.22, 10.0
    n_core, n_clad = 3.48, 1.444
    box = ((-3.20, 3.20), (-1.60, 1.60), (-1.49, 1.71))  # 6.40 x 3.20 x 3.20 um
    pml = 1.0
    line_x = (-1.4, 2.0)
    trim = 0.3

    (x0, x1), (y0, y1), (z0, z1) = box
    c = gf.Component()
    h = width / 2
    c.add_polygon([(x0, -h), (x1, -h), (x1, h), (x0, h)], layer=gf.gpdk.LAYER.WG)
    c.add_port(name="o1", center=(x0, 0.0), width=width, orientation=180, layer=gf.gpdk.LAYER.WG)
    c.add_port(name="o2", center=(x1, 0.0), width=width, orientation=0, layer=gf.gpdk.LAYER.WG)
    ls = LayerStack(
        layers={
            "core": LayerLevel(
                layer=gf.gpdk.LAYER.WG,
                zmin=0.0,
                thickness=thickness,
                material="si",
                mesh_order=1,
                sidewall_angle=angle,
            )
        }
    )
    mats = {"si": fdtdx.Material(permittivity=n_core**2), "clad": fdtdx.Material(permittivity=n_clad**2)}
    stack = bf.StackDevice.from_layer_stack(
        c,
        ls,
        mats,
        background="clad",
        zrange_layers=["core"],
        name="stack",
        diagnostics=False,
        boundary_offset=(0.0, 0.0, y1 - h, y0 + h, z1 - thickness, z0),
        voxelizer="greens",
        averaging=averaging,
    )

    summary = None
    if kind.startswith("uniform-"):
        res_um = float(kind.split("-", 1)[1]) * 1e-3
        grid = fdtdx.core.grid.UniformGrid(spacing=res_um * 1e-6)
        bound = fdtdx.BoundaryConfig.from_uniform_bound(thickness=round(pml / res_um))
        on_grid: dict[str, Any] = {}
    elif kind.startswith("mesh-"):
        ppw = float(kind.split("-", 1)[1])
        m = bf.mesh(stack, wavelength, ppw=ppw, pml=pml)
        summary = m.summary(time=200e-15)
        grid, bound = m.grid, m.boundary
    else:
        raise ValueError(f"Unknown sloped_guide kind {kind!r}")

    config = stack.get_simulation_config(time=200e-15, grid=grid)
    if not kind.startswith("uniform-"):
        on_grid = {"grid": config}

    wave = fdtdx.WaveCharacter(wavelength=wavelength * 1e-6)
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=wave, spectral_width=fdtdx.WaveCharacter(frequency=0.1 * wave.get_frequency())
    )
    volume, constraints = stack.get_simulation_volume()
    source, placing = stack.create_mode_plane_source_at(
        "o1", (0.6, 0.6), wave_character=wave, temporal_profile=pulse, port_offset=1.3, name="source",
        mode_index=0, filter_pol="te", **on_grid,
    )
    constraints += placing
    line = fdtdx.PhasorDetector(
        name="line",
        partial_real_shape=((line_x[1] - line_x[0]) * 1e-6, None, None),
        partial_grid_shape=(None, 1, 1),
        wave_characters=(wave,),
        components=("Ey",),
        scaling_mode="pulse",
    )
    mid = [0.5 * (lo + hi) for lo, hi in box]
    constraints.append(
        line.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(0, 0, 0),
            other_positions=(0, 0, 0),
            margins=((0.5 * (line_x[0] + line_x[1]) - mid[0]) * 1e-6, (0.0 - mid[1]) * 1e-6, (thickness / 2 - mid[2]) * 1e-6),
        )
    )
    boundaries, b_placing = fdtdx.boundary_objects_from_config(bound, volume)
    constraints += b_placing

    key = jax.random.PRNGKey(0)
    t0 = time.perf_counter()
    objects, arrays, params, config, _ = fdtdx.place_objects(
        [volume, stack, source, line, *boundaries.values()], config, constraints, key
    )
    all_params = {**params, **bf.init_stack_params(objects)}
    arrays, objects, _ = _apply_stack(arrays, objects, all_params, key)
    jax.block_until_ready(arrays.inv_permittivities)
    t_setup = time.perf_counter() - t0
    neff_mode = complex(np.asarray(objects["source"]._neff).reshape(-1)[0]).real

    t0 = time.perf_counter()
    with jax.enable_x64(False):
        _, state = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key, show_progress=False)
        jax.block_until_ready(state.detector_states)
    t_run = time.perf_counter() - t0

    e = np.asarray(jnp.reshape(state.detector_states["line"]["phasor"], (-1,)))
    edges_x = np.asarray(config.resolved_grid.edges(0), dtype=np.float64) * 1e6
    (i0, i1), _, _ = objects["line"].grid_slice_tuple
    x = 0.5 * (edges_x[i0:i1] + edges_x[i0 + 1 : i1 + 1]) - 0.5 * (edges_x[0] + edges_x[-1])
    keep = (x > line_x[0] + trim) & (x < line_x[1] - trim)
    phase = np.unwrap(np.angle(e[keep]))
    slope = np.polyfit(x[keep], phase, 1)[0]
    neff_fdtd = float(abs(slope) / (2 * np.pi / wavelength))

    shape, total_cells, min_nm, max_nm = _grid_stats(config)
    return {
        "bench": "sloped_guide",
        "kind": kind,
        "voxelizer": "greens",
        "averaging": averaging,
        "cells": shape,
        "total_cells": total_cells,
        "min_cell_nm": min_nm,
        "max_cell_nm": max_nm,
        "steps": int(config.time_steps_total),
        "neff_mode": float(neff_mode),
        "neff_fdtd": neff_fdtd,
        "setup_s": round(t_setup, 2),
        "run_s": round(t_run, 2),
        "peak_gb": _peak_gb(jax),
        "summary": summary,
    }


# ---------------------------------------------------------------- 3. soi_bragg


def _bragg_component():
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    gf.snap.SNAP_ENABLED = False

    w0, w1, w2 = 0.50, 0.56, 0.44
    period, n_periods = 0.320, 10
    half_p = 0.5 * period
    x_start = -0.5 * n_periods * period  # -1.60 um
    x_end = 0.5 * n_periods * period     # +1.60 um
    lead = 0.80  # straight leads [-2.40, -1.60] and [1.60, 2.40]

    pts: list[tuple[float, float]] = [(-x_end - lead, -w0 / 2), (x_start, -w0 / 2)]
    for k in range(n_periods):
        xk = x_start + k * period
        pts.extend([(xk, -w1 / 2), (xk + half_p, -w1 / 2), (xk + half_p, -w2 / 2), (xk + period, -w2 / 2)])
    pts.extend([(x_end, -w0 / 2), (x_end + lead, -w0 / 2), (x_end + lead, w0 / 2), (x_end, w0 / 2)])
    for k in range(n_periods - 1, -1, -1):
        xk = x_start + k * period
        pts.extend([(xk + period, w2 / 2), (xk + half_p, w2 / 2), (xk + half_p, w1 / 2), (xk, w1 / 2)])
    pts.extend([(x_start, w0 / 2), (-x_end - lead, w0 / 2)])

    c = gf.Component()
    c.add_polygon(pts, layer=gf.gpdk.LAYER.WG)
    c.add_port("o1", center=(-x_end - lead, 0.0), width=w0, orientation=180, layer=gf.gpdk.LAYER.WG)
    c.add_port("o2", center=(x_end + lead, 0.0), width=w0, orientation=0, layer=gf.gpdk.LAYER.WG)
    return c


def run_soi_bragg(kind: str, averaging: str) -> dict[str, Any]:
    import fdtdx
    import gdsfactory as gf
    import jax
    from gdsfactory.technology import LayerLevel, LayerStack

    import bifrost as bf

    assert Path(bf.__file__).resolve().is_relative_to(WT_ROOT), bf.__file__
    gf.gpdk.PDK.activate()
    gf.snap.SNAP_ENABLED = False

    ls = LayerStack(
        layers={
            "box": LayerLevel(layer="WAFER", zmin=-3.0, thickness=3.0, material="sio2", mesh_order=9),
            "core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=0.22, material="si", mesh_order=1),
        }
    )
    mats = {
        "si": fdtdx.Material(permittivity=3.48**2),
        "sio2": fdtdx.Material(permittivity=1.444**2),
        "air": fdtdx.Material(permittivity=1.0),
    }
    stack = bf.StackDevice.from_layer_stack(
        _bragg_component(),
        ls,
        mats,
        background="air",
        zrange_layers=["core"],
        extend_ports=(("o1", 1.60), ("o2", 1.60)),
        boundary_offset=(0.0, 0.0, 1.32, -1.32, 1.49, -1.49),  # 8.00 x 3.20 x 3.20 um
        name="stack",
        diagnostics=False,
        voxelizer="greens",
        averaging=averaging,
    )

    pml = 0.80
    sim_time = 360e-15
    summary = None
    if kind.startswith("uniform-"):
        res_um = float(kind.split("-", 1)[1]) * 1e-3
        grid = fdtdx.core.grid.UniformGrid(spacing=res_um * 1e-6)
        bound = fdtdx.BoundaryConfig.from_uniform_bound(thickness=round(pml / res_um))
        on_grid: dict[str, Any] = {}
    elif kind.startswith("mesh-"):
        ppw = float(kind.split("-", 1)[1])
        m = bf.mesh(stack, (1.48, 1.60), ppw=ppw, pml=pml)
        summary = m.summary(time=sim_time)
        grid, bound = m.grid, m.boundary
    else:
        raise ValueError(f"Unknown soi_bragg kind {kind!r}")

    config = stack.get_simulation_config(time=sim_time, grid=grid)
    if not kind.startswith("uniform-"):
        on_grid = {"grid": config}

    center_wl = 1.54
    freqs_inv_um = np.linspace(1.0 / 1.60, 1.0 / 1.48, 31)
    band = [fdtdx.WaveCharacter(wavelength=1e-6 / f) for f in freqs_inv_um]
    center = fdtdx.WaveCharacter(wavelength=center_wl * 1e-6)
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=center, spectral_width=fdtdx.WaveCharacter(frequency=0.08 * center.get_frequency())
    )
    te0 = dict(mode_index=0, filter_pol="te")
    mode = dict(wave_characters=band, scaling_mode="pulse", **te0)
    pad = (0.5, 0.45)

    volume, constraints = stack.get_simulation_volume()
    objects = [volume, stack]
    for obj, pl in (
        stack.create_mode_plane_source_at(
            "o1", pad, wave_character=center, temporal_profile=pulse, port_offset=-0.60, name="source", **te0, **on_grid
        ),
        stack.create_mode_overlap_detector_at("o1", pad, direction="in", port_offset=-0.25, name="in", **mode, **on_grid),
        stack.create_mode_overlap_detector_at("o2", pad, port_offset=-0.25, name="out", **mode, **on_grid),
    ):
        objects.append(obj)
        constraints += pl
    boundaries, b_placing = fdtdx.boundary_objects_from_config(bound, volume)
    objects += list(boundaries.values())
    constraints += b_placing

    key = jax.random.PRNGKey(0)
    t0 = time.perf_counter()
    objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
    all_params = {**params, **bf.init_stack_params(objects)}
    arrays, objects, _ = _apply_stack(arrays, objects, all_params, key)
    jax.block_until_ready(arrays.inv_permittivities)
    t_setup = time.perf_counter() - t0

    t0 = time.perf_counter()
    with jax.enable_x64(False):
        _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key, show_progress=False)
        jax.block_until_ready(arrays.detector_states)
    t_run = time.perf_counter() - t0

    a_in = objects["in"].compute_overlap(arrays.detector_states["in"])
    a_out = objects["out"].compute_overlap(arrays.detector_states["out"])
    t_spec = (np.abs(np.asarray(a_out / a_in)) ** 2).tolist()
    shape, total_cells, min_nm, max_nm = _grid_stats(config)
    return {
        "bench": "soi_bragg",
        "kind": kind,
        "voxelizer": "greens",
        "averaging": averaging,
        "cells": shape,
        "total_cells": total_cells,
        "min_cell_nm": min_nm,
        "max_cell_nm": max_nm,
        "steps": int(config.time_steps_total),
        "wavelength_nm": (1e3 / freqs_inv_um).tolist(),
        "T_spectrum": t_spec,
        "T_min": float(min(t_spec)),
        "setup_s": round(t_setup, 2),
        "run_s": round(t_run, 2),
        "peak_gb": _peak_gb(jax),
        "summary": summary,
    }


# ---------------------------------------------------------------- 4. elevator_coupler


def run_elevator_coupler(kind: str, averaging: str) -> dict[str, Any]:
    import fdtdx
    import fdtdx_elevator_coupler as ec
    import jax

    import bifrost as bf

    assert Path(bf.__file__).resolve().is_relative_to(WT_ROOT), bf.__file__
    overlap = 3.80
    stack = bf.StackDevice.from_layer_stack(
        ec.elevator_coupler(overlap),
        ec.LAYER_STACK,
        ec.MATERIALS,
        background="air",
        name="stack",
        zrange_layers=["lower", "upper"],
        extend_ports=(("o1", ec.EXTENSION), ("o2", ec.EXTENSION)),
        boundary_offset=(0.0, 0.0, 2.0 - ec.W_MAX / 2, -2.0 + ec.W_MAX / 2, 1.60 - 0.19, -1.60 + 0.21),
        voxelizer="greens",
        averaging=averaging,
    )

    pml = ec.PML
    sim_time = 400e-15
    summary = None
    if kind.startswith("uniform-"):
        res_um = float(kind.split("-", 1)[1]) * 1e-3
        grid = fdtdx.core.grid.UniformGrid(spacing=res_um * 1e-6)
        bound = fdtdx.BoundaryConfig.from_uniform_bound(thickness=round(pml / res_um))
        on_grid: dict[str, Any] = {}
    elif kind.startswith("mesh-"):
        ppw = float(kind.split("-", 1)[1])
        m = bf.mesh(stack, ec.WAVELENGTH, ppw=ppw, pml=pml)
        summary = m.summary(time=sim_time)
        grid, bound = m.grid, m.boundary
    else:
        raise ValueError(f"Unknown elevator_coupler kind {kind!r}")

    config = stack.get_simulation_config(time=sim_time, grid=grid)
    if not kind.startswith("uniform-"):
        on_grid = {"grid": config}

    wave = fdtdx.WaveCharacter(wavelength=ec.WAVELENGTH * 1e-6)
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=wave, spectral_width=fdtdx.WaveCharacter(frequency=0.05 * wave.get_frequency())
    )
    te0 = dict(mode_index=0, filter_pol="te")
    mode = dict(wave_characters=(wave,), scaling_mode="pulse", **te0)

    volume, constraints = stack.get_simulation_volume()
    objects = [volume, stack]
    for obj, pl in (
        stack.create_mode_plane_source_at(
            "o1", (1.838, 1.074), wave_character=wave, temporal_profile=pulse, port_offset=-0.3, name="source",
            **te0, **on_grid,
        ),
        stack.create_mode_overlap_detector_at("o1", (1.838, 1.074), direction="in", name="in", **mode, **on_grid),
        stack.create_mode_overlap_detector_at("o2", (1.838, 1.511), name="out", **mode, **on_grid),
    ):
        objects.append(obj)
        constraints += pl
    boundaries, b_placing = fdtdx.boundary_objects_from_config(bound, volume)
    objects += list(boundaries.values())
    constraints += b_placing

    key = jax.random.PRNGKey(0)
    t0 = time.perf_counter()
    objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
    all_params = {**params, **bf.init_stack_params(objects)}
    arrays, objects, _ = _apply_stack(arrays, objects, all_params, key)
    jax.block_until_ready(arrays.inv_permittivities)
    t_setup = time.perf_counter() - t0

    t0 = time.perf_counter()
    with jax.enable_x64(False):
        _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key, show_progress=False)
        jax.block_until_ready(arrays.detector_states)
    t_run = time.perf_counter() - t0

    a_in = objects["in"].compute_overlap(arrays.detector_states["in"])
    a_out = objects["out"].compute_overlap(arrays.detector_states["out"])
    t_val = float(np.ravel(abs(a_out / a_in) ** 2)[0])
    shape, total_cells, min_nm, max_nm = _grid_stats(config)
    return {
        "bench": "elevator_coupler",
        "kind": kind,
        "voxelizer": "greens",
        "averaging": averaging,
        "cells": shape,
        "total_cells": total_cells,
        "min_cell_nm": min_nm,
        "max_cell_nm": max_nm,
        "steps": int(config.time_steps_total),
        "T": t_val,
        "setup_s": round(t_setup, 2),
        "run_s": round(t_run, 2),
        "peak_gb": _peak_gb(jax),
        "summary": summary,
    }


# ---------------------------------------------------------------- 5. directional_coupler


def run_directional_coupler(kind: str, averaging: str) -> dict[str, Any]:
    sys.path.insert(0, str(WT_ROOT / "benchmarks" / "greens"))
    import dc_convergence_compare as dcc

    row = dcc._run_one(kind, averaging, "greens")
    row["bench"] = "directional_coupler"
    return row


# ---------------------------------------------------------------- 6. sin_bend


def run_sin_bend(kind: str, averaging: str) -> dict[str, Any]:
    import fdtdx
    import inverse_design as invd
    import jax
    import jax.numpy as jnp

    import bifrost as bf

    assert Path(bf.__file__).resolve().is_relative_to(WT_ROOT), bf.__file__
    z_margin = (invd.BOX_HEIGHT - invd.THICKNESS) / 2
    stack = bf.StackDevice.from_layer_stack(
        invd.tight_bend(invd.UNIFORM),
        invd.layer_stack,
        invd.materials,
        background="air",
        zrange_layers=["core"],
        extend_ports=(("o1", invd.EXTENSION), ("o2", invd.EXTENSION)),
        boundary_offset=(invd.MARGIN, 0, 0, -invd.MARGIN, z_margin, -z_margin),
        name="stack",
        averaging=averaging,
        voxelizer="greens",
        dtype=jnp.float32,
    )

    summary = None
    if kind.startswith("uniform-"):
        res_um = float(kind.split("-", 1)[1]) * 1e-3
        grid = fdtdx.core.grid.UniformGrid(spacing=res_um * 1e-6)
        bound = fdtdx.BoundaryConfig.from_uniform_bound(thickness=round(invd.PML / res_um))
        on_grid: dict[str, Any] = {}
    elif kind == "meshclad-40-80":
        ppw_core = invd.WAVELENGTH / (invd.CORE_INDEX * 0.04)
        m = bf.mesh(
            stack,
            invd.WAVELENGTH,
            ppw=ppw_core,
            pml=invd.PML,
            refine=bf.Refine(dilation={"WG": 0.45}, material_cells={"silica": 0.08}),
        )
        summary = m.summary(time=invd.DURATION)
        grid, bound = m.grid, m.boundary
    elif kind.startswith("meshclad-"):
        ppw = float(kind.split("-", 1)[1])
        clad_cell = invd.WAVELENGTH / (invd.BOX_INDEX * min(ppw, 13.0))
        m = bf.mesh(
            stack,
            invd.WAVELENGTH,
            ppw=ppw,
            pml=invd.PML,
            refine=bf.Refine(dilation={"WG": 0.45}, material_cells={"silica": clad_cell}),
        )
        summary = m.summary(time=invd.DURATION)
        grid, bound = m.grid, m.boundary
    elif kind.startswith("meshsub-"):
        ppw = float(kind.split("-", 1)[1])
        clad_cell = invd.WAVELENGTH / (invd.BOX_INDEX * min(ppw, 12.0))
        m = bf.mesh(
            stack,
            invd.WAVELENGTH,
            ppw=ppw,
            pml=invd.PML,
            refine=bf.Refine(material_cells={"silica": clad_cell}),
        )
        summary = m.summary(time=invd.DURATION)
        grid, bound = m.grid, m.boundary
    elif kind.startswith("mesh-"):
        ppw = float(kind.split("-", 1)[1])
        m = bf.mesh(stack, invd.WAVELENGTH, ppw=ppw, pml=invd.PML)
        summary = m.summary(time=invd.DURATION)
        grid, bound = m.grid, m.boundary
    else:
        raise ValueError(f"Unknown sin_bend kind {kind!r}")

    config = stack.get_simulation_config(time=invd.DURATION, grid=grid)
    if not kind.startswith("uniform-"):
        on_grid = {"grid": config}

    wave = fdtdx.WaveCharacter(wavelength=invd.WAVELENGTH * 1e-6)
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=wave, spectral_width=fdtdx.WaveCharacter(frequency=0.05 * wave.get_frequency())
    )
    te0 = dict(mode_index=0, filter_pol="te")
    vol, constraints = stack.get_simulation_volume()
    objects = [vol, stack]
    for obj, pl in (
        stack.create_mode_plane_source_at(
            "o1", invd.PADDING, wave_character=wave, temporal_profile=pulse, port_offset=-1.0, name="source",
            **te0, **on_grid,
        ),
        stack.create_mode_overlap_detector_at(
            "o1", invd.PADDING, direction="in", wave_characters=(wave,), scaling_mode="pulse", port_offset=-0.5,
            name="in", **te0, **on_grid,
        ),
        stack.create_mode_overlap_detector_at(
            "o2", invd.PADDING, wave_characters=(wave,), scaling_mode="pulse", port_offset=-0.5,
            name="out", **te0, **on_grid,
        ),
    ):
        objects.append(obj)
        constraints += pl
    bounds, b_placing = fdtdx.boundary_objects_from_config(bound, vol)
    objects += list(bounds.values())
    constraints += b_placing

    t0 = time.perf_counter()
    objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, invd.KEY)
    all_params = {**params, **bf.init_stack_params(objects)}
    arrays, objects, _ = _apply_stack(arrays, objects, all_params, invd.KEY)
    jax.block_until_ready(arrays.inv_permittivities)
    t_setup = time.perf_counter() - t0

    t0 = time.perf_counter()
    with jax.enable_x64(False):
        _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=invd.KEY, show_progress=False)
        jax.block_until_ready(arrays.detector_states)
    t_run = time.perf_counter() - t0

    a_in = objects["in"].compute_overlap(arrays.detector_states["in"])[0]
    a_out = objects["out"].compute_overlap(arrays.detector_states["out"])[0]
    s21 = complex(np.asarray(a_out / a_in))
    shape, total_cells, min_nm, max_nm = _grid_stats(config)
    return {
        "bench": "sin_bend",
        "kind": kind,
        "voxelizer": "greens",
        "averaging": averaging,
        "cells": shape,
        "total_cells": total_cells,
        "min_cell_nm": min_nm,
        "max_cell_nm": max_nm,
        "steps": int(config.time_steps_total),
        "T": float(abs(s21) ** 2),
        "phase_rad": float(math.atan2(s21.imag, s21.real)),
        "setup_s": round(t_setup, 2),
        "run_s": round(t_run, 2),
        "peak_gb": _peak_gb(jax),
        "summary": summary,
    }


# ---------------------------------------------------------------- 7. linbo3_bend


def run_linbo3_bend(kind: str, averaging: str) -> dict[str, Any]:
    import fdtdx
    import jax
    import jax.numpy as jnp
    import linbo3_bend as lb

    import bifrost as bf

    assert Path(bf.__file__).resolve().is_relative_to(WT_ROOT), bf.__file__
    z_margin = (lb.BOX_HEIGHT - lb.THICKNESS) / 2
    stack = bf.StackDevice.from_layer_stack(
        lb.component,
        lb.LAYER_STACK,
        lb.MATERIALS,
        background="air",
        zrange_layers=["core"],
        extend_ports=(("o1", lb.EXTENSION), ("o2", lb.EXTENSION)),
        boundary_offset=(lb.MARGIN, 0.0, 0.0, -lb.MARGIN, z_margin, -z_margin),
        name="stack",
        diagnostics=False,
        averaging=averaging,
        voxelizer="greens",
        dtype=jnp.float32,
    )

    summary = None
    if kind.startswith("uniform-"):
        res_um = float(kind.split("-", 1)[1]) * 1e-3
        grid = fdtdx.core.grid.UniformGrid(spacing=res_um * 1e-6)
        bound = fdtdx.BoundaryConfig.from_uniform_bound(thickness=round(lb.PML / res_um))
        on_grid: dict[str, Any] = {}
    elif kind.startswith("meshsub-"):
        ppw = float(kind.split("-", 1)[1])
        clad_cell = lb.WAVELENGTH / (lb.BOX_INDEX * min(ppw, 12.0))
        m = bf.mesh(
            stack,
            lb.WAVELENGTH,
            ppw=ppw,
            pml=lb.PML,
            refine=bf.Refine(material_cells={"silica": clad_cell}),
        )
        summary = m.summary(time=lb.DURATION)
        grid, bound = m.grid, m.boundary
    elif kind.startswith("mesh-"):
        ppw = float(kind.split("-", 1)[1])
        m = bf.mesh(stack, lb.WAVELENGTH, ppw=ppw, pml=lb.PML)
        summary = m.summary(time=lb.DURATION)
        grid, bound = m.grid, m.boundary
    else:
        raise ValueError(f"Unknown linbo3_bend kind {kind!r}")

    config = stack.get_simulation_config(time=lb.DURATION, grid=grid)
    if not kind.startswith("uniform-"):
        on_grid = {"grid": config}

    wave = fdtdx.WaveCharacter(wavelength=lb.WAVELENGTH * 1e-6)
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=wave, spectral_width=fdtdx.WaveCharacter(frequency=0.05 * wave.get_frequency())
    )
    te0 = dict(mode_index=0, filter_pol="te")
    tm0 = dict(mode_index=0, filter_pol="tm")
    vol, constraints = stack.get_simulation_volume()
    objects = [vol, stack]
    for obj, pl in (
        stack.create_mode_plane_source_at(
            "o1", lb.PLANE_PADDING, wave_character=wave, temporal_profile=pulse, port_offset=-1.0,
            name="source", **te0, **on_grid,
        ),
        stack.create_mode_overlap_detector_at(
            "o1", lb.PLANE_PADDING, direction="in", wave_characters=(wave,), scaling_mode="pulse",
            port_offset=-0.5, name="in", **te0, **on_grid,
        ),
        stack.create_mode_overlap_detector_at(
            "o2", lb.PLANE_PADDING, wave_characters=(wave,), scaling_mode="pulse",
            port_offset=-0.5, name="out", **te0, **on_grid,
        ),
        stack.create_mode_overlap_detector_at(
            "o2", lb.PLANE_PADDING, wave_characters=(wave,), scaling_mode="pulse",
            port_offset=-0.5, name="out_tm", **tm0, **on_grid,
        ),
    ):
        objects.append(obj)
        constraints += pl
    bounds, b_placing = fdtdx.boundary_objects_from_config(bound, vol)
    objects += list(bounds.values())
    constraints += b_placing

    key = jax.random.PRNGKey(0)
    t0 = time.perf_counter()
    objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
    all_params = {**params, **bf.init_stack_params(objects)}
    arrays, objects, _ = _apply_stack(arrays, objects, all_params, key)
    jax.block_until_ready(arrays.inv_permittivities)
    t_setup = time.perf_counter() - t0

    t0 = time.perf_counter()
    with jax.enable_x64(False):
        _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key, show_progress=False)
        jax.block_until_ready(arrays.detector_states)
    t_run = time.perf_counter() - t0

    a_in = objects["in"].compute_overlap(arrays.detector_states["in"])[0]
    a_out = objects["out"].compute_overlap(arrays.detector_states["out"])[0]
    a_tm = objects["out_tm"].compute_overlap(arrays.detector_states["out_tm"])[0]
    s21 = complex(np.asarray(a_out / a_in))
    s21_tm = complex(np.asarray(a_tm / a_in))
    neff_in = float(complex(np.asarray(objects["in"]._mode_neff).reshape(-1)[0]).real)
    neff_out = float(complex(np.asarray(objects["out"]._mode_neff).reshape(-1)[0]).real)
    shape, total_cells, min_nm, max_nm = _grid_stats(config)
    return {
        "bench": "linbo3_bend",
        "kind": kind,
        "voxelizer": "greens",
        "averaging": averaging,
        "cells": shape,
        "total_cells": total_cells,
        "min_cell_nm": min_nm,
        "max_cell_nm": max_nm,
        "steps": int(config.time_steps_total),
        "T": float(abs(s21) ** 2),
        "T_tm0": float(abs(s21_tm) ** 2),
        "phase_rad": float(math.atan2(s21.imag, s21.real)),
        "neff_in": neff_in,
        "neff_out": neff_out,
        "setup_s": round(t_setup, 2),
        "run_s": round(t_run, 2),
        "peak_gb": _peak_gb(jax),
        "summary": summary,
    }


RUNNERS = {
    "aniso_slab": run_aniso_slab,
    "sloped_guide": run_sloped_guide,
    "soi_bragg": run_soi_bragg,
    "elevator_coupler": run_elevator_coupler,
    "directional_coupler": run_directional_coupler,
    "sin_bend": run_sin_bend,
    "linbo3_bend": run_linbo3_bend,
}


DC_KINDS = (
    "uniform-100", "uniform-50", "uniform-40", "uniform-25", "uniform-23.5", "uniform-20",
    "mesh-6", "mesh-8", "mesh-10", "mesh-12", "mesh-12.5", "mesh-14", "mesh-16",
)
BEND_IMPORT_KINDS = (
    "uniform-120", "uniform-60", "uniform-40", "uniform-30", "uniform-24",
    "mesh-12", "mesh-16", "mesh-20", "mesh-24",
)


def _import_existing(results: list[dict[str, Any]]) -> None:
    seen = {(r["bench"], r["kind"], r["averaging"]) for r in results}

    dc_path = WT_ROOT / "benchmarks" / "greens" / "results" / "dc_convergence.json"
    if dc_path.exists():
        for r in json.loads(dc_path.read_text()):
            if r.get("voxelizer") == "greens" and r.get("averaging") in AVERAGINGS:
                k, avg = r["kind"], r["averaging"]
                if k in DC_KINDS and ("directional_coupler", k, avg) not in seen:
                    results.append({**r, "bench": "directional_coupler", "source": "benchmarks/greens/results/dc_convergence.json"})
                    seen.add(("directional_coupler", k, avg))

    bend_path = WT_ROOT / "benchmarks" / "greens" / "results" / "bend_convergence.json"
    if bend_path.exists():
        for r in json.loads(bend_path.read_text()):
            if r.get("bench") == "sin_bend" and r.get("voxelizer") == "greens" and r.get("averaging") in AVERAGINGS:
                k, avg = r["kind"], r["averaging"]
                if k in BEND_IMPORT_KINDS and ("sin_bend", k, avg) not in seen:
                    results.append({**r, "bench": "sin_bend", "source": "benchmarks/greens/results/bend_convergence.json"})
                    seen.add(("sin_bend", k, avg))


def _job_list(averaging: str = "subpixel") -> list[tuple[str, str, bool]]:
    """Return ``(bench, kind, use_platform_allocator)`` jobs in execution order."""
    sub = averaging == "subpixel"
    jobs: list[tuple[str, str, bool]] = []
    if not sub:
        for k in (
            "uniform-60", "uniform-45", "uniform-36", "uniform-30", "uniform-24",
            "uniform-18", "uniform-15", "uniform-12", "uniform-10", "uniform-7.5",
            "mesh-6", "mesh-8", "mesh-10", "mesh-14", "mesh-20", "mesh-28", "mesh-40", "mesh-55",
        ):
            jobs.append(("aniso_slab", k, False))

    sg_kinds: tuple[str, ...] = (
        "uniform-64", "uniform-50", "uniform-40", "uniform-32", "uniform-25", "uniform-20", "uniform-16",
        "uniform-12.5", "uniform-11",
        *(("uniform-9", "uniform-7.2") if sub else ()),
        "mesh-8", "mesh-10", "mesh-12", "mesh-15", "mesh-18", "mesh-22", "mesh-27", "mesh-34",
        "mesh-46", "mesh-59",
        *(("mesh-74", "mesh-86") if sub else ()),
    )
    for k in sg_kinds:
        jobs.append(("sloped_guide", k, k == "uniform-12.5"))

    for k in (
        "uniform-64", "uniform-50", "uniform-40", "uniform-32", "uniform-25", "uniform-20", "uniform-16", "uniform-12.5",
        "mesh-8", "mesh-10", "mesh-12", "mesh-15", "mesh-18", "mesh-22", "mesh-26",
    ):
        jobs.append(("soi_bragg", k, k in ("uniform-16", "uniform-12.5")))

    ec_kinds: tuple[str, ...] = (
        "uniform-80", "uniform-50", "uniform-40", "uniform-32", "uniform-25", "uniform-20", "uniform-17", "uniform-16",
        *(("uniform-12.5", "uniform-12") if sub else ()),
        "mesh-6", "mesh-8", "mesh-10", "mesh-12", "mesh-15", "mesh-18", "mesh-22",
        *(("mesh-28", "mesh-33") if sub else ()),
    )
    for k in ec_kinds:
        jobs.append(("elevator_coupler", k, k in ("uniform-20", "uniform-16", "mesh-18")))

    dc_kinds: tuple[str, ...] = (
        *DC_KINDS,
        *(("uniform-16.5", "mesh-18") if sub else ()),
    )
    for k in dc_kinds:
        jobs.append(("directional_coupler", k, k in ("uniform-25", "uniform-20", "mesh-10", "mesh-12", "mesh-14", "mesh-16")))

    sb_kinds: tuple[str, ...] = (
        *BEND_IMPORT_KINDS,
        *(("uniform-20", "uniform-16.2") if sub else ()),
        "mesh-30", "mesh-36",
        *(("mesh-46", "mesh-54") if sub else ()),
        "meshsub-12", "meshsub-16", "meshsub-20", "meshsub-24", "meshsub-32", "meshsub-38",
        *(("meshsub-46", "meshsub-57") if sub else ()),
        "meshclad-14", "meshclad-17", "meshclad-20", "meshclad-24", "meshclad-40-80",
    )
    for k in sb_kinds:
        jobs.append(("sin_bend", k, k in ("uniform-30", "uniform-24")))

    lb_kinds: tuple[str, ...] = (
        "uniform-120", "uniform-60", "uniform-40", "uniform-30", "uniform-24",
        *(("uniform-20", "uniform-17") if sub else ()),
        "mesh-10", "mesh-12", "mesh-16", "mesh-20", "mesh-26", "mesh-34",
        *(("mesh-42", "mesh-48") if sub else ()),
        "meshsub-10", "meshsub-12", "meshsub-16", "meshsub-20", "meshsub-27", "meshsub-36",
        *(("meshsub-46", "meshsub-51") if sub else ()),
    )
    for k in lb_kinds:
        jobs.append(("linbo3_bend", k, False))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(WT_ROOT / "benchmarks/meshing/results/accuracy_per_cell.json"))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--bench", default="all")
    parser.add_argument("--kind", default=None, help="one kind (worker), or a comma-separated subset to run")
    parser.add_argument("--averaging", default="subpixel", choices=AVERAGINGS)
    parser.add_argument(
        "--platform-allocator",
        action="store_true",
        help="Use XLA_PYTHON_CLIENT_ALLOCATOR=platform for the largest grids (omits JAX peak_gb stats)",
    )
    args = parser.parse_args()

    if args.worker:
        if args.bench not in RUNNERS or not args.kind:
            raise SystemExit("--worker requires valid --bench and --kind")
        import jax

        row = RUNNERS[args.bench](args.kind, args.averaging)
        row.setdefault("gpu", jax.devices()[0].device_kind)
        print("RESULT_JSON:" + json.dumps(row), flush=True)
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text())
        except Exception:
            results = []
    _import_existing(results)
    out_path.write_text(json.dumps(results, indent=2))

    done = {(r["bench"], r["kind"], r["averaging"]) for r in results}
    for bench, kind, use_platform in _job_list(args.averaging):
        if args.bench != "all" and bench != args.bench:
            continue
        if args.kind and kind not in args.kind.split(","):
            continue
        if (bench, kind, args.averaging) in done:
            print(f"Skipping ({bench}, {kind}, {args.averaging})", flush=True)
            continue
        env = os.environ.copy()
        if args.platform_allocator and use_platform:
            env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
            env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
            env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        else:
            env.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.97"
        proc = subprocess.run(
            [sys.executable, __file__, "--worker", "--bench", bench, "--kind", kind, "--averaging", args.averaging],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if proc.returncode != 0:
            print(f"FAILED ({bench}, {kind}, {args.averaging}):\n{proc.stderr[-1500:]}", flush=True)
            continue
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON:")), None)
        if line is None:
            print(f"NO RESULT ({bench}, {kind}, {args.averaging}):\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}", flush=True)
            continue
        row = json.loads(line.split("RESULT_JSON:", 1)[1])
        results.append(row)
        done.add((bench, kind, args.averaging))
        out_path.write_text(json.dumps(results, indent=2))
        pk = f"{row['peak_gb']:.2f}GB" if row.get("peak_gb") is not None else "n/a"
        print(
            f"{bench:20s} {kind:14s} {args.averaging:13s} | cells={row['total_cells']:>10,d} steps={row['steps']:>6d} "
            f"run={row['run_s']:.2f}s peak={pk}",
            flush=True,
        )


if __name__ == "__main__":
    main()
