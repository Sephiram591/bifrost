"""Convergence & gradient check on both air-clad 90-degree bends: greens vs legacy.

Bend A: Compact tight air-clad SOI bend (R = 2.0 um, w = 0.5 um, h = 0.22 um, n = 3.48 on SiO2).
Bend B: Air-clad SiN tight_bend (R = 7.0 um, w = 1.6 um, h = 0.75 um, n = 1.996 on SiO2, 144 vertices)
matching ``examples/guide/inverse_design.py``.
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


# ---------------------------------------------------------------- Bend A: SOI R=2.0 um


def _soi_bend_component(bulge: float = 0.0):
    import gdsfactory as gf
    import jax.numpy as jnp

    gf.gpdk.PDK.activate()
    gf.snap.SNAP_ENABLED = False
    r_mid = 2.0
    th = jnp.linspace(0.0, 0.5 * math.pi, 72)
    hw = 0.25 + jnp.asarray(bulge) * jnp.sin(2.0 * th)
    r_out = r_mid + hw
    r_in = r_mid - hw
    outer = jnp.stack([r_out * jnp.sin(th), r_mid - r_out * jnp.cos(th)], axis=1)
    inner = jnp.stack([r_in * jnp.sin(th), r_mid - r_in * jnp.cos(th)], axis=1)
    c = gf.Component()
    c.add_polygon(jnp.concatenate([outer, inner[::-1]]), layer=gf.gpdk.LAYER.WG)
    c.add_port("o1", center=(0.0, 0.0), width=0.50, orientation=180, layer=gf.gpdk.LAYER.WG)
    c.add_port("o2", center=(r_mid, r_mid), width=0.50, orientation=90, layer=gf.gpdk.LAYER.WG)
    return c


def _apply_stack_safe(arrays, objects, all_params, key):
    import importlib.util

    import jax.numpy as jnp

    import bifrost as bf

    stack_obj = objects["stack"]
    if stack_obj.averaging != "subpixel_full" or importlib.util.find_spec("tidy3d_extras") is not None:
        return bf.apply_stack(arrays, objects, all_params, key)
    full_arrays, status = stack_obj.write_with_status(arrays, all_params.get("stack"), None, True)
    diag_inv = jnp.zeros_like(full_arrays.inv_permittivities)
    for idx_d in (0, 4, 8):
        diag_inv = diag_inv.at[idx_d].set(full_arrays.inv_permittivities[idx_d])
    diag_arrays = full_arrays.at["inv_permittivities"].set(diag_inv)
    objects, reapplied, reused = bf.stack._device._reapply_stack_objects(
        diag_arrays, objects, [stack_obj], key
    )
    objects = bf.stack._device._cast_pml_coefficients(objects, stack_obj.dtype)
    return full_arrays, objects, {
        "stack_reapplied": reapplied,
        "stack_reused": reused,
        "stack_status": {"stack": status},
    }


def _run_soi(kind: str, averaging: str, voxelizer: str, shift_frac: float = 0.0) -> dict[str, Any]:
    import fdtdx
    import jax
    import jax.numpy as jnp
    from gdsfactory.technology import LayerLevel, LayerStack

    import bifrost as bf
    import gdsfactory as gf

    assert Path(bf.__file__).resolve().is_relative_to(WT_ROOT), bf.__file__
    res_um = float(kind.split("-")[1]) * 1e-3
    shift_um = shift_frac * res_um
    # Extension 2.1 um gives [-2.1, 2.25] x [-0.25, 4.1]; box 5.40 x 5.40 x 2.40 um divides evenly at 40/30/25/20/15/12 nm
    lstack = LayerStack(
        layers={
            "box": LayerLevel(layer="WAFER", zmin=-2.0, thickness=2.0, material="sio2", mesh_order=9),
            "core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=0.22, material="si", mesh_order=1),
        }
    )
    mats = {
        "si": fdtdx.Material(permittivity=3.48**2),
        "sio2": fdtdx.Material(permittivity=1.444**2),
        "air": fdtdx.Material(permittivity=1.0),
    }
    ext = 2.1
    boff = (1.10 + shift_um, 0.05 + shift_um, -0.05 + shift_um, -1.10 + shift_um, 1.09 + shift_um, -1.09 + shift_um)
    stack = bf.StackDevice.from_layer_stack(
        _soi_bend_component(),
        lstack,
        mats,
        background="air",
        zrange_layers=["core"],
        extend_ports=(("o1", ext), ("o2", ext)),
        boundary_offset=boff,
        name="stack",
        averaging=averaging,
        voxelizer=voxelizer,
        dtype=jnp.float32,
    )
    grid = fdtdx.core.grid.UniformGrid(spacing=res_um * 1e-6)
    pml = 0.8
    config = stack.get_simulation_config(time=300e-15, grid=grid)
    wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=wave, spectral_width=fdtdx.WaveCharacter(frequency=0.05 * wave.get_frequency())
    )
    te0 = dict(mode_index=0, filter_pol="te")
    pad = (0.5, 0.5)
    vol, constraints = stack.get_simulation_volume()
    objects = [vol, stack]
    for obj, pl in (
        stack.create_mode_plane_source_at("o1", pad, wave_character=wave, temporal_profile=pulse, port_offset=-0.95, name="source", **te0),
        stack.create_mode_overlap_detector_at("o1", pad, direction="in", wave_characters=(wave,), scaling_mode="pulse", port_offset=-0.45, name="in", **te0),
        stack.create_mode_overlap_detector_at("o2", pad, wave_characters=(wave,), scaling_mode="pulse", port_offset=-0.45, name="out", **te0),
    ):
        objects.append(obj)
        constraints += pl
    bounds, pl = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=round(pml / res_um)), vol)
    objects += list(bounds.values())
    constraints += pl

    key = jax.random.PRNGKey(0)
    t0 = time.perf_counter()
    objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
    all_params = {**params, **bf.init_stack_params(objects)}
    arrays, objects, status = _apply_stack_safe(arrays, objects, all_params, key)
    jax.block_until_ready(arrays.inv_permittivities)
    t_setup = time.perf_counter() - t0

    t0 = time.perf_counter()
    with jax.enable_x64(False):
        _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key, show_progress=False)
        jax.block_until_ready(arrays.detector_states)
    t_run = time.perf_counter() - t0

    a_in = objects["in"].compute_overlap(arrays.detector_states["in"])[0]
    a_out = objects["out"].compute_overlap(arrays.detector_states["out"])[0]
    s21 = complex(np.asarray(a_out / a_in))
    mem = jax.devices()[0].memory_stats() or {}
    return {
        "bench": "soi_bend",
        "kind": kind,
        "shift_frac": shift_frac,
        "voxelizer": voxelizer,
        "averaging": averaging,
        "res_nm": round(res_um * 1e3, 2),
        "cells": [int(x) for x in config.resolved_grid.shape],
        "steps": int(config.time_steps_total),
        "T": float(abs(s21) ** 2),
        "phase_rad": float(math.atan2(s21.imag, s21.real)),
        "setup_s": round(t_setup, 2),
        "run_s": round(t_run, 2),
        "peak_gb": round(mem["peak_bytes_in_use"] / 1e9, 3) if "peak_bytes_in_use" in mem else None,
    }


def _run_soi_grad_check(averaging: str) -> dict[str, Any]:
    import fdtdx
    import jax
    import jax.numpy as jnp
    from gdsfactory.technology import LayerLevel, LayerStack

    import bifrost as bf
    import gdsfactory as gf

    assert Path(bf.__file__).resolve().is_relative_to(WT_ROOT), bf.__file__
    res_um = 0.03
    lstack = LayerStack(
        layers={
            "box": LayerLevel(layer="WAFER", zmin=-2.0, thickness=2.0, material="sio2", mesh_order=9),
            "core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=0.22, material="si", mesh_order=1),
        }
    )
    mats = {
        "si": fdtdx.Material(permittivity=3.48**2),
        "sio2": fdtdx.Material(permittivity=1.444**2),
        "air": fdtdx.Material(permittivity=1.0),
    }
    boff = (1.10, 0.05, -0.05, -1.10, 1.09, -1.09)
    stack = bf.StackDevice.from_layer_stack(
        _soi_bend_component(0.0),
        lstack,
        mats,
        background="air",
        zrange_layers=["core"],
        extend_ports=(("o1", 2.1), ("o2", 2.1)),
        boundary_offset=boff,
        name="stack",
        averaging=averaging,
        voxelizer="greens",
        dtype=jnp.float32,
    )
    config = stack.get_simulation_config(
        time=260e-15,
        grid=fdtdx.core.grid.UniformGrid(spacing=res_um * 1e-6),
        gradient_config=fdtdx.GradientConfig(method="checkpointed", num_checkpoints=45),
    )
    wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=wave, spectral_width=fdtdx.WaveCharacter(frequency=0.05 * wave.get_frequency())
    )
    te0 = dict(mode_index=0, filter_pol="te")
    pad = (0.5, 0.5)
    vol, constraints = stack.get_simulation_volume()
    objects = [vol, stack]
    for obj, pl in (
        stack.create_mode_plane_source_at("o1", pad, wave_character=wave, temporal_profile=pulse, port_offset=-0.95, name="source", **te0),
        stack.create_mode_overlap_detector_at("o1", pad, direction="in", wave_characters=(wave,), scaling_mode="pulse", port_offset=-0.45, name="in", **te0),
        stack.create_mode_overlap_detector_at("o2", pad, wave_characters=(wave,), scaling_mode="pulse", port_offset=-0.45, name="out", **te0),
    ):
        objects.append(obj)
        constraints += pl
    bounds, pl = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=round(0.8 / res_um)), vol)
    objects += list(bounds.values())
    constraints += pl
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
    params = {**params, **bf.init_stack_params(objects)}

    def trans(bulge):
        comp = _soi_bend_component(bulge)
        p = {**params, "stack": {**params["stack"], **objects["stack"].pack(comp)}}
        arrs, objs, _ = bf.apply_stack(arrays, objects, p, key)
        _, arrs = fdtdx.run_fdtd(arrays=arrs, objects=objs, config=config, key=key, show_progress=False)
        a_in = objs["in"].compute_overlap(arrs.detector_states["in"])[0]
        a_out = objs["out"].compute_overlap(arrs.detector_states["out"])[0]
        return jnp.abs(a_out / a_in) ** 2

    with jax.enable_x64(False):
        t0_val, g_ad = jax.value_and_grad(trans)(jnp.asarray(0.0, dtype=jnp.float32))
        jax.block_until_ready(g_ad)
        eps = 2e-3
        fd_val = float(trans(jnp.asarray(eps, dtype=jnp.float32)) - trans(jnp.asarray(-eps, dtype=jnp.float32))) / (2 * eps)
    return {
        "bench": "soi_grad_check",
        "averaging": averaging,
        "res_nm": 30.0,
        "T0": float(t0_val),
        "grad_ad": float(g_ad),
        "grad_fd": float(fd_val),
        "rel_err": abs(float(g_ad) - fd_val) / max(abs(fd_val), 1e-9),
    }


# ---------------------------------------------------------------- Bend B: SiN tight_bend (examples/guide/inverse_design.py)


def _run_sin(kind: str, averaging: str, voxelizer: str) -> dict[str, Any]:
    import fdtdx
    import jax
    import jax.numpy as jnp

    sys.path.insert(0, str(WT_ROOT / "examples" / "guide"))
    import inverse_design as invd
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
        voxelizer=voxelizer,
        dtype=jnp.float32,
    )
    if kind.startswith("uniform-"):
        res_um = float(kind.split("-")[1]) * 1e-3
        grid = fdtdx.core.grid.UniformGrid(spacing=res_um * 1e-6)
        bound = fdtdx.BoundaryConfig.from_uniform_bound(thickness=round(invd.PML / res_um))
        on_grid: dict[str, Any] = {}
        summary = None
    elif kind.startswith("mesh-"):
        ppw = float(kind.split("-")[1])
        m = bf.mesh(stack, (invd.WAVELENGTH, invd.WAVELENGTH), ppw=ppw, pml=invd.PML, center_faces=True)
        summary = m.summary(time=invd.DURATION)
        grid, bound = m.grid, m.boundary
    else:
        raise ValueError(f"Unknown kind {kind!r}")

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
        stack.create_mode_plane_source_at("o1", invd.PADDING, wave_character=wave, temporal_profile=pulse, port_offset=-1.0, name="source", **te0, **on_grid),
        stack.create_mode_overlap_detector_at("o1", invd.PADDING, direction="in", wave_characters=(wave,), scaling_mode="pulse", port_offset=-0.5, name="in", **te0, **on_grid),
        stack.create_mode_overlap_detector_at("o2", invd.PADDING, wave_characters=(wave,), scaling_mode="pulse", port_offset=-0.5, name="out", **te0, **on_grid),
    ):
        objects.append(obj)
        constraints += pl
    bounds, pl = fdtdx.boundary_objects_from_config(bound, vol)
    objects += list(bounds.values())
    constraints += pl

    t0 = time.perf_counter()
    objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, invd.KEY)
    all_params = {**params, **bf.init_stack_params(objects)}
    arrays, objects, status = _apply_stack_safe(arrays, objects, all_params, invd.KEY)
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
    mem = jax.devices()[0].memory_stats() or {}
    return {
        "bench": "sin_bend",
        "kind": kind,
        "voxelizer": voxelizer,
        "averaging": averaging,
        "cells": [int(x) for x in config.resolved_grid.shape],
        "total_cells": int(np.prod(config.resolved_grid.shape)),
        "steps": int(config.time_steps_total),
        "T": float(abs(s21) ** 2),
        "phase_rad": float(math.atan2(s21.imag, s21.real)),
        "setup_s": round(t_setup, 2),
        "run_s": round(t_run, 2),
        "peak_gb": round(mem["peak_bytes_in_use"] / 1e9, 3) if "peak_bytes_in_use" in mem else None,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="benchmarks/greens/results/bend_convergence.json")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--task", default="soi")
    parser.add_argument("--kind", default="uniform-40")
    parser.add_argument("--voxelizer", default="greens")
    parser.add_argument("--averaging", default="subpixel")
    parser.add_argument("--shift", type=float, default=0.0)
    args = parser.parse_args()

    if args.worker:
        if args.task == "soi":
            row = _run_soi(args.kind, args.averaging, args.voxelizer, shift_frac=args.shift)
        elif args.task == "soi_grad":
            row = _run_soi_grad_check(args.averaging)
        elif args.task == "sin":
            row = _run_sin(args.kind, args.averaging, args.voxelizer)
        else:
            raise ValueError(args.task)
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

    def already(match: dict[str, Any]) -> bool:
        return any(all(r.get(k) == v for k, v in match.items()) for r in results)

    jobs: list[tuple[dict[str, Any], list[str]]] = []
    # SOI bend: fine reference first, then uniform sweep, shift jitter, and AD grad check
    jobs.append((
        {"bench": "soi_bend", "kind": "uniform-12", "shift_frac": 0.0, "voxelizer": "greens", "averaging": "subpixel_full"},
        ["--task", "soi", "--kind", "uniform-12", "--voxelizer", "greens", "--averaging", "subpixel_full"],
    ))
    for res in (40, 30, 25, 20, 15):
        for vox in ("greens", "legacy"):
            for avg in ("subpixel", "subpixel_full"):
                jobs.append((
                    {"bench": "soi_bend", "kind": f"uniform-{res}", "shift_frac": 0.0, "voxelizer": vox, "averaging": avg},
                    ["--task", "soi", "--kind", f"uniform-{res}", "--voxelizer", vox, "--averaging", avg],
                ))
    for sh in (0.0, 0.2, 0.4, 0.6, 0.8):
        for vox in ("greens", "legacy"):
            for avg in ("subpixel", "subpixel_full"):
                jobs.append((
                    {"bench": "soi_bend", "kind": "uniform-30", "shift_frac": sh, "voxelizer": vox, "averaging": avg},
                    ["--task", "soi", "--kind", "uniform-30", "--voxelizer", vox, "--averaging", avg, "--shift", str(sh)],
                ))
    for avg in ("subpixel", "subpixel_full"):
        jobs.append((
            {"bench": "soi_grad_check", "averaging": avg},
            ["--task", "soi_grad", "--averaging", avg],
        ))

    # SiN tight_bend: fine ref, uniform & bf.mesh sweeps (12.36 x 12.36 x 4.80 um divides evenly by 120, 60, 40, 30, 24 nm)
    jobs.append((
        {"bench": "sin_bend", "kind": "uniform-24", "voxelizer": "greens", "averaging": "subpixel_full"},
        ["--task", "sin", "--kind", "uniform-24", "--voxelizer", "greens", "--averaging", "subpixel_full"],
    ))
    jobs.append((
        {"bench": "sin_bend", "kind": "mesh-28", "voxelizer": "greens", "averaging": "subpixel_full"},
        ["--task", "sin", "--kind", "mesh-28", "--voxelizer", "greens", "--averaging", "subpixel_full"],
    ))
    for kind in ("uniform-120", "uniform-60", "uniform-40", "uniform-30", "mesh-12", "mesh-16", "mesh-20", "mesh-24"):
        for vox in ("greens", "legacy"):
            for avg in ("subpixel", "subpixel_full"):
                jobs.append((
                    {"bench": "sin_bend", "kind": kind, "voxelizer": vox, "averaging": avg},
                    ["--task", "sin", "--kind", kind, "--voxelizer", vox, "--averaging", avg],
                ))

    for key_d, cli in jobs:
        if already(key_d):
            continue
        env = os.environ.copy()
        if key_d.get("kind") in ("uniform-12", "uniform-24"):
            env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
            env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        proc = subprocess.run(
            [sys.executable, __file__, "--worker", *cli],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if proc.returncode != 0:
            print(f"FAILED {key_d}:\n{proc.stderr[-1500:]}", flush=True)
            continue
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON:")), None)
        if line is None:
            print(f"NO RESULT {key_d}", flush=True)
            continue
        row = json.loads(line.split("RESULT_JSON:", 1)[1])
        results.append(row)
        out_path.write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
