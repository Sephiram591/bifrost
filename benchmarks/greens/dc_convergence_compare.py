"""Broadband SiN directional coupler convergence: greens vs legacy across uniform, graded bf.mesh, and shifted grids.

Mirrors ``benchmarks/rectilinear/dc_convergence.py`` using clean per-configuration
worker subprocesses so peak GPU memory and compile/setup times are isolated.
"""

from __future__ import annotations

import argparse
import json
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


def _run_one(kind: str, averaging: str, voxelizer: str) -> dict[str, Any]:
    import fdtdx
    import fdtdx_directional_coupler as dc
    import jax

    import bifrost as bf

    assert Path(bf.__file__).resolve().is_relative_to(WT_ROOT), bf.__file__
    dc.AVERAGING = averaging

    name = f"{kind}:{voxelizer}:{averaging}"
    frac = ""
    base_kind, _, frac = kind.partition("%")

    if frac:
        f_val, h = float(frac), float(base_kind.split("-")[1]) * 1e-3
        dy = dc.HALF_SPAN[0] - (dc.PORT_GAP + 2 * dc.WIDTH) / 2
        dz = dc.HALF_SPAN[1] - dc.HEIGHT / 2
        stack = bf.StackDevice.from_layer_stack(
            dc.directional_coupler(),
            dc.LAYER_STACK,
            dc.MATERIALS,
            background="clad",
            name="stack",
            zrange_layers=["core"],
            extend_ports=tuple((p, dc.EXTENSION) for p in ("o1", "o2", "o3", "o4")),
            boundary_offset=(0.0, 0.0, dy, -dy, dz + (1.0 - f_val) * h, -(dz + f_val * h)),
            averaging=averaging,
            voxelizer=voxelizer,
        )
    else:
        # Rebuild exact dc.build_stack() with requested voxelizer & averaging
        half_w = (dc.PORT_GAP + 2 * dc.WIDTH) / 2
        dy = dc.HALF_SPAN[0] - half_w
        dz = dc.HALF_SPAN[1] - dc.HEIGHT / 2
        stack = bf.StackDevice.from_layer_stack(
            dc.directional_coupler(),
            dc.LAYER_STACK,
            dc.MATERIALS,
            background="clad",
            name="stack",
            zrange_layers=["core"],
            extend_ports=tuple((p, dc.EXTENSION) for p in ("o1", "o2", "o3", "o4")),
            boundary_offset=(0.0, 0.0, dy, -dy, dz, -dz),
            averaging=averaging,
            voxelizer=voxelizer,
        )

    summary = None
    if base_kind.startswith("uniform-"):
        res = float(base_kind.split("-")[1]) * 1e-3
        grid = fdtdx.core.grid.UniformGrid(spacing=res * 1e-6)
        bound = fdtdx.BoundaryConfig.from_uniform_bound(thickness=round(dc.PML / res))
        on_grid: dict[str, Any] = {}
    elif base_kind.startswith("mesh-"):
        ppw = float(base_kind.split("-")[1])
        wavelengths = 1.0 / dc.frequencies()
        band_um = (float(wavelengths.min()), float(wavelengths.max()))
        m = bf.mesh(stack, band_um, ppw=ppw, pml=dc.PML, center_faces=True)
        summary = m.summary(time=dc.TIME)
        grid, bound = m.grid, m.boundary
    else:
        raise ValueError(f"Unknown grid kind {kind!r}")

    config = stack.get_simulation_config(time=dc.TIME, grid=grid)
    if not base_kind.startswith("uniform-"):
        on_grid = {"grid": config}

    center = fdtdx.WaveCharacter(wavelength=dc.WAVELENGTH * 1e-6)
    band = [fdtdx.WaveCharacter(wavelength=1e-6 / f) for f in dc.frequencies()]
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=center,
        spectral_width=fdtdx.WaveCharacter(frequency=dc.PULSE_WIDTH * center.get_frequency()),
    )
    te0 = {"mode_index": 0, "filter_pol": "te"}
    mode = dict(wave_characters=band, scaling_mode="pulse", **te0)
    volume, constraints = stack.get_simulation_volume()
    objects = [volume, stack]
    for obj, c in (
        stack.create_mode_plane_source_at(
            "o1", dc.padding([center]), wave_character=center, temporal_profile=pulse,
            port_offset=-0.3, name="source", **te0, **on_grid
        ),
        stack.create_mode_overlap_detector_at("o1", dc.padding(band), direction="in", name="in", **mode, **on_grid),
        stack.create_mode_overlap_detector_at("o4", dc.padding(band), name="through", **mode, **on_grid),
        stack.create_mode_overlap_detector_at("o3", dc.padding(band), name="cross", **mode, **on_grid),
    ):
        objects.append(obj)
        constraints += c
    boundaries, c = fdtdx.boundary_objects_from_config(bound, volume)
    objects += list(boundaries.values())
    constraints += c
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

    states = arrays.detector_states
    a_in, a_through, a_cross = (objects[n].compute_overlap(states[n]) for n in ("in", "through", "cross"))
    through = np.abs(np.asarray(a_through / a_in)) ** 2
    cross = np.abs(np.asarray(a_cross / a_in)) ** 2
    mid = dc.N_FREQ // 2
    edges = [np.asarray(config.resolved_grid.edges(a), dtype=np.float64) * 1e6 for a in range(3)]
    shape = [len(e) - 1 for e in edges]
    mem = jax.devices()[0].memory_stats() or {}
    return {
        "name": name,
        "kind": kind,
        "voxelizer": voxelizer,
        "averaging": averaging,
        "cells": shape,
        "total_cells": int(np.prod(shape)),
        "min_cell_nm": [round(float(np.diff(e).min()) * 1e3, 2) for e in edges],
        "max_cell_nm": [round(float(np.diff(e).max()) * 1e3, 2) for e in edges],
        "steps": int(config.time_steps_total),
        "wavelength_nm": (1e3 / dc.frequencies()).tolist(),
        "through": through.tolist(),
        "cross": cross.tolist(),
        "split_mid": float(cross[mid] / (cross[mid] + through[mid])),
        "total_mid": float(cross[mid] + through[mid]),
        "setup_s": round(t_setup, 2),
        "run_s": round(t_run, 2),
        "peak_gb": round(mem["peak_bytes_in_use"] / 1e9, 3) if "peak_bytes_in_use" in mem else None,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="benchmarks/greens/results/dc_convergence.json")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--kind", default="uniform-50")
    parser.add_argument("--averaging", default="subpixel")
    parser.add_argument("--voxelizer", default="greens")
    args = parser.parse_args()

    if args.worker:
        row = _run_one(args.kind, args.averaging, args.voxelizer)
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
    done = {(r["kind"], r["voxelizer"], r["averaging"]) for r in results}

    # Reference first, then uniform, mesh, and sub-cell shift sweep
    jobs: list[tuple[str, str, str]] = [
        ("mesh-16", "greens", "subpixel"),
        ("uniform-20", "greens", "subpixel"),
    ]
    for kind in ("uniform-100", "uniform-50", "uniform-40", "uniform-25", "mesh-8", "mesh-10", "mesh-12", "mesh-14"):
        for vox in ("greens", "legacy"):
            for avg in ("subpixel", "subpixel_full"):
                jobs.append((kind, vox, avg))
    for f in (0.0, 0.2, 0.4, 0.6, 0.8):
        for vox in ("greens", "legacy"):
            for avg in ("subpixel", "subpixel_full"):
                jobs.append((f"uniform-40%{f}", vox, avg))

    for kind, vox, avg in jobs:
        if (kind, vox, avg) in done:
            print(f"Skipping ({kind}, {vox}, {avg})", flush=True)
            continue
        env = os.environ.copy()
        if kind in ("mesh-16", "uniform-20", "uniform-25", "mesh-10", "mesh-12", "mesh-14"):
            env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
            env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        cmd = [sys.executable, __file__, "--worker", "--kind", kind, "--voxelizer", vox, "--averaging", avg]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        if proc.returncode != 0:
            print(f"FAILED ({kind}, {vox}, {avg}):\n{proc.stderr[-1500:]}", flush=True)
            continue
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON:")), None)
        if line is None:
            print(f"NO RESULT ({kind}, {vox}, {avg})", flush=True)
            continue
        row = json.loads(line.split("RESULT_JSON:", 1)[1])
        results.append(row)
        out_path.write_text(json.dumps(results, indent=2))
        pk_str = f"{row['peak_gb']:.2f}GB" if row["peak_gb"] is not None else "n/a"
        print(
            f"{kind:14s} {vox:6s} {avg:13s} | split={row['split_mid']:.5f} T={row['total_mid']:.4f} "
            f"setup={row['setup_s']:.1f}s run={row['run_s']:.1f}s peak={pk_str}",
            flush=True,
        )


if __name__ == "__main__":
    main()
