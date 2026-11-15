"""The inverse-design bend on graded x, y and z grids: transmission, compile-once and cost against uniform grids.

The silicon nitride bend of examples/guide/inverse_design.py (its geometry, materials, ports and objective), on

* ``uniform-60``: 60 nm cells (the grid the example optimizes on);
* ``uniform-40``: 40 nm cells (the grid the example checks on: the reference here);
* ``graded``: 40 nm cells over the guide and its mode's reach in x and y (where the optimized edges can go) and
  around the core in z, growing to 80 nm in the margins, the silica and the air; every PML about 1 um, on 40 nm
  cells where a port's straight lead runs through it and 80 nm elsewhere;
* ``graded-60``: the same with 60 and 120 nm cells (to optimize on instead of uniform-60);
* ``mesh``: :func:`bifrost.mesh` with graded's choices (40 nm in the core, 80 nm in the air and, through
  ``material_cells``, in the silica), the edges' reach as ``dilation={"WG": 0.45}`` and the default tail;
* ``mesh-auto``, ``mesh-auto-<ppw>``: :func:`bifrost.mesh` with only a ppw (14 by default) and the dilation;
* ``lattice``, ``lattice-auto``, ``lattice-auto-<ppw>``: the same on the 1 nm lattice (``quantum=0.001``).

Per grid (one process each, for clean peak memory): the TE0 transmission of the uniform-width bend (``bf.jit`` of the
objective, compile and run timed apart), and with ``--steps`` that many Adam steps of the example's
``bf.jit(jax.value_and_grad(objective))``, counting JAX backend compiles per step and the jit's cache size (a fixed
grid must compile once). Also the cells, time step and steps, and the device's peak memory.

Run on a GPU:
    python benchmarks/rectilinear/graded_bend.py out.json --grid graded --steps 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "guide"))

PML = 1.0  # um
GRADED = {"graded": (0.04, 0.08), "graded-60": (0.06, 0.12)}  # um: the graded grids' fine and coarse cells
# fine regions (um): the guide with its leads and the reach of the optimized edges (half-widths up to 1.25 um)
# plus the mode's tail; the core with its mode's reach into the silica and the air
FINE_X, FINE_Y, FINE_Z = (None, 8.7), (-1.7, None), (-0.45, 1.2)  # None: from the box's face


def pml_cells(e: np.ndarray, low: bool) -> int:
    """Cells from a face of edges e covering PML um."""
    w = np.diff(e) if low else np.diff(e)[::-1]
    return int(np.searchsorted(np.cumsum(w), PML - 1e-9) + 1)


def graded_axis(lo: float, hi: float, region: tuple, fine: float, coarse: float) -> tuple[np.ndarray, int, int]:
    """Edges over [lo, hi] with fine cells in region (None: up to that face) and their PML cells per face."""
    import bifrost as bf

    a = lo if region[0] is None else region[0]
    b = hi if region[1] is None else region[1]
    n = math.ceil(PML / coarse - 1e-9)  # whole coarse cells covering the PML
    ends = (0 if region[0] is None else n, 0 if region[1] is None else n)
    e = bf.graded_edges(lo, hi, coarse, fine=[(a, b, fine)], growth=1.25, uniform_ends=ends)
    return e, pml_cells(e, True), pml_cells(e, False)


def place(kind: str, checkpoints: int = 60):
    """The uniform bend on this grid, with the example's source and monitors."""
    import fdtdx
    import inverse_design as ex

    import bifrost as bf

    z_margin = (ex.BOX_HEIGHT - ex.THICKNESS) / 2
    stack = bf.StackDevice.from_layer_stack(
        ex.tight_bend(ex.UNIFORM), ex.layer_stack, ex.materials, background="air", zrange_layers=["core"],
        extend_ports=(("o1", ex.EXTENSION), ("o2", ex.EXTENSION)),
        boundary_offset=(ex.MARGIN, 0, 0, -ex.MARGIN, z_margin, -z_margin), name="stack",
    )
    middle = (*stack.center, stack.z_center)
    size = [s * 1e6 for s in stack.partial_real_shape]
    box = [(m - s / 2, m + s / 2) for m, s in zip(middle, size, strict=True)]
    if kind.startswith(("mesh", "lattice")):
        quantum = 0.001 if kind.startswith("lattice") else None
        if kind in ("mesh", "lattice"):  # graded's choices: 40 nm in the core, 80 nm in the air and the silica
            m = bf.mesh(stack, ex.WAVELENGTH, ppw=ex.WAVELENGTH / (ex.CORE_INDEX * 0.04), pml=PML, quantum=quantum,
                        refine=bf.Refine(dilation={"WG": 0.45}, material_cells={"silica": 0.08}))
        else:  # <mesh|lattice>-auto or <mesh|lattice>-auto-<ppw>
            ppw = float(kind.split("-")[2]) if kind.count("-") == 2 else 14.0
            m = bf.mesh(stack, ex.WAVELENGTH, ppw=ppw, pml=PML, quantum=quantum, refine=bf.Refine(dilation={"WG": 0.45}))
        print(m.summary(time=ex.DURATION), flush=True)
        grid, bound = m.grid, m.boundary
    elif kind in GRADED:
        fine, coarse = GRADED[kind]
        regions = (FINE_X, FINE_Y, FINE_Z)
        axes = [graded_axis(lo, hi, r, fine, coarse) for (lo, hi), r in zip(box, regions, strict=True)]
        grid = stack.rectilinear_grid(coarse, *(e for e, _, _ in axes))
        (nx0, nx1), (ny0, ny1), (nz0, nz1) = ((a, b) for _, a, b in axes)
        bound = fdtdx.BoundaryConfig(thickness_grid_minx=nx0, thickness_grid_maxx=nx1, thickness_grid_miny=ny0,
                                     thickness_grid_maxy=ny1, thickness_grid_minz=nz0, thickness_grid_maxz=nz1)
    else:
        res = float(kind.split("-")[1]) * 1e-3
        grid = fdtdx.core.grid.UniformGrid(spacing=res * 1e-6)
        bound = fdtdx.BoundaryConfig.from_uniform_bound(thickness=round(PML / res))
    config = stack.get_simulation_config(
        time=ex.DURATION, grid=grid,
        gradient_config=fdtdx.GradientConfig(method="checkpointed", num_checkpoints=checkpoints),
    )
    wave = fdtdx.WaveCharacter(wavelength=ex.WAVELENGTH * 1e-6)
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=wave, spectral_width=fdtdx.WaveCharacter(frequency=0.05 * wave.get_frequency())
    )
    te0 = {"mode_index": 0, "filter_pol": "te"}
    on_grid = {"grid": config} if kind in GRADED or kind.startswith(("mesh", "lattice")) else {}
    volume, constraints = stack.get_simulation_volume()
    objects = [volume, stack]
    for obj, placing in (
        stack.create_mode_plane_source_at("o1", ex.PADDING, wave_character=wave, temporal_profile=pulse,
                                          port_offset=-1.0, name="source", **te0, **on_grid),
        stack.create_mode_overlap_detector_at("o1", ex.PADDING, direction="in", wave_characters=(wave,),
                                              scaling_mode="pulse", port_offset=-0.5, name="in", **te0, **on_grid),
        stack.create_mode_overlap_detector_at("o2", ex.PADDING, wave_characters=(wave,), scaling_mode="pulse",
                                              port_offset=-0.5, name="out", **te0, **on_grid),
    ):
        objects.append(obj)
        constraints += placing
    boundaries, placing = fdtdx.boundary_objects_from_config(bound, volume)
    objects += boundaries.values()
    constraints += placing
    objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, ex.KEY)
    return objects, arrays, {**params, **bf.init_stack_params(objects)}, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out")
    parser.add_argument("--grid", required=True,
                        help=f"uniform-60, uniform-40, {', '.join(GRADED)}, mesh, mesh-auto or mesh-auto-<ppw>")
    parser.add_argument("--checkpoints", type=int, default=60, help="checkpoints of the backward pass")
    parser.add_argument("--steps", type=int, default=0, help="Adam steps of the jitted value_and_grad")
    parser.add_argument("--commit", action="store_true",
                        help="commit the starting parameters to the device (jax.device_put), as the updated ones are")
    args = parser.parse_args()
    import inverse_design as ex
    import jax
    import jax.numpy as jnp
    import optax

    import bifrost as bf

    compiles: list[float] = []

    def listener(event: str, duration: float, **_: object) -> None:
        if event == "/jax/core/compile/backend_compile_duration":
            compiles.append(duration)

    jax.monitoring.register_event_duration_secs_listener(listener)
    device = jax.devices()[0]
    print("jax", jax.__version__, "on", device.device_kind, flush=True)
    t0 = time.perf_counter()
    placed = place(args.grid, args.checkpoints)
    objects, arrays, params, config = placed
    stack = objects["stack"]
    t_place = time.perf_counter() - t0
    edges = [np.asarray(config.resolved_grid.edges(a)) * 1e6 for a in range(3)]
    facts = {
        "grid": args.grid, "cells": [len(e) - 1 for e in edges], "total_cells": int(np.prod([len(e) - 1 for e in edges])),
        "min_cell_nm": [round(float(np.min(np.diff(e))) * 1e3, 2) for e in edges],
        "max_cell_nm": [round(float(np.max(np.diff(e))) * 1e3, 2) for e in edges],
        "dt_fs": float(config.time_step_duration) * 1e15, "steps": int(config.time_steps_total),
        "corner_columns": stack.corner_columns, "place_s": round(t_place, 1),
    }
    print(json.dumps(facts), flush=True)

    def objective(z, arrays, params):
        return ex.simulate(ex.half_widths_of(z), (objects, arrays, params, config))[0]

    z = jnp.asarray(ex.parameters_of(ex.UNIFORM), dtype=jnp.float64)
    forward = bf.jit(objective)
    n0 = len(compiles)
    t0 = time.perf_counter()
    transmission = float(forward(z, arrays, params))
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    float(forward(z, arrays, params))
    t_second = time.perf_counter() - t0
    out = {
        **facts, "transmission": transmission, "forward_first_s": round(t_first, 2),
        "forward_s": round(t_second, 2), "forward_compile_s": round(sum(compiles[n0:]), 2),
        "forward_compiles": len(compiles) - n0, "ns_per_cell_step": t_second / facts["total_cells"] / facts["steps"] * 1e9,
    }
    stats = device.memory_stats() or {}
    out["peak_gb_forward"] = round(stats.get("peak_bytes_in_use", 0) / 1e9, 2)
    print(json.dumps(out), flush=True)

    if args.steps:
        if args.commit:
            z = jax.device_put(z, device)
        value_and_grad = bf.jit(jax.value_and_grad(objective))
        optimizer = optax.adam(0.03)
        state = optimizer.init(z)
        history = []
        for step in range(args.steps):
            n0 = len(compiles)
            t0 = time.perf_counter()
            value, gradient = value_and_grad(z, arrays, params)
            value = float(value)
            seconds = time.perf_counter() - t0
            history.append({"step": step, "z_dtype": str(z.dtype), "z_weak": bool(z.weak_type),
                            "z_committed": getattr(z, "_committed", None), "T": value, "seconds": round(seconds, 2),
                            "backend_compiles": len(compiles) - n0, "compile_s": round(sum(compiles[n0:]), 2),
                            "grad_norm": float(jnp.linalg.norm(gradient))})
            print(json.dumps(history[-1]), flush=True)
            updates, state = optimizer.update(-gradient, state)
            z = optax.apply_updates(z, updates)
        stats = device.memory_stats() or {}
        out.update({"optimization": history, "jit_cache_size": getattr(value_and_grad, "_cache_size", lambda: None)(),
                    "peak_gb": round(stats.get("peak_bytes_in_use", 0) / 1e9, 2)})
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "optimization"}), flush=True)


if __name__ == "__main__":
    main()
