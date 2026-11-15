"""A silicon strip with sloped walls on grids graded in y and z: its effective index against uniform grids.

A 500 x 220 nm silicon strip in silica with 10 degree walls, along x through a 6 x 3.2 x 3.2 um box, TE0 at
1.55 um. Grids:

* ``uniform-20`` and ``uniform-40``: 20 and 40 nm cells;
* ``graded``: 20 nm along x (the same dispersion along the guide as uniform-20); in y and z 20 nm over the core
  and 0.2 um around it, growing to 60 nm in the cladding (17 uniform cells under every PML);
* ``graded-snap``: the same with the fine block's faces on uniform-20's edges, so its cells are uniform-20's and
  the painted permittivity can be compared cell by cell (``--save-eps``);
* ``graded-wide``: 20 nm cells on uniform-20's edges 0.6 um past the core, growing to 60 nm (the PMLs about 1 um,
  partly on the graded cells);
* ``tail-<um>``: 20 nm cells on uniform-20's edges ``<um>`` past the core (0 for none), growing to 60 nm, as
  graded-wide: the evanescent tail's own effect;
* ``mesh-<l>``: :func:`bifrost.mesh` with 20 nm cells in silicon (and its tails), 60 nm in silica
  (``material_cells``) and a global tail of ``<l>`` decay lengths, the fine cells on uniform-20's lattice;
* ``meshppw-<ppw>``: :func:`bifrost.mesh` with ``ppw`` cells per wavelength in every material and its default
  tail;
* ``lattice-<l>``, ``latticeppw-<ppw>``: the same on the 1 nm lattice (``quantum=0.001``).

``--save-eps`` saves the device's inverse permittivity on the yz plane at the box's middle, with the y and z edges
(um).

Per grid, the effective index two ways: ``neff_mode`` from fdtdx's mode solver on the stack's averaged cross-section
(the source's mode, solved on the grid; no time stepping), and ``neff_fdtd`` from the phase slope of Ey along the
guide's axis in a full run (it adds the Yee dispersion along x, the same for uniform-20 and graded). With
``--reference``, tidy3d's mode solver on the exact geometry (its own subpixel averaging) at 10 and 5 nm, and their
Richardson extrapolation. Also the cells, time step and steps, the run's time and the device's peak memory.

Run on a GPU (fdtdx) and on a CPU (the reference):
    python benchmarks/rectilinear/graded_guide.py out.json --grids uniform-20 uniform-40 graded
    python benchmarks/rectilinear/graded_guide.py ref.json --reference
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

WAVELENGTH = 1.55  # um
WIDTH, THICKNESS, ANGLE = 0.5, 0.22, 10.0  # um, um, degrees
N_CORE, N_CLAD = 3.48, 1.444
BOX = ((-3.0, 3.0), (-1.6, 1.6), (-1.48, 1.72))  # um
PML = 1.0  # um
LINE = (-1.2, 1.8)  # um: the Ey phasor line along the guide's axis (past the source at -1.7)
TRIM = 0.3  # um left out of the fit at both ends of the line


def pml_cells(e: np.ndarray, low: bool) -> int:
    """Cells from a face of edges e covering PML um."""
    w = np.diff(e) if low else np.diff(e)[::-1]
    return int(np.searchsorted(np.cumsum(w), PML - 1e-9) + 1)


def build(kind: str):
    """The placed objects, arrays, params and config on one grid."""
    import fdtdx
    import gdsfactory as gf
    import jax
    from gdsfactory.technology import LayerLevel, LayerStack

    import bifrost as bf

    gf.gpdk.PDK.activate()
    gf.snap.SNAP_ENABLED = False
    (x0, x1), (y0, y1), (z0, z1) = BOX
    c = gf.Component()
    h = WIDTH / 2
    c.add_polygon([(x0, -h), (x1, -h), (x1, h), (x0, h)], layer=gf.gpdk.LAYER.WG)
    c.add_port(name="o1", center=(x0, 0.0), width=WIDTH, orientation=180, layer=gf.gpdk.LAYER.WG)
    c.add_port(name="o2", center=(x1, 0.0), width=WIDTH, orientation=0, layer=gf.gpdk.LAYER.WG)
    ls = LayerStack(layers={"core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=THICKNESS, material="si",
                                               mesh_order=1, sidewall_angle=ANGLE)})
    mats = {"si": fdtdx.Material(permittivity=N_CORE**2), "clad": fdtdx.Material(permittivity=N_CLAD**2)}
    stack = bf.StackDevice.from_layer_stack(
        c, ls, mats, background="clad", zrange_layers=["core"], name="stack", diagnostics=False,
        boundary_offset=(0.0, 0.0, y1 - h, y0 + h, z1 - THICKNESS, z0),
    )
    if kind.startswith(("mesh", "lattice")):
        head, val = kind.split("-", 1)
        quantum = 0.001 if head.startswith("lattice") else None
        if head.endswith("ppw"):
            m = bf.mesh(stack, WAVELENGTH, ppw=float(val), pml=PML, quantum=quantum)
        else:  # 20 nm in silicon (on uniform-20's lattice), 60 nm in silica, the tail in decay lengths
            tail = float(val)
            m = bf.mesh(stack, WAVELENGTH, ppw=WAVELENGTH / (N_CORE * 0.02), pml=PML, center_faces=False,
                        quantum=quantum,
                        refine=bf.Refine(global_dilation=tail, global_z_padding=tail, material_cells={"clad": 0.06}))
        print(m.summary(), flush=True)
        grid, bound = m.grid, m.boundary
    elif kind.startswith(("graded", "tail")):
        n = math.ceil(PML / 0.06 - 1e-9)
        if kind == "graded":
            fy, fz, snap, ends = (-h - 0.2, h + 0.2), (-0.2, THICKNESS + 0.2), False, (n, n)
        elif kind == "graded-snap":  # the fine block's faces on uniform-20's edges (-1.6 + 0.02 j, -1.48 + 0.02 k)
            fy, fz, snap, ends = (-0.46, 0.46), (-0.2, 0.42), True, (n, n)
        elif kind.startswith("tail-"):  # the block t um past the core, out to uniform-20's next edges
            t = float(kind.split("-")[1])

            def lattice(a: float, base: float, up: bool) -> float:
                """The uniform-20 edge (from base) at or past a, outward."""
                k = math.ceil((a - base) / 0.02 - 1e-9) if up else math.floor((a - base) / 0.02 + 1e-9)
                return base + 0.02 * k

            fy = (lattice(-h - t, y0, False), lattice(h + t, y0, True))
            fz = (lattice(-t, z0, False), lattice(THICKNESS + t, z0, True))
            snap, ends = True, (0, 0)
        else:
            fy, fz, snap, ends = (-0.86, 0.86), (-0.6, 0.82), True, (0, 0)
        ey, ez = (
            bf.graded_edges(lo, hi, 0.06, fine=[(*f, 0.02)], growth=1.25, uniform_ends=ends, snap=f if snap else ())
            for (lo, hi), f in (((y0, y1), fy), ((z0, z1), fz))
        )
        grid = stack.rectilinear_grid(0.02, y_um=ey, z_um=ez)
        nx = round(PML / 0.02)
        ny0, ny1, nz0, nz1 = (pml_cells(e, low) for e in (ey, ez) for low in (True, False))
        bound = fdtdx.BoundaryConfig(thickness_grid_minx=nx, thickness_grid_maxx=nx, thickness_grid_miny=ny0,
                                     thickness_grid_maxy=ny1, thickness_grid_minz=nz0, thickness_grid_maxz=nz1)
    else:
        res = float(kind.split("-")[1]) * 1e-3
        grid = fdtdx.core.grid.UniformGrid(spacing=res * 1e-6)
        bound = fdtdx.BoundaryConfig.from_uniform_bound(thickness=round(PML / res))
    config = stack.get_simulation_config(time=200e-15, grid=grid)
    wave = fdtdx.WaveCharacter(wavelength=WAVELENGTH * 1e-6)
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=wave, spectral_width=fdtdx.WaveCharacter(frequency=0.1 * wave.get_frequency())
    )
    volume, constraints = stack.get_simulation_volume()
    source, placing = stack.create_mode_plane_source_at(
        "o1", (0.6, 0.6), wave_character=wave, temporal_profile=pulse, port_offset=1.3, name="source",
        mode_index=0, filter_pol="te", **({} if kind.startswith("uniform") else {"grid": config}),
    )
    constraints += placing
    line = fdtdx.PhasorDetector(
        name="line", partial_real_shape=((LINE[1] - LINE[0]) * 1e-6, None, None), partial_grid_shape=(None, 1, 1),
        wave_characters=(wave,), components=("Ey",), scaling_mode="pulse",
    )
    mid = [0.5 * (lo + hi) for lo, hi in BOX]
    constraints.append(line.place_relative_to(
        volume, axes=(0, 1, 2), own_positions=(0, 0, 0), other_positions=(0, 0, 0),
        margins=((0.5 * (LINE[0] + LINE[1]) - mid[0]) * 1e-6, (0.0 - mid[1]) * 1e-6, (THICKNESS / 2 - mid[2]) * 1e-6),
    ))
    boundaries, placing = fdtdx.boundary_objects_from_config(bound, volume)
    constraints += placing
    objects, arrays, params, config, _ = fdtdx.place_objects(
        [volume, stack, source, line, *boundaries.values()], config, constraints, jax.random.PRNGKey(0)
    )
    return objects, arrays, {**params, **bf.init_stack_params(objects)}, config


def run(kind: str, save_eps: str | None = None, mode_only: bool = False) -> dict:
    """neff from the mode solver and from the run on one grid, with its facts."""
    import fdtdx
    import jax
    import jax.numpy as jnp

    import bifrost as bf

    t0 = time.perf_counter()
    objects, arrays, params, config = build(kind)
    t_place = time.perf_counter() - t0
    key = jax.random.PRNGKey(0)
    t0 = time.perf_counter()
    arrays, objects, _ = bf.apply_stack(arrays, objects, params, key)
    jax.block_until_ready(arrays.inv_permittivities)
    t_apply = time.perf_counter() - t0
    neff_mode = complex(np.asarray(objects["source"]._neff).reshape(-1)[0])
    if save_eps:  # the yz plane at the box's middle, and its edges in stack coordinates (um)
        (sx0, sx1), (sy0, sy1), (sz0, sz1) = objects["stack"].grid_slice_tuple
        inv = np.asarray(arrays.inv_permittivities[:, (sx0 + sx1) // 2, sy0:sy1, sz0:sz1])
        ys, zs = (np.asarray(config.resolved_grid.edges(a), dtype=np.float64)[lo : hi + 1] * 1e6
                  for a, (lo, hi) in ((1, (sy0, sy1)), (2, (sz0, sz1))))
        (_, _), (by0, by1), (bz0, bz1) = BOX
        np.savez(save_eps, inv=inv, y=ys - 0.5 * (ys[0] + ys[-1]) + 0.5 * (by0 + by1),
                 z=zs - 0.5 * (zs[0] + zs[-1]) + 0.5 * (bz0 + bz1))

    if mode_only:  # the mode solver's neff alone (no time stepping)
        shape = [len(np.asarray(config.resolved_grid.edges(a))) - 1 for a in range(3)]
        return {"grid": kind, "cells": shape, "neff_mode": neff_mode.real, "apply_stack_s": round(t_apply, 2)}

    def solve(arrays):
        _, state = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key, show_progress=False)
        return state.detector_states["line"]["phasor"]

    solve = jax.jit(solve)
    t0 = time.perf_counter()
    phasor = jax.block_until_ready(solve(arrays))
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    phasor = jax.block_until_ready(solve(arrays))
    t_run = time.perf_counter() - t0
    e = np.asarray(jnp.reshape(phasor, (-1,)))
    edges = np.asarray(config.resolved_grid.edges(0)) * 1e6
    (i0, i1), _, _ = objects["line"].grid_slice_tuple
    x = 0.5 * (edges[i0:i1] + edges[i0 + 1 : i1 + 1]) - 0.5 * (edges[0] + edges[-1])  # um from the box's middle
    keep = (x > LINE[0] + TRIM) & (x < LINE[1] - TRIM)
    phase = np.unwrap(np.angle(e[keep]))
    slope = np.polyfit(x[keep], phase, 1)[0]
    k0 = 2 * np.pi / WAVELENGTH
    shape = [len(np.asarray(config.resolved_grid.edges(a))) - 1 for a in range(3)]
    stats = jax.devices()[0].memory_stats() or {}
    steps = int(config.time_steps_total)
    return {
        "grid": kind, "cells": shape, "total_cells": int(np.prod(shape)),
        "min_cell_nm": [round(float(np.min(np.diff(np.asarray(config.resolved_grid.edges(a))))) * 1e9, 2)
                        for a in range(3)],
        "dt_fs": float(config.time_step_duration) * 1e15, "steps": steps,
        "neff_mode": neff_mode.real, "neff_fdtd": abs(slope) / k0,
        "amplitude_ripple": float(np.std(np.abs(e[keep])) / np.mean(np.abs(e[keep]))),
        "place_s": round(t_place, 1), "apply_stack_s": round(t_apply, 2), "run_first_s": round(t_first, 2),
        "run_s": round(t_run, 2), "ns_per_cell_step": t_run / np.prod(shape) / steps * 1e9,
        "peak_gb": round(stats.get("peak_bytes_in_use", 0) / 1e9, 2),
    }


def reference() -> dict:
    """tidy3d's mode solver on the exact sloped strip at 10, 5 and 2.5 nm (a 2 um square: the mode is gone by
    its sides), the observed order of convergence and the extrapolation with it."""
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver

    h = WIDTH / 2
    core = td.Structure(
        geometry=td.PolySlab(vertices=[(-5.0, -h), (5.0, -h), (5.0, h), (-5.0, h)], slab_bounds=(0.0, THICKNESS),
                             axis=2, sidewall_angle=math.radians(ANGLE), reference_plane="bottom"),
        medium=td.Medium(permittivity=N_CORE**2),
    )
    out = {}
    for dl in (0.01, 0.005, 0.0025):
        sim = td.Simulation(
            size=(1.0, 2.0, 2.0), center=(0.0, 0.0, THICKNESS / 2), grid_spec=td.GridSpec.uniform(dl=dl),
            structures=[core], medium=td.Medium(permittivity=N_CLAD**2), run_time=1e-12,
            boundary_spec=td.BoundarySpec.all_sides(td.PML()),
        )
        solver = ModeSolver(simulation=sim, plane=td.Box(center=(0.0, 0.0, THICKNESS / 2), size=(0.0, 2.0, 2.0)),
                            mode_spec=td.ModeSpec(num_modes=2, target_neff=3.0), freqs=[td.C_0 / WAVELENGTH])
        data = solver.solve()
        out[f"tidy3d_{dl * 1e3:g}nm"] = float(np.asarray(data.n_eff.values).reshape(-1)[0])
    a, b, c = out["tidy3d_10nm"], out["tidy3d_5nm"], out["tidy3d_2.5nm"]
    order = math.log2((a - b) / (b - c))
    out["tidy3d_order"] = order
    out["tidy3d_extrapolated"] = c + (c - b) / (2**order - 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out")
    parser.add_argument("--grids", nargs="+", default=[])
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--save-eps", default=None, help="with one grid: save its yz inverse permittivity here")
    parser.add_argument("--mode-only", action="store_true", help="the mode solver's neff only (no run)")
    args = parser.parse_args()
    rows = []
    if args.reference:
        rows.append(reference())
        print(json.dumps(rows[-1]), flush=True)
    if args.grids:
        import jax

        print("jax", jax.__version__, "on", jax.devices()[0].device_kind, flush=True)
    for kind in args.grids:
        rows.append(run(kind, args.save_eps, args.mode_only))
        print(json.dumps(rows[-1]), flush=True)
    Path(args.out).write_text(json.dumps({"rows": rows}, indent=1))


if __name__ == "__main__":
    main()
