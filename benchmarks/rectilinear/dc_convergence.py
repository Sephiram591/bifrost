"""The directional coupler's splitting ratio against the total cell count, on uniform grids and on bf.mesh's.

The silicon-nitride directional coupler of examples/fdtdx_directional_coupler.py (its geometry, materials, pulse,
source and mode monitors; the |E| plane is left out). The splitting ratio is the cross port's share of the power
that reaches the two outputs, ``P_cross / (P_cross + P_through)``, at the band's center (619 nm); the full spectra
are saved too. Grids:

* ``uniform-<nm>``: uniform cells (the box, 43 x 6.4 x 2.6 um, is whole cells at 20, 25, 40, 50 and 100 nm);
* ``mesh-<ppw>``: :func:`bifrost.mesh` over the band (538 to 728 nm) with ``ppw`` cells per wavelength in each
  material and the default tail, the core's faces centered in their cells (``center_faces``, the default);
  nothing moves;
* ``meshoff-<ppw>``: the same with ``center_faces=False``;
* ``lattice-<ppw>``, ``latticeoff-<ppw>``: the same on the 1 nm lattice (``quantum=0.001``);
* ``cells-<core nm>-<clad nm>``: :func:`bifrost.mesh` with the materials' cells given (``material_cells``);
* ``<grid>@<half y>[,<half z>]``: the box's half height in y (and z) changed, which shifts the lattice against
  the guides;
* ``uniform-<nm>%<f>``: the box one cell taller, so that both core faces sit at the fraction ``f`` of their cells.

One grid per process (clean peak memory); each appends one JSON line to ``out``::

    python benchmarks/rectilinear/dc_convergence.py results/dc_convergence.jsonl --grid mesh-14
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "examples"))


def run(kind: str) -> dict:
    """The spectra and the splitting ratio on one grid, with the grid's facts."""
    import fdtdx
    import fdtdx_directional_coupler as dc
    import jax

    import bifrost as bf

    name = kind
    # <grid>@<half box height in y>[,<in z>] (um): the lattice shifted against the guides
    kind, _, span = kind.partition("@")
    if span:
        half = [float(v) for v in span.split(",")]
        dc.HALF_SPAN = (half[0], half[1] if len(half) > 1 else dc.HALF_SPAN[1])
    # uniform-<nm>%<f>: the box one cell taller, f cells of it below and 1 - f above, so that both core faces
    # (whole cells from the box's faces before) sit at f of their cells
    kind, _, frac = kind.partition("%")
    if frac:
        f, h = float(frac), float(kind.split("-")[1]) * 1e-3
        dy = dc.HALF_SPAN[0] - (dc.PORT_GAP + 2 * dc.WIDTH) / 2
        dz = dc.HALF_SPAN[1] - dc.HEIGHT / 2
        stack = bf.stack.StackDevice.from_layer_stack(
            dc.directional_coupler(), dc.LAYER_STACK, dc.MATERIALS, background="clad", name="stack",
            zrange_layers=["core"], extend_ports=tuple((p, dc.EXTENSION) for p in ("o1", "o2", "o3", "o4")),
            boundary_offset=(0.0, 0.0, dy, -dy, dz + (1 - f) * h, -(dz + f * h)), averaging=dc.AVERAGING,
        )
    else:
        stack = dc.build_stack()
    summary = None
    if kind.startswith("uniform-"):
        res = float(kind.split("-")[1]) * 1e-3
        grid = fdtdx.core.grid.UniformGrid(spacing=res * 1e-6)
        bound = fdtdx.BoundaryConfig.from_uniform_bound(thickness=round(dc.PML / res))
        on_grid: dict = {}
    else:
        wavelengths = 1 / dc.frequencies()  # um
        band = (float(wavelengths.min()), float(wavelengths.max()))
        if kind.startswith("cells-"):  # cells-<core nm>-<clad nm>: bf.mesh on the uniform lattice (center_faces=False)
            core, clad = (float(v) * 1e-3 for v in kind.split("-")[1:3])
            m = bf.mesh(stack, band, pml=dc.PML, center_faces=False,
                        refine=bf.Refine(material_cells={"core": core, "clad": clad}))
        else:  # mesh-<ppw>, meshoff-<ppw>, lattice-<ppw> or latticeoff-<ppw>
            head, ppw = kind.split("-", 1)
            quantum = 0.001 if head.startswith("lattice") else None
            m = bf.mesh(stack, band, ppw=float(ppw), pml=dc.PML,
                        center_faces=not head.endswith("off"), quantum=quantum)
        summary = m.summary(time=dc.TIME)
        print(summary, flush=True)
        grid, bound = m.grid, m.boundary
    config = stack.get_simulation_config(time=dc.TIME, grid=grid)
    if not kind.startswith("uniform-"):
        on_grid = {"grid": config}
    center = fdtdx.WaveCharacter(wavelength=dc.WAVELENGTH * 1e-6)
    band = [fdtdx.WaveCharacter(wavelength=1e-6 / f) for f in dc.frequencies()]
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=center, spectral_width=fdtdx.WaveCharacter(frequency=dc.PULSE_WIDTH * center.get_frequency())
    )
    te0 = {"mode_index": 0, "filter_pol": "te"}
    mode = dict(wave_characters=band, scaling_mode="pulse", **te0)
    volume, constraints = stack.get_simulation_volume()
    objects = [volume, stack]
    for obj, c in (
        stack.create_mode_plane_source_at("o1", dc.padding([center]), wave_character=center, temporal_profile=pulse,
                                          port_offset=-0.3, name="source", **te0, **on_grid),
        stack.create_mode_overlap_detector_at("o1", dc.padding(band), direction="in", name="in", **mode, **on_grid),
        stack.create_mode_overlap_detector_at("o4", dc.padding(band), name="through", **mode, **on_grid),
        stack.create_mode_overlap_detector_at("o3", dc.padding(band), name="cross", **mode, **on_grid),
    ):
        objects.append(obj)
        constraints += c
    boundaries, c = fdtdx.boundary_objects_from_config(bound, volume)
    objects += boundaries.values()
    constraints += c
    key = jax.random.PRNGKey(0)
    t0 = time.perf_counter()
    objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
    arrays, objects, _ = bf.apply_stack(arrays, objects, {**params, **bf.init_stack_params(objects)}, key)
    jax.block_until_ready(arrays.inv_permittivities)
    t_setup = time.perf_counter() - t0
    t0 = time.perf_counter()
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
    stats = jax.devices()[0].memory_stats() or {}
    middle = [0.5 * (e[0] + e[-1]) for e in edges]  # um: the domain's middle in fdtdx's frame
    placement = {}  # each source / monitor's extent in stack coordinates (um), from the grid's edges
    for obj in ("source", "in", "through", "cross"):
        box = []
        for a, (lo, hi) in enumerate(objects[obj].grid_slice_tuple):
            offset = (stack.center[0], stack.center[1], stack.z_center)[a] - middle[a]
            box.append([round(float(edges[a][lo] + offset), 4), round(float(edges[a][hi] + offset), 4)])
        placement[obj] = box
    return {
        "grid": name, "cells": shape, "total_cells": int(np.prod(shape)),
        "min_cell_nm": [round(float(np.diff(e).min()) * 1e3, 2) for e in edges],
        "max_cell_nm": [round(float(np.diff(e).max()) * 1e3, 2) for e in edges],
        "steps": int(config.time_steps_total), "dt_fs": float(config.time_step_duration) * 1e15,
        "wavelength_nm": (1e3 / dc.frequencies()).tolist(), "through": through.tolist(), "cross": cross.tolist(),
        "split": float(cross[mid] / (cross[mid] + through[mid])),
        "setup_s": round(t_setup, 1), "run_s": round(t_run, 1),
        "peak_gb": round(stats.get("peak_bytes_in_use", 0) / 1e9, 2), "summary": summary,
        "placement": placement,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out")
    parser.add_argument("--grid", required=True, help="uniform-<nm> or mesh-<ppw>")
    args = parser.parse_args()
    import jax

    print("jax", jax.__version__, "on", jax.devices()[0].device_kind, flush=True)
    row = run(args.grid)
    with Path(args.out).open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k not in ("through", "cross", "wavelength_nm", "summary")}),
          flush=True)


if __name__ == "__main__":
    main()
