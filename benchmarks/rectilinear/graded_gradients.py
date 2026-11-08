"""Gradients on graded x, y and z grids against central finite differences (float64).

A silicon strip in silica (0.5 um wide, 0.22 um thick, 8 degree walls) with a bump on one side (corners for the
corner model), in a 2.4 x 2.0 x 1.32 um box, on a grid graded along x, y and z (25 nm cells around the bump and the
core, 50 nm elsewhere; 7 uniform cells under every PML face), or on a uniform 40 nm grid as the control. For the
core level's thickness, zmin and sidewall angle and one vertex of the bump, the derivative from ``jax.grad`` is
compared with central differences at two steps, of:

* ``weights``: the silicon fractions weighted by a fixed random field (``material_weights``);
* ``fdtd``: the energy of Ey at 1.55 um on a plane past the bump, from a point dipole in the core (a full fdtdx
  run, backward through the checkpointed time loop). A dipole and a phasor detector read no mode, so both sides
  differentiate the same function (``apply_stack`` re-solves a mode source's mode outside autodiff).

Run on a CPU node (float64):
    python benchmarks/rectilinear/graded_gradients.py out.json --averaging subpixel subpixel_full
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

BOX = ((-1.2, 1.2), (-1.0, 1.0), (-0.56, 0.76))  # um
BUMP = (0.0, 0.4, 0.25, 0.4)  # x0, x1, y0, y1 (um): a bump on the strip's +y side
PARAMS = {  # name: (finite-difference steps)
    "thickness/core": (1e-4, 1e-5),
    "zmin/core": (1e-4, 1e-5),
    "sidewall_angle/core": (1e-3, 1e-4),
    "vertex": (1e-4, 1e-5),  # x of the bump's outer corner (0.4, 0.4)
}


def edges(kind: str) -> list[np.ndarray]:
    """x, y, z edges (um) over BOX."""
    import bifrost as bf

    if kind == "uniform":
        return [np.linspace(lo, hi, round((hi - lo) / 0.04) + 1) for lo, hi in BOX]
    fine = ((-0.15, 0.55), (-0.4, 0.55), (-0.08, 0.3))  # around the bump, the strip, the core
    return [bf.graded_edges(lo, hi, 0.05, fine=[(a, b, 0.025)], growth=1.25, uniform_ends=(7, 7))
            for (lo, hi), (a, b) in zip(BOX, fine, strict=True)]


def build(kind: str, averaging: str, zmin: float = 0.0):
    """The placed objects, arrays, params and config, and the bump vertex's index."""
    import fdtdx
    import gdsfactory as gf
    import jax
    import jax.numpy as jnp
    from gdsfactory.technology import LayerLevel, LayerStack

    import bifrost as bf

    gf.gpdk.PDK.activate()
    gf.snap.SNAP_ENABLED = False
    x0, x1 = BOX[0]
    bx0, bx1, by0, by1 = BUMP
    c = gf.Component()
    c.add_polygon([(x0, -0.25), (x1, -0.25), (x1, by0), (bx1, by0), (bx1, by1), (bx0, by1), (bx0, by0), (x0, by0)],
                  layer=gf.gpdk.LAYER.WG)
    c.add_port(name="o1", center=(x0, 0.0), width=0.5, orientation=180, layer=gf.gpdk.LAYER.WG)
    c.add_port(name="o2", center=(x1, 0.0), width=0.5, orientation=0, layer=gf.gpdk.LAYER.WG)
    ls = LayerStack(layers={"core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=zmin, thickness=0.22, material="si",
                                               mesh_order=1, sidewall_angle=8.0)})
    mats = {"si": fdtdx.Material(permittivity=3.48**2), "clad": fdtdx.Material(permittivity=1.444**2)}
    (_, _), (y0, y1), (z0, z1) = BOX
    stack = bf.StackDevice.from_layer_stack(
        c, ls, mats, background="clad", zrange_layers=["core"], name="stack", dtype=jnp.float64,
        boundary_offset=(0.0, 0.0, y1 - by1, y0 + 0.25, z1 - 0.22 - zmin, z0 - zmin), diagnostics=False,
        averaging=averaging,
    )
    ex, ey, ez = edges(kind)
    grid = stack.rectilinear_grid(0.05, x_um=ex, y_um=ey, z_um=ez)
    config = stack.get_simulation_config(
        time=80e-15, grid=grid, gradient_config=fdtdx.GradientConfig(method="checkpointed", num_checkpoints=40),
    )
    wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=wave, spectral_width=fdtdx.WaveCharacter(frequency=0.2 * wave.get_frequency())
    )
    volume, constraints = stack.get_simulation_volume()
    objects = [volume, stack]
    for obj, placing in (
        stack.create_point_dipole_source_at("o1", wave_character=wave, polarization=1, temporal_profile=pulse,
                                            port_offset=0.5, name="dipole"),
        stack.create_phasor_detector_at("o2", (0.2, 0.2), wave_characters=(wave,), components=("Ey",),
                                        port_offset=0.45, name="out", grid=grid),
    ):
        objects.append(obj)
        constraints += placing
    boundaries, placing = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=7),
                                                             volume)
    objects += boundaries.values()
    constraints += placing
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
    params = {**params, **bf.init_stack_params(objects)}
    (vname,) = [k for k in params["stack"] if k.startswith("vertices/")]
    v = np.asarray(params["stack"][vname])
    index = int(np.argmin(np.hypot(v[:, 0] - bx1, v[:, 1] - by1)))
    return objects, arrays, params, config, vname, index


def check(kind: str, averaging: str, zmin: float = 0.0, params: tuple = tuple(PARAMS),
          targets: tuple = ("weights", "fdtd")) -> list[dict]:
    """The rows of one grid and averaging: every parameter, through the weights and through the run."""
    import fdtdx
    import jax
    import jax.numpy as jnp

    import bifrost as bf

    names = params
    objects, arrays, params, config, vname, index = build(kind, averaging, zmin)
    placed = objects["stack"]
    nx, ny, nz = placed.grid_shape
    key = jax.random.PRNGKey(0)
    field = jax.random.uniform(jax.random.PRNGKey(7), (nx, ny, nz), dtype=jnp.float64)

    def perturbed(name: str, delta: jax.Array) -> dict:
        stack_p = dict(params["stack"])
        if name == "vertex":
            stack_p[vname] = stack_p[vname].at[index, 0].add(delta)
        else:
            stack_p[name] = stack_p[name] + delta
        return {**params, "stack": stack_p}

    def weights_objective(delta: jax.Array, name: str) -> jax.Array:
        names, w = placed.material_weights(perturbed(name, delta)["stack"])
        return jnp.sum(w[names.index("si")] * field)

    def fdtd_objective(delta: jax.Array, name: str) -> jax.Array:
        arrays2, objects2, _ = bf.apply_stack(arrays, objects, perturbed(name, delta), key)
        _, arrays2 = fdtdx.run_fdtd(arrays=arrays2, objects=objects2, config=config, key=key, show_progress=False)
        return jnp.sum(jnp.abs(arrays2.detector_states["out"]["phasor"]) ** 2)

    rows = []
    for target, objective in (("weights", weights_objective), ("fdtd", fdtd_objective)):
        if target not in targets:
            continue
        value = bf.jit(objective, static_argnames="name")
        grad = bf.jit(jax.grad(objective), static_argnames="name")
        for name in names:
            steps = PARAMS[name]
            t0 = time.time()
            zero = jnp.zeros((), dtype=jnp.float64)
            g = float(grad(zero, name=name))
            fds = [float((value(zero + h, name=name) - value(zero - h, name=name)) / (2 * h)) for h in steps]
            rows.append({
                "kind": kind, "averaging": averaging, "zmin": zmin, "target": target, "param": name, "grad": g,
                "fd": fds,
                "steps": list(steps), "rel_err": abs(g - fds[-1]) / max(abs(fds[-1]), 1e-300),
                "fd_spread": abs(fds[0] - fds[1]) / max(abs(fds[-1]), 1e-300),
                "seconds": round(time.time() - t0, 1), "cells": [nx, ny, nz],
            })
            print(json.dumps(rows[-1]), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out")
    parser.add_argument("--kinds", nargs="+", default=["graded", "uniform"])
    parser.add_argument("--averaging", nargs="+", default=["subpixel", "subpixel_full"])
    parser.add_argument("--zmin", type=float, default=0.0, help="the core's bottom (um) in the fixed box")
    parser.add_argument("--params", nargs="+", default=list(PARAMS), choices=list(PARAMS))
    parser.add_argument("--targets", nargs="+", default=["weights", "fdtd"], choices=["weights", "fdtd"])
    args = parser.parse_args()
    import jax

    print("jax", jax.__version__, "on", jax.devices()[0].device_kind, "x64", jax.config.jax_enable_x64, flush=True)
    rows = []
    for kind in args.kinds:
        for averaging in args.averaging:
            started = time.time()
            rows += check(kind, averaging, args.zmin, tuple(args.params), tuple(args.targets))
            print(f"{kind} {averaging}: {time.time() - started:.0f} s", flush=True)
    Path(args.out).write_text(json.dumps({"rows": rows}, indent=1))


if __name__ == "__main__":
    main()
