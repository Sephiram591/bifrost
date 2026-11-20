"""Adiabatic elevator coupler between two SiN tapers, simulated with fdtdx.

The gdsfactory component and its LayerStack go straight into the simulation as
one StackDevice; its ports get straight extensions through the PML, and the
source and mode monitors are placed on the ports. TE0 at 619 nm is launched into the upper taper (190 nm SiN on
the oxide) and its transfer into the TE0 mode of the lower taper (110 nm SiN,
top 100 nm below the oxide surface) is measured while sweeping the overlap of
the two 10 um tapers (650 nm -> 50 nm wide).

Run on a GPU node:  python examples/fdtdx_elevator_coupler.py
"""

import sys
import time as clock
from pathlib import Path

import fdtdx
import jax
import numpy as np

import gdsfactory as gf
import bifrost as bf
from gdsfactory.technology import LayerLevel, LayerStack

gf.gpdk.PDK.activate()

WAVELENGTH = 0.619  # um
N_SIN, N_OXIDE = 2.0, 1.457
UPPER, LOWER = (34, 0), (35, 0)
LENGTH, W_MAX, W_MIN = 10.0, 0.65, 0.05  # taper length and widths (um)
PML = 0.5  # um
PPW = 20  # points per wavelength in each material
LEAD = 0.5  # straight sections carrying the ports (um)
EXTENSION = 1.0  # port extensions (um): to the box faces, through the PML

LAYER_STACK = LayerStack(
    layers={
        "oxide": LayerLevel(layer="WAFER", zmin=-5.0, thickness=5.0, material="oxide", mesh_order=9),
        "lower": LayerLevel(layer=LOWER, zmin=-0.21, thickness=0.11, material="sin", mesh_order=1),
        "upper": LayerLevel(layer=UPPER, zmin=0.0, thickness=0.19, material="sin", mesh_order=1),
    }
)
MATERIALS = {
    "air": fdtdx.Material(),
    "oxide": fdtdx.Material(permittivity=N_OXIDE**2),
    "sin": fdtdx.Material(permittivity=N_SIN**2),
}


def elevator_coupler(overlap: float) -> gf.Component:
    """Upper taper (wide -> narrow) whose tip overlaps the lower taper (narrow -> wide).

    Short straight leads on both ends carry the ports: o1 on the upper guide, o2
    on the lower one.
    """
    c = gf.Component()
    a, b, x = W_MAX / 2, W_MIN / 2, LENGTH - overlap  # x: where the lower taper starts
    c.add_polygon([(-LEAD, -a), (0, -a), (LENGTH, -b), (LENGTH, b), (0, a), (-LEAD, a)], layer=UPPER)
    end = x + LENGTH
    c.add_polygon([(x, -b), (end, -a), (end + LEAD, -a), (end + LEAD, a), (end, a), (x, b)], layer=LOWER)
    c.add_port("o1", center=(-LEAD, 0.0), width=W_MAX, orientation=180, layer=UPPER)
    c.add_port("o2", center=(end + LEAD, 0.0), width=W_MAX, orientation=0, layer=LOWER)
    return c


def domain(overlap: float) -> tuple[tuple[float, float], ...]:
    """Simulation box (um): the port extensions run through the PML to its x faces."""
    reach = LEAD + EXTENSION
    return (-reach, 2 * LENGTH - overlap + reach), (-2.0, 2.0), (-1.6, 1.4)


def build_stack(overlap: float) -> bf.stack.StackDevice:
    """The component with its stack; the box is the polygons' bbox (extensions included),
    grown in y and z by ``boundary_offset`` to the domain."""
    (_, _), (y0, y1), (z0, z1) = domain(overlap)
    return bf.stack.StackDevice.from_layer_stack(
        elevator_coupler(overlap), LAYER_STACK, MATERIALS, background="air", name="stack",
        zrange_layers=["lower", "upper"],    # z: the two SiN levels, -0.21 to 0.19 um
        extend_ports=(("o1", EXTENSION), ("o2", EXTENSION)),
        boundary_offset=(0.0, 0.0, y1 - W_MAX / 2, y0 + W_MAX / 2, z1 - 0.19, z0 + 0.21),
    )


def simulate(overlap: float, ppw: float = PPW, time: float = 400e-15) -> dict:
    wave = fdtdx.WaveCharacter(wavelength=WAVELENGTH * 1e-6)
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=wave, spectral_width=fdtdx.WaveCharacter(frequency=0.05 * wave.get_frequency())
    )
    te0 = dict(mode_index=0, filter_pol="te")  # TE0: E_y dominant
    mode = dict(wave_characters=(wave,), scaling_mode="pulse", **te0)

    stack = build_stack(overlap)
    mesh = bf.mesh(stack, WAVELENGTH, ppw=ppw, pml=PML)  # 0.5 um PML fits inside the 1 um port extensions
    config = stack.get_simulation_config(time=time, grid=mesh.grid)
    volume, constraints = stack.get_simulation_volume()
    objects = [volume, stack]
    # port planes 2.9 um wide and 1.5 (upper) / 2.0 (lower) um tall around each guide, outside the PML:
    # port_padding is per side, in wavelengths (0.619 um), past the guide's width and its level's height
    for obj, c in (
        stack.create_mode_plane_source_at(
            "o1", (1.838, 1.074), wave_character=wave, temporal_profile=pulse, port_offset=-0.3,
            name="source", grid=config, **te0,
        ),
        stack.create_mode_overlap_detector_at(
            "o1", (1.838, 1.074), direction="in", name="in", grid=config, **mode,
        ),
        stack.create_mode_overlap_detector_at("o2", (1.838, 1.511), name="out", grid=config, **mode),
    ):
        objects.append(obj)
        constraints += c
    field = fdtdx.PhasorDetector(  # |E| on the xz plane through the taper axis
        name="field", wave_characters=(wave,), components=("Ex", "Ey", "Ez"),
        scaling_mode="pulse", partial_grid_shape=(None, 1, None),
    )
    objects.append(field)
    constraints += [field.same_size(volume, axes=(0, 2)), field.place_at_center(volume)]
    boundaries, c = fdtdx.boundary_objects_from_config(mesh.boundary, volume)
    objects += boundaries.values()
    constraints += c

    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
    arrays, objects, _ = bf.stack.apply_stack(arrays, objects, {**params, **bf.stack.init_stack_params(objects)}, key)
    start = clock.perf_counter()
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key)
    jax.block_until_ready(arrays.detector_states)
    runtime = clock.perf_counter() - start

    states = arrays.detector_states
    a_in = objects["in"].compute_overlap(states["in"])
    a_out = objects["out"].compute_overlap(states["out"])
    j = objects["field"].grid_slice[1].start  # y index of the field slice (y = 0)
    E = np.asarray(states["field"]["phasor"][0, 0, :, :, 0, :])  # (3, nx, nz) at the wavelength
    x_edges = np.asarray(mesh.edges[0], dtype=float)
    z_edges = np.asarray(mesh.edges[2], dtype=float)
    return {
        "T": float(np.ravel(abs(a_out / a_in) ** 2)[0]),  # one overlap per wave character
        "x": 0.5 * (x_edges[:-1] + x_edges[1:]),
        "z": 0.5 * (z_edges[:-1] + z_edges[1:]),
        "z_edges": z_edges,
        "E2": np.sum(np.abs(E) ** 2, axis=0),  # |E|^2 on the xz plane through the taper axis
        "eps": 1 / np.asarray(arrays.inv_permittivities[0, :, j, :]),
        "cells": int(np.prod(mesh.shape)),
        "runtime": runtime,
        "ppw": float(ppw),
    }


if __name__ == "__main__":
    ppw = float(sys.argv[1]) if len(sys.argv) > 1 else PPW
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    for overlap in np.linspace(1, 6, 4):
        result = simulate(overlap, ppw=ppw)
        print(
            f"overlap {overlap:.2f} um (ppw={ppw:g}, {result['cells'] / 1e6:.1f}M cells, "
            f"{result['runtime']:.0f} s): T = {result['T']:.4f}"
        )
        np.savez(out / f"fdtdx_overlap_{overlap:.2f}.npz", **result)
