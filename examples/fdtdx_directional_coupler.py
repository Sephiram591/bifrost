"""Broadband silicon-nitride directional coupler, simulated with fdtdx.

Two 400 nm wide, 200 nm tall SiN (n = 1.95) waveguides in oxide (n = 1.457). They start 3 um apart, are brought
together by two cosine S-bends to a 300 nm gap over a straight coupling section, and are separated again. The
gdsfactory component and its LayerStack go straight into the simulation as one StackDevice; the ports get straight
extensions through the PML, and the source and mode monitors sit on the ports.

A short Gaussian pulse launches TE0 into o1 (lower left). One run covers the whole band: mode-overlap monitors on the
input and on the two outputs, o4 (through, same guide) and o3 (cross, other guide), record the TE0 amplitudes at
31 frequencies centered on 619 nm, and ``T = |a_out / a_in|^2`` gives the power spectra.

Run on a GPU node:  python examples/fdtdx_directional_coupler.py [resolution_um] [averaging]
                    averaging: subpixel (default), subpixel_full or inverse, see StackOptions
Compare with MEEP:  examples/meep_directional_coupler.py, examples/compare_directional_coupler.py
"""

import sys
import time as clock
from pathlib import Path

import fdtdx
import jax
import jax.numpy as jnp
import numpy as np

import gdsfactory as gf
import bifrost as bf
from gdsfactory.technology import LayerLevel, LayerStack

gf.gpdk.PDK.activate()

WAVELENGTH = 0.619  # um, the center of the band
N_SIN, N_OXIDE = 1.95, 1.457
CORE = gf.gpdk.LAYER.WG
WIDTH, HEIGHT = 0.4, 0.2  # waveguide cross-section (um)
GAP = 0.3  # edge-to-edge gap in the coupling section (um)
PORT_GAP = 3.0  # edge-to-edge gap at the ports (um)
COUPLING = 20.0  # straight coupling section (um)
BEND = 10.0  # x length of each cosine S-bend (um)
LEAD = 0.5  # straight sections between a port and its bend (um)
EXTENSION = 1.0  # port extensions (um): to the box faces, through the PML
PML = 0.5  # um
HALF_SPAN = (3.2, 1.3)  # half the box height in y and z (um)
PLANE = (2.0, 1.4)  # y and z size of the source and monitor planes around a guide (um)

BANDWIDTH = 0.15  # the band is the center frequency +- this fraction of it
N_FREQ = 31  # frequencies in the band
PULSE_WIDTH = 0.10  # Gaussian spectral width (sigma of the amplitude spectrum) / center frequency
RESOLUTION = 0.02  # um
TIME = 400e-15  # s: the pulse leaves the last monitor at ~350 fs
AVERAGING = "subpixel"  # how the stack averages permittivity at interfaces: "subpixel", "subpixel_full", "inverse"

LENGTH = 2 * (LEAD + BEND) + COUPLING  # x distance between the left and right ports

LAYER_STACK = LayerStack(
    layers={"core": LayerLevel(layer=CORE, zmin=-HEIGHT / 2, thickness=HEIGHT, material="core", mesh_order=1)}
)
MATERIALS = {
    "core": fdtdx.Material(permittivity=N_SIN**2),
    "clad": fdtdx.Material(permittivity=N_OXIDE**2),
}


def frequencies() -> np.ndarray:
    """The N_FREQ frequencies (1 / um), evenly spaced and centered on 1 / WAVELENGTH."""
    return np.linspace(1 - BANDWIDTH, 1 + BANDWIDTH, N_FREQ) / WAVELENGTH


def half_separation(x: np.ndarray, gap: float = GAP) -> tuple[np.ndarray, np.ndarray]:
    """Half the center-to-center distance of the guides at x, and its slope d/dx (cosine S-bends)."""
    c, p = (gap + WIDTH) / 2, (PORT_GAP + WIDTH) / 2
    u = np.clip((x - LEAD) / BEND, 0, 1)  # 0 -> 1 over the first bend
    v = np.clip((x - LEAD - BEND - COUPLING) / BEND, 0, 1)  # over the second
    h = c + (p - c) * ((1 + np.cos(np.pi * u)) + (1 - np.cos(np.pi * v))) / 2
    dh = (p - c) / 2 * np.pi / BEND * (np.sin(np.pi * v) - np.sin(np.pi * u))
    return h, dh


def directional_coupler(gap: float = GAP) -> gf.Component:
    """The two guides as polygons on the core layer.

    Ports: o1 lower left, o2 upper left, o3 upper right, o4 lower right.
    """
    bend = np.linspace(0, BEND, round(BEND / 0.1) + 1)  # vertices every 0.1 um along the bends
    second = LEAD + BEND + COUPLING
    x = np.unique(np.concatenate([[0, LEAD], LEAD + bend, [second], second + bend, [LENGTH]]))
    h, dh = half_separation(x, gap)
    c = gf.Component()
    for sign in (-1, 1):
        n = np.stack([-sign * dh, np.ones_like(x)], axis=1) / np.sqrt(1 + dh**2)[:, None]  # unit normal of the guide
        center = np.stack([x, sign * h], axis=1)
        edge_a, edge_b = center + WIDTH / 2 * n, center - WIDTH / 2 * n
        c.add_polygon(np.concatenate([edge_a, edge_b[::-1]]).tolist(), layer=CORE)
    h0 = float(half_separation(np.array(0.0), gap)[0])
    c.add_port("o1", center=(0.0, -h0), width=WIDTH, orientation=180, layer=CORE)
    c.add_port("o2", center=(0.0, h0), width=WIDTH, orientation=180, layer=CORE)
    c.add_port("o3", center=(LENGTH, h0), width=WIDTH, orientation=0, layer=CORE)
    c.add_port("o4", center=(LENGTH, -h0), width=WIDTH, orientation=0, layer=CORE)
    return c


def domain() -> tuple[tuple[float, float], ...]:
    """Simulation box (um): the port extensions run through the PML to its x faces."""
    return (-EXTENSION, LENGTH + EXTENSION), (-HALF_SPAN[0], HALF_SPAN[0]), (-HALF_SPAN[1], HALF_SPAN[1])


def build_stack(gap: float = GAP, averaging: str = AVERAGING) -> bf.stack.StackDevice:
    """The component with its stack; the box is the polygons' bbox (extensions included),
    grown in y and z by ``boundary_offset`` to the domain."""
    y_edge = (PORT_GAP + 2 * WIDTH) / 2
    dy, dz = HALF_SPAN[0] - y_edge, HALF_SPAN[1] - HEIGHT / 2
    return bf.stack.StackDevice.from_layer_stack(
        directional_coupler(gap), LAYER_STACK, MATERIALS, background="clad", name="stack",
        zrange_layers=["core"],  # z: the core level, -0.1 to 0.1 um
        extend_ports=tuple((port, EXTENSION) for port in ("o1", "o2", "o3", "o4")),
        boundary_offset=(0.0, 0.0, dy, -dy, dz, -dz),
        averaging=averaging,
    )


def padding(wave_characters: list[fdtdx.WaveCharacter]) -> tuple[float, float]:
    """``port_padding`` (in mean wavelengths) that makes a plane of PLANE um around a guide."""
    mean = np.mean([w.get_wavelength() for w in wave_characters]) * 1e6
    return (PLANE[0] - WIDTH) / 2 / mean, (PLANE[1] - HEIGHT) / 2 / mean


def simulate(resolution: float = RESOLUTION, time: float = TIME, averaging: str = AVERAGING) -> dict:
    (x0, _), (y0, _), _ = domain()
    config = fdtdx.SimulationConfig(
        time=time, grid=fdtdx.core.grid.UniformGrid(spacing=resolution * 1e-6), dtype=jnp.float32
    )
    center = fdtdx.WaveCharacter(wavelength=WAVELENGTH * 1e-6)
    band = [fdtdx.WaveCharacter(wavelength=1e-6 / f) for f in frequencies()]
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=center, spectral_width=fdtdx.WaveCharacter(frequency=PULSE_WIDTH * center.get_frequency())
    )
    te0 = dict(mode_index=0, filter_pol="te")  # TE0: E_y dominant
    mode = dict(wave_characters=band, scaling_mode="pulse", **te0)

    stack = build_stack(averaging=averaging)
    volume, constraints = stack.get_simulation_volume()
    objects = [volume, stack]
    for obj, c in (
        stack.create_mode_plane_source_at(
            "o1", padding([center]), wave_character=center, temporal_profile=pulse, port_offset=-0.3, name="source",
            **te0,
        ),
        stack.create_mode_overlap_detector_at("o1", padding(band), direction="in", name="in", **mode),
        stack.create_mode_overlap_detector_at("o4", padding(band), name="through", **mode),
        stack.create_mode_overlap_detector_at("o3", padding(band), name="cross", **mode),
    ):
        objects.append(obj)
        constraints += c
    field = fdtdx.PhasorDetector(  # |E| at the center wavelength on the xy plane through the core's middle
        name="field", wave_characters=(center,), components=("Ex", "Ey", "Ez"),
        scaling_mode="pulse", partial_grid_shape=(None, None, 1),
    )
    objects.append(field)
    constraints += [field.same_size(volume, axes=(0, 1)), field.place_at_center(volume)]
    pml = fdtdx.BoundaryConfig.from_uniform_bound(thickness=round(PML / resolution))
    boundaries, c = fdtdx.boundary_objects_from_config(pml, volume)
    objects += boundaries.values()
    constraints += c

    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
    arrays, objects, _ = bf.stack.apply_stack(arrays, objects, {**params, **bf.stack.init_stack_params(objects)}, key)
    # gdsfactory turns on x64; fdtdx 0.6.3's PML would then promote its float32 state inside the solve loop
    start = clock.perf_counter()
    with jax.enable_x64(False):
        _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key)
        jax.block_until_ready(arrays.detector_states)
    runtime = clock.perf_counter() - start
    peak = (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30  # GiB, the whole process

    states = arrays.detector_states
    a_in, a_through, a_cross = (objects[n].compute_overlap(states[n]) for n in ("in", "through", "cross"))
    k = objects["field"].grid_slice[2].start  # z index of the field slice
    E = np.asarray(states["field"]["phasor"][0, 0, :, :, :, 0])  # (3, nx, ny) at the center wavelength
    nx, ny = E.shape[1:]
    return {
        "frequency": frequencies(),  # 1 / um
        "wavelength": 1 / frequencies(),  # um
        "through": np.abs(np.asarray(a_through / a_in)) ** 2,  # one overlap per frequency
        "cross": np.abs(np.asarray(a_cross / a_in)) ** 2,
        "x": x0 + resolution * (np.arange(nx) + 0.5),
        "y": y0 + resolution * (np.arange(ny) + 0.5),
        "E2": np.sum(np.abs(E) ** 2, axis=0),  # |E|^2 on the xy plane
        "eps": 1 / np.asarray(arrays.inv_permittivities[0, :, :, k]),
        "runtime": runtime,  # s, the FDTD run including its JIT compilation
        "cells": int(np.prod(arrays.inv_permittivities.shape[1:])),
        "peak_gb": peak,
        "averaging": averaging,
    }


if __name__ == "__main__":
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    resolution = float(sys.argv[1]) if len(sys.argv) > 1 else RESOLUTION
    averaging = sys.argv[2] if len(sys.argv) > 2 else AVERAGING
    result = simulate(resolution, averaging=averaging)
    mid = N_FREQ // 2
    print(
        f"fdtdx ({averaging}), resolution {resolution * 1e3:g} nm, {result['cells'] / 1e6:.1f}M cells, "
        f"{result['runtime']:.0f} s, peak {result['peak_gb']:.1f} GiB: "
        f"{WAVELENGTH * 1e3:.0f} nm through {result['through'][mid]:.4f}, cross {result['cross'][mid]:.4f}"
    )
    suffix = "" if averaging == AVERAGING else f"_{averaging}"  # the default's files keep their plain names
    np.savez(out / f"fdtdx_dc_{resolution * 1e3:g}nm{suffix}.npz", **result)
