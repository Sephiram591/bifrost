"""A 90 degree silicon waveguide bend, simulated end to end with bifrost and fdtdx.

The device is gdsfactory's ``bend_circular`` pcell. TE0 at 1.55 um is launched into
the straight extension of port ``o1``, and the fraction that leaves port ``o2``
in TE0 is measured. The same script produces the field image
and the result the viewer loads (see docs/getting-started.md).

Run it on a GPU node:  python examples/waveguide_bend.py
Options:               python examples/waveguide_bend.py --resolution 0.04 --figure out.png
"""

# --8<-- [start:imports]
import argparse
import time
from dataclasses import dataclass
from typing import Any

import fdtdx
import jax
import jax.numpy as jnp
import numpy as np

import gdsfactory as gf
import bifrost as bf
from gdsfactory.technology import LayerLevel, LayerStack

gf.gpdk.PDK.activate()  # the generic PDK: the pcells below and the waveguide layer come from it
# --8<-- [end:imports]

# --8<-- [start:parameters]
WAVELENGTH = 1.55  # um, in vacuum
CORE_INDEX, CLAD_INDEX = 3.48, 1.444  # silicon in silica, constant (no dispersion)
WIDTH, THICKNESS = 0.5, 0.22  # um
RADIUS = 5.0  # um, of the bend's centerline
EXTENSION = 2.5  # um, how far both ports are extended, straight, to the box faces
PML = 1.0  # um of perfectly matched layer inside each face of the box
MARGIN = 2.25  # um of cladding between the bend and the two box faces that have no port
BOX_HEIGHT = 3.4  # um, the box in z; with MARGIN this makes the box exactly 10 x 10 x 3.4 um
PLANE_PADDING = (0.5, 0.3)  # source and monitor planes: wavelengths past the guide, per side (width, height)
RESOLUTION = 0.04  # um, the grid cell
DURATION = 400e-15  # s of simulated time
MOVIE_FRAMES = 40  # time steps the field movie records, evenly spaced
# --8<-- [end:parameters]


# --8<-- [start:component]
component = gf.components.bend_circular(radius=RADIUS, angle=90, width=WIDTH)  # ports: o1 faces west, o2 north
# --8<-- [end:component]


# --8<-- [start:stack]
LAYER_STACK = LayerStack(
    layers={"core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=THICKNESS, material="core", mesh_order=1)}
)
MATERIALS = {
    "core": fdtdx.Material(permittivity=CORE_INDEX**2),
    "clad": fdtdx.Material(permittivity=CLAD_INDEX**2),  # also the background
}


def build_stack() -> bf.StackDevice:
    """The bend voxelized with its layer stack, with both ports extended straight to the box."""
    z_margin = (BOX_HEIGHT - THICKNESS) / 2  # cladding above and below the core
    return bf.StackDevice.from_layer_stack(
        component, LAYER_STACK, MATERIALS,
        background="clad",
        included_layers=["core"],  # the box's z range starts as the core's: 0 to 0.22 um
        extend_ports=(("o1", EXTENSION), ("o2", EXTENSION)),  # both end at a box face
        # (x+, x-, y+, y-, z+, z-): the faces holding a port stay put, the others move out. The box must
        # be a whole number of grid cells in each direction: 10, 10 and 3.4 um suit 50, 40, 25 and 20 nm.
        boundary_offset=(MARGIN, 0.0, 0.0, -MARGIN, z_margin, -z_margin),
        name="stack",
    )
# --8<-- [end:stack]


@dataclass
class BendResult:
    """What a run hands back: the placed simulation (for the viewer) and the numbers."""

    stack: Any
    objects: Any
    arrays: Any
    config: Any
    transmission: float
    field: np.ndarray  # |E|^2 on the xy plane through the middle of the core, shape (nx, ny)
    extent: tuple[float, float, float, float]  # x0, x1, y0, y1 of that plane, um
    eps: np.ndarray  # the permittivity on that plane, shape (nx, ny)
    seconds: float  # wall time of place + apply + run, compilation included
    peak_gib: float | None  # peak device memory, when the backend reports it


def simulate(resolution: float = RESOLUTION, duration: float = DURATION) -> BendResult:
    started = time.perf_counter()
    stack = build_stack()

    # --8<-- [start:settings]
    config = fdtdx.SimulationConfig(
        time=duration,
        grid=fdtdx.core.grid.UniformGrid(spacing=resolution * 1e-6),
        dtype=jnp.float32,
    )
    wave = fdtdx.WaveCharacter(wavelength=WAVELENGTH * 1e-6)
    pulse = fdtdx.GaussianPulseProfile(  # a pulse 5 % wide in frequency, so it ramps up smoothly
        center_wave=wave, spectral_width=fdtdx.WaveCharacter(frequency=0.05 * wave.get_frequency())
    )
    te0 = dict(mode_index=0, filter_pol="te")  # the fundamental TE mode
    # --8<-- [end:settings]

    # --8<-- [start:objects]
    volume, constraints = stack.get_simulation_volume()
    objects = [volume, stack]
    for obj, placing in (
        # launch 1 um outside port o1 (a negative port_offset moves a plane out of the device,
        # into the port extension), clear of the absorbing boundary at the box face
        stack.create_mode_plane_source_at(
            "o1", PLANE_PADDING, wave_character=wave, temporal_profile=pulse, port_offset=-1.0,
            name="source", **te0,
        ),
        # one monitor 0.5 um outside each port, in the straight extensions, before and after the bend
        stack.create_mode_overlap_detector_at(
            "o1", PLANE_PADDING, direction="in", wave_characters=(wave,), scaling_mode="pulse",
            port_offset=-0.5, name="in", **te0,
        ),
        stack.create_mode_overlap_detector_at(
            "o2", PLANE_PADDING, wave_characters=(wave,), scaling_mode="pulse",
            port_offset=-0.5, name="out", **te0,
        ),
    ):
        objects.append(obj)
        constraints += placing
    field = fdtdx.PhasorDetector(  # E on the xy plane through the middle of the core
        name="field", wave_characters=(wave,), components=("Ex", "Ey", "Ez"),
        scaling_mode="pulse", partial_grid_shape=(None, None, 1),
    )
    movie = fdtdx.FieldDetector(  # the same plane at 40 evenly spaced time steps: a movie of the pulse
        name="movie", components=("Ex", "Ey"), partial_grid_shape=(None, None, 1),
        switch=fdtdx.OnOffSwitch(interval=max(1, config.time_steps_total // MOVIE_FRAMES)),
    )
    for monitor in (field, movie):
        objects.append(monitor)
        constraints += [monitor.same_size(volume, axes=(0, 1)), monitor.place_at_center(volume)]
    boundaries, placing = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=round(PML / resolution)), volume
    )
    objects += boundaries.values()
    constraints += placing
    # --8<-- [end:objects]

    # --8<-- [start:run]
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
    # write the layer stack into the permittivity arrays (instead of fdtdx.apply_params)
    arrays, objects, _ = bf.apply_stack(arrays, objects, {**params, **bf.init_stack_params(objects)}, key)
    # gdsfactory turns on 64-bit floats; fdtdx's float32 solver loop needs them off
    with jax.enable_x64(False):
        _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key)
    # --8<-- [end:run]

    # --8<-- [start:results]
    states = arrays.detector_states
    a_in = objects["in"].compute_overlap(states["in"])  # complex amplitude of TE0 on each monitor
    a_out = objects["out"].compute_overlap(states["out"])
    transmission = float(np.ravel(abs(a_out / a_in) ** 2)[0])  # one entry per wavelength
    # --8<-- [end:results]

    k = objects["field"].grid_slice[2].start  # z index of the field plane
    phasor = np.asarray(states["field"]["phasor"][0, 0, :, :, :, 0])  # (3, nx, ny) at the wavelength
    nx, ny = phasor.shape[1:]
    sx, sy, _ = (float(v) * 1e6 for v in stack.partial_real_shape)
    x0, y0 = float(stack.center[0]) - sx / 2, float(stack.center[1]) - sy / 2
    stats = getattr(jax.devices()[0], "memory_stats", lambda: None)()
    return BendResult(
        stack=stack, objects=objects, arrays=arrays, config=config, transmission=transmission,
        field=np.sum(np.abs(phasor) ** 2, axis=0),
        extent=(x0, x0 + nx * resolution, y0, y0 + ny * resolution),
        eps=1 / np.asarray(arrays.inv_permittivities[0, :, :, k]),
        seconds=time.perf_counter() - started,
        peak_gib=None if not stats else stats.get("peak_bytes_in_use", 0) / 2**30,
    )


# --8<-- [start:viewer]
def open_viewer(result: BendResult) -> Any:
    """The viewer with the finished simulation loaded into its Result tab.

    In a Jupyter or marimo cell, end the cell with the returned viewer to display it.
    """
    viewer = bf.view(result.stack)
    viewer.load_result(result.objects, result.arrays, result.config, stack=result.stack)
    return viewer
# --8<-- [end:viewer]


# --8<-- [start:plot]
def plot_field(result: BendResult, path: str) -> None:
    """Saves |E|^2 on the xy plane through the core, with the guide's outline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0, x1, y0, y1 = result.extent
    fig, ax = plt.subplots(figsize=(6, 5.4))
    image = ax.imshow(
        result.field.T / result.field.max(), origin="lower", extent=(x0, x1, y0, y1), cmap="inferno",
    )
    xs, ys = np.linspace(x0, x1, result.eps.shape[0]), np.linspace(y0, y1, result.eps.shape[1])
    ax.contour(xs, ys, result.eps.T, levels=[(CORE_INDEX**2 + CLAD_INDEX**2) / 2], colors="white", linewidths=0.6)
    ax.set(xlabel="x (um)", ylabel="y (um)", title=f"|E|$^2$ at {WAVELENGTH} um, T = {result.transmission:.3f}")
    fig.colorbar(image, label="|E|$^2$ (relative to the peak)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
# --8<-- [end:plot]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resolution", type=float, default=RESOLUTION, help="grid cell in um")
    parser.add_argument("--duration", type=float, default=DURATION, help="simulated time in s")
    parser.add_argument("--figure", default="waveguide_bend_field.png", help="where to save the field image")
    args = parser.parse_args()

    result = simulate(args.resolution, args.duration)
    print(f"transmission: {result.transmission:.4f}")
    print(f"wall time:    {result.seconds:.1f} s (compilation included)")
    if result.peak_gib is not None:
        print(f"peak memory:  {result.peak_gib:.1f} GiB on {jax.devices()[0]}")
    plot_field(result, args.figure)
    print(f"field image:  {args.figure}")
