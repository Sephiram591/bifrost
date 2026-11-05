"""Inverse design: the widths of a bend too tight for a uniform-width guide.

An air-clad silicon nitride guide on silica, 1.6 um wide and 0.75 um thick, bent 90 degrees at 7 um radius,
transmits only 56 % of its TE0 mode as TE0 at the uniform width: most of the rest leaves as TE1. The width on
each side of its centerline, at 70 stations along the bend, is optimized for TE0 transmission at 1.55 um by
gradient ascent through the simulation; both ends stay 1.6 um wide and centered, so the ports and their
straight leads never change. The optimized design is then checked on a finer grid.
Run it on a GPU:  python examples/guide/inverse_design.py
Options:  --iterations 50 --resolution 0.06 --check 0.04 --figure inverse_design.png
"""

import argparse
import time

import fdtdx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import gdsfactory as gf
import bifrost as bf
from gdsfactory.technology import LayerLevel, LayerStack

gf.gpdk.PDK.activate()

# --8<-- [start:geometry]
WAVELENGTH = 1.55  # um
CORE_INDEX, BOX_INDEX, CLAD_INDEX = 1.996, 1.444, 1.0  # silicon nitride on silica, under air
WIDTH, THICKNESS = 1.6, 0.75  # um: the guide at both ports
RADIUS = 7.0  # um, of the bend's centerline
N_STATIONS = 70  # evenly spaced along the bend between its two ends
H_MIN, H_MAX = 0.25, 1.25  # um: bounds of every half-width (H_MAX < RADIUS keeps the inner edge off the center)
ANGLES = jnp.linspace(0.0, jnp.pi / 2, N_STATIONS + 2)  # one vertex per station and end on each edge: a fixed topology


def tight_bend(half_widths):
    """The bend whose inner and outer edges are ``half_widths`` (2, N_STATIONS) um from its centerline.

    Row 0 is the side toward the bend's center, row 1 the side away from it. At both ends the guide is
    WIDTH wide and centered; in between, each edge is linear from station to station. Every profile has
    the same 144 vertices, so the device built for one takes the vertices of any other (``StackDevice.pack``).
    """
    t = ANGLES / ANGLES[-1]  # along the bend, 0 to 1
    stations = jnp.linspace(0.0, 1.0, N_STATIONS + 2)
    end = jnp.array([WIDTH / 2])
    h_in, h_out = (jnp.interp(t, stations, jnp.concatenate([end, h, end])) for h in jnp.asarray(half_widths))
    edges = []
    for r in (RADIUS + h_out, RADIUS - h_in):  # the arc's center is at (0, RADIUS)
        edges.append(jnp.stack([r * jnp.sin(ANGLES), RADIUS - r * jnp.cos(ANGLES)], axis=1))
    bend = gf.Component()
    bend.add_polygon(jnp.concatenate([edges[0], edges[1][::-1]]), layer=gf.gpdk.LAYER.WG)
    bend.add_port(name="o1", center=(0.0, 0.0), width=WIDTH, orientation=180, layer=gf.gpdk.LAYER.WG)
    bend.add_port(name="o2", center=(RADIUS, RADIUS), width=WIDTH, orientation=90, layer=gf.gpdk.LAYER.WG)
    return bend


UNIFORM = jnp.full((2, N_STATIONS), WIDTH / 2)  # the 1.6 um guide, centered everywhere
# --8<-- [end:geometry]


# --8<-- [start:simulation]
EXTENSION = 2.5  # um: both ports run straight out to the box faces
MARGIN = 2.06  # um past the bend's two free sides; the box is 12.36 x 12.36 x 4.8 um, whole cells at 60 and 40 nm
BOX_HEIGHT, PML = 4.8, 1.0  # um
RESOLUTION, CHECK_RESOLUTION = 0.06, 0.04  # um: the grid it is optimized on, and the finer one it is checked on
DURATION = 500e-15  # s
PADDING = (0.5, 0.5)  # wavelengths past the port's width and the core's height: the mode reaches into the silica
KEY = jax.random.PRNGKey(0)

layer_stack = LayerStack(
    layers={
        "box": LayerLevel(layer="WAFER", zmin=-3.0, thickness=3.0, material="silica", mesh_order=9),
        "core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=THICKNESS, material="core", mesh_order=1),
    }
)
materials = {
    "core": fdtdx.Material(permittivity=CORE_INDEX**2),
    "silica": fdtdx.Material(permittivity=BOX_INDEX**2),
    "air": fdtdx.Material(permittivity=CLAD_INDEX**2),
}


def place(resolution):
    """The uniform bend, placed on a grid of this cell (um) with its source and monitors."""
    z_margin = (BOX_HEIGHT - THICKNESS) / 2  # the silica runs through the bottom face, the air through the top
    stack = bf.StackDevice.from_layer_stack(
        tight_bend(UNIFORM), layer_stack, materials, background="air", zrange_layers=["core"],
        extend_ports=(("o1", EXTENSION), ("o2", EXTENSION)),
        boundary_offset=(MARGIN, 0, 0, -MARGIN, z_margin, -z_margin), name="stack",
    )
    config = stack.get_simulation_config(
        time=DURATION, grid=fdtdx.core.grid.UniformGrid(spacing=resolution * 1e-6),
        gradient_config=fdtdx.GradientConfig(method="checkpointed", num_checkpoints=60),  # bounded backward memory
    )
    wave = fdtdx.WaveCharacter(wavelength=WAVELENGTH * 1e-6)
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=wave, spectral_width=fdtdx.WaveCharacter(frequency=0.05 * wave.get_frequency())
    )
    te0 = dict(mode_index=0, filter_pol="te")
    volume, constraints = stack.get_simulation_volume()
    objects = [volume, stack]
    for obj, placing in (
        # source and monitors in the straight leads, clear of the bend whose edges move
        stack.create_mode_plane_source_at(
            "o1", PADDING, wave_character=wave, temporal_profile=pulse, port_offset=-1.0, name="source", **te0,
        ),
        stack.create_mode_overlap_detector_at(
            "o1", PADDING, direction="in", wave_characters=(wave,), scaling_mode="pulse", port_offset=-0.5,
            name="in", **te0,
        ),
        stack.create_mode_overlap_detector_at(
            "o2", PADDING, wave_characters=(wave,), scaling_mode="pulse", port_offset=-0.5, name="out", **te0,
        ),
    ):
        objects.append(obj)
        constraints += placing
    field = fdtdx.PhasorDetector(  # E on the xy plane through the middle of the core, for the figure
        name="field", wave_characters=(wave,), components=("Ex", "Ey", "Ez"), scaling_mode="pulse",
        partial_grid_shape=(None, None, 1),
    )
    objects.append(field)
    constraints += [field.same_size(volume, axes=(0, 1)), field.place_at_center(volume)]
    boundaries, placing = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=round(PML / resolution)), volume
    )
    objects += boundaries.values()
    constraints += placing
    objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, KEY)
    return objects, arrays, {**params, **bf.init_stack_params(objects)}, config
# --8<-- [end:simulation]


# --8<-- [start:objective]
def simulate(half_widths, placed):
    """Transmission (TE0 power out over TE0 power in) of the bend with these half-widths, and the final arrays."""
    objects, arrays, params, config = placed
    stack = objects["stack"]
    p = {**params, "stack": {**params["stack"], **stack.pack(tight_bend(half_widths))}}  # the new edges' vertices
    arrays, objects, _ = bf.apply_stack(arrays, objects, p, KEY)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=KEY, show_progress=False)
    a_in = objects["in"].compute_overlap(arrays.detector_states["in"])
    a_out = objects["out"].compute_overlap(arrays.detector_states["out"])
    return jnp.abs(a_out / a_in)[0] ** 2, arrays


def half_widths_of(z):
    """Unbounded parameters to half-widths within (H_MIN, H_MAX): the optimizer never leaves the bounds."""
    return H_MIN + (H_MAX - H_MIN) * jax.nn.sigmoid(z)


def parameters_of(half_widths):
    """The inverse of half_widths_of."""
    u = (jnp.asarray(half_widths) - H_MIN) / (H_MAX - H_MIN)
    return jnp.log(u / (1 - u))
# --8<-- [end:objective]


# --8<-- [start:optimize]
def describe(half_widths):
    """The range of a profile's width, and of its middle's shift toward the bend's center (um)."""
    inner, outer = np.asarray(half_widths)
    width, shift = inner + outer, (inner - outer) / 2
    return f"width {width.min():.2f} to {width.max():.2f}, shift {shift.min():+.2f} to {shift.max():+.2f} um"


def optimize(placed, iterations, learning_rate=0.03):
    """Adam on the transmission; returns the best half-widths seen and the transmission of every step."""
    value_and_grad = jax.value_and_grad(lambda z: simulate(half_widths_of(z), placed)[0])  # eager: no jax.jit
    optimizer = optax.adam(learning_rate)
    z = parameters_of(UNIFORM)  # start from the uniform bend
    state = optimizer.init(z)
    history, best = [], (-1.0, None)
    for step in range(iterations):
        transmission, gradient = value_and_grad(z)
        transmission = float(transmission)
        history.append(transmission)
        print(f"step {step:2d}  T = {transmission:.4f}  {describe(half_widths_of(z))}")
        if transmission > best[0]:
            best = (transmission, half_widths_of(z))
        updates, state = optimizer.update(-gradient, state)  # optax minimizes: ascend on T
        z = optax.apply_updates(z, updates)
    return best[1], history
# --8<-- [end:optimize]


def field_image(half_widths, placed, resolution):
    """The transmission, and |E|^2, the permittivity and the extent (um) of the xy plane through the core."""
    transmission, arrays = simulate(half_widths, placed)
    objects = placed[0]
    stack, k = objects["stack"], objects["field"].grid_slice[2].start
    phasor = np.asarray(arrays.detector_states["field"]["phasor"][0, 0, :, :, :, 0])
    nx, ny = phasor.shape[1:]
    sx, sy, _ = (float(v) * 1e6 for v in stack.partial_real_shape)
    x0, y0 = float(stack.center[0]) - sx / 2, float(stack.center[1]) - sy / 2
    extent = (x0, x0 + nx * resolution, y0, y0 + ny * resolution)
    eps = 1 / np.asarray(arrays.inv_permittivities[0, :, :, k])
    return float(transmission), np.sum(np.abs(phasor) ** 2, axis=0), eps, extent


def plot(images, history, path):
    """The uniform and optimized fields side by side, and the transmission per step."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    peak = max(field.max() for _, field, _, _ in images.values())
    for ax, (name, (transmission, field, eps, (x0, x1, y0, y1))) in zip(axes, images.items()):
        ax.imshow(field.T / peak, origin="lower", extent=(x0, x1, y0, y1), cmap="inferno", vmax=1)
        xs, ys = np.linspace(x0, x1, eps.shape[0]), np.linspace(y0, y1, eps.shape[1])
        ax.contour(xs, ys, eps.T, levels=[(CORE_INDEX**2 + CLAD_INDEX**2) / 2], colors="white", linewidths=0.6)
        ax.set(xlabel="x (um)", ylabel="y (um)", title=f"{name}: T = {transmission:.3f}")
    axes[2].plot(range(len(history)), history, "o-")
    axes[2].set(xlabel="step", ylabel="transmission", title="optimization")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=50, help="optimization steps")
    parser.add_argument("--resolution", type=float, default=RESOLUTION, help="grid cell (um) to optimize on")
    parser.add_argument("--check", type=float, default=CHECK_RESOLUTION, help="finer grid cell (um); 0 skips it")
    parser.add_argument("--figure", default="inverse_design.png", help="where to save the figure; '' skips it")
    args = parser.parse_args()

    started = time.perf_counter()
    # --8<-- [start:run]
    placed = place(args.resolution)
    half_widths, history = optimize(placed, args.iterations)
    print(f"optimized: {describe(half_widths)}")
    images = {name: field_image(h, placed, args.resolution) for name, h in (("uniform", UNIFORM), ("optimized", half_widths))}
    t_uniform, t_optimized = images["uniform"][0], images["optimized"][0]
    print(f"transmission at {args.resolution * 1e3:.0f} nm: uniform {t_uniform:.4f}, optimized {t_optimized:.4f}")
    if args.check:
        fine = place(args.check)  # the same two designs on a finer grid: is the gain real?
        t_uniform, t_optimized = (float(simulate(h, fine)[0]) for h in (UNIFORM, half_widths))
        print(f"transmission at {args.check * 1e3:.0f} nm: uniform {t_uniform:.4f}, optimized {t_optimized:.4f}")
    # --8<-- [end:run]
    print(f"wall time: {time.perf_counter() - started:.0f} s on {jax.devices()[0].device_kind}")
    if args.figure:
        plot(images, history, args.figure)
        print(f"figure: {args.figure}")
