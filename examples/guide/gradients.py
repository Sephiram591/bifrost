"""Gradients: the derivative of the phase a bend adds with respect to its radius, through the simulation.

The derivative is that of the discretized simulation; it is compared with a central finite difference
of the same simulation. Run it on a GPU:  python examples/guide/gradients.py
"""

# --8<-- [start:gradient]
import fdtdx
import jax
import jax.numpy as jnp
import gdsfactory as gf
import bifrost as bf
from gdsfactory.technology import LayerLevel, LayerStack

gf.gpdk.PDK.activate()

RADIUS = 2.0  # um: the parameter we differentiate with respect to
RESOLUTION = 0.04  # um, the grid of the bend tutorial

layer_stack = LayerStack(
    layers={"core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=0.22, material="core", mesh_order=1)}
)
materials = {"core": fdtdx.Material(permittivity=3.48**2), "clad": fdtdx.Material(permittivity=1.444**2)}

# a gdsfactory bend (any differentiable cell works); the ports are extended 2 um, and the box is 6 x 6 x 3.4 um.
# 2 um is tighter than the strip's 3.5 um minimum radius, so that check is off.
component = gf.components.bend_circular(radius=RADIUS, angle=90, allow_min_radius_violation=True)
stack = bf.StackDevice.from_layer_stack(
    component, layer_stack, materials, background="clad", included_layers=["core"],
    extend_ports=(("o1", 2.0), ("o2", 2.0)), boundary_offset=(1.75, 0, 0, -1.75, 1.59, -1.59), name="stack",
)

config = fdtdx.SimulationConfig(
    time=400e-15, grid=fdtdx.core.grid.UniformGrid(spacing=RESOLUTION * 1e-6), dtype=jnp.float32,
    # how the backward pass trades memory for time: it re-simulates from 60 stored states
    # instead of keeping every step, so the memory stays small
    gradient_config=fdtdx.GradientConfig(method="checkpointed", num_checkpoints=60),
)
wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
pulse = fdtdx.GaussianPulseProfile(
    center_wave=wave, spectral_width=fdtdx.WaveCharacter(frequency=0.05 * wave.get_frequency())
)
te0 = dict(mode_index=0, filter_pol="te")
volume, constraints = stack.get_simulation_volume()
objects = [volume, stack]
for obj, placing in (
    stack.create_mode_plane_source_at(
        "o1", (0.5, 0.3), wave_character=wave, temporal_profile=pulse, port_offset=-0.5, name="source", **te0,
    ),
    stack.create_mode_overlap_detector_at(
        "o1", (0.5, 0.3), direction="in", wave_characters=(wave,), scaling_mode="pulse", name="in", **te0,
    ),
    # the output monitor sits 0.5 um out in the straight extension (port_offset < 0 moves a plane out of the
    # component), so that it stays in the straight lead for every radius we try; on the port itself it would
    # sit inside the bend as soon as the radius grows
    stack.create_mode_overlap_detector_at(
        "o2", (0.5, 0.3), wave_characters=(wave,), scaling_mode="pulse", port_offset=-0.5, name="out", **te0,
    ),
):
    objects.append(obj)
    constraints += placing
boundaries, placing = fdtdx.boundary_objects_from_config(
    fdtdx.BoundaryConfig.from_uniform_bound(thickness=round(1.0 / RESOLUTION)), volume
)
objects += boundaries.values()
constraints += placing

key = jax.random.PRNGKey(0)
objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
params = {**params, **bf.init_stack_params(objects)}
placed = objects["stack"]  # the device as placed: it can swap in the vertices of another component


def ratio(radius):
    """The complex ratio of the TE0 amplitude on the output monitor to that on the input monitor."""
    bend = gf.components.bend_circular(radius=radius, angle=90, allow_min_radius_violation=True)  # new radius
    p = {**params, "stack": {**params["stack"], **placed.pack(bend)}}  # its vertices into the stack's parameters
    arrays2, objects2, _ = bf.apply_stack(arrays, objects, p, key)
    with jax.enable_x64(False):
        _, arrays2 = fdtdx.run_fdtd(arrays=arrays2, objects=objects2, config=config, key=key, show_progress=False)
    a_in = objects2["in"].compute_overlap(arrays2.detector_states["in"])
    a_out = objects2["out"].compute_overlap(arrays2.detector_states["out"])
    return (a_out / a_in)[0]


# The overall sign of a mode is arbitrary in each mode solve, so the phase of the ratio is only defined
# modulo pi (ratio**2 is not affected). We compare it with the phase at the base radius, which keeps the
# result away from the branch cut: it is the extra phase (rad) that a bend of this radius adds.
reference = jnp.angle(ratio(RADIUS) ** 2) / 2


def phase_shift(radius):
    """The phase (rad, modulo pi) of the bend of this radius relative to the base radius."""
    return jnp.angle(ratio(radius) ** 2 * jnp.exp(-2j * reference)) / 2


value, gradient = jax.value_and_grad(phase_shift)(RADIUS)  # the simulation runs forward and backward
h = 0.05  # um: a central difference over +-h
finite = (phase_shift(RADIUS + h) - phase_shift(RADIUS - h)) / (2 * h)
print(f"phase shift at {RADIUS} um: {float(value):+.3f} rad (zero by construction)")
print(f"d(phase)/dR: autodiff {float(gradient):+.2f} rad per um, finite difference {float(finite):+.2f} rad per um")
# --8<-- [end:gradient]
