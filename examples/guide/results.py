"""Results and field monitors: run a short simulation of a straight waveguide and read the monitors.

The grid is coarse (50 nm) and the run short, so this finishes in a minute or two on a CPU or in seconds on a GPU.
Run it:  python examples/guide/results.py
"""

# --8<-- [start:simulate]
import fdtdx
import jax
import jax.numpy as jnp
import numpy as np
import gdsfactory as gf
import bifrost as bf
from gdsfactory.technology import LayerLevel, LayerStack

gf.gpdk.PDK.activate()

component = gf.components.straight(length=5)  # a 5 um long, 500 nm wide waveguide with ports o1 and o2
layer_stack = LayerStack(
    layers={"core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=0.22, material="core", mesh_order=1)}
)
materials = {"core": fdtdx.Material(permittivity=3.48**2), "clad": fdtdx.Material(permittivity=1.444**2)}
stack = bf.StackDevice.from_layer_stack(
    component, layer_stack, materials, background="clad", zrange_layers=["core"],
    extend_ports=(("o1", 2.0), ("o2", 2.0)), name="stack",
    boundary_offset=(0, 0, 1.6, -1.6, 1.59, -1.59),  # a 9 x 3.7 x 3.4 um box: whole cells at 50 nm
)

RESOLUTION = 0.05  # um
config = fdtdx.SimulationConfig(
    time=300e-15, grid=fdtdx.core.grid.UniformGrid(spacing=RESOLUTION * 1e-6), dtype=jnp.float32
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
    # mode monitors: how much of TE0 passes each port
    stack.create_mode_overlap_detector_at(
        "o1", (0.5, 0.3), direction="in", wave_characters=(wave,), scaling_mode="pulse", name="in", **te0,
    ),
    stack.create_mode_overlap_detector_at(
        "o2", (0.5, 0.3), wave_characters=(wave,), scaling_mode="pulse", name="out", **te0,
    ),
    # Poynting-flux monitors: the power through a plane at every time step (no wave of their own, so
    # the padding's unit is wavelength_um)
    stack.create_poynting_flux_detector_at("o1", (0.5, 0.3), wavelength_um=1.55, direction="in", name="flux_in"),
    stack.create_poynting_flux_detector_at("o2", (0.5, 0.3), wavelength_um=1.55, name="flux_out"),
):
    objects.append(obj)
    constraints += placing
# a field monitor: E on the xy plane through the middle of the core, at the wavelength
field = fdtdx.PhasorDetector(
    name="field", wave_characters=(wave,), components=("Ex", "Ey", "Ez"),
    scaling_mode="pulse", partial_grid_shape=(None, None, 1),
)
objects.append(field)
constraints += [field.same_size(volume, axes=(0, 1)), field.place_at_center(volume)]
boundaries, placing = fdtdx.boundary_objects_from_config(
    fdtdx.BoundaryConfig.from_uniform_bound(thickness=round(1.0 / RESOLUTION)), volume
)
objects += boundaries.values()
constraints += placing

key = jax.random.PRNGKey(0)
objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
arrays, objects, _ = bf.apply_stack(arrays, objects, {**params, **bf.init_stack_params(objects)}, key)
with jax.enable_x64(False):  # gdsfactory turns on 64-bit floats; the solver loop needs them off
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key)
# --8<-- [end:simulate]

# --8<-- [start:read]
states = arrays.detector_states  # one entry per monitor, by name

# mode monitors: the complex amplitude of TE0, one entry per wave character
a_in = objects["in"].compute_overlap(states["in"])
a_out = objects["out"].compute_overlap(states["out"])
transmission = float(np.ravel(abs(a_out / a_in) ** 2)[0])
print(f"transmission: {transmission:.3f}")

# the Poynting-flux monitors: the power through each plane, one row per time step. Summed over the run it is
# the energy that crossed, so the ratio of the two is the fraction of the light that got through
flux_in = np.asarray(states["flux_in"]["poynting_flux"]).sum()
flux_out = np.asarray(states["flux_out"]["poynting_flux"]).sum()
print(f"flux ratio: {float(flux_out / flux_in):.3f}")

# the field monitor: a phasor per component, on the monitor's grid
phasor = np.asarray(states["field"]["phasor"][0, 0, :, :, :, 0])  # (3 components, nx, ny)
intensity = np.sum(np.abs(phasor) ** 2, axis=0)                   # |E|^2
print("field grid:", intensity.shape, "peak |E|^2:", float(intensity.max()))
# --8<-- [end:read]

# --8<-- [start:viewer]
viewer = bf.view(stack)
viewer.load_result(objects, arrays, config, stack=stack)  # shows the monitors' fields in the Result tab
# --8<-- [end:viewer]
