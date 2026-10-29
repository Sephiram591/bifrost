"""Simulation setup and ports: sources and monitors on a waveguide, placed on a coarse grid.

It builds and places everything and stops before the solve, so it runs in seconds on a CPU.
Run it:  python examples/guide/simulation_setup.py
"""

# --8<-- [start:device]
import fdtdx
import jax
import jax.numpy as jnp
import gdsfactory as gf
import bifrost as bf
from gdsfactory.technology import LayerLevel, LayerStack

gf.gpdk.PDK.activate()

CORE = (1, 0)
component = gf.Component()
component.add_polygon([(0, -0.25), (5, -0.25), (5, 0.25), (0, 0.25)], layer=CORE)
component.add_port("o1", center=(0, 0), width=0.5, orientation=180, layer=CORE)
component.add_port("o2", center=(5, 0), width=0.5, orientation=0, layer=CORE)

layer_stack = LayerStack(
    layers={"core": LayerLevel(layer=CORE, zmin=0.0, thickness=0.22, material="core", mesh_order=1)}
)
materials = {
    "core": fdtdx.Material(permittivity=3.48**2),
    "clad": fdtdx.Material(permittivity=1.444**2),
}
stack = bf.StackDevice.from_layer_stack(
    component, layer_stack, materials, background="clad", included_layers=["core"],
    extend_ports=(("o1", 2.0), ("o2", 2.0)),          # the leads end at the box faces ...
    # ... which are 1.6 um from the guide in y and 1.59 um from the core in z. The box is then
    # 9 x 3.7 x 3.4 um: a whole number of cells at the 50 nm grid below.
    boundary_offset=(0, 0, 1.6, -1.6, 1.59, -1.59),
    name="stack",
)
# --8<-- [end:device]

# --8<-- [start:settings]
RESOLUTION = 0.05  # um, coarse enough for a quick check
config = fdtdx.SimulationConfig(
    time=100e-15,                                                    # seconds of simulated time
    grid=fdtdx.core.grid.UniformGrid(spacing=RESOLUTION * 1e-6),     # the cell edge, in meters
    dtype=jnp.float32,
)
wave = fdtdx.WaveCharacter(wavelength=1.55e-6)                       # fdtdx works in meters; bifrost's lengths are um
pulse = fdtdx.GaussianPulseProfile(
    center_wave=wave, spectral_width=fdtdx.WaveCharacter(frequency=0.05 * wave.get_frequency())
)
# --8<-- [end:settings]

# --8<-- [start:ports]
volume, constraints = stack.get_simulation_volume()   # the box, from the device's bounding box
objects = [volume, stack]
te0 = dict(mode_index=0, filter_pol="te")             # the fundamental TE mode
for obj, placing in (
    # a mode source 0.5 um outside port o1 (negative port_offset moves it out of the component)
    stack.create_mode_plane_source_at(
        "o1", (0.5, 0.3), wave_character=wave, temporal_profile=pulse, port_offset=-0.5, name="source", **te0,
    ),
    # mode monitors on both ports; "in" counts the mode travelling into the component
    stack.create_mode_overlap_detector_at(
        "o1", (0.5, 0.3), direction="in", wave_characters=(wave,), scaling_mode="pulse", name="in", **te0,
    ),
    stack.create_mode_overlap_detector_at(
        "o2", (0.5, 0.3), wave_characters=(wave,), scaling_mode="pulse", name="out", **te0,
    ),
):
    objects.append(obj)
    constraints += placing
# --8<-- [end:ports]

# --8<-- [start:place]
boundaries, placing = fdtdx.boundary_objects_from_config(
    fdtdx.BoundaryConfig.from_uniform_bound(thickness=round(1.0 / RESOLUTION)), volume   # 1 um of PML
)
objects += boundaries.values()
constraints += placing

key = jax.random.PRNGKey(0)
objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
print("grid cells:", objects.volume.grid_shape)
for monitor in (*objects.sources, *objects.detectors):
    print(f"{monitor.name:8} {monitor.grid_shape} cells at {[(s.start, s.stop) for s in monitor.grid_slice]}")
# --8<-- [end:place]
