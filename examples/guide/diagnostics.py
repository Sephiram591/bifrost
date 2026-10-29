"""Geometry diagnostics: a strip narrower than the grid resolves, and the device says so.

Run it:  python examples/guide/diagnostics.py
"""

# --8<-- [start:diagnostics]
import warnings

import fdtdx
import jax
import jax.numpy as jnp
import gdsfactory as gf
import bifrost as bf
from gdsfactory.technology import LayerLevel, LayerStack

gf.gpdk.PDK.activate()

CORE = (1, 0)
component = gf.Component()
component.add_polygon([(0, -0.02), (5, -0.02), (5, 0.02), (0, 0.02)], layer=CORE)  # a 40 nm wide strip
component.add_port("o1", center=(0, 0), width=0.04, orientation=180, layer=CORE)
component.add_port("o2", center=(5, 0), width=0.04, orientation=0, layer=CORE)
layer_stack = LayerStack(
    layers={"core": LayerLevel(layer=CORE, zmin=0.0, thickness=0.22, material="core", mesh_order=1)}
)
materials = {"core": fdtdx.Material(permittivity=3.48**2), "clad": fdtdx.Material(permittivity=1.444**2)}
stack = bf.StackDevice.from_layer_stack(
    component, layer_stack, materials, background="clad", included_layers=["core"],
    extend_ports=(("o1", 1.0), ("o2", 1.0)),
    boundary_offset=(0, 0, 1.48, -1.48, 1.39, -1.39),  # a 7 x 3 x 3 um box: whole cells at 50 nm
    name="stack",
)

config = fdtdx.SimulationConfig(
    time=50e-15, grid=fdtdx.core.grid.UniformGrid(spacing=50e-9), dtype=jnp.float32  # 50 nm cells
)
volume, constraints = stack.get_simulation_volume()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    fdtdx.place_objects([volume, stack], config, constraints, jax.random.PRNGKey(0))  # placing is when the checks run

for warning in caught:
    if issubclass(warning.category, bf.StackDiagnosticWarning):
        print(warning.message)
# --8<-- [end:diagnostics]
