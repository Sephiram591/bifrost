"""A straight silicon waveguide in bifrost's viewer: the shortest complete example.

Run it in a Jupyter or marimo cell and end the cell with ``viewer`` to display the widget.
"""

# --8<-- [start:straight]
import fdtdx
import gdsfactory as gf
import bifrost as bf
from gdsfactory.technology import LayerLevel, LayerStack

gf.gpdk.PDK.activate()

component = gf.components.straight(length=5)  # a 5 um long, 500 nm wide waveguide with ports o1 and o2

# one 220 nm thick level on the waveguide layer; silicon core in a silica cladding
layer_stack = LayerStack(
    layers={"core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=0.22, material="core", mesh_order=1)}
)
materials = {
    "core": fdtdx.Material(permittivity=3.48**2),
    "clad": fdtdx.Material(permittivity=1.444**2),
}

stack = bf.StackDevice.from_layer_stack(
    component, layer_stack, materials,
    background="clad",                               # fills everything the core does not
    zrange_layers=["core"],                          # the box's z range is the core's
    extend_ports=(("o1", 1.0), ("o2", 1.0)),         # 1 um straight leads out of both ports
    boundary_offset=(0, 0, 1.5, -1.5, 1.5, -1.5),    # 1.5 um of cladding in y and z
    name="stack",
)
viewer = bf.view(stack)  # the interactive 3D viewer
# --8<-- [end:straight]
