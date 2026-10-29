"""A straight silicon waveguide in bifrost's viewer: the shortest complete example.

Run it in a Jupyter or marimo cell and end the cell with ``viewer`` to display the widget.
"""

# --8<-- [start:straight]
import fdtdx
import gdsfactory as gf
import bifrost as bf
from gdsfactory.technology import LayerLevel, LayerStack

gf.gpdk.PDK.activate()

CORE = (1, 0)  # (layer, datatype) the waveguide is drawn on

# a 5 um long, 500 nm wide strip with a port on each end
component = gf.Component()
component.add_polygon([(0, -0.25), (5, -0.25), (5, 0.25), (0, 0.25)], layer=CORE)
component.add_port("o1", center=(0, 0), width=0.5, orientation=180, layer=CORE)
component.add_port("o2", center=(5, 0), width=0.5, orientation=0, layer=CORE)

# one 220 nm thick level on that layer; silicon core in a silica cladding
layer_stack = LayerStack(
    layers={"core": LayerLevel(layer=CORE, zmin=0.0, thickness=0.22, material="core", mesh_order=1)}
)
materials = {
    "core": fdtdx.Material(permittivity=3.48**2),
    "clad": fdtdx.Material(permittivity=1.444**2),
}

stack = bf.StackDevice.from_layer_stack(
    component, layer_stack, materials,
    background="clad",                               # fills everything the core does not
    included_layers=["core"],                        # the box's z range is the core's
    extend_ports=(("o1", 1.0), ("o2", 1.0)),         # 1 um straight leads out of both ports
    boundary_offset=(0, 0, 1.5, -1.5, 1.5, -1.5),    # 1.5 um of cladding in y and z
    name="stack",
)
viewer = bf.view(stack)  # the interactive 3D viewer
# --8<-- [end:straight]
