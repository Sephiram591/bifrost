"""Layer stacks and materials: the same waveguide in a bare cladding and on a substrate.

Run it:  python examples/guide/layer_stacks.py
"""

# --8<-- [start:imports]
import fdtdx
import gdsfactory as gf
import bifrost as bf
from gdsfactory.technology import LayerLevel, LayerStack

gf.gpdk.PDK.activate()

component = gf.components.straight(length=5)  # a 5 um long, 500 nm wide waveguide with ports o1 and o2
# --8<-- [end:imports]

# --8<-- [start:bare]
# 1. One level and two materials: a silicon core in a uniform silica cladding.
layer_stack = LayerStack(
    layers={"core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=0.22, material="core", mesh_order=1)}
)
materials = {
    "core": fdtdx.Material(permittivity=3.48**2),
    "clad": fdtdx.Material(permittivity=1.444**2),
}
stack = bf.StackDevice(
    component, layer_stack, materials,
    background="clad",                              # fills everything no level covers
    zrange_layers=["core"],                         # the box's z range starts as the core's
    extend_ports=(("o1", 1.0), ("o2", 1.0)),        # straight leads out of both ports
    boundary_offset=(0, 0, 1.5, -1.5, 1.5, -1.5),   # (x+, x-, y+, y-, z+, z-) in um
    name="stack",
)
print("box (um):", [round(float(s) * 1e6, 3) for s in stack.partial_real_shape])
# --8<-- [end:bare]

# --8<-- [start:substrate]
# 2. The same guide on a 2 um silica substrate under air. "WAFER" is in inverted_layers by
#    default, so with no polygons drawn on it its inverse covers the whole plane.
substrate_stack = LayerStack(
    layers={
        "box": LayerLevel(layer="WAFER", zmin=-2.0, thickness=2.0, material="silica", mesh_order=9),
        "core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=0.22, material="silicon", mesh_order=1),
    }
)
substrate_materials = {
    "silicon": fdtdx.Material(permittivity=3.48**2),
    "silica": fdtdx.Material(permittivity=1.444**2),
    "air": fdtdx.Material(),
}
on_substrate = bf.StackDevice(
    component, substrate_stack, substrate_materials,
    background="air",
    zrange_layers=["core"],
    extend_ports=(("o1", 1.0), ("o2", 1.0)),
    boundary_offset=(0, 0, 1.5, -1.5, 1.5, -1.5),
    name="on_substrate",
)
print("box (um):", [round(float(s) * 1e6, 3) for s in on_substrate.partial_real_shape])
# --8<-- [end:substrate]

# --8<-- [start:inverted]
# 3. Negative-tone guide: polygons on DEEP_ETCH define etched trenches, so inverting
#    DEEP_ETCH applies silicon everywhere outside them, leaving a 500 nm core between.
etched = gf.Component()
etched.add_polygon([(0.0, 0.25), (5.0, 0.25), (5.0, 2.0), (0.0, 2.0)], layer=gf.gpdk.LAYER.DEEP_ETCH)
etched.add_polygon([(0.0, -2.0), (5.0, -2.0), (5.0, -0.25), (0.0, -0.25)], layer=gf.gpdk.LAYER.DEEP_ETCH)
etched.add_port("o1", center=(0.0, 0.0), width=0.5, orientation=180, layer=gf.gpdk.LAYER.DEEP_ETCH)
etched.add_port("o2", center=(5.0, 0.0), width=0.5, orientation=0, layer=gf.gpdk.LAYER.DEEP_ETCH)

negative_stack = LayerStack(
    layers={
        "box": LayerLevel(layer="WAFER", zmin=-2.0, thickness=2.0, material="silica", mesh_order=9),
        "core": LayerLevel(layer=gf.gpdk.LAYER.DEEP_ETCH, zmin=0.0, thickness=0.22, material="silicon", mesh_order=1),
    }
)
negative_guide = bf.StackDevice(
    etched, negative_stack, substrate_materials,
    background="air",
    zrange_layers=["core"],
    extend_ports=(("o1", 1.0), ("o2", 1.0)),
    boundary_offset=(0, 0, 0, 0, 1.5, -1.5),
    inverted_layers=("WAFER", gf.gpdk.LAYER.DEEP_ETCH),
    name="negative_guide",
)
# --8<-- [end:inverted]

viewer = bf.view(on_substrate)  # in a notebook cell, end the cell with `viewer`
