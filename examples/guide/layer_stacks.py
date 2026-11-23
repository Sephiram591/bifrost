"""Layer stacks and materials: common photonic LayerStack and fdtdx.Material definitions.

Run it:  python examples/guide/layer_stacks.py
"""

# --8<-- [start:imports]
import fdtdx
import gdsfactory as gf
from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer

gf.gpdk.PDK.activate()
# --8<-- [end:imports]

# --8<-- [start:bare]
# 1. One level and two isotropic materials: a 220 nm silicon core in silica cladding.
layer_stack = LayerStack(
    layers={
        "core": LayerLevel(
            layer=gf.gpdk.LAYER.WG,
            zmin=0.0,
            thickness=0.22,
            material="core",
            mesh_order=1,
        ),
    }
)
materials = {
    "core": fdtdx.Material(permittivity=3.48**2),
    "clad": fdtdx.Material(permittivity=1.444**2),
}
# --8<-- [end:bare]

# --8<-- [start:substrate]
# 2. A silicon core on a 2 um buried-oxide (BOX) substrate under air.
substrate_stack = LayerStack(
    layers={
        "box": LayerLevel(layer="WAFER", zmin=-2.0, thickness=2.0, material="silica", mesh_order=9),
        "core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=0.22, material="silicon", mesh_order=1),
    }
)
substrate_materials = {
    "silicon": fdtdx.Material(permittivity=3.48**2),
    "silica": fdtdx.Material(permittivity=1.444**2),
    "air": fdtdx.Material(permittivity=1.0),
}
# --8<-- [end:substrate]

# --8<-- [start:linbo3]
# 3. X-cut thin-film lithium niobate (TFLN) ridge with 20 deg sloped sidewalls and
#    diagonally anisotropic permittivity on a 3 um silica bottom oxide in air.
n_o, n_e = 2.211, 2.138  # ordinary (x, z) and extraordinary (optic axis along y) at 1.55 um

linbo3_stack = LayerStack(
    layers={
        "box": LayerLevel(layer="WAFER", zmin=-3.0, thickness=3.0, material="silica", mesh_order=9),
        "core": LayerLevel(
            layer=gf.gpdk.LAYER.WG,
            zmin=0.0,
            thickness=0.50,
            sidewall_angle=20.0,
            material="linbo3",
            mesh_order=1,
        ),
    }
)
linbo3_materials = {
    "linbo3": fdtdx.Material(permittivity=(n_o**2, n_e**2, n_o**2)),
    "silica": fdtdx.Material(permittivity=1.444**2),
    "air": fdtdx.Material(permittivity=1.0),
}
# --8<-- [end:linbo3]

# --8<-- [start:derived]
# 4. Partial-etch rib waveguide using LogicalLayer boolean expressions: a 220 nm rib
#    where WG is not etched by SLAB150, over a 90 nm slab across the union of both.
wg = LogicalLayer(layer=gf.gpdk.LAYER.WG)
etch = LogicalLayer(layer=gf.gpdk.LAYER.SLAB150)

rib_stack = LayerStack(
    layers={
        "box": LayerLevel(layer="WAFER", zmin=-2.0, thickness=2.0, material="silica", mesh_order=9),
        "slab": LayerLevel(
            layer=wg | etch, derived_layer=wg, zmin=0.0, thickness=0.09, material="silicon", mesh_order=2
        ),
        "rib": LayerLevel(
            layer=wg - etch, derived_layer=wg, zmin=0.0, thickness=0.22, material="silicon", mesh_order=1
        ),
    }
)
# --8<-- [end:derived]

# --8<-- [start:inverted]
# 5. Negative-tone trench-etch stack: DEEP_ETCH polygons mark etched trenches, so
#    treating DEEP_ETCH as inverted applies silicon everywhere outside the trenches.
negative_stack = LayerStack(
    layers={
        "box": LayerLevel(layer="WAFER", zmin=-2.0, thickness=2.0, material="silica", mesh_order=9),
        "core": LayerLevel(layer=gf.gpdk.LAYER.DEEP_ETCH, zmin=0.0, thickness=0.22, material="silicon", mesh_order=1),
    }
)
# --8<-- [end:inverted]
