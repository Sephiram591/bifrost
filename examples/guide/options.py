"""Geometry options: build the same device with the default (most accurate) and the fast options.

Run it:  python examples/guide/options.py
"""

# --8<-- [start:options]
import fdtdx
import gdsfactory as gf
import bifrost as bf
from gdsfactory.technology import LayerLevel, LayerStack

gf.gpdk.PDK.activate()

component = gf.components.straight(length=5)  # a 5 um long, 500 nm wide waveguide with ports o1 and o2
layer_stack = LayerStack(
    layers={"core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=0.22, material="core", mesh_order=1)}
)
materials = {"core": fdtdx.Material(permittivity=3.48**2), "clad": fdtdx.Material(permittivity=1.444**2)}


def build(options: bf.StackOptions) -> bf.StackDevice:
    return bf.StackDevice.from_layer_stack(
        component, layer_stack, materials, background="clad", zrange_layers=["core"],
        extend_ports=(("o1", 1.0), ("o2", 1.0)), boundary_offset=(0, 0, 1.5, -1.5, 1.5, -1.5),
        options=options, name="stack",
    )


accurate = build(bf.StackOptions())        # every accuracy feature on: the default
quick = build(bf.StackOptions.fast())      # for a quick look: no corner model, no Yee staggering, no diagnostics

for name, device in (("default", accurate), ("fast", quick)):
    o = device.options
    print(f"{name:8} averaging={o.averaging} yee_staggered={o.yee_staggered} corner_lines={o.corner_lines} diagnostics={o.diagnostics}")
# --8<-- [end:options]
