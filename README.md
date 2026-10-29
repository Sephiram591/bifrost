<p align="center"><img src="docs/images/bifrost-logo-rounded-rectangle.svg" alt="bifrost" width="360"></p>

# bifrost

[![docs](https://github.com/Sephiram591/bifrost/actions/workflows/pages.yml/badge.svg)](https://sephiram591.github.io/bifrost/)
[![tests](https://github.com/Sephiram591/bifrost/actions/workflows/test.yml/badge.svg)](https://github.com/Sephiram591/bifrost/actions/workflows/test.yml)

The rainbow bridge between your GDSFactory(X) pipeline and FDTD(X) simulation.

bifrost turns a [gdsfactory](https://gdsfactory.github.io/gdsfactory/) component and its layer stack into a
differentiable [fdtdx](https://github.com/ymahlau/fdtdx) simulation, and ships a 3D viewer for setting up the run and
inspecting its fields. It builds on [gdsfactoryx](https://github.com/Sephiram591/gdsfactoryx), the JAX port of gdsfactory.

**Documentation:** https://sephiram591.github.io/bifrost/ (with a [live viewer demo](https://sephiram591.github.io/bifrost/guide/viewer/#live-demo))

## Install

Python 3.12 to 3.14. fdtdx is a required dependency, so one command installs everything. Use a fresh environment:
gdsfactoryx's distribution is named `gdsfactory`, so it replaces an upstream gdsfactory installed in the same one.

```bash
python -m venv bifrost-env && source bifrost-env/bin/activate
pip install "bifrost @ git+https://github.com/Sephiram591/bifrost.git"
```

For GPU simulations also install a CUDA build of JAX (see the [installation page](https://sephiram591.github.io/bifrost/installation/)).

## Quick start

A straight silicon waveguide in the viewer (in a Jupyter or marimo cell, end the cell with `viewer`):

```python
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
    included_layers=["core"],                        # the box's z range is the core's
    extend_ports=(("o1", 1.0), ("o2", 1.0)),         # 1 um straight leads out of both ports
    boundary_offset=(0, 0, 1.5, -1.5, 1.5, -1.5),    # 1.5 um of cladding in y and z
    name="stack",
)
viewer = bf.view(stack)  # the interactive 3D viewer
```

Next: [run your first simulation](https://sephiram591.github.io/bifrost/getting-started/), a 90 degree bend from an empty
file to a transmission value and a field image.
