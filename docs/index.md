![bifrost](images/bifrost-logo-final.svg){ width="360" }

# bifrost

The rainbow bridge between your [GDSFactory(X)](https://gdsfactory.github.io/gdsfactory/) pipeline and
[FDTD(X)](https://github.com/ymahlau/fdtdx) simulation.

bifrost takes a gdsfactory component and its `LayerStack` and turns them into a differentiable fdtdx simulation, with
a 3D/field viewer for setting up and inspecting the run. No GDS export, no re-describing the geometry with fdtdx
primitives.

```python
import bifrost as bf
import gdsfactory as gf

gf.gpdk.PDK.activate()
stack = bf.StackDevice.from_layer_stack(component, layer_stack, materials=materials, background="sio2")
viewer = bf.view(stack)   # interactive 3D viewer, sources/detectors, run
```

```
component + LayerStack --> bf.StackDevice --> fdtdx simulation --> bf.view
   (gdsfactory(x))        (voxelize, ports)   (apply_stack, run)    (fields, video)
```

## Features

- **Layer-stack voxelization**: every `LayerLevel` (logical and derived layers, `mesh_order`, sidewall angle, bias) into
  one multi-material region, with sub-pixel smoothing and full permittivity tensors.
- **Gradients**: differentiable with respect to polygon vertices and level heights, so `jax.grad` flows from the
  simulation result back to the cell function parameters
  (needs [gdsfactoryx](https://github.com/Sephiram591/gdsfactoryx)'s JAX backend).
- **Ports to sources and detectors**: simulation volume and sources/detectors placed on the component's ports.
- **Viewer**: an `anywidget` with a 3D viewport, cross-sections, field display, timeline and video export.
- **Diagnostics and timing** of the geometry steps.

## Where to go next

- [Installation](installation.md)
- [Getting started](getting-started.md)
- User guide: [StackDevice](guide/stack-device.md), [options](guide/options.md),
  [gradients](guide/gradients.md), [diagnostics](guide/diagnostics.md), [geometry model](guide/geometry-model.md)
- [Viewer](viewer.md)
- [Validation and benchmarks](validation.md)
- [Examples](examples/elevator-coupler.md)
- [API reference](api/index.md)
- [Development](development.md)

## Related projects

- [gdsfactory](https://gdsfactory.github.io/gdsfactory/): components, cells, PDKs and routing. For anything about
  building a layout, use its documentation.
- [gdsfactoryx](https://github.com/Sephiram591/gdsfactoryx): the JAX port of gdsfactory that bifrost builds on, which
  makes layouts differentiable.
- [fdtdx](https://github.com/ymahlau/fdtdx): the differentiable FDTD solver that bifrost drives.
