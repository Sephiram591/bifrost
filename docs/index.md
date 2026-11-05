![bifrost](images/bifrost-logo-rounded-rectangle.svg){ width="360" }

# bifrost

The rainbow bridge between your [GDSFactory](https://gdsfactory.github.io/gdsfactory/) pipeline and
[FDTD](https://github.com/ymahlau/fdtdx) simulation.

bifrost turns a [gdsfactoryx](https://github.com/Sephiram591/gdsfactoryx) (the JAX port
of gdsfactory that makes layouts differentiable) component and its layer stack into a differentiable [fdtdx](https://github.com/ymahlau/fdtdx)
simulation. It also ships a 3D viewer for setting up the run and inspecting its fields. Below is an example of a straight silicon waveguide, opened in the viewer:

```python
--8<-- "examples/straight_waveguide.py:straight"
```

`bf.StackDevice` is the device the solver sees: the component's polygons, extruded using the layer stack data, with ports ready to take sources and monitors. 

## Where to go next

<div class="grid cards" markdown>

- **[Installation](installation.md)**

    How to set up the bifrost with the JAX port of GDSFactory.

- **[Run your first simulation](getting-started.md)**

    A 90° waveguide bend from an empty file to a transmission value and a field image.

- **[Explore the live demo](live-demo.md)**

    The result of that simulation in the viewer, in your browser, with no installation.

- **[Read the user guide](guide/layer-stacks.md)**

    Layer stacks, simulation setup, results, the viewer, gradients, inverse design and diagnostics.

</div>