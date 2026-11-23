![bifrost](images/bifrost-logo-rounded-rectangle.svg){ width="360" }

# bifrost

The rainbow bridge between your [gdsfactory](https://gdsfactory.github.io/gdsfactory/) pipeline and
[fdtdx](https://github.com/ymahlau/fdtdx) simulation.

bifrost turns a [gdsfactoryx](https://github.com/Sephiram591/gdsfactoryx) (the JAX port
of gdsfactory) component and its layer stack into a differentiable [fdtdx](https://github.com/ymahlau/fdtdx)
simulation. It also ships a 3D viewer for setting up the run and inspecting its fields. Below is an example of simulating a straight silicon waveguide:

```python
import fdtdx
import jax
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

stack = bf.StackDevice(
    component, layer_stack, materials,
    background="clad",                               # fills everything the core does not
    zrange_layers=["core"],                          # the box's z range is the core's
    extend_ports=(("o1", 1.5), ("o2", 1.5)),         # straight leads through the 1 um PMLs
    boundary_offset=(0, 0, 1.5, -1.5, 1.5, -1.5),    # 1.5 um of cladding in y and z
    name="stack",
)

mesh = bf.mesh(stack, 1.55)
config = stack.get_simulation_config(time=300e-15, grid=mesh.grid)
wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
volume, constraints = stack.get_simulation_volume()
source, s_c = stack.create_mode_plane_source_at("o1", (0.5, 0.3), wave, grid=config)
detector, d_c = stack.create_mode_overlap_detector_at(
    "o2", (0.5, 0.3), (wave,), name="out", grid=config, switch=fdtdx.OnOffSwitch(start_time=200e-15),
)
boundaries, b_c = fdtdx.boundary_objects_from_config(mesh.boundary, volume)

key = jax.random.PRNGKey(0)
objects, arrays, params, config, _ = fdtdx.place_objects(
    [volume, stack, source, detector, *boundaries.values()], config, constraints + s_c + d_c + b_c, key,
)
arrays, objects, _ = bf.apply_stack(arrays, objects, {**params, **bf.init_stack_params(objects)}, key)
_, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key)
print("transmission:", float(abs(objects["out"].compute_overlap(arrays.detector_states["out"])[0]) ** 2))
```

`bf.StackDevice` is the helper class where all the magic happens, handling adaptive FDTD meshing and allowing you to create sources/detectors on ports. 

!!! warning "LLM Disclaimer"
    bifrost is largely developed using LLM models from providers including Claude, ChatGPT, and Google. We should always validate any code shipped by "AI". Conveniently, the MEEP FDTD package provides a reference against which to test FDTD simulations built by bifrost and run by FDTDX. [See our convergence testing section](guide/convergence-testing.md) for comparison results between bifrost and MEEP.

## Where to go next

<div class="grid cards" markdown>

- **[Installation](installation.md)**

    How to set up bifrost with [gdsfactoryx](https://github.com/Sephiram591/gdsfactoryx), the JAX port of gdsfactory.

- **[Run your first simulation](getting-started.md)**

    Simulate a 90° waveguide bend and upload results, including a time domain video, to the viewer.

- **[Explore the Widget Demo](live-demo.md)**

    The result of the "Getting Started" simulation in an interactive notebook in your browser (or open the standalone [Explorer demo](demo/index.html)).

- **[Read the user guide](guide/layer-stacks.md)**

    Layer stacks, simulation setup, convergence testing, results, gradients, inverse design and diagnostics.

</div>