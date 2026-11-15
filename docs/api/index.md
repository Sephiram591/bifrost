# API reference

Import as `import bifrost as bf`. For `Component`, cells, routing and PDKs see the
[gdsfactory API reference](https://gdsfactory.github.io/gdsfactory/api/).

| Name | What it is |
| --- | --- |
| [`bf.StackDevice`][bifrost.stack.StackDevice] | a component and its layer stack as one fdtdx object; places sources and monitors on ports |
| [`bf.StackOptions`][bifrost.stack.StackOptions] | the geometry options that trade accuracy for time |
| [`bf.TopRoughness`][bifrost.stack.TopRoughness] | a random rough top surface for one level |
| [`bf.apply_stack`][bifrost.stack.apply_stack] | writes the stacks into the permittivity arrays, in place of `fdtdx.apply_params` |
| [`bf.init_stack_params`][bifrost.stack.init_stack_params] | the stacks' differentiable parameters (vertices, heights, thicknesses) |
| [`bf.StackDiagnosticWarning`][bifrost.stack.StackDiagnosticWarning] | the warning for features the grid does not resolve |
| [`bf.ModePlaneSource`][bifrost.stack.ModePlaneSource] | `fdtdx.ModePlaneSource` normalized to `min_spacing**2` across graded port planes |
| [`bf.ModeOverlapDetector`][bifrost.stack.ModeOverlapDetector] | `fdtdx.ModeOverlapDetector` normalized to `min_spacing**2` across graded port planes |
| [`bf.mesh`][bifrost.meshing.mesh] | a device's graded grid from its geometry, the band and where it may move |
| [`bf.Refine`][bifrost.meshing.Refine] | where `bf.mesh` refines: the geometry's motion (dilation, z padding) and the tails |
| [`bf.Mesh`][bifrost.meshing.Mesh] | `bf.mesh`'s result: the edges, the fdtdx grid and boundaries, a summary |
| [`bf.CoarseVertices`][bifrost.meshing.CoarseVertices] | `StackDevice.coarse_vertices`' report: vertices in cells wider than their target |
| [`bf.axis_edges`][bifrost.grids.axis_edges] | continuous cell edges of one axis within their targets, PMLs and centered faces |
| [`bf.lattice_edges`][bifrost.grids.lattice_edges] | exact cell edges of one axis on a lattice of quanta, within their targets |
| [`bf.graded_edges`][bifrost.grids.graded_edges] | cell edges of a graded axis, for `StackDevice.rectilinear_grid` |
| [`bf.jit`][bifrost.jitting.jit] | `jax.jit` with the XLA option gradients through fdtdx's time loop need |
| [`bf.compiler_options`][bifrost.jitting.compiler_options] | those options merged with your own, for another jit wrapper |
| [`bf.record_timings`][bifrost.timing.record_timings] | times the geometry steps |
| [`bf.view`][bifrost.viewer.widget.view] | the 3D viewer widget for a `StackDevice` |

The modules: [bf.stack](stack.md), [bf.geometry](geometry.md), [bf.grids](grids.md), [bf.meshing](meshing.md),
[bf.jit](jitting.md),
[bf.subpixel and bf.timing](subpixel-timing.md) and [bf.viewer](viewer.md).
