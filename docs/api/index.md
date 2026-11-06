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
| [`bf.jit`][bifrost.jitting.jit] | `jax.jit` with the XLA option gradients through fdtdx's time loop need |
| [`bf.compiler_options`][bifrost.jitting.compiler_options] | those options merged with your own, for another jit wrapper |
| [`bf.record_timings`][bifrost.timing.record_timings] | times the geometry steps |
| [`bf.view`][bifrost.viewer.widget.view] | the 3D viewer widget for a `StackDevice` |

The modules: [bf.stack](stack.md), [bf.geometry](geometry.md), [bf.jit](jitting.md),
[bf.subpixel and bf.timing](subpixel-timing.md) and [bf.viewer](viewer.md).
