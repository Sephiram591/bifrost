# API reference

Import the package as `import bifrost as bf`. For `Component`, parametric cells, routing, and PDKs, see [gdsfactoryx](https://github.com/Sephiram591/gdsfactoryx) (the JAX port of gdsfactory) and the [gdsfactory API reference](https://gdsfactory.github.io/gdsfactory/api/).

| Name | Description |
| --- | --- |
| [`bf.StackDevice`][bifrost.stack.StackDevice] | Component and `LayerStack` packaged as one differentiable fdtdx object with port-attached sources and monitors |
| [`bf.StackOptions`][bifrost.stack.StackOptions] | Discretization and subpixel accuracy options that trade accuracy for build/solve time |
| [`bf.TopRoughness`][bifrost.stack.TopRoughness] | Gaussian random-field top-surface roughness for a stack level |
| [`bf.apply_stack`][bifrost.stack.apply_stack] | Writes `StackDevice` layers into simulation permittivity arrays in place of `fdtdx.apply_params` |
| [`bf.init_stack_params`][bifrost.stack.init_stack_params] | Returns differentiable stack parameters (polygon vertices, heights, thicknesses, sidewall angles) |
| [`bf.StackDiagnosticWarning`][bifrost.stack.StackDiagnosticWarning] | Warning emitted when features or gaps are narrower than the grid resolves |
| [`bf.ModePlaneSource`][bifrost.stack.ModePlaneSource] | `fdtdx.ModePlaneSource` subclass with port-area-invariant power normalization on graded grids |
| [`bf.ModeOverlapDetector`][bifrost.stack.ModeOverlapDetector] | `fdtdx.ModeOverlapDetector` subclass with port-area-invariant overlap normalization on graded grids |
| [`bf.mesh`][bifrost.meshing.mesh] | Generates an adaptive non-uniform grid from device geometry, wavelength band, and `Refine` bounds |
| [`bf.Refine`][bifrost.meshing.Refine] | Configuration for moving-boundary dilation, z padding, evanescent tails, explicit boxes, and material targets |
| [`bf.Mesh`][bifrost.meshing.Mesh] | Output of `bf.mesh` holding axis edges, `fdtdx` grid and boundary configs, and a summary helper |
| [`bf.CoarseVertices`][bifrost.meshing.CoarseVertices] | Report from `StackDevice.coarse_vertices` counting vertices outside their target cell resolution |
| [`bf.axis_edges`][bifrost.grids.axis_edges] | Places continuous 1D cell edges along an axis meeting target cell widths, PMLs, and centered interfaces |
| [`bf.lattice_edges`][bifrost.grids.lattice_edges] | Places 1D cell edges restricted to whole multiples of a discrete `quantum` step |
| [`bf.graded_edges`][bifrost.grids.graded_edges] | Builds smoothly graded 1D cell edges with optional `snap` coordinates for `StackDevice.rectilinear_grid` |
| [`bf.AxisEdges`][bifrost.grids.AxisEdges] | Structured result (edges, PML cell counts, alignment, symmetry, centered faces) from `axis_edges` / `lattice_edges` |
| [`bf.jit`][bifrost.jitting.jit] | `jax.jit` wrapper adding the XLA options needed for checkpointed `fdtdx` gradients |
| [`bf.compiler_options`][bifrost.jitting.compiler_options] | Merges bifrost's XLA compiler flags with custom `jax.jit` `compiler_options` |
| [`bf.COMPILER_OPTIONS`][bifrost.jitting.COMPILER_OPTIONS] | XLA compiler flags added by `bf.jit` (`simplify-conditional` disabled) |
| [`bf.record_timings`][bifrost.timing.record_timings] | Context manager for recording wall, trace, compile, and run times of stack geometry steps |
| [`bf.view`][bifrost.viewer.widget.view] | Launches the interactive 3D/2D viewer widget for a `StackDevice` |

Modules: [bf.stack](stack.md), [bf.geometry](geometry.md), [bf.grids](grids.md), [bf.meshing](meshing.md), [bf.jitting](jitting.md), [bf.subpixel and bf.timing](subpixel-timing.md), and [bf.viewer](viewer.md).
