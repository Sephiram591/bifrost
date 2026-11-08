# Changelog

## Unreleased

- [Graded grids](guide/simulation-setup.md#graded-grids): a `StackDevice` runs on an fdtdx `RectilinearGrid` whose
  cells differ along any axis (every averaging). Fill fractions stay exact, each field component averages over the dual
  cells between cell centers along the axes it is staggered on, and `averaging="subpixel_full"` solves a junction
  voxel's cell problem on its own box (`bf.subpixel.periodic_cell_tensor(..., sides=...)`). Where the x and y cells
  differ, a voxel is painted as a rectangle with Euclidean normals and round corners; on graded x or y every column on
  its own (half-planes, corner model, sub-samples and junctions), with roughness drawn on an even lattice and
  interpolated to the columns, and finite-difference normals from the distances between cell centers. `bf.graded_edges` places graded edges (fine regions, a growth limit,
  interfaces on edges, uniform cells under the PML), `StackDevice.rectilinear_grid` builds the grid over the box, and
  the `create_*_at` methods take `grid=` to give port planes whole cells there. Uniform grids are unchanged, bit for bit.
- The viewer adopts results recorded on non-uniform grids (detectors at their physical extent, from the grid's
  edges) and draws their field planes resampled onto an even grid at each graded axis's finest cell.
- `place_on_grid` warns when a level that `from_layer_stack` left out (outside the box by more than `level_margin` and
  a 0.1 um slack) is sampled by a coarse cell at the box's face; build the device with a larger `level_margin` then.

## 0.2.0

- [Inverse design](guide/inverse-design.md) guide and example (`examples/guide/inverse_design.py`): the width on each
  side of an air-clad silicon nitride bend, at 70 stations, optimized for TE0 transmission through the simulation
  (56 % to 99.5 %; the uniform bend converts most of the rest to TE1).
- `bf.jit`: `jax.jit` with an XLA option (`simplify-conditional` disabled) for gradients through fdtdx's checkpointed
  time loop, which plain `jax.jit` compiles into a backward loop that computes a time step's VJP on every iteration.
  The inverse-design example compiles its whole optimization step with it.
- Dependencies: gdsfactoryx 9.51.0.post2 or newer, whose geometry traces under `jax.jit`, so a whole objective that
  builds a component can be compiled, and whose GDS output, reference names and cross-section names match upstream
  gdsfactory's.
- Installation docs are backend-neutral: bifrost runs on any JAX backend; set JAX up with the
  [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
- The viewer design spec moved out of the documentation site to `design/viewer-spec.md` in the repository.

## 0.1.0

First release on PyPI, as `bifrost-photonics` (the import is still `import bifrost as bf`); the fdtdx stack and viewer
moved here from gdsfactoryx.

- `bf.StackDevice`: a gdsfactory component and its `LayerStack` voxelized into one differentiable fdtdx object, with
  sources and monitors placed on ports: sidewalls, derived layers, exact corners, subpixel and full-tensor averaging,
  roughness, dispersive and anisotropic materials, diagnostics and timing.
- `StackDevice(dtype=...)`: the whole simulation in float32 (default) or float64.
- `bf.view`: the 3D viewer widget for setting up, running and inspecting a simulation, with a precomputed
  [live demo](live-demo.md).
- Examples: the [waveguide bend](examples/waveguide-bend.md) tutorial, and the
  [elevator](examples/elevator-coupler.md) and [directional](examples/directional-coupler.md) couplers compared with
  MEEP.
- Dependencies: fdtdx (required, no extra), gdsfactoryx from PyPI, tidy3d 2.12 or newer.
