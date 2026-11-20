# Changelog

## Unreleased

- [Automatic meshing](guide/simulation-setup.md#automatic-meshing): `bf.mesh(stack, band, refine=bf.Refine(...))` builds a graded simulation grid from the device geometry, wavelength band, and optimization motion bounds (`Refine.dilation` per layer or per vertex, `Refine.z_padding` per level):
  - Refines materials to `lambda / (n ppw)` where they exist or may move, extends fine grids by 3 evanescent decay lengths past lower-index boundaries, resolves narrow features and gaps (`feature_cells`), and supports explicit refinement regions (`Refine.regions`) and substrate coarsening (`Refine.material_cells`).
  - Places continuous real-valued cell edges along each axis (`bf.axis_edges`, or discrete multiples via `bf.lattice_edges` when `quantum` is set), satisfying dual-cell targets, neighbor `growth` limits, uniform PML regions, mirror symmetry, and mid-cell centering of static horizontal interfaces (`center_faces=True`).
  - Returns `Mesh` (`mesh.grid`, `mesh.boundary`, `mesh.summary()`) alongside `StackDevice.coarse_vertices` for tracking moving vertices under `jax.jit`.
- Exact Green's-theorem grid-line voxelizer (`voxelizer="greens"`, now the default) with unified Kottke homogenization and dynamic junction handling, eliminating static corner-column and junction capacities.
- [Graded grids](guide/simulation-setup.md#graded-grids): `StackDevice` supports `fdtdx.core.grid.RectilinearGrid` with non-uniform cells along any axis across all averaging modes (`"subpixel"`, `"subpixel_full"`, and `"inverse"`), using `bf.graded_edges`, `StackDevice.rectilinear_grid`, and `grid=` on port factory methods.
- `bf.ModePlaneSource` and `bf.ModeOverlapDetector` (returned by `create_mode_plane_source_at` and `create_mode_overlap_detector_at`) normalize launched mode fields and overlap area weights against the simulation-wide `min_spacing**2` on non-uniform `RectilinearGrid`s, removing port-area bias across planes of different sizes, paddings, or orientations.
- Viewer support for non-uniform grids, adaptive meshing controls (defaulting to adaptive `ppw=15` with per-layer dilation sub-cards), and client-side MP4 video export via WebCodecs and `wrap_video`.
- `averaging="subpixel_full"`: junction sub-cells whose two dominant interface directions carry nearly equal weight blend smoothly toward axis-weighted laminates so the homogenized tensor and its derivatives stay continuous across corner crossings.
- `place_on_grid` warns when a level excluded during `from_layer_stack` (beyond `level_margin + 0.1` µm) is sampled by a coarse cell at a box face.

## 0.2.0

- [Inverse design](guide/inverse-design.md) guide and example (`examples/guide/inverse_design.py`): optimizes the inner and outer width profile of an air-clad silicon nitride bend across 70 stations for TE0 transmission (raising transmission from 56 % to 99.3 % at `ppw=20`).
- `bf.jit`: wraps `jax.jit` with XLA's `simplify-conditional` pass disabled so gradients through fdtdx's checkpointed time loop avoid evaluating a time step's VJP on forward recomputation iterations.
- Requires `gdsfactoryx >= 9.51.0.post2`, supporting traceable geometry inside `jax.jit` and matching upstream gdsfactory GDS output and naming conventions.
- Backend-neutral installation docs: bifrost works with any JAX accelerator backend configured per the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
- Moved the viewer design specification out of the documentation site to `design/viewer-spec.md` in the repository.

## 0.1.0

Initial release on PyPI as `bifrost-photonics` (`import bifrost as bf`), separating the fdtdx stack integration and 3D viewer from `gdsfactoryx`.

- `bf.StackDevice`: voxelizes a gdsfactory component and its `LayerStack` into one differentiable fdtdx object with port-attached sources and detectors, sidewalls, derived layers, corner handling, subpixel and full-tensor averaging, roughness, dispersive and anisotropic materials, diagnostics, and timing.
- `StackDevice(dtype=...)`: supports `jnp.float32` (default) or `jnp.float64` precision across the simulation.
- `bf.view`: interactive 3D/2D viewer widget with precomputed [Widget Demo](live-demo.md) and standalone [Explorer demo](demo/index.html).
- Examples: [waveguide bend](examples/waveguide-bend.md), [elevator coupler](examples/elevator-coupler.md), and [directional coupler](examples/directional-coupler.md) benchmarked against MEEP.
- Dependencies: `fdtdx` (required), `gdsfactoryx` from PyPI, and `tidy3d >= 2.12`.
