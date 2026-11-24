# Advanced

Beyond the basic workflow in [Simulation Setup](simulation-setup.md), bifrost supports virtual plane ports and grid-snapping controls, customizable adaptive meshing, and double-precision (`float64`) simulations.

## Plane ports, z centering and grid snapping

- **Plane ports (`PLANE_PORTS` in [`bifrost.stack`][bifrost.stack]):** Passing `"xy"`, `"xz"`, or `"yz"` as `port` places the source, detector, or point dipole on the plane through the simulation box's center, spanning the box's in-plane extents rather than a component port (they are virtual frames, not entries in `stack.ports`). For plane ports, `"in"` faces `-normal`, `"out"` faces `+normal`, and positive `port_offset` shifts along `"in"`; `port_padding=(0, 0)` keeps the box extent unchanged.
- **Pinning z to box center (`Z_CENTER_LEVEL` in [`bifrost.stack`][bifrost.stack]):** Passing `level=bf.stack.Z_CENTER_LEVEL` centers the plane at `stack.z_center` instead of the middle of the port's or named level's z span. For component ports, the height still comes from the port's level span; for plane ports, `level=None` also centers on `stack.z_center`, while a level name places the plane at that level's span middle.
- **Non-uniform grid snapping (`grid=`):** Passing `grid=config` (a `SimulationConfig` or resolved `RectilinearGrid`) to any plane factory on [`StackDevice`][bifrost.stack.StackDevice] snaps the plane's faces to the nearest grid edges along any non-uniform in-plane axis and expresses the size in whole cells rather than physical length. This requires the `StackDevice` to fill the simulation volume from [`get_simulation_volume`][bifrost.stack.StackDevice.get_simulation_volume]. [`create_point_dipole_source_at`][bifrost.stack.StackDevice.create_point_dipole_source_at] always occupies one cell and does not take `grid=`.

## Adaptive meshing

[`bf.mesh`][bifrost.meshing.mesh] builds a graded grid once before a run or an optimization: fine wherever high-index materials exist or may move, coarse elsewhere. You specify the wavelength band and optional bounds on where geometry may move during optimization ([`Refine`][bifrost.meshing.Refine]), while the parameters themselves remain unconstrained:

```python
mesh = bf.mesh(
    stack, (1.50, 1.60),                       # the band (um)
    refine=bf.Refine(dilation={"WG": 0.45}),   # how far the WG layer's boundary may move (um)
)
print(mesh.summary(time=500e-15))              # cells, cell sizes, time step, memory
config = stack.get_simulation_config(time=500e-15, grid=mesh.grid)
boundaries, placing = fdtdx.boundary_objects_from_config(mesh.boundary, volume)
source, placing = stack.create_mode_plane_source_at("o1", (0.5, 0.5), wave, grid=config)
```

- **Target cell sizes and motion bounds.** Each level requests cells of at most `lambda / (n ppw)` across its geometry, within `Refine.dilation` of moving layers (a scalar per layer, or one value per vertex matching `stack.init_params()["vertices/<layer>"]`), and across faces shifted by `Refine.z_padding`. Overlapping levels follow painting order: a region occluded by a static higher-priority level is not refined further, whereas regions exposed by a moving level are.
- **Evanescent tails.** Past every boundary toward a lower index, fine cells continue for `global_dilation` (laterally) and `global_z_padding` (vertically) decay lengths `l = lambda / (2 pi sqrt(n_high^2 - n_low^2))`—3 decay lengths by default. Stopping fine cells right at a silicon core boundary (`0` decay lengths) introduces roughly an order of magnitude more effective-index error than extending them by 1–3 decay lengths; see [Convergence testing](convergence-testing.md#why-adaptive-meshing-wins-per-cell).
- **Narrow features and full-wafer layers.** Narrow parts and gaps in the starting geometry receive `feature_cells` refinement, `Refine.regions` adds explicit fine boxes, and `min_cell` prevents thin films or narrow gaps from forcing an overly small time step. Conductive or metallic materials require explicit targets in `Refine.material_cells`. Because the grid is a tensor product, each axis coordinate takes the smallest target anywhere in its cross-sectional plane: coarsen a full-wafer substrate or cladding with `Refine.material_cells` (13 to 14 cells per wavelength are usually enough) to keep margins and PMLs coarse.
- **Edge placement.** [`bf.axis_edges`][bifrost.grids.axis_edges] places continuous real-valued cell edges so every primal cell and dual cell stays within its target, neighboring cells (and periodic wrap-around cells) differ by at most `growth`, and each PML boundary receives uniform cells covering `pml` µm (`boundaries` can set faces to `"periodic"`, `"pec"`, `"pmc"`, or `"bloch"` instead). The solver maximizes the smallest cell (which governs the CFL time step), keeps cell counts minimal, preserves mirror symmetry when the geometry is symmetric, aligns uniform runs where compatible, and accepts any box size (or pass `quantum` to snap edges to multiples of a discrete step via [`bf.lattice_edges`][bifrost.grids.lattice_edges]).
- **Mid-cell interface centering (`center_faces=True`).** Static horizontal material interfaces are placed at the exact midpoint of a z cell rather than on a cell edge, avoiding asymmetric dual-cell averaging errors and sub-cell boundary oscillations. Two faces a fractional number of cells apart may require slightly narrower cells (allowing the smallest z cell up to 20 % below the uncentered optimum); faces under a PML or moved by `Refine.z_padding` remain uncentered. [`Mesh.summary`][bifrost.meshing.Mesh.summary] lists the centered faces.
- **Tracking vertices outside refined zones.** [`stack.coarse_vertices`][bifrost.stack.StackDevice.coarse_vertices] counts vertices that move into cells wider than their target, per layer and axis. It runs inside `jax.jit` (useful as an auxiliary output of an optimization objective), and `mesh.summary()` reports it for the starting geometry. A level whose `z_padding` moves it into the box from beyond `level_margin` is rejected at mesh creation, since placing the device would drop it.
- **Accuracy vs cell count.** Across benchmarks including sloped waveguides, anisotropic lithium niobate bends, multi-level tapers, directional couplers, and curved SiN bends, `bf.mesh` matches uniform fine-grid accuracy with **1.7× to 3.6× fewer cells** and **1.7× to 5.0× shorter runtime** (and up to **6×–12× fewer cells** at tighter fine-grid tolerances; `ppw=14` offers a faster setting for initial design sweeps). See [Convergence testing](convergence-testing.md) for full convergence curves and tuning examples, and [`bf.meshing`][bifrost.meshing] for every parameter and default of `mesh` and `Refine`.

## Precision

The whole simulation runs in the [`StackDevice`][bifrost.stack.StackDevice]'s `dtype`, `jnp.float32` (the default) or `jnp.float64`, and the device is the single place to configure it:

```python
stack = bf.StackDevice(
    component, layer_stack, materials, background="clad", zrange_layers=["core"], dtype=jnp.float64
)
mesh = bf.mesh(stack, 1.55)
config = stack.get_simulation_config(time=100e-15, grid=mesh.grid)
```

- [`get_simulation_config`][bifrost.stack.StackDevice.get_simulation_config] builds `fdtdx.SimulationConfig` in the device's dtype and forwards every other argument (`time`, `grid`, `backend`, `gradient_config`, ...); passing `dtype=` directly to it is an error. A manually constructed config must match the device's dtype or placement raises a `ValueError`.
- The `create_*_at` monitors match the device precision automatically: phasor, mode-overlap, and diffractive monitors use `complex64` or `complex128`, and field, energy, and Poynting-flux monitors use `float32` or `float64`, unless overridden with `dtype=`. fdtdx sources follow the config dtype.
- Importing `gdsfactory` enables JAX's 64-bit mode globally, and [`bf.apply_stack`][bifrost.stack.apply_stack] keeps fdtdx's PML coefficients in the device's dtype, so `float32` simulations run cleanly with x64 enabled. A `dtype=jnp.float64` device requires JAX's 64-bit mode to remain enabled at placement time.
- `float64` doubles array memory and is substantially slower on GPUs tuned for single precision. Note that in fdtdx 0.6.3, phasor monitor time windows and uniform-grid mode-overlap face weights still run in `float32`; on non-uniform `RectilinearGrid`s, [`bf.ModeOverlapDetector`][bifrost.stack.ModeOverlapDetector] computes face weights in `float64` when configured for double-precision overlap monitors and `float32` otherwise.
