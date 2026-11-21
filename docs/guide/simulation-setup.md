# Simulation setup and ports

A bifrost simulation is an ordinary [fdtdx](https://github.com/ymahlau/fdtdx) simulation in which the device, the sources, and the monitors come from `StackDevice`. Because `StackDevice` knows where its ports are, you attach sources and monitors to ports instead of computing box coordinates by hand. The example below builds and places a setup without running the solver, finishing in seconds on a CPU.

## Build, place and inspect

Create the `StackDevice` and configure the grid, time window, and optical pulse:

```python
--8<-- "examples/guide/simulation_setup.py:device"

--8<-- "examples/guide/simulation_setup.py:settings"
```

- **Units.** bifrost's lengths (`boundary_offset`, `extend_ports`, `port_offset`, `bf.mesh`) are in **µm**. fdtdx expects SI units (**meters** and **seconds**): `SimulationConfig(time=...)` and `WaveCharacter(wavelength=...)` take seconds and meters (`* 1e-6`).
- **The grid.** `bf.mesh(stack, 1.55)` builds an adaptive non-uniform grid (`mesh.grid`, 20 points per local wavelength by default) and 1 µm PML boundaries (`mesh.boundary`) over any box size; see [Automatic meshing](#automatic-meshing).
- **The pulse.** A Gaussian pulse with a spectral width of 5 % of the center frequency ramps up smoothly and lets one run cover a band; the monitors below record the center wavelength.
- **The precision.** `stack.get_simulation_config(...)` creates an `fdtdx.SimulationConfig` using the device's `dtype`; see [Precision](#precision).

Next, create the simulation volume along with the source and monitors attached to ports `o1` and `o2`:

```python
--8<-- "examples/guide/simulation_setup.py:ports"
```

Add the PML boundaries from `mesh.boundary`, then run `fdtdx.place_objects` to fit every object onto the grid:

```python
--8<-- "examples/guide/simulation_setup.py:place"
```

Output:

```text
grid cells: (405, 95, 83)
source   (1, 67, 43) cells at [(67, 68), (14, 81), (20, 63)]
in       (1, 67, 43) cells at [(90, 91), (14, 81), (20, 63)]
out      (1, 67, 43) cells at [(315, 316), (14, 81), (20, 63)]
```

The box is 405 × 95 × 83 non-uniform cells. Each slice is `(start, stop)` cell indices along x, y, and z: each plane is one cell thick along its normal and 67 × 43 cells across, snapped via `grid=config` to the cell edges around the 0.5 µm guide plus 0.5 wavelengths per side in y and 0.3 wavelengths per side in z. The source sits 0.5 µm before the input monitor at x = 0.

## Sources and monitors on ports

`stack.get_simulation_volume()` returns the simulation box and its placement constraints. Each `stack.create_<object>_at(port, port_padding, ...)` method returns an fdtdx object together with the constraints that place it on that port, centered and normal to the port's direction. Add both to your lists:

```python
for obj, placing in (stack.create_mode_plane_source_at("o1", (0.5, 0.3), ...),):
    objects.append(obj)
    constraints += placing
```

The arguments every method shares:

| Argument | Meaning |
| --- | --- |
| `port` | a component port name, or a simulation-box plane port (`"xy"`, `"xz"`, `"yz"` in `bf.stack.PLANE_PORTS`) |
| `port_padding` | `(lateral, vertical)`: the plane extends this far past the port's width and the level's height on each side, in units of the mean wavelength of the object's wave characters (negative values shrink the plane). Field, energy and Poynting-flux monitors have none, so they take `wavelength_um` for the unit |
| `direction` | `"in"` (into the component, or `-normal` on a plane port) or `"out"`; sources default to `"in"`, monitors to `"out"` |
| `port_offset` | shift in µm along the injection direction; a negative value moves the plane out of the component, e.g. into a port extension |
| `level` | the level whose z span sets the plane's height (default: the levels that draw the port's layer, or volume mid-height for a plane port); pass `bf.stack.Z_CENTER_LEVEL` (`"z_center"`) to pin the z center to `stack.z_center` |
| `grid` | optional `SimulationConfig` or `RectilinearGrid` for plane methods (`create_point_dipole_source_at` is always one cell); snaps faces to the nearest grid edges along any non-uniform in-plane axis |

!!! note "Where to put a source and its monitors"
    - Keep source and monitor planes **outside the PML** and well clear of the box faces: the box's absorbing layer lies inside the box, so the port extension has to be longer than the PML plus the offsets.
    - Keep the plane **inside the box**. A plane larger than the box is clipped to it. With 1.55 µm light, a padding of 3 wavelengths is larger than a small box, so give the box room or reduce the padding.
    - Launch **just outside the port** (`port_offset=-0.5`) so the source sits in the straight extension, and put the input monitor on the port itself, as above.
    - Planes must be tall enough to hold the mode: the mode of a 220 nm silicon guide reaches about 0.5 µm into the cladding above and below, so a 1.1 µm tall plane is sufficient.

The methods and the objects they create (`bf.*` wrappers live in `bifrost.stack` and subclass the matching `fdtdx` class):

| Method | Object created |
| --- | --- |
| `create_mode_plane_source_at` | `bf.ModePlaneSource` (`fdtdx.ModePlaneSource` subclass) |
| `create_gaussian_plane_source_at` | `fdtdx.GaussianPlaneSource` |
| `create_uniform_plane_source_at` | `fdtdx.UniformPlaneSource` |
| `create_hard_constant_amplitude_plane_source_at` | `HardConstantAmplitudePlanceSource` (`fdtdx`) |
| `create_point_dipole_source_at` | `fdtdx.PointDipoleSource` |
| `create_mode_overlap_detector_at` | `bf.ModeOverlapDetector` (`fdtdx.ModeOverlapDetector` subclass) |
| `create_poynting_flux_detector_at` | `fdtdx.PoyntingFluxDetector` |
| `create_phasor_detector_at` | `fdtdx.PhasorDetector` |
| `create_field_detector_at` | `fdtdx.FieldDetector` |
| `create_energy_detector_at` | `fdtdx.EnergyDetector` |
| `create_diffractive_detector_at` | `fdtdx.DiffractiveDetector` |

On non-uniform grids, `bf.ModePlaneSource` and `bf.ModeOverlapDetector` normalize mode power against a common reference area (`min_spacing**2`) so launched power and overlap ratios do not depend on port-plane padding, width, or orientation; on uniform grids they behave identically to `fdtdx`'s classes.

Other fdtdx fields go in as keyword arguments. Boundaries, regular fdtdx devices, and other objects are added as usual.

### Plane ports, z centering and grid snapping

- **Plane ports (`bf.stack.PLANE_PORTS = {"yz": 0, "xz": 1, "xy": 2}`):** Passing `"xy"`, `"xz"`, or `"yz"` as `port` places the source, detector, or point dipole on the plane through the simulation box's center, spanning the box's in-plane extents (`width`, `height`) rather than a component port (they are virtual frames, not entries in `stack.ports`). For plane ports, `"in"` faces the negative normal axis (`-normal`), `"out"` faces `+normal`, and positive `port_offset` shifts along `"in"` (`-normal`); `port_padding=(0, 0)` keeps the box extent unchanged.
- **Pinning z to box center (`bf.stack.Z_CENTER_LEVEL = "z_center"`):** Passing `level=bf.stack.Z_CENTER_LEVEL` centers the plane at `stack.z_center` instead of the middle of the port's or named level's z span. For component ports, the height still comes from the port's level span; for plane ports, `level=None` also centers on `stack.z_center`, while a level name places the plane at that level's span middle.
- **Non-uniform grid snapping (`grid=`):** Passing `grid=config` (a `SimulationConfig` or resolved `RectilinearGrid`) to any plane `create_*_at` method snaps the plane's faces to the nearest grid edges along any non-uniform in-plane axis and expresses the size in whole cells (`partial_grid_shape`) rather than physical length (`partial_real_shape`). This requires the `StackDevice` to fill the simulation volume from `get_simulation_volume(...)`. `create_point_dipole_source_at` always uses `partial_grid_shape=(1, 1, 1)` and does not take `grid=`.

## Place, apply and run

After `fdtdx.place_objects`, `bf.apply_stack` writes the stack into the permittivity arrays, in place of `fdtdx.apply_params` (see [Waveguide bend](../examples/waveguide-bend.md) for a full end-to-end run):

```python
objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
arrays, objects, _ = bf.apply_stack(arrays, objects, {**params, **bf.init_stack_params(objects)}, key)
_, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key)
```

fdtdx's own `Device` objects only mix two materials, so `fdtdx.apply_params` does not update a `StackDevice`. `apply_stack` writes every stack into the arrays first and then calls `fdtdx.apply_params`, so regular devices and sources are still updated, and devices win where they overlap a stack. `bf.init_stack_params(objects)` supplies the stack's own parameters (vertices, heights, thicknesses) for differentiation; see [Gradients](gradients.md).

## Automatic meshing

`bf.mesh` builds a graded grid once before a run or an optimization: fine wherever high-index materials exist or may move, coarse elsewhere. You specify the wavelength band and optional bounds on where geometry may move during optimization (`Refine`), while the parameters themselves remain unconstrained:

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

- **Target cell sizes and motion bounds.** Each level requests cells of at most `lambda / (n ppw)` (`ppw=20` by default) across its geometry, within `Refine.dilation` of moving layers (a scalar per layer, or one value per vertex matching `stack.init_params()["vertices/<layer>"]`), and across faces shifted by `Refine.z_padding`. Overlapping levels follow painting order: a region occluded by a static higher-priority level is not refined further, whereas regions exposed by a moving level are.
- **Evanescent tails.** Past every boundary toward a lower index, fine cells continue for `global_dilation` (laterally) and `global_z_padding` (vertically) decay lengths `l = lambda / (2 pi sqrt(n_high^2 - n_low^2))`—3 decay lengths by default. Stopping fine cells right at a silicon core boundary (`0` decay lengths) introduces roughly an order of magnitude more effective-index error than extending them by 1–3 decay lengths; see [Adaptive meshing](adaptive-meshing.md#why-adaptive-meshing-wins-per-cell).
- **Narrow features and full-wafer layers.** Narrow parts and gaps in the starting geometry receive `feature_cells` (1.5) cells across, `Refine.regions` adds explicit fine boxes, and `min_cell` prevents thin films or narrow gaps from forcing an overly small time step. Conductive or metallic materials require explicit targets in `Refine.material_cells`. Because the grid is a tensor product, each axis coordinate takes the smallest target anywhere in its cross-sectional plane: coarsen a full-wafer substrate or cladding with `Refine.material_cells` (13 to 14 cells per wavelength are usually enough) to keep margins and PMLs coarse.
- **Edge placement.** `bf.axis_edges` places continuous real-valued cell edges so every primal cell and dual cell stays within its target, neighboring cells (and periodic wrap-around cells) differ by at most `growth` (1.25), and each PML boundary receives uniform cells covering `pml` µm (`boundaries` can set faces to `"periodic"`, `"pec"`, `"pmc"`, or `"bloch"` instead). The solver maximizes the smallest cell (which governs the CFL time step), keeps cell counts minimal, preserves mirror symmetry when the geometry is symmetric, aligns uniform runs where compatible (`align_fine`), and accepts any box size (or pass `quantum=0.001` to snap edges to multiples of a discrete step via `bf.lattice_edges`).
- **Mid-cell interface centering (`center_faces=True`).** Static horizontal material interfaces are placed at the exact midpoint of a z cell rather than on a cell edge, avoiding asymmetric dual-cell averaging errors and sub-cell boundary oscillations. Two faces a fractional number of cells apart may require slightly narrower cells (allowing the smallest z cell up to 20 % below the uncentered optimum); faces under a PML or moved by `Refine.z_padding` remain uncentered. `mesh.summary()` lists the centered faces.
- **Tracking vertices outside refined zones.** `stack.coarse_vertices(params, mesh)` counts vertices that move into cells wider than their target, per layer and axis. It runs inside `jax.jit` (useful as an auxiliary output of an optimization objective), and `mesh.summary()` reports it for the starting geometry. A level whose `z_padding` moves it into the box from beyond `level_margin` is rejected at mesh creation, since placing the device would drop it.
- **Accuracy vs cell count.** Across benchmarks including sloped waveguides, anisotropic lithium niobate bends, multi-level tapers, directional couplers, and curved SiN bends, `bf.mesh` matches uniform fine-grid accuracy with **1.7× to 3.6× fewer cells** and **1.7× to 5.0× shorter runtime** (and up to **6×–12× fewer cells** at tighter fine-grid tolerances; `ppw=14` offers a faster setting for initial design sweeps). See [Adaptive meshing](adaptive-meshing.md) for full convergence curves and tuning examples.

## Graded grids

When you need explicit control over per-axis edge placement rather than automatic meshing, `bf.graded_edges` builds custom 1D edge coordinates and `StackDevice.rectilinear_grid` turns them into an `fdtdx.core.grid.RectilinearGrid` over the box:

```python
half = 0.5 * stack.partial_real_shape[2] * 1e6  # the box's half height (um)
z = bf.graded_edges(
    stack.z_center - half, stack.z_center + half, 0.06,  # at most 60 nm cells
    fine=[(-0.05, 0.27, 0.02)],    # 20 nm around the 220 nm core
    snap=[0.0, 0.22],              # the core's faces on cell edges
    uniform_ends=(10, 10),         # 10 cells of 60 nm under each PML, whose thickness fdtdx counts in cells
)
config = stack.get_simulation_config(time=100e-15, grid=stack.rectilinear_grid(0.06, z_um=z))
source, placing = stack.create_mode_plane_source_at("o1", (0.5, 0.5), wave, grid=config)
```

- **Grading x, y, and z.** Pass `x_um` and `y_um` edges to `rectilinear_grid` from `graded_edges` (for example, fine across a narrow gap and coarse along straight leads). Every column of voxels is painted on its own lateral cell rectangle with Euclidean normals, sidewall slopes, and round corners; with `averaging="subpixel_full"`, junction voxels solve their cell problem on the voxel's own aspect ratio. Surface roughness on graded x or y is sampled on an even lattice at the finest spacing and interpolated to each column. Keep cell aspect ratios within a factor of 16 (placement emits a warning otherwise).
- **Time step and neighbor growth.** The smallest cell anywhere in the box sets the global CFL time step, so grading saves cells, field memory, and gradient checkpoint memory rather than time steps. `graded_edges` bounds neighboring cell ratios by `growth` (1.25) and drops any `snap` coordinate within half a cell of another edge with a warning to avoid slivers.
- **Port snapping and dtype consistency.** Always pass `grid=config` (or the `RectilinearGrid`) to `create_*_at` so port planes snap to local cell edges at their height rather than scaling from the domain origin. Always build the config with `stack.get_simulation_config(grid=...)` so grid edge coordinates match the device's `dtype`; placing a device with mismatched edge precision raises `ValueError`.
- **Subpixel accuracy and JIT caching.** Material fill fractions stay exact on non-uniform cells, and each Yee E component averages over the dual cells between neighboring cell centers along its staggered axes (Ex along y and z, Ey along x and z, Ez along x and y). On a fixed graded grid, `jax.jit` compiles once just as on a uniform grid, with per-step cost proportional to total cell count. Keep fine cells across the mode's evanescent tail rather than growing cell sizes immediately outside the core; see [Adaptive meshing](adaptive-meshing.md#why-adaptive-meshing-wins-per-cell).

## Precision

The whole simulation runs in the `StackDevice`'s `dtype`, `jnp.float32` (the default) or `jnp.float64`, and the device is the single place to configure it:

```python
stack = bf.StackDevice.from_layer_stack(
    component, layer_stack, materials, background="clad", zrange_layers=["core"], dtype=jnp.float64
)
mesh = bf.mesh(stack, 1.55)
config = stack.get_simulation_config(time=100e-15, grid=mesh.grid)
```

- `get_simulation_config` builds `fdtdx.SimulationConfig` in the device's dtype and forwards every other argument (`time`, `grid`, `backend`, `gradient_config`, ...); passing `dtype=` directly to it is an error. A manually constructed config must match the device's dtype or placement raises a `ValueError`.
- The `create_*_at` monitors match the device precision automatically: phasor, mode-overlap, and diffractive monitors use `complex64` or `complex128`, and field, energy, and Poynting-flux monitors use `float32` or `float64`, unless overridden with `dtype=`. fdtdx sources follow the config dtype.
- Importing `gdsfactory` enables JAX's 64-bit mode globally, and `bf.apply_stack` keeps fdtdx's PML coefficients in the device's dtype, so `float32` simulations run cleanly with x64 enabled. A `dtype=jnp.float64` device requires JAX's 64-bit mode to remain enabled at placement time.
- `float64` doubles array memory and is substantially slower on GPUs tuned for single precision. Note that in fdtdx 0.6.3, phasor monitor time windows and uniform-grid mode-overlap face weights still run in `float32`; on non-uniform `RectilinearGrid`s, `bf.ModeOverlapDetector` computes face weights in `float64` when configured for double-precision overlap monitors and `float32` otherwise.

## Troubleshooting

- **`Configured grid shape ... does not match simulation volume shape`**: Only occurs with a manual `UniformGrid` when a box side lies halfway between two cell counts (`bf.mesh` meshes any box size directly). Adjust `boundary_offset` by a fraction of a cell or change the uniform grid spacing.
- **A source or monitor is in the wrong place or clipped.** Print `monitor.grid_slice` as in the setup example above and compare with the box's cell count. Check the sign of `port_offset` (negative moves the plane *out* of the component) and `direction`.
- **The result looks a million times off.** A length was passed to fdtdx in µm rather than meters (`1e-6`).
