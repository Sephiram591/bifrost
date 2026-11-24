# Simulation Setup

A bifrost simulation is an ordinary [fdtdx](https://github.com/ymahlau/fdtdx) simulation in which the device, the sources, and the monitors come from [`StackDevice`][bifrost.stack.StackDevice]. Because `StackDevice` knows where its ports are, you attach sources and monitors to ports instead of computing box coordinates by hand. The setup below builds and places a straight-waveguide simulation without running the solver, finishing in seconds on a CPU.

## StackDevice Setup

Create the [`StackDevice`][bifrost.stack.StackDevice] from a gdsfactory component, its `LayerStack`, and its `materials` dictionary (see [Layer stacks and materials](layer-stacks.md)):

```python
--8<-- "examples/guide/simulation_setup.py:device"
```

- **Box and extensions.** `zrange_layers` determines the vertical span of the simulation box before `boundary_offset` adds cladding margins around the GDS cell, while `extend_ports` continues the waveguide straight out through the absorbing boundaries. `background="clad"` fills every voxel not covered by a stack level.
- **Units.** bifrost's lengths (`boundary_offset`, `extend_ports`, `port_offset`, [`bf.mesh`][bifrost.meshing.mesh]) are in **µm**. fdtdx expects SI units (**meters** and **seconds**), so wavelengths and durations passed to fdtdx are in meters (`* 1e-6`) and seconds.

## Placing Sources/Detectors

[`stack.get_simulation_volume`][bifrost.stack.StackDevice.get_simulation_volume] returns the simulation box and its placement constraints. Define the center wavelength and pulse profile, then use the port factory methods on [`StackDevice`][bifrost.stack.StackDevice] to return each fdtdx source or detector together with the constraints that place it on a port, centered and normal to the port's direction (here `config` comes from [Running the Simulation](#running-the-simulation) below):

```python
--8<-- "examples/guide/simulation_setup.py:pulse"

--8<-- "examples/guide/simulation_setup.py:ports"
```

- **The pulse.** A Gaussian pulse with a spectral width of 5 % of the center frequency ramps up smoothly and lets one run cover a band (`WaveCharacter` takes wavelengths in meters, `* 1e-6`).

Add each object and its constraints to your lists:

```python
for obj, placing in (stack.create_mode_plane_source_at("o1", (0.5, 0.3), ...),):
    objects.append(obj)
    constraints += placing
```

`StackDevice` provides port-attached factories for mode, Gaussian, uniform, constant-amplitude plane, and point-dipole sources (such as [`create_mode_plane_source_at`][bifrost.stack.StackDevice.create_mode_plane_source_at]), as well as mode-overlap, Poynting-flux, phasor, field, energy, and diffractive monitors (such as [`create_mode_overlap_detector_at`][bifrost.stack.StackDevice.create_mode_overlap_detector_at]). See [`StackDevice`][bifrost.stack.StackDevice] in the API reference for the full list of `create_*_at` methods, their shared port, padding, direction, offset, level, and grid-snapping arguments, and forwarded `fdtdx` keyword arguments.

!!! note "Where to put a source and its monitors"
    - Keep source and monitor planes **outside the PML** and well clear of the box faces: the box's absorbing layer lies inside the box, so the port extension has to be longer than the PML plus the offsets.
    - Keep the plane **inside the box**. A plane larger than the box is clipped to it. With 1.55 µm light, a padding of 3 wavelengths is larger than a small box, so give the box room or reduce the padding.
    - Launch **just outside the port** (`port_offset=-0.5`) so the source sits in the straight extension, and put the input monitor on the port itself, as above.
    - Planes must be tall enough to hold the mode: the mode of a 220 nm silicon guide reaches about 0.5 µm into the cladding above and below, so a 1.1 µm tall plane is sufficient.

On non-uniform grids, [`bf.ModePlaneSource`][bifrost.stack.ModePlaneSource] and [`bf.ModeOverlapDetector`][bifrost.stack.ModeOverlapDetector] (subclasses of `fdtdx`'s classes in [`bifrost.stack`][bifrost.stack]) normalize mode power against a common reference area (`min_spacing**2`) so launched power and overlap ratios do not depend on port-plane padding, width, or orientation; on uniform grids they behave identically to `fdtdx`'s classes.

Other fdtdx fields go in as keyword arguments. Boundaries, regular fdtdx devices, and other objects are added as usual.

### Plane ports, z centering and grid snapping

- **Plane ports (`PLANE_PORTS` in [`bifrost.stack`][bifrost.stack]):** Passing `"xy"`, `"xz"`, or `"yz"` as `port` places the source, detector, or point dipole on the plane through the simulation box's center, spanning the box's in-plane extents rather than a component port (they are virtual frames, not entries in `stack.ports`). For plane ports, `"in"` faces `-normal`, `"out"` faces `+normal`, and positive `port_offset` shifts along `"in"`; `port_padding=(0, 0)` keeps the box extent unchanged.
- **Pinning z to box center (`Z_CENTER_LEVEL` in [`bifrost.stack`][bifrost.stack]):** Passing `level=bf.stack.Z_CENTER_LEVEL` centers the plane at `stack.z_center` instead of the middle of the port's or named level's z span. For component ports, the height still comes from the port's level span; for plane ports, `level=None` also centers on `stack.z_center`, while a level name places the plane at that level's span middle.
- **Non-uniform grid snapping (`grid=`):** Passing `grid=config` (a `SimulationConfig` or resolved `RectilinearGrid`) to any plane factory on [`StackDevice`][bifrost.stack.StackDevice] snaps the plane's faces to the nearest grid edges along any non-uniform in-plane axis and expresses the size in whole cells rather than physical length. This requires the `StackDevice` to fill the simulation volume from [`get_simulation_volume`][bifrost.stack.StackDevice.get_simulation_volume]. [`create_point_dipole_source_at`][bifrost.stack.StackDevice.create_point_dipole_source_at] always occupies one cell and does not take `grid=`.

## Running the Simulation

Build the adaptive grid and simulation configuration, add the PML boundaries from `mesh.boundary`, and run `fdtdx.place_objects` to fit every object onto the grid:

```python
--8<-- "examples/guide/simulation_setup.py:config"
```

- **The grid.** [`bf.mesh`][bifrost.meshing.mesh] builds an adaptive non-uniform grid (`mesh.grid`, 20 points per local wavelength by default) and 1 µm PML boundaries (`mesh.boundary`) over any box size; see [Automatic meshing](#automatic-meshing).
- **The configuration and precision.** [`stack.get_simulation_config`][bifrost.stack.StackDevice.get_simulation_config] creates an `fdtdx.SimulationConfig` using the device's `dtype` and forwards time, grid, and solver settings; see [Precision](#precision).

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

After `fdtdx.place_objects`, [`bf.apply_stack`][bifrost.stack.apply_stack] writes the stack into the permittivity arrays, in place of `fdtdx.apply_params`, before calling `fdtdx.run_fdtd` (see [Waveguide bend](../examples/waveguide-bend.md) for a full end-to-end run):

```python
objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
arrays, objects, _ = bf.apply_stack(arrays, objects, {**params, **bf.init_stack_params(objects)}, key)
_, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key)
```

fdtdx's own `Device` objects only mix two materials, so `fdtdx.apply_params` does not update a `StackDevice`. `apply_stack` writes every stack into the arrays first and then calls `fdtdx.apply_params`, so regular devices and sources are still updated, and devices win where they overlap a stack. [`bf.init_stack_params`][bifrost.stack.init_stack_params] supplies the stack's own parameters (vertices, heights, thicknesses) for differentiation; see [Inverse design](inverse-design.md).

### Automatic meshing

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

### Precision

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

## Results

When `fdtdx.run_fdtd` returns, everything the monitors recorded is in `arrays.detector_states`, keyed by monitor name:

```python
--8<-- "examples/guide/results.py:read"
```

For the 300 fs straight waveguide run in `examples/guide/results.py`, both the mode-overlap transmission and the Poynting-flux ratio are `1.000`.

- **Mode monitors** ([`ModeOverlapDetector`][bifrost.stack.ModeOverlapDetector]) give the complex amplitude of a guided mode on their plane: `monitor.compute_overlap(states[name])`, one entry per wave character. The squared ratio of the output to the input amplitude is the **transmission**, normalized without a reference run because both monitors measure the same mode of the same guide.
- **Poynting-flux monitors** (`PoyntingFluxDetector`) record the power through their plane at every time step, in fdtdx's units. Summed over the run it is the energy that crossed, so the ratio of two of them cross-checks the transmission without assuming a mode (it also counts light in other modes).
- **Field monitors** (`PhasorDetector`) record the field's complex amplitude at the wave characters you give them, on the monitor's grid: `states[name]["phasor"][0, 0, :, :, :, 0]` is `(components, nx, ny)` for an xy plane one cell thick. `|E|²` is the sum of the squared magnitudes over the components. For the field at many *times*, a `FieldDetector` records the same plane at time steps you choose (the [bend example](../examples/waveguide-bend.md) records 40 of them, which the viewer plays as a movie).

### Choosing a monitor

| You want | Use | Read it with |
| --- | --- | --- |
| the power in a guided mode (transmission) | [`create_mode_overlap_detector_at`][bifrost.stack.StackDevice.create_mode_overlap_detector_at] | `compute_overlap(state)` |
| the power through a plane, any mode | [`create_poynting_flux_detector_at`][bifrost.stack.StackDevice.create_poynting_flux_detector_at] | `state["poynting_flux"]` |
| the field's amplitude and phase at a wavelength | `PhasorDetector` | `state["phasor"]` |
| the field at a sequence of times (a movie) | `FieldDetector` with an `OnOffSwitch(interval=...)` | the recorded frames |
| the energy in a region | [`create_energy_detector_at`][bifrost.stack.StackDevice.create_energy_detector_at] | `state["energy"]` |

### Open the result in the viewer

```python
--8<-- "examples/guide/results.py:viewer"
```

[`viewer.load_result`][bifrost.viewer.widget.Viewer.load_result] takes what fdtdx hands out: the placed objects, the arrays and the config. The **Result** tab then has one read-only entry per monitor, with the field of every plane-shaped monitor drawn on its plane and a frame slider for monitors that recorded several frames; the **Sources / Detectors** tab lists the same objects. In a notebook, end the cell with `viewer`. More on the [Widget Demo](../live-demo.md) page.

## Troubleshooting

- **`Configured grid shape ... does not match simulation volume shape`**: Only occurs with a manual `UniformGrid` when a box side lies halfway between two cell counts (`bf.mesh` meshes any box size directly). Adjust `boundary_offset` by a fraction of a cell or change the uniform grid spacing.
- **A source or monitor is in the wrong place or clipped.** Print `monitor.grid_slice` as in the placement example above and compare with the box's cell count. Check the sign of `port_offset` (negative moves the plane *out* of the component) and `direction`.
- **The result looks a million times off.** A length was passed to fdtdx in µm rather than meters (`1e-6`).
- **The transmission is far from 1 for a straight guide.** The run was too short: the output monitor must see the whole pulse. Increase `time` so the pulse finishes crossing the device (for example, shortening the run to 200 fs drops the measured transmission in `examples/guide/results.py` to 0.87).
- **A monitor reports nothing, or the viewer shows no field for it.** The viewer draws monitors that record a plane (one cell thick along its normal), and not ones that sum over their volume (`reduce_volume=True`).
- **The numbers are tiny or huge.** fdtdx's flux and amplitudes are in its own units; compare ratios, not absolute values.
