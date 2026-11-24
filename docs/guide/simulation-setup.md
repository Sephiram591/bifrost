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

Other fdtdx fields go in as keyword arguments. Boundaries, regular fdtdx devices, and other objects are added as usual. See [Plane ports, z centering and grid snapping](advanced.md#plane-ports-z-centering-and-grid-snapping) for virtual box planes, vertical centering, and non-uniform grid snapping.

## Running the Simulation

Build the adaptive grid and simulation configuration, add the PML boundaries from `mesh.boundary`, and run `fdtdx.place_objects` to fit every object onto the grid:

```python
--8<-- "examples/guide/simulation_setup.py:config"
```

- **The grid.** [`bf.mesh`][bifrost.meshing.mesh] builds an adaptive non-uniform grid (`mesh.grid`, 20 points per local wavelength by default) and 1 µm PML boundaries (`mesh.boundary`) over any box size; see [Adaptive meshing](advanced.md#adaptive-meshing).
- **The configuration and precision.** [`stack.get_simulation_config`][bifrost.stack.StackDevice.get_simulation_config] creates an `fdtdx.SimulationConfig` using the device's `dtype` and forwards time, grid, and solver settings; see [Precision](advanced.md#precision).

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

## Results

When `fdtdx.run_fdtd` returns, everything the monitors recorded is in `arrays.detector_states`, keyed by monitor name:

```python
--8<-- "examples/guide/results.py:read"
```

For the 300 fs straight waveguide run in `examples/guide/results.py`, both the mode-overlap transmission and the Poynting-flux ratio are `1.000`.

- **Mode monitors** ([`ModeOverlapDetector`][bifrost.stack.ModeOverlapDetector]) give the complex amplitude of a guided mode on their plane: `monitor.compute_overlap(states[name])`, one entry per wave character. The squared ratio of the output to the input amplitude is the **transmission**, normalized without a reference run because both monitors measure the same mode of the same guide.
- **Poynting-flux monitors** (`PoyntingFluxDetector`) record the power through their plane at every time step, in fdtdx's units. Summed over the run it is the energy that crossed, so the ratio of two of them cross-checks the transmission without assuming a mode (it also counts light in other modes).
- **Field monitors** (`PhasorDetector`) record the field's complex amplitude at the wave characters you give them, on the monitor's grid: `states[name]["phasor"][0, 0, :, :, :, 0]` is `(components, nx, ny)` for an xy plane one cell thick. `|E|²` is the sum of the squared magnitudes over the components. For the field at many *times*, a `FieldDetector` records the same plane at time steps you choose (the [bend example](../examples/waveguide-bend.md) records 40 of them, which the viewer plays as a movie).

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
