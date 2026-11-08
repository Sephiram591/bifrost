# Simulation setup and ports

A bifrost simulation is an ordinary [fdtdx](https://github.com/ymahlau/fdtdx) simulation in which the device, the
sources and the monitors come from the `StackDevice`: it knows where its ports are, so you ask for a source or monitor
*on a port* instead of computing coordinates. The example below sets one up and places it without running the solve;
it finishes in seconds on a CPU.

## Build, place and inspect

The device and the numerical settings:

```python
--8<-- "examples/guide/simulation_setup.py:device"

--8<-- "examples/guide/simulation_setup.py:settings"
```

- **Units.** bifrost's lengths (`boundary_offset`, `extend_ports`, `port_offset`) are in **µm**. fdtdx's are in
  **meters**: `SimulationConfig(time=...)` is in seconds, `UniformGrid(spacing=...)` and `WaveCharacter(wavelength=...)`
  are in meters. The `* 1e-6` conversions are the usual source of a factor-of-a-million mistake.
- **The grid.** `UniformGrid(spacing=...)` is the cell edge. fdtdx rounds the box to a whole number of cells in every
  direction, so keep its sides on the grid (see [Layer stacks and materials](layer-stacks.md)).
- **The pulse.** A Gaussian pulse with a spectral width of 5 % of the center frequency ramps up smoothly and lets one run
  cover a band; the monitors below record the center wavelength.
- **The precision.** `stack.get_simulation_config(...)` is `fdtdx.SimulationConfig` with the device's `dtype` filled in;
  see [Precision](#precision).

The box, the source and the monitors, each placed on a port:

```python
--8<-- "examples/guide/simulation_setup.py:ports"
```

The absorbing boundary and the placement:

```python
--8<-- "examples/guide/simulation_setup.py:place"
```

Output:

```text
grid cells: (180, 74, 68)
source   (1, 41, 23) cells at [(30, 31), (16, 57), (22, 45)]
in       (1, 41, 23) cells at [(40, 41), (16, 57), (22, 45)]
out      (1, 41, 23) cells at [(140, 141), (16, 57), (22, 45)]
```

The box is 180 × 74 × 68 cells of 50 nm. Each slice is `(start, stop)` cell indices along x, y and z: each plane is
one cell thick along its normal and 41 × 23 cells (2.05 × 1.15 µm) across, the 0.5 µm guide plus 0.5 wavelengths per
side, and the 0.22 µm core plus 0.3 wavelengths per side. The source is 10 cells (0.5 µm) before the input monitor,
which sits on the port at x = 0.

## Sources and monitors on ports

`stack.get_simulation_volume()` returns the simulation box and its placement constraints. Each
`stack.create_<object>_at(port, port_padding, ...)` method returns an fdtdx object together with the constraints that
place it on that port, centered and normal to the port's direction. Add both to your lists:

```python
for obj, placing in (stack.create_mode_plane_source_at("o1", (0.5, 0.3), ...),):
    objects.append(obj)
    constraints += placing
```

The arguments every method shares:

| Argument | Meaning |
| --- | --- |
| `port` | the port's name |
| `port_padding` | `(lateral, vertical)`: the plane extends this far past the port's width and the level's height on each side, in units of the mean wavelength of the object's wave characters (negative values shrink the plane). Field, energy and Poynting-flux monitors have none, so they take `wavelength_um` for the unit |
| `direction` | `"in"` (into the component) or `"out"`; sources default to `"in"`, monitors to `"out"` |
| `port_offset` | shift in µm along the injection direction; a negative value moves the plane out of the component, e.g. into a port extension |
| `level` | the level whose z span sets the plane's height (default: the levels that draw the port's layer) |

!!! note "Where to put a source and its monitors"
    - Keep source and monitor planes **outside the PML** and well clear of the box faces: the box's absorbing layer
      lies inside the box, so the port extension has to be longer than the PML plus the offsets.
    - Keep the plane **inside the box**. A plane larger than the box is clipped to it. With 1.55 µm light, a padding of
      3 wavelengths is far larger than a small box, so give the box room or reduce the padding.
    - Launch **just outside the port** (`port_offset=-0.5`) so the source sits in the straight extension, and put the input
      monitor on the port itself, as above.
    - Planes must be tall enough to hold the mode: the mode of a 220 nm silicon guide reaches about 0.5 µm into the
      cladding above and below, so a 1.1 µm tall plane is plenty.

The methods and the fdtdx objects they make:

| Method | fdtdx object |
| --- | --- |
| `create_mode_plane_source_at` | `ModePlaneSource` |
| `create_gaussian_plane_source_at` | `GaussianPlaneSource` |
| `create_uniform_plane_source_at` | `UniformPlaneSource` |
| `create_hard_constant_amplitude_plane_source_at` | `HardConstantAmplitudePlanceSource` |
| `create_point_dipole_source_at` | `PointDipoleSource` |
| `create_mode_overlap_detector_at` | `ModeOverlapDetector` |
| `create_poynting_flux_detector_at` | `PoyntingFluxDetector` |
| `create_phasor_detector_at` | `PhasorDetector` |
| `create_field_detector_at` | `FieldDetector` |
| `create_energy_detector_at` | `EnergyDetector` |
| `create_diffractive_detector_at` | `DiffractiveDetector` |

Other fdtdx fields go in as keyword arguments. Boundaries, regular fdtdx devices and other objects are added as usual.

## Place, apply and run

After `fdtdx.place_objects`, `bf.apply_stack` writes the stack into the permittivity arrays, in place of
`fdtdx.apply_params` (the [waveguide bend](../examples/waveguide-bend.md) runs this end to end):

```python
objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
arrays, objects, _ = bf.apply_stack(arrays, objects, {**params, **bf.init_stack_params(objects)}, key)
_, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key)
```

fdtdx's own `Device` objects only mix two materials, so `fdtdx.apply_params` doesn't know about a `StackDevice`.
`apply_stack` writes every stack into the arrays first and then calls `fdtdx.apply_params`, so regular devices and
sources are still updated, and devices win where they overlap a stack. `bf.init_stack_params(objects)` supplies the
stack's own parameters (vertices, heights, thicknesses), which is what you differentiate; see
[Gradients](gradients.md).

## Graded grids

The cells can have different sizes along every axis: fine cells where the layers and the edges are, coarse ones in the
cladding around them. `bf.graded_edges` places the edges of one axis and `StackDevice.rectilinear_grid` turns them into
fdtdx's grid over the box, here graded along z:

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

- **x and y may be graded too**, with any averaging: pass `x_um` / `y_um` edges to `rectilinear_grid` (from
  `graded_edges`, e.g. fine across a gap and coarse along the waveguides, or uniform ones of a size other than
  `spacing_um`). Every column of voxels is painted on its own lateral cell, a rectangle of the cell's aspect in a frame
  scaled the same way along x and y, so normals, sidewall slopes and round corners stay Euclidean; the corner model and
  `xy_subsamples` work per column, and with `"subpixel_full"` a junction voxel's closure solves its cell problem on the
  voxel's own box. Rough surfaces on graded x or y are drawn on an even lattice at the finest cell and interpolated to
  the columns, so they come out slightly smoothed in the coarse ones. Keep the cells within a factor 16 of square (a
  placement warning says when they are not).
- **The smallest cell sets the time step** of the whole simulation. Grading saves cells and memory (also the gradient's
  checkpoints), not time steps. `graded_edges` keeps neighbouring cells within a factor `growth` (1.25) and does not
  make slivers: a `snap` coordinate within half a cell of another edge is dropped, with a warning.
- **Pass `grid=` to the `create_*_at` methods** (the config or the grid). fdtdx converts a plane's physical size to
  cells from the bottom of the domain, which is wrong where the cells differ; with `grid=` the plane's faces snap to the
  nearest edges at its own height.
- **Fill fractions stay exact.** A level's part of a cell is its overlap with the cell, and each field component
  averages over the cells between neighbouring cell centers (the dual cells) along the axes fdtdx staggers it on: Ex
  along y and z, Ey along x and z, Ez along x and y.

## Precision

The whole simulation runs in the `StackDevice`'s `dtype`, `jnp.float32` (the default) or `jnp.float64`, and the device
is the only place to set it:

```python
stack = bf.StackDevice.from_layer_stack(
    component, layer_stack, materials, background="clad", zrange_layers=["core"], dtype=jnp.float64
)
config = stack.get_simulation_config(time=100e-15, grid=fdtdx.core.grid.UniformGrid(spacing=50e-9))
```

- `get_simulation_config` builds the `fdtdx.SimulationConfig` with the device's dtype and passes every other argument
  (`time`, `grid`, `backend`, `gradient_config`, ...) on. Passing `dtype=` to it is an error. A config you build yourself
  must use the device's dtype as well, or placing the device raises a `ValueError`.
- The `create_*_at` monitors follow the device: phasor, mode-overlap and diffractive monitors are complex64 or complex128,
  field, energy and Poynting-flux monitors float32 or float64, unless you pass `dtype=` yourself. fdtdx's sources already
  follow the config.
- Nothing needs switching around the run. Importing gdsfactory turns JAX's 64-bit mode on, and `bf.apply_stack` keeps
  fdtdx's absorbing-boundary (PML) coefficients in the device's dtype, so `fdtdx.run_fdtd` works in either precision.
  Scripts written before this wrapped the run in `jax.enable_x64(False)`; that still works for float32, but a float64
  device refuses to be placed while the 64-bit mode is off.
- float64 doubles the memory of every array and is slow on GPUs built for single precision. fdtdx 0.6.3 still
  computes a few auxiliary quantities in float32 (the time window of phasor monitors and the face weights of mode-overlap
  monitors), so a float64 run has float64 fields and permittivity, not every intermediate in double precision.

## Troubleshooting

- **`Configured grid shape ... does not match simulation volume shape`**: a side of the box is exactly halfway between
  two cell counts. Move `boundary_offset` by a fraction of a cell, or change the grid spacing.
- **A source or monitor is in the wrong place or clipped.** Print `monitor.grid_slice` as above and compare with the
  box's cell count. Check the sign of `port_offset` (negative moves the
  plane *out* of the component) and `direction`.
- **The result looks a million times off.** A length went into fdtdx in µm. fdtdx wants meters.
