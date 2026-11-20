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

- **Units.** bifrost's lengths (`boundary_offset`, `extend_ports`, `port_offset`, `bf.mesh`) are in **µm**. fdtdx's are in
  **meters**: `SimulationConfig(time=...)` and `WaveCharacter(wavelength=...)` take seconds and meters (`* 1e-6`).
  The conversion is the usual source of a factor-of-a-million mistake.
- **The grid.** `bf.mesh(stack, 1.55)` builds an adaptive non-uniform grid (`mesh.grid`, 20 points per local wavelength
  by default) and 1 µm PML boundaries (`mesh.boundary`) over any box size; see [Automatic meshing](#automatic-meshing).
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
grid cells: (405, 95, 83)
source   (1, 67, 43) cells at [(67, 68), (14, 81), (20, 63)]
in       (1, 67, 43) cells at [(90, 91), (14, 81), (20, 63)]
out      (1, 67, 43) cells at [(315, 316), (14, 81), (20, 63)]
```

The box is 405 × 95 × 83 non-uniform cells. Each slice is `(start, stop)` cell indices along x, y and z: each plane is
one cell thick along its normal and 67 × 43 cells across, snapped via `grid=config` to the cell edges around the
0.5 µm guide plus 0.5 wavelengths per side in y and 0.3 wavelengths per side in z. The source sits 0.5 µm before the
input monitor at `x = 0`.

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
| `port` | a component port name, or a simulation-box plane port (`"xy"`, `"xz"`, `"yz"` in `bf.stack.PLANE_PORTS`) |
| `port_padding` | `(lateral, vertical)`: the plane extends this far past the port's width and the level's height on each side, in units of the mean wavelength of the object's wave characters (negative values shrink the plane). Field, energy and Poynting-flux monitors have none, so they take `wavelength_um` for the unit |
| `direction` | `"in"` (into the component, or `-normal` on a plane port) or `"out"`; sources default to `"in"`, monitors to `"out"` |
| `port_offset` | shift in µm along the injection direction; a negative value moves the plane out of the component, e.g. into a port extension |
| `level` | the level whose z span sets the plane's height (default: the levels that draw the port's layer, or volume mid-height for a plane port); pass `bf.stack.Z_CENTER_LEVEL` (`"z_center"`) to pin the z center to `stack.z_center` |
| `grid` | optional `SimulationConfig` or `RectilinearGrid` for plane methods (`create_point_dipole_source_at` is always one cell); snaps faces to the nearest grid edges along any non-uniform in-plane axis |

!!! note "Where to put a source and its monitors"
    - Keep source and monitor planes **outside the PML** and well clear of the box faces: the box's absorbing layer
      lies inside the box, so the port extension has to be longer than the PML plus the offsets.
    - Keep the plane **inside the box**. A plane larger than the box is clipped to it. With 1.55 µm light, a padding of
      3 wavelengths is far larger than a small box, so give the box room or reduce the padding.
    - Launch **just outside the port** (`port_offset=-0.5`) so the source sits in the straight extension, and put the input
      monitor on the port itself, as above.
    - Planes must be tall enough to hold the mode: the mode of a 220 nm silicon guide reaches about 0.5 µm into the
      cladding above and below, so a 1.1 µm tall plane is plenty.

The methods and the objects they make (`bf.*` wrappers live in `bifrost.stack` and subclass the matching `fdtdx` class):

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

`bf.ModePlaneSource` and `bf.ModeOverlapDetector` normalize against the simulation-wide `min_spacing**2` on non-uniform
`RectilinearGrid`s so launched power and overlap ratios do not shift with plane size or orientation, while leaving uniform
grids unchanged bit for bit.

Other fdtdx fields go in as keyword arguments. Boundaries, regular fdtdx devices and other objects are added as usual.

### Plane ports, z centering and grid snapping

- **Plane ports (`bf.stack.PLANE_PORTS = {"yz": 0, "xz": 1, "xy": 2}`):** passing `"xy"`, `"xz"`, or `"yz"` as `port`
  places the source, detector, or point dipole on the plane through the simulation box's center, spanning the box's
  in-plane extents (`width`, `height`) rather than a component port (they are virtual frames, not entries in `stack.ports`).
  For these plane ports, `"in"` faces the negative normal axis (`-normal`), `"out"` faces `+normal`, and positive
  `port_offset` shifts along `"in"` (`-normal`); `port_padding=(0, 0)` keeps the box extent unchanged.
- **Pinning z to box center (`bf.stack.Z_CENTER_LEVEL = "z_center"`):** passing `level=bf.stack.Z_CENTER_LEVEL` centers
  the plane at `stack.z_center` instead of the middle of the port's or named level's z span (for component ports, the
  height still comes from the port's level span; for plane ports, `level=None` also centers on `stack.z_center`, while a
  level name places the plane at that level's span middle).
- **Non-uniform grid snapping (`grid=`):** passing `grid=config` (a `SimulationConfig` or resolved `RectilinearGrid`) to
  any plane `create_*_at` method snaps the plane's faces to the nearest grid edges along any non-uniform in-plane axis
  and expresses the size in whole cells (`partial_grid_shape`) rather than physical length (`partial_real_shape`). This
  requires the `StackDevice` to fill the simulation volume from `get_simulation_volume(...)`. `create_point_dipole_source_at`
  always uses `partial_grid_shape=(1, 1, 1)` and does not take `grid=`.

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
  nearest edges at its own height. The device must fill the simulation volume (`get_simulation_volume`, any
  `boundary_offset`); a grid that spans more than its box is refused.
- **Build the config with `get_simulation_config`.** fdtdx scales the fields by the cell widths in the dtype of the
  grid's edges, so a float32 simulation needs float32 edges; `rectilinear_grid` makes them in the device's dtype and
  `get_simulation_config(grid=...)` converts those of a `RectilinearGrid` built otherwise. Placing a device with a
  config whose graded edges are in another dtype raises a `ValueError`.
- **Fill fractions stay exact.** A level's part of a cell is its overlap with the cell, and each field component
  averages over the cells between neighbouring cell centers (the dual cells) along the axes fdtdx staggers it on: Ex
  along y and z, Ey along x and z, Ez along x and y.
- **Optimizations compile once** on a fixed graded grid, like on a uniform one; a graded grid's step costs what its
  cells cost. In `benchmarks/rectilinear` a sloped silicon strip graded in y and z gives the uniform 20 nm grid's
  effective index within 7e-5 with 27 % of its cells, and the inverse-design bend graded to 80 nm away from the
  guide gives the uniform 40 nm grid's transmission within 4.4e-4 with 63 %. Keep the fine cells over the mode's
  evanescent tail, not only over the core: cells that grow too early (120 nm in the bend's silica) cost accuracy.

## Automatic meshing

`bf.mesh` builds the graded grid for you, once, before a run or an optimization: fine wherever a material is or may
come to be, coarse elsewhere. You tell it the wavelength band and where the geometry may move; it never limits the
parameters themselves.

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

- **Where it refines.** Every level asks for cells of at most `lambda / (n ppw)` of its material (`ppw=20`) wherever
  it is or may come to be: its shape, the bands within `dilation` of its moving layers (one value per layer, or one
  per vertex in the layout of `stack.init_params()["vertices/<layer>"]`, 0 where nothing moves; along an edge the band
  tapers between its vertices), and its faces moved by `z_padding`. Overlapping levels follow the painting order: a
  level that one which never moves hides asks for nothing there, one that a moving level may uncover does.
- **The evanescent tail.** Past each boundary toward a lower index the fine cells continue for `global_dilation`
  (laterally) and `global_z_padding` (above and below) decay lengths `l = lambda / (2 pi sqrt(n_high^2 - n_low^2))`,
  3 by default, also around geometry that never moves. On the silicon strip of `benchmarks/rectilinear` 1, 2 and 3
  of them give the uniform 20 nm grid's effective index within 5.8e-4, 1.4e-4 and 3.7e-5 (with 31 % of its cells at 3);
  without a tail the graded grid is as far off (3.3e-3) as a uniform grid of twice the cell.
- **Narrow parts and gaps** of the starting geometry get `feature_cells` (1.5) cells across, `Refine.regions` adds
  explicit sizes, and `min_cell` keeps a thin film or a narrow gap from setting a tiny time step. Conducting or
  metallic materials need their size in `Refine.material_cells`.
- **Each axis takes, at each coordinate, the smallest target anywhere in the plane through it** (the grid is a tensor
  product). A substrate or a cladding that spans every plane caps the lateral cells everywhere: coarsen it with
  `material_cells` (13 to 14 cells per wavelength are enough there).
- **The edges** are placed in continuous real coordinates (`bf.axis_edges`): no cell wider than its target, also over
  half its width on either side (so the dual cells, where the transverse E components are sampled, are within it too),
  neighbours (and periodic wrap-around cells) within `growth` (1.25); maximizing the smallest cell (which sets the time
  step), keeping cell counts minimal, and smoothing transitions across plateaus and ramps. Each PML face gets uniform
  cells covering `pml` um (`boundaries` may make faces `"periodic"`, `"pec"`, `"pmc"` or `"bloch"` instead);
  mirror-symmetric devices get mirror-symmetric grids, an axis with a single target gets exact uniform `L / N` cells,
  and the finest cells align to the lattice of their own width (`align_fine`) when compatible. Any box size is accepted
  (pass `quantum=0.001` to restrict edges to whole multiples of a discrete lattice via `bf.lattice_edges`).
- **Faces in the middle of their cells** (`center_faces=True`, the default). Every static level face where the material
  changes sits at the exact middle of a z cell, not on an edge: with subpixel averaging that leaves a markedly smaller
  error (a third, on the directional coupler of `benchmarks/rectilinear`). Two faces a fractional number of cells apart
  need some narrower cells, so the smallest z cell may come out up to 20 % below the largest possible; a face that still
  finds no room is left off with a warning, faces under a PML are left out, and faces moved by `z_padding` are left
  uncentered so vertical derivatives do not start on a dual-cell boundary. `mesh.summary()` lists the centered faces.
- **The geometry may leave its dilation.** `stack.coarse_vertices(params, mesh)` counts the vertices that sit in cells
  wider than their target, per layer and axis; it runs under `jax.jit` (return it as auxiliary output of the
  objective), and `mesh.summary()` reports it for the starting geometry. A level whose `z_padding` moves it into the
  box from beyond `level_margin` is refused, since placing the device would drop it.
- **How close it gets.** On the inverse-design bend with `dilation={"WG": 0.45}` (the half-widths' reach), 40 nm in the
  core and 80 nm in the air and the silica, `bf.mesh` gives the uniform 40 nm grid's transmission within 2e-5 with 65 %
  of its cells (the hand-set graded grid above: 4.4e-4); with `ppw=20` alone, within 1.4e-4 with 81 %. `ppw=14` (within
  2.6e-3 with 28 % of the cells) is twice as close as a uniform grid of half again the core's cells, for fast
  optimizations. See [Adaptive meshing](adaptive-meshing.md) for accuracy-per-cell benchmarks across six simulations.

## Precision

The whole simulation runs in the `StackDevice`'s `dtype`, `jnp.float32` (the default) or `jnp.float64`, and the device
is the only place to set it:

```python
stack = bf.StackDevice.from_layer_stack(
    component, layer_stack, materials, background="clad", zrange_layers=["core"], dtype=jnp.float64
)
mesh = bf.mesh(stack, 1.55)
config = stack.get_simulation_config(time=100e-15, grid=mesh.grid)
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
  computes phasor monitor time windows and uniform-grid mode-overlap face weights in float32 (on non-uniform
  `RectilinearGrid`s, `bf.ModeOverlapDetector.place_on_grid` computes its face weights in `placed._config.dtype`:
  `float64` in double precision when overlap monitors are `complex128`, and `float32` otherwise), so a float64 run
  has float64 fields and permittivity, not every intermediate in double precision.

## Troubleshooting

- **`Configured grid shape ... does not match simulation volume shape`**: only applies to a manual `UniformGrid` when
  a side of the box is halfway between two cell counts (default `bf.mesh` accepts any box size). Move `boundary_offset`
  by a fraction of a cell, or change the grid spacing.
- **A source or monitor is in the wrong place or clipped.** Print `monitor.grid_slice` as above and compare with the
  box's cell count. Check the sign of `port_offset` (negative moves the
  plane *out* of the component) and `direction`.
- **The result looks a million times off.** A length went into fdtdx in µm. fdtdx wants meters.
