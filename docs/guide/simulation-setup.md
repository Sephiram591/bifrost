# Simulation setup and ports

A bifrost simulation is an ordinary [fdtdx](https://github.com/ymahlau/fdtdx) simulation in which the device, the
sources and the monitors come from the `StackDevice`: it knows where its ports are, so you ask for a source or monitor
*on a port* instead of computing coordinates. This page sets one up and places it, without running the solve. The
script is complete and finishes in seconds on a CPU.

## Build, place and inspect

First the device and the numerical settings.

```python
--8<-- "examples/guide/simulation_setup.py:device"

--8<-- "examples/guide/simulation_setup.py:settings"
```

- **Units.** bifrost's lengths (`boundary_offset`, `extend_ports`, `port_offset`) are in **µm**. fdtdx's are in
  **meters**: `SimulationConfig(time=...)` is in seconds, `UniformGrid(spacing=...)` and `WaveCharacter(wavelength=...)`
  are in meters. The `* 1e-6` conversions are the usual place for a factor-of-a-million mistake.
- **The grid.** `UniformGrid(spacing=...)` is the cell edge. The box has to be a whole number of cells in every
  direction (see [Layer stacks and materials](layer-stacks.md)).
- **The pulse.** A Gaussian pulse with a spectral width of 5 % of the center frequency ramps up smoothly and lets one run
  cover a band; the monitors below record the center wavelength.

Now the box, the source and the monitors, each placed on a port:

```python
--8<-- "examples/guide/simulation_setup.py:ports"
```

and finally the absorbing boundary and the placement:

```python
--8<-- "examples/guide/simulation_setup.py:place"
```

Run on a CPU, it prints the placed grid and each object's position:

```text
grid cells: (180, 74, 68)
source   (1, 41, 23) cells at [(30, 31), (16, 57), (22, 45)]
in       (1, 41, 23) cells at [(40, 41), (16, 57), (22, 45)]
out      (1, 41, 23) cells at [(140, 141), (16, 57), (22, 45)]
```

The box is 180 × 74 × 68 cells of 50 nm. Each slice is `(start, stop)` cell indices along x, y and z, so each plane is
one cell thick along its normal and 41 × 23 cells (2.05 × 1.15 µm) across: the 0.5 µm guide plus 0.5 wavelengths per
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

The objects available:

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

Any other fdtdx fields can be passed as keyword arguments. Add boundaries, regular fdtdx devices and other objects as
usual.

## Place, apply and run

After `fdtdx.place_objects`, the stack is written into the permittivity arrays by `bf.apply_stack` instead of
`fdtdx.apply_params`:

```python
objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
arrays, objects, _ = bf.apply_stack(arrays, objects, {**params, **bf.init_stack_params(objects)}, key)
with jax.enable_x64(False):
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key)
```

(That is part of the complete [waveguide bend](../examples/waveguide-bend.md) example, which runs it.)

fdtdx's own `Device` objects only mix two materials, so `fdtdx.apply_params` doesn't know about a `StackDevice`.
`apply_stack` writes every stack into the arrays first and then calls `fdtdx.apply_params`, so regular devices and
sources are still updated, and devices win where they overlap a stack. `bf.init_stack_params(objects)` supplies the
stack's own parameters (vertices, heights, thicknesses), which is what you differentiate; see
[Gradients](gradients.md).

The 64-bit switch around `run_fdtd` is needed because importing gdsfactory turns on JAX's 64-bit mode while fdtdx's
solver loop is written for 32 bits.

## Troubleshooting

- **`Configured grid shape ... does not match simulation volume shape`**: the box is not a whole number of cells.
  Change `boundary_offset` or the grid spacing so each side divides evenly.
- **A source or monitor is in the wrong place or clipped.** Print `monitor.grid_slice` as above; each is
  `(start, stop)` cell indices. Compare with the box's cell count. Check the sign of `port_offset` (negative moves the
  plane *out* of the component) and `direction`.
- **The result looks a million times off.** A length went into fdtdx in µm. fdtdx wants meters.
