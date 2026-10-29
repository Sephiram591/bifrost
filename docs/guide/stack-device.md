# Layer stacks and StackDevice

`bifrost.stack.StackDevice` turns a component and its
[`LayerStack`](https://gdsfactory.github.io/gdsfactory/notebooks/03_layer_stack/)
into a single [fdtdx](https://github.com/ymahlau/fdtdx) object. Every
`LayerLevel` of the stack is voxelized into one multi-material region, and the
result is differentiable with respect to:

- the polygon vertices of every layer, and therefore the parameters of the cell
  function that drew them;
- the thickness and sidewall angle of every level.

The device also builds a simulation volume that fits it and places sources and
detectors directly on the component's ports. You don't export a GDS, and you
don't re-describe the geometry with fdtdx primitives.

Install the integration with the `fdtdx` extra (requires `fdtdx >= 0.6.3`):

```bash
pip install "gdsfactory[fdtdx] @ git+https://github.com/Sephiram591/gdsfactoryx.git"
```

## Workflow

1. Build the device with `StackDevice.from_layer_stack(component, layer_stack, materials, background, ...)`.
2. Get the matching simulation volume with `stack.get_simulation_volume()`.
3. Put sources and detectors on ports with the `stack.create_<object>_at(port, ...)` methods.
4. Place everything with `fdtdx.place_objects` and merge `init_stack_params(objects)` into the params.
5. Apply the device with `apply_stack` instead of `fdtdx.apply_params`, then run `fdtdx.run_fdtd`.

### 1. Build the device

```python
import fdtdx
import bifrost as bf
import gdsfactory as gf
from gdsfactory.technology import LayerLevel, LayerStack

gf.gpdk.PDK.activate()

layer_stack = LayerStack(
    layers={
        "box": LayerLevel(layer="WAFER", zmin=-3.0, thickness=3.0, material="sio2", mesh_order=9),
        "core": LayerLevel(layer=(1, 0), zmin=0.0, thickness=0.22, material="si", mesh_order=1),
    }
)
materials = {
    "air": fdtdx.Material(),
    "si": fdtdx.Material(permittivity=3.48**2),
    "sio2": fdtdx.Material(permittivity=1.444**2),
}

stack = bf.StackDevice.from_layer_stack(
    gf.components.mmi1x2(),
    layer_stack,
    materials,
    background="air",
    included_layers=["core"],                 # z range of the box: the core ...
    extend_ports=(("o1", 1.0), ("o2", 1.0), ("o3", 1.0)),  # ... plus straight leads out of the ports ...
    boundary_offset=(0, 0, 1, -1, 1, -1),     # ... grown by 1 um in y and z
    name="stack",
)
```

- `materials` maps every `LayerLevel.material` that is drawn, plus `background`,
  to an `fdtdx.Material`. The background fills everything no level covers.
- The box starts as the xy bounding box of the polygons, including the port
  extensions, and the z range of `included_layers`.
  `boundary_offset = (x+, x-, y+, y-, z+, z-)` moves its faces in µm without
  changing the geometry. Every level that overlaps the box is drawn, including
  levels outside `included_layers` such as the `box` oxide above.
- `extend_ports` continues each listed port's cross-section straight out of the
  component. Every stack layer that reaches the port face, over a stretch
  overlapping the port's width, is extended with that stretch's width, so a rib
  waveguide keeps its slab and an etch over the port continues. Use it to run
  waveguides through the PML. To extend only some layers, give them as a third
  element: `("o1", 1.0, [LAYER.WG])`. The extensions are polygons like any
  other: they get the level's thickness, sidewall angle and bias, and they
  follow the port and the polygon edges when the component's parameters change.
- Layers in `fill_layers` (default `WAFER`) cover the whole plane.

The voxelization supports most of `LayerLevel`:

- logical and derived layers;
- `mesh_order` priorities, where the lower value wins;
- `sidewall_angle`, `width_to_z`, `bias` and `z_to_bias`;
- background levels.

### 2. Volume, sources and detectors

```python
volume, constraints = stack.get_simulation_volume()
objects = [volume, stack]
wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
for obj, c in (
    stack.create_mode_plane_source_at("o1", (1.0, 1.0), wave_character=wave, port_offset=-0.5),
    stack.create_mode_overlap_detector_at("o2", (1.0, 1.0), wave_characters=(wave,), name="out"),
):
    objects.append(obj)
    constraints += c
```

Each `create_*_at` method centers a plane on a port and orients the plane
normal to the port. It returns the object together with its position
constraints. The common arguments are:

| Argument | Meaning |
| --- | --- |
| `port` | port name |
| `port_padding` | `(lateral, vertical)`: the plane extends this far past the port's width and the level's height on each side, in units of the mean wavelength of the object's wave characters (negative values shrink the plane). Field, energy and Poynting-flux detectors have none, so they take `wavelength_um` for the unit |
| `direction` | `"in"` (into the component) or `"out"`; sources default to `"in"`, detectors to `"out"` |
| `port_offset` | shift in µm along the injection direction; negative values move the plane out of the component, e.g. into a port extension |
| `level` | level whose z span sets the plane's height (default: the levels drawing the port's layer) |

Available objects:

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

Any other fdtdx fields can be passed as keyword arguments. Add boundaries,
regular fdtdx devices and other objects as usual.

### 3. Place, apply and run

```python
import jax
import jax.numpy as jnp

config = fdtdx.SimulationConfig(
    time=200e-15, grid=fdtdx.core.grid.UniformGrid(spacing=20e-9), dtype=jnp.float32
)
pml = fdtdx.BoundaryConfig.from_uniform_bound(thickness=25)
boundaries, c = fdtdx.boundary_objects_from_config(pml, volume)
objects += boundaries.values()
constraints += c

key = jax.random.PRNGKey(0)
objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
params = {**params, **bf.init_stack_params(objects)}
arrays, objects, _ = bf.apply_stack(arrays, objects, params, key)
_, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key)
```

fdtdx's own `Device` objects only mix two materials, so `fdtdx.apply_params`
doesn't know about a `StackDevice`. `apply_stack` writes every stack into the
arrays first and then calls `fdtdx.apply_params`, so regular devices and sources
are still updated, and devices win where they overlap a stack.
