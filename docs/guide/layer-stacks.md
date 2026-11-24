# Layer stacks and materials

Vertical structure and optical properties in bifrost are described by two objects:

- a gdsfactory [`LayerStack`](https://gdsfactory.github.io/gdsfactory/notebooks/03_layer_stack/), which maps 2D layout layers (or boolean expressions of layers) to 3D `LayerLevel`s with heights, thicknesses, sidewall profiles, and material names;
- a `dict[str, fdtdx.Material]` mapping every material name referenced by the stack—plus the background medium—to an [`fdtdx.Material`](https://github.com/ymahlau/fdtdx).

Once defined, a component, its `LayerStack`, and its `materials` dictionary are passed to [`bf.StackDevice`][bifrost.stack.StackDevice] in [Simulation setup and ports](simulation-setup.md).

## A silicon waveguide in silica

The simplest stack has one level and two isotropic materials—a 220 nm silicon strip core embedded in uniform silica cladding:

```python
--8<-- "examples/guide/layer_stacks.py:imports"

--8<-- "examples/guide/layer_stacks.py:bare"
```

- **`layer_stack`** maps level names to `LayerLevel` definitions:
  - `layer`: the GDS `(layer, datatype)` tuple, PDK layer name, or `LogicalLayer` that defines the level's 2D footprint;
  - `zmin` and `thickness`: vertical start coordinate and thickness in **µm** (`z` spans `[zmin, zmin + thickness]`);
  - `material`: string key looked up in `materials`;
  - `mesh_order`: integer priority where levels overlap in 3D (**lower `mesh_order` wins**).
- **`materials`** maps each material name to an `fdtdx.Material`. Lossless isotropic dielectrics set `permittivity=n**2` (`fdtdx.Material()` defaults to `permittivity=1.0` for vacuum/air). Include the background medium (`"clad"` here) alongside the level materials so regions outside every level have a defined permittivity.

## Adding a substrate

Multi-layer stacks combine planar wafer layers with patterned waveguide layers. Here the silicon core sits on a 2 µm buried-oxide (`"box"`) substrate with air above:

```python
--8<-- "examples/guide/layer_stacks.py:substrate"
```

The `"box"` level uses `"WAFER"`, which bifrost treats as an inverted (background-plane) layer by default: when no polygons are drawn on `WAFER`, its complement covers the entire xy plane from `zmin=-2.0` to `0.0` µm. Giving `"box"` a higher `mesh_order` (`9`) than `"core"` (`1`) ensures the core takes precedence wherever their vertical spans meet or overlap.

## Sloped sidewalls and anisotropic materials: lithium niobate

`LayerLevel` supports angled sidewalls and `fdtdx.Material` supports anisotropic permittivity tensors. The stack below models an X-cut thin-film lithium niobate (TFLN) ridge waveguide (`0.50` µm thick, `20°` trapezoidal sidewalls) on a `3.0` µm silica bottom oxide in air at `1.55` µm (used in the [Lithium niobate bend](../examples/linbo3-bend.md) example):

```python
--8<-- "examples/guide/layer_stacks.py:linbo3"
```

- **Trapezoidal sidewalls (`sidewall_angle`, `width_to_z`).** `sidewall_angle` tilts the boundary inward with height by `(z - z_ref) * tan(sidewall_angle)`, where `z_ref = zmin + width_to_z * thickness`. Because `width_to_z` defaults to `0.0`, the drawn GDS width applies at the bottom of the ridge (`z = zmin`) and a positive angle narrows the ridge toward the top (`width_to_z=0.5` anchors the drawn width at mid-height instead). Constant and height-varying horizontal offsets can also be specified with `bias` and `z_to_bias`.
- **Anisotropic permittivity.** In simulation coordinates with crystallographic X vertical along `z`, the optic axis Z along `y`, and Y along `x`, the ordinary (`n_o = 2.211`) and extraordinary (`n_e = 2.138`) refractive indices give the diagonal permittivity 3-tuple `(n_o**2, n_e**2, n_o**2)`.

## Derived layers and foundry stacks

Foundry stacks often define partial etches, slabs, or claddings as boolean combinations (`|`, `&`, `-`, `^`) of drawn mask layers via `LogicalLayer` (pass `derived_layer` to satisfy gdsfactory's `LayerLevel` validation when `layer` is a boolean `DerivedLayer`):

```python
--8<-- "examples/guide/layer_stacks.py:derived"
```

Here `"rib"` keeps the full 220 nm silicon thickness where `WG` is not covered by the `SLAB150` etch mask, while `"slab"` leaves a 90 nm slab across their union (`mesh_order=1` on `"rib"` overrides `mesh_order=2` on `"slab"` inside the rib). Outer sizing on a layer expression (`.sized(...)`) is also supported. To use a foundry PDK's built-in stack directly (such as `gf.gpdk.LAYER_STACK`), supply an entry in `materials` for each level that overlaps your simulation region.

## Negative-tone and inverted layers

In negative-tone or trench-etch processes, drawn polygons mark where material is **removed** rather than where the core sits. Define the `LayerStack` on the trench layer in the same way:

```python
--8<-- "examples/guide/layer_stacks.py:inverted"
```

Listing `DEEP_ETCH` alongside `"WAFER"` in `inverted_layers` when constructing the device (see [Simulation setup and ports](simulation-setup.md)) applies `"silicon"` to the complement of the drawn `DEEP_ETCH` trenches—leaving a ridge waveguide between two trenches—and offsets the remaining silicon when `sidewall_angle` or `bias` is set. Similarly, drawing polygons on `WAFER` cuts openings out of a `"WAFER"` substrate or cladding level.

## Summary of `LayerLevel` and `Material` features

| Feature | How to specify |
| --- | --- |
| Vertical span | `zmin` and `thickness` (in µm) on `LayerLevel` |
| Overlap priority | `mesh_order` on `LayerLevel` (lower integer wins) |
| Boolean & sized masks | `LogicalLayer` operators `\|`, `&`, `-`, `^`, and outer `.sized(...)` (with `derived_layer=...`), or `background=True` (`background_exclude_layers=...`) |
| Sidewalls & offsets | `sidewall_angle` (degrees), `width_to_z` (`0.0`..`1.0`), `bias`, and `z_to_bias` on `LayerLevel` |
| Isotropic dielectric | `fdtdx.Material(permittivity=n**2)` |
| Diagonal anisotropy | `fdtdx.Material(permittivity=(eps_x, eps_y, eps_z))` |
| Full 3 × 3 anisotropy | `fdtdx.Material(permittivity=((e_xx, e_xy, e_xz), ...))` (real symmetric positive-definite tensor; requires `averaging="subpixel_full"`, see [Geometry options](options.md)) |
| Dispersion & loss | `fdtdx.Material(permittivity=eps_inf, dispersion=fdtdx.DispersionModel(poles=(...)), electric_conductivity=sigma)` (see [Geometry model](geometry-model.md)) |

Next: [Simulation setup and ports](simulation-setup.md) turns a component, `LayerStack`, and `materials` dictionary into a [`bf.StackDevice`][bifrost.stack.StackDevice] with adaptive meshing, port sources, and monitors.
