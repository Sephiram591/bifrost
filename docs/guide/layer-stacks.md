# Layer stacks and materials

`bf.StackDevice` turns a gdsfactory component and its
[`LayerStack`](https://gdsfactory.github.io/gdsfactory/notebooks/03_layer_stack/) into one
[fdtdx](https://github.com/ymahlau/fdtdx) object. Every `LayerLevel` of the stack is voxelized into a single
multi-material region, and the result is differentiable with respect to the polygon vertices of every layer (and so to
the parameters of the cell function that drew them) and to the thickness, height and sidewall angle of every level. You
never export a GDS or re-describe the geometry with fdtdx primitives.

This page builds devices; [Simulation setup and ports](simulation-setup.md) puts sources and monitors on them.

## A silicon waveguide in silica

The smallest useful stack has one level and two materials: a core and a uniform cladding. This script is complete: it
builds the component, the stack and the device. The component is gdsfactory's `straight` cell, and the waveguide
layer `gf.gpdk.LAYER.WG` comes from the generic PDK that `gf.gpdk.PDK.activate()` loads.

```python
--8<-- "examples/guide/layer_stacks.py:imports"

--8<-- "examples/guide/layer_stacks.py:bare"
```

It prints the simulation box that `from_layer_stack` derived:

```text
box (um): [7.0, 3.5, 3.22]
```

That is 5 µm of guide plus the two 1 µm port extensions in x, the 0.5 µm guide plus 1.5 µm of cladding on both sides in y,
and the 0.22 µm core plus 1.5 µm above and below in z.

- **`layer_stack`** lists the levels. A `LayerLevel` says which layer it draws, where it starts (`zmin`) and how thick it
  is (both in µm), its `material` by name, and its `mesh_order` (where levels overlap, the lower value wins).
- **`materials`** maps every material name that is drawn, plus the `background`, to an `fdtdx.Material`. Here the
  permittivity is constant: the indices are 3.48 (silicon) and 1.444 (silica).
- **`background`** fills everything no level covers: here the cladding.
- **`zrange_layers`** picks the levels whose z range sets the box's height before `boundary_offset` is applied.
- **`extend_ports`** continues a port's cross-section straight out of the component, so the waveguide can run through
  the absorbing boundary instead of ending inside the simulation.
- **`boundary_offset`** `= (x+, x-, y+, y-, z+, z-)` moves the box's faces by that many µm; positive values move a face
  toward +x, +y or +z. It changes the box, not the geometry.

!!! note "Keep the box on the grid"
    `from_layer_stack` uses `boundary_offset` exactly. When `fdtdx.place_objects` places the box it rounds each side to
    the nearest whole number of grid cells (on a 40 nm grid a 9.85 µm side becomes 9.84 µm), so the box you simulate can
    differ from the one you asked for by up to half a cell. Choose `boundary_offset` so the sides are whole numbers of
    cells at the grid you will use (a 10 × 10 × 3.4 µm box is one at 50, 40, 25 and 20 nm). A side exactly halfway
    between two cell counts can make `place_objects` stop with `Configured grid shape ... does not match simulation
    volume shape`.

Display the device with `viewer = bf.view(stack)`; see [Viewer and live demo](viewer.md).

## Adding a substrate

A foundry stack usually has a thick oxide under the core. The `WAFER` layer covers the whole plane, so a level on it
needs no polygon of its own. This second device is the same guide on a 2 µm silica substrate under air:

```python
--8<-- "examples/guide/layer_stacks.py:substrate"
```

Every level that overlaps the box is drawn, including the `box` level here, which `zrange_layers` does not list. The
box starts as the xy bounding box of the polygons (extensions included) and the z range of `zrange_layers`, so the
substrate shows up in the geometry without enlarging the simulation.

## Parameters

| Argument | Meaning |
| --- | --- |
| `materials` | name → `fdtdx.Material` for every drawn level and the `background` |
| `background` | the material outside every level |
| `zrange_layers` | levels whose z span sets the box's height |
| `boundary_offset` | the six box faces' offsets in µm |
| `extend_ports` | `(port, length)` pairs, or `(port, length, [layers])` to extend only some layers |
| `fill_layers` | layers that cover the whole plane (default `WAFER`) |
| `level_margin` | include levels within this distance of the box, so their `zmin` can move into it |

`extend_ports` extends, for each listed port, every stack layer that reaches the port face with the width of that layer
at the port, so a rib waveguide keeps its slab and an etch over the port continues. The extensions are polygons like
any other: they get the level's thickness, sidewall angle and bias, and they follow the port and the polygon edges when
the component's parameters change.

The voxelization supports most of `LayerLevel`:

- logical and derived layers;
- `mesh_order` priorities;
- `sidewall_angle`, `width_to_z`, `bias` and `z_to_bias`;
- background levels.

## Derived layers and foundry stacks

Real stacks define some layers from others (a slab is the core grown by some margin; an etch is subtracted from it).
The sketch below is an **illustration of the idea, not a runnable example**: the layer names stand for your own PDK's.

```python
# Illustration only: a rib waveguide, with the slab derived from the waveguide layer.
rib_stack = LayerStack(
    layers={
        "slab": LayerLevel(layer=SLAB, zmin=0.0, thickness=0.09, material="silicon", mesh_order=2),
        "rib": LayerLevel(layer=WG, zmin=0.0, thickness=0.22, material="silicon", mesh_order=1),
    }
)
```

To use a foundry PDK's own stack, pass its `LayerStack` (for example `gf.gpdk.LAYER_STACK`) in place of
`layer_stack` and give a material for every level it draws. How the geometry is turned into voxels, including derived
levels and sloped walls, is described in the [geometry model](geometry-model.md); the cost and accuracy of each option is
in [Geometry options](options.md).
