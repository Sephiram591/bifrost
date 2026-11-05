# Layer stacks and materials

`bf.StackDevice` turns a gdsfactory component and its
[`LayerStack`](https://gdsfactory.github.io/gdsfactory/notebooks/03_layer_stack/) into one
[fdtdx](https://github.com/ymahlau/fdtdx) object. Every `LayerLevel` of the stack is voxelized into a single
multi-material region, differentiable with respect to the polygon vertices of every layer (and so to the parameters of
the cell function that drew them) and to the thickness, height and sidewall angle of every level.

## A silicon waveguide in silica

The smallest useful stack has one level and two materials: a core in a uniform cladding. The component is gdsfactory's
`straight` cell on the generic PDK's waveguide layer `gf.gpdk.LAYER.WG`.

```python
--8<-- "examples/guide/layer_stacks.py:imports"

--8<-- "examples/guide/layer_stacks.py:bare"
```

The box that `from_layer_stack` derives:

```text
box (um): [7.0, 3.5, 3.22]
```

In x, 5 µm of guide plus the two 1 µm port extensions; in y, the 0.5 µm guide plus 1.5 µm of cladding on both sides; in
z, the 0.22 µm core plus 1.5 µm above and below.

- **`layer_stack`** lists the levels. A `LayerLevel` gives the layer it draws, where it starts (`zmin`) and how thick it
  is (both in µm), its `material` by name, and its `mesh_order` (where levels overlap, the lower value wins).
- **`materials`** maps every drawn material name, plus the `background`, to an `fdtdx.Material`. Here the permittivity
  is constant: the indices are 3.48 (silicon) and 1.444 (silica).
- **`background`** fills everything no level covers: here the cladding.
- **`zrange_layers`** picks the levels whose z range sets the box's height before `boundary_offset` is applied.
- **`extend_ports`** continues a port's cross-section straight out of the component, so the waveguide runs through the
  absorbing boundary instead of ending inside the simulation.
- **`boundary_offset`** `= (x+, x-, y+, y-, z+, z-)` moves the box's faces by that many µm; positive values move a face
  toward +x, +y or +z. It changes the box, not the geometry.

!!! note "Keep the box on the grid"
    `fdtdx.place_objects` rounds each side of the box to the nearest whole number of grid cells (on a 40 nm grid a
    9.85 µm side becomes 9.84 µm), so the simulated box can differ from `boundary_offset` by up to half a cell. Choose
    sides that are whole numbers of cells at your grid (10 × 10 × 3.4 µm is one at 50, 40, 25 and 20 nm). A side exactly
    halfway between two cell counts can make `place_objects` stop with
    `Configured grid shape ... does not match simulation volume shape`.

Display the device with `viewer = bf.view(stack)`; see [Viewer](viewer.md).

## Adding a substrate

The same guide on a 2 µm silica substrate under air. The substrate is a level on the `WAFER` layer, which covers the
whole plane, so it needs no polygon of its own:

```python
--8<-- "examples/guide/layer_stacks.py:substrate"
```

Every level that overlaps the box is drawn, including the `box` level here, which `zrange_layers` does not list. The
box starts as the xy bounding box of the polygons (extensions included) and the z range of `zrange_layers`, so the
substrate appears in the geometry without enlarging the simulation.

## Parameters

Besides the arguments above, `from_layer_stack` takes:

| Argument | Meaning |
| --- | --- |
| `extend_ports` | also `(port, length, [layers])`, to extend only some layers |
| `fill_layers` | layers that cover the whole plane (default `WAFER`) |
| `level_margin` | include levels within this distance of the box, so their `zmin` can move into it |
| `dtype` | `jnp.float32` (default) or `jnp.float64`: the precision of the whole simulation, see [Precision](simulation-setup.md#precision) |

`extend_ports` extends, for each listed port, every stack layer that reaches the port face, with that layer's width at
the port, so a rib waveguide keeps its slab and an etch over the port continues. The extensions are polygons like any
other: they get the level's thickness, sidewall angle and bias, and follow the port and the polygon edges when the
component's parameters change.

The voxelization supports most of `LayerLevel`:

- logical and derived layers;
- `mesh_order` priorities;
- `sidewall_angle`, `width_to_z`, `bias` and `z_to_bias`;
- background levels.

## Derived layers and foundry stacks

Real stacks define some layers from others (a slab is the core grown by some margin; an etch is subtracted from it). A
rib waveguide, as a sketch (`SLAB` and `WG` stand for your PDK's layers):

```python
rib_stack = LayerStack(
    layers={
        "slab": LayerLevel(layer=SLAB, zmin=0.0, thickness=0.09, material="silicon", mesh_order=2),
        "rib": LayerLevel(layer=WG, zmin=0.0, thickness=0.22, material="silicon", mesh_order=1),
    }
)
```

To use a foundry PDK's own stack, pass its `LayerStack` (for example `gf.gpdk.LAYER_STACK`) and give a material for
every level it draws. The [geometry model](geometry-model.md) describes how geometry becomes voxels, including derived
levels and sloped walls; [Geometry options](options.md) gives the cost and accuracy of each option.
