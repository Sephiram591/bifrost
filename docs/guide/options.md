# Geometry options

`StackOptions` collects the geometry features that cost time. All are on by default, the most accurate setting;
`StackOptions.fast()` turns off those a quick look at a device does not need. The same device, built both ways:

```python
--8<-- "examples/guide/options.py:options"
```

Output:

```text
default  averaging=subpixel yee_staggered=True corner_lines=2 diagnostics=True
fast     averaging=inverse yee_staggered=False corner_lines=0 diagnostics=False
```

Pass `options=` to `from_layer_stack`, or individual fields such as `corner_lines=0`, but not both. `device.options`
returns a built device's options.

## What each option buys

| Option (default) | What it buys |
|---|---|
| `corner_lines=2` (0: off) | exact fill at polygon corners and edge crossings: corner voxels off by up to 0.25 (convex) and 0.31 (rotated polygons) without, 1e-6 with |
| `averaging="subpixel"` (`"inverse"`) | Kottke averaging (the diagonal of the averaged tensor): second-order accurate at interfaces along the grid; first order at oblique ones (see below) |
| `averaging="subpixel_full"` (opt-in) | complete tensors: anisotropic materials (full tensors), Kottke's anisotropic laminate at planar interfaces in every entry, a junction closure at corners (see below) |
| `subpixel_normal="voxel"` (`"gradient"`) | the normal of each voxel's own model: exact at slanted walls (0.01 deg), where finite differences of the mean eps are 8-33 deg off |
| `yee_staggered=True` | each E component's own Yee position (first-order error without) |
| `xy_subsamples=1` | s > 1 refines tips and dense curves, at about s² the cost |
| `derived_shapes="exact"` (`"composite"`) | no slivers where edges of a derived level's layers coincide (0.88 of a voxel before) |
| `diagnostics=True` | warnings about unresolved parts and gaps |

The corner model and the full-tensor averaging are the expensive options; the rest are cheap. Measured costs:
[Validation and benchmarks](../validation.md#geometry-option-costs).
