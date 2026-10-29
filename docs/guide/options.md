# Geometry options

The geometry features that cost time are collected in `StackOptions`. They are all on by default, which is the most
accurate setting; `StackOptions.fast()` turns off the ones you do not need for a quick look at a device. This script is
complete: it builds the same device twice and reports the options each one carries.

```python
--8<-- "examples/guide/options.py:options"
```

It prints:

```text
default  averaging=subpixel yee_staggered=True corner_lines=2 diagnostics=True
fast     averaging=inverse yee_staggered=False corner_lines=0 diagnostics=False
```

Pass `options=` to `from_layer_stack`, as above, or give individual fields (`corner_lines=0`) instead; not both at once.
`device.options` returns the options of a built device.

## What each option buys

| Option (default) | What it buys |
|---|---|
| `corner_lines=2` (0: off) | exact fill at polygon corners and edge crossings: corner voxels off by up to 0.25 (convex) and 0.31 (rotated polygons) without, 1e-6 with |
| `averaging="subpixel"` (`"inverse"`) | Kottke averaging (the diagonal of the averaged tensor): second-order accurate at interfaces along the grid; first order at oblique ones (see below) |
| `averaging="subpixel_full"` (opt-in) | complete tensors: anisotropic materials (full tensors), Kottke's anisotropic laminate at planar interfaces in every entry, a junction closure at corners (see below) |
| `subpixel_normal="voxel"` (`"gradient"`) | the normal of each voxel's own model: exact at slanted walls (0.01 deg), where finite differences of the mean eps are 8-33 deg off |
| `yee_staggered=True` | each E component's own Yee position (first-order error without) |
| `xy_subsamples=1` | (s > 1 refines tips and dense curves; cost ~ s^2) |
| `derived_shapes="exact"` (`"composite"`) | no slivers where edges of a derived level's layers coincide (0.88 of a voxel before) |
| `diagnostics=True` | warnings about unresolved parts and gaps |

The cost of each option, measured over the test suites, is in
[Validation and benchmarks](../validation.md#geometry-option-costs). Roughly: the corner model and the full-tensor
averaging are the expensive ones; the rest are cheap.
