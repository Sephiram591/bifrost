# Geometry options

[`StackOptions`][bifrost.stack.StackOptions] collects the geometry discretization and accuracy settings for a [`StackDevice`][bifrost.stack.StackDevice]. All accuracy features are enabled by default; [`StackOptions.fast`][bifrost.stack.StackOptions.fast] turns off the slower options when you want a quick preview of a device. The example below builds the same waveguide with both presets:

```python
--8<-- "examples/guide/options.py:options"
```

Output:

```text
default  averaging=subpixel yee_staggered=True corner_lines=2 diagnostics=True
fast     averaging=inverse yee_staggered=False corner_lines=0 diagnostics=False
```

Pass `options=` to `StackDevice`, or pass individual fields such as `corner_lines=0`, but not both. [`device.options`][bifrost.stack.StackDevice.options] returns the resolved options of a built device.

## What each option buys

See [`StackOptions`][bifrost.stack.StackOptions] in the API reference for every field, accepted value, and default. In practice, the trade-offs are:

- **Voxelizer and corners (`voxelizer`, `corner_lines`, `xy_subsamples`, `derived_shapes`):** By default, bifrost uses exact 2D/2.5D Green's-theorem grid-line integration (`voxelizer="greens"`) and exact boolean polygon recipes (`derived_shapes="exact"`). Switching to `voxelizer="legacy"` uses fixed-capacity signed-distance painting with local corner splitting (`corner_lines`), and `xy_subsamples` refines narrow tips and curves laterally at roughly quadratic cost.
- **Permittivity averaging (`averaging`, `subpixel_normal`, `yee_staggered`):** `"subpixel"` (the default) applies Kottke subpixel averaging along voxel-local geometry normals on staggered Yee locations ([Permittivity averaging](geometry-model.md#permittivity-averaging)). `"subpixel_full"` writes full 3 × 3 anisotropic permittivity tensors ([Full tensors](geometry-model.md#full-tensors)), and `"inverse"` uses fdtdx's cheaper isotropic `1 / eps` mean.
- **Diagnostics (`diagnostics`):** Emits [`StackDiagnosticWarning`][bifrost.stack.StackDiagnosticWarning] when `fdtdx.place_objects` encounters features or gaps narrower than the grid resolves ([Geometry diagnostics](diagnostics.md)).

Enabling the corner model or full-tensor averaging accounts for most of the geometry cost; the remaining options are inexpensive. See [Validation and benchmarks](../validation.md#geometry-option-costs) for measured forward and gradient timings.
