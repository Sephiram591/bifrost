# Geometry options

`StackOptions` collects the geometry discretization and accuracy settings for a `StackDevice`. All accuracy features are enabled by default; `StackOptions.fast()` turns off the slower options when you want a quick preview of a device. The example below builds the same waveguide with both presets:

```python
--8<-- "examples/guide/options.py:options"
```

Output:

```text
default  averaging=subpixel yee_staggered=True corner_lines=2 diagnostics=True
fast     averaging=inverse yee_staggered=False corner_lines=0 diagnostics=False
```

Pass `options=` to `StackDevice`, or pass individual fields such as `corner_lines=0`, but not both. `device.options` returns the resolved options of a built device.

## What each option buys

| Option (default) | Behavior |
|---|---|
| `voxelizer="greens"` (`"legacy"`) | Uses exact 2D/2.5D Green's-theorem grid-line integration with unified Kottke homogenization and dynamic junction handling by default; `"legacy"` selects the fixed-capacity signed-distance and local corner-model backend |
| `corner_lines=2` (0: off) | Computes exact fill fractions at polygon corners and edge crossings (sub-1e-6 fill error vs up to ~0.3 voxel error on convex or rotated corners with `"legacy"` and `corner_lines=0`) |
| `averaging="subpixel"` (`"inverse"`) | Kottke subpixel averaging using the diagonal of the homogenized permittivity tensor: second-order accurate at grid-aligned interfaces and first-order at oblique ones ([Permittivity averaging](geometry-model.md#permittivity-averaging)) |
| `averaging="subpixel_full"` (opt-in) | Writes full 3 × 3 permittivity tensors: supports anisotropic materials, Kottke's anisotropic laminate at planar interfaces, and periodic cell homogenization at junctions ([Full tensors](geometry-model.md#full-tensors)) |
| `subpixel_normal="voxel"` (`"gradient"`) | Computes interface normals from each voxel's local geometry rather than finite differences of the voxel-mean permittivity, avoiding 8–33° normal errors at slanted walls |
| `yee_staggered=True` | Samples each electric field component (`Ex`, `Ey`, `Ez`) at its staggered Yee grid location instead of the voxel center |
| `xy_subsamples=1` | Subdivides each lateral cell into `s × s` sub-squares (`s > 1` refines narrow tips and tight curves at roughly `s²` cost) |
| `derived_shapes="exact"` (`"composite"`) | Evaluates exact derived-layer polygons so coincident edges (such as `WG - ETCH`) leave no sub-voxel slivers |
| `diagnostics=True` | Emits `StackDiagnosticWarning` when `fdtdx.place_objects` encounters features or gaps narrower than the grid resolves |

Enabling the corner model or full-tensor averaging accounts for most of the geometry cost; the remaining options are inexpensive. See [Validation and benchmarks](../validation.md#geometry-option-costs) for measured forward and gradient timings.
