# Geometry options and their cost

The geometry features that cost time are collected in `StackOptions`, all on
(the most accurate) by default:

```python

device = bf.StackDevice.from_layer_stack(..., options=bf.StackOptions(corner_lines=0))
device.options                                  # bf.StackOptions(...) of a device
bf.StackDevice.from_layer_stack(..., options=bf.StackOptions.fast())  # quick look
```

Each option is also a StackDevice field and can still be passed on its own
(`corner_lines=0`), but not both ways at once. The costs below are wall time
over the full fdtdx test suites (205 tests, 34 min on one L40S, timed with the
plugin above), where most of the time is compiling: tests build many small
devices and compile each once. They are lower bounds: the steps leave out the
lowering and compilation of jitted callers and the backward passes (about a
third of the test time in that run), and no suite was run with a feature off.

| Option (default) | What it buys | Suite cost |
|---|---|---|
| `corner_lines=2` (0: off) | exact fill at polygon corners and edge crossings: corner voxels off by up to 0.25 (convex) and 0.31 (rotated polygons) without, 1e-6 with | 5.7 min (`corner_pass` 176 s, `corner_inputs` 123 s, `corner_budget` 38 s, `corner_plan` 6 s) |
| `averaging="subpixel"` (`"inverse"`) | Kottke averaging (the diagonal of the averaged tensor): second-order accurate at interfaces along the grid; first order at oblique ones (see below) | part of `composite` painting, at most 4.2 min with everything else painted there |
| `averaging="subpixel_full"` (opt-in) | complete tensors: anisotropic materials (full tensors), Kottke's anisotropic laminate at planar interfaces in every entry, a junction closure at corners (see below) | 9-component arrays: fdtdx's fully anisotropic update in the whole simulation, ~4.9x per time step and ~1.8x GPU memory on a 33M-cell device; writing (L40S, 130 x 90 x 45 test device, 1514 junction voxels over the Yee positions): 0.21 s forward and 0.96 s gradient against 0.096 s and 0.70 s, peak memory 2.5 GB against 1.3 GB (260 x 180 x 90: 0.85 s, 5.3 s, 17 GB against 1.4 s, 7.9 s, 5.2 GB); compiling the write takes about 60 s and its gradient 270 s, once per device shape |
| `subpixel_normal="voxel"` (`"gradient"`) | the normal of each voxel's own model: exact at slanted walls (0.01 deg), where finite differences of the mean eps are 8-33 deg off | part of the same painting; per call about 1.7x write forward and 3.0x gradient over `"gradient"` (0.095 s / 0.59 s against 0.056 s / 0.20 s on the 130 x 90 x 45 test device) |
| `yee_staggered=True` | each E component's own Yee position (first-order error without) | 1.1 min (`write.Ey`, `write.Ez`) plus the corner budget's extra grids |
| `xy_subsamples=1` | (s > 1 refines tips and dense curves; cost ~ s^2) | 1.4 min for the tests that use 2 |
| `derived_shapes="exact"` (`"composite"`) | no slivers where edges of a derived level's layers coincide (0.88 of a voxel before) | < 0.3 min (`derived_shapes` 0.3 s; evaluated within `layer_geometry`, 15 s for all layers) |
| `diagnostics=True` | warnings about unresolved parts and gaps | 0.4 s |

Not switchable, because they are cheap or are the geometry itself: the
per-sub-interval sidewall offsets (inside the painting), the topology check of
`pack` (0.2 s; `pack(check=False)` skips it) and `retopologize` (1 s).
The largest single cost is not a feature: `fdtdx.place_objects` evaluates
`material_weights` eagerly once per placement (for its material mapping),
which compiles the geometry op by op: 8.4 of the suite's 34 min. That mapping
skips the corner model (it only rounds to the dominant material, and
`apply_stack` overwrites it): with it, the first placement of a 260 x 180 x 90
device spent 42 s autotuning the corner pass (49 s in all; now 7.3 s, as
before the corner model). The first jitted `write` still compiles and
autotunes the corner pass, once per device shape.

Against the code before these accuracy changes, on the 130 x 90 x 45 test
device (L40S, jitted, all options at their defaults), `write` takes 0.095 s
forward (2.3x the old 0.041 s) and 0.59 s for its gradient (4.4x the old
0.13 s); at 260 x 180 x 90 the gradient takes 7.0 s (3.8x the old 1.9 s).
`StackOptions.fast()` brings `material_weights` back to the old cost (12.5 ms
forward, 32.6 ms gradient, against 13.2 ms and 32.2 ms).
