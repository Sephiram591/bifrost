# Validation and benchmarks

The drivers of the StackDevice tensor and corner work are in `benchmarks/subpixel/`, their results in
`benchmarks/subpixel/results/`. Reproduce a run with `python benchmarks/subpixel/<driver>.py out.json` on a GPU. Every
driver prints its versions (jax, fdtdx, the gdsfactory it imported) and uses fixed seeds; GPU runs were on an NVIDIA
L40S (jax 0.11.2, fdtdx 0.6.3, float32 simulations), CPU references with numpy/scipy/shapely and meep.

--8<-- "benchmarks/subpixel/README.md:results"

## Geometry option costs

Wall time of each [geometry option](guide/options.md) over the full fdtdx test suites (205 tests, 34 min on one L40S,
timed with `FDTDX_TIMINGS=<name> pytest`, see [Timing the geometry steps](guide/diagnostics.md#timing-the-geometry-steps)).
Most of that time is compiling: the tests build many small devices and compile each once. These are lower bounds: the
steps leave out the lowering and compilation of jitted callers and the backward passes (about a third of the test time
in that run), and no suite was run with a feature off.

| Option (default) | Suite cost |
|---|---|
| `corner_lines=2` (0: off) | 5.7 min (`corner_pass` 176 s, `corner_inputs` 123 s, `corner_budget` 38 s, `corner_plan` 6 s) |
| `averaging="subpixel"` (`"inverse"`) | part of `composite` painting, at most 4.2 min with everything else painted there |
| `averaging="subpixel_full"` (opt-in) | 9-component arrays: fdtdx's fully anisotropic update in the whole simulation, ~4.9x per time step and ~1.8x GPU memory on a 33M-cell device; writing: see the write-cost table above (compiled once per device shape) |
| `subpixel_normal="voxel"` (`"gradient"`) | part of the same painting; per call about 1.7x write forward and 3.0x gradient over `"gradient"` (0.095 s / 0.59 s against 0.056 s / 0.20 s on the 130 x 90 x 45 test device) |
| `yee_staggered=True` | 1.1 min (`write.Ey`, `write.Ez`) plus the corner budget's extra grids |
| `xy_subsamples=1` | 1.4 min for the tests that use 2 |
| `derived_shapes="exact"` (`"composite"`) | < 0.3 min (`derived_shapes` 0.3 s; evaluated within `layer_geometry`, 15 s for all layers) |
| `diagnostics=True` | 0.4 s |

Not switchable, because they are cheap or are the geometry itself: the per-sub-interval sidewall offsets (inside the
painting), the topology check of `pack` (0.2 s; `pack(check=False)` skips it) and `retopologize` (1 s).

The largest single cost is not a feature: `fdtdx.place_objects` evaluates `material_weights` eagerly once per placement
(for its material mapping), which compiles the geometry op by op: 8.4 of the suite's 34 min. That mapping skips the
corner model (it only rounds to the dominant material, and `apply_stack` overwrites it): with it, the first placement of
a 260 x 180 x 90 device spent 42 s autotuning the corner pass (49 s in all; now 7.3 s, as before the corner model). The
first jitted `write` still compiles and autotunes the corner pass, once per device shape.

Against the code before these accuracy changes, on the 130 x 90 x 45 test device (L40S, jitted, all options at their
defaults), `write` takes 0.095 s forward (2.3x the old 0.041 s) and 0.59 s for its gradient (4.4x the old 0.13 s); at
260 x 180 x 90 the gradient takes 7.0 s (3.8x the old 1.9 s). `StackOptions.fast()` brings `material_weights` back to
the old cost (12.5 ms forward, 32.6 ms gradient, against 13.2 ms and 32.2 ms).

