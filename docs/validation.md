# Validation and benchmarks

Validation and accuracy benchmarks for `StackDevice` subpixel homogenization, corner handling, and anisotropic materials live in `benchmarks/subpixel/`, with saved results in `benchmarks/subpixel/results/`. Reproduce any benchmark on a GPU with `python benchmarks/subpixel/<driver>.py out.json`. Each driver prints the installed `jax`, `fdtdx`, and `gdsfactory` versions and uses fixed seeds; GPU runs were recorded on an NVIDIA L40S (`jax` 0.11.2, `fdtdx` 0.6.3, `float32` simulations), and CPU references use NumPy, SciPy, Shapely, and MEEP.

--8<-- "benchmarks/subpixel/README.md:results"

## Geometry option costs

The table below lists cumulative wall times for each [geometry option](guide/options.md) across the 205 tests in the full test suite (34 min total on one NVIDIA L40S, measured with `FDTDX_TIMINGS=<name> pytest`; see [Timing the geometry steps](guide/diagnostics.md#timing-the-geometry-steps)). Most of that time is first-call compilation across many small device shapes; inner step totals exclude outer JIT lowering/compilation and backward passes.

| Option (default) | Suite cost |
|---|---|
| `corner_lines=2` (0: off) | 5.7 min (`corner_pass` 176 s, `corner_inputs` 123 s, `corner_budget` 38 s, `corner_plan` 6 s) |
| `averaging="subpixel"` (`"inverse"`) | part of `composite` painting (at most 4.2 min combined across all painting) |
| `averaging="subpixel_full"` (opt-in) | 9-component tensors using fdtdx's fully anisotropic FDTD update (~4.8–6.5× median runtime and ~1.2–2.8× median peak memory across full simulations; see `Write cost` above for per-call array writing) |
| `subpixel_normal="voxel"` (`"gradient"`) | ~1.7× write forward and ~3.0× write gradient over `"gradient"` (0.095 s / 0.59 s vs 0.056 s / 0.20 s on a 130 × 90 × 45 device) |
| `yee_staggered=True` | 1.1 min (`write.Ey`, `write.Ez`) plus staggered corner budgets |
| `xy_subsamples=1` | 1.4 min across tests that enable `xy_subsamples=2` |
| `derived_shapes="exact"` (`"composite"`) | < 0.3 min (`derived_shapes` 0.3 s; evaluated within `layer_geometry`, 15 s across all layers) |
| `diagnostics=True` | 0.4 s |

Sidewall offsets along z sub-intervals, the topology check in `pack` (0.2 s across the suite; `pack(check=False)` skips it), and `retopologize` (1 s) add negligible overhead.

During `fdtdx.place_objects`, `material_weights` runs eagerly once per placement to build fdtdx's initial material-index map (8.4 min across the 205-test suite). Because `apply_stack` overwrites the permittivity arrays later, that initial mapping rounds to the dominant material and skips the corner pass, keeping first-time placement of a 260 × 180 × 90 device at 7.3 s; the first jitted `write` call compiles and autotunes the corner pass once per device shape.

On a jitted 130 × 90 × 45 device on an NVIDIA L40S, `write` with all default `StackOptions()` takes 0.095 s forward and 0.59 s for its gradient (7.0 s gradient at 260 × 180 × 90). `StackOptions.fast()` turns off subpixel averaging, Yee staggering, the corner model, and diagnostics for quick prototyping, evaluating `material_weights` in 12.5 ms forward and 32.6 ms gradient.

## Adaptive meshing benchmarks

Accuracy-per-cell, VRAM, and MEEP reference convergence sweeps comparing `bf.mesh` against `UniformGrid` across six photonic simulations—under `averaging="subpixel"`, `averaging="subpixel_full"`, and native `mp.Prism`/`mp.Block` uniform-grid MEEP—live in `benchmarks/meshing/` (`accuracy_per_cell.py`, `meep_uniform.py`, `plot_accuracy_per_cell.py`, results in `benchmarks/meshing/results/accuracy_per_cell.json` and `benchmarks/meshing/results/meep_uniform.json`) and are summarized in [Convergence testing](guide/convergence-testing.md).
