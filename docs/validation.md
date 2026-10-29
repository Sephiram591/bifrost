# Validation and benchmarks

Drivers and results of the sub-pixel and tensor work, under `benchmarks/subpixel/`. Reproduce a run with `python benchmarks/subpixel/<driver>.py out.json` on a GPU.


Drivers and their results (`results/`) for the StackDevice tensor and corner work. Every driver prints its
versions (jax, fdtdx, the gdsfactory it imported) and uses fixed seeds; GPU runs were on an NVIDIA L40S
(jax 0.11.2, fdtdx 0.6.3, float32 simulations), CPU references with numpy/scipy/shapely and meep.

| Driver | What it checks | Result |
|---|---|---|
| `reference_fem.py` | independent P1 finite-element homogenization of polygon partitions (periodic, Dirichlet, mean-field boundary conditions; interface-fitted, refined) | the reference for the closure studies |
| `study_boundary_conditions.py` | which cell problem a voxel tensor should solve, against the voxel embedded in a 5 x 5 patch | voxel-alone periodic: exact for grid-aligned laminates; Dirichlet and mean-field biased (16 %, 38 %) |
| `study_closure.py` | sub-cell models for the closure (Voigt, laminate sub-cells; 2 to 8 sub-cells) | laminate sub-cells converge 3 to 5x faster; thresholds r = 0.002 to 0.02 |
| `validate_closure.py` | the production closure on junction voxels of placed devices (an L, a rotated square, an acute triangle) against the reference | closure vs embedded: 5.1 % median, 14.6 % p90; laminate along one normal: 31 %, 42 %; closure vs exact periodic: 2.2 % median |
| `legacy_fingerprint.py` | the legacy averagings' weights and tensors, bitwise, against the branch point (deterministic GPU ops) | all identical |
| `berreman.py` | 4 x 4 transfer matrix for anisotropic layers at normal incidence (self-checks against Fabry-Perot and decoupled eigenpolarizations) | reference for `em_slab.py` |
| `em_slab.py` | a 137 nm slab of a rotated uniaxial crystal with an xz entry in fdtdx vs Berreman, 40 to 5 nm, three grid offsets; staircase and diagonal baselines; reciprocity; `--grad` gradient through the solve | see below |
| `em_grating.py` | corner scattering: a grating of rotated rods (scalar and anisotropic) in fdtdx vs meep's subpixel smoothing | see below |
| `em_stability.py` | a closed periodic box with full tensors in every material and junctions everywhere, 52450 steps | energy within 1 % after the pulse, no growth |
| `bench_write.py` | cost of `write` and its gradient against `"subpixel"` | see below |
| `analyze_em.py` | tables and observed orders of the EM runs | |

## Anisotropic slab (`results/slab.json`)

Error of the Jones transmission against Berreman, the worst of three slab positions against the grid:

| averaging / tensor | 40 nm | 20 nm | 10 nm | 5 nm | order |
|---|---|---|---|---|---|
| `subpixel_full`, rotated + xz (polarization conversion 0.035) | 1.3e-2 | 5.0e-3 | 9.9e-4 | 1.9e-4 | 2.4 |
| faces snapped to the cells (staircase) | 0.12 | 3.6e-2 | 1.7e-2 | 9.2e-3 | 0.9 |
| `subpixel_full`, diagonal | 1.4e-2 | 5.4e-3 | 1.1e-3 | 2.0e-4 | 2.4 |
| `subpixel`, diagonal | same as above | | | | |
| `inverse`, diagonal | 6.9e-2 | 3.6e-2 | 2.6e-2 | 1.3e-2 | 1.0 |

|r|^2 + |t|^2 = 1 within 2e-5; t symmetric within 6e-8; the backward run's t is t^T within 4e-6.

Gradient through the whole solve (`results/slab_grad.json`), d|t_xx|^2/d(thickness) and d|t_yx|^2/d(thickness):
jax against central differences of the simulation 8e-4 and 1e-4 (20 nm), 1e-4 and 3e-4 (10 nm). Against the
transfer matrix's derivative they are 84 % and 43 % off at 20 nm, 28 % and 14 % at 10 nm: the discretized
response is piecewise smooth in an interface's position (`results/slab_scan_20.0_subpixel_full.json`: within
one voxel the slope is that of the averaged voxel, with a kink where the interface crosses a sample voxel's
side), while |t|^2 itself is within 0.6 %.

## Write cost (`results/bench*.json`, L40S)

| device | averaging | write | gradient | peak memory |
|---|---|---|---|---|
| 130 x 90 x 45 | `subpixel` | 0.096 s | 0.70 s | 1.3 GB |
| 130 x 90 x 45 | `subpixel_full` (1514 junction voxels) | 0.21 s | 0.96 s | 2.5 GB |
| 260 x 180 x 90 | `subpixel` | 1.4 s | 7.9 s | 5.2 GB |
| 260 x 180 x 90 | `subpixel_full` (3049 junction voxels) | 0.85 s | 5.3 s | 17 GB |

First calls compile: about 60 s for the write and 270 s for its gradient with `subpixel_full`.

## Corner scattering: grating of rotated rods (`results/grating_*.json`)

Zeroth-order transmission of a grating of 0.3 x 0.2 um rods rotated by 30 degrees (period 0.6 um, in-plane E,
1.55 um), against meep's subpixel smoothing at 50 to 800 pixels/um extrapolated with its observed order
(about 1.1; the reference is uncertain to about 2e-3). Worst of three rod positions against the grid:

| rods, averaging | 40 nm | 20 nm | 10 nm | 5 nm |
|---|---|---|---|---|
| eps 12, `subpixel_full` | 3.2e-3 | 1.6e-3 | 7.7e-4 | 6.5e-4 |
| eps 12, previous `subpixel_full` (one averaged normal per voxel) | 7.5e-3 | 4.4e-3 | 2.0e-3 | 7.4e-4 |
| eps 12, `subpixel` (diagonal) | 2.7e-2 | 1.8e-2 | 9.8e-3 | 5.0e-3 |
| eps 12, `inverse` | 9.4e-2 | 5.6e-2 | 3.1e-2 | 1.7e-2 |
| rotated uniaxial crystal with xz entry, `subpixel_full` | 4.2e-3 | 1.8e-3 | 3.8e-4 | 4.2e-4 |

For the crystal rods, fdtdx's own extrapolation (changes 1.2e-3, 5.3e-4, 1.0e-4 between successive
resolutions) agrees with meep's within 3e-4. At 45 degrees meep is not yet in its asymptotic range (uncertain
to about 5e-3); both `subpixel_full` versions are within that from 20 nm on.

## Stability (`results/stability.json`)

A closed periodic box (130 x 113 x 45 at 20 nm: a waveguide with 10 degree sidewalls, a slab, a deep etch, a
rotated pillar and a pad; every material a different full tensor; 2555 junction voxels over the Yee
positions), excited by a dipole pulse: over 52450 steps the total energy after the pulse stays between 1.000
and 1.010 of its first value (last: 1.003).

## Geometry option costs

The costs of the [geometry options](guide/options.md) are wall time over the full fdtdx test suites (205 tests, 34 min on
one L40S, timed with the plugin that `FDTDX_TIMINGS=<name> pytest` turns on, see [Contributing and tests](development.md)),
where most of the time is compiling: the tests build many small devices and compile each once. They are lower bounds: the
steps leave out the lowering and compilation of jitted callers and the backward passes (about a third of the test time in
that run), and no suite was run with a feature off.

| Option (default) | Suite cost |
|---|---|
| `corner_lines=2` (0: off) | 5.7 min (`corner_pass` 176 s, `corner_inputs` 123 s, `corner_budget` 38 s, `corner_plan` 6 s) |
| `averaging="subpixel"` (`"inverse"`) | part of `composite` painting, at most 4.2 min with everything else painted there |
| `averaging="subpixel_full"` (opt-in) | 9-component arrays: fdtdx's fully anisotropic update in the whole simulation, ~4.9x per time step and ~1.8x GPU memory on a 33M-cell device; writing (L40S, 130 x 90 x 45 test device, 1514 junction voxels over the Yee positions): 0.21 s forward and 0.96 s gradient against 0.096 s and 0.70 s, peak memory 2.5 GB against 1.3 GB (260 x 180 x 90: 0.85 s, 5.3 s, 17 GB against 1.4 s, 7.9 s, 5.2 GB); compiling the write takes about 60 s and its gradient 270 s, once per device shape |
| `subpixel_normal="voxel"` (`"gradient"`) | part of the same painting; per call about 1.7x write forward and 3.0x gradient over `"gradient"` (0.095 s / 0.59 s against 0.056 s / 0.20 s on the 130 x 90 x 45 test device) |
| `yee_staggered=True` | 1.1 min (`write.Ey`, `write.Ez`) plus the corner budget's extra grids |
| `xy_subsamples=1` | 1.4 min for the tests that use 2 |
| `derived_shapes="exact"` (`"composite"`) | < 0.3 min (`derived_shapes` 0.3 s; evaluated within `layer_geometry`, 15 s for all layers) |
| `diagnostics=True` | 0.4 s |

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

