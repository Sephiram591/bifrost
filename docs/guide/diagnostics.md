# Geometry diagnostics

A grid can only draw what it resolves. When a device is placed, bifrost checks what each level draws against the grid and
warns about the parts it cannot represent. This script is complete: a 40 nm strip on a 50 nm grid, placed without a run.

```python
--8<-- "examples/guide/diagnostics.py:diagnostics"
```

It prints one warning for the strip:

```text
StackDevice 'stack', level 'core': 1 part narrower than 0.075 um (1.5 cells); the narrowest is 0.04 um (0.8 cells) wide, at (2.5000, 0.0000) um and z = 0.0000 um. The grid does not resolve it: refine the grid, raise xy_subsamples, or widen the part (diagnostics=False turns these warnings off)
```

The rest of this page describes what is checked.

## What is checked

When fdtdx places the device, the device checks the shape every level draws at
the grid's resolution. It warns with a `StackDiagnosticWarning`, one per level
and kind of problem, about:

- **Narrow parts and gaps.** Parts of a level, or gaps in it, narrower than
  1.5 cells divided by `xy_subsamples` (sub-samples resolve thinner features),
  between any two edges that face each other at less than 90°. That includes
  a corner pointing at an edge or at another corner, and a V-notch that nearly
  cuts through a part. The two sides of one corner (sharp, cut, blunt or
  rounded) are the corner, not a narrow part, where they meet at 45° or more.
  Sides meeting at a smaller angle (a needle or a taper) are measured where
  they do not touch, so a sharp tip is never reported but a blunt one is.
  - The shape checked is the one the device draws. Each layer is offset by its
    signed distance, with round joins where it grows at a convex corner or
    shrinks at a reflex one, drawn to within 1% of the threshold. For example,
    two squares whose corners face each other diagonally 0.08 µm apart, grown
    by 0.02 µm, are 0.04 µm apart, not the 0.023 µm of square-cornered
    sizing. A round join belongs to its corner: it is not a narrow part
    together with the edges next to it, so a strip grown with round ends keeps
    its width.
  - Where a sidewall angle, bias or `z_to_bias` offsets the level, the shape is
    checked at enough heights to also find a part that narrows or vanishes, or
    a gap that closes, within the level.
  - Only the box counts. A width or gap is found where one of its two edges
    enters the box, and it is measured and located on the parts of its edges
    in the box. So a gap that narrows toward a point outside the box is
    reported with its width at the box.
  - The warning gives the narrowest one in µm and in cells, a point in its
    middle (x, y and z in µm) and how many places there are. Widths and gaps
    of 1% of the threshold or less are round-off and are left out.
  - Port extensions that end at a box face are not checked at that end.
  - Derived levels are checked on the exact shape they draw (see
    [Geometry model](geometry-model.md)). With `derived_shapes="composite"`, or
    for a level that fell back to it, a narrow part can also come from layers
    whose edges coincide: `WG - ETCH` with the etch ending exactly where the
    waveguide ends then draws a sliver twice the offset wide where the level
    grows.
- **Overlapping polygons on a layer that a level shrinks** (only with
  `derived_shapes="composite"`, or for a level that fell back to it), through a
  sidewall angle above `z_ref`, a negative bias, `z_to_bias` or sizing, or a
  growing level that subtracts the layer. The composite distance of a layer is
  the minimum of its polygons' distances, so the level draws each polygon
  shrunk instead of their union shrunk. That opens a slit where an overlap is
  narrower than about twice the shrink. Polygons that only abut are not
  reported (their shared edges are seams), and grows are exact. With the
  default `derived_shapes="exact"`, such a level draws the exact union and is
  not reported.

`stack.diagnose(resolution)` returns the same findings as data, as
`StackDiagnostic` tuples (level, kind, size in µm and cells, threshold,
location, z, count, layers and message). Before the device is placed, pass the
resolution in µm. To turn the warnings off, pass `diagnostics=False` to
`from_layer_stack`, or filter them:

```python
import warnings

warnings.filterwarnings("ignore", category=bf.StackDiagnosticWarning)
```

The checks run KLayout on the polygons and the nominal level heights the device
was built with (its initial parameters), not on traced values. They take
milliseconds for typical components (40 ms for a ring resonator). A 2000-tooth
grating whose 1 µm thick level has a 30° sidewall is checked at 64 heights and
takes about 9 s.

## Timing the geometry steps

Every step that builds the geometry can be timed. Timing is off by default and
then costs one global lookup per step; it never changes values or gradients.

```python

with bf.record_timings() as timings:
    device = bf.StackDevice.from_layer_stack(...)
    objects, arrays, params, config, _ = fdtdx.place_objects(...)
    arrays, objects, _ = bf.apply_stack(arrays, objects, params, key)
print(timings.table())  # per step: calls, wall, trace, compile, run (s)
```

The steps are `from_layer_stack` (laps `compile_levels`, `polygons`,
`port_extensions`, `select_levels`), `derived_shapes`, `place_on_grid`
(`keep_reachable_levels`, `pole_alignment`, `corner_budget`, `diagnose`),
`pack` (`pack_vertices`, `topology_check`), `check_topology`, `retopologize`,
`apply_stack` (`write_stacks`, `apply_params`), `write` (laps `Ex`, `Ey`, `Ez`
or `E`, `set_permittivity`, `conductivity`, step `dispersion`) and, inside
`write` or `material_weights`, `weights` with `level_surfaces`,
`layer_geometry` (where derived shapes are evaluated), `source_sdfs`,
`corner_plan`, `corner_capacity`, `corner_inputs`, `composite` or
`composite_samples`, `corner_pass` and `normal_tensor`. Nested steps also
count in their callers. A step that runs while JAX traces a function for
`jax.jit` counts its wall time as tracing time, which is part of compiling. A
step on concrete values, also under eager `jax.grad` (without `jit`), waits for
its outputs (`jax.block_until_ready`; their primals under autodiff) when it
ends, and splits its time into compile (JAX's tracing, lowering and XLA
compile events while it runs) and run. Nothing waits inside traced code. The
backward pass of an eager `jax.grad` and the lowering and compilation of jitted
callers happen outside every step: `StepTimings.outside_compile` sums the
compile events that fire while no step is open, and `timed_step(name)` around
the whole call (forward and backward) times them.

The fdtdx test suites time themselves with
`FDTDX_TIMINGS=<name> pytest tests/test_stack.py tests/test_geometry.py`
(or `--fdtdx-timings=<name>`): the plugin `tests/fdtdx_timing_plugin.py` also
times `fdtdx.place_objects` and `fdtdx.run_fdtd` (step `fdtd_solve`; the test
`test_fdtd_solve_forward_and_gradient` adds `fdtd_solve_forward` and
`fdtd_solve_gradient`, the whole forward and `jax.value_and_grad` calls with
compile and backward pass), prints the per-step table, the time outside the
steps (per test and in total) and the slowest tests with their steps, and
writes the data to `.scratch/fdtdx_work/timings/<name>.json`. Under pytest-xdist
each worker writes `<name>.<worker id>.json`.
