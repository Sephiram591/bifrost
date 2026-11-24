# Geometry diagnostics

When a [`StackDevice`][bifrost.stack.StackDevice] is placed, bifrost checks each level's geometry against the grid and warns if any parts or gaps are too narrow for the grid to resolve. [`bf.mesh`][bifrost.meshing.mesh] automatically refines narrow features and gaps (`feature_cells`) unless capped by `min_cell`; a `min_cell`-limited mesh (shown below) or a coarse `UniformGrid` emits a [`StackDiagnosticWarning`][bifrost.stack.StackDiagnosticWarning]. Here is a 40 nm strip placed on a mesh capped at `min_cell=0.05` (~50 nm cells):

```python
--8<-- "examples/guide/diagnostics.py:diagnostics"
```

The warning:

```text
StackDevice 'stack', level 'core': 1 part narrower than 0.07895 um (1.5 cells); the narrowest is 0.04 um (0.76 cells) wide, at (2.5000, 0.0000) um and z = 0.0000 um. The grid does not resolve it: refine the grid, raise xy_subsamples, or widen the part (diagnostics=False turns these warnings off)
```

## What is checked

One `StackDiagnosticWarning` is emitted per level and diagnostic category:

- **Narrow parts and gaps.** Parts of a level, or gaps in it, narrower than 1.5 cells divided by `xy_subsamples` (using the larger cell width where x and y differ) between any two edges facing each other at less than 90°. That includes a corner pointing at an edge or another corner, and a V-notch that nearly cuts through a part. The two sides of one corner (sharp, cut, blunt, or rounded) are treated as the corner rather than a narrow part where they meet at 45° or more; sides meeting at a shallower angle (such as a needle or taper tip) are measured away from the tip, so sharp tips are not flagged while blunt ones are.
  - Each layer is offset by its signed distance with round joins where it grows at a convex corner or shrinks at a reflex one, matching the geometry drawn on the grid to within 1 % of the threshold. A round join belongs to its corner so grown corners do not trigger false narrow-part warnings.
  - Where a sidewall angle, bias, or `z_to_bias` offsets the level, cross-sections are checked across enough heights to catch parts that pinch or gaps that close inside the level.
  - Only features inside the simulation box are reported: a width or gap is detected whenever one of its edges enters the box, and it is measured and located within the box.
  - The warning reports the narrowest width or gap in µm and in cells, the `(x, y, z)` coordinates at its midpoint in µm, and the total count of such places (ignoring features at or below 1 % of the threshold).
  - Port extensions ending at a box face are excluded at that end.
  - Derived levels are checked on the exact shape they draw (see [Geometry model](geometry-model.md)). With `derived_shapes="composite"`, or for expressions that fall back to composite evaluation, coincident edges across layers (such as `WG - ETCH` ending flush with the waveguide) produce a sliver of twice the offset width when grown.
- **Overlapping polygons on a shrunk layer** (only with `derived_shapes="composite"`, or when an expression falls back to it), due to a sidewall angle above `z_ref`, negative `bias`, `z_to_bias`, or sizing, or a growing level that subtracts the layer. Composite distance evaluation shrinks each polygon individually rather than their union, opening a slit where an overlap is narrower than twice the shrink. Abutting polygons sharing an edge are not flagged, and grows are exact. With the default `derived_shapes="exact"`, overlapping polygons are unioned before offsetting and do not trigger this issue.

[`stack.diagnose`][bifrost.stack.StackDevice.diagnose] returns the findings programmatically as [`StackDiagnostic`][bifrost.stack.StackDiagnostic] tuples (pass the grid resolution in µm when calling `diagnose` before placement; see the API reference for every field and argument). To suppress warnings, pass `diagnostics=False` to `StackDevice`, or filter the warning class:

```python
import warnings

warnings.filterwarnings("ignore", category=bf.StackDiagnosticWarning)
```

Diagnostics use KLayout on the component polygons and nominal level heights (they are not re-run inside traced `jax.jit` or `jax.grad` calls). Typical components take tens of milliseconds (40 ms for a ring resonator); a 2000-tooth grating with a 30° sidewall over 1 µm thickness checks 64 heights in about 9 s.

## Timing the geometry steps

Step timing is disabled by default (costing a single global lookup per call) and never alters simulation values or gradients. Enable it with [`bf.record_timings`][bifrost.timing.record_timings] (see [`bf.timing`][bifrost.timing] for the full timing API):

```python
with bf.record_timings() as timings:
    device = bf.StackDevice(...)
    objects, arrays, params, config, _ = fdtdx.place_objects(...)
    arrays, objects, _ = bf.apply_stack(arrays, objects, params, key)
print(timings.table())  # per step: calls, wall, trace, compile, run (s)
```

Recorded steps and sub-step laps:

- **`init`** (laps `compile_levels`, `polygons`, `port_extensions`, `select_levels`) and **`derived_shapes`**
- **`pack`** (`pack_vertices`, `topology_check`), **`check_topology`**, and **`retopologize`**
- **`place_on_grid`** (`keep_reachable_levels`, `pole_alignment`, `corner_budget`, `junction_budget` for `"subpixel_full"`, `diagnose`)
- **`apply_stack`** (runs `write_complete` per device, then `write_stacks`, `apply_params`, `reapply_objects`)
- **`write`** (for `"inverse"` and `"subpixel"`, with laps `Ex`, `Ey`, `Ez` or `E`, `set_permittivity`, `conductivity`, and step `dispersion`) and **`weights`** (`level_surfaces`, `layer_geometry`, `source_sdfs`, `corner_plan`, `corner_capacity`, `corner_inputs`, `composite` or `composite_samples`, `corner_pass`, and `normal_tensor`)
- **`write_complete`** (for `"subpixel_full"`, delegating to `write` otherwise): **`interfaces`** and **`complete_tensor`**, plus `Ex`, `Ey`, `Ez`, `set_permittivity`, `conductivity`, and `dispersion`

Nested steps are included in their callers' totals.

A step called while JAX traces a function for `jax.jit` attributes its wall time to tracing time (part of compiling). On concrete inputs (including eager `jax.grad` without `jit`), each step blocks on its outputs (`jax.block_until_ready`) when exiting and splits wall time into JAX compile time (tracing, lowering, and XLA compilation events) and run time. Lowering and compilation of outer jitted functions, as well as the backward pass of eager `jax.grad`, occur outside inner steps: read [`StepTimings.outside_compile`][bifrost.timing.StepTimings.outside_compile] or wrap the entire call in [`bf.timing.timed_step`][bifrost.timing.timed_step] to measure them.

Contributors running the test suite can record timings across tests with `FDTDX_TIMINGS=<name> pytest`; see [Contributing and tests](../development.md).
