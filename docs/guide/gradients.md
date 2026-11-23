# Gradients through the simulation

Every number in a `StackDevice` (the vertices of every polygon, the heights, thicknesses and sidewall angles of every
level) is differentiable through the whole simulation. The example differentiates the **phase** a 90° bend adds to the
light with respect to the bend's **radius** with `jax.value_and_grad`, and compares the result with a central finite
difference of the same simulation. Run it on a GPU.

```python
--8<-- "examples/guide/gradients.py:gradient"
```

Output:

```text
phase shift at 2.0 um: +0.000 rad (zero by construction)
d(phase)/dR: autodiff -6.43 rad per um, finite difference -5.64 rad per um
```

On an NVIDIA L40S the script took 319 s: four simulations, one of them also run backward. On this `bf.mesh` grid
(`~22 nm` cells in the core), `autodiff` gives the local slope inside a single voxel at `R = 2.0` (`-6.43 rad per µm`),
while central finite differences of ±0.01, ±0.02 and ±0.05 µm step across voxel faces and yield `-5.39`, `-5.61` and
`-5.64` rad per µm (see [What the derivatives are](#what-the-derivatives-are)).

How the script works:

- `stack.pack(component)` returns the stack's vertex parameters for a new component *of the same polygon topology*.
  Rebuilding the bend with a new radius inside the objective and merging its vertices into the parameters
  (`{**params["stack"], **placed.pack(bend)}`) makes the result a function of the radius.
- `bf.apply_stack` writes the stack into the arrays, `fdtdx.run_fdtd` solves, and JAX differentiates through all of it.
  `fdtdx.GradientConfig(method="checkpointed", ...)` bounds the memory of the backward pass: without a gradient
  configuration the run stores every time step and runs out of memory quickly. Simulations with lossy materials need the
  checkpointed method, because fdtdx's reversible gradients amplify round-off in lossy voxels; `StackDevice` warns about
  that combination.
- This objective runs eagerly: `bend_circular` picks its number of points from the radius, which needs a concrete
  value. A cell function whose structure does not depend on the parameters (the same polygons and number of vertices,
  no `if` on a parameter) can be traced, and then the whole objective can be wrapped in
  [`bf.jit`][bifrost.jitting.jit] (`jax.jit` with an XLA option for gradients through fdtdx's time loop), as
  [Inverse design](inverse-design.md) does; discrete choices and KLayout operations work only eagerly (see
  [the gdsfactoryx README](https://github.com/Sephiram591/gdsfactoryx#differentiability)). So does `apply_stack` in
  dispersive simulations with mode or plane-wave sources (see
  [dispersive materials](geometry-model.md#dispersive-materials)).

Worth copying into your own objectives:

- **Choose an objective with a signal.** The transmission of a good bend is nearly flat in the radius (it barely moves
  around 2 µm), and the grid's cell-scale noise is as large as its slope, so its derivative tells you little. The phase
  changes by several radians per µm. A bend tight enough to lose light is the opposite case: its transmission has a
  signal, and [Inverse design](inverse-design.md) optimizes it.
- **Keep monitors in the straight extension for every parameter value.** The output monitor sits 0.5 µm out in the port
  extension. On the port itself, a larger radius would put the end of the arc past the monitor.
- **A mode's overall sign is arbitrary in each mode solve.** The ratio of two mode amplitudes is defined only up to a
  sign, so its phase jumps by π between some parameter values, which would ruin a finite difference. The script compares
  `ratio**2`, which does not depend on the sign, with a reference at the base radius.

## The stack's parameters

The stack's parameters live in `params[name]`, in stack units (µm and degrees):

```python
params["stack"] = {
    "vertices/<layer>": ...,       # (E, 2), one entry per source layer with polygons
    "zmin/<level>": ...,           # ()
    "thickness/<level>": ...,      # ()
    "sidewall_angle/<level>": ..., # ()
}
```

Thicknesses, heights and sidewall angles are ordinary entries and can be differentiated directly. Every level's `zmin`
is independent: making a film thicker does not lift the levels deposited on it. To keep them stacked, derive their `zmin`
from the level below inside the objective, e.g. for a cladding deposited on the core:

```python
def stacked(t_core):
    return {**params["stack"], "thickness/core": t_core, "zmin/clad": params["stack"]["zmin/core"] + t_core}
```

Every level in the box is computed over the whole box height, so `zmin`, thickness and roughness can move it anywhere
inside the box. Levels outside the box are not part of the device: `level_margin=` (µm) in `StackDevice` includes
the levels within that distance of the box, so they can be moved into it.

To differentiate with respect to a cell parameter, rebuild the component inside the objective and swap in its vertices
with `stack.pack`, as the script does. `pack` requires the polygon topology the device was built from: the same number
of polygons, and of vertices per polygon, on every layer.

## What the derivatives are

They are the derivatives of the discretized simulation, not of the continuum device; across a full fdtdx solve they
agree with finite differences of the simulation to 1e-3. Subpixel averaging blends an interface's material across the
voxel it crosses, so a response is piecewise smooth in an interface's position, with a slope kink where the interface
crosses a sample voxel's face. At high refractive-index contrast, the derivative inside a single voxel can differ from
the continuum slope even while the objective value itself is already close: for example, across a high-contrast dielectric
slab (`ε` contrast ~9, `benchmarks/subpixel/em_slab.py --grad`), `d|t|^2/d(thickness)` is 0.13 at 20 nm and 0.59 at
10 nm against 0.82 from the exact transfer matrix, while `|t|^2` is already within 0.6 %. Refine the grid where an
optimum is sensitive to the position of a high-contrast interface.

Where a surface lies exactly on a voxel face (a thickness or `zmin` that is a whole number of cells from the box's
bottom, at some Yee position), the voxel fractions have a kink: the derivative there is the mean of the two one-sided
ones, half of the surface's movement in each of the two voxels, with every material's fraction changing consistently.

On a [non-uniform grid](simulation-setup.md#graded-grids) the grid is part of the compiled program, like a uniform
one: a fixed grid compiles once, and the derivatives are those of that grid's discretization (none with respect
to the cell edges). The kinks above sit on that grid's own voxel faces, so they are closer together in fine cells;
refine where an interface whose position you optimize can move.

## Polygon topology

When a device is built, `StackDevice` extracts combinatorial topology from the initial vertices and uses it to index later parameter updates by polygon and vertex: T-junctions (where a corner lies on another polygon's edge, such as a taper meeting an MMI side), seams between abutting polygons, polygon ordering and orientation, and where port extensions meet polygons at port faces. Any parameter update that breaks these relationships would corrupt the geometry—for instance, pulling a taper off an MMI side, opening a gap between abutting polygons, or reordering polygons.

Whenever `component` is concrete (outside `jax.jit` or `jax.grad`), `pack` runs `stack.check_topology(component)` and raises a `ValueError` naming any layer whose topology changed. Traced components inside `jax.grad` or `jax.jit` skip the check, so it adds zero overhead inside optimization loops. Small numerical noise, layout snapping, and topology-preserving moves (such as sliding a taper along an MMI side or moving abutting polygons together) pass; pass `pack(component, check=False)` to skip the concrete check explicitly.

All topology checks share one distance tolerance, `topology_tol`: vertices within `topology_tol` of an edge lie on it, vertices within `topology_tol` of each other coincide, and vertices within `topology_tol` of a port face lie on it. The default is 2 database units (`gf.kcl.dbu`, typically 2 nm) plus `float32` coordinate round-off (`2 * eps * max_coord`, about 1.2 nm at 5000 µm from the origin), which accommodates 1 nm grid snapping on rotated edges (`sqrt(2)` nm) without leaving slits between polygons. Pass `topology_tol=...` (µm) to `StackDevice` to override it.

The check also reports polygons on a stack layer that had none when the device was built: the levels read such a layer
as empty, and a level that needs it (e.g. a slab level of a device built from a strip waveguide) was not drawn at all.
Only `StackDevice` builds a device for them; `retopologize` raises.

When the new topology is intended, `stack.retopologize(component)` returns the same device (box, levels, materials,
placement) with the topology of the new component, whose polygons may also change in number and size. Swap it into the
placed objects and refresh its parameters:

```python
objects = objects.copy()
objects["stack"] = objects["stack"].retopologize(component)
params = {**params, "stack": {**params["stack"], **objects["stack"].pack(component)}}
```

`check_topology` and `retopologize` also take a parameter dict (e.g. `params["stack"]`) instead of a component, for
vertices optimized directly; port extensions and new layers are then not checked, and the extensions are not found
again.
