# Gradients through the simulation

Every number in a `StackDevice` (the vertices of every polygon, the heights, thicknesses and sidewall angles of every
level) is differentiable through the whole simulation. This script is complete: it differentiates the **phase** a 90° bend
adds to the light with respect to the bend's **radius** with `jax.value_and_grad`, and compares the result with a central
finite difference of the same simulation. Run it on a GPU.

```python
--8<-- "examples/guide/gradients.py:gradient"
```

It prints:

```text
phase shift at 2.0 um: +0.000 rad (zero by construction)
d(phase)/dR: autodiff -5.64 rad per um, finite difference -5.64 rad per um
```

The two derivatives agree to the digits shown. (On an NVIDIA L40S the script took 133 s: four simulations, one of them
also run backward. In a scan of the phase shift at radii from 1.95 to 2.1 µm, the finite difference was -5.39, -5.58
and -5.64 rad per µm for steps of ±0.01, ±0.02 and ±0.05 µm: the smallest steps are the noisiest, because the grid's
cells are 0.04 µm wide.) How the script works:

- `stack.pack(component)` returns the stack's vertex parameters for a new component *of the same polygon topology*. Rebuilding
  the bend with a new radius inside the objective and merging its vertices into the parameters
  (`{**params["stack"], **placed.pack(bend)}`) makes the result a function of the radius.
- `bf.apply_stack` writes the stack into the arrays, `fdtdx.run_fdtd` solves, and JAX differentiates through all of it.
  `fdtdx.GradientConfig(method="checkpointed", ...)` bounds the memory of the backward pass: without a gradient
  configuration the run stores every time step and a GPU runs out of memory quickly.
- As with the layout backend, do not wrap the objective in `jax.jit` (the cell functions need concrete values). The
  derivative is that of the *discretized* simulation, as explained below.

Three things in the script are worth copying into your own objectives:

- **Choose an objective with a signal.** The transmission of a good bend is nearly flat in the radius (it barely moves around
  2 µm), and the grid's cell-scale noise is as large as its slope, so its derivative tells you little. The phase changes by
  several radians per µm.
- **Keep monitors in the straight lead for every parameter value.** The output monitor sits 0.5 µm out in the port
  extension. On the port itself, a larger radius would put the end of the arc past the monitor.
- **A mode's overall sign is arbitrary in each mode solve.** The ratio of two mode amplitudes is therefore defined only up to
  a sign, and the phase of the ratio jumps by π between some parameter values, which would ruin a finite difference. The
  script compares `ratio**2`, which does not depend on the sign, with a reference at the base radius.

The rest of this page is reference.

## The stack's parameters

The stack's parameters live in `params[name]`, in stack units (µm and degrees). This is an illustration of the
dictionary's layout, not a runnable example:

```python
params["stack"] = {
    "vertices/<layer>": ...,       # (E, 2), one entry per source layer with polygons
    "zmin/<level>": ...,           # ()
    "thickness/<level>": ...,      # ()
    "sidewall_angle/<level>": ..., # ()
}
```

Every level's `zmin` is independent: making a film thicker does not lift the
levels deposited on it. To keep them stacked, derive their `zmin` from the
level below inside the objective, e.g. for a cladding deposited on a 0.22 µm
core (a sketch, not runnable by itself):

```python
def stacked(t_core):
    return {**params["stack"], "thickness/core": t_core, "zmin/clad": params["stack"]["zmin/core"] + t_core}
```

Every level in the box is computed over the whole box height, so `zmin`,
thickness and roughness can move it anywhere inside the box. Levels outside
the box are not part of the device: pass `level_margin=` (µm) to
`from_layer_stack` to include the levels within that distance of the box, so
that they can be moved into it.

To differentiate with respect to a cell parameter, rebuild the component inside the objective and swap in its vertices
with `stack.pack`, as the script above does. `pack` requires the same polygon topology as the component the device was built
from, meaning the same number of polygons and vertices per polygon on every
layer. Thicknesses and sidewall angles are ordinary entries of the params dict
and can be differentiated directly. Where a surface lies exactly on a voxel
face (a thickness or `zmin` that is a whole number of cells from the box's
bottom, at some Yee position), the voxel fractions have a kink: the
derivative there is the mean of the two one-sided ones, half of the surface's
movement in each of the two voxels, with every material's fraction changing
consistently.

These are the derivatives of the discretized simulation (through a whole
fdtdx solve they agree with its finite differences to 1e-3), not those of the
physical device. Averaging puts an interface's material into the voxel it
crosses, so a response is piecewise smooth in an interface's position, with a
kink where the interface crosses a sample voxel's side, and at high contrast
its slope inside one voxel can be far from the physical one while its values
and its slope over a voxel follow them. In the anisotropic slab test
(`benchmarks/subpixel/em_slab.py --grad`, contrast about 9),
d|t|^2/d(thickness) is 0.13 at 20 nm and 0.59 at 10 nm, against 0.82 from
the transfer matrix, while |t|^2 is within 0.6 %. Refine the grid where the
optimum depends on a gradient with respect to an interface's position.

The device also takes a few things from the vertices it is built with, once,
and indexes later vertices with them by polygon and vertex: which corners lie
inside another polygon's edge (T-junctions, such as a taper on the side of an
MMI), which edges are seams between abutting polygons (left out of the
distance), the order of the polygons, which way each one runs, and where each
port extension meets the polygons on the port face. A parameter change that
breaks one of these would otherwise give a wrong geometry without an error: a
taper moved off an MMI side leaves a bump on that side, two abutting polygons
moved apart keep the gap between them filled, and a component that lists two
polygons of the same size in the other order hands each one the other's
topology. So whenever the component is concrete (not traced), `pack` calls
`stack.check_topology(component)`, which raises `ValueError` naming each layer
and what changed in it. Traced components, inside `jax.grad` or `jax.jit`, are
never checked, so the check costs nothing in the objective. Round-off, layout
snapping and moves that keep the topology, such as a taper sliding along an MMI
side or abutting polygons moving together, pass.

All of this is found within one distance, the device's `topology_tol`: a corner
this close to another polygon's edge lies on it, vertices this close to each
other coincide, and a vertex this close to a port face is on it. The default is
2 database units of the layout (`gf.kcl.dbu`, so 2 nm), because snapping to the
1 nm grid can move a corner of a rotated component up to about 1.4 nm off the
slanted edge it lies on; float32 round-off of the coordinates (2 epsilons times
the largest one, about 1.2 nm at 5000 µm from the origin) is added. A corner
within it bends the other polygon's edge through itself, by at most that
distance, instead of leaving a slit between the polygons. The check, a new
device (`from_layer_stack`) and `retopologize` all use the same tolerance, so
whatever the check accepts, a device built for the new vertices would also have.
Moves smaller than it (a taper 1 nm off an MMI side, a waveguide end 1 nm short
of an extended port's face) are accepted as keeping the topology. Abutting
edges whose corners are up to that far apart but not at the same point (which
snapping does not produce from corners that coincide) are a seam too: a voxel
whose center falls in the gap between them, if there is one, is filled as if it
were far outside. Pass `topology_tol=...` (µm) to `from_layer_stack` to change
it.

The check also reports polygons that the component now has on a stack layer
that had none when the device was built: the levels read such a layer as
empty, and a level that needs it (e.g. a slab level of a device built from a
strip waveguide) was not drawn at all. Only `StackDevice.from_layer_stack`
builds a device for them; `retopologize` raises.

When the new topology is intended, `stack.retopologize(component)` returns the
same device (box, levels, materials, placement) with the topology of the new
component; its polygons may also change in number and size. Swap it into the
placed objects and refresh its parameters (a sketch, not runnable by itself):

```python
objects = objects.copy()
objects["stack"] = objects["stack"].retopologize(component)
params = {**params, "stack": {**params["stack"], **objects["stack"].pack(component)}}
```

`check_topology` and `retopologize` also take a parameter dict (e.g.
`params["stack"]`) instead of a component, for vertices optimized directly;
port extensions and new layers are then not checked, and the extensions are not
found again. `pack(component,
check=False)` skips the check.

As with the layout backend, don't wrap the cell function in `jax.jit`; see
[the gdsfactoryx README](https://github.com/Sephiram591/gdsfactoryx#differentiability). Simulations with lossy materials
should use `fdtdx.GradientConfig(method="checkpointed")`, because fdtdx's
reversible gradients amplify round-off in lossy voxels. `StackDevice` warns
when it detects that combination.
