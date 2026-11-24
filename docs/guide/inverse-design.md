# Inverse design

An air-clad silicon nitride guide on silica, 1.6 µm wide and 0.75 µm thick, bent 90° at 7 µm radius, transmits only
56 % of TE0 at 1.55 µm as TE0 at its uniform width: most of the rest leaves the bend as TE1.
[`examples/guide/inverse_design.py`](https://github.com/Sephiram591/bifrost/blob/main/examples/guide/inverse_design.py)
optimizes the width on each side of its centerline, at 70 stations, for TE0 transmission, by gradient ascent through
the simulation, and raises it to 99.3 %: the bend's loss falls from 2.5 dB to 0.03 dB, and a finer mesh confirms it.
Run it on a GPU; on an NVIDIA L40S its 50 steps took 13489 s.

![The uniform and the optimized bend: |E|² on the middle of the core, and the transmission per step](../images/inverse_design.png)

## The bend and its parameters

```python
--8<-- "examples/guide/inverse_design.py:geometry"
```

- **The guide:** 1.6 µm wide and 0.75 µm thick, it guides four modes: TE0 (effective index 1.81), TM0 (1.76), TE1
  (1.63) and TM1 (1.61); TE2 (1.436) is below the silica's 1.444.
- **Where the light goes:** the uniform bend transmits 0.558 of TE0 as TE0 and 0.402 as TE1 (at `ppw = 20`; 0.558 and
  0.404 at `ppw = 24`). Nearly all the light gets through, but 40 % of the input leaves as TE1: the straight guide's
  mode does not match the bend's, which sits toward its outer edge, so the junctions excite both bend modes. Only 4 %
  is lost otherwise.
- **Both ends are fixed:** 1.6 µm wide and centered, so the ports, their straight leads and the monitors in them never
  change. Only the 70 stations in between move, each edge linear from one to the next.
- **The topology is fixed:** every profile has the same 144 vertices, one per station and end on each edge, so the
  device built once, for the uniform bend, takes any profile's vertices with [`stack.pack`][bifrost.stack.StackDevice.pack] (see
  [polygon topology](gradients.md#polygon-topology)).
- **Two widths per station,** toward the bend's center and away from it. A single width, symmetric about the
  centerline, helps here: narrowing the bend to 1.2 µm raises the transmission to 0.85 (widening it to 2.0 µm lowers
  it to 0.25). Moving the guide toward the center, so the straight guide's mode lines up with the bend's, helps more:
  0.1, 0.2 and 0.3 µm give 0.82, 0.96 and 0.92. Only two independent edges can express both.

## The simulation

This example places the source 1 µm and both monitors 0.5 µm out in the straight leads so they stay clear of the moving bend edges (a monitor directly on the port would see the bend edges move and distort the reference mode). The silica substrate sits on the `WAFER` layer across the bottom of the box under an air background (see [Adding a substrate](layer-stacks.md#adding-a-substrate)). Because the guided mode penetrates into the silica, the box is 4.8 µm tall and the source and monitor planes extend 0.5 wavelengths (0.78 µm) above and below the core, where the mode field decays to about 3 % while remaining clear of the absorbing boundaries.

```python
--8<-- "examples/guide/inverse_design.py:simulation"
```

## Objective and optimizer

The objective is the normalized TE0 transmission `|a_out / a_in|²`. Each half-width is a bounded function of an
unbounded parameter, so the optimizer cannot leave 0.25 to 1.25 µm (a guide 0.5 to 2.5 µm wide), and
[optax](https://optax.readthedocs.io)'s Adam (installed with fdtdx) steps on those parameters. `jax.value_and_grad` of
the objective is wrapped in [`bf.jit`][bifrost.jitting.jit], so the cell function, [`apply_stack`][bifrost.stack.apply_stack], the simulation and its
backward pass compile into one program, once, at the first step: every step has the same shapes, and the cell
function's structure (144 vertices, no branch on a half-width) does not depend on the parameters, so gdsfactoryx can
trace it (see [the gdsfactoryx README](https://github.com/Sephiram591/gdsfactoryx#differentiability)). The arrays and
parameters are arguments of the compiled function rather than constants captured by the trace.

[`bf.jit`][bifrost.jitting.jit] wraps `jax.jit` with an XLA compiler option disabling the `simplify-conditional` pass: under plain `jax.jit`, XLA turns the conditional inside fdtdx's checkpointed backward loop into a `select` and evaluates a time step's VJP on every iteration, including forward recomputations (see [`bf.jitting`][bifrost.jitting] for the speedup). With `bf.jit`, the `ppw = 20` optimization below completes in 13489 s on an NVIDIA L40S.

```python
--8<-- "examples/guide/inverse_design.py:objective"

--8<-- "examples/guide/inverse_design.py:optimize"
```

## Result

```python
--8<-- "examples/guide/inverse_design.py:run"
```

Output (steps 2 to 4, 6 to 9, 11 to 14, 16 to 19, 21 to 29 and 31 to 48 left out):

```text
step  0  T = 0.5584  width 1.60 to 1.60, shift +0.00 to +0.00 um
step  1  T = 0.6271  width 1.59 to 1.60, shift -0.01 to +0.01 um
step  5  T = 0.8554  width 1.53 to 1.60, shift -0.04 to +0.04 um
step 10  T = 0.9632  width 1.46 to 1.61, shift -0.07 to +0.07 um
step 15  T = 0.9747  width 1.43 to 1.64, shift -0.08 to +0.09 um
step 20  T = 0.9737  width 1.45 to 1.68, shift -0.09 to +0.11 um
step 30  T = 0.9879  width 1.54 to 1.74, shift -0.07 to +0.11 um
step 49  T = 0.9934  width 1.54 to 1.77, shift -0.07 to +0.12 um
optimized: width 1.54 to 1.77, shift -0.07 to +0.12 um
transmission at ppw=20: uniform 0.5584, optimized 0.9934
transmission at ppw=24: uniform 0.5576, optimized 0.9930
wall time: 13489 s on NVIDIA L40S
```

- **The design:** the guide stays 1.54 to 1.77 µm wide, and its middle moves by up to 0.12 µm toward the center and
  0.07 µm away from it along the bend. In the figure, the uniform bend's field beats across the guide as TE0 and TE1
  interfere; the optimized bend carries one mode along its outer edge. With no penalty on curvature, the inner edge,
  where little light goes, keeps ripples a few stations long; a design for fabrication would smooth or penalize them.
- **The best step is kept:** here it is the last, though the transmission dips slightly between steps 16 and 19. Because
  noisy steps (for example at a higher learning rate) can overshoot before recovering, `optimize` returns the best
  widths it has seen, not the last.
- **The gain is the device's, not the grid's:** on the finer `ppw = 24` mesh the uniform bend transmits 0.5576 and the
  optimized one 0.9930. The derivatives are those of the discretized simulation (see
  [what the derivatives are](gradients.md#what-the-derivatives-are)), so an optimum is worth checking on a finer grid.
- **The time:** 50 steps, each a forward and a backward solve, then four forward solves for the figure and the check.

## For your own designs

- Choose parameters that can express the improvement: here, two edges rather than one symmetric width.
- Scale the step to the parameters: with 140 of them, Adam's early steps move every edge by about the same amount, so
  a smaller learning rate gives a steadier rise and a smoother profile.
- Bound them through a smooth map (here a sigmoid) instead of clipping, which would stop their gradients at the bounds.
- Keep the ports and the leads that hold the monitors fixed, and keep the number of vertices fixed; for a change of
  topology, rebuild the device with [`retopologize`][bifrost.stack.StackDevice.retopologize] (see [polygon topology](gradients.md#polygon-topology)).
- Keep the best design seen, and check it on a finer grid.
