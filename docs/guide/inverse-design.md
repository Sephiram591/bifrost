# Inverse design

A 90° silicon bend of 0.8 µm radius transmits only 81 % of TE0 at the uniform 0.5 µm width.
[`examples/guide/inverse_design.py`](https://github.com/Sephiram591/bifrost/blob/main/examples/guide/inverse_design.py)
optimizes the width on each side of its centerline for transmission, by gradient ascent through the simulation, and
raises it to 90 %: the bend's loss halves, from 0.91 dB to 0.45 dB, and a finer grid confirms it. Run it on a GPU; on
an NVIDIA L40S it took 977 s.

![The uniform and the optimized bend: |E|² on the middle of the core, and the transmission per step](../images/inverse_design.png)

## The bend and its parameters

```python
--8<-- "examples/guide/inverse_design.py:geometry"
```

- **Both ends are fixed:** 0.5 µm wide and centered, so the ports, their straight leads and the monitors in them never
  change. Only the six stations in between move, each edge linear from one to the next.
- **The topology is fixed:** every profile has the same 122 vertices, so the device built once, for the uniform bend,
  takes any profile's vertices with `stack.pack` (see [polygon topology](gradients.md#polygon-topology)).
- **Two widths per station,** toward the bend's center and away from it. A single width, symmetric about the
  centerline, has nowhere to go here: 0.4 and 0.6 µm transmit 0.74 against the uniform 0.5 µm's 0.81, and optimizing a
  symmetric profile gained only 0.007. Moving the guide toward the center is what helps, and only two independent
  edges can express that.

## The simulation

As in [Gradients](gradients.md), with the source 1 µm and both monitors 0.5 µm out in the straight leads, clear of the
moving edges: a monitor on the port itself would see the bend's edges move and change the mode it measures.

```python
--8<-- "examples/guide/inverse_design.py:simulation"
```

## Objective and optimizer

The objective is the normalized TE0 transmission `|a_out / a_in|²`. Each half-width is a bounded function of an
unbounded parameter, so the optimizer cannot leave 0.1 to 0.5 µm, and [optax](https://optax.readthedocs.io)'s Adam
(installed with fdtdx) steps on those parameters. `jax.value_and_grad` runs eagerly: the cell function needs concrete
values, so the objective is not wrapped in `jax.jit`.

```python
--8<-- "examples/guide/inverse_design.py:objective"

--8<-- "examples/guide/inverse_design.py:optimize"
```

## Result

```python
--8<-- "examples/guide/inverse_design.py:run"
```

Output (steps 1 to 6 and 10 to 13 left out):

```text
step  0  T = 0.8118  inner [0.25 0.25 0.25 0.25 0.25 0.25]  outer [0.25 0.25 0.25 0.25 0.25 0.25]
step  7  T = 0.9011  inner [0.334 0.351 0.377 0.379 0.356 0.347]  outer [0.186 0.192 0.189 0.192 0.2   0.189]
step  8  T = 0.9018  inner [0.346 0.36  0.389 0.392 0.366 0.36 ]  outer [0.18  0.19  0.182 0.185 0.196 0.182]
step  9  T = 0.9017  inner [0.359 0.367 0.398 0.402 0.374 0.373]  outer [0.173 0.189 0.177 0.179 0.194 0.176]
step 14  T = 0.8972  inner [0.41  0.384 0.425 0.43  0.392 0.418]  outer [0.152 0.189 0.17  0.174 0.196 0.154]
optimized half-widths (um): inner [0.346 0.36  0.389 0.392 0.366 0.36 ], outer [0.18  0.19  0.182 0.185 0.196 0.182]
transmission at 40 nm: uniform 0.8118, optimized 0.9018
transmission at 25 nm: uniform 0.8121, optimized 0.9052
wall time: 977 s on NVIDIA L40S
```

- **The design:** the inner edge moves 0.10 to 0.14 µm toward the center and the outer edge 0.05 to 0.07 µm in, so the
  guide is 0.53 to 0.58 µm wide and its middle about 0.09 µm closer to the center than the uniform bend's. In the
  figure, the uniform bend radiates from its outer edge after the turn; the optimized one keeps that light.
- **The best step is kept:** the transmission peaks at step 8 and then falls slowly as Adam overshoots, so `optimize`
  returns the best widths it has seen, not the last.
- **The gain is the device's, not the grid's:** on the 25 nm grid the uniform bend transmits 0.8121 and the optimized
  one 0.9052. The derivatives are those of the discretized simulation (see
  [what the derivatives are](gradients.md#what-the-derivatives-are)), so an optimum is worth checking on a finer grid.
- **The time:** 15 steps, each a forward and a backward solve, then four forward solves for the figure and the check.

## For your own designs

- Choose parameters that can express the improvement: here, two edges rather than one symmetric width.
- Bound them through a smooth map (here a sigmoid) instead of clipping, which would stop their gradients at the bounds.
- Keep the ports and the leads that hold the monitors fixed, and keep the number of vertices fixed; for a change of
  topology, rebuild the device with `retopologize` (see [polygon topology](gradients.md#polygon-topology)).
- Keep the best design seen, and check it on a finer grid.
