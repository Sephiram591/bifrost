# Waveguide bend

A 90° silicon waveguide bend, simulated with bifrost and fdtdx: the example behind [Getting started](../getting-started.md)
and the [live demo](../guide/viewer.md#live-demo). The page there explains the script step by step; this page is the
reference for running and changing it.

## The script

```python
--8<-- "examples/waveguide_bend.py"
```

## Run it

```bash
python examples/waveguide_bend.py                          # 40 nm grid, 400 fs: about half a minute on a GPU
python examples/waveguide_bend.py --resolution 0.1         # a coarse CPU-sized try-out
python examples/waveguide_bend.py --resolution 0.025 --figure bend_fine.png
```

The options are `--resolution` (the grid cell in µm; it must divide the 10 × 10 × 3.4 µm box evenly), `--duration` (the simulated
time in seconds) and `--figure` (where to save the field image). It prints the transmission, the wall time (compilation
included) and, on a GPU, the peak memory.

## Things to try

- **Change the geometry.** `RADIUS`, `WIDTH` and `THICKNESS` at the top of the file are in µm. A smaller radius radiates more.
  Keep the box a whole number of cells when you change the sizes that set it (`RADIUS`, `EXTENSION`, `MARGIN`).
- **Use the result elsewhere.** `simulate()` returns a `BendResult` with the stack, the placed objects, the arrays, the
  transmission and the field, so you can import the function and post-process it. `open_viewer(result)` returns the viewer.
- **Differentiate it.** The transmission is differentiable with respect to the radius and width; see
  [Gradients and inverse design](../guide/gradients.md).

## Settings and cost

| Grid cell | Transmission | Wall time | Peak GPU memory |
|---|---|---|---|
| 50 nm | 1.0012 | 24 s | 0.7 GiB |
| 40 nm (the script's default) | 1.0008 | 32 s | 1.2 GiB |
| 25 nm | 0.9998 | 106 s | 4.6 GiB |
| 20 nm | 0.9994 | 227 s | 8.7 GiB |

Measured on an NVIDIA L40S, each row in its own process; A run 1.5 times as long (600 fs) gives the same transmission to four digits at 25 nm (0.9998), so 400 fs is enough time for the pulse to cross the device and ring down. `scripts/bend_sweep.py` reproduces this table.
