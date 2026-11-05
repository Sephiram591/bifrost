# Waveguide bend

A 90° silicon waveguide bend: the script that [Getting started](../getting-started.md) walks through step by step, and
the result in the [live demo](../live-demo.md).

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

| Option | Meaning |
| --- | --- |
| `--resolution` | the grid cell in µm; the 10 × 10 × 3.4 µm box is a whole number of cells at 50, 40, 25 and 20 nm, and fdtdx rounds it to whole cells at other values |
| `--duration` | the simulated time in seconds |
| `--figure` | where to save the field image |

It prints the transmission, the wall time (compilation included) and, on a GPU, the peak memory. Transmission, time and
memory at four grids are in [Choosing the settings](../getting-started.md#choosing-the-settings).

## Things to try

- **Change the geometry.** `RADIUS`, `WIDTH` and `THICKNESS` at the top of the file are in µm. A smaller radius radiates
  more. Keep each side of the box a whole number of cells when you change the sizes that set it (`RADIUS`,
  `EXTENSION`, `MARGIN`): fdtdx rounds it to the nearest whole cell otherwise.
- **Use the result elsewhere.** `simulate()` returns a `BendResult` with the stack, the placed objects, the arrays, the
  transmission and the field, for post-processing; `open_viewer(result)` returns the viewer.
- **Differentiate it.** The transmission is differentiable with respect to the radius and width; see
  [Gradients and inverse design](../guide/gradients.md).
