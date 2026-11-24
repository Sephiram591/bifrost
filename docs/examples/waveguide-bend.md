# Waveguide bend

This example simulates a 90° silicon waveguide bend—the same script walked through step by step in [Getting started](../getting-started.md), with its precomputed output available in the [Widget Demo](../live-demo.md) and standalone [Explorer demo](../demo/index.html).

## The script

```python
--8<-- "examples/waveguide_bend.py"
```

## Run it

```bash
python examples/waveguide_bend.py                    # ppw=20 adaptive grid, 400 fs: about a minute on a GPU
python examples/waveguide_bend.py --ppw 10           # a coarse CPU-sized try-out
python examples/waveguide_bend.py --ppw 24 --figure bend_fine.png
```

| Option | Meaning |
| --- | --- |
| `--ppw` | points per local wavelength passed to [`bf.mesh`][bifrost.meshing.mesh]; default `20` |
| `--duration` | the simulated time in seconds |
| `--figure` | where to save the field image |

It prints the transmission, the wall time (compilation included) and, on a GPU, the peak memory. Transmission, time and
memory at four grids are in [Choosing the settings](../getting-started.md#choosing-the-settings).

## Things to try

- **Change the geometry.** `RADIUS`, `WIDTH` and `THICKNESS` at the top of the file are in µm. A smaller radius radiates
  more; changing `RADIUS`, `EXTENSION` or `MARGIN` changes the simulation box directly, and `bf.mesh` meshes any box size.
- **Use the result elsewhere.** `simulate()` returns a `BendResult` with the stack, the placed objects, the arrays, the
  transmission and the field, for post-processing; `open_viewer(result)` returns the viewer.
- **Differentiate it.** The transmission is differentiable with respect to the radius and width; see
  [Inverse design](../guide/inverse-design.md).
