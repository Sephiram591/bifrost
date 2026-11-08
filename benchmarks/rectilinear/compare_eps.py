"""Cell by cell: the inverse permittivity a graded grid paints against a uniform one, where their cells coincide.

Reads two ``graded_guide.py --save-eps`` files (numpy only) and compares every component in the cells whose y and z
boxes, and those of their neighbours (the Yee-shifted components average over dual cells reaching into them), are
the same in both grids.

    python benchmarks/rectilinear/compare_eps.py uniform-20-eps.npz graded-snap-eps.npz
"""

from __future__ import annotations

import json
import sys

import numpy as np


def matches(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
    """For every cell of ``other``, the index of the same cell in ``ref`` (-1 where none), its neighbours too."""
    index = np.full(len(other) - 1, -1)
    for i in range(len(other) - 1):
        j = int(np.argmin(np.abs(ref - other[i])))
        if j + 1 < len(ref) and abs(ref[j] - other[i]) < 1e-6 and abs(ref[j + 1] - other[i + 1]) < 1e-6:
            index[i] = j
    inner = index.copy()
    for i in range(len(index)):  # the neighbours coincide too
        if i == 0 or i == len(index) - 1 or index[i - 1] < 0 or index[i + 1] < 0:
            inner[i] = -1
    return inner


def main() -> None:
    ref, other = (np.load(path) for path in sys.argv[1:3])
    iy, iz = matches(ref["y"], other["y"]), matches(ref["z"], other["z"])
    jy, jz = np.flatnonzero(iy >= 0), np.flatnonzero(iz >= 0)
    a = other["inv"][:, jy][:, :, jz]
    b = ref["inv"][:, iy[jy]][:, :, iz[jz]]
    mixed = np.abs(b - b.max(axis=(1, 2), keepdims=True)) > 1e-6  # not pure cladding
    diff = np.abs(a - b)
    out = {
        "cells_compared": [len(jy), len(jz)],
        "max_abs_diff": [float(d.max()) for d in diff],
        "max_rel_diff": [float((d / np.abs(r)).max()) for d, r in zip(diff, b, strict=True)],
        "max_abs_diff_interface": [float(d[m].max()) if m.any() else 0.0 for d, m in zip(diff, mixed, strict=True)],
        "interface_cells": [int(m.sum()) for m in mixed],
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
