# Adaptive meshing (`bf.mesh`) accuracy-per-cell benchmarks

Drivers and results comparing `bf.mesh` against `fdtdx`'s `UniformGrid` and native-geometry MEEP uniform-grid simulations across six photonic benchmarks. The `bifrost` sweeps run on `voxelizer="greens"` under both subpixel averaging modes: `averaging="subpixel"` (the diagonal of the averaged tensor, bifrost's default) and `averaging="subpixel_full"` (the complete 3 × 3 tensor). The rotated anisotropic crystal slab runs only under `"subpixel_full"`, since `"subpixel"` rejects materials with off-diagonal permittivity.

All `bifrost` L40S sweep runs executed in isolated worker subprocesses (`float32`) with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.97` out to the 48 GB ceiling of an NVIDIA L40S (~140M–175M cells / 32.0–43.3 GB under `"subpixel"`, and ~50M–55M cells / 30.2–33.3 GB under `"subpixel_full"`), except the `"subpixel_full"` `directional_coupler` `uniform-20` and `mesh-16` grids (timed on NVIDIA H200) and four large `"subpixel_full"` `peak_gb` measurements exceeding L40S setup headroom (`elevator_coupler` `uniform-16`, `directional_coupler` `mesh-14` / `uniform-20` / `mesh-16`, recorded with `peak_gb_gpu: "NVIDIA B200"`). Fine-grid extension points recorded on an NVIDIA B200 (`505M–697M` cells / `114.1–153.8 GB` under `"subpixel"`, plus `aniso_slab` `mesh-160` and `uniform-1.5` under `"subpixel_full"`) and their matched-VRAM MEEP `UniformGrid` runs are included in `accuracy_per_cell.json`, `meep_uniform.json`, and all regenerated convergence figures.

## Reproducing

Run the full `bifrost` sweep for each averaging mode on a GPU node from the repository root; `--bench <name>` restricts a run to one benchmark and `--kind <k1,k2,...>` to a subset of its grids (`--platform-allocator` switches large grids to `XLA_PYTHON_CLIENT_ALLOCATOR=platform`, omitting `peak_gb`):

```bash
python benchmarks/meshing/accuracy_per_cell.py --averaging subpixel
python benchmarks/meshing/accuracy_per_cell.py --averaging subpixel_full
```

Run the matching native-geometry MEEP uniform-grid reference sweeps on CPU nodes (`8` MPI ranks for 1D `aniso_slab`, `16` MPI ranks for 3D benchmarks):

```bash
mpirun -np 8 python benchmarks/meshing/meep_uniform.py --bench aniso_slab --out benchmarks/meshing/results/meep_uniform.json
mpirun -np 16 python benchmarks/meshing/meep_uniform.py --bench <name> --out benchmarks/meshing/results/meep_uniform.json
```

`meep_uniform.py` constructs each device directly from `mp.Prism` and `mp.Block` primitives with `mp.Medium` (`eps_averaging=True`, never exporting permittivity arrays from `bifrost`) and computes `equiv_peak_gb` by linear interpolation against `total_cells` from `bifrost`'s `"subpixel"` `UniformGrid` sweep.

Rows in both JSONs are keyed cleanly so interrupted sweeps resume where they left off. `accuracy_per_cell.py` imports existing `voxelizer="greens"` rows of both averaging modes for `directional_coupler` (`benchmarks/greens/results/dc_convergence.json`) and `sin_bend` (`benchmarks/greens/results/bend_convergence.json`), executes all remaining configurations in clean subprocesses, and writes `benchmarks/meshing/results/accuracy_per_cell.json`. Legacy `soi_bragg` rows remain in `accuracy_per_cell.json` for historical reproducibility. Under `"subpixel_full"` without `tidy3d_extras`, mode sources and monitors are solved on the diagonal of the averaged tensor while the FDTD update uses the complete tensor (`_apply_stack_safe` in `benchmarks/greens/dc_convergence_compare.py`).

Regenerate all convergence plots with:

```bash
python benchmarks/meshing/plot_accuracy_per_cell.py
```

`plot_accuracy_per_cell.py` renders `docs/images/adaptive_meshing_vram.png`, `docs/images/adaptive_meshing_raw.png`, `benchmarks/meshing/results/adaptive_meshing_vram.png`, `benchmarks/meshing/results/adaptive_meshing_raw.png`, and `benchmarks/meshing/results/adaptive_meshing_accuracy.png`.

## Benchmarks

1. **`aniso_slab`** — 137 nm rotated anisotropic crystal slab (`eps = R(25°) diag(9, 12, 14) R(25°)^T + 0.8 (x z^T + z x^T)`) in air at 1.55 µm, normal incidence, `averaging="subpixel_full"` only, against the exact 4 × 4 Berreman transfer-matrix Jones transmission of `benchmarks/subpixel/berreman.py` (1D MEEP uses non-zero `eps_xz` / `eps_zx` off-diagonal permittivity) out to `1.5 nm` / `ppw=160` on B200.
2. **`sloped_guide`** — 500 × 220 nm silicon strip waveguide with 10° sidewalls in silica (6.40 × 3.20 × 3.20 µm box, `lambda = 1.55 um`), comparing `neff_mode` and `neff_fdtd` against the extrapolated `tidy3d` continuum reference (`2.381043` from `benchmarks/rectilinear/results/guide_tidy3d_reference.json`) out to `7.2 nm` / `ppw=142` under `"subpixel"` (`4.71 nm` for MEEP).
3. **`sin_bend`** — 90° air-clad SiN waveguide bend (`R = 7 um`, `1.6 × 0.75 um`, on `WAFER` SiO2 substrate, 12.36 × 12.36 × 4.80 µm) at 1.55 µm out to `11 nm` / `ppw=54` / `ppw=84 + material_cells`, comparing `UniformGrid`, default `bf.mesh`, `bf.mesh` with `Refine(material_cells={"silica": ...})`, and the inverse-design configuration `Refine(dilation={"WG": 0.45}, material_cells={"silica": 0.08})`.
4. **`elevator_coupler`** — Two-level SiN adiabatic elevator coupler (`overlap = 3.80 um`, 110 nm lower taper and 190 nm upper taper separated by a 100 nm vertical oxide gap, 50 nm tips, 19.20 × 4.00 × 3.20 µm box) at 619 nm out to `12 nm` / `ppw=52` (`7.24 nm` for MEEP).
5. **`directional_coupler`** — Broadband SiN directional coupler (400 × 200 nm waveguides, 300 nm gap over 20 µm, 43 × 6.4 × 2.6 µm box, 538–728 nm) out to `10.13 nm` / `ppw=30`, comparing splitting ratio at 619 nm against the joint uniform continuum fit `S0 = 0.3558` (`benchmarks/rectilinear/README.md`).
6. **`linbo3_bend`** — 90° X-cut LiNbO₃ ridge bend (`R = 7 um`, `1.2 × 0.50 um`, `20°` sidewalls, diagonal permittivity `(2.211^2, 2.138^2, 2.211^2)` on `WAFER` SiO2 substrate in air, 12.00 × 12.00 × 4.80 µm) at 1.55 µm out to `11.15 nm` / `ppw=48` / `ppw=78 + material_cells`, recording both `TE0` transmission and `TE0 -> TM0` polarization conversion.
