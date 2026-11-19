# Adaptive meshing (`bf.mesh`) accuracy-per-cell benchmarks

Drivers and results comparing `bf.mesh` against `fdtdx`'s `UniformGrid` across six photonic simulations on `voxelizer="greens"`, each under both subpixel averaging modes: `averaging="subpixel"` (the diagonal of the averaged tensor, bifrost's default) and `averaging="subpixel_full"` (the complete tensor). The rotated anisotropic crystal slab runs only under `"subpixel_full"`, since `"subpixel"` rejects materials with off-diagonal permittivity. All GPU runs executed in isolated worker subprocesses (float32) on an NVIDIA L40S, except the `"subpixel_full"` `directional_coupler` `uniform-20` and `mesh-16` grids, which need more than 48 GB and ran on an NVIDIA H200 (each row records its `gpu`).

## Reproducing

Run the full sweep for each averaging mode on a GPU node from the repository root; `--bench <name>` restricts a run to one benchmark and `--kind <k1,k2,...>` to a subset of its grids:

```bash
python benchmarks/meshing/accuracy_per_cell.py --averaging subpixel
python benchmarks/meshing/accuracy_per_cell.py --averaging subpixel_full
python benchmarks/meshing/plot_accuracy_per_cell.py
```

Rows are keyed by `(bench, kind, averaging)`, and existing rows are skipped, so an interrupted sweep resumes where it stopped. The script imports existing `voxelizer="greens"` rows of both averaging modes for `directional_coupler` (`benchmarks/greens/results/dc_convergence.json`) and `sin_bend` (`benchmarks/greens/results/bend_convergence.json`), executes all remaining configurations in clean subprocesses, and writes `benchmarks/meshing/results/accuracy_per_cell.json`. Under `"subpixel_full"` without `tidy3d_extras`, mode sources and monitors are solved on the diagonal of the averaged tensor while the FDTD update uses the complete tensor (`_apply_stack_safe` in `benchmarks/greens/dc_convergence_compare.py`). `plot_accuracy_per_cell.py` renders `docs/images/adaptive_meshing_raw.png`, `benchmarks/meshing/results/adaptive_meshing_raw.png`, and `benchmarks/meshing/results/adaptive_meshing_accuracy.png`.

## Benchmarks

1. **`aniso_slab`** — 137 nm rotated anisotropic crystal slab (`eps = R(25°) diag(9, 12, 14) R(25°)^T + 0.8 (x z^T + z x^T)`) in air at 1.55 µm, normal incidence, `averaging="subpixel_full"` only, against the exact 4 × 4 Berreman transfer-matrix Jones transmission of `benchmarks/subpixel/berreman.py`.
2. **`sloped_guide`** — 500 × 220 nm silicon strip waveguide with 10° sidewalls in silica (6.40 × 3.20 × 3.20 µm box, `lambda = 1.55 um`), comparing `neff_mode` (and `neff_fdtd`) against the extrapolated `tidy3d` continuum reference (`2.381043` from `benchmarks/rectilinear/results/guide_tidy3d_reference.json`) and `uniform-12.5`.
3. **`soi_bragg`** — 10-period corrugated SOI sidewall Bragg grating (`Lambda = 320 nm`, widths `560 / 440 nm` on 500 nm waveguide, 220 nm Si on SiO2 box under air, 8.00 × 3.20 × 3.20 µm) over 1.48–1.60 µm (31 wavelengths), comparing spectral RMSE against `uniform-12.5` (41.94M cells).
4. **`elevator_coupler`** — Two-level SiN adiabatic elevator coupler (`overlap = 3.80 um`, 110 nm lower taper and 190 nm upper taper separated by a 100 nm vertical oxide gap, 50 nm tips, 19.20 × 4.00 × 3.20 µm box) at 619 nm, against `uniform-16` (60.00M cells).
5. **`directional_coupler`** — Broadband SiN directional coupler (400 × 200 nm waveguides, 300 nm gap over 20 µm, 43 × 6.4 × 2.6 µm box, 538–728 nm), comparing splitting ratio at 619 nm against the joint uniform continuum fit `S0 = 0.3558` (`benchmarks/rectilinear/README.md`).
6. **`sin_bend`** — 90° air-clad SiN waveguide bend (`R = 7 um`, `1.6 x 0.75 um`, on SiO2 substrate, 12.36 × 12.36 × 4.80 µm) at 1.55 µm against `uniform-24` (53.05M cells), comparing `UniformGrid`, default `bf.mesh`, `bf.mesh` with `Refine(material_cells={"silica": ...})`, and the inverse-design configuration `Refine(dilation={"WG": 0.45}, material_cells={"silica": 0.08})`.
