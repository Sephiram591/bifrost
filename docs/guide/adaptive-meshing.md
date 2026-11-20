# Adaptive meshing

`bf.mesh(stack, wavelength, ppw=..., refine=...)` builds a non-uniform tensor-product grid once from a device's `StackDevice`, a wavelength or `(min, max)` wavelength band in µm, a target grid density (`ppw`, points per local wavelength), and an optional `bf.Refine` configuration specifying where parameters may move. It returns `mesh.grid` for `stack.get_simulation_config(...)` and `mesh.boundary` for `fdtdx.boundary_objects_from_config(...)`.

Across photonic simulations—covering 1D anisotropic crystals, 10° sloped waveguides, Bragg gratings, multi-level taper couplers, directional couplers, and curved bends on substrates—`bf.mesh` cuts total grid cells by **2× to 4.8×** and cuts runtime by **2.2× to 10.9×** at matched accuracy compared to `fdtdx.core.grid.UniformGrid`.

## Convergence against grid cell count

The benchmarks below run in `float32` precision under both Kottke subpixel homogenization modes: `averaging="subpixel"` (the default; solid lines, filled markers) and `averaging="subpixel_full"` (dashed lines, hollow markers), compared in [Diagonal vs full-tensor averaging](#diagonal-vs-full-tensor-averaging). Values quoted in this section are for `averaging="subpixel"`. All runs used an NVIDIA L40S except the two largest `"subpixel_full"` directional-coupler grids (`UniformGrid` at 20 nm and `bf.mesh` at `ppw=16`), which exceed 48 GB and ran on an NVIDIA H200 (`benchmarks/meshing/accuracy_per_cell.py`). Each panel plots the simulated effective index, transmission, or splitting ratio directly against total cell count:

![Simulated effective index and transmission against grid cell count](../images/adaptive_meshing_raw.png)

- **(a) Sloped SOI strip waveguide: mode solver (10° sidewalls, 1.55 µm).** A 500 × 220 nm silicon strip in silica (6.40 × 3.20 × 3.20 µm box). On `UniformGrid`, sub-cell boundary crossings cause `neff_mode` to swing non-monotonically (2.3758 at 64 nm, 2.3654 at 50 nm, 2.3863 at 32 nm, 2.3845 at 20 nm, and 2.3801 at 12.5 nm with 33.55M cells). `bf.mesh` centers both core faces mid-cell and converges monotonically from 2.4062 at `ppw=8` through 2.3845 at `ppw=22` to 2.3823 at `ppw=34` (9.84M cells).
- **(b) Sloped SOI strip waveguide: FDTD phase slope (10° sidewalls, 1.55 µm).** Unwrapped `Ey` phase slope (`neff_fdtd`) in the 3D FDTD run. `UniformGrid` steps non-monotonically (2.4112 at 64 nm, 2.3863 at 50 nm, 2.3949 at 32 nm, 2.3879 at 20 nm with 8.19M cells in 16.73 s / 2.48 GB, and 2.3814 at 12.5 nm with 33.55M cells), whereas `bf.mesh` decreases smoothly from 2.4338 at `ppw=8` to 2.3837 at `ppw=34`, reaching 2.3879 at `ppw=22` (2.80M cells, 3.87 s, 1.46 GB)—matching 20 nm `UniformGrid` in both `neff_mode` (within 9e-5) and `neff_fdtd` (within 2e-5) with **2.9× fewer cells** and **4.3× shorter runtime**.
- **(c) Broadband SOI sidewall Bragg grating (1.48–1.60 µm, stop-band minimum).** A 10-period corrugated silicon waveguide (320 nm pitch, widths 560 / 440 nm, 8.00 × 3.20 × 3.20 µm box) across 31 wavelengths. On `UniformGrid`, the stop-band minimum `min T(λ)` oscillates across coarse grids (0.9176 at 50 nm, 0.9269 at 40 nm, 0.8929 at 25 nm, and a dip to 0.8349 at 20 nm with 10.24M cells in 48.98 s / 2.86 GB) before settling to 0.8680 at 16 nm (20.00M cells, 102.94 s) and 0.8640 at 12.5 nm (41.94M cells, 267.27 s). `bf.mesh` converges smoothly from 0.8867 at `ppw=18` (2.13M cells, 4.48 s, 1.29 GB) through 0.8752 at `ppw=22` to 0.8686 at `ppw=26` (6.22M cells, 23.50 s, 2.19 GB)—matching 16 nm `UniformGrid`'s stop-band depth in **3.2× fewer cells** and **4.4× shorter runtime** (and matching 12.5 nm `UniformGrid`'s full-band `T(λ)` within 4.8e-3 RMSE; `ppw=18` matches 20 nm `UniformGrid`'s full-band accuracy with **4.8× fewer cells** and **10.9× faster runtime**).
- **(d) Two-level SiN adiabatic elevator coupler (619 nm).** Upper (190 nm) and lower (110 nm) SiN tapers (650 to 50 nm wide, 3.80 µm overlap, 19.20 × 4.00 × 3.20 µm box) separated vertically by a 100 nm oxide gap. With all four horizontal SiN interfaces centered mid-cell, `bf.mesh` reaches T = 0.53447 at `ppw=10` (5.09M cells, 10.12 s, 1.29 GB) and plateaus within 1.1e-4 across `ppw=12` (8.64M, 0.53402), `ppw=15` (16.66M, 0.53398), and `ppw=18` (28.56M, 0.53409). By contrast, `UniformGrid` places the z = 0 gap wall on a cell edge and drifts slowly downward from 0.54016 at 40 nm through 0.53668 at 25 nm (15.73M cells, 50.59 s, 3.98 GB) to 0.53553 at 16 nm (60.00M cells, 289.85 s).
- **(e) Broadband SiN directional coupler (538–728 nm, 300 nm gap).** Two 400 × 200 nm SiN waveguides in oxide (43 × 6.4 × 2.6 µm box), showing the 619 nm cross-splitting ratio `S = P_cross / (P_cross + P_through)`. `UniformGrid` descends from 0.4513 at 50 nm (5.72M cells) and 0.4188 at 40 nm to 0.3778 at 25 nm (45.79M cells, 9.84 GB) and 0.3701 at 20 nm (89.44M cells, 344.05 s), while `bf.mesh` reaches 0.3968 at `ppw=6` (5.79M cells), 0.3754 at `ppw=8` (13.43M cells, 31.87 s, 3.29 GB—already past 25 nm `UniformGrid` with **3.4× fewer cells** and **3.0× less memory**), 0.3616 at `ppw=10` (26.34M cells, 78.29 s), and 0.3583 at `ppw=16` (106.58M cells).
- **(f) Curved SiN bend on a full-wafer silica substrate (1.55 µm).** A 7 µm radius 90° air-clad SiN bend on `WAFER` silica (12.36 × 12.36 × 4.80 µm box). Both `UniformGrid` (0.5636 at 60 nm, 0.5586 at 40 nm, 0.5571 at 30 nm in 27.16M cells / 92.40 s, and 0.5583 at 24 nm in 53.05M cells) and default `bf.mesh` (0.5755 at `ppw=12`, 0.5650 at `ppw=16`, 0.5583 at `ppw=20` in 9.05M cells / 23.67 s, and 0.5576 at `ppw=24` in 15.52M cells) settle around T ≈ 0.558. Coarsening the full-wafer silica level via `Refine(material_cells=...)` reduces cell count by 13 % (`ppw=20`: 7.87M cells, 18.18 s, 2.37 GB, T = 0.5584; `ppw=24`: 13.53M cells, 42.66 s, 3.74 GB, T = 0.5576) while tracking default `bf.mesh` within 1.5e-4—**2.0× fewer cells** and **2.2× faster runtime** than 30 nm `UniformGrid`. Adding a 0.45 µm optimization dilation around the moving bend (`Refine(dilation={"WG": 0.45}, material_cells={"silica": 0.08})` with 40 nm core cells) reaches T = 0.5586 in 7.45M cells (16.44 s, 2.28 GB, 35 % fewer cells than 40 nm `UniformGrid`).

The benchmark suite also tests a 1D rotated anisotropic crystal slab under `averaging="subpixel_full"` against the exact 4 × 4 Berreman transfer-matrix Jones transmission (`benchmarks/subpixel/berreman.py`), where `bf.mesh` at `ppw=28` (100 z cells, 9.57e-4 max `|t - t_Berreman|`) matches 10 nm `UniformGrid` (360 z cells, 9.80e-4) with **3.6× fewer cells**. (Off-diagonal permittivity tensors require `averaging="subpixel_full"`.)

### Summary of simulated values

| Simulation (metric) | `UniformGrid` (cells / value) | `bf.mesh` (cells / value) | Finest `UniformGrid` | Reduction (cells / time) |
| --- | --- | --- | --- | --- |
| Sloped SOI strip (`n_eff,FDTD`) | 20 nm: 8.19M / 2.3879 | `ppw=22`: 2.80M / 2.3879 | 12.5 nm (33.55M): 2.3814 | **2.9×** cells, **4.3×** faster |
| SOI Bragg grating (`min T(λ)`) | 16 nm: 20.00M / 0.8680 | `ppw=26`: 6.22M / 0.8686 | 12.5 nm (41.94M): 0.8640 | **3.2×** cells, **4.4×** faster |
| Elevator coupler (T at 619 nm) | 25 nm: 15.73M / 0.5367 | `ppw=10`: 5.09M / 0.5345 | 16 nm (60.00M): 0.5355 | **3.1×** cells, **5.0×** faster |
| Directional coupler (S at 619 nm) | 25 nm: 45.79M / 0.3778 | `ppw=8`: 13.43M / 0.3754 | 20 nm (89.44M): 0.3701 | **3.4×** cells, **4.5×** faster |
| SiN bend (T at 1.55 µm) | 30 nm: 27.16M / 0.5571 | `ppw=24` + `material_cells`: 13.53M / 0.5576 | 24 nm (53.05M): 0.5583 | **2.0×** cells, **2.2×** faster |

## Convergence against peak VRAM

The six panels below plot the same benchmark runs against process-isolated peak live XLA device memory (`jax.devices()[0].memory_stats()["peak_bytes_in_use"]`, in decimal GB, `10^9` bytes), measured in fresh worker subprocesses with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.97` so compilation caches do not carry across runs. All `averaging="subpixel"` runs and most `averaging="subpixel_full"` runs were measured on an NVIDIA L40S (48 GB); the four largest `"subpixel_full"` configurations—the 16 nm elevator coupler (36.18 GB live peak, exceeding 48 GB during setup) and the `ppw=14`, 20 nm, and `ppw=16` directional couplers (43.32 GB, 53.87 GB, and 64.14 GB)—were measured on an NVIDIA B200.

![Simulated effective index and transmission against peak VRAM](../images/adaptive_meshing_vram.png)

- **VRAM reduction at matched accuracy.** Under `averaging="subpixel"`, `bf.mesh` cuts peak GPU memory by **1.7× to 3.3×** at matched accuracy: 2.48 → 1.46 GB (**1.7×**) on the sloped SOI strip (`ppw=22` vs 20 nm), 5.01 → 2.19 GB (**2.3×**) on the SOI Bragg grating (`ppw=26` vs 16 nm; and 2.86 → 1.29 GB, **2.2×**, at `ppw=18` vs 20 nm), 3.98 → 1.29 GB (**3.1×**) on the elevator coupler (`ppw=10` vs 25 nm), 9.84 → 3.29 GB (**3.0×**) on the directional coupler (`ppw=8` vs 25 nm; and 19.05 → 5.84 GB, **3.3×**, at `ppw=10` vs 20 nm), and 6.46 → 3.74 GB (**1.7×**) on the SiN bend (`ppw=24` + `material_cells` vs 30 nm). Under `averaging="subpixel_full"`, the reduction is **1.9× to 4.4×**.
- **Baseline overhead vs linear per-cell scaling.** Below ~1–2M cells, XLA compilation workspaces and 2D port mode solves create a ~0.25–0.9 GB floor. Above a few million cells, peak VRAM scales linearly with cell count at **0.21–0.25 GB per million cells** under `"subpixel"` and **0.60–0.62 GB per million cells** under `"subpixel_full"`, so cell-count reductions translate almost directly into VRAM savings.
- **Horizontal separation of `"subpixel"` and `"subpixel_full"`.** Unlike the cell-count panels where both modes share the same x-coordinates, plotting against peak VRAM shifts the dashed `"subpixel_full"` curves rightward. On four of the five 3D devices, `bf.mesh` with full-tensor `"subpixel_full"` uses **less peak VRAM than `UniformGrid` with diagonal `"subpixel"`** at matched accuracy—for example, 8.52 GB at `ppw=8` (`"subpixel_full"`) vs 9.84 GB at 25 nm (`"subpixel"`) on the directional coupler, 3.10 GB vs 3.98 GB on the elevator coupler, 3.92 GB vs 5.01 GB on the SOI Bragg grating, and 1.74 GB vs 2.48 GB on the sloped SOI strip.

### Peak VRAM at matched accuracy

| Simulation (metric) | `UniformGrid` (`subpixel`) | `bf.mesh` (`subpixel`) | Finest `UniformGrid` (`subpixel`) | Reduction (`subpixel` / `subpixel_full`) |
| --- | --- | --- | --- | --- |
| Sloped SOI strip (`n_eff,FDTD`) | 20 nm: 2.48 GB | `ppw=22`: 1.46 GB | 12.5 nm: 8.48 GB | **1.7×** / **3.0×** |
| SOI Bragg grating (`min T(λ)`) | 16 nm: 5.01 GB | `ppw=26`: 2.19 GB | 12.5 nm: 10.29 GB | **2.3×** / **3.1×** |
| Elevator coupler (T at 619 nm) | 25 nm: 3.98 GB | `ppw=10`: 1.29 GB | 16 nm: 13.66 GB | **3.1×** / **3.1×** |
| Directional coupler (S at 619 nm) | 25 nm: 9.84 GB | `ppw=8`: 3.29 GB | 20 nm: 19.05 GB | **3.0×** / **3.2×** |
| SiN bend (T at 1.55 µm) | 30 nm: 6.46 GB | `ppw=24` + `material_cells`: 3.74 GB | 24 nm: 12.26 GB | **1.7×** / **1.9×** |

## Diagonal vs full-tensor averaging

`averaging="subpixel"` writes the diagonal of each voxel's Kottke-averaged permittivity tensor. `averaging="subpixel_full"` also writes the off-diagonal entries produced by interfaces oblique to the grid and runs `fdtdx`'s fully anisotropic update throughout the simulation. In the benchmarks below (dashed lines and hollow markers in the figures above), mode sources and monitors solve on the diagonal of the averaged tensor (since `tidy3d`'s mode solver requires `tidy3d_extras` for fully anisotropic media) while the 3D FDTD update uses the complete 3 × 3 tensor.

Shifts are `"subpixel_full"` minus `"subpixel"` on the same grid. Run-time and peak-memory ratios are medians (with ranges) over the grids where both modes ran on the same GPU type; peak memory ratios compare runs under JAX's BFC allocator (`XLA_PYTHON_CLIENT_PREALLOCATE=false`).

| Simulation (metric) | Shift, all grids | Shift, finest `UniformGrid` | Shift, finest `bf.mesh` | Run time, full / diagonal | Peak memory, full / diagonal |
| --- | --- | --- | --- | --- | --- |
| Anisotropic slab (Berreman error) | — (`"subpixel"` rejects off-diagonal permittivity) | — | — | — | — |
| Sloped SOI strip (`n_eff,FDTD`) | -1.3e-4 to +4.0e-4 | +1.2e-4 (12.5 nm) | +1.0e-4 (`ppw=34`) | 4.5× (2.1–6.7×) | 1.14× (0.92–2.46×) |
| SOI Bragg grating (`min T(λ)`) | -6.3e-2 to -5.7e-3 | -9.2e-3 (12.5 nm; full-band RMS 6.5e-3) | -1.1e-2 (`ppw=26`) | 4.8× (1.7–6.2×) | 1.15× (1.02–2.50×) |
| Elevator coupler (T at 619 nm) | +5.2e-4 to +1.2e-2 | +5.2e-4 (16 nm) | +5.8e-4 (`ppw=18`) | 5.3× (2.4–6.7×) | 2.41× (1.89–2.77×) |
| Directional coupler (S at 619 nm) | -8.9e-4 to +3.4e-3 | -8.9e-4 (20 nm) | -2.3e-5 (`ppw=16`) | 6.0× (2.9–8.1×) | 2.65× (0.74–2.89×) |
| SiN bend (T at 1.55 µm) | +4.8e-4 to +1.5e-3 | +5.7e-4 (24 nm) | +5.6e-4 (`ppw=24`) | 5.3× (2.3–6.4×) | 2.03× (0.51–2.63×) |

- **When to choose `"subpixel"` vs `"subpixel_full"`.** Use the default `averaging="subpixel"` for isotropic and diagonally anisotropic devices: `"subpixel_full"` adds 4.5–6.0× median runtime and 1.1–2.7× median peak GPU memory across the full 3D FDTD update. Switch to `averaging="subpixel_full"` when materials carry off-diagonal permittivity entries (such as rotated crystals) or when off-diagonal coupling at sub-cell oblique corners matters.
- **Where the two modes differ.** The largest differences appear on coarse grids with strong corrugated or stepped interfaces—the SOI Bragg grating's `min T(λ)` shifts by 0.038–0.063 at 40–64 nm and 0.009–0.018 at 12.5–20 nm, and the elevator coupler's T shifts by 1.2e-2 at 80 nm down to 5.2e-4 at 16 nm. On the directional coupler and the SiN bend, the shift stays below 1e-3 on fine uniform grids and across `bf.mesh` (within 4.1e-4 on the directional coupler and 4.8e-4 to 9.4e-4 on the bend). On the finest grids overall, the two averaging modes differ by roughly the same amount (5e-4 to 9e-4 on the smooth devices, 6.5e-3 full-band RMS on the Bragg grating) as the discretization differences between `bf.mesh` and fine uniform grids.
- **Interface placement vs off-diagonal averaging.** `UniformGrid` keeps the same non-monotonic oscillations under `"subpixel_full"` (`n_eff,FDTD` on the sloped strip steps 2.4110 → 2.3867 → 2.3951 → 2.3881, and the Bragg grating's `min T(λ)` still dips to 0.8262 at 20 nm): where material interfaces fall within grid cells—which `bf.mesh` controls via `center_faces=True`—rather than omitted off-diagonal tensor entries drives those swings. `bf.mesh` achieves the same cell-count and runtime reductions under both modes:

### Summary of simulated values under `averaging="subpixel_full"`

| Simulation (metric) | `UniformGrid` (cells / value) | `bf.mesh` (cells / value) | Finest `UniformGrid` | Reduction (cells / time) |
| --- | --- | --- | --- | --- |
| Sloped SOI strip (`n_eff,FDTD`) | 20 nm: 8.19M / 2.3881 | `ppw=22`: 2.80M / 2.3881 | 12.5 nm (33.55M): 2.3816 | **2.9×** cells, **3.6×** faster |
| SOI Bragg grating (`min T(λ)`) | 16 nm: 20.00M / 0.8498 | `ppw=26`: 6.22M / 0.8572 | 12.5 nm (41.94M): 0.8549 | **3.2×** cells, **3.6×** faster |
| Elevator coupler (T at 619 nm) | 25 nm: 15.73M / 0.5379 | `ppw=10`: 5.09M / 0.5366 | 16 nm (60.00M): 0.5361 | **3.1×** cells, **4.0×** faster |
| Directional coupler (S at 619 nm) | 25 nm: 45.79M / 0.3777 | `ppw=8`: 13.43M / 0.3753 | 20 nm (89.44M): 0.3692 | **3.4×** cells, **7.0×** faster |
| SiN bend (T at 1.55 µm) | 30 nm: 27.16M / 0.5577 | `ppw=24` + `material_cells`: 13.53M / 0.5581 | 24 nm (53.05M): 0.5588 | **2.0×** cells, **2.4×** faster |

## Why adaptive meshing wins per cell

Three mechanisms in `bf.mesh` work together so fewer cells achieve equal or lower error:

1. **Material-scaled spacing plus `3l` evanescent tails (`ppw`, `global_dilation`, `global_z_padding`).** Every region asks for cells no wider than `lambda / (n ppw)`, grading smoothly outward with neighbor ratio at most `growth=1.25`. Fine cells extend past every high-index boundary into lower-index cladding by `3` evanescent decay lengths `l = lambda_max / (2 pi sqrt(n_high^2 - n_low^2))` by default (`global_dilation=3.0`, `global_z_padding=3.0`). On the sloped SOI waveguide, stopping fine cells right at the silicon boundary (0 decay lengths) loses -3.3e-3 in `neff`—as much as doubling the uniform cell size—whereas 1, 2, and 3 decay lengths reduce that error to -5.8e-4, -1.4e-4, and -3.7e-5 (using only 31 % of the 20 nm uniform grid's cells at `3l`).
2. **Planar interfaces centered mid-cell (`center_faces=True`).** In Yee grids, horizontal material interfaces on primal cell edges coincide with `Ex` and `Ey` dual-cell faces, creating an asymmetric O(hz^2) averaging error that can be three times larger than placing those interfaces at exact cell midpoints (0.500 of a z cell). On the directional coupler, `center_faces=True` cuts error up to **2.6×** at fixed `ppw` (for example, 8.6e-3 down to 3.3e-3 at `ppw=17`) and keeps `neff_mode` monotone on the sloped waveguide instead of jumping across sub-cell boundary crossings.
3. **Automatic resolution of thin tips and narrow gaps (`feature_cells=1.5`).** KLayout width and space checks on each level's starting geometry automatically target at most `w / 1.5` locally around narrow tapers and gaps, such as the elevator coupler's 50 nm taper tips (producing 33.3 nm minimum spacing even at `ppw=6` and `ppw=8`), without refining the rest of the cladding or waveguide bulk. Use `min_cell=` when a sub-wavelength film or sliver would otherwise force an unnecessarily small time step.

## Practical trade-offs and tuning

- **Memory savings are unconditional; time-step savings follow GPU saturation.** Because the smallest cell anywhere in the domain sets the global CFL time step `dt`, grading saves **cells, field memory, and adjoint-checkpoint memory** proportionally while keeping roughly the same number of time steps as a uniform grid at the core spacing. On small domains below ~2–3M cells (such as 1D slabs or coarse strip waveguides), kernel launch latency dominates and wall times are similar; once grids exceed a few million cells and saturate GPU compute (Bragg gratings, couplers, and 3D bends), wall time drops nearly in proportion to cell count (4×–11× faster at matched accuracy).
- **Coarsen full-wafer substrates on tensor-product grids with `Refine.material_cells`.** Because a rectilinear mesh is a tensor product of 1D axis edge arrays, each coordinate takes the finest target anywhere in its cross-sectional plane. A `WAFER` substrate or uniform cladding spanning all x and y planes caps the coarse lateral spacing everywhere at `lambda / (n_sub ppw)`. Coarsening low-contrast substrates with `material_cells` to ~13 cells per wavelength recovers lateral grading in margins and PMLs without hurting accuracy, and combines directly with optimization `dilation`:

```python
import fdtdx
import gdsfactory as gf
from gdsfactory.technology import LayerLevel, LayerStack
import bifrost as bf

gf.gpdk.PDK.activate()
component = gf.components.bend_circular(radius=5, width=0.5)
layer_stack = LayerStack(
    layers={
        "box": LayerLevel(layer="WAFER", zmin=-3.0, thickness=3.0, material="silica", mesh_order=9),
        "core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=0.22, material="si", mesh_order=1),
    }
)
materials = {
    "si": fdtdx.Material(permittivity=3.48**2),
    "silica": fdtdx.Material(permittivity=1.444**2),
    "air": fdtdx.Material(permittivity=1.0),
}
stack = bf.StackDevice.from_layer_stack(
    component, layer_stack, materials, background="air", zrange_layers=["core"],
    extend_ports=(("o1", 2.0), ("o2", 2.0)), boundary_offset=(1.5, 0.0, 0.0, -1.5, 1.4, -1.4), name="stack",
)

# 1.55 um band; allow 0.35 um boundary motion during inverse design while coarsening the wafer box
mesh = bf.mesh(
    stack, (1.50, 1.60), ppw=18,
    refine=bf.Refine(dilation={"WG": 0.35}, material_cells={"silica": 0.08}),
)
config = stack.get_simulation_config(time=300e-15, grid=mesh.grid)
source, src_placing = stack.create_mode_plane_source_at(
    "o1", (0.5, 0.5), wave_character=fdtdx.WaveCharacter(wavelength=1.55e-6), grid=config,
)
```

For full placement rules (`grid=config` on port monitors and `mesh.boundary`), see [Simulation setup and ports](simulation-setup.md#automatic-meshing); for every parameter of `mesh`, `Refine`, and `CoarseVertices`, see [`bf.meshing`](../api/meshing.md).
