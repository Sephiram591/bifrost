# Adaptive meshing

`bf.mesh` builds a non-uniform tensor-product grid once from the device's `StackDevice`, the simulated wavelength band, and an optional statement of where parameters may move (`Refine`). Across photonic simulations—covering 1D anisotropic crystals, 10° sloped waveguides, Bragg gratings, multi-level taper couplers, directional couplers, and curved bends on substrates—`bf.mesh` cuts total grid cells by **2× to 4.8×** and cuts runtime by **2.2× to 10.9×** at matched accuracy compared to `fdtdx.core.grid.UniformGrid`.

## Convergence against grid cell count

All benchmarks below use bifrost's default Green's-theorem voxelizer (`voxelizer="greens"`) with float32 fields and run under both Kottke subpixel homogenization modes: `averaging="subpixel"` (bifrost's default; solid lines, filled markers) and `averaging="subpixel_full"` (dashed lines, hollow markers), compared in [Diagonal vs full-tensor averaging](#diagonal-vs-full-tensor-averaging). Values quoted in this section are for `averaging="subpixel"`. All runs used an NVIDIA L40S except the two largest `"subpixel_full"` directional-coupler grids (`uniform-20` and `mesh-16`), which exceed its 48 GB and ran on an NVIDIA H200 (`benchmarks/meshing/accuracy_per_cell.py`). Rather than subtracting a chosen reference, each panel plots the simulated effective index, transmission, or splitting ratio directly against total cell count:

![Simulated effective index and transmission against grid cell count](../images/adaptive_meshing_raw.png)

- **(a) Sloped SOI strip waveguide: mode solver (10° sidewalls, 1.55 µm).** A 500 × 220 nm silicon strip in silica (6.40 × 3.20 × 3.20 µm box). On `UniformGrid`, sub-cell boundary crossings swing `neff_mode` non-monotonically (`2.3758` at `64 nm` → `2.3654` at `50 nm` → `2.3863` at `32 nm` → `2.3845` at `20 nm` → `2.3794` at `16 nm` → `2.3801` at `12.5 nm`, `33.55M` cells); `bf.mesh` centers both core faces mid-cell and converges monotonically (`2.4062` at `ppw=8` → `2.3845` at `ppw=22` → `2.3823` at `ppw=34`, `9.84M` cells).
- **(b) Sloped SOI strip waveguide: FDTD phase slope (10° sidewalls, 1.55 µm).** Unwrapped `Ey` phase slope (`neff_fdtd`) in the 3D FDTD run. `UniformGrid` steps non-monotonically (`2.4112` at `64 nm` → `2.3863` at `50 nm` → `2.3949` at `32 nm` → `2.3879` at `20 nm`, `8.19M` cells, `16.73 s`, `2.48 GB` → `2.3814` at `12.5 nm`, `33.55M` cells), whereas `bf.mesh` decreases smoothly (`2.4338` at `ppw=8` to `2.3837` at `ppw=34`) and reaches `2.3879` at `mesh-22` (`2.80M` cells, `3.87 s`, `1.46 GB`)—matching `uniform-20` in both `neff_mode` (`9e-5`) and `neff_fdtd` (`2e-5`) with **2.9× fewer cells** and **4.3× shorter runtime**.
- **(c) Broadband SOI sidewall Bragg grating (1.48–1.60 µm, stop-band minimum).** A 10-period corrugated silicon waveguide (`Lambda = 320 nm`, widths `560 / 440 nm`, 8.00 × 3.20 × 3.20 µm box) across 31 wavelengths. On `UniformGrid`, the stop-band minimum `min T(λ)` oscillates (`0.9176` at `50 nm` → `0.9269` at `40 nm` → `0.8929` at `25 nm`, `5.24M` cells → `0.8349` at `20 nm`, `10.24M` cells, `48.98 s`, `2.86 GB`) before settling to `0.8680` (`16 nm`, `20.00M` cells, `102.94 s`) and `0.8640` (`12.5 nm`, `41.94M` cells, `267.27 s`). `bf.mesh` converges smoothly from `0.8867` at `mesh-18` (`2.13M` cells, `4.48 s`, `1.29 GB`) to `0.8752` at `mesh-22` (`3.75M` cells, `10.22 s`) and `0.8686` at `mesh-26` (`6.22M` cells, `23.50 s`, `2.19 GB`)—matching `uniform-16`'s stop-band depth in **3.2× fewer cells** and **4.4× shorter runtime** (and matching `uniform-12.5`'s full-band `T(λ)` within `4.8e-3` RMSE; `mesh-18` beats `uniform-20`'s full-band RMSE with **4.8× fewer cells** and **10.9× faster runtime**).
- **(d) Two-level SiN adiabatic elevator coupler (619 nm).** Upper (`190 nm`) and lower (`110 nm`) SiN tapers (`650 -> 50 nm`) separated vertically by a `100 nm` oxide gap (`overlap = 3.80 um`, 19.20 × 4.00 × 3.20 µm box). With all four horizontal SiN interfaces centered mid-cell, `bf.mesh` reaches `T = 0.53447` at `mesh-10` (`5.09M` cells, `10.12 s`, `1.29 GB`) and plateaus within `1.1e-4` across `mesh-12` (`8.64M`, `0.53402`), `mesh-15` (`16.66M`, `0.53398`), and `mesh-18` (`28.56M`, `0.53409`), whereas `UniformGrid` places the `z = 0` gap wall on a cell edge and drifts slowly downward (`0.54016` at `40 nm` → `0.53979` at `32 nm`, `7.50M` cells → `0.53668` at `25 nm`, `15.73M` cells, `50.59 s`, `3.98 GB` → `0.53641` at `20 nm` → `0.53553` at `16 nm`, `60.00M` cells, `289.85 s`).
- **(e) Broadband SiN directional coupler (538–728 nm, 300 nm gap).** Two 400 × 200 nm SiN waveguides in oxide (43 × 6.4 × 2.6 µm box), showing the 619 nm cross-splitting ratio `S = P_cross / (P_cross + P_through)`. `UniformGrid` descends from `0.4513` (`50 nm`, `5.72M` cells) → `0.4188` (`40 nm`) → `0.3778` (`25 nm`, `45.79M` cells, `9.84 GB`) → `0.3701` (`20 nm`, `89.44M` cells, `344.05 s`), while `bf.mesh` reaches `0.3968` at `mesh-6` (`5.79M` cells), `0.3754` at `mesh-8` (`13.43M` cells, `31.87 s`, `3.29 GB`—already past `uniform-25` with **3.4× fewer cells** and **3.0× less memory**), `0.3616` at `mesh-10` (`26.34M` cells, `78.29 s`), `0.3592` at `mesh-14` (`71.97M` cells), and `0.3583` at `mesh-16` (`106.58M` cells).
- **(f) Curved SiN bend on a full-wafer silica substrate (1.55 µm).** A 7 µm radius 90° air-clad SiN bend on `WAFER` silica (12.36 × 12.36 × 4.80 µm box). Both `UniformGrid` (`0.5636` at `60 nm` → `0.5586` at `40 nm`, `11.46M` cells, `29.97 s` → `0.5571` at `30 nm`, `27.16M` cells, `92.40 s` → `0.5583` at `24 nm`, `53.05M` cells) and `bf.mesh` (`0.5755` at `ppw=12` → `0.5650` at `ppw=16` → `0.5583` at `ppw=20`, `9.05M` cells, `23.67 s` → `0.5576` at `ppw=24`, `15.52M` cells) settle around `T ≈ 0.558`. Coarsening the full-wafer silica level via `Refine(material_cells=...)` reduces cells by 13 % (`meshsub-20`: `7.87M` cells, `18.18 s`, `2.37 GB`, `T = 0.5584`; `meshsub-24`: `13.53M` cells, `42.66 s`, `3.74 GB`, `T = 0.5576`) while tracking default `bf.mesh` within `1.5e-4`—**2.0× fewer cells** and **2.2× faster runtime** than `uniform-30`; `meshclad-40-80` adds a `0.45` µm optimization dilation around the moving bend and still reaches `T = 0.5586` in `7.45M` cells (`16.44 s`, `2.28 GB`, 35 % fewer cells than `uniform-40`).

The benchmark suite also tests a 1D rotated anisotropic crystal slab (`averaging="subpixel_full"`, against the exact 4 × 4 Berreman transfer-matrix Jones transmission in `benchmarks/subpixel/berreman.py`), where `mesh-28` (`100` `z` cells, `9.57e-4` max `|t - t_Berreman|`) matches `uniform-10` (`360` `z` cells, `9.80e-4`) with **3.6× fewer cells**. This benchmark runs only with `"subpixel_full"`: `averaging="subpixel"` rejects materials with off-diagonal permittivity entries.

### Summary of simulated values

| Simulation (metric) | `UniformGrid` (cells / value) | `bf.mesh` (cells / value) | Finest `UniformGrid` | Reduction (cells / time) |
| --- | --- | --- | --- | --- |
| Sloped SOI strip (`n_eff,FDTD`) | `uniform-20`: `8.19M` / `2.3879` | `mesh-22`: `2.80M` / `2.3879` | `12.5 nm` (`33.55M`): `2.3814` | **2.9×** cells, **4.3×** faster |
| SOI Bragg grating (`min T(λ)`) | `uniform-16`: `20.00M` / `0.8680` | `mesh-26`: `6.22M` / `0.8686` | `12.5 nm` (`41.94M`): `0.8640` | **3.2×** cells, **4.4×** faster |
| Elevator coupler (`T` at 619 nm) | `uniform-25`: `15.73M` / `0.5367` | `mesh-10`: `5.09M` / `0.5345` | `16 nm` (`60.00M`): `0.5355` | **3.1×** cells, **5.0×** faster |
| Directional coupler (`S` at 619 nm) | `uniform-25`: `45.79M` / `0.3778` | `mesh-8`: `13.43M` / `0.3754` | `20 nm` (`89.44M`): `0.3701` | **3.4×** cells, **4.5×** faster |
| SiN bend (`T` at 1.55 µm) | `uniform-30`: `27.16M` / `0.5571` | `meshsub-24`: `13.53M` / `0.5576` | `24 nm` (`53.05M`): `0.5583` | **2.0×** cells, **2.2×** faster |

## Diagonal vs full-tensor averaging

`averaging="subpixel"` writes the diagonal of each voxel's Kottke-averaged permittivity tensor. `averaging="subpixel_full"` also writes the off-diagonal entries that walls oblique to the grid produce, and runs fdtdx's fully anisotropic update throughout the simulation. Every benchmark above was repeated with `"subpixel_full"` (dashed lines and hollow markers in the figure). In those runs, mode sources and monitors are solved on the diagonal of the averaged tensor, because tidy3d's mode solver needs `tidy3d_extras` for fully anisotropic media; the FDTD update itself uses the complete tensor.

Shifts are `"subpixel_full"` minus `"subpixel"` on the same grid. Run-time and peak-memory ratios are medians (with ranges) over the grids both modes ran on the same GPU type; peak memory is reported only for grids run with JAX's default allocator.

| Simulation (metric) | Shift, all grids | Shift, finest `UniformGrid` | Shift, finest `bf.mesh` | Run time, full / diagonal | Peak memory, full / diagonal |
| --- | --- | --- | --- | --- | --- |
| Anisotropic slab (Berreman error) | — (`"subpixel"` rejects the off-diagonal material) | — | — | — | — |
| Sloped SOI strip (`n_eff,FDTD`) | `-1.3e-4` to `+4.0e-4` | `+1.2e-4` (`12.5 nm`) | `+1.0e-4` (`ppw=34`) | `4.5×` (`2.1–6.7×`) | `1.14×` (`0.92–2.42×`) |
| SOI Bragg grating (`min T(λ)`) | `-6.3e-2` to `-5.7e-3` | `-9.2e-3` (`12.5 nm`; full-band RMS `6.5e-3`) | `-1.1e-2` (`ppw=26`) | `4.8×` (`1.7–6.2×`) | `1.12×` (`1.02–2.21×`) |
| Elevator coupler (`T` at 619 nm) | `+5.2e-4` to `+1.2e-2` | `+5.2e-4` (`16 nm`) | `+5.8e-4` (`ppw=18`) | `5.3×` (`2.4–6.7×`) | `2.38×` (`1.89–2.77×`) |
| Directional coupler (`S` at 619 nm) | `-8.9e-4` to `+3.4e-3` | `-8.9e-4` (`20 nm`) | `-2.3e-5` (`ppw=16`) | `6.0×` (`2.9–8.1×`) | `2.02×` (`0.74–2.89×`) |
| SiN bend (`T` at 1.55 µm) | `+4.8e-4` to `+1.5e-3` | `+5.7e-4` (`24 nm`) | `+5.6e-4` (`ppw=24`) | `5.3×` (`2.3–6.4×`) | `1.19×` (`0.51–2.31×`) |

- **The averaging mode does not cause the `UniformGrid` oscillations.** Under `"subpixel_full"` the uniform sequences keep their shape: the sloped strip's `n_eff,FDTD` still steps `2.4110` (`64 nm`) → `2.3867` (`50 nm`) → `2.3951` (`32 nm`) → `2.3881` (`20 nm`), within `4e-4` of the diagonal values, and the Bragg grating's `min T(λ)` still dips at `20 nm` (`0.8262`, against `0.8498` at `16 nm`). Where interfaces fall within cells, not the dropped off-diagonal entries, drives those swings.
- **The shift is largest on the Bragg grating and on coarse elevator-coupler grids, and shrinks with refinement there.** The Bragg grating's `min T(λ)` moves by `0.038–0.063` at `40–64 nm` and by `0.009–0.018` at `12.5–20 nm`; the elevator coupler's `T` moves by `1.2e-2` at `80 nm` and `5.2e-4` at `16 nm`. On the directional coupler the shift stays at or below `4.1e-4` on every `bf.mesh` grid and on `uniform-25` through `uniform-50`; on the bend it stays between `4.8e-4` and `9.4e-4` from `7M` to `53M` cells, comparable to the bend's own `uniform-30`-to-`uniform-24` spread (`1.1e-3`).
- **The `bf.mesh`-versus-`UniformGrid` comparisons hold under both modes** (table below). The sloped strip's `mesh-22` still matches `uniform-20` in `n_eff,FDTD` to `2e-5`. The elevator coupler's gap between `bf.mesh` and the finest uniform grid also persists: under `"subpixel"`, `bf.mesh` plateaus at `0.5340–0.5341` against `uniform-16`'s `0.5355`; under `"subpixel_full"`, `bf.mesh` keeps descending (`0.5366` at `ppw=10` → `0.5347` at `ppw=18`) against `uniform-16`'s `0.5361`, a `1.4e-3` gap in both modes.
- **The averaging choice moves fine-grid values about as much as the differences the summary tables compare.** On the finest uniform grids the two modes differ by `6.5e-3` full-band RMS on the Bragg grating (against the `4.8e-3` RMSE match quoted for `mesh-26` above), `5.2e-4` on the elevator coupler, `8.9e-4` on the directional coupler, and `5.7e-4` on the bend.
- **`"subpixel_full"` costs `4.5–6.0×` the median run time and `1.1–2.4×` the peak memory.** `"subpixel"` remains the default; `"subpixel_full"` is required for materials with off-diagonal permittivity, such as the anisotropic slab.

### Summary of simulated values under `averaging="subpixel_full"`

| Simulation (metric) | `UniformGrid` (cells / value) | `bf.mesh` (cells / value) | Finest `UniformGrid` | Reduction (cells / time) |
| --- | --- | --- | --- | --- |
| Sloped SOI strip (`n_eff,FDTD`) | `uniform-20`: `8.19M` / `2.3881` | `mesh-22`: `2.80M` / `2.3881` | `12.5 nm` (`33.55M`): `2.3816` | **2.9×** cells, **3.6×** faster |
| SOI Bragg grating (`min T(λ)`) | `uniform-16`: `20.00M` / `0.8498` | `mesh-26`: `6.22M` / `0.8572` | `12.5 nm` (`41.94M`): `0.8549` | **3.2×** cells, **3.6×** faster |
| Elevator coupler (`T` at 619 nm) | `uniform-25`: `15.73M` / `0.5379` | `mesh-10`: `5.09M` / `0.5366` | `16 nm` (`60.00M`): `0.5361` | **3.1×** cells, **4.0×** faster |
| Directional coupler (`S` at 619 nm) | `uniform-25`: `45.79M` / `0.3777` | `mesh-8`: `13.43M` / `0.3753` | `20 nm` (`89.44M`): `0.3692` | **3.4×** cells, **7.0×** faster |
| SiN bend (`T` at 1.55 µm) | `uniform-30`: `27.16M` / `0.5577` | `meshsub-24`: `13.53M` / `0.5581` | `24 nm` (`53.05M`): `0.5588` | **2.0×** cells, **2.4×** faster |

## Why adaptive meshing wins per cell

Three mechanisms in `bf.mesh` work together so fewer cells achieve equal or lower error:

1. **Material-scaled spacing plus `3l` evanescent tails (`ppw`, `global_dilation`, `global_z_padding`).** Every region asks for cells no wider than `lambda / (n ppw)`, grading smoothly outward with neighbor ratio at most `growth=1.25`. Crucially, fine cells extend past every high-index boundary into lower-index cladding by `3` evanescent decay lengths `l = lambda_max / (2 pi sqrt(n_high^2 - n_low^2))` by default (`global_dilation=3.0`, `global_z_padding=3.0`). On the sloped SOI waveguide (`benchmarks/rectilinear/results/guide_tails.json`), stopping fine cells right at the silicon boundary (`0` decay lengths) loses `-3.3e-3` in `neff`—as much as doubling the uniform cell size—whereas `1`, `2`, and `3` decay lengths cut that error to `-5.8e-4`, `-1.4e-4`, and `-3.7e-5` (with only 31 % of the uniform 20 nm grid's cells at `3`).
2. **Planar interfaces centered mid-cell (`center_faces=True`).** In Yee grids, horizontal material interfaces on primal cell edges coincide with `Ex` and `Ey` dual-cell faces, creating an asymmetric `O(hz^2)` averaging error that can be three times larger than placing those interfaces at exact cell midpoints (`0.500` of a z cell). On the directional coupler (`benchmarks/rectilinear/README.md`), `center_faces=True` cuts error up to **2.6×** at fixed `ppw` (e.g., `8.6e-3` down to `3.3e-3` at `ppw=17`) and keeps `neff_mode` monotone on the sloped waveguide instead of jumping across sub-cell boundary crossings.
3. **Automatic resolution of thin tips and narrow gaps (`feature_cells=1.5`).** KLayout width and space checks on each level's starting geometry automatically target at most `w / 1.5` locally around narrow tapers and gaps, such as the elevator coupler's 50 nm taper tips (`min_dx = 33.3 nm` even at `ppw=6` and `ppw=8`), without refining the rest of the cladding or waveguide bulk. Use `min_cell=` when a sub-wavelength film or sliver would otherwise force an unnecessarily small time step.

## Practical trade-offs and tuning

- **Memory savings are unconditional; time-step savings follow GPU saturation.** Because the smallest cell anywhere in the domain sets the global CFL time step `dt`, grading saves **cells, field memory, and adjoint-checkpoint memory** proportionally while keeping roughly the same number of time steps as a uniform grid at the core spacing. On small domains below ~2–3M cells (`aniso_slab`, coarse `sloped_guide`), kernel launch latency dominates and wall times are similar; once grids exceed a few million cells and saturate GPU compute (`soi_bragg`, `elevator_coupler`, `directional_coupler`, `sin_bend`), wall time drops nearly in proportion to cell count (4×–11× faster at matched accuracy).
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
