# Harmonic inversion (`bf.harminv`) vs. Meep (`libharminv`)

Benchmarks comparing `bifrost.harminv` / `bifrost.harminv_detector` (JAX CPU and GPU) against Meep's C++ `libharminv` (`meep.py_do_harminv` and native `meep.Simulation` + `mp.Harminv`) across accuracy regimes, runtime and autodiff scaling, multi-channel spatial detectors, and an end-to-end differentiable 1D Fabry–Pérot resonator simulation.

## Hardware & Environment

Measured in isolated worker subprocesses on a compute node with:

- **CPU:** Intel Xeon Platinum 8562Y+ (`JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`)
- **GPU:** NVIDIA L40S (`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `JAX_ENABLE_X64=1`)
- **Software:** Python `3.12.14`, JAX `0.11.2`, `fdtdx` `0.6.3`, Meep `1.34.0`, `bifrost` `0.2.0`

## Reproducing

Run on a GPU compute node (or via SLURM `srun`):

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 python benchmarks/harminv/bench_harminv.py
```

Each suite (`accuracy`, `scaling_meep`, `scaling_cpu`, `scaling_gpu`, `fdtd_meep`, `fdtd_bifrost`) spawns a fresh worker process so XLA compilation, CUDA memory, and OpenMPI state stay isolated. Raw structured metrics are written to `benchmarks/harminv/results/bench_harminv.json`.

---

## 1. Accuracy vs. Analytical Ground Truth

All accuracy tests run in `float64` (`JAX_ENABLE_X64=1`) on identical real-valued time traces ($\Delta t = 0.4\text{ fs}$). `meep.py_do_harminv` returns accurate amplitude magnitudes (`max_amp_mag_rel`), while on real-valued signals it flips between returning $a_k$ and $\bar{a}_k$ across $Q$ and record lengths (visible in `max_amp_rel` vs. `max_amp_mag_rel` below); `bf.harminv` preserves consistent `exp(-i omega t)` phasor signs throughout.

### Clean 3-mode signal ($N = 2{,}000$, $[175, 215]\text{ THz}$)

| Metric | Meep (`py_do_harminv`) | `bf.harminv` |
|---|---|---|
| Modes matched | 3 / 3 | 3 / 3 |
| Max rel. freq error | `9.84e-7` | `4.84e-16` |
| Max rel. $Q$ error | `2.86e-4` | `7.04e-13` |
| Max rel. $\|a\|$ error | `5.50e-4` | `3.68e-13` |
| Max rel. complex $a$ error | `7.14e-4` | `3.70e-13` |

### High-$Q$ sweep ($N = 4{,}000$, $f_0 = 193.4\text{ THz}$, $[185, 202]\text{ THz}$)

| Target $Q$ | Meep freq rel | Meep $Q$ rel | Meep $\|a\|$ rel | `bf.harminv` freq rel | `bf.harminv` $Q$ rel | `bf.harminv` complex $a$ rel |
|---|---|---|---|---|---|---|
| `1e2` | `1.62e-16` | `1.59e-14` | `2.89e-15` | `1.62e-16` | `1.99e-13` | `7.18e-14` |
| `1e3` | `3.23e-16` | `7.36e-13` | `5.83e-13` | `9.69e-16` | `1.79e-13` | `7.19e-13` |
| `1e4` | `3.23e-16` | `6.15e-13` | `3.34e-12` (conj. $a$: `7.79e-1`) | `0.00e+0` | `1.31e-11` | `7.04e-13` |
| `1e5` | `1.62e-16` | `1.02e-10` | `1.77e-11` (conj. $a$: `7.79e-1`) | `6.46e-16` | `3.31e-10` | `1.47e-12` |
| `1e6` | `3.23e-16` | `8.64e-12` | `1.11e-10` (conj. $a$: `7.79e-1`) | `9.69e-16` | `9.05e-10` | `1.22e-12` |

### Sub-Rayleigh close doublet ($N = 1{,}500$, $T = 600\text{ fs}$, $Q_1 = 300, Q_2 = 350$)

| $\Delta f \cdot T$ | $\Delta f$ (THz) | Meep matched | Meep freq rel | Meep $Q$ rel | `bf` matched | `bf` freq rel | `bf` $Q$ rel | `bf` complex $a$ rel |
|---|---|---|---|---|---|---|---|---|
| `2.00` | `3.3333` | 2 / 2 | `2.70e-6` | `1.88e-3` | 2 / 2 | `8.15e-16` | `1.11e-12` | `5.00e-13` |
| `1.00` | `1.6667` | 2 / 2 | `6.49e-16` | `2.12e-12` | 2 / 2 | `1.77e-15` | `1.47e-12` | `9.76e-13` |
| `0.50` | `0.8333` | 2 / 2 | `1.92e-14` | `1.21e-11` | 2 / 2 | `8.54e-15` | `3.73e-12` | `4.83e-12` |
| `0.25` | `0.4167` | 2 / 2 | `1.05e-3` | `4.43e-1` | 2 / 2 | `8.57e-15` | `6.50e-12` | `9.68e-12` |
| `0.15` | `0.2500` | 2 / 2 | `6.04e-4` | `2.89e-1` | 2 / 2 | `4.36e-15` | `1.07e-12` | `4.98e-12` |

### Short-record sweep ($f_1 = 191\text{ THz}, Q_1 = 80$; $f_2 = 199\text{ THz}, Q_2 = 110$)

| $N$ | Optical cycles | Meep matched | Meep freq rel | Meep $Q$ rel | `bf` matched | `bf` freq rel | `bf` $Q$ rel | `bf` complex $a$ rel |
|---|---|---|---|---|---|---|---|---|
| `78` | `6.08` | 1 / 2 | `1.48e-2` | `7.43e-1` | 2 / 2 | `2.06e-12` | `2.61e-10` | `1.09e-10` |
| `130` | `10.14` | 2 / 2 | `2.06e-14` | `1.01e-11` | 2 / 2 | `7.82e-13` | `9.18e-11` | `4.51e-11` |
| `260` | `20.28` | 2 / 2 | `4.08e-15` | `2.96e-13` | 2 / 2 | `3.27e-16` | `3.91e-13` | `1.04e-13` |
| `520` | `40.56` | 2 / 2 | `1.35e-2` | `2.63e+0` | 2 / 2 | `4.71e-16` | `1.98e-13` | `8.06e-14` |
| `1040` | `81.12` | 2 / 2 | `6.54e-16` | `6.87e-13` | 2 / 2 | `7.85e-16` | `9.84e-14` | `1.02e-13` |

### Additive Gaussian noise sweep ($N = 1{,}500$, 10 seeds per SNR)

Median worst-mode error over matched runs (`10 / 10` runs matched both modes across all SNR levels):

| SNR (dB) | Meep freq | Meep $Q$ | Meep $\|a\|$ | `bf` default freq | `bf` default $Q$ | `bf` default $\|a\|$ | `bf` tuned `svd_tol` | Tuned freq | Tuned $Q$ | Tuned $\|a\|$ |
|---|---|---|---|---|---|---|---|---|---|---|
| `80` | `2.87e-8` | `1.48e-5` | `1.10e-5` | `4.18e-8` | `1.83e-5` | `1.71e-5` | `2.00e-5` | `3.19e-8` | `1.15e-5` | `6.24e-6` |
| `60` | `2.18e-6` | `5.92e-4` | `5.99e-4` | `3.87e-7` | `2.23e-4` | `1.09e-4` | `2.00e-4` | `8.92e-6` | `2.30e-3` | `8.18e-4` |
| `40` | `8.04e-6` | `4.55e-3` | `4.79e-3` | `3.97e-6` | `1.65e-3` | `1.17e-3` | `2.00e-3` | `9.21e-6` | `2.92e-3` | `1.24e-3` |
| `30` | `2.40e-5` | `7.88e-3` | `6.27e-3` | `1.40e-5` | `4.51e-3` | `4.60e-3` | `6.32e-3` | `1.58e-5` | `5.23e-3` | `2.32e-3` |
| `20` | `4.67e-5` | `1.77e-2` | `1.43e-2` | `3.80e-5` | `1.09e-2` | `1.08e-2` | `2.00e-2` | `4.15e-5` | `1.08e-2` | `5.92e-3` |

### Antisymmetric spatial mode recovery ($M_{\text{ch}} = 65$, $N = 1{,}000$)

Signal combines an even mode (`191.0 THz`, $Q=120$) and an odd mode (`198.0 THz`, $Q=150$, proportional to $x e^{-2x^2}$ on $x \in [-1, 1]$ with zero spatial mean and zero node on center index 32):

- **Meep single center cell (`x = 0`):** matched `1 / 2` modes (`[191.0 THz]`) — misses the odd mode.
- **Meep spatial mean:** matched `1 / 2` modes (`[191.0 THz]`) — misses the odd mode.
- **`bf.harminv` multi-channel:** matched `2 / 2` modes (`[191.0, 198.0] THz`), recovering the full 65-cell complex odd spatial amplitude profile with relative $L_2$ error `2.33e-13`.

---

## 2. Runtime & Autodiff Scaling

Warm times are median execution latencies in milliseconds across repeated runs (`jax.block_until_ready` for JAX CPU/GPU). Meep reports both the raw C call (`py_do_harminv` on a pre-built Python `list[complex]`) and total Python call time including `[complex(v) for v in array]`.

### Single-channel trace length scaling ($J = 16$ at $N = 500$; $J = 24$ for $N \ge 2{,}000$)

| $N$ | $J$ | Meep C (ms) | Meep total (ms) | `bf` CPU fwd (ms) | `bf` CPU `val+grad` (ms) | `bf` GPU fwd (ms) | `bf` GPU `val+grad` (ms) |
|---|---|---|---|---|---|---|---|
| `500` | `16` | `0.20` | `0.24` | `0.77` | `0.89` | `7.01` | `7.30` |
| `2,000` | `24` | `0.82` | `0.94` | `1.40` | `3.98` | `8.25` | `8.72` |
| `10,000` | `24` | `3.15` | `3.75` | `4.03` | `13.83` | `10.57` | `10.95` |
| `50,000` | `24` | `13.85` | `16.66` | `18.21` | `58.37` | `10.65` | `11.03` |
| `200,000` | `24` | `55.68` | `68.40` | `81.03` | `124.22` | `13.22` | `13.91` |

*Cold XLA compilation times on `length_N`: `391–484 ms` (CPU fwd), `637–734 ms` (CPU grad), `639–840 ms` (GPU fwd), `965–1053 ms` (GPU grad). Peak GPU VRAM across the scaling suite: `1.214 GB`.*

### Basis size $J$ scaling ($N = 10{,}000$)

| $J$ | Meep C (ms) | `bf` CPU fwd (ms) | `bf` CPU `val+grad` (ms) | `bf` GPU fwd (ms) | `bf` GPU `val+grad` (ms) |
|---|---|---|---|---|---|
| `8` | `2.17` | `1.94` | `2.29` | `5.95` | `6.24` |
| `16` | `2.44` | `3.70` | `4.23` | `9.00` | `9.34` |
| `32` | `4.13` | `3.95` | `18.07` | `13.81` | `14.37` |
| `64` | `8.81` | `18.30` | `30.39` | `23.04` | `23.65` |
| `128` | `37.04` | `62.37` | `74.56` | `52.89` | `54.57` |

### Multi-channel `FieldDetector` scaling ($N = 4{,}000$, $J = 16$, `float32`)

For Meep, $M_{\text{ch}} \in \{1, 16, 256\}$ are measured directly in a per-channel `py_do_harminv` loop; $M_{\text{ch}} \in \{4{,}096, 16{,}384\}$ are linear extrapolations (`*`) from the measured $M_{\text{ch}} = 256$ time (`315.07 ms`).

| $M_{\text{ch}}$ | Meep per-ch (ms) | `bf` CPU fwd (ms) | `bf` CPU `val+grad` (ms) | `bf` GPU fwd (ms) | `bf` GPU `val+grad` (ms) |
|---|---|---|---|---|---|
| `1` | `2.00` | `1.78` | `2.10` | `8.27` | `8.58` |
| `16` | `19.52` | `2.23` | `2.96` | `9.93` | `11.25` |
| `256` | `315.07` | `8.70` | `10.22` | `11.02` | `11.38` |
| `4,096` | `5,041.15*` | `100.38` | `101.27` | `13.10` | `14.01` |
| `16,384` | `20,164.62*` | `197.92` | `229.12` | `22.06` | `25.76` |

At $M_{\text{ch}} = 16{,}384$ channels, `bf.harminv` runs in **22.06 ms** on L40S GPU (with full reverse-mode gradients in **25.76 ms**), compared to ~`20.16 s` for per-channel `libharminv`.

---

## 3. End-to-End 1D Fabry–Pérot Resonator FDTD Comparison

1D-periodic Silicon slab in air ($L_0 = 1.36\,\mu\text{m}$, $n = 3.48$, grid spacing $\Delta x = 0.02\,\mu\text{m}$, $160 \times 3 \times 3$ GPU box, PML $0.40\,\mu\text{m}$, modes $m = 5, 6, 7$ in $[148, 235]\text{ THz}$):

| Mode | Analytic $(f\text{ THz},\ Q)$ | Meep 1D FDTD + `mp.Harminv` | `bifrost` + `fdtdx` GPU + `bf.harminv_detector` | `mp.py_do_harminv` on `fdtdx` trace |
|---|---|---|---|---|
| $m = 5$ | `158.359`, `13.281` | `157.559`, `13.046` | `158.015`, `13.251` | `158.014`, `13.256` |
| $m = 6$ | `190.031`, `15.937` | `189.424`, `15.984` | `189.438`, `15.887` | `189.436`, `15.881` |
| $m = 7$ | `221.703`, `18.594` | `220.836`, `18.684` | `220.762`, `18.513` | `220.766`, `18.529` |

- **Meep FDTD runtime (`wall_s`):** `0.338 s` (CPU, 14,001 Harminv samples).
- **`bifrost` GPU FDTD runtime:** `5.68 s` cold compile, `0.615 s` warm forward (`14,686` total steps, `5,769` detector samples at `dt_eff = 0.076263 fs`), `24.49 s` cold / `12.56 s` warm `bf.jit(jax.value_and_grad)` end-to-end through `fdtdx` (peak GPU VRAM `0.269 GB`).
- **End-to-end geometry derivative $\mathrm{d}f_6/\mathrm{d}L$ at $L_0 = 1.36\,\mu\text{m}$:**
  - `bf.jit(jax.value_and_grad)`: **`-139.561 THz/um`**
  - Central finite difference ($h = 0.01\,\mu\text{m}$): **`-138.456 THz/um`** (0.80% difference)
  - Analytical scaling $-f_6 / L_0$: **`-139.292 THz/um`** (0.19% difference)
