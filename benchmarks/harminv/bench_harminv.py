"""Accuracy, runtime/autodiff scaling, and FDTD resonator benchmark: bifrost.harminv vs Meep.

Run from the repository root on a compute/GPU node (via SLURM on Engaging):

    python benchmarks/harminv/bench_harminv.py

Each backend suite (accuracy, Meep CPU scaling, JAX CPU scaling, JAX GPU scaling,
Meep FDTD, and bifrost+fdtdx GPU FDTD) executes in a clean worker subprocess so
XLA compilation, OpenMPI/BLAS state, and CUDA memory remain completely isolated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "benchmarks" / "harminv" / "results" / "bench_harminv.json"
sys.path.insert(0, str(REPO_ROOT))


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _iqr_ms(samples_s: list[float]) -> float:
    if len(samples_s) < 2:
        return 0.0
    q = statistics.quantiles(samples_s, n=4)
    return float(q[2] - q[0]) * 1e3


def _med_ms(samples_s: list[float]) -> float:
    return float(statistics.median(samples_s)) * 1e3


def _match_modes(
    true_modes: list[tuple[float, float, complex]],
    extracted: list[dict[str, float | complex]],
    tol_rel: float = 0.05,
) -> dict[str, Any]:
    """One-to-one nearest-frequency match between ground-truth and extracted modes."""
    pairs: list[tuple[float, int, int]] = []
    for ti, (f_true, _q_true, _a_true) in enumerate(true_modes):
        for ei, cand in enumerate(extracted):
            rel = abs(float(cand["frequency"]) - f_true) / f_true
            if rel <= tol_rel:
                pairs.append((rel, ti, ei))
    pairs.sort()

    used_t: set[int] = set()
    used_e: set[int] = set()
    f_errs: list[float] = []
    q_errs: list[float] = []
    a_errs: list[float] = []
    a_mag_errs: list[float] = []
    a_conj_errs: list[float] = []
    for rel, ti, ei in pairs:
        if ti in used_t or ei in used_e:
            continue
        used_t.add(ti)
        used_e.add(ei)
        f_true, q_true, a_true = true_modes[ti]
        cand = extracted[ei]
        f_errs.append(rel)
        q_val = float(cand["Q"])
        q_errs.append(abs(q_val - q_true) / abs(q_true) if math.isfinite(q_val) else float("inf"))
        a_cand = complex(cand["amplitude"])
        a_errs.append(abs(a_cand - a_true) / abs(a_true))
        a_mag_errs.append(abs(abs(a_cand) - abs(a_true)) / abs(a_true))
        a_conj_errs.append(min(abs(a_cand - a_true), abs(complex(a_cand.conjugate()) - a_true)) / abs(a_true))

    return {
        "matched": len(used_t),
        "total": len(true_modes),
        "num_extracted": len(extracted),
        "max_freq_rel": float(max(f_errs)) if f_errs else None,
        "max_q_rel": float(max(q_errs)) if q_errs else None,
        "max_amp_rel": float(max(a_errs)) if a_errs else None,
        "max_amp_mag_rel": float(max(a_mag_errs)) if a_mag_errs else None,
        "max_amp_up_to_conj_rel": float(max(a_conj_errs)) if a_conj_errs else None,
    }


def _synth_real_modes(
    n_steps: int,
    dt: float,
    modes: list[tuple[float, float, complex]],
) -> Any:
    import numpy as np

    t = np.arange(n_steps, dtype=np.float64) * dt
    y = np.zeros(n_steps, dtype=np.float64)
    for f_hz, q_val, amp in modes:
        gamma = math.pi * f_hz / q_val
        omega = 2.0 * math.pi * f_hz - 1j * gamma
        y += 2.0 * np.real(amp * np.exp(-1j * omega * t))
    return y


def _run_meep_trace(
    trace_np: Any,
    dt_s: float,
    f_min_hz: float,
    f_max_hz: float,
    *,
    maxbands: int = 16,
    spectral_density: float = 1.1,
    Q_thresh: float = 10.0,
    err_thresh: float = 0.01,
) -> list[dict[str, float | complex]]:
    import meep as mp
    import numpy as np

    dt_ps = float(dt_s) * 1e12
    raw = mp.py_do_harminv(
        [complex(v) for v in np.asarray(trace_np, dtype=np.float64)],
        dt_ps,
        float(f_min_hz) * 1e-12,
        float(f_max_hz) * 1e-12,
        int(maxbands),
        float(spectral_density),
        float(Q_thresh),
        mp.inf,
        float(err_thresh),
        -1.0,
        -1.0,
    )
    out = []
    for m_freq, m_amp, m_err in raw:
        q_val = m_freq.real / (-2.0 * m_freq.imag) if m_freq.imag != 0.0 else math.inf
        out.append(
            {
                "frequency": float(m_freq.real) * 1e12,
                "Q": float(q_val),
                "amplitude": complex(m_amp),
                "error": float(abs(m_err)),
            }
        )
    out.sort(key=lambda d: float(d["frequency"]))
    return out


def _run_bifrost_trace(
    trace_np: Any,
    dt_s: float,
    f_min_hz: float,
    f_max_hz: float,
    *,
    max_modes: int = 16,
    spectral_density: float = 1.1,
    Q_thresh: float = 10.0,
    err_thresh: float = 0.01,
    svd_tol: float | None = None,
) -> list[dict[str, float | complex]]:
    import jax.numpy as jnp

    import bifrost as bf

    res = bf.harminv(
        jnp.asarray(trace_np),
        dt_s,
        f_min_hz,
        f_max_hz,
        max_modes=max_modes,
        spectral_density=spectral_density,
        Q_thresh=Q_thresh,
        err_thresh=err_thresh,
        svd_tol=svd_tol,
    )
    return [
        {
            "frequency": float(m["frequency"]),
            "Q": float(m["Q"]),
            "amplitude": complex(m["amplitude"]),
            "error": float(m["error"]),
        }
        for m in res.to_modes()
    ]


def _suite_accuracy() -> dict[str, Any]:
    import numpy as np

    results: dict[str, Any] = {}

    # 1. Clean 3-mode real signal
    dt = 0.4e-15
    n_clean = 2000
    modes_clean = [
        (186.0e12, 120.0, 1.0 * np.exp(0.3j)),
        (193.5e12, 450.0, 0.8 * np.exp(-0.7j)),
        (204.0e12, 280.0, 0.5 * np.exp(1.1j)),
    ]
    sig_clean = _synth_real_modes(n_clean, dt, modes_clean)
    m_ext = _run_meep_trace(sig_clean, dt, 175e12, 215e12, Q_thresh=10.0, err_thresh=0.01)
    b_ext = _run_bifrost_trace(sig_clean, dt, 175e12, 215e12, Q_thresh=10.0, err_thresh=0.01)
    results["clean_3mode"] = {
        "n_steps": n_clean,
        "dt_fs": dt * 1e15,
        "meep": _match_modes(modes_clean, m_ext),
        "bifrost": _match_modes(modes_clean, b_ext),
    }

    # 2. High-Q sweep
    high_q_rows = []
    n_hq = 4000
    f_hq = 193.4e12
    for q_target in (1e2, 1e3, 1e4, 1e5, 1e6):
        modes = [(f_hq, float(q_target), 1.0 * np.exp(0.4j))]
        sig = _synth_real_modes(n_hq, dt, modes)
        me = _run_meep_trace(sig, dt, 185e12, 202e12, Q_thresh=10.0, err_thresh=0.05)
        be = _run_bifrost_trace(sig, dt, 185e12, 202e12, Q_thresh=10.0, err_thresh=0.05)
        high_q_rows.append(
            {
                "Q": float(q_target),
                "meep": _match_modes(modes, me),
                "bifrost": _match_modes(modes, be),
            }
        )
    results["high_q"] = {"n_steps": n_hq, "dt_fs": dt * 1e15, "rows": high_q_rows}

    # 3. Sub-Rayleigh close doublet sweep
    doublet_rows = []
    n_dbl = 1500
    t_dur = n_dbl * dt
    f_center = 193.5e12
    for alpha in (2.0, 1.0, 0.5, 0.25, 0.15):
        sep_hz = alpha / t_dur
        modes = [
            (f_center - 0.5 * sep_hz, 300.0, 1.0 * np.exp(0.2j)),
            (f_center + 0.5 * sep_hz, 350.0, 0.85 * np.exp(-0.6j)),
        ]
        sig = _synth_real_modes(n_dbl, dt, modes)
        me = _run_meep_trace(sig, dt, 183e12, 204e12, Q_thresh=10.0, err_thresh=0.1)
        be = _run_bifrost_trace(sig, dt, 183e12, 204e12, Q_thresh=10.0, err_thresh=0.1)
        doublet_rows.append(
            {
                "df_T": alpha,
                "sep_thz": round(sep_hz * 1e-12, 4),
                "meep": _match_modes(modes, me),
                "bifrost": _match_modes(modes, be),
            }
        )
    results["doublet"] = {"n_steps": n_dbl, "t_fs": t_dur * 1e15, "rows": doublet_rows}

    # 4. Short-record sweep
    short_rows = []
    modes_short = [
        (191.0e12, 80.0, 1.0 * np.exp(0.3j)),
        (199.0e12, 110.0, 0.7 * np.exp(-0.5j)),
    ]
    for n_short in (78, 130, 260, 520, 1040):
        sig = _synth_real_modes(n_short, dt, modes_short)
        cycles = n_short * dt * 195e12
        me = _run_meep_trace(sig, dt, 175e12, 215e12, Q_thresh=10.0, err_thresh=0.05)
        be = _run_bifrost_trace(sig, dt, 175e12, 215e12, Q_thresh=10.0, err_thresh=0.05)
        short_rows.append(
            {
                "n_steps": n_short,
                "cycles": round(cycles, 2),
                "meep": _match_modes(modes_short, me),
                "bifrost": _match_modes(modes_short, be),
            }
        )
    results["short_record"] = {"dt_fs": dt * 1e15, "rows": short_rows}

    # 5. Additive Gaussian noise sweep (10 seeds per SNR)
    n_noise = 1500
    modes_noise = [
        (191.0e12, 180.0, 1.0 * np.exp(0.3j)),
        (198.0e12, 220.0, 0.75 * np.exp(-0.8j)),
    ]
    clean_n = _synth_real_modes(n_noise, dt, modes_noise)
    sig_rms = float(np.sqrt(np.mean(clean_n**2)))
    noise_rows = []
    for snr_db in (80, 60, 40, 30, 20):
        noise_std = sig_rms * (10.0 ** (-snr_db / 20.0))
        tuned_tol = min(0.2 * (10.0 ** (-snr_db / 20.0)), 0.1)
        stats: dict[str, dict[str, list[float] | int]] = {
            name: {"full_match": 0, "f_errs": [], "q_errs": [], "a_errs": [], "a_mag_errs": []}
            for name in ("meep", "bifrost_default", "bifrost_tuned")
        }
        for seed in range(10):
            rng = np.random.default_rng(1000 * snr_db + seed)
            noisy = clean_n + noise_std * rng.standard_normal(n_noise)
            trials = {
                "meep": _run_meep_trace(noisy, dt, 180e12, 210e12, Q_thresh=10.0, err_thresh=0.05),
                "bifrost_default": _run_bifrost_trace(
                    noisy, dt, 180e12, 210e12, Q_thresh=10.0, err_thresh=0.05, svd_tol=None
                ),
                "bifrost_tuned": _run_bifrost_trace(
                    noisy, dt, 180e12, 210e12, Q_thresh=10.0, err_thresh=0.05, svd_tol=tuned_tol
                ),
            }
            for key, ext in trials.items():
                m = _match_modes(modes_noise, ext)
                if m["matched"] == 2:
                    stats[key]["full_match"] = int(stats[key]["full_match"]) + 1
                    stats[key]["f_errs"].append(float(m["max_freq_rel"]))  # type: ignore[union-attr]
                    stats[key]["q_errs"].append(float(m["max_q_rel"]))  # type: ignore[union-attr]
                    stats[key]["a_errs"].append(float(m["max_amp_rel"]))  # type: ignore[union-attr]
                    stats[key]["a_mag_errs"].append(float(m["max_amp_mag_rel"]))  # type: ignore[union-attr]
        row: dict[str, Any] = {"snr_db": snr_db, "trials": 10, "tuned_svd_tol": tuned_tol}
        for key, st in stats.items():
            f_list: list[float] = st["f_errs"]  # type: ignore[assignment]
            q_list: list[float] = st["q_errs"]  # type: ignore[assignment]
            a_list: list[float] = st["a_errs"]  # type: ignore[assignment]
            am_list: list[float] = st["a_mag_errs"]  # type: ignore[assignment]
            row[key] = {
                "full_match_count": st["full_match"],
                "med_freq_rel": float(statistics.median(f_list)) if f_list else None,
                "med_q_rel": float(statistics.median(q_list)) if q_list else None,
                "med_amp_rel": float(statistics.median(a_list)) if a_list else None,
                "med_amp_mag_rel": float(statistics.median(am_list)) if am_list else None,
            }
        noise_rows.append(row)
    results["noise_snr"] = {"n_steps": n_noise, "rows": noise_rows}

    # 6. Antisymmetric odd spatial mode demo (M_ch = 65, mirror plane at index 32)
    n_odd = 1000
    t_odd = np.arange(n_odd, dtype=np.float64) * dt
    x = np.linspace(-1.0, 1.0, 65, dtype=np.float64)
    prof_even = np.exp(-2.0 * x**2)
    prof_odd = x * np.exp(-2.0 * x**2)  # strictly zero on mirror plane x=0 (idx 32) and zero spatial mean
    a_even = 1.0 * np.exp(0.2j) * prof_even
    a_odd = 1.2 * np.exp(-0.4j) * prof_odd
    w_e = 2.0 * math.pi * 191.0e12 - 1j * (math.pi * 191.0e12 / 120.0)
    w_o = 2.0 * math.pi * 198.0e12 - 1j * (math.pi * 198.0e12 / 150.0)
    fields_odd = 2.0 * np.real(
        np.exp(-1j * w_e * t_odd)[:, None] * a_even[None, :] + np.exp(-1j * w_o * t_odd)[:, None] * a_odd[None, :]
    )
    center_trace = fields_odd[:, 32]
    mean_trace = np.mean(fields_odd, axis=1)
    true_odd_pair = [(191.0e12, 120.0, 1.0 + 0j), (198.0e12, 150.0, 1.0 + 0j)]

    import jax.numpy as jnp

    import bifrost as bf

    res_odd_bf = bf.harminv(jnp.asarray(fields_odd), dt, 180e12, 210e12, max_modes=8, Q_thresh=10.0)
    bf_odd_modes = [
        {"frequency": float(m["frequency"]), "Q": float(m["Q"]), "amplitude": 1.0 + 0j}
        for m in res_odd_bf.to_modes()
    ]
    results["antisymmetric_odd_mode"] = {
        "n_steps": n_odd,
        "m_ch": 65,
        "meep_center_cell": {
            "matched": _match_modes(true_odd_pair, _run_meep_trace(center_trace, dt, 180e12, 210e12))["matched"],
            "frequencies_thz": [round(float(m["frequency"]) * 1e-12, 3) for m in _run_meep_trace(center_trace, dt, 180e12, 210e12)],
        },
        "meep_spatial_mean": {
            "matched": _match_modes(true_odd_pair, _run_meep_trace(mean_trace, dt, 180e12, 210e12))["matched"],
            "frequencies_thz": [round(float(m["frequency"]) * 1e-12, 3) for m in _run_meep_trace(mean_trace, dt, 180e12, 210e12)],
        },
        "bifrost_multichannel": {
            "matched": _match_modes(true_odd_pair, bf_odd_modes)["matched"],
            "frequencies_thz": [round(float(m["frequency"]) * 1e-12, 3) for m in bf_odd_modes],
            "odd_profile_rel_err": float(
                np.linalg.norm(np.asarray(res_odd_bf.amplitude[1]) - a_odd) / np.linalg.norm(a_odd)
            ),
        },
    }
    return results


def _length_band_for(n_steps: int, dt: float) -> tuple[float, float, int]:
    j_target = 16 if n_steps <= 500 else 24
    df = (j_target - 0.5) / (1.1 * (n_steps - 1) * dt)
    f_cen = 195.0e12
    return f_cen - 0.5 * df, f_cen + 0.5 * df, j_target


def _basis_band_for(n_steps: int, dt: float, j_target: int) -> tuple[float, float]:
    df = (j_target - 0.5) / (1.1 * (n_steps - 1) * dt)
    f_cen = 195.0e12
    return f_cen - 0.5 * df, f_cen + 0.5 * df


def _suite_scaling_meep() -> dict[str, Any]:
    import meep as mp
    import numpy as np

    dt = 0.4e-15
    dt_ps = dt * 1e12
    length_rows = []
    for n_steps in (500, 2000, 10000, 50000, 200000):
        f_min, f_max, j_target = _length_band_for(n_steps, dt)
        df_modes = min(6.0e12, 0.25 * (f_max - f_min))
        q_scale = max(180.0, 0.4 * math.pi * 195.0e12 * n_steps * dt)
        modes = [
            (195.0e12 - df_modes, q_scale, 1.0 * np.exp(0.3j)),
            (195.0e12 + df_modes, 1.25 * q_scale, 0.7 * np.exp(-0.5j)),
        ]
        sig = _synth_real_modes(n_steps, dt, modes)
        vals = [complex(v) for v in sig]
        f_lo_thz, f_hi_thz = f_min * 1e-12, f_max * 1e-12
        # Warmup
        _ = mp.py_do_harminv(vals, dt_ps, f_lo_thz, f_hi_thz, 16, 1.1, 10.0, mp.inf, 0.01, -1.0, -1.0)
        reps = 7 if n_steps >= 200000 else 20
        c_times = []
        tot_times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = mp.py_do_harminv(vals, dt_ps, f_lo_thz, f_hi_thz, 16, 1.1, 10.0, mp.inf, 0.01, -1.0, -1.0)
            c_times.append(time.perf_counter() - t0)
            assert len(out) >= 1

            t1 = time.perf_counter()
            vals_conv = [complex(v) for v in sig]
            out2 = mp.py_do_harminv(vals_conv, dt_ps, f_lo_thz, f_hi_thz, 16, 1.1, 10.0, mp.inf, 0.01, -1.0, -1.0)
            tot_times.append(time.perf_counter() - t1)
            assert len(out2) >= 1
        length_rows.append(
            {
                "n_steps": n_steps,
                "J": j_target,
                "meep_c_med_ms": round(_med_ms(c_times), 4),
                "meep_c_iqr_ms": round(_iqr_ms(c_times), 4),
                "meep_total_med_ms": round(_med_ms(tot_times), 4),
                "meep_total_iqr_ms": round(_iqr_ms(tot_times), 4),
            }
        )

    basis_rows = []
    n_basis_len = 10000
    for j_target in (8, 16, 32, 64, 128):
        f_min, f_max = _basis_band_for(n_basis_len, dt, j_target)
        df_modes = min(2.0e12, 0.25 * (f_max - f_min))
        modes = [
            (195.0e12 - df_modes, 800.0, 1.0 * np.exp(0.2j)),
            (195.0e12 + df_modes, 1000.0, 0.7 * np.exp(-0.4j)),
        ]
        sig = _synth_real_modes(n_basis_len, dt, modes)
        vals = [complex(v) for v in sig]
        f_lo_thz, f_hi_thz = f_min * 1e-12, f_max * 1e-12
        _ = mp.py_do_harminv(vals, dt_ps, f_lo_thz, f_hi_thz, 16, 1.1, 10.0, mp.inf, 0.01, -1.0, -1.0)
        reps = 10 if j_target >= 64 else 20
        c_times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = mp.py_do_harminv(vals, dt_ps, f_lo_thz, f_hi_thz, 16, 1.1, 10.0, mp.inf, 0.01, -1.0, -1.0)
            c_times.append(time.perf_counter() - t0)
            assert len(out) >= 1
        basis_rows.append(
            {
                "n_steps": n_basis_len,
                "J": j_target,
                "meep_c_med_ms": round(_med_ms(c_times), 4),
                "meep_c_iqr_ms": round(_iqr_ms(c_times), 4),
            }
        )

    multichannel_rows = []
    n_mc = 4000
    f_min_mc, f_max_mc = _basis_band_for(n_mc, dt, 16)
    f_lo_mc, f_hi_mc = f_min_mc * 1e-12, f_max_mc * 1e-12
    t256_med_ms = None
    for m_ch in (1, 16, 256, 4096, 16384):
        if m_ch <= 256:
            rng = np.random.default_rng(7)
            spatial = rng.standard_normal((2, m_ch)).astype(np.float64)
            base1 = _synth_real_modes(n_mc, dt, [(193.0e12, 180.0, 1.0 + 0.2j)])
            base2 = _synth_real_modes(n_mc, dt, [(197.0e12, 220.0, 0.7 - 0.4j)])
            fields = (base1[:, None] * spatial[0][None, :] + base2[:, None] * spatial[1][None, :]).astype(np.float32)

            def _run_loop() -> None:
                for ch in range(m_ch):
                    vals_ch = [complex(v) for v in fields[:, ch]]
                    mp.py_do_harminv(vals_ch, dt_ps, f_lo_mc, f_hi_mc, 16, 1.1, 10.0, mp.inf, 0.01, -1.0, -1.0)

            _run_loop()
            reps = 5 if m_ch == 256 else 15
            times = []
            for _ in range(reps):
                t0 = time.perf_counter()
                _run_loop()
                times.append(time.perf_counter() - t0)
            med = _med_ms(times)
            iqr = _iqr_ms(times)
            if m_ch == 256:
                t256_med_ms = med
            multichannel_rows.append(
                {
                    "n_steps": n_mc,
                    "m_ch": m_ch,
                    "J": 16,
                    "meep_med_ms": round(med, 4),
                    "meep_iqr_ms": round(iqr, 4),
                    "extrapolated": False,
                }
            )
        else:
            assert t256_med_ms is not None
            multichannel_rows.append(
                {
                    "n_steps": n_mc,
                    "m_ch": m_ch,
                    "J": 16,
                    "meep_med_ms": round(t256_med_ms * (m_ch / 256.0), 4),
                    "meep_iqr_ms": None,
                    "extrapolated": True,
                }
            )

    return {"length_N": length_rows, "basis_J": basis_rows, "multichannel_M": multichannel_rows}


def _suite_scaling_jax(expected_backend: str) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    import bifrost as bf

    backend = jax.default_backend()
    if backend != expected_backend:
        raise RuntimeError(f"Expected JAX backend {expected_backend!r}, got {backend!r} ({jax.devices()}).")
    dev_kind = getattr(jax.devices()[0], "device_kind", str(jax.devices()[0]))

    dt = 0.4e-15

    def _bench_fn(fn: Any, arg: Any, reps: int) -> tuple[float, float, float]:
        t0 = time.perf_counter()
        out0 = fn(arg)
        jax.block_until_ready(out0)
        cold_ms = (time.perf_counter() - t0) * 1e3

        # Extra warmup call
        jax.block_until_ready(fn(arg))
        samples = []
        for _ in range(reps):
            t1 = time.perf_counter()
            out = fn(arg)
            jax.block_until_ready(out)
            samples.append(time.perf_counter() - t1)
        return round(cold_ms, 4), round(_med_ms(samples), 4), round(_iqr_ms(samples), 4)

    length_rows = []
    for n_steps in (500, 2000, 10000, 50000, 200000):
        f_min, f_max, j_target = _length_band_for(n_steps, dt)
        df_modes = min(6.0e12, 0.25 * (f_max - f_min))
        q_scale = max(180.0, 0.4 * math.pi * 195.0e12 * n_steps * dt)
        modes = [
            (195.0e12 - df_modes, q_scale, 1.0 * np.exp(0.3j)),
            (195.0e12 + df_modes, 1.25 * q_scale, 0.7 * np.exp(-0.5j)),
        ]
        sig = jnp.asarray(_synth_real_modes(n_steps, dt, modes))

        def _fwd(s: jax.Array, f_lo: float = f_min, f_hi: float = f_max, nb: int = j_target) -> jax.Array:
            res = bf.harminv(s, dt, f_lo, f_hi, max_modes=16, num_basis=nb, Q_thresh=10.0)
            return jnp.sum(jnp.where(res.valid, res.frequency, 0.0))

        fwd_jit = bf.jit(_fwd)
        grad_jit = bf.jit(jax.value_and_grad(_fwd))
        reps = 7 if n_steps >= 200000 else 20
        f_cold, f_med, f_iqr = _bench_fn(fwd_jit, sig, reps)
        g_cold, g_med, g_iqr = _bench_fn(grad_jit, sig, reps)
        length_rows.append(
            {
                "n_steps": n_steps,
                "J": j_target,
                "fwd_cold_ms": f_cold,
                "fwd_med_ms": f_med,
                "fwd_iqr_ms": f_iqr,
                "grad_cold_ms": g_cold,
                "grad_med_ms": g_med,
                "grad_iqr_ms": g_iqr,
            }
        )

    basis_rows = []
    n_basis_len = 10000
    for j_target in (8, 16, 32, 64, 128):
        f_min, f_max = _basis_band_for(n_basis_len, dt, j_target)
        df_modes = min(2.0e12, 0.25 * (f_max - f_min))
        modes = [
            (195.0e12 - df_modes, 800.0, 1.0 * np.exp(0.2j)),
            (195.0e12 + df_modes, 1000.0, 0.7 * np.exp(-0.4j)),
        ]
        sig = jnp.asarray(_synth_real_modes(n_basis_len, dt, modes))

        def _fwd_j(s: jax.Array, f_lo: float = f_min, f_hi: float = f_max, nb: int = j_target) -> jax.Array:
            res = bf.harminv(s, dt, f_lo, f_hi, max_modes=16, num_basis=nb, Q_thresh=10.0)
            return jnp.sum(jnp.where(res.valid, res.frequency, 0.0))

        fwd_jit = bf.jit(_fwd_j)
        grad_jit = bf.jit(jax.value_and_grad(_fwd_j))
        reps = 10 if j_target >= 64 else 20
        f_cold, f_med, f_iqr = _bench_fn(fwd_jit, sig, reps)
        g_cold, g_med, g_iqr = _bench_fn(grad_jit, sig, reps)
        basis_rows.append(
            {
                "n_steps": n_basis_len,
                "J": j_target,
                "fwd_cold_ms": f_cold,
                "fwd_med_ms": f_med,
                "fwd_iqr_ms": f_iqr,
                "grad_cold_ms": g_cold,
                "grad_med_ms": g_med,
                "grad_iqr_ms": g_iqr,
            }
        )

    multichannel_rows = []
    n_mc = 4000
    f_min_mc, f_max_mc = _basis_band_for(n_mc, dt, 16)
    base1 = _synth_real_modes(n_mc, dt, [(193.0e12, 180.0, 1.0 + 0.2j)])
    base2 = _synth_real_modes(n_mc, dt, [(197.0e12, 220.0, 0.7 - 0.4j)])
    for m_ch in (1, 16, 256, 4096, 16384):
        rng = np.random.default_rng(7)
        spatial = rng.standard_normal((2, m_ch)).astype(np.float64)
        fields = jnp.asarray(
            (base1[:, None] * spatial[0][None, :] + base2[:, None] * spatial[1][None, :]).astype(np.float32)
        )

        def _fwd_mc(s: jax.Array) -> jax.Array:
            res = bf.harminv(s, dt, f_min_mc, f_max_mc, max_modes=16, num_basis=16, Q_thresh=10.0)
            return jnp.sum(jnp.where(res.valid, res.frequency, 0.0))

        fwd_jit = bf.jit(_fwd_mc)
        grad_jit = bf.jit(jax.value_and_grad(_fwd_mc))
        reps = 7 if m_ch >= 4096 else 15
        f_cold, f_med, f_iqr = _bench_fn(fwd_jit, fields, reps)
        g_cold, g_med, g_iqr = _bench_fn(grad_jit, fields, reps)
        multichannel_rows.append(
            {
                "n_steps": n_mc,
                "m_ch": m_ch,
                "J": 16,
                "fwd_cold_ms": f_cold,
                "fwd_med_ms": f_med,
                "fwd_iqr_ms": f_iqr,
                "grad_cold_ms": g_cold,
                "grad_med_ms": g_med,
                "grad_iqr_ms": g_iqr,
            }
        )

    mem_stats = jax.devices()[0].memory_stats() or {}
    return {
        "backend": backend,
        "device_kind": dev_kind,
        "peak_gb": round(float(mem_stats.get("peak_bytes_in_use", 0)) / 1e9, 4),
        "length_N": length_rows,
        "basis_J": basis_rows,
        "multichannel_M": multichannel_rows,
    }


def _analytic_fp_modes(l0_um: float = 1.36, n_core: float = 3.48) -> list[dict[str, float]]:
    c_m_s = 299792458.0
    r_f = (n_core - 1.0) / (n_core + 1.0)
    log_r = math.log(abs(r_f))
    out = []
    for m in (5, 6, 7):
        f_hz = m * c_m_s / (2.0 * n_core * l0_um * 1e-6)
        q_val = -m * math.pi / (2.0 * log_r)
        out.append({"m": m, "frequency_thz": f_hz * 1e-12, "Q": q_val})
    return out


def _suite_fdtd_meep() -> dict[str, Any]:
    import meep as mp

    c_um_s = 299792458.0 * 1e6  # 299.792458 um / ps -> THz = meep_f * 299.792458
    f_lo_thz, f_hi_thz = 148.0, 235.0
    fcen = 0.5 * (f_lo_thz + f_hi_thz) / (c_um_s * 1e-12)
    df = (f_hi_thz - f_lo_thz) / (c_um_s * 1e-12)
    f_src = 1.0 / 1.55

    sim = mp.Simulation(
        cell_size=mp.Vector3(3.2, 0, 0),
        resolution=50,
        boundary_layers=[mp.PML(0.4, direction=mp.X)],
        k_point=mp.Vector3(),
        geometry=[mp.Block(size=mp.Vector3(1.36, mp.inf, mp.inf), material=mp.Medium(index=3.48))],
        sources=[
            mp.Source(
                mp.GaussianSource(f_src, fwidth=0.35),
                component=mp.Ey,
                center=mp.Vector3(-0.92),
            )
        ],
    )
    h = mp.Harminv(mp.Ey, mp.Vector3(0.92), fcen, df, 16)
    h.Q_thresh = 5.0

    t0 = time.perf_counter()
    sim.run(mp.after_sources(h), until_after_sources=140)
    wall_s = time.perf_counter() - t0

    modes = sorted(
        [
            {
                "frequency_thz": float(m.freq) * (c_um_s * 1e-12),
                "Q": float(m.Q),
                "amp": float(abs(m.amp)),
                "error": float(abs(m.err)),
            }
            for m in h.modes
        ],
        key=lambda d: d["frequency_thz"],
    )
    return {
        "resolution_per_um": 50,
        "dx_um": 0.02,
        "wall_s": round(wall_s, 4),
        "num_harminv_samples": len(h.data),
        "modes": modes,
    }


def _suite_fdtd_bifrost() -> dict[str, Any]:
    import fdtdx
    import gdsfactory as gf
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gdsfactory.technology import LayerLevel, LayerStack

    import bifrost as bf

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"Expected gpu backend for fdtd_bifrost, got {jax.default_backend()!r}")

    gf.gpdk.PDK.activate()
    l0 = 1.36
    n_core = 3.48
    ls = LayerStack(
        layers={"core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=-0.20, thickness=0.40, material="core", mesh_order=1)}
    )
    materials = {
        "core": fdtdx.Material(permittivity=n_core**2),
        "air": fdtdx.Material(permittivity=1.0),
    }
    comp = gf.components.straight(length=l0, width=0.40)
    stack = bf.StackDevice(
        comp,
        ls,
        materials,
        background="air",
        zrange_layers=["core"],
        boundary_offset=(0.92, -0.92, -0.17, 0.17, -0.17, 0.17),
        name="stack",
    )
    config = stack.get_simulation_config(
        time=560e-15,
        grid=fdtdx.core.grid.UniformGrid(spacing=0.02e-6),
        backend="gpu",
        gradient_config=fdtdx.GradientConfig(method="checkpointed", num_checkpoints=16),
    )
    wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=wave,
        spectral_width=fdtdx.WaveCharacter(frequency=45e12),
    )
    volume, c_vol = stack.get_simulation_volume()
    source = fdtdx.UniformPlaneSource(
        name="source",
        partial_grid_shape=(1, None, None),
        wave_character=wave,
        temporal_profile=pulse,
        fixed_E_polarization_vector=(0.0, 1.0, 0.0),
        direction="+",
    )
    c_src = [
        source.place_relative_to(
            volume, axes=(0, 1, 2), own_positions=(-1, 0, 0), other_positions=(-1, 0, 0), grid_margins=(34, 0, 0)
        )
    ]
    probe = fdtdx.FieldDetector(
        name="probe",
        partial_grid_shape=(1, None, None),
        components=("Ey",),
        reduce_volume=True,
        switch=fdtdx.OnOffSwitch(start_time=120e-15, interval=2),
    )
    c_det = [
        probe.place_relative_to(
            volume, axes=(0, 1, 2), own_positions=(1, 0, 0), other_positions=(1, 0, 0), grid_margins=(-34, 0, 0)
        )
    ]
    b_cfg = fdtdx.BoundaryConfig(
        thickness_grid_minx=20,
        thickness_grid_maxx=20,
        thickness_grid_miny=1,
        thickness_grid_maxy=1,
        thickness_grid_minz=1,
        thickness_grid_maxz=1,
        boundary_type_miny="periodic",
        boundary_type_maxy="periodic",
        boundary_type_minz="periodic",
        boundary_type_maxz="periodic",
    )
    boundaries, c_bnd = fdtdx.boundary_objects_from_config(b_cfg, volume)

    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        [volume, stack, source, probe, *boundaries.values()],
        config,
        [*c_vol, *c_src, *c_det, *c_bnd],
        key,
    )
    params = {**params, **bf.init_stack_params(objects)}
    placed_stack = objects["stack"]
    placed_probe = objects["probe"]
    f_lo, f_hi = 148.0e12, 235.0e12
    m6_target_hz = 6.0 * fdtdx.constants.c / (2.0 * n_core * l0 * 1e-6)

    def run_fwd(length_um: jax.Array) -> tuple[bf.HarminvResult, fdtdx.ArrayContainer]:
        c_new = gf.components.straight(length=length_um, width=0.40)
        p = {**params, "stack": {**params["stack"], **placed_stack.pack(c_new)}}
        arrays_in, objs_in, _ = bf.apply_stack(arrays, objects, p, key)
        _, final_arrays = fdtdx.run_fdtd(arrays=arrays_in, objects=objs_in, config=config, key=key, show_progress=False)
        res = bf.harminv_detector(
            placed_probe, final_arrays, f_lo, f_hi, components="Ey", Q_thresh=5.0, rel_amp_thresh=1e-3
        )
        return res, final_arrays

    def freq6_thz(length_um: jax.Array) -> jax.Array:
        res, _ = run_fwd(length_um)
        return res.select(m6_target_hz)["frequency"] / 1e12

    fwd_jit = bf.jit(run_fwd)
    freq_jit = bf.jit(freq6_thz)
    grad_jit = bf.jit(jax.value_and_grad(freq6_thz))
    l0_arr = jnp.asarray(l0, dtype=stack.dtype)

    t0 = time.perf_counter()
    res0, final0 = fwd_jit(l0_arr)
    jax.block_until_ready(res0.frequency)
    fwd_cold_s = time.perf_counter() - t0

    fwd_samples = []
    for _ in range(3):
        t1 = time.perf_counter()
        r_tmp, _ = fwd_jit(l0_arr)
        jax.block_until_ready(r_tmp.frequency)
        fwd_samples.append(time.perf_counter() - t1)

    t2 = time.perf_counter()
    f6_val, df6_dl = grad_jit(l0_arr)
    jax.block_until_ready((f6_val, df6_dl))
    grad_cold_s = time.perf_counter() - t2

    grad_samples = []
    for _ in range(3):
        t3 = time.perf_counter()
        v_tmp, g_tmp = grad_jit(l0_arr)
        jax.block_until_ready((v_tmp, g_tmp))
        grad_samples.append(time.perf_counter() - t3)

    h = 0.01
    fd_df_dl = (
        float(freq_jit(jnp.asarray(l0 + h, dtype=stack.dtype)))
        - float(freq_jit(jnp.asarray(l0 - h, dtype=stack.dtype)))
    ) / (2.0 * h)

    bf_modes = [
        {
            "frequency_thz": float(m["frequency"]) * 1e-12,
            "Q": float(m["Q"]),
            "amp": float(abs(m["amplitude"])),
            "error": float(m["error"]),
        }
        for m in res0.to_modes()
    ]

    dt_eff, _ = bf.detector_time_grid(placed_probe)
    trace = np.asarray(final0.detector_states["probe"]["fields"][:, 0], dtype=np.float64)
    mem = jax.devices()[0].memory_stats() or {}

    # Cross-check Meep py_do_harminv directly on the exact fdtdx GPU trace
    t_m0 = time.perf_counter()
    meep_on_fdtdx = _run_meep_trace(trace, dt_eff, f_lo, f_hi, Q_thresh=5.0, err_thresh=0.01)
    meep_post_ms = (time.perf_counter() - t_m0) * 1e3

    return {
        "dx_um": 0.02,
        "grid_shape": [int(v) for v in objects.volume.grid_shape],
        "time_steps_total": int(config.time_steps_total),
        "detector_samples": int(trace.shape[0]),
        "dt_eff_fs": round(dt_eff * 1e15, 6),
        "fwd_cold_s": round(fwd_cold_s, 4),
        "fwd_warm_med_s": round(float(statistics.median(fwd_samples)), 4),
        "grad_cold_s": round(grad_cold_s, 4),
        "grad_warm_med_s": round(float(statistics.median(grad_samples)), 4),
        "peak_gb": round(float(mem.get("peak_bytes_in_use", 0)) / 1e9, 4),
        "modes": bf_modes,
        "meep_on_fdtdx_trace": [
            {
                "frequency_thz": float(m["frequency"]) * 1e-12,
                "Q": float(m["Q"]),
                "amp": float(abs(m["amplitude"])),
                "error": float(m["error"]),
            }
            for m in meep_on_fdtdx
        ],
        "meep_post_ms": round(meep_post_ms, 4),
        "m6_freq_thz": float(f6_val),
        "df6_dl_autodiff_thz_per_um": float(df6_dl),
        "df6_dl_fd_thz_per_um": float(fd_df_dl),
        "df6_dl_analytic_thz_per_um": -float(f6_val) / l0,
    }


def _run_worker(suite: str) -> dict[str, Any]:
    if suite == "accuracy":
        return _suite_accuracy()
    if suite == "scaling_meep":
        return _suite_scaling_meep()
    if suite == "scaling_cpu":
        return _suite_scaling_jax("cpu")
    if suite == "scaling_gpu":
        return _suite_scaling_jax("gpu")
    if suite == "fdtd_meep":
        return _suite_fdtd_meep()
    if suite == "fdtd_bifrost":
        return _suite_fdtd_bifrost()
    raise ValueError(f"Unknown suite: {suite!r}")


def _spawn_suite(suite: str, env_overrides: dict[str, str]) -> dict[str, Any]:
    env = {**os.environ, **env_overrides}
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--suite", suite]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Worker suite={suite!r} failed (exit {proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
        )
    marker = "BENCH_RESULT_JSON:"
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    raise RuntimeError(f"Worker suite={suite!r} emitted no {marker} line.\nSTDOUT:\n{proc.stdout[-2000:]}")


def _bifrost_version_from_init() -> str:
    init_text = (REPO_ROOT / "bifrost" / "__init__.py").read_text()
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_text, re.M)
    return m.group(1) if m else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--suite",
        choices=["accuracy", "scaling_meep", "scaling_cpu", "scaling_gpu", "fdtd_meep", "fdtd_bifrost"],
    )
    parser.add_argument(
        "--only-suites",
        help="Comma-separated subset of suites to run (merging with existing --out JSON if present).",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.worker:
        if not args.suite:
            parser.error("--worker requires --suite")
        payload = _run_worker(args.suite)
        print("BENCH_RESULT_JSON:" + json.dumps(payload), flush=True)
        return

    all_suites = {"accuracy", "scaling_meep", "scaling_cpu", "scaling_gpu", "fdtd_meep", "fdtd_bifrost"}
    selected = set(all_suites)
    if args.only_suites:
        selected = {s.strip() for s in args.only_suites.split(",") if s.strip()}
        unknown = selected - all_suites
        if unknown:
            parser.error(f"Unknown suites in --only-suites: {sorted(unknown)}")

    existing: dict[str, Any] = {}
    if args.only_suites and args.out.exists():
        existing = json.loads(args.out.read_text())

    acc = existing.get("accuracy")
    if "accuracy" in selected:
        print("Running accuracy suite (CPU x64 + Meep)...", flush=True)
        acc = _spawn_suite("accuracy", {"JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1"})

    sc_existing = existing.get("scaling", {})
    sc_meep = sc_existing.get("meep")
    if "scaling_meep" in selected:
        print("Running Meep C scaling suite...", flush=True)
        sc_meep = _spawn_suite("scaling_meep", {"JAX_PLATFORMS": "cpu"})

    sc_cpu = sc_existing.get("cpu")
    if "scaling_cpu" in selected:
        print("Running bifrost CPU scaling suite...", flush=True)
        sc_cpu = _spawn_suite("scaling_cpu", {"JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1"})

    sc_gpu = sc_existing.get("gpu")
    if "scaling_gpu" in selected:
        print("Running bifrost GPU scaling suite...", flush=True)
        sc_gpu = _spawn_suite(
            "scaling_gpu",
            {"JAX_PLATFORMS": "cuda,cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "JAX_ENABLE_X64": "1"},
        )

    fdtd_existing = existing.get("fdtd_resonator", {})
    fdtd_meep = fdtd_existing.get("meep")
    if "fdtd_meep" in selected:
        print("Running Meep 1D FDTD + Harminv suite...", flush=True)
        fdtd_meep = _spawn_suite("fdtd_meep", {"JAX_PLATFORMS": "cpu"})

    fdtd_bf = fdtd_existing.get("bifrost_gpu")
    if "fdtd_bifrost" in selected:
        print("Running bifrost + fdtdx GPU 1D FDTD + harminv_detector + grad suite...", flush=True)
        fdtd_bf = _spawn_suite(
            "fdtd_bifrost",
            {"JAX_PLATFORMS": "cuda,cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "JAX_ENABLE_X64": "1"},
        )

    import importlib.metadata

    import meep as mp

    def _pkg_version(name: str, fallback: str = "unknown") -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return fallback

    gpu_device = (
        sc_gpu["device_kind"]
        if isinstance(sc_gpu, dict) and "device_kind" in sc_gpu
        else existing.get("metadata", {}).get("gpu_device", "unknown")
    )
    report = {
        "metadata": {
            "python": platform.python_version(),
            "cpu_model": _cpu_model(),
            "gpu_device": gpu_device,
            "jax": _pkg_version("jax"),
            "fdtdx": _pkg_version("fdtdx"),
            "meep": str(getattr(mp, "__version__", _pkg_version("pymeep"))),
            "bifrost": _pkg_version("bifrost", fallback=_bifrost_version_from_init()),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "accuracy": acc,
        "scaling": {
            "meep": sc_meep,
            "cpu": sc_cpu,
            "gpu": sc_gpu,
        },
        "fdtd_resonator": {
            "analytic": _analytic_fp_modes(),
            "meep": fdtd_meep,
            "bifrost_gpu": fdtd_bf,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote benchmark report to {args.out}", flush=True)


if __name__ == "__main__":
    main()
