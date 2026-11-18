"""Generate convergence and write-performance comparison plots from benchmark JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STYLES: dict[tuple[str, str], dict[str, Any]] = {
    ("greens", "subpixel_full"): dict(color="#1f77b4", marker="o", ls="-", lw=2.2, label="Green's (subpixel_full)"),
    ("greens", "subpixel"): dict(color="#2ca02c", marker="s", ls="-", lw=2.0, label="Green's (subpixel)"),
    ("legacy", "subpixel_full"): dict(color="#d62728", marker="^", ls="--", lw=1.8, label="Legacy (subpixel_full)"),
    ("legacy", "subpixel"): dict(color="#ff7f0e", marker="v", ls="--", lw=1.8, label="Legacy (subpixel)"),
}


def _plot_soi_bend(rows: list[dict[str, Any]], out_dir: Path) -> None:
    soi = [r for r in rows if r.get("bench") == "soi_bend"]
    grads = [r for r in rows if r.get("bench") == "soi_grad_check"]
    if not soi:
        return
    ref_rows = [r for r in soi if r["kind"] == "uniform-12" and r["shift_frac"] == 0.0]
    t_ref = ref_rows[0]["T"] if ref_rows else None

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)

    # Panel A: T vs resolution
    ax = axes[0]
    for key, st in STYLES.items():
        pts = sorted(
            [r for r in soi if r["shift_frac"] == 0.0 and (r["voxelizer"], r["averaging"]) == key and r["kind"] != "uniform-12"],
            key=lambda r: r["res_nm"],
        )
        if pts:
            ax.plot([p["res_nm"] for p in pts], [p["T"] for p in pts], **st)
    if t_ref is not None:
        ax.axhline(t_ref, color="black", ls=":", lw=1.4, label=f"Ref 12 nm ({t_ref:.4f})")
    ax.invert_xaxis()
    ax.set_xlabel("Grid spacing dx (nm)")
    ax.set_ylabel("TE0 Transmission |S21|^2")
    ax.set_title("SOI Tight Air-Clad Bend (R=2 um): Transmission")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Panel B: Log-log convergence error
    ax = axes[1]
    if t_ref is not None:
        for key, st in STYLES.items():
            pts = sorted(
                [r for r in soi if r["shift_frac"] == 0.0 and (r["voxelizer"], r["averaging"]) == key and r["kind"] != "uniform-12"],
                key=lambda r: r["res_nm"],
            )
            if pts:
                xs = np.array([p["res_nm"] for p in pts])
                errs = np.maximum(np.abs(np.array([p["T"] for p in pts]) - t_ref), 1e-6)
                ax.loglog(xs, errs, **st)
        xs_g = np.array([15.0, 40.0])
        ax.loglog(xs_g, 0.015 * (xs_g / 40.0) ** 2, color="gray", ls="-.", alpha=0.7, label="O(dx^2)")
    ax.set_xlabel("Grid spacing dx (nm)")
    ax.set_ylabel("|T - T_ref|")
    ax.set_title("Convergence Rate vs 12 nm Reference")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    # Panel C: Sub-cell shift jitter at 30 nm
    ax = axes[2]
    for key, st in STYLES.items():
        pts = sorted(
            [r for r in soi if r["kind"] == "uniform-30" and (r["voxelizer"], r["averaging"]) == key],
            key=lambda r: r["shift_frac"],
        )
        if pts:
            vals = [p["T"] for p in pts]
            ptp = max(vals) - min(vals)
            st_lbl = {**st, "label": f"{st['label']} (pp={ptp:.4f})"}
            ax.plot([p["shift_frac"] for p in pts], vals, **st_lbl)
    if grads:
        txt = "\n".join(
            f"{g['averaging']}: AD={g['grad_ad']:.4f} FD={g['grad_fd']:.4f} (rel={g['rel_err']:.2%})"
            for g in grads
        )
        ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=7.5, va="bottom", bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.set_xlabel("Diagonal sub-cell shift (fraction of 30 nm cell)")
    ax.set_ylabel("TE0 Transmission |S21|^2")
    ax.set_title("Sub-Cell Grid Shift Sensitivity (dx = 30 nm)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)

    fig.savefig(out_dir / "soi_bend_convergence.png", dpi=200)
    plt.close(fig)


def _plot_sin_bend(rows: list[dict[str, Any]], out_dir: Path) -> None:
    sin = [r for r in rows if r.get("bench") == "sin_bend"]
    if not sin:
        return
    ref_u = [r for r in sin if r["kind"] == "uniform-24"]
    ref_m = [r for r in sin if r["kind"] == "mesh-28"]
    t_ref = (ref_u[0]["T"] if ref_u else (ref_m[0]["T"] if ref_m else None))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), constrained_layout=True)

    # Panel A: Uniform T vs dx
    ax = axes[0]
    for key, st in STYLES.items():
        pts = sorted(
            [r for r in sin if r["kind"].startswith("uniform-") and r["kind"] != "uniform-24" and (r["voxelizer"], r["averaging"]) == key],
            key=lambda r: float(r["kind"].split("-")[1]),
        )
        if pts:
            ax.plot([float(p["kind"].split("-")[1]) for p in pts], [p["T"] for p in pts], **st)
    if t_ref is not None:
        ax.axhline(t_ref, color="black", ls=":", lw=1.4, label=f"Ref ({t_ref:.4f})")
    ax.invert_xaxis()
    ax.set_xlabel("Uniform spacing dx (nm)")
    ax.set_ylabel("TE0 Transmission |S21|^2")
    ax.set_title("Air-Clad SiN tight_bend (R=7 um): Uniform Grids")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Panel B: T vs Mcells comparing uniform vs bf.mesh
    ax = axes[1]
    for key, st in STYLES.items():
        pts_u = sorted(
            [r for r in sin if r["kind"].startswith("uniform-") and r["kind"] != "uniform-24" and (r["voxelizer"], r["averaging"]) == key],
            key=lambda r: r["total_cells"],
        )
        pts_m = sorted(
            [r for r in sin if r["kind"].startswith("mesh-") and r["kind"] != "mesh-28" and (r["voxelizer"], r["averaging"]) == key],
            key=lambda r: r["total_cells"],
        )
        if pts_u:
            ax.semilogx([p["total_cells"] / 1e6 for p in pts_u], [p["T"] for p in pts_u], **{**st, "alpha": 0.45, "label": f"{st['label']} [unif]"})
        if pts_m:
            ax.semilogx([p["total_cells"] / 1e6 for p in pts_m], [p["T"] for p in pts_m], **{**st, "ls": "-", "marker": "D" if key[0] == "greens" else "x", "label": f"{st['label']} [bf.mesh]"})
    if t_ref is not None:
        ax.axhline(t_ref, color="black", ls=":", lw=1.4)
    ax.set_xlabel("Total grid cells (millions)")
    ax.set_ylabel("TE0 Transmission |S21|^2")
    ax.set_title("Uniform vs Graded bf.mesh Efficiency")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)

    fig.savefig(out_dir / "sin_bend_convergence.png", dpi=200)
    plt.close(fig)


def _plot_dc(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    ref_m16 = [r for r in rows if r["kind"] == "mesh-16" and r["voxelizer"] == "greens"]
    ref_unif = [r for r in rows if r["kind"] == "uniform-20" and r["voxelizer"] == "greens"]
    ref = ref_m16[0] if ref_m16 else (ref_unif[0] if ref_unif else None)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)

    # Panel A: Center split vs Mcells (separate uniform vs bf.mesh curves)
    ax = axes[0]
    for key, st in STYLES.items():
        pts_u = sorted(
            [r for r in rows if r["kind"].startswith("uniform-") and "%" not in r["kind"] and r["kind"] != "uniform-20" and (r["voxelizer"], r["averaging"]) == key],
            key=lambda r: r["total_cells"],
        )
        pts_m = sorted(
            [r for r in rows if r["kind"].startswith("mesh-") and r["kind"] != "mesh-16" and (r["voxelizer"], r["averaging"]) == key],
            key=lambda r: r["total_cells"],
        )
        if pts_u:
            ax.semilogx([p["total_cells"] / 1e6 for p in pts_u], [p["split_mid"] for p in pts_u], **{**st, "alpha": 0.45, "label": f"{st['label']} [unif]"})
        if pts_m:
            ax.semilogx([p["total_cells"] / 1e6 for p in pts_m], [p["split_mid"] for p in pts_m], **{**st, "ls": "-", "marker": "D" if key[0] == "greens" else "x", "label": f"{st['label']} [bf.mesh]"})
    ax.axhline(0.3558, color="gray", ls="-.", lw=1.2, label="Continuum fit S0 (0.3558)")
    if ref is not None:
        ax.axhline(ref["split_mid"], color="black", ls=":", lw=1.4, label=f"Ref ({ref['kind']}: {ref['split_mid']:.4f})")
    ax.set_xlabel("Total grid cells (millions)")
    ax.set_ylabel("Cross Split Ratio @ 619 nm")
    ax.set_title("Directional Coupler Split Convergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=6.5)

    # Panel B: Broadband spectral RMS error vs reference (separate uniform vs bf.mesh curves)
    ax = axes[1]
    if ref is not None:
        ref_cross = np.array(ref["cross"])
        for key, st in STYLES.items():
            pts_u = sorted(
                [r for r in rows if r["kind"].startswith("uniform-") and "%" not in r["kind"] and r["kind"] != "uniform-20" and (r["voxelizer"], r["averaging"]) == key],
                key=lambda r: r["total_cells"],
            )
            pts_m = sorted(
                [r for r in rows if r["kind"].startswith("mesh-") and r["kind"] != "mesh-16" and (r["voxelizer"], r["averaging"]) == key],
                key=lambda r: r["total_cells"],
            )
            if pts_u:
                rms_u = [float(np.sqrt(np.mean((np.array(p["cross"]) - ref_cross) ** 2))) for p in pts_u]
                ax.loglog([p["total_cells"] / 1e6 for p in pts_u], np.maximum(rms_u, 1e-6), **{**st, "alpha": 0.45, "label": f"{st['label']} [unif]"})
            if pts_m:
                rms_m = [float(np.sqrt(np.mean((np.array(p["cross"]) - ref_cross) ** 2))) for p in pts_m]
                ax.loglog([p["total_cells"] / 1e6 for p in pts_m], np.maximum(rms_m, 1e-6), **{**st, "ls": "-", "marker": "D" if key[0] == "greens" else "x", "label": f"{st['label']} [bf.mesh]"})
    ax.set_xlabel("Total grid cells (millions)")
    ax.set_ylabel("Cross Spectrum RMSE vs Reference")
    ax.set_title("Broadband Spectral Error (538–728 nm)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=6.5)

    # Panel C: Sub-cell shift jitter at 40 nm
    ax = axes[2]
    for key, st in STYLES.items():
        pts = sorted(
            [r for r in rows if r["kind"].startswith("uniform-40%") and (r["voxelizer"], r["averaging"]) == key],
            key=lambda r: float(r["kind"].split("%")[1]),
        )
        if pts:
            fracs = [float(p["kind"].split("%")[1]) for p in pts]
            vals = [p["split_mid"] for p in pts]
            pp = max(vals) - min(vals)
            ax.plot(fracs, vals, **{**st, "label": f"{st['label']} (pp={pp:.4f})"})
    ax.set_xlabel("Core face sub-cell fraction f (40 nm)")
    ax.set_ylabel("Cross Split Ratio @ 619 nm")
    ax.set_title("Sub-Cell Boundary Shift Jitter (40 nm)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)

    fig.savefig(out_dir / "dc_convergence.png", dpi=200)
    plt.close(fig)


def _plot_write_perf(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)

    grids = sorted({tuple(r["grid_shape"]) for r in rows})
    combos = [(vox, avg) for vox in ("greens", "legacy") for avg in ("subpixel", "subpixel_full")]
    labels = [f"{vox}\n{avg}" for vox, avg in combos]
    x = np.arange(len(combos))
    w = 0.35

    ax = axes[0]
    for gi, gshape in enumerate(grids):
        g_lbl = "x".join(str(v) for v in gshape)
        vals, errs = [], []
        for vox, avg in combos:
            match = [r for r in rows if tuple(r["grid_shape"]) == gshape and r["voxelizer"] == vox and r["averaging"] == avg]
            vals.append(match[0]["write_med_s"] * 1e3 if match else 0.0)
            errs.append(match[0]["write_iqr_s"] * 1e3 if match else 0.0)
        ax.bar(x + (gi - 0.5 * (len(grids) - 1)) * w, vals, w, yerr=errs, capsize=3, label=f"Forward {g_lbl}")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Warm Write Time (ms)")
    ax.set_title("StackDevice.write Warm Median (+/- IQR)")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for gi, gshape in enumerate(grids):
        g_lbl = "x".join(str(v) for v in gshape)
        vals, errs = [], []
        for vox, avg in combos:
            match = [r for r in rows if tuple(r["grid_shape"]) == gshape and r["voxelizer"] == vox and r["averaging"] == avg]
            vals.append(match[0]["grad_med_s"] * 1e3 if match else 0.0)
            errs.append(match[0]["grad_iqr_s"] * 1e3 if match else 0.0)
        ax.bar(x + (gi - 0.5 * (len(grids) - 1)) * w, vals, w, yerr=errs, capsize=3, label=f"Grad {g_lbl}")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Warm Grad Time (ms)")
    ax.set_title("jax.grad(write) Warm Median (+/- IQR)")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    max_gb = 0.0
    for gi, gshape in enumerate(grids):
        g_lbl = "x".join(str(v) for v in gshape)
        vals, cold_lbls = [], []
        for vox, avg in combos:
            match = [r for r in rows if tuple(r["grid_shape"]) == gshape and r["voxelizer"] == vox and r["averaging"] == avg]
            if match:
                gb = float(match[0].get("peak_gb") or 0.0)
                cold_s = float(match[0]["place_s"]) + float(match[0]["write_cold_s"])
                vals.append(gb)
                cold_lbls.append(f"{cold_s:.1f}s")
                max_gb = max(max_gb, gb)
            else:
                vals.append(0.0)
                cold_lbls.append("")
        xpos = x + (gi - 0.5 * (len(grids) - 1)) * w
        bars = ax.bar(xpos, vals, w, label=g_lbl)
        for bar, lbl in zip(bars, cold_lbls):
            if lbl:
                ax.text(
                    bar.get_x() + 0.5 * bar.get_width(),
                    bar.get_height(),
                    lbl,
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                )
    if max_gb > 0.0:
        ax.set_ylim(0.0, max_gb * 1.20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Peak GPU Memory (GB)")
    ax.set_title("Peak GPU Memory & Cold Setup (place + write)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    fig.savefig(out_dir / "write_perf_compare.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="benchmarks/greens/results")
    parser.add_argument("--out-dir", default="benchmarks/greens/figures")
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bend_f = res_dir / "bend_convergence.json"
    if bend_f.exists():
        bend_rows = json.loads(bend_f.read_text())
        _plot_soi_bend(bend_rows, out_dir)
        _plot_sin_bend(bend_rows, out_dir)

    dc_f = res_dir / "dc_convergence.json"
    if dc_f.exists():
        _plot_dc(json.loads(dc_f.read_text()), out_dir)

    write_f = res_dir / "bench_write.json"
    if write_f.exists():
        _plot_write_perf(json.loads(write_f.read_text()), out_dir)


if __name__ == "__main__":
    main()
