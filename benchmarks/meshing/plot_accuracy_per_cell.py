"""Plot Pareto accuracy-vs-cell-count curves across six simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Palette from benchmarks/rectilinear/plot_dc_convergence.py (CVD-safe, high contrast on #fcfcfb)
SURFACE = "#fcfcfb"
INK = "#1f1f1e"
MUTED = "#6b6a63"
GRID = "#e6e5e0"
UNIF = "#2a78d6"
MESH = "#eb6834"
MESH_SUB = "#1baf7a"

TIDY3D_NEFF_REF = 2.381043200767292  # benchmarks/rectilinear/results/guide_tidy3d_reference.json
DC_S0_REF = 0.3558  # benchmarks/rectilinear/README.md joint uniform continuum fit


def _by_bench(rows: list[dict[str, Any]], bench: str) -> dict[str, dict[str, Any]]:
    return {r["kind"]: r for r in rows if r.get("bench") == bench}


def _style_ax(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    plain = ticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(plain)
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="medium", color=INK, pad=8)
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK)


def _plot_series(
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
    *,
    color: str,
    marker: str,
    label: str,
    ls: str = "-",
    zorder: float = 3,
) -> None:
    order = np.argsort(xs)
    x_arr = np.asarray(xs)[order]
    y_arr = np.maximum(np.asarray(ys)[order], 1e-7)
    ax.plot(
        x_arr,
        y_arr,
        color=color,
        marker=marker,
        linestyle=ls,
        linewidth=2.0,
        markersize=6.2,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        label=label,
        zorder=zorder,
    )


def _annotate_points(
    ax: plt.Axes,
    pts: list[tuple[float, float, str, tuple[int, int], str, str]],
) -> None:
    for x, y, text, offset, ha, va in pts:
        ax.annotate(
            text,
            (x, max(y, 1e-7)),
            textcoords="offset points",
            xytext=offset,
            ha=ha,
            va=va,
            fontsize=7.6,
            color=MUTED,
        )


def make_figure(src: Path, dsts: list[Path]) -> None:
    rows = json.loads(src.read_text())

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.6), facecolor=SURFACE)

    # (a) Anisotropic crystal slab
    ax = axes[0, 0]
    _style_ax(ax, "(a) Anisotropic crystal slab (1.55 µm)", "z cells", "max |t - t_Berreman|")
    slab = _by_bench(rows, "aniso_slab")
    u_slab = [r for k, r in slab.items() if k.startswith("uniform-")]
    m_slab = [r for k, r in slab.items() if k.startswith("mesh-") and float(k.split("-")[1]) >= 10]
    _plot_series(
        ax,
        [r["cells_z"] for r in u_slab],
        [r["t_error"] for r in u_slab],
        color=UNIF,
        marker="o",
        label="UniformGrid",
    )
    _plot_series(
        ax,
        [r["cells_z"] for r in m_slab],
        [r["t_error"] for r in m_slab],
        color=MESH,
        marker="s",
        label="bf.mesh (default)",
    )
    ax.set_xlim(28, 620)
    ax.set_ylim(3e-4, 3.0e-2)
    ax.set_xticks([40, 60, 100, 150, 240, 360, 480])
    ann_a = []
    for kind, off, ha, va in (("uniform-60", (5, 3), "left", "bottom"), ("uniform-10", (5, 4), "left", "bottom")):
        if kind in slab:
            r = slab[kind]
            ann_a.append((r["cells_z"], r["t_error"], f"{kind.split('-')[1]} nm", off, ha, va))
    for kind, off, ha, va in (("mesh-10", (-3, -6), "right", "top"), ("mesh-28", (-5, -5), "right", "top"), ("mesh-55", (5, 4), "left", "bottom")):
        if kind in slab:
            r = slab[kind]
            ann_a.append((r["cells_z"], r["t_error"], f"ppw {kind.split('-')[1]}", off, ha, va))
    _annotate_points(ax, ann_a)

    # (b) Sloped SOI waveguide
    ax = axes[0, 1]
    _style_ax(ax, "(b) Sloped SOI strip (10° walls)", "total cells (millions)", "|n_eff - n_eff,ref|")
    sg = _by_bench(rows, "sloped_guide")
    u_sg = [r for k, r in sg.items() if k.startswith("uniform-") and k != "uniform-12.5"]
    m_sg = [r for k, r in sg.items() if k.startswith("mesh-")]
    _plot_series(
        ax,
        [r["total_cells"] / 1e6 for r in u_sg],
        [abs(r["neff_mode"] - TIDY3D_NEFF_REF) for r in u_sg],
        color=UNIF,
        marker="o",
        label="UniformGrid",
    )
    _plot_series(
        ax,
        [r["total_cells"] / 1e6 for r in m_sg],
        [abs(r["neff_mode"] - TIDY3D_NEFF_REF) for r in m_sg],
        color=MESH,
        marker="s",
        label="bf.mesh (default)",
    )
    ax.set_xlim(0.16, 25)
    ax.set_ylim(8e-4, 3.6e-2)
    ann_b = []
    for kind, off, ha, va in (("uniform-50", (5, 4), "left", "bottom"), ("uniform-20", (5, 4), "left", "bottom"), ("uniform-16", (5, 3), "left", "bottom")):
        if kind in sg:
            r = sg[kind]
            ann_b.append((r["total_cells"] / 1e6, abs(r["neff_mode"] - TIDY3D_NEFF_REF), f"{kind.split('-')[1]} nm", off, ha, va))
    for kind, off, ha, va in (("mesh-12", (-5, -5), "right", "top"), ("mesh-22", (-5, -5), "right", "top"), ("mesh-34", (-5, -4), "right", "top")):
        if kind in sg:
            r = sg[kind]
            ann_b.append((r["total_cells"] / 1e6, abs(r["neff_mode"] - TIDY3D_NEFF_REF), f"ppw {kind.split('-')[1]}", off, ha, va))
    _annotate_points(ax, ann_b)

    # (c) Broadband SOI Bragg grating
    ax = axes[0, 2]
    _style_ax(ax, "(c) SOI Bragg grating (1.48–1.60 µm)", "total cells (millions)", "T(λ) RMSE vs ref")
    br = _by_bench(rows, "soi_bragg")
    br_ref_key = "uniform-12.5" if "uniform-12.5" in br else ("uniform-16" if "uniform-16" in br else None)
    if br_ref_key:
        br_ref = np.asarray(br[br_ref_key]["T_spectrum"])
        u_br = [r for k, r in br.items() if k.startswith("uniform-") and k != br_ref_key]
        m_br = [r for k, r in br.items() if k.startswith("mesh-")]
        _plot_series(
            ax,
            [r["total_cells"] / 1e6 for r in u_br],
            [float(np.sqrt(np.mean((np.asarray(r["T_spectrum"]) - br_ref) ** 2))) for r in u_br],
            color=UNIF,
            marker="o",
            label="UniformGrid",
        )
        _plot_series(
            ax,
            [r["total_cells"] / 1e6 for r in m_br],
            [float(np.sqrt(np.mean((np.asarray(r["T_spectrum"]) - br_ref) ** 2))) for r in m_br],
            color=MESH,
            marker="s",
            label="bf.mesh (default)",
        )
        ax.set_xlim(0.18, 28)
        ax.set_ylim(2.0e-3, 1.05e-1)
        ann_c = []
        for kind, off, ha, va in (("uniform-50", (0, 6), "center", "bottom"), ("uniform-20", (4, 4), "left", "bottom"), ("uniform-16", (-5, 3), "right", "bottom")):
            if kind in br:
                r = br[kind]
                err = float(np.sqrt(np.mean((np.asarray(r["T_spectrum"]) - br_ref) ** 2)))
                ann_c.append((r["total_cells"] / 1e6, err, f"{kind.split('-')[1]} nm", off, ha, va))
        for kind, off, ha, va in (("mesh-12", (-5, -5), "right", "top"), ("mesh-22", (-5, -5), "right", "top"), ("mesh-26", (5, 3), "left", "bottom")):
            if kind in br:
                r = br[kind]
                err = float(np.sqrt(np.mean((np.asarray(r["T_spectrum"]) - br_ref) ** 2)))
                ann_c.append((r["total_cells"] / 1e6, err, f"ppw {kind.split('-')[1]}", off, ha, va))
        _annotate_points(ax, ann_c)

    # (d) Elevator coupler
    ax = axes[1, 0]
    _style_ax(ax, "(d) Two-level SiN elevator coupler", "total cells (millions)", "|T - T_ref| at 619 nm")
    ec = _by_bench(rows, "elevator_coupler")
    ec_ref_key = "uniform-16" if "uniform-16" in ec else ("uniform-20" if "uniform-20" in ec else None)
    if ec_ref_key:
        ec_ref = ec[ec_ref_key]["T"]
        u_ec = [r for k, r in ec.items() if k.startswith("uniform-") and k != ec_ref_key]
        m_ec = [r for k, r in ec.items() if k.startswith("mesh-")]
        _plot_series(
            ax,
            [r["total_cells"] / 1e6 for r in u_ec],
            [abs(r["T"] - ec_ref) for r in u_ec],
            color=UNIF,
            marker="o",
            label="UniformGrid",
        )
        _plot_series(
            ax,
            [r["total_cells"] / 1e6 for r in m_ec],
            [abs(r["T"] - ec_ref) for r in m_ec],
            color=MESH,
            marker="s",
            label="bf.mesh (default)",
        )
        ax.set_xlim(0.38, 42)
        ax.set_ylim(6.0e-4, 8.5e-2)
        ann_d = []
        for kind, off, ha, va in (("uniform-50", (-4, -5), "right", "top"), ("uniform-25", (-5, -5), "right", "top")):
            if kind in ec:
                r = ec[kind]
                ann_d.append((r["total_cells"] / 1e6, abs(r["T"] - ec_ref), f"{kind.split('-')[1]} nm", off, ha, va))
        for kind, off, ha, va in (("mesh-8", (5, 4), "left", "bottom"), ("mesh-10", (-5, -5), "right", "top"), ("mesh-18", (-4, 5), "right", "bottom")):
            if kind in ec:
                r = ec[kind]
                ann_d.append((r["total_cells"] / 1e6, abs(r["T"] - ec_ref), f"ppw {kind.split('-')[1]}", off, ha, va))
        _annotate_points(ax, ann_d)

    # (e) Directional coupler
    ax = axes[1, 1]
    _style_ax(ax, "(e) SiN directional coupler (300 nm gap)", "total cells (millions)", "|S_619 - S_0|")
    dc = _by_bench(rows, "directional_coupler")
    u_dc = [r for k, r in dc.items() if k.startswith("uniform-") and float(k.split("-")[1]) <= 50]
    m_dc = [r for k, r in dc.items() if k.startswith("mesh-")]
    _plot_series(
        ax,
        [r["total_cells"] / 1e6 for r in u_dc],
        [abs(r["split_mid"] - DC_S0_REF) for r in u_dc],
        color=UNIF,
        marker="o",
        label="UniformGrid",
    )
    _plot_series(
        ax,
        [r["total_cells"] / 1e6 for r in m_dc],
        [abs(r["split_mid"] - DC_S0_REF) for r in m_dc],
        color=MESH,
        marker="s",
        label="bf.mesh (default)",
    )
    ax.set_xlim(4.4, 135)
    ax.set_ylim(1.8e-3, 1.45e-1)
    ann_e = []
    for kind, off, ha, va in (("uniform-50", (5, 3), "left", "bottom"), ("uniform-20", (4, 4), "left", "bottom")):
        if kind in dc:
            r = dc[kind]
            ann_e.append((r["total_cells"] / 1e6, abs(r["split_mid"] - DC_S0_REF), f"{kind.split('-')[1]} nm", off, ha, va))
    for kind, off, ha, va in (("mesh-6", (5, 3), "left", "bottom"), ("mesh-10", (-4, -5), "right", "top"), ("mesh-14", (-5, -5), "right", "top")):
        if kind in dc:
            r = dc[kind]
            ann_e.append((r["total_cells"] / 1e6, abs(r["split_mid"] - DC_S0_REF), f"ppw {kind.split('-')[1]}", off, ha, va))
    _annotate_points(ax, ann_e)

    # (f) SiN bend + Refine.material_cells substrate study
    ax = axes[1, 2]
    _style_ax(ax, "(f) Curved SiN bend on silica substrate", "total cells (millions)", "|T - T_ref| at 1.55 µm")
    sb = _by_bench(rows, "sin_bend")
    sb_ref_key = "uniform-24" if "uniform-24" in sb else ("uniform-30" if "uniform-30" in sb else None)
    if sb_ref_key:
        sb_ref = sb[sb_ref_key]["T"]
        u_sb = [r for k, r in sb.items() if k.startswith("uniform-") and k != sb_ref_key and float(k.split("-")[1]) <= 60]
        m_sb = [r for k, r in sb.items() if k.startswith("mesh-")]
        ms_sb = [r for k, r in sb.items() if k.startswith("meshsub-")]
        _plot_series(
            ax,
            [r["total_cells"] / 1e6 for r in u_sb],
            [abs(r["T"] - sb_ref) for r in u_sb],
            color=UNIF,
            marker="o",
            label="UniformGrid",
        )
        _plot_series(
            ax,
            [r["total_cells"] / 1e6 for r in m_sb],
            [abs(r["T"] - sb_ref) for r in m_sb],
            color=MESH,
            marker="s",
            label="bf.mesh (default)",
        )
        if ms_sb:
            _plot_series(
                ax,
                [r["total_cells"] / 1e6 for r in ms_sb],
                [abs(r["T"] - sb_ref) for r in ms_sb],
                color=MESH_SUB,
                marker="^",
                label="bf.mesh + material_cells",
            )
        ax.set_xlim(1.6, 36)
        ax.set_ylim(4.5e-5, 3.6e-2)
        ann_f = []
        for kind, off, ha, va in (("uniform-60", (-5, -4), "right", "top"), ("uniform-40", (5, -5), "left", "top"), ("uniform-30", (-4, 5), "right", "bottom")):
            if kind in sb:
                r = sb[kind]
                ann_f.append((r["total_cells"] / 1e6, abs(r["T"] - sb_ref), f"{kind.split('-')[1]} nm", off, ha, va))
        for kind, off, ha, va in (("meshsub-20", (-5, -3), "right", "top"), ("meshsub-24", (0, 14), "center", "bottom")):
            if kind in sb:
                r = sb[kind]
                ann_f.append((r["total_cells"] / 1e6, abs(r["T"] - sb_ref), f"ppw {kind.split('-')[1]} (mc)", off, ha, va))
        _annotate_points(ax, ann_f)
        ax.legend(frameon=False, loc="upper right", fontsize=7.6)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9.8, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    for dst in dsts:
        dst.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dst, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=root / "benchmarks/meshing/results/accuracy_per_cell.json")
    parser.add_argument(
        "--out",
        type=Path,
        nargs="+",
        default=[
            root / "docs/images/adaptive_meshing_accuracy.png",
            root / "benchmarks/meshing/results/adaptive_meshing_accuracy.png",
        ],
    )
    args = parser.parse_args()
    make_figure(args.src, args.out)


if __name__ == "__main__":
    main()
