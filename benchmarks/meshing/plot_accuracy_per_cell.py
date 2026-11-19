"""Plot accuracy-vs-cell-count curves across six simulations for both subpixel averaging modes.

Writes two figures: the error against each benchmark's reference (``adaptive_meshing_accuracy.png``) and
the measured quantity itself, with no reference subtracted (``adaptive_meshing_raw.png``). Solid filled
markers are ``averaging="subpixel"``; dashed hollow markers are ``averaging="subpixel_full"``."""

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


AVERAGINGS = ("subpixel", "subpixel_full")
# solid filled markers: averaging="subpixel"; dashed hollow markers: averaging="subpixel_full"
AVG_LS = {"subpixel": "-", "subpixel_full": "--"}
AVG_LABEL = {"subpixel": "subpixel", "subpixel_full": "subpixel_full"}


def _by_bench(rows: list[dict[str, Any]], bench: str, averaging: str) -> dict[str, dict[str, Any]]:
    return {r["kind"]: r for r in rows if r.get("bench") == bench and r.get("averaging") == averaging}


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
    hollow: bool = False,
    yscale_floor: float | None = 1e-7,
) -> None:
    if not xs:
        return
    order = np.argsort(xs)
    x_arr = np.asarray(xs)[order]
    y_arr = np.asarray(ys, dtype=float)[order]
    if yscale_floor is not None:
        y_arr = np.maximum(y_arr, yscale_floor)
    ax.plot(
        x_arr,
        y_arr,
        color=color,
        marker=marker,
        linestyle=ls,
        linewidth=2.0,
        markersize=7.0 if marker == "^" else 6.2,
        markeredgecolor=color if hollow else SURFACE,
        markerfacecolor=SURFACE if hollow else color,
        markeredgewidth=1.4 if hollow else (0.9 if marker == "^" else 1.2),
        label=label,
        zorder=zorder,
    )


def _annotate_points(
    ax: plt.Axes,
    pts: list[tuple[float, float, str, tuple[int, int], str, str]],
    *,
    yscale_floor: float | None = 1e-7,
) -> None:
    for x, y, text, offset, ha, va in pts:
        y_pos = max(y, yscale_floor) if yscale_floor is not None else y
        ax.annotate(
            text,
            (x, y_pos),
            textcoords="offset points",
            xytext=offset,
            ha=ha,
            va=va,
            fontsize=7.5,
            color=MUTED,
        )


def _rmse(r: dict[str, Any], ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(r["T_spectrum"]) - ref) ** 2)))


def _series(
    ax: plt.Axes, rows: list[dict[str, Any]], fx, fy, *, color: str, marker: str, label: str, avg: str,
    label_ends: bool | None = None, show_avg: bool = True, **kw: Any,
) -> None:
    """One curve; ``label_ends`` (default: only on ``"subpixel"`` curves) tags its coarsest and finest points."""
    if label_ends is None:
        label_ends = avg == "subpixel"
    if rows and label_ends:
        ends = sorted(rows, key=fx)
        for r, (dx, ha) in ((ends[0], (-5, "right")), (ends[-1], (5, "left"))):
            kind = r["kind"]
            size = kind.split("-", 1)[1]
            text = f"{size} nm" if kind.startswith("uniform-") else f"ppw {size}"
            y = fy(r)
            if kw.get("yscale_floor", 1e-7) is not None:
                y = max(y, 1e-7)
            ax.annotate(text, (fx(r), y), textcoords="offset points", xytext=(dx, 4), ha=ha, va="bottom", fontsize=7.2, color=MUTED)
    _plot_series(
        ax,
        [fx(r) for r in rows],
        [fy(r) for r in rows],
        color=color,
        marker=marker,
        label=f"{label} · {AVG_LABEL[avg]}" if show_avg else label,
        ls=AVG_LS[avg],
        hollow=avg == "subpixel_full",
        **kw,
    )


def _uniform_nm(kind: str) -> float:
    return float(kind.split("-")[1])


def _setup_rc() -> None:
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


def _finish(fig: plt.Figure, axes: np.ndarray, dsts: list[Path], *, show_avg: bool = True) -> None:
    from matplotlib.lines import Line2D

    present = {lab for ax in axes.flat for lab in ax.get_legend_handles_labels()[1]}
    handles, labels = [], []
    if show_avg:
        for avg in AVERAGINGS:
            for color, marker, name in ((UNIF, "o", "UniformGrid"), (MESH, "s", "bf.mesh"), (MESH_SUB, "^", "bf.mesh + material_cells")):
                lab = f"{name} · {AVG_LABEL[avg]}"
                if lab not in present:
                    continue
                hollow = avg == "subpixel_full"
                handles.append(
                    Line2D([], [], color=color, marker=marker, linestyle=AVG_LS[avg], linewidth=2.0, markersize=6.2,
                           markeredgecolor=color if hollow else SURFACE, markerfacecolor=SURFACE if hollow else color,
                           markeredgewidth=1.4 if hollow else 1.2)
                )
                labels.append(lab)
        # one legend column per averaging mode (matplotlib fills columns first)
        ncol = max(1, len({lab.rsplit(" · ", 1)[1] for lab in labels}))
        rect_top = 0.91
    else:
        for color, marker, name in ((UNIF, "o", "UniformGrid"), (MESH, "s", "bf.mesh (default)"), (MESH_SUB, "^", "bf.mesh + material_cells")):
            if name not in present:
                continue
            handles.append(
                Line2D([], [], color=color, marker=marker, linestyle="-", linewidth=2.0,
                       markersize=7.0 if marker == "^" else 6.2,
                       markeredgecolor=SURFACE, markerfacecolor=color,
                       markeredgewidth=0.9 if marker == "^" else 1.2)
            )
            labels.append(name)
        ncol = max(1, len(labels))
        rect_top = 0.94
    fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False, fontsize=9.2, bbox_to_anchor=(0.5, 0.997),
               handlelength=3.0)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, rect_top))
    for dst in dsts:
        dst.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dst, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def make_figure(src: Path, dsts: list[Path]) -> None:
    """Error against cell count, both averaging modes per panel. References: (a) Berreman, (b) tidy3d
    continuum, (e) joint uniform continuum fit; (c), (d), (f) the finest UniformGrid of the same averaging mode."""
    rows = json.loads(src.read_text())
    _setup_rc()
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.9), facecolor=SURFACE)
    mb = lambda r: r["total_cells"] / 1e6  # noqa: E731

    # (a) Anisotropic crystal slab: averaging="subpixel" cannot write off-diagonal material tensors
    ax = axes[0, 0]
    _style_ax(ax, "(a) Anisotropic crystal slab (1.55 µm)", "z cells", "max |t - t_Berreman|")
    for avg in AVERAGINGS:
        slab = _by_bench(rows, "aniso_slab", avg)
        u = [r for k, r in slab.items() if k.startswith("uniform-")]
        m = [r for k, r in slab.items() if k.startswith("mesh-") and float(k.split("-")[1]) >= 10]
        _series(ax, u, lambda r: r["cells_z"], lambda r: r["t_error"], color=UNIF, marker="o", label="UniformGrid", avg=avg, label_ends=True)
        _series(ax, m, lambda r: r["cells_z"], lambda r: r["t_error"], color=MESH, marker="s", label="bf.mesh", avg=avg, label_ends=True)
    ax.set_xlim(28, 620)
    ax.set_ylim(3e-4, 3.0e-2)
    ax.set_xticks([40, 60, 100, 150, 240, 360, 480])
    ax.text(0.97, 0.97, "subpixel_full only\n(off-diagonal material)", transform=ax.transAxes, ha="right", va="top", fontsize=7.6, color=MUTED)

    # (b) Sloped SOI waveguide: mode-solver n_eff against the tidy3d continuum reference (the FDTD
    # phase-slope n_eff is in the raw-value figure)
    ax = axes[0, 1]
    _style_ax(ax, "(b) Sloped SOI strip (10° walls)", "total cells (millions)", "|n_eff - n_eff,ref|")
    for avg in AVERAGINGS:
        sg = _by_bench(rows, "sloped_guide", avg)
        u = [r for k, r in sg.items() if k.startswith("uniform-") and k != "uniform-12.5"]
        m = [r for k, r in sg.items() if k.startswith("mesh-")]
        fy = lambda r: abs(r["neff_mode"] - TIDY3D_NEFF_REF)  # noqa: E731
        _series(ax, u, mb, fy, color=UNIF, marker="o", label="UniformGrid", avg=avg)
        _series(ax, m, mb, fy, color=MESH, marker="s", label="bf.mesh", avg=avg)

    # (c) Broadband SOI Bragg grating
    ax = axes[0, 2]
    _style_ax(ax, "(c) SOI Bragg grating (1.48–1.60 µm)", "total cells (millions)", "T(λ) RMSE vs uniform-12.5")
    for avg in AVERAGINGS:
        br = _by_bench(rows, "soi_bragg", avg)
        if "uniform-12.5" not in br:
            continue
        ref = np.asarray(br["uniform-12.5"]["T_spectrum"])
        u = [r for k, r in br.items() if k.startswith("uniform-") and k != "uniform-12.5"]
        m = [r for k, r in br.items() if k.startswith("mesh-")]
        _series(ax, u, mb, lambda r: _rmse(r, ref), color=UNIF, marker="o", label="UniformGrid", avg=avg)
        _series(ax, m, mb, lambda r: _rmse(r, ref), color=MESH, marker="s", label="bf.mesh", avg=avg)

    # (d) Elevator coupler
    ax = axes[1, 0]
    _style_ax(ax, "(d) Two-level SiN elevator coupler", "total cells (millions)", "|T - T(uniform-16)| at 619 nm")
    for avg in AVERAGINGS:
        ec = _by_bench(rows, "elevator_coupler", avg)
        if "uniform-16" not in ec:
            continue
        ref_t = ec["uniform-16"]["T"]
        u = [r for k, r in ec.items() if k.startswith("uniform-") and k != "uniform-16"]
        m = [r for k, r in ec.items() if k.startswith("mesh-")]
        _series(ax, u, mb, lambda r: abs(r["T"] - ref_t), color=UNIF, marker="o", label="UniformGrid", avg=avg)
        _series(ax, m, mb, lambda r: abs(r["T"] - ref_t), color=MESH, marker="s", label="bf.mesh", avg=avg)

    # (e) Directional coupler
    ax = axes[1, 1]
    _style_ax(ax, "(e) SiN directional coupler (300 nm gap)", "total cells (millions)", "|S_619 - S_0|")
    for avg in AVERAGINGS:
        dc = _by_bench(rows, "directional_coupler", avg)
        u = [r for k, r in dc.items() if k.startswith("uniform-") and _uniform_nm(k) <= 50]
        m = [r for k, r in dc.items() if k.startswith("mesh-")]
        fy = lambda r: abs(r["split_mid"] - DC_S0_REF)  # noqa: E731
        _series(ax, u, mb, fy, color=UNIF, marker="o", label="UniformGrid", avg=avg)
        _series(ax, m, mb, fy, color=MESH, marker="s", label="bf.mesh", avg=avg)

    # (f) SiN bend + Refine.material_cells substrate study
    ax = axes[1, 2]
    _style_ax(ax, "(f) Curved SiN bend on silica substrate", "total cells (millions)", "|T - T(uniform-24)| at 1.55 µm")
    for avg in AVERAGINGS:
        sb = _by_bench(rows, "sin_bend", avg)
        if "uniform-24" not in sb:
            continue
        ref_t = sb["uniform-24"]["T"]
        u = [r for k, r in sb.items() if k.startswith("uniform-") and k != "uniform-24" and _uniform_nm(k) <= 60]
        m = [r for k, r in sb.items() if k.startswith("mesh-")]
        ms = [r for k, r in sb.items() if k.startswith("meshsub-")]
        fy = lambda r: abs(r["T"] - ref_t)  # noqa: E731
        _series(ax, u, mb, fy, color=UNIF, marker="o", label="UniformGrid", avg=avg)
        _series(ax, m, mb, fy, color=MESH, marker="s", label="bf.mesh", avg=avg)
        _series(ax, ms, mb, fy, color=MESH_SUB, marker="^", label="bf.mesh + material_cells", avg=avg, label_ends=False)

    _finish(fig, axes, dsts)


def make_raw_figure(src: Path, dsts: list[Path]) -> None:
    """Actual simulated quantities (effective index, transmission, splitting ratio) against cell count."""
    rows = json.loads(src.read_text())
    _setup_rc()
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 8.0), facecolor=SURFACE)
    mb = lambda r: r["total_cells"] / 1e6  # noqa: E731
    panels = (
        (axes[0, 0], "(a) Sloped SOI strip: mode solver", "sloped_guide", "n_eff (mode solver)", lambda r: r["neff_mode"]),
        (axes[0, 1], "(b) Sloped SOI strip: FDTD phase slope", "sloped_guide", "n_eff (FDTD)", lambda r: r["neff_fdtd"]),
        (axes[0, 2], "(c) SOI Bragg grating: stop-band minimum", "soi_bragg", "min T(λ), 1.48–1.60 µm", lambda r: r["T_min"]),
        (axes[1, 0], "(d) Two-level SiN elevator coupler", "elevator_coupler", "T at 619 nm", lambda r: r["T"]),
        (axes[1, 1], "(e) SiN directional coupler (300 nm gap)", "directional_coupler", "splitting ratio S at 619 nm", lambda r: r["split_mid"]),
        (axes[1, 2], "(f) Curved SiN bend on silica substrate", "sin_bend", "T at 1.55 µm", lambda r: r["T"]),
    )
    coarse = {"sloped_guide": 64, "soi_bragg": 64, "elevator_coupler": 50, "directional_coupler": 50, "sin_bend": 60}
    present_avgs = {r.get("averaging") for r in rows if r.get("bench") in {p[2] for p in panels}}
    show_avg = len(present_avgs) > 1
    mesh_label = "bf.mesh" if show_avg else "bf.mesh (default)"

    annotations: dict[str, list[tuple[str, tuple[int, int], str, str, str]]] = {
        "sloped_guide_mode": [
            ("uniform-50", (5, -4), "left", "top", "50 nm"),
            ("uniform-20", (5, 5), "left", "bottom", "20 nm"),
            ("uniform-12.5", (0, -7), "center", "top", "12.5 nm"),
            ("mesh-8", (6, 2), "left", "bottom", "ppw 8"),
            ("mesh-22", (4, 7), "left", "bottom", "ppw 22"),
            ("mesh-34", (-5, -5), "right", "top", "ppw 34"),
        ],
        "sloped_guide_fdtd": [
            ("uniform-50", (-4, -5), "right", "top", "50 nm"),
            ("uniform-20", (5, 4), "left", "bottom", "20 nm"),
            ("uniform-12.5", (0, 6), "center", "bottom", "12.5 nm"),
            ("mesh-8", (6, 0), "left", "center", "ppw 8"),
            ("mesh-22", (-6, -7), "right", "top", "ppw 22"),
            ("mesh-34", (-5, -6), "right", "top", "ppw 34"),
        ],
        "soi_bragg": [
            ("uniform-64", (-4, -5), "right", "top", "64 nm"),
            ("uniform-20", (0, -6), "center", "top", "20 nm"),
            ("uniform-12.5", (-4, 5), "right", "bottom", "12.5 nm"),
            ("mesh-8", (5, 2), "left", "bottom", "ppw 8"),
            ("mesh-18", (-5, -5), "right", "top", "ppw 18"),
            ("mesh-26", (5, 4), "left", "bottom", "ppw 26"),
        ],
        "elevator_coupler": [
            ("uniform-50", (0, 6), "center", "bottom", "50 nm"),
            ("uniform-25", (4, 5), "left", "bottom", "25 nm"),
            ("uniform-16", (-4, 5), "right", "bottom", "16 nm"),
            ("mesh-6", (6, -1), "left", "center", "ppw 6"),
            ("mesh-10", (0, -6), "center", "top", "ppw 10"),
            ("mesh-18", (-4, -6), "right", "top", "ppw 18"),
        ],
        "directional_coupler": [
            ("uniform-50", (5, 2), "left", "bottom", "50 nm"),
            ("uniform-20", (0, 6), "center", "bottom", "20 nm"),
            ("mesh-6", (5, 2), "left", "bottom", "ppw 6"),
            ("mesh-10", (-5, -5), "right", "top", "ppw 10"),
            ("mesh-16", (-4, -6), "right", "top", "ppw 16"),
        ],
        "sin_bend": [
            ("uniform-60", (-4, -5), "right", "top", "60 nm"),
            ("uniform-40", (5, 4), "left", "bottom", "40 nm"),
            ("uniform-24", (-4, 5), "right", "bottom", "24 nm"),
            ("mesh-12", (5, 2), "left", "bottom", "ppw 12"),
            ("meshsub-20", (-5, -5), "right", "top", "ppw 20 (mc)"),
            ("mesh-24", (-6, -6), "right", "top", "ppw 24"),
        ],
    }

    for idx, (ax, title, bench, ylabel, fy) in enumerate(panels):
        _style_ax(ax, title, "total cells (millions)", ylabel)
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_yscale("linear")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
        ax.margins(x=0.08, y=0.14)
        for avg in AVERAGINGS:
            rr = _by_bench(rows, bench, avg)
            u = [r for k, r in rr.items() if k.startswith("uniform-") and _uniform_nm(k) <= coarse[bench] and fy(r) is not None]
            m = [r for k, r in rr.items() if k.startswith("mesh-")]
            ms = [r for k, r in rr.items() if k.startswith("meshsub-")]
            kw: dict[str, Any] = {"yscale_floor": None, "show_avg": show_avg}
            if not show_avg:
                kw["label_ends"] = False
            _series(ax, u, mb, fy, color=UNIF, marker="o", label="UniformGrid", avg=avg, **kw)
            _series(ax, m, mb, fy, color=MESH, marker="s", label=mesh_label, avg=avg, **kw)
            _series(ax, ms, mb, fy, color=MESH_SUB, marker="^", label="bf.mesh + material_cells", avg=avg, label_ends=False, yscale_floor=None, show_avg=show_avg)
        if not show_avg:
            ann_key = ("sloped_guide_mode", "sloped_guide_fdtd")[idx] if bench == "sloped_guide" else bench
            rr_sub = _by_bench(rows, bench, "subpixel")
            pts = []
            for kind, off, ha, va, text in annotations[ann_key]:
                if kind in rr_sub and fy(rr_sub[kind]) is not None:
                    r = rr_sub[kind]
                    pts.append((mb(r), fy(r), text, off, ha, va))
            _annotate_points(ax, pts, yscale_floor=None)
    _finish(fig, axes, dsts, show_avg=show_avg)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=root / "benchmarks/meshing/results/accuracy_per_cell.json")
    parser.add_argument(
        "--out",
        type=Path,
        nargs="+",
        default=[
            root / "benchmarks/meshing/results/adaptive_meshing_accuracy.png",
        ],
    )
    parser.add_argument(
        "--raw-out",
        type=Path,
        nargs="+",
        default=[
            root / "docs/images/adaptive_meshing_raw.png",
            root / "benchmarks/meshing/results/adaptive_meshing_raw.png",
        ],
    )
    args = parser.parse_args()
    make_figure(args.src, args.out)
    make_raw_figure(args.src, args.raw_out)


if __name__ == "__main__":
    main()
