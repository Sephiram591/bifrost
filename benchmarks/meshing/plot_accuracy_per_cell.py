"""Plot accuracy and simulated convergence curves across six simulations for bifrost and MEEP.

Writes three figures: peak GPU VRAM (``adaptive_meshing_vram.png``), grid cell count
(``adaptive_meshing_raw.png``), and error against reference (``adaptive_meshing_accuracy.png``).
Solid filled markers are ``averaging="subpixel"``; dashed hollow markers are
``averaging="subpixel_full"``; purple dash-dot diamonds are MEEP on a uniform grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

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
MEEP = "#7a49a5"

TIDY3D_NEFF_REF = 2.381043200767292  # benchmarks/rectilinear/results/guide_tidy3d_reference.json
DC_S0_REF = 0.3558  # benchmarks/rectilinear/README.md joint uniform continuum fit


AVERAGINGS = ("subpixel", "subpixel_full")
# solid filled markers: averaging="subpixel"; dashed hollow markers: averaging="subpixel_full"
AVG_LS = {"subpixel": "-", "subpixel_full": "--"}
AVG_LABEL = {"subpixel": "subpixel", "subpixel_full": "subpixel_full"}


def _load_meep(src: Path, meep_src: Path | None) -> list[dict[str, Any]]:
    candidate = src.parent / "meep_uniform.json" if meep_src is None else meep_src
    return json.loads(candidate.read_text()) if candidate.exists() else []


def _by_bench(rows: list[dict[str, Any]], bench: str, averaging: str) -> dict[str, dict[str, Any]]:
    return {r["kind"]: r for r in rows if r.get("bench") == bench and r.get("averaging") == averaging}


def _meep_by_bench(rows: list[dict[str, Any]], bench: str) -> dict[str, dict[str, Any]]:
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


def _markersize(marker: str) -> float:
    if marker == "^":
        return 7.0
    if marker == "D":
        return 5.4
    return 6.2


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
        markersize=_markersize(marker),
        markeredgecolor=color if hollow else SURFACE,
        markerfacecolor=SURFACE if hollow else color,
        markeredgewidth=1.4 if hollow else (0.9 if marker in ("^", "D") else 1.2),
        label=label,
        zorder=zorder,
    )


def _annotate_points(
    ax: plt.Axes,
    pts: list[tuple[float, float, str, tuple[int, int], str, str]],
    *,
    yscale_floor: float | None = 1e-7,
    fontsize: float = 7.5,
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
            fontsize=fontsize,
            color=MUTED,
        )


def _series(
    ax: plt.Axes, rows: list[dict[str, Any]], fx, fy, *, color: str, marker: str, label: str, avg: str | None,
    label_ends: bool | None = None, show_avg: bool = True, ls: str | None = None,
    start_offset: tuple[tuple[int, int], str, str] = ((-4, 4), "right", "bottom"),
    fin_offset: tuple[tuple[int, int], str, str] = ((5, 4), "left", "bottom"),
    **kw: Any,
) -> None:
    """One curve; ``label_ends`` (default: only on ``"subpixel"`` curves) tags its coarsest and finest points."""
    if label_ends is None:
        label_ends = avg == "subpixel"
    if rows and label_ends:
        ends = sorted(rows, key=fx)
        end_specs = (
            (ends[0], start_offset[0], start_offset[1], start_offset[2]),
            (ends[-1], fin_offset[0], fin_offset[1], fin_offset[2]),
        )
        for r, off, ha, va in end_specs:
            kind = r["kind"]
            size = kind.split("-", 1)[1]
            text = f"{size} nm" if kind.startswith("uniform-") else f"ppw {size}"
            y = fy(r)
            if kw.get("yscale_floor", 1e-7) is not None:
                y = max(y, 1e-7)
            ax.annotate(text, (fx(r), y), textcoords="offset points", xytext=off, ha=ha, va=va, fontsize=7.2, color=MUTED)
    resolved_ls = ls if ls is not None else AVG_LS[avg or "subpixel"]
    resolved_label = f"{label} · {AVG_LABEL[avg]}" if (show_avg and avg is not None) else label
    _plot_series(
        ax,
        [fx(r) for r in rows],
        [fy(r) for r in rows],
        color=color,
        marker=marker,
        label=resolved_label,
        ls=resolved_ls,
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


def _finish(
    fig: plt.Figure,
    axes: np.ndarray,
    dsts: list[Path],
    *,
    show_avg: bool = True,
    meep_label: str = "MEEP · UniformGrid",
) -> None:
    from matplotlib.lines import Line2D

    present = {lab for ax in axes.flat for lab in ax.get_legend_handles_labels()[1]}
    handles, labels = [], []
    if show_avg:
        # Pair (subpixel, subpixel_full) per meshing family so column-major ncol=4 aligns modes cleanly.
        for color, marker, name in ((UNIF, "o", "UniformGrid"), (MESH, "s", "bf.mesh"), (MESH_SUB, "^", "bf.mesh + material_cells")):
            for avg in AVERAGINGS:
                lab = f"{name} · {AVG_LABEL[avg]}"
                if lab not in present:
                    continue
                hollow = avg == "subpixel_full"
                handles.append(
                    Line2D([], [], color=color, marker=marker, linestyle=AVG_LS[avg], linewidth=2.0, markersize=_markersize(marker),
                           markeredgecolor=color if hollow else SURFACE, markerfacecolor=SURFACE if hollow else color,
                           markeredgewidth=1.4 if hollow else 1.2)
                )
                labels.append(lab)
        if meep_label in present:
            handles.append(
                Line2D([], [], color=MEEP, marker="D", linestyle="-.", linewidth=2.0, markersize=_markersize("D"),
                       markeredgecolor=SURFACE, markerfacecolor=MEEP, markeredgewidth=0.9)
            )
            labels.append(meep_label)
        ncol = 4 if meep_label in present else 3
        rect_top = 0.905
        leg_kw: dict[str, Any] = {"fontsize": 8.6, "handlelength": 2.7, "columnspacing": 1.3}
    else:
        for color, marker, name, ls in (
            (UNIF, "o", "UniformGrid", "-"),
            (MESH, "s", "bf.mesh (default)", "-"),
            (MESH_SUB, "^", "bf.mesh + material_cells", "-"),
            (MEEP, "D", meep_label, "-."),
        ):
            if name not in present:
                continue
            handles.append(
                Line2D([], [], color=color, marker=marker, linestyle=ls, linewidth=2.0,
                       markersize=_markersize(marker),
                       markeredgecolor=SURFACE, markerfacecolor=color,
                       markeredgewidth=0.9 if marker in ("^", "D") else 1.2)
            )
            labels.append(name)
        ncol = max(1, len(labels))
        rect_top = 0.94
        leg_kw = {"fontsize": 9.2, "handlelength": 3.0}
    fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.997), **leg_kw)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, rect_top))
    for dst in dsts:
        dst.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dst, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def _plot_self_conv_panel(
    ax: plt.Axes,
    title: str,
    ylabel: str,
    rows: list[dict[str, Any]],
    meep_rows: list[dict[str, Any]],
    bench: str,
    *,
    max_unif_nm: float | None = None,
    has_meshsub: bool = False,
    mesh_start_offset: tuple[tuple[int, int], str, str] = ((-4, 4), "right", "bottom"),
) -> None:
    _style_ax(ax, title, "total cells (millions)", ylabel)
    ax.margins(x=0.14, y=0.16)
    mb = lambda r: r["total_cells"] / 1e6  # noqa: E731
    in_range = lambda k: k.startswith("uniform-") and (max_unif_nm is None or _uniform_nm(k) <= max_unif_nm)  # noqa: E731
    for avg in AVERAGINGS:
        d = _by_bench(rows, bench, avg)
        u_keys = [k for k in d if in_range(k)]
        if not u_keys:
            continue
        finest_u = min(u_keys, key=_uniform_nm)
        ref_t = d[finest_u]["T"]
        u = [d[k] for k in u_keys if k != finest_u]
        m = [r for k, r in d.items() if k.startswith("mesh-")]
        fy = lambda r, rt=ref_t: abs(r["T"] - rt)  # noqa: E731
        _series(ax, u, mb, fy, color=UNIF, marker="o", label="UniformGrid", avg=avg, fin_offset=((0, 7), "center", "bottom"))
        _series(ax, m, mb, fy, color=MESH, marker="s", label="bf.mesh", avg=avg, start_offset=mesh_start_offset)
        if has_meshsub:
            ms = [r for k, r in d.items() if k.startswith("meshsub-")]
            _series(ax, ms, mb, fy, color=MESH_SUB, marker="^", label="bf.mesh + material_cells", avg=avg, label_ends=False)
    md = _meep_by_bench(meep_rows, bench)
    m_keys = [k for k in md if in_range(k)]
    if len(m_keys) > 1:
        m_finest = min(m_keys, key=_uniform_nm)
        m_ref = md[m_finest]["T"]
        m_u = [md[k] for k in m_keys if k != m_finest]
        _series(
            ax, m_u, mb, lambda r, rt=m_ref: abs(r["T"] - rt),
            color=MEEP, marker="D", ls="-.", label="MEEP · UniformGrid", avg=None, label_ends=False, zorder=2.8,
        )


def make_figure(src: Path, dsts: list[Path], *, meep_src: Path | None = None) -> None:
    """Error against cell count for bifrost (both averaging modes) and MEEP UniformGrid.

    References: (a) Berreman, (b) tidy3d continuum, (e) joint uniform continuum fit; (c), (d), (f)
    each solver/averaging mode's own finest UniformGrid.
    """
    rows = json.loads(src.read_text())
    meep_rows = _load_meep(src, meep_src)
    _setup_rc()
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 8.0), facecolor=SURFACE)
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
    m_slab = list(_meep_by_bench(meep_rows, "aniso_slab").values())
    _series(ax, m_slab, lambda r: r["cells_z"], lambda r: r["t_error"], color=MEEP, marker="D", ls="-.", label="MEEP · UniformGrid", avg=None, label_ends=False, zorder=2.8)
    ax.set_xlim(28, 620)
    ax.set_ylim(3e-4, 3.0e-2)
    ax.set_xticks([40, 60, 100, 150, 240, 360, 480])
    ax.text(0.97, 0.97, "subpixel_full only\n(off-diagonal material)", transform=ax.transAxes, ha="right", va="top", fontsize=7.6, color=MUTED)

    # (b) Sloped SOI waveguide: mode-solver n_eff against the tidy3d continuum reference
    ax = axes[0, 1]
    _style_ax(ax, "(b) Sloped SOI strip (10° walls)", "total cells (millions)", "|n_eff - n_eff,ref|")
    ax.margins(x=0.14, y=0.16)
    fy_sg = lambda r: abs(r["neff_mode"] - TIDY3D_NEFF_REF)  # noqa: E731
    for avg in AVERAGINGS:
        sg = _by_bench(rows, "sloped_guide", avg)
        u = [r for k, r in sg.items() if k.startswith("uniform-")]
        m = [r for k, r in sg.items() if k.startswith("mesh-")]
        _series(ax, u, mb, fy_sg, color=UNIF, marker="o", label="UniformGrid", avg=avg)
        _series(ax, m, mb, fy_sg, color=MESH, marker="s", label="bf.mesh", avg=avg)
    m_sg = list(_meep_by_bench(meep_rows, "sloped_guide").values())
    _series(ax, m_sg, mb, fy_sg, color=MEEP, marker="D", ls="-.", label="MEEP · UniformGrid", avg=None, label_ends=False, zorder=2.8)

    # (c) SiN bend (moved from (f) replacing soi_bragg)
    _plot_self_conv_panel(
        axes[0, 2], "(c) Curved SiN bend on silica substrate", "|T - T(finest uniform)| at 1.55 µm",
        rows, meep_rows, "sin_bend", max_unif_nm=60.0, has_meshsub=True,
    )

    # (d) Elevator coupler
    _plot_self_conv_panel(
        axes[1, 0], "(d) Two-level SiN elevator coupler", "|T - T(finest uniform)| at 619 nm",
        rows, meep_rows, "elevator_coupler", max_unif_nm=None, has_meshsub=False,
        mesh_start_offset=((6, 3), "left", "bottom"),
    )

    # (e) Directional coupler
    ax = axes[1, 1]
    _style_ax(ax, "(e) SiN directional coupler (300 nm gap)", "total cells (millions)", "|S_619 - S_0|")
    ax.margins(x=0.14, y=0.16)
    fy_dc = lambda r: abs(r["split_mid"] - DC_S0_REF)  # noqa: E731
    for avg in AVERAGINGS:
        dc = _by_bench(rows, "directional_coupler", avg)
        u = [r for k, r in dc.items() if k.startswith("uniform-") and _uniform_nm(k) <= 50]
        m = [r for k, r in dc.items() if k.startswith("mesh-")]
        _series(ax, u, mb, fy_dc, color=UNIF, marker="o", label="UniformGrid", avg=avg)
        _series(ax, m, mb, fy_dc, color=MESH, marker="s", label="bf.mesh", avg=avg)
    m_dc = [r for k, r in _meep_by_bench(meep_rows, "directional_coupler").items() if k.startswith("uniform-") and _uniform_nm(k) <= 50]
    _series(ax, m_dc, mb, fy_dc, color=MEEP, marker="D", ls="-.", label="MEEP · UniformGrid", avg=None, label_ends=False, zorder=2.8)

    # (f) X-cut LiNbO3 bend (20 deg sidewalls, anisotropic)
    _plot_self_conv_panel(
        axes[1, 2], "(f) X-cut LiNbO3 bend (20° walls)", "|T - T(finest uniform)| at 1.55 µm",
        rows, meep_rows, "linbo3_bend", max_unif_nm=60.0, has_meshsub=True,
    )

    _finish(fig, axes, dsts)


def _make_value_figure(src: Path, dsts: list[Path], *, x_axis: Literal["cells", "vram"], meep_src: Path | None = None) -> None:
    """Actual simulated quantities (effective index, transmission, splitting ratio) against cells or peak VRAM."""
    rows = json.loads(src.read_text())
    meep_rows = _load_meep(src, meep_src)
    _setup_rc()
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 8.0), facecolor=SURFACE)
    if x_axis == "cells":
        fx = lambda r: r["total_cells"] / 1e6  # noqa: E731
        mfx = fx
        xlabel = "total cells (millions)"
        x_margin = 0.14
        meep_label = "MEEP · UniformGrid"
    elif x_axis == "vram":
        fx = lambda r: r.get("peak_gb")  # noqa: E731
        mfx = lambda r: r.get("equiv_peak_gb")  # noqa: E731
        xlabel = "peak / equiv. VRAM (GB)"
        x_margin = 0.15
        meep_label = "MEEP · UniformGrid (equiv. VRAM)"
    else:
        raise ValueError(f"Unknown x_axis {x_axis!r}")
    panels = (
        (axes[0, 0], "(a) Sloped SOI strip: mode solver", "sloped_guide", "n_eff (mode solver)", lambda r: r["neff_mode"]),
        (axes[0, 1], "(b) Sloped SOI strip: FDTD phase slope", "sloped_guide", "n_eff (FDTD)", lambda r: r["neff_fdtd"]),
        (axes[0, 2], "(c) Curved SiN bend on silica substrate", "sin_bend", "T at 1.55 µm", lambda r: r["T"]),
        (axes[1, 0], "(d) Two-level SiN elevator coupler", "elevator_coupler", "T at 619 nm", lambda r: r["T"]),
        (axes[1, 1], "(e) SiN directional coupler (300 nm gap)", "directional_coupler", "splitting ratio S at 619 nm", lambda r: r["split_mid"]),
        (axes[1, 2], "(f) X-cut LiNbO3 bend (20° walls)", "linbo3_bend", "TE0 T at 1.55 µm", lambda r: r["T"]),
    )
    coarse = {"sloped_guide": 64, "sin_bend": 60, "elevator_coupler": 50, "directional_coupler": 50, "linbo3_bend": 60}
    present_avgs = {r.get("averaging") for r in rows if r.get("bench") in {p[2] for p in panels}}
    show_avg = len(present_avgs) > 1
    mesh_label = "bf.mesh" if show_avg else "bf.mesh (default)"

    annotations: dict[str, list[tuple[str, tuple[int, int], str, str, str]]] = {
        "sloped_guide_mode": [
            ("uniform-50", (5, -4), "left", "top", "50 nm"),
            ("uniform-20", (5, 5), "left", "bottom", "20 nm"),
            ("uniform-7.2", (0, -7), "center", "top", "7.2 nm"),
            ("mesh-8", (6, 2), "left", "bottom", "ppw 8"),
            ("mesh-22", (4, 7), "left", "bottom", "ppw 22"),
            ("mesh-86", (-5, -5), "right", "top", "ppw 86"),
        ],
        "sloped_guide_fdtd": [
            ("uniform-50", (-4, -5), "right", "top", "50 nm"),
            ("uniform-20", (5, 4), "left", "bottom", "20 nm"),
            ("uniform-7.2", (0, 6), "center", "bottom", "7.2 nm"),
            ("mesh-8", (6, 0), "left", "center", "ppw 8"),
            ("mesh-22", (-6, -7), "right", "top", "ppw 22"),
            ("mesh-86", (-5, -6), "right", "top", "ppw 86"),
        ],
        "sin_bend": [
            ("uniform-60", (-4, -5), "right", "top", "60 nm"),
            ("uniform-40", (5, 4), "left", "bottom", "40 nm"),
            ("uniform-16.2", (-4, 5), "right", "bottom", "16.2 nm"),
            ("mesh-12", (5, 2), "left", "bottom", "ppw 12"),
            ("meshsub-20", (-5, -5), "right", "top", "ppw 20 (mc)"),
            ("mesh-54", (-6, -6), "right", "top", "ppw 54"),
        ],
        "elevator_coupler": [
            ("uniform-50", (0, 6), "center", "bottom", "50 nm"),
            ("uniform-25", (4, 5), "left", "bottom", "25 nm"),
            ("uniform-12", (-4, 5), "right", "bottom", "12 nm"),
            ("mesh-6", (6, -1), "left", "center", "ppw 6"),
            ("mesh-10", (0, -6), "center", "top", "ppw 10"),
            ("mesh-33", (-4, -6), "right", "top", "ppw 33"),
        ],
        "directional_coupler": [
            ("uniform-50", (5, 2), "left", "bottom", "50 nm"),
            ("uniform-16.5", (0, 6), "center", "bottom", "16.5 nm"),
            ("mesh-6", (5, 2), "left", "bottom", "ppw 6"),
            ("mesh-10", (-5, -5), "right", "top", "ppw 10"),
            ("mesh-18", (-4, -6), "right", "top", "ppw 18"),
        ],
        "linbo3_bend": [
            ("uniform-60", (-4, -5), "right", "top", "60 nm"),
            ("uniform-30", (5, 4), "left", "bottom", "30 nm"),
            ("uniform-17", (0, 6), "center", "bottom", "17 nm"),
            ("mesh-10", (5, 2), "left", "bottom", "ppw 10"),
            ("mesh-26", (-5, -5), "right", "top", "ppw 26"),
            ("mesh-48", (0, -7), "center", "top", "ppw 48"),
        ],
    }

    cells_end_annotations: dict[str, list[tuple[str, tuple[int, int], str, str, str]]] = {
        "sloped_guide_mode": [
            ("mesh-8", (5, 3), "left", "bottom", "ppw 8"),
            ("mesh-86", (0, 8), "center", "bottom", "ppw 86"),
            ("uniform-64", (4, -5), "left", "top", "64 nm"),
            ("uniform-7.2", (0, -9), "center", "top", "7.2 nm"),
        ],
        "sloped_guide_fdtd": [
            ("mesh-8", (6, 0), "left", "center", "ppw 8"),
            ("mesh-86", (0, 7), "center", "bottom", "ppw 86"),
            ("uniform-64", (-3, -8), "center", "top", "64 nm"),
            ("uniform-7.2", (0, -9), "center", "top", "7.2 nm"),
        ],
        "sin_bend": [
            ("mesh-12", (5, 3), "left", "bottom", "ppw 12"),
            ("mesh-54", (0, 7), "center", "bottom", "ppw 54"),
            ("uniform-60", (0, -8), "center", "top", "60 nm"),
            ("uniform-16.2", (0, -9), "center", "top", "16.2 nm"),
        ],
        "elevator_coupler": [
            ("mesh-6", (6, -2), "left", "center", "ppw 6"),
            ("mesh-33", (4, -8), "left", "top", "ppw 33"),
            ("uniform-50", (0, -8), "center", "top", "50 nm"),
            ("uniform-12", (-6, -8), "right", "top", "12 nm"),
        ],
        "directional_coupler": [
            ("mesh-6", (5, 2), "left", "bottom", "ppw 6"),
            ("mesh-18", (0, -9), "center", "top", "ppw 18"),
            ("uniform-50", (5, 2), "left", "bottom", "50 nm"),
            ("uniform-16.5", (5, 2), "left", "bottom", "16.5 nm"),
        ],
        "linbo3_bend": [
            ("mesh-10", (5, 0), "left", "center", "ppw 10"),
            ("mesh-48", (0, -9), "center", "top", "ppw 48"),
            ("uniform-60", (0, -8), "center", "top", "60 nm"),
            ("uniform-17", (0, 7), "center", "bottom", "17 nm"),
        ],
    }

    vram_end_annotations: dict[str, list[tuple[str, tuple[int, int], str, str, str]]] = {
        "sloped_guide_mode": [
            ("mesh-8", (5, 3), "left", "bottom", "ppw 8"),
            ("mesh-86", (-3, 7), "center", "bottom", "ppw 86"),
            ("uniform-64", (4, -5), "left", "top", "64 nm"),
            ("uniform-7.2", (0, -9), "center", "top", "7.2 nm"),
        ],
        "sloped_guide_fdtd": [
            ("mesh-8", (5, 0), "left", "center", "ppw 8"),
            ("mesh-86", (-2, 7), "center", "bottom", "ppw 86"),
            ("uniform-64", (-2, -8), "center", "top", "64 nm"),
            ("uniform-7.2", (0, -9), "center", "top", "7.2 nm"),
        ],
        "sin_bend": [
            ("mesh-12", (0, 7), "center", "bottom", "ppw 12"),
            ("mesh-54", (0, 7), "center", "bottom", "ppw 54"),
            ("uniform-60", (-4, -6), "right", "top", "60 nm"),
            ("uniform-16.2", (0, -9), "center", "top", "16.2 nm"),
        ],
        "elevator_coupler": [
            ("mesh-6", (6, -1), "left", "center", "ppw 6"),
            ("mesh-33", (3, -8), "left", "top", "ppw 33"),
            ("uniform-50", (0, 7), "center", "bottom", "50 nm"),
            ("uniform-12", (-6, -8), "right", "top", "12 nm"),
        ],
        "directional_coupler": [
            ("mesh-6", (5, 2), "left", "bottom", "ppw 6"),
            ("mesh-18", (0, -9), "center", "top", "ppw 18"),
            ("uniform-50", (5, 2), "left", "bottom", "50 nm"),
            ("uniform-16.5", (6, 11), "left", "bottom", "16.5 nm"),
        ],
        "linbo3_bend": [
            ("mesh-10", (5, -2), "left", "center", "ppw 10"),
            ("mesh-48", (0, -9), "center", "top", "ppw 48"),
            ("uniform-60", (-4, -7), "center", "top", "60 nm"),
            ("uniform-17", (0, 7), "center", "bottom", "17 nm"),
        ],
    }

    for idx, (ax, title, bench, ylabel, fy) in enumerate(panels):
        _style_ax(ax, title, xlabel, ylabel)
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_yscale("linear")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
        ax.margins(x=x_margin, y=0.14)
        for avg in AVERAGINGS:
            rr = _by_bench(rows, bench, avg)
            keep = lambda r: fx(r) is not None and fy(r) is not None  # noqa: E731
            u = [r for k, r in rr.items() if k.startswith("uniform-") and _uniform_nm(k) <= coarse[bench] and keep(r)]
            m = [r for k, r in rr.items() if k.startswith("mesh-") and keep(r)]
            ms = [r for k, r in rr.items() if k.startswith("meshsub-") and keep(r)]
            kw: dict[str, Any] = {"yscale_floor": None, "show_avg": show_avg, "label_ends": False}
            _series(ax, u, fx, fy, color=UNIF, marker="o", label="UniformGrid", avg=avg, **kw)
            _series(ax, m, fx, fy, color=MESH, marker="s", label=mesh_label, avg=avg, **kw)
            _series(ax, ms, fx, fy, color=MESH_SUB, marker="^", label="bf.mesh + material_cells", avg=avg, **kw)
        m_u = [
            r for k, r in _meep_by_bench(meep_rows, bench).items()
            if k.startswith("uniform-") and _uniform_nm(k) <= coarse[bench] and mfx(r) is not None and fy(r) is not None
        ]
        _series(ax, m_u, mfx, fy, color=MEEP, marker="D", ls="-.", label=meep_label, avg=None, label_ends=False, yscale_floor=None, show_avg=show_avg, zorder=2.8)
        ann_key = ("sloped_guide_mode", "sloped_guide_fdtd")[idx] if bench == "sloped_guide" else bench
        rr_sub = _by_bench(rows, bench, "subpixel")
        spec_map = annotations if not show_avg else (cells_end_annotations if x_axis == "cells" else vram_end_annotations)
        pts = []
        for kind, off, ha, va, text in spec_map[ann_key]:
            if kind in rr_sub and fx(rr_sub[kind]) is not None and fy(rr_sub[kind]) is not None:
                r = rr_sub[kind]
                pts.append((fx(r), fy(r), text, off, ha, va))
        _annotate_points(ax, pts, yscale_floor=None, fontsize=7.5 if not show_avg else 7.2)
    _finish(fig, axes, dsts, show_avg=show_avg, meep_label=meep_label)


def make_raw_figure(src: Path, dsts: list[Path], *, meep_src: Path | None = None) -> None:
    """Actual simulated quantities (effective index, transmission, splitting ratio) against cell count."""
    _make_value_figure(src, dsts, x_axis="cells", meep_src=meep_src)


def make_vram_figure(src: Path, dsts: list[Path], *, meep_src: Path | None = None) -> None:
    """Actual simulated quantities (effective index, transmission, splitting ratio) against peak GPU VRAM."""
    _make_value_figure(src, dsts, x_axis="vram", meep_src=meep_src)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=root / "benchmarks/meshing/results/accuracy_per_cell.json")
    parser.add_argument("--meep-src", type=Path, default=root / "benchmarks/meshing/results/meep_uniform.json")
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
    parser.add_argument(
        "--vram-out",
        type=Path,
        nargs="+",
        default=[
            root / "docs/images/adaptive_meshing_vram.png",
            root / "benchmarks/meshing/results/adaptive_meshing_vram.png",
        ],
    )
    args = parser.parse_args()
    make_figure(args.src, args.out, meep_src=args.meep_src)
    make_raw_figure(args.src, args.raw_out, meep_src=args.meep_src)
    make_vram_figure(args.src, args.vram_out, meep_src=args.meep_src)


if __name__ == "__main__":
    main()
