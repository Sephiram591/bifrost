"""Plot the directional coupler's splitting ratio against the total cell count (results of dc_convergence.py).

    python benchmarks/rectilinear/plot_dc_convergence.py results/dc_convergence.jsonl results/dc_convergence.png

Top: the splitting ratio at 619 nm on uniform grids (the core's faces on cell edges, as the box puts them), on uniform
grids shifted half a cell in z (the faces mid-cell) and on bf.mesh's. Bottom: its distance from the converged value,
from one fit S0 + C h^p to every uniform grid of 25 nm and finer, S0 and p shared by both placements and C their own
(the faces' place in their cells sets C, not the limit).
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# the reference palette's first four categorical slots (validated: CVD dE >= 9.1, normal dE >= 22.9; slots 3 and 4
# are under 3:1 contrast on the surface, so every series has its own marker and its points are labelled)
EDGE, MESH, MID, MESH_OFF = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a63", "#e6e5e0"
MID_SHIFT = "@3.2,"  # the uniform grids shifted half a cell in z: <grid>@3.2,<half height>


def joint_fit(groups: list[list[tuple[float, float]]]) -> tuple[float, float, list[float], float]:
    """(S0, p, C per group, rms) of the least-squares fit s = S0 + C_g h^p, S0 and p shared, p over [1, 4]."""
    best = None
    rows = [(g, h, s) for g, points in enumerate(groups) for h, s in points]
    b = np.array([s for _, _, s in rows])
    for p in np.linspace(1.0, 4.0, 3001):
        a = np.zeros((len(rows), 1 + len(groups)))
        a[:, 0] = 1.0
        for i, (g, h, _) in enumerate(rows):
            a[i, 1 + g] = h**p
        coef, *_ = np.linalg.lstsq(a, b, rcond=None)
        res = float(np.sum((a @ coef - b) ** 2))
        if best is None or res < best[0]:
            best = (res, float(p), coef)
    res, p, coef = best
    return float(coef[0]), p, [float(c) for c in coef[1:]], float(np.sqrt(res / len(rows)))


def main(src: str, dst: str) -> None:
    with open(src) as f:
        rows = {(r := json.loads(line))["grid"]: r for line in f}  # the last run of a grid wins

    def series(keep) -> list[dict]:
        return sorted((r for r in rows.values() if keep(r["grid"])), key=lambda r: r["total_cells"])

    edge = series(lambda g: g.startswith("uniform-") and "@" not in g)
    mid = series(lambda g: g.startswith("uniform-") and MID_SHIFT in g)
    mesh = series(lambda g: g.startswith("mesh-"))  # faces centered (center_faces, the default)
    mesh_off = series(lambda g: g.startswith("meshoff-"))  # center_faces=False
    def cell(r: dict) -> float:
        """A uniform grid's cell (nm), from its name."""
        return float(r["grid"].split("-")[1].split("@")[0])

    groups = [[(cell(r), r["split"]) for r in rs if cell(r) <= 25] for rs in (edge, mid)]  # the asymptotic range
    s0, p, cs, rms = joint_fit(groups)
    print(f"converged: S0 = {s0:.5f}, p = {p:.2f}, C (faces on edges, mid-cell) = {cs[0]:.3e}, {cs[1]:.3e}, "
          f"rms {rms:.1e}")
    for r in edge + mid + mesh + mesh_off:
        print(f"{r['grid']:>22} {r['total_cells'] / 1e6:7.1f}M  split {r['split']:.5f}  |S - S0| "
              f"{abs(r['split'] - s0):.2e}  run {r['run_s']:.0f} s")

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.4, 7.8), sharex=True, facecolor=SURFACE,
                                      gridspec_kw={"height_ratios": [1.2, 1]})
    for ax in (top, bottom):
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    bottom.set_xscale("log")
    bottom.set_yscale("log")
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    for axis in (bottom.xaxis, bottom.yaxis):
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    bottom.set_xticks([5, 10, 20, 50, 100, 200, 400])
    bottom.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    groups = (
        ("uniform, core faces on cell edges", edge, EDGE, "o", lambda r: f"{cell(r):g} nm", (6, 6), "left", "bottom"),
        ("uniform, core faces mid-cell", mid, MID, "^", lambda r: f"{cell(r):g} nm", (-6, 4), "right", "bottom"),
        ("bf.mesh, faces centered (default)", mesh, MESH, "s", lambda r: f"ppw {r['grid'].split('-')[1]}", (-6, -6),
         "right", "top"),
        ("bf.mesh, center_faces=False", mesh_off, MESH_OFF, "D", lambda r: f"ppw {r['grid'].split('-')[1]}", (6, 6),
         "left", "bottom"),
    )
    for name, rs, color, marker, label, offset, ha, va in groups:
        if not rs:
            continue
        cells = np.array([r["total_cells"] for r in rs]) / 1e6
        split = np.array([r["split"] for r in rs])
        style = {"color": color, "linewidth": 2, "marker": marker, "markersize": 7, "markeredgecolor": SURFACE,
                 "markeredgewidth": 1.5, "label": name, "zorder": 3}
        top.plot(cells, split, **style)
        bottom.plot(cells, np.abs(split - s0), **style)
        # label the uniform grids on top and bf.mesh's (default) at the bottom, where the series stand apart; the
        # mid-cell and center_faces=False points share those grids' cell counts, so the legend names them
        if rs is mid or rs is mesh_off:
            continue
        ax, ys = (top, split) if rs is edge else (bottom, np.abs(split - s0))
        for r, x, y in zip(rs, cells, ys, strict=True):
            ax.annotate(label(r), (x, y), textcoords="offset points", xytext=offset, ha=ha, va=va, fontsize=8,
                        color=MUTED)
    top.axhline(s0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    top.annotate(f"converged: {s0:.4f} (fit of the uniform grids, order {p:.2f})", (0.01, s0),
                 xycoords=("axes fraction", "data"), xytext=(0, 4), textcoords="offset points", fontsize=8,
                 color=MUTED)
    top.set_ylabel("splitting ratio at 619 nm\nP_cross / (P_cross + P_through)")
    top.set_title("Directional coupler: convergence with the cell count", loc="left", fontsize=11, color=INK)
    top.legend(frameon=False, loc="upper right", fontsize=9)
    bottom.set_xlabel("total cells (millions)")
    bottom.set_ylabel("|splitting ratio - converged|")
    fig.tight_layout()
    fig.savefig(dst, dpi=150, facecolor=SURFACE)
    print("wrote", dst)


if __name__ == "__main__":
    main(*sys.argv[1:3])
