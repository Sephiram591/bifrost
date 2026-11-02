"""Compares the fdtdx and MEEP directional-coupler runs: power spectra, 50:50 point, field maps, speed.

    python examples/compare_directional_coupler.py [resolution_nm]   # reads examples/output/*_dc_<nm>nm*.npz

The fdtdx runs (``python examples/fdtdx_directional_coupler.py <resolution_um> [averaging]``) are compared with the MEEP
run; an fdtdx run that is missing is left out.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

OUT = Path(__file__).parent / "output"
RUNS = {  # label: file stem
    "fdtdx subpixel": "fdtdx_dc_{nm}nm",
    "fdtdx subpixel_full": "fdtdx_dc_{nm}nm_subpixel_full",
    "MEEP": "meep_dc_{nm}nm",
}
REFERENCE = "MEEP"
LINES = {"fdtdx subpixel": "-", "fdtdx subpixel_full": "-.", "MEEP": "--"}
SHORT = {"fdtdx subpixel": "subpixel", "fdtdx subpixel_full": "full", "MEEP": "MEEP"}  # table column names
EPS_MID = (1.95**2 + 1.457**2) / 2  # the permittivity contour that outlines the guides


def load(stem: str) -> dict:
    d = dict(np.load(OUT / f"{stem}.npz"))
    d["E2"], d["eps"] = np.squeeze(d["E2"]), np.squeeze(d["eps"])
    # normalize to the input mode: peak |E|^2 in the straight lead of the input guide (o1: x in 0..0.5, y < 0)
    x, y = d["x"], d["y"]
    d["E2"] = d["E2"] / d["E2"][np.ix_((x > 0.1) & (x < 0.4), y < 0)].max()
    return d


def on_grid(d: dict, x: np.ndarray, y: np.ndarray, key: str = "E2") -> np.ndarray:
    xs, ys = (d["eps_x"], d["eps_y"]) if key == "eps" and "eps_x" in d else (d["x"], d["y"])
    f = RegularGridInterpolator((np.asarray(xs), np.asarray(ys)), d[key], bounds_error=False, fill_value=None)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.clip(f((X, Y)), 0, None) if key == "E2" else f((X, Y))


def split_wavelength(d: dict) -> float:
    """Wavelength (nm) where the cross port carries half of the input power (linear interpolation)."""
    order = np.argsort(d["cross"])
    return float(np.interp(0.5, d["cross"][order], d["wavelength"][order] * 1e3))


def main(nm: float = 20) -> None:
    stems = {label: stem.format(nm=f"{nm:g}") for label, stem in RUNS.items()}
    runs = {label: load(stem) for label, stem in stems.items() if (OUT / f"{stem}.npz").exists()}
    ref = runs[REFERENCE]
    others = [label for label in runs if label != REFERENCE]
    wl = ref["wavelength"] * 1e3

    fig, (ax, ax_d) = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    for key, color, port in (("through", "C0", "through (o4)"), ("cross", "C1", "cross (o3)")):
        for label, d in runs.items():
            ax.plot(wl, d[key], LINES[label], color=color, label=f"{port}, {label}")
            if label != REFERENCE:
                ax_d.plot(wl, d[key] - ref[key], LINES[label], color=color, label=f"{port}, {label}")
    ax.set(ylabel="TE0 power transmission from o1", ylim=(0, 1.05))
    ax.set_title(f"SiN directional coupler, 300 nm gap, dx = {nm:g} nm", fontsize=10)
    ax.legend(fontsize=7.5, ncol=2, loc="lower center")
    ax_d.axhline(0, color="0.7", lw=0.5)
    ax_d.set(xlabel="wavelength (nm)", ylabel=f"fdtdx - {REFERENCE}")
    ax_d.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "directional_coupler_spectra.png", dpi=130)

    first = runs[others[0]] if others else ref
    x, y = first["x"], first["y"]
    interior = (x > 0.0) & (x < x.max() - 1.0)  # after the source, before the PML
    maps = {label: on_grid(d, x, y) for label, d in runs.items()}
    err = {
        label: np.linalg.norm((maps[label] - maps[REFERENCE])[interior]) / np.linalg.norm(maps[REFERENCE][interior])
        for label in others
    }
    extent = (x.min(), x.max(), y.min(), y.max())
    style = dict(extent=extent, origin="lower", aspect="auto")
    fig, axes = plt.subplots(len(runs) + 1, 1, figsize=(14, 2.3 * (len(runs) + 1)), sharex=True)
    axes[0].imshow(first["eps"].T, cmap="Greys", **style)
    axes[0].contour(x, y, on_grid(ref, x, y, "eps").T, levels=[EPS_MID], colors="r", linewidths=0.6)
    shown = others[0] if others else REFERENCE
    axes[0].set_title(f"eps ({shown}, grey) / {REFERENCE} contour at eps = {EPS_MID:.2f} (red)", fontsize=9)
    for ax, (label, d) in zip(axes[1:], maps.items()):
        ax.imshow(np.sqrt(d).T, cmap="inferno", vmin=0, vmax=1.2, **style)
        ax.set_title(f"{label}: |E| (z = 0) at 619 nm", fontsize=9)
    for ax in axes:
        ax.set(ylabel="y (um)")
    axes[-1].set(xlabel="x (um)")
    fig.tight_layout()
    fig.savefig(OUT / "directional_coupler_fields.png", dpi=130)

    names = [SHORT[label] for label in runs]
    width = 9 * len(runs) - 1
    print(f"{'wavelength':>10} {'through power':^{width}}  {'cross power':^{width}}")
    print(f"{'(nm)':>10} " + " ".join(f"{n:>8}" for n in names) + "  " + " ".join(f"{n:>8}" for n in names))
    for i in range(0, len(wl), 3):
        through = " ".join(f"{d['through'][i]:8.4f}" for d in runs.values())
        cross = " ".join(f"{d['cross'][i]:8.4f}" for d in runs.values())
        print(f"{wl[i]:10.1f} {through}  {cross}")
    print("columns: " + ", ".join(f"{SHORT[label]} = {label}" for label in runs))
    splits = ", ".join(f"{label} {split_wavelength(d):.1f} nm" for label, d in runs.items())
    print(f"\n50:50 point (cross = 0.5): {splits}")
    for label in others:
        d = runs[label]
        d_t, d_c = d["through"] - ref["through"], d["cross"] - ref["cross"]
        print(
            f"{label} - {REFERENCE}: 50:50 point {split_wavelength(d) - split_wavelength(ref):+.1f} nm, "
            f"max |diff| through {abs(d_t).max():.4f} cross {abs(d_c).max():.4f}, "
            f"rms through {np.sqrt((d_t**2).mean()):.4f} cross {np.sqrt((d_c**2).mean()):.4f}, "
            f"|E|^2 relative difference at 619 nm {err[label]:.3f}"
        )
    for label, d in runs.items():
        peak = f", peak GPU memory {float(d['peak_gb']):.0f} GiB" if "peak_gb" in d else ""
        print(f"{label}: {float(d['cells']) / 1e6:.1f}M cells, {float(d['runtime']):.0f} s{peak}")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 20)
