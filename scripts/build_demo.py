"""Regenerates the precomputed viewer demo from the waveguide bend tutorial.

Runs examples/waveguide_bend.py's simulation, loads the result into the viewer and
exports it (see bifrost/viewer/export.py) to docs/demo/data/, with the field image
the tutorial shows. The documentation build only publishes these files: it never
reruns the simulation. Run this on a GPU node after the example, the viewer or the
export format changes, then commit the result.

    python scripts/build_demo.py
    python scripts/build_demo.py --resolution 0.1 --out /tmp/demo   # a quick CPU try-out
"""

import argparse
import datetime
import importlib.metadata as metadata
import json
import subprocess
import sys
from pathlib import Path

import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import waveguide_bend as bend  # noqa: E402

import bifrost as bf  # noqa: E402


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resolution", type=float, default=bend.RESOLUTION, help="grid cell in um")
    parser.add_argument("--duration", type=float, default=bend.DURATION, help="simulated time in s")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "demo" / "data")
    parser.add_argument("--figure", type=Path, default=ROOT / "docs" / "images" / "waveguide_bend_field.png")
    parser.add_argument("--section-step", type=float, default=0.1, help="spacing of the stored slices, um")
    args = parser.parse_args()

    result = bend.simulate(args.resolution, args.duration)
    print(f"transmission {result.transmission:.4f}; {result.seconds:.1f} s; peak {result.peak_gib} GiB")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    bend.plot_field(result, str(args.figure))

    viewer = bend.open_viewer(result)
    sx, sy, sz = (float(v) * 1e6 for v in result.stack.partial_real_shape)
    manifest = {
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "source": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "bifrost", "examples", "js"))},
        "packages": {name: metadata.version(name) for name in ("bifrost", "gdsfactory", "fdtdx", "jax", "jaxlib")},
        "device": str(jax.devices()[0]),
        "parameters": {
            "example": "examples/waveguide_bend.py",
            "wavelength_um": bend.WAVELENGTH,
            "core_index": bend.CORE_INDEX,
            "clad_index": bend.CLAD_INDEX,
            "width_um": bend.WIDTH,
            "thickness_um": bend.THICKNESS,
            "radius_um": bend.RADIUS,
            "resolution_um": args.resolution,
            "duration_s": args.duration,
            "pml_um": bend.PML,
            "box_um": [round(sx, 3), round(sy, 3), round(sz, 3)],
            "movie_frames": bend.MOVIE_FRAMES,
            "section_step_um": args.section_step,
        },
        "result": {
            "transmission": round(result.transmission, 4),
            "wall_seconds": round(result.seconds, 1),
            "peak_gib": None if result.peak_gib is None else round(result.peak_gib, 2),
        },
    }
    # every monitor opens with its field shown (|E|), the movie at its most energetic frame, except the
    # static phasor "field": it shares the movie's plane, so its box starts unchecked (one click shows it)
    data = bf.viewer.export.export_demo(
        viewer, args.out, section_step_um=args.section_step, unchecked=("field",), manifest=manifest
    )
    print(f"wrote {args.out}: {data['total_bytes'] / 1e6:.1f} MB in total")
    for spec_id, entry in data["fields"].items():
        print(f"  field {entry['name']}: {entry['frames']} frames, {sum(entry['bytes'].values()) / 1e6:.2f} MB")
    print(json.dumps(manifest["result"]))


if __name__ == "__main__":
    main()
