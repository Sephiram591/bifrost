"""Measures what the grid costs and buys for examples/waveguide_bend.py, as a Markdown table.

Each ppw runs in its own process, so the peak memory is that run's alone. A last row-less
run executes the simulation twice in one process, to show how much of the first run is compilation.
Run it on the GPU node whose numbers you want to quote, and paste the table into docs/getting-started.md
and docs/examples/waveguide-bend.md:

    python scripts/bend_sweep.py
    python scripts/bend_sweep.py --ppws 8 10     # a quick CPU try-out
"""

import argparse
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "waveguide_bend.py"
sys.path.insert(0, str(ROOT / "examples"))

import waveguide_bend as bend  # noqa: E402

import bifrost as bf  # noqa: E402


def run(*args: str) -> dict[str, str]:
    out = subprocess.run([sys.executable, str(EXAMPLE), *args], capture_output=True, text=True, check=True).stdout
    return {
        "T": re.search(r"transmission:\s+([\d.]+)", out).group(1),
        "seconds": re.search(r"wall time:\s+([\d.]+) s", out).group(1),
        "memory": (re.search(r"peak memory:\s+([\d.]+) GiB", out) or [None, "n/a"])[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ppws", type=float, nargs="+", default=[12, 16, 20, 24])
    parser.add_argument("--long", type=float, default=24, help="the ppw of the 600 fs check")
    args = parser.parse_args()

    stack = bend.build_stack()
    with tempfile.TemporaryDirectory() as tmp:
        figure = str(Path(tmp) / "field.png")
        rows = [
            (p, math.prod(bf.mesh(stack, bend.WAVELENGTH, ppw=p).shape), run("--ppw", str(p), "--figure", figure))
            for p in args.ppws
        ]
        longer = run("--ppw", str(args.long), "--duration", "600e-15", "--figure", figure)
        warm = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'examples'); import waveguide_bend as b; "
             "first = b.simulate().seconds; second = b.simulate().seconds; print(f'{first:.1f} {second:.1f}')"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.split()[-2:]

    print("| Points per wavelength (ppw) | Grid cells | Transmission | Wall time | Peak GPU memory |")
    print("|---|---|---|---|---|")
    for ppw, cells, row in rows:
        label = f"{ppw:g} (the script's default)" if ppw == bend.PPW else f"{ppw:g}"
        print(f"| {label} | {cells / 1e6:.2f}M | {row['T']} | {float(row['seconds']):.0f} s | {row['memory']} GiB |")
    print()
    print(f"600 fs at ppw={args.long:g}: transmission {longer['T']}, {float(longer['seconds']):.0f} s, {longer['memory']} GiB.")
    print(f"Default settings twice in one process: {warm[0]} s, then {warm[1]} s.")


if __name__ == "__main__":
    main()
