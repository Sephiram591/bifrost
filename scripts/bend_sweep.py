"""Measures what the grid costs and buys for examples/waveguide_bend.py, as a Markdown table.

Each resolution runs in its own process, so the peak memory is that run's alone. A last row-less
run executes the simulation twice in one process, to show how much of the first run is compilation.
Run it on the GPU node whose numbers you want to quote, and paste the table into docs/getting-started.md
and docs/examples/waveguide-bend.md:

    python scripts/bend_sweep.py
    python scripts/bend_sweep.py --resolutions 0.1 0.08     # a quick CPU try-out
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "waveguide_bend.py"


def run(*args: str) -> dict[str, str]:
    out = subprocess.run([sys.executable, str(EXAMPLE), *args], capture_output=True, text=True, check=True).stdout
    return {
        "T": re.search(r"transmission:\s+([\d.]+)", out).group(1),
        "seconds": re.search(r"wall time:\s+([\d.]+) s", out).group(1),
        "memory": (re.search(r"peak memory:\s+([\d.]+) GiB", out) or [None, "n/a"])[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resolutions", type=float, nargs="+", default=[0.05, 0.04, 0.025, 0.02])
    parser.add_argument("--long", type=float, default=0.025, help="the resolution of the 600 fs check")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        figure = str(Path(tmp) / "field.png")
        rows = [(r, run("--resolution", str(r), "--figure", figure)) for r in args.resolutions]
        longer = run("--resolution", str(args.long), "--duration", "600e-15", "--figure", figure)
        warm = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'examples'); import waveguide_bend as b; "
             "first = b.simulate().seconds; second = b.simulate().seconds; print(f'{first:.1f} {second:.1f}')"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.split()[-2:]

    print("| Grid cell | Transmission | Wall time | Peak GPU memory |")
    print("|---|---|---|---|")
    for resolution, row in rows:
        print(f"| {resolution * 1000:g} nm | {row['T']} | {float(row['seconds']):.0f} s | {row['memory']} GiB |")
    print()
    print(f"600 fs at {args.long * 1000:g} nm: transmission {longer['T']}, {float(longer['seconds']):.0f} s, {longer['memory']} GiB.")
    print(f"Default settings twice in one process: {warm[0]} s, then {warm[1]} s.")


if __name__ == "__main__":
    main()
