"""Captures docs/images/viewer-demo.png: the picture shown behind the live demo and when it cannot load.

It opens docs/demo in headless Chromium, waits for the result's fields, and presses the viewer's own
camera (save image) button, which writes the 3D viewport to a PNG. Nothing else is captured. Run it
after build_demo.py:

    python scripts/capture_demo_poster.py
    python scripts/capture_demo_poster.py --root /path/to/docs --out poster.png

Needs Playwright with Chromium (`pip install playwright && playwright install chromium`).
"""

import argparse
import functools
import http.server
import socketserver
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
CHROMIUM_ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT / "docs", help="the docs folder holding demo/")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "images" / "viewer-demo.png")
    parser.add_argument("--size", type=int, nargs=2, default=(1200, 760), metavar=("W", "H"))
    args = parser.parse_args()

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(args.root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=CHROMIUM_ARGS)
            page = browser.new_page(viewport={"width": args.size[0], "height": args.size[1]}, color_scheme="dark")
            page.goto(f"http://127.0.0.1:{server.server_address[1]}/demo/index.html")
            page.wait_for_selector(".gfx-shell .gfx-menubar", timeout=120000)
            page.wait_for_function(
                "document.querySelector('#status')?.hidden === true && document.querySelector('.gfx-colorbar')?.hidden === false",
                timeout=120000,
            )
            page.wait_for_timeout(1500)  # let the last frame draw
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with page.expect_download() as download:
                page.locator(".gfx-save").click()  # the viewer's own save-image button
            download.value.save_as(args.out)
            browser.close()
    finally:
        server.shutdown()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
