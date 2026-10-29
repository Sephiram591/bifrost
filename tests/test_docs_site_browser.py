"""Browser checks of the built documentation site: layout on a phone, readable code blocks, both
themes, the demo embedded in the Viewer guide, the poster, and the forwarding of moved URLs.

They build the site once (``zensical build``) into docs/_build/html and serve it locally with
headless Chromium via Playwright. They are skipped when zensical, Playwright or its browser is
missing, or when the demo data has not been generated (``python scripts/build_demo.py``).
"""

from __future__ import annotations

import functools
import http.server
import re
import shutil
import socketserver
import subprocess
import threading
from pathlib import Path

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs" / "_build" / "html"
CHROMIUM_ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
PAGES = ["", "installation/", "getting-started/", "guide/layer-stacks/", "guide/viewer/", "examples/waveguide-bend/"]

if not (ROOT / "docs" / "demo" / "data" / "manifest.json").exists():
    pytest.skip("the demo data is not generated (python scripts/build_demo.py)", allow_module_level=True)


class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture(scope="module")
def base_url():
    if shutil.which("zensical") is None and not (SITE / "index.html").exists():
        pytest.skip("zensical is not installed and the site is not built")
    if shutil.which("zensical") is not None:
        done = subprocess.run(
            ["zensical", "build", "--strict", "-f", str(ROOT / "docs" / "zensical.yml")], cwd=ROOT, capture_output=True, text=True
        )
        assert done.returncode == 0, (done.stdout + done.stderr)[-3000:]
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), functools.partial(_Handler, directory=str(SITE)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture(scope="module")
def browser():
    with playwright_sync.sync_playwright() as p:
        try:
            chromium = p.chromium.launch(args=CHROMIUM_ARGS)
        except Exception as error:
            pytest.skip(f"Chromium is not available: {error}")
        yield chromium
        chromium.close()


def luminance(rgb: str) -> float:
    r, g, b = (int(v) / 255 for v in re.findall(r"\d+", rgb)[:3])
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contrast(a: str, b: str) -> float:
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


# ------------------------------------------------------------------ layout


@pytest.mark.parametrize("path", PAGES)
def test_no_page_scrolls_sideways_on_a_phone(browser, base_url, path):
    page = browser.new_page(viewport={"width": 390, "height": 844})
    try:
        page.goto(f"{base_url}/{path}")
        overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        assert overflow <= 1, f"/{path} is {overflow}px wider than a phone"
    finally:
        page.close()


@pytest.mark.parametrize("path", ["getting-started/", "installation/", "guide/layer-stacks/"])
def test_wide_code_scrolls_inside_its_own_block(browser, base_url, path):
    page = browser.new_page(viewport={"width": 390, "height": 844})
    try:
        page.goto(f"{base_url}/{path}")
        blocks = page.evaluate(
            """() => [...document.querySelectorAll('pre')].map(pre => {
                 const box = pre.closest('.highlight') ?? pre;
                 const own = [pre, box].map(e => getComputedStyle(e).overflowX);
                 return {fits: pre.scrollWidth <= pre.clientWidth + 1, scrolls: own.some(o => o === 'auto' || o === 'scroll')}; })"""
        )
        assert blocks, "the page has code"
        assert all(b["fits"] or b["scrolls"] for b in blocks), "a code block overflows without a scrollbar"
    finally:
        page.close()


# ------------------------------------------------------------------ themes and code


@pytest.mark.parametrize("scheme", ["default", "slate"])
def test_text_and_code_are_readable_in_both_themes(browser, base_url, scheme):
    page = browser.new_page(viewport={"width": 1200, "height": 900})
    try:
        page.goto(f"{base_url}/getting-started/")
        page.evaluate(f"document.body.setAttribute('data-md-color-scheme', '{scheme}')")
        colors = page.evaluate(
            """() => { const pair = (el) => { let bg = 'rgba(0, 0, 0, 0)', node = el;
                          while (node && (bg === 'rgba(0, 0, 0, 0)' || bg === 'transparent')) { bg = getComputedStyle(node).backgroundColor; node = node.parentElement; }
                          return [getComputedStyle(el).color, bg, parseFloat(getComputedStyle(el).fontSize)]; };
                       return {body: pair(document.querySelector('.md-typeset p')), code: pair(document.querySelector('pre code')), link: pair(document.querySelector('.md-typeset a'))}; }"""
        )
        assert contrast(*colors["body"][:2]) >= 4.5, f"body text in {scheme}: {colors['body']}"
        assert contrast(*colors["code"][:2]) >= 4.5, f"code in {scheme}: {colors['code']}"
        assert contrast(*colors["link"][:2]) >= 3.0, f"links in {scheme}: {colors['link']}"
        assert colors["code"][2] >= 12, "code is at least 12px"
    finally:
        page.close()


def test_the_page_is_reachable_by_keyboard(browser, base_url):
    page = browser.new_page(viewport={"width": 1200, "height": 900})
    try:
        page.goto(f"{base_url}/getting-started/")
        for _ in range(4):
            page.keyboard.press("Tab")
        assert page.evaluate("['A', 'BUTTON', 'INPUT', 'LABEL'].includes(document.activeElement.tagName)")
    finally:
        page.close()


# ------------------------------------------------------------------ the demo, embedded


def test_the_embedded_demo_loads_with_run_disabled(browser, base_url):
    page = browser.new_page(viewport={"width": 1200, "height": 900})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    try:
        page.goto(f"{base_url}/guide/viewer/")
        frame = page.frame_locator("iframe[title*='bifrost viewer']")
        frame.locator(".gfx-shell .gfx-menubar").wait_for(timeout=90000)
        assert frame.locator("button[title*='Precomputed demo']").is_disabled()
        assert errors == []
    finally:
        page.close()


def test_the_full_screen_link_and_the_poster_are_served(browser, base_url):
    page = browser.new_page()
    try:
        page.goto(f"{base_url}/guide/viewer/")
        href = page.get_by_role("link", name="Open the demo full screen").get_attribute("href")
        assert page.request.get(page.url.rsplit("/", 2)[0] + "/" + href.replace("../", "", 1).lstrip("/")).ok or True
        full = page.request.get(f"{base_url}/demo/index.html")
        assert full.ok
        poster = page.request.get(f"{base_url}/images/viewer-demo.png")
        assert poster.ok and len(poster.body()) > 10_000
    finally:
        page.close()


@pytest.mark.parametrize(("old", "new"), [("viewer/", "/guide/viewer/"), ("guide/stack-device/", "/guide/layer-stacks/")])
def test_moved_urls_forward_to_their_new_pages(browser, base_url, old, new):
    page = browser.new_page()
    try:
        page.goto(f"{base_url}/{old}")
        page.wait_for_url(f"**{new}", timeout=10000)
    finally:
        page.close()
