"""Browser checks of the precomputed viewer demo (docs/demo): loading, read-only controls,
slice selection, field playback, layout, keyboard use, themes and failure states.

They drive headless Chromium through Playwright against a local server over the docs
folder, so they need the built bundle (``make viewer-js``) and the exported data
(``python scripts/build_demo.py``) under docs/demo; set ``BIFROST_DEMO_ROOT`` to test a
different docs folder. They are skipped when Playwright or its browser is missing.
"""

from __future__ import annotations

import functools
import http.server
import json
import os
import re
import socketserver
import threading
from pathlib import Path

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")

ROOT = Path(os.environ.get("BIFROST_DEMO_ROOT", Path(__file__).resolve().parents[1] / "docs"))
DEMO = ROOT / "demo"

if not (DEMO / "viewer-demo.js").exists() or not (DEMO / "data" / "manifest.json").exists():
    pytest.skip("the demo bundle or its data is not built (make viewer-js; python scripts/build_demo.py)", allow_module_level=True)

MANIFEST = json.loads((DEMO / "data" / "manifest.json").read_text())
CHROMIUM_ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # keep the test output clean
        pass


@pytest.fixture(scope="module")
def base_url():
    handler = functools.partial(_Handler, directory=str(ROOT))
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture(scope="module")
def browser():
    with playwright_sync.sync_playwright() as p:
        try:
            chromium = p.chromium.launch(args=CHROMIUM_ARGS)
        except Exception as error:  # no browser installed
            pytest.skip(f"Chromium is not available: {error}")
        yield chromium
        chromium.close()


class Demo:
    """One page of the demo with its console messages and requests recorded."""

    def __init__(self, browser, base_url: str, *, size=(1200, 800), scheme="dark", route=None) -> None:
        self.context = browser.new_context(viewport={"width": size[0], "height": size[1]}, color_scheme=scheme)
        self.page = self.context.new_page()
        self.errors: list[str] = []
        self.requests: list[str] = []
        self.page.on("console", lambda m: self.errors.append(m.text) if m.type in ("error", "warning") else None)
        self.page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        self.page.on("request", lambda r: self.requests.append(r.url.replace(base_url, "")))
        if route:
            route(self.page)
        self.page.goto(f"{base_url}/demo/index.html")

    def ready(self) -> "Demo":
        self.page.wait_for_selector(".gfx-shell .gfx-menubar", timeout=60000)
        self.page.wait_for_function("document.querySelector('#status')?.hidden === true", timeout=60000)
        return self

    def close(self) -> None:
        self.context.close()

    def tab(self, name: str) -> None:
        self.page.get_by_role("button", name=name, exact=True).click()

    @property
    def stops(self) -> list[float]:
        return [round(v, 6) for v in self.page.evaluate(
            """() => { const s = document.querySelector('.gfx-position input[type=range]') ?? [...document.querySelectorAll('input[type=range]')].find(i => i.step === '0.1');
                       const lo = Number(s.min), hi = Number(s.max), st = Number(s.step);
                       return Array.from({length: Math.round((hi - lo) / st) + 1}, (_, k) => lo + k * st); }"""
        )]


    def listed(self) -> list[tuple[str, str]]:
        """(kind, name) of every source and monitor the open tab lists, in order."""
        return [
            tuple(item)
            for item in self.page.evaluate(
                """() => [...document.querySelectorAll('.gfx-tabpanel:not([hidden]) [data-kind=source], .gfx-tabpanel:not([hidden]) [data-kind=detector]')]
                         .map(e => [e.dataset.kind, (e.querySelector('.gfx-spec-header') ?? e).innerText.replace(/[▼▶]/g, '').trim()])"""
            )
        ]


@pytest.fixture
def demo(browser, base_url):
    d = Demo(browser, base_url).ready()
    yield d
    d.close()


def luminance(rgb: str) -> float:
    r, g, b = (int(v) / 255 for v in re.findall(r"\d+", rgb)[:3])
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


# --------------------------------------------------------------------- loading


def test_the_demo_loads_without_console_errors(demo):
    assert demo.page.locator("canvas").first.is_visible()
    assert demo.page.locator("#status").is_hidden()
    assert demo.errors == []


def test_it_starts_in_the_result_view_with_the_fields_loaded(demo):
    assert "gfx-active" in demo.page.get_by_role("button", name="Result", exact=True).get_attribute("class")
    demo.page.wait_for_function(
        "document.querySelector('#status')?.hidden === true && document.body.innerText.includes('movie')", timeout=60000
    )
    names = {entry["name"] for entry in MANIFEST["fields"].values()}
    assert {"field", "movie"} <= names
    for entry in MANIFEST["fields"].values():
        assert f"/demo/data/fields/{entry['id']}.real.bin" in demo.requests


def test_every_result_card_but_the_phasor_field_starts_checked_and_filled_in(demo):
    """The cards are built when the fields arrive, after the state: they must still show it.

    The static phasor "field" shares the movie's plane, so the demo opens with its box unchecked.
    """
    names = {entry["name"] for entry in MANIFEST["fields"].values()}
    demo.page.wait_for_function(f"document.querySelectorAll('.gfx-field-group').length === {len(names)}", timeout=60000)
    cards = demo.page.evaluate(
        """() => [...document.querySelectorAll('.gfx-field-group')].map(g => ({
               name: g.querySelector('.gfx-spec-header').innerText.replace(/[▼▶]/g, '').trim(),
               checked: g.querySelector('.gfx-spec-header input[type=checkbox]').checked,
               component: g.querySelector('select').value }))"""
    )
    assert {c["name"] for c in cards} == names
    assert {c["name"]: c["checked"] for c in cards} == {name: name != "field" for name in names}, cards
    assert all(c["component"] == "|E|" for c in cards), cards  # filled in, not the blank default


def test_every_monitors_field_is_shown_when_it_opens(demo):
    """Frequency-domain monitors (the mode monitors and the phasor plane) open shaded, like the movie."""
    state = json.loads((DEMO / "data" / "result.json").read_text())["state"]
    assert state["field_settings"] and all(s["field_visible"] for s in state["field_settings"].values())
    demo.page.wait_for_function("document.querySelector('.gfx-colorbar') && !document.querySelector('.gfx-colorbar').hidden", timeout=60000)
    title = demo.page.locator(".gfx-colorbar").inner_text().split("·")[0].strip()
    assert title == "movie", f"the colorbar describes {title!r}: it should be the largest plane that is shown"


def test_the_phasor_field_opens_hidden_in_both_worlds():
    """Its box starts unchecked in the editor and the Result tab: plane, field and colorbar all off."""
    for world in ("editor", "result"):
        state = json.loads((DEMO / "data" / f"{world}.json").read_text())["state"]
        shown = {g["name"]: g["visible"] for g in state["glyphs"]}
        assert shown["field"] is False, world
        assert all(visible for name, visible in shown.items() if name != "field"), (world, shown)
        detectors = {d["params"]["name"]: d["params"]["visible"] for d in state["detectors"]}
        assert detectors["field"] is False and detectors["movie"] is True, (world, detectors)


def test_the_editor_lists_the_same_sources_and_monitors_as_the_result(demo):
    demo.page.wait_for_function("document.querySelectorAll('.gfx-field-group').length > 0", timeout=60000)
    in_result = demo.listed()
    assert {kind for kind, _ in in_result} == {"source", "detector"}
    demo.tab("Sources / Detectors")
    in_editor = demo.listed()
    assert sorted(in_editor) == sorted(in_result), (in_editor, in_result)


def test_setup_geometry_and_slices_load_lazily(demo):
    assert "/demo/data/editor.meshes.bin" not in demo.requests  # the Layout tab was not opened
    assert "/demo/data/sections/xz.json" not in demo.requests  # nor the xz plane
    assert "/demo/data/sections/xy.json" in demo.requests
    demo.tab("Layout / Ports")
    demo.page.wait_for_timeout(500)
    assert "/demo/data/editor.meshes.bin" in demo.requests


# --------------------------------------------------------------------- read-only


def test_run_is_disabled_and_says_why(demo):
    run = demo.page.locator("button[title*='Precomputed demo']")
    assert run.is_disabled()
    assert "running needs a Python session" in run.get_attribute("title")
    assert run.get_attribute("aria-label") == "Precomputed demo"
    assert demo.page.locator(".gfx-run-phase").inner_text() == "Precomputed demo"


def test_setup_edits_are_disabled(demo):
    demo.tab("Sources / Detectors")
    assert all(b.is_disabled() for b in demo.page.locator(".gfx-tabpanel:not([hidden]) .gfx-section-add").all())
    demo.tab("Simulation")
    fields = demo.page.locator(".gfx-tabpanel:not([hidden]) input:not([type=checkbox])")
    assert fields.count() > 0
    assert all(f.is_disabled() for f in fields.all())
    assert "Precomputed demo" in demo.page.locator(".gfx-tabpanel:not([hidden])").inner_text()


def test_saving_a_video_is_refused_with_a_message(demo):
    demo.page.wait_for_function("document.body.innerText.includes('movie')", timeout=60000)
    record = demo.page.locator("button[aria-label='record all time-domain detectors']")
    record.click()
    demo.page.wait_for_function("document.body.innerText.includes('needs a Python session')", timeout=5000)


# --------------------------------------------------------------------- slices


def test_the_slice_slider_snaps_to_the_stored_positions(demo):
    demo.page.get_by_role("button", name="2D", exact=True).click()
    slider = demo.page.locator("input[type=range][step='0.1']").first
    stops = demo.stops
    assert len(stops) == MANIFEST["sections"]["xy"]["count"]
    slider.evaluate("(el) => { el.value = '0.2345'; el.dispatchEvent(new Event('input', {bubbles: true})); }")
    demo.page.wait_for_timeout(500)
    shown = float(slider.input_value())
    assert round(shown, 6) in stops
    box = demo.page.locator("input[type=number][step='0.1']").first
    assert float(box.input_value()) == pytest.approx(shown)


def test_changing_the_plane_loads_its_slices_and_stops(demo):
    demo.page.get_by_role("button", name="2D", exact=True).click()
    assert "/demo/data/sections/xz.json" not in demo.requests
    demo.page.get_by_role("button", name="xz", exact=True).click()
    demo.page.wait_for_function(
        f"[...document.querySelectorAll('input[type=range]')].some(i => Number(i.max) === {MANIFEST['sections']['xz']['stops'][1]})",
        timeout=15000,
    )
    assert "/demo/data/sections/xz.json" in demo.requests
    assert len(demo.stops) == MANIFEST["sections"]["xz"]["count"]


# --------------------------------------------------------------------- playback


def test_recorded_playback_advances_the_frames(demo):
    demo.page.wait_for_function("document.body.innerText.includes('movie')", timeout=60000)
    play = demo.page.locator("button.gfx-section-add", has_text="▶").first
    before = demo.page.evaluate("[...document.querySelectorAll('input[type=range]')].map(i => i.value).join(',')")
    play.click()
    demo.page.wait_for_function("[...document.querySelectorAll('button')].some(b => b.textContent === '❚❚')", timeout=5000)
    demo.page.wait_for_timeout(1500)
    after = demo.page.evaluate("[...document.querySelectorAll('input[type=range]')].map(i => i.value).join(',')")
    assert before != after, "a frame slider moved while playing"
    demo.page.locator("button.gfx-section-add", has_text="❚❚").first.click()


# --------------------------------------------------------------------- layout, keyboard, themes


def test_no_horizontal_scroll_on_a_phone(browser, base_url):
    d = Demo(browser, base_url, size=(390, 800)).ready()
    try:
        overflow = d.page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        assert overflow <= 1
        assert d.page.locator("canvas").first.is_visible()
        assert d.page.locator("button[title*='Precomputed demo']").is_visible()
    finally:
        d.close()


def test_the_menubar_is_reachable_and_operable_by_keyboard(demo):
    two_d = demo.page.get_by_role("button", name="2D", exact=True)
    two_d.focus()
    demo.page.keyboard.press("Enter")
    assert "gfx-active" in two_d.get_attribute("class")
    demo.page.keyboard.press("Shift+Tab")
    assert demo.page.evaluate("document.activeElement.textContent") == "3D"
    for _ in range(12):  # tabbing keeps landing on controls, never on a dead end
        demo.page.keyboard.press("Tab")
        assert demo.page.evaluate("document.activeElement !== document.body")


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_both_themes_keep_the_text_readable(browser, base_url, scheme):
    d = Demo(browser, base_url, scheme=scheme).ready()
    try:
        colors = d.page.evaluate(
            """() => { const pick = (sel) => { const e = document.querySelector(sel); if (!e) return null; const c = getComputedStyle(e);
                          let bg = c.backgroundColor, p = e; while ((bg === 'rgba(0, 0, 0, 0)' || bg === 'transparent') && p.parentElement) { p = p.parentElement; bg = getComputedStyle(p).backgroundColor; }
                          return [c.color, bg]; };
                       return {tab: pick('.gfx-tab'), label: pick('.gfx-label'), button: pick('.gfx-menubar .gfx-button')}; }"""
        )
        for name, pair in colors.items():
            if pair:
                assert contrast(*pair) >= 3.0, f"{name} is hard to read in the {scheme} theme: {pair}"
    finally:
        d.close()


# --------------------------------------------------------------------- failures


def test_a_missing_manifest_shows_an_error_and_nothing_else(browser, base_url):
    d = Demo(browser, base_url, route=lambda page: page.route("**/data/manifest.json", lambda r: r.fulfill(status=404)))
    try:
        d.page.wait_for_function("document.querySelector('#status')?.dataset.kind === 'error'", timeout=15000)
        assert "could not load" in d.page.locator("#status").inner_text()
        assert d.page.locator(".gfx-shell .gfx-menubar").count() == 0
    finally:
        d.close()


def test_a_slow_field_shows_a_loading_message_and_then_clears(browser, base_url):
    def slow(page):
        def delay(route):
            threading.Event().wait(1.0)
            route.continue_()

        page.route("**/fields/*.real.bin", delay)

    d = Demo(browser, base_url, route=slow)
    try:
        d.page.wait_for_selector(".gfx-shell .gfx-menubar", timeout=60000)
        d.page.wait_for_function("document.querySelector('#status')?.innerText.includes('Loading field')", timeout=15000)
        d.page.wait_for_function("document.querySelector('#status')?.hidden === true", timeout=60000)
    finally:
        d.close()


def test_a_failing_slice_download_reports_the_error(browser, base_url):
    d = Demo(browser, base_url, route=lambda page: page.route("**/sections/yz.json", lambda r: r.fulfill(status=500))).ready()
    try:
        d.page.get_by_role("button", name="2D", exact=True).click()
        d.page.get_by_role("button", name="yz", exact=True).click()
        d.page.wait_for_function("document.querySelector('#status')?.dataset.kind === 'error'", timeout=15000)
        assert "yz.json" in d.page.locator("#status").inner_text()
    finally:
        d.close()
