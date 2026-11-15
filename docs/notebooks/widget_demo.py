# /// script
# [tool.marimo.display]
# theme = "dark"
# ///
"""The Widget Demo page's notebook: the bifrost viewer on a precomputed 90 degree waveguide bend.

It never imports bifrost. The viewer is the same front end `bf.view(stack)` shows, played back from the
files `scripts/build_demo.py` exported to docs/demo/data, so it runs in a browser with no jax, no fdtdx
and no GPU. The docs build exports this file with `marimo export html-wasm` into docs/demo/notebook/
(`make demo-notebook`), and docs/live-demo.md embeds the result in an iframe, which keeps the site's menu.

All the text of the Widget Demo page is here, under the viewer; the page itself holds only the iframe.
The notebook starts in dark mode (the header above, which `marimo export` reads), and the viewer starts
with its cards collapsed (js/viewer/demo.ts). Links to other pages of the site are written from the
exported notebook (docs/demo/notebook/) and open in the whole window (`target="_top"`), not in the iframe.

The widget finds the viewer bundle and the data relative to the exported page (docs/demo/notebook/),
so the notebook is meant to be read from there; `marimo edit` on this file shows the text but not the viewer.
"""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full", app_title="bifrost viewer demo")


@app.cell(hide_code=True)
def _():
    import json
    import pathlib

    import anywidget
    import marimo as mo
    import traitlets

    # the text keeps a readable line length under the viewer, which spans the whole window
    PROSE = {"max-width": "61rem", "margin": "0 auto", "padding": "0 1rem"}
    return PROSE, anywidget, json, mo, pathlib, traitlets


@app.cell(hide_code=True)
def _(anywidget, mo, traitlets):
    class _Replay(anywidget.AnyWidget):
        """Mounts the demo bundle on the exported files; the bundle is `docs/demo/viewer-demo.js`."""

        _esm = """
        export default {
          async render({ model, el }) {
            // paths are relative to the notebook page, docs/demo/notebook/index.html once exported
            const bundle = new URL(model.get("bundle"), document.baseURI).href;
            const sheet = new URL("viewer.css", bundle).href;
            const root = el.getRootNode();
            const into = root instanceof ShadowRoot ? root : document.head;
            if (![...into.querySelectorAll("link")].some((l) => l.href === sheet)) {
              const link = document.createElement("link");
              link.rel = "stylesheet";
              link.href = sheet;
              into.append(link);
            }
            const eczar = "https://fonts.googleapis.com/css2?family=Eczar:wght@400&display=swap";
            if (![...document.head.querySelectorAll("link")].some((l) => l.href === eczar)) {
              const font = document.createElement("link");
              font.rel = "stylesheet";
              font.href = eczar;
              document.head.append(font);
            }
            el.style.height = model.get("height");
            const demo = await import(bundle);
            await demo.mountInto(el, model.get("src"));
          },
        };
        """

        src = traitlets.Unicode("../data/").tag(sync=True)
        bundle = traitlets.Unicode("../viewer-demo.js").tag(sync=True)
        height = traitlets.Unicode("max(420px, calc(100vh - 11rem))").tag(sync=True)

    viewer = mo.ui.anywidget(_Replay())
    viewer
    return


@app.cell(hide_code=True)
def _(PROSE, mo):
    mo.md(
        """
        The viewer on the result of the <a href="../../getting-started/" target="_top">bend tutorial</a>, in your browser with
        no installation and no Python. Drag to orbit, scroll to zoom, and use the **Result** tab to choose what to display.

        <a href="index.html" target="_blank">Open the demo full screen</a> ·
        <a href="../index.html" target="_blank">the same viewer without the notebook</a>

        The demo is a <a href="https://marimo.io" target="_blank">marimo</a> notebook that runs in your browser: it starts a
        Python runtime there (Pyodide), so the first load takes a few seconds longer than the plain viewer. It needs a
        browser with WebGL. If it does not load, the picture behind it shows the same result; reload the page to try again,
        or use the plain viewer.
        """
    ).style(PROSE)
    return


@app.cell(hide_code=True)
def _(PROSE, json, mo, pathlib):
    def _read(where) -> str:
        if str(where).startswith("http"):  # in the browser, where the notebook's location is a URL
            from pyodide.http import open_url

            return open_url(str(where)).read()
        return pathlib.Path(where).read_text()

    _here = mo.notebook_location()
    # exported to docs/demo/notebook, the data is a sibling folder; from docs/notebooks it is under demo/
    _data = _here.parent / "data" if str(_here).startswith("http") else _here.parent / "demo" / "data"
    _m = json.loads(_read(_data / "manifest.json"))
    _p, _r = _m["parameters"], _m["result"]
    mo.md(
        f"A {_p['width_um']} × {_p['thickness_um']} µm silicon waveguide ({_p['core_index']}) in oxide "
        f"({_p['clad_index']}) turning through a {_p['radius_um']} µm bend at {_p['wavelength_um']} µm, "
        f"simulated with fdtdx on `{_m['device']}` at {_p['resolution_um']} µm resolution: "
        f"transmission **{_r['transmission']:.3f}** in {_r['wall_seconds']} s. "
        f"Source: `{_p['example']}` at commit `{_m['source']['commit'][:7]}`."
    ).style(PROSE)
    return


@app.cell(hide_code=True)
def _(PROSE, mo):
    mo.md(
        """
        ## What the demo can and cannot do

        The demo plays back a **precomputed** result, so only what needs no Python works:

        - Orbit, zoom, fit the view, switch between 3D and 2D, and toggle layers and ports.
        - Show and style the fields: pick the component and the transform (real, imaginary, magnitude, phase, log), the
          colors and the opacity, and choose the frame of a monitor that recorded several.
        - **Play** the recorded movie of the pulse crossing the bend.
        - Move the cross-section slider. The 2D slices are stored at evenly spaced positions, so the slider **snaps** to
          them and shows the exact coordinate; there is no field *between* the recorded planes. A field is drawn only on
          the plane of the monitor that recorded it, and in a 2D slice only where the slice is that plane: the demo has
          no field volume.
        - **Run** is disabled and labeled "Precomputed demo"; the setup (sources, monitors, simulation settings) is
          read-only, and saving a video needs Python, so it says so. Saving a PNG of the view works.
        """
    ).style(PROSE)
    return


@app.cell(hide_code=True)
def _(PROSE, mo):
    mo.md(
        """
        ## How it was made

        The demo's data is the output of one real run of `examples/waveguide_bend.py`, exported by `scripts/build_demo.py`
        (see <a href="../../development/#the-live-demo" target="_top">Contributing and tests</a>).
        `docs/demo/data/manifest.json` records the source revision, the package versions, the simulation parameters, and
        the size of every file. The documentation build only publishes those prepared files; it never reruns the
        simulation.

        The page around the viewer is this notebook,
        <a href="https://github.com/Sephiram591/bifrost/blob/main/docs/notebooks/widget_demo.py" target="_blank"><code>docs/notebooks/widget_demo.py</code></a>,
        exported with `marimo export html-wasm` when the docs are built. It does not import <span class="bf-word" style="font-family: 'Eczar', serif; font-weight: 400;">bifrost</span> (jax, fdtdx and
        gdsfactory do not run in a browser): its widget loads the same viewer bundle on the exported files, and one of its
        cells reads the run's `manifest.json` for the summary under the viewer.

        To run the viewer on your own devices, see the <a href="../../guide/viewer/" target="_top">Viewer guide</a>.
        """
    ).style(PROSE)
    return


if __name__ == "__main__":
    app.run()
