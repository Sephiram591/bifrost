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
    import anywidget
    import marimo as mo
    import traitlets

    # the text keeps a readable line length under the viewer, which spans the whole window
    PROSE = {"max-width": "61rem", "margin": "0 auto", "padding": "0 1rem"}
    return PROSE, anywidget, mo, traitlets


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
        r"""
        # Viewer

        <a href="../../api/bifrost/viewer/widget/#bifrost.viewer.widget.view" target="_top"><code>bf.view</code></a> returns an <a href="https://anywidget.dev" target="_blank">anywidget</a> viewer for a <a href="../../api/bifrost/stack/" target="_top"><code>bf.StackDevice</code></a> in Jupyter and marimo. It displays the device in 3D and 2D cross-sections, lets you configure sources and monitors, runs simulations in the background, and visualizes the recorded fields.

        ## Open the viewer

        Display a straight silicon waveguide in the viewer by ending a Jupyter or marimo cell with `viewer`:

        ```python
        import fdtdx
        import gdsfactory as gf
        import bifrost as bf
        from gdsfactory.technology import LayerLevel, LayerStack

        gf.gpdk.PDK.activate()

        component = gf.components.straight(length=5)  # a 5 um long, 500 nm wide waveguide with ports o1 and o2

        # one 220 nm thick level on the waveguide layer; silicon core in a silica cladding
        layer_stack = LayerStack(
            layers={"core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=0.22, material="core", mesh_order=1)}
        )
        materials = {
            "core": fdtdx.Material(permittivity=3.48**2),
            "clad": fdtdx.Material(permittivity=1.444**2),
        }

        stack = bf.StackDevice(
            component, layer_stack, materials,
            background="clad",                               # fills everything the core does not
            zrange_layers=["core"],                          # the box's z range is the core's
            extend_ports=(("o1", 1.0), ("o2", 1.0)),         # 1 um straight leads out of both ports
            boundary_offset=(0, 0, 1.5, -1.5, 1.5, -1.5),    # 1.5 um of cladding in y and z
            name="stack",
        )
        viewer = bf.view(stack)  # the interactive 3D viewer
        viewer
        ```

        (In a standalone Python script, `bf.view(stack)` only constructs the widget object without displaying a UI. The precomputed demo above runs in your browser without Python, or you can open the standalone <a href="../index.html" target="_blank">Explorer demo</a>.)

        The viewer has four tabs on the left:

        | Tab | What it does |
        | --- | --- |
        | **Layout / Ports** | Configure level colors, opacities, and visibility, and toggle port markers |
        | **Sources / Detectors** | Add sources and monitors on component ports, simulation-box plane ports (`xy`, `xz`, `yz`), or absolute coordinates (`None`); double-click a port in 3D to add a monitor |
        | **Simulation** | Configure wavelength, **Meshing** (`adaptive` or `uniform`), run time, absorbing boundary, and simulation box offsets |
        | **Result** | Inspect recorded fields on monitor planes, scrub multi-frame detectors, play animations, and export MP4 videos |

        Use the **3D** and **2D** buttons to switch views. In 2D mode, select a plane (`xy`, `xz`, or `yz`) and scrub the slice slider through the device. The camera button in the menubar exports the current view as a PNG.

        ### Placing plane ports in Sources / Detectors

        In addition to component ports, the **port** dropdown in **Sources / Detectors** offers simulation-box plane ports (`xy`, `xz`, `yz`) and `None` (these virtual planes are not listed in **Layout / Ports**):

        - **`xy`, `xz`, `yz` (plane ports):** Places the source or monitor through the simulation box's center spanning the box's in-plane extent (`port_padding` defaults to `(0, 0)`). Here `"in"` faces the negative normal axis (`-z`, `-y`, or `-x`), `"out"` faces the positive axis, and a positive `port_offset` shifts along `"in"` (`-axis`). `level` defaults to `(auto)` (the box's mid-height) and can also target a stack level or `z_center`.
        - **`None` (absolute placement):** Switches to `axis` (`0` = x / yz plane, `1` = y / xz plane, `2` = z / xy plane), `center` `(x, y, z)` in µm (stack coordinates), and `size` `(u, v)` in µm (`"in"` points along `+axis`).

        ## Run a simulation from the viewer

        Add a source and a monitor in the **Sources / Detectors** tab, set the grid and duration in **Simulation**, and click **Run**. The **Meshing** card defaults to `adaptive` (`ppw` = 15, `PML (µm)`, `global dilation`, `global z pad`, `max cell growth`, and per-layer **Dilation layers** sub-cards backed by <a href="../../guide/simulation-setup/#automatic-meshing" target="_top">Automatic meshing</a>), and can be switched to `uniform` (`resolution`, `PML (cells)`). The simulation runs on a background thread with the setup locked while running, and the results populate the **Result** tab. You can also access the prepared objects in Python via `viewer.objects`, `viewer.config`, and `viewer.constraints`, or click **copy launch code** to copy a reproducible script.

        ## Load a finished simulation

        To inspect a simulation run from a script, pass its outputs to `viewer.load_result` (as in <a href="../../getting-started/" target="_top">Getting started</a> and <a href="../../guide/results/" target="_top">Results and field monitors</a>):

        ```python
        viewer = bf.view(stack)
        viewer.load_result(objects, arrays, config, stack=stack)
        ```

        `objects`, `arrays`, and `config` come directly from `fdtdx.place_objects` and `fdtdx.run_fdtd`, and `stack` is the `StackDevice` the result was recorded on.

        The **Result** and **Sources / Detectors** tabs list the run's sources and monitors with their names, types, and planes as read-only entries (representing the placed `fdtdx` objects). To run a new simulation from the viewer, add new sources and monitors; clicking **Run** replaces the adopted ones.

        ## Exporting video and images

        - **PNG screenshots:** The menubar camera button saves the active 3D or 2D viewport (or the entire widget when **Screenshot** is switched to `whole widget` in the **Viewer settings** panel), and the copy button writes the PNG to the clipboard.
        - **MP4 video recording:** In **Result**, multi-frame detectors (multi-frequency phasor monitors and time-domain `FieldDetector`s) expose **play**, an **fps** input (1–120 in the UI), and a **save video** button that encodes every frame client-side with WebCodecs (`H.264`, or `VP9`/`VP8` fallback). When time-domain detectors are present, the **Detectors** header also shows **play all** and **record all** buttons to export every time-domain detector together as `time-domain-detectors.mp4`.
        - **Save destination and folder:** In **Viewer settings**, **Save to** switches between `browser (download)` (default) and `server (kernel)`. Server saves write non-clobbering files into `viewer.save_dir`, configurable via `bf.view(stack, save_dir=...)`, `viewer.save_dir`, or the **Save folder** input.
        - **Video packaging (`wrap_video`):** MP4 packaging from WebCodecs streams runs in the kernel through <a href="../../api/bifrost/viewer/widget/#bifrost.viewer.widget.wrap_video" target="_top"><code>bifrost.viewer.widget.wrap_video</code></a>, which remuxes Annex B `"h264"` streams or transcodes IVF `"vp8"`/`"vp9"` streams to H.264 using `imageio-ffmpeg`'s bundled `ffmpeg`. Video export requires a live Python kernel and is unavailable in the static browser demos.

        ## Troubleshooting

        - **The widget displays as text such as `Viewer(...)`, or nothing appears.** The viewer requires an interactive notebook frontend that supports `anywidget` (Jupyter or marimo), and `viewer` must be the final expression of the cell. Plain scripts do not render the widget.
        - **The viewer is blank or the 3D view does not draw.** The 3D viewport requires WebGL. Verify that hardware acceleration is enabled in your browser or try another browser.
        - **Run does nothing or fails.** A simulation requires at least one source, and requires a monitor to record fields. Setup errors are displayed directly in the viewer; for full control over custom placement constraints, use a Python script as shown in <a href="../../guide/simulation-setup/" target="_top">Simulation setup and ports</a>.
        - **The Result tab says no result is loaded.** Run a simulation from the viewer or call `viewer.load_result(...)` as shown above.
        """
    ).style(PROSE)
    return


if __name__ == "__main__":
    app.run()
