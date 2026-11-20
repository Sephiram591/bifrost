# Viewer

`bf.view(stack)` returns an [anywidget](https://anywidget.dev) viewer for a `bf.StackDevice` in Jupyter and marimo. It displays the device in 3D and 2D cross-sections, lets you configure sources and monitors, runs simulations in the background, and visualizes the recorded fields.

## Open the viewer

Display a straight silicon waveguide in the viewer by ending a Jupyter or marimo cell with `viewer`:

```python
--8<-- "examples/straight_waveguide.py:straight"
```

(In a standalone Python script, `bf.view(stack)` only constructs the widget object without displaying a UI. To try the viewer in your browser without Python, open the [Widget Demo](../live-demo.md) or the standalone [Explorer demo](../demo/index.html).)

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

Add a source and a monitor in the **Sources / Detectors** tab, set the grid and duration in **Simulation**, and click **Run**. The **Meshing** card defaults to `adaptive` (`ppw` = 15, `PML (µm)`, `global dilation`, `global z pad`, `max cell growth`, and per-layer **Dilation layers** sub-cards backed by [Automatic meshing](simulation-setup.md#automatic-meshing)), and can be switched to `uniform` (`resolution`, `PML (cells)`). The simulation runs on a background thread with the setup locked while running, and the results populate the **Result** tab. You can also access the prepared objects in Python via `viewer.objects`, `viewer.config`, and `viewer.constraints`, or click **copy launch code** to copy a reproducible script.

## Load a finished simulation

To inspect a simulation run from a script, pass its outputs to `viewer.load_result` (as in [Getting started](../getting-started.md) and [Results and field monitors](results.md)):

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
- **Video packaging (`wrap_video`):** MP4 packaging from WebCodecs streams runs in the kernel through [`bifrost.viewer.widget.wrap_video`][bifrost.viewer.widget.wrap_video] (`wrap_video(stream: bytes, codec: str, fps: int) -> bytes`, `1 <= fps <= 240`), which remuxes Annex B `"h264"` streams or transcodes IVF `"vp8"`/`"vp9"` streams to H.264 using `imageio-ffmpeg`'s bundled `ffmpeg`. Video export requires a live Python kernel and is unavailable in the static browser demos.

## Troubleshooting

- **The widget displays as text such as `Viewer(...)`, or nothing appears.** The viewer requires an interactive notebook frontend that supports `anywidget` (Jupyter or marimo), and `viewer` must be the final expression of the cell. Plain scripts do not render the widget.
- **The viewer is blank or the 3D view does not draw.** The 3D viewport requires WebGL. Verify that hardware acceleration is enabled in your browser or try another browser.
- **Run does nothing or fails.** A simulation requires at least one source, and requires a monitor to record fields. Setup errors are displayed directly in the viewer; for full control over custom placement constraints, use a Python script as shown in [Simulation setup and ports](simulation-setup.md).
- **The Result tab says no result is loaded.** Run a simulation from the viewer or call `viewer.load_result(...)` as shown above.
