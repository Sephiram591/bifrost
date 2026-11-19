# Viewer

`bf.view(stack)` returns an [anywidget](https://anywidget.dev) viewer for a `bf.StackDevice`, for Jupyter and marimo. It
shows the device in 3D and in 2D cross-sections, sets up sources and monitors, runs the simulation, and shows the fields
a finished simulation recorded.

## Open the viewer

A straight silicon waveguide in the viewer (in Jupyter or marimo, end the cell with `viewer`):

```python
--8<-- "examples/straight_waveguide.py:straight"
```

(Outside a notebook, a script only builds the widget. To look at one without Python, use the [Widget Demo](../live-demo.md)
or the standalone [Explorer demo](../demo/index.html).)

The viewer has four tabs on its left:

| Tab | What it does |
| --- | --- |
| **Layout / Ports** | the layers (color, opacity, visibility) and ports of the device |
| **Sources / Detectors** | add sources and monitors on component ports, simulation-box plane ports (`xy`, `xz`, `yz`), or absolute coordinates (`None`); double-click a port in 3D to add a monitor |
| **Simulation** | the wavelength, **Meshing** card (`uniform` or `adaptive`), run time, absorbing boundary and the simulation box |
| **Result** | the last result: a field per monitor on its plane, frame sliders, playback and MP4 recording |

The **3D** and **2D** buttons switch the view. In 2D you pick a plane (`xy`, `xz` or `yz`) and move a slider through the
device. The menubar camera button saves the current view as a PNG.

### Placing plane ports in Sources / Detectors

In addition to component ports, the **port** dropdown in **Sources / Detectors** offers simulation-box plane ports
(`xy`, `xz`, `yz`) and `None` (these virtual planes are not listed in **Layout / Ports**):

- **`xy`, `xz`, `yz` (plane ports):** places the source or monitor through the simulation box's center spanning the box's
  in-plane extent (`port_padding` defaults to `(0, 0)`). Here `"in"` faces the negative normal axis (`-z`, `-y`, or `-x`),
  `"out"` faces the positive axis, and a positive `port_offset` shifts along `"in"` (`-axis`). `level` defaults to `(auto)`
  (the box's mid-height) and can also target a stack level or `z_center`.
- **`None` (absolute placement):** switches to `axis` (`0` = x / yz plane, `1` = y / xz plane, `2` = z / xy plane),
  `center` `(x, y, z)` in µm (stack coordinates), and `size` `(u, v)` in µm (`"in"` points along `+axis`).

## Run a simulation from the viewer

Add a source and a monitor in the **Sources / Detectors** tab, choose the grid and time in **Simulation**, and press
**Run**. The **Meshing** card switches between `uniform` (`resolution`, `PML (cells)`) and `adaptive` (`ppw`,
`PML (µm)`, `global dilation`, `global z pad`, `max cell growth`, and per-layer **Dilation layers** sub-cards backed
by [Automatic meshing](simulation-setup.md#automatic-meshing)). The run happens in the Python session on a background
thread, with the setup locked while it goes, and its field appears in the **Result** tab. To place and run the objects
yourself, they are in `viewer.objects`, `viewer.config` and `viewer.constraints`; the **copy launch code** button copies
a script that reproduces the setup.

## Load a finished simulation

A result simulated in a script goes to the viewer as the last step of that script (as in the
[bend tutorial](../getting-started.md) and [Results and field monitors](results.md)):

```python
viewer = bf.view(stack)
viewer.load_result(objects, arrays, config, stack=stack)
```

`objects`, `arrays` and `config` are what `fdtdx.place_objects` and `fdtdx.run_fdtd` return, and `stack` is the
`StackDevice` the result was recorded on.

The **Result** and **Sources / Detectors** tabs then list the run's sources and monitors with their names, types and
planes. They are read-only, because the viewer has the objects fdtdx placed, not the calls that made them. To simulate
again from the viewer, add your own sources and monitors: running replaces the listed ones.

## Exporting video and images

- **PNG screenshots:** the menubar camera button saves the active 3D or 2D viewport (or the whole widget when
  **Screenshot** is switched to `whole widget` in the gear **Viewer settings** panel), and the copy button writes the
  same PNG to the clipboard.
- **MP4 video recording:** in **Result**, multi-frame detectors (multi-frequency phasor monitors and time-domain
  `FieldDetector`s) expose **play**, an **fps** input (1–120 in the UI), and a **save video** (video camera) button
  that encodes every frame of the active viewport client-side with WebCodecs (`H.264`, or `VP9`/`VP8` fallback). When
  time-domain detectors are present, the **Detectors** header also shows **play all** / **record all** buttons to
  export every time-domain detector together as `time-domain-detectors.mp4`.
- **Save destination and folder:** in **Viewer settings**, **Save to** switches between `browser (download)` (default)
  and `server (kernel)`. Server saves write non-clobbering files into `viewer.save_dir`, set by
  `bf.view(stack, save_dir=...)`, `viewer.save_dir`, or the **Save folder** text field.
- **Python helper (`wrap_video`):** `Viewer` does not define an `export_video()` method; kernel-side MP4 packaging runs
  through `bifrost.viewer.widget.wrap_video(stream: bytes, codec: str, fps: int) -> bytes` (`1 <= fps <= 240`), which
  remuxes Annex B `"h264"` streams (`-c:v copy`) or transcodes IVF `"vp8"`/`"vp9"` streams to H.264 using
  `imageio-ffmpeg`'s bundled `ffmpeg` (available only in a live Python kernel, not the static demos).

## Troubleshooting

- **A widget shows as text such as `Viewer(...)`, or not at all.** The viewer needs a notebook that supports widgets
  (Jupyter with `anywidget` installed, or marimo). End the cell with the viewer object. In a plain script nothing is
  displayed.
- **The viewer is blank or the 3D view does not draw.** It needs WebGL. Check that the browser has hardware acceleration
  on, or try another browser.
- **Run does nothing, or fails.** A run needs a source, and a monitor to record anything; a problem with the setup is
  reported in the viewer. For a run you control, use the scripts in [Simulation setup and ports](simulation-setup.md).
- **The Result tab says no result is loaded.** Run from the viewer, or call `viewer.load_result(...)` as above.
