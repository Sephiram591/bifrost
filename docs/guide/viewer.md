# Viewer and live demo

`bf.view(stack)` returns an [anywidget](https://anywidget.dev) viewer for a `bf.StackDevice`. It works in Jupyter and
marimo, and shows the device in 3D and in 2D cross-sections, lets you set up sources and monitors, can run the simulation,
and shows the fields a finished simulation recorded.

## Open the viewer

This script is complete. It builds a straight silicon waveguide and creates the viewer; in a Jupyter or marimo cell, end the
cell with `viewer` to display it.

```python
--8<-- "examples/straight_waveguide.py:straight"
```

(Outside a notebook, a script only builds the widget. To look at one without Python, use the [live demo](#live-demo).)

The viewer has four tabs on its left:

| Tab | What it does |
| --- | --- |
| **Layout / Ports** | the layers (color, opacity, visibility) and ports of the device |
| **Sources / Detectors** | add sources and monitors on ports and set their parameters; double-click a port in 3D to add a monitor |
| **Simulation** | the wavelength, grid, run time, absorbing boundary and the simulation box |
| **Result** | the last result: a field per monitor on its plane, frame sliders, playback and a movie recorder |

The **3D** and **2D** buttons switch the view. In 2D you pick a plane (`xy`, `xz` or `yz`) and move a slider through the
device. The camera button saves the current view as a PNG.

## Run a simulation from the viewer

Add a source and a monitor in the **Sources / Detectors** tab, choose the grid and time in **Simulation**, and press
**Run**. The run happens in the Python session on a background thread, with the setup locked while it goes, and its field
appears in the **Result** tab. The same objects are available in code (`viewer.objects`, `viewer.config` and
`viewer.constraints`) if you want to place and run them yourself, and the **copy launch code** button copies the script that
reproduces the setup.

## Load a finished simulation

If you ran the simulation in a script, hand its result to the viewer. This is the end of the
[bend tutorial](../getting-started.md) and of [Results and field monitors](results.md), where the complete scripts are:

```python
viewer = bf.view(stack)
viewer.load_result(objects, arrays, config, stack=stack)
```

`objects`, `arrays` and `config` are what `fdtdx.place_objects` and `fdtdx.run_fdtd` return, and `stack` is the
`StackDevice` the result was recorded on. (This snippet is not runnable by itself: it is the last step of those scripts.)

## Live demo

This is the viewer on the result of the [bend tutorial](../getting-started.md): a 90° silicon waveguide bend simulated
with the settings of that page. It runs in your browser with no installation and no Python. Drag to orbit, scroll to zoom,
and use the **Result** tab to choose what to display.

<div style="position: relative; width: 100%; height: min(72vh, 640px); min-height: 380px; border-radius: 8px; overflow: hidden; background: #101014 url(../../images/viewer-demo.png) center / cover no-repeat;">
  <iframe src="../demo/index.html" title="bifrost viewer: a precomputed 90 degree waveguide bend" loading="lazy" allowfullscreen style="width: 100%; height: 100%; border: 0;"></iframe>
</div>

[Open the demo full screen](../demo/index.html){ target="_blank" }

If the demo does not load, the picture above is the same result; reload the page to try again. The demo needs a
browser with WebGL.

### What the demo can and cannot do

The demo is **precomputed**: it plays back a result that was simulated beforehand, so only what needs no Python works.

- Orbit, zoom, fit the view, switch between 3D and 2D, and toggle layers and ports.
- Show and style the fields: pick the component and the transform (real, imaginary, magnitude, phase, log), the colors and
  the opacity, and choose the frame of a monitor that recorded several.
- **Play** the recorded movie of the pulse crossing the bend.
- Move the cross-section slider. The 2D slices are stored at evenly spaced positions, so the slider **snaps** to them and shows
  the exact coordinate; there is no field *between* the recorded planes. A field is drawn only on the plane of the
  monitor that recorded it, and in a 2D slice only where the slice is that plane: the demo has no field volume.
- **Run** is disabled and labeled "Precomputed demo"; the setup (sources, monitors, simulation settings) is read-only,
  and saving a video needs Python, so it says so. Saving a PNG of the view works.

### How it was made

The demo's data is the output of one real run of `examples/waveguide_bend.py`, exported by `scripts/build_demo.py` (see
[Contributing and tests](../development.md#the-live-demo)). `docs/demo/data/manifest.json` records the source revision, the package
versions, the simulation parameters, and the size of every file. The documentation build only publishes those prepared
files; it never reruns the simulation.

## Troubleshooting

- **A widget shows as text such as `Viewer(...)`, or not at all.** The viewer needs a notebook that supports widgets
  (Jupyter with `anywidget` installed, or marimo). End the cell with the viewer object. In a plain script nothing is
  displayed.
- **The viewer is blank or the 3D view does not draw.** It needs WebGL. Check that the browser has hardware acceleration
  on, or try another browser.
- **Run does nothing, or fails.** A run needs a source, and a monitor to record anything; a problem with the setup is
  reported in the viewer. For a run you control, use the scripts in [Simulation setup and ports](simulation-setup.md).
- **The Result tab says no result is loaded.** Run from the viewer, or call `viewer.load_result(...)` as above.
