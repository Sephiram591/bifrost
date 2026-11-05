---
title: Widget Demo
---

<div class="bf-live-demo" style="background: #101014 url(../images/viewer-demo.png) center / cover no-repeat;">
  <iframe src="demo/index.html" title="bifrost viewer: a precomputed 90 degree waveguide bend" loading="lazy" allowfullscreen></iframe>
</div>

<div class="bf-live-demo-notes" markdown>

The viewer on the result of the [bend tutorial](getting-started.md), in your browser with no installation and no
Python. Drag to orbit, scroll to zoom, and use the **Result** tab to choose what to display.

[Open the demo full screen](demo/index.html){ target="_blank" }

The demo needs a browser with WebGL. If it does not load, the picture behind it shows the same result; reload the page to
try again.

## What the demo can and cannot do

The demo plays back a **precomputed** result, so only what needs no Python works:

- Orbit, zoom, fit the view, switch between 3D and 2D, and toggle layers and ports.
- Show and style the fields: pick the component and the transform (real, imaginary, magnitude, phase, log), the colors and
  the opacity, and choose the frame of a monitor that recorded several.
- **Play** the recorded movie of the pulse crossing the bend.
- Move the cross-section slider. The 2D slices are stored at evenly spaced positions, so the slider **snaps** to them and shows
  the exact coordinate; there is no field *between* the recorded planes. A field is drawn only on the plane of the
  monitor that recorded it, and in a 2D slice only where the slice is that plane: the demo has no field volume.
- **Run** is disabled and labeled "Precomputed demo"; the setup (sources, monitors, simulation settings) is read-only,
  and saving a video needs Python, so it says so. Saving a PNG of the view works.

## How it was made

The demo's data is the output of one real run of `examples/waveguide_bend.py`, exported by `scripts/build_demo.py` (see
[Contributing and tests](development.md#the-live-demo)). `docs/demo/data/manifest.json` records the source revision, the package
versions, the simulation parameters, and the size of every file. The documentation build only publishes those prepared
files; it never reruns the simulation.

To run the viewer on your own devices, see the [Viewer guide](guide/viewer.md).

</div>
