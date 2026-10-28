# Viewer

`bf.view(stack)` returns an [anywidget](https://anywidget.dev) app shell for a `bf.StackDevice`. It works in marimo and
Jupyter.

```python
import bifrost as bf

viewer = bf.view(stack)
viewer          # displays the widget
```

What it shows and does:

- the 2D level regions and 3D solids of the layer stack, painted by `mesh_order`, with port glyphs;
- cross-sections through the device;
- sources and detectors as declarative specs, which can be edited in the widget;
- a run: `viewer.run()` launches the simulation on a background thread with the setup locked meanwhile;
  `viewer.objects`, `viewer.config` and `viewer.constraints` expose the prepared fdtdx objects;
- field display with a ramp and timeline, and video export (uses `imageio-ffmpeg`);
- a server-side image save (`save_dir` argument), writing a PNG of the viewport.

Modules: `bf.viewer.geometry` (analytic level regions), `solids` (3D meshes), `state` (shared state), `payload` (wire
format), `sim` (source/detector specs and the run), `fields`, `section`, `adopt` (adopting a foreign fdtdx result) and
`widget` (the shell).

The design decisions are recorded in the [viewer design spec](design/viewer-spec.md). The front end is TypeScript under
`js/viewer/` and bundled into `bifrost/viewer/static/viewer.js` (`make viewer-js`).
