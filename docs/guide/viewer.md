# Viewer

`bf.view(stack)` returns an [anywidget](https://anywidget.dev) viewer for a `bf.StackDevice`, for Jupyter and marimo. It
shows the device in 3D and in 2D cross-sections, sets up sources and monitors, runs the simulation, and shows the fields
a finished simulation recorded.

## Open the viewer

A straight silicon waveguide in the viewer (in Jupyter or marimo, end the cell with `viewer`):

```python
--8<-- "examples/straight_waveguide.py:straight"
```

(Outside a notebook, a script only builds the widget. To look at one without Python, use the [live demo](../live-demo.md).)

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
appears in the **Result** tab. To place and run the objects yourself, they are in `viewer.objects`, `viewer.config` and
`viewer.constraints`; the **copy launch code** button copies a script that reproduces the setup.

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

## Troubleshooting

- **A widget shows as text such as `Viewer(...)`, or not at all.** The viewer needs a notebook that supports widgets
  (Jupyter with `anywidget` installed, or marimo). End the cell with the viewer object. In a plain script nothing is
  displayed.
- **The viewer is blank or the 3D view does not draw.** It needs WebGL. Check that the browser has hardware acceleration
  on, or try another browser.
- **Run does nothing, or fails.** A run needs a source, and a monitor to record anything; a problem with the setup is
  reported in the viewer. For a run you control, use the scripts in [Simulation setup and ports](simulation-setup.md).
- **The Result tab says no result is loaded.** Run from the viewer, or call `viewer.load_result(...)` as above.
