# Results and field monitors

When `fdtdx.run_fdtd` returns, everything the monitors recorded is in `arrays.detector_states`, keyed by monitor name.
The example is a short simulation of a straight waveguide (about a minute on 8 CPU cores).

## Run a short simulation

The device and settings of [Simulation setup and ports](simulation-setup.md), plus two Poynting-flux monitors and a
field monitor, run for 300 fs so the whole pulse crosses the guide:

```python
--8<-- "examples/guide/results.py:simulate"
```

## Read the monitors

```python
--8<-- "examples/guide/results.py:read"
```

Output:

```text
transmission: 1.000
flux ratio: 1.000
field grid: (180, 74) peak |E|^2: 1132.8427734375
```

- **Mode monitors** (`ModeOverlapDetector`) give the complex amplitude of a guided mode on their plane:
  `monitor.compute_overlap(states[name])`, one entry per wave character. The squared ratio of the output to the input
  amplitude is the **transmission**, normalized without a reference run because both monitors measure the same mode of
  the same guide. A straight guide transmits about 1: a quick check of the source, the monitors and the run time.
- **Poynting-flux monitors** (`PoyntingFluxDetector`) record the power through their plane at every time step, in
  fdtdx's units. Summed over the run it is the energy that crossed, so the ratio of two of them cross-checks the
  transmission without assuming a mode (it also counts light in other modes).
- **Field monitors** (`PhasorDetector`) record the field's complex amplitude at the wave characters you give them, on the
  monitor's grid: `states[name]["phasor"][0, 0, :, :, :, 0]` is `(components, nx, ny)` for an xy plane one cell thick.
  `|E|²` is the sum of the squared magnitudes over the components. For the field at many *times*, a `FieldDetector`
  records the same plane at time steps you choose (the [bend example](../examples/waveguide-bend.md) records 40 of them,
  which the viewer plays as a movie).

## Choosing a monitor

| You want | Use | Read it with |
| --- | --- | --- |
| the power in a guided mode (transmission) | `create_mode_overlap_detector_at` | `compute_overlap(state)` |
| the power through a plane, any mode | `create_poynting_flux_detector_at` | `state["poynting_flux"]` |
| the field's amplitude and phase at a wavelength | `PhasorDetector` | `state["phasor"]` |
| the field at a sequence of times (a movie) | `FieldDetector` with an `OnOffSwitch(interval=...)` | the recorded frames |
| the energy in a region | `create_energy_detector_at` | `state["energy"]` |

Mode and flux monitors are placed on ports (see [Simulation setup and ports](simulation-setup.md)); a field monitor is
any fdtdx detector, sized and placed with constraints, as in the script.

## Open the result in the viewer

```python
--8<-- "examples/guide/results.py:viewer"
```

`viewer.load_result` takes what fdtdx hands out: the placed objects, the arrays and the config. The **Result** tab then
has one read-only entry per monitor, with the field of every plane-shaped monitor drawn on its plane and a frame slider
for monitors that recorded several frames; the **Sources / Detectors** tab lists the same objects. In a notebook, end the
cell with `viewer`. More in [Viewer](viewer.md).

## Troubleshooting

- **The transmission is far from 1 for a straight guide.** The run was too short: the output monitor must see the whole
  pulse. Make `time` longer than the pulse needs to cross the device (at 200 fs this script reported 0.87).
- **A monitor reports nothing, or the viewer shows no field for it.** The viewer draws monitors that record a plane (one
  cell thick along its normal), and not ones that sum over their volume (`reduce_volume=True`).
- **The numbers are tiny or huge.** fdtdx's flux and amplitudes are in its own units; compare ratios, not absolute values.
