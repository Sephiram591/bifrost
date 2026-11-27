---
name: bifrost-field-manual
description: Deep reference for bifrost + gdsfactoryx + fdtdx. It explains how data flows from a layout to detector states, the coordinate and unit frames, how to size the box and planes, how to estimate GPU memory and run time, how to make results converge, what is differentiable, and a symptom-to-fix debugging playbook. Use it to design or review a forward simulation or inverse-design study from first principles, to diagnose wrong numbers or out-of-memory failures, or to tune speed and VRAM without losing physical accuracy.
---

# bifrost field manual: understand the pipeline, then budget it, debug it, and verify it

This skill teaches the **reasoning** behind correct bifrost scripts so you can handle devices no template covers. Each section ends with the **rules it implies**. Every symbol named here was checked in the bifrost source, docs, or examples (paths are given). §11 lists what was not verified.

**Reference paths**
- bifrost repo (or installed `bifrost` package; docs at https://sephiram591.github.io/bifrost/): key files are `bifrost/stack/_device.py` (StackDevice, port factories, `apply_stack`), `bifrost/meshing.py` (`mesh`, `Refine`, `Mesh`), `bifrost/jitting.py`, `bifrost/harminv.py`, `docs/guide/*.md`, `examples/`.
- **gdsfactory / gdsfactoryx APIs (mandatory):** `docs/llms-full.txt` in the [gdsfactoryx](https://github.com/Sephiram591/gdsfactoryx) repo (index: `docs/llms.txt`) covers component factories and defaults, ports, cross-sections, paths, PDK layers, and `@gf.cell` caching. Check both for `LayerStack` / `LayerLevel` first. In the version inspected, neither contained LayerStack/LayerLevel material, so take those APIs from `docs/guide/layer-stacks.md` and `examples/guide/layer_stacks.py` in bifrost.
- gdsfactoryx differentiability caveats: `docs/differentiable/` in the gdsfactoryx repo.
- Don't follow the `pip install gdsfactory` / `gplugins` lines in `llms-full.txt`. gdsfactoryx already provides the `gdsfactory` package and cannot share an environment with upstream gdsfactory.

---

## 1. Mental model: the pipeline and who owns what

```text
gdsfactoryx Component (um, gds xy)      LayerStack + materials dict
        |  polygons, ports                     |  levels: layer expr, zmin, thickness, mesh_order, material
        +-------------------+------------------+
                            v
bf.StackDevice ---- owns: the BOX (xy bbox of polygons + port extensions, z span of zrange_layers,
        |                 then boundary_offset), dtype, geometry options, ports, parameter layout
        |-- bf.mesh(stack, wavelength_um, ppw, refine) -> Mesh: fixed rectilinear edges + PML cell counts
        |-- stack.get_simulation_config(time=s, grid=mesh.grid, [gradient_config]) -> fdtdx.SimulationConfig
        |-- stack.get_simulation_volume() -> (volume, [center-the-device constraint])
        |-- stack.create_*_at(port, padding, ..., grid=config) -> (fdtdx object, constraints)
        |-- fdtdx.boundary_objects_from_config(mesh.boundary, volume)
                            v
fdtdx.place_objects(objects, config, constraints, key) -> (objects, arrays, params, config, _)
                            v                        (static: shapes, slices, grid. Compiles against these.)
bf.apply_stack(arrays, objects, {**params, **bf.init_stack_params(objects)}, key)
        writes every stack's eps (subpixel-averaged) -> fdtdx.apply_params for regular devices
        -> re-solves port modes on the FINAL eps (LRU-cached) -> casts PML coefficients to the device dtype
                            v                        (dynamic: differentiable in params["stack"])
fdtdx.run_fdtd(arrays=, objects=, config=, key=) -> (_, arrays);  arrays.detector_states[name]
                            v
objects[name].compute_overlap(state) / poynting_flux sums / phasors / bf.harminv_detector
```

**Rules this implies**
- **Static vs dynamic.** Everything before `place_objects` is static: box, grid, plane slices, number of vertices. Changing any of it means re-placing and recompiling. Everything in `params` is dynamic: vertices, `zmin/…`, `thickness/…`, `sidewall_angle/…`. For sweeps and optimization, place once and then feed new params through `stack.pack(...)` + `bf.apply_stack`. Never rebuild the `StackDevice` inside a loop.
- **`apply_stack` replaces `fdtdx.apply_params`.** fdtdx `Device`s mix only two materials, so `apply_params` alone does not write a StackDevice. It also re-solves source and monitor modes on the subpixel-averaged permittivity. Skipping it gives modes solved on a staircased painting.
- **The device must fill the volume** (`get_simulation_volume`) for `grid=` snapping to work. The snapper raises if the grid spans something other than the box.

---

## 2. Frames and units

| Frame | Used by | Unit | Origin |
|---|---|---|---|
| Layout / stack coordinates | component polygons, ports, `LayerLevel.zmin`, `Mesh.edges`, `Refine.regions`, `rectilinear_grid(x_um=…)` | µm | gds origin; z = stack height |
| fdtdx grid | `RectilinearGrid` edges, `partial_real_shape`, `WaveCharacter(wavelength=…)` | m | **box center** |
| Time | `config time=`, `OnOffSwitch(start_time=…)`, `harminv dt` | s | t = 0 at start |
| Spectral | `HarminvResult.frequency` (Hz), `.wavelength` (m), `.decay_rate` (1/s) | SI | — |
| Plane padding | `port_padding` | **wavelengths**: the mean of the object's `wave_characters`, or `wavelength_um` for flux/field/energy | — |

Faces and orientations:
- `boundary_offset = (x+, x-, y+, y-, z+, z-)`. Positive moves *any* face toward +axis, so `(m, -m, …)` grows the box.
- `bf.mesh(..., pml=, boundaries=)` six-tuples use `FACES = ("minx", "maxx", "miny", "maxy", "minz", "maxz")`. That is the reverse order within each axis, which is easy to get wrong.
- Port orientation is 0 = East, 90 = North, 180 = West, 270 = South (`llms-full.txt`).
- `extend_ports` continues each port straight out along its orientation.
- `port_offset > 0` moves a plane **into** the component, and `< 0` moves it **out** into the extension.
- For virtual plane ports `"xy"` / `"xz"` / `"yz"`, `"in"` faces −normal and positive `port_offset` shifts along `"in"`.

**Precision.**
- `StackDevice(dtype=jnp.float32)` is the default. Pass `jnp.float64` for double precision, which needs `jax.config.update("jax_enable_x64", True)` before placement.
- `get_simulation_config` copies the device dtype and **rejects `dtype=`**. It also converts `RectilinearGrid` edges to that dtype. Edges in a different dtype would make fdtdx promote the fields inside the time loop.
- Importing `gdsfactory` turns on x64 globally. `apply_stack` casts PML coefficients so float32 runs still work; the docs say no `jax.enable_x64(False)` wrapper is needed, although one older example still uses one.
- Large float32 vertex coordinates trigger a ULP warning. Center the component near (0, 0).

---

## 3. Box and plane geometry: the clearance arithmetic

The default PML is **inside** the box: `bf.mesh(pml=1.0)` µm on every face. Every source and port monitor must sit outside the PML and inside the box. A plane larger than the box is clipped.

Let `L_ext` be the extension length, `t_pml` the PML thickness, `o_src < o_mon ≤ 0` the port offsets, `W` the plane's lateral padding in µm (`port_padding[0] × λ`), `H` its vertical padding in µm, `m_xy` the lateral margin from `boundary_offset`, and `m_z` the z margin beyond the `zrange_layers` span.

| Constraint | Inequality | Example values (waveguide_bend.py) |
|---|---|---|
| Source clear of the PML along the port axis | `L_ext ≥ t_pml + |o_src| + ≥0.2` | 2.5 ≥ 1.0 + 1.0 + 0.5 |
| Plane clear of the PML laterally | `m_xy ≥ t_pml + W` (+ margin) | 2.25 ≥ 1.0 + 0.775 |
| Plane clear of the PML vertically | `m_z ≥ t_pml + H` (+ margin) | 1.59 ≥ 1.0 + 0.465 |
| Mode fits in the plane | 220 nm Si: about 0.5 µm into the cladding, so `(0.5, 0.3)` λ is enough. Low contrast (SiN on silica) needs `(0.5, 0.5)` | inverse_design.py |

- Faces that an extended port runs through get offset **0**, so the guide passes through the PML to the box face instead of ending inside the simulation. Faces without a port get the margin.
- Compact PMLs (`pml=0.5`) are used in the coupler and elevator examples, where the extensions are only 1 µm.

**Placement of the source and monitors.**
- Source at `port_offset=-1.0`, monitors at `-0.5`, both in the straight leads.
- In optimization, this keeps them clear of moving edges. A monitor on the port itself would see the bend move and distort its reference mode.
- The input monitor sits downstream of the source and uses `direction="in"`, so it counts only the forward wave and ignores reflections.

**Verify placement, don't assume it.** After `place_objects`, print `objects.volume.grid_shape` and each `(o.name, o.grid_shape, [(s.start, s.stop) for s in o.grid_slice])`. Compare the slices with `mesh.pml_cells`. This runs in seconds on a CPU (`examples/guide/simulation_setup.py`).

---

## 4. Memory and time model (be GPU-memory-conscious by construction)

**Cells set everything.** `Mesh.shape` gives the cells. `mesh.summary(time=T)` prints the cell count, min/max cell size, the CFL time step, the number of steps, and `fields (E and H, nbit): 6·cells·bytes GB`. That figure counts field arrays only.

**Measured peak VRAM** (convergence-testing.md, forward):
- About **0.21–0.25 GB per million cells** with `averaging="subpixel"`.
- About **0.60–0.62 GB per million cells** with `"subpixel_full"`.
- A **0.25–0.9 GB floor** below 1–2 M cells (XLA workspaces and 2D mode solves).
- float64 doubles array memory.
- The bend's cell count grows roughly as ppw³: 12 → 3.05 M, 16 → 7.02 M, 20 → 13.46 M, 24 → 23.03 M cells, which used 1.1 / 1.8 / 2.9 / 5.0 GiB.

**Where cells come from, with the levers that cut them:**
1. **Target size `λ / (n · ppw)` in each material.** ppw 20 is the default and is accurate for bends and couplers. ppw 14 is the documented fast setting for early optimization.
2. **Tensor-product penalty.** Each axis coordinate takes the smallest target anywhere in its cross-plane. A full-wafer substrate or cladding therefore forces its fine spacing across all x and y. Fix it with `bf.Refine(material_cells={"silica": λ / (n_silica · 13)})`; 13–14 cells per wavelength are enough there. The LiNbO₃ example uses `λ / (n_box · min(ppw, 12))`.
3. **Evanescent tails.** `global_dilation = global_z_padding = 3.0` decay lengths `l = λ / (2π √(n_hi² − n_lo²))`. Don't cut these to save cells: dropping the tail to 0 costs about 10× the n_eff error.
4. **Narrow features.** `feature_cells=1.5` cells across the narrowest part or gap. Use `min_cell` (µm) to stop a thin film from collapsing the time step.
5. **Box margins and PML thickness.** These are pure overhead, so size them with the §3 inequalities and no larger.

**The time step** is set by the **smallest cell anywhere** (CFL). Grading saves memory roughly in proportion to cells. Wall time falls in proportion only once the GPU is saturated (more than a few million cells). Small problems are dominated by kernel-launch latency.

**Other memory consumers you control:**
- Monitors. Use planes (`partial_grid_shape=(None, None, 1)`), not volumes. Keep `wave_characters` lists small. Give `FieldDetector` movies `switch=fdtdx.OnOffSwitch(interval=max(1, config.time_steps_total // N_FRAMES))`. Use `reduce_volume=True` when only an integral is needed.
- **Backward pass.** Use `fdtdx.GradientConfig(method="checkpointed", num_checkpoints=K)`. It stores K states and re-simulates between them (the examples use K = 30–35). The exact memory-vs-K curve is not documented; measure it.
- **Averaging mode.** `"subpixel_full"` runs fdtdx's full anisotropic update everywhere: 4.8–6.5× the median run time and 1.2–2.8× the median memory. Use it only for off-diagonal permittivity (rotated crystals) or when oblique-corner coupling matters.
- **Allocator.** The 200 M-cell coupler example sets `os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")` **before importing jax**, to avoid BFC fragmentation and its 75 % default cap.
- **Measure.** Read `(jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0)`, and profile geometry steps with `with bf.record_timings() as t: ...; print(t.table())`.

**Budget procedure.**
1. Build the device and mesh.
2. Read `summary`.
3. Estimate the forward peak with the per-million-cell rates above.
4. If it is over budget, pull the levers in order: material_cells → ppw → margins/PML → monitors → `StackOptions.fast()`. Use `fast()` only for geometry previews, never for final numbers.

---

## 5. Accuracy model and convergence protocol

What bifrost already does for accuracy, by default:
- Kottke subpixel averaging along each voxel's own normal (`averaging="subpixel"`, `yee_staggered=True`).
- An exact Green's-theorem voxelizer (`voxelizer="greens"`).
- Exact derived booleans (`derived_shapes="exact"`).
- Static horizontal interfaces centered mid-cell (`bf.mesh(center_faces=True)`).
- Graded meshing with 3-decay-length tails.

Don't turn these off for production numbers.

**Physics setup rules**
- **Materials.**
  - Lossless isotropic: `fdtdx.Material(permittivity=n**2)`.
  - Diagonal anisotropy: `permittivity=(ex, ey, ez)`.
  - Full tensor: requires `averaging="subpixel_full"`.
  - Dispersion and loss: `permittivity=eps_inf, dispersion=fdtdx.DispersionModel(poles=(...)), electric_conductivity=σ`.
  - Every level material in the box **and** the `background` must be keys of `materials`. Otherwise you get `KeyError`; restrict with `levels=[...]`.
- **Overlap priority.** `mesh_order` decides overlaps, and the **lower value wins** (core = 1, `WAFER` substrate = 9). `"WAFER"` is inverted by default: with no polygons drawn on it, it fills the plane.
- **Mode selection.** `mode_index` (0 = fundamental) plus `filter_pol="te"` or `"tm"`. Use identical settings on both monitors whose ratio you take.
- **Pulse.** Use `fdtdx.GaussianPulseProfile(center_wave=wave, spectral_width=fdtdx.WaveCharacter(frequency=0.05 * wave.get_frequency()))` for one wavelength. Widen the spectral width to about 0.10 for a band, and mesh that band as `bf.mesh(stack, (λmin, λmax))`. Pulsed source → `scaling_mode="pulse"`.
- **Duration.** The output monitor must see the whole pulse cross and ring down. Check it by re-running at 1.5× the time (the bend gave identical four digits at 400 fs vs 600 fs). Too short reads low.

**Convergence protocol (do this before reporting a number)**
1. Pick ppw 20 (or 14 for exploration). Record `T(ppw)`.
2. Re-run at ppw + 4, or at the documented check ppw = 24. Report both. On the bend, 20 → 24 changed T by 5e-4. Sensitive devices (multimode, couplers) need more.
3. Re-run at 1.5× the time.
4. For inverse designs, evaluate **both** the start design and the optimum on the finer grid. The gain must persist (0.5584 → 0.9934 at ppw 20 vs 0.5576 → 0.9930 at ppw 24 in the docs).
5. Cross-check the mode-overlap `T` against the Poynting-flux ratio. Flux ≥ mode `T`, and the difference is power in other modes.

**Geometry diagnostics.**
- `StackDiagnosticWarning` reports parts or gaps narrower than 1.5 cells. Treat it as an accuracy bug: refine, raise `xy_subsamples`, or widen the feature.
- `stack.diagnose(resolution=…)` returns the same findings as data.
- `mesh.summary()` reports how many starting vertices sit in cells wider than their target. `stack.coarse_vertices(params, mesh)` reports the same for moving vertices and runs under `jax.jit`, so it can be an auxiliary output of an objective.

---

## 6. Differentiability model

**What carries gradients**
- `params["stack"]` holds `vertices/<layer name>` (E, 2), `zmin/<level>`, `thickness/<level>`, and `sidewall_angle/<level>`, all in µm and degrees.
- `stack.pack(component)` maps a (possibly traced) gdsfactoryx component to the vertex entries, port extensions included.
- gdsfactoryx differentiates polygon vertices, ports, references, bounding boxes, paths, extrusion, and routes.
- gdsfactoryx does **not** differentiate booleans, `offset`, `over_under`, `fill`, or `get_polygons(merge=True)`.
- Grid snapping (1 nm) passes gradients straight through. Set `gf.snap.SNAP_ENABLED = False` when comparing against finite differences on unsnapped geometry.

**Topology is frozen at construction.**
- The device memorizes polygon count, vertices per polygon, T-junctions, seams, orientation, and where extensions meet port faces.
- `pack` raises if counts change, and checks topology on concrete inputs (never under tracing).
- If you change topology on purpose, call `objects[name].retopologize(c)` and refresh `params[name]`.
- For optimization, design cells with a **fixed vertex count and no Python branches on traced values**, and keep ports and leads fixed.

**Eager vs jit (the two docs disagree; here is how to reconcile them)**
- gdsfactoryx docs: "Differentiation is eager: wrap your objective in `jax.grad` but not in `jax.jit`." Cell functions keep Python control flow that needs concrete values.
- bifrost compiles `bf.jit(jax.value_and_grad(objective))` successfully when the cell's structure does not depend on the parameters. Verified cases: hand-built `jnp` polygons in `inverse_design.py`, and `gf.components.straight(length=traced)` in `tests/test_harminv.py`. `gradients.py` differentiates `bend_circular(radius=traced)` **eagerly**.
- Procedure: try `bf.jit`. If tracing fails on a concretization error, use eager `jax.value_and_grad` or rewrite the cell with fixed structure.

**Why `bf.jit` rather than `jax.jit`.** XLA's `simplify-conditional` pass turns the `lax.cond` in fdtdx's checkpointed backward loop into evaluating both branches, which runs a time step's VJP on every iteration. `bf.jit` disables that pass, scoped to the compiled function: 3146 s vs 4848 s on the inverse-design benchmark, with identical results. To use another jit wrapper, pass `compiler_options=bf.compiler_options()`.

**Compile once.** Every step must have identical shapes and dtypes:
- Pass `arrays` and `params` as **arguments**, not closure constants.
- Start the optimizer state committed and with a fixed dtype: `jax.device_put(jnp.asarray(z0, dtype=jnp.float64), jax.devices()[0])`. A weakly typed or uncommitted start compiles the step twice.

**Bounds.** Map unbounded variables through a sigmoid (`lo + (hi - lo) * jax.nn.sigmoid(z)`). Clipping kills gradients at the bounds.
- Pair the bounds with `bf.mesh(refine=bf.Refine(dilation={"WG": max_motion_um}))`, so the fixed grid is fine wherever edges may go. Dilation can also be per-vertex, in the layout of `stack.init_params()["vertices/WG"]`.
- The mesher never limits the parameters. Watch for vertices that leave the refined band with `coarse_vertices`.

**Phase objectives.** A mode's overall sign is arbitrary in each solve, so `angle(a_out / a_in)` is defined only modulo π. Use `jnp.angle(r**2 * jnp.exp(-2j * ref)) / 2`, as in `gradients.py`.

**Limitation.** With dispersive materials *and* mode or plane-wave sources, `apply_stack` cannot run inside `jax.jit` (fdtdx 0.6.2 builds a filter with NumPy). Plain `jax.grad` still works.

**Gradient check.** Compare against a central difference of the same discretized simulation (`h` = 0.05 µm for a radius, 0.005–0.01 µm for thickness or length in the tests).

---

## 7. Choosing the measurement

| You want | Use | Read it as |
|---|---|---|
| Power in one guided mode | `create_mode_overlap_detector_at(..., scaling_mode="pulse")`, with `direction="in"` on the input | `objects[n].compute_overlap(states[n])` gives one complex value per wave character; `T = abs(a_out / a_in) ** 2` |
| Total power through a plane (all modes) | `create_poynting_flux_detector_at(..., wavelength_um=λ)` (`reduce_volume=True` by default; `direction="in"` counts power entering as positive) | `np.asarray(states[n]["poynting_flux"]).sum()`, one row per step; take ratios |
| Field picture at λ | `fdtdx.PhasorDetector(..., partial_grid_shape=(None, None, 1))` + `same_size(volume, axes=(0, 1))` + `place_at_center(volume)`, or `create_phasor_detector_at` | `states[n]["phasor"][0, 0, :, :, :, 0]` gives `(components, nx, ny)` in the examples |
| Time movie | `FieldDetector` with an `OnOffSwitch(interval=…)` | viewer: `bf.view(stack).load_result(objects, arrays, config, stack=stack)` (notebook only) |
| Resonance f, Q, decay | `create_field_detector_at(..., wavelength_um=λ, switch=fdtdx.OnOffSwitch(start_time=after_pulse, interval=k))` + `bf.harminv_detector(det, arrays, f_min, f_max, components="Ey", Q_thresh=…)` | `res.to_modes()` (concrete); `res.select(f0)` returns a differentiable dict with `frequency`, `Q`, `inv_Q`, `amplitude`, … |
| Diffraction orders | `create_diffractive_detector_at(..., frequencies, orders)` | needs periodic boundaries: `bf.mesh(boundaries=…)` |

**harminv internals that matter**
- It needs ≥ 5 uniformly strided samples from a forward, non-inverse `FieldDetector`. The band must be below Nyquist, `0.5 / (stride · dt)`.
- `f_min` / `f_max` can be Hz or `WaveCharacter`s, and two WaveCharacters are auto-ordered.
- Defaults: `max_modes=16`, `Q_thresh=10.0`, `err_thresh=1e-2`.
- It runs in float64 when x64 is on, and warns otherwise.
- With both E and H components and no weights, H is scaled by η₀ internally.
- `HarminvResult` has a fixed shape (`max_modes`), with valid modes first. Mask unused slots with `.valid`.

---

## 8. Debugging playbook (symptom → most likely cause → fix)

| Symptom | Likely cause | Fix |
|---|---|---|
| T ≈ 1e±6 off, or absurd | a µm value passed to fdtdx | `WaveCharacter(wavelength=λ_um * 1e-6)`; `time` in s |
| T ≈ 0 | source inside the PML; plane clipped; input monitor with the default `"out"`; wrong mode spec | §3 inequalities; print `grid_slice`; `direction="in"`; same `mode_index` / `filter_pol` |
| T well below 1 for a straight guide | run too short | increase `time`; check at 1.5× |
| T slightly above 1 | discretization error in the mode amplitudes on a coarse grid | refine; report the convergence pair |
| T jumps when a parameter changes slightly | a monitor sits inside the moving geometry | move monitors into the leads (`port_offset=-0.5`) |
| `TypeError` from `get_simulation_config` | passed `dtype=` | set `dtype` on `StackDevice` |
| `ValueError: ... dtype ... simulation config` | config built by hand with another dtype, or grid edges in another dtype | build with `stack.get_simulation_config(grid=...)` |
| `RuntimeError` about 64-bit mode | float64 device with x64 off | `jax.config.update("jax_enable_x64", True)` before placing |
| `KeyError ... material ... not in materials` | a level in the box has no material | add it, or pass `levels=[...]` |
| `ValueError: Polygon topology of vertices/... changed` | the cell's vertex or polygon count depends on the parameter | fixed-structure cell, or `retopologize` |
| `grid= needs the grid of the device's own simulation volume` | the volume is not `stack.get_simulation_volume()` | use the device's volume, or drop `grid=` and give cells yourself |
| `Configured grid shape ... does not match` | manual `UniformGrid` with a box between two cell counts | adjust `boundary_offset` by a fraction of a cell, or use `bf.mesh` |
| `mesh: ... not a whole number of ... quanta` | `quantum=` given | `quantum=None` (the default) |
| `mesh: ... material without an automatic size` | a metal or conductive material | `Refine(material_cells={"metal": size_um})` |
| `StackDiagnosticWarning` narrow part or gap | the feature is under 1.5 cells | refine, raise `xy_subsamples`, or widen; `min_cell` may be capping |
| float32 vertex ULP warning | coordinates far from the origin | center the component near (0, 0), or use float64 |
| `RESOURCE_EXHAUSTED` / OOM | too many cells, or monitor/movie/checkpoint memory | §4 levers; `cuda_async` allocator; fewer checkpoints only if memory is in the forward states |
| first step slow, later steps fast | normal compilation | — |
| every step slow (recompiling) | changing shapes or dtypes; uncommitted or weakly typed inputs; arrays captured by closure | keep shapes fixed; `device_put` a float64 start; pass arrays as arguments |
| gradient wildly off FD | non-differentiable gdsfactoryx op (boolean, offset, merge); topology switch; FD step crosses a snapping or topology change | use differentiable constructions; `gf.snap.SNAP_ENABLED = False` for FD |
| harminv returns no valid modes | samples start before the pulse ends; band too wide or narrow; `Q_thresh` too high; too few samples | later `start_time`; adjust the band; lower `Q_thresh`; longer run |
| viewer blank in a script | the viewer is a notebook widget | use Jupyter/marimo, or plot with matplotlib |

---

## 9. Performance-tuning ladder (each rung keeps the physics; go in order and stop when it fits)

1. **Read `mesh.summary(time=T)`** and estimate VRAM (§4). Do a place-only dry run.
2. **Shrink the box.** Minimal margins that still satisfy §3. A compact PML (0.5 µm) where the extensions are short.
3. **`Refine.material_cells`** on full-wafer or uniform-cladding layers (~13 cells per wavelength).
4. **ppw 14** for exploration and early optimization. Final numbers come from ppw 20 and a check at a higher ppw.
5. **Lean monitors.** Planes not volumes, few wave characters, movie intervals, `reduce_volume=True` integrals.
6. **Gradients.** `GradientConfig(method="checkpointed", num_checkpoints≈30–35)`, `bf.jit`, compile once, reuse placed objects across steps.
7. **Keep `averaging="subpixel"`** unless the physics needs `"subpixel_full"`. Keep float32 unless float64 is needed (it doubles memory and is much slower on single-precision-tuned GPUs).
8. **Allocator.** `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`, set before importing jax, for very large grids.
9. **Previews only.** `StackOptions.fast()` (`"inverse"` averaging, no Yee staggering, no corner model, no diagnostics) for checking geometry and placement, never for reported results.

---

## 10. Minimal correct skeleton (reason from §1–§7 to fill it in)

```python
stack = bf.StackDevice(component, layer_stack, materials, background="clad", zrange_layers=["core"],
                       extend_ports=(("o1", EXT), ("o2", EXT)), boundary_offset=OFFSETS, name="stack")
mesh = bf.mesh(stack, WL_UM, ppw=PPW, refine=bf.Refine(...)); print(mesh.summary(time=TIME))
config = stack.get_simulation_config(time=TIME, grid=mesh.grid)  # + gradient_config for gradients
wave = fdtdx.WaveCharacter(wavelength=WL_UM * 1e-6)
volume, constraints = stack.get_simulation_volume(); objects = [volume, stack]
for obj, c in (stack.create_mode_plane_source_at("o1", PAD, wave_character=wave, temporal_profile=pulse,
                                                 port_offset=-1.0, name="source", grid=config, **te0),
               stack.create_mode_overlap_detector_at("o1", PAD, direction="in", wave_characters=(wave,),
                                                     scaling_mode="pulse", port_offset=-0.5, name="in", grid=config, **te0),
               stack.create_mode_overlap_detector_at("o2", PAD, wave_characters=(wave,), scaling_mode="pulse",
                                                     port_offset=-0.5, name="out", grid=config, **te0)):
    objects.append(obj); constraints += c
b, c = fdtdx.boundary_objects_from_config(mesh.boundary, volume); objects += b.values(); constraints += c
objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
arrays, objects, info = bf.apply_stack(arrays, objects, {**params, **bf.init_stack_params(objects)}, key)
_, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key)
T = abs(objects["out"].compute_overlap(arrays.detector_states["out"]) /
        objects["in"].compute_overlap(arrays.detector_states["in"])) ** 2
```

`info["stack_reapplied"]` and `info["stack_reused"]` name the objects whose modes were re-solved or served from the cache.

---

## 11. Not verified (say so instead of guessing)

- **fdtdx.** Signatures and defaults beyond the exact usages in the bifrost examples. The fdtdx source was not read; the docs mention fdtdx 0.6.2 and 0.6.3.
- **Phasor indexing.** What each leading index of `states[n]["phasor"]` means. Only `[0, 0, :, :, :, 0]` → `(components, nx, ny)` for a single-wave, one-cell-thick xy plane is shown.
- **Backward pass.** Peak memory as a function of `num_checkpoints`, and the multiplier of backward over forward VRAM.
- **Cell tracing.** Which built-in gdsfactoryx cells trace under `bf.jit`. Rely on §6's procedure.
- **`LayerLevel`.** The complete field list. Verified in bifrost use: `layer`, `zmin`, `thickness`, `material`, `mesh_order`, `sidewall_angle`, `width_to_z`, `bias`, `z_to_bias`, `derived_layer`, `background`. `LogicalLayer` booleans `| & - ^` and `.sized(...)` are documented in layer-stacks.md.
- **`StackDevice.interface_status(...)`.** Its argument list was not read.
- **Small grids.** Whether the per-million-cell VRAM rates hold below ~2 M cells; the floor dominates there.
