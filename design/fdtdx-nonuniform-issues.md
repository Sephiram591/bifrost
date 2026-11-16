# fdtdx issues found while running bifrost on non-uniform grids (drafts, not filed)

Drafts for upstream fdtdx (0.6.3, the version bifrost 0.2 pins), found while making bifrost's `StackDevice` run
on `RectilinearGrid`. Each says how it was established: **observed** (seen in a run here) or **read** (from
fdtdx's source, worked around in bifrost, not reproduced on its own). File them only after review.

## 1. A physical size becomes a cell count measured from the domain's lower face (read)

`fdtd/initialization.py:1265-1295` (`_real_length_to_grid_size`, used by `_resolve_static_shapes` at 1664 and by
size constraints at 1906) converts an object's `partial_real_shape` to cells with
`grid.length_to_cell_count(axis, length)`, which counts from `edges[0]`, the domain's lower face. Sizes are fixed
before positions, so on a graded axis an object gets as many cells as the same length has at the bottom of the
domain, not where it sits: a 0.13 um plane centered in fine cells can get one coarse cell's worth.

- **Suggested fix:** resolve sizes with positions (snap both faces at the anchor, e.g. with
  `bounds_for_anchor`'s interval search over physical faces), or document that non-uniform axes need
  `partial_grid_shape`.
- **bifrost's workaround:** `StackDevice.create_*_at(..., grid=)` snaps both faces of a port plane to edges at
  its own position and gives `partial_grid_shape` along graded axes.

## 2. Grid constraints and offsets raise on non-uniform grids (read)

`GridCoordinateConstraint` (`initialization.py:1747`) and nonzero `grid_margins` / `grid_offsets` / `grid_offset`
(1330-1340, called at 1812, 1886, 1939) raise on a `RectilinearGrid`, although an offset in cells is well defined
there (index arithmetic on the edges). Suggest accepting them as index offsets.

## 3. Plane-source energy normalization is not area weighted (read)

`objects/sources/linear_polarization.py:289-302`: with `normalize_by_energy=True` (the default) the energy is
re-normalized by an unweighted sum over cells, so on graded cells a source's power depends on the cell sizes
under it. The `normalize_by_energy=False` path normalizes by `sum(profile * cell_area)` on non-uniform grids
(432-452) but by `sum(profile)` in index units on uniform ones (455-464), a scale difference of about
`1 / (dx dy)` between the two (inferred, not run). Suggest weighting by the cells' face areas in both.

- **bifrost's resolution (mode ports):** `bf.ModePlaneSource` and `bf.ModeOverlapDetector` in `bifrost/stack.py`
  wrap the corresponding `fdtdx` classes to normalize graded port planes by `min_spacing**2`
  (`ModePlaneSource.apply` rescales `_E` and `_H` by `sqrt(mean(face_area) / min_spacing**2)`, and
  `ModeOverlapDetector.place_on_grid` sets `_cached_face_area_weights` to
  `face_area / sqrt(mean(face_area) * min_spacing**2)`), leaving uniform grids unchanged. The upstream `fdtdx`
  `LinearPolarization` note above remains open.

## 4. Expose the conductivity scale (read)

Conductivities are stored times `c * dt / courant_number` (`initialization.py:727-729`, inverted in
`detectors/mode.py:342`), which equals the spacing only on a uniform grid. A device that writes conductivity
itself (bifrost's `StackDevice`) has to copy the expression. Suggest a `SimulationConfig.conductivity_scale`
property.

## 5. Coordinate frames disagree for a grid not centered on 0 (read)

`_real_coord_to_edge_index` (`initialization.py:1298-1310`) uses absolute edge coordinates, while
`_center_to_bounds_for_grid` and the GDS helpers (e.g. `objects/static_material/gds_layer_stack.py:537`) measure
from the simulation volume's center. They agree only when the `RectilinearGrid` is centered on the origin
(bifrost's `StackDevice.rectilinear_grid` builds it so). Suggest one convention, documented.

## 6. Grid edges are float32 metres (observed)

`RectilinearGrid` stores its edges as `jnp.asarray(...)` in the default float32, so a 20 nm cell's width carries
a relative round-off of about 2e-6 (0.019999959 um observed for 0.02 um cells over a 1.6 um domain). A uniform
grid's `uniform_spacing` hides this by rounding to 1e-14 m; a non-uniform one exposes it to every consumer of
`cell_widths`. Suggest float64 edges for the setup-time geometry (they are not traced), or documenting it.
bifrost derives its frame unit from an evenly spaced axis's whole extent, rounded like `uniform_spacing`.

## 7. Smaller points (read)

- **Time step of a `QuasiUniformGrid`:** before placement, `config.time_step_duration` uses
  `courant_number * min_spacing / c` (`config.py:331-332`); after placement, with the resolved
  `RectilinearGrid`, the per-axis CFL formula (`core/grid.py:495-512`). The step count changes between the two;
  only the placed config's is the one run.
- **Constructor:** `RectilinearGrid` takes keyword arguments only (`x_edges=`, `y_edges=`, `z_edges=`); a positional
  call fails inside pytreeclass with an unhelpful `TypeError`.
- **`QuasiUniformGrid` cell counts:** `QuasiUniformGrid.resolve` appears to require even cell counts
  (`core/grid.py` around 190; not checked in a run).
