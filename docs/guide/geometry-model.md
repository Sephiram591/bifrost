# Geometry model

A `StackDevice` turns polygons and `LayerStack` levels into grid permittivity in stages: polygon preparation and derived boolean shapes, sidewall offsets, painting by `mesh_order` (using exact 2D/2.5D Green's-theorem grid-line integration with `voxelizer="greens"` by default, or signed-distance fields and a local corner model with `voxelizer="legacy"`), and Kottke subpixel permittivity averaging.

## Signed distances

Source layers are prepared with differentiable vertex arrays and exact seam handling (`bifrost.geometry.signed_distance`). Overlapping and abutting polygons merge without seams, and partly shared junction edges (such as a taper meeting the side of an MMI within `topology_tol`) are split so the shared segment becomes an exact seam.

## Derived levels

With `derived_shapes="exact"` (the default), any level defined by a boolean expression of layers (`and`, `or`, `not`, `xor`), or a single layer containing overlapping polygons, is evaluated from the exact polygons of that boolean (`bifrost.geometry.derive_polygons`). Those polygons are compiled once when building the device as a recipe whose vertices are either copies of source vertices or line-line intersections of source edges. Evaluating the recipe from traced vertex coordinates keeps derived shapes differentiable with respect to every source vertex. Where the boundary passes from one layer's edge to a collinear edge of another (abutting polygons sharing an edge), the recipe keeps one vertex from each layer joined at zero length so each moves with its own layer. Levels referencing the same boolean expression share one recipe, and shared edge crossings evaluate identically.

The level's offset (sidewall angle, `bias`, `z_to_bias`, and whole-expression sizing) then offsets that exact shape with round joins. This eliminates sub-voxel slivers where coincident edges meet (such as `WG - ETCH` ending flush with the waveguide), produces round exterior corners on `and` intersections, and avoids slits when shrinking an `or` union of abutting or overlapping polygons. Compiling the derived-shape recipes takes only a few seconds even on large layouts (about 3 s for 52.8k edges).

- **Frozen topology.** Like polygon seams, the derived-shape recipe assumes fixed combinatorial topology. `check_topology` and `pack` raise an error if moved vertices cause a part to appear or vanish, open a seam (for example, extending `WG` past `ETCH`), move a crossing off one of its edges, or break a collinear joint (`WG | SLAB` with equal tops after moving one edge). Call `retopologize` to rebuild the recipes for the new topology. Both the builder and the topology check tolerate parts narrower than `2 * topology_tol`. Between concrete checks, gradients follow the frozen recipe (with one-sided derivatives at exact coincidences); crossings of nearly parallel edges scale in sensitivity as `1 / sin(angle)`.
- **Sizing below the root.** Expressions such as `WG.sized(...) - ETCH` or anisotropic sizing cannot be represented by a single derived recipe; levels with inner sizing fall back to composite signed-distance evaluation and emit a `StackDiagnosticWarning` when the device is built. Sizing applied to the outer expression is combined directly into the level's offset with round joins.
- **`derived_shapes="composite"`** evaluates derived layers directly from the individual layers' signed distances (`or` takes `min`, `and` takes `max`, `not` takes `max(a, -b)`), then applies the level offset. This matches the exact boolean offset only where a single layer's edge is closest; coincident edges between layers produce slivers, and shrinking overlapping polygons leaves slits (see [Geometry diagnostics](diagnostics.md)).
- **Known limit.** Under `voxelizer="legacy"`, when two corners of a single derived level fall inside the same voxel (such as an etch tip touching a waveguide edge), the corner model blends back to a half-plane, whereas `derived_shapes="composite"` keeps one literal per layer. At zero bias both draw the same shape (with multi-corner voxels differing by up to 0.26 in worst-case fill); with nonzero bias, `derived_shapes="exact"` avoids the much larger sliver errors of composite evaluation, while isolated multi-corner voxels can differ by up to 0.12. Levels sharing a crossing with equal offsets (`WG - ETCH` and `WG & ETCH` with identical sidewall or bias) remain exact there.

## Sidewalls

At height `z`, a level offsets its boundary inward by `(z - z_ref) * tan(sidewall_angle) - bias - z_to_bias(z)`, where `z_ref = zmin + width_to_z * thickness`. Within each voxel, the offset is averaged over the vertical interval the level occupies inside that voxel (for a straight sidewall, the offset at the midpoint of that sub-interval) rather than at the voxel center. This vertical mean is exact across piecewise-linear `z_to_bias` knots and depends only on the level's own top and bottom surfaces. As a result, a top or bottom interface cutting through a voxel uses the sidewall position at its own height, and straight sidewalls conserve volume away from polygon corners even with `z_to_bias` knots inside the voxel.

- **Per-voxel vs total volume.** Because polygon area is quadratic in offset where an edge crosses a voxel corner within the voxel height, evaluating the cross-section at the mean offset differs slightly per voxel from the exact 3D integral—within 5e-3 of a voxel at 20° for axis-aligned edges and 2e-2 at 17° oblique angles—while canceling along the edge so total volume remains exact. Where a steep `z_to_bias` ramp has a knot inside a voxel (for instance, a 30 nm bias change over the top 6 nm of a 15° sidewall on 20 nm cells), single-voxel differences can reach 0.055 (axis-aligned) to 0.14 (17°), while layer volume remains within 4e-3 of a cell.
- **Near polygon corners.** Near corners, the level is evaluated at the two Gauss-Legendre heights of its sub-interval in the voxel and averaged. Because a sloped corner's area is quadratic in `z`, two-point Gauss-Legendre quadrature integrates the 3D volume to within 5e-5 of a voxel at ±20° (compared to 1.1e-2 error from evaluating only at the midpoint height), up to sub-voxel corner crossings or interior `z_to_bias` knots.
- **Continuity between straight edges and corners.** Under `voxelizer="legacy"`, straight sloped edges entering corner columns blend smoothly between the ordinary columns' mean-offset half-plane and the two-height corner evaluation (over a transition proportional to the seam size `<= tan(angle)/(4 * sqrt(3))`, about 2.5e-2 at 10°) so vertices bending away from collinearity or corners entering a voxel produce smooth derivatives matching finite differences. This blending runs level by level in `mesh_order`, keeping every voxel fill between the midpoint cross-section and the two-point 3D integral.

## Painting

Levels are painted in ascending `mesh_order` by clipping each voxel laterally and vertically between level surfaces, with `background` filling any remaining fraction. Away from corners, each level within a voxel acts as a half-plane defined by its local edge distance and normal.

On a [non-uniform grid](simulation-setup.md#graded-grids), voxels are the grid's own cells. Along a graded z axis, each voxel layer is clipped between its bottom and top cell edges, and staggered field components (`Ex` and `Ey` on z cell faces, and similarly for components staggered in x and y) average over the dual cells spanning neighboring cell centers. When x and y cell widths differ, each column is painted on its own physical cell rectangle so distances, normals, sidewall angles, and round joins remain Euclidean.

## Corners

Under the default `voxelizer="greens"`, polygon fill fractions and interface moments are integrated directly across every cell via Green's theorem with no corner-column limits or edge caps.

Under `voxelizer="legacy"`, voxel columns within reach of a polygon vertex or edge crossing use a local multi-edge model: each source layer is represented by the intersection or union of its two nearest edges' offset half-planes, combined through the level's boolean expression with round joins where an offset grows a convex corner or shrinks a reflex one. This handles arbitrary convex or reflex vertices, T-junction seams, hole cut lines, narrow strips or gaps between two edges, multi-layer corners (`WG - ETCH`), and overlapping levels sorted by `mesh_order`.

Where a third feature enters the voxel (edges shorter than roughly 1.4 cells plus twice the offset, dense curves, or sharp tips), the legacy corner model smoothly blends back to the half-plane using a single weight per voxel and z sub-interval (the minimum certification across `xy_subsamples` sub-squares) so half-plane errors cannot cancel unevenly. Known limits of `voxelizer="legacy"`:

- On sloped walls, the two-point corner rule integrates the 3D volume while the half-plane evaluates the midpoint cross-section; in blended voxels at steep angles (up to 25°), per-voxel fill can exceed the half-plane error by up to 1.6e-2.
- Without lateral sub-sampling, a voxel split vertically by multiple surfaces where only some sub-intervals are certified can deviate slightly from the single-half-plane fill.
- Hole cut lines and T-junctions split an outer edge; within roughly 1.4 cells plus twice the offset of that split point, the short edge segment acts as a third feature and reverts to the half-plane result (up to 2.3e-2 fill error at a shrunk corner).

In `voxelizer="legacy"`, `corner_lines` (default 2) sets the maximum number of split lines per voxel cell (cost scales roughly as `2^corner_lines + 1`; `0` disables the corner model), `corner_columns` caps the corner columns per sample grid (automatically sized during placement), and `xy_subsamples` refines dense curves and tips at roughly `s²` cost.

## Permittivity averaging

`averaging="subpixel"` (the default) applies Kottke subpixel averaging along the interface normal derived from each voxel's own geometry, preventing interfaces in adjacent voxels (such as a slab beneath a rib sidewall or a thin film next to a wall) from contaminating the normal. The permittivity variance in each voxel is decomposed into each z sub-interval's lateral flux (tilted out of plane by the sidewall angle) and its vertical profile (tilted by roughness slopes). Where opposite interfaces cancel flux inside a single voxel (such as a thin strip embedded in another material), the normal falls back smoothly to one-sided finite differences of the voxel-mean permittivity. This yields exact interface normals for any single planar wall, slanted wall, or multilayer film of arbitrary thickness and position, as well as 2D simulations.

With `xy_subsamples > 1`, sub-square moments are summed per z sub-interval before computing the normal, preserving exact normals across internal sub-square boundaries wherever the sub-squares share a consistent edge or certified corner. Setting `edge_width` different from 1 decouples the sub-square kernels, which causes interfaces in the gap between kernels to partially use one-sided differences. Gradients remain finite in `float32` even at low-contrast interfaces or corner slivers.

With `yee_staggered=True` (the default), each electric field component (`Ex`, `Ey`, `Ez`) is evaluated at its staggered Yee position with its own local normal. On a 130 × 90 × 45 test device, `subpixel_normal="voxel"` takes about 1.7× longer forward and 3.0× longer in gradient than finite-difference normals (`subpixel_normal="gradient"`). Raising `xy_subsamples=2` increases forward and gradient evaluation cost by roughly 4–5× (0.41 s forward and 2.8 s gradient vs 0.095 s and 0.59 s on an NVIDIA L40S, after ~40 s / ~80 s compilation).

`averaging="inverse"` computes the isotropic arithmetic mean of `1 / eps` (fdtdx's built-in convention), with no subpixel normal overhead. Electric conductivity is always averaged linearly.

## Full tensors

With `averaging="subpixel_full"`, materials may specify full anisotropic permittivity tensors: `fdtdx.Material(permittivity=...)` accepts a scalar, a diagonal 3-tuple, a nested 3 × 3 tuple, or a row-major 9-tuple. Permittivity tensors must be real, finite, symmetric, and positive definite (validated relative to tensor scale at device build and placement). A crystal with principal permittivities `e1, e2, e3` along the columns of a rotation matrix `R` has tensor `R @ diag(e1, e2, e3) @ R.T`. Because `"subpixel"` and `"inverse"` write only the three diagonal components, they reject materials with off-diagonal entries rather than silently dropping them.

`"subpixel_full"` writes, for each E component `c` at its Yee location, row `c` of the inverse of that position's effective 3 × 3 permittivity tensor according to the local voxel contents:

- **Uniform voxel:** the exact inverse of the material tensor.
- **Planar interface along normal `n`** (straight or sloped walls, films of any thickness, parallel multilayers, rough surfaces): Kottke's anisotropic laminate along `n`, `eps_eff = Q tau^-1(sum_m f_m tau(Q^T eps_m Q)) Q^T` (reducing for scalar materials to `<1/eps> n n^T + (I - n n^T) / <eps>`).
- **Junctions** (corners, walls meeting horizontal surfaces, three or more materials): periodic cell homogenization. The voxel is subdivided into `4 × 4 × 4` sub-cells (`junction_cells`), each assigned its local laminate (blending smoothly toward axis-weighted laminates where a corner inside a sub-cell makes the leading direction ambiguous), and homogenized via a periodic trilinear finite-element cell solve factored once with Cholesky for all three unit fields.

A voxel transitions smoothly into the junction closure as its interface direction tensor `D = sum_faces (area * contrast^2 * fade) N N^T` spreads away from a single direction (`r = 1 - lambda_1 / tr D`, blended smoothly across `r` from `0.002` to `0.02`). Every effective tensor is symmetric positive definite and bounded between the harmonic and arithmetic means of its constituent materials, with exact sub-cell volume fractions and implicit derivatives through the linear solve. Because the blend crosses from laminate to junction over roughly 0.03 cells when a corner enters a voxel, verify geometry derivatives there with small finite-difference steps (around 1e-3 cells; for example, on 20 nm cells, central differences agree to 2 % at 1e-4 µm and 0.1 % at 3e-5 µm). Sub-cell faces (quarters of a voxel) introduce piecewise-smooth derivative kinks analogous to voxel faces (see [What the derivatives are](gradients.md#what-the-derivatives-are)).

**Accuracy and stability.**

- **Homogenized tensor error:** Compared against an independent interface-fitted finite-element reference embedded in a 5 × 5 patch across L-corners, rotated squares, and acute triangles, the junction closure matches to 5 % median error (15 % at the 90th percentile), vs 31 % (42 %) for a single-normal laminate; most of the remaining 5 % discrepancy comes from periodic vs embedded boundary conditions. Planar interfaces converge at second order.
- **Rotated anisotropic slab (`benchmarks/subpixel/em_slab.py`):** For a 137 nm uniaxial crystal slab (9, 12, 14) rotated 25° about z with an xz entry of 0.8 in air at 1.55 µm, Jones transmission error against the exact Berreman transfer matrix (including polarization conversion `|t_yx| = 0.035`) is 1.3e-2, 5.0e-3, 9.9e-4, and 1.9e-4 at 40, 20, 10, and 5 nm (second-order convergence), compared to 0.12 down to 9.2e-3 (first-order) for cell-snapped staircase voxels. Energy conservation (`|r|^2 + |t|^2 = 1` within 2e-5), symmetry (6e-8), and reciprocity (4e-6) hold across resolutions.
- **Rotated rod grating (`benchmarks/subpixel/em_grating.py`):** For a grating of 0.3 × 0.2 µm rods rotated by 30° (0.6 µm period, 1.55 µm, in-plane E) against extrapolated MEEP subpixel smoothing, zeroth-order transmission error for `eps = 12` rods is 3.2e-3, 1.6e-3, 7.7e-4, and 6.5e-4 at 40, 20, 10, and 5 nm (vs 2.7e-2 to 5.0e-3 for `"subpixel"` and 9.4e-2 to 1.7e-2 for `"inverse"`). For rotated uniaxial crystal rods (polarization conversion 0.04), errors run from 4.2e-3 at 40 nm to 4.2e-4 at 5 nm, matching MEEP's continuum extrapolation within 3e-4.
- **Long-time stability (`benchmarks/subpixel/em_stability.py`):** In a closed periodic cavity containing sloped waveguides, a slab, a deep etch, a rotated pillar, and a pad with full anisotropic tensors in every material (2555 junction voxels), total field energy after a dipole pulse remains within 1 % over 52,450 time steps (2 ps) with no instability.

**Capacity and failure handling.** Under `voxelizer="greens"`, junction voxels execute dynamically with no static slot cap. Under `voxelizer="legacy"`, junction voxels and corner columns use static per-grid capacities (`junction_voxels`, sized at 1.5× the initial junctions plus 64, and `corner_columns`); overflowing those capacities or exceeding a 1e-4 solve residual triggers a coverage failure rather than a silent fallback. With `interface_failure="raise"` (the default), concrete calls raise an error and traced calls write NaN; with `"warn"`, a warning is emitted and the voxel keeps the laminate or half-plane tensor. `apply_stack` returns status counters in `info["stack_status"][name]`, and `device.interface_status(params)` inspects them without writing arrays.

Kottke averaging applies to `Material.permittivity` (`eps_inf`); conductivity and dispersive pole strengths are mixed linearly. Enabling `averaging="subpixel_full"` switches the entire FDTD simulation to fdtdx's 3 × 3 anisotropic update, which typically costs **4.5× to 6.0×** the runtime and **1.1× to 2.4×** the peak GPU memory of `"subpixel"` (see [Diagonal vs full-tensor averaging](adaptive-meshing.md#diagonal-vs-full-tensor-averaging)).

## Dispersive materials

Materials can carry an `fdtdx.DispersionModel` (Lorentz or Drude poles), in which case `Material.permittivity` sets `eps_inf`. All stack materials share unified pole slots—one per distinct `(omega_0, gamma)`—and each voxel mixes pole strengths linearly, giving exact susceptibility mixing: `eps(omega) = eps_inf,mix + sum_m w_m chi_m(omega)`. Because each pole slot adds memory and compute across the entire simulation volume, materials sharing pole parameters automatically share slots. In dispersive simulations with mode or plane-wave sources, `apply_stack` cannot be traced inside `jax.jit` (a limitation of fdtdx 0.6.2), though `jax.grad` still works.

## Surface roughness

`TopRoughness` displaces the top surface of a level with a Gaussian random field:

```python
stack = bf.StackDevice.from_layer_stack(
    ...,
    roughness=[bf.TopRoughness(level="core", rms=0.005, correlation_length=0.05)],
)
```

Dips are filled by levels deposited on that surface (with `fill_above=True`, their bottoms follow the rough surface downward) or by any overlapping level according to `mesh_order`. Pass `noise_key` to `apply_stack` to draw a fresh realization, or `apply_roughness=False` to evaluate the nominal unperturbed geometry.
