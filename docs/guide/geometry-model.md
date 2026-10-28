# Geometry model

- **Signed distances.** Each source layer gets a 2D signed distance field to the
  union of its polygons, differentiable in the vertices
  (`bifrost.geometry.signed_distance`). Overlapping and abutting
  polygons merge without seams, and partly shared junction edges (a taper on the
  side of an MMI, its corners within `topology_tol` of the side) are made
  exactly shared.
- **Derived levels** (`derived_shapes="exact"`, the default). A level whose
  layer is a boolean of layers (`and`, `or`, `not`, `xor`), or one layer whose
  polygons overlap, is drawn from the exact polygons of that boolean, with its
  own signed distance. They are built once, when the device is built
  (`bifrost.geometry.derive_polygons`), as a recipe in which every
  vertex is a copy of a layer vertex or the crossing of the lines of two layer
  edges, and evaluated from the (traced) vertices, so the shape stays
  differentiable in every source vertex. Where the boundary passes from one
  layer's edge to a collinear edge of another (abutting polygons, an edge
  shared by two layers), it gets one vertex from each, joined at zero length,
  so each moves with its own layer. Levels that draw the same boolean share
  one shape, and crossings shared by several shapes evaluate to the same
  numbers. The level's offset (sidewall, bias, `z_to_bias`, and the sizing of
  the whole expression) then offsets that shape with round joins: no slivers
  where edges of its layers coincide (`WG - ETCH` with the etch ending with
  the waveguide), round grown corners of an `and`, and no slits where an `or`
  of abutting or overlapping parts shrinks. On a test device this removed a
  sliver of 0.88 Si at the etched end and halved the corner columns.
  - The recipe is frozen topology, like the polygons' seams:
    `check_topology` (and `pack`) report a shape the new vertices no longer
    draw (a part that appeared or vanished, a seam that opened: e.g. the
    waveguide now reaching past the etch's end), a crossing that left one
    of its edges, and a joint of collinear edges that opened into a step
    (`WG | SLAB` with equal tops, one top moved); `retopologize` builds the
    shapes again. The build and the check accept the same differences from
    the boolean: parts narrower than twice the topology tolerance. Between those
    concrete checks, gradients follow the frozen recipe; at exact
    coincidences they are one-sided. Crossings of nearly parallel edges are
    ill-conditioned (about 1/sin of their angle).
  - Sizing below the expression's root (`WG.sized(...) - ETCH`, anisotropic
    sizing) cannot be drawn this way: such a level keeps the composite model
    below, with a `StackDiagnosticWarning` when the device is built. Sizing of
    the whole expression is part of the level's offset (round joins, not
    KLayout's corner modes).
  - `derived_shapes="composite"` restores the earlier model: derived layers
    combine the layers' fields (`or` takes the min, `and` the max, `not` is
    `max(a, -b)`), offset as a whole. That is the offset of the derived shape
    only where one layer's edge is nearest; where layers' edges coincide it
    draws slivers, and where a level shrinks a layer, overlapping polygons are
    shrunk one by one (see [Geometry diagnostics](diagnostics.md)).
  - Known limit: where two corners of one derived level share a voxel (an
    etch tip on the waveguide's edge), its corner model keeps the half-plane,
    while the composite model, with one literal per layer, is exact there. At
    bias 0 both draw the same shape, and on random stars the exact mode has a
    summed fill error of 0.62 against 0.35 (worst voxel 0.26 against 0); with
    nonzero biases it is far better overall (0.9 against 6.5) with single
    voxels up to 0.12 worse. It is never worse than its own half-plane.
    Levels sharing a crossing with equal offsets (`WG - ETCH` and
    `WG & ETCH` with one sidewall or bias) are exact there: each rounds its
    own tip, in the cone of its own edge normals (before: 0.20-0.25 of a voxel
    at the crossing).
  - Building the shapes costs seconds for tens of thousands of edges (52.8k
    edges: 3 s; the first version took 85 s and 7.7 GB).
- **Sidewalls.** A level at height `z` offsets its distance by
  `(z - z_ref)·tan(sidewall_angle) - bias - z_to_bias(z)`, with
  `z_ref = zmin + width_to_z·thickness`. In each voxel a level takes the mean
  of this offset over the part of the voxel it fills along z (for a straight
  sidewall, the offset at the middle of that part), not the offset at the voxel
  center. The mean is exact for a piecewise-linear `z_to_bias` (its knots split
  the part), and only the level's own surfaces enter it: another level's
  surface cutting the voxel changes nothing. A top or bottom surface that cuts
  a voxel therefore gets the sidewall position at its own height, and a
  sidewall gets its exact volume away from polygon corners, also with a
  `z_to_bias` knot inside a voxel.
  - Per voxel, the area at the mean offset differs from the 3D volume where the
    edge crosses a corner of the voxel's square within the voxel's height (the
    area is quadratic in the offset there). Audits measured up to 5e-3 of a
    voxel at 20° for axis-aligned edges and 2e-2 for edges at 17°; these errors
    cancel along the edge, so the row volume stays exact. This bound does not
    hold where a steep `z_to_bias` ramp has a knot inside the voxel (the offset
    then sweeps over a cell or more within one voxel height): a 30 nm bias
    change over the top 6 nm of a 15° sidewall (`z_to_bias=((0, 0.97, 1),
    (0, 0, -0.03))`, 20 nm voxels) gave up to 0.055 of a voxel axis-aligned
    and 0.14 at 17° (the old code: 0.27), while the layer's volume stays within
    4e-3 of a cell.
  - Near polygon corners, the corner model (below) paints each level with its
    offsets at the two Gauss-Legendre heights of its part of the voxel and
    takes the mean (again independent of other levels' surfaces). A sloped
    corner's area is quadratic in z, so this is the exact 3D volume, except
    where a vertex or edge crosses a corner or side of the voxel's square
    within the part, or a `z_to_bias` knot lies inside it (a residual of the
    same order as on straight edges). At ±20°, a corner on a voxel center is now
    within 5e-5 of the 3D fill (the cross-section at the middle misses
    `tan²(angle)/12`, 1.1e-2).
  - Where the edge is straight within a corner column's square (a junction of
    two polygons, a port extension, a collinear vertex, or the straight sides
    next to a corner), the two heights alone would differ from the ordinary
    columns' mean offset where the wall crosses a side of the square within
    the voxel (by up to `tan(angle)/(4√3)`, 2.5e-2 of a voxel at 10°), a seam
    along the waveguide. So the corner columns add that seam (the half-plane
    at the mean offset less the half-plane at the two heights): where the
    corner model equals the half-plane at both heights, they get exactly the
    ordinary columns' result, and a straight edge is painted the same in
    every column. The seam fades out smoothly (and differentiably) as the
    corner model leaves the half-plane, over a difference of half the seam's
    size, so a vertex bending off collinear or a corner entering a voxel's
    square moves the fill continuously, with gradients matching finite
    differences (at 10-20°, the slope is at most 2 per cell of motion, where
    the corner first enters the square at one of the two heights and the
    seam fades out, and about 0.5 past the square's side), and a real corner
    keeps the 2-point value. Vertical walls have no seam. The seam is
    added level by level in `mesh_order`, on the union of the levels painted
    up to each, so a level's fill depends only on its own and the earlier
    levels' geometry: a later level's corner in the square, or its surface
    cutting the voxel, changes nothing of it (the straight edge of a sloped
    core next to a corner of a lower-priority level keeps the ordinary
    columns' fill). An earlier level's corner in the square does bend that
    union, so there a later level's straight sloped edge keeps the 2-point
    value (up to `tan(angle)/(4√3)` off the ordinary columns along it).
    Every corner-column voxel thus lies between the cross-section at the
    middle and the 2-point 3D value, to 5e-4 of a voxel where a corner just
    enters the square. On a polygon with corners of all kinds at ±20°, the summed error
    against the 3D fill is 0.93-0.972 of the cross-section's at the middle;
    the largest error per voxel is the straight edges' 2e-2, as in the
    ordinary columns.
- **Painting.** Levels are painted by increasing `mesh_order` by clipping each
  voxel exactly, both laterally and along z between the level surfaces. The
  background gets the rest. Laterally, each level is one half-plane per voxel,
  taken from its distance and gradient at the voxel center. That is exact
  wherever at most one edge of the level crosses the voxel.
- **Corners.** In voxel columns where a polygon corner or a crossing of layer
  edges is within reach, the voxel is painted with the exact local geometry
  instead: each source layer is the intersection or union of its two nearest
  edges' half-planes, combined through the level's layer expression, offset by
  the sidewall, bias and sizing, with round joins where an offset grows a
  convex corner or shrinks a reflex one. This is exact for:
  - any vertex of a source layer, convex or reflex, at any angle, inside or
    just outside the voxel, also across T-junction seams and hole cut lines
    (except near where they split an edge, see below);
  - crossings of edges of overlapping polygons of one layer, and strips or gaps
    between two edges;
  - corners of derived levels where edges of different layers meet, such as
    `WG - ETCH` with the etch edge across the waveguide (with exact derived
    shapes these are ordinary corners of the level's own polygon, with round
    joins where it grows);
  - several levels in one voxel (seams, nesting, `mesh_order`).

  The local model is checked in every voxel. Where a third feature comes within
  reach (edges shorter than about 1.4 cells plus twice the offset, dense curves,
  narrow tips), the result blends back to the half-plane. It is therefore never
  further from the exact fill than the half-plane alone, per z sub-interval
  and, with `xy_subsamples > 1`, for the voxel's average too: all sub-squares
  of a voxel are blended by one weight, the least certain of them, so their
  half-plane errors can no longer cancel unevenly (a weight per sub-square left
  voxels up to 6.8e-3 further off). That costs accuracy where only some
  sub-squares are certified (the largest error in one random test went from
  1.2e-2 to 2.2e-2), and with sub-samples the corner columns reach 0.05 sides
  further. Known limits:
  - With sloped walls, "exact" is the 3D volume where the 2-point rule is
    exact (see Sidewalls), and the half-plane is exact for the cross-section
    at the mean offset only. Against the 3D fill, the never-worse comparison
    then holds up to those residuals: in random tests at up to 25°, a few
    voxels per layer came out further off than the half-plane (up to 1.6e-2,
    against 3.6e-2 before the 2-point rule; on the corner test polygon at
    ±20°, at most 2 voxels per layer by more than 1e-4).
  - Without sub-samples, never-worse holds per z sub-interval; a voxel split
    by level surfaces into sub-intervals where only some are certified can in
    principle still be further off than the half-plane.
  - A hole's cut line ends on an outer edge and splits it, and so does a
    T-junction. Within about 1.4 cells plus twice the offset of a corner of
    that edge, the short piece counts as a third feature, and that corner
    keeps the half-plane result (2.3e-2 off in a test ring shrunk by 0.6
    cells). Fixing this needs a local model of three edges (the corner and the
    split point), or treating collinear pieces as one edge, which the 1 nm
    snapping of a cut point (not exactly collinear) makes inexact.

  `corner_lines` (default 2) is how many lines may split a voxel into cells in
  which every level is a single half-plane. The cost grows about as
  `2^corner_lines + 1`, and 0 turns the corner model off. `corner_columns` is
  the number of columns per sample grid painted this way. It is sized when the
  device is placed, and further columns keep the half-plane result. On a
  130 x 90 x 45 test device, the corner model makes `material_weights` about
  1.5x slower, and its gradient about 1.9x slower. `xy_subsamples` still
  refines the lateral fill at tips, short edges and dense curves, at a cost of
  roughly `s²`.
- **Permittivity averaging.** `averaging="subpixel"` (default) is Kottke subpixel
  averaging along the normal of each voxel's own model, so interfaces in
  neighbouring voxels (a slab under a rib's sidewall, a film next to a wall) do
  not mix. The voxel's ε variance is split into each z sub-interval's
  lateral part, along its kernel-weighted interface flux (tilted out of the
  plane by the sidewall angle), and the vertical profile (along z, tilted by the
  roughness slopes). Variance whose flux cancels inside the voxel (a strip of
  one level between two parts of another) falls back to one-sided differences
  of the mean ε∞. The normal is exact for any single planar interface (slanted,
  horizontal or leaning) and for films of any thickness and position. At
  corners it is a mix of the edges in the voxel. It also handles 2D
  simulations. With `xy_subsamples`, the sub-squares' moments are added up
  per z sub-interval before the normal is taken, so the variance between
  sub-squares counts and a plane on the face between two of them keeps its
  exact normal. The normal stays continuous as an edge crosses that face
  wherever the sub-squares' models agree on the edge: straight edges, and
  corners the corner model certifies. Near a corner painted with half-planes
  (`corner_lines=0`, or beyond the corner model's certified reach), it can
  jump there, by up to about 0.25 in 1/ε for Si in air. The half-plane model
  already jumps by as much without sub-samples (about 0.2 where a voxel
  center crosses the bisector of a corner). With `edge_width` other than 1,
  the sub-squares do not tile the voxel and each gets its own kernel, which
  vanishes on its own sides. A wall in a gap between them, or close to their
  sides, then partly falls back to the one-sided differences. The gradients
  stay finite in float32, also for low-contrast levels (an oxide next to
  silicon) at corner slivers. With `yee_staggered=True` (default), each E
  component is sampled at its own Yee position, with its own normal. On a
  130 x 90 x 45 test device, this normal makes the subpixel `apply_stack`
  about 1.7x slower, and its gradient about 3.0x slower, than finite
  differences of the mean ε would (`subpixel_normal="gradient"`).
  With `xy_subsamples=2` it takes about 4x (forward) and 5x (gradient) as
  long as with 1: 0.41 s and 2.8 s instead of 0.095 s and 0.59 s on an L40S
  GPU, after compiling for about 40 s and 80 s.
  `averaging="inverse"` is fdtdx's inverse-permittivity average; it and
  `material_weights` cost nothing extra. Conductivity is averaged linearly.
- **Full tensors (`averaging="subpixel_full"`).** Materials may be
  anisotropic: `fdtdx.Material(permittivity=...)` as a scalar, a diagonal
  3-tuple, a nested 3 x 3 tuple or a row-major 9-tuple; it must be real,
  finite, symmetric and positive definite (checked, relative to the tensor's
  scale, when the device is built and placed). A crystal with principal
  values `e1, e2, e3` along the columns of a rotation `R` is
  `R @ diag(e1, e2, e3) @ R.T`. `"subpixel"` and `"inverse"` write each E
  component's diagonal entry only, so they reject materials with off-diagonal
  entries rather than drop them. `"subpixel_full"` writes, for every E
  component c at its own Yee position, row c of the inverse of that
  position's complete effective tensor (all three entries; fdtdx applies
  them to the other components averaged onto that position), by what the
  voxel holds there:
  - one material: the exact inverse of its tensor;
  - interfaces along one normal n (a wall, a sloped wall, a film of any
    thickness, parallel multilayers, a rough surface): Kottke's anisotropic
    laminate along n, `eps_eff = Q tau^-1(sum_m f_m tau(Q^T eps_m Q)) Q^T`,
    exact for parallel layers of any number and order (with scalar
    materials it is the usual `<1/eps> n n^T + (I - n n^T) / <eps>` for the
    inverse);
  - a junction (a corner, a wall meeting a surface, three materials): the
    closure. The voxel's own model is cut into 4 x 4 x 4 sub-cells
    (`junction_cells`), each the laminate of its own faces, and homogenized
    by a periodic cell problem (trilinear finite elements, one Cholesky
    factorization for the three fields, the energy of the computed fields).

  A voxel is a junction where its interface direction tensor
  `D = sum_faces (area contrast^2 fade) N N^T` spreads off its leading
  direction: `r = 1 - lambda_1 / tr D`; the closure is blended in smoothly
  between r = 0.002 and 0.02, so a corner entering a voxel changes the
  tensor continuously. Every written tensor is symmetric positive definite
  and between the harmonic and arithmetic means of the voxel's materials
  (the sub-cells add up to the voxel's fractions exactly, also in corner
  columns; with `xy_subsamples` > 1 they are painted with each sample's
  half-planes and then made to hold the voxel's materials, which blend the
  corner model in), and the closure's derivative is the solve's implicit
  one. The blend reaches the closure within about 0.03 cells of a corner
  entering a voxel, so derivatives with respect to the geometry are
  continuous but change quickly there: check them with finite-difference steps well below
  that (about 1e-3 cells; moving a corner's vertex on 20 nm cells, central
  differences are 2 % off at 1e-4 um and 0.1 % at 3e-5 um). A surface on a
  face of the closure's sub-cells (a quarter of a voxel) is a kink, like one
  on a voxel face (see "Gradients through the simulation").
  Against an independent finite-element reference of the exact geometry
  (the voxel embedded in a 5 x 5 patch, on junction voxels of an L shape, a
  rotated square and an acute triangle), the closure's tensor is off by 5 %
  (median; 15 % at the 90th percentile), where the laminate along the
  faces' leading direction is off by 31 % (42 %); most of the closure's
  error is its periodic boundary condition (the exact periodic tensor is
  itself 5.6 % from the embedded one), its sub-cells add 2.2 %. Planar
  interfaces (walls, films, multilayers) converge at second order (below);
  at sharp corners the field is singular and no voxel tensor makes them so.
  In fdtdx, a 137 nm slab of a uniaxial crystal (9, 12, 14) rotated by 25
  degrees about z, with an xz entry of 0.8, in air at 1.55 um, at normal
  incidence, against the Berreman transfer matrix
  (`benchmarks/subpixel/em_slab.py`): the error of its Jones
  transmission (polarization conversion |t_yx| = 0.035 included) is
  1.3e-2, 5.0e-3, 9.9e-4 and 1.9e-4 at 40, 20, 10 and 5 nm (second order;
  the worst of three positions of the slab against the grid), where the
  same slab snapped to the cells (what voxel-center sampling paints) is off
  by 0.12, 3.6e-2, 1.7e-2 and 9.2e-3 (first order). |r|^2 + |t|^2 = 1
  within 2e-5, the Jones matrix is symmetric within 6e-8 and the wave sent
  the other way gets its transpose within 4e-6 (reciprocity). For a
  diagonal tensor the result is that of `"subpixel"` (same errors), and
  `"inverse"` converges at first order. At corners
  (`benchmarks/subpixel/em_grating.py`: a grating of 0.3 x 0.2 um
  rods rotated by 30 degrees, period 0.6 um, in-plane E, zeroth-order
  transmission against meep's subpixel smoothing extrapolated from 50 to 800
  pixels/um, itself uncertain to about 2e-3), the error is 3.2e-3, 1.6e-3,
  7.7e-4 and 6.5e-4 at 40, 20, 10 and 5 nm for eps = 12 rods, against
  7.5e-3, 4.4e-3, 2.0e-3 and 7.4e-4 for the previous `"subpixel_full"`
  (one averaged normal per voxel), 2.7e-2 to 5.0e-3 for `"subpixel"` and
  9.4e-2 to 1.7e-2 for `"inverse"`. For rods of the rotated uniaxial crystal
  above (polarization conversion 0.04), the error is 4.2e-3, 1.8e-3,
  3.8e-4 and 4.2e-4, and the two codes' extrapolations agree within 3e-4.
  fdtdx's fully anisotropic update stays stable with these tensors: in a
  closed periodic box (a waveguide with sloped walls, slab, etch, a rotated
  pillar and a pad, every material a different full tensor; 2555 junction
  voxels), the total energy after a dipole pulse stays within 1 % for 52450
  time steps (2 ps), without growth
  (`benchmarks/subpixel/em_stability.py`).

  Junction voxels take slots of a static capacity per Yee position,
  `junction_voxels`, sized when the device is placed (1.5 x the junctions at
  the initial parameters, plus 64). Junctions beyond it, corner columns
  beyond `corner_columns` and a closure residual above 1e-4 are coverage
  failures, never a silent one-plane fallback: with
  `interface_failure="raise"` (default) `write` raises where the values are
  concrete and writes NaN under tracing; with `"warn"` it warns and keeps the
  laminate or half-plane result there. Voxels of corner columns whose corner
  model is not certified (more edges or tips than it resolves) are painted
  partly with one half-plane per level; they are counted, not failed, and a
  placed device with `diagnostics` warns about them (and about
  `corner_lines=0`, which paints every corner that way). `apply_stack`
  returns the counts in `info["stack_status"][name]`, and
  `device.interface_status(params)` computes them without writing (check
  them at the end of an optimization). The averaging applies to
  `Material.permittivity` (eps_inf); conductivity and pole strengths are
  mixed linearly, as with the other averagings. The cost is the whole
  simulation's: every voxel takes fdtdx's 3 x 3 update (measured with the
  earlier scalar version: 4.9x per time step and 1.8x the GPU memory on the
  elevator coupler's 33M cells), so it is opt-in.

### Dispersive materials

Materials can carry an `fdtdx.DispersionModel` (Lorentz or Drude poles), in
which case `Material.permittivity` is ε∞. All stack materials share the same
pole slots, one per distinct `(omega_0, gamma)`. Each voxel then mixes the pole
strengths, which mixes the susceptibilities exactly:
`ε(ω) = ε∞,mix + Σ w_m χ_m(ω)`. Each slot costs memory and time over the whole
simulation, so materials that share poles also share slots. With mode or
plane-wave sources in a dispersive simulation, `apply_stack` can't run inside
`jax.jit` (a limitation of fdtdx 0.6.2), but `jax.grad` works.

### Surface roughness

`TopRoughness` displaces the top surface of one level with a Gaussian random
field:

```python

stack = bf.StackDevice.from_layer_stack(
    ...,
    roughness=[bf.TopRoughness(level="core", rms=0.005, correlation_length=0.05)],
)
```

Dips are filled by the levels deposited on that surface. With
`fill_above=True`, their bottoms follow the surface down. Any other level
covering that height also fills them, by `mesh_order`. Pass `noise_key` to
`apply_stack` to draw a new realization, or `apply_roughness=False` for the
nominal geometry.
