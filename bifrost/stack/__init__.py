"""Differentiable fdtdx object for a whole component with its LayerStack.

:class:`StackDevice` voxelizes every :class:`~gdsfactory.technology.LayerLevel` of
a LayerStack (logical and derived layers, ``mesh_order`` priorities, sidewall
angle, ``width_to_z``, ``bias``, ``z_to_bias``, background levels) into one
N-material region, differentiable with respect to the polygon vertices of every
source layer and to the bottom height (zmin), thickness and sidewall angle of
every level. The device
also builds its simulation volume (:meth:`StackDevice.get_simulation_volume`)
and places sources and detectors on the component's ports
(``StackDevice.create_<source or detector>_at``).

fdtdx ``Device`` objects only mix two materials and :func:`fdtdx.apply_params`
only knows about them, so the stack is applied with :func:`apply_stack`, which
writes the stack into the arrays and then calls ``fdtdx.apply_params`` (regular
devices and sources are updated as usual, after the stack).

Geometry (lengths in grid cells inside, gds/stack units are um):

- each source layer gets a 2D signed distance field (see
  :func:`bifrost.geometry.signed_distance`); a derived level (a
  boolean of layers, or one layer whose polygons overlap) gets the distance to
  the exact polygons of its boolean, built once as a recipe of copied layer
  vertices and crossings of layer edges and evaluated from the traced
  vertices (``derived_shapes="exact"``; :func:`derive_polygons`), or, with
  ``derived_shapes="composite"`` and where a level's sizing below its root
  cannot be drawn that way, the layers' distances combined with ``or = min``,
  ``and = max``, ``not = max(a, -b)``; layers whose material is applied to
  the inverse of their geometry (``inverted_layers``, default WAFER, which
  covers the whole plane when empty) and background levels are everywhere
  inside outside their polygons;
- a level at height z offsets its distance by
  ``(z - z_ref) * tan(sidewall_angle) - bias - z_to_bias(z)`` with
  ``z_ref = zmin + width_to_z * thickness``; along z each voxel is split at the
  (possibly rough) level surfaces, and on each part each level is the
  half-plane given by its distance and gradient at the voxel center, or, in
  columns where a polygon corner or a crossing of layer edges is within reach,
  the exact intersection / union of the nearby edges' offset half-planes, with
  round joins at grown corners, wherever that local model is certified
  (otherwise the half-plane; ``corner_lines``), offset for the part's mid
  height (so a surface cutting a voxel gets the sidewall at its own height);
  the levels are painted by increasing ``mesh_order`` (lower wins) by clipping
  the voxel square exactly; the background material gets the rest;
- :class:`TopRoughness` displaces the top surface of a level; dips are filled
  by the levels deposited on it (whose bottom follows) or by whatever level
  covers that height;
- the permittivity of a voxel is the Kottke subpixel average along the normal
  of the voxel's own model: the voxel's eps variance is split into each z
  sub-interval's lateral part, along its kernel-weighted interface flux
  (tilted out of the plane by the sidewall angle), and the vertical profile
  (along z, tilted by the roughness slopes), with the moments of all its
  ``xy_subsamples`` sub-squares added up; variance whose flux cancels
  inside the voxel falls back to one-sided differences of the mean eps_inf.
  So interfaces in neighbouring voxels do not mix, and films thinner than a
  voxel and 2D simulations work (``averaging="subpixel"``, the default; needs
  diagonally anisotropic arrays, which the device requests). Or the
  inverse-permittivity average of its materials (``averaging="inverse"``,
  fdtdx's convention). With 3-component arrays each E component is sampled at
  its own Yee position, with its own normal (``yee_staggered``). Electric
  conductivity is averaged linearly.

When it is placed, the device warns (:class:`StackDiagnosticWarning`) about
parts of a level and gaps narrower than about 1.5 cells, and, for levels drawn
with the composite model, about overlapping polygons on a layer that a level
shrinks (each polygon's distance is shrunk, not their union's);
:meth:`StackDevice.diagnose` returns the findings as data.

Dispersion (fdtdx >= 0.6.2): materials may carry a ``fdtdx.DispersionModel``
(Lorentz / Drude poles; ``Material.permittivity`` is then eps_inf). All stack
materials get the same pole slots (one per distinct (omega_0, gamma), see
:func:`bifrost.geometry.align_material_poles`), so fdtdx allocates
exactly those slots. In every voxel the slot constants c1, c2 are written and
the pole strengths are mixed, ``c3 = sum_m w_m c3_m``, which mixes the
susceptibilities exactly: ``eps(omega) = eps_inf_mix + sum_m w_m chi_m(omega)``
with ``eps_inf_mix`` from ``averaging``. (With ``"subpixel"`` the components
along an interface are then the exact arithmetic mean of eps(omega); fdtdx's
pole coefficients are isotropic, so the normal component keeps an arithmetic
mean of chi.) Each slot costs memory and time over the whole simulation, so
poles shared between materials share a slot.

Example:
    ```python
    import fdtdx, jax
    import gdsfactory as gf
    import bifrost as bf

    stack = bf.stack.StackDevice.from_layer_stack(
        my_circuit(gap=0.2), gf.gpdk.LAYER_STACK,
        materials={"si": fdtdx.Material(permittivity=12.1), "sio2": fdtdx.Material(permittivity=2.1)},
        background="sio2",
        zrange_layers=["core"],                   # z range of the box: the core ...
        extend_ports=(("o1", 1.0), ("o2", 1.0)),  # ... straight waveguides out of both ports ...
        boundary_offset=(0, 0, 1, -1, 1, -1),     # ... and 1 um more in y and z
        name="stack",
    )
    volume, constraints = stack.get_simulation_volume()
    objects = [volume, stack]
    wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
    for obj, c in (
        stack.create_mode_plane_source_at("o1", (0.5, 0.5), wave_character=wave, port_offset=-0.5),
        stack.create_mode_overlap_detector_at("o2", (0.5, 0.5), wave_characters=(wave,)),
    ):
        objects.append(obj)
        constraints += c
    objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
    params = {**params, **bf.stack.init_stack_params(objects)}

    def loss(gap):
        p = {**params, "stack": {**params["stack"], **objects["stack"].pack(my_circuit(gap=gap))}}
        arrays2, objects2, _ = bf.stack.apply_stack(arrays, objects, p, key)
        ...
    ```
"""

# ruff: noqa: F401

from ..timing import StepTimings, record_timings
from ._layers import (
    Array,
    Averaging,
    PortDirection,
    _BIG,
    _SELECT_SLACK,
    _ANISOTROPY_SENTINEL,
    _YEE_E_SHIFTS,
    _OFFDIAG_PAIRS,
    _CORNER_TOL,
    _PLANE_TOL,
    _CORNER_RAMP,
    _VOXEL_RAMP,
    _TIP_AREA,
    _MITRE_MIN,
    _STRAIGHT_WIDTH,
    _STRAIGHT_TOL,
    _PART_TINY,
    _GL3,
    _JUNCTION_LOW,
    _JUNCTION_HIGH,
    _SUBCELL_GAP,
    _NO_FACES,
    _FACE_THICKNESS,
    _ON_LINE,
    _PARALLEL,
    _T_FAR,
    _JUNCTION_CHUNK,
    _CERTIFIED,
    _CLOSURE_RESIDUAL,
    _HIGHEST,
    _KOTTKE_ETA,
    _KOTTKE_TINY,
    _FACE_TOL,
    _TOPOLOGY_DBU,
    _TOPOLOGY_EPS,
    _MIN_FEATURE,
    _CHECK_ANGLE,
    _NARROW_ANGLE,
    _CHECK_DBU,
    _CHECK_HEIGHTS,
    _ROUND_TOL,
    _Level,
    _Port,
    _topology_tol,
    _read_port,
    _FaceEnd,
    _PORT_CORNER,
    _face_frame,
    _face_point,
    _extension_rect,
    _face_intervals,
    _port_extensions,
    _face_spans,
    _span,
    _FRAME_RTOL,
    _MAX_ASPECT,
    _SampleAxis,
    _Frame,
    _grid_frame,
    _conductivity_scale,
    _sizing,
    _compile_layer,
    _compile_level,
    _add_grow,
    _simplify_expr,
    _expr_layers,
    _expr_can_draw,
    _expr_positive_layers,
    Field,
    _eval_expr,
    _TRUE,
    _FALSE,
    _literals,
    _eval_bool,
    _unit_normal,
    _layer_key,
    StackDiagnosticWarning,
    _kdb_region,
    _NO_LAYER,
    _DerivedShape,
    _root_grow,
    _with_root_grow,
    _inner_grows,
    _slot_expr,
    _shapely_polygons,
    _union_overlaps,
    _DERIVED_ERROR_WIDTH,
    _part_index,
    _shape_region,
    _recipe_region,
    _kdb_fold_rings,
    _expr_region,
    _ring_area,
    _derived_shapes,
    _primal,
)
from ._diagnostics import (
    StackDiagnostic,
    _wall_offset,
    _check_heights,
    _corners,
    _seeds,
    _fans,
    _round_offset,
    _tree_region,
    _clip_segments,
    _closest_points,
    _on_joins,
    _hull_key,
    _joined,
    _narrow_places,
    _overlaps,
)
from ._placement import (
    PLANE_PORTS,
    detector_plane_own_positions,
    Z_CENTER_LEVEL,
    _IN_PLANE,
    plane_extents,
    _SIMULATION_DTYPES,
    _simulation_dtype,
    _same_dtype,
    _complex_dtype,
    _plane_face_areas,
    _reference_face_area,
    ModePlaneSource,
    ModeOverlapDetector,
    _axis_direction,
    _given,
)
from ._corner_model import (
    _clip,
    _area,
    _Cells,
    _cells_expand,
    _cells_twice,
    _cells_take,
    _half_sides,
    _cell_corners,
    _cell_area,
    _cell_reach,
    _cell_standard,
    _cell_diagonal,
    _kernel_scales,
    _moments,
    _cross,
    _angle,
    _disk_area,
    _sidewall_shift,
    _paint_levels,
    _weigh_levels,
    _normal_tensor,
    _kernel_mass,
    _level_walls,
    _first_moment,
    _paint_moments,
    _line_range,
    _keep_side,
    _take_side,
    _take_interval,
    _line_partition,
    _line_faces,
    _face_lengths,
    _face_weights,
    _dyad,
    _interface_parts,
    _vertical_tensor,
    _CornerPlan,
    _LayerCorner,
    _CornerInputs,
    _Slot,
    _layer_corner,
    _same_plane,
    _literal_value,
    _assign,
    _round_tips,
    _corner_pass,
    _corner_fill,
    _mean_shift,
    _straight_seams,
    _CornerCells,
    _corner_cells,
    _corner_polygons,
    _corner_square,
    _corner_interfaces,
    _interface_corner_pass,
    _blend,
    _lateral_range,
    _corner_candidates,
)
from ._profiles import (
    _point_slopes,
    _split_layer,
    _paint_layer,
    _gauss_heights,
    _composite,
    _lateral_samples,
    _composite_samples,
    _interface_composite,
    TopRoughness,
    _roughness_reach,
    _anisotropy,
    _center_distances,
    _normal_squared,
    _normal_dyad,
)
from ._junctions import (
    _is_finite_spd,
    _safe_symmetric_inverse,
    _junction_cells,
    _match_fractions,
    _junction_tensor,
    _LayerGeometry,
    _JunctionInputs,
    _Interfaces,
)
from ._device import (
    _check_nonbool_int,
    _validate_stack_options,
    StackOptions,
    StackDevice,
    _check_materials,
    _drop_empty_pole_slots,
    pole_slots,
    layer_stack_wafer_layer,
    _PML_COEFFICIENTS,
    _cast_pml_coefficients,
    stack_devices,
    init_stack_params,
    _interface_status_core,
    apply_stack,
    _material_slices,
    _hashable_config_value,
    _mode_solver_config_key,
    _slice_cache_key,
    _PENDING_REAPPLY,
    _reapply_cache,
    _boxes_overlap,
    _reapply_stack_objects,
)

__all__ = [
    "ModeOverlapDetector",
    "ModePlaneSource",
    "PLANE_PORTS",
    "detector_plane_own_positions",
    "Z_CENTER_LEVEL",
    "StackDevice",
    "StackDiagnostic",
    "StackDiagnosticWarning",
    "StackOptions",
    "TopRoughness",
    "apply_stack",
    "init_stack_params",
    "record_timings",
    "StepTimings",
]
