"""StackOptions, StackDevice, apply_stack, and bounded mode-solver caching."""

import sys
import contextvars
import dataclasses
import hashlib
import math
import warnings
import fdtdx
import jax
import jax.numpy as jnp
import numpy as np
from collections.abc import Callable, Sequence
from enum import Enum
from functools import lru_cache, partial
from typing import Any, Literal, Self
from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import autoinit, field, frozen_field
from fdtdx.dispersion import DispersionModel, Pole, compute_pole_coefficients
from fdtdx.fdtd.container import ObjectContainer
from fdtdx.materials import (
    compute_allowed_dispersive_coefficients,
    compute_allowed_electric_conductivities,
    compute_allowed_permittivities,
    compute_ordered_names,
)
from fdtdx.objects.detectors.diffractive import DiffractiveDetector
from fdtdx.objects.object import PositionConstraint, SimulationObject
from fdtdx.objects.sources.source import HardConstantAmplitudePlanceSource
from fdtdx.objects.static_material.static import StaticMultiMaterialObject
from fdtdx.typing import INVALID_SLICE_TUPLE_3D
from ..subpixel import (
    check_permittivity,
    from_six,
    is_diagonal,
    junction_weight,
    laminate,
    material_tensor,
    material_tensors,
    periodic_cell_tensor,
    to_six,
)
from ..timing import lap, timed, timed_step
from ..geometry import (
    _boundary_links,
    _edge_topology,
    align_material_poles,
    crossing_changes,
    has_tracers,
    joint_changes,
    nearest_edges,
    pack_polygons,
    recipe_rings,
    run_recipe,
    signed_distance,
    split_t_junctions,
    to_numpy,
    topology_changes,
    warn_reversible_losses,
)
from ._layers import (
    Array,
    Averaging,
    Field,
    PortDirection,
    StackDiagnosticWarning,
    _ANISOTROPY_SENTINEL,
    _BIG,
    _CHECK_ANGLE,
    _CHECK_DBU,
    _CLOSURE_RESIDUAL,
    _CORNER_TOL,
    _DERIVED_ERROR_WIDTH,
    _DerivedShape,
    _FALSE,
    _FRAME_RTOL,
    _FaceEnd,
    _Frame,
    _HIGHEST,
    _JUNCTION_CHUNK,
    _JUNCTION_HIGH,
    _JUNCTION_LOW,
    _Level,
    _MAX_ASPECT,
    _MIN_FEATURE,
    _NO_LAYER,
    _PORT_CORNER,
    _Port,
    _ROUND_TOL,
    _SELECT_SLACK,
    _SampleAxis,
    _TRUE,
    _VOXEL_RAMP,
    _YEE_E_SHIFTS,
    _compile_layer,
    _compile_level,
    _conductivity_scale,
    _derived_shapes,
    _eval_expr,
    _expr_can_draw,
    _expr_layers,
    _expr_positive_layers,
    _expr_region,
    _extension_rect,
    _face_intervals,
    _face_spans,
    _grid_frame,
    _kdb_fold_rings,
    _kdb_region,
    _layer_key,
    _literals,
    _part_index,
    _port_extensions,
    _primal,
    _read_port,
    _root_grow,
    _shape_region,
    _shapely_polygons,
    _span,
    _topology_tol,
    _unit_normal,
    _with_root_grow,
)
from ._diagnostics import (
    StackDiagnostic,
    _check_heights,
    _corners,
    _narrow_places,
    _overlaps,
    _seeds,
    _tree_region,
    _wall_offset,
)
from ._placement import (
    ModeOverlapDetector,
    ModePlaneSource,
    PLANE_PORTS,
    Z_CENTER_LEVEL,
    _IN_PLANE,
    _axis_direction,
    _complex_dtype,
    _given,
    _same_dtype,
    _simulation_dtype,
    detector_plane_own_positions,
    plane_extents,
)
from ._corners import (
    _Cells,
    _CornerInputs,
    _CornerPlan,
    _cell_diagonal,
    _cells_take,
    _corner_candidates,
    _lateral_range,
    _layer_corner,
    _sidewall_shift,
)
from ._profiles import (
    TopRoughness,
    _anisotropy,
    _center_distances,
    _composite,
    _composite_samples,
    _interface_composite,
    _lateral_samples,
    _normal_squared,
    _point_slopes,
    _roughness_reach,
)
from ._junctions import (
    _Interfaces,
    _JunctionInputs,
    _LayerGeometry,
    _is_finite_spd,
    _junction_cells,
    _safe_symmetric_inverse,
)


def _check_nonbool_int(name: str, value: Any) -> int:
    """Validates that ``value`` is a non-boolean integer and returns it as Python int."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _validate_stack_options(
    averaging: Any,
    yee_staggered: Any,
    corner_lines: Any,
    xy_subsamples: Any,
    derived_shapes: Any,
    diagnostics: Any,
    junction_cells: Any,
    interface_failure: Any,
) -> None:
    """Validates shared option fields of :class:`StackOptions` and :class:`StackDevice`."""
    if averaging not in ("inverse", "subpixel", "subpixel_full"):
        raise ValueError(f"averaging must be 'inverse', 'subpixel', or 'subpixel_full', got {averaging!r}")
    if not isinstance(yee_staggered, bool):
        raise TypeError(f"yee_staggered must be a bool, got {yee_staggered!r}")
    if _check_nonbool_int("corner_lines", corner_lines) < 0:
        raise ValueError(f"corner_lines must be >= 0, got {corner_lines!r}")
    n_sub = _check_nonbool_int("xy_subsamples", xy_subsamples)
    if n_sub < 1:
        raise ValueError(f"xy_subsamples must be >= 1, got {xy_subsamples!r}")
    if derived_shapes not in ("exact", "composite"):
        raise ValueError(f"derived_shapes must be 'exact' or 'composite', got {derived_shapes!r}")
    if not isinstance(diagnostics, bool):
        raise TypeError(f"diagnostics must be a bool, got {diagnostics!r}")
    if isinstance(junction_cells, (str, bytes)) or not isinstance(junction_cells, Sequence) or len(junction_cells) != 3:
        raise ValueError(f"junction_cells must be a 3-tuple of positive integers, got {junction_cells!r}")
    jc = tuple(_check_nonbool_int(f"junction_cells[{i}]", c) for i, c in enumerate(junction_cells))
    if any(c < 1 for c in jc):
        raise ValueError(f"junction_cells entries must be >= 1, got {junction_cells!r}")
    if averaging == "subpixel_full" and (jc[0] % n_sub != 0 or jc[1] % n_sub != 0):
        raise ValueError(f"junction_cells {jc} must split each of the {n_sub} xy_subsamples per axis evenly")
    if interface_failure not in ("raise", "warn"):
        raise ValueError(f"interface_failure must be 'raise' or 'warn', got {interface_failure!r}")


@dataclasses.dataclass(frozen=True)
class StackOptions:
    """Geometry features of a :class:`StackDevice` that cost time, all on by default.

    Each field is the StackDevice field of the same name; pass the options to
    :meth:`StackDevice.from_layer_stack` (``options=``), or the fields one by
    one as before. The defaults are the most accurate settings. What each
    option buys is in docs/guide/options.md and what it costs in
    docs/validation.md ("Geometry option costs"). ``StackOptions.fast()`` turns
    off the ones that are not needed for a quick look at a device.
    """

    #: "subpixel": Kottke averaging along the normal of each voxel's own model, the
    #: diagonal of the averaged tensor (needs 3-component arrays); "subpixel_full": the
    #: same with its off-diagonal entries (9-component arrays: fdtdx then runs its fully
    #: anisotropic update in the whole simulation, ~5x per step); "inverse": the mean of
    #: 1 / eps (fdtdx's convention)
    averaging: Averaging = "subpixel"
    #: normal of "subpixel" averaging: "voxel" (each voxel's own model) or "gradient"
    #: (finite differences of the mean eps; cheaper, up to 15-33 degrees off at slanted walls)
    subpixel_normal: Literal["voxel", "gradient"] = "voxel"
    #: sample each E component's materials at its own Yee position
    yee_staggered: bool = True
    #: split lines of the exact fill at polygon corners and edge crossings (0: off)
    corner_lines: int = 2
    #: lateral sub-samples per voxel and axis (cost ~ s^2)
    xy_subsamples: int = 1
    #: "exact" or "composite" drawing of derived levels (see ``StackDevice.derived_shapes``)
    derived_shapes: Literal["exact", "composite"] = "exact"
    #: warn about unresolved parts, gaps and shrinking overlaps when placed
    diagnostics: bool = True
    #: averaging="subpixel_full": sub-cells per voxel along x, y and z of the junction closure's cell
    #: problem (n x n x m unknowns per junction voxel; n a multiple of xy_subsamples)
    junction_cells: tuple[int, int, int] = (4, 4, 4)
    #: averaging="subpixel_full": a coverage failure (junction voxels or corner columns beyond their static
    #: capacity, a cell problem solved inaccurately) "raise"s (an error where the result is concrete; under
    #: tracing the written box is NaN, so the result cannot be accepted), or "warn"s and keeps the laminate
    #: or half-plane result (counted in the status, see StackDevice.interface_status)
    interface_failure: Literal["raise", "warn"] = "raise"

    def __post_init__(self) -> None:
        _validate_stack_options(
            self.averaging,
            self.yee_staggered,
            self.corner_lines,
            self.xy_subsamples,
            self.derived_shapes,
            self.diagnostics,
            self.junction_cells,
            self.interface_failure,
        )

    @classmethod
    def fast(cls) -> "StackOptions":
        """Every costly feature off: "inverse" averaging, no Yee staggering, no corner model or diagnostics
        (``derived_shapes`` stays ``"exact"``)."""
        return cls(averaging="inverse", yee_staggered=False, corner_lines=0, diagnostics=False)


@autoinit
class StackDevice(StaticMultiMaterialObject):
    """A component with its LayerStack as one differentiable fdtdx object.

    Build it with :meth:`from_layer_stack`, get the matching volume with
    :meth:`get_simulation_volume` and sources / detectors on the component's
    ports with the ``create_*_at`` methods. Merge :func:`init_stack_params` into
    the fdtdx params, update the vertices with :meth:`pack` and apply the device
    with :func:`apply_stack`. Parameters (all in stack units, um / degrees)::

        params[name] = {
            "vertices/<layer>": (E, 2),  # one per source layer with polygons
            "zmin/<level>": (),
            "thickness/<level>": (),
            "sidewall_angle/<level>": (),
        }

    The device box is overwritten on every :func:`apply_stack` (permittivity,
    conductivity and, in dispersive simulations, all pole coefficients):
    everything not covered by a level is ``background``. fdtdx Devices are
    applied afterwards and win where they overlap.

    The polygon topology (T-junctions, seams, polygon order and orientation,
    where the port extensions meet the polygons) is taken from the vertices the
    device is built with, within ``topology_tol``, and indexes any later
    vertices. :meth:`pack` checks that it still fits (:meth:`check_topology`)
    and :meth:`retopologize` takes it from new vertices.

    Attributes:
        levels: compiled levels in the box.
        source_layers: (parameter key, layer index) per source layer with polygons.
        source_vertices: initial packed vertices (um) per source layer; the
            static topology is computed from them.
        source_sizes: vertices per polygon, per source layer.
        background: material where no level is.
        center: gds point (um) at the xy center of the box.
        z_center: stack height (um) at the z center of the box.
        source_extensions: per source layer, (port, length, face end a,
            face end b) of the port extensions it ends with.
        extended_ports: (port, length, layers it may continue) per extended port.
        empty_layers: stack layers the levels in the box read that had no
            polygons, with those levels.
        topology_tol: distance (um) within which the polygon topology is found
            (None: 2 database units of the layout), plus float32 round-off.
        ports: the component's ports.
        roughness: top-surface roughness per level.
        noise_seed: roughness seed when no ``noise_key`` is passed.
        averaging: "subpixel" (Kottke, along the normal of each voxel's own
            model; the diagonal of the averaged tensor), "subpixel_full" (with
            its off-diagonal entries: exact first-order interface correction
            also at walls oblique to the grid; 9-component arrays, and fdtdx's
            fully anisotropic update in the whole simulation, ~5x slower per
            step and ~1.8x the memory) or "inverse".
        subpixel_normal: "voxel" (the normal of each voxel's own model) or
            "gradient" (finite differences of the mean eps; cheaper, less exact).
        yee_staggered: sample each E component at its own Yee position.
        level_margin: how far (um) outside the box a level may start and still
            be drawn, so that zmin / thickness / roughness can move it in.
        edge_width: lateral width (cells) of the voxel square used for areas.
        xy_subsamples: lateral sub-samples per voxel and axis.
        edge_chunk: edges per step of the distance computation.
        corner_lines: split lines of the exact fill in columns near polygon
            corners and crossings of layer edges (0: one half-plane per level
            everywhere).
        corner_columns: columns per lateral sample grid painted with the corner
            model (None: sized when the device is placed; 0: none).
        diagnostics: warn when the device is placed about parts of a level
            and gaps narrower than the grid resolves, and overlapping polygons
            a level shrinks (:meth:`diagnose`).
        derived_shapes: "exact" draws each derived level as the offset of the
            exact polygons of its layer expression (built from
            ``source_vertices``, see ``derived_layers``); "composite" draws the
            min / max of its layers' offset distances.
        derived_layers: the exact shapes of the derived levels.
        dtype: floating-point precision of the simulation, ``jnp.float32``
            (default) or ``jnp.float64``. :meth:`get_simulation_config` builds
            the ``fdtdx.SimulationConfig`` in it, the device refuses a config of
            another dtype, and the parameters, the detectors of the ``create_*_at``
            methods and the arrays :func:`apply_stack` hands back follow it.
            :func:`apply_stack` casts PML coefficients to ``dtype`` so ``float32``
            runs cleanly even when JAX's 64-bit mode is on, whereas ``float64``
            requires JAX's 64-bit mode enabled (``jax.config.update("jax_enable_x64", True)``).
        junction_cells: ``(nx, ny, nz)`` sub-cell grid per junction voxel in
            ``"subpixel_full"`` averaging (default ``(4, 4, 4)``).
        junction_voxels: junction voxels per sample grid solved with the periodic
            cell closure in ``"subpixel_full"`` averaging (None: sized when the
            device is placed; 0: laminate everywhere).
        interface_failure: ``"raise"`` or ``"warn"`` when ``"subpixel_full"`` hits
            capacity overflow or a failed cell solve.
        dropped_levels: ``(name, zmin, zmax)`` of nonempty levels outside the
            device box at construction, checked against grid sampling when placed.
        grid_frame: cell-edge coordinates and uniformity flags captured when the
            device is placed on a grid (None before placement).
    """

    #: compiled levels (see :meth:`from_layer_stack`)
    levels: tuple[_Level, ...] = frozen_field()
    #: (param key, layer index) per source layer with polygons
    source_layers: tuple[tuple[str, int], ...] = frozen_field()
    #: initial packed vertices (um) per source layer; the static topology
    #: (T-junctions, seams, edge directions) is computed from them
    source_vertices: tuple[np.ndarray, ...] = frozen_field()
    #: vertices per polygon, per source layer
    source_sizes: tuple[tuple[int, ...], ...] = frozen_field()
    #: material filling everything that no level covers
    background: str = frozen_field()
    #: gds point (um) at the xy center of the device
    center: tuple[float, float] = frozen_field(default=(0.0, 0.0))
    #: stack height (um) at the z center of the device
    z_center: float = frozen_field(default=0.0)
    #: per source layer: (port, length, a, b) of each port extension, whose 4-vertex
    #: polygons end the layer's vertices; a / b are its corners on the port face:
    #: (i, i) copies the layer's vertex i, (i, j) is where edge i -> j crosses the
    #: face, (-1, -1) is the port's own corner
    source_extensions: tuple[tuple[tuple[str, float, _FaceEnd, _FaceEnd], ...], ...] = frozen_field(
        default=()
    )
    #: (port, length, layer indices it may continue) per port in ``extend_ports``,
    #: to find the extensions again from new polygons (empty: the layers that
    #: have extensions in ``source_extensions``)
    extended_ports: tuple[tuple[str, float, tuple[int, ...]], ...] = frozen_field(default=())
    #: (layer index, (level, nominal bottom, top) of each level in the box that reads
    #: it) per stack layer without polygons when the device was built: the levels
    #: read it as empty, and those that need it are not drawn, so polygons on it
    #: need a new device (:meth:`check_topology` reports them)
    empty_layers: tuple[tuple[int, tuple[tuple[str, float, float], ...]], ...] = frozen_field(default=())
    #: distance (um) within which a vertex lies on an edge of another polygon (a
    #: T-junction, split so that the partly shared edge is a seam), two vertices
    #: coincide (seams of abutting polygons) and a vertex is on a port face, when
    #: the polygon topology is found, by the device and by check_topology and
    #: retopologize alike; float32 round-off (2 epsilons times the largest
    #: coordinate) is added. None: 2 database units of the layout (gf.kcl.dbu),
    #: since snapping to its grid moves a vertex on a slanted edge up to sqrt(2)
    #: units off it. A vertex this close to an edge bends the edge through it;
    #: abutting edges whose ends are this close but apart are a seam too, and a
    #: voxel centered in the gap between them, if any, is filled as if far outside
    topology_tol: float | None = frozen_field(default=None)
    #: the component's ports, for the ``create_*_at`` methods
    ports: tuple[_Port, ...] = frozen_field(default=())
    #: top-surface roughness per level, sampled with ``noise_key``
    roughness: Sequence["TopRoughness"] = field(default=())
    #: seed used for the roughness when no ``noise_key`` is passed
    noise_seed: int = frozen_field(default=0)
    #: "subpixel" (Kottke, along the interface normal of each voxel's own model:
    #: the flux of its interfaces and its vertical profile; the diagonal of the
    #: averaged tensor), "subpixel_full" (the same tensor with its off-diagonal
    #: entries, which oblique walls need; 9-component arrays) or "inverse" (fdtdx
    #: convention)
    averaging: Averaging = frozen_field(default="subpixel")
    #: normal of "subpixel" averaging: "voxel" (from each voxel's own sub-cell model:
    #: exact for straight interfaces, also slanted, thin and sloped ones) or
    #: "gradient" (from finite differences of the voxel-mean eps, as before the
    #: voxel model: errors up to 15-33 degrees at slanted walls, but cheaper)
    subpixel_normal: Literal["voxel", "gradient"] = frozen_field(default="voxel")
    #: with 3-component permittivity arrays (e.g. ``averaging="subpixel"``),
    #: sample each E component's materials at its own Yee position (half a cell
    #: apart); otherwise all components use the voxel center, a first-order error
    yee_staggered: bool = frozen_field(default=True)
    #: levels whose nominal span is within this distance (um) of the box are
    #: drawn, so that zmin / thickness / roughness can move them into it; levels
    #: in the box are drawn over its whole height and may move anywhere in it
    level_margin: float = frozen_field(default=0.0)
    #: lateral width of the voxel square used for the areas, in cells
    edge_width: float = frozen_field(default=1.0)
    #: in-plane sub-samples per voxel and axis: the lateral fill is averaged over
    #: s x s sub-squares, which makes it closer to the exact area fraction at
    #: polygon corners and narrow tips (cost ~ s^2); the subpixel normal comes
    #: from the moments of all of them together
    xy_subsamples: int = frozen_field(default=1)
    #: edges per step in the signed distance computation
    edge_chunk: int = frozen_field(default=64)
    #: in voxel columns near polygon corners and crossings of layer edges, paint
    #: the exact local geometry with up to this many split lines; 0 means one
    #: half-plane per level everywhere, as before (bit-identical). Cost grows
    #: about as 2^M + 1
    corner_lines: int = frozen_field(default=2)
    #: static number of columns per lateral sample grid painted with the corner
    #: model; None sizes it in place_on_grid. Candidates beyond it (raster
    #: order) keep the half-plane result
    corner_columns: int | None = frozen_field(default=None)
    #: when the device is placed, warn (StackDiagnosticWarning) about parts of a
    #: level and gaps narrower than the grid resolves, and about overlapping
    #: polygons a level shrinks; :meth:`diagnose` returns them as data
    diagnostics: bool = frozen_field(default=True)
    #: "exact": a derived level (and / or / not / xor of layers, or one layer whose
    #: polygons overlap) is drawn from the exact polygons of its boolean, built
    #: once as copies of layer vertices and crossings of layer edges, so its
    #: offset is the offset of that shape; "composite": from the min / max of its
    #: layers' distances, as before (slivers where edges of its layers coincide,
    #: sharp grown corners of an and, slits where an or of overlapping parts shrinks)
    derived_shapes: Literal["exact", "composite"] = frozen_field(default="exact")
    #: the exact shapes of the derived levels (with ``derived_shapes="exact"``),
    #: built from ``source_vertices``; part of the frozen topology (:meth:`check_topology`)
    derived_layers: tuple[_DerivedShape, ...] = frozen_field(default=())
    #: averaging="subpixel_full": sub-cells of the junction closure along x, y, z (see StackOptions)
    junction_cells: tuple[int, int, int] = frozen_field(default=(4, 4, 4))
    #: averaging="subpixel_full": static number of junction voxels per sample grid (Yee position); None sizes
    #: it in place_on_grid. Candidates beyond it are a coverage failure (see interface_failure)
    junction_voxels: int | None = frozen_field(default=None)
    #: averaging="subpixel_full": what a coverage failure does, "raise" or "warn" (see StackOptions)
    interface_failure: Literal["raise", "warn"] = frozen_field(default="raise")
    #: precision of the simulation, jnp.float32 or jnp.float64 (see the class docstring); any
    #: spelling of the two is accepted and stored as jnp.float32 / jnp.float64
    dtype: jnp.dtype = frozen_field(default=jnp.float32)
    #: (name, nominal bottom, top in um) of the levels from_layer_stack left out as outside the box
    #: (beyond level_margin and a small slack); place_on_grid warns when the grid samples one of them
    dropped_levels: tuple[tuple[str, float, float], ...] = frozen_field(default=())
    #: the static grid frame (:class:`_Frame`), set by place_on_grid from the resolved grid
    grid_frame: _Frame | None = frozen_field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", _simulation_dtype(self.dtype))
        _validate_stack_options(
            self.averaging,
            self.yee_staggered,
            self.corner_lines,
            self.xy_subsamples,
            self.derived_shapes,
            self.diagnostics,
            self.junction_cells,
            self.interface_failure,
        )
        if isinstance(self.edge_width, bool) or not isinstance(self.edge_width, (int, float, np.number)):
            raise TypeError(f"edge_width must be a positive number, got {self.edge_width!r}")
        ew = float(self.edge_width)
        if not math.isfinite(ew) or ew <= 0.0:
            raise ValueError(f"edge_width must be finite and > 0, got {self.edge_width!r}")
        if _check_nonbool_int("edge_chunk", self.edge_chunk) < 1:
            raise ValueError(f"edge_chunk must be >= 1, got {self.edge_chunk!r}")
        if _check_nonbool_int("corner_columns", self.corner_columns) < 0 if self.corner_columns is not None else False:
            raise ValueError(f"corner_columns must be >= 0 or None, got {self.corner_columns!r}")
        if _check_nonbool_int("junction_voxels", self.junction_voxels) < 0 if self.junction_voxels is not None else False:
            raise ValueError(f"junction_voxels must be >= 0 or None, got {self.junction_voxels!r}")
        if isinstance(self.level_margin, bool) or not isinstance(self.level_margin, (int, float, np.number)):
            raise TypeError(f"level_margin must be a non-negative number, got {self.level_margin!r}")
        lm = float(self.level_margin)
        if not math.isfinite(lm) or lm < 0.0:
            raise ValueError(f"level_margin must be finite and >= 0, got {self.level_margin!r}")
        if self.topology_tol is not None:
            if isinstance(self.topology_tol, bool) or not isinstance(self.topology_tol, (int, float, np.number)):
                raise TypeError(f"topology_tol must be a non-negative number or None, got {self.topology_tol!r}")
            tt = float(self.topology_tol)
            if not math.isfinite(tt) or tt < 0.0:
                raise ValueError(f"topology_tol must be finite and >= 0, got {self.topology_tol!r}")
        _check_nonbool_int("noise_seed", self.noise_seed)

    # ------------------------------------------------------------ construction

    @classmethod
    @timed("from_layer_stack")
    def from_layer_stack(
        cls,
        component: Any,
        layer_stack: Any,
        materials: dict[str, fdtdx.Material],
        background: str,
        *,
        zrange_layers: Sequence[str],
        extend_ports: Sequence[tuple[str, float] | tuple[str, float, Sequence[Any]]] = (),
        boundary_offset: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        levels: Sequence[str] | None = None,
        fill_layers: Sequence[Any] | None = None,
        options: StackOptions | None = None,
        **kwargs: Any,
    ) -> Self:
        """Builds the device and its box from a component and a LayerStack.

        ``extend_ports`` first adds a straight waveguide to each listed port. The
        box then starts as the xy bounding box of the polygons on the stack's
        layers (extensions included; all polygons if the levels only fill the
        plane) and the z range of ``zrange_layers``, and ``boundary_offset``
        moves its faces without changing the geometry. Every level that overlaps
        the final box (or comes within ``level_margin`` of it) is drawn, also
        levels outside ``zrange_layers``, unless no polygon is on its layers.
        Levels within 0.1 um more need a material too; those that do not reach
        the box once fdtdx has placed it on its grid are dropped then.

        Args:
            component: the component; polygons and ports are read with gradients
                stopped.
            layer_stack: a :class:`~gdsfactory.technology.LayerStack`.
            materials: fdtdx materials by ``LayerLevel.material``; needed for
                ``background`` and every level that is drawn.
            background: material (key of ``materials``) where no level is.
            zrange_layers: level names; the lowest bottom and the highest top
                among them are the initial z faces of the box.
            extend_ports: ``((port, length), ...)`` in um, or
                ``((port, length, layers), ...)``. Each port's cross-section is
                continued straight out of the component along the port's
                orientation (opposite to the direction light is injected through
                the port): every stack layer (or only ``layers``) whose polygons
                reach the port face, over a stretch that overlaps the port's
                width, gets a rectangle as wide as that stretch, so a rib
                waveguide continues with its slab and an etch over the port
                continues too. On the port's own layer, a port without polygons
                at its face gets a rectangle of the port's width, unless that
                layer is a fill layer: it already covers the plane, so only the
                other layers reaching the face are extended. The extensions
                are polygons like the others: every level draws them with its
                thickness, sidewall angle and bias, and they follow the port and
                the polygons' edges when the component's parameters change.
            boundary_offset: (x+, x-, y+, y-, z+, z-) in um. Moves each face of the
                box along its axis, positive towards +x, +y or +z for both faces
                of an axis (so (1, -1, ...) grows the box by 1 um on each side
                in x). The geometry does not change: the box shows more or less
                of the same layer stack.
            levels: levels that may be drawn (default: every level with a
                material and nonzero thickness).
            fill_layers: layers that cover the whole plane (default: WAFER).
            options: the geometry features (:class:`StackOptions`; default: all
                on, the most accurate). Its fields are StackDevice fields, which
                can also be passed one by one in ``kwargs``, but not both.
            **kwargs: other StackDevice fields (``name``, ``roughness``,
                ``averaging``, ...). The box comes from the arguments above, so
                ``partial_real_shape``, ``partial_grid_shape``, ``center`` and
                ``z_center`` are not accepted.

        Returns:
            The device; its ``partial_real_shape`` is the box (fdtdx rounds it
            to whole cells around the box center).

        Raises:
            KeyError: unknown level or port names, or a level in the box (or the
                background) whose material is not in ``materials``.
            ValueError: invalid offsets or extensions (no orientation or width,
                length <= 0, a port layer no level draws and that is not a fill
                layer), an empty box, a
                negative ``topology_tol``, or a component without polygons.
            TypeError: a box field passed in ``kwargs``.
        """
        from gdsfactory.pdk import get_layer

        if options is not None:
            given = sorted(set(kwargs) & {f.name for f in dataclasses.fields(StackOptions)})
            if given:
                raise TypeError(f"Pass {given} either in options or as arguments, not both")
            kwargs.update(dataclasses.asdict(options))
        for key in ("partial_real_shape", "partial_grid_shape", "center", "z_center"):
            if key in kwargs:
                raise TypeError(
                    f"from_layer_stack sets {key!r} from zrange_layers, extend_ports and boundary_offset"
                )
        offset = tuple(float(o) for o in boundary_offset)
        if len(offset) != 6 or not all(math.isfinite(o) for o in offset):
            raise ValueError(
                f"boundary_offset must be 6 finite lengths (x+, x-, y+, y-, z+, z-), got {boundary_offset}"
            )
        if isinstance(zrange_layers, str):
            zrange_layers = [zrange_layers]
        if not zrange_layers:
            raise ValueError("zrange_layers must name at least one level")
        levels_seq: list[str] | None = None
        if levels is not None:
            if isinstance(levels, (str, bytes)):
                raise TypeError(f"levels must be a sequence of level names, got {levels!r}")
            levels_seq = list(levels)
            if not levels_seq:
                raise ValueError("levels must name at least one level")
            if len(set(levels_seq)) != len(levels_seq):
                raise ValueError(f"Duplicate level names in levels={levels_seq!r}")
        unknown = [n for n in (*zrange_layers, *(levels_seq or ())) if n not in layer_stack.layers]
        if unknown:
            raise KeyError(f"Unknown levels {unknown}; the stack has {sorted(layer_stack.layers)}")
        if background not in materials:
            raise KeyError(f"background {background!r} is not in materials")
        if _ANISOTROPY_SENTINEL in materials:
            raise ValueError(f"{_ANISOTROPY_SENTINEL!r} is reserved for the device's internal placeholder material")
        if fill_layers is None:
            fill_layers = [layer_stack_wafer_layer()]
        fills = {int(get_layer(lay)) for lay in fill_layers if lay is not None}

        # z faces: the included levels, then the offsets
        spans = [_span(layer_stack.layers[n]) for n in zrange_layers]
        z0 = min(lo for lo, _ in spans) + offset[5]
        z1 = max(hi for _, hi in spans) + offset[4]

        names = levels_seq if levels_seq is not None else list(layer_stack.layers)
        candidates = []
        for name in names:
            level = layer_stack.layers[name]
            if level.material is None or level.thickness == 0:
                if levels is not None:
                    raise ValueError(f"Level {name!r} has no material or zero thickness")
                continue
            candidates.append((name, level, _compile_level(level, fills)))
        stack_layers = set().union(*(_expr_layers(expr) for *_, expr in candidates))
        lap("compile_levels")

        ports = [_read_port(port) for port in component.ports]

        # static polygons per layer (um), then the port extensions on their layers
        polygons = {
            index: [np.asarray(to_numpy(p), dtype=float).reshape(-1, 2) for p in polys]
            for index, polys in component.get_polygons_points().items()
        }
        lap("polygons")
        by_name = {p.name: p for p in ports}
        extended = []
        seen_ext: set[tuple[str, int]] = set()
        seen_empty: set[str] = set()
        for item in extend_ports:
            if isinstance(item, (str, bytes)) or not isinstance(item, Sequence) or len(item) not in (2, 3):
                raise ValueError(
                    f"extend_ports entries must be (port_name, length[, layers]), got {item!r}"
                )
            name, length, *only = item
            if name not in by_name:
                raise KeyError(f"Unknown port {name!r} in extend_ports; the component has {sorted(by_name)}")
            p = by_name[name]
            if (
                p.orientation is None
                or p.width <= 0
                or isinstance(length, bool)
                or not math.isfinite(float(length))
                or not float(length) > 0
            ):
                raise ValueError(f"Cannot extend port {name!r}: needs an orientation, a width and a length > 0")
            if p.layer not in stack_layers and p.layer not in fills:
                raise ValueError(f"Port {name!r} is on layer {p.layer}, which no level of the stack draws")
            layers = sorted(stack_layers)
            if only:
                layers = [index for index in layers if index in {int(get_layer(lay)) for lay in only[0]}]
            if layers:
                for idx in layers:
                    if (str(name), idx) in seen_ext:
                        raise ValueError(f"Duplicate extend_ports entry for port {name!r} on layer {idx}")
                    seen_ext.add((str(name), idx))
            else:
                if str(name) in seen_empty:
                    raise ValueError(f"Duplicate extend_ports entry for port {name!r}")
                seen_empty.add(str(name))
            extended.append((str(name), float(length), tuple(layers)))
        topology_tol = kwargs.get("topology_tol")
        if topology_tol is not None and not float(topology_tol) >= 0:
            raise ValueError(f"topology_tol must be >= 0, got {topology_tol}")
        extensions, rectangles = _port_extensions(polygons, by_name, extended, topology_tol)
        lap("port_extensions")

        # xy faces: bounding box of the polygons on the stack's layers (any z;
        # all polygons if the levels only fill), then the offsets
        points = [q for index in stack_layers for q in (*polygons.get(index, []), *rectangles.get(index, []))]
        if not points:
            points = [q for polys in polygons.values() for q in polys]
        if not points:
            raise ValueError("The component has no polygons")
        points = np.concatenate(points)
        (bx0, by0), (bx1, by1) = points.min(axis=0), points.max(axis=0)
        x0, x1 = bx0 + offset[1], bx1 + offset[0]
        y0, y1 = by0 + offset[3], by1 + offset[2]
        if not (x1 > x0 and y1 > y0 and z1 > z0):
            raise ValueError(f"Empty box x [{x0}, {x1}], y [{y0}, {y1}], z [{z0}, {z1}] um")

        # levels just outside the box count too (see _SELECT_SLACK); the exact
        # selection needs the grid and happens in place_on_grid
        level_margin = float(kwargs.get("level_margin", 0.0))
        if level_margin < 0:
            raise ValueError(f"level_margin must be >= 0, got {level_margin}")
        margin = level_margin + _SELECT_SLACK
        nonempty = {index for index, polys in polygons.items() if polys} | set(rectangles)
        roughness_arg = kwargs.get("roughness", ())
        drawn_spans = [(name, *_span(level)) for name, level, expr in candidates if _expr_can_draw(expr, nonempty)]
        keep_drawn, _ = _roughness_reach(drawn_spans, roughness_arg, z0 - margin, z1 + margin)
        compiled = []
        empty: dict[int, list[tuple[str, float, float]]] = {}
        dropped: list[tuple[str, float, float]] = []
        for name, level, expr in candidates:
            lo, hi = _span(level)
            if _expr_can_draw(expr, nonempty):
                if name not in keep_drawn:
                    dropped.append((name, float(lo), float(hi)))  # place_on_grid checks it is not sampled
                    continue
            elif hi <= z0 - margin or lo >= z1 + margin:
                continue  # outside the box
            for index in sorted(_expr_layers(expr) - nonempty):
                empty.setdefault(index, []).append((name, float(lo), float(hi)))
            if not _expr_can_draw(expr, nonempty):
                continue  # no polygons on its layers (e.g. a metal level of a passive component)
            if level.material not in materials:
                raise KeyError(
                    f"Level {name!r} (z {lo:g} to {hi:g} um) is in the box but its material "
                    f"{level.material!r} is not in materials {sorted(materials)}; pass it, or "
                    "restrict levels=[...]"
                )
            zmin_val = float(level.zmin)
            thick_val = float(level.thickness)
            w2z_val = float(level.width_to_z)
            angle_val = float(level.sidewall_angle)
            if not (math.isfinite(zmin_val) and math.isfinite(thick_val) and math.isfinite(w2z_val)):
                raise ValueError(f"Level {name!r}: zmin, thickness, and width_to_z must be finite")
            if not math.isfinite(angle_val) or abs(angle_val) >= 90.0:
                raise ValueError(
                    f"Level {name!r}: abs(sidewall_angle) must be < 90 degrees, got {level.sidewall_angle!r}"
                )
            bias = level.bias
            if isinstance(bias, (tuple, list)):
                if len(bias) != 2 or bias[0] != bias[1]:
                    raise NotImplementedError(f"Anisotropic bias {bias} is not supported")
                bias = bias[0]
            bias_val = float(bias or 0.0)
            if not math.isfinite(bias_val):
                raise ValueError(f"Level {name!r}: bias must be finite, got {level.bias!r}")
            z_to_bias = None
            if level.z_to_bias is not None:
                if (
                    isinstance(level.z_to_bias, (str, bytes))
                    or not isinstance(level.z_to_bias, Sequence)
                    or len(level.z_to_bias) != 2
                ):
                    raise ValueError(
                        f"Level {name!r}: z_to_bias must be a 2-tuple (heights, biases), got {level.z_to_bias!r}"
                    )
                zs_raw, bs_raw = level.z_to_bias
                zs = tuple(float(v) for v in zs_raw)
                bs = tuple(float(v) for v in bs_raw)
                if not zs or len(zs) != len(bs):
                    raise ValueError(f"Level {name!r}: z_to_bias heights and biases must be non-empty and equal length")
                if not all(math.isfinite(v) for v in (*zs, *bs)):
                    raise ValueError(f"Level {name!r}: z_to_bias values must be finite")
                if any(zs[i + 1] <= zs[i] for i in range(len(zs) - 1)):
                    raise ValueError(f"Level {name!r}: z_to_bias heights must be strictly increasing, got {zs}")
                z_to_bias = (zs, bs)
            fill_layer = None
            if expr[0] == "fill" and not level.background:
                try:
                    declared = _expr_positive_layers(_compile_layer(level.layer, set()))
                except (KeyError, TypeError, ValueError):
                    declared = set()
                if len(declared) == 1:
                    fill_layer = declared.pop()
            compiled.append(
                _Level(
                    name=name,
                    expr=expr,
                    zmin=zmin_val,
                    thickness=thick_val,
                    sidewall_angle=angle_val,
                    width_to_z=w2z_val,
                    bias=bias_val,
                    z_to_bias=z_to_bias,
                    mesh_order=int(level.mesh_order),
                    material=level.material,
                    fill_layer=fill_layer,
                )
            )
        level_names = {lv.name for lv in compiled}
        for r in kwargs.get("roughness", ()):
            if r.level not in level_names:
                raise KeyError(f"roughness level {r.level!r} is not among {sorted(level_names)}")

        layer_ids = sorted(set().union(*(_expr_layers(lv.expr) for lv in compiled)))
        # extensions on layers no level in the box draws are dropped with them,
        # but the port's own layer must be drawn, or fill the plane
        missing = [
            n for n, *_ in extend_ports
            if by_name[n].layer not in layer_ids and by_name[n].layer not in fills
        ]
        if missing:
            raise ValueError(f"No level in the box draws the layer of the extended ports {missing}")
        sources, verts, sizes, exts = [], [], [], []
        for index in layer_ids:
            polys = [*polygons.get(index, []), *rectangles.get(index, [])]
            if not polys:
                continue  # empty layer: constant "outside"
            sources.append((_layer_key(index), index))
            verts.append(np.concatenate(polys))
            sizes.append(tuple(len(q) for q in polys))
            exts.append(tuple(extensions.get(index, ())))

        used = {lv.material for lv in compiled} | {background}
        mats = {k: v for k, v in materials.items() if k in used}
        _check_materials(mats, kwargs.get("averaging", "subpixel"), kwargs.get("name", "stack"))

        lap("select_levels")
        device = cls(
            materials=align_material_poles(mats),
            levels=tuple(compiled),
            source_layers=tuple(sources),
            source_vertices=tuple(verts),
            source_sizes=tuple(sizes),
            source_extensions=tuple(exts),
            extended_ports=tuple(extended),
            empty_layers=tuple((index, tuple(levels)) for index, levels in sorted(empty.items())),
            dropped_levels=tuple(dropped),
            background=background,
            partial_real_shape=((x1 - x0) * 1e-6, (y1 - y0) * 1e-6, (z1 - z0) * 1e-6),
            center=(0.5 * (x0 + x1), 0.5 * (y0 + y1)),
            z_center=0.5 * (z0 + z1),
            ports=tuple(ports),
            **kwargs,
        )
        return device._with_derived_layers()

    @property
    def options(self) -> StackOptions:
        """The device's geometry features as a :class:`StackOptions`."""
        return StackOptions(**{f.name: getattr(self, f.name) for f in dataclasses.fields(StackOptions)})

    @timed("derived_shapes")
    def _with_derived_layers(self) -> Self:
        """The device with ``derived_layers`` built from its ``source_vertices`` (see ``derived_shapes``)."""
        shapes: tuple[_DerivedShape, ...] = ()
        if self.derived_shapes == "exact":
            shapes = _derived_shapes(
                self.levels, self.source_layers, self.source_vertices, self.source_sizes, self._topology_tol,
                self.name,
            )
        elif self.derived_shapes != "composite":
            raise ValueError(f"derived_shapes must be 'exact' or 'composite', got {self.derived_shapes!r}")
        return self.aset("derived_layers", shapes)

    def _slots(self) -> list[tuple[str, int]]:
        """(key, layer index) per layer slot: the source layers, then the derived shapes' nonempty parts."""
        return [*self.source_layers, *(
            (f"derived/{'+'.join(d.levels)}/{j}", _part_index(k, j))
            for k, d in enumerate(self.derived_layers) for j, r in enumerate(d.recipes) if len(r.kind)
        )]

    def _drawn_expr(self, level: _Level) -> tuple:
        """The expression a level is drawn with: the union of its derived shape's parts (offset by its root
        sizing), or its own.
        """
        for k, d in enumerate(self.derived_layers):
            if level.name in d.levels:
                grow = _root_grow(level.expr)
                parts = [("layer", _part_index(k, j), 0.0) for j, r in enumerate(d.recipes) if len(r.kind)]
                if not parts:
                    return ("fill", grow) if d.recipes[0].negate else ("layer", _NO_LAYER, 0.0)
                union = parts[0]
                for part in parts[1:]:
                    union = ("or", union, part, 0.0)
                if d.recipes[0].negate:
                    return ("not", ("fill", 0.0), union, grow)
                return _with_root_grow(union, grow)
        return level.expr

    # ------------------------------------------------------------ parameters

    def init_params(self, key: Array | None = None) -> dict[str, Array]:
        """Initial parameters: the component's vertices and the stack's thicknesses and angles.

        Args:
            key: unused (the parameters are deterministic).

        Returns:
            ``{"vertices/<layer>": (E, 2), "zmin/<level>": (), "thickness/<level>": (),
            "sidewall_angle/<level>": ()}``.
        """
        del key
        p: dict[str, Array] = {}
        for (k, _), v in zip(self.source_layers, self.source_vertices, strict=True):
            p[k] = jnp.asarray(v, dtype=self.dtype)
        for lv in self.levels:
            p[f"zmin/{lv.name}"] = jnp.asarray(lv.zmin, dtype=self.dtype)
            p[f"thickness/{lv.name}"] = jnp.asarray(lv.thickness, dtype=self.dtype)
            p[f"sidewall_angle/{lv.name}"] = jnp.asarray(lv.sidewall_angle, dtype=self.dtype)
        return p

    @timed("pack")
    def pack(self, component: Any, check: bool = True) -> dict[str, Array]:
        """Vertex parameters of a (possibly traced) component, port extensions included.

        Args:
            component: a component with the polygon topology of the one the
                device was built from (the same number of polygons and vertices
                per polygon on every source layer, see :meth:`check_topology`)
                and the extended ports.
            check: if the component is concrete (not traced), check that its
                vertices still fit the device's topology
                (:meth:`check_topology`). Traced components are never checked.

        Returns:
            ``{"vertices/<layer>": (E, 2)}``, differentiable with respect to the
            parameters the component was built from; the extensions follow their
            ports and the polygon edges they continue. Which vertices and edges
            an extension continues is fixed when the device is built.

        Raises:
            ValueError: if the number of polygons or vertices changed or, with
                ``check``, the topology no longer fits (:meth:`retopologize`
                gives a device for the new vertices) or the component has
                polygons on a stack layer the device was built without.
        """
        out = self._pack(component, cast=False)
        if check and not has_tracers(out):
            self._raise_on_changes(out, component)
        return {k: jnp.asarray(v, dtype=self.dtype) for k, v in out.items()}

    @timed("pack_vertices")
    def _pack(self, component: Any, *, cast: bool = True) -> dict[str, Any]:
        """:meth:`pack` without the topology check."""
        polygons = component.get_polygons_points()
        out: dict[str, Any] = {}
        extensions = self.source_extensions or tuple(() for _ in self.source_layers)
        for (k, index), sizes, exts in zip(self.source_layers, self.source_sizes, extensions, strict=True):
            polys = polygons.get(index, [])
            own = sizes[: len(sizes) - len(exts)]
            s = tuple(int(np.shape(q)[0]) for q in polys)
            if s != own:
                raise ValueError(
                    f"Polygon topology of {k} changed: device has {len(own)} polygons with sizes "
                    f"{own}, got {len(s)} with {s}. Build a device for the new polygons with "
                    "device.retopologize(component), or StackDevice.from_layer_stack."
                )
            ext_ports = [component.ports[name] for name, *_ in exts]
            concrete = not has_tracers(polys) and not has_tracers(
                [(p.center, p.width, p.orientation) for p in ext_ports]
            )
            if concrete:
                v = (
                    np.concatenate([np.asarray(to_numpy(q), dtype=np.float64).reshape(-1, 2) for q in polys], axis=0)
                    if polys
                    else np.zeros((0, 2), dtype=np.float64)
                )
                rects = []
                for (name, length, a, b), port in zip(exts, ext_ports, strict=True):
                    center = np.asarray([float(c) for c in port.center], dtype=np.float64)
                    rects.append(
                        np.asarray(
                            _extension_rect(
                                (a, b), v, center, float(port.width), float(port.orientation), float(length), np
                            ),
                            dtype=np.float64,
                        )
                    )
                out[k] = np.concatenate([v, *rects], axis=0) if rects else v
            else:
                v = pack_polygons(polys)[0] if polys else jnp.zeros((0, 2))
                rects = []
                for (name, length, a, b), port in zip(exts, ext_ports, strict=True):
                    # face corners on the polygons' own vertices and edges: an exact seam
                    center = jnp.stack([jnp.asarray(c, dtype=v.dtype) for c in port.center])
                    width = jnp.asarray(port.width, dtype=v.dtype)
                    orientation = jnp.asarray(port.orientation, dtype=v.dtype)
                    rects.append(_extension_rect((a, b), v, center, width, orientation, length, jnp).astype(v.dtype))
                out[k] = jnp.concatenate([v, *rects]) if rects else v
        if cast:
            return {k: jnp.asarray(v, dtype=self.dtype) for k, v in out.items()}
        return out

    # ------------------------------------------------------------ topology

    @timed("check_topology")
    def check_topology(self, vertices: Any) -> None:
        """Checks that new vertices still fit the polygon topology the device was built with.

        Once, from the vertices it is built with (``source_vertices``), the
        device finds which polygon vertices lie inside edges of other polygons
        (T-junctions, split so that partly shared edges become seams), which
        edges are seams (abutting polygons, hole cut lines) and are left out of
        the distance, which way each polygon runs (for the corner model) and,
        for each port extension, where its layer meets the port face. Later
        vertices are indexed with all that, by polygon and vertex, so it must
        still fit them. Per source layer, this finds:

        - polygons that look reordered among the polygons of their size: one now
          lies where another one was, much closer than to its own place, in a
          cycle of such matches (identical polygons moving far, as in a
          stretched grating, do not count);
        - T-junctions that came apart (e.g. a taper moved off the side of an
          MMI, whose side would keep a bump through the taper's corners) and
          new ones;
        - seams that opened (e.g. abutting polygons moved apart: the gap would
          stay filled, and points inside it get wrong distances) and new ones;
        - polygons that now run the other way round (with ``corner_lines``);
        - per exact derived shape (``derived_layers``): where the polygons its
          recipe draws from the new vertices differ from the exact boolean of
          them by more than the tolerance (a part appeared or vanished, a seam
          opened), and crossings that left one of their two edges;
        - with a component: port extensions whose layer now meets the port face
          elsewhere (a vertex left the face, an edge no longer crosses it, or
          the layer reaches the face in more, fewer or other places), and
          polygons on a stack layer that had none when the device was built
          (``empty_layers``): the levels read it as empty and those that need
          it are not drawn, so only :meth:`from_layer_stack` builds a device
          for them.

        Relations are found with the device's topology tolerance
        (``topology_tol``, 2 database units by default, plus float32 round-off;
        computed from the new vertices for them), as :meth:`retopologize` and
        :meth:`from_layer_stack` find them: what held for the device's vertices
        counts as kept while it holds within it, and what holds within it now
        but did not is new. So layout snapping, round-off and moves that keep
        the topology (a taper sliding along an MMI side, abutting polygons
        moving together) pass, and when the check passes, a device built for
        the new vertices has the same T-junctions, seams and extensions: the
        device's edges bend through the new vertices by at most the tolerance.
        Traced vertices (inside ``jax.grad`` or ``jax.jit``) are not checked,
        so this costs nothing in a differentiated objective.

        Args:
            vertices: a component, packed like :meth:`pack` does (its ports
                locate the port faces), or device parameters with
                ``vertices/<layer>`` entries (um; other entries are ignored),
                such as the output of :meth:`pack` or ``params[name]``. Port
                extensions and new layers are only checked with a component.

        Raises:
            ValueError: if the topology no longer fits, naming each layer and what
                changed in it (:meth:`retopologize` gives a device for the new
                vertices), if the component has polygons on a layer the device
                was built without, or if the parameters have none of the
                device's layers, or another number of vertices.
        """
        component = None
        if hasattr(vertices, "get_polygons_points"):
            component, vertices = vertices, self._pack(vertices, cast=False)
        else:
            keys = [k for k, _ in self.source_layers]
            if not keys:
                return
            vertices = {k: v for k, v in vertices.items() if k in keys}
            if not vertices:
                raise ValueError(
                    f"No vertices of StackDevice {self.name!r} in the parameters: pass params[{self.name!r}], "
                    f"the output of pack(), or a component (layers {keys})"
                )
        if not has_tracers(vertices):
            self._raise_on_changes(vertices, component)

    @timed("retopologize")
    def retopologize(self, vertices: Any) -> Self:
        """A copy of the device with the polygon topology of new vertices.

        When :meth:`pack` or :meth:`check_topology` reports that new vertices no
        longer fit the device's topology, this gives the device built for them:
        its ``source_vertices`` (and :meth:`init_params`) are the new vertices,
        so their T-junctions, seams, polygon order and orientations are found
        again, within the same tolerance as the check uses (``topology_tol``):
        whatever the check counts as kept, this keeps; the exact shapes of the
        derived levels (``derived_layers``) are built again from them too.
        Everything else is kept:
        the box, the levels and source layers, the materials, the other fields
        and the placement of a placed device, whose budgets of corner columns
        and junction voxels only grow if the new vertices need more. Swap it into the placed
        objects and refresh its parameters, e.g. ``objects = objects.copy();
        objects[name] = objects[name].retopologize(c)`` and ``params[name] =
        {**params[name], **objects[name].pack(c)}``.

        Args:
            vertices: a component, read like :meth:`from_layer_stack` does: its
                polygons may also differ in number and size, the ports are read
                again and the port extensions are found again from its polygons
                and ports, on the layers they may continue. Or device parameters
                with every ``vertices/<layer>`` entry (um) and the device's
                number of vertices per layer: the polygon sizes and the
                extensions' face ends are kept (use a component when those
                change).

        Returns:
            The new device.

        Raises:
            TypeError: if the vertices are traced (inside ``jax.grad`` or
                ``jax.jit``).
            KeyError: if an extended port is not among the component's ports, or
                the parameters lack a layer's vertices.
            ValueError: if a source layer has no polygons left, the component
                has polygons on a stack layer the device was built without
                (``empty_layers``: only :meth:`from_layer_stack` adds the levels
                and source layers for them), or the parameters have another
                number of vertices.
        """
        if hasattr(vertices, "get_polygons_points"):
            component = vertices
            raw = component.get_polygons_points()
            if has_tracers(raw) or has_tracers([(p.center, p.width, p.orientation) for p in component.ports]):
                raise TypeError("retopologize needs concrete vertices; call it outside jax.grad and jax.jit")
            new = self._new_layers(raw)
            if new:
                raise ValueError(
                    f"StackDevice {self.name!r} cannot take the component's polygons on layers it was built "
                    "without:\n" + "\n".join(f"  - {p}" for p in new)
                    + "\nBuild a new device for the component with StackDevice.from_layer_stack."
                )
            polygons = {
                index: [np.asarray(to_numpy(p), dtype=float).reshape(-1, 2) for p in polys]
                for index, polys in raw.items()
            }
            ports = tuple(_read_port(port) for port in component.ports)
            by_name = {p.name: p for p in ports}
            extended = self._extended_ports()
            missing = [name for name, *_ in extended if name not in by_name]
            if missing:
                raise KeyError(f"The extended ports {missing} are not among the component's ports {sorted(by_name)}")
            drawn = {index for _, index in self.source_layers}
            extensions, rectangles = _port_extensions(
                polygons,
                by_name,
                [(name, length, tuple(i for i in ls if i in drawn)) for name, length, ls in extended],
                self.topology_tol,
            )
            verts, sizes, exts = [], [], []
            for k, index in self.source_layers:
                polys = [*polygons.get(index, []), *rectangles.get(index, [])]
                if not polys:
                    raise ValueError(
                        f"{k} has no polygons in the component; build a new device with StackDevice.from_layer_stack"
                    )
                verts.append(np.concatenate(polys))
                sizes.append(tuple(len(q) for q in polys))
                exts.append(tuple(extensions.get(index, ())))
            out = self.aset("source_vertices", tuple(verts))
            out = out.aset("source_sizes", tuple(sizes))
            out = out.aset("source_extensions", tuple(exts))
            out = out.aset("extended_ports", extended)
            out = out.aset("ports", ports)
        else:
            verts = []
            for (k, _), v0 in zip(self.source_layers, self.source_vertices, strict=True):
                if k not in vertices:
                    raise KeyError(f"retopologize needs every layer's vertices; {k!r} is missing")
                if has_tracers(vertices[k]):
                    raise TypeError("retopologize needs concrete vertices; call it outside jax.grad and jax.jit")
                v = np.asarray(vertices[k], dtype=float)
                if v.shape != np.shape(v0):
                    raise ValueError(
                        f"{k} has {v.shape[0]} vertices, the device {len(v0)}; pass a component to change the "
                        "polygons' sizes"
                    )
                verts.append(v)
            out = self.aset("source_vertices", tuple(verts))
        out = out._with_derived_layers()
        if out._grid_slice_tuple != INVALID_SLICE_TUPLE_3D and out.corner_lines > 0 and out.corner_columns:
            budget = out._corner_budget()
            if budget > out.corner_columns:
                out = out.aset("corner_columns", budget)
        if out._grid_slice_tuple != INVALID_SLICE_TUPLE_3D and out.averaging == "subpixel_full":
            jbudget = out._junction_budget()
            if out.junction_voxels is None or jbudget > out.junction_voxels:
                out = out.aset("junction_voxels", jbudget)
        return out

    def _extended_ports(self) -> tuple[tuple[str, float, tuple[int, ...]], ...]:
        """``extended_ports``, or for devices built without it, the layers their extensions are on."""
        if self.extended_ports or not self.source_extensions:
            return tuple(self.extended_ports)
        layers: dict[tuple[str, float], list[int]] = {}
        for (_, index), exts in zip(self.source_layers, self.source_extensions, strict=True):
            for name, length, *_ in exts:
                found = layers.setdefault((name, float(length)), [])
                if index not in found:
                    found.append(index)
        return tuple((name, length, tuple(found)) for (name, length), found in layers.items())

    def _topology_tol(self, vertices: Any) -> float:
        """The device's topology tolerance (um) for a layer's vertices (see ``topology_tol``)."""
        return _topology_tol(self.topology_tol, vertices)

    def _new_layers(self, polygons: dict[int, Sequence[Any]]) -> list[str]:
        """Stack layers the device was built without that now have polygons (see ``empty_layers``).

        Args:
            polygons: the component's polygons per layer index.

        Returns:
            One line per such layer, starting with the parameter key it would have.
        """
        drawn = {lv.name for lv in self.levels}
        found = []
        for index, readers in self.empty_layers:
            count = len(polygons.get(index, ()))
            if count:
                levels = ", ".join(
                    f"level {name!r} {'reads it as empty' if name in drawn else 'is not drawn'}" for name, *_ in readers
                )
                found.append(
                    f"{_layer_key(index)}: {count} polygon{'s' if count > 1 else ''} on a layer the device was "
                    f"built without ({levels})"
                )
        return found

    @timed("topology_check")
    def _raise_on_changes(self, vertices: dict[str, Any], component: Any) -> None:
        """Raises a ValueError that lists :meth:`_new_layers` and :meth:`_topology_changes`, if there are any."""
        new = [] if component is None else self._new_layers(component.get_polygons_points())
        problems = new + self._topology_changes(vertices, component)
        if not problems:
            return
        if new:
            fix = (
                "Build a new device for the component with StackDevice.from_layer_stack (retopologize keeps the "
                "levels and source layers)."
            )
        else:
            fix = (
                "Build a device for them with device.retopologize(component) (it keeps the box, the levels and "
                "the placement; then refresh its params), or pass vertices that keep the topology "
                "(pack(component, check=False) skips this check)."
            )
        raise ValueError(
            f"The vertices no longer fit the polygon topology StackDevice {self.name!r} was built with "
            "(lengths in um):\n" + "\n".join(f"  - {p}" for p in problems) + "\n" + fix
        )

    def _topology_changes(self, vertices: dict[str, Any], component: Any) -> list[str]:
        """What no longer fits in concrete vertices, per layer (see :meth:`check_topology`).

        The device's relations are those found within its topology tolerance
        for ``source_vertices``; the new vertices' are those found within the
        tolerance for them, as a device built for them would find them.

        Args:
            vertices: ``{"vertices/<layer>": (E, 2)}`` (um, concrete), port
                extensions included; layers that are missing are skipped.
            component: the component, to check the port extensions, or None.

        Returns:
            One line per change, starting with the layer's parameter key.
        """
        extensions = self.source_extensions or tuple(() for _ in self.source_layers)
        extended = self._extended_ports()
        ports = {}
        if component is not None:
            ports = {name: _read_port(component.ports[name]) for name, *_ in extended}
        problems = []
        for (k, index), v0, sizes, exts in zip(
            self.source_layers, self.source_vertices, self.source_sizes, extensions, strict=True
        ):
            if k not in vertices:
                continue
            v = np.asarray(vertices[k]).astype(float)
            if v.shape != np.shape(v0):
                problems.append(f"{k}: {v.shape[0]} vertices, the device has {len(v0)}")
                continue
            found = []
            own_sizes = sizes[: len(sizes) - len(exts)]
            if ports:
                own = v[: sum(own_sizes)]
                found += self._extension_changes(
                    index, own, own_sizes, exts, extended, ports, self._topology_tol(own)
                )
            if not np.isfinite(v).all():
                found.append(f"vertex {int(np.flatnonzero(~np.isfinite(v).all(axis=1))[0])} is not finite")
            else:
                found += topology_changes(
                    v0, v, sizes, self._topology_tol(v), self._topology_tol(v0), oriented=self.corner_lines > 0
                )
            problems += [f"{k}: {p}" for p in found]
        return problems + self._derived_changes(vertices)

    def _derived_changes(self, vertices: dict[str, Any]) -> list[str]:
        """What no longer fits the derived shapes' recipes in concrete vertices (see :meth:`check_topology`).

        Per shape: a crossing that left one of its two edges (beyond the
        topology tolerance of its ends), and where the recipe's polygons
        differ from the exact boolean of the new vertices by more than the
        tolerance (a part that appeared or vanished, a seam that opened).

        Args:
            vertices: ``{"vertices/<layer>": (E, 2)}`` (um, concrete); shapes
                with a missing layer are skipped.

        Returns:
            One line per change, starting with the levels that draw the shape.
        """
        import shapely

        keys = {index: k for k, index in self.source_layers}
        sizes = {index: n for (_, index), n in zip(self.source_layers, self.source_sizes, strict=True)}
        problems = []
        for d in self.derived_layers:
            if not all(keys[index] in vertices for index in d.layers):
                continue
            pool = [np.asarray(vertices[keys[index]], dtype=float) for index in d.layers]
            if any(v.shape != (sum(sizes[i]), 2) or not np.isfinite(v).all() for v, i in zip(pool, d.layers, strict=True)):
                continue  # reported per layer
            levels = ", ".join(repr(n) for n in d.levels)
            prefix = f"level {levels} (exact shape of {', '.join(keys[i] for i in d.layers)})"
            flat = np.concatenate(pool)
            tol = max(self._topology_tol(v) for v in pool)
            starts = np.cumsum([0, *(len(v) for v in pool)])
            for _, a, b, x in [c for r in d.recipes for c in crossing_changes(r, flat, tol)][:3]:
                layer = int(np.searchsorted(starts, a, side="right")) - 1
                problems.append(
                    f"{prefix}: the crossing of edge {a - starts[layer]}->{b - starts[layer]} of "
                    f"{keys[d.layers[layer]]} with another edge left that edge, now at ({x[0]:.6g}, {x[1]:.6g})"
                )
            for _, _, p, q in [c for r in d.recipes for c in joint_changes(r, flat, tol)][:3]:
                problems.append(
                    f"{prefix}: a joint of two layers' collinear edges opened into a step from "
                    f"({p[0]:.6g}, {p[1]:.6g}) to ({q[0]:.6g}, {q[1]:.6g})"
                )
            lo, hi = flat.min(axis=0) - 1.0, flat.max(axis=0) + 1.0
            universe = shapely.box(*lo, *hi)
            drawn = _shape_region(d.recipes, flat)
            if d.recipes[0].negate:
                drawn = universe.difference(drawn)
            regions = {i: _shapely_polygons(v, sizes[i]) for i, v in zip(d.layers, pool, strict=True)}
            error = drawn.symmetric_difference(_expr_region(d.expr, regions, universe))
            if not error.buffer(-_DERIVED_ERROR_WIDTH * tol).is_empty:
                circle = shapely.maximum_inscribed_circle(
                    error.buffer(-_DERIVED_ERROR_WIDTH * tol).buffer(_DERIVED_ERROR_WIDTH * tol)
                )
                c = np.asarray(circle.coords[0])
                width = 2 * float(circle.length)
                problems.append(
                    f"{prefix}: the shape it draws differs from the boolean of the new vertices by a part "
                    f"{width:.4g} um wide at ({c[0]:.6g}, {c[1]:.6g}): a part appeared or vanished, or a seam opened"
                )
        return problems

    @staticmethod
    def _extension_changes(
        index: int,
        own: np.ndarray,
        own_sizes: Sequence[int],
        exts: Sequence[tuple[str, float, _FaceEnd, _FaceEnd]],
        extended: Sequence[tuple[str, float, tuple[int, ...]]],
        ports: dict[str, _Port],
        tol: float,
    ) -> list[str]:
        """Port faces where a layer's polygons no longer meet the face as its extensions assume.

        The face corners of the device's extensions (from their frozen face ends,
        with the current vertices and ports) are compared with those found again
        (:func:`_face_intervals`, within ``tol`` as a device built now would find
        them); they may move by up to ``tol``. The corners set where each
        extension starts and how wide it is over its whole length, so a corner
        that moved along the face counts as much as one that moved off it.

        Args:
            index: the layer.
            own: (E, 2) current vertices (um) of its own polygons (no extensions).
            own_sizes: vertices per own polygon.
            exts: the device's extensions on the layer (``source_extensions``).
            extended: (port, length, layers it may continue) per extended port.
            ports: the component's current ports by name.
            tol: the layer's topology tolerance (um) for ``own``.

        Returns:
            One line per port whose extensions changed.
        """

        def spans(c: np.ndarray) -> str:
            if not len(c):
                return "no extension"
            return "extensions from " + ", ".join(f"({a[0]:.6g}, {a[1]:.6g}) to ({b[0]:.6g}, {b[1]:.6g})" for a, b in c)

        found = []
        for name, length, layers in extended:
            if index not in layers:
                continue
            port = ports[name]
            now = _face_intervals(port, own, own_sizes, tol)
            if index == port.layer and not now:
                now = [(_PORT_CORNER, _PORT_CORNER)]
            built = _face_spans([(a, b) for n, ln, a, b in exts if n == name and ln == length], own, port)
            seen = _face_spans(now, own, port)
            # non-finite where an edge no longer crosses the face: never within tol
            if built.shape == seen.shape and bool(np.all(np.linalg.norm(built - seen, axis=-1) <= tol)):
                continue
            found.append(
                f"its cross-section at port {name!r} changed: the device has {spans(built)} on the port face, "
                f"a device built now would have {spans(seen)}"
            )
        return found

    # ------------------------------------------------------------ diagnostics

    @timed("diagnose")
    def diagnose(self, resolution: float | None = None) -> list[StackDiagnostic]:
        """Geometry the grid does not resolve, and overlaps a level's offset does not draw as their union's.

        Every level but the fills is checked in the box, on the shape it draws
        from the device's polygons (``source_vertices``, port extensions
        included) with its layer expression, sizings, bias, z_to_bias and
        sidewall angle at its nominal zmin, thickness and angle:

        - ``"width"`` and ``"gap"``: parts of the shape, and gaps in it,
          narrower than 1.5 cells divided by ``xy_subsamples`` (sub-samples
          resolve thinner features), between any two edges that face each
          other at less than 90 degrees: also a corner pointing at an edge or
          at another corner, and a V-notch nearly cutting through a part. The
          two sides of one corner (sharp, cut, blunt or rounded) are the
          corner, not a narrow part, where they meet at 45 degrees or more;
          sides meeting at a smaller angle (a needle, a taper) are measured
          where they do not touch. The layers are offset as the device draws
          them, by their signed distance: with round joins where a layer grows
          at a convex corner or shrinks at a reflex one (drawn to within 1% of
          the threshold), and a round join is part of its corner, not a narrow
          part with the edges next to it. Where the level's boundary is
          offset, the shape changes with height, so it is checked at
          cross-sections over the level's height in the box, close enough that
          nothing narrows by more than half the threshold from one to the next
          (a part that vanishes, or a gap that closes, within the level is
          found too). Only the box counts: a width or gap is found where one
          of its two edges enters the box, and measured, and located, on the
          parts of its edges in the box. Port extensions that end at a box
          face are continued past it for the checks, so their artificial ends
          are not reported.
        - ``"overlap"`` (levels drawn with the composite model only; exact
          derived shapes draw the union): polygons of a source layer overlap
          with positive area (not only abut), and the level shrinks that layer somewhere in its
          height: a sidewall angle above ``z_ref``, a negative bias, z_to_bias
          or sizing, or a level that grows and subtracts the layer. A layer's
          distance is the minimum of its polygons' distances, so the level
          draws the union of the shrunk polygons, not their shrunk union,
          which can open slits where they overlap (where an overlap is
          narrower than about twice the shrink). Grows are exact.

        The checks run on concrete numpy data with KLayout (never tracers).
        :meth:`place_on_grid` warns with the message of every finding (a
        :class:`StackDiagnosticWarning`) unless ``diagnostics`` is False.

        Args:
            resolution: grid cell size (um); default: the placed device's.

        Returns:
            At most one finding per level and kind, in the order of ``levels``.

        Raises:
            ValueError: if the device is neither placed nor given a resolution,
                or has no box, or the resolution is not positive.
        """
        import klayout.db as kdb

        placed = self._grid_slice_tuple != INVALID_SLICE_TUPLE_3D
        if resolution is None:
            if not placed:
                raise ValueError(f"StackDevice {self.name!r} is not placed; pass the grid resolution (um)")
            # the larger lateral cell where x and y differ (the largest, graded): what the coarser one resolves
            if self._lateral_graded():
                e = self.grid_frame.edges
                resolution = self._res_um() * float(max(np.max(np.diff(e[0])), np.max(np.diff(e[1]))))
            else:
                resolution = self._res_um() * max(self._lateral_widths())
        cell = float(resolution)
        if not cell > 0:
            raise ValueError(f"resolution must be a positive length (um), got {resolution}")
        if placed:
            size = tuple(n * self._res_um() for n in self.grid_shape)
            if self.grid_frame is not None and not all(self.grid_frame.uniform):  # the box from its edges
                size = tuple(float(e[-1]) * self._res_um() for e in self.grid_frame.edges)
        elif self.partial_real_shape is None or None in tuple(self.partial_real_shape):
            raise ValueError(f"StackDevice {self.name!r} has no box (partial_real_shape)")
        else:
            size = tuple(float(s) * 1e6 for s in self.partial_real_shape)
        threshold = _MIN_FEATURE * cell / self.xy_subsamples
        z_box = (self.z_center - 0.5 * size[2], self.z_center + 0.5 * size[2])
        center = np.asarray(self.center, dtype=float)
        half = 0.5 * np.array(size[:2])

        # every level's literals (offsets in um) and the heights of its checked cross-sections
        source = {index: i for i, (_, index) in enumerate(self._slots())}
        plans = []
        reach = 0.0  # largest offset (um) of a literal
        for lv in self.levels:
            expr = self._drawn_expr(lv)
            if expr[0] == "fill":
                continue
            tree, literals = _literals(expr, 1.0, source)
            lo, hi = _span(lv)
            lo, hi = max(lo, z_box[0]), min(hi, z_box[1])
            if tree in (_TRUE, _FALSE) or not hi > lo:
                continue
            # a part narrows by twice the offset's change: at most half the threshold between heights
            heights = _check_heights(lv, lo, hi, 0.25 * threshold)
            offsets = _wall_offset(lv, heights)
            reach = max([reach, *(float(np.abs(a * offsets + b).max()) for _, a, b in literals)])
            plans.append((lv, tree, literals, heights, offsets))
        if not plans:
            return []

        # the polygons (um, from the device center); extensions that end near
        # or beyond a box face continue past it (their ends there are artificial)
        margin = threshold + reach + 2 * cell  # how far the plane the checks see reaches past the box
        extensions = self.source_extensions or tuple(() for _ in self.source_layers)
        polygons: dict[int, list[np.ndarray]] = {}
        for (_, index), v, sizes, exts in zip(
            self.source_layers, self.source_vertices, self.source_sizes, extensions, strict=True
        ):
            polys = np.split(np.asarray(v, dtype=float) - center, np.cumsum(sizes)[:-1])
            for rect in polys[len(polys) - len(exts) :]:
                along = rect[1] - rect[0]  # corners 0, 3 on the port face, 1, 2 at the end
                length = float(np.linalg.norm(along))
                if not length > 0:
                    continue
                n, far = along / length, 0.5 * (rect[1] + rect[2])
                axes = np.abs(n) > 1e-12
                exit_ = np.min((np.sign(n[axes]) * half[axes] - far[axes]) / n[axes])  # to the box face ahead
                if exit_ < margin:
                    rect[1:3] += (2 * float(half.sum()) + margin) * n
            polygons[index] = polys
        extent = max(float(np.abs(np.concatenate([np.concatenate(p) for p in polygons.values()])).max()),
                     *(half + margin))
        dbu = max(_CHECK_DBU, 4 * extent / 2**31)  # 32-bit coordinates, with room for the offsets

        def box_region(h: np.ndarray) -> Any:
            """The region |x| <= h[0], |y| <= h[1] (um)."""
            x, y = (round(float(c) / dbu) for c in h)
            return kdb.Region(kdb.Box(-x, -y, x, y))

        box, universe = box_region(half), box_region(half + margin)
        inside = (round(float(half[0]) / dbu), round(float(half[1]) / dbu))
        # the layers within reach of the universe decide its shapes (offsets reach at most `reach`)
        clip = box_region(half + 2 * margin)
        regions = {index: _kdb_region(polys, dbu) & clip for index, polys in polygons.items()}
        for k, d in enumerate(self.derived_layers):  # exact shapes, from the same (continued) polygons
            for j, r in enumerate(d.recipes):
                if len(r.kind):
                    pool = np.concatenate([np.concatenate(polygons[index]) for index in d.layers])
                    rings = recipe_rings(r, run_recipe(r.kind, r.index, r.positions - center, pool, np))
                    regions[_part_index(k, j)] = _kdb_fold_rings(rings, dbu) & clip
        limit = round(threshold / dbu)
        tol = max(1.0, _ROUND_TOL * limit)  # dbu: round joins are drawn this close to round
        corners: dict[int, tuple[tuple[np.ndarray, Any], tuple[np.ndarray, Any]]] = {}
        keys = {index: k for k, index in self.source_layers}
        overlaps: dict[int, tuple[int, float, np.ndarray] | None] = {}
        n_sub = f" / xy_subsamples {self.xy_subsamples}" if self.xy_subsamples > 1 else ""
        off = " (diagnostics=False turns these warnings off)"

        findings = []
        for lv, tree, literals, heights, offsets in plans:
            prefix = f"StackDevice {self.name!r}, level {lv.name!r}"
            # the layers offset with round joins, as the device draws them (see _round_offset)
            near = kdb.Edges()
            for index in sorted({index for index, _, _ in literals}):
                near += regions[index].edges()
                if index not in corners:
                    corners[index] = tuple((v, _seeds(v)) for v in _corners(regions[index]))
            near.merged_semantics = False
            cache: dict[tuple[int, int, int], tuple[Any, np.ndarray]] = {}
            joins: list[np.ndarray] = []  # of the shape being built
            other = [0]  # largest offset (dbu) of the level's layers at the height being built

            def literal(
                index: int, grow: int, near: Any = near, cache: dict = cache, joins: list = joins, other: list = other
            ) -> Any:
                """A layer grown by ``grow`` dbu with round joins (recorded in ``joins``)."""
                key = (index, grow, other[0])
                if key not in cache:
                    # a width or gap at a join is within the limit of the join, which is within |grow| of
                    # its corner, and of another boundary, which is within other[0] of its layer's edges
                    round_offset_fn = sys.modules["bifrost.stack"]._round_offset
                    cache[key] = round_offset_fn(
                        regions[index], grow, corners[index], near, limit + other[0] + math.ceil(tol) + 2, tol
                    )
                joins.append(cache[key][1])
                return cache[key][0]

            # parts and gaps narrower than the threshold, at every checked height
            found: dict[str, list[tuple[float, np.ndarray, np.ndarray, float]]] = {"width": [], "gap": []}
            seen = set()
            for z, s in zip(heights, offsets, strict=True):
                if round(s / dbu) in seen:
                    continue
                seen.add(round(s / dbu))
                joins.clear()
                other[0] = max(abs(round(-(a * s + b) / dbu)) for _, a, b in literals)
                shape = _tree_region(tree, literals, literal, float(s), universe, dbu)
                drawn = np.concatenate(joins) if joins else np.zeros((0, 3))
                for kind, check in (("width", shape.width_check), ("gap", shape.space_check)):
                    pairs = check(limit, ignore_angle=_CHECK_ANGLE)
                    places = _narrow_places(pairs, inside, limit, shape, drawn, tol)
                    found[kind] += [(d, c, e, float(z)) for d, c, e in places]
            for kind, places in found.items():
                if not places:
                    continue
                # the narrowest place, or the longest one within the joins' tolerance of it
                least = min(p[0] for p in places)
                d, c, _, z = max(
                    (p for p in places if p[0] <= least + tol), key=lambda p: float(np.hypot(*(p[2][1] - p[2][0])))
                )
                spots = kdb.Region()
                for _, _, e, _ in places:
                    (x0, y0), (x1, y1) = np.floor(e[0]).astype(np.int64), np.ceil(e[1]).astype(np.int64)
                    spots.insert(kdb.Box(int(x0) - 1, int(y0) - 1, int(x1) + 1, int(y1) + 1))
                count = spots.merged().count()
                size = d * dbu
                location = (float(center[0] + c[0] * dbu), float(center[1] + c[1] * dbu))
                what = ("part", "parts") if kind == "width" else ("gap", "gaps")
                findings.append(StackDiagnostic(
                    level=lv.name,
                    kind=kind,
                    size=size,
                    cells=size / cell,
                    threshold=threshold,
                    location=location,
                    z=z,
                    count=count,
                    layers=(),
                    message=(
                        f"{prefix}: {count} {what[count > 1]} narrower than {threshold:.4g} um "
                        f"({_MIN_FEATURE:g} cells{n_sub}); the narrowest is {size:.4g} um ({size / cell:.3g} "
                        f"cells) wide, at ({location[0]:.4f}, {location[1]:.4f}) um and z = {z:.4f} um. The grid "
                        f"does not resolve it: refine the grid, raise xy_subsamples, or widen the {what[0]}{off}"
                    ),
                ))

            # overlapping polygons on the layers the level shrinks
            shrink: dict[int, tuple[float, float]] = {}  # largest shrink (um) of each layer and its height
            for index, a, b in literals:
                if index < 0:
                    continue  # an exact shape: its distance is the offset of its union
                r = a * offsets + b
                k = int(np.argmax(r))
                if r[k] > dbu and r[k] > shrink.get(index, (0.0, 0.0))[0]:
                    shrink[index] = (float(r[k]), float(heights[k]))
            hits = {}
            for index in shrink:
                if index not in overlaps:
                    overlaps[index] = _overlaps(polygons[index], dbu, box)
                if overlaps[index] is not None:
                    hits[index] = overlaps[index]
            if hits:
                worst = max(hits, key=lambda i: shrink[i][0])
                size, z = shrink[worst]
                count = sum(h[0] for h in hits.values())
                area = sum(h[1] for h in hits.values())
                c = center + max(hits.values(), key=lambda h: h[1])[2]
                location = (float(c[0]), float(c[1]))
                layers = tuple(keys[i] for i in hits)
                findings.append(StackDiagnostic(
                    level=lv.name,
                    kind="overlap",
                    size=size,
                    cells=size / cell,
                    threshold=0.0,
                    location=location,
                    z=z,
                    count=count,
                    layers=layers,
                    message=(
                        f"{prefix} shrinks {', '.join(layers)} by up to {size:.4g} um ({size / cell:.3g} cells; "
                        f"sidewall angle, bias, z_to_bias or sizing), and {'its' if len(layers) == 1 else 'their'} "
                        f"polygons overlap at {count} {'place' if count == 1 else 'places'} in the box "
                        f"({area:.4g} um^2 in all; e.g. at ({location[0]:.4f}, {location[1]:.4f}) um). The level "
                        "draws the union of the shrunk polygons, not their shrunk union, which can open slits where "
                        "they overlap (where an overlap is narrower than about twice the shrink). Make the polygons "
                        f"abut instead of overlap, or merge them{off}"
                    ),
                ))
        return findings

    # ------------------------------------------------------------ geometry

    def _res_um(self) -> float:
        """The frame unit in um (:class:`_Frame`): the grid resolution on a uniform grid."""
        if self.grid_frame is not None:
            return self.grid_frame.h_um
        return self._config.uniform_spacing() * 1e6

    def _lateral_widths(self) -> tuple[float, float]:
        """(wx, wy): the width (frame units) of the device's cells along x and y (1 and 1 on a uniform grid)."""
        if self.grid_frame is None:
            return 1.0, 1.0
        wx, wy = self.grid_frame.widths[:2]
        if wx is None or wy is None:
            raise NotImplementedError(f"StackDevice {self.name!r}: graded x or y cells are painted per column")
        return wx, wy

    def _lateral_graded(self) -> bool:
        """Whether the device's x or y cells are graded (painted per column, see :class:`_Cells`)."""
        return self.grid_frame is not None and None in self.grid_frame.widths[:2]

    def _sample_axes(self, shift: tuple[float, float]) -> tuple[_SampleAxis, _SampleAxis]:
        """The x and y sample axes of the grid whose points are ``shift`` cells (0 or -0.5) from the centers."""
        return self.grid_frame.sample_axis(0, shift[0]), self.grid_frame.sample_axis(1, shift[1])

    def _sample_cells(self, shift: tuple[float, float]) -> _Cells:
        """The lateral cells (dense (nx, ny), the device's dtype) of the sample grid at the Yee shift (sx, sy).

        On a grid with graded x or y cells: each column's sample box (the cell, or the dual cell between
        neighbouring cell centers along a shifted axis), ``wx`` by ``wy`` frame units, as the rectangle of area 1
        with half sides ``(sqrt(wx / wy), sqrt(wy / wx)) / 2`` in a frame of ``sqrt(wx wy) edge_width /
        xy_subsamples`` frame units per side (see :meth:`_lateral_cell`).
        """
        xa, ya = self._sample_axes(shift)
        wx, wy = (xa.hi - xa.lo)[:, None], (ya.hi - ya.lo)[None, :]
        r = np.sqrt(wx / wy)
        hx, hy = 0.5 * r, 0.5 / r
        cell = np.sqrt(wx * wy) * self.edge_width / self.xy_subsamples
        dtype = self._config.dtype
        return _Cells(
            jnp.asarray(hx, dtype=dtype), jnp.asarray(hy, dtype=dtype), jnp.asarray(cell, dtype=dtype),
            2.0 * float(max(np.max(hx), np.max(hy))),
        )

    def _lateral_cell(self) -> tuple[Any, float]:
        """(half, cell): the lateral cell a sample is painted on, and frame units per side of it.

        Where the x and y cells are one size: the square of half side 0.5, ``cell = edge_width / xy_subsamples``
        (as always). Where they differ (``wx``, ``wy`` frame units): the rectangle of the same aspect and area 1,
        half sides ``(sqrt(wx / wy), sqrt(wy / wx)) / 2``, in an isotropic frame of ``cell = sqrt(wx wy)
        edge_width / xy_subsamples`` frame units per unit, so that distances, normals and round joins stay
        Euclidean (see :func:`_cell_corners`).
        """
        wx, wy = self._lateral_widths()
        if wx == wy:
            return 0.5, self.edge_width / self.xy_subsamples
        r = math.sqrt(wx / wy)
        return (0.5 * r, 0.5 / r), math.sqrt(wx * wy) * self.edge_width / self.xy_subsamples

    def _column_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """The x and y centers (frame units) of the device's columns."""
        e = self.grid_frame.edges
        return 0.5 * (e[0][:-1] + e[0][1:]), 0.5 * (e[1][:-1] + e[1][1:])

    def _surface_slopes(self, z: Array) -> Array:
        """(nx, ny, 2) lateral gradient of a surface z (frame units per frame unit); 0 along an axis of one cell."""
        if self._lateral_graded():  # over the columns' centers
            centers = self._column_centers()
            return jnp.stack(
                [jnp.gradient(z, jnp.asarray(centers[a], dtype=z.dtype), axis=a) if z.shape[a] > 1
                 else jnp.zeros_like(z) for a in (0, 1)],
                axis=-1,
            )
        wx, wy = self._lateral_widths()
        if wx == wy:
            return jnp.stack(
                [jnp.gradient(z, axis=a) if z.shape[a] > 1 else jnp.zeros_like(z) for a in (0, 1)], axis=-1
            )
        return jnp.stack(
            [jnp.gradient(z, axis=a) / w if z.shape[a] > 1 else jnp.zeros_like(z) for a, w in ((0, wx), (1, wy))],
            axis=-1,
        )

    def _fd_spacing(self, shift: tuple[float, float, float]) -> tuple[np.ndarray | None, ...] | None:
        """Distances between neighbouring sample centers for :func:`_normal_squared` (None: one cell everywhere).

        Args:
            shift: the shift (cells) of the sample grid (its Yee position).

        Returns:
            Per axis None (one cell) or the (n + 1,) distances in frame units; None where all are one cell.
        """
        nx, ny, _ = self.grid_shape
        zaxis = self._z_axis(shift[2])
        z = None if zaxis.uniform else _center_distances(zaxis)
        if self._lateral_graded():
            return (*(_center_distances(a) for a in self._sample_axes((shift[0], shift[1]))), z)
        wx, wy = self._lateral_widths()
        if wx == wy and zaxis.uniform:
            return None
        return (
            None if wx == wy else np.full(nx + 1, wx),
            None if wx == wy else np.full(ny + 1, wy),
            z,
        )

    def _to_square(self) -> float:
        """Frame units -> sides of a sample's lateral cell (1 / cell of :meth:`_lateral_cell`)."""
        wx, wy = self._lateral_widths()
        if wx == wy:
            return self.xy_subsamples / self.edge_width
        return 1.0 / self._lateral_cell()[1]

    def _z_axis(self, z_shift: float) -> _SampleAxis:
        """The z axis of the sample grid whose points are ``z_shift`` cells from the voxel centers.

        Args:
            z_shift: 0 (voxel centers) or -0.5 (Ex, Ey Yee positions).

        Returns:
            The sample axis (:meth:`_Frame.sample_axis`); without a frame, the
            uniform voxel layers.
        """
        if self.grid_frame is None:
            lo = np.arange(self.grid_shape[2], dtype=float)
            return _SampleAxis(float(z_shift), lo, lo + 1.0, True)
        return self.grid_frame.sample_axis(2, z_shift)

    def interface_offsets(self, key: Array | None) -> list[Array]:
        """Samples every roughness entry.

        Args:
            key: PRNG key (default: ``PRNGKey(noise_seed)``).

        Returns:
            One (nx, ny) height offset in cells per entry of ``roughness``.
        """
        nx, ny, _ = self.grid_shape
        if key is None:
            key = jax.random.PRNGKey(self.noise_seed)
        if not self.roughness:
            return []
        if self._lateral_graded():
            return [self._graded_offsets(r, jax.random.fold_in(key, i)) for i, r in enumerate(self.roughness)]
        wx, wy = self._lateral_widths()
        widths = {} if wx == wy else {"widths": (wx, wy)}
        return [
            r.sample((nx, ny), self._res_um(), jax.random.fold_in(key, i), **widths)
            for i, r in enumerate(self.roughness)
        ]

    def _graded_offsets(self, roughness: "TopRoughness", key: Array) -> Array:
        """One rough surface on graded x or y cells: (nx, ny) height offsets (frame units) at the columns.

        The field is drawn on an auxiliary lattice of one frame unit (the finest lateral cell) over the box and
        interpolated bilinearly to the columns' centers (which smooths it a little in coarse cells); without a
        correlation length it is white noise per column, as on a uniform grid.
        """
        nx, ny, _ = self.grid_shape
        if not roughness.correlation_length > 0:
            return roughness.sample((nx, ny), self._res_um(), key)
        e = self.grid_frame.edges
        lattice = (max(2, math.ceil(e[0][-1] - 1e-9)), max(2, math.ceil(e[1][-1] - 1e-9)))
        f = roughness.sample(lattice, self._res_um(), key)  # lattice point (i, j) at (i + 0.5, j + 0.5)
        index, weight = [], []
        for c, n in zip(self._column_centers(), lattice, strict=True):
            i = np.clip(np.floor(c - 0.5).astype(int), 0, n - 2)
            index.append(i)
            weight.append(jnp.asarray(np.clip(c - 0.5 - i, 0.0, 1.0), dtype=f.dtype))
        (ix, iy), (tx, ty) = index, weight
        tx, ty = tx[:, None], ty[None, :]
        return (
            (1 - tx) * (1 - ty) * f[ix[:, None], iy[None, :]] + tx * (1 - ty) * f[ix[:, None] + 1, iy[None, :]]
            + (1 - tx) * ty * f[ix[:, None], iy[None, :] + 1] + tx * ty * f[ix[:, None] + 1, iy[None, :] + 1]
        )

    @timed("layer_geometry")
    def _layer_geometry(self, params: dict[str, Array]) -> list[_LayerGeometry]:
        """Every source layer's polygons in grid cells, with their static edge structure.

        Partly shared junction edges are split first (static topology, from
        ``source_vertices`` within the topology tolerance, ``topology_tol``; see
        :meth:`check_topology`), so abutting polygons, such as a port extension
        and its waveguide or a taper on an MMI side, join without a seam.

        Args:
            params: device parameters (``vertices/<layer>`` in um).

        A derived shape (``derived_layers``) is one polygon whose vertices its
        recipe evaluates from its layers' vertices (:func:`run_recipe`); its
        zero-length joints and glue cuts are left out of the distance.

        Returns:
            One entry per layer slot (:meth:`_slots`: the source layers, then
            the nonempty derived shapes); the boundary links only with
            ``corner_lines``.
        """
        nx, ny, _ = self.grid_shape
        res_um = self._res_um()
        if self._lateral_graded():  # the box's middle from its edges (frame units)
            offset = np.array([self.grid_frame.edges[0][-1] / 2, self.grid_frame.edges[1][-1] / 2])
        else:
            wx, wy = self._lateral_widths()
            offset = np.array([nx / 2, ny / 2]) if wx == wy else np.array([nx * wx / 2, ny * wy / 2])
        center = np.asarray(self.center, dtype=np.float64)
        dtype = self.dtype

        def to_cells(coords: Array | np.ndarray) -> Array:
            if has_tracers(coords):
                calc = jnp.float64 if jax.config.jax_enable_x64 else dtype
                c = jnp.asarray(center, dtype=calc)
                o = jnp.asarray(offset, dtype=calc)
                return ((jnp.asarray(coords, dtype=calc) - c) / calc(res_um) + o).astype(dtype)
            local = (np.asarray(to_numpy(coords), dtype=np.float64) - center) / res_um + offset
            return jnp.asarray(local, dtype=dtype)

        out = []
        keys = {index: k for k, index in self.source_layers}
        for d in self.derived_layers:
            pool = None
            for r in d.recipes:
                if not len(r.kind):
                    continue
                if pool is None:
                    pool = jnp.concatenate([to_cells(params[keys[index]]) for index in d.layers])
                topology = (np.asarray(r.positions, dtype=float) - center) / res_um + offset
                v = run_recipe(r.kind, r.index, topology, pool)
                tol = d.tol / res_um
                sizes = (len(topology),)
                nxt, ids, boundary = _edge_topology(topology, sizes, tol)
                links = (None, None, None)
                if self.corner_lines > 0:
                    links = _boundary_links(topology, sizes, nxt, ids, boundary, tol)
                out.append(_LayerGeometry(v, nxt, ids, boundary, *links))
        derived = out
        out = []
        for (k, _), v0, sizes in zip(self.source_layers, self.source_vertices, self.source_sizes, strict=True):
            topology = (np.asarray(v0, dtype=float) - center) / res_um + offset
            tol = self._topology_tol(v0) / res_um  # cells
            # partly shared junction edges -> exactly shared seams (static)
            split, split_sizes = split_t_junctions(topology, sizes, tol)
            v = to_cells(params[k])[split]
            nxt, ids, boundary = _edge_topology(topology[split], split_sizes, tol)
            links = (None, None, None)
            if self.corner_lines > 0:
                links = _boundary_links(topology[split], split_sizes, nxt, ids, boundary, tol)
            out.append(_LayerGeometry(v, nxt, ids, boundary, *links))
        return out + derived

    def _sample_points(self, shift: tuple[float, float] = (0.0, 0.0)) -> Array:
        """Lateral sample points (nx * ny, 2) in frame units: the voxel centers moved by ``shift`` cells, i-major.

        With graded x or y cells ``shift`` is (sx, sy, a, b): the Yee shift (0 or -0.5) of the sample grid, whose
        points are its sample boxes' centers, then a sub-sample's offset in widths of its box.
        """
        nx, ny, _ = self.grid_shape
        if self._lateral_graded():
            sx, sy, a, b = shift if len(shift) == 4 else (*shift, 0.0, 0.0)
            xa, ya = self._sample_axes((sx, sy))
            cx = 0.5 * (xa.lo + xa.hi) + xa.shift + a * (xa.hi - xa.lo)
            cy = 0.5 * (ya.lo + ya.hi) + ya.shift + b * (ya.hi - ya.lo)
            xx, yy = jnp.meshgrid(jnp.asarray(cx), jnp.asarray(cy), indexing="ij")
            return jnp.stack([xx.ravel(), yy.ravel()], axis=1)
        wx, wy = self._lateral_widths()
        if wx != wy:  # cells of wx and wy frame units
            xx, yy = jnp.meshgrid(
                (jnp.arange(nx) + 0.5 + shift[0]) * wx, (jnp.arange(ny) + 0.5 + shift[1]) * wy, indexing="ij"
            )
            return jnp.stack([xx.ravel(), yy.ravel()], axis=1)
        xx, yy = jnp.meshgrid(
            jnp.arange(nx) + 0.5 + shift[0], jnp.arange(ny) + 0.5 + shift[1], indexing="ij"
        )
        return jnp.stack([xx.ravel(), yy.ravel()], axis=1)

    @timed("source_sdfs")
    def _source_sdfs(
        self,
        params: dict[str, Array],
        shift: tuple[float, float] = (0.0, 0.0),
        geometry: Sequence[_LayerGeometry] | None = None,
    ) -> dict[int, Field]:
        """Signed distance of every source layer at the voxel centers.

        Args:
            params: device parameters (``vertices/<layer>`` in um).
            shift: offset (cells) of the sample points from the voxel centers.
            geometry: :meth:`_layer_geometry` of ``params``, if already known.

        Returns:
            (distance, gx, gy) per layer slot index (:meth:`_slots`) that a
            level is drawn with, (nx, ny) each, in cells, negative inside;
            (gx, gy) is the gradient with respect to the point.
        """
        nx, ny, _ = self.grid_shape
        if geometry is None:
            geometry = self._layer_geometry(params)
        read = set().union(*(_expr_layers(self._drawn_expr(lv)) for lv in self.levels))
        sdf = {}
        for (_, index), (v, nxt, ids, boundary, *_) in zip(self._slots(), geometry, strict=True):
            if index not in read:
                continue
            points = self._sample_points(shift).astype(v.dtype)

            def distance(q: Array, v: Array = v, nxt: np.ndarray = nxt, ids: np.ndarray = ids,
                         boundary: np.ndarray = boundary) -> Array:
                """Signed distance (cells) at the points q (P, 2)."""
                return signed_distance(q, v, v[nxt], ids, boundary, self.edge_chunk)

            # value and gradient with respect to the sample point (forward mode)
            tangents = jnp.stack([jnp.zeros_like(points).at[:, a].set(1.0) for a in (0, 1)])
            sd, grad = jax.vmap(
                lambda t, f=distance: jax.jvp(f, (points,), (t,)), out_axes=(None, 0)
            )(tangents)
            sdf[index] = (sd.reshape(nx, ny), grad[0].reshape(nx, ny), grad[1].reshape(nx, ny))
        return sdf

    @timed("level_surfaces")
    def _level_surfaces(
        self, p: dict[str, Array], noise_key: Array | None, apply_roughness: bool, z_shift: float
    ) -> tuple[list[tuple[_Level, tuple[Array, Array, Array] | None]], list[Array], list[Array]]:
        """The levels by priority, with their sidewalls and (rough) surfaces.

        Args:
            p: all device parameters.
            noise_key: PRNG key for the roughness.
            apply_roughness: if False, the nominal geometry.
            z_shift: offset (cells) of the sampled voxels from the grid's voxels along z
                (the ``shift`` of :meth:`_z_axis`: 0 on a graded axis, whose
                sample voxels are the layers' bounds instead).

        Returns:
            active: (level, its (zmin, thickness, angle) for
                :func:`_sidewall_shift`, or None for fill levels) by increasing
                ``mesh_order``.
            lows: (nx, ny) bottom surface of each level (cells).
            highs: (nx, ny) top surface of each level (cells).
        """
        nx, ny, nz = self.grid_shape
        res_um = self._res_um()
        frame = self.grid_frame
        if frame is None or frame.uniform[2]:
            z0 = self.z_center - 0.5 * nz * res_um  # stack z at the device bottom
        else:  # the box's height from the graded cells
            z0 = self.z_center - 0.5 * float(frame.edges[2][-1]) * res_um
        offsets_z = self.interface_offsets(noise_key) if apply_roughness and self.roughness else []
        dtype = self._config.dtype
        order = sorted(range(len(self.levels)), key=lambda i: (self.levels[i].mesh_order, i))
        tops = {lv.name: max(lv.zmin, lv.zmin + lv.thickness) for lv in self.levels}
        active = []  # (level, (zmin, thickness, angle) of its sidewall or None for fills)
        lows, highs = [], []
        for i in order:
            lv = self.levels[i]
            # every level spans the whole box height: which voxels it fills
            # comes from its (traced) surfaces alone, so zmin, thickness and
            # roughness may move it anywhere in the box
            zb = (p[f"zmin/{lv.name}"] - z0) / res_um - z_shift
            t = p[f"thickness/{lv.name}"] / res_um
            angle = p[f"sidewall_angle/{lv.name}"]

            # surfaces (cells), with roughness shared per interface
            z_lo, z_hi = (zb, zb + t) if lv.thickness > 0 else (zb + t, zb)
            lo_nominal = lv.zmin if lv.thickness > 0 else lv.zmin + lv.thickness
            z_lo = jnp.broadcast_to(z_lo, (nx, ny))
            z_hi = jnp.broadcast_to(z_hi, (nx, ny))
            for r, dz in zip(self.roughness if offsets_z else (), offsets_z, strict=True):
                if r.level == lv.name:
                    z_hi = z_hi + dz
                elif r.fill_above and r.level in tops and abs(lo_nominal - tops[r.level]) < 1e-6:
                    z_lo = z_lo + dz

            # the sidewall (angle, bias, z_to_bias) is evaluated in _composite at
            # the height of each part of a voxel the level fills
            wall = None if self._drawn_expr(lv)[0] == "fill" else (zb, t, angle)
            active.append((lv, wall))
            lows.append(z_lo.astype(dtype))
            highs.append(z_hi.astype(dtype))
        return active, lows, highs

    @timed("corner_plan")
    def _corner_plan(
        self,
        active: Sequence[tuple[_Level, Any]],
        lows: Sequence[Array],
        highs: Sequence[Array],
        zaxis: _SampleAxis | None = None,
    ) -> tuple[_CornerPlan | None, list[list[Array]]]:
        """The levels' literals for the corner model, and how far from a column each matters.

        Args:
            active: the levels by priority (:meth:`_level_surfaces`).
            lows: (nx, ny) bottom surface of each level (cells).
            highs: (nx, ny) top surface of each level (cells).
            zaxis: the z axis of the sample grid (:meth:`_z_axis`); None: the
                voxel layers [k, k + 1].

        Returns:
            plan: the literals, or None without ``corner_lines`` or when no level
                reads a source layer.
            radii: per level, per literal: the distance (cells, no gradient)
                within which its boundary can cut the lateral square somewhere
                in the box: the square's half-diagonal plus the largest offset
                of the literal over the heights the level covers. On graded x
                or y without the half-diagonal, which is each column's own
                (:meth:`_graded_radii` adds it per sample grid).
        """
        if self.corner_lines <= 0:
            return None, []
        nz = self.grid_shape[2]
        z_span = (0.0, nz) if zaxis is None or zaxis.uniform else (float(zaxis.lo[0]), float(zaxis.hi[-1]))
        res_um = self._res_um()
        if self._lateral_graded():
            half_diagonal = 0.0
        else:
            # with sub-samples, _VOXEL_RAMP more: a sample's certificate for its voxel
            # is 1 where its columns end (see _composite_samples)
            half, cell = self._lateral_cell()
            reach = _cell_diagonal(half) + _CORNER_TOL + (_VOXEL_RAMP if self.xy_subsamples > 1 else 0.0)
            half_diagonal = (
                reach * self.edge_width / self.xy_subsamples if not isinstance(half, tuple) else reach * cell
            )
        position = {index: i for i, (_, index) in enumerate(self._slots())}
        trees, literals, radii, layers = [], [], [], []
        for (lv, wall), lo, hi in zip(active, lows, highs, strict=True):
            if wall is None:
                trees.append(None)
                literals.append(())
                radii.append([])
                continue
            tree, lits = _literals(self._drawn_expr(lv), res_um, position)
            s_lo, s_hi = _lateral_range(lv, wall, lo, hi, z_span, res_um)
            slots, rad = [], []
            for index, a, b in lits:
                if position[index] not in layers:
                    layers.append(position[index])
                slots.append((layers.index(position[index]), a, b))
                rad.append(half_diagonal + jnp.maximum(jnp.abs(a * s_lo + b), jnp.abs(a * s_hi + b)))
            trees.append(tree)
            literals.append(tuple(slots))
            radii.append(rad)
        if not layers:
            return None, []
        return _CornerPlan(trees=tuple(trees), literals=tuple(literals), layers=tuple(layers)), radii

    def _graded_radii(self, radii: Sequence[Sequence[Array]], cells: _Cells) -> list[list[Array]]:
        """:meth:`_corner_plan`'s radii on graded x or y, with each column's half-diagonal: (nx * ny,) each.

        Args:
            radii: per level, per literal, the radius without the half-diagonal (cells).
            cells: the sample grid's lateral cells (:meth:`_sample_cells`).
        """
        reach = _cell_diagonal(cells) + _CORNER_TOL + (_VOXEL_RAMP if self.xy_subsamples > 1 else 0.0)
        half_diagonal = jnp.reshape(reach * cells.cell, (-1,))  # frame units
        return [[r + half_diagonal for r in rad] for rad in radii]

    @timed("corner_capacity")
    def _corner_capacity(self, plan: _CornerPlan, geometry: Sequence[_LayerGeometry]) -> int:
        """Static number of corner columns per sample grid.

        ``corner_columns`` once the device is placed; otherwise a bound from the
        boundary vertices and the levels' nominal offsets, with a warning.

        Args:
            plan: the levels' literals.
            geometry: the source layers (:meth:`_layer_geometry`).

        Returns:
            The number of columns (at most nx * ny).
        """
        nx, ny, _ = self.grid_shape
        if self.corner_columns is not None:
            return max(0, min(nx * ny, int(self.corner_columns)))
        warnings.warn(
            f"StackDevice {self.name!r} has no corner_columns (it is set when the device is placed); "
            "using a static bound, which may be much larger than needed",
            stacklevel=3,
        )
        res_um = self._res_um()
        offsets = [0.0] * len(plan.layers)  # bound on the literals' offsets (cells), per layer slot
        for lv, literals in zip(sorted(self.levels, key=lambda lv: lv.mesh_order), plan.literals, strict=True):
            t = abs(lv.thickness) / res_um
            biases = [abs(lv.bias)] + [abs(b) for b in (lv.z_to_bias[1] if lv.z_to_bias else ())]
            wall = (1 + abs(lv.width_to_z)) * t * abs(math.tan(math.radians(lv.sidewall_angle)))
            for layer, _, b in literals:
                offsets[layer] = max(offsets[layer], abs(b) + wall + max(biases) / res_um)
        if self._lateral_graded():  # the largest cell's
            cells = self._sample_cells((0.0, 0.0))
            reach = _cell_diagonal(cells) + (_VOXEL_RAMP if self.xy_subsamples > 1 else 0.0)
            half_diagonal = float(jnp.max(reach * cells.cell))
        else:
            half, cell = self._lateral_cell()
            reach = _cell_diagonal(half) + (_VOXEL_RAMP if self.xy_subsamples > 1 else 0.0)
            half_diagonal = (
                reach * self.edge_width / self.xy_subsamples if not isinstance(half, tuple) else reach * cell
            )
        bound = 64
        for layer, offset in zip(plan.layers, offsets, strict=True):
            vertices = int(np.count_nonzero(geometry[layer].boundary))
            bound += vertices * math.ceil(math.pi * (half_diagonal + offset + 1.5) ** 2)
        return min(nx * ny, bound)

    @timed("corner_inputs")
    def _corner_inputs(
        self,
        plan: _CornerPlan,
        radii: Sequence[Sequence[Array]],
        geometry: Sequence[_LayerGeometry],
        sdf: dict[int, Field],
        shift: tuple[float, float],
        capacity: int,
    ) -> _CornerInputs:
        """The corner columns of one lateral sample grid and their local models.

        Args:
            plan: the levels' literals.
            radii: per level, per literal, the radius of :meth:`_corner_plan`.
            geometry: the source layers (:meth:`_layer_geometry`).
            sdf: the source layers' distances on this grid (:meth:`_source_sdfs`).
            shift: offset (cells) of the grid's points from the voxel centers; on graded x or y the Yee
                shift and the sub-sample's offset, (sx, sy, a, b) (see :meth:`_sample_points`).
            capacity: number of corner columns (static).

        Returns:
            The corner columns for :func:`_composite`.
        """
        nx, ny, _ = self.grid_shape
        num = nx * ny
        graded_xy = self._lateral_graded()
        if graded_xy:  # each column's own cell
            cells = self._sample_cells((shift[0], shift[1]))
            radii = self._graded_radii(radii, cells)
            sc = None
        else:
            sc = self._to_square()
        near = []
        base_points = self._sample_points(shift)
        points_by_dtype: dict[Any, Array] = {}

        def sample_points(dt: Any) -> Array:
            if dt not in points_by_dtype:
                points_by_dtype[dt] = base_points.astype(dt)
            return points_by_dtype[dt]

        for layer in plan.layers:
            v, _, _, boundary, src, dst, _ = geometry[layer]
            points = sample_points(v.dtype)
            near.append(nearest_edges(points, v[src], v[dst], boundary, self.edge_chunk))
        candidate = _corner_candidates(plan, radii, [d for d, _ in near])
        # the first candidates in raster order; unused slots hold nx * ny
        cols = jnp.nonzero(candidate, size=capacity, fill_value=num)[0]
        dropped = candidate & (jnp.cumsum(candidate) > capacity)
        cs = jnp.minimum(cols, num - 1)
        taken_cells = _cells_take(cells, cs) if graded_xy else None
        slots = self._slots()
        models = []
        for layer, (dist, ids) in zip(plan.layers, near, strict=True):
            v, _, pid, _, src, dst, succ = geometry[layer]
            sd, gx, gy = sdf[slots[layer][1]]
            points = sample_points(v.dtype)
            normal = _unit_normal(gx, gy).reshape(num, 2)[cs]
            to_square = 1.0 / taken_cells.cell if graded_xy else sc
            models.append(
                _layer_corner(
                    v, src, dst, succ, pid, ids[cs], dist[cs], points[cs], sd.reshape(num)[cs], normal, to_square
                )
            )
        dtype = self._config.dtype  # like the half-plane painting (the distances may be wider)
        layers = jax.tree.map(
            lambda *x: jnp.stack(x).astype(dtype) if jnp.issubdtype(x[0].dtype, jnp.floating) else jnp.stack(x), *models
        )
        return _CornerInputs(
            cols=cols,
            taken=cols < num,
            layers=layers,
            trees=plan.trees,
            literals=plan.literals,
            lines=self.corner_lines,
            sc=sc,
            dropped=dropped,
            cells=taken_cells,
        )

    def _corner_candidate_count(self) -> int:
        """Most corner candidates of any lateral sample grid, at the initial parameters.

        Covers every grid :meth:`material_weights` samples for :meth:`write`
        and fdtdx's initial arrays: the voxel centers and the Yee positions,
        each with its xy sub-samples (the default roughness realization).

        Returns:
            The number of candidate columns.
        """
        n_sub = self.xy_subsamples
        sub = [(k + 0.5) / n_sub - 0.5 for k in range(n_sub)]
        shifts = [(0.0, 0.0, 0.0), *(_YEE_E_SHIFTS if self.yee_staggered else ())]
        if self.grid_frame is not None and (not self.grid_frame.uniform[2] or self._lateral_graded()):
            return self._graded_candidate_count(shifts, sub)

        @jax.jit
        def count(p: dict[str, Array], shift: Array) -> Array:
            """Candidates of the grid whose points are moved by shift (3,) from the voxel centers."""
            plan, radii = self._corner_plan(*self._level_surfaces(p, None, True, shift[2]))
            if plan is None:
                return jnp.zeros((), dtype=jnp.int32)
            geometry = self._layer_geometry(p)
            dist = []
            for layer in plan.layers:
                v, _, _, boundary, src, dst, _ = geometry[layer]
                points = self._sample_points((shift[0], shift[1])).astype(v.dtype)
                dist.append(nearest_edges(points, v[src], v[dst], boundary, self.edge_chunk)[0])
            return jnp.sum(_corner_candidates(plan, radii, dist), dtype=jnp.int32)

        p = self.init_params()
        return max(int(count(p, jnp.asarray([x + a, y + b, z]))) for x, y, z in shifts for a in sub for b in sub)

    def _graded_candidate_count(self, shifts: Sequence[tuple[float, float, float]], sub: Sequence[float]) -> int:
        """:meth:`_corner_candidate_count` on a graded grid: one compiled count per z sample axis (and on graded
        x or y per lateral Yee shift, whose cells are static).

        Args:
            shifts: the sample grids' shifts (cells) from the voxel centers.
            sub: the lateral sub-sample offsets (cells).

        Returns:
            The number of candidate columns.
        """
        graded_xy = self._lateral_graded()

        def counter(zaxis: _SampleAxis, yee: tuple[float, float] | None) -> Callable[[dict[str, Array], Array], Array]:
            """The compiled count for the grids on this z axis (static: its layers' bounds) and, on graded x or y,
            at this lateral Yee shift (static: its cells)."""

            @jax.jit
            def count(p: dict[str, Array], xy: Array) -> Array:
                """Candidates of the grid whose points are moved by xy (2,) from the voxel centers (graded x or
                y: from the Yee positions, the sub-sample's offset)."""
                plan, radii = self._corner_plan(*self._level_surfaces(p, None, True, zaxis.shift), zaxis)
                if plan is None:
                    return jnp.zeros((), dtype=jnp.int32)
                grid = (xy[0], xy[1]) if yee is None else (*yee, xy[0], xy[1])
                if yee is not None:
                    radii = self._graded_radii(radii, self._sample_cells(yee))
                geometry = self._layer_geometry(p)
                dist = []
                for layer in plan.layers:
                    v, _, _, boundary, src, dst, _ = geometry[layer]
                    points = self._sample_points(grid).astype(v.dtype)
                    dist.append(nearest_edges(points, v[src], v[dst], boundary, self.edge_chunk)[0])
                return jnp.sum(_corner_candidates(plan, radii, dist), dtype=jnp.int32)

            return count

        p = self.init_params()
        most = 0
        for z in sorted({s[2] for s in shifts}):
            grids = [(x, y) for x, y, zz in shifts if zz == z]
            if graded_xy:
                for x, y in grids:
                    count = counter(self._z_axis(z), (x, y))
                    most = max(most, *(int(count(p, jnp.asarray([a, b]))) for a in sub for b in sub))
                continue
            count = counter(self._z_axis(z), None)
            most = max(most, *(int(count(p, jnp.asarray([x + a, y + b]))) for x, y in grids for a in sub for b in sub))
        return most

    def material_weights(
        self,
        params: dict[str, Array] | None = None,
        noise_key: Array | None = None,
        apply_roughness: bool = True,
        shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> tuple[list[str], Array]:
        """Volume fraction of every material in every voxel.

        Args:
            params: device parameters (defaults: :meth:`init_params`).
            noise_key: PRNG key for the roughness.
            apply_roughness: if False, the nominal geometry.
            shift: offset (cells) of the sampled voxels from the grid's voxels,
                e.g. to a Yee field position.

        Returns:
            names: material names in fdtdx order (``compute_ordered_names``).
            weights: (num_materials, nx, ny, nz), summing to one over materials.
        """
        return self._weights_and_normals(params, noise_key, apply_roughness, shift)[:2]

    def _half_planes(
        self,
        active: Sequence[tuple[_Level, Any]],
        is_fill: Sequence[bool],
        sdf: dict[int, Field],
        to_square: Array | float,
        dtype: jnp.dtype,
    ) -> tuple[Array, Array]:
        """(nx, ny, K) offsets (sides of the square) and (nx, ny, K, 2) normals of the levels, by priority."""
        nx, ny, _ = self.grid_shape
        res_um = self._res_um()
        offsets, normals = [], []
        for (lv, _), fill in zip(active, is_fill, strict=True):
            if fill:
                offsets.append(jnp.full((nx, ny), -_BIG, dtype=dtype))
                normals.append(jnp.zeros((nx, ny, 2), dtype=dtype))
                continue
            sd, gx, gy = _eval_expr(self._drawn_expr(lv), sdf, (nx, ny), res_um)
            offsets.append((sd * to_square).astype(dtype))
            # (1, 0) where no layer is near (BIG distances, zero gradient)
            normals.append(_unit_normal(gx, gy).astype(dtype))
        return jnp.stack(offsets, axis=-1), jnp.stack(normals, axis=-2)

    @timed("weights")
    def _weights_and_normals(
        self,
        params: dict[str, Array] | None,
        noise_key: Array | None,
        apply_roughness: bool,
        shift: tuple[float, float, float],
        eps_cols: np.ndarray | None = None,
        offdiag: bool = False,
    ) -> tuple[list[str], Array, Array | None]:
        """Volume fractions of the materials and, with ``eps_cols``, the voxels' normal tensors.

        The tensor is :func:`_normal_tensor` of each voxel's own model for the
        permittivities ``eps_cols``, with contrasts normalized by their range
        over the levels' materials and the background. With lateral
        sub-samples, their moments are added up per z sub-interval first
        (:func:`_composite_samples`), so the variance between them counts; the
        flux kernel spans the union of their squares where these tile the
        voxel (see :func:`_lateral_samples`).

        Args:
            params: device parameters (defaults: :meth:`init_params`).
            noise_key: PRNG key for the roughness.
            apply_roughness: if False, the nominal geometry.
            shift: offset (cells) of the sampled voxels from the grid's voxels.
            eps_cols: (num_materials, C) static permittivity of every material
                (fdtdx order) for C E components, or None: fractions only.
            offdiag: the tensor also has its off-diagonal entries.

        Returns:
            names: material names in fdtdx order.
            weights: (num_materials, nx, ny, nz), summing to one over materials.
            tensor: (C, 4, nx, ny, nz) variance along x, y, z and unexplained
                variance (see :func:`_normal_tensor`), with ``offdiag``
                (C, 7, nx, ny, nz) followed by the xy, xz, yz entries, or None
                without ``eps_cols``.
        """
        p = {**self.init_params(), **(params or {})}
        nx, ny, nz = self.grid_shape
        res_um = self._res_um()
        names = compute_ordered_names(self.materials)
        dtype = self._config.dtype
        zaxis = self._z_axis(shift[2])
        active, lows, highs = self._level_surfaces(p, noise_key, apply_roughness, zaxis.shift)
        if not active:
            w = jnp.zeros((len(names), nx, ny, nz), dtype=dtype)
            channels = 7 if offdiag else 4
            tensor = None if eps_cols is None else jnp.zeros((eps_cols.shape[1], channels, nx, ny, nz), dtype=dtype)
            return names, w.at[names.index(self.background)].set(1.0), tensor
        is_fill = tuple(self._drawn_expr(lv)[0] == "fill" for lv, *_ in active)
        level_materials = [lv.material for lv, *_ in active]

        n_sub = self.xy_subsamples
        graded_xy = self._lateral_graded()
        if graded_xy:  # each column's own cell: its offsets scaled here, its walls in _paint_layer
            cells = self._sample_cells((shift[0], shift[1]))
            to_square = 1.0 / cells.cell
        else:
            to_square = self._to_square()  # frame units -> sides of the lateral cell

        def lateral(z: Array) -> Array:
            """(..., K) offsets (scaled cells; frame units per column) of the levels' boundaries at heights z."""
            if graded_xy:
                return jnp.stack(
                    [jnp.zeros_like(z) if wall is None else _sidewall_shift(lv, z, *wall, res_um) for lv, wall in active],
                    axis=-1,
                )
            return jnp.stack(
                [
                    jnp.zeros_like(z) if wall is None else to_square * _sidewall_shift(lv, z, *wall, res_um)
                    for lv, wall in active
                ],
                axis=-1,
            )

        # heights (cells) where a level's offset bends (its z_to_bias knots): each
        # level takes the exact mean of its offset over its part of a voxel
        bends = [[] if wall is None or lv.z_to_bias is None else [wall[0] + f * wall[1] for f in lv.z_to_bias[0]]
                 for lv, wall in active]
        knots = None
        if any(bends):
            # padded by repeating a level's last knot (a repeated knot, or one on a
            # straight offset, changes nothing)
            rows = [b or [0.0] for b in bends]
            width = max(len(r) for r in rows)
            knots = jnp.stack([jnp.stack([jnp.asarray(x, dtype=dtype) for x in r + r[-1:] * (width - len(r))])
                               for r in rows])  # (K, M)

        # the sample's lateral cell, frame units per side of it (per column on graded x or y)
        lat_half, cell = (cells, cells.cell) if graded_xy else self._lateral_cell()
        normal_args = {}
        if eps_cols is not None:
            # each level's eps minus the background's, over the range of both (per component)
            eps_levels = np.asarray([eps_cols[names.index(m)] for m in level_materials], dtype=float)  # (K, C)
            eps_background = np.asarray(eps_cols[names.index(self.background)], dtype=float)
            spread = np.ptp(np.concatenate([eps_levels, eps_background[None]]), axis=0)
            spread = np.where(spread > 0, spread, 1.0)
            normal_args["level_eps"] = ((eps_levels - eps_background) / spread).T  # (C, K)
            normal_args["cell"] = cell
            normal_args["offdiag"] = offdiag
            if apply_roughness and self.roughness:  # the surfaces' lateral slopes (cells per cell)
                normal_args["surface_slopes"] = jnp.stack(
                    [self._surface_slopes(z) for z in (*lows, *highs)], axis=-2
                )

        surfaces = (jnp.stack(lows, axis=-1), jnp.stack(highs, axis=-1))  # (nx, ny, K) each
        geometry = self._layer_geometry(p)
        plan, radii = self._corner_plan(active, lows, highs, zaxis)
        capacity = 0 if plan is None else self._corner_capacity(plan, geometry)
        graded = {} if zaxis.uniform else {"zaxis": zaxis}
        if isinstance(lat_half, tuple):  # a rectangle: the composites paint on it
            graded["half"] = lat_half
        # several sub-samples are added up per sub-interval (_composite_samples) with the tensor,
        # and with corner columns (their corner model is blended by the voxel's certificate)
        together = n_sub > 1 and (bool(normal_args) or capacity > 0)
        weights, tensor, samples = 0.0, None, []
        for a, b, kernel in _lateral_samples(n_sub, self.edge_width):
            # graded x or y: the Yee shift and the sub-sample's offset apart (see _sample_points)
            grid = (shift[0], shift[1], a, b) if graded_xy else (shift[0] + a, shift[1] + b)
            sdf = self._source_sdfs(p, grid, geometry)
            corner = None
            if capacity:
                corner = self._corner_inputs(plan, radii, geometry, sdf, grid, capacity)
            offsets, normals = self._half_planes(active, is_fill, sdf, to_square, dtype)
            if together:
                samples.append((offsets, normals, corner, kernel))
                continue
            out = _composite(
                offsets, normals, is_fill, *surfaces, lateral, nz, corner=corner, kernel=kernel, knots=knots,
                **normal_args, **graded,
            )
            if normal_args:  # one sub-sample
                out, tensor = out
            weights = weights + out
        if together:
            offsets, normals, corners, kernels = zip(*samples, strict=True)
            weights, tensor = _composite_samples(
                jnp.stack(offsets),
                jnp.stack(normals),
                is_fill,
                *surfaces,
                lateral,
                nz,
                normal_args.get("level_eps"),
                kernels,
                cell,
                corners if capacity else None,
                normal_args.get("surface_slopes"),
                knots=knots,
                offdiag=offdiag,
                **graded,
            )
        weights = weights / n_sub**2
        out = []
        for m in names:
            cols = [j for j, lm in enumerate(level_materials) if lm == m]
            if m == self.background:
                cols.append(len(level_materials))
            out.append(weights[..., cols].sum(axis=-1) if cols else jnp.zeros((nx, ny, nz), dtype))
        if tensor is not None:
            tensor = jnp.moveaxis(tensor, (3, 4), (0, 1))  # (C, 4 or 7, nx, ny, nz)
        return names, jnp.stack(out), tensor

    # ------------------------------------------------------------ complete tensors (averaging="subpixel_full")

    def _material_table(self) -> tuple[list[str], np.ndarray]:
        """Material names in fdtdx order and their (M, 3, 3) permittivity tensors (checked; the placeholder is I)."""
        names = compute_ordered_names(self.materials)
        return names, material_tensors(self.materials, names, ignore=[_ANISOTROPY_SENTINEL])

    def _lateral_model(
        self, active: Sequence[tuple[_Level, Any]], dtype: Any
    ) -> tuple[Callable[[Array], Array], Array | None]:
        """The levels' sidewall offsets at heights (sides of a sample square) and their z_to_bias knots (cells).

        As in :meth:`_weights_and_normals` (see :func:`_composite`); on graded x or y the offsets are in frame
        units (each column's cell scales them).
        """
        res_um = self._res_um()
        graded_xy = self._lateral_graded()
        to_square = 1.0 if graded_xy else self._to_square()

        def lateral(z: Array) -> Array:
            if graded_xy:
                return jnp.stack(
                    [jnp.zeros_like(z) if wall is None else _sidewall_shift(lv, z, *wall, res_um) for lv, wall in active],
                    axis=-1,
                )
            return jnp.stack(
                [
                    jnp.zeros_like(z) if wall is None else to_square * _sidewall_shift(lv, z, *wall, res_um)
                    for lv, wall in active
                ],
                axis=-1,
            )

        bends = [[] if wall is None or lv.z_to_bias is None else [wall[0] + f * wall[1] for f in lv.z_to_bias[0]]
                 for lv, wall in active]
        knots = None
        if any(bends):
            rows = [b or [0.0] for b in bends]
            width = max(len(r) for r in rows)
            knots = jnp.stack([jnp.stack([jnp.asarray(x, dtype=dtype) for x in r + r[-1:] * (width - len(r))])
                               for r in rows])  # (K, M)
        return lateral, knots

    @timed("interfaces")
    def _interfaces(
        self,
        params: dict[str, Array] | None,
        noise_key: Array | None,
        apply_roughness: bool,
        shift: tuple[float, float, float],
    ) -> "_Interfaces":
        """The voxels of one sample grid (e.g. a Yee position) with their interfaces.

        Every voxel's material fractions (those of :meth:`material_weights`) and its interface direction
        tensor (:func:`_interface_composite`: the faces of its own model, half-planes or, in corner
        columns, the corner model, plus the variance of its material moments along z), with what the
        junction closure needs to paint the voxel again (:func:`_junction_tensor`).

        Args:
            params: device parameters (defaults: :meth:`init_params`).
            noise_key: PRNG key for the roughness.
            apply_roughness: if False, the nominal geometry.
            shift: offset (cells) of the sampled voxels from the grid's voxels.

        Returns:
            The grid's interfaces (see :class:`_Interfaces`).
        """
        p = {**self.init_params(), **(params or {})}
        nx, ny, nz = self.grid_shape
        names, eps = self._material_table()
        dtype = self._config.dtype
        zaxis = self._z_axis(shift[2])
        graded = {} if zaxis.uniform else {"zaxis": zaxis}
        active, lows, highs = self._level_surfaces(p, noise_key, apply_roughness, zaxis.shift)
        if not active:
            w = jnp.zeros((len(names), nx, ny, nz), dtype=dtype).at[names.index(self.background)].set(1.0)
            return _Interfaces(names=names, weights=w, direction=jnp.zeros((nx, ny, nz, 6), dtype=dtype), voxel=None)
        is_fill = tuple(self._drawn_expr(lv)[0] == "fill" for lv, *_ in active)
        level_materials = [lv.material for lv, *_ in active]
        # the levels' materials, then the background: tensors, normalized 6-vectors, squared contrasts
        level_eps = eps[[names.index(m) for m in (*level_materials, self.background)]]  # (K + 1, 3, 3)
        scale = float(max(np.linalg.norm(e) for e in level_eps))
        vectors = np.stack([
            np.array([e[0, 0], e[1, 1], e[2, 2], math.sqrt(2) * e[0, 1], math.sqrt(2) * e[0, 2], math.sqrt(2) * e[1, 2]])
            for e in level_eps
        ]) / scale
        contrast2 = np.sum((level_eps[:, None] - level_eps[None, :]) ** 2, axis=(-2, -1)) / scale**2

        n_sub = self.xy_subsamples
        graded_xy = self._lateral_graded()
        if graded_xy:  # each column's own cell (see _weights_and_normals)
            lat_half = self._sample_cells((shift[0], shift[1]))
            to_square, cell = 1.0 / lat_half.cell, lat_half.cell
        else:
            to_square = self._to_square()  # frame units -> sides of a sample's cell
            lat_half, cell = self._lateral_cell()
        lateral, knots = self._lateral_model(active, dtype)
        surface_slopes = None
        if apply_roughness and self.roughness:
            surface_slopes = jnp.stack([self._surface_slopes(z) for z in (*lows, *highs)], axis=-2)

        surfaces = (jnp.stack(lows, axis=-1), jnp.stack(highs, axis=-1))
        geometry = self._layer_geometry(p)
        plan, radii = self._corner_plan(active, lows, highs, zaxis)
        capacity = 0 if plan is None else self._corner_capacity(plan, geometry)
        direction = 0.0
        fractions = 0.0
        dropped = jnp.zeros((), dtype=jnp.int32)
        uncertified = jnp.zeros((), dtype=jnp.int32)
        planes, corners, kernels = [], [], []
        for a, b, kernel in _lateral_samples(n_sub, self.edge_width):
            # graded x or y: the Yee shift and the sub-sample's offset apart (see _sample_points)
            grid = (shift[0], shift[1], a, b) if graded_xy else (shift[0] + a, shift[1] + b)
            sdf = self._source_sdfs(p, grid, geometry)
            corner = self._corner_inputs(plan, radii, geometry, sdf, grid, capacity) if capacity else None
            if corner is not None and corner.dropped is not None:
                dropped = dropped + jnp.sum(corner.dropped, dtype=jnp.int32)
            offsets, normals = self._half_planes(active, is_fill, sdf, to_square, dtype)
            planes.append((offsets, normals))
            corners.append(corner)
            kernels.append(kernel)
            f, d, u = _interface_composite(
                offsets, normals, is_fill, *surfaces, lateral, nz, lat_half, knots, cell, vectors, contrast2,
                surface_slopes, corner=corner, **graded,
            )
            fractions = fractions + f
            direction = direction + d
            uncertified = uncertified + u
        if n_sub == 1:
            per_level = fractions
        elif capacity:
            # with sub-samples and corner columns, blend the corner model by the voxel's certificate
            # (the least of its samples', see _composite_samples) using the already-computed samples
            per_level = (
                _composite_samples(
                    jnp.stack([o for o, _ in planes]),
                    jnp.stack([n for _, n in planes]),
                    is_fill,
                    *surfaces,
                    lateral,
                    nz,
                    None,
                    kernels,
                    cell,
                    corners,
                    half=lat_half,
                    knots=knots,
                    **graded,
                )[0]
                / n_sub**2
            )
        else:
            per_level = fractions / n_sub**2
        out = []
        for m in names:
            cols = [j for j, lm in enumerate(level_materials) if lm == m]
            if m == self.background:
                cols.append(len(level_materials))
            out.append(per_level[..., cols].sum(axis=-1) if cols else jnp.zeros((nx, ny, nz), dtype))
        w = jnp.stack(out)
        voxel = _JunctionInputs(
            offsets=jnp.stack([o for o, _ in planes]),
            normals=jnp.stack([n for _, n in planes]),
            lows=surfaces[0],
            highs=surfaces[1],
            point_slopes=_point_slopes(surfaces[0], surface_slopes),
            lateral=lateral,
            knots=knots,
            is_fill=is_fill,
            level_eps=level_eps,
            vectors=vectors,
            contrast2=contrast2,
            cell=cell,
            corner=corner if n_sub == 1 else None,
            level_material=tuple(names.index(m) for m in (*level_materials, self.background)),
            zaxis=graded.get("zaxis"),
            half=lat_half,
            widths=(
                tuple(a.hi - a.lo for a in self._sample_axes((shift[0], shift[1])))
                if graded_xy else self._lateral_widths()
            ),
        )
        return _Interfaces(
            names=names, weights=w.astype(dtype), direction=direction, voxel=voxel, corner_dropped=dropped,
            corner_uncertified=uncertified,
        )

    def _junction_capacity(self) -> int:
        """Static number of junction voxels per sample grid (``junction_voxels``, or a bound with a warning)."""
        nx, ny, nz = self.grid_shape
        if self.junction_voxels is not None:
            return max(0, min(nx * ny * nz, int(self.junction_voxels)))
        warnings.warn(
            f"StackDevice {self.name!r} has no junction_voxels (it is set when the device is placed); using "
            "an eighth of the voxels",
            stacklevel=3,
        )
        return max(64, nx * ny * nz // 8)

    def _junction_inputs(
        self, g: "_Interfaces", voxels: Array
    ) -> tuple[dict[str, Any], Callable[[dict[str, Any]], tuple[Array, Array]]]:
        """The junction closure's inputs for some voxels of a sample grid, and the function painting their sub-cells.

        Args:
            g: the grid's interfaces (:meth:`_interfaces`).
            voxels: (N,) flat voxel indices (of (nx, ny, nz), i-major).

        Returns:
            inputs: per-voxel arrays (leading axis N), to batch as needed.
            cells: maps (a batch of) the inputs to the voxels' sub-cells: their fractions (N, n, n, m, K + 1)
                and laminates (N, n, n, m, 3, 3) (:func:`_junction_cells`).

        Raises:
            ValueError: ``junction_cells`` that do not split the sample squares evenly.
        """
        nx, ny, nz = self.grid_shape
        dtype = g.weights.dtype
        col, layer = voxels // nz, (voxels % nz).astype(dtype)
        v = g.voxel
        cells = self.junction_cells
        n_sub = self.xy_subsamples
        if cells[0] % n_sub or cells[1] % n_sub:
            raise ValueError(f"junction_cells {cells} must split each of the {n_sub} sample squares per axis evenly")
        per_x = cells[0] // n_sub  # sub-squares per sample square along x
        per_y = cells[1] // n_sub  # sub-squares per sample square along y
        per_column = isinstance(v.half, _Cells)  # graded x or y: every voxel's own cell (set per voxel below)
        rectangle = isinstance(v.half, tuple) and not per_column
        if rectangle:  # sub-rectangles of the sample's rectangle
            hx, hy = v.half
            sub: tuple[float, float] | float = (hx / per_x, hy / per_y)
        elif not per_column:
            sub = 0.5 / per_x if per_x == per_y else (0.5 / per_x, 0.5 / per_y)
        else:
            sub = 0.5 / per_x
        # every sub-square (x-major): the sample square holding it and its center in that square's frame
        sample_of, centers, unit = [], [], []
        sx, sy = (sub if isinstance(sub, tuple) else (sub, sub))
        for i in range(cells[0]):
            for j in range(cells[1]):
                sample_of.append((i // per_x) * n_sub + j // per_y)
                # per column: in units of the voxel's half sides (hx, hy)
                unit.append((-1.0 + (i % per_x + 0.5) * 2 / per_x, -1.0 + (j % per_y + 0.5) * 2 / per_y))
                if rectangle:
                    centers.append((-hx + (i % per_x + 0.5) * 2 * sx, -hy + (j % per_y + 0.5) * 2 * sy))
                elif not per_column:
                    centers.append((-0.5 + (i % per_x + 0.5) * 2 * sx, -0.5 + (j % per_y + 0.5) * 2 * sy))
        sample_of = np.asarray(sample_of)
        centers = None if per_column else jnp.asarray(centers, dtype=dtype)

        def gather(x: Array) -> Array:
            return x.reshape(nx * ny, *x.shape[2:])[col]

        inputs = {
            "offsets": jnp.moveaxis(jnp.stack([gather(o) for o in v.offsets]), 0, 1)[:, sample_of],  # (N, Q, K)
            "normals": jnp.moveaxis(jnp.stack([gather(n) for n in v.normals]), 0, 1)[:, sample_of],
            "lows": gather(v.lows),
            "highs": gather(v.highs),
            "layer": layer,
            # the voxels' material fractions, which their sub-cells are made to add up to
            "target": jnp.moveaxis(g.weights, 0, -1).reshape(-1, g.weights.shape[0])[voxels],
        }
        if v.zaxis is not None:  # a graded layer: its bottom and top (frame units) instead of k, k + 1
            k = voxels % nz
            inputs["layer"] = jnp.asarray(v.zaxis.lo, dtype=dtype)[k]
            inputs["top"] = jnp.asarray(v.zaxis.hi, dtype=dtype)[k]
        if per_column:  # every voxel's own cell and lateral widths (frame units)
            for name, x in (("hx", v.half.hx), ("hy", v.half.hy), ("cell", v.half.cell)):
                inputs[name] = jnp.reshape(x, (-1,))[col]
            inputs["wx"] = jnp.asarray(v.widths[0], dtype=dtype)[col // ny]
            inputs["wy"] = jnp.asarray(v.widths[1], dtype=dtype)[col % ny]
        if v.point_slopes is not None:
            inputs["point_slopes"] = gather(v.point_slopes)
        corner = v.corner
        if corner is not None:
            # each voxel's corner slot, if its column has one; elsewhere taken is False (no weight)
            slots = corner.cols.shape[0]
            slot_of = jnp.full((nx * ny,), -1, dtype=jnp.int32).at[corner.cols].set(
                jnp.arange(slots, dtype=jnp.int32), mode="drop"
            )
            slot = slot_of[col]
            inputs["taken"] = (slot >= 0) & corner.taken[jnp.maximum(slot, 0)]
            inputs["layers"] = jax.tree.map(lambda x: jnp.moveaxis(x[:, jnp.maximum(slot, 0)], 1, 0), corner.layers)
            inputs["planes"] = (gather(v.offsets[0]), gather(v.normals[0]))

        def cells_of(args: dict[str, Any]) -> tuple[Array, Array]:
            half, sub_j, centers_j, sc = v.half, sub, centers, None if corner is None else corner.sc
            if per_column:  # the voxels' own cells, their sub-cells and the sub-cells' centers
                half = _Cells(args["hx"], args["hy"], args["cell"], v.half.reach)
                sub_j = _Cells(
                    args["hx"] / per_x,
                    args["hy"] / per_y,
                    args["cell"],
                    v.half.reach / min(per_x, per_y),
                    1.0 / (per_x * per_y),
                )
                centers_j = jnp.asarray(unit, dtype=dtype)[None] * jnp.stack([args["hx"], args["hy"]], axis=-1)[:, None]
                sc = (1.0 / args["cell"])[:, None]
            corner_j = None
            if corner is not None:  # the voxels' corner models: the layer axis first again
                corner_j = _CornerInputs(
                    cols=None, taken=args["taken"], layers=jax.tree.map(lambda x: jnp.moveaxis(x, 0, 1), args["layers"]),
                    trees=corner.trees, literals=corner.literals, lines=corner.lines, sc=sc,
                )
            return _junction_cells(
                args["offsets"], args["normals"], centers_j, v.is_fill, args["lows"], args["highs"], args["layer"],
                v.lateral, sub_j, v.knots, v.cell, v.vectors, v.contrast2, v.level_eps, cells,
                args.get("point_slopes"), corner_j, args.get("planes"), args["target"], v.level_material,
                **({"top": args["top"]} if "top" in args else {}),
                **({"half": half} if rectangle or per_column else {}),
            )

        return inputs, cells_of

    @timed("complete_tensor")
    def _complete_tensor(self, g: "_Interfaces", capacity: int) -> tuple[Array, dict[str, Array]]:
        """The inverse effective permittivity of every voxel of a sample grid (averaging="subpixel_full").

        A voxel of one material keeps that material's exact inverse. Otherwise its interface direction tensor
        D (:func:`_interface_composite`) decides (:func:`bifrost.subpixel.junction_weight`): where
        its interfaces share one normal (also parallel films and multilayers, a sloped wall, a rough
        surface) the laminate along it (Kottke's planar average of the anisotropic materials,
        :func:`bifrost.subpixel.laminate`); where they do not (a corner, a wall meeting a surface,
        three materials), the junction closure (:func:`_junction_tensor`), blended smoothly with the
        laminate along D's leading direction as the second direction appears (symmetric, positive definite,
        between the materials' harmonic and arithmetic means). Junction voxels are compacted into
        ``capacity`` slots; candidates beyond it keep the laminate and are counted in the status.

        Args:
            g: the grid's interfaces (:meth:`_interfaces`).
            capacity: number of junction slots (static).

        Returns:
            (nx, ny, nz, 6) the inverse tensor (xx, yy, zz, xy, xz, yz) and the status: ``junctions``
            (candidates), ``junction_overflow`` (candidates without a slot) and ``closure_residual``
            (largest relative residual of the cell problems' solves).
        """
        nx, ny, nz = self.grid_shape
        names, eps = self._material_table()
        dtype = g.weights.dtype
        inv_eps = np.linalg.inv(eps)
        # an axis of one cell (2D simulations) has no interface normal along it
        keep = np.asarray([n > 1 for n in self.grid_shape], dtype=float)
        d = from_six(g.direction) * jnp.asarray(np.outer(keep, keep), dtype=dtype)
        omega, normal = junction_weight(d, _JUNCTION_LOW, _JUNCTION_HIGH)
        f = jnp.moveaxis(g.weights, 0, -1)  # (nx, ny, nz, M)

        def planar(inputs: tuple[Array, Array]) -> Array:
            """(nx, ny, 6) of one voxel layer: the material's own inverse, or the laminate's."""
            fk, nk = inputs
            k, _ = _safe_symmetric_inverse(laminate(eps, fk, nk))
            pure = jnp.max(fk, axis=-1) >= 1.0
            exact = jnp.einsum("...m,mij->...ij", fk, jnp.asarray(inv_eps, dtype=dtype), precision=_HIGHEST)
            return to_six(jnp.where(pure[..., None, None], exact, k))

        out = jax.lax.map(planar, (jnp.moveaxis(f, 2, 0), jnp.moveaxis(normal, 2, 0)))  # (nz, nx, ny, 6)
        out = jnp.moveaxis(out, 0, 2).reshape(-1, 6)
        candidate = (omega > 0).ravel()
        count = jnp.sum(candidate, dtype=jnp.int32)
        status = {
            "junctions": count,
            "junction_overflow": jnp.maximum(count - capacity, 0),
            "closure_residual": jnp.zeros((), dtype=dtype),
        }
        if g.voxel is None or capacity == 0:
            return out.reshape(nx, ny, nz, 6), status
        num = nx * ny * nz
        idx = jnp.nonzero(candidate, size=capacity, fill_value=num)[0]
        safe = jnp.minimum(idx, num - 1)
        inputs, cells_of = self._junction_inputs(g, safe)
        chunk = min(capacity, _JUNCTION_CHUNK)
        pad = (-capacity) % chunk

        def batched(x: Array) -> Array:
            x = jnp.concatenate([x, jnp.repeat(x[-1:], pad, axis=0)]) if pad else x
            return x.reshape(-1, chunk, *x.shape[1:])

        wx, wy = g.voxel.widths

        def closure(args: dict[str, Any]) -> tuple[Array, Array]:
            if "wx" in args:  # graded x or y: every voxel's own box (frame units)
                height = args["top"] - args["layer"] if "top" in args else jnp.ones_like(args["layer"])
                return periodic_cell_tensor(cells_of(args)[1], jnp.stack([args["wx"], args["wy"], height], axis=-1))
            if "top" not in args and wx == wy:
                return periodic_cell_tensor(cells_of(args)[1])
            # the voxel is a box (frame units): its lateral widths, as high as its layer (a graded z) or 1
            ones = jnp.ones_like(args["layer"])
            height = args["top"] - args["layer"] if "top" in args else ones
            return periodic_cell_tensor(cells_of(args)[1], jnp.stack([wx * ones, wy * ones, height], axis=-1))

        tensor, residual = jax.lax.map(jax.checkpoint(closure), jax.tree.map(batched, inputs))
        tensor = tensor.reshape(-1, 3, 3)[:capacity].astype(dtype)
        residual = residual.reshape(-1)[:capacity]
        w = omega.ravel()[safe]
        planar_eps = laminate(eps, f.reshape(-1, len(names))[safe], normal.reshape(-1, 3)[safe])
        planar_ok = _is_finite_spd(planar_eps)
        eye = jnp.eye(3, dtype=dtype)
        planar_safe = jnp.where(planar_ok[:, None, None], planar_eps, eye)
        tensor_ok = _is_finite_spd(tensor)
        residual_ok = jnp.isfinite(residual) & (residual <= _CLOSURE_RESIDUAL)
        tensor_safe = jnp.where(tensor_ok[:, None, None], tensor, planar_safe)
        blended = (1.0 - w)[:, None, None] * planar_safe + w[:, None, None] * tensor_safe
        inv_blended, blended_ok = _safe_symmetric_inverse(blended)
        certified = (idx < num) & residual_ok & tensor_ok & planar_ok & blended_ok
        completed = jnp.where(certified[:, None], to_six(inv_blended), out[safe])
        out = out.at[idx].set(completed, mode="drop")
        bad_residual = jnp.where(tensor_ok & planar_ok & blended_ok, residual, jnp.asarray(jnp.nan, dtype=dtype))
        status["closure_residual"] = jnp.max(jnp.where(idx < num, bad_residual, jnp.zeros((), dtype=dtype)))
        return out.reshape(nx, ny, nz, 6), status

    # ------------------------------------------------------------ fdtdx hooks

    @timed("place_on_grid")
    def place_on_grid(
        self, grid_slice_tuple: Any, config: SimulationConfig, key: Array
    ) -> Self:
        """fdtdx hook: aligns pole slots, requests anisotropic arrays (subpixel), sizes corner columns, diagnoses.

        Without ``corner_columns``, the corner model gets 1.5 times the most
        candidate columns of any sample grid at the initial parameters, plus 64,
        rounded up to a multiple of 64 (at most every column). With
        ``diagnostics``, every finding of :meth:`diagnose` at the grid's
        resolution is a :class:`StackDiagnosticWarning`.

        Args:
            grid_slice_tuple: the device's grid slices.
            config: simulation config.
            key: PRNG key.

        Returns:
            The placed device.

        Raises:
            ValueError: if ``config.dtype`` is not the device's ``dtype``
                (:meth:`get_simulation_config` builds one that is).
            RuntimeError: if a float64 device is placed with JAX's 64-bit mode off.
        """
        self._check_precision(config)
        frame = _grid_frame(config, grid_slice_tuple)
        if frame.widths[0] is None or frame.widths[1] is None:  # graded x or y: painted per column
            wx, wy = np.diff(frame.edges[0]), np.diff(frame.edges[1])
            aspect = float(max(np.max(wx) / np.min(wy), np.max(wy) / np.min(wx)))
        else:
            aspect = max(frame.widths[0] / frame.widths[1], frame.widths[1] / frame.widths[0])
        if aspect > _MAX_ASPECT:
            warnings.warn(
                f"StackDevice {self.name!r}: the grid's cells are {aspect:.3g} times longer in one lateral direction "
                f"than in the other (above {_MAX_ASPECT:g}); the subpixel model and its closure lose accuracy there",
                StackDiagnosticWarning,
                stacklevel=2,
            )
        self = self.aset("grid_frame", frame)
        self = self._keep_reachable_levels(grid_slice_tuple, config)
        # aligned pole slots also for devices built without from_layer_stack;
        # fdtdx allocates the slots from the placed objects' materials
        mats = {k: v for k, v in self.materials.items() if k != _ANISOTROPY_SENTINEL}
        _check_materials(mats, self.averaging, self.name)
        mats = _drop_empty_pole_slots(align_material_poles(mats))
        if self.averaging == "subpixel":
            # makes fdtdx allocate diagonally anisotropic arrays (never painted)
            mats[_ANISOTROPY_SENTINEL] = fdtdx.Material(permittivity=(1.0, 1.0, 2.0))
        elif self.averaging == "subpixel_full":
            # an off-diagonal entry makes fdtdx allocate fully anisotropic arrays (never painted)
            mats[_ANISOTROPY_SENTINEL] = fdtdx.Material(permittivity=((2.0, 0.5, 0.0), (0.5, 2.0, 0.0), (0.0, 0.0, 2.0)))
        self = self.aset("materials", mats)
        warn_reversible_losses(mats, config, self.name, conductivity=True)
        placed = super().place_on_grid(grid_slice_tuple=grid_slice_tuple, config=config, key=key)
        lap("pole_alignment")
        if placed.corner_lines > 0 and placed.corner_columns is None:
            placed = placed.aset("corner_columns", placed._corner_budget())
        if placed.averaging == "subpixel_full" and placed.corner_lines == 0 and placed.diagnostics:
            warnings.warn(
                f"StackDevice {placed.name!r}: averaging='subpixel_full' with corner_lines=0 paints every "
                "polygon corner with one half-plane per level, so its voxels get the laminate along one "
                "normal; the junction closure resolves corners with the corner model (corner_lines > 0)",
                StackDiagnosticWarning,
                stacklevel=2,
            )
        if placed.averaging == "subpixel_full" and placed.junction_voxels is None:
            placed = placed.aset("junction_voxels", placed._junction_budget())
        if placed.diagnostics:
            for finding in placed.diagnose():
                warnings.warn(finding.message, StackDiagnosticWarning, stacklevel=2)
        return placed

    @timed("corner_budget")
    def _corner_budget(self) -> int:
        """Corner columns for the initial parameters of a placed device (see :meth:`place_on_grid`)."""
        nx, ny, _ = self.grid_shape
        return min(nx * ny, 64 * math.ceil((1.5 * self._corner_candidate_count() + 64) / 64))

    @timed("junction_budget")
    def _junction_budget(self) -> int:
        """Junction voxels for the initial parameters of a placed device: 1.5 times the most candidates of any
        Yee position, plus 64, rounded up to a multiple of 64 (at most every voxel)."""
        nx, ny, nz = self.grid_shape
        shifts = _YEE_E_SHIFTS if self.yee_staggered else ((0.0, 0.0, 0.0),)
        keep = np.asarray([n > 1 for n in self.grid_shape], dtype=float)

        @partial(jax.jit, static_argnums=1)
        def count(p: dict[str, Array], shift: tuple[float, float, float]) -> tuple[Array, Array]:
            g = self._interfaces(p, None, True, shift)
            d = from_six(g.direction) * jnp.asarray(np.outer(keep, keep), dtype=g.direction.dtype)
            omega, _ = junction_weight(d, _JUNCTION_LOW, _JUNCTION_HIGH)
            return jnp.sum(omega > 0, dtype=jnp.int32), jnp.asarray(g.corner_uncertified, dtype=jnp.int32)

        p = self.init_params()
        counts = [count(p, tuple(float(x) for x in shift)) for shift in shifts]
        most = max(int(c) for c, _ in counts)
        uncertified = sum(int(u) for _, u in counts)
        if self.diagnostics and uncertified:
            warnings.warn(
                f"StackDevice {self.name!r}: {uncertified} voxels of corner columns (summed over the Yee "
                "positions) are painted partly with one half-plane per level, where the corner model is not "
                "certified (more edges or tips than it resolves); their junction closure sees that model. "
                "interface_status() reports the count for other parameters.",
                StackDiagnosticWarning,
                stacklevel=3,
            )
        return min(nx * ny * nz, 64 * math.ceil((1.5 * most + 64) / 64))

    @timed("keep_reachable_levels")
    def _keep_reachable_levels(self, grid_slice_tuple: Any, config: SimulationConfig) -> Self:
        """Drops the levels that cannot reach the placed box, with their materials and source layers.

        The box's exact height is known only now: fdtdx rounds it to whole cells
        around ``z_center``. A level is kept if its nominal span comes within
        ``level_margin`` of the box, or within half a cell below it, which the
        staggered Ex and Ey samples of the bottom voxels see. Materials of
        dropped levels are not allocated by fdtdx (e.g. no conductivity array
        for a metal level above the box).

        Args:
            grid_slice_tuple: the device's grid slices.
            config: simulation config.

        Returns:
            The device with the reachable levels only.
        """
        frame = self.grid_frame if self.grid_frame is not None else _grid_frame(config, grid_slice_tuple)
        res_um = frame.h_um
        # on a uniform grid: half the box's cells, and half a cell below it
        half = 0.5 * float(frame.edges[2][-1]) * res_um
        bottom = self.z_center - half - 0.5 * frame.below[2] * res_um - self.level_margin
        top = self.z_center + half + self.level_margin
        self._warn_dropped_levels(bottom, top)
        keep_names, padded = _roughness_reach(
            [(lv.name, *_span(lv)) for lv in self.levels], self.roughness, bottom, top
        )
        levels = tuple(lv for lv in self.levels if lv.name in keep_names)
        # empty layers matter only to levels that reach the box
        empty = tuple(
            (index, reach)
            for index, readers in self.empty_layers
            if (
                reach := tuple(
                    (name, lo, hi)
                    for name, lo, hi in readers
                    if name in keep_names
                    or (padded.get(name, (lo, hi))[1] > bottom and padded.get(name, (lo, hi))[0] < top)
                )
            )
        )
        if empty != tuple(self.empty_layers):
            self = self.aset("empty_layers", empty)
        roughness = tuple(r for r in self.roughness if r.level in keep_names)
        if len(levels) == len(self.levels) and len(roughness) == len(self.roughness):
            return self
        layers = set().union(*(_expr_layers(lv.expr) for lv in levels))
        keep = [i for i, (_, index) in enumerate(self.source_layers) if index in layers]
        used = {lv.material for lv in levels} | {self.background, _ANISOTROPY_SENTINEL}
        out = self.aset("levels", levels)
        out = out.aset("roughness", roughness)
        out = out.aset("materials", {k: m for k, m in self.materials.items() if k in used})
        for name in ("source_layers", "source_vertices", "source_sizes"):
            out = out.aset(name, tuple(getattr(self, name)[i] for i in keep))
        if self.source_extensions:
            out = out.aset("source_extensions", tuple(self.source_extensions[i] for i in keep))
        # the shapes of the kept levels (their pools are on the kept layers)
        names = {lv.name for lv in levels}
        shapes = tuple(
            d._replace(levels=tuple(n for n in d.levels if n in names))
            for d in self.derived_layers if any(n in names for n in d.levels)
        )
        return out.aset("derived_layers", shapes)

    def _warn_dropped_levels(self, bottom: float, top: float) -> None:
        """Warns about levels left out when the device was built that the placed grid samples.

        ``from_layer_stack`` keeps the levels within ``level_margin`` plus a slack of the box; a
        coarse cell at the box's bottom face samples below that, so a level there is missing.

        Args:
            bottom: lowest stack height (um) the grid samples, ``level_margin`` included.
            top: highest one.
        """
        z_lo = self.z_center - 0.5 * self.partial_real_shape[2] * 1e6
        z_hi = self.z_center + 0.5 * self.partial_real_shape[2] * 1e6
        for name, lo, hi in self.dropped_levels:
            if hi > bottom and lo < top:
                gap = max(z_lo - hi, lo - z_hi)
                warnings.warn(
                    f"StackDevice {self.name!r}: level {name!r} (z {lo:g} to {hi:g} um) was left out when the "
                    f"device was built, {gap:.3g} um outside the box, but this grid samples from {bottom:g} to "
                    f"{top:g} um, so it is missing; build the device with level_margin >= "
                    f"{max(gap - _SELECT_SLACK, 0.0) + 1e-3:.3g}",
                    StackDiagnosticWarning,
                    stacklevel=4,
                )

    def get_voxel_mask_for_shape(self) -> Array:
        """fdtdx hook: the device fills its whole box.

        Returns:
            (nx, ny, nz) all-True mask.
        """
        return jnp.ones(self.grid_shape, dtype=jnp.bool)

    def get_material_mapping(self) -> Array:
        """fdtdx hook: the nominal geometry rounded to the dominant material.

        Only used by fdtdx to paint the initial arrays; :func:`apply_stack`
        overwrites them with the mixed materials. It is computed without the
        corner model (``corner_lines=0``), which only refines the fractions of
        corner voxels: compiling the corner pass for this one rounding cost the
        first placement of a device 42 s of GPU autotuning at 260x180x90 voxels.

        Returns:
            (nx, ny, nz) material indices in fdtdx's order.
        """
        _, w = self.aset("corner_lines", 0).material_weights(apply_roughness=False)
        return jnp.argmax(w, axis=0).astype(jnp.int32)

    @timed("write")
    def write(
        self,
        arrays: Any,
        params: dict[str, Array] | None = None,
        noise_key: Array | None = None,
        apply_roughness: bool = True,
    ) -> Any:
        """Writes the mixed inverse permittivity, conductivity and pole coefficients.

        With ``averaging="subpixel_full"`` (9-component arrays) row c of the
        inverse of the voxel's complete effective permittivity tensor is
        written at E_c's Yee position (:meth:`write_with_status`): the
        materials' full tensors, Kottke's anisotropic laminate along the
        voxel's interface normal where its interfaces are parallel, and the
        junction closure where they are not. Other averagings take each E
        component's diagonal entry of a diagonally anisotropic material (a
        material with off-diagonal entries needs "subpixel_full"), and write
        the diagonal and zero off-diagonal entries into 9-component arrays
        (made by another object).

        Args:
            arrays: fdtdx ArrayContainer.
            params: device parameters (default: :meth:`init_params`).
            noise_key: PRNG key for the roughness.
            apply_roughness: if False, the nominal geometry.

        Returns:
            The arrays with the device box overwritten.

        Raises:
            ValueError: unknown averaging, "subpixel" with isotropic arrays, or
                "subpixel_full" without 9-component arrays.
        """
        n_comp = arrays.inv_permittivities.shape[0]
        if n_comp not in (1, 3, 9):
            raise ValueError(f"Unexpected number of permittivity components {n_comp}")
        if self.averaging not in ("inverse", "subpixel", "subpixel_full"):
            raise ValueError(f"Unknown averaging {self.averaging!r}")
        if self.subpixel_normal not in ("voxel", "gradient"):
            raise ValueError(f"subpixel_normal must be 'voxel' or 'gradient', got {self.subpixel_normal!r}")
        if self.averaging == "subpixel" and n_comp == 1:
            raise ValueError(
                "averaging='subpixel' needs diagonally anisotropic arrays; build the "
                "device with from_layer_stack(..., averaging='subpixel')"
            )
        if self.averaging == "subpixel_full" and n_comp != 9:
            raise ValueError(
                "averaging='subpixel_full' needs fully anisotropic (9-component) arrays; build the "
                "device with from_layer_stack(..., averaging='subpixel_full')"
            )
        if self.averaging == "subpixel_full":
            return self.write_with_status(arrays, params, noise_key, apply_roughness)[0]
        _check_materials(self.materials, self.averaging, self.name)
        full = n_comp == 9
        rows = 3 if full else n_comp  # the E components
        dtype = arrays.inv_permittivities.dtype
        iso = {"isotropic": n_comp == 1, "diagonally_anisotropic": n_comp == 3}
        eps_np = np.asarray(compute_allowed_permittivities(self.materials, **iso), dtype=float)  # (M, n_comp)
        if full:
            eps_np = eps_np[:, [0, 4, 8]]  # each E component's diagonal entry
        eps = jnp.asarray(eps_np, dtype=dtype)
        sigma = self._allowed_conductivity(arrays, dtype)
        # each E component sees the materials around its own Yee position;
        # components are processed one at a time to bound the memory
        shifts = _YEE_E_SHIFTS if rows == 3 and self.yee_staggered else ((0.0, 0.0, 0.0),)
        subpixel = self.averaging in ("subpixel", "subpixel_full")
        voxel_normal = subpixel and self.subpixel_normal == "voxel"
        if subpixel:
            # the axes of one cell (2D simulations) get no normal component
            keep = jnp.asarray([float(n > 1) for n in self.grid_shape], dtype=dtype)[:, None, None, None]
            sentinel = np.asarray([m == _ANISOTROPY_SENTINEL for m in compute_ordered_names(self.materials)])
        keep_yee_w = (
            len(shifts) == 3
            and arrays.dispersive_c1 is not None
            and arrays.dispersive_c3.shape[1] in (3, 9)
        )
        inv, cond, w_cell, yee_weights = {}, [], 0.0, []
        for c in range(rows):
            if c < len(shifts):
                if voxel_normal:  # the normals of the component's own voxels (all three if they share them)
                    cols = eps_np[:, [c]] if len(shifts) == rows else eps_np
                    _, w, tensors = self._weights_and_normals(params, noise_key, apply_roughness, shifts[c], cols)
                    tensors = tensors.astype(dtype)
                else:
                    w = self.material_weights(params, noise_key, apply_roughness, shifts[c])[1]
                w = w.astype(dtype)
                w_cell = w_cell + w / len(shifts)  # cell value for the isotropic quantities
                if keep_yee_w:
                    yee_weights.append(w)
                self._append_staggered_sigma(cond, sigma, w, c, len(shifts))
            e = eps[:, c, None, None, None]
            if subpixel:
                # n2 <1/eps> + (1 - n2) / <eps> = (1 + n2 a) / <eps>, with the normal
                # of the voxel's own model (the unexplained variance takes the
                # one-sided differences of the mean eps), or only the latter
                # with subpixel_normal="gradient"
                mean_eps = jnp.sum(w * e, axis=0)
                a = _anisotropy(w, eps_np[:, c], sentinel)
                fd = _normal_squared(mean_eps, self._fd_spacing(shifts[min(c, len(shifts) - 1)]))[c]
                n2 = fd
                if voxel_normal:
                    tensor = tensors[0 if len(shifts) == rows else c]  # (4, nx, ny, nz)
                    along = tensor[:3] * keep
                    total = jnp.sum(along, axis=0) + tensor[3] + 1e-12
                    n2 = (along[c] + (tensor[3] + 1e-12) * fd) / total
                inv[c, c] = (1 + n2 * a) / mean_eps
            else:
                inv[c, c] = jnp.sum(w / e, axis=0)
            lap(f"E{'xyz'[c]}" if rows == 3 else "E", inv[c, c])
        zero = jnp.zeros_like(inv[0, 0])
        entries = [inv.get((c, d), zero) for c in range(3) for d in range(3)] if full else [inv[c, c] for c in range(rows)]
        arrays = arrays.at["inv_permittivities"].set(
            arrays.inv_permittivities.at[:, *self.grid_slice].set(jnp.stack(entries).astype(dtype))
        )
        lap("set_permittivity", arrays)
        return self._write_conductivity_and_dispersion(arrays, sigma, cond, w_cell, yee_weights or None, dtype)

    @timed("write_complete")
    def write_with_status(
        self,
        arrays: Any,
        params: dict[str, Array] | None = None,
        noise_key: Array | None = None,
        apply_roughness: bool = True,
    ) -> tuple[Any, dict[str, Array]]:
        """:meth:`write`, with the status of the interface model (``averaging="subpixel_full"``).

        Every E component c is sampled at its own Yee position (``yee_staggered``): the inverse of the
        voxel's complete effective permittivity there (:meth:`_complete_tensor`) is computed whole, and
        its row c written, all three entries (so the nine entries of a voxel come from three
        positions; each is a row of a symmetric positive-definite tensor of its own position). The
        materials' full tensors enter (scalars, diagonals and off-diagonal entries alike); a voxel of one
        material gets the exact inverse of its tensor. Conductivity and pole strengths are mixed linearly,
        as with the other averagings.

        Coverage failures (junction voxels beyond ``junction_voxels``, corner candidates beyond
        ``corner_columns``, a cell problem solved with a relative residual above 1e-4) are never kept
        silently: with ``interface_failure="raise"`` they raise where the result is concrete, and make the
        box NaN under tracing (so the result cannot be accepted); with ``"warn"`` they warn and keep the
        laminate or half-plane result. The other averagings return an empty status.

        Args:
            arrays: fdtdx ArrayContainer.
            params: device parameters (default: :meth:`init_params`).
            noise_key: PRNG key for the roughness.
            apply_roughness: if False, the nominal geometry.

        Returns:
            The arrays and the status: ``junctions``, ``junction_overflow``, ``corner_dropped`` and
            ``corner_uncertified`` (counts, summed over the Yee positions; the last: voxels of corner columns
            whose corner model is not certified, so partly one half-plane per level, reported but not a
            failure), ``closure_residual`` (the largest) and ``failed``.

        Raises:
            ValueError: a coverage failure with ``interface_failure="raise"`` and concrete values.
        """
        if self.averaging != "subpixel_full":
            return self.write(arrays, params, noise_key, apply_roughness), {}
        if arrays.inv_permittivities.shape[0] != 9:
            raise ValueError(
                "averaging='subpixel_full' needs fully anisotropic (9-component) arrays; build the "
                "device with from_layer_stack(..., averaging='subpixel_full')"
            )
        _check_materials(self.materials, self.averaging, self.name)
        dtype = arrays.inv_permittivities.dtype
        sigma = self._allowed_conductivity(arrays, dtype)
        shifts = _YEE_E_SHIFTS if self.yee_staggered else ((0.0, 0.0, 0.0),)
        capacity = self._junction_capacity()
        # the rows of the symmetric inverse in its 6-entry storage (xx, yy, zz, xy, xz, yz)
        row_entries = ((0, 3, 4), (3, 1, 5), (4, 5, 2))
        keep_yee_w = (
            len(shifts) == 3
            and arrays.dispersive_c1 is not None
            and arrays.dispersive_c3.shape[1] in (3, 9)
        )
        entries, cond, w_cell, statuses, yee_weights = [], [], 0.0, [], []
        for c in range(3):
            if c < len(shifts):
                g = self._interfaces(params, noise_key, apply_roughness, shifts[c])
                w = g.weights.astype(dtype)
                w_cell = w_cell + w / len(shifts)
                if keep_yee_w:
                    yee_weights.append(w)
                self._append_staggered_sigma(cond, sigma, w, c, len(shifts))
                inv6, status = self._complete_tensor(g, capacity)
                statuses.append({**status, "corner_dropped": g.corner_dropped, "corner_uncertified": g.corner_uncertified})
            entries += [inv6[..., e] for e in row_entries[c]]
            lap(f"E{'xyz'[c]}", inv6)
        status = {
            "junctions": sum(s["junctions"] for s in statuses),
            "junction_overflow": sum(s["junction_overflow"] for s in statuses),
            "corner_dropped": sum(s["corner_dropped"] for s in statuses),
            "corner_uncertified": sum(s["corner_uncertified"] for s in statuses),
            "closure_residual": jnp.max(jnp.stack([s["closure_residual"] for s in statuses])),
        }
        failed = (status["junction_overflow"] > 0) | (status["corner_dropped"] > 0) | ~(
            status["closure_residual"] <= _CLOSURE_RESIDUAL
        )
        status["failed"] = failed
        values = _primal(failed)
        if not isinstance(values, jax.core.Tracer) and bool(values):
            message = (
                f"StackDevice {self.name!r}: the interface model's coverage failed: "
                f"{int(_primal(status['junction_overflow']))} junction voxels beyond junction_voxels="
                f"{capacity}, {int(_primal(status['corner_dropped']))} corner columns beyond corner_columns="
                f"{self.corner_columns}, closure residual {float(_primal(status['closure_residual'])):.3g}. "
                "Rebuild the capacities for these parameters (place the device again, or set junction_voxels "
                "/ corner_columns larger), or pass interface_failure='warn' to keep the laminate / half-plane "
                "result there"
            )
            if self.interface_failure == "raise":
                raise ValueError(message)
            warnings.warn(message, StackDiagnosticWarning, stacklevel=2)
        stacked = jnp.stack(entries).astype(dtype)
        if self.interface_failure == "raise":  # a traced failure: no result to accept
            stacked = jnp.where(failed, jnp.nan, stacked)
        arrays = arrays.at["inv_permittivities"].set(arrays.inv_permittivities.at[:, *self.grid_slice].set(stacked))
        lap("set_permittivity", arrays)
        arrays = self._write_conductivity_and_dispersion(arrays, sigma, cond, w_cell, yee_weights or None, dtype)
        return arrays, status

    def _allowed_conductivity(self, arrays: Any, dtype: jnp.dtype) -> Array | None:
        """Allowed electric conductivity table (M, n_c) in ``dtype``, or None."""
        if arrays.electric_conductivity is None:
            return None
        n_c = arrays.electric_conductivity.shape[0]
        return jnp.asarray(
            compute_allowed_electric_conductivities(
                self.materials, isotropic=n_c == 1, diagonally_anisotropic=n_c == 3
            ),
            dtype=dtype,
        )

    @staticmethod
    def _append_staggered_sigma(
        cond: list[Array], sigma: Array | None, w: Array, component: int, n_shifts: int
    ) -> None:
        """Appends Yee-weighted conductivity row(s) for component ``c`` when ``sigma`` has 3 or 9 entries."""
        if sigma is None or n_shifts != 3:
            return
        if sigma.shape[1] == 3:
            cond.append(jnp.sum(w * sigma[:, component, None, None, None], axis=0))
        elif sigma.shape[1] == 9:
            for d in range(3):
                cond.append(jnp.sum(w * sigma[:, 3 * component + d, None, None, None], axis=0))

    def _write_conductivity_and_dispersion(
        self,
        arrays: Any,
        sigma: Array | None,
        cond: list[Array],
        w_cell: Array,
        yee_weights: Sequence[Array] | None,
        dtype: jnp.dtype,
    ) -> Any:
        """Writes mixed electric conductivity and dispersive pole coefficients for this box."""
        if sigma is not None:
            if not cond:
                cond = [jnp.sum(w_cell * sigma[:, k, None, None, None], axis=0) for k in range(sigma.shape[1])]
            mixed_cond = jnp.stack(cond) * _conductivity_scale(self._config)
            gradient_config = self._config.gradient_config
            if gradient_config is not None and gradient_config.method == "reversible":
                # fdtdx's reversible backward pass closes over the conductivity:
                # it gets no gradient there, and a traced one cannot be closed
                # over (place_on_grid warns that these gradients are inaccurate)
                mixed_cond = jax.lax.stop_gradient(mixed_cond)
            arrays = arrays.at["electric_conductivity"].set(
                arrays.electric_conductivity.at[:, *self.grid_slice].set(mixed_cond.astype(dtype))
            )
            lap("conductivity", arrays)
        if arrays.dispersive_c1 is not None:
            arrays = self._write_dispersion(arrays, w_cell, yee_weights=yee_weights)
        return arrays

    def interface_status(
        self, params: dict[str, Array] | None = None, noise_key: Array | None = None, apply_roughness: bool = True
    ) -> dict[str, Any]:
        """The interface model's status for these parameters, as numbers (``averaging="subpixel_full"``).

        Computes every Yee position's interfaces (no arrays are written) and returns the counts of
        :meth:`write_with_status`: junction voxels, those beyond ``junction_voxels``, corner columns beyond
        ``corner_columns``, voxels of corner columns the corner model does not certify, and the closure's
        largest residual. Use it to check coverage before an optimization, or after one at its final
        parameters.
        """
        p = {**self.init_params(), **(params or {})}
        shifts = _YEE_E_SHIFTS if self.yee_staggered else ((0.0, 0.0, 0.0),)
        capacity = self._junction_capacity()
        junctions = overflow = dropped = uncertified = jnp.zeros((), dtype=jnp.int32)
        residual = jnp.zeros((), dtype=self.dtype)
        for shift in shifts:
            j, o, d, u, r = _interface_status_core(
                self, p, noise_key, bool(apply_roughness), capacity, shift
            )
            junctions = junctions + j
            overflow = overflow + o
            dropped = dropped + d
            uncertified = uncertified + u
            residual = jnp.maximum(residual, r)
        return {
            "junctions": int(junctions),
            "junction_overflow": int(overflow),
            "corner_dropped": int(dropped),
            "corner_uncertified": int(uncertified),
            "closure_residual": float(residual),
            "junction_voxels": capacity,
        }

    def coarse_vertices(self, params: dict[str, Array], mesh: Any) -> Any:
        """How many vertices sit in grid cells wider than their target, per source layer.

        The target of a layer's vertices is the cell :func:`bifrost.mesh` asks of the materials that meet
        there: ``lambda / (n ppw)`` of the levels reading the layer and of their lateral neighbours, the
        smallest. A vertex counts along x or y where the cell holding it (the wider one on an edge) is wider
        than that, and along z where a level reading its layer, at its current zmin and thickness, spans a
        z cell wider than the level's own target. Vertices outside the box are counted apart.

        A diagnostic only: nothing limits the parameters, and a vertex that leaves its dilation shows up
        here. It runs under ``jax.jit`` (the mesh is static) and costs a search per vertex, so it can leave
        an optimizer's objective as auxiliary output::

            loss, report = objective(p)  # report = stack.coarse_vertices(p["stack"], mesh)
            print(report)                # a table, on concrete values

        It does not check how the device is painted: :meth:`interface_status` reports the corner and
        junction coverage.

        Args:
            params: the device's parameters (``{"vertices/<layer>": ..., "zmin/<level>": ...,
                "thickness/<level>": ...}``), concrete or traced.
            mesh: the :class:`bifrost.Mesh` the device runs on (a placed device must be on its grid).

        Returns:
            A :class:`bifrost.meshing.CoarseVertices`.
        """
        from .meshing import coarse_vertices

        return coarse_vertices(self, params, mesh)

    @timed("dispersion")
    def _write_dispersion(self, arrays: Any, w: Array, yee_weights: Sequence[Array] | None = None) -> Any:
        """Writes the pole coefficients of the box.

        c1, c2 are the slot constants in every voxel, also where no material
        with that pole is (c3 = 0 keeps P = 0 there, and the adjoint of c3 sees
        the material's recurrence); ``c3 = sum_m w_m c3_m`` (using per-axis Yee
        fractions ``yee_weights = (w_Ex, w_Ey, w_Ez)`` for 3- or 9-component
        couplings when staggered). Slots that other objects need beyond the
        stack's own are zeroed in the box.

        Args:
            arrays: fdtdx ArrayContainer with dispersive arrays.
            w: (M, nx, ny, nz) material fractions in fdtdx's material order.
            yee_weights: optional ``(w_Ex, w_Ey, w_Ez)`` staggered fractions.

        Returns:
            The arrays with c1, c2 and c3 overwritten in the box.

        Raises:
            ValueError: if the arrays have fewer pole slots than the device needs,
                or ``yee_weights`` does not have length 3.
        """
        n_slots, n_comp = arrays.dispersive_c1.shape[:2]
        n_coupling = arrays.dispersive_c3.shape[1]
        dtype = arrays.dispersive_c1.dtype
        dt = self._config.time_step_duration
        slots = pole_slots(self.materials)
        n = len(slots)
        if n > n_slots:
            raise ValueError(
                f"{self.name!r} needs {n} pole slots but the arrays have {n_slots}; "
                "were the arrays initialized with this device?"
            )
        c1 = jnp.zeros((n_slots, n_comp, *self.grid_shape), dtype=dtype)
        c2 = jnp.zeros((n_slots, n_comp, *self.grid_shape), dtype=dtype)
        c3 = jnp.zeros((n_slots, n_coupling, *self.grid_shape), dtype=dtype)
        if n:
            a1, a2, _ = compute_pole_coefficients(slots, dt)
            # (M, n, n_coupling): the slots' K (isotropic, diagonal, or row-major 3x3)
            _, _, a3 = compute_allowed_dispersive_coefficients(
                self.materials, dt, max_num_poles=n, num_components=n_comp, coupling_components=n_coupling
            )
            shape = (n, n_comp, *self.grid_shape)

            def per_slot(a: np.ndarray) -> Array:
                """Broadcasts one coefficient per slot to the box."""
                return jnp.broadcast_to(jnp.asarray(a, dtype)[:, None, None, None, None], shape)

            c1 = c1.at[:n].set(per_slot(a1))
            c2 = c2.at[:n].set(per_slot(a2))
            a3_arr = jnp.asarray(a3, dtype)
            if yee_weights is not None and len(yee_weights) != 3:
                raise ValueError(f"yee_weights must have 3 entries (Ex, Ey, Ez), got {len(yee_weights)}")
            if n_coupling == 1 or yee_weights is None:
                mixed = jnp.einsum(  # full precision: GPUs default to TF32 for matmuls
                    "msc,mxyz->scxyz",
                    a3_arr,
                    w.astype(dtype),
                    precision=jax.lax.Precision.HIGHEST,
                )
            elif n_coupling == 3:
                w_yee = jnp.stack([wy.astype(dtype) for wy in yee_weights])
                mixed = jnp.einsum(
                    "msc,cmxyz->scxyz",
                    a3_arr,
                    w_yee,
                    precision=jax.lax.Precision.HIGHEST,
                )
            elif n_coupling == 9:
                w_yee = jnp.stack([wy.astype(dtype) for wy in yee_weights])
                mixed = jnp.einsum(
                    "msab,amxyz->sabxyz",
                    a3_arr.reshape(a3_arr.shape[0], n, 3, 3),
                    w_yee,
                    precision=jax.lax.Precision.HIGHEST,
                ).reshape(n, 9, *self.grid_shape)
            else:
                raise ValueError(f"Unexpected dispersive_c3 coupling components {n_coupling}")
            c3 = c3.at[:n].set(mixed)
        for name, value in (
            ("dispersive_c1", c1),
            ("dispersive_c2", c2),
            ("dispersive_c3", c3),
        ):
            current = getattr(arrays, name)
            arrays = arrays.at[name].set(current.at[:, :, *self.grid_slice].set(value))
        return arrays

    # ------------------------------------------------------------ simulation setup

    def get_simulation_volume(
        self, name: str = "volume", material: fdtdx.Material | None = None
    ) -> tuple[fdtdx.SimulationVolume, list[PositionConstraint]]:
        """Simulation volume that is exactly the device box.

        Args:
            name: name of the volume.
            material: material of the volume (default: the background); the
                device overwrites the whole box anyway.

        Returns:
            The volume and the constraint that centers this device in it.
        """
        volume = fdtdx.SimulationVolume(
            name=name,
            partial_real_shape=self.partial_real_shape,
            material=material or self.materials[self.background],
        )
        return volume, [self.place_at_center(volume)]

    def rectilinear_grid(
        self,
        spacing_um: float,
        x_um: Sequence[float] | None = None,
        y_um: Sequence[float] | None = None,
        z_um: Sequence[float] | None = None,
    ) -> fdtdx.core.grid.RectilinearGrid:
        """An fdtdx ``RectilinearGrid`` over this device's box (the simulation volume of
        :meth:`get_simulation_volume`), from cell edges in stack coordinates::

            z = bf.graded_edges(z0, z1, 0.06, fine=[(-0.05, 0.3, 0.02)], snap=[0.0, 0.22])
            config = stack.get_simulation_config(time=..., grid=stack.rectilinear_grid(0.06, z_um=z))

        Any axis may be graded. Its edges are in stack coordinates, from the box's lower face to
        its upper one (the box after ``boundary_offset``, which need not be symmetric).

        Args:
            spacing_um: cell size (um) along the axes whose edges are not given; the box must be
                a whole number of them there.
            x_um: x edges (um, gds coordinates) from the box's lower face to its upper one, or None.
            y_um: y edges, or None.
            z_um: z edges (um, stack heights), or None.

        Returns:
            The grid, its edges in metres from the box's center (fdtdx's frame), in the device's dtype.

        Raises:
            ValueError: if the box is not whole cells along an axis without edges, or given edges
                are not strictly increasing or do not end on the box's faces (within 1 pm).
        """
        if self.partial_real_shape is None or None in tuple(self.partial_real_shape):
            raise ValueError(f"StackDevice {self.name!r} has no box (partial_real_shape)")
        if isinstance(spacing_um, bool) or not isinstance(spacing_um, (int, float, np.number)):
            raise TypeError(f"spacing_um must be a positive finite number, got {spacing_um!r}")
        spacing = float(spacing_um)
        if not math.isfinite(spacing) or spacing <= 0.0:
            raise ValueError(f"spacing_um must be finite and > 0, got {spacing_um!r}")
        sizes = [float(s) * 1e6 for s in self.partial_real_shape]
        middles = (float(self.center[0]), float(self.center[1]), float(self.z_center))
        out = []
        for axis, given in enumerate((x_um, y_um, z_um)):
            mid, size = middles[axis], sizes[axis]
            lo, hi = mid - 0.5 * size, mid + 0.5 * size
            if given is None:
                n = round(size / spacing)
                if n < 1 or abs(n * spacing - size) > 1e-6:
                    raise ValueError(
                        f"StackDevice {self.name!r}: the box is {size:g} um along {'xyz'[axis]}, not a whole "
                        f"number of {spacing:g} um cells; pass its edges ({'xyz'[axis]}_um)"
                    )
                edges = spacing * (np.arange(n + 1) - 0.5 * n)
            else:
                e = np.array(given, dtype=np.float64)
                if e.ndim != 1 or len(e) < 2 or not np.all(np.isfinite(e)) or np.any(np.diff(e) <= 0):
                    raise ValueError(f"{'xyz'[axis]}_um must be finite strictly increasing edges, got {given!r}")
                if abs(e[0] - lo) > 1e-6 or abs(e[-1] - hi) > 1e-6:
                    raise ValueError(
                        f"StackDevice {self.name!r}: {'xyz'[axis]}_um must run from the box's face {lo:g} to "
                        f"{hi:g} um, not {e[0]:g} to {e[-1]:g}"
                    )
                e[0], e[-1] = lo, hi
                edges = e - mid
            edges_grid = jnp.asarray(edges * 1e-6, dtype=self.dtype)
            if not bool(jnp.all(jnp.diff(edges_grid) > 0)):
                raise ValueError(
                    f"StackDevice {self.name!r}: {'xyz'[axis]}_um collapses adjacent edges after casting to "
                    f"{np.dtype(self.dtype).name}"
                )
            out.append(edges_grid)
        return fdtdx.core.grid.RectilinearGrid(x_edges=out[0], y_edges=out[1], z_edges=out[2])

    def _grid_in_dtype(self, grid: Any) -> Any:
        """A ``RectilinearGrid`` with its edges in the device's dtype (other grids as they are).

        fdtdx's curl on a non-uniform grid scales the fields by the cells' widths, in the edges' dtype: float64
        edges (e.g. numpy edges when JAX's 64-bit mode is on) would turn a float32
        simulation's fields into float64 in the time loop, which fails.
        """
        if not isinstance(grid, fdtdx.core.grid.RectilinearGrid):
            return grid
        edges = [grid.edges(a) for a in range(3)]
        if all(_same_dtype(e.dtype, self.dtype) for e in edges):
            return grid
        return fdtdx.core.grid.RectilinearGrid(
            **{f"{name}_edges": jnp.asarray(e, dtype=self.dtype) for name, e in zip("xyz", edges, strict=True)}
        )

    def get_simulation_config(self, **kwargs: Any) -> SimulationConfig:
        """The ``fdtdx.SimulationConfig`` for this device, in the device's precision.

        Sets ``dtype`` from :attr:`dtype` and passes everything else on to
        ``fdtdx.SimulationConfig``, so the precision is chosen once, on the
        device (a ``RectilinearGrid``'s edges are converted to it too)::

            stack = bf.StackDevice.from_layer_stack(..., dtype=jnp.float64)
            config = stack.get_simulation_config(time=100e-15, grid=UniformGrid(spacing=50e-9))

        Args:
            **kwargs: the ``fdtdx.SimulationConfig`` arguments (``time``,
                ``grid``, ``backend``, ``gradient_config``, ...), but not ``dtype``.

        Returns:
            The simulation config, with the device's ``dtype``.

        Raises:
            TypeError: if ``dtype`` is passed (the device sets it).
        """
        if "dtype" in kwargs:
            raise TypeError(
                "get_simulation_config takes its dtype from the StackDevice: "
                "build the device with StackDevice(dtype=...) instead of passing dtype here"
            )
        if "grid" in kwargs:
            kwargs["grid"] = self._grid_in_dtype(kwargs["grid"])
        return SimulationConfig(dtype=self.dtype, **kwargs)

    def _check_precision(self, config: SimulationConfig) -> None:
        """Checks that the simulation config and JAX's 64-bit mode fit the device's dtype.

        Args:
            config: the simulation config the device is placed with.

        Raises:
            ValueError: if ``config.dtype`` is not the device's dtype, or a non-uniform grid's edges are in
                another dtype (fdtdx would run the fields in theirs).
            RuntimeError: if a float64 device is placed with JAX's 64-bit mode off.
        """
        if not _same_dtype(config.dtype, self.dtype):
            raise ValueError(
                f"StackDevice {self.name!r} has dtype {np.dtype(self.dtype)} but the simulation config has "
                f"dtype {np.dtype(config.dtype)}; build the config with "
                "device.get_simulation_config(...), which takes the device's dtype"
            )
        grid = config.resolved_grid
        if config.has_nonuniform_grid and isinstance(grid, fdtdx.core.grid.RectilinearGrid):
            dtypes = {np.dtype(grid.edges(a).dtype).name for a in range(3)}
            if dtypes != {np.dtype(self.dtype).name}:
                raise ValueError(
                    f"StackDevice {self.name!r} has dtype {np.dtype(self.dtype)} but the grid's edges are "
                    f"{', '.join(sorted(dtypes))}: fdtdx would turn the fields into those in its time loop. Build "
                    "the config with device.get_simulation_config(grid=...) (or the grid with "
                    "device.rectilinear_grid), which converts them"
                )
        if _same_dtype(self.dtype, np.float64) and not jax.config.jax_enable_x64:
            raise RuntimeError(
                f"StackDevice {self.name!r} has dtype float64, but JAX's 64-bit mode is off, which would "
                "truncate everything to float32; enable it with jax.config.update('jax_enable_x64', True) "
                "before placing or running a float64 device"
            )
        if _same_dtype(self.dtype, np.float32) and self.source_vertices:
            if grid is not None:
                min_sp = getattr(grid, "min_spacing", None)
                if min_sp is None:
                    min_sp = min(float(np.min(np.diff(np.asarray(grid.edges(a))))) for a in range(3))
                h_um = float(min_sp) * 1e6
            elif hasattr(config.grid, "min_spacing"):
                h_um = float(config.grid.min_spacing) * 1e6
            else:
                h_um = float(config.uniform_spacing()) * 1e6
            max_coord = max((float(np.abs(v).max()) for v in self.source_vertices if v.size), default=0.0)
            quant_um = max_coord * float(np.finfo(np.float32).eps)
            if h_um > 0.0 and quant_um > 0.01 * h_um:
                warnings.warn(
                    f"StackDevice {self.name!r}: float32 vertex coordinates up to {max_coord:g} um have ULP "
                    f"rounding (~{quant_um:.3g} um) exceeding 1% of the finest grid spacing ({h_um:.3g} um); "
                    "use dtype=jnp.float64 or center the component near (0, 0)",
                    UserWarning,
                    stacklevel=3,
                )

    def _port(self, port: str) -> _Port:
        """Looks up a port by name.

        Args:
            port: port name.

        Returns:
            The port.

        Raises:
            KeyError: if the component has no such port.
        """
        for p in self.ports:
            if p.name == port:
                return p
        raise KeyError(f"Unknown port {port!r}; {self.name!r} has ports {[p.name for p in self.ports]}")

    def _plane_frame(
        self, plane: str, port_offset: float, level: str | None = None
    ) -> tuple[_Port, int, int, list[float], float]:
        """Frame of a plane through the simulation volume.

        The three :data:`PLANE_PORTS` give the plane sources and detectors a
        frame without a component port: a plane parallel to the volume's faces,
        centered in the volume. The volume is the device box, so its center is
        ``(self.center[0], self.center[1], self.z_center)``.

        A plane normal to z (``"xy"``) cannot come from a component port —
        ports run along x / y — so it uses the same synthetic frame, with the
        two in-plane extents as width and height.

        ``level`` sets the plane's height: a level name puts the plane at the
        middle of that level's z span, and :data:`Z_CENTER_LEVEL` (or ``None``)
        at the volume's own mid-height. For the ``"xy"`` plane, whose normal is
        z, that *is* the plane's position; for ``"xz"`` / ``"yz"`` it moves the
        plane's in-plane vertical center.

        Args:
            plane: ``"xy"``, ``"xz"`` or ``"yz"``.
            port_offset: shift (um) along ``"in"`` (``-normal``); negative moves toward
                the outward direction (``+normal``). Note: absolute viewer placement
                with ``port="None"`` keeps ``"in"`` along ``+axis``.
            level: a level name, :data:`Z_CENTER_LEVEL`, or ``None`` for the
                volume's mid-height.

        Returns:
            A synthetic port whose ``width`` / ``height`` are the volume's
            in-plane extents, the normal axis (0 = x, 1 = y, 2 = z), the
            outward sign (+1), the center (x, y, z) in um and the height.

        Raises:
            ValueError: the device has no box (``partial_real_shape``), so the
                volume's extent is unknown.
            KeyError: ``level`` is not a level of this device.
        """
        axis = PLANE_PORTS[plane]
        if self.partial_real_shape is None or None in tuple(self.partial_real_shape):
            raise ValueError(f"StackDevice {self.name!r} has no box (partial_real_shape)")
        size = tuple(float(s) * 1e6 for s in self.partial_real_shape)
        u_axis, v_axis = _IN_PLANE[axis]
        width, height = size[u_axis], size[v_axis]
        z = float(self.z_center)
        if level is not None and level != Z_CENTER_LEVEL:
            match = [lv for lv in self.levels if lv.name == level]
            if not match:
                raise KeyError(f"Level {level!r} is not in {self.name!r}: {[lv.name for lv in self.levels]}")
            spans = [_span(lv) for lv in match]
            z = 0.5 * (min(a for a, _ in spans) + max(b for _, b in spans))
        center = [float(self.center[0]), float(self.center[1]), z]
        # Keep outward = +1 so "in" points along -normal and +port_offset moves along "in";
        # viewer placement with port="None" (no port) keeps "in" -> +axis.
        center[axis] -= port_offset
        port = _Port(name=plane, x=center[0], y=center[1], orientation=None, width=width, layer=-1)
        return port, axis, 1, center, height

    def _port_frame(
        self, port: str, level: str | None, port_offset: float
    ) -> tuple[_Port, int, int, list[float], float]:
        """Position and orientation of a port in the stack.

        Args:
            port: port name.
            level: level whose z span is used (default: the union of the levels
                that draw the port's layer). For a :data:`PLANE_PORTS` name it
                sets the plane's height: a level's span middle, or
                :data:`Z_CENTER_LEVEL` / ``None`` for the volume's mid-height.
            port_offset: shift (um) along the port's injection direction (into
                the component); negative moves out of it.

        Returns:
            The port, its normal axis (0 = x, 1 = y), its outward sign (+1 / -1),
            the center (x, y, z) in um (z: middle of the span) and the height of
            the span in um.

        ``port`` may also name one of :data:`PLANE_PORTS` (``"xy"``, ``"xz"``,
        ``"yz"``): the plane through the simulation volume's center. Those are
        not ports of the device — they are not in :attr:`ports` — they only
        give the plane sources and detectors a frame, sized by the volume's own
        in-plane extent rather than a port's.

        Raises:
            KeyError: unknown port or level.
            ValueError: a port without orientation or not along x / y, or a port
                layer that no level draws.
        """
        if port in PLANE_PORTS:
            return self._plane_frame(port, port_offset, level)
        p = self._port(port)
        if p.orientation is None:
            raise ValueError(f"Port {port!r} has no orientation")
        quarter = round(p.orientation / 90.0)
        if abs(p.orientation - 90.0 * quarter) > 1e-6:
            raise ValueError(f"Port {port!r} points at {p.orientation} deg; fdtdx planes need 0, 90, 180 or 270")
        quarter %= 4
        axis, sign = quarter % 2, (1 if quarter < 2 else -1)
        if level is not None and level != Z_CENTER_LEVEL:
            match = [lv for lv in self.levels if lv.name == level]
            if not match:
                raise KeyError(f"Level {level!r} is not in {self.name!r}: {[lv.name for lv in self.levels]}")
        else:
            match = [
                lv for lv in self.levels
                if p.layer in _expr_positive_layers(lv.expr) or p.layer == lv.fill_layer
            ]
            if not match:
                raise ValueError(f"No level of {self.name!r} draws the layer of port {port!r}; pass level=...")
        spans = [_span(lv) for lv in match]
        lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
        # Z_CENTER_LEVEL pins the plane to the volume's mid-height while keeping
        # the port's own span height; otherwise the span's middle sets z.
        z = float(self.z_center) if level == Z_CENTER_LEVEL else 0.5 * (lo + hi)
        center = [p.x, p.y, z]
        center[axis] -= sign * port_offset  # sign: the port's outward orientation
        return p, axis, sign, center, hi - lo

    def _place_at(
        self, obj: Any, center: Sequence[float], own_positions: Sequence[float] = (0, 0, 0)
    ) -> list[PositionConstraint]:
        """Constraint that puts the center of ``obj`` at a stack point.

        Args:
            obj: fdtdx object.
            center: (x, y, z) in um (gds xy, stack z).
            own_positions: the point of ``obj`` that sits on ``center`` along each
                axis, ``-1`` (lower side) to ``1`` (upper side); the default is
                its center.

        Returns:
            One position constraint relative to this device.
        """
        reference = (self.center[0], self.center[1], self.z_center)
        margins = tuple((c - r) * 1e-6 for c, r in zip(center, reference, strict=True))
        return [
            obj.place_relative_to(
                self, axes=(0, 1, 2), own_positions=tuple(own_positions), other_positions=(0, 0, 0),
                margins=margins,
            )
        ]

    def _at_port(
        self,
        cls: type,
        port: str,
        port_padding: Sequence[float],
        suffix: str,
        *,
        wavelengths_um: Sequence[float] | None,
        direction: PortDirection | None,
        level: str | None,
        port_offset: float,
        name: str | None,
        grid: Any = None,
        **fields: Any,
    ) -> tuple[Any, list[PositionConstraint]]:
        """Builds a plane source / detector on a port.

        Args:
            cls: fdtdx source or detector class.
            port: port name.
            port_padding: see :func:`plane_extents`.
            suffix: default name is ``<port>_<suffix>``.
            wavelengths_um: the wavelengths (um) whose mean is the padding's unit.
            direction: "in" / "out" relative to the component, or None for
                classes without a direction.
            level: see :meth:`_port_frame`.
            port_offset: see :meth:`_port_frame`.
            name: object name.
            grid: the simulation's config or ``RectilinearGrid``, or None. Along
                an in-plane axis where its cells differ, the plane is given in whole
                cells: its faces snap to the nearest edges at its own position, since
                fdtdx counts a physical size's cells from the domain's lower face. The
                device must fill the simulation volume (:meth:`get_simulation_volume`,
                any ``boundary_offset``): the grid's middle is the box's center.
            **fields: other fields of ``cls``.

        Returns:
            The object and the constraints placing it.
        """
        p, axis, sign, center, height = self._port_frame(port, level, port_offset)
        if p.width <= 0:
            raise ValueError(f"Port {port!r} has no width")
        width_um, height_um = plane_extents(p.width, height, port_padding, wavelengths_um)
        real: list[float | None] = [None, None, None]
        cells: list[int | None] = [None, None, None]
        cells[axis] = 1  # one cell thick, normal to the plane
        u_axis, v_axis = _IN_PLANE[axis]
        real[u_axis] = width_um * 1e-6
        real[v_axis] = height_um * 1e-6
        snapped_extents = [width_um, height_um]
        if grid is not None:
            for idx, (a, extent) in enumerate(((u_axis, width_um), (v_axis, height_um))):
                snapped = self._snap_extent_span(grid, a, center[a], extent)
                if snapped is not None:
                    cells[a], center[a], snapped_extents[idx] = snapped
                    real[a] = None
        if issubclass(cls, fdtdx.GaussianPlaneSource) and fields.get("radius") is None:
            fields["radius"] = 0.5 * min(snapped_extents) * 1e-6
        if direction is not None:
            fields["direction"] = _axis_direction(direction, sign)
        if "dtype" not in fields:
            # fdtdx's detectors keep their own dtype (float32 / complex64) instead of config.dtype, and
            # would store a float64 run's fields in single precision; sources follow config.dtype
            if issubclass(cls, (fdtdx.PhasorDetector, DiffractiveDetector)):
                fields["dtype"] = _complex_dtype(self.dtype)
            elif issubclass(cls, fdtdx.Detector):
                fields["dtype"] = self.dtype
        obj = cls(
            name=name or f"{port}_{suffix}",
            partial_real_shape=tuple(real),
            partial_grid_shape=tuple(cells),
            **fields,
        )
        own = detector_plane_own_positions(axis) if isinstance(obj, fdtdx.Detector) else (0, 0, 0)
        return obj, self._place_at(obj, center, own)

    def _snap_extent_span(
        self, grid: Any, axis: int, center_um: float, extent_um: float
    ) -> tuple[int, float, float] | None:
        """Whole cells, snapped center (um), and snapped span (um) along a non-uniform axis."""
        resolved = grid.resolved_grid if isinstance(grid, SimulationConfig) else grid
        if not isinstance(resolved, fdtdx.core.grid.RectilinearGrid):
            return None
        raw = np.asarray(resolved.edges(axis))
        edges = raw.astype(np.float64) * 1e6
        widths = np.diff(edges)
        eps = float(np.finfo(raw.dtype).eps) if np.issubdtype(raw.dtype, np.floating) else 0.0
        floor = 8.0 * eps * float(np.max(np.abs(edges)))
        if np.max(np.abs(widths - widths[0])) <= _FRAME_RTOL * widths[0] + floor:
            return None
        size = float(self.partial_real_shape[axis]) * 1e6
        if abs((edges[-1] - edges[0]) - size) > 0.01 * float(np.min(widths)):
            raise ValueError(
                f"StackDevice {self.name!r}: grid= needs the grid of the device's own simulation volume "
                f"(get_simulation_volume): along {'xyz'[axis]} it spans {edges[-1] - edges[0]:g} um, the box "
                f"{size:g} um. In a larger volume, leave grid= out and give the plane's cells yourself."
            )
        middle = (self.center[0], self.center[1], self.z_center)[axis]
        offset = 0.5 * (edges[0] + edges[-1]) - middle  # stack -> grid coordinates (the box's center)
        lo = int(np.argmin(np.abs(edges - (center_um + offset - 0.5 * extent_um))))
        hi = int(np.argmin(np.abs(edges - (center_um + offset + 0.5 * extent_um))))
        if hi <= lo:
            hi = min(lo + 1, len(edges) - 1)
            lo = hi - 1
        return hi - lo, 0.5 * (edges[lo] + edges[hi]) - offset, float(edges[hi] - edges[lo])

    def _snap_extent(self, grid: Any, axis: int, center_um: float, extent_um: float) -> tuple[int, float] | None:
        """Whole cells for an extent along a non-uniform axis of the grid the device fills.

        Args:
            grid: a ``SimulationConfig`` or a ``RectilinearGrid`` spanning this device's box.
            axis: 0, 1 or 2.
            center_um: the extent's center (stack coordinates, um).
            extent_um: its length (um).

        Returns:
            (cells, snapped center in um): the extent's faces moved to the nearest edges, at
            least one cell; None where the grid's cells along ``axis`` are all one size (or
            the grid is not resolved), which fdtdx places exactly.

        Raises:
            ValueError: if the grid does not span the device's box along ``axis``: the device
                must fill the simulation volume, whose middle is then the box's center.
        """
        snapped = self._snap_extent_span(grid, axis, center_um, extent_um)
        return None if snapped is None else (snapped[0], snapped[1])

    # ------------------------------------------------------------ sources on ports

    def create_mode_plane_source_at(
        self,
        port: str,
        port_padding: Sequence[float],
        wave_character: fdtdx.WaveCharacter,
        temporal_profile: fdtdx.TemporalProfile | None = None,
        mode_index: int = 0,
        filter_pol: Literal["te", "tm"] | None = None,
        direction: PortDirection = "in",
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        grid: Any = None,
        **kwargs: Any,
    ) -> tuple[ModePlaneSource, list[PositionConstraint]]:
        """Waveguide mode source on a port.

        Args:
            port: port name; the plane is centered on it (in z on the middle of
                the port's level) and normal to it, or ``"xy"``, ``"xz"``,
                ``"yz"`` (:data:`PLANE_PORTS`) for a full-volume cross-section.
            port_padding: (lateral, vertical): the plane extends this far past the
                port's width and the level's height on each side, in units of the
                mean wavelength of this object's wave characters.
            wave_character: wavelength / frequency of the mode.
            temporal_profile: pulse shape (default: fdtdx's single frequency).
            mode_index: mode order (0 = fundamental).
            filter_pol: only "te" or "tm" modes (default: all).
            direction: "in" launches into the component, "out" away from it.
            level: level whose height and middle set the plane's z (default:
                the levels that draw the port's layer), or :data:`Z_CENTER_LEVEL`
                (``"z_center"``) for the stack's vertical middle.
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the plane out, e.g. into a port extension.
            name: object name (default ``<port>_mode_source``).
            grid: optional ``SimulationConfig`` or ``RectilinearGrid`` spanning this
                device's box to snap the plane's extents to non-uniform cell edges.
            **kwargs: other ModePlaneSource fields.

        Returns:
            The source and the constraints placing it relative to this device.
        """
        fields = _given(temporal_profile=temporal_profile, filter_pol=filter_pol)
        return self._at_port(
            ModePlaneSource, port, port_padding, "mode_source",
            wavelengths_um=[wave_character.get_wavelength() * 1e6],
            direction=direction, level=level,
            port_offset=port_offset, name=name, grid=grid, wave_character=wave_character, mode_index=mode_index,
            **fields, **kwargs,
        )

    def create_gaussian_plane_source_at(
        self,
        port: str,
        port_padding: Sequence[float],
        wave_character: fdtdx.WaveCharacter,
        temporal_profile: fdtdx.TemporalProfile | None = None,
        radius: float | None = None,
        std: float = 1 / 3,
        fixed_E_polarization_vector: tuple[float, float, float] | None = None,
        fixed_H_polarization_vector: tuple[float, float, float] | None = None,
        direction: PortDirection = "in",
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        grid: Any = None,
        **kwargs: Any,
    ) -> tuple[fdtdx.GaussianPlaneSource, list[PositionConstraint]]:
        """Gaussian beam source on a port.

        Args:
            port: port name; the plane is centered on it and normal to it, or
                ``"xy"``, ``"xz"``, ``"yz"`` (:data:`PLANE_PORTS`) for a
                full-volume cross-section.
            port_padding: (lateral, vertical): the plane extends this far past the
                port's width and the level's height on each side, in units of the
                mean wavelength of this object's wave characters.
            wave_character: wavelength / frequency.
            temporal_profile: pulse shape (default: fdtdx's single frequency).
            radius: beam radius in um (default: half the smaller plane side).
            std: standard deviation of the Gaussian relative to ``radius``.
            fixed_E_polarization_vector: E polarization (global x, y, z).
            fixed_H_polarization_vector: H polarization (alternative to E).
            direction: "in" launches into the component, "out" away from it.
            level: level whose height and middle set the plane's z, or
                :data:`Z_CENTER_LEVEL` (``"z_center"``) for the stack's vertical middle.
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the plane out, e.g. into a port extension.
            name: object name (default ``<port>_gaussian_source``).
            grid: optional ``SimulationConfig`` or ``RectilinearGrid`` spanning this
                device's box to snap the plane's extents to non-uniform cell edges.
            **kwargs: other GaussianPlaneSource fields.

        Returns:
            The source and the constraints placing it relative to this device.
        """
        wavelengths_um = [wave_character.get_wavelength() * 1e6]
        fields = _given(
            radius=None if radius is None else radius * 1e-6,
            temporal_profile=temporal_profile,
            fixed_E_polarization_vector=fixed_E_polarization_vector,
            fixed_H_polarization_vector=fixed_H_polarization_vector,
        )
        return self._at_port(
            fdtdx.GaussianPlaneSource, port, port_padding, "gaussian_source",
            wavelengths_um=wavelengths_um,
            direction=direction,
            level=level, port_offset=port_offset, name=name, grid=grid, wave_character=wave_character,
            std=std, **fields, **kwargs,
        )

    def create_uniform_plane_source_at(
        self,
        port: str,
        port_padding: Sequence[float],
        wave_character: fdtdx.WaveCharacter,
        temporal_profile: fdtdx.TemporalProfile | None = None,
        amplitude: float = 1.0,
        fixed_E_polarization_vector: tuple[float, float, float] | None = None,
        fixed_H_polarization_vector: tuple[float, float, float] | None = None,
        direction: PortDirection = "in",
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        grid: Any = None,
        **kwargs: Any,
    ) -> tuple[fdtdx.UniformPlaneSource, list[PositionConstraint]]:
        """Uniform plane-wave source on a port.

        Args:
            port: port name; the plane is centered on it and normal to it, or
                ``"xy"``, ``"xz"``, ``"yz"`` (:data:`PLANE_PORTS`) for a
                full-volume cross-section.
            port_padding: (lateral, vertical): the plane extends this far past the
                port's width and the level's height on each side, in units of the
                mean wavelength of this object's wave characters.
            wave_character: wavelength / frequency.
            temporal_profile: pulse shape (default: fdtdx's single frequency).
            amplitude: field amplitude.
            fixed_E_polarization_vector: E polarization (global x, y, z).
            fixed_H_polarization_vector: H polarization (alternative to E).
            direction: "in" launches into the component, "out" away from it.
            level: level whose height and middle set the plane's z, or
                :data:`Z_CENTER_LEVEL` (``"z_center"``) for the stack's vertical middle.
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the plane out, e.g. into a port extension.
            name: object name (default ``<port>_uniform_source``).
            grid: optional ``SimulationConfig`` or ``RectilinearGrid`` spanning this
                device's box to snap the plane's extents to non-uniform cell edges.
            **kwargs: other UniformPlaneSource fields.

        Returns:
            The source and the constraints placing it relative to this device.
        """
        fields = _given(
            temporal_profile=temporal_profile,
            fixed_E_polarization_vector=fixed_E_polarization_vector,
            fixed_H_polarization_vector=fixed_H_polarization_vector,
        )
        return self._at_port(
            fdtdx.UniformPlaneSource, port, port_padding, "uniform_source",
            wavelengths_um=[wave_character.get_wavelength() * 1e6],
            direction=direction,
            level=level, port_offset=port_offset, name=name, grid=grid, wave_character=wave_character,
            amplitude=amplitude, **fields, **kwargs,
        )

    def create_hard_constant_amplitude_plane_source_at(
        self,
        port: str,
        port_padding: Sequence[float],
        wave_character: fdtdx.WaveCharacter,
        temporal_profile: fdtdx.TemporalProfile | None = None,
        amplitude: float = 1.0,
        fixed_E_polarization_vector: tuple[float, float, float] | None = None,
        fixed_H_polarization_vector: tuple[float, float, float] | None = None,
        direction: PortDirection = "in",
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        grid: Any = None,
        **kwargs: Any,
    ) -> tuple[HardConstantAmplitudePlanceSource, list[PositionConstraint]]:
        """Hard (field-overwriting) constant-amplitude plane source on a port.

        Args:
            port: port name; the plane is centered on it and normal to it, or
                ``"xy"``, ``"xz"``, ``"yz"`` (:data:`PLANE_PORTS`) for a
                full-volume cross-section.
            port_padding: (lateral, vertical): the plane extends this far past the
                port's width and the level's height on each side, in units of the
                mean wavelength of this object's wave characters.
            wave_character: wavelength / frequency.
            temporal_profile: pulse shape (default: fdtdx's single frequency).
            amplitude: field amplitude.
            fixed_E_polarization_vector: E polarization (global x, y, z).
            fixed_H_polarization_vector: H polarization (alternative to E).
            direction: "in" launches into the component, "out" away from it.
            level: level whose height and middle set the plane's z, or
                :data:`Z_CENTER_LEVEL` (``"z_center"``) for the stack's vertical middle.
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the plane out, e.g. into a port extension.
            name: object name (default ``<port>_hard_source``).
            grid: optional ``SimulationConfig`` or ``RectilinearGrid`` spanning this
                device's box to snap the plane's extents to non-uniform cell edges.
            **kwargs: other HardConstantAmplitudePlanceSource fields.

        Returns:
            The source and the constraints placing it relative to this device.
        """
        fields = _given(
            temporal_profile=temporal_profile,
            fixed_E_polarization_vector=fixed_E_polarization_vector,
            fixed_H_polarization_vector=fixed_H_polarization_vector,
        )
        return self._at_port(
            HardConstantAmplitudePlanceSource, port, port_padding, "hard_source",
            wavelengths_um=[wave_character.get_wavelength() * 1e6],
            direction=direction,
            level=level, port_offset=port_offset, name=name, grid=grid, wave_character=wave_character,
            amplitude=amplitude, **fields, **kwargs,
        )

    def create_point_dipole_source_at(
        self,
        port: str,
        wave_character: fdtdx.WaveCharacter,
        polarization: int | Literal["x", "y", "z"],
        temporal_profile: fdtdx.TemporalProfile | None = None,
        azimuth_angle: float = 0.0,
        elevation_angle: float = 0.0,
        source_type: Literal["electric", "magnetic"] = "electric",
        amplitude: float = 1.0,
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        **kwargs: Any,
    ) -> tuple[fdtdx.PointDipoleSource, list[PositionConstraint]]:
        """Point dipole at a port (one cell; a point has no size to scale).

        Args:
            port: port name; the dipole sits at its center, in z at the middle
                of the port's level, or ``"xy"``, ``"xz"``, ``"yz"``
                (:data:`PLANE_PORTS`) to center on the stack.
            wave_character: wavelength / frequency.
            polarization: dipole axis, 0 / 1 / 2 or "x" / "y" / "z".
            temporal_profile: pulse shape (default: fdtdx's single frequency).
            azimuth_angle: rotation (degrees) about the vertical axis.
            elevation_angle: rotation (degrees) about the horizontal axis.
            source_type: "electric" or "magnetic".
            amplitude: dipole amplitude.
            level: level whose middle sets the dipole's z, or
                :data:`Z_CENTER_LEVEL` (``"z_center"``) for the stack's vertical middle.
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the dipole out, e.g. into a port extension.
            name: object name (default ``<port>_dipole_source``).
            **kwargs: other PointDipoleSource fields.

        Returns:
            The source and the constraints placing it relative to this device.
        """
        _, _, _, center, _ = self._port_frame(port, level, port_offset)
        if isinstance(polarization, bool) or not isinstance(polarization, (str, int, np.integer)):
            raise TypeError(
                f"polarization must be 0, 1, 2 or 'x', 'y', 'z', got {type(polarization).__name__}: {polarization!r}"
            )
        if isinstance(polarization, str):
            axis_map = {"x": 0, "y": 1, "z": 2}
            if polarization not in axis_map:
                raise ValueError(f"polarization must be 0, 1, 2 or 'x', 'y', 'z', got {polarization!r}")
            axis = axis_map[polarization]
        else:
            axis = int(polarization)
            if axis not in (0, 1, 2):
                raise ValueError(f"polarization must be 0, 1, 2 or 'x', 'y', 'z', got {polarization!r}")
        source = fdtdx.PointDipoleSource(
            name=name or f"{port}_dipole_source",
            partial_grid_shape=(1, 1, 1),
            wave_character=wave_character,
            polarization=axis,
            azimuth_angle=azimuth_angle,
            elevation_angle=elevation_angle,
            source_type=source_type,
            amplitude=amplitude,
            **_given(temporal_profile=temporal_profile),
            **kwargs,
        )
        return source, self._place_at(source, center)

    # ------------------------------------------------------------ detectors on ports

    def create_mode_overlap_detector_at(
        self,
        port: str,
        port_padding: Sequence[float],
        wave_characters: Sequence[fdtdx.WaveCharacter],
        mode_index: int = 0,
        filter_pol: Literal["te", "tm"] | None = None,
        scaling_mode: Literal["continuous", "pulse"] | None = None,
        direction: PortDirection = "out",
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        grid: Any = None,
        **kwargs: Any,
    ) -> tuple[ModeOverlapDetector, list[PositionConstraint]]:
        """Mode overlap (mode amplitude) detector on a port.

        Args:
            port: port name; the plane is centered on it and normal to it, or
                ``"xy"``, ``"xz"``, ``"yz"`` (:data:`PLANE_PORTS`) for a
                full-volume cross-section.
            port_padding: (lateral, vertical): the plane extends this far past the
                port's width and the level's height on each side, in units of the
                mean wavelength of this object's wave characters.
            wave_characters: wavelengths / frequencies to detect.
            mode_index: mode order (0 = fundamental).
            filter_pol: only "te" or "tm" modes (default: all).
            scaling_mode: "continuous" or "pulse" phasor normalization (default:
                fdtdx's, "continuous"); use "pulse" with pulsed sources.
            direction: "out" measures the mode leaving the component, "in" the
                mode entering it.
            level: level whose height and middle set the plane's z, or
                :data:`Z_CENTER_LEVEL` (``"z_center"``) for the stack's vertical middle.
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the plane out, e.g. into a port extension.
            name: object name (default ``<port>_mode_detector``).
            grid: optional ``SimulationConfig`` or ``RectilinearGrid`` spanning this
                device's box to snap the plane's extents to non-uniform cell edges.
            **kwargs: other ModeOverlapDetector fields (``dtype`` defaults to the
                device's precision: complex64 or complex128).

        Returns:
            The detector and the constraints placing it relative to this device.
        """
        fields = _given(filter_pol=filter_pol, scaling_mode=scaling_mode)
        return self._at_port(
            ModeOverlapDetector, port, port_padding, "mode_detector",
            wavelengths_um=[w.get_wavelength() * 1e6 for w in wave_characters],
            direction=direction,
            level=level, port_offset=port_offset, name=name, grid=grid,
            wave_characters=tuple(wave_characters), mode_index=mode_index, **fields, **kwargs,
        )

    def create_poynting_flux_detector_at(
        self,
        port: str,
        port_padding: Sequence[float],
        reduce_volume: bool = True,
        keep_all_components: bool = False,
        direction: PortDirection = "out",
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        wavelength_um: float | None = None,
        grid: Any = None,
        **kwargs: Any,
    ) -> tuple[fdtdx.PoyntingFluxDetector, list[PositionConstraint]]:
        """Poynting flux (power vs time) detector on a port.

        Args:
            port: port name; the plane is centered on it and normal to it, or
                ``"xy"``, ``"xz"``, ``"yz"`` (:data:`PLANE_PORTS`) for a
                full-volume cross-section.
            port_padding: (lateral, vertical): the plane extends this far past the
                port's width and the level's height on each side, in units of
                ``wavelength_um``.
            reduce_volume: sum the flux over the plane.
            keep_all_components: keep all three flux components.
            direction: "out" counts power leaving the component as positive,
                "in" power entering it.
            level: level whose height and middle set the plane's z, or
                :data:`Z_CENTER_LEVEL` (``"z_center"``) for the stack's vertical middle.
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the plane out, e.g. into a port extension.
            wavelength_um: the wavelength (um) whose multiple the padding is; required
                unless the padding is zero.
            name: object name (default ``<port>_flux_detector``).
            grid: optional ``SimulationConfig`` or ``RectilinearGrid`` spanning this
                device's box to snap the plane's extents to non-uniform cell edges.
            **kwargs: other PoyntingFluxDetector fields (``dtype`` defaults to the
                device's precision: float32 or float64).

        Returns:
            The detector and the constraints placing it relative to this device.
        """
        _, axis, *_ = self._port_frame(port, level, port_offset)
        return self._at_port(
            fdtdx.PoyntingFluxDetector, port, port_padding, "flux_detector",
            wavelengths_um=None if wavelength_um is None else [wavelength_um],
            direction=direction,
            level=level, port_offset=port_offset, name=name, grid=grid, reduce_volume=reduce_volume,
            keep_all_components=keep_all_components, fixed_propagation_axis=axis, **kwargs,
        )

    def create_phasor_detector_at(
        self,
        port: str,
        port_padding: Sequence[float],
        wave_characters: Sequence[fdtdx.WaveCharacter],
        components: Sequence[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] | None = None,
        reduce_volume: bool = False,
        scaling_mode: Literal["continuous", "pulse"] | None = None,
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        grid: Any = None,
        **kwargs: Any,
    ) -> tuple[fdtdx.PhasorDetector, list[PositionConstraint]]:
        """Frequency-domain field (phasor) detector on a port plane.

        Args:
            port: port name; the plane is centered on it and normal to it, or
                ``"xy"``, ``"xz"``, ``"yz"`` (:data:`PLANE_PORTS`) for a
                full-volume cross-section.
            port_padding: (lateral, vertical): the plane extends this far past the
                port's width and the level's height on each side, in units of the
                mean wavelength of this object's wave characters.
            wave_characters: wavelengths / frequencies to record.
            components: field components (default: fdtdx's, all six).
            reduce_volume: sum over the plane.
            scaling_mode: "continuous" or "pulse" normalization (default:
                fdtdx's, "continuous").
            level: level whose height and middle set the plane's z, or
                :data:`Z_CENTER_LEVEL` (``"z_center"``) for the stack's vertical middle.
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the plane out, e.g. into a port extension.
            name: object name (default ``<port>_phasor_detector``).
            grid: optional ``SimulationConfig`` or ``RectilinearGrid`` spanning this
                device's box to snap the plane's extents to non-uniform cell edges.
            **kwargs: other PhasorDetector fields (``dtype`` defaults to the
                device's precision: complex64 or complex128).

        Returns:
            The detector and the constraints placing it relative to this device.
        """
        fields = _given(components=None if components is None else tuple(components), scaling_mode=scaling_mode)
        return self._at_port(
            fdtdx.PhasorDetector, port, port_padding, "phasor_detector",
            wavelengths_um=[w.get_wavelength() * 1e6 for w in wave_characters],
            direction=None, level=level,
            port_offset=port_offset, name=name, grid=grid, wave_characters=tuple(wave_characters),
            reduce_volume=reduce_volume, **fields, **kwargs,
        )

    def create_field_detector_at(
        self,
        port: str,
        port_padding: Sequence[float],
        components: Sequence[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] | None = None,
        reduce_volume: bool = False,
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        wavelength_um: float | None = None,
        grid: Any = None,
        **kwargs: Any,
    ) -> tuple[fdtdx.FieldDetector, list[PositionConstraint]]:
        """Time-domain field detector on a port plane.

        Args:
            port: port name; the plane is centered on it and normal to it, or
                ``"xy"``, ``"xz"``, ``"yz"`` (:data:`PLANE_PORTS`) for a
                full-volume cross-section.
            port_padding: (lateral, vertical): the plane extends this far past the
                port's width and the level's height on each side, in units of
                ``wavelength_um``.
            components: field components (default: fdtdx's, all six).
            reduce_volume: sum over the plane.
            level: level whose height and middle set the plane's z, or
                :data:`Z_CENTER_LEVEL` (``"z_center"``) for the stack's vertical middle.
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the plane out, e.g. into a port extension.
            wavelength_um: the wavelength (um) whose multiple the padding is; required
                unless the padding is zero.
            name: object name (default ``<port>_field_detector``).
            grid: optional ``SimulationConfig`` or ``RectilinearGrid`` spanning this
                device's box to snap the plane's extents to non-uniform cell edges.
            **kwargs: other FieldDetector fields (``dtype`` defaults to the
                device's precision: float32 or float64).

        Returns:
            The detector and the constraints placing it relative to this device.
        """
        fields = _given(components=None if components is None else tuple(components))
        return self._at_port(
            fdtdx.FieldDetector, port, port_padding, "field_detector",
            wavelengths_um=None if wavelength_um is None else [wavelength_um],
            direction=None, level=level,
            port_offset=port_offset, name=name, grid=grid, reduce_volume=reduce_volume, **fields, **kwargs,
        )

    def create_energy_detector_at(
        self,
        port: str,
        port_padding: Sequence[float],
        reduce_volume: bool = False,
        aggregate: str | None = None,
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        wavelength_um: float | None = None,
        grid: Any = None,
        **kwargs: Any,
    ) -> tuple[fdtdx.EnergyDetector, list[PositionConstraint]]:
        """Electromagnetic energy density detector on a port plane.

        Args:
            port: port name; the plane is centered on it and normal to it, or
                ``"xy"``, ``"xz"``, ``"yz"`` (:data:`PLANE_PORTS`) for a
                full-volume cross-section.
            port_padding: (lateral, vertical): the plane extends this far past the
                port's width and the level's height on each side, in units of
                ``wavelength_um``.
            reduce_volume: sum over the plane.
            aggregate: aggregation over time, e.g. "mean" (default: none).
            level: level whose height and middle set the plane's z, or
                :data:`Z_CENTER_LEVEL` (``"z_center"``) for the stack's vertical middle.
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the plane out, e.g. into a port extension.
            wavelength_um: the wavelength (um) whose multiple the padding is; required
                unless the padding is zero.
            name: object name (default ``<port>_energy_detector``).
            grid: optional ``SimulationConfig`` or ``RectilinearGrid`` spanning this
                device's box to snap the plane's extents to non-uniform cell edges.
            **kwargs: other EnergyDetector fields (``dtype`` defaults to the
                device's precision: float32 or float64).

        Returns:
            The detector and the constraints placing it relative to this device.
        """
        return self._at_port(
            fdtdx.EnergyDetector, port, port_padding, "energy_detector",
            wavelengths_um=None if wavelength_um is None else [wavelength_um],
            direction=None, level=level,
            port_offset=port_offset, name=name, grid=grid, reduce_volume=reduce_volume, aggregate=aggregate, **kwargs,
        )

    def create_diffractive_detector_at(
        self,
        port: str,
        port_padding: Sequence[float],
        frequencies: Sequence[float],
        orders: Sequence[tuple[int, int]] = ((0, 0),),
        direction: PortDirection = "out",
        level: str | None = None,
        port_offset: float = 0.0,
        name: str | None = None,
        grid: Any = None,
        **kwargs: Any,
    ) -> tuple[DiffractiveDetector, list[PositionConstraint]]:
        """Diffraction-order detector on a port plane (needs periodic boundaries).

        Args:
            port: port name; the plane is centered on it and normal to it, or
                ``"xy"``, ``"xz"``, ``"yz"`` (:data:`PLANE_PORTS`) for a
                full-volume cross-section.
            port_padding: (lateral, vertical): the plane extends this far past the
                port's width and the level's height on each side, in units of the
                mean wavelength of ``frequencies``.
            frequencies: frequencies (Hz) to analyze.
            orders: diffraction orders (m, n) to record.
            direction: "out" for orders leaving the component, "in" entering it.
            level: level whose height and middle set the plane's z, or
                :data:`Z_CENTER_LEVEL` (``"z_center"``) for the stack's vertical middle.
            port_offset: shift (um) along the port's injection direction (into the
                component); negative moves the plane out, e.g. into a port extension.
            name: object name (default ``<port>_diffractive_detector``).
            grid: optional ``SimulationConfig`` or ``RectilinearGrid`` spanning this
                device's box to snap the plane's extents to non-uniform cell edges.
            **kwargs: other DiffractiveDetector fields (``dtype`` defaults to the
                device's precision: complex64 or complex128).

        Returns:
            The detector and the constraints placing it relative to this device.
        """
        return self._at_port(
            DiffractiveDetector, port, port_padding, "diffractive_detector",
            wavelengths_um=[fdtdx.constants.c / f * 1e6 for f in frequencies],
            direction=direction,
            level=level, port_offset=port_offset, name=name, grid=grid, frequencies=tuple(frequencies),
            orders=tuple(orders), **kwargs,
        )


def _check_materials(materials: dict[str, fdtdx.Material], averaging: str, name: str) -> None:
    """Checks the permittivity and magnetic properties of the materials a StackDevice paints.

    Every permittivity tensor must be finite, symmetric and positive definite
    (:func:`bifrost.subpixel.check_permittivity`); off-diagonal entries are written by
    ``averaging="subpixel_full"`` only, so with another averaging they are an error rather than dropped.
    Magnetic materials and magnetic conductivities are not supported by StackDevice.

    Args:
        materials: the device's materials (the anisotropy placeholder is skipped).
        averaging: the device's averaging.
        name: the device's name, for the message.

    Raises:
        ValueError: an invalid tensor, off-diagonal entries without "subpixel_full",
            or a magnetic / magnetically conductive material.
    """
    for key, material in materials.items():
        if key == _ANISOTROPY_SENTINEL:
            continue
        if getattr(material, "is_magnetic", False) or getattr(material, "is_magnetically_conductive", False):
            raise ValueError(
                f"Material {key!r} of StackDevice {name!r} has magnetic properties or magnetic conductivity, "
                "which StackDevice does not mix"
            )
        eps = material_tensor(material)
        check_permittivity(eps, key)
        if averaging != "subpixel_full" and not is_diagonal(eps):
            raise ValueError(
                f"Material {key!r} of StackDevice {name!r} has off-diagonal permittivity entries, which "
                f"averaging={averaging!r} cannot write (it takes each E component's diagonal entry); use "
                "averaging='subpixel_full', which writes the complete tensor"
            )


def _drop_empty_pole_slots(materials: dict[str, fdtdx.Material]) -> dict[str, fdtdx.Material]:
    """Removes the pole slots that no material uses (zero strength everywhere).

    Aligned materials keep a slot for every pole of every material they were
    aligned with; after levels are dropped, slots of their materials would cost
    memory and time in every voxel for nothing.

    Args:
        materials: materials after :func:`align_material_poles`.

    Returns:
        The materials with the used slots only (no dispersion if none is used).
    """
    poles = {k: m.dispersion.poles for k, m in materials.items() if m.dispersion is not None}
    if not poles:
        return materials
    num = len(next(iter(poles.values())))
    live = [s for s in range(num) if any(p[s].coupling_sq != 0 for p in poles.values())]
    if len(live) == num:
        return materials
    out = dict(materials)
    for k in poles:
        m = materials[k]
        out[k] = fdtdx.Material(
            permittivity=m.permittivity,
            permeability=m.permeability,
            electric_conductivity=m.electric_conductivity,
            magnetic_conductivity=m.magnetic_conductivity,
            dispersion=DispersionModel(poles=tuple(poles[k][s] for s in live)) if live else None,
        )
    return out


def pole_slots(materials: dict[str, fdtdx.Material]) -> tuple[Pole, ...]:
    """The shared pole slots of aligned materials.

    Args:
        materials: materials after :func:`align_material_poles`.

    Returns:
        The poles of the slots (empty if no material is dispersive).

    Raises:
        ValueError: if the materials' slots differ (not aligned).
    """
    slots: tuple[Pole, ...] | None = None
    for name, m in materials.items():
        if m.dispersion is None or m.dispersion.num_poles == 0:
            continue
        keys = [(p.omega_0, p.gamma) for p in m.dispersion.poles]
        if slots is None:
            slots = tuple(m.dispersion.poles)
        elif keys != [(p.omega_0, p.gamma) for p in slots]:
            raise ValueError(
                f"Material {name!r} has pole slots {keys} that differ from the other "
                "materials'; align them with align_material_poles"
            )
    return slots or ()


def layer_stack_wafer_layer() -> Any:
    """The WAFER layer of the active PDK.

    Returns:
        The layer, or None if the PDK has none.
    """
    from gdsfactory.pdk import get_layer

    try:
        return get_layer("WAFER")
    except Exception:
        return None


_PML_COEFFICIENTS = ("pml_a_E", "pml_b_E", "inv_kappa_E", "pml_a_H", "pml_b_H", "inv_kappa_H")


def _cast_pml_coefficients(objects: Any, dtype: Any) -> Any:
    """Re-casts the PMLs' CPML coefficients to ``dtype`` (fdtdx 0.6.3).

    When JAX's 64-bit mode is on, fdtdx's
    ``place_on_grid`` then derives ``pml_a/b`` and ``inv_kappa`` as float64
    whatever ``config.dtype`` is. In a float32 simulation the CPML update
    ``b * psi + a * d`` would promote the float32 psi fields, and
    ``eqxi.while_loop`` rejects the loop body whose output structure changed.
    This is why runs used to be wrapped in ``jax.enable_x64(False)``; with the
    coefficients in the simulation's dtype, the 64-bit mode can stay on for
    either dtype.

    Args:
        objects: fdtdx ObjectContainer.
        dtype: the simulation's dtype (the device's).

    Returns:
        ``objects``, or a container with the re-cast PMLs if any coefficient had another dtype.
    """
    out = []
    changed = False
    for obj in objects.object_list:
        if isinstance(obj, fdtdx.PerfectlyMatchedLayer):
            for name in _PML_COEFFICIENTS:
                value = getattr(obj, name, None)
                if value is not None and not _same_dtype(value.dtype, dtype):
                    obj = obj.aset(name, value.astype(dtype))
                    changed = True
        out.append(obj)
    return ObjectContainer(object_list=out, volume_idx=objects.volume_idx) if changed else objects


def stack_devices(objects: Any) -> list[StackDevice]:
    """All StackDevices of an fdtdx ObjectContainer."""
    return [o for o in objects.objects if isinstance(o, StackDevice)]


def init_stack_params(objects: Any) -> dict[str, dict[str, Array]]:
    """Initial parameters of all StackDevices.

    Args:
        objects: fdtdx ObjectContainer (from ``fdtdx.place_objects``).

    Returns:
        ``{device name: device params}``; merge it into fdtdx's params.
    """
    return {o.name: o.init_params() for o in stack_devices(objects)}


@partial(jax.jit, static_argnames=("apply_roughness", "capacity", "shift"))
def _interface_status_core(
    device: StackDevice,
    params: dict[str, Array],
    noise_key: Array | None,
    apply_roughness: bool,
    capacity: int,
    shift: tuple[float, float, float],
) -> tuple[Array, Array, Array, Array, Array]:
    """JIT-compiled interface status computation for one Yee shift."""
    g = device._interfaces(params, noise_key, apply_roughness, shift)
    _, status = device._complete_tensor(g, capacity)
    return (
        jnp.asarray(status["junctions"], dtype=jnp.int32),
        jnp.asarray(status["junction_overflow"], dtype=jnp.int32),
        jnp.asarray(g.corner_dropped, dtype=jnp.int32),
        jnp.asarray(g.corner_uncertified, dtype=jnp.int32),
        jnp.asarray(status["closure_residual"], dtype=device.dtype),
    )


@timed("apply_stack")
def apply_stack(
    arrays: Any,
    objects: Any,
    params: dict[str, Any],
    key: Array,
    noise_key: Array | None = None,
    apply_roughness: bool = True,
    **transform_kwargs: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """``fdtdx.apply_params`` that also applies every :class:`StackDevice`.

    Stacks are written first, then ``fdtdx.apply_params`` applies the regular
    devices. ``transform_kwargs`` are forwarded to ``fdtdx.apply_params``.
    Last, the objects whose ``apply`` reads the materials (sources and mode
    detectors: their modes, impedances and time offsets) and that overlap a
    stack are applied again with the final permittivity and pole coefficients.
    fdtdx applies objects that overlap none of its own ``Device`` objects once,
    in ``place_objects``, on the initial painting (each voxel's dominant
    material, see :meth:`StackDevice.get_material_mapping`), and
    ``fdtdx.apply_params`` (0.6.3) re-applies only those overlapping a
    ``Device``; without this step a mode source would launch the mode of that
    painting, not of the stack, and scatter where it is injected. An object
    is skipped while its materials match a previous solve in a bounded LRU
    cache keyed by configuration and compact content digests (under ``jax.grad``
    by the primal values), so repeated calls with unchanged parameters solve no
    modes. ``info["stack_reapplied"]`` and ``info["stack_reused"]`` name the
    objects.

    In dispersive simulations with mode or plane-wave sources, fdtdx 0.6.2
    builds the sources' dispersive filter with numpy, so this call (like
    ``fdtdx.apply_params``) cannot run inside ``jax.jit``; ``jax.grad`` works.

    The returned objects also have their PML coefficients in the stacks'
    ``dtype``: when 64-bit JAX mode is active, fdtdx 0.6.3 derives them as
    float64 and float32 simulations would otherwise fail in ``fdtdx.run_fdtd``.
    No ``jax.enable_x64(False)`` is needed around the run, for either dtype.

    Args:
        arrays: fdtdx ArrayContainer.
        objects: fdtdx ObjectContainer.
        params: fdtdx params including the stacks' (:func:`init_stack_params`).
        key: PRNG key for ``fdtdx.apply_params``.
        noise_key: PRNG key for the roughness (default: each device's seed).
        apply_roughness: if False, the nominal geometry.
        **transform_kwargs: forwarded to ``fdtdx.apply_params``.

    Returns:
        (arrays, objects, info) as ``fdtdx.apply_params`` returns them.
    """
    stacks = stack_devices(objects)
    if stacks:
        expected_dtype = stacks[0].dtype
        for o in stacks[1:]:
            if not _same_dtype(o.dtype, expected_dtype):
                raise ValueError(
                    f"apply_stack: StackDevices must share one dtype, got "
                    f"{stacks[0].name!r} ({expected_dtype}) and {o.name!r} ({o.dtype})"
                )
        if not _same_dtype(arrays.inv_permittivities.dtype, expected_dtype):
            raise ValueError(
                f"apply_stack: arrays.inv_permittivities has dtype {arrays.inv_permittivities.dtype}, "
                f"expected StackDevice dtype {expected_dtype}"
            )
        for i, a in enumerate(stacks):
            for b in stacks[i + 1 :]:
                if _boxes_overlap(a, b):
                    raise ValueError(
                        f"apply_stack: StackDevices {a.name!r} and {b.name!r} have overlapping grid boxes "
                        f"{a.grid_slice_tuple} and {b.grid_slice_tuple}"
                    )
    statuses = {}
    for o in stacks:
        arrays, status = o.write_with_status(arrays, params.get(o.name), noise_key, apply_roughness)
        if status:
            statuses[o.name] = status
    lap("write_stacks", arrays)
    skipped_devices = {d.name: d for d in objects.devices if d.name not in params}
    if skipped_devices:
        device_params = {d.name: params[d.name] for d in objects.devices if d.name in params}
        volume_obj = objects.volume
        filtered_list = [o for o in objects.object_list if o.name not in skipped_devices]
        apply_objects = ObjectContainer(object_list=filtered_list, volume_idx=filtered_list.index(volume_obj))
        arrays, applied_objects, info = fdtdx.apply_params(
            arrays, apply_objects, device_params, key, **transform_kwargs
        )
        by_name = {o.name: o for o in applied_objects.object_list}
        objects = ObjectContainer(
            object_list=[skipped_devices.get(o.name, by_name[o.name]) for o in objects.object_list],
            volume_idx=objects.volume_idx,
        )
    else:
        device_params = {d.name: params[d.name] for d in objects.devices}
        arrays, objects, info = fdtdx.apply_params(arrays, objects, device_params, key, **transform_kwargs)
    lap("apply_params", arrays)
    objects, reapplied, reused = _reapply_stack_objects(arrays, objects, stacks, key)
    lap("reapply_objects", objects)
    if stacks:
        objects = _cast_pml_coefficients(objects, stacks[0].dtype)
    return arrays, objects, {**info, "stack_reapplied": reapplied, "stack_reused": reused, "stack_status": statuses}


def _material_slices(arrays: Any, obj: Any) -> tuple[Any, ...]:
    """The material arrays an object's ``apply`` reads, sliced to its grid region."""
    gs = obj.grid_slice
    mu = arrays.inv_permeabilities
    out = [arrays.inv_permittivities[:, *gs], mu[:, *gs] if isinstance(mu, jax.Array) and mu.ndim > 0 else mu]
    for name in ("dispersive_c1", "dispersive_c2", "dispersive_c3"):
        a = getattr(arrays, name, None)
        out.append(None if a is None else a[:, :, *gs])
    sigma = getattr(arrays, "electric_conductivity", None)
    out.append(None if sigma is None else sigma[:, *gs])
    return tuple(out)


def _hashable_config_value(val: Any, seen: set[int] | None = None) -> Any:
    """Converts an object field or configuration value into a hashable key."""
    if val is None or isinstance(val, (bool, int, float, str, bytes, np.dtype, jnp.dtype, type, Enum)):
        return val
    val = _primal(val)
    if isinstance(val, jax.core.Tracer):
        return ("tracer", id(val))
    if isinstance(val, (np.ndarray, jax.Array)):
        arr = np.ascontiguousarray(np.asarray(val))
        return ("array", arr.dtype.str, arr.shape, hashlib.blake2b(arr, digest_size=16).digest())
    if isinstance(val, np.generic):
        return (type(val).__name__, val.item())
    if seen is None:
        seen = set()
    obj_id = id(val)
    if obj_id in seen:
        return ("cycle", obj_id)
    if isinstance(val, (tuple, list)):
        seen.add(obj_id)
        try:
            return (type(val).__name__, tuple(_hashable_config_value(v, seen) for v in val))
        finally:
            seen.remove(obj_id)
    if isinstance(val, dict):
        seen.add(obj_id)
        try:
            return (
                "dict",
                tuple((_hashable_config_value(k, seen), _hashable_config_value(v, seen)) for k, v in val.items()),
            )
        finally:
            seen.remove(obj_id)
    if hasattr(val, "get_class_fields"):
        seen.add(obj_id)
        try:
            return (
                type(val),
                tuple((f.name, _hashable_config_value(getattr(val, f.name), seen)) for f in val.get_class_fields()),
            )
        finally:
            seen.remove(obj_id)
    if dataclasses.is_dataclass(val) and not isinstance(val, type):
        seen.add(obj_id)
        try:
            return (
                type(val),
                tuple((f.name, _hashable_config_value(getattr(val, f.name), seen)) for f in dataclasses.fields(val)),
            )
        finally:
            seen.remove(obj_id)
    try:
        hash(val)
        return val
    except TypeError:
        if hasattr(val, "__dict__"):
            seen.add(obj_id)
            try:
                return (type(val), _hashable_config_value(vars(val), seen))
            finally:
                seen.remove(obj_id)
        return ("id", obj_id)


def _mode_solver_config_key(obj: Any) -> Any | None:
    """Compact hashable key of all configuration fields affecting ``obj.apply()``, or None if stochastic."""
    if (
        getattr(obj, "max_angle_random_offset", 0.0) != 0.0
        or getattr(obj, "max_vertical_offset", 0.0) != 0.0
        or getattr(obj, "max_horizontal_offset", 0.0) != 0.0
    ):
        return None
    fields = tuple((f.name, _hashable_config_value(f.value)) for f in obj.get_public_fields())
    return (
        type(obj),
        obj.grid_slice_tuple,
        _hashable_config_value(getattr(obj, "_config", None)),
        _hashable_config_value(getattr(obj, "_cached_face_area_weights", None)),
        fields,
    )


def _slice_cache_key(slices: tuple[Any, ...]) -> tuple[Any, ...] | None:
    """Tuple of ``(dtype, shape, blake2b_16B)`` digests for concrete slices, or None if traced."""
    out: list[Any] = []
    for item in slices:
        if item is None:
            out.append(None)
            continue
        item = _primal(item)
        if isinstance(item, jax.core.Tracer):
            return None
        arr = np.ascontiguousarray(np.asarray(item))
        out.append((arr.dtype.str, arr.shape, hashlib.blake2b(arr, digest_size=16).digest()))
    return tuple(out)


_PENDING_REAPPLY: contextvars.ContextVar[Callable[[], Any] | None] = contextvars.ContextVar(
    "_PENDING_REAPPLY", default=None
)


@lru_cache(maxsize=32)
def _reapply_cache(mode_solver_config_key: Any, slice_key: tuple[Any, ...]) -> Any:
    """Bounded LRU cache storing concrete ``obj.apply()`` outputs keyed by config and slice digests."""
    fn = _PENDING_REAPPLY.get()
    assert fn is not None
    return fn()


def _boxes_overlap(a: Any, b: Any) -> bool:
    """Whether two placed objects' grid regions share a voxel (half-open slices on all three axes).

    Not ``SimulationObject.check_overlap``: that tests whether an end of one
    region lies in the other on some axis, so a large box containing a plane
    (a stack containing a port's source) counts as no overlap.
    """
    return all(
        sa < ob and oa < sb
        for (sa, sb), (oa, ob) in zip(a.grid_slice_tuple, b.grid_slice_tuple, strict=True)
    )


def _reapply_stack_objects(
    arrays: Any, objects: Any, stacks: Sequence[StackDevice], key: Array
) -> tuple[Any, list[str], list[str]]:
    """Applies the material-reading objects that overlap a stack again, with the final arrays (see :func:`apply_stack`).

    Returns:
        The objects, and the names of those applied again and of those reused
        from the bounded content-digest cache.
    """
    if not stacks:
        return objects, [], []
    devices = objects.devices
    reapplied, reused, new_objects = [], [], []
    for i, obj in enumerate(objects.object_list):
        if (
            isinstance(obj, StackDevice)
            or type(obj).apply is SimulationObject.apply  # ignores the materials
            or not any(_boxes_overlap(s, obj) for s in stacks)
            # fdtdx.apply_params applied it already (its own overlap test, so exactly those objects)
            or any(d.check_overlap(obj) for d in devices)
        ):
            new_objects.append(obj)
            continue

        def frozen(name: str) -> Any:
            a = getattr(arrays, name, None)
            return a if a is None or not isinstance(a, jax.Array) else jax.lax.stop_gradient(a)

        slices = _material_slices(arrays, obj)
        slice_key = _slice_cache_key(slices)
        config_key = None if slice_key is None else _mode_solver_config_key(obj)
        ran = False

        def solve(cur_obj: Any = obj, idx: int = i) -> Any:
            nonlocal ran
            ran = True
            with timed_step("reapply_object") as step:
                return step.done(
                    cur_obj.apply(
                        key=jax.random.fold_in(key, idx),
                        inv_permittivities=frozen("inv_permittivities"),
                        inv_permeabilities=frozen("inv_permeabilities"),
                        dispersive_c1=frozen("dispersive_c1"),
                        dispersive_c2=frozen("dispersive_c2"),
                        dispersive_c3=frozen("dispersive_c3"),
                        electric_conductivity=frozen("electric_conductivity"),
                    )
                )

        if config_key is not None and slice_key is not None:
            token = _PENDING_REAPPLY.set(solve)
            try:
                applied = _reapply_cache(config_key, slice_key)
            finally:
                _PENDING_REAPPLY.reset(token)
            if ran:
                reapplied.append(applied.name)
            else:
                reused.append(applied.name)
        else:
            applied = solve()
            reapplied.append(applied.name)
        new_objects.append(applied)
    return type(objects)(object_list=new_objects, volume_idx=objects.volume_idx), reapplied, reused
