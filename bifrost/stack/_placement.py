"""Port/plane source & detector placement helpers and area-normalized wrappers."""

import math
import fdtdx
import jax.numpy as jnp
import numpy as np
from collections.abc import Sequence
from typing import Any, Literal, Self
from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import autoinit
from ._layers import Array, PortDirection


#: The simulation-volume planes a source or detector can sit on, as virtual
#: ports: the plane through the box center parallel to the named faces, mapped
#: to its normal axis (0 = x, 1 = y, 2 = z). They are not device ports.
PLANE_PORTS: dict[str, int] = {"yz": 0, "xz": 1, "xy": 2}


def detector_plane_own_positions(axis: int) -> tuple[float, float, float]:
    """The `own_positions` that put a detector plane's samples on its target.

    fdtdx records every detector field at `(i·Δx, j·Δy, (k+½)·Δz)` (see
    `fdtdx.core.physics.curl.interpolate_fields`): x and y samples lie on a
    cell's lower edge, z samples at its centre. A one-cell-thick plane is placed
    by anchoring a point of the object on the target, and fdtdx picks the cell
    whose anchor is nearest, so the anchor must be the sampled one: `-1` (the
    lower edge) along a normal that is x or y, `0` (the centre) along z. The
    in-plane axes keep the centre, which centres the many-cell plane on the
    target.

    Args:
        axis: the plane's normal axis (0 = x, 1 = y, 2 = z).

    Returns:
        `own_positions` for `place_relative_to`, in fdtdx's `-1 .. 1`
        convention (`-1` lower side, `0` centre).
    """
    positions = [0.0, 0.0, 0.0]
    if axis in (0, 1):
        positions[axis] = -1.0
    return positions[0], positions[1], positions[2]


#: The `level` value naming the simulation volume's vertical middle, offered
#: alongside level names so a source or detector can be pinned to the volume's
#: mid-height instead of a level's span. `None` has the same meaning for a
#: plane port, and selects a port's own level union otherwise.
Z_CENTER_LEVEL: str = "z_center"


#: The two in-plane axes of each normal axis (0 = x, 1 = y, 2 = z), in the
#: order a plane's ``port_padding`` and its two extents use: (u, v).
_IN_PLANE: tuple[tuple[int, int], ...] = ((1, 2), (0, 2), (0, 1))


def plane_extents(
    width_um: float,
    height_um: float,
    port_padding: Sequence[float],
    wavelengths_um: Sequence[float] | None,
) -> tuple[float, float]:
    """The in-plane size of a source or detector plane on a port, in um.

    The plane extends `port_padding` mean wavelengths past the port's width and
    the level's height on each side, so each extent is the port's plus twice the
    padding.

    Args:
        width_um: the port's width.
        height_um: the level's height.
        port_padding: `(lateral, vertical)`, each finite and possibly negative (a
            negative padding shrinks the plane inside the port), in units of the
            mean of `wavelengths_um`.
        wavelengths_um: the vacuum wavelengths of the object's wave characters.
            May be empty or None only when the padding is zero.

    Returns:
        The plane's (width, height) in um.

    Raises:
        ValueError: if the padding is not two finite numbers, a wavelength is
            missing for a non-zero padding, or a negative padding leaves a
            plane no larger than zero.
    """
    if len(port_padding) != 2 or not all(math.isfinite(v) for v in port_padding):
        raise ValueError(f"port_padding must be two finite numbers, got {port_padding}")
    if not wavelengths_um:
        if any(v != 0 for v in port_padding):
            raise ValueError(
                "port_padding is in units of the mean wavelength: pass wavelength_um (or a wave character)"
            )
        return float(width_um), float(height_um)
    mean = float(sum(wavelengths_um)) / len(wavelengths_um)
    size = (width_um + 2 * port_padding[0] * mean, height_um + 2 * port_padding[1] * mean)
    if min(size) <= 0:
        raise ValueError(
            f"port_padding {tuple(port_padding)} leaves a plane of {size[0]:.4g} x {size[1]:.4g} um: "
            "a negative padding must stay smaller than half the port's width and the level's height"
        )
    return size


_SIMULATION_DTYPES = {np.dtype(np.float32): jnp.float32, np.dtype(np.float64): jnp.float64}


def _simulation_dtype(dtype: Any) -> Any:
    """``jnp.float32`` or ``jnp.float64`` for any spelling of those two dtypes.

    Args:
        dtype: a jnp / np scalar type, ``np.dtype`` or dtype name (``"float64"``).

    Returns:
        The canonical ``jnp.float32`` or ``jnp.float64``.

    Raises:
        ValueError: for anything else (the solver runs in float32 or float64).
    """
    try:
        resolved = None if dtype is None else _SIMULATION_DTYPES.get(np.dtype(dtype))
    except TypeError:
        resolved = None
    if resolved is None:
        raise ValueError(f"dtype must be jnp.float32 or jnp.float64, got {dtype!r}")
    return resolved


def _same_dtype(a: Any, b: Any) -> bool:
    """Whether two dtype spellings (jnp / np type, ``np.dtype``) are the same dtype."""
    return np.dtype(a) == np.dtype(b)


def _complex_dtype(dtype: Any) -> Any:
    """The complex dtype that goes with a simulation dtype (phasors and mode overlaps): complex64 or complex128."""
    return jnp.complex128 if _same_dtype(dtype, np.float64) else jnp.complex64


def _plane_face_areas(obj: Any) -> Array:
    """Per-cell transverse areas (m^2) of a placed plane source or detector."""
    calc_dtype = obj._config.dtype
    grid = obj._config.resolved_grid
    if grid is not None:
        return jnp.asarray(
            grid.face_area(axis=obj.propagation_axis, slice_tuple=obj.grid_slice_tuple),
            dtype=calc_dtype,
        )
    spacing = jnp.asarray(obj._config.uniform_spacing(), dtype=calc_dtype)
    return jnp.full(obj.grid_shape, spacing * spacing, dtype=calc_dtype)


def _reference_face_area(config: SimulationConfig, calc_dtype: Any) -> Array:
    """Simulation-wide reference cell area ``min_spacing**2`` (m^2)."""
    grid = config.resolved_grid
    h = grid.min_spacing if grid is not None else config.uniform_spacing()
    h_arr = jnp.asarray(h, dtype=calc_dtype)
    return h_arr * h_arr


@autoinit
class ModePlaneSource(fdtdx.ModePlaneSource):
    """Waveguide mode source with port-independent power normalization on graded grids.

    On a non-uniform `RectilinearGrid`, `fdtdx.core.physics.modes.compute_mode`
    normalizes its fields with `area_weights / mean(area_weights)`, leaving the
    launched Poynting flux proportional to each source plane's own `mean(face_area)`.
    Rescaling `_E` and `_H` by `sqrt(mean(face_area) / min_spacing**2)` keeps field
    magnitudes `O(1)` in `float32`, behaves identically to fdtdx on uniform grids,
    and launches the same physical power `min_spacing**2` regardless of the source
    plane's padding, width, or orientation.
    """

    def apply(
        self: Self,
        key: Array,
        inv_permittivities: Array,
        inv_permeabilities: Array | float,
        dispersive_c1: Array | None = None,
        dispersive_c2: Array | None = None,
        dispersive_c3: Array | None = None,
        electric_conductivity: Array | None = None,
    ) -> Self:
        applied = super().apply(
            key=key,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
            dispersive_c1=dispersive_c1,
            dispersive_c2=dispersive_c2,
            dispersive_c3=dispersive_c3,
            electric_conductivity=electric_conductivity,
        )
        if not applied._config.has_nonuniform_grid:
            return applied
        areas = _plane_face_areas(applied)
        ref_area = _reference_face_area(applied._config, areas.dtype)
        scale = jnp.sqrt(jnp.mean(areas) / ref_area).astype(applied._config.dtype)
        applied = applied.aset("_E", applied._E / scale)
        return applied.aset("_H", applied._H / scale)


@autoinit
class ModeOverlapDetector(fdtdx.ModeOverlapDetector):
    """Waveguide mode-overlap detector with port-independent power normalization on graded grids.

    On a non-uniform ``RectilinearGrid``, ``fdtdx.core.physics.modes.compute_mode`` scales
    the reference mode by ``sqrt(mean(face_area))``, while
    ``fdtdx.BaseModeOverlapDetector.place_on_grid`` divides the overlap area weights by
    ``mean(face_area)`` a second time, biasing ``|alpha|^2`` by ``1 / mean(face_area)`` and
    two-port ratios ``|alpha_out / alpha_in|^2`` by ``mean(A_in) / mean(A_out)`` whenever two
    port planes span different cell-area distributions (for instance on 90-degree bends with
    non-identical x/y grading, or ports with different widths or paddings). Weighting by
    ``face_area / sqrt(mean(face_area) * min_spacing**2)`` cancels each detector plane's
    ``mean(face_area)`` against the simulation-wide ``min_spacing**2`` reference while
    leaving uniform grids unchanged bit for bit.
    """

    def place_on_grid(
        self: Self,
        grid_slice_tuple: Any,
        config: SimulationConfig,
        key: Array,
    ) -> Self:
        placed = super().place_on_grid(
            grid_slice_tuple=grid_slice_tuple,
            config=config,
            key=key,
        )
        if not placed._config.has_nonuniform_grid:
            return placed
        real_dtype = placed._config.dtype
        areas = _plane_face_areas(placed)
        ref_area = _reference_face_area(placed._config, areas.dtype)
        weights = (areas / jnp.sqrt(jnp.mean(areas) * ref_area)).astype(real_dtype)
        return placed.aset("_cached_face_area_weights", weights, create_new_ok=True)


def _axis_direction(direction: PortDirection, outward: int) -> Literal["+", "-"]:
    """fdtdx direction ("+" / "-") of "in" / "out" at a port.

    Args:
        direction: "in" (into the component) or "out".
        outward: +1 if the port points towards +axis, else -1.

    Returns:
        "+" or "-" along the port's normal axis.

    Raises:
        ValueError: for anything but "in" / "out".
    """
    if direction not in ("in", "out"):
        raise ValueError(f"direction must be 'in' or 'out', got {direction!r}")
    return "+" if (outward > 0) == (direction == "out") else "-"


def _given(**kwargs: Any) -> dict[str, Any]:
    """The keyword arguments that are not None (None keeps fdtdx's default)."""
    return {k: v for k, v in kwargs.items() if v is not None}
