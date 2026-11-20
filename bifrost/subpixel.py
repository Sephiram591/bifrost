"""Material permittivity tensors and their subpixel averages for `bifrost.stack`.

Pure matrix routines, independent of the stack's geometry:

- the material tensor contract: every material's relative permittivity as a
  real, symmetric, positive-definite 3 x 3 matrix in simulation coordinates
  (fdtdx's row-major 9-tuple), checked once (`material_tensors`);
- the planar (laminate) average of anisotropic materials of Kottke,
  Farjadpour and Johnson, Phys. Rev. E 77, 036611 (2008), Section VI
  (`laminate`), written with the projectors onto the interface normal
  and onto its plane, so that it needs no tangent frame;
- the junction closure: the effective tensor of a voxel's own arrangement of
  materials from the periodic cell problem `div(A (e_i + grad phi_i)) = 0`
  on a grid of sub-cells (`periodic_cell_tensor`), and the weight that
  blends it with the laminate where a voxel's interfaces are close to one
  plane (`junction_weight`).

Tensors are `(..., 3, 3)` arrays. Rows and columns are x, y, z of the
simulation; nothing here rotates a material: a crystal whose principal axes are
rotated by R relative to the simulation axes has the tensor `R @ eps @ R.T`.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array

# full float32 precision in contractions (GPUs default to TF32 for them: 10 mantissa bits)
_HIGHEST = jax.lax.Precision.HIGHEST
# sparse_voxel_map compacts the active entries before chunking (True), instead of scanning every spatial
# brick of the grid with its own lax.cond (one device->host predicate sync per brick).
_COMPACT_SPARSE_MAP = True
# Junction closures per sparse_voxel_map call (chunk) and per skippable group in homogenize_full
_JUNCTION_CHUNK = 128
_JUNCTION_GROUP = 16384
# size of an interface direction tensor (its largest entry, its trace) below which it has no direction (a
# full face weighs ~1); its square stays a normal float32, which the derivatives of ratios with it need
_NO_DIRECTION = 1e-15

#: largest |eps - eps^T| accepted, relative to the largest |eps| entry
SYMMETRY_RTOL = 1e-6
#: smallest eigenvalue accepted, relative to the largest one
DEFINITE_RTOL = 1e-6
#: default junction transition bounds r = 1 - lambda_1 / tr(D)
JUNCTION_LOW = 0.002
JUNCTION_HIGH = 0.02
#: sub-cell gap (lambda_1 - lambda_2) / tr(D) below which a sub-cell blends from leading normal to axis laminates
SUBCELL_GAP = (0.02, 0.1)
#: maximum normwise backward residual accepted for a certified periodic cell solve
CLOSURE_RESIDUAL = 1e-4


# ---------------------------------------------------------------- material tensor contract


def material_tensor(material: Any) -> np.ndarray:
    """The relative permittivity of an fdtdx Material as a (3, 3) matrix in simulation coordinates.

    fdtdx stores a scalar, a 3-tuple (the diagonal) and a nested 3 x 3 tuple
    alike as one row-major 9-tuple (xx, xy, xz, yx, yy, yz, zx, zy, zz).

    Args:
        material: an ``fdtdx.Material``.

    Returns:
        (3, 3) float64 array.

    Raises:
        ValueError: if the permittivity is not 9 numbers.
    """
    eps = np.asarray(material.permittivity, dtype=float)
    if eps.shape != (9,):
        raise ValueError(f"Expected fdtdx's 9-component permittivity, got shape {eps.shape}")
    return eps.reshape(3, 3)


def is_diagonal(eps: np.ndarray) -> bool:
    """Whether a (3, 3) tensor has no off-diagonal entry (exactly, as fdtdx decides its array layout)."""
    return not np.any(eps[~np.eye(3, dtype=bool)])


def check_permittivity(eps: np.ndarray, name: str) -> None:
    """Raises unless a permittivity tensor is finite, symmetric and positive definite.

    The tolerances are relative to the tensor's own scale (``SYMMETRY_RTOL``,
    ``DEFINITE_RTOL``). Invalid input is rejected, never repaired: eigenvalues
    are not clipped.

    Args:
        eps: (3, 3) tensor.
        name: the material's name, for the message.

    Raises:
        ValueError: with the reason and how to fix it.
    """
    if not np.all(np.isfinite(eps)):
        raise ValueError(f"Material {name!r} has a non-finite permittivity {eps.tolist()}")
    scale = float(np.abs(eps).max())
    asymmetry = float(np.abs(eps - eps.T).max())
    if not asymmetry <= SYMMETRY_RTOL * scale:
        raise ValueError(
            f"Material {name!r} has a non-symmetric permittivity {eps.tolist()} (largest |eps - eps^T| = "
            f"{asymmetry:.3g}). A StackDevice needs the real, symmetric permittivity of a lossless dielectric "
            "(losses go into electric_conductivity or a DispersionModel); a tensor whose principal axes are "
            "rotated by R is R @ eps @ R.T"
        )
    eigenvalues = np.linalg.eigvalsh(0.5 * (eps + eps.T))
    if not eigenvalues[0] > DEFINITE_RTOL * abs(eigenvalues[-1]):
        raise ValueError(
            f"Material {name!r} has a permittivity that is not positive definite (eigenvalues "
            f"{eigenvalues.tolist()}). Explicit FDTD is unstable for it; a material with Re(eps) < 0 in band "
            "needs eps_inf >= 1 and a DispersionModel. Eigenvalues are not clipped"
        )


def material_tensors(materials: Mapping[str, Any], names: Sequence[str], ignore: Sequence[str] = ()) -> np.ndarray:
    """The (M, 3, 3) permittivity tensors of materials in a given order, each checked.

    Args:
        materials: fdtdx Materials by name.
        names: the order (e.g. fdtdx's ``compute_ordered_names``).
        ignore: names not checked (internal placeholders), returned as the identity.

    Returns:
        (M, 3, 3) float64 array.

    Raises:
        ValueError: see `check_permittivity`.
    """
    out = []
    for name in names:
        if name in ignore:
            out.append(np.eye(3))
            continue
        eps = material_tensor(materials[name])
        check_permittivity(eps, name)
        out.append(0.5 * (eps + eps.T))  # exactly symmetric
    return np.stack(out) if out else np.zeros((0, 3, 3))


def tensor_contrast(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    """Rotation-invariant contrast of two tensors: ``|a - b|_F / scale``."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / scale)


def contrast_matrix(eps: np.ndarray) -> np.ndarray:
    """(M, M) pairwise contrasts (`tensor_contrast`), scaled by the largest Frobenius norm of the materials.

    Zero exactly between materials with equal tensors, so that an interface
    between them is not an interface.
    """
    if len(eps) == 0:
        return np.zeros((0, 0))
    scale = float(max(np.linalg.norm(e) for e in eps))
    diff = eps[:, None] - eps[None, :]
    return np.sqrt(np.sum(diff**2, axis=(-2, -1))) / scale


# ---------------------------------------------------------------- matrix helpers


def symmetric_inverse(a: Array) -> Array:
    """Inverse of (..., 3, 3) symmetric matrices from their adjugate (exact symmetry, cheap derivatives)."""
    a00, a01, a02 = a[..., 0, 0], a[..., 0, 1], a[..., 0, 2]
    a11, a12, a22 = a[..., 1, 1], a[..., 1, 2], a[..., 2, 2]
    c00 = a11 * a22 - a12 * a12
    c01 = a02 * a12 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    c11 = a00 * a22 - a02 * a02
    c12 = a01 * a02 - a00 * a12
    c22 = a00 * a11 - a01 * a01
    det = a00 * c00 + a01 * c01 + a02 * c02
    rows = [[c00, c01, c02], [c01, c11, c12], [c02, c12, c22]]
    return jnp.stack([jnp.stack(r, axis=-1) for r in rows], axis=-2) / det[..., None, None]


def to_six(a: Array) -> Array:
    """(..., 3, 3) symmetric -> (..., 6): xx, yy, zz, xy, xz, yz."""
    return jnp.stack([a[..., 0, 0], a[..., 1, 1], a[..., 2, 2], a[..., 0, 1], a[..., 0, 2], a[..., 1, 2]], axis=-1)


def from_six(s: Array) -> Array:
    """(..., 6) xx, yy, zz, xy, xz, yz -> (..., 3, 3) symmetric."""
    xx, yy, zz, xy, xz, yz = (s[..., i] for i in range(6))
    rows = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
    return jnp.stack([jnp.stack(r, axis=-1) for r in rows], axis=-2)


# ---------------------------------------------------------------- planar (laminate) average


def laminate(eps: Any, fractions: Array, normal: Array) -> Array:
    """Effective permittivity of parallel layers of anisotropic materials (Kottke's planar average).

    In a frame Q whose first axis is the interface normal n, the continuous
    components (normal D, tangential E) and the averaged ones (normal E,
    tangential D) are related by `tau(Q^T eps Q)`, so for layers of volume
    fractions f_m:

    ```text
    eps_eff = Q tau^-1(sum_m f_m tau(Q^T eps_m Q)) Q^T
    ```

    with, for a local tensor [[a, b^T], [b, C]] (a = n.eps n, b its normal /
    tangential coupling, C its tangential block), ``tau = [[-1/a, b^T/a],
    [b/a, C - b b^T/a]]`` and its inverse ``[[-1/t, -s^T/t], [-s/t, G - s
    s^T/t]]`` for ``tau = [[t, s^T], [s, G]]``. Here a, b and C are taken with
    the projectors ``n n^T`` and ``I - n n^T`` directly in simulation
    coordinates, so the result does not depend on a choice of tangent axes,
    nor on the sign of n. For isotropic materials it is the scalar formula,
    ``eps_eff^-1 = <1/eps> n n^T + (I - n n^T) / <eps>``. The order of the
    layers does not enter, so it is exact for any number of parallel layers.

    Args:
        eps: (M, 3, 3) symmetric positive-definite tensors (static numpy or traced).
        fractions: (..., M) volume fractions; normalized to sum to one.
        normal: (..., 3) interface normal (any length; sign does not matter).

    Returns:
        (..., 3, 3) the effective permittivity, symmetric positive definite.
    """
    dtype = fractions.dtype
    eps = jnp.asarray(eps, dtype=dtype)
    num = eps.shape[0]
    if num == 0:
        return jnp.broadcast_to(jnp.eye(3, dtype=dtype), (*fractions.shape[:-1], 3, 3))
    normal = jnp.asarray(normal, dtype=dtype)
    # Safe thresholds so squaring small divisors in reverse-mode AD never underflows tiny in float32/float64.
    eps2 = float(jnp.finfo(dtype).tiny)
    min_scale = 10.0 * math.sqrt(eps2)
    f_sum = jnp.sum(fractions, axis=-1, keepdims=True)
    ok_f = (f_sum > min_scale) & jnp.all(jnp.isfinite(fractions), axis=-1, keepdims=True)
    f_sum_safe = jnp.where(ok_f, f_sum, 1.0)
    f = jnp.where(ok_f, fractions / f_sum_safe, 0.0)
    # Normalize normal by its stop-gradient infinity norm so u^2 lies in [1, 3] and rsqrt'(u^2) cannot overflow.
    n_scale = jnp.max(jnp.abs(normal), axis=-1, keepdims=True)
    has_dir = (n_scale > min_scale) & jnp.all(jnp.isfinite(normal), axis=-1, keepdims=True)
    n_scale_safe = jax.lax.stop_gradient(jnp.where(has_dir, n_scale, 1.0))
    u = jnp.where(has_dir, normal / n_scale_safe, 0.0)
    u2 = jnp.sum(u * u, axis=-1, keepdims=True)
    u2_safe = jnp.where(has_dir & (u2 > 0), u2, 1.0)
    n = u * jax.lax.rsqrt(u2_safe)
    n = jnp.where(has_dir, n, jnp.asarray([0.0, 0.0, 1.0], dtype=dtype))
    # sum_m f_m tau_m, part by part; materials one at a time (M is small and static), so that no
    # (..., M, 3, 3) array is formed over a whole grid
    t = jnp.zeros(f.shape[:-1], dtype=dtype)  # -sum f / a
    s = jnp.zeros((*f.shape[:-1], 3), dtype=dtype)  # sum f b / a
    bb = jnp.zeros((*f.shape[:-1], 3, 3), dtype=dtype)  # sum f b b^T / a
    mean = jnp.zeros((*f.shape[:-1], 3, 3), dtype=dtype)  # sum f eps (its tangential block is sum f C)
    ok_terms = jnp.ones(f.shape[:-1], dtype=bool)
    for m in range(num):
        en = jnp.einsum("ij,...j->...i", eps[m], n, precision=_HIGHEST)  # eps n
        a = jnp.sum(en * n, axis=-1)  # n . eps n > 0
        ok_a = jnp.isfinite(a) & (jnp.abs(a) > min_scale)
        a_safe = jnp.where(ok_a, a, 1.0)
        w = jnp.where(ok_a, f[..., m] / a_safe, 0.0)
        ok_terms = ok_terms & (ok_a | (f[..., m] == 0))
        b = en - a[..., None] * n  # (I - n n^T) eps n
        t = t - w
        s = s + w[..., None] * b
        bb = bb + w[..., None, None] * b[..., :, None] * b[..., None, :]
        mean = mean + f[..., m, None, None] * eps[m]
    # tangential block of the mean: (I - n n^T) mean (I - n n^T)
    mn = jnp.einsum("...ij,...j->...i", mean, n, precision=_HIGHEST)
    tangential = (
        mean
        - n[..., :, None] * mn[..., None, :]
        - mn[..., :, None] * n[..., None, :]
        + jnp.sum(mn * n, axis=-1)[..., None, None] * n[..., :, None] * n[..., None, :]
    )
    gamma = tangential - bb  # sum f (C - b b^T / a)
    # tau^-1: a_eff = -1/t, b_eff = -s/t, C_eff = gamma - s s^T / t
    ok_t = jnp.isfinite(t) & (jnp.abs(t) > min_scale)
    t_safe = jnp.where(ok_t, t, -1.0)
    a_eff = -1.0 / t_safe
    b_eff = -s / t_safe[..., None]
    c_eff = gamma - s[..., :, None] * s[..., None, :] / t_safe[..., None, None]
    nn = n[..., :, None] * n[..., None, :]
    out = (
        a_eff[..., None, None] * nn
        + n[..., :, None] * b_eff[..., None, :]
        + b_eff[..., :, None] * n[..., None, :]
        + c_eff
    )
    ok = ok_f[..., 0] & ok_t & ok_terms & jnp.all(jnp.isfinite(out), axis=(-2, -1))
    return jnp.where(ok[..., None, None], out, jnp.eye(3, dtype=dtype))


# ---------------------------------------------------------------- junction closure


def _q1_gauss_gradients(sizes: tuple[float, float, float]) -> np.ndarray:
    """(8, 8, 3) gradients of the trilinear basis functions at the 2 x 2 x 2 Gauss points of a brick.

    [q, a, c]: point q = 4 q_x + 2 q_y + q_z, local node a = 4 a_x + 2 a_y + a_z, component c.
    """
    g = np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)])
    h = np.asarray(sizes, dtype=float)
    out = np.zeros((8, 8, 3))
    for q in range(8):
        xi = [g[(q >> 2) & 1], g[(q >> 1) & 1], g[q & 1]]
        for a in range(8):
            corner = [(a >> 2) & 1, (a >> 1) & 1, a & 1]
            for c in range(3):
                value = 1.0 / h[c]
                for k in range(3):
                    if k == c:
                        value *= -1.0 if corner[k] == 0 else 1.0
                    else:
                        value *= (1 - xi[k]) if corner[k] == 0 else xi[k]
                out[q, a, c] = value
    return out


def _q1_reference(sizes: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Trilinear (Q1) brick of the given sides: stiffness blocks and gradient integrals.

    Returns:
        stiffness: (3, 3, 8, 8) ``int_e d_c N_a d_d N_b``, local node a = 4 a_x + 2 a_y + a_z.
        gradient: (8, 3) ``int_e grad N_a``.
    """
    corners = np.array([[(a >> 2) & 1, (a >> 1) & 1, a & 1] for a in range(8)])
    h = np.asarray(sizes, dtype=float)
    vol = float(np.prod(h))
    mass = np.array([[1 / 3, 1 / 6], [1 / 6, 1 / 3]])  # int l_a l_b on [0, 1]
    slope = np.array([-1.0, 1.0])  # l_a'
    stiffness = np.zeros((3, 3, 8, 8))
    for c in range(3):
        for d in range(3):
            for a in range(8):
                for b in range(8):
                    value = vol / (h[c] * h[d])
                    for k in range(3):
                        ak, bk = corners[a, k], corners[b, k]
                        if k == c and k == d:
                            value *= slope[ak] * slope[bk]
                        elif k == c:
                            value *= slope[ak] * 0.5
                        elif k == d:
                            value *= slope[bk] * 0.5
                        else:
                            value *= mass[ak, bk]
                    stiffness[c, d, a, b] = value
    gradient = np.zeros((8, 3))
    for a in range(8):
        for c in range(3):
            gradient[a, c] = vol / h[c] * slope[corners[a, c]] * 0.25
    return stiffness, gradient


def _periodic_assembly(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Global node of every element's 8 local nodes on a periodic grid of sub-cells.

    Returns:
        nodes: (E, 8) global node index (E = n1 n2 n3, i-major), local node 4 a_x + 2 a_y + a_z.
        offsets: the same grid's node count, (n1 n2 n3,).
    """
    n1, n2, n3 = shape
    i, j, k = np.meshgrid(np.arange(n1), np.arange(n2), np.arange(n3), indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    nodes = np.stack(
        [((i + ax) % n1) * n2 * n3 + ((j + ay) % n2) * n3 + (k + az) % n3 for ax in (0, 1) for ay in (0, 1) for az in (0, 1)],
        axis=-1,
    )
    return nodes, np.arange(n1 * n2 * n3)


def _factor_and_solve_spd(kr: Array, rhs: Array, ke_scale: Array) -> tuple[Array, Array, Array]:
    """Factors SPD ``kr`` via Cholesky and solves ``kr x = rhs``.

    Returns ``(x, ok, chol)`` with ``ok`` certifying finite inputs, Cholesky diagonals
    above both the underflow floor and element-stiffness rounding noise, and finite
    solutions. Non-certified entries use ``x = 0`` and ``chol = I`` so linear solves
    and reverse-mode cotangents remain finite.
    """
    dtype = kr.dtype
    n = kr.shape[-1]
    finfo = jnp.finfo(dtype)
    eye = jnp.broadcast_to(jnp.eye(n, dtype=dtype), kr.shape)
    min_diag_abs = 10.0 * math.sqrt(float(finfo.tiny))
    scale_fin = jnp.where(jnp.isfinite(ke_scale) & (ke_scale > 0.0), ke_scale, 0.0)
    min_diag = jnp.maximum(min_diag_abs, jnp.sqrt(32.0 * float(finfo.eps) * scale_fin))
    finite = jnp.all(jnp.isfinite(kr), axis=(-2, -1)) & jnp.all(jnp.isfinite(rhs), axis=(-2, -1))
    kr_fin = jnp.where(finite[..., None, None], kr, eye)
    rhs_fin = jnp.where(finite[..., None, None], rhs, jnp.zeros_like(rhs))
    with jax.default_matmul_precision("highest"):
        chol_raw = jnp.linalg.cholesky(kr_fin)
    chol_ok = (
        finite
        & jnp.all(jnp.isfinite(chol_raw), axis=(-2, -1))
        & jnp.all(jnp.diagonal(chol_raw, axis1=-2, axis2=-1) > min_diag[..., None], axis=-1)
    )
    chol_safe = jnp.where(chol_ok[..., None, None], chol_raw, eye)
    with jax.default_matmul_precision("highest"):
        y = jax.scipy.linalg.solve_triangular(chol_safe, rhs_fin, lower=True)
        x_raw = jax.scipy.linalg.solve_triangular(jnp.swapaxes(chol_safe, -1, -2), y, lower=False)
    ok = chol_ok & jnp.all(jnp.isfinite(x_raw), axis=(-2, -1))
    x = jnp.where(ok[..., None, None], x_raw, jnp.zeros_like(rhs))
    chol = jnp.where(ok[..., None, None], chol_safe, eye)
    return x, ok, chol


@jax.custom_jvp
def _spd_solve(kr: Array, rhs: Array, ke_scale: Array) -> tuple[Array, Array]:
    """Solves ``kr x = rhs`` with SPD Cholesky certification, returning ``(x, ok_float)``."""
    x, ok, _ = _factor_and_solve_spd(kr, rhs, ke_scale)
    return x, ok.astype(kr.dtype)


@_spd_solve.defjvp
def _spd_solve_jvp(
    primals: tuple[Array, Array, Array], tangents: tuple[Array, Array, Array]
) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
    """Implicit JVP for ``_spd_solve``, avoiding Cholesky differentiation and zeroing non-SPD cotangents."""
    kr, rhs, ke_scale = primals
    d_kr, d_rhs, _ = tangents
    x, ok, chol = _factor_and_solve_spd(kr, rhs, ke_scale)
    mask = ok[..., None, None]
    with jax.default_matmul_precision("highest"):
        res = jnp.where(mask, d_rhs - jnp.matmul(d_kr, x, precision=_HIGHEST), jnp.zeros_like(rhs))
        y = jax.scipy.linalg.solve_triangular(chol, res, lower=True)
        dx = jax.scipy.linalg.solve_triangular(jnp.swapaxes(chol, -1, -2), y, lower=False)
    dx = jnp.where(mask, dx, jnp.zeros_like(x))
    ok_f = ok.astype(kr.dtype)
    return (x, ok_f), (dx, jnp.zeros_like(ok_f))


def periodic_cell_tensor(coefficients: Array, sides: Array | None = None) -> tuple[Array, Array]:
    """Effective tensor of a periodic cell made of constant-tensor sub-cells.

    Solves, for the three unit fields e_i, the cell problem ``div(A (e_i +
    grad phi_i)) = 0`` with periodic phi, by trilinear (Q1) finite elements on
    the grid of sub-cells (every element has its sub-cell's constant tensor):
    one stiffness matrix K per cell, factorized once (Cholesky) for the three
    right-hand sides. It returns the energy bilinear form of the computed
    fields, ``A_eff_ij = <(e_i + grad phi_i) . A (e_j + grad phi_j)>``,
    integrated exactly (2 x 2 x 2 Gauss points per element): a sum of the
    elements' positive semidefinite contributions, so symmetric and positive
    semidefinite by construction, and, as the energy of admissible fields, not
    below the exact cell tensor and second order in the solve's error (the
    equal ``<A> - F^T K^-1 F`` would be first order, and lose about the
    condition number of K in float32). The mean field is e_i (phi is
    periodic), so this is the cell's homogenized tensor, between the harmonic
    and arithmetic means of A in the matrix sense. Its derivative is the one
    of the linear solve (implicit), not of an iteration.

    A box with sides ``D = diag(sides)`` (a voxel of a graded grid) is the unit
    cube for ``D^-1 A D^-1`` (the coordinates scaled by D), whose effective
    tensor gives the box's as ``D A_eff D``; the scale of D cancels.

    Args:
        coefficients: (..., n1, n2, n3, 3, 3) symmetric positive-definite
            tensor of every sub-cell; the cell is the unit cube, sub-cells
            1/n1 x 1/n2 x 1/n3 (an axis of one sub-cell is constant along it).
        sides: (..., 3) the cell's sides (any common scale), or None: a cube.

    Returns:
        tensor: (..., 3, 3) the effective tensor.
        residual: (...,) the solve's normwise backward error ``|K phi + F| / (|K| |phi| + |F|)`` (Frobenius
            norms; NaN where the factorization failed).
    """
    if sides is not None:
        s = jnp.asarray(sides, dtype=coefficients.dtype)
        s = s / jnp.exp(jnp.mean(jnp.log(s), axis=-1, keepdims=True))  # geometric mean 1 (conditioning)
        inv = 1.0 / s[..., None, None, None, :]
        tensor, residual = periodic_cell_tensor(coefficients * inv[..., :, None] * inv[..., None, :])
        return tensor * s[..., :, None] * s[..., None, :], residual
    *batch, n1, n2, n3, _, _ = coefficients.shape
    dtype = coefficients.dtype
    num = n1 * n2 * n3
    stiffness, gradient = _q1_reference((1.0 / n1, 1.0 / n2, 1.0 / n3))
    nodes, _ = _periodic_assembly((n1, n2, n3))
    a = coefficients.reshape(*batch, num, 3, 3)
    # element matrices (..., E, 8, 8) = sum_cd A_cd S_cd, and loads (..., E, 8, 3) = int grad N_a . A e_i
    ke = jnp.einsum("...ecd,cdab->...eab", a, jnp.asarray(stiffness, dtype=dtype), precision=_HIGHEST)
    fe = jnp.einsum("ac,...eci->...eai", jnp.asarray(gradient, dtype=dtype), a, precision=_HIGHEST)
    rows = np.broadcast_to(nodes[:, :, None], (num, 8, 8))
    cols = np.broadcast_to(nodes[:, None, :], (num, 8, 8))
    k = jnp.zeros((*batch, num, num), dtype=dtype).at[..., rows, cols].add(ke)
    f = jnp.zeros((*batch, num, 3), dtype=dtype).at[..., nodes, :].add(fe)
    mean = jnp.mean(a, axis=-3)
    if num == 1:
        return mean, jnp.zeros(batch, dtype=dtype)
    # phi is defined up to a constant: fix it at node 0
    kr, fr = k[..., 1:, 1:], f[..., 1:, :]
    ke_scale = jax.lax.stop_gradient(jnp.max(jnp.abs(ke), axis=(-3, -2, -1)))
    phi, ok_f = _spd_solve(kr, -fr, ke_scale)
    ok = ok_f > 0
    # the fields e_i + grad phi_i at every element's Gauss points (..., E, 8 points, 3 fields, 3 components)
    full_phi = jnp.concatenate([jnp.zeros_like(phi[..., :1, :]), phi], axis=-2)  # (..., num, 3)
    grads = jnp.asarray(_q1_gauss_gradients((1.0 / n1, 1.0 / n2, 1.0 / n3)), dtype=dtype)  # (8, 8, 3)
    nodal = full_phi[..., nodes, :]  # (..., E, 8 nodes, 3 fields)
    fields = jnp.eye(3, dtype=dtype) + jnp.einsum("qac,...eai->...eqic", grads, nodal, precision=_HIGHEST)
    a_safe = jnp.where(ok[..., None, None, None], a, jnp.zeros_like(a))
    flux = jnp.einsum("...ecd,...eqjd->...eqjc", a_safe, fields, precision=_HIGHEST)  # A (e_j + grad phi_j)
    tensor = jnp.einsum("...eqic,...eqjc->...ij", fields, flux, precision=_HIGHEST) / (num * 8)
    tensor = 0.5 * (tensor + jnp.swapaxes(tensor, -1, -2))
    # Fallback for singular / non-SPD cells (e.g. lossless plasmonic eps1 + eps2 == 0):
    # well-conditioned isotropic average magnitude so primal stays finite & SPD and bwd stays NaN-free.
    a_fin = jnp.where(jnp.isfinite(a), a, 0.0)
    diag_mean = jnp.mean(jnp.abs(jnp.diagonal(a_fin, axis1=-2, axis2=-1)), axis=(-2, -1))
    min_scale = max(1e-12, 10.0 * float(jnp.finfo(dtype).tiny) ** (1.0 / 3.0))
    ok_iso = (diag_mean > min_scale) & jnp.isfinite(diag_mean)
    iso = jnp.where(ok_iso, diag_mean, jnp.ones_like(diag_mean))
    fallback = iso[..., None, None] * jnp.eye(3, dtype=dtype)
    t_scale = jax.lax.stop_gradient(jnp.max(jnp.abs(tensor), axis=(-2, -1)))
    t_norm = tensor / jnp.where(t_scale > 0, t_scale, 1.0)[..., None, None]
    t00, t01, t02 = t_norm[..., 0, 0], t_norm[..., 0, 1], t_norm[..., 0, 2]
    t11, t12, t22 = t_norm[..., 1, 1], t_norm[..., 1, 2], t_norm[..., 2, 2]
    minor2 = t00 * t11 - t01 * t01
    det3 = t00 * (t11 * t22 - t12 * t12) - t01 * (t01 * t22 - t02 * t12) + t02 * (t01 * t12 - t02 * t11)
    tensor_spd = (t00 > 0) & (minor2 > 0) & (det3 > 0)
    tensor_ok = ok & jnp.all(jnp.isfinite(tensor), axis=(-2, -1)) & tensor_spd
    out_tensor = jnp.where(tensor_ok[..., None, None], tensor, fallback)
    # the solve's normwise backward error |K phi + F| / (|K| |phi| + |F|) (NaN where the factorization failed)
    kr_safe = jnp.where(ok[..., None, None], kr, jnp.zeros_like(kr))
    fr_safe = jnp.where(ok[..., None, None], fr, jnp.zeros_like(fr))
    with jax.default_matmul_precision("highest"):
        misfit = jnp.sqrt(jnp.sum((jnp.matmul(kr_safe, phi, precision=_HIGHEST) + fr_safe) ** 2, axis=(-2, -1)))
    scale = jnp.sqrt(jnp.sum(kr_safe**2, axis=(-2, -1)) * jnp.sum(phi**2, axis=(-2, -1))) + jnp.sqrt(
        jnp.sum(fr_safe**2, axis=(-2, -1))
    )
    residual = jnp.where(scale > 0, misfit / jnp.where(scale > 0, scale, 1.0), 0.0)
    residual = jnp.where(tensor_ok, residual, jnp.asarray(jnp.nan, dtype=dtype))
    return out_tensor, jax.lax.stop_gradient(residual)


# ---------------------------------------------------------------- classification


def principal_direction(d: Array, squarings: int = 4) -> tuple[Array, Array]:
    """The leading eigenvector and eigenvalue of (..., 3, 3) positive semidefinite matrices.

    By repeated squaring (the columns of ``d^(2^squarings)``), which unlike an
    eigendecomposition has finite derivatives at repeated eigenvalues. The
    error of the direction is of order ``(lambda_2 / lambda_1)^(2^squarings)``:
    exact for rank one, and below 1e-9 where ``lambda_2 <= lambda_1 / 4``.

    Args:
        d: (..., 3, 3) symmetric positive semidefinite.
        squarings: number of squarings.

    Returns:
        direction: (..., 3) unit vector (sign arbitrary); e_z where d vanishes (entries below 1e-15).
        value: (...,) its Rayleigh quotient.
    """
    scale = jnp.max(jnp.abs(d), axis=(-2, -1), keepdims=True)
    ok = scale > _NO_DIRECTION
    # The direction does not depend on d's scale (it is homogeneous of degree 0), so the normalizations
    # carry no derivative: differentiating them would only add terms that cancel, through d / m^2, whose
    # m^2 underflows in float32 for small d and makes 0 * inf = NaN. Where d is (numerically) zero the
    # identity is squared instead. For a symmetric p with largest entry 1, p p has a diagonal entry >= 1.
    p = jnp.where(ok, d / jax.lax.stop_gradient(jnp.where(ok, scale, 1.0)), jnp.eye(3, dtype=d.dtype))
    for _ in range(squarings):
        p = jnp.matmul(p, p, precision=_HIGHEST)
        p = p / jax.lax.stop_gradient(jnp.max(jnp.abs(p), axis=(-2, -1), keepdims=True))
    norms = jnp.sum(p * p, axis=-2)  # (..., 3) squared column norms
    column = jnp.argmax(jax.lax.stop_gradient(norms), axis=-1)
    v = jnp.take_along_axis(p, column[..., None, None], axis=-1)[..., 0]
    v2 = jnp.sum(v * v, axis=-1, keepdims=True)
    good = ok[..., 0] & (v2 > 0)
    v = jnp.where(good, v * jax.lax.rsqrt(jnp.where(good, v2, 1.0)), jnp.asarray([0.0, 0.0, 1.0], dtype=d.dtype))
    value = jnp.einsum("...i,...ij,...j->...", v, d, v, precision=_HIGHEST)
    return v, value


def junction_weight(d: Array, low: float, high: float) -> tuple[Array, Array]:
    """Weight of the junction closure for a voxel with interface direction tensor d, and its planar normal.

    ``d = sum_faces w n n^T`` of a voxel's interfaces; its spread off its leading
    direction, ``r = 1 - lambda_1 / tr d`` (0 where all faces are parallel, up
    to 2/3), sets the weight ``omega``: 0 for r <= low (the laminate along the
    leading direction is used), 1 for r >= high (the closure), a smooth step
    in between. Only where omega < 1 is the direction used, and there the
    leading eigenvalue is separated (r < high <= 1/3), so it is well defined
    and differentiable; where r >= high its derivative is cut.

    Args:
        d: (..., 3, 3) direction tensors.
        low: r below which the voxel is planar.
        high: r above which it is a junction (at most 1/3).

    Returns:
        omega: (...,) in [0, 1].
        normal: (..., 3) the leading direction.
    """
    normal, value = principal_direction(d)
    trace = jnp.trace(d, axis1=-2, axis2=-1)
    some = trace > _NO_DIRECTION
    r = jnp.where(some, 1.0 - value / jnp.where(some, trace, 1.0), 0.0)
    x = jnp.clip((r - low) / (high - low), 0.0, 1.0)
    omega = x * x * (3.0 - 2.0 * x)
    normal = jnp.where((r < high)[..., None], normal, jax.lax.stop_gradient(normal))
    return omega, normal


def is_finite_spd(a: Array) -> Array:
    """Certify that ``(..., 3, 3)`` symmetric matrices are finite and strictly positive definite."""
    finite = jnp.all(jnp.isfinite(a), axis=(-2, -1))
    safe = jnp.where(finite[..., None, None], a, jnp.eye(3, dtype=a.dtype))
    sym = 0.5 * (safe + jnp.swapaxes(safe, -1, -2))
    a00 = sym[..., 0, 0]
    a01, a02 = sym[..., 0, 1], sym[..., 0, 2]
    a11, a12, a22 = sym[..., 1, 1], sym[..., 1, 2], sym[..., 2, 2]
    c00 = a11 * a22 - a12 * a12
    c01 = a02 * a12 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    m2 = a00 * a11 - a01 * a01
    det = a00 * c00 + a01 * c01 + a02 * c02
    min_det = 2.0 * math.sqrt(float(jnp.finfo(a.dtype).tiny))
    return finite & (a00 > 0.0) & (m2 > 0.0) & (det > min_det)


def safe_symmetric_inverse(a: Array) -> tuple[Array, Array]:
    """Symmetric 3x3 inverse with finite-SPD certification and double-where gradient guard."""
    ok = is_finite_spd(a)
    eye = jnp.eye(3, dtype=a.dtype)
    a_safe = jnp.where(ok[..., None, None], a, eye)
    sym = 0.5 * (a_safe + jnp.swapaxes(a_safe, -1, -2))
    inv_safe = symmetric_inverse(sym)
    return jnp.where(ok[..., None, None], inv_safe, eye), ok


def subcell_direction_blend(
    d: Array,
    gap: tuple[float, float] = SUBCELL_GAP,
) -> tuple[Array, Array, Array]:
    """Eigenvalue-free blend between leading-normal laminate and axis laminates for a sub-cell.

    Computes `lambda_1` via `principal_direction` and the second eigenvalue via the
    3x3 Frobenius trace/norm identity without ``eigvalsh``, preventing ``NaN`` gradients at
    repeated eigenvalues (e.g. 90-degree corners).

    Args:
        d: ``(..., 3, 3)`` positive-semidefinite interface direction tensor.
        gap: ``(low, high)`` bounds on ``(lambda_1 - lambda_2) / tr(D)``.

    Returns:
        ``(w_dom, normal, axis_weights)`` with ``w_dom`` shaped ``d.shape[:-2]``,
        ``normal`` shaped ``(*d.shape[:-2], 3)``, and ``axis_weights`` shaped
        ``(*d.shape[:-2], 3)`` summing to 1.
    """
    low, high = gap
    if not (0.0 <= low < high <= 1.0):
        raise ValueError(f"subcell gap must satisfy 0 <= low < high <= 1, got {gap!r}")
    dtype = d.dtype
    sym = 0.5 * (d + jnp.swapaxes(d, -1, -2))
    normal, lam1 = principal_direction(sym, squarings=8)
    trace = jnp.trace(sym, axis1=-2, axis2=-1)
    some = trace > _NO_DIRECTION
    safe_tr = jnp.where(some, trace, jnp.ones((), dtype=dtype))
    frob2 = jnp.sum(sym * sym, axis=(-2, -1))
    s = jnp.maximum(trace - lam1, 0.0)
    q = jnp.maximum(frob2 - lam1 * lam1, 0.0)
    disc = jnp.maximum(2.0 * q - s * s, 0.0) + (1e-6 * safe_tr) ** 2
    lam2 = jnp.where(some, 0.5 * (s + jnp.sqrt(jnp.where(some, disc, jnp.ones((), dtype=dtype)))), 0.0)
    gap_ratio = jnp.where(some, jnp.clip((lam1 - lam2) / safe_tr, 0.0, 1.0), 0.0)
    t = jnp.clip((gap_ratio - low) / (high - low), 0.0, 1.0)
    w_dom = t * t * (3.0 - 2.0 * t)
    normal = jnp.where((gap_ratio > low)[..., None], normal, jax.lax.stop_gradient(normal))
    diag = jnp.maximum(jnp.diagonal(sym, axis1=-2, axis2=-1), 0.0)
    diag_sum = jnp.sum(diag, axis=-1, keepdims=True)
    has_diag = diag_sum > _NO_DIRECTION
    axis_weights = jnp.where(
        has_diag,
        diag / jnp.where(has_diag, diag_sum, jnp.ones((), dtype=dtype)),
        jnp.full_like(diag, 1.0 / 3.0),
    )
    return w_dom, normal, axis_weights


def subcell_laminate(
    eps: Any,
    fractions: Array,
    direction: Array,
    *,
    gap: tuple[float, float] = SUBCELL_GAP,
) -> Array:
    """Smooth ``(..., 3, 3)`` Kottke laminate tensor with eigenvalue-free corner blending."""
    dtype = fractions.dtype
    eps_arr = jnp.asarray(eps, dtype=dtype)
    dir_arr = jnp.asarray(direction, dtype=dtype)
    w_dom, normal, axis_weights = subcell_direction_blend(dir_arr, gap=gap)
    lam_dom = laminate(eps_arr, fractions, normal)
    axes = jnp.eye(3, dtype=dtype)
    shape_axes = (1,) * (fractions.ndim - 1) + (3, 3)
    lam_axes = laminate(eps_arr, fractions[..., None, :], axes.reshape(shape_axes))
    lam_iso = jnp.einsum("...k,...kij->...ij", axis_weights, lam_axes, precision=_HIGHEST)
    w = w_dom[..., None, None]
    tensor = w * lam_dom + (1.0 - w) * lam_iso
    pure = jnp.max(fractions, axis=-1) >= 1.0
    exact = jnp.einsum("...m,mij->...ij", fractions, eps_arr, precision=_HIGHEST)
    return jnp.where(pure[..., None, None], exact, tensor)


def sparse_voxel_map(
    fn: Callable[[Array, Array, Any], Any],
    mask: Array,
    xs: Any = None,
    *,
    chunk_size: int = 64,
    group_size: int = 1024,
    max_items: int | None = None,
) -> Any:
    """Evaluate ``fn(chunk_indices, chunk_mask, chunk_xs)`` only on the True ``mask`` entries.

    The True entries are first compacted to the front of a flat index list (a cumsum scatter, no
    static capacity), then processed ``chunk_size`` at a time. A two-level ``jax.lax.scan`` with
    ``jax.lax.cond`` skips every group of ``group_size`` entries past the active count, so the loop
    runs ``ceil(active / chunk_size)`` evaluations of ``fn`` plus one cheap skipped ``cond`` per
    remaining group, instead of one ``cond`` per spatial brick of the whole grid. Supports
    ``jax.jit``, ``jax.grad`` and ``jax.jvp``; ``max_items`` caps the executed entries.
    """
    if not _COMPACT_SPARSE_MAP:
        return _sparse_voxel_map_bricks(fn, mask, xs, chunk_size=chunk_size, group_size=group_size, max_items=max_items)
    mask_arr = jnp.asarray(mask, dtype=jnp.bool_)
    grid_shape = mask_arr.shape
    n_total = math.prod(grid_shape)
    if chunk_size <= 0 or group_size <= 0:
        raise ValueError(f"chunk_size and group_size must be positive, got {chunk_size}, {group_size}")

    eff_chunk = min(chunk_size, max(1, n_total))
    idx_struct = jax.ShapeDtypeStruct((eff_chunk,), jnp.int32)
    mask_struct = jax.ShapeDtypeStruct((eff_chunk,), jnp.bool_)
    xs_struct = (
        None
        if xs is None
        else jax.tree.map(
            lambda leaf: jax.ShapeDtypeStruct((eff_chunk, *np.shape(leaf)[len(grid_shape) :]), jnp.asarray(leaf).dtype),
            xs,
        )
    )
    out_spec = jax.eval_shape(fn, idx_struct, mask_struct, xs_struct)
    if n_total == 0:
        return jax.tree.map(lambda s: jnp.zeros((*grid_shape, *s.shape[1:]), dtype=s.dtype), out_spec)

    flat_mask = mask_arr.reshape((n_total,))
    if max_items is not None:
        if max_items < 0:
            raise ValueError(f"max_items must be non-negative, got {max_items}")
        if max_items == 0:
            return jax.tree.map(lambda s: jnp.zeros((*grid_shape, *s.shape[1:]), dtype=s.dtype), out_spec)
        flat_mask = flat_mask & (jnp.cumsum(flat_mask.astype(jnp.int32)) <= max_items)

    chunks_per_group = max(1, group_size // eff_chunk)
    n_chunks = math.ceil(n_total / eff_chunk)
    chunks_per_group = min(chunks_per_group, n_chunks)
    n_groups = math.ceil(n_chunks / chunks_per_group)
    n_pad = n_groups * chunks_per_group * eff_chunk

    # Compact: active flat indices first (in grid order), the tail points at index 0 and is masked out
    rank = jnp.cumsum(flat_mask.astype(jnp.int32)) - 1
    count = rank[-1] + 1
    dest = jnp.where(flat_mask, rank, n_pad)
    perm = jnp.zeros((n_pad,), dtype=jnp.int32).at[dest].set(jnp.arange(n_total, dtype=jnp.int32), mode="drop")
    slot_live = jnp.arange(n_pad, dtype=jnp.int32) < count

    flat_xs = None if xs is None else jax.tree.map(lambda leaf: jnp.asarray(leaf).reshape((n_total, *jnp.shape(leaf)[len(grid_shape) :])), xs)
    # Gather xs into compacted order up front and feed them as scanned inputs: a gather from a closed-over
    # full-grid array inside the loop would make reverse mode stack one full-grid cotangent per chunk
    xs_c = None if flat_xs is None else jax.tree.map(lambda a: a[perm], flat_xs)

    @jax.checkpoint
    def _run_chunk(operand: tuple[Array, Array, Any]) -> Any:
        c_idx, c_mask, c_xs = operand
        raw = fn(c_idx, c_mask, c_xs)
        return jax.tree.map(
            lambda o: jnp.where(c_mask.reshape((eff_chunk,) + (1,) * (o.ndim - 1)), o, jnp.zeros((), dtype=o.dtype)),
            raw,
        )

    def _skip_chunk(_: tuple[Array, Array, Any]) -> Any:
        return jax.tree.map(lambda s: jnp.zeros((eff_chunk, *s.shape[1:]), dtype=s.dtype), out_spec)

    def _chunk_step(_: None, op: tuple[Array, Array, Any]) -> tuple[None, Any]:
        return None, jax.lax.cond(op[1][0], _run_chunk, _skip_chunk, op)

    def _run_group(op: tuple[Array, Array, Any]) -> Any:
        if chunks_per_group == 1:
            first = jax.tree.map(lambda a: a[0], op)
            return jax.tree.map(lambda a: a[None], _run_chunk(first))
        return jax.lax.scan(_chunk_step, None, op)[1]

    def _skip_group(_: tuple[Array, Array, Any]) -> Any:
        return jax.tree.map(lambda s: jnp.zeros((chunks_per_group, eff_chunk, *s.shape[1:]), dtype=s.dtype), out_spec)

    def _group_step(_: None, op: tuple[Array, Array, Any]) -> tuple[None, Any]:
        return None, jax.lax.cond(op[1][0, 0], _run_group, _skip_group, op)

    def _grouped(a: Array) -> Array:
        return a.reshape((n_groups, chunks_per_group, eff_chunk, *a.shape[1:]))

    g_ops = (_grouped(perm), _grouped(slot_live), None if xs_c is None else jax.tree.map(_grouped, xs_c))
    if n_groups == 1:
        g0 = jax.tree.map(lambda a: a[0], g_ops)
        stacked = jax.tree.map(lambda a: a[None], jax.lax.cond(g0[1][0, 0], _run_group, _skip_group, g0))
    else:
        _, stacked = jax.lax.scan(_group_step, None, g_ops)
    out_c = jax.tree.map(lambda a: a.reshape((n_pad, *a.shape[3:])), stacked)
    scatter_to = jnp.where(slot_live, perm, n_total)
    return jax.tree.map(
        lambda a: jnp.zeros((n_total, *a.shape[1:]), dtype=a.dtype)
        .at[scatter_to]
        .set(a, mode="drop")
        .reshape((*grid_shape, *a.shape[1:])),
        out_c,
    )


def _sparse_voxel_map_bricks(
    fn: Callable[[Array, Array, Any], Any],
    mask: Array,
    xs: Any = None,
    *,
    chunk_size: int = 64,
    group_size: int = 1024,
    max_items: int | None = None,
) -> Any:
    """Evaluate ``fn(chunk_indices, chunk_mask, chunk_xs)`` only on bricks containing True ``mask`` entries.

    Uses fixed spatial bricks with two-level ``jax.lax.scan`` + ``jax.lax.cond`` +
    ``jax.checkpoint``, supporting ``jax.jit``, ``jax.grad``, and ``jax.jvp`` without
    a static buffer cap when ``max_items=None``, and capping executed entries to
    ``max_items`` when an integer cap is provided.
    """
    mask_arr = jnp.asarray(mask, dtype=jnp.bool_)
    grid_shape = mask_arr.shape
    n_total = math.prod(grid_shape)
    if chunk_size <= 0 or group_size <= 0:
        raise ValueError(f"chunk_size and group_size must be positive, got {chunk_size}, {group_size}")

    eff_chunk = min(chunk_size, max(1, n_total))
    chunks_per_group = max(1, group_size // eff_chunk)
    eff_group = eff_chunk * chunks_per_group

    idx_struct = jax.ShapeDtypeStruct((eff_chunk,), jnp.int32)
    mask_struct = jax.ShapeDtypeStruct((eff_chunk,), jnp.bool_)
    if xs is None:
        xs_struct = None
    else:
        xs_struct = jax.tree.map(
            lambda leaf: jax.ShapeDtypeStruct((eff_chunk, *np.shape(leaf)[len(grid_shape) :]), jnp.asarray(leaf).dtype),
            xs,
        )
    out_spec = jax.eval_shape(fn, idx_struct, mask_struct, xs_struct)

    def _empty_output(n_rows: int) -> Any:
        return jax.tree.map(lambda s: jnp.zeros((n_rows, *s.shape[1:]), dtype=s.dtype), out_spec)

    if n_total == 0:
        return jax.tree.map(lambda s: jnp.zeros((*grid_shape, *s.shape[1:]), dtype=s.dtype), out_spec)

    flat_mask = mask_arr.reshape((n_total,))
    if max_items is not None:
        if max_items < 0:
            raise ValueError(f"max_items must be non-negative, got {max_items}")
        if max_items == 0:
            return jax.tree.map(lambda s: jnp.zeros((*grid_shape, *s.shape[1:]), dtype=s.dtype), out_spec)
        rank = jnp.cumsum(flat_mask.astype(jnp.int32))
        flat_mask = flat_mask & (rank <= max_items)

    n_chunks = math.ceil(n_total / eff_chunk)
    use_two_level = n_chunks > chunks_per_group
    n_groups = math.ceil(n_total / eff_group) if use_two_level else 1
    chunks_in_group = chunks_per_group if use_two_level else n_chunks
    n_pad = n_groups * chunks_in_group * eff_chunk
    pad = n_pad - n_total

    flat_idx = jnp.arange(n_total, dtype=jnp.int32)
    if pad > 0:
        flat_idx = jnp.pad(flat_idx, (0, pad), constant_values=0)
        flat_mask = jnp.pad(flat_mask, (0, pad), constant_values=False)

    def _flatten_and_pad(leaf: Array) -> Array:
        arr = jnp.asarray(leaf)
        flat = arr.reshape((n_total, *arr.shape[len(grid_shape) :]))
        if pad > 0:
            pad_widths = ((0, pad),) + ((0, 0),) * (flat.ndim - 1)
            flat = jnp.pad(flat, pad_widths)
        return flat

    flat_xs = None if xs is None else jax.tree.map(_flatten_and_pad, xs)

    @jax.checkpoint
    def _run_chunk(operand: tuple[Array, Array, Any]) -> Any:
        c_idx, c_mask, c_xs = operand
        raw = fn(c_idx, c_mask, c_xs)
        return jax.tree.map(
            lambda o: jnp.where(
                c_mask.reshape((eff_chunk,) + (1,) * (o.ndim - 1)),
                o,
                jnp.zeros((), dtype=o.dtype),
            ),
            raw,
        )

    def _skip_chunk(_: tuple[Array, Array, Any]) -> Any:
        return _empty_output(eff_chunk)

    def _chunk_step(_: None, chunk_op: tuple[Array, Array, Any]) -> tuple[None, Any]:
        c_idx, c_mask, c_xs = chunk_op
        out = jax.lax.cond(jnp.any(c_mask), _run_chunk, _skip_chunk, (c_idx, c_mask, c_xs))
        return None, out

    if not use_two_level:
        b_idx = flat_idx.reshape((n_chunks, eff_chunk))
        b_mask = flat_mask.reshape((n_chunks, eff_chunk))
        b_xs = (
            None
            if flat_xs is None
            else jax.tree.map(lambda a: a.reshape((n_chunks, eff_chunk, *a.shape[1:])), flat_xs)
        )
        if n_chunks == 1:
            out_flat = jax.lax.cond(
                jnp.any(b_mask[0]),
                _run_chunk,
                _skip_chunk,
                (b_idx[0], b_mask[0], None if b_xs is None else jax.tree.map(lambda a: a[0], b_xs)),
            )
        else:
            _, stacked = jax.lax.scan(_chunk_step, None, (b_idx, b_mask, b_xs))
            out_flat = jax.tree.map(lambda a: a.reshape((n_pad, *a.shape[2:])), stacked)
    else:
        g_idx = flat_idx.reshape((n_groups, chunks_in_group, eff_chunk))
        g_mask = flat_mask.reshape((n_groups, chunks_in_group, eff_chunk))
        g_xs = (
            None
            if flat_xs is None
            else jax.tree.map(lambda a: a.reshape((n_groups, chunks_in_group, eff_chunk, *a.shape[1:])), flat_xs)
        )

        @jax.checkpoint
        def _run_group(group_op: tuple[Array, Array, Any]) -> Any:
            _, res = jax.lax.scan(_chunk_step, None, group_op)
            return res

        def _skip_group(_: tuple[Array, Array, Any]) -> Any:
            return jax.tree.map(
                lambda s: jnp.zeros((chunks_in_group, eff_chunk, *s.shape[1:]), dtype=s.dtype), out_spec
            )

        def _group_step(_: None, group_op: tuple[Array, Array, Any]) -> tuple[None, Any]:
            _, gm, _ = group_op
            out = jax.lax.cond(jnp.any(gm), _run_group, _skip_group, group_op)
            return None, out

        _, stacked = jax.lax.scan(_group_step, None, (g_idx, g_mask, g_xs))
        out_flat = jax.tree.map(lambda a: a.reshape((n_pad, *a.shape[3:])), stacked)

    return jax.tree.map(
        lambda a: a[:n_total].reshape((*grid_shape, *a.shape[1:])),
        out_flat,
    )


def _inv_eps_array(eps: Any, dtype: Any) -> Array:
    """Inverse permittivity table computed in float64 when static, or traced when inside jit/grad."""
    if isinstance(eps, jax.core.Tracer):
        return jnp.linalg.inv(jnp.asarray(eps, dtype=dtype))
    return jnp.asarray(np.linalg.inv(np.asarray(eps, dtype=np.float64)), dtype=dtype)


def _exact_bulk_inverse(eps: Any, fractions: Array) -> Array:
    """Inverse permittivity tensor computed in float64 on pure single-material voxels."""
    inv_eps = _inv_eps_array(eps, fractions.dtype)
    return jnp.einsum("...m,mij->...ij", fractions, inv_eps, precision=_HIGHEST)


@partial(jax.jit, static_argnames=("is_diag_eps",))
def _homogenize_diagonal_jit(
    eps_arr: Array,
    inv_eps_arr: Array,
    fractions: Array,
    direction: Array,
    is_diag_eps: bool = True,
) -> Array:
    """JIT-fused diagonal inverse permittivity kernel avoiding 3x3 matrix intermediates.

    Weights the three coordinate-axis Kottke laminates *in inverse space* by ``D_kk / tr(D)``:
    ``sum_k (D_kk / tr D) inv(L_k)_cc``. For isotropic or diagonally anisotropic materials and any
    planar interface (oblique, sloped, or axis-aligned), this is algebraically identical to Kottke's
    ``n_c^2 <1/eps_c> + (1 - n_c^2) / <eps_c>``; at multi-face corners (``D`` of rank >= 2) it retains
    all faces smoothly with no eigenvalue-gap switch. Accepts ``direction`` shaped either ``(..., 3)``
    (diagonal ``(D_xx, D_yy, D_zz)``) or ``(..., 3, 3)``.
    """
    dtype = fractions.dtype
    if direction.ndim == fractions.ndim and direction.shape[-1] == 3:
        diag_d = jnp.maximum(direction, 0.0)
    else:
        sym = 0.5 * (direction + jnp.swapaxes(direction, -1, -2))
        diag_d = jnp.maximum(jnp.diagonal(sym, axis1=-2, axis2=-1), 0.0)
    d_sum = jnp.sum(diag_d, axis=-1, keepdims=True)
    has_d = d_sum > _NO_DIRECTION
    axis_weights = jnp.where(
        has_d,
        diag_d / jnp.where(has_d, d_sum, jnp.ones((), dtype=dtype)),
        jnp.full_like(diag_d, 1.0 / 3.0),
    )
    pure = jnp.max(fractions, axis=-1, keepdims=True) >= 1.0
    inv_eps_d = jnp.diagonal(inv_eps_arr, axis1=-2, axis2=-1)
    mean_inv = jnp.einsum("...m,mc->...c", fractions, inv_eps_d, precision=_HIGHEST)
    if is_diag_eps:
        eps_d = jnp.diagonal(eps_arr, axis1=-2, axis2=-1)
        mean_eps = jnp.einsum("...m,mc->...c", fractions, eps_d, precision=_HIGHEST)
        tiny = jnp.asarray(float(jnp.finfo(dtype).tiny), dtype=dtype)
        inv_mean = jnp.where(mean_eps > tiny, 1.0 / jnp.maximum(mean_eps, tiny), jnp.ones_like(mean_eps))
        diag = axis_weights * mean_inv + (1.0 - axis_weights) * inv_mean
        return jnp.where(pure, mean_inv, diag)
    axes = jnp.eye(3, dtype=dtype)
    lam_axes = laminate(eps_arr, fractions[..., None, :], axes.reshape((1,) * (fractions.ndim - 1) + (3, 3)))
    inv_axes, _ = safe_symmetric_inverse(lam_axes)
    diag_axes = jnp.diagonal(inv_axes, axis1=-2, axis2=-1)
    diag = jnp.einsum("...k,...kc->...c", axis_weights, diag_axes, precision=_HIGHEST)
    return jnp.where(pure, mean_inv, diag)


def homogenize_diagonal(
    eps: Any,
    fractions: Array,
    direction: Array,
    *,
    gap: tuple[float, float] | None = None,
) -> Array:
    """Diagonal inverse permittivity ``(..., 3)`` from voxel ``(fractions, direction)``."""
    _ = gap
    dtype = fractions.dtype
    eps_arr = jnp.asarray(eps, dtype=dtype)
    inv_eps_arr = _inv_eps_array(eps, dtype)
    dir_arr = jnp.asarray(direction, dtype=dtype)
    if isinstance(eps, jax.core.Tracer):
        diag_flag = False
    else:
        eps_np = np.asarray(eps, dtype=np.float64)
        off = eps_np - np.eye(3) * eps_np
        diag_flag = bool(np.max(np.abs(off)) <= 1e-12 * max(float(np.max(np.abs(eps_np))), 1.0))
    if fractions.ndim >= 4 and math.prod(fractions.shape[:-1]) > 8_000_000:
        out_z = jax.lax.map(
            lambda pair: _homogenize_diagonal_jit(eps_arr, inv_eps_arr, pair[0], pair[1], is_diag_eps=diag_flag),
            (jnp.moveaxis(fractions, 2, 0), jnp.moveaxis(dir_arr, 2, 0)),
        )
        return jnp.moveaxis(out_z, 0, 2)
    return _homogenize_diagonal_jit(eps_arr, inv_eps_arr, fractions, dir_arr, is_diag_eps=diag_flag)


@partial(jax.jit, static_argnames=("junction_low", "junction_high"))
def _junction_active_jit(
    eps_arr: Array,
    fractions: Array,
    dir_arr: Array,
    junction_low: float,
    junction_high: float,
) -> tuple[Array, Array, Array, Array]:
    """JIT-fused planar Kottke laminate and junction classifier mask."""
    omega, normal = junction_weight(dir_arr, junction_low, junction_high)
    planar = laminate(eps_arr, fractions, normal)
    active = omega > 0
    return omega, planar, active, jnp.sum(active, dtype=jnp.int32)


@jax.jit
def _planar_only_jit(
    inv_eps_arr: Array,
    fractions: Array,
    planar: Array,
    total_junctions: Array,
) -> tuple[Array, Array]:
    """JIT-fused planar-only inverse permittivity without junction closure."""
    pure = jnp.max(fractions, axis=-1) >= 1.0
    exact = jnp.einsum("...m,mij->...ij", fractions, inv_eps_arr, precision=_HIGHEST)
    inv_planar = jnp.where(pure[..., None, None], exact, safe_symmetric_inverse(planar)[0])
    return to_six(inv_planar), jnp.maximum(total_junctions, 0)


@partial(jax.jit, static_argnames=("closure_residual_tol",))
def _blend_full_jit(
    inv_eps_arr: Array,
    fractions: Array,
    planar: Array,
    omega: Array,
    executed_mask: Array,
    junc_tensor: Array,
    residuals: Array,
    closure_residual_tol: float,
) -> tuple[Array, Array, Array]:
    """JIT-fused certification, planar/junction blending, and safe inversion."""
    dtype = fractions.dtype
    pure = jnp.max(fractions, axis=-1) >= 1.0
    exact = jnp.einsum("...m,mij->...ij", fractions, inv_eps_arr, precision=_HIGHEST)
    eye3 = jnp.eye(3, dtype=dtype)
    planar_ok = is_finite_spd(planar)
    planar_safe = jnp.where(planar_ok[..., None, None], planar, eye3)
    junc_ok = is_finite_spd(junc_tensor)
    res_ok = jnp.isfinite(residuals) & (residuals <= closure_residual_tol)
    junc_cert = executed_mask & junc_ok & res_ok
    junc_safe = jnp.where(junc_cert[..., None, None], junc_tensor, planar_safe)
    eff_omega = jnp.where(junc_cert, omega, jnp.zeros((), dtype=dtype))
    w = eff_omega[..., None, None]
    blended = (1.0 - w) * planar_safe + w * junc_safe
    inv_mixed, blended_ok = safe_symmetric_inverse(blended)
    planar_inv, _ = safe_symmetric_inverse(planar)
    inv_safe_mix = jnp.where(blended_ok[..., None, None], inv_mixed, planar_inv)
    inv_eff = jnp.where(pure[..., None, None], exact, inv_safe_mix)

    executed_count = jnp.sum(executed_mask, dtype=jnp.int32)
    bad_res = jnp.where(junc_ok & planar_ok & blended_ok, residuals, jnp.asarray(jnp.nan, dtype=dtype))
    return (
        to_six(inv_eff),
        executed_count,
        jnp.max(jnp.where(executed_mask, bad_res, jnp.zeros((), dtype=dtype))),
    )


def homogenize_full(
    eps: Any,
    fractions: Array,
    direction: Array,
    junction_cell_fn: Callable[[Array, Array, Any], tuple[Array, Array, Array]] | None,
    *,
    junction_xs: Any = None,
    junction_low: float = JUNCTION_LOW,
    junction_high: float = JUNCTION_HIGH,
    subcell_gap: tuple[float, float] = SUBCELL_GAP,
    closure_residual_tol: float = CLOSURE_RESIDUAL,
    chunk_size: int | None = None,
    group_size: int | None = None,
    max_items: int | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Unified full-tensor inverse permittivity ``(..., 6)`` blending planar Kottke laminate and cell closures."""
    chunk_size = _JUNCTION_CHUNK if chunk_size is None else chunk_size
    group_size = _JUNCTION_GROUP if group_size is None else group_size
    dtype = fractions.dtype
    eps_arr = jnp.asarray(eps, dtype=dtype)
    inv_eps_arr = _inv_eps_array(eps, dtype)
    dir_arr = jnp.asarray(direction, dtype=dtype)
    large_3d = fractions.ndim >= 4 and math.prod(fractions.shape[:-1]) > 8_000_000
    if large_3d:
        omega_z, planar_z, active_z, cnt_z = jax.lax.map(
            lambda pair: _junction_active_jit(
                eps_arr, pair[0], pair[1], junction_low=junction_low, junction_high=junction_high
            ),
            (jnp.moveaxis(fractions, 2, 0), jnp.moveaxis(dir_arr, 2, 0)),
        )
        omega = jnp.moveaxis(omega_z, 0, 2)
        planar = jnp.moveaxis(planar_z, 0, 2)
        active = jnp.moveaxis(active_z, 0, 2)
        total_junctions = jnp.sum(cnt_z, dtype=jnp.int32)
    else:
        omega, planar, active, total_junctions = _junction_active_jit(
            eps_arr, fractions, dir_arr, junction_low=junction_low, junction_high=junction_high
        )

    if junction_cell_fn is None or max_items == 0:
        if large_3d:
            inv6_z, _ = jax.lax.map(
                lambda pair: _planar_only_jit(inv_eps_arr, pair[0], pair[1], jnp.zeros((), dtype=jnp.int32)),
                (jnp.moveaxis(fractions, 2, 0), jnp.moveaxis(planar, 2, 0)),
            )
            inv6_planar = jnp.moveaxis(inv6_z, 0, 2)
            overflow = jnp.maximum(total_junctions, 0)
        else:
            inv6_planar, overflow = _planar_only_jit(inv_eps_arr, fractions, planar, total_junctions)
        status = {
            "junctions": total_junctions,
            "junction_overflow": overflow if max_items == 0 else jnp.zeros((), dtype=jnp.int32),
            "closure_residual": jnp.zeros((), dtype=dtype),
        }
        return inv6_planar, status

    if max_items is None:
        executed_mask = active
    else:
        if max_items < 0:
            raise ValueError(f"max_items must be non-negative, got {max_items}")
        rank = jnp.cumsum(active.astype(jnp.int32).ravel()).reshape(active.shape)
        executed_mask = active & (rank <= max_items)

    def _solve_chunk(chunk_indices: Array, chunk_mask: Array, chunk_xs: Any) -> tuple[Array, Array]:
        sub_fractions, sub_dir, sub_sides = junction_cell_fn(chunk_indices, chunk_mask, chunk_xs)
        sub_coeffs = subcell_laminate(eps_arr, sub_fractions, sub_dir, gap=subcell_gap)
        eye3 = jnp.eye(3, dtype=dtype)
        safe = chunk_mask.reshape((chunk_mask.shape[0],) + (1,) * (sub_coeffs.ndim - 1))
        safe_coeffs = jnp.where(safe, sub_coeffs, eye3)
        safe_sides = jnp.where(chunk_mask[:, None], jnp.asarray(sub_sides, dtype=dtype), 1.0)
        return periodic_cell_tensor(safe_coeffs, safe_sides)

    scan_xs = fractions if junction_xs is None else (fractions, junction_xs)
    junc_tensor, residuals = sparse_voxel_map(
        _solve_chunk,
        active,
        scan_xs,
        chunk_size=chunk_size,
        group_size=group_size,
        max_items=max_items,
    )
    if large_3d:
        inv6_z, exec_z, res_z = jax.lax.map(
            lambda xs_z: _blend_full_jit(
                inv_eps_arr,
                xs_z[0],
                xs_z[1],
                xs_z[2],
                xs_z[3],
                xs_z[4],
                xs_z[5],
                closure_residual_tol=closure_residual_tol,
            ),
            (
                jnp.moveaxis(fractions, 2, 0),
                jnp.moveaxis(planar, 2, 0),
                jnp.moveaxis(omega, 2, 0),
                jnp.moveaxis(executed_mask, 2, 0),
                jnp.moveaxis(junc_tensor, 2, 0),
                jnp.moveaxis(residuals, 2, 0),
            ),
        )
        inv6_eff = jnp.moveaxis(inv6_z, 0, 2)
        executed_count = jnp.sum(exec_z, dtype=jnp.int32)
        closure_residual = jnp.max(res_z)
    else:
        inv6_eff, executed_count, closure_residual = _blend_full_jit(
            inv_eps_arr,
            fractions,
            planar,
            omega,
            executed_mask,
            junc_tensor,
            residuals,
            closure_residual_tol=closure_residual_tol,
        )
    status = {
        "junctions": total_junctions,
        "junction_overflow": jnp.maximum(total_junctions - executed_count, 0),
        "closure_residual": closure_residual,
    }
    return inv6_eff, status
