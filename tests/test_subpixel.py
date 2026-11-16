"""Tests for the material tensor contract and the subpixel tensor averages (bifrost.subpixel).

Analytic fixtures run in float64 (``jax.experimental.enable_x64``) against independent references: the
paper's tau transform in an explicit tangent frame, and the layered continuity conditions solved directly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import bifrost as bf

fdtdx = pytest.importorskip("fdtdx")


RTOL64 = 1e-10  # float64 acceptance on well-conditioned analytic fixtures
RTOL32 = 1e-4  # float32


def enable_x64():
    """float64 inside the block (jax's flag as a context manager)."""
    return jax.enable_x64(True)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def random_spd(rng: np.random.Generator, lo: float = 1.5, hi: float = 12.0) -> np.ndarray:
    """A symmetric positive-definite tensor with unequal principal values and rotated axes."""
    r = random_rotation(rng)
    return r @ np.diag(rng.uniform(lo, hi, size=3)) @ r.T


def frame_laminate(eps: np.ndarray, f: np.ndarray, n: np.ndarray, tangent: np.ndarray | None = None) -> np.ndarray:
    """Kottke's tau average written as in the paper: in an explicit frame Q = [n, t1, t2] (numpy, float64)."""
    n = n / np.linalg.norm(n)
    seed = tangent if tangent is not None else (np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0]))
    t1 = seed - (seed @ n) * n
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    q = np.stack([n, t1, t2], axis=1)
    taus = []
    for e in eps:
        loc = q.T @ e @ q
        t = np.empty((3, 3))
        t[0, 0] = -1 / loc[0, 0]
        for i in (1, 2):
            t[0, i] = loc[0, i] / loc[0, 0]
            t[i, 0] = loc[i, 0] / loc[0, 0]
            for j in (1, 2):
                t[i, j] = loc[i, j] - loc[i, 0] * loc[0, j] / loc[0, 0]
        taus.append(t)
    tau = np.einsum("m,mij->ij", f / f.sum(), np.stack(taus))
    out = np.empty((3, 3))
    out[0, 0] = -1 / tau[0, 0]
    for i in (1, 2):
        out[0, i] = -tau[0, i] / tau[0, 0]
        out[i, 0] = -tau[i, 0] / tau[0, 0]
        for j in (1, 2):
            out[i, j] = tau[i, j] - tau[i, 0] * tau[0, j] / tau[0, 0]
    return q @ out @ q.T


def layered_reference(eps: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Effective tensor of layers normal to z from the continuity conditions, solved directly (numpy).

    In every layer E_x, E_y and D_z are the same; for a mean field <E> the layers' E_z and then <D> follow
    from a linear solve. No tau transform is used.
    """
    f = f / f.sum()
    out = np.empty((3, 3))
    for k in range(3):
        mean = np.eye(3)[k]
        # unknowns: E_x, E_y (shared), D_z (shared); E_z,m = (D_z - eps_zx E_x - eps_zy E_y) / eps_zz
        # <E_z> = mean_z gives the third equation; <E_x>, <E_y> = mean_x, mean_y the first two
        a = np.zeros((3, 3))
        b = np.array([mean[0], mean[1], mean[2]])
        a[0, 0] = a[1, 1] = 1.0
        for fm, e in zip(f, eps, strict=True):
            a[2, 0] += fm * (-e[2, 0] / e[2, 2])
            a[2, 1] += fm * (-e[2, 1] / e[2, 2])
            a[2, 2] += fm / e[2, 2]
        ex, ey, dz = np.linalg.solve(a, b)
        d = np.zeros(3)
        for fm, e in zip(f, eps, strict=True):
            field = np.array([ex, ey, (dz - e[2, 0] * ex - e[2, 1] * ey) / e[2, 2]])
            d += fm * (e @ field)
        out[:, k] = d
    return out


def rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


# ---------------------------------------------------------------- material tensor contract


def test_material_tensor_formats_and_fdtdx_ordering() -> None:
    """Scalars, diagonals, nested 3 x 3 and 9-tuples become the (3, 3) matrix in fdtdx's row-major order."""
    assert np.array_equal(bf.subpixel.material_tensor(fdtdx.Material(permittivity=2.5)), 2.5 * np.eye(3))
    assert np.array_equal(bf.subpixel.material_tensor(fdtdx.Material(permittivity=(2.0, 3.0, 4.0))), np.diag([2.0, 3.0, 4.0]))
    nested = ((4.0, 0.5, 0.25), (0.5, 3.0, -0.1), (0.25, -0.1, 2.0))
    assert np.array_equal(bf.subpixel.material_tensor(fdtdx.Material(permittivity=nested)), np.asarray(nested))
    flat = (4.0, 0.5, 0.25, 0.5, 3.0, -0.1, 0.25, -0.1, 2.0)
    assert np.array_equal(bf.subpixel.material_tensor(fdtdx.Material(permittivity=flat)), np.asarray(nested))


@pytest.mark.parametrize(
    ("eps", "match"),
    [
        (((4.0, 0.5, 0.0), (0.4, 3.0, 0.0), (0.0, 0.0, 2.0)), "non-symmetric"),
        (((1.0, 2.0, 0.0), (2.0, 1.0, 0.0), (0.0, 0.0, 2.0)), "not positive definite"),
        (((float("nan"), 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), "non-finite"),
        (((-2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), "not positive definite"),
    ],
)
def test_invalid_permittivities_are_rejected(eps, match: str) -> None:
    """Non-finite, non-symmetric and indefinite tensors raise; nothing is clipped or symmetrized silently."""
    with pytest.raises(ValueError, match=match):
        bf.subpixel.check_permittivity(np.asarray(eps), "bad")


def test_symmetry_and_definiteness_tolerances_are_relative() -> None:
    """A large tensor with float32 round-off asymmetry passes; the same absolute asymmetry on a small one does
    not.
    """
    big = 1e4 * np.diag([1.0, 2.0, 3.0])
    big[0, 1] += 1e-3  # 1e-7 relative
    bf.subpixel.check_permittivity(big, "big")
    small = np.diag([1.0, 2.0, 3.0])
    small[0, 1] += 1e-3
    with pytest.raises(ValueError, match="non-symmetric"):
        bf.subpixel.check_permittivity(small, "small")


def test_material_tensors_skip_placeholders_and_symmetrize_round_off() -> None:
    mats = {"a": fdtdx.Material(permittivity=2.0), "x": fdtdx.Material(permittivity=(1.0, 1.0, 2.0))}
    eps = bf.subpixel.material_tensors(mats, ["x", "a"], ignore=["x"])
    assert np.array_equal(eps[0], np.eye(3)) and np.array_equal(eps[1], 2 * np.eye(3))
    assert np.array_equal(eps, np.swapaxes(eps, -1, -2))


def test_contrast_is_rotation_invariant_and_zero_for_equal_tensors() -> None:
    rng = np.random.default_rng(0)
    a, b = random_spd(rng), random_spd(rng)
    r = random_rotation(rng)
    c = bf.subpixel.contrast_matrix(np.stack([a, b, a]))
    cr = bf.subpixel.contrast_matrix(np.stack([r @ a @ r.T, r @ b @ r.T, r @ a @ r.T]))
    np.testing.assert_allclose(c, cr, rtol=1e-12, atol=1e-15)
    assert c[0, 2] == 0.0 and c[0, 1] > 0
    # a trace or the diagonal cannot see an off-diagonal-only difference; the contrast does
    d = np.diag([3.0, 3.0, 2.0])
    e = d.copy()
    e[0, 1] = e[1, 0] = 0.7
    # |e - d|_F = 0.7 sqrt(2) over the larger norm |e|_F = 4.79
    assert np.trace(d) == np.trace(e) and bf.subpixel.contrast_matrix(np.stack([d, e]))[0, 1] == pytest.approx(0.2065, abs=1e-4)


# ---------------------------------------------------------------- matrix helpers


def test_symmetric_inverse_and_six_entry_storage() -> None:
    with enable_x64():
        rng = np.random.default_rng(1)
        a = np.stack([random_spd(rng) for _ in range(20)])
        a = 0.5 * (a + np.swapaxes(a, -1, -2))  # exactly symmetric
        inv = np.asarray(bf.subpixel.symmetric_inverse(jnp.asarray(a)))
        assert rel(inv, np.linalg.inv(a)) < RTOL64
        assert np.array_equal(np.asarray(bf.subpixel.from_six(bf.subpixel.to_six(jnp.asarray(a)))), a)


# ---------------------------------------------------------------- laminate


def _laminate64(eps, f, n) -> np.ndarray:
    with enable_x64():
        return np.asarray(bf.subpixel.laminate(np.asarray(eps), jnp.asarray(f, dtype=jnp.float64), jnp.asarray(n, dtype=jnp.float64)))


@pytest.mark.parametrize("seed", range(6))
def test_laminate_matches_the_papers_tau_in_an_explicit_frame(seed: int) -> None:
    """Coordinate-free projectors == the paper's tau, tau^-1 in a frame [n, t1, t2], for random anisotropic
    materials (unequal principal values, rotated axes), fractions and normals; and the frame's tangent axes do
    not matter.
    """
    rng = np.random.default_rng(seed)
    num = 2 + seed % 3
    eps = np.stack([random_spd(rng) for _ in range(num)])
    f = rng.dirichlet(np.ones(num))
    n = rng.normal(size=3)
    want = frame_laminate(eps, f, n)
    assert rel(_laminate64(eps, f, n), want) < RTOL64
    other = frame_laminate(eps, f, n, tangent=rng.normal(size=3))
    assert rel(other, want) < RTOL64


def test_laminate_matches_the_layered_continuity_solution() -> None:
    """For layers normal to z (any anisotropy, also off-diagonal entries coupling z), the laminate is the
    effective tensor of the continuity conditions solved directly; fractions near 0 and 1 included.
    """
    rng = np.random.default_rng(2)
    for f0 in (1e-6, 0.3, 0.5, 1 - 1e-6):
        eps = np.stack([random_spd(rng), random_spd(rng), random_spd(rng)])
        f = np.array([f0, (1 - f0) * 0.4, (1 - f0) * 0.6])
        assert rel(_laminate64(eps, f, [0.0, 0.0, 1.0]), layered_reference(eps, f)) < RTOL64


def test_laminate_with_off_diagonal_only_contrast() -> None:
    """Two materials whose tensors differ only in xy (same trace and diagonal) still form an interface: the
    laminate across x differs from both and from their mean, and matches the layered reference (rotated so the
    layers are normal to z).
    """
    d = np.diag([3.0, 3.0, 2.0])
    e = d.copy()
    e[0, 1] = e[1, 0] = 0.8
    eps = np.stack([d, e])
    f = np.array([0.5, 0.5])
    out = _laminate64(eps, f, [1.0, 0.0, 0.0])
    assert rel(out, 0.5 * (d + e)) > 1e-3
    # rotate x -> z: layers normal to z
    p = np.array([[0, 0, 1.0], [0, 1.0, 0], [1.0, 0, 0]])
    ref = p.T @ layered_reference(np.stack([p @ d @ p.T, p @ e @ p.T]), f) @ p
    assert rel(out, ref) < RTOL64


def test_laminate_symmetries() -> None:
    """Normal reversal and material relabeling change nothing; rotating materials and normal together rotates
    the result (covariance); homogeneous fractions give the material itself.
    """
    rng = np.random.default_rng(3)
    eps = np.stack([random_spd(rng) for _ in range(3)])
    f = rng.dirichlet(np.ones(3))
    n = rng.normal(size=3)
    out = _laminate64(eps, f, n)
    assert rel(_laminate64(eps, f, -n), out) < RTOL64
    perm = [2, 0, 1]
    assert rel(_laminate64(eps[perm], f[perm], n), out) < RTOL64
    r = random_rotation(rng)
    rotated = _laminate64(np.einsum("ij,mjk,lk->mil", r, eps, r), f, r @ n)
    assert rel(rotated, r @ out @ r.T) < RTOL64
    for m in range(3):
        assert rel(_laminate64(eps, np.eye(3)[m], n), eps[m]) < 1e-13


def test_laminate_scalar_special_case_and_bounds() -> None:
    """Isotropic materials: eps_eff^-1 = <1/eps> n n^T + (I - n n^T)/<eps>. Anisotropic: symmetric, between
    the harmonic and arithmetic means in the matrix sense (Wiener bounds).
    """
    n = np.array([0.3, -0.5, 0.8])
    n = n / np.linalg.norm(n)
    eps = np.stack([12.0 * np.eye(3), 2.1 * np.eye(3)])
    f = np.array([0.37, 0.63])
    inv = np.linalg.inv(_laminate64(eps, f, n))
    nn = np.outer(n, n)
    want = (f / np.array([12.0, 2.1])).sum() * nn + (np.eye(3) - nn) / (f * np.array([12.0, 2.1])).sum()
    assert rel(inv, want) < RTOL64
    rng = np.random.default_rng(4)
    for _ in range(20):
        eps = np.stack([random_spd(rng, 1.0, 20.0) for _ in range(3)])
        f = rng.dirichlet(np.ones(3))
        out = _laminate64(eps, f, rng.normal(size=3))
        assert np.abs(out - out.T).max() < 1e-12
        arithmetic = np.einsum("m,mij->ij", f, eps)
        harmonic = np.linalg.inv(np.einsum("m,mij->ij", f, np.linalg.inv(eps)))
        scale = np.abs(arithmetic).max()
        assert np.linalg.eigvalsh(arithmetic - out).min() > -1e-12 * scale
        assert np.linalg.eigvalsh(out - harmonic).min() > -1e-12 * scale


def test_laminate_float32_and_gradients() -> None:
    """float32 within 1e-4 of float64; jax derivatives in the fractions and the normal match central finite
    differences in float64.
    """
    rng = np.random.default_rng(5)
    eps = np.stack([random_spd(rng) for _ in range(2)])
    f = np.array([0.3, 0.7])
    n = np.array([0.2, 0.9, -0.4])
    out32 = np.asarray(bf.subpixel.laminate(eps, jnp.asarray(f, jnp.float32), jnp.asarray(n, jnp.float32)))
    assert rel(out32, _laminate64(eps, f, n)) < RTOL32
    with enable_x64():

        def g(x):
            return jnp.sum(jnp.sin(jnp.arange(9.0).reshape(3, 3)) * bf.subpixel.laminate(eps, jnp.stack([x[0], 1 - x[0]]), x[1:]))

        x = jnp.asarray([0.3, 0.2, 0.9, -0.4])
        grad = np.asarray(jax.grad(g)(x))
        for i in range(4):
            h = 1e-6
            fd = (g(x.at[i].add(h)) - g(x.at[i].add(-h))) / (2 * h)
            assert abs(grad[i] - float(fd)) < 1e-7 * max(1.0, abs(float(fd)))


# ---------------------------------------------------------------- periodic cell problem


def _cell64(coefficients, sides=None) -> tuple[np.ndarray, float]:
    with enable_x64():
        s = None if sides is None else jnp.asarray(sides, dtype=jnp.float64)
        t, r = bf.subpixel.periodic_cell_tensor(jnp.asarray(coefficients, dtype=jnp.float64), s)
        return np.asarray(t), float(np.max(r))


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("aspect", [0.25, 4.0])
def test_periodic_box_of_aligned_layers_is_the_laminate(axis: int, aspect: float) -> None:
    """A box (a voxel of a graded grid) of layers normal to an axis: still Kottke's laminate along it, whatever
    the box's sides, also for anisotropic materials with off-diagonal coupling (which a wrong scaling of the
    tensors to the unit cube would change).
    """
    rng = np.random.default_rng(17 + axis)
    eps = np.stack([random_spd(rng) for _ in range(3)])
    layers = np.array([0, 2, 1, 1, 0, 2])
    shape = [2, 3, 2]
    shape[axis] = len(layers)
    coefficients = eps[np.broadcast_to(layers.reshape([-1 if a == axis else 1 for a in range(3)]), shape)]
    f = np.bincount(layers, minlength=3) / len(layers)
    for stretched in range(3):
        sides = np.ones(3)
        sides[stretched] = aspect
        out, residual = _cell64(coefficients, sides)
        assert rel(out, _laminate64(eps, f, np.eye(3)[axis])) < RTOL64, (axis, stretched)
        assert residual < 1e-10


def test_periodic_box_scale_symmetry_and_bounds() -> None:
    """The box's sides: only their ratios count, a cube is the unit cell, the result permutes with the axes,
    and stays within the Wiener bounds.
    """
    rng = np.random.default_rng(23)
    coefficients = np.stack([random_spd(rng, 1.0, 15.0) for _ in range(4 * 3 * 2)]).reshape(4, 3, 2, 3, 3)
    sides = np.array([1.0, 0.5, 3.0])
    out, residual = _cell64(coefficients, sides)
    assert residual < 1e-10
    assert rel(_cell64(coefficients, 7.0 * sides)[0], out) < RTOL64
    assert rel(_cell64(coefficients, np.ones(3))[0], _cell64(coefficients)[0]) < RTOL64
    assert np.abs(out - out.T).max() < 1e-12
    flat = coefficients.reshape(-1, 3, 3)
    assert np.linalg.eigvalsh(flat.mean(axis=0) - out).min() > -1e-12
    assert np.linalg.eigvalsh(out - np.linalg.inv(np.linalg.inv(flat).mean(axis=0))).min() > -1e-12
    # swap the x and y axes of the grid, the tensors and the sides
    p = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 1.0]])
    swapped = np.einsum("ij,abcjk,lk->bacil", p, coefficients, p)
    assert rel(_cell64(swapped, sides[[1, 0, 2]])[0], p @ out @ p.T) < RTOL64
    # stretching the box changes the result (the sides are used)
    assert rel(out, _cell64(coefficients)[0]) > 1e-3


def test_periodic_cell_of_one_material_is_that_material() -> None:
    rng = np.random.default_rng(6)
    e = random_spd(rng)
    out, residual = _cell64(np.broadcast_to(e, (3, 4, 2, 3, 3)))
    assert rel(out, e) < 1e-13 and residual < 1e-12


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_periodic_cell_of_aligned_layers_is_the_laminate(axis: int) -> None:
    """Layers on whole sub-cells normal to an axis: Q1 represents the layered potential exactly, so the cell
    problem gives Kottke's laminate (anisotropic materials with off-diagonal coupling).
    """
    rng = np.random.default_rng(7 + axis)
    eps = np.stack([random_spd(rng) for _ in range(3)])
    layers = np.array([0, 0, 1, 2, 2, 1, 0, 2])  # material of each layer
    shape = [2, 3, 2]
    shape[axis] = len(layers)
    idx = np.broadcast_to(layers.reshape([-1 if a == axis else 1 for a in range(3)]), shape)
    coefficients = eps[idx]
    out, residual = _cell64(coefficients)
    f = np.bincount(layers, minlength=3) / len(layers)
    assert rel(out, _laminate64(eps, f, np.eye(3)[axis])) < RTOL64
    assert residual < 1e-10


def test_periodic_cell_bounds_symmetry_and_grid_covariance() -> None:
    """Random sub-cell tensors: symmetric, within the Wiener bounds, and covariant under the grid's symmetries
    (swapping axes, mirroring one).
    """
    rng = np.random.default_rng(9)
    coefficients = np.stack([random_spd(rng, 1.0, 15.0) for _ in range(4 * 3 * 2)]).reshape(4, 3, 2, 3, 3)
    out, residual = _cell64(coefficients)
    assert residual < 1e-10
    assert np.abs(out - out.T).max() < 1e-12
    flat = coefficients.reshape(-1, 3, 3)
    arithmetic = flat.mean(axis=0)
    harmonic = np.linalg.inv(np.linalg.inv(flat).mean(axis=0))
    assert np.linalg.eigvalsh(arithmetic - out).min() > -1e-12
    assert np.linalg.eigvalsh(out - harmonic).min() > -1e-12
    # swap the x and y axes of the grid and of the tensors
    p = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 1.0]])
    swapped = np.einsum("ij,abcjk,lk->bacil", p, coefficients, p)
    assert rel(_cell64(swapped)[0], p @ out @ p.T) < RTOL64
    # mirror x: the grid reversed, x components negated
    m = np.diag([-1.0, 1.0, 1.0])
    mirrored = np.einsum("ij,abcjk,lk->abcil", m, coefficients[::-1], m)
    assert rel(_cell64(mirrored)[0], m @ out @ m.T) < RTOL64


def test_periodic_cell_gradients_match_finite_differences() -> None:
    rng = np.random.default_rng(10)
    base = np.stack([random_spd(rng) for _ in range(8)]).reshape(2, 2, 2, 3, 3)
    direction = np.stack([random_spd(rng) for _ in range(8)]).reshape(2, 2, 2, 3, 3)
    with enable_x64():

        def g(t):
            out, _ = bf.subpixel.periodic_cell_tensor(jnp.asarray(base) + t * jnp.asarray(direction))
            return jnp.sum(out * jnp.asarray([[1.0, 2, 3], [2, 5, 7], [3, 7, 11]]))

        grad = float(jax.grad(g)(0.0))
        h = 1e-6
        fd = float((g(h) - g(-h)) / (2 * h))
        assert abs(grad - fd) < 1e-7 * max(1.0, abs(fd))


# ---------------------------------------------------------------- classification


def test_principal_direction_and_junction_weight() -> None:
    """Rank one: the exact direction and omega 0. Two equal directions (a tie): omega 1, and finite gradients
    (the direction's derivative is cut there). A small second direction: a small omega.
    """
    with enable_x64():
        n = jnp.asarray([0.6, 0.0, 0.8])
        d = 2.0 * jnp.outer(n, n)
        v, value = bf.subpixel.principal_direction(d)
        assert abs(abs(float(v @ n)) - 1) < 1e-14 and abs(float(value) - 2.0) < 1e-13
        omega, normal = bf.subpixel.junction_weight(d, 0.02, 0.2)
        assert float(omega) == 0.0 and abs(abs(float(normal @ n)) - 1) < 1e-14
        tie = jnp.diag(jnp.asarray([1.0, 1.0, 0.0]))
        assert float(bf.subpixel.junction_weight(tie, 0.02, 0.2)[0]) == 1.0

        def loss(t):
            w, nrm = bf.subpixel.junction_weight(t * tie + jnp.diag(jnp.asarray([0.0, 0.0, 1e-9])), 0.02, 0.2)
            return w + jnp.sum(nrm**2)

        assert np.isfinite(float(jax.grad(loss)(1.0)))
        small = d + 0.01 * jnp.diag(jnp.asarray([0.0, 1.0, 0.0]))
        omega_small = float(bf.subpixel.junction_weight(small, 0.001, 0.2)[0])
        assert 0.0 < omega_small < 0.05


def test_no_interface_has_finite_zero_derivatives_in_float32() -> None:
    """A voxel without interfaces (d = 0) gets e_z and omega 0, and the derivatives of its normal, and of a
    laminate along it, are finite in float32 (the norm of the squared matrix must not vanish in the
    backward pass: 0 * inf there would make every gradient through a homogeneous voxel NaN)."""
    eps = jnp.asarray(np.stack([np.eye(3), 12.0 * np.eye(3)]), dtype=jnp.float32)
    zero = jnp.zeros((3, 3), dtype=jnp.float32)
    v, value = bf.subpixel.principal_direction(zero)
    assert np.allclose(np.asarray(v), [0.0, 0.0, 1.0]) and float(value) == 0.0
    jac = jax.jacobian(lambda d: bf.subpixel.junction_weight(d, 0.002, 0.02)[1])(zero)
    assert np.all(np.isfinite(np.asarray(jac)))

    def loss(d):
        _, normal = bf.subpixel.junction_weight(d, 0.002, 0.02)
        return jnp.sum(bf.subpixel.laminate(eps, jnp.asarray([1.0, 0.0], dtype=jnp.float32), normal))

    grad = np.asarray(jax.grad(loss)(zero))
    assert np.all(np.isfinite(grad)) and np.all(grad == 0.0)


def _check_laminate_guards(dtype: jnp.dtype) -> None:
    eps = jnp.asarray(np.stack([np.eye(3), 12.0 * np.eye(3)]), dtype=dtype)
    f_ref = jnp.asarray([0.4, 0.6], dtype=dtype)
    n_ez = jnp.asarray([0.0, 0.0, 1.0], dtype=dtype)
    ref_ez = bf.subpixel.laminate(eps, f_ref, n_ez)
    ref_ex = bf.subpixel.laminate(eps, f_ref, jnp.asarray([1.0, 0.0, 0.0], dtype=dtype))

    def eval_loss(f, n, e):
        return jnp.sum(bf.subpixel.laminate(e, f, n))

    grad_fn = jax.grad(eval_loss, argnums=(0, 1, 2))

    # Zero normal falls back cleanly to e_z with zero d/d(normal).
    n_zero = jnp.zeros(3, dtype=dtype)
    val_zero = bf.subpixel.laminate(eps, f_ref, n_zero)
    assert np.allclose(np.asarray(val_zero), np.asarray(ref_ez))
    gf, gn, ge = grad_fn(f_ref, n_zero, eps)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in (gf, gn, ge))
    assert np.all(np.asarray(gn) == 0.0)

    # Tiny nonzero normal where |n|^(-3) would overflow float32 stays exact and finite in AD.
    n_tiny = jnp.asarray([1e-13, 0.0, 0.0], dtype=dtype)
    val_tiny = bf.subpixel.laminate(eps, f_ref, n_tiny)
    assert np.allclose(np.asarray(val_tiny), np.asarray(ref_ex))
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grad_fn(f_ref, n_tiny, eps))

    # Sub-threshold normal falls back to e_z without overflow.
    n_sub = jnp.asarray([1e-20, 0.0, 0.0], dtype=dtype)
    assert np.all(np.isfinite(np.asarray(bf.subpixel.laminate(eps, f_ref, n_sub))))
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grad_fn(f_ref, n_sub, eps))

    # All-zero fractions fall back to identity with finite zero gradients, while batched neighbors keep exact grads.
    f_batch = jnp.asarray([[0.0, 0.0], [0.4, 0.6]], dtype=dtype)
    n_batch = jnp.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype)
    out_batch = bf.subpixel.laminate(eps, f_batch, n_batch)
    assert np.allclose(np.asarray(out_batch[0]), np.eye(3))
    assert np.allclose(np.asarray(out_batch[1]), np.asarray(ref_ex))
    gfb, gnb, geb = jax.grad(lambda f, n, e: jnp.sum(bf.subpixel.laminate(e, f, n)), argnums=(0, 1, 2))(
        f_batch, n_batch, eps
    )
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in (gfb, gnb, geb))
    gf_ref, _, _ = grad_fn(f_ref, jnp.asarray([1.0, 0.0, 0.0], dtype=dtype), eps)
    assert np.allclose(np.asarray(gfb[1]), np.asarray(gf_ref), rtol=1e-5, atol=1e-6)


def test_laminate_handles_zero_fractions_and_tiny_normals_in_float32_and_float64() -> None:
    _check_laminate_guards(jnp.float32)
    with enable_x64():
        _check_laminate_guards(jnp.float64)


def _check_periodic_plasmonic_and_jvp(dtype: jnp.dtype) -> None:
    eye3 = np.eye(3)
    # 2x1x1 resonant cell with eps1 + eps2 == 0 -> singular stiffness K == 0.
    coeffs_2x1 = np.zeros((2, 1, 1, 3, 3))
    coeffs_2x1[0, 0, 0] = -2.0 * eye3
    coeffs_2x1[1, 0, 0] = 2.0 * eye3
    # 2x2x2 indefinite plasmonic cell with half -2*I and half +2*I.
    coeffs_2x2 = np.zeros((2, 2, 2, 3, 3))
    coeffs_2x2[0] = -2.0 * eye3
    coeffs_2x2[1] = 2.0 * eye3

    for arr in (coeffs_2x1, coeffs_2x2):
        c = jnp.asarray(arr, dtype=dtype)
        for sides in (None, jnp.asarray([1.0, 2.0, 3.0], dtype=dtype)):
            tensor, res = bf.subpixel.periodic_cell_tensor(c, sides=sides)
            t_np = np.asarray(tensor)
            assert np.all(np.isfinite(t_np))
            assert np.allclose(t_np, t_np.T)
            assert np.all(np.linalg.eigvalsh(t_np) > 0.0)
            assert bool(jnp.isnan(res))

            grad_c = jax.grad(lambda x, s=sides: jnp.sum(bf.subpixel.periodic_cell_tensor(x, sides=s)[0]))(c)
            assert bool(jnp.all(jnp.isfinite(grad_c)))
            tangent = jnp.ones_like(c)
            primal, dt = jax.jvp(
                lambda x, s=sides: bf.subpixel.periodic_cell_tensor(x, sides=s)[0], (c,), (tangent,)
            )
            assert bool(jnp.all(jnp.isfinite(primal))) and bool(jnp.all(jnp.isfinite(dt)))

    # Mixed batch under vmap: SPD cell 0 and singular plasmonic cell 1 stay isolated in primal and grad.
    rng = np.random.default_rng(7)
    spd_cell = np.empty((2, 2, 2, 3, 3))
    a_spd, b_spd = random_spd(rng, 1.5, 4.0), random_spd(rng, 6.0, 12.0)
    spd_cell[0] = a_spd
    spd_cell[1] = b_spd
    batch = jnp.asarray(np.stack([spd_cell, coeffs_2x2]), dtype=dtype)

    vmap_fn = jax.vmap(bf.subpixel.periodic_cell_tensor)
    t_b, res_b = vmap_fn(batch)
    t0_single, res0_single = bf.subpixel.periodic_cell_tensor(batch[0])
    assert np.allclose(np.asarray(t_b[0]), np.asarray(t0_single), rtol=1e-5, atol=1e-6)
    assert float(res_b[0]) == pytest.approx(float(res0_single), rel=1e-5, abs=1e-6)
    assert bool(jnp.isnan(res_b[1])) and bool(jnp.all(jnp.isfinite(t_b[1])))

    g_b = jax.grad(lambda b: jnp.sum(vmap_fn(b)[0]))(batch)
    g0_single = jax.grad(lambda x: jnp.sum(bf.subpixel.periodic_cell_tensor(x)[0]))(batch[0])
    assert bool(jnp.all(jnp.isfinite(g_b)))
    assert np.allclose(np.asarray(g_b[0]), np.asarray(g0_single), rtol=1e-4, atol=1e-6)

    # Implicit custom_jvp matches jax.grad directional derivative on the SPD cell.
    direction = jnp.asarray(rng.normal(size=batch[0].shape), dtype=dtype)
    _, jvp_val = jax.jvp(lambda x: jnp.sum(bf.subpixel.periodic_cell_tensor(x)[0]), (batch[0],), (direction,))
    assert float(jvp_val) == pytest.approx(float(jnp.sum(g0_single * direction)), rel=1e-4, abs=1e-5)


def test_periodic_cell_tensor_plasmonic_resonance_and_gradients() -> None:
    _check_periodic_plasmonic_and_jvp(jnp.float32)
    with enable_x64():
        _check_periodic_plasmonic_and_jvp(jnp.float64)
