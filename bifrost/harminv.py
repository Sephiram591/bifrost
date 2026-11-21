"""JAX-differentiable harmonic inversion (Filter Diagonalization Method) for ``fdtdx.FieldDetector``.

Extracts complex resonant frequencies, decay rates, quality factors, wavelengths,
and complex modal amplitudes from uniformly sampled time-domain signals (or
``fdtdx.FieldDetector`` recordings) using Mandelshtam & Taylor's Filter Diagonalization
Method (FDM) combined with narrow-band Fourier-projected Vandermonde amplitude recovery.

Conventions match ``fdtdx`` and Meep/``harminv``:

.. math::
    y(t_0 + n\\,\\Delta t) = \\sum_{k} a_k e^{-i \\omega_k n\\,\\Delta t},
    \\qquad \\omega_k = 2\\pi f_k - i \\gamma_k,
    \\qquad Q_k = \\frac{\\operatorname{Re}(\\omega_k)}{-2\\operatorname{Im}(\\omega_k)} = \\frac{\\pi f_k}{\\gamma_k}.

For real-valued signals with positive frequency bands (``f_min > 0``), each mode
is implicitly paired with its negative-frequency conjugate partner so that
``y(t_0 + n dt) = 2 * Re(sum_k a_k exp(-i omega_k n dt))`` (for example,
``A cos(2 pi f t + phi) exp(-gamma t)`` yields ``|a| = A / 2``).
"""

from __future__ import annotations

import functools
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import fdtdx
import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "HarminvResult",
    "detector_time_grid",
    "harminv",
    "harminv_detector",
]

_CHUNK_CHANNELS = 2048
_GOLDEN_ANGLE = 2.0 * math.pi / (((1.0 + math.sqrt(5.0)) / 2.0) ** 2)


def _abs_sq(z: jax.Array) -> jax.Array:
    """Squared magnitude ``|z|^2`` safe to differentiate at ``z == 0``."""
    if jnp.iscomplexobj(z):
        re = jnp.real(z)
        im = jnp.imag(z)
        return re * re + im * im
    return z * z


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "frequency",
        "decay_rate",
        "Q",
        "omega",
        "wavelength",
        "amplitude",
        "error",
        "valid",
        "dt",
        "t0",
    ],
    meta_fields=["is_real_signal", "paired_conjugate"],
)
@dataclass(frozen=True)
class HarminvResult:
    """Fixed-shape harmonic inversion output compatible with ``jax.jit`` and ``jax.grad``.

    Modes are sorted with all ``valid == True`` entries first. Unused slots up to
    ``K = frequency.shape[0]`` are zero-filled with ``valid == False``.

    Attributes:
        frequency: Resonant frequencies ``f_k`` in Hz, shape ``(K,)``.
        decay_rate: Amplitude decay rates ``gamma_k = -Im(omega_k)`` in ``1/s``
            (positive for decaying modes), shape ``(K,)``.
        Q: Quality factors ``pi * frequency / decay_rate``, shape ``(K,)``.
        omega: Complex angular frequencies ``2*pi*f_k - 1j*gamma_k`` in rad/s, shape ``(K,)``.
        wavelength: Free-space wavelengths ``c / f_k`` in metres, shape ``(K,)``.
        amplitude: Complex mode amplitudes referenced to ``t0``, shape ``(K, *channel_shape)``.
        error: Relative pole residual error from the shifted pencil ``U2``, shape ``(K,)``.
        valid: Boolean mask identifying physical modes passing all filter thresholds, shape ``(K,)``.
        dt: Effective sample spacing in seconds (scalar array).
        t0: Simulation time of the first sample in seconds (scalar array).
        is_real_signal: Static metadata; ``True`` when the input signal was real-valued.
        paired_conjugate: Static metadata; ``True`` when positive-band amplitudes represent
            real-signal conjugate pairs ``2 * Re(a_k exp(-i omega_k t))``.
    """

    frequency: jax.Array
    decay_rate: jax.Array
    Q: jax.Array
    omega: jax.Array
    wavelength: jax.Array
    amplitude: jax.Array
    error: jax.Array
    valid: jax.Array
    dt: jax.Array
    t0: jax.Array
    is_real_signal: bool
    paired_conjugate: bool

    @property
    def inv_Q(self) -> jax.Array:
        """Inverse quality factors ``1 / Q_k = gamma_k / (pi * f_k)`` (smooth as ``gamma_k -> 0``)."""
        safe_f = jnp.where(jnp.abs(self.frequency) > 1e-30, self.frequency, 1.0)
        raw = self.decay_rate / (math.pi * safe_f)
        return jnp.where(self.valid & (jnp.abs(self.frequency) > 1e-30), raw, 0.0)

    def select(self, target_frequency: float | fdtdx.WaveCharacter) -> dict[str, jax.Array]:
        """Select the valid mode closest to ``target_frequency`` in a differentiable way.

        Uses ``stop_gradient`` on the index lookup so gradients flow cleanly through
        the selected mode's frequency, decay rate, Q, and amplitude.

        Args:
            target_frequency: Target frequency in Hz or an ``fdtdx.WaveCharacter``.

        Returns:
            Dict with keys ``"frequency"``, ``"decay_rate"``, ``"Q"``, ``"inv_Q"``,
            ``"omega"``, ``"wavelength"``, ``"amplitude"``, ``"error"``, and ``"valid"``.
        """
        f0 = _coerce_frequency(target_frequency)
        dist = jnp.where(self.valid, jnp.abs(self.frequency - f0), jnp.inf)
        idx = jax.lax.stop_gradient(jnp.argmin(dist))
        return {
            "frequency": self.frequency[idx],
            "decay_rate": self.decay_rate[idx],
            "Q": self.Q[idx],
            "inv_Q": self.inv_Q[idx],
            "omega": self.omega[idx],
            "wavelength": self.wavelength[idx],
            "amplitude": self.amplitude[idx],
            "error": self.error[idx],
            "valid": self.valid[idx],
        }

    def reconstruct(
        self,
        num_steps: int | None = None,
        *,
        time: jax.Array | None = None,
    ) -> jax.Array:
        """Reconstruct the time-domain signal from the valid extracted modes.

        Args:
            num_steps: Number of uniformly spaced steps ``0 .. num_steps - 1`` starting
                at ``t0`` with step ``dt``. Mutually exclusive with ``time``.
            time: Explicit 1D array of simulation times (seconds) at which to evaluate
                the reconstructed signal (uses ``tau = time - t0``).

        Returns:
            Reconstructed signal of shape ``(T, *channel_shape)``, real if
            ``self.is_real_signal`` else complex.
        """
        if (num_steps is None) == (time is None):
            raise ValueError("Provide exactly one of 'num_steps' or 'time' to reconstruct().")
        if time is not None:
            tau = jnp.asarray(time) - self.t0
        else:
            assert num_steps is not None
            tau = jnp.arange(num_steps, dtype=jnp.result_type(self.dt)) * self.dt
        if tau.ndim != 1:
            raise ValueError(f"'time' must be a 1D array, got shape {tau.shape}.")

        # Mask omega and amplitude BEFORE exponentiating so unused slots never overflow.
        omega_safe = jnp.where(self.valid, self.omega, 0.0 + 0.0j)
        amp_mask = self.valid[(slice(None),) + (None,) * (self.amplitude.ndim - 1)]
        amp_safe = jnp.where(amp_mask, self.amplitude, 0.0 + 0.0j)

        basis = jnp.exp(-1j * tau[:, None] * omega_safe[None, :])  # (T, K)
        amp_flat = amp_safe.reshape((amp_safe.shape[0], -1))  # (K, M_ch)
        series = (basis @ amp_flat).reshape((tau.shape[0], *self.amplitude.shape[1:]))
        if self.paired_conjugate:
            return 2.0 * jnp.real(series)
        if self.is_real_signal:
            return jnp.real(series)
        return series

    def to_modes(self) -> list[dict[str, Any]]:
        """Return concrete Python dictionaries for each valid mode (eager/non-jitted)."""
        valid_np = np.asarray(self.valid)
        freq_np = np.asarray(self.frequency)
        decay_np = np.asarray(self.decay_rate)
        q_np = np.asarray(self.Q)
        omega_np = np.asarray(self.omega)
        wl_np = np.asarray(self.wavelength)
        amp_np = np.asarray(self.amplitude)
        err_np = np.asarray(self.error)
        modes = []
        for idx in np.flatnonzero(valid_np):
            amp_val = complex(amp_np[idx]) if amp_np.ndim == 1 else amp_np[idx]
            modes.append(
                {
                    "frequency": float(freq_np[idx]),
                    "decay_rate": float(decay_np[idx]),
                    "Q": float(q_np[idx]),
                    "omega": complex(omega_np[idx]),
                    "wavelength": float(wl_np[idx]),
                    "amplitude": amp_val,
                    "error": float(err_np[idx]),
                }
            )
        return modes


def _coerce_frequency(value: float | fdtdx.WaveCharacter) -> float:
    if isinstance(value, fdtdx.WaveCharacter):
        return float(value.get_frequency())
    return float(value)


def _resolve_freq_bounds(
    f_min: float | fdtdx.WaveCharacter,
    f_max: float | fdtdx.WaveCharacter,
) -> tuple[float, float]:
    both_wave = isinstance(f_min, fdtdx.WaveCharacter) and isinstance(f_max, fdtdx.WaveCharacter)
    f1 = _coerce_frequency(f_min)
    f2 = _coerce_frequency(f_max)
    if not (math.isfinite(f1) and math.isfinite(f2)):
        raise ValueError(f"Frequency bounds must be finite, got ({f1}, {f2}).")
    if both_wave:
        f_low, f_high = (f1, f2) if f1 <= f2 else (f2, f1)
    else:
        f_low, f_high = f1, f2
    if not (f_low < f_high):
        raise ValueError(f"Require f_min < f_max (in Hz), got f_min={f_low}, f_max={f_high}.")
    return f_low, f_high


def detector_time_grid(detector: fdtdx.FieldDetector) -> tuple[float, float]:
    """Extract uniform ``(dt, t0)`` in seconds from a placed ``fdtdx.FieldDetector``.

    Uses static detector placement configuration (``_calculate_on_list()``) rather than
    dynamic JAX arrays so it remains safe under closure-captured ``bf.jit``.

    Args:
        detector: Placed forward ``fdtdx.FieldDetector``.

    Returns:
        ``(dt, t0)`` where ``dt`` is the effective interval between recorded samples (seconds)
        and ``t0`` is the simulation time of the first recorded sample (seconds).
    """
    if not isinstance(detector, fdtdx.FieldDetector):
        raise TypeError(f"Expected fdtdx.FieldDetector, got {type(detector).__name__}.")
    if detector.inverse:
        raise ValueError(f"FieldDetector '{detector.name}' has inverse=True, which is not supported.")
    if detector._config is None or not hasattr(detector._config, "time_steps_total"):
        raise ValueError(
            f"FieldDetector '{detector.name}' is not placed yet; run fdtdx.place_objects(...) first."
        )
    try:
        dt_sim = float(detector._config.time_step_duration)
        total_steps = int(detector._config.time_steps_total)
        on_list = detector._calculate_on_list()
    except jax.errors.ConcretizationTypeError as exc:
        raise ValueError(
            f"FieldDetector '{detector.name}' has traced config fields inside jit; "
            "extract (dt, t0) outside jit or capture the placed detector by closure."
        ) from exc

    if total_steps <= 0 or dt_sim <= 0.0:
        raise ValueError(f"Invalid simulation time configuration: time_steps_total={total_steps}, dt={dt_sim}.")

    active = [t for t, is_on in enumerate(on_list) if is_on]
    if len(active) < 5:
        raise ValueError(
            f"FieldDetector '{detector.name}' records only {len(active)} time steps; need at least 5."
        )
    strides = {b - a for a, b in zip(active, active[1:])}
    if len(strides) != 1:
        raise ValueError(
            f"FieldDetector '{detector.name}' has non-uniform sampling strides {sorted(strides)}; "
            "harmonic inversion requires uniformly spaced samples."
        )
    stride = next(iter(strides))
    dt = stride * dt_sim
    t0 = active[0] * dt_sim
    return dt, t0


def _make_projection_weights(
    num_channels: int,
    real_dtype: jnp.dtype,
    channel_scales: jax.Array | None,
) -> tuple[jax.Array | None, int]:
    """Build static orthonormal channel projection matrix ``(M_ch, P)`` or ``None`` when ``M_ch <= 4``."""
    if num_channels <= 4:
        return None, num_channels
    num_proj = 4
    idx = jnp.arange(1, num_channels + 1, dtype=real_dtype)
    cols = jnp.stack(
        [
            jnp.ones(num_channels, dtype=real_dtype),
            jnp.cos(idx * _GOLDEN_ANGLE),
            jnp.sin(idx * _GOLDEN_ANGLE),
            jnp.cos(2.0 * idx * _GOLDEN_ANGLE),
        ],
        axis=1,
    )
    q_cols, _ = jnp.linalg.qr(cols, mode="reduced")
    if channel_scales is not None:
        q_cols = channel_scales[:, None] * q_cols
    return q_cols, num_proj


def _project_channels(
    y_mat: jax.Array,
    w_proj: jax.Array | None,
    channel_scales: jax.Array | None,
    real_dtype: jnp.dtype,
    complex_dtype: jnp.dtype,
) -> jax.Array:
    """Project ``(N, M_ch)`` signal onto ``(N, P)`` traces without promoting full large tensors."""
    if w_proj is None:
        c_proj = y_mat if channel_scales is None else y_mat * channel_scales[None, :]
        return c_proj.astype(complex_dtype)

    n_steps, m_ch = y_mat.shape
    target_dtype = complex_dtype if jnp.iscomplexobj(y_mat) or jnp.iscomplexobj(w_proj) else real_dtype
    if m_ch <= _CHUNK_CHANNELS:
        return (y_mat.astype(target_dtype) @ w_proj.astype(target_dtype)).astype(complex_dtype)

    num_chunks = (m_ch + _CHUNK_CHANNELS - 1) // _CHUNK_CHANNELS
    pad_ch = num_chunks * _CHUNK_CHANNELS - m_ch
    w_pad = w_proj.astype(target_dtype) if pad_ch == 0 else jnp.pad(w_proj.astype(target_dtype), ((0, pad_ch), (0, 0)))
    w_chunks = w_pad.reshape((num_chunks, _CHUNK_CHANNELS, w_proj.shape[1]))
    y_padded = y_mat if pad_ch == 0 else jnp.pad(y_mat, ((0, 0), (0, pad_ch)))

    def body(acc: jax.Array, idx: jax.Array) -> tuple[jax.Array, None]:
        start = idx * _CHUNK_CHANNELS
        y_blk = jax.lax.dynamic_slice(
            y_padded,
            (0, start),
            (n_steps, _CHUNK_CHANNELS),
        ).astype(target_dtype)
        return acc + y_blk @ w_chunks[idx], None

    init = jnp.zeros((n_steps, w_proj.shape[1]), dtype=target_dtype)
    out, _ = jax.lax.scan(body, init, jnp.arange(num_chunks))
    return out.astype(complex_dtype)


def _narrowband_dft(
    y_mat: jax.Array,
    phi_dft: jax.Array,
    real_dtype: jnp.dtype,
    complex_dtype: jnp.dtype,
) -> jax.Array:
    """Compute ``D = exp(1j * phi_dft[:, None] * n[None, :]) @ Y`` of shape ``(J_f, M_ch)``."""
    n_steps, m_ch = y_mat.shape
    n_idx = jnp.arange(n_steps, dtype=real_dtype)
    angles = phi_dft[:, None] * n_idx[None, :]
    cos_mat = jnp.cos(angles)
    sin_mat = jnp.sin(angles)
    is_complex_y = jnp.iscomplexobj(y_mat)

    def _dft_block(y_blk: jax.Array) -> jax.Array:
        if not is_complex_y:
            y_r = y_blk.astype(real_dtype)
            return jax.lax.complex(cos_mat @ y_r, sin_mat @ y_r)
        y_c = y_blk.astype(complex_dtype)
        return jax.lax.complex(cos_mat, sin_mat) @ y_c

    if m_ch <= _CHUNK_CHANNELS:
        return _dft_block(y_mat)

    num_chunks = (m_ch + _CHUNK_CHANNELS - 1) // _CHUNK_CHANNELS
    pad_ch = num_chunks * _CHUNK_CHANNELS - m_ch
    j_f = phi_dft.shape[0]
    y_padded = y_mat if pad_ch == 0 else jnp.pad(y_mat, ((0, 0), (0, pad_ch)))

    def body(_: None, idx: jax.Array) -> tuple[None, jax.Array]:
        y_blk = jax.lax.dynamic_slice(y_padded, (0, idx * _CHUNK_CHANNELS), (n_steps, _CHUNK_CHANNELS))
        return None, _dft_block(y_blk)

    _, blocks = jax.lax.scan(body, None, jnp.arange(num_chunks))  # (num_chunks, J_f, _CHUNK_CHANNELS)
    d_full = jnp.transpose(blocks, (1, 0, 2)).reshape((j_f, num_chunks * _CHUNK_CHANNELS))
    return d_full[:, :m_ch]


def _build_fdm_matrices(
    c_norm: jax.Array,
    phi_basis: jax.Array,
    real_dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Construct stacked ``U0, U1, U2`` of shape ``(P * J, J)`` in ``O(P * N * J + P * J^2)``."""
    n_steps, num_proj = c_norm.shape
    half_m = (n_steps - 3) // 2
    num_basis = phi_basis.shape[0]

    n_all = jnp.arange(2 * half_m + 1, dtype=real_dtype)
    e_all = jnp.exp(1j * phi_basis[:, None] * n_all[None, :])  # (J, 2M+1)
    z = jnp.exp(1j * phi_basis)  # (J,)
    e1 = e_all[:, : half_m + 1] * z[:, None]  # sum_{s=0}^M c_s z^{s+1}
    e2 = e_all[:, 1 : half_m + 1]  # sum_{s=M+1}^{2M} c_s z^{s-M}, empty if M == 0
    z_m1 = jnp.exp(1j * (half_m + 1) * phi_basis)  # (J,)

    s1 = jnp.arange(half_m + 1, dtype=real_dtype)
    s2 = jnp.arange(half_m + 1, 2 * half_m + 1, dtype=real_dtype)
    tri_weights = jnp.concatenate([s1 + 1.0, 2.0 * half_m + 1.0 - s2])  # (2M+1,)

    eye = jnp.eye(num_basis, dtype=jnp.bool_)
    safe_denom = jnp.where(eye, 1.0 + 0.0j, z[:, None] - z[None, :])  # (J, J)
    phase_half = jnp.exp(-0.5j * half_m * phi_basis)  # z^{-M/2}, (J,)
    phase_mat = phase_half[:, None] * phase_half[None, :]  # (J, J)

    u_mats = []
    for p in (0, 1, 2):
        c_p = c_norm[p : p + 2 * half_m + 1, :]  # (2M+1, P)
        f1 = (e1 @ c_p[: half_m + 1, :]).T  # (P, J)
        f2 = (e2 @ c_p[half_m + 1 :, :]).T  # (P, J)
        diag_vals = (e_all @ (tri_weights[:, None] * c_p)).T  # (P, J)

        num = (
            f1[:, :, None]
            - f1[:, None, :]
            + z_m1[None, :, None] * f2[:, None, :]
            - z_m1[None, None, :] * f2[:, :, None]
        )
        off_diag = num / safe_denom[None, :, :]
        u_p = jnp.where(eye[None, :, :], diag_vals[:, :, None], off_diag) / (half_m + 1.0)
        u_p = u_p * phase_mat[None, :, :]
        u_mats.append(u_p.reshape((num_proj * num_basis, num_basis)))

    return u_mats[0], u_mats[1], u_mats[2]


def _fdm_primal_decompose(
    u0: jax.Array,
    u1: jax.Array,
    u2: jax.Array,
    eff_svd_tol: float,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Run SVD truncation and right-eigenvector generalized solve on stacked ``U0, U1, U2``."""
    num_basis = u0.shape[1]
    real_dtype = jnp.finfo(u0.dtype).dtype
    complex_dtype = u0.dtype

    v_mat, sigma, wh_mat = jnp.linalg.svd(u0, full_matrices=False)
    vh_mat = jnp.conj(v_mat.T)
    w_mat = jnp.conj(wh_mat.T)

    active = (sigma > eff_svd_tol * sigma[0]) & (sigma[0] > 0)
    safe_sigma = jnp.where(active, sigma, 1.0)
    inv_sqrt_s = jnp.where(active, 1.0 / jnp.sqrt(safe_sigma), 0.0)
    inv_s = inv_sqrt_s * inv_sqrt_s

    # Distinct dummy diagonal entries far outside the unit circle so inactive slots stay decoupled.
    dummy = jnp.where(active, 0.0, -(2.0 + 0.1 * jnp.arange(num_basis, dtype=real_dtype))).astype(complex_dtype)

    u1_core = vh_mat @ u1 @ w_mat
    u1_tilde = (inv_sqrt_s[:, None] * u1_core * inv_sqrt_s[None, :]) + jnp.diag(dummy)

    u_eig, x_mat = jnp.linalg.eig(u1_tilde)

    active_w = jnp.sum(jnp.abs(x_mat * active[:, None]) ** 2, axis=0)
    inactive_w = jnp.sum(jnp.abs(x_mat * (~active)[:, None]) ** 2, axis=0)
    is_mode = (active_w > inactive_w) & (jnp.abs(u_eig) > 1e-12)

    u2_tilde = inv_sqrt_s[:, None] * (vh_mat @ u2 @ w_mat) * inv_sqrt_s[None, :]
    res2 = u2_tilde @ x_mat - x_mat * (u_eig * u_eig)[None, :]
    # Normalize error by 2 * |u|^2 * max(|log(u)|, 1e-6) * ||x|| to match harminv's fractional frequency error.
    u_safe_err = jnp.where(jnp.abs(u_eig) > 1e-30, u_eig, 1.0 + 0.0j)
    log_mag = jnp.maximum(jnp.abs(jnp.log(u_safe_err)), 1e-6)
    denom_err = jnp.maximum(
        2.0 * log_mag * jnp.maximum(jnp.abs(u_eig) ** 2, 1e-30) * jnp.linalg.norm(x_mat, axis=0),
        1e-30,
    )
    err = jnp.where(is_mode, jnp.linalg.norm(res2, axis=0) / denom_err, jnp.inf)

    return u_eig, err, is_mode, x_mat, v_mat, vh_mat, w_mat, wh_mat, active, inv_sqrt_s, inv_s, sigma


@functools.partial(jax.custom_jvp, nondiff_argnums=(3,))
def _solve_fdm_poles(
    u0: jax.Array,
    u1: jax.Array,
    u2: jax.Array,
    eff_svd_tol: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Solve ``U1 b_k = u_k U0 b_k`` with rank truncation and GPU-compatible right-only ``eig``."""
    u_eig, err, is_mode, *_ = _fdm_primal_decompose(u0, u1, u2, eff_svd_tol)
    real_dtype = jnp.finfo(u0.dtype).dtype
    return u_eig, err.astype(real_dtype), is_mode.astype(real_dtype)


@_solve_fdm_poles.defjvp
def _solve_fdm_poles_jvp(
    eff_svd_tol: float,
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
    """Exact truncated-SVD subspace JVP for non-degenerate retained FDM poles."""
    u0, u1, u2 = primals
    du0, du1, _du2 = tangents
    num_basis = u0.shape[1]
    real_dtype = jnp.finfo(u0.dtype).dtype
    complex_dtype = u0.dtype

    u_eig, err, is_mode, x_mat, v_mat, vh_mat, w_mat, wh_mat, active, inv_sqrt_s, inv_s, sigma = (
        _fdm_primal_decompose(u0, u1, u2, eff_svd_tol)
    )

    # Plain LU solve avoids squaring condition number of X; sanitize in case of exact degeneracy.
    eye_j = jnp.eye(num_basis, dtype=complex_dtype)
    y_t = jnp.linalg.solve(x_mat.T, eye_j).T
    y_t = jnp.where(jnp.isfinite(y_t), y_t, 0.0 + 0.0j)

    r_mat = (w_mat * inv_sqrt_s[None, :]) @ x_mat  # (J, J)
    lt_mat = y_t @ (inv_sqrt_s[:, None] * vh_mat)  # (J, P*J)
    du0_h = jnp.conj(du0.T)  # (J, P*J)

    # 1. Active-block contribution (all intra-active SVD rotations cancel identically into -u_eig * l^T dU0 r).
    term1 = jnp.sum(lt_mat.T * (du1 @ r_mat), axis=0) - u_eig * jnp.sum(lt_mat.T * (du0 @ r_mat), axis=0)

    # 2. Orthogonal complement (I - V V^H) left-subspace rotation when P > 1 (M = P*J > J).
    inv_s32 = inv_s * inv_sqrt_s
    a_perp = y_t @ (inv_s32[:, None] * wh_mat)  # (J, J)
    u1_r = u1 @ r_mat  # (P*J, J)
    vh_u1_r = vh_mat @ u1_r  # (J, J)
    b_perp = u1_r - v_mat @ vh_u1_r  # (P*J, J)
    term_perp = jnp.sum(a_perp.T * (du0_h @ b_perp), axis=0)

    # 3. Exact SVD subspace rotations between active (i in A) and truncated (j in I) singular vectors.
    p_mat = vh_mat @ du0 @ w_mat  # (J, J)
    p_dag = jnp.conj(p_mat.T)  # (J, J)
    cross_ai = active[:, None] & (~active)[None, :]  # (J, J), True for (i in A, j in I)
    s_col = sigma[:, None]
    s_row = sigma[None, :]
    diff_sq = s_col * s_col - s_row * s_row
    min_gap = 1e-12 * jnp.maximum(s_col * s_col, 1e-30)
    use_exact_gap = cross_ai & (jnp.abs(diff_sq) > min_gap)
    use_pinv_fallback = cross_ai & (~use_exact_gap)
    safe_diff_sq = jnp.where(use_exact_gap, diff_sq, 1.0)

    omega_v_ai = jnp.where(
        use_exact_gap,
        -(s_row * p_mat + s_col * p_dag) / safe_diff_sq,
        jnp.where(use_pinv_fallback, -inv_s[:, None] * p_dag, 0.0 + 0.0j),
    )
    omega_w_ia = jnp.where(
        use_exact_gap,
        (s_row * p_mat.T + s_col * jnp.conj(p_mat)) / safe_diff_sq,
        jnp.where(use_pinv_fallback, inv_s[:, None] * jnp.conj(p_mat), 0.0 + 0.0j),
    ).T

    y_s = y_t * inv_sqrt_s[None, :]  # (J, J)
    term_v_ai = -jnp.sum(y_s.T * (omega_v_ai @ vh_u1_r), axis=0)

    lt_u1_w = (lt_mat @ u1) @ w_mat  # (J, J)
    s_x = inv_sqrt_s[:, None] * x_mat  # (J, J)
    term_w_ia = jnp.sum(lt_u1_w.T * (omega_w_ia @ s_x), axis=0)

    du = jnp.where(is_mode, term1 + term_perp + term_v_ai + term_w_ia, 0.0 + 0.0j)
    zero_r = jnp.zeros_like(err, dtype=real_dtype)
    return (u_eig, err.astype(real_dtype), is_mode.astype(real_dtype)), (du, zero_r, zero_r)


def _geom_sum(w: jax.Array, n_steps: int) -> jax.Array:
    """Differentiable ``sum_{n=0}^{N-1} w^n`` with Taylor fallback near ``w = 1``."""
    delta = 1.0 - w
    near_one = jnp.abs(delta) < 1e-6
    safe_delta = jnp.where(near_one, 1.0 + 0.0j, delta)
    w_safe = jnp.where(near_one, 1.0 + 0.0j, w)
    exact = (1.0 - w_safe**n_steps) / safe_delta

    # (1 - (1 - d)^N) / d = N - N(N-1)/2 d + N(N-1)(N-2)/6 d^2 - N(N-1)(N-2)(N-3)/24 d^3
    n_f = float(n_steps)
    c1 = 0.5 * n_f * (n_f - 1.0)
    c2 = n_f * (n_f - 1.0) * (n_f - 2.0) / 6.0
    c3 = n_f * (n_f - 1.0) * (n_f - 2.0) * (n_f - 3.0) / 24.0
    taylor = n_f - delta * (c1 - delta * (c2 - delta * c3))
    return jnp.where(near_one, taylor, exact)


def _solve_amplitudes(
    dft_data: jax.Array,
    phi_dft: jax.Array,
    u_k: jax.Array,
    pre_valid: jax.Array,
    n_steps: int,
    paired_conjugate: bool,
    real_dtype: jnp.dtype,
    complex_dtype: jnp.dtype,
) -> jax.Array:
    """Solve for complex amplitudes ``(K, M_ch)`` over narrow-band DFT projections via QR."""
    num_modes = u_k.shape[0]
    rot = jnp.exp(1j * phi_dft[:, None])  # (J_f, 1)
    g_pos = _geom_sum(rot * u_k[None, :], n_steps) * pre_valid[None, :]  # (J_f, K)
    tiny_reg = 1e-14 if real_dtype == jnp.float64 else 1e-7

    if paired_conjugate:
        g_neg = _geom_sum(rot * jnp.conj(u_k)[None, :], n_steps) * pre_valid[None, :]  # (J_f, K)
        g_sum = g_pos + g_neg
        g_diff = 1j * (g_pos - g_neg)
        b_top = jnp.concatenate([jnp.real(g_sum), jnp.real(g_diff)], axis=1)  # (J_f, 2K)
        b_bot = jnp.concatenate([jnp.imag(g_sum), jnp.imag(g_diff)], axis=1)  # (J_f, 2K)
        b_mat = jnp.concatenate([b_top, b_bot], axis=0)  # (2*J_f, 2K)
        d_real = jnp.concatenate([jnp.real(dft_data), jnp.imag(dft_data)], axis=0).astype(real_dtype)  # (2*J_f, M_ch)

        mask2 = jnp.concatenate([pre_valid, pre_valid])
        b_masked = jnp.where(mask2[None, :], b_mat, 0.0)
        b_scale = jax.lax.stop_gradient(jnp.maximum(jnp.max(jnp.abs(b_masked)), 1.0))
        reg_diag = jnp.where(mask2, tiny_reg * b_scale, 1.0)
        a_aug = jnp.concatenate([b_masked, jnp.diag(reg_diag)], axis=0)
        d_aug = jnp.concatenate([d_real, jnp.zeros((2 * num_modes, d_real.shape[1]), dtype=real_dtype)], axis=0)
        q_mat, r_mat = jnp.linalg.qr(a_aug, mode="reduced")
        sol = jnp.linalg.solve(r_mat, q_mat.T @ d_aug)
        amps = jax.lax.complex(sol[:num_modes], sol[num_modes:])
        return amps * pre_valid[:, None]

    g_masked = jnp.where(pre_valid[None, :], g_pos, 0.0 + 0.0j)
    g_scale = jax.lax.stop_gradient(jnp.maximum(jnp.max(jnp.abs(g_masked)), 1.0))
    reg_c = jnp.where(pre_valid, tiny_reg * g_scale, 1.0).astype(complex_dtype)
    a_aug_c = jnp.concatenate([g_masked, jnp.diag(reg_c)], axis=0)
    d_aug_c = jnp.concatenate([dft_data, jnp.zeros((num_modes, dft_data.shape[1]), dtype=complex_dtype)], axis=0)
    q_c, r_c = jnp.linalg.qr(a_aug_c, mode="reduced")
    amps_c = jnp.linalg.solve(r_c, jnp.conj(q_c.T) @ d_aug_c)
    return amps_c * pre_valid[:, None]


def harminv(
    signal: jax.Array,
    dt: float,
    f_min: float | fdtdx.WaveCharacter,
    f_max: float | fdtdx.WaveCharacter,
    *,
    time_axis: int = 0,
    weights: jax.Array | None = None,
    max_modes: int = 16,
    spectral_density: float = 1.1,
    num_basis: int | None = None,
    svd_tol: float | None = None,
    err_thresh: float = 1e-2,
    rel_err_thresh: float = math.inf,
    Q_thresh: float = 10.0,
    amp_thresh: float = 0.0,
    rel_amp_thresh: float = 0.0,
    sort_by: Literal["frequency", "amplitude", "Q", "error"] = "frequency",
    t0: float = 0.0,
    _channel_scales: jax.Array | None = None,
) -> HarminvResult:
    """Extract complex resonant frequencies, decay rates, Q factors, and amplitudes via FDM.

    Supports both ``jax.jvp`` and ``jax.grad`` / ``jax.value_and_grad`` under ``bf.jit``
    and ``jax.jit``.

    Args:
        signal: Uniformly sampled real or complex time-domain field array.
        dt: Sample spacing in seconds (must be positive).
        f_min: Lower frequency bound in Hz or an ``fdtdx.WaveCharacter``. If both
            ``f_min`` and ``f_max`` are ``WaveCharacter`` objects, they are ordered
            automatically.
        f_max: Upper frequency bound in Hz or an ``fdtdx.WaveCharacter``.
        time_axis: Axis of ``signal`` corresponding to time steps (default ``0``).
        weights: Optional channel weights of shape matching ``signal`` with ``time_axis``
            removed. If ``None``, shared poles are extracted across all channels via a
            non-canceling stacked projection pencil.
        max_modes: Static output capacity ``K`` of the returned :class:`HarminvResult`.
        spectral_density: Basis oversampling factor ``J ~ spectral_density * df * T`` when
            ``num_basis`` is ``None`` (default ``1.1``, matching ``harminv``).
        num_basis: Optional explicit static Fourier basis count ``J``.
        svd_tol: Relative singular-value cutoff for ``U0`` rank truncation. If ``None``
            (default), uses ``1e-12`` for 64-bit inputs when ``jax_enable_x64`` is enabled
            and ``1e-5`` for 32-bit inputs (such as default ``float32``
            ``fdtdx.FieldDetector`` recordings) or when ``jax_enable_x64`` is disabled.
        err_thresh: Maximum allowed relative frequency residual error.
        rel_err_thresh: Maximum ratio of mode error to the lowest in-band mode error.
        Q_thresh: Minimum allowed ``|Q|`` for valid modes (default ``10.0``).
        amp_thresh: Minimum allowed RMS amplitude magnitude for valid modes.
        rel_amp_thresh: Minimum ratio of RMS amplitude to the strongest valid mode amplitude.
        sort_by: Ordering of valid modes: ``"frequency"``, ``"amplitude"``, ``"Q"``, or ``"error"``.
        t0: Simulation time of the first sample (seconds).

    Returns:
        Fixed-shape :class:`HarminvResult` PyTree of leading length ``max_modes``.
    """
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError(f"dt must be finite and positive, got {dt}.")
    if max_modes < 1:
        raise ValueError(f"max_modes must be >= 1, got {max_modes}.")
    if sort_by not in ("frequency", "amplitude", "Q", "error"):
        raise ValueError(f"Invalid sort_by={sort_by!r}; expected 'frequency', 'amplitude', 'Q', or 'error'.")

    f_low, f_high = _resolve_freq_bounds(f_min, f_max)
    nyquist = 0.5 / dt
    if max(abs(f_low), abs(f_high)) >= nyquist:
        raise ValueError(
            f"Frequency band [{f_low:.4e}, {f_high:.4e}] Hz exceeds Nyquist {nyquist:.4e} Hz "
            f"(for dt={dt:.4e} s). Reduce sampling interval or lower f_max."
        )

    arr = jnp.asarray(signal)
    if arr.ndim < 1:
        raise ValueError("signal must have at least 1 dimension (time).")
    arr = jnp.moveaxis(arr, time_axis, 0)
    n_steps = int(arr.shape[0])
    if n_steps < 5:
        raise ValueError(f"Need at least 5 time samples for harmonic inversion, got {n_steps}.")
    half_m = (n_steps - 3) // 2
    if half_m < 1:
        raise ValueError(f"Need at least 5 time samples so M >= 1, got N={n_steps}.")

    is_real_signal = not jnp.iscomplexobj(arr)
    paired_conjugate = bool(is_real_signal and f_low > 0.0)
    use_x64 = bool(jax.config.jax_enable_x64)
    real_dtype = jnp.float64 if use_x64 else jnp.float32
    complex_dtype = jnp.complex128 if use_x64 else jnp.complex64
    if jnp.issubdtype(arr.dtype, jnp.floating) or jnp.issubdtype(arr.dtype, jnp.complexfloating):
        input_bits = jnp.finfo(arr.dtype).bits
    else:
        input_bits = 64 if use_x64 else 32
    if svd_tol is None:
        eff_svd_tol = 1e-5 if (not use_x64 or input_bits <= 32) else 1e-12
    else:
        if svd_tol < 0.0 or not math.isfinite(svd_tol):
            raise ValueError(f"svd_tol must be finite and non-negative, got {svd_tol}.")
        eff_svd_tol = float(svd_tol) if use_x64 else max(float(svd_tol), 1e-6)
    if not use_x64:
        warnings.warn(
            "jax_enable_x64 is disabled; FDM is running in float32/complex64, which may lose accuracy "
            "for high-Q modes or narrow frequency bands.",
            UserWarning,
            stacklevel=2,
        )

    channel_shape = arr.shape[1:]
    m_ch = math.prod(channel_shape) if channel_shape else 1
    y_mat = arr.reshape((n_steps, m_ch))

    if weights is not None:
        w_arr = jnp.asarray(weights)
        if w_arr.shape != channel_shape:
            raise ValueError(f"weights shape {w_arr.shape} does not match signal channel shape {channel_shape}.")
        w_proj = w_arr.reshape((m_ch, 1))
        if _channel_scales is not None:
            w_proj = _channel_scales.reshape((m_ch, 1)) * w_proj
    else:
        scales_flat = None if _channel_scales is None else _channel_scales.reshape((m_ch,))
        w_proj, _ = _make_projection_weights(m_ch, real_dtype, scales_flat)

    c_proj = _project_channels(
        y_mat,
        w_proj,
        None if _channel_scales is None else _channel_scales.reshape((m_ch,)),
        real_dtype,
        complex_dtype,
    )
    c_sq = jnp.sum(_abs_sq(c_proj))
    c_has_energy = c_sq > 0
    safe_c_scale = jnp.sqrt(jnp.where(c_has_energy, c_sq, 1.0))
    c_norm = (c_proj / safe_c_scale).astype(complex_dtype)

    if num_basis is None:
        raw_j = math.ceil(spectral_density * (f_high - f_low) * max(n_steps - 1, 1) * dt)
        num_j = max(min(raw_j, 200), 4)
        num_j = min(num_j, half_m) if half_m >= 2 else 1
    else:
        num_j = int(num_basis)
        if num_j < 1:
            raise ValueError(f"num_basis must be >= 1, got {num_j}.")

    phi_low = 2.0 * math.pi * f_low * dt
    phi_high = 2.0 * math.pi * f_high * dt
    phi_mid = 0.5 * (phi_low + phi_high)
    min_dphi = math.pi / max(2 * half_m, 4)
    half_span = max(0.5 * (phi_high - phi_low), 0.5 * max(num_j - 1, 0) * min_dphi)
    b_low = phi_mid - half_span
    b_high = phi_mid + half_span
    if phi_low > 0.0:
        b_low = max(1e-6, b_low)
    b_low = max(-0.99 * math.pi, b_low)
    b_high = min(0.99 * math.pi, b_high)
    if b_high <= b_low:
        b_low, b_high = phi_low, phi_high

    if num_j == 1:
        phi_basis = jnp.asarray([phi_mid], dtype=real_dtype)
    else:
        phi_basis = jnp.linspace(b_low, b_high, num_j, dtype=real_dtype)

    u0, u1, u2 = _build_fdm_matrices(c_norm, phi_basis, real_dtype)
    u_eig, err, is_mode_f = _solve_fdm_poles(u0, u1, u2, eff_svd_tol)
    is_mode = is_mode_f > 0.5

    u_nz = jnp.where(is_mode & (jnp.abs(u_eig) > 1e-30), u_eig, jnp.exp(-1j * phi_mid))
    phi = phi_mid - jnp.angle(u_nz * jnp.exp(1j * phi_mid))
    freq = phi / (2.0 * math.pi * dt)
    decay = -jnp.log(jnp.abs(u_nz)) / dt
    safe_decay = jnp.where(jnp.abs(decay) > 1e-30, decay, jnp.where(decay >= 0, 1e-30, -1e-30))
    q_factor = (math.pi * freq) / safe_decay

    in_band = (freq >= f_low) & (freq <= f_high)
    min_inband_err = jnp.min(jnp.where(is_mode & in_band, err, jnp.inf))
    err_ok = (err <= err_thresh) & jnp.where(
        jnp.isfinite(min_inband_err),
        err <= rel_err_thresh * min_inband_err,
        False,
    )
    q_ok = jnp.abs(q_factor) >= Q_thresh
    growth_ok = (-decay * n_steps * dt) <= 40.0
    pre_valid_all = is_mode & in_band & err_ok & q_ok & growth_ok

    cand_k = min(max_modes, num_j)
    # In lexsort, the LAST key is the primary sort key.
    init_order = jax.lax.stop_gradient(
        jnp.lexsort((freq, err, jnp.where(pre_valid_all, 0, 1)))[:cand_k]
    )
    u_sel = u_nz[init_order]
    freq_sel = freq[init_order]
    decay_sel = decay[init_order]
    q_sel = q_factor[init_order]
    err_sel = err[init_order]
    pre_valid = pre_valid_all[init_order]

    # Guard u_k BEFORE Vandermonde geometric series so unused slots never overflow.
    u_k = jnp.where(pre_valid, u_sel, jnp.exp(-1j * phi_mid))

    j_f = max(num_j, 2 * cand_k + 6)
    phi_dft = jnp.linspace(phi_low, phi_high, j_f, dtype=real_dtype)
    dft_data = _narrowband_dft(y_mat, phi_dft, real_dtype, complex_dtype)
    amps_flat = _solve_amplitudes(
        dft_data=dft_data,
        phi_dft=phi_dft,
        u_k=u_k,
        pre_valid=pre_valid,
        n_steps=n_steps,
        paired_conjugate=paired_conjugate,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )

    amp_sq = jnp.mean(_abs_sq(amps_flat), axis=1)
    amp_nonzero = amp_sq > 0
    amp_mag = jnp.where(amp_nonzero, jnp.sqrt(jnp.where(amp_nonzero, amp_sq, 1.0)), 0.0)
    max_amp = jnp.max(jnp.where(pre_valid, amp_mag, 0.0))
    amp_ok = (amp_mag >= amp_thresh) & (amp_mag >= rel_amp_thresh * max_amp)
    valid_sel = pre_valid & amp_ok

    if sort_by == "frequency":
        sec_key = freq_sel
    elif sort_by == "amplitude":
        sec_key = -amp_mag
    elif sort_by == "Q":
        sec_key = -jnp.abs(q_sel)
    else:
        sec_key = err_sel

    final_order = jax.lax.stop_gradient(
        jnp.lexsort((freq_sel, sec_key, jnp.where(valid_sel, 0, 1)))
    )
    valid = valid_sel[final_order]
    freq_out = jnp.where(valid, freq_sel[final_order], 0.0).astype(real_dtype)
    decay_out = jnp.where(valid, decay_sel[final_order], 0.0).astype(real_dtype)
    q_out = jnp.where(valid, q_sel[final_order], 0.0).astype(real_dtype)
    err_out = jnp.where(valid, err_sel[final_order], 0.0).astype(real_dtype)

    omega_raw = 2.0 * math.pi * freq_out - 1j * decay_out
    omega_out = jnp.where(valid, omega_raw, 0.0 + 0.0j).astype(complex_dtype)

    safe_f = jnp.where(valid & (jnp.abs(freq_out) > 1e-30), freq_out, 1.0)
    wavelength_out = jnp.where(
        valid & (jnp.abs(freq_out) > 1e-30),
        fdtdx.constants.c / safe_f,
        0.0,
    ).astype(real_dtype)

    amps_sorted = amps_flat[final_order]
    amps_masked = jnp.where(valid[:, None], amps_sorted, 0.0 + 0.0j).astype(complex_dtype)

    pad_k = max_modes - cand_k
    if pad_k > 0:
        freq_out = jnp.pad(freq_out, (0, pad_k))
        decay_out = jnp.pad(decay_out, (0, pad_k))
        q_out = jnp.pad(q_out, (0, pad_k))
        omega_out = jnp.pad(omega_out, (0, pad_k))
        wavelength_out = jnp.pad(wavelength_out, (0, pad_k))
        err_out = jnp.pad(err_out, (0, pad_k))
        valid = jnp.pad(valid, (0, pad_k), constant_values=False)
        amps_masked = jnp.pad(amps_masked, ((0, pad_k), (0, 0)))

    amplitude_out = amps_masked.reshape((max_modes, *channel_shape))

    return HarminvResult(
        frequency=freq_out,
        decay_rate=decay_out,
        Q=q_out,
        omega=omega_out,
        wavelength=wavelength_out,
        amplitude=amplitude_out,
        error=err_out,
        valid=valid,
        dt=jnp.asarray(dt, dtype=real_dtype),
        t0=jnp.asarray(t0, dtype=real_dtype),
        is_real_signal=is_real_signal,
        paired_conjugate=paired_conjugate,
    )


def _extract_detector_fields(
    detector: fdtdx.FieldDetector,
    state_or_arrays: fdtdx.ArrayContainer | Mapping[str, Any] | jax.Array,
) -> jax.Array:
    if isinstance(state_or_arrays, fdtdx.ArrayContainer):
        return state_or_arrays.detector_states[detector.name]["fields"]
    if isinstance(state_or_arrays, Mapping):
        if "fields" in state_or_arrays:
            return jnp.asarray(state_or_arrays["fields"])
        if detector.name in state_or_arrays:
            entry = state_or_arrays[detector.name]
            if isinstance(entry, Mapping) and "fields" in entry:
                return jnp.asarray(entry["fields"])
        raise KeyError(
            f"Could not find 'fields' or detector '{detector.name}' in state mapping keys {sorted(state_or_arrays)}."
        )
    return jnp.asarray(state_or_arrays)


def harminv_detector(
    detector: fdtdx.FieldDetector,
    state_or_arrays: fdtdx.ArrayContainer | Mapping[str, Any] | jax.Array,
    f_min: float | fdtdx.WaveCharacter,
    f_max: float | fdtdx.WaveCharacter,
    *,
    components: str | Sequence[str] | None = None,
    reduce_spatial: bool = False,
    **kwargs: Any,
) -> HarminvResult:
    """Run differentiable harmonic inversion directly on an ``fdtdx.FieldDetector`` recording.

    Note that when ``detector.exact_interpolation=True`` (the ``fdtdx`` default), ``H``
    components are time-averaged ``(H_prev + H) / 2`` which preserves all pole frequencies
    and decay rates while scaling ``H`` amplitudes by ``cos(omega * dt_sim / 2)``. When both
    ``E`` and ``H`` components are included and ``weights`` is not provided, ``H`` components
    are scaled by free-space impedance ``eta0`` internally during pole extraction so magnetic
    modes are not drowned out by electric components.

    Args:
        detector: Placed forward ``fdtdx.FieldDetector``.
        state_or_arrays: ``fdtdx.ArrayContainer``, full ``detector_states`` dict, single
            ``DetectorState`` dict (``{"fields": ...}``), or raw recorded ``fields`` array.
        f_min: Lower frequency bound in Hz or an ``fdtdx.WaveCharacter``.
        f_max: Upper frequency bound in Hz or an ``fdtdx.WaveCharacter``.
        components: Optional single component name (e.g. ``"Ey"``, squeezing the component
            axis) or sequence of component names from ``detector.components``.
        reduce_spatial: If ``True`` and ``not detector.reduce_volume``, average spatial axes
            using ``detector._volume_weighted_spatial_mean``.
        **kwargs: Forwarded to :func:`harminv`.

    Returns:
        Fixed-shape :class:`HarminvResult` PyTree.
    """
    dt, t0_det = detector_time_grid(detector)
    t0 = float(kwargs.pop("t0", t0_det))
    fields = _extract_detector_fields(detector, state_or_arrays)

    det_comps = tuple(detector.components)
    squeeze_comp = False
    if components is None:
        selected_comps = det_comps
    elif isinstance(components, str):
        if components not in det_comps:
            raise ValueError(f"Component {components!r} not in detector '{detector.name}' components {det_comps}.")
        idx = det_comps.index(components)
        fields = fields[:, idx, ...]
        selected_comps = (components,)
        squeeze_comp = True
    else:
        selected_comps = tuple(components)
        if not selected_comps:
            raise ValueError("components sequence cannot be empty.")
        indices = []
        for comp in selected_comps:
            if comp not in det_comps:
                raise ValueError(f"Component {comp!r} not in detector '{detector.name}' components {det_comps}.")
            indices.append(det_comps.index(comp))
        fields = fields[:, jnp.asarray(indices, dtype=jnp.int32), ...]

    if reduce_spatial and not detector.reduce_volume:
        leading_dims = 1 if squeeze_comp else 2
        fields = detector._volume_weighted_spatial_mean(fields, leading_dims=leading_dims)

    channel_scales = None
    has_e = any(c.startswith("E") for c in selected_comps)
    has_h = any(c.startswith("H") for c in selected_comps)
    if has_e and has_h and not squeeze_comp and kwargs.get("weights") is None:
        # Match H scale to E scale during pole extraction so magnetic modes are weighted comparably.
        comp_scales = jnp.asarray(
            [fdtdx.constants.eta0 if c.startswith("H") else 1.0 for c in selected_comps],
            dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32,
        )
        spatial_shape = fields.shape[2:]
        channel_scales = jnp.broadcast_to(
            comp_scales[(slice(None),) + (None,) * len(spatial_shape)],
            fields.shape[1:],
        )

    return harminv(
        fields,
        dt=dt,
        f_min=f_min,
        f_max=f_max,
        time_axis=0,
        t0=t0,
        _channel_scales=channel_scales,
        **kwargs,
    )
