"""Unit, reference, autodiff, and FDTD tests for ``bifrost.harminv``."""

from __future__ import annotations

import math

import fdtdx
import gdsfactory as gf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gdsfactory.technology import LayerLevel, LayerStack

import bifrost as bf


def _complex_two_mode_signal(
    n_steps: int,
    dt: float,
    f1: float,
    g1: float,
    a1: complex,
    f2: float,
    g2: float,
    a2: complex,
) -> jax.Array:
    t = jnp.arange(n_steps, dtype=jnp.float64) * dt
    w1 = 2.0 * math.pi * f1 - 1j * g1
    w2 = 2.0 * math.pi * f2 - 1j * g2
    return a1 * jnp.exp(-1j * w1 * t) + a2 * jnp.exp(-1j * w2 * t)


def _real_two_mode_signal(
    n_steps: int,
    dt: float,
    f1: float,
    g1: float,
    a1: complex,
    f2: float,
    g2: float,
    a2: complex,
) -> jax.Array:
    return 2.0 * jnp.real(_complex_two_mode_signal(n_steps, dt, f1, g1, a1, f2, g2, a2))


# ------------------------------------------------------------------ 3.1 pure-JAX & reference tests


def test_single_and_two_mode_complex_and_real_recovery() -> None:
    dt = 2.5e-16
    n_steps = 800
    f1, g1, a1 = 193.0e12, 4.0e12, 1.2 * np.exp(0.4j)
    f2, g2, a2 = 196.5e12, 2.5e12, 0.7 * np.exp(-0.9j)

    sig_c = _complex_two_mode_signal(n_steps, dt, f1, g1, a1, f2, g2, a2)
    res_c = bf.harminv(sig_c, dt, 180.0e12, 210.0e12, max_modes=8, Q_thresh=5.0)
    assert int(jnp.sum(res_c.valid)) == 2
    assert bool(res_c.valid[0]) and bool(res_c.valid[1]) and not bool(res_c.valid[2])
    np.testing.assert_allclose(res_c.frequency[:2], [f1, f2], rtol=1e-10)
    np.testing.assert_allclose(res_c.decay_rate[:2], [g1, g2], rtol=1e-9)
    np.testing.assert_allclose(res_c.Q[:2], [math.pi * f1 / g1, math.pi * f2 / g2], rtol=1e-9)
    np.testing.assert_allclose(res_c.omega[:2], [2 * math.pi * f1 - 1j * g1, 2 * math.pi * f2 - 1j * g2], rtol=1e-9)
    np.testing.assert_allclose(res_c.wavelength[:2], [fdtdx.constants.c / f1, fdtdx.constants.c / f2], rtol=1e-10)
    np.testing.assert_allclose(res_c.amplitude[:2], [a1, a2], rtol=1e-9)
    np.testing.assert_allclose(res_c.reconstruct(n_steps), sig_c, atol=1e-10, rtol=1e-10)

    # Short ~10-cycle real signal: conjugate-pair amplitude recovery must stay accurate.
    dt_short = 5.0e-16
    n_short = 120  # 60 fs ~ 11.6 cycles at 193 THz
    sig_r = _real_two_mode_signal(n_short, dt_short, f1, g1, a1, f2, g2, a2)
    res_r = bf.harminv(sig_r, dt_short, 175.0e12, 215.0e12, max_modes=8, Q_thresh=5.0)
    assert int(jnp.sum(res_r.valid)) == 2
    np.testing.assert_allclose(res_r.frequency[:2], [f1, f2], rtol=1e-8)
    np.testing.assert_allclose(res_r.decay_rate[:2], [g1, g2], rtol=1e-7)
    np.testing.assert_allclose(res_r.amplitude[:2], [a1, a2], rtol=1e-7)
    np.testing.assert_allclose(res_r.reconstruct(n_short), sig_r, atol=1e-9, rtol=1e-9)

    modes = res_r.to_modes()
    assert len(modes) == 2
    assert modes[0]["frequency"] == pytest.approx(f1, rel=1e-8)
    sel = res_r.select(fdtdx.WaveCharacter(frequency=f2))
    assert float(sel["frequency"]) == pytest.approx(f2, rel=1e-8)


def test_matches_meep_py_do_harminv() -> None:
    mp = pytest.importorskip("meep")
    dt = 4.0e-16
    n_steps = 1200
    f1, g1, a1 = 190.0e12, 1.5e12, 1.0 * np.exp(0.3j)
    f2, g2, a2 = 202.0e12, 1.8e12, 0.6 * np.exp(-0.7j)
    sig = _real_two_mode_signal(n_steps, dt, f1, g1, a1, f2, g2, a2)

    # Meep normalizes time in dimensionless units; scale dt * f0 identically.
    dt_norm = dt * 1e12  # units of ps (frequencies in THz)
    f_min_thz, f_max_thz = 175.0, 215.0
    meep_bands = mp.py_do_harminv(
        [complex(v) for v in np.asarray(sig, dtype=np.float64)],
        dt_norm,
        f_min_thz,
        f_max_thz,
        16,
        1.1,
        50.0,
        mp.inf,
        0.01,
        -1.0,
        -1.0,
    )
    meep_bands = sorted(meep_bands, key=lambda item: item[0].real)
    assert len(meep_bands) == 2

    res = bf.harminv(sig, dt, f_min_thz * 1e12, f_max_thz * 1e12, max_modes=16, Q_thresh=50.0, err_thresh=0.01)
    assert int(jnp.sum(res.valid)) == 2
    for idx, (m_freq, m_amp, _m_err) in enumerate(meep_bands):
        assert float(res.frequency[idx]) == pytest.approx(m_freq.real * 1e12, rel=1e-8)
        assert float(res.decay_rate[idx]) == pytest.approx(-2.0 * math.pi * m_freq.imag * 1e12, rel=1e-7)
        m_q = m_freq.real / (-2.0 * m_freq.imag)
        assert float(res.Q[idx]) == pytest.approx(m_q, rel=1e-7)
        # Meep's do_harminv returns conjugated amplitudes relative to exp(-i omega t).
        assert abs(complex(res.amplitude[idx]) - np.conj(m_amp)) < 1e-7


def test_multichannel_and_antisymmetric_spatial_modes() -> None:
    dt = 4.0e-16
    n_steps = 900
    t = jnp.arange(n_steps, dtype=jnp.float64) * dt
    f_even, g_even = 188.0e12, 2.0e12
    f_odd, g_odd = 197.0e12, 2.8e12

    x = jnp.linspace(-1.0, 1.0, 8, dtype=jnp.float64)
    y = jnp.linspace(-1.0, 1.0, 6, dtype=jnp.float64)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    # Even profile in channel 0; strictly odd (zero-mean) spatial profile in channel 1.
    prof_even = jnp.exp(-(xx**2 + yy**2))[:, :, None]
    prof_odd = (xx * jnp.exp(-0.5 * (xx**2 + yy**2)))[:, :, None]

    amp_true_1 = jnp.stack([0.8 * jnp.exp(0.2j) * prof_even, jnp.zeros_like(prof_even)], axis=0)
    amp_true_2 = jnp.stack([jnp.zeros_like(prof_odd), 1.1 * jnp.exp(-0.5j) * prof_odd], axis=0)

    w1 = 2.0 * math.pi * f_even - 1j * g_even
    w2 = 2.0 * math.pi * f_odd - 1j * g_odd
    fields = 2.0 * jnp.real(
        jnp.exp(-1j * w1 * t)[:, None, None, None, None] * amp_true_1[None, ...]
        + jnp.exp(-1j * w2 * t)[:, None, None, None, None] * amp_true_2[None, ...]
    )
    assert fields.shape == (n_steps, 2, 8, 6, 1)

    res = bf.harminv(fields, dt, 175.0e12, 210.0e12, max_modes=8, Q_thresh=10.0)
    assert int(jnp.sum(res.valid)) == 2
    assert res.amplitude.shape == (8, 2, 8, 6, 1)
    np.testing.assert_allclose(res.frequency[:2], [f_even, f_odd], rtol=1e-9)
    np.testing.assert_allclose(res.decay_rate[:2], [g_even, g_odd], rtol=1e-8)
    np.testing.assert_allclose(res.amplitude[0], amp_true_1, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(res.amplitude[1], amp_true_2, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(res.reconstruct(n_steps), fields, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("noise_level", [0.0, 1e-4])
def test_autodiff_jvp_and_grad_match_finite_differences(noise_level: float) -> None:
    dt = 4.0e-16
    n_steps = 600
    f0, g0, amp0, phi0 = 194.0e12, 2.2e12, 1.3, 0.35
    f1, g1, amp1, phi1 = 201.0e12, 1.9e12, 0.7, -0.4
    t = jnp.arange(n_steps, dtype=jnp.float64) * dt

    key = jax.random.PRNGKey(42)
    noise = noise_level * jax.random.normal(key, shape=(n_steps, 6), dtype=jnp.float64)
    ch_weights = jnp.asarray([1.0, -0.5, 0.8, 0.3, -1.1, 0.6], dtype=jnp.float64)

    def make_field(params_vec: jax.Array) -> jax.Array:
        df, dg, da, dphi = params_vec
        s1 = 2.0 * amp0 * da * jnp.cos(2.0 * math.pi * f0 * df * t + phi0 + dphi) * jnp.exp(-(g0 * dg) * t)
        s2 = 2.0 * amp1 * jnp.cos(2.0 * math.pi * f1 * t + phi1) * jnp.exp(-g1 * t)
        return (s1 + 0.4 * s2)[:, None] * ch_weights[None, :] + noise

    # Use a raised svd_tol on noisy data so truncation active mask has r < J,
    # directly exercising all three Golub-Pereyra pseudoinverse terms; keep clean
    # tolerance well above the 1e-12 negative-frequency tail to avoid rank flips.
    tol = 1e-2 if noise_level > 0 else 1e-9

    def loss_metrics(params_vec: jax.Array) -> jax.Array:
        sig = make_field(params_vec)
        res = bf.harminv(
            sig,
            dt,
            180.0e12,
            212.0e12,
            max_modes=16,
            Q_thresh=5.0,
            svd_tol=tol,
        )
        m0 = res.select(f0)
        amp_sq = jnp.sum(m0["amplitude"].real ** 2 + m0["amplitude"].imag ** 2)
        return jnp.array([m0["frequency"] / 1e12, m0["decay_rate"] / 1e12, m0["Q"], amp_sq])

    p0 = jnp.ones(4, dtype=jnp.float64)
    p0 = p0.at[3].set(0.0)
    tangent = jnp.asarray([0.3, -0.2, 0.5, 0.4], dtype=jnp.float64)

    # Forward-mode JVP check (under bf.jit as well)
    primals_out, tangents_out = bf.jit(lambda p, v: jax.jvp(loss_metrics, (p,), (v,)))(p0, tangent)
    assert jnp.all(jnp.isfinite(primals_out))
    assert jnp.all(jnp.isfinite(tangents_out))

    eps = 1e-5
    fd_jvp = (loss_metrics(p0 + eps * tangent) - loss_metrics(p0 - eps * tangent)) / (2.0 * eps)
    np.testing.assert_allclose(tangents_out, fd_jvp, rtol=1e-5, atol=1e-4)

    # Reverse-mode bf.jit(jax.value_and_grad) check across each scalar metric
    for idx in range(4):
        def scalar_obj(p: jax.Array, i: int = idx) -> jax.Array:
            return loss_metrics(p)[i]

        val, grad = bf.jit(jax.value_and_grad(scalar_obj))(p0)
        assert jnp.isfinite(val)
        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.dot(grad, tangent)) == pytest.approx(float(tangents_out[idx]), rel=1e-5, abs=1e-5)
        fd_grad = np.zeros(4)
        for j in range(4):
            e = jnp.zeros(4, dtype=jnp.float64).at[j].set(eps)
            fd_grad[j] = float(scalar_obj(p0 + e) - scalar_obj(p0 - e)) / (2.0 * eps)
        np.testing.assert_allclose(grad, fd_grad, rtol=1e-5, atol=1e-4)


def test_float32_input_preserves_grad_dtype_and_zero_signal_is_safe() -> None:
    dt = 4.0e-16
    n_steps = 400
    sig32 = _real_two_mode_signal(n_steps, dt, 193e12, 2e12, 1.0, 200e12, 2e12, 0.5).astype(jnp.float32)

    def freq_loss(s: jax.Array) -> jax.Array:
        res = bf.harminv(s, dt, 180e12, 210e12, max_modes=8, Q_thresh=5.0)
        return jnp.sum(jnp.where(res.valid, res.frequency, 0.0)) / 1e12

    val, grad = bf.jit(jax.value_and_grad(freq_loss))(sig32)
    assert jnp.isfinite(val)
    assert grad.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(grad)) and float(jnp.linalg.norm(grad)) > 0.0

    # Pure-zero input must return all-invalid zeros with zero finite gradient (no NaNs).
    zeros = jnp.zeros(n_steps, dtype=jnp.float32)
    z_val, z_grad = bf.jit(jax.value_and_grad(freq_loss))(zeros)
    assert float(z_val) == 0.0
    assert z_grad.dtype == jnp.float32
    np.testing.assert_array_equal(z_grad, jnp.zeros_like(zeros))


def test_field_detector_interface_and_errors() -> None:
    gf.gpdk.PDK.activate()
    comp = gf.components.straight(length=0.8, width=0.5)
    ls = LayerStack(layers={"core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=0.22, material="core")})
    mats = {"core": fdtdx.Material(permittivity=3.48**2), "clad": fdtdx.Material(permittivity=1.444**2)}
    stack = bf.StackDevice.from_layer_stack(
        comp,
        ls,
        mats,
        background="clad",
        zrange_layers=["core"],
        boundary_offset=(0.2, -0.2, 0.2, -0.2, 0.2, -0.2),
        name="stack",
    )
    config = stack.get_simulation_config(
        time=120e-15,
        grid=fdtdx.core.grid.UniformGrid(spacing=0.1e-6),
        backend=jax.default_backend(),
    )
    vol, c_vol = stack.get_simulation_volume()
    sw = fdtdx.OnOffSwitch(start_time=20e-15, interval=2)
    det_full, c_det = stack.create_field_detector_at(
        "o2", (0.1, 0.1), wavelength_um=1.55, components=("Ey", "Hz"), reduce_volume=False, switch=sw, name="det"
    )
    det_inv, c_inv = stack.create_field_detector_at(
        "o1", (0.1, 0.1), wavelength_um=1.55, components=("Ey",), reduce_volume=True, inverse=True, name="det_inv"
    )

    # Unplaced detector error
    with pytest.raises(ValueError, match="not placed"):
        bf.detector_time_grid(det_full)

    objs, arrays, _, config, _ = fdtdx.place_objects(
        [vol, stack, det_full, det_inv], config, [*c_vol, *c_det, *c_inv], jax.random.PRNGKey(0)
    )

    # Inverse detector rejected
    with pytest.raises(ValueError, match="inverse=True"):
        bf.detector_time_grid(objs["det_inv"])

    dt_eff, t0 = bf.detector_time_grid(objs["det"])
    assert dt_eff == pytest.approx(2.0 * config.time_step_duration)
    expected_first = next(t for t, on in enumerate(objs["det"]._calculate_on_list()) if on) * config.time_step_duration
    assert t0 == pytest.approx(expected_first)

    # Inject synthetic damped wave into det's state tensor
    n_rec = objs["det"].num_time_steps_recorded
    tau = jnp.arange(n_rec, dtype=jnp.float64) * dt_eff
    f0, g0 = 193.0e12, 1.5e12
    w0 = 2.0 * math.pi * f0 - 1j * g0
    spatial = jnp.ones(objs["det"].grid_shape, dtype=jnp.float64)
    ey = 2.0 * jnp.real(1.5 * jnp.exp(0.2j) * jnp.exp(-1j * w0 * tau))[:, None, None, None] * spatial[None, ...]
    hz = 2.0 * jnp.real(0.004 * jnp.exp(-0.3j) * jnp.exp(-1j * w0 * tau))[:, None, None, None] * spatial[None, ...]
    fields = jnp.stack([ey, hz], axis=1).astype(jnp.float32)
    arrays_with_data = arrays.aset("detector_states", {**arrays.detector_states, "det": {"fields": fields}})

    # Accept ArrayContainer, detector_states, DetectorState, WaveCharacter in either wavelength order
    wc_a = fdtdx.WaveCharacter(wavelength=1.65e-6)
    wc_b = fdtdx.WaveCharacter(wavelength=1.45e-6)
    r1 = bf.harminv_detector(objs["det"], arrays_with_data, wc_a, wc_b, Q_thresh=5.0)
    r2 = bf.harminv_detector(objs["det"], arrays_with_data.detector_states, wc_b, wc_a, Q_thresh=5.0)
    r3 = bf.harminv_detector(
        objs["det"], arrays_with_data.detector_states["det"], wc_a, wc_b, components="Ey", reduce_spatial=True, Q_thresh=5.0
    )
    assert int(jnp.sum(r1.valid)) == 1
    assert float(r1.frequency[0]) == pytest.approx(f0, rel=1e-5)
    assert float(r2.frequency[0]) == pytest.approx(f0, rel=1e-5)
    assert float(r3.frequency[0]) == pytest.approx(f0, rel=1e-5)
    assert r3.amplitude.shape == (r3.frequency.shape[0],)  # single component + reduce_spatial squeezed

    # Nyquist violation
    with pytest.raises(ValueError, match="Nyquist"):
        bf.harminv(fields[:, 0, 0, 0, 0], dt=dt_eff, f_min=100e12, f_max=0.6 / dt_eff)

    # Non-uniform fixed_on_time_steps rejected
    det_nonunif = fdtdx.FieldDetector(
        name="nonunif",
        partial_grid_shape=(1, 1, 1),
        switch=fdtdx.OnOffSwitch(fixed_on_time_steps=[0, 1, 3, 6, 10, 15]),
    )
    objs2, _, _, _, _ = fdtdx.place_objects([vol, det_nonunif], config, [det_nonunif.place_at_center(vol)], jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="non-uniform"):
        bf.detector_time_grid(objs2["nonunif"])


# ------------------------------------------------------------------ 3.2 simplest bifrost FDTD resonator tests


def test_simplest_bifrost_fabry_perot_resonator_forward_meep_and_grad() -> None:
    """1D-periodic Si slab resonator in bifrost: analytical mode, Meep comparison, and d(f)/dL."""
    gf.gpdk.PDK.activate()
    mp = pytest.importorskip("meep")

    L0 = 1.36  # um
    n_core = 3.48
    ls = LayerStack(
        layers={"core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=-0.20, thickness=0.40, material="core", mesh_order=1)}
    )
    materials = {
        "core": fdtdx.Material(permittivity=n_core**2),
        "air": fdtdx.Material(permittivity=1.0),
    }

    comp = gf.components.straight(length=L0, width=0.40)
    stack = bf.StackDevice.from_layer_stack(
        comp,
        ls,
        materials,
        background="air",
        zrange_layers=["core"],
        boundary_offset=(0.72, -0.72, -0.14, 0.14, -0.14, 0.14),
        name="stack",
    )
    config = stack.get_simulation_config(
        time=500e-15,
        grid=fdtdx.core.grid.UniformGrid(spacing=0.04e-6),
        backend=jax.default_backend(),
        gradient_config=fdtdx.GradientConfig(method="checkpointed", num_checkpoints=16),
    )

    wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=wave,
        spectral_width=fdtdx.WaveCharacter(frequency=25e12),
    )
    volume, c_vol = stack.get_simulation_volume()
    source = fdtdx.UniformPlaneSource(
        name="source",
        partial_grid_shape=(1, None, None),
        wave_character=wave,
        temporal_profile=pulse,
        fixed_E_polarization_vector=(0.0, 1.0, 0.0),
        direction="+",
    )
    c_src = [
        source.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(-1, 0, 0),
            other_positions=(-1, 0, 0),
            grid_margins=(10, 0, 0),
        )
    ]
    probe = fdtdx.FieldDetector(
        name="probe",
        partial_grid_shape=(1, None, None),
        components=("Ey",),
        reduce_volume=True,
        switch=fdtdx.OnOffSwitch(start_time=140e-15, interval=2),
    )
    c_det = [
        probe.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(1, 0, 0),
            other_positions=(1, 0, 0),
            grid_margins=(-10, 0, 0),
        )
    ]
    b_cfg = fdtdx.BoundaryConfig(
        thickness_grid_minx=8,
        thickness_grid_maxx=8,
        thickness_grid_miny=1,
        thickness_grid_maxy=1,
        thickness_grid_minz=1,
        thickness_grid_maxz=1,
        boundary_type_miny="periodic",
        boundary_type_maxy="periodic",
        boundary_type_minz="periodic",
        boundary_type_maxz="periodic",
    )
    boundaries, c_bnd = fdtdx.boundary_objects_from_config(b_cfg, volume)

    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        [volume, stack, source, probe, *boundaries.values()],
        config,
        [*c_vol, *c_src, *c_det, *c_bnd],
        key,
    )
    params = {**params, **bf.init_stack_params(objects)}
    placed_stack = objects["stack"]
    placed_probe = objects["probe"]

    f_lo = fdtdx.WaveCharacter(wavelength=1.75e-6)
    f_hi = fdtdx.WaveCharacter(wavelength=1.40e-6)

    def run_resonator(length_um: jax.Array) -> tuple[bf.HarminvResult, fdtdx.ArrayContainer]:
        c_new = gf.components.straight(length=length_um, width=0.40)
        p = {**params, "stack": {**params["stack"], **placed_stack.pack(c_new)}}
        arrays_in, objs_in, _ = bf.apply_stack(arrays, objects, p, key)
        _, final_arrays = fdtdx.run_fdtd(arrays=arrays_in, objects=objs_in, config=config, key=key, show_progress=False)
        res = bf.harminv_detector(placed_probe, final_arrays, f_lo, f_hi, components="Ey", Q_thresh=5.0)
        return res, final_arrays

    # Forward accuracy vs analytical Fabry-Perot m=6 mode and Meep harminv
    res0, final0 = run_resonator(jnp.asarray(L0, dtype=stack.dtype))
    assert int(jnp.sum(res0.valid)) >= 1
    m6_analytic_f = 6.0 * fdtdx.constants.c / (2.0 * n_core * L0 * 1e-6)
    r_fresnel = (n_core - 1.0) / (n_core + 1.0)
    m6_analytic_q = -6.0 * math.pi / (2.0 * math.log(abs(r_fresnel)))

    mode = res0.select(m6_analytic_f)
    assert bool(mode["valid"])
    assert float(mode["frequency"]) == pytest.approx(m6_analytic_f, rel=0.025)
    assert float(mode["Q"]) == pytest.approx(m6_analytic_q, rel=0.15)

    # Direct comparison against Meep py_do_harminv on the identical FDTD detector trace
    dt_eff, _ = bf.detector_time_grid(placed_probe)
    trace = np.asarray(final0.detector_states["probe"]["fields"][:, 0], dtype=np.float64)
    meep_modes = mp.py_do_harminv(
        [complex(v) for v in trace],
        dt_eff * 1e12,
        f_lo.get_frequency() / 1e12,
        f_hi.get_frequency() / 1e12,
        16,
        1.1,
        5.0,
        mp.inf,
        0.01,
        -1.0,
        -1.0,
    )
    assert len(meep_modes) >= 1
    closest_meep = min(meep_modes, key=lambda item: abs(item[0].real * 1e12 - float(mode["frequency"])))
    assert float(mode["frequency"]) == pytest.approx(closest_meep[0].real * 1e12, rel=1e-3)
    meep_q = closest_meep[0].real / (-2.0 * closest_meep[0].imag)
    assert float(mode["Q"]) == pytest.approx(meep_q, rel=1e-2)

    # End-to-end differentiable gradient df/dL via bf.jit(jax.value_and_grad)
    def resonant_freq_thz(length_um: jax.Array) -> jax.Array:
        res, _ = run_resonator(length_um)
        return res.select(m6_analytic_f)["frequency"] / 1e12

    val_fn = bf.jit(resonant_freq_thz)
    grad_fn = bf.jit(jax.value_and_grad(resonant_freq_thz))

    f_val, df_dl = grad_fn(jnp.asarray(L0, dtype=stack.dtype))
    h = 0.01  # um
    fd_df_dl = (
        float(val_fn(jnp.asarray(L0 + h, dtype=stack.dtype)))
        - float(val_fn(jnp.asarray(L0 - h, dtype=stack.dtype)))
    ) / (2.0 * h)

    assert np.isfinite(float(f_val)) and np.isfinite(float(df_dl))
    expected_df_dl = -(float(f_val) / L0)
    assert float(df_dl) < 0.0
    assert float(df_dl) == pytest.approx(expected_df_dl, rel=0.1)
    assert float(df_dl) == pytest.approx(fd_df_dl, rel=0.05)


def test_3d_strip_resonator_multichannel_detector_and_thickness_grad() -> None:
    """Minimal 3D strip waveguide resonator with reduce_volume=False and d(f)/d(thickness)."""
    gf.gpdk.PDK.activate()
    ls = LayerStack(
        layers={"core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=0.22, material="core", mesh_order=1)}
    )
    materials = {
        "core": fdtdx.Material(permittivity=3.48**2),
        "air": fdtdx.Material(permittivity=1.0),
    }
    comp = gf.components.straight(length=1.68, width=0.48)
    stack = bf.StackDevice.from_layer_stack(
        comp,
        ls,
        materials,
        background="air",
        zrange_layers=["core"],
        boundary_offset=(0.60, -0.60, 0.56, -0.56, 0.53, -0.53),
        name="stack",
    )
    config = stack.get_simulation_config(
        time=220e-15,
        grid=fdtdx.core.grid.UniformGrid(spacing=0.08e-6),
        backend=jax.default_backend(),
        gradient_config=fdtdx.GradientConfig(method="checkpointed", num_checkpoints=16),
    )
    wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=wave,
        spectral_width=fdtdx.WaveCharacter(frequency=25e12),
    )
    volume, c_vol = stack.get_simulation_volume()
    source, c_src = stack.create_point_dipole_source_at(
        "o1",
        wave_character=wave,
        polarization="y",
        temporal_profile=pulse,
        port_offset=0.4,
        name="dipole",
    )
    probe, c_det = stack.create_field_detector_at(
        "o2",
        (0.2, 0.2),
        wavelength_um=1.55,
        components=("Ey", "Hz"),
        reduce_volume=False,
        port_offset=0.32,
        switch=fdtdx.OnOffSwitch(start_time=95e-15, interval=2),
        name="probe",
    )
    boundaries, c_bnd = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=5),
        volume,
    )

    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        [volume, stack, source, probe, *boundaries.values()],
        config,
        [*c_vol, *c_src, *c_det, *c_bnd],
        key,
    )
    params = {**params, **bf.init_stack_params(objects)}
    placed_probe = objects["probe"]

    f_lo = fdtdx.WaveCharacter(wavelength=1.80e-6)
    f_hi = fdtdx.WaveCharacter(wavelength=1.35e-6)

    def run_strip(thickness_um: jax.Array) -> bf.HarminvResult:
        p = {**params, "stack": {**params["stack"], "thickness/core": thickness_um}}
        arrays_in, objs_in, _ = bf.apply_stack(arrays, objects, p, key)
        _, final_arrays = fdtdx.run_fdtd(arrays=arrays_in, objects=objs_in, config=config, key=key, show_progress=False)
        return bf.harminv_detector(placed_probe, final_arrays, f_lo, f_hi, Q_thresh=3.0)

    res0 = run_strip(jnp.asarray(0.22, dtype=stack.dtype))
    assert int(jnp.sum(res0.valid)) >= 1
    assert res0.amplitude.shape == (res0.frequency.shape[0], 2, *placed_probe.grid_shape)
    assert float(jnp.linalg.norm(res0.amplitude[0, 0])) > 0.0
    assert float(jnp.linalg.norm(res0.amplitude[0, 1])) > 0.0

    # Pick strongest valid mode as reference frequency for thickness sensitivity
    best_idx = int(jnp.argmax(jnp.where(res0.valid, jnp.sum(jnp.abs(res0.amplitude[:, 0]) ** 2, axis=(1, 2, 3)), -1.0)))
    target_f = float(res0.frequency[best_idx])

    def tracked_freq_thz(thickness_um: jax.Array) -> jax.Array:
        return run_strip(thickness_um).select(target_f)["frequency"] / 1e12

    f_val, df_dt = bf.jit(jax.value_and_grad(tracked_freq_thz))(jnp.asarray(0.22, dtype=stack.dtype))
    h = 0.005  # um
    val_fn = bf.jit(tracked_freq_thz)
    fd_df_dt = (
        float(val_fn(jnp.asarray(0.22 + h, dtype=stack.dtype)))
        - float(val_fn(jnp.asarray(0.22 - h, dtype=stack.dtype)))
    ) / (2.0 * h)

    # Increasing core thickness increases n_eff -> decreases resonant frequency (negative df/dt)
    assert np.isfinite(float(f_val)) and np.isfinite(float(df_dt))
    assert float(df_dt) < 0.0
    assert float(df_dt) == pytest.approx(fd_df_dt, rel=0.1)
