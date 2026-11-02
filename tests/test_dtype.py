"""Tests for ``StackDevice(dtype=...)``: the precision (float32 or float64) the fdtdx simulation runs in."""

from __future__ import annotations

import functools
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

fdtdx = pytest.importorskip("fdtdx")

import gdsfactory as gf  # noqa: E402
import bifrost as bf  # noqa: E402
from bifrost.stack import _simulation_dtype  # noqa: E402
from gdsfactory.technology import LayerLevel, LayerStack  # noqa: E402

WAVE = fdtdx.WaveCharacter(wavelength=1.55e-6)
PADDING = (0.5, 0.3)
DTYPES = [
    pytest.param(jnp.float32, jnp.complex64, id="float32"),
    pytest.param(jnp.float64, jnp.complex128, id="float64"),
]


@pytest.fixture(autouse=True)
def pdk() -> Iterator[None]:
    gf.gpdk.PDK.activate()
    yield


def _component() -> gf.Component:
    return gf.components.straight(length=5)  # 5 um long, 500 nm wide, ports o1 and o2


_DEFAULT = object()  # "no dtype argument" (None is a value the device must refuse)


def _device(dtype: Any = _DEFAULT) -> bf.StackDevice:
    """The shortest complete device: a straight silicon waveguide in silica with both ports extended to the box."""
    layer_stack = LayerStack(
        layers={"core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=0.22, material="core", mesh_order=1)}
    )
    materials = {"core": fdtdx.Material(permittivity=3.48**2), "clad": fdtdx.Material(permittivity=1.444**2)}
    extra = {} if dtype is _DEFAULT else {"dtype": dtype}
    return bf.StackDevice.from_layer_stack(
        _component(), layer_stack, materials, background="clad", zrange_layers=["core"],
        extend_ports=(("o1", 1.5), ("o2", 1.5)), boundary_offset=(0, 0, 1.0, -1.0, 1.0, -1.0),
        name="stack", **extra,
    )


def _grid(spacing_um: float = 0.1) -> Any:
    return fdtdx.core.grid.UniformGrid(spacing=spacing_um * 1e-6)


# --------------------------------------------------------------- the setting


def test_the_default_is_float32() -> None:
    assert _device().dtype is jnp.float32


@pytest.mark.parametrize(
    "spelling, expected",
    [
        (jnp.float32, jnp.float32),
        (jnp.float64, jnp.float64),
        (np.float32, jnp.float32),
        (np.float64, jnp.float64),
        (np.dtype("float64"), jnp.float64),
        (jnp.dtype("float32"), jnp.float32),
        ("float32", jnp.float32),
        ("float64", jnp.float64),
    ],
)
def test_any_spelling_of_the_two_dtypes_is_stored_canonically(spelling: Any, expected: Any) -> None:
    assert _simulation_dtype(spelling) is expected
    assert _device(spelling).dtype is expected


@pytest.mark.parametrize("bad", [jnp.float16, jnp.int32, jnp.complex64, jnp.bfloat16, "bogus", None, 3.5])
def test_other_dtypes_are_refused(bad: Any) -> None:
    with pytest.raises(ValueError, match="float32 or jnp.float64"):
        _simulation_dtype(bad)
    with pytest.raises(ValueError, match="float32 or jnp.float64"):
        _device(bad)


def test_the_dtype_survives_pytree_round_trips() -> None:
    device = _device(jnp.float64)
    assert jax.tree_util.tree_map(lambda x: x, device).dtype is jnp.float64
    assert device.aset("name", "other").dtype is jnp.float64


# ------------------------------------------------------ get_simulation_config


@pytest.mark.parametrize("dtype, _complex", DTYPES)
def test_get_simulation_config_sets_the_dtype_and_passes_the_rest_on(dtype: Any, _complex: Any) -> None:
    device = _device(dtype)
    config = device.get_simulation_config(time=50e-15, grid=_grid(), backend=jax.default_backend(), courant_factor=0.5)
    assert isinstance(config, fdtdx.SimulationConfig)
    assert np.dtype(config.dtype) == np.dtype(dtype)
    assert config.time == 50e-15
    assert config.grid.spacing == pytest.approx(0.1e-6)
    assert config.courant_factor == 0.5
    assert config.backend == jax.default_backend()


def test_get_simulation_config_refuses_a_dtype_argument() -> None:
    with pytest.raises(TypeError, match="StackDevice"):
        _device().get_simulation_config(time=50e-15, grid=_grid(), dtype=jnp.float32)


def test_a_config_of_another_dtype_is_refused_when_the_device_is_placed() -> None:
    device = _device(jnp.float64)
    config = fdtdx.SimulationConfig(time=10e-15, grid=_grid(), backend=jax.default_backend(), dtype=jnp.float32)
    volume, constraints = device.get_simulation_volume()
    with pytest.raises(ValueError, match="get_simulation_config"):
        fdtdx.place_objects([volume, device], config, constraints, jax.random.PRNGKey(0))


def test_a_hand_built_config_of_the_same_dtype_is_accepted() -> None:
    for dtype in (jnp.float32, jnp.float64):
        device = _device(dtype)
        config = fdtdx.SimulationConfig(time=10e-15, grid=_grid(), backend=jax.default_backend(), dtype=dtype)
        volume, constraints = device.get_simulation_volume()
        objects, arrays, *_ = fdtdx.place_objects([volume, device], config, constraints, jax.random.PRNGKey(0))
        assert np.dtype(arrays.inv_permittivities.dtype) == np.dtype(dtype)


def test_a_float64_device_needs_jax_x64() -> None:
    device = _device(jnp.float64)
    config = device.get_simulation_config(time=10e-15, grid=_grid(), backend=jax.default_backend())
    device._check_precision(config)  # importing bifrost turned the 64-bit mode on
    with jax.enable_x64(False):
        assert not jax.config.jax_enable_x64
        with pytest.raises(RuntimeError, match="64-bit mode is off"):
            device._check_precision(config)
        _device(jnp.float32)._check_precision(  # a float32 device does not care
            fdtdx.SimulationConfig(time=10e-15, grid=_grid(), backend=jax.default_backend(), dtype=jnp.float32)
        )


# ----------------------------------------------------------------- parameters


@pytest.mark.parametrize("dtype, _complex", DTYPES)
def test_the_parameters_follow_the_device(dtype: Any, _complex: Any) -> None:
    device = _device(dtype)
    params = device.init_params()
    assert params and all(np.dtype(v.dtype) == np.dtype(dtype) for v in params.values())


def test_a_float64_device_never_packs_below_float64() -> None:
    device = _device(jnp.float64)
    packed = device.pack(_component())
    assert packed and all(v.dtype == jnp.float64 for v in packed.values())


def test_a_float64_parameter_gradient_is_float64() -> None:
    device = _device(jnp.float64)
    config = device.get_simulation_config(time=10e-15, grid=_grid(), backend=jax.default_backend())
    volume, constraints = device.get_simulation_volume()
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects([volume, device], config, constraints, key)
    params = {**params, **bf.init_stack_params(objects)}
    assert params["stack"]["thickness/core"].dtype == jnp.float64

    def total(thickness: Any) -> Any:
        p = {**params, "stack": {**params["stack"], "thickness/core": thickness}}
        new, _, _ = bf.apply_stack(arrays, objects, p, key)
        return jnp.sum(new.inv_permittivities)

    grad = jax.grad(total)(params["stack"]["thickness/core"])
    assert grad.dtype == jnp.float64 and np.isfinite(float(grad)) and float(grad) != 0.0


# ------------------------------------------------------------------ detectors


@pytest.mark.parametrize("dtype, complex_dtype", DTYPES)
def test_detectors_follow_the_device_dtype(dtype: Any, complex_dtype: Any) -> None:
    device = _device(dtype)
    frequency = WAVE.get_frequency()
    complex_ones = [
        device.create_mode_overlap_detector_at("o2", PADDING, (WAVE,))[0],
        device.create_phasor_detector_at("o2", PADDING, (WAVE,))[0],
        device.create_diffractive_detector_at("o2", PADDING, [frequency])[0],
    ]
    real_ones = [
        device.create_field_detector_at("o2", PADDING, wavelength_um=1.55)[0],
        device.create_energy_detector_at("o2", PADDING, wavelength_um=1.55)[0],
        device.create_poynting_flux_detector_at("o2", PADDING, wavelength_um=1.55)[0],
    ]
    for detector in complex_ones:
        assert np.dtype(detector.dtype) == np.dtype(complex_dtype), type(detector).__name__
    for detector in real_ones:
        assert np.dtype(detector.dtype) == np.dtype(dtype), type(detector).__name__


def test_an_explicit_detector_dtype_wins() -> None:
    device = _device(jnp.float64)
    assert device.create_field_detector_at("o2", PADDING, wavelength_um=1.55, dtype=jnp.float32)[0].dtype is jnp.float32
    assert device.create_phasor_detector_at("o2", PADDING, (WAVE,), dtype=jnp.complex64)[0].dtype is jnp.complex64


def test_sources_are_left_to_the_config() -> None:
    """fdtdx's sources take config.dtype themselves, so no dtype field is injected into them."""
    device = _device(jnp.float64)
    pulse = fdtdx.GaussianPulseProfile(center_wave=WAVE, spectral_width=fdtdx.WaveCharacter(frequency=1e13))
    source, _ = device.create_mode_plane_source_at("o1", PADDING, wave_character=WAVE, temporal_profile=pulse)
    assert not hasattr(source, "dtype")


# ------------------------------------------------------------- end to end


PML_FIELDS = ("pml_a_E", "pml_b_E", "inv_kappa_E", "pml_a_H", "pml_b_H", "inv_kappa_H")


@functools.lru_cache(maxsize=None)
def _simulate(dtype_name: str) -> dict[str, Any]:
    """A mode launched down a straight waveguide with PMLs, in one precision, with nothing but the device's dtype set:
    no ``enable_x64`` around the run, no dtype on any detector."""
    dtype = jnp.dtype(dtype_name).type
    device = _device(dtype)
    config = device.get_simulation_config(time=300e-15, grid=_grid(), backend=jax.default_backend())
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=WAVE, spectral_width=fdtdx.WaveCharacter(frequency=0.05 * WAVE.get_frequency())
    )
    te0 = dict(mode_index=0, filter_pol="te")
    volume, constraints = device.get_simulation_volume()
    objects = [volume, device]
    for obj, placing in (
        device.create_mode_plane_source_at(
            "o1", PADDING, wave_character=WAVE, temporal_profile=pulse, port_offset=-1.0, name="source", **te0
        ),
        device.create_mode_overlap_detector_at(
            "o1", PADDING, direction="in", wave_characters=(WAVE,), scaling_mode="pulse", port_offset=-0.5,
            name="in", **te0,
        ),
        device.create_mode_overlap_detector_at(
            "o2", PADDING, wave_characters=(WAVE,), scaling_mode="pulse", port_offset=-0.5, name="out", **te0
        ),
        device.create_phasor_detector_at("o2", PADDING, (WAVE,), port_offset=-0.5, name="phasor"),
        device.create_field_detector_at(
            "o2", PADDING, components=("Ex",), port_offset=-0.5, wavelength_um=1.55, name="movie",
            switch=fdtdx.OnOffSwitch(interval=100),
        ),
    ):
        objects.append(obj)
        constraints += placing
    boundaries, placing = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=5), volume)
    objects += boundaries.values()
    constraints += placing

    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
    arrays, objects, _ = bf.apply_stack(arrays, objects, {**params, **bf.init_stack_params(objects)}, key)
    pmls = [o for o in objects.object_list if isinstance(o, fdtdx.PerfectlyMatchedLayer)]
    pml_dtypes = {np.dtype(getattr(o, n).dtype) for o in pmls for n in PML_FIELDS}
    permittivity_dtype = np.dtype(arrays.inv_permittivities.dtype)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key, show_progress=False)
    states = arrays.detector_states
    a_in = np.ravel(objects["in"].compute_overlap(states["in"]))[0]
    a_out = np.ravel(objects["out"].compute_overlap(states["out"]))[0]
    return {
        "transmission": float(abs(a_out / a_in) ** 2),
        "pml_dtypes": pml_dtypes,
        "permittivity": permittivity_dtype,
        "E": np.dtype(arrays.fields.E.dtype),
        "H": np.dtype(arrays.fields.H.dtype),
        "overlap": np.dtype(states["in"]["phasor"].dtype),
        "phasor": np.dtype(states["phasor"]["phasor"].dtype),
        "movie": np.dtype(states["movie"]["fields"].dtype),
        "n_pml": len(pmls),
    }


@pytest.mark.parametrize("dtype, complex_dtype", DTYPES)
def test_a_simulation_runs_in_the_devices_dtype(dtype: Any, complex_dtype: Any) -> None:
    out = _simulate(np.dtype(dtype).name)
    real, cplx = np.dtype(dtype), np.dtype(complex_dtype)
    assert out["n_pml"] == 6
    assert out["permittivity"] == real
    assert out["pml_dtypes"] == {real}, "apply_stack puts the PML coefficients in the device's dtype"
    assert out["E"] == real and out["H"] == real
    assert out["overlap"] == cplx and out["phasor"] == cplx
    assert out["movie"] == real
    assert np.isfinite(out["transmission"]) and 0.9 < out["transmission"] < 1.05


@pytest.mark.parametrize("dtype, _complex", DTYPES)
def test_a_solve_with_pml_is_differentiable_in_the_devices_dtype(dtype: Any, _complex: Any) -> None:
    """``jax.grad`` of a detected energy with respect to the core thickness, through ``apply_stack`` and a checkpointed
    ``run_fdtd`` with PMLs, and no ``enable_x64(False)`` around any of it."""
    device = _device(dtype)
    config = device.get_simulation_config(
        time=120e-15, grid=_grid(), backend=jax.default_backend(),
        gradient_config=fdtdx.GradientConfig(method="checkpointed", num_checkpoints=8),
    )
    source, c1 = device.create_point_dipole_source_at("o1", wave_character=WAVE, polarization="y", port_offset=0.2)
    probe, c2 = device.create_energy_detector_at(
        "o2", PADDING, wavelength_um=1.55, reduce_volume=True, port_offset=0.2, name="probe"
    )
    volume, constraints = device.get_simulation_volume()
    boundaries, c3 = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=5), volume)
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        [volume, device, source, probe, *boundaries.values()], config, [*constraints, *c1, *c2, *c3], key
    )
    params = {**params, **bf.init_stack_params(objects)}

    def energy(thickness: Any) -> Any:
        p = {**params, "stack": {**params["stack"], "thickness/core": thickness}}
        new, objs, _ = bf.apply_stack(arrays, objects, p, key)
        _, final = fdtdx.run_fdtd(arrays=new, objects=objs, config=config, key=key, show_progress=False)
        return 1e24 * jnp.sum(final.detector_states["probe"]["energy"])  # SI energies are ~1e-25 J

    value, grad = jax.value_and_grad(energy)(jnp.asarray(0.22, dtype=dtype))
    assert np.isfinite(float(value)) and float(value) > 0
    assert np.isfinite(float(grad)) and float(grad) != 0.0
    assert np.dtype(grad.dtype) == np.dtype(dtype)


def test_float32_and_float64_give_the_same_physics() -> None:
    single, double = _simulate("float32"), _simulate("float64")
    assert abs(single["transmission"] - double["transmission"]) < 1e-3
