"""The viewer's extra defaults against the installed fdtdx and the stack's factories.

``EXTRA_DEFAULTS`` (bifrost/viewer/sim.py) is a hand-written copy of defaults that
live in fdtdx and in ``StackDevice``'s ``create_*_at`` signatures. These tests
fail when fdtdx changes one, so the table cannot go stale unnoticed. When one
fails, follow the "Maintenance" section of docs/development.md.
"""

from __future__ import annotations

import inspect
from typing import Any

import fdtdx
import pytest
import pytreeclass as tc

import bifrost as bf

EXTRA_DEFAULTS = bf.viewer.sim.EXTRA_DEFAULTS


def _fdtdx_default(type_name: str, key: str) -> Any:
    """The default of field ``key`` of fdtdx's class ``type_name``."""
    cls = getattr(fdtdx, type_name)
    (field,) = [f for f in tc.fields(cls) if f.name == key]
    return field.default


def _plain(value: Any) -> Any:
    """Tuples as lists, the way the table (and the browser's JSON) holds them."""
    return list(value) if isinstance(value, tuple) else value


CASES = [
    (type_name, key, default)
    for type_name, extras in EXTRA_DEFAULTS.items()
    for key, default in extras.items()
    if key not in ("record_interval", *bf.viewer.sim._REQUIRED_EXTRAS)
]


@pytest.mark.parametrize(("type_name", "key", "default"), CASES, ids=[f"{t}.{k}" for t, k, _ in CASES])
def test_extra_default_matches_fdtdx(type_name: str, key: str, default: Any) -> None:
    assert _plain(_fdtdx_default(type_name, key)) == default


@pytest.mark.parametrize("type_name", [t for t, e in EXTRA_DEFAULTS.items() if "record_interval" in e])
def test_record_interval_default_is_fdtdxs_every_step_switch(type_name: str) -> None:
    """``record_interval`` becomes ``switch=OnOffSwitch(interval=n)``; 1 is fdtdx's default."""
    assert fdtdx.OnOffSwitch().interval == EXTRA_DEFAULTS[type_name]["record_interval"]
    assert any(f.name == "switch" for f in tc.fields(getattr(fdtdx, type_name)))


def test_required_extras_have_no_fdtdx_default() -> None:
    """The viewer supplies a default only where fdtdx has none (a dipole's polarization)."""
    for key in bf.viewer.sim._REQUIRED_EXTRAS:
        (type_name,) = [t for t, e in EXTRA_DEFAULTS.items() if key in e]
        assert not isinstance(_fdtdx_default(type_name, key), (str, int, float, bool, tuple)), (
            f"fdtdx now defaults {type_name}.{key}: drop it from _REQUIRED_EXTRAS"
        )


@pytest.mark.parametrize(("type_name", "key", "default"), CASES, ids=[f"{t}.{k}" for t, k, _ in CASES])
def test_extra_default_matches_the_factory(type_name: str, key: str, default: Any) -> None:
    """A factory that spells out the default (``reduce_volume``) must agree with the table."""
    factory = getattr(bf.stack.StackDevice, bf.viewer.sim.TYPES[type_name].method)
    parameter = inspect.signature(factory).parameters.get(key)
    if parameter is None or parameter.default in (inspect.Parameter.empty, None):
        pytest.skip("the factory leaves it to fdtdx")
    assert _plain(parameter.default) == default
