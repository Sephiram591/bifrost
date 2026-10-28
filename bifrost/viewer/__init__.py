"""The bifrost viewer (spec ``docs/design/viewer-spec.md``).

The v1 deliverable is an ``anywidget`` (spec D2), returned by ``view(stack)``
(spec D7). The analytic geometry builder builds the 2D level regions
(:mod:`bifrost.viewer.geometry`, milestone M0, spec D8) and the 3D level
solids, their ``mesh_order`` painting and the port glyphs
(:mod:`bifrost.viewer.solids`, milestone M1, spec D3). The explorer and the
3D viewport are M2 (spec R1-R3): :mod:`bifrost.viewer.state` holds the shared
state, :mod:`bifrost.viewer.payload` the wire format and
:mod:`bifrost.viewer.widget` the app shell. The sources, detectors and the
run are M4 (spec R12-R14, D11, D15): :mod:`bifrost.viewer.sim` holds the
declarative specs and the preparation path, and ``widget`` exposes
``.objects``/``.config``/``.constraints`` (mode 2) and ``.run()`` (mode 1, on a
background thread with the setup locked while it runs).

Nothing here is imported by ``import bifrost`` (spec C3); the extra
dependencies (``manifold3d``, ``anywidget``) ship with bifrost (spec C6).
``view`` is reached lazily, so importing this package does not import them.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["view"]

_SUBMODULES = frozenset(
    {"adopt", "fields", "geometry", "palette", "payload", "section", "sim", "solids", "state", "widget"}
)


def __getattr__(name: str) -> Any:
    """Lazily exposes :func:`bifrost.viewer.widget.view` and the submodules (spec C3)."""
    if name == "view":
        return importlib.import_module(".widget", __name__).view
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
