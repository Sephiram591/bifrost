"""bifrost: the bridge between gdsfactory(x) layouts and fdtdx simulations.

Import it as ``import bifrost as bf``::

    import bifrost as bf

    stack = bf.stack.StackDevice.from_layer_stack(component, layer_stack, ...)
    viewer = bf.view(stack)
"""

from __future__ import annotations

import importlib
from typing import Any

from . import geometry, jitting, stack, subpixel, timing
from .jitting import COMPILER_OPTIONS, compiler_options, jit
from .stack import (
    StackDevice,
    StackDiagnosticWarning,
    StackOptions,
    TopRoughness,
    apply_stack,
    init_stack_params,
)
from .timing import record_timings

__version__ = "0.1.0"

__all__ = [
    "COMPILER_OPTIONS",
    "StackDevice",
    "StackDiagnosticWarning",
    "StackOptions",
    "TopRoughness",
    "__version__",
    "apply_stack",
    "compiler_options",
    "geometry",
    "init_stack_params",
    "jit",
    "jitting",
    "record_timings",
    "stack",
    "subpixel",
    "timing",
    "view",
    "viewer",
]


def __getattr__(name: str) -> Any:
    """Lazily exposes ``viewer`` and ``view`` so anywidget/manifold3d load on use."""
    if name == "viewer":
        return importlib.import_module(".viewer", __name__)
    if name == "view":
        return importlib.import_module(".viewer", __name__).view
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
