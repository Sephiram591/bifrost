"""bifrost: the bridge between gdsfactory(x) layouts and fdtdx simulations.

Import it as ``import bifrost as bf``::

    import bifrost as bf

    stack = bf.stack.StackDevice.from_layer_stack(component, layer_stack, ...)
    viewer = bf.view(stack)
"""

from __future__ import annotations

import importlib
from typing import Any

from . import geometry, grids, jitting, meshing, stack, subpixel, timing
from .grids import AxisEdges, graded_edges, lattice_edges
from .jitting import COMPILER_OPTIONS, compiler_options, jit
from .meshing import CoarseVertices, Mesh, Refine, mesh
from .stack import (
    StackDevice,
    StackDiagnosticWarning,
    StackOptions,
    TopRoughness,
    apply_stack,
    init_stack_params,
)
from .timing import record_timings

__version__ = "0.2.0"

__all__ = [
    "COMPILER_OPTIONS",
    "AxisEdges",
    "CoarseVertices",
    "Mesh",
    "Refine",
    "StackDevice",
    "StackDiagnosticWarning",
    "StackOptions",
    "TopRoughness",
    "__version__",
    "apply_stack",
    "compiler_options",
    "geometry",
    "graded_edges",
    "grids",
    "init_stack_params",
    "jit",
    "jitting",
    "lattice_edges",
    "mesh",
    "meshing",
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
