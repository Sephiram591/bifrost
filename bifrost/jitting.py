"""``jax.jit`` with the XLA options that differentiating through fdtdx's checkpointed time loop needs.

fdtdx runs its time loop as equinox's checkpointed ``while_loop``. Its backward pass alternates, in one loop, between
recomputing forward steps from a checkpoint and making a "U-turn" (one step's VJP), chosen by a ``lax.cond``. When a
gradient and its forward pass are compiled into one program, XLA's ``simplify-conditional`` pass can replace that
conditional by computing both branches and selecting one, since neither branch holds an op it counts as expensive. The
step's VJP then runs on every iteration of the backward loop, recomputations included. On an NVIDIA L40S, the
inverse-design example (50 optimization steps on a 60 nm grid, 4371 time steps each) took 4848 s with ``jax.jit`` and
3146 s with this pass disabled, with the same results.

:func:`jit` is ``jax.jit`` with :data:`COMPILER_OPTIONS` merged into its ``compiler_options``: the options apply to
the functions it compiles only, not to the rest of the process (unlike ``XLA_FLAGS``)::

    value_and_grad = bf.jit(jax.value_and_grad(objective))

For another jit wrapper, pass :func:`compiler_options` (or :data:`COMPILER_OPTIONS`) to it.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from typing import Any

import jax

__all__ = ["COMPILER_OPTIONS", "compiler_options", "jit"]

#: The XLA compiler options :func:`jit` adds: ``simplify-conditional`` disabled, so the backward loop of fdtdx's
#: checkpointed time loop keeps its conditional (see the module docstring).
COMPILER_OPTIONS: Mapping[str, Any] = {"xla_disable_hlo_passes": "simplify-conditional"}


def compiler_options(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """:data:`COMPILER_OPTIONS` merged with ``extra``.

    Args:
        extra: compiler options of your own (``jax.jit``'s ``compiler_options``). They win over bifrost's, except
            for ``xla_disable_hlo_passes``, whose comma-separated pass names are combined with bifrost's.

    Returns:
        A new dict of compiler options.
    """
    out = dict(COMPILER_OPTIONS)
    for key, value in (extra or {}).items():
        if key == "xla_disable_hlo_passes" and key in out:
            names = [*str(out[key]).split(","), *str(value).split(",")]
            value = ",".join(dict.fromkeys(n.strip() for n in names if n.strip()))
        out[key] = value
    return out


def jit(fun: Callable[..., Any] | None = None, /, **kwargs: Any) -> Any:
    """``jax.jit`` with :data:`COMPILER_OPTIONS`; it takes the same arguments.

    Use it for functions that differentiate through ``fdtdx.run_fdtd`` (a ``jax.value_and_grad``, ``jax.grad`` or
    ``jax.vjp`` of an objective), as a call or as a decorator, with or without arguments::

        value_and_grad = bf.jit(jax.value_and_grad(objective))

        @bf.jit(static_argnames="steps")
        def step(params, arrays, steps): ...

    Args:
        fun: the function to compile; omitted when ``bf.jit(...)`` is used as a decorator with arguments.
        **kwargs: ``jax.jit``'s keyword arguments. ``compiler_options`` are merged with bifrost's
            (:func:`compiler_options`).

    Returns:
        The compiled function (``jax.jit``'s result), or a decorator when ``fun`` is omitted.
    """
    kwargs["compiler_options"] = compiler_options(kwargs.get("compiler_options"))
    if fun is None:
        return functools.partial(jit, **kwargs)
    return jax.jit(fun, **kwargs)
