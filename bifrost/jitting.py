"""`jax.jit` with the XLA options needed to differentiate through fdtdx's checkpointed time loop.

fdtdx runs its time loop using equinox's checkpointed `while_loop`. Its backward pass alternates, in one loop,
between recomputing forward steps from a checkpoint and taking a reverse step (one step's VJP), chosen by a
`lax.cond`. When a gradient and its forward pass are compiled into one program, XLA's `simplify-conditional` pass can
replace that conditional by evaluating both branches and selecting one, since neither branch contains an operation it
treats as expensive. The step's VJP then runs on every iteration of the backward loop, recomputations included. On an
NVIDIA L40S, the inverse-design example (50 optimization steps on a 60 nm grid, 4371 time steps each) runs in 3146 s
with this pass disabled vs 4848 s with plain `jax.jit`, with identical results.

`jit` is `jax.jit` with `COMPILER_OPTIONS` merged into its `compiler_options`, scoping the flags to the compiled
function rather than the whole process (unlike `XLA_FLAGS`):

```python
value_and_grad = bf.jit(jax.value_and_grad(objective))
```

For another `jit` wrapper, pass `compiler_options` (or `COMPILER_OPTIONS`) to it.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from typing import Any

import jax

__all__ = ["COMPILER_OPTIONS", "compiler_options", "jit"]

#: XLA compiler options added by `jit`: disables `simplify-conditional` so the backward loop of fdtdx's
#: checkpointed time loop keeps its conditional.
COMPILER_OPTIONS: Mapping[str, Any] = {"xla_disable_hlo_passes": "simplify-conditional"}


def compiler_options(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """`COMPILER_OPTIONS` merged with `extra`.

    Args:
        extra: Custom compiler options (passed to `jax.jit`'s `compiler_options`). They take precedence over
            bifrost's defaults, except for `xla_disable_hlo_passes`, whose comma-separated pass names are combined.

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
    """`jax.jit` with `COMPILER_OPTIONS`; accepts the same arguments.

    Use it for functions that differentiate through `fdtdx.run_fdtd` (`jax.value_and_grad`, `jax.grad`, or
    `jax.vjp` of an objective), as a call or as a decorator, with or without arguments:

    ```python
    value_and_grad = bf.jit(jax.value_and_grad(objective))

    @bf.jit(static_argnames="steps")
    def step(params, arrays, steps): ...
    ```

    Args:
        fun: Function to compile; omitted when `bf.jit(...)` is used as a decorator with arguments.
        **kwargs: Forwarded to `jax.jit`. Any `compiler_options` are merged with `compiler_options()`.

    Returns:
        The compiled function, or a decorator when `fun` is omitted.
    """
    kwargs["compiler_options"] = compiler_options(kwargs.get("compiler_options"))
    if fun is None:
        return functools.partial(jit, **kwargs)
    return jax.jit(fun, **kwargs)
