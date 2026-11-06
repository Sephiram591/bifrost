"""bf.jit: jax.jit with the XLA option for gradients through fdtdx's checkpointed time loop."""

import re

import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import numpy as np

import bifrost as bf


def test_compiler_options_default_and_merge():
    assert bf.compiler_options() == {"xla_disable_hlo_passes": "simplify-conditional"}
    # disabled passes are combined (each once), other options are added or win over bifrost's
    assert bf.compiler_options({"xla_disable_hlo_passes": "foo, simplify-conditional"}) == {
        "xla_disable_hlo_passes": "simplify-conditional,foo"
    }
    assert bf.compiler_options({"xla_gpu_autotune_level": 0}) == {
        "xla_disable_hlo_passes": "simplify-conditional",
        "xla_gpu_autotune_level": 0,
    }
    options = bf.compiler_options()
    options["xla_gpu_autotune_level"] = 0
    assert dict(bf.COMPILER_OPTIONS) == {"xla_disable_hlo_passes": "simplify-conditional"}


def test_jit_call_and_decorator_forms():
    x = jnp.arange(4.0)
    expected = jax.jit(lambda x, n: jnp.sum(x**n), static_argnames="n")(x, 3)
    assert bf.jit(lambda x, n: jnp.sum(x**n), static_argnames="n")(x, 3) == expected

    @bf.jit
    def squares(x):
        return jnp.sum(x**2)

    @bf.jit(static_argnames="n")
    def powers(x, n):
        return jnp.sum(x**n)

    assert squares(x) == jnp.sum(x**2)
    assert powers(x, 3) == expected


def _loss(x):
    """A loop run like fdtdx runs its time loop: equinox's checkpointed while loop, fewer checkpoints than steps."""

    def body(carry):
        i, v = carry
        return i + 1, jnp.sin(v) * 1.01 + 0.1 * v

    _, v = eqxi.while_loop(lambda c: c[0] < 50, body, (0, x), max_steps=50, kind="checkpointed", checkpoints=5)
    return jnp.sum(v**2)


def _conditionals(jitted, x):
    return len(re.findall(r" conditional\(", jitted.lower(x).compile().as_text()))


def test_jit_keeps_the_backward_loop_conditional():
    x = jnp.linspace(0.0, 1.0, 1000)
    value_and_grad = jax.value_and_grad(_loss)
    plain, ours = jax.jit(value_and_grad), bf.jit(value_and_grad)
    for a, b in zip(jax.tree.leaves(plain(x)), jax.tree.leaves(ours(x)), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    n_plain, n_ours = _conditionals(plain, x), _conditionals(ours, x)
    if jax.default_backend() == "gpu":
        # XLA's GPU pipeline turns the U-turn conditional of the backward loop into a select; bf.jit keeps it
        assert n_ours > n_plain, (
            f"bf.jit kept {n_ours} conditionals, jax.jit {n_plain}: the XLA option no longer takes effect "
            "(renamed pass?) or XLA no longer removes this conditional"
        )
    else:
        assert n_ours >= n_plain
