# Installation

## Requirements

- Python 3.12, 3.13 or 3.14.
- A fresh virtual environment (see the note on gdsfactory below).
- For simulations of any size, a CUDA GPU. Everything on this page also works on a CPU, which is enough for the
  [live demo](guide/viewer.md#live-demo), for building devices and for the viewer.

## Basic installation

```bash
python -m venv bifrost-env
source bifrost-env/bin/activate
pip install "bifrost @ git+https://github.com/Sephiram591/bifrost.git"
```

This installs everything bifrost needs, including [fdtdx](https://github.com/ymahlau/fdtdx), the FDTD solver it drives.
fdtdx is a required dependency, not an extra: there is no `[fdtdx]` option to add, and a plain `pip install` is
complete. The viewer's `anywidget` and `manifold3d` are required too.

!!! warning "gdsfactoryx replaces upstream gdsfactory"
    bifrost builds on [gdsfactoryx](https://github.com/Sephiram591/gdsfactoryx), the JAX port of gdsfactory. Its
    distribution is still named `gdsfactory`, so installing bifrost replaces an upstream gdsfactory in the same
    environment, and installing upstream gdsfactory afterwards replaces gdsfactoryx and breaks bifrost. Keep bifrost in
    its own environment. In your code nothing changes: `import gdsfactory as gf` works as before, and you import
    bifrost as `bf`:

    ```python
    import bifrost as bf
    ```

## Check the installation

Run this in the environment you installed into. It prints the versions bifrost depends on and the devices JAX can use.

```python
--8<-- "examples/check_install.py:check"
```

A working GPU installation prints something like this (the versions will differ):

```text
python     3.12.14
bifrost    0.1.0
gdsfactory 9.51.0
fdtdx      0.6.3
jax        0.11.2
jaxlib     0.11.2
devices    [CudaDevice(id=0)]
```

`gdsfactory` reports the gdsfactoryx version here. If `devices` shows `[CpuDevice(id=0)]` on a machine with a GPU, see
[GPU setup](#gpu-setup).

## GPU setup

The default `jax` wheel runs on the CPU. To use an NVIDIA GPU, install the CUDA build that matches your driver by
following the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html), for example:

```bash
pip install -U "jax[cuda12]"
```

and run the [installation check](#check-the-installation) again: `devices` should now list a `CudaDevice`. bifrost
itself needs no GPU-specific configuration. On a shared cluster, request the GPU through your scheduler (for example
`srun --gres=gpu:1 python examples/waveguide_bend.py`) instead of running on a login node.

## Contributor installation

```bash
git clone https://github.com/Sephiram591/bifrost.git
cd bifrost
pip install -e ".[dev,docs]"
```

| Command | What it does |
| --- | --- |
| `make test` | the Python tests (`pytest tests`) |
| `make test-js` | the viewer's JavaScript tests and typecheck |
| `make test-docs` | checks of the docs and the live demo, including browser tests (run `playwright install chromium` once first) |
| `make test-hardware` | the tutorial's full simulations (needs a GPU) |
| `make viewer-js` | rebuilds the viewer bundle in `bifrost/viewer/static/` after editing `js/viewer/*.ts` |
| `make docs` | builds this documentation with strict checks |
| `make docs-serve` | serves it at `localhost:8080` |

See [Contributing and tests](development.md) for the conventions.
