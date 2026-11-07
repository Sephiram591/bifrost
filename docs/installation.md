# Installation

## Requirements

- Python 3.12, 3.13 or 3.14.
- A fresh virtual environment (see the warning on gdsfactory below).
- For simulations of any size, an accelerator that JAX supports, such as a GPU or TPU (see [JAX setup](#jax-setup)). A
  CPU is enough for the [live demo](live-demo.md), for building devices and for the viewer.

## Basic installation

```bash
pip install bifrost-photonics
```

The latest development version, from GitHub:

```bash
pip install "bifrost-photonics @ git+https://github.com/Sephiram591/bifrost.git"
```

!!! warning "gdsfactoryx cannot share an environment with upstream gdsfactory"
    bifrost builds on [gdsfactoryx](https://github.com/Sephiram591/gdsfactoryx), the JAX port of gdsfactory, installed
    from PyPI as the `gdsfactoryx` distribution. It provides the same `gdsfactory` import package as upstream gdsfactory,
    so the two overwrite each other's files and upstream gdsfactory in the same environment breaks bifrost. Remove an
    existing upstream gdsfactory first (`pip uninstall -y gdsfactory`). If `import gdsfactory` raises an error saying
    both distributions are installed, run `pip uninstall -y gdsfactory gdsfactoryx && pip install gdsfactoryx`. Your
    code does not change: `import gdsfactory as gf` works as before.

## Check the installation

In the new environment, run:

```python
--8<-- "examples/check_install.py:check"
```

Expected output (the versions will differ, and `devices` lists your JAX backend's devices, here one CUDA GPU):

```text
python             3.12.14
bifrost-photonics  0.1.0
gdsfactoryx        9.51.0.post2
fdtdx              0.6.3
jax                0.11.2
jaxlib             0.11.2
devices            [CudaDevice(id=0)]
```

## JAX setup

bifrost works with any JAX backend and needs no backend-specific configuration. pip installs JAX's default build, which
runs on the CPU. For a GPU, TPU or other accelerator, set up JAX for your hardware with the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html), then run the
[installation check](#check-the-installation) again: `devices` should list your accelerator.

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

Conventions: [Contributing and tests](development.md).
