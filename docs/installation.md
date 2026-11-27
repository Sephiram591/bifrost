# Installation

## Requirements

- Python 3.12, 3.13 or 3.14.
- A fresh virtual environment (see the warning on [gdsfactoryx](https://github.com/Sephiram591/gdsfactoryx) vs upstream gdsfactory below).
- For simulations of any size, an accelerator that JAX supports, such as a GPU or TPU (see [JAX setup](#jax-setup)). A
  CPU is enough for the [Widget Demo](live-demo.md) (or standalone [Explorer demo](demo/index.html)), for building devices and for the viewer.

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
    code does not change: `import gdsfactory as gf` works unchanged.

## Check the installation

In the new environment, run:

```python
--8<-- "examples/check_install.py:check"
```

Expected output (the versions will differ, and `devices` lists your JAX backend's devices, here one CUDA GPU):

```text
python             3.12.14
bifrost-photonics  0.3.0
gdsfactoryx        9.51.0.post3
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
| `make test` | Run the Python tests (`pytest tests`) |
| `make test-js` | Run the viewer's JavaScript tests and typecheck |
| `make test-docs` | Check the docs and widget demo, including browser tests (run `playwright install chromium` once first) |
| `make test-hardware` | Run the tutorial's full simulations (requires a GPU) |
| `make viewer-js` | Rebuild the viewer bundle in `bifrost/viewer/static/` after editing `js/viewer/*.ts` |
| `make docs` | Build this documentation with strict checks |
| `make docs-serve` | Serve the documentation locally at `localhost:8080` |

See [Contributing and tests](development.md) for project conventions.
