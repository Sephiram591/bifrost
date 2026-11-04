# Installation

## Requirements

- Python 3.12, 3.13 or 3.14.
- A fresh virtual environment (see the note on gdsfactory below).
- For simulations of any size, a JAX backend with an accelerator (a GPU or TPU). bifrost runs on whichever backend your
  JAX installation provides (see [JAX setup](#jax-setup)). Everything on this page also works on a CPU, which is enough
  for the [live demo](guide/viewer.md#live-demo), for building devices and for the viewer.

## Basic installation

```bash
pip install bifrost-photonics
```

To install the latest development version from GitHub
instead:

```bash
pip install "bifrost-photonics @ git+https://github.com/Sephiram591/bifrost.git"
```

!!! warning "gdsfactoryx cannot share an environment with upstream gdsfactory"
    bifrost builds on [gdsfactoryx](https://github.com/Sephiram591/gdsfactoryx), the JAX port of gdsfactory, which pip
    installs from PyPI as the `gdsfactoryx` distribution. It provides the same `gdsfactory` import package as upstream
    gdsfactory, so the two overwrite each other's files: installing upstream gdsfactory in the same environment breaks
    bifrost. Keep bifrost in its own environment, and remove an existing upstream gdsfactory first
    (`pip uninstall -y gdsfactory`). If `import gdsfactory` raises an error saying both distributions are installed, run
    `pip uninstall -y gdsfactory gdsfactoryx && pip install gdsfactoryx`. In your code nothing changes:
    `import gdsfactory as gf` works as before.

## Check the installation

Run this in the environment you installed into. It prints the versions bifrost depends on and the devices JAX can use.

```python
--8<-- "examples/check_install.py:check"
```

A working installation prints something like this (the versions will differ, and `devices` lists the devices of your
JAX backend, here one CUDA GPU):

```text
python             3.12.14
bifrost-photonics  0.1.0
gdsfactoryx        9.51.0
fdtdx              0.6.3
jax                0.11.2
jaxlib             0.11.2
devices            [CudaDevice(id=0)]
```

## JAX setup

bifrost works with any backend JAX supports. Installing bifrost installs JAX's default build, which runs on the CPU. To
run on a GPU, a TPU or another accelerator, set up JAX for your hardware as described in the
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

See [Contributing and tests](development.md) for the conventions.
