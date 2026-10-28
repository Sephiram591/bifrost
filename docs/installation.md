# Installation

bifrost depends on [gdsfactoryx](https://github.com/Sephiram591/gdsfactoryx), the JAX port of gdsfactory. Its
distribution is still named `gdsfactory`, so it replaces an upstream gdsfactory installed in the same environment; use a
separate environment if you need both.

```bash
pip install "bifrost @ git+https://github.com/Sephiram591/bifrost.git"
```

For GPU simulations, install a CUDA-enabled `jax` as described in the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Development install

```bash
git clone https://github.com/Sephiram591/bifrost.git
cd bifrost
pip install -e ".[dev,docs]"
```

Always import the package as `bf`:

```python
import bifrost as bf
```

## Rebuilding the viewer bundle

The viewer's JavaScript is committed under `bifrost/viewer/static/`. After editing `js/viewer/*.ts`:

```bash
make viewer-js
```
