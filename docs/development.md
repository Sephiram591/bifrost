# Development

```bash
pip install -e ".[dev,docs]"
make test       # pytest tests
make test-js    # viewer JS tests and typecheck
make docs       # strict docs build
```

Simulation tests need `fdtdx` and a working jax; run them on a GPU node. To time the geometry steps and the FDTD
solve, run pytest with `FDTDX_TIMINGS=<name>` (or `--fdtdx-timings=<name>`); the plugin is `tests/fdtdx_timing_plugin.py`.

## Conventions

- Import bifrost as `bf` everywhere outside the package (`import bifrost as bf`); inside the package, use relative
  imports.
- Source of truth for the viewer's design is the [viewer spec](design/viewer-spec.md).

## Releases

Version in `pyproject.toml` and `bifrost/__init__.py`. Docs deploy from `main` through the `pages` workflow.
