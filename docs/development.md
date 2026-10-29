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

## Maintenance

### Keeping the viewer's defaults in line with fdtdx

The viewer's source and detector cards show a default in every field a spec
leaves unset, and a field cleared in the browser returns to it. The defaults
come from three places; fdtdx changes can invalidate the third.

| Field | Default | Source of truth |
| --- | --- | --- |
| `name` | `<port>_<type>` | `ObjectSpec.name` in `bifrost/viewer/sim.py` |
| `wavelengths`, `port_padding`, and every Simulation-tab field | the settings | `SimSettings` in `bifrost/viewer/sim.py` |
| each type's extras (`mode_index`, `filter_pol`, `scaling_mode`, `dft_subsample`, `reduce_volume`, `record_interval`, `components`, `aggregate`, `polarization`, `source_type`) | fdtdx's own default | `EXTRA_DEFAULTS` in `bifrost/viewer/sim.py` |

`EXTRA_DEFAULTS` is a hand-written copy of fdtdx's class defaults and of
`StackDevice`'s factory defaults (for example `reduce_volume` is `True` for the
Poynting-flux detector and `False` for the others). After upgrading fdtdx, or
when adding a type or an extra:

1. Check the latest fdtdx conventions: read the class of each viewer type in
   the installed `fdtdx` (`fdtdx/objects/sources`, `fdtdx/objects/detectors`)
   and the matching `create_*_at` signature in `bifrost/stack.py`. Look for
   renamed, added, removed or re-defaulted fields, and for new fdtdx
   conventions such as a different scaling mode or sampling parameter.
2. Update `EXTRA_DEFAULTS`, the type's `extra` tuple in `TYPES`, and, if a field
   has no fdtdx default (like a dipole's `polarization`), `_REQUIRED_EXTRAS`.
3. Run `pytest tests/test_viewer_defaults.py`. It compares every entry against
   the installed fdtdx and fails on a mismatch, so a stale table is caught.
4. `scaling_mode` is special: the browser shows `"pulse"` instead of the table's
   value when the settings' pulse width is positive (the kernel makes the same
   substitution when it builds the objects). Change both places together.
5. Rebuild the viewer bundle only if the browser code changed (`make viewer-js`).

A field whose default is `None` (`filter_pol`, `aggregate`) shows as empty:
empty means "off" or "unrestricted", and clearing the field leaves it empty.

