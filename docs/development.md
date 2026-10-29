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

## The live demo

The [live demo](guide/viewer.md#live-demo) is the viewer's own front end playing back a precomputed result, with no Python.
Its pieces:

| Piece | Where |
| --- | --- |
| the simulation it shows | `examples/waveguide_bend.py` |
| the exporter (geometry, fields, 2D slices) | `bifrost/viewer/export.py` |
| the regeneration script | `scripts/build_demo.py` |
| the browser adapter that stands in for the kernel | `js/viewer/static.ts`, entry `js/viewer/demo.ts` |
| the page, bundle and data that are published | `docs/demo/` (`index.html`, `viewer-demo.js`, `viewer.css`, `data/`) |

To regenerate it after changing the example, the viewer or the export format, run on a GPU node, then commit `docs/demo/`:

```bash
make viewer-js                  # rebuilds the viewer and the demo bundle
python scripts/build_demo.py    # reruns the bend, writes docs/demo/data and docs/images/waveguide_bend_field.png
python scripts/capture_demo_poster.py   # the fallback picture, from the viewer's own save button
```

`docs/demo/data/manifest.json` records the source revision (and whether the working tree was dirty), the package versions, the
device, the simulation parameters, the measured transmission, wall time and memory, and the size of every file. The
documentation build only copies `docs/demo/`; it never reruns the simulation. The demo is kept small (about 17 MB, 15 of them the field movie): the
2D slices are stored at 0.1 µm spacing and the fields as 16-bit numbers against a per-file range.

