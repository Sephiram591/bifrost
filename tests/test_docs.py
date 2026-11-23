"""Checks that keep the documentation honest: its code runs, its links resolve, its install
commands match the package. They need no GPU. The full simulations behind the tutorial are
checked separately, on suitable hardware: ``BIFROST_HARDWARE=1 pytest tests/test_docs.py -k hardware``.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MARKDOWN = sorted(p for p in DOCS.rglob("*.md") if "_build" not in p.parts)
HARDWARE = os.environ.get("BIFROST_HARDWARE") == "1"
#: folders `make docs` generates before the build; links into them may be missing in a fresh checkout
GENERATED = (DOCS / "demo" / "notebook",)

SNIPPET = re.compile(r'--8<--\s+"([^":]+)(?::([\w-]+))?"')
FENCE = re.compile(r"^(```|~~~).*?^\1", re.S | re.M)
LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
HTML_REF = re.compile(r"""(?:src|href)=["']([^"']+)["']|url\(\s*["']?([^)"']+)["']?\s*\)""")
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.M)


def slug(text: str) -> str:
    """The anchor Markdown's default table of contents gives a heading."""
    text = re.sub(r"[`*_]|<[^>]+>|\{[^}]*\}", "", text)
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", text)


def prose(path: Path) -> str:
    """A page's text without code, where link-like syntax is not a link."""
    text = FENCE.sub("", path.read_text())
    return re.sub(r"`[^`\n]*`", "", text)


# ------------------------------------------------------------------ snippets


@pytest.mark.parametrize("page", MARKDOWN, ids=lambda p: str(p.relative_to(DOCS)))
def test_snippet_includes_point_at_real_example_sections(page: Path) -> None:
    for name, section in SNIPPET.findall(page.read_text()):
        source = ROOT / name
        assert source.is_file(), f"{page.name} includes {name}, which does not exist"
        if section:
            text = source.read_text()
            assert f"--8<-- [start:{section}]" in text and f"--8<-- [end:{section}]" in text, (
                f"{page.name} includes {name}:{section}, which has no such section"
            )


def test_no_unfilled_placeholders() -> None:
    """Measured numbers and outputs are pasted in from real runs; a leftover @@MARKER@@ means one was not."""
    for page in [ROOT / "README.md", *MARKDOWN]:
        assert not re.search(r"@@[A-Z_]+@@", page.read_text()), f"{page.relative_to(ROOT)} has an unfilled placeholder"


# ------------------------------------------------------------------ links


def _links(page: Path) -> list[tuple[str, Path]]:
    """Every link with the folder it resolves against.

    A Markdown link is relative to the source file, and so is a `src` or `href` in raw HTML: the build
    rewrites both (an iframe's `demo/index.html` in docs/live-demo.md becomes `../demo/index.html`
    in the page, and a `../demo/...` written there would climb out of the site). Only a CSS `url(...)`
    in a `style` attribute is left alone, so it is relative to the built page, whose URL is one folder
    deeper (`live-demo/`).
    """
    text = prose(page)
    built = page.parent if page.stem == "index" else page.parent / page.stem
    return [
        *((link, page.parent) for link in LINK.findall(text)),
        *((a, page.parent) for a, _ in HTML_REF.findall(text) if a),
        *((b, built) for _, b in HTML_REF.findall(text) if b),
    ]


@pytest.mark.parametrize("page", MARKDOWN, ids=lambda p: str(p.relative_to(DOCS)))
def test_internal_links_and_images_exist(page: Path) -> None:
    anchors_of: dict[Path, set[str]] = {}

    def anchors(path: Path) -> set[str]:
        if path not in anchors_of:
            anchors_of[path] = {slug(h) for h in HEADING.findall(FENCE.sub("", path.read_text()))}
        return anchors_of[path]

    for link, base in _links(page):
        if re.match(r"^(https?:|mailto:|data:|//)", link):
            continue
        target, _, anchor = link.partition("#")
        resolved = (base / target).resolve() if target else page
        if any(resolved.is_relative_to(folder) for folder in GENERATED):
            continue
        if resolved.is_dir():
            resolved = resolved / "index.md"
        assert resolved.exists(), f"{page.relative_to(DOCS)} links to {link}, which does not exist"
        if anchor and resolved.suffix == ".md":
            assert anchor in anchors(resolved), f"{page.relative_to(DOCS)} links to {link}: no such heading"


def test_every_page_is_in_the_navigation_or_a_forwarding_stub() -> None:
    config = (DOCS / "zensical.yml").read_text()
    listed = set(re.findall(r":\s+([\w/.-]+\.md)\s*$", config, re.M))
    for page in MARKDOWN:
        relative = str(page.relative_to(DOCS))
        if relative in listed:
            continue
        assert "(moved)" in page.read_text().splitlines()[0], f"{relative} is not in the nav and is not a forwarding stub"


# ------------------------------------------------------------------ installation


def test_no_obsolete_install_commands() -> None:
    """fdtdx is a required dependency: there is no [fdtdx] extra to install."""
    pattern = re.compile(r"pip install[^\n]*\[fdtdx\]")
    for path in [ROOT / "README.md", *MARKDOWN, *ROOT.glob("examples/**/*.py")]:
        assert not pattern.search(path.read_text()), f"{path.relative_to(ROOT)} installs an fdtdx extra"
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert '"fdtdx>=' in pyproject.split("[project.optional-dependencies]")[0], "fdtdx must be a required dependency"


def test_the_readme_quick_start_is_the_straight_waveguide_example() -> None:
    example = (ROOT / "examples" / "straight_waveguide.py").read_text()
    code = re.search(r"# --8<-- \[start:straight\]\n(.*?)# --8<-- \[end:straight\]", example, re.S).group(1).rstrip("\n")
    assert code in (ROOT / "README.md").read_text()


# ------------------------------------------------------------------ the examples run


def _run(path: str, *args: str, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    if not HARDWARE:  # the light examples never need a GPU; the hardware tests must use the one there is
        env["JAX_PLATFORMS"] = os.environ.get("JAX_PLATFORMS", "cpu")
    return subprocess.run(
        [sys.executable, str(ROOT / path), *args], capture_output=True, text=True, timeout=timeout, env=env,
        cwd=ROOT / "docs",  # nowhere near the example's own folder: no hidden imports from it
    )


LIGHT = [
    ("examples/check_install.py", ["devices", "bifrost", "fdtdx"]),
    ("examples/straight_waveguide.py", []),
    ("examples/guide/layer_stacks.py", []),
    ("examples/guide/simulation_setup.py", ["grid cells: (405, 95, 83)", "source", "out"]),
    ("examples/guide/options.py", ["default", "fast", "corner_lines=0"]),
    ("examples/guide/diagnostics.py", ["narrower than"]),
]


OUTPUT_PAGES = {  # the guide pages that show an example's printed output, in a ```text block
    "examples/guide/simulation_setup.py": "guide/simulation-setup.md",
    "examples/guide/options.py": "guide/options.md",
    "examples/guide/diagnostics.py": "guide/diagnostics.md",
}
TEXT_BLOCK = re.compile(r"^```text\n(.*?)^```", re.S | re.M)


@pytest.mark.parametrize(("path", "expected"), LIGHT, ids=[p for p, _ in LIGHT])
def test_instructional_examples_run_in_a_fresh_process(path: str, expected: list[str]) -> None:
    """Each example is complete: run from nowhere else, in a new interpreter, with no notebook state.

    Its page shows what it prints, line for line, so a changed number cannot go stale in the docs.
    """
    done = _run(path)
    assert done.returncode == 0, done.stderr[-2000:]
    for text in expected:
        assert text in done.stdout, f"{path} did not print {text!r}:\n{done.stdout}"
    if path in OUTPUT_PAGES:
        printed = done.stdout + done.stderr
        shown = TEXT_BLOCK.findall((DOCS / OUTPUT_PAGES[path]).read_text())
        assert shown, f"docs/{OUTPUT_PAGES[path]} shows no output block"
        for line in (line.strip() for block in shown for line in block.splitlines()):
            assert not line or line in printed, f"docs/{OUTPUT_PAGES[path]} shows {line!r}, which {path} does not print"


@pytest.mark.skipif(not HARDWARE, reason="runs full simulations; set BIFROST_HARDWARE=1 on a GPU node")
def test_hardware_the_bend_tutorial_reproduces_its_numbers(tmp_path: Path) -> None:
    done = _run("examples/waveguide_bend.py", "--figure", str(tmp_path / "field.png"))
    assert done.returncode == 0, done.stderr[-2000:]
    transmission = float(re.search(r"transmission:\s+([\d.]+)", done.stdout).group(1))
    assert 0.99 < transmission <= 1.005, done.stdout  # the tutorial reports 1.0008: the grid's error, not gain
    assert (tmp_path / "field.png").stat().st_size > 10_000


@pytest.mark.skipif(not HARDWARE, reason="runs full simulations; set BIFROST_HARDWARE=1 on a GPU node")
def test_hardware_the_linbo3_bend_example_runs(tmp_path: Path) -> None:
    done = _run("examples/linbo3_bend.py", "--figure", str(tmp_path / "field.png"))
    assert done.returncode == 0, done.stderr[-2000:]
    transmission = float(re.search(r"TE0 trans\.:\s+([\d.]+)", done.stdout).group(1))
    tm_conv = float(re.search(r"TE0->TM0 conv\.:\s+([\d.]+)", done.stdout).group(1))
    assert 0.90 < transmission < 0.98, done.stdout
    assert 0.01 < tm_conv < 0.10, done.stdout
    assert transmission + tm_conv <= 1.01, done.stdout
    assert (tmp_path / "field.png").stat().st_size > 10_000


@pytest.mark.skipif(not HARDWARE, reason="runs full simulations; set BIFROST_HARDWARE=1 on a GPU node")
def test_hardware_the_results_guide_reads_its_monitors() -> None:
    done = _run("examples/guide/results.py")
    assert done.returncode == 0, done.stderr[-2000:]
    assert abs(float(re.search(r"transmission:\s+([\d.]+)", done.stdout).group(1)) - 1.0) < 0.02
    assert abs(float(re.search(r"flux ratio:\s+([\d.]+)", done.stdout).group(1)) - 1.0) < 0.05


@pytest.mark.skipif(not HARDWARE, reason="runs full simulations; set BIFROST_HARDWARE=1 on a GPU node")
def test_hardware_the_gradient_example_agrees_with_finite_differences() -> None:
    done = _run("examples/guide/gradients.py")
    assert done.returncode == 0, done.stderr[-2000:]
    autodiff, finite = (float(v) for v in re.search(r"autodiff ([+-][\d.]+).*finite difference ([+-][\d.]+)", done.stdout).groups())
    assert autodiff == pytest.approx(finite, rel=0.2, abs=2e-3), done.stdout


@pytest.mark.skipif(not HARDWARE, reason="runs full simulations; set BIFROST_HARDWARE=1 on a GPU node")
def test_hardware_the_width_optimization_improves_the_bend() -> None:
    done = _run("examples/guide/inverse_design.py", "--iterations", "3", "--check", "0", "--figure", "", timeout=1800)
    assert done.returncode == 0, done.stderr[-2000:]
    found = re.search(r"transmission at ppw=[\d.]+: uniform ([\d.]+), optimized ([\d.]+)", done.stdout)
    uniform, optimized = (float(v) for v in found.groups())
    assert optimized > uniform, done.stdout


# ------------------------------------------------------------------ inline code & recipes


PY_BLOCK = re.compile(r"^```python[ \t]*\n(.*?)^```", re.S | re.M)


def _extract_snippet(name: str, section: str | None) -> str:
    text = (ROOT / name).read_text()
    if not section:
        return text
    pattern = re.compile(
        rf"^[^\n]*--8<-- \[start:{re.escape(section)}\]\s*\n(.*?)^[ \t]*# --8<-- \[end:{re.escape(section)}\]",
        re.S | re.M,
    )
    match = pattern.search(text)
    assert match is not None, f"{name} has no snippet section {section!r}"
    return textwrap.dedent(match.group(1))


def _expand_python_block(block: str) -> str:
    lines: list[str] = []
    for line in block.splitlines():
        match = SNIPPET.fullmatch(line.strip())
        if match:
            lines.append(_extract_snippet(match.group(1), match.group(2)).rstrip("\n"))
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("page", MARKDOWN, ids=lambda p: str(p.relative_to(DOCS)))
def test_python_code_blocks_parse_cleanly(page: Path) -> None:
    """Every ``python`` fenced block in docs/ (after expanding snippet includes) is valid Python syntax."""
    for index, block in enumerate(PY_BLOCK.findall(page.read_text()), start=1):
        code = _expand_python_block(block)
        ast.parse(code, filename=f"{page.relative_to(DOCS)}#block-{index}")


def _simulation_setup_device():
    import fdtdx
    import gdsfactory as gf
    from gdsfactory.technology import LayerLevel, LayerStack

    import bifrost as bf

    gf.gpdk.PDK.activate()
    component = gf.components.straight(length=5)
    layer_stack = LayerStack(
        layers={"core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=0.22, material="core", mesh_order=1)}
    )
    materials = {
        "core": fdtdx.Material(permittivity=3.48**2),
        "clad": fdtdx.Material(permittivity=1.444**2),
    }
    stack = bf.StackDevice(
        component, layer_stack, materials, background="clad", zrange_layers=["core"],
        extend_ports=(("o1", 2.0), ("o2", 2.0)), boundary_offset=(0.0, 0.0, 1.55, -1.55, 1.59, -1.59), name="stack",
    )
    wave = fdtdx.WaveCharacter(wavelength=1.55e-6)
    return stack, wave


def test_graded_grids_recipe_in_simulation_setup() -> None:
    """Exercise the graded-grid and port-source recipe in docs/guide/simulation-setup.md."""
    import fdtdx
    import jax
    import numpy as np

    import bifrost as bf

    stack, wave = _simulation_setup_device()
    half = 0.5 * stack.partial_real_shape[2] * 1e6
    z = bf.graded_edges(
        stack.z_center - half, stack.z_center + half, 0.06,
        fine=[(-0.05, 0.27, 0.02)],
        snap=[0.0, 0.22],
        uniform_ends=(10, 10),
    )
    config = stack.get_simulation_config(time=100e-15, grid=stack.rectilinear_grid(0.06, z_um=z))
    volume, constraints = stack.get_simulation_volume()
    source, placing = stack.create_mode_plane_source_at("o1", (0.5, 0.5), wave, grid=config)
    boundaries, b_placing = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig(
            thickness_grid_minx=15, thickness_grid_maxx=15,
            thickness_grid_miny=15, thickness_grid_maxy=15,
            thickness_grid_minz=10, thickness_grid_maxz=10,
        ),
        volume,
    )
    objects = [volume, stack, source, *boundaries.values()]
    placed, arrays, params, placed_config, _ = fdtdx.place_objects(
        objects, config, [*constraints, *placing, *b_placing], jax.random.PRNGKey(0)
    )
    arrays, placed, _ = bf.apply_stack(
        arrays, placed, {**params, **bf.init_stack_params(placed)}, jax.random.PRNGKey(0)
    )
    assert z[0] == pytest.approx(stack.z_center - half)
    assert z[-1] == pytest.approx(stack.z_center + half)
    assert np.allclose(np.diff(z[:11]), 0.06)
    assert np.allclose(np.diff(z[-11:]), 0.06)
    assert placed.volume.grid_shape == (150, 60, len(z) - 1)
    assert placed[source.name].grid_shape[0] == 1
    assert len(placed_config.resolved_grid.edges(2)) == len(z)
    assert np.isfinite(np.asarray(arrays.inv_permittivities)).all()


def test_automatic_mesh_recipe_in_simulation_setup() -> None:
    """Exercise the bf.mesh automatic-meshing recipe in docs/guide/simulation-setup.md."""
    import fdtdx
    import jax
    import numpy as np

    import bifrost as bf

    stack, wave = _simulation_setup_device()
    mesh = bf.mesh(
        stack, (1.50, 1.60),
        refine=bf.Refine(dilation={"WG": 0.45}),
    )
    summary = mesh.summary(time=500e-15)
    assert "cells" in summary
    config = stack.get_simulation_config(time=500e-15, grid=mesh.grid)
    volume, constraints = stack.get_simulation_volume()
    boundaries, b_placing = fdtdx.boundary_objects_from_config(mesh.boundary, volume)
    source, placing = stack.create_mode_plane_source_at("o1", (0.5, 0.5), wave, grid=config)
    objects = [volume, stack, source, *boundaries.values()]
    placed, arrays, params, _, _ = fdtdx.place_objects(
        objects, config, [*constraints, *b_placing, *placing], jax.random.PRNGKey(0)
    )
    arrays, placed, _ = bf.apply_stack(
        arrays, placed, {**params, **bf.init_stack_params(placed)}, jax.random.PRNGKey(0)
    )
    assert placed.volume.grid_shape == mesh.shape
    assert placed[source.name].grid_shape[0] == 1
    assert np.isfinite(np.asarray(arrays.inv_permittivities)).all()


def test_convergence_testing_guide_recipe() -> None:
    """Exercise the substrate-coarsening + dilation bf.mesh recipe in docs/guide/convergence-testing.md."""
    import fdtdx
    import gdsfactory as gf
    import jax
    import numpy as np
    from gdsfactory.technology import LayerLevel, LayerStack

    import bifrost as bf

    gf.gpdk.PDK.activate()
    component = gf.components.bend_circular(radius=5, width=0.5)
    layer_stack = LayerStack(
        layers={
            "box": LayerLevel(layer="WAFER", zmin=-3.0, thickness=3.0, material="silica", mesh_order=9),
            "core": LayerLevel(layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=0.22, material="si", mesh_order=1),
        }
    )
    materials = {
        "si": fdtdx.Material(permittivity=3.48**2),
        "silica": fdtdx.Material(permittivity=1.444**2),
        "air": fdtdx.Material(permittivity=1.0),
    }
    stack = bf.StackDevice(
        component, layer_stack, materials, background="air", zrange_layers=["core"],
        extend_ports=(("o1", 2.0), ("o2", 2.0)), boundary_offset=(1.5, 0.0, 0.0, -1.5, 1.4, -1.4), name="stack",
    )
    mesh = bf.mesh(
        stack, (1.50, 1.60), ppw=18,
        refine=bf.Refine(dilation={"WG": 0.35}, material_cells={"silica": 0.08}),
    )
    config = stack.get_simulation_config(time=300e-15, grid=mesh.grid)
    volume, constraints = stack.get_simulation_volume()
    boundaries, b_placing = fdtdx.boundary_objects_from_config(mesh.boundary, volume)
    source, src_placing = stack.create_mode_plane_source_at(
        "o1", (0.5, 0.5), wave_character=fdtdx.WaveCharacter(wavelength=1.55e-6), grid=config,
    )
    objects = [volume, stack, source, *boundaries.values()]
    placed, arrays, params, _, _ = fdtdx.place_objects(
        objects, config, [*constraints, *b_placing, *src_placing], jax.random.PRNGKey(0)
    )
    arrays, placed, _ = bf.apply_stack(
        arrays, placed, {**params, **bf.init_stack_params(placed)}, jax.random.PRNGKey(0)
    )
    assert placed.volume.grid_shape == mesh.shape
    assert placed[source.name].grid_shape[0] == 1
    assert np.isfinite(np.asarray(arrays.inv_permittivities)).all()


# ------------------------------------------------------------------ the build


@pytest.mark.skipif(shutil.which("zensical") is None, reason="zensical is not installed (pip install -e '.[docs]')")
def test_the_documentation_builds_strictly(tmp_path: Path) -> None:
    staged_docs = tmp_path / "docs"
    shutil.copytree(DOCS, staged_docs, ignore=shutil.ignore_patterns("_build"))
    config_text = (DOCS / "zensical.yml").read_text()
    config_text = re.sub(r"(?m)^docs_dir:\s*.*$", "docs_dir: docs", config_text)
    config_text = re.sub(r"(?m)^site_dir:\s*.*$", "site_dir: site", config_text)
    config_text = re.sub(
        r"(?m)^(\s*base_path:\s*).*$",
        lambda m: f'{m.group(1)}["{ROOT.as_posix()}", "{staged_docs.as_posix()}"]',
        config_text,
    )
    config_text = re.sub(
        r"(?m)^(\s*paths:\s*).*$",
        lambda m: f'{m.group(1)}["{ROOT.as_posix()}"]',
        config_text,
    )
    staged_config = tmp_path / "zensical.yml"
    staged_config.write_text(config_text)
    done = subprocess.run(
        ["zensical", "build", "--strict", "-f", str(staged_config)],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert done.returncode == 0, (done.stdout + done.stderr)[-3000:]
    assert (tmp_path / "site" / "index.html").is_file()


# ------------------------------------------------------------------ the Widget Demo notebook

NOTEBOOK = DOCS / "notebooks" / "widget_demo.py"


def test_the_widget_demo_page_embeds_the_exported_notebook() -> None:
    """`make demo-notebook` exports to docs/demo/notebook, and the page's iframe points there."""
    assert 'src="demo/notebook/index.html"' in (DOCS / "live-demo.md").read_text()
    assert "-o docs/demo/notebook" in (ROOT / "Makefile").read_text()


def test_the_widget_demo_page_holds_only_the_iframe_and_the_notebook_holds_the_text() -> None:
    """The page's text lives in the notebook, under the viewer, so the page itself is a title and the iframe."""
    body = (DOCS / "live-demo.md").read_text().split("---", 2)[2]
    assert re.sub(r"<div.*</div>", "", body, flags=re.S).strip() == ""
    source = NOTEBOOK.read_text()
    for heading in ("What the demo can and cannot do", "How it was made"):
        assert f"## {heading}" in source


def test_the_notebooks_site_links_point_at_real_pages_and_headings() -> None:
    """The notebook's links are not in a Markdown page, so the link check above never reads them.

    They are written from docs/demo/notebook/ (where the notebook is exported), so `../../x/` is the page docs/x.md.
    """
    hrefs = re.findall(r'href="\.\./\.\./([^"]+)"', NOTEBOOK.read_text())
    assert hrefs, "the notebook links to no page of the site"
    for href in hrefs:
        path, _, anchor = href.partition("#")
        page = DOCS / f"{path.rstrip('/')}.md"
        assert page.is_file(), f"the notebook links to ../../{href}, which is not a page"
        if anchor:
            headings = {slug(h) for h in HEADING.findall(FENCE.sub("", page.read_text()))}
            assert anchor in headings, f"the notebook links to ../../{href}: no such heading"


def test_the_notebook_opens_in_dark_mode() -> None:
    """`marimo export` reads the display theme from the script's PEP 723 header."""
    header = NOTEBOOK.read_text().split('"""', 1)[0]
    assert re.search(r"^# \[tool\.marimo\.display\]\n# theme = \"dark\"$", header, re.M)


def test_the_notebook_never_imports_bifrost_and_the_bundle_still_offers_what_it_calls() -> None:
    """A browser kernel has no jax, fdtdx or gdsfactory, so the notebook only replays the exported files."""
    source = NOTEBOOK.read_text()
    assert not re.search(r"^\s*(import|from)\s+bifrost", source, re.MULTILINE)
    assert "mountInto" in source and "mountInto" in (DOCS / "demo" / "viewer-demo.js").read_text()


@pytest.mark.skipif(shutil.which("marimo") is None, reason="marimo is not installed (pip install -e '.[docs]')")
def test_the_widget_demo_notebook_exports_to_wasm(tmp_path: Path) -> None:
    command = ["marimo", "export", "html-wasm", str(NOTEBOOK), "-o", str(tmp_path)]
    done = subprocess.run(
        [*command, "--mode", "run", "--no-show-code", "--no-sandbox", "-f"],
        check=False, capture_output=True, text=True, cwd=ROOT,
    )
    assert done.returncode == 0, (done.stdout + done.stderr)[-3000:]
    assert (tmp_path / "index.html").is_file() and (tmp_path / "assets").is_dir()
