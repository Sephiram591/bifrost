"""Checks that keep the documentation honest: its code runs, its links resolve, its install
commands match the package. They need no GPU. The full simulations behind the tutorial are
checked separately, on suitable hardware: ``BIFROST_HARDWARE=1 pytest tests/test_docs.py -k hardware``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MARKDOWN = sorted(p for p in DOCS.rglob("*.md") if "_build" not in p.parts)
HARDWARE = os.environ.get("BIFROST_HARDWARE") == "1"

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

    A Markdown link is relative to the source file (the build rewrites it); a path in raw HTML is
    left alone, so it is relative to the built page, whose URL is one folder deeper (`guide/viewer/`).
    """
    text = prose(page)
    built = page.parent if page.stem == "index" else page.parent / page.stem
    return [
        *((link, page.parent) for link in LINK.findall(text)),
        *((a or b, built) for a, b in HTML_REF.findall(text)),
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
    ("examples/guide/layer_stacks.py", ["box (um): [7.0, 3.5, 3.22]"]),
    ("examples/guide/simulation_setup.py", ["grid cells: (180, 74, 68)", "source", "out"]),
    ("examples/guide/options.py", ["default", "fast", "corner_lines=0"]),
    ("examples/guide/diagnostics.py", ["narrower than"]),
]


@pytest.mark.parametrize(("path", "expected"), LIGHT, ids=[p for p, _ in LIGHT])
def test_instructional_examples_run_in_a_fresh_process(path: str, expected: list[str]) -> None:
    """Each example is complete: run from nowhere else, in a new interpreter, with no notebook state."""
    done = _run(path)
    assert done.returncode == 0, done.stderr[-2000:]
    for text in expected:
        assert text in done.stdout, f"{path} did not print {text!r}:\n{done.stdout}"


@pytest.mark.skipif(not HARDWARE, reason="runs full simulations; set BIFROST_HARDWARE=1 on a GPU node")
def test_hardware_the_bend_tutorial_reproduces_its_numbers(tmp_path: Path) -> None:
    done = _run("examples/waveguide_bend.py", "--figure", str(tmp_path / "field.png"))
    assert done.returncode == 0, done.stderr[-2000:]
    transmission = float(re.search(r"transmission:\s+([\d.]+)", done.stdout).group(1))
    assert 0.99 < transmission <= 1.001, done.stdout  # the tutorial reports 0.9991
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


# ------------------------------------------------------------------ the build


@pytest.mark.skipif(shutil.which("zensical") is None, reason="zensical is not installed (pip install -e '.[docs]')")
def test_the_documentation_builds_strictly(tmp_path: Path) -> None:
    done = subprocess.run(
        ["zensical", "build", "--strict", "-f", str(DOCS / "zensical.yml")], capture_output=True, text=True, cwd=ROOT,
    )
    assert done.returncode == 0, (done.stdout + done.stderr)[-3000:]
