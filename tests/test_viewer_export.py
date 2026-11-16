"""The static demo's exporter (bifrost/viewer/export.py): the files the browser adapter reads."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import bifrost as bf

sys.path.insert(0, str(Path(__file__).parent))
from test_viewer_fields import _detector_viewer, _run  # noqa: E402

export = bf.viewer.export


# ------------------------------------------------------------------ helpers


def test_stops_are_evenly_spaced_and_include_the_anchor() -> None:
    stops = export._stops(-1.59, 1.81, 0.11, 0.1)
    assert stops[0] == pytest.approx(-1.59) and stops[-1] == pytest.approx(1.81)
    assert 0.11 in stops
    assert np.allclose(np.diff(stops), 0.1)
    assert export._stops(0.0, 1.0, 0.5, 0.3) == pytest.approx([0.2, 0.5, 0.8])


def test_quantize_round_trips_within_one_step_of_the_range() -> None:
    values = np.linspace(-3.0, 5.0, 1000, dtype=np.float32)
    data, low, high = export._quantize(values)
    decoded = low + np.frombuffer(data, dtype="<u2") * (high - low) / 65535
    assert (low, high) == (-3.0, 5.0)
    assert np.abs(decoded - values).max() <= (high - low) / 65535


def test_quantize_handles_a_constant_field() -> None:
    data, low, high = export._quantize(np.zeros(4, dtype=np.float32))
    assert (low, high) == (0.0, 0.0) and np.frombuffer(data, dtype="<u2").tolist() == [0, 0, 0, 0]


def test_write_json_and_quantize_refuse_non_finite_values(tmp_path: Path) -> None:
    """E-BF-20: non-finite values are refused rather than emitting invalid JSON."""
    with pytest.raises(ValueError):
        export._write_json(tmp_path / "bad.json", {"metric": float("nan")})
    with pytest.raises(ValueError):
        export._write_json(tmp_path / "bad_inf.json", {"metric": float("inf")})
    with pytest.raises(ValueError, match="non-finite"):
        export._quantize(np.array([0.0, np.nan], dtype=np.float32))


def _entry(real, imag=None, components=("Ex", "Ey", "Ez")) -> dict:
    frames, comps, rows, cols = real.shape
    entry = {
        "id": "d", "name": "field", "components": list(components), "grid": [rows, cols],
        "real": np.ascontiguousarray(real, dtype="<f4").data,
        "freqs": [1.0] * frames,
    }
    if imag is not None:
        entry["imag"] = np.ascontiguousarray(imag, dtype="<f4").data
    return entry


def test_the_peak_frame_is_the_most_energetic() -> None:
    real = np.zeros((4, 1, 3, 3), dtype=np.float32)
    real[2] = 5.0
    assert export._peak_frame(_entry(real, components=("Ex",))) == 2


def test_unchecked_hides_the_named_monitor_in_its_spec_and_its_glyph() -> None:
    none = {"port": "None", "type": "PhasorDetector"}
    wire = {
        "sources": [{"id": "s", "port": "None", "type": "ModePlaneSource", "params": {"name": "source", "visible": True}}],
        "detectors": [
            {"id": "a", **none, "params": {"name": "field", "visible": True}},
            {"id": "b", **none, "params": {"name": "movie"}},
            {"id": "c", "port": "o1", "type": "PhasorDetector", "params": {}},  # no name param: o1_PhasorDetector
        ],
        "glyphs": [{"name": n, "visible": True} for n in ("source", "field", "movie", "o1_PhasorDetector")],
    }
    export._uncheck(wire, {"field", "o1_PhasorDetector"})
    assert [d["params"].get("visible", True) for d in wire["detectors"]] == [False, True, False]
    assert [g["visible"] for g in wire["glyphs"]] == [True, False, True, False]
    assert wire["sources"][0]["params"]["visible"] is True


# ------------------------------------------------------------------ a real export


def test_export_demo_writes_every_file_the_browser_reads(tmp_path: Path) -> None:
    viewer = _detector_viewer()
    result = _run(viewer)
    viewer.load_result(result.objects, result.arrays, result.config, stack=viewer.stack)
    manifest = export.export_demo(
        viewer, tmp_path, section_step_um=0.5,
        field_defaults={"o1_PhasorDetector": {"field_component": "|E|", "field_visible": False}},
        unchecked=("o1_PhasorDetector",), manifest={"source": {"commit": "abc"}},
    )
    assert manifest["format"] == export.FORMAT and manifest["source"] == {"commit": "abc"}
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest

    for world in ("editor", "result"):
        data = json.loads((tmp_path / f"{world}.json").read_text())
        assert data["state"]["readonly"] == {"reason": "Precomputed demo"}
        assert data["state"]["active_tab"] == ("result" if world == "result" else "layout")
        assert data["state"]["detectors"], f"the {world} world lists the monitor"
        assert all(not d["params"]["visible"] for d in data["state"]["detectors"])  # opened unchecked ...
        assert all(not g["visible"] for g in data["state"]["glyphs"] if g["kind"] == "detector")  # ... and hidden
        size = (tmp_path / f"{world}.meshes.bin").stat().st_size
        refs = [data["meshes"]["background"]] + [m for lv in data["meshes"]["levels"] for m in (lv["painted"], lv["full"])]
        for ref in refs:  # every buffer lies inside the file, 4-byte aligned
            for offset, length in (ref["positions"], ref["indices"]):
                assert offset % 4 == 0 and offset + length <= size
            assert ref["positions"][1] == ref["count"] * 12 and ref["indices"][1] == ref["tris"] * 12

    for plane, entry in manifest["sections"].items():
        section = json.loads((tmp_path / "sections" / f"{plane}.json").read_text())
        assert len(section["stops"]) == entry["count"] and section["stops"][0] == pytest.approx(entry["stops"][0])
        assert np.allclose(np.diff(section["stops"]), 0.5)
        assert entry["default"] in [pytest.approx(s) for s in section["stops"]]
        assert all(s["plane"] == plane and s["position"] == pytest.approx(p) for s, p in zip(section["slices"], section["stops"]))

    assert manifest["fields"], "the result's detectors have fields"
    for spec_id, entry in manifest["fields"].items():
        meta = json.loads((tmp_path / "fields" / f"{spec_id}.json").read_text())
        rows, cols = meta["grid"]
        samples = meta["frames"] * len(meta["components"]) * rows * cols
        assert (tmp_path / "fields" / f"{spec_id}.real.bin").stat().st_size == samples * 2
        assert entry["derived"] == [] and "|E|" not in meta["components"]
        assert meta["components"] == ["Ex", "Ey"]
        assert meta["quant"]["real"][0] <= meta["quant"]["real"][1]

    state = json.loads((tmp_path / "result.json").read_text())["state"]
    (spec_id,) = [s["id"] for s in state["detectors"]]
    assert state["field_settings"][spec_id]["field_component"] == "|E|"
    assert state["field_settings"][spec_id]["field_visible"] is False


def test_export_demo_needs_a_result(tmp_path: Path) -> None:
    viewer = bf.viewer.widget.view(_detector_viewer().stack)
    with pytest.raises(ValueError, match="no result"):
        export.export_demo(viewer, tmp_path)
