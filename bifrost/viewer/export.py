"""Exports a viewer holding a result as static files for the precomputed browser demo.

The live viewer is an ``anywidget`` driven by a Python kernel. The demo shows the
same front end with no kernel: this module writes everything the kernel would
have sent, as files, and ``js/viewer/static.ts`` plays them back.

``export_demo(viewer, out_dir)`` writes::

    manifest.json              what is here, how it was made, and its sizes
    <world>.json               the world's state, meta and mesh index
    <world>.meshes.bin         the world's mesh buffers (float32 / uint32)
    sections/<plane>.json      precomputed 2D slices at evenly spaced positions
    fields/<id>.json           one detector's field metadata
    fields/<id>.real.bin       its samples, as uint16 against a per-file range
    fields/<id>.imag.bin       (phasors only)

A *world* is what the viewer calls it (spec R17): ``editor`` (the device with its
setup) and ``result`` (the loaded result, with its detectors adopted). Fields are
loaded on demand, and so are the slices, one plane at a time.

The viewer shades one field component at a time, and in a bend the dominant
in-plane component rotates, so every plane field gets one extra *derived*
component first in its list, ``|E|``: the square root of the summed squared
magnitudes of its recorded ``E`` components. The manifest says which components
are derived; every other component is exactly what the solver recorded.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .payload import build_field_payload, build_section_payload
from .state import PLANES

__all__ = ["FORMAT", "export_demo"]

#: The layout version of the exported files; the browser refuses another one.
FORMAT = 1

READONLY_REASON = "Precomputed demo"


def _write_json(path: Path, value: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, separators=(",", ":"))
    path.write_text(text)
    return len(text.encode())


def _write_bytes(path: Path, data: bytes) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)


def _export_meshes(geometry: dict[str, Any], base: Path) -> tuple[dict[str, Any], int]:
    """Writes a geometry payload's buffers to one file; returns its JSON index and the size."""
    chunks: list[bytes] = []
    offset = 0

    def put(buffer: Any) -> list[int]:
        nonlocal offset
        raw = bytes(buffer)
        pad = (-offset) % 4  # keep every buffer 4-byte aligned for typed-array views
        chunks.append(b"\0" * pad + raw)
        start = offset + pad
        offset = start + len(raw)
        return [start, len(raw)]

    def mesh(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "count": payload["count"],
            "tris": payload["tris"],
            "positions": put(payload["positions"]),
            "indices": put(payload["indices"]),
        }

    index = {
        "origin": geometry["origin"],
        "size": geometry["size"],
        "levels": [
            {"name": level["name"], "painted": mesh(level["painted"]), "full": mesh(level["full"])}
            for level in geometry["levels"]
        ],
        "background": mesh(geometry["background"]),
    }
    return index, _write_bytes(base, b"".join(chunks))


def _quantize(values: Any) -> tuple[bytes, float, float]:
    """float32 samples as uint16 against their own (min, max)."""
    array = np.asarray(values, dtype=np.float32)
    low, high = float(array.min()), float(array.max())
    scale = (high - low) or 1.0
    quantized = np.round((array - low) / scale * 65535.0).astype("<u2")
    return quantized.tobytes(), low, high


MAGNITUDE = "|E|"


def _with_magnitude(entry: dict[str, Any]) -> dict[str, Any]:
    """The field with a derived ``|E|`` component put first, or unchanged without two E components."""
    components = list(entry["components"])
    electric = [i for i, name in enumerate(components) if name in ("Ex", "Ey", "Ez")]
    if len(electric) < 2:
        return entry
    frames, (rows, cols) = len(entry.get("freqs") or entry.get("times") or [0]), entry["grid"]
    shape = (frames, len(components), rows, cols)
    real = np.frombuffer(entry["real"], dtype=np.float32).reshape(shape)
    imag = np.frombuffer(entry["imag"], dtype=np.float32).reshape(shape) if "imag" in entry else None
    power = real[:, electric] ** 2 + (imag[:, electric] ** 2 if imag is not None else 0.0)
    magnitude = np.sqrt(power.sum(axis=1, keepdims=True)).astype(np.float32)
    derived = dict(entry, components=[MAGNITUDE, *components], derived=[MAGNITUDE])
    derived["real"] = np.concatenate([magnitude, real], axis=1)
    if imag is not None:
        derived["imag"] = np.concatenate([np.zeros_like(magnitude), imag], axis=1)  # a magnitude has no phase
    return derived


def _peak_frame(entry: dict[str, Any]) -> int:
    """The recorded frame with the most field energy: where a still of a movie shows the most."""
    components = len(entry["components"])
    frames = len(entry.get("freqs") or entry.get("times") or [0])
    real = np.frombuffer(entry["real"], dtype=np.float32).reshape(frames, components, -1)
    return int(np.argmax((real**2).sum(axis=(1, 2))))


def _export_field(entry: dict[str, Any], out: Path) -> dict[str, Any]:
    """Writes one detector's field; returns its manifest entry."""
    spec_id = entry["id"]
    meta = {key: value for key, value in entry.items() if key not in ("real", "imag")}
    sizes: dict[str, int] = {}
    quant: dict[str, list[float]] = {}
    for part in ("real", "imag"):
        if part not in entry:
            continue
        data, low, high = _quantize(np.asarray(entry[part], dtype=np.float32))
        sizes[part] = _write_bytes(out / "fields" / f"{spec_id}.{part}.bin", data)
        quant[part] = [low, high]
    meta["quant"] = quant
    meta["frames"] = int(len(meta.get("freqs") or meta.get("times") or [0]))
    sizes["json"] = _write_json(out / "fields" / f"{spec_id}.json", meta)
    return {
        "id": spec_id,
        "name": entry["name"],
        "components": entry["components"],
        "derived": entry.get("derived", []),
        "frames": meta["frames"],
        "grid": entry["grid"],
        "bytes": sizes,
    }


def _stops(lo: float, hi: float, anchor: float, step: float) -> list[float]:
    """Evenly spaced positions in [lo, hi] that include ``anchor``."""
    first = math.ceil((lo - anchor) / step - 1e-9)
    last = math.floor((hi - anchor) / step + 1e-9)
    return [round(anchor + k * step, 9) for k in range(first, last + 1)]


def _export_sections(state: Any, out: Path, step: float) -> dict[str, Any]:
    """Slices the device at evenly spaced positions of every plane; returns the index."""
    saved_plane, saved_positions = state.plane, dict(state.plane_positions)
    index: dict[str, Any] = {}
    try:
        for plane in PLANES:
            lo, hi = state.plane_bounds(plane)
            anchor = saved_positions[plane]  # the box center, the default position
            stops = _stops(lo, hi, anchor, step)
            slices = []
            for position in stops:
                state.plane = plane
                state.plane_positions[plane] = position
                payload = build_section_payload(state)
                slices.append(payload)
            size = _write_json(out / "sections" / f"{plane}.json", {"stops": stops, "slices": slices})
            index[plane] = {
                "stops": [stops[0], stops[-1]],
                "count": len(stops),
                "step": step,
                "default": round(anchor, 9),  # where the plane starts: the box center
                "bytes": size,
            }
    finally:
        state.plane, state.plane_positions = saved_plane, saved_positions
    return index


def export_demo(
    viewer: Any,
    out_dir: str | Path,
    *,
    section_step_um: float = 0.1,
    field_defaults: dict[str, dict[str, Any]] | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Writes a viewer that holds a result (``load_result`` or ``run``) as demo files.

    Args:
        viewer: a :class:`~bifrost.viewer.widget.Viewer` whose result world exists.
        out_dir: the directory to write; created if missing.
        section_step_um: the spacing of the stored 2D slices along each plane's normal. The
            positions include the box center, which is where the field plane of a device
            centered on its core sits.
        field_defaults: how each detector's field is shown when the demo opens, by detector
            name: any of the ``field_*`` display settings (``field_component``,
            ``field_transform``, ``field_visible``, ...). A time-domain detector opens on the
            frame with the most field energy (set ``field_freq`` to override).
        manifest: extra entries for ``manifest.json`` (source revision, package versions,
            simulation parameters).

    Returns:
        The manifest, also written to ``manifest.json``.

    Raises:
        ValueError: if the viewer holds no result.
    """
    if viewer._result_state is None:
        raise ValueError("the viewer holds no result: call viewer.load_result(...) or viewer.run() first")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    editor, result = viewer._viewer_state, viewer._result_state
    worlds: dict[str, Any] = {}
    total = 0

    fields = {
        spec_id: _with_magnitude(entry) for spec_id, entry in build_field_payload(result, viewer._run_result).items()
    }
    shown: dict[str, dict[str, Any]] = {}  # the opening display settings, by spec id
    for spec_id, entry in fields.items():
        settings: dict[str, Any] = {}
        if entry.get("frame_kind") == "time":
            settings["field_freq"] = _peak_frame(entry)
        settings.update((field_defaults or {}).get(entry["name"], {}))
        shown[spec_id] = settings
    for name, state, payloads in (
        ("editor", editor, viewer._cached_payloads(editor)),
        ("result", result, (viewer._result_geometry, viewer._result_meta)),
    ):
        geometry, meta = payloads
        mesh_index, mesh_bytes = _export_meshes(geometry, out / f"{name}.meshes.bin")
        wire_state = state.to_state()
        wire_state["active_tab"] = "result" if name == "result" else "layout"
        wire_state["readonly"] = {"reason": READONLY_REASON}
        if name == "result":
            wire_state["field_settings"] = {
                spec_id: {**wire_state["field_settings"].get(spec_id, {}), **settings}
                for spec_id, settings in shown.items()
            } | {k: v for k, v in wire_state["field_settings"].items() if k not in shown}
        size = _write_json(
            out / f"{name}.json", {"state": wire_state, "meta": meta, "meshes": mesh_index}
        )
        worlds[name] = {"json_bytes": size, "mesh_bytes": mesh_bytes}
        total += size + mesh_bytes

    field_index = {}
    for spec_id, entry in fields.items():
        field_index[spec_id] = _export_field(entry, out)
        total += sum(field_index[spec_id]["bytes"].values())

    sections = _export_sections(result, out, section_step_um)
    total += sum(item["bytes"] for item in sections.values())

    data = {
        "format": FORMAT,
        "worlds": worlds,
        "fields": field_index,
        "sections": sections,
        "total_bytes": total,
        **(manifest or {}),
    }
    _write_json(out / "manifest.json", data)
    return data
