"""Benchmark StackDevice write/grad compile time, warm median/IQR, and peak GPU memory: greens vs legacy.

Runs each (voxelizer, averaging, grid) configuration in a clean worker subprocess so
XLA compile time and CUDA ``peak_bytes_in_use`` are uncontaminated across cases.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _make_bend_device(target_nx: int, target_nz: int, averaging: str, voxelizer: str):
    import fdtdx
    import gdsfactory as gf
    import jax.numpy as jnp
    import numpy as np
    from gdsfactory.technology import LayerLevel, LayerStack

    import bifrost as bf

    gf.gpdk.PDK.activate()
    gf.snap.SNAP_ENABLED = False

    # Tight air-clad SOI 90-degree bend (R = 2.0 um, w = 0.5 um, h = 0.22 um, side edges every ~5 deg)
    r_mid = 2.0
    width = 0.5
    th = np.linspace(0.0, 0.5 * math.pi, 37)
    r_out = r_mid + 0.5 * width
    r_in = r_mid - 0.5 * width
    outer = np.stack([r_out * np.sin(th), r_mid - r_out * np.cos(th)], axis=1)
    inner = np.stack([r_in * np.sin(th), r_mid - r_in * np.cos(th)], axis=1)
    poly = np.concatenate([outer, inner[::-1]])

    c = gf.Component()
    c.add_polygon(poly, layer=gf.gpdk.LAYER.WG)
    c.add_port("o1", center=(0.0, 0.0), width=width, orientation=180, layer=gf.gpdk.LAYER.WG)
    c.add_port("o2", center=(r_mid, r_mid), width=width, orientation=90, layer=gf.gpdk.LAYER.WG)

    lstack = LayerStack(
        layers={
            "box": LayerLevel(layer="WAFER", zmin=-1.0, thickness=1.0, material="sio2", mesh_order=9),
            "core": LayerLevel(
                layer=gf.gpdk.LAYER.WG, zmin=0.0, thickness=0.22, material="si", mesh_order=1, sidewall_angle=6.0
            ),
        }
    )
    mats = {
        "si": fdtdx.Material(permittivity=3.48**2),
        "sio2": fdtdx.Material(permittivity=1.444**2),
        "air": fdtdx.Material(permittivity=1.0),
    }
    span_xy = 3.84
    res_um = span_xy / target_nx
    box_z = target_nz * res_um
    # Center box exactly to hit (target_nx, target_nx, target_nz)
    ext_len = 0.75
    raw_span_x = r_mid + 0.5 * width + ext_len  # o1 shifts min_x to -ext_len, o2 shifts max_y to r_mid + ext_len
    pad_xy = span_xy - raw_span_x
    pad_z = 0.5 * (box_z - 0.22)
    stack = bf.StackDevice(
        c,
        lstack,
        mats,
        background="air",
        zrange_layers=["core"],
        extend_ports=(("o1", ext_len), ("o2", ext_len)),
        boundary_offset=(0.5 * pad_xy, -0.5 * pad_xy, 0.5 * pad_xy, -0.5 * pad_xy, pad_z, -pad_z),
        name="stack",
        averaging=averaging,
        voxelizer=voxelizer,
        dtype=jnp.float32,
    )
    cfg = stack.get_simulation_config(time=50e-15, grid=fdtdx.core.grid.UniformGrid(spacing=res_um * 1e-6))
    vol = fdtdx.SimulationVolume(partial_real_shape=stack.partial_real_shape, material=mats["air"])
    t0 = time.perf_counter()
    import jax

    objects, arrays, params, cfg, _ = fdtdx.place_objects(
        object_list=[vol, stack],
        config=cfg,
        constraints=[stack.place_relative_to(vol, axes=(0, 1, 2), own_positions=(0, 0, 0), other_positions=(0, 0, 0))],
        key=jax.random.PRNGKey(0),
    )
    t_place = time.perf_counter() - t0
    return objects["stack"], arrays, {**params, **bf.init_stack_params(objects)}["stack"], res_um, t_place


def _iqr(samples: list[float]) -> float:
    q = statistics.quantiles(samples, n=4)
    return float(q[2] - q[0])


def _bench_single(nx: int, nz: int, averaging: str, voxelizer: str, repeat: int = 10) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    import bifrost as bf

    assert Path(bf.__file__).resolve().is_relative_to(REPO_ROOT), bf.__file__
    d, arrays, p, res_um, t_place = _make_bend_device(nx, nz, averaging, voxelizer)

    def written(q):
        return d.write(arrays, q).inv_permittivities[:, *d.grid_slice]

    write_jit = jax.jit(written)
    t0 = time.perf_counter()
    w0 = write_jit(p)
    jax.block_until_ready(w0)
    fwd_cold = time.perf_counter() - t0
    fwd_samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(write_jit(p))
        fwd_samples.append(time.perf_counter() - t0)

    weights = jnp.ones_like(w0)
    grad_fn = jax.jit(jax.grad(lambda q: jnp.sum(weights * written(q))))
    t0 = time.perf_counter()
    g0 = grad_fn(p)
    jax.block_until_ready(g0)
    bwd_cold = time.perf_counter() - t0
    bwd_samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(grad_fn(p))
        bwd_samples.append(time.perf_counter() - t0)

    status = {}
    if averaging == "subpixel_full":
        import numpy as np

        st = d.interface_status(p)
        status = {k: np.asarray(v).item() for k, v in st.items()}
    mem = jax.devices()[0].memory_stats() or {}
    return {
        "grid_shape": [int(x) for x in d.grid_shape],
        "res_nm": round(res_um * 1e3, 3),
        "voxelizer": voxelizer,
        "averaging": averaging,
        "place_s": round(t_place, 4),
        "write_cold_s": round(fwd_cold, 4),
        "write_med_s": round(statistics.median(fwd_samples), 5),
        "write_iqr_s": round(_iqr(fwd_samples), 5),
        "grad_cold_s": round(bwd_cold, 4),
        "grad_med_s": round(statistics.median(bwd_samples), 5),
        "grad_iqr_s": round(_iqr(bwd_samples), 5),
        "peak_gb": round(mem.get("peak_bytes_in_use", 0) / 1e9, 4),
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="benchmarks/greens/results/bench_write.json")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--nx", type=int, default=160)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--averaging", default="subpixel")
    parser.add_argument("--voxelizer", default="greens")
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()

    if args.worker:
        row = _bench_single(args.nx, args.nz, args.averaging, args.voxelizer, repeat=args.repeat)
        print("RESULT_JSON:" + json.dumps(row), flush=True)
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text())
        except Exception:
            results = []

    configs = [(160, 48), (256, 64)]
    for nx, nz in configs:
        for voxelizer in ("greens", "legacy"):
            for averaging in ("subpixel", "subpixel_full"):
                if any(r["voxelizer"] == voxelizer and r["averaging"] == averaging and r["grid_shape"][0] in (nx, nx - 1, nx + 1) for r in results):
                    print(f"Skipping already completed ({nx}x{nx}x{nz}, {voxelizer}, {averaging})", flush=True)
                    continue
                cmd = [
                    sys.executable,
                    __file__,
                    "--worker",
                    "--nx",
                    str(nx),
                    "--nz",
                    str(nz),
                    "--voxelizer",
                    voxelizer,
                    "--averaging",
                    averaging,
                    "--repeat",
                    str(args.repeat),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if proc.returncode != 0:
                    print(f"FAILED ({nx}, {voxelizer}, {averaging}):\n{proc.stderr}", flush=True)
                    continue
                line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON:")), None)
                if line is None:
                    print(f"NO RESULT ({nx}, {voxelizer}, {averaging}): {proc.stdout}", flush=True)
                    continue
                row = json.loads(line.split("RESULT_JSON:", 1)[1])
                results.append(row)
                out_path.write_text(json.dumps(results, indent=2))
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
