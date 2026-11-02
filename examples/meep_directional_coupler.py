"""MEEP reference for fdtdx_directional_coupler.py: same component, LayerStack, source, monitors and frequencies.

1. Export the gdsfactory geometry once (one process; imports gdsfactory/fdtdx):
       python examples/meep_directional_coupler.py export
2. Run MEEP (MPI; imports only meep/numpy):
       mpirun -np 48 python examples/meep_directional_coupler.py run [resolution_um]
"""

import json
import sys
import time as clock
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "output"
C = 299792458.0  # m/s, the speed of light fdtdx uses


def export() -> None:
    """Writes the component's polygons, the LayerStack levels and the run settings to JSON."""
    sys.path.insert(0, str(Path(__file__).parent))
    import fdtdx_directional_coupler as ex

    OUT.mkdir(exist_ok=True)
    # the polygons the fdtdx StackDevice draws: the component plus its port extensions
    stack = ex.build_stack()
    polygons = {
        index: np.split(np.asarray(v), np.cumsum(sizes)[:-1])
        for (_, index), v, sizes in zip(stack.source_layers, stack.source_vertices, stack.source_sizes, strict=True)
    }
    import gdsfactory as gf

    levels = []
    for name, lv in ex.LAYER_STACK.layers.items():
        index = gf.get_layer(lv.layer.layer if hasattr(lv.layer, "layer") else lv.layer)
        levels.append({
            "name": name,
            "zmin": lv.zmin,
            "thickness": lv.thickness,
            "mesh_order": lv.mesh_order,
            "index": float(np.sqrt(ex.MATERIALS[lv.material].permittivity[0])),
            "polygons": [np.asarray(p).tolist() for p in polygons.get(index, [])],
        })
    geometry = {
        "levels": levels,
        "domain": ex.domain(),
        "background_index": float(np.sqrt(ex.MATERIALS["clad"].permittivity[0])),
        "wavelength": ex.WAVELENGTH,
        "frequencies": ex.frequencies().tolist(),
        "bandwidth": ex.BANDWIDTH,
        "pulse_width": ex.PULSE_WIDTH,
        "pml": ex.PML,
        "plane": ex.PLANE,
        "time": ex.TIME,
        "resolution": ex.RESOLUTION,
        # port centers (x, y): o1 is the input, o4 the through port, o3 the cross port
        "ports": {p.name: [float(p.center[0]), float(p.center[1])] for p in ex.directional_coupler().ports},
    }
    (OUT / "geometry_dc.json").write_text(json.dumps(geometry))


def run(resolution: float | None = None) -> None:
    import meep as mp

    g = json.loads((OUT / "geometry_dc.json").read_text())
    (x0, x1), (y0, y1), (z0, z1) = g["domain"]
    res, pml = resolution or g["resolution"], g["pml"]
    f0, freqs = 1 / g["wavelength"], g["frequencies"]
    cell = mp.Vector3(x1 - x0, y1 - y0, z1 - z0)
    cells = round(cell.x * cell.y * cell.z / res**3)
    center = mp.Vector3((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)

    geometry = []  # MEEP: later objects win, and lower mesh_order has priority
    for lv in sorted(g["levels"], key=lambda lv: -lv["mesh_order"]):
        medium = mp.Medium(index=lv["index"])
        for p in lv["polygons"]:
            vertices = [mp.Vector3(x, y, lv["zmin"]) for x, y in p]
            geometry.append(mp.Prism(vertices, height=lv["thickness"], material=medium))

    plane = mp.Vector3(0, g["plane"][0], g["plane"][1])
    at = lambda port, dx=0.0: mp.Vector3(g["ports"][port][0] + dx, g["ports"][port][1], 0)
    # same Gaussian envelope as fdtdx: sigma_t = 1 / (2 pi PULSE_WIDTH f0), peak at 6 sigma_t
    pulse = mp.GaussianSource(f0, fwidth=2 * np.pi * g["pulse_width"] * f0, cutoff=6)
    source = mp.EigenModeSource(pulse, center=at("o1", -0.3), size=plane, eig_band=1, eig_match_freq=True)
    sim = mp.Simulation(
        cell_size=cell,
        geometry_center=center,
        geometry=geometry,
        default_material=mp.Medium(index=g["background_index"]),
        sources=[source],
        resolution=1 / res,
        boundary_layers=[mp.PML(pml)],
    )
    nfreq, df = len(freqs), freqs[-1] - freqs[0]
    monitors = {
        name: sim.add_mode_monitor(f0, df, nfreq, mp.ModeRegion(center=at(port), size=plane))
        for name, port in (("in", "o1"), ("through", "o4"), ("cross", "o3"))
    }
    slice_center, slice_size = mp.Vector3(center.x, center.y, 0), mp.Vector3(cell.x, cell.y, 0)
    dft = sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez], f0, 0, 1, center=slice_center, size=slice_size)
    start = clock.perf_counter()
    sim.run(until=g["time"] * C * 1e6)  # the same physical time as fdtdx, in um / c
    runtime = clock.perf_counter() - start

    alpha = {n: sim.get_eigenmode_coefficients(m, [1]).alpha[0, :, 0] for n, m in monitors.items()}
    E2 = sum(np.abs(np.squeeze(sim.get_dft_array(dft, c, 0))) ** 2 for c in (mp.Ex, mp.Ey, mp.Ez))
    x, y, _, _ = sim.get_array_metadata(dft_cell=dft)
    eps = np.squeeze(sim.get_array(center=slice_center, size=slice_size, component=mp.Dielectric))
    ex_, ey_, _, _ = sim.get_array_metadata(center=slice_center, size=slice_size)
    if mp.am_master():
        through, cross = (np.abs(alpha[n] / alpha["in"]) ** 2 for n in ("through", "cross"))
        mid = nfreq // 2
        print(
            f"MEEP, resolution {res * 1e3:.0f} nm, {cells / 1e6:.1f}M cells, {runtime:.0f} s: "
            f"{1e3 / f0:.0f} nm through {through[mid]:.4f}, cross {cross[mid]:.4f}",
            flush=True,
        )
        np.savez(
            OUT / f"meep_dc_{res * 1e3:g}nm.npz",
            frequency=np.asarray(freqs), wavelength=1 / np.asarray(freqs), through=through, cross=cross,
            x=np.asarray(x), y=np.asarray(y), E2=E2, eps=eps, eps_x=np.asarray(ex_), eps_y=np.asarray(ey_),
            runtime=runtime, cells=cells,
        )


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export()
    else:
        run(float(sys.argv[2]) if len(sys.argv) > 2 else None)
