"""A slab on graded z grids: fdtdx with a graded StackDevice against the Berreman transfer matrix.

The 1D problem of benchmarks/subpixel/em_slab.py (4 x 4 cells periodic in x and y, PML in z; a plane-wave pulse at
normal incidence through a 137 nm slab of diag(9, 12, 14), averaging="subpixel"): the Jones matrices at 1.55 um
against berreman.jones, each run divided by a run without the slab on the same grid. Per grid it reports the error
of t and |r|, the cells along z and the time steps:

* uniform at the fine spacing, and at the coarse one;
* graded: the fine spacing within ``--band`` of the slab, growing to the coarse one elsewhere (bf.graded_edges);
* graded with the slab's faces on cell edges (``snap``).

The device's box is the whole volume, so fdtdx places it exactly; the source and the monitor planes are one cell
thick (no physical size along z, which fdtdx would count in cells from the domain's bottom).

Run on a CPU or GPU node from the checkout:
    python benchmarks/rectilinear/graded_slab.py out.json --fine 10 --coarse 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "subpixel"))
from berreman import jones

import bifrost as bf

WAVELENGTH = 1.55  # um
THICKNESS = 0.137  # um
SLAB_Z = 0.013  # um: the slab's bottom (not on a cell edge of the uniform grids)
EPS = np.diag([9.0, 12.0, 14.0])


def full_tensor() -> np.ndarray:
    """The rotated uniaxial tensor with an xz entry of benchmarks/subpixel/em_slab.py."""
    a = np.radians(25.0)
    r = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    eps = r @ EPS @ r.T
    eps[0, 2] = eps[2, 0] = 0.8
    return eps


CASES = {"diag": (EPS, "subpixel"), "full": (full_tensor(), "subpixel_full")}  # (tensor, averaging)
PML = 0.6  # um
HALF = 1.8  # um: the volume runs from -HALF to HALF in z (PML included)
Z_SRC, Z_T, Z_T2, Z_R = -0.7, 0.75, 0.95, -1.0  # um: source, transmission planes, reflection plane


def z_edges(kind: str, fine: float, coarse: float, band: float) -> np.ndarray:
    """z edges (um) of the volume: 'uniform-fine', 'uniform-coarse', 'graded' or 'graded-snap'."""
    if kind.startswith("uniform"):
        res = fine if kind == "uniform-fine" else coarse
        n = round(2 * HALF / res)
        return np.linspace(-HALF, HALF, n + 1)
    n_pml = round(PML / coarse)
    region = [(SLAB_Z - band, SLAB_Z + THICKNESS + band, fine)]
    snap = [SLAB_Z, SLAB_Z + THICKNESS] if kind == "graded-snap" else []
    return bf.graded_edges(-HALF, HALF, coarse, fine=region, snap=snap, uniform_ends=(n_pml, n_pml))


def run(kind: str, fine: float, coarse: float, band: float, with_slab: bool, pol: int,
        case: str = "diag") -> tuple[np.ndarray, dict]:
    """(3, 2) phasors (trans, trans2, refl) x (Ex, Ey) of one run, and facts about its grid."""
    import fdtdx
    import gdsfactory as gf
    import jax
    import jax.numpy as jnp
    from gdsfactory.technology import LayerLevel, LayerStack

    gf.gpdk.PDK.activate()
    gf.snap.SNAP_ENABLED = False
    lateral = 4 * coarse
    eps, averaging = CASES[case]
    if not with_slab:
        eps = np.eye(3)
    mats = {"air": fdtdx.Material(), "slab": fdtdx.Material(permittivity=tuple(tuple(float(x) for x in row)
                                                                                for row in eps))}
    c = gf.Component()
    h = lateral / 2
    c.add_polygon([(-h, -h), (h, -h), (h, h), (-h, h)], layer=gf.gpdk.LAYER.WAFER)
    ls = LayerStack(layers={"slab": LayerLevel(layer=gf.gpdk.LAYER.WAFER, zmin=SLAB_Z, thickness=THICKNESS,
                                                material="slab", mesh_order=1)})
    # the box is the whole volume, -HALF to HALF
    device = bf.StackDevice.from_layer_stack(
        c, ls, mats, background="air", name="stack", zrange_layers=["slab"],
        boundary_offset=(0.0, 0.0, 0.0, 0.0, HALF - SLAB_Z - THICKNESS, -HALF - SLAB_Z), diagnostics=False,
        averaging=averaging,
    )
    z = z_edges(kind, fine, coarse, band)
    grid = device.rectilinear_grid(coarse, z_um=z)
    config = device.get_simulation_config(time=120e-15, grid=grid, backend=jax.default_backend())
    volume, constraints = device.get_simulation_volume(material=mats["air"])
    n_pml_min = round(PML / float(z[1] - z[0]))
    n_pml_max = round(PML / float(z[-1] - z[-2]))
    bound = fdtdx.BoundaryConfig(
        boundary_type_minx="periodic", boundary_type_maxx="periodic", boundary_type_miny="periodic",
        boundary_type_maxy="periodic", thickness_grid_minz=n_pml_min, thickness_grid_maxz=n_pml_max,
    )
    boundaries, b_constraints = fdtdx.boundary_objects_from_config(bound, volume)
    wave = fdtdx.WaveCharacter(wavelength=WAVELENGTH * 1e-6)
    profile = fdtdx.GaussianPulseProfile(spectral_width=fdtdx.WaveCharacter(wavelength=10 * WAVELENGTH * 1e-6),
                                         center_wave=wave)
    source = fdtdx.UniformPlaneSource(
        name="source", partial_grid_shape=(None, None, 1), direction="+", wave_character=wave, temporal_profile=profile,
        fixed_E_polarization_vector=(1.0, 0.0, 0.0) if pol == 0 else (0.0, 1.0, 0.0),
    )
    planes = {"trans": Z_T, "trans2": Z_T2, "refl": Z_R}
    detectors = [
        fdtdx.PhasorDetector(name=n, partial_grid_shape=(None, None, 1), wave_characters=(wave,), components=("Ex", "Ey"),
                             reduce_volume=True, scaling_mode="pulse")
        for n in planes
    ]

    def at(obj, zz):
        return obj.place_relative_to(volume, axes=(0, 1, 2), own_positions=(0, 0, 0), other_positions=(0, 0, 0),
                                     margins=(0, 0, zz * 1e-6))

    constraints += [*b_constraints, at(source, Z_SRC), *(at(d, planes[d.name]) for d in detectors)]
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=[volume, device, source, *detectors, *boundaries.values()], config=config,
        constraints=constraints, key=key,
    )
    params = {**params, **bf.init_stack_params(objects)}
    arrays, objects, _ = bf.apply_stack(arrays, objects, params, key)
    _, state = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key, show_progress=False)
    rows = [np.asarray(jnp.reshape(state.detector_states[n]["phasor"], (-1,))[:2]) for n in ("trans", "trans2", "refl")]
    facts = {"cells_z": len(z) - 1, "min_dz_nm": float(np.min(np.diff(z)) * 1e3), "steps": int(config.time_steps_total)}
    return np.stack(rows), facts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out")
    parser.add_argument("--fine", type=float, default=10.0, help="fine spacing (nm)")
    parser.add_argument("--coarse", type=float, default=40.0, help="coarse spacing (nm)")
    parser.add_argument("--band", type=float, default=0.1, help="fine band around the slab (um)")
    parser.add_argument("--kinds", nargs="+", default=["uniform-fine", "uniform-coarse", "graded", "graded-snap"])
    parser.add_argument("--case", choices=sorted(CASES), default="diag",
                        help="diag: diag(9, 12, 14), subpixel; full: rotated with an xz entry, subpixel_full")
    args = parser.parse_args()
    import fdtdx
    import jax

    fine, coarse = args.fine * 1e-3, args.coarse * 1e-3
    r_ref, t_ref = jones([(CASES[args.case][0], THICKNESS)], 1.0, 1.0, WAVELENGTH)
    t_ref = t_ref * np.exp(-2j * np.pi / WAVELENGTH * THICKNESS)
    print("jax", jax.__version__, "fdtdx", getattr(fdtdx, "__version__", "?"), "on", jax.devices()[0].device_kind,
          flush=True)
    rows = []
    for kind in args.kinds:
        started = time.time()
        t = np.zeros((2, 2), dtype=complex)
        r = np.zeros((2, 2))
        for pol in (0, 1):
            with_slab, facts = run(kind, fine, coarse, args.band, True, pol, args.case)
            without, _ = run(kind, fine, coarse, args.band, False, pol, args.case)
            inc = without[0, pol]
            t[:, pol] = with_slab[0] / inc
            r[:, pol] = np.abs(with_slab[2] / inc)
        row = {
            "kind": kind, **facts, "seconds": round(time.time() - started, 1),
            "t_error": float(np.max(np.abs(t - t_ref))), "r_error": float(np.max(np.abs(r - np.abs(r_ref)))),
            "energy": float(np.max(np.abs(np.sum(np.abs(t) ** 2 + r**2, axis=0) - 1.0))),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path(args.out).write_text(json.dumps({"case": args.case, "fine_nm": args.fine, "coarse_nm": args.coarse,
                                          "band_um": args.band, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
