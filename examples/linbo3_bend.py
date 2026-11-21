"""An X-cut LiNbO3 (TFLN) 90 degree waveguide bend on silica, simulated with bifrost and fdtdx.

The waveguide is an air-clad 1.2 x 0.50 um lithium niobate ridge with 20 degree
sloped sidewalls on a silica bottom oxide (BOX). In simulation coordinates
(crystallographic X along z, optic axis Z along y, Y along x), the ordinary and
extraordinary refractive indices give an anisotropic diagonal permittivity
tensor ``(n_o^2, n_e^2, n_o^2)`` with ``n_o = 2.211`` and ``n_e = 2.138`` at
1.55 um:

- Launching TE0 into port ``o1`` (+x propagation, E predominantly along y/Z) sees
  the extraordinary index ``n_e``.
- Exiting port ``o2`` (+y propagation, E predominantly along x/Y) sees the
  ordinary index ``n_o``.
- The 20 degree trapezoidal sidewalls on an asymmetric SiO2/air stack break
  vertical symmetry and convert a measurable fraction of TE0 into TM0 around the
  bend.

Run it on a GPU node:  python examples/linbo3_bend.py
Options:               python examples/linbo3_bend.py --ppw 20 --figure linbo3_bend_field.png
"""

# --8<-- [start:imports]
import argparse
import importlib.util
import time
from dataclasses import dataclass
from typing import Any

import fdtdx
import gdsfactory as gf
import jax
import jax.numpy as jnp
import numpy as np
from gdsfactory.technology import LayerLevel, LayerStack

import bifrost as bf

gf.gpdk.PDK.activate()
# --8<-- [end:imports]

# --8<-- [start:parameters]
WAVELENGTH = 1.55  # um, in vacuum
N_O, N_E = 2.211, 2.138  # X-cut LiNbO3 at 1.55 um (ordinary along x/z, optic axis along y)
BOX_INDEX, CLAD_INDEX = 1.444, 1.0  # SiO2 buried oxide beneath the core, air above and beside it
WIDTH, THICKNESS, SIDEWALL_ANGLE = 1.2, 0.50, 20.0  # um, deg (trapezoidal TFLN ridge)
RADIUS = 7.0  # um, of the bend's centerline
EXTENSION = 2.5  # um, straight leads through the absorbing boundaries
MARGIN = 1.90  # um beyond the bend on the free sides -> 12.00 x 12.00 um lateral box
BOX_HEIGHT, PML = 4.80, 1.0  # um -> 12.00 x 12.00 x 4.80 um simulation volume
PLANE_PADDING = (0.5, 0.5)  # mode plane padding in wavelengths beyond width and thickness
PPW = 20  # points per local wavelength for bf.mesh
DURATION = 500e-15  # s of simulated time
MOVIE_FRAMES = 40
# --8<-- [end:parameters]


# --8<-- [start:component]
component = gf.components.bend_circular(radius=RADIUS, angle=90, width=WIDTH)
# --8<-- [end:component]


# --8<-- [start:layerstack]
LAYER_STACK = LayerStack(
    layers={
        "box": LayerLevel(layer="WAFER", zmin=-3.0, thickness=3.0, material="silica", mesh_order=9),
        "core": LayerLevel(
            layer=gf.gpdk.LAYER.WG,
            zmin=0.0,
            thickness=THICKNESS,
            sidewall_angle=SIDEWALL_ANGLE,
            material="linbo3",
            mesh_order=1,
        ),
    }
)
MATERIALS = {
    "linbo3": fdtdx.Material(permittivity=(N_O**2, N_E**2, N_O**2)),
    "silica": fdtdx.Material(permittivity=BOX_INDEX**2),
    "air": fdtdx.Material(permittivity=CLAD_INDEX**2),
}
# --8<-- [end:layerstack]


# --8<-- [start:stack]
def build_stack(averaging: str = "subpixel") -> bf.StackDevice:
    """Build the voxelized X-cut LiNbO3 bend with straight extensions to the box faces."""
    z_margin = (BOX_HEIGHT - THICKNESS) / 2
    return bf.StackDevice.from_layer_stack(
        component,
        LAYER_STACK,
        MATERIALS,
        background="air",
        zrange_layers=["core"],
        extend_ports=(("o1", EXTENSION), ("o2", EXTENSION)),
        boundary_offset=(MARGIN, 0.0, 0.0, -MARGIN, z_margin, -z_margin),
        averaging=averaging,
        name="stack",
    )
# --8<-- [end:stack]


@dataclass
class Linbo3BendResult:
    """Placed simulation objects, arrays, modal metrics, and field slice for the LiNbO3 bend."""

    stack: Any
    objects: Any
    arrays: Any
    config: Any
    transmission: float
    tm_conversion: float
    neff_in: float
    neff_out: float
    total_cells: int
    field: np.ndarray
    ez_field: np.ndarray
    edges: tuple[np.ndarray, np.ndarray]
    eps: np.ndarray
    seconds: float
    peak_gib: float | None


def _apply_stack(arrays: Any, objects: Any, all_params: dict[str, Any], key: jax.Array) -> tuple[Any, Any, Any]:
    """Apply the stack, solving port modes on the diagonal tensor if tidy3d_extras is absent under subpixel_full."""
    stack_obj = objects["stack"]
    if stack_obj.averaging != "subpixel_full" or importlib.util.find_spec("tidy3d_extras") is not None:
        return bf.apply_stack(arrays, objects, all_params, key)
    full_arrays, status = stack_obj.write_with_status(arrays, all_params.get("stack"), None, True)
    diag_inv = jnp.zeros_like(full_arrays.inv_permittivities)
    for idx_d in (0, 4, 8):
        diag_inv = diag_inv.at[idx_d].set(full_arrays.inv_permittivities[idx_d])
    diag_arrays = full_arrays.at["inv_permittivities"].set(diag_inv)
    objects, reapplied, reused = bf.stack._device._reapply_stack_objects(
        diag_arrays, objects, [stack_obj], key
    )
    objects = bf.stack._device._cast_pml_coefficients(objects, stack_obj.dtype)
    return full_arrays, objects, {
        "stack_reapplied": reapplied,
        "stack_reused": reused,
        "stack_status": {"stack": status},
    }


def simulate(
    ppw: float = PPW,
    duration: float = DURATION,
    averaging: str = "subpixel",
    coarsen_substrate: bool = True,
) -> Linbo3BendResult:
    """Run the 3D X-cut LiNbO3 bend simulation at the requested mesh density."""
    started = time.perf_counter()
    stack = build_stack(averaging=averaging)

    # --8<-- [start:settings]
    refine = (
        bf.Refine(material_cells={"silica": WAVELENGTH / (BOX_INDEX * min(float(ppw), 12.0))})
        if coarsen_substrate
        else None
    )
    mesh = bf.mesh(stack, WAVELENGTH, ppw=ppw, pml=PML, refine=refine)
    config = stack.get_simulation_config(time=duration, grid=mesh.grid)
    wave = fdtdx.WaveCharacter(wavelength=WAVELENGTH * 1e-6)
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=wave, spectral_width=fdtdx.WaveCharacter(frequency=0.05 * wave.get_frequency())
    )
    te0 = dict(mode_index=0, filter_pol="te")
    tm0 = dict(mode_index=0, filter_pol="tm")
    # --8<-- [end:settings]

    # --8<-- [start:objects]
    volume, constraints = stack.get_simulation_volume()
    objects = [volume, stack]
    for obj, placing in (
        stack.create_mode_plane_source_at(
            "o1", PLANE_PADDING, wave_character=wave, temporal_profile=pulse, port_offset=-1.0,
            name="source", grid=config, **te0,
        ),
        stack.create_mode_overlap_detector_at(
            "o1", PLANE_PADDING, direction="in", wave_characters=(wave,), scaling_mode="pulse",
            port_offset=-0.5, name="in", grid=config, **te0,
        ),
        stack.create_mode_overlap_detector_at(
            "o2", PLANE_PADDING, wave_characters=(wave,), scaling_mode="pulse",
            port_offset=-0.5, name="out", grid=config, **te0,
        ),
        stack.create_mode_overlap_detector_at(
            "o2", PLANE_PADDING, wave_characters=(wave,), scaling_mode="pulse",
            port_offset=-0.5, name="out_tm", grid=config, **tm0,
        ),
    ):
        objects.append(obj)
        constraints += placing
    field = fdtdx.PhasorDetector(
        name="field", wave_characters=(wave,), components=("Ex", "Ey", "Ez"),
        scaling_mode="pulse", partial_grid_shape=(None, None, 1),
    )
    movie = fdtdx.FieldDetector(
        name="movie", components=("Ex", "Ey", "Ez"), partial_grid_shape=(None, None, 1),
        switch=fdtdx.OnOffSwitch(interval=max(1, config.time_steps_total // MOVIE_FRAMES)),
    )
    for monitor in (field, movie):
        objects.append(monitor)
        constraints += [monitor.same_size(volume, axes=(0, 1)), monitor.place_at_center(volume)]
    boundaries, placing = fdtdx.boundary_objects_from_config(mesh.boundary, volume)
    objects += boundaries.values()
    constraints += placing
    # --8<-- [end:objects]

    # --8<-- [start:run]
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
    arrays, objects, _ = _apply_stack(arrays, objects, {**params, **bf.init_stack_params(objects)}, key)
    jax.block_until_ready(arrays.inv_permittivities)
    with jax.enable_x64(False):
        _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key, show_progress=False)
        jax.block_until_ready(arrays.detector_states)
    # --8<-- [end:run]

    # --8<-- [start:results]
    states = arrays.detector_states
    a_in = objects["in"].compute_overlap(states["in"])
    a_out = objects["out"].compute_overlap(states["out"])
    a_tm = objects["out_tm"].compute_overlap(states["out_tm"])
    transmission = float(np.ravel(abs(a_out / a_in) ** 2)[0])
    tm_conversion = float(np.ravel(abs(a_tm / a_in) ** 2)[0])
    neff_in = float(complex(np.asarray(objects["in"]._mode_neff).reshape(-1)[0]).real)
    neff_out = float(complex(np.asarray(objects["out"]._mode_neff).reshape(-1)[0]).real)
    # --8<-- [end:results]

    k = objects["field"].grid_slice[2].start
    phasor = np.asarray(states["field"]["phasor"][0, 0, :, :, :, 0])
    total_e2 = np.sum(np.abs(phasor) ** 2, axis=0)
    ez2 = np.abs(phasor[2]) ** 2
    stats = getattr(jax.devices()[0], "memory_stats", lambda: None)()
    return Linbo3BendResult(
        stack=stack,
        objects=objects,
        arrays=arrays,
        config=config,
        transmission=transmission,
        tm_conversion=tm_conversion,
        neff_in=neff_in,
        neff_out=neff_out,
        total_cells=int(np.prod(objects["volume"].grid_shape)),
        field=total_e2,
        ez_field=ez2,
        edges=(np.asarray(mesh.edges[0], dtype=float), np.asarray(mesh.edges[1], dtype=float)),
        eps=1.0 / np.asarray(arrays.inv_permittivities[0, :, :, k]),
        seconds=time.perf_counter() - started,
        peak_gib=None if not stats else stats.get("peak_bytes_in_use", 0) / 2**30,
    )


def open_viewer(result: Linbo3BendResult) -> Any:
    """Open the interactive 3D viewer preloaded with the LiNbO3 bend simulation result."""
    viewer = bf.view(result.stack)
    viewer.load_result(result.objects, result.arrays, result.config, stack=result.stack)
    return viewer


def plot_field(result: Linbo3BendResult, path: str) -> None:
    """Save total |E|^2 and hybridized |Ez|^2 on the core mid-plane with waveguide contours."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_edges, y_edges = result.edges
    xs = 0.5 * (x_edges[:-1] + x_edges[1:])
    ys = 0.5 * (y_edges[:-1] + y_edges[1:])
    peak = float(result.field.max()) or 1.0

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    panels = (
        (result.field / peak, f"Total $|E|^2$ (TE0 T = {result.transmission:.3f})", "inferno"),
        (result.ez_field / peak, f"Vertical $|E_z|^2$ (TE0$\\to$TM0 = {result.tm_conversion:.3f})", "magma"),
    )
    for ax, (data, title, cmap) in zip(axes, panels, strict=True):
        im = ax.pcolormesh(x_edges, y_edges, data.T, cmap=cmap, shading="flat")
        ax.contour(xs, ys, result.eps.T, levels=[(N_O**2 + CLAD_INDEX**2) / 2], colors="white", linewidths=0.6)
        ax.set_aspect("equal")
        ax.set(xlabel="x (um)", ylabel="y (um)", title=title)
        fig.colorbar(im, ax=ax, shrink=0.88)
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ppw", type=float, default=PPW, help="cells per local wavelength")
    parser.add_argument("--duration", type=float, default=DURATION, help="simulated time in s")
    parser.add_argument("--averaging", default="subpixel", choices=("subpixel", "subpixel_full"))
    parser.add_argument("--no-coarsen-substrate", action="store_true", help="refine full-wafer silica BOX at full ppw")
    parser.add_argument("--figure", default="linbo3_bend_field.png", help="where to save the field figure")
    args = parser.parse_args()

    res = simulate(
        ppw=args.ppw,
        duration=args.duration,
        averaging=args.averaging,
        coarsen_substrate=not args.no_coarsen_substrate,
    )
    print(f"cells:          {res.total_cells:,d}")
    print(f"neff (o1/o2):   {res.neff_in:.4f} (n_e side) -> {res.neff_out:.4f} (n_o side)")
    print(f"TE0 trans.:     {res.transmission:.4f}")
    print(f"TE0->TM0 conv.: {res.tm_conversion:.4f}")
    print(f"wall time:      {res.seconds:.1f} s (compilation included)")
    if res.peak_gib is not None:
        print(f"peak memory:    {res.peak_gib:.2f} GiB on {jax.devices()[0]}")
    if args.figure:
        plot_field(res, args.figure)
        print(f"field image:    {args.figure}")
