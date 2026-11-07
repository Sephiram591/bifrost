# Non-uniform rectilinear grids

Drivers and results for a `StackDevice` on fdtdx's `RectilinearGrid` (graded z). Reproduce a run with
`python benchmarks/rectilinear/<driver>.py out.json` from the checkout.

## A slab on graded z grids (`graded_slab.py`, `results/graded_slab.json`)

A 137 nm slab of diag(9, 12, 14) at normal incidence (4 x 4 cells periodic in x and y, PML in z,
`averaging="subpixel"`), against the Berreman transfer matrix of `benchmarks/subpixel/berreman.py`; the
Jones matrices at 1.55 um, each run divided by a run without the slab on the same grid. x and y are 40 nm; z
is uniform at 10 or 40 nm, or graded with `bf.graded_edges`: 10 nm within 0.1 um of the slab, growing by at
most 1.25 to 40 nm, 40 nm under the PML. CPU (jax 0.11.2, fdtdx 0.6.3, float32).

| z grid | z cells | time steps | max \|t - t_ref\| | max \|\|r\| - \|r_ref\|\| | energy error |
|---|---|---|---|---|---|
| uniform 10 nm | 360 | 3854 | 1.05e-3 | 5.8e-4 | 5e-6 |
| uniform 40 nm | 90 | 1574 | 1.26e-2 | 5.4e-3 | 1.4e-5 |
| graded 10 / 40 nm | 121 | 3878 | 1.14e-3 | 2.7e-4 | 5.5e-5 |
| graded, the slab's faces on edges | 122 | 3929 | 1.44e-3 | 7.8e-4 | 1.2e-4 |

The graded grid is as accurate as the uniform fine one with a third of its cells; the time steps are the fine
grid's, since the smallest cell sets the time step. Putting the slab's faces on cell edges does not help here:
subpixel averaging already resolves faces inside cells.

With `--case full` (`results/graded_slab_full.json`): the rotated uniaxial tensor with an xz entry of
`benchmarks/subpixel/em_slab.py` (polarization conversion), `averaging="subpixel_full"`:

| z grid | z cells | max \|t - t_ref\| | max \|\|r\| - \|r_ref\|\| | energy error |
|---|---|---|---|---|
| uniform 10 nm | 360 | 9.8e-4 | 5.5e-4 | 5e-6 |
| uniform 40 nm | 90 | 1.17e-2 | 5.1e-3 | 1.3e-5 |
| graded 10 / 40 nm | 121 | 1.10e-3 | 2.3e-4 | 5.4e-5 |
| graded, the slab's faces on edges | 122 | 1.36e-3 | 7.3e-4 | 1.2e-4 |
