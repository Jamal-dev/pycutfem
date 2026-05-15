# Turek 2D-3 Navier-Stokes MOR

This example is a native-online MOR workflow for the 2D-3 Turek cylinder
benchmark.  It is intentionally outside `examples/NIRB`.

The example demonstrates:

- full-order Taylor-Hood Navier-Stokes snapshots on the Turek channel/cylinder
  mesh;
- Dirichlet lifting before POD, so velocity snapshots are homogeneous on inlet,
  walls, and the cylinder;
- generic mixed-field lift enrichment, used here as pressure supremizers for
  inf-sup stability;
- projection cross-validation for velocity, pressure, lift, and QDEIM
  collateral mode counts;
- QDEIM/DEIM non-affine decomposition of the convective residual;
- native C++ online Gauss-Newton solves against generated UFL kernels, with no
  Python callbacks inside the nonlinear iteration.

The fluid stress is written in 2D form:

```python
sigma_f = -p * I + 2 * mu * dev(eps(u))
dev(eps(u)) = eps(u) - 0.5 * trace(eps(u)) * I
```

Use the conda environment from the repo:

```bash
conda run --no-capture-output -n fenicsx python examples/turek_navier_stokes_mor/turek_2d3_mor.py smoke --output-dir /tmp/turek_mor_smoke
conda run --no-capture-output -n fenicsx python examples/turek_navier_stokes_mor/turek_2d3_mor.py offline-snapshots --output-dir examples/turek_navier_stokes_mor/artifacts
conda run --no-capture-output -n fenicsx python examples/turek_navier_stokes_mor/turek_2d3_mor.py train --snapshot-file examples/turek_navier_stokes_mor/artifacts/turek_2d3_snapshots.npz --output-dir examples/turek_navier_stokes_mor/artifacts --select-modes
conda run --no-capture-output -n fenicsx python examples/turek_navier_stokes_mor/turek_2d3_mor.py online-native --rom-file examples/turek_navier_stokes_mor/artifacts/turek_2d3_rom.npz
conda run --no-capture-output -n fenicsx python examples/turek_navier_stokes_mor/turek_2d3_mor.py validate --snapshot-file examples/turek_navier_stokes_mor/artifacts/turek_2d3_snapshots.npz --rom-file examples/turek_navier_stokes_mor/artifacts/turek_2d3_rom.npz
```

One verified coarse end-to-end command is:

```bash
conda run --no-capture-output -n fenicsx python examples/turek_navier_stokes_mor/turek_2d3_mor.py benchmark --output-dir /tmp/turek_mor_e2e_final --mesh-size 0.055 --max-steps 6 --velocity-modes 2 --pressure-modes 1 --supremizer-modes 1 --deim-modes 16 --backend cpp --validation-index 0
```

The script records the boundary-condition time used by the theta scheme
(`t_n + theta * dt`) and uses that same time for native online lifting during
validation.  This is essential for pressure accuracy because pressure reacts
strongly to a small mismatch in the imposed inlet state.

For production timing, increase `--max-steps`, decrease `--mesh-size`, and use
`--mesh-backend gmsh --mesh-file ...` if you want a persistent Gmsh mesh.  The
default command line uses the built-in structured O-grid backend so the example
does not depend on Gmsh for smoke runs.

For longer runs, launch the benchmark detached and write a log:

```bash
tmux new-session -d -s turek_mor_realistic -c /home/bhatti/Documents/pycutfem 'conda run --no-capture-output -n fenicsx python examples/turek_navier_stokes_mor/turek_2d3_mor.py benchmark --output-dir /tmp/turek_mor_realistic --mesh-size 0.04 --max-steps 12 --select-modes --velocity-mode-candidates 2,4,6 --pressure-mode-candidates 1,2,3 --supremizer-mode-candidates 1,2,3 --deim-mode-candidates 24,32,48 --deim-modes 32 --backend cpp --validation-index -1 2>&1 | tee /tmp/turek_mor_realistic.log'
```

Latest validation on 2026-05-15 used `--mesh-size 0.04 --max-steps 12
--select-modes` and selected 6 velocity modes, 3 pressure modes, 2 lift modes,
and 48 QDEIM modes on a 6669-DOF problem.  The native online solve converged in
3 iterations with residual `1.48e-10`, velocity relative error `1.07e-03`,
pressure shifted relative error `3.07e-03`, projection relative error
`2.05e-03`, and speedup `6.15x` against the mean FOM step.
