# NIRB Reproduction

This directory is being reworked so the primary benchmark path is generated inside
`pycutfem` and the downloaded paper repositories are used only as validation
references.

Current local-first commands:

```bash
conda run --no-capture-output -n fenicsx python examples/NIRB/run_example2_reference.py
conda run --no-capture-output -n fenicsx python examples/NIRB/assemble_example2_local.py \
  --output-dir examples/NIRB/artifacts/example2_local_smoke \
  --mesh-size 0.20 --mesh-order 1 --field-order 1 --reference-velocity 2.5 --backend python
conda run --no-capture-output -n fenicsx python examples/NIRB/run_example2_local.py \
  --output-dir examples/NIRB/artifacts/example2_local_fom_smoke_h04 \
  --mesh-size 0.04 --mesh-order 1 --poly-order 1 --pressure-order 1 \
  --dt 0.008 --end-time 0.008 --max-steps 1 --max-coupling-iters 12 \
  --coupling-abs-tol 1e-6 --coupling-rel-tol 1e-6 \
  --force-update constant --force-relaxation 0.25 --force-history 50 --force-regularization 1e-10 \
  --newton-tol 1e-6 --max-newton-iter 8 \
  --bossak-alpha -0.3 --dynamic-tau 1.0 --pressure-gauge 1e-5 \
  --backend cpp --linear-backend petsc --verbose
bash examples/NIRB/launch_example2_local_tmux.sh
```

Reference-only utilities that still exist:

```bash
conda run --no-capture-output -n fenicsx python examples/NIRB/run_example1.py
conda run --no-capture-output -n fenicsx python examples/NIRB/run_example2.py
```

Notes:

- `run_example2_reference.py` parses the downloaded `DoubleFlap` JSON and MDPA
  files and writes the geometry, interface, and boundary-condition summary that
  the local pycutfem port will target.
- `assemble_example2_local.py` builds a conforming local DoubleFlap mesh,
  classifies the local fluid/solid subdomains, applies the Example 2 boundary
  conditions, and assembles the local Neo-Hookean FSI residual/Jacobian once.
- `run_example2_local.py` is the local strong staggered FSI driver. It solves
  the solid, harmonic mesh extension, and ALE fluid subproblems in a fixed-point
  loop on the interface load `f_{k+1} = F(S(f_k))`, supports relaxed
  `constant`, `aitken`, and local `iqnils` load updates, exports `coSimData`
  snapshot arrays, and writes a per-subiteration `timeseries.csv`.
- The local fluid step now mirrors the Kratos benchmark configuration more
  closely:
  - Bossak time integration with the same default `alpha = -0.3`.
  - ALE convective velocity `u - w_mesh`.
  - DVMS-style algebraic stabilization with `C1 = 8`, `C2 = 2`,
    `dynamic_tau = 1.0`, `tau_one`, `tau_two`, and the pressure-subscale
    tracking term `tau_p`, implemented locally as a supported streamline /
    pressure / grad-div split in the pycutfem form language.
  - Persistent fluid acceleration history between time steps, so the local
    first-step and subsequent-step solves use the same state variables that the
    Kratos Bossak scheme advances.
- The Bossak/DVMS terms were mirrored from the Kratos source used by the paper:
  - `applications/FluidDynamicsApplication/python_scripts/navier_stokes_monolithic_solver.py`
  - `applications/FluidDynamicsApplication/python_scripts/navier_stokes_ale_fluid_solver.py`
  - `applications/FluidDynamicsApplication/custom_strategies/schemes/residualbased_predictorcorrector_velocity_bossak_scheme_turbulent.h`
  - `applications/FluidDynamicsApplication/custom_elements/d_vms.cpp`
  - `applications/FluidDynamicsApplication/custom_elements/qs_vms.cpp`
- The local implementation is still an algebraic pycutfem port of those terms,
  not a line-for-line copy of the Kratos element. The validation target remains
  the Kratos first-step `u`/`v` fields before launching the long snapshot job.
- The current best local step-1 benchmark match uses `mesh-size=0.04`,
  `pressure-gauge=1e-5`, and a conservative `constant` relaxation of `0.25`.
  Against the live Kratos step-1 fields, snapshot 2 of that run currently gives
  `disp_rel_l2 ≈ 0.405` and `velocity_rel_l2 ≈ 0.641`; see
  `artifacts/example2_local_re301_step1_h004_g1e-5_const025/comparison_snapshot2.json`.
- `launch_example2_local_tmux.sh` now runs the local staggered driver in `tmux`
  with a persistent log file using the current best one-step monitoring
  configuration on the `cpp` assembly + PETSc linear-solver path.
- `launch_example2_dealii_tmux.sh` still exists as a bridge to the older
  `examples/fsi_dealii_reference.py` path while the full Example 2 driver is
  being ported.
- The old clone-and-train scripts are no longer the intended primary workflow.
