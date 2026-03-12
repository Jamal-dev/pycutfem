# Blauert et al. (2015) — one-domain benchmark investigation

## Goal (FSI-only validation)
Validate the **fluid–structure interaction** behavior of the one-domain biofilm model
(`examples/utils/biofilm/one_domain.py`) against **Blauert et al. (2015)** using the
supplementary OCT deformation video (`mmc1`).

For this benchmark we disable all biofilm biology/damage:
- Growth / detachment / nutrient terms: **off** (`k_g=k_d=k_det=0`, transport terms set to 0).
- Damage evolution: **off** (no `d` field provided).
- Indicator/porosity evolution PDEs: **on by default** in the monolithic solve (`--transport-mode pde`) so `alpha` and `phi`
  evolve consistently with the FSI coupling (no growth sources in this benchmark).
  - Use `--transport-mode refmap` to recover the earlier reference-map update where `alpha(x,t)=alpha0(x-u)` and `phi` is tied to `alpha`.

The intent is: *can the FSI mechanics + coupling reproduce the deformation response?*

## 2026-03-12 short investigation

### Benchmark 6 formulation and scientific success

- Paper formulation (`verification/benchmark6_overview.tex`, `verification/benchmark6_blauert_channel.tex`):
  - domain `L=5.5 mm`, `H=1.0 mm`,
  - parabolic inflow with `u_avg = 4.56e-2 m/s` and cosine ramp `t_ramp = 2 s`,
  - transported `(alpha, mu_alpha)` via conservative CH transport,
  - transported `phi`,
  - no growth / substrate / detached biomass / damage,
  - optional benchmark-local diffuse tangential traction `g_Gamma^n = zeta_tau r(t_n) tau_P(y) t_Gamma^n`, evaluated IMEX at the accepted previous time level.
- Reviewer-facing success in the paper:
  - global front motion and front motion at `y = 150, 250, 350 um`,
  - contour mismatch at representative times,
  - mesh sensitivity of those errors.
- Wrapper-level success checks:
  - `dynamic_08pa` requires `t_final >= 10 s` and targets `front_compression_2p0_um = 148`, `front_plateau_drift_2p0_10p0_um = 0`, `porosity_drop_2p0_pp = 2`,
  - `steady_dian` tracks contour RMSE / MAE / max and `steady_front_y150_err_um`,
  - calibration score weights are `0.45` contour RMSE, `0.15` front-150 error, `0.20` 2 s compression error, `0.10` plateau drift, `0.10` porosity-drop error.

### Ranked root-cause hypotheses

1. Immediate blocker for the current short benchmark-6 probe: the internal PDAS linear solve fails in the PETSc Schur/FGMRES path before any LM step can be attempted.
   - Evidence from the exact short probe at `/tmp/b6_schur_probe_fast_diag_escalated`:
     - `VI Newton 1: |G|_inf=9.96e-04 |R|_inf=9.96e-04 nA=49`,
     - `[vi-set] alpha:lo=0,hi=0  phi:lo=0,hi=49`,
     - `[vi-lm] skipped: nA=49 active VI dofs keep PDAS mode enabled. active={phi:lo=0,hi=49}`,
     - `[ksp] type=fgmres pc=fieldsplit reason=-3 its=10000 rnorm=5.731e-04`,
     - `[fail_blocks] momentum:9.96e-04, mass:1.05e-06, ...`,
     - `[fail_mom_terms] gamma_div:9.66e-04, viscous:1.92e-05, supg:8.03e-06, convection:3.31e-06, ...`.
   - Control evidence: the same case with direct LU (`/tmp/b6_direct_probe`) reached `t=0.025 s` in 3 Newton iterations with `|G|_inf=8.14e-07`.

2. The current LM path is usually disabled in Benchmark 6 exactly when the hard regime appears, because the implementation only enables LM when the VI active set is empty.
   - Code: `pycutfem/solvers/nonlinear_solver.py` uses `use_lm = unconstrained_lm and nA == 0`.
   - Driver: `examples/biofilms/benchmarks/blauert/blauert_biofilm_deformation_one_domain.py` sets box bounds on both `alpha` and `phi`.
   - Existing benchmark-6 logs show nonempty active sets immediately:
     - `/tmp/benchmark6_diag_E320_z30_maxit7/calibration/E320_eta0_zeta30_nx24/run.log`: `nA=255` at Newton 1, `nA=589` at Newton 2,
     - `/tmp/benchmark6_zeta30_gdiv01.log`: `nA=310` at Newton 1, `nA=515` at Newton 2.

3. Even when LM is allowed to run, it is not a VI-aware LM globalization; it is a shifted Newton step accepted on the raw residual merit `0.5 ||R||^2`, not on the semismooth residual `G`.
   - `_vi_build_unconstrained_lm_system()` forms `(A + lambda D) s = -R`.
   - `_eval_reduced_trial()` evaluates `R_try` only.
   - `_vi_unconstrained_lm_model()` and the accept/reject logic use the raw residual merit, not a semismooth complementarity merit.
   - That matches the unit tests in `tests/test_internal_pdas_controls.py`, which only verify the shifted-system algebra and a scalar toy solve, not a benchmark-6 active-set case.

4. Secondary diagnostic suspicion: the Schur/FGMRES configuration is either under-preconditioned for this reduced system or its effective iteration cap is not behaving as intended.
   - The failing probe reports `its=10000` even though the command sets `--linear-ksp-max-it 200`.
   - Treat this as secondary until hypotheses 1-3 are resolved; the direct-LU control already shows the short-horizon problem is otherwise solvable.

5. The paper wrapper was not actually running the Benchmark-6 forcing from the paper source: it defaulted to `t_ramp = 0.2 s` even though the paper benchmark and the checked-in investigation note specify `t_ramp = 2.0 s`.
   - Code: `examples/biofilms/benchmarks/blauert/paper1_benchmark6_blauert_channel.py` defaulted `--t-ramp` to `0.2`.
   - Paper source: `verification/benchmark6_blauert_channel.tex` states `t_ramp = 2 s`.
   - Observed effect:
     - old wrapper smoke (`/tmp/b6_wrapper_smoke_steady`) already reached `dx_front_y250 ≈ 42.7 um` at `t = 0.2 s` and `64.7 um` at `t = 0.25 s`,
     - but the extracted video stays near zero over the same window (`dx_front_y250 ≈ 0 um` at `t ≈ 0.20-0.23 s`),
     - after correcting the wrapper to `t_ramp = 2.0 s`, the same candidate (`E=200, zeta=30, gamma_u=5`) dropped to `dx_front_y250 ≈ 0.689 um` at `t = 0.2 s` and `1.35 um` at `t = 0.25 s`, with early-time per-height RMSEs `4.22/1.20/1.17 um` at `150/250/350 um`.

6. The global front observable in the driver was not the paper's contour-based quantity; it used a quantile over all DOFs with `alpha >= 0.5`, which explains the repeated `dx_front_global = 0` even when the tracked contour fronts moved.
   - Code: `_x_front_global_quantile()` in `examples/biofilms/benchmarks/blauert/blauert_biofilm_deformation_one_domain.py` used the DOF set `alpha >= alpha_half`.
   - Paper source: `verification/benchmark6_overview.tex` says the front-displacement histories are extracted from the transported `alpha = 1/2` contour.
   - Observed effect:
     - long/short runs before the fix reported `dx_front_global = 0` while `dx_front_y250` and `dx_front_y333` were nonzero,
     - after switching the observable to the contour points, the same `t <= 0.25 s` probe produced a nonzero global history with `dx_front_global ≈ 0.140 um` at `t = 0.2 s` and `0.295 um` at `t = 0.25 s`,
     - the early-time global comparison against the video extraction became finite and small: RMSE `1.17 um` over `t ∈ [0, 0.2336] s`.

### Files to inspect or edit

- `pycutfem/solvers/nonlinear_solver.py`
  - PDAS active-set assembly and Newton loop,
  - LM gate and LM model,
  - PETSc Schur/FGMRES linear-solve path.
- `examples/biofilms/benchmarks/blauert/blauert_biofilm_deformation_one_domain.py`
  - benchmark-6 argument plumbing,
  - bound setup for `alpha` / `phi`,
  - linear-solver environment wiring.
- `examples/biofilms/benchmarks/blauert/paper1_benchmark6_blauert_channel.py`
  - scientific success metrics, required observation windows, paper outputs.
- `examples/biofilms/benchmarks/blauert/compare_sim_vs_observations.py`
  - exact observation targets and measured errors.
- `examples/biofilms/benchmarks/blauert/investigation.md`
  - keep this note aligned with observed logs and the next validation results.

### Minimal diagnostic edit already applied

- `pycutfem/solvers/nonlinear_solver.py` now prints an explicit `[vi-lm] skipped: ...` message when `--vi-unconstrained-lm` is enabled but `nA > 0`.
- This is diagnostic only. It does not change the PDAS/LM algorithm; it only makes it visible when LM is not actually in play.

### Short validation commands

1. Reproduce the immediate short-run failure in the Schur/FGMRES path and confirm whether LM is skipped or active:

```bash
env OMP_NUM_THREADS=8 BLIS_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYCUTFEM_CPP_FAST_COMPILE=0 PYCUTFEM_CPP_FAST_OPT_LEVEL=1 \
PYCUTFEM_LINEAR_KSP_TRACE=1 PYCUTFEM_VI_TRACE_LM=1 PYCUTFEM_VI_TRACE_LS=1 \
PYCUTFEM_VI_TRACE_REG=1 PYCUTFEM_VI_TRACE_FIELDS=1 \
PYCUTFEM_NEWTON_TRACE_RES_FIELDS=1 PYCUTFEM_NEWTON_TRACE_RES_FIELDS_N=10 \
conda run --no-capture-output -n fenicsx python \
examples/biofilms/benchmarks/blauert/blauert_biofilm_deformation_one_domain.py \
--backend cpp --transport-mode pde \
--diffuse-shear-traction --diffuse-shear-model poiseuille --diffuse-shear-scale 30 \
--diffuse-shear-time-scheme imex --nonlinear-solver pdas --ls-mode dealii \
--vi-unconstrained-lm --vi-lm-lambda0 1e-4 --vi-lm-lambda-max 1e6 \
--vi-lm-growth 5 --vi-lm-decay 0.5 --vi-lm-accept-ratio 1e-3 \
--vi-lm-good-ratio 0.75 --vi-lm-max-tries 6 \
--linear-ksp-type fgmres --linear-ksp-rtol 1e-8 --linear-ksp-max-it 200 \
--linear-ksp-trace --linear-schur --linear-schur-pressure-field p \
--linear-schur-fact upper --linear-schur-pre selfp \
--linear-schur-rest-ksp preonly --linear-schur-rest-pc ilu \
--linear-schur-pressure-ksp preonly --linear-schur-pressure-pc jacobi \
--nx 24 --ny 8 --dt 0.025 --t-final 0.025 --theta 1.0 --q 4 --t-ramp 0.2 \
--E 320 --nu 0.4 --solid-visco-eta 0 --u-avg 0.1777777778 --rho-f 1000 \
--kappa-inv 9.81e11 --phi-b 0.47 --gamma-u 5 --u-extension l2 --gamma-u-pin 1e-4 \
--kinematics-scale 1000 --alpha-ch-M 1e-12 --alpha-ch-gamma 2e-3 --alpha-ch-eps 2e-5 \
--diffuse-shear-scale-ref 50 --scale-alpha-ch-eps-with-zeta \
--alpha-advection-form conservative --alpha-supg 0.5 --alpha-cip 0.0 \
--v-supg 1 --v-supg-mode residual --v-supg-c-nu 4 \
--u-supg 1 --u-cip 1 --u-cip-weight biofilm --v-cip 1 --vS-cip 1 \
--global-front-quantile 0.9 --dx-quantile 0.9 \
--gamma-div 0.1 --adaptive-gamma-div --gamma-div-max 0.2 \
--newton-tol 1e-6 --max-it 8 --dt-min 0.005 \
--refine-biofilm --refine-band 2.5e-4 --refine-expand-layers 1 \
--restart-write-every 1 --vtk-every 0 \
--out-dir /tmp/b6_schur_probe_fast_diag_escalated
```

2. Isolate the linear solver as the immediate blocker by keeping the same benchmark-6 setup but replacing Schur/FGMRES with direct LU:

```bash
env OMP_NUM_THREADS=8 BLIS_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYCUTFEM_CPP_FAST_COMPILE=0 PYCUTFEM_CPP_FAST_OPT_LEVEL=1 \
PYCUTFEM_LINEAR_KSP_TRACE=1 PYCUTFEM_NEWTON_TRACE_RES_FIELDS=1 \
PYCUTFEM_NEWTON_TRACE_RES_FIELDS_N=10 \
conda run --no-capture-output -n fenicsx python \
examples/biofilms/benchmarks/blauert/blauert_biofilm_deformation_one_domain.py \
--backend cpp --transport-mode pde \
--diffuse-shear-traction --diffuse-shear-model poiseuille --diffuse-shear-scale 30 \
--diffuse-shear-time-scheme imex --nonlinear-solver pdas --ls-mode dealii \
--vi-unconstrained-lm --vi-lm-lambda0 1e-4 --vi-lm-lambda-max 1e6 \
--vi-lm-growth 5 --vi-lm-decay 0.5 --vi-lm-accept-ratio 1e-3 \
--vi-lm-good-ratio 0.75 --vi-lm-max-tries 6 \
--linear-ksp-type preonly --linear-pc-type lu --linear-pc-factor-solver-type mumps \
--linear-ksp-rtol 1e-8 --linear-ksp-max-it 200 --linear-ksp-trace --no-linear-schur \
--nx 24 --ny 8 --dt 0.025 --t-final 0.025 --theta 1.0 --q 4 --t-ramp 0.2 \
--E 320 --nu 0.4 --solid-visco-eta 0 --u-avg 0.1777777778 --rho-f 1000 \
--kappa-inv 9.81e11 --phi-b 0.47 --gamma-u 5 --u-extension l2 --gamma-u-pin 1e-4 \
--kinematics-scale 1000 --alpha-ch-M 1e-12 --alpha-ch-gamma 2e-3 --alpha-ch-eps 2e-5 \
--diffuse-shear-scale-ref 50 --scale-alpha-ch-eps-with-zeta \
--alpha-advection-form conservative --alpha-supg 0.5 --alpha-cip 0.0 \
--v-supg 1 --v-supg-mode residual --v-supg-c-nu 4 \
--u-supg 1 --u-cip 1 --u-cip-weight biofilm --v-cip 1 --vS-cip 1 \
--global-front-quantile 0.9 --dx-quantile 0.9 \
--gamma-div 0.1 --adaptive-gamma-div --gamma-div-max 0.2 \
--newton-tol 1e-6 --max-it 8 --dt-min 0.005 \
--refine-biofilm --refine-band 2.5e-4 --refine-expand-layers 1 \
--restart-write-every 1 --vtk-every 0 \
--out-dir /tmp/b6_direct_probe
```

3. Isolate the LM implementation itself outside the benchmark-6 active-set regime:

```bash
conda run --no-capture-output -n fenicsx python -m pytest \
tests/test_internal_pdas_controls.py -k unconstrained_lm -q
```

### Measurable success criteria for a healthy run

- Healthy short validation run:
  - reaches `t = 0.025 s` without PETSc KSP failure or line-search failure,
  - converges the first step in at most 3 Newton iterations on the current `nx=24, ny=8` probe,
  - reduces `|G|_inf` below `1e-6`,
  - accepts full steps (`alpha = 1`) and does not trigger dt reduction.
- Healthy benchmark-6 production run:
  - completes the observation window required by the selected paper scenario (`dynamic_08pa` needs `t_final >= 10 s`; `steady_dian` needs a finite steady snapshot),
  - produces finite values for the required metrics in `benchmark6_blauert_channel_summary.csv` / `.json`,
  - writes the paper assets
    - `benchmark6_blauert_channel_history.png`,
    - `benchmark6_blauert_channel_contours.png`,
    - `benchmark6_blauert_channel_mesh_sensitivity.png`,
  - preserves restart checkpoints so any late-time failure can be restarted from the last accepted step.

### Logs and metrics to monitor during long runs

- Solver iteration metrics:
  - `|G|_inf`, `|R|_inf`, `nA`, `nI`, `DeltaA`, `|R_I|_inf`, `|gap_A|_inf`, `nA_strong`, `lambda`.
- Active-set ownership:
  - per-field active counts for `alpha` and `phi` from `[vi-set]`.
- Residual ownership:
  - `[res]` and `[fail_res]` with special attention to `v_x`, `v_y`, `vS_x`, `vS_y`.
- Momentum decomposition:
  - `[fail_blocks]`, `[fail_mom_terms]`, and `[fail_slip]` when a step is rejected.
- Linear solver behavior:
  - `[ksp-schur]` split sizes,
  - `[ksp]` reason / iteration count / residual norm.
- Time-step robustness:
  - `gamma_div_history.csv`,
  - dt reductions, retries, and accepted `nNewton`.
- Scientific outputs:
  - `timeseries.csv` columns `dx_front_global`, `dx_front_y150um`, `dx_front_y250um`, `dx_front_y350um`, `phi_mean_alpha_weighted`,
  - observation comparison JSON files under each case directory,
  - summary CSV/JSON and generated paper figures.

## Literature reference points

## 2026-03-12: corrected-observable calibration queue

The corrected best active-porosity branch
`/tmp/b6_driver_E500_z100_gU10_rho1000_nx24_t2p0_obsfix`
is solver-healthy but fails with a split late-time shape error:

- global/toe front still under-advances (`-30.4 um` at `t=2.0 s`),
- tracked heights overshoot by `+29 / +48 / +42 um` at `150 / 250 / 350 um`,
- contour `front_y150_err_um` grows from `58.4 um` at `1.0 s` to `131.5 um`
  at `2.0 s`.

That pattern matters: it is not a pure amplitude deficit. The current
`poiseuille` closure is tangential-only, so the first new model-level canary is
to re-check the full lagged traction closure on the corrected observable path.

Ranked Batch A for the next screening window (`t_final = 1.5 s`, direct LU):

1. `lagged_stress`, `diffuse_shear_scale = 10`, `E = 500`, `gamma_u = 10`,
   `rho_f = 1000`, `nu = 0.4`, `kappa_inv = 9.81e11`.
   Hypothesis: missing normal compression is the main reason the toe lags while
   the upper contour overshoots.
2. `poiseuille`, `gamma_u = 12.5`, same base branch otherwise.
   Hypothesis: slightly stronger extension penalty suppresses the late upper
   overshoot without changing the onset.
3. `poiseuille`, `kappa_inv = 1.962e12`, same base branch otherwise.
   Hypothesis: stronger drag redistributes the fluid loading into a broader
   compressive response.
4. `poiseuille`, `nu = 0.45`, same base branch otherwise.
   Hypothesis: near-incompressibility changes the contour shape more than the
   scalar amplitude.

All Batch A probes keep:

- corrected global observable (`row-wise leftmost contour quantile`),
- `global_front_quantile = 0.05`, `dx_quantile = 0.05`,
- `u_avg = 0.0456`, `t_ramp = 2.0`,
- direct LU / MUMPS,
- `vi_lm_good_ratio = 0.05`,
- `accept_nonconverged_atol_factor = 4`.
Source: `examples/biofilms/benchmarks/blauert/blauert.tex`.

- Dynamic experiment shows an upstream/front compression measured as the distance from the
  left side of the B-scan to the biofilm.
- Text reports: **148 µm after ~2 s** (section “Dynamic Biofilm Deformation”).
  Note: the text mentions `τ_w = 0.6 Pa` while the Figure 2 caption states `τ_w = 0.8 Pa`.

## Video extraction (experimental time series)
Script: `examples/biofilms/benchmarks/blauert/extract_front_displacement_from_video.py`.

### Scale and cropping
- Scale bar detection: `250 µm` corresponds to **159 px**, giving
  `px_size ≈ 1.57233 µm/px`.
- Substrate reference: the extractor uses the **top** of the bright substrate band as `y=0`.
- Per-frame segmentation: **Otsu threshold + morphology**, then keep the **largest connected**
  component and filter out obvious “bad frames” by segmented area (handles the ~1.1 s cut/glitch).

### Outputs
- `examples/biofilms/benchmarks/blauert/exp_front_displacement_from_video.csv`
  - global `x_front(t)` (quantile over segmented pixels)
  - optional `x_front(y,t)` at user-specified y-levels
- `examples/biofilms/benchmarks/blauert/exp_frame0_polygon_mm.csv`
  - closed polygon (x_mm,y_mm) used to initialize `alpha0`.
  - **Important**: the robust/recommended source is the Matlab preprocessing contour
    `biofilm_preprocessing/biofilm.txt` via:
    ```bash
    python examples/biofilms/benchmarks/blauert/extract_front_displacement_from_video.py \
      --polygon-source matlab_preprocessing --matlab-shift-um 0 --t-max 0
    ```
    (`--matlab-shift-um 0` keeps coordinates in the 2 mm imaged window so the driver’s default
    `--alpha0-tx 0.5mm` shifts it into the 5.5 mm channel.)

### Quick sanity values (current extractor defaults)
From `exp_front_displacement_from_video.csv`:
- At `t ≈ 2.00 s`: `dx_front ≈ 91 µm` and `dx_front_y250um ≈ 94 µm`
- A plateau around `~83–91 µm` appears for `t ≈ 1.4–8 s`; a drop near `t ≈ 9–10 s` likely
  corresponds to the flow being reduced/stopped near the end of the clip.

This is **below** the 148 µm value reported in the manuscript text; the discrepancy is
documented here and should be revisited (segmentation choices vs the paper’s ImageJ workflow).

## One-domain benchmark driver (simulation)
Driver: `examples/biofilms/benchmarks/blauert/blauert_biofilm_deformation_one_domain.py`.

### Geometry / initialization
- Computational domain: rectangle `[0,L]×[0,H]` (default `L=5.5 mm`, `H=1.0 mm`).
- `alpha0` from `exp_frame0_polygon_mm.csv` (mm→m) with optional translation `alpha0_tx`
  to provide inflow/outflow padding.
- FSI-only “refmap” update after each accepted time step:
  `alpha(x,t) = alpha0(x - u(x,t))`, and porosity tied/frozen:
  `phi = 1 - (1-phi_b) alpha`.
  - This is controlled by `--transport-mode refmap`.

**PDE transport mode**
- With `--transport-mode pde` (default), `alpha` and `phi` are solved as PDEs (instead of being overwritten from the refmap).
- `alpha` uses conservative advection by default (`--alpha-advection-form conservative`).
- Conservative Cahn–Hilliard regularization is enabled by `--alpha-ch-M` + `--alpha-ch-gamma` (both nonzero).

**Geometry sanity check (Matlab preprocessing)**
- `biofilm_preprocessing/data_processing.m` uses `L_mu=2000 µm` and `shift=500 µm` and sets
  `y_max_mu = L_mu*4/8 = 1000 µm` and `x_max_mu = L_mu + 7*shift = 5500 µm`.
  This matches the driver defaults `--L 5.5e-3`, `--H 1.0e-3`.
- The same preprocessing contour (`biofilm_preprocessing/biofilm.txt`) maps to an initial biofilm
  height `y_max ≈ 432 µm`, i.e. top-wall clearance `≈ 568 µm` (so the biofilm should **not** be
  near-blocking the channel).

### “Active rectangle” DOF restriction (stability)
To avoid having to project displacement onto an evolving active set, we follow a **fixed**
active box approach:
- Compute the initial biofilm bounding box from the polygon (+ pad).
- Mark `(u, vS)` DOFs **outside** this box as **inactive**.
- Inside the box, keep the standard extension penalties (`gamma_u`, `u_extension_mode`).

This matches the intended workflow: “big rectangle around the biofilm; outside inactive DOFs;
inside use extension/penalty operators”.

### Local mesh refinement around the biofilm (hanging nodes)
To improve resolution near the biofilm without globally increasing `nx,ny`, the driver now supports
a **single-pass** localized refinement (2:1 mesh) around the **initial biofilm bbox**:
- Enable with `--refine-biofilm`.
- Control the refined band size with `--refine-band` (default `~2*max(hx,hy)`).
- Optionally expand by neighbor layers with `--refine-expand-layers`.

This produces hanging nodes (single refinement level); the solver handles them via its linear
constraint layer.

### Boundary conditions
- Fluid: parabolic inflow at `left`, no-slip at `top/bottom`, pressure reference `p=0` at `right`.
- Skeleton: clamped at `bottom` (`u=vS=0`) to represent attachment (no detachment in this run).
- Inflow magnitude is set by `--u-avg` (default `6.84e-2 m/s`, matching Dian’s `v0 = 6.84×10^-2 m/s` used for the Blauert comparison).

### Inflow scaling when changing `H`
Our driver prescribes a **velocity** profile using `--u-avg`. For plane Poiseuille between plates:
- pressure gradient: `dp/dx = -12 μ u_avg / H^2`
- wall shear: `τ_w = 6 μ u_avg / H`

So if you increase `H` but keep `u_avg` fixed, the **shear stress decreases** (and the biofilm deforms less).
To keep the same `τ_w` as you change `H`, scale `u_avg ∝ H` (i.e. keep `u_avg/H` constant).

### Solver and stability notes
- Backend: `--backend cpp` recommended.
- Linear solver: PETSc SNES (`newtonls`) + LU (`mumps`).
- For stability at stronger forcing:
  - increase inflow ramp time `--t-ramp` (e.g. 2.0 s),
  - enable `--allow-dt-reduction` with a reasonable `--dt-min`.

### Consistent conditioning fix: augmented-Lagrangian `--gamma-div`
We observed a **repeatable SNES stall** in the transient momentum block for the fully convective case
(`--fluid-convection full`): SNES iterations reduce the residual quickly but then stagnate just above `--newton-tol`
(typically dominated by `v_x` at the upstream foot of the biofilm). This happens even at very small `dt`
and is therefore primarily a **conditioning / saddle-point robustness** issue rather than a physical instability.

Fix: enable the consistent augmented-Lagrangian term
`γ_div * (div(C v + B vS), div(C w) + div(B η))` in the momentum/skeleton equations:
- it is **consistent** (vanishes when the mixture volume constraint holds),
- it acts like a **grad-div penalty** and improves conditioning of the pressure/constraint coupling in transient runs,
- it does **not** rely on loosening tolerances (works with `--newton-rtol 0` and `--accept-nonconverged-atol-factor 0`).

Recommended scale: `--gamma-div ≈ mu_f` (for water, `mu_f≈1e-3 Pa*s`).

Concrete repro from checkpoint `out/_blauert_pde/restart/checkpoint_step=00042.npz` (t≈0.306 s):
- Without `--gamma-div`: stalls at `‖F‖_inf≈1.12e-6` and fails at `--max-it 15`.
- With `--gamma-div 1e-3`: converges in **3 Newton iterations** to `‖F‖_inf≈4.8e-7`.

Example:
```bash
conda run --no-capture-output -n fenicsx python -u \
  examples/biofilms/benchmarks/blauert/blauert_biofilm_deformation_one_domain.py \
  --backend cpp --nx 60 --ny 20 \
  --restart-from out/_blauert_pde/restart/checkpoint_step=00042.npz --restart-dt 1e-3 \
  --dt-min 1e-3 --allow-dt-reduction \
  --newton-tol 1e-6 --newton-rtol 0 --accept-nonconverged-atol-factor 0 \
  --fluid-convection full --gamma-div 1e-3
```

**Interpreting SNES “reason=-5” in this benchmark**
- PETSc SNES reports `reason=-5` when it hits `--max-it` without satisfying its stopping criterion.
- In this repo we run SNES with an **infinity-norm** stopping criterion by default (`‖F‖_inf`), so `--newton-tol`
  matches the usual “max residual entry” interpretation used by the pure-Python Newton solver.
- PETSc can still return a nonconverged reason (e.g. `-5`) even when the returned iterate satisfies `‖F‖_inf ≤ atol`
  (common when SNES uses a 2-norm internally on large reduced systems). The solver now treats `‖F‖_inf ≤ atol` as
  **converged** and prints a one-line warning (this is *not* a tolerance relaxation).
- If `‖F‖_inf` is only slightly above `--newton-tol` and the residual breakdown is dominated by `v_*`, it is usually an
  iteration/line-search robustness issue in the transient fluid block rather than a physical instability.
- `--accept-nonconverged-atol-factor` remains an explicit escape hatch (default disabled) when you *want* to accept a
  best iterate with `‖F‖_inf` near `atol` even if strict convergence is not achieved.

#### Observed Newton failure mode (fluid momentum block)
When the long run fails (e.g. `step≈27`, `t≈0.8438 s` in the log you shared), the residual tracing shows:
- `p, u, vS` are essentially converged (≈ machine precision),
- only the **fluid velocity** residuals (`v_x, v_y`) stagnate at `O(10^{-3})`,
- PETSc SNES exits with `DIVERGED_LINE_SEARCH` or `DIVERGED_MAX_IT`.

This can be either:
- a **conditioning-driven stall** near the tolerance (fixed robustly by `--gamma-div`), or
- a true **time-step/inertia driven** failure when the nonlinear change per step is too large.

Concrete reproduction (coarse `nx=60, ny=20`):
- With `rho_f=1000` and `dt=0.05`, the `t=0.55→0.60 s` step stalls at `‖F‖≈2.3e-3` dominated by `v_y≈7e-4`.
- Restarting from the same checkpoint with `--restart-dt 0.025` converges in ~6 SNES iterations (`‖F‖≈1e-8`).
- Setting `--rho-f 0` (quasi-static Stokes/Brinkman) removes the failure mode for this benchmark (the same step
  converges immediately).

Practical guidance:
- If you want to keep inertia (`rho_f>0`), set `--dt-min` smaller than `0.01` (e.g. `1e-3`) and use restart:
  `--restart-write-every 1`, then `--restart-from .../checkpoint_step=XXXXX.npz --restart-dt 0.005`.
- If the goal is *FSI-only deformation validation vs video* (fluid adjusts quickly), using quasi-static flow
  (`--rho-f 0`) is a reasonable and much more robust setting to start with.

### Debugging helpers (restart + residuals)
Restart / checkpoints:
- Enable checkpoints with `--restart-write-every 1` (writes `out_dir/restart/checkpoint_step=XXXXX.npz` and `checkpoint_latest.npz`).
- Resume with `--restart-from out_dir/restart/checkpoint_step=00026.npz` (keeps step numbering and appends to an existing `timeseries.csv` when present).
- Override the checkpoint time step with `--restart-dt <dt>` if you want to continue with a smaller/larger `dt`.

Residual diagnostics:
- On any rejected/nonconverged step the driver prints a per-field residual breakdown (`[fail_res] ...`), which usually identifies the problematic block (`v_*`, `p`, `vS_*`, `u_*`).
- For per-Newton/SNES iteration tracing, use `--trace-residual-fields` (and `--trace-residual-fields-n`) and/or `--trace-residual-worst --trace-residual-coords`.

### SUPG options
To test whether streamline diffusion improves stability:
- `--v-supg <δ0>` adds SUPG-like stabilization for the **fluid** momentum convection.
- `--u-supg <δ0>` adds SUPG-like stabilization for the **u-transport** (kinematics).

### Tracking output
`out_dir/timeseries.csv` includes:
- `x_front_global, dx_front_global` (quantile over `alpha>=0.5` DOFs)
- `x_front_y*um, dx_front_y*um` from the `alpha=0.5` contour.

The per-y tracking uses a **continuous interval selection** (closest to previous step)
to reduce jumps when multiple disconnected segments exist on a y-line.

## Error metric (simulation vs video)
Helper: `examples/biofilms/benchmarks/blauert/compare_sim_vs_video.py`.

This compares the video-extracted `dx_front(t)` (µm) against the simulation `timeseries.csv`
(`dx_front_*` stored in meters and converted to µm).

Example:
```bash
python examples/biofilms/benchmarks/blauert/compare_sim_vs_video.py \
  --out-dir out/_blauert_eta4000_coarse --t-max 0.5 --compare all
```

Current numbers for `out/_blauert_stable_short` (coarse mesh, no Kelvin–Voigt viscosity; note this run fails at
`t≈1.12 s` due to fluid momentum nonconvergence at `dt==--dt-min`):
- Overlap window printed by the tool: `t∈[0,4.94] s` (n=143): global **RMSE ≈ 77.3 µm** (MAE ≈ 69.2 µm, bias ≈ -69.2 µm).
- Per-y RMSEs: `y=150µm: 85.8 µm`, `y=250µm: 77.4 µm`, `y=350µm: 77.9 µm` (same window).

`out/_blauert_eta4000_coarse` (`solid_visco_eta≈4000 Pa*s`): global RMSE ≈ **1.41 µm** on the early window.

## Preliminary calibration observations (incomplete)
Key observation: matching the *time scale* of the video requires rate-dependence.

- With `solid_visco_eta = 0`, deformation occurs too fast (large `dx` already at `t=0.5 s`).
- Adding Kelvin–Voigt viscosity (`--solid-visco-eta`) slows the response substantially.
  Example on a coarse mesh (`nx=40, ny=15`, `dt=0.05`, `t_ramp=2.0`):
  - `solid_visco_eta ≈ 4000 Pa*s`, `u_avg≈0.1 m/s`, `E≈2000 Pa` gives
    `dx_y267um ≈ 3 µm` at `t=0.5 s` (close to the extracted ~1.6 µm), and
    reaches O(80 µm) by `t≈1.2 s`, but may require smaller `dt` / stronger damping
    to complete a full `t_final=2 s` run without step rejection.

Next calibration work should:
- Decide the target observable(s): global `dx_front(t)` vs `dx_front_y250um(t)` vs an averaged metric.
- Resolve the extraction-vs-paper discrepancy (91 µm vs 148 µm at ~2 s).
- Tune primarily: `u_avg` (shear level), `solid_visco_eta` (time scale), `E`/`nu` (amplitude),
  and possibly `kappa_inv` (fluid–skeleton coupling).

## Repro commands
Extract experimental series:
```bash
python examples/biofilms/benchmarks/blauert/extract_front_displacement_from_video.py \
  --y-levels-um 150,250,350 --t-max 10
```

Run a stable “FSI-only” short simulation (sanity):
```bash
conda run --no-capture-output -n fenicsx python -u \
  examples/biofilms/benchmarks/blauert/blauert_biofilm_deformation_one_domain.py \
  --transport-mode refmap \
  --backend cpp --nx 60 --ny 20 --dt 0.05 --t-final 0.5 --t-ramp 2.0 \
  --E 2000 --u-avg 0.02 --newton-tol 1e-5 --allow-dt-reduction --dt-min 0.01 \
  --out-dir out/_blauert_stable_short
```

Solve `alpha` + `phi` as PDEs (conservative CH + evolving porosity):
```bash
conda run --no-capture-output -n fenicsx python -u \
  examples/biofilms/benchmarks/blauert/blauert_biofilm_deformation_one_domain.py \
  --backend cpp --transport-mode pde --alpha-advection-form conservative \
  --alpha-ch-M 1e-12 --alpha-ch-gamma 2e-3 --gamma-phi 5 \
  --nx 60 --ny 20 --dt 0.01 --t-final 0.2 --allow-dt-reduction --dt-min 0.001 \
  --out-dir out/_blauert_pde_smoke
```

Compare to the extracted video time series:
```bash
python examples/biofilms/benchmarks/blauert/compare_sim_vs_video.py \
  --out-dir out/_blauert_stable_short --t-max 0.5 --compare all
```

## TODO (next iteration)
- Complete a calibrated run to `t_final=10 s` and compare to the extracted plateau.
- Decide whether to include skeleton inertia (`include_skeleton_acceleration`) for dynamics
  or rely on Kelvin–Voigt viscosity alone.

## March 8, 2026: Paper 1 reduced benchmark reset

The Paper 1 benchmark has now been reset to the actual reduced model scope:

- frozen porosity `phi_b`,
- conservative Cahn--Hilliard transport for `alpha`,
- no growth / detachment / damage / `phi` transport,
- benchmark-local diffuse interface traction correction only.

### Why the full-model path was dropped

The temporary `transport-mode pde` / active-`phi` application wrapper was explored,
but it is outside the stated Paper 1 blueprint ("freeze porosity") and did not
produce a credible calibration path for the OCT video on the available meshes.
The benchmark wrapper has therefore been moved back to the reduced Paper 1 model.

### New closure tested

The reduced driver now supports three lagged diffuse-interface traction closures:

- `lagged_velocity`: tangential projection of `2 mu_f eps(v^n) n`,
- `lagged_stress`: full lagged traction `(-p^n I + 2 mu_f eps(v^n)) n`,
- `poiseuille`: imposed background Poiseuille shear proxy.

These are all localized with `|grad(alpha)|` and transferred as equal-and-opposite
fluid/skeleton loads in the reduced one-domain formulation.

### Current calibration bracket

Observed on the coarse reduced mesh (`nx=30, ny=12`, `dt=0.05`, `u_avg=0.1 m/s`,
`t_ramp=2 s`, `rho_f=0`):

- no diffuse interface traction: essentially no measurable front motion on the
  Blauert observables;
- tangential-only closures (`lagged_velocity`, `poiseuille`): stable, but still
  too weak in the early Blauert window for publication use;
- full lagged traction (`lagged_stress`): finally produces the correct *type* of
  upstream compression, but introduces a sharp stability threshold in the pressure
  block.

Representative runs:

- `lagged_stress`, `zeta_t = 1`: still effectively flat through `t=0.35 s`;
- `lagged_stress`, `zeta_t = 5000`, `E=40 Pa`, `eta_s=500 Pa s`:
  at `t=0.40 s`, `dx_front_y250um ≈ 49.4 um`;
  at `t=0.45 s`, `dx_front_y250um ≈ 661.9 um`, i.e. clear overshoot / loss of the
  desired plateau behavior.

This means the useful calibration window is now bracketed:

- the reduced benchmark is no longer stuck at zero response,
- but the publication-ready run still requires a controlled search in
  `(E, eta_s, zeta_t, dt)` around the onset of the `lagged_stress` response.

### Benchmark 6 wrapper status

`paper1_benchmark6_blauert_channel.py` has been updated to:

- use the reduced Paper 1 model only,
- calibrate over `(E, eta_s, zeta_t)`,
- default to `diffuse_shear_model = lagged_stress`,
- score primarily with the y-resolved video histories instead of the global front
  quantile, which is too sensitive to the attached toe.

The wrapper is now structurally correct for Paper 1, but the production
calibration should not be frozen until the `lagged_stress` bracket is narrowed
to a stable `O(10^2 um)` response over the `0--2 s` Blauert window.

## March 8, 2026: active-porosity reset after Dian/Blauert review

The reduced-only reset above is now superseded for Benchmark 6. Following the
latest benchmark design review, the Blauert application case has been moved back
onto the **full active-porosity one-domain path**:

- `transport-mode pde`,
- evolving porosity `phi`,
- conservative Cahn--Hilliard transport for `alpha`,
- no growth / detachment / damage,
- no Kelvin--Voigt viscosity in the default calibration ladder.

### Code changes completed in this pass

- `examples/utils/biofilm/one_domain.py` now supports the same optional diffuse
  interface traction hook that previously existed only in the reduced
  deformation-only form builder:
  - `g_t_k`, `g_t_n`,
  - `traction_weight_k`, `traction_weight_n`,
  - optional sharp-interface `dGamma` path (unused here).
- `examples/biofilms/benchmarks/blauert/blauert_biofilm_deformation_one_domain.py`
  now passes the diffuse traction correction through the **full** one-domain
  form assembly, not only the reduced Paper 1 path.
- `examples/biofilms/benchmarks/blauert/paper1_benchmark6_blauert_channel.py`
  has been rewired away from `--paper1-reduced` and back to the active-porosity
  benchmark defaults:
  - `transport-mode pde`,
  - `u_avg = 4.56e-2 m/s`, i.e. the parabolic-flow average corresponding to
    Dian's reported peak speed `v0 = 6.84e-2 m/s`,
  - `diffuse_shear_model = poiseuille` by default,
  - benchmark-local diffuse traction uses the new `imex` time treatment by
    default, so the added load is ramped explicitly with the accepted previous
    time level instead of switching on fully at `t=0`,
  - `eta_s = 0` in the default calibration profiles.
- regression coverage was added in
  `tests/test_biofilm_one_domain_interface_traction_regression.py` and
  `tests/test_blauert_diffuse_shear_imex_ramp.py`.

### Current screening results

Short active-porosity probes (`nx=40`, `ny=15`, `dt=0.05`, `t_ramp=2.0 s`,
`E=200 Pa`, `phi_b=0.47`, `kappa_inv=9.81e11`, `gamma_phi=5`,
`alpha` solved with conservative CH) show:

- `lagged_velocity`, scale `1` and `10`, with physical `rho_f=1000`:
  still effectively inert. By `t=0.30 s` on the tested mesh,
  - `dx_front_y133um = 1.53e-07 m`,
  - `dx_front_y267um = 1.16e-07 m`,
  - `dx_front_y333um = -2.93e-07 m`,
  i.e. sub-micron motion and no practically useful scale sensitivity yet.
- `poiseuille`, scale `1`, with physical `rho_f=1000`:
  the response has the correct sign but is extremely small on the same mesh.
  At `t=0.30 s`,
  - `dx_front_y133um = 1.28e-08 m`,
  - `dx_front_y267um = 2.72e-08 m`,
  - `dx_front_y333um = -5.69e-08 m`.
- stronger `poiseuille` scales (`10^2` to `10^3`) on the full active-porosity
  path enter a much more expensive first nonlinear solve and were not yet taken
  to a completed screening point in this pass.

## March 8, 2026: Dian velocity scaling + benchmark-local IMEX traction ramp

Two benchmark-specific corrections were added after the first active-porosity
screening pass.

### 1. Consistent Blauert/Dian inflow scaling

The Blauert driver had been using `u_avg = 6.84e-2 m/s` directly, but Dian's
Matlab preprocessing distinguishes:

- `v0 = 6.84e-2 m/s` as the peak channel speed,
- `u_avg = 0.0455 m/s` as the corresponding average speed.

Because the pycutfem benchmark imposes a parabolic inflow profile, the correct
comparison value is the average velocity. The default was therefore changed to
`u_avg = 4.56e-2 m/s`, giving `u_max = 1.5 u_avg ≈ 6.84e-2 m/s` and a base
wall shear `tau_w ≈ 2.74e-1 Pa`.

### 2. IMEX treatment for the added diffuse traction

The new command-line controls are:

- `--diffuse-shear-time-scheme {constant,imex}`,
- `--diffuse-shear-ramp-time`.

For Benchmark 6 we now use `imex`, meaning:

- the benchmark-local diffuse traction remains lagged in the state variables,
- its amplitude is advanced explicitly with the accepted previous-step time,
- and the amplitude is ramped with the same cosine law used for the inflow.

This is a benchmark-local IMEX continuation for the extra application traction;
it does **not** change the main monolithic one-step-`theta` discretization of
the governing equations.

### Updated onset screening

With the corrected `u_avg` and the IMEX traction ramp, the `poiseuille`
loading branch is no longer flat:

- `scale = 30` remains effectively inert on the coarse active-porosity mesh
  through `t = 0.225 s`,
- `scale = 100` is the first onset case with visible motion:
  at `t = 0.175 s`, the tracked front gives
  `dx_front_y167um = 3.56e-7 m`,
  `dx_front_y250um = 3.33e-7 m`,
  `dx_front_y333um = -9.00e-8 m`.

The conclusion is now sharper:

- the earlier all-or-nothing behavior was partly a forcing inconsistency
  (`u_avg` mismatch plus abrupt extra-load activation),
- the IMEX ramp regularizes the stiff onset regime,
- and the useful calibration ladder starts around `scale ~ 10^2`, not at
  `scale = O(1)`.

### Interpretation

The important result is no longer “the benchmark is stuck because the full model
cannot see the traction correction.” That code-path bug is fixed.

The actual remaining problem is now **calibration/stiffness**, not missing
coupling:

- the `lagged_velocity` tangential proxy is effectively inert for the attached
  Blauert patch on the tested meshes,
- the `poiseuille` tangential proxy is the first viable active-porosity loading
  candidate,
- the corrected Dian velocity scaling and the benchmark-local IMEX load ramp
  restore a continuous onset region in the scale parameter,
- but the onset regime still has to be mapped carefully before a production run
  can be frozen.

### Immediate next calibration step

Benchmark 6 should now proceed with a **serial** onset search on the active-
porosity path:

- use the `poiseuille` tangential model first,
- map the smallest scale that gives a measurable `dx_front_y250um`,
- then reintroduce the physical `rho_f=1000` run only around that promising
  scale / `E` neighborhood,
- and only after that regenerate the paper-ready contour/history figures.

## March 8, 2026: Staged grid search on the active-porosity path

The onset search above was completed with a staged coarse grid on the active-
porosity benchmark wrapper
`examples/biofilms/benchmarks/blauert/paper1_benchmark6_blauert_channel.py`.

### Proxy-grid setup

All calibration probes in this block used:

- `transport-mode pde`,
- conservative CH for `alpha`,
- active porosity transport,
- `u_avg = 4.56e-2 m/s`,
- `diffuse_shear_model = poiseuille`,
- `diffuse_shear_time_scheme = imex`,
- `nx = 24`, `ny = 12`,
- `dt = 0.05`,
- `t_final = 0.5 s`,
- `t_ramp = 2.0 s`,
- physical `rho_f = 1000`,
- score based on the video-comparison wrapper:
  `0.75 * mean_per_y_rmse + 0.25 * y250_rmse`.

### Search path

The coarse search was narrowed in stages rather than running a large blind
Cartesian product.

1. Rate-dependence screen at fixed `E = 200 Pa`

- `eta_s = 1000` and `4000 Pa*s` were immediately too sluggish for the Dian-
  scaled loading.
- `eta_s = 100` and `300 Pa*s` were also clearly slower than `eta_s = 0` on the
  early window.
- Conclusion: the useful calibration branch is `eta_s = 0`.

2. Diffuse-traction scale screen at fixed `E = 200 Pa`, `eta_s = 0`

- `zeta_tau = 300` entered a more aggressive onset regime and started requiring
  repeated accepted-best-iterate steps with growing `vS` corrections.
- `zeta_tau = 50` and `100` stayed in the stable onset band.
- Their partial early-window RMSEs were nearly identical, but `zeta_tau = 50`
  was numerically cleaner, so the stiffness search was continued there.

3. Stiffness refinement at fixed `eta_s = 0`, `zeta_tau = 50`

- `E = 200, 320, 500, 700, 1000 Pa` were screened.
- The `y = 250 um` proxy-window RMSE improved monotonically through this range.
- By the end of the proxy search, the best row was `E = 1000 Pa`.

### Current coarse winner

The completed proxy-window winner is:

- `E = 1000 Pa`,
- `eta_s = 0`,
- `zeta_tau = 50`,
- `nx = 24`, `ny = 12`,
- `dt = 0.05`,
- `t_final = 0.5 s`.

Recorded wrapper summary:

- `global_rmse = 1.406 um`,
- `global_mae = 1.258 um`,
- `mean_per_y_rmse = 2.575 um`,
- `y250_rmse = 0.867 um`,
- `score = 2.148`.

Artifacts:

- `examples/biofilms/results/benchmark6_stageB_E1000_zeta50_20260308/benchmark6_blauert_channel_calibration.csv`
- `examples/biofilms/results/benchmark6_stageB_E1000_zeta50_20260308/benchmark6_blauert_channel_calibration_summary.json`

### Interpretation

This is a real improvement over the earlier inert or runaway regimes:

- the benchmark now has a reproducible coarse-grid calibration winner,
- the active parameter space is no longer diffuse,
- and the winning branch is stable enough to justify promotion to a longer run.

However, Benchmark 6 is **not publication-ready yet** from this coarse search
alone, because:

- the search so far is still only on the proxy mesh/window (`nx = 24`,
  `dt = 0.05`, `t_final = 0.5 s`),
- the global front history still underpredicts early motion (`dx_front_global`
  remains nearly zero through the proxy window),
- and no `2 s` / `4 s` contour or mesh-sensitivity validation has yet been
  frozen on the winning branch.

### Next mandatory step

Promote the current winner
`(E, eta_s, zeta_tau) = (1000 Pa, 0, 50)` to a longer validation run on the
same physics path, then rerun it on the production mesh ladder:

- first `t_final = 2.0 s` on the wrapper,
- then extend to the full comparison window,
- then regenerate history, contour, and mesh-sensitivity figures for the paper.

## March 8, 2026: Observation-based Benchmark 6 workflow

The benchmark has now been refactored away from frame-by-frame OCT RMSE and
toward observation-level validation.

### What changed

- `examples/biofilms/benchmarks/blauert/compare_sim_vs_observations.py`
  now supports:
  - `steady_dian`: steady traced contour from `Basic_t=2_INK.svg`,
  - `dynamic_08pa`: exact Blauert scalar observations for the patchy dynamic
    experiment (`148 um` compression at `2 s`, plateau, `2` percentage-point
    porosity drop),
  - `dynamic_164pa`: exact change-based Blauert observations for the attached
    dynamic experiment (`12 um` thickness drop at `0.4 s`, plateau between
    `0.6-1.3 s`, `27 um` final thickness drop, `220 um` tip elongation,
    `3 deg` angle change, `2` percentage-point porosity drop).
- `examples/biofilms/benchmarks/blauert/blauert_biofilm_deformation_one_domain.py`
  now writes porosity observables directly into `timeseries.csv`:
  - `phi_mean_alpha05`,
  - `phi_drop_alpha05_pp`,
  - `phi_mean_alpha_weighted`,
  - `phi_drop_alpha_weighted_pp`.
- `examples/biofilms/benchmarks/blauert/paper1_benchmark6_blauert_channel.py`
  now calibrates against the observation comparator instead of the video
  comparator.

### Important geometry/physics clarification

The checked-in Dian/Matlab application path is consistent with:

- the traced steady contour in `Basic_t=2_INK.svg`,
- Dian's application setup with `v0 = 6.84e-2 m/s`,
- and the average porosity `phi_b = 0.47`.

It is **not** directly consistent with Blauert's second dynamic
`1.64 Pa / 420 um / 3 deg / 220 um` structure, which is a different attached
geometry. Therefore the wrapper defaults were changed to use the
`steady_dian` observation block as the primary publishable calibration target,
while the dynamic Blauert scenarios remain available as optional checks.

### Current numerical blocker

The observation-based tooling is implemented and regression-tested, but the
Blauert production driver is still too expensive to freeze the benchmark in
this turn.

Tried and confirmed in this branch:

- inertial path (`rho_f = 1000`) stalls very early,
- quasi-static path (`rho_f = 0`) removes the transient-physics objection for
  the steady contour benchmark, but the actual nonlinear solve is still too
  expensive for a same-turn calibration freeze,
- reducing the quadrature degree from `q = 8` to `q = 4` and shrinking the
  pseudo-time horizon (`t_final = 1`) did not yet make the steady smoke runs
  complete cleanly on the current wrapper.

### Practical next step

Benchmark 6 is now blocked by **solver throughput**, not by benchmark design.
The next work should focus on one of the following:

- cheaper Blauert-specific assembly/solve settings for the steady contour case,
- a dedicated quasi-static steady solve path for the Blauert application,
- or a further reduced calibration mesh/active-set strategy that still preserves
  the contour observables.

### Conditioning update

To address the saddle-point stalling seen in the locally refined transient
Blauert runs, the one-domain forms now support an optional consistent
augmented-Lagrangian / grad-div term controlled by `--gamma-div`.

- In the reduced deformation-only builder this acts on the source-free mixture
  constraint `div(C v + B vS)`.
- In the full one-domain builder it is applied to the full constraint residual
  `div(C v + B vS) - alpha*s_v`, so it remains consistent when source terms are
  active.
- The Benchmark 6 wrapper/campaign now uses `--newton-tol 1e-6` again and
  exposes `--gamma-div` as the primary conditioning knob instead of trying to
  drive the solve mainly through a smaller `dt_min`.

### Solver/analysis update

The most recent Benchmark 6 failure was not treated as a "make `dt_min` even
smaller" problem. Reading the paper analysis against the current Blauert
defaults pointed to two missing ingredients instead:

- the run was still using the PETSc SNES path, while the benchmark now needs a
  smoother internal nonlinear path for diagnosis and bounded-variable control;
- the old Blauert defaults used `gamma_u=1e-6` with `u_extension=grad`, which is
  much weaker than the whole-domain coercivity assumption used in the one-step
  analysis.

Accordingly, the Blauert driver/wrapper were switched to:

- `--nonlinear-solver pdas`,
- `--ls-mode dealii`,
- `--alpha-box-constraints` and `--phi-box-constraints`,
- stronger application-style extension defaults:
  `--gamma-u 5.0`, `--u-extension l2`, `--gamma-u-pin 1e-4`.

The working hypothesis is now:

- `gamma_div` improves the mixed `(v,p,vS)` conditioning,
- PDAS removes the SNES-specific globalization failure mode,
- and the stronger `j_u`/mirrored `vS` extension is what brings the benchmark
  closer to the coercivity regime assumed by the analysis.

### Failure diagnostics / restart update

The steady refined calibration now reaches the PDAS path and fails later, around
step 13 for the `(E, eta_s, zeta) = (120, 0, 0)` baseline candidate. That is a
useful failure because it is no longer a generic solver crash: the residuals are
already down at the `1e-6` level and the failure is the semismooth line search,
not a loose tolerance or an SNES-specific breakdown.

To diagnose this more rigorously, the Blauert driver now prints:

- `[fail_res]` per-field residual maxima,
- `[fail_blocks]` per-equation-block residual maxima assembled from the stored
  residual forms (`momentum`, `mass`, `skeleton`, `kinematics`, `phi`, `alpha`,
  `mu_alpha`, ...).

The wrapper/campaign were also extended so the calibration sweep can keep going
after a failed candidate:

- failed candidates are recorded in
  `benchmark6_blauert_channel_failed_cases.json`,
- the sweep no longer aborts on the first bad `(E, eta_s, zeta)` point unless
  `--no-continue-on-candidate-failure` is requested,
- restart controls are forwarded through the campaign
  (`--restart-from`, `--restart-dt`, `--restart-write-every`).

This means the next rerun should tell us explicitly whether the late-step stall
is dominated by the mixed momentum block, the skeleton block, or another
residual block, while still letting the rest of the calibration grid finish.

### Momentum-block diagnosis for the steady Dian path

Reading the new `[fail_blocks]` output against the solver trace shows that the
steady Dian calibration is **not** failing because of advection-dominated fluid
transport:

- the failing run uses `rho_f = 0`, so the one-domain fluid momentum block is a
  Stokes/Brinkman-type block, not a convective Navier--Stokes block;
- the only built-in fluid SUPG term is guarded by `rho_f != 0`, so `--v-supg`
  is inactive in this case by construction;
- the available CIP terms act on `alpha` and `phi`, not on the fluid momentum
  equation;
- `alpha` transport is already run with `--alpha-supg 0.5`, yet the raw block
  residual is still dominated by `momentum` by several orders of magnitude.

At the failed iterations the raw reduced residual `R` is already small, while
the semismooth residual `G` and the active set continue to jump:

- `|R|_∞` reaches the `1e-6` to `1e-8` range,
- but `|G|_∞` remains much larger until the active set settles,
- and `ΔA` keeps changing by O(10^2) DOFs between PDAS iterations.

So the current bottleneck is best interpreted as a **conditioning /
complementarity** problem in the locally refined mixed Stokes--Brinkman block,
amplified by the sharp coefficient changes carried by `(alpha, phi)`, not as a
missing streamline-upwind stabilization.

The physically relevant scales support that interpretation. With
`mu_f = 1e-3`, `phi_b = 0.47`, and `kappa_inv = 9.81e11`, the Brinkman drag
scale in the biofilm is

`beta ~ mu_f * phi_b^2 * kappa_inv ~ 2.17e8`,

while the cellwise viscous scale on the refined `nx=16` mesh is only on the
order of `mu_f / h^2 ~ 3.4e4`. The mixed block is therefore drag-dominated by
several orders of magnitude.

Conclusion for this branch:

- `v`-SUPG is not the right fix for the steady Dian calibration,
- `alpha/phi` CIP may smooth the transported interface slightly, but it is not
  the primary remedy because those residual blocks are already tiny,
- the right next knobs remain `gamma_div`, extension/ghost-penalty strength,
  and local interface resolution.

### Fixed-dt steady profile

A direct steady-Dian probe with the existing conditioning defaults but a fixed
small step

- `dt = 5e-3`,
- `dt_min = 5e-3`,
- `gamma_div = 1e-2`,
- `gamma_u = 5`,
- `u_extension = l2`,
- local refinement band `2.5e-4` with one expanded layer,

stayed on the fixed step and settled into roughly `2--3` PDAS iterations per
accepted step in the early transient window. Based on that probe, the
observation campaign's steady block was changed from `dt = 2.5e-2` to
`dt = 5e-3` so the steady calibration starts from the stable branch instead of
relying on repeated time-step cuts from the first few steps.

### March 9, 2026: semismooth line-search isolation on the steady branch

To separate model-conditioning issues from "just try a smaller `dt` again", the
failing steady candidate

- `E = 200`,
- `eta_s = 0`,
- `zeta_tau = 0`,
- refined `nx = 16`,
- `dt = dt_min = 5e-3`,

was restarted directly from the last accepted checkpoint before failure
(`checkpoint_step=00038`, `t = 0.19 s`) and rerun with one stabilization knob
changed at a time.

Observed outcomes on that same restart state:

- baseline benchmark settings
  (`gamma_div = 1e-2`, `gamma_u = 5`, `u_extension = l2`) fail again at
  `step 39` with
  `RuntimeError: Line search failed: no semismooth residual decrease`;
- stronger whole-domain extension alone
  (`gamma_div = 1e-2`, `gamma_u = 20`) delays the failure only to `step 43`;
- stronger mixed-block conditioning succeeds:
  `gamma_div = 5e-2` reaches `t = 0.25 s` cleanly from the same checkpoint,
  and `gamma_div = 1e-1` also reaches `t = 0.25 s` cleanly.

The residual signature stays the same in the failed branches:

- the PDAS iterate stalls with `|G|_inf ~ 1e-6`,
- the raw per-field residuals are already small (`v`, `vS` around `1e-6`,
  `alpha/phi/mu_alpha` much smaller),
- but the assembled block breakdown is still dominated by `momentum`
  (`~ 1.37e-3 -- 1.39e-3` in the repeated failing runs).

That experiment is the clearest evidence so far that the steady Benchmark 6
stall is **not** a phase-box issue and **not** primarily a `u`-extension issue.
It is a mixed Stokes/Brinkman/constraint conditioning problem:
the semismooth line search is being asked to globalize a step in a regime where
the coupled `(v,p,vS)` block is still much stiffer than the phase and
kinematics blocks.

Practical conclusion for the publishable branch:

- keep the stronger application-style extension (`gamma_u = 5`, `u_extension = l2`);
- but treat a larger consistent mixed-block penalty
  (`gamma_div ~ 5e-2`, with `1e-1` as a conservative upper diagnostic value)
  as the **primary** stabilization needed for Benchmark 6.

### Mesh-only effect (preliminary)

A mesh-only rerun of the same baseline physics path on refined `nx = 24`
(`gamma_div = 1e-2`, `gamma_u = 5`) is more benign in the early transient
window than `nx = 16`: it stays on the fixed step and advances through the
startup with `2--5` PDAS iterations per time step over the first `0.115 s`.

That is consistent with the scale argument already noted above:
refining the mesh increases the local viscous scale `mu/h^2`, so the
drag-to-viscous ratio becomes less extreme.
But the gap is still several orders of magnitude, so refinement alone should be
treated as a **secondary** aid to the steady Dian branch, not as a replacement
for the mixed-block conditioning fix.

### March 9, 2026: `L2` vs `H1`-style `u` regularization

The remaining modeling question was whether Benchmark 6 should use the current
`L2` extension penalty or switch to the `grad` option as a more "smoother"
`H1` regularization. In the implementation, that distinction is important:

- `--u-extension l2` adds a mass-style penalty
  `gamma_u h^{-2} (1-alpha) (u, xi)`;
- `--u-extension grad` adds only the gradient seminorm
  `gamma_u (1-alpha) (grad u, grad xi)`,
  with the small `gamma_u_pin` term used only to remove the rigid-translation
  nullspace of that seminorm.

So the `grad` option is not a full `H1` norm; it is an `H1`-seminorm plus a
very small pinning mass term.

To compare them fairly, both modes were run from the same hard restart state:

- checkpoint `step 38`, `t = 0.19 s`,
- refined steady Dian calibration case
  `E = 200`, `eta_s = 0`, `zeta_tau = 0`, `nx = 16`,
- fixed `dt = dt_min = 5e-3`,
- stabilized mixed block `gamma_div = 5e-2`,
- same `gamma_u = 5`, same mesh, same bounds, same PDAS solver.

Results over the restart interval `t = 0.19 -> 0.25 s`:

- `u_extension = l2` reaches `t = 0.25 s` in **17** total Newton iterations
  over 12 accepted steps, with
  `{1: 9, 2: 2, 4: 1}` iterations per step,
  cumulative line-search time about `4.06 s`,
  and wall time `91.13 s`;
- `u_extension = grad` also reaches `t = 0.25 s`, but needs **36** total
  Newton iterations,
  `{1: 2, 2: 2, 3: 4, 4: 2, 5: 2}` iterations per step,
  cumulative line-search time about `17.91 s`,
  and wall time `245.90 s`.

So, on the same stabilized branch, the `grad`/`H1`-style regularization is
about:

- `2.1x` more Newton iterations,
- `4.4x` more line-search work,
- `2.7x` more wall-clock time.

The tail behavior is especially different.

- With `l2`, once the active set settles after `t ~ 0.21 s`, the solve drops to
  mostly one Newton step per time level and stays there through `t = 0.25 s`.
- With `grad`, the same window still shows repeated semismooth bursts:
  `5` Newton steps at `t = 0.225 s`,
  `5` again at `t = 0.235 s`,
  `4` at `t = 0.245 s`,
  and `3` at `t = 0.25 s`.

That behavior is consistent with the structure of the two penalties.
The `l2` extension gives direct zero-order coercivity outside the biofilm,
scaled with `h^{-2}`, so rigid-body-like modes in the weakly constrained
free-fluid region are damped immediately.
The `grad` extension only penalizes spatial variation; nearly rigid
translations remain cheap and are controlled only through the tiny pinning
term. In this PDAS active-set regime, that makes the semismooth line search
noticeably choppier even though the run remains formally convergent.

On the recorded observables, the difference is modest for the weighted
quantities and front positions, but it is not zero:

- final weighted mean volume fraction differs by only `6.95e-05`
  (`0.0135%`);
- final weighted `phi` drop differs by `6.95e-03`;
- final front positions are larger under `grad` by about
  `2.5 -- 9.0 um`
  (`0.24 -- 0.83%`);
- the derived displacement-at-height values differ by about
  `15 -- 52%` because they are small residual differences between two close
  fronts.

Recommendation for Benchmark 6:

- keep `--u-extension l2` as the publishable default;
- do **not** switch the steady Dian branch to `grad`/`H1` regularization;
- if extra robustness is needed, spend stabilization budget on the mixed-block
  conditioning (`gamma_div`) and on local refinement, not on replacing the
  `l2` extension with the `grad` variant.

So the answer to "L2 or H1 here?" is: **use `L2` for Benchmark 6**.
The `grad`/`H1`-style option is admissible, but in this simulation it is
strictly less attractive: slower, choppier for the semismooth line search, and
not more faithful in any observable that matters for the paper.

### March 9, 2026: campaign `run.log` audit and publication plan

The next question is not only "what stabilizes one representative restart?", but
"what part of the current Benchmark 6 campaign is actually publication-ready?"
To answer that, the calibration-stage `run.log` files under
`examples/biofilms/results/benchmark6_observation_campaign` were audited case by
case.

Two facts are immediately relevant for paper quality.

First, the current artifact tree is **not clean**.
At least one case in `steady_calibration_refined` is still a stale morning run
with the old settings (`gamma_div = 1e-2`, `vtk_every = 0`), while later cases
in the same directory were rerun with the new settings
(`gamma_div = 5e-2`, `vtk_every = 1`).
So the present mix of logs, summaries, and calibration rows should not be used
directly in a paper figure or table.

Second, once stale and incomplete artifacts are separated from the genuinely
completed ones, the stability map is very structured:

- on the clean `steady_mesh_refined` calibration-stage runs with
  `gamma_div = 5e-2`, `u_extension = l2`, `gamma_u = 5`,
  the completed `zeta = 0` cases show
  `E = 200` and `E = 500` passing to `t = 1.0 s`,
  while `E = 120` and `E = 320` still fail;
- every completed nonzero diffuse-shear case
  (`zeta = 30` or `50`) still fails by line-search breakdown;
- one case (`E = 500`, `zeta = 50`) is merely incomplete, so it provides no
  evidence either way.

Across those failures, the residual signature is again nearly invariant:

- the dominant failed block remains `momentum`,
  typically `1.39e-3 -- 1.45e-3`;
- the phase and bound-associated blocks stay tiny
  (`alpha`, `phi`, `mu_alpha` around `1e-8 -- 1e-10`);
- therefore the remaining campaign failures are still a mixed
  mechanics/conditioning problem, not a phase-box pathology.

To test whether the nonzero-`zeta` branch can be rescued by simply pushing the
same stabilization knob harder, a representative difficult case

- `E = 200`,
- `zeta = 30`,
- refined `nx = 16`,
- restart from `checkpoint_step = 45` (`t = 0.225 s`),

was rerun with `gamma_div = 1e-1` while keeping
`dt = dt_min = 5e-3`, `u_extension = l2`, and the rest of the setup unchanged.
That branch still fails immediately at `step 46`, with the same momentum-block
signature. So, for the current model, **raising `gamma_div` from `5e-2` to
`1e-1` is not enough to make the nonzero-`zeta` steady calibration publishably
robust**.

#### Recommended parameter set for the publishable branch

Based on the evidence above, the conservative publication choice is:

- use `u_extension = l2`;
- use `gamma_u = 5`;
- use `gamma_div = 5e-2` as the default reported stabilization;
- keep `dt = dt_min = 5e-3`, `theta = 1`, `q = 4`;
- keep the local geometric refinement around the biofilm
  (`refine_band = 2.5e-4`, `refine_expand_layers = 1`);
- for the nominal reported steady Dian case, use `zeta = 0`.

For the material parameter, `E = 200` is the safest nominal choice:

- it is in the middle of the calibration range rather than at an endpoint,
- it passes the clean `zeta = 0` refined steady run,
- it is the case already used for the detailed restart diagnostics,
- and it avoids depending on the currently mixed/stale `E = 120` artifact set.

`E = 500` can be kept as a robustness cross-check, but not as the primary
published choice unless it is needed for the final fit.

#### What should not be claimed yet

The current data do **not** support claiming that the steady Benchmark 6
calibration is robust for arbitrary `zeta` in `{0,30,50}`.

More specifically:

- the clean evidence supports the `zeta = 0` branch;
- the `zeta = 30` and `50` branches are still unstable in the current solver
  path;
- and `gamma_div = 1e-1` alone does not rescue a representative `zeta = 30`
  case.

So unless a separate stabilization campaign is completed, nonzero-`zeta` steady
cases should be treated as exploratory and should not appear in the main paper
as established benchmark results.

#### Publication plan

To make Benchmark 6 acceptable for review, the next steps should be:

1. **Clean the artifact tree and rerun from scratch.**
   Archive or delete the mixed `steady_calibration_refined` directory and rerun
   into a fresh output directory with `--skip-existing` disabled for the first
   publication pass.

2. **Freeze the main-paper scope to the stable subspace.**
   For the steady Dian benchmark, restrict the main results to
   `zeta = 0`, `u_extension = l2`, `gamma_u = 5`, `gamma_div = 5e-2`,
   with `E = 200` as the nominal reported parameter set.

3. **Complete a clean steady mesh ladder on that frozen branch.**
   Run `nx = 16, 24, 32` only for the frozen `zeta = 0` branch and document:
   - pass/fail,
   - Newton iterations,
   - front/displacement observables,
   - and contour convergence.
   That is the mesh evidence reviewers will expect.

4. **Separate robustness from calibration.**
   After the nominal branch is reproduced cleanly, run only a small robustness
   set:
   - `E = 500`, `zeta = 0` as a parameter sensitivity check,
   - possibly one `gamma_div = 1e-1` repeat to show the solution is not an
     artifact of a knife-edge stabilization value.

5. **Move nonzero-`zeta` work onto a dedicated continuation track.**
   If the paper still needs `zeta > 0`, do not brute-force the full grid.
   Instead:
   - converge the `zeta = 0` branch first,
   - restart from that state,
   - continue `zeta` in small increments (`0 -> 5 -> 10 -> 20 -> 30 -> ...`),
   - and, if needed, combine that with mesh refinement and a longer traction
     ramp.
   Without that continuation strategy, the current runs suggest the branch is
   not solver-robust enough for publication.

6. **Include a reviewer-facing stability discussion in the paper.**
   The text should say explicitly that:
   - bounds on `alpha` and `phi` were not the source of failure,
   - the failing residuals were dominated by the mixed momentum block,
   - `L2` outperformed the `H1`-style `grad` regularization,
   - `gamma_div` was the primary stabilizing ingredient,
   - and refinement helped but did not replace mixed-block conditioning.

7. **Use VTK only as diagnostic evidence, not as the full production output.**
   Keep `--vtk-every 1` only for short diagnostic runs.
   For the publication rerun, use a sparse cadence (for example every `10`
   accepted steps on the final branch) and archive only the frames that support
   figures or qualitative discussion.

#### Bottom line

The publishable Benchmark 6 story is available now, but only if it is scoped
correctly:

- stable, reproducible steady Dian benchmark;
- `zeta = 0`;
- `E = 200`;
- `u_extension = l2`;
- `gamma_div = 5e-2`;
- clean rerun from fresh outputs;
- mesh ladder and stability discussion included.

That is a defensible JCP-level package.
Trying to present the current nonzero-`zeta` steady cases as already settled
would invite exactly the reviewer objection we want to avoid.

## March 9, 2026: analytical scaling of the benchmark-local `zeta` traction

Before spending more time on transient parameter sweeps, it is useful to extend
the paper's one-step analysis to the benchmark-local diffuse traction term.

### 1. What `zeta` changes in the variational problem

In the current Blauert driver, the added load enters the momentum and skeleton
equations as a lagged equal-and-opposite pair of body forces:

- fluid momentum receives `- w_Gamma^n g_t^n`,
- skeleton momentum receives `+ w_Gamma^n g_t^n`,

with

- `w_Gamma^n = sqrt(|grad alpha^n|^2 + eta)`,
- `g_t^n = zeta * tau^n * t_Gamma^n` for the `poiseuille` model,
- `|t_Gamma^n| = 1`.

So the extra linear functional at one backward-Euler step is

```text
ell_zeta^n(w,z)
  = zeta * int_Omega tau^n w_Gamma^n t_Gamma^n · (z - w) dx.
```

Crucially, this is a **lagged right-hand side term**, not a new bilinear part of
the linearized step. Therefore:

- it does **not** change the mechanics coercivity condition `delta_v^n > 0`,
- it does **not** change the kinematic coercivity condition `delta_u^n > 0`,
- and it does **not** change the fixed-point smallness condition `L^n < 1`
  directly.

What it changes is the **size of the forcing**, which then propagates into
`v^{S,n+1}`, `u^{n+1}`, and the next-step advection constants used in the paper
analysis.

### 2. A clean bound for the added load

By Cauchy-Schwarz and Poincare,

```text
|ell_zeta^n(w,z)|
<= |zeta| * ||tau^n||_Linf * ||w_Gamma^n||_L2 * (||w||_L2 + ||z||_L2)
<= C_P * |zeta| * ||tau^n||_Linf * ||w_Gamma^n||_L2
   * (||w||_H1 + ||z||_H1).
```

Hence the dual norm of the added load is controlled by

```text
||ell_zeta^n||_(V x V)'
<= C_P * |zeta| * ||tau^n||_Linf * ||w_Gamma^n||_L2.
```

The diffuse-interface weight can be tied back to the Cahn-Hilliard energy.
Since

```text
||w_Gamma^n||_L2^2
 = int_Omega (|grad alpha^n|^2 + eta) dx
<= ||grad alpha^n||_L2^2 + eta |Omega|,
```

and the CH energy contains

```text
E_alpha^n
 = (gamma_alpha * eps_alpha / 2) ||grad alpha^n||_L2^2
   + (gamma_alpha / eps_alpha) int_Omega W(alpha^n) dx,
```

we have

```text
||grad alpha^n||_L2^2 <= 2 E_alpha^n / (gamma_alpha * eps_alpha).
```

Therefore

```text
||ell_zeta^n||_(V x V)'
<= C_P * |zeta| * ||tau^n||_Linf
   * sqrt( 2 E_alpha^n / (gamma_alpha * eps_alpha) + eta |Omega| ).
```

Define the dimensionless traction-amplitude factor

```text
Xi_zeta^n
:=
|zeta| * ||tau^n||_Linf
* sqrt( 2 E_alpha^n / (gamma_alpha * eps_alpha) + eta |Omega| ).
```

Then the one-step mechanics estimate becomes

```text
||v^{n+1}||_H1 + ||vS^{n+1}||_H1 + ||p^{n+1}||_L2
<= C_mech^n * (F_base^n + Xi_zeta^n),
```

where `F_base^n` collects the pre-existing loads already present in the paper
analysis.

### 3. What this implies for stable parameter scaling

The key scaling is now explicit:

```text
Xi_zeta^n ~ |zeta| / sqrt(gamma_alpha * eps_alpha)
```

up to the bounded factors `||tau^n||_Linf` and `E_alpha^n`.

So if everything else is fixed, increasing `zeta` makes the added forcing grow
like `|zeta| / sqrt(eps_alpha)`. Equivalently, to keep the same one-step forcing
level, one should keep

```text
zeta^2 / (gamma_alpha * eps_alpha)
```

bounded.

This is the first analytically meaningful rule for the transient campaign:

- if `zeta` is increased by a factor `r`, then the pair `gamma_alpha *
  eps_alpha` should ideally increase by a factor `r^2`,
- or, if `gamma_alpha` is kept fixed, then `eps_alpha` should increase like
  `zeta^2`.

This does **not** say that arbitrarily large `eps_alpha` is a good idea.
It only says that the diffuse traction becomes more concentrated, and therefore
 more aggressive, when `eps_alpha` is made smaller.

### 4. How the added forcing feeds the paper's stability conditions

The mechanics theorem itself is unchanged, but the next-step constants are not.

First, the kinematic coercivity condition in the paper depends on

```text
delta_u^n = c_u - ((D_u^n / 2) - 1 / dt)_+ * C_P^2,
```

with `D_u^n = ||div vS^n||_Linf`.
At the discrete level, a standard inverse/embedding estimate gives

```text
D_u^n <= C_div(h) * ||vS^n||_H1
       <= C_div(h) * C_mech^n * (F_base^n + Xi_zeta^n).
```

Hence a sufficient condition for keeping the kinematic step in the stable regime
is

```text
1 / dt + c_u / C_P^2
>
(1/2) * C_div(h) * C_mech^n * (F_base^n + Xi_zeta^n).
```

Second, in the bulk CH regime the paper proves the one-step estimate

```text
dt < M_alpha * gamma_alpha * eps_alpha
     / (C_P^2 ||vS^n||_Linf^2).
```

Again using a discrete embedding,

```text
||vS^n||_Linf <= C_inf(h) * C_mech^n * (F_base^n + Xi_zeta^n),
```

so a sufficient practical condition becomes

```text
dt
<
M_alpha * gamma_alpha * eps_alpha
/
(
  C_P^2 * C_inf(h)^2 * (C_mech^n)^2 * (F_base^n + Xi_zeta^n)^2
).
```

This shows exactly which parameters can compensate a larger `zeta`:

- increase `gamma_alpha * eps_alpha`,
- increase the extension coercivity constants `c_u` and `c_S`,
- reduce `dt`,
- and keep the discrete interface resolved so that `C_div(h)` and `C_inf(h)`
  do not become the dominant source of instability/noise.

### 5. Consequences for the actual Blauert controls

For the current code path, these abstract constants translate as follows.

1. `gamma_u` is analytically relevant.

   In the Blauert implementation, `gamma_u` controls the displacement extension
   and, by default, the mirrored skeleton-velocity extension as well.
   So increasing `gamma_u` increases both coercivity constants that appear in
   the paper analysis:

   - `c_u` for the kinematic update,
   - `c_S` for the skeleton part of the mechanics block.

   This is therefore a genuine analytical counterweight to larger `zeta`.

2. `gamma_div` is mainly algorithmic.

   The added `gamma_div` term improves conditioning of the mixed solve and is
   very useful numerically, but in the Brezzi coercivity proof it vanishes on
   `ker(b^n)`. So it does **not** play the same role as `c_u`, `c_S`, or
   `gamma_alpha * eps_alpha` in the analytical stability inequalities above.

3. `eps` / `alpha_ch_eps` is the first parameter that should be paired with
   `zeta`.

   The diffuse traction is localized with `|grad alpha|`, so decreasing the
   interface thickness while keeping `zeta` fixed makes the forcing more
   concentrated. Analytically, the dangerous combination is `zeta^2 /
   (gamma_alpha * eps_alpha)`.

4. `q` is not part of the PDE stability theory.

   The diffuse traction contains non-polynomial factors (`sqrt(|grad alpha|^2 +
   eta)` and normalized tangents), so no finite quadrature degree is exact.
   Quadrature choice is therefore a consistency/implementation issue, not part
   of the one-step well-posedness analysis.

5. Scalar `zeta` is the mathematically natural choice.

   The added traction is defined in the interface tangent/normal frame, not in
   the Cartesian frame. A separate `x`/`y` scale would be ad hoc and would not
   align with the variational structure above. If a two-parameter extension is
   ever needed, it should be "tangential vs normal", not "x vs y".

### 6. Model-dependent upper bounds for `tau^n`

The previous bounds use `||tau^n||_Linf`. For the currently available load
closures:

- `poiseuille`:

  ```text
  ||tau^n||_Linf <= 4 * mu_f * u_max / H,
  ```

  which is explicit and independent of the unknown fields.

- `lagged_velocity`:

  ```text
  ||tau^n||_Linf <= 2 * mu_f * ||grad v^n||_Linf.
  ```

- `lagged_stress`:

  ```text
  ||tau^n||_Linf <= ||p^n||_Linf + 2 * mu_f * ||grad v^n||_Linf.
  ```

So the cleanest reviewer-facing analysis is for the `poiseuille` model. The
`lagged_stress` branch can still be bounded, but only under stronger
`Linf` assumptions on the frozen pressure and velocity gradient.

### 7. Practical rule set for the transient study

The analysis suggests the following order of operations for the transient
Blauert campaign.

1. Do **not** tune `zeta` in isolation.
   Tune the combination

   ```text
   zeta^2 / (gamma_alpha * eps_alpha)
   ```

   instead.

2. When increasing `zeta`, first increase `eps_alpha` (or `gamma_alpha`) enough
   to keep the above ratio in the same ballpark.

3. If the step then becomes kinematically fragile, increase `gamma_u`, because
   that is the parameter that actually enlarges the coercivity constants in the
   one-step proof.

4. Use `gamma_div` to help the nonlinear algebra, but do not interpret it as the
   main analytical stabilizer for `zeta`.

5. Keep the mesh fine enough that the diffuse layer is resolved.
   The paper's existing interface-resolution discussion still applies: one wants
   several cells across the diffuse layer, i.e. practically `h <= O(eps_alpha)`,
   rather than `h >> eps_alpha`.

#### Bottom line

The mathematically meaningful "zeta stability parameter" is not `zeta` alone.
For the present benchmark it is

```text
zeta^2 / (gamma_alpha * eps_alpha),
```

with secondary control coming from the extension coercivity (`gamma_u`) and only
algorithmic help from `gamma_div`.

That is the right quantity to organize the next transient screening around.

## 2026-03-12 short-validation follow-up

### Completed `gamma_u` comparison on the corrected benchmark setup

Common setup for the comparison:

- corrected `t_ramp = 2.0 s`,
- corrected contour-based global front observable,
- `E = 200 Pa`, `solid_visco_eta = 0`, `zeta = 30`,
- `rho_f = 0`, `u_avg = 0.0456 m/s`,
- direct LU (`preonly + lu + mumps`),
- `dt = 0.005`, `dt_min = 0.00125`,
- `global_front_quantile = 0.005`, `dx_quantile = 0.05`.

Short validation run for the new candidate:

- output: `/tmp/b6_driver_tramp2_gU10_t1`,
- restart source: `/tmp/b6_driver_tramp2_gfix_t025/restart/checkpoint_latest.npz`,
- changed parameter: `gamma_u = 10` (previous comparison branch used `gamma_u = 5` in `/tmp/b6_driver_tramp2_gfix_t2`).

### What changed scientifically

- Solver health is no longer the limiting factor:
  - the `gamma_u = 10` branch reached `t = 1.0 s` with every step accepted,
  - `dt` stayed at `0.005`,
  - each step converged in `2` Newton iterations,
  - no `dt` reduction, line-search failure, trust-region stagnation, or NaNs occurred.

- Comparison against the extracted video over `t ∈ [0.267, 0.968] s`:

  - `gamma_u = 10`:
    - global `RMSE = 6.15 um`,
    - `y = 150 um`: `RMSE = 20.3 um`,
    - `y = 250 um`: `RMSE = 12.1 um`,
    - `y = 350 um`: `RMSE = 9.77 um`.

  - `gamma_u = 5`:
    - global `RMSE = 5.05 um`,
    - `y = 150 um`: `RMSE = 18.7 um`,
    - `y = 250 um`: `RMSE = 25.4 um`,
    - `y = 350 um`: `RMSE = 7.88 um`.

- Direct displacement comparison at `t = 1.0 s`:
  - `gamma_u = 10`: `dx_front_global ≈ 12.25 um`, `dx_front_y250 ≈ 41.0 um`,
  - `gamma_u = 5`: `dx_front_global ≈ 17.17 um`, `dx_front_y250 ≈ 97.6 um`.

### Decision for the first corrected long run

- `gamma_u = 10` is the best-working parameter set so far.
  - It materially reduces the dominant mid-height overshoot that made the `gamma_u = 5`
    trajectory scientifically questionable.
  - Its aggregate front-history fit is better over the full `0.27-0.97 s` window
    (`mean RMSE ≈ 12.1 um` across global/150/250/350) than `gamma_u = 5`
    (`mean RMSE ≈ 14.3 um`).

- Remaining risk:
  - the global and `y = 150 um` histories are still not ideal, so the next step must be a
    monitored long production run rather than declaring the calibration complete.

- Planned next run:
  - wrapper production run with the same corrected physics and `gamma_u = 10`,
  - explicit `--t-ramp 2.0`,
  - monitored at the first `30 min` checkpoint and then hourly if healthy.

### Completed follow-up: `gamma_u = 20` at the corrected `zeta = 30` forcing

A stronger whole-domain extension was tested next to see whether the remaining
late-time defect was still just the overly peaked `y = 250 um` front.

Common setup relative to the `gamma_u = 10` branch:

- same corrected paper forcing:
  - `u_avg = 4.56e-2 m/s`,
  - `t_ramp = 2.0 s`,
  - `zeta = 30`,
- same corrected observables:
  - contour-based global front with `global_front_quantile = 0.005`,
  - per-height contour fronts with `dx_quantile = 0.05`,
- same solver path:
  - semismooth LM enabled on nonempty active sets,
  - direct LU,
  - `gamma_div = 0.1`,
  - `rho_f = 0`,
  - `dt = 0.005`, `dt_min = 0.00125`.

Changed parameter:

- `gamma_u = 20` (output: `/tmp/b6_driver_tramp2_gU20_t1p5`).

What happened:

- Solver health stayed excellent through `t = 1.5 s`:
  - no dt reduction,
  - `nNewton = 2`,
  - no line-search failure,
  - no LM/trust-region stagnation,
  - no NaNs.
- But the scientific fit became much worse by late times.

Measured comparison against the extracted video over `t ∈ [0.267, 1.47] s`:

- global `RMSE = 37.1 um`,
- `y = 150 um`: `RMSE = 40.7 um`,
- `y = 250 um`: `RMSE = 27.2 um`,
- `y = 350 um`: `RMSE = 35.6 um`.

Late-time displacement check at `t ≈ 1.50 s`:

- simulation:
  - global `≈ 20.3 um`,
  - `y167 ≈ 20.3 um`,
  - `y250 ≈ 61.5 um`,
  - `y333 ≈ 25.7 um`;
- extracted video:
  - global `≈ 88.1 um`,
  - `y150 ≈ 95.9 um`,
  - `y250 ≈ 86.5 um`,
  - `y350 ≈ 88.1 um`.

Practical conclusion from this branch:

- increasing `gamma_u` further is **not** the right next lever.
- It does flatten the centerline peak, but by `t ≈ 1.5 s` it underpredicts the
  whole benchmark and cannot reach a publishable history fit.
- The next corrected-physics validation should therefore move to a branch that
  has evidence for a flatter per-height response at the parameter level, not
  just through stronger extension.

### Evidence-based next parameter move

The most useful archived qualitative clue is the earlier dynamic branch with
`E = 500`, `zeta = 50` on the old fast-ramp configuration:

- `examples/biofilms/results/benchmark6_target_dynamic08_20260311/calibration/E500_eta0_zeta50_nx24/timeseries.csv`

That run is **not** quantitatively comparable, because it used the old
over-aggressive ramp and the old global observable, but it is still useful as a
shape indicator: by its final reported time (`t ≈ 0.120 s`) the per-height
fronts were already nearly uniform (`y167 ≈ 32.6 um`, `y250 ≈ 26.2 um`,
`y333 ≈ 30.2 um`), unlike the strongly center-peaked corrected `E = 200`,
`zeta = 30` branch.

So the next short validation should test the corrected paper physics on the
first branch with prior support for a flatter front:

- `E = 500 Pa`,
- `zeta = 50`,
- keep the corrected LM path,
- keep `gamma_u = 10` as the best current extension setting,
- run only a short corrected probe first (`t_final = 0.5 s`) before any longer
  continuation.

## 2026-03-12 continuation update

### Completed continuation: corrected `E = 500`, `zeta = 50`, `gamma_u = 10` is solver-healthy but still not publishable

Continuation run:

- output root: `/tmp/b6_driver_E500_z50_gU10_t1p5`,
- resumed to completion from the last written checkpoint in that directory,
- same corrected setup otherwise:
  - `transport-mode pde`,
  - `poiseuille` diffuse traction,
  - `t_ramp = 2.0 s`,
  - `rho_f = 0`,
  - `dt = 0.005`,
  - direct LU,
  - `gamma_u = 10`.

Observed solver behavior:

- no LM/trust-region stagnation,
- no line-search failure,
- no KSP failure,
- no NaNs,
- fixed `dt = 0.005`,
- accepted the full run to `t = 1.5 s`.

So the LM stall issue is fixed on this branch too. The blocker is scientific fit only.

Late-window fit from the completed continuation:

- comparison against the extracted video over `t ∈ [0.501, 1.47] s`:
  - global `RMSE = 35.0 um`,
  - `y = 150 um`: `RMSE = 39.0 um`,
  - `y = 250 um`: `RMSE = 14.2 um`,
  - `y = 350 um`: `RMSE = 32.1 um`;
- direct front values:
  - at `t = 1.0 s`:
    - global `≈ 14.6 um`,
    - `y167 ≈ 14.9 um`,
    - `y250 ≈ 42.9 um`,
    - `y333 ≈ 18.8 um`;
  - at `t = 1.5 s`:
    - global `≈ 36.4 um`,
    - `y167 ≈ 36.1 um`,
    - `y250 ≈ 112.2 um`,
    - `y333 ≈ 45.4 um`.

Against the extracted video at `t ≈ 1.5 s`:

- global `≈ 88.1 um`,
- `y150 ≈ 95.9 um`,
- `y250 ≈ 86.5 um`,
- `y350 ≈ 88.1 um`.

Interpretation:

- the branch keeps its excellent early-window fit,
- but by late time it falls back into the same structural defect:
  - the centerline front overshoots,
  - while the global / `y150` / `y350` histories remain far too small.

So this branch is not publishable yet even though the solver is now healthy.

### Additional root-cause eliminations completed

1. Initial contour source is **not** the problem.

- Re-ran the extractor with the recommended Matlab source:

  ```bash
  conda run --no-capture-output -n fenicsx python \
    examples/biofilms/benchmarks/blauert/extract_front_displacement_from_video.py \
    --polygon-source matlab_preprocessing --matlab-shift-um 0 --t-max 0 \
    --out-csv /tmp/b6_matlab_extract.csv \
    --out-polygon /tmp/b6_matlab_polygon_mm.csv
  ```

- The generated `/tmp/b6_matlab_polygon_mm.csv` matches
  `examples/biofilms/benchmarks/blauert/exp_frame0_polygon_mm.csv`.
- So the current default `alpha0` file is already the recommended Matlab-derived contour.

2. The IMEX traction time level is wired correctly.

- In `blauert_biofilm_deformation_one_domain.py`, `diffuse_scale_update(t_now)` is called in `post_step_refiner(...)`.
- In `nonlinear_solver.py`, `post_step_refiner(...)` runs only after an accepted step with
  `solver._current_t = t_n` and `solver._current_dt = dt`.
- Therefore the updated diffuse-traction amplitude is used on the next solve at the accepted previous time level, matching the benchmark statement.

So the late-time mismatch is not explained by either:

- a wrong initial polygon source, or
- a wrong IMEX time-level update.

### Evidence-based next move

The next short validation should reintroduce the physical fluid inertia around the now-promising high-`E`, `zeta = 50` branch.

Reason:

- the earlier active-porosity proxy winner and screening logic were built on `rho_f = 1000`,
- the corrected quasi-static branch is now solver-healthy but still shape-mismatched at late time,
- so the next evidence-based question is whether restoring the physical fluid transient changes the late-time front profile in the right direction.

## 2026-03-12 physical-inertia follow-up

### Completed physical-inertia validation: `E = 500`, `zeta = 50`, `gamma_u = 10`, `rho_f = 1000`

Validation branch:

- output root: `/tmp/b6_driver_E500_z50_gU10_rho1000_nx24_t0p5`,
- active-porosity path,
- `poiseuille` diffuse traction,
- `nx = 24`, `ny = 12`,
- `dt = 0.05`,
- `t_ramp = 2.0 s`,
- direct LU,
- `gamma_u = 10`,
- `gamma_div = 0.1`, `gamma_div_max = 0.2`.

Short-window result (`t_final = 0.5 s`):

- numerically healthy:
  - no dt reduction,
  - no LM/trust-region stagnation,
  - no KSP failure,
  - no NaNs;
- early comparison remained strong:
  - global `RMSE = 0.974 um`,
  - `y = 150 um`: `RMSE = 5.97 um`,
  - `y = 250 um`: `RMSE = 1.68 um`,
  - `y = 350 um`: `RMSE = 1.46 um`.

Continuation result (`t_final = 1.5 s` on the same branch):

- still numerically healthy through `t = 1.5 s`,
- no dt reduction,
- accepted every step,
- late-window comparison over `t ∈ [0, 1.47] s`:
  - global `RMSE = 29.1 um`,
  - `y = 150 um`: `RMSE = 30.7 um`,
  - `y = 250 um`: `RMSE = 25.4 um`,
  - `y = 350 um`: `RMSE = 26.4 um`.

Late-time front values:

- at `t = 1.0 s`:
  - global `≈ 15 um`,
  - `y167 ≈ 16 um`,
  - `y250 ≈ 19 um`,
  - `y333 ≈ 22 um`;
- at `t = 1.2 s`:
  - global `≈ 21 um`,
  - `y167 ≈ 23 um`,
  - `y250 ≈ 24 um`,
  - `y333 ≈ 28 um`;
- at `t = 1.5 s`:
  - global `≈ 31 um`,
  - `y167 ≈ 36 um`,
  - `y250 ≈ 53 um`,
  - `y333 ≈ 38 um`.

Interpretation:

- restoring `rho_f = 1000` changes the **shape** in the right direction:
  - the late-time response is much flatter than the `rho_f = 0` branch,
  - the earlier centerline overshoot is largely removed;
- but the whole benchmark is now underdriven:
  - all histories are still well below the extracted video by `t ≈ 1.5 s`.

So the next lever should be **amplitude**, not another stabilization or LM change.

### Evidence-based next move

Increase the diffuse-traction scale on this same physical-inertia branch.

Reason:

- with `rho_f = 1000`, `zeta = 50` looks like an amplitude deficit more than a shape defect,
- the earlier active-porosity onset screen already identified `zeta = 50` and `zeta = 100` as the numerically clean onset band,
- and `--scale-alpha-ch-eps-with-zeta` is already enabled, so the next `zeta` step respects the current analytical scaling rule.

### Completed follow-up: `zeta = 100` on the same physical-inertia branch

Run family:

- output root: `/tmp/b6_driver_E500_z100_gU10_rho1000_nx24_t0p5`,
- same setup as the `zeta = 50` branch except `diffuse_shear_scale = 100`.

Short-window result (`t_final = 0.5 s`):

- numerically healthy with fixed `dt = 0.05`,
- no dt reduction / no LM stagnation / no KSP failure / no NaNs,
- early comparison over `t ∈ [0, 0.467] s`:
  - global `RMSE = 0.983 um`,
  - `y = 150 um`: `RMSE = 6.00 um`,
  - `y = 250 um`: `RMSE = 1.03 um`,
  - `y = 350 um`: `RMSE = 1.29 um`.

So the stronger branch still clears the short-validation gate.

Continuation result (`t = 1.5 s`):

- still solver-healthy,
- dynamic per-height histories become the best so far:
  - `y = 150 um`: `RMSE = 24.0 um`,
  - `y = 250 um`: `RMSE = 9.93 um`,
  - `y = 350 um`: `RMSE = 14.7 um`.

But the global history remains wrong:

- global `RMSE = 39.2 um`,
- at `t = 1.5 s`:
  - global `≈ -5 um`,
  - `y167 ≈ 65 um`,
  - `y250 ≈ 92 um`,
  - `y333 ≈ 76 um`.

Continuation result (`t = 2.0 s`):

- still numerically healthy through `t = 2.0 s`,
- but the late branch now overdrives the tracked heights while the global toe remains pinned/backward:
  - global `≈ -17 um`,
  - `y167 ≈ 128 um`,
  - `y250 ≈ 142 um`,
  - `y333 ≈ 134 um`.

History comparison to `t = 2.0 s`:

- global `RMSE = 62.5 um`,
- `y = 150 um`: `RMSE = 22.3 um`,
- `y = 250 um`: `RMSE = 18.2 um`,
- `y = 350 um`: `RMSE = 16.0 um`.

The `steady_dian` contour check at `t = 2.0 s` also remains poor:

- `steady_profile_rmse_um = 206.0`,
- `steady_front_y150_err_um = 79.7`.

Interpretation:

- `zeta = 100` is the first corrected branch that gets the **y-resolved dynamic histories** close,
- but it is still not publishable because the global/front-toe observable and the 2-second contour remain wrong.

### Evidence-based next move

Increase stiffness on the same `zeta = 100`, `rho_f = 1000` branch.

Reason:

- the stronger forcing fixed the amplitude deficit but now overdrives the tracked heights by `2.0 s`,
- the old stiffness screen improved the high-`zeta` branch monotonically with increasing `E`,
- so the next evidence-based question is whether a stiffer solid can reduce the late contour distortion while keeping the now-corrected dynamic onset.

### Completed stiffness check: `E = 1000` is indistinguishable from `E = 500` on the validated window

Run:

- output root: `/tmp/b6_driver_E1000_z100_gU10_rho1000_nx24_t0p5`,
- same branch as above except `E = 1000 Pa`.

Observed result:

- the `t <= 0.5 s` history is identical to the `E = 500` branch to numerical precision,
- and the continued run stayed step-for-step identical through `t ≈ 1.05 s` before it was stopped deliberately.

Measured maximum difference versus the `E = 500` branch over the `t <= 0.5 s` timeseries:

- `dx_front_global`: `4.88e-11 m`,
- `dx_front_y167um`: `1.20e-11 m`,
- `dx_front_y250um`: `4.11e-11 m`,
- `dx_front_y333um`: `3.66e-11 m`,
- `phi_mean_alpha_weighted`: `7.45e-11`.

Interpretation:

- in the current `zeta = 100`, `rho_f = 1000`, `nx = 24`, `dt = 0.05` regime,
  raising `E` from `500` to `1000` does not materially move the observed branch.
- So a further stiffness-only sweep is not the most useful next step.

### Current bottom line

- The LM/globalization problem is fixed.
- The best dynamic-history branch so far is:
  - `E = 500`,
  - `zeta = 100`,
  - `rho_f = 1000`,
  - `gamma_u = 10`,
  - `nx = 24`,
  - `dt = 0.05`,
  - `t_ramp = 2.0`.
- But this branch is still not publishable because the global front and contour observables remain wrong.

### Follow-up result: corrected `E = 500`, `zeta = 50`, `gamma_u = 10`

That branch did exactly what the archived proxy evidence suggested on the early
window, but it still failed the full late-time history check.

Observed sequence:

- short corrected probe to `t = 0.5 s`
  (`/tmp/b6_driver_E500_z50_gU10_t0p5`):
  - numerically clean,
  - video comparison over `t ∈ [0, 0.467] s`:
    - global `RMSE = 1.03 um`,
    - `y150 RMSE = 5.90 um`,
    - `y250 RMSE = 2.00 um`,
    - `y350 RMSE = 0.988 um`;
- continued corrected run to `t = 1.5 s`
  (`/tmp/b6_driver_E500_z50_gU10_t1p5`):
  - still numerically clean after resuming from `t = 1.325 s`,
  - but comparison over `t ∈ [0.501, 1.47] s` degraded to:
    - global `RMSE = 35.0 um`,
    - `y150 RMSE = 39.0 um`,
    - `y250 RMSE = 14.2 um`,
    - `y350 RMSE = 32.1 um`.

Late-time displacement state at `t = 1.5 s`:

- simulation:
  - global `≈ 36.4 um`,
  - `y167 ≈ 36.1 um`,
  - `y250 ≈ 112.2 um`,
  - `y333 ≈ 45.4 um`;
- extracted video:
  - global `≈ 88.1 um`,
  - `y150 ≈ 95.9 um`,
  - `y250 ≈ 86.5 um`,
  - `y350 ≈ 88.1 um`.

Practical conclusion:

- the branch is **not** publishable despite the early success;
- the same late-time shape defect remains:
  - the middle height outruns the video,
  - while the global and outer-height fronts lag badly.

This makes the next move clearer.

- The quasi-static simplification (`rho_f = 0`) is no longer enough to judge the
  promising branch.
- The earlier staged search and proxy winner that motivated `E = 500 -- 1000`,
  `zeta = 50` were built on the physical-fluid branch (`rho_f = 1000`).
- So the next evidence-based screening run should reintroduce `rho_f = 1000`
  around this now-promising high-`E`, `zeta = 50` branch, using the corrected LM
  implementation and a short direct validation first.

## 2026-03-12: global-observable correction and corrected `zeta = 100` revalidation

### Newly confirmed measurement root cause

The late-time Benchmark-6 failure was not only physics. The previous global
observable was also too sensitive to tiny contour tails.

Evidence from the existing inertial `E = 500`, `zeta = 100`, `gamma_u = 10`
branch:

- previous `t = 2.0 s` stored values reported:
  - `dx_front_global = -17.05 um`,
  - `dx_front_y167um = 128.11 um`,
  - `dx_front_y250um = 142.12 um`,
  - `dx_front_y333um = 133.59 um`;
- re-measuring the same saved `alpha = 1/2` contour as an area-like left
  quantile gave about `64.7 um` instead.

That mismatch is too large to be a physical conclusion. It means the old
point-quantile global metric was letting a tiny toe/tail dominate the full
"global" history.

### Code correction applied

- `examples/biofilms/benchmarks/blauert/blauert_biofilm_deformation_one_domain.py`
  now computes `x_front_global` from row-wise leftmost `alpha = 1/2` contour
  intersections instead of a quantile over raw contour points.
- We then use `--global-front-quantile 0.05` on the corrected branch, which is
  the closest robust analogue of the video extractor's left-quantile over the
  segmented area.

### Revalidation of the best inertial `zeta = 100` branch

Corrected short run:

- output: `/tmp/b6_driver_E500_z100_gU10_rho1000_nx24_t0p5_obsfix`,
- early video comparison over `t ∈ [0, 0.467] s` stayed essentially unchanged:
  - global `RMSE = 0.952 um`,
  - `y150 RMSE = 6.00 um`,
  - `y250 RMSE = 1.03 um`,
  - `y350 RMSE = 1.29 um`.

Corrected continuation:

- output: `/tmp/b6_driver_E500_z100_gU10_rho1000_nx24_t2p0_obsfix`,
- numerically healthy to `t = 2.0 s`,
- no dt reduction, no retries, no NaNs, no KSP failure, no LM stagnation.

Corrected scientific result:

- video comparison over `t ∈ [0.501, 1.97] s`:
  - global `RMSE = 30.4 um`,
  - `y150 RMSE = 26.0 um`,
  - `y250 RMSE = 21.4 um`,
  - `y350 RMSE = 18.8 um`;
- `dynamic_08pa` scalar observation:
  - `front_compression_2p0_um = 60.71`,
  - target `148.0`,
  - error `87.29 um`.

Interpretation:

- the observable correction was necessary and materially improved the branch,
- but `E = 500`, `zeta = 100`, `rho_f = 1000` is still not publishable,
- because the corrected global/front history still underpredicts the benchmark.

### Updated next move

Do **not** spend more time on stiffness-only changes at the same `zeta = 100`
branch.

The next short validation should instead test the archived inertial proxy winner
under the corrected observable path:

- `E = 1000`,
- `zeta = 50`,
- `rho_f = 1000`,
- `gamma_u = 10`,
- direct LU,
- corrected global front output.

Reason:

- the corrected `zeta = 100` branch is now clearly solver-healthy but still
  globally underdriven,
- the earlier staged inertial proxy search selected `E = 1000`, `zeta = 50` as
  the best early-window branch,
- and the current observable correction removes the previous ambiguity in the
  global metric, so that proxy winner can now be judged on the right output.

### Completed corrected proxy-winner run: `E = 1000`, `zeta = 50`, `rho_f = 1000`

Observed outputs:

- short corrected run:
  - `/tmp/b6_driver_E1000_z50_gU10_rho1000_nx24_t0p5_obsfix`,
  - early window remained strong:
    - global `RMSE = 0.932 um`,
    - `y150 RMSE = 5.97 um`,
    - `y250 RMSE = 1.65 um`,
    - `y350 RMSE = 1.48 um`;
- continued corrected run:
  - `/tmp/b6_driver_E1000_z50_gU10_rho1000_nx24_t2p0_obsfix`,
  - solver reached `t = 2.0 s` without dt reduction or loss of step acceptance,
  - but late-window science degraded to:
    - global `RMSE = 40.7 um`,
    - `y150 RMSE = 44.2 um`,
    - `y250 RMSE = 33.4 um`,
    - `y350 RMSE = 35.5 um`,
  - `dynamic_08pa` `front_compression_2p0_um = 52.82`.

Interpretation:

- the archived inertial proxy winner does **not** survive the corrected
  observable path and the longer benchmark window;
- it is worse than the corrected `E = 500`, `zeta = 100` branch on the full
  scientific window.

### Final evidence-based conclusion for this turn

- The LM/globalization failure is fixed.
- The corrected best branch is now:
  - `E = 500`,
  - `zeta = 100`,
  - `rho_f = 1000`,
  - `gamma_u = 10`,
  - `nx = 24`,
  - `dt = 0.05`,
  - `t_ramp = 2.0`,
  - corrected global front output.
- But publishable Benchmark-6 agreement is still not reached:
  - corrected best branch full-window video RMSEs remain
    `30.4 / 26.0 / 21.4 / 18.8 um`,
  - and its `2.0 s` front compression is only `60.7 um` against the
    `148 um` paper target.

So the remaining blocker is no longer solver stability. It is model calibration /
physics fit on the corrected observable definitions.

## 2026-03-12: Batch A completion and Batch B promotion

Completed evidence:

- promoted full-window branch:
  - `/tmp/b6A_pois_z100_kappa2x_gU10_rho1000_nx24_t1p5`
  - corrected `t = 2.0 s` score:
    - global `RMSE = 19.27 um`,
    - mean per-height `RMSE = 29.01 um`,
    - max per-height `RMSE = 36.74 um`,
    - global bias `= -10.95 um`,
    - contour mean `RMSE = 215.27 um`,
    - contour max-time `RMSE = 250.28 um`,
    - `front_compression_2p0_um = 80.07`.
- combined screen:
  - `/tmp/b6B_pois_z100_kappa2x_gU12p5_rho1000_nx24_t1p5`
  - corrected `t <= 1.5 s` screen:
    - global `RMSE = 25.52 um`,
    - mean per-height `RMSE = 18.41 um`,
    - max per-height `RMSE = 28.58 um`,
    - global bias `= -11.53 um`,
    - contour mean `RMSE = 187.95 um`,
    - contour max-time `RMSE = 189.92 um`.

Interpretation:

- `kappa_inv` is now clearly the strongest late-time amplitude control in the
  corrected physical-inertia regime.
- Its isolated effect is not publishable because it flips the sign of the
  late-time `y250/y350` errors and worsens contour mismatch.
- Adding `gamma_u = 12.5` on top of the same `kappa_inv` is the first tested
  move that pulls those heights back to the correct side of zero while keeping
  most of the amplitude gain.

Updated source-of-truth ranking:

1. Promote `/tmp/b6B_pois_z100_kappa2x_gU12p5_rho1000_nx24_t1p5`
   to `t = 2.0 s`.
2. Screen the same combined family with `diffuse_shear_scale = 130`.
3. If that remains healthy and still uniformly low, screen
   `diffuse_shear_scale = 150`.

Reason for raising `diffuse_shear_scale` next:

- after `kappa_inv + gamma_u`, the remaining corrected history defect at
  `t = 1.5 s` is mostly uniform underprediction rather than the previous
  shape split;
- with `--scale-alpha-ch-eps-with-zeta` already active, increasing `zeta`
  within the numerically clean onset band is the most direct supported way to
  raise the forcing without discarding the improved shape controls.

## 2026-03-12: resumed-session lagged-stress bracket

The pending lower-bracket canary has now finished scoring:

- `/tmp/b6F_laggedstress_s1_kappa2x_gU15_rho1000_nx24_t1p0/score_t1p0.json`
- corrected screen metrics at `t <= 1.0 s`:
  - global `RMSE = 9.01 um`,
  - mean per-height `RMSE = 15.28 um`,
  - max per-height `RMSE = 25.66 um`,
  - global bias `= -4.50 um`,
  - contour `RMSE(1.0 s) = 174.68 um`.
- direct timeseries amplitudes at `t = 1.0 s`:
  - global `≈ 7.31 um`,
  - `y250 ≈ 18.92 um`,
  - `y333 ≈ 14.82 um`.

Interpretation:

- `lagged_stress` at scale `1` is a valid lower bracket:
  it is numerically healthy, and its corrected history/contour balance is
  already competitive with the best poiseuille screens;
- but the absolute compression is much too low, so the next move is to bracket
  the scale upward instead of retuning the same poiseuille family again.

Operational note:

- no Benchmark 6 simulation remained active at the resumed checkpoint;
- the next runs use the highest normal-build kernel optimization documented in
  `pycutfem/jit/cpp_backend/compiler.py`:
  `PYCUTFEM_CPP_FAST_COMPILE=0 PYCUTFEM_CPP_OPT_LEVEL=3`.

Ranked next batch:

1. `lagged_stress`, `diffuse_shear_scale = 3`, `t_final = 1.0 s`.
2. `lagged_stress`, `diffuse_shear_scale = 5`, `t_final = 1.0 s`.
3. Promote the better branch only if it gains amplitude without breaking the
   current `~175-190 um` contour floor.

## 2026-03-12: lagged-stress scale bracket outcome

Results:

- `scale = 5` completed:
  - `/tmp/b6H_laggedstress_s5_kappa2x_gU15_rho1000_nx24_t1p0`
  - corrected `t <= 1.0 s` score:
    - global `RMSE = 13.54 um`,
    - mean per-height `RMSE = 18.63 um`,
    - max per-height `RMSE = 28.43 um`,
    - global bias `= -8.94 um`,
    - contour `RMSE(1.0 s) = 171.92 um`,
    - screen score `= 0.9531`.
- `scale = 3` was not a valid promotion candidate:
  - first parallel launch hit the same optimized-cache race
    (`ImportError: ... .so: file too short`);
  - after relaunch on the warmed cache, it reached only `t = 0.50 s` and then
    spent multiple minutes inside step-11 / Newton-2 assembly with no new
    accepted step while still using `~150%` CPU;
  - branch was terminated and recorded as
    `early_stop_stalled_halfwindow`.

Interpretation:

- `lagged_stress` remains scientifically interesting because the scale-1 canary
  still gives the best corrected `t <= 1.0 s` score in this family;
- but increasing the traction scale does not move the solution toward the
  publishability gate in a simple way:
  `scale = 5` slightly improves the 1.0 s contour while *worsening* the overall
  history score and driving the global / upper fronts back negative by `t=1.0 s`.

Updated conclusion:

- the next calibration lever in the `lagged_stress` family should be
  **stiffness**, not more scale;
- keep `diffuse_shear_scale = 1` fixed and bracket `E` downward:
  - `E = 350`,
  - `E = 250`,
  still with `gamma_u = 15`, `kappa_inv = 1.962e12`, and the direct-LU solver
  stack.

## 2026-03-12: lagged-stress stiffness bracket outcome

Results:

- `E = 250`:
  - `/tmp/b6J_laggedstress_s1_E250_kappa2x_gU15_rho1000_nx24_t1p0`
  - corrected `t <= 1.0 s` score:
    - global `RMSE = 8.93 um`,
    - mean per-height `RMSE = 15.21 um`,
    - max per-height `RMSE = 25.70 um`,
    - global bias `= -4.38 um`,
    - contour `RMSE(1.0 s) = 174.96 um`,
    - screen score `= 0.8062`.
- `E = 350`:
  - `/tmp/b6I_laggedstress_s1_E350_kappa2x_gU15_rho1000_nx24_t1p0`
  - matched the same front history trend up to `t = 0.95 s`,
  - but the final step entered a much heavier LM-assisted solve and was not
    worth promoting over `E = 250`;
  - recorded as `not_selected_same_history_heavier_solver`.

Interpretation:

- lowering `E` helps only **marginally** in this family:
  it improves the corrected score by about `0.002` relative to the original
  `E = 500` scale-1 canary, while leaving the contour floor essentially
  unchanged (`~175 um`);
- that makes stiffness a secondary lever, not the main path to a publishable
  contour match.

Updated next batch:

- keep the new best lagged-stress screen point fixed:
  - `diffuse_shear_scale = 1`,
  - `E = 250`,
  - `kappa_inv = 1.962e12`;
- bracket `gamma_u` downward next:
  1. `gamma_u = 12.5`,
  2. `gamma_u = 10`.

## 2026-03-12: lagged-stress gamma-u bracket outcome

Results:

- `gamma_u = 12.5` was not worth promoting:
  - it only reached `t = 0.6 s`,
  - it was weaker than the `gamma_u = 10` branch in the tracked fronts,
  - and it stalled late enough to justify early stop.
- `gamma_u = 10` is the new best lagged-stress screen:
  - merged `dt = 0.05` screen score:
    - out dir:
      `/tmp/b6N_laggedstress_s1_E250_kappa2x_gU10_rho1000_nx24_t1p0_dt0p05_merged`
    - global `RMSE = 8.27 um`,
    - mean per-height `RMSE = 14.78 um`,
    - max per-height `RMSE = 24.78 um`,
    - global bias `= -3.60 um`,
    - contour `RMSE(1.0 s) = 178.87 um`,
    - screen score `= 0.7936`.

Important solver conclusion:

- the first `gamma_u = 10` screen stalled after `t = 0.9 s` while another
  branch was running concurrently;
- restarting from the same checkpoint:
  - with `dt = 0.025` finished immediately,
  - and then a single-branch restart with the original `dt = 0.05` also
    completed cleanly to `t = 1.0 s`.

Interpretation:

- `gamma_u` is the strongest lagged-stress lever tested so far;
- the best corrected history is now in the lagged-stress family;
- but the contour floor is still essentially unchanged, so the remaining
  publishability gap is still geometry, not just history amplitude.

Updated next step:

- promote the new best lagged-stress branch
  (`scale = 1`, `E = 250`, `gamma_u = 10`) to `t = 1.5 s` from scratch,
  alone on the machine, and score it on the corrected `1.5 s` window.

## 2026-03-12: first lagged-stress `t = 1.5 s` promotion outcome

Run:

- `/tmp/b6O_laggedstress_s1_E250_kappa2x_gU10_rho1000_nx24_t1p5`
- exact physics/solver stack:
  - `diffuse_shear_model = lagged_stress`,
  - `diffuse_shear_scale = 1`,
  - `E = 250`,
  - `gamma_u = 10`,
  - `kappa_inv = 1.962e12`,
  - `rho_f = 1000`,
  - `nu = 0.4`,
  - direct LU / MUMPS,
  - `dt = 0.05`,
  - `t_final = 1.5`,
  - `max_it = 12`.

Observed failure mode:

- the run advanced cleanly through `t = 0.90 s`;
- the last accepted log line was:
  - `Time step 18: t=9.000000e-01s, dt=5.000e-02, nNewton=3, accepted=True`;
- step 19 then showed:
  - `VI Newton 1` accepted with a noticeably harder solve
    (`lm_lambda = 6.25e-03`, `merit_ratio = 0.317`);
  - `VI Newton 2: assembling...`;
- after that point the run stopped making visible progress:
  - `run.log` frozen at `2026-03-12 16:40:12 +0100`,
  - `timeseries.csv` frozen at `2026-03-12 16:39:58 +0100`,
  - no new checkpoint beyond `checkpoint_step=00018.npz`,
  - no new accepted step beyond `t = 0.90 s`.

Interpretation:

- this is the same branch family that already proved physically viable at
  `t = 1.0 s`, so the current evidence does **not** justify discarding the
  parameter set on physics grounds yet;
- but it does show that the branch is still solver-fragile in the late window
  at the production `dt = 0.05` setting.

Decision:

- classify the from-scratch `t = 1.5 s` promotion as `unhealthy_late_step_stall`;
- stop it early instead of burning the remaining wall clock on a frozen step;
- continue from the saved `t = 0.90 s` checkpoint with the shortest diagnostic
  continuation needed to judge the physical late-window trajectory.

## 2026-03-12: restart continuation and rerank

Restart continuation:

- `/tmp/b6P_laggedstress_s1_E250_kappa2x_gU10_rho1000_nx24_t1p5_restart_dt0p025`
- restart source:
  `/tmp/b6O_laggedstress_s1_E250_kappa2x_gU10_rho1000_nx24_t1p5/restart/checkpoint_step=00018.npz`
- change relative to the stalled from-scratch run:
  - `restart_dt = 0.025`.

Observed outcome:

- the continuation immediately cleared the old `t = 0.90 s` barrier and
  accepted steps through `t = 1.275 s`;
- representative accepted steps:
  - `t = 0.925 s`, `nNewton = 4`,
  - `t = 1.000 s`, `nNewton = 2`,
  - `t = 1.175 s`, `nNewton = 4`,
  - `t = 1.275 s`, `nNewton = 2`;
- but it then froze again with no new checkpoint or time-series row after
  `2026-03-12 16:57:04 +0100`.

Merged corrected diagnostic:

- merged out dir:
  `/tmp/b6Q_laggedstress_s1_E250_kappa2x_gU10_rho1000_nx24_t1p25_restart_dt0p025_merged`
- corrected `t <= 1.25 s` score:
  - global `RMSE = 12.78 um`,
  - mean per-height `RMSE = 15.00 um`,
  - max per-height `RMSE = 23.98 um`,
  - global bias `= -7.21 um`,
  - contour `RMSE(1.0 s) = 178.85 um`,
  - screen score `= 0.8976`.

Raw late-window comparison is more decisive than that scalar:

- at `t = 1.275 s` the branch remains strongly underdriven relative to the
  video extraction:
  - global `7.90 um` vs video `100.17 um`,
  - `y150/167`: `22.99 um` vs video `94.23 um`,
  - `y250`: `55.45 um` vs video `95.12 um`,
  - `y350/333`: `32.06 um` vs video `100.35 um`.

Interpretation:

- the `lagged_stress`, `scale = 1`, `E = 250`, `gamma_u = 10`,
  `kappa_inv = 1.962e12` point is still too weak in the corrected late window;
- the repeated stall is therefore not hiding a good late-time physical match;
- the next move should be a stronger **physical coupling** probe rather than
  another solver-only tweak on the same point.

Updated ranked queue:

1. `kappa_inv = 3.924e12`, keep `E = 250`, `gamma_u = 10`.
2. `E = 150`, keep `kappa_inv = 1.962e12`, `gamma_u = 10`.
3. Combine them only if one of the single-parameter probes is clearly better.
