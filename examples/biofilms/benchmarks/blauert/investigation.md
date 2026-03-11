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

## Literature reference points
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
