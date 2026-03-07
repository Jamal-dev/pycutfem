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

**Interpreting SNES “reason=-5” in this benchmark**
- PETSc SNES reports `reason=-5` when it hits `--max-it` without satisfying its stopping criterion.
- In this repo we run SNES with an **infinity-norm** stopping criterion by default (`‖F‖_inf`), so `--newton-tol`
  matches the usual “max residual entry” interpretation used by the pure-Python Newton solver.
- If you see `reason=-5` with `‖F‖_inf` already below `--newton-tol`, that indicates a solver/norm mismatch (should
  not happen); if `‖F‖_inf` is only slightly above `--newton-tol` and the residual breakdown is dominated by `v_*`,
  it is typically an iteration-limit issue (increase `--max-it`) rather than a physical instability.
- `--accept-nonconverged-atol-factor` exists as an escape hatch (default disabled) for cases where SNES reports a
  nonconverged reason but returns a best iterate with sufficiently small `‖F‖_inf`.

#### Observed Newton failure mode (fluid momentum block)
When the long run fails (e.g. `step≈27`, `t≈0.8438 s` in the log you shared), the residual tracing shows:
- `p, u, vS` are essentially converged (≈ machine precision),
- only the **fluid velocity** residuals (`v_x, v_y`) stagnate at `O(10^{-3})`,
- PETSc SNES exits with `DIVERGED_LINE_SEARCH` or `DIVERGED_MAX_IT`.

This is a **time-step/inertia driven** robustness issue in the transient Navier–Stokes part: the solver needs a
smaller `dt` than the current `--dt-min` to keep the nonlinear change per step small enough.

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
