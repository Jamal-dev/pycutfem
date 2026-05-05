# Investigation: Li et al. (2020) deformation benchmark vs one-domain poroelastic model

## Goal
Create a **publishable** benchmark setup that validates the one-domain biofilm model in:
- `examples/utils/biofilm/one_domain.py`

against the **experimental deformation** reported by:
- Li et al., *Biotechnology and Bioengineering* (2020), DOI `10.1002/bit.27491`
- Local copy: `examples/biofilms/benchmarks/lie/Li-2020.pdf`

This benchmark targets **deformation under flow** (Figure 7 / Video S1), *not* growth/detachment.

---

## Experimental inputs (Video S1)
Source:
- `examples/biofilms/benchmarks/lie/additional_data/bit27491-sup-0001-si_v1.avi`

From the paper (Section “Biofilm deformation experiments”):
- Flow cell cross-section: **10 mm × 10 mm** (width × height)
- Rigid support: **1 mm diameter × 3 mm height**
- Biofilm assumed fixed to the top of the support (in simulation).

### What we extract
We extract a contour and a deformation time series `dx(t)` from the OCT video:
- We track the **leftmost** contour intersection (upstream edge) with three horizontal lines:
  - `y = 0.75 H_b`, `0.50 H_b`, `0.25 H_b` where `H_b` is the initial protruding-biofilm height.
- We enforce a *straight* base (biofilm stands on the support).
- We scale pixels → meters by enforcing: **base width = 1 mm**.

Script:
- `examples/biofilms/benchmarks/lie/extract_deformation_timeseries_from_experimental_video_s1.py`

Repro command (writes both `dx(t)` and the frame-0 polygon in mm):
```bash
python -u examples/biofilms/benchmarks/lie/extract_deformation_timeseries_from_experimental_video_s1.py \
  --x-intersection leftmost \
  --straighten-base \
  --out-csv out/_lie_exp_s1_dx_leftmost_sb/timeseries.csv \
  --out-poly0-mm-csv examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm_ts.csv \
  --debug-every 40 \
  --debug-dir out/_lie_exp_s1_dx_leftmost_sb_debug
```

Key outputs used downstream:
- Experimental target time series: `out/_lie_exp_s1_dx_leftmost_sb/timeseries.csv`
- Scaled + straightened frame-0 polygon: `examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm_ts.csv`

---

## Simulation geometry + mesh
We model a 2D channel with a “slit” where the rigid support sits:
- Channel: `L × H = 15 mm × 10 mm`
- Support block (removed from the fluid mesh): `1 mm × 3 mm`
- Support centered at `x = L/2`
- Biofilm polygon placed on the **top** of the block.

Mesh construction is done inside:
- `examples/biofilms/benchmarks/lie/lie_synthetic_deformation_one_domain.py`
  - structured quads, partitioned into 5 rectangles and merged to remove the block region.

Optional local mesh refinement (recommended for publishable stress/contour plots):
- `--refine-biofilm` refines the mesh around the *initial* biofilm bounding box by **one level**
  (supports ≤1 hanging node per edge).
- `--refine-biofilm-pad` controls how much padding is added to the initial biofilm box to define the
  refinement region.
- `--refine-biofilm-levels` increases the refinement depth (e.g. `3` gives a much tighter `α=0.5` contour
  match to the SVG at `t=0`, but increases DOFs significantly).

Mesh visualization:
- `examples/biofilms/benchmarks/lie/plot_background_mesh_slit_plot2.py`
- `examples/biofilms/benchmarks/lie/plot_mesh_setup_with_biofilm.py`

---

## One-domain model choices (what’s enabled/disabled)
We use the one-domain poroelastic model with the following active pieces:
- 2× momentum balance (mixture + skeleton)
- 2× mass balance (mixture incompressibility + porosity)
- 1× indicator transport (`α`)
- 1× kinematic constraint (`u_t = v^S`)

We disable for this benchmark:
- growth/chemistry (`S` frozen)
- detachment/damage

### Indicator transport (requested settings)
Per requirement, we keep:
- `D_alpha = 0` (avoid artificial depletion/diffusion of α)
- **conservative** advection form: `∂t α + div(α v^S) = …`
- Cahn–Hilliard (CH) regularization enabled (mass-conserving smoothing)

In `lie_synthetic_deformation_one_domain.py` defaults:
- `--alpha-advection-form conservative`
- `--alpha-ch-M 1e-12`
- `--alpha-ch-gamma 2e-3`
- `--alpha-ch-eps` defaults to `--eps` (default `2e-4 m`)
- mobility model: `degenerate`

---

## Boundary conditions (important)
Fluid:
- no-slip `v=0` on channel walls and on the removed-block surfaces (`block_*`)
- parabolic inflow at `x=0` with a linear ramp over `t_ramp`
- outlet pressure pinned at `x=L` (`p=0`)

Biofilm attachment (critical for stress development):
- **clamped** on `block_top`:
  - `u = 0`
  - `v^S = 0`

### α “corner wetting” and why the base can appear to go below the support
With **PDE** transport + **Cahn–Hilliard** regularization, the diffuse interface can “wet” down the
vertical block walls at the top corner. In the step-channel geometry, this means the `α=0.5` contour
can enter the *floor region* (left/right of the block, where `y<block_h` exists), which looks like the
biofilm base is “tilting into the block” when overlaid on the experimental video.

This is a **simulation** effect (not just VTK visualization): CH has natural no-flux type boundary
conditions unless an explicit wall-energy/contact-angle model is added, so the contact line can move.

For the Lie benchmark we assume **no detachment** and a **fixed footprint** on the support. To keep
the biofilm indicator out of the floor region, the driver supports:
- `--alpha-clip-below-block` (recommended): freezes `α=0` (and `φ=1`) for all DOFs with `y<block_h`.

Optional extra pinning (usually not needed if clipping is enabled):
- `--alpha-pin-block-top`: sets `α=α0` on `block_top` (pins contact line to the initial footprint).
  - stronger variant: add `--alpha-pin-block-top-value 1` to enforce `α=1` at the base (no detachment), at the cost of strict mass conservation.
    - use `--alpha-pin-block-top-alpha0-min` (default `0.9`) to avoid forcing `α=1` on diffuse corner/edge nodes (prevents a “tilted base” artifact).
- `--alpha-zero-block-sides`: sets `α=0` on `block_left/block_right` (prevents wall “wetting”).

---

## Solver robustness notes
For speed and robustness:
- use `--backend cpp` (PETSc linear solves; nonlinear solve via SNES or PDAS)
- LU/MUMPS (default in driver)
- optionally allow time-step reduction when Newton fails (`--allow-dt-reduction`)

### Early-stop vs full time horizon (`--stop-on-steady`)
If you run with `--stop-on-steady`, the time stepper can terminate early when the Newton update
falls below `--steady-tol`. For reproducibility:
- `timeseries.csv` will still contain a final row at `t_final` with the last `dx` value repeated.
- VTK output will stop at the last accepted step (it does **not** synthesize intermediate frames).

This is why a run with `--t-final 20` can end at (say) 9 time steps but still show a `t=20` row.

### Alpha bounds (0<=alpha<=1)
In several Lie calibration runs (e.g. `out/_lie_pde_ch/vtk`), `alpha` can overshoot outside `[0,1]`
once the interface starts moving (typical for CG advection of sharp fronts):
- observed examples: `min(alpha)≈-0.32`, `max(alpha)≈1.63` starting at `step=0005`.

The driver now supports a **box-constrained** nonlinear solve via PDAS:
- default: `--newton-solver pdas` (enforces `0<=alpha<=1`)
- to reproduce the older PETSc SNES behavior: `--newton-solver snes`

For diagnostics, enable:
- `--alpha-metrics` → writes `alpha_metrics.csv` with `∫alpha dx`, centroid, and min/max per step.

Additionally, `pycutfem` was extended with an opt-in “accept best SNES iterate” behavior:
- `pycutfem/solvers/nonlinear_solver.py`: `NewtonParameters.accept_nonconverged_atol_factor`
- used in the driver via `--snes-accept-factor` (typical value: `3`)

Interpreting solver logs (important for publishability):
- PETSc SNES may report **0 Newton iterations** when the predictor already satisfies `--newton-tol` (absolute residual test).
- The printed `ΔU_step∞` is a **time-step increment** (current minus previous state), not the Newton correction.
  It can be dominated by auxiliary unknowns like `mu_alpha` (Cahn–Hilliard) even when displacements are small.
- For SNES runs, the log also prints `Δx_snes∞`, which is the **change applied by SNES** to the reduced unknown vector.

Jacobian/residual consistency (finite-difference check):
```bash
pytest -q tests/test_biofilm_one_domain_alpha_cahn_hilliard_jacobian_fd.py::test_biofilm_one_domain_alpha_cahn_hilliard_jacobian_fd_consistency
```

---

## Calibration to experiment (current best fit)
Directly using Li Table-1 values (`G_b≈69736 Pa`, `μ_b≈30494 Pa·s`) in this **poroelastic** one-domain surrogate yields near-zero deformation under the very low shear stresses of the slow (`u_avg=6e-4 m/s`) mm-scale channel.

For matching the **experimental deformation curves**, we treat the poroelastic parameters as **effective** (because this model couples flow → skeleton primarily via Brinkman drag / pore-pressure mechanisms rather than explicit surface-traction as in Li’s two-fluid viscoelastic model).

### Best-fit parameter set (produces nonlinear rise + plateau)
Run directory:
- `out/_lie_best_fit_v1_linear`

Parameters (tracked in git for reproducibility):
- `examples/biofilms/benchmarks/lie/best_fit_v1.json`

### Note on Figure 8 velocity comparisons (synthetic case)
Li et al. ramp the inflow from 0→`u_avg` over **1 s** (Section “Numerical implementation” in `examples/biofilms/benchmarks/lie/main.tex:208`).
Many calibration runs here use a slower ramp (e.g. `--t-ramp 8`) to reduce transient solver difficulty. This has a *direct* effect:
- at `t=2 s`, the applied inflow is only `2/8=0.25` of its final value, so the velocity field is ~**4× smaller** than the paper’s Figure 8 snapshot at `t=2 s`.

Also note:
- with `--allow-dt-reduction`, the physical time corresponding to `vtk/step=XXXX.vtu` is **not** `XXXX*dt` unless `dt` stayed constant; always read `timeseries.csv` for the actual `t_s` grid.
- if you run with the effective best-fit parameters (`--mu-b-model mu --mu-b-fluid 1e-3 --kappa-inv 1e8`), the biofilm is intentionally made *much* more fluid-like / permeable than Li’s synthetic-biofilm material (Table 1), so the velocity penetration pattern will differ from Figure 8 even if the inflow magnitude matches.

Command:
```bash
conda run --no-capture-output -n fenicsx python -u examples/biofilms/benchmarks/lie/lie_synthetic_deformation_one_domain.py \
  --backend cpp \
  --transport-mode pde \
  --alpha-advection-form conservative \
  --alpha-ch-M 1e-12 --alpha-ch-gamma 2e-3 --D-alpha 0 \
  --mu-b-model mu --mu-b-fluid 1e-3 \
  --solid-model linear \
  --G-b 5e-3 --mu-b 3e-2 --kappa-inv 1e8 \
  --u-avg 6e-4 --t-ramp 8 \
  --dt 1.0 --t-final 20 \
  --allow-dt-reduction --dt-min 0.1 --dt-reduction-factor 0.5 \
  --stop-on-steady --steady-tol 1e-12 \
  --nx 80 --ny 40 --nx-left 28 --nx-mid 24 --nx-right 28 --ny-bottom 8 --ny-top 32 \
  --vtk-every 0 \
  --out-dir out/_lie_best_fit_v1_linear
```

Comparison plot:
```bash
python -u examples/biofilms/benchmarks/lie/compare_exp_sim_timeseries_dx.py \
  --sim-dir out/_lie_best_fit_v1_linear \
  --smooth-exp 5
```

Outputs:
- `out/_lie_best_fit_v1_linear/timeseries.csv`
- `out/_lie_best_fit_v1_linear/compare_exp_sim_dx.png`

Current error level (Video-S1 extracted target; 3 tracking lines):
- total RMSE is O(20–25 µm) depending on early-stop behavior and smoothing window.

---

## Why is “sim line1” mismatching more than the others?
For the current best fit, the per-line RMSE is dominated by **line 1** (top-most tracking line).
This happens because the **shape of the displacement profile vs height** is different:

- Experiment: peak displacement is around the middle line (line 2), with line 1 smaller.
- One-domain model (current settings): displacement tends to increase monotonically with height,
  so the top line (line 1) is the largest.

This makes it difficult to match all three lines simultaneously with only the current effective
poroelastic parameters.

Additionally, line 1 is the most sensitive to small contour irregularities near the free surface
(segmentation noise + local protrusions), so small differences in the extracted contour lead to
larger dx differences on line 1 than on the lower lines.

---

## Finite strain vs small strain (does it help?)
We added a switch to the driver:
- `--solid-model {linear,hencky,svk,neo_hookean}`

On the calibrated Lie benchmark deformation levels (O(100 µm)), switching from **small-strain**
(`--solid-model linear`) to **finite-strain** (`hencky` / `neo_hookean`) did **not** improve the
exp-vs-sim RMSE in quick tests; the dx(t) curves are essentially unchanged within the current
error level.

Quick check (coarser mesh `--nx 40 --ny 20`, same parameters as best-fit):
- `linear`: total RMSE ≈ 27.5 µm
- `hencky`: total RMSE ≈ 27.5 µm (no improvement)
- `neo_hookean`: total RMSE ≈ 29.1 µm (slightly worse)

Practical note: finite-strain models can be significantly more expensive/less robust for the
same settings (larger nonlinear systems), so for this benchmark we currently keep `linear`.

---

## Overlay video (exp + sim)
To generate a publishable overlay video, first run the best-fit case with VTU output:
```bash
conda run --no-capture-output -n fenicsx python -u examples/biofilms/benchmarks/lie/lie_synthetic_deformation_one_domain.py \
  --backend cpp \
  --transport-mode pde \
  --alpha-advection-form conservative \
  --alpha-ch-M 1e-12 --alpha-ch-gamma 2e-3 --D-alpha 0 \
  --mu-b-model mu --mu-b-fluid 1e-3 \
  --solid-model linear \
  --G-b 5e-3 --mu-b 3e-2 --kappa-inv 1e8 \
  --u-avg 6e-4 --t-ramp 8 \
  --dt 1.0 --t-final 20 \
  --allow-dt-reduction --dt-min 0.1 --dt-reduction-factor 0.5 \
  --stop-on-steady --steady-tol 1e-12 \
  --nx 80 --ny 40 --nx-left 28 --nx-mid 24 --nx-right 28 --ny-bottom 8 --ny-top 32 \
  --vtk-every 1 \
  --out-dir out/_lie_best_fit_v1_linear_vtk
```

Then create the overlay video (writes `<sim-dir>/exp_sim_overlay.mp4` by default):
```bash
conda run --no-capture-output -n fenicsx python -u examples/biofilms/benchmarks/lie/make_overlay_video_exp_sim.py \
  --sim-dir out/_lie_best_fit_v1_linear_vtk \
  --straighten-base \
  --base-per-frame
```

Outputs:
- Overlay video: `out/_lie_best_fit_v1_linear_vtk/exp_sim_overlay.mp4`
- VTK snapshots: `out/_lie_best_fit_v1_linear_vtk/vtk/step=0000.vtu` … (open in ParaView)

Note: if you use `--transport-mode refmap`, the alpha PDE regularization flags (`--alpha-ch-*`, `--alpha-cahn-*`)
are ignored (they only apply to `--transport-mode pde`). To silence the warning, pass `--alpha-ch-M 0 --alpha-ch-gamma 0`.

---

## 2026-02-27 — Scale-bar anchored extraction + smoothed t=0 geometry (Figure 5(b) intent)

This update standardizes the experimental extraction so it matches the paper’s geometry conventions:

- Anchor at the paper’s ⊗ (left end of the rigid-support top).
- Straighten the base on the rigid support.
- Scale pixels → meters using the **embedded 100 µm scale bar** in Video S1.
- Use a **smoothed/simplified** t=0 geometry consistent with Figure 5(b) (remove fissures, keep macroscopic humps).

### New tracked inputs (benchmark artifacts)
- Experimental target time series (Video S1): `examples/biofilms/benchmarks/lie/exp_s1_dx_leftmost_scalebar.csv`
- Smoothed t=0 polygon (mm, origin at ⊗, straight base 0→1 mm): `examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm_scalebar_smooth.csv`

### Geometry extraction (frame 0)
Script:
- `examples/biofilms/benchmarks/lie/extract_geometry_from_video_s1_frame0.py`

Command (writes debug overlays to `out/` and the tracked polygon CSV):
```bash
python -u examples/biofilms/benchmarks/lie/extract_geometry_from_video_s1_frame0.py \
  --clahe \
  --out-poly-mm-csv examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm_scalebar_smooth.csv \
  --debug-dir out/_lie_exp_geom_debug_s1_frame0
```

Measured (from script stdout, frame 0):
- scale bar: `100 µm / 49 px` → `m_per_px = 2.040816e-06 m/px`
- inferred support width: `490 px` (consistent with `block_w = 1 mm`)
- initial exposed-biofilm height: `H_b ≈ 1.332653 mm` (mask-row method; robust to thin top artifacts)

### Deformation extraction (Video S1)
Script:
- `examples/biofilms/benchmarks/lie/extract_deformation_timeseries_from_experimental_video_s1.py`

Command (writes the tracked CSV):
```bash
python -u examples/biofilms/benchmarks/lie/extract_deformation_timeseries_from_experimental_video_s1.py \
  --clahe \
  --straighten-base \
  --scale-mode scalebar \
  --hb-method mask \
  --x-method mask \
  --x-intersection leftmost \
  --debug-every 40 \
  --debug-dir out/_lie_exp_s1_dx_leftmost_scalebar_debug \
  --out-csv examples/biofilms/benchmarks/lie/exp_s1_dx_leftmost_scalebar.csv
```

Notes:
- `--hb-method mask` defines `H_b` from the segmented mask (top-most row with row-count ≥ `0.2 * max_row_count`), which avoids including the thin diagonal bright streak above the specimen in Video S1.
- `--scale-mode scalebar` is the source-of-truth scaling; the rigid support width (`1 mm`) is only used to set the straightened base span (and as a sanity check).

### Solver-side tracking-line robustness (benchmark driver)
`lie_synthetic_deformation_one_domain.py` now enforces monotone ordering when snapping tracking-line y-levels to available alpha DOF y-levels, preventing accidental line reordering on coarse meshes.

### Current exp-vs-sim agreement (effective poroelastic surrogate)
Best-fit (calibration mesh; cpp backend) is recorded in:
- `examples/biofilms/benchmarks/lie/best_fit_v2_scalebar.json`

For the recorded best-fit run:
- RMSE total ≈ `31.7 µm`
- Relative error (paper definition; “every second datum”) total ≈ `38.7 %`

This is **still larger** than the synthetic-biofilm relative errors reported in Li et al. (2020) (3.2–21.1%). The likely causes are:
- remaining uncertainty in extracting a DIC-equivalent displacement signal from noisy OCT frames, and
- physics mismatch: Li et al. validate a **two-fluid viscoelastic PF Oldroyd-B** model, whereas this benchmark uses an **effective one-domain poroelastic surrogate** (parameters are “effective”, not directly transferable from Table 1).

---

## Next improvements (if tighter agreement is needed)
1) Improve experimental `dx(t)` robustness (reduce top-line sensitivity to small protrusions):
   - tune segmentation (`--blur-ksize`, morphology, `--simplify-eps`)
   - consider tracking a “main-body” intersection rather than the extreme leftmost outlier if multiple intersections exist.
2) Parameter optimization automation:
   - `examples/biofilms/benchmarks/lie/optimize_params_to_experiment_dx.py`
   - run with `--use-conda-run fenicsx --allow-dt-reduction --stop-on-steady` and a larger eval budget.
3) Final publishable run:
   - re-run best-fit on a finer mesh (e.g. `--nx 200 --ny 80`) and confirm convergence of `dx(t)`.

---

## 2026-02-28 — Fix upper-right hump + make exp/sim tracking lines consistent

### Geometry extraction: keep the upper-right hump (avoid the “flat roof”)
Problem observed:
- The previous frame-0 polygon was missing the upper-right hump / appeared too flat near the top-right, compared to Fig. 5(b).

Fix (in `extract_geometry_from_video_s1_frame0.py`):
- For contour-mode extraction, replace the global “top row” clipping with a **per-column cap** computed from the **longest contiguous foreground run** in each column (rejects the thin diagonal streak without flattening the true top curvature).
- Add `--cap-run-quantile` (default `0.10`) so the cap uses a small quantile within the longest run (robust to 1–2 pixel spikes at the very top).
- Compare the extracted top-envelope to Fig. 5(b) traced coordinates on a restricted x-range where the trace actually contains the top boundary.

Repro (writes the tracked polygon CSV + debug overlays/plots):
```bash
python -u examples/biofilms/benchmarks/lie/extract_geometry_from_video_s1_frame0.py \
  --polygon-method contour \
  --fig5b-traced-csv examples/biofilms/benchmarks/lie/additional_data/traced_coordinates_mm_5b.csv \
  --out-poly-mm-csv examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm_scalebar_smooth.csv \
  --debug-dir out/_lie_exp_geom_debug_s1_frame0
```

Key outputs:
- Frame overlay (video frame 0): `out/_lie_exp_geom_debug_s1_frame0/overlay_polygon_frame.png`
- Fig. 5(b) top-envelope comparison: `out/_lie_exp_geom_debug_s1_frame0/compare_fig5b_top_envelope.png`
- Polygon plot (mm): `out/_lie_exp_geom_debug_s1_frame0/polygon_mm.png`

Measured (stdout, frame 0):
- scale bar: `100 µm / 49 px` → `m_per_px = 2.040816e-06 m/px`
- inferred support width: `490 px` (consistent with `block_w = 1 mm`)
- extracted polygon height: `H_b = 1.282756 mm` (from the saved polygon)
- Fig. 5(b) top-envelope match on `x∈[0.60,0.86] mm`: `RMSE = 0.0664 mm`, `max|Δ| = 0.1005 mm`

### Experimental dx(t): enforce the same H_b used in simulation
Problem observed:
- The experimental extractor’s default `--hb-method mask` produced `H_b ≈ 1.33 mm`, while the **simulation tracking lines** were based on the **smoothed polygon** height (`≈ 1.283 mm`), so the y-levels didn’t match exactly.

Fix (in `extract_deformation_timeseries_from_experimental_video_s1.py`):
- Add `--hb-from-polygon-mm-csv` to override `H_b` using the tracked smoothed polygon (max `y_mm`).

Repro (writes the tracked experimental CSV):
```bash
python -u examples/biofilms/benchmarks/lie/extract_deformation_timeseries_from_experimental_video_s1.py \
  --clahe \
  --straighten-base \
  --scale-mode scalebar \
  --hb-from-polygon-mm-csv examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm_scalebar_smooth.csv \
  --x-method mask \
  --x-intersection leftmost \
  --mask-y-band-px 3 \
  --mask-x-quantile-left 0.10 \
  --debug-every 0 \
  --out-csv examples/biofilms/benchmarks/lie/exp_s1_dx_leftmost_scalebar.csv
```

### Current exp-vs-sim agreement (after the hump + H_b consistency fixes)
Comparison command:
```bash
python -u examples/biofilms/benchmarks/lie/compare_exp_sim_timeseries_dx.py \
  --exp-csv examples/biofilms/benchmarks/lie/exp_s1_dx_leftmost_scalebar.csv \
  --sim-dir out/_lie_opt_scalebar_v3_mu/run_219554f3cd97
```

Metrics (paper relative error definition, “every 2nd datum”):
- RMSE total ≈ `35.5 µm`
- Relative error total ≈ `44.6 %` (line1≈48%, line2≈40.5%, line3≈43.6%)

This is still above the low synthetic-case errors reported by Li et al. (2020). The dominant remaining issues appear to be:
- **experimental intersection noise** (especially for the lowest line), and
- **step-like simulated dx(t)** on coarse meshes when tracking the `α=0.5` contour via nodal values (dx remains 0 until the interface crosses enough DOFs to change the detected intersection).

### Reproducible current run (2026-02-28, cpp backend)
Simulation run used for the reported current error:
```bash
conda run --no-capture-output -n fenicsx python -u examples/biofilms/benchmarks/lie/lie_synthetic_deformation_one_domain.py \
  --backend cpp \
  --transport-mode pde \
  --alpha-advection-form conservative \
  --alpha-ch-M 1e-12 --alpha-ch-gamma 2e-3 --D-alpha 0 \
  --alpha0-file examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm_scalebar_smooth.csv \
  --mu-b-model mu --mu-b-fluid 1e-3 \
  --solid-model linear --G-b 1.20536e-4 --mu-b 3e-2 --kappa-inv 1e8 \
  --u-avg 6e-4 --t-ramp 8 \
  --dt 1.0 --t-final 20 --newton-tol 1e-5 --max-it 15 \
  --snes-accept-factor 5 --allow-dt-reduction --dt-min 0.2 --dt-reduction-factor 0.5 \
  --stop-on-steady --steady-tol 1e-11 \
  --nx 60 --ny 30 --nx-left 21 --nx-mid 18 --nx-right 21 --ny-bottom 6 --ny-top 24 \
  --q 6 --vtk-every 0 \
  --out-dir examples/biofilms/benchmarks/lie/_run_sim_v2
```

Comparison:
```bash
python -u examples/biofilms/benchmarks/lie/compare_exp_sim_timeseries_dx.py \
  --exp-csv examples/biofilms/benchmarks/lie/exp_s1_dx_leftmost_scalebar.csv \
  --sim-dir examples/biofilms/benchmarks/lie/_run_sim_v2
```

Reported:
- RMSE total: `35.46 µm`
- Relative errors (paper metric, every second datum): line1=`48.0%`, line2=`40.5%`, line3=`43.6%`, total=`44.6%`

### Model-comparison plan (paper-facing)
1. Use the same geometric anchor/scaling convention for experiment and simulation:
   - origin at ⊗, straight base `x∈[0,1] mm`, scale bar `100 µm`.
2. Match tracking-line definitions exactly:
   - `y = 0.75/0.50/0.25 * H_b` with `H_b` taken from the same smoothed polygon file in both extraction and simulation setup.
3. Compare on two metrics:
   - RMSE in `µm` (optimization objective),
   - Li-style relative error (`sum|Dm-De|/sum|De|`, every second datum) for publication reporting.
4. State physics mismatch explicitly:
   - Li: two-fluid viscoelastic PF with Oldroyd-B;
   - this benchmark: one-domain poroelastic surrogate with effective parameters.
5. Publish both:
   - best achievable fit and stability envelope,
   - remaining mismatch attribution (measurement noise + surrogate-physics limitation).

### Bug finding + regression guard
Bug fixed in benchmark extraction:
- A global top-row clipping strategy could flatten the upper-right hump when a thin diagonal streak exists above the biofilm in Video S1.

Fix:
- contour-mode clipping now uses a per-column cap from the **longest contiguous foreground run**.

Regression tests added:
- `tests/test_lie_benchmark_geometry_helpers.py`
  - verifies top-envelope binning does not stay all-NaN,
  - verifies longest-run cap ignores short upper streaks (prevents hump flattening regression).

---

## 2026-02-28 — Similarity-aligned ⊗ anchor + slit-sealed polygon + α\_min tuning (MAE beats paper)

The scale-bar workflow above produced a usable benchmark, but the anchor placement and boundary profile
still depended on segmentation heuristics. To make the setup **paper-aligned** and less sensitive to
pixel artifacts, we switched to a similarity-based alignment using Fig. 5(a), then extracted a smoothed
polygon directly from Video S1.

### Geometry (t=0): similarity-based extraction with smoothing
Script:
- `examples/biofilms/benchmarks/lie/extract_biofilm_similarity_stripes.py`

Key improvements for a publishable polygon:
- **Similarity transform** (ECC affine) from Fig. 5(a) → Video S1 frame 0 to map the paper’s ⊗ anchor.
- Scale from the embedded **100 µm scale bar**.
- **Slit sealing** (`--seal-slits-k`) to remove thin vertical fissures/segmentation dropouts that would
  otherwise create large zig-zags in the external contour.
- Arclength **resample + smoothing** (`--smooth-*`) to remove pixel jaggedness.

Tracked outputs:
- Anchor (pixel + scale): `examples/biofilms/benchmarks/lie/anchor_frame0_similarity.json`
- Smoothed polygon (mm, origin at ⊗, straight base 0→1 mm): `examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm_similarity_smooth.csv`

Repro:
```bash
python -u examples/biofilms/benchmarks/lie/extract_biofilm_similarity_stripes.py \
  --video examples/biofilms/benchmarks/lie/additional_data/bit27491-sup-0001-si_v1.avi \
  --frame 0 \
  --fig5a examples/biofilms/benchmarks/lie/additional_data/figure_5_a.jpg \
  --anchor-json examples/biofilms/benchmarks/lie/anchor_frame0_similarity.json \
  --prior-csv examples/biofilms/benchmarks/lie/additional_data/traced_coordinates_mm_5b.csv \
  --seal-slits-k 15 \
  --smooth-window-mm 0.03 --smooth-ds-mm 0.004 --n-verts 260 \
  --out-csv examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm_similarity_smooth.csv \
  --debug-dir out/_lie_similarity_frame0
```

### Experimental dx(t): DIC-like optical-flow tracking
Script:
- `examples/biofilms/benchmarks/lie/extract_deformation_timeseries_from_experimental_video_s1.py`

We use LK optical flow to track the three paper-style boundary points (Fig. 7(a) green dots), with
tracking-line heights defined from the **same** smoothed polygon (`H_b = max(y_mm)`).

Tracked output:
- `examples/biofilms/benchmarks/lie/exp_s1_dx_optflow_leftmost_similarity.csv`

Repro:
```bash
python -u examples/biofilms/benchmarks/lie/extract_deformation_timeseries_from_experimental_video_s1.py \
  --video examples/biofilms/benchmarks/lie/additional_data/bit27491-sup-0001-si_v1.avi \
  --clahe --straighten-base --scale-mode scalebar \
  --hb-from-polygon-mm-csv examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm_similarity_smooth.csv \
  --x-intersection leftmost --x-method mask --mask-y-band-px 3 --mask-x-quantile-left 0.10 \
  --dx-method opticalflow --flow-offset-in-px 3 \
  --debug-every 40 --debug-dir out/_lie_exp_s1_dx_optflow_leftmost_similarity_dbg \
  --out-csv examples/biofilms/benchmarks/lie/exp_s1_dx_optflow_leftmost_similarity.csv
```

### Simulation: the skeleton-DOF restriction is a dominant calibration lever
Key finding:
- When using `--restrict-skeleton-method alpha`, the threshold `--restrict-skeleton-alpha-min` strongly controls the deformation magnitude.
  Setting it too high effectively clamps interface-adjacent skeleton DOFs and yields under-predicted dx.
- For a time-independent active set that does not need updating as the interface advects, prefer
  `--restrict-skeleton-method box` (default) with a sufficiently large `--restrict-skeleton-box-pad`.

Best reproducible run (effective poroelastic surrogate; cpp backend):
```bash
conda run --no-capture-output -n fenicsx python -u examples/biofilms/benchmarks/lie/lie_synthetic_deformation_one_domain.py \
  --backend cpp \
  --transport-mode pde \
  --alpha-advection-form conservative \
  --alpha-ch-M 1e-12 --alpha-ch-gamma 2e-3 --D-alpha 0 \
  --alpha0-kind polygon \
  --alpha0-file examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm_similarity_smooth.csv \
  --alpha0-scale 1e-3 --alpha0-align block \
  --mu-b-model mu --mu-b-fluid 1e-3 \
  --solid-model linear --G-b 1.20536e-4 --mu-b 3e-2 --kappa-inv 1e8 \
  --u-avg 6e-4 --t-ramp 8 \
  --dt 1.0 --t-final 20 --newton-tol 1e-5 --max-it 15 \
  --snes-accept-factor 5 --allow-dt-reduction --dt-min 0.2 --dt-reduction-factor 0.5 \
  --stop-on-steady --steady-tol 1e-11 \
  --restrict-skeleton-method alpha --restrict-skeleton-alpha-min 0.01 \
  --nx 60 --ny 30 --nx-left 21 --nx-mid 18 --nx-right 21 --ny-bottom 6 --ny-top 24 \
  --q 6 --vtk-every 0 --dx-intersection leftmost \
  --out-dir out/_lie_bestfit_v3_similarity_optflow
```

Compare (prints RMSE/MAE and Li-style relative errors; writes a plot next to the sim CSV):
```bash
python -u examples/biofilms/benchmarks/lie/compare_exp_sim_timeseries_dx.py \
  --exp-csv examples/biofilms/benchmarks/lie/exp_s1_dx_optflow_leftmost_similarity.csv \
  --sim-dir out/_lie_bestfit_v3_similarity_optflow \
  --smooth-exp 5
```

Result (smooth-exp=5):
- RMSE: line1≈20.2 µm, line2≈24.9 µm, line3≈14.7 µm, total≈20.4 µm
- MAE:  line1≈15.9 µm, line2≈20.4 µm, line3≈12.2 µm, total≈16.2 µm
- Relative errors (paper metric, every second datum): line1≈19.8%, line2≈25.4%, line3≈21.1%, total≈22.2%

Paper reference point (Fig. 7, synthetic biofilm; reported “averaged error”):
- line1≈24 µm, line2≈25 µm, line3≈13 µm

On this pipeline, the **MAE per line is below the paper’s reported average errors**, but the
Li-style **relative error** is still larger (likely dominated by differences in how the displacement
signal is extracted from noisy OCT frames, and by the one-domain-poroelastic surrogate physics).

Tracked best-fit record:
- `examples/biofilms/benchmarks/lie/best_fit_v3_similarity_optflow.json`

### Bug finding + regression guard (similarity extractor)
Bug:
- `extract_biofilm_similarity_stripes.py --no-cap-artifacts` could crash with `NameError` due to an
  uninitialized variable in an optional code path.

Fix:
- refactor artifact clipping into `_clip_thin_artifacts_above_cap(..., enabled=...)`.

Regression test:
- `tests/test_lie_similarity_stripes_helpers.py`

---

## 2026-03-01 — Publishable SVG-traced outlines (Inkscape) + refmap calibration + overlay video

The OCT video is hard to segment robustly (jittered contour + anchor drift). To make the benchmark
publishable and reproducible, we switched to **manual per-frame SVG tracing**:

### Traced inputs
- SVG series (one per frame): `examples/biofilms/benchmarks/lie/svg_fles/frame_0000.svg` … `frame_0160.svg`
  - Note: currently **missing** `frame_0123.svg` (extractors skip missing frames).
- Each SVG contains:
  - a traced **base line** on the rigid-support top (anchor + scaling),
  - a traced **biofilm outline**.

Helper module (path parsing + base/outline extraction):
- `examples/biofilms/benchmarks/lie/svg_trace_utils.py`

Regression guard:
- `tests/test_lie_svg_trace_utils.py` ensures disconnected “mark” paths don’t break extraction.

### Experimental dx(t) extraction from SVGs (robust interior tracking)
Script:
- `examples/biofilms/benchmarks/lie/extract_deformation_timeseries_from_svg_series.py`

Key choices:
- Scaling: `--scale-mode base_frame0` enforces **base width = 1 mm** using the traced base in frame 0
  (keeps scaling consistent across frames and avoids scale-bar noise).
- Tracking point: `--dx-quantile 0.1` (interior point) instead of the extreme boundary intersection.
- Ambiguity control: `--track-mode continuity --max-jump-mm 0.2` to avoid cross-section “jumping” when
  multiple intersection segments exist.
- Preferred measurement lines (“middle + upper”): `--y-fracs 0.7,0.6,0.5`

Repro command (writes exp target CSV and the t=0 polygon for the solver):
```bash
python -u examples/biofilms/benchmarks/lie/extract_deformation_timeseries_from_svg_series.py \
  --scale-mode base_frame0 \
  --x-intersection leftmost \
  --dx-quantile 0.1 \
  --track-mode continuity --max-jump-mm 0.2 \
  --y-fracs 0.7,0.6,0.5 \
  --out-csv examples/biofilms/benchmarks/lie/exp_s1_dx_leftmost_svgtrace_y706050_q010_base_cont.csv \
  --out-poly0-mm-csv examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm_svgtrace_q010_base_cont.csv
```

### Best-fit parameters (effective poroelastic surrogate; transport_mode=refmap)
The skeleton-active region restriction is a dominant lever for deformation amplitude.

Two options exist:
- `--restrict-skeleton-method box` (**default, recommended**): keep `(u,vS)` DOFs inside a fixed rectangle
  around the *initial* biofilm (computed from the alpha0 bounding box + `--restrict-skeleton-box-pad`).
  This avoids having to update the active set as the interface advects.
- `--restrict-skeleton-method alpha`: keep `(u,vS)` DOFs where `alpha0 > --restrict-skeleton-alpha-min`.
  This was used for earlier calibration runs and remains reproducible, but the threshold can implicitly
  clamp interface-adjacent skeleton DOFs.

For the recorded `best_fit_v4_svgtrace_refmap.json`, the main “fit lever” was:
- `--restrict-skeleton-method alpha --restrict-skeleton-alpha-min 0.001`

Tracked best-fit record:
- `examples/biofilms/benchmarks/lie/best_fit_v4_svgtrace_refmap.json`

Summary metrics for the recorded fit (see JSON):
- RMSE total ≈ **18.3 µm**
- MAE total ≈ **15.3 µm**
- Relative error (Li metric, every 2nd datum): total ≈ **24.1%**

### PDE-transport validation status
Using the same parameter set in `--transport-mode pde` (with CH enabled) produced a large mismatch
for the lowest tracking line (line 3) in the coarse-mesh run:
- `out/_lie_sim_pde_y706050_best`
- line3 relative error ≈ **134%** (dominating total error)

This likely reflects a combination of:
- coarse-mesh tracking-line snapping (y-lines land on sparse alpha DOF levels), and/or
- multi-intersection ambiguity of the simulated `α=0.5` contour under PDE transport.

For publishable overlay and parameter comparison, the current recommended benchmark path uses
`--transport-mode refmap` (until the PDE-mode tracking/mesh is revisited).

### Overlay video (SVG-traced exp contour + simulated prediction)
1) Run the solver with VTU output:
```bash
conda run --no-capture-output -n fenicsx python -u examples/biofilms/benchmarks/lie/lie_synthetic_deformation_one_domain.py \
  --backend cpp \
  --transport-mode refmap \
  --alpha0-file examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm_svgtrace_q010_base_cont.csv \
  --alpha0-scale 1e-3 --alpha0-align block \
  --restrict-skeleton-dofs --restrict-skeleton-method alpha --restrict-skeleton-alpha-min 0.001 \
  --G-b 9.682e-3 --mu-b 2.182e-2 --kappa-inv 1e9 \
  --mu-b-model mu --mu-b-fluid 30494 \
  --solid-model linear \
  --u-avg 6e-4 --t-ramp 8 \
  --dt 1 --t-final 20 \
  --nx 40 --ny 20 --nx-left 14 --nx-mid 12 --nx-right 14 --ny-bottom 4 --ny-top 16 \
  --q 4 --vtk-every 1 \
  --alpha-ch-M 0 --alpha-ch-gamma 0 \
  --dx-intersection leftmost --dx-quantile 0.1 --y-fracs 0.7,0.6,0.5 \
  --out-dir out/_lie_refmap_y706050_best_vtk_dt1
```

2) Create the overlay (writes `<sim-dir>/exp_sim_overlay_svgtrace.mp4`):
```bash
python -u examples/biofilms/benchmarks/lie/make_overlay_video_exp_sim_svgtrace.py \
  --sim-dir out/_lie_refmap_y706050_best_vtk_dt1
```

The overlay script uses the SVG base line per frame as the anchor, draws the traced exp outline in
green, the predicted outline in red, and **linearly interpolates** between VTU steps (α or `u`,
depending on `--sim-outline`) by default for smooth playback at 8 fps.

---

## 2026-03-02 — Paper Fig. 7 digitization (DIC curves) + conservative Allen–Cahn + publishable tracking overlay

### What the green/red overlays mean
In `make_overlay_video_exp_sim_svgtrace.py`:
- **green** = experimental outline (SVG trace)
- **red** = simulated outline from VTU
  - default: `--sim-outline alpha` (α=0.5 contour, works for both PDE and refmap transport)
  - option: `--sim-outline u_deform_poly` (deform the t=0 polygon with `u`, reference-map style)

Use `--show-tracking` to draw the **exact 3 lines + points** used for `dx(t)` comparison.
For `--track-mode continuity`, the overlay seeds the first-frame point locations from `lines.csv` (`x_ref_m`)
so the points shown at `t=0` match the simulation’s reference locations.

### Are we using DIC?
Yes, in two ways:
- **(A) Paper-DIC digitization:** we digitize the paper’s Fig. 7(b–d) circle-marker curves into a CSV target.
- **(B) Video-S1 DIC (this repo):** we implement a subset-based **translation DIC** pipeline (with SVG-traced
  rigid-base stabilization + optional subpixel refinement) to extract `dx(t)` directly from Video S1.

We also keep a legacy **contour intersection** extractor (non-DIC) for comparison/debugging.

Limitations of the current in-repo DIC:
- translation-only subsets (no local affine warp / strain fitting),
- accuracy depends on the traced SVG base endpoints being correct for each frame.

### DIC extraction from Video S1 (subset correlation + SVG stabilization)
Script:
- `examples/biofilms/benchmarks/lie/extract_deformation_timeseries_dic_video_s1.py`

Example (writes a `dx(t)` CSV + a debug video with the tracked points/vectors):
```bash
python -u examples/biofilms/benchmarks/lie/extract_deformation_timeseries_dic_video_s1.py \
  --out-csv out/_lie_exp_s1_dic_dx/timeseries.csv \
  --out-video out/_lie_exp_s1_dic_dx/dic_debug.mp4 \
  --x-quantile 0.5 --x-span 0.4 --samples-per-line 5
```

If `frame_0000.svg` contains **3 manual mark points**, track exactly those (top/mid/bottom) instead:
```bash
python -u examples/biofilms/benchmarks/lie/extract_deformation_timeseries_dic_video_s1.py \
  --mode marks \
  --out-csv out/_lie_exp_s1_dic_marks_dx/timeseries.csv \
  --out-video out/_lie_exp_s1_dic_marks_dx/dic_debug.mp4
```

For a denser DIC displacement field, use `--mode grid` (optionally with `--max-grid-points` to limit runtime).

Then compare a simulation to the DIC-extracted target:
```bash
python -u examples/biofilms/benchmarks/lie/compare_exp_sim_timeseries_dx.py \
  --exp-csv out/_lie_exp_s1_dic_dx/timeseries.csv \
  --sim-dir <SIM_OUT_DIR>
```

If you use `--mode marks`, you can also compare *simulation* displacements at the same SVG marks
(apples-to-apples) by postprocessing the VTU displacement field `u`:
```bash
python -u examples/biofilms/benchmarks/lie/postprocess_sim_dx_at_svg_marks.py \
  --sim-dir <SIM_OUT_DIR>

python -u examples/biofilms/benchmarks/lie/compare_exp_sim_timeseries_dx.py \
  --exp-csv out/_lie_exp_s1_dic_marks.csv \
  --sim-csv <SIM_OUT_DIR>/timeseries_svg_marks.csv
```

### Digitize the paper’s Fig. 7 displacement curves
Script:
- `examples/biofilms/benchmarks/lie/extract_exp_dx_from_li_fig7_plots.py`

Command (writes a reproducible CSV target + debug images):
```bash
python -u examples/biofilms/benchmarks/lie/extract_exp_dx_from_li_fig7_plots.py \
  --debug-dir out/_lie_fig7_digitize_debug
```

Output:
- `examples/biofilms/benchmarks/lie/exp_fig7_dx_digitized.csv` with columns
  `t_s, dx_line1_m, dx_line2_m, dx_line3_m`.

### Compare any simulation run to the digitized Fig. 7 curves
```bash
python -u examples/biofilms/benchmarks/lie/compare_exp_sim_timeseries_dx.py \
  --exp-csv examples/biofilms/benchmarks/lie/exp_fig7_dx_digitized.csv \
  --sim-dir <SIM_OUT_DIR> \
  --smooth-exp 1
```

### Conservative Allen–Cahn (mass-conserving α evolution)
The one-domain model supports a **mass-conserving conservative Allen–Cahn** option for `α`:
- enable PDE transport: `--transport-mode pde`
- enable Allen–Cahn: `--alpha-cahn-M ... --alpha-cahn-gamma ...`
- enforce mass conservation: `--alpha-cahn-conservative`

Example (skeleton/flow params omitted):
```bash
conda run --no-capture-output -n fenicsx python -u examples/biofilms/benchmarks/lie/lie_synthetic_deformation_one_domain.py \
  --backend cpp \
  --transport-mode pde \
  --alpha-cahn-M 1e-10 --alpha-cahn-gamma 2e-3 --alpha-cahn-eps 2e-4 \
  --alpha-cahn-mobility constant \
  --alpha-cahn-conservative \
  --out-dir out/_lie_pde_ac
```

Note: Allen–Cahn and Cahn–Hilliard are mutually exclusive in the driver.

### Overlay video with tracked points (publishable)
Run solver with VTU output (`--vtk-every 1`) and then:
```bash
python -u examples/biofilms/benchmarks/lie/make_overlay_video_exp_sim_svgtrace.py \
  --sim-dir <SIM_OUT_DIR> \
  --show-tracking \
  --draw-base
```

Example output (includes tracked points/lines):
- `out/_lie_fig8_stress_run/exp_sim_overlay_svgtrace_tracking.mp4`

### Rigid-body drift / center-of-mass motion: why it happens
If the biofilm outline is provided as a **traced polygon of the external interface**, then the polygon boundary corresponds to **`α≈0.5`** in the smooth-step initialization. In the Lie geometry the traced outline includes the **attached base** on `block_top`, so the raw `α0(x,block_h)` is typically `≈0.5` along the footprint.

Consequence:
- If you run with `--alpha-pin-block-top-value 1 --alpha-pin-block-top-alpha0-min 0.9`, the constant value would **not** be applied (because `α0(x,block_h)` never reaches `0.9`), leaving the base under-pinned and allowing apparent “rigid” translation of the phase field.

Fix:
- Use `--alpha-pin-block-top --alpha-pin-block-top-value 1` to enforce **full attachment** on the footprint.
- Also enable `--alpha-clip-below-block` and (recommended) `--alpha-zero-block-sides` to prevent CH “wetting” down the block sides.

The driver now decides whether a block-top node is inside the footprint by probing `α0` slightly *above* the block (rather than at `y=block_h`), so the `0.9` threshold becomes meaningful for polygon-traced outlines.

### Fig. 8-style stress run (dt=0.25) + exp overlay timing
Run directory:
- `out/_lie_fig8_stress_run`

Relative errors (same definition as in Li Fig. 7 caption: sum over every 2nd datum):
- vs digitized paper curves (`examples/biofilms/benchmarks/lie/exp_fig7_dx_digitized.csv`): total ≈ **49%**
- vs SVG-traced Video-S1 target (`examples/biofilms/benchmarks/lie/exp_s1_dx_leftmost_svgtrace_y706050_q010_base_cont.csv`): total ≈ **40.4%**

Timing alignment (why 81 VTUs can overlay 161 frames):
- Video S1 has **161 frames** at **8 fps** → `Δt_video = 0.125 s`, duration `(161-1)/8 = 20 s`.
- With `--dt 0.25` the solver writes VTU steps `0..80` (81 files) → `Δt_sim = 0.25 s`.
- `make_overlay_video_exp_sim_svgtrace.py` maps each frame time `t = frame/fps` to the bracketing
  simulation times in `timeseries.csv` and **linearly interpolates** `α`/`u` between VTU steps by default.
  If you want a 1:1 mapping without interpolation, either run the solver with `--dt 0.125` or use
  `--no-time-interp` in the overlay script.

Stress magnitude sanity check (why your VTU shows ~1e-2 but Li Fig. 8 shows ~2.5e3):
- The exported fields `sigma_newtonian_dev_norm_alpha` and `tau_skel_visc_dev_norm_alpha` scale like
  `μ * |D(v)|` and `η_s * |D(v^S)|` (units: Pa).
- If you run with `--mu-b-model mu --mu-b-fluid 1e-3`, the mixture viscosity is essentially water everywhere,
  so the Newtonian stress scale is `O(1e-3 * u_avg/H) ≈ O(1e-4 Pa)`.
- Li Fig. 8 stresses are `O(1e3 Pa)` because their synthetic-biofilm viscosity is huge (`μ_b≈3e4 Pa·s`,
  Table 1). To get comparable magnitudes in this driver, use a viscosity model that actually applies `μ_b`
  in the biofilm region (e.g. `--mu-b-model alpha_mu --mu-b-fluid 30494`) and match the ramp time
  (`--t-ramp 1` for Fig. 8 at `t=2 s`).
