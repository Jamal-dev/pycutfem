# Investigation: `biofilm_channel_sloughing.py` u-block stability

## Goal

Make the skeleton displacement block (`u_x,u_y`) *robustly solvable* in the channel sloughing demo, including **post-failure chunk motion**, without relying on fragile tuning of `dt`/Newton parameters.

Success criteria:
- No Newton failures in the early-time sloughing run (baseline reproducer below) for reasonable `dt` (target: `2e-4`).
- Add **edge-case regression tests** that isolate the u-block well-posedness (fully damaged / no-biofilm / inertia+drag only).
- Keep the formulation **coercive / well-conditioned** when stiffness and permeability are degraded by damage and/or when `alpha→0`.
- Once stable: update `examples/biofilms/model/model.tex` to match the implemented stabilization/kinematics.

Related code:
- Driver: `examples/biofilms/biofilm_channel_sloughing.py`
- One-domain forms: `examples/utils/biofilm/one_domain.py` (skeleton equation + inertia + u-extension + u-CIP)
- Documentation: `examples/biofilms/model/model.tex`

---

## Baseline failure (reproducer)

### Fast reproducer (fails at time step 3)

This caps Newton to 20 iterations so the failure is reached quickly (otherwise it can spend many iterations stagnating):

```bash
outdir=garbage/repro_sloughing_u_block_t3_maxit20
rm -rf "$outdir" && mkdir -p "$outdir"

PYCUTFEM_RESIDUAL_TRACE=1 \
PYCUTFEM_RESIDUAL_TRACE_FIELDS=1 \
PYCUTFEM_RESIDUAL_TRACE_COORDS=1 \
conda run --no-capture-output -n fenicsx \
  python -u examples/biofilms/biofilm_channel_sloughing.py \
    --backend cpp \
    --case dian_paper_sloughing_gap \
    --nx 30 --ny 20 --q 6 \
    --dt 2e-4 --t-final 6e-4 \
    --eps 4e-5 \
    --fix-base \
    --no-freeze-alpha --no-alpha-from-refmap \
    --D-alpha 0 \
    --alpha-cahn-M 0.1 --alpha-cahn-gamma 1.0 \
    --alpha-cahn-conservative --alpha-cahn-mobility degenerate \
    --no-conserve-alpha \
    --mass-every 1 \
    --gamma-u-pin 1e-3 \
    --u-extension l2 --gamma-u 2 \
    --u-cip 2 --u-cip-weight fluid \
    --solid-model linear \
    --solid-visco-eta 0.1 \
    --Umax 9e-2 \
    --damage-kappa-stiff 1e-6 --damage-kappa-perm 1e-6 \
    --vtk-every 0 \
    --newton-tol 1e-6 --newton-rtol 0 \
    --max-it 20 \
    --dump-state-every 1 \
    --no-stop-on-steady \
    --outdir "$outdir" \
  2>&1 | tee "$outdir/run.log"
```

Observed outcome (Feb 18, 2026, local run):
- Step 2: Newton stagnates around `|R|_∞ ~ 1e-5` and is **accepted** by the solver’s “stagnated near tolerance” logic.
- Step 3: Newton does not converge within 20 iterations and errors with:
  - `Newton failed at step 3 with dt=2.000e-04: Newton did not converge – adjust Δt or verify Jacobian.`
  - Worst residuals are consistently in `u_x/u_y` (often near the biofilm base / crack region).

### Optional diagnostics

To collect matrix stats / eigenvalues on every Newton iteration (expensive):

```bash
export PYCUTFEM_MATRIX_STATS=1
export PYCUTFEM_MATRIX_STATS_STEP=1
export PYCUTFEM_MATRIX_STATS_IT=1
export PYCUTFEM_MATRIX_EIGS_K=6
```

---

## Working hypotheses (what likely causes the u-block instability)

H1. **Loss of coercivity / near-nullspaces in `u`** during damage + sloughing onset.
- When `g_stiff(d)` is small locally, the elastic contribution to the `u` Jacobian becomes weak.
- Kelvin–Voigt viscosity does **not** damp rigid translation/rotation of detached chunks (∇vS≈0 ⇒ viscous stress≈0).
- Drag `β(v-vS)` couples to the fluid; if the chunk moves nearly with the fluid, this provides little *direct* damping for rigid-body-like `u` modes.

H2. **Eulerian inertia + CG discretization behaves hyperbolically** in damaged/slipping zones.
- With stiffness weakened and/or drag effectively reduced, the skeleton block behaves closer to transport/hyperbolic dynamics where plain H¹ Galerkin can stagnate without strong stabilization.

H3. **Kinematic inconsistency** when interpreting `u` as an Eulerian reference-map variable.
- In neo-Hookean refmap mode, `u` is tied to `X(x,t)`; then the consistent kinematic evolution is
  `∂t u + w·∇u = w` with a solid velocity `w`.
- Current formulation uses `vS=(u^{n+1}-u^n)/dt` and also uses `vS` as the advecting velocity in several Eulerian equations, which can mix frames once motion becomes large.

H4. **Non-smooth operators + global absolute tolerance** create a Newton floor.
- Even if the physics is fine, max/history/positive-part constructs can cap Newton at ~1e-5–1e-6 unless smoothing and/or scaling is adjusted.

---

## Experiments (results)

### E1. Switch `--u-extension l2` → `--u-extension grad`

Rationale: `u-extension l2` adds a strong `(1-α) u` penalty in the fluid that *anchors* `u` and therefore fights rigid-body translation of a detached chunk (since `u` is CG and continuous). This can create a residual floor / slow line-search progress once inertia-driven chunk motion starts.

Result (local run, Feb 18, 2026):
- Baseline (`--u-extension l2`) fails at step 3 with `--max-it 20` (see reproducer above).
- With `--u-extension grad --gamma-u-pin 1e-12` (keeping `--u-cip 2 --u-cip-weight fluid`), the same run:
  - Step 2 is accepted near convergence (`|R|_∞≈4e-6`).
  - Step 3 converges to tolerance (`|R|_∞≈5e-7`) in 4 Newton iterations.

Conclusion: for **post-failure chunk motion**, `u-extension grad` is the correct stabilization choice for a continuous `u` field.

### E2. Guardrail: override `u-extension l2` in sloughing+inertia runs

Rationale: even though `u-extension l2` is coercive and makes the `u` block full rank, it anchors a continuous CG `u` in the free-fluid region. When solid inertia is enabled (sloughing runs), this can directly oppose rigid-body-like chunk translation and lead to Newton stagnation/failure once motion starts.

Result (local run, Feb 18, 2026, `dian_paper_sloughing_gap`, `nx=30,ny=20,q=6,dt=2e-4`, strict `newton_tol=1e-6`, `max-it=10`):
- with explicit `--u-extension l2`: Newton fails at step 3 (worst residual in `u_y`, `|R|_∞≈8e-5`).
- with `--u-extension grad`: step 3 converges (`|R|_∞≈4e-7`).

Fix (implemented): in `examples/biofilms/biofilm_channel_sloughing.py`, if `--process` includes sloughing, solid inertia is enabled, and the user requests `--u-extension l2`, the driver prints a warning and overrides `u-extension` to `grad` (keeping the user’s `--gamma-u-pin` if provided, otherwise using a tiny default).

### E3. Alpha vs damage: why `alpha` is not 0 where `d≈1` (and why high `v` does not imply “alpha moved”)

Observed “puzzle” (user report): as detachment/cracking begins, `d` reaches 1 in parts of the biofilm and the *fluid* velocity increases there, but `alpha` does not go to 0 (so the region still looks like biofilm in ParaView).

What the code actually models today:
- `alpha` is an **indicator / tracer** for “biofilm region” (and is used in mixture coefficients like `rho(alpha,phi)`, `mu(alpha,phi)`).
- `d` is a **bulk damage / cohesion-loss** field used only through degradation factors:
  - stiffness: `g_stiff(d) = (1-κ_stiff)(1-d)^2 + κ_stiff`
  - permeability / drag: `g_perm(d) = (1-κ_perm)(1-d)^2 + κ_perm`
  (`examples/utils/biofilm/one_domain.py`).
- There is **no coupling** in the alpha evolution that enforces `alpha→0` when `d→1`.

So, in this formulation, **`d=1` does not mean “biofilm absent”**; it means “biofilm present but mechanically failed / highly permeable”.

Why “high velocity in damaged regions” happens while `alpha` stays nonzero:
- The drag coupling is `β (v - vS)` with `β ∝ alpha * phi^2 * kappa_inv * g_perm(d)`.
- When `d→1`, `g_perm(d)≈κ_perm`, so `β` can collapse by many orders of magnitude.
- Then the fluid can slip relative to the skeleton (`|v-vS|` becomes large), and `v` can be high in the damaged zone even if the skeleton does not translate away.

Concrete evidence from the dumped VTK output at step ~178 (local run, Feb 2026, `nx=30,ny=20,dt=2e-4`):
- In nodes with `d>0.9`: `beta` is O(1e4–1e5) and `|vS|` is much smaller than `|v|`, so `|v-vS|` is O(1e-1).
- In intact biofilm nodes (e.g. `d<0.1` and `alpha>0.5`): `beta` is O(1e8) and `v≈vS` (very small slip).
Interpretation: the “gap” seen in velocity is primarily a **low-drag / high-slip region**, not necessarily an `alpha=0` void.

Actionable visualization/diagnostics to disambiguate:
- Plot `vS` and the **slip** `|v-vS|` (already exported as `vS`; slip can be computed in ParaView).
- Plot `beta` (drag proxy) and `g_perm(d)` (compute `g_perm` in ParaView from `d` and `damage_kappa_perm`).
- Plot “intact biofilm indicator” `alpha_intact := alpha*(1-d)` if you want a scalar that vanishes when material is fully damaged.

Notes on the chosen sloughing preset:
- In `--case dian_paper_sloughing_gap`, growth is disabled (`k_g=0`) and detachment is disabled (`k_det=0`) by design; so there is **no sink** that would remove `alpha` mass.
- If you want “alpha follows the chunk and leaves a true void”, the preset’s intended choice is `--alpha-from-refmap` (transport `alpha(x,t)=alpha0(x-u)`), not solving an Allen–Cahn/Cahn–Hilliard PDE for `alpha`.

Why does `alpha_area` (and sometimes `int_alpha`) increase?
- `alpha_area` in the log is **not** the domain area of `alpha>0.5`. It is the *bottom-wall contact measure*
  \[
    \alpha_{\text{area}} := \int_{\Gamma_{\text{wall}}} \alpha \, ds
  \]
  assembled in `examples/utils/biofilm/adhesion.py` (used to normalize the RMS shear). As the diffuse interface spreads or more of the wall has `alpha>0`, this number can increase even if the total mass is unchanged.
- Naming: `--alpha-cahn-*` is **Allen–Cahn** regularization; true **Cahn–Hilliard** is `--alpha-ch-*`.
- In the reported run, `int_alpha := ∫_Ω α dx` also increases significantly (e.g. step 1: `6.982e-07` vs step 178: `9.577e-07` in `examples/biofilms/results/dian_paper_sloughing_gap_fix_base_ch/run.log`). This is **not** a ParaView artifact: these integrals are assembled from the FE solution in the driver.
- Root cause (before fix): when you solve `alpha` (`--no-freeze-alpha --no-alpha-from-refmap`) the transport term was written in **non-conservative advective form** `vS·∇α`. If `div(vS)≠0` (common in the poro/mixture setting) then `∫_Ω α dx` drifts even with no boundary flux. The conservative Allen–Cahn constraint only constrains the Allen–Cahn regularization part, not this advection term.
- Fix (implemented): the alpha transport now uses the conservative form `div(α vS)` (implemented as `vS·∇α + α div(vS)` for backend robustness) in `examples/utils/biofilm/one_domain.py`. With `k_det=0`, `k_g=0`, and no boundary flux, this removes the systematic mass drift.
- Remaining safety net: keep `--conserve-alpha` enabled if you want to enforce `int_alpha(t)=int_alpha(0)` exactly in long runs despite discretization errors and/or any boundary flux.

VTK gotcha (important for interpretation):
- Many of these sloughing runs use a Q2 geometry mesh with Q1 scalars (`alpha,phi,d,a,p,…`). If you export Q1 scalars naively as point data on the Q2 nodes, you will create spurious zeros on mid-edge nodes and ParaView can show misleading patterns.
- Fix (implemented): `pycutfem/io/vtk.py` now upsamples Q1-on-Qp quad scalars to all mesh nodes by bilinear interpolation of the element corner values, so point-data fields like `alpha` and `d` are meaningful in ParaView.

### E4. Convergence past the ~179-step stall

The long sloughing run was reported to fail at step ~179 with residuals dominated by `u_x/u_y` and a collapsing line-search.

With the current driver defaults/guardrails (notably `--u-extension grad` for sloughing+inertia) and the existing parameter set used in the report, restarting from the accepted state at step 178 and advancing several more steps succeeds without Newton failure (local check: steps 179–184 converge).

If the run still stalls on your machine:
- Ensure you are using `--ls-mode dealii` (default) for the sloughing run; Armijo can be much slower and is more prone to “tiny α” stagnation on non-smooth residuals.
- Increase stabilization where the `u` block loses coercivity:
  - try `--u-cip 5` (or 10) and `--u-cip-weight both`,
  - increase floors: `--damage-kappa-perm 1e-5`–`1e-4` and/or `--damage-kappa-stiff 1e-6`–`1e-5`.
- Enable adaptive time-step reduction as a safety net:
  - `--allow-dt-reduction --dt-min 5e-5 --dt-reduction-factor 0.5`

Checkpointing reminder:
- `--dump-state-every` does **nothing** unless `--dump-state` (or env `PYCUTFEM_DUMP_STATE=1`) is enabled.

---

## Test plan (edge cases for the u-block)

We need tests that fail today and pass after stabilization changes, but run fast (small meshes).

T1. **α≡0 everywhere** (no biofilm):
- Expectation: the coupled system reduces to “fluid + harmless u-extension”; the `u` DOFs must not create singular/ill-conditioned solves.

T2. **d≡1 everywhere, α≡1 everywhere, elasticity off**:
- Expectation: with stiffness removed, the `u` equation should still be solvable via inertia and/or drag and/or extension.
- Purpose: isolate “post-failure” modes without elastic regularization.

T3. **Inertia+drag only, rigid translation MMS**:
- Expectation: a manufactured rigid-body motion should be reproducible (or at least converge) under the chosen stabilization, demonstrating “chunk motion” is numerically controlled.

T4. **Regression: sloughing-gap early steps**
- Very small version of the driver (reduced mesh/short time) that must not throw a Newton exception.

---

## Fix plan (phased)

P0. Instrumentation / scaling
- Add per-field norms/tolerances (or block scaling) so u does not dominate the global `|R|_∞` decision.

P1. Guarantee minimal coercivity in `u`
- Ensure that in any region where stiffness is degraded, there remains a robust *direct* stabilizing mechanism:
  - drag floor, or
  - extension/pinning inside weak-α zones, or
  - a dedicated rigid-mode damping term (careful: should not overdamp physical motion).

P2. Improve hyperbolic robustness when inertia dominates
- Revisit u-CIP scaling and/or add SUPG/CIP targeted to the inertial/transport-like part.
- Consider switching to `u-extension grad` by default for sloughing (with pinning) if it improves robustness.

P3. (Optional but likely) introduce a solid velocity unknown `w`
- Momentum in `w` + kinematics `∂t u + w·∇u = w` (Eulerian-consistent).
- This makes inertia stabilization “NS-like” on `w` and removes implicit/inconsistent differentiation through `u`.

P4. Update documentation
- Once changes are in, update `examples/biofilms/model/model.tex` to reflect the final stabilized formulation and any new unknowns.

---

## Status

- [x] Reproduced Newton failure at step 3 (see reproducer).
- [x] Added u-block edge-case tests (T1/T2 + grad+pin variant).
- [x] Identified and validated a robust u-extension choice for chunk motion (`--u-extension grad` + tiny pin).
- [x] Added a sloughing+inertia guardrail: override `--u-extension l2` → `grad` to avoid Newton failure during chunk motion.
- [x] Updated `examples/biofilms/model/model.tex` with u-block edge-case test discussion.
- [x] Added a dynamic chunk-motion regression (T3; MMS 3 moving cylinder / rigid-chunk transport):
  `tests/test_biofilm_one_domain_mms_moving_cylinder_smoke.py` (backend=python, nx=2, Δt=0.02, 1 step),
  asserting bounded errors (`||u-u_exact||_{L2}<1e-3`, `||α-α_exact||_{L2}<0.3`) and successful Newton solve.
- [ ] Add a sloughing-gap early-step integration regression (T4; must not throw Newton exceptions).
