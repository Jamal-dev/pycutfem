# CutFEM Cylinder (DFG 2D-2) — Debug Plan

Goal: match FeatFlow/DFG benchmark reference for **drag (Cd)**, **lift (Cl)**, **Δp**, and the **front stagnation point** for the cylinder using the CutFEM formulation.

This file is meant to be *kept up to date* during the investigation so that we can keep the log of work.

---

## Reference

- Benchmark geometry (DFG 2D cylinder):
  - Channel: `L=2.2`, `H=0.41`
  - Cylinder: `D=0.1` at `(0.2, 0.2)`
  - Parameters: `rho=1.0`, `mu=1e-3`, inflow with mean `U_mean=1.5`
- FeatFlow reference data (bundled in this repo):
  - `examples/turek_cylinder/data/featflow/bdforces_lv*` (Cd/Cl)
  - `examples/turek_cylinder/data/featflow/pointvalues_lv*` (p(A), p(B), Δp)

---

## Run/Compare Harness

- [x] Make CutFEM benchmark write a single `functionals.csv` with `time,Cd,Cl,dp` (no plotting required).
- [x] Add a tiny compare script that loads FeatFlow + our csv and prints RMS/max errors on the overlapping time window.
- [x] Add a CutFEM-vs-volume-only compare script + parity mode toggles (`--disable-mass/--disable-convection`) to isolate Navier terms.
- [ ] Choose a “fast debug mode” (small mesh + short `t_end`) and a “validation mode” (Feat-like settings).

---

## Baselines (must be true before tuning CutFEM)

- [ ] `examples/turek_benchmark_volume_only.py` reproduces FeatFlow trends (Cd/Cl/Δp) with the same `(dt, theta, inflow)` used for CutFEM comparisons.
- [ ] CutFEM and volume-only use *the same* nondimensionalization for Cd/Cl:
  - `Cd = 2 F_x / (rho * U_mean^2 * D)`, `Cl = 2 F_y / (rho * U_mean^2 * D)`

---

## CutFEM Formulation Checks (high priority)

### Geometry / Sign conventions

- [ ] Verify level-set sign convention matches the formulation:
  - `phi < 0` inside cylinder (rigid), `phi > 0` in fluid.
- [ ] Verify interface normal orientation used everywhere is consistent:
  - `n` points from `Ω⁻` (obstacle) to `Ω⁺` (fluid).
- [ ] Verify `physical_domain` for the PDE is *fluid + cut* only (no inside elements).

### Boundary conditions / constraints

- [x] **No pressure pin** is applied.
- [x] **No mean-pressure Lagrange multiplier** is used (do-nothing outlet fixes the pressure constant).

### Nitsche on Γ (cylinder interface)

- [x] Stress tensor consistency:
  - If volume uses `2μ ε(u)` then Nitsche traction must also use `2μ ε(u) n` (not `2μ (∇u) n`).
- [ ] Verify the symmetric term matches the chosen stress:
  - Consistent term: `-(σ(u,p)n, v)_Γ`
  - Symmetric term: `-(σ(v,q)n, u-g)_Γ`
  - Penalty term scaling: `(β (u-g), v)_Γ` with `β ~ μ/h + ρ h/dt`

---

## Stabilization Checks

- [ ] Ghost penalty region equals the intended “cut-neighborhood” (not the whole mesh).
- [ ] Ghost penalty scaling (h, μ, ρ, dt) matches standard theory for the chosen term(s).
- [ ] Pressure stabilization terms (if any) are consistent with mass conservation and do not over-damp.

---

## Physical Diagnostics (must match qualitatively)

- [ ] Front stagnation point:
  - `u(c_x - D/2, c_y)` approximately `(0,0)` (or small, depending on penalty tolerance).
- [ ] Pressure field:
  - high pressure at front stagnation point, low pressure in wake.
- [ ] Lift oscillation:
  - `Cl` oscillates about ~0 (no bias drift).

---

## Hypotheses (track + resolve)

- [x] H1: Nitsche uses the wrong viscous stress (`2μ ∇u` instead of `2μ ε(u)`), corrupting no-slip enforcement and force evaluation.
- [x] H2: Mean-pressure constraint / Lagrange multiplier is active and shifts the pressure/traction incorrectly (should be removed).
- [~] H3: Ghost penalty is applied on an incorrect facet set or with inconsistent scaling (over/under-stabilization).
- [~] H4: Force computation uses mismatched normals/sign conventions vs the interface enforcement.
- [ ] H5: Quadrature order on cut/interface/ghost is too low for Q2 and under-integrates forces.
- [ ] H6: Deformation/level-set mismatch (using analytic φ for some measures and FE φ for others) causes inconsistent geometry.
- [x] H7 (rejected): “too few pressure DOFs near Γ” → equal-order Q2/Q2 should fix the mismatch (it does not; unstable without extra stabilization).
- [x] H8 (rejected): “set velocity DOFs inside the cylinder to 0” improves no-slip (it overconstrains CutFEM extension DOFs and destroys the solution).
- [x] H9: DFG uses a *non-symmetric* stress/traction convention (`σ := ν∇u - pI`) and a matching do-nothing outflow (`ν∂ₙu - p n = 0`). Our previous symmetric traction (`ν(∇u+∇uᵀ) - pI`) was inconsistent with FeatFlow’s reported Cd/Cl and with the outlet BC.

---

## Findings Log

### 2026-01-14

- [x] Moved benchmark into `examples/turek_cylinder/` and relocated FeatFlow data to `examples/turek_cylinder/data/featflow/`.
- [x] Removed mean-pressure constraint / Lagrange multiplier path from the CutFEM benchmark.
- [x] Fixed Nitsche viscous traction to use `∇u+∇uᵀ` (consistent with `2μ ε(u)` in the volume term).
- [x] Aligned DFG scaling: `U_mean=1.0`, `U_max=1.5`, and wrote `functionals.csv` + `compare_featflow.py` for RMS/max comparisons.
- [ ] (next) Re-run with Feat-like `(dt, theta)` and tune stabilization parameters until Cd/Cl/Δp and `u_stag` match the reference.

### 2026-01-15

**Reference (FeatFlow lv6)**, first two output points:
- `t=0.0003125`: `Cd=0.13706829152`, `Cl=-2.1752442712e-04`, `dp=0.08654680758`
- `t=0.0009375`: `Cd=0.14059913852`, `Cl=-2.1666557243e-04`, `dp=0.08780832355`

**CPP backend baseline reproduces the same mismatch (Q2/Q1)**:
- Command:
  - `conda run --no-capture-output -n xfemcustom python examples/turek_cylinder/turek_benchmark.py --backend cpp --dt 0.000625 --theta 0.5 --max-steps 2 --inflow dfg --init stokes --nx 65 --ny 50 --refine-level 1 --with-deformation --ghost-measure patch --gamma-gp 0 --vtk-every 0 --force-eval surface`
- Results:
  - `t=0.0003125`: `Cd≈0.153106`, `Cl≈0.005812`, `dp≈0.063225`
  - `t=0.0009375`: `Cd≈0.209738`, `Cl≈0.010742`, `dp≈0.099687`

**Geometry diagnostics (with deformation)**:
- [x] `|Γ| ≈ π D` and `∫_Γ n ds ≈ 0` (closed curve consistency) on the refined quad mesh.
  - Example: `|Γ|≈3.141601e-01`, `∫n ds≈(3.0e-17, 6.5e-17)` for `D=0.1`.

**Baseline mismatch (CutFEM, Q2/Q1, quad mesh, refine_level=1)**:
- [x] With deformation + Stokes init, no ghost penalty (`--gamma-gp 0`):
  - `t=0.0003125`: `Cd≈0.153106`, `Cl≈0.005812`, `dp≈0.063225`
  - `t=0.0009375`: `Cd≈0.209738`, `Cl≈0.010742`, `dp≈0.099687`
  - Conclusion: drag too large, lift has strong spurious bias, Δp too small early.

**Stabilization sweeps (all with deformation + Stokes init)**:
- [~] Velocity ghost penalty (`--gamma-gp 1e-2 --gamma-gp-p 0`) reduces Cd/Cl slightly, but does not remove lift bias.
  - Example: `t=0.0003125` gives `Cd≈0.1418`, `Cl≈0.00466`, `dp≈0.0603`.
- [x] Pressure ghost penalty (`jump(p)jump(q)` on facet-patch) is *very* sensitive:
  - `--gamma-gp-p 1e-2` strongly damps pressures/forces (`dp` drops to ~`0.036` at the first point and `Cd` to ~`0.10`), i.e. it is not a valid “match the benchmark” setting.
  - Smaller values (e.g. `1e-4`) still bias `dp` noticeably.
  - Conclusion: the current pressure stabilization term/scaling likely needs redesign for this benchmark.

**Ghost region selection**:
- [~] Tested both “fluid-side-only” (`ghost_pos|ghost_both`) and “full ghost band” (`ghost`) facet sets. This changes Cd/Cl significantly for nonzero `gamma_gp`, but does not resolve the mismatch.

**Initialization**:
- [x] `--init stokes` is stable and gives physically reasonable magnitudes.
- [x] `--init zero` currently produces extremely large forces/pressures at the first step for many parameter choices; treat as a separate stability issue (not used for validation until resolved).

**Mesh refinement around the interface**:
- [x] Increasing `--refine-level` from 1→2 (more local refinement) did not materially change the early-time Cd/Cl mismatch; suggests the issue is not simply coarse resolution of the circle.

**Equal-order hypothesis (Q2/Q2) does NOT help (unstable without extra stabilization)**:
- Command:
  - `conda run --no-capture-output -n xfemcustom python examples/turek_cylinder/turek_benchmark.py --backend cpp --fe-order 2 --p-order 2 --dt 0.000625 --theta 0.5 --max-steps 2 --inflow dfg --init stokes --nx 65 --ny 50 --refine-level 1 --with-deformation --ghost-measure patch --gamma-gp 0 --vtk-every 0 --force-eval surface`
- Observations:
  - Forces and Δp flip sign and blow up in magnitude (`ΔU` jumps to ~`9e8` by step 2).
  - Results are physically meaningless (negative Cd and negative Δp at the first two output points).

**“Clamp velocity DOFs inside the cylinder” hypothesis is WRONG for CutFEM extension DOFs**:
- Command:
  - `conda run --no-capture-output -n xfemcustom python examples/turek_cylinder/turek_benchmark.py --backend cpp --dt 0.000625 --theta 0.5 --max-steps 2 --inflow dfg --init stokes --nx 65 --ny 50 --refine-level 1 --with-deformation --ghost-measure patch --gamma-gp 0 --vtk-every 0 --force-eval surface --zero-inside-vel`
- Observations:
  - Clamps ~`906` velocity DOFs (mostly extension DOFs on cut cells) and massively overconstrains the method.
  - Results become wildly wrong:
    - `t=0.0003125`: `Cd≈2.729`, `Cl≈-1.290`, `dp≈0.457`
    - `t=0.0009375`: `Cd≈7.213`, `Cl≈-3.582`, `dp≈0.721`
  - Conclusion: in CutFEM, DOFs whose *nodes* lie inside the obstacle are still part of the polynomial extension on cut cells; clamping them is not a valid way to enforce no-slip.

**Large time step experiment (dt=0.5) is stable but does NOT match FeatFlow (time-discretization error dominates)**:
- CutFEM (Q2/Q1, `--init stokes`, `--dt 0.5`, 4 steps) runs and converges, but Cl shows a visible bias at early times:
  - `t=0.25`: `Cd≈0.2369`, `Cl≈0.00715`, `dp≈0.1055`
  - `t=0.75`: `Cd≈0.5269`, `Cl≈0.00449`, `dp≈0.2810`
- CutFEM (Q2/Q1, `--init zero`, `--dt 0.5`, 4 steps) gives much smaller early-time lift (closer to FeatFlow sign/magnitude):
  - `t=0.25`: `Cd≈0.2467`, `Cl≈-1.0e-4`, `dp≈0.1099`
  - `t=0.75`: `Cd≈0.5776`, `Cl≈ 2.2e-4`, `dp≈0.2999`
- Volume-only (`examples/turek_benchmark_volume_only.py`) was updated to support `--backend cpp`, `--inflow dfg`, `--init stokes|zero`, correct FeatFlow probes, and correct force sign.
  - With `--dt 0.5 --init zero` on the structured O-grid (`--mesh-backend structured --mesh-size 0.02`):
    - `t=0.25`: `Cd≈0.2348`, `Cl≈-7.6e-05`, `dp≈0.1055`
    - `t=0.75`: `Cd≈0.5589`, `Cl≈-5.7e-05`, `dp≈0.2798`
- Conclusion: `dt=0.5` is useful for **fast CutFEM vs volume-only parity checks**, but cannot be used to match FeatFlow quantitatively.

### 2026-01-15 (term-by-term parity: CutFEM vs volume-only, constant inflow)

This isolates the mismatch source between the CutFEM benchmark and the body-fitted (volume-only) benchmark.

**A) Stokes (no mass, no convection)** — matches well
- Volume-only (`mesh_size=0.015`, `--disable-mass --disable-convection`, `dt=0.5`, `theta=0.5`, 1 step):
  - `Cd≈0.225460`, `Cl≈0.002123`, `dp≈0.056553`
- CutFEM (`nx=65, ny=50, refine=1`, `--disable-mass --disable-convection`, `dt=0.5`, `theta=0.5`, 1 step):
  - `Cd≈0.230475`, `Cl≈0.001857`, `dp≈0.056886`
- Conclusion: interface traction + symmetric Nitsche terms are consistent for Stokes; mismatch is not in diffusion/pressure/interface integration.

**B) Transient Stokes (mass on, no convection)** — matches well
- Volume-only (`mesh_size=0.015`, `--disable-convection`, `dt=0.5`, `theta=0.5`, 4 steps):
  - `t=0.25`: `Cd≈0.789472`, `dp≈0.335640`
  - `t=1.75`: `Cd≈0.642790`, `dp≈0.228707`
- CutFEM (`nx=65, ny=50, refine=1`, `--disable-convection`, `dt=0.5`, `theta=0.5`, 4 steps):
  - `t=0.25`: `Cd≈0.814900`, `dp≈0.336974`
  - `t=1.75`: `Cd≈0.635461`, `dp≈0.230213`
- Conclusion: transient mass term is consistent; mismatch is not in the mass matrix / θ-time handling.

**C) Steady Navier–Stokes (mass off, convection on)** — mismatch shrinks with mesh refinement (discretization effect)
- Volume-only (`mesh_size=0.015`, `--disable-mass`, `dt=0.5`, `theta=1.0`, 1 step):
  - `Cd≈2.640952`, `Cl≈-0.013107`, `dp≈2.295545`
- CutFEM (`nx=65, ny=50, refine=1`, `--disable-mass`, `dt=0.5`, `theta=1.0`, 1 step, `beta0=40…200`):
  - `Cd≈2.884–2.890`, `dp≈2.314`
  - Increasing `beta0` reduces slip strongly, but **does not** materially change `Cd` → not a Nitsche-penalty issue.
- Volume-only refinement shows Cd increases toward CutFEM:
  - `mesh_size=0.01`, `--disable-mass`, `--init stokes`: `Cd≈2.799577`, `dp≈2.305528`
- CutFEM local refinement around the interface increases Cd:
  - `refine=2`: `Cd≈2.932322`, `dp≈2.311187`
- Conclusion: the “CutFEM vs volume-only” mismatch in the full NS regime is dominated by **discretization (mesh-quality / boundary-layer resolution)**, not by an obvious bug in cut integration or the Stokes/Nitsche pieces.

---

## Next experiments (short list)

- [ ] H5: Increase quadrature order on cut volumes + interface (`q`) and check if Cd/Cl/dp move toward FeatFlow (under-integration can cause systematic force bias).
- [ ] Compare against the volume-only solver at the same polynomial order and `(dt, theta)` to verify the *Navier–Stokes core* is not the source of the mismatch.
- [ ] Inspect pressure stabilization options: consider switching from value-jump to grad-jump and/or adding time-/h-scaling consistent with CutFEM Stokes inf–sup theory; validate by convergence w.r.t. `gamma_gp_p`.
- [ ] Add more diagnostics: divergence norm on Ω⁺, outlet traction integral, and split forces into viscous vs pressure parts to identify which term drives the Cd/Cl bias.

---

## 2026-01-15 (skew-symmetric convection check)

Hypothesis: using the skew-symmetric (energy-conserving) convection form

`0.5*( dot(grad(u)*w, v) - dot(grad(v)*w, u) )`  (with `w=u` for NS)

would improve Cd/Cl agreement (either vs FeatFlow or vs the body-fitted volume-only solver).

Findings (constant inflow, steady NS surrogate via `--disable-mass --theta 1 --dt 0.5`, cpp backend):

- Volume-only (structured O-grid, `--mesh-size 0.03`, `--init stokes`):
  - standard: `Cd≈2.4376`, `Cl≈-2.21e-02`, `Δp≈2.3215`
  - skew:     `Cd≈2.3223`, `Cl≈-1.17e-02`, `Δp≈2.2662`
  - Skew still **reduces Cd/Δp** (moves *away* from the expected ~3.1 mean Cd for the DFG benchmark).
- CutFEM (background mesh `nx=65,ny=50,refine=1`, `--with-deformation`, `beta0=100`, `gamma_gp=0`, `--init stokes`):
  - skew: `Cd≈2.8840`, `Cl≈-8.50e-03`, `Δp≈2.3055`
  - (From earlier standard runs at similar settings: `Cd≈2.884–2.890`), so skew changes Cd only marginally on CutFEM.

Conclusion:
- Skew convection **does not** improve agreement for this benchmark in the tested regime; it makes the body-fitted solver’s Cd smaller while leaving CutFEM nearly unchanged → parity does not improve.

Backend note:
- Fix required: C++ backend was previously mis-compiling `dot(dot(grad(v_test), u_trial), u_k)` (used in skew Jacobian) by treating a mixed stack as a gradient stack, leading to corrupted Jacobians and factorization failure. This is now fixed and covered by regression tests.

---

## 2026-01-15 (DFG do-nothing outlet + stress convention)

DFG benchmark 2D-2 defines:

- do-nothing outflow: `ν ∂ₙ u - p n = 0` on Γ_out
- stress used for forces: `σ := ν ∇u - p I`

Changes:

- Added the outlet correction term `+⟨ν (∇uᵀ·n), v⟩_{Γ_out}` so that the symmetric interior form `2ν ε(u):ε(v)` produces the DFG do-nothing traction `ν ∂ₙu - p n` on the outlet.
- Updated drag/lift postprocessing to use `σ := ν ∇u - p I` (and updated Babuska–Miller volume functional accordingly).
- Fixed `compile_multi` for `dS` integrals by always including `(0,0)` in the requested derivative set; added regression tests.

Quick validation (volume-only, structured O-grid, coarse, cpp backend):

- Run: `examples/turek_benchmark_volume_only.py` with `--mesh-size 0.05 --dt 0.01 --max-steps 500 --inflow constant --init stokes`.
- Last-cycle window (T≈0.33): `Cl_min≈-1.028`, `Cl_max≈1.003` (close to FeatFlow’s ~±1), with extrema separated by ~`0.18s` (close to the expected ~`0.165s` half-period).
- Cd is still low on this coarse mesh (`Cd_mean≈2.79` vs FeatFlow ~`3.13`), but the **Cl amplitude is now in-family**.
