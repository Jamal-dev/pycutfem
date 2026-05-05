# FSI benchmark investigation log

## 2026-03-08: Jonas-inspired exact shear benchmark (publication candidate)

### Final Benchmark 5 design
The publication Benchmark 5 is now the Jonas-inspired exact shear benchmark:

- exact driver: `examples/biofilms/benchmarks/FSI/paper1_benchmark5_jonas_shear.py`
- exact benchmark definition: `examples/utils/biofilm/benchmark5_jonas_shear_exact.py`

The final benchmark is tailored to the reduced Paper 1 one-domain variables
`(v,p,vS,u,alpha,mu_alpha)` and keeps the conservative Cahn--Hilliard interface active.
Instead of reusing the sharp Jonas annulus directly, the benchmark introduces a
benchmark-local **diffuse tangential traction transfer**
localized by the conserved interface weight `omega_Gamma = d_y alpha`.
That gives an exact verification problem for the actual reduced Paper 1 model.

### Frozen production result
Run tag:

- `benchmark5_jonas_publish_ready_20260308`

Finest production mesh (`n_x = 32`):

- `||e_v||_{L^2} = 7.630e-04`
- `||e_p||_{L^2} = 7.864e-03`
- `||e_u||_{L^2} = 1.410e-03`
- `||e_alpha||_{L^2} = 1.549e-03`
- `||e_mu_alpha||_{L^2} = 1.368e-04`
- `|e_{u,Gamma}| = 1.421e-03`

Production rates from `16 -> 32`:

- `EOC(v_L2) = 1.692`
- `EOC(p_L2) = 1.727`
- `EOC(vS_L2) = 1.948`
- `EOC(u_L2) = 1.948`
- `EOC(alpha_L2) = 1.981`
- `EOC(mu_alpha_L2) = 1.962`
- `EOC(v_H1) = 1.932`
- `EOC(u_H1) = 1.927`

### Assessment
This is the first structurally consistent Benchmark 5 variant.
It verifies:

- tangential traction transfer,
- reference-map deformation response,
- and the conserved Cahn--Hilliard interface

inside the reduced one-domain formulation itself.
This benchmark is strong enough to serve as the Benchmark 5 manuscript block for Paper 1.

## 2026-03-08: Jonas annulus impermeable-limit prototype

### Goal
Test whether the reduced Paper 1 one-domain mechanics block can recover the
steady analytic annulus benchmark from Jonas--Heuveline (2016):
- Taylor--Couette fluid annulus with inner angular velocity,
- surrounding elastic ring,
- exact steady fluid velocity / pressure and exact solid tangential displacement.

### Historical prototype files
The annulus prototype was evaluated locally during development and then retired.
It is kept in this log only as a failure analysis record.

### What was tested
- annulus mesh generated with gmsh and loaded into pycutfem,
- reduced one-domain unknown set `(v,p,vS,u,phi,alpha,mu_alpha,S)`,
- `phi`, `alpha`, `mu_alpha`, and `S` frozen so only the mechanics block is active,
- transient relaxation from rest to a steady state under the analytic inner-wall
  tangential velocity,
- comparison against the Jonas steady fields on fluid and solid core regions.

### Main finding
The prototype converges to a steady state numerically, but it does **not**
recover the Jonas solid displacement field. The mismatch is structural, not a
parameter-tuning issue.

Representative run:
```bash
conda run --no-capture-output -n fenicsx python -u \
  examples/biofilms/benchmarks/FSI/paper1_benchmark5_jonas_annulus.py \
  --outdir /tmp/benchmark5_transient_probe2 \
  --h-list 0.12 --kappa-inv-list 1e4,1e5 \
  --dt 0.1 --final-time 1.0 --steady-tol 1e-5 --png-dpi 100 --quiet
```

Observed steady states on `h_max ≈ 0.118`:
- `kappa_inv = 1e4`: `fluid_v_rel_l2 = 5.38e-01`, `solid_u_rel_l2 = 1.00e+00`,
  `u_probe_final = 1.10e-05`, `u_probe_exact = -1.63e-02`
- `kappa_inv = 1e5`: `fluid_v_rel_l2 = 7.41e-01`, `solid_u_rel_l2 = 1.00e+00`,
  `u_probe_final = 1.24e-06`, `u_probe_exact = -1.63e-02`

So the solver reaches steady state quickly, leakage decreases with larger
`kappa_inv`, but the solid tangential displacement remains essentially zero.

### Interpretation
This is consistent with the reduced Paper 1 mechanics model.
The Jonas annulus deformation is driven by **tangential interfacial viscous
traction**.
In the current reduced one-domain mechanics block, the skeleton is driven by:
- pressure coupling through `B(alpha,phi) grad p`,
- Darcy/Brinkman drag through `beta(alpha,phi) (v - vS)`,
- elastic stress and the Eulerian kinematic update.

The model does **not** include a sharp-interface tangential traction transfer of
the Jonas type. Therefore the Jonas `u_theta` field is not the correct target
for this reduced model, even though the annulus fluid solve itself is well posed.

### Consequence for Benchmark 5
Do **not** wire the Jonas annulus benchmark into the paper-ready suite as a
publication benchmark in its current form.

There are two viable paths:
1. Change the model so the impermeable limit carries the missing tangential
   traction transfer, then rerun Jonas.
2. Replace Benchmark 5 with an impermeable-limit comparison whose deformation is
   driven by the couplings that are actually present in the reduced model
   (pressure / drag / reference-map kinematics), not by classical tangential
   FSI traction continuity.

# Turek–Hron FSI-2 (one-domain) investigation log

## Goal
Validate the **FSI coupling** in the one-domain diffuse-interface model (`examples/utils/biofilm/one_domain.py`) against the canonical **Turek–Hron FSI-2** benchmark:
- drag coefficient `C_D(t)`
- lift coefficient `C_L(t)`
- tip displacement of point A at the beam end

We run the **biofilm one-domain** model in a “pure FSI” configuration:
- `mu_max = k_g = k_d = k_det = 0` (no growth / detachment)
- `damage_* = 0` (no damage evolution)
- use finite-strain solid model (`solid_model=svk` by default)
- keep `phi` frozen and tie it to `alpha` (see below)
- evolve `alpha` via conservative advection + **Cahn–Hilliard** regularization (mass conserving)

## Setup (geometry + params)
Geometry constants are in `examples/biofilms/benchmarks/FSI/turek_hron_beam_one_domain.py`:
- channel: `L=2.2`, `H=0.41`
- cylinder: center `(0.2,0.2)`, `R=0.05`
- beam: length `0.35`, height `0.02`
- tip reference (point A): `(0.6,0.2)`

FSI-2 reference parameter set (matches the CutFEM driver in this repo):
- fluid: `rho_f=1000`, `mu_f=1`, `U_mean=1` (parabolic inflow, ramped over 2s)
- solid: `nu_s=0.4`, `E_s≈1.4e6` (so `mu_s≈5e5`), `rho_s=1e4`

## Mesh (structured channel with a hole)
We use the structured **O-grid** channel mesh with the cylinder removed from the mesh:
- mesh builder: `examples/utils/fsi/turek_fsi2.py:build_structured_channel_mesh`
- boundary tags include: `inlet`, `outlet`, `walls`, `cylinder`

## “Solidification” via frozen phi
We represent the **beam** by a diffuse indicator `alpha(x,t)`.

The cylinder is treated as a **rigid obstacle** by using a *channel-with-a-hole*
mesh and applying no-slip on the cylinder boundary (no solid mechanics inside it).

To emulate an (almost) impermeable solid while staying inside the one-domain
poroelastic formulation, we **freeze** `phi` and enforce

`phi(x,t) = 1 - (1-phi_solid) * alpha(x,t)`.

This yields:
- in fluid (`alpha≈0`): `phi≈1`
- in solid (`alpha≈1`): `phi≈phi_solid` (small)

Important: the skeleton inertia term uses `B = alpha*(1-phi)`, so with `phi` tied
to `alpha` we set

`rho_s0_tilde = rho_s / (1 - phi_solid)`

so that `rho_S = rho_s0_tilde * alpha*(1-phi) ≈ rho_s` in the solid.

## Beam root attachment (Dirichlet `alpha=1`)
To prevent numerical “detachment” of the beam from the rigid cylinder, we enforce
`alpha=1` on the **beam-root arc** on the cylinder boundary (right half, within
the beam-height band). In the driver this is implemented by tagging alpha DOFs
into `dh.dof_tags["beam_root_alpha"]` and applying a Dirichlet BC to `alpha`.

## Active DOF rectangle for (u, vS)
We follow the “big rectangle” approach:
- choose a fixed rectangle that encloses all expected beam motion
- mark (u,vS) DOFs outside that rectangle **inactive**
- keep the standard extension penalties inside the rectangle to stabilize (u,vS)
  in regions where `alpha` is small (free fluid)

This avoids dynamically changing active sets and removes the need to project
displacements onto inactive DOFs.

## Local refinement around the beam (1 hanging node)
The beam is thin (`H_beam=0.02`), so we use a **single** local refinement level
around the initial beam bounding box to increase resolution near the structure.

In `turek_hron_beam_one_domain.py` this is enabled by default:
- `--refine-beam` / `--no-refine-beam`
- `--refine-beam-pad <float>`

To keep the mesh in the “one hanging node per edge” regime, the benchmark driver
currently assumes `--refine-levels 1`.

## Drag/lift computation
The benchmark driver reports forces as a **sum** of:
- cylinder force from a traction integral on the cylinder boundary (sharp, since the cylinder is a mesh hole),
- beam force from the diffuse Brinkman penalization reaction.

### Beam contribution (diffuse, Brinkman reaction)
For the one-domain diffuse formulation we do not have a sharp interface measure on the beam.
We compute the hydrodynamic force on the beam using the Brinkman penalization identity:

`F ≈ ∫ beta * (v - vS) dx`

### Cylinder contribution (sharp, traction)
The cylinder force uses the fluid Cauchy stress
`σ_f = -p I + μ_f (∇v + ∇v^T)` and

`F_cyl = -∫ σ_f n ds`

where `n` is the outward normal of the *fluid* domain (points into the hole).

### Coefficients
We then form
`C_D = 2*F_D/(rho_f*U_mean^2*D)` and `C_L = 2*F_L/(rho_f*U_mean^2*D)`, `D=2R`.

## Outlet / pressure gauge
We use a **do-nothing outlet** (natural traction condition) and avoid enforcing
`p=0` on the whole outlet boundary (which can over-constrain the pressure field
and artificially steepen the pressure gradient in the channel).

To remove the constant-pressure nullspace, we **pin a single pressure DOF**
near the outlet mid-height (`--pressure-bc point`, default).

You can disable gauge fixing entirely with `--pressure-bc none` (pure do-nothing outlet).
If PETSc reports a singular/indefinite solve in that mode, switch back to `--pressure-bc point`.

## Nonlinear solver diagnostics (SNES vs PDAS)
The benchmark driver supports two nonlinear solvers:
- `--newton-solver snes` (default): PETSc SNES Newton with line-search. Robust and has many PETSc tuning knobs.
- `--newton-solver pdas`: internal PDAS / semismooth Newton for box-constraints. Prints active-set sizes (`nA/nI`) and can be useful for debugging constraint behavior.

Why Newton iterations may “not show”:
- SNES is quiet by default. Enable iteration output with `--petsc-monitor` (or `PYCUTFEM_PETSC_MONITOR=1`), which adds PETSc monitors like `snes_monitor` / `snes_converged_reason`.
- PDAS prints iterations by design (lines starting with `VI Newton ...`).

Extra debug knobs (optional):
- `PYCUTFEM_NEWTON_TRACE_RES_FIELDS=1` prints per-field residual norms (helps identify which block dominates).
- `PYCUTFEM_RESIDUAL_TRACE=1` prints the worst residual entry (and optionally coordinates).

## Tip tracking
The Eulerian reference-map kinematics in the one-domain model uses
`X(x,t) = x - u(x,t)` as the reference coordinate of the material point currently
at `x`.

For the Turek–Hron tip reference point `X_ref = (0.6, 0.2)`, the current tip
position is obtained from the implicit relation

`x_tip = X_ref + u(x_tip,t)`.

In `turek_hron_beam_one_domain.py` this is solved by a few fixed-point iterations
using cheap inverse-distance-weighting (IDW) interpolation of `u` from Q2 nodes.

## Penalization / drag scaling (`kappa_inv`, `phi_solid`)
The Brinkman drag in the one-domain formulation is (scalar path):

`beta = alpha * mu_f * phi^2 * kappa_inv`.

Inside the solid (`alpha≈1`, `phi≈phi_solid`) the effective penalty magnitude is

`beta_solid ≈ mu_f * (phi_solid^2) * kappa_inv`.

For Brinkman penalization to approximate an impermeable solid, `beta_solid`
must be large enough compared to the viscous and inertial scales; a common
practical guideline is to scale the penalty like

`beta_solid ~ mu_f / h^2`

so that the penalization length scale is comparable to the mesh size `h`.

This suggests the rough rule of thumb:

`kappa_inv ~ 1 / (phi_solid^2 * h^2)`.

Trade-off:
- larger `kappa_inv` → better no-slip enforcement / more accurate forces, but stiffer nonlinear solve
- larger `phi_solid` → stronger `beta` for fixed `kappa_inv`, but reduces solid volume fraction `B = alpha*(1-phi)`

## Conservative alpha + Cahn–Hilliard regularization
We use:
- `alpha_advection_form = conservative` (mass-conserving advection),
- `alpha_ch_*` enabled to keep a well-posed diffuse interface and avoid area loss.

Tuning knobs:
- `alpha_ch_eps` should be a few mesh sizes (so the interface is resolved).
- `alpha_ch_M` (mobility) controls how fast the interface regularizes; too large
  can smear the interface or damp motion; too small can make the system stiff.
- `alpha_ch_gamma` controls the regularization strength.

## Outputs
`out_dir/timeseries.csv` columns:
- `t`: time
- `FD`, `FL`: **total** force components `FD = FD_cyl + FD_beam`, `FL = FL_cyl + FL_beam`
- `dp`: pressure drop `p(A)-p(B)` at fixed probes `A=(0.15,0.2)`, `B=(0.25,0.2)`
- `CD`, `CL`: coefficients using `D=2R`, `U_mean`
- `FD_cyl`, `FL_cyl`, `CD_cyl`, `CL_cyl`: cylinder traction force and its coefficients
- `FD_beam`, `FL_beam`, `CD_beam`, `CL_beam`: beam Brinkman-reaction force and its coefficients
- `tip_x`, `tip_y`: tracked current tip position
- `tip_dx`, `tip_dy`: tip displacement relative to `(0.6, 0.2)`

Notes:
- By construction, `FD = FD_cyl + FD_beam` and similarly for lift.
- `alpha` box constraints are **enabled by default** in the benchmark driver; disable with `--no-alpha-box-constraints`.

## Runs / findings
Add runs here as you iterate:

### Run 0 (smoke: CLI + assembly)
Purpose:
- verify the benchmark driver imports cleanly
- verify the fixed active-DOF rectangle is applied (inactive DOFs dropped)
- verify the one-domain forms assemble with CH enabled

Notes:
- use `--vtk-every 0` for speed
- python backend is extremely verbose during assembly (expected); prefer `--backend cpp` for real runs

### Run 1 (baseline scaffold)
Command:
```bash
conda run --no-capture-output -n fenicsx python -u examples/biofilms/benchmarks/FSI/turek_hron_beam_one_domain.py \
  --backend cpp --mesh-size 0.01 --dt 0.005 --t-final 1.0 --vtk-every 0
```

Notes:
- TODO: fill in stability observations (Newton convergence, dt sensitivity).
- TODO: compare `C_D/C_L` and tip displacement vs reference.

### Run 1a (cpp+snes, coarse, single step)
Command:
```bash
conda run --no-capture-output -n fenicsx python -u examples/biofilms/benchmarks/FSI/turek_hron_beam_one_domain.py \
  --backend cpp --newton-solver snes --mesh-size 0.01 --q 6 --dt 0.01 --t-final 0.01 --vtk-every 1 \
  --kappa-inv 1e6 --phi-solid 0.2 --eps 0.02 --alpha-ch-eps 0.02 --alpha-ch-M 1e-6 --alpha-ch-gamma 1e-2
```

Observed:
- converged in 2 Newton iterations for step 1
- first-step forces are small (still in ramp-up); tip displacement near zero (as expected at `t=0.01`)

### Run 2a (hole mesh + cylinder traction, cpp+snes, single step)
Command:
```bash
conda run --no-capture-output -n fenicsx python -u examples/biofilms/benchmarks/FSI/turek_hron_beam_one_domain.py \
  --backend cpp --newton-solver snes --mesh-size 0.02 --q 6 --dt 0.01 --t-final 0.01 --vtk-every 0 \
  --kappa-inv 1e6 --phi-solid 0.2 --eps 0.02 --alpha-ch-eps 0.02 --alpha-ch-M 1e-6 --alpha-ch-gamma 1e-2
```

Observed:
- converged in 2 Newton iterations for step 1
- note: first call to the traction postprocessing (`dS` on `cylinder`) triggers a one-time C++ kernel compilation
- at `t=0.01`: `CD≈4.41e-3`, `CL≈-4.38e-6`, tip displacement ≈ `(2.2e-7, -6.0e-10)`

### Run 2b (cpp+snes, PETSc monitor output, single step)
Command:
```bash
conda run --no-capture-output -n fenicsx python -u examples/biofilms/benchmarks/FSI/turek_hron_beam_one_domain.py \
  --backend cpp --mesh-size 0.05 --no-refine-beam --no-refine-obstacle --vtk-every 0 \
  --t-final 0.005 --dt 0.005 --petsc-monitor
```

Observed:
- PETSc prints SNES/KSP iteration info (e.g. `SNES Function norm ...`), which is otherwise silent by default.
- at `t=0.005`: `dp≈2.1e-1` (order-one / not spuriously huge for this very short run)

Variant:
- `--pressure-bc none` also converged for this single-step smoke run and produced essentially the same `dp`/forces (as expected: only the absolute pressure gauge changes).
