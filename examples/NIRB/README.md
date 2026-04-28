# NIRB Example 2: DoubleFlap FSI Reproduction

This directory reproduces the paper's Example 2 inside `pycutfem`.
The target is not to reuse the author's reduced-order outputs as the primary
workflow, but to generate the full-order snapshots locally, validate them
against the public Kratos benchmark, and then feed those snapshots into the
general `pycutfem.mor` and `pycutfem.nirb` modules.

The benchmark path is now:

1. generate the full-order coupled FSI data inside `pycutfem`,
2. validate local fluid, solid, interface, and coupled-step gates against the
   Kratos benchmark on the same meshes,
3. launch the long snapshot run in `tmux`,
4. use the exported `coSimData` matrices as the offline NIRB training data.

This README documents the full formulation, the discrete operator, the hidden
subscale handling, the coupling algorithm, the validation gates, the debugging
path, and the current production commands.

## Current validated status

The main Example 2 gates that must hold before the long snapshot run are:

- Rigid-wall fluid gate on the exact Kratos mesh and exact Kratos nodal state:
  - velocity relative L2 error: `0`
  - pressure relative L2 error: `0`
  - interface reaction relative L2 error: `9.125e-4`
  - interface reaction cosine similarity: `0.99999958`
- Solid constitutive gate on the same mesh/state:
  - displacement relative L2 error: `~1e-4`
- Interface transfer gate:
  - exact sign and ordering match after the Kratos fluid-to-structure sign flip
- Coupled first-step gate:
  - displacement relative L2 error: `2.196e-2`
  - interface velocity relative L2 error: `2.200e-2`

Those numbers come from the validated exact fluid path, exact interface
reaction path, Kratos-consistent hyperelastic solid, and Bossak-consistent
mesh/interface velocity.

The long snapshot run should use the same exact path.

## Production commands

One-step exact validation:

```bash
conda run --no-capture-output -n fenicsx python examples/NIRB/run_example2_local.py \
  --output-dir examples/NIRB/artifacts/example2_coupled_step1_re301_exact \
  --mesh-source reference \
  --poly-order 1 \
  --pressure-order 1 \
  --reynolds 301 \
  --dt 0.008 \
  --end-time 0.008 \
  --max-steps 1 \
  --max-coupling-iters 6 \
  --coupling-abs-tol 1e-6 \
  --coupling-rel-tol 1e-6 \
  --load-transfer reaction \
  --force-update iqnils \
  --force-relaxation 0.5 \
  --force-iteration-horizon 50 \
  --force-history 3 \
  --force-regularization 1e-10 \
  --newton-tol 1e-6 \
  --max-newton-iter 8 \
  --bossak-alpha -0.3 \
  --dynamic-tau 1.0 \
  --pressure-gauge 1e-5 \
  --fluid-operator exact \
  --backend cpp \
  --linear-backend petsc \
  --snapshot-mode all \
  --save-vtk \
  --vtk-every 1 \
  --monitor-interface-loads \
  --verbose
```

Long snapshot run in `tmux`:

```bash
bash examples/NIRB/launch_example2_local_tmux.sh
```

Optional environment overrides for the launcher:

```bash
END_TIME=6.008 VTK_EVERY=1 MAX_COUPLING_ITERS=12 \
  bash examples/NIRB/launch_example2_local_tmux.sh my_session_name
```

Kratos comparison utilities:

```bash
python examples/NIRB/debug/run_kratos_example2_reference.py --help
python examples/NIRB/debug/dump_kratos_example2_stage_state.py --help
python examples/NIRB/debug/check_example2_fluid_gate.py --help
python examples/NIRB/debug/check_example2_solid_gate.py --help
python examples/NIRB/debug/check_example2_interface_gate.py --help
```

## Benchmark definition

The benchmark is the double-flap FSI case distributed with the public
DoubleFlap/Kratos reference repository. The production local setup is built by:

- `examples/NIRB/double_flap_reference.py`
- `examples/NIRB/example2_local_setup.py`
- `examples/NIRB/example2_problem.py`
- `examples/NIRB/run_example2_local.py`

The validation reference is the Kratos co-simulation setup:

- `DoubleFlap_fsi_parameters_ROM.json`
- `ProjectParametersCFD.json`
- `ProjectParametersCSM.json`
- `FluidMaterials.json`

Local production settings:

- mesh source: `reference`
- Reynolds number: `301`
- fluid density: from the benchmark JSON
- kinematic viscosity: from `FluidMaterials.json`
- time step: `0.008`
- final time: `6.008`
- Bossak alpha: `-0.3`
- fluid element: exact local Kratos-style `DVMS`
- structure law: Kratos `HyperElasticPlaneStrain2DLaw`
- coupling: strong Gauss-Seidel with IQN-ILS load update

## Unknowns and notation

We solve on partitioned fluid and solid meshes.

Fluid:

- reference fluid domain: `\hat{\Omega}_f`
- ALE mesh displacement: `d_m`
- ALE deformation gradient:
  `F_m = I + \nabla_X d_m`
- ALE Jacobian:
  `J_m = det(F_m)`
- fluid velocity: `u`
- fluid pressure: `p`
- mesh velocity: `w_m`
- convective velocity:
  `c = u - w_m`

Solid:

- reference solid domain: `\hat{\Omega}_s`
- solid displacement: `d_s`
- solid deformation gradient:
  `F_s = I + \nabla_X d_s`
- solid Jacobian:
  `J_s = det(F_s)`

Interface:

- fluid-structure interface: `\Gamma_{fsi}`
- Dirichlet data passed to the fluid: interface displacement / mesh motion
- Neumann data passed to the solid: interface load in the reference frame

Test functions:

- fluid velocity test: `v`
- fluid pressure test: `q`
- solid displacement test: `\eta`
- mesh-extension test: `\psi`

## Strong staggered FSI algorithm

The production coupling is a fixed-point problem on the interface load:

```math
(F \circ S)(f) = f
```

where:

- `S`: solve the quasi-static solid with interface load `f`, return interface
  displacement `d_\Gamma`
- `F`: solve the ALE fluid on the moved mesh induced by `d_\Gamma`, return the
  interface load

Per time step, the local driver performs:

1. choose an initial interface load guess `f_k`
2. solve the solid on `\hat{\Omega}_s`
3. extend the interface displacement into the fluid mesh
4. compute Bossak mesh velocity from the mesh displacement history
5. solve the fluid on `\hat{\Omega}_f`
6. extract the returned interface load
7. compute the load/displacement residual
8. update the next load guess using constant relaxation, Aitken, or IQN-ILS
9. repeat until both displacement and load satisfy
   `abs <= 1e-6 or rel <= 1e-6`

The benchmark-relevant update is IQN-ILS.

## Solid model

The solid is solved in the Lagrangian frame and must not be driven by a raw
Cauchy traction. The structure law in the local path matches Kratos
`HyperElasticPlaneStrain2DLaw`.

### Kinematics

```math
F_s = I + \nabla_X d_s,
\qquad
J_s = det(F_s)
```

### Constitutive law

The first Piola-Kirchhoff stress is

```math
P(F_s)
= \mu_s F_s + (\lambda_s \log J_s - \mu_s) F_s^{-T}
```

with Lamé parameters

```math
\mu_s = \frac{E}{2(1+\nu)},
\qquad
\lambda_s = \frac{E\nu}{(1+\nu)(1-2\nu)}
```

### Weak form

The quasi-static solid residual is

```math
R_s(d_s;\eta)
= \int_{\hat{\Omega}_s} P(F_s) : \nabla_X \eta \, dX
- \int_{\Gamma_{fsi}} f \cdot \eta \, dS
```

with clamp Dirichlet conditions on the fixed support.

### Consistent tangent

The exact variation used in the local code is

```math
\delta P
= \mu_s \nabla_X \delta d_s
+ \lambda_s tr(F_s^{-1}\nabla_X \delta d_s) F_s^{-T}
+ (\mu_s - \lambda_s \log J_s)
  F_s^{-T} (\nabla_X \delta d_s)^T F_s^{-T}
```

and the Jacobian is

```math
K_s(\delta d_s,\eta)
= \int_{\hat{\Omega}_s} \delta P : \nabla_X \eta \, dX
```

## Mesh-motion model

The fluid mesh is moved by a harmonic extension problem.

### Weak form

Let `d_m` be the mesh displacement and `d_\Gamma` the interface displacement
coming from the structure. The mesh extension problem is:

```math
R_m(d_m;\psi)
= \int_{\hat{\Omega}_f} \nabla_X d_m : \nabla_X \psi \, dX
```

with Dirichlet conditions:

- `d_m = d_\Gamma` on `\Gamma_{fsi}`
- `d_m = 0` on inlet, outlet, walls, cylinder

The current local path uses this harmonic extension and reproduces the coupled
benchmark once the mesh velocity is treated with the correct Bossak kinematics.

## Bossak time integration

The benchmark uses Bossak with `alpha = -0.3`.

Define

```math
\gamma = 0.5 - \alpha,
\qquad
\beta = 0.25 (1-\alpha)^2
```

For a displacement-like unknown `d`,

```math
a^{n+1}
= \frac{d^{n+1} - d^n - \Delta t\, v^n - \Delta t^2 (0.5-\beta) a^n}
        {\beta \Delta t^2}
```

```math
v^{n+1}
= v^n + \Delta t \left[(1-\gamma)a^n + \gamma a^{n+1}\right]
```

The Bossak weighted acceleration used in the momentum equation is

```math
a_{n+\alpha}
= (1-\alpha) a^{n+1} + \alpha a^n
```

### Important implementation detail

The local code initially used a backward-difference mesh/interface velocity.
That was wrong for this benchmark.

The final validated path computes:

- fluid acceleration from Bossak
- mesh velocity from Bossak on the mesh-displacement history
- interface velocity from the Bossak-consistent fluid mesh history

This is the fix that removed the last moved-domain mismatch in the coupled
first step.

## Fluid model

The fluid solve is an ALE monolithic Bossak step with a Kratos-style `DVMS`
element.

### ALE operators

The physical gradient and divergence are

```math
\nabla_x u = (\nabla_X u) F_m^{-1}
```

```math
div_x(u) = \frac{cof(F_m) : \nabla_X u}{J_m}
```

The physical Cauchy stress used by the final local path is the Kratos-style
deviatoric Newtonian stress:

```math
\sigma_f(u,p)
= -p I + \mu_f \left(
    \nabla_x u + \nabla_x u^T - \frac{2}{3} div_x(u) I
  \right)
```

The fluid traction in the current configuration is

```math
t_f = \sigma_f n_f
```

and the reference-frame load passed to the structure is its pull-back

```math
G_{N,s} = J_m \sigma_f F_m^{-T} N_s
```

### Continuous ALE weak form

The Galerkin part of the fluid residual on `\hat{\Omega}_f` is

```math
R_f^{gal}(u,p;v,q)
= \int_{\hat{\Omega}_f} \rho_f J_m a_{n+\alpha} \cdot v \, dX
+ \int_{\hat{\Omega}_f} \rho_f J_m \left[(\nabla_x u)c\right] \cdot v \, dX
+ \int_{\hat{\Omega}_f} J_m \sigma_f(u,p) F_m^{-T} : \nabla_X v \, dX
+ \int_{\hat{\Omega}_f} J_m div_x(u) q \, dX
+ \int_{\hat{\Omega}_f} \varepsilon_p p q \, dX
```

where

```math
c = u - w_m
```

and `w_m` is the Bossak-consistent mesh velocity.

### Large scales and small scales

The DVMS interpretation is

```math
u = u_h + u',
\qquad
p = p_h + p'
```

where:

- `(u_h,p_h)` are the resolved finite-element large scales
- `(u',p')` are the local unresolved small scales

The unresolved velocity subscale is modeled from the local momentum residual.
In the benchmark this is not only a static algebraic term; it has transient
history through the old subscale velocity and the old mass residual.

### Stabilization parameters

The Kratos-style parameters used in the local path are

```math
\tau_1
= \frac{dynamic\_tau}
        {c_1 \mu_f / h^2 + \rho_f\left(m_{am} + c_2 |c_{full}|/h\right)}
```

```math
\tau_2
= \mu_f + \rho_f c_2 |c_{full}| h / c_1
```

```math
\tau_p
= \rho_f h^2 / (c_1 \Delta t)
```

with:

- `c_1 = 8`
- `c_2 = 2`
- `dynamic_tau = 1.0`
- `h` = Kratos element size
- `c_full = u - w_m + u'_{pred}`

### Why the tau scaling matters

The local solve did not match Kratos until:

- the same `h` scaling was used,
- the same Bossak `m_am` contribution entered `tau_1`,
- the same predicted subscale entered `c_full`,
- and the same old mass residual was used in the transient stabilization terms.

## Discrete Kratos DVMS operator

Matching the continuous weak form was not enough.
The benchmark matches only when the local code reproduces the *discrete*
Kratos operator.

Kratos assembles the element operator in three parts:

1. `AddVelocitySystem`
2. `AddMassLHS`
3. `AddMassStabilization`

and uses the Bossak scheme-level combination:

```math
LHS = D + m_{am} M
```

```math
RHS = r_{vel} - M a_{n+\alpha}
```

where:

- `D` is the discrete velocity contribution
- `M` is the discrete mass contribution
- `m_am = (1-\alpha) / (\gamma \Delta t)`

### Local block structure

Using the nodal convection operator

```math
AGradN_i = \rho_f (c_{full} \cdot \nabla N_i)
```

the dominant discrete blocks are:

```math
K_{ij}^{uu}
= w \left[
N_i AGradN_j
+ \left(AGradN_i - \rho_f N_i/\Delta t\right)\tau_1 AGradN_j
\right] I
+ w (\tau_2+\tau_p) (\nabla N_i \otimes \nabla N_j)
```

```math
K_{i,j,d}^{up}
= w \left[
\tau_1 AGradN_i \partial_d N_j
- \tau_1 (\rho_f N_i/\Delta t)\partial_d N_j
- (\partial_d N_i) N_j
\right]
```

```math
K_{i,j,d}^{pu}
= w \left[
\tau_1 AGradN_j \partial_d N_i
+ (\partial_d N_i) N_j
\right]
```

```math
K_{ij}^{pp}
= w \tau_1 (\nabla N_i \cdot \nabla N_j)
```

Mass blocks:

```math
M_{ij}^{uu} = w \rho_f N_i N_j I
```

```math
M_{ij}^{stab,uu}
= w \tau_1 \rho_f \left(AGradN_i - N_i/\Delta t\right) N_j I
```

```math
M_{i,j,d}^{stab,pu}
= w \tau_1 \rho_f (\partial_d N_i) N_j
```

### DVMS history variables

The exact local path carries explicit integration-point history:

- old subscale velocity
- predicted subscale velocity
- old mass residual
- momentum projection
- mass projection

The old terms enter the residual as explicit forcing, not as a symbolic
`u_prev` substitution only.

## Hidden state and the consistent condensed Jacobian

The small scales are local hidden variables.
The clean exact path is:

```math
R(U,z) = 0,
\qquad
G(U,z) = 0
```

where:

- `U = [u,p]`
- `z` = local DVMS hidden state

After eliminating `z = z*(U)`, the condensed residual is

```math
\hat{R}(U) = R(U, z^*(U))
```

and the exact consistent tangent is

```math
\frac{d\hat{R}}{dU}
= R_U - R_z G_z^{-1} G_U
```

In block form:

```math
\hat{K} = K_{uu} - K_{uz} K_{zz}^{-1} K_{zu}
```

```math
\hat{r} = r_u - K_{uz} K_{zz}^{-1} r_z
```

This is the formulation used by the fused exact local operator.
The important engineering conclusion from the debugging was:

- the condensed correction must be built inside one fused local assembly path
- the production solve must not assemble one base operator and then perform a
  second full-mesh correction pass

That was the main performance fix.

## Interface transfer

### Dirichlet data sent to the fluid

The fluid does not receive a backward-difference velocity from the solid.
The validated path is:

1. solve the solid for interface displacement
2. extend that displacement through the fluid mesh
3. compute the fluid mesh velocity from the mesh displacement history with
   Bossak
4. apply that mesh velocity on the moving fluid interface

This is what finally matched the Kratos moved-domain stages.

### Neumann data sent to the structure

The structure must not be driven by a raw Cauchy traction sampled on the fluid
boundary ordering.

The final local path:

1. computes the fluid-side reaction on the constrained interface rows from the
   exact discrete fluid system
2. applies the Kratos fluid-to-structure sign convention
3. resamples onto the solid interface ordering
4. optionally also exports the pulled-back reference traction

The exact Kratos relation on the constrained interface rows is:

```math
REACTION = -b_c
```

after Dirichlet elimination on zero constrained increments.

## Why the debugging took several stages

The final result required closing several distinct gaps.

### 1. UFL and backend support

The exact ALE pressure-gradient terms and the exact Jacobian contractions had to
assemble identically in Python, JIT, and C++.
That was checked term-by-term in:

- `examples/debug/comparison_with_fenics.py`

### 2. Fluid gate

The first rigid-wall fluid mismatch was traced to two non-physics issues:

- truncated VTK values from the public benchmark were not precise enough for
  the reaction gate
- the exact fluid production path was adding an extra inhomogeneous Dirichlet
  lift during the Newton solve

Once that was fixed, the rigid-wall fluid gate matched.

### 3. Solid gate

The solid mismatch was removed by replacing the old local Neo-Hookean variant
with the exact Kratos plane-strain hyperelastic law and tangent.

### 4. Interface gate

The interface mismatch required:

- the Kratos sign flip on fluid-to-structure transfer
- solid-interface ordering instead of assuming fluid and solid interface node
  orders already match

### 5. Coupled moved-domain gate

The last major mismatch was not the fluid large-scale operator.
It was the mesh/interface velocity.

The local path originally used backward-difference/BDF-like mesh velocity.
Kratos uses Bossak mesh velocity in the ALE mesh solver.
Once the local path used the same Bossak kinematics for:

- `MESH_VELOCITY`
- interface moving-wall velocity
- exact stage history

the coupled first-step gate collapsed from large errors to about `2.2%`.

## Final validated local architecture

The production local Example 2 path now consists of:

- reference fluid mesh
- reference solid mesh
- exact Kratos-style DVMS fluid operator
- Bossak mesh velocity
- Kratos hyperelastic plane-strain solid
- exact discrete reaction transfer
- strong Gauss-Seidel outer coupling
- IQN-ILS load update
- `cpp` assembly backend
- PETSc linear solves

Key implementation files:

- `examples/NIRB/run_example2_local.py`
- `examples/NIRB/dvms/symbolics.py`
- `examples/NIRB/dvms/update.py`
- `examples/NIRB/dvms/runtime_operator.py`
- `examples/NIRB/dvms/local_operator.py`
- `examples/NIRB/example2_problem.py`
- `examples/NIRB/example2_local_setup.py`
- `examples/NIRB/double_flap_reference.py`

## Output layout

Every run writes:

- `summary.json`
- `timeseries.csv`
- `snapshot_metadata.csv`
- `coSimData/`

The `coSimData` folder contains:

- `disp_data.npy`
- `load_data.npy`
- `load_guess_data.npy`
- `load_return_data.npy`
- `interface_disp_data.npy`
- `interface_velocity_data.npy`
- `interface_traction_data.npy`
- `load_reaction_data.npy` and `load_stress_data.npy` when monitoring is on
- `iters.npy`
- `fluid_time.npy`
- `structure_time.npy`
- `increment_time.npy`
- `total_solving_time.npy`

When `--save-vtk` is enabled, the run also writes:

- `vtk_data/vtk_output_fsi_cfd/FluidParts_FluidPart_0_XXXX.vtu`
- `vtk_data/vtk_output_fsi_csm/Structure_0_XXXX.vtu`
- `vtk_manifest.csv`

Fluid VTK fields:

- `VELOCITY`
- `PRESSURE`
- `MESH_DISPLACEMENT`
- `MESH_VELOCITY`
- `ACCELERATION`

Solid VTK fields:

- `DISPLACEMENT`
- `POINT_LOAD`

## Snapshot-matrix job

The long snapshot run is the production full-order training job.
It should use:

- `--mesh-source reference`
- `--fluid-operator exact`
- `--load-transfer reaction`
- `--backend cpp`
- `--linear-backend petsc`
- `--snapshot-mode all`
- `--save-vtk`

That run writes:

- the full per-subiteration FSI snapshot matrices in `coSimData`
- the VTK files for later visual comparison
- the timing files needed for MOR/NIRB performance studies

## If you want to reimplement this from scratch

The minimal faithful recipe is:

1. Recreate the reference fluid and solid meshes from the benchmark.
2. Use the exact same material data, Reynolds number, time step, and Bossak
   alpha.
3. Solve the solid in the reference frame with the Kratos plane-strain
   hyperelastic law and exact tangent.
4. Extend interface displacement into the fluid mesh with a harmonic extension.
5. Compute `MESH_VELOCITY` from the mesh displacement history with Bossak,
   not backward difference.
6. Build the ALE fluid operator with the same:
   - `DVMS` blocks
   - mass/stabilization split
   - subscale history
   - `tau` scaling
   - constrained-row reaction extraction
7. Couple fluid and solid with strong Gauss-Seidel on the interface load.
8. Use IQN-ILS to update the interface load guess.
9. Validate in this order:
   - term-level
   - rigid-wall fluid
   - solid
   - interface
   - coupled first step
10. Only then launch the long snapshot run.

## Reference-only utilities

The old reference scripts still exist:

```bash
conda run --no-capture-output -n fenicsx python examples/NIRB/run_example1.py
conda run --no-capture-output -n fenicsx python examples/NIRB/run_example2.py
```

They are not the intended primary workflow.
The intended primary workflow is the local full-order run plus Kratos-backed
validation.
