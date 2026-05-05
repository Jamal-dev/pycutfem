# Sanitized Partitioned FSI Operators

This note documents the reusable operators exposed by:

- `examples/utils/fsi/fluid_solver.py`
- `examples/utils/fsi/structure_solver.py`

The implementations wrap the sanitized NIRB/DoubleFlap operators that were matched against Kratos. They are example-level utilities, not a stable `pycutfem` core API.

## Fluid Model

The fluid solve is an ALE incompressible Navier-Stokes problem with Kratos-style dynamic VMS stabilization and Bossak time integration. Unknowns are velocity `u`, pressure `p`, and the prescribed mesh displacement `d_m` used to define the ALE map.

Reference-to-current kinematics:

```text
F = I + Grad(d_m)
J = det(F)
grad_x(u) = Grad(u) F^{-1}
div_x(u) = cof(F) : Grad(u) / J
w_m = mesh velocity
```

The strong residual matched by the stabilized local operator is:

```text
rho (a_B(u) + ((u - w_m + u_s) . grad_x) u) - div_x sigma(u,p) = 0
div_x u = 0
```

where `u_s` is the predicted dynamic subscale and:

```text
sigma(u,p) = -p I + 2 mu (epsilon(u) - (1/3) div_x(u) I)
epsilon(u) = 0.5 (grad_x(u) + grad_x(u)^T)
```

The weak residual for tests `(v,q)` contains:

```text
int_Omega J rho a_B(u) . v dX
+ int_Omega J rho ((grad_x u) (u - w_m + u_s)) . v dX
+ int_Omega (J sigma F^{-T}) : Grad(v) dX
+ int_Omega (cof(F) : Grad(u)) q dX
+ int_Omega pressure_gauge p q dX
+ DVMS stabilization terms
```

The DVMS parameters are the Kratos-matched element parameters:

```text
tau_1 = dynamic_tau / (8 mu / h^2 + rho (1/dt + 2 |u - w_m + u_s| / h))
tau_2 = mu + rho |u - w_m + u_s| h / 4
tau_p = rho h^2 / (8 dt)
```

The stabilization terms include the dynamic residual, static convective and pressure residuals, old subscale velocity, OSS momentum projection, mass projection, and old mass residual. In the exact path, `SanitizedDVMSFluidSolver.exact_placeholder_forms()` supplies a structurally nonzero placeholder form while `FluidDVMSCondensedLocalSystemOperator` injects the Kratos-matched element matrix/vector blocks. The runtime operator sequence is:

```text
1. Bossak acceleration operator: refresh a_k from accepted u_prev/a_prev.
2. DVMS predictor operator: update predicted_subscale_velocity at quadrature points.
3. Condensed local system operator: assemble the exact local velocity-pressure block.
```

The local block uses the grouped element layout `[ux, uy, p]`, condenses the hidden subscale contribution in the same convention used by Kratos, and scatters the resulting local system into the active Newton system.

## Structure Model

The structure solve is total-Lagrangian plane-strain hyperelasticity. Unknown displacement is `d`; test displacement is `w`.

Kinematics and constitutive law:

```text
F = I + Grad(d)
J = det(F)
P(F) = mu F + (lambda log(J) - mu) F^{-T}
```

The material tangent used in the Newton Jacobian is:

```text
dP(F; Grad(dd)) =
    mu Grad(dd)
  + lambda tr(F^{-1} Grad(dd)) F^{-T}
  + (mu - lambda log(J)) F^{-T} Grad(dd)^T F^{-T}
```

The strong form is:

```text
-Div P(F) = 0                    in Omega_s
d = 0                            on Gamma_clamp
P(F) N = t                       on Gamma_interface
```

The weak residual and Jacobian are:

```text
R_s(d; w) = int_Omega_s P(F) : Grad(w) dX
            - int_Gamma_interface t . w dS

J_s(d; dd, w) = int_Omega_s dP(F; Grad(dd)) : Grad(w) dX
```

For partitioned FSI coupling, `SanitizedHyperelasticStructureSolver` supports both analytic interface traction and a nodal point-load path. The point-load path builds a full structural residual vector on the interface displacement DOFs and injects it through `_ReducedResidualShiftOperator`, matching the sanitized NIRB coupling path.

## Reuse Contract

Use these utilities when an example wants the Kratos-matched partitioned fluid or solid operators through a small solver-facing API instead of importing the full NIRB driver at each call site. Benchmark-specific mesh construction, coupling acceleration, checkpointing, and VTK output should stay in the example driver; only the reusable operator setup belongs here.
