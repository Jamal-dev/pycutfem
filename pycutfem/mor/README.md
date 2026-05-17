# pycutfem.mor

`pycutfem.mor` contains the reduced-order modeling tools used by pycutfem for
offline basis construction, non-affine decomposition, native C++ online solves,
GNAT/DEIM/QDEIM hyper-reduction, mixed-field stability gates, adjoints, DWR
certification, adaptive enrichment, and readiness checks for harder models.

The module is problem-generic.  It does not assume a specific PDE.  A problem
such as Navier-Stokes, FSI, poromechanics, or a biofilm multi-constituent model
provides UFL residuals, tangents, field layouts, snapshots, boundary conditions,
and quantities of interest.  The MOR layer provides the algebra, native runtime
contracts, certificates, and artifact schema.

## Documentation Plan

This README is organized as a practical onboarding document:

1. Mathematical model and notation.
2. Recommended workflow from snapshots to certified native online solves.
3. Algorithms for POD, mixed-field bases, DEIM/QDEIM, GNAT, Gauss-Newton,
   branch globalization, DWR, and adaptive enrichment.
4. Code examples for the common paths.
5. Public API inventory by module.
6. Performance and certification rules.

## Core Mathematical Model

The full-order nonlinear problem is written algebraically as

$$
R(x_n, x_{n-1}; \mu) = 0,
$$

where `x_n` is the full finite-element state at time step `n`,
`x_{n-1}` is the previous state, and `mu` denotes parameters.  A reduced affine
decoder uses

$$
x(q) = x_0 + Vq,
$$

where `x_0` is an offset or lifting state, `V` is the trial basis, and `q` is
the reduced coefficient vector.

For a nonlinear online solve, the residual and tangent projected to the reduced
space can be one of several targets:

True Galerkin:

$$
g(q) = V^T R(x_0 + Vq),
\qquad
G(q) = V^T J(x_0 + Vq)V.
$$

Full-row LSPG:

$$
\min_{\delta q}
\left\| R(x_0 + Vq) + J(x_0 + Vq)V\delta q \right\|_2^2.
$$

Sampled LSPG/GNAT:

$$
\min_{\delta q}
\left\| W^{1/2} S
\left(R(x_0 + Vq) + J(x_0 + Vq)V\delta q \right)
\right\|_2^2,
$$

where `S` selects sampled residual rows and `W` contains row weights.

Reference-regularized branch selection adds a reduced reference state `q_ref`:

$$
\min_{\delta q}
\left\| r + A\delta q \right\|_2^2
+ \lambda \left\| V(q + \delta q - q_{\mathrm{ref}}) \right\|_2^2.
$$

Decoded physical bounds are represented in full-state rows:

$$
\ell \le C(x_0 + Vq) \le u.
$$

The reduced solver can enforce these bounds using PDAS or IPM-style native
Gauss-Newton drivers.

For a scalar quantity of interest `Q`, the fully discrete adjoint uses

$$
J_n(x_n)^T z_n =
\frac{\partial Q}{\partial x_n}
- \left(\frac{\partial R_{n+1}}{\partial x_n}\right)^T z_{n+1}.
$$

The DWR estimate is

$$
\eta_Q =
\sum_n z_n^T R_n(x_r),
$$

up to the sign convention configured for the residual.

## Recommended Workflow

Use this sequence for a production nonlinear MOR:

1. Run a stable FOM trajectory and collect snapshots.
2. Remove lift/Dirichlet data from snapshots.
3. Gauge-correct pressure-like fields.
4. Fit fieldwise POD bases and any lift/supremizer enrichments.
5. Certify mixed stability and per-field projection errors.
6. Fit collateral bases for non-affine residual terms.
7. Select DEIM/QDEIM rows or build block-balanced GNAT sampling.
8. Certify sampled/full residual norm-equivalence.
9. Create a native reduced artifact with kernels, target, bounds, state updates,
   reference policy, and adjoint/DWR metadata.
10. Run native online Gauss-Newton without Python inside the nonlinear loop.
11. Certify the trajectory with DWR and guard the certificate with branch,
    gauge, and norm-equivalence gates.
12. Use adaptive enrichment if any gate fails.
13. Accept only validated speedup and write a readiness certificate before
    moving to a harder model.

## Basic POD Example

```python
import numpy as np
from pycutfem.mor import fit_pod, project_to_basis, reconstruct_from_basis

# Snapshot matrix shape is (n_dofs, n_snapshots).
snapshots = np.random.default_rng(0).normal(size=(100, 20))

pod = fit_pod(snapshots, energy=0.999, center=True)
q = project_to_basis(snapshots, pod.basis, pod.mean)
reconstructed = reconstruct_from_basis(q, pod.basis, pod.mean)

relative_error = np.linalg.norm(reconstructed - snapshots) / np.linalg.norm(snapshots)
```

The POD approximation is

$$
X \approx \bar{x}\mathbf{1}^T + VQ,
\qquad
Q = V^T(X-\bar{x}\mathbf{1}^T).
$$

## Mixed-Field Basis Example

Use fieldwise bases for mixed systems where small fields can be hidden by large
fields in a global Euclidean POD.

```python
import numpy as np
from pycutfem.mor import (
    fit_fieldwise_pod_basis,
    gauge_correct_snapshots,
    reduced_coupling_rank_certificate,
    certify_mixed_stability_basis,
)

snapshots = np.random.default_rng(1).normal(size=(12, 30))
velocity_rows = np.arange(0, 4)
pressure_rows = np.arange(4, 7)
scalar_rows = np.arange(7, 12)

gauge = {"name": "pressure", "rows": pressure_rows}
corrected = gauge_correct_snapshots(snapshots, [gauge])

fieldwise = fit_fieldwise_pod_basis(
    corrected.corrected_snapshots,
    (
        {"name": "velocity", "rows": velocity_rows},
        {"name": "pressure", "rows": pressure_rows},
        {"name": "scalar", "rows": scalar_rows},
    ),
    n_modes_per_block={"velocity": 3, "pressure": 2, "scalar": 3},
)

certificate = certify_mixed_stability_basis(
    snapshots,
    fieldwise.basis,
    offset=fieldwise.offset,
    row_blocks=(
        {"name": "velocity", "rows": velocity_rows},
        {"name": "pressure", "rows": pressure_rows},
        {"name": "scalar", "rows": scalar_rows},
    ),
    pressure_gauge_blocks=[gauge],
)

assert certificate.passed
```

For saddle-point or constrained mixed systems, check reduced coupling rank:

```python
rank = reduced_coupling_rank_certificate(
    coupling_operator,     # shape (coupled_rows, primary_rows)
    primary_basis,
    coupled_basis,
    name="pressure_divergence",
    required_rank=coupled_basis.shape[1],
)
```

## Lift and Supremizer Enrichment

The generic lift equation is

$$
A_p S = C^T \Phi_c,
$$

where `A_p` is a primary-field operator, `C` is a coupling operator, and
`\Phi_c` is a coupled-field basis.  This covers pressure supremizers and other
mixed-field lift enrichments.

```python
from pycutfem.mor import solve_coupled_lift_snapshots, fit_lift_enriched_basis

lift_snapshots = solve_coupled_lift_snapshots(
    primary_operator,
    coupling_operator,
    coupled_basis,
    solver_backend="auto",
)

enriched = fit_lift_enriched_basis(
    primary_basis,
    coupled_basis,
    lift_snapshots,
    n_lift_modes=4,
)
```

Compatibility wrappers for velocity-pressure problems are also available:
`compute_supremizer_snapshots` and `fit_supremizer_enriched_velocity_basis`.

## Non-Affine Decomposition: DEIM and QDEIM

For a nonlinear feature or residual snapshot matrix `F`, a collateral basis
`\Phi_f` approximates

$$
F(\mu) \approx \Phi_f c(\mu).
$$

DEIM/QDEIM evaluates only selected rows `P^T F` and reconstructs coefficients:

$$
c(\mu) =
\left(P^T \Phi_f\right)^\dagger P^T F(\mu).
$$

```python
from pycutfem.mor import (
    fit_collateral_basis,
    build_qdeim_interpolation_rule,
    reconstruct_from_interpolation,
)

collateral = fit_collateral_basis(residual_snapshots, n_modes=20)
rule = build_qdeim_interpolation_rule(collateral)
approx = reconstruct_from_interpolation(sampled_values, rule)
```

For mixed problems, use the weighted helper:

```python
from pycutfem.mor import build_nonaffine_reduced_decomposition

decomp = build_nonaffine_reduced_decomposition(
    residual_snapshots,
    trial_basis,
    n_modes=20,
    method="qdeim",
    row_weights=row_weights,
)
```

## GNAT Sampling and Norm-Equivalence

Sparse GNAT should be accepted only if sampled residuals see the same local
branch as full residuals:

$$
\gamma \|R(q)\|_2
\le
\|S R(q)\|_W
\le
\Gamma \|R(q)\|_2.
$$

```python
from pycutfem.mor import (
    build_block_balanced_gnat_sampling,
    certify_sampled_residual_norm_equivalence,
    augment_rows_for_residual_norm_equivalence,
)

sampling = build_block_balanced_gnat_sampling(
    dof_handler,
    trial_basis,
    snapshot_matrix=snapshots,
    row_blocks=row_blocks,
    sample_rows=128,
    mandatory_element_ids=interface_elements,
    min_rows_per_block=4,
)

certificate = certify_sampled_residual_norm_equivalence(
    residual_matrix,
    sampling.row_dofs,
    row_weights=sampling.row_weights,
    row_blocks=row_blocks,
    lower_bound=1.0e-3,
    upper_bound=1.0e3,
)

if not certificate.passed:
    augmented = augment_rows_for_residual_norm_equivalence(
        residual_matrix,
        sampling.row_dofs,
        row_weights=sampling.row_weights,
        row_blocks=row_blocks,
        mandatory_rows=sampling.row_dofs,
        max_rows=256,
    )
```

Mandatory elements are intended for diffuse interfaces, active damage zones,
detachment fronts, or any region that must not be omitted by the sampling
matrix.

## Dense and Constrained Gauss-Newton

The dense step solves

$$
\min_{\delta}
\|W^{1/2}(r + J\delta)\|_2^2
+ \lambda \|D\delta\|_2^2.
$$

```python
from pycutfem.mor import gauss_newton_step

step = gauss_newton_step(
    jacobian,
    residual,
    weights=row_weights,
    damping=1.0e-8,
    backend="cpp",
)
```

Active decoded bounds become equality constraints in a PDAS step:

$$
C\delta = h.
$$

```python
from pycutfem.mor import equality_constrained_gauss_newton_step

constrained = equality_constrained_gauss_newton_step(
    jacobian,
    residual,
    constraint_matrix=C,
    constraint_rhs=h,
    backend="cpp",
)
```

Decoded bounds are built from full-state rows:

```python
from pycutfem.mor import bound_constraints_from_fields

bounds = bound_constraints_from_fields(
    total_dofs=n_total,
    fields={
        "alpha": {"rows": alpha_rows, "lower": 0.0, "upper": 1.0},
        "phi": {"rows": phi_rows, "lower": 0.0, "upper": 1.0},
    },
)
```

## Native Online Solves

The native online drivers call generated UFL residual/tangent kernels directly
from C++ and keep Python outside the nonlinear iteration.

Primary entry points:

- `solve_native_online_gauss_newton`
- `solve_native_deim_online_gauss_newton`
- `solve_native_bound_constrained_online_gauss_newton`
- `solve_native_bound_constrained_deim_online_gauss_newton`
- `solve_native_bound_constrained_galerkin_online_gauss_newton`

The native loop supports:

- generated residual/tangent kernels,
- sampled rows and element restrictions,
- dense and sparse GNAT lifts,
- DEIM/QDEIM selected basis and residual terms,
- decoded bounds with PDAS/IPM,
- line search, damping, trust radius, branch radius,
- affine and symbolic state updates,
- timing counters and convergence telemetry.

Convergence for LSPG/GNAT should normally use optimality:

$$
\|J(q)^T R(q)\| \le \tau_{\mathrm{opt}},
$$

not raw residual norm alone.

Use `native_online_convergence_status` to evaluate the result consistently.

## Branch Prediction and Reference Policies

Transient nonlinear reduced solves can jump to a different local root.  The
branch layer provides reduced predictors and reference policies:

- `ConstantReducedPredictor`
- `LinearHistoryReducedPredictor`
- `TimeParameterizedReducedPredictor`
- `fit_time_parameterized_predictor`
- `ReferencePolicy`

Example:

```python
from pycutfem.mor import fit_time_parameterized_predictor, ReferencePolicy

predictor = fit_time_parameterized_predictor(
    train_coefficients,  # shape (n_snapshots, n_modes)
    train_times,
    degree=24,
    ridge=1.0e-12,
)

policy = ReferencePolicy(
    predictor=predictor,
    reference_weight=1.0e-4,
    max_reference_distance=1.0,
    max_step_norm=0.8,
    metric_basis=trial_basis,
)

prediction = policy.predict(time=t_next, q_current=q_n)
native_options = prediction.native_options()
```

## State Updates

Native online solves often need arrays refreshed during each nonlinear trial:

$$
y(q) = y_0 + Bq.
$$

Use:

- `AffineStateUpdateSpec`
- `SymbolicStateUpdateKernelSpec`
- `NativeStateUpdateKernelCall`
- `StateTransactionSpec`
- `build_dirichlet_lift_state_updates`
- `apply_affine_state_updates`

This is how lifted Dirichlet values, local coefficient arrays, quadrature
history, or symbolic state-update kernels are kept consistent during line
search and accepted/rejected trials.

## Artifacts

`NativeReducedArtifact` is the portable schema for native reduced models.  It
stores:

- schema version,
- problem id,
- trial basis and offset,
- residual and tangent kernel references,
- sampled target or GNAT target,
- decoded bound constraints,
- non-affine evaluation graph,
- state transaction metadata,
- adjoint/DWR metadata,
- reference policy,
- solver options,
- free-form metadata.

```python
from pycutfem.mor import NativeReducedArtifact, save_native_reduced_artifact

artifact = NativeReducedArtifact(
    problem_id="my_problem",
    trial_basis=trial_basis,
    offset=offset,
    residual_kernel=residual_ref,
    tangent_kernel=tangent_ref,
    target=gnat_target,
    bound_constraints=bounds,
    reference_policy=policy,
    adjoint_dwr=adjoint_spec,
)

save_native_reduced_artifact(artifact, "my_problem_rom.npz")
```

Reload with `load_native_reduced_artifact`.  A loaded artifact can instantiate a
`NativeReducedRuntimeProblem` when supplied with live compiled kernel capsules
and static arguments.

## Discrete Adjoints and DWR

Define QoIs with:

- `QoIFunctionalSpec`
- `QoIKernelSpec`
- `QoIStatePolicy`
- `TransientResidualDependencySpec`

Assemble and check QoIs:

```python
from pycutfem.mor import (
    evaluate_qoi_functional,
    assemble_qoi_gradient,
    check_qoi_gradient,
    reduced_qoi_gradient_from_full,
)
```

Solve adjoints:

```python
from pycutfem.mor import solve_discrete_adjoint, solve_reduced_discrete_adjoint

adjoint = solve_discrete_adjoint(
    jacobians,
    qoi_gradients,
    previous_state_jacobians=previous_state_jacobians,
    backend="cpp",
)
```

Certify with DWR:

```python
from pycutfem.mor import certify_dual_weighted_residual

certificate = certify_dual_weighted_residual(
    residuals,
    jacobians,
    qoi_gradients,
    previous_state_jacobians=previous_state_jacobians,
    reference_qoi_error=qoi_error,
    effectivity_bounds=(0.1, 20.0),
    backend="cpp",
)
```

For saved trajectories:

```python
from pycutfem.mor import (
    DWRReducedTrajectory,
    save_dwr_reduced_trajectory,
    certify_dual_weighted_residual_from_artifact_trajectory,
)

trajectory = DWRReducedTrajectory(
    residuals=residual_history,
    jacobians=jacobian_history,
    qoi_gradients=qoi_gradient_history,
    previous_state_jacobians=previous_jacobian_history,
    reference_qoi_error=qoi_error,
)
save_dwr_reduced_trajectory(trajectory, "dwr_trajectory.npz")

certificate = certify_dual_weighted_residual_from_artifact_trajectory(
    "rom_artifact.npz",
    "dwr_trajectory.npz",
)
```

Guard DWR certificates:

```python
from pycutfem.mor import dwr_certification_guard

guard = dwr_certification_guard(
    certificate.estimate,
    branch_certificate={"passed": True},
    norm_equivalence_certificate=norm_certificate,
    gauge_certificate={"passed": True},
    require_norm_equivalence=True,
    require_gauge=True,
)
assert guard.passed
```

## Adaptive Enrichment

When a certificate fails, the adaptive layer maps failure modes to actions:

- high field/projection error -> add primal basis modes,
- high QoI/DWR indicator -> add adjoint basis modes,
- failed mixed-coupling rank -> add lift or supremizer modes,
- failed norm-equivalence -> add GNAT rows,
- failed branch certificate -> strengthen predictor/reference model.

```python
from pycutfem.mor import select_certified_adaptive_enrichment_actions

decision = select_certified_adaptive_enrichment_actions(
    field_errors=mixed_certificate.field_errors,
    dwr_estimate=dwr_certificate.estimate,
    norm_equivalence_certificate=norm_certificate,
    branch_certificate=branch_certificate,
    mixed_stability_certificate=mixed_certificate,
)

for action in decision.actions:
    print(action.kind, action.target, action.reason)
```

DWR-localized row augmentation:

```python
from pycutfem.mor import augment_rows_from_dwr_localization

new_rows = augment_rows_from_dwr_localization(
    residuals,
    adjoints,
    row_dofs=current_rows,
    mandatory_rows=interface_rows,
    max_new_rows=16,
)
```

## Readiness Gate

Before moving from a validated benchmark to a harder model, use
`certify_mor_readiness`.

```python
from pycutfem.mor import certify_mor_readiness

ready = certify_mor_readiness(
    validation_summary,
    artifact=loaded_artifact,
    milestone_statuses={24: "Completed", 33: "Completed", 34: "Completed"},
)

if not ready.passed:
    for gate in ready.failed_gates:
        print(gate.name, gate.value, gate.threshold)
```

The readiness gate checks state error, projection error, validated speedup,
predictive validation, decoded bounds, DWR, interface sampling, artifact
metadata, and per-field error metadata.

## Module Inventory

### `pod.py`

- `PODBasis`
- `fit_pod`
- `project`
- `project_to_basis`
- `reconstruct`
- `reconstruct_from_basis`

### `scaling.py`

- `MeanCenterer`
- `StandardScaler`

### `cross_validation.py`

- `ModeSweepEntry`
- `ModeSweepResult`
- `run_mode_cross_validation`

### `mixed_reduction.py`

- `MixedBasisBlock`
- `FieldwisePODBasis`
- `LiftEnrichment`
- `SupremizerEnrichment`
- `NonAffineReducedDecomposition`
- `PressureGaugeBlock`
- `GaugeCorrectionResult`
- `FieldProjectionError`
- `CouplingRankCertificate`
- `MixedStabilityCertificate`
- `field_dof_indices`
- `build_dirichlet_lifting_vector`
- `remove_lifting_from_snapshots`
- `restore_lifting_to_snapshots`
- `solve_coupled_lift_snapshots`
- `compute_supremizer_snapshots`
- `orthonormalize_columns`
- `fit_lift_enriched_basis`
- `fit_supremizer_enriched_velocity_basis`
- `fit_fieldwise_pod_basis`
- `build_mixed_field_basis`
- `build_mixed_velocity_pressure_basis`
- `pressure_gauge_history`
- `gauge_correct_snapshots`
- `field_projection_errors`
- `reduced_coupling_rank_certificate`
- `certify_mixed_stability_basis`
- `build_block_row_weights`
- `build_nonaffine_reduced_decomposition`

### `decomposition.py`

- `CollateralBasis`
- `InterpolationRule`
- `EmpiricalCubatureRule`
- `ReducedOperatorTerm`
- `NativeReducedEvaluationGraph`
- `fit_collateral_basis`
- `select_deim_rows`
- `select_qdeim_rows`
- `build_interpolation_rule`
- `build_deim_interpolation_rule`
- `build_qdeim_interpolation_rule`
- `interpolation_coefficients`
- `reconstruct_from_interpolation`
- `compose_reduced_operator`

### `sampling.py`

- `SamplingBlock`
- `BlockBalancedGnatSampling`
- `ResidualNormEquivalenceCertificate`
- `AugmentedNormEquivalenceResult`
- `field_row_blocks`
- `support_element_ids_from_rows`
- `rows_supported_on_elements`
- `select_coordinate_band_elements`
- `build_block_balanced_gnat_sampling`
- `certify_sampled_residual_norm_equivalence`
- `augment_rows_for_residual_norm_equivalence`

### `gauss_newton.py`

- `GaussNewtonNormalEquations`
- `GaussNewtonStepResult`
- `form_normal_equations`
- `gauss_newton_step`

### `constrained_gauss_newton.py`

- `EqualityConstrainedGaussNewtonStepResult`
- `ConstrainedGaussNewtonBackend`
- `ConstrainedGaussNewtonMethod`
- `equality_constrained_gauss_newton_step`

### `constraints.py`

- `BoundActivity`
- `ActiveBoundEquations`
- `ReducedBoundConstraintSpec`
- `BoundConstraintSpec`
- `bound_constraints_from_fields`
- `project_reduced_coefficients_to_bounds`

### `globalization.py`

- `BranchGlobalizationSpec`
- `ContinuationAttempt`
- `ContinuationResult`
- `clip_step_to_trust_region`
- `step_alpha_to_branch_radius`
- `solve_with_branch_backtracking`

### `predictors.py` and `reference.py`

- `ReducedReferencePrediction`
- `ConstantReducedPredictor`
- `LinearHistoryReducedPredictor`
- `TimeParameterizedReducedPredictor`
- `fit_time_parameterized_predictor`
- `predictor_from_native_dict`
- `ReferencePolicy`
- `ReferencePolicyResult`
- `clip_reference_distance`

### `state_updates.py`

- `NativeStateArraySpec`
- `AffineStateUpdateSpec`
- `SymbolicStateUpdateKernelSpec`
- `NativeStateUpdateKernelCall`
- `StateTransactionSpec`
- `coerce_affine_state_update`
- `coerce_affine_state_updates`
- `apply_affine_state_updates`
- `build_dirichlet_lift_state_updates`

### `native_assembly.py`

- `native_kernel_metadata_from_runner`
- `call_native_kernel`
- `reduced_target_from_native_kernel_pair`
- `gnat_system_from_native_kernel`
- `sampled_lspg_rows_from_native_kernel`
- `sampled_galerkin_reduced_system_from_native_kernel`
- `apply_affine_updates_to_static_args`

### `reduced_assembly.py`

- `AffineReducedState`
- `ReducedLocalAssembler`
- `validate_local_blocks`
- `validate_element_weights`
- `decode_element_values`
- `decode_values_on_dofs`
- `sampled_lspg_rows_from_local_blocks`
- `sampled_lspg_element_contributions_from_local_blocks`
- `sampled_galerkin_reduced_system_from_local_blocks`
- `sampled_galerkin_element_contributions_from_local_blocks`
- `sampled_galerkin_reduced_system_from_native_kernel`
- `reduced_reaction_from_local_blocks`
- `constrained_reaction_rows_from_local_blocks`
- `apply_gnat_lift`

### `online_gauss_newton.py`

- `NativeOnlineGaussNewtonResult`
- `NativeOnlineConvergenceStatus`
- `native_online_convergence_status`
- `solve_native_online_gauss_newton`
- `solve_native_deim_online_gauss_newton`
- `solve_native_bound_constrained_online_gauss_newton`
- `solve_native_bound_constrained_deim_online_gauss_newton`
- `solve_native_bound_constrained_galerkin_online_gauss_newton`

### `sparse.py`

- `NativeSparseMatrix`
- `is_sparse_matrix_like`
- `apply_sparse_gnat_lift`
- `sparse_gnat_normal_equations`

### `artifacts.py`

- `NATIVE_REDUCED_ARTIFACT_SCHEMA_VERSION`
- `NativeKernelReference`
- `NativeGnatTargetSpec`
- `NativeAdjointDWRSpec`
- `NativeReducedArtifact`
- `NativeReducedRuntimeProblem`
- `save_native_reduced_artifact`
- `load_native_reduced_artifact`

### `dwr.py`

- `QoIFunctionalSpec`
- `QoIKernelSpec`
- `QoIStatePolicy`
- `TransientResidualDependencySpec`
- `QoIGradientCheck`
- `DiscreteAdjointResult`
- `DWREstimate`
- `DWRCertificationResult`
- `DWRReducedTrajectory`
- `DWRGuardResult`
- `solve_transpose_system`
- `solve_discrete_adjoint`
- `solve_reduced_discrete_adjoint`
- `finite_difference_gradient`
- `check_qoi_gradient`
- `linearize_qoi_functional`
- `evaluate_qoi_functional`
- `assemble_qoi_gradient`
- `reduced_qoi_gradient_from_full`
- `dual_weighted_residual_estimate`
- `dominant_dwr_contributions`
- `dwr_certification_guard`
- `save_dwr_reduced_trajectory`
- `load_dwr_reduced_trajectory`
- `certify_dual_weighted_residual`
- `certify_dual_weighted_residual_from_artifact_trajectory`

### `adaptive.py`

- `AdaptiveEnrichmentAction`
- `AdaptiveMORDecision`
- `select_certified_adaptive_enrichment_actions`
- `augment_rows_from_dwr_localization`

### `readiness.py`

- `MORReadinessGate`
- `MORReadinessCriteria`
- `MORReadinessCertificate`
- `certify_mor_readiness`

### `interface.py`

- `InterfaceRestriction`
- `build_interface_restriction`
- `build_restriction_matrix`

### `quadratic_manifold.py`

- `QuadraticFeatureMap`
- `QuadraticManifoldDecoder`
- `quadratic_feature_matrix`
- `fit_quadratic_decoder`
- `fit_quadratic_manifold`

### `regressors.py`

- `PolynomialFeatureMap`
- `PolynomialLeastSquaresRegressor`
- `PolynomialLassoRegressor`
- `ThinPlateSplineRBF`
- `fit_poly_least_squares`
- `fit_poly_lasso`
- `fit_tps_rbf`

### `metrics.py`, `timing.py`, `snapshots.py`, and `io.py`

- `mean_sample_l2_error`
- `snapshot_l2_error`
- `reduced_regression_error`
- `online_relative_displacement_error`
- `max_online_relative_displacement_error`
- `accumulated_iteration_overhead`
- `speedup`
- `Timer`
- `TimingAccumulator`
- `build_speedup_report`
- `SnapshotBatch`
- `SnapshotReader`
- `SnapshotWriter`
- `load_config`
- `save_config`
- `load_model`
- `save_model`
- `save_results`

### `incompressible.py`

This is a compatibility module that re-exports velocity-pressure helper names
from `mixed_reduction.py`.  New code should import from `pycutfem.mor`
or `pycutfem.mor.mixed_reduction` directly.

### `cpp_backend/`

The C++ backend modules provide native kernels for dense Gauss-Newton,
constrained Gauss-Newton, reduced projection, sparse GNAT, native reduced
assembly, DEIM online evaluation, online Gauss-Newton, and adjoint algebra.
Users normally call the high-level Python wrappers listed above; the wrappers
compile/import the C++ extensions as needed.

## Efficient Usage Rules

- Keep offline training in Python, but keep online nonlinear loops in native C++
  once the artifact is built.
- Use fieldwise POD for mixed systems.
- Gauge-correct pressure-like fields before POD and before QoI certification.
- Use lift/supremizer enrichment when a reduced coupling rank gate fails.
- Use full-row LSPG or true Galerkin as the diagnostic baseline before sparse
  GNAT.
- Do not enable sparse GNAT unless norm-equivalence passes on the validation
  neighborhood.
- Use `ReferencePolicy` for transient nonlinear branch selection.
- Use decoded bounds for physical constraints such as volume fractions,
  concentrations, damage, or detachment variables.
- Treat DWR as a certificate and adaptivity signal, not as a replacement for a
  branch-correct primal solve.
- Report validated speedup only when state, bounds, branch, gauge, DWR, and
  sampling gates pass.

## Production Acceptance Checklist

A reduced model is production-ready only when:

- the FOM trajectory is stable and reproducible,
- the reduced trajectory is predictive, not exact trajectory replay,
- per-field projection errors pass,
- mixed stability and gauge gates pass,
- decoded bounds pass,
- native nonlinear convergence uses the correct residual/optimality gate,
- sampled GNAT passes residual norm-equivalence,
- DWR effectivity and guard checks pass for the selected QoIs,
- the native artifact can be reloaded,
- validated online speedup is positive,
- `certify_mor_readiness` passes before moving to the next harder model.
