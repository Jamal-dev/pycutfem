# Gateaux Derivative Implementation Plan

## Goal

Add a native pycutfem UFL Gateaux derivative so a nonlinear weak form residual
can be linearized into a Jacobian form from the symbolic pycutfem expression
tree itself, without relying on the old SymPy prototype.

The feature must be **fail-closed**:

- If the residual contains an unsupported or non-differentiable construct, the
  code must raise a clear error.
- It is explicitly better to reject a form than to assemble a wrong Jacobian.

## Required Outcomes

1. A public API can linearize `Integral` / `Form` residuals with respect to one
   or more coefficient functions and directions.
2. The API works for representative Navier-Stokes-style residuals and for
   coupled benchmark-7-style residuals.
3. The implementation supports the operator set already used by pycutfem’s
   nonlinear formulations, including mixed scalar/vector/H(div) fields.
4. Unsupported hard non-smooth constructs fail explicitly.
5. The solver stack can opt into auto-linearization from `residual_form` plus
   `(coefficient, direction)` pairs instead of requiring a manually written
   `jacobian_form`.

## Planned API

Add a new native autodiff module under `pycutfem/ufl/`.

Public entry points:

- `gateaux_derivative(obj, coefficients, directions, *, strict=True)`
- `linearize_form(residual_form, coefficients, directions, *, strict=True)`
- `linearize_equation(residual_form, coefficients, directions, *, strict=True)`

Planned semantics:

- `obj` may be an `Expression`, `Integral`, or `Form`.
- `coefficients` / `directions` may be a single object or parallel sequences.
- Scalar coefficients map to scalar trial directions.
- `VectorFunction` maps to `VectorTrialFunction`.
- `HdivFunction` maps to `HdivTrialFunction`.
- `linearize_equation(...)` returns `Equation(jacobian_form, residual_form)`.

Solver integration:

- Extend `NewtonSolver` so `jacobian_form=None` is allowed when explicit
  linearization pairs are provided.
- If the auto-linearization request is incomplete or ambiguous, raise.

## Native Differentiation Rules

### Leaves treated as differentiable unknowns

- `Function`
- `VectorFunction`
- `HdivFunction`
- their component views when they belong to one of the mapped coefficients

### Leaves treated as constants with zero Gateaux derivative

- `Constant`
- `Identity`
- `Analytic`
- `ElementWiseConstant`
- `CellDiameter`
- `MeshSize`
- `FacetNormal`
- `NormalComponent`
- `NodalFunction`

### Composite nodes to support

- `Sum`, `Sub`
- `Prod`, `Div`
- `Dot`, `Inner`, `Outer`
- `Transpose`
- `Grad`, `Derivative`, `DivOperation`, `Hessian`, `Laplacian`
- `Restriction`, `Side`, `Avg`, `Jump`, `Pos`, `Neg`
- `Power`
- `Log`, `Exp`, `Tanh`
- `Sin`, `Cos`, `Tan`, `Asin`, `Acos`, `Atan`
- `Sinh`, `Cosh`, `Asinh`, `Acosh`, `Atanh`
- `Trace`
- `Determinant`
- `Inverse`
- `Cofactor`

### Nodes that must fail by default

- `PositivePart`
- `Heaviside`
- any unknown expression node not in the supported list

Failure rule:

- If a hard non-smooth node does not depend on the differentiated
  coefficients, derivative is zero and the form may proceed.
- If it does depend on the differentiated coefficients, raise a dedicated
  autodiff error.

## Robustness Constraints

1. Shape/type checks must be enforced before differentiation starts.
2. Vector and H(div) coefficients must only map to matching direction types.
3. Component views must resolve back to their owning coefficient so mixed forms
   with `u_k[0]`, `v_k[1]`, or H(div) components differentiate correctly.
4. The resulting Jacobian form must remain symbolic pycutfem UFL, not a side
   channel representation.
5. Unknown-node handling must be centralized so new expression types do not
   silently pass through as zero.

## Pass / Fail Study

### Pass study

The feature is accepted only if all of the following pass:

1. Unit-level symbolic rules:
   product rule, quotient rule, chain rule, gradient/divergence propagation,
   determinant/inverse/cofactor rules.
2. Navier-Stokes-style residual:
   auto-linearized Jacobian matches a manually written Jacobian on assembly.
3. Benchmark-7-style coupled pressure/alpha/H(div) block:
   auto-linearized Jacobian matches the manually written Jacobian on assembly.
4. Coupled benchmark-7 residual form:
   auto-linearized full Jacobian matches the existing manual Jacobian on a
   representative small problem.
5. Optional Fenicsx cross-check where available:
   a representative auto-linearized pycutfem form matches `ufl.derivative(...)`
   assembled in the `fenicsx` environment.

### Fail study

The feature is rejected if any of the following occur:

1. A residual containing `PositivePart` or `Heaviside` depending on the chosen
   coefficients produces a Jacobian instead of raising.
2. A residual containing an unknown expression node is silently differentiated.
3. Mismatched coefficient/direction pairings are accepted.
4. The auto-linearized Jacobian differs from the existing manual Jacobian for
   the benchmark forms within the chosen tolerances.
5. The solver silently auto-linearizes when `jacobian_form=None` but no valid
   linearization data was supplied.

## Initial Validation Matrix

### Must-pass tests

- symbolic scalar/vector/H(div) rule tests
- unsupported-node failure tests
- Navier-Stokes-style manual-vs-auto Jacobian parity
- benchmark-7 pressure/alpha/H(div) block parity
- benchmark-7 full coupled residual parity on a small mesh

### Optional pass when environment exists

- Fenicsx autodiff parity via
  `conda run --no-capture-output -n fenicsx python ...`

## Implementation Notes

- Prefer a native pycutfem-tree visitor over the old `pycutfem/ufl/symops.py`
  path. The SymPy prototype is useful as a reference only.
- Reuse existing pycutfem symbolic helpers such as `trace`, `det`, `inv`,
  `cof`, `grad`, `div`, `dot`, `inner`, and `log` in the generated derivative
  tree so all backends see standard pycutfem nodes.
- Keep simplification conservative. Exact zero propagation is useful; any
  algebraic rewrite beyond that should be minimal.

## Acceptance Criteria

The work is complete when:

- the new API is documented and exported,
- the solver can optionally auto-build a Jacobian from a residual,
- the pass study above is green,
- the fail study above raises the intended explicit errors,
- and the remaining unsupported cases are documented instead of being guessed.
