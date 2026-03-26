# Dot / Tensor Refactor Design

## Status

- [x] Objective 1: Introduce one shared tensor-signature model for Python, JIT, and C++ backends.
- [x] Objective 2: Move `dot` contraction decisions into one shared rule engine.
- [x] Objective 3: Move scalar/vector/matrix `*` promotion and mixed-basis multiplication decisions into one shared rule engine.
- [x] Objective 4: Make `grad`/`Hessian` semantics rank-based and tag-based instead of singleton-axis-based.
- [x] Objective 5: Support composite scalar operations under `grad` through linearity and chain-rule expansion.
- [x] Objective 6: Preserve current FenicsX parity on `examples/debug/comparison_with_fenics.py`.
- [x] Objective 7: Keep all three backends aligned on the same result metadata and contraction orientation.
- [x] Objective 8: Distinguish field-component provenance from derivative-channel provenance without changing algebraic tensor rank.

### UFL Reference Note

The maintained developer-facing semantic note now lives in:

- `pycutfem/ufl/tensor_semantics.md`

That file should be treated as the compact long-term reference for tensor rank,
`dot`, `inner`, `*`, basis ownership, mixed layout, and compile-time storage rules
inside the `ufl` package.

### Completed Now

- `pycutfem.ufl.tensor_algebra` is the shared semantic layer for runtime op-info objects and backend stack items.
- The compact long-term UFL developer note now lives at `pycutfem/ufl/tensor_semantics.md`.
- The Python compiler plus the JIT/C++ code generators now consume the shared dot-plan logic for the contraction-sensitive vector/gradient/Hessian cases.
- JIT and C++ stack items now carry shared `ExpressionMeta` and mixed-layout tags, so later operations reuse semantic tensor/provenance information instead of re-inferring from legacy storage shape.
- Non-scalar `*` now follows tensor-product semantics at the expression layer: scalar scaling stays scalar, while vector/matrix/tensor products append free axes instead of silently contracting.
- Scalar gradients are now inferred semantically as spatial vectors, not fake leading-component tensors.
- Scalar basis gradients now carry in canonical basis form `(d, n)` in the Python/JIT algebra path instead of legacy `(1, n, d)`.
- Mixed basis and mixed gradient stack items now infer the same basis-axis ordering in the shared rule layer.
- Shared product lowering now uses derivative provenance as well as tensor rank, so promoted scalar-gradient vectors keep gradient semantics across backend lowering.
- Scalar basis row `*` scalar basis row now goes through the shared mixed outer-product path before scalar scaling, so `test_row * trial_row` lowers to a true `(n_test, n_trial)` mixed matrix instead of a wrapped scalar carrier.
- Value-side `dot(grad(...), grad(...))` now lowers by semantic tensor rank rather than raw stored shape, so rank-1 wrappers and rank-2 matrices follow the same shared contraction rule in JIT/C++.
- Composite scalar gradients are covered by backend-parity regression tests that exercise linearity and chain-rule expansion.
- Focused Fenics comparison terms covering `grad`/vector convection contractions and identity-times-gradient products pass on `python`, `jit`, and `cpp`.
- Multiplication-specific Fenics comparison terms covering dyads, vector-by-matrix constants, and vector-by-gradient tensor products pass on `python`, `jit`, and `cpp`.
- Mixed/component-contraction Fenics comparison terms covering scalar-gradient dots, row extraction, and expanded vector-gradient contractions pass on `python`, `jit`, and `cpp`.
- `grad · basis` and `basis · grad` now lower to different mixed layouts according to tensor-contraction order instead of reusing one ambiguous helper path.
- Shared provenance metadata now distinguishes vector-field components from scalar-gradient derivative channels, and the shared rule engine exposes sum-planning on top of that model.
- Shared sum planning now computes tensor-aware role selection for `+`/`-` in the JIT and C++ backends, and broadcasts raw shapes when the legacy storage layouts are already compatible.
- Mixed-matrix sums now preserve the concrete emitted runtime shape from the shared product planner, so the JIT add/sub path no longer drops a leading row through stale singleton-wrapper metadata.
- Shared inner planning now computes scalar-result basis layout and metadata from the same tensor signature model used by `dot` and `*`, so JIT/C++ no longer need to invent `(-1,)` or `(-1,-1)` result shapes for supported inner products.
- Shared scalar-division planning now preserves carrier tensor metadata for `/`, so Python, JIT, and C++ all use the same result role/shape/layout rules for scalar denominator and reciprocal-style scalar numerator cases.
- Shared dot/product lowering now carries an explicit `StorageSpec` with exact emitted shapes, free-axis positions, and basis-axis positions.
- Shared lowering now also exposes a backend-facing `KernelValueSpec`, so JIT/C++ can consume `kind`, `role`, `shape`, `layout`, and `ExpressionMeta` from the same source instead of recomputing those pieces ad hoc.
- Dot lowering now has one compile-time source for contraction axes and result shapes, instead of recovering them from backend-local `shape == ...` branches.
- H(div) vector expressions now participate in symbolic shape inference as true rank-1 vectors before the generic scalar-`Function` fallback, so `grad(v)` is typed as rank-2 and `Transpose(grad(v))` remains a matrix transpose instead of a vector hack.
- Mixed rank-1 intermediates created by contractions such as `dot(grad(test), trial)` are now treated as true rank-1 mixed tensors, and their subsequent contraction with value vectors uses the same first-free-axis rule as the shared tensor planner.
- Ghost/interface/facet restricted derivatives now compile through the same semantic storage planning, including sided gradient loads and mixed layouts.
- C++ and JIT dot lowering now consume planned storage/layout metadata for the mixed basis cases as well as the value/value cases; branch-local fallback shapes are only compatibility defaults when the planner truly has no exact extent.
- C++ rank-1 basis contractions now support both matrix-backed storage and carried `std::vector<Eigen::MatrixXd>` storage, so planner-driven dot paths also work for Hessian-derived gradient carriers on ghost/interface integrals.
- Shared storage inference now classifies scalar Hessian singleton wrappers semantically, so scalar Hessian basis/value dots lower through the same planner path as rank-1 and rank-2 tensor contractions instead of falling back on raw `(1, ...)` shape guesses.
- JIT basis-carrying dot results now package `role`, `shape`, `layout`, and semantic flags through one shared basis-result helper, and rank-1 `dot(basis-vector, grad(value))` now chooses the scalar-vs-tensor path from the planner tensor rank instead of `shape[0] == 1`.
- C++ dot lowering now has a shared `push_dot_result(...)` path for planner-owned result metadata, so rank-1 `vec·grad` / `grad·vec` contractions stop reintroducing local kind/shape drift after helper emission.
- Non-verbose C++ kernel builds now suppress setuptools/distutils INFO chatter; successful builds stay quiet and only real compiler diagnostics are surfaced on failure.
- Fresh facet/interface/ghost regression reruns covering exterior facets, interface side maps, aligned interfaces, grad-only facet kernels, Hessian normal-jump ghost penalties, and JIT FSI interface traction all pass.
- Fresh full-suite parity reruns now pass for:
  - `python`: `253 / 0`
  - `jit` with `NUMBA_DISABLE_JIT=1`: `253 / 0`
- Fresh targeted C++ parity reruns for the previously failing solid cross-tangent and vector-Hessian terms now pass with `0` failures.
- Fresh full-suite Fenics parity rerun now passes for:
  - `cpp`: `253 / 0`
- Fresh ghost/interface/facet pytest rerun passes:
  - `176 passed, 3 skipped, 177 deselected`
- Scoped `awk` audits of the main JIT/C++ dot and product lowering sections are now clean when checking for planner-owned result flow:
  - JIT `dot`: `dot_lowering|dot_plan|dot_value_spec|dot_case|_mul_scalar_vector`
  - JIT `*`: `product_lowering|product_case|product_value_spec|sum_lowering|sum_plan|push_bin|_mul_scalar_vector`
  - C++ `dot`: `dot_lowering|dot_plan|dot_value_spec|dot_case|push_dot_result`
  - C++ `*`: `product_lowering|product_case|product_value_spec|push_bin`

### Verification Closeout

The all-term Fenics comparison closeout is complete:

- `python`: `253 / 0`
- `jit` with `NUMBA_DISABLE_JIT=1`: `253 / 0`
- `cpp`: `253 / 0`

### Product State

The mathematical meaning of `*` is now:

- scalar `*` anything: scalar scaling
- non-scalar `*` non-scalar: tensor product / dyad

Examples:

- `a * I_ij -> c_ij`
- `a_i * b_j -> c_ij`
- `a_i * b_jk -> c_ijk`

This semantic rule is now enforced at the symbolic expression layer and is verified by backend parity tests plus a dedicated Fenics comparison suite.

What remains backend-specific is helper selection, not product semantics:

- some JIT/C++ multiplication code paths still choose which helper to emit through legacy shape-driven branching
- the algebraic meaning, result metadata, mixed-basis orientation, and compile-time storage layout now come from shared lowering

## Problem

Today the code uses raw array shape plus ad hoc `role` branches as the source of truth for tensor algebra.
That is the main bottleneck.

Two expressions can have the same numeric length `2` but different meaning:

- a velocity field `v` has a field-component axis
- `grad(a)` for scalar `a` has a spatial axis

The current implementation often has to guess which axis is being contracted from raw shape alone.
That is why cases such as

- `dot(v, grad(a))`
- `dot(grad(v), v)`
- `dot(v, grad(v))`

need different special cases even though they are all rank-1/rank-2 contractions.

The root cause is:

1. tensor meaning is implicit
2. basis axes are mixed with free tensor axes
3. singleton promotion for scalar gradients (`(1,n,2)`) leaks into algebra decisions
4. Python, JIT, and C++ backends each re-encode the same semantics independently

## Design Goals

The refactor must satisfy these rules.

1. Tensor meaning must be explicit and backend-independent.
2. `dot` must be decided from axis tags, not from raw shape heuristics.
3. `grad(a)` must be treated semantically as a spatial vector.
4. Scalar basis gradients should carry canonically as `(d, n)` wherever the backend can do so, and legacy `(1, n, d)` storage must not affect contraction semantics.
5. `grad(v)` must be treated semantically as a matrix with axes `(field_component, spatial)`.
6. Higher derivatives must extend the free-axis list in a predictable way.
7. Basis axes and free tensor axes must be tracked separately in the design, even if a backend keeps a legacy storage layout.
8. The source of truth must live in one module and be reused by Python, JIT, and C++ code generation.
9. Existing FenicsX parity must not regress.
10. Kernel metadata should carry exact basis extents whenever they are known at code-generation time.

## UFL / FEniCSX Source Semantics

These are the primary-source semantics from the installed UFL implementation in the `fenicsx` environment, and they are the target model for this refactor.

- `grad` appends one spatial free axis.
  - scalar: `() -> (d,)`
  - vector: `(k,) -> (k, d)`
  - matrix: `(i, j) -> (i, j, d)`
- `dot(lhs, rhs)` contracts `lhs` last free axis with `rhs` first free axis.
  - result shape: `lhs_shape[:-1] + rhs_shape[1:]`
- `outer(lhs, rhs)` concatenates free axes and performs no contraction.
- `transpose` is only defined for rank-2 tensors.

This is why the refactor must be rank/axis based rather than storage-shape based.

## Terminology

### Free Tensor Axis

An axis that participates in algebraic tensor operations such as `dot`, `inner`, `trace`, transpose, and contraction.

Examples:

- vector component axis of `v`
- spatial axis of `grad(a)`
- row/column axes of `grad(v)`

### Basis Axis

An axis indexing local test or trial basis functions.
Basis axes do not participate in tensor contraction.
They determine matrix/vector assembly orientation.

### Axis Tags

Each free axis has a tag:

- `field_component`
- `spatial`

Each basis axis has a tag:

- `test_basis`
- `trial_basis`

## Canonical Semantic Model

We introduce a shared semantic object:

```text
TensorSignature(
    free_axes: tuple[AxisTag, ...],
    basis_axes: tuple[BasisAxisTag, ...],
    family: "value" | "basis" | "mixed_basis",
    role: "const" | "value" | "function" | "test" | "trial" | "mixed",
)
```

This is the semantic source of truth.
Backends may still store arrays in their current physical layout, but all algebra decisions are taken from `TensorSignature`, not from raw shape alone.

## Storage Plan

Semantic tensor rank is not enough to emit kernels.
We also need one shared compile-time storage description that tells every backend where the free axes and basis axes live in the stored array.

```text
StorageSpec(
    stored_shape: tuple[int, ...],
    free_axis_positions: tuple[int, ...],
    basis_axis_positions: tuple[int, ...],
    basis_sizes: tuple[int, ...],
    canonical_shape: tuple[int, ...],   # free axes followed by basis axes
)
```

This is the bridge between the clean UFL-style tensor algebra and the backend storage layouts.

Rules:

- values: store free axes in logical order
- one basis axis:
  - scalar basis keeps a compatibility row wrapper `(1, n)` for now
  - rank-1 basis tensors store as `(free, n)`
  - rank-2 basis tensors store as `(free0, n, free1)`
- mixed basis:
  - scalar mixed tensor stores as `(n_test, n_trial)`
  - rank-1 mixed tensor stores as `(free, n_test, n_trial)`
  - rank-2 mixed tensor stores as `(free0, n_test, n_trial, free1)`

Examples:

- scalar trial/test: `(1, n)`
- vector trial/test: `(d, n)`
- scalar `grad(trial)`: `(d, n)`
- vector `grad(trial)`: `(k, n, d)`
- mixed vector tensor: `(k, n_test, n_trial)`
- mixed gradient tensor: `(k, n_test, n_trial, d)`

`plan_dot_lowering()` and `plan_product_lowering()` must return these storage specs, not just logical tensor ranks.

In the implemented closeout state, these storage specs are also the compile-time shape-prediction path for code generation:

- if basis extents are known when writing the kernel, `stored_shape` carries the exact emitted shape
- JIT/C++ dot/product lowering read that planned shape instead of rediscovering it from emitted arrays
- runtime `-1` extents remain only for genuinely dynamic union-style temporaries or compatibility wrappers that are not yet fully normalized

## Backend Result Spec

The shared planner must also tell the backends what kind of stack item to push after lowering.

```text
KernelValueSpec(
    kind: "scalar" | "vec" | "mat" | "grad" | "hess" | "mixed",
    role: str,
    shape: tuple[int, ...],
    layout: MixedLayout,
    is_vector: bool,
    is_gradient: bool,
    is_hessian: bool,
    meta: ExpressionMeta,
)
```

This is the compile-time contract between the shared tensor planner and the backend code generators.
Backends should not guess `kind`/`layout` again from row-vs-column hacks or `shape[0] == 1`.

## Provenance Model

Algebraic tensor rank is not enough to recover field ownership.

- `v` and `grad(a)` are both rank-1 tensors
- `grad(v)` and `grad(grad(a))` are both rank-2 tensors

That is correct algebraically, but it is not enough for robust field handling.
We therefore track provenance separately from tensor rank.

```text
ExpressionMeta(
    tensor: TensorSignature,
    provenance: ProvenanceSignature(
        sources: tuple[
            FieldSource(
                parent: str,
                fields: tuple[str, ...],
                kind: "field_components" | "derivative_channels" | "constant" | "unknown",
                derivative_depth: int,
            ),
            ...
        ]
    ),
)
```

Rules:

- tensor algebra (`dot`, `inner`, `*` promotion, transpose, trace) uses `TensorSignature`
- assembly orientation uses basis axes
- field ownership and derivative lineage use `ProvenanceSignature`
- addition requires matching tensor/basis signatures, but may merge different provenance sources

Examples:

- `v = (v_x, v_y)`:
  provenance kind = `field_components`
- `grad(a) = (a_x, a_y)`:
  provenance kind = `derivative_channels`
- `grad(a) + v`:
  algebraically valid rank-1 sum, provenance becomes the union of both sources

## Canonical Semantics of Core Objects

### Scalar

- value/function: no free axes
- trial/test basis: no free axes, one basis axis

### Vector Field

`v`

- free axes: `(field_component,)`
- value shape semantics: `(2,)`
- basis semantics: one basis axis plus one free `field_component` axis

### Scalar Gradient

`grad(a)`

- free axes: `(spatial,)`
- value semantics: spatial vector
- basis semantics: scalar-basis differentiated once, still a spatial vector basis
- canonical basis storage target: `(d, n)`
- legacy `(1, n, d)` wrappers may still exist in compatibility paths, but they are not part of the algebraic model

Important:
`grad(a)` is not semantically a matrix and it is not semantically a tensor with a fake leading component axis.

### Vector Gradient

`grad(v)`

- free axes: `(field_component, spatial)`
- value semantics: matrix
- basis semantics: gradient basis with the same two free axes

### Higher Derivatives

`grad` appends one `spatial` free axis to the operand semantics.

Examples:

- `grad(a)` -> `(spatial,)`
- `grad(v)` -> `(field_component, spatial)`
- `grad(grad(a))` -> `(spatial, spatial)`
- `grad(grad(v))` -> `(field_component, spatial, spatial)`

For basis-valued objects, the basis axis remains a basis axis and never becomes part of the free-axis list.

### Hessian

`Hessian(a)` is equivalent to `grad(grad(a))`.

`Hessian(v)` is equivalent to `grad(grad(v))`.

## Raw Storage vs Semantic Axes

This refactor separates:

1. semantic axes used for algebra
2. backend storage axes used for computation

This is critical because current backends already use different practical layouts:

- Python helper objects
- Numba stack items
- C++ stack items

The refactor does **not** require every backend to switch storage immediately.
It does require every backend to expose a `TensorSignature`.

That lets us refactor safely in two stages:

1. unify decisions first
2. normalize storage later if still useful

## Dot Rules

`dot(a, b)` contracts:

- the last free axis of `a`
- with the first free axis of `b`

This is valid only if those axis tags are compatible.

### Tag Compatibility

Allowed:

- `field_component` with `field_component`
- `spatial` with `spatial`

Not allowed:

- `field_component` with `spatial`

unless one operand is already semantically represented so that the contracted axis is the correct one.
This is exactly why the signature matters.

Examples:

- `dot(v, grad(a))`
  - `v.free_axes = (field_component,)`
  - `grad(a).free_axes = (spatial,)`
  - this is valid only because both are physical vectors in the same ambient dimension
  - the rule engine maps this through an explicit vector-compatibility rule, not through singleton-shape guessing

- `dot(v, grad(v))`
  - contracts `field_component` of `v`
  - with first free axis of `grad(v)`, which is `field_component`
  - result free axes: `(spatial,)`

- `dot(grad(v), v)`
  - contracts last free axis of `grad(v)`, which is `spatial`
  - with first free axis of `v`, which is the physical vector axis
  - result free axes: `(field_component,)`

The engine must distinguish those two cases from tags and operand families, not from special-case shape branches.

## Inner Rules

`inner(a, b)` fully contracts all free axes.

Requirements:

1. same number of free axes
2. pairwise compatible axis tags
3. basis axes are preserved and ordered as:
   - test first
   - trial second

## Product Rules

`*` is used for three different semantic categories today.
The refactor makes them explicit.

### 1. Scalar Scaling

If either operand has no free axes, multiply pointwise and preserve the other operand signature.

Examples:

- scalar value `*` vector value
- scalar function `*` grad(function)
- scalar trial/test `*` vector value

### 2. Basis Outer Product

If both operands are scalar bases, the result has no free axes and two basis axes:

- `(test_basis, trial_basis)`

### 3. Basis Promotion by Scalar Factors

If one operand is a scalar basis and the other operand has free axes, preserve the other operand free axes and attach the basis axis from the scalar basis.

Examples:

- scalar trial `*` vector value -> vector trial
- scalar trial `*` grad(value) -> gradient trial
- scalar test `*` grad(trial) -> mixed gradient

This removes the need to invent fake tensor ranks just to make addition possible.

## Mixed Basis Orientation

Any bilinear object must carry basis axes in canonical order:

- rows: `test_basis`
- cols: `trial_basis`

This rule is backend-independent.
If operands arrive reversed, the rule engine still returns the same canonical semantic orientation.

## Transpose Rules

Transpose acts only on free axes.
It never moves basis axes.

Examples:

- transpose of `grad(v)` swaps `(field_component, spatial)`
- transpose of a mixed bilinear block swaps matrix free axes, not test/trial basis axes

## Trace Rules

Trace is only valid when the two contracted free axes are compatible and square.
Basis axes remain untouched.

Examples:

- `trace(grad(v))`
- `trace(dot(Finv, dF))`

## Gradient / Derivative Rules for Composite Operations

The refactor explicitly supports composite scalar differentiation through one recursive rule set:

### Linearity

- `D(a + b) = D(a) + D(b)`
- `D(a - b) = D(a) - D(b)`

### Product Rule

- `D(a * b) = D(a) * b + a * D(b)`

### Quotient Rule

- `D(a / b) = (D(a) * b - a * D(b)) / (b * b)`

### Chain Rule

- `D(exp(a)) = exp(a) * D(a)`
- `D(log(a)) = D(a) / a`
- `D(a^p) = p * a^(p-1) * D(a)` for scalar constant `p`

Scope for this refactor:

- full support for scalar composite expressions under `grad` / `Derivative`
- no new automatic tensor-valued chain-rule system beyond the current scalar use cases

That matches current actual use cases in the biofilm and FPI forms.

## Shared Rule Engine

We add one shared class, used by all three backends:

```text
TensorRuleEngine
  - infer_signature(...)
  - plan_dot(lhs, rhs)
  - plan_inner(lhs, rhs)
  - plan_division(lhs, rhs)
  - plan_product(lhs, rhs)
  - plan_grad(sig)
  - plan_transpose(sig)
  - plan_trace(sig)
```

Each plan returns:

- input semantic signatures
- result semantic signature
- contraction description
- canonical basis orientation
- a small backend-agnostic operation kind

The backend-agnostic operation kind is then lowered by:

- Python compiler to `VecOpInfo` / `GradOpInfo` / `HessOpInfo`
- Numba codegen to helper calls
- C++ codegen to helper calls

## Backend Integration Plan

### Python Backend

1. Add `TensorSignature` adapters for `VecOpInfo`, `GradOpInfo`, and `HessOpInfo`.
2. Route `_visit_Dot` through `TensorRuleEngine.plan_dot`.
3. Route `_visit_Prod` through `TensorRuleEngine.plan_product`.
4. Keep `VecOpInfo` / `GradOpInfo` raw arrays for now, but stop deciding semantics from singleton shape.

### Numba Backend

1. Extend `StackItem` with semantic tensor metadata.
2. Infer signature at load time for values, bases, gradients, Hessians, and mixed terms.
3. Replace hand-written `dot` case selection with `TensorRuleEngine.plan_dot`.
4. Lower the returned plan to existing helpers where possible.
5. Add small helpers only when a plan cannot be expressed cleanly with the current primitives.

### C++ Backend

1. Extend C++ `StackItem` metadata generation with the same semantic signature.
2. Use the same shared Python-side rule engine during code generation.
3. Lower plans to existing Eigen helpers.
4. Keep rows=test / cols=trial canonical ordering for every mixed result.

## Migration Strategy

### Stage 1

- create `design_dots.md`
- add `TensorSignature` and `TensorRuleEngine`
- add unit tests for signature inference and rule planning

### Stage 2

- wire Python backend to the new rule engine
- keep behavior identical

### Stage 3

- wire Numba and C++ codegen to the same rule engine
- preserve existing helper kernels where possible

### Stage 4

- run parity tests and `examples/debug/comparison_with_fenics.py`
- fix any remaining mismatches

## Acceptance Criteria

The refactor is complete only when:

1. `dot(v, grad(a))` and `dot(grad(a), v)` are handled without scalar-gradient singleton hacks.
2. `dot(v, grad(v))` and `dot(grad(v), v)` are handled by the same rule engine and produce the correct contraction axis.
3. scalar-gradient promotions used in biofilm/FPI Jacobians assemble without broadcast workarounds.
4. composite scalar gradients such as `grad(dalpha * coeff + alpha * dphi)` work by explicit linearity/product-rule expansion.
5. Python, JIT, and C++ backends use the same semantic rule layer.
6. `examples/debug/comparison_with_fenics.py` remains at machine precision.

## Verification

Verified now:

- Focused planner/algebra suite:
  - `tests/test_tensor_algebra_rules.py`
  - `tests/test_scalar_grad_dot_backend_parity.py`
  - result: `32 passed`
- Ghost/interface deformation validation:
  - `tests/ufl/test_ghost_deformation_validation.py`
  - result: `9 passed`
- Clean-cache targeted Fenics comparison:
  - `Seboldt Hdiv momentum (jac)` on `jit` and `cpp`
  - `Biofilm momentum (jac)` on `jit` and `cpp`
  - `Biofilm current total jacobian` on `jit` and `cpp`
  - `Biofilm total jacobian` on `jit` and `cpp`
  - result: all targeted terms match FEniCSx at machine precision in the fresh reruns
- Full clean-cache Python Fenics comparison:
  - log: `/home/bhatti/tmp/pycutfem_full_python_v8.log`
  - result: `253 passed / 0 failed`
- Full clean-cache JIT Fenics comparison:
  - log: `/home/bhatti/tmp/pycutfem_full_jit_v8.log`
  - result: `253 passed / 0 failed`

Live long-run review sessions:

- full Python Fenics comparison:
  - tmux session: `full_compare_python_v8`
  - log: `/home/bhatti/tmp/pycutfem_full_python_v8.log`
- full JIT Fenics comparison:
  - tmux session: `full_compare_jit_v8`
  - log: `/home/bhatti/tmp/pycutfem_full_jit_v8.log`
- full C++ Fenics comparison:
  - tmux session: `full_compare_cpp_v45`
  - log: `/home/bhatti/tmp/pycutfem_full_cpp_v45.log`
- whole-tests ghost/interface/facet subset:
  - tmux session: `pytest_ghost_interface_v8`
  - log: `/home/bhatti/tmp/pycutfem_pytest_ghost_v8.log`

Objective 6 is now complete:

- full clean-cache Python Fenics comparison:
  - log: `/home/bhatti/tmp/pycutfem_full_python_v9.log`
  - result: `253 passed / 0 failed`
- full clean-cache JIT Fenics comparison:
  - log: `/home/bhatti/tmp/pycutfem_full_jit_v9.log`
  - result: `253 passed / 0 failed`
- full clean-cache C++ Fenics comparison:
  - log: `/home/bhatti/tmp/pycutfem_full_cpp_v46.log`
  - result: `253 passed / 0 failed`
- ghost/interface/facet pytest subset:
  - log: `/home/bhatti/tmp/pycutfem_pytest_ghost_v9.log`
  - result: `176 passed, 3 skipped, 177 deselected`

## Notes

The important architectural decision is this:

Raw array shape is an implementation detail.
Tensor meaning is not.

This refactor succeeds only if tensor meaning becomes explicit and shared.

## Verification Notes

- Fresh long review runs now use `TMPDIR=/home/bhatti/tmp` and cache roots under `/home/bhatti/tmp/...` so they avoid the machine's full `/tmp` tmpfs.
- The maintained branch audit is now clean for the actual lowering dispatch blocks:
  - JIT `dot`: only planner-owned branches plus the intentional scalar-dot delegation to `_mul_scalar_vector`
  - JIT `*`: planner-owned branches only
  - C++ `dot`: planner-owned branches only
  - C++ `*`: planner-owned branches only
- Final C++ facet closeout:
  - sided `grad(Function)` on cut skeleton residuals must not apply the restriction mask a second time after loading the side-specific coefficient block
  - regression coverage now includes restricted scalar CIP residual parity on `dCutSkeleton`
