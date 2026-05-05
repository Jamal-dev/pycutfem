# Tensor Semantics for UFL Development

This note records the tensor model that `pycutfem.ufl.tensor_algebra` now enforces.
It is the intended source of truth for future work in the `ufl` package.

## Core Separation

Every expression is described by three independent layers:

1. `TensorSignature`
   - algebraic free axes only
   - no basis axes
   - no field-name guessing

2. `BasisSignature`
   - test/trial ownership only
   - basis axes never participate in tensor contraction

3. `ProvenanceSignature`
   - where the expression came from
   - distinguishes vector-field components from scalar-gradient derivative channels

Storage layout is a fourth concern and is handled by `StorageSpec`.

## Canonical Algebra Rules

### Rank Model

- scalar: rank 0
- vector field value: rank 1
- `grad(scalar)`: rank 1
- `grad(vector)`: rank 2
- `grad(grad(scalar))`: rank 2
- `grad(grad(vector))`: rank 3

`grad` appends one physical free axis.

### Dot

`dot(lhs, rhs)` contracts:

- the last free physical axis of `lhs`
- with the first free physical axis of `rhs`

Result free axes are:

- `lhs.free_axes[:-1] + rhs.free_axes[1:]`

This matches UFL / FEniCS semantics.

Important consequences:

- `dot(v, grad(a))` contracts the vector axis of `v` with the first free axis of `grad(a)`
- `dot(grad(v), v)` contracts the last free axis of `grad(v)` with the first free axis of `v`
- `dot(H, c)` for a Hessian contracts the last free axis of `H`
- `dot(c, H)` contracts the first free axis of `H`

No dot rule is allowed to inspect raw storage shape first and decide the contraction from that.

### Inner

`inner(lhs, rhs)` is full Frobenius contraction over all free physical axes after tensor-shape compatibility is established.

For basis-carrying tensors this means:

- basis ownership is preserved for assembly
- free physical axes are fully contracted
- basis axes are never contracted by `inner`

### Multiplication

`*` has only two meanings:

- scalar scaling if either operand is algebraically scalar
- tensor product otherwise

Important special case for basis carriers:

- scalar test-row `*` scalar trial-row is not scalar scaling
- it is a mixed outer product and must lower to a true local matrix `(n_test, n_trial)`

Examples:

- `a * I_ij -> c_ij`
- `a_i * b_j -> c_ij`
- `a_i * b_jk -> c_ijk`

`*` is not a hidden contraction operator.
Any contraction must go through `dot`, `inner`, `tr`, or explicit derived operators.

### Transpose

`transpose` is only defined for algebraic rank-2 tensors.

This means:

- transposing a matrix is valid
- transposing a rank-1 vector is not a tensor operation in this system
- backend storage views may reorder rank-1 arrays for code generation, but that is not a semantic transpose

## Basis Rules

- scalar basis value: compatibility storage may still appear as `(1, n)`
- scalar basis gradient: canonical storage is `(d, n)`
- vector basis value: `(k, n)`
- vector basis gradient: `(k, n, d)`
- mixed scalar: `(n_test, n_trial)`
- mixed rank-1: `(free, n_test, n_trial)`
- mixed rank-2: `(free0, n_test, n_trial, free1)`

Canonical mixed ordering is always:

- `test`
- then `trial`

## Compile-Time Shape Planning

Backends must not rediscover tensor meaning from emitted arrays.

The shared planner provides:

- `StorageSpec`
  - exact emitted shape when known
  - free-axis positions
  - basis-axis positions
  - canonical free-then-basis shape

- `KernelValueSpec`
  - backend stack kind
  - role
  - emitted shape
  - mixed layout
  - semantic flags
  - `ExpressionMeta`

This is the compile-time shape-prediction path.

`-1` extents are allowed only for genuinely dynamic compatibility paths.
They must not be used as part of algebraic reasoning.

When a backend helper already emits a concrete collapsed runtime shape, that
shape must be preserved. Semantic wrapper shapes such as a fake leading
singleton may not be reintroduced later, because sum/product lowering would
otherwise index or broadcast the wrong runtime axes.

## Mixed Layout

Mixed basis tensors may carry one of two layouts:

- `component_first`
- `component_last`

The layout is chosen by the shared planner from contraction order.

Examples:

- `dot(grad(test), trial)` and `dot(trial, grad(test))` are not the same mixed layout
- the emitted backend helper may differ
- the algebraic source remains the same planner result

## Hessian Rule After One Contraction

After one dot contraction against a vector, a Hessian no longer behaves like a full Hessian object.

- rank-1 result: vector-like
- rank-2 result: gradient-like / matrix-like

It must not be sent back into plain mass-matrix or full-Hessian fallback code just because the stored object is 2D.

## Backend Contract

Python, JIT, and C++ may choose different helper functions, but they must agree on:

- tensor rank
- contraction axes
- basis ownership
- mixed layout
- emitted shape
- provenance metadata

Backend-local branches are allowed only for helper emission.
They are not allowed to redefine tensor algebra.

## Audit Rule

For the maintained JIT/C++ lowering sections, the branch audit is:

- `dot` branches must route through planner-owned metadata:
  - `dot_lowering`
  - `dot_plan`
  - `dot_value_spec`
  - `dot_case`
  - backend push helpers such as `push_dot_result`
  - scalar-dot delegation through `_mul_scalar_vector` is allowed, because in
    this DSL `dot(scalar, X)` and `dot(X, scalar)` are defined as scalar
    multiplication and therefore intentionally reuse the shared product planner
- `*` branches must route through planner-owned metadata:
  - `product_lowering`
  - `product_case`
  - `product_value_spec`
  - backend push helpers such as `push_bin`
  - JIT push helpers such as `_push_product_result`

This means helper selection may still branch, but result kind, role, layout,
shape, and semantic flags must come from the shared planner.

## Practical Checks for Future Changes

When adding a new operator or helper:

1. infer `TensorSignature` first
2. plan storage with `StorageSpec`
3. emit backend code from `KernelValueSpec`
4. do not infer contraction direction from raw shape
5. do not use transpose as a fix for rank-1 ambiguity
6. keep basis axes out of tensor contraction logic
7. keep provenance separate from algebraic rank

If a new branch needs `shape[0] == 1` to decide tensor meaning, the design is probably being violated.
