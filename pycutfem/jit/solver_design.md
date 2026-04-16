# C++ Shared-Loop Volume Assembly Design

This note describes the repo-level design for the C++ grouped volume assembly
path in `pycutfem.jit`.

## Goal

For compatible plain `dx` integrals:

- one element loop,
- one quadrature loop,
- multiple integral bodies executed inside that shared loop,
- no repeated full kernel passes over the same element set and quadrature rule.

The target is the same loop structure used by Kratos-style local operators:
shared geometry and basis setup, then multiple local contributions accumulated
inside one fused element/QP traversal.

## Scope

Phase 1 covers only:

- standard whole-cell volume integrals,
- no level-set cut logic,
- no facet/interface/ghost kernels,
- same element subset,
- same quadrature degree after geometric inflation,
- same deformation and measure metadata signature.

## Design Invariants

1. Field order is frozen by `MixedElement.field_names`.
   `active_fields`, `gdofs_map`, basis tables, and coefficient slices must all
   follow the same order.

2. Integral order is preserved.
   Grouped execution must evaluate integral bodies in original source order.

3. The shared-loop path is a backend execution strategy, not a symbolic
   correctness trick.
   The robust design is ordered grouped execution, not symbolic summation of a
   giant monolithic integrand.

4. Global assembly and local-volume assembly must use the same grouping logic.
   Otherwise the Kratos-style local operator path remains slower than the
   global path.

## Current Architecture

The repo-level pieces are:

- `pycutfem.jit._plan_integral_execution_units(...)`
  This groups compatible integrals into execution units while preserving source
  order.

- `pycutfem.jit._FusedIntegralGroup`
  This is the execution unit for grouped C++ plain-volume kernels.

- `pycutfem.jit.cpp_backend.compile_backend_cpp_group(...)`
  This compiles multiple original IR programs into one kernel with one shared
  element/QP loop.

- `pycutfem.ufl.compilers.FormCompiler.assemble_volume_local_contributions(...)`
  This now routes compatible C++ plain-volume local batches through the same
  grouped execution planner rather than assembling integrals one by one.

## Whole-Equation Direct Assembly Design

The next durable step is to plan assembly at the `Equation(a, L)` level rather
than treating the Jacobian and residual as two unrelated top-level passes.

The robust design is:

- one equation plan with explicit request flags:
  - `need_vector`
  - `need_matrix`
- residual-first scheduling:
  - assemble `L` first,
  - assemble `a` only when requested,
  - keep residual-only Newton evaluations off the matrix path.
- one execution signature per compatible measure group:
  - same domain,
  - same entity subset,
  - same quadrature degree,
  - same deformation/measure metadata.
- one future C++ target per execution signature:
  - gather element-local coefficient blocks once,
  - enter one element loop,
  - enter one quadrature loop,
  - accumulate both residual and Jacobian bodies inside that shared traversal.

Field-order safety rules for the whole-equation path:

1. The mixed-field order remains frozen by `MixedElement.field_names`.
2. Residual and Jacobian outputs must use the same `gdofs_map` ordering.
3. Grouped execution may reorder only by explicit plan stage:
   `L` before `a`; never by ad hoc dictionary ordering.
4. Within each stage, source integral order is preserved.

Output targets should evolve in two phases:

- Phase A:
  keep local `K_elem/F_elem` kernels, but drive them from one equation plan and
  allow residual-only requests to skip the matrix path entirely.
- Phase B:
  replace Python scatter with a direct sparse target:
  - prebuilt CSR structure,
  - per-element CSR slot maps,
  - C++ writes directly to `values` and `rhs`.

If the direct-CSR phase underperforms, investigate in this order:

1. repeated coefficient gather across residual/Jacobian bodies,
2. CSR slot lookup cost,
3. Python-side sparse allocation/reallocation,
4. thread-safety strategy:
   coloring vs thread-local reduction vs atomics,
5. memory traffic from writing sparse values in poor row order.

## Success Criteria

1. Correctness:
   grouped C++ local/global volume assembly matches the unfused path.

2. Structural:
   generated grouped C++ code contains one element loop and one quadrature loop
   per compatible group.

3. Performance:
   grouped local/global volume assembly materially outperforms the old
   per-integral path on forms with coarse volume groups.

4. End-to-end:
   the grouped C++ path should move the runtime toward Kratos-style operator
   costs rather than only improving compile-time bookkeeping.

## Failure Analysis

If the runtime target is not met, check in this order:

1. grouped vs unfused correctness,
2. generated loop structure,
3. inlining and scratch reinitialization overhead,
4. coefficient gather / scatter overhead,
5. linear solve cost after assembly.
