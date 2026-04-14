# Runtime Operator Layer

## What This Layer Is

`pycutfem.operators` is the backend-neutral runtime hook layer for method-specific
discrete logic that does not belong inside the ordinary continuous UFL weak
form.

The core idea is:

- keep continuous PDE terms in UFL,
- keep mutable history/state in `pycutfem.state`,
- keep method-specific discrete updates in runtime operators,
- let the same solver lifecycle drive `python`, `jit`, and `cpp` backends.

This avoids building a second symbolic assembly compiler before it is actually
needed.

## What Problem It Solves

Some methods are not just "one more weak form term".

Typical examples are:

- lagged subscale predictions,
- history-dependent stabilization,
- reduced-space load shifts,
- per-step or per-iteration coefficient refreshes,
- discrete algebra that depends on the current iterate but should not change
  kernel structure.

For those cases, forcing everything into pure UFL usually becomes awkward:

- the math may be naturally defined as a local algorithm, not as a single weak
  form,
- the data may be lagged state, not unknown fields,
- the update may need to happen before every residual/Jacobian assembly,
- the result may be a reduced-space correction instead of another volume term.

The runtime operator layer gives one clean place for that logic.

## Relationship To The State Layer

The operator layer and the state layer are intentionally separate.

The state layer is the data plane:

- `StateRegistry`
- `CellStateField`
- `CellStateCoefficient`

It provides mutable runtime values that can be read inside the existing UFL/JIT
pipeline without recompiling kernels when values change.

Storage rule:

- scalar cell state should be stored as shape `(n_cells,)`,
- vector cell state should be stored as shape `(n_cells, ncomp)`,
- rank-2 tensor cell state should be stored as shape `(n_cells, n0, n1)`.

Do not store scalar state as `(n_cells, 1)` unless you explicitly want a
length-1 vector. The backends treat shape as part of the operator semantics.

The operator layer is the control plane:

- decide when state is updated,
- decide when reduced systems are post-processed,
- decide how step acceptance/rejection affects method-specific history.

This split is important. A discrete method usually needs both:

- state storage,
- lifecycle rules for updating that state.

## Current Design

The runtime operator interface is defined in `base.py`.

Available hooks:

- `bind(solver)`
- `on_step_begin(...)`
- `before_assembly(...)`
- `after_assembly(...)`
- `on_step_accept(...)`
- `on_step_reject(...)`

`OperatorManager` composes multiple operators in a stable order.

The solver integration is intentionally narrow:

- `before_assembly` runs before every reduced residual/Jacobian assembly,
- `after_assembly` can modify the reduced matrix/vector after backend assembly,
- step hooks follow the pseudo-time step lifecycle.

Because the hooks sit above the backend assembly path, they work uniformly for:

- `backend="python"`
- `backend="jit"`
- `backend="cpp"`

## Local Contribution Design

This package now includes a generic local-contribution operator API.

The purpose is narrow:

- let an operator assemble explicit element-local `K_elem` / `F_elem`,
- keep the method-specific algebra outside core,
- reuse solver-owned scatter/reduction machinery,
- run the same operator contract under `python`, `jit`, and `cpp`.

The intended split is:

- core owns the local-operator contract,
- core owns batch/result normalization and scatter helpers,
- examples/applications own the actual local kernel math.

This is not meant to replace UFL.

UFL still owns continuous weak forms and their ordinary backend compilation.
The local-contribution operator only covers the complementary case where a
method has explicit discrete local algebra that should be assembled and
scattered after the continuous part.

That means the sustainable pattern becomes:

1. assemble the continuous part through UFL,
2. evaluate method-specific local batches through a runtime operator,
3. scatter those local blocks into the reduced or global system.

The first concrete core APIs are:

- `LocalAssemblyWorkset`
- `LocalAssemblyResult`
- `LocalStateUpdate`
- `LocalAssemblyOperator`
- `CallbackLocalAssemblyOperator`
- `FusedLocalAssemblyOperator`
- `CallbackFusedLocalAssemblyOperator`
- `SymbolicQuadratureStateUpdateSpec`
- `SymbolicFusedLocalAssemblyOperator`
- `PointwiseQuadratureWorkset`
- `PointwiseQuadratureResult`
- `PointwiseQuadratureOperator`
- `CallbackPointwiseQuadratureOperator`
- `SymbolicPointwiseNewtonOperator`

The solver now exposes both:

- `NewtonSolver.scatter_element_contribs_reduced(...)`
- `NewtonSolver.scatter_element_contribs_full(...)`

Example 2 uses this split in two ways:

- `examples/NIRB/dvms/local_operator.py` keeps the Kratos-style
  local block definitions in the example while using the repo-level
  symbolic local-batch assembly contract for matrix/vector evaluation and scatter.
- `examples/NIRB/dvms/runtime_operator.py` uses the general symbolic/UFL quadrature path
  through `SymbolicPointwiseNewtonOperator` for the nonlinear predicted-subscale refresh.

## First-Class Fused Local Operators

The operator layer now also supports a first-class fused local/state contract.

This closes an important architectural gap between:

- local matrix/vector assembly, and
- hidden runtime state updates (for example quadrature-level DVMS subscales).

The key idea is that one runtime operator can now:

1. evaluate backend-native local element contributions,
2. evaluate backend-native hidden-state updates,
3. stage those updates on runtime state fields,
4. scatter the resulting local `K_elem` / `F_elem` into the solver-owned system,
5. commit or rollback step-persistent state through the normal solver lifecycle.

The new core pieces are:

- `LocalStateUpdate`
  - one state-field update attached to a local assembly result
- `FusedLocalAssemblyOperator`
  - generic runtime operator that owns both local assembly and hidden-state transaction hooks
- `CallbackFusedLocalAssemblyOperator`
  - fused operator backed by user-provided python/jit/cpp kernels
- `SymbolicQuadratureStateUpdateSpec`
  - symbolic quadrature-state update definition
- `SymbolicFusedLocalAssemblyOperator`
  - backend-neutral symbolic fused operator that combines:
    - local FE assembly from `FormCompiler.assemble_local_contributions(...)`
    - quadrature-state updates from `FormCompiler.evaluate_volume_expressions_on_quadrature(...)`

The lifecycle is:

1. `on_step_begin`
   - reset iteration-persistent staged state
2. `after_assembly`
   - run the backend-specific fused local operator
   - apply staged hidden-state updates
   - scatter local matrix/vector contributions
3. `on_step_accept`
   - commit step-persistent staged state
4. `on_step_reject`
   - rollback step-persistent staged state

This is the first place in the repo where hidden-state management is solver-owned
instead of being hand-written inside a benchmark driver.

## What This Gives Us

This fused contract is important because it lets advanced methods keep:

- one runtime operator object,
- one solver lifecycle,
- one backend-dispatch contract,
- and no benchmark-specific Python-side state bookkeeping.

In particular, methods with quadrature-level hidden state can now be expressed as:

```python
result = LocalAssemblyResult(
    K_elem=...,
    F_elem=...,
    state_updates=(
        LocalStateUpdate(field=qfield, values=z_new, entity_ids=eids, staged=True),
    ),
)
```

and the solver/operator layer will take care of:

- staging the hidden state,
- committing or rolling it back with the step lifecycle,
- and scattering the local algebra.

## Compiler-Level Condensed Local Systems

The deeper compiler support is now available too.

The UFL/compiler layer exposes:

- `CondensedQuadratureLocalSystem`

This is a first-class compiler IR object for local hidden-state elimination.
It describes:

- a base FE form or equation,
- left coupling blocks `C`,
- right coupling blocks `B`,
- hidden-state Jacobian blocks `G`,
- hidden-state residual blocks `r`,
- and the quadrature layout used for the hidden-state algebra.

The compiler then assembles the condensed local system directly as:

- `K_hat = K_base + sign * C^T G^{-1} B`
- `F_hat = F_base + sign * C^T G^{-1} r`

with `sign = -1` recovering the usual Schur-complement elimination.

This path is native on all three backends:

- `python`: NumPy backend
- `jit`: Numba backend
- `cpp`: Eigen/pybind11 backend

The symbolic local operators can now also call this path through direct reduced
scatter, so condensed local systems no longer need to round-trip through:

- compiler batch assembly,
- runtime result materialization,
- operator-side merge,
- and a second scatter step.

Instead, the compiler can now assemble and inject local condensed
contributions directly into the solver-owned reduced system.

The important boundary is:

- hidden-state local algebra and Schur-complement condensation are now backend-native and compiler-owned
- benchmark code no longer needs Python-side post-processing to build condensed `K_elem/F_elem`

What is still not fully collapsed into one single generated FE kernel is the
orchestration boundary between:

- compiled FE local assembly,
- compiled/local quadrature-state updates,
- backend-native Schur-complement condensation,
- and final reduced/global scatter

That means the architecture gap is now much smaller, but not mathematically
larger than it needs to be for production use.

The remaining future optimization target for a true Kratos-style single native
local operator is:

- one compiler pass that emits FE assembly plus local-state elimination plus
  scatter in one fused generated kernel

So the repo now has the important ingredients in core:

- first-class compiler IR for hidden-state elimination,
- local block algebra with native Schur complements,
- backend parity on `python` / `jit` / `cpp`,
- and solver-owned hidden-state transactions.

Design rule:

- core stays generic and backend-neutral,
- method-specific operators stay in the example or application.

## Phase Targets

- [x] Phase 1: add the generic local-contribution API and Python reference path.
- [x] Phase 2: add `jit` / Numba backend dispatch and parity coverage.
- [x] Phase 3: add `cpp` backend dispatch and parity coverage.
- [x] Phase 4: migrate Example 2 DVMS/local discrete usage onto the new API and verify the end-to-end path.
- [x] Phase 5: add a generic symbolic pointwise quadrature-update operator and migrate the DVMS predicted-subscale solve onto it.

## What This Is Not

This is not yet a second symbolic local-assembly language.

There is still no second symbolic operator IR or UFL replacement in this
layer. That is deliberate.

Right now the repo only needs:

- mutable state coefficients,
- solver-time lifecycle hooks,
- backend-neutral reduced-system post-processing,
- and a narrow local-contribution API for explicit element-local algebra.

That is enough to streamline Example 2 without duplicating the existing UFL
compiler stack.

If a future method truly needs a richer element-local language that cannot be
expressed through:

- UFL terms reading state coefficients, plus
- runtime hook callbacks,

then this package is the right place to grow a richer discrete-operator API.

Current note for this repo:

- the solver now exposes
  `NewtonSolver.scatter_element_contribs_reduced(...)` as the stable runtime
  entry point for operators that already assembled their own element-local
  blocks and only need reduced-system scattering.
- that is enough for methods such as the Example 2 DVMS port to inject exact
  discrete local contributions without inventing a separate operator compiler.

## Sustainable Rule For Core Code

Core code should remain generic.

Core owns:

- operator lifecycle hooks,
- operator composition,
- state storage abstractions,
- backend-neutral solver integration.

Examples or applications own:

- method-specific formulas,
- history update rules,
- problem-specific operator classes.

This keeps repository-level architecture clean while still making advanced
methods possible.

## How It Helps With Discrete Operators

In practice, many "discrete operators" can be split into two parts:

1. Continuous part:
   assembled from UFL as usual.
2. Discrete part:
   computed from lagged state and current iterate, then injected through
   runtime hooks.

That pattern covers a large class of methods:

- residual-based stabilization with stored history,
- local predictor-corrector state updates,
- external/interface load corrections,
- algebraic subscale models,
- projection-based auxiliary fields.

The key point is that the continuous and discrete parts can coexist cleanly in
one solve without inventing a new compiler interface.

## Example Pattern

Below is the intended pattern for mixing a continuous operator with a discrete
operator.

```python
import numpy as np

from pycutfem.operators import RuntimeOperator
from pycutfem.state import StateRegistry
from pycutfem.solvers.nonlinear_solver import NewtonSolver


# 1. Register mutable state once.
state = StateRegistry()
u_ss_old = state.register_cell(
    "my_method_old_subscale",
    n_cells=mesh.n_elements,
    tensor_shape=(2,),
    persistence="step",
)
u_ss_pred = state.register_cell(
    "my_method_predicted_subscale",
    n_cells=mesh.n_elements,
    tensor_shape=(2,),
    persistence="iteration",
)
old_mass_residual = state.register_cell(
    "my_method_old_mass_residual",
    n_cells=mesh.n_elements,
    tensor_shape=(),
    persistence="step",
)


# 2. Use the state inside the ordinary UFL form.
u_ss_old_coeff = u_ss_old.coefficient()
u_ss_pred_coeff = u_ss_pred.coefficient()
old_mass_residual_coeff = old_mass_residual.coefficient()

residual = (
    galerkin_residual
    + dot(u_ss_pred_coeff, test_v) * dx()
    + div(test_v) * old_mass_residual_coeff * dx()
)


# 3. Update the state through a runtime operator.
class MyDiscreteOperator(RuntimeOperator):
    def __init__(self, *, state_registry, u_k, p_k):
        self.state_registry = state_registry
        self.u_k = u_k
        self.p_k = p_k

    def before_assembly(self, *, solver, coeffs, need_matrix):
        del solver, coeffs, need_matrix
        predicted = compute_local_subscale_prediction(
            u=self.u_k,
            p=self.p_k,
            old_subscale=self.state_registry["my_method_old_subscale"].values,
        )
        self.state_registry["my_method_predicted_subscale"].assign(predicted)

    def on_step_accept(self, *, solver, functions, prev_functions, aux_functions, step, step_no, t, dt, bcs):
        del solver, functions, prev_functions, aux_functions, step, step_no, t, dt, bcs
        self.state_registry["my_method_old_subscale"].assign(
            self.state_registry["my_method_predicted_subscale"].values
        )


solver = NewtonSolver(
    residual_form=residual,
    jacobian_form=jacobian,
    dof_handler=dh,
    mixed_element=me,
    bcs=bcs,
    bcs_homog=bcs_homog,
    backend="cpp",
    operators=[MyDiscreteOperator(state_registry=state, u_k=u_k, p_k=p_k)],
)
```

What happens here:

- the continuous Galerkin part is still plain UFL,
- the lagged/discrete state is stored once in the registry,
- the operator updates runtime state before each assembly,
- the backend sees changed coefficient values but the same kernel structure,
- accepted steps promote iteration state into step history.

## Example 2 Mapping

Example 2 uses the same pattern.

Continuous part:

- Bossak/ALE residual and Jacobian in UFL,
- DVMS history read through `CellStateCoefficient`s.

Discrete part:

- `_FluidDVMSSolverOperator`
  updates the predicted subscale before each assembly,
- `_ReducedResidualShiftOperator`
  injects the interface point-load correction in reduced space.

That keeps the example-specific method logic in the example while core code only
provides the general runtime mechanism.

## When To Extend This Layer Further

The next step should only be taken if a real method needs more than the current
hook/state pattern.

A stronger operator abstraction becomes justified when all of these are true:

- the method is naturally defined as explicit local algebra,
- the local algebra should be reusable across problems,
- the operator cannot be represented cleanly as UFL plus runtime coefficient
  updates,
- post-processing reduced vectors is not enough.

At that point, a true local discrete-operator API or IR can be added here.

Until then, this runtime operator layer is the smaller and cleaner design.
