from __future__ import annotations

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.operators import SymbolicPointwiseNewtonOperator
from pycutfem.ufl.expressions import Function, VectorFunction

from .helpers import (
    _bossak_coefficients,
    _kratos_dvms_current_element_size_array,
    _kratos_dvms_current_element_size_coefficient,
)
from .state import FluidDVMSState
from .symbolics import build_fluid_dvms_predictor_iteration_symbolics
from .update import _clear_fluid_dvms_oss_projections
from .update import _update_fluid_dvms_predicted_subscale
from .update import _update_fluid_dvms_oss_projections


def _predictor_element_char_length_coefficient(state: FluidDVMSState, mesh: Mesh, dh: DofHandler, d_mesh: VectorFunction):
    cache = getattr(state, "_predictor_element_char_length_coefficient_cache", None)
    key = (int(id(mesh)), int(id(dh)), int(id(d_mesh)))
    if isinstance(cache, dict) and key in cache:
        return cache[key]
    coeff = _kratos_dvms_current_element_size_coefficient(mesh, dh, d_mesh)
    if not isinstance(cache, dict):
        cache = {}
        state._predictor_element_char_length_coefficient_cache = cache
    cache[key] = coeff
    return coeff


class FluidDVMSSolverOperator(SymbolicPointwiseNewtonOperator):
    """Runtime operator that refreshes the DVMS predicted subscale before assembly."""

    def __init__(
        self,
        *,
        state: FluidDVMSState,
        dh: DofHandler,
        mesh: Mesh,
        u_k: VectorFunction,
        u_prev: VectorFunction,
        a_prev: VectorFunction,
        a_curr: VectorFunction | None = None,
        p_k: Function,
        d_mesh: VectorFunction,
        d_prev: VectorFunction,
        d_prev2: VectorFunction | None = None,
        mesh_v: VectorFunction | None = None,
        mesh_v_prev: VectorFunction | None = None,
        mesh_a_prev: VectorFunction | None = None,
        rho_f: float,
        mu_f: float,
        dt: float,
        bossak_alpha: float,
        dynamic_tau: float,
        # Kratos DVMS hard-codes ten Newton iterations; non-converged
        # quadrature-point predictors are zeroed in the update routine.
        max_iterations: int = 10,
        rel_tol: float = 1.0e-14,
        abs_tol: float = 1.0e-14,
        use_oss: bool = False,
    ) -> None:
        self.state = state
        self.dh = dh
        self.mesh = mesh
        self.u_k = u_k
        self.u_prev = u_prev
        self.a_prev = a_prev
        self.a_curr = a_curr
        self.p_k = p_k
        self.d_mesh = d_mesh
        self.d_prev = d_prev
        self.d_prev2 = d_prev2
        self.mesh_v = mesh_v
        self.mesh_v_prev = mesh_v_prev
        self.mesh_a_prev = mesh_a_prev
        self.rho_f = float(rho_f)
        self.mu_f = float(mu_f)
        self.dt = float(dt)
        self.bossak_alpha = float(bossak_alpha)
        self.dynamic_tau = float(dynamic_tau)
        self.max_iterations = max(int(max_iterations), 1)
        self.rel_tol = float(rel_tol)
        self.abs_tol = float(abs_tol)
        self.use_oss = bool(use_oss)

        dt_value = max(self.dt, 1.0e-14)
        bossak = _bossak_coefficients(alpha=self.bossak_alpha, dt=dt_value)
        self.h_coefficient = _predictor_element_char_length_coefficient(self.state, self.mesh, self.dh, self.d_mesh)
        self.symbolics = build_fluid_dvms_predictor_iteration_symbolics(
            u_k=self.u_k,
            u_prev=self.u_prev,
            a_prev=self.a_prev,
            a_curr=self.a_curr,
            p_k=self.p_k,
            d_mesh=self.d_mesh,
            d_prev=self.d_prev,
            d_prev2=self.d_prev2,
            mesh_v=self.mesh_v,
            mesh_v_prev=self.mesh_v_prev,
            mesh_a_prev=self.mesh_a_prev,
            dt=dt_value,
            bossak_ma0=float(bossak["ma0"]),
            bossak_ma2=float(bossak["ma2"]),
            bossak_alpha=float(bossak["alpha"]),
            rho=self.rho_f,
            mu=self.mu_f,
            h=self.h_coefficient,
            predicted_subscale=self.state.coefficient("predicted_subscale_velocity"),
            old_subscale=self.state.coefficient("old_subscale_velocity"),
            momentum_projection=self.state.coefficient("momentum_projection"),
        )
        super().__init__(
            dof_handler=self.dh,
            unknown_field=self.state._state_fields["predicted_subscale_velocity"],
            residual_expr=self.symbolics.fixed_point_residual,
            jacobian_expr=self.symbolics.linearization_matrix,
            quadrature_order=int(self.state.quadrature_order),
            max_iterations=self.max_iterations,
            rel_tol=self.rel_tol,
            abs_tol=self.abs_tol,
            failure_mode="zero",
        )

    def before_assembly(self, *, solver, coeffs, need_matrix: bool) -> None:
        del coeffs
        # Kratos updates the hidden predicted subscale in
        # InitializeNonLinearIteration(), i.e. once at the start of each
        # nonlinear build/solve cycle. It does not refresh that hidden state
        # again for the residual-only checks that happen after UpdateDatabase()
        # and FinalizeNonLinIteration(). Re-refreshing here on
        # need_matrix=False advances the hidden DVMS state machine one extra
        # time per Newton iteration and perturbs the carried fluid state.
        if not bool(need_matrix):
            return
        if not bool(self.use_oss):
            _clear_fluid_dvms_oss_projections(self.state)
        self.h_coefficient.values[...] = _kratos_dvms_current_element_size_array(self.mesh, self.dh, self.d_mesh)
        _update_fluid_dvms_predicted_subscale(
            state=self.state,
            dh=self.dh,
            mesh=self.mesh,
            u_k=self.u_k,
            u_prev=self.u_prev,
            a_prev=self.a_prev,
            a_curr=self.a_curr,
            p_k=self.p_k,
            d_mesh=self.d_mesh,
            d_prev=self.d_prev,
            d_prev2=self.d_prev2,
            mesh_v=self.mesh_v,
            mesh_v_prev=self.mesh_v_prev,
            mesh_a_prev=self.mesh_a_prev,
            rho_f=self.rho_f,
            mu_f=self.mu_f,
            dt=self.dt,
            bossak_alpha=self.bossak_alpha,
            dynamic_tau=self.dynamic_tau,
            max_iterations=self.max_iterations,
            rel_tol=self.rel_tol,
            abs_tol=self.abs_tol,
            backend=str(getattr(solver, "backend", "python")),
            use_oss=bool(self.use_oss),
        )

    def after_nonlinear_update(self, *, solver, functions) -> None:
        del functions
        if not bool(self.use_oss):
            _clear_fluid_dvms_oss_projections(self.state)
            return
        _update_fluid_dvms_oss_projections(
            state=self.state,
            dh=self.dh,
            mesh=self.mesh,
            u_k=self.u_k,
            p_k=self.p_k,
            d_mesh=self.d_mesh,
            d_prev=self.d_prev,
            d_prev2=self.d_prev2,
            mesh_v=self.mesh_v,
            mesh_v_prev=self.mesh_v_prev,
            mesh_a_prev=self.mesh_a_prev,
            rho_f=self.rho_f,
            dt=self.dt,
            bossak_alpha=self.bossak_alpha,
        )

    def on_step_accept(
        self,
        *,
        solver,
        functions,
        prev_functions,
        aux_functions,
        step: int,
        step_no: int,
        t: float,
        dt: float,
        bcs,
    ) -> None:
        del solver, functions, prev_functions, aux_functions, step, step_no, t, dt, bcs


def build_fluid_dvms_predictor_pointwise_operator(
    *,
    state: FluidDVMSState,
    dh: DofHandler,
    mesh: Mesh,
    u_k: VectorFunction,
    u_prev: VectorFunction,
    a_prev: VectorFunction,
    a_curr: VectorFunction | None = None,
    p_k: Function,
    d_mesh: VectorFunction,
    d_prev: VectorFunction,
    d_prev2: VectorFunction | None = None,
    mesh_v: VectorFunction | None = None,
    mesh_v_prev: VectorFunction | None = None,
    mesh_a_prev: VectorFunction | None = None,
    rho_f: float,
    mu_f: float,
    dt: float,
    bossak_alpha: float,
    dynamic_tau: float,
    max_iterations: int = 10,
    rel_tol: float = 1.0e-14,
    abs_tol: float = 1.0e-14,
    use_oss: bool = False,
) -> FluidDVMSSolverOperator:
    cache = getattr(state, "_predictor_pointwise_operator_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        state._predictor_pointwise_operator_cache = cache
    key = (
        int(id(dh)),
        int(id(mesh)),
        int(id(u_k)),
        int(id(u_prev)),
        int(id(a_prev)),
        int(id(a_curr)) if a_curr is not None else -1,
        int(id(p_k)),
        int(id(d_mesh)),
        int(id(d_prev)),
        int(id(d_prev2)) if d_prev2 is not None else -1,
        int(id(mesh_v)) if mesh_v is not None else -1,
        int(id(mesh_v_prev)) if mesh_v_prev is not None else -1,
        int(id(mesh_a_prev)) if mesh_a_prev is not None else -1,
        float(rho_f),
        float(mu_f),
        float(dt),
        float(bossak_alpha),
        float(dynamic_tau),
        int(max_iterations),
        float(rel_tol),
        float(abs_tol),
        bool(use_oss),
    )
    cached = cache.get(key)
    if isinstance(cached, FluidDVMSSolverOperator):
        return cached
    operator = FluidDVMSSolverOperator(
        state=state,
        dh=dh,
        mesh=mesh,
        u_k=u_k,
        u_prev=u_prev,
        a_prev=a_prev,
        a_curr=a_curr,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        d_prev2=d_prev2,
        mesh_v=mesh_v,
        mesh_v_prev=mesh_v_prev,
        mesh_a_prev=mesh_a_prev,
        rho_f=float(rho_f),
        mu_f=float(mu_f),
        dt=float(dt),
        bossak_alpha=float(bossak_alpha),
        dynamic_tau=float(dynamic_tau),
        max_iterations=int(max_iterations),
        rel_tol=float(rel_tol),
        abs_tol=float(abs_tol),
        use_oss=bool(use_oss),
    )
    cache[key] = operator
    return operator


__all__ = [
    "FluidDVMSSolverOperator",
    "build_fluid_dvms_predictor_pointwise_operator",
]
