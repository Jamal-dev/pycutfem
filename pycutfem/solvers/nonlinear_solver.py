r"""
nonlinear_solver.py  –  High‑level Newton driver for pycutfem
===========================================================
A reusable, backend‑agnostic Newton solver that takes UFL residual and
Jacobian forms, compiles high‑performance kernels via *pycutfem.jit* once,
and then orchestrates Newton iterations, an optional Armijo line‑search,
pseudo‑time stepping and boundary‑condition handling.

The module is intentionally lightweight: all heavy lifting (element loops,
local matrices, gradients, etc.) happens in the Numba/LLVM kernels
emitted by `compile_backend`.  The Python layer below is mere glue and
control logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Callable, Optional, Tuple
import inspect

import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sp

from pycutfem.jit import compile_backend
from pycutfem.ufl.helpers_jit import _build_jit_kernel_args

# ----------------------------------------------------------------------------
#  Parameter dataclasses
# ----------------------------------------------------------------------------

@dataclass
class NewtonParameters:
    """Settings that govern a *single* Newton solve."""

    newton_tol: float = 1e-8            # ‖R‖_∞ convergence threshold
    max_newton_iter: int = 20           # hard cap on inner Newton iterations

    # Armijo back‑tracking line‑search
    line_search: bool = True
    ls_max_iter: int = 8
    ls_reduction: float = 0.5           # α ← β·α after reject
    ls_c1: float = 1.0e-4               # sufficient‑decrease parameter


@dataclass
class TimeStepperParameters:
    """Controls the pseudo‑time (or real time) advancement."""

    dt: float = 1.0                     # time‑step size
    max_steps: int = 1_000              # stop after this many steps
    steady_tol: float = 1e-6            # ‖ΔU‖_∞ < tol ⇒ steady
    theta: float = 1.0                  # 1.0 = backward Euler, 0.5 = CN
    stop_on_steady: bool = True         # early exit when steady‑state reached
    final_time: float = max_steps * dt  


@dataclass
class LinearSolverParameters:
    """Sparse linear solver settings (expandable for PETSc/Krylov/etc.)."""

    backend: str = "scipy"              # placeholder for future options
    tol: float = 1e-12
    maxit: int = 10_000


# ----------------------------------------------------------------------------
#  NewtonSolver class
# ----------------------------------------------------------------------------

class NewtonSolver:
    r"""Generic monolithic Newton solver with optional back‑tracking search.

    The solver is **problem agnostic**: you feed it UFL residual / Jacobian
    forms *and* the concrete `Function` / `VectorFunction` objects that own
    the DOF vectors.  Everything else (assembly, linear solve, Newton
    updates, line‑search, time stepping) is handled internally.
    """

    # ------------------------------------------------------------------
    #  Construction & JIT compilation
    # ------------------------------------------------------------------
    def __init__(
        self,
        residual_form,
        jacobian_form,
        *,
        dof_handler,
        mixed_element,
        bcs: List,                       # inhomogeneous BCs  (values added)
        bcs_homog: List,                 # homogeneous BCs    (rows/cols 0)
        newton_params: NewtonParameters = NewtonParameters(),
        lin_params: LinearSolverParameters = LinearSolverParameters(),
        quad_order: int = 6,
        preproc_cb: Optional[Callable[[List], None]] = None,
        postproc_cb: Optional[Callable[[List], None]] = None,
    ) -> None:
        self.dh = dof_handler
        self.me = mixed_element
        self.bcs = bcs
        self.bcs_homog = bcs_homog
        self.np = newton_params
        self.lp = lin_params
        self.pre_cb = preproc_cb
        self.post_cb = postproc_cb

        # Keep original forms handy (their .integrand is needed later)
        self._residual_form = residual_form
        self._jacobian_form = jacobian_form

        # 1) compile kernels ------------------------------------------------
        self._res_runner, self._res_ir = compile_backend(
            residual_form, dof_handler=self.dh, mixed_element=self.me
        )
        self._jac_runner, self._jac_ir = compile_backend(
            jacobian_form, dof_handler=self.dh, mixed_element=self.me
        )

        # 2) pre‑compute static arguments (geometry, basis tables, …) -------
        self.static_args = self._precompute_static_args(quad_order)

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------
    def solve_time_interval(
        self,
        *,
        functions: List,                    # current‑step unknowns  Uⁿ
        prev_functions: List,               # previous‑step snapshot Uⁿ⁻¹
        time_params: TimeStepperParameters = TimeStepperParameters(),
        aux_functions: Optional[Dict[str, "Function"]] = None,
    ) -> Tuple[np.ndarray, int, float]:
        r"""Advance the problem in *pseudo*‑time until steady‑state.

        Parameters
        ----------
        functions : list[Function | VectorFunction]
            Primary unknowns at the **current** time level *n*.  Updated
            in‑place by the solver.
        prev_functions : list[Function | VectorFunction]
            Snapshot from the previous time level *n‑1* (kept intact so the
            residual/Jacobian can access both states).
        time_params : TimeStepperParameters, optional
            Controls Δt, maximum number of steps and the steady‑state test.
        aux_functions : dict[str, Function], optional
            Additional coefficient fields (material properties, body forces,
            etc.) that appear in the forms but are *not* unknowns.

        Returns
        -------
        last_delta : ndarray
            Final Newton increment (can be used for external checks).
        n_steps : int
            Number of successful time steps.
        elapsed : float
            Wall‑clock time in seconds.
        """

        dh = self.dh
        dt = time_params.dt
        delta_U = np.zeros(dh.total_dofs)

        t_start = time.perf_counter()
        for step in range(time_params.max_steps):
            t_n = step * dt

            # Predictor: copy previous solution ---------------------
            for f, f_prev in zip(functions, prev_functions):
                f.nodal_values[:] = f_prev.nodal_values[:]

            # Time‑dependent BCs -----------------------------------
            bcs_now = self._freeze_bcs(self.bcs, t_n)
            dh.apply_bcs(bcs_now, *functions)

            # Newton loop -----------------------------------------
            delta_U = self._newton_loop(functions, prev_functions, aux_functions, bcs_now)

            # Steady‑state test -----------------------------------
            if (
                time_params.stop_on_steady
                and step > 0
                and np.linalg.norm(delta_U, ord=np.inf) < time_params.steady_tol
            ):
                break

            # Accept: promote current → previous ------------------
            for f_prev, f in zip(prev_functions, functions):
                f_prev.nodal_values[:] = f.nodal_values[:]
            if t_n + dt >= time_params.final_time:
                break

        elapsed = time.perf_counter() - t_start
        return delta_U, step + 1, elapsed

    # ------------------------------------------------------------------
    #  Newton iteration (internal)
    # ------------------------------------------------------------------
    def _newton_loop(self, funcs, prev_funcs, aux_funcs, bcs_now):
        dh = self.dh

        for it in range(self.np.max_newton_iter):
            if self.pre_cb is not None:
                self.pre_cb(funcs)

            current: Dict[str, "Function"] = {f.name: f for f in funcs}
            current.update({f.name: f for f in prev_funcs})
            if aux_funcs:
                current.update(aux_funcs)

            K_loc, _ = self._jac_runner(current, self.static_args)
            _, F_loc = self._res_runner(current, self.static_args)
            A, R = self._assemble_system(K_loc, F_loc)

            norm_R = np.linalg.norm(R, ord=np.inf)
            print(f"        Newton {it+1}: |R|_∞ = {norm_R:.2e}")
            if norm_R < self.np.newton_tol:
                # Return *time‑step* increment --------------------
                delta = np.hstack([
                    f.nodal_values - f_prev.nodal_values
                    for f, f_prev in zip(funcs, prev_funcs)
                ])
                return delta

            dU = self._solve_linear_system(A, -R)

            if self.np.line_search:
                dU = self._line_search(funcs, dU, current, norm_R, bcs_now)

            dh.add_to_functions(dU, funcs)
            dh.apply_bcs(bcs_now, *funcs)

            if self.post_cb is not None:
                self.post_cb(funcs)

        raise RuntimeError("Newton did not converge – adjust Δt or verify Jacobian.")

    # ------------------------------------------------------------------
    #  Static kernel arguments (geometry + basis tables, *not* coeffs)
    # ------------------------------------------------------------------
    def _precompute_static_args(self, quad_order: int) -> Dict[str, np.ndarray]:
        dh = self.dh
        me = self.me

        static_args: Dict[str, np.ndarray] = dh.precompute_geometric_factors(quad_order)
        static_args["gdofs_map"] = np.vstack([
            dh.get_elemental_dofs(e) for e in range(me.mesh.n_elements)
        ]).astype(np.int32)
        static_args["node_coords"] = dh.get_all_dof_coords()

        # Basis / gradient lookup tables -------------------------------
        basis_jac = _build_jit_kernel_args(
            self._jac_ir,
            self._jacobian_form.integrand,
            me,
            quad_order,
            dh,
            param_order=self._jac_runner.param_order,
        )
        basis_res = _build_jit_kernel_args(
            self._res_ir,
            self._residual_form.integrand,
            me,
            quad_order,
            dh,
            param_order=self._res_runner.param_order,
        )
        static_args.update(basis_jac)
        static_args.update(basis_res)

        # strip really dynamic blocks (u_k_loc, p_k_loc, …) ------------
        static_args = {k: v for k, v in static_args.items() if not k.endswith("_loc")}
        return static_args

    # ------------------------------------------------------------------
    #  Linear system & BC handling
    # ------------------------------------------------------------------
    def _assemble_system(self, K_loc, F_loc):
        r"""Assemble global matrix **A** and residual vector **R**.

        If *K_loc* has shape *(n_elem, n_loc, n_loc)* it is interpreted as the
        per‑element stiffness/Jacobian contribution.  If it has shape
        *(n_elem, n_loc)* (i.e. rank‑2), this signals that the caller is only
        interested in the **residual**.  In that case we **skip** the sparse
        matrix construction entirely – a huge speed‑up for the line‑search,
        which only needs \|R\|.
        """
        dh = self.dh
        n_total = dh.total_dofs

        # ---------------------------- assemble residual vector R --------
        R = np.zeros(n_total)
        for e, Fe in enumerate(F_loc):
            gdofs = dh.get_elemental_dofs(e)
            np.add.at(R, gdofs, Fe)

        # ---------------------------- fast path: residual only ----------
        if K_loc.ndim == 2:
            # Caller does not need the matrix (line‑search, diagnostics …)
            A = sp.csr_matrix((n_total, n_total))  # empty / zero
        else:
            # Full matrix assembly -------------------------------------
            n_elem, n_loc, _ = K_loc.shape
            data = np.zeros(n_elem * n_loc * n_loc)
            rows = np.zeros_like(data, dtype=np.int32)
            cols = np.zeros_like(data, dtype=np.int32)

            for e in range(n_elem):
                gdofs = dh.get_elemental_dofs(e)
                r, c = np.meshgrid(gdofs, gdofs, indexing="ij")
                offset = e * n_loc * n_loc
                rows[offset:offset + n_loc * n_loc] = r.ravel()
                cols[offset:offset + n_loc * n_loc] = c.ravel()
                data[offset:offset + n_loc * n_loc] = K_loc[e].ravel()

            A = sp.coo_matrix((data, (rows, cols)), shape=(n_total, n_total)).tocsr()

            # --- Dirichlet BCs ----------------------------------------
            if self.bcs_homog:
                bc_data = dh.get_dirichlet_data(self.bcs_homog)
                if bc_data:
                    bc_dofs = np.fromiter(bc_data.keys(), dtype=int)
                    bc_vals = np.fromiter(bc_data.values(), dtype=float)
                    R -= A @ np.bincount(bc_dofs, weights=bc_vals, minlength=n_total)
                    A = _zero_rows_cols(A, bc_dofs)
                    R[bc_dofs] = bc_vals
        return A, R

    def _solve_linear_system(self, A: sp.csr_matrix, rhs: np.ndarray) -> np.ndarray:
        if self.lp.backend == "scipy":
            return spla.spsolve(A, rhs)
        else:
            raise ValueError(f"Unknown linear solver backend '{self.lp.backend}'.")


    # ------------------------------------------------------------------
    #  Armijo back‑tracking line‑search
    # ------------------------------------------------------------------
    def _line_search(self, funcs, dU, coeffs, R0, bcs_now):
        dh = self.dh
        alpha = 1.0

        for ls_it in range(self.np.ls_max_iter):
            # trial update ------------------------------
            dh.add_to_functions(alpha * dU, funcs)
            dh.apply_bcs(bcs_now, *funcs)

            # residual ---------------------------------
            _, F_loc_trial = self._res_runner(coeffs, self.static_args)
            _, R_trial = self._assemble_system(np.zeros_like(F_loc_trial), F_loc_trial)

            if np.linalg.norm(R_trial, ord=np.inf) <= (1.0 - self.np.ls_c1 * alpha) * R0:
                return alpha * dU  # accept

            # reject: rollback --------------------------
            dh.add_to_functions(-alpha * dU, funcs)
            alpha *= self.np.ls_reduction

        print("        Line search failed – proceeding with full step anyway.")
        return dU
    # ------------------------------------------------------------------
    #  Boundary‑condition helper
    # ------------------------------------------------------------------
    @staticmethod
    def _freeze_bcs(bcs, t_now):
        """Return a shallow copy where callable BCs are bound to *t_now*."""
        frozen = []
        for bc in bcs:
            func = bc.value
            if callable(func):
                sig = inspect.signature(func)
                if len(sig.parameters) == 3:  # expects (x, y, t)
                    func = (lambda x, y, _f=func, _t=t_now: _f(x, y, _t))
            frozen.append(type(bc)(bc.field, bc.method, bc.domain_tag, func))
        return frozen

# ----------------------------------------------------------------------------
#  Helper utility
# ----------------------------------------------------------------------------

def _zero_rows_cols(A: sp.csr_matrix, rows: np.ndarray) -> sp.csr_matrix:
    """Zero out rows *and* columns and put 1.0 on the diagonal (in‑place)."""
    A = A.tolil()
    A[rows, :] = 0.0
    A[:, rows] = 0.0
    A[rows, rows] = 1.0
    return A.tocsr()
