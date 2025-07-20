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
# from pycutfem.ufl.helpers_jit import _scatter_element_contribs
from pycutfem.ufl.helpers_jit import _build_jit_kernel_args
from pycutfem.jit import compile_multi        # NEW import

def _scatter_element_contribs(
    *,
    K_elem: np.ndarray | None,
    F_elem: np.ndarray | None,
    J_elem: np.ndarray | None,
    element_ids: np.ndarray,
    gdofs_map: np.ndarray,
    target: np.ndarray,            # lil_matrix for Jacobian, 1‑D array for RHS
    ctx: dict,
    integrand,
    hook,
):
    """
    Add local element (or facet) contributions into the global matrix / vector.

    Behaviour is controlled via *ctx*:
        ctx["rhs"]  : True  → F_elem is assembled (vector mode)
                      False → K_elem is assembled (matrix mode)
        ctx["add"]  : True  → contributions are *added* to *target*
                      False → target[...] is overwritten
    """
    rhs    = ctx.get("rhs", False)
    do_add = ctx.get("add", True)

    # ------------------------------------------------------------------ #
    # Matrix contributions                                               #
    # ------------------------------------------------------------------ #
    if not rhs and K_elem is not None and K_elem.ndim == 3:
        for i in range(len(element_ids)):
            gdofs = gdofs_map[i]
            valid = gdofs >= 0                      # ignore padding (‑1)

            if not np.any(valid):
                continue

            rows = gdofs[valid]
            Ke   = K_elem[i][np.ix_(valid, valid)]

            r, c = np.meshgrid(rows, rows, indexing="ij")
            if do_add:
                target[r, c] += Ke
            else:
                target[r, c]  = Ke

    # ------------------------------------------------------------------ #
    # Vector contributions                                               #
    # ------------------------------------------------------------------ #
    if rhs and F_elem is not None:
        for i in range(len(element_ids)):
            gdofs = gdofs_map[i]
            valid = gdofs >= 0

            if not np.any(valid):
                continue

            if do_add:
                np.add.at(target, gdofs[valid], F_elem[i][valid])
            else:
                target[gdofs[valid]] = F_elem[i][valid]

    # ------------------------------------------------------------------ #
    # Functional contributions (unchanged)                               #
    # ------------------------------------------------------------------ #
    if hook and J_elem is not None:
        total = J_elem.sum(axis=0) if J_elem.ndim > 1 else J_elem.sum()
        acc = ctx.setdefault("scalar_results", {}).setdefault(
            hook["name"], np.zeros_like(total)
        )
        acc += total

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
        backend: str = "jit",
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

        # --- compile one kernel list for K, one for F ----------------------
        self.kernels_K = compile_multi(        # Jacobian
                jacobian_form,
                dof_handler   = self.dh,
                mixed_element = self.me,
                backend       = backend,
        )

        self.kernels_F = compile_multi(        # Residual
                residual_form,
                dof_handler   = self.dh,
                mixed_element = self.me,
                backend       = backend,
        )


        
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
            print(f"    Time step {step+1}: ΔU = {np.linalg.norm(delta_U, ord=np.inf):.2e}")

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

            A, R = self._assemble_system(current)

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
    #  Linear system & BC handling
    # ------------------------------------------------------------------
    def _assemble_system(self, coeffs, *, need_matrix: bool = True):
        """
        Assemble global Jacobian A (optional) and residual R.

        The key difference from the old version is that we *accumulate* every
        kernel’s contribution into one global matrix / vector instead of overwriting
        them on each loop iteration.
        """
        dh   = self.dh
        ndof = dh.total_dofs

        A_glob = sp.lil_matrix((ndof, ndof)) if need_matrix else None
        R_glob = np.zeros(ndof)

        # ------------------------------------------------------------------ #
        # 1) Jacobian                                                        #
        # ------------------------------------------------------------------ #
        if need_matrix:
            for ker in self.kernels_K:
                Kloc, _, _ = ker.exec(coeffs)

                _scatter_element_contribs(
                    K_elem      = Kloc,
                    F_elem      = None,
                    J_elem      = None,
                    element_ids = ker.static_args["eids"],
                    gdofs_map   = ker.static_args["gdofs_map"],
                    target      = A_glob,               # accumulate into A_glob
                    ctx         = {"rhs": False, "add": True},
                    integrand   = ker,
                    hook        = None,
                )

        # ------------------------------------------------------------------ #
        # 2) Residual                                                        #
        # ------------------------------------------------------------------ #
        for ker in self.kernels_F:
            _, Floc, _ = ker.exec(coeffs)

            R_inc = np.zeros_like(R_glob)                # per‑kernel buffer
            _scatter_element_contribs(
                K_elem      = None,
                F_elem      = Floc,
                J_elem      = None,
                element_ids = ker.static_args["eids"],
                gdofs_map   = ker.static_args["gdofs_map"],
                target      = R_inc,                     # write into buffer
                ctx         = {"rhs": True, "add": True},
                integrand   = ker,
                hook        = None,
            )

            # print individual contribution (handy for debugging)
            print(f"{ker.domain:9}: |R|_∞ = {np.linalg.norm(R_inc, np.inf):.3e}")

            R_glob += R_inc                              # accumulate

        # ------------------------------------------------------------------ #
        # 3) Homogeneous Dirichlet rows                                       #
        # ------------------------------------------------------------------ #
        if need_matrix and self.bcs_homog:
            bc_data = dh.get_dirichlet_data(self.bcs_homog)
            if bc_data:
                rows = np.fromiter(bc_data.keys(),  dtype=int)
                vals = np.fromiter(bc_data.values(), dtype=float)

                bc_vec = np.zeros(ndof)
                bc_vec[rows] = vals

                R_glob -= A_glob @ bc_vec          # move prescribed values to RHS
                A_glob  = _zero_rows_cols(A_glob, rows)
                R_glob[rows] = vals               # enforce value in residual

        return (A_glob.tocsr() if need_matrix else None), R_glob


    def _solve_linear_system(self, A: sp.csr_matrix, rhs: np.ndarray) -> np.ndarray:
        if self.lp.backend == "scipy":
            return spla.spsolve(A, rhs)
        else:
            raise ValueError(f"Unknown linear solver backend '{self.lp.backend}'.")


    # ------------------------------------------------------------------
    #  Armijo back‑tracking line‑search
    # ------------------------------------------------------------------
    def _line_search(self, funcs, dU, coeffs, R0, bcs_now):
        dh    = self.dh
        alpha = 1.0

        for _ in range(self.np.ls_max_iter):
            # 1) trial update ------------------------------------------------
            dh.add_to_functions(alpha * dU, funcs)
            dh.apply_bcs(bcs_now, *funcs)

            # 2) residual with *matrix‑free* assembly ------------------------
            _, R_trial = self._assemble_system(coeffs, need_matrix=False)

            # 3) Armijo test -------------------------------------------------
            if np.linalg.norm(R_trial, np.inf) <= (1.0 - self.np.ls_c1 * alpha) * R0:
                return alpha * dU          # accept step

            # 4) reject → rollback & shrink ---------------------------------
            dh.add_to_functions(-alpha * dU, funcs)
            alpha *= self.np.ls_reduction

        print("        Line search failed – taking full Newton step.")
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
