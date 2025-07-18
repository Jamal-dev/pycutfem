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
from pycutfem.jit import compile_multi        # NEW import


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
        self._res_runner, self._res_ir = compile_multi(
            residual_form, dof_handler=self.dh, mixed_element=self.me
        )
        self._jac_runner, self._jac_ir = compile_multi(
            jacobian_form, dof_handler=self.dh, mixed_element=self.me
        )

        # Normalise to lists for easier iteration later
        self._res_runners = getattr(self._res_runner, "runners", [self._res_runner])
        self._jac_runners = getattr(self._jac_runner, "runners", [self._jac_runner])

        # 2) pre‑compute static arguments (geometry, basis tables, …) -------
        self._precompute_static_args(quad_order)

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

            import scipy.sparse as sp
            n_glob = dh.total_dofs
            A = sp.csr_matrix((n_glob, n_glob))
            R = np.zeros(n_glob)

            # --- Jacobian contributions ---
            for runner, sa in zip(self._jac_runners, self.static_args_jac):
                K_loc = runner(current, sa)[0]
                A_part, _ = self._assemble_system(K_loc, np.zeros((K_loc.shape[0], K_loc.shape[1])), sa["gdofs_map"])
                A = A + A_part

            # --- Residual contributions ---
            for runner, sa in zip(self._res_runners, self.static_args_res):
                F_loc = runner(current, sa)[1]
                _, R_part = self._assemble_system(np.zeros((F_loc.shape[0], F_loc.shape[1])), F_loc, sa["gdofs_map"])
                R += R_part

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
    def _precompute_static_args(self, quad_order: int):
        """
        Build and return static kernel arguments **per integral**.
        """
        import numpy as np
        from typing import Optional
        from pycutfem.ufl.helpers import required_multi_indices

        dh, me = self.dh, self.me
        self.union_dofs = dh.union_dofs
        mesh = me.mesh
        ghost_ls: Optional[callable] = None
        ghost_derivs: set = set()
        interface_ls: Optional[callable] = None

        # ------------------------------------------------------ 1) scan integrals
        def _scan(ints):
            nonlocal ghost_ls, ghost_derivs, interface_ls
            for I in ints:
                if I.measure.domain_type == "ghost_edge":
                    ghost_ls      = I.measure.level_set
                    ghost_derivs |= required_multi_indices(I.integrand)
                if I.measure.domain_type == "interface":
                    interface_ls = I.measure.level_set

        _scan(self._jacobian_form.integrals
            if hasattr(self._jacobian_form, "integrals") else
            [self._jacobian_form])
        _scan(self._residual_form.integrals
            if hasattr(self._residual_form, "integrals") else
            [self._residual_form])

        # ------------------------------------------------------ 2) place‑holders
        _empty_J = np.empty((0, 0, 2, 2))      # re‑usable singleton
        vol_geo = dh.precompute_geometric_factors(quad_order)
        vol_geo["gdofs_map"] = np.vstack(
            [dh.get_elemental_dofs(e) for e in range(mesh.n_elements)]
        ).astype(np.int32)
        vol_geo["node_coords"] = dh.get_all_dof_coords()

        ghost_geo = {}
        edge_ids = mesh.edge_bitset("ghost")
        if ghost_ls is not None and edge_ids.cardinality():
            ghost_geo = dh.precompute_ghost_factors(edge_ids, quad_order, ghost_ls, ghost_derivs)

        interface_geo = {}
        if interface_ls is not None:
            cut_eids = mesh.element_bitset("cut")
            if cut_eids.cardinality():
                interface_geo = dh.precompute_interface_factors(cut_eids, quad_order, interface_ls)
                interface_geo["gdofs_map"] = vol_geo["gdofs_map"]

        def _build(ir, integrand, runner, base):
            args = dict(base)
            args.update(
                _build_jit_kernel_args(ir, integrand, me, quad_order, dh,
                                       param_order=runner.param_order,
                                       pre_built=base)
            )
            # pad tables to union size
            n_union = self.union_dofs
            for k, v in list(args.items()):
                if not isinstance(v, np.ndarray) or v.ndim < 3:
                    continue
                if v.shape[-1] < n_union and v.shape[-2] != 2:
                    pad = np.zeros(v.shape[:-1] + (n_union,), dtype=v.dtype)
                    pad[..., :v.shape[-1]] = v
                    args[k] = pad
                elif v.shape[-2] < n_union and v.shape[-1] == 2:
                    pad = np.zeros(v.shape[:-2] + (n_union, 2), dtype=v.dtype)
                    pad[..., :v.shape[-2], :] = v
                    args[k] = pad
            return {k: v for k, v in args.items() if not k.endswith("_loc")}

        self.static_args_jac = []
        for ir, I, r in zip(self._jac_ir, self._jacobian_form.integrals, self._jac_runners):
            base = vol_geo
            if I.measure.domain_type == "ghost_edge":
                base = ghost_geo
            elif I.measure.domain_type == "interface":
                base = interface_geo
            self.static_args_jac.append(_build(ir, I.integrand, r, base))

        self.static_args_res = []
        for ir, I, r in zip(self._res_ir, self._residual_form.integrals, self._res_runners):
            base = vol_geo
            if I.measure.domain_type == "ghost_edge":
                base = ghost_geo
            elif I.measure.domain_type == "interface":
                base = interface_geo
            self.static_args_res.append(_build(ir, I.integrand, r, base))

        return






    # ------------------------------------------------------------------
    #  Linear system & BC handling
    #  Assemble global matrix & residual from local JIT blocks
    # ----------------------------------------------------------------------
    def _assemble_system(self, K_loc: np.ndarray, F_loc: np.ndarray, gdofs_map: np.ndarray):
        """
        Scatter JIT‑returned (K_loc, F_loc) into a CSR matrix A and vector R.

        *  Each row corresponds to either an **element** or an **edge**.
        *  The mapping from that row to global DOFs is the same
        ``static_args['gdofs_map'][row]`` that was used in the kernel.
        *  Rows may contain −1 padding; those columns are skipped.
        """
        import numpy as np
        import scipy.sparse as sp
        from scipy.sparse import lil_matrix

        dh          = self.dh
        n_glob      = dh.total_dofs

        # ---------------- residual ----------------------------------------
        R = np.zeros(n_glob)
        for row, Fe in enumerate(F_loc):
            gmap  = gdofs_map[row]
            valid = gmap >= 0
            np.add.at(R, gmap[valid], Fe[valid])

        # ---------------- fast path: residual only ------------------------
        if K_loc.ndim == 2:
            A = sp.csr_matrix((n_glob, n_glob))   # empty shell
        else:
            # sparse accumulate -------------------------------------------
            A = lil_matrix((n_glob, n_glob))
            for row, Ke in enumerate(K_loc):
                gmap  = gdofs_map[row]
                valid = gmap >= 0
                rows  = gmap[valid]
                cols  = gmap[valid]
                A[np.ix_(rows, cols)] += Ke[np.ix_(valid, valid)]
            A = A.tocsr()

            # ------------- homogeneous Dirichlet BCs ----------------------
            if self.bcs_homog:
                bc_data = dh.get_dirichlet_data(self.bcs_homog)
                if bc_data:
                    bc_dofs = np.fromiter(bc_data.keys(), dtype=int)
                    bc_vals = np.fromiter(bc_data.values(), dtype=float)

                    # subtract known values from RHS
                    R -= A @ np.bincount(bc_dofs, weights=bc_vals,
                                        minlength=n_glob)

                    # zero rows & columns, put 1 on diag
                    A_lil = A.tolil()
                    A_lil[bc_dofs, :] = 0
                    A_lil[:, bc_dofs] = 0
                    A_lil[bc_dofs, bc_dofs] = 1.0
                    A = A_lil.tocsr()

                    # enforce values in RHS
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
            n_glob = dh.total_dofs
            R_trial = np.zeros(n_glob)
            for runner, sa in zip(self._res_runners, self.static_args_res):
                F_loc_trial = runner(coeffs, sa)[1]
                _, part = self._assemble_system(np.zeros((F_loc_trial.shape[0], F_loc_trial.shape[1])), F_loc_trial, sa["gdofs_map"])
                R_trial += part

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
