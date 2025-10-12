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

from re import A
import time
import os
from dataclasses import dataclass
from typing import Dict, List, Callable, Optional, Tuple
import inspect


import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sp
# from pycutfem.ufl.helpers_jit import _scatter_element_contribs
from pycutfem.ufl.helpers_jit import _build_jit_kernel_args
from pycutfem.jit import compile_multi        
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.helpers import analyze_active_dofs
try:
    from pycutfem.solvers.giant import giant_solver
    HAS_GIANT = True
except ImportError:
    giant_solver = None
    HAS_GIANT = False
_skip_petsc = os.getenv("PYCUTFEM_SKIP_PETSC", "").lower() in {"1", "true", "yes"}
if not _skip_petsc:
    try:
        from petsc4py import PETSc
        HAS_PETSC = True
    except Exception:  # noqa: PERF203
        PETSc = None
        HAS_PETSC = False
else:
    PETSc = None
    HAS_PETSC = False
import sys

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

def _scatter_element_contribs_reduced(
    *,
    K_elem: np.ndarray | None,
    F_elem: np.ndarray | None,
    J_elem: np.ndarray | None,
    element_ids: np.ndarray,
    gdofs_map: np.ndarray,
    A_red,                 # csr/lil for matrix or None
    R_red,                 # 1-D array for rhs or None
    full_to_red: np.ndarray,  # length = ndof, maps full→reduced (−1 if dropped)
    hook=None,
):
    """
    Scatter element contributions directly into a reduced system.
    Rows/cols with full_to_red == -1 (Dirichlet / inactive) are ignored.
    """

    nloc_ok = lambda arr: (arr is not None) and (arr.ndim >= 2)

    # --- Matrix block: keep only (free, free) -------------------------
    if A_red is not None and K_elem is not None and K_elem.ndim == 3:
        for i in range(len(element_ids)):
            gdofs = gdofs_map[i]
            rmap  = full_to_red[gdofs]
            rmask = rmap >= 0
            if not np.any(rmask):
                continue
            rows = rmap[rmask]

            # columns use the same element gdofs (bilinear form)
            cmap  = full_to_red[gdofs]
            cmask = cmap >= 0
            if not np.any(cmask):
                continue
            cols = cmap[cmask]

            Ke = K_elem[i][np.ix_(rmask, cmask)]
            A_red[np.ix_(rows, cols)] += Ke

    # --- Vector block: keep only free rows ----------------------------
    if R_red is not None and F_elem is not None:
        for i in range(len(element_ids)):
            gdofs = gdofs_map[i]
            rmap  = full_to_red[gdofs]
            rmask = rmap >= 0
            if not np.any(rmask):
                continue
            rows = rmap[rmask]
            R_red[rows] += F_elem[i][rmask]

    # --- Scalar functionals unchanged (optional) ----------------------
    if hook is not None and J_elem is not None:
        total = J_elem.sum(axis=0) if J_elem.ndim > 1 else J_elem.sum()
        acc = hook.setdefault("acc", 0.0)
        hook["acc"] = acc + total


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
    ls_c1: float = 0.1#1.0e-4               # sufficient‑decrease parameter


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
#  Restriction helper
# ----------------------------------------------------------------------------
# put this in nonlinear_solver.py (top-level) -------------------------
class _ActiveReducer:
    def __init__(self, dh, active_dofs):
        self.dh = dh
        self.active = np.asarray(active_dofs, dtype=int)
        self.full2red = {i: j for j, i in enumerate(self.active)}

    # vectors ----------------------------------------------------------
    def restrict_vec(self, v_full):
        return v_full[self.active]

    def expand_vec(self, v_red):
        v_full = np.zeros(self.dh.total_dofs, dtype=v_red.dtype)
        v_full[self.active] = v_red
        return v_full

    # matrices ---------------------------------------------------------
    def restrict_mat(self, A_full):
        aid = self.active
        return A_full.tocsr()[np.ix_(aid, aid)]

    # systems (full → reduced with BCs already handled in the full space)
    def reduce_system(self, A_full, R_full, dh, bcs):
        K_red = self.restrict_mat(A_full)
        F_red = self.restrict_vec(R_full)

        # Dirichlet on the reduced system (identical to your current code)
        bc_data = dh.get_dirichlet_data(bcs)
        if bc_data:
            rows_full = np.fromiter(bc_data.keys(), dtype=int)
            vals_full = np.fromiter(bc_data.values(), dtype=float)

            # keep only rows that are active
            mask = np.isin(rows_full, self.active)
            rows_full = rows_full[mask]
            vals_full = vals_full[mask]
            if rows_full.size:
                rows_red = np.array([self.full2red[i] for i in rows_full])

                F_red -= K_red @ np.bincount(rows_red, weights=vals_full,
                                             minlength=self.active.size)

                K_red = K_red.tolil()
                K_red[rows_red, :] = 0
                K_red[:, rows_red] = 0
                K_red[rows_red, rows_red] = 1.0
                K_red = K_red.tocsr()

                F_red[rows_red] = vals_full
        return K_red, F_red


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
        deformation: Optional[object] = None,
        # callback returning the updated nodal displacement array given the
        # coefficient dictionary used for assembly. Returning ``None`` skips
        # geometry refresh for that call.
        deformation_update: Optional[Callable[[Dict[str, object]], np.ndarray]] = None,
        preproc_cb: Optional[Callable[[List], None]] = None,
        postproc_cb: Optional[Callable[[List], None]] = None,
        backend: str = "jit",
        postproc_timeloop_cb: Optional[Callable[[List], None]] = None,
    ) -> None:
        self.dh = dof_handler
        self.me = mixed_element
        self.bcs = bcs
        self.bcs_homog = bcs_homog
        self.np = newton_params
        self.lp = lin_params
        self.pre_cb = preproc_cb
        self.post_cb = postproc_cb
        self.post_timeloop_cb = postproc_timeloop_cb
        self.backend = backend
        self.quad_order = quad_order
        self.deformation = deformation
        self._deformation_update = deformation_update

        # Keep original forms handy (their .integrand is needed later)
        self._residual_form = residual_form
        self._jacobian_form = jacobian_form

        # --- compile one kernel list for K, one for F ----------------------
        self._compile_all_kernels()
        # --- NEW: A PRIORI DOF ANALYSIS ---
        # Analyze the forms once to get the definitive set of active DOFs.
        self.equation = Equation(jacobian_form, residual_form)
        # Which BC list marks the fixed rows? Prefer homogeneous set.
        bcs_for_active = self.bcs_homog if getattr(self, "bcs_homog", None) else self.bcs
        # 1) What DOFs are “touched” by Restriction (if any)?
        active_by_restr, has_restriction = analyze_active_dofs(
            self.equation, self.dh, self.me, bcs_for_active
        )
        ndof = self.dh.total_dofs
        bc_dofs = set(self.dh.get_dirichlet_data(bcs_for_active).keys())

        # Candidate DOFs that would remain without explicit drops
        candidate = set(active_by_restr) if has_restriction else set(range(ndof))

        # Optional DOF tags (e.g. CutFEM inactive regions) are honoured here.
        inactive_dofs: set[int] = set()
        dh_tags = getattr(self.dh, "dof_tags", None)
        if dh_tags:
            inactive_dofs = set(dh_tags.get("inactive", set()))
        # Remove DOFs tagged inactive but not already fixed by Dirichlet BCs
        inactive_free = inactive_dofs - bc_dofs
        inactive_removed = len(inactive_free & candidate)

        # 2) Free DOFs = candidate DOFs \ (Dirichlet ∪ inactive)
        free_set = (candidate - bc_dofs) - inactive_free
        free = sorted(free_set)

        self.active_dofs = np.asarray(free, dtype=int)
        nfree = self.active_dofs.size

        # 3) Maps full ↔ reduced
        self.full_to_red = -np.ones(ndof, dtype=int)
        self.full_to_red[self.active_dofs] = np.arange(nfree, dtype=int)
        self.red_to_full = self.active_dofs

        # 4) Always run the reduced path when there are any fixed DOFs
        self.use_reduced = (nfree < ndof)
        if not self.use_reduced:
            print("NewtonSolver: Operating on the full, unrestricted system.")
        else:
            print(f"NewtonSolver: Reduced system with {nfree}/{ndof} DOFs.")

        # Optional sanity log:
        print(f"  Dirichlet DOFs detected: {len(bc_dofs)}; Free DOFs: {nfree}")
        if inactive_dofs:
            print(f"  Inactive DOFs tagged: {len(inactive_dofs)} (unique)")
            if inactive_removed:
                print(f"  Inactive DOFs dropped: {inactive_removed}")
            else:
                print("  Inactive DOFs already excluded by restriction domains.")

        # Build once: CSR structure & per-element scatter plan for reduced system
        self._build_reduced_pattern()

        self.restrictor = _ActiveReducer(self.dh, self.active_dofs)




    def _compile_all_kernels(self) -> None:
        """(Re)compile residual and Jacobian kernels with current metadata."""
        self.kernels_K = compile_multi(
            self._jacobian_form,
            dof_handler=self.dh,
            mixed_element=self.me,
            quad_order=self.quad_order,
            backend=self.backend,
        )

        self.kernels_F = compile_multi(
            self._residual_form,
            dof_handler=self.dh,
            mixed_element=self.me,
            quad_order=self.quad_order,
            backend=self.backend,
        )

        self._pattern_stale = True
        self._last_jacobian = None
        for attr in ("_csr_indptr", "_csr_indices", "_elem_pos", "_elem_lidx"):
            if hasattr(self, attr):
                delattr(self, attr)

    def _maybe_refresh_deformation(self, coeffs: Dict[str, object]) -> None:
        if self.deformation is None or self._deformation_update is None:
            return

        disp = self._deformation_update(coeffs)
        if disp is None:
            return

        disp_arr = np.asarray(disp, dtype=float)
        target = self.deformation.node_displacements
        if disp_arr.shape != target.shape:
            raise ValueError(
                "deformation_update returned displacements with shape "
                f"{disp_arr.shape}, expected {target.shape}"
            )

        if np.allclose(disp_arr, target):
            return

        np.copyto(target, disp_arr)
        self._compile_all_kernels()

        
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
        post_step_refiner=None
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
        post_step_refiner : Callable, optional
            It can be another newton method that refines the solution

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
            
            # Post-step refiner (VI clip) **before** promotion so prev matches clipped state
            if post_step_refiner is not None:
                post_step_refiner(step, bcs_now, functions, prev_functions)
            
             # Accept: promote current → previous
            for f_prev, f in zip(prev_functions, functions):
                f_prev.nodal_values[:] = f.nodal_values[:]

            if step > 0:
                # Reject and retry with smaller Δt if the update blows up
                if np.linalg.norm(delta_U, np.inf) > 5.0 * (np.linalg.norm(self.delta_U_prev, np.inf) if step else 1.0):
                    # restore current to previous and cut dt
                    for f, fp in zip(functions, prev_functions):
                        f.nodal_values[:] = fp.nodal_values[:]
                    time_params.dt *= 0.5
                    print(f"    Rejecting step {step+1}; reducing Δt → {time_params.dt:.3e} and retrying.")
                    continue
            self.delta_U_prev = delta_U.copy()
            # Steady‑state test -----------------------------------
            if (
                time_params.stop_on_steady
                and step > 0
                and np.linalg.norm(delta_U, ord=np.inf) < time_params.steady_tol
            ):
                break

            # # Accept: promote current → previous ------------------
            # dh.apply_bcs(bcs_now, *functions)
            # for f_prev, f in zip(prev_functions, functions):
            #     f_prev.nodal_values[:] = f.nodal_values[:]
            # Post time-loop callback
            if self.post_timeloop_cb is not None:
                self.post_timeloop_cb(functions)
            if t_n + dt >= time_params.final_time:
                break

        elapsed = time.perf_counter() - t_start
        return delta_U, step + 1, elapsed

    # ------------------------------------------------------------------
    #  Newton iteration (internal)
    # ------------------------------------------------------------------
    def _newton_loop(self, funcs, prev_funcs, aux_funcs, bcs_now):
        """
        Newton iterations that ALWAYS operate in the reduced space (ff block).
        - Dirichlet DOFs are excluded via self.active_dofs.
        - Inhomogeneous Dirichlet values are NOT re-applied inside the iteration,
        only after an accepted step (BCs were already set before entering the loop).
        """
        dh = self.dh
        ndof = dh.total_dofs

        # Quick safety: make sure we actually have a reduced set
        # (This is only for logging; we still run the reduced pipeline.)
        nfree = len(self.active_dofs)
        if nfree == ndof:
            print("        [warn] active_dofs == total_dofs — reduced path = full size."
                " Check that bcs_homog correctly marks Dirichlet nodes.")

        self._last_iter_timings = []
        totals = {"assembly": 0.0, "linear_solve": 0.0, "line_search": 0.0}
        temp_t0 = time.perf_counter()
        for it in range(self.np.max_newton_iter):
            if self.pre_cb is not None:
                self.pre_cb(funcs)

            # Build the coefficients dict expected by kernels
            current: Dict[str, "Function"] = {f.name: f for f in funcs}
            current.update({f.name: f for f in prev_funcs})
            if aux_funcs:
                current.update(aux_funcs)

            # 1) Assemble reduced system: A_ff δU_f = -R_f
            assembly_time = 0.0
            t_asm = time.perf_counter()
            A_red, R_red = self._assemble_system_reduced(current, need_matrix=True)
            assembly_time += time.perf_counter() - t_asm

            # 1a) PRUNE decoupled rows caused by Restriction (nnz==0)
            row_nnz = np.diff(A_red.indptr)
            inactive = np.where(row_nnz == 0)[0]
            if inactive.size:
                keep_mask = np.ones(len(self.active_dofs), dtype=bool)
                keep_mask[inactive] = False
                # rebuild maps
                self.active_dofs = self.active_dofs[keep_mask]
                self.full_to_red = -np.ones(ndof, dtype=int)
                self.full_to_red[self.active_dofs] = np.arange(self.active_dofs.size, dtype=int)
                self.red_to_full = self.active_dofs
                self._pattern_stale = True
                self.restrictor = _ActiveReducer(self.dh, self.active_dofs)
                # rebuild reduced CSR scatter pattern
                self._build_reduced_pattern()
                # re-assemble on the cleaned support
                t_asm = time.perf_counter()
                A_red, R_red = self._assemble_system_reduced(current, need_matrix=True)
                assembly_time += time.perf_counter() - t_asm

            # Log residual with a full-size view (zeros on fixed DOFs) for readability
            R_full = np.zeros(ndof)
            R_full[self.active_dofs] = R_red
            norm_R = np.linalg.norm(R_full, ord=np.inf)
            t_current = time.perf_counter()
            t_iteration = t_current - temp_t0
            temp_t0 = t_current
            print(f"        Newton {it+1}: |R|_∞ = {norm_R:.2e}, time = {t_iteration}s")
            linear_time = 0.0
            line_search_time = 0.0
            if norm_R < self.np.newton_tol:
                # Converged — return *time-step* increment for all fields
                delta = np.hstack([
                    f.nodal_values - f_prev.nodal_values
                    for f, f_prev in zip(funcs, prev_funcs)
                ])
                totals["assembly"] += assembly_time
                self._last_iter_timings.append(
                    {
                        "iteration": it + 1,
                        "assembly": assembly_time,
                        "linear_solve": 0.0,
                        "line_search": 0.0,
                    }
                )
                self._last_iteration_totals = totals
                print(
                    "          timings: assembly={:.3e}s, solve={:.3e}s, line-search={:.3e}s".format(
                        assembly_time, 0.0, 0.0
                    )
                )
                return delta

            # 2) Compute reduced Newton direction
            t_lin = time.perf_counter()
            dU_red = self._solve_linear_system(A_red, -R_red)
            linear_time = time.perf_counter() - t_lin

            # 3) Optional Armijo backtracking in reduced space (no BC re-application inside)
            if getattr(self.np, "line_search", False):
                t_ls = time.perf_counter()
                dU_red = self._line_search_reduced(A_red, R_red, dU_red, funcs, current, bcs_now)
                line_search_time = time.perf_counter() - t_ls

            # 4) Apply accepted step: expand reduced → full, update fields, then re-impose BCs
            dU_full = np.zeros(ndof)
            dU_full[self.active_dofs] = dU_red
            dh.add_to_functions(dU_full, funcs)

            # Re-impose the (possibly inhomogeneous) Dirichlet values AFTER the update
            dh.apply_bcs(bcs_now, *funcs)

            if self.post_cb is not None:
                self.post_cb(funcs)

            totals["assembly"] += assembly_time
            totals["linear_solve"] += linear_time
            totals["line_search"] += line_search_time
            self._last_iter_timings.append(
                {
                    "iteration": it + 1,
                    "assembly": assembly_time,
                    "linear_solve": linear_time,
                    "line_search": line_search_time,
                }
            )
            print(
                "          timings: assembly={:.3e}s, solve={:.3e}s, line-search={:.3e}s".format(
                    assembly_time, linear_time, line_search_time
                )
            )

        # If we get here, Newton did not converge within the iteration budget
        self._last_iteration_totals = totals
        raise RuntimeError("Newton did not converge – adjust Δt or verify Jacobian.")


   
    def _line_search_reduced(self, A_red, R_red, S_red, funcs, coeffs, bcs_now):
        """
        Backtracking Armijo search performed entirely in the reduced space.
        - A_red, R_red: current reduced Jacobian and residual
        - S_red: proposed search direction in reduced space
        """
        dh   = self.dh
        np_  = self.np
        c1   = getattr(np_, "ls_c1", 1.0e-4)

        # Gradient g_f = J_ff^T R_f  (reduced variables only)
        g = A_red.T @ R_red
        gTS = float(g @ S_red)
        if gTS >= 0.0:
            # Not a descent direction – reject and take zero step
            print("        Warning: Not a descent direction in reduced space.")
            return np.zeros_like(S_red)

        phi0 = 0.5 * float(R_red @ R_red)
        alpha = 1.0
        best_alpha, best_phi = 0.0, phi0

        # Snapshot the *full* iterate
        snap = [f.nodal_values.copy() for f in funcs]

        for _ in range(np_.ls_max_iter):
            # Trial update: expand to full, apply, and re-impose BCs
            for f, buf in zip(funcs, snap): f.nodal_values[:] = buf
            dU_full = np.zeros(dh.total_dofs)
            dU_full[self.active_dofs] = alpha * S_red
            dh.add_to_functions(dU_full, funcs)
            dh.apply_bcs(bcs_now, *funcs)

            # Evaluate residual in reduced space (matrix not needed)
            _, R_try = self._assemble_system_reduced(coeffs, need_matrix=False)
            phi = 0.5 * float(R_try @ R_try)

            # Track best effort
            if phi < best_phi:
                best_phi, best_alpha = phi, alpha

            # Armijo condition with reduced quantities
            if phi <= phi0 + c1 * alpha * gTS:
                for f, buf in zip(funcs, snap): f.nodal_values[:] = buf
                print(f"        Armijo search accepted α = {alpha:.2e}")
                return alpha * S_red

            alpha *= np_.ls_reduction

        # Fallback (best effort seen)
        for f, buf in zip(funcs, snap): f.nodal_values[:] = buf
        if best_alpha > 0.0:
            print(f"        Armijo failed, using best-effort α = {best_alpha:.2e}.")
            return best_alpha * S_red
        print("        Line search failed – no descent.")
        return np.zeros_like(S_red)

    # ------------------------------------------------------------------
    #  Reduced Linear system & BC handling
    # ------------------------------------------------------------------
    def _assemble_system_reduced(self, coeffs, *, need_matrix: bool = True):
        self._maybe_refresh_deformation(coeffs)
        nred = len(self.active_dofs)
        if need_matrix:
            A_red, R_red = self._assemble_system_reduced_fast(coeffs)
            return A_red, R_red
        else:
            # residual only: keep a light path that avoids fancy indexing
            R_red = np.zeros(nred)
            for ker in self.kernels_F:
                _, Floc, _ = ker.exec(coeffs)
                eids = ker.static_args["eids"]; gdofs = ker.static_args["gdofs_map"]
                for e in range(len(eids)):
                    rmap = self.full_to_red[gdofs[e]]
                    m = rmap >= 0
                    if np.any(m):
                        np.add.at(R_red, rmap[m], Floc[e][m])
            return None, R_red


    # ------------------------------------------------------------------
    #  Linear system & BC handling        (REF‑ACTORED)
    # ------------------------------------------------------------------
    def _assemble_system(self, coeffs, *, need_matrix: bool = True):
        """
        Assemble global Jacobian A (optional) and residual R.

        * When ``need_matrix=False`` it returns (None, R) but **still**
          enforces homogeneous Dirichlet rows in *R* so that line‑search
          evaluations are meaningful.
        """
        self._maybe_refresh_deformation(coeffs)
        dh   = self.dh
        ndof = dh.total_dofs

        A_glob = sp.lil_matrix((ndof, ndof)) if need_matrix else None
        R_glob = np.zeros(ndof)

        # 1) Jacobian ---------------------------------------------------
        if need_matrix:
            for ker in self.kernels_K:
                Kloc, _, _ = ker.exec(coeffs)

                _scatter_element_contribs(
                    K_elem      = Kloc,
                    F_elem      = None,
                    J_elem      = None,
                    element_ids = ker.static_args["eids"],
                    gdofs_map   = ker.static_args["gdofs_map"],
                    target      = A_glob,
                    ctx         = {"rhs": False, "add": True},
                    integrand   = ker,
                    hook        = None,
                )
            self._last_jacobian = A_glob          # keep for line‑search

        # 2) Residual ---------------------------------------------------
        for ker in self.kernels_F:
            _, Floc, _ = ker.exec(coeffs)

            R_inc = np.zeros_like(R_glob)
            _scatter_element_contribs(
                K_elem      = None,
                F_elem      = Floc,
                J_elem      = None,
                element_ids = ker.static_args["eids"],
                gdofs_map   = ker.static_args["gdofs_map"],
                target      = R_inc,
                ctx         = {"rhs": True, "add": True},
                integrand   = ker,
                hook        = None,
            )
            print(f"{ker.domain:9}: |R|_∞ = {np.linalg.norm(R_inc, np.inf):.3e}")
            R_glob += R_inc

        # 3) Homogeneous Dirichlet rows  (always) -----------------------
        #    OLD: executed only when `need_matrix` was True
        if self.bcs_homog:
            bc_data = dh.get_dirichlet_data(self.bcs_homog)
            if bc_data:
                rows = np.fromiter(bc_data.keys(),  dtype=int)
                vals = np.fromiter(bc_data.values(), dtype=float)

                if need_matrix:
                    # Move prescribed values to RHS and zero the rows/cols
                    bc_vec = np.zeros(ndof);  bc_vec[rows] = vals
                    R_glob -= A_glob @ bc_vec
                    A_glob  = _zero_rows_cols(A_glob, rows)   # ← moved

                # Always override residual entries so ‖R‖ ignores BC rows
                R_glob[rows] = vals                        # NEW

        return (A_glob.tocsr() if need_matrix else None), R_glob



    def _solve_linear_system(self, A: sp.csr_matrix, rhs: np.ndarray) -> np.ndarray:
        if self.lp.backend == "scipy":
            return spla.spsolve(A, rhs)
        else:
            raise ValueError(f"Unknown linear solver backend '{self.lp.backend}'.")


    def _phi(self, vec):                 # ½‖·‖² helper
        return 0.5 * np.dot(vec, vec)

    def _line_search(self, funcs, dU, coeffs, R0_vec, bcs_now):
        """
        A robust "best-effort" backtracking line search. It finds a step
        that satisfies the Armijo condition. If it fails, it returns the
        step that produced the largest observed residual reduction.
        """
        dh = self.dh
        np_ = self.np
        c1 = 1e-4

        snap = [f.nodal_values.copy() for f in funcs]
        phi0 = 0.5 * np.dot(R0_vec, R0_vec)

        # Initialize "best-effort" tracking variables
        best_alpha = 1.0
        best_phi = phi0
        # print(f"initial φ = {phi0:.2e}, starting Armijo search...")

        # This will now work correctly because of the change in _newton_loop
        J = self._last_jacobian
        if J is None:
             print("        Warning: Jacobian not found, cannot perform Armijo search.")
             return dU # Fallback to full step

        g = J.T @ R0_vec
        gTd = np.dot(g, dU)
        if gTd >= 0:
            print("        Warning: Not a descent direction.")
            return np.zeros_like(dU)

        alpha = 1.0
        for i in range(np_.ls_max_iter):
            # Restore state and apply new trial step
            for f, buf in zip(funcs, snap): f.nodal_values[:] = buf
            dh.add_to_functions(alpha * dU, funcs)
            dh.apply_bcs(bcs_now, *funcs)

            _, R_trial = self._assemble_system(coeffs, need_matrix=False)
            # print(f"R_trial_norm_inf : {np.linalg.norm(R_trial, ord=np.inf):.2e}")
            phi_trial = 0.5 * np.dot(R_trial, R_trial)
            # print(f"  Armijo search α = {alpha:.2e}: φ = {phi_trial:.2e} (Δφ = {phi_trial - phi0:.2e})")

            # Keep track of the best result found so far
            if phi_trial < best_phi and alpha > 0:
                best_phi = phi_trial
                # print(f" Current best α = {alpha:.2e}")
                best_alpha = alpha

            # Check the strict Armijo condition
            if phi_trial <= phi0 + c1 * alpha * gTd:
                # Restore state before returning the scaled step
                for f, buf in zip(funcs, snap): f.nodal_values[:] = buf
                print(f"        Armijo search accepted α = {alpha:.2e}")
                return alpha * dU

            alpha *= np_.ls_reduction
        
        # Restore state before returning
        for f, buf in zip(funcs, snap): f.nodal_values[:] = buf

        # If the loop finishes, fall back to the best step we found, if any.
        if best_alpha > 0:
            print(f"        Armijo failed, using best-effort α = {best_alpha:.2e} instead.")
            return best_alpha * dU
        else:
            print(f"        Line search failed catastrophically. No descent found.")
            return np.zeros_like(dU)
    
    def _create_reduced_system(self, K_full, F_full, bcs):
        """
        Reduces the linear system using the pre-computed self.active_dofs.
        This method is now simpler and more robust.
        """
        # We already have self.active_dofs, computed once at the start.
        active_dofs = self.active_dofs
        
        # The rest of the logic is very similar, but no longer needs to find active DOFs.
        full_to_red_map = {old_dof: new_dof for new_dof, old_dof in enumerate(active_dofs)}

        K_red = K_full.tocsr()[np.ix_(active_dofs, active_dofs)]
        F_red = F_full[active_dofs]

        # Apply BCs to the reduced system
        # Get all dirichlet data again to apply to the reduced system
        bc_data = self.dh.get_dirichlet_data(bcs)
        
        # Filter for BCs that are part of the active system
        active_bc_data = {dof: val for dof, val in bc_data.items() if dof in active_dofs}
        
        if active_bc_data:
            # Map full DOF indices to reduced indices
            reduced_rows = np.array([full_to_red_map[dof] for dof in active_bc_data.keys()], dtype=int)
            reduced_vals = np.array(list(active_bc_data.values()), dtype=float)
            
            # Apply to RHS vector F_red
            F_red -= K_red @ np.bincount(reduced_rows, weights=reduced_vals, minlength=len(active_dofs))

            # Zero out rows and columns in the reduced matrix
            K_red_lil = K_red.tolil()
            K_red_lil[reduced_rows, :] = 0
            K_red_lil[:, reduced_rows] = 0
            K_red_lil[reduced_rows, reduced_rows] = 1.0
            K_red = K_red_lil.tocsr()

            # Set values in the RHS vector
            F_red[reduced_rows] = reduced_vals
        
        return K_red, F_red, full_to_red_map

    # ------------------------------------------------------------------
    #  Fast assembly Path
    # ------------------------------------------------------------------
    def _build_reduced_pattern(self):
        """
        Build (once) the CSR sparsity of the reduced system and a per-element
        scatter plan that works even when the number of active local DOFs varies
        across elements (due to BCs/cuts). Stores:
        - self._csr_indptr, self._csr_indices
        - self._elem_pos  : per-kernel list of [ per-element 1D position arrays ]
        - self._elem_lidx : per-kernel list of [ per-element local active idx ]
        """

        if getattr(self, "_pattern_stale", True) is False and hasattr(self, "_csr_indptr"):
            return

        n = len(self.active_dofs)
        if n == 0:
            # degenerate case; keep minimal structures
            self._csr_indptr = np.zeros(1, dtype=np.int32)
            self._csr_indices = np.zeros(0, dtype=np.int32)
            self._elem_pos = [[] for _ in self.kernels_K]
            self._elem_lidx = [[] for _ in self.kernels_K]
            self._pattern_stale = False
            return

        # 1) Collect column sets per reduced row
        rows_cols = [set() for _ in range(n)]
        for ker in self.kernels_K:
            gdofs = ker.static_args["gdofs_map"]
            for e in range(gdofs.shape[0]):
                rmap = self.full_to_red[gdofs[e]]
                rows = rmap[rmap >= 0]
                if rows.size == 0:
                    continue
                cols_set = set(rows.tolist())
                for r in rows:
                    rows_cols[r].update(cols_set)

        # 2) Build CSR (row-wise sorted for determinism)
        indptr = np.zeros(n + 1, dtype=np.int32)
        for i in range(n):
            indptr[i + 1] = indptr[i] + len(rows_cols[i])
        indices = np.empty(indptr[-1], dtype=np.int32)
        for i in range(n):
            cols = sorted(rows_cols[i])
            indices[indptr[i] : indptr[i + 1]] = cols
        self._csr_indptr, self._csr_indices = indptr, indices

        # 3) Map (row, col) → absolute position in CSR "data"
        pos = {}
        for i in range(n):
            s, t = indptr[i], indptr[i + 1]
            for k, c in enumerate(indices[s:t]):
                pos[(i, c)] = s + k

        # 4) Ragged per-element scatter plans (store 1-D position arrays + local idx)
        self._elem_pos = []   # per-kernel: list of 1-D arrays (len = n_act^2)
        self._elem_lidx = []  # per-kernel: list of local indices kept (len = n_act)
        for ker in self.kernels_K:
            gdofs = ker.static_args["gdofs_map"]
            pos_list = []
            lidx_list = []
            for e in range(gdofs.shape[0]):
                full = gdofs[e]
                rmap = self.full_to_red[full]
                mask = rmap >= 0
                rows = rmap[mask]
                if rows.size == 0:
                    pos_list.append(np.empty(0, dtype=np.int32))
                    lidx_list.append(np.empty(0, dtype=np.int32))
                    continue
                # flattened positions, row-major order matching Ke[np.ix_(lidx,lidx)].ravel()
                nact = rows.size
                pflat = np.empty(nact * nact, dtype=np.int32)
                t = 0
                for a in range(nact):
                    ra = rows[a]
                    for b in range(nact):
                        cb = rows[b]
                        pflat[t] = pos[(ra, cb)]
                        t += 1
                pos_list.append(pflat)
                lidx_list.append(np.nonzero(mask)[0].astype(np.int32))
            self._elem_pos.append(pos_list)
            self._elem_lidx.append(lidx_list)

        self._pattern_stale = False


    def _assemble_system_reduced_fast(self, coeffs):
        """
        Assemble reduced (A, R) using the prebuilt CSR pattern and ragged
        per-element scatter plans from _build_reduced_pattern().
        Returns: (A_csr, R)
        """

        if getattr(self, "_pattern_stale", True):
            self._build_reduced_pattern()

        indptr, indices = self._csr_indptr, self._csr_indices
        n = indptr.size - 1
        data = np.zeros(indices.size, dtype=float)
        R = np.zeros(n, dtype=float)

        profile = os.getenv("PYCUTFEM_PROFILE_KERNELS", "").lower() in {"1", "true", "yes"}
        prof_entries = []

        # Matrix (Jacobian) blocks
        for ker, pos_list, lidx_list in zip(self.kernels_K, self._elem_pos, self._elem_lidx):
            t_exec = time.perf_counter()
            Kloc, _, _ = ker.exec(coeffs)  # shape [nel, nloc, nloc]
            exec_time = time.perf_counter() - t_exec
            t_scatter = time.perf_counter()
            for e, (pflat, lidx) in enumerate(zip(pos_list, lidx_list)):
                if pflat.size == 0:
                    continue
                Kel = Kloc[e][np.ix_(lidx, lidx)].ravel()
                data[pflat] += Kel
            scatter_time = time.perf_counter() - t_scatter
            if profile:
                prof_entries.append(
                    ("jacobian", getattr(ker, "domain", "unknown"), exec_time, scatter_time, len(pos_list))
                )

        # Residual blocks
        for ker in self.kernels_F:
            t_exec = time.perf_counter()
            _, Floc, _ = ker.exec(coeffs)  # [nel, nloc]
            exec_time = time.perf_counter() - t_exec
            t_scatter = time.perf_counter()
            gdofs = ker.static_args["gdofs_map"]
            for e in range(gdofs.shape[0]):
                rmap = self.full_to_red[gdofs[e]]
                mask = rmap >= 0
                if not np.any(mask):
                    continue
                rows = rmap[mask]
                np.add.at(R, rows, Floc[e][mask])
            scatter_time = time.perf_counter() - t_scatter
            if profile:
                prof_entries.append(
                    ("residual", getattr(ker, "domain", "unknown"), exec_time, scatter_time, gdofs.shape[0])
                )

        A = sp.csr_matrix((data, indices, indptr), shape=(n, n))

        if profile and prof_entries:
            print("        [profile] kernel timings (type, domain, exec, scatter, n_elem):")
            for kind, domain, exec_time, scatter_time, nelem in prof_entries:
                print(
                    f"          {kind:8} {domain:12} exec={exec_time:.3e}s scatter={scatter_time:.3e}s elems={nelem}"
                )

        return A, R


    
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





#-----------------------------------------------------------------------------
# petsc-like solvers ---------------------------------------------------------
#-----------------------------------------------------------------------------
# ============================================================================
#  PETSc/SNES Inexact Newton-Krylov Solver
# ============================================================================



class PetscSnesNewtonSolver(NewtonSolver):
    """
    PETSc-backed Newton solver that mirrors the Python Newton:
      - Direct linear solve by default (preonly+lu)
      - Backtracking line search (newtonls + bt)
      - Absolute nonlinear tolerance (SNES atol)
      - Reuses SNES/KSP/Mats/Vectors across time steps
      - Assembles reduced residual/Jacobian; gives SNES the true R(u)
    """

    def __init__(self, *args, petsc_options: Optional[Dict] = None, **kwargs):
        if not HAS_PETSC:
            raise RuntimeError("petsc4py is not available; PetscSnesNewtonSolver cannot be used.")
        super().__init__(*args, **kwargs)
        self.petsc_options = dict(petsc_options or {})
        self._petsc_ctx: Dict = {}

        # persistent PETSc objects
        self._snes: Optional[PETSc.SNES] = None
        self._x_red: Optional[PETSc.Vec] = None
        self._r_red: Optional[PETSc.Vec] = None
        self._J: Optional[PETSc.Mat] = None
        self._P: Optional[PETSc.Mat] = None
        self._box_lower_full = None   # full-size np.array (total_dofs) or None
        self._box_upper_full = None
        self._XL = None               # PETSc Vec (reduced) for lower bounds
        self._XU = None               # PETSc Vec (reduced) for upper bounds


        # Sensible defaults; user options may override via setFromOptions()
        self.petsc_options.setdefault("snes_type", "newtonls")
        self.petsc_options.setdefault("snes_linesearch_type", "bt")
        self.petsc_options.setdefault("snes_linesearch_damping", 1.0)

        # Robust baseline linear solve: exact Newton like the Python path
        self.petsc_options.setdefault("ksp_type", "preonly")
        self.petsc_options.setdefault("pc_type", "lu")
        self.petsc_options.setdefault("pc_factor_mat_solver_type", "mumps")  # or superlu_dist/superlu

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def _apply_petsc_options(self) -> None:
        """Load self.petsc_options into PETSc.Options so setFromOptions() sees them."""
        opts = PETSc.Options()
        for k, v in self.petsc_options.items():
            key = k if k.startswith("-") else k
            opts[key] = "" if v is None else str(v)
    
    def set_box_bounds(self, lower=None, upper=None, by_field: dict | None=None):
        """
        Define box bounds on DOFs. Accepts:
        - lower/upper: scalar or full-size numpy arrays (total_dofs)
        - by_field: dict like {'ux': (lo, hi), 'uy': (lo, hi), 'p': (None, None)}
        Any missing bound → ±inf.
        Call BEFORE solve_time_interval().
        """
        n = self.dh.total_dofs
        lo_full = np.full(n, -np.inf, dtype=float)
        hi_full = np.full(n,  np.inf, dtype=float)

        if isinstance(lower, (int, float)): lo_full[:] = lower
        elif isinstance(lower, np.ndarray): lo_full[:] = lower
        if isinstance(upper, (int, float)): hi_full[:] = upper
        elif isinstance(upper, np.ndarray): hi_full[:] = upper

        if by_field:
            for name, (lo, hi) in by_field.items():
                sl = self.dh.get_field_slice(name)
                if lo is not None: lo_full[sl] = lo
                if hi is not None: hi_full[sl] = hi

        # store; we’ll map to reduced each step/size
        self._box_lower_full = lo_full
        self._box_upper_full = hi_full


    def _ensure_snes(self, n_free: int, comm) -> None:
        """Create/resize SNES and persistent objects if needed; set defaults then apply options."""
        if self._snes is not None and self._x_red.getSize() == n_free:
            return

        # --- Vectors (reduced) ---
        self._x_red = PETSc.Vec().create(comm=comm)
        self._x_red.setSizes(n_free)
        self._x_red.setFromOptions()

        self._r_red = PETSc.Vec().create(comm=comm)
        self._r_red.setSizes(n_free)
        self._r_red.setFromOptions()

        # --- Matrices (reduced) ---
        self._J = PETSc.Mat().createAIJ(size=(n_free, n_free), comm=comm)
        self._P = PETSc.Mat().createAIJ(size=(n_free, n_free), comm=comm)

        # Optional preallocation from reduced CSR pattern
        if hasattr(self, "_csr_indptr") and hasattr(self, "_csr_indices"):
            try:
                ia = np.asarray(self._csr_indptr, dtype=PETSc.IntType)
                ja = np.asarray(self._csr_indices, dtype=PETSc.IntType)
                self._J.setPreallocationCSR((ia, ja))
                self._P.setPreallocationCSR((ia, ja))
            except Exception:
                pass

        # --- SNES and callbacks ---
        self._snes = PETSc.SNES().create(comm=comm)
        self._snes.setFunction(self._eval_residual_reduced, self._r_red)
        self._snes.setJacobian(self._eval_jacobian_reduced, self._J, self._P)

        # --- Optional VI bounds (reduced) ---
        if (getattr(self, "_box_lower_full", None) is not None) and (getattr(self, "_box_upper_full", None) is not None):
            lo_red = self._box_lower_full[self.active_dofs].astype(float, copy=True)
            hi_red = self._box_upper_full[self.active_dofs].astype(float, copy=True)

            self._XL = PETSc.Vec().create(comm=comm)
            self._XL.setSizes(n_free)
            self._XL.setFromOptions()
            self._XU = PETSc.Vec().create(comm=comm)
            self._XU.setSizes(n_free)
            self._XU.setFromOptions()

            idx = np.arange(n_free, dtype=PETSc.IntType)
            self._XL.setValues(idx, lo_red, addv=PETSc.InsertMode.INSERT_VALUES)
            self._XU.setValues(idx, hi_red, addv=PETSc.InsertMode.INSERT_VALUES)
            self._XL.assemblyBegin(); self._XL.assemblyEnd()
            self._XU.assemblyBegin(); self._XU.assemblyEnd()

            # Default to VI Newton unless the user chose something else
            self.petsc_options.setdefault("snes_type", "vinewtonrsls")
            # Attach bounds to SNES (now that SNES exists)
            self._snes.setVariableBounds(self._XL, self._XU)

        # --- Programmatic defaults (robust baseline; options may override) ---
        # Use absolute tolerance to mimic your Python Newton's stopping rule
        self._snes.setTolerances(rtol=0.0, atol=self.np.newton_tol, max_it=self.np.max_newton_iter)
        try:
            self._snes.getLineSearch().setType("bt")
        except Exception:
            pass

        ksp = self._snes.getKSP()
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        try:
            pc.setFactorSolverType("mumps")  # or "superlu_dist"/"superlu"
        except Exception:
            pass

        # If you configured a Schur fieldsplit elsewhere, install it now (optional)
        if getattr(self, "_schur_split_cfg", None) is not None:
            self._apply_schur_fieldsplit(ksp)

        # --- Apply user options last so they override all defaults ---
        self._apply_petsc_options()
        self._snes.setFromOptions()


    # ------------------------------------------------------------------ #
    # Newton loop
    # ------------------------------------------------------------------ #
    def _newton_loop(self, funcs, prev_funcs, aux_funcs, bcs_now):
        dh = self.dh
        comm = PETSc.COMM_WORLD

        # Ensure we have the reduced pattern/active dofs
        if getattr(self, "_pattern_stale", True) or not hasattr(self, "_csr_indptr"):
            self._build_reduced_pattern()

        # Apply BCs to the current iterate and form the reduced initial guess
        dh.apply_bcs(bcs_now, *funcs)
        x0_full = np.hstack([f.nodal_values for f in funcs]).copy()
        x0_red = x0_full[self.active_dofs].copy()

        # Context for callbacks
        self._petsc_ctx = dict(
            funcs=funcs, prev_funcs=prev_funcs, aux_funcs=aux_funcs,
            bcs_now=bcs_now, x0_full=x0_full
        )

        # Create/reuse SNES stack for this reduced size
        self._ensure_snes(len(self.active_dofs), comm)

        # Load initial guess into PETSc vector
        idx = np.arange(x0_red.size, dtype=PETSc.IntType)

        # --- PROJECT x0 INTO [XL, XU] if VI is active ---
        if getattr(self, "_XL", None) is not None and getattr(self, "_XU", None) is not None:
            lo = self._XL.getArray(readonly=True)
            hi = self._XU.getArray(readonly=True)
            x0_red = np.minimum(np.maximum(x0_red, lo), hi)

        self._x_red.setValues(idx, x0_red, addv=PETSc.InsertMode.INSERT_VALUES)
        self._x_red.assemblyBegin(); self._x_red.assemblyEnd()


        # Solve the nonlinear system on the reduced space
        self._snes.solve(None, self._x_red)

        # Write back the absolute solution (not an increment)
        x_fin = self._x_red.getArray(readonly=True)
        new_full = x0_full.copy()
        new_full[self.active_dofs] = x_fin
        for f in funcs:
            g = f._g_dofs
            f.set_nodal_values(g, new_full[g])
        dh.apply_bcs(bcs_now, *funcs)

        prev_vals = np.hstack([f.nodal_values for f in prev_funcs])
        return np.hstack([f.nodal_values for f in funcs]) - prev_vals

    # ------------------------------------------------------------------ #
    # SNES callbacks: residual and Jacobian on the reduced space
    # ------------------------------------------------------------------ #
    def _eval_residual_reduced(self, snes, x_red: PETSc.Vec, r_red: PETSc.Vec):
        ctx = self._petsc_ctx

        # Lift reduced iterate to full space on a fresh copy, then apply BCs
        x_full = ctx["x0_full"].copy()
        x_full[self.active_dofs] = x_red.getArray(readonly=True)

        res_funcs = [f.copy() for f in ctx["funcs"]]
        for f in res_funcs:
            g = f._g_dofs
            f.set_nodal_values(g, x_full[g])
        self.dh.apply_bcs(ctx["bcs_now"], *res_funcs)

        coeffs = {f.name: f for f in res_funcs}
        coeffs.update({f.name: f for f in ctx["prev_funcs"]})
        if ctx["aux_funcs"]:
            coeffs.update(ctx["aux_funcs"])

        # Assemble residual R(u) on the reduced space (SNES expects +R)
        _, R_red = self._assemble_system_reduced(coeffs, need_matrix=False)

        r_red.set(0.0)
        idx = np.arange(R_red.size, dtype=PETSc.IntType)
        r_red.setValues(idx, R_red, addv=PETSc.InsertMode.INSERT_VALUES)
        r_red.assemblyBegin(); r_red.assemblyEnd()
        return 0

    def _eval_jacobian_reduced(self, snes, x_red: PETSc.Vec, J: PETSc.Mat, P: PETSc.Mat):
        ctx = self._petsc_ctx

        x_full = ctx["x0_full"].copy()
        x_full[self.active_dofs] = x_red.getArray(readonly=True)

        jac_funcs = [f.copy() for f in ctx["funcs"]]
        for f in jac_funcs:
            g = f._g_dofs
            f.set_nodal_values(g, x_full[g])
        self.dh.apply_bcs(ctx["bcs_now"], *jac_funcs)

        coeffs = {f.name: f for f in jac_funcs}
        coeffs.update({f.name: f for f in ctx["prev_funcs"]})
        if ctx["aux_funcs"]:
            coeffs.update(ctx["aux_funcs"])

        # Assemble reduced Jacobian
        A_red, _ = self._assemble_system_reduced(coeffs, need_matrix=True)

        # Load into PETSc AIJ(s)
        ia = A_red.indptr.astype(PETSc.IntType, copy=False)
        ja = A_red.indices.astype(PETSc.IntType, copy=False)
        a = A_red.data
        try:
            J.setValuesCSR((ia, ja), a)
        except TypeError:
            J.setValuesCSR(ia, ja, a)
        J.assemblyBegin(); J.assemblyEnd()

        if P.handle != J.handle:
            try:
                P.setValuesCSR((ia, ja), a)
            except TypeError:
                P.setValuesCSR(ia, ja, a)
            P.assemblyBegin(); P.assemblyEnd()

        return J, P, PETSc.Mat.Structure.SAME_NONZERO_PATTERN

    # ------------------------------------------------------------------ #
    # Optional: Schur field-split wiring (only if you call set_schur_fieldsplit)
    # ------------------------------------------------------------------ #
    def set_schur_fieldsplit(
        self,
        split_map: dict,
        *,
        schur_fact: str = "full",      # "full", "upper", "lower", "diag"
        schur_pre:  str = "selfp",     # "selfp", "a11", "user"
        sub_pc: dict | None = None,    # e.g. {"u":"hypre", "p":"jacobi"}
    ) -> None:
        self._schur_split_cfg = {
            "split_map": dict(split_map),
            "schur_fact": schur_fact.lower(),
            "schur_pre":  schur_pre.lower(),
            "sub_pc": sub_pc or {},
        }

    def _apply_schur_fieldsplit(self, ksp) -> None:
        pc = ksp.getPC()
        pc.setType("fieldsplit")
        pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)

        cfg = getattr(self, "_schur_split_cfg", None)
        if cfg is None:
            return

        opts = PETSc.Options()
        opts["pc_fieldsplit_schur_fact_type"]    = cfg["schur_fact"]
        opts["pc_fieldsplit_schur_precondition"] = cfg["schur_pre"]

        blocks = []
        for name, fields in cfg["split_map"].items():
            full_idx = []
            for fld in fields:
                full_idx.extend(self.dh.get_field_slice(fld))
            full_idx = np.array(sorted(full_idx), dtype=int)
            red_idx = self.full_to_red[full_idx]
            red_idx = red_idx[red_idx >= 0]
            iset = PETSc.IS().createGeneral(red_idx.astype(PETSc.IntType))
            blocks.append((name, iset))

        pc.setFieldSplitIS(*blocks)
        for name, _ in blocks:
            if f"fieldsplit_{name}_ksp_type" not in self.petsc_options:
                opts[f"fieldsplit_{name}_ksp_type"] = "preonly"
            if f"fieldsplit_{name}_pc_type" not in self.petsc_options:
                opts[f"fieldsplit_{name}_pc_type"] = cfg["sub_pc"].get(name, "jacobi")
    # ------------------------------------------------------------------ #
    # -------------------- Additional methods ---------------------------
    # ------------------------------------------------------------------ #
    def set_vi_on_interface_band(
        self,
        level_set,
        *,
        fields=("ux", "uy"),      # constrain only velocity by default
        side="+",                 # '+' => φ>=0 (fluid), '-' => φ<=0 (solid)
        band_width=1.0,           # thickness ~ band_width * h_interface
        bounds_by_field=None,     # e.g. {"ux": (-Ucap, Ucap), "uy": (-Ucap, Ucap)}
        Ucap=None,                # if given, symmetric cap (-Ucap, +Ucap) for listed fields
        eps=None                  # absolute band half-width (override if you prefer |φ|<=eps)
    ):
        """
        Build VI bounds only for DOFs in an interface band and only for the requested
        fields; all other DOFs remain unbounded (±inf). Call BEFORE solve_time_interval().
        """
        dh   = self.dh
        mesh = dh.mixed_element.mesh

        # 1) Coordinates and level-set at DOFs
        X = dh.get_all_dof_coords()                  # shape (total_dofs, 2)  :contentReference[oaicite:3]{index=3}
        phi = level_set(X)

        # 2) Choose band half-width in physical units
        if eps is None:
            # representative h near the interface (use cut elements if available)
            try:
                cut_ids = mesh.element_bitset("cut").to_indices()
                if len(cut_ids) > 0:
                    h0 = float(np.mean([mesh.element_char_length(int(e)) for e in cut_ids]))
                else:
                    h0 = float(np.mean([mesh.element_char_length(e) for e in range(len(mesh.elements_list))]))
            except Exception:
                h0 = float(np.mean([mesh.element_char_length(e) for e in range(len(mesh.elements_list))]))
            eps = band_width * h0

        # 3) Start with ±inf everywhere
        n = dh.total_dofs
        lo_full = np.full(n, -np.inf, dtype=float)
        hi_full = np.full(n,  np.inf, dtype=float)

        # 4) Field-wise mask inside the band on the requested side
        def _side_ok(arr):
            if side == "+":  return arr >= 0.0
            if side == "-":  return arr <= 0.0
            raise ValueError("side must be '+' or '-'")

        # bounds to apply
        if bounds_by_field is None:
            if Ucap is None:
                raise ValueError("Provide either bounds_by_field or Ucap.")
            bounds_by_field = {f: (-float(Ucap), float(Ucap)) for f in fields}

        for f in fields:
            gdofs = np.asarray(dh.get_field_slice(f), dtype=int)
            band  = (np.abs(phi[gdofs]) <= eps) & _side_ok(phi[gdofs])
            if not np.any(band):
                continue
            lo, hi = bounds_by_field.get(f, (None, None))
            if lo is not None:
                lo_full[gdofs[band]] = float(lo)
            if hi is not None:
                hi_full[gdofs[band]] = float(hi)

        # Leave pressure (or other fields) unbounded by simply not touching them.
        # Store for SNES setup; they’ll be reduced to the active/free DOFs later.
        self.set_box_bounds(lower=lo_full, upper=hi_full)     # uses existing API  :contentReference[oaicite:4]{index=4}








# --- Helper functions for scattering into global NumPy arrays for PETSc ---
# These can be added to the bottom of nonlinear_solver.py

def _scatter_element_contribs_petsc(F_elem, element_ids, gdofs_map, target_vec):
    """Scatters local vectors into a global NumPy vector."""
    for i in range(len(element_ids)):
        gdofs = gdofs_map[i]
        valid = gdofs >= 0
        if not np.any(valid):
            continue
        np.add.at(target_vec, gdofs[valid], F_elem[i][valid])

def _scatter_mat_vec_petsc(K_elem, vec_loc_vals, element_ids, gdofs_map, target_vec):
    """Computes local mat-vec products and scatters them into a global vector."""
    for i in range(len(element_ids)):
        gdofs = gdofs_map[i]
        valid = gdofs >= 0
        if not np.any(valid):
            continue
        
        # Get local part of the input vector
        v_loc = vec_loc_vals[gdofs[valid]]
        
        # Local mat-vec product
        res_loc = K_elem[i][np.ix_(valid, valid)] @ v_loc
        
        # Add to global result vector
        np.add.at(target_vec, gdofs[valid], res_loc)





#-----------------------------------------------------------------------------
# Inexact Newton solver (not working)
#------------------------------------------------------------------------------

if HAS_GIANT:
    class GiantInexactNewtonSolver(NewtonSolver):
        """
        Uses the GIANT Fortran package (matrix-free inexact/semi-smooth Newton)
        as the outer nonlinear solver. Residual/Jacobian-vector products are
        still assembled/evaluated by your fast Numba kernels.
        """
        def _newton_loop(self, funcs, prev_funcs, aux_funcs, bcs_now):

            dh      = self.dh
            n_total = dh.total_dofs
            active  = self.active_dofs         # free dofs (length n_free)
            n_free  = active.size

            # 1) pack the CURRENT iterate into a reduced vector x_red
            def _pack_reduced():
                x_full = np.hstack([f.nodal_values for f in funcs])
                return x_full[active].copy()

            def _unpack_into_funcs(x_red):
                # expand reduced → full, add to fields, re-apply BCs
                dU_full = np.zeros(n_total)
                dU_full[active] = x_red - _pack_reduced()  # delta on free dofs
                dh.add_to_functions(dU_full, funcs)
                dh.apply_bcs(bcs_now, *funcs)

            # 2) Build a shared "coeffs" dict each evaluation uses
            def _current_coeffs():
                coeffs = {f.name: f for f in funcs}
                coeffs.update({f.name: f for f in prev_funcs})
                if aux_funcs:
                    coeffs.update(aux_funcs)
                return coeffs

            # 3) GIANT callbacks ------------------------------------------------
            # IMPORTANT: GIANT calls these many times, so avoid needless allocs.
            # Residual: fcn(n, x, f, ierr, ...)
            # Reuse your local helpers: active, dh, _current_coeffs(), etc.

            def fcn(x_in, n_opt=None, *unused):
                # Snapshot the current fields
                snap = [f.nodal_values.copy() for f in funcs]
                try:
                    # Write reduced iterate into the live functions
                    x_full = np.hstack([buf for buf in snap])
                    x_full[active] = x_in
                    off = 0
                    for f, buf in zip(funcs, snap):
                        nloc = f.nodal_values.size
                        f.nodal_values[:] = x_full[off:off+nloc]; off += nloc
                    dh.apply_bcs(bcs_now, *funcs)

                    # Assemble reduced residual
                    _, R_red = self._assemble_system_reduced(_current_coeffs(), need_matrix=False)
                    return np.asarray(R_red, dtype=np.float64), 0
                except Exception:
                    return np.zeros_like(x_in, dtype=np.float64), 1
                finally:
                    # Restore snapshots
                    for f, buf in zip(funcs, snap):
                        f.nodal_values[:] = buf

            # Optional tiny cache so we don’t rebuild A_red multiple times at same x
            _jac_cache = {"x_id": None, "x_copy": None, "A": None}
            _last_x = None
            _last_A = None

            def muljac(x_in, v_in, n_opt=None, *unused):
                snap = [f.nodal_values.copy() for f in funcs]
                try:
                    # If x_in hasn't changed, reuse cached A_red
                    use_cache = (_jac_cache["x_copy"] is not None
                                    and _jac_cache["x_copy"].shape == x_in.shape
                                    and np.array_equal(_jac_cache["x_copy"], x_in))


                    if not use_cache or _jac_cache["A"] is None:
                        x_full = np.hstack([buf for buf in snap])
                        x_full[active] = x_in
                        off = 0
                        for f, buf in zip(funcs, snap):
                            nloc = f.nodal_values.size
                            f.nodal_values[:] = x_full[off:off+nloc]; off += nloc
                        dh.apply_bcs(bcs_now, *funcs)

                        A_red, _ = self._assemble_system_reduced(_current_coeffs(), need_matrix=True)
                        _jac_cache["A"] = A_red
                        _jac_cache["x_copy"] = x_in.copy()

                    jv = _jac_cache["A"] @ v_in
                    return np.asarray(jv, dtype=np.float64), 0
                except Exception:
                    return np.zeros_like(v_in, dtype=np.float64), 1
                finally:
                    for f, buf in zip(funcs, snap):
                        f.nodal_values[:] = buf


            # 4) Run GIANT on the reduced vector (GIANT updates x0 in place)
            # Make sure dtypes are correct
            x0 = _pack_reduced()
            xscal = np.maximum(1.0, np.abs(x0))
            rtol  = self.np.newton_tol
            iopt  = np.zeros(50, dtype=np.int32)
            ierr  = np.int32(0)
            x0    = np.asarray(x0,    dtype=np.float64)
            xscal = np.asarray(xscal, dtype=np.float64)
            iopt  = np.asarray(iopt,  dtype=np.int32)
            iopt[30] = 1   # print Newton iteration header
            iopt[31] = 1   # print inner GMRES info


            ierr = giant_solver.giant_wrapper_mod.giant_shim(
                x0,                         # x : in/out, length n_free
                xscal,                      # xscal : length n_free
                float(rtol),                # rtol : scalar
                iopt,                       # iopt : int32[50], in/out
                fcn,                        # fcn   : callback (function, not tuple)
                muljac,                     # muljac: callback (function, not tuple)
                n_free,                     # optional; may omit since len(x0) is used
                (),                         # fcn_extra_args
                ()                          # muljac_extra_args
            )
            if int(ierr) != 0:
                raise RuntimeError(f"GIANT failed with ierr={int(ierr)}")

            # 5) Accept GIANT’s iterate: write it back once
            # (delta = x* - x_old on active dofs)
            dU_full = np.zeros(n_total)
            dU_full[active] = x0 - _pack_reduced()
            dh.add_to_functions(dU_full, funcs)
            dh.apply_bcs(bcs_now, *funcs)

            # return the *time-step* increment like your base class
            delta = np.hstack([f.nodal_values - fp.nodal_values for f, fp in zip(funcs, prev_funcs)])
            return delta







#------------------------------------------------------------------------------
# nonlinear_solver.py  ---------------------------------------------------
# --------------------------------------------------------------------------
#  Hybrid Newton–Adam solver
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#  AdaGrad‑Preconditioned Newton with Trust‑Region Switch
# --------------------------------------------------------------------------
class AdamNewtonSolver(NewtonSolver):
    """
    Newton search direction pre‑conditioned by AdaGrad/Adam, with a
    trust‑region fallback to a pure (scaled) gradient step.

    * Keeps quadratic convergence near the solution (when dU_H is small).
    * Never takes an uphill step thanks to the best‑α fallback.
    * Call signature identical to NewtonSolver – no driver changes.
    """

    # --- hyper‑parameters ---------------------------------------------
    beta1 = 0.9
    beta2 = 0.999
    eps   = 1e-8
    eta   = 2.0          # trust‑region radius  (||dU_H|| > η||dU_A||)
    ls_c1 = 1e-4         # Armijo parameter
    ls_min_alpha = 1e-6  # abort search below this

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.m = None
        self.v = None
        self.t = 0

    # ------------------------------------------------------------------
    #  Newton loop (same signature as parent)
    # ------------------------------------------------------------------
    def _newton_loop(self, funcs, prev_funcs, aux_funcs, bcs_now):
        dh     = self.dh
        np_    = self.np

        # allocate Adam buffers lazily
        ndof = sum(f.nodal_values.size for f in funcs)
        if self.m is None or self.m.size != ndof:
            self.m = np.zeros(ndof)
            self.v = np.zeros(ndof)
            self.t = 0

        for it in range(1, np_.max_newton_iter + 1):

            # 1) assemble J, R -----------------------------------------
            coeffs = {f.name: f for f in funcs}
            coeffs.update({f.name: f for f in prev_funcs})
            if aux_funcs:
                coeffs.update(aux_funcs)

            A_full, R_full = self._assemble_system(coeffs, need_matrix=True)
            self._last_jacobian = A_full          # for fallback gradient step
            norm_R = np.linalg.norm(R_full, np.inf)
            print(f"        Newton {it}: |R|_∞ = {norm_R:.2e}")
            if norm_R < np_.newton_tol:
                delta = np.hstack([f.nodal_values - fp.nodal_values
                                   for f, fp in zip(funcs, prev_funcs)])
                return delta

            # 2) gradient & AdaGrad preconditioner ---------------------
            K_red, R_red   = self.restrictor.reduce_system(A_full, R_full,
                                                 self.dh, bcs_now)
            g = A_full.T @ R_full                      # true gradient of ½‖R‖²
            self.t += 1
            self.m = self.beta1 * self.m + (1-self.beta1) * g
            self.v = self.beta2 * self.v + (1-self.beta2) * (g*g)
            m_hat  = self.m / (1 - self.beta1**self.t)
            v_hat  = self.v / (1 - self.beta2**self.t)
            P_diag = 1.0 / (np.sqrt(v_hat) + self.eps)
            P_diag = np.clip(P_diag, 1/20.0, 20.0)       # bounded scaling

            # 3) three candidate directions ----------------------------
            dU_N_red = self._solve_linear_system(K_red, -R_red)   # Newton
            dU_N = self.restrictor.expand_vec(dU_N_red)
            dU_A = -P_diag * m_hat                   # AdaGrad / Adam
            dU_H =  P_diag * dU_N                    # hybrid (scaled Newton)

            # trust‑region switch
            gTS = lambda v: np.dot(g, v)
            candidates = {"hybrid": dU_H, "adam": dU_A, "newton": dU_N,
                          "steepest": -g}
            name, S = min(candidates.items(), key=lambda kv: gTS(kv[1]))

            if gTS(S) >= 0.0:
                raise RuntimeError("No descent direction – try smaller Δt")

            # 4) Armijo back‑tracking (with gᵀS) -----------------------
            ΔU = self._armijo_search(S, g, coeffs, bcs_now, funcs)
            if ΔU.size == 0:                      # gave up – cut Δt outside
                raise RuntimeError("Line search failed – try smaller Δt")
            dh.add_to_functions(ΔU, funcs)
            dh.apply_bcs(bcs_now, *funcs)

        raise RuntimeError("Newton max_iter reached without convergence")

    # ------------------------------------------------------------------
    # Armijo search on a single direction S (Corrected Version)
    # ------------------------------------------------------------------
    def _armijo_search(self, S, g, coeffs, bcs_now, funcs):
        dh = self.dh
        alpha = 1.0
        _, R0 = self._assemble_system(coeffs, need_matrix=False)
        phi0 = 0.5 * np.dot(R0, R0)
        gTS = np.dot(g, S)

        # 1. Save the pristine state at the beginning of the search.
        snap = [f.nodal_values.copy() for f in funcs]

        # 2. Initialize best_alpha to 0.0 for clean failure signaling.
        best_alpha = 0.0
        best_phi = phi0

        while alpha >= self.ls_min_alpha:
            # 3. Always apply the trial step from the clean snapshot.
            for f, buf in zip(funcs, snap):
                f.nodal_values[:] = buf
            dh.add_to_functions(alpha * S, funcs)
            dh.apply_bcs(bcs_now, *funcs)

            _, R_trial = self._assemble_system(coeffs, need_matrix=False)
            phi = 0.5 * np.dot(R_trial, R_trial)

            # Update the best-effort trackers
            if phi < best_phi:
                best_phi = phi
                best_alpha = alpha

            # Armijo condition for sufficient decrease
            if phi <= phi0 + self.ls_c1 * alpha * gTS:
                # Success! Restore state and return the scaled step.
                print(f"        Armijo search accepted α = {alpha:.2e} (φ = {phi:.2e})")
                for f, buf in zip(funcs, snap):
                    f.nodal_values[:] = buf
                return alpha * S

            # No need for an explicit rollback here, as the top of the
            # loop will restore from the snapshot anyway.
            alpha *= 0.5

        # If the loop finishes, the strict condition failed. Use the best-effort result.
        for f, buf in zip(funcs, snap):
            f.nodal_values[:] = buf # Ensure state is restored before exiting.

        if best_alpha > 0.0:
            print(f"        Search failed – using best α = {best_alpha:.3e}")
            return best_alpha * S

        print("        Line search failed – no descent direction.")
        return np.zeros_like(S)  # Signal failure to the caller

    def _maybe_refresh_deformation(self, coeffs: Dict[str, object]) -> None:
        if self.deformation is None or self._deformation_update is None:
            return

        disp = self._deformation_update(coeffs)
        if disp is None:
            return

        disp_arr = np.asarray(disp, dtype=float)
        target = self.deformation.node_displacements
        if disp_arr.shape != target.shape:
            raise ValueError(
                "deformation_update returned displacements with shape "
                f"{disp_arr.shape}, expected {target.shape}"
            )

        if np.allclose(disp_arr, target):
            return

        np.copyto(target, disp_arr)
        self._compile_all_kernels()
