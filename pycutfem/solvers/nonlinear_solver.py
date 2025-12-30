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
import math
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
            valid_full = gdofs >= 0
            rmap = -np.ones_like(gdofs, dtype=int)
            if np.any(valid_full):
                rmap[valid_full] = full_to_red[gdofs[valid_full]]
            rmask = rmap >= 0
            if not np.any(rmask):
                continue
            rows = rmap[rmask]

            # columns use the same element gdofs (bilinear form)
            cmap = rmap
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
            valid_full = gdofs >= 0
            rmap = -np.ones_like(gdofs, dtype=int)
            if np.any(valid_full):
                rmap[valid_full] = full_to_red[gdofs[valid_full]]
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
    # Line-search mode:
    # - "armijo": backtracking on ½‖R‖² with Armijo sufficient decrease
    # - "dealii": backtracking that only requires ‖R‖_∞ to decrease (step-fsi.cc style)
    ls_mode: str = "armijo"
    # Allow deeper backtracking; some stiff problems only descend for very small α
    ls_max_iter: int = 12
    ls_reduction: float = 0.5           # α ← β·α after reject
    # NOTE: A too-large `ls_c1` makes the Armijo test overly strict and can
    # stall Newton (tiny accepted steps). Use a standard small value.
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
    allow_dt_reduction: bool = False    # opt-in adaptive dt reduction
    dt_reduction_factor: float = 0.5    # dt <- factor * dt on rejection
    dt_reduction_threshold: float = 5.0 # reject if ||ΔU|| grows by this factor
    on_dt_change: Optional[Callable[[float], None]] = None
    adjust_max_steps_on_dt_change: bool = True  # keep final_time reachable


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
    def __init__(self, dh, active_dofs, constraint=None):
        self.dh = dh
        self.constraint = constraint
        self.active = np.asarray(active_dofs, dtype=int)
        self.full_size = constraint.n_master if constraint is not None else dh.total_dofs
        self.phys_size = dh.total_dofs
        self.full2red = {int(i): j for j, i in enumerate(self.active)}

    # vectors ----------------------------------------------------------
    def restrict_vec(self, v_full):
        if self.constraint is None:
            return v_full[self.active]
        v_master = self.constraint.restrict_full(v_full)
        return v_master[self.active]

    def expand_vec(self, v_red):
        if self.constraint is None:
            v_full = np.zeros(self.dh.total_dofs, dtype=v_red.dtype)
            v_full[self.active] = v_red
            return v_full
        v_master = np.zeros(self.full_size, dtype=v_red.dtype)
        v_master[self.active] = v_red
        return self.constraint.prolong(v_master)

    # matrices ---------------------------------------------------------
    def restrict_mat(self, A_full):
        aid = self.active
        if self.constraint is None:
            return A_full.tocsr()[np.ix_(aid, aid)]
        A_master = self.constraint.E_T @ (A_full @ self.constraint.E)
        return A_master.tocsr()[np.ix_(aid, aid)]

    # systems (full → reduced with BCs already handled in the full space)
    def reduce_system(self, A_full, R_full, dh, bcs):
        if self.constraint is None:
            K_red = self.restrict_mat(A_full)
            F_red = self.restrict_vec(R_full)
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

        # Constraint-aware path: condense first, then apply BCs in master space
        A_master = self.constraint.E_T @ (A_full @ self.constraint.E)
        R_master = self.constraint.E_T @ R_full

        bc_data_full = dh.get_dirichlet_data(bcs)
        bc_master = self.constraint.project_dirichlet(bc_data_full)
        if bc_master:
            rows_master = np.fromiter(bc_master.keys(), dtype=int)
            vals_master = np.fromiter(bc_master.values(), dtype=float)
            if rows_master.size:
                R_master -= A_master @ np.bincount(
                    rows_master, weights=vals_master, minlength=self.constraint.n_master
                )
                A_master = _zero_rows_cols(A_master, rows_master)
                R_master[rows_master] = vals_master

        A_red = A_master.tocsr()[np.ix_(self.active, self.active)]
        R_red = R_master[self.active]
        return A_red, R_red


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
        # Optional hanging-node constraints → master space
        self.constraints = None
        if hasattr(self.dh, "build_hanging_node_constraints"):
            try:
                self.constraints = self.dh.build_hanging_node_constraints()
            except Exception:
                self.constraints = None
        ndof_effective = self.constraints.n_master if self.constraints is not None else self.dh.total_dofs

        def _map_to_master(ids):
            if self.constraints is None:
                return set(ids)
            return self.constraints.to_master_set(ids)
        # 1) What DOFs are “touched” by Restriction (if any)?
        active_by_restr, has_restriction = analyze_active_dofs(
            self.equation, self.dh, self.me, bcs_for_active
        )
        ndof = self.dh.total_dofs
        bc_dofs = set(self.dh.get_dirichlet_data(bcs_for_active).keys())

        # Candidate DOFs that would remain without explicit drops
        candidate_master = _map_to_master(active_by_restr) if has_restriction else set(range(ndof_effective))

        # Optional DOF tags (e.g. CutFEM inactive regions) are honoured here.
        inactive_dofs: set[int] = set()
        dh_tags = getattr(self.dh, "dof_tags", None)
        if dh_tags:
            inactive_dofs = set(dh_tags.get("inactive", set()))
        # Remove DOFs tagged inactive but not already fixed by Dirichlet BCs
        inactive_free = inactive_dofs - bc_dofs
        inactive_removed = len(inactive_free & set(active_by_restr if has_restriction else range(ndof)))

        # 2) Free DOFs = candidate DOFs \ (Dirichlet ∪ inactive)
        bc_master = _map_to_master(bc_dofs)
        inactive_master = _map_to_master(inactive_free)
        free_set = (candidate_master - bc_master) - inactive_master
        free = sorted(free_set)

        self.active_dofs = np.asarray(free, dtype=int)
        nfree = self.active_dofs.size

        # 3) Maps full ↔ reduced
        self.full_to_red = -np.ones(ndof_effective, dtype=int)
        self.full_to_red[self.active_dofs] = np.arange(nfree, dtype=int)
        self.red_to_full = self.active_dofs

        # 4) Always run the reduced path when there are any fixed DOFs
        self.use_reduced = (nfree < ndof_effective)
        if not self.use_reduced:
            print("NewtonSolver: Operating on the full, unrestricted system.")
        else:
            print(f"NewtonSolver: Reduced system with {nfree}/{ndof_effective} DOFs.")

        # Optional sanity log:
        print(f"  Dirichlet DOFs detected: {len(bc_dofs)}; Free DOFs: {nfree}")
        if inactive_dofs:
            print(f"  Inactive DOFs tagged: {len(inactive_dofs)} (unique)")
            if inactive_removed:
                print(f"  Inactive DOFs dropped: {inactive_removed}")
            else:
                print("  Inactive DOFs already excluded by restriction domains.")
        print(f"NewtonSolver using backend '{self.backend}'.")
        # Build once: CSR structure & per-element scatter plan for reduced system
        self._build_reduced_pattern()

        self.restrictor = _ActiveReducer(self.dh, self.active_dofs, constraint=self.constraints)




    def _compile_all_kernels(self) -> None:
        """(Re)compile residual and Jacobian kernels with current metadata."""
        if self.backend == "jit":
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
            self._python_fc = None
        elif self.backend == "python":
            # Python backend: defer to the pure-Python FormCompiler at assembly time.
            from pycutfem.ufl.compilers import FormCompiler

            self._python_fc = FormCompiler(
                self.dh, quadrature_order=self.quad_order, backend="python"
            )
            self.kernels_K = []
            self.kernels_F = []
        else:
            raise ValueError(f"Unknown backend '{self.backend}'.")

        self._pattern_stale = True
        self._last_jacobian = None
        for attr in ("_csr_indptr", "_csr_indices", "_elem_pos", "_elem_lidx"):
            if hasattr(self, attr):
                delattr(self, attr)

    def refresh_levelset_kernels(self, level_set):
        """
        Refresh precomputed static arguments for kernels that depend on a moving
        level set without re-JIT compilation. Marks the scatter pattern as stale
        so it will be rebuilt on the next assembly.
        """
        if self.backend != "jit":
            return
        t0 = time.perf_counter()
        changed = 0
        for ker in list(getattr(self, "kernels_K", [])) + list(getattr(self, "kernels_F", [])):
            if getattr(ker, "level_set", None) is not level_set:
                continue
            try:
                updated = ker.refresh(level_set)
            except Exception:
                updated = False
            if updated:
                changed += 1
        if changed:
            self._pattern_stale = True
            print(f"[jit] refreshed {changed} kernels for moving level set in {time.perf_counter() - t0:.3f}s")
    def _python_form_compiler(self):
        """Lazily construct the pure-Python FormCompiler."""
        if getattr(self, "_python_fc", None) is None:
            from pycutfem.ufl.compilers import FormCompiler

            self._python_fc = FormCompiler(
                self.dh, quadrature_order=self.quad_order, backend="python"
            )
        return self._python_fc

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
        delta_U = np.zeros(dh.total_dofs)

        t_start = time.perf_counter()
        step = 0
        t_n = 0.0
        prev_delta_inf: float | None = None
        last_dt: float | None = None
        while step < time_params.max_steps and t_n < time_params.final_time:
            dt = float(time_params.dt)
            if last_dt is None or not math.isclose(dt, last_dt, rel_tol=0.0, abs_tol=0.0):
                on_dt_change = getattr(time_params, "on_dt_change", None)
                if callable(on_dt_change):
                    on_dt_change(dt)
                last_dt = dt

            # Predictor: copy previous solution ---------------------
            for f, f_prev in zip(functions, prev_functions):
                f.nodal_values[:] = f_prev.nodal_values[:]

            # Time‑dependent BCs -----------------------------------
            # For theta-schemes, apply time-dependent Dirichlet data at t_{n+θ}
            # (implicit Euler θ=1 → end-of-step, CN θ=0.5 → midpoint).
            t_bc = t_n + float(getattr(time_params, "theta", 1.0)) * dt
            bcs_now = self._freeze_bcs(self.bcs, t_bc)
            dh.apply_bcs(bcs_now, *functions)

            # Newton loop -----------------------------------------
            try:
                delta_U, converged, n_iters = self._newton_loop(functions, prev_functions, aux_functions, bcs_now)
                reduce_dt = False
            except Exception as e:
                reduce_dt = True
                print(f"    Newton failed at step {step+1} with dt={dt:.3e}: {e}")
            # Crieteria if converged but number of Newton iterations are more than 20, reduce dt
            if (converged and n_iters >= 20) or reduce_dt:
                if not bool(getattr(time_params, "allow_dt_reduction", False)):
                    raise RuntimeError(
                        "Time step reduction requested but disabled. "
                        "Reduce dt manually or enable allow_dt_reduction with on_dt_change."
                    )
                on_dt_change = getattr(time_params, "on_dt_change", None)
                if not callable(on_dt_change):
                    raise RuntimeError(
                        "Time step reduction requires TimeStepperParameters.on_dt_change "
                        "to keep dt-dependent coefficients in sync."
                    )
                factor = float(getattr(time_params, "dt_reduction_factor", 0.5))
                time_params.dt = float(time_params.dt) * factor
                if bool(getattr(time_params, "adjust_max_steps_on_dt_change", True)):
                    try:
                        final_time = float(time_params.final_time)
                    except Exception:
                        final_time = None
                    if final_time is not None and final_time > t_n:
                        remaining = final_time - t_n
                        extra = int(math.ceil(remaining / max(float(time_params.dt), 1.0e-16)))
                        time_params.max_steps = max(int(time_params.max_steps), int(step + extra))
                on_dt_change(float(time_params.dt))
                last_dt = float(time_params.dt)
                print(f"    Reducing Δt → {time_params.dt:.3e} due to slow Newton convergence and retrying.")
                continue
            delta_inf = float(np.linalg.norm(delta_U, ord=np.inf))
            print(f"    Time step {step+1}: ΔU = {delta_inf:.2e}")
            
            # Post-step refiner (VI clip) **before** promotion so prev matches clipped state
            if post_step_refiner is not None:
                post_step_refiner(step, bcs_now, functions, prev_functions)

            # # Reject and retry with smaller Δt if the update blows up
            # if step > 0 and prev_delta_inf is not None and prev_delta_inf > 0.0:
            #     threshold = float(getattr(time_params, "dt_reduction_threshold", 5.0))
            #     if delta_inf > threshold * prev_delta_inf:
            #         if not bool(getattr(time_params, "allow_dt_reduction", False)):
            #             raise RuntimeError(
            #                 "Time step reduction requested but disabled. "
            #                 "Reduce dt manually or enable allow_dt_reduction with on_dt_change."
            #             )
            #         on_dt_change = getattr(time_params, "on_dt_change", None)
            #         if not callable(on_dt_change):
            #             raise RuntimeError(
            #                 "Time step reduction requires TimeStepperParameters.on_dt_change "
            #                 "to keep dt-dependent coefficients in sync."
            #             )
            #         factor = float(getattr(time_params, "dt_reduction_factor", 0.5))
            #         time_params.dt = float(time_params.dt) * factor
            #         if bool(getattr(time_params, "adjust_max_steps_on_dt_change", True)):
            #             try:
            #                 final_time = float(time_params.final_time)
            #             except Exception:
            #                 final_time = None
            #             if final_time is not None and final_time > t_n:
            #                 remaining = final_time - t_n
            #                 extra = int(math.ceil(remaining / max(float(time_params.dt), 1.0e-16)))
            #                 time_params.max_steps = max(int(time_params.max_steps), int(step + extra))
            #         on_dt_change(float(time_params.dt))
            #         last_dt = float(time_params.dt)
            #         print(f"    Rejecting step {step+1}; reducing Δt → {time_params.dt:.3e} and retrying.")
            #         continue

            # Accept: promote current → previous
            for f_prev, f in zip(prev_functions, functions):
                f_prev.nodal_values[:] = f.nodal_values[:]
            prev_delta_inf = delta_inf

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
            t_n += dt
            step += 1

        elapsed = time.perf_counter() - t_start
        return delta_U, step, elapsed

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
        self._current_bcs = bcs_now
        dh = self.dh
        ndof_eff = getattr(self.restrictor, "full_size", dh.total_dofs)

        # Quick safety: make sure we actually have a reduced set
        # (This is only for logging; we still run the reduced pipeline.)
        nfree = len(self.active_dofs)
        if nfree == ndof_eff:
            print("        [warn] active_dofs == total_dofs — reduced path = full size."
                " Check that bcs_homog correctly marks Dirichlet nodes.")

        self._last_iter_timings = []
        totals = {"assembly": 0.0, "linear_solve": 0.0, "line_search": 0.0}
        temp_t0 = time.perf_counter()
        converged = False
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
                self.full_to_red = -np.ones(ndof_eff, dtype=int)
                self.full_to_red[self.active_dofs] = np.arange(self.active_dofs.size, dtype=int)
                self.red_to_full = self.active_dofs
                self._pattern_stale = True
                self.restrictor = _ActiveReducer(self.dh, self.active_dofs, constraint=getattr(self, "constraints", None))
                # rebuild reduced CSR scatter pattern
                self._build_reduced_pattern()
                # re-assemble on the cleaned support
                t_asm = time.perf_counter()
                A_red, R_red = self._assemble_system_reduced(current, need_matrix=True)
                assembly_time += time.perf_counter() - t_asm

            # Log residual with a full-size view (zeros on fixed DOFs) for readability
            R_full = np.zeros(ndof_eff)
            R_full[self.active_dofs] = R_red
            norm_R = np.linalg.norm(R_full, ord=np.inf)
            t_current = time.perf_counter()
            t_iteration = t_current - temp_t0
            temp_t0 = t_current
            print(f"        Newton {it+1}: |R|_∞ = {norm_R:.2e}, time = {t_iteration}s")
            if norm_R <= self.np.newton_tol:
                # Converged — return *time-step* increment for all fields
                converged = True
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
                        "residual_norm": norm_R,
                        "converged": True,
                    }
                )
                self._last_iteration_totals = totals
                print(
                    "          timings: assembly={:.3e}s, solve={:.3e}s, line-search={:.3e}s".format(
                        assembly_time, 0.0, 0.0
                    )
                )
                return delta, converged, it + 1
            linear_time = 0.0
            line_search_time = 0.0

            # 2) Compute reduced Newton direction
            t_lin = time.perf_counter()
            dU_red = self._solve_linear_system(A_red, -R_red)
            linear_time = time.perf_counter() - t_lin
            if os.getenv("PYCUTFEM_NEWTON_TRACE", "").lower() in {"1", "true", "yes"}:
                lin_inf = float(np.linalg.norm(A_red @ dU_red + R_red, ord=np.inf))
                du_inf = float(np.linalg.norm(dU_red, ord=np.inf))
                dU_full_dbg = self.restrictor.expand_vec(dU_red)
                field_norms = []
                for fld in getattr(self.dh, "field_names", []):
                    try:
                        sl = self.dh.get_field_slice(fld)
                    except Exception:
                        continue
                    if sl is None or len(sl) == 0:
                        continue
                    field_norms.append(f"{fld}:{np.linalg.norm(dU_full_dbg[sl], ord=np.inf):.2e}")
                extra = ("  [" + ", ".join(field_norms) + "]") if field_norms else ""
                print(f"        [lin] ‖A·δU+R‖∞={lin_inf:.3e}  ‖δU‖∞={du_inf:.3e}{extra}")

            if os.getenv("PYCUTFEM_DIRDERIV_CHECK", "").lower() in {"1", "true", "yes"}:
                # Directional derivative check: (R(u+ε·δ) - R(u))/ε ≈ J·δ = -R(u)
                eps = float(os.getenv("PYCUTFEM_DIRDERIV_EPS", "1e-6"))
                snap = [f.nodal_values.copy() for f in funcs]
                dU_full = self.restrictor.expand_vec(eps * dU_red)
                dh.add_to_functions(dU_full, funcs)
                dh.apply_bcs(bcs_now, *funcs)
                _, R_eps = self._assemble_system_reduced(current, need_matrix=False)
                for f, buf in zip(funcs, snap):
                    f.nodal_values[:] = buf
                derr = (R_eps - R_red) / eps + R_red
                dir_err = float(np.linalg.norm(derr, ord=np.inf))
                worst_red = int(np.argmax(np.abs(derr))) if derr.size else -1
                worst_full = int(self.active_dofs[worst_red]) if worst_red >= 0 else -1
                worst_val = float(derr[worst_red]) if worst_red >= 0 else 0.0
                field = None
                try:
                    field = getattr(self.dh, "_dof_to_node_map", {}).get(worst_full, (None, None))[0]
                except Exception:
                    field = None
                fmsg = f", field={field}" if field is not None else ""
                print(
                    f"        [dd] ‖(R(u+ε·δ)-R(u))/ε + R(u)‖∞={dir_err:.3e} "
                    f"(ε={eps:.1e}, worst_gdof={worst_full}, worst_val={worst_val:.3e}{fmsg})"
                )

            # 3) Optional Armijo backtracking in reduced space (no BC re-application inside)
            if getattr(self.np, "line_search", False):
                t_ls = time.perf_counter()
                dU_red = self._line_search_reduced(A_red, R_red, dU_red, funcs, current, bcs_now)
                line_search_time = time.perf_counter() - t_ls

            # 4) Apply accepted step: expand reduced → full, update fields, then re-impose BCs
            dU_full = self.restrictor.expand_vec(dU_red)

            if os.getenv("PYCUTFEM_CHECK_DIRICHLET_INCREMENT", "").lower() in {"1", "true", "yes"}:
                try:
                    bc_rows = np.fromiter(dh.get_dirichlet_data(bcs_now).keys(), dtype=int)
                    if bc_rows.size:
                        max_bc_du = float(np.max(np.abs(dU_full[bc_rows])))
                        print(f"        [bc] max|δU| on Dirichlet DOFs = {max_bc_du:.3e}")
                except Exception as exc:
                    print(f"        [bc] Dirichlet increment check skipped: {exc}")
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
                    "residual_norm": norm_R,
                    "converged": False,
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
        mode = getattr(np_, "ls_mode", "armijo")
        c1   = getattr(np_, "ls_c1", 1.0e-4)

        if mode not in {"armijo", "dealii"}:
            raise ValueError(f"Unknown line-search mode '{mode}'.")

        # Deal.II style: accept the first step that reduces the infinity norm of the residual.
        if mode == "dealii":
            trace = os.getenv("PYCUTFEM_LS_TRACE", "").lower() in {"1", "true", "yes"}
            # Warm-start: reuse last accepted α (try slightly larger first).
            alpha = float(getattr(self, "_ls_alpha_prev", 1.0))
            alpha = min(1.0, alpha / float(np_.ls_reduction))
            norm0 = float(np.linalg.norm(R_red, ord=np.inf))
            best_alpha, best_norm = 0.0, norm0
            snap = [f.nodal_values.copy() for f in funcs]

            for _ in range(np_.ls_max_iter):
                for f, buf in zip(funcs, snap):
                    f.nodal_values[:] = buf
                dU_full = self.restrictor.expand_vec(alpha * S_red)
                dh.add_to_functions(dU_full, funcs)
                dh.apply_bcs(bcs_now, *funcs)

                _, R_try = self._assemble_system_reduced(coeffs, need_matrix=False)
                norm_try = float(np.linalg.norm(R_try, ord=np.inf))
                if trace:
                    print(f"        [ls] α={alpha:.6e}  ‖R‖∞={norm_try:.6e}  (‖R‖∞0={norm0:.6e})")

                if norm_try < best_norm:
                    best_norm, best_alpha = norm_try, alpha
                if norm_try < norm0:
                    for f, buf in zip(funcs, snap):
                        f.nodal_values[:] = buf
                    print(f"        Line search accepted α = {alpha:.2e} (‖R‖∞ decreased)")
                    self._ls_alpha_prev = alpha
                    return alpha * S_red

                alpha *= np_.ls_reduction

            for f, buf in zip(funcs, snap):
                f.nodal_values[:] = buf
            if best_alpha > 0.0:
                print(f"        Line search failed, using best-effort α = {best_alpha:.2e} (‖R‖∞ decreased).")
                self._ls_alpha_prev = best_alpha
                return best_alpha * S_red
            print("        Line search failed – no decreasing step found; taking α = 0.")
            return np.zeros_like(S_red)

        # Gradient g_f = J_ff^T R_f  (reduced variables only)
        g = A_red.T @ R_red
        gTS = float(g @ S_red)
        if gTS >= 0.0:
            # Newton direction is not a descent; fall back to steepest descent.
            # This keeps the iteration moving instead of taking a zero step.
            print("        Warning: Not a descent direction in reduced space; using steepest descent.")
            S_red = -g
            gTS = float(g @ S_red)  # = -||g||^2 <= 0

        phi0 = 0.5 * float(R_red @ R_red)
        # Warm-start: reuse last accepted α (try slightly larger first).
        alpha = float(getattr(self, "_ls_alpha_prev", 1.0))
        alpha = min(1.0, alpha / float(np_.ls_reduction))
        best_alpha, best_phi = 0.0, phi0

        # Snapshot the *full* iterate
        snap = [f.nodal_values.copy() for f in funcs]

        for _ in range(np_.ls_max_iter):
            # Trial update: expand to full, apply, and re-impose BCs
            for f, buf in zip(funcs, snap): f.nodal_values[:] = buf
            dU_full = self.restrictor.expand_vec(alpha * S_red)
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
                self._ls_alpha_prev = alpha
                return alpha * S_red

            alpha *= np_.ls_reduction

        # Fallback (best effort seen)
        for f, buf in zip(funcs, snap): f.nodal_values[:] = buf
        if best_alpha > 0.0:
            print(f"        Armijo failed, using best-effort α = {best_alpha:.2e}.")
            self._ls_alpha_prev = best_alpha
            return best_alpha * S_red
        # No decrease found; still take the smallest tried step to avoid stagnation.
        min_alpha = alpha
        print(f"        Line search failed – taking minimal step α = {min_alpha:.2e}.")
        self._ls_alpha_prev = min_alpha
        return min_alpha * S_red

    # ------------------------------------------------------------------
    #  Reduced Linear system & BC handling
    # ------------------------------------------------------------------
    def _assemble_system_reduced(self, coeffs, *, need_matrix: bool = True):
        if self.backend == "python":
            return self._assemble_system_reduced_python(coeffs, need_matrix=need_matrix)
        if getattr(self, "constraints", None) is not None:
            return self._assemble_system_with_constraints(coeffs, need_matrix=need_matrix)
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
                gdofs = ker.static_args["gdofs_map"]
                for e in range(gdofs.shape[0]):
                    full = gdofs[e]
                    valid_full = full >= 0
                    if not np.any(valid_full):
                        continue
                    rmap = -np.ones_like(full, dtype=int)
                    rmap[valid_full] = self.full_to_red[full[valid_full]]
                    m = rmap >= 0
                    if np.any(m):
                        np.add.at(R_red, rmap[m], Floc[e][m])
            return None, R_red

    def _assemble_full_system_python(self, apply_bcs=None):
        """
        Assemble the full-space Jacobian and residual using the Python backend.
        Dirichlet conditions are applied only if *apply_bcs* is provided.
        """
        fc = self._python_form_compiler()
        return fc.assemble(self.equation, bcs=apply_bcs)

    def _assemble_system_reduced_python(self, coeffs, *, need_matrix: bool = True):
        """
        Python-backend assembly: build the full system once and condense to the
        active DOFs (and optional constraints) via the existing reducer.
        """
        self._maybe_refresh_deformation(coeffs)
        A_full, R_full = self._assemble_full_system_python(apply_bcs=None)

        # Prefer the time-frozen BCs when available
        bcs_apply = getattr(self, "_current_bcs", None)
        if bcs_apply is None:
            bcs_apply = self.bcs_homog if self.bcs_homog else self.bcs

        A_red, R_red = self.restrictor.reduce_system(A_full, R_full, self.dh, bcs_apply)
        if need_matrix:
            return A_red, R_red
        return None, R_red

    def _assemble_system_with_constraints(self, coeffs, *, need_matrix: bool = True):
        """
        Assemble in the full space and condense hanging nodes via Eᵀ A E / Eᵀ R.
        """
        A_full, R_full = self._assemble_system_raw(coeffs, need_matrix=True)
        if need_matrix:
            A_red, R_red = self.restrictor.reduce_system(
                A_full, R_full, self.dh, self.bcs_homog if self.bcs_homog else self.bcs
            )
            return A_red, R_red
        # Residual-only path: condense and select active entries
        R_master = self.constraints.restrict_full(R_full)
        bc_map = self.constraints.project_dirichlet(self.dh.get_dirichlet_data(self.bcs_homog or self.bcs))
        if bc_map:
            rows = np.fromiter(bc_map.keys(), dtype=int)
            vals = np.fromiter(bc_map.values(), dtype=float)
            R_master[rows] = vals
        return None, R_master[self.active_dofs]


    # ------------------------------------------------------------------
    #  Linear system & BC handling        (REF‑ACTORED)
    # ------------------------------------------------------------------
    def _assemble_system_raw(self, coeffs, *, need_matrix: bool = True):
        """
        Assemble full-space Jacobian and residual *without* applying Dirichlet BCs.
        Used when hanging-node condensation is handled separately.
        """
        if self.backend == "python":
            self._maybe_refresh_deformation(coeffs)
            A_full, R_full = self._assemble_full_system_python(apply_bcs=None)
            return (A_full if need_matrix else None), R_full

        self._maybe_refresh_deformation(coeffs)
        ndof = self.dh.total_dofs
        A_glob = sp.lil_matrix((ndof, ndof)) if need_matrix else None
        R_glob = np.zeros(ndof)

        if need_matrix:
            for ker in self.kernels_K:
                Kloc, _, _ = ker.exec(coeffs)
                _scatter_element_contribs(
                    K_elem=Kloc,
                    F_elem=None,
                    J_elem=None,
                    element_ids=ker.static_args["eids"],
                    gdofs_map=ker.static_args["gdofs_map"],
                    target=A_glob,
                    ctx={"rhs": False, "add": True},
                    integrand=ker,
                    hook=None,
                )
            self._last_jacobian = A_glob

        for ker in self.kernels_F:
            _, Floc, _ = ker.exec(coeffs)
            R_inc = np.zeros_like(R_glob)
            _scatter_element_contribs(
                K_elem=None,
                F_elem=Floc,
                J_elem=None,
                element_ids=ker.static_args["eids"],
                gdofs_map=ker.static_args["gdofs_map"],
                target=R_inc,
                ctx={"rhs": True, "add": True},
                integrand=ker,
                hook=None,
            )
            R_glob += R_inc

        return (A_glob.tocsr() if need_matrix else None), R_glob

    def _assemble_system(self, coeffs, *, need_matrix: bool = True):
        """
        Assemble global Jacobian A (optional) and residual R.

        * When ``need_matrix=False`` it returns (None, R) but **still**
          enforces homogeneous Dirichlet rows in *R* so that line‑search
          evaluations are meaningful.
        """
        if self.backend == "python":
            self._maybe_refresh_deformation(coeffs)
            A_glob, R_glob = self._assemble_full_system_python(apply_bcs=None)
            dh = self.dh
            ndof = dh.total_dofs

            if self.bcs_homog:
                bc_data = dh.get_dirichlet_data(self.bcs_homog)
                if bc_data:
                    rows = np.fromiter(bc_data.keys(), dtype=int)
                    vals = np.fromiter(bc_data.values(), dtype=float)

                    if need_matrix and A_glob is not None:
                        bc_vec = np.zeros(ndof)
                        bc_vec[rows] = vals
                        R_glob -= A_glob @ bc_vec
                        A_glob = _zero_rows_cols(A_glob, rows)

                    R_glob[rows] = vals

            return (A_glob if need_matrix else None), R_glob

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
        backend = str(getattr(self.lp, "backend", "scipy") or "scipy").lower()
        if backend == "scipy":
            return spla.spsolve(A, rhs)
        if backend == "petsc":
            if not HAS_PETSC:
                raise RuntimeError(
                    "LinearSolverParameters.backend='petsc' requested but petsc4py is not available."
                )
            return self._solve_linear_system_petsc(A, rhs)
        raise ValueError(f"Unknown linear solver backend '{self.lp.backend}'.")

    def _solve_linear_system_petsc(self, A: sp.csr_matrix, rhs: np.ndarray) -> np.ndarray:
        """
        Solve A x = rhs using PETSc KSP on COMM_SELF (serial) while reusing
        the matrix sparsity pattern across Newton iterations.
        """
        from petsc4py import PETSc  # type: ignore

        if not sp.isspmatrix_csr(A):
            A = A.tocsr()

        n = int(A.shape[0])
        if int(A.shape[1]) != n:
            raise ValueError("PETSc linear solve expects a square matrix.")

        cache = getattr(self, "_petsc_linear_cache", None)
        if cache is None or cache.get("n") != n:
            comm = PETSc.COMM_SELF
            mat = PETSc.Mat().createAIJ(size=(n, n), comm=comm)
            # Preallocate once using the CSR pattern (indices/indptr).
            ia = np.asarray(A.indptr, dtype=PETSc.IntType)
            ja = np.asarray(A.indices, dtype=PETSc.IntType)
            try:
                mat.setPreallocationCSR((ia, ja))
            except Exception:
                pass
            mat.setUp()

            ksp = PETSc.KSP().create(comm=comm)
            ksp.setOperators(mat)
            # Robust defaults: exact Newton step.
            ksp.setType("preonly")
            pc = ksp.getPC()
            pc.setType("lu")
            # Prefer MUMPS when available; otherwise let PETSc pick.
            try:
                pc.setFactorSolverType("mumps")
            except Exception:
                pass
            # Allow users to override via PETSc options (e.g. -pc_factor_mat_solver_type superlu_dist).
            ksp.setFromOptions()

            b = PETSc.Vec().createSeq(n, comm=comm)
            x = PETSc.Vec().createSeq(n, comm=comm)

            cache = {"n": n, "mat": mat, "ksp": ksp, "b": b, "x": x}
            self._petsc_linear_cache = cache

        mat: PETSc.Mat = cache["mat"]
        ksp: PETSc.KSP = cache["ksp"]
        b: PETSc.Vec = cache["b"]
        x: PETSc.Vec = cache["x"]

        # Update numeric values (pattern is the same).
        mat.zeroEntries()
        ia = np.asarray(A.indptr, dtype=PETSc.IntType)
        ja = np.asarray(A.indices, dtype=PETSc.IntType)
        a = np.asarray(A.data, dtype=np.float64)
        try:
            mat.setValuesCSR((ia, ja), a)
        except TypeError:
            mat.setValuesCSR(ia, ja, a)
        mat.assemblyBegin()
        mat.assemblyEnd()

        # Load RHS and solve.
        b_arr = b.getArray()
        b_arr[:] = rhs
        x.set(0.0)
        ksp.solve(b, x)
        return x.getArray(readonly=True).copy()


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
        mode = getattr(np_, "ls_mode", "armijo")
        c1 = getattr(np_, "ls_c1", 1.0e-4)

        if mode not in {"armijo", "dealii"}:
            raise ValueError(f"Unknown line-search mode '{mode}'.")

        snap = [f.nodal_values.copy() for f in funcs]
        phi0 = 0.5 * np.dot(R0_vec, R0_vec)
        norm0 = float(np.linalg.norm(R0_vec, ord=np.inf))

        if mode == "dealii":
            alpha = 1.0
            best_alpha, best_norm = 0.0, norm0
            for _ in range(np_.ls_max_iter):
                for f, buf in zip(funcs, snap):
                    f.nodal_values[:] = buf
                dh.add_to_functions(alpha * dU, funcs)
                dh.apply_bcs(bcs_now, *funcs)
                _, R_trial = self._assemble_system(coeffs, need_matrix=False)
                norm_trial = float(np.linalg.norm(R_trial, ord=np.inf))
                if norm_trial < best_norm:
                    best_norm, best_alpha = norm_trial, alpha
                if norm_trial < norm0:
                    for f, buf in zip(funcs, snap):
                        f.nodal_values[:] = buf
                    print(f"        Line search accepted α = {alpha:.2e} (‖R‖∞ decreased)")
                    return alpha * dU
                alpha *= np_.ls_reduction

            for f, buf in zip(funcs, snap):
                f.nodal_values[:] = buf
            if best_alpha > 0.0:
                print(f"        Line search failed, using best-effort α = {best_alpha:.2e} (‖R‖∞ decreased).")
                return best_alpha * dU
            print("        Line search failed – no decreasing step found; taking α = 0.")
            return np.zeros_like(dU)

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

        if self.backend == "python":
            # Python backend assembles in the full space; reduced scatter is unused.
            n = len(getattr(self, "active_dofs", []))
            self._csr_indptr = np.zeros(n + 1, dtype=np.int32)
            self._csr_indices = np.zeros(0, dtype=np.int32)
            self._elem_pos = []
            self._elem_lidx = []
            self._pattern_stale = False
            return

        if getattr(self, "constraints", None) is not None:
            # Constraint handling assembles in the full space and condenses;
            # skip pattern building to avoid indexing physical DOF ids here.
            self._csr_indptr = np.zeros(1, dtype=np.int32)
            self._csr_indices = np.zeros(0, dtype=np.int32)
            self._elem_pos = [[] for _ in self.kernels_K]
            self._elem_lidx = [[] for _ in self.kernels_K]
            self._pattern_stale = False
            return

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
                full = gdofs[e]
                valid_full = full >= 0
                if not np.any(valid_full):
                    continue
                rmap = self.full_to_red[full[valid_full]]
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
                valid_full = full >= 0
                if not np.any(valid_full):
                    pos_list.append(np.empty(0, dtype=np.int32))
                    lidx_list.append(np.empty(0, dtype=np.int32))
                    continue
                rmap = -np.ones_like(full, dtype=int)
                rmap[valid_full] = self.full_to_red[full[valid_full]]
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
                full = gdofs[e]
                valid_full = full >= 0
                if not np.any(valid_full):
                    continue
                rmap = -np.ones_like(full, dtype=int)
                rmap[valid_full] = self.full_to_red[full[valid_full]]
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
        self._current_bcs = bcs_now
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
        reason = int(self._snes.getConvergedReason())
        if reason <= 0:
            its = int(self._snes.getIterationNumber())
            try:
                fnorm = float(self._snes.getFunctionNorm())
            except Exception:
                fnorm = float("nan")
            print(f"    [warn] SNES did not converge (reason={reason}, iters={its}, ‖F‖={fnorm:.3e}). Continuing with best iterate.")

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
            if getattr(self, "constraints", None) is not None:
                red_idx_list = []
                for gd in full_idx:
                    mcol = self.constraints.master_index_for(int(gd))
                    if mcol is None:
                        continue
                    ridx = self.full_to_red[mcol]
                    if ridx >= 0:
                        red_idx_list.append(int(ridx))
                red_idx = np.array(sorted(set(red_idx_list)), dtype=int)
            else:
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
            self._current_bcs = bcs_now
            n_total_eff = getattr(self.restrictor, "full_size", dh.total_dofs)
            n_phys = getattr(self.restrictor, "phys_size", dh.total_dofs)
            active  = self.active_dofs         # free dofs (length n_free)
            n_free  = active.size

            # 1) pack the CURRENT iterate into a reduced vector x_red
            def _pack_reduced():
                x_full = np.hstack([f.nodal_values for f in funcs])
                if getattr(self, "constraints", None) is not None:
                    x_master = self.constraints.restrict_full(x_full)
                    return x_master[active].copy()
                return x_full[active].copy()

            def _unpack_into_funcs(x_red):
                # expand reduced → full, add to fields, re-apply BCs
                delta_red = x_red - _pack_reduced()  # delta on free dofs
                if getattr(self, "constraints", None) is not None:
                    dU_full = self.restrictor.expand_vec(delta_red)
                else:
                    dU_full = np.zeros(n_phys)
                    dU_full[active] = delta_red
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
        self._current_bcs = bcs_now

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
