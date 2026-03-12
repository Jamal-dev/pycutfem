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
import hashlib
import time
import os
import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Tuple
import inspect
import warnings


import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sp
try:
    from scipy.sparse.linalg import MatrixRankWarning
except Exception:
    MatrixRankWarning = None
# from pycutfem.jit.kernel_args import _scatter_element_contribs
from pycutfem.jit.kernel_args import _build_jit_kernel_args
from pycutfem.jit import compile_multi        
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.helpers import analyze_active_dofs
from pycutfem.solvers.dt_controller import AdaptiveDtController, DtControllerParams
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

_HAVE_NUMBA_REDUCED_PATTERN = False
try:
    from numba import njit  # type: ignore

    _HAVE_NUMBA_REDUCED_PATTERN = True
except Exception:  # pragma: no cover - optional dependency
    njit = None  # type: ignore[assignment]


if _HAVE_NUMBA_REDUCED_PATTERN:
    @njit(cache=True)
    def _csr_find_pos(indices: np.ndarray, start: int, end: int, target: int) -> int:
        lo = start
        hi = end
        while lo < hi:
            mid = (lo + hi) // 2
            v = int(indices[mid])
            if v < target:
                lo = mid + 1
            else:
                hi = mid
        if lo < end and int(indices[lo]) == target:
            return lo
        return -1

    @njit(cache=True)
    def _build_pos_flat_numba(red: np.ndarray, indptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
        n_ent = int(red.shape[0])
        n_loc = int(red.shape[1])
        out = -np.ones((n_ent, n_loc * n_loc), dtype=np.int32)
        for e in range(n_ent):
            for i in range(n_loc):
                ra = int(red[e, i])
                if ra < 0:
                    continue
                s = int(indptr[ra])
                epos = int(indptr[ra + 1])
                base = i * n_loc
                for j in range(n_loc):
                    c = int(red[e, j])
                    if c < 0:
                        continue
                    out[e, base + j] = _csr_find_pos(indices, s, epos, c)
        return out

# ---------------------------------------------------------------------------
# Numba reduced-pattern precompile (hide first-call JIT latency)
# ---------------------------------------------------------------------------
_NUMBA_REDUCED_PATTERN_PRECOMPILE_EVENT: threading.Event | None = None
_NUMBA_REDUCED_PATTERN_PRECOMPILE_ERROR: Exception | None = None
_NUMBA_REDUCED_PATTERN_PRECOMPILE_LOCK = threading.Lock()


def _start_numba_reduced_pattern_precompile() -> None:
    """
    Trigger compilation of the reduced-pattern numba kernels in a background
    thread so the first solver startup does not pay the full JIT cost.
    """
    global _NUMBA_REDUCED_PATTERN_PRECOMPILE_EVENT, _NUMBA_REDUCED_PATTERN_PRECOMPILE_ERROR
    if not _HAVE_NUMBA_REDUCED_PATTERN:
        return
    if os.getenv("PYCUTFEM_NUMBA_PRECOMPILE_REDUCED_PATTERN", "1").lower() in {"0", "false", "no"}:
        return
    with _NUMBA_REDUCED_PATTERN_PRECOMPILE_LOCK:
        if _NUMBA_REDUCED_PATTERN_PRECOMPILE_EVENT is not None:
            return
        ev = threading.Event()
        _NUMBA_REDUCED_PATTERN_PRECOMPILE_EVENT = ev

        def _worker() -> None:
            global _NUMBA_REDUCED_PATTERN_PRECOMPILE_ERROR
            try:
                red = np.zeros((1, 1), dtype=np.int32)
                indptr = np.zeros((2,), dtype=np.int32)
                indices = np.zeros((0,), dtype=np.int32)
                _build_pos_flat_numba(red, indptr, indices)
            except Exception as exc:  # pragma: no cover - best effort
                _NUMBA_REDUCED_PATTERN_PRECOMPILE_ERROR = exc
            finally:
                ev.set()

        t = threading.Thread(
            target=_worker,
            name="pycutfem-numba-precompile-reduced-pattern",
            daemon=True,
        )
        t.start()


def _wait_numba_reduced_pattern_precompile() -> None:
    global _NUMBA_REDUCED_PATTERN_PRECOMPILE_EVENT, _NUMBA_REDUCED_PATTERN_PRECOMPILE_ERROR
    ev = _NUMBA_REDUCED_PATTERN_PRECOMPILE_EVENT
    if ev is None:
        return
    ev.wait()
    if _NUMBA_REDUCED_PATTERN_PRECOMPILE_ERROR is not None:
        # Disable the numba fast-path for this process; fall back to the pure-numpy plan.
        global _HAVE_NUMBA_REDUCED_PATTERN
        _HAVE_NUMBA_REDUCED_PATTERN = False

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
    newton_rtol: float = 0.0           # optional relative tolerance (‖R‖_∞ ≤ rtol·‖R₀‖_∞)
    max_newton_iter: int = 20           # hard cap on inner Newton iterations
    # PETSc SNES robustness: when SNES returns a best iterate but reports a
    # non-converged reason (e.g. line-search divergence), optionally accept the
    # step if the reported ‖F‖ is still "close enough" to the absolute tolerance.
    #
    # - 0 disables this behavior (default; preserves strict rejection)
    # - A value like 2 means accept if ‖F‖ ≤ 2*newton_tol.
    #
    # NOTE: SNES uses its own norm for ‖F‖ (typically an L2 norm of the residual
    # vector), while `newton_tol` is documented as an infinity-norm threshold for
    # the pure-Python Newton path. For SNES we interpret `newton_tol` as the
    # absolute tolerance supplied via `snes_atol`.
    accept_nonconverged_atol_factor: float = 0.0
    # Logging verbosity:
    #   0 = quiet (no per-step/iteration prints)
    #   1 = time-step summary only
    #   2 = Newton iteration summary (default)
    #   3 = line-search accept/reject details
    print_level: int = 2

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
    t0: float = 0.0                     # start time (for restarts)
    step0: int = 0                      # global step offset (for logs/callbacks)
    final_time: Optional[float] = None  # default: max_steps * dt (set in __post_init__)
    allow_dt_reduction: bool = False    # opt-in adaptive dt reduction
    dt_reduction_factor: float = 0.5    # dt <- factor * dt on rejection
    dt_reduction_threshold: float = 5.0 # reject if ||ΔU|| grows by this factor
    dt_min: float = 0.0                 # minimum allowed dt when reducing
    on_dt_change: Optional[Callable[[float], None]] = None
    adjust_max_steps_on_dt_change: bool = True  # keep final_time reachable

    # Iteration-count based dt adaptation (used when allow_dt_reduction=True)
    dt_max: Optional[float] = None
    dt_increase_factor: float = 1.1
    dt_decrease_factor_slow: float = 0.9
    dt_iters_increase_threshold: int = 8
    dt_iters_decrease_threshold: int = 20
    dt_easy_steps_before_increase: int = 2
    dt_slow_steps_before_decrease: int = 1
    dt_reject_on_slow: bool = False

    # Optional: step-failure handler for debugging/recovery.
    #
    # Called when Newton fails or returns non-converged at a step.
    # The callback may mutate coefficients (e.g. penalty Constants) and request a retry.
    #
    # Return values:
    # - False / None: not handled; solver applies its default failure handling (e.g. dt reduction / raise)
    # - True or "retry": retry the same step (keeps the default predictor: current <- prev)
    # - "retry_keep_guess": retry the same step but keep the current iterate (skip predictor reset)
    on_step_failure: Optional[Callable[..., object]] = None

    # Predictor for the initial guess at each new time step.
    #
    # - "prev" (default): U^{n+1,0} <- U^n
    # - "delta":          U^{n+1,0} <- U^n + (dt/dt_prev) * (U^n - U^{n-1})
    #                     implemented via the previous accepted step increment.
    #
    # NOTE: this only affects the *initial guess* for Newton/SNES and does not
    # change the converged solution (it can, however, strongly affect robustness
    # and iteration counts).
    predictor: str = "prev"
    predictor_damping: float = 1.0
    predictor_clip_01: bool = False

    def __post_init__(self) -> None:
        if self.final_time is None:
            self.final_time = float(self.t0) + float(self.max_steps) * float(self.dt)
        if self.dt_max is None:
            self.dt_max = float(self.dt)


@dataclass
class LinearSolverParameters:
    """Sparse linear solver settings (expandable for PETSc/Krylov/etc.)."""

    backend: str = ("petsc" if HAS_PETSC else "scipy")
    tol: float = 1e-12
    maxit: int = 10_000


@dataclass
class VIParameters:
    """
    Parameters for box-constrained variational inequalities (VI) solved via
    PDAS / semismooth Newton.

    We support a PDAS / semismooth Newton strategy for box constraints by
    predicting active sets from a (KKT-style) complementarity indicator.

    For lower/upper bounds we use the standard max-based NCP logic:

      lower-active  if   F(x) - c * (x - lo) > 0
      upper-active  if   F(x) + c * (hi - x) < 0

    where F(x) is the (reduced) residual vector and c>0 is a scaling parameter
    with the units of "stiffness" (force/length) for mechanics-like problems.
    """

    c: float = 1.0
    c_by_field: dict[str, float] = field(default_factory=dict)
    active_tol: float = 0.0
    enter_tol: float | None = None
    leave_tol: float | None = None
    active_set_persistence: int = 0
    project_initial_guess: bool = True
    project_each_iteration: bool = False
    inactive_reg_lambda0: float = 0.0
    inactive_reg_lambda_max: float = 1.0e6
    inactive_reg_growth: float = 5.0
    inactive_reg_decay: float = 0.5
    inactive_reg_min_diag: float = 1.0e-12
    inactive_reg_delta_active_trigger: int = 0
    inactive_reg_stall_alpha: float = 0.25
    inactive_reg_stall_merit_ratio: float = 0.95
    merit_mode: str = "split"
    merit_residual_weight: float = 1.0
    merit_gap_weight: float = 1.0
    filter_max_residual_growth: float = 1.25
    filter_max_gap_growth: float = 1.25
    active_step_delta_active_trigger: int = 0
    active_step_soft_alpha: float = 1.0
    active_step_strong_factor: float = 5.0
    filter_max_delta_active: int = 0
    unconstrained_lm: bool = False
    unconstrained_lm_lambda0: float = 1.0e-4
    unconstrained_lm_lambda_max: float = 1.0e6
    unconstrained_lm_growth: float = 5.0
    unconstrained_lm_decay: float = 0.5
    unconstrained_lm_min_diag: float = 1.0e-12
    unconstrained_lm_accept_ratio: float = 1.0e-3
    unconstrained_lm_good_ratio: float = 0.75
    unconstrained_lm_max_tries: int = 6

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
        constraints: Optional[object] = None,
        use_hanging_constraints: bool = True,
        preproc_cb: Optional[Callable[[List], None]] = None,
        # Optional callback invoked *before every assembly* (Jacobian/residual,
        # including line-search residual evaluations). This is intended for
        # updating dependent coefficients (e.g. projected Lagrange multipliers)
        # so residual evaluations stay consistent with the current iterate.
        preassemble_cb: Optional[Callable[[Dict[str, object]], None]] = None,
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
        self.preassemble_cb = preassemble_cb
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
        _profile_setup = os.getenv("PYCUTFEM_PROFILE_SETUP", "").lower() in {"1", "true", "yes"}
        _t_setup0 = time.perf_counter() if _profile_setup else 0.0
        parallel_compile = os.getenv("PYCUTFEM_PARALLEL_COMPILE", "1").lower() not in {"0", "false", "no"}
        _start_numba_reduced_pattern_precompile()
        _t_compile0 = time.perf_counter()
        if self._is_jit_backend():
            msg = f"[setup] preparing kernels (backend='{self.backend}', quad_order={int(self.quad_order)}, cache-aware)"
            if parallel_compile:
                msg += " (residual in background)"
            print(msg, flush=True)
        # Overlap residual-kernel compilation with active-DOF analysis + reduced-pattern
        # building. This cuts time-to-first-Newton on small problems.
        self._compile_all_kernels(parallel_residual=parallel_compile)
        if self._is_jit_backend():
            dt_compile = time.perf_counter() - _t_compile0
            if parallel_compile and getattr(self, "_kernels_F_future", None) is not None:
                print(f"[setup] kernels ready (jacobian) in {dt_compile:.3f}s; residual will finish during setup.", flush=True)
            else:
                print(f"[setup] kernels ready in {dt_compile:.3f}s", flush=True)
        if _profile_setup:
            dt = time.perf_counter() - _t_setup0
            if getattr(self, "_kernels_F_future", None) is not None:
                print(f"[setup] prepare kernels (jacobian): {dt:.3f}s  (residual preparing in background)")
            else:
                print(f"[setup] prepare kernels: {dt:.3f}s")
        # --- NEW: A PRIORI DOF ANALYSIS ---
        # Analyze the forms once to get the definitive set of active DOFs.
        self.equation = Equation(jacobian_form, residual_form)
        # Which BC list marks the fixed rows? Prefer homogeneous set.
        bcs_for_active = self.bcs_homog if getattr(self, "bcs_homog", None) else self.bcs
        # Optional linear constraints → master space (AgFEM / hanging nodes)
        self.constraints = constraints
        if self.constraints is None and use_hanging_constraints and hasattr(self.dh, "build_hanging_node_constraints"):
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
        print(f"NewtonSolver linear solver backend '{str(getattr(self.lp, 'backend', 'scipy') or 'scipy')}'.")
        # Build once: CSR structure & per-element scatter plan for reduced system
        _t_pat0 = time.perf_counter() if _profile_setup else 0.0
        self._build_reduced_pattern()
        if _profile_setup:
            print(f"[setup] reduced pattern: {time.perf_counter() - _t_pat0:.3f}s")

        # Residual kernels may still be compiling in the background when
        # `PYCUTFEM_PARALLEL_COMPILE=1`. Synchronize before leaving __init__
        # so failures surface early and no background threads linger.
        _t_wait0 = time.perf_counter() if _profile_setup else 0.0
        self._finish_residual_kernel_compilation()
        if _profile_setup:
            dt_wait = time.perf_counter() - _t_wait0
            # Only print if we actually waited a noticeable amount.
            if dt_wait >= 1.0e-3:
                print(f"[setup] compile kernels (residual wait): {dt_wait:.3f}s")
        self.restrictor = _ActiveReducer(self.dh, self.active_dofs, constraint=self.constraints)


    def _is_jit_backend(self) -> bool:
        return self.backend in {"jit", "cpp", "c++"}

    def _gather_full_iterate(self, funcs: list) -> np.ndarray:
        """
        Gather the current iterate into a full global DOF vector (global index order).

        Do not assume `func.nodal_values` is stored in global-DOF order. Always map
        via each function's `_g_dofs`.
        """
        n = int(self.dh.total_dofs)
        x_full = np.zeros(n, dtype=float)
        for f in funcs:
            g = getattr(f, "_g_dofs", None)
            vals = getattr(f, "nodal_values", None)
            if g is None or vals is None:
                continue
            g_arr = np.asarray(g, dtype=int).ravel()
            if g_arr.size == 0:
                continue
            v_arr = np.asarray(vals, dtype=float).ravel()
            if v_arr.size != g_arr.size:
                raise ValueError(
                    f"Function '{getattr(f, 'name', '<unnamed>')}' has nodal_values size {int(v_arr.size)} "
                    f"but _g_dofs size {int(g_arr.size)}."
                )
            x_full[g_arr] = v_arr
        return x_full

    def _enforce_constraints_on_functions(self, functions: list) -> None:
        """
        Ensure the *stored* Function/VectorFunction values satisfy the active
        linear constraints (slaves are overwritten from masters).

        This is required because the solver stores solution vectors in the full
        DOF space even when assembling/solving in the constrained master space.
        """
        if getattr(self, "constraints", None) is None:
            return
        slave_to_master = getattr(self.constraints, "slave_to_master", None)
        if not isinstance(slave_to_master, dict) or not slave_to_master:
            return

        from pycutfem.ufl.expressions import Function, VectorFunction

        ndof = int(self.dh.total_dofs)
        U = np.zeros(ndof, dtype=float)

        # Gather full vector from the provided function objects.
        for func in functions:
            if not isinstance(func, (Function, VectorFunction)):
                continue
            g2l = getattr(func, "_g2l", None)
            vals = getattr(func, "nodal_values", None)
            if g2l is None or vals is None:
                continue
            for gdof, lidx in g2l.items():
                U[int(gdof)] = float(vals[int(lidx)])

        # Overwrite slave DOFs from masters.
        for sdof, combo in slave_to_master.items():
            U[int(sdof)] = sum(float(w) * U[int(mdof)] for (mdof, w) in combo)

        # Scatter back into the function objects.
        for func in functions:
            if not isinstance(func, (Function, VectorFunction)):
                continue
            g2l = getattr(func, "_g2l", None)
            vals = getattr(func, "nodal_values", None)
            if g2l is None or vals is None:
                continue
            for gdof, lidx in g2l.items():
                vals[int(lidx)] = U[int(gdof)]

    def _assert_functions_finite(self, functions: list, *, context: str) -> None:
        from pycutfem.ufl.expressions import Function, VectorFunction

        for func in functions:
            if not isinstance(func, (Function, VectorFunction)):
                continue
            vals = np.asarray(getattr(func, "nodal_values", np.array([], dtype=float)), dtype=float)
            if vals.size and not np.all(np.isfinite(vals)):
                raise RuntimeError(
                    f"Non-finite iterate detected {context} in field '{getattr(func, 'name', '<unnamed>')}'."
                )




    def _compile_all_kernels(self, *, parallel_residual: bool = False) -> None:
        """(Re)compile residual and Jacobian kernels with current metadata."""
        # Ensure no async compilation is left dangling from a previous call.
        self._finish_residual_kernel_compilation()
        if self._is_jit_backend():
            _compress_cache: dict = {}
            _gdofs_cache: dict = {}
            self.kernels_K = compile_multi(
                self._jacobian_form,
                dof_handler=self.dh,
                mixed_element=self.me,
                quad_order=self.quad_order,
                backend=self.backend,
                compress_cache=_compress_cache,
                gdofs_cache=_gdofs_cache,
            )

            if parallel_residual:
                from concurrent.futures import ThreadPoolExecutor

                self.kernels_F = []
                self._kernels_F_executor = ThreadPoolExecutor(max_workers=1)
                self._kernels_F_future = self._kernels_F_executor.submit(
                    compile_multi,
                    self._residual_form,
                    dof_handler=self.dh,
                    mixed_element=self.me,
                    quad_order=self.quad_order,
                    backend=self.backend,
                    compress_cache=_compress_cache,
                    gdofs_cache=_gdofs_cache,
                )
            else:
                self.kernels_F = compile_multi(
                    self._residual_form,
                    dof_handler=self.dh,
                    mixed_element=self.me,
                    quad_order=self.quad_order,
                    backend=self.backend,
                    compress_cache=_compress_cache,
                    gdofs_cache=_gdofs_cache,
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
            raise ValueError(
                f"Unknown backend '{self.backend}'. Use 'python', 'jit', or 'cpp'."
            )

        self._pattern_stale = True
        self._last_jacobian = None
        for attr in ("_csr_indptr", "_csr_indices", "_elem_pos", "_elem_lidx"):
            if hasattr(self, attr):
                delattr(self, attr)

    def _finish_residual_kernel_compilation(self) -> None:
        fut = getattr(self, "_kernels_F_future", None)
        if fut is None:
            return
        ex = getattr(self, "_kernels_F_executor", None)
        try:
            self.kernels_F = fut.result()
        finally:
            try:
                if ex is not None:
                    ex.shutdown(wait=True)
            finally:
                self._kernels_F_future = None
                self._kernels_F_executor = None

    def refresh_levelset_kernels(self, level_set):
        """
        Refresh precomputed static arguments for kernels that depend on a moving
        level set without re-JIT compilation. Marks the scatter pattern as stale
        so it will be rebuilt on the next assembly **only if** the kernel/entity
        layout changed (eids/gdofs_map shape). Pure geometry updates (same entity
        sets, new quadrature/weights) should not force a sparsity rebuild.
        """
        if not self._is_jit_backend():
            return
        refresh_mode = os.getenv("PYCUTFEM_LEVELSET_KERNEL_REFRESH", "").strip().lower()
        if refresh_mode in {"recompile", "compile", "full"}:
            t0 = time.perf_counter()
            self._compile_all_kernels()
            print(
                f"[jit] recompiled kernels for moving level set in {time.perf_counter() - t0:.3f}s"
            )
            return
        debug_refresh = os.getenv("PYCUTFEM_LEVELSET_REFRESH_DEBUG", "").lower() in {"1", "true", "yes"}
        raise_on_fail = os.getenv("PYCUTFEM_LEVELSET_REFRESH_RAISE", "").lower() in {"1", "true", "yes"}
        trace_refresh = os.getenv("PYCUTFEM_LEVELSET_REFRESH_TRACE", "").lower() in {"1", "true", "yes"}
        log_summary = os.getenv("PYCUTFEM_LEVELSET_REFRESH_LOG", "").lower() in {"1", "true", "yes"}
        assert_full_refresh = os.getenv("PYCUTFEM_LEVELSET_REFRESH_ASSERT_FULL", "").lower() in {"1", "true", "yes"}
        validate_refresh = os.getenv("PYCUTFEM_LEVELSET_REFRESH_VALIDATE", "").lower() in {"1", "true", "yes"}
        validate_raise = os.getenv("PYCUTFEM_LEVELSET_REFRESH_VALIDATE_RAISE", "").lower() in {"1", "true", "yes"}
        profile_refresh = os.getenv("PYCUTFEM_LEVELSET_REFRESH_PROFILE", "").lower() in {"1", "true", "yes"}
        try:
            validate_samples = int(os.getenv("PYCUTFEM_LEVELSET_REFRESH_VALIDATE_SAMPLES", "25") or "25")
        except Exception:
            validate_samples = 25
        rebuild_no_reuse = refresh_mode in {"rebuild", "rebuild_static", "no_reuse", "noreuse"}

        kernels_all = list(getattr(self, "kernels_K", [])) + list(getattr(self, "kernels_F", []))
        kernels_ls = [ker for ker in kernels_all if getattr(ker, "level_set", None) is level_set]
        kernels_ls_buildable = [ker for ker in kernels_ls if callable(getattr(ker, "builder", None))]

        if trace_refresh:
            from collections import Counter

            def _kind_of(ker) -> str:
                try:
                    if isinstance(getattr(ker, "static_args", None), dict):
                        k = ker.static_args.get("entity_kind", None)
                        if k:
                            return str(k)
                except Exception:
                    pass
                if str(getattr(ker, "domain", "")).lower() in {"ghost_edge"}:
                    return "edge"
                return "element"

            key_counts = Counter()
            for ker in kernels_ls:
                dom = str(getattr(ker, "domain", "") or "unknown")
                side = str(getattr(ker, "side", "") or "")
                kind = _kind_of(ker)
                key_counts[(dom, side, kind)] += 1

            print(
                "[jit] level-set refresh: total_kernels={} (K={}, F={}), ls_dependent={}, buildable={}".format(
                    int(len(kernels_all)),
                    int(len(getattr(self, "kernels_K", []) or [])),
                    int(len(getattr(self, "kernels_F", []) or [])),
                    int(len(kernels_ls)),
                    int(len(kernels_ls_buildable)),
                )
            )
            if key_counts:
                pretty = ", ".join(
                    f"{dom}{('/'+side) if side else ''}/{kind}:{n}"
                    for (dom, side, kind), n in sorted(key_counts.items())
                )
                print(f"[jit] level-set kernels by domain: {pretty}")

        if kernels_ls and not kernels_ls_buildable:
            msg = "[jit] level-set refresh: found ls-dependent kernels but none are refreshable (missing builder hooks)."
            print(msg)
            if assert_full_refresh:
                raise RuntimeError(msg)

        def _pattern_sig(ker) -> tuple:
            """
            Return a lightweight signature of the kernel layout that impacts
            reduced sparsity/scatter plans. We intentionally ignore quadrature
            data and other geometry-dependent arrays.
            """
            static = getattr(ker, "static_args", None)
            if not isinstance(static, dict):
                return (0, b"", 0, 0)
            eids = static.get("eids", None)
            if eids is None:
                try:
                    eids = getattr(ker, "eids", None)
                except Exception:
                    eids = None
            if eids is None:
                eids = np.asarray([], dtype=np.int64)
            else:
                eids = np.asarray(eids, dtype=np.int64).ravel()
            if eids.size:
                eids_u8 = np.ascontiguousarray(eids).view(np.uint8)
                eids_digest = hashlib.blake2b(eids_u8, digest_size=16).digest()
            else:
                eids_digest = b""

            gdofs = static.get("gdofs_map", None)
            if isinstance(gdofs, np.ndarray):
                gdofs_shape0 = int(gdofs.shape[0])
                gdofs_shape1 = int(gdofs.shape[1]) if gdofs.ndim >= 2 else 0
            else:
                gdofs_shape0, gdofs_shape1 = 0, 0

            return (int(eids.size), eids_digest, gdofs_shape0, gdofs_shape1)

        # Default: refresh static args in-place (fast).
        t0 = time.perf_counter()
        changed = 0
        pattern_changed = False
        any_fail = False
        first_fail: tuple[str, str | None, str | None, Exception] | None = None
        refreshed_kernels: list = []
        prof_rows: list[tuple[float, str, str | None, str | None, int]] = []
        for ker in list(getattr(self, "kernels_K", [])) + list(getattr(self, "kernels_F", [])):
            if getattr(ker, "level_set", None) is not level_set:
                continue
            sig_before = _pattern_sig(ker)
            t_ker0 = time.perf_counter() if profile_refresh else 0.0
            try:
                if rebuild_no_reuse:
                    if getattr(ker, "builder", None) is None:
                        updated = False
                    else:
                        new_args = ker.builder(level_set)
                        if new_args is None:
                            updated = False
                        else:
                            old_args = ker.static_args if isinstance(ker.static_args, dict) else None
                            if old_args is not None and isinstance(new_args, dict):
                                for key, val in old_args.items():
                                    if key not in new_args:
                                        new_args[key] = val
                            ker.static_args = new_args
                            ker.eids = np.asarray(new_args.get("eids", []), dtype=np.int32)
                            updated = True
                else:
                    updated = ker.refresh(level_set)
            except Exception as exc:
                any_fail = True
                if first_fail is None:
                    try:
                        dom = str(getattr(ker, "domain", "unknown"))
                        side = getattr(ker, "side", None)
                        kind = None
                        try:
                            kind = ker.static_args.get("entity_kind", None)
                        except Exception:
                            kind = None
                        first_fail = (dom, side, kind, exc)
                    except Exception:
                        first_fail = ("unknown", None, None, exc)
                if debug_refresh:
                    dom = getattr(ker, "domain", "unknown")
                    side = getattr(ker, "side", None)
                    kind = None
                    try:
                        kind = ker.static_args.get("entity_kind", None)
                    except Exception:
                        kind = None
                    try:
                        eids_len = int(np.asarray(ker.static_args.get("eids", [])).shape[0])
                    except Exception:
                        eids_len = -1
                    try:
                        gdofs_shape = tuple(np.asarray(ker.static_args.get("gdofs_map")).shape)
                    except Exception:
                        gdofs_shape = ()
                    print(f"[jit] refresh failed: dom={dom} side={side} kind={kind} err={exc}")
                    print(f"[jit]   old static: nEnt={eids_len} gdofs_map={gdofs_shape}")
                    try:
                        import traceback as _tb
                        print(_tb.format_exc().rstrip())
                    except Exception:
                        pass
                if raise_on_fail:
                    raise
                updated = False
            if updated:
                changed += 1
                refreshed_kernels.append(ker)
            sig_after = _pattern_sig(ker)
            if sig_after != sig_before:
                pattern_changed = True
            if profile_refresh:
                dt_ker = time.perf_counter() - t_ker0
                dom = str(getattr(ker, "domain", "unknown"))
                side = getattr(ker, "side", None)
                kind = None
                try:
                    if isinstance(getattr(ker, "static_args", None), dict):
                        kind = ker.static_args.get("entity_kind", None)
                except Exception:
                    kind = None
                n_ent = 0
                try:
                    n_ent = int(np.asarray(getattr(ker, "eids", np.asarray([]))).shape[0])
                except Exception:
                    n_ent = 0
                prof_rows.append((float(dt_ker), dom, side, str(kind) if kind is not None else None, int(n_ent)))
        if any_fail and not raise_on_fail:
            # Avoid running with a partially refreshed kernel set (inconsistent
            # Jacobian/residual). Fall back to a full recompile.
            t_re = time.perf_counter()
            self._compile_all_kernels()
            if first_fail is not None:
                dom, side, kind, exc = first_fail
                print(
                    "[jit] kernel refresh failed (dom={}, side={}, kind={}): {}; recompiled kernels in {:.3f}s".format(
                        dom,
                        side,
                        kind,
                        exc,
                        time.perf_counter() - t_re,
                    )
                )
            else:
                print(f"[jit] kernel refresh failed; recompiled kernels in {time.perf_counter() - t_re:.3f}s")
            return
        if kernels_ls_buildable and changed != len(kernels_ls_buildable):
            msg = "[jit] level-set refresh: refreshed {}/{} refreshable kernels (stale kernels possible).".format(
                int(changed), int(len(kernels_ls_buildable))
            )
            print(msg)
            if assert_full_refresh:
                raise RuntimeError(msg)

        if validate_refresh and refreshed_kernels:
            mesh = getattr(getattr(self, "me", None), "mesh", None) or getattr(self.dh.mixed_element, "mesh", None)
            if mesh is not None:
                rng = np.random.default_rng(0)

                def _sample(arr: np.ndarray, k: int) -> np.ndarray:
                    arr = np.asarray(arr, dtype=int)
                    if arr.size <= k:
                        return arr
                    idx = rng.choice(arr.size, size=int(k), replace=False)
                    return arr[idx]

                bad: list[str] = []
                for ker in refreshed_kernels:
                    static = getattr(ker, "static_args", None)
                    if not isinstance(static, dict):
                        continue
                    eids = np.asarray(static.get("eids", []), dtype=int)
                    kind = static.get("entity_kind", None)
                    dom = str(getattr(ker, "domain", "") or "unknown")
                    side = getattr(ker, "side", None)
                    if kind not in {"edge", "element"}:
                        kind = "edge" if dom == "ghost_edge" else "element"

                    gdofs = static.get("gdofs_map", None)
                    if isinstance(gdofs, np.ndarray):
                        try:
                            if int(gdofs.shape[0]) != int(eids.shape[0]):
                                bad.append(
                                    f"gdofs_map rows != eids ({dom}, kind={kind}, side={side}): {gdofs.shape[0]} vs {eids.shape[0]}"
                                )
                        except Exception:
                            pass

                    samp = _sample(eids, max(0, int(validate_samples)))
                    if samp.size == 0:
                        continue
                    if kind == "element":
                        tags = [str(getattr(mesh.elements_list[int(e)], "tag", "")) for e in samp]
                        tagset = set(tags)
                        if dom == "interface":
                            if any(t != "cut" for t in tags):
                                bad.append(f"interface/element has non-cut tags: {sorted(tagset)}")
                        elif dom == "volume":
                            # Full-volume kernels should be pure inside/outside; cut-volume kernels are separate.
                            if static.get("_full_fixed", False):
                                # Fixed-size full kernels may include all element ids, with inactive
                                # elements masked out by zero quadrature weights.
                                qw = static.get("qw", None)
                                if (
                                    isinstance(qw, np.ndarray)
                                    and qw.ndim >= 2
                                    and int(qw.shape[0]) >= int(mesh.n_elements)
                                ):
                                    wrong = []
                                    for eid, t in zip(samp, tags):
                                        if str(side) == "+" and t != "outside":
                                            wrong.append(int(eid))
                                        if str(side) == "-" and t != "inside":
                                            wrong.append(int(eid))
                                    if wrong:
                                        if not np.allclose(qw[np.asarray(wrong, dtype=int)], 0.0):
                                            bad.append(
                                                f"volume/{side} fixed kernel has nonzero weights on non-{('outside' if str(side)== '+' else 'inside')} ids"
                                            )
                            else:
                                if "cut" in tagset and (len(tagset) > 1):
                                    bad.append(f"volume kernel mixes 'cut' with {sorted(tagset)} (side={side})")
                                if "cut" not in tagset:
                                    if str(side) == "+" and any(t != "outside" for t in tags):
                                        bad.append(f"volume/+ has non-outside tags: {sorted(tagset)}")
                                    if str(side) == "-" and any(t != "inside" for t in tags):
                                        bad.append(f"volume/- has non-inside tags: {sorted(tagset)}")
                    else:
                        tags = [str(getattr(mesh.edges_list[int(g)], "tag", "")) for g in samp]
                        tagset = set(tags)
                        if dom == "interface":
                            if any(t != "interface" for t in tags):
                                bad.append(f"interface/edge has non-interface tags: {sorted(tagset)}")
                        elif dom == "ghost_edge":
                            if any((not t.startswith("ghost")) or t == "interface" for t in tags):
                                bad.append(f"ghost_edge has unexpected tags: {sorted(tagset)}")

                if bad:
                    msg = "[jit] level-set refresh validation failed: " + " | ".join(bad[:6])
                    print(msg)
                    if validate_raise:
                        raise RuntimeError(msg)

        if changed:
            if pattern_changed:
                self._pattern_stale = True
                if hasattr(self, "_decoupled_full_mask"):
                    try:
                        delattr(self, "_decoupled_full_mask")
                    except Exception:
                        pass
            total_dt = time.perf_counter() - t0
            if profile_refresh and prof_rows:
                prof_rows.sort(reverse=True, key=lambda t: t[0])
                print("[jit] level-set refresh slow kernels (dt, domain, side, kind, nEnt):")
                for dt_ker, dom, side, kind, n_ent in prof_rows[:12]:
                    print(f"  {dt_ker:7.3f}s  dom={dom}  side={side}  kind={kind}  n={n_ent}")
            if log_summary or trace_refresh or debug_refresh or profile_refresh:
                print(f"[jit] refreshed {changed} kernels for moving level set in {total_dt:.3f}s")
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
        step = 0  # local step counter within this solve call
        t_n = float(getattr(time_params, "t0", 0.0) or 0.0)
        step0 = int(getattr(time_params, "step0", 0) or 0)
        prev_delta_inf: float | None = None
        last_dt: float | None = None
        last_success_delta: np.ndarray | None = None
        last_success_dt: float | None = None

        abort_on_dt_reduction = os.getenv("PYCUTFEM_ABORT_ON_DT_REDUCTION", "").lower() in {"1", "true", "yes"}
        keep_guess = False  # set by on_step_failure="retry_keep_guess"

        dt_controller: AdaptiveDtController | None = None
        if bool(getattr(time_params, "allow_dt_reduction", False)):
            dt_controller = AdaptiveDtController(
                dt0=float(time_params.dt),
                params=DtControllerParams(
                    dt_min=float(getattr(time_params, "dt_min", 0.0)),
                    dt_max=getattr(time_params, "dt_max", None),
                    iters_increase_threshold=int(getattr(time_params, "dt_iters_increase_threshold", 8)),
                    iters_decrease_threshold=int(getattr(time_params, "dt_iters_decrease_threshold", 20)),
                    easy_steps_before_increase=int(getattr(time_params, "dt_easy_steps_before_increase", 2)),
                    slow_steps_before_decrease=int(getattr(time_params, "dt_slow_steps_before_decrease", 1)),
                    increase_factor=float(getattr(time_params, "dt_increase_factor", 1.1)),
                    decrease_factor_slow=float(getattr(time_params, "dt_decrease_factor_slow", 0.9)),
                    decrease_factor_fail=float(getattr(time_params, "dt_reduction_factor", 0.5)),
                    reject_on_slow=bool(getattr(time_params, "dt_reject_on_slow", False)),
                ),
            )

        def _adjust_max_steps_for_final_time(*, t_now: float, step_now: int, dt_new: float) -> None:
            if not bool(getattr(time_params, "adjust_max_steps_on_dt_change", True)):
                return
            final_time = getattr(time_params, "final_time", None)
            try:
                final_time_val = float(final_time) if final_time is not None else None
            except Exception:
                final_time_val = None
            if final_time_val is None or final_time_val <= t_now:
                return
            remaining = float(final_time_val) - float(t_now)
            needed = int(math.ceil(remaining / max(float(dt_new), 1.0e-16)))
            time_params.max_steps = max(int(time_params.max_steps), int(step_now + needed))

        def _require_dt_callback() -> Callable[[float], None]:
            on_dt_change = getattr(time_params, "on_dt_change", None)
            if not callable(on_dt_change):
                raise RuntimeError(
                    "Adaptive dt requires TimeStepperParameters.on_dt_change "
                    "to keep dt-dependent coefficients in sync."
                )
            return on_dt_change

        def _classify_newton_exception(exc: Exception) -> str:
            msg = str(exc)
            if isinstance(exc, RuntimeError) and "Line search failed" in msg:
                return "line_search"
            if "did not converge" in msg:
                return "max_iter"
            return type(exc).__name__

        while step < time_params.max_steps and t_n < time_params.final_time:
            global_step = int(step0 + step)
            global_step_no = int(global_step + 1)
            dt_nominal = float(time_params.dt)
            dt = float(dt_nominal)
            # Cap the last time step to hit final_time exactly.
            # Without this, the time loop can overshoot (e.g. t_n=0.46875, dt=0.125, final_time=0.5).
            try:
                final_time_val = float(time_params.final_time)
            except Exception:
                final_time_val = None
            if final_time_val is not None:
                remaining = float(final_time_val) - float(t_n)
                # Avoid taking a numerically tiny "last" step due to floating-point
                # drift (e.g. t_n=0.49999999999999994 < 0.5). Such steps can make
                # the transient blocks nearly singular (inv_dt -> huge) and break
                # Newton at the end of an otherwise successful run.
                tol_final = 1.0e-12 * max(1.0, abs(float(final_time_val)))
                if remaining <= tol_final:
                    break
                if dt > remaining:
                    dt = float(remaining)
            dt_min = float(getattr(time_params, "dt_min", 0.0))
            if dt_min > 0.0 and dt < dt_min:
                # Allow a shorter *last* step to hit final_time exactly.
                tol_last = 1.0e-12 * max(1.0, abs(float(final_time_val))) if final_time_val is not None else 0.0
                is_last_step = bool(
                    final_time_val is not None
                    and dt < dt_nominal
                    and abs((t_n + dt) - final_time_val) <= tol_last
                )
                if not is_last_step:
                    raise RuntimeError(
                        f"Δt={dt:.3e} dropped below dt_min={dt_min:.3e}; aborting time stepping."
                    )
                if int(getattr(self.np, "print_level", 2) or 0) >= 1:
                    print(
                        f"    [warn] Last step uses Δt={dt:.3e} < dt_min={dt_min:.3e} to hit final_time={final_time_val:.3e}."
                    )
            if last_dt is None or not math.isclose(dt, last_dt, rel_tol=0.0, abs_tol=0.0):
                on_dt_change = getattr(time_params, "on_dt_change", None)
                if callable(on_dt_change):
                    on_dt_change(dt)
                last_dt = dt

            # Predictor: copy previous solution ---------------------
            if not keep_guess:
                for f, f_prev in zip(functions, prev_functions):
                    f.nodal_values[:] = f_prev.nodal_values[:]
                pred_key = str(getattr(time_params, "predictor", "prev")).strip().lower()
                if pred_key in {"delta", "increment", "extrap", "extrapolate"}:
                    if (
                        last_success_delta is not None
                        and last_success_dt is not None
                        and float(last_success_dt) > 0.0
                        and float(getattr(time_params, "predictor_damping", 1.0) or 0.0) != 0.0
                    ):
                        scale = float(dt) / float(last_success_dt)
                        damp = float(getattr(time_params, "predictor_damping", 1.0) or 0.0)
                        dh.add_to_functions(damp * scale * last_success_delta, functions)
                        if bool(getattr(time_params, "predictor_clip_01", False)):
                            for f in functions:
                                fld = getattr(f, "field_name", None)
                                if fld in {"alpha", "phi"}:
                                    try:
                                        f.nodal_values[:] = np.clip(np.asarray(f.nodal_values, dtype=float), 0.0, 1.0)
                                    except Exception:
                                        pass
            else:
                keep_guess = False

            # Time‑dependent BCs -----------------------------------
            # For theta-schemes, apply time-dependent Dirichlet data at t_{n+θ}
            # (implicit Euler θ=1 → end-of-step, CN θ=0.5 → midpoint).
            t_bc = t_n + float(getattr(time_params, "theta", 1.0)) * dt
            bcs_now = self._freeze_bcs(self.bcs, t_bc)
            dh.apply_bcs(bcs_now, *functions)

            # Newton loop -----------------------------------------
            # Expose step metadata for debug hooks (e.g. step-scoped FD checks).
            self._current_step_no = int(global_step_no)
            self._current_t = float(t_n)
            self._current_dt = float(dt)
            # Clear last-step diagnostics (set by backend-specific Newton loops).
            self._last_nonlinear_norm = None
            self._last_nonlinear_norm_label = None
            self._last_nonlinear_reason = None
            self._last_nonlinear_update_inf = None
            self._last_nonlinear_update_label = None
            self._last_nonlinear_accepted = False
            self._last_nonlinear_iterations = None
            try:
                delta_U, converged, n_iters = self._newton_loop(functions, prev_functions, aux_functions, bcs_now)
            except Exception as e:
                print(f"    Newton failed at step {global_step_no} with dt={dt:.3e}: {e}")
                on_fail = getattr(time_params, "on_step_failure", None)
                if callable(on_fail):
                    try:
                        action = on_fail(
                            step=int(global_step),
                            step_no=int(global_step_no),
                            global_step=int(global_step),
                            global_step_no=int(global_step_no),
                            local_step=int(step),
                            local_step_no=int(step + 1),
                            t=float(t_n),
                            dt=float(dt),
                            exception=e,
                            functions=functions,
                            prev_functions=prev_functions,
                            bcs=bcs_now,
                            aux_functions=aux_functions,
                        )
                    except TypeError:
                        action = on_fail(step, t_n, dt, e)
                    except Exception as cb_exc:  # noqa: PERF203
                        print(f"    [warn] on_step_failure callback raised: {cb_exc}")
                        action = None
                    if action in (True, "retry"):
                        continue
                    if action == "retry_keep_guess":
                        keep_guess = True
                        continue
                    if action == "abort":
                        raise
                if not bool(getattr(time_params, "allow_dt_reduction", False)) or dt_controller is None:
                    raise
                _require_dt_callback()
                reason = _classify_newton_exception(e)
                decision = dt_controller.on_failure(dt=dt, reason=reason)
                # Guard against infinite retry loops when dt hits dt_min (or when
                # a controller misconfiguration yields a non-decreasing dt).
                new_dt = float(decision.dt)
                if new_dt >= float(dt) - 0.0:
                    raise RuntimeError(
                        f"Newton failed at step {global_step_no} with dt={dt:.3e}, "
                        f"and Δt cannot be reduced further (proposed dt={new_dt:.3e}, reason={decision.reason})."
                    ) from e
                if dt_min > 0.0 and decision.dt < dt_min:
                    raise RuntimeError(
                        f"Δt reduction would drop below dt_min={dt_min:.3e} (new_dt={decision.dt:.3e})."
                    ) from e
                time_params.dt = new_dt
                _adjust_max_steps_for_final_time(t_now=t_n, step_now=step, dt_new=float(time_params.dt))
                print(
                    f"    Rejecting step {global_step_no}; reducing Δt → {time_params.dt:.3e} ({decision.reason}) and retrying."
                )
                # Reset line-search warm-start between attempts: after a failed step the
                # previous accepted α can be extremely small, which makes retries at a
                # reduced Δt stagnate unnecessarily.
                self._ls_alpha_prev = 1.0
                if abort_on_dt_reduction and float(time_params.dt) < float(dt):
                    raise RuntimeError(
                        f"Aborting after Δt reduction request at step {global_step_no}: "
                        f"{dt:.3e} -> {float(time_params.dt):.3e} ({decision.reason})."
                    ) from e
                continue

            if not converged:
                # Some solver backends (e.g. SNES) may return the best iterate without raising.
                on_fail = getattr(time_params, "on_step_failure", None)
                if callable(on_fail):
                    try:
                        action = on_fail(
                            step=int(global_step),
                            step_no=int(global_step_no),
                            global_step=int(global_step),
                            global_step_no=int(global_step_no),
                            local_step=int(step),
                            local_step_no=int(step + 1),
                            t=float(t_n),
                            dt=float(dt),
                            exception=RuntimeError("Newton did not converge."),
                            functions=functions,
                            prev_functions=prev_functions,
                            bcs=bcs_now,
                            aux_functions=aux_functions,
                        )
                    except TypeError:
                        action = on_fail(step, t_n, dt, None)
                    except Exception as cb_exc:  # noqa: PERF203
                        print(f"    [warn] on_step_failure callback raised: {cb_exc}")
                        action = None
                    if action in (True, "retry"):
                        continue
                    if action == "retry_keep_guess":
                        keep_guess = True
                        continue
                    if action == "abort":
                        raise RuntimeError(
                            f"Newton did not converge at step {global_step_no} with dt={dt:.3e}."
                        )
                if not bool(getattr(time_params, "allow_dt_reduction", False)) or dt_controller is None:
                    raise RuntimeError(
                        f"Newton did not converge at step {global_step_no} with dt={dt:.3e}."
                    )
                _require_dt_callback()
                decision = dt_controller.on_failure(dt=dt, reason="not_converged")
                # Guard against infinite retry loops when dt hits dt_min.
                new_dt = float(decision.dt)
                if new_dt >= float(dt) - 0.0:
                    raise RuntimeError(
                        f"Newton did not converge at step {global_step_no} with dt={dt:.3e}, "
                        f"and Δt cannot be reduced further (proposed dt={new_dt:.3e}, reason={decision.reason})."
                    )
                if dt_min > 0.0 and decision.dt < dt_min:
                    raise RuntimeError(
                        f"Δt reduction would drop below dt_min={dt_min:.3e} (new_dt={decision.dt:.3e})."
                    )
                time_params.dt = new_dt
                _adjust_max_steps_for_final_time(t_now=t_n, step_now=step, dt_new=float(time_params.dt))
                print(
                    f"    Rejecting step {global_step_no}; reducing Δt → {time_params.dt:.3e} ({decision.reason}) and retrying."
                )
                # Same rationale as in the exception path above.
                self._ls_alpha_prev = 1.0
                if abort_on_dt_reduction and float(time_params.dt) < float(dt):
                    raise RuntimeError(
                        f"Aborting after Δt reduction request at step {global_step_no}: "
                        f"{dt:.3e} -> {float(time_params.dt):.3e} ({decision.reason})."
                    )
                continue

            # Optional dt adaptation based on Newton iteration count (success path)
            if bool(getattr(time_params, "allow_dt_reduction", False)) and dt_controller is not None:
                decision = dt_controller.on_success(dt=dt, n_iters=int(n_iters))
                if decision.dt != dt:
                    _require_dt_callback()
                    dt_min = float(getattr(time_params, "dt_min", 0.0))
                    if dt_min > 0.0 and decision.dt < dt_min:
                        raise RuntimeError(
                            f"Δt update would drop below dt_min={dt_min:.3e} (new_dt={decision.dt:.3e})."
                        )
                    time_params.dt = float(decision.dt)
                    # For reject-on-slow mode, retry the same step at the new dt.
                    if decision.retry_step:
                        _adjust_max_steps_for_final_time(t_now=t_n, step_now=step, dt_new=float(time_params.dt))
                        print(
                            f"    Rejecting step {global_step_no}; setting Δt → {time_params.dt:.3e} ({decision.reason}) and retrying."
                        )
                        if abort_on_dt_reduction and float(time_params.dt) < float(dt):
                            raise RuntimeError(
                                f"Aborting after Δt reduction request at step {global_step_no}: "
                                f"{dt:.3e} -> {float(time_params.dt):.3e} ({decision.reason})."
                            )
                        continue
                    # Otherwise, apply on next step (after advancing time).
                    _adjust_max_steps_for_final_time(
                        t_now=float(t_n + dt), step_now=int(step + 1), dt_new=float(time_params.dt)
                    )
                    if abort_on_dt_reduction and float(time_params.dt) < float(dt):
                        raise RuntimeError(
                            f"Aborting after Δt reduction request at step {global_step_no}: "
                            f"{dt:.3e} -> {float(time_params.dt):.3e} ({decision.reason})."
                        )
                    if decision.reason == "slow_newton":
                        print(
                            f"    Slow Newton convergence ({n_iters} iters); setting next Δt → {time_params.dt:.3e}."
                        )
                    elif decision.reason == "fast_newton":
                        print(
                            f"    Fast Newton convergence ({n_iters} iters); setting next Δt → {time_params.dt:.3e}."
                        )

            delta_inf = float(np.linalg.norm(delta_U, ord=np.inf))
            if int(getattr(self.np, "print_level", 2) or 0) >= 1:
                dom_name = None
                try:
                    dom = []
                    for f_prev, f in zip(prev_functions, functions):
                        df = np.asarray(f.nodal_values, dtype=float) - np.asarray(f_prev.nodal_values, dtype=float)
                        dom.append((getattr(f, "name", type(f).__name__), float(np.linalg.norm(df.ravel(), ord=np.inf))))
                    if dom:
                        dom_name, _dom_val = max(dom, key=lambda t: t[1])
                except Exception:
                    dom_name = None

                msg = (
                    f"    Time step {global_step_no}: t={float(t_n + dt):.6e}s, dt={float(dt):.3e}, "
                    f"nNewton={int(n_iters)}, ΔU_step∞ = {delta_inf:.2e}"
                )
                if dom_name:
                    msg += f" (dom={dom_name})"
                norm_val = getattr(self, "_last_nonlinear_norm", None)
                norm_label = getattr(self, "_last_nonlinear_norm_label", None)
                if norm_label and norm_val is not None:
                    try:
                        msg += f", {str(norm_label)} = {float(norm_val):.3e}"
                    except Exception:
                        pass
                upd_val = getattr(self, "_last_nonlinear_update_inf", None)
                upd_label = getattr(self, "_last_nonlinear_update_label", None)
                if upd_label and upd_val is not None:
                    try:
                        msg += f", {str(upd_label)} = {float(upd_val):.3e}"
                    except Exception:
                        pass
                reason = getattr(self, "_last_nonlinear_reason", None)
                if reason is not None:
                    msg += f", reason={reason}"
                if bool(getattr(self, "_last_nonlinear_accepted", False)):
                    msg += ", accepted=True"
                print(msg)
            self._last_nonlinear_iterations = int(n_iters)
            
            # Post-step refiner (VI clip) **before** promotion so prev matches clipped state
            if post_step_refiner is not None:
                post_step_refiner(step, bcs_now, functions, prev_functions)

            # Store last accepted increment for predictor use on the next step.
            #
            # IMPORTANT: Build this in *global DOF order* using `_g_dofs` maps.
            # Concatenating `f.nodal_values` is not guaranteed to match the global
            # DOF numbering (mixed/vector fields, cut/xfem splitting, constraints),
            # and using a mis-ordered delta can catastrophically destabilize the
            # "delta" predictor.
            try:
                last_success_delta = self._gather_full_iterate(functions) - self._gather_full_iterate(prev_functions)
                last_success_dt = float(dt)
            except Exception:
                last_success_delta = None
                last_success_dt = None

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

            # # Accept: promote current → previous ------------------
            # dh.apply_bcs(bcs_now, *functions)
            # for f_prev, f in zip(prev_functions, functions):
            #     f_prev.nodal_values[:] = f.nodal_values[:]
            # Post time-loop callback
            if self.post_timeloop_cb is not None:
                self.post_timeloop_cb(functions)
            t_n += dt
            step += 1

            # Steady‑state test -----------------------------------
            # If enabled, still run the post-step callback and advance the
            # time/step counters before exiting so the final accepted step is
            # consistently recorded/exported by drivers.
            if time_params.stop_on_steady and step > 0 and delta_inf < time_params.steady_tol:
                if int(getattr(self.np, "print_level", 2) or 0) >= 1:
                    try:
                        tol = float(getattr(time_params, "steady_tol", 0.0))
                    except Exception:
                        tol = float("nan")
                    print(
                        f"    Steady-state reached at t={float(t_n):.6e}s (step={int(global_step_no)}): "
                        f"ΔU_step∞={float(delta_inf):.3e} < steady_tol={tol:.3e}; stopping (--stop-on-steady)."
                    )
                break

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
            if int(getattr(self.np, "print_level", 2) or 0) >= 2:
                print(
                    "        [warn] active_dofs == total_dofs — reduced path = full size."
                    " Check that bcs_homog correctly marks Dirichlet nodes."
                )

        self._last_iter_timings = []
        totals = {"assembly": 0.0, "linear_solve": 0.0, "line_search": 0.0}
        temp_t0 = time.perf_counter()
        converged = False
        norm_R0: float | None = None
        norm_hist: list[float] = []
        for it in range(self.np.max_newton_iter):
            if self.pre_cb is not None:
                self.pre_cb(funcs)

            if getattr(self, "constraints", None) is not None:
                # Keep stored full vectors consistent with master DOFs.
                self._enforce_constraints_on_functions(funcs)
                self._enforce_constraints_on_functions(prev_funcs)

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

            # 1a) PRUNE decoupled rows/cols caused by Restriction (nnz==0)
            A_red, R_red, _pruned, extra_asm = self._prune_decoupled_rows_cols(
                current, A_red, R_red, ndof_eff
            )
            assembly_time += extra_asm

            # Log residual with a full-size view (zeros on fixed DOFs) for readability
            R_full = np.zeros(ndof_eff)
            R_full[self.active_dofs] = R_red
            norm_R = np.linalg.norm(R_full, ord=np.inf)
            norm_hist.append(float(norm_R))
            # Expose convergence diagnostics to the outer time-stepper.
            self._last_nonlinear_norm = float(norm_R)
            self._last_nonlinear_norm_label = "|R|_∞"

            # Optional matrix diagnostics (expensive; debug-only).
            if os.getenv("PYCUTFEM_MATRIX_STATS", "").lower() in {"1", "true", "yes"}:
                step_target_raw = os.getenv("PYCUTFEM_MATRIX_STATS_STEP", "").strip()
                it_target_raw = os.getenv("PYCUTFEM_MATRIX_STATS_IT", "").strip()
                try:
                    step_now = int(getattr(self, "_current_step_no", -1))
                except Exception:
                    step_now = -1
                try:
                    want_step = int(step_target_raw) if step_target_raw else None
                except Exception:
                    want_step = None
                try:
                    want_it = int(it_target_raw) if it_target_raw else 1
                except Exception:
                    want_it = 1
                do_stats = (want_step is None or want_step == step_now) and (int(it + 1) == int(want_it))
                if do_stats and A_red is not None:
                    try:
                        A_csr = A_red.tocsr() if hasattr(A_red, "tocsr") else A_red
                        fro_A = float(np.sqrt(A_csr.multiply(A_csr).sum()))
                        A_skew = A_csr - A_csr.T
                        fro_skew = float(np.sqrt(A_skew.multiply(A_skew).sum()))
                        ratio = fro_skew / max(fro_A, 1.0e-300)
                        nnz = int(getattr(A_csr, "nnz", 0))
                        print(f"        [mat] nnz={nnz}  ||A-A^T||_F/||A||_F={ratio:.3e}  ||A||_F={fro_A:.3e}")

                        k_raw = os.getenv("PYCUTFEM_MATRIX_EIGS_K", "").strip()
                        if k_raw:
                            k = int(k_raw)
                        else:
                            k = 0
                        if k > 0:
                            which = os.getenv("PYCUTFEM_MATRIX_EIGS_WHICH", "LM").strip() or "LM"
                            try:
                                evals = spla.eigs(A_csr, k=min(k, max(1, A_csr.shape[0] - 2)), which=which, return_eigenvectors=False)
                                evals = np.asarray(evals, dtype=np.complex128)
                                order = np.argsort(-np.abs(evals))
                                evals = evals[order]
                                e_fmt = ", ".join([f"{z.real:+.3e}{z.imag:+.3e}j" for z in evals[:k]])
                                print(f"        [mat] eigs(k={k}, which='{which}') ≈ [{e_fmt}]")
                            except Exception as exc:  # noqa: PERF203
                                print(f"        [mat] eigs failed: {exc}")
                    except Exception as exc:  # noqa: PERF203
                        print(f"        [mat] stats failed: {exc}")
            if os.getenv("PYCUTFEM_SAVE_MATRIX", "").lower() in {"1", "true", "yes"}:
                step_target_raw = os.getenv("PYCUTFEM_SAVE_MATRIX_STEP", "").strip()
                it_target_raw = os.getenv("PYCUTFEM_SAVE_MATRIX_IT", "").strip()
                outdir = str(os.getenv("PYCUTFEM_SAVE_MATRIX_DIR", "") or "").strip()
                if not outdir:
                    outdir = os.path.join(os.getcwd(), "_pycutfem_matrix_dumps")
                try:
                    step_now = int(getattr(self, "_current_step_no", -1))
                except Exception:
                    step_now = -1
                try:
                    want_step = int(step_target_raw) if step_target_raw else None
                except Exception:
                    want_step = None
                try:
                    want_it = int(it_target_raw) if it_target_raw else 1
                except Exception:
                    want_it = 1
                do_dump = (want_step is None or want_step == step_now) and (int(it + 1) == int(want_it))
                if do_dump:
                    dumped = getattr(self, "_saved_matrix_keys", None)
                    if dumped is None:
                        dumped = set()
                        setattr(self, "_saved_matrix_keys", dumped)
                    key = (step_now, int(it + 1))
                    if key not in dumped and A_red is not None:
                        dumped.add(key)
                        try:
                            os.makedirs(outdir, exist_ok=True)
                            stem = f"step{int(step_now):04d}_it{int(it + 1):02d}"
                            A_csr = A_red.tocsr() if hasattr(A_red, "tocsr") else A_red
                            sp.save_npz(os.path.join(outdir, f"{stem}_A_red.npz"), A_csr)
                            np.save(os.path.join(outdir, f"{stem}_R_red.npy"), np.asarray(R_red, dtype=np.float64))
                            np.save(os.path.join(outdir, f"{stem}_active_dofs.npy"), np.asarray(self.active_dofs, dtype=np.int64))
                            print(f"        [mat-save] saved reduced matrix/vector to {outdir}/{stem}_*")
                        except Exception as exc:
                            print(f"        [mat-save] failed: {exc}")
            if os.getenv("PYCUTFEM_RESIDUAL_TRACE", "").lower() in {"1", "true", "yes"}:
                worst_full = int(np.argmax(np.abs(R_full))) if R_full.size else -1
                worst_val = float(R_full[worst_full]) if worst_full >= 0 else 0.0
                field = None
                try:
                    field = getattr(self.dh, "_dof_to_node_map", {}).get(worst_full, (None, None))[0]
                except Exception:
                    field = None
                fmsg = f", field={field}" if field is not None else ""

                extra = ""
                if os.getenv("PYCUTFEM_RESIDUAL_TRACE_CLASSIFY", "").lower() in {"1", "true", "yes"}:
                    try:
                        constraints = getattr(self, "constraints", None)
                        bc_full = dh.get_dirichlet_data(bcs_now)
                        if constraints is not None:
                            bc_master = constraints.project_dirichlet(bc_full)
                            is_dirichlet = int(worst_full) in bc_master
                            inactive_master = constraints.to_master_set(dh.dof_tags.get("inactive", set()))
                            is_inactive = int(worst_full) in inactive_master
                        else:
                            is_dirichlet = int(worst_full) in bc_full
                            is_inactive = int(worst_full) in set(dh.dof_tags.get("inactive", set()))
                        extra = f", dirichlet={int(bool(is_dirichlet))}, inactive={int(bool(is_inactive))}"
                    except Exception:
                        extra = ""

                coords_msg = ""
                if os.getenv("PYCUTFEM_RESIDUAL_TRACE_COORDS", "").lower() in {"1", "true", "yes"} and field is not None:
                    try:
                        sl = np.asarray(self.dh.get_field_slice(field), dtype=int).ravel()
                        hit = np.nonzero(sl == int(worst_full))[0]
                        if hit.size:
                            coords = np.asarray(self.dh.get_dof_coords(field), dtype=float)[int(hit[0])]
                            if coords.size >= 2:
                                coords_msg = f", x={coords[0]:.3e}, y={coords[1]:.3e}"
                    except Exception:
                        coords_msg = ""

                print(f"        [res] worst_gdof={worst_full} worst_val={worst_val:.3e}{fmsg}{extra}{coords_msg}")
                if os.getenv("PYCUTFEM_RESIDUAL_TRACE_FIELDS", "").lower() in {"1", "true", "yes"}:
                    field_norms = []
                    for fld in getattr(self.dh, "field_names", []):
                        try:
                            sl = self.dh.get_field_slice(fld)
                        except Exception:
                            continue
                        if sl is None or len(sl) == 0:
                            continue
                        field_norms.append(f"{fld}:{np.linalg.norm(R_full[sl], ord=np.inf):.2e}")
                    if field_norms:
                        print("        [res] per-field |R|_∞: " + ", ".join(field_norms))

                # Optional per-kernel/domain breakdown of the residual at the worst DOF.
                # Useful to separate volume/interface/ghost contributions in CutFEM.
                if os.getenv("PYCUTFEM_RESIDUAL_BREAKDOWN", "").lower() in {"1", "true", "yes"}:
                    it_target = os.getenv("PYCUTFEM_RESIDUAL_BREAKDOWN_IT", "").strip()
                    if not it_target or int(it_target) == int(it + 1):
                        try:
                            worst_red = int(self.full_to_red[worst_full]) if worst_full >= 0 else -1
                        except Exception:
                            worst_red = -1
                        if worst_red >= 0:
                            def _residual_contrib_at_red(kernel, red_index: int) -> float:
                                gdofs = kernel.static_args.get("gdofs_map")
                                if not isinstance(gdofs, np.ndarray) or gdofs.ndim != 2 or gdofs.shape[0] == 0:
                                    return 0.0
                                _, Floc, _ = kernel.exec(current)
                                acc = 0.0
                                constraints = getattr(self, "constraints", None)
                                if constraints is None:
                                    ndof_eff = int(self.full_to_red.size)
                                    for e in range(gdofs.shape[0]):
                                        full = np.asarray(gdofs[e], dtype=int)
                                        valid_full = (full >= 0) & (full < ndof_eff)
                                        if not np.any(valid_full):
                                            continue
                                        rmap = -np.ones_like(full, dtype=int)
                                        rmap[valid_full] = self.full_to_red[full[valid_full]]
                                        hit = np.nonzero(rmap == int(red_index))[0]
                                        if hit.size:
                                            acc += float(np.sum(Floc[e][hit]))
                                    return float(acc)

                                full_to_master = getattr(self, "_constr_full_to_master", None)
                                is_slave = getattr(self, "_constr_is_slave", None)
                                slave_map = getattr(self, "_constr_slave_map", None)
                                if (
                                    not isinstance(full_to_master, np.ndarray)
                                    or not isinstance(is_slave, np.ndarray)
                                    or not isinstance(slave_map, dict)
                                ):
                                    return 0.0

                                for e in range(gdofs.shape[0]):
                                    full = np.asarray(gdofs[e], dtype=int)
                                    valid_full = full >= 0
                                    if not np.any(valid_full):
                                        continue
                                    for loc_i, gd in zip(np.nonzero(valid_full)[0].tolist(), full[valid_full].tolist()):
                                        gd_i = int(gd)
                                        if gd_i < 0 or gd_i >= int(full_to_master.size):
                                            continue
                                        val = float(Floc[e][int(loc_i)])
                                        if not np.isfinite(val) or val == 0.0:
                                            continue
                                        if bool(is_slave[gd_i]):
                                            combo = slave_map.get(gd_i)
                                            if combo is None:
                                                continue
                                            mcols, wts = combo
                                            for mcol, wv in zip(mcols.tolist(), wts.tolist()):
                                                red = int(self.full_to_red[int(mcol)])
                                                if red == int(red_index):
                                                    acc += float(wv) * val
                                        else:
                                            mcol = int(full_to_master[gd_i])
                                            if mcol < 0:
                                                continue
                                            red = int(self.full_to_red[mcol])
                                            if red == int(red_index):
                                                acc += val

                                return float(acc)

                            entries = []
                            for k_idx, kerF in enumerate(self.kernels_F):
                                r0 = _residual_contrib_at_red(kerF, worst_red)
                                if r0 == 0.0:
                                    continue
                                entries.append(
                                    (
                                        abs(r0),
                                        r0,
                                        k_idx,
                                        getattr(kerF, "domain", "unknown"),
                                        getattr(kerF, "side", None),
                                        getattr(kerF.static_args, "get", lambda _k, _d=None: _d)("entity_kind", None),
                                        int(getattr(kerF.static_args.get("gdofs_map"), "shape", (0,))[0]),
                                    )
                                )
                            entries.sort(reverse=True, key=lambda t: t[0])
                            print("        [res] per-kernel contrib at worst DOF (|Rk|, Rk, k, domain, side, kind, nEnt):")
                            for item in entries[:12]:
                                _, r0, k_idx, dom_k, side_k, kind_k, n_ent = item
                                print(
                                    f"          |{abs(r0):.3e}|  Rk={r0:+.3e}  k={k_idx:03d}  "
                                    f"dom={dom_k}  side={side_k}  kind={kind_k}  n={n_ent}"
                                )
            if norm_R0 is None:
                norm_R0 = float(norm_R)
            tol_eff = float(getattr(self.np, "newton_tol", 0.0))
            rtol = float(getattr(self.np, "newton_rtol", 0.0))
            if rtol > 0.0 and norm_R0 > 0.0:
                tol_eff = max(tol_eff, rtol * norm_R0)
            t_current = time.perf_counter()
            t_iteration = t_current - temp_t0
            temp_t0 = t_current
            if int(getattr(self.np, "print_level", 2) or 0) >= 2:
                print(f"        Newton {it+1}: |R|_∞ = {norm_R:.2e}, time = {t_iteration}s")
            if os.getenv("PYCUTFEM_NEWTON_TRACE_RES_FIELDS", "").lower() in {"1", "true", "yes"}:
                try:
                    R_full_dbg = self.restrictor.expand_vec(R_red)
                    field_norms = []
                    for fld in getattr(self.dh, "field_names", []):
                        try:
                            sl = self.dh.get_field_slice(fld)
                        except Exception:
                            continue
                        if sl is None or len(sl) == 0:
                            continue
                        sl = np.asarray(sl, dtype=int).ravel()
                        field_norms.append((float(np.linalg.norm(R_full_dbg[sl], ord=np.inf)), str(fld)))
                    field_norms.sort(reverse=True, key=lambda t: t[0])
                    n_show_raw = os.getenv("PYCUTFEM_NEWTON_TRACE_RES_FIELDS_N", "8").strip()
                    try:
                        n_show = max(1, int(n_show_raw))
                    except Exception:
                        n_show = 8
                    items = ", ".join(f"{name}:{val:.2e}" for val, name in field_norms[:n_show])
                    print(f"        [res] {items}")
                except Exception as exc:
                    print(f"        [res] trace failed: {exc}")
            # Stagnation accept (near tolerance) --------------------------------
            #
            # In strongly penalty-dominated regimes (e.g. very large Brinkman drag β and/or
            # inertia terms containing 1/Δt factors), driving the *raw* residual infinity norm
            # below a very small absolute tolerance can become limited by floating-point noise
            # and subtractive cancellation. In such cases Newton can stagnate with vanishing
            # progress while still being effectively "as converged as possible".
            #
            # This guard accepts an iterate when:
            #   (a) the residual is within a modest factor of the requested tolerance, and
            #   (b) residual improvement over a window is negligible.
            #
            # Set the accept factor to 0 to disable.
            accept_factor_raw = os.getenv("PYCUTFEM_NEWTON_STAGNATION_ACCEPT_FACTOR", "").strip()
            if accept_factor_raw:
                try:
                    stag_accept_factor = float(accept_factor_raw)
                except Exception:
                    stag_accept_factor = 0.0
            else:
                try:
                    stag_accept_factor = float(os.getenv("PYCUTFEM_LS_FAIL_ACCEPT_FACTOR", "20.0"))
                except Exception:
                    stag_accept_factor = 0.0

            if stag_accept_factor > 0.0 and tol_eff > 0.0 and norm_R <= stag_accept_factor * tol_eff:
                win_raw = os.getenv("PYCUTFEM_NEWTON_STAGNATION_WINDOW", "8").strip()
                try:
                    win = max(3, int(win_raw))
                except Exception:
                    win = 8

                rel_raw = os.getenv("PYCUTFEM_NEWTON_STAGNATION_REL_DROP", "1e-3").strip()
                abs_raw = os.getenv("PYCUTFEM_NEWTON_STAGNATION_ABS_DROP_TOLFAC", "0.5").strip()
                try:
                    rel_drop_tol = float(rel_raw)
                except Exception:
                    rel_drop_tol = 1.0e-3
                try:
                    abs_drop_tolfac = float(abs_raw)
                except Exception:
                    abs_drop_tolfac = 0.5

                if len(norm_hist) >= win:
                    recent = norm_hist[-win:]
                    r_max = float(max(recent))
                    r_min = float(min(recent))
                    abs_drop = r_max - r_min
                    rel_drop = abs_drop / max(r_max, 1.0e-300)
                    if (rel_drop <= rel_drop_tol) or (abs_drop <= abs_drop_tolfac * tol_eff):
                        if int(getattr(self.np, "print_level", 2) or 0) >= 2:
                            print(
                                "        Newton stagnated near tolerance "
                                f"(|R|_∞={float(norm_R):.2e}, tol={tol_eff:.2e}, "
                                f"window={win}, rel_drop={rel_drop:.2e}, abs_drop={abs_drop:.2e}); accepting iterate."
                            )
                        converged = True
                        delta = np.hstack(
                            [
                                f.nodal_values - f_prev.nodal_values
                                for f, f_prev in zip(funcs, prev_funcs)
                            ]
                        )
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
                        if int(getattr(self.np, "print_level", 2) or 0) >= 2:
                            print(
                                "          timings: assembly={:.3e}s, solve={:.3e}s, line-search={:.3e}s".format(
                                    assembly_time, 0.0, 0.0
                                )
                            )
                        return delta, converged, it + 1
            if norm_R <= tol_eff:
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
                if int(getattr(self.np, "print_level", 2) or 0) >= 2:
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
                if os.getenv("PYCUTFEM_NEWTON_TRACE_WORST", "").lower() in {"1", "true", "yes"}:
                    try:
                        worst_red = int(np.argmax(np.abs(R_red))) if R_red.size else -1
                        worst_full = int(self.active_dofs[worst_red]) if worst_red >= 0 else -1
                        du_w = float(dU_red[worst_red]) if worst_red >= 0 else 0.0
                        r_w = float(R_red[worst_red]) if worst_red >= 0 else 0.0
                        start = int(A_red.indptr[worst_red]) if worst_red >= 0 else 0
                        end = int(A_red.indptr[worst_red + 1]) if worst_red >= 0 else 0
                        row_nnz = int(end - start) if worst_red >= 0 else 0
                        row_abs = float(np.sum(np.abs(A_red.data[start:end]))) if row_nnz else 0.0
                        diag = 0.0
                        if row_nnz:
                            cols = A_red.indices[start:end]
                            hit = np.nonzero(cols == worst_red)[0]
                            if hit.size:
                                diag = float(A_red.data[start + int(hit[0])])
                        col_nnz = 0
                        col_abs = 0.0
                        try:
                            col = A_red.getcol(worst_red) if worst_red >= 0 else None
                            if col is not None:
                                col_nnz = int(getattr(col, "nnz", 0))
                                col_abs = float(np.sum(np.abs(getattr(col, "data", np.asarray([], dtype=float))))) if col_nnz else 0.0
                        except Exception:
                            col_nnz = 0
                            col_abs = 0.0
                        print(
                            f"        [lin] worst_red={worst_red} worst_gdof={worst_full} "
                            f"R={r_w:+.3e} dU={du_w:+.3e} "
                            f"row_nnz={row_nnz} row_abs={row_abs:.3e} diag={diag:+.3e} "
                            f"col_nnz={col_nnz} col_abs={col_abs:.3e}"
                        )
                    except Exception:
                        pass

            dd_enabled = (
                os.getenv("PYCUTFEM_DIRDERIV_CHECK", "").lower() in {"1", "true", "yes"}
                or os.getenv("PYCUTFEM_NEWTON_FD_CHECK", "").lower() in {"1", "true", "yes"}
            )
            dd_step = os.getenv("PYCUTFEM_DIRDERIV_STEP", "").strip() or os.getenv("PYCUTFEM_NEWTON_FD_STEP", "").strip()
            if dd_enabled and dd_step:
                try:
                    dd_step_i = int(dd_step)
                except Exception:
                    dd_step_i = None
                if dd_step_i is not None and int(getattr(self, "_current_step_no", -1)) != dd_step_i:
                    dd_enabled = False
            if dd_enabled:
                # Directional derivative check: (R(u+ε·δ) - R(u))/ε ≈ J·δ = -R(u)
                eps = float(os.getenv("PYCUTFEM_DIRDERIV_EPS", os.getenv("PYCUTFEM_NEWTON_FD_EPS", "1e-6")))
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
                dd_breakdown = os.getenv("PYCUTFEM_DIRDERIV_BREAKDOWN", "").lower() in {"1", "true", "yes"}
                dd_breakdown_tol = float(os.getenv("PYCUTFEM_DIRDERIV_BREAKDOWN_TOL", "0.0") or "0.0")
                dd_breakdown_once = os.getenv("PYCUTFEM_DIRDERIV_BREAKDOWN_ONCE", "").lower() in {"1", "true", "yes"}
                if dd_breakdown and worst_red >= 0:
                    if dd_breakdown_tol > 0.0 and dir_err < dd_breakdown_tol:
                        dd_breakdown = False
                    if dd_breakdown_once and getattr(self, "_dd_breakdown_done", False):
                        dd_breakdown = False

                if dd_breakdown and worst_red >= 0:
                    # Optional per-kernel breakdown of the DD mismatch at the worst DOF.
                    # This runs extra kernel evaluations; enable only when debugging.
                    self._dd_breakdown_done = True
                    def _residual_contrib_at_red(kernel, red_index: int) -> float:
                        gdofs = kernel.static_args.get("gdofs_map")
                        if not isinstance(gdofs, np.ndarray) or gdofs.ndim != 2 or gdofs.shape[0] == 0:
                            return 0.0
                        _, Floc, _ = kernel.exec(current)
                        acc = 0.0
                        constraints = getattr(self, "constraints", None)
                        if constraints is None:
                            ndof_eff = int(self.full_to_red.size)
                            for e in range(gdofs.shape[0]):
                                full = np.asarray(gdofs[e], dtype=int)
                                valid_full = (full >= 0) & (full < ndof_eff)
                                if not np.any(valid_full):
                                    continue
                                rmap = -np.ones_like(full, dtype=int)
                                rmap[valid_full] = self.full_to_red[full[valid_full]]
                                hit = np.nonzero(rmap == int(red_index))[0]
                                if hit.size:
                                    acc += float(np.sum(Floc[e][hit]))
                            return float(acc)

                        full_to_master = getattr(self, "_constr_full_to_master", None)
                        is_slave = getattr(self, "_constr_is_slave", None)
                        slave_map = getattr(self, "_constr_slave_map", None)
                        if (
                            not isinstance(full_to_master, np.ndarray)
                            or not isinstance(is_slave, np.ndarray)
                            or not isinstance(slave_map, dict)
                        ):
                            return 0.0

                        for e in range(gdofs.shape[0]):
                            full = np.asarray(gdofs[e], dtype=int)
                            valid_full = full >= 0
                            if not np.any(valid_full):
                                continue
                            for loc_i, gd in zip(np.nonzero(valid_full)[0].tolist(), full[valid_full].tolist()):
                                gd_i = int(gd)
                                if gd_i < 0 or gd_i >= int(full_to_master.size):
                                    continue
                                val = float(Floc[e][int(loc_i)])
                                if not np.isfinite(val) or val == 0.0:
                                    continue
                                if bool(is_slave[gd_i]):
                                    combo = slave_map.get(gd_i)
                                    if combo is None:
                                        continue
                                    mcols, wts = combo
                                    for mcol, wv in zip(mcols.tolist(), wts.tolist()):
                                        red = int(self.full_to_red[int(mcol)])
                                        if red == int(red_index):
                                            acc += float(wv) * val
                                else:
                                    mcol = int(full_to_master[gd_i])
                                    if mcol < 0:
                                        continue
                                    red = int(self.full_to_red[mcol])
                                    if red == int(red_index):
                                        acc += val

                        return float(acc)

                    base = []
                    for k_idx, kerF in enumerate(self.kernels_F):
                        base.append((k_idx, kerF, _residual_contrib_at_red(kerF, worst_red)))

                    snap2 = [f.nodal_values.copy() for f in funcs]
                    dh.add_to_functions(dU_full, funcs)  # already scaled by eps above
                    dh.apply_bcs(bcs_now, *funcs)
                    pert = []
                    for k_idx, kerF in enumerate(self.kernels_F):
                        pert.append((k_idx, kerF, _residual_contrib_at_red(kerF, worst_red)))
                    for f, buf in zip(funcs, snap2):
                        f.nodal_values[:] = buf

                    entries = []
                    for (k_idx, kerF, r0), (_, _, r1) in zip(base, pert):
                        dd_k = (r1 - r0) / eps + r0
                        if dd_k == 0.0:
                            continue
                        entries.append(
                            (
                                abs(dd_k),
                                dd_k,
                                r0,
                                r1,
                                k_idx,
                                getattr(kerF, "domain", "unknown"),
                                getattr(kerF, "side", None),
                                getattr(kerF.static_args, "get", lambda _k, _d=None: _d)("entity_kind", None),
                                int(getattr(kerF.static_args.get("gdofs_map"), "shape", (0,))[0]),
                            )
                        )
                    entries.sort(reverse=True, key=lambda t: t[0])
                    print("        [dd] per-kernel breakdown (|dd_k|, dd_k, R0_k, R1_k, k, domain, side, kind, nEnt):")
                    for item in entries[:12]:
                        _, dd_k, r0, r1, k_idx, dom_k, side_k, kind_k, n_ent = item
                        print(
                            f"          |{abs(dd_k):.3e}|  dd={dd_k:+.3e}  R0={r0:+.3e}  R1={r1:+.3e}  "
                            f"k={k_idx:03d}  dom={dom_k}  side={side_k}  kind={kind_k}  n={n_ent}"
                        )

            # 3) Optional Armijo backtracking in reduced space (no BC re-application inside)
            if getattr(self.np, "line_search", False):
                t_ls = time.perf_counter()
                try:
                    dU_red = self._line_search_reduced(A_red, R_red, dU_red, funcs, current, bcs_now)
                    line_search_time = time.perf_counter() - t_ls
                except RuntimeError as exc:
                    msg = str(exc)
                    if "Line search failed" in msg:
                        accept_factor = float(os.getenv("PYCUTFEM_LS_FAIL_ACCEPT_FACTOR", "20.0"))
                        tol_accept = float(tol_eff) if "tol_eff" in locals() else float(getattr(self.np, "newton_tol", 0.0))
                        if accept_factor > 0.0 and tol_accept > 0.0 and norm_R <= accept_factor * tol_accept:
                            line_search_time = time.perf_counter() - t_ls
                            if int(getattr(self.np, "print_level", 2) or 0) >= 2:
                                print(
                                    "        Line search failed near convergence "
                                    f"(|R|_∞={norm_R:.2e}, tol={tol_accept:.2e}); accepting iterate."
                                )
                            converged = True
                            delta = np.hstack(
                                [
                                    f.nodal_values - f_prev.nodal_values
                                    for f, f_prev in zip(funcs, prev_funcs)
                                ]
                            )
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
                                    "converged": True,
                                }
                            )
                            self._last_iteration_totals = totals
                            if int(getattr(self.np, "print_level", 2) or 0) >= 2:
                                print(
                                    "          timings: assembly={:.3e}s, solve={:.3e}s, line-search={:.3e}s".format(
                                        assembly_time, linear_time, line_search_time
                                    )
                                )
                            return delta, converged, it + 1

                    # Robustness: if the Deal.II-style infinity-norm line-search fails,
                    # try Armijo (L2 residual) before giving up.
                    fallback = os.getenv("PYCUTFEM_LS_FALLBACK", "armijo").strip().lower()
                    if (
                        fallback in {"1", "true", "yes", "armijo"}
                        and getattr(self.np, "ls_mode", "armijo") == "dealii"
                    ):
                        if int(getattr(self.np, "print_level", 2) or 0) >= 2:
                            print("        Line search failed in 'dealii' mode; retrying with Armijo.")
                        self.np.ls_mode = "armijo"
                        # Reset the warm-start alpha: the Deal.II backtracking may have
                        # driven `_ls_alpha_prev` to ~0, which makes Armijo fallback fail
                        # trivially (no measurable residual decrease).
                        self._ls_alpha_prev = 1.0
                        try:
                            dU_red = self._line_search_reduced(A_red, R_red, dU_red, funcs, current, bcs_now)
                        except Exception as exc_fallback:  # noqa: PERF203
                            line_search_time = time.perf_counter() - t_ls
                            if int(getattr(self.np, "print_level", 2) or 0) >= 2:
                                print(f"        [ls] Armijo fallback also failed: {exc_fallback}")
                            raise
                        line_search_time = time.perf_counter() - t_ls
                    else:
                        line_search_time = time.perf_counter() - t_ls
                        raise
            else:
                line_search_time = 0.0

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
            if getattr(self, "constraints", None) is not None:
                self._enforce_constraints_on_functions(funcs)

            if self.post_cb is not None:
                self.post_cb(funcs)

            # Reuse the residual computed during line search to detect convergence
            # without a second full assembly on the updated iterate.
            norm_after: float | None = None
            use_ls_cache = (
                bool(getattr(self.np, "line_search", False))
                and getattr(self, "preassemble_cb", None) is None
                and self.post_cb is None
            )
            if use_ls_cache:
                R_after = getattr(self, "_ls_last_residual_red", None)
                if isinstance(R_after, np.ndarray):
                    norm_after = float(np.linalg.norm(R_after, ord=np.inf))
                    if not np.isfinite(norm_after):
                        norm_after = None
            converged_after = bool(norm_after is not None and float(norm_after) <= float(tol_eff))

            totals["assembly"] += assembly_time
            totals["linear_solve"] += linear_time
            totals["line_search"] += line_search_time
            self._last_iter_timings.append(
                {
                    "iteration": it + 1,
                    "assembly": assembly_time,
                    "linear_solve": linear_time,
                    "line_search": line_search_time,
                    "residual_norm": float(norm_after) if converged_after else norm_R,
                    "converged": bool(converged_after),
                }
            )
            if int(getattr(self.np, "print_level", 2) or 0) >= 2:
                print(
                    "          timings: assembly={:.3e}s, solve={:.3e}s, line-search={:.3e}s".format(
                        assembly_time, linear_time, line_search_time
                    )
                )

            if converged_after:
                converged = True
                delta = np.hstack(
                    [f.nodal_values - f_prev.nodal_values for f, f_prev in zip(funcs, prev_funcs)]
                )
                self._last_iteration_totals = totals
                return delta, converged, it + 1

        # If we get here, Newton did not converge within the iteration budget
        accept_factor_raw = os.getenv("PYCUTFEM_NEWTON_MAXITER_ACCEPT_FACTOR", "").strip()
        if accept_factor_raw:
            try:
                accept_factor = float(accept_factor_raw)
            except Exception:
                accept_factor = 0.0
        else:
            # Default: mirror the line-search "near convergence" accept factor so stiff, penalty-dominated
            # problems can make progress even when the raw residual hits a floating-point floor.
            try:
                accept_factor = float(os.getenv("PYCUTFEM_LS_FAIL_ACCEPT_FACTOR", "20.0"))
            except Exception:
                accept_factor = 0.0
        if accept_factor > 0.0:
            tol_eff_last = float(tol_eff) if "tol_eff" in locals() else float(getattr(self.np, "newton_tol", 0.0))
            if tol_eff_last > 0.0 and float(norm_R) <= accept_factor * tol_eff_last:
                if int(getattr(self.np, "print_level", 2) or 0) >= 2:
                    print(
                        "        Newton hit max iterations but is near tolerance "
                        f"(|R|_∞={float(norm_R):.2e}, tol={tol_eff_last:.2e}, factor={accept_factor:.2g}); accepting iterate."
                    )
                converged = True
                delta = np.hstack(
                    [
                        f.nodal_values - f_prev.nodal_values
                        for f, f_prev in zip(funcs, prev_funcs)
                    ]
                )
                self._last_iteration_totals = totals
                return delta, converged, int(self.np.max_newton_iter)

        self._last_iteration_totals = totals
        raise RuntimeError("Newton did not converge – adjust Δt or verify Jacobian.")


   
    def _prune_decoupled_rows_cols(self, coeffs, A_red, R_red, ndof_eff: int):
        """
        Drop decoupled (zero) rows/cols in the reduced system and rebuild the
        reduced pattern. Returns (A_red, R_red, pruned, extra_asm_time).
        """
        pruned = False
        extra_asm = 0.0
        trace = os.getenv("PYCUTFEM_PRUNE_TRACE", "").lower() in {"1", "true", "yes"}
        for _ in range(2):
            row_nnz = np.diff(A_red.indptr)
            if A_red.indices.size:
                col_nnz = np.bincount(A_red.indices, minlength=A_red.shape[1])
            else:
                col_nnz = np.zeros(A_red.shape[1], dtype=int)
            inactive_mask = (row_nnz == 0) | (col_nnz == 0)
            drop_tol = float(os.getenv("PYCUTFEM_DROP_TOL", "1e-12"))
            if drop_tol > 0.0 and A_red.data.size:
                abs_data = np.abs(A_red.data)
                row_abs = np.zeros_like(row_nnz, dtype=abs_data.dtype)
                starts = A_red.indptr[:-1]
                ends = A_red.indptr[1:]
                nonempty = ends > starts
                if np.any(nonempty):
                    row_abs[nonempty] = np.add.reduceat(abs_data, starts[nonempty])
                col_abs = np.bincount(A_red.indices, weights=abs_data, minlength=A_red.shape[1])
                # Treat PYCUTFEM_DROP_TOL as an *absolute* row/col L1 threshold.
                # A global relative threshold (scaled by max|A|) is brittle: a single
                # huge penalty block can cause legitimate physics rows to be dropped.
                inactive_mask |= (row_abs <= drop_tol) | (col_abs <= drop_tol)
            inactive = np.where(inactive_mask)[0]
            if not inactive.size:
                break
            if trace:
                print(f"        [prune] dropping {int(inactive.size)}/{int(A_red.shape[0])} decoupled rows/cols")
            # Track structurally decoupled full-space DOFs so external active-DOF
            # recomputation (e.g. on moving interfaces) can avoid re-introducing
            # them when the entity layout did not change.
            try:
                removed_full = np.asarray(self.active_dofs[inactive], dtype=int)
                if removed_full.size:
                    mask = getattr(self, "_decoupled_full_mask", None)
                    if (
                        not isinstance(mask, np.ndarray)
                        or mask.dtype != bool
                        or int(mask.size) != int(ndof_eff)
                    ):
                        mask = np.zeros(int(ndof_eff), dtype=bool)
                        self._decoupled_full_mask = mask
                    mask[removed_full] = True
            except Exception:
                pass
            keep_mask = np.ones(len(self.active_dofs), dtype=bool)
            keep_mask[inactive] = False
            self.active_dofs = self.active_dofs[keep_mask]
            self.full_to_red = -np.ones(ndof_eff, dtype=int)
            self.full_to_red[self.active_dofs] = np.arange(self.active_dofs.size, dtype=int)
            self.red_to_full = self.active_dofs
            self._pattern_stale = True
            self.restrictor = _ActiveReducer(
                self.dh, self.active_dofs, constraint=getattr(self, "constraints", None)
            )
            self._build_reduced_pattern()
            t_asm = time.perf_counter()
            A_red, R_red = self._assemble_system_reduced(coeffs, need_matrix=True)
            extra_asm += time.perf_counter() - t_asm
            pruned = True
        return A_red, R_red, pruned, extra_asm

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
        # Cache the reduced residual at the *accepted* line-search iterate so the
        # Newton loop can reuse it (avoid an extra assembly just to detect convergence).
        self._ls_last_residual_red = None

        if mode not in {"armijo", "dealii"}:
            raise ValueError(f"Unknown line-search mode '{mode}'.")

        # Deal.II style: accept the first step that reduces the infinity norm of the residual.
        if mode == "dealii":
            trace = os.getenv("PYCUTFEM_LS_TRACE", "").lower() in {"1", "true", "yes"}
            norm0 = float(np.linalg.norm(R_red, ord=np.inf))
            best_alpha, best_norm = 0.0, norm0
            best_R = None
            snap = [f.nodal_values.copy() for f in funcs]

            def _eval(alpha_try: float):
                for f, buf in zip(funcs, snap):
                    f.nodal_values[:] = buf
                dU_full = self.restrictor.expand_vec(alpha_try * S_red)
                dh.add_to_functions(dU_full, funcs)
                dh.apply_bcs(bcs_now, *funcs)
                if getattr(self, "constraints", None) is not None:
                    self._enforce_constraints_on_functions(funcs)

                _, R_try = self._assemble_system_reduced(coeffs, need_matrix=False)
                norm_try = float(np.linalg.norm(R_try, ord=np.inf))
                if trace:
                    print(f"        [ls] α={alpha_try:.6e}  ‖R‖∞={norm_try:.6e}  (‖R‖∞0={norm0:.6e})")
                return norm_try, R_try

            # Always try the full Newton step first.
            # Warm-starting from a tiny α (from a previous near-stagnation) can otherwise
            # "lock" the search into microscopic steps and stall progress on the next
            # Newton iteration / time step.
            norm_try, R_try = _eval(1.0)
            if norm_try < best_norm:
                best_norm, best_alpha, best_R = norm_try, 1.0, R_try
            if norm_try < norm0:
                for f, buf in zip(funcs, snap):
                    f.nodal_values[:] = buf
                if int(getattr(self.np, "print_level", 2) or 0) >= 3:
                    print("        Line search accepted α = 1.00e+00 (‖R‖∞ decreased)")
                self._ls_last_residual_red = np.asarray(R_try, dtype=float)
                self._ls_alpha_prev = 1.0
                return 1.0 * S_red

            # Backtracking sequence after the full-step trial.
            #
            # IMPORTANT: do *not* warm-start from the previously accepted α here.
            # After a near-stagnation, the last accepted α can be microscopic.
            # Starting from such a value would prevent trying moderate steps
            # (e.g. 0.5, 0.25, ...) even when they would reduce ‖R‖∞.
            alpha = float(np_.ls_reduction)
            if not (0.0 < alpha < 1.0):
                alpha = 0.5

            for _ in range(np_.ls_max_iter):
                norm_try, R_try = _eval(alpha)
                if norm_try < best_norm:
                    best_norm, best_alpha, best_R = norm_try, alpha, R_try
                if norm_try < norm0:
                    for f, buf in zip(funcs, snap):
                        f.nodal_values[:] = buf
                    if int(getattr(self.np, "print_level", 2) or 0) >= 3:
                        print(f"        Line search accepted α = {alpha:.2e} (‖R‖∞ decreased)")
                    self._ls_last_residual_red = np.asarray(R_try, dtype=float)
                    self._ls_alpha_prev = alpha
                    return alpha * S_red

                alpha *= np_.ls_reduction

            for f, buf in zip(funcs, snap):
                f.nodal_values[:] = buf
            if best_alpha > 0.0:
                if int(getattr(self.np, "print_level", 2) or 0) >= 2:
                    print(f"        Line search failed, using best-effort α = {best_alpha:.2e} (‖R‖∞ decreased).")
                if best_R is not None:
                    self._ls_last_residual_red = np.asarray(best_R, dtype=float)
                self._ls_alpha_prev = best_alpha
                return best_alpha * S_red
            min_alpha = alpha
            if int(getattr(self.np, "print_level", 2) or 0) >= 2:
                print(f"        Line search failed – taking minimal step α = {min_alpha:.2e}.")
            if os.getenv("PYCUTFEM_LS_FAIL_HARD", "1").lower() not in {"0", "false", "no"}:
                raise RuntimeError("Line search failed: no residual decrease.")
            self._ls_alpha_prev = min_alpha
            return min_alpha * S_red

        # Gradient g_f = J_ff^T R_f  (reduced variables only)
        g = A_red.T @ R_red
        gTS = float(g @ S_red)
        if gTS >= 0.0:
            # Newton direction is not a descent; fall back to steepest descent.
            # This keeps the iteration moving instead of taking a zero step.
            if int(getattr(self.np, "print_level", 2) or 0) >= 2:
                print("        Warning: Not a descent direction in reduced space; using steepest descent.")
            S_red = -g
            gTS = float(g @ S_red)  # = -||g||^2 <= 0
            if os.getenv("PYCUTFEM_LS_DEBUG", "").lower() in {"1", "true", "yes"}:
                try:
                    r_inf = float(np.linalg.norm(R_red, ord=np.inf))
                    g_inf = float(np.linalg.norm(g, ord=np.inf))
                    s_inf = float(np.linalg.norm(S_red, ord=np.inf))
                    print(f"        [ls] ‖R‖∞={r_inf:.3e}  ‖g‖∞={g_inf:.3e}  ‖S‖∞={s_inf:.3e}")
                except Exception:
                    pass

        phi0 = 0.5 * float(R_red @ R_red)
        # Warm-start: reuse last accepted α (try slightly larger first).
        alpha = float(getattr(self, "_ls_alpha_prev", 1.0))
        alpha = min(1.0, alpha / float(np_.ls_reduction))
        best_alpha, best_phi = 0.0, phi0
        best_R = None

        # Snapshot the *full* iterate
        snap = [f.nodal_values.copy() for f in funcs]

        for _ in range(np_.ls_max_iter):
            # Trial update: expand to full, apply, and re-impose BCs
            for f, buf in zip(funcs, snap): f.nodal_values[:] = buf
            dU_full = self.restrictor.expand_vec(alpha * S_red)
            dh.add_to_functions(dU_full, funcs)
            dh.apply_bcs(bcs_now, *funcs)
            if getattr(self, "constraints", None) is not None:
                self._enforce_constraints_on_functions(funcs)

            # Evaluate residual in reduced space (matrix not needed)
            _, R_try = self._assemble_system_reduced(coeffs, need_matrix=False)
            phi = 0.5 * float(R_try @ R_try)

            # Track best effort
            if phi < best_phi:
                best_phi, best_alpha, best_R = phi, alpha, R_try

            # Armijo condition with reduced quantities
            if phi <= phi0 + c1 * alpha * gTS:
                self._ls_last_residual_red = np.asarray(R_try, dtype=float)
                for f, buf in zip(funcs, snap): f.nodal_values[:] = buf
                if int(getattr(self.np, "print_level", 2) or 0) >= 3:
                    print(f"        Armijo search accepted α = {alpha:.2e}")
                self._ls_alpha_prev = alpha
                return alpha * S_red

            alpha *= np_.ls_reduction

        # Fallback (best effort seen)
        for f, buf in zip(funcs, snap): f.nodal_values[:] = buf
        if best_alpha > 0.0:
            if int(getattr(self.np, "print_level", 2) or 0) >= 2:
                print(f"        Armijo failed, using best-effort α = {best_alpha:.2e}.")
            if best_R is not None:
                self._ls_last_residual_red = np.asarray(best_R, dtype=float)
            self._ls_alpha_prev = best_alpha
            return best_alpha * S_red

        # As a last resort, try a simple residual-direction backtracking step.
        # This is useful when the assembled Jacobian is inconsistent enough that
        # the Gauss-Newton gradient g = JᵀR does not produce an actual decrease.
        try_alt = os.getenv("PYCUTFEM_LS_ALT_RESIDUAL", "1").lower() in {"1", "true", "yes"}
        if try_alt:
            S_alt = -np.asarray(R_red, dtype=float)
            alt_norm = float(np.linalg.norm(S_alt, ord=np.inf))
            if np.isfinite(alt_norm) and alt_norm > 0.0:
                alpha_alt = float(getattr(self, "_ls_alpha_prev", 1.0))
                alpha_alt = min(1.0, alpha_alt / float(np_.ls_reduction))
                best_alpha_alt = 0.0
                best_phi_alt = phi0
                best_R_alt = None
                for _ in range(np_.ls_max_iter):
                    for f, buf in zip(funcs, snap):
                        f.nodal_values[:] = buf
                    dU_full = self.restrictor.expand_vec(alpha_alt * S_alt)
                    dh.add_to_functions(dU_full, funcs)
                    dh.apply_bcs(bcs_now, *funcs)
                    if getattr(self, "constraints", None) is not None:
                        self._enforce_constraints_on_functions(funcs)
                    _, R_try = self._assemble_system_reduced(coeffs, need_matrix=False)
                    phi = 0.5 * float(R_try @ R_try)
                    if phi < best_phi_alt:
                        best_phi_alt, best_alpha_alt, best_R_alt = phi, alpha_alt, R_try
                    if phi < phi0:
                        for f, buf in zip(funcs, snap):
                            f.nodal_values[:] = buf
                        if int(getattr(self.np, "print_level", 2) or 0) >= 3:
                            print(f"        Armijo alt (S=-R) accepted α = {alpha_alt:.2e}")
                        self._ls_last_residual_red = np.asarray(R_try, dtype=float)
                        self._ls_alpha_prev = alpha_alt
                        return alpha_alt * S_alt
                    alpha_alt *= np_.ls_reduction
                if best_alpha_alt > 0.0:
                    for f, buf in zip(funcs, snap):
                        f.nodal_values[:] = buf
                    if int(getattr(self.np, "print_level", 2) or 0) >= 2:
                        print(f"        Armijo alt (S=-R) best-effort α = {best_alpha_alt:.2e}.")
                    if best_R_alt is not None:
                        self._ls_last_residual_red = np.asarray(best_R_alt, dtype=float)
                    self._ls_alpha_prev = best_alpha_alt
                    return best_alpha_alt * S_alt
        # No decrease found; still take the smallest tried step to avoid stagnation.
        min_alpha = alpha
        if int(getattr(self.np, "print_level", 2) or 0) >= 2:
            print(f"        Line search failed – taking minimal step α = {min_alpha:.2e}.")
        if os.getenv("PYCUTFEM_LS_FAIL_HARD", "1").lower() not in {"0", "false", "no"}:
            raise RuntimeError("Line search failed: no residual decrease.")
        self._ls_alpha_prev = min_alpha
        return min_alpha * S_red

    def _assemble_residual_reduced_fast(self, coeffs) -> np.ndarray:
        """Assemble reduced residual only, using precomputed reduced scatter maps when available."""
        if getattr(self, "_pattern_stale", True):
            self._build_reduced_pattern()
        self._finish_residual_kernel_compilation()

        nred = int(len(self.active_dofs))
        R_red = np.zeros(nred, dtype=float)
        res_maps = getattr(self, "_res_red_maps", None)
        has_constraints = getattr(self, "constraints", None) is not None

        for k_idx, ker in enumerate(self.kernels_F):
            gdofs = ker.static_args.get("gdofs_map")
            if isinstance(gdofs, np.ndarray) and gdofs.shape[0] == 0:
                continue
            _, Floc, _ = ker.exec(coeffs)  # [nEnt, nLoc]

            red_map = None
            if (
                not has_constraints
                and isinstance(res_maps, list)
                and k_idx < len(res_maps)
                and isinstance(res_maps[k_idx], np.ndarray)
            ):
                cand = res_maps[k_idx]
                if getattr(cand, "shape", None) == getattr(Floc, "shape", None):
                    red_map = cand

            if red_map is not None:
                rm = np.asarray(red_map, dtype=np.int32).ravel()
                ff = np.asarray(Floc, dtype=float).ravel()
                mask = rm >= 0
                if np.any(mask):
                    np.add.at(R_red, rm[mask], ff[mask])
                continue

            # Fallback: per-entity scatter (supports constraint-aware mappings).
            if not isinstance(gdofs, np.ndarray):
                gdofs = ker.static_args["gdofs_map"]
            for e in range(int(gdofs.shape[0])):
                full = gdofs[e]
                valid_full = full >= 0
                if not np.any(valid_full):
                    continue
                if not has_constraints:
                    rmap = -np.ones_like(full, dtype=int)
                    rmap[valid_full] = self.full_to_red[full[valid_full]]
                    m = rmap >= 0
                    if np.any(m):
                        np.add.at(R_red, rmap[m], Floc[e][m])
                else:
                    full_to_master = getattr(self, "_constr_full_to_master", None)
                    is_slave = getattr(self, "_constr_is_slave", None)
                    slave_map = getattr(self, "_constr_slave_map", None)
                    if (
                        not isinstance(full_to_master, np.ndarray)
                        or not isinstance(is_slave, np.ndarray)
                        or not isinstance(slave_map, dict)
                    ):
                        continue
                    for loc_i, gd in zip(np.nonzero(valid_full)[0].tolist(), full[valid_full].tolist()):
                        gd_i = int(gd)
                        if gd_i < 0 or gd_i >= int(full_to_master.size):
                            continue
                        val = float(Floc[e][int(loc_i)])
                        if not np.isfinite(val) or val == 0.0:
                            continue
                        if bool(is_slave[gd_i]):
                            combo = slave_map.get(gd_i)
                            if combo is None:
                                continue
                            mcols, wts = combo
                            for mcol, wv in zip(mcols.tolist(), wts.tolist()):
                                red = int(self.full_to_red[int(mcol)])
                                if red >= 0:
                                    R_red[red] += float(wv) * val
                        else:
                            mcol = int(full_to_master[gd_i])
                            if mcol < 0:
                                continue
                            red = int(self.full_to_red[mcol])
                            if red >= 0:
                                R_red[red] += val

        return R_red

    # ------------------------------------------------------------------
    #  Reduced Linear system & BC handling
    # ------------------------------------------------------------------
    def _assemble_system_reduced(self, coeffs, *, need_matrix: bool = True):
        if getattr(self, "preassemble_cb", None) is not None:
            self.preassemble_cb(coeffs)
        if self.backend == "python":
            return self._assemble_system_reduced_python(coeffs, need_matrix=need_matrix)
        self._maybe_refresh_deformation(coeffs)
        nred = len(self.active_dofs)
        if need_matrix:
            A_red, R_red = self._assemble_system_reduced_fast(coeffs)
            return A_red, R_red
        else:
            return None, self._assemble_residual_reduced_fast(coeffs)

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
        trace_kernel_residuals = os.getenv("PYCUTFEM_TRACE_KERNEL_RESIDUALS", "").lower() in {"1", "true", "yes"}
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
            if trace_kernel_residuals:
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
            try:
                with warnings.catch_warnings():
                    if MatrixRankWarning is not None:
                        warnings.filterwarnings("error", category=MatrixRankWarning)
                    sol = spla.spsolve(A, rhs)
                if not np.isfinite(sol).all():
                    raise np.linalg.LinAlgError("spsolve returned non-finite values")
                return sol
            except Exception as exc:
                if not np.isfinite(A.data).all() or not np.isfinite(rhs).all():
                    raise
                shift_scale = float(os.getenv("PYCUTFEM_LIN_SHIFT", "1e-8"))
                diag = A.diagonal()
                diag_scale = float(np.max(np.abs(diag))) if diag.size else 1.0
                shift = max(shift_scale * max(1.0, diag_scale), 1.0e-14)
                A_reg = A + (shift * sp.eye(A.shape[0], format="csr"))
                print(f"        [lin] adding diagonal shift {shift:.3e} after solver failure: {exc}")
                return spla.spsolve(A_reg, rhs)
        if backend == "petsc":
            if not HAS_PETSC:
                raise RuntimeError(
                    "LinearSolverParameters.backend='petsc' requested but petsc4py is not available."
                )
            return self._solve_linear_system_petsc(A, rhs)
        raise ValueError(f"Unknown linear solver backend '{self.lp.backend}'.")

    def set_linear_schur_fieldsplit(
        self,
        *,
        pressure_field: str = "p",
        schur_fact: str = "full",
        schur_pre: str = "selfp",
        outer_ksp: str | None = None,
        outer_pc: str | None = None,
        rest_ksp: str | None = None,
        rest_pc: str | None = None,
        rest_factor_solver_type: str | None = None,
        pressure_ksp: str | None = None,
        pressure_pc: str | None = None,
        pressure_factor_solver_type: str | None = None,
    ) -> None:
        self._linear_schur_cfg = {
            "pressure_field": str(pressure_field),
            "schur_fact": str(schur_fact).lower(),
            "schur_pre": str(schur_pre).lower(),
            "outer_ksp": None if outer_ksp is None else str(outer_ksp).lower(),
            "outer_pc": None if outer_pc is None else str(outer_pc).lower(),
            "rest_ksp": None if rest_ksp is None else str(rest_ksp).lower(),
            "rest_pc": None if rest_pc is None else str(rest_pc).lower(),
            "rest_factor_solver_type": None if rest_factor_solver_type is None else str(rest_factor_solver_type).lower(),
            "pressure_ksp": None if pressure_ksp is None else str(pressure_ksp).lower(),
            "pressure_pc": None if pressure_pc is None else str(pressure_pc).lower(),
            "pressure_factor_solver_type": None if pressure_factor_solver_type is None else str(pressure_factor_solver_type).lower(),
        }
        try:
            self._petsc_linear_cache = None
        except Exception:
            pass

    def _petsc_linear_schur_cfg_from_env(self) -> dict | None:
        flag = str(os.getenv("PYCUTFEM_LINEAR_SCHUR", "") or "").strip().lower()
        if flag not in {"1", "true", "yes"}:
            return None
        return {
            "pressure_field": str(os.getenv("PYCUTFEM_LINEAR_SCHUR_PRESSURE_FIELD", "p") or "p").strip(),
            "schur_fact": str(os.getenv("PYCUTFEM_LINEAR_SCHUR_FACT_TYPE", "full") or "full").strip().lower(),
            "schur_pre": str(os.getenv("PYCUTFEM_LINEAR_SCHUR_PRECONDITION", "selfp") or "selfp").strip().lower(),
            "outer_ksp": str(os.getenv("PYCUTFEM_LINEAR_KSP_TYPE", "") or "").strip().lower() or None,
            "outer_pc": "fieldsplit",
            "rest_ksp": str(os.getenv("PYCUTFEM_LINEAR_SCHUR_REST_KSP_TYPE", "preonly") or "preonly").strip().lower(),
            "rest_pc": str(os.getenv("PYCUTFEM_LINEAR_SCHUR_REST_PC_TYPE", "ilu") or "ilu").strip().lower(),
            "rest_factor_solver_type": str(os.getenv("PYCUTFEM_LINEAR_SCHUR_REST_FACTOR_SOLVER_TYPE", "") or "").strip().lower() or None,
            "pressure_ksp": str(os.getenv("PYCUTFEM_LINEAR_SCHUR_PRESSURE_KSP_TYPE", "preonly") or "preonly").strip().lower(),
            "pressure_pc": str(os.getenv("PYCUTFEM_LINEAR_SCHUR_PRESSURE_PC_TYPE", "jacobi") or "jacobi").strip().lower(),
            "pressure_factor_solver_type": str(os.getenv("PYCUTFEM_LINEAR_SCHUR_PRESSURE_FACTOR_SOLVER_TYPE", "") or "").strip().lower() or None,
        }

    def _reduced_indices_for_field(self, field_name: str) -> np.ndarray:
        try:
            full_idx = np.asarray(self.dh.get_field_slice(field_name), dtype=int).ravel()
        except Exception:
            return np.empty((0,), dtype=int)
        if full_idx.size == 0:
            return np.empty((0,), dtype=int)
        if getattr(self, "constraints", None) is not None:
            red_idx_list: list[int] = []
            for gd in full_idx.tolist():
                try:
                    mcol = self.constraints.master_index_for(int(gd))
                except Exception:
                    mcol = None
                if mcol is None:
                    continue
                ridx = int(self.full_to_red[int(mcol)]) if 0 <= int(mcol) < int(self.full_to_red.size) else -1
                if ridx >= 0:
                    red_idx_list.append(ridx)
            if not red_idx_list:
                return np.empty((0,), dtype=int)
            return np.asarray(sorted(set(red_idx_list)), dtype=int)
        red_idx = np.asarray(self.full_to_red[full_idx], dtype=int).ravel()
        red_idx = red_idx[red_idx >= 0]
        if red_idx.size == 0:
            return np.empty((0,), dtype=int)
        return np.unique(red_idx)

    def _apply_linear_schur_fieldsplit(self, ksp, n: int) -> bool:
        from petsc4py import PETSc  # type: ignore

        cfg = getattr(self, "_linear_schur_cfg", None)
        if cfg is None:
            cfg = self._petsc_linear_schur_cfg_from_env()
        if cfg is None:
            return False

        p_red = self._reduced_indices_for_field(str(cfg.get("pressure_field", "p") or "p"))
        if p_red.size == 0 or p_red.size >= int(n):
            return False
        rest_red = np.setdiff1d(np.arange(int(n), dtype=int), p_red, assume_unique=False)
        if rest_red.size == 0:
            return False

        pc = ksp.getPC()
        if cfg.get("outer_ksp"):
            ksp.setType(str(cfg["outer_ksp"]))
        pc.setType(str(cfg.get("outer_pc", "fieldsplit") or "fieldsplit"))
        pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)

        rest_is = PETSc.IS().createGeneral(rest_red.astype(PETSc.IntType), comm=PETSc.COMM_SELF)
        p_is = PETSc.IS().createGeneral(p_red.astype(PETSc.IntType), comm=PETSc.COMM_SELF)
        pc.setFieldSplitIS(("rest", rest_is), ("p", p_is))

        opts = PETSc.Options()
        opts["pc_fieldsplit_schur_fact_type"] = str(cfg.get("schur_fact", "full") or "full")
        opts["pc_fieldsplit_schur_precondition"] = str(cfg.get("schur_pre", "selfp") or "selfp")
        if cfg.get("rest_ksp"):
            opts["fieldsplit_rest_ksp_type"] = str(cfg["rest_ksp"])
        if cfg.get("rest_pc"):
            opts["fieldsplit_rest_pc_type"] = str(cfg["rest_pc"])
        if cfg.get("rest_factor_solver_type"):
            opts["fieldsplit_rest_pc_factor_mat_solver_type"] = str(cfg["rest_factor_solver_type"])
        if cfg.get("pressure_ksp"):
            opts["fieldsplit_p_ksp_type"] = str(cfg["pressure_ksp"])
        if cfg.get("pressure_pc"):
            opts["fieldsplit_p_pc_type"] = str(cfg["pressure_pc"])
        if cfg.get("pressure_factor_solver_type"):
            opts["fieldsplit_p_pc_factor_mat_solver_type"] = str(cfg["pressure_factor_solver_type"])
        self._linear_schur_last_info = {
            "pressure_field": str(cfg.get("pressure_field", "p") or "p"),
            "p_dofs": int(p_red.size),
            "rest_dofs": int(rest_red.size),
        }
        return True

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
            try:
                # When the interface moves or the active set changes, the CSR
                # pattern can legitimately change; do not hard-fail on new nonzeros.
                mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
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
            # Optional env overrides for the internal PETSc linear solve used by
            # the custom Newton/PDAS path. This keeps the benchmark CLI stable
            # while allowing direct LU vs Krylov experiments.
            try:
                ksp_type = str(os.getenv("PYCUTFEM_LINEAR_KSP_TYPE", "") or "").strip()
                if ksp_type:
                    ksp.setType(ksp_type)
            except Exception:
                pass
            try:
                pc_type = str(os.getenv("PYCUTFEM_LINEAR_PC_TYPE", "") or "").strip()
                if pc_type:
                    pc.setType(pc_type)
            except Exception:
                pass
            try:
                factor_type = str(os.getenv("PYCUTFEM_LINEAR_PC_FACTOR_SOLVER_TYPE", "") or "").strip()
                if factor_type:
                    pc.setFactorSolverType(factor_type)
            except Exception:
                pass
            schur_applied = False
            try:
                schur_applied = bool(self._apply_linear_schur_fieldsplit(ksp, n))
            except Exception:
                schur_applied = False
            try:
                rtol_raw = str(os.getenv("PYCUTFEM_LINEAR_KSP_RTOL", "") or "").strip()
                atol_raw = str(os.getenv("PYCUTFEM_LINEAR_KSP_ATOL", "") or "").strip()
                dtol_raw = str(os.getenv("PYCUTFEM_LINEAR_KSP_DTOL", "") or "").strip()
                maxit_raw = str(os.getenv("PYCUTFEM_LINEAR_KSP_MAX_IT", "") or "").strip()
                rtol = float(rtol_raw) if rtol_raw else None
                atol = float(atol_raw) if atol_raw else None
                dtol = float(dtol_raw) if dtol_raw else None
                max_it = int(maxit_raw) if maxit_raw else None
                if any(v is not None for v in (rtol, atol, dtol, max_it)):
                    ksp.setTolerances(
                        rtol=PETSc.DECIDE if rtol is None else rtol,
                        atol=PETSc.DECIDE if atol is None else atol,
                        divtol=PETSc.DECIDE if dtol is None else dtol,
                        max_it=PETSc.DECIDE if max_it is None else max_it,
                    )
            except Exception:
                pass
            # Allow users to override via PETSc options (e.g. -pc_factor_mat_solver_type superlu_dist).
            ksp.setFromOptions()
            if os.getenv("PYCUTFEM_LINEAR_KSP_TRACE", "").lower() in {"1", "true", "yes"} and schur_applied:
                try:
                    info = dict(getattr(self, "_linear_schur_last_info", {}) or {})
                    print(
                        "        [ksp-schur] pressure_field="
                        + str(info.get("pressure_field", "p"))
                        + f" p_dofs={int(info.get('p_dofs', 0))} rest_dofs={int(info.get('rest_dofs', 0))}"
                    )
                except Exception:
                    pass

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
        rhs = np.asarray(rhs, dtype=np.float64)
        if (not np.all(np.isfinite(a))) or (not np.all(np.isfinite(rhs))):
            # Non-finite values can corrupt direct solvers (e.g. MUMPS) and even
            # leave cached PETSc objects in a bad internal state. Clear the cache
            # so adaptive dt retries start with a clean KSP/Mat.
            try:
                self._petsc_linear_cache = None
            except Exception:
                pass
            raise RuntimeError("PETSc linear solve received non-finite entries in matrix or rhs.")
        try:
            mat.setValuesCSR(ia, ja, a)
        except TypeError:
            mat.setValuesCSR((ia, ja), a)
        mat.assemblyBegin()
        mat.assemblyEnd()

        # Load RHS and solve.
        #
        # IMPORTANT: petsc4py's Vec.getArray() returns a NumPy view that can keep
        # the underlying Vec "locked" for array access while the view exists.
        # If a PETSc solve throws an exception while such a view is live, the lock
        # can persist and break subsequent solves (e.g. during adaptive-Δt retries).
        b_arr = b.getArray()
        b_arr[:] = rhs
        del b_arr

        x.set(0.0)
        try:
            ksp.solve(b, x)
        except PETSc.Error:
            # Clear cache so the next attempt rebuilds PETSc objects from scratch.
            try:
                self._petsc_linear_cache = None
            except Exception:
                pass
            raise

        try:
            reason = int(ksp.getConvergedReason())
        except Exception:
            reason = 0
        if os.getenv("PYCUTFEM_LINEAR_KSP_TRACE", "").lower() in {"1", "true", "yes"}:
            try:
                its = int(ksp.getIterationNumber())
            except Exception:
                its = -1
            try:
                rnorm = float(ksp.getResidualNorm())
            except Exception:
                rnorm = float("nan")
            print(f"        [ksp] type={ksp.getType()} pc={ksp.getPC().getType()} reason={reason} its={its} rnorm={rnorm:.3e}")
        if reason <= 0:
            try:
                self._petsc_linear_cache = None
            except Exception:
                pass
            its = None
            try:
                its = int(ksp.getIterationNumber())
            except Exception:
                pass
            raise RuntimeError(
                f"PETSc KSP failed to converge (reason={reason}"
                + (f", its={its}" if its is not None else "")
                + ")."
            )

        x_arr = x.getArray(readonly=True)
        out = x_arr.copy()
        del x_arr
        if not np.all(np.isfinite(out)):
            try:
                self._petsc_linear_cache = None
            except Exception:
                pass
            raise RuntimeError("PETSc linear solve returned non-finite solution values.")
        return out


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
            if os.getenv("PYCUTFEM_LS_FAIL_HARD", "1").lower() not in {"0", "false", "no"}:
                raise RuntimeError("Line search failed: no residual decrease.")
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
            print("        Line search failed catastrophically. No descent found.")
            if os.getenv("PYCUTFEM_LS_FAIL_HARD", "1").lower() not in {"0", "false", "no"}:
                raise RuntimeError("Line search failed: no residual decrease.")
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

        if getattr(self, "constraints", None) is not None:
            self._build_reduced_pattern_constraints()
            return

        _profile = os.getenv("PYCUTFEM_PROFILE_SETUP", "").lower() in {"1", "true", "yes"}
        _t0 = time.perf_counter() if _profile else 0.0

        # 1) Build reduced sparsity as COO pairs then compress to CSR.
        #    Use an incidence-matrix approach to avoid O(n_elem * n_loc^2) COO expansion:
        #      - Build sparse B with B[e, i] = 1 if reduced DOF i is present in entity e
        #      - Pattern is then (B.T @ B) > 0 (boolean adjacency)
        #    This keeps startup fast even when each entity contributes a dense local block.
        _t_pat0 = time.perf_counter() if _profile else 0.0
        ent_row_blocks: list[np.ndarray] = []
        ent_col_blocks: list[np.ndarray] = []
        ent_rows = 0
        seen_gdofs_maps: set[int] = set()
        ndof_full = int(getattr(self.full_to_red, "size", 0))
        if _profile:
            try:
                print(f"[setup] kernels_K: {int(len(self.kernels_K))}")
            except Exception:
                pass
        for ker in self.kernels_K:
            gdofs = ker.static_args.get("gdofs_map")
            if not isinstance(gdofs, np.ndarray) or gdofs.ndim != 2:
                continue
            gid = int(id(gdofs))
            if gid in seen_gdofs_maps:
                continue
            seen_gdofs_maps.add(gid)
            full = np.asarray(gdofs, dtype=np.int64)
            if full.size == 0:
                continue
            valid = (full >= 0) & (full < ndof_full)
            if not np.any(valid):
                continue
            red = -np.ones_like(full, dtype=np.int32)
            red[valid] = self.full_to_red[full[valid]]
            mask = red >= 0
            if not np.any(mask):
                continue
            cnt = np.asarray(mask.sum(axis=1), dtype=np.int32)
            nonempty = np.nonzero(cnt)[0]
            if nonempty.size == 0:
                continue
            # Flatten reduced DOF indices in row-major order over the selected entity rows.
            cols = red[nonempty][mask[nonempty]].astype(np.int32, copy=False)
            # Row indices aligned with the boolean flattening above.
            rows = np.repeat(
                np.arange(ent_rows, ent_rows + int(nonempty.size), dtype=np.int32),
                cnt[nonempty],
            ).astype(np.int32, copy=False)
            ent_row_blocks.append(rows)
            ent_col_blocks.append(cols)
            ent_rows += int(nonempty.size)

        if ent_row_blocks:
            row_all = np.concatenate(ent_row_blocks)
            col_all = np.concatenate(ent_col_blocks)
            B = sp.coo_matrix(
                (np.ones(row_all.shape[0], dtype=np.int8), (row_all, col_all)),
                shape=(int(ent_rows), int(n)),
            ).tocsr()
            pat = (B.T @ B).tocsr()
            # pattern only – drop counts
            try:
                pat.data[:] = 1
            except Exception:
                pat.data = np.ones_like(pat.data, dtype=np.int8)
            pat.sort_indices()
            indptr = np.asarray(pat.indptr, dtype=np.int32)
            indices = np.asarray(pat.indices, dtype=np.int32)
        else:
            indptr = np.zeros(n + 1, dtype=np.int32)
            indices = np.zeros(0, dtype=np.int32)
        self._csr_indptr, self._csr_indices = indptr, indices
        if _profile:
            print(f"[setup] pattern csr: {time.perf_counter() - _t_pat0:.3f}s  nnz={int(indices.size)} n={int(n)}")

        # PETSc matrix cache: sparsity can change when the interface moves and/or the
        # active DOF set is recomputed. The cached PETSc.Mat preallocation must be
        # rebuilt when the CSR pattern changes, otherwise PETSc can raise
        # "new nonzero" insertion errors.
        try:
            if str(getattr(self.lp, "backend", "scipy") or "scipy").lower() == "petsc":
                if hasattr(self, "_petsc_linear_cache"):
                    delattr(self, "_petsc_linear_cache")
        except Exception:
            pass

        # CSR positions as a sparse matrix (data = position index).
        #
        # NOTE: SciPy CSR fancy indexing can dominate startup for medium-sized
        # problems (it allocates many temporary submatrices). Default to a
        # row-wise `searchsorted` path which is usually faster end-to-end for
        # CutFEM-style kernels. The old sparse-indexing path can be re-enabled
        # via `PYCUTFEM_PATTERN_POS_CSR=1` for comparison/regressions.
        pos_csr = None
        use_pos_csr = os.getenv("PYCUTFEM_PATTERN_POS_CSR", "").lower() in {"1", "true", "yes"}
        if use_pos_csr:
            try:
                if indices.size:
                    pos_data = np.arange(int(indices.size), dtype=np.int32)
                    pos_csr = sp.csr_matrix((pos_data, indices, indptr), shape=(int(n), int(n)))
            except Exception:
                pos_csr = None

        # 3) Ragged per-element scatter plans (store 1-D position arrays + local idx).
        #    Compute CSR data positions with row-wise searchsorted to avoid a large
        #    Python dict of (row,col) -> idx.
        _t_pl0 = time.perf_counter() if _profile else 0.0
        self._elem_pos = []   # per-kernel: list of 1-D arrays (len = n_act^2)
        self._elem_lidx = []  # per-kernel: list of local indices kept (len = n_act)
        plan_cache: dict[int, tuple[object, object]] = {}
        cache_hits = 0
        for ker in self.kernels_K:
            gdofs = ker.static_args.get("gdofs_map")
            if isinstance(gdofs, np.ndarray) and gdofs.ndim == 2:
                cached = plan_cache.get(int(id(gdofs)))
                if cached is not None:
                    self._elem_pos.append(cached[0])
                    self._elem_lidx.append(cached[1])
                    cache_hits += 1
                    continue
            if not isinstance(gdofs, np.ndarray) or gdofs.ndim != 2:
                self._elem_pos.append([])
                self._elem_lidx.append([])
                continue

            full = np.asarray(gdofs, dtype=np.int64)
            n_ent, n_loc = int(full.shape[0]), int(full.shape[1])
            if n_ent == 0:
                pos_flat = np.empty((0, int(n_loc * n_loc)), dtype=np.int32)
                plan_cache[int(id(gdofs))] = (pos_flat, None)
                self._elem_pos.append(pos_flat)
                self._elem_lidx.append(None)
                continue

            # Map full -> reduced once for the entire entity batch.
            red = -np.ones_like(full, dtype=np.int32)
            valid = (full >= 0) & (full < ndof_full)
            if np.any(valid):
                red[valid] = self.full_to_red[full[valid]]

            # Dense per-entity position plan:
            # - shape (n_ent, n_loc*n_loc)
            # - local ordering matches `Kloc[e].ravel()` for the kernel.
            # - inactive row/col pairs are marked with -1 and skipped at scatter time.
            if _HAVE_NUMBA_REDUCED_PATTERN:
                _wait_numba_reduced_pattern_precompile()
                pos_flat = _build_pos_flat_numba(red, indptr, indices)
            else:
                pos_flat = -np.ones((n_ent, int(n_loc * n_loc)), dtype=np.int32)
                for e in range(n_ent):
                    row_red = red[e]
                    col_mask = row_red >= 0
                    if not np.any(col_mask):
                        continue
                    cols_act = row_red[col_mask].astype(np.int32, copy=False)
                    all_cols_active = bool(np.all(col_mask))
                    for i in range(n_loc):
                        ra = int(row_red[i])
                        if ra < 0:
                            continue
                        s = int(indptr[ra])
                        epos = int(indptr[ra + 1])
                        row_cols = indices[s:epos]
                        row_pos = s + np.searchsorted(row_cols, cols_act)
                        seg = pos_flat[e, int(i * n_loc) : int((i + 1) * n_loc)]
                        if all_cols_active:
                            seg[:] = row_pos
                        else:
                            seg[col_mask] = row_pos

            plan_cache[int(id(gdofs))] = (pos_flat, None)
            self._elem_pos.append(pos_flat)
            self._elem_lidx.append(None)

        # No constraint metadata in this mode.
        self._elem_extra = None

        # Precompute residual scatter maps (full gdof -> reduced row index) for
        # each residual kernel. This avoids per-element Python loops and repeated
        # allocation in _assemble_system_reduced_fast().
        try:
            res_maps: list[np.ndarray | None] = []
            res_cache: dict[int, np.ndarray] = {}
            ndof_full = int(getattr(self.full_to_red, "size", 0))
            for kerF in list(getattr(self, "kernels_F", []) or []):
                gdofs = kerF.static_args.get("gdofs_map")
                if not isinstance(gdofs, np.ndarray) or gdofs.ndim != 2:
                    res_maps.append(None)
                    continue
                gid = int(id(gdofs))
                cached = res_cache.get(gid)
                if cached is None:
                    full = np.asarray(gdofs, dtype=np.int64)
                    red_map = -np.ones_like(full, dtype=np.int32)
                    valid = (full >= 0) & (full < ndof_full)
                    if np.any(valid):
                        red_map[valid] = self.full_to_red[full[valid]]
                    res_cache[gid] = red_map
                    cached = red_map
                res_maps.append(cached)
            self._res_red_maps = res_maps
        except Exception:
            self._res_red_maps = None

        self._pattern_stale = False
        if _profile:
            try:
                print(f"[setup] unique gdofs_maps: {int(len(seen_gdofs_maps))}  plan cache hits: {int(cache_hits)}")
            except Exception:
                pass
            print(f"[setup] scatter plans: {time.perf_counter() - _t_pl0:.3f}s")
            print(f"[setup] build_reduced_pattern total: {time.perf_counter() - _t0:.3f}s")


    def _build_reduced_pattern_constraints(self) -> None:
        """
        Build reduced CSR pattern + per-entity scatter plans when hanging-node
        constraints are present.

        We assemble directly in the master space (u_full = E @ u_master) by
        distributing slave DOF contributions to their master DOFs using the
        constraint weights, avoiding the expensive full-space assembly and the
        subsequent Eᵀ A E condensation.
        """

        constraints = getattr(self, "constraints", None)
        if constraints is None:
            raise RuntimeError("_build_reduced_pattern_constraints called without constraints.")

        n = int(len(self.active_dofs))
        if n <= 0:
            self._csr_indptr = np.zeros(1, dtype=np.int32)
            self._csr_indices = np.zeros(0, dtype=np.int32)
            self._elem_pos = [[] for _ in self.kernels_K]
            self._elem_lidx = [[] for _ in self.kernels_K]
            self._elem_extra = [[] for _ in self.kernels_K]
            self._pattern_stale = False
            return

        # Lookup tables:
        # - full gdof -> master column index (or -1 if slave/unmapped)
        # - slave full gdof -> (master_cols[], weights[])
        ndof_full = int(getattr(self.dh, "total_dofs", 0))
        full_to_master = np.full(ndof_full, -1, dtype=np.int32)
        master_ids = np.asarray(getattr(constraints, "master_ids", np.array([], dtype=int)), dtype=int)
        for col, gd in enumerate(master_ids.tolist()):
            gd_i = int(gd)
            if 0 <= gd_i < ndof_full:
                full_to_master[gd_i] = int(col)

        slave_to_master_cols: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        is_slave = np.zeros(ndof_full, dtype=bool)
        for sdof, combo in getattr(constraints, "slave_to_master", {}).items():
            sdof_i = int(sdof)
            if sdof_i < 0 or sdof_i >= ndof_full:
                continue
            is_slave[sdof_i] = True
            mcols: list[int] = []
            wts: list[float] = []
            for mdof, w in combo:
                mdof_i = int(mdof)
                if mdof_i < 0 or mdof_i >= ndof_full:
                    continue
                mcol = int(full_to_master[mdof_i])
                if mcol < 0:
                    continue
                mcols.append(mcol)
                wts.append(float(w))
            if mcols:
                slave_to_master_cols[sdof_i] = (np.asarray(mcols, dtype=np.int32), np.asarray(wts, dtype=float))

        # Save for residual-only assembly path.
        self._constr_full_to_master = full_to_master
        self._constr_is_slave = is_slave
        self._constr_slave_map = slave_to_master_cols

        full_to_red_master = np.asarray(self.full_to_red, dtype=int)

        def _add_red(red_set: set[int], gd: int) -> None:
            if gd < 0 or gd >= ndof_full:
                return
            if bool(is_slave[gd]):
                combo = slave_to_master_cols.get(int(gd))
                if combo is None:
                    return
                mcols, _w = combo
                for mcol in mcols.tolist():
                    red = int(full_to_red_master[int(mcol)])
                    if red >= 0:
                        red_set.add(red)
                return
            mcol = int(full_to_master[int(gd)])
            if mcol < 0:
                return
            red = int(full_to_red_master[mcol])
            if red >= 0:
                red_set.add(red)

        # 1) Collect column sets per reduced row
        rows_cols: list[set[int]] = [set() for _ in range(n)]
        for ker in self.kernels_K:
            gdofs = ker.static_args.get("gdofs_map")
            if not isinstance(gdofs, np.ndarray) or gdofs.ndim != 2 or gdofs.shape[0] == 0:
                continue
            for e in range(int(gdofs.shape[0])):
                full = np.asarray(gdofs[e], dtype=int)
                red_set: set[int] = set()
                for gd in full.tolist():
                    _add_red(red_set, int(gd))
                if not red_set:
                    continue
                for r in red_set:
                    rows_cols[r].update(red_set)

        # 2) Build CSR (row-wise sorted for determinism)
        indptr = np.zeros(n + 1, dtype=np.int32)
        for i in range(n):
            indptr[i + 1] = indptr[i] + len(rows_cols[i])
        indices = np.empty(indptr[-1], dtype=np.int32)
        for i in range(n):
            cols = sorted(rows_cols[i])
            indices[indptr[i] : indptr[i + 1]] = np.asarray(cols, dtype=np.int32)
        self._csr_indptr, self._csr_indices = indptr, indices

        # Pattern changed → drop PETSc cache
        try:
            if str(getattr(self.lp, "backend", "scipy") or "scipy").lower() == "petsc":
                if hasattr(self, "_petsc_linear_cache"):
                    delattr(self, "_petsc_linear_cache")
        except Exception:
            pass

        # 3) Per-kernel scatter plans
        self._elem_pos = []
        self._elem_lidx = []
        self._elem_extra = []

        def _pflat(rows: np.ndarray) -> np.ndarray:
            rows = np.asarray(rows, dtype=np.int32)
            m = int(rows.size)
            if m <= 0:
                return np.empty(0, dtype=np.int32)
            out = np.empty(m * m, dtype=np.int32)
            for ai, r in enumerate(rows.tolist()):
                r_i = int(r)
                s = int(indptr[r_i])
                t = int(indptr[r_i + 1])
                cols = indices[s:t]
                jj = np.searchsorted(cols, rows)
                out[ai * m : (ai + 1) * m] = s + jj.astype(np.int32, copy=False)
            return out

        for ker in self.kernels_K:
            gdofs = ker.static_args.get("gdofs_map")
            if not isinstance(gdofs, np.ndarray) or gdofs.ndim != 2:
                self._elem_pos.append([])
                self._elem_lidx.append([])
                self._elem_extra.append([])
                continue
            nel = int(gdofs.shape[0])
            nloc = int(gdofs.shape[1]) if gdofs.ndim == 2 else 0
            pos_list: list[np.ndarray] = []
            lidx_list: list[np.ndarray] = []
            extra_list: list[object | None] = []

            for e in range(nel):
                full = np.asarray(gdofs[e], dtype=int)
                valid = full >= 0
                if not np.any(valid):
                    pos_list.append(np.empty(0, dtype=np.int32))
                    lidx_list.append(np.empty(0, dtype=np.int32))
                    extra_list.append(None)
                    continue

                full_valid = full[valid].astype(int, copy=False)
                has_slave = bool(np.any(is_slave[full_valid]))

                if not has_slave:
                    mcols = full_to_master[full_valid]
                    ok = mcols >= 0
                    if not np.any(ok):
                        pos_list.append(np.empty(0, dtype=np.int32))
                        lidx_list.append(np.empty(0, dtype=np.int32))
                        extra_list.append(None)
                        continue
                    reds = full_to_red_master[mcols[ok]]
                    keep = reds >= 0
                    if not np.any(keep):
                        pos_list.append(np.empty(0, dtype=np.int32))
                        lidx_list.append(np.empty(0, dtype=np.int32))
                        extra_list.append(None)
                        continue
                    rows = np.asarray(reds[keep], dtype=np.int32)
                    lidx = np.nonzero(valid)[0][ok][keep].astype(np.int32, copy=False)
                    pos_list.append(_pflat(rows))
                    lidx_list.append(lidx)
                    extra_list.append(None)
                    continue

                # Constrained entity: local mapping P via ragged rows
                maps: list[list[tuple[int, float]]] = [[] for _ in range(nloc)]
                red_set: set[int] = set()
                for i in range(nloc):
                    gd = int(full[i])
                    if gd < 0:
                        continue
                    if bool(is_slave[gd]):
                        combo = slave_to_master_cols.get(gd)
                        if combo is None:
                            continue
                        mcols, wts = combo
                        for mcol, w in zip(mcols.tolist(), wts.tolist()):
                            red = int(full_to_red_master[int(mcol)])
                            if red >= 0:
                                maps[i].append((red, float(w)))
                                red_set.add(red)
                    else:
                        mcol = int(full_to_master[gd])
                        if mcol < 0:
                            continue
                        red = int(full_to_red_master[mcol])
                        if red >= 0:
                            maps[i].append((red, 1.0))
                            red_set.add(red)

                if not red_set:
                    pos_list.append(np.empty(0, dtype=np.int32))
                    lidx_list.append(np.empty(0, dtype=np.int32))
                    extra_list.append(None)
                    continue

                rows_unique = np.asarray(sorted(red_set), dtype=np.int32)
                red_to_local = {int(r): j for j, r in enumerate(rows_unique.tolist())}

                ptr = np.zeros(nloc + 1, dtype=np.int32)
                j_acc: list[int] = []
                w_acc: list[float] = []
                for i in range(nloc):
                    tmp: dict[int, float] = {}
                    for red, w in maps[i]:
                        jj = int(red_to_local[int(red)])
                        tmp[jj] = tmp.get(jj, 0.0) + float(w)
                    for jj, w in tmp.items():
                        j_acc.append(int(jj))
                        w_acc.append(float(w))
                    ptr[i + 1] = int(len(j_acc))
                rep_j = np.asarray(j_acc, dtype=np.int32)
                rep_w = np.asarray(w_acc, dtype=float)

                pos_list.append(_pflat(rows_unique))
                lidx_list.append(np.empty(0, dtype=np.int32))
                extra_list.append((rows_unique, ptr, rep_j, rep_w))

            self._elem_pos.append(pos_list)
            self._elem_lidx.append(lidx_list)
            self._elem_extra.append(extra_list)

        self._pattern_stale = False


    def _assemble_system_reduced_fast(self, coeffs):
        """
        Assemble reduced (A, R) using the prebuilt CSR pattern and ragged
        per-element scatter plans from _build_reduced_pattern().
        Returns: (A_csr, R)
        """

        if getattr(self, "_pattern_stale", True):
            self._build_reduced_pattern()

        # Residual kernels may be compiling asynchronously (parallel setup).
        self._finish_residual_kernel_compilation()

        indptr, indices = self._csr_indptr, self._csr_indices
        n = indptr.size - 1
        data = np.zeros(indices.size, dtype=float)
        R = np.zeros(n, dtype=float)

        profile = os.getenv("PYCUTFEM_PROFILE_KERNELS", "").lower() in {"1", "true", "yes"}
        prof_entries = []
        debug_kernel = os.getenv("PYCUTFEM_DEBUG_KERNELS", "").lower() in {"1", "true", "yes"}
        debug_kernel_min_elems = int(os.getenv("PYCUTFEM_DEBUG_KERNELS_MIN_ELEMS", "0") or "0")
        trace_exec = os.getenv("PYCUTFEM_TRACE_KERNEL_EXEC", "").lower() in {"1", "true", "yes"}
        debug_seen = getattr(self, "_debug_kernel_seen", None)
        if debug_kernel and debug_seen is None:
            debug_seen = set()
            self._debug_kernel_seen = debug_seen

        # Matrix (Jacobian) blocks
        extra_lists = getattr(self, "_elem_extra", None)
        if extra_lists is None:
            extra_lists = [None for _ in self.kernels_K]

        for ker, pos_list, lidx_list, extra_list in zip(self.kernels_K, self._elem_pos, self._elem_lidx, extra_lists):
            gdofs = ker.static_args.get("gdofs_map")
            if isinstance(gdofs, np.ndarray) and gdofs.shape[0] == 0:
                continue
            t_exec = time.perf_counter()
            if trace_exec:
                print(
                    f"        [kern] exec jacobian dom={getattr(ker,'domain','?')} side={getattr(ker,'side',None)}",
                    flush=True,
                )
            Kloc, _, _ = ker.exec(coeffs)  # shape [nel, nloc, nloc]
            exec_time = time.perf_counter() - t_exec
            if trace_exec:
                print(
                    f"        [kern] done jacobian dom={getattr(ker,'domain','?')} exec={exec_time:.3e}s",
                    flush=True,
                )
            if debug_kernel and isinstance(debug_seen, set) and (id(ker) not in debug_seen):
                try:
                    n_ent = int(getattr(Kloc, "shape", (0,))[0])
                except Exception:
                    n_ent = 0
                if n_ent >= debug_kernel_min_elems:
                    try:
                        qw = ker.static_args.get("qw")
                        qw_sum = float(np.nansum(np.abs(qw))) if isinstance(qw, np.ndarray) else float("nan")
                        qw_min = float(np.nanmin(qw)) if isinstance(qw, np.ndarray) and qw.size else float("nan")
                        qw_max = float(np.nanmax(qw)) if isinstance(qw, np.ndarray) and qw.size else float("nan")
                        kmax = float(np.nanmax(np.abs(Kloc))) if isinstance(Kloc, np.ndarray) and Kloc.size else float("nan")
                        n_valid = n_active = 0
                        active_frac = float("nan")
                        if isinstance(gdofs, np.ndarray):
                            valid = gdofs >= 0
                            if np.any(valid):
                                rmap = -np.ones_like(gdofs, dtype=int)
                                rmap[valid] = self.full_to_red[gdofs[valid]]
                                active = rmap >= 0
                                n_valid = int(np.count_nonzero(valid))
                                n_active = int(np.count_nonzero(active))
                                active_frac = float(n_active) / float(max(1, n_valid))
                        side = getattr(ker, "side", None)
                        print(
                            f"[kern] jacobian dom={getattr(ker,'domain','?')} side={side} "
                            f"elems={n_ent} max|K|={kmax:.3e} "
                            f"qw_sum={qw_sum:.3e} qw=[{qw_min:.3e},{qw_max:.3e}] "
                            f"active_frac={active_frac:.3e} ({n_active}/{n_valid})"
                        )
                    except Exception:
                        pass
                debug_seen.add(id(ker))
            t_scatter = time.perf_counter()
            if extra_list is None:
                if isinstance(pos_list, np.ndarray):
                    # Dense per-entity plan: pos_list[e] has length n_loc*n_loc and
                    # matches Kloc[e].ravel(); entries are -1 for inactive pairs.
                    Kflat = np.asarray(Kloc, dtype=float).reshape(int(Kloc.shape[0]), -1)
                    for e in range(int(pos_list.shape[0])):
                        pflat = np.asarray(pos_list[e], dtype=np.int32)
                        if pflat.size == 0:
                            continue
                        m = pflat >= 0
                        if not np.any(m):
                            continue
                        data[pflat[m]] += Kflat[e, m]
                else:
                    for e, (pflat, lidx) in enumerate(zip(pos_list, lidx_list)):
                        if pflat.size == 0:
                            continue
                        Kel = Kloc[e][np.ix_(lidx, lidx)].ravel()
                        data[pflat] += Kel
            else:
                # Constraint-aware scatter for entities with hanging-node slaves.
                for e, (pflat, lidx, meta) in enumerate(zip(pos_list, lidx_list, extra_list)):
                    if pflat.size == 0:
                        continue
                    if meta is None:
                        Kel = Kloc[e][np.ix_(lidx, lidx)].ravel()
                        data[pflat] += Kel
                        continue
                    rows_unique, ptr, rep_j, rep_w = meta
                    m = int(rows_unique.size)
                    if m <= 0:
                        continue
                    Ke = Kloc[e]
                    nloc = int(Ke.shape[0])

                    # B = Ke @ P, where P maps local full dofs to local reduced rows.
                    B = np.zeros((nloc, m), dtype=float)
                    for j_loc in range(nloc):
                        s0 = int(ptr[j_loc])
                        s1 = int(ptr[j_loc + 1])
                        if s1 <= s0:
                            continue
                        col = Ke[:, j_loc]
                        for k in range(s0, s1):
                            a = int(rep_j[k])
                            wv = float(rep_w[k])
                            B[:, a] += wv * col

                    # Scatter K_m = Pᵀ B directly into global CSR data.
                    for i_loc in range(nloc):
                        s0 = int(ptr[i_loc])
                        s1 = int(ptr[i_loc + 1])
                        if s1 <= s0:
                            continue
                        brow = B[i_loc, :]
                        for k in range(s0, s1):
                            a = int(rep_j[k])
                            wv = float(rep_w[k])
                            data[pflat[a * m : (a + 1) * m]] += wv * brow
            scatter_time = time.perf_counter() - t_scatter
            if profile:
                prof_entries.append(
                    ("jacobian", getattr(ker, "domain", "unknown"), exec_time, scatter_time, len(pos_list))
                )

        # Residual blocks
        res_maps = getattr(self, "_res_red_maps", None)
        has_constraints = getattr(self, "constraints", None) is not None
        for k_idx, ker in enumerate(self.kernels_F):
            gdofs = ker.static_args.get("gdofs_map")
            if isinstance(gdofs, np.ndarray) and gdofs.shape[0] == 0:
                continue
            t_exec = time.perf_counter()
            if trace_exec:
                print(
                    f"        [kern] exec residual dom={getattr(ker,'domain','?')} side={getattr(ker,'side',None)}",
                    flush=True,
                )
            _, Floc, _ = ker.exec(coeffs)  # [nel, nloc]
            exec_time = time.perf_counter() - t_exec
            if trace_exec:
                print(
                    f"        [kern] done residual dom={getattr(ker,'domain','?')} exec={exec_time:.3e}s",
                    flush=True,
                )
            if debug_kernel and isinstance(debug_seen, set) and (id(ker) not in debug_seen):
                try:
                    n_ent = int(getattr(Floc, "shape", (0,))[0])
                except Exception:
                    n_ent = 0
                if n_ent >= debug_kernel_min_elems:
                    try:
                        qw = ker.static_args.get("qw")
                        qw_sum = float(np.nansum(np.abs(qw))) if isinstance(qw, np.ndarray) else float("nan")
                        qw_min = float(np.nanmin(qw)) if isinstance(qw, np.ndarray) and qw.size else float("nan")
                        qw_max = float(np.nanmax(qw)) if isinstance(qw, np.ndarray) and qw.size else float("nan")
                        absF = np.abs(Floc) if isinstance(Floc, np.ndarray) else None
                        fmax = float(np.nanmax(absF)) if isinstance(absF, np.ndarray) and absF.size else float("nan")
                        n_valid = n_active = 0
                        active_frac = float("nan")
                        fmax_active = 0.0
                        fmax_dropped = 0.0
                        if isinstance(gdofs, np.ndarray) and isinstance(absF, np.ndarray) and absF.size:
                            valid = gdofs >= 0
                            if np.any(valid):
                                rmap = -np.ones_like(gdofs, dtype=int)
                                rmap[valid] = self.full_to_red[gdofs[valid]]
                                active = rmap >= 0
                                n_valid = int(np.count_nonzero(valid))
                                n_active = int(np.count_nonzero(active))
                                active_frac = float(n_active) / float(max(1, n_valid))
                                if np.any(active):
                                    fmax_active = float(np.nanmax(absF[active]))
                                dropped = valid & ~active
                                if np.any(dropped):
                                    fmax_dropped = float(np.nanmax(absF[dropped]))
                        side = getattr(ker, "side", None)
                        print(
                            f"[kern] residual dom={getattr(ker,'domain','?')} side={side} "
                            f"elems={n_ent} max|F|={fmax:.3e} "
                            f"qw_sum={qw_sum:.3e} qw=[{qw_min:.3e},{qw_max:.3e}] "
                            f"active_frac={active_frac:.3e} ({n_active}/{n_valid}) "
                            f"max|F_active|={fmax_active:.3e} max|F_drop|={fmax_dropped:.3e}"
                        )
                    except Exception:
                        pass
                debug_seen.add(id(ker))
            t_scatter = time.perf_counter()
            if not has_constraints and isinstance(res_maps, list) and k_idx < len(res_maps) and isinstance(res_maps[k_idx], np.ndarray):
                red_map = res_maps[k_idx]
                if red_map.shape == getattr(Floc, "shape", None):
                    rm = np.asarray(red_map, dtype=np.int32).ravel()
                    ff = np.asarray(Floc, dtype=float).ravel()
                    mask = rm >= 0
                    if np.any(mask):
                        np.add.at(R, rm[mask], ff[mask])
                else:
                    red_map = None
            else:
                red_map = None

            if red_map is None:
                if not isinstance(gdofs, np.ndarray):
                    gdofs = ker.static_args["gdofs_map"]
                for e in range(gdofs.shape[0]):
                    full = gdofs[e]
                    valid_full = full >= 0
                    if not np.any(valid_full):
                        continue
                    if not has_constraints:
                        rmap = -np.ones_like(full, dtype=int)
                        rmap[valid_full] = self.full_to_red[full[valid_full]]
                        mask = rmap >= 0
                        if not np.any(mask):
                            continue
                        rows = rmap[mask]
                        np.add.at(R, rows, Floc[e][mask])
                    else:
                        full_to_master = getattr(self, "_constr_full_to_master", None)
                        is_slave = getattr(self, "_constr_is_slave", None)
                        slave_map = getattr(self, "_constr_slave_map", None)
                        if (
                            not isinstance(full_to_master, np.ndarray)
                            or not isinstance(is_slave, np.ndarray)
                            or not isinstance(slave_map, dict)
                        ):
                            continue
                        for loc_i, gd in zip(np.nonzero(valid_full)[0].tolist(), full[valid_full].tolist()):
                            gd_i = int(gd)
                            if gd_i < 0 or gd_i >= int(full_to_master.size):
                                continue
                            val = float(Floc[e][int(loc_i)])
                            if not np.isfinite(val) or val == 0.0:
                                continue
                            if bool(is_slave[gd_i]):
                                combo = slave_map.get(gd_i)
                                if combo is None:
                                    continue
                                mcols, wts = combo
                                for mcol, wv in zip(mcols.tolist(), wts.tolist()):
                                    red = int(self.full_to_red[int(mcol)])
                                    if red >= 0:
                                        R[red] += float(wv) * val
                            else:
                                mcol = int(full_to_master[gd_i])
                                if mcol < 0:
                                    continue
                                red = int(self.full_to_red[mcol])
                                if red >= 0:
                                    R[red] += val
            scatter_time = time.perf_counter() - t_scatter
            if profile:
                try:
                    n_ent = int(getattr(Floc, "shape", (0,))[0])
                except Exception:
                    n_ent = 0
                prof_entries.append(
                    ("residual", getattr(ker, "domain", "unknown"), exec_time, scatter_time, n_ent)
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
        self._trace_monitor_installed = False


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
    def _gather_full_iterate(self, funcs: list) -> np.ndarray:
        """
        Gather the current iterate into a full global DOF vector (global index order).

        Do *not* use `np.hstack([f.nodal_values ...])` here: Function/VectorFunction
        storage order follows `._g_dofs`, which is not guaranteed to be sorted by
        global DOF id.
        """
        n = int(self.dh.total_dofs)
        x_full = np.zeros(n, dtype=float)
        for f in funcs:
            g = getattr(f, "_g_dofs", None)
            vals = getattr(f, "nodal_values", None)
            if g is None or vals is None:
                continue
            g_arr = np.asarray(g, dtype=int).ravel()
            if g_arr.size == 0:
                continue
            v_arr = np.asarray(vals, dtype=float).ravel()
            if v_arr.size != g_arr.size:
                raise ValueError(
                    f"Function '{getattr(f, 'name', '<unnamed>')}' has nodal_values size {int(v_arr.size)} "
                    f"but _g_dofs size {int(g_arr.size)}."
                )
            x_full[g_arr] = v_arr
        return x_full

    def _apply_petsc_options(self) -> None:
        """Load self.petsc_options into PETSc.Options so setFromOptions() sees them."""
        opts = PETSc.Options()
        for k, v in self.petsc_options.items():
            key = k if k.startswith("-") else k
            opts[key] = "" if v is None else str(v)

    def _invalidate_petsc_cache(self) -> None:
        """Drop cached PETSc objects so they are rebuilt with the current pattern."""
        self._snes = None
        self._x_red = None
        self._r_red = None
        self._J = None
        self._P = None
        self._XL = None
        self._XU = None
        self._trace_monitor_installed = False
        self._inf_ls_precheck_installed = False
        self._inf_ls_x_trial = None
        self._inf_ls_r_trial = None
    
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

        # If we are in a constraint (hanging-node) mode, bounds on slave DOFs do
        # not map cleanly to box constraints in the reduced/master space.
        # We apply bounds on master DOFs only.
        if getattr(self, "constraints", None) is not None:
            try:
                slaves = np.asarray(self.constraints.slaves, dtype=int)
                if slaves.size:
                    lo_s = lo_full[slaves]
                    hi_s = hi_full[slaves]
                    if np.any(np.isfinite(lo_s)) or np.any(np.isfinite(hi_s)):
                        warnings.warn(
                            "Box bounds were set on slave/hanging DOFs; these bounds are ignored in the "
                            "condensed master-space VI solve. Apply bounds to master DOFs instead.",
                            RuntimeWarning,
                        )
            except Exception:
                pass


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

        # ------------------------------------------------------------------
        # Convergence norm: use infinity norm by default.
        #
        # Rationale:
        # - Our `NewtonParameters.newton_tol` is documented as an infinity-norm
        #   threshold (‖R‖_∞) for the pure-Python Newton implementation.
        # - PETSc SNES defaults to a 2-norm on the residual vector. On large
        #   systems this can be *much* stricter (scales like √N) and may cause
        #   unnecessary max-iteration failures even when every field has
        #   already reached the intended per-DOF tolerance.
        #
        # We first try to set the SNES norm type directly (if supported by
        # petsc4py/PETSc). If not available, we install a custom convergence
        # test that checks ‖F‖_∞ against (atol, rtol).
        # ------------------------------------------------------------------
        self._snes_norm_is_infinity = False
        try:
            self._snes.setNormType(PETSc.NormType.NORM_INFINITY)  # type: ignore[attr-defined]
            self._snes_norm_is_infinity = True
        except Exception:
            fnorm0_box: dict[str, float | None] = {"fnorm0": None}

            # NOTE: petsc4py changed the callback signature across versions.
            # Accept optional args so this works with both:
            #   (snes, it, xnorm, gnorm, fnorm)  and  (snes, it, xnorm)
            def _conv_test(snes, it, xnorm=None, gnorm=None, fnorm=None, *args, **kwargs):  # noqa: ANN001
                try:
                    f_vec, _ctx = snes.getFunction()
                    f_inf = float(f_vec.norm(PETSc.NormType.NORM_INFINITY))
                except Exception:
                    try:
                        f_inf = float(self._r_red.norm(PETSc.NormType.NORM_INFINITY))
                    except Exception:
                        try:
                            f_inf = float(fnorm) if fnorm is not None else float("nan")
                        except Exception:
                            f_inf = float("nan")

                # Reset the "initial norm" at the start of every SNES solve (it=0),
                # even when SNES is reused across time steps.
                if it == 0 and np.isfinite(f_inf):
                    fnorm0_box["fnorm0"] = float(f_inf)

                atol = float(getattr(self.np, "newton_tol", 0.0) or 0.0)
                rtol = float(getattr(self.np, "newton_rtol", 0.0) or 0.0)
                if atol > 0.0 and np.isfinite(f_inf) and f_inf <= atol:
                    return PETSc.SNES.ConvergedReason.CONVERGED_FNORM_ABS
                if (
                    rtol > 0.0
                    and fnorm0_box["fnorm0"] is not None
                    and np.isfinite(f_inf)
                    and f_inf <= rtol * float(fnorm0_box["fnorm0"])
                ):
                    rel_reason = getattr(PETSc.SNES.ConvergedReason, "CONVERGED_FNORM_RELATIVE", None)
                    if rel_reason is None:
                        rel_reason = getattr(PETSc.SNES.ConvergedReason, "CONVERGED_FNORM_REL", None)
                    if rel_reason is None:
                        rel_reason = PETSc.SNES.ConvergedReason.CONVERGED_FNORM_ABS
                    return rel_reason
                return PETSc.SNES.ConvergedReason.ITERATING

            self._snes.setConvergenceTest(_conv_test)
            self._snes_norm_is_infinity = True

        # --- Optional VI bounds (reduced) ---
        if (getattr(self, "_box_lower_full", None) is not None) and (getattr(self, "_box_upper_full", None) is not None):
            if getattr(self, "constraints", None) is not None:
                master_ids = np.asarray(self.constraints.master_ids, dtype=int)
                lo_master = np.asarray(self._box_lower_full, dtype=float)[master_ids]
                hi_master = np.asarray(self._box_upper_full, dtype=float)[master_ids]
                lo_red = lo_master[self.active_dofs].astype(float, copy=True)
                hi_red = hi_master[self.active_dofs].astype(float, copy=True)
            else:
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
        self._snes.setTolerances(
            rtol=float(getattr(self.np, "newton_rtol", 0.0) or 0.0),
            atol=float(getattr(self.np, "newton_tol", 0.0) or 0.0),
            max_it=int(getattr(self.np, "max_newton_iter", 0) or 0),
        )
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
        self._maybe_install_residual_trace_monitor()
        self._maybe_install_inf_ls_precheck()

    def _maybe_install_residual_trace_monitor(self) -> None:
        """
        Install a lightweight SNES monitor to print per-field residual diagnostics.

        This mirrors the env-var based tracing supported by the Python Newton/PDAS
        implementations:
          - PYCUTFEM_NEWTON_TRACE_RES_FIELDS=1  (top-N per-field |R|_∞)
          - PYCUTFEM_RESIDUAL_TRACE=1           (worst DOF residual; optionally coords)
        """
        if self._trace_monitor_installed:
            return
        snes = getattr(self, "_snes", None)
        r_red = getattr(self, "_r_red", None)
        if snes is None or r_red is None:
            return

        want_fields = os.getenv("PYCUTFEM_NEWTON_TRACE_RES_FIELDS", "").lower() in {"1", "true", "yes"}
        want_worst = os.getenv("PYCUTFEM_RESIDUAL_TRACE", "").lower() in {"1", "true", "yes"}
        if not (want_fields or want_worst):
            return

        def _monitor(_snes, it, _norm) -> None:
            try:
                if PETSc.COMM_WORLD.getRank() != 0:
                    return
            except Exception:
                pass

            try:
                R_red = np.asarray(r_red.getArray(readonly=True), dtype=float).ravel()
                R_full = np.asarray(self.restrictor.expand_vec(R_red), dtype=float).ravel()
            except Exception as exc:
                print(f"        [res] snes trace failed: {exc}")
                return

            if want_fields:
                try:
                    field_norms = []
                    for fld in getattr(self.dh, "field_names", []):
                        try:
                            sl = self.dh.get_field_slice(fld)
                        except Exception:
                            continue
                        if sl is None or len(sl) == 0:
                            continue
                        sl = np.asarray(sl, dtype=int).ravel()
                        field_norms.append((float(np.linalg.norm(R_full[sl], ord=np.inf)), str(fld)))
                    field_norms.sort(reverse=True, key=lambda t: t[0])
                    n_show_raw = os.getenv("PYCUTFEM_NEWTON_TRACE_RES_FIELDS_N", "8").strip()
                    try:
                        n_show = max(1, int(n_show_raw))
                    except Exception:
                        n_show = 8
                    items = ", ".join(f"{name}:{val:.2e}" for val, name in field_norms[:n_show])
                    print(f"        [res] it={int(it)} {items}")
                except Exception as exc:
                    print(f"        [res] snes per-field trace failed: {exc}")

            if want_worst:
                try:
                    if R_full.size == 0:
                        return
                    worst_full = int(np.argmax(np.abs(R_full)))
                    worst_val = float(R_full[worst_full])
                    field = None
                    try:
                        field = getattr(self.dh, "_dof_to_node_map", {}).get(worst_full, (None, None))[0]
                    except Exception:
                        field = None
                    fmsg = f", field={field}" if field is not None else ""

                    coords_msg = ""
                    if os.getenv("PYCUTFEM_RESIDUAL_TRACE_COORDS", "").lower() in {"1", "true", "yes"} and field is not None:
                        try:
                            sl = np.asarray(self.dh.get_field_slice(field), dtype=int).ravel()
                            hit = np.nonzero(sl == int(worst_full))[0]
                            if hit.size:
                                coords = np.asarray(self.dh.get_dof_coords(field), dtype=float)[int(hit[0])]
                                if coords.size >= 2:
                                    coords_msg = f", x={coords[0]:.3e}, y={coords[1]:.3e}"
                        except Exception:
                            coords_msg = ""
                    print(f"        [res] it={int(it)} worst_gdof={worst_full} worst_val={worst_val:.3e}{fmsg}{coords_msg}")
                except Exception as exc:
                    print(f"        [res] snes worst trace failed: {exc}")

        snes.setMonitor(_monitor)
        self._trace_monitor_installed = True

    def _maybe_install_inf_ls_precheck(self) -> None:
        """
        Install an infinity-norm based line-search *pre-check* to improve robustness.

        Motivation:
        - PETSc line searches/globalization are typically based on a 2-norm of F.
          For large mixed systems this can accept steps that reduce the 2-norm
          while increasing the per-DOF maximum residual (∞-norm), which is the
          metric we use for our time-step accept/reject diagnostics.
        - In some cases SNES can then terminate with DIVERGED_LINE_SEARCH (-6)
          even though an earlier iterate had a small ∞-norm residual.

        This pre-check dampens the Newton direction `y` so that a trial step
          x_trial = x - α y
        reduces ‖F‖_∞ whenever possible (or at least avoids catastrophic growth).

        Notes:
        - petsc4py's callback signature for `setLineSearchPreCheck` varies; in the
          version used by our fenicsx environment it is `precheck(x: Vec, y: Vec)`.
        - We keep this lightweight (few trial alphas) and only scale the direction.
        """
        if getattr(self, "_inf_ls_precheck_installed", False):
            return
        snes = getattr(self, "_snes", None)
        x_red = getattr(self, "_x_red", None)
        r_red = getattr(self, "_r_red", None)
        if snes is None or x_red is None or r_red is None:
            return
        if not hasattr(snes, "setLineSearchPreCheck"):
            return

        # Work vectors for trial residual evaluations.
        try:
            self._inf_ls_x_trial = x_red.duplicate()
            self._inf_ls_r_trial = r_red.duplicate()
        except Exception:
            return

        # Pre-check tuning via env vars (keep conservative defaults).
        try:
            max_tries = int(os.getenv("PYCUTFEM_SNES_INF_LS_MAX_TRIES", "8"))
        except Exception:
            max_tries = 8
        max_tries = max(1, min(20, int(max_tries)))
        try:
            reduction = float(os.getenv("PYCUTFEM_SNES_INF_LS_REDUCTION", "0.5"))
        except Exception:
            reduction = 0.5
        if not (0.0 < reduction < 1.0):
            reduction = 0.5

        x_trial = self._inf_ls_x_trial
        r_trial = self._inf_ls_r_trial

        def _precheck(x: PETSc.Vec, y: PETSc.Vec, *args, **kwargs):  # noqa: ANN001
            # Guard: if direction is zero, nothing to do.
            try:
                y_inf = float(y.norm(PETSc.NormType.NORM_INFINITY))
            except Exception:
                y_inf = float("nan")
            if not (np.isfinite(y_inf) and y_inf > 0.0):
                return False

            # Current infinity norm (try to reuse SNES' stored function vector).
            try:
                f_vec, _ctx = snes.getFunction()
                norm0 = float(f_vec.norm(PETSc.NormType.NORM_INFINITY))
            except Exception:
                try:
                    snes.computeFunction(x, r_red)
                    norm0 = float(r_red.norm(PETSc.NormType.NORM_INFINITY))
                except Exception:
                    norm0 = float("nan")
            if not np.isfinite(norm0):
                return False

            # First try the full Newton step (α=1). If it already decreases the
            # ∞-norm, do not interfere with PETSc's native globalization (keeps
            # overhead low and avoids overly conservative damping).
            try:
                x_trial.waxpy(-1.0, y, x)
                snes.computeFunction(x_trial, r_trial)
                n_full = float(r_trial.norm(PETSc.NormType.NORM_INFINITY))
            except Exception:
                n_full = float("nan")
            if np.isfinite(n_full) and n_full <= float(norm0):
                return False

            alpha = 1.0
            best_alpha = 0.0
            best_norm = float(norm0)
            smallest_tried: float | None = None
            if np.isfinite(n_full):
                best_alpha = 1.0
                best_norm = float(n_full)

            # Try a short backtracking sequence and pick the best trial.
            # If nothing improves, apply the smallest alpha to avoid blow-ups
            # that can trigger DIVERGED_LINE_SEARCH.
            for _ in range(max_tries - 1):
                try:
                    alpha *= float(reduction)
                    smallest_tried = float(alpha)
                    x_trial.waxpy(-alpha, y, x)  # x_trial = x - α y
                    snes.computeFunction(x_trial, r_trial)
                    n_try = float(r_trial.norm(PETSc.NormType.NORM_INFINITY))
                except Exception:
                    n_try = float("nan")

                if np.isfinite(n_try) and n_try < best_norm:
                    best_norm = float(n_try)
                    best_alpha = float(alpha)

                if alpha <= 0.0:
                    break

            if best_alpha <= 0.0:
                best_alpha = float(smallest_tried) if smallest_tried is not None else float(reduction)

            # Only scale if we are actually damping.
            if best_alpha < 1.0 - 1.0e-14:
                try:
                    y.scale(best_alpha)
                    return True
                except Exception:
                    return False
            return False

        try:
            snes.setLineSearchPreCheck(_precheck)
            self._inf_ls_precheck_installed = True
        except Exception:
            # If this fails, do not block the solve; proceed with PETSc defaults.
            self._inf_ls_precheck_installed = False


    # ------------------------------------------------------------------ #
    # Newton loop
    # ------------------------------------------------------------------ #
    def _newton_loop(self, funcs, prev_funcs, aux_funcs, bcs_now):
        dh = self.dh
        self._current_bcs = bcs_now
        comm = PETSc.COMM_WORLD

        pattern_was_stale = getattr(self, "_pattern_stale", True) or not hasattr(self, "_csr_indptr")

        # Apply BCs to the current iterate and form the reduced initial guess
        dh.apply_bcs(bcs_now, *funcs)
        if getattr(self, "constraints", None) is not None:
            self._enforce_constraints_on_functions(funcs)
        x0_full = self._gather_full_iterate(funcs)

        # Assemble once to prune any decoupled rows/cols (matches Python Newton behavior).
        current: Dict[str, "Function"] = {f.name: f for f in funcs}
        current.update({f.name: f for f in prev_funcs})
        if aux_funcs:
            current.update(aux_funcs)
        ndof_eff = getattr(self.restrictor, "full_size", dh.total_dofs)
        A_red, R_red = self._assemble_system_reduced(current, need_matrix=True)
        A_red, R_red, pruned, _extra_asm = self._prune_decoupled_rows_cols(
            current, A_red, R_red, ndof_eff
        )
        if pattern_was_stale or pruned:
            self._invalidate_petsc_cache()
        # Ensure we have a CSR sparsity pattern for PETSc preallocation even on the
        # Python backend (where we don't build the pattern from kernels).
        try:
            if A_red is not None:
                if not sp.isspmatrix_csr(A_red):
                    A_red = A_red.tocsr()
                self._csr_indptr = A_red.indptr.copy()
                self._csr_indices = A_red.indices.copy()
                self._pattern_stale = False
        except Exception:
            pass

        if getattr(self, "constraints", None) is not None:
            master_ids = np.asarray(self.constraints.master_ids, dtype=int)
            x0_master = x0_full[master_ids]
            x0_red = x0_master[self.active_dofs].copy()
        else:
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

        x0_red_guess = np.asarray(x0_red, dtype=float).copy()
        self._x_red.setValues(idx, x0_red, addv=PETSc.InsertMode.INSERT_VALUES)
        self._x_red.assemblyBegin(); self._x_red.assemblyEnd()

        # Keep SNES tolerances in sync with `NewtonParameters`. This allows runtime
        # adjustments (e.g. temporary max-it boosts) via callbacks without needing
        # to recreate the SNES object.
        try:
            self._snes.setTolerances(
                rtol=float(getattr(self.np, "newton_rtol", 0.0) or 0.0),
                atol=float(getattr(self.np, "newton_tol", 0.0) or 0.0),
                max_it=int(getattr(self.np, "max_newton_iter", 0) or 0),
            )
        except Exception:
            pass

        # Solve the nonlinear system on the reduced space
        self._snes.solve(None, self._x_red)
        reason_snes = int(self._snes.getConvergedReason())
        reason = int(reason_snes)
        converged = reason_snes > 0
        accepted = False
        n_iters = int(self._snes.getIterationNumber())
        # Re-evaluate the residual at the *returned* iterate. In particular for
        # DIVERGED_LINE_SEARCH (-6), PETSc can exit with a solution vector that
        # does not match the last stored function vector.
        fnorm = float("nan")
        fnorm_snes = float("nan")
        try:
            self._snes.computeFunction(self._x_red, self._r_red)
            fnorm = float(self._r_red.norm(PETSc.NormType.NORM_INFINITY))
            fnorm_snes = float(self._r_red.norm(PETSc.NormType.NORM_2))
        except Exception:
            try:
                fnorm_snes = float(self._snes.getFunctionNorm())
            except Exception:
                fnorm_snes = float("nan")
            try:
                f_vec, _ctx = self._snes.getFunction()
                fnorm = float(f_vec.norm(PETSc.NormType.NORM_INFINITY))
            except Exception:
                fnorm = float(fnorm_snes)

        # PETSc may report a nonconverged reason (e.g. DIVERGED_MAX_IT) even when the
        # returned iterate already satisfies our intended infinity-norm tolerance.
        # Treat such cases as converged by our criterion (does not relax tolerances).
        atol = float(getattr(self.np, "newton_tol", 0.0) or 0.0)
        rtol = float(getattr(self.np, "newton_rtol", 0.0) or 0.0)
        tol_met_abs = bool(atol > 0.0 and np.isfinite(fnorm) and float(fnorm) <= float(atol))
        tol_met_rel = False  # rtol currently handled by SNES convergence tests (if enabled)
        if not converged and (tol_met_abs or tol_met_rel):
            try:
                conv_abs = int(getattr(PETSc.SNES.ConvergedReason, "CONVERGED_FNORM_ABS", 2))
            except Exception:
                conv_abs = 2
            print(
                f"    [warn] SNES returned reason={reason_snes} after {n_iters} iters, but ‖F‖_inf={fnorm:.3e} ≤ atol={atol:.3e}; "
                "treating as converged."
            )
            converged = True
            reason = int(conv_abs)

        # Expose convergence diagnostics to the outer time-stepper.
        self._last_nonlinear_norm = float(fnorm)
        self._last_nonlinear_norm_label = "‖F‖_inf"
        self._last_nonlinear_reason = int(reason)
        if not converged:
            # Ensure norms correspond to the returned iterate.
            try:
                self._snes.computeFunction(self._x_red, self._r_red)
                fnorm = float(self._r_red.norm(PETSc.NormType.NORM_INFINITY))
                fnorm_snes = float(self._r_red.norm(PETSc.NormType.NORM_2))
            except Exception:
                pass
            accept_factor = float(getattr(self.np, "accept_nonconverged_atol_factor", 0.0) or 0.0)
            accept_step = bool(
                accept_factor > 0.0
                and atol > 0.0
                and np.isfinite(fnorm)
                and fnorm <= accept_factor * atol
            )
            print(
                f"    [warn] SNES did not converge (reason={reason_snes}, iters={n_iters}, ‖F‖_inf={fnorm:.3e}, ‖F‖_2={fnorm_snes:.3e}). "
                "Continuing with best iterate."
            )
            if accept_step:
                print(
                    f"    [warn] Accepting SNES best iterate since ‖F‖_inf={fnorm:.3e} ≤ {accept_factor:g}·atol "
                    f"(atol={atol:.3e})."
                )
                converged = True
                accepted = True

        # Write back the absolute solution (not an increment)
        x_fin = self._x_red.getArray(readonly=True)
        try:
            dx_inf = float(np.linalg.norm(np.asarray(x_fin, dtype=float) - x0_red_guess, ord=np.inf))
        except Exception:
            dx_inf = float("nan")
        self._last_nonlinear_update_inf = float(dx_inf)
        self._last_nonlinear_update_label = "Δx_snes∞"
        self._last_nonlinear_accepted = bool(accepted)
        if getattr(self, "constraints", None) is not None:
            master_ids = np.asarray(self.constraints.master_ids, dtype=int)
            new_master = x0_full[master_ids].copy()
            new_master[self.active_dofs] = np.asarray(x_fin, dtype=float)
            new_full = self.constraints.prolong(new_master)
        else:
            new_full = x0_full.copy()
            new_full[self.active_dofs] = np.asarray(x_fin, dtype=float)
        for f in funcs:
            g = f._g_dofs
            f.set_nodal_values(g, new_full[g])
        dh.apply_bcs(bcs_now, *funcs)
        if getattr(self, "constraints", None) is not None:
            self._enforce_constraints_on_functions(funcs)

        prev_vals = np.hstack([f.nodal_values for f in prev_funcs])
        delta = np.hstack([f.nodal_values for f in funcs]) - prev_vals
        return delta, converged, n_iters

    # ------------------------------------------------------------------ #
    # SNES callbacks: residual and Jacobian on the reduced space
    # ------------------------------------------------------------------ #
    def _eval_residual_reduced(self, snes, x_red: PETSc.Vec, r_red: PETSc.Vec):
        ctx = self._petsc_ctx

        # Lift reduced iterate to full space on a fresh copy, then apply BCs
        x_fin = np.asarray(x_red.getArray(readonly=True), dtype=float)
        if getattr(self, "constraints", None) is not None:
            master_ids = np.asarray(self.constraints.master_ids, dtype=int)
            master = ctx["x0_full"][master_ids].copy()
            master[self.active_dofs] = x_fin
            x_full = self.constraints.prolong(master)
        else:
            x_full = ctx["x0_full"].copy()
            x_full[self.active_dofs] = x_fin

        # IMPORTANT: the pure-Python backend assembles forms using the Function
        # objects that were captured when the UFL expression was built. In that
        # mode, assembling on copies would ignore updated coefficients and
        # produce a constant residual, which breaks SNES line-search.
        res_funcs = ctx["funcs"] if self.backend == "python" else [f.copy() for f in ctx["funcs"]]
        for f in res_funcs:
            g = f._g_dofs
            f.set_nodal_values(g, x_full[g])
        self.dh.apply_bcs(ctx["bcs_now"], *res_funcs)
        if getattr(self, "constraints", None) is not None:
            self._enforce_constraints_on_functions(res_funcs)

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

        x_fin = np.asarray(x_red.getArray(readonly=True), dtype=float)
        if getattr(self, "constraints", None) is not None:
            master_ids = np.asarray(self.constraints.master_ids, dtype=int)
            master = ctx["x0_full"][master_ids].copy()
            master[self.active_dofs] = x_fin
            x_full = self.constraints.prolong(master)
        else:
            x_full = ctx["x0_full"].copy()
            x_full[self.active_dofs] = x_fin

        jac_funcs = ctx["funcs"] if self.backend == "python" else [f.copy() for f in ctx["funcs"]]
        for f in jac_funcs:
            g = f._g_dofs
            f.set_nodal_values(g, x_full[g])
        self.dh.apply_bcs(ctx["bcs_now"], *jac_funcs)
        if getattr(self, "constraints", None) is not None:
            self._enforce_constraints_on_functions(jac_funcs)

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
        X = dh.get_all_dof_coords()                  # shape (total_dofs, 2)
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
        self.set_box_bounds(lower=lo_full, upper=hi_full)     # uses existing API


# --------------------------------------------------------------------------
#  Internal PDAS / Semismooth Newton solver for box VIs
# --------------------------------------------------------------------------

class PdasNewtonSolver(NewtonSolver):
    """
    Internal Primal-Dual Active Set (PDAS) / semismooth Newton solver for
    box-constrained variational inequalities:

        find x in [lo, hi] such that 0 in F(x) + N_[lo,hi](x)

    This solver reuses pycutfem's reduced assembly pipeline and PETSc-backed
    linear solves (via LinearSolverParameters.backend='petsc').
    """

    def __init__(self, *args, vi_params: VIParameters = VIParameters(), **kwargs):
        super().__init__(*args, **kwargs)
        self.vi_params = vi_params
        self._box_lower_full: np.ndarray | None = None
        self._box_upper_full: np.ndarray | None = None
        self._vi_prev_state: np.ndarray | None = None
        self._vi_pending_state: np.ndarray | None = None
        self._vi_pending_count: np.ndarray | None = None
        self._vi_lambda_current: float = float(getattr(self.vi_params, "inactive_reg_lambda0", 0.0) or 0.0)
        self._vi_lm_lambda_current: float = float(getattr(self.vi_params, "unconstrained_lm_lambda0", 1.0e-4) or 1.0e-4)
        self._vi_last_metrics: dict[str, float] | None = None
        self._vi_last_retry_count: int = 0
        self._vi_red_field_names = self._build_vi_reduced_field_names()

    def _build_vi_reduced_field_names(self) -> np.ndarray:
        names: list[str] = []
        full_to_field = getattr(self.dh, "_dof_to_node_map", {})
        master_ids = None
        if getattr(self, "constraints", None) is not None:
            try:
                master_ids = np.asarray(self.constraints.master_ids, dtype=int)
            except Exception:
                master_ids = None
        for rid in np.asarray(self.active_dofs, dtype=int).tolist():
            full_id = int(rid)
            if master_ids is not None and 0 <= full_id < int(master_ids.size):
                full_id = int(master_ids[full_id])
            names.append(str(full_to_field.get(full_id, ("", None))[0] or ""))
        return np.asarray(names, dtype=object)

    def _vi_c_vector(self, A_red: sp.csr_matrix, lo_red: np.ndarray, hi_red: np.ndarray) -> np.ndarray:
        bounded = np.isfinite(lo_red) | np.isfinite(hi_red)
        c_base = float(getattr(self.vi_params, "c", 1.0) or 1.0)
        c_red = np.full(lo_red.shape, c_base, dtype=float)

        # Auto-scale by field from the current Jacobian diagonal when requested.
        auto_c = (not np.isfinite(c_base)) or (c_base <= 0.0)
        diag_full = np.abs(np.asarray(A_red.diagonal(), dtype=float))
        diag_full = np.where(np.isfinite(diag_full) & (diag_full > 0.0), diag_full, np.nan)

        field_vals: dict[str, float] = {}
        if auto_c and np.any(bounded):
            for fld in np.unique(self._vi_red_field_names[bounded]):
                if not str(fld):
                    continue
                mask = bounded & (self._vi_red_field_names == fld)
                vals = diag_full[mask]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    field_vals[str(fld)] = float(np.median(vals))
            if field_vals:
                finite_vals = np.asarray(list(field_vals.values()), dtype=float)
                finite_vals = finite_vals[np.isfinite(finite_vals) & (finite_vals > 0.0)]
                self.vi_params.c = float(np.median(finite_vals)) if finite_vals.size else 1.0
            else:
                self.vi_params.c = 1.0
            c_red[:] = float(getattr(self.vi_params, "c", 1.0) or 1.0)

        # Explicit per-field overrides always win.
        for fld, c_val in dict(getattr(self.vi_params, "c_by_field", {}) or {}).items():
            try:
                c_f = float(c_val)
            except Exception:
                continue
            if np.isfinite(c_f) and c_f > 0.0:
                field_vals[str(fld)] = c_f

        for fld, c_f in field_vals.items():
            c_red[self._vi_red_field_names == fld] = float(c_f)

        c_red = np.where(np.isfinite(c_red) & (c_red > 0.0), c_red, 1.0)
        self._vi_c_field_current = dict(field_vals)
        return c_red

    def _vi_state_from_sets(self, act_lo: np.ndarray, act_hi: np.ndarray) -> np.ndarray:
        state = np.zeros(act_lo.shape, dtype=np.int8)
        state[np.asarray(act_lo, dtype=bool)] = 1
        state[np.asarray(act_hi, dtype=bool)] = -1
        return state

    def _vi_apply_state_persistence(self, proposed_state: np.ndarray) -> np.ndarray:
        persist = max(0, int(getattr(self.vi_params, "active_set_persistence", 0) or 0))
        if persist <= 0:
            self._vi_pending_state = proposed_state.copy()
            self._vi_pending_count = np.zeros(proposed_state.shape, dtype=np.int16)
            return proposed_state

        prev = getattr(self, "_vi_prev_state", None)
        if prev is None or prev.shape != proposed_state.shape:
            self._vi_pending_state = proposed_state.copy()
            self._vi_pending_count = np.zeros(proposed_state.shape, dtype=np.int16)
            return proposed_state

        pending = getattr(self, "_vi_pending_state", None)
        counts = getattr(self, "_vi_pending_count", None)
        if pending is None or counts is None or pending.shape != proposed_state.shape or counts.shape != proposed_state.shape:
            pending = prev.copy()
            counts = np.zeros(proposed_state.shape, dtype=np.int16)

        changed = proposed_state != prev
        same_pending = proposed_state == pending
        counts = np.where(changed & same_pending, counts + 1, np.where(changed, 1, 0)).astype(np.int16, copy=False)
        pending = np.where(changed, proposed_state, prev).astype(np.int8, copy=False)

        accepted = prev.copy()
        promote = changed & (counts > persist)
        accepted[promote] = proposed_state[promote]

        self._vi_pending_state = pending.astype(np.int8, copy=False)
        self._vi_pending_count = counts.astype(np.int16, copy=False)
        return accepted.astype(np.int8, copy=False)

    def _vi_metrics(
        self,
        x_red: np.ndarray,
        R_red: np.ndarray,
        lo_red: np.ndarray,
        hi_red: np.ndarray,
        act_lo: np.ndarray,
        act_hi: np.ndarray,
        G_red: np.ndarray,
    ) -> dict[str, float]:
        active = np.asarray(act_lo | act_hi, dtype=bool)
        inactive = ~active
        gap = np.zeros_like(x_red, dtype=float)
        gap[act_lo] = np.abs(x_red[act_lo] - lo_red[act_lo])
        gap[act_hi] = np.abs(hi_red[act_hi] - x_red[act_hi])
        R_inactive = np.asarray(R_red[inactive], dtype=float)
        gap_active = np.asarray(gap[active], dtype=float)
        res_inf = float(np.linalg.norm(R_inactive, ord=np.inf)) if R_inactive.size else 0.0
        gap_inf = float(np.linalg.norm(gap_active, ord=np.inf)) if gap_active.size else 0.0
        res_l2 = float(R_inactive @ R_inactive) if R_inactive.size else 0.0
        gap_l2 = float(gap_active @ gap_active) if gap_active.size else 0.0
        merit = (
            0.5 * float(getattr(self.vi_params, "merit_residual_weight", 1.0) or 1.0) * res_l2
            + 0.5 * float(getattr(self.vi_params, "merit_gap_weight", 1.0) or 1.0) * gap_l2
        )
        return {
            "G_inf": float(np.linalg.norm(np.asarray(G_red, dtype=float), ord=np.inf)),
            "G_half": 0.5 * float(np.asarray(G_red, dtype=float) @ np.asarray(G_red, dtype=float)),
            "inactive_res_inf": res_inf,
            "active_gap_inf": gap_inf,
            "inactive_res_l2_sq": res_l2,
            "active_gap_l2_sq": gap_l2,
            "merit": merit,
        }

    def _vi_apply_inactive_regularization(
        self,
        A_mod: sp.csr_matrix,
        A_base: sp.csr_matrix,
        active: np.ndarray,
        lam: float,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        inactive = ~np.asarray(active, dtype=bool)
        if lam <= 0.0 or not np.any(inactive):
            return A_mod, np.zeros(active.shape, dtype=float)

        diag = np.abs(np.asarray(A_base.diagonal(), dtype=float))
        floor = float(getattr(self.vi_params, "inactive_reg_min_diag", 1.0e-12) or 1.0e-12)
        diag = np.where(np.isfinite(diag) & (diag > floor), diag, floor)
        add = np.zeros(active.shape, dtype=float)
        add[inactive] = float(lam) * diag[inactive]

        A_reg = A_mod.tolil(copy=True)
        for idx in np.flatnonzero(inactive).tolist():
            A_reg[int(idx), int(idx)] = float(A_reg[int(idx), int(idx)]) + float(add[int(idx)])
        return A_reg.tocsr(), add

    def _vi_build_unconstrained_lm_system(
        self,
        A_red: sp.csr_matrix,
        R_red: np.ndarray,
        lam: float,
    ) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        rhs = -np.asarray(R_red, dtype=float)
        return self._vi_build_lm_system(A_red, rhs, lam)

    def _vi_build_lm_system(
        self,
        A_red: sp.csr_matrix,
        rhs_red: np.ndarray,
        lam: float,
    ) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        if not sp.isspmatrix_csr(A_red):
            A_red = A_red.tocsr()
        diag = np.abs(np.asarray(A_red.diagonal(), dtype=float))
        floor = float(getattr(self.vi_params, "unconstrained_lm_min_diag", 1.0e-12) or 1.0e-12)
        diag = np.where(np.isfinite(diag) & (diag > floor), diag, floor)
        H = A_red.copy().tocsr()
        if lam > 0.0:
            H = H + sp.diags(float(lam) * diag, format="csr")
        rhs = np.asarray(rhs_red, dtype=float).copy()
        return H, rhs, diag

    def _vi_unconstrained_lm_model(
        self,
        R_red: np.ndarray,
        A_red: sp.csr_matrix,
        step: np.ndarray,
        lam: float,
        diag: np.ndarray,
        *,
        phi_try: float | None = None,
    ) -> tuple[float, float, float, float]:
        phi0 = 0.5 * float(np.asarray(R_red, dtype=float) @ np.asarray(R_red, dtype=float))
        lin_res = np.asarray(R_red, dtype=float) + np.asarray(A_red @ step, dtype=float)
        model_phi = 0.5 * float(lin_res @ lin_res)
        # IMPORTANT:
        # The current unconstrained "LM" branch solves a shifted Newton system
        #   (A + λ D) s = -R
        # for speed, not the classical least-squares LM normal equations.
        # Therefore `phi0 - model_phi` is *not* a reliable predicted reduction
        # for the accepted step, and using it in rho can produce ±inf / nonsense.
        #
        # We still return the linearized merit model value for diagnostics, but
        # the acceptance ratio uses the *actual* relative reduction instead.
        rho = float("nan")
        if phi_try is not None:
            actual = phi0 - float(phi_try)
            rho = actual / max(phi0, 1.0e-30)
        pred_model = phi0 - model_phi
        return phi0, model_phi, pred_model, rho

    def _vi_strong_active_mask(
        self,
        x_red: np.ndarray,
        R_red: np.ndarray,
        lo_red: np.ndarray,
        hi_red: np.ndarray,
        act_lo: np.ndarray,
        act_hi: np.ndarray,
        c_red: np.ndarray,
    ) -> np.ndarray:
        tol0 = float(getattr(self.vi_params, "active_tol", 0.0) or 0.0)
        enter_tol = getattr(self.vi_params, "enter_tol", None)
        leave_tol = getattr(self.vi_params, "leave_tol", None)
        enter_tol = tol0 if enter_tol is None else float(enter_tol)
        leave_tol = enter_tol if leave_tol is None else float(leave_tol)
        base_tol = max(abs(enter_tol), abs(leave_tol), 1.0e-16)
        factor = float(getattr(self.vi_params, "active_step_strong_factor", 5.0) or 5.0)
        gap_lo = x_red - lo_red
        gap_hi = hi_red - x_red
        ind_lo = R_red - c_red * gap_lo
        ind_hi = R_red + c_red * gap_hi
        strong_lo = np.asarray(act_lo, dtype=bool) & (ind_lo >= factor * base_tol)
        strong_hi = np.asarray(act_hi, dtype=bool) & ((-ind_hi) >= factor * base_tol)
        return np.asarray(strong_lo | strong_hi, dtype=bool)

    def _vi_apply_active_step_policy(
        self,
        *,
        alpha: float,
        S_red: np.ndarray,
        active_mask: np.ndarray | None,
        strong_active_mask: np.ndarray | None,
        active_delta_ref: int,
    ) -> tuple[np.ndarray, bool]:
        step = alpha * np.asarray(S_red, dtype=float)
        if active_mask is None or active_mask.size != step.size:
            return step, False
        active_mask = np.asarray(active_mask, dtype=bool)
        if not np.any(active_mask):
            return step, False

        trigger = int(getattr(self.vi_params, "active_step_delta_active_trigger", 0) or 0)
        soft_alpha = float(getattr(self.vi_params, "active_step_soft_alpha", 1.0) or 1.0)
        use_soft = trigger > 0 and active_delta_ref >= trigger and soft_alpha < 1.0
        if not use_soft:
            step = step.copy()
            step[active_mask] = np.asarray(S_red, dtype=float)[active_mask]
            return step, False

        strong_active_mask = np.zeros(step.shape, dtype=bool) if strong_active_mask is None else np.asarray(strong_active_mask, dtype=bool)
        if strong_active_mask.shape != step.shape:
            strong_active_mask = np.zeros(step.shape, dtype=bool)
        weak_active_mask = active_mask & (~strong_active_mask)
        weak_alpha = min(max(alpha, 0.0), soft_alpha)
        step = step.copy()
        step[strong_active_mask] = np.asarray(S_red, dtype=float)[strong_active_mask]
        step[weak_active_mask] = weak_alpha * np.asarray(S_red, dtype=float)[weak_active_mask]
        return step, True

    def set_box_bounds(self, lower=None, upper=None, by_field: dict | None = None) -> None:
        """
        Define componentwise bounds on the full DOF vector.

        Parameters
        ----------
        lower/upper:
            Scalar or full-size numpy arrays (total_dofs). Missing bound -> ±inf.
        by_field:
            Dict like {'ux': (lo, hi), 'uy': (lo, hi), 'p': (None, None)} applied on
            the corresponding field slices.
        """
        n = int(self.dh.total_dofs)
        lo_full = np.full(n, -np.inf, dtype=float)
        hi_full = np.full(n, np.inf, dtype=float)

        if isinstance(lower, (int, float)):
            lo_full[:] = float(lower)
        elif isinstance(lower, np.ndarray):
            if int(lower.size) != n:
                raise ValueError(f"lower bound array has size {int(lower.size)}, expected {n}.")
            lo_full[:] = np.asarray(lower, dtype=float)

        if isinstance(upper, (int, float)):
            hi_full[:] = float(upper)
        elif isinstance(upper, np.ndarray):
            if int(upper.size) != n:
                raise ValueError(f"upper bound array has size {int(upper.size)}, expected {n}.")
            hi_full[:] = np.asarray(upper, dtype=float)

        if by_field:
            for name, (lo, hi) in by_field.items():
                sl = self.dh.get_field_slice(name)
                if sl is None:
                    raise KeyError(f"Unknown field '{name}'.")
                if lo is not None:
                    lo_full[sl] = float(lo)
                if hi is not None:
                    hi_full[sl] = float(hi)

        # sanitize NaNs to ±inf
        lo_full = np.where(np.isfinite(lo_full), lo_full, -np.inf)
        hi_full = np.where(np.isfinite(hi_full), hi_full, np.inf)

        if np.any(lo_full > hi_full):
            bad = np.flatnonzero(lo_full > hi_full)
            raise ValueError(f"Inconsistent bounds: lower > upper at {int(bad.size)} DOFs (e.g. {int(bad[0])}).")

        self._box_lower_full = lo_full
        self._box_upper_full = hi_full

        if getattr(self, "constraints", None) is not None:
            try:
                slaves = np.asarray(self.constraints.slaves, dtype=int)
                if slaves.size:
                    lo_s = lo_full[slaves]
                    hi_s = hi_full[slaves]
                    if np.any(np.isfinite(lo_s)) or np.any(np.isfinite(hi_s)):
                        warnings.warn(
                            "Box bounds were set on slave/hanging DOFs; these bounds are ignored in the "
                            "condensed master-space VI solve. Apply bounds to master DOFs instead.",
                            RuntimeWarning,
                        )
            except Exception:
                pass

    def _bounds_reduced(self) -> tuple[np.ndarray, np.ndarray]:
        lo_full = getattr(self, "_box_lower_full", None)
        hi_full = getattr(self, "_box_upper_full", None)
        if lo_full is None or hi_full is None:
            raise RuntimeError("Box bounds are not set. Call set_box_bounds() before solving.")
        if getattr(self, "constraints", None) is not None:
            master_ids = np.asarray(self.constraints.master_ids, dtype=int)
            lo_master = np.asarray(lo_full, dtype=float)[master_ids]
            hi_master = np.asarray(hi_full, dtype=float)[master_ids]
            return lo_master[self.active_dofs].copy(), hi_master[self.active_dofs].copy()
        return np.asarray(lo_full, dtype=float)[self.active_dofs].copy(), np.asarray(hi_full, dtype=float)[self.active_dofs].copy()

    def _gather_full_iterate(self, funcs: list) -> np.ndarray:
        """
        Gather the current iterate into a full global DOF vector in *global index*
        order.

        IMPORTANT: Function/VectorFunction nodal storage is in their local order
        (the order of `._g_dofs`), which is not necessarily sorted by global DOF id.
        Do not use `np.hstack([f.nodal_values ...])` when you need a global vector.
        """
        n = int(self.dh.total_dofs)
        x_full = np.zeros(n, dtype=float)
        for f in funcs:
            g = getattr(f, "_g_dofs", None)
            vals = getattr(f, "nodal_values", None)
            if g is None or vals is None:
                continue
            g_arr = np.asarray(g, dtype=int).ravel()
            if g_arr.size == 0:
                continue
            v_arr = np.asarray(vals, dtype=float).ravel()
            if v_arr.size != g_arr.size:
                raise ValueError(
                    f"Function '{getattr(f, 'name', '<unnamed>')}' has nodal_values size {int(v_arr.size)} "
                    f"but _g_dofs size {int(g_arr.size)}."
                )
            x_full[g_arr] = v_arr
        return x_full

    def _pack_reduced_iterate(self, funcs: list) -> np.ndarray:
        x_full = self._gather_full_iterate(funcs)
        if getattr(self, "constraints", None) is not None:
            master_ids = np.asarray(self.constraints.master_ids, dtype=int)
            x_master = np.asarray(x_full, dtype=float)[master_ids]
            return x_master[self.active_dofs].copy()
        return np.asarray(x_full, dtype=float)[self.active_dofs].copy()

    def _project_funcs_to_bounds(self, funcs: list, bcs_now, lo_red: np.ndarray, hi_red: np.ndarray) -> None:
        x_red = self._pack_reduced_iterate(funcs)
        x_proj = np.minimum(np.maximum(x_red, lo_red), hi_red)
        d_red = x_proj - x_red
        if not np.any(d_red):
            return
        dU_full = self.restrictor.expand_vec(d_red)
        self.dh.add_to_functions(dU_full, funcs)
        self.dh.apply_bcs(bcs_now, *funcs)
        if getattr(self, "constraints", None) is not None:
            self._enforce_constraints_on_functions(funcs)

    def _pdas_residual_and_sets(
        self,
        x_red: np.ndarray,
        R_red: np.ndarray,
        lo_red: np.ndarray,
        hi_red: np.ndarray,
        c_red: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the semismooth residual for the bound VI and the corresponding
        PDAS active sets.

        Residual (componentwise):
          - inactive:    F(x) = 0
          - lower-active: x - lo = 0
          - upper-active: x - hi = 0

        Active set prediction:
          - lower-active if   F(x) - c(x-lo) >  tol
          - upper-active if   F(x) + c(hi-x) < -tol
        """
        tol0 = float(getattr(self.vi_params, "active_tol", 0.0) or 0.0)
        enter_tol = getattr(self.vi_params, "enter_tol", None)
        leave_tol = getattr(self.vi_params, "leave_tol", None)
        enter_tol = tol0 if enter_tol is None else float(enter_tol)
        leave_tol = enter_tol if leave_tol is None else float(leave_tol)
        if c_red is None:
            c_red = np.full(x_red.shape, float(getattr(self.vi_params, "c", 1.0) or 1.0), dtype=float)
        c_red = np.where(np.isfinite(c_red) & (c_red > 0.0), c_red, 1.0)

        lo_f = np.isfinite(lo_red)
        hi_f = np.isfinite(hi_red)
        gap_lo = x_red - lo_red
        gap_hi = hi_red - x_red

        ind_lo = R_red - c_red * gap_lo
        ind_hi = R_red + c_red * gap_hi

        prev_state = getattr(self, "_vi_prev_state", None)
        prev_lo = prev_state is not None and prev_state.shape == x_red.shape
        if prev_lo:
            prev_lo_mask = prev_state == 1
            prev_hi_mask = prev_state == -1
        else:
            prev_lo_mask = np.zeros(x_red.shape, dtype=bool)
            prev_hi_mask = np.zeros(x_red.shape, dtype=bool)

        # Candidate actives (masked to finite bounds) with hysteresis.
        enter_lo = lo_f & (ind_lo > enter_tol)
        enter_hi = hi_f & (ind_hi < -enter_tol)
        keep_lo = prev_lo_mask & lo_f & (ind_lo > -leave_tol)
        keep_hi = prev_hi_mask & hi_f & (ind_hi < leave_tol)

        act_lo = enter_lo | keep_lo
        act_hi = enter_hi | keep_hi

        # Resolve overlaps (should not happen for consistent bounds, but be safe).
        both = act_lo & act_hi
        if np.any(both):
            # Prefer the closer bound in terms of current gap.
            choose_lo = gap_lo[both] <= gap_hi[both]
            act_lo[both] &= choose_lo
            act_hi[both] &= ~choose_lo

        proposed_state = self._vi_state_from_sets(act_lo, act_hi)
        state = self._vi_apply_state_persistence(proposed_state)
        act_lo = state == 1
        act_hi = state == -1

        # Semismooth residual.
        G = np.asarray(R_red, dtype=float).copy()
        G[act_lo] = gap_lo[act_lo]
        G[act_hi] = -gap_hi[act_hi]  # x - hi
        self._vi_prev_state = state.copy()
        return G, act_lo, act_hi

    def _apply_identity_rows(self, A: sp.csr_matrix, rows: np.ndarray) -> sp.csr_matrix:
        if rows.size == 0:
            return A
        if not sp.isspmatrix_csr(A):
            A = A.tocsr()
        A_mod = A.copy()
        indptr = A_mod.indptr
        indices = A_mod.indices
        data = A_mod.data
        missing_diag = False
        for r in rows.tolist():
            rr = int(r)
            start = int(indptr[rr])
            end = int(indptr[rr + 1])
            if end <= start:
                missing_diag = True
                continue
            data[start:end] = 0.0
            hit = np.nonzero(indices[start:end] == rr)[0]
            if hit.size:
                data[start + int(hit[0])] = 1.0
            else:
                missing_diag = True
        if not missing_diag:
            return A_mod

        # Fallback: allow pattern change for rows missing a diagonal entry.
        A_lil = A.tolil(copy=True)
        for r in rows.tolist():
            rr = int(r)
            A_lil.rows[rr] = [rr]
            A_lil.data[rr] = [1.0]
        return A_lil.tocsr()

    def _line_search_vi_reduced(
        self,
        *,
        x0_red: np.ndarray,
        R0_red: np.ndarray,
        G0: np.ndarray,
        act_lo0: np.ndarray,
        act_hi0: np.ndarray,
        S_red: np.ndarray,
        c_red: np.ndarray,
        strong_active_mask: np.ndarray | None = None,
        active_delta_ref: int = 0,
        active_mask: np.ndarray | None = None,
        funcs: list,
        coeffs: dict,
        bcs_now,
        lo_red: np.ndarray,
        hi_red: np.ndarray,
    ) -> np.ndarray:
        dh = self.dh
        np_ = self.np
        mode = str(getattr(np_, "ls_mode", "armijo") or "armijo").lower()
        c1 = float(getattr(np_, "ls_c1", 1.0e-4))
        merit_mode = str(getattr(self.vi_params, "merit_mode", "split") or "split").lower()
        trace_ls = os.getenv("PYCUTFEM_VI_TRACE_LS", "").lower() in {"1", "true", "yes"}

        snap = [f.nodal_values.copy() for f in funcs]
        metrics0 = self._vi_metrics(x0_red, R0_red, lo_red, hi_red, act_lo0, act_hi0, G0)
        norm0 = float(metrics0["G_inf"])
        phi0 = float(metrics0["G_half"])
        merit0 = norm0 if merit_mode == "legacy" and mode == "dealii" else (phi0 if merit_mode == "legacy" else float(metrics0["merit"]))
        gap0 = float(metrics0["active_gap_inf"])
        res0 = float(metrics0["inactive_res_inf"])
        best_alpha = 0.0
        best_merit = merit0
        best_metrics = dict(metrics0)

        def _trial(alpha: float) -> tuple[dict[str, float], np.ndarray]:
            for f, buf in zip(funcs, snap):
                f.nodal_values[:] = buf
            step, soft_active = self._vi_apply_active_step_policy(
                alpha=alpha,
                S_red=S_red,
                active_mask=active_mask,
                strong_active_mask=strong_active_mask,
                active_delta_ref=active_delta_ref,
            )
            dU_full = self.restrictor.expand_vec(step)
            dh.add_to_functions(dU_full, funcs)
            dh.apply_bcs(bcs_now, *funcs)
            if getattr(self, "constraints", None) is not None:
                self._enforce_constraints_on_functions(funcs)
            if bool(getattr(self.vi_params, "project_each_iteration", True)):
                self._project_funcs_to_bounds(funcs, bcs_now, lo_red, hi_red)

            _, R_try = self._assemble_system_reduced(coeffs, need_matrix=False)
            x_try = self._pack_reduced_iterate(funcs)
            prev_state_save = None if self._vi_prev_state is None else self._vi_prev_state.copy()
            prev_pending = None if self._vi_pending_state is None else self._vi_pending_state.copy()
            prev_counts = None if self._vi_pending_count is None else self._vi_pending_count.copy()
            G_try, act_lo_try, act_hi_try = self._pdas_residual_and_sets(x_try, R_try, lo_red, hi_red, c_red=c_red)
            metrics = self._vi_metrics(x_try, R_try, lo_red, hi_red, act_lo_try, act_hi_try, G_try)
            metrics["delta_active"] = float(
                np.count_nonzero(self._vi_state_from_sets(act_lo_try, act_hi_try) != self._vi_state_from_sets(act_lo0, act_hi0))
            )
            metrics["soft_active"] = 1.0 if soft_active else 0.0
            metrics["alpha_trial"] = float(alpha)
            if prev_state_save is None:
                self._vi_prev_state = None
            else:
                self._vi_prev_state = prev_state_save
            self._vi_pending_state = prev_pending
            self._vi_pending_count = prev_counts
            return metrics, step

        # Always try full step first (robust, avoids warm-start lock-in).
        metrics1, step1 = _trial(1.0)
        merit1 = metrics1["G_inf"] if merit_mode == "legacy" and mode == "dealii" else (metrics1["G_half"] if merit_mode == "legacy" else metrics1["merit"])
        if merit1 < best_merit:
            best_merit = merit1
            best_alpha = 1.0
            best_metrics = dict(metrics1)

        def _accept(metrics: dict[str, float], alpha: float) -> bool:
            if merit_mode == "legacy":
                if mode == "dealii":
                    return float(metrics["G_inf"]) < norm0
                return float(metrics["G_half"]) <= phi0 - c1 * alpha * float(G0 @ G0)
            res_ok = float(metrics["inactive_res_inf"]) <= max(
                float(getattr(self.vi_params, "filter_max_residual_growth", 1.25) or 1.25) * max(res0, 1.0e-16),
                1.0e-16,
            )
            gap_ok = float(metrics["active_gap_inf"]) <= max(
                float(getattr(self.vi_params, "filter_max_gap_growth", 1.25) or 1.25) * max(gap0, 1.0e-16),
                1.0e-16,
            )
            if not (res_ok and gap_ok):
                return False
            max_delta_active = int(getattr(self.vi_params, "filter_max_delta_active", 0) or 0)
            if max_delta_active > 0 and int(metrics.get("delta_active", 0.0)) > max_delta_active:
                return False
            merit = float(metrics["merit"])
            # Use a relative Armijo decrease on the VI merit. Scaling by 1.0 makes
            # late-time steps with merit << 1 fail unconditionally even when merit decreases.
            return merit <= merit0 - c1 * alpha * max(merit0, 1.0e-30)

        def _trace_trial(metrics: dict[str, float], accepted: bool) -> None:
            if not trace_ls:
                return
            msg = (
                f"        [vi-ls] alpha={float(metrics.get('alpha_trial', 0.0)):.3e} "
                f"merit={float(metrics.get('merit', 0.0)):.3e} "
                f"resI={float(metrics.get('inactive_res_inf', 0.0)):.3e} "
                f"gapA={float(metrics.get('active_gap_inf', 0.0)):.3e} "
                f"ΔA={int(metrics.get('delta_active', 0.0))} "
                f"soft={int(metrics.get('soft_active', 0.0))} "
                f"accepted={int(bool(accepted))}"
            )
            if not np.any(act_lo0 | act_hi0):
                alpha_trial = float(metrics.get("alpha_trial", 0.0))
                pred_ratio = (1.0 - alpha_trial) ** 2
                actual_ratio = float(metrics.get("merit", 0.0)) / max(merit0, 1.0e-30)
                msg += f" pred_ratio={pred_ratio:.3e} model_ratio={actual_ratio / max(pred_ratio, 1.0e-30):.3e}"
            print(msg)

        ok1 = _accept(metrics1, 1.0)
        _trace_trial(metrics1, ok1)
        if ok1:
            for f, buf in zip(funcs, snap):
                f.nodal_values[:] = buf
            self._ls_alpha_prev = 1.0
            self._vi_last_metrics = dict(metrics1)
            return step1

        alpha = float(getattr(np_, "ls_reduction", 0.5))
        if not (0.0 < alpha < 1.0):
            alpha = 0.5

        for _ in range(int(getattr(np_, "ls_max_iter", 12))):
            metrics, step = _trial(alpha)
            merit = metrics["G_inf"] if merit_mode == "legacy" and mode == "dealii" else (metrics["G_half"] if merit_mode == "legacy" else metrics["merit"])
            if merit < best_merit:
                best_merit = merit
                best_alpha = alpha
                best_metrics = dict(metrics)
            ok = _accept(metrics, alpha)
            _trace_trial(metrics, ok)
            if ok:
                for f, buf in zip(funcs, snap):
                    f.nodal_values[:] = buf
                self._ls_alpha_prev = alpha
                self._vi_last_metrics = dict(metrics)
                return step
            alpha *= float(getattr(np_, "ls_reduction", 0.5))

        for f, buf in zip(funcs, snap):
            f.nodal_values[:] = buf
        if best_alpha > 0.0:
            self._ls_alpha_prev = best_alpha
            self._vi_last_metrics = dict(best_metrics)
            step, _ = self._vi_apply_active_step_policy(
                alpha=best_alpha,
                S_red=S_red,
                active_mask=active_mask,
                strong_active_mask=strong_active_mask,
                active_delta_ref=active_delta_ref,
            )
            return step

        min_alpha = alpha
        if os.getenv("PYCUTFEM_LS_FAIL_HARD", "1").lower() not in {"0", "false", "no"}:
            raise RuntimeError("Line search failed: no semismooth residual decrease.")
        self._ls_alpha_prev = min_alpha
        self._vi_last_metrics = dict(metrics0)
        step, _ = self._vi_apply_active_step_policy(
            alpha=min_alpha,
            S_red=S_red,
            active_mask=active_mask,
            strong_active_mask=strong_active_mask,
            active_delta_ref=active_delta_ref,
        )
        return step

    def _newton_loop(self, funcs, prev_funcs, aux_funcs, bcs_now):
        # If no bounds are set, fall back to the standard Newton solver.
        if getattr(self, "_box_lower_full", None) is None or getattr(self, "_box_upper_full", None) is None:
            return super()._newton_loop(funcs, prev_funcs, aux_funcs, bcs_now)

        self._current_bcs = bcs_now
        dh = self.dh
        ndof_eff = getattr(self.restrictor, "full_size", dh.total_dofs)

        if getattr(self, "constraints", None) is not None:
            self._enforce_constraints_on_functions(funcs)
            self._enforce_constraints_on_functions(prev_funcs)

        # Optionally project the initial guess to be feasible.
        lo_red, hi_red = self._bounds_reduced()
        if bool(getattr(self.vi_params, "project_initial_guess", True)):
            self._project_funcs_to_bounds(funcs, bcs_now, lo_red, hi_red)

        norm_G0: float | None = None
        last_active: np.ndarray | None = None
        last_norm_G = float("inf")
        last_norm_R = float("inf")
        last_it = 0
        accept_factor = float(getattr(self.np, "accept_nonconverged_atol_factor", 0.0) or 0.0)
        atol = float(getattr(self.np, "newton_tol", 0.0) or 0.0)
        # Each accepted time step defines a new nonlinear subproblem. Reusing a
        # previously saturated LM damping parameter across solves can freeze the
        # next step at an almost-zero trust-region correction.
        self._vi_lm_lambda_current = float(getattr(self.vi_params, "unconstrained_lm_lambda0", 1.0e-4) or 1.0e-4)

        for it in range(int(self.np.max_newton_iter)):
            if self.pre_cb is not None:
                self.pre_cb(funcs)
            if getattr(self, "constraints", None) is not None:
                self._enforce_constraints_on_functions(funcs)
                self._enforce_constraints_on_functions(prev_funcs)

            current: Dict[str, "Function"] = {f.name: f for f in funcs}
            current.update({f.name: f for f in prev_funcs})
            if aux_funcs:
                current.update(aux_funcs)

            # Assemble reduced Jacobian and residual.
            if str(getattr(self, "backend", "")).lower() == "cpp":
                print(f"        VI Newton {it+1}: assembling...", flush=True)
            t_asm = time.perf_counter()
            A_red, R_red = self._assemble_system_reduced(current, need_matrix=True)
            asm_time = time.perf_counter() - t_asm

            # Prune decoupled rows/cols (CutFEM restriction).
            A_red, R_red, pruned, extra_asm = self._prune_decoupled_rows_cols(current, A_red, R_red, ndof_eff)
            asm_time += extra_asm
            if pruned:
                lo_red, hi_red = self._bounds_reduced()

            # Optional Jacobian eigenvalue tracing (debugging/conditioning studies).
            #
            # IMPORTANT:
            # - This is potentially expensive; it is off by default and can be
            #   restricted to a single step/iteration via env vars.
            # - These are eigenvalues of the *reduced, pruned* Jacobian A_red
            #   (before PDAS identity-row modification). Row/field scaling of
            #   equations will generally change these eigenvalues.
            if os.getenv("PYCUTFEM_TRACE_JAC_EIGS", "").lower() in {"1", "true", "yes"}:
                try:
                    step_filter = int(os.getenv("PYCUTFEM_TRACE_JAC_EIGS_STEP", "-1") or "-1")
                except Exception:
                    step_filter = -1
                try:
                    it_filter = int(os.getenv("PYCUTFEM_TRACE_JAC_EIGS_IT", "1") or "1")
                except Exception:
                    it_filter = 1

                step_now = int(getattr(self, "_current_step_no", -1))
                do_trace = (step_filter < 0 or step_now == step_filter) and (it + 1 == it_filter)
                if do_trace:
                    traced = getattr(self, "_trace_jac_eigs_done", None)
                    if traced is None:
                        traced = set()
                        setattr(self, "_trace_jac_eigs_done", traced)
                    key = (step_now, it + 1)
                    if key not in traced:
                        traced.add(key)
                        try:
                            n = int(getattr(A_red, "shape", (0, 0))[0])
                            nnz = int(getattr(A_red, "nnz", 0))
                        except Exception:
                            n = 0
                            nnz = 0
                        print(f"        [vi-jac] A_red: n={n} nnz={nnz}")
                        try:
                            k_req = int(os.getenv("PYCUTFEM_TRACE_JAC_EIGS_K", "6") or "6")
                        except Exception:
                            k_req = 6
                        try:
                            which = str(os.getenv("PYCUTFEM_TRACE_JAC_EIGS_WHICH", "LM") or "LM").strip().upper()
                        except Exception:
                            which = "LM"
                        try:
                            maxiter = int(os.getenv("PYCUTFEM_TRACE_JAC_EIGS_MAXITER", "300") or "300")
                        except Exception:
                            maxiter = 300
                        try:
                            tol = float(os.getenv("PYCUTFEM_TRACE_JAC_EIGS_TOL", "1e-6") or "1e-6")
                        except Exception:
                            tol = 1.0e-6

                        if n >= 3 and k_req >= 1:
                            k_eigs = int(min(max(k_req, 1), n - 2))
                            try:
                                evals = spla.eigs(A_red, k=k_eigs, which=which, tol=tol, maxiter=maxiter, return_eigenvectors=False)
                                evals = np.asarray(evals)
                                # Sort by magnitude (descending) for stable printing.
                                order = np.argsort(-np.abs(evals))
                                evals = evals[order]

                                def _fmt(ev: complex) -> str:
                                    evc = complex(ev)
                                    if abs(evc.imag) < 1.0e-12:
                                        return f"{evc.real:.3e}"
                                    sgn = "+" if evc.imag >= 0.0 else "-"
                                    return f"{evc.real:.3e}{sgn}{abs(evc.imag):.3e}j"

                                print(
                                    f"        [vi-jac] eigs(A_red) which={which} k={k_eigs}: "
                                    + ", ".join(_fmt(ev) for ev in evals)
                                )
                            except Exception as e:
                                print(f"        [vi-jac] eigs(A_red) failed: {e}")
                        else:
                            print("        [vi-jac] eigs(A_red) skipped: matrix too small.")
                # else: not traced

            # Current reduced iterate and semismooth residual.
            x_red = self._pack_reduced_iterate(funcs)
            c_red = self._vi_c_vector(A_red, lo_red, hi_red)
            G_red, act_lo, act_hi = self._pdas_residual_and_sets(x_red, R_red, lo_red, hi_red, c_red=c_red)
            metrics0 = self._vi_metrics(x_red, R_red, lo_red, hi_red, act_lo, act_hi, G_red)
            norm_G = float(np.linalg.norm(G_red, ord=np.inf))
            norm_R = float(np.linalg.norm(R_red, ord=np.inf))
            last_norm_G = float(norm_G)
            last_norm_R = float(norm_R)
            last_it = int(it + 1)
            self._last_nonlinear_norm = float(norm_G)
            self._last_nonlinear_norm_label = "|G|_∞"
            self._last_nonlinear_reason = None
            self._last_nonlinear_accepted = False
            self._last_nonlinear_relaxed_accept = False

            if norm_G0 is None:
                norm_G0 = norm_G
            tol_eff = max(float(self.np.newton_tol), float(getattr(self.np, "newton_rtol", 0.0) or 0.0) * float(norm_G0))

            active = act_lo | act_hi

            nA = int(np.count_nonzero(active))
            nI = int(active.size - nA)
            changed = -1
            if last_active is not None and last_active.shape == active.shape:
                changed = int(np.count_nonzero(last_active != active))
            last_active = active.copy()
            strong_active = self._vi_strong_active_mask(x_red, R_red, lo_red, hi_red, act_lo, act_hi, c_red)

            print(
                f"        VI Newton {it+1}: |G|_∞={norm_G:.2e}  |R|_∞={norm_R:.2e}  nA={nA} nI={nI}"
                + (f"  ΔA={changed}" if changed >= 0 else "")
                + f"  |R_I|_∞={metrics0['inactive_res_inf']:.2e}  |gap_A|_∞={metrics0['active_gap_inf']:.2e}"
                + f"  λ={float(getattr(self, '_vi_lambda_current', 0.0)):.2e}"
                + f"  nA_strong={int(np.count_nonzero(strong_active))}"
                + f"  (asm={asm_time:.2e}s)"
            )
            if os.getenv("PYCUTFEM_VI_TRACE_FIELDS", "").lower() in {"1", "true", "yes"}:
                items = []
                for fld in np.unique(self._vi_red_field_names):
                    fld = str(fld)
                    if not fld:
                        continue
                    mask = self._vi_red_field_names == fld
                    n_lo = int(np.count_nonzero(act_lo & mask))
                    n_hi = int(np.count_nonzero(act_hi & mask))
                    c_f = float(np.median(c_red[mask])) if np.any(mask) else float(getattr(self.vi_params, "c", 1.0) or 1.0)
                    if n_lo or n_hi or np.any(np.isfinite(lo_red[mask]) | np.isfinite(hi_red[mask])):
                        items.append(f"{fld}:lo={n_lo},hi={n_hi},c={c_f:.2e}")
                if items:
                    print("        [vi-set] " + "  ".join(items))

            # Match the standard Newton field-residual tracing output for VI solves.
            # (Useful to identify which PDE block dominates the raw residual.)
            if os.getenv("PYCUTFEM_NEWTON_TRACE_RES_FIELDS", "").lower() in {"1", "true", "yes"}:
                try:
                    R_full_dbg = self.restrictor.expand_vec(R_red)
                    field_norms = []
                    for fld in getattr(self.dh, "field_names", []):
                        try:
                            sl = self.dh.get_field_slice(fld)
                        except Exception:
                            continue
                        if sl is None or len(sl) == 0:
                            continue
                        sl = np.asarray(sl, dtype=int).ravel()
                        field_norms.append((float(np.linalg.norm(R_full_dbg[sl], ord=np.inf)), str(fld)))
                    field_norms.sort(reverse=True, key=lambda t: t[0])
                    n_show_raw = os.getenv("PYCUTFEM_NEWTON_TRACE_RES_FIELDS_N", "8").strip()
                    try:
                        n_show = max(1, int(n_show_raw))
                    except Exception:
                        n_show = 8
                    items = ", ".join(f"{name}:{val:.2e}" for val, name in field_norms[:n_show])
                    print(f"        [res] {items}")
                except Exception as exc:
                    print(f"        [res] trace failed: {exc}")

            # Optional residual tracing (same env vars as the standard Newton solver).
            # For VI, we expose both the semismooth residual G and the raw residual R
            # so users can identify whether non-convergence comes from active-set
            # complementarity (gap) or from a particular PDE block.
            if os.getenv("PYCUTFEM_RESIDUAL_TRACE", "").lower() in {"1", "true", "yes"}:
                try:
                    G_full = self.restrictor.expand_vec(G_red)
                    R_full = self.restrictor.expand_vec(R_red)
                except Exception:
                    G_full = None
                    R_full = None

                def _trace_vec(tag: str, vec: np.ndarray | None) -> None:
                    if vec is None:
                        return
                    vec = np.asarray(vec, dtype=float).ravel()
                    if vec.size == 0:
                        return
                    worst = int(np.argmax(np.abs(vec)))
                    worst_val = float(vec[worst])
                    field = None
                    try:
                        field = getattr(self.dh, "_dof_to_node_map", {}).get(worst, (None, None))[0]
                    except Exception:
                        field = None
                    fmsg = f", field={field}" if field is not None else ""

                    coords_msg = ""
                    if os.getenv("PYCUTFEM_RESIDUAL_TRACE_COORDS", "").lower() in {"1", "true", "yes"}:
                        try:
                            self.dh._ensure_dof_coords()
                            coords = np.asarray(getattr(self.dh, "_dof_coords", None), dtype=float)
                            if coords.ndim == 2 and coords.shape[0] > worst and coords.shape[1] >= 2:
                                coords_msg = f", x={coords[worst, 0]:.3e}, y={coords[worst, 1]:.3e}"
                        except Exception:
                            coords_msg = ""

                    print(f"        [vi-res] worst_{tag}: gdof={worst} val={worst_val:.3e}{fmsg}{coords_msg}")

                    if os.getenv("PYCUTFEM_RESIDUAL_TRACE_FIELDS", "").lower() in {"1", "true", "yes"}:
                        field_norms = []
                        for fld in getattr(self.dh, "field_names", []):
                            try:
                                sl = self.dh.get_field_slice(fld)
                            except Exception:
                                continue
                            if sl is None or len(sl) == 0:
                                continue
                            try:
                                field_norms.append(f"{fld}:{np.linalg.norm(vec[sl], ord=np.inf):.2e}")
                            except Exception:
                                continue
                        if field_norms:
                            print(f"        [vi-res] per-field |{tag}|_∞: " + ", ".join(field_norms))

                _trace_vec("G", G_full)
                _trace_vec("R", R_full)

            if norm_G <= tol_eff:
                self._last_nonlinear_reason = 2
                self._last_nonlinear_accepted = True
                self._last_nonlinear_relaxed_accept = False
                delta = np.hstack([f.nodal_values - f_prev.nodal_values for f, f_prev in zip(funcs, prev_funcs)])
                return delta, True, it + 1

            # Build semismooth Newton system (PDAS): inactive rows use Newton, active rows are identity.
            rhs = (-R_red).astype(float, copy=True)
            if nA:
                rhs[act_lo] = lo_red[act_lo] - x_red[act_lo]
                rhs[act_hi] = hi_red[act_hi] - x_red[act_hi]
                A_mod = self._apply_identity_rows(A_red, np.flatnonzero(active))
            else:
                A_mod = A_red

            lam = float(getattr(self, "_vi_lambda_current", 0.0) or 0.0)
            delta_trigger = int(getattr(self.vi_params, "inactive_reg_delta_active_trigger", 0) or 0)
            if delta_trigger > 0 and changed >= delta_trigger:
                lam = max(lam, float(getattr(self.vi_params, "inactive_reg_lambda0", 0.0) or 0.0))
                lam = min(
                    float(getattr(self.vi_params, "inactive_reg_lambda_max", 1.0e6) or 1.0e6),
                    lam * float(getattr(self.vi_params, "inactive_reg_growth", 5.0) or 5.0) if lam > 0.0 else 0.0,
                )

            lin_time = 0.0
            ls_time = 0.0
            reg_retries = 0
            used_lm = False
            lm_info: dict[str, float] = {}

            def _eval_reduced_trial(
                step_red: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
                step_red = np.asarray(step_red, dtype=float)
                if not np.all(np.isfinite(step_red)):
                    raise RuntimeError("Trial step contains non-finite entries.")
                snap = [f.nodal_values.copy() for f in funcs]
                prev_state_save = None if self._vi_prev_state is None else self._vi_prev_state.copy()
                prev_pending = None if self._vi_pending_state is None else self._vi_pending_state.copy()
                prev_counts = None if self._vi_pending_count is None else self._vi_pending_count.copy()
                try:
                    dU_full = self.restrictor.expand_vec(step_red)
                    if not np.all(np.isfinite(dU_full)):
                        raise RuntimeError("Expanded trial step contains non-finite entries.")
                    dh.add_to_functions(dU_full, funcs)
                    dh.apply_bcs(bcs_now, *funcs)
                    if getattr(self, "constraints", None) is not None:
                        self._enforce_constraints_on_functions(funcs)
                    for f in funcs:
                        vals = np.asarray(getattr(f, "nodal_values", np.array([], dtype=float)), dtype=float)
                        if vals.size and not np.all(np.isfinite(vals)):
                            raise RuntimeError(
                                f"Trial iterate became non-finite after constraint enforcement in field "
                                f"'{getattr(f, 'name', '<unnamed>')}'."
                            )
                    if bool(getattr(self.vi_params, "project_each_iteration", True)):
                        self._project_funcs_to_bounds(funcs, bcs_now, lo_red, hi_red)
                    _, R_try = self._assemble_system_reduced(current, need_matrix=False)
                    R_try = np.asarray(R_try, dtype=float)
                    if not np.all(np.isfinite(R_try)):
                        raise RuntimeError("Trial residual contains non-finite entries.")
                    x_try = self._pack_reduced_iterate(funcs)
                    if not np.all(np.isfinite(x_try)):
                        raise RuntimeError("Trial reduced iterate contains non-finite entries.")
                    G_try, act_lo_try, act_hi_try = self._pdas_residual_and_sets(x_try, R_try, lo_red, hi_red, c_red=c_red)
                    metrics_try = self._vi_metrics(x_try, R_try, lo_red, hi_red, act_lo_try, act_hi_try, G_try)
                    metrics_try["delta_active"] = float(
                        np.count_nonzero(self._vi_state_from_sets(act_lo_try, act_hi_try) != self._vi_state_from_sets(act_lo, act_hi))
                    )
                    return x_try, R_try, G_try, act_lo_try, act_hi_try, metrics_try
                finally:
                    if prev_state_save is None:
                        self._vi_prev_state = None
                    else:
                        self._vi_prev_state = prev_state_save
                    self._vi_pending_state = prev_pending
                    self._vi_pending_count = prev_counts
                    for f, buf in zip(funcs, snap):
                        f.nodal_values[:] = buf

            lm_requested = bool(getattr(self.vi_params, "unconstrained_lm", False))
            lm_trace = os.getenv("PYCUTFEM_VI_TRACE_LM", "").lower() in {"1", "true", "yes"}
            use_lm = lm_requested
            if lm_requested and nA > 0 and lm_trace:
                active_items = []
                for fld in np.unique(self._vi_red_field_names):
                    fld = str(fld)
                    if not fld:
                        continue
                    mask = self._vi_red_field_names == fld
                    n_lo = int(np.count_nonzero(act_lo & mask))
                    n_hi = int(np.count_nonzero(act_hi & mask))
                    if n_lo or n_hi:
                        active_items.append(f"{fld}:lo={n_lo},hi={n_hi}")
                msg = f"        [vi-lm] active-set mode: nA={nA} semismooth LM will use the PDAS-modified system."
                if active_items:
                    msg += " active={" + ", ".join(active_items) + "}"
                print(msg)
            if use_lm:
                lm_lam = float(getattr(self, "_vi_lm_lambda_current", 0.0) or 0.0)
                lm0 = float(getattr(self.vi_params, "unconstrained_lm_lambda0", 1.0e-4) or 1.0e-4)
                lm_max = float(getattr(self.vi_params, "unconstrained_lm_lambda_max", 1.0e6) or 1.0e6)
                lm_growth = float(getattr(self.vi_params, "unconstrained_lm_growth", 5.0) or 5.0)
                lm_decay = float(getattr(self.vi_params, "unconstrained_lm_decay", 0.5) or 0.5)
                lm_accept = float(getattr(self.vi_params, "unconstrained_lm_accept_ratio", 1.0e-3) or 1.0e-3)
                lm_good = float(getattr(self.vi_params, "unconstrained_lm_good_ratio", 0.75) or 0.75)
                lm_max_tries = int(getattr(self.vi_params, "unconstrained_lm_max_tries", 6) or 6)
                phi0_lm = float(metrics0["merit"])
                best_step = None
                best_phi = phi0_lm
                best_rho = -np.inf
                best_metrics = None
                res0_lm = float(metrics0["inactive_res_inf"])
                gap0_lm = float(metrics0["active_gap_inf"])
                max_delta_active = int(getattr(self.vi_params, "filter_max_delta_active", 0) or 0)
                max_res_growth = float(getattr(self.vi_params, "filter_max_residual_growth", 1.25) or 1.25)
                max_gap_growth = float(getattr(self.vi_params, "filter_max_gap_growth", 1.25) or 1.25)
                fallback_rho_min = max(1.0e-6, 0.1 * lm_accept)

                for lm_try in range(max(lm_max_tries, 1)):
                    try:
                        H_lm, rhs_lm, lm_diag = self._vi_build_lm_system(A_mod, rhs, lm_lam)
                        t_lin = time.perf_counter()
                        step_lm = self._solve_linear_system(H_lm, rhs_lm)
                        lin_time += time.perf_counter() - t_lin
                    except RuntimeError as exc:
                        if lm_trace:
                            print(f"        [vi-lm] try={lm_try+1} λ={lm_lam:.2e} linear solve failed: {exc}")
                        lm_lam = min(lm_max, max(lm_lam, lm0) * lm_growth if max(lm_lam, lm0) > 0.0 else lm0)
                        continue
                    try:
                        _, _, _, _, _, metrics_try = _eval_reduced_trial(step_lm)
                        phi_try = float(metrics_try["merit"])
                    except RuntimeError as exc:
                        if lm_trace:
                            print(f"        [vi-lm] try={lm_try+1} λ={lm_lam:.2e} rejected: {exc}")
                        lm_lam = min(lm_max, max(lm_lam, lm0) * lm_growth if max(lm_lam, lm0) > 0.0 else lm0)
                        continue
                    if not np.isfinite(phi_try):
                        if lm_trace:
                            print(f"        [vi-lm] try={lm_try+1} λ={lm_lam:.2e} rejected: non-finite phi_try")
                        lm_lam = min(lm_max, max(lm_lam, lm0) * lm_growth if max(lm_lam, lm0) > 0.0 else lm0)
                        continue
                    _, model_phi, pred_model, _ = self._vi_unconstrained_lm_model(G_red, A_mod, step_lm, lm_lam, lm_diag, phi_try=phi_try)
                    actual = phi0_lm - phi_try
                    rho = actual / max(phi0_lm, 1.0e-30)
                    res_ok = float(metrics_try["inactive_res_inf"]) <= max(max_res_growth * max(res0_lm, 1.0e-16), 1.0e-16)
                    gap_ok = float(metrics_try["active_gap_inf"]) <= max(max_gap_growth * max(gap0_lm, 1.0e-16), 1.0e-16)
                    delta_ok = max_delta_active <= 0 or int(metrics_try.get("delta_active", 0.0)) <= max_delta_active
                    if lm_trace:
                        print(
                            f"        [vi-lm] try={lm_try+1} λ={lm_lam:.2e} "
                            f"phi0={phi0_lm:.3e} phitry={phi_try:.3e} "
                            f"pred_model={pred_model:.3e} act={actual:.3e} rho={rho:.3e} "
                            f"res_ok={int(res_ok)} gap_ok={int(gap_ok)} delta_ok={int(delta_ok)}"
                        )
                    if phi_try < best_phi:
                        best_phi = phi_try
                        best_step = np.asarray(step_lm, dtype=float).copy()
                        best_rho = float(rho)
                        best_metrics = dict(metrics_try)
                    if np.isfinite(rho) and actual > 0.0 and rho >= lm_accept and res_ok and gap_ok and delta_ok:
                        used_lm = True
                        S_red = np.asarray(step_lm, dtype=float)
                        accepted_alpha = 1.0
                        merit_ratio = phi_try / max(phi0_lm, 1.0e-30)
                        lm_info = {"rho": float(rho), "phi_try": float(phi_try), "phi0": float(phi0_lm)}
                        if rho >= lm_good:
                            lm_lam = max(lm0, lm_lam * lm_decay if lm_lam > 0.0 else lm0)
                        self._vi_lm_lambda_current = float(min(max(lm_lam, lm0), lm_max))
                        self._ls_alpha_prev = 1.0
                        self._vi_last_metrics = dict(metrics_try)
                        break
                    lm_lam = min(lm_max, max(lm_lam, lm0) * lm_growth if max(lm_lam, lm0) > 0.0 else lm0)
                if not used_lm and best_step is not None and best_phi < phi0_lm and np.isfinite(best_rho) and best_rho >= fallback_rho_min:
                    used_lm = True
                    S_red = np.asarray(best_step, dtype=float)
                    accepted_alpha = 1.0
                    merit_ratio = best_phi / max(phi0_lm, 1.0e-30)
                    lm_info = {"rho": float(best_rho), "phi_try": float(best_phi), "phi0": float(phi0_lm)}
                    self._vi_lm_lambda_current = float(min(max(lm_lam, lm0), lm_max))
                    self._ls_alpha_prev = 1.0
                    self._vi_last_metrics = dict(best_metrics or {"merit": float(best_phi)})
                elif not used_lm and best_step is not None and best_phi < phi0_lm and lm_trace:
                    print(
                        f"        [vi-lm] best trial rejected for insufficient improvement: "
                        f"rho={float(best_rho):.3e} < fallback_min={fallback_rho_min:.3e}"
                    )
                if not used_lm:
                    self._vi_lm_lambda_current = float(lm0)

            if used_lm:
                self._vi_last_retry_count = 0
                dU_full = self.restrictor.expand_vec(S_red)
                if not np.all(np.isfinite(dU_full)):
                    raise RuntimeError("Accepted LM step expanded to non-finite full increment.")
                dh.add_to_functions(dU_full, funcs)
                dh.apply_bcs(bcs_now, *funcs)
                if getattr(self, "constraints", None) is not None:
                    self._enforce_constraints_on_functions(funcs)
                if bool(getattr(self.vi_params, "project_each_iteration", True)):
                    self._project_funcs_to_bounds(funcs, bcs_now, lo_red, hi_red)
                self._assert_functions_finite(funcs, context="after accepted LM step")
                print(
                    f"          timings: solve={lin_time:.2e}s, line-search={ls_time:.2e}s, "
                    f"alpha=1.00e+00, merit_ratio={float(lm_info.get('phi_try', 0.0)) / max(float(lm_info.get('phi0', 1.0)), 1.0e-30):.3f}, "
                    f"lm_rho={float(lm_info.get('rho', 0.0)):.3f}, lm_lambda={float(getattr(self, '_vi_lm_lambda_current', 0.0)):.2e}"
                )
                continue

            while True:
                A_work, reg_diag = self._vi_apply_inactive_regularization(A_mod, A_red, active, lam)
                try:
                    t_lin = time.perf_counter()
                    S_red = self._solve_linear_system(A_work, rhs)
                    lin_time += time.perf_counter() - t_lin
                except RuntimeError as exc:
                    lam_max = float(getattr(self.vi_params, "inactive_reg_lambda_max", 1.0e6) or 1.0e6)
                    lam_growth = float(getattr(self.vi_params, "inactive_reg_growth", 5.0) or 5.0)
                    lam0 = float(getattr(self.vi_params, "inactive_reg_lambda0", 0.0) or 0.0)
                    if lam < lam_max and (lam > 0.0 or lam0 > 0.0):
                        lam = max(lam, lam0)
                        lam = min(lam_max, lam * lam_growth if lam > 0.0 else lam0)
                        reg_retries += 1
                        if os.getenv("PYCUTFEM_VI_TRACE_REG", "").lower() in {"1", "true", "yes"}:
                            print(f"        [vi-reg] retry={reg_retries}  λ->{lam:.2e}  after linear solve failure: {exc}")
                        continue
                    raise

                if bool(getattr(self.np, "line_search", True)):
                    t_ls = time.perf_counter()
                    try:
                        S_red = self._line_search_vi_reduced(
                                x0_red=x_red,
                                R0_red=R_red,
                                G0=G_red,
                                act_lo0=act_lo,
                                act_hi0=act_hi,
                                S_red=S_red,
                                c_red=c_red,
                                strong_active_mask=strong_active,
                                active_delta_ref=max(changed, 0),
                                active_mask=active,
                                funcs=funcs,
                                coeffs=current,
                                bcs_now=bcs_now,
                                lo_red=lo_red,
                                hi_red=hi_red,
                            )
                        ls_time += time.perf_counter() - t_ls
                        break
                    except RuntimeError:
                        ls_time += time.perf_counter() - t_ls
                        lam_max = float(getattr(self.vi_params, "inactive_reg_lambda_max", 1.0e6) or 1.0e6)
                        lam_growth = float(getattr(self.vi_params, "inactive_reg_growth", 5.0) or 5.0)
                        lam0 = float(getattr(self.vi_params, "inactive_reg_lambda0", 0.0) or 0.0)
                        if lam < lam_max and (lam > 0.0 or lam0 > 0.0):
                            lam = max(lam, lam0)
                            lam = min(lam_max, lam * lam_growth if lam > 0.0 else lam0)
                            reg_retries += 1
                            if os.getenv("PYCUTFEM_VI_TRACE_REG", "").lower() in {"1", "true", "yes"}:
                                print(f"        [vi-reg] retry={reg_retries}  λ->{lam:.2e}  diag∞={float(np.max(reg_diag)):.2e}")
                            continue
                        raise
                else:
                    break
            self._vi_lambda_current = float(lam)
            self._vi_last_retry_count = int(reg_retries)
            accepted_alpha = float(getattr(self, "_ls_alpha_prev", 1.0) or 1.0)
            accepted_metrics = dict(getattr(self, "_vi_last_metrics", {}) or {})
            merit_ratio = float(accepted_metrics.get("merit", metrics0["merit"])) / max(float(metrics0["merit"]), 1.0e-30)

            # Apply the accepted step.
            dU_full = self.restrictor.expand_vec(S_red)
            if not np.all(np.isfinite(dU_full)):
                raise RuntimeError("Accepted semismooth Newton step expanded to non-finite full increment.")
            dh.add_to_functions(dU_full, funcs)
            dh.apply_bcs(bcs_now, *funcs)
            if getattr(self, "constraints", None) is not None:
                self._enforce_constraints_on_functions(funcs)
            if bool(getattr(self.vi_params, "project_each_iteration", True)):
                self._project_funcs_to_bounds(funcs, bcs_now, lo_red, hi_red)
            self._assert_functions_finite(funcs, context="after accepted semismooth Newton step")

            lam0 = float(getattr(self.vi_params, "inactive_reg_lambda0", 0.0) or 0.0)
            lam_max = float(getattr(self.vi_params, "inactive_reg_lambda_max", 1.0e6) or 1.0e6)
            lam_growth = float(getattr(self.vi_params, "inactive_reg_growth", 5.0) or 5.0)
            stall_alpha = float(getattr(self.vi_params, "inactive_reg_stall_alpha", 0.25) or 0.25)
            stall_ratio = float(getattr(self.vi_params, "inactive_reg_stall_merit_ratio", 0.95) or 0.95)
            if lam0 > 0.0 and accepted_alpha < stall_alpha and merit_ratio > stall_ratio and self._vi_lambda_current < lam_max:
                self._vi_lambda_current = min(lam_max, max(self._vi_lambda_current, lam0) * lam_growth)
                if os.getenv("PYCUTFEM_VI_TRACE_REG", "").lower() in {"1", "true", "yes"}:
                    print(f"        [vi-reg] accepted-stall  alpha={accepted_alpha:.2e}  merit_ratio={merit_ratio:.3f}  λ->{self._vi_lambda_current:.2e}")
            elif accepted_alpha >= 0.999 and reg_retries == 0:
                lam_decay = float(getattr(self.vi_params, "inactive_reg_decay", 0.5) or 0.5)
                if 0.0 < lam_decay < 1.0:
                    self._vi_lambda_current *= lam_decay
                    if self._vi_lambda_current < lam0:
                        self._vi_lambda_current = lam0

            print(f"          timings: solve={lin_time:.2e}s, line-search={ls_time:.2e}s, alpha={accepted_alpha:.2e}, merit_ratio={merit_ratio:.3f}")

        if accept_factor > 0.0 and atol > 0.0 and np.isfinite(last_norm_G) and last_norm_G <= accept_factor * atol:
            print(
                f"    [warn] Accepting VI Newton best iterate since |G|_∞={last_norm_G:.3e} "
                f"≤ {accept_factor:g}·atol (atol={atol:.3e})."
            )
            self._last_nonlinear_norm = float(last_norm_G)
            self._last_nonlinear_norm_label = "|G|_∞"
            self._last_nonlinear_reason = 2
            self._last_nonlinear_accepted = True
            self._last_nonlinear_relaxed_accept = True
            delta = np.hstack([f.nodal_values - f_prev.nodal_values for f, f_prev in zip(funcs, prev_funcs)])
            return delta, True, int(last_it)

        raise RuntimeError("VI Newton did not converge – adjust tolerances/Δt or verify Jacobian.")









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
