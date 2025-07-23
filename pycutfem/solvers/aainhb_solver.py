# aainhb_solver.py – AA‑INHB nonlinear solver for CutFEM
"""AA‑INHB = Anderson–AdaGrad Inexact‑Newton with Hessian Bounds
----------------------------------------------------------------

This module plugs a *globally convergent* variant of Newton’s method into
pycutfem.  It combines four inexpensive safeguards that together remove the
classic divergence cycle on tough CutFEM interface problems:

    • **IN**  Inexact Newton direction        (Krylov solve to loose tol)
    • **HB**  Hessian‑bounded diagonal pre‑conditioning  (clip to [γ⁻¹,γ])
    • **AA(m)**  Anderson Acceleration with window *m*   (non‑linear Krylov)
    • **LS**  Monotone Armijo back‑tracking line search   (never uphill)

The implementation below is a *drop‑in* replacement for ``NewtonSolver`` –
just instantiate ``AAINHBSolver`` instead.  No driver code changes are needed.

Usage
-----
>>> solver = AAINHBSolver(residual_form, jacobian_form,
...                       dof_handler=dh, mixed_element=me,
...                       bcs=bcs_inh, bcs_homog=bcs_hom)
>>> deltaU, nSteps, wall = solver.solve_time_interval(functions=U,
...                                                   prev_functions=Uold)

Notes
-----
* The class re‑uses *all* assembly utilities from ``nonlinear_solver.py``.
* Only the private ``_newton_loop`` is overridden.
* Krylov solve uses GMRES with restart=50 and relative tolerance η₍CG₎‖R‖₂.
* Running‑RMS pre‑conditioner follows AdaGrad and is clipped to [γ⁻¹,γ].
* Anderson window depth *m* and Armijo parameters are exposed as attributes.
"""
from __future__ import annotations

from collections import deque
import re
from typing import List, Dict, Optional, Tuple
import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import spilu, LinearOperator


from pycutfem.solvers.nonlinear_solver import NewtonSolver, TimeStepperParameters, LinearSolverParameters, NewtonParameters  # noqa: E402

# ---------------------------------------------------------------------------
# Helper: gather global DOF vector and restore snapshots
# ---------------------------------------------------------------------------

def _gather_global(dh, funcs):
    """Return one *global* nodal vector that matches *dh.total_dofs*."""
    vec = np.zeros(dh.total_dofs)
    for f in funcs:
        for gdof, lidx in f._g2l.items():
            vec[gdof] = f.nodal_values[lidx]
    return vec


def _snapshot_funcs(funcs):
    """Shallow snapshot of *all* nodal arrays (returns a list)."""
    return [f.nodal_values.copy() for f in funcs]


def _restore_funcs(funcs, snaps):
    for f, buf in zip(funcs, snaps):
        f.nodal_values[:] = buf


# ---------------------------------------------------------------------------
#  Main solver class
# ---------------------------------------------------------------------------
class AAINHBSolver(NewtonSolver):
    """Anderson‑accelerated, Hessian‑bounded Inexact Newton solver."""

    # -------- hyper‑parameters (can be tuned per instance) ----------------
    gamma: float = 20.0           # Hessian‑bound γ (clip P to [γ⁻¹,γ])
    m_AA: int = 4                 # Anderson depth m
    eta_CG: float = 1e-2          # relative Krylov tolerance η₍CG₎
    eps: float = 1e-8             # AdaGrad epsilon
    alpha_0: float = 1.0          # initial step length
    c1: float = 1e-6              # New Armijo slope reduction (smaller to accept larger steps)
    alpha_min: float = 1e-6       # minimum step length

    # ------------------------------------------------------------------
    #  Construction (identical signature to NewtonSolver)
    # ------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Uhist: deque[np.ndarray] = deque(maxlen=self.m_AA)
        self.Fhist: deque[np.ndarray] = deque(maxlen=self.m_AA)
        self.v: Optional[np.ndarray] = None   # running RMS for AdaGrad
        self.last_alpha: float = 1.0  # for adaptive initial line search alpha

    # ------------------------------------------------------------------
    #  Internal Krylov solve (GMRES with loose tol)
    # ------------------------------------------------------------------
    def _resprint(self, rk):
        self._it += 1
        if self._it % 20 == 0:          # every 20 iter
            print(f"          GMRES |r|₂ = {rk:8.2e}")
    def _solve_krylov(self, A, rhs, reltol):
        # Jacobi / AdaGrad diagonal
        

        ilu = spilu(A.tocsc(), drop_tol=1e-3, fill_factor=10)
        M   = LinearOperator(A.shape, ilu.solve)

        x, info = spla.gmres(A, rhs,
                            rtol=reltol, atol=0.0,
                            restart=100, maxiter=200,
                            M=M)
        if info == 0:
            return x
        return spla.spsolve(A, rhs)          # fallback


    # ------------------------------------------------------------------
    #  Overridden Newton loop implementing AA‑INHB (Algorithm 3.1)
    # ------------------------------------------------------------------
    def _newton_loop(self, funcs, prev_funcs, aux_funcs, bcs_now):
        dh = self.dh
        np_ = self.np

        self.Uhist.clear()  # New (reset histories for each nonlinear solve)
        self.Fhist.clear()  # New

        # allocate running‑RMS buffer lazily -----------------------------
        ndof = dh.total_dofs
        if self.v is None or self.v.size != ndof:
            self.v = np.zeros(ndof)

        for k in range(1, np_.max_newton_iter + 1):
            # ------------------------------------------------------ 1)
            # Assemble residual R and Jacobian J
            # ----------------------------------------------------------
            coeffs: Dict[str, "Function"] = {f.name: f for f in funcs}
            coeffs.update({f.name: f for f in prev_funcs})
            if aux_funcs:
                coeffs.update(aux_funcs)

            J, R = self._assemble_system(coeffs, need_matrix=True)
            norm_R_inf = np.linalg.norm(R, ord=np.inf)
            print(f"        AA‑INHB {k}: |R|_∞ = {norm_R_inf:.2e}")
            if norm_R_inf < np_.newton_tol:
                # time‑step increment to report
                delta = np.hstack([
                    f.nodal_values - f_prev.nodal_values
                    for f, f_prev in zip(funcs, prev_funcs)
                ])
                return delta

            # Newton direction (inexact) ---------------------------------
            tol_krylov = self.eta_CG  # New
            if k == 1: tol_krylov = 5e-2  # New (slightly looser for first step)
            dU_N = self._solve_krylov(J, -R, reltol=tol_krylov)  # New (passed corrected reltol)

            # Hessian‑bounded diagonal scaling ---------------------------
            g = J.T @ R                              # true gradient
            # running RMS (AdaGrad style)
            if k == 1:
                self.v = g * g + 1e-6  # New (initialize v based on first gradient to avoid zero)
            else:
                self.v = 0.9 * self.v + 0.1 * (g * g)  # New (faster adaptation)
            P = 1.0 / (np.sqrt(self.v) + self.eps)
            P = np.clip(P, 1.0 / self.gamma, self.gamma)
            dU_H = P * dU_N

            # Gather current solution vector U
            U_vec = _gather_global(dh, funcs)
            U_bar = U_vec + dU_H

            # ------------------------------------------------------ 3)
            # Anderson Acceleration candidate
            # ----------------------------------------------------------
            # Apply dU_H *once* to compute F_bar (we need residual there)
            snap = _snapshot_funcs(funcs)
            dh.add_to_functions(dU_H, funcs)
            dh.apply_bcs(bcs_now, *funcs)
            _, F_bar = self._assemble_system(coeffs, need_matrix=False)
            _restore_funcs(funcs, snap)              # rollback

            Ucand_vec = U_bar.copy()
            if len(self.Uhist) > 0:
                ΔF = np.column_stack([F_bar - f for f in self.Fhist])
                ΔU = np.column_stack([U_bar - u for u in self.Uhist])
                # least squares: argmin ‖ΔF θ – F_bar‖₂ s.t. sum θ = 1
                Q = ΔF.T @ ΔF
                s = ΔF.T @ F_bar
                theta_un, *_ = np.linalg.lstsq(Q, s, rcond=1e-8)  # New (use lstsq for robustness)
                ones_vec = np.ones(len(self.Uhist))
                z, *_ = np.linalg.lstsq(Q, ones_vec, rcond=1e-8)  # New
                sum_z = np.sum(z)
                if abs(sum_z) > 1e-10:
                    lambda_ = (1 - np.sum(theta_un)) / sum_z
                else:
                    lambda_ = 0
                theta = theta_un + lambda_ * z
                U_AA = U_bar - ΔU @ theta            # extrapolated point
                dU_AA = U_AA - U_vec
                # keep only if it is a *better* descent direction
                if np.dot(g, dU_AA) < np.dot(g, dU_H):
                    Ucand_vec = U_AA
            dU_cand = Ucand_vec - U_vec

            # ------------------------------------------------------ 4)
            # Monotone Armijo back‑tracking line search
            # ----------------------------------------------------------
            alpha = self.last_alpha  # New (start with previous accepted alpha)
            phi0 = 0.5 * np.dot(R, R)
            gTd = np.dot(g, dU_cand)
            assert gTd < 0.0, "Candidate is not a descent direction!"

            accepted = False
            while alpha >= self.alpha_min:
                dh.add_to_functions(alpha * dU_cand, funcs)
                dh.apply_bcs(bcs_now, *funcs)

                _, R_try = self._assemble_system(coeffs, need_matrix=False)
                phi_try = 0.5 * np.dot(R_try, R_try)

                # Armijo:  φ(trial) ≤ φ0 + c1 α gᵀd   (gᵀd < 0 ⇒ RHS < φ0)
                if phi_try <= phi0 + self.c1 * alpha * gTd:
                    accepted = True
                    print(f"        Armijo accepted α = {alpha:.2e}")  # New (added for debugging)
                    break    # accept this alpha
                # rollback and halve alpha
                dh.add_to_functions(-alpha * dU_cand, funcs)
                alpha *= 0.5

            if not accepted:
                raise RuntimeError("Armijo search failed – suggest to cut Δt")

            self.last_alpha = min(1.0, alpha * 4.0)  # New (more aggressive growth *4)

            # ------------------------------------------------------ 5)
            # Accept step, update histories, go to next Newton iter
            # ----------------------------------------------------------
            # (functions *already* contain the accepted step)
            self.Uhist.append(_gather_global(dh, funcs))
            self.Fhist.append(R_try)
            # loop continues – fresh residual assembled at top

        # Newton failed to converge within max iterations
        raise RuntimeError("AA‑INHB reached max_newton_iter without converging.")


# ---------------------------------------------------------------------------
#  Factory convenience – so user code can fall back easily
# ---------------------------------------------------------------------------

def get_solver(solver: str = "aainhb"):
    """Return the requested solver class by *string* name."""
    if solver.lower() in {"aainhb", "aa-inhb", "in"}:
        return AAINHBSolver
    from pycutfem.solvers.nonlinear_solver import NewtonSolver
    return NewtonSolver