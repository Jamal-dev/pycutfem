from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, NewtonSolver
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    Identity,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    div,
    grad,
    heaviside,
    inner,
    pos_part,
)
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.linalg import (
    d_spectral_positive_part_2x2_sym,
    smooth_pos,
    smooth_pos_derivative,
    spectral_positive_part_2x2_sym,
    sym,
)
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad
from pycutfem.utils.gmsh_loader import mesh_from_gmsh


def _tag_rectangle_boundaries(mesh: Mesh, *, L: float, H: float, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - float(L)) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - float(H)) <= tol,
        }
    )


def _one_minus(expr):
    # Keep the "function-like" operand on the left so the python backend
    # dispatches to VecOpInfo arithmetic (float - VecOpInfo is not supported).
    return (-expr) + Constant(1.0)


def _tensile_energy_mass_lumped_on_d(
    *,
    dof_handler: DofHandler,
    u: VectorFunction,
    dx_domain,
    mu_s: Constant,
    lambda_s: Constant,
    eta_pos: float,
    disc_reg: float,
    backend: str,
    quad_order: int,
    w_eps: float = 1.0e-14,
) -> np.ndarray:
    """Mass-lumped nodal ψ⁺(u) in the scalar `d` space."""
    E = sym(grad(u))
    E_plus, _, _, _, _ = spectral_positive_part_2x2_sym(E, eta_pos=float(eta_pos), disc_reg=float(disc_reg))
    trE = div(u)
    tr_pos = smooth_pos(trE, eta=float(eta_pos))
    # Match deal.II dynamic_fracture Miehe driver:
    # ψ⁺ := 1/2 σ⁺(E):E with σ⁺ = 2μE⁺ + λ⟨trE⟩_+ I.
    sig_plus = Constant(2.0) * mu_s * E_plus + Identity(2) * (lambda_s * tr_pos)
    psi_plus = Constant(0.5) * inner(sig_plus, E)

    psi = TestFunction("d", dof_handler=dof_handler)
    w_form = (psi) * dx_domain
    rhs_form = (psi * psi_plus) * dx_domain

    _, w_vec = assemble_form(Equation(None, w_form), dof_handler=dof_handler, bcs=[], quad_order=quad_order, backend=backend)
    _, rhs_vec = assemble_form(Equation(None, rhs_form), dof_handler=dof_handler, bcs=[], quad_order=quad_order, backend=backend)

    sl = np.asarray(dof_handler.get_field_slice("d"), dtype=int)
    w = np.asarray(w_vec[sl], dtype=float)
    rhs = np.asarray(rhs_vec[sl], dtype=float)

    out = np.zeros_like(w)
    mask = w > float(w_eps)
    if np.any(mask):
        out[mask] = rhs[mask] / (w[mask] + float(w_eps))
    return out


def _make_initial_notch(
    *,
    dof_handler: DofHandler,
    d0: Function,
    L: float,
    H: float,
    x0: float,
    y0: float,
    thickness: float,
) -> np.ndarray:
    """Initialize d with a sharp slit-like notch (d=1 in a thin band)."""
    xy = dof_handler.get_dof_coords("d")
    x = xy[:, 0]
    y = xy[:, 1]
    band = (x >= float(x0)) & (np.abs(y - float(y0)) <= 0.5 * float(thickness))
    d0_vals = np.zeros_like(d0.nodal_values, dtype=float)
    d0_vals[band] = 1.0
    d0.nodal_values[:] = d0_vals
    return d0_vals.copy()


def _write_point_csv(*, filename: str, dof_handler: DofHandler, u, v, d) -> None:
    """Write a pointwise CSV on the scalar `d` support points (Q1/Q2 nodal points)."""
    xy = np.asarray(dof_handler.get_dof_coords("d"), dtype=float)
    u_x = np.asarray(u.components[0].nodal_values, dtype=float)
    u_y = np.asarray(u.components[1].nodal_values, dtype=float)
    v_x = np.asarray(v.components[0].nodal_values, dtype=float)
    v_y = np.asarray(v.components[1].nodal_values, dtype=float)
    d_vals = np.asarray(d.nodal_values, dtype=float)
    pf = 1.0 - d_vals

    out = np.column_stack([xy[:, 0], xy[:, 1], u_x, u_y, v_x, v_y, pf])
    np.savetxt(
        filename,
        out,
        delimiter=",",
        header="x,y,u_x,u_y,v_x,v_y,pf",
        comments="",
        fmt="%.17e",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="examples/debug/out/mode_I_crack")
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--q", type=int, default=4, help="Quadrature order.")

    ap.add_argument(
        "--mesh-file",
        type=str,
        default="",
        help="Optional Gmsh .msh file. When set, overrides --nx/--ny/--order and uses physical-line tags for boundaries.",
    )
    ap.add_argument("--L", type=float, default=1.0)
    ap.add_argument("--H", type=float, default=1.0)
    ap.add_argument("--nx", type=int, default=40)
    ap.add_argument("--ny", type=int, default=40)
    ap.add_argument(
        "--order",
        type=int,
        default=1,
        choices=(1, 2),
        help="Polynomial degree for all fields (Q1 matches deal.II dynamic_fracture default).",
    )

    ap.add_argument("--dt", type=float, default=1.0e-4)
    ap.add_argument("--t-final", type=float, default=2.0e-2)
    ap.add_argument("--theta", type=float, default=1.0, help="One-step theta for u_t=v constraint and stress averaging.")

    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--mu", type=float, default=80.77e3)
    ap.add_argument("--lambda", dest="lambda_", type=float, default=121.15e3)
    ap.add_argument("--Gc", type=float, default=2.7)
    ap.add_argument("--ell", type=float, default=0.02, help="Damage length scale (phase-field).")
    ap.add_argument("--eta-d", type=float, default=0.0, help="Damage viscosity (0 for quasi-static damage solve).")
    ap.add_argument("--kappa", type=float, default=1.0e-9, help="Crack regularization floor in g(d).")
    ap.add_argument("--eta-pos", type=float, default=1.0e-12, help="Smoothing for ⟨·⟩_+ in spectral split.")
    ap.add_argument("--disc-reg", type=float, default=1.0e-16, help="Regularization under sqrt in eigen split.")
    ap.add_argument(
        "--history",
        type=str,
        default="max",
        choices=("max", "none"),
        help="Driving field H: 'max' uses H=max_t ψ⁺(u) (standard), 'none' uses instantaneous ψ⁺(u).",
    )
    ap.add_argument(
        "--mech-d",
        type=str,
        default="current",
        choices=("current", "lagged", "extrapolated"),
        help="Damage coefficient used in mechanics: current d_k, lagged d_n, or extrapolated (2*d_n-d_{n-1}, clipped).",
    )

    ap.add_argument(
        "--solve-mode",
        type=str,
        default="staggered",
        choices=("staggered", "monolithic_al"),
        help="Solve strategy. 'staggered' alternates (u,v) and d; 'monolithic_al' matches deal.II dynamic_fracture (monolithic Newton + augmented Lagrangian).",
    )
    ap.add_argument("--gamma-penal", type=float, default=1.0, help="Augmented Lagrangian penalty γ.")
    ap.add_argument("--al-max-it", type=int, default=10, help="Max augmented Lagrangian iterations per time step.")
    ap.add_argument("--al-tol-abs", type=float, default=1.0e-5, help="Absolute AL stopping tolerance.")
    ap.add_argument("--al-tol-rel", type=float, default=1.0e-5, help="Relative AL stopping tolerance.")
    ap.add_argument(
        "--al-eta",
        type=float,
        default=0.0,
        help="Smoothing η for positive-part in the AL term (η=0 uses hard max(0,x), matching deal.II).",
    )

    ap.add_argument("--newton-tol", type=float, default=1.0e-10, help="Newton |R|_∞ tolerance (monolithic mode).")
    ap.add_argument("--max-newton-it", type=int, default=100, help="Max Newton iterations (monolithic mode).")
    ap.add_argument(
        "--velocity-bcs",
        action="store_true",
        help="Also prescribe v_y on top/bottom (deal.II does NOT do this for Miehe tension).",
    )

    ap.add_argument("--load-rate", type=float, default=1.0, help="Top displacement rate: u_y(top)=load_rate*t.")
    ap.add_argument("--vtk-every", type=int, default=1, help="Write VTK every N accepted steps (0 disables).")
    ap.add_argument("--csv-every", type=int, default=0, help="Write point CSV every N accepted steps (0 disables).")

    ap.add_argument("--notch-x0", type=float, default=0.5)
    ap.add_argument("--notch-y0", type=float, default=0.5)
    ap.add_argument("--notch-thickness", type=float, default=0.02)

    ap.add_argument("--max-alt-it", type=int, default=3, help="Max staggered (u,v)<->d iterations per time step.")
    ap.add_argument("--alt-tol", type=float, default=1.0e-6, help="Stopping tolerance for staggered iterations on d.")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    L = float(args.L)
    H = float(args.H)
    dt_val = float(args.dt)
    if dt_val <= 0.0:
        raise ValueError("--dt must be positive.")

    # ------------------------------------------------------------------
    # Mesh / spaces
    # ------------------------------------------------------------------
    mesh_file = str(args.mesh_file).strip()
    if mesh_file:
        mesh = mesh_from_gmsh(mesh_file, apply_boundary_tags=True)
        order = int(mesh.poly_order)
    else:
        order = int(args.order)
        nodes, elems, _, corners = structured_quad(L, H, nx=int(args.nx), ny=int(args.ny), poly_order=order)
        mesh = Mesh(
            nodes=nodes,
            element_connectivity=elems,
            elements_corner_nodes=corners,
            element_type="quad",
            poly_order=order,
        )
        _tag_rectangle_boundaries(mesh, L=L, H=H)

    me = MixedElement(
        mesh,
        field_specs={
            "u_x": order,
            "u_y": order,
            "v_x": order,
            "v_y": order,
            "d": order,
        },
    )
    dh = DofHandler(me, method="cg")

    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)
    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)

    du = VectorTrialFunction(space=U, dof_handler=dh)
    dv = VectorTrialFunction(space=V, dof_handler=dh)
    dd = TrialFunction("d", dof_handler=dh)

    u_test = VectorTestFunction(space=U, dof_handler=dh)
    v_test = VectorTestFunction(space=V, dof_handler=dh)
    d_test = TestFunction("d", dof_handler=dh)

    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    d_k = Function("d_k", "d", dof_handler=dh)

    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    d_n = Function("d_n", "d", dof_handler=dh)
    d_nm1 = Function("d_nm1", "d", dof_handler=dh)  # previous-previous (for extrapolation)
    d_mech = Function("d_mech", "d", dof_handler=dh)  # damage seen by mechanics (pf_extra analogue)

    # History field (ψ⁺ max) stored on the d space.
    H_hist = Function("H_hist", "d", dof_handler=dh)
    H_hist.nodal_values.fill(0.0)

    # Augmented Lagrangian multiplier λ (deal.II stores this as a separate vector field,
    # updated explicitly between monolithic Newton solves).
    lambda_penal = Function("lambda_penal", "d", dof_handler=dh)
    lambda_penal.nodal_values.fill(0.0)
    lambda_diff = Function("lambda_diff", "d", dof_handler=dh)
    lambda_diff.nodal_values.fill(0.0)

    # Initial notch / crack seed.
    d_notch_min = _make_initial_notch(
        dof_handler=dh,
        d0=d_n,
        L=L,
        H=H,
        x0=float(args.notch_x0),
        y0=float(args.notch_y0),
        thickness=float(args.notch_thickness),
    )
    d_k.nodal_values[:] = d_n.nodal_values[:]
    d_nm1.nodal_values[:] = d_n.nodal_values[:]
    d_mech.nodal_values[:] = d_n.nodal_values[:]

    # ------------------------------------------------------------------
    # Parameters / measures
    # ------------------------------------------------------------------
    qdeg = int(args.q)
    dx_form = dx(metadata={"q": qdeg})

    rho = Constant(float(args.rho))
    mu_s = Constant(float(args.mu))
    lambda_s = Constant(float(args.lambda_))
    Gc = Constant(float(args.Gc))
    ell = Constant(float(args.ell))
    eta_d = Constant(float(args.eta_d))
    kappa = Constant(float(args.kappa))
    gamma_penal = Constant(float(args.gamma_penal))
    inv_dt = Constant(1.0 / float(dt_val))
    dt = Constant(float(dt_val))
    theta = Constant(float(args.theta))
    one = Constant(1.0)
    one_m_theta = one - theta

    solve_mode = str(args.solve_mode).strip().lower()
    if solve_mode == "monolithic_al" and str(args.mech_d).strip().lower() == "current":
        print("[mode_I] monolithic_al requires lagged/extrapolated mechanics damage (deal.II pf_extra). Forcing --mech-d extrapolated.")
        args.mech_d = "extrapolated"

    # g(d) = (1-κ)(1-d)^2 + κ  (d=0 intact, d=1 broken)
    one_m_d_mech = _one_minus(d_mech)
    g_d_mech = (one - kappa) * (one_m_d_mech * one_m_d_mech) + kappa

    # ------------------------------------------------------------------
    # Mechanics (u,v) with Miehe split: σ = g(d) σ⁺ + σ⁻
    # ------------------------------------------------------------------
    I = Identity(2)
    eta_pos = float(args.eta_pos)
    disc_reg = float(args.disc_reg)
    al_eta = float(args.al_eta)

    def _miehe_split(u, g_coeff):
        E = sym(grad(u))
        E_plus, E_minus, _, _, _ = spectral_positive_part_2x2_sym(E, eta_pos=eta_pos, disc_reg=disc_reg)
        trE = div(u)
        tr_pos = smooth_pos(trE, eta=eta_pos)
        sig_plus = Constant(2.0) * mu_s * E_plus + I * (lambda_s * tr_pos)
        sig_minus = Constant(2.0) * mu_s * E_minus + I * (lambda_s * (trE - tr_pos))
        sigma = g_coeff * sig_plus + sig_minus
        psi_plus = Constant(0.5) * inner(sig_plus, E)
        return E, E_plus, E_minus, trE, tr_pos, sig_plus, sig_minus, sigma, psi_plus

    E_k, E_plus_k, E_minus_k, trE_k, tr_pos_k, sig_plus_k, sig_minus_k, sig_k, psi_plus_k = _miehe_split(u_k, g_d_mech)

    # For deal.II "pf_extra" parity (lagged/extrapolated mechanics coefficient),
    # the old-time stress uses the same extrapolated coefficient.
    if solve_mode == "monolithic_al" and str(args.mech_d).strip().lower() in {"lagged", "extrapolated"}:
        E_n, E_plus_n, E_minus_n, trE_n, tr_pos_n, sig_plus_n, sig_minus_n, sig_n, psi_plus_n = _miehe_split(u_n, g_d_mech)
    else:
        one_m_d_n = _one_minus(d_n)
        g_d_n = (one - kappa) * (one_m_d_n * one_m_d_n) + kappa
        E_n, E_plus_n, E_minus_n, trE_n, tr_pos_n, sig_plus_n, sig_minus_n, sig_n, psi_plus_n = _miehe_split(u_n, g_d_n)

    eps_utest = sym(grad(u_test))

    # Match deal.II dynamic_fracture block structure:
    # - kinematic update tested with the velocity test space
    # - momentum (virtual work) tested with the displacement test space
    # NOTE: deal.II multiplies the balance laws by Δt, i.e. uses (u-u_n) - Δt v_{n+θ} and ρ(v-v_n) - Δt div σ.
    r_kin = inner((u_k - u_n) - dt * (theta * v_k + one_m_theta * v_n), v_test) * dx_form
    r_mom = inner(rho * (v_k - v_n), u_test) * dx_form
    r_mom += dt * theta * inner(sig_k, eps_utest) * dx_form + dt * one_m_theta * inner(sig_n, eps_utest) * dx_form
    r_mech = r_kin + r_mom

    # Jacobian for mechanics (k-part only).
    a_kin = inner(du - dt * theta * dv, v_test) * dx_form

    # Tangent of σ(u) for the split model (d is treated as a coefficient here).
    dE = sym(grad(du))
    dE_plus = d_spectral_positive_part_2x2_sym(E_k, dE, eta_pos=eta_pos, disc_reg=disc_reg)
    dtrE = div(du)
    dtr_pos = smooth_pos_derivative(trE_k, eta=eta_pos) * dtrE

    dsig_plus = Constant(2.0) * mu_s * dE_plus + I * (lambda_s * dtr_pos)
    dsig_minus = Constant(2.0) * mu_s * (dE - dE_plus) + I * (lambda_s * (dtrE - dtr_pos))
    dsig = g_d_mech * dsig_plus + dsig_minus
    a_mom = inner(rho * dv, u_test) * dx_form + dt * theta * inner(dsig, eps_utest) * dx_form

    a_mech = a_kin + a_mom

    # ------------------------------------------------------------------
    # Damage / phase-field (two variants)
    #   (A) staggered: AT2 with history H=max ψ⁺
    #   (B) monolithic_al: deal.II-style monolithic + augmented Lagrangian, driven by instantaneous ψ⁺
    # ------------------------------------------------------------------
    # (A) staggered, history-driven
    r_dmg_hist = (eta_d * (d_k - d_n) * inv_dt) * d_test * dx_form
    r_dmg_hist += (Gc / ell) * d_k * d_test * dx_form
    r_dmg_hist += Gc * ell * inner(grad(d_k), grad(d_test)) * dx_form
    r_dmg_hist += -Constant(2.0) * (one - kappa) * _one_minus(d_k) * H_hist * d_test * dx_form

    a_dmg_hist = (eta_d * dd * inv_dt) * d_test * dx_form
    a_dmg_hist += (Gc / ell) * dd * d_test * dx_form
    a_dmg_hist += Gc * ell * inner(grad(dd), grad(d_test)) * dx_form
    a_dmg_hist += Constant(2.0) * (one - kappa) * H_hist * dd * d_test * dx_form

    # (B) monolithic + AL (matches deal.II dynamic_fracture residual structure)
    one_m_d_k = _one_minus(d_k)
    one_m_d_n = _one_minus(d_n)

    # AL term: enforce irreversibility d^{k} >= d^{n} (i.e. no healing).
    # deal.II uses: max(0, λ + γ (pf - pf_old)), with pf=1-d => pf-pf_old = d_old - d.
    al_arg = lambda_penal + gamma_penal * (d_n - d_k)
    if al_eta > 0.0:
        al_term = smooth_pos(al_arg, eta=al_eta)
        chi_al = smooth_pos_derivative(al_arg, eta=al_eta)
    else:
        al_term = pos_part(al_arg)
        # deal.II convention: use a <0 branch, i.e. H(0)=1 for the Jacobian weight.
        chi_al = one - heaviside(-al_arg)

    # Driving energy density ψ⁺ (instantaneous; no max-history in deal.II)
    drive_k = -Constant(2.0) * (one - kappa) * one_m_d_k * psi_plus_k
    drive_n = -Constant(2.0) * (one - kappa) * one_m_d_n * psi_plus_n

    # NOTE: d is the complement of deal.II's phase-field variable pf=1-d.
    # The irreversibility constraint is d^k >= d^n, so the AL multiplier enters
    # the d-weak form with a negative sign (∂g/∂d=-1 for g=d^n-d^k).
    r_dmg_al = (-al_term * d_test) * dx_form
    r_dmg_al += dt * theta * ((Gc / ell) * d_k * d_test + Gc * ell * inner(grad(d_k), grad(d_test)) + drive_k * d_test) * dx_form
    r_dmg_al += dt * one_m_theta * ((Gc / ell) * d_n * d_test + Gc * ell * inner(grad(d_n), grad(d_test)) + drive_n * d_test) * dx_form

    # Gateaux derivative of ψ⁺(u) used in the monolithic tangent.
    delta_psi_plus = Constant(0.5) * (inner(dsig_plus, E_k) + inner(sig_plus_k, dE))

    a_dmg_al = chi_al * gamma_penal * dd * d_test * dx_form
    a_dmg_al += dt * theta * ((Gc / ell) * dd * d_test + Gc * ell * inner(grad(dd), grad(d_test))) * dx_form
    a_dmg_al += dt * theta * (
        (Constant(2.0) * (one - kappa) * psi_plus_k) * dd * d_test
        - Constant(2.0) * (one - kappa) * one_m_d_k * delta_psi_plus * d_test
    ) * dx_form

    # ------------------------------------------------------------------
    # Boundary conditions (mode-I tension)
    # ------------------------------------------------------------------
    load_rate = float(args.load_rate)

    def _uy_top(_x, _y, t=0.0):
        return load_rate * float(t)

    def _vy_top(_x, _y, t=0.0):
        return load_rate

    bcs = [
        BoundaryCondition("u_y", "dirichlet", "bottom", lambda x, y, t=0.0: 0.0),
        BoundaryCondition("u_y", "dirichlet", "top", _uy_top),
    ]
    if bool(args.velocity_bcs):
        # Optional: keep velocity BCs consistent with prescribed displacement.
        bcs.extend(
            [
                BoundaryCondition("v_y", "dirichlet", "bottom", lambda x, y, t=0.0: 0.0),
                BoundaryCondition("v_y", "dirichlet", "top", _vy_top),
            ]
        )
    bcs_homog = [
        BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y, t=0.0: 0.0) for bc in bcs
    ]

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------
    mech_solver = None
    dmg_solver = None
    mono_solver = None
    if solve_mode == "staggered":
        mech_solver = NewtonSolver(
            r_mech,
            a_mech,
            dof_handler=dh,
            mixed_element=me,
            bcs=bcs,
            bcs_homog=bcs_homog,
            newton_params=NewtonParameters(newton_tol=1.0e-7, max_newton_iter=30, ls_mode="dealii"),
            lin_params=LinearSolverParameters(backend="scipy"),
            quad_order=qdeg,
            backend=str(args.backend),
        )

        dmg_solver = NewtonSolver(
            r_dmg_hist,
            a_dmg_hist,
            dof_handler=dh,
            mixed_element=me,
            bcs=[],
            bcs_homog=[],
            newton_params=NewtonParameters(newton_tol=1.0e-10, max_newton_iter=10, ls_mode="dealii"),
            lin_params=LinearSolverParameters(backend="scipy"),
            quad_order=qdeg,
            backend=str(args.backend),
        )
    elif solve_mode == "monolithic_al":
        r_total = r_mech + r_dmg_al
        a_total = a_mech + a_dmg_al
        mono_solver = NewtonSolver(
            r_total,
            a_total,
            dof_handler=dh,
            mixed_element=me,
            bcs=bcs,
            bcs_homog=bcs_homog,
            newton_params=NewtonParameters(
                newton_tol=float(args.newton_tol),
                max_newton_iter=int(args.max_newton_it),
                line_search=True,
                ls_mode="dealii",
            ),
            lin_params=LinearSolverParameters(backend="scipy"),
            quad_order=qdeg,
            backend=str(args.backend),
        )
    else:
        raise ValueError(f"Unknown --solve-mode {solve_mode!r}.")

    # ------------------------------------------------------------------
    # Time loop
    # ------------------------------------------------------------------
    t = 0.0
    step = 0

    vtk_every = int(args.vtk_every)
    if vtk_every > 0:
        export_vtk(
            filename=os.path.join(str(outdir), f"solution_{step:04d}.vtu"),
            mesh=mesh,
            dof_handler=dh,
            functions={"u": u_n, "v": v_n, "d": d_n, "H": H_hist},
        )
    csv_every = int(args.csv_every)
    if csv_every > 0:
        _write_point_csv(
            filename=os.path.join(str(outdir), f"points_{step:06d}.csv"),
            dof_handler=dh,
            u=u_n,
            v=v_n,
            d=d_n,
        )

    while t < float(args.t_final) - 1.0e-15:
        t_next = t + dt_val
        t_bc = t_next if float(args.theta) == 1.0 else (t + float(args.theta) * dt_val)

        # Predictor: start from previous step.
        u_k.nodal_values[:] = u_n.nodal_values[:]
        v_k.nodal_values[:] = v_n.nodal_values[:]
        d_k.nodal_values[:] = d_n.nodal_values[:]

        # Freeze time-dependent BCs for this step.
        bcs_now = NewtonSolver._freeze_bcs(bcs, t_bc)
        dh.apply_bcs(bcs_now, u_k, v_k)

        # deal.II-style lagging/extrapolation of the mechanics damage coefficient.
        mech_d_mode = str(args.mech_d)
        if mech_d_mode == "lagged":
            d_mech.nodal_values[:] = d_n.nodal_values[:]
        elif mech_d_mode == "extrapolated":
            d_mech.nodal_values[:] = np.clip(2.0 * d_n.nodal_values - d_nm1.nodal_values, 0.0, 1.0)

        if solve_mode == "staggered":
            # Staggered (alternate minimization) solve.
            d_prev_iter = d_k.nodal_values.copy()
            for _alt in range(int(args.max_alt_it)):
                # (1) mechanics for (u,v) with fixed d
                if mech_d_mode == "current":
                    d_mech.nodal_values[:] = d_k.nodal_values[:]
                dh.apply_bcs(bcs_now, u_k, v_k)
                _delta, converged, _nits = mech_solver._newton_loop(
                    [u_k, v_k], [u_n, v_n], {"d_mech": d_mech, "d_n": d_n}, bcs_now
                )
                if not converged:
                    raise RuntimeError("Mechanical Newton did not converge.")

                # (2) update history H = max(H, ψ⁺(u))
                psi_nodes = _tensile_energy_mass_lumped_on_d(
                    dof_handler=dh,
                    u=u_k,
                    dx_domain=dx_form,
                    mu_s=mu_s,
                    lambda_s=lambda_s,
                    eta_pos=eta_pos,
                    disc_reg=disc_reg,
                    backend=str(args.backend),
                    quad_order=qdeg,
                )
                if str(args.history) == "max":
                    H_hist.nodal_values[:] = np.maximum(H_hist.nodal_values, psi_nodes)
                else:
                    H_hist.nodal_values[:] = psi_nodes

                # (3) damage solve for d with fixed H
                _delta, converged, _nits = dmg_solver._newton_loop([d_k], [d_n], {"H_hist": H_hist}, [])
                if not converged:
                    raise RuntimeError("Damage Newton did not converge.")

                # (4) irreversibility & bounds: d^{n+1} >= d^n, plus keep initial notch
                d_k.nodal_values[:] = np.maximum(d_k.nodal_values, d_n.nodal_values)
                d_k.nodal_values[:] = np.maximum(d_k.nodal_values, d_notch_min)
                d_k.nodal_values[:] = np.clip(d_k.nodal_values, 0.0, 1.0)

                d_change = float(np.max(np.abs(d_k.nodal_values - d_prev_iter)))
                d_prev_iter[:] = d_k.nodal_values
                if d_change < float(args.alt_tol):
                    break
        else:
            # deal.II-style augmented Lagrangian loop + monolithic Newton.
            # Reset λ at the start of each *time step*.
            lambda_penal.nodal_values.fill(0.0)
            gamma_val = float(args.gamma_penal)
            al_tol_abs = float(args.al_tol_abs)
            al_tol_rel = float(args.al_tol_rel)
            al_max_it = int(args.al_max_it)

            rms0 = None
            for al_it in range(al_max_it):
                dh.apply_bcs(bcs_now, u_k, v_k)
                _delta, converged, _nits = mono_solver._newton_loop(
                    [u_k, v_k, d_k],
                    [u_n, v_n, d_n],
                    {"d_mech": d_mech, "lambda_penal": lambda_penal},
                    bcs_now,
                )
                if not converged:
                    raise RuntimeError("Monolithic Newton did not converge.")

                lam_old = lambda_penal.nodal_values.copy()
                lambda_penal.nodal_values[:] = np.maximum(
                    0.0, lambda_penal.nodal_values + gamma_val * (d_n.nodal_values - d_k.nodal_values)
                )

                dl = lambda_penal.nodal_values - lam_old
                rms = float(np.linalg.norm(dl) / np.sqrt(dl.size))
                if rms0 is None:
                    rms0 = max(rms, 1.0e-30)
                rel = float(rms / rms0)
                print(
                    f"[mode_I][AL] it={al_it+1} rms(dl)={rms:.3e} rel={rel:.3e} "
                    f"lambda[max]={float(np.max(lambda_penal.nodal_values)):.3e}"
                )
                if (rms <= al_tol_abs) or (rel <= al_tol_rel):
                    break

        # Accept: promote current -> previous.
        u_n.nodal_values[:] = u_k.nodal_values[:]
        v_n.nodal_values[:] = v_k.nodal_values[:]
        d_nm1.nodal_values[:] = d_n.nodal_values[:]
        d_n.nodal_values[:] = d_k.nodal_values[:]

        t = t_next
        step += 1
        print(f"[mode_I] step={step} t={t:.6e} d[max]={float(np.max(d_n.nodal_values)):.3e}")

        if vtk_every > 0 and (step % vtk_every) == 0:
            export_vtk(
                filename=os.path.join(str(outdir), f"solution_{step:04d}.vtu"),
                mesh=mesh,
                dof_handler=dh,
                functions={"u": u_n, "v": v_n, "d": d_n, "H": H_hist},
            )
        if csv_every > 0 and (step % csv_every) == 0:
            _write_point_csv(
                filename=os.path.join(str(outdir), f"points_{step:06d}.csv"),
                dof_handler=dh,
                u=u_n,
                v=v_n,
                d=d_n,
            )


if __name__ == "__main__":
    main()
