#!/usr/bin/env python3
"""
Finite-difference consistency check for a coupled FSI + semi-smooth contact system.

This script assembles a *trimmed* monolithic fully Eulerian FSI residual/Jacobian
and adds the relaxed contact term from `examples.utils.fsi.contact`:

  R(U) += γ_C ⟨P(U)⟩₊ (v_s · n_s) on Γ_i
  A(U) += γ_C H(P(U)) P'(U)[δU] (v_s · n_s) on Γ_i

and verifies the directional derivative:

  A(U)·δ ≈ (R(U + ε δ) - R(U)) / ε.

It is meant as a regression tool for:
  - semi-smooth Newton linearization (PositivePart + Heaviside),
  - mixed +/− restrictions on moving-interface CutFEM domains,
  - backend parity (python / jit / cpp).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import CircleLevelSet
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.meshgen import structured_quad

from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    FacetNormal,
    Function,
    TrialFunction,
    TestFunction,
    VectorFunction,
    VectorTrialFunction,
    VectorTestFunction,
    ElementWiseConstant,
    Pos,
    Neg,
    restrict,
    Identity,
)
from pycutfem.solvers.nonlinear_solver import NewtonSolver, NewtonParameters

from examples.utils.fsi.contact import (
    RelaxedWallContact,
    sigma_f_newtonian,
    dsigma_f_newtonian,
    sigma_s_stvk,
    dsigma_s_stvk,
)
from pycutfem.ufl.analytic import Analytic, y

from examples.utils.fsi.fully_eulerian import (
    make_domain_sets,
    build_measures,
    build_extension_measures,
    hansbo_kappa,
    build_fsi_eulerian_forms,
)


@dataclass(frozen=True)
class Case:
    label: str
    wall_eps: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FD check for coupled FSI+contact (semi-smooth Newton).")
    p.add_argument("--backends", type=str, default="python,jit,cpp", help="Comma-separated list (or 'all').")
    p.add_argument("--nx", type=int, default=4)
    p.add_argument("--ny", type=int, default=4)
    p.add_argument("--quad-order", type=int, default=4)
    p.add_argument("--eps-fd", type=float, default=1.0e-7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nonlinear-solid", action="store_true", help="Use nonlinear StVK (paper) instead of linear.")
    p.add_argument("--paper-terms", action="store_true", help="Enable paper-style SUPG + implicit extension + Hessian ghost + w(kappa) weights.")
    p.add_argument("--wmax", type=float, default=2.0, help="Paper ghost weight parameter w_max (used when --paper-terms).")
    p.add_argument("--no-stab", action="store_true", help="Skip ghost-penalty stabilization terms (Eq. 13).")
    p.add_argument("--no-extension", action="store_true", help="Disable extension domains and extension ghost penalties (Eq. 14).")
    p.add_argument("--no-ext", action="store_true", help="Skip extension ghost penalty terms but keep extension domains (only relevant if extension is on).")
    p.add_argument("--no-supg", action="store_true", help="Skip SUPG artificial diffusion terms (Sec. 3.3).")
    p.add_argument("--no-hessian-ghost", action="store_true", help="Disable the Q2 fluid Hessian ghost penalty (Eq. 13, g_F^2 term).")
    p.add_argument("--no-weights", action="store_true", help="Disable the cut-dependent weight function w(kappa).")
    p.add_argument("--no-u-mom", action="store_true", help="Disable the optional displacement-in-momentum stabilization (paper Eq. 13).")
    p.add_argument("--no-contact", action="store_true", help="Skip adding the relaxed wall contact term (isolate FSI/stabilizations).")
    return p.parse_args()


def _fd_check(*, backend: str, case: Case, args: argparse.Namespace) -> tuple[float, float, int]:
    # Geometry (paper Sec. 6)
    x0, x1 = -0.04, 0.04
    y0, y1 = 0.0, 0.08
    r = 0.011
    h0 = 0.039
    center = (0.0, h0 + r)

    # Mesh
    nodes, elems, _, corners = structured_quad(
        x1 - x0,
        y1 - y0,
        nx=int(args.nx),
        ny=int(args.ny),
        poly_order=2,
        offset=(x0, y0),
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )

    # Level set (solid = φ<0)
    level_set = CircleLevelSet(center=center, radius=r)
    mesh.classify_elements(level_set, tol=1.0e-12)
    mesh.classify_edges(level_set, tol=1.0e-12)
    mesh.build_interface_segments(level_set, tol=1.0e-12)

    paper_terms = bool(args.paper_terms)
    use_extension = paper_terms and (not bool(args.no_extension))
    extension_layers = 1 if use_extension else 0
    domains = make_domain_sets(mesh, extension_layers=extension_layers)
    dx_f, dx_s, dΓ, dG_f, dG_s = build_measures(mesh, level_set, domains, qvol=int(args.quad_order))
    dG_f_ext = None
    dG_s_ext = None
    if use_extension and (not bool(args.no_ext)):
        dG_f_ext, dG_s_ext = build_extension_measures(mesh, level_set, domains, qvol=int(args.quad_order))

    # Hansbo weights
    theta_pos_vals, theta_neg_vals = hansbo_kappa(mesh, level_set, theta_min=1.0e-3)
    theta_pos = ElementWiseConstant(theta_pos_vals)
    theta_neg = ElementWiseConstant(theta_neg_vals)
    theta_sum = Pos(theta_pos) + Neg(theta_neg) + Constant(1.0e-12)
    kappa_pos = Pos(theta_pos) / theta_sum
    kappa_neg = Neg(theta_neg) / theta_sum

    w_fluid = None
    w_solid = None
    if paper_terms and (not bool(args.no_weights)):
        wmax = max(float(args.wmax), 1.0)
        w_fluid_vals = 0.5 * (wmax ** (1.0 - 2.0 * theta_pos_vals))
        w_solid_vals = 0.5 * (wmax ** (1.0 - 2.0 * theta_neg_vals))
        w_fluid = ElementWiseConstant(w_fluid_vals)
        w_solid = ElementWiseConstant(w_solid_vals)

    # Unknowns: (u_f, p_f, v_s, d_s)
    deg_uf, deg_pf, deg_us, deg_ds = 2, 1, 1, 1
    fields = {
        "u_pos_x": deg_uf,
        "u_pos_y": deg_uf,
        "p_pos_": deg_pf,
        "vs_neg_x": deg_us,
        "vs_neg_y": deg_us,
        "d_neg_x": deg_ds,
        "d_neg_y": deg_ds,
    }
    me = MixedElement(mesh, field_specs=fields)
    dh = DofHandler(me, method="cg")

    # Spaces (sided)
    Vf = FunctionSpace("Vf", ["u_pos_x", "u_pos_y"], side="+")
    Pf = FunctionSpace("Pf", ["p_pos_"], side="+")
    Vs = FunctionSpace("Vs", ["vs_neg_x", "vs_neg_y"], side="-")
    Ds = FunctionSpace("Ds", ["d_neg_x", "d_neg_y"], side="-")

    du_f = VectorTrialFunction(Vf, dof_handler=dh)
    dp_f = TrialFunction(name="dp_f", field_name="p_pos_", dof_handler=dh, side="+")
    du_s = VectorTrialFunction(Vs, dof_handler=dh)
    dd_s = VectorTrialFunction(Ds, dof_handler=dh)

    tv_f = VectorTestFunction(Vf, dof_handler=dh)
    tq_f = TestFunction(name="q_f", field_name="p_pos_", dof_handler=dh, side="+")
    tv_s = VectorTestFunction(Vs, dof_handler=dh)
    td_s = VectorTestFunction(Ds, dof_handler=dh)

    uf_k = VectorFunction("uf_k", ["u_pos_x", "u_pos_y"], dh)
    pf_k = Function("pf_k", "p_pos_", dh)
    vs_k = VectorFunction("vs_k", ["vs_neg_x", "vs_neg_y"], dh)
    ds_k = VectorFunction("ds_k", ["d_neg_x", "d_neg_y"], dh)

    uf_n = VectorFunction("uf_n", ["u_pos_x", "u_pos_y"], dh)
    pf_n = Function("pf_n", "p_pos_", dh)
    vs_n = VectorFunction("vs_n", ["vs_neg_x", "vs_neg_y"], dh)
    ds_n = VectorFunction("ds_n", ["d_neg_x", "d_neg_y"], dh)

    rng = np.random.default_rng(int(args.seed))
    for f in (uf_k, pf_k, vs_k, ds_k):
        f.nodal_values[:] = 1.0e-2 * rng.standard_normal(f.nodal_values.shape)
    for f in (uf_n, pf_n, vs_n, ds_n):
        f.nodal_values[:] = 0.0

    # Restrict to active domains (mirrors the bouncing_ball example)
    fluid_active = domains.get("has_pos_ext") if use_extension else domains["has_pos"]
    solid_active = domains.get("has_neg_ext") if use_extension else domains["has_neg"]

    du_f_R = restrict(du_f, fluid_active)
    dp_f_R = restrict(dp_f, fluid_active)
    tv_f_R = restrict(tv_f, fluid_active)
    tq_f_R = restrict(tq_f, fluid_active)
    uf_k_R = restrict(uf_k, fluid_active)
    uf_n_R = restrict(uf_n, fluid_active)
    pf_k_R = restrict(pf_k, fluid_active)
    pf_n_R = restrict(pf_n, fluid_active)

    du_s_R = restrict(du_s, solid_active)
    dd_s_R = restrict(dd_s, solid_active)
    tv_s_R = restrict(tv_s, solid_active)
    td_s_R = restrict(td_s, solid_active)
    vs_k_R = restrict(vs_k, solid_active)
    vs_n_R = restrict(vs_n, solid_active)
    ds_k_R = restrict(ds_k, solid_active)
    ds_n_R = restrict(ds_n, solid_active)

    # Parameters (mostly paper values; only those needed for assembly)
    dt = Constant(1.0e-4)
    theta = Constant(1.0)
    cell_h = CellDiameter()

    rho_f = Constant(1141.0)
    nu_f = Constant(7.0114e-5)
    mu_f = rho_f * nu_f

    rho_s = Constant(1361.0)
    mu_s = Constant(20.0e3)
    lambda_s = Constant(80.0e3)

    gamma_N = Constant(1.0e7)
    beta_N = Constant(1.0) * gamma_N  # build_fsi_eulerian_forms expects "beta_N" multiplier

    # Build baseline FSI terms (trimmed subset by default; paper_terms adds stabilizations)
    use_stab = paper_terms and (not bool(args.no_stab))
    use_ext = paper_terms and use_extension and (not bool(args.no_ext))
    use_supg = paper_terms and (not bool(args.no_supg))
    use_hessian = paper_terms and (not bool(args.no_hessian_ghost))
    use_u_mom = paper_terms and (not bool(args.no_u_mom)) and use_stab

    # Paper parameters: γ_{v_f}=0.5, γ_{v_s}=γ_p=γ_u=0.1
    gamma_v = Constant(0.5) if use_stab else Constant(0.0)
    gamma_v_s = Constant(0.1) if use_stab else Constant(0.0)
    gamma_p = Constant(0.1) if use_stab else Constant(0.0)
    gamma_v_grad = Constant(0.1) if use_stab else Constant(0.0)
    supg_delta0 = Constant(1.0e-5) if use_supg else None
    gamma_v_ext = Constant(10.0) if use_ext else None
    gamma_p_ext = Constant(10.0) if use_ext else None
    gamma_vs_ext = Constant(10.0) if use_ext else None
    gamma_u_ext = Constant(10.0) if use_ext else None
    gamma_u_psi_ext = Constant(1.0e-4) if use_ext else None

    terms = build_fsi_eulerian_forms(
        du_f=du_f_R,
        dp_f=dp_f_R,
        du_s=du_s_R,
        ddisp_s=dd_s_R,
        test_vel_f=tv_f_R,
        test_q_f=tq_f_R,
        test_vel_s=tv_s_R,
        test_disp_s=td_s_R,
        uf_k=uf_k_R,
        pf_k=pf_k_R,
        uf_n=uf_n_R,
        pf_n=pf_n_R,
        us_k=vs_k_R,
        us_n=vs_n_R,
        disp_k=ds_k_R,
        disp_n=ds_n_R,
        dx_fluid=dx_f,
        dx_solid=dx_s,
        dGamma=dΓ,
        dG_fluid=dG_f,
        dG_solid=dG_s,
        kappa_pos=kappa_pos,
        kappa_neg=kappa_neg,
        cell_h=cell_h,
        beta_N=beta_N,
        rho_f=rho_f,
        rho_s=rho_s,
        mu_f=mu_f,
        mu_s=mu_s,
        lambda_s=lambda_s,
        dt=dt,
        theta=theta,
        gamma_v=gamma_v,
        gamma_v_s=gamma_v_s,
        gamma_p=gamma_p,
        gamma_v_grad=gamma_v_grad,
        gamma_u_mom=gamma_v_grad if use_u_mom else None,
        w_fluid=w_fluid,
        w_solid=w_solid,
        dG_fluid_ext=dG_f_ext,
        dG_solid_ext=dG_s_ext,
        gamma_v_ext=gamma_v_ext,
        gamma_p_ext=gamma_p_ext,
        gamma_vs_ext=gamma_vs_ext,
        gamma_u_ext=gamma_u_ext,
        gamma_u_psi_ext=gamma_u_psi_ext,
        supg_delta0_vs=supg_delta0,
        supg_delta0_u=supg_delta0,
        fluid_hessian_ghost=bool(use_hessian),
        solid_reg_eps=Constant(0.0),
        use_linear_solid=not bool(args.nonlinear_solid),
        solid_stvk_paper=True,
        solid_advect_lagged=False,
        s_nitsche_value=0.0,  # keep the interface term minimal for this check
        interface_form="paper",
    )

    residual_form = terms.r_vol_f + terms.R_int + terms.r_vol_s + terms.r_svc
    jacobian_form = terms.a_vol_f + terms.J_int + terms.a_vol_s + terms.a_svc
    if use_stab:
        residual_form = residual_form + terms.r_stab
        jacobian_form = jacobian_form + terms.a_stab
    if use_ext:
        if terms.r_ext is not None:
            residual_form = residual_form + terms.r_ext
        if terms.a_ext is not None:
            jacobian_form = jacobian_form + terms.a_ext
    if use_supg:
        if terms.r_supg is not None:
            residual_form = residual_form + terms.r_supg
        if terms.a_supg is not None:
            jacobian_form = jacobian_form + terms.a_supg

    if not bool(args.no_contact):
        # Add relaxed wall contact (paper Eq. (21) + semi-smooth Jacobian)
        n_s = FacetNormal()
        n_f = Constant(-1.0) * n_s
        penalty_nitsche = rho_f * nu_f * gamma_N / cell_h

        # Paper sets γ_C := γ_C^0 μ_s / h with γ_C^0 = 500.
        gamma_C0 = Constant(500.0)
        gamma_C = gamma_C0 * mu_s / cell_h

        gap_eps = Analytic(y) - Constant(float(case.wall_eps))
        contact = RelaxedWallContact(
            gamma_C=gamma_C,
            gap_eps=gap_eps,
            n_s=n_s,
            n_f=n_f,
            penalty_nitsche=penalty_nitsche,
        )

        sigma_f_k = sigma_f_newtonian(uf_k_R, pf_k_R, rho_f=rho_f, nu_f=nu_f)
        dsigma_f = dsigma_f_newtonian(du_f_R, dp_f_R, rho_f=rho_f, nu_f=nu_f)
        sigma_s_k = sigma_s_stvk(ds_k_R, mu_s=mu_s, lambda_s=lambda_s)
        dsigma_s = dsigma_s_stvk(ds_k_R, dd_s_R, mu_s=mu_s, lambda_s=lambda_s)

        P = contact.P_gammaC(
            u=ds_k_R,
            u_prev=ds_n_R,
            v_f=uf_k_R,
            p_f=pf_k_R,
            v_s=vs_k_R,
            sigma_s=sigma_s_k,
            sigma_f=sigma_f_k,
        )
        dP = contact.dP_gammaC(
            du=dd_s_R,
            dv_f=du_f_R,
            dp_f=dp_f_R,
            dv_s=du_s_R,
            dsigma_s=dsigma_s,
            dsigma_f=dsigma_f,
        )

        residual_form = residual_form + contact.residual_term(P=P, k=Constant(1.0), test_v_s=tv_s_R) * dΓ
        jacobian_form = jacobian_form + contact.jacobian_term(P=P, dP=dP, k=Constant(1.0), test_v_s=tv_s_R) * dΓ

    solver = NewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(newton_tol=1.0e-14, line_search=False),
        backend=str(backend),
        quad_order=int(args.quad_order),
    )

    funcs = [uf_k, pf_k, vs_k, ds_k]
    coeffs = {f.name: f for f in funcs}
    coeffs.update({uf_n.name: uf_n, pf_n.name: pf_n, vs_n.name: vs_n, ds_n.name: ds_n})

    A, R = solver._assemble_system(coeffs, need_matrix=True)  # pylint: disable=protected-access

    ndof = dh.total_dofs
    delta = rng.standard_normal(ndof)
    delta *= 1.0e-3 / (np.linalg.norm(delta, np.inf) + 1.0e-16)

    eps_fd = float(args.eps_fd)
    snap = [f.nodal_values.copy() for f in funcs]
    try:
        dh.add_to_functions(eps_fd * delta, funcs)
        A1, R1 = solver._assemble_system(coeffs, need_matrix=False)  # type: ignore[assignment]
        assert A1 is None
    finally:
        for f, buf in zip(funcs, snap):
            f.nodal_values[:] = buf

    dR_fd = (R1 - R) / eps_fd
    dR_lin = A @ delta

    err = np.linalg.norm(dR_fd - dR_lin, np.inf)
    ref = max(1.0, np.linalg.norm(dR_fd, np.inf), np.linalg.norm(dR_lin, np.inf))
    rel = err / ref
    return float(err), float(rel), int(ndof)


def main() -> None:
    args = _parse_args()
    if args.backends.strip().lower() == "all":
        backends = ["python", "jit", "cpp"]
    else:
        backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    cases = (
        Case("inactive (P<0)", wall_eps=1.0e-4),
        Case("active (P>0)", wall_eps=1.0e-1),
    )

    for case in cases:
        print(f"\n--- {case.label}: wall_eps={case.wall_eps:g} ---")
        for backend in backends:
            try:
                err, rel, ndof = _fd_check(backend=backend, case=case, args=args)
            except Exception as exc:  # pragma: no cover - debug script
                print(f"[{backend:6}] FAILED: {exc}")
                continue
            print(f"[{backend:6}] ndof={ndof:5d}  |FD - A·d|_inf={err:.3e}  rel={rel:.3e}")


if __name__ == "__main__":
    main()
