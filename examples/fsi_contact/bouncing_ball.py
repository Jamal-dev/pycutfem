#!/usr/bin/env python3
"""
Fully Eulerian fluid–structure interaction with relaxed wall contact (CutFEM).

Implements the semi-smooth Newton contact term from:
  "Numerical simulations of fully Eulerian fluid-structure contact interaction using a
   ghost-penalty cut finite element approach" (Frei et al., ACSE 2025),
see `examples/fsi_contact/paper/main.tex`, Problems 5–7.

This example focuses on:
  - consistent residual/Jacobian assembly for the contact term (PositivePart + Heaviside),
  - integration into a monolithic CutFEM FSI setup,
  - a moving level-set updated via φ(x) = φ0(x - d(x)) on (inside∪cut) nodes.

Notes
-----
- The full paper setup includes additional extension terms and an adaptive time-step controller.
  This example implements the implicit extension terms (paper Sec. 3.4). A lightweight subset of
  the paper’s adaptive time-step heuristic is available via `--adaptive-dt` (Newton-failure refine +
  10-step coarsen), but it does not reproduce the full rollback logic from Sec. 6.4.
- The discrete forms are written in a time-divided algebraic form (with `1/dt` mass terms).
  Before handing them to the Newton solver, we rescale residual and Jacobian by `dt` so the
  absolute Newton tolerance matches the paper’s (unscaled) formulation (Problem 5).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import json

from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import CircleLevelSet, LevelSetGridFunction
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
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
    Pos,
    Neg,
    ElementWiseConstant,
    restrict,
    grad,
    dot,
    inner,
    div,
    Identity,
)
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.solvers.nonlinear_solver import NewtonSolver, NewtonParameters, TimeStepperParameters

from examples.utils.fsi.contact import (
    RelaxedWallContact,
    sigma_f_newtonian,
    dsigma_f_newtonian,
    green_lagrange_strain,
    sigma_s_stvk as sigma_s_stvk_paper,
    dsigma_s_stvk as dsigma_s_stvk_paper,
)
from pycutfem.ufl.analytic import Analytic, y
from pycutfem.utils.functionals import NamedFunctionalEvaluator

from examples.utils.fsi.fully_eulerian import (
    make_domain_sets,
    refresh_domain_sets,
    build_measures,
    build_extension_measures,
    hansbo_kappa,
    retag_inactive,
    recompute_active_dofs,
    extend_newly_active_dofs_nearest,
    build_fsi_eulerian_forms,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bouncing ball: fully Eulerian FSI + relaxed wall contact.")
    p.add_argument("--nx", type=int, default=int(os.getenv("NX", "32")), help="Elements in x (uniform). Env: NX.")
    p.add_argument("--ny", type=int, default=int(os.getenv("NY", "32")), help="Elements in y (uniform). Env: NY.")
    p.add_argument("--dt", type=float, default=float(os.getenv("DT", "1e-4")), help="Time step size. Env: DT.")
    p.add_argument(
        "--adaptive-dt",
        action="store_true",
        help="Enable paper-style adaptive dt (refine on Newton failure; coarsen after 10 accepted steps without refinement).",
    )
    p.add_argument(
        "--dt-min",
        type=float,
        default=float(os.getenv("DT_MIN", "1e-8")),
        help="Minimum dt allowed when --adaptive-dt is active. Env: DT_MIN.",
    )
    p.add_argument("--final-time", type=float, default=float(os.getenv("FINAL_TIME", "0.01")), help="Final time.")
    p.add_argument("--backend", type=str, default=os.getenv("PYCUTFEM_JIT_BACKEND", "jit"), help="python|jit|cpp")
    p.add_argument("--quad-order", type=int, default=int(os.getenv("QUAD_ORDER", "6")), help="Quadrature order.")
    p.add_argument("--newton-tol", type=float, default=float(os.getenv("NEWTON_TOL", "1e-7")), help="Newton tol.")
    p.add_argument(
        "--max-newton-iter",
        type=int,
        default=int(os.getenv("MAX_NEWTON_ITER", "50")),
        help="Maximum Newton iterations per time step (paper reports averages ~1–2; higher max helps robustness).",
    )
    p.add_argument(
        "--newton-print-level",
        type=int,
        default=int(os.getenv("NEWTON_PRINT_LEVEL", "0")),
        help="NewtonSolver verbosity: 0=quiet, 1=per-step, 2=per-Newton, 3=line-search. Env: NEWTON_PRINT_LEVEL.",
    )
    p.add_argument("--no-line-search", action="store_true", help="Disable line search.")
    p.add_argument("--no-extension", action="store_true", help="Disable implicit extension ghost penalties.")
    p.add_argument("--no-supg", action="store_true", help="Disable SUPG convection stabilization in the solid.")
    p.add_argument("--no-hessian-ghost", action="store_true", help="Disable 2nd-derivative ghost penalty for Q2 fluid vel.")
    p.add_argument("--wmax", type=float, default=float(os.getenv("WMAX", "2.0")), help="Paper ghost weight parameter w_max.")
    p.add_argument("--save-vtk", action="store_true", help="Write VTK output (slow).")
    p.add_argument(
        "--vtk-every",
        type=int,
        default=int(os.getenv("VTK_EVERY", "0")),
        help="Write VTK every N accepted steps (0 disables). Env: VTK_EVERY. Implies --save-vtk.",
    )
    p.add_argument("--out-dir", type=str, default=os.getenv("OUT_DIR", "output_bouncing_ball"), help="Output dir.")
    p.add_argument("--no-metrics", action="store_true", help="Disable paper quantities-of-interest tracking.")
    p.add_argument(
        "--metrics-stride",
        type=int,
        default=int(os.getenv("METRICS_STRIDE", "10")),
        help="Compute expensive energies every N steps. Env: METRICS_STRIDE.",
    )
    p.add_argument(
        "--log-stride",
        type=int,
        default=int(os.getenv("LOG_STRIDE", "1")),
        help="Print step summary every N accepted steps (set >1 for long runs). Env: LOG_STRIDE.",
    )
    return p.parse_args()


def _min_interface_y(mesh: Mesh) -> float:
    """Approximate min y-coordinate of Γ by scanning cut-element interface segments."""
    y_min = np.inf
    for elem in mesh.elements_list:
        if getattr(elem, "tag", "") != "cut":
            continue
        segs = getattr(elem, "interface_segments", []) or []
        for seg in segs:
            for pt in seg:
                y_min = min(y_min, float(np.asarray(pt, float)[1]))
    return float(y_min) if np.isfinite(y_min) else float("nan")


def _nearest_field_dof(dh: DofHandler, field: str, xy: np.ndarray) -> int:
    coords = np.asarray(dh.get_dof_coords(field), dtype=float)
    gdofs = np.asarray(dh.get_field_slice(field), dtype=int)
    if coords.size == 0 or gdofs.size == 0:
        raise RuntimeError(f"No DOFs found for field '{field}'.")
    diff = coords - np.asarray(xy, dtype=float).reshape(1, 2)
    idx = int(np.argmin(np.einsum("ij,ij->i", diff, diff)))
    return int(gdofs[idx])


def update_ball_levelset_from_displacement(
    ls_ball: LevelSetGridFunction,
    disp_vec: VectorFunction,
    ref_ls: CircleLevelSet,
    *,
    phi_disp_cutoff: float = 0.0,
    tol_commit: float = 1e-12,
) -> None:
    """
    Update FE level set by nodal advection:
      φ_new(x_node) = φ0(x_node - d(x_node)).

    Displacement is only sampled on nodes that belong to inside/cut elements and
    are not "far" on the positive side (phi <= phi_disp_cutoff) to avoid using
    uncontrolled values in the fluid-only region.
    """
    dh_ls = ls_ball.dh
    mesh = dh_ls.mixed_element.mesh
    disp_dh = disp_vec._dof_handler

    gphi = np.asarray(dh_ls.get_field_slice(ls_ball.field), dtype=int)
    node_ids = np.array([dh_ls._dof_to_node_map[int(gd)][1] for gd in gphi], dtype=int)
    phi_vals = ls_ball.nodal_values()
    phi_new_vals = np.empty_like(phi_vals)

    n_nodes = mesh.nodes_x_y_pos.shape[0]
    phi_by_node = np.full(n_nodes, np.nan, dtype=float)
    phi_by_node[node_ids] = phi_vals

    disp_x = disp_vec.components[0]
    disp_y = disp_vec.components[1]
    fld_x = disp_x.field_name
    fld_y = disp_y.field_name

    li_x = np.full(n_nodes, -1, dtype=int)
    li_y = np.full(n_nodes, -1, dtype=int)
    g2l = getattr(disp_vec, "_g2l", {}) or {}
    for nid, gd in disp_dh.dof_map.get(fld_x, {}).items():
        li = g2l.get(int(gd))
        if li is not None:
            li_x[int(nid)] = int(li)
    for nid, gd in disp_dh.dof_map.get(fld_y, {}).items():
        li = g2l.get(int(gd))
        if li is not None:
            li_y[int(nid)] = int(li)

    try:
        inside_eids = mesh.element_bitset("inside").to_indices()
    except Exception:
        inside_eids = np.array([], dtype=int)
    try:
        cut_eids = mesh.element_bitset("cut").to_indices()
    except Exception:
        cut_eids = np.array([], dtype=int)
    if inside_eids.size or cut_eids.size:
        active_eids = np.unique(np.concatenate([inside_eids, cut_eids]).astype(int, copy=False))
        active_nodes = np.unique(mesh.elements_connectivity[active_eids].ravel())
        node_active = np.zeros(n_nodes, dtype=bool)
        node_active[active_nodes.astype(int, copy=False)] = True
    else:
        node_active = np.zeros(n_nodes, dtype=bool)

    for gd_phi, nid in zip(gphi, node_ids):
        x, y0 = mesh.nodes_x_y_pos[int(nid)]
        ux = 0.0
        uy = 0.0
        phi_n = phi_by_node[int(nid)]
        if node_active[int(nid)] and np.isfinite(phi_n) and float(phi_n) <= float(phi_disp_cutoff):
            lix = int(li_x[int(nid)])
            liy = int(li_y[int(nid)])
            if lix >= 0 and liy >= 0:
                ux = float(disp_vec.nodal_values[lix])
                uy = float(disp_vec.nodal_values[liy])
        X_ref = np.array([float(x) - ux, float(y0) - uy], float)
        phi_new = float(ref_ls(X_ref))
        li = ls_ball._g2l[int(gd_phi)]
        phi_new_vals[int(li)] = phi_new

    phi_vals[:] = phi_new_vals
    ls_ball.commit(tol=float(tol_commit))


def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------
    # Geometry / parameters (paper Sec. 6)
    # ------------------------------------------------------------------
    # Paper geometry (Sec. 6.1 / Fig. 4): x ∈ [-0.04, 0.04], y ∈ [0, 0.08].
    x0, x1 = -0.04, 0.04
    y0, y1 = 0.0, 0.08
    Lx = x1 - x0
    Ly = y1 - y0

    r = 0.011
    h0 = 0.039
    ball_center = (0.0, h0 + r)

    nu_f = 7.0114e-5
    rho_f = 1141.0
    rho_s = 1361.0
    mu_s = 20.0e3
    lambda_s = 80.0e3
    mu_f = rho_f * nu_f  # dynamic viscosity

    gamma_N = 1.0e7
    gamma_C0 = 500.0
    eps_relax = 1.0e-4
    wmax = max(float(args.wmax), 1.0)
    g_vec = np.array([0.0, -9.81], float)

    # FE orders (paper: Q2/Q1/Q1/Q1)
    deg_uf = 2
    deg_pf = 1
    deg_us = 1
    deg_ds = 1

    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------
    nodes, elems, edges, corners = structured_quad(
        Lx,
        Ly,
        nx=int(args.nx),
        ny=int(args.ny),
        poly_order=max(deg_uf, 1),
        offset=(x0, y0),
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=max(deg_uf, 1),
    )

    # ------------------------------------------------------------------
    # Moving level set: FE φ field
    # ------------------------------------------------------------------
    ball_ref_ls = CircleLevelSet(center=ball_center, radius=r)
    ls_me = MixedElement(mesh, field_specs={"phi_ball": max(deg_uf, 2)})
    ls_dh = DofHandler(ls_me, method="cg")
    ls_ball = LevelSetGridFunction(ls_dh, field="phi_ball")
    ls_ball.interpolate(lambda x, y: float(ball_ref_ls(np.array([x, y], float))))
    ls_ball.commit(tol=1e-12)
    mesh.classify_elements(ls_ball, tol=1e-12)
    mesh.classify_edges(ls_ball, tol=1e-12)
    mesh.build_interface_segments(ls_ball, tol=1e-12)

    # boundaries
    bc_tags = {
        "top": lambda x, y: np.isclose(y, y1),
        "walls": lambda x, y: np.isclose(y, y0) | np.isclose(x, x0) | np.isclose(x, x1),
    }
    mesh.tag_boundary_edges(bc_tags)

    # ------------------------------------------------------------------
    # Domains / measures (BitSets updated in-place each step)
    # ------------------------------------------------------------------
    use_extension = not bool(args.no_extension)
    extension_layers = 1 if use_extension else 0
    domains = make_domain_sets(mesh, extension_layers=extension_layers)
    qvol = int(args.quad_order)
    dx_fluid, dx_solid, dGamma, dG_fluid, dG_solid = build_measures(
        mesh, ls_ball, domains, qvol=qvol, use_facet_patch_ghost=True
    )
    dG_fluid_ext = None
    dG_solid_ext = None
    if use_extension:
        dG_fluid_ext, dG_solid_ext = build_extension_measures(
            mesh, ls_ball, domains, qvol=qvol, use_facet_patch_ghost=True
        )

    # Hansbo weights (updated in-place)
    theta_pos_vals, theta_neg_vals = hansbo_kappa(mesh, ls_ball, theta_min=1.0e-3)
    theta_pos_ewc = ElementWiseConstant(theta_pos_vals)
    theta_neg_ewc = ElementWiseConstant(theta_neg_vals)
    theta_sum = Pos(theta_pos_ewc) + Neg(theta_neg_ewc) + Constant(1.0e-12)
    kappa_pos = Pos(theta_pos_ewc) / theta_sum
    kappa_neg = Neg(theta_neg_ewc) / theta_sum

    w_fluid_vals = 0.5 * (wmax ** (1.0 - 2.0 * theta_pos_vals))
    w_solid_vals = 0.5 * (wmax ** (1.0 - 2.0 * theta_neg_vals))
    w_fluid_ewc = ElementWiseConstant(w_fluid_vals)
    w_solid_ewc = ElementWiseConstant(w_solid_vals)

    # ------------------------------------------------------------------
    # Mixed element (monolithic)
    # ------------------------------------------------------------------
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
    dof_handler = DofHandler(me, method="cg")

    # Tag initial inactive DOFs (CutFEM FE_Nothing style)
    retag_inactive(
        dof_handler,
        mesh,
        theta_neg=theta_neg_vals,
        solid_cut_drop=0.0,
        fluid_ext_domain=domains.get("fluid_ext_domain"),
        solid_ext_domain=domains.get("solid_ext_domain"),
    )

    # Pressure pin (avoid (0,0) so we can measure p_bc there like in the paper).
    pin_pt = np.array([x0, y1], float)
    dof_handler.tag_dof_by_locator(
        "p_pin",
        "p_pos_",
        locator=lambda x, y, x0=float(pin_pt[0]), y0=float(pin_pt[1]): np.isclose(x, x0) and np.isclose(y, y0),
        find_first=True,
    )

    # ------------------------------------------------------------------
    # Function spaces, trial/test, current/prev state
    # ------------------------------------------------------------------
    Vf = FunctionSpace("Vf", ["u_pos_x", "u_pos_y"], side="+")
    Pf = FunctionSpace("Pf", ["p_pos_"], side="+")
    Vs = FunctionSpace("Vs", ["vs_neg_x", "vs_neg_y"], side="-")
    Ds = FunctionSpace("Ds", ["d_neg_x", "d_neg_y"], side="-")

    du_f = VectorTrialFunction(Vf, dof_handler=dof_handler)
    dp_f = TrialFunction(name="dp_f", field_name="p_pos_", dof_handler=dof_handler, side="+")
    du_s = VectorTrialFunction(Vs, dof_handler=dof_handler)
    ddisp_s = VectorTrialFunction(Ds, dof_handler=dof_handler)

    test_vel_f = VectorTestFunction(Vf, dof_handler=dof_handler)
    test_q_f = TestFunction(name="q_f", field_name="p_pos_", dof_handler=dof_handler, side="+")
    test_vel_s = VectorTestFunction(Vs, dof_handler=dof_handler)
    test_disp_s = VectorTestFunction(Ds, dof_handler=dof_handler)

    uf_k = VectorFunction("uf_k", ["u_pos_x", "u_pos_y"], dof_handler)
    pf_k = Function("pf_k", "p_pos_", dof_handler)
    us_k = VectorFunction("us_k", ["vs_neg_x", "vs_neg_y"], dof_handler)
    disp_k = VectorFunction("disp_k", ["d_neg_x", "d_neg_y"], dof_handler)

    uf_n = VectorFunction("uf_n", ["u_pos_x", "u_pos_y"], dof_handler)
    pf_n = Function("pf_n", "p_pos_", dof_handler)
    us_n = VectorFunction("us_n", ["vs_neg_x", "vs_neg_y"], dof_handler)
    disp_n = VectorFunction("disp_n", ["d_neg_x", "d_neg_y"], dof_handler)

    for f in (uf_k, pf_k, us_k, disp_k, uf_n, pf_n, us_n, disp_n):
        f.nodal_values.fill(0.0)

    # BCs (paper Sec. 6.2)
    bcs = [
        BoundaryCondition("u_pos_y", "dirichlet", "top", lambda x, y: 0.0),  # slip: v_y=0
        BoundaryCondition("u_pos_x", "dirichlet", "walls", lambda x, y: 0.0),
        BoundaryCondition("u_pos_y", "dirichlet", "walls", lambda x, y: 0.0),
        BoundaryCondition("p_pos_", "dirichlet", "p_pin", lambda x, y: 0.0),
    ]
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, lambda x, y: 0.0) for b in bcs]

    dof_handler.apply_bcs(bcs, uf_k, pf_k, us_k, disp_k, uf_n, pf_n, us_n, disp_n)

    # ------------------------------------------------------------------
    # Build FSI residual/Jacobian (baseline) + forcing + contact
    # ------------------------------------------------------------------
    dt = Constant(float(args.dt))
    theta = Constant(1.0)
    cell_h = CellDiameter()
    beta_N = Constant(gamma_N)

    rho_f_c = Constant(rho_f)
    rho_s_c = Constant(rho_s)
    nu_f_c = Constant(nu_f)
    mu_f_c = Constant(mu_f)
    mu_s_c = Constant(mu_s)
    lambda_s_c = Constant(lambda_s)

    gamma_v = Constant(0.5)
    gamma_v_s = Constant(0.1)
    gamma_p = Constant(0.1)
    gamma_v_grad = Constant(0.1)

    # Restrict unknowns to the current active submeshes via Restriction nodes
    # (keeps active-DOF analysis meaningful and avoids unconstrained blocks).
    supg_delta0 = None if bool(args.no_supg) else Constant(1.0e-5)

    gamma_v_ext = Constant(10.0) if use_extension else None
    gamma_p_ext = Constant(10.0) if use_extension else None
    gamma_vs_ext = Constant(10.0) if use_extension else None
    gamma_u_ext = Constant(10.0) if use_extension else None
    gamma_u_psi_ext = Constant(1.0e-4) if use_extension else None

    fluid_active = domains.get("has_pos_ext") if use_extension else domains["has_pos"]
    solid_active = domains.get("has_neg_ext") if use_extension else domains["has_neg"]

    du_f_R = restrict(du_f, fluid_active)
    dp_f_R = restrict(dp_f, fluid_active)
    test_vel_f_R = restrict(test_vel_f, fluid_active)
    test_q_f_R = restrict(test_q_f, fluid_active)
    uf_k_R = restrict(uf_k, fluid_active)
    uf_n_R = restrict(uf_n, fluid_active)
    pf_k_R = restrict(pf_k, fluid_active)
    pf_n_R = restrict(pf_n, fluid_active)

    du_s_R = restrict(du_s, solid_active)
    ddisp_s_R = restrict(ddisp_s, solid_active)
    test_vel_s_R = restrict(test_vel_s, solid_active)
    test_disp_s_R = restrict(test_disp_s, solid_active)
    us_k_R = restrict(us_k, solid_active)
    us_n_R = restrict(us_n, solid_active)
    disp_k_R = restrict(disp_k, solid_active)
    disp_n_R = restrict(disp_n, solid_active)

    terms = build_fsi_eulerian_forms(
        du_f=du_f_R,
        dp_f=dp_f_R,
        du_s=du_s_R,
        ddisp_s=ddisp_s_R,
        test_vel_f=test_vel_f_R,
        test_q_f=test_q_f_R,
        test_vel_s=test_vel_s_R,
        test_disp_s=test_disp_s_R,
        uf_k=uf_k_R,
        pf_k=pf_k_R,
        uf_n=uf_n_R,
        pf_n=pf_n_R,
        us_k=us_k_R,
        us_n=us_n_R,
        disp_k=disp_k_R,
        disp_n=disp_n_R,
        dx_fluid=dx_fluid,
        dx_solid=dx_solid,
        dGamma=dGamma,
        dG_fluid=dG_fluid,
        dG_solid=dG_solid,
        kappa_pos=kappa_pos,
        kappa_neg=kappa_neg,
        cell_h=cell_h,
        beta_N=beta_N,
        rho_f=rho_f_c,
        rho_s=rho_s_c,
        mu_f=mu_f_c,
        mu_s=mu_s_c,
        lambda_s=lambda_s_c,
        dt=dt,
        theta=theta,
        gamma_v=gamma_v,
        gamma_v_s=gamma_v_s,
        gamma_p=gamma_p,
        gamma_v_grad=gamma_v_grad,
        gamma_u_mom=gamma_v_grad,
        w_fluid=w_fluid_ewc,
        w_solid=w_solid_ewc,
        dG_fluid_ext=dG_fluid_ext,
        dG_solid_ext=dG_solid_ext,
        gamma_v_ext=gamma_v_ext,
        gamma_p_ext=gamma_p_ext,
        gamma_vs_ext=gamma_vs_ext,
        gamma_u_ext=gamma_u_ext,
        gamma_u_psi_ext=gamma_u_psi_ext,
        supg_delta0_vs=supg_delta0,
        supg_delta0_u=supg_delta0,
        fluid_hessian_ghost=not bool(args.no_hessian_ghost),
        solid_reg_eps=Constant(0.0),
        use_linear_solid=False,
        solid_stvk_paper=True,
        solid_advect_lagged=False,
        s_nitsche_value=1.0,
        interface_form="paper",
    )

    # Body force (gravity)
    f_ext = Constant(g_vec, dim=1)
    residual_form = terms.residual_form - rho_f_c * dot(f_ext, test_vel_f_R) * dx_fluid - rho_s_c * dot(f_ext, test_vel_s_R) * dx_solid
    jacobian_form = terms.jacobian_form

    # Contact (paper Eq. (21) and semi-smooth Jacobian)
    n_s = FacetNormal()
    n_f = Constant(-1.0) * n_s
    I2 = Identity(2)
    sigma_f_k = sigma_f_newtonian(uf_k_R, pf_k_R, rho_f=rho_f_c, nu_f=nu_f_c)
    dsigma_f = dsigma_f_newtonian(du_f_R, dp_f_R, rho_f=rho_f_c, nu_f=nu_f_c)
    sigma_s_k = sigma_s_stvk_paper(disp_k_R, mu_s=mu_s_c, lambda_s=lambda_s_c)
    dsigma_s = dsigma_s_stvk_paper(disp_k_R, ddisp_s_R, mu_s=mu_s_c, lambda_s=lambda_s_c)
    penalty_nitsche = beta_N * mu_f_c / cell_h
    gap_eps = Analytic(y) - Constant(eps_relax)
    # Paper: γ_C := γ_C^0 μ_s / h with γ_C^0 = 500 (Sec. 6.3)
    gamma_C = Constant(gamma_C0) * mu_s_c / cell_h
    contact = RelaxedWallContact(
        gamma_C=gamma_C,
        gap_eps=gap_eps,
        n_s=n_s,
        n_f=n_f,
        penalty_nitsche=penalty_nitsche,
    )
    P = contact.P_gammaC(
        u=disp_k_R,
        u_prev=disp_n_R,
        v_f=uf_k_R,
        p_f=pf_k_R,
        v_s=us_k_R,
        sigma_s=sigma_s_k,
        sigma_f=sigma_f_k,
    )
    dP = contact.dP_gammaC(
        du=ddisp_s_R,
        dv_f=du_f_R,
        dp_f=dp_f_R,
        dv_s=du_s_R,
        dsigma_s=dsigma_s,
        dsigma_f=dsigma_f,
    )
    residual_form = residual_form + contact.residual_term(P=P, k=Constant(1.0), test_v_s=test_vel_s_R) * dGamma
    jacobian_form = jacobian_form + contact.jacobian_term(P=P, dP=dP, k=Constant(1.0), test_v_s=test_vel_s_R) * dGamma

    # Paper scaling (Problem 5): keep the algebraic system identical but scale
    # residual/Jacobian by `dt` so absolute Newton tolerances match the paper.
    residual_form = residual_form * dt
    jacobian_form = jacobian_form * dt

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    dt0 = float(args.dt)
    newton_tol0 = float(args.newton_tol)

    newton_params = NewtonParameters(
        newton_tol=newton_tol0,
        max_newton_iter=int(args.max_newton_iter),
        line_search=not bool(args.no_line_search),
        ls_mode=os.getenv("LS_MODE", "dealii"),
        print_level=int(args.newton_print_level),
    )

    solver = NewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dof_handler,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=newton_params,
        backend=str(args.backend),
        quad_order=int(args.quad_order),
    )

    bc_data_init = dof_handler.get_dirichlet_data(bcs_homog if bcs_homog else bcs) or {}
    n_dirichlet_init = int(len(bc_data_init))
    n_free_init = int(np.asarray(getattr(solver, "active_dofs", np.empty(0, dtype=int)), dtype=int).size)

    # Paper Sec. 6.4: adaptive dt + Newton tol tied to dt, with factor α_k=0.1.
    def _on_dt_change(dt_new: float) -> None:
        dt_new = float(dt_new)
        if dt_new <= 0.0:
            raise ValueError(f"dt must be positive, got {dt_new!r}")
        dt.value = dt_new
        # Paper: when dt is refined by α_k, also refine Newton tolerance by α_k.
        solver.np.newton_tol = newton_tol0 * (dt_new / max(dt0, 1.0e-300))

    time_params = TimeStepperParameters(
        dt=dt0,
        max_steps=int(np.ceil(float(args.final_time) / max(dt0, 1.0e-16))),
        theta=1.0,
        final_time=float(args.final_time),
        # This is a transient benchmark; do not early-exit based on a small Newton update.
        stop_on_steady=False,
        allow_dt_reduction=bool(args.adaptive_dt),
        dt_reduction_factor=0.1,
        dt_min=float(args.dt_min),
        # Disable iteration-count based dt changes; we implement the paper’s
        # coarsening rule explicitly in the post-step callback.
        dt_increase_factor=1.0,
        dt_decrease_factor_slow=1.0,
        on_dt_change=_on_dt_change if bool(args.adaptive_dt) else None,
    )

    out_dir = Path(args.out_dir)
    metrics_enabled = not bool(args.no_metrics)
    metrics_stride = int(args.metrics_stride)
    compute_energies = metrics_enabled and metrics_stride > 0

    vtk_every = int(getattr(args, "vtk_every", 0) or 0)
    if bool(args.save_vtk) and vtk_every <= 0:
        vtk_every = 1
    save_vtk = bool(args.save_vtk) or (vtk_every > 0)

    if save_vtk or metrics_enabled:
        out_dir.mkdir(parents=True, exist_ok=True)
    vtk_dir = out_dir / "vtk"
    if save_vtk:
        vtk_dir.mkdir(parents=True, exist_ok=True)

    step_idx = {"k": 0}
    t_state = {"t": 0.0}
    log_stride = max(int(args.log_stride), 1)

    def _save_vtk(step_no: int, t: float) -> None:
        if not save_vtk:
            return
        if vtk_every <= 0:
            return
        if step_no < 0:
            raise ValueError(f"step_no must be >= 0, got {step_no!r}")
        fname = vtk_dir / f"solution_{int(step_no):04d}.vtu"
        phi_nodes = ls_ball.evaluate_on_nodes(mesh)
        tag_map = {"outside": 0, "cut": 1, "inside": 2}
        cell_tags = np.fromiter(
            (tag_map.get(str(getattr(elem, "tag", "")), -1) for elem in mesh.elements_list),
            dtype=int,
            count=len(mesh.elements_list),
        )
        export_vtk(
            filename=str(fname),
            mesh=mesh,
            dof_handler=dof_handler,
            functions={
                "uf": uf_k,
                "pf": pf_k,
                "us": us_k,
                "disp": disp_k,
                "phi_ball": phi_nodes,
            },
            cell_data={
                "elem_tag": cell_tags,
                "time": np.full_like(cell_tags, float(t), dtype=float),
            },
        )

    # --- quantities of interest (paper Sec. 6.6) ----------------------
    p_bc_dof = _nearest_field_dof(dof_handler, "p_pos_", np.array([0.0, 0.0], float))
    p_coords = np.asarray(dof_handler.get_dof_coords("p_pos_"), dtype=float)
    p_slice = np.asarray(dof_handler.get_field_slice("p_pos_"), dtype=int)
    p_bc_xy = p_coords[int(np.where(p_slice == p_bc_dof)[0][0])]

    # For QoIs like max_{Q_f} |v_f| we must exclude implicit-extension DOFs
    # (Ω^{f,ext}_{h,n} \\ Ω^{f}_{h,n}). Those are not physical and can fluctuate.
    uxs_all = np.asarray(dof_handler.get_field_slice("u_pos_x"), dtype=int)
    uys_all = np.asarray(dof_handler.get_field_slice("u_pos_y"), dtype=int)
    if int(uxs_all.size) != int(uys_all.size):
        raise RuntimeError("Unexpected mismatch between u_pos_x and u_pos_y DOF counts.")
    nx_u = int(uxs_all.size)
    ux_g2i = {int(gd): i for i, gd in enumerate(uxs_all.tolist())}
    uy_g2i = {int(gd): i for i, gd in enumerate(uys_all.tolist())}
    elem_ux_li: list[np.ndarray] = []
    elem_uy_li: list[np.ndarray] = []
    for eid in range(len(mesh.elements_list)):
        ux_g = np.asarray(dof_handler.element_maps["u_pos_x"][int(eid)], dtype=int)
        uy_g = np.asarray(dof_handler.element_maps["u_pos_y"][int(eid)], dtype=int)
        li_x = np.fromiter((ux_g2i.get(int(gd), -1) for gd in ux_g), dtype=int)
        li_y = np.fromiter((uy_g2i.get(int(gd), -1) for gd in uy_g), dtype=int)
        elem_ux_li.append(li_x[li_x >= 0])
        elem_uy_li.append(li_y[li_y >= 0])

    phys_ux = np.zeros(nx_u, dtype=bool)
    phys_uy = np.zeros(nx_u, dtype=bool)

    qoi = {
        "t0": None,
        "t_star": None,
        "v_star": None,
        "f_star": None,
        "t_cont": None,
        "t_jump": None,
        "h_jump": None,
        "max_p_bc": -np.inf,
        "max_v_f": 0.0,
        "max_E_el": 0.0,
        "max_E_kin_f": 0.0,
        "max_E_kin_s": 0.0,
        "newton_steps_sum": 0.0,
        "n_steps": 0,
    }
    hist = {"t": [], "dt": [], "min_dist": [], "p_bc": [], "vmax_v_f": []}
    prev = {"t": None, "dist": None}
    rebound_window: list[tuple[float, float]] = []

    e2 = Constant(np.array([0.0, 1.0], float), dim=1)
    qoi_events = None
    qoi_energies = None
    if metrics_enabled:
        qoi_events = NamedFunctionalEvaluator(
            forms={
                "vol_s": Constant(1.0) * dx_solid,
                "vy_s": dot(us_k_R, e2) * dx_solid,
                "f_y": dot(dot(sigma_f_newtonian(uf_k_R, pf_k_R, rho_f=rho_f_c, nu_f=nu_f_c), n_f), e2) * dGamma,
            },
            dof_handler=dof_handler,
            mixed_element=me,
            backend=str(args.backend),
            quad_order=int(args.quad_order),
        )
        if compute_energies:
            qoi_energies = NamedFunctionalEvaluator(
                forms={
                    "E_el": inner(
                        sigma_s_stvk_paper(disp_k_R, mu_s=mu_s_c, lambda_s=lambda_s_c),
                        green_lagrange_strain(disp_k_R),
                    )
                    * dx_solid,
                    "E_kin_f": Constant(0.5) * rho_f_c * dot(uf_k_R, uf_k_R) * dx_fluid,
                    "E_kin_s": Constant(0.5) * rho_s_c * dot(us_k_R, us_k_R) * dx_solid,
                },
                dof_handler=dof_handler,
                mixed_element=me,
                backend=str(args.backend),
                quad_order=int(args.quad_order),
            )

    dt_prev_accepted = dt0
    steps_since_refine = 0

    if save_vtk and vtk_every > 0:
        _save_vtk(0, 0.0)

    def post_step_cb(functions: Sequence, prev_functions: Sequence | None = None) -> None:
        old_active = np.asarray(getattr(solver, "active_dofs", []), dtype=int).copy()
        dt_step = float(dt.value)
        t_state["t"] = float(t_state["t"]) + dt_step
        t = float(t_state["t"])
        # Update level set from displacement at the new time step.
        update_ball_levelset_from_displacement(
            ls_ball,
            disp_k,
            ball_ref_ls,
            phi_disp_cutoff=float(2.0 * eps_relax),
            tol_commit=1e-12,
        )
        # Mesh already reclassified by ls_ball.commit(); keep explicit for safety:
        mesh.classify_elements(ls_ball, tol=1e-12)
        mesh.classify_edges(ls_ball, tol=1e-12)
        mesh.build_interface_segments(ls_ball, tol=1e-12)
        refresh_domain_sets(mesh, domains, extension_layers=extension_layers)
        # Update Hansbo cut ratios (in-place arrays backing ElementWiseConstant)
        thp, thn = hansbo_kappa(mesh, ls_ball, theta_min=1.0e-3)
        theta_pos_vals[:] = thp
        theta_neg_vals[:] = thn
        w_fluid_vals[:] = 0.5 * (wmax ** (1.0 - 2.0 * theta_pos_vals))
        w_solid_vals[:] = 0.5 * (wmax ** (1.0 - 2.0 * theta_neg_vals))
        # Retag inactive DOFs after topology change
        retag_inactive(
            dof_handler,
            mesh,
            theta_neg=theta_neg_vals,
            solid_cut_drop=0.0,
            fluid_ext_domain=domains.get("fluid_ext_domain"),
            solid_ext_domain=domains.get("solid_ext_domain"),
        )
        # Refresh precomputed kernel geometry for the moving interface.
        solver.refresh_levelset_kernels(ls_ball)
        if qoi_events is not None:
            qoi_events.refresh_levelset(ls_ball)
        if qoi_energies is not None:
            qoi_energies.refresh_levelset(ls_ball)
        # Reduced-system DOFs depend on Restriction domains + inactive tags.
        active_changed = recompute_active_dofs(solver, bcs_homog if bcs_homog else bcs)
        if active_changed:
            new_active = np.asarray(getattr(solver, "active_dofs", []), dtype=int)
            newly_active = np.setdiff1d(new_active, old_active, assume_unique=False)
            if newly_active.size:
                extend_newly_active_dofs_nearest(
                    dh=dof_handler,
                    newly_active=newly_active,
                    active_old=old_active,
                    active_new=new_active,
                    field_to_current={
                        "u_pos_x": uf_k,
                        "u_pos_y": uf_k,
                        "p_pos_": pf_k,
                        "vs_neg_x": us_k,
                        "vs_neg_y": us_k,
                        "d_neg_x": disp_k,
                        "d_neg_y": disp_k,
                    },
                    field_to_prev={
                        "u_pos_x": uf_n,
                        "u_pos_y": uf_n,
                        "p_pos_": pf_n,
                        "vs_neg_x": us_n,
                        "vs_neg_y": us_n,
                        "d_neg_x": disp_n,
                        "d_neg_y": disp_n,
                    },
                    k=int(os.getenv("PYCUTFEM_EXTEND_NEWLY_ACTIVE_K", "4") or "4"),
                    trace=os.getenv("PYCUTFEM_EXTEND_NEWLY_ACTIVE_TRACE", "").lower() in {"1", "true", "yes"},
                )
        # Re-apply BCs (includes pressure pin)
        dof_handler.apply_bcs(bcs, uf_k, pf_k, us_k, disp_k, uf_n, pf_n, us_n, disp_n)

        dist = _min_interface_y(mesh) - y0

        # Metrics / QoI tracking ------------------------------------
        if metrics_enabled:
            hist["t"].append(float(t))
            hist["dt"].append(float(dt_step))
            hist["min_dist"].append(float(dist))
            p_bc = float(pf_k.get_nodal_values(np.array([p_bc_dof], dtype=int))[0])
            hist["p_bc"].append(float(p_bc))
            qoi["max_p_bc"] = max(float(qoi["max_p_bc"]), float(p_bc))

            # Physical-domain DOFs for max_{Q_f} |v_f|.
            phys_ux.fill(False)
            phys_uy.fill(False)
            for eid in domains["has_pos"].to_indices():
                li_x = elem_ux_li[int(eid)]
                li_y = elem_uy_li[int(eid)]
                if li_x.size:
                    phys_ux[li_x] = True
                if li_y.size:
                    phys_uy[li_y] = True

            # nodal max |v_f| (proxy for max_{Q_f})
            vmax_now = 0.0
            if nx_u:
                ux_vals = np.asarray(uf_k.nodal_values[:nx_u], dtype=float)
                uy_vals = np.asarray(uf_k.nodal_values[nx_u : nx_u + nx_u], dtype=float)
                full_to_red = getattr(solver, "full_to_red", None)
                if isinstance(full_to_red, np.ndarray) and full_to_red.shape == (int(dof_handler.total_dofs),):
                    active = (full_to_red[uxs_all] >= 0) & (full_to_red[uys_all] >= 0)
                else:
                    inactive = set(getattr(dof_handler, "dof_tags", {}).get("inactive", set()))
                    if inactive:
                        inactive_arr = np.fromiter(inactive, dtype=int)
                        active = ~np.isin(uxs_all, inactive_arr)
                    else:
                        active = np.ones(nx_u, dtype=bool)
                active = active & phys_ux & phys_uy
                if np.any(active):
                    vmax_now = float(np.max(np.sqrt(ux_vals[active] ** 2 + uy_vals[active] ** 2)))
                    qoi["max_v_f"] = max(float(qoi["max_v_f"]), vmax_now)
            hist["vmax_v_f"].append(float(vmax_now))

            # Newton iteration stats (average over accepted time steps)
            n_it = float(len(getattr(solver, "_last_iter_timings", []) or []))
            qoi["newton_steps_sum"] = float(qoi["newton_steps_sum"]) + n_it
            qoi["n_steps"] = int(qoi["n_steps"]) + 1

            # Event detection on the distance-to-wall signal
            dist_prev = prev["dist"]
            t_prev = prev["t"]
            prev["dist"] = float(dist)
            prev["t"] = float(t)

            def _crossing_time(threshold: float) -> float | None:
                if dist_prev is None or t_prev is None:
                    return None
                if float(dist_prev) > threshold >= float(dist):
                    denom = float(dist_prev) - float(dist)
                    if abs(denom) < 1.0e-16:
                        return float(t)
                    frac = (float(dist_prev) - float(threshold)) / denom
                    return float(t_prev) + frac * (float(t) - float(t_prev))
                return None

            # t0: center reaches height h0  ->  min_dist == (h0 - r)
            if qoi["t0"] is None:
                te = _crossing_time(float(h0 - r))
                if te is not None:
                    qoi["t0"] = float(te)

            # t*: min_dist == 2r (relative to t0)
            if qoi["t0"] is not None and qoi["t_star"] is None:
                te = _crossing_time(float(2.0 * r))
                if te is not None:
                    qoi["t_star"] = float(te) - float(qoi["t0"])
                    if qoi_events is not None:
                        coeffs = {
                            uf_k.name: uf_k,
                            pf_k.name: pf_k,
                            us_k.name: us_k,
                            disp_k.name: disp_k,
                            uf_n.name: uf_n,
                            pf_n.name: pf_n,
                            us_n.name: us_n,
                            disp_n.name: disp_n,
                        }
                        vals = qoi_events.evaluate(coeffs)
                        vol = float(vals.get("vol_s", 0.0))
                        qoi["v_star"] = float(vals.get("vy_s", 0.0)) / max(vol, 1.0e-16)
                        qoi["f_star"] = float(vals.get("f_y", 0.0))

            # t_cont: first contact with relaxed wall (min_dist == eps_relax)
            if qoi["t0"] is not None and qoi["t_cont"] is None:
                te = _crossing_time(float(eps_relax))
                if te is not None:
                    qoi["t_cont"] = float(te) - float(qoi["t0"])

            # t_jump/h_jump: first local maximum after contact
            if qoi["t_cont"] is not None and qoi["t_jump"] is None:
                rebound_window.append((float(t), float(dist)))
                if len(rebound_window) >= 3:
                    (t0w, d0w), (t1w, d1w), (t2w, d2w) = rebound_window[-3:]
                    if d1w > d0w and d1w > d2w and d1w > float(1.1 * eps_relax):
                        qoi["t_jump"] = float(t1w) - float(qoi["t0"])
                        qoi["h_jump"] = float(d1w)

            if compute_energies and qoi_energies is not None:
                # Energies peak around impact/contact; sample densely in a near-wall band.
                near_contact = float(dist) < 1.0e-2
                if near_contact or (step_idx["k"] % max(1, metrics_stride) == 0):
                    coeffs = {
                        uf_k.name: uf_k,
                        pf_k.name: pf_k,
                        us_k.name: us_k,
                        disp_k.name: disp_k,
                        uf_n.name: uf_n,
                        pf_n.name: pf_n,
                        us_n.name: us_n,
                        disp_n.name: disp_n,
                    }
                    vals = qoi_energies.evaluate(coeffs)
                    qoi["max_E_el"] = max(float(qoi["max_E_el"]), float(vals.get("E_el", 0.0)))
                    qoi["max_E_kin_f"] = max(float(qoi["max_E_kin_f"]), float(vals.get("E_kin_f", 0.0)))
                    qoi["max_E_kin_s"] = max(float(qoi["max_E_kin_s"]), float(vals.get("E_kin_s", 0.0)))

        # Paper Sec. 6.4: coarsen dt after 10 accepted steps without refinement.
        nonlocal dt_prev_accepted, steps_since_refine
        if bool(args.adaptive_dt):
            if dt_step < dt_prev_accepted - 1.0e-16:
                dt_prev_accepted = dt_step
                steps_since_refine = 0
            else:
                steps_since_refine += 1

            if steps_since_refine >= 10 and dt_step < dt0 - 1.0e-16:
                dt_new = min(dt0, 10.0 * dt_step)
                if dt_new > dt_step + 1.0e-16:
                    time_params.dt = float(dt_new)  # takes effect next step
                    dt_prev_accepted = float(dt_new)
                    steps_since_refine = 0
                    print(f"    [dt] coarsen: next dt -> {float(dt_new):.3e}")

        if step_idx["k"] % log_stride == 0:
            if metrics_enabled and hist["p_bc"]:
                print(
                    f"[step {step_idx['k']:04d}] t={t:.6e}  min_dist≈{dist:.6e}  "
                    f"p_bc({p_bc_xy[0]:+.3f},{p_bc_xy[1]:+.3f})={float(hist['p_bc'][-1]):.3e}"
                )
            else:
                print(f"[step {step_idx['k']:04d}] t={t:.6e}  min_dist≈{dist:.6e}")

        step_no = int(step_idx["k"]) + 1
        if save_vtk and vtk_every > 0 and (step_no % vtk_every) == 0:
            _save_vtk(step_no, t)
        step_idx["k"] += 1

    solver.post_timeloop_cb = post_step_cb

    try:
        solver.solve_time_interval(
            functions=[uf_k, pf_k, us_k, disp_k],
            prev_functions=[uf_n, pf_n, us_n, disp_n],
            time_params=time_params,
            aux_functions={"dt": dt},
        )
    finally:
        if metrics_enabled:
            bc_data = dof_handler.get_dirichlet_data(bcs_homog if bcs_homog else bcs) or {}
            n_dirichlet = int(len(bc_data))
            n_free = int(np.asarray(getattr(solver, "active_dofs", np.empty(0, dtype=int)), dtype=int).size)
            metrics_out = {
                "mesh": {"nx": int(args.nx), "ny": int(args.ny)},
                "time": {
                    "dt0": float(dt0),
                    "dt_min": float(args.dt_min),
                    "adaptive_dt": bool(args.adaptive_dt),
                    "final_time": float(args.final_time),
                },
                "params": {"r": float(r), "h0": float(h0), "eps_relax": float(eps_relax)},
                "dofs": {
                    "total": int(dof_handler.total_dofs),
                    "free_initial": n_free_init,
                    "dirichlet_initial": n_dirichlet_init,
                    "paper_initial": n_free_init + n_dirichlet_init,
                    "free": n_free,
                    "dirichlet": n_dirichlet,
                    # The paper reports the DoFs of the *active* CutFEM system (including
                    # Dirichlet rows/cols but excluding dropped inactive DOFs).
                    "paper": n_free + n_dirichlet,
                },
                "qoi": {
                    **{k: (None if v is None else float(v)) for k, v in qoi.items() if k != "n_steps"},
                    "n_steps": int(qoi["n_steps"]),
                    "n_newton_avg": float(qoi["newton_steps_sum"]) / max(int(qoi["n_steps"]), 1),
                },
                "p_bc": {"gdof": int(p_bc_dof), "xy": [float(p_bc_xy[0]), float(p_bc_xy[1])]},
            }
            (out_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2, sort_keys=True))
            # minimal distance signal (for Fig. 5/8-style plots)
            csv_lines = ["t,dt,min_dist,p_bc,vmax_v_f"]
            for tt, dtt, dd, pp, vv in zip(hist["t"], hist["dt"], hist["min_dist"], hist["p_bc"], hist["vmax_v_f"]):
                csv_lines.append(f"{tt:.16e},{dtt:.16e},{dd:.16e},{pp:.16e},{vv:.16e}")
            (out_dir / "min_dist.csv").write_text("\n".join(csv_lines) + "\n")
            print("\n[QoI] wrote:", out_dir / "metrics.json")
            print("[QoI] wrote:", out_dir / "min_dist.csv")
            print("[QoI] summary:", json.dumps(metrics_out["qoi"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
