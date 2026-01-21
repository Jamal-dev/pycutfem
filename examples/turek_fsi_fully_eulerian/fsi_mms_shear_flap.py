#!/usr/bin/env python
"""
Fully-Eulerian FSI MMS (Variant A: shear-flap, fixed interface).

Goal:
  - Exercise the same Eulerian FSI weak forms as the Turek script, but on a
    minimal fixed-interface setup that is easy to diagnose.
  - Intended primarily for backend validation (including the C++ JIT backend).

Run (example):
  PYCUTFEM_JIT_BACKEND=cpp conda run --no-capture-output -n fenicsx \
    python -u examples/turek_fsi_fully_eulerian/fsi_mms_shear_flap.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    dot,
    restrict,
)
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.utils.fsi_fully_eulerian import build_fsi_eulerian_forms, build_measures, make_domain_sets, retag_inactive
from pycutfem.utils.meshgen import structured_quad


def _tag_rect_boundaries(mesh: Mesh, *, x0: float, x1: float, y0: float, y1: float, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - x0) <= tol,
            "right": lambda x, y: abs(x - x1) <= tol,
            "bottom": lambda x, y: abs(y - y0) <= tol,
            "top": lambda x, y: abs(y - y1) <= tol,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=5)
    parser.add_argument("--ny", type=int, default=3)
    parser.add_argument("--poly-order-u", type=int, default=2)
    parser.add_argument("--poly-order-p", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--q", type=int, default=6)
    parser.add_argument("--beta-n", type=float, default=20.0)
    parser.add_argument("--mu-f", type=float, default=1.0)
    parser.add_argument("--mu-s", type=float, default=2.0)
    parser.add_argument("--A", type=float, default=0.05)
    parser.add_argument("--newton-tol", type=float, default=1.0e-12)
    parser.add_argument("--max-newton-iter", type=int, default=12)
    parser.add_argument("--cache-dir", type=str, default=os.getenv("PYCUTFEM_CACHE_DIR", ""))
    args = parser.parse_args()

    if args.cache_dir:
        os.environ["PYCUTFEM_CACHE_DIR"] = args.cache_dir

    # ------------------------------------------------------------------
    # Geometry: rectangle with vertical interface x=0 (solid on x<0, fluid on x>0)
    # ------------------------------------------------------------------
    Lx, Ly = 2.0, 1.0
    x0, x1 = -1.0, 1.0
    y0, y1 = -0.5, 0.5

    nodes, elems, edges, corners = structured_quad(
        Lx,
        Ly,
        nx=int(args.nx),
        ny=int(args.ny),
        poly_order=int(args.poly_order_u),
        offset=(x0, y0),
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=int(args.poly_order_u),
    )
    _tag_rect_boundaries(mesh, x0=x0, x1=x1, y0=y0, y1=y1)

    level_set = AffineLevelSet(a=1.0, b=0.0, c=0.0)  # Γ: x=0
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)

    domains = make_domain_sets(mesh, use_aligned_interface=False)
    dx_fluid, dx_solid, dGamma, dG_fluid, dG_solid = build_measures(mesh, level_set, domains, qvol=int(args.q))

    # ------------------------------------------------------------------
    # Unknowns (matching build_fsi_eulerian_forms conventions)
    # ------------------------------------------------------------------
    me = MixedElement(
        mesh,
        field_specs={
            "u_pos_x": int(args.poly_order_u),
            "u_pos_y": int(args.poly_order_u),
            "p_pos_": int(args.poly_order_p),
            "vs_neg_x": int(args.poly_order_u),
            "vs_neg_y": int(args.poly_order_u),
            "d_neg_x": int(args.poly_order_u),
            "d_neg_y": int(args.poly_order_u),
        },
    )
    dh = DofHandler(me, method="cg")
    # Drop DOFs that live strictly on the wrong side (fluid fields inside solid, etc.).
    # Without this, the MMS is underdetermined and the solver can converge with those
    # inactive DOFs left at the initial guess, polluting the reported max errors.
    retag_inactive(dh, mesh)

    velocity_fluid = FunctionSpace(name="velocity_fluid", field_names=["u_pos_x", "u_pos_y"], dim=1, side="+")
    pressure_fluid = FunctionSpace(name="pressure_fluid", field_names=["p_pos_"], dim=0, side="+")
    velocity_solid = FunctionSpace(name="velocity_solid", field_names=["vs_neg_x", "vs_neg_y"], dim=1, side="-")
    displacement_solid = FunctionSpace(name="displacement", field_names=["d_neg_x", "d_neg_y"], dim=1, side="-")

    du_f = VectorTrialFunction(space=velocity_fluid, dof_handler=dh)
    dp_f = TrialFunction(name="trial_pressure_fluid", field_name="p_pos_", dof_handler=dh, side="+")
    du_s = VectorTrialFunction(space=velocity_solid, dof_handler=dh)
    ddisp_s = VectorTrialFunction(space=displacement_solid, dof_handler=dh)

    v_f = VectorTestFunction(space=velocity_fluid, dof_handler=dh)
    q_f = TestFunction(name="test_pressure_fluid", field_name="p_pos_", dof_handler=dh, side="+")
    v_s = VectorTestFunction(space=velocity_solid, dof_handler=dh)
    w_s = VectorTestFunction(space=displacement_solid, dof_handler=dh)

    uf_k = VectorFunction(name="u_f_k", field_names=["u_pos_x", "u_pos_y"], dof_handler=dh, side="+")
    pf_k = Function(name="p_f_k", field_name="p_pos_", dof_handler=dh, side="+")
    uf_n = VectorFunction(name="u_f_n", field_names=["u_pos_x", "u_pos_y"], dof_handler=dh, side="+")
    pf_n = Function(name="p_f_n", field_name="p_pos_", dof_handler=dh, side="+")
    us_k = VectorFunction(name="u_s_k", field_names=["vs_neg_x", "vs_neg_y"], dof_handler=dh, side="-")
    us_n = VectorFunction(name="u_s_n", field_names=["vs_neg_x", "vs_neg_y"], dof_handler=dh, side="-")
    disp_k = VectorFunction(name="disp_k", field_names=["d_neg_x", "d_neg_y"], dof_handler=dh, side="-")
    disp_n = VectorFunction(name="disp_n", field_names=["d_neg_x", "d_neg_y"], dof_handler=dh, side="-")

    for f in (uf_k, pf_k, uf_n, pf_n, us_k, us_n, disp_k, disp_n):
        f.nodal_values.fill(0.0)

    # ------------------------------------------------------------------
    # MMS data (global smooth extension)
    # ------------------------------------------------------------------
    mu_f = float(args.mu_f)
    mu_s = float(args.mu_s)
    alpha = mu_s / mu_f
    A = float(args.A)

    def psi(x):
        return x + x**2

    def psi_dd(x):
        return 2.0 + 0.0 * x

    def w_exact(x, t):
        return A * np.exp(alpha * t) * psi(x)

    def u_exact_y(x, t):
        return alpha * w_exact(x, t)

    def u_exact_y_xx(x, t):
        return alpha * A * np.exp(alpha * t) * psi_dd(x)

    def w_exact_xx(x, t):
        return A * np.exp(alpha * t) * psi_dd(x)

    time_state = {"t_prev": 0.0, "t_curr": 0.0, "dt": 0.0}

    def f_f_y_discrete(x):
        t0 = float(time_state["t_prev"])
        t1 = float(time_state["t_curr"])
        dt = float(time_state["dt"])
        u0 = u_exact_y(x, t0)
        u1 = u_exact_y(x, t1)
        return (u1 - u0) / max(dt, 1.0e-16) - mu_f * u_exact_y_xx(x, t1)

    def f_s_y_discrete(x):
        t0 = float(time_state["t_prev"])
        t1 = float(time_state["t_curr"])
        dt = float(time_state["dt"])
        u0 = u_exact_y(x, t0)
        u1 = u_exact_y(x, t1)
        return (u1 - u0) / max(dt, 1.0e-16) - mu_s * w_exact_xx(x, t1)

    def g_disp_y_discrete(x):
        t0 = float(time_state["t_prev"])
        t1 = float(time_state["t_curr"])
        dt = float(time_state["dt"])
        d0 = w_exact(x, t0)
        d1 = w_exact(x, t1)
        u1 = u_exact_y(x, t1)
        return (d1 - d0) / max(dt, 1.0e-16) - u1

    f_f = Analytic(lambda x, y: np.stack((0.0 * x, f_f_y_discrete(x)), axis=-1), degree=4)
    f_s = Analytic(lambda x, y: np.stack((0.0 * x, f_s_y_discrete(x)), axis=-1), degree=4)
    g_d = Analytic(lambda x, y: np.stack((0.0 * x, g_disp_y_discrete(x)), axis=-1), degree=4)

    # ------------------------------------------------------------------
    # Forms + forcing
    # ------------------------------------------------------------------
    dt = Constant(float(args.dt))
    dt._jit_name = "dt"
    theta = Constant(1.0)

    du_f_R = restrict(du_f, domains["has_pos"])
    dp_f_R = restrict(dp_f, domains["has_pos"])
    v_f_R = restrict(v_f, domains["has_pos"])
    q_f_R = restrict(q_f, domains["has_pos"])
    uf_k_R = restrict(uf_k, domains["has_pos"])
    pf_k_R = restrict(pf_k, domains["has_pos"])
    uf_n_R = restrict(uf_n, domains["has_pos"])
    pf_n_R = restrict(pf_n, domains["has_pos"])

    du_s_R = restrict(du_s, domains["has_neg"])
    ddisp_s_R = restrict(ddisp_s, domains["has_neg"])
    v_s_R = restrict(v_s, domains["has_neg"])
    w_s_R = restrict(w_s, domains["has_neg"])
    us_k_R = restrict(us_k, domains["has_neg"])
    us_n_R = restrict(us_n, domains["has_neg"])
    disp_k_R = restrict(disp_k, domains["has_neg"])
    disp_n_R = restrict(disp_n, domains["has_neg"])

    forms = build_fsi_eulerian_forms(
        du_f=du_f_R,
        dp_f=dp_f_R,
        du_s=du_s_R,
        ddisp_s=ddisp_s_R,
        test_vel_f=v_f_R,
        test_q_f=q_f_R,
        test_vel_s=v_s_R,
        test_disp_s=w_s_R,
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
        kappa_pos=Constant(0.5),
        kappa_neg=Constant(0.5),
        cell_h=CellDiameter(),
        beta_N=Constant(float(args.beta_n)),
        rho_f=Constant(1.0),
        rho_s=Constant(1.0),
        mu_f=Constant(mu_f),
        mu_s=Constant(mu_s),
        lambda_s=Constant(0.0),
        dt=dt,
        theta=theta,
        gamma_v=Constant(0.0),
        gamma_p=Constant(0.0),
        gamma_v_grad=Constant(0.0),
        solid_reg_eps=Constant(0.0),
        use_linear_solid=True,
        solid_advect_lagged=True,
        s_nitsche_value=1.0,
    )

    residual_form = (
        forms.residual_form
        - dot(f_f, v_f_R) * dx_fluid
        - dot(f_s, v_s_R) * dx_solid
        - dot(g_d, w_s_R) * dx_solid
    )

    # ------------------------------------------------------------------
    # Time-dependent Dirichlet BCs (applied at t_{n+1} because theta=1)
    # ------------------------------------------------------------------
    def u_x_bc(x, y, t):
        return 0.0

    def u_y_bc(x, y, t):
        return float(u_exact_y(np.asarray(x), float(t)))

    def p_bc(x, y, t):
        return 0.0

    def d_x_bc(x, y, t):
        return 0.0

    def d_y_bc(x, y, t):
        return float(w_exact(np.asarray(x), float(t)))

    bcs = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                BoundaryCondition("u_pos_x", "dirichlet", tag, u_x_bc),
                BoundaryCondition("u_pos_y", "dirichlet", tag, u_y_bc),
                BoundaryCondition("p_pos_", "dirichlet", tag, p_bc),
                BoundaryCondition("vs_neg_x", "dirichlet", tag, u_x_bc),
                BoundaryCondition("vs_neg_y", "dirichlet", tag, u_y_bc),
                BoundaryCondition("d_neg_x", "dirichlet", tag, d_x_bc),
                BoundaryCondition("d_neg_y", "dirichlet", tag, d_y_bc),
            ]
        )
    bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]

    # ------------------------------------------------------------------
    # Newton solve: 1 BE step t0 -> t1
    # ------------------------------------------------------------------
    solver = None  # assigned below; used in closure

    def _pre_cb(_funcs):
        assert solver is not None
        t0 = float(getattr(solver, "_current_t", 0.0))
        dt_step = float(getattr(solver, "_current_dt", float(dt.value)))
        time_state["t_prev"] = t0
        time_state["t_curr"] = t0 + dt_step
        time_state["dt"] = dt_step

    solver = NewtonSolver(
        residual_form,
        forms.jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(
            newton_tol=float(args.newton_tol),
            max_newton_iter=int(args.max_newton_iter),
            line_search=False,
        ),
        quad_order=int(args.q),
        backend="jit",
        preproc_cb=_pre_cb,
    )

    # ICs at t0
    t0 = 0.0
    gd = np.asarray(dh.get_field_slice("u_pos_x"), dtype=int)
    uf_n.set_nodal_values(gd, np.zeros_like(gd, dtype=float))
    uf_k.set_nodal_values(gd, np.zeros_like(gd, dtype=float))
    gd = np.asarray(dh.get_field_slice("u_pos_y"), dtype=int)
    xy = dh.get_dof_coords("u_pos_y")
    uf_n.set_nodal_values(gd, u_exact_y(xy[:, 0], t0))
    uf_k.set_nodal_values(gd, u_exact_y(xy[:, 0], t0))

    gd = np.asarray(dh.get_field_slice("vs_neg_x"), dtype=int)
    us_n.set_nodal_values(gd, np.zeros_like(gd, dtype=float))
    us_k.set_nodal_values(gd, np.zeros_like(gd, dtype=float))
    gd = np.asarray(dh.get_field_slice("vs_neg_y"), dtype=int)
    xy = dh.get_dof_coords("vs_neg_y")
    us_n.set_nodal_values(gd, u_exact_y(xy[:, 0], t0))
    us_k.set_nodal_values(gd, u_exact_y(xy[:, 0], t0))

    gd = np.asarray(dh.get_field_slice("d_neg_x"), dtype=int)
    disp_n.set_nodal_values(gd, np.zeros_like(gd, dtype=float))
    disp_k.set_nodal_values(gd, np.zeros_like(gd, dtype=float))
    gd = np.asarray(dh.get_field_slice("d_neg_y"), dtype=int)
    xy = dh.get_dof_coords("d_neg_y")
    disp_n.set_nodal_values(gd, w_exact(xy[:, 0], t0))
    disp_k.set_nodal_values(gd, w_exact(xy[:, 0], t0))

    def _on_dt_change(dt_new: float) -> None:
        dt.value = float(dt_new)

    dt0 = float(dt.value)
    solver.solve_time_interval(
        functions=[uf_k, pf_k, us_k, disp_k],
        prev_functions=[uf_n, pf_n, us_n, disp_n],
        aux_functions={"dt": dt},
        time_params=TimeStepperParameters(
            dt=dt0,
            max_steps=1,
            final_time=dt0,
            theta=float(theta.value),
            stop_on_steady=False,
            on_dt_change=_on_dt_change,
        ),
    )

    # Report max nodal errors at t1
    t1 = dt0
    inactive = set(getattr(dh, "dof_tags", {}).get("inactive", set()))

    def _max_active_err(field: str, values: np.ndarray) -> float:
        gdofs = [int(d) for d in dh.get_field_slice(field) if int(d) not in inactive]
        if not gdofs:
            return 0.0
        gdofs = np.asarray(gdofs, dtype=int)
        return float(np.max(np.abs(values[gdofs] - values_exact[gdofs])))

    # Report max nodal errors at t1 on *active* DOFs only.
    values_exact = np.zeros(dh.total_dofs, dtype=float)

    # u_f_y exact
    gd = np.asarray(dh.get_field_slice("u_pos_y"), dtype=int)
    xy = dh.get_dof_coords("u_pos_y")
    values_exact[gd] = u_exact_y(xy[:, 0], t1)
    values = np.zeros(dh.total_dofs, dtype=float)
    values[gd] = uf_k.components[1].nodal_values
    e_u = _max_active_err("u_pos_y", values)

    # u_s_y exact
    gd = np.asarray(dh.get_field_slice("vs_neg_y"), dtype=int)
    xy = dh.get_dof_coords("vs_neg_y")
    values_exact[gd] = u_exact_y(xy[:, 0], t1)
    values = np.zeros(dh.total_dofs, dtype=float)
    values[gd] = us_k.components[1].nodal_values
    e_us = _max_active_err("vs_neg_y", values)

    # d_s_y exact
    gd = np.asarray(dh.get_field_slice("d_neg_y"), dtype=int)
    xy = dh.get_dof_coords("d_neg_y")
    values_exact[gd] = w_exact(xy[:, 0], t1)
    values = np.zeros(dh.total_dofs, dtype=float)
    values[gd] = disp_k.components[1].nodal_values
    e_d = _max_active_err("d_neg_y", values)
    print(f"[mms] max|u_f_y - u_exact| = {e_u:.3e}")
    print(f"[mms] max|u_s_y - u_exact| = {e_us:.3e}")
    print(f"[mms] max|d_s_y - d_exact| = {e_d:.3e}")


if __name__ == "__main__":
    main()
