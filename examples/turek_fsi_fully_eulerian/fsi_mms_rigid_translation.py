#!/usr/bin/env python
"""
Fully-Eulerian FSI MMS: rigid translation with moving interface.

This is a minimal "sanity MMS" to validate:
  - moving level set updates (commit + refresh_domain_sets),
  - inactive-DOF retagging under motion,
  - residual consistency for a solution with zero stresses and zero gradients.

Exact solution
--------------
Pick a constant translation velocity U = (Ux,Uy). Then
  u_f = u_s = U,   p_f = 0,
  d_s(t) = t * U,
solves the fully-Eulerian monolithic FSI system (with zero forcing), regardless
of the interface position, because all gradients vanish and the kinematic
constraint is satisfied exactly by construction.
"""

from __future__ import annotations

import argparse

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import LevelSetGridFunction
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import CellDiameter, Constant, Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction, restrict
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.utils.fsi_fully_eulerian import build_fsi_eulerian_forms, build_measures, make_domain_sets, refresh_domain_sets, retag_inactive
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


def _set_const_field(dh: DofHandler, fun, field: str, value: float) -> None:
    gd = np.asarray(dh.get_field_slice(field), dtype=int)
    fun.set_nodal_values(gd, np.full_like(gd, float(value), dtype=float))


def _free_dofs(dh: DofHandler, bcs: list[BoundaryCondition]) -> np.ndarray:
    dirichlet = dh.get_dirichlet_data(bcs) or {}
    bc = set(map(int, dirichlet.keys()))
    inactive = set(getattr(dh, "dof_tags", {}).get("inactive", set()))
    blocked = bc | inactive
    return np.array([i for i in range(int(dh.total_dofs)) if i not in blocked], dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=10)
    ap.add_argument("--ny", type=int, default=5)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--ux", type=float, default=0.2)
    ap.add_argument("--uy", type=float, default=-0.15)
    ap.add_argument("--c0", type=float, default=0.0, help="Initial interface position for φ=x-c(t)")
    ap.add_argument("--dc", type=float, default=0.1, help="Interface shift per step")
    ap.add_argument("--backend", choices=("python", "jit", "cpp"), default="python")
    ap.add_argument("--assert-tol", type=float, default=1.0e-12, help="Fail if |R|_inf exceeds this (0 disables)")
    args = ap.parse_args()

    # Mesh / geometry
    x0, x1 = -1.0, 1.0
    y0, y1 = -0.5, 0.5
    Lx, Ly = x1 - x0, y1 - y0
    nodes, elems, edges, corners = structured_quad(Lx, Ly, nx=int(args.nx), ny=int(args.ny), poly_order=1, offset=(x0, y0))
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    _tag_rect_boundaries(mesh, x0=x0, x1=x1, y0=y0, y1=y1)

    # Level set (grid function)
    ls_me = MixedElement(mesh, field_specs={"phi": 1})
    ls_dh = DofHandler(ls_me, method="cg")
    level_set = LevelSetGridFunction(ls_dh, field="phi")
    ls_coords = np.asarray(ls_dh.get_dof_coords("phi"), dtype=float)

    def _commit_interface(c: float) -> None:
        phi = ls_coords[:, 0] - float(c)
        level_set.set_from_array(phi)
        level_set.commit(tol=1.0e-12)

    _commit_interface(float(args.c0))
    domains = make_domain_sets(mesh, use_aligned_interface=False)
    dx_fluid, dx_solid, dGamma, dG_fluid, dG_solid = build_measures(mesh, level_set, domains, qvol=3)

    # Unknowns
    me = MixedElement(
        mesh,
        field_specs={
            "u_pos_x": 1,
            "u_pos_y": 1,
            "p_pos_": 1,
            "vs_neg_x": 1,
            "vs_neg_y": 1,
            "d_neg_x": 1,
            "d_neg_y": 1,
        },
    )
    dh = DofHandler(me, method="cg")

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

    dt = Constant(float(args.dt))
    theta = Constant(1.0)
    rho_f = Constant(1.0)
    rho_s = Constant(1.0)

    R = domains
    forms = build_fsi_eulerian_forms(
        du_f=restrict(du_f, R["has_pos"]),
        dp_f=restrict(dp_f, R["has_pos"]),
        du_s=restrict(du_s, R["has_neg"]),
        ddisp_s=restrict(ddisp_s, R["has_neg"]),
        test_vel_f=restrict(v_f, R["has_pos"]),
        test_q_f=restrict(q_f, R["has_pos"]),
        test_vel_s=restrict(v_s, R["has_neg"]),
        test_disp_s=restrict(w_s, R["has_neg"]),
        uf_k=restrict(uf_k, R["has_pos"]),
        pf_k=restrict(pf_k, R["has_pos"]),
        uf_n=restrict(uf_n, R["has_pos"]),
        pf_n=restrict(pf_n, R["has_pos"]),
        us_k=restrict(us_k, R["has_neg"]),
        us_n=restrict(us_n, R["has_neg"]),
        disp_k=restrict(disp_k, R["has_neg"]),
        disp_n=restrict(disp_n, R["has_neg"]),
        dx_fluid=dx_fluid,
        dx_solid=dx_solid,
        dGamma=dGamma,
        dG_fluid=dG_fluid,
        dG_solid=dG_solid,
        kappa_pos=Constant(0.5),
        kappa_neg=Constant(0.5),
        cell_h=CellDiameter(),
        beta_N=Constant(20.0),
        rho_f=rho_f,
        rho_s=rho_s,
        mu_f=Constant(1.0),
        mu_s=Constant(1.0),
        lambda_s=Constant(0.0),
        dt=dt,
        theta=theta,
        gamma_v=Constant(0.0),
        gamma_p=Constant(0.0),
        gamma_v_grad=Constant(0.0),
        svc_scale=rho_s / dt,
        solid_reg_eps=Constant(0.0),
        use_linear_solid=True,
        solid_advect_lagged=True,
        s_nitsche_value=0.0,
    )

    ux, uy = float(args.ux), float(args.uy)
    bcs: list[BoundaryCondition] = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                BoundaryCondition("u_pos_x", "dirichlet", tag, lambda x, y, t: ux),
                BoundaryCondition("u_pos_y", "dirichlet", tag, lambda x, y, t: uy),
                BoundaryCondition("vs_neg_x", "dirichlet", tag, lambda x, y, t: ux),
                BoundaryCondition("vs_neg_y", "dirichlet", tag, lambda x, y, t: uy),
                BoundaryCondition("d_neg_x", "dirichlet", tag, lambda x, y, t: ux * float(t)),
                BoundaryCondition("d_neg_y", "dirichlet", tag, lambda x, y, t: uy * float(t)),
            ]
        )

    for n in range(int(args.steps)):
        t0 = float(n) * float(dt.value)
        t1 = float(n + 1) * float(dt.value)

        c = float(args.c0) + float(n + 1) * float(args.dc)
        _commit_interface(c)
        refresh_domain_sets(mesh, domains)
        retag_inactive(dh, mesh)

        # u^{n+1} = u^n = U, p=0, d^{n+1}-d^n = dt*U
        _set_const_field(dh, uf_n, "u_pos_x", ux)
        _set_const_field(dh, uf_n, "u_pos_y", uy)
        _set_const_field(dh, uf_k, "u_pos_x", ux)
        _set_const_field(dh, uf_k, "u_pos_y", uy)

        _set_const_field(dh, us_n, "vs_neg_x", ux)
        _set_const_field(dh, us_n, "vs_neg_y", uy)
        _set_const_field(dh, us_k, "vs_neg_x", ux)
        _set_const_field(dh, us_k, "vs_neg_y", uy)

        _set_const_field(dh, disp_n, "d_neg_x", ux * t0)
        _set_const_field(dh, disp_n, "d_neg_y", uy * t0)
        _set_const_field(dh, disp_k, "d_neg_x", ux * t1)
        _set_const_field(dh, disp_k, "d_neg_y", uy * t1)

        pf_n.nodal_values.fill(0.0)
        pf_k.nodal_values.fill(0.0)

        _, F = assemble_form(Equation(None, forms.residual_form), dof_handler=dh, bcs=[], backend=str(args.backend))
        free = _free_dofs(dh, bcs)
        res_inf = float(np.linalg.norm(F[free], ord=np.inf)) if free.size else 0.0
        print(f"[rigid] step={n+1:03d} t={t1:.3e} c={c:+.3e}  |R|_inf(free)={res_inf:.3e}")

        tol = float(args.assert_tol)
        if tol > 0.0 and res_inf > tol:
            raise RuntimeError(f"Rigid-translation MMS residual too large: {res_inf:.3e} > {tol:.3e}")


if __name__ == "__main__":
    main()

