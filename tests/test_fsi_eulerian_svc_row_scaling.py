import numpy as np
import pytest
import scipy.sparse as sp

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import CellDiameter, Constant, Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction, restrict
from pycutfem.ufl.forms import Equation, assemble_form
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


def test_fsi_eulerian_svc_scale_is_row_scaling() -> None:
    """
    `svc_scale` is intended as a *pure row scaling* of the Eulerian kinematic constraint.

    This test asserts that assembling `a_svc` with `svc_scale=rho_s/dt` produces a matrix
    that is exactly scaled compared to `svc_scale=1`.
    """
    # Small mesh for a fast assembly-only test.
    x0, x1 = -1.0, 1.0
    y0, y1 = -0.5, 0.5
    Lx, Ly = x1 - x0, y1 - y0
    nx, ny = 2, 1
    poly_u = 2
    poly_p = 1
    poly_d = 1
    q = 3

    nodes, elems, edges, corners = structured_quad(Lx, Ly, nx=nx, ny=ny, poly_order=poly_u, offset=(x0, y0))
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_u,
    )
    _tag_rect_boundaries(mesh, x0=x0, x1=x1, y0=y0, y1=y1)

    level_set = AffineLevelSet(a=1.0, b=1.0, c=0.0)
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)
    domains = make_domain_sets(mesh, use_aligned_interface=False)

    dx_fluid, dx_solid, dGamma, dG_fluid, dG_solid = build_measures(mesh, level_set, domains, qvol=q)

    me = MixedElement(
        mesh,
        field_specs={
            "u_pos_x": poly_u,
            "u_pos_y": poly_u,
            "p_pos_": poly_p,
            "vs_neg_x": poly_u,
            "vs_neg_y": poly_u,
            "d_neg_x": poly_d,
            "d_neg_y": poly_d,
        },
    )
    dh = DofHandler(me, method="cg")
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

    dt_val = 0.01
    dt = Constant(dt_val)
    theta = Constant(1.0)
    rho_f = Constant(1.0)
    rho_s = Constant(2.5)  # non-1 to make the scaling factor obvious

    def _forms(svc_scale):
        return build_fsi_eulerian_forms(
            du_f=restrict(du_f, domains["has_pos"]),
            dp_f=restrict(dp_f, domains["has_pos"]),
            du_s=restrict(du_s, domains["has_neg"]),
            ddisp_s=restrict(ddisp_s, domains["has_neg"]),
            test_vel_f=restrict(v_f, domains["has_pos"]),
            test_q_f=restrict(q_f, domains["has_pos"]),
            test_vel_s=restrict(v_s, domains["has_neg"]),
            test_disp_s=restrict(w_s, domains["has_neg"]),
            uf_k=restrict(uf_k, domains["has_pos"]),
            pf_k=restrict(pf_k, domains["has_pos"]),
            uf_n=restrict(uf_n, domains["has_pos"]),
            pf_n=restrict(pf_n, domains["has_pos"]),
            us_k=restrict(us_k, domains["has_neg"]),
            us_n=restrict(us_n, domains["has_neg"]),
            disp_k=restrict(disp_k, domains["has_neg"]),
            disp_n=restrict(disp_n, domains["has_neg"]),
            dx_fluid=dx_fluid,
            dx_solid=dx_solid,
            dGamma=dGamma,
            dG_fluid=dG_fluid,
            dG_solid=dG_solid,
            kappa_pos=Constant(0.5),
            kappa_neg=Constant(0.5),
            cell_h=CellDiameter(),
            beta_N=Constant(0.0),
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
            svc_scale=svc_scale,
            solid_reg_eps=Constant(0.0),
            use_linear_solid=True,
            solid_advect_lagged=True,
            s_nitsche_value=0.0,
        )

    forms1 = _forms(Constant(1.0))
    forms2 = _forms(rho_s / dt)

    A1, _ = assemble_form(Equation(forms1.a_svc, None), dof_handler=dh, bcs=[], backend="python")
    A2, _ = assemble_form(Equation(forms2.a_svc, None), dof_handler=dh, bcs=[], backend="python")

    assert sp.isspmatrix(A1) and sp.isspmatrix(A2)
    scale = float(rho_s.value) / float(dt.value)
    diff = (A2 - scale * A1).tocsr()
    if diff.nnz:
        assert float(np.max(np.abs(diff.data))) < 1.0e-12
