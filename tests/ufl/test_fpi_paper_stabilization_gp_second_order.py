import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    restrict,
)
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.utils.fpi_fully_eulerian import build_fpi_eulerian_forms
from pycutfem.utils.fsi_fully_eulerian import build_measures, make_domain_sets
from pycutfem.utils.meshgen import structured_quad


def _build_cut_problem(*, nx: int, poly_order: int, qdeg: int):
    nodes, elems, edges, corners = structured_quad(
        2.0, 2.0, nx=nx, ny=nx, poly_order=poly_order, offset=(-1.0, -1.0)
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )
    # Non-aligned interface so ghost facets exist.
    level_set = AffineLevelSet(a=1.0, b=0.0, c=0.1)
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)
    domains = make_domain_sets(mesh, use_aligned_interface=False)
    dx_f, dx_p, dGamma, dG_f, dG_p = build_measures(mesh, level_set, domains, qvol=qdeg)

    me = MixedElement(
        mesh,
        field_specs={
            "v_pos_x": poly_order,
            "v_pos_y": poly_order,
            "p_pos_": poly_order,
            "v_neg_x": poly_order,
            "v_neg_y": poly_order,
            "u_neg_x": poly_order,
            "u_neg_y": poly_order,
            "p_neg_": poly_order,
        },
    )
    dh = DofHandler(me, method="cg")

    Vf = FunctionSpace(name="Vf", field_names=["v_pos_x", "v_pos_y"], dim=1, side="+")
    Vp = FunctionSpace(name="Vp", field_names=["v_neg_x", "v_neg_y"], dim=1, side="-")
    Up = FunctionSpace(name="Up", field_names=["u_neg_x", "u_neg_y"], dim=1, side="-")

    dvF = VectorTrialFunction(space=Vf, dof_handler=dh)
    dpF = TrialFunction(name="dpF", field_name="p_pos_", dof_handler=dh, side="+")
    dvP = VectorTrialFunction(space=Vp, dof_handler=dh)
    duP = VectorTrialFunction(space=Up, dof_handler=dh)
    dpP = TrialFunction(name="dpP", field_name="p_neg_", dof_handler=dh, side="-")

    vF_test = VectorTestFunction(space=Vf, dof_handler=dh)
    qF_test = TestFunction(name="qF", field_name="p_pos_", dof_handler=dh, side="+")
    vP_test = VectorTestFunction(space=Vp, dof_handler=dh)
    uP_test = VectorTestFunction(space=Up, dof_handler=dh)
    qP_test = TestFunction(name="qP", field_name="p_neg_", dof_handler=dh, side="-")

    vF_k = VectorFunction(name="vF_k", field_names=["v_pos_x", "v_pos_y"], dof_handler=dh, side="+")
    pF_k = Function(name="pF_k", field_name="p_pos_", dof_handler=dh, side="+")
    vP_k = VectorFunction(name="vP_k", field_names=["v_neg_x", "v_neg_y"], dof_handler=dh, side="-")
    uP_k = VectorFunction(name="uP_k", field_names=["u_neg_x", "u_neg_y"], dof_handler=dh, side="-")
    pP_k = Function(name="pP_k", field_name="p_neg_", dof_handler=dh, side="-")

    vF_n = VectorFunction(name="vF_n", field_names=["v_pos_x", "v_pos_y"], dof_handler=dh, side="+")
    pF_n = Function(name="pF_n", field_name="p_pos_", dof_handler=dh, side="+")
    vP_n = VectorFunction(name="vP_n", field_names=["v_neg_x", "v_neg_y"], dof_handler=dh, side="-")
    uP_n = VectorFunction(name="uP_n", field_names=["u_neg_x", "u_neg_y"], dof_handler=dh, side="-")
    pP_n = Function(name="pP_n", field_name="p_neg_", dof_handler=dh, side="-")

    for f in (vF_k, pF_k, vP_k, uP_k, pP_k, vF_n, pF_n, vP_n, uP_n, pP_n):
        f.nodal_values.fill(0.0)

    has_pos = domains["has_pos"]
    has_neg = domains["has_neg"]

    return dict(
        level_set=level_set,
        dx_f=dx_f,
        dx_p=dx_p,
        dGamma=dGamma,
        dG_f=dG_f,
        dG_p=dG_p,
        vF_kR=restrict(vF_k, has_pos),
        pF_kR=restrict(pF_k, has_pos),
        vP_kR=restrict(vP_k, has_neg),
        uP_kR=restrict(uP_k, has_neg),
        pP_kR=restrict(pP_k, has_neg),
        vF_nR=restrict(vF_n, has_pos),
        pF_nR=restrict(pF_n, has_pos),
        vP_nR=restrict(vP_n, has_neg),
        uP_nR=restrict(uP_n, has_neg),
        pP_nR=restrict(pP_n, has_neg),
        uP_nm1R=restrict(uP_n, has_neg),
        dvF_R=restrict(dvF, has_pos),
        dpF_R=restrict(dpF, has_pos),
        dvP_R=restrict(dvP, has_neg),
        duP_R=restrict(duP, has_neg),
        dpP_R=restrict(dpP, has_neg),
        vF_testR=restrict(vF_test, has_pos),
        qF_testR=restrict(qF_test, has_pos),
        vP_testR=restrict(vP_test, has_neg),
        uP_testR=restrict(uP_test, has_neg),
        qP_testR=restrict(qP_test, has_neg),
    )


def test_paper_gp_second_order_terms_are_present_in_form_repr():
    prob = _build_cut_problem(nx=2, poly_order=2, qdeg=4)

    common = dict(
        vF_k=prob["vF_kR"],
        pF_k=prob["pF_kR"],
        vP_k=prob["vP_kR"],
        uP_k=prob["uP_kR"],
        pP_k=prob["pP_kR"],
        vF_n=prob["vF_nR"],
        pF_n=prob["pF_nR"],
        vP_n=prob["vP_nR"],
        uP_n=prob["uP_nR"],
        pP_n=prob["pP_nR"],
        uP_nm1=prob["uP_nm1R"],
        dvF=prob["dvF_R"],
        dpF=prob["dpF_R"],
        dvP=prob["dvP_R"],
        duP=prob["duP_R"],
        dpP=prob["dpP_R"],
        vF_test=prob["vF_testR"],
        qF_test=prob["qF_testR"],
        vP_test=prob["vP_testR"],
        uP_test=prob["uP_testR"],
        qP_test=prob["qP_testR"],
        dx_f=prob["dx_f"],
        dx_p=prob["dx_p"],
        dGamma=prob["dGamma"],
        dG_f=prob["dG_f"],
        dG_p=prob["dG_p"],
        level_set=prob["level_set"],
        dt=Constant(0.05),
        theta=1.0,
        rho_f=Constant(1.0),
        mu_f=Constant(1.0),
        rho_s0_tilde=Constant(1.0),
        porosity=Constant(0.5),
        K_inv=Constant(((1.0 / 0.1), 0.0, 0.0, (1.0 / 0.1)), dim=2),
        c_nh=Constant(1.0),
        beta_nh=Constant(1.0),
        beta_BJ=Constant(1.0),
        kappa=Constant(1.0),
        gamma_n=Constant(1.0),
        gamma_t=Constant(1.0),
        zeta=1.0,
        use_interface_terms=False,
        use_stabilization=True,
        use_paper_stabilization=True,
        poly_order=2,
    )

    forms_no_j2 = build_fpi_eulerian_forms(**common, gp_second_weight=0.0)
    forms_j2 = build_fpi_eulerian_forms(**common, gp_second_weight=0.05)

    rep_no_j2 = repr(forms_no_j2.a_gp)
    rep_j2 = repr(forms_j2.a_gp)

    # Second-normal-derivative term (dn2) appears as nested Grad(Dot(Grad(...), n)).
    assert "Grad(Dot(Grad" not in rep_no_j2
    assert "Grad(Dot(Grad" in rep_j2

    # Measure metadata must request second derivatives when gp_second_weight is active.
    has_second_derivs = ("(0, 2)" in rep_j2) or ("(2, 0)" in rep_j2) or ("(1, 1)" in rep_j2)
    assert "(0, 2)" not in rep_no_j2 and "(2, 0)" not in rep_no_j2 and "(1, 1)" not in rep_no_j2
    assert has_second_derivs
