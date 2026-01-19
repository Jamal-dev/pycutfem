import numpy as np
import pytest

from pycutfem.core.mesh import Mesh
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.meshgen import structured_quad
from pycutfem.utils.fsi_fully_eulerian import build_fsi_eulerian_forms, build_measures, make_domain_sets
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    FacetNormal,
    Function,
    Identity,
    Neg,
    Pos,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    dot,
    grad,
    restrict,
    trace,
)
from pycutfem.ufl.forms import Equation, assemble_form


def _set_linear_field(component, coords, *, x_coeff=1.0, y_coeff=0.0, offset=0.0):
    values = x_coeff * coords[:, 0] + y_coeff * coords[:, 1] + offset
    try:
        dh = component._dof_handler
        gdofs = np.asarray(dh.get_field_slice(component.field_name), dtype=int)
        component.set_nodal_values(gdofs, values)
    except Exception:
        component.nodal_values[:] = values


def test_fsi_eulerian_interface_traction_balances_for_matching_fields():
    poly_order = 1
    nodes, elems, edges, corners = structured_quad(
        2.0,
        2.0,
        nx=1,
        ny=1,
        poly_order=poly_order,
        offset=(-1.0, -1.0),
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )

    level_set = AffineLevelSet(a=1.0, b=0.0, c=0.0)  # interface x=0
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)

    domains = make_domain_sets(mesh, use_aligned_interface=False)
    dx_fluid, dx_solid, dGamma, dG_fluid, dG_solid = build_measures(mesh, level_set, domains, qvol=3)

    me = MixedElement(
        mesh,
        field_specs={
            "u_pos_x": poly_order,
            "u_pos_y": poly_order,
            "p_pos_": poly_order,
            "vs_neg_x": poly_order,
            "vs_neg_y": poly_order,
            "d_neg_x": poly_order,
            "d_neg_y": poly_order,
        },
    )
    dh = DofHandler(me, method="cg")

    velocity_fluid_space = FunctionSpace(name="velocity_fluid", field_names=["u_pos_x", "u_pos_y"], dim=1, side="+")
    velocity_solid_space = FunctionSpace(name="velocity_solid", field_names=["vs_neg_x", "vs_neg_y"], dim=1, side="-")
    displacement_space = FunctionSpace(name="displacement", field_names=["d_neg_x", "d_neg_y"], dim=1, side="-")

    du_f = VectorTrialFunction(space=velocity_fluid_space, dof_handler=dh)
    dp_f = TrialFunction(name="trial_pressure_fluid", field_name="p_pos_", dof_handler=dh, side="+")
    du_s = VectorTrialFunction(space=velocity_solid_space, dof_handler=dh)
    ddisp_s = VectorTrialFunction(space=displacement_space, dof_handler=dh)
    test_vel_f = VectorTestFunction(space=velocity_fluid_space, dof_handler=dh)
    test_q_f = TestFunction(name="test_pressure_fluid", field_name="p_pos_", dof_handler=dh, side="+")
    test_vel_s = VectorTestFunction(space=velocity_solid_space, dof_handler=dh)
    test_disp_s = VectorTestFunction(space=displacement_space, dof_handler=dh)

    uf_k = VectorFunction(name="u_f_k", field_names=["u_pos_x", "u_pos_y"], dof_handler=dh, side="+")
    pf_k = Function(name="p_f_k", field_name="p_pos_", dof_handler=dh, side="+")
    uf_n = VectorFunction(name="u_f_n", field_names=["u_pos_x", "u_pos_y"], dof_handler=dh, side="+")
    pf_n = Function(name="p_f_n", field_name="p_pos_", dof_handler=dh, side="+")
    us_k = VectorFunction(name="u_s_k", field_names=["vs_neg_x", "vs_neg_y"], dof_handler=dh, side="-")
    us_n = VectorFunction(name="u_s_n", field_names=["vs_neg_x", "vs_neg_y"], dof_handler=dh, side="-")
    disp_k = VectorFunction(name="disp_k", field_names=["d_neg_x", "d_neg_y"], dof_handler=dh, side="-")
    disp_n = VectorFunction(name="disp_n", field_names=["d_neg_x", "d_neg_y"], dof_handler=dh, side="-")

    for func in (uf_k, pf_k, uf_n, pf_n, us_k, us_n, disp_k, disp_n):
        func.nodal_values.fill(0.0)

    # Linear field u(x) = x, v = 0; same for solid velocity and displacement.
    _set_linear_field(uf_k.components[0], dh.get_dof_coords("u_pos_x"))
    uf_k.components[1].nodal_values.fill(0.0)
    _set_linear_field(us_k.components[0], dh.get_dof_coords("vs_neg_x"))
    us_k.components[1].nodal_values.fill(0.0)
    _set_linear_field(disp_k.components[0], dh.get_dof_coords("d_neg_x"))
    disp_k.components[1].nodal_values.fill(0.0)

    du_f_R = restrict(du_f, domains["has_pos"])
    dp_f_R = restrict(dp_f, domains["has_pos"])
    test_vel_f_R = restrict(test_vel_f, domains["has_pos"])
    test_q_f_R = restrict(test_q_f, domains["has_pos"])
    uf_k_R = restrict(uf_k, domains["has_pos"])
    uf_n_R = restrict(uf_n, domains["has_pos"])
    pf_k_R = restrict(pf_k, domains["has_pos"])
    pf_n_R = restrict(pf_n, domains["has_pos"])

    du_s_R = restrict(du_s, domains["has_neg"])
    ddisp_s_R = restrict(ddisp_s, domains["has_neg"])
    test_vel_s_R = restrict(test_vel_s, domains["has_neg"])
    test_disp_s_R = restrict(test_disp_s, domains["has_neg"])
    us_k_R = restrict(us_k, domains["has_neg"])
    us_n_R = restrict(us_n, domains["has_neg"])
    disp_k_R = restrict(disp_k, domains["has_neg"])
    disp_n_R = restrict(disp_n, domains["has_neg"])

    forms = build_fsi_eulerian_forms(
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
        kappa_pos=Constant(0.5),
        kappa_neg=Constant(0.5),
        cell_h=CellDiameter(),
        beta_N=Constant(0.0),
        rho_f=Constant(1.0),
        rho_s=Constant(1.0),
        mu_f=Constant(1.0),
        mu_s=Constant(1.0),
        lambda_s=Constant(0.0),
        dt=Constant(1.0),
        theta=Constant(1.0),
        gamma_v=Constant(0.0),
        gamma_p=Constant(0.0),
        gamma_v_grad=Constant(0.0),
        solid_reg_eps=Constant(0.0),
        use_linear_solid=True,
        solid_advect_lagged=True,
        s_nitsche_value=0.0,
    )

    n = FacetNormal()
    I2 = Identity(2)

    def _epsilon(u):
        return 0.5 * (grad(u) + grad(u).T)

    # IBP boundary term on Γ:
    #   B_Γ(v_f,v_s) = ∫_Γ t_f·v_f − t_s·v_s
    # with t_f = σ_f n and t_s = σ_s n using the same normal n (solid→fluid).
    t_f = 2.0 * Constant(1.0) * dot(_epsilon(Pos(uf_k_R)), n) - Pos(pf_k_R) * n
    eps_s = _epsilon(Neg(disp_k_R))
    sigma_s = Constant(2.0) * Constant(1.0) * eps_s + Constant(0.0) * trace(eps_s) * I2
    t_s = dot(sigma_s, n)
    B = (dot(t_f, Pos(test_vel_f_R)) - dot(t_s, Neg(test_vel_s_R))) * dGamma

    W = B + forms.R_int
    _, residual = assemble_form(Equation(None, W), dof_handler=dh, bcs=[], backend="python")
    fluid_dofs = np.concatenate([dh.get_field_slice("u_pos_x"), dh.get_field_slice("u_pos_y")])
    solid_dofs = np.concatenate([dh.get_field_slice("vs_neg_x"), dh.get_field_slice("vs_neg_y")])
    assert np.linalg.norm(residual[fluid_dofs], ord=np.inf) < 1.0e-10
    assert np.linalg.norm(residual[solid_dofs], ord=np.inf) < 1.0e-10


def test_fsi_eulerian_interface_traction_nonzero_for_mismatched_fields():
    poly_order = 1
    nodes, elems, edges, corners = structured_quad(
        2.0,
        2.0,
        nx=1,
        ny=1,
        poly_order=poly_order,
        offset=(-1.0, -1.0),
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )

    level_set = AffineLevelSet(a=1.0, b=0.0, c=0.0)  # interface x=0
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)

    domains = make_domain_sets(mesh, use_aligned_interface=False)
    dx_fluid, dx_solid, dGamma, dG_fluid, dG_solid = build_measures(mesh, level_set, domains, qvol=3)

    me = MixedElement(
        mesh,
        field_specs={
            "u_pos_x": poly_order,
            "u_pos_y": poly_order,
            "p_pos_": poly_order,
            "vs_neg_x": poly_order,
            "vs_neg_y": poly_order,
            "d_neg_x": poly_order,
            "d_neg_y": poly_order,
        },
    )
    dh = DofHandler(me, method="cg")

    velocity_fluid_space = FunctionSpace(name="velocity_fluid", field_names=["u_pos_x", "u_pos_y"], dim=1, side="+")
    velocity_solid_space = FunctionSpace(name="velocity_solid", field_names=["vs_neg_x", "vs_neg_y"], dim=1, side="-")
    displacement_space = FunctionSpace(name="displacement", field_names=["d_neg_x", "d_neg_y"], dim=1, side="-")

    du_f = VectorTrialFunction(space=velocity_fluid_space, dof_handler=dh)
    dp_f = TrialFunction(name="trial_pressure_fluid", field_name="p_pos_", dof_handler=dh, side="+")
    du_s = VectorTrialFunction(space=velocity_solid_space, dof_handler=dh)
    ddisp_s = VectorTrialFunction(space=displacement_space, dof_handler=dh)
    test_vel_f = VectorTestFunction(space=velocity_fluid_space, dof_handler=dh)
    test_q_f = TestFunction(name="test_pressure_fluid", field_name="p_pos_", dof_handler=dh, side="+")
    test_vel_s = VectorTestFunction(space=velocity_solid_space, dof_handler=dh)
    test_disp_s = VectorTestFunction(space=displacement_space, dof_handler=dh)

    uf_k = VectorFunction(name="u_f_k", field_names=["u_pos_x", "u_pos_y"], dof_handler=dh, side="+")
    pf_k = Function(name="p_f_k", field_name="p_pos_", dof_handler=dh, side="+")
    uf_n = VectorFunction(name="u_f_n", field_names=["u_pos_x", "u_pos_y"], dof_handler=dh, side="+")
    pf_n = Function(name="p_f_n", field_name="p_pos_", dof_handler=dh, side="+")
    us_k = VectorFunction(name="u_s_k", field_names=["vs_neg_x", "vs_neg_y"], dof_handler=dh, side="-")
    us_n = VectorFunction(name="u_s_n", field_names=["vs_neg_x", "vs_neg_y"], dof_handler=dh, side="-")
    disp_k = VectorFunction(name="disp_k", field_names=["d_neg_x", "d_neg_y"], dof_handler=dh, side="-")
    disp_n = VectorFunction(name="disp_n", field_names=["d_neg_x", "d_neg_y"], dof_handler=dh, side="-")

    for func in (uf_k, pf_k, uf_n, pf_n, us_k, us_n, disp_k, disp_n):
        func.nodal_values.fill(0.0)

    # Fluid is nonzero, solid is zero -> interface traction should not cancel.
    _set_linear_field(uf_k.components[0], dh.get_dof_coords("u_pos_x"))
    uf_k.components[1].nodal_values.fill(0.0)

    du_f_R = restrict(du_f, domains["has_pos"])
    dp_f_R = restrict(dp_f, domains["has_pos"])
    test_vel_f_R = restrict(test_vel_f, domains["has_pos"])
    test_q_f_R = restrict(test_q_f, domains["has_pos"])
    uf_k_R = restrict(uf_k, domains["has_pos"])
    uf_n_R = restrict(uf_n, domains["has_pos"])
    pf_k_R = restrict(pf_k, domains["has_pos"])
    pf_n_R = restrict(pf_n, domains["has_pos"])

    du_s_R = restrict(du_s, domains["has_neg"])
    ddisp_s_R = restrict(ddisp_s, domains["has_neg"])
    test_vel_s_R = restrict(test_vel_s, domains["has_neg"])
    test_disp_s_R = restrict(test_disp_s, domains["has_neg"])
    us_k_R = restrict(us_k, domains["has_neg"])
    us_n_R = restrict(us_n, domains["has_neg"])
    disp_k_R = restrict(disp_k, domains["has_neg"])
    disp_n_R = restrict(disp_n, domains["has_neg"])

    forms = build_fsi_eulerian_forms(
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
        kappa_pos=Constant(0.5),
        kappa_neg=Constant(0.5),
        cell_h=CellDiameter(),
        beta_N=Constant(0.0),
        rho_f=Constant(1.0),
        rho_s=Constant(1.0),
        mu_f=Constant(1.0),
        mu_s=Constant(1.0),
        lambda_s=Constant(0.0),
        dt=Constant(1.0),
        theta=Constant(1.0),
        gamma_v=Constant(0.0),
        gamma_p=Constant(0.0),
        gamma_v_grad=Constant(0.0),
        solid_reg_eps=Constant(0.0),
        use_linear_solid=True,
        solid_advect_lagged=True,
        s_nitsche_value=0.0,
    )

    n = FacetNormal()

    def _epsilon(u):
        return 0.5 * (grad(u) + grad(u).T)

    t_f = 2.0 * Constant(1.0) * dot(_epsilon(Pos(uf_k_R)), n) - Pos(pf_k_R) * n
    # Solid displacement is zero => σ_s = 0 => t_s = 0.
    t_s = Constant(0.0) * n
    B = (dot(t_f, Pos(test_vel_f_R)) - dot(t_s, Neg(test_vel_s_R))) * dGamma

    W = B + forms.R_int
    _, residual = assemble_form(Equation(None, W), dof_handler=dh, bcs=[], backend="python")
    solid_dofs = np.concatenate([dh.get_field_slice("vs_neg_x"), dh.get_field_slice("vs_neg_y")])
    assert np.linalg.norm(residual[solid_dofs], ord=np.inf) > 1.0e-6


@pytest.mark.parametrize("backend", ["python", "jit"])
def test_fsi_eulerian_interface_traction_nonzero_aligned_interface(backend):
    poly_order = 1
    nodes, elems, edges, corners = structured_quad(
        2.0,
        2.0,
        nx=2,
        ny=1,
        poly_order=poly_order,
        offset=(-1.0, -1.0),
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )

    level_set = AffineLevelSet(a=1.0, b=0.0, c=0.0)  # interface x=0 (aligned edge)
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set, tol=1.0e-8)
    mesh.build_interface_segments(level_set)

    assert mesh.edge_bitset("interface").cardinality() > 0
    assert mesh.element_bitset("cut").cardinality() == 0

    domains = make_domain_sets(mesh, use_aligned_interface=True)
    dx_fluid, dx_solid, dGamma, dG_fluid, dG_solid = build_measures(mesh, level_set, domains, qvol=3)

    me = MixedElement(
        mesh,
        field_specs={
            "u_pos_x": poly_order,
            "u_pos_y": poly_order,
            "p_pos_": poly_order,
            "vs_neg_x": poly_order,
            "vs_neg_y": poly_order,
            "d_neg_x": poly_order,
            "d_neg_y": poly_order,
        },
    )
    dh = DofHandler(me, method="cg")

    velocity_fluid_space = FunctionSpace(name="velocity_fluid", field_names=["u_pos_x", "u_pos_y"], dim=1, side="+")
    velocity_solid_space = FunctionSpace(name="velocity_solid", field_names=["vs_neg_x", "vs_neg_y"], dim=1, side="-")
    displacement_space = FunctionSpace(name="displacement", field_names=["d_neg_x", "d_neg_y"], dim=1, side="-")

    du_f = VectorTrialFunction(space=velocity_fluid_space, dof_handler=dh)
    dp_f = TrialFunction(name="trial_pressure_fluid", field_name="p_pos_", dof_handler=dh, side="+")
    du_s = VectorTrialFunction(space=velocity_solid_space, dof_handler=dh)
    ddisp_s = VectorTrialFunction(space=displacement_space, dof_handler=dh)
    test_vel_f = VectorTestFunction(space=velocity_fluid_space, dof_handler=dh)
    test_q_f = TestFunction(name="test_pressure_fluid", field_name="p_pos_", dof_handler=dh, side="+")
    test_vel_s = VectorTestFunction(space=velocity_solid_space, dof_handler=dh)
    test_disp_s = VectorTestFunction(space=displacement_space, dof_handler=dh)

    uf_k = VectorFunction(name="u_f_k", field_names=["u_pos_x", "u_pos_y"], dof_handler=dh, side="+")
    pf_k = Function(name="p_f_k", field_name="p_pos_", dof_handler=dh, side="+")
    uf_n = VectorFunction(name="u_f_n", field_names=["u_pos_x", "u_pos_y"], dof_handler=dh, side="+")
    pf_n = Function(name="p_f_n", field_name="p_pos_", dof_handler=dh, side="+")
    us_k = VectorFunction(name="u_s_k", field_names=["vs_neg_x", "vs_neg_y"], dof_handler=dh, side="-")
    us_n = VectorFunction(name="u_s_n", field_names=["vs_neg_x", "vs_neg_y"], dof_handler=dh, side="-")
    disp_k = VectorFunction(name="disp_k", field_names=["d_neg_x", "d_neg_y"], dof_handler=dh, side="-")
    disp_n = VectorFunction(name="disp_n", field_names=["d_neg_x", "d_neg_y"], dof_handler=dh, side="-")

    for func in (uf_k, pf_k, uf_n, pf_n, us_k, us_n, disp_k, disp_n):
        func.nodal_values.fill(0.0)

    _set_linear_field(uf_k.components[0], dh.get_dof_coords("u_pos_x"))
    uf_k.components[1].nodal_values.fill(0.0)

    du_f_R = restrict(du_f, domains["has_pos"])
    dp_f_R = restrict(dp_f, domains["has_pos"])
    test_vel_f_R = restrict(test_vel_f, domains["has_pos"])
    test_q_f_R = restrict(test_q_f, domains["has_pos"])
    uf_k_R = restrict(uf_k, domains["has_pos"])
    uf_n_R = restrict(uf_n, domains["has_pos"])
    pf_k_R = restrict(pf_k, domains["has_pos"])
    pf_n_R = restrict(pf_n, domains["has_pos"])

    du_s_R = restrict(du_s, domains["has_neg"])
    ddisp_s_R = restrict(ddisp_s, domains["has_neg"])
    test_vel_s_R = restrict(test_vel_s, domains["has_neg"])
    test_disp_s_R = restrict(test_disp_s, domains["has_neg"])
    us_k_R = restrict(us_k, domains["has_neg"])
    us_n_R = restrict(us_n, domains["has_neg"])
    disp_k_R = restrict(disp_k, domains["has_neg"])
    disp_n_R = restrict(disp_n, domains["has_neg"])

    forms = build_fsi_eulerian_forms(
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
        kappa_pos=Constant(0.5),
        kappa_neg=Constant(0.5),
        cell_h=CellDiameter(),
        beta_N=Constant(0.0),
        rho_f=Constant(1.0),
        rho_s=Constant(1.0),
        mu_f=Constant(1.0),
        mu_s=Constant(1.0),
        lambda_s=Constant(0.0),
        dt=Constant(1.0),
        theta=Constant(1.0),
        gamma_v=Constant(0.0),
        gamma_p=Constant(0.0),
        gamma_v_grad=Constant(0.0),
        solid_reg_eps=Constant(0.0),
        use_linear_solid=True,
        solid_advect_lagged=True,
        s_nitsche_value=0.0,
    )

    _, residual = assemble_form(Equation(None, forms.R_int), dof_handler=dh, bcs=[], backend=backend)
    solid_dofs = np.concatenate([dh.get_field_slice("vs_neg_x"), dh.get_field_slice("vs_neg_y")])
    assert np.linalg.norm(residual[solid_dofs], ord=np.inf) > 1.0e-6


def test_fsi_eulerian_interface_pressure_sign_aligned():
    poly_order = 1
    nodes, elems, edges, corners = structured_quad(
        2.0,
        2.0,
        nx=2,
        ny=1,
        poly_order=poly_order,
        offset=(-1.0, -1.0),
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )

    level_set = AffineLevelSet(a=1.0, b=0.0, c=0.0)  # interface x=0 (aligned edge)
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set, tol=1.0e-8)
    mesh.build_interface_segments(level_set)

    domains = make_domain_sets(mesh, use_aligned_interface=True)
    dx_fluid, dx_solid, dGamma, dG_fluid, dG_solid = build_measures(mesh, level_set, domains, qvol=3)

    me = MixedElement(
        mesh,
        field_specs={
            "u_pos_x": poly_order,
            "u_pos_y": poly_order,
            "p_pos_": poly_order,
            "vs_neg_x": poly_order,
            "vs_neg_y": poly_order,
            "d_neg_x": poly_order,
            "d_neg_y": poly_order,
        },
    )
    dh = DofHandler(me, method="cg")

    velocity_fluid_space = FunctionSpace(name="velocity_fluid", field_names=["u_pos_x", "u_pos_y"], dim=1, side="+")
    pressure_fluid_space = FunctionSpace(name="pressure_fluid", field_names=["p_pos_"], dim=0, side="+")
    velocity_solid_space = FunctionSpace(name="velocity_solid", field_names=["vs_neg_x", "vs_neg_y"], dim=1, side="-")
    displacement_space = FunctionSpace(name="displacement", field_names=["d_neg_x", "d_neg_y"], dim=1, side="-")

    du_f = VectorTrialFunction(space=velocity_fluid_space, dof_handler=dh)
    dp_f = TrialFunction(name="trial_pressure_fluid", field_name="p_pos_", dof_handler=dh, side="+")
    du_s = VectorTrialFunction(space=velocity_solid_space, dof_handler=dh)
    ddisp_s = VectorTrialFunction(space=displacement_space, dof_handler=dh)
    test_vel_f = VectorTestFunction(space=velocity_fluid_space, dof_handler=dh)
    test_q_f = TestFunction(name="test_pressure_fluid", field_name="p_pos_", dof_handler=dh, side="+")
    test_vel_s = VectorTestFunction(space=velocity_solid_space, dof_handler=dh)
    test_disp_s = VectorTestFunction(space=displacement_space, dof_handler=dh)

    uf_k = VectorFunction(name="u_f_k", field_names=["u_pos_x", "u_pos_y"], dof_handler=dh, side="+")
    pf_k = Function(name="p_f_k", field_name="p_pos_", dof_handler=dh, side="+")
    uf_n = VectorFunction(name="u_f_n", field_names=["u_pos_x", "u_pos_y"], dof_handler=dh, side="+")
    pf_n = Function(name="p_f_n", field_name="p_pos_", dof_handler=dh, side="+")
    us_k = VectorFunction(name="u_s_k", field_names=["vs_neg_x", "vs_neg_y"], dof_handler=dh, side="-")
    us_n = VectorFunction(name="u_s_n", field_names=["vs_neg_x", "vs_neg_y"], dof_handler=dh, side="-")
    disp_k = VectorFunction(name="disp_k", field_names=["d_neg_x", "d_neg_y"], dof_handler=dh, side="-")
    disp_n = VectorFunction(name="disp_n", field_names=["d_neg_x", "d_neg_y"], dof_handler=dh, side="-")

    for func in (uf_k, pf_k, uf_n, pf_n, us_k, us_n, disp_k, disp_n):
        func.nodal_values.fill(0.0)

    du_f_R = restrict(du_f, domains["has_pos"])
    dp_f_R = restrict(dp_f, domains["has_pos"])
    test_vel_f_R = restrict(test_vel_f, domains["has_pos"])
    test_q_f_R = restrict(test_q_f, domains["has_pos"])
    uf_k_R = restrict(uf_k, domains["has_pos"])
    pf_k_R = restrict(pf_k, domains["has_pos"])
    uf_n_R = restrict(uf_n, domains["has_pos"])
    pf_n_R = restrict(pf_n, domains["has_pos"])

    du_s_R = restrict(du_s, domains["has_neg"])
    ddisp_s_R = restrict(ddisp_s, domains["has_neg"])
    test_vel_s_R = restrict(test_vel_s, domains["has_neg"])
    test_disp_s_R = restrict(test_disp_s, domains["has_neg"])
    us_k_R = restrict(us_k, domains["has_neg"])
    us_n_R = restrict(us_n, domains["has_neg"])
    disp_k_R = restrict(disp_k, domains["has_neg"])
    disp_n_R = restrict(disp_n, domains["has_neg"])

    kappa_pos = Constant(0.5)
    kappa_neg = Constant(0.5)
    forms = build_fsi_eulerian_forms(
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
        cell_h=CellDiameter(),
        beta_N=Constant(0.0),
        rho_f=Constant(1.0),
        rho_s=Constant(1.0),
        mu_f=Constant(1.0),
        mu_s=Constant(1.0),
        lambda_s=Constant(0.0),
        dt=Constant(1.0),
        theta=Constant(1.0),
        gamma_v=Constant(0.0),
        gamma_p=Constant(0.0),
        gamma_v_grad=Constant(0.0),
        solid_reg_eps=Constant(0.0),
        use_linear_solid=True,
        solid_advect_lagged=True,
        s_nitsche_value=1.0,
    )

    assert forms.J_int_sym_fluid is not None
    K_sym, _ = assemble_form(Equation(forms.J_int_sym_fluid, None), dof_handler=dh, bcs=[], backend="python")

    n = FacetNormal()
    jump_vel_trial = Pos(du_f_R) - Neg(du_s_R)
    expected_form = (-kappa_pos * dot(Pos(test_q_f_R) * n, jump_vel_trial)) * dGamma
    K_expected, _ = assemble_form(Equation(expected_form, None), dof_handler=dh, bcs=[], backend="python")

    p_rows = np.asarray(dh.get_field_slice("p_pos_"), dtype=int)
    u_cols = np.concatenate(
        [
            np.asarray(dh.get_field_slice("u_pos_x"), dtype=int),
            np.asarray(dh.get_field_slice("u_pos_y"), dtype=int),
        ]
    )
    block_sym = K_sym.tocsr()[p_rows[:, None], u_cols].toarray()
    block_expected = K_expected.tocsr()[p_rows[:, None], u_cols].toarray()
    np.testing.assert_allclose(block_sym, block_expected, rtol=1.0e-10, atol=1.0e-12)
