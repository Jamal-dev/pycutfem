import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
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
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.utils.fsi_fully_eulerian import build_fsi_eulerian_forms, build_measures, make_domain_sets
from pycutfem.utils.meshgen import structured_quad


def _set_affine_field(component, coords: np.ndarray, *, ax: float = 0.0, ay: float = 0.0, b: float = 0.0) -> None:
    values = ax * coords[:, 0] + ay * coords[:, 1] + b
    dh = component._dof_handler
    gdofs = np.asarray(dh.get_field_slice(component.field_name), dtype=int)
    component.set_nodal_values(gdofs, np.asarray(values, dtype=float))


def test_interface_consistency_cancels_ibp_boundary_terms_for_equal_tractions():
    """
    Mathematical sign test for the *consistency* part of the FSI Nitsche coupling.

    Conventions (matching the implementation):
      - single interface normal `n` is oriented from solid (-) to fluid (+)
      - define tractions with that same `n`:  t_f = σ_f n,  t_s = σ_s n

    Integration by parts on Ω_f and Ω_s yields the interface boundary contribution
      B_Γ(v_f, v_s) = ∫_Γ t_f·v_f ds  - ∫_Γ t_s·v_s ds

    Under the exact dynamic interface condition t_f = t_s = t, this becomes
      B_Γ = ∫_Γ t·(v_f - v_s) ds = ∫_Γ t·jump(v) ds.

    A *consistent* Nitsche flux term must cancel this for all discontinuous test pairs:
      N_Γ(u;v) = -∫_Γ {t(u)}·jump(v) ds,
      {t} = κ⁺ t_f + κ⁻ t_s,  κ⁺+κ⁻=1  ⇒  {t}=t when t_f=t_s.

    This test constructs a manufactured state with constant nonzero traction and checks:
      - B_Γ + (code's interface consistency term) == 0
      - a "difference-average" (κ⁺ t_f - κ⁻ t_s) does *not* cancel B_Γ
    """

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

    # Flat interface Γ: x = 0 with Ω⁻={x<0} (solid) and Ω⁺={x>0} (fluid)
    level_set = AffineLevelSet(a=1.0, b=0.0, c=0.0)
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)

    domains = make_domain_sets(mesh, use_aligned_interface=False)
    dx_fluid, dx_solid, dGamma, dG_fluid, dG_solid = build_measures(mesh, level_set, domains, qvol=5)

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

    # Manufactured constant nonzero traction:
    #   fluid: u_f=0, p=p0  => t_f = -p0 n
    #   solid: choose linear-elastic σ_s = -p0 I  => t_s = -p0 n
    p0 = 1.234
    pf_k.nodal_values.fill(p0)
    pf_n.nodal_values.fill(p0)

    # With mu_s=0, lambda_s=1: d(x,y)=(-p0*x, 0) yields σ_s = trace(ε(d)) I = (-p0) I.
    _set_affine_field(disp_k.components[0], dh.get_dof_coords("d_neg_x"), ax=-p0, ay=0.0, b=0.0)
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

    kappa_pos = Constant(0.5)
    kappa_neg = Constant(0.5)

    # Build *only* the consistency part of the interface terms (no penalty, no symmetric adjoint term).
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
        mu_s=Constant(0.0),
        lambda_s=Constant(1.0),
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

    # B_Γ(v_f, v_s) = ∫ t_f·v_f - ∫ t_s·v_s
    t_f = -Pos(pf_k_R) * n
    eps_s = _epsilon(Neg(disp_k_R))
    sigma_s = Constant(0.0) * eps_s + Constant(1.0) * trace(eps_s) * I2
    t_s = dot(sigma_s, n)
    B = (dot(t_f, Pos(test_vel_f_R)) - dot(t_s, Neg(test_vel_s_R))) * dGamma

    # Code's interface term should be the *negative* of B on the exact interface conditions.
    W = B + forms.R_int
    _, residual = assemble_form(Equation(None, W), dof_handler=dh, bcs=[], backend="python")

    fluid_vel_dofs = np.concatenate([dh.get_field_slice("u_pos_x"), dh.get_field_slice("u_pos_y")])
    solid_vel_dofs = np.concatenate([dh.get_field_slice("vs_neg_x"), dh.get_field_slice("vs_neg_y")])
    assert np.linalg.norm(residual[fluid_vel_dofs], ord=np.inf) < 1.0e-10
    assert np.linalg.norm(residual[solid_vel_dofs], ord=np.inf) < 1.0e-10

    # A "difference-average" does *not* cancel B (diagnostic).
    jump_v = Pos(test_vel_f_R) - Neg(test_vel_s_R)
    avg_wrong = kappa_pos * t_f - kappa_neg * t_s
    N_wrong = (-dot(avg_wrong, jump_v)) * dGamma
    W_wrong = B + N_wrong
    _, residual_wrong = assemble_form(Equation(None, W_wrong), dof_handler=dh, bcs=[], backend="python")
    assert np.linalg.norm(residual_wrong[fluid_vel_dofs], ord=np.inf) > 1.0e-6

