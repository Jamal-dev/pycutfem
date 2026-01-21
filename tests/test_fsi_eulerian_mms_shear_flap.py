import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import CellDiameter, Constant, Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction, dot, restrict
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.utils.fsi_fully_eulerian import build_fsi_eulerian_forms, build_measures, make_domain_sets
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


def _build_mms_problem(*, nx: int = 3, ny: int = 2, q: int = 3):
    # Geometry: rectangle with interface x=0 (solid on x<0, fluid on x>0)
    Lx, Ly = 2.0, 1.0
    x0, x1 = -1.0, 1.0
    y0, y1 = -0.5, 0.5
    poly_order_u = 1
    poly_order_p = 1

    nodes, elems, edges, corners = structured_quad(
        Lx,
        Ly,
        nx=int(nx),
        ny=int(ny),
        poly_order=poly_order_u,
        offset=(x0, y0),
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order_u,
    )
    _tag_rect_boundaries(mesh, x0=x0, x1=x1, y0=y0, y1=y1)

    level_set = AffineLevelSet(a=1.0, b=0.0, c=0.0)  # Γ: x=0
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)

    domains = make_domain_sets(mesh, use_aligned_interface=False)
    dx_fluid, dx_solid, dGamma, dG_fluid, dG_solid = build_measures(mesh, level_set, domains, qvol=int(q))

    me = MixedElement(
        mesh,
        field_specs={
            "u_pos_x": poly_order_u,
            "u_pos_y": poly_order_u,
            "p_pos_": poly_order_p,
            "vs_neg_x": poly_order_u,
            "vs_neg_y": poly_order_u,
            "d_neg_x": poly_order_u,
            "d_neg_y": poly_order_u,
        },
    )
    dh = DofHandler(me, method="cg")

    # Spaces
    velocity_fluid = FunctionSpace(name="velocity_fluid", field_names=["u_pos_x", "u_pos_y"], dim=1, side="+")
    velocity_solid = FunctionSpace(name="velocity_solid", field_names=["vs_neg_x", "vs_neg_y"], dim=1, side="-")
    displacement_solid = FunctionSpace(name="displacement", field_names=["d_neg_x", "d_neg_y"], dim=1, side="-")

    # Unknowns / tests
    du_f = VectorTrialFunction(space=velocity_fluid, dof_handler=dh)
    dp_f = TrialFunction(name="trial_pressure_fluid", field_name="p_pos_", dof_handler=dh, side="+")
    du_s = VectorTrialFunction(space=velocity_solid, dof_handler=dh)
    ddisp_s = VectorTrialFunction(space=displacement_solid, dof_handler=dh)

    v_f = VectorTestFunction(space=velocity_fluid, dof_handler=dh)
    q_f = TestFunction(name="test_pressure_fluid", field_name="p_pos_", dof_handler=dh, side="+")
    v_s = VectorTestFunction(space=velocity_solid, dof_handler=dh)
    w_s = VectorTestFunction(space=displacement_solid, dof_handler=dh)

    # State
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

    # Restrict to active domains
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

    return {
        "mesh": mesh,
        "dh": dh,
        "me": me,
        "domains": domains,
        "level_set": level_set,
        "dx_fluid": dx_fluid,
        "dx_solid": dx_solid,
        "dGamma": dGamma,
        "dG_fluid": dG_fluid,
        "dG_solid": dG_solid,
        "du_f": du_f,
        "dp_f": dp_f,
        "du_s": du_s,
        "ddisp_s": ddisp_s,
        "v_f": v_f,
        "q_f": q_f,
        "v_s": v_s,
        "w_s": w_s,
        "uf_k": uf_k,
        "pf_k": pf_k,
        "uf_n": uf_n,
        "pf_n": pf_n,
        "us_k": us_k,
        "us_n": us_n,
        "disp_k": disp_k,
        "disp_n": disp_n,
        "du_f_R": du_f_R,
        "dp_f_R": dp_f_R,
        "v_f_R": v_f_R,
        "q_f_R": q_f_R,
        "uf_k_R": uf_k_R,
        "pf_k_R": pf_k_R,
        "uf_n_R": uf_n_R,
        "pf_n_R": pf_n_R,
        "du_s_R": du_s_R,
        "ddisp_s_R": ddisp_s_R,
        "v_s_R": v_s_R,
        "w_s_R": w_s_R,
        "us_k_R": us_k_R,
        "us_n_R": us_n_R,
        "disp_k_R": disp_k_R,
        "disp_n_R": disp_n_R,
    }


def _residual_inf_on_free_dofs(dh: DofHandler, res_form, bcs, *, backend: str) -> float:
    """
    Assemble the *raw* residual (no BC elimination) and measure the infinity norm
    on non-Dirichlet DOFs.

    Note: applying non-homogeneous Dirichlet elimination to a Newton system can
    change the RHS; for MMS residual checks we therefore inspect the unmodified
    residual and simply drop boundary DOFs from the norm.
    """
    _, F = assemble_form(Equation(None, res_form), dof_handler=dh, bcs=[], backend=backend)
    dirichlet = dh.get_dirichlet_data(bcs) or {}
    rows = np.fromiter(dirichlet.keys(), dtype=int) if dirichlet else np.zeros((0,), dtype=int)
    free = np.setdiff1d(np.arange(dh.total_dofs, dtype=int), rows)
    if free.size == 0:
        return 0.0
    return float(np.linalg.norm(F[free], ord=np.inf))


@pytest.mark.parametrize("backend", ("python", "cpp"))
def test_fsi_eulerian_mms_variant_a_shear_flap_residual_zero_python_backend(backend):
    """
    MMS Variant A (shear-flap): choose α = μ_s/μ_f to make interface shear tractions match
    without needing any interface forcing.
    """
    prob = _build_mms_problem()
    dh = prob["dh"]
    mesh = prob["mesh"]
    domains = prob["domains"]

    mu_f = 1.0
    mu_s = 2.0
    alpha = mu_s / mu_f
    A = 0.1
    dt_val = 0.05
    t0, t1 = 0.0, dt_val

    # Linear shape: exactly representable in CG1.
    def psi(x):
        return x

    def w_exact(x, t):
        return A * np.exp(alpha * t) * psi(x)

    def u_exact_y(x, t):
        return alpha * w_exact(x, t)

    # Discrete BE forcing (ρ=1); u_xx = 0 for linear psi.
    def f_f_y_discrete(x):
        u0 = u_exact_y(x, t0)
        u1 = u_exact_y(x, t1)
        return (u1 - u0) / dt_val

    def f_s_y_discrete(x):
        u0 = u_exact_y(x, t0)
        u1 = u_exact_y(x, t1)
        return (u1 - u0) / dt_val

    def g_disp_y_discrete(x):
        d0 = w_exact(x, t0)
        d1 = w_exact(x, t1)
        u1 = u_exact_y(x, t1)
        return (d1 - d0) / dt_val - u1

    f_f = Analytic(lambda x, y: np.stack((0.0 * x, f_f_y_discrete(x)), axis=-1), degree=2)
    f_s = Analytic(lambda x, y: np.stack((0.0 * x, f_s_y_discrete(x)), axis=-1), degree=2)
    g_d = Analytic(lambda x, y: np.stack((0.0 * x, g_disp_y_discrete(x)), axis=-1), degree=2)

    dt = Constant(dt_val)
    theta = Constant(1.0)

    forms = build_fsi_eulerian_forms(
        du_f=prob["du_f_R"],
        dp_f=prob["dp_f_R"],
        du_s=prob["du_s_R"],
        ddisp_s=prob["ddisp_s_R"],
        test_vel_f=prob["v_f_R"],
        test_q_f=prob["q_f_R"],
        test_vel_s=prob["v_s_R"],
        test_disp_s=prob["w_s_R"],
        uf_k=prob["uf_k_R"],
        pf_k=prob["pf_k_R"],
        uf_n=prob["uf_n_R"],
        pf_n=prob["pf_n_R"],
        us_k=prob["us_k_R"],
        us_n=prob["us_n_R"],
        disp_k=prob["disp_k_R"],
        disp_n=prob["disp_n_R"],
        dx_fluid=prob["dx_fluid"],
        dx_solid=prob["dx_solid"],
        dGamma=prob["dGamma"],
        dG_fluid=prob["dG_fluid"],
        dG_solid=prob["dG_solid"],
        kappa_pos=Constant(0.5),
        kappa_neg=Constant(0.5),
        cell_h=CellDiameter(),
        beta_N=Constant(0.0),
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
        - dot(f_f, prob["v_f_R"]) * prob["dx_fluid"]
        - dot(f_s, prob["v_s_R"]) * prob["dx_solid"]
        - dot(g_d, prob["w_s_R"]) * prob["dx_solid"]
    )

    # Fill exact state at t0 / t1.
    for comp in (0, 1):
        gd = np.asarray(dh.get_field_slice("u_pos_x" if comp == 0 else "u_pos_y"), dtype=int)
        xy = dh.get_dof_coords("u_pos_x" if comp == 0 else "u_pos_y")
        prob["uf_n"].set_nodal_values(gd, 0.0 * xy[:, 0] if comp == 0 else u_exact_y(xy[:, 0], t0))
        prob["uf_k"].set_nodal_values(gd, 0.0 * xy[:, 0] if comp == 0 else u_exact_y(xy[:, 0], t1))

        gd = np.asarray(dh.get_field_slice("vs_neg_x" if comp == 0 else "vs_neg_y"), dtype=int)
        xy = dh.get_dof_coords("vs_neg_x" if comp == 0 else "vs_neg_y")
        prob["us_n"].set_nodal_values(gd, 0.0 * xy[:, 0] if comp == 0 else u_exact_y(xy[:, 0], t0))
        prob["us_k"].set_nodal_values(gd, 0.0 * xy[:, 0] if comp == 0 else u_exact_y(xy[:, 0], t1))

        gd = np.asarray(dh.get_field_slice("d_neg_x" if comp == 0 else "d_neg_y"), dtype=int)
        xy = dh.get_dof_coords("d_neg_x" if comp == 0 else "d_neg_y")
        prob["disp_n"].set_nodal_values(gd, 0.0 * xy[:, 0] if comp == 0 else w_exact(xy[:, 0], t0))
        prob["disp_k"].set_nodal_values(gd, 0.0 * xy[:, 0] if comp == 0 else w_exact(xy[:, 0], t1))

    prob["pf_n"].nodal_values.fill(0.0)
    prob["pf_k"].nodal_values.fill(0.0)

    # Dirichlet: exact outer boundary at t1.
    def _u_x_bc(x, y):
        return 0.0

    def _u_y_bc(x, y):
        return float(u_exact_y(np.asarray(x), t1))

    def _p_bc(x, y):
        return 0.0

    def _d_x_bc(x, y):
        return 0.0

    def _d_y_bc(x, y):
        return float(w_exact(np.asarray(x), t1))

    bcs = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                BoundaryCondition("u_pos_x", "dirichlet", tag, _u_x_bc),
                BoundaryCondition("u_pos_y", "dirichlet", tag, _u_y_bc),
                BoundaryCondition("p_pos_", "dirichlet", tag, _p_bc),
                BoundaryCondition("vs_neg_x", "dirichlet", tag, _u_x_bc),
                BoundaryCondition("vs_neg_y", "dirichlet", tag, _u_y_bc),
                BoundaryCondition("d_neg_x", "dirichlet", tag, _d_x_bc),
                BoundaryCondition("d_neg_y", "dirichlet", tag, _d_y_bc),
            ]
        )

    res_inf = _residual_inf_on_free_dofs(dh, residual_form, bcs, backend=backend)
    assert res_inf < 1.0e-10


@pytest.mark.parametrize("backend", ("python", "cpp"))
def test_fsi_eulerian_mms_variant_b_sinusoidal_with_interface_forcing_residual_zero_python_backend(backend):
    """
    MMS Variant B (sinusoidal flapping): add interface forcing g = t_f - t_s so
    the manufactured fields satisfy the interface traction balance.
    """
    prob = _build_mms_problem()
    dh = prob["dh"]

    mu_f = 1.0
    mu_s = 2.0
    A = 0.1
    omega = 1.7
    dt_val = 0.05
    t0, t1 = 0.0, dt_val

    def psi(x):
        return x

    def psi_d(x):
        return 1.0 + 0.0 * x

    def w_exact(x, t):
        return A * np.sin(omega * t) * psi(x)

    def w_exact_x(x, t):
        return A * np.sin(omega * t) * psi_d(x)

    def u_exact_y(x, t):
        return A * omega * np.cos(omega * t) * psi(x)

    def u_exact_y_x(x, t):
        return A * omega * np.cos(omega * t) * psi_d(x)

    # Discrete BE forcing (ρ=1); u_xx = w_xx = 0 for linear psi.
    def f_f_y_discrete(x):
        u0 = u_exact_y(x, t0)
        u1 = u_exact_y(x, t1)
        return (u1 - u0) / dt_val

    def f_s_y_discrete(x):
        u0 = u_exact_y(x, t0)
        u1 = u_exact_y(x, t1)
        return (u1 - u0) / dt_val

    def g_disp_y_discrete(x):
        d0 = w_exact(x, t0)
        d1 = w_exact(x, t1)
        u1 = u_exact_y(x, t1)
        return (d1 - d0) / dt_val - u1

    # Interface traction mismatch (same normal n from solid->fluid):
    #   t_f_y = μ_f ∂_x u_y,   t_s_y = μ_s ∂_x d_y  =>  g = t_f - t_s.
    def g_ifc_y(x):
        return mu_f * u_exact_y_x(x, t1) - mu_s * w_exact_x(x, t1)

    f_f = Analytic(lambda x, y: np.stack((0.0 * x, f_f_y_discrete(x)), axis=-1), degree=2)
    f_s = Analytic(lambda x, y: np.stack((0.0 * x, f_s_y_discrete(x)), axis=-1), degree=2)
    g_d = Analytic(lambda x, y: np.stack((0.0 * x, g_disp_y_discrete(x)), axis=-1), degree=2)
    g_ifc = Analytic(lambda x, y: np.stack((0.0 * x, g_ifc_y(x)), axis=-1), degree=2)

    dt = Constant(dt_val)
    theta = Constant(1.0)

    forms = build_fsi_eulerian_forms(
        du_f=prob["du_f_R"],
        dp_f=prob["dp_f_R"],
        du_s=prob["du_s_R"],
        ddisp_s=prob["ddisp_s_R"],
        test_vel_f=prob["v_f_R"],
        test_q_f=prob["q_f_R"],
        test_vel_s=prob["v_s_R"],
        test_disp_s=prob["w_s_R"],
        uf_k=prob["uf_k_R"],
        pf_k=prob["pf_k_R"],
        uf_n=prob["uf_n_R"],
        pf_n=prob["pf_n_R"],
        us_k=prob["us_k_R"],
        us_n=prob["us_n_R"],
        disp_k=prob["disp_k_R"],
        disp_n=prob["disp_n_R"],
        dx_fluid=prob["dx_fluid"],
        dx_solid=prob["dx_solid"],
        dGamma=prob["dGamma"],
        dG_fluid=prob["dG_fluid"],
        dG_solid=prob["dG_solid"],
        kappa_pos=Constant(0.5),
        kappa_neg=Constant(0.5),
        cell_h=CellDiameter(),
        beta_N=Constant(0.0),
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
        - dot(f_f, prob["v_f_R"]) * prob["dx_fluid"]
        - dot(f_s, prob["v_s_R"]) * prob["dx_solid"]
        - dot(g_d, prob["w_s_R"]) * prob["dx_solid"]
        + Constant(0.5) * dot(g_ifc, prob["v_f_R"]) * prob["dGamma"]
        + Constant(0.5) * dot(g_ifc, prob["v_s_R"]) * prob["dGamma"]
    )

    # Fill exact state at t0 / t1.
    for comp in (0, 1):
        gd = np.asarray(dh.get_field_slice("u_pos_x" if comp == 0 else "u_pos_y"), dtype=int)
        xy = dh.get_dof_coords("u_pos_x" if comp == 0 else "u_pos_y")
        prob["uf_n"].set_nodal_values(gd, 0.0 * xy[:, 0] if comp == 0 else u_exact_y(xy[:, 0], t0))
        prob["uf_k"].set_nodal_values(gd, 0.0 * xy[:, 0] if comp == 0 else u_exact_y(xy[:, 0], t1))

        gd = np.asarray(dh.get_field_slice("vs_neg_x" if comp == 0 else "vs_neg_y"), dtype=int)
        xy = dh.get_dof_coords("vs_neg_x" if comp == 0 else "vs_neg_y")
        prob["us_n"].set_nodal_values(gd, 0.0 * xy[:, 0] if comp == 0 else u_exact_y(xy[:, 0], t0))
        prob["us_k"].set_nodal_values(gd, 0.0 * xy[:, 0] if comp == 0 else u_exact_y(xy[:, 0], t1))

        gd = np.asarray(dh.get_field_slice("d_neg_x" if comp == 0 else "d_neg_y"), dtype=int)
        xy = dh.get_dof_coords("d_neg_x" if comp == 0 else "d_neg_y")
        prob["disp_n"].set_nodal_values(gd, 0.0 * xy[:, 0] if comp == 0 else w_exact(xy[:, 0], t0))
        prob["disp_k"].set_nodal_values(gd, 0.0 * xy[:, 0] if comp == 0 else w_exact(xy[:, 0], t1))

    prob["pf_n"].nodal_values.fill(0.0)
    prob["pf_k"].nodal_values.fill(0.0)

    # Dirichlet: exact outer boundary at t1.
    def _u_x_bc(x, y):
        return 0.0

    def _u_y_bc(x, y):
        return float(u_exact_y(np.asarray(x), t1))

    def _p_bc(x, y):
        return 0.0

    def _d_x_bc(x, y):
        return 0.0

    def _d_y_bc(x, y):
        return float(w_exact(np.asarray(x), t1))

    bcs = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                BoundaryCondition("u_pos_x", "dirichlet", tag, _u_x_bc),
                BoundaryCondition("u_pos_y", "dirichlet", tag, _u_y_bc),
                BoundaryCondition("p_pos_", "dirichlet", tag, _p_bc),
                BoundaryCondition("vs_neg_x", "dirichlet", tag, _u_x_bc),
                BoundaryCondition("vs_neg_y", "dirichlet", tag, _u_y_bc),
                BoundaryCondition("d_neg_x", "dirichlet", tag, _d_x_bc),
                BoundaryCondition("d_neg_y", "dirichlet", tag, _d_y_bc),
            ]
        )

    res_inf = _residual_inf_on_free_dofs(dh, residual_form, bcs, backend=backend)
    assert res_inf < 1.0e-10
