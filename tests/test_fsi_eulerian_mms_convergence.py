import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import CellDiameter, Constant, Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction, dot, restrict
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.spaces import FunctionSpace
from examples.utils.fsi.fully_eulerian import build_fsi_eulerian_forms, build_measures, make_domain_sets, retag_inactive
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


def _build_solver_problem(*, nx: int, ny: int, q: int, backend: str):
    # Domain: [-1,1]×[-0.5,0.5] with a cut interface x + c = 0 (c != 0 ensures cut pattern)
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

    level_set = AffineLevelSet(a=1.0, b=0.0, c=0.27)  # Γ: x=-0.27 (cut)
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
    retag_inactive(dh, mesh)

    velocity_fluid = FunctionSpace(name="velocity_fluid", field_names=["u_pos_x", "u_pos_y"], dim=1, side="+")
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

    # Restrictions
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
        "level_set": level_set,
        "domains": domains,
        "dx_fluid": dx_fluid,
        "dx_solid": dx_solid,
        "dGamma": dGamma,
        "dG_fluid": dG_fluid,
        "dG_solid": dG_solid,
        "du_f_R": du_f_R,
        "dp_f_R": dp_f_R,
        "du_s_R": du_s_R,
        "ddisp_s_R": ddisp_s_R,
        "v_f_R": v_f_R,
        "q_f_R": q_f_R,
        "v_s_R": v_s_R,
        "w_s_R": w_s_R,
        "uf_k": uf_k,
        "pf_k": pf_k,
        "uf_n": uf_n,
        "pf_n": pf_n,
        "us_k": us_k,
        "us_n": us_n,
        "disp_k": disp_k,
        "disp_n": disp_n,
        "uf_k_R": uf_k_R,
        "pf_k_R": pf_k_R,
        "uf_n_R": uf_n_R,
        "pf_n_R": pf_n_R,
        "us_k_R": us_k_R,
        "us_n_R": us_n_R,
        "disp_k_R": disp_k_R,
        "disp_n_R": disp_n_R,
        "backend": backend,
        "q": int(q),
    }


def _set_field(dh: DofHandler, field: str, values: np.ndarray, *, func) -> None:
    gdofs = np.asarray(dh.get_field_slice(field), dtype=int)
    func.set_nodal_values(gdofs, np.asarray(values, dtype=float))


def _set_scalar_from_callable(dh: DofHandler, field: str, *, func, value_cb):
    xy = np.asarray(dh.get_dof_coords(field), dtype=float)
    vals = np.asarray(value_cb(xy[:, 0], xy[:, 1]), dtype=float)
    _set_field(dh, field, vals, func=func)


def _solve_one_step_variant_a(*, nx: int, backend: str) -> dict[str, float]:
    prob = _build_solver_problem(nx=nx, ny=max(2, nx // 2), q=6, backend=backend)
    dh: DofHandler = prob["dh"]
    ls = prob["level_set"]

    mu_f = 1.0
    mu_s = 2.0
    alpha = mu_s / mu_f
    A = 0.1
    dt_val = 0.1
    t0, t1 = 0.0, dt_val

    def psi(x):
        return x + x * x

    def psi_dd(_x):
        return 2.0 + 0.0 * _x

    def w_exact(x, t):
        return A * np.exp(alpha * t) * psi(x)

    def u_exact_y(x, t):
        return alpha * w_exact(x, t)

    def u_exact_y_xx(x, t):
        return alpha * A * np.exp(alpha * t) * psi_dd(x)

    def w_exact_xx(x, t):
        return A * np.exp(alpha * t) * psi_dd(x)

    # BE forcing at t1 (consistent with θ=1)
    def f_f_y(x):
        return (u_exact_y(x, t1) - u_exact_y(x, t0)) / dt_val - mu_f * u_exact_y_xx(x, t1)

    def f_s_y(x):
        return (u_exact_y(x, t1) - u_exact_y(x, t0)) / dt_val - mu_s * w_exact_xx(x, t1)

    def g_disp_y(x):
        return (w_exact(x, t1) - w_exact(x, t0)) / dt_val - u_exact_y(x, t1)

    f_f = Analytic(lambda x, y: np.stack((0.0 * x, f_f_y(x)), axis=-1), degree=4)
    f_s = Analytic(lambda x, y: np.stack((0.0 * x, f_s_y(x)), axis=-1), degree=4)
    g_d = Analytic(lambda x, y: np.stack((0.0 * x, g_disp_y(x)), axis=-1), degree=4)

    dt = Constant(dt_val)
    dt._jit_name = "dt"
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
        beta_N=Constant(20.0),
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

    # Dirichlet BCs at t1
    bcs = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                ("u_pos_x", tag, lambda x, y: 0.0),
                ("u_pos_y", tag, lambda x, y: float(u_exact_y(np.asarray(x), t1))),
                ("p_pos_", tag, lambda x, y: 0.0),
                ("vs_neg_x", tag, lambda x, y: 0.0),
                ("vs_neg_y", tag, lambda x, y: float(u_exact_y(np.asarray(x), t1))),
                ("d_neg_x", tag, lambda x, y: 0.0),
                ("d_neg_y", tag, lambda x, y: float(w_exact(np.asarray(x), t1))),
            ]
        )
    bcs = [BoundaryCondition(f, "dirichlet", tag, cb) for (f, tag, cb) in bcs]
    bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]

    # Initial conditions at t0
    _set_scalar_from_callable(dh, "u_pos_x", func=prob["uf_n"].components[0], value_cb=lambda x, y: 0.0 * x)
    _set_scalar_from_callable(dh, "u_pos_y", func=prob["uf_n"].components[1], value_cb=lambda x, y: u_exact_y(x, t0))
    prob["uf_k"].nodal_values[:] = prob["uf_n"].nodal_values

    _set_scalar_from_callable(dh, "vs_neg_x", func=prob["us_n"].components[0], value_cb=lambda x, y: 0.0 * x)
    _set_scalar_from_callable(dh, "vs_neg_y", func=prob["us_n"].components[1], value_cb=lambda x, y: u_exact_y(x, t0))
    prob["us_k"].nodal_values[:] = prob["us_n"].nodal_values

    _set_scalar_from_callable(dh, "d_neg_x", func=prob["disp_n"].components[0], value_cb=lambda x, y: 0.0 * x)
    _set_scalar_from_callable(dh, "d_neg_y", func=prob["disp_n"].components[1], value_cb=lambda x, y: w_exact(x, t0))
    prob["disp_k"].nodal_values[:] = prob["disp_n"].nodal_values

    prob["pf_n"].nodal_values.fill(0.0)
    prob["pf_k"].nodal_values.fill(0.0)

    solver = NewtonSolver(
        residual_form,
        forms.jacobian_form,
        dof_handler=dh,
        mixed_element=prob["me"],
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=1.0e-10, max_newton_iter=25, line_search=False),
        quad_order=prob["q"],
        backend=backend,
    )

    solver.solve_time_interval(
        functions=[prob["uf_k"], prob["pf_k"], prob["us_k"], prob["disp_k"]],
        prev_functions=[prob["uf_n"], prob["pf_n"], prob["us_n"], prob["disp_n"]],
        aux_functions={"dt": dt},
        time_params=TimeStepperParameters(dt=dt_val, max_steps=1, final_time=dt_val, theta=1.0),
    )

    err_u = dh.l2_error_on_side_compiled(
        functions=prob["uf_k"],
        exact={"u_pos_x": lambda x, y: 0.0 * np.asarray(x), "u_pos_y": lambda x, y: u_exact_y(np.asarray(x), t1)},
        level_set=ls,
        side="+",
        fields=["u_pos_x", "u_pos_y"],
        backend=backend,
    )
    err_us = dh.l2_error_on_side_compiled(
        functions=prob["us_k"],
        exact={"vs_neg_x": lambda x, y: 0.0 * np.asarray(x), "vs_neg_y": lambda x, y: u_exact_y(np.asarray(x), t1)},
        level_set=ls,
        side="-",
        fields=["vs_neg_x", "vs_neg_y"],
        backend=backend,
    )
    err_d = dh.l2_error_on_side_compiled(
        functions=prob["disp_k"],
        exact={"d_neg_x": lambda x, y: 0.0 * np.asarray(x), "d_neg_y": lambda x, y: w_exact(np.asarray(x), t1)},
        level_set=ls,
        side="-",
        fields=["d_neg_x", "d_neg_y"],
        backend=backend,
    )
    return {"u_f": float(err_u), "u_s": float(err_us), "d_s": float(err_d)}


def _solve_one_step_variant_b(*, nx: int, backend: str) -> dict[str, float]:
    prob = _build_solver_problem(nx=nx, ny=max(2, nx // 2), q=6, backend=backend)
    dh: DofHandler = prob["dh"]
    ls = prob["level_set"]

    mu_f = 1.0
    mu_s = 2.0
    A = 0.1
    omega = 1.7
    dt_val = 0.1
    t0, t1 = 0.0, dt_val

    def psi(x):
        return x + x * x

    def psi_d(x):
        return 1.0 + 2.0 * x

    def psi_dd(_x):
        return 2.0 + 0.0 * _x

    def w_exact(x, t):
        return A * np.sin(omega * t) * psi(x)

    def u_exact_y(x, t):
        return A * omega * np.cos(omega * t) * psi(x)

    def u_exact_y_x(x, t):
        return A * omega * np.cos(omega * t) * psi_d(x)

    def u_exact_y_xx(x, t):
        return A * omega * np.cos(omega * t) * psi_dd(x)

    def w_exact_x(x, t):
        return A * np.sin(omega * t) * psi_d(x)

    def w_exact_xx(x, t):
        return A * np.sin(omega * t) * psi_dd(x)

    # BE forcing at t1 (θ=1)
    def f_f_y(x):
        return (u_exact_y(x, t1) - u_exact_y(x, t0)) / dt_val - mu_f * u_exact_y_xx(x, t1)

    def f_s_y(x):
        return (u_exact_y(x, t1) - u_exact_y(x, t0)) / dt_val - mu_s * w_exact_xx(x, t1)

    def g_disp_y(x):
        return (w_exact(x, t1) - w_exact(x, t0)) / dt_val - u_exact_y(x, t1)

    def g_ifc_y(x):
        return mu_f * u_exact_y_x(x, t1) - mu_s * w_exact_x(x, t1)

    f_f = Analytic(lambda x, y: np.stack((0.0 * x, f_f_y(x)), axis=-1), degree=4)
    f_s = Analytic(lambda x, y: np.stack((0.0 * x, f_s_y(x)), axis=-1), degree=4)
    g_d = Analytic(lambda x, y: np.stack((0.0 * x, g_disp_y(x)), axis=-1), degree=4)
    g_ifc = Analytic(lambda x, y: np.stack((0.0 * x, g_ifc_y(x)), axis=-1), degree=4)

    dt = Constant(dt_val)
    dt._jit_name = "dt"
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
        beta_N=Constant(20.0),
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

    # Dirichlet BCs at t1
    bcs = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                ("u_pos_x", tag, lambda x, y: 0.0),
                ("u_pos_y", tag, lambda x, y: float(u_exact_y(np.asarray(x), t1))),
                ("p_pos_", tag, lambda x, y: 0.0),
                ("vs_neg_x", tag, lambda x, y: 0.0),
                ("vs_neg_y", tag, lambda x, y: float(u_exact_y(np.asarray(x), t1))),
                ("d_neg_x", tag, lambda x, y: 0.0),
                ("d_neg_y", tag, lambda x, y: float(w_exact(np.asarray(x), t1))),
            ]
        )
    bcs = [BoundaryCondition(f, "dirichlet", tag, cb) for (f, tag, cb) in bcs]
    bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]

    # Initial conditions at t0
    _set_scalar_from_callable(dh, "u_pos_x", func=prob["uf_n"].components[0], value_cb=lambda x, y: 0.0 * x)
    _set_scalar_from_callable(dh, "u_pos_y", func=prob["uf_n"].components[1], value_cb=lambda x, y: u_exact_y(x, t0))
    prob["uf_k"].nodal_values[:] = prob["uf_n"].nodal_values

    _set_scalar_from_callable(dh, "vs_neg_x", func=prob["us_n"].components[0], value_cb=lambda x, y: 0.0 * x)
    _set_scalar_from_callable(dh, "vs_neg_y", func=prob["us_n"].components[1], value_cb=lambda x, y: u_exact_y(x, t0))
    prob["us_k"].nodal_values[:] = prob["us_n"].nodal_values

    _set_scalar_from_callable(dh, "d_neg_x", func=prob["disp_n"].components[0], value_cb=lambda x, y: 0.0 * x)
    _set_scalar_from_callable(dh, "d_neg_y", func=prob["disp_n"].components[1], value_cb=lambda x, y: w_exact(x, t0))
    prob["disp_k"].nodal_values[:] = prob["disp_n"].nodal_values

    prob["pf_n"].nodal_values.fill(0.0)
    prob["pf_k"].nodal_values.fill(0.0)

    solver = NewtonSolver(
        residual_form,
        forms.jacobian_form,
        dof_handler=dh,
        mixed_element=prob["me"],
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=1.0e-10, max_newton_iter=25, line_search=False),
        quad_order=prob["q"],
        backend=backend,
    )

    solver.solve_time_interval(
        functions=[prob["uf_k"], prob["pf_k"], prob["us_k"], prob["disp_k"]],
        prev_functions=[prob["uf_n"], prob["pf_n"], prob["us_n"], prob["disp_n"]],
        aux_functions={"dt": dt},
        time_params=TimeStepperParameters(dt=dt_val, max_steps=1, final_time=dt_val, theta=1.0),
    )

    err_u = dh.l2_error_on_side_compiled(
        functions=prob["uf_k"],
        exact={"u_pos_x": lambda x, y: 0.0 * np.asarray(x), "u_pos_y": lambda x, y: u_exact_y(np.asarray(x), t1)},
        level_set=ls,
        side="+",
        fields=["u_pos_x", "u_pos_y"],
        backend=backend,
    )
    err_us = dh.l2_error_on_side_compiled(
        functions=prob["us_k"],
        exact={"vs_neg_x": lambda x, y: 0.0 * np.asarray(x), "vs_neg_y": lambda x, y: u_exact_y(np.asarray(x), t1)},
        level_set=ls,
        side="-",
        fields=["vs_neg_x", "vs_neg_y"],
        backend=backend,
    )
    err_d = dh.l2_error_on_side_compiled(
        functions=prob["disp_k"],
        exact={"d_neg_x": lambda x, y: 0.0 * np.asarray(x), "d_neg_y": lambda x, y: w_exact(np.asarray(x), t1)},
        level_set=ls,
        side="-",
        fields=["d_neg_x", "d_neg_y"],
        backend=backend,
    )
    return {"u_f": float(err_u), "u_s": float(err_us), "d_s": float(err_d)}


def _assert_converges(errors_coarse: dict[str, float], errors_fine: dict[str, float]) -> None:
    for key in ("u_f", "u_s", "d_s"):
        ec = float(errors_coarse[key])
        ef = float(errors_fine[key])
        assert np.isfinite(ec) and np.isfinite(ef)
        # Avoid degenerate passes where the coarse error is essentially zero.
        assert ec > 1.0e-12
        assert ef < 0.9 * ec


def test_fsi_eulerian_mms_convergence_variant_a_cut_interface_cpp_backend():
    e0 = _solve_one_step_variant_a(nx=6, backend="cpp")
    e1 = _solve_one_step_variant_a(nx=12, backend="cpp")
    _assert_converges(e0, e1)


def test_fsi_eulerian_mms_convergence_variant_b_cut_interface_cpp_backend():
    e0 = _solve_one_step_variant_b(nx=6, backend="cpp")
    e1 = _solve_one_step_variant_b(nx=12, backend="cpp")
    _assert_converges(e0, e1)
