import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
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
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.jit import compile_multi
from pycutfem.jit.kernel_args import _scatter_element_contribs
from examples.utils.fsi.fully_eulerian import (
    build_fsi_eulerian_forms,
    build_measures,
    make_domain_sets,
    refresh_domain_sets,
)
from pycutfem.utils.meshgen import structured_quad


BACKENDS = ("python", "cpp")
JIT_BACKENDS = ("cpp",)


def _tag_rect_boundaries(mesh: Mesh, *, x0: float, x1: float, y0: float, y1: float, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - x0) <= tol,
            "right": lambda x, y: abs(x - x1) <= tol,
            "bottom": lambda x, y: abs(y - y0) <= tol,
            "top": lambda x, y: abs(y - y1) <= tol,
        }
    )


def _build_problem(
    *,
    level_set,
    nx: int = 3,
    ny: int = 2,
    poly_order_u: int = 1,
    poly_order_p: int = 1,
    q: int = 5,
    use_aligned_interface: bool = False,
    use_facet_patch_ghost: bool = False,
):
    # Geometry: rectangle with an internal interface defined by `level_set`
    Lx, Ly = 2.0, 1.0
    x0, x1 = -1.0, 1.0
    y0, y1 = -0.5, 0.5

    nodes, elems, edges, corners = structured_quad(
        Lx,
        Ly,
        nx=int(nx),
        ny=int(ny),
        poly_order=int(poly_order_u),
        offset=(x0, y0),
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=int(poly_order_u),
    )
    _tag_rect_boundaries(mesh, x0=x0, x1=x1, y0=y0, y1=y1)

    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)

    domains = make_domain_sets(mesh, use_aligned_interface=bool(use_aligned_interface))
    dx_fluid, dx_solid, dGamma, dG_fluid, dG_solid = build_measures(
        mesh, level_set, domains, qvol=int(q), use_facet_patch_ghost=bool(use_facet_patch_ghost)
    )

    me = MixedElement(
        mesh,
        field_specs={
            "u_pos_x": int(poly_order_u),
            "u_pos_y": int(poly_order_u),
            "p_pos_": int(poly_order_p),
            "vs_neg_x": int(poly_order_u),
            "vs_neg_y": int(poly_order_u),
            "d_neg_x": int(poly_order_u),
            "d_neg_y": int(poly_order_u),
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

    for f in (uf_k, pf_k, uf_n, pf_n, us_k, us_n, disp_k, disp_n):
        f.nodal_values.fill(0.0)

    # Restrictions to active domains
    du_f_R = restrict(du_f, domains["has_pos"])
    dp_f_R = restrict(dp_f, domains["has_pos"])
    v_f_R = restrict(v_f, domains["has_pos"])
    q_f_R = restrict(q_f, domains["has_pos"])
    uf_k_R = restrict(uf_k, domains["has_pos"])
    uf_n_R = restrict(uf_n, domains["has_pos"])
    pf_k_R = restrict(pf_k, domains["has_pos"])
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
        "level_set": level_set,
        "domains": domains,
        "dx_fluid": dx_fluid,
        "dx_solid": dx_solid,
        "dGamma": dGamma,
        "dG_fluid": dG_fluid,
        "dG_solid": dG_solid,
        "me": me,
        "dh": dh,
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
    }


def _set_field(dh: DofHandler, field: str, values: np.ndarray, *, func) -> None:
    gdofs = np.asarray(dh.get_field_slice(field), dtype=int)
    func.set_nodal_values(gdofs, np.asarray(values, dtype=float))


def _set_affine_vector(
    dh: DofHandler,
    *,
    func: VectorFunction,
    field_x: str,
    field_y: str,
    ax: float,
    bx: float,
    ay: float,
    by: float,
    cx: float = 0.0,
    cy: float = 0.0,
):
    xy = dh.get_dof_coords(field_x)
    vx = ax * xy[:, 0] + bx * xy[:, 1] + cx
    _set_field(dh, field_x, vx, func=func)
    xy = dh.get_dof_coords(field_y)
    vy = ay * xy[:, 0] + by * xy[:, 1] + cy
    _set_field(dh, field_y, vy, func=func)


def _set_affine_scalar(dh: DofHandler, *, func: Function, field: str, a: float, b: float, c: float):
    xy = dh.get_dof_coords(field)
    vals = a * xy[:, 0] + b * xy[:, 1] + c
    _set_field(dh, field, vals, func=func)


def _residual_inf(dh: DofHandler, res_form, bcs, *, backend: str) -> float:
    _, F = assemble_form(Equation(None, res_form), dof_handler=dh, bcs=[], backend=backend)
    dirichlet = dh.get_dirichlet_data(bcs) or {}
    bc_rows = np.fromiter(dirichlet.keys(), dtype=int) if dirichlet else np.zeros((0,), dtype=int)
    inactive = set(getattr(dh, "dof_tags", {}).get("inactive", set()))
    mask = np.ones(dh.total_dofs, dtype=bool)
    if bc_rows.size:
        mask[bc_rows] = False
    if inactive:
        mask[np.fromiter(inactive, dtype=int)] = False
    if not np.any(mask):
        return 0.0
    return float(np.linalg.norm(F[mask], ord=np.inf))


def _manufactured_affine_shear_with_pressure(
    prob,
    *,
    dt_val: float,
    a_vel: float,
    p_lin: tuple[float, float, float],
    solid_advect_lagged: bool,
):
    """
    Manufactured fields:
      u(x,y) = a_vel * [y, x]
      p(x,y) = p0 + px*x + py*y
      u_s = u_f,  (so jump(u)=0)
      d(t1) = dt * u_s(t1),  d(t0)=0

    Forcing is computed in strong form (theta=1), and an interface traction forcing
    g = t_f(u,p) - t_s(d) is added (vector), distributed symmetrically (kappa=0.5).
    """
    dh: DofHandler = prob["dh"]
    level_set = prob["level_set"]

    rho_f = 1.0
    rho_s = 1.0
    # NOTE: `build_fsi_eulerian_forms` uses a grad-grad viscous volume form but a
    # symmetric-gradient traction in the Nitsche flux. To avoid conflating MMS
    # verification with that modeling choice, we set `mu_f=0` here so the test
    # focuses on (i) pressure traction sign and (ii) nonlinear convection/Jacobian.
    mu_f = 0.0
    mu_s = 2.0
    lambda_s = 0.5

    theta = Constant(1.0)
    dt = Constant(float(dt_val))

    # MMS residual checks target the *physics* terms and interface signs; keep the
    # optional ghost/patch stabilizations off so the manufactured solution can be
    # satisfied exactly in the discrete space.
    gamma_v = Constant(0.0)
    gamma_p = Constant(0.0)
    gamma_v_grad = Constant(0.0)
    solid_reg_eps = Constant(0.0)

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
        beta_N=Constant(50.0),
        rho_f=Constant(rho_f),
        rho_s=Constant(rho_s),
        mu_f=Constant(mu_f),
        mu_s=Constant(mu_s),
        lambda_s=Constant(lambda_s),
        dt=dt,
        theta=theta,
        gamma_v=gamma_v,
        gamma_p=gamma_p,
        gamma_v_grad=gamma_v_grad,
        solid_reg_eps=solid_reg_eps,
        use_linear_solid=True,
        solid_advect_lagged=bool(solid_advect_lagged),
        s_nitsche_value=1.0,
    )

    # State at t0: all zero.
    for f in (prob["uf_n"], prob["pf_n"], prob["us_n"], prob["disp_n"]):
        f.nodal_values.fill(0.0)

    # State at t1: affine u, affine p, and displacement d = dt*u (solid).
    _set_affine_vector(
        dh,
        func=prob["uf_k"],
        field_x="u_pos_x",
        field_y="u_pos_y",
        ax=0.0,
        bx=a_vel,
        ay=a_vel,
        by=0.0,
    )
    _set_affine_vector(
        dh,
        func=prob["us_k"],
        field_x="vs_neg_x",
        field_y="vs_neg_y",
        ax=0.0,
        bx=a_vel,
        ay=a_vel,
        by=0.0,
    )
    px, py, p0 = p_lin
    _set_affine_scalar(dh, func=prob["pf_k"], field="p_pos_", a=px, b=py, c=p0)

    # Displacement at t1: d = dt * u_s(t1)
    _set_affine_vector(
        dh,
        func=prob["disp_k"],
        field_x="d_neg_x",
        field_y="d_neg_y",
        ax=0.0,
        bx=float(dt_val) * a_vel,
        ay=float(dt_val) * a_vel,
        by=0.0,
    )

    # Forcing terms ----------------------------------------------------
    # u(x,y) = a_vel * [y, x]  => grad(u) = [[0, a],[a,0]], (grad u) u = [a^2 x, a^2 y]
    def u_exact(x, y):
        return np.stack((a_vel * y, a_vel * x), axis=-1)

    def conv_u(x, y):
        return np.stack((a_vel**2 * x, a_vel**2 * y), axis=-1)

    def grad_p():
        return np.array([px, py], dtype=float)

    def p_exact(x, y):
        return px * x + py * y + p0

    f_f = Analytic(
        lambda x, y: (rho_f / dt_val) * u_exact(x, y) + rho_f * conv_u(x, y) + grad_p(),
        degree=2,
    )

    if solid_advect_lagged:
        f_s = Analytic(lambda x, y: (rho_s / dt_val) * u_exact(x, y), degree=2)
        g_d = Analytic(lambda x, y: np.stack((0.0 * x, 0.0 * x), axis=-1), degree=2)
    else:
        f_s = Analytic(lambda x, y: (rho_s / dt_val) * u_exact(x, y) + rho_s * conv_u(x, y), degree=2)
        g_d = Analytic(lambda x, y: float(dt_val) * conv_u(x, y), degree=2)

    # Interface traction forcing g = t_f - t_s (vector, full normal + shear).
    # n is from solid(-) -> fluid(+): align with level_set.gradient().
    n = np.asarray(level_set.gradient(np.array([0.0, 0.0], dtype=float)), dtype=float).reshape(2,)
    n = n / max(np.linalg.norm(n), 1.0e-16)

    eps_u = np.array([[0.0, a_vel], [a_vel, 0.0]], dtype=float)
    eps_d = float(dt_val) * eps_u
    sigma_s = 2.0 * mu_s * eps_d + lambda_s * float(np.trace(eps_d)) * np.eye(2)
    t_s = sigma_s @ n

    t_f_visc = 2.0 * mu_f * (eps_u @ n)

    def g_ifc(x, y):
        pvals = np.asarray(p_exact(x, y), dtype=float)
        return (t_f_visc - t_s) - pvals[..., None] * n

    g_ifc_fun = Analytic(lambda x, y: g_ifc(x, y), degree=2)

    residual_form = (
        forms.residual_form
        - dot(f_f, prob["v_f_R"]) * prob["dx_fluid"]
        - dot(f_s, prob["v_s_R"]) * prob["dx_solid"]
        - dot(g_d, prob["w_s_R"]) * prob["dx_solid"]
        + Constant(0.5) * dot(g_ifc_fun, prob["v_f_R"]) * prob["dGamma"]
        + Constant(0.5) * dot(g_ifc_fun, prob["v_s_R"]) * prob["dGamma"]
    )

    # Dirichlet BCs at t1.
    def u_bc_x(x, y):
        return float(a_vel * np.asarray(y))

    def u_bc_y(x, y):
        return float(a_vel * np.asarray(x))

    def p_bc(x, y):
        return float(px * np.asarray(x) + py * np.asarray(y) + p0)

    def d_bc_x(x, y):
        return float(dt_val * a_vel * np.asarray(y))

    def d_bc_y(x, y):
        return float(dt_val * a_vel * np.asarray(x))

    bcs = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                BoundaryCondition("u_pos_x", "dirichlet", tag, u_bc_x),
                BoundaryCondition("u_pos_y", "dirichlet", tag, u_bc_y),
                BoundaryCondition("p_pos_", "dirichlet", tag, p_bc),
                BoundaryCondition("vs_neg_x", "dirichlet", tag, u_bc_x),
                BoundaryCondition("vs_neg_y", "dirichlet", tag, u_bc_y),
                BoundaryCondition("d_neg_x", "dirichlet", tag, d_bc_x),
                BoundaryCondition("d_neg_y", "dirichlet", tag, d_bc_y),
            ]
        )

    return residual_form, bcs


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "level_set",
    [
        AffineLevelSet(a=1.0, b=0.0, c=0.27),  # fixed vertical interface (cut)
        AffineLevelSet(a=1.0, b=1.0, c=0.05),  # fixed oblique interface (cut)
    ],
)
def test_fsi_eulerian_mms_pressure_normal_traction_and_convection_residual_zero(backend, level_set):
    """
    MMS (A/B): exercises pressure + normal traction coupling and nonzero fluid convection.
    Uses a full traction-vector forcing g = t_f(u,p) - t_s(d) on Γ.
    """
    prob = _build_problem(level_set=level_set, nx=3, ny=2, q=6, use_facet_patch_ghost=False)
    residual_form, bcs = _manufactured_affine_shear_with_pressure(
        prob,
        dt_val=0.1,
        a_vel=0.7,
        p_lin=(0.3, -0.2, 0.9),  # p = px*x + py*y + p0
        solid_advect_lagged=True,
    )
    res = _residual_inf(prob["dh"], residual_form, bcs, backend=backend)
    assert res < 1.0e-9


@pytest.mark.parametrize("backend", BACKENDS)
def test_fsi_eulerian_mms_solid_advection_term_residual_zero(backend):
    """
    MMS (C): exercises the Eulerian solid advection terms u_s·∇u_s and u_s·∇d,
    plus their coupling with the rest of the FSI residual.
    """
    level_set = AffineLevelSet(a=1.0, b=0.0, c=0.31)
    prob = _build_problem(level_set=level_set, nx=3, ny=2, q=6, use_facet_patch_ghost=False)
    residual_form, bcs = _manufactured_affine_shear_with_pressure(
        prob,
        dt_val=0.1,
        a_vel=0.5,
        p_lin=(0.0, 0.0, 0.0),
        solid_advect_lagged=False,
    )
    res = _residual_inf(prob["dh"], residual_form, bcs, backend=backend)
    assert res < 1.0e-9


@pytest.mark.parametrize("backend", BACKENDS)
def test_fsi_eulerian_mms_aligned_interface_kernel_residual_zero(backend):
    """
    Coverage for the USE_ALIGNED_INTERFACE=1 path: Γ coincides with a mesh edge.

    Uses the pressure/traction MMS so the aligned-interface kernel exercises:
      - normal orientation
      - pressure traction sign (-p n)
      - interface traction forcing g_Γ (full vector)
    """
    level_set = AffineLevelSet(a=1.0, b=0.0, c=0.0)  # Γ: x=0 (aligned for nx even)
    prob = _build_problem(level_set=level_set, nx=4, ny=2, q=6, use_aligned_interface=True, use_facet_patch_ghost=False)
    residual_form, bcs = _manufactured_affine_shear_with_pressure(
        prob,
        dt_val=0.1,
        a_vel=0.7,
        p_lin=(0.3, -0.2, 0.9),
        solid_advect_lagged=True,
    )
    res = _residual_inf(prob["dh"], residual_form, bcs, backend=backend)
    assert res < 1.0e-9


def _assemble_residual_from_kernels(kernels, dh: DofHandler, coeffs: dict) -> np.ndarray:
    ndof = int(dh.total_dofs)
    R = np.zeros(ndof, dtype=float)
    for ker in kernels:
        _Kloc, Floc, _Jloc = ker.exec(coeffs)
        if Floc is None:
            continue
        R_inc = np.zeros_like(R)
        _scatter_element_contribs(
            K_elem=None,
            F_elem=Floc,
            J_elem=None,
            element_ids=np.asarray(ker.static_args.get("eids", []), dtype=np.int32),
            gdofs_map=np.asarray(ker.static_args.get("gdofs_map", []), dtype=np.int32),
            matvec=R_inc,
            ctx={"rhs": True, "add": True},
            integrand=ker,
            hook=None,
        )
        R += R_inc
    return R


@pytest.mark.parametrize("backend", JIT_BACKENDS)
def test_fsi_eulerian_mms_moving_interface_translation_refresh(backend):
    """
    MMS (D): exercise moving-interface refresh without re-JIT compilation.

    We compile kernels once, then move a straight interface so the cut pattern and
    active entity sets change, update domain BitSets in-place, refresh kernel static
    args, and ensure assembly remains consistent (residual stays ~0).
    """
    level_set = AffineLevelSet(a=1.0, b=0.0, c=0.27)
    prob = _build_problem(
        level_set=level_set,
        nx=6,
        ny=4,
        poly_order_u=1,
        poly_order_p=1,
        q=4,
        use_facet_patch_ghost=False,
    )
    dh = prob["dh"]
    mesh = prob["mesh"]
    domains = prob["domains"]

    dt_val = 0.1
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
        beta_N=Constant(10.0),
        rho_f=Constant(1.0),
        rho_s=Constant(1.0),
        mu_f=Constant(1.0),
        mu_s=Constant(1.0),
        lambda_s=Constant(1.0),
        dt=dt,
        theta=theta,
        gamma_v=Constant(1.0e-2),
        gamma_p=Constant(1.0e-2),
        gamma_v_grad=Constant(1.0e-2),
        solid_reg_eps=Constant(0.0),
        use_linear_solid=True,
        solid_advect_lagged=True,
        s_nitsche_value=1.0,
    )

    residual_form = forms.residual_form

    # Constant translation field (exactly satisfies PDE with natural outer BCs).
    u0 = np.array([0.2, -0.1], dtype=float)

    for f in (prob["uf_n"], prob["uf_k"], prob["us_n"], prob["us_k"]):
        f.nodal_values.fill(0.0)
    for f in (prob["pf_n"], prob["pf_k"], prob["disp_n"], prob["disp_k"]):
        f.nodal_values.fill(0.0)

    for vec, fx, fy in (
        (prob["uf_n"], "u_pos_x", "u_pos_y"),
        (prob["uf_k"], "u_pos_x", "u_pos_y"),
        (prob["us_n"], "vs_neg_x", "vs_neg_y"),
        (prob["us_k"], "vs_neg_x", "vs_neg_y"),
    ):
        _set_affine_vector(
            dh,
            func=vec,
            field_x=fx,
            field_y=fy,
            ax=0.0,
            bx=0.0,
            ay=0.0,
            by=0.0,
            cx=float(u0[0]),
            cy=float(u0[1]),
        )

    # Displacement at t0 = 0; at t1 = dt*u.
    _set_affine_vector(
        dh,
        func=prob["disp_n"],
        field_x="d_neg_x",
        field_y="d_neg_y",
        ax=0.0,
        bx=0.0,
        ay=0.0,
        by=0.0,
        cx=0.0,
        cy=0.0,
    )
    _set_affine_vector(
        dh,
        func=prob["disp_k"],
        field_x="d_neg_x",
        field_y="d_neg_y",
        ax=0.0,
        bx=0.0,
        ay=0.0,
        by=0.0,
        cx=float(dt_val * u0[0]),
        cy=float(dt_val * u0[1]),
    )

    coeffs = {
        "u_f_k": prob["uf_k"],
        "p_f_k": prob["pf_k"],
        "u_f_n": prob["uf_n"],
        "p_f_n": prob["pf_n"],
        "u_s_k": prob["us_k"],
        "u_s_n": prob["us_n"],
        "disp_k": prob["disp_k"],
        "disp_n": prob["disp_n"],
        "dt": dt,
    }

    kernels = compile_multi(
        residual_form,
        dof_handler=dh,
        mixed_element=prob["me"],
        quad_order=4,
        backend=backend,
    )

    R0 = _assemble_residual_from_kernels(kernels, dh, coeffs)
    assert float(np.linalg.norm(R0, ord=np.inf)) < 1.0e-10

    # Move the interface enough to change the cut pattern, then refresh.
    level_set.c = -0.11
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)
    refresh_domain_sets(mesh, domains, use_aligned_interface=False)

    for ker in kernels:
        ker.refresh(level_set)

    R1 = _assemble_residual_from_kernels(kernels, dh, coeffs)
    assert float(np.linalg.norm(R1, ord=np.inf)) < 1.0e-10
