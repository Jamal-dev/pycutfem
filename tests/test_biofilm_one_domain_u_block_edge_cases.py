import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction as UFLTestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
)
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import ds, dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad

from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms


def _build_one_domain_problem(*, nx: int = 2, ny: int = 2, q: int = 5):
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=int(nx), ny=int(ny), poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )

    me = MixedElement(
        mesh,
        field_specs={
            "v_x": 2,
            "v_y": 2,
            "p": 1,
            "u_x": 2,
            "u_y": 2,
            "phi": 1,
            "alpha": 1,
            "S": 1,
        },
    )
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    dv = VectorTrialFunction(space=V, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    dp = TrialFunction("p", dof_handler=dh)
    dphi = TrialFunction("phi", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    dS = TrialFunction("S", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = UFLTestFunction("p", dof_handler=dh)
    phi_test = UFLTestFunction("phi", dof_handler=dh)
    alpha_test = UFLTestFunction("alpha", dof_handler=dh)
    S_test = UFLTestFunction("S", dof_handler=dh)

    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    S_k = Function("S_k", "S", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    S_n = Function("S_n", "S", dof_handler=dh)

    return {
        "mesh": mesh,
        "me": me,
        "dh": dh,
        "q": int(q),
        "dv": dv,
        "dp": dp,
        "du": du,
        "dphi": dphi,
        "dalpha": dalpha,
        "dS": dS,
        "v_test": v_test,
        "q_test": q_test,
        "u_test": u_test,
        "phi_test": phi_test,
        "alpha_test": alpha_test,
        "S_test": S_test,
        "v_k": v_k,
        "p_k": p_k,
        "u_k": u_k,
        "phi_k": phi_k,
        "alpha_k": alpha_k,
        "S_k": S_k,
        "v_n": v_n,
        "p_n": p_n,
        "u_n": u_n,
        "phi_n": phi_n,
        "alpha_n": alpha_n,
        "S_n": S_n,
    }


def _u_dofs(dh: DofHandler) -> np.ndarray:
    ux = np.asarray(dh.get_field_slice("u_x"), dtype=int).ravel()
    uy = np.asarray(dh.get_field_slice("u_y"), dtype=int).ravel()
    return np.unique(np.concatenate([ux, uy]))


def _min_singular_value_dense(A) -> float:
    Ad = np.asarray(A.todense(), dtype=float)
    s = np.linalg.svd(Ad, compute_uv=False)
    return float(np.min(s)) if s.size else 0.0


def test_u_block_alpha_zero_has_l2_extension_coercivity() -> None:
    """
    Edge case T1: alpha ≡ 0 everywhere (no biofilm).

    Expectation: the u-extension term provides direct L2 coercivity for u so the
    u-u Jacobian block is full rank (no rigid-mode nullspace).
    """
    prob = _build_one_domain_problem(nx=2, ny=2, q=5)
    dh: DofHandler = prob["dh"]

    # State: trivial fields.
    prob["v_k"].nodal_values.fill(0.0)
    prob["v_n"].nodal_values.fill(0.0)
    prob["u_k"].nodal_values.fill(0.0)
    prob["u_n"].nodal_values.fill(0.0)
    prob["p_k"].nodal_values.fill(0.0)
    prob["p_n"].nodal_values.fill(0.0)
    prob["phi_k"].nodal_values.fill(1.0)
    prob["phi_n"].nodal_values.fill(1.0)
    prob["alpha_k"].nodal_values.fill(0.0)
    prob["alpha_n"].nodal_values.fill(0.0)
    prob["S_k"].nodal_values.fill(0.2)
    prob["S_n"].nodal_values.fill(0.2)

    forms = build_biofilm_one_domain_forms(
        v_k=prob["v_k"],
        p_k=prob["p_k"],
        u_k=prob["u_k"],
        phi_k=prob["phi_k"],
        alpha_k=prob["alpha_k"],
        S_k=prob["S_k"],
        v_n=prob["v_n"],
        p_n=prob["p_n"],
        u_n=prob["u_n"],
        phi_n=prob["phi_n"],
        alpha_n=prob["alpha_n"],
        S_n=prob["S_n"],
        dv=prob["dv"],
        dp=prob["dp"],
        du=prob["du"],
        dphi=prob["dphi"],
        dalpha=prob["dalpha"],
        dS=prob["dS"],
        v_test=prob["v_test"],
        q_test=prob["q_test"],
        u_test=prob["u_test"],
        phi_test=prob["phi_test"],
        alpha_test=prob["alpha_test"],
        S_test=prob["S_test"],
        dx=dx(metadata={"q": int(prob["q"])}),
        ds_cip=ds(metadata={"q": int(prob["q"])}),
        dt=Constant(0.1),
        theta=1.0,
        rho_f=Constant(1.0),
        mu_f=Constant(1.0e-2),
        kappa_inv=Constant(10.0),
        mu_s=Constant(0.0),
        lambda_s=Constant(0.0),
        D_phi=0.0,
        gamma_phi=0.0,
        D_alpha=0.1,
        gamma_u=2.0,
        u_extension_mode="l2",
        D_S=0.0,
        mu_max=0.0,
        K_S=1.0,
        k_g=0.0,
        k_d=0.0,
        Y=1.0,
        k_det=0.0,
    )

    K, _R = assemble_form(
        Equation(forms.jacobian_form, forms.residual_form),
        dof_handler=dh,
        bcs=[],
        quad_order=int(prob["q"]),
        backend="python",
    )

    u = _u_dofs(dh)
    Kuu = K.tocsr()[u[:, None], u]
    smin = _min_singular_value_dense(Kuu)
    assert smin > 1.0e-12


def test_u_block_inertia_and_drag_without_elasticity_is_full_rank() -> None:
    """
    Edge case T2: no elasticity (mu_s=lambda_s=0), but alpha ≡ 1.

    Expectation: inertia and/or drag provides direct (mass-like) coercivity for u
    through vS=(u^{n+1}-u^n)/dt, so the u-u Jacobian block is full rank.
    """
    prob = _build_one_domain_problem(nx=2, ny=2, q=5)
    dh: DofHandler = prob["dh"]

    prob["v_k"].nodal_values.fill(0.0)
    prob["v_n"].nodal_values.fill(0.0)
    prob["u_k"].nodal_values.fill(0.0)
    prob["u_n"].nodal_values.fill(0.0)
    prob["p_k"].nodal_values.fill(0.0)
    prob["p_n"].nodal_values.fill(0.0)
    prob["phi_k"].nodal_values.fill(0.5)
    prob["phi_n"].nodal_values.fill(0.5)
    prob["alpha_k"].nodal_values.fill(1.0)
    prob["alpha_n"].nodal_values.fill(1.0)
    prob["S_k"].nodal_values.fill(0.2)
    prob["S_n"].nodal_values.fill(0.2)

    forms = build_biofilm_one_domain_forms(
        v_k=prob["v_k"],
        p_k=prob["p_k"],
        u_k=prob["u_k"],
        phi_k=prob["phi_k"],
        alpha_k=prob["alpha_k"],
        S_k=prob["S_k"],
        v_n=prob["v_n"],
        p_n=prob["p_n"],
        u_n=prob["u_n"],
        phi_n=prob["phi_n"],
        alpha_n=prob["alpha_n"],
        S_n=prob["S_n"],
        dv=prob["dv"],
        dp=prob["dp"],
        du=prob["du"],
        dphi=prob["dphi"],
        dalpha=prob["dalpha"],
        dS=prob["dS"],
        v_test=prob["v_test"],
        q_test=prob["q_test"],
        u_test=prob["u_test"],
        phi_test=prob["phi_test"],
        alpha_test=prob["alpha_test"],
        S_test=prob["S_test"],
        dx=dx(metadata={"q": int(prob["q"])}),
        ds_cip=ds(metadata={"q": int(prob["q"])}),
        dt=Constant(0.1),
        theta=1.0,
        rho_f=Constant(1.0),
        mu_f=Constant(1.0e-2),
        kappa_inv=Constant(10.0),
        mu_s=Constant(0.0),
        lambda_s=Constant(0.0),
        D_phi=0.0,
        gamma_phi=0.0,
        D_alpha=0.0,
        gamma_u=0.0,
        include_skeleton_acceleration=True,
        rho_s0_tilde=Constant(1.0),
        D_S=0.0,
        mu_max=0.0,
        K_S=1.0,
        k_g=0.0,
        k_d=0.0,
        Y=1.0,
        k_det=0.0,
    )

    K, _R = assemble_form(
        Equation(forms.jacobian_form, forms.residual_form),
        dof_handler=dh,
        bcs=[],
        quad_order=int(prob["q"]),
        backend="python",
    )

    u = _u_dofs(dh)
    Kuu = K.tocsr()[u[:, None], u]
    smin = _min_singular_value_dense(Kuu)
    assert smin > 1.0e-12


def test_u_block_alpha_zero_grad_extension_with_pin_is_full_rank() -> None:
    """
    Edge case T1b: alpha ≡ 0 everywhere with H1 (grad) extension.

    Expectation: the grad penalty alone has a global-translation nullspace; the
    optional gamma_u_pin removes it so the u-u block is full rank.
    """
    prob = _build_one_domain_problem(nx=2, ny=2, q=5)
    dh: DofHandler = prob["dh"]

    prob["v_k"].nodal_values.fill(0.0)
    prob["v_n"].nodal_values.fill(0.0)
    prob["u_k"].nodal_values.fill(0.0)
    prob["u_n"].nodal_values.fill(0.0)
    prob["p_k"].nodal_values.fill(0.0)
    prob["p_n"].nodal_values.fill(0.0)
    prob["phi_k"].nodal_values.fill(1.0)
    prob["phi_n"].nodal_values.fill(1.0)
    prob["alpha_k"].nodal_values.fill(0.0)
    prob["alpha_n"].nodal_values.fill(0.0)
    prob["S_k"].nodal_values.fill(0.2)
    prob["S_n"].nodal_values.fill(0.2)

    forms = build_biofilm_one_domain_forms(
        v_k=prob["v_k"],
        p_k=prob["p_k"],
        u_k=prob["u_k"],
        phi_k=prob["phi_k"],
        alpha_k=prob["alpha_k"],
        S_k=prob["S_k"],
        v_n=prob["v_n"],
        p_n=prob["p_n"],
        u_n=prob["u_n"],
        phi_n=prob["phi_n"],
        alpha_n=prob["alpha_n"],
        S_n=prob["S_n"],
        dv=prob["dv"],
        dp=prob["dp"],
        du=prob["du"],
        dphi=prob["dphi"],
        dalpha=prob["dalpha"],
        dS=prob["dS"],
        v_test=prob["v_test"],
        q_test=prob["q_test"],
        u_test=prob["u_test"],
        phi_test=prob["phi_test"],
        alpha_test=prob["alpha_test"],
        S_test=prob["S_test"],
        dx=dx(metadata={"q": int(prob["q"])}),
        ds_cip=ds(metadata={"q": int(prob["q"])}),
        dt=Constant(0.1),
        theta=1.0,
        rho_f=Constant(1.0),
        mu_f=Constant(1.0e-2),
        kappa_inv=Constant(10.0),
        mu_s=Constant(0.0),
        lambda_s=Constant(0.0),
        D_phi=0.0,
        gamma_phi=0.0,
        D_alpha=0.1,
        gamma_u=2.0,
        u_extension_mode="grad",
        gamma_u_pin=1.0e-3,
        D_S=0.0,
        mu_max=0.0,
        K_S=1.0,
        k_g=0.0,
        k_d=0.0,
        Y=1.0,
        k_det=0.0,
    )

    K, _R = assemble_form(
        Equation(forms.jacobian_form, forms.residual_form),
        dof_handler=dh,
        bcs=[],
        quad_order=int(prob["q"]),
        backend="python",
    )

    u = _u_dofs(dh)
    Kuu = K.tocsr()[u[:, None], u]
    smin = _min_singular_value_dense(Kuu)
    assert smin > 1.0e-12
