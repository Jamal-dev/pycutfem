import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant, Function, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction
from pycutfem.ufl.expressions import TestFunction as UflTestFunction
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad

from examples.utils.biofilm.deformation_only import build_deformation_only_forms


def _build_system(*, gamma_div: float):
    qdeg = 1
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=2)
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
            "vS_x": 2,
            "vS_y": 2,
            "u_x": 2,
            "u_y": 2,
            "alpha": 1,
            "mu_alpha": 1,
        },
    )
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
    VS = FunctionSpace("VS", ["vS_x", "vS_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    dv = VectorTrialFunction(space=V, dof_handler=dh)
    dvS = VectorTrialFunction(space=VS, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    dp = TrialFunction("p", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    dmu = TrialFunction("mu_alpha", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    vS_test = VectorTestFunction(space=VS, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = UflTestFunction("p", dof_handler=dh)
    alpha_test = UflTestFunction("alpha", dof_handler=dh)
    mu_test = UflTestFunction("mu_alpha", dof_handler=dh)

    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    vS_k = VectorFunction("vS_k", ["vS_x", "vS_y"], dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    mu_k = Function("mu_k", "mu_alpha", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    vS_n = VectorFunction("vS_n", ["vS_x", "vS_y"], dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    mu_n = Function("mu_n", "mu_alpha", dof_handler=dh)

    rng = np.random.default_rng(7)
    for vf in (v_k, vS_k, u_k, v_n, vS_n, u_n):
        vf.nodal_values[:] = 1.0e-2 * rng.standard_normal(vf.nodal_values.shape)
    for sf in (p_k, p_n, mu_k, mu_n):
        sf.nodal_values[:] = 1.0e-2 * rng.standard_normal(sf.nodal_values.shape)

    alpha_k.nodal_values[:] = 0.55
    alpha_n.nodal_values[:] = 0.45

    forms = build_deformation_only_forms(
        v_k=v_k,
        p_k=p_k,
        vS_k=vS_k,
        u_k=u_k,
        alpha_k=alpha_k,
        mu_alpha_k=mu_k,
        v_n=v_n,
        p_n=p_n,
        vS_n=vS_n,
        u_n=u_n,
        alpha_n=alpha_n,
        mu_alpha_n=mu_n,
        dv=dv,
        dp=dp,
        dvS=dvS,
        du=du,
        dalpha=dalpha,
        dmu_alpha=dmu,
        v_test=v_test,
        q_test=q_test,
        vS_test=vS_test,
        u_test=u_test,
        alpha_test=alpha_test,
        mu_alpha_test=mu_test,
        dx=dx(metadata={"q": int(qdeg)}),
        dt=Constant(0.1),
        theta=1.0,
        rho_f=Constant(1.0),
        mu_f=Constant(1.0e-2),
        mu_b=Constant(1.0e-2),
        kappa_inv=Constant(10.0),
        mu_s=Constant(1.0),
        lambda_s=Constant(1.0),
        solid_visco_eta=0.0,
        gamma_div=float(gamma_div),
        phi_b=0.47,
        M_alpha=1.0,
        gamma_alpha=1.0,
        eps_alpha=0.1,
    )

    eq = Equation(forms.jacobian_form, forms.residual_form)
    K, R = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=int(qdeg), backend="python")
    return np.asarray(K.todense(), dtype=float), np.asarray(R, dtype=float)


def test_deformation_only_gamma_div_changes_system_for_nonzero_mixture_divergence() -> None:
    K0, R0 = _build_system(gamma_div=0.0)
    K1, R1 = _build_system(gamma_div=0.25)

    assert np.linalg.norm(K1 - K0, ord=np.inf) > 1.0e-12
    assert np.linalg.norm(R1 - R0, ord=np.inf) > 1.0e-12
