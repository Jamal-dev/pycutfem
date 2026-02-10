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
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.measures import dx
from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms
from pycutfem.utils.meshgen import structured_quad


def test_constant_pressure_does_not_create_spurious_bulk_force_for_variable_capacity():
    """
    Regression for the momentum-pressure coupling:
      R_v^p = -∫ p div(C w) dx
    should not create an interior body-force term when p is constant.

    With the old non-adjoint coupling -∫ (C p) div(w) dx, constant p produced a
    spurious p*grad(C) force. This test probes interior velocity DOFs only.
    """
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=2)
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

    # Isolate pressure coupling in momentum:
    # v=u=0, constant p, variable C(alpha,phi), all other terms inactive.
    for vf in (v_k, v_n, u_k, u_n):
        vf.nodal_values[:] = 0.0
    p_k.set_values_from_function(lambda x, y: 1.0)
    p_n.set_values_from_function(lambda x, y: 1.0)
    phi_k.set_values_from_function(lambda x, y: 0.55 + 0.20 * x)
    phi_n.set_values_from_function(lambda x, y: 0.55 + 0.20 * x)
    alpha_k.set_values_from_function(lambda x, y: 0.45 + 0.20 * y)
    alpha_n.set_values_from_function(lambda x, y: 0.45 + 0.20 * y)
    S_k.set_values_from_function(lambda x, y: 0.0)
    S_n.set_values_from_function(lambda x, y: 0.0)

    forms = build_biofilm_one_domain_forms(
        v_k=v_k,
        p_k=p_k,
        u_k=u_k,
        phi_k=phi_k,
        alpha_k=alpha_k,
        S_k=S_k,
        v_n=v_n,
        p_n=p_n,
        u_n=u_n,
        phi_n=phi_n,
        alpha_n=alpha_n,
        S_n=S_n,
        dv=dv,
        dp=dp,
        du=du,
        dphi=dphi,
        dalpha=dalpha,
        dS=dS,
        v_test=v_test,
        q_test=q_test,
        u_test=u_test,
        phi_test=phi_test,
        alpha_test=alpha_test,
        S_test=S_test,
        dx=dx(metadata={"q": 5}),
        dt=Constant(1.0),
        theta=1.0,
        rho_f=Constant(1.0),
        mu_f=Constant(0.0),
        kappa_inv=Constant(1.0),
        mu_s=Constant(1.0),
        lambda_s=Constant(1.0),
        D_phi=0.0,
        gamma_phi=0.0,
        D_alpha=0.0,
        D_S=0.0,
        mu_max=0.0,
        K_S=1.0,
        k_g=0.0,
        k_d=0.0,
        Y=1.0,
        k_det=0.0,
    )

    _, R = assemble_form(Equation(None, forms.r_momentum), dof_handler=dh, bcs=[], quad_order=5, backend="python")
    R = np.asarray(R, dtype=float)

    tol = 1.0e-12
    for field in ("v_x", "v_y"):
        sl = np.asarray(dh.get_field_slice(field), dtype=int)
        xy = np.asarray(dh.get_field_dof_coords(field), dtype=float)
        interior_mask = (xy[:, 0] > tol) & (xy[:, 0] < 1.0 - tol) & (xy[:, 1] > tol) & (xy[:, 1] < 1.0 - tol)
        interior_gdofs = sl[interior_mask]
        assert interior_gdofs.size > 0
        err = float(np.linalg.norm(R[interior_gdofs], ord=np.inf))
        assert err < 1.0e-10
