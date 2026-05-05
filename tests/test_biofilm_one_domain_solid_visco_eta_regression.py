import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
)
from pycutfem.ufl.expressions import TestFunction as UflTestFunction
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad

from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms


def test_one_domain_solid_viscosity_and_frozen_sv_jacobian_assembles_python_backend() -> None:
    """
    Regression test for two related assembly issues:

    1) `solid_visco_eta != 0` (Kelvin–Voigt skeleton viscosity) previously triggered a
       trial/function VecOpInfo shape mismatch in the Jacobian.
    2) passing a frozen/Picard volumetric-source Jacobian as `ds_v=Constant(0.0)` could
       also trigger the same mismatch via `-alpha_k*ds_v` in the mass block.
    """
    qdeg = 1  # keep this tiny; we only care about compilation/assembly robustness
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
            "phi": 1,
            "alpha": 1,
            "S": 1,
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
    dphi = TrialFunction("phi", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    dS = TrialFunction("S", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    vS_test = VectorTestFunction(space=VS, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = UflTestFunction("p", dof_handler=dh)
    phi_test = UflTestFunction("phi", dof_handler=dh)
    alpha_test = UflTestFunction("alpha", dof_handler=dh)
    S_test = UflTestFunction("S", dof_handler=dh)

    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    vS_k = VectorFunction("vS_k", ["vS_x", "vS_y"], dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    S_k = Function("S_k", "S", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    vS_n = VectorFunction("vS_n", ["vS_x", "vS_y"], dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    S_n = Function("S_n", "S", dof_handler=dh)

    rng = np.random.default_rng(0)
    for vf in (v_k, vS_k, u_k, v_n, vS_n, u_n):
        vf.nodal_values[:] = 1.0e-2 * rng.standard_normal(vf.nodal_values.shape)
    for sf in (p_k, p_n):
        sf.nodal_values[:] = 1.0e-2 * rng.standard_normal(sf.nodal_values.shape)

    phi_k.nodal_values[:] = 0.7
    phi_n.nodal_values[:] = 0.7
    alpha_k.nodal_values[:] = 0.5
    alpha_n.nodal_values[:] = 0.5
    S_k.nodal_values[:] = 0.2
    S_n.nodal_values[:] = 0.2

    forms = build_biofilm_one_domain_forms(
        v_k=v_k,
        p_k=p_k,
        vS_k=vS_k,
        u_k=u_k,
        phi_k=phi_k,
        alpha_k=alpha_k,
        S_k=S_k,
        v_n=v_n,
        p_n=p_n,
        vS_n=vS_n,
        u_n=u_n,
        phi_n=phi_n,
        alpha_n=alpha_n,
        S_n=S_n,
        dv=dv,
        dp=dp,
        dvS=dvS,
        du=du,
        dphi=dphi,
        dalpha=dalpha,
        dS=dS,
        v_test=v_test,
        q_test=q_test,
        vS_test=vS_test,
        u_test=u_test,
        phi_test=phi_test,
        alpha_test=alpha_test,
        S_test=S_test,
        dx=dx(metadata={"q": int(qdeg)}),
        dt=Constant(0.1),
        theta=1.0,
        rho_f=Constant(1.0),
        mu_f=Constant(1.0e-2),
        kappa_inv=Constant(10.0),
        mu_s=Constant(1.0),
        lambda_s=Constant(1.0),
        solid_visco_eta=1.0,
        # Exercise ds_v handling for frozen/Picard linearization
        s_v=Constant(0.0),
        ds_v=Constant(0.0),
    )

    eq = Equation(forms.jacobian_form, forms.residual_form)
    K, R = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=int(qdeg), backend="python")
    assert K.shape[0] == K.shape[1] == dh.total_dofs
    assert R.shape == (dh.total_dofs,)
