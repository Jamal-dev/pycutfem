import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant, Function, TestFunction as UFLTestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import ds, dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad

from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms


def _assemble_damage_residual(*, damage_H_prev):
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
            "u_x": 2,
            "u_y": 2,
            "phi": 1,
            "alpha": 1,
            "d": 1,
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
    dd = TrialFunction("d", dof_handler=dh)
    dS = TrialFunction("S", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = UFLTestFunction("p", dof_handler=dh)
    phi_test = UFLTestFunction("phi", dof_handler=dh)
    alpha_test = UFLTestFunction("alpha", dof_handler=dh)
    d_test = UFLTestFunction("d", dof_handler=dh)
    S_test = UFLTestFunction("S", dof_handler=dh)

    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    d_k = Function("d_k", "d", dof_handler=dh)
    S_k = Function("S_k", "S", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    d_n = Function("d_n", "d", dof_handler=dh)
    S_n = Function("S_n", "S", dof_handler=dh)

    # Deterministic state: keep u=0 so the instantaneous ψ⁺ drive is 0.0 and any
    # nonzero residual difference must come from the provided history field.
    for vf in (v_k, u_k, v_n, u_n):
        vf.nodal_values[:] = 0.0
    for sf in (p_k, p_n):
        sf.nodal_values[:] = 0.0

    phi_k.nodal_values[:] = 0.6
    phi_n.nodal_values[:] = 0.6
    alpha_k.nodal_values[:] = 1.0
    alpha_n.nodal_values[:] = 1.0
    d_k.nodal_values[:] = 0.2
    d_n.nodal_values[:] = 0.2
    S_k.nodal_values[:] = 0.2
    S_n.nodal_values[:] = 0.2

    if damage_H_prev is not None and not hasattr(damage_H_prev, "nodal_values"):
        H_prev = Function("H_prev", "d", dof_handler=dh)
        H_prev.nodal_values[:] = float(damage_H_prev)
        damage_H_prev = H_prev

    forms = build_biofilm_one_domain_forms(
        v_k=v_k,
        p_k=p_k,
        u_k=u_k,
        phi_k=phi_k,
        alpha_k=alpha_k,
        d_k=d_k,
        S_k=S_k,
        v_n=v_n,
        p_n=p_n,
        u_n=u_n,
        phi_n=phi_n,
        alpha_n=alpha_n,
        d_n=d_n,
        S_n=S_n,
        dv=dv,
        dp=dp,
        du=du,
        dphi=dphi,
        dalpha=dalpha,
        dd=dd,
        dS=dS,
        v_test=v_test,
        q_test=q_test,
        u_test=u_test,
        phi_test=phi_test,
        alpha_test=alpha_test,
        d_test=d_test,
        S_test=S_test,
        dx=dx(metadata={"q": 3}),
        ds_cip=ds(metadata={"q": 3}),
        dt=Constant(0.1),
        theta=1.0,
        rho_f=Constant(1.0),
        mu_f=Constant(1.0e-2),
        kappa_inv=Constant(10.0),
        mu_s=Constant(1.0),
        lambda_s=Constant(1.0),
        D_phi=0.0,
        gamma_phi=0.0,
        D_alpha=0.0,
        D_S=0.0,
        damage_k=0.0,
        damage_sigma_cr=0.0,
        damage_m=2.0,
        damage_D=0.0,
        damage_gamma_out=0.0,
        damage_kappa_stiff=1.0e-12,
        damage_kappa_perm=1.0e-12,
        damage_model="phase_field",
        damage_eta=1.0,
        damage_Gc=1.0,
        damage_l=0.1,
        damage_psi0=0.0,
        damage_pf_driver="miehe_energy",
        damage_H_prev=damage_H_prev,
        damage_stiff_split="full",
    )
    assert forms.r_damage is not None

    _, R = assemble_form(Equation(None, forms.r_damage), dof_handler=dh, bcs=[], quad_order=5, backend="python")
    R = np.asarray(R, dtype=float)
    sl_d = np.asarray(dh.get_field_slice("d"), dtype=int)
    return R[sl_d]


def test_biofilm_one_domain_phase_field_damage_history_field_changes_residual():
    # No history: equivalent to H_prev=0 for u=0.
    R0 = _assemble_damage_residual(damage_H_prev=None)

    # With history: inject a nonzero H_prev and verify the residual changes.
    R1 = _assemble_damage_residual(damage_H_prev=10.0)

    diff = np.max(np.abs(R1 - R0))
    assert np.isfinite(diff)
    assert diff > 1.0e-8
