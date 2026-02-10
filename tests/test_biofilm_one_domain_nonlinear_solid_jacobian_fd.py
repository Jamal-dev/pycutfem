import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction,
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


def _build_problem(*, nx: int = 1, ny: int = 1, q: int = 4):
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=2)
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
    q_test = TestFunction("p", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    S_test = TestFunction("S", dof_handler=dh)

    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    S_k = Function("S_k", "S", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    u_nm1 = VectorFunction("u_nm1", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    S_n = Function("S_n", "S", dof_handler=dh)

    rng = np.random.default_rng(0)
    for vf in (v_k, u_k, v_n, u_n, u_nm1):
        vf.nodal_values[:] = 1.0e-2 * rng.standard_normal(vf.nodal_values.shape)
    for sf in (p_k, p_n):
        sf.nodal_values[:] = 1.0e-2 * rng.standard_normal(sf.nodal_values.shape)

    # Keep nonlinear scalars in safe ranges (avoid K_S + S ~= 0).
    phi_k.nodal_values[:] = np.clip(0.7 + 0.05 * rng.standard_normal(phi_k.nodal_values.shape), 0.2, 0.95)
    phi_n.nodal_values[:] = np.clip(0.7 + 0.05 * rng.standard_normal(phi_n.nodal_values.shape), 0.2, 0.95)
    alpha_k.nodal_values[:] = np.clip(0.5 + 0.05 * rng.standard_normal(alpha_k.nodal_values.shape), 0.05, 0.95)
    alpha_n.nodal_values[:] = np.clip(0.5 + 0.05 * rng.standard_normal(alpha_n.nodal_values.shape), 0.05, 0.95)
    S_k.nodal_values[:] = np.clip(0.2 + 0.05 * rng.standard_normal(S_k.nodal_values.shape), 0.01, 1.0)
    S_n.nodal_values[:] = np.clip(0.2 + 0.05 * rng.standard_normal(S_n.nodal_values.shape), 0.01, 1.0)

    dt = Constant(0.1)
    th = 1.0

    # Reference inverse permeability K^{-1} (matrix) for Eulerian ref-map push-forward.
    K_inv = Constant(10.0 * np.eye(2, dtype=float))

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
        u_nm1=u_nm1,
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
        dx=dx(metadata={"q": int(q)}),
        dt=dt,
        theta=th,
        rho_f=Constant(1.0),
        mu_f=Constant(1.0e-2),
        kappa_inv=K_inv,
        kappa_inv_model="refmap",
        solid_model="neo_hookean",
        rho_s0_tilde=Constant(0.5),
        include_skeleton_acceleration=True,
        mu_s=Constant(1.0),
        lambda_s=Constant(1.0),
        D_phi=0.1,
        gamma_phi=1.0,
        D_alpha=0.1,
        D_S=0.1,
        mu_max=0.4,
        K_S=0.3,
        k_g=0.5,
        k_d=0.1,
        Y=0.8,
        k_det=0.2,
    )

    field_to_func_k = {
        "v_x": v_k.components[0],
        "v_y": v_k.components[1],
        "p": p_k,
        "u_x": u_k.components[0],
        "u_y": u_k.components[1],
        "phi": phi_k,
        "alpha": alpha_k,
        "S": S_k,
    }

    return dh, forms, field_to_func_k


def test_biofilm_one_domain_nh_jacobian_fd_consistency():
    """
    FD check for the biofilm Jacobian with:
      - Neo-Hookean Eulerian reference-map solid
      - Eulerian push-forward inverse permeability k^{-1}(u)
      - skeleton acceleration term enabled
    """
    dh, forms, field_to_func_k = _build_problem(nx=1, ny=1, q=4)
    eq = Equation(forms.jacobian_form, forms.residual_form)
    K, R0 = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=4, backend="python")
    R0 = np.asarray(R0, dtype=float)

    def assemble_residual():
        _, R = assemble_form(Equation(None, forms.residual_form), dof_handler=dh, bcs=[], quad_order=4, backend="python")
        return np.asarray(R, dtype=float)

    # Probe a few DOFs (one per block) to cover couplings.
    probes = []
    for fld in ("v_x", "p", "u_x", "phi", "alpha", "S"):
        sl = dh.get_field_slice(fld)
        if sl:
            probes.append(int(sl[len(sl) // 2]))

    eps = 1.0e-8
    for j in probes:
        fld, _ = dh._dof_to_node_map[j]
        func = field_to_func_k[fld]
        old = float(func.get_nodal_values(np.asarray([j], dtype=int))[0])
        func.set_nodal_values(np.asarray([j], dtype=int), np.asarray([old + eps], dtype=float))
        R1 = assemble_residual()
        func.set_nodal_values(np.asarray([j], dtype=int), np.asarray([old], dtype=float))

        fd = (R1 - R0) / eps
        col = np.asarray(K.getcol(j).toarray()).reshape(-1)
        denom = max(1.0, float(np.linalg.norm(fd, ord=np.inf)), float(np.linalg.norm(col, ord=np.inf)))
        rel = float(np.linalg.norm(fd - col, ord=np.inf)) / denom
        assert rel < 5.0e-6, f"FD mismatch at dof {j} ({fld}): rel={rel:.2e}"

