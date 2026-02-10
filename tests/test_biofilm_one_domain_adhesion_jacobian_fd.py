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
from pycutfem.ufl.measures import dS, dx
from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms
from pycutfem.utils.meshgen import structured_quad


def _tag_unit_square_boundaries(mesh: Mesh, *, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - 1.0) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - 1.0) <= tol,
        }
    )


def _build_problem(*, nx: int = 2, ny: int = 2, q: int = 5):
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    _tag_unit_square_boundaries(mesh)

    # Mixed unknowns: (v,p,u,phi,alpha,S) with 2D vectors for v,u.
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
    dS_trial = TrialFunction("S", dof_handler=dh)

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
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    S_n = Function("S_n", "S", dof_handler=dh)

    rng = np.random.default_rng(0)
    for vf in (v_k, u_k, v_n, u_n):
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

    ds_bottom = dS(defined_on=mesh.edge_bitset("bottom"), metadata={"q": int(q)})

    # Nonzero adhesion parameters to exercise the added boundary term.
    a_prev = Constant(0.8)

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
        dS=dS_trial,
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
        kappa_inv=Constant(10.0),
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
        ds_adh=ds_bottom,
        adhesion_k_n=5.0,
        adhesion_k_t=2.0,
        adhesion_gamma_n=1.0,
        adhesion_gamma_t=0.5,
        adhesion_a_prev=a_prev,
    )

    # Map global dof -> state function owning it (k-level only).
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


def test_biofilm_one_domain_adhesion_backend_parity_python_cpp():
    dh, forms, _ = _build_problem(nx=2, ny=2, q=5)
    eq = Equation(forms.jacobian_form, forms.residual_form)

    K_py, R_py = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="python")
    K_cpp, R_cpp = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="cpp")

    A_py = K_py.tocsr().toarray()
    A_cpp = K_cpp.tocsr().toarray()
    assert np.allclose(A_py, A_cpp, rtol=1.0e-10, atol=1.0e-12)
    assert np.allclose(np.asarray(R_py, float), np.asarray(R_cpp, float), rtol=1.0e-10, atol=1.0e-12)


def test_biofilm_one_domain_adhesion_jacobian_fd_consistency():
    """
    FD check for the added adhesion boundary term contributions.
    """
    dh, forms, field_to_func_k = _build_problem(nx=2, ny=2, q=5)
    eq = Equation(forms.jacobian_form, forms.residual_form)
    K, R0 = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="python")
    R0 = np.asarray(R0, dtype=float)

    def assemble_residual():
        _, R = assemble_form(Equation(None, forms.residual_form), dof_handler=dh, bcs=[], quad_order=5, backend="python")
        return np.asarray(R, dtype=float)

    # Probe DOFs that couple into the adhesion term (u, alpha).
    probes = []
    for fld in ("u_x", "u_y", "alpha"):
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
        assert rel < 1.0e-6, f"FD mismatch at dof {j} ({fld}): rel={rel:.2e}"
