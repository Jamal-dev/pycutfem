import numpy as np

from tests.subprocess_utils import run_module_func_in_subprocess

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
from pycutfem.ufl.measures import ds, dx
from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms
from pycutfem.utils.meshgen import structured_quad


def _build_problem(
    *,
    nx: int = 2,
    ny: int = 2,
    q: int = 5,
    D_alpha: float = 0.1,
    alpha_supg: float = 0.0,
    alpha_cip: float = 0.0,
    u_cip: float = 0.0,
    kappa_inv_model: str = "spatial",
    kappa_inv_phi_ref: float | None = None,
    alpha_cahn_M: float = 0.0,
    alpha_cahn_gamma: float = 0.0,
    alpha_cahn_eps: float = 1.0,
    alpha_crack_k: float = 0.0,
    alpha_crack_Dc: float = 0.0,
    alpha_crack_gamma_kappa: float = 0.0,
    alpha_crack_driver: str = "shear",
    # Cahn–Hilliard regularization for alpha (adds mu_alpha block)
    alpha_ch_M: float = 0.0,
    alpha_ch_gamma: float = 0.0,
    alpha_ch_eps: float = 0.1,
    alpha_ch_mobility: str = "constant",
    v_supg: float = 0.0,
    v_supg_mode: str = "streamline",
    v_supg_c_nu: float = 4.0,
    v_cip: float = 0.0,
    fluid_convection: str = "full",
    theta: float = 1.0,
    substrate_reaction_scheme: str = "theta",
    substrate_diffusion_scheme: str = "theta",
):
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )

    ch_enabled = float(alpha_ch_M) != 0.0 and float(alpha_ch_gamma) != 0.0

    # Mixed unknowns: (v,p,vS,u,phi,alpha,[mu_alpha],S) with 2D vectors for v,vS,u.
    field_specs = {
        "v_x": 2,
        "v_y": 2,
        "p": 1,
        "vS_x": 2,
        "vS_y": 2,
        "u_x": 2,
        "u_y": 2,
        "phi": 1,
        "alpha": 1,
    }
    if ch_enabled:
        field_specs["mu_alpha"] = 1
    field_specs["S"] = 1

    me = MixedElement(
        mesh,
        field_specs=field_specs,
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
    dmu_alpha = TrialFunction("mu_alpha", dof_handler=dh) if ch_enabled else None
    dS = TrialFunction("S", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    vS_test = VectorTestFunction(space=VS, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    mu_alpha_test = TestFunction("mu_alpha", dof_handler=dh) if ch_enabled else None
    S_test = TestFunction("S", dof_handler=dh)

    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    vS_k = VectorFunction("vS_k", ["vS_x", "vS_y"], dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    mu_alpha_k = Function("mu_alpha_k", "mu_alpha", dof_handler=dh) if ch_enabled else None
    S_k = Function("S_k", "S", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    vS_n = VectorFunction("vS_n", ["vS_x", "vS_y"], dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    mu_alpha_n = Function("mu_alpha_n", "mu_alpha", dof_handler=dh) if ch_enabled else None
    S_n = Function("S_n", "S", dof_handler=dh)

    rng = np.random.default_rng(0)
    for vf in (v_k, vS_k, u_k, v_n, vS_n, u_n):
        vf.nodal_values[:] = 1.0e-2 * rng.standard_normal(vf.nodal_values.shape)
    for sf in (p_k, p_n):
        sf.nodal_values[:] = 1.0e-2 * rng.standard_normal(sf.nodal_values.shape)

    # Keep nonlinear scalars in safe ranges (avoid K_S + S ~= 0).
    phi_k.nodal_values[:] = np.clip(0.7 + 0.05 * rng.standard_normal(phi_k.nodal_values.shape), 0.2, 0.95)
    phi_n.nodal_values[:] = np.clip(0.7 + 0.05 * rng.standard_normal(phi_n.nodal_values.shape), 0.2, 0.95)
    alpha_k.nodal_values[:] = np.clip(0.5 + 0.05 * rng.standard_normal(alpha_k.nodal_values.shape), 0.05, 0.95)
    alpha_n.nodal_values[:] = np.clip(0.5 + 0.05 * rng.standard_normal(alpha_n.nodal_values.shape), 0.05, 0.95)
    if ch_enabled:
        mu_alpha_k.nodal_values[:] = 1.0e-2 * rng.standard_normal(mu_alpha_k.nodal_values.shape)
        mu_alpha_n.nodal_values[:] = 1.0e-2 * rng.standard_normal(mu_alpha_n.nodal_values.shape)
    S_k.nodal_values[:] = np.clip(0.2 + 0.05 * rng.standard_normal(S_k.nodal_values.shape), 0.01, 1.0)
    S_n.nodal_values[:] = np.clip(0.2 + 0.05 * rng.standard_normal(S_n.nodal_values.shape), 0.01, 1.0)

    dt = Constant(0.1)
    th = float(theta)

    # Volume source in the mixture constraint (matches the Warner-style benchmark):
    #   div(C v + B vS) = alpha * s_v,  with  s_v=(monod(S)-k_d)*(1-phi).
    # This exercises the coupling δ(α s_v) = (δα)s_v + α(δs_v) in the Jacobian.
    mu_max_c = Constant(0.4)
    K_S_c = Constant(0.3)
    k_d_c = Constant(0.1)
    monod = mu_max_c * (S_k / (S_k + K_S_c))
    one_m_phi = (-phi_k) + Constant(1.0)
    s_v = (monod - k_d_c) * one_m_phi
    denom = S_k + K_S_c
    dmonod = mu_max_c * (K_S_c / (denom * denom)) * dS
    ds_v = dmonod * one_m_phi - (monod - k_d_c) * dphi

    forms = build_biofilm_one_domain_forms(
        v_k=v_k,
        p_k=p_k,
        vS_k=vS_k,
        u_k=u_k,
        phi_k=phi_k,
        alpha_k=alpha_k,
        mu_alpha_k=mu_alpha_k,
        S_k=S_k,
        v_n=v_n,
        p_n=p_n,
        vS_n=vS_n,
        u_n=u_n,
        phi_n=phi_n,
        alpha_n=alpha_n,
        mu_alpha_n=mu_alpha_n,
        S_n=S_n,
        dv=dv,
        dp=dp,
        dvS=dvS,
        du=du,
        dphi=dphi,
        dalpha=dalpha,
        dmu_alpha=dmu_alpha,
        dS=dS,
        v_test=v_test,
        q_test=q_test,
        vS_test=vS_test,
        u_test=u_test,
        phi_test=phi_test,
        alpha_test=alpha_test,
        mu_alpha_test=mu_alpha_test,
        S_test=S_test,
        dx=dx(metadata={"q": int(q)}),
        ds_cip=ds(metadata={"q": int(q)}),
        dt=dt,
        theta=th,
        rho_f=Constant(1.0),
        mu_f=Constant(1.0e-2),
        kappa_inv=Constant(10.0),
        kappa_inv_model=str(kappa_inv_model),
        kappa_inv_phi_ref=float(kappa_inv_phi_ref) if kappa_inv_phi_ref is not None else 0.7,
        mu_s=Constant(1.0),
        lambda_s=Constant(1.0),
        D_phi=0.1,
        gamma_phi=1.0,
        D_alpha=float(D_alpha),
        alpha_ch_M=float(alpha_ch_M),
        alpha_ch_gamma=float(alpha_ch_gamma),
        alpha_ch_eps=float(alpha_ch_eps),
        alpha_ch_mobility=str(alpha_ch_mobility),
        v_supg=float(v_supg),
        v_supg_mode=str(v_supg_mode),
        v_supg_c_nu=float(v_supg_c_nu),
        v_cip=float(v_cip),
        fluid_convection=str(fluid_convection),
        alpha_cahn_M=float(alpha_cahn_M),
        alpha_cahn_gamma=float(alpha_cahn_gamma),
        alpha_cahn_eps=float(alpha_cahn_eps),
        alpha_crack_k=float(alpha_crack_k),
        alpha_crack_Dc=float(alpha_crack_Dc),
        alpha_crack_gamma_kappa=float(alpha_crack_gamma_kappa),
        alpha_crack_driver=str(alpha_crack_driver),
        alpha_supg=float(alpha_supg),
        alpha_cip=float(alpha_cip),
        u_cip=float(u_cip),
        D_S=0.1,
        substrate_reaction_scheme=str(substrate_reaction_scheme),
        substrate_diffusion_scheme=str(substrate_diffusion_scheme),
        mu_max=0.4,
        K_S=0.3,
        k_g=0.5,
        k_d=0.1,
        Y=0.8,
        k_det=0.2,
        s_v=s_v,
        ds_v=ds_v,
    )

    # Map global dof -> state function owning it (k-level only).
    field_to_func_k = {
        "v_x": v_k.components[0],
        "v_y": v_k.components[1],
        "p": p_k,
        "vS_x": vS_k.components[0],
        "vS_y": vS_k.components[1],
        "u_x": u_k.components[0],
        "u_y": u_k.components[1],
        "phi": phi_k,
        "alpha": alpha_k,
        **({"mu_alpha": mu_alpha_k} if ch_enabled else {}),
        "S": S_k,
    }

    return dh, forms, field_to_func_k


def _cpp_backend_parity_impl() -> None:
    dh, forms, _ = _build_problem(nx=2, ny=2, q=5)
    eq = Equation(forms.jacobian_form, forms.residual_form)

    K_py, R_py = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="python")
    K_cpp, R_cpp = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="cpp")

    A_py = K_py.tocsr().toarray()
    A_cpp = K_cpp.tocsr().toarray()
    assert np.allclose(A_py, A_cpp, rtol=1.0e-10, atol=1.0e-12)
    assert np.allclose(np.asarray(R_py, float), np.asarray(R_cpp, float), rtol=1.0e-10, atol=1.0e-12)


def test_biofilm_one_domain_backend_parity_python_cpp():
    run_module_func_in_subprocess(__name__, "_cpp_backend_parity_impl")


def test_biofilm_one_domain_jacobian_fd_consistency():
    """
    FD check for the full (v,p,u,phi,alpha,S) one-domain biofilm Jacobian.

    This is intentionally tiny to keep runtime low while catching sign/index bugs.
    """
    for theta in (1.0, 0.5):
        schemes = (("theta", "theta"),) if float(theta) == 1.0 else (("theta", "theta"), ("implicit", "theta"), ("implicit", "implicit"))
        for r_scheme, d_scheme in schemes:
            dh, forms, field_to_func_k = _build_problem(
                nx=2,
                ny=2,
                q=5,
                theta=float(theta),
                substrate_reaction_scheme=str(r_scheme),
                substrate_diffusion_scheme=str(d_scheme),
            )
            eq = Equation(forms.jacobian_form, forms.residual_form)
            K, R0 = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="python")
            R0 = np.asarray(R0, dtype=float)

            def assemble_residual():
                _, R = assemble_form(
                    Equation(None, forms.residual_form), dof_handler=dh, bcs=[], quad_order=5, backend="python"
                )
                return np.asarray(R, dtype=float)

            # Probe a few DOFs (one per field) to cover block couplings.
            probes = []
            for fld in ("v_x", "v_y", "p", "vS_x", "u_x", "phi", "alpha", "S"):
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
                assert rel < 1.0e-6, (
                    f"FD mismatch at dof {j} ({fld}, theta={theta}, substrate_reaction_scheme={r_scheme}, "
                    f"substrate_diffusion_scheme={d_scheme}): rel={rel:.2e}"
                )


def _cpp_backend_parity_v_supg_residual_impl() -> None:
    dh, forms, _ = _build_problem(
        nx=2,
        ny=2,
        q=5,
        v_supg=1.0,
        v_supg_mode="residual",
        v_supg_c_nu=4.0,
        fluid_convection="full",
    )
    eq = Equation(forms.jacobian_form, forms.residual_form)

    K_py, R_py = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="python")
    K_cpp, R_cpp = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="cpp")

    A_py = K_py.tocsr().toarray()
    A_cpp = K_cpp.tocsr().toarray()
    assert np.allclose(A_py, A_cpp, rtol=1.0e-10, atol=1.0e-12)
    assert np.allclose(np.asarray(R_py, float), np.asarray(R_cpp, float), rtol=1.0e-10, atol=1.0e-12)


def test_biofilm_one_domain_v_supg_residual_backend_parity_python_cpp():
    run_module_func_in_subprocess(__name__, "_cpp_backend_parity_v_supg_residual_impl")


def _all_backend_parity_v_supg_residual_momentum_impl() -> None:
    dh, forms, _ = _build_problem(
        nx=1,
        ny=1,
        q=4,
        v_supg=1.0,
        v_supg_mode="residual",
        v_supg_c_nu=4.0,
        fluid_convection="full",
    )
    equations = {
        "momentum_residual": Equation(None, forms.r_momentum),
        "momentum_jacobian": Equation(forms.a_momentum, None),
    }

    for name, eq in equations.items():
        assembled = {}
        for backend in ("python", "jit", "cpp"):
            K, R = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=4, backend=backend)
            assembled[backend] = K.tocsr().toarray() if K is not None else np.asarray(R, dtype=float)

        ref = assembled["python"]
        for backend in ("jit", "cpp"):
            got = assembled[backend]
            assert np.allclose(got, ref, rtol=1.0e-10, atol=1.0e-12), (
                f"{name} backend parity failed for {backend}: "
                f"max_abs={float(np.max(np.abs(got - ref))):.3e}"
            )


def test_biofilm_one_domain_v_supg_residual_momentum_backend_parity_all_backends():
    run_module_func_in_subprocess(__name__, "_all_backend_parity_v_supg_residual_momentum_impl")


def test_biofilm_one_domain_v_supg_residual_jacobian_fd_consistency():
    dh, forms, field_to_func_k = _build_problem(
        nx=2,
        ny=2,
        q=5,
        v_supg=1.0,
        v_supg_mode="residual",
        v_supg_c_nu=4.0,
        fluid_convection="full",
    )
    eq = Equation(forms.jacobian_form, forms.residual_form)
    K, R0 = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="python")
    R0 = np.asarray(R0, dtype=float)

    def assemble_residual():
        _, R = assemble_form(Equation(None, forms.residual_form), dof_handler=dh, bcs=[], quad_order=5, backend="python")
        return np.asarray(R, dtype=float)

    probes = []
    for fld in ("v_x", "v_y", "p", "vS_x", "phi", "alpha"):
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


def _cpp_backend_parity_v_cip_impl() -> None:
    dh, forms, _ = _build_problem(
        nx=2,
        ny=2,
        q=5,
        v_cip=1.0,
        fluid_convection="full",
    )
    eq = Equation(forms.jacobian_form, forms.residual_form)

    K_py, R_py = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="python")
    K_cpp, R_cpp = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="cpp")

    A_py = K_py.tocsr().toarray()
    A_cpp = K_cpp.tocsr().toarray()
    assert np.allclose(A_py, A_cpp, rtol=1.0e-10, atol=1.0e-12)
    assert np.allclose(np.asarray(R_py, float), np.asarray(R_cpp, float), rtol=1.0e-10, atol=1.0e-12)


def test_biofilm_one_domain_v_cip_backend_parity_python_cpp():
    run_module_func_in_subprocess(__name__, "_cpp_backend_parity_v_cip_impl")


def test_biofilm_one_domain_v_cip_jacobian_fd_consistency():
    dh, forms, field_to_func_k = _build_problem(
        nx=2,
        ny=2,
        q=5,
        v_cip=1.0,
        fluid_convection="full",
    )
    eq = Equation(forms.jacobian_form, forms.residual_form)
    K, R0 = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="python")
    R0 = np.asarray(R0, dtype=float)

    def assemble_residual():
        _, R = assemble_form(Equation(None, forms.residual_form), dof_handler=dh, bcs=[], quad_order=5, backend="python")
        return np.asarray(R, dtype=float)

    probes = []
    for fld in ("v_x", "v_y", "alpha"):
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


def _cpp_backend_parity_alpha_stabilization_impl() -> None:
    dh, forms, _ = _build_problem(nx=2, ny=2, q=5, D_alpha=0.0, alpha_supg=0.5, alpha_cip=2.0)
    eq = Equation(forms.jacobian_form, forms.residual_form)

    K_py, R_py = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="python")
    K_cpp, R_cpp = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="cpp")

    A_py = K_py.tocsr().toarray()
    A_cpp = K_cpp.tocsr().toarray()
    assert np.allclose(A_py, A_cpp, rtol=1.0e-10, atol=1.0e-12)
    assert np.allclose(np.asarray(R_py, float), np.asarray(R_cpp, float), rtol=1.0e-10, atol=1.0e-12)


def test_biofilm_one_domain_alpha_stabilization_backend_parity_python_cpp():
    run_module_func_in_subprocess(__name__, "_cpp_backend_parity_alpha_stabilization_impl")


def test_biofilm_one_domain_alpha_stabilization_jacobian_fd_consistency():
    """
    FD check that the (SUPG + CIP) stabilization additions to the alpha equation
    remain consistent with the manually coded Jacobian.
    """
    dh, forms, field_to_func_k = _build_problem(nx=2, ny=2, q=5, D_alpha=0.0, alpha_supg=0.5, alpha_cip=2.0)
    eq = Equation(forms.jacobian_form, forms.residual_form)
    K, R0 = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="python")
    R0 = np.asarray(R0, dtype=float)

    def assemble_residual():
        _, R = assemble_form(Equation(None, forms.residual_form), dof_handler=dh, bcs=[], quad_order=5, backend="python")
        return np.asarray(R, dtype=float)

    probes = []
    for fld in ("v_x", "p", "u_x", "alpha"):
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


def _cpp_backend_parity_u_cip_impl() -> None:
    dh, forms, _ = _build_problem(nx=2, ny=2, q=5, u_cip=1.0)
    eq = Equation(forms.jacobian_form, forms.residual_form)

    K_py, R_py = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="python")
    K_cpp, R_cpp = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="cpp")

    A_py = K_py.tocsr().toarray()
    A_cpp = K_cpp.tocsr().toarray()
    assert np.allclose(A_py, A_cpp, rtol=1.0e-10, atol=1.0e-12)
    assert np.allclose(np.asarray(R_py, float), np.asarray(R_cpp, float), rtol=1.0e-10, atol=1.0e-12)


def test_biofilm_one_domain_u_cip_backend_parity_python_cpp():
    run_module_func_in_subprocess(__name__, "_cpp_backend_parity_u_cip_impl")


def test_biofilm_one_domain_u_cip_jacobian_fd_consistency():
    """
    FD check that the u-CIP stabilization remains consistent with the manually coded Jacobian.
    """
    dh, forms, field_to_func_k = _build_problem(nx=2, ny=2, q=5, u_cip=1.0)
    eq = Equation(forms.jacobian_form, forms.residual_form)
    K, R0 = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="python")
    R0 = np.asarray(R0, dtype=float)

    def assemble_residual():
        _, R = assemble_form(Equation(None, forms.residual_form), dof_handler=dh, bcs=[], quad_order=5, backend="python")
        return np.asarray(R, dtype=float)

    probes = []
    for fld in ("u_x", "u_y"):
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


def _cpp_backend_parity_kc_permeability_impl() -> None:
    dh, forms, _ = _build_problem(nx=2, ny=2, q=5, kappa_inv_model="kozeny_carman", kappa_inv_phi_ref=0.7)
    eq = Equation(forms.jacobian_form, forms.residual_form)

    K_py, R_py = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="python")
    K_cpp, R_cpp = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="cpp")

    A_py = K_py.tocsr().toarray()
    A_cpp = K_cpp.tocsr().toarray()
    assert np.allclose(A_py, A_cpp, rtol=1.0e-10, atol=1.0e-12)
    assert np.allclose(np.asarray(R_py, float), np.asarray(R_cpp, float), rtol=1.0e-10, atol=1.0e-12)


def test_biofilm_one_domain_kc_permeability_backend_parity_python_cpp():
    run_module_func_in_subprocess(__name__, "_cpp_backend_parity_kc_permeability_impl")


def test_biofilm_one_domain_kc_permeability_jacobian_fd_consistency():
    """
    FD check that Kozeny–Carman permeability dependence on phi remains consistent.
    """
    dh, forms, field_to_func_k = _build_problem(nx=2, ny=2, q=5, kappa_inv_model="kc", kappa_inv_phi_ref=0.7)
    eq = Equation(forms.jacobian_form, forms.residual_form)
    K, R0 = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="python")
    R0 = np.asarray(R0, dtype=float)

    def assemble_residual():
        _, R = assemble_form(Equation(None, forms.residual_form), dof_handler=dh, bcs=[], quad_order=5, backend="python")
        return np.asarray(R, dtype=float)

    # Probe a mid phi DOF (kappa_inv(phi) affects beta and therefore both momentum and skeleton drag).
    probes = []
    sl = dh.get_field_slice("phi")
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


def _cpp_backend_parity_alpha_cahn_crack_impl() -> None:
    dh, forms, _ = _build_problem(
        nx=2,
        ny=2,
        q=5,
        D_alpha=0.05,
        alpha_cahn_M=0.2,
        alpha_cahn_gamma=1.0,
        alpha_cahn_eps=0.1,
        alpha_crack_k=0.5,
        alpha_crack_Dc=0.0,
        alpha_crack_gamma_kappa=0.0,
        alpha_crack_driver="shear",
    )
    eq = Equation(forms.jacobian_form, forms.residual_form)

    K_py, R_py = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="python")
    K_cpp, R_cpp = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="cpp")

    A_py = K_py.tocsr().toarray()
    A_cpp = K_cpp.tocsr().toarray()
    assert np.allclose(A_py, A_cpp, rtol=1.0e-10, atol=1.0e-12)
    assert np.allclose(np.asarray(R_py, float), np.asarray(R_cpp, float), rtol=1.0e-10, atol=1.0e-12)


def test_biofilm_one_domain_alpha_cahn_crack_backend_parity_python_cpp():
    run_module_func_in_subprocess(__name__, "_cpp_backend_parity_alpha_cahn_crack_impl")


def test_biofilm_one_domain_alpha_cahn_crack_jacobian_fd_consistency():
    """
    FD check for the Allen–Cahn + crack-propagation additions to the alpha equation.
    """
    dh, forms, field_to_func_k = _build_problem(
        nx=2,
        ny=2,
        q=5,
        D_alpha=0.05,
        alpha_cahn_M=0.2,
        alpha_cahn_gamma=1.0,
        alpha_cahn_eps=0.1,
        alpha_crack_k=0.5,
        alpha_crack_Dc=0.0,
        alpha_crack_gamma_kappa=0.0,
        alpha_crack_driver="shear",
    )
    eq = Equation(forms.jacobian_form, forms.residual_form)
    K, R0 = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="python")
    R0 = np.asarray(R0, dtype=float)

    def assemble_residual():
        _, R = assemble_form(Equation(None, forms.residual_form), dof_handler=dh, bcs=[], quad_order=5, backend="python")
        return np.asarray(R, dtype=float)

    probes = []
    sl = dh.get_field_slice("alpha")
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
        assert rel < 2.0e-6, f"FD mismatch at dof {j} ({fld}): rel={rel:.2e}"


def _cpp_backend_parity_alpha_cahn_hilliard_impl() -> None:
    dh, forms, _ = _build_problem(
        nx=2,
        ny=2,
        q=5,
        D_alpha=0.0,
        alpha_ch_M=0.4,
        alpha_ch_gamma=1.0,
        alpha_ch_eps=0.1,
        alpha_ch_mobility="degenerate",
    )
    assert forms.r_mu_alpha is not None
    assert forms.a_mu_alpha is not None

    eq = Equation(forms.jacobian_form, forms.residual_form)
    K_py, R_py = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="python")
    K_cpp, R_cpp = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="cpp")

    assert np.allclose(K_py.tocsr().toarray(), K_cpp.tocsr().toarray(), rtol=1.0e-10, atol=1.0e-12)
    assert np.allclose(np.asarray(R_py, float), np.asarray(R_cpp, float), rtol=1.0e-10, atol=1.0e-12)


def test_biofilm_one_domain_alpha_cahn_hilliard_backend_parity_python_cpp():
    run_module_func_in_subprocess(__name__, "_cpp_backend_parity_alpha_cahn_hilliard_impl")


def test_biofilm_one_domain_alpha_cahn_hilliard_jacobian_fd_consistency():
    """
    FD check for the coupled (alpha, mu_alpha) Cahn–Hilliard additions.
    """
    dh, forms, field_to_func_k = _build_problem(
        nx=2,
        ny=2,
        q=5,
        D_alpha=0.0,
        alpha_ch_M=0.4,
        alpha_ch_gamma=1.0,
        alpha_ch_eps=0.1,
        alpha_ch_mobility="degenerate",
    )
    eq = Equation(forms.jacobian_form, forms.residual_form)
    K, R0 = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=5, backend="python")
    R0 = np.asarray(R0, dtype=float)

    def assemble_residual():
        _, R = assemble_form(Equation(None, forms.residual_form), dof_handler=dh, bcs=[], quad_order=5, backend="python")
        return np.asarray(R, dtype=float)

    probes = []
    for fld in ("alpha", "mu_alpha"):
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
