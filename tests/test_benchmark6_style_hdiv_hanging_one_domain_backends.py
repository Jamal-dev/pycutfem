import numpy as np
import pytest

from examples.biofilms.benchmarks.blauert.blauert_biofilm_deformation_one_domain import refine_around_biofilm_bbox
from examples.utils.biofilm.one_domain import _epsilon, _sqrt, build_biofilm_one_domain_forms
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver
from pycutfem.ufl.expressions import (
    Constant,
    div,
    dot,
    Function,
    grad,
    HdivFunction,
    HdivTestFunction,
    HdivTrialFunction,
    inner,
    TestFunction as UFLTestFunction,
    TrialFunction as UFLTrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
)
from pycutfem.ufl.measures import dS as dS_measure, ds, dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def _build_benchmark6_style_hdiv_case(
    *,
    v_supg: float = 0.0,
    v_supg_mode: str = "streamline",
    weak_tangential_dirichlet: bool = False,
    weak_tangential_method: str = "penalty",
    fluid_hdiv_order: int = 0,
):
    L = 5.5e-3
    H = 1.0e-3
    nodes, elems, _edges, corners = structured_quad(L, H, nx=2, ny=1, poly_order=1, offset=(0.0, 0.0))
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=np.asarray(elems, dtype=int),
        elements_corner_nodes=np.asarray(corners, dtype=int),
        element_type="quad",
        poly_order=1,
    )
    poly = np.asarray(
        [
            [0.6e-3, 0.10e-3],
            [1.9e-3, 0.10e-3],
            [1.9e-3, 0.55e-3],
            [0.6e-3, 0.55e-3],
            [0.6e-3, 0.10e-3],
        ],
        dtype=float,
    )
    mesh = refine_around_biofilm_bbox(mesh, poly=poly, band=1.0e-6, expand_layers=0, L=L, H=H)
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= 1.0e-12,
            "right": lambda x, y: abs(x - L) <= 1.0e-12,
            "bottom": lambda x, y: abs(y - 0.0) <= 1.0e-12,
            "top": lambda x, y: abs(y - H) <= 1.0e-12,
        }
    )

    me = MixedElement(
        mesh,
        field_specs={
            "v": ("RT", int(fluid_hdiv_order)),
            "p": ("DG", 0),
            "vS_x": 1,
            "vS_y": 1,
            "u_x": 1,
            "u_y": 1,
            "phi": 1,
            "alpha": 1,
            "mu_alpha": 1,
            "S": 1,
        },
    )
    dh = DofHandler(me, method="cg")

    VS = FunctionSpace("VS", ["vS_x", "vS_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    dv = HdivTrialFunction("v")
    dp = UFLTrialFunction("p", dof_handler=dh)
    dvS = VectorTrialFunction(space=VS, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    dphi = UFLTrialFunction("phi", dof_handler=dh)
    dalpha = UFLTrialFunction("alpha", dof_handler=dh)
    dmu_alpha = UFLTrialFunction("mu_alpha", dof_handler=dh)
    dS = UFLTrialFunction("S", dof_handler=dh)

    v_test = HdivTestFunction("v")
    q_test = UFLTestFunction("p", dof_handler=dh)
    vS_test = VectorTestFunction(space=VS, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    phi_test = UFLTestFunction("phi", dof_handler=dh)
    alpha_test = UFLTestFunction("alpha", dof_handler=dh)
    mu_alpha_test = UFLTestFunction("mu_alpha", dof_handler=dh)
    S_test = UFLTestFunction("S", dof_handler=dh)

    v_k = HdivFunction("v_k", "v", dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    vS_k = VectorFunction("vS_k", ["vS_x", "vS_y"], dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    mu_alpha_k = Function("mu_alpha_k", "mu_alpha", dof_handler=dh)
    S_k = Function("S_k", "S", dof_handler=dh)

    v_n = HdivFunction("v_n", "v", dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    vS_n = VectorFunction("vS_n", ["vS_x", "vS_y"], dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    mu_alpha_n = Function("mu_alpha_n", "mu_alpha", dof_handler=dh)
    S_n = Function("S_n", "S", dof_handler=dh)

    v_xy = np.asarray(dh.get_dof_coords("v"), dtype=float)
    v_k.nodal_values[:] = 2.5e-4 + 8.0e-5 * v_xy[:, 0] / L - 5.0e-5 * v_xy[:, 1] / H
    v_n.nodal_values[:] = 1.7e-4 - 4.0e-5 * v_xy[:, 0] / L + 3.0e-5 * v_xy[:, 1] / H

    p_xy = np.asarray(dh.get_dof_coords("p"), dtype=float)
    p_k.nodal_values[:] = 20.0 + 2.0 * p_xy[:, 0] / L - 1.0 * p_xy[:, 1] / H
    p_n.nodal_values[:] = 15.0 + 1.0 * p_xy[:, 0] / L - 0.5 * p_xy[:, 1] / H

    vS_k.set_values_from_function(lambda x, y: np.asarray([1.0e-5 + 3.0e-6 * x / L, -0.8e-5 - 2.0e-6 * y / H]))
    vS_n.set_values_from_function(lambda x, y: np.asarray([0.8e-5 + 2.0e-6 * x / L, -0.5e-5 - 1.0e-6 * y / H]))
    u_k.set_values_from_function(lambda x, y: np.asarray([1.5e-5 + 2.0e-5 * x / L, -1.2e-5 - 1.5e-5 * y / H]))
    u_n.set_values_from_function(lambda x, y: np.asarray([1.0e-5 + 1.0e-5 * x / L, -0.8e-5 - 1.0e-5 * y / H]))

    for fk, fn, basek, basen in (
        (phi_k, phi_n, 0.55, 0.58),
        (alpha_k, alpha_n, 0.42, 0.40),
        (mu_alpha_k, mu_alpha_n, 1.0e-3, 8.0e-4),
        (S_k, S_n, 0.25, 0.22),
    ):
        xy = np.asarray(dh.get_dof_coords(fk.field_name), dtype=float)
        fk.nodal_values[:] = basek + 0.03 * xy[:, 0] / L - 0.02 * xy[:, 1] / H
        fn.nodal_values[:] = basen + 0.02 * xy[:, 0] / L - 0.01 * xy[:, 1] / H

    phi_k.nodal_values[:] = np.clip(phi_k.nodal_values, 0.25, 0.85)
    phi_n.nodal_values[:] = np.clip(phi_n.nodal_values, 0.25, 0.85)
    alpha_k.nodal_values[:] = np.clip(alpha_k.nodal_values, 0.05, 0.95)
    alpha_n.nodal_values[:] = np.clip(alpha_n.nodal_values, 0.05, 0.95)
    S_k.nodal_values[:] = np.clip(S_k.nodal_values, 0.05, 0.80)
    S_n.nodal_values[:] = np.clip(S_n.nodal_values, 0.05, 0.80)

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
        dx=dx(metadata={"q": 4}),
        ds_cip=ds(metadata={"q": 4}),
        dt=Constant(0.05),
        theta=1.0,
        rho_f=Constant(1000.0),
        mu_f=Constant(1.0e-3),
        kappa_inv=Constant(1.4715e12),
        mu_s=Constant(50.0),
        lambda_s=Constant(75.0),
        gamma_div=1.0e-3,
        D_phi=0.0,
        gamma_phi=5.0,
        D_alpha=0.0,
        alpha_advect_with="vS",
        alpha_advection_form="conservative",
        alpha_ch_M=0.0,
        alpha_ch_gamma=0.0,
        alpha_mu_aux_pin=1.0,
        alpha_supg=0.0,
        alpha_cip=0.0,
        u_cip=0.0,
        v_supg=float(v_supg),
        v_supg_mode=str(v_supg_mode),
        ds_hdiv_tangential=(
            dS_measure(
                defined_on=mesh.edge_bitset("left") | mesh.edge_bitset("bottom") | mesh.edge_bitset("top"),
                metadata={"q": 4},
            )
            if weak_tangential_dirichlet
            else None
        ),
        hdiv_tangential_gamma=20.0,
        hdiv_tangential_method=str(weak_tangential_method),
        fluid_hdiv_order=int(fluid_hdiv_order),
        v_cip=0.0,
        vS_cip=0.0,
        mu_max=0.4,
        K_S=0.3,
        k_g=0.5,
        k_d=0.1,
        Y=0.8,
        k_det=0.2,
        s_v=Constant(0.0),
        ds_v=Constant(0.0),
    )

    all_funcs = [
        v_k,
        p_k,
        vS_k,
        u_k,
        phi_k,
        alpha_k,
        mu_alpha_k,
        S_k,
        v_n,
        p_n,
        vS_n,
        u_n,
        phi_n,
        alpha_n,
        mu_alpha_n,
        S_n,
    ]
    coeffs = {f.name: f for f in all_funcs}
    return dh, me, forms, coeffs, all_funcs


def _assemble_reduced_system(backend: str):
    dh, me, forms, coeffs, all_funcs = _build_benchmark6_style_hdiv_case()
    solver = NewtonSolver(
        forms.residual_form,
        forms.jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(newton_tol=1.0e-10, max_newton_iter=2, line_search=False),
        quad_order=4,
        backend=backend,
    )
    assert solver.constraints is not None, "expected hanging-node constraints on the benchmark-6-style refined mesh"
    v_gids = set(map(int, dh.get_field_slice("v")))
    assert any(int(sdof) in v_gids for sdof in solver.constraints.slaves), "expected RT slave DOFs on the refined mesh"
    solver._enforce_constraints_on_functions(all_funcs)
    A_red, R_red = solver._assemble_system_reduced(coeffs, need_matrix=True)
    return A_red.toarray(), np.asarray(R_red, dtype=float)


def _assemble_reduced_system_with_supg(
    backend: str,
    *,
    v_supg: float,
    v_supg_mode: str,
    weak_tangential_dirichlet: bool = False,
    weak_tangential_method: str = "penalty",
    fluid_hdiv_order: int = 0,
):
    dh, me, forms, coeffs, all_funcs = _build_benchmark6_style_hdiv_case(
        v_supg=v_supg,
        v_supg_mode=v_supg_mode,
        weak_tangential_dirichlet=weak_tangential_dirichlet,
        weak_tangential_method=weak_tangential_method,
        fluid_hdiv_order=fluid_hdiv_order,
    )
    solver = NewtonSolver(
        forms.residual_form,
        forms.jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(newton_tol=1.0e-10, max_newton_iter=2, line_search=False),
        quad_order=4,
        backend=backend,
    )
    solver._enforce_constraints_on_functions(all_funcs)
    A_red, R_red = solver._assemble_system_reduced(coeffs, need_matrix=True)
    return A_red.toarray(), np.asarray(R_red, dtype=float)


@pytest.mark.parametrize(
    ("backend", "v_supg", "v_supg_mode", "weak_tangential_dirichlet", "weak_tangential_method"),
    (
        ("cpp", 0.0, "streamline", False, "penalty"),
        ("cpp", 0.5, "residual", False, "penalty"),
        ("cpp", 0.5, "residual", True, "penalty"),
        ("cpp", 0.5, "residual", True, "nitsche"),
    ),
)
def test_benchmark6_style_hdiv_hanging_one_domain_backend_parity_matches_python(
    monkeypatch,
    tmp_path,
    backend,
    v_supg,
    v_supg_mode,
    weak_tangential_dirichlet,
    weak_tangential_method,
):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"jit_cache_{backend}"))
    if backend == "cpp":
        monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")
    else:
        monkeypatch.delenv("PYCUTFEM_JIT_BACKEND", raising=False)

    A_ref, R_ref = _assemble_reduced_system_with_supg(
        "python",
        v_supg=v_supg,
        v_supg_mode=v_supg_mode,
        weak_tangential_dirichlet=weak_tangential_dirichlet,
        weak_tangential_method=weak_tangential_method,
    )
    A, R = _assemble_reduced_system_with_supg(
        backend,
        v_supg=v_supg,
        v_supg_mode=v_supg_mode,
        weak_tangential_dirichlet=weak_tangential_dirichlet,
        weak_tangential_method=weak_tangential_method,
    )

    np.testing.assert_allclose(A, A_ref, rtol=1.0e-9, atol=2.0e-7)
    np.testing.assert_allclose(R, R_ref, rtol=1.0e-10, atol=1.0e-10)


def test_hdiv_tangential_nitsche_rt1_changes_reduced_operator():
    A_pen, R_pen = _assemble_reduced_system_with_supg(
        "python",
        v_supg=0.0,
        v_supg_mode="streamline",
        weak_tangential_dirichlet=True,
        weak_tangential_method="penalty",
        fluid_hdiv_order=1,
    )
    A_nit, R_nit = _assemble_reduced_system_with_supg(
        "python",
        v_supg=0.0,
        v_supg_mode="streamline",
        weak_tangential_dirichlet=True,
        weak_tangential_method="nitsche",
        fluid_hdiv_order=1,
    )

    assert float(np.max(np.abs(A_pen - A_nit))) > 1.0e-6
    assert float(np.max(np.abs(R_pen - R_nit))) > 1.0e-8


def test_benchmark6_style_hdiv_rt1_nitsche_backend_parity_matches_python(monkeypatch, tmp_path):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "jit_cache_cpp_rt1_nitsche"))
    monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")

    A_ref, R_ref = _assemble_reduced_system_with_supg(
        "python",
        v_supg=0.0,
        v_supg_mode="streamline",
        weak_tangential_dirichlet=True,
        weak_tangential_method="nitsche",
        fluid_hdiv_order=1,
    )
    A_cpp, R_cpp = _assemble_reduced_system_with_supg(
        "cpp",
        v_supg=0.0,
        v_supg_mode="streamline",
        weak_tangential_dirichlet=True,
        weak_tangential_method="nitsche",
        fluid_hdiv_order=1,
    )

    np.testing.assert_allclose(A_cpp, A_ref, rtol=1.0e-9, atol=5.0e-7)
    np.testing.assert_allclose(R_cpp, R_ref, rtol=1.0e-10, atol=1.0e-10)


def test_benchmark6_style_hdiv_rt1_residual_supg_nitsche_cpp_matches_python(monkeypatch, tmp_path):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "jit_cache_cpp_rt1_residual"))
    monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")

    A_ref, R_ref = _assemble_reduced_system_with_supg(
        "python",
        v_supg=0.5,
        v_supg_mode="residual",
        weak_tangential_dirichlet=True,
        weak_tangential_method="nitsche",
        fluid_hdiv_order=1,
    )
    A_cpp, R_cpp = _assemble_reduced_system_with_supg(
        "cpp",
        v_supg=0.5,
        v_supg_mode="residual",
        weak_tangential_dirichlet=True,
        weak_tangential_method="nitsche",
        fluid_hdiv_order=1,
    )

    np.testing.assert_allclose(A_cpp, A_ref, rtol=1.0e-9, atol=5.0e-7)
    np.testing.assert_allclose(R_cpp, R_ref, rtol=1.0e-10, atol=1.0e-10)


def test_benchmark6_style_hdiv_alpha_residual_matches_manual_transport_on_hanging_mesh():
    dh, me, forms, coeffs, all_funcs = _build_benchmark6_style_hdiv_case(v_supg=0.0, v_supg_mode="streamline")
    solver = NewtonSolver(
        forms.residual_form,
        forms.jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(newton_tol=1.0e-10, max_newton_iter=2, line_search=False),
        quad_order=4,
        backend="python",
    )
    solver._enforce_constraints_on_functions(all_funcs)

    alpha_k = coeffs["alpha_k"]
    alpha_n = coeffs["alpha_n"]
    alpha_test = UFLTestFunction("alpha", dof_handler=dh)
    vS_k = coeffs["vS_k"]
    v_n = coeffs["v_n"]
    phi_k = coeffs["phi_k"]
    S_k = coeffs["S_k"]

    inv_dt = Constant(20.0)
    mu_max_c = Constant(0.4)
    K_S_c = Constant(0.3)
    k_g_c = Constant(0.5)
    k_det_c = Constant(0.2)
    one_c = Constant(1.0)

    G_k = k_g_c * (mu_max_c * (S_k / (S_k + K_S_c))) * (one_c - phi_k)
    delta_k = Constant(4.0) * alpha_k * (one_c - alpha_k)
    D_det_prev = k_det_c * _sqrt(inner(_epsilon(v_n), _epsilon(v_n)) + Constant(1.0e-12))
    adv_k = dot(grad(alpha_k), vS_k) + alpha_k * div(vS_k)
    manual_alpha = alpha_test * (
        ((alpha_k - alpha_n) * inv_dt)
        + adv_k
        - G_k * alpha_k * (one_c - alpha_k)
        + D_det_prev * delta_k
    ) * dx(metadata={"q": 4})

    _, R_alpha = assemble_form(Equation(None, forms.r_alpha), dh, bcs=[], backend="python")
    _, R_manual = assemble_form(Equation(None, manual_alpha), dh, bcs=[], backend="python")

    if solver.constraints is not None:
        R_alpha = solver.constraints.restrict_full(np.asarray(R_alpha, dtype=float))
        R_manual = solver.constraints.restrict_full(np.asarray(R_manual, dtype=float))
    else:
        R_alpha = np.asarray(R_alpha, dtype=float)
        R_manual = np.asarray(R_manual, dtype=float)

    np.testing.assert_allclose(R_alpha, R_manual, rtol=1.0e-10, atol=1.0e-10)
