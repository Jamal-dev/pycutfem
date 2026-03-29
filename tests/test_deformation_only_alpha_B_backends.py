import numpy as np
import pytest
from scipy import sparse

from examples.utils.biofilm.deformation_only import build_deformation_only_forms
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant, Function, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction
from pycutfem.ufl.expressions import TestFunction as UflTestFunction
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def _assemble_vector(*, problem, form, qdeg: int, backend: str) -> np.ndarray:
    _, residual = assemble_form(
        Equation(None, form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=int(qdeg),
        backend=str(backend),
    )
    return np.asarray(residual, dtype=float)


def _assemble_matrix(*, problem, form, qdeg: int, backend: str) -> np.ndarray:
    matrix, _ = assemble_form(
        Equation(form, None),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=int(qdeg),
        backend=str(backend),
    )
    if sparse.issparse(matrix):
        return matrix.toarray()
    return matrix.to_scipy().toarray()


def _build_problem():
    nodes, elems, _, corners = structured_quad(1.0, 0.25, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(
        mesh,
        field_specs={
            "v_x": 1,
            "v_y": 1,
            "p": 1,
            "vS_x": 1,
            "vS_y": 1,
            "u_x": 1,
            "u_y": 1,
            "alpha": 1,
            "B": 1,
            "mu_alpha": 1,
        },
    )
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
    VS = FunctionSpace("VS", ["vS_x", "vS_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    problem = {
        "dh": dh,
        "dv": VectorTrialFunction(space=V, dof_handler=dh),
        "dp": TrialFunction("p", dof_handler=dh),
        "dvS": VectorTrialFunction(space=VS, dof_handler=dh),
        "du": VectorTrialFunction(space=U, dof_handler=dh),
        "dalpha": TrialFunction("alpha", dof_handler=dh),
        "dB": TrialFunction("B", dof_handler=dh),
        "dmu": TrialFunction("mu_alpha", dof_handler=dh),
        "v_test": VectorTestFunction(space=V, dof_handler=dh),
        "q_test": UflTestFunction("p", dof_handler=dh),
        "vS_test": VectorTestFunction(space=VS, dof_handler=dh),
        "u_test": VectorTestFunction(space=U, dof_handler=dh),
        "alpha_test": UflTestFunction("alpha", dof_handler=dh),
        "B_test": UflTestFunction("B", dof_handler=dh),
        "mu_test": UflTestFunction("mu_alpha", dof_handler=dh),
        "v_k": VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh),
        "p_k": Function("p_k", "p", dof_handler=dh),
        "vS_k": VectorFunction("vS_k", ["vS_x", "vS_y"], dof_handler=dh),
        "u_k": VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh),
        "alpha_k": Function("alpha_k", "alpha", dof_handler=dh),
        "B_k": Function("B_k", "B", dof_handler=dh),
        "mu_k": Function("mu_k", "mu_alpha", dof_handler=dh),
        "v_n": VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh),
        "p_n": Function("p_n", "p", dof_handler=dh),
        "vS_n": VectorFunction("vS_n", ["vS_x", "vS_y"], dof_handler=dh),
        "u_n": VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh),
        "alpha_n": Function("alpha_n", "alpha", dof_handler=dh),
        "B_n": Function("B_n", "B", dof_handler=dh),
        "mu_n": Function("mu_n", "mu_alpha", dof_handler=dh),
    }

    rng = np.random.default_rng(17)
    for key in ("v_k", "v_n", "vS_k", "vS_n", "u_k", "u_n"):
        problem[key].nodal_values[:] = 1.0e-2 * rng.standard_normal(problem[key].nodal_values.shape)
    for key in ("p_k", "p_n", "mu_k", "mu_n"):
        problem[key].nodal_values[:] = 1.0e-2 * rng.standard_normal(problem[key].nodal_values.shape)

    alpha_k = 0.35 + 0.45 * rng.random(problem["alpha_k"].nodal_values.shape)
    alpha_n = 0.35 + 0.45 * rng.random(problem["alpha_n"].nodal_values.shape)
    phi_k = 0.25 + 0.50 * rng.random(problem["B_k"].nodal_values.shape)
    phi_n = 0.25 + 0.50 * rng.random(problem["B_n"].nodal_values.shape)
    problem["alpha_k"].nodal_values[:] = alpha_k
    problem["alpha_n"].nodal_values[:] = alpha_n
    problem["B_k"].nodal_values[:] = alpha_k * (1.0 - phi_k)
    problem["B_n"].nodal_values[:] = alpha_n * (1.0 - phi_n)
    return problem


def _build_forms(problem, *, qdeg: int):
    return build_deformation_only_forms(
        v_k=problem["v_k"],
        p_k=problem["p_k"],
        vS_k=problem["vS_k"],
        u_k=problem["u_k"],
        alpha_k=problem["alpha_k"],
        B_k=problem["B_k"],
        mu_alpha_k=problem["mu_k"],
        v_n=problem["v_n"],
        p_n=problem["p_n"],
        vS_n=problem["vS_n"],
        u_n=problem["u_n"],
        alpha_n=problem["alpha_n"],
        B_n=problem["B_n"],
        mu_alpha_n=problem["mu_n"],
        dv=problem["dv"],
        dp=problem["dp"],
        dvS=problem["dvS"],
        du=problem["du"],
        dalpha=problem["dalpha"],
        dB=problem["dB"],
        dmu_alpha=problem["dmu"],
        v_test=problem["v_test"],
        q_test=problem["q_test"],
        vS_test=problem["vS_test"],
        u_test=problem["u_test"],
        alpha_test=problem["alpha_test"],
        B_test=problem["B_test"],
        mu_alpha_test=problem["mu_test"],
        dx=dx(metadata={"q": int(qdeg)}),
        dt=Constant(0.1),
        theta=1.0,
        rho_f=Constant(1.0),
        mu_f=Constant(0.035),
        mu_b=Constant(0.035),
        kappa_inv=Constant(1.0e5),
        mu_s=Constant(1.67785e5),
        lambda_s=Constant(8.22148e6),
        solid_model="linear",
        kappa_inv_model="refmap",
        M_alpha=1.0,
        gamma_alpha=1.0,
        eps_alpha=0.1,
        support_physics="internal_conversion",
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        fluid_convection="full",
        include_skeleton_acceleration=True,
        rho_s0_tilde=Constant(1.1),
        skeleton_inertia_convection="full",
        skeleton_pressure_mode="whole_domain",
    )


@pytest.mark.parametrize("backend", ("cpp",))
def test_deformation_only_alpha_B_backend_matches_python(monkeypatch, tmp_path, backend):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"jit_cache_{backend}"))
    if backend == "cpp":
        monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")
    else:
        monkeypatch.delenv("PYCUTFEM_JIT_BACKEND", raising=False)

    qdeg = 4
    problem_ref = _build_problem()
    forms_ref = _build_forms(problem_ref, qdeg=qdeg)
    problem = _build_problem()
    forms = _build_forms(problem, qdeg=qdeg)

    assert forms_ref.r_B is not None
    assert forms_ref.a_B is not None

    A_ref = _assemble_matrix(problem=problem_ref, form=forms_ref.jacobian_form, qdeg=qdeg, backend="python")
    R_ref = _assemble_vector(problem=problem_ref, form=forms_ref.residual_form, qdeg=qdeg, backend="python")

    A = _assemble_matrix(problem=problem, form=forms.jacobian_form, qdeg=qdeg, backend=backend)
    R = _assemble_vector(problem=problem, form=forms.residual_form, qdeg=qdeg, backend=backend)

    np.testing.assert_allclose(A, A_ref, rtol=1.0e-9, atol=1.0e-9)
    np.testing.assert_allclose(R, R_ref, rtol=1.0e-9, atol=1.0e-9)
