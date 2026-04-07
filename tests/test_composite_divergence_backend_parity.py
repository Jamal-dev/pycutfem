import numpy as np
import pytest


def _compiled_backends() -> list[str]:
    out = ["jit"]
    try:
        from pycutfem.jit.cpp_backend import compile_backend_cpp  # noqa: F401
    except Exception:
        return out
    out.append("cpp")
    return out


def _build_problem():
    from pycutfem.core.dofhandler import DofHandler
    from pycutfem.core.mesh import Mesh
    from pycutfem.fem.mixedelement import MixedElement
    from pycutfem.ufl.expressions import Function, VectorFunction
    from pycutfem.ufl.spaces import FunctionSpace
    from pycutfem.utils.meshgen import structured_quad

    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    me = MixedElement(
        mesh,
        field_specs={
            "u_x": 2,
            "u_y": 2,
            "phi": 1,
            "psi": 1,
        },
    )
    dh = DofHandler(me, method="cg")

    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    psi_k = Function("psi_k", "psi", dof_handler=dh)

    coords_ux = np.asarray(dh.get_dof_coords("u_x"), dtype=float)
    coords_uy = np.asarray(dh.get_dof_coords("u_y"), dtype=float)
    coords_phi = np.asarray(dh.get_dof_coords("phi"), dtype=float)
    coords_psi = np.asarray(dh.get_dof_coords("psi"), dtype=float)

    u_k.components[0].nodal_values[:] = 0.15 + 0.25 * coords_ux[:, 0] - 0.08 * coords_ux[:, 1]
    u_k.components[1].nodal_values[:] = -0.10 + 0.05 * coords_uy[:, 0] + 0.18 * coords_uy[:, 1]
    phi_k.nodal_values[:] = 0.55 - 0.12 * coords_phi[:, 0] + 0.07 * coords_phi[:, 1]
    psi_k.nodal_values[:] = 0.20 + 0.10 * coords_psi[:, 0] - 0.04 * coords_psi[:, 1]

    return dh, U, u_k, phi_k, psi_k


def _assemble_matrix(form, *, dh, backend: str):
    from pycutfem.ufl.forms import Equation, assemble_form

    K, F = assemble_form(Equation(form, None), dof_handler=dh, bcs=[], quad_order=6, backend=backend)
    return np.asarray(K.toarray(), dtype=float), np.asarray(F, dtype=float).reshape(-1)


def _assemble_rhs(form, *, dh, backend: str):
    from pycutfem.ufl.forms import Equation, assemble_form

    _, F = assemble_form(Equation(None, form), dof_handler=dh, bcs=[], quad_order=6, backend=backend)
    return np.asarray(F, dtype=float).reshape(-1)


@pytest.mark.parametrize("backend", _compiled_backends())
@pytest.mark.parametrize(
    "case",
    (
        "function_trial",
        "trial_function",
        "test_function",
        "constant_test",
    ),
)
def test_div_scalar_vector_product_identity_matches_python(backend, case, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_div_product_{backend}_{case}"))

    from pycutfem.ufl.expressions import Constant, TestFunction, TrialFunction, VectorTestFunction, VectorTrialFunction, div, dot, grad
    from pycutfem.ufl.measures import dx

    dh, U, u_k, phi_k, _psi_k = _build_problem()

    du = VectorTrialFunction(U, dof_handler=dh)
    w = VectorTestFunction(U, dof_handler=dh)
    dphi = TrialFunction("phi", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    psi_trial = TrialFunction("psi", dof_handler=dh)
    psi_test = TestFunction("psi", dof_handler=dh)
    phi_c = Constant(1.75)

    dΩ = dx(metadata={"q": 6})
    if case == "function_trial":
        compact = psi_test * div(phi_k * du) * dΩ
        expanded = psi_test * (dot(du, grad(phi_k)) + phi_k * div(du)) * dΩ
    elif case == "trial_function":
        compact = psi_test * div(dphi * u_k) * dΩ
        expanded = psi_test * (dot(u_k, grad(dphi)) + dphi * div(u_k)) * dΩ
    elif case == "test_function":
        compact = psi_trial * div(phi_test * u_k) * dΩ
        expanded = psi_trial * (dot(u_k, grad(phi_test)) + phi_test * div(u_k)) * dΩ
    elif case == "constant_test":
        compact = psi_trial * div(phi_c * w) * dΩ
        expanded = psi_trial * (dot(w, grad(phi_c)) + phi_c * div(w)) * dΩ
    else:  # pragma: no cover
        raise AssertionError(f"Unhandled case {case!r}.")

    A_compact_py, F_compact_py = _assemble_matrix(compact, dh=dh, backend="python")
    A_expanded_py, F_expanded_py = _assemble_matrix(expanded, dh=dh, backend="python")
    A_compact_backend, F_compact_backend = _assemble_matrix(compact, dh=dh, backend=backend)
    A_expanded_backend, F_expanded_backend = _assemble_matrix(expanded, dh=dh, backend=backend)

    np.testing.assert_allclose(A_expanded_py, A_compact_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(F_expanded_py, F_compact_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(A_compact_backend, A_compact_py, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(F_compact_backend, F_compact_py, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(A_expanded_backend, A_compact_py, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(F_expanded_backend, F_compact_py, rtol=1.0e-10, atol=1.0e-10)


@pytest.mark.parametrize("backend", _compiled_backends())
@pytest.mark.parametrize("case", ("trial_function", "function_trial"))
def test_div_dyad_bilinear_identity_matches_python(backend, case, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_div_dyad_{backend}_{case}"))

    from pycutfem.ufl.expressions import VectorTestFunction, VectorTrialFunction, div, dot, dyad, grad, inner
    from pycutfem.ufl.measures import dx

    dh, U, u_k, _phi_k, _psi_k = _build_problem()

    du = VectorTrialFunction(U, dof_handler=dh)
    w = VectorTestFunction(U, dof_handler=dh)

    dΩ = dx(metadata={"q": 6})
    if case == "trial_function":
        compact = inner(div(dyad(du, u_k)), w) * dΩ
        expanded = inner(u_k * div(du) + dot(grad(u_k), du), w) * dΩ
    elif case == "function_trial":
        compact = inner(div(dyad(u_k, du)), w) * dΩ
        expanded = inner(du * div(u_k) + dot(grad(du), u_k), w) * dΩ
    else:  # pragma: no cover
        raise AssertionError(f"Unhandled case {case!r}.")

    A_compact_py, F_compact_py = _assemble_matrix(compact, dh=dh, backend="python")
    A_expanded_py, F_expanded_py = _assemble_matrix(expanded, dh=dh, backend="python")
    A_compact_backend, F_compact_backend = _assemble_matrix(compact, dh=dh, backend=backend)
    A_expanded_backend, F_expanded_backend = _assemble_matrix(expanded, dh=dh, backend=backend)

    np.testing.assert_allclose(A_expanded_py, A_compact_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(F_expanded_py, F_compact_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(A_compact_backend, A_compact_py, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(F_compact_backend, F_compact_py, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(A_expanded_backend, A_compact_py, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(F_expanded_backend, F_compact_py, rtol=1.0e-10, atol=1.0e-10)


@pytest.mark.parametrize("backend", _compiled_backends())
@pytest.mark.parametrize("case", ("constant_first", "constant_second"))
def test_div_dyad_constant_identities_match_python(backend, case, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_div_dyad_const_{backend}_{case}"))

    from pycutfem.ufl.expressions import Constant, VectorTestFunction, div, dot, dyad, grad, inner
    from pycutfem.ufl.measures import dx

    dh, U, u_k, _phi_k, _psi_k = _build_problem()

    w = VectorTestFunction(U, dof_handler=dh)
    beta = Constant(np.array([0.6, -0.2], dtype=float))

    dΩ = dx(metadata={"q": 6})
    if case == "constant_first":
        compact = inner(div(dyad(beta, u_k)), w) * dΩ
        expanded = inner(u_k * div(beta) + dot(grad(u_k), beta), w) * dΩ
    elif case == "constant_second":
        compact = inner(div(dyad(u_k, beta)), w) * dΩ
        expanded = inner(beta * div(u_k) + dot(grad(beta), u_k), w) * dΩ
    else:  # pragma: no cover
        raise AssertionError(f"Unhandled case {case!r}.")

    F_compact_py = _assemble_rhs(compact, dh=dh, backend="python")
    F_expanded_py = _assemble_rhs(expanded, dh=dh, backend="python")
    F_compact_backend = _assemble_rhs(compact, dh=dh, backend=backend)
    F_expanded_backend = _assemble_rhs(expanded, dh=dh, backend=backend)

    np.testing.assert_allclose(F_expanded_py, F_compact_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(F_compact_backend, F_compact_py, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(F_expanded_backend, F_compact_py, rtol=1.0e-10, atol=1.0e-10)


@pytest.mark.parametrize("backend", _compiled_backends())
def test_div_grad_equals_laplacian_matches_python(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_div_grad_{backend}"))

    from pycutfem.ufl.expressions import TestFunction, TrialFunction, VectorTestFunction, VectorTrialFunction, div, grad, inner, laplacian
    from pycutfem.ufl.measures import dx

    dh, U, _u_k, _phi_k, _psi_k = _build_problem()

    du = VectorTrialFunction(U, dof_handler=dh)
    w = VectorTestFunction(U, dof_handler=dh)
    dphi = TrialFunction("phi", dof_handler=dh)
    psi_test = TestFunction("psi", dof_handler=dh)

    dΩ = dx(metadata={"q": 6})
    scalar_compact = psi_test * div(grad(dphi)) * dΩ
    scalar_expanded = psi_test * laplacian(dphi) * dΩ
    vector_compact = inner(div(grad(du)), w) * dΩ
    vector_expanded = inner(laplacian(du), w) * dΩ

    A_scalar_py, F_scalar_py = _assemble_matrix(scalar_compact, dh=dh, backend="python")
    A_scalar_expanded_py, F_scalar_expanded_py = _assemble_matrix(scalar_expanded, dh=dh, backend="python")
    A_scalar_backend, F_scalar_backend = _assemble_matrix(scalar_compact, dh=dh, backend=backend)
    A_scalar_expanded_backend, F_scalar_expanded_backend = _assemble_matrix(scalar_expanded, dh=dh, backend=backend)

    np.testing.assert_allclose(A_scalar_expanded_py, A_scalar_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(F_scalar_expanded_py, F_scalar_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(A_scalar_backend, A_scalar_py, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(F_scalar_backend, F_scalar_py, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(A_scalar_expanded_backend, A_scalar_py, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(F_scalar_expanded_backend, F_scalar_py, rtol=1.0e-10, atol=1.0e-10)

    A_vector_py, F_vector_py = _assemble_matrix(vector_compact, dh=dh, backend="python")
    A_vector_expanded_py, F_vector_expanded_py = _assemble_matrix(vector_expanded, dh=dh, backend="python")
    A_vector_backend, F_vector_backend = _assemble_matrix(vector_compact, dh=dh, backend=backend)
    A_vector_expanded_backend, F_vector_expanded_backend = _assemble_matrix(vector_expanded, dh=dh, backend=backend)

    np.testing.assert_allclose(A_vector_expanded_py, A_vector_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(F_vector_expanded_py, F_vector_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(A_vector_backend, A_vector_py, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(F_vector_backend, F_vector_py, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(A_vector_expanded_backend, A_vector_py, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(F_vector_expanded_backend, F_vector_py, rtol=1.0e-10, atol=1.0e-10)
