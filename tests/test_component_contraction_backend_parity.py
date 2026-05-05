import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant, VectorFunction, VectorTestFunction, VectorTrialFunction, dot, grad, inner
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def _compiled_backends() -> list[str]:
    out = ["jit"]
    try:
        from pycutfem.jit.cpp_backend import compile_backend_cpp  # noqa: F401
    except Exception:
        return out
    out.append("cpp")
    return out


def _build_vector_problem():
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    me = MixedElement(mesh, field_specs={"ux": 2, "uy": 2})
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["ux", "uy"], dim=1)
    du = VectorTrialFunction(space=V, dof_handler=dh)
    v = VectorTestFunction(space=V, dof_handler=dh)
    u_k = VectorFunction(name="u_k", field_names=["ux", "uy"], dof_handler=dh)

    coords_x = np.asarray(dh.get_dof_coords("ux"), dtype=float)
    coords_y = np.asarray(dh.get_dof_coords("uy"), dtype=float)
    u_k.components[0].nodal_values[:] = 1.0 + coords_x[:, 0] + 0.25 * coords_x[:, 1]
    u_k.components[1].nodal_values[:] = -0.5 + 0.75 * coords_y[:, 0] + 2.0 * coords_y[:, 1]

    B = Constant(np.array([[2.0, -0.5], [1.0, 3.0]], dtype=float), dim=2)
    dΩ = dx(metadata={"q": 6})
    return dh, du, v, u_k, B, dΩ


def _assemble_matrix(form, dh, backend):
    K, _ = assemble_form(Equation(form, None), dof_handler=dh, bcs=[], backend=backend)
    return K.toarray()


def _assemble_vector(form, dh, backend):
    _, rhs = assemble_form(Equation(None, form), dof_handler=dh, bcs=[], backend=backend)
    return np.asarray(rhs, dtype=float).reshape(-1)


@pytest.mark.parametrize("backend", _compiled_backends())
def test_grad_vector_row_access_matches_component_gradient(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_component_rows_{backend}"))

    dh, du, v, u_k, _, dΩ = _build_vector_problem()

    bilinear_row0 = inner(grad(du)[0], grad(v)[0]) * dΩ
    bilinear_row1 = inner(grad(du)[1], grad(v)[1]) * dΩ
    bilinear_comp0 = inner(grad(du[0]), grad(v[0])) * dΩ
    bilinear_comp1 = inner(grad(du[1]), grad(v[1])) * dΩ

    residual_row0 = inner(grad(u_k)[0], grad(v)[0]) * dΩ
    residual_row1 = inner(grad(u_k)[1], grad(v)[1]) * dΩ
    residual_comp0 = inner(grad(u_k[0]), grad(v[0])) * dΩ
    residual_comp1 = inner(grad(u_k[1]), grad(v)[1]) * dΩ

    A_row0_py = _assemble_matrix(bilinear_row0, dh, "python")
    A_row1_py = _assemble_matrix(bilinear_row1, dh, "python")
    A_comp0_py = _assemble_matrix(bilinear_comp0, dh, "python")
    A_comp1_py = _assemble_matrix(bilinear_comp1, dh, "python")
    np.testing.assert_allclose(A_row0_py, A_comp0_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(A_row1_py, A_comp1_py, rtol=1.0e-12, atol=1.0e-12)

    A_row0_backend = _assemble_matrix(bilinear_row0, dh, backend)
    A_row1_backend = _assemble_matrix(bilinear_row1, dh, backend)
    np.testing.assert_allclose(A_row0_backend, A_comp0_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(A_row1_backend, A_comp1_py, rtol=1.0e-12, atol=1.0e-12)

    R_row0_py = _assemble_vector(residual_row0, dh, "python")
    R_row1_py = _assemble_vector(residual_row1, dh, "python")
    R_comp0_py = _assemble_vector(residual_comp0, dh, "python")
    R_comp1_py = _assemble_vector(residual_comp1, dh, "python")
    np.testing.assert_allclose(R_row0_py, R_comp0_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(R_row1_py, R_comp1_py, rtol=1.0e-12, atol=1.0e-12)

    R_row0_backend = _assemble_vector(residual_row0, dh, backend)
    R_row1_backend = _assemble_vector(residual_row1, dh, backend)
    np.testing.assert_allclose(R_row0_backend, R_comp0_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(R_row1_backend, R_comp1_py, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("backend", _compiled_backends())
def test_vector_grad_component_expansions_match_compact_dot_forms(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_component_adv_{backend}"))

    dh, du, v, u_k, _, dΩ = _build_vector_problem()

    left_compact = dot(dot(grad(du), u_k), v) * dΩ
    left_components = sum(
        grad(du)[i][j] * u_k[j] * v[i]
        for i in range(2)
        for j in range(2)
    ) * dΩ
    left_components_alt = sum(
        grad(du)[i, j] * u_k[j] * v[i]
        for i in range(2)
        for j in range(2)
    ) * dΩ

    right_compact = dot(dot(u_k, grad(du)), v) * dΩ
    right_components = sum(
        u_k[i] * grad(du)[i][j] * v[j]
        for i in range(2)
        for j in range(2)
    ) * dΩ
    right_components_alt = sum(
        u_k[i] * grad(du)[i, j] * v[j]
        for i in range(2)
        for j in range(2)
    ) * dΩ

    left_residual_compact = dot(dot(grad(u_k), u_k), v) * dΩ
    left_residual_components = sum(
        grad(u_k)[i][j] * u_k[j] * v[i]
        for i in range(2)
        for j in range(2)
    ) * dΩ
    right_residual_compact = dot(dot(u_k, grad(u_k)), v) * dΩ
    right_residual_components = sum(
        u_k[i] * grad(u_k)[i][j] * v[j]
        for i in range(2)
        for j in range(2)
    ) * dΩ

    A_left_py = _assemble_matrix(left_compact, dh, "python")
    A_left_comp_py = _assemble_matrix(left_components, dh, "python")
    A_left_alt_py = _assemble_matrix(left_components_alt, dh, "python")
    A_right_py = _assemble_matrix(right_compact, dh, "python")
    A_right_comp_py = _assemble_matrix(right_components, dh, "python")
    A_right_alt_py = _assemble_matrix(right_components_alt, dh, "python")

    np.testing.assert_allclose(A_left_py, A_left_comp_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(A_left_py, A_left_alt_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(A_right_py, A_right_comp_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(A_right_py, A_right_alt_py, rtol=1.0e-12, atol=1.0e-12)

    np.testing.assert_allclose(_assemble_matrix(left_compact, dh, backend), A_left_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(_assemble_matrix(left_components, dh, backend), A_left_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(_assemble_matrix(left_components_alt, dh, backend), A_left_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(_assemble_matrix(right_compact, dh, backend), A_right_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(_assemble_matrix(right_components, dh, backend), A_right_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(_assemble_matrix(right_components_alt, dh, backend), A_right_py, rtol=1.0e-12, atol=1.0e-12)

    R_left_py = _assemble_vector(left_residual_compact, dh, "python")
    R_left_comp_py = _assemble_vector(left_residual_components, dh, "python")
    R_right_py = _assemble_vector(right_residual_compact, dh, "python")
    R_right_comp_py = _assemble_vector(right_residual_components, dh, "python")

    np.testing.assert_allclose(R_left_py, R_left_comp_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(R_right_py, R_right_comp_py, rtol=1.0e-12, atol=1.0e-12)

    np.testing.assert_allclose(_assemble_vector(left_residual_compact, dh, backend), R_left_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(_assemble_vector(left_residual_components, dh, backend), R_left_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(_assemble_vector(right_residual_compact, dh, backend), R_right_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(_assemble_vector(right_residual_components, dh, backend), R_right_py, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("backend", _compiled_backends())
def test_component_matrix_vector_component_forms_match_python(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_component_matvec_{backend}"))

    dh, _, v, u_k, B, dΩ = _build_vector_problem()

    left_components = sum(
        B[i, j] * u_k[j] * v[i]
        for i in range(2)
        for j in range(2)
    ) * dΩ

    right_components = sum(
        u_k[i] * B[i, j] * v[j]
        for i in range(2)
        for j in range(2)
    ) * dΩ

    R_left_py = _assemble_vector(left_components, dh, "python")
    R_right_py = _assemble_vector(right_components, dh, "python")

    np.testing.assert_allclose(_assemble_vector(left_components, dh, backend), R_left_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(_assemble_vector(right_components, dh, backend), R_right_py, rtol=1.0e-12, atol=1.0e-12)
