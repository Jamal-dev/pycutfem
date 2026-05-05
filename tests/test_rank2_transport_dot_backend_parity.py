import numpy as np
import os
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction,
    Transpose,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    dot,
    grad,
    inner,
    inv,
)
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def _compiled_backends() -> list[str]:
    out = ["jit"]
    try:
        from pycutfem.jit.cpp_backend import compile_backend_cpp  # noqa: F401
    except Exception:
        available = out
    else:
        available = out + ["cpp"]

    requested = os.environ.get("PYCUTFEM_TEST_BACKENDS", "").strip()
    if not requested:
        return available
    wanted = [backend.strip() for backend in requested.split(",") if backend.strip()]
    return [backend for backend in available if backend in wanted]


def _build_problem():
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    me = MixedElement(mesh, field_specs={"ux": 2, "uy": 2, "p": 2})
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["ux", "uy"], dim=1)
    du = VectorTrialFunction(space=V, dof_handler=dh)
    v = VectorTestFunction(space=V, dof_handler=dh)
    dp = TrialFunction("p", dof_handler=dh)
    q = TestFunction("p", dof_handler=dh)
    u_k = VectorFunction(name="u_k", field_names=["ux", "uy"], dof_handler=dh)
    p_k = Function(name="p_k", field_name="p", dof_handler=dh)

    coords_ux = np.asarray(dh.get_dof_coords("ux"), dtype=float)
    coords_uy = np.asarray(dh.get_dof_coords("uy"), dtype=float)
    coords_p = np.asarray(dh.get_dof_coords("p"), dtype=float)
    u_k.components[0].nodal_values[:] = 0.35 + 0.65 * coords_ux[:, 0] - 0.15 * coords_ux[:, 1]
    u_k.components[1].nodal_values[:] = -0.2 + 0.25 * coords_uy[:, 0] + 0.55 * coords_uy[:, 1]
    p_k.nodal_values[:] = 0.1 + 0.8 * coords_p[:, 0] - 0.45 * coords_p[:, 1]

    F = Constant(np.array([[1.15, -0.2], [0.35, 0.9]], dtype=float), dim=2)
    B = Constant(np.array([[0.7, 0.1], [-0.25, 1.05]], dtype=float), dim=2)
    c = Constant(np.array([0.45, -0.35], dtype=float), dim=1)
    dΩ = dx(metadata={"q": 6})

    return {"dh": dh, "du": du, "v": v, "dp": dp, "q": q, "u_k": u_k, "p_k": p_k, "F": F, "B": B, "c": c, "dx": dΩ}


def _assemble_matrix(form, dh, backend):
    K, _ = assemble_form(Equation(form, None), dof_handler=dh, bcs=[], backend=backend)
    return np.asarray(K.toarray(), dtype=float)


def _assemble_vector(form, dh, backend):
    _, rhs = assemble_form(Equation(None, form), dof_handler=dh, bcs=[], backend=backend)
    return np.asarray(rhs, dtype=float).reshape(-1)


def _test_rank2_exprs(ctx):
    grad_v = grad(ctx["v"])
    return {
        "grad(v)·F": dot(grad_v, ctx["F"]),
        "F·grad(v)": dot(ctx["F"], grad_v),
        "grad(v)^T·F": dot(Transpose(grad_v), ctx["F"]),
        "F·grad(v)^T": dot(ctx["F"], Transpose(grad_v)),
    }


def _trial_rank2_exprs(ctx):
    grad_du = grad(ctx["du"])
    return {
        "grad(du)·F": dot(grad_du, ctx["F"]),
        "F·grad(du)": dot(ctx["F"], grad_du),
        "grad(du)^T·F": dot(Transpose(grad_du), ctx["F"]),
        "F·grad(du)^T": dot(ctx["F"], Transpose(grad_du)),
    }


@pytest.mark.parametrize("backend", _compiled_backends())
def test_rank2_transport_dot_matrix_closure_matches_python(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_rank2_transport_dot_mat_{backend}"))

    ctx = _build_problem()
    form = dot(dot(dot(grad(ctx["v"]), ctx["F"]), ctx["du"]), ctx["c"]) * ctx["dx"]

    expected = _assemble_matrix(form, ctx["dh"], "python")
    got = _assemble_matrix(form, ctx["dh"], backend)

    np.testing.assert_allclose(got, expected, rtol=1.0e-11, atol=1.0e-11)


@pytest.mark.parametrize("rank2_name", ("grad(v)·F", "F·grad(v)", "grad(v)^T·F", "F·grad(v)^T"))
@pytest.mark.parametrize("backend", _compiled_backends())
def test_rank2_transport_dot_test_side_family_matches_python(rank2_name, backend, tmp_path, monkeypatch):
    monkeypatch.setenv(
        "PYCUTFEM_CACHE_DIR",
        str(tmp_path / f"pycutfem_rank2_transport_test_family_{backend}_{rank2_name.replace(' ', '_')}"),
    )

    ctx = _build_problem()
    rank2_expr = _test_rank2_exprs(ctx)[rank2_name]
    form = dot(dot(rank2_expr, ctx["du"]), ctx["c"]) * ctx["dx"]

    np.testing.assert_allclose(
        _assemble_matrix(form, ctx["dh"], backend),
        _assemble_matrix(form, ctx["dh"], "python"),
        rtol=1.0e-11,
        atol=1.0e-11,
    )


@pytest.mark.parametrize("backend", _compiled_backends())
def test_rank2_transport_dot_matrix_closure_left_vector_matches_python(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_rank2_transport_dot_left_{backend}"))

    ctx = _build_problem()
    form = dot(ctx["c"], dot(ctx["v"], dot(ctx["F"], grad(ctx["du"])))) * ctx["dx"]

    expected = _assemble_matrix(form, ctx["dh"], "python")
    got = _assemble_matrix(form, ctx["dh"], backend)

    np.testing.assert_allclose(got, expected, rtol=1.0e-11, atol=1.0e-11)


@pytest.mark.parametrize("rank2_name", ("grad(du)·F", "F·grad(du)", "grad(du)^T·F", "F·grad(du)^T"))
@pytest.mark.parametrize("backend", _compiled_backends())
def test_rank2_transport_dot_trial_side_family_matches_python(rank2_name, backend, tmp_path, monkeypatch):
    monkeypatch.setenv(
        "PYCUTFEM_CACHE_DIR",
        str(tmp_path / f"pycutfem_rank2_transport_trial_family_{backend}_{rank2_name.replace(' ', '_')}"),
    )

    ctx = _build_problem()
    rank2_expr = _trial_rank2_exprs(ctx)[rank2_name]
    form = dot(ctx["c"], dot(ctx["v"], rank2_expr)) * ctx["dx"]

    np.testing.assert_allclose(
        _assemble_matrix(form, ctx["dh"], backend),
        _assemble_matrix(form, ctx["dh"], "python"),
        rtol=1.0e-11,
        atol=1.0e-11,
    )


@pytest.mark.parametrize("backend", _compiled_backends())
def test_scalar_basis_matrix_products_inside_inner_match_python(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_scalar_basis_matrix_inner_{backend}"))

    ctx = _build_problem()
    left_form = inner(ctx["F"], ctx["q"] * ctx["B"]) * ctx["dx"]
    right_form = inner(ctx["F"], ctx["B"] * ctx["q"]) * ctx["dx"]

    np.testing.assert_allclose(
        _assemble_vector(left_form, ctx["dh"], backend),
        _assemble_vector(left_form, ctx["dh"], "python"),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    np.testing.assert_allclose(
        _assemble_vector(right_form, ctx["dh"], backend),
        _assemble_vector(right_form, ctx["dh"], "python"),
        rtol=1.0e-11,
        atol=1.0e-11,
    )


def _vector_value_exprs(ctx):
    return {
        "const": ctx["c"],
        "function": ctx["u_k"],
    }


_VECTOR_BASIS_INNER_FORM_NAMES = (
    "trial grad · value -> inner with test",
    "value · trial grad -> inner with test",
    "test grad · value -> inner with trial",
    "value · test grad -> inner with trial",
)


def _vector_basis_inner_forms(ctx, vec_expr):
    return {
        "trial grad · value -> inner with test": inner(dot(grad(ctx["du"]), vec_expr), ctx["v"]) * ctx["dx"],
        "value · trial grad -> inner with test": inner(dot(vec_expr, grad(ctx["du"])), ctx["v"]) * ctx["dx"],
        "test grad · value -> inner with trial": inner(dot(grad(ctx["v"]), vec_expr), ctx["du"]) * ctx["dx"],
        "value · test grad -> inner with trial": inner(dot(vec_expr, grad(ctx["v"])), ctx["du"]) * ctx["dx"],
    }


_VECTOR_VALUE_INNER_FORM_NAMES = (
    "grad(function) · const -> inner with test",
    "const · grad(function) -> inner with test",
    "grad(function) · function -> inner with test",
    "function · grad(function) -> inner with test",
)


def _vector_value_inner_forms(ctx):
    return {
        "grad(function) · const -> inner with test": inner(dot(grad(ctx["u_k"]), ctx["c"]), ctx["v"]) * ctx["dx"],
        "const · grad(function) -> inner with test": inner(dot(ctx["c"], grad(ctx["u_k"])), ctx["v"]) * ctx["dx"],
        "grad(function) · function -> inner with test": inner(dot(grad(ctx["u_k"]), ctx["u_k"]), ctx["v"]) * ctx["dx"],
        "function · grad(function) -> inner with test": inner(dot(ctx["u_k"], grad(ctx["u_k"])), ctx["v"]) * ctx["dx"],
    }


_TRANSPORTED_BASIS_INNER_FORM_NAMES = (
    "inner(grad(v)·c, grad(du)·c)",
    "inner(c·grad(v), c·grad(du))",
)


def _transported_basis_inner_forms(ctx):
    return {
        "inner(grad(v)·c, grad(du)·c)": inner(dot(grad(ctx["v"]), ctx["c"]), dot(grad(ctx["du"]), ctx["c"])) * ctx["dx"],
        "inner(c·grad(v), c·grad(du))": inner(dot(ctx["c"], grad(ctx["v"])), dot(ctx["c"], grad(ctx["du"]))) * ctx["dx"],
    }


def _cache_token(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)


@pytest.mark.parametrize("vector_name", ("const", "function"))
@pytest.mark.parametrize("form_name", _VECTOR_BASIS_INNER_FORM_NAMES)
@pytest.mark.parametrize("backend", _compiled_backends())
def test_vector_grad_dot_inner_basis_closure_matches_python(vector_name, form_name, backend, tmp_path, monkeypatch):
    cache_key = f"pycutfem_vector_grad_inner_basis_{backend}_{vector_name}_{_cache_token(form_name)}"
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / cache_key))

    ctx = _build_problem()
    vec_expr = _vector_value_exprs(ctx)[vector_name]
    form = _vector_basis_inner_forms(ctx, vec_expr)[form_name]

    np.testing.assert_allclose(
        _assemble_matrix(form, ctx["dh"], backend),
        _assemble_matrix(form, ctx["dh"], "python"),
        rtol=1.0e-11,
        atol=1.0e-11,
    )


@pytest.mark.parametrize("form_name", _VECTOR_VALUE_INNER_FORM_NAMES)
@pytest.mark.parametrize("backend", _compiled_backends())
def test_vector_grad_dot_inner_value_closure_matches_python(form_name, backend, tmp_path, monkeypatch):
    cache_key = f"pycutfem_vector_grad_inner_value_{backend}_{_cache_token(form_name)}"
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / cache_key))

    ctx = _build_problem()
    form = _vector_value_inner_forms(ctx)[form_name]

    np.testing.assert_allclose(
        _assemble_vector(form, ctx["dh"], backend),
        _assemble_vector(form, ctx["dh"], "python"),
        rtol=1.0e-11,
        atol=1.0e-11,
    )


@pytest.mark.parametrize("form_name", _TRANSPORTED_BASIS_INNER_FORM_NAMES)
@pytest.mark.parametrize("backend", _compiled_backends())
def test_transported_vector_basis_inner_matches_python(form_name, backend, tmp_path, monkeypatch):
    cache_key = f"pycutfem_transported_basis_inner_{backend}_{_cache_token(form_name)}"
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / cache_key))

    ctx = _build_problem()
    form = _transported_basis_inner_forms(ctx)[form_name]

    np.testing.assert_allclose(
        _assemble_matrix(form, ctx["dh"], backend),
        _assemble_matrix(form, ctx["dh"], "python"),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
