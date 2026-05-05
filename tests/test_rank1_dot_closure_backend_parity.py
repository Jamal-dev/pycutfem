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
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    dot,
    grad,
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


def _build_rank1_problem():
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
    u_k.components[0].nodal_values[:] = 0.5 + 0.8 * coords_ux[:, 0] - 0.2 * coords_ux[:, 1]
    u_k.components[1].nodal_values[:] = -0.3 + 0.4 * coords_uy[:, 0] + 0.9 * coords_uy[:, 1]
    p_k.nodal_values[:] = 0.2 + 1.1 * coords_p[:, 0] - 0.7 * coords_p[:, 1]

    A_arr = np.array([[1.2, -0.15], [0.25, 0.85]], dtype=float)
    A = Constant(A_arr, dim=2)
    AT = Constant(A_arr.T, dim=2)
    c = Constant(np.array([0.4, -0.6], dtype=float), dim=1)
    dΩ = dx(metadata={"q": 6})
    return {
        "dh": dh,
        "du": du,
        "v": v,
        "dp": dp,
        "q": q,
        "u_k": u_k,
        "p_k": p_k,
        "A": A,
        "AT": AT,
        "c": c,
        "dx": dΩ,
    }


def _assemble_matrix(form, dh, backend):
    K, _ = assemble_form(Equation(form, None), dof_handler=dh, bcs=[], backend=backend)
    return np.asarray(K.toarray(), dtype=float)


def _assemble_vector(form, dh, backend):
    _, rhs = assemble_form(Equation(None, form), dof_handler=dh, bcs=[], backend=backend)
    return np.asarray(rhs, dtype=float).reshape(-1)


def _rank1_test_exprs(ctx):
    return {
        "grad_q": grad(ctx["q"]),
        "AT_grad_q": dot(ctx["AT"], grad(ctx["q"])),
        "grad_q_A": dot(grad(ctx["q"]), ctx["A"]),
        "v": ctx["v"],
        "A_v": dot(ctx["A"], ctx["v"]),
        "v_AT": dot(ctx["v"], ctx["AT"]),
    }


def _rank1_trial_exprs(ctx):
    return {
        "grad_dp": grad(ctx["dp"]),
        "AT_grad_dp": dot(ctx["AT"], grad(ctx["dp"])),
        "grad_dp_A": dot(grad(ctx["dp"]), ctx["A"]),
        "du": ctx["du"],
        "A_du": dot(ctx["A"], ctx["du"]),
        "du_AT": dot(ctx["du"], ctx["AT"]),
    }


def _rank1_value_exprs(ctx):
    return {
        "c": ctx["c"],
        "A_c": dot(ctx["A"], ctx["c"]),
        "c_AT": dot(ctx["c"], ctx["AT"]),
        "u_k": ctx["u_k"],
        "A_u_k": dot(ctx["A"], ctx["u_k"]),
        "u_k_AT": dot(ctx["u_k"], ctx["AT"]),
        "grad_p_k": grad(ctx["p_k"]),
        "AT_grad_p_k": dot(ctx["AT"], grad(ctx["p_k"])),
        "grad_p_k_A": dot(grad(ctx["p_k"]), ctx["A"]),
    }


def _mixed_rank1_exprs(ctx):
    return {
        "grad_v_dot_du": dot(grad(ctx["v"]), ctx["du"]),
        "grad_v_A_dot_du": dot(dot(grad(ctx["v"]), ctx["A"]), ctx["du"]),
        "v_dot_grad_du": dot(ctx["v"], grad(ctx["du"])),
        "v_dot_A_grad_du": dot(ctx["v"], dot(ctx["A"], grad(ctx["du"]))),
    }


def _assert_allclose_forms(forms, dh, backend, assemble_fn, *, rtol=1.0e-11, atol=1.0e-11):
    failures: list[str] = []
    for name, form in forms:
        expected = assemble_fn(form, dh, "python")
        got = assemble_fn(form, dh, backend)
        try:
            np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")
    assert not failures, "\n".join(failures)


def _combine_linear_terms(terms, coeffs):
    expr = coeffs[0] * terms[0]
    for coeff, term in zip(coeffs[1:], terms[1:], strict=True):
        expr = expr + coeff * term
    return expr


def _expand_dot_product(lhs_terms, lhs_coeffs, rhs_terms, rhs_coeffs):
    expanded = None
    for lhs_coeff, lhs_term in zip(lhs_coeffs, lhs_terms, strict=True):
        for rhs_coeff, rhs_term in zip(rhs_coeffs, rhs_terms, strict=True):
            term = lhs_coeff * rhs_coeff * dot(lhs_term, rhs_term)
            expanded = term if expanded is None else expanded + term
    return expanded


def _assert_combined_matches_expanded_and_python(
    name,
    combined_form,
    expanded_form,
    dh,
    backend,
    assemble_fn,
    *,
    rtol=1.0e-11,
    atol=1.0e-11,
):
    combined_python = assemble_fn(combined_form, dh, "python")
    combined_backend = assemble_fn(combined_form, dh, backend)
    expanded_backend = assemble_fn(expanded_form, dh, backend)
    try:
        np.testing.assert_allclose(combined_backend, combined_python, rtol=rtol, atol=atol)
        np.testing.assert_allclose(combined_backend, expanded_backend, rtol=rtol, atol=atol)
    except AssertionError as exc:
        raise AssertionError(f"{name}: {exc}") from exc


def _rank1_linear_families(ctx):
    test_exprs = _rank1_test_exprs(ctx)
    trial_exprs = _rank1_trial_exprs(ctx)
    value_exprs = _rank1_value_exprs(ctx)
    mixed_exprs = _mixed_rank1_exprs(ctx)
    return {
        "test_component": [
            test_exprs["v"],
            test_exprs["A_v"],
            test_exprs["v_AT"],
        ],
        "test_derivative": [
            test_exprs["grad_q"],
            test_exprs["AT_grad_q"],
            test_exprs["grad_q_A"],
        ],
        "trial_component": [
            trial_exprs["du"],
            trial_exprs["A_du"],
            trial_exprs["du_AT"],
        ],
        "trial_derivative": [
            trial_exprs["grad_dp"],
            trial_exprs["AT_grad_dp"],
            trial_exprs["grad_dp_A"],
        ],
        "value_component": [
            value_exprs["c"],
            value_exprs["A_c"],
            value_exprs["c_AT"],
            value_exprs["u_k"],
            value_exprs["A_u_k"],
            value_exprs["u_k_AT"],
        ],
        "value_derivative": [
            value_exprs["grad_p_k"],
            value_exprs["AT_grad_p_k"],
            value_exprs["grad_p_k_A"],
        ],
        "mixed_component": [
            mixed_exprs["grad_v_dot_du"],
            mixed_exprs["grad_v_A_dot_du"],
        ],
        "mixed_derivative": [
            mixed_exprs["v_dot_grad_du"],
            mixed_exprs["v_dot_A_grad_du"],
        ],
    }


def _linear_coeffs(values):
    return [Constant(float(value)) for value in values]


@pytest.mark.parametrize("backend", _compiled_backends())
def test_rank1_dot_residual_closure_matches_python(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_rank1_residual_{backend}"))

    ctx = _build_rank1_problem()
    test_exprs = _rank1_test_exprs(ctx)
    value_exprs = _rank1_value_exprs(ctx)
    forms = []

    for test_name, test_expr in test_exprs.items():
        for value_name, value_expr in value_exprs.items():
            forms.append((f"{test_name}·{value_name}", dot(test_expr, value_expr) * ctx["dx"]))
            forms.append((f"{value_name}·{test_name}", dot(value_expr, test_expr) * ctx["dx"]))

    _assert_allclose_forms(forms, ctx["dh"], backend, _assemble_vector)


@pytest.mark.parametrize("backend", _compiled_backends())
def test_rank1_dot_bilinear_closure_matches_python(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_rank1_bilinear_{backend}"))

    ctx = _build_rank1_problem()
    test_exprs = _rank1_test_exprs(ctx)
    trial_exprs = _rank1_trial_exprs(ctx)
    forms = []

    for test_name, test_expr in test_exprs.items():
        for trial_name, trial_expr in trial_exprs.items():
            forms.append((f"{test_name}·{trial_name}", dot(test_expr, trial_expr) * ctx["dx"]))
            forms.append((f"{trial_name}·{test_name}", dot(trial_expr, test_expr) * ctx["dx"]))

    _assert_allclose_forms(forms, ctx["dh"], backend, _assemble_matrix)


@pytest.mark.parametrize("backend", _compiled_backends())
def test_mixed_rank1_dot_value_closure_matches_python(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_mixed_rank1_{backend}"))

    ctx = _build_rank1_problem()
    mixed_exprs = _mixed_rank1_exprs(ctx)
    value_exprs = _rank1_value_exprs(ctx)
    forms = []

    for mixed_name, mixed_expr in mixed_exprs.items():
        for value_name, value_expr in value_exprs.items():
            forms.append((f"{mixed_name}·{value_name}", dot(mixed_expr, value_expr) * ctx["dx"]))
            forms.append((f"{value_name}·{mixed_name}", dot(value_expr, mixed_expr) * ctx["dx"]))

    _assert_allclose_forms(forms, ctx["dh"], backend, _assemble_matrix)


@pytest.mark.parametrize("backend", _compiled_backends())
def test_rank1_linear_closure_matches_python_and_expansion(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_rank1_linear_closure_{backend}"))

    ctx = _build_rank1_problem()
    families = _rank1_linear_families(ctx)

    cases = [
        (
            "residual_component",
            families["test_component"],
            _linear_coeffs([1.2, -0.65, 0.45]),
            families["value_component"],
            _linear_coeffs([0.9, -0.5, 0.35, -1.1, 0.75, -0.4]),
            _assemble_vector,
        ),
        (
            "residual_derivative",
            families["test_derivative"],
            _linear_coeffs([1.1, -0.55, 0.8]),
            families["value_derivative"],
            _linear_coeffs([0.95, -1.2, 0.6]),
            _assemble_vector,
        ),
        (
            "bilinear_component",
            families["test_component"],
            _linear_coeffs([1.05, -0.7, 0.5]),
            families["trial_component"],
            _linear_coeffs([0.85, -1.15, 0.65]),
            _assemble_matrix,
        ),
        (
            "bilinear_derivative",
            families["test_derivative"],
            _linear_coeffs([1.15, -0.6, 0.75]),
            families["trial_derivative"],
            _linear_coeffs([0.9, -1.05, 0.55]),
            _assemble_matrix,
        ),
        (
            "mixed_component",
            families["mixed_component"],
            _linear_coeffs([1.25, -0.85]),
            families["value_component"],
            _linear_coeffs([0.8, -0.45, 0.3, -1.0, 0.55, -0.7]),
            _assemble_matrix,
        ),
        (
            "mixed_derivative",
            families["mixed_derivative"],
            _linear_coeffs([-1.1, 0.9]),
            families["value_derivative"],
            _linear_coeffs([0.7, -1.3, 0.5]),
            _assemble_matrix,
        ),
    ]

    for name, lhs_terms, lhs_coeffs, rhs_terms, rhs_coeffs, assemble_fn in cases:
        lhs = _combine_linear_terms(lhs_terms, lhs_coeffs)
        rhs = _combine_linear_terms(rhs_terms, rhs_coeffs)

        combined_lr = dot(lhs, rhs) * ctx["dx"]
        expanded_lr = _expand_dot_product(lhs_terms, lhs_coeffs, rhs_terms, rhs_coeffs) * ctx["dx"]
        _assert_combined_matches_expanded_and_python(
            f"{name}: lhs·rhs",
            combined_lr,
            expanded_lr,
            ctx["dh"],
            backend,
            assemble_fn,
        )

        combined_rl = dot(rhs, lhs) * ctx["dx"]
        expanded_rl = _expand_dot_product(rhs_terms, rhs_coeffs, lhs_terms, lhs_coeffs) * ctx["dx"]
        _assert_combined_matches_expanded_and_python(
            f"{name}: rhs·lhs",
            combined_rl,
            expanded_rl,
            ctx["dh"],
            backend,
            assemble_fn,
        )
