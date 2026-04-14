from __future__ import annotations

import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import (
    Constant,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    dot,
    grad,
    inner,
)
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def _available_backends() -> list[str]:
    out = ["jit"]
    try:
        from pycutfem.jit.cpp_backend import compile_backend_cpp  # noqa: F401
    except Exception:
        return out
    out.append("cpp")
    return out


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
    me = MixedElement(mesh, field_specs={"ux": 2, "uy": 2})
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["ux", "uy"], dim=1)
    du = VectorTrialFunction(space=V, dof_handler=dh)
    v = VectorTestFunction(space=V, dof_handler=dh)
    u_k = VectorFunction(name="u_k", field_names=["ux", "uy"], dof_handler=dh)

    coords_ux = np.asarray(dh.get_dof_coords("ux"), dtype=float)
    coords_uy = np.asarray(dh.get_dof_coords("uy"), dtype=float)
    u_k.components[0].nodal_values[:] = 0.5 + 0.8 * coords_ux[:, 0] - 0.2 * coords_ux[:, 1]
    u_k.components[1].nodal_values[:] = -0.3 + 0.4 * coords_uy[:, 0] + 0.9 * coords_uy[:, 1]

    c = Constant(np.array([0.4, -0.6], dtype=float), dim=1)
    A = Constant(np.array([[1.2, -0.15], [0.25, 0.85]], dtype=float), dim=2)
    qdx = dx(metadata={"q": 6})

    return {
        "dh": dh,
        "du": du,
        "v": v,
        "u_k": u_k,
        "c": c,
        "A": A,
        "dx": qdx,
    }


def _build_mixed_problem():
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
    u_k = VectorFunction(name="u_k", field_names=["ux", "uy"], dof_handler=dh)

    coords_ux = np.asarray(dh.get_dof_coords("ux"), dtype=float)
    coords_uy = np.asarray(dh.get_dof_coords("uy"), dtype=float)
    u_k.components[0].nodal_values[:] = 0.5 + 0.8 * coords_ux[:, 0] - 0.2 * coords_ux[:, 1]
    u_k.components[1].nodal_values[:] = -0.3 + 0.4 * coords_uy[:, 0] + 0.9 * coords_uy[:, 1]

    A = Constant(np.array([[2.0, -0.5], [1.0, 3.0]], dtype=float), dim=2)
    qdx = dx(metadata={"q": 6})

    return {"dh": dh, "du": du, "v": v, "u_k": u_k, "A": A, "dx": qdx}


def _forms(ctx):
    return {
        "value_trial_right_vec": dot(dot(ctx["u_k"] * ctx["du"], ctx["c"]), ctx["v"]) * ctx["dx"],
        "value_trial_left_vec": dot(dot(ctx["c"], ctx["u_k"] * ctx["du"]), ctx["v"]) * ctx["dx"],
        "value_trial_right_mat": inner(dot(ctx["u_k"] * ctx["du"], ctx["A"]), grad(ctx["v"])) * ctx["dx"],
        "value_trial_left_mat": inner(dot(ctx["A"], ctx["u_k"] * ctx["du"]), grad(ctx["v"])) * ctx["dx"],
        "basis_trial_right_vec": dot(dot(ctx["v"] * ctx["du"], ctx["c"]), ctx["c"]) * ctx["dx"],
        "basis_trial_left_vec": dot(dot(ctx["c"], ctx["v"] * ctx["du"]), ctx["c"]) * ctx["dx"],
        "basis_trial_right_mat": inner(dot(ctx["v"] * ctx["du"], ctx["A"]), ctx["A"]) * ctx["dx"],
        "basis_trial_left_mat": inner(dot(ctx["A"], ctx["v"] * ctx["du"]), ctx["A"]) * ctx["dx"],
    }


def _assemble_matrix(form, dh: DofHandler, backend: str) -> np.ndarray:
    matrix, _ = assemble_form(Equation(form, None), dof_handler=dh, bcs=[], backend=backend)
    return np.asarray(matrix.toarray(), dtype=float)


@pytest.mark.parametrize("backend", _available_backends())
def test_dyad_dot_vector_and_matrix_closure_matches_python_backend(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_dyad_dot_{backend}"))

    ctx = _build_problem()
    failures: list[str] = []
    for name, form in _forms(ctx).items():
        expected = _assemble_matrix(form, ctx["dh"], "python")
        got = _assemble_matrix(form, ctx["dh"], backend)
        try:
            np.testing.assert_allclose(got, expected, rtol=1.0e-11, atol=1.0e-11)
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")

    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("backend", _available_backends())
def test_value_trial_right_matrix_dyad_matches_python_in_mixed_space(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_dyad_dot_mixed_{backend}"))

    ctx = _build_mixed_problem()
    form = inner(dot(ctx["u_k"] * ctx["du"], ctx["A"]), grad(ctx["v"])) * ctx["dx"]

    expected = _assemble_matrix(form, ctx["dh"], "python")
    got = _assemble_matrix(form, ctx["dh"], backend)

    np.testing.assert_allclose(got, expected, rtol=1.0e-11, atol=1.0e-11)
