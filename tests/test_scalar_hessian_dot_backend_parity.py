import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant, FacetNormal, Function, Hessian, TestFunction, TrialFunction, dot, inner
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dS, dx
from pycutfem.utils.meshgen import structured_quad


def _compiled_backends() -> list[str]:
    backends = ["jit"]
    try:
        from pycutfem.jit.cpp_backend import compile_backend_cpp  # noqa: F401
    except Exception:
        return backends
    return backends + ["cpp"]


def _build_scalar_hessian_problem():
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=4, ny=4, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    mesh.tag_boundary_edges({"all": lambda x, y: True})

    me = MixedElement(mesh, field_specs={"p": 2})
    dh = DofHandler(me, method="cg")

    p = TrialFunction("p", dof_handler=dh)
    q = TestFunction("p", dof_handler=dh)
    p_k = Function(name="p_k", field_name="p", dof_handler=dh)
    p_k.set_values_from_function(lambda x, y: np.sin(1.7 * x) + 0.2 * np.cos(2.3 * y) + 0.15 * x * y)

    c = Constant(np.array([0.35, -0.55], dtype=float), dim=1)
    n = FacetNormal()
    dΩ = dx(metadata={"q": 4})
    dSb = dS(mesh.edge_bitset("all"), metadata={"q": 4})

    return {"dh": dh, "p": p, "q": q, "p_k": p_k, "c": c, "n": n, "dx": dΩ, "dS": dSb}


def _assemble_matrix(form, dh, backend):
    mat, _ = assemble_form(Equation(form, None), dof_handler=dh, bcs=[], backend=backend)
    return np.asarray(mat.toarray(), dtype=float)


def _assemble_vector(form, dh, backend):
    _, rhs = assemble_form(Equation(None, form), dof_handler=dh, bcs=[], backend=backend)
    return np.asarray(rhs, dtype=float).reshape(-1)


_VOLUME_BILINEAR_FORMS = (
    "Hdotc",
    "cdotH",
    "nHn",
)


def _volume_bilinear_forms(ctx):
    return {
        "Hdotc": inner(dot(Hessian(ctx["p"]), ctx["c"]), dot(Hessian(ctx["q"]), ctx["c"])) * ctx["dx"],
        "cdotH": inner(dot(ctx["c"], Hessian(ctx["p"])), dot(ctx["c"], Hessian(ctx["q"]))) * ctx["dx"],
        "nHn": inner(
            dot(ctx["c"], dot(Hessian(ctx["p"]), ctx["c"])),
            dot(ctx["c"], dot(Hessian(ctx["q"]), ctx["c"])),
        ) * ctx["dx"],
    }


_VOLUME_RESIDUAL_FORMS = (
    "Hdotc",
    "cdotH",
    "nHn",
)


def _volume_residual_forms(ctx):
    return {
        "Hdotc": inner(dot(Hessian(ctx["p_k"]), ctx["c"]), dot(Hessian(ctx["q"]), ctx["c"])) * ctx["dx"],
        "cdotH": inner(dot(ctx["c"], Hessian(ctx["p_k"])), dot(ctx["c"], Hessian(ctx["q"]))) * ctx["dx"],
        "nHn": inner(
            dot(ctx["c"], dot(Hessian(ctx["p_k"]), ctx["c"])),
            dot(ctx["c"], dot(Hessian(ctx["q"]), ctx["c"])),
        ) * ctx["dx"],
    }


_BOUNDARY_BILINEAR_FORMS = (
    "Hdotn",
    "ndotH",
    "nHn",
)


def _boundary_bilinear_forms(ctx):
    return {
        "Hdotn": inner(dot(Hessian(ctx["p"]), ctx["n"]), dot(Hessian(ctx["q"]), ctx["n"])) * ctx["dS"],
        "ndotH": inner(dot(ctx["n"], Hessian(ctx["p"])), dot(ctx["n"], Hessian(ctx["q"]))) * ctx["dS"],
        "nHn": inner(
            dot(ctx["n"], dot(Hessian(ctx["p"]), ctx["n"])),
            dot(ctx["n"], dot(Hessian(ctx["q"]), ctx["n"])),
        ) * ctx["dS"],
    }


_BOUNDARY_RESIDUAL_FORMS = (
    "Hdotn",
    "ndotH",
    "nHn",
)


def _boundary_residual_forms(ctx):
    return {
        "Hdotn": inner(dot(Hessian(ctx["p_k"]), ctx["n"]), dot(Hessian(ctx["q"]), ctx["n"])) * ctx["dS"],
        "ndotH": inner(dot(ctx["n"], Hessian(ctx["p_k"])), dot(ctx["n"], Hessian(ctx["q"]))) * ctx["dS"],
        "nHn": inner(
            dot(ctx["n"], dot(Hessian(ctx["p_k"]), ctx["n"])),
            dot(ctx["n"], dot(Hessian(ctx["q"]), ctx["n"])),
        ) * ctx["dS"],
    }


def _cache_token(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)


@pytest.mark.parametrize("form_name", _VOLUME_BILINEAR_FORMS)
@pytest.mark.parametrize("backend", _compiled_backends())
def test_scalar_hessian_volume_bilinear_matches_python(form_name, backend, tmp_path, monkeypatch):
    monkeypatch.setenv(
        "PYCUTFEM_CACHE_DIR",
        str(tmp_path / f"pycutfem_scalar_hessian_volume_bilinear_{backend}_{_cache_token(form_name)}"),
    )

    ctx = _build_scalar_hessian_problem()
    form = _volume_bilinear_forms(ctx)[form_name]

    np.testing.assert_allclose(
        _assemble_matrix(form, ctx["dh"], backend),
        _assemble_matrix(form, ctx["dh"], "python"),
        rtol=1.0e-11,
        atol=1.0e-11,
    )


@pytest.mark.parametrize("form_name", _VOLUME_RESIDUAL_FORMS)
@pytest.mark.parametrize("backend", _compiled_backends())
def test_scalar_hessian_volume_residual_matches_python(form_name, backend, tmp_path, monkeypatch):
    monkeypatch.setenv(
        "PYCUTFEM_CACHE_DIR",
        str(tmp_path / f"pycutfem_scalar_hessian_volume_residual_{backend}_{_cache_token(form_name)}"),
    )

    ctx = _build_scalar_hessian_problem()
    form = _volume_residual_forms(ctx)[form_name]

    np.testing.assert_allclose(
        _assemble_vector(form, ctx["dh"], backend),
        _assemble_vector(form, ctx["dh"], "python"),
        rtol=1.0e-11,
        atol=1.0e-11,
    )


@pytest.mark.parametrize("form_name", _BOUNDARY_BILINEAR_FORMS)
@pytest.mark.parametrize("backend", _compiled_backends())
def test_scalar_hessian_boundary_bilinear_matches_python(form_name, backend, tmp_path, monkeypatch):
    monkeypatch.setenv(
        "PYCUTFEM_CACHE_DIR",
        str(tmp_path / f"pycutfem_scalar_hessian_boundary_bilinear_{backend}_{_cache_token(form_name)}"),
    )

    ctx = _build_scalar_hessian_problem()
    form = _boundary_bilinear_forms(ctx)[form_name]

    np.testing.assert_allclose(
        _assemble_matrix(form, ctx["dh"], backend),
        _assemble_matrix(form, ctx["dh"], "python"),
        rtol=1.0e-11,
        atol=1.0e-11,
    )


@pytest.mark.parametrize("form_name", _BOUNDARY_RESIDUAL_FORMS)
@pytest.mark.parametrize("backend", _compiled_backends())
def test_scalar_hessian_boundary_residual_matches_python(form_name, backend, tmp_path, monkeypatch):
    monkeypatch.setenv(
        "PYCUTFEM_CACHE_DIR",
        str(tmp_path / f"pycutfem_scalar_hessian_boundary_residual_{backend}_{_cache_token(form_name)}"),
    )

    ctx = _build_scalar_hessian_problem()
    form = _boundary_residual_forms(ctx)[form_name]

    np.testing.assert_allclose(
        _assemble_vector(form, ctx["dh"], backend),
        _assemble_vector(form, ctx["dh"], "python"),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
