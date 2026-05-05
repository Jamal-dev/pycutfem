import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl import (
    Equation,
    Function,
    HdivFunction,
    HdivTestFunction,
    HdivTrialFunction,
    TestFunction as UflTestFunction,
    TrialFunction,
    assemble_form,
    div,
    dot,
    dx,
    grad,
    inner,
)
from pycutfem.ufl.expressions import FacetNormal
from pycutfem.ufl.measures import dS
from pycutfem.utils.meshgen import structured_quad
from examples.utils.biofilm.one_domain import _tangential_component_2d


def _epsilon(v):
    return 0.5 * (grad(v) + grad(v).T)


def _assemble_hdiv_grad_equation(dh: DofHandler, *, backend: str):
    u = HdivTrialFunction("u")
    v = HdivTestFunction("u")
    u_k = HdivFunction(name="u_k", field_name="u", dof_handler=dh)

    rng = np.random.default_rng(7)
    u_k.nodal_values[:] = rng.standard_normal(u_k.nodal_values.size)

    a = (
        inner(grad(u), grad(v))
        + 0.25 * dot(dot(grad(u), u_k), v)
        + 0.5 * div(u) * div(v)
    ) * dx()
    L = (inner(grad(u_k), grad(v)) + 0.75 * inner(u_k, v)) * dx()

    K, F = assemble_form(Equation(a, L), dof_handler=dh, bcs=[], backend=backend)
    return K.tocsr(), np.asarray(F, dtype=float)


def _assemble_hdiv_value_symmetric_gradient_equation(dh: DofHandler, *, backend: str):
    v = HdivTrialFunction("v")
    w = HdivTestFunction("v")
    dalpha = TrialFunction("alpha", dof_handler=dh)
    alpha_test = UflTestFunction("alpha", dof_handler=dh)
    v_k = HdivFunction(name="v_k", field_name="v", dof_handler=dh)
    alpha_k = Function(name="alpha_k", field_name="alpha", dof_handler=dh)

    rng = np.random.default_rng(11)
    v_k.nodal_values[:] = rng.standard_normal(v_k.nodal_values.size)
    alpha_k.nodal_values[:] = rng.standard_normal(alpha_k.nodal_values.size)

    a = (inner(_epsilon(v), _epsilon(w)) + dalpha * inner(_epsilon(v_k), _epsilon(w)) + dalpha * alpha_test) * dx()
    L = (0.5 * inner(_epsilon(v_k), _epsilon(w)) + alpha_k * alpha_test) * dx()

    K, F = assemble_form(Equation(a, L), dof_handler=dh, bcs=[], backend=backend)
    return K.tocsr(), np.asarray(F, dtype=float)


def _assemble_boundary_hdiv_grad_equation(dh: DofHandler, *, backend: str):
    u = HdivTrialFunction("u")
    v = HdivTestFunction("u")
    u_k = HdivFunction(name="u_k", field_name="u", dof_handler=dh)

    rng = np.random.default_rng(19)
    u_k.nodal_values[:] = rng.standard_normal(u_k.nodal_values.size)

    n = FacetNormal()

    def tcomp(w):
        return _tangential_component_2d(w, n)

    qmeta = {"q": 4}
    a = (inner(grad(u), grad(v)) - tcomp(dot(2.0 * grad(u), n)) * tcomp(v)) * dS(metadata=qmeta)
    L = (inner(grad(u_k), grad(v)) - tcomp(dot(2.0 * grad(u_k), n)) * tcomp(v)) * dS(metadata=qmeta)

    K, F = assemble_form(Equation(a, L), dof_handler=dh, bcs=[], backend=backend)
    return K.tocsr(), np.asarray(F, dtype=float)


@pytest.mark.parametrize("backend", ("jit", "cpp"))
def test_hdiv_grad_assembly_matches_python(backend, monkeypatch, tmp_path):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"jit_cache_{backend}"))
    if backend == "cpp":
        monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")

    nodes, elem_conn, edges, corner = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=1)
    mesh = Mesh(
        nodes,
        elem_conn,
        edges_connectivity=edges,
        elements_corner_nodes=corner,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, {"u": ("RT", 0)})
    dh = DofHandler(me, method="cg")

    K_ref, F_ref = _assemble_hdiv_grad_equation(dh, backend="python")
    K, F = _assemble_hdiv_grad_equation(dh, backend=backend)

    np.testing.assert_allclose(K.toarray(), K_ref.toarray(), atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(F, F_ref, atol=1.0e-12, rtol=0.0)


@pytest.mark.parametrize("backend", ("jit", "cpp"))
def test_hdiv_value_symmetric_gradient_matches_python(backend, monkeypatch, tmp_path):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"jit_cache_value_symgrad_{backend}"))
    if backend == "cpp":
        monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")

    nodes, elem_conn, edges, corner = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes,
        elem_conn,
        edges_connectivity=edges,
        elements_corner_nodes=corner,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, {"v": ("RT", 0), "alpha": 1})
    dh = DofHandler(me, method="cg")

    K_ref, F_ref = _assemble_hdiv_value_symmetric_gradient_equation(dh, backend="python")
    K, F = _assemble_hdiv_value_symmetric_gradient_equation(dh, backend=backend)

    K_arr = K.toarray()
    assert np.isfinite(K_arr).all()
    assert np.isfinite(F).all()
    np.testing.assert_allclose(K_arr, K_ref.toarray(), atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(F, F_ref, atol=1.0e-12, rtol=0.0)


@pytest.mark.parametrize("backend", ("jit", "cpp"))
def test_boundary_hdiv_grad_rt1_matches_python(backend, monkeypatch, tmp_path):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"jit_cache_boundary_hdiv_grad_{backend}"))
    if backend == "cpp":
        monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")

    nodes, elem_conn, edges, corner = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=1)
    mesh = Mesh(
        nodes,
        elem_conn,
        edges_connectivity=edges,
        elements_corner_nodes=corner,
        element_type="quad",
        poly_order=1,
    )
    mesh.tag_boundary_edges({"all": lambda x, y: True})
    me = MixedElement(mesh, {"u": ("RT", 1)})
    dh = DofHandler(me, method="cg")

    K_ref, F_ref = _assemble_boundary_hdiv_grad_equation(dh, backend="python")
    K, F = _assemble_boundary_hdiv_grad_equation(dh, backend=backend)

    np.testing.assert_allclose(K.toarray(), K_ref.toarray(), atol=5.0e-13, rtol=0.0)
    np.testing.assert_allclose(F, F_ref, atol=5.0e-13, rtol=0.0)
