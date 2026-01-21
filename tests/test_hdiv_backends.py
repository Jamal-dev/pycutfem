import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl import (
    Equation,
    HdivFunction,
    HdivTestFunction,
    HdivTrialFunction,
    TestFunction as UFLTestFunction,
    TrialFunction as UFLTrialFunction,
    assemble_form,
    div,
    dx,
    inner,
)
from pycutfem.utils.meshgen import structured_quad


def _assemble_u_p_equation(dh: DofHandler, *, backend: str, k: int):
    mesh = dh.mixed_element.mesh
    assert mesh.element_type == "quad"

    u = HdivTrialFunction("u")
    v = HdivTestFunction("u")
    p = UFLTrialFunction("p")
    q = UFLTestFunction("p")

    u_k = HdivFunction(name="u_k", field_name="u", dof_handler=dh)
    rng = np.random.default_rng(0)
    u_k.nodal_values[:] = rng.standard_normal(u_k.nodal_values.size)

    # One mixed form that exercises:
    # - RT basis values (inner(u,v))
    # - RT divergence (div(u), div(v))
    # - RT coefficient evaluation (u_k)
    a = (inner(u, v) - p * div(v) + div(u) * q + 1.0e-3 * p * q) * dx()
    L = inner(u_k, v) * dx()

    K, F = assemble_form(Equation(a, L), dof_handler=dh, bcs=[], backend=backend)
    return K.tocsr(), np.asarray(F, dtype=float)


@pytest.mark.parametrize("backend", ("jit", "cpp"))
def test_hdiv_rt0_ufl_assembly_matches_python(backend, monkeypatch, tmp_path):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"jit_cache_{backend}"))

    nodes, elem_conn, edges, corner = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=1)
    mesh = Mesh(
        nodes,
        elem_conn,
        edges_connectivity=edges,
        elements_corner_nodes=corner,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, {"u": ("RT", 0), "p": ("DG", 0)})
    dh = DofHandler(me, method="cg")

    K_ref, F_ref = _assemble_u_p_equation(dh, backend="python", k=0)
    K, F = _assemble_u_p_equation(dh, backend=backend, k=0)

    np.testing.assert_allclose(K.toarray(), K_ref.toarray(), atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(F, F_ref, atol=1.0e-12, rtol=0.0)


def test_hdiv_rt1_ufl_assembly_matches_python_jit(monkeypatch, tmp_path):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "jit_cache_rt1"))

    nodes, elem_conn, edges, corner = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=1)
    mesh = Mesh(
        nodes,
        elem_conn,
        edges_connectivity=edges,
        elements_corner_nodes=corner,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, {"u": ("RT", 1), "p": ("DG", 1)})
    dh = DofHandler(me, method="cg")

    K_ref, F_ref = _assemble_u_p_equation(dh, backend="python", k=1)
    K, F = _assemble_u_p_equation(dh, backend="jit", k=1)

    np.testing.assert_allclose(K.toarray(), K_ref.toarray(), atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(F, F_ref, atol=1.0e-12, rtol=0.0)
