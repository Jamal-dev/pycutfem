import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl import Equation, HdivTestFunction, HdivTrialFunction, assemble_form, dot, inner
from pycutfem.ufl.expressions import FacetNormal, Jump, Neg, Pos
from pycutfem.ufl.measures import dGhost, dInterface
from pycutfem.utils.meshgen import structured_quad


def _cut_quad_mesh():
    nodes, elem_conn, edges, corner = structured_quad(1.0, 1.0, nx=4, ny=3, poly_order=1)
    mesh = Mesh(
        nodes,
        elem_conn,
        edges_connectivity=edges,
        elements_corner_nodes=corner,
        element_type="quad",
        poly_order=1,
    )
    # Non-aligned line that cuts the mesh.
    level_set = AffineLevelSet(a=1.0, b=0.21, c=-0.53)
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)

    cut = mesh.element_bitset("cut")
    ghost = mesh.edge_bitset("ghost")
    assert cut.cardinality() > 0
    assert ghost.cardinality() > 0
    return mesh, level_set, cut, ghost


def _assemble_matrix(dh: DofHandler, a, *, backend: str):
    K, F = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend=backend)
    return K.tocsr(), np.asarray(F, dtype=float)


@pytest.mark.parametrize("backend", ("jit", "cpp"))
def test_hdiv_rt0_dghost_ufl_assembly_matches_python(backend, monkeypatch, tmp_path):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"jit_cache_{backend}_rt0_dghost"))

    mesh, level_set, _cut, ghost = _cut_quad_mesh()
    me = MixedElement(mesh, {"u": ("RT", 0)})
    dh = DofHandler(me, method="cg")

    u = HdivTrialFunction("u")
    v = HdivTestFunction("u")
    a = inner(Jump(u), Jump(v)) * dGhost(defined_on=ghost, level_set=level_set, metadata={"q": 6})

    K_ref, F_ref = _assemble_matrix(dh, a, backend="python")
    K, F = _assemble_matrix(dh, a, backend=backend)

    np.testing.assert_allclose(K.toarray(), K_ref.toarray(), atol=2.0e-10, rtol=0.0)
    np.testing.assert_allclose(F, F_ref, atol=2.0e-10, rtol=0.0)


def test_hdiv_rt1_dghost_ufl_assembly_matches_python_jit(monkeypatch, tmp_path):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "jit_cache_rt1_dghost"))

    mesh, level_set, _cut, ghost = _cut_quad_mesh()
    me = MixedElement(mesh, {"u": ("RT", 1)})
    dh = DofHandler(me, method="cg")

    u = HdivTrialFunction("u")
    v = HdivTestFunction("u")
    a = inner(Jump(u), Jump(v)) * dGhost(defined_on=ghost, level_set=level_set, metadata={"q": 8})

    K_ref, F_ref = _assemble_matrix(dh, a, backend="python")
    K, F = _assemble_matrix(dh, a, backend="jit")

    np.testing.assert_allclose(K.toarray(), K_ref.toarray(), atol=2.0e-10, rtol=0.0)
    np.testing.assert_allclose(F, F_ref, atol=2.0e-10, rtol=0.0)


@pytest.mark.parametrize("backend", ("jit", "cpp"))
def test_hdiv_rt0_dinterface_ufl_assembly_matches_python(backend, monkeypatch, tmp_path):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"jit_cache_{backend}_rt0_dinterface"))

    mesh, level_set, cut, _ghost = _cut_quad_mesh()
    me = MixedElement(mesh, {"u": ("RT", 0)})
    dh = DofHandler(me, method="cg")

    u = HdivTrialFunction("u")
    v = HdivTestFunction("u")
    n = FacetNormal()
    a = (
        dot(Pos(u), n) * dot(Pos(v), n)
        + dot(Neg(u), n) * dot(Neg(v), n)
    ) * dInterface(defined_on=cut, level_set=level_set, metadata={"q": 6})

    K_ref, F_ref = _assemble_matrix(dh, a, backend="python")
    K, F = _assemble_matrix(dh, a, backend=backend)

    np.testing.assert_allclose(K.toarray(), K_ref.toarray(), atol=1.0e-11, rtol=0.0)
    np.testing.assert_allclose(F, F_ref, atol=1.0e-11, rtol=0.0)


def test_hdiv_rt1_dinterface_ufl_assembly_matches_python_jit(monkeypatch, tmp_path):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "jit_cache_rt1_dinterface"))

    mesh, level_set, cut, _ghost = _cut_quad_mesh()
    me = MixedElement(mesh, {"u": ("RT", 1)})
    dh = DofHandler(me, method="cg")

    u = HdivTrialFunction("u")
    v = HdivTestFunction("u")
    n = FacetNormal()
    a = (
        dot(Pos(u), n) * dot(Pos(v), n)
        + dot(Neg(u), n) * dot(Neg(v), n)
    ) * dInterface(defined_on=cut, level_set=level_set, metadata={"q": 8})

    K_ref, F_ref = _assemble_matrix(dh, a, backend="python")
    K, F = _assemble_matrix(dh, a, backend="jit")

    np.testing.assert_allclose(K.toarray(), K_ref.toarray(), atol=1.0e-11, rtol=0.0)
    np.testing.assert_allclose(F, F_ref, atol=1.0e-11, rtol=0.0)
