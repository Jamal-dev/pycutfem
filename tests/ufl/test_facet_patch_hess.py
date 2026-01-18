import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet, CircleLevelSet, LevelSetMeshAdaptation
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import TrialFunction, TestFunction, Jump, Hessian, inner
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dFacetPatch
from pycutfem.utils.meshgen import structured_quad


@pytest.mark.parametrize("jit_backend", ["cpp", "numba"])
def test_facet_patch_hessian_penalty_spd_and_backend_parity(monkeypatch, tmp_path, jit_backend: str) -> None:
    """
    Regression test: dFacetPatch must support Hessian-based penalties (order-2 jets)
    and match Python vs JIT assembly.
    """
    monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", str(jit_backend))
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_jit_cache"))

    nodes, elems, edges, corners = structured_quad(2.0, 1.0, nx=10, ny=4, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )

    me = MixedElement(mesh, {"u": 2})
    dh = DofHandler(me, method="cg")

    ls = AffineLevelSet(1.0, 0.2, -0.93)  # non-aligned cut
    dh.classify_from_levelset(ls)

    ghost = mesh.edge_bitset("ghost")
    assert ghost.cardinality() > 0

    u_pos = TrialFunction("u", dof_handler=dh, side="+")
    u_neg = TrialFunction("u", dof_handler=dh, side="-")
    v_pos = TestFunction("u", dof_handler=dh, side="+")
    v_neg = TestFunction("u", dof_handler=dh, side="-")

    a = inner(Hessian(Jump(u_pos, u_neg)), Hessian(Jump(v_pos, v_neg))) * dFacetPatch(
        defined_on=ghost,
        level_set=ls,
        metadata={"q": 6, "derivs": {(2, 0), (1, 1), (0, 2)}},
    )

    K_py, _ = assemble_form(Equation(a, None), dof_handler=dh, backend="python")
    K_jit, _ = assemble_form(Equation(a, None), dof_handler=dh, backend="jit")

    K_py = K_py.toarray()
    K_jit = K_jit.toarray()

    assert np.max(np.abs(K_py)) > 1.0e-12
    assert np.allclose(K_jit, K_jit.T, atol=1e-12)

    # PSD (robust relative tolerance)
    evals = np.linalg.eigvalsh(K_jit)
    lam_min = float(evals[0])
    lam_max = float(evals[-1])
    eps = np.finfo(float).eps
    tol = 200 * eps * max(1.0, lam_max)
    assert lam_min >= -tol

    diff = K_py - K_jit
    max_diff = float(np.max(np.abs(diff))) if diff.size else 0.0
    assert max_diff < 1.0e-9


@pytest.mark.parametrize("jit_backend", ["cpp", "numba"])
def test_facet_patch_hessian_penalty_deformation_backend_parity(monkeypatch, tmp_path, jit_backend: str) -> None:
    """
    Regression test: Hessian facet-patch penalties must remain stable under
    isoparametric deformation and agree between backends.
    """
    monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", str(jit_backend))
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_jit_cache"))
    monkeypatch.delenv("PYCUTFEM_FACET_PATCH_GEO_MODE", raising=False)

    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=6, ny=6, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )

    me = MixedElement(mesh, {"u": 2})
    dh = DofHandler(me, method="cg")

    ls = CircleLevelSet(center=(0.5, 0.5), radius=0.33)
    dh.classify_from_levelset(ls)

    ghost = mesh.edge_bitset("ghost")
    assert ghost.cardinality() > 0

    adapter = LevelSetMeshAdaptation(mesh, order=2, threshold=1.0, max_steps=6)
    deformation = adapter.calc_deformation(ls, q_vol=6)

    u_pos = TrialFunction("u", dof_handler=dh, side="+")
    u_neg = TrialFunction("u", dof_handler=dh, side="-")
    v_pos = TestFunction("u", dof_handler=dh, side="+")
    v_neg = TestFunction("u", dof_handler=dh, side="-")

    dW = dFacetPatch(defined_on=ghost, level_set=ls, metadata={"q": 6}, deformation=deformation)
    a = inner(Hessian(Jump(u_pos, u_neg)), Hessian(Jump(v_pos, v_neg))) * dW

    K_py, _ = assemble_form(Equation(a, None), dof_handler=dh, backend="python")
    K_jit, _ = assemble_form(Equation(a, None), dof_handler=dh, backend="jit")

    K_py = K_py.tocsr()
    K_jit = K_jit.tocsr()

    assert K_py.nnz > 0
    # Hessian penalties can be larger than value-jump penalties; still guard against blow-up.
    assert float(np.max(np.abs(K_py.data))) < 1.0e9

    diff = (K_py - K_jit).tocoo()
    max_diff = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
    assert max_diff < 1.0e-8
