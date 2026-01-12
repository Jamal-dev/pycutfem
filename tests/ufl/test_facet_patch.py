import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import TestFunction as UFLTestFunction
from pycutfem.ufl.expressions import TrialFunction, jump
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dFacetPatch
from pycutfem.utils.meshgen import structured_quad


@pytest.mark.parametrize("p", [1, 2])
def test_facet_patch_jump_cg_nonzero_and_backend_parity(monkeypatch, p: int) -> None:
    """
    Regression test: NGSolve-style facet-patch integrals must produce a non-zero
    stabilization matrix even for CG spaces.

    If `dFacetPatch` accidentally degenerates to a facet integral (like `dGhost`),
    then `jump(u)` is identically zero for CG and the matrix becomes zero.
    """
    monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")

    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=6, ny=6, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )

    me = MixedElement(mesh, {"u": int(p)})
    dh = DofHandler(me, method="cg")

    # Non-aligned cut so we actually get a ghost edge band.
    ls = AffineLevelSet(1.0, 0.2, -0.47)
    dh.classify_from_levelset(ls)

    ghost = mesh.edge_bitset("ghost")
    assert ghost.cardinality() > 0

    u = TrialFunction("u", dof_handler=dh)
    v = UFLTestFunction("u", dof_handler=dh)

    dW = dFacetPatch(defined_on=ghost, level_set=ls, metadata={"q": 6})
    a = jump(u) * jump(v) * dW

    K_py, _ = assemble_form(Equation(a, None), dof_handler=dh, backend="python")
    K_jit, _ = assemble_form(Equation(a, None), dof_handler=dh, backend="jit")

    K_py = K_py.tocsr()
    K_jit = K_jit.tocsr()

    assert K_py.nnz > 0
    assert float(np.max(np.abs(K_py.data))) > 1.0e-12

    diff = (K_py - K_jit).tocoo()
    max_diff = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
    assert max_diff < 1.0e-10
