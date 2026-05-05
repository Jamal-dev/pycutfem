import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import FacetNormal, Hessian, Jump, TrialFunction, TestFunction, dot, inner, restrict
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dFacetPatch
from pycutfem.utils.meshgen import structured_quad


def test_facet_patch_hessian_nn_jump_python_cpp_backend_parity(monkeypatch, tmp_path) -> None:
    """
    Regression test: the C++ backend must correctly assemble the "double normal
    derivative" Hessian ghost penalty used in the Turek-cylinder benchmark:

      ⟨ [nᵀ H(u) n], [nᵀ H(v) n] ⟩_{patch}

    including element restriction via `restrict(…, physical_domain)`.
    """
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_jit_cache"))

    nodes, elems, edges, corners = structured_quad(2.0, 1.0, nx=8, ny=4, poly_order=1)
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

    ls = AffineLevelSet(a=1.0, b=0.2, c=-0.93)
    dh.classify_from_levelset(ls)

    ghost = mesh.edge_bitset("ghost")
    assert ghost.cardinality() > 0

    physical = mesh.element_bitset("outside") | mesh.element_bitset("cut")

    u_pos = TrialFunction("u", dof_handler=dh, side="+")
    u_neg = TrialFunction("u", dof_handler=dh, side="-")
    v_pos = TestFunction("u", dof_handler=dh, side="+")
    v_neg = TestFunction("u", dof_handler=dh, side="-")

    u_jump = Jump(restrict(u_pos, physical), restrict(u_neg, physical))
    v_jump = Jump(restrict(v_pos, physical), restrict(v_neg, physical))

    n = FacetNormal()
    d2n_u = dot(dot(Hessian(u_jump), n), n)
    d2n_v = dot(dot(Hessian(v_jump), n), n)

    dW = dFacetPatch(
        defined_on=ghost,
        level_set=ls,
        metadata={"q": 6, "derivs": {(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)}},
    )
    a = inner(d2n_u, d2n_v) * dW

    K_py, _ = assemble_form(Equation(a, None), dof_handler=dh, backend="python")
    K_cpp, _ = assemble_form(Equation(a, None), dof_handler=dh, backend="cpp")

    K_py = K_py.tocsr()
    K_cpp = K_cpp.tocsr()

    assert K_py.nnz > 0
    assert K_cpp.nnz > 0

    diff = (K_py - K_cpp).tocoo()
    max_diff = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
    assert max_diff < 1.0e-9

