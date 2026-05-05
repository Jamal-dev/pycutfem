import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import TestFunction as UFLTestFunction
from pycutfem.ufl.expressions import TrialFunction, jump
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dFacetPatch, dGhost, dx
from pycutfem.utils.meshgen import structured_quad


def _eig_min_max_sym(A) -> tuple[float, float]:
    Ad = A.toarray()
    Ad = 0.5 * (Ad + Ad.T)
    w = np.linalg.eigvalsh(Ad)
    return float(w[0]), float(w[-1])


def test_facet_patch_stabilizes_tiny_sliver_cg() -> None:
    """
    Regression test: `dFacetPatch` must add coercivity for *CG* spaces on
    sliver cuts, where plain volume terms can become nearly singular.

    Setup:
      - Q2 CG field on a 4x1 structured mesh on [0,1]x[0,1].
      - Vertical level set very close to x=0.25, producing a tiny '+' sliver in
        the first element.

    Observations:
      - `jump(u)` on `dGhost` is identically 0 for CG → no stabilization.
      - `jump(u)` on `dFacetPatch` is non-zero (polynomial extension) and
        increases the smallest eigenvalue by orders of magnitude.
    """
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=4, ny=1, poly_order=1)
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

    # Tiny '+' sliver in the first element: phi = x - 0.249999.
    ls = AffineLevelSet(1.0, 0.0, -0.249999)
    dh.classify_from_levelset(ls)

    outside = mesh.element_bitset("outside")
    cut = mesh.element_bitset("cut")
    has_outside = outside | cut
    assert cut.cardinality() > 0

    ghost_pos = mesh.edge_bitset("ghost_pos")
    ghost_both = mesh.edge_bitset("ghost_both")
    ghost = ghost_pos | ghost_both
    assert ghost.cardinality() > 0

    u = TrialFunction("u", dof_handler=dh)
    v = UFLTestFunction("u", dof_handler=dh)

    q = 6
    dx_pos = dx(defined_on=has_outside, level_set=ls, metadata={"side": "+", "q": q})
    dG = dGhost(defined_on=ghost, level_set=ls, metadata={"q": q, "derivs": {(0, 0)}})
    dW = dFacetPatch(defined_on=ghost, level_set=ls, metadata={"q": q})

    M, _ = assemble_form(Equation((u * v) * dx_pos, None), dof_handler=dh, bcs=[], backend="python")
    Sg, _ = assemble_form(Equation(jump(u) * jump(v) * dG, None), dof_handler=dh, bcs=[], backend="python")
    Sf, _ = assemble_form(Equation(jump(u) * jump(v) * dW, None), dof_handler=dh, bcs=[], backend="python")

    M = M.tocsr()
    Sg = Sg.tocsr()
    Sf = Sf.tocsr()

    assert Sg.nnz == 0  # CG value-jumps on facets are zero
    assert Sf.nnz > 0
    assert float(np.max(np.abs(Sf.data))) > 1.0e-12

    lam_min_M, lam_max_M = _eig_min_max_sym(M)
    lam_min_A, lam_max_A = _eig_min_max_sym(M + Sf)

    # The sliver makes the (volume-only) operator extremely ill-conditioned.
    # (Allow tiny negative due to roundoff from eigendecomposition.)
    assert lam_min_M < 1.0e-12

    # Facet-patch stabilization couples the sliver to its neighbor and restores
    # a meaningful smallest eigenvalue.
    assert lam_min_A > 1.0e-8
    assert lam_max_A > lam_max_M
