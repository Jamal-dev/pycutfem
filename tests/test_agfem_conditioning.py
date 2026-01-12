import numpy as np
import scipy.sparse as sp
import pytest

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.xfem import AgFEMMapper

from pycutfem.ufl.measures import dx
from pycutfem.ufl.expressions import TrialFunction, TestFunction
from pycutfem.ufl.forms import Equation, assemble_form


def _cond_symmetric(A: np.ndarray, *, rtol: float = 1.0e-14) -> float:
    """
    Condition number estimate for (nearly) symmetric PSD matrices.

    For extreme sliver cuts, the mass matrix can be numerically singular and
    eigvalsh may return a tiny negative eigenvalue (~roundoff). Clamp the
    minimum eigenvalue to a relative tolerance to keep the metric finite and
    comparable before/after aggregation.
    """
    w = np.linalg.eigvalsh(0.5 * (A + A.T))
    w = np.asarray(w, dtype=float)
    wmax = float(np.max(w))
    if wmax <= 0.0:
        return float("inf")
    wmin = float(np.min(w))
    tol = float(rtol) * wmax
    if wmin < -tol:
        return float("inf")
    wmin_eff = max(wmin, tol)
    return wmax / wmin_eff


@pytest.mark.parametrize("backend", ["python"])
def test_agfem_sliver_conditioning_improves(backend):
    """
    Ill-conditioned sliver sanity check (AgFEM):

    Assemble an L2 mass matrix on Ω⁺ with a near-zero cut fraction, then
    constrain ghost DOFs by aggregation and verify the condition number drops.
    """
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=8, ny=2, poly_order=1)
    mesh = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=1)
    me = MixedElement(mesh, {"u": 1})
    dh = DofHandler(me, method="cg")

    # Sliver: interface very close to x=0.5 so one side of the cut is tiny on one column
    eps = 2.5e-7  # ~1e-6 fraction for h=0.25
    ls = AffineLevelSet(1.0, 0.0, -(0.5 - eps))  # x - (0.5 - eps)
    dh.classify_from_levelset(ls)

    u = TrialFunction("u", dof_handler=dh)
    v = TestFunction("u", dof_handler=dh)
    dx_pos = dx(level_set=ls, metadata={"side": "+", "q": 4})

    # Full mass matrix on Ω⁺ (matrix is singular on the full space; restrict to active DOFs)
    K_full, _ = assemble_form(Equation(u * v * dx_pos, None), dof_handler=dh, backend=backend)
    if not sp.issparse(K_full):
        K_full = sp.csr_matrix(K_full)
    K_full = K_full.tocsr()

    outside = mesh.element_bitset("outside").to_indices()
    cut = mesh.element_bitset("cut").to_indices()
    active_eids = np.unique(np.concatenate([outside, cut]).astype(int))
    active_dofs = sorted({int(g) for e in active_eids for g in dh.get_elemental_dofs(int(e))})
    A = K_full[active_dofs][:, active_dofs].toarray()
    cond_full = _cond_symmetric(A)

    mapper = AgFEMMapper(dh)
    ag = mapper.build_aggregation_map(ls, side="+", theta_min=0.05)
    cons = mapper.build_constraints(ag, fields=["u"])

    # Condense and restrict to master DOFs that are active in Ω⁺
    K_red = (cons.E_T @ (K_full @ cons.E)).tocsr()
    active_master_cols = [i for i, gd in enumerate(cons.master_ids.tolist()) if int(gd) in set(active_dofs)]
    A_red = K_red[active_master_cols][:, active_master_cols].toarray()
    cond_red = _cond_symmetric(A_red)

    assert np.isfinite(cond_full)
    assert np.isfinite(cond_red)
    assert cond_red < cond_full
