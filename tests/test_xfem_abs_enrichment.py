import numpy as np
import pytest
import scipy.sparse.linalg as spla

from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.meshgen import structured_quad
from pycutfem.xfem import XFEMDofHandler

from pycutfem.ufl.expressions import CellDiameter, Constant, Function, TestFunction, TrialFunction, grad, inner
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dInterface, dx


def test_xfem_abs_enrichment_penalty_dirichlet_smoke(monkeypatch, tmp_path):
    """
    Regression smoke test for shifted-|phi| ("abs") enrichment.

    This exercises:
      - cut-volume basis/grad precompute for abs enrichment (dx on both sides),
      - interface basis precompute for abs enrichment (penalty term on dInterface),
    on the C++ JIT backend.
    """
    monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "jit_cache"))

    # --- mesh / level set -----------------------------------------------------
    ll = (-1.0, -1.0)
    L = 2.0
    nx = ny = 16
    nodes, elems, edges, corners = structured_quad(L, L, nx=nx, ny=ny, poly_order=1, offset=ll)
    mesh = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=1)
    mesh.tag_boundary_edges({"boundary": lambda *_: True})

    # Interface y=0.03 (negative below), intentionally not mesh-aligned to
    # exercise cut-cell and interface precomputations.
    level_set = AffineLevelSet(a=0.0, b=1.0, c=-0.03).normalised()

    me = MixedElement(mesh, {"u": 1})
    dh0 = DofHandler(me, method="cg")
    dh0.classify_from_levelset(level_set)
    assert mesh.element_bitset("cut").cardinality() > 0

    dh = XFEMDofHandler(dh0)
    dh.rebuild_enrichment(level_set, enrich={"u": "abs"})
    assert dh.total_dofs > dh.base_total_dofs

    # --- weak form ------------------------------------------------------------
    u = TrialFunction("u", dof_handler=dh)
    v = TestFunction("u", dof_handler=dh)
    h = CellDiameter()

    q = 4
    dx_pos = dx(level_set=level_set, metadata={"side": "+", "q": q})
    dx_neg = dx(level_set=level_set, metadata={"side": "-", "q": q})
    dGamma = dInterface(defined_on=mesh.element_bitset("cut"), level_set=level_set, metadata={"q": q})

    pen = Constant(1.0e6)
    a = inner(grad(u), grad(v)) * dx_pos + inner(grad(u), grad(v)) * dx_neg
    a += (pen / h) * u * v * dGamma
    Lform = Constant(1.0) * v * dx_neg

    bc_boundary = BoundaryCondition("u", "dirichlet", "boundary", lambda *_: 0.0)

    K, F = assemble_form(Equation(a, Lform), dof_handler=dh, bcs=[bc_boundary], backend="jit")
    sol = spla.spsolve(K.tocsc(), F)
    assert np.all(np.isfinite(sol))

    uh = Function(name="uh", field_name="u", dof_handler=dh)
    gd = dh.get_field_slice("u")
    uh.set_nodal_values(gd, sol[gd])

    # Base nodal values reflect the physical function values at mesh nodes for abs enrichment.
    n_base = int(np.asarray(dh.base.get_field_slice("u"), dtype=int).size)
    u_base = np.asarray(uh.nodal_values[:n_base], dtype=float)
    coords = np.asarray(dh.base.get_dof_coords("u"), dtype=float)

    # Outside (y>0): should be ~0 due to u=0 on Γ and on outer boundary.
    pos = coords[:, 1] > 0.4
    assert float(np.max(np.abs(u_base[pos]))) < 5.0e-4

    # Inside (y<0): should be non-trivial due to source term.
    neg = coords[:, 1] < -0.4
    assert float(np.max(np.abs(u_base[neg]))) > 1.0e-6
