import numpy as np
import pytest
import scipy.sparse.linalg as spla
import sympy as sp

from pycutfem.core.levelset import AnnulusLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.meshgen import structured_quad
from pycutfem.xfem import AgFEMMapper

from pycutfem.ufl.analytic import Analytic, x, y
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    FacetNormal,
    Function,
    TestFunction,
    TrialFunction,
    dot,
    grad,
    inner,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dInterface, dx


def test_poisson_annulus_agfem_smoke(monkeypatch):
    """
    Regression smoke test for AgFEM constraints in a real CutFEM solve.

    Mirrors the NGSXFEM `fictdom_aggfem.py` setup (annulus + Nitsche BC) but:
    - uses a structured quad background mesh
    - eliminates outside DOFs and applies AgFEM constraints on cut cells
    - runs through the C++ JIT backend
    """
    monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")

    ll = (-1.0, -1.0)
    L = 2.0
    maxh = 0.1
    nx = ny = int(round(L / maxh))
    nodes, elems, edges, corners = structured_quad(L, L, nx=nx, ny=ny, poly_order=1, offset=ll)
    mesh = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=1)

    degree = 2
    me = MixedElement(mesh, {"u": degree})
    dh = DofHandler(me, method="cg")

    r1, r2 = 1.0 / 4.0, 3.0 / 4.0
    level_set = AnnulusLevelSet(center=(0.0, 0.0), r_inner=r1, r_outer=r2)
    dh.classify_from_levelset(level_set)

    inside = mesh.element_bitset("inside")
    cut = mesh.element_bitset("cut")
    outside = mesh.element_bitset("outside")
    physical = inside | cut
    assert cut.cardinality() > 0

    dh.tag_dof_bitset("inactive", "u", elem_mask=outside, strict=True)
    bc_inactive = BoundaryCondition("u", "dirichlet", "inactive", lambda x_, y_: 0.0)

    r = sp.sqrt(x**2 + y**2)
    exact_expr = 20.0 * (float(r2) - r) * (r - float(r1))
    rhs_expr = -sp.diff(exact_expr, x, 2) - sp.diff(exact_expr, y, 2)
    exact = Analytic(exact_expr)
    rhs = Analytic(rhs_expr)

    u = TrialFunction("u", dof_handler=dh)
    v = TestFunction("u", dof_handler=dh)
    n = FacetNormal()
    h = CellDiameter()

    q = 2 * degree + 2
    dx_neg = dx(defined_on=physical, level_set=level_set, metadata={"side": "-", "q": q})
    dGamma = dInterface(defined_on=cut, level_set=level_set, metadata={"q": q})

    gamma = Constant(10.0 * degree * degree)
    a = inner(grad(u), grad(v)) * dx_neg
    a += -dot(grad(u), n) * v * dGamma
    a += -dot(grad(v), n) * u * dGamma
    a += (gamma / h) * u * v * dGamma
    Lform = rhs * v * dx_neg

    K, F = assemble_form(Equation(a, Lform), dof_handler=dh, bcs=[bc_inactive], backend="jit")

    mapper = AgFEMMapper(dh)
    ag = mapper.build_aggregation_map(level_set, side="-", theta_min=0.999, defined_on=physical)
    assert ag.ghost_eids.size > 0
    cons = mapper.build_constraints(ag, fields=["u"])

    K_red = (cons.E_T @ (K @ cons.E)).tocsr()
    F_red = cons.E_T @ F
    u_master = spla.spsolve(K_red.tocsc(), F_red)
    u_full = cons.E @ u_master

    uh = Function(name="uh", field_name="u", dof_handler=dh)
    gd = dh.get_field_slice("u")
    uh.set_nodal_values(gd, u_full[gd])

    err = (uh - exact) * (uh - exact)
    res = assemble_form(
        Equation(err * dx_neg, None),
        dof_handler=dh,
        assembler_hooks={err: {"name": "err2"}},
        backend="jit",
    )
    err2 = float(np.asarray(res["err2"]).ravel()[0])
    l2 = float(np.sqrt(err2))

    assert np.isfinite(l2)
    # Coarse-mesh sanity: should be in the few-percent range or better.
    assert l2 < 5.0e-2

