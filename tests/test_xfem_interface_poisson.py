import numpy as np
import pytest
import scipy.sparse.linalg as spla
import sympy as sp

from pycutfem.core.geometry import hansbo_cut_ratio
from pycutfem.core.levelset import SuperellipseLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.meshgen import structured_quad
from pycutfem.xfem import XFEMDofHandler

from pycutfem.ufl.analytic import Analytic, x, y
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    ElementWiseConstant,
    FacetNormal,
    Function,
    Neg,
    Pos,
    TestFunction,
    TrialFunction,
    dot,
    grad,
    inner,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dInterface, dx


def test_xfem_interface_nitsche_smoke(monkeypatch):
    """
    Regression smoke test for interface Nitsche terms on an XFEM space.

    This mirrors the NGSXFEM `mpi_nxfem.py` manufactured solution but runs
    on a structured quad mesh and exercises the C++ JIT backend.
    """
    monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")

    ll = (-1.5, -1.5)
    L = 3.0
    maxh = 0.25
    nx = ny = int(round(L / maxh))
    nodes, elems, edges, corners = structured_quad(L, L, nx=nx, ny=ny, poly_order=1, offset=ll)
    mesh = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=1)

    degree = 2
    me = MixedElement(mesh, {"u": degree})
    dh0 = DofHandler(me, method="cg")

    level_set = SuperellipseLevelSet(center=(0.0, 0.0), radius=1.0)
    dh0.classify_from_levelset(level_set)
    assert mesh.element_bitset("cut").cardinality() > 0

    mesh.tag_boundary_edges({"boundary": lambda x_, y_: True})

    dh = XFEMDofHandler(dh0)
    dh.rebuild_enrichment(level_set, enrich={"u": "heaviside"})
    assert dh.total_dofs > dh.base_total_dofs

    r44 = x**4 + y**4
    r41 = sp.sqrt(sp.sqrt(r44))

    u_neg_expr = 1 + sp.pi / 2 - sp.sqrt(2.0) * sp.cos(sp.pi / 4 * r44)
    u_pos_expr = (sp.pi / 2) * r41

    alpha_neg = 1.0
    alpha_pos = 2.0
    f_neg_expr = -alpha_neg * (sp.diff(u_neg_expr, x, 2) + sp.diff(u_neg_expr, y, 2))
    f_pos_expr = -alpha_pos * (sp.diff(u_pos_expr, x, 2) + sp.diff(u_pos_expr, y, 2))

    u_neg = Analytic(u_neg_expr)
    u_pos = Analytic(u_pos_expr)
    f_neg = Analytic(f_neg_expr)
    f_pos = Analytic(f_pos_expr)

    u_pos_fun = sp.lambdify((x, y), u_pos_expr, "numpy")
    bc_boundary = BoundaryCondition(
        "u",
        "dirichlet",
        "boundary",
        lambda xx, yy: float(u_pos_fun(float(xx), float(yy))),
    )

    theta_pos = hansbo_cut_ratio(mesh, level_set, side="+")
    theta_neg = hansbo_cut_ratio(mesh, level_set, side="-")
    kappa_pos = ElementWiseConstant(theta_pos)
    kappa_neg = ElementWiseConstant(theta_neg)

    u = TrialFunction("u", dof_handler=dh)
    v = TestFunction("u", dof_handler=dh)
    n = FacetNormal()
    h = CellDiameter()

    q = 2 * degree + 2
    dx_pos = dx(level_set=level_set, metadata={"side": "+", "q": q})
    dx_neg = dx(level_set=level_set, metadata={"side": "-", "q": q})
    dGamma = dInterface(defined_on=mesh.element_bitset("cut"), level_set=level_set, metadata={"q": q})

    a = Constant(alpha_pos) * inner(grad(u), grad(v)) * dx_pos
    a += Constant(alpha_neg) * inner(grad(u), grad(v)) * dx_neg

    flux_u_pos = -Constant(alpha_pos) * dot(grad(Pos(u)), n)
    flux_u_neg = -Constant(alpha_neg) * dot(grad(Neg(u)), n)
    flux_v_pos = -Constant(alpha_pos) * dot(grad(Pos(v)), n)
    flux_v_neg = -Constant(alpha_neg) * dot(grad(Neg(v)), n)
    avg_flux_u = kappa_pos * flux_u_pos + kappa_neg * flux_u_neg
    avg_flux_v = kappa_pos * flux_v_pos + kappa_neg * flux_v_neg
    jump_u = Neg(u) - Pos(u)
    jump_v = Neg(v) - Pos(v)

    stab = Constant(20.0 * (alpha_pos + alpha_neg)) / h
    a += (avg_flux_u * jump_v + avg_flux_v * jump_u + stab * jump_u * jump_v) * dGamma

    Lform = f_pos * v * dx_pos + f_neg * v * dx_neg
    K, F = assemble_form(Equation(a, Lform), dof_handler=dh, bcs=[bc_boundary], backend="jit")
    sol = spla.spsolve(K.tocsc(), F)

    uh = Function(name="uh", field_name="u", dof_handler=dh)
    gd = dh.get_field_slice("u")
    uh.set_nodal_values(gd, sol[gd])

    err_pos = (uh - u_pos) * (uh - u_pos)
    err_neg = (uh - u_neg) * (uh - u_neg)
    err_form = err_pos * dx_pos + err_neg * dx_neg
    res = assemble_form(
        Equation(err_form, None),
        dof_handler=dh,
        assembler_hooks={err_pos: {"name": "err2"}, err_neg: {"name": "err2"}},
        backend="jit",
    )
    err2 = float(np.asarray(res["err2"]).ravel()[0])
    l2 = float(np.sqrt(err2))

    assert np.isfinite(l2)
    # Coarse-mesh sanity: XFEM+Nitsche should be accurate at the percent level.
    assert l2 < 5.0e-2

