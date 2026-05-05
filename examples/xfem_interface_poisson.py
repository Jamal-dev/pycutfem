#!/usr/bin/env python
# coding: utf-8

"""
XFEM interface demo mirroring NGSXFEM `mpi_nxfem.py` (serial run).

Solve a scalar interface problem on Ω=[-1.5,1.5]^2 with an L^4 level-set:

  φ(x,y) = (x^4 + y^4)^(1/4) - 1

Diffusion coefficients are piecewise-constant on Ω⁻ (φ<0) / Ω⁺ (φ>0).
We use a symmetric Nitsche interface formulation with Hansbo weights κᵖ/κⁿ
and an XFEM enrichment (shifted Heaviside) on cut elements.
"""

from __future__ import annotations

import argparse
from math import pi

import numpy as np
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


def solve_interface_xfem(*, maxh: float, degree: int, backend: str) -> float:
    # --- domain / mesh ---------------------------------------------------------
    ll = (-1.5, -1.5)
    L = 3.0
    nx = max(4, int(round(L / float(maxh))))
    ny = nx
    nodes, elems, edges, corners = structured_quad(L, L, nx=nx, ny=ny, poly_order=1, offset=ll)
    mesh = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=1)

    me = MixedElement(mesh, {"u": int(degree)})
    dh0 = DofHandler(me, method="cg")

    level_set = SuperellipseLevelSet(center=(0.0, 0.0), radius=1.0)
    dh0.classify_from_levelset(level_set)

    # Tag all boundary edges with a single tag for Dirichlet elimination.
    mesh.tag_boundary_edges({"boundary": lambda x_, y_: True})

    dh = XFEMDofHandler(dh0)
    dh.rebuild_enrichment(level_set, enrich={"u": "heaviside"})

    # --- manufactured solution ------------------------------------------------
    r44 = x**4 + y**4
    r41 = sp.sqrt(sp.sqrt(r44))

    # NGSXFEM demo convention: NEG first, POS second.
    u_neg_expr = 1 + pi / 2 - sp.sqrt(2.0) * sp.cos(pi / 4 * r44)
    u_pos_expr = (pi / 2) * r41

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

    # --- Hansbo weights (per element) ----------------------------------------
    theta_pos = hansbo_cut_ratio(mesh, level_set, side="+")
    theta_neg = hansbo_cut_ratio(mesh, level_set, side="-")
    kappa_pos = ElementWiseConstant(theta_pos)
    kappa_neg = ElementWiseConstant(theta_neg)

    # --- weak form ------------------------------------------------------------
    u = TrialFunction("u", dof_handler=dh)
    v = TestFunction("u", dof_handler=dh)
    n = FacetNormal()
    h = CellDiameter()

    q = 2 * int(degree) + 2
    dx_pos = dx(level_set=level_set, metadata={"side": "+", "q": q})
    dx_neg = dx(level_set=level_set, metadata={"side": "-", "q": q})
    dGamma = dInterface(defined_on=mesh.element_bitset("cut"), level_set=level_set, metadata={"q": q})

    a = Constant(alpha_pos) * inner(grad(u), grad(v)) * dx_pos
    a += Constant(alpha_neg) * inner(grad(u), grad(v)) * dx_neg

    # Physical normal flux q = -α ∇u·n (mirrors NGSXFEM)
    flux_u_pos = -Constant(alpha_pos) * dot(grad(Pos(u)), n)
    flux_u_neg = -Constant(alpha_neg) * dot(grad(Neg(u)), n)
    flux_v_pos = -Constant(alpha_pos) * dot(grad(Pos(v)), n)
    flux_v_neg = -Constant(alpha_neg) * dot(grad(Neg(v)), n)

    avg_flux_u = kappa_pos * flux_u_pos + kappa_neg * flux_u_neg
    avg_flux_v = kappa_pos * flux_v_pos + kappa_neg * flux_v_neg

    jump_u = Neg(u) - Pos(u)
    jump_v = Neg(v) - Pos(v)

    lambda_nitsche = 20.0
    stab = Constant(lambda_nitsche * (alpha_pos + alpha_neg)) / h
    a += (avg_flux_u * jump_v + avg_flux_v * jump_u + stab * jump_u * jump_v) * dGamma

    L = f_pos * v * dx_pos + f_neg * v * dx_neg

    K, F = assemble_form(Equation(a, L), dof_handler=dh, bcs=[bc_boundary], backend=backend)
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
        backend=backend,
    )
    err2 = float(np.asarray(res["err2"]).ravel()[0])
    return float(np.sqrt(err2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("python", "jit"), default="jit")
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--maxh", type=float, default=0.2)
    args = p.parse_args()

    l2 = solve_interface_xfem(maxh=float(args.maxh), degree=int(args.degree), backend=str(args.backend))
    print(f"L2 error : {l2:.16e}")


if __name__ == "__main__":
    main()

