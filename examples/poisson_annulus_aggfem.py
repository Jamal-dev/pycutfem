#!/usr/bin/env python
# coding: utf-8

"""
AgFEM (element aggregation) demo mirroring NGSXFEM `fictdom_aggfem.py`.

Solve -Δu = f on an annulus Ω = { r1 < sqrt(x^2+y^2) < r2 } embedded in [-1,1]^2
with homogeneous Dirichlet BC on ∂Ω enforced by a Nitsche interface term.

We eliminate DOFs on Ωᶜ (outside elements) and apply AgFEM-style aggregation
constraints on *cut* elements to improve conditioning.
"""

from __future__ import annotations

import argparse

import numpy as np
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


def solve_annulus_agfem(*, maxh: float, degree: int, theta_min: float, backend: str) -> float:
    # --- domain / mesh ---------------------------------------------------------
    ll = (-1.0, -1.0)
    L = 2.0
    nx = max(4, int(round(L / float(maxh))))
    ny = nx
    nodes, elems, edges, corners = structured_quad(L, L, nx=nx, ny=ny, poly_order=1, offset=ll)
    mesh = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=1)

    me = MixedElement(mesh, {"u": int(degree)})
    dh = DofHandler(me, method="cg")

    # --- level set / active domain -------------------------------------------
    r1, r2 = 1.0 / 4.0, 3.0 / 4.0
    level_set = AnnulusLevelSet(center=(0.0, 0.0), r_inner=r1, r_outer=r2)
    dh.classify_from_levelset(level_set)

    inside = mesh.element_bitset("inside")
    cut = mesh.element_bitset("cut")
    outside = mesh.element_bitset("outside")
    physical = inside | cut

    # Deactivate DOFs on elements fully outside Ω (mimics FE_Nothing restriction).
    dh.tag_dof_bitset("inactive", "u", elem_mask=outside, strict=True)
    bc_inactive = BoundaryCondition("u", "dirichlet", "inactive", lambda x_, y_: 0.0)

    # --- manufactured solution ------------------------------------------------
    r = sp.sqrt(x**2 + y**2)
    exact_expr = 20.0 * (float(r2) - r) * (r - float(r1))
    rhs_expr = -sp.diff(exact_expr, x, 2) - sp.diff(exact_expr, y, 2)
    exact = Analytic(exact_expr)
    rhs = Analytic(rhs_expr)

    # --- weak form ------------------------------------------------------------
    u = TrialFunction("u", dof_handler=dh)
    v = TestFunction("u", dof_handler=dh)
    n = FacetNormal()
    h = CellDiameter()

    q = 2 * int(degree) + 2
    dx_neg = dx(defined_on=physical, level_set=level_set, metadata={"side": "-", "q": q})
    dGamma = dInterface(defined_on=cut, level_set=level_set, metadata={"q": q})

    lambda_nitsche = 10.0 * float(degree) * float(degree)
    gamma = Constant(lambda_nitsche)

    a = inner(grad(u), grad(v)) * dx_neg
    a += -dot(grad(u), n) * v * dGamma
    a += -dot(grad(v), n) * u * dGamma
    a += (gamma / h) * u * v * dGamma

    L = rhs * v * dx_neg

    K, F = assemble_form(Equation(a, L), dof_handler=dh, bcs=[bc_inactive], backend=backend)

    # --- AgFEM constraints & solve -------------------------------------------
    mapper = AgFEMMapper(dh)
    ag = mapper.build_aggregation_map(level_set, side="-", theta_min=float(theta_min), defined_on=physical)
    cons = mapper.build_constraints(ag, fields=["u"])

    K_red = (cons.E_T @ (K @ cons.E)).tocsr()
    F_red = cons.E_T @ F

    u_master = spla.spsolve(K_red.tocsc(), F_red)
    u_full = cons.E @ u_master

    # --- error ---------------------------------------------------------------
    uh = Function(name="uh", field_name="u", dof_handler=dh)
    gd = dh.get_field_slice("u")
    uh.set_nodal_values(gd, u_full[gd])

    err = (uh - exact) * (uh - exact)
    res = assemble_form(
        Equation(err * dx_neg, None),
        dof_handler=dh,
        assembler_hooks={err: {"name": "err2"}},
        backend=backend,
    )
    err2 = float(np.asarray(res["err2"]).ravel()[0])
    return float(np.sqrt(err2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("python", "jit"), default="jit")
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--maxh", type=float, default=0.05)
    p.add_argument("--theta-min", type=float, default=0.999)
    args = p.parse_args()

    l2 = solve_annulus_agfem(
        maxh=float(args.maxh),
        degree=int(args.degree),
        theta_min=float(args.theta_min),
        backend=str(args.backend),
    )
    print(f"L2 Error: {l2:.16e}")


if __name__ == "__main__":
    main()

