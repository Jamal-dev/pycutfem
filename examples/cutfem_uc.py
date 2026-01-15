#!/usr/bin/env python
"""
Unique continuation (elliptic interface) CutFEM demo with CIP/GLS/interface/Tikhonov stabilizations.

This is the PyCutFEM counterpart of the NGSXFEM notebook:
  https://github.com/ngsxfem/ngsxfem-jupyter/blob/94beb9d693b4424e2ea640173c956aaab86faf51/cutfem_uc.ipynb

For mesh-parity comparisons, run the NGSolve reference:
  `examples/debug/ngsolve_cutfem_uc.py`
"""

from __future__ import annotations

import argparse
import os
from math import pi, sqrt

import numpy as np
import scipy.sparse.linalg as spla
import sympy as sp

from pycutfem.core.levelset import LevelSetMeshAdaptation, SuperellipseLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.analytic import Analytic, x, y
from pycutfem.ufl.expressions import (
    FacetNormal,
    Function,
    Laplacian,
    MeshSize,
    Neg,
    Pos,
    TestFunction,
    TrialFunction,
    dot,
    grad,
    inner,
    jump,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dInterface, ds, dx, dCutSkeleton
from pycutfem.utils.bitset import BitSet
from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.dofhandler import DofHandler


def _quad_deg_to_npoints(deg: int) -> int:
    """Map polynomial degree to Gauss-Legendre points per direction (exact for polynomials)."""
    return max(1, int(np.ceil((int(deg) + 1) / 2)))


def _elem_bitset_by_center_box(mesh: Mesh, *, xmin: float, xmax: float, ymin: float, ymax: float) -> BitSet:
    cc = np.asarray(mesh.corner_connectivity, dtype=int)
    XY = np.asarray(mesh.nodes_x_y_pos, dtype=float)
    centers = XY[cc].mean(axis=1)
    mask = (centers[:, 0] >= float(xmin)) & (centers[:, 0] <= float(xmax)) & (centers[:, 1] >= float(ymin)) & (centers[:, 1] <= float(ymax))
    return BitSet(mask)


def _edge_bitset_with_both_neighbors_in(mesh: Mesh, elem_mask: np.ndarray) -> BitSet:
    m = np.zeros((len(mesh.edges_list),), dtype=bool)
    for i, e in enumerate(mesh.edges_list):
        if e.right is None:
            continue
        if bool(elem_mask[int(e.left)]) and bool(elem_mask[int(e.right)]):
            m[int(i)] = True
    return BitSet(m)


def solve_uc(
    *,
    order: int,
    geom_order: int,
    backend: str,
    use_cpp: bool | None = None,
    use_deformation: bool = True,
    q: int | None = None,
    q_degree: int | None = None,
    q_err: int | None = None,
    q_err_degree: int | None = None,
    gamma_data: float = 1.0e5,
    gamma_cip: float = 5.0e-2,
    gamma_gls: float = 5.0e-2,
    gamma_if: float = 1.0e-3,
    alpha0: float = 1.0e-5,
    alpha1: float = 1.0e-2,
) -> float:
    # JIT backend selection (when backend='jit')
    if use_cpp is not None:
        os.environ["PYCUTFEM_JIT_BACKEND"] = "cpp" if use_cpp else "numba"

    # --- mesh: structured quads on [-1.5,1.5]^2 --------------------------------
    L = 3.0
    maxh = 0.125
    nx = int(round(L / maxh))
    ny = nx
    nodes, elems, edges, corners = structured_quad(L, L, nx=nx, ny=ny, poly_order=int(geom_order), offset=(-L / 2, -L / 2))
    mesh = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=int(geom_order))

    omega = _elem_bitset_by_center_box(mesh, xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5)
    B = _elem_bitset_by_center_box(mesh, xmin=-1.25, xmax=1.25, ymin=-1.25, ymax=1.25)

    # --- level set + deformation ----------------------------------------------
    level_set = SuperellipseLevelSet(center=(0.0, 0.0), radius=1.0)
    lsetadap = LevelSetMeshAdaptation(mesh, order=int(order))
    deformation_full = lsetadap.calc_deformation(level_set)
    deformation = deformation_full if bool(use_deformation) else None
    lsetp1 = lsetadap.lset_p1
    if lsetp1 is None:
        raise RuntimeError("LevelSetMeshAdaptation did not build lset_p1.")

    # Classify with the P1 surrogate (mirrors NGSXFEM usage).
    mesh.classify_elements(lsetp1)
    mesh.classify_edges(lsetp1)
    mesh.build_interface_segments(lsetp1)

    inside = mesh.element_bitset("inside")
    outside = mesh.element_bitset("outside")
    cut = mesh.element_bitset("cut")
    hasneg = inside | cut
    haspos = outside | cut

    # CIP facet sets (both neighbors in hasneg / haspos)
    hasneg_mask = np.asarray(hasneg.mask, dtype=bool)
    haspos_mask = np.asarray(haspos.mask, dtype=bool)
    facets_hasneg = _edge_bitset_with_both_neighbors_in(mesh, hasneg_mask)
    facets_haspos = _edge_bitset_with_both_neighbors_in(mesh, haspos_mask)

    # --- FE space --------------------------------------------------------------
    me = MixedElement(mesh, {"u_neg": int(order), "u_pos": int(order), "z": int(order)})
    dh = DofHandler(me, method="cg")

    # Outer boundary tag for z=0
    mesh.tag_boundary_edges({"bc_Omega": lambda x_, y_: True})

    # Compress-like inactive DOFs:
    # - u_neg is only supported on HASNEG; deactivate DOFs whose support lies fully in outside elements.
    # - u_pos is only supported on HASPOS; deactivate DOFs whose support lies fully in inside elements.
    dh.tag_dof_bitset("inactive", "u_neg", outside, strict=True)
    dh.tag_dof_bitset("inactive", "u_pos", inside, strict=True)

    bcs = [
        BoundaryCondition("u_neg", "dirichlet", "inactive", lambda _x, _y: 0.0),
        BoundaryCondition("u_pos", "dirichlet", "inactive", lambda _x, _y: 0.0),
        BoundaryCondition("z", "dirichlet", "bc_Omega", lambda _x, _y: 0.0),
    ]

    # --- manufactured solution -------------------------------------------------
    mu_neg = 2.0
    mu_pos = 20.0
    mu_sum = mu_neg + mu_pos

    r44 = x**4 + y**4
    r41 = sp.sqrt(sp.sqrt(r44))
    u_neg_expr = (1.0 / sp.sqrt(2.0)) * (1.0 + sp.pi * mu_neg / mu_pos) - sp.cos(sp.pi / 4.0 * r44)
    u_pos_expr = (mu_neg / mu_pos) * (sp.pi / sp.sqrt(2.0)) * r41

    f_neg_expr = -mu_neg * (sp.diff(u_neg_expr, x, 2) + sp.diff(u_neg_expr, y, 2))
    f_pos_expr = -mu_pos * (sp.diff(u_pos_expr, x, 2) + sp.diff(u_pos_expr, y, 2))

    u_exact_neg = Analytic(u_neg_expr)
    u_exact_pos = Analytic(u_pos_expr)
    f_neg = Analytic(f_neg_expr)
    f_pos = Analytic(f_pos_expr)

    # --- unknowns --------------------------------------------------------------
    u_neg = TrialFunction("u_neg", dof_handler=dh)
    u_pos = TrialFunction("u_pos", dof_handler=dh)
    z = TrialFunction("z", dof_handler=dh)

    v_neg = TestFunction("u_neg", dof_handler=dh)
    v_pos = TestFunction("u_pos", dof_handler=dh)
    w = TestFunction("z", dof_handler=dh)

    # --- measures --------------------------------------------------------------
    if q is None and q_degree is None:
        q = 2 * int(order) + 2
    if q_degree is not None:
        q = _quad_deg_to_npoints(int(q_degree))
    if q is None:
        raise ValueError("q is None after resolving q/q_degree.")

    dx_neg = dx(level_set=lsetp1, deformation=deformation, metadata={"side": "-", "q": int(q)})
    dx_pos = dx(level_set=lsetp1, deformation=deformation, metadata={"side": "+", "q": int(q)})
    dGamma = dInterface(level_set=lsetp1, deformation=deformation, metadata={"q": int(q)})

    # --- primal/dual main coupling --------------------------------------------
    n = FacetNormal()
    h = MeshSize()

    kappa_neg = mu_pos / mu_sum
    kappa_pos = mu_neg / mu_sum
    avg_flux_w = -(kappa_neg * mu_neg * dot(grad(Neg(w)), n) + kappa_pos * mu_pos * dot(grad(Pos(w)), n))
    avg_flux_z = -(kappa_neg * mu_neg * dot(grad(Neg(z)), n) + kappa_pos * mu_pos * dot(grad(Pos(z)), n))

    jump_u = Neg(u_neg) - Pos(u_pos)
    jump_v = Neg(v_neg) - Pos(v_pos)

    a = mu_neg * inner(grad(u_neg), grad(w)) * dx_neg
    a += mu_pos * inner(grad(u_pos), grad(w)) * dx_pos
    a += avg_flux_w * jump_u * dGamma

    a += mu_neg * inner(grad(v_neg), grad(z)) * dx_neg
    a += mu_pos * inner(grad(v_pos), grad(z)) * dx_pos
    a += avg_flux_z * jump_v * dGamma

    L = f_neg * w * dx_neg + f_pos * w * dx_pos

    # --- data constraint on omega ---------------------------------------------
    dx_neg_omega = dx(defined_on=omega, level_set=lsetp1, deformation=deformation, metadata={"side": "-", "q": q})
    dx_pos_omega = dx(defined_on=omega, level_set=lsetp1, deformation=deformation, metadata={"side": "+", "q": q})
    a += gamma_data * u_neg * v_neg * dx_neg_omega
    a += gamma_data * u_pos * v_pos * dx_pos_omega
    L += gamma_data * u_exact_neg * v_neg * dx_neg_omega
    L += gamma_data * u_exact_pos * v_pos * dx_pos_omega

    # --- CIP (continuous interior penalty) ------------------------------------
    nF = FacetNormal()
    dsk_neg = dCutSkeleton(defined_on=facets_hasneg, level_set=lsetp1, deformation=deformation, metadata={"side": "-", "q": q})
    dsk_pos = dCutSkeleton(defined_on=facets_haspos, level_set=lsetp1, deformation=deformation, metadata={"side": "+", "q": q})

    a += gamma_cip * h * mu_neg * jump(dot(grad(u_neg), nF)) * jump(dot(grad(v_neg), nF)) * dsk_neg
    a += gamma_cip * h * mu_pos * jump(dot(grad(u_pos), nF)) * jump(dot(grad(v_pos), nF)) * dsk_pos

    # --- GLS ------------------------------------------------------------------
    Lu_neg = -mu_neg * Laplacian(u_neg)
    Lu_pos = -mu_pos * Laplacian(u_pos)
    Lv_neg = -mu_neg * Laplacian(v_neg)
    Lv_pos = -mu_pos * Laplacian(v_pos)

    a += gamma_gls * (h * h) * Lu_neg * Lv_neg * dx_neg
    a += gamma_gls * (h * h) * Lu_pos * Lv_pos * dx_pos
    L += gamma_gls * (h * h) * f_neg * Lv_neg * dx_neg
    L += gamma_gls * (h * h) * f_pos * Lv_pos * dx_pos

    # --- interface stabilization ----------------------------------------------
    mubar = 0.5 * (mu_neg + mu_pos)

    flux_jump_u = dot(mu_neg * grad(Neg(u_neg)) - mu_pos * grad(Pos(u_pos)), n)
    flux_jump_v = dot(mu_neg * grad(Neg(v_neg)) - mu_pos * grad(Pos(v_pos)), n)

    # Tangential gradient jump:
    #   P(a) = a - (a·n)n, so P(a)·P(b) = a·b - (a·n)(b·n) (since |n|=1).
    grad_jump_u = grad(Neg(u_neg)) - grad(Pos(u_pos))
    grad_jump_v = grad(Neg(v_neg)) - grad(Pos(v_pos))
    tan_jump_inner = inner(grad_jump_u, grad_jump_v) - dot(grad_jump_u, n) * dot(grad_jump_v, n)

    a += gamma_if * (mubar / h) * jump_u * jump_v * dGamma
    a += gamma_if * h * flux_jump_u * flux_jump_v * dGamma
    a += gamma_if * mubar * h * tan_jump_inner * dGamma

    # --- (weak) Tikhonov stabilization ----------------------------------------
    dx_geom_neg = dx(defined_on=hasneg, metadata={"q": q})
    dx_geom_pos = dx(defined_on=haspos, metadata={"q": q})
    power = 2 * int(order)

    a += alpha0 * (h ** power) * u_neg * v_neg * dx_geom_neg
    a += alpha0 * (h ** power) * u_pos * v_pos * dx_geom_pos
    a += alpha1 * (h ** power) * inner(grad(u_neg), grad(v_neg)) * dx_geom_neg
    a += alpha1 * (h ** power) * inner(grad(u_pos), grad(v_pos)) * dx_geom_pos

    # --- dual stabilization ----------------------------------------------------
    a += -mu_neg * inner(grad(z), grad(w)) * dx_neg
    a += -mu_pos * inner(grad(z), grad(w)) * dx_pos

    K, F = assemble_form(Equation(a, L), dof_handler=dh, bcs=bcs, backend=backend)
    sol = spla.spsolve(K.tocsc(), F)

    uh_neg = Function(name="uh_neg", field_name="u_neg", dof_handler=dh)
    uh_pos = Function(name="uh_pos", field_name="u_pos", dof_handler=dh)
    uh_neg.set_nodal_values(dh.get_field_slice("u_neg"), sol[dh.get_field_slice("u_neg")])
    uh_pos.set_nodal_values(dh.get_field_slice("u_pos"), sol[dh.get_field_slice("u_pos")])

    # error on B
    if q_err is None and q_err_degree is None:
        q_err = 2 * int(order)
    if q_err_degree is not None:
        q_err = _quad_deg_to_npoints(int(q_err_degree))
    if q_err is None:
        raise ValueError("q_err is None after resolving q_err/q_err_degree.")

    dx_neg_B = dx(defined_on=B, level_set=lsetp1, deformation=deformation, metadata={"side": "-", "q": q_err})
    dx_pos_B = dx(defined_on=B, level_set=lsetp1, deformation=deformation, metadata={"side": "+", "q": q_err})
    err_neg = (uh_neg - u_exact_neg) * (uh_neg - u_exact_neg)
    err_pos = (uh_pos - u_exact_pos) * (uh_pos - u_exact_pos)
    err_form = err_neg * dx_neg_B + err_pos * dx_pos_B
    res = assemble_form(
        Equation(err_form, None),
        dof_handler=dh,
        assembler_hooks={err_neg: {"name": "err2"}, err_pos: {"name": "err2"}},
        backend=backend,
    )
    err2 = float(np.asarray(res["err2"]).ravel()[0])
    return float(np.sqrt(err2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("python", "jit"), default="jit")
    p.add_argument("--jit-backend", choices=("cpp", "numba"), default=None)
    p.add_argument("--order", type=int, default=2)
    p.add_argument("--geom-order", type=int, default=None, help="Geometry (mapping) order. Defaults to --order.")
    p.add_argument("--q", type=int, default=None, help="Assembly quadrature: Gauss points per direction (quad).")
    p.add_argument("--q-degree", type=int, default=None, help="Assembly quadrature: polynomial degree (quad). Converted internally to Gauss points.")
    p.add_argument("--q-err", type=int, default=None, help="Error quadrature: Gauss points per direction (quad).")
    p.add_argument("--q-err-degree", type=int, default=None, help="Error quadrature: polynomial degree (quad). Converted internally to Gauss points.")
    p.add_argument("--gamma-data", type=float, default=1.0e5)
    p.add_argument("--gamma-cip", type=float, default=5.0e-2)
    p.add_argument("--gamma-gls", type=float, default=5.0e-2)
    p.add_argument("--gamma-if", type=float, default=1.0e-3)
    p.add_argument("--no-deformation", action="store_true", help="Disable isoparametric mesh deformation (debug).")
    args = p.parse_args()

    geom_order = int(args.order) if args.geom_order is None else int(args.geom_order)

    use_cpp = None
    if args.jit_backend is not None:
        use_cpp = str(args.jit_backend) == "cpp"

    l2 = solve_uc(
        order=int(args.order),
        geom_order=geom_order,
        backend=str(args.backend),
        use_cpp=use_cpp,
        use_deformation=not bool(args.no_deformation),
        q=args.q,
        q_degree=args.q_degree,
        q_err=args.q_err,
        q_err_degree=args.q_err_degree,
        gamma_data=float(args.gamma_data),
        gamma_cip=float(args.gamma_cip),
        gamma_gls=float(args.gamma_gls),
        gamma_if=float(args.gamma_if),
    )
    b = str(args.backend)
    jb = os.getenv("PYCUTFEM_JIT_BACKEND", "") if b == "jit" else "-"
    print(f"PyCutFEM UC L2 error ({b}, jit={jb}, geom_order={geom_order}): {l2:.16e}")


if __name__ == "__main__":
    main()
