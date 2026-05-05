#!/usr/bin/env python
"""
XFEM moving-domain runner using the gmsh background mesh.

Imports the gmsh mesh with ``netgen.read_gmsh.ReadGmsh`` so that the same
background grid can be shared with the pycutfem driver. Outputs final error
norms in a machine-readable line for the comparison script.
"""
from __future__ import annotations

import argparse
import json
from math import pi, ceil, sqrt, cos, sin
import numpy as np

from netgen.read_gmsh import ReadGmsh
from ngsolve import Mesh, H1, GridFunction, Parameter, CoefficientFunction, InnerProduct, Grad, Integrate, specialcf, sqrt as ngs_sqrt, x, y, dx as ngs_dx, BitArray, LinearForm
from ngsolve import ngsglobals, Draw, Redraw, SetNumThreads
from ngsolve import sin as ngsin, cos as ngcos
from xfem import (
    CutInfo,
    GetFacetsWithNeighborTypes,
    GetDofsOfElements,
    dCut,
    dFacetPatch,
    HASNEG,
    NEG,
    RestrictedBilinearForm,
)
from xfem.lsetcurv import LevelSetMeshAdaptation, InterpolateToP1

ngsglobals.msg_level = 1
SetNumThreads(2)

r0 = 0.5
r1 = pi / (2 * r0)
nu = 1e-5
c_gamma = 1.0
velmax = 2.0


def levelset_expr(rho_param):
    return ngs_sqrt((x - rho_param) ** 2 + y**2) - r0


def rhs_expr(rho_param):
    rr = ngs_sqrt((x - rho_param) ** 2 + y**2)
    return nu * (
        -(pi / r0) * r1 * (ngsin(r1 * rr) ** 2 - ngcos(r1 * rr) ** 2)
        + (pi / r0) * ngcos(r1 * rr) * ngsin(r1 * rr) * (1 / rr)
    )


def u_exact_expr(rho_param):
    rr = ngs_sqrt((x - rho_param) ** 2 + y**2)
    return ngcos(r1 * rr) ** 2


def grad_u_exact_expr(rho_param):
    rr = ngs_sqrt((x - rho_param) ** 2 + y**2)
    return CoefficientFunction(
        (
            -pi * ngsin(pi / r0 * rr) * (x - rho_param) / rr,
            -pi * ngsin(pi / r0 * rr) * y / rr,
        )
    )


def edge_length_max(ngmesh) -> float:
    pts = np.array([p.p for p in ngmesh.Points()], dtype=float)  # 1-based indexing
    h_max = 0.0
    for e in ngmesh.Elements1D():
        p0 = pts[int(e.points[0].nr) - 1][:2]
        p1 = pts[int(e.points[1].nr) - 1][:2]
        h_max = max(h_max, float(np.linalg.norm(p0 - p1)))
    return h_max


def run(args):
    ngmesh = ReadGmsh(str(args.mesh_file))
    for _ in range(args.Lx):
        ngmesh.Refine()
    mesh = Mesh(ngmesh)
    h_max = edge_length_max(ngmesh)

    V = H1(mesh, order=args.k, dgjumps=True)
    gfu = GridFunction(V)

    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=args.k, threshold=0.1, discontinuous_qn=True)
    deformation = lsetmeshadap.deform
    lsetp1 = lsetmeshadap.lset_p1

    ci_main = CutInfo(mesh)
    ci_inner = CutInfo(mesh)
    ci_outer = CutInfo(mesh)

    els_hasneg = ci_main.GetElementsOfType(HASNEG)
    els_outer = ci_outer.GetElementsOfType(HASNEG)
    els_inner = ci_inner.GetElementsOfType(NEG)

    els_ring = BitArray(mesh.ne)
    facets_ring = BitArray(mesh.nedge)
    els_outer_old = BitArray(mesh.ne)
    els_test = BitArray(mesh.ne)

    t_param = Parameter(0.0)
    w_param = Parameter(0.0)
    rho_param = Parameter(0.0)
    dt = args.t0 * 0.5 ** args.Lt
    T_end = args.T_end
    delta = dt * velmax
    K_tilde = int(ceil(delta / h_max))
    gamma_s = c_gamma * K_tilde

    u, v = V.TnT()
    h = specialcf.mesh_size

    dx_neg = dCut(levelset=lsetp1, domain_type=NEG, definedonelements=els_hasneg, deformation=deformation)
    dw = dFacetPatch(definedonelements=facets_ring, deformation=deformation)

    a = RestrictedBilinearForm(V, element_restriction=els_outer, facet_restriction=facets_ring, check_unused=False)
    w_cf = CoefficientFunction((w_param, 0))
    a += (1 / dt) * u * v * dx_neg
    a += nu * InnerProduct(Grad(u), Grad(v)) * dx_neg
    a += (InnerProduct(w_cf, Grad(u)) * v) * dx_neg
    a += gamma_s * (1 / h**2) * (u - u.Other()) * (v - v.Other()) * dw

    rhs_cf = rhs_expr(rho_param)
    f = LinearForm(V)
    f += rhs_cf * v * dx_neg
    f += (1 / dt) * gfu * v * dx_neg

    errors_L2 = []
    errors_H1 = []
    dx_2k = dx_neg.order(2 * args.k)

    def comp_errs():
        l2 = sqrt(Integrate((gfu - u_ex_cf) ** 2 * dx_2k, mesh))
        h1 = sqrt(Integrate((Grad(gfu) - grad_u_ex_cf) ** 2 * dx_2k, mesh))
        errors_L2.append(l2)
        errors_H1.append(h1)
        return l2, h1

    lset_cf = levelset_expr(rho_param)
    rhs_cf = rhs_expr(rho_param)
    u_ex_cf = u_exact_expr(rho_param)
    grad_u_ex_cf = grad_u_exact_expr(rho_param)

    lsetmeshadap.ProjectOnUpdate(gfu)
    gfu.Set(u_ex_cf)
    els_outer_old.Set()

    for it in range(1, int(T_end / dt + 0.5) + 1):
        t_param.Set(it * dt)
        w_param.Set(2 * cos(2 * pi * t_param.Get()))
        rho_param.Set(1.0 / pi * sin(2 * pi * t_param.Get()))
        deformation = lsetmeshadap.CalcDeformation(lset_cf)

        InterpolateToP1(lset_cf - delta, lsetp1)
        ci_outer.Update(lsetp1)
        InterpolateToP1(lset_cf + delta, lsetp1)
        ci_inner.Update(lsetp1)

        els_ring.Clear()
        els_ring |= els_outer & ~els_inner
        facets_ring.Clear()
        facets_ring |= GetFacetsWithNeighborTypes(mesh, a=els_outer, b=els_ring)

        InterpolateToP1(lset_cf, lsetp1)
        ci_main.Update(lsetp1)

        active_dofs = GetDofsOfElements(V, els_outer)

        els_test[:] = els_hasneg & ~els_outer_old
        if sum(els_test) != 0:
            raise RuntimeError("Some active elements do not have a history")
        els_outer_old[:] = els_outer

        a.Assemble(reallocate=True)
        f.Assemble()
        inv = a.mat.Inverse(active_dofs, inverse="")
        gfu.vec.data = inv * f.vec

        l2err, h1err = comp_errs()
        print(f"Lx={args.Lx}, dt={dt:8.6f}, t={t_param.Get():8.6f}, L2={l2err:5.3e}, active={sum(els_outer)}, K={K_tilde}")

        if args.draw:
            Draw(gfu, mesh, "gfu")
            Redraw(blocking=False)

    err_l2l2 = float(sqrt(dt * sum([e ** 2 for e in errors_L2])))
    err_l2h1 = float(sqrt(dt * sum([e ** 2 for e in errors_H1])))
    linf_l2 = float(max(errors_L2))

    print("XFEM summary:")
    print(f"L2(0,T;L2) = {err_l2l2:6.4e}")
    print(f"Linf(0,T;L2) = {linf_l2:6.4e}")
    print(f"L2(0,T;H1) = {err_l2h1:6.4e}")

    if args.output_json:
        payload = {
            "Lx": args.Lx,
            "Lt": args.Lt,
            "h_max": h_max,
            "dt": dt,
            "n_steps": int(T_end / dt + 0.5),
            "err_l2l2": err_l2l2,
            "err_l2h1": err_l2h1,
            "linf_l2": linf_l2,
            "gamma_s": gamma_s,
            "delta": delta,
            "mesh_file": str(args.mesh_file),
            "poly_order": args.k,
            "step_errors": [],
        }
        print("RESULT_JSON", json.dumps(payload))


def _parse():
    ap = argparse.ArgumentParser(description="Run xfem moving_domain with a gmsh mesh.")
    ap.add_argument("--mesh-file", type=str, required=True)
    ap.add_argument("--k", type=int, default=1, help="FE order.")
    ap.add_argument("--poly-order", type=int, dest="k", help="Alias for --k.")
    ap.add_argument("--h0", type=float, default=0.2)
    ap.add_argument("--t0", type=float, default=0.1)
    ap.add_argument("--Lx", type=int, default=3)
    ap.add_argument("--Lt", type=int, default=3)
    ap.add_argument("--T-end", type=float, default=0.4, dest="T_end")
    ap.add_argument("--draw", action="store_true", help="Enable NGSolve draw windows.")
    ap.add_argument("--output-json", action="store_true", help="Emit RESULT_JSON line for parsing.")
    return ap.parse_args()


if __name__ == "__main__":
    run(_parse())
