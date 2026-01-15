#!/usr/bin/env python
"""
NGSolve/NGSXFEM reference for the unique-continuation CutFEM example (CIP/GLS/IF/Tikhonov).

This mirrors the notebook:
  https://github.com/ngsxfem/ngsxfem-jupyter/blob/94beb9d693b4424e2ea640173c956aaab86faf51/cutfem_uc.ipynb

Differences:
- Uses a structured quad mesh (same geometry as the PyCutFEM reference) so that we can
  compare results on identical meshes.
- Uses element BitArrays (selected by coordinates) instead of Netgen materials for
  omega/B subdomains.

Run (inside xfemcustom env)
---------------------------
conda run --no-capture-output -n xfemcustom \\
  python examples/debug/ngsolve_cutfem_uc.py --order 2
"""

from __future__ import annotations

import argparse
from math import pi, sqrt

import numpy as np
from ngsolve import (
    BitArray,
    BilinearForm,
    Compress,
    GridFunction,
    H1,
    Integrate,
    InnerProduct,
    LinearForm,
    cos,
    dx,
    grad,
    specialcf,
    sqrt as ngsqrt,
    x,
    y,
)
from ngsolve.meshes import MakeStructured2DMesh
from xfem import HAS, HASNEG, HASPOS, IF, NEG, POS, CutInfo, GetDofsOfElements, GetFacetsWithNeighborTypes, dCut
from xfem.lsetcurv import LevelSetMeshAdaptation


def _elem_centers_bitarray(mesh: Mesh, *, xmin: float, xmax: float, ymin: float, ymax: float) -> BitArray:
    ngm = mesh.ngmesh
    pts = [p.p for p in ngm.Points()]  # list[(x,y,z)], 0-based python list
    out = BitArray(mesh.ne)
    out.Clear()
    for ei, el in enumerate(ngm.Elements2D()):
        vs = el.vertices  # point numbers (1-based)
        coords = np.asarray([pts[int(v.nr) - 1] for v in vs], dtype=float)
        cx, cy = coords[:, 0].mean(), coords[:, 1].mean()
        if (xmin <= cx <= xmax) and (ymin <= cy <= ymax):
            out.Set(int(ei))
    return out


def solve_uc(
    *,
    order: int,
    geom_order: int,
    intorder: int | None,
    use_deformation: bool,
    gamma_data: float,
    gamma_cip: float,
    gamma_gls: float,
    gamma_if: float,
    alpha0: float,
    alpha1: float,
) -> float:
    # --- mesh: structured quads on [-1.5,1.5]^2 --------------------------------
    L = 3.0
    maxh = 0.125
    nx = int(round(L / maxh))
    ny = nx
    mesh = MakeStructured2DMesh(
        quads=True,
        nx=nx,
        ny=ny,
        mapping=lambda xx, yy: (L * float(xx) - 0.5 * L, L * float(yy) - 0.5 * L),
    )
    if int(geom_order) > 1:
        mesh.Curve(int(geom_order))

    # coordinate-selected subdomains (aligned with mesh for maxh=0.125)
    omega_elems = _elem_centers_bitarray(mesh, xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5)
    B_elems = _elem_centers_bitarray(mesh, xmin=-1.25, xmax=1.25, ymin=-1.25, ymax=1.25)

    # --- level set: L^4 ball ---------------------------------------------------
    r44 = x**4 + y**4
    r41 = ngsqrt(ngsqrt(r44))
    levelset = r41 - 1.0

    lsetadap = LevelSetMeshAdaptation(mesh, order=int(order), levelset=levelset)
    lsetp1 = lsetadap.lset_p1
    ci = CutInfo(mesh, lsetp1)

    # --- manufactured solution -------------------------------------------------
    mu = (2.0, 20.0)  # (NEG, POS)
    solution = [
        (1.0 / sqrt(2.0)) * (1.0 + pi * mu[0] / mu[1]) - cos(pi / 4.0 * r44),
        (mu[0] / mu[1]) * (pi / sqrt(2.0)) * r41,
    ]
    coef_f = [-mu[i] * (solution[i].Diff(x).Diff(x) + solution[i].Diff(y).Diff(y)) for i in range(2)]

    # --- FE spaces -------------------------------------------------------------
    Vh = H1(mesh, order=int(order), dirichlet=[], dgjumps=True)
    Vh0 = H1(mesh, order=int(order), dirichlet="bottom|right|top|left", dgjumps=False)

    hasneg = ci.GetElementsOfType(HASNEG)
    haspos = ci.GetElementsOfType(HASPOS)

    Vh_Gamma = Compress(Vh, GetDofsOfElements(Vh, hasneg)) * Compress(Vh, GetDofsOfElements(Vh, haspos)) * Vh0

    u1, u2, z0 = Vh_Gamma.TrialFunction()
    v1, v2, w0 = Vh_Gamma.TestFunction()

    u = [u1, u2]
    v = [v1, v2]
    z = [z0, z0]
    w = [w0, w0]

    gradu = [grad(u[i]) for i in range(2)]
    gradv = [grad(v[i]) for i in range(2)]
    gradz = [grad(z[i]) for i in range(2)]
    gradw = [grad(w[i]) for i in range(2)]

    # --- cut measures (with deformation) --------------------------------------
    deform = getattr(lsetadap, "deform", None) if bool(use_deformation) else None
    cut_kwargs = {}
    if deform is not None:
        cut_kwargs["deformation"] = deform
    if intorder is not None:
        cut_kwargs["order"] = int(intorder)

    dC = tuple([dCut(lsetp1, dt, definedonelements=ci.GetElementsOfType(HAS(dt)), **cut_kwargs) for dt in [NEG, POS]])
    dGamma = dCut(lsetp1, IF, **cut_kwargs)

    n = 1.0 / grad(lsetp1).Norm() * grad(lsetp1)
    kappa = (mu[1] / sum(mu), mu[0] / sum(mu))
    average_flux_z = sum([-kappa[i] * mu[i] * gradz[i] * n for i in [0, 1]])
    average_flux_w = sum([-kappa[i] * mu[i] * gradw[i] * n for i in [0, 1]])

    # --- main coupling (primal/dual) ------------------------------------------
    a = BilinearForm(Vh_Gamma, symmetric=True)
    a += sum(mu[i] * gradu[i] * gradw[i] * dC[i] for i in [0, 1])
    a += average_flux_w * (u[0] - u[1]) * dGamma

    a += sum(mu[i] * gradv[i] * gradz[i] * dC[i] for i in [0, 1])
    a += average_flux_z * (v[0] - v[1]) * dGamma

    f = LinearForm(Vh_Gamma)
    f += sum(coef_f[i] * w[i] * dC[i] for i in [0, 1])

    # --- data constraint on omega ---------------------------------------------
    gamma_data = float(gamma_data)
    for i, dt in enumerate([NEG, POS]):
        a += gamma_data * u[i] * v[i] * dCut(lsetp1, dt, definedonelements=omega_elems, **cut_kwargs)
        f += gamma_data * solution[i] * v[i] * dCut(lsetp1, dt, definedonelements=omega_elems, **cut_kwargs)

    # --- CIP (continuous interior penalty) ------------------------------------
    ba_facets = {
        NEG: GetFacetsWithNeighborTypes(mesh, a=hasneg, b=hasneg),
        POS: GetFacetsWithNeighborTypes(mesh, a=haspos, b=haspos),
    }
    dk = tuple(
        [
            dCut(
                lsetp1,
                dt,
                skeleton=True,
                definedonelements=ba_facets[dt],
                **cut_kwargs,
            )
            for dt in [NEG, POS]
        ]
    )
    nF = specialcf.normal(mesh.dim)
    h = specialcf.mesh_size
    gamma_CIP = float(gamma_cip)
    a += sum(
        [
            gamma_CIP
            * h
            * mu[i]
            * InnerProduct((gradu[i] - gradu[i].Other()) * nF, (gradv[i] - gradv[i].Other()) * nF)
            * dk[i]
            for i in [0, 1]
        ]
    )

    # --- GLS ------------------------------------------------------------------
    def calL(fun):
        hesse = [fun[i].Operator("hesse") for i in [0, 1]]
        return (
            -mu[0] * hesse[0][0, 0] - mu[0] * hesse[0][1, 1],
            -mu[1] * hesse[1][0, 0] - mu[1] * hesse[1][1, 1],
        )

    gamma_GLS = float(gamma_gls)
    a += sum([gamma_GLS * h**2 * calL(u)[i] * calL(v)[i] * dC[i] for i in [0, 1]])
    f += sum([gamma_GLS * h**2 * coef_f[i] * calL(v)[i] * dC[i] for i in [0, 1]])

    # --- interface stabilization ----------------------------------------------
    def P(fun):
        return fun - (fun * n) * n

    jump_flux_u = (mu[0] * gradu[0] - mu[1] * gradu[1]) * n
    jump_flux_v = (mu[0] * gradv[0] - mu[1] * gradv[1]) * n
    jump_tangential_u = P(gradu[0]) - P(gradu[1])
    jump_tangential_v = P(gradv[0]) - P(gradv[1])

    gamma_IF = float(gamma_if)
    mubar = 0.5 * (mu[0] + mu[1])
    a += gamma_IF * (mubar / h) * (u[0] - u[1]) * (v[0] - v[1]) * dGamma
    a += gamma_IF * h * jump_flux_u * jump_flux_v * dGamma
    a += gamma_IF * mubar * h * jump_tangential_u * jump_tangential_v * dGamma

    # --- (weak) Tikhonov ------------------------------------------------------
    alpha = [float(alpha0), float(alpha1)]
    dGeom = tuple([dx(definedonelements=ci.GetElementsOfType(dt)) for dt in [HASNEG, HASPOS]])
    a += sum(alpha[0] * h ** (2 * order) * u[i] * v[i] * dGeom[i] for i in [0, 1])
    a += sum(alpha[1] * h ** (2 * order) * grad(u[i]) * grad(v[i]) * dGeom[i] for i in [0, 1])

    # --- dual stabilization ----------------------------------------------------
    a += sum(-mu[i] * gradz[i] * gradw[i] * dC[i] for i in [0, 1])

    # --- solve ----------------------------------------------------------------
    a.Assemble()
    f.Assemble()
    gfu = GridFunction(Vh_Gamma)
    gfu.vec.data = a.mat.Inverse(Vh_Gamma.FreeDofs(), inverse="sparsecholesky") * f.vec

    # --- error on B -----------------------------------------------------------
    err_kwargs = {"order": 2 * int(order)}
    if deform is not None:
        err_kwargs["deformation"] = deform
    err_sqr = sum(
        [
            (gfu.components[i] - solution[i]) ** 2
            * dCut(
                lsetp1,
                dt,
                definedonelements=B_elems,
                **err_kwargs,
            )
            for i, dt in zip([0, 1], [NEG, POS])
        ]
    )
    l2_err = float(ngsqrt(Integrate(err_sqr, mesh)))
    return l2_err


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--order", type=int, default=2)
    p.add_argument("--geom-order", type=int, default=None, help="Geometry (mapping) order. Defaults to --order.")
    p.add_argument(
        "--intorder",
        type=int,
        default=None,
        help="Integration order override for dCut(...) measures (volume/interface/skeleton).",
    )
    p.add_argument("--no-deformation", action="store_true", help="Disable mesh deformation in dCut(...).")
    p.add_argument("--gamma-data", type=float, default=1.0e5)
    p.add_argument("--gamma-cip", type=float, default=5.0e-2)
    p.add_argument("--gamma-gls", type=float, default=5.0e-2)
    p.add_argument("--gamma-if", type=float, default=1.0e-3)
    p.add_argument("--alpha0", type=float, default=1.0e-5)
    p.add_argument("--alpha1", type=float, default=1.0e-2)
    args = p.parse_args()
    geom_order = int(args.order) if args.geom_order is None else int(args.geom_order)
    l2 = solve_uc(
        order=int(args.order),
        geom_order=geom_order,
        intorder=args.intorder,
        use_deformation=not bool(args.no_deformation),
        gamma_data=float(args.gamma_data),
        gamma_cip=float(args.gamma_cip),
        gamma_gls=float(args.gamma_gls),
        gamma_if=float(args.gamma_if),
        alpha0=float(args.alpha0),
        alpha1=float(args.alpha1),
    )
    print(f"NGSolve / NGSXFEM UC L2 error (geom_order={geom_order}): {l2:.16e}")


if __name__ == "__main__":
    main()
