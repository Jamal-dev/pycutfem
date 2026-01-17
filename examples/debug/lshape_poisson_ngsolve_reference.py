#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from typing import Any


def _solve_lshape(
    *,
    order: int,
    maxh: float,
    levels: int,
    refine_r0: float,
    refine_shrink: float,
    int_order: int,
) -> list[dict[str, Any]]:
    import ngsolve as ngs
    from netgen.geom2d import SplineGeometry

    geo = SplineGeometry()
    p1 = geo.AppendPoint(0, 0)
    p2 = geo.AppendPoint(1, 0)
    p3 = geo.AppendPoint(1, 1)
    p4 = geo.AppendPoint(-1, 1)
    p5 = geo.AppendPoint(-1, -1)
    p6 = geo.AppendPoint(0, -1)
    geo.Append(["line", p1, p2], bc="outer")
    geo.Append(["line", p2, p3], bc="outer")
    geo.Append(["line", p3, p4], bc="outer")
    geo.Append(["line", p4, p5], bc="outer")
    geo.Append(["line", p5, p6], bc="outer")
    geo.Append(["line", p6, p1], bc="outer")

    mesh = ngs.Mesh(geo.GenerateMesh(maxh=float(maxh)))

    # Exact: u = r^(2/3) * sin(2/3 theta) on the 270° wedge, theta in [0,2π).
    func_angle = ngs.atan2(ngs.y, ngs.x)
    # NGSolve's IfPos(a, b, c) takes the b-branch only for strictly-positive 'a'.
    # Keep θ=0 on the +x axis (Dirichlet cut boundary) by biasing with +eps.
    func_angle = ngs.IfPos(func_angle + 1e-14, func_angle, func_angle + 2 * math.pi)
    exact = (ngs.x * ngs.x + ngs.y * ngs.y) ** (1.0 / 3.0) * ngs.sin((2.0 / 3.0) * func_angle)

    fes = ngs.H1(mesh, order=int(order), dirichlet="outer")
    u = fes.TrialFunction()
    v = fes.TestFunction()

    a = ngs.BilinearForm(fes, symmetric=True)
    a += ngs.grad(u) * ngs.grad(v) * ngs.dx
    f = ngs.LinearForm(fes)  # RHS is 0

    gfu = ngs.GridFunction(fes)
    out: list[dict[str, Any]] = []

    for level in range(int(levels)):
        fes.Update()
        gfu.Update()

        a.Assemble()
        f.Assemble()

        gfu.Set(exact, ngs.BND)
        inv = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky")
        gfu.vec.data += inv * (f.vec - a.mat * gfu.vec)

        err = gfu - exact
        l2_err = float(ngs.sqrt(ngs.Integrate(err * err, mesh, order=int(int_order))))
        out.append({"level": level, "ndof": int(fes.ndof), "l2_error": l2_err})

        if level == int(levels) - 1:
            break

        # Simple geometric marking to concentrate refinement near the singularity.
        r_mark = float(refine_r0) * (float(refine_shrink) ** level)
        r2_mark = r_mark * r_mark

        for el in mesh.Elements():
            pts = [mesh.vertices[v.nr].point for v in el.vertices]
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            if (cx * cx + cy * cy) <= r2_mark:
                mesh.SetRefinementFlag(el, True)

        mesh.Refine()

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="NGSolve reference for L-shape Poisson singular benchmark.")
    ap.add_argument("--order", type=int, default=2, help="H1 polynomial order.")
    ap.add_argument("--maxh", type=float, default=0.5, help="Initial mesh size.")
    ap.add_argument("--levels", type=int, default=6, help="Number of refine+solve cycles.")
    ap.add_argument("--refine-r0", type=float, default=0.8, help="Marking radius at level 0.")
    ap.add_argument("--refine-shrink", type=float, default=0.5, help="Radius shrink factor per level.")
    ap.add_argument("--int-order", type=int, default=12, help="Quadrature order for L2 error integration.")
    ap.add_argument("--format", choices=("table", "json"), default="table")
    args = ap.parse_args()

    results = _solve_lshape(
        order=args.order,
        maxh=args.maxh,
        levels=args.levels,
        refine_r0=args.refine_r0,
        refine_shrink=args.refine_shrink,
        int_order=args.int_order,
    )

    if args.format == "json":
        print(json.dumps(results))
        return

    print(f"{'level':>5}  {'ndof':>10}  {'L2 error':>12}")
    for row in results:
        print(f"{row['level']:5d}  {row['ndof']:10d}  {row['l2_error']:12.5e}")


if __name__ == "__main__":
    main()
