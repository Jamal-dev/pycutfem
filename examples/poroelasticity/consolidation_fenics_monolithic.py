#!/usr/bin/env python3
"""
FEniCS (dolfin) monolithic reference for 2D Biot poroelastic consolidation.

This solves the same model/mesh/time-stepping as
`examples/poroelasticity/consolidation_fenics_reference.py` (multiphenics block),
but uses a standard mixed FunctionSpace and `assemble_system`.

Run (in the dolfin env):
  conda run --no-capture-output -n fenics \\
    python examples/poroelasticity/consolidation_fenics_monolithic.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from consolidation_fenics_reference import (
    ProblemParameters,
    IsotropicMaterialParameters,
    _delaunay_cells,
    _generate_points,
    _mesh_from_points_cells,
    _structured_cells,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="examples/poroelasticity/fenics_monolithic_out")
    ap.add_argument(
        "--triangulation",
        choices=("delaunay", "structured"),
        default="delaunay",
        help="How to triangulate the (nx,ny) point set when --mesh-file is not provided.",
    )
    ap.add_argument(
        "--mesh-file",
        default=None,
        help="Optional .npz with arrays {points,cells}. If it doesn't exist, it is created.",
    )
    ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--H", type=float, default=10.0)
    ap.add_argument("--nx", type=int, default=31)
    ap.add_argument("--ny", type=int, default=14)
    ap.add_argument("--num-time-steps", type=int, default=120)
    ap.add_argument("--final-time", type=float, default=36.0)
    ap.add_argument("--pressure-region", type=float, default=5.0)
    ap.add_argument("--t1", type=float, default=None, help="Optional override for the drainage time t_1.")
    args = ap.parse_args()

    pp = ProblemParameters(
        final_time=float(args.final_time),
        t_1=float(args.t1) if args.t1 is not None else float(args.final_time) / 10.0,
        p_1=1.54e9,
        p_d=380e6,
        p_0=380e6,
        pressure_region=float(args.pressure_region),
        L=float(args.L),
        H=float(args.H),
        num_time_steps=int(args.num_time_steps),
    )
    mp = IsotropicMaterialParameters(
        porosity=0.19,
        poisson_ratio=0.2,
        biot_coef=0.78,
        E=14.4e9,
        biot_modulus=13.5e9,
        permeability=2e-10,
    )

    mesh_file = Path(args.mesh_file).resolve() if args.mesh_file else None
    if mesh_file is not None and mesh_file.exists():
        data = np.load(mesh_file)
        points = np.asarray(data["points"], dtype=float)
        cells = np.asarray(data["cells"], dtype=int)
    else:
        points = _generate_points(L=pp.L, H=pp.H, nx=args.nx, ny=args.ny)
        if args.triangulation == "structured":
            cells = _structured_cells(nx=args.nx, ny=args.ny)
        else:
            cells = _delaunay_cells(points)
        if mesh_file is not None:
            mesh_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(mesh_file, points=points, cells=cells)
            print(f"[fenics-monolithic] wrote mesh file {mesh_file}")

    mesh = _mesh_from_points_cells(points, cells)
    mp.update_parameters_by_E(dim=mesh.geometry().dim())
    mp.convert_to_fenics()

    from dolfin import (
        Constant,
        DirichletBC,
        FiniteElement,
        Function,
        FunctionSpace,
        Identity,
        Measure,
        MeshFunction,
        MixedElement,
        SubDomain,
        TestFunctions,
        TrialFunctions,
        VectorElement,
        XDMFFile,
        assemble_system,
        assign,
        div,
        dot,
        grad,
        inner,
        interpolate,
        near,
        solve,
        split,
        sym,
        tr,
    )

    # ------------------------------------------------------------------
    # Boundary markers (same as the multiphenics script)
    # ------------------------------------------------------------------
    split_tol = 1e-12

    class TopBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], pp.H, 0.5) and (x[0] > pp.pressure_region + split_tol) and on_boundary

    class BottomBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0.0) and on_boundary

    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0) and on_boundary

    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], pp.L) and on_boundary

    class PressureLoadBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return (x[0] <= pp.pressure_region + split_tol) and near(x[1], pp.H, 0.5) and on_boundary

    class BM:
        top = 1
        bottom = 2
        left = 3
        right = 4
        pressure_load = 5

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    LeftBoundary().mark(boundaries, BM.left)
    RightBoundary().mark(boundaries, BM.right)
    TopBoundary().mark(boundaries, BM.top)
    BottomBoundary().mark(boundaries, BM.bottom)
    PressureLoadBoundary().mark(boundaries, BM.pressure_load)

    out_dir = Path(args.output_dir).resolve()
    (out_dir / "mesh_files").mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    with XDMFFile(str(out_dir / "mesh_files" / "consolidation_test_boundaries.xdmf")) as f:
        f.write(boundaries)

    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

    dt = float(pp.final_time) / float(pp.num_time_steps)
    dt_c = Constant(dt)
    theta = Constant(0.5)

    def eps(uu):
        return sym(grad(uu))

    def stress_s(uu):
        return 2.0 * mp.mu * eps(uu) + mp.lambda_ * tr(eps(uu)) * Identity(2)

    element_u = VectorElement("CG", mesh.ufl_cell(), 2)
    element_p = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, MixedElement([element_u, element_p]))

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    w = Function(W)
    w0 = Function(W)
    (u0, p0) = split(w0)

    traction_rate = Constant((0.0, 0.0))

    a = (
        inner(stress_s(u), eps(v)) * dx
        - mp.biot_coef * p * div(v) * dx
        + mp.biot_coef * div(u) * q * dx
        + (1.0 / mp.biot_modulus) * p * q * dx
        + theta * dt_c * dot(mp.permeability * grad(p), grad(q)) * dx
    )
    L = (
        inner(stress_s(u0), eps(v)) * dx
        - mp.biot_coef * p0 * div(v) * dx
        + dt_c * dot(traction_rate, v) * ds(BM.pressure_load)
        + mp.biot_coef * div(u0) * q * dx
        + (1.0 / mp.biot_modulus) * p0 * q * dx
        - (1.0 - theta) * dt_c * dot(mp.permeability * grad(p0), grad(q)) * dx
    )

    # Dirichlet BCs
    bcs = [
        DirichletBC(W.sub(0).sub(0), Constant(0.0), boundaries, BM.left),
        DirichletBC(W.sub(0).sub(0), Constant(0.0), boundaries, BM.right),
        DirichletBC(W.sub(0).sub(1), Constant(0.0), boundaries, BM.bottom),
        DirichletBC(W.sub(1), Constant(pp.p_d), boundaries, BM.top),
    ]

    # Initial conditions (u=0, p=p_d)
    u_init = interpolate(Constant((0.0, 0.0)), W.sub(0).collapse())
    p_init = interpolate(Constant(pp.p_d), W.sub(1).collapse())
    assign(w0.sub(0), u_init)
    assign(w0.sub(1), p_init)

    times: list[float] = []
    p_w_max: list[float] = []
    delta_u_max: list[float] = []
    delta_p_max: list[float] = []
    traction_rate_y: list[float] = []

    t = 0.0
    step = 0
    while t < pp.final_time - 1e-14:
        t += dt
        step += 1

        tr_y = (-pp.p_1 / pp.t_1) if t < pp.t_1 else 0.0
        traction_rate.assign(Constant((0.0, float(tr_y))))

        A, b = None, None
        A, b = assemble_system(a, L, bcs)
        solve(A, w.vector(), b, "mumps")

        (u_sol, p_sol) = w.split()
        (u_prev, p_prev) = w0.split()
        delta_u_max.append((u_sol.vector() - u_prev.vector()).max())
        delta_p_max.append((p_sol.vector() - p_prev.vector()).max())
        traction_rate_y.append(float(tr_y))
        p_w_max.append(p_sol.vector().max())
        times.append(float(t))

        w0.assign(w)

        if step % max(1, pp.num_time_steps // 10) == 0:
            print(f"[fenics-monolithic] step={step:4d}/{pp.num_time_steps}  t={t:.3e}  p_max={p_w_max[-1]:.3e}")

    import pandas as pd

    df = pd.DataFrame(
        {
            "time": times,
            "p_w_max": p_w_max,
            "delta_u_max": delta_u_max,
            "delta_p_max": delta_p_max,
            "traction_rate_y": traction_rate_y,
            "theta_val": float(theta),
        }
    )
    out_csv = out_dir / "results" / "nonIncremental_test_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"[fenics-monolithic] wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
