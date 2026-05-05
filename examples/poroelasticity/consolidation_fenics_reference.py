#!/usr/bin/env python3
"""
FEniCS (dolfin) reference implementation for 2D Biot poroelastic consolidation.

This is a lightly-scripted version of the notebook code provided in the prompt:
- unstructured triangular mesh from Delaunay triangulation of a stretched grid
- quasi-static linear elasticity coupled to Darcy flow (Biot)
- monolithic mixed solve in time with a one-step theta scheme

Run (in the dolfin env):
  conda run --no-capture-output -n fenics \
    python examples/poroelasticity/consolidation_fenics_reference.py
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _generate_points(*, L: float, H: float, nx: int, ny: int) -> np.ndarray:
    # Equally spaced x; exponentially stretched y (matches the provided code).
    x_unique = np.linspace(0.0, float(L), int(nx))
    y_unique = float(H) * (1.0 - np.exp(-np.linspace(0.0, float(H) / 3.3, int(ny))))
    x_coords = np.repeat(x_unique, int(ny))
    y_coords = np.tile(y_unique, int(nx))
    return np.column_stack([x_coords, y_coords])


def _structured_cells(*, nx: int, ny: int) -> np.ndarray:
    """
    Deterministic triangulation of the tensor-product point set produced by
    `_generate_points`: each logical quad is split into two triangles.
    """
    nx = int(nx)
    ny = int(ny)
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be >= 2 for triangulation")

    cells: list[list[int]] = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            v00 = i * ny + j
            v10 = (i + 1) * ny + j
            v01 = i * ny + (j + 1)
            v11 = (i + 1) * ny + (j + 1)
            # CCW orientation
            cells.append([v00, v10, v11])
            cells.append([v00, v11, v01])
    return np.asarray(cells, dtype=int)


def _ccw_cells(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    tri = np.asarray(cells, dtype=int).copy()
    a = pts[tri[:, 0]]
    b = pts[tri[:, 1]]
    c = pts[tri[:, 2]]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    flip = area2 < 0.0
    tri[flip, 1], tri[flip, 2] = tri[flip, 2], tri[flip, 1]
    return tri


def _delaunay_cells(points: np.ndarray) -> np.ndarray:
    from scipy.spatial import Delaunay

    tri = Delaunay(points)
    return np.asarray(tri.simplices, dtype=int)


def _mesh_from_points_cells(points: np.ndarray, cells: np.ndarray):
    vertices = np.asarray(points, dtype=float)
    tri = _ccw_cells(vertices, cells)

    from dolfin import Mesh, MeshEditor

    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, "triangle", 2, 2)
    editor.init_vertices(vertices.shape[0])
    editor.init_cells(tri.shape[0])
    for i, xy in enumerate(vertices):
        editor.add_vertex(int(i), xy)
    for i, cell in enumerate(tri):
        editor.add_cell(int(i), cell)
    editor.close()
    return mesh


def _mesh_from_delaunay(points: np.ndarray):
    return _mesh_from_points_cells(points, _delaunay_cells(points))


@dataclass
class ProblemParameters:
    final_time: float
    t_1: float
    p_1: float
    p_d: float
    p_0: float
    pressure_region: float
    L: float
    H: float
    num_time_steps: int


@dataclass
class IsotropicMaterialParameters:
    porosity: float
    poisson_ratio: float
    biot_coef: float
    E: float
    biot_modulus: float
    permeability: float  # k/mu^w

    def update_parameters_by_E(self, *, dim: int, Ks: float | None = None) -> None:
        nu = float(self.poisson_ratio)
        E = float(self.E)

        # Lamé parameters (3D formula used in the provided notebook; keep it as-is).
        self.lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.mu = E / (2.0 * (1.0 + nu))
        self.G = self.mu

        if dim == 2:
            self.K = self.lambda_ + self.mu
        elif dim == 3:
            self.K = self.lambda_ + 2.0 * self.mu / 3.0
        else:
            raise ValueError(f"Invalid dimension {dim}")

        if Ks is not None:
            self.b = 1.0 - (self.K / float(Ks))
            self.biot_coef = self.b
            self.K_s = float(Ks)
        else:
            self.b = float(self.biot_coef)
            self.K_s = self.K / (1.0 - float(self.biot_coef))
            self.K_f = float(self.porosity) / (
                1.0 / float(self.biot_modulus) - (float(self.biot_coef) - float(self.porosity)) / self.K_s
            )

    def convert_to_fenics(self) -> None:
        from dolfin import Constant

        for field, value in list(self.__dict__.items()):
            setattr(self, field, Constant(value))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="examples/poroelasticity/fenics_out")
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

    out_dir = Path(args.output_dir).resolve()
    mesh_dir = out_dir / "mesh_files"
    results_dir = out_dir / "results"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    mesh_file = Path(args.mesh_file).resolve() if args.mesh_file else None
    if mesh_file is not None and mesh_file.exists():
        data = np.load(mesh_file)
        points = np.asarray(data["points"], dtype=float)
        cells = np.asarray(data["cells"], dtype=int)
    else:
        points = _generate_points(L=args.L, H=args.H, nx=args.nx, ny=args.ny)
        if args.triangulation == "structured":
            cells = _structured_cells(nx=args.nx, ny=args.ny)
        else:
            cells = _delaunay_cells(points)
        if mesh_file is not None:
            mesh_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(mesh_file, points=points, cells=cells)
            print(f"[fenics] wrote mesh file {mesh_file}")

    spatial_mesh = _mesh_from_points_cells(points, cells)

    # ------------------------------------------------------------------
    # Parameters (match the provided notebook defaults)
    # ------------------------------------------------------------------
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
    mp.update_parameters_by_E(dim=spatial_mesh.geometry().dim())
    mp.convert_to_fenics()

    # ------------------------------------------------------------------
    # Boundary marking (matches the notebook logic; note the large top tol)
    # ------------------------------------------------------------------
    from dolfin import MeshFunction, SubDomain, near, XDMFFile
    from mpi4py import MPI

    mesh_comm = MPI.COMM_WORLD
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

    @dataclass(frozen=True)
    class BoundaryMarkers:
        top: int = 1
        bottom: int = 2
        left: int = 3
        right: int = 4
        pressure_load: int = 5

    bm = BoundaryMarkers()

    boundaries = MeshFunction("size_t", spatial_mesh, spatial_mesh.topology().dim() - 1)
    boundaries.set_all(0)
    LeftBoundary().mark(boundaries, bm.left)
    RightBoundary().mark(boundaries, bm.right)
    TopBoundary().mark(boundaries, bm.top)
    BottomBoundary().mark(boundaries, bm.bottom)
    PressureLoadBoundary().mark(boundaries, bm.pressure_load)
    with XDMFFile(mesh_comm, str(mesh_dir / "consolidation_test_boundaries.xdmf")) as f:
        f.write(boundaries)

    # ------------------------------------------------------------------
    # Monolithic solve (multiphenics block formulation, as in the notebook)
    # ------------------------------------------------------------------
    from dolfin import (
        Constant,
        FiniteElement,
        VectorElement,
        Identity,
        Measure,
        grad,
        inner,
        sym,
        tr,
        dot,
        div,
        interpolate,
        assign,
    )
    from multiphenics import BlockElement, BlockFunctionSpace, BlockTestFunction, BlockTrialFunction, BlockFunction
    from multiphenics import DirichletBC
    from multiphenics import block_split, BlockDirichletBC, block_assemble, block_solve, block_assign

    theta_val = Constant(0.5)
    dt = float(pp.final_time) / float(pp.num_time_steps)
    dt_c = Constant(dt)

    dx = Measure("dx", domain=spatial_mesh)
    ds = Measure("ds", domain=spatial_mesh, subdomain_data=boundaries)

    def eps(uu):
        return sym(grad(uu))

    def stress_s(uu):
        return 2.0 * mp.mu * eps(uu) + mp.lambda_ * tr(eps(uu)) * Identity(2)

    element_v = VectorElement("CG", spatial_mesh.ufl_cell(), 2)
    element_q = FiniteElement("CG", spatial_mesh.ufl_cell(), 1)

    traction_rate = Constant((0.0, 0.0))  # overwritten each step

    W_element = BlockElement(element_v, element_q)
    W = BlockFunctionSpace(spatial_mesh, W_element)
    vq = BlockTestFunction(W)
    up = BlockTrialFunction(W)
    w = BlockFunction(W)
    w0 = BlockFunction(W)
    (v, q) = block_split(vq)
    (u, p) = block_split(up)
    (u0, p0) = block_split(w0)

    # Dirichlet BCs
    bc_p_top = DirichletBC(W.sub(1), Constant(pp.p_d), boundaries, bm.top)
    bc_u_bottom = DirichletBC(W.sub(0).sub(1), Constant(0.0), boundaries, bm.bottom)
    bc_u_left = DirichletBC(W.sub(0).sub(0), Constant(0.0), boundaries, bm.left)
    bc_u_right = DirichletBC(W.sub(0).sub(0), Constant(0.0), boundaries, bm.right)
    bcs = BlockDirichletBC([bc_u_right, bc_u_left, bc_u_bottom, bc_p_top])

    # Initial conditions
    u_s_init = interpolate(Constant((0.0, 0.0)), W.sub(0))
    p_w_init = interpolate(Constant(pp.p_d), W.sub(1))
    assign(w0.sub(0), u_s_init)
    assign(w0.sub(1), p_w_init)

    # Weak forms
    K = inner(stress_s(u), eps(v)) * dx
    Q = mp.biot_coef * p * div(v) * dx
    Q_trans = mp.biot_coef * div(u) * q * dx
    S = (1.0 / mp.biot_modulus) * p * q * dx
    H = dot(mp.permeability * grad(p), grad(q)) * dx

    aa = [[-K, Q], [Q_trans, theta_val * dt_c * H + S]]

    # Previous state contributions
    K0 = inner(stress_s(u0), eps(v)) * dx
    Q0 = mp.biot_coef * p0 * div(v) * dx
    Q_trans0 = mp.biot_coef * div(u0) * q * dx
    S0 = (1.0 / mp.biot_modulus) * p0 * q * dx
    H0 = dot(mp.permeability * grad(p0), grad(q)) * dx

    # Note: traction is applied as an *increment* dt * traction_rate, matching the notebook.
    dfu_dt = dot(traction_rate, v) * ds(bm.pressure_load)

    f_1 = (-K0 + Q0 - dt_c * dfu_dt)
    f_2 = (Q_trans0 + S0 - (1.0 - theta_val) * dt_c * H0)
    ff = [f_1, f_2]

    # Time loop
    times: list[float] = []
    p_w_max: list[float] = []
    delta_u_max: list[float] = []
    delta_p_max: list[float] = []
    traction_rate_y: list[float] = []

    t = 0.0
    step = 0
    while t < pp.final_time - 1.0e-14:
        t += dt
        step += 1

        # Update traction rate (piecewise linear ramp in time).
        tr_y = (-pp.p_1 / pp.t_1) if t < pp.t_1 else 0.0
        traction_rate.assign(Constant((0.0, float(tr_y))))

        FF = block_assemble(ff)
        AA = block_assemble(aa)
        bcs.apply(FF)
        bcs.apply(AA)
        block_solve(AA, w.block_vector(), FF, "mumps")

        (u_s_sol, p_w_sol) = w
        (u_s_prev, p_w_prev) = w0

        delta_u_max.append((u_s_sol.vector() - u_s_prev.vector()).max())
        delta_p_max.append((p_w_sol.vector() - p_w_prev.vector()).max())
        traction_rate_y.append(float(tr_y))
        p_w_max.append(p_w_sol.vector().max())
        times.append(t)

        block_assign(w0, w)

        if step % max(1, pp.num_time_steps // 10) == 0:
            print(f"[fenics] step={step:4d}/{pp.num_time_steps}  t={t:.3e}  p_max={p_w_max[-1]:.3e}")

    # Write CSV output compatible with the notebook filenames.
    import pandas as pd

    df = pd.DataFrame(
        {
            "time": times,
            "p_w_max": p_w_max,
            "delta_u_max": delta_u_max,
            "delta_p_max": delta_p_max,
            "traction_rate_y": traction_rate_y,
            "theta_val": float(theta_val),
        }
    )
    df.to_csv(results_dir / "nonIncremental_test_results.csv", index=False)
    print(f"[fenics] wrote {results_dir / 'nonIncremental_test_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
