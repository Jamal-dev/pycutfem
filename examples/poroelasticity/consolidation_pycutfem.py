#!/usr/bin/env python3
"""
pycutfem implementation of 2D Biot poroelastic consolidation.

This matches the *monolithic* (non-incremental unknowns, incremental loading)
scheme used in `examples/poroelasticity/consolidation_fenics_reference.py`.

Run (in fenicsx or any env with numpy/scipy):
  conda run --no-capture-output -n fenicsx \
    python examples/poroelasticity/consolidation_pycutfem.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la
from scipy.spatial import Delaunay

# Allow running this example without installing pycutfem into the active env.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pycutfem.core.mesh import Mesh
from pycutfem.core.topology import Node
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
)
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.measures import dS, dx
from examples.utils.poromechanics import (
    UPlMaterial2D,
    build_upl_theta_system_2d,
    displacement_neumann_rhs,
)


def _generate_points(*, L: float, H: float, nx: int, ny: int) -> np.ndarray:
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


def _ccw_cells(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    cells = np.asarray(simplices, dtype=int).copy()
    a = points[cells[:, 0]]
    b = points[cells[:, 1]]
    c = points[cells[:, 2]]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    flip = area2 < 0.0
    cells[flip, 1], cells[flip, 2] = cells[flip, 2], cells[flip, 1]
    return cells


def _build_p2_tri_mesh(points: np.ndarray, cells_p1: np.ndarray) -> Mesh:
    """
    Build a P2 triangular mesh for pycutfem from a P1 Delaunay triangulation.

    Local node ordering for each triangle (k=2) matches pycutfem's P2 convention:
      [V0, mid(V0,V1), V1, mid(V0,V2), mid(V1,V2), V2]
    """
    points = np.asarray(points, dtype=float)
    cells = _ccw_cells(points, cells_p1)

    nodes: list[Node] = [Node(id=i, x=float(x), y=float(y)) for i, (x, y) in enumerate(points)]

    edge_to_mid: dict[tuple[int, int], int] = {}

    def _mid_node(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        mid = edge_to_mid.get(key)
        if mid is not None:
            return int(mid)
        xa, ya = points[key[0]]
        xb, yb = points[key[1]]
        nid = len(nodes)
        nodes.append(Node(id=nid, x=float(0.5 * (xa + xb)), y=float(0.5 * (ya + yb))))
        edge_to_mid[key] = nid
        return nid

    elements = np.empty((cells.shape[0], 6), dtype=int)
    corners = np.empty((cells.shape[0], 3), dtype=int)
    for eid, (v0, v1, v2) in enumerate(cells):
        m01 = _mid_node(int(v0), int(v1))
        m02 = _mid_node(int(v0), int(v2))
        m12 = _mid_node(int(v1), int(v2))
        elements[eid, :] = [int(v0), int(m01), int(v1), int(m02), int(m12), int(v2)]
        corners[eid, :] = [int(v0), int(v1), int(v2)]

    return Mesh(
        nodes=nodes,
        element_connectivity=elements,
        edges_connectivity=None,  # infer edges
        elements_corner_nodes=corners,
        element_type="tri",
        poly_order=2,
    )


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class MaterialParameters:
    porosity: float
    poisson_ratio: float
    biot_coef: float
    E: float
    biot_modulus: float
    permeability: float

    @property
    def lambda_(self) -> float:
        nu = float(self.poisson_ratio)
        E = float(self.E)
        return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    @property
    def mu(self) -> float:
        nu = float(self.poisson_ratio)
        E = float(self.E)
        return E / (2.0 * (1.0 + nu))


def solve_consolidation_pycutfem(
    *,
    output_dir: str | Path | None = "examples/poroelasticity/pycutfem_out",
    L: float = 20.0,
    H: float = 10.0,
    nx: int = 31,
    ny: int = 14,
    num_time_steps: int = 120,
    final_time: float = 36.0,
    pressure_region: float = 5.0,
    t_1: float | None = None,
    triangulation: str = "delaunay",
    mesh_file: str | Path | None = None,
    backend: str = "jit",
    compare_with: str | Path | None = None,
    write_csv: bool = True,
    print_progress: bool = True,
) -> dict[str, list[float] | float]:
    """
    Run the 2D Biot consolidation benchmark with pycutfem.

    Returns a dict with time-history arrays (time, p_w_max, ...) plus theta_val.
    """
    if t_1 is None:
        t_1 = float(final_time) / 10.0

    results_dir: Path | None = None
    if write_csv:
        if output_dir is None:
            raise ValueError("output_dir must be provided when write_csv=True")
        out_dir = Path(output_dir).resolve()
        results_dir = out_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

    pp = ProblemParameters(
        final_time=float(final_time),
        t_1=float(t_1),
        p_1=1.54e9,
        p_d=380e6,
        p_0=380e6,
        pressure_region=float(pressure_region),
        L=float(L),
        H=float(H),
        num_time_steps=int(num_time_steps),
    )
    mp = MaterialParameters(
        porosity=0.19,
        poisson_ratio=0.2,
        biot_coef=0.78,
        E=14.4e9,
        biot_modulus=13.5e9,
        permeability=2e-10,
    )

    mesh_path = Path(mesh_file).resolve() if mesh_file is not None else None
    if mesh_path is not None and mesh_path.exists():
        data = np.load(mesh_path)
        points = np.asarray(data["points"], dtype=float)
        cells_p1 = np.asarray(data["cells"], dtype=int)
    else:
        points = _generate_points(L=pp.L, H=pp.H, nx=nx, ny=ny)
        if triangulation == "structured":
            cells_p1 = _structured_cells(nx=nx, ny=ny)
        elif triangulation == "delaunay":
            tri = Delaunay(points)
            cells_p1 = np.asarray(tri.simplices, dtype=int)
        else:
            raise ValueError(f"Unknown triangulation '{triangulation}' (expected 'delaunay' or 'structured').")

        if mesh_path is not None:
            mesh_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(mesh_path, points=points, cells=cells_p1)
            if print_progress:
                print(f"[pycutfem] wrote mesh file {mesh_path}")

    mesh = _build_p2_tri_mesh(points, cells_p1)

    # Boundary tags (match FEniCS script logic; note the large top tolerance).
    top_tol = 0.5
    mesh.tag_boundary_edges(
        {
            "bottom": lambda x, y: np.isclose(y, 0.0),
            "left": lambda x, y: np.isclose(x, 0.0),
            "right": lambda x, y: np.isclose(x, pp.L),
            "top_drained": lambda x, y: (abs(y - pp.H) <= top_tol) and (x > pp.pressure_region),
            # Use a strict split to match how dolfin facet marking behaves at x=pressure_region.
            "pressure_load": lambda x, y: (abs(y - pp.H) <= top_tol) and (x < pp.pressure_region),
        }
    )

    mixed_element = MixedElement(mesh, field_specs={"ux": 2, "uy": 2, "p": 1})
    dh = DofHandler(mixed_element, method="cg")

    disp_space = FunctionSpace("displacement", ["ux", "uy"], dim=1)
    pres_space = FunctionSpace("pressure", ["p"], dim=0)

    u = VectorTrialFunction(disp_space, dof_handler=dh)
    v = VectorTestFunction(disp_space, dof_handler=dh)
    p = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)

    u0 = VectorFunction(name="u0", field_names=["ux", "uy"], dof_handler=dh)
    p0 = Function(name="p0", field_name="p", dof_handler=dh)
    u0.nodal_values.fill(0.0)
    p0.nodal_values.fill(pp.p_d)

    # Time-dependent traction rate (vector) used on the pressure_load boundary.
    traction_rate = Constant(np.asarray([0.0, 0.0], dtype=float))
    traction_rate._jit_name = "traction_rate"

    theta = Constant(0.5)
    theta._jit_name = "theta"
    dt = float(pp.final_time) / float(pp.num_time_steps)
    dt_c = Constant(dt)
    dt_c._jit_name = "dt"

    upl_material = UPlMaterial2D(
        young_modulus=mp.E,
        poisson_ratio=mp.poisson_ratio,
        porosity=mp.porosity,
        biot_coefficient=mp.biot_coef,
        permeability_xx=mp.permeability,
        storage_inverse=1.0 / mp.biot_modulus,
    )
    volume_measure = dx(metadata={"q": 5})
    upl_system = build_upl_theta_system_2d(
        u_trial=u,
        p_trial=p,
        u_test=v,
        p_test=q,
        u_prev=u0,
        p_prev=p0,
        material=upl_material,
        dt=dt_c,
        theta=theta,
        dx_measure=volume_measure,
    )

    a = upl_system.lhs_form
    L = upl_system.rhs_form + displacement_neumann_rhs(
        v,
        traction_rate,
        dS(mesh.edge_bitset("pressure_load"), metadata={"q": 5}),
        scale=dt_c,
    )

    bcs = [
        BoundaryCondition("ux", "dirichlet", "left", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "right", lambda x, y: 0.0),
        BoundaryCondition("uy", "dirichlet", "bottom", lambda x, y: 0.0),
        BoundaryCondition("p", "dirichlet", "top_drained", lambda x, y: float(pp.p_d)),
    ]

    times: list[float] = []
    p_w_max: list[float] = []
    delta_u_max: list[float] = []
    delta_p_max: list[float] = []
    traction_rate_y: list[float] = []

    # ------------------------------------------------------------------
    # Assemble and factorize the (time-invariant) matrix once.
    # Only the RHS depends on u0/p0 and the traction rate.
    # ------------------------------------------------------------------
    compiler = FormCompiler(dh, quadrature_order=None, backend=backend)
    ndofs = dh.total_dofs

    K_lil = sp.lil_matrix((ndofs, ndofs))
    compiler._basis_cache.clear()
    compiler._coeff_cache.clear()
    compiler._collapsed_cache.clear()
    compiler.ctx["rhs"] = False
    compiler._assemble_form(a, K_lil)
    K_raw = K_lil.tocsr()

    dirichlet = dh.get_dirichlet_data(bcs)
    if dirichlet:
        bc_rows = np.fromiter(dirichlet.keys(), dtype=int)
        bc_vals = np.fromiter(dirichlet.values(), dtype=float)
    else:
        bc_rows = np.zeros(0, dtype=int)
        bc_vals = np.zeros(0, dtype=float)

    u_bc = np.zeros(ndofs, dtype=float)
    if bc_rows.size:
        u_bc[bc_rows] = bc_vals
    bc_shift = K_raw @ u_bc

    K_bc = compiler._apply_bcs(K_lil.copy(), np.zeros(ndofs, dtype=float), bcs)
    lu = sp_la.splu(K_bc.tocsc())

    t = 0.0
    w_prev = np.zeros(ndofs, dtype=float)
    # Start consistent with u0/p0.
    w_prev[u0._g_dofs] = u0.nodal_values
    w_prev[p0._g_dofs] = p0.nodal_values

    for step in range(1, pp.num_time_steps + 1):
        t = step * dt

        tr_y = (-pp.p_1 / pp.t_1) if t < pp.t_1 else 0.0
        traction_rate.value[0] = 0.0
        traction_rate.value[1] = float(tr_y)

        compiler._basis_cache.clear()
        compiler._coeff_cache.clear()
        compiler._collapsed_cache.clear()
        compiler.ctx["rhs"] = True
        F_raw = np.zeros(ndofs, dtype=float)
        compiler._assemble_form(L, F_raw)

        F = F_raw - bc_shift
        if bc_rows.size:
            F[bc_rows] = bc_vals

        w = lu.solve(F)

        # Extract and record diagnostics on the full global vector.
        p_slice = np.asarray(dh.get_field_slice("p"), dtype=int)
        u_slice = np.concatenate([dh.get_field_slice("ux"), dh.get_field_slice("uy")]).astype(int)
        p_w_max.append(float(np.max(w[p_slice])))
        delta_u_max.append(float(np.max(w[u_slice] - w_prev[u_slice])))
        delta_p_max.append(float(np.max(w[p_slice] - w_prev[p_slice])))
        traction_rate_y.append(float(tr_y))
        times.append(float(t))

        # Update previous state functions.
        u0.nodal_values = w[u0._g_dofs]
        p0.nodal_values = w[p0._g_dofs]
        w_prev[:] = w

        if print_progress and step % max(1, pp.num_time_steps // 10) == 0:
            print(f"[pycutfem] step={step:4d}/{pp.num_time_steps}  t={t:.3e}  p_max={p_w_max[-1]:.3e}")

    results: dict[str, list[float] | float] = {
        "time": times,
        "p_w_max": p_w_max,
        "delta_u_max": delta_u_max,
        "delta_p_max": delta_p_max,
        "traction_rate_y": traction_rate_y,
        "theta_val": float(theta),
    }

    if write_csv and results_dir is not None:
        import pandas as pd

        df = pd.DataFrame(results)
        out_csv = results_dir / "nonIncremental_test_results.csv"
        df.to_csv(out_csv, index=False)
        if print_progress:
            print(f"[pycutfem] wrote {out_csv}")

        if compare_with:
            ref = pd.read_csv(Path(compare_with))
            df_cmp = df.copy()
            ref_cmp = ref.copy()
            df_cmp["time_r"] = df_cmp["time"].round(10)
            ref_cmp["time_r"] = ref_cmp["time"].round(10)
            merged = df_cmp.merge(ref_cmp, on="time_r", suffixes=("_pycutfem", "_fenics"), how="inner")
            if not merged.empty:
                diff = merged["p_w_max_pycutfem"].to_numpy() - merged["p_w_max_fenics"].to_numpy()
                print(
                    "[compare] p_w_max: "
                    f"n={diff.size}  max|Δ|={np.max(np.abs(diff)):.3e}  "
                    f"rms={np.sqrt(np.mean(diff**2)):.3e}"
                )
            else:
                print("[compare] no matching time points found between CSVs (after rounding).")

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="examples/poroelasticity/pycutfem_out")
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
    ap.add_argument("--backend", choices=("jit", "python"), default="jit")
    ap.add_argument("--compare-with", default=None, help="Optional path to reference CSV for p_w_max comparison.")
    args = ap.parse_args()

    solve_consolidation_pycutfem(
        output_dir=args.output_dir,
        L=args.L,
        H=args.H,
        nx=args.nx,
        ny=args.ny,
        num_time_steps=args.num_time_steps,
        final_time=args.final_time,
        pressure_region=args.pressure_region,
        t_1=args.t1,
        triangulation=args.triangulation,
        mesh_file=args.mesh_file,
        backend=args.backend,
        compare_with=args.compare_with,
        write_csv=True,
        print_progress=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
