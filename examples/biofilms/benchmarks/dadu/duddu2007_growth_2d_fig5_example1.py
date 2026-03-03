"""
Duddu et al. (2007) growth-only model (XFEM + level set): 2D Example 1 (Fig. 5).

Paper reference
---------------
R. Duddu, S. Bordas, D. L. Chopp, B. Moran (2007)
"A combined extended finite element and level set method for biofilm growth"
Int. J. Numer. Meth. Engng.

We reproduce Example 1 (Fig. 5):
  - one semi-circular colony attached to the bottom wall,
  - growth driven by quasi-steady substrate diffusion + velocity potential,
  - interface advanced by level set φ_t + F ||∇φ|| = 0 with adaptive dt.

Outputs (in --outdir)
---------------------
- fig5a_interface.png   : interface motion (colored by time)
- fig5b_S.png           : substrate concentration at final time (color)
- fig5c_Phi.png         : velocity potential at final time (color)
- y_top_timeseries.csv  : interface top height vs time
- summary.json
- snaps_phi.npz         : stored φ(x,y) snapshots on the FD grid at paper target times
- vtk/step=XXXX.vtu     : VTK snapshots (S, Phi, levelset phi, extended speed F_ext)
- vtk/series.pvd        : ParaView time-series collection for the VTK snapshots
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from pycutfem.core.levelset import AffineLevelSet, PiecewiseLinearLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    grad,
    inner,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dInterface, dx
from pycutfem.utils.meshgen import structured_triangles
from pycutfem.xfem import XFEMDofHandler

from examples.biofilms.benchmarks.dadu.duddu2007_levelset_fd import (
    extend_speed_nearest_interface,
    level_set_update,
    make_uniform_grid,
    phi_union_disks_on_wall,
    reinitialize_signed_distance,
)
from examples.biofilms.benchmarks.dadu.duddu2007_speed import (
    compute_interface_segment_speeds_duddu2007,
    sample_segment_speeds,
)


try:
    from petsc4py import PETSc  # type: ignore

    _HAS_PETSC = True
except Exception:
    PETSc = None
    _HAS_PETSC = False


@dataclass(frozen=True)
class Duddu2007Kinetics:
    f_active: float = 0.5
    rho_x: float = 1.0250  # mg VS / mm^3
    rho_w: float = 1.0125  # mg VS / mm^3
    Y_xO: float = 0.583  # mg VS / mg O2
    Y_wO: float = 0.215  # mg VS / mg O2 (Table I)
    qhat0: float = 8.0  # mg O2 / (mg VS day)
    K0: float = 5.0e-7  # mg O2 / mm^3
    b: float = 0.3  # 1/day
    f_D: float = 0.8  # -
    g: float = 1.42  # mg O2 / mg VS

    def monod(self, S):
        return S / (Constant(float(self.K0)) + S)

    def consumption(self, S):
        f = Constant(float(self.f_active))
        rho_x = Constant(float(self.rho_x))
        qhat0 = Constant(float(self.qhat0))
        g = Constant(float(self.g))
        f_D = Constant(float(self.f_D))
        b = Constant(float(self.b))
        return f * rho_x * (qhat0 + g * f_D * b) * self.monod(S)

    def d_consumption(self, S):
        f = Constant(float(self.f_active))
        rho_x = Constant(float(self.rho_x))
        qhat0 = Constant(float(self.qhat0))
        g = Constant(float(self.g))
        f_D = Constant(float(self.f_D))
        b = Constant(float(self.b))
        K0 = Constant(float(self.K0))
        return f * rho_x * (qhat0 + g * f_D * b) * (K0 / (K0 + S) / (K0 + S))

    def divU(self, S):
        f = Constant(float(self.f_active))
        Y_xO = Constant(float(self.Y_xO))
        Y_wO = Constant(float(self.Y_wO))
        qhat0 = Constant(float(self.qhat0))
        b = Constant(float(self.b))
        f_D = Constant(float(self.f_D))
        rho_x = Constant(float(self.rho_x))
        rho_w = Constant(float(self.rho_w))
        mon = self.monod(S)
        rho_x_rate = (Y_xO * qhat0 - b) * mon
        rho_w_rate = (rho_x / rho_w) * ((Constant(1.0) - f_D) * b + Y_wO * qhat0) * mon
        return f * (rho_x_rate + rho_w_rate)


def _solve_linear_system(A, rhs, *, linear_solver: str) -> np.ndarray:
    linear_solver = str(linear_solver).strip().lower()
    if linear_solver in {"scipy", "spsolve", "direct"}:
        return spla.spsolve(A.tocsc(), rhs)
    if linear_solver in {"petsc", "ksp"}:
        if not _HAS_PETSC:
            raise RuntimeError("petsc4py is not available but --linear-solver petsc was requested.")
        A_csr = A.tocsr()
        Ap = PETSc.Mat().createAIJ(size=A_csr.shape, csr=(A_csr.indptr, A_csr.indices, A_csr.data))
        bp = PETSc.Vec().createWithArray(np.asarray(rhs, dtype=float))
        xp = bp.duplicate()
        ksp = PETSc.KSP().create()
        ksp.setOperators(Ap)
        # Default to an iterative solve that scales to the paper mesh sizes.
        # Users can override via PETSc options, e.g.:
        #   -ksp_type preonly -pc_type lu
        ksp.setType("cg")
        pc = ksp.getPC()
        pc.setType("gamg")
        ksp.setTolerances(rtol=1.0e-12, atol=1.0e-50, max_it=2000)
        ksp.setFromOptions()
        ksp.solve(bp, xp)
        return np.asarray(xp.getArray(), dtype=float)
    raise ValueError(f"Unknown linear_solver={linear_solver!r}. Use 'scipy' or 'petsc'.")


def _tag_rectangle_boundaries(mesh: Mesh, *, L: float, H: float, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - float(L)) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - float(H)) <= tol,
        }
    )


def _write_pvd(*, out_pvd: Path, datasets: list[tuple[float, str]]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for t, rel in datasets:
        lines.append(f'    <DataSet timestep="{float(t):.12e}" group="" part="0" file="{rel}"/>')
    lines += ["  </Collection>", "</VTKFile>", ""]
    out_pvd.write_text("\n".join(lines), encoding="utf-8")


def _sample_grid_bilinear(*, xg: np.ndarray, yg: np.ndarray, Z: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Bilinear sample of Z(yg, xg) onto point arrays x,y (broadcastable).
    Assumes xg and yg are uniform 1D grids with constant spacing.
    """
    xg = np.asarray(xg, dtype=float).ravel()
    yg = np.asarray(yg, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float)
    xx, yy = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if xg.size < 2 or yg.size < 2:
        raise ValueError("Need at least 2 grid points in each direction.")
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    nx = int(xg.size)
    ny = int(yg.size)
    # clamp indices to valid interior for i+1/j+1
    ii = np.floor((xx - float(xg[0])) / dx).astype(int)
    jj = np.floor((yy - float(yg[0])) / dy).astype(int)
    ii = np.clip(ii, 0, nx - 2)
    jj = np.clip(jj, 0, ny - 2)
    x0 = xg[ii]
    y0 = yg[jj]
    tx = (xx - x0) / dx
    ty = (yy - y0) / dy
    p00 = Z[jj, ii]
    p10 = Z[jj, ii + 1]
    p01 = Z[jj + 1, ii]
    p11 = Z[jj + 1, ii + 1]
    return (1.0 - tx) * (1.0 - ty) * p00 + tx * (1.0 - ty) * p10 + (1.0 - tx) * ty * p01 + tx * ty * p11


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _biofilm_top_y(*, grid_y: np.ndarray, phi_grid: np.ndarray) -> float:
    """
    Approximate y_max on Γ_int from the FD level set grid using vertical
    sign-change detection and linear interpolation.
    """
    phi_grid = np.asarray(phi_grid, float)
    y = np.asarray(grid_y, float)
    top = 0.0
    for i in range(int(phi_grid.shape[1])):
        col = np.asarray(phi_grid[:, i], float).ravel()
        for j in range(int(col.size) - 1):
            a = float(col[j])
            b = float(col[j + 1])
            if (a <= 0.0) and (b > 0.0) and (b != a):
                t01 = a / (a - b)
                y_int = float(y[j] + t01 * (y[j + 1] - y[j]))
                top = max(top, y_int)
    return float(top)


def _plot_level_set_contours_colored(
    phi_snaps: list[tuple[float, np.ndarray]],
    *,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    outpng: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    X, Y = np.meshgrid(grid_x, grid_y, indexing="xy")
    times = np.asarray([t for t, _ in phi_snaps], dtype=float)
    tmin = float(np.min(times))
    tmax = float(np.max(times))
    norm = plt.Normalize(vmin=tmin, vmax=tmax)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    for t, phi in phi_snaps:
        c = cmap(norm(float(t)))
        ax.contour(X, Y, np.asarray(phi, float), levels=[0.0], colors=[c], linewidths=1.0, alpha=0.95)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.9)
    cbar.set_label("t (days)")

    ax.set_xlim(float(grid_x[0]), float(grid_x[-1]))
    ax.set_ylim(float(grid_y[0]), float(grid_y[-1]))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(str(title))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    fig.savefig(outpng, dpi=220)
    plt.close(fig)


def _plot_scalar_on_mesh(
    *,
    dh_base: DofHandler,
    values_base: np.ndarray,
    field: str,
    title: str,
    outpng: Path,
    cmap: str,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    phi_grid: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    import matplotlib.tri as mtri  # noqa: E402

    values_base = np.asarray(values_base, dtype=float).ravel()
    mesh = dh_base.mixed_element.mesh
    n_nodes = int(np.asarray(mesh.nodes_x_y_pos, dtype=float).shape[0])

    node_to_gdof = (getattr(dh_base, "dof_map", {}) or {}).get(field) or {}
    if len(node_to_gdof) != n_nodes:
        raise RuntimeError(f"Field {field!r}: expected dof_map on all mesh nodes, got {len(node_to_gdof)} entries.")

    # Build node-ordered values robustly.
    u_node = np.zeros(n_nodes, dtype=float)
    sl = np.asarray(dh_base.get_field_slice(field), dtype=int).ravel()
    if sl.size != values_base.size:
        raise ValueError(f"{field}: values size {values_base.size} != base dofs {sl.size}")
    gd_to_val = {int(gd): float(values_base[i]) for i, gd in enumerate(sl.tolist())}
    for nid in range(n_nodes):
        gd = int(node_to_gdof[int(nid)])
        u_node[int(nid)] = float(gd_to_val.get(gd, 0.0))

    tri_conn = np.asarray(mesh.corner_connectivity, dtype=int)
    tri = mtri.Triangulation(mesh.nodes_x_y_pos[:, 0], mesh.nodes_x_y_pos[:, 1], tri_conn)

    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    tcf = ax.tricontourf(tri, u_node, levels=24, cmap=str(cmap))
    fig.colorbar(tcf, ax=ax, shrink=0.9)

    # Overlay the sharp interface (φ=0) from the FD grid so the panel matches the interface used by the level set.
    X, Y = np.meshgrid(grid_x, grid_y, indexing="xy")
    ax.contour(X, Y, np.asarray(phi_grid, float), levels=[0.0], colors="k", linewidths=1.1, alpha=0.9)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(str(title))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    fig.savefig(outpng, dpi=220)
    plt.close(fig)


def main() -> None:
    # Keep compiler logs quiet by default (they can dominate runtime on long runs).
    logging.getLogger("pycutfem").setLevel(logging.WARNING)
    logging.getLogger("pycutfem.ufl.compilers").setLevel(logging.WARNING)

    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="examples/biofilms/benchmarks/dadu/results/duddu2007_fig5_example1_xfem")
    p.add_argument("--backend", choices=("cpp",), default="cpp")
    p.add_argument("--linear-solver", choices=("petsc", "scipy"), default="petsc")
    p.add_argument("--q", type=int, default=4)

    # Geometry
    p.add_argument("--L", type=float, default=0.5, help="Domain width (mm).")
    p.add_argument("--H", type=float, default=0.5, help="Domain height (mm).")
    p.add_argument("--mesh-nx", type=int, default=100, help="Background mesh divisions in x (quads split into triangles).")
    p.add_argument("--mesh-ny", type=int, default=100, help="Background mesh divisions in y (quads split into triangles).")
    p.add_argument("--grid-nx", type=int, default=200, help="FD level set grid cells in x.")
    p.add_argument("--grid-ny", type=int, default=200, help="FD level set grid cells in y.")

    # Example 1 initial colony
    p.add_argument("--center-x", type=float, default=0.25, help="Colony center x-position (mm).")
    p.add_argument("--radius", type=float, default=0.01, help="Colony radius (mm).")

    # Model params
    p.add_argument("--Sbar", type=float, default=8.3e-6)
    p.add_argument("--Db", type=float, default=146.88)
    p.add_argument("--Df", type=float, default=183.6)
    p.add_argument("--Y-xO", type=float, default=Duddu2007Kinetics.Y_xO)
    p.add_argument("--Y-wO", type=float, default=Duddu2007Kinetics.Y_wO)

    # Time stepping (paper: adaptive dt=0.8*min(dx,dy)/max(F))
    p.add_argument("--t-final", type=float, default=44.5, help="Final time (days).")
    p.add_argument("--dt-min", type=float, default=1.0e-4, help="Lower bound on adaptive dt (days).")
    p.add_argument("--max-steps", type=int, default=200000)
    p.add_argument("--reinit-every", type=int, default=1, help="Reinitialize signed distance every N steps (0=off).")

    # Substrate source boundary (paper: 0.1mm above top-most biofilm point)
    p.add_argument("--Ls", type=float, default=0.1, help="Distance from top-most biofilm point to Γ_S^d (mm).")
    p.add_argument("--S-penalty", type=float, default=1.0e6, help="Penalty parameter for S Dirichlet on Γ_S^d.")

    # Newton / penalty
    p.add_argument("--newton-tol", type=float, default=1.0e-10)
    p.add_argument("--max-it", type=int, default=50)
    p.add_argument("--newton-verbose", action="store_true")
    p.add_argument("--newton-damping", type=float, default=1.0, help="Constant damping factor for Newton updates (0<d≤1).")
    p.add_argument("--S-min", type=float, default=0.0)
    p.add_argument("--phi-penalty", type=float, default=1.0e6)

    # Speed evaluation
    p.add_argument(
        "--speed-mode",
        choices=("duddu", "qp"),
        default="duddu",
        help="Interface speed evaluation: 'duddu' uses Fig.3 shaded-triangle logic; 'qp' samples n·∇Phi at interface quadrature points.",
    )
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--timings", action="store_true", help="Print per-step timing breakdown.")

    # Output
    p.add_argument("--no-vtk", action="store_true", help="Disable VTK snapshots under <outdir>/vtk.")
    p.add_argument("--cmap-S", type=str, default="viridis", help="Colormap for substrate panel.")
    p.add_argument("--cmap-Phi", type=str, default="RdBu_r", help="Colormap for velocity potential panel.")

    args = p.parse_args()

    if str(args.linear_solver).lower() == "petsc" and not _HAS_PETSC:
        raise RuntimeError("PETSc requested but not available in this environment.")

    outdir = Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    # Paper times for Fig.5(a) (Example 1): "interface plotted after every 20 time steps".
    t_targets = np.asarray(
        [
            0.0,
            1.0,
            2.3,
            3.7,
            5.3,
            7.0,
            8.9,
            10.8,
            12.8,
            14.7,
            16.6,
            18.5,
            20.4,
            21.9,
            23.7,
            25.6,
            27.6,
            29.4,
            31.3,
            33.2,
            35.0,
            36.8,
            38.5,
            40.2,
            42.1,
            43.7,
            44.5,
        ],
        dtype=float,
    )

    # --- FD level set grid -------------------------------------------------
    grid = make_uniform_grid(Lx=float(args.L), Ly=float(args.H), nx=int(args.grid_nx), ny=int(args.grid_ny))
    phi = phi_union_disks_on_wall(
        grid,
        centers_x=[float(args.center_x)],
        radii=[float(args.radius)],
        wall_y=0.0,
    )
    phi = reinitialize_signed_distance(phi, dx=grid.dx, dy=grid.dy)

    # --- XFEM mesh ---------------------------------------------------------
    nodes, elems, edges, corners = structured_triangles(
        Lx=float(args.L),
        Ly=float(args.H),
        nx_quads=int(args.mesh_nx),
        ny_quads=int(args.mesh_ny),
        poly_order=1,
        offset=(0.0, 0.0),
    )
    mesh = Mesh(nodes, elems, edges, corners, element_type="tri", poly_order=1)
    _tag_rectangle_boundaries(mesh, L=float(args.L), H=float(args.H))

    # Mixed element (S, Phi) with optional enrichment on cut elements.
    me = MixedElement(mesh, field_specs={"S": 1, "Phi": 1})
    dh0 = DofHandler(me, method="cg")
    dh = XFEMDofHandler(dh0)

    kin = Duddu2007Kinetics(Y_xO=float(args.Y_xO), Y_wO=float(args.Y_wO))

    # State carry-over (base DOFs only; enriched DOFs rebuilt every step)
    S_base: np.ndarray | None = None
    Phi_base: np.ndarray | None = None

    # Time series for y_top
    y_top0 = _biofilm_top_y(grid_y=grid.y, phi_grid=phi)
    y_D0 = float(min(float(args.H), y_top0 + float(args.Ls)))
    rows: list[dict[str, object]] = [
        {
            "t_days": 0.0,
            "step": 0,
            "dt_days": 0.0,
            "y_top_mm": float(y_top0),
            "y_D_mm": float(y_D0),
            "maxF_mm_per_day": float("nan"),
        }
    ]

    # Snapshots (paper target times): store φ on FD grid.
    phi_snaps: list[tuple[float, np.ndarray]] = []
    next_target = 0

    # VTK snapshots at target times
    vtk_dir = outdir / "vtk"
    vtk_datasets: list[tuple[float, str]] = []
    if not bool(args.no_vtk):
        vtk_dir.mkdir(parents=True, exist_ok=True)

    # Convenience for VTK export (base functions only)
    dh_base = dh.base
    S_out = Function("S", "S", dof_handler=dh_base)
    Phi_out = Function("Phi", "Phi", dof_handler=dh_base)

    def _write_vtk_snapshot(*, snap_idx: int, t_days: float, phi_grid: np.ndarray, F_ext: np.ndarray, S_base: np.ndarray, Phi_base: np.ndarray) -> None:
        if bool(args.no_vtk):
            return
        S_out.nodal_values[:] = np.asarray(S_base, dtype=float)
        Phi_out.nodal_values[:] = np.asarray(Phi_base, dtype=float)
        xy = np.asarray(mesh.nodes_x_y_pos, dtype=float)
        phi_nodes = _sample_grid_bilinear(xg=grid.x, yg=grid.y, Z=np.asarray(phi_grid, float), x=xy[:, 0], y=xy[:, 1])
        F_nodes = _sample_grid_bilinear(xg=grid.x, yg=grid.y, Z=np.asarray(F_ext, float), x=xy[:, 0], y=xy[:, 1])
        out = vtk_dir / f"step={int(snap_idx):04d}.vtu"
        export_vtk(
            str(out),
            mesh=mesh,
            dof_handler=dh_base,
            functions={
                "phi_ls": np.asarray(phi_nodes, dtype=float),
                "F_ext": np.asarray(F_nodes, dtype=float),
                "S": S_out,
                "Phi": Phi_out,
            },
        )
        vtk_datasets.append((float(t_days), out.name))

    # Main loop: advance interface in time
    t = 0.0
    step = 0
    while t < float(args.t_final) - 1.0e-12:
        if step >= int(args.max_steps):
            raise RuntimeError(f"Reached max_steps={args.max_steps} at t={t:.3g} days.")

        t_step0 = time.perf_counter()

        # Build a piecewise-linear level set on the XFEM mesh for classification/integration.
        class _GridSampler:
            def __init__(self, x: np.ndarray, y: np.ndarray, phi: np.ndarray):
                self.x = np.asarray(x, float)
                self.y = np.asarray(y, float)
                self.phi = np.asarray(phi, float)

            def __call__(self, x):
                xx = float(np.asarray(x, float).ravel()[0])
                yy = float(np.asarray(x, float).ravel()[1])
                return float(_sample_grid_bilinear(xg=self.x, yg=self.y, Z=self.phi, x=xx, y=yy))

        t_ls0 = time.perf_counter()
        ls_mesh = PiecewiseLinearLevelSet.from_level_set(mesh, _GridSampler(grid.x, grid.y, phi))
        t_ls = time.perf_counter() - t_ls0

        # Refresh cut-element segments/tags + rebuild enrichment
        t_class0 = time.perf_counter()
        dh0.classify_from_levelset(ls_mesh)
        dh.rebuild_enrichment(ls_mesh, enrich={"S": "abs", "Phi": "abs"})
        t_class = time.perf_counter() - t_class0

        # Unknowns on current dof layout
        S_k = Function(name="S", field_name="S", dof_handler=dh)
        Phi_k = Function(name="Phi", field_name="Phi", dof_handler=dh)
        n_base_S = int(np.asarray(dh.base.get_field_slice("S"), dtype=int).size)
        n_base_Phi = int(np.asarray(dh.base.get_field_slice("Phi"), dtype=int).size)

        if S_base is None:
            S_base = np.full(n_base_S, float(args.Sbar), dtype=float)
            Phi_base = np.zeros(n_base_Phi, dtype=float)
        else:
            if int(np.asarray(S_base).size) != int(n_base_S):
                raise RuntimeError("Base DOF count changed unexpectedly for S.")
            if int(np.asarray(Phi_base).size) != int(n_base_Phi):
                raise RuntimeError("Base DOF count changed unexpectedly for Phi.")

        # Initialize iterates (base carry-over + 0 enrichment)
        S_k.nodal_values[:] = 0.0
        Phi_k.nodal_values[:] = 0.0
        S_k.nodal_values[:n_base_S] = np.asarray(S_base, dtype=float)
        Phi_k.nodal_values[:n_base_Phi] = np.asarray(Phi_base, dtype=float)

        # Moving substrate Dirichlet boundary Γ_S^d at y = y_top + Ls
        y_top = _biofilm_top_y(grid_y=grid.y, phi_grid=phi)
        y_D = float(min(float(args.H), y_top + float(args.Ls)))
        ls_Sd = AffineLevelSet(0.0, 1.0, -float(y_D))

        q = int(args.q)
        dx_pos = dx(level_set=ls_mesh, metadata={"side": "+", "q": q})
        dx_neg = dx(level_set=ls_mesh, metadata={"side": "-", "q": q})
        dGamma = dInterface(defined_on=mesh.element_bitset("cut"), level_set=ls_mesh, metadata={"q": q})
        dGammaS = dInterface(level_set=ls_Sd, metadata={"q": q, "linear_interface": True})

        vS = TestFunction("S", dof_handler=dh)
        dS = TrialFunction("S", dof_handler=dh)
        vPhi = TestFunction("Phi", dof_handler=dh)
        dPhi = TrialFunction("Phi", dof_handler=dh)

        Db = Constant(float(args.Db))
        Df = Constant(float(args.Df))
        pen = Constant(float(args.phi_penalty))
        h = CellDiameter()
        penS = Constant(float(args.S_penalty))
        Sbar = Constant(float(args.Sbar))

        R = kin.consumption(S_k)
        dR = kin.d_consumption(S_k)
        f_src = kin.divU(S_k)

        r_S = Db * inner(grad(S_k), grad(vS)) * dx_neg + Df * inner(grad(S_k), grad(vS)) * dx_pos + R * vS * dx_neg
        a_S = Db * inner(grad(dS), grad(vS)) * dx_neg + Df * inner(grad(dS), grad(vS)) * dx_pos + dR * dS * vS * dx_neg
        r_S += (penS / h) * (S_k - Sbar) * vS * dGammaS
        a_S += (penS / h) * dS * vS * dGammaS

        r_Phi = inner(grad(Phi_k), grad(vPhi)) * dx_neg + inner(grad(Phi_k), grad(vPhi)) * dx_pos
        r_Phi += f_src * vPhi * dx_neg + (pen / h) * Phi_k * vPhi * dGamma
        a_Phi = inner(grad(dPhi), grad(vPhi)) * dx_neg + inner(grad(dPhi), grad(vPhi)) * dx_pos
        a_Phi += (pen / h) * dPhi * vPhi * dGamma

        slS = np.asarray(dh.get_field_slice("S"), dtype=int)
        slPhi = np.asarray(dh.get_field_slice("Phi"), dtype=int)

        # Newton iterations for substrate S (nonlinear)
        S_min = float(args.S_min)
        user_damp = float(args.newton_damping)
        if not (0.0 < user_damp <= 1.0):
            raise ValueError("--newton-damping must satisfy 0 < d <= 1.")
        t_newton0 = time.perf_counter()
        newton_iters = 0
        for it in range(int(args.max_it)):
            A_full, r_vec_full = assemble_form(
                Equation(a_S, r_S),
                dof_handler=dh,
                bcs=[],
                backend=str(args.backend),
            )
            rS = np.asarray(r_vec_full[slS], dtype=float)
            rS_norm = float(np.linalg.norm(rS))
            if bool(getattr(args, "newton_verbose", False)):
                print(f"[newton] step={step:04d} it={it:02d}  ||r_S||={rS_norm:.3e}")
            if rS_norm <= float(args.newton_tol):
                newton_iters = int(it) + 1
                break

            A_SS = A_full[slS, :][:, slS]
            dS_vec = _solve_linear_system(A_SS, -rS, linear_solver=str(args.linear_solver))
            S_old = np.asarray(S_k.nodal_values, dtype=float)
            damp = float(user_damp)
            if S_min > 0.0:
                ds_base = np.asarray(dS_vec[:n_base_S], dtype=float)
                s_base = np.asarray(S_old[:n_base_S], dtype=float)
                mask = ds_base < 0.0
                if bool(np.any(mask)):
                    alpha = np.min((s_base[mask] - S_min) / (-ds_base[mask]))
                    if np.isfinite(alpha):
                        damp = float(min(damp, 0.99 * float(alpha)))
            S_k.nodal_values[:] = S_old + float(damp) * dS_vec
        else:
            raise RuntimeError("Substrate Newton did not converge.")
        if newton_iters == 0:
            newton_iters = int(args.max_it)
        t_newton = time.perf_counter() - t_newton0

        # Solve Φ once (linear) on converged S
        t_phi0 = time.perf_counter()
        f_src = kin.divU(S_k)
        r_Phi = inner(grad(Phi_k), grad(vPhi)) * dx_neg + inner(grad(Phi_k), grad(vPhi)) * dx_pos
        r_Phi += f_src * vPhi * dx_neg + (pen / h) * Phi_k * vPhi * dGamma
        a_Phi = inner(grad(dPhi), grad(vPhi)) * dx_neg + inner(grad(dPhi), grad(vPhi)) * dx_pos
        a_Phi += (pen / h) * dPhi * vPhi * dGamma

        A_full, r_vec_full = assemble_form(Equation(a_Phi, r_Phi), dof_handler=dh, bcs=[], backend=str(args.backend))
        rP = np.asarray(r_vec_full[slPhi], dtype=float)
        A_PP = A_full[slPhi, :][:, slPhi]
        dPhi_vec = _solve_linear_system(A_PP, -rP, linear_solver=str(args.linear_solver))
        Phi_k.nodal_values[:] = np.asarray(Phi_k.nodal_values, dtype=float) + np.asarray(dPhi_vec, dtype=float)
        t_phi = time.perf_counter() - t_phi0

        # Update base carry-over arrays (for next step)
        S_base = np.asarray(S_k.nodal_values[:n_base_S], dtype=float).copy()
        Phi_base = np.asarray(Phi_k.nodal_values[:n_base_Phi], dtype=float).copy()

        # Record snapshots at paper target times *at the current interface*.
        if next_target < int(t_targets.size) and abs(float(t_targets[next_target]) - t) <= 1.0e-10:
            phi_snaps.append((float(t), np.asarray(phi, float).copy()))
            # Use the current speed field for VTK export.
            snap_idx = int(next_target)
        else:
            snap_idx = -1

        # --- speed on interface (Duddu 5.2) ---------------------------------
        t_speed0 = time.perf_counter()
        speed_mode = str(getattr(args, "speed_mode", "qp")).strip().lower()
        if speed_mode == "duddu":
            segs = compute_interface_segment_speeds_duddu2007(dof_handler=dh, level_set=ls_mesh, Phi=Phi_k, field="Phi")
            pts, spd = sample_segment_speeds(segs, samples_per_segment=3)
        elif speed_mode == "qp":
            cut_eids = np.asarray(mesh.element_bitset("cut").to_indices(), dtype=int)
            geo = dh.precompute_interface_factors(
                cut_eids,
                qdeg=int(q),
                level_set=ls_mesh,
                linear_interface=True,
                reuse=False,
            )
            eids = np.asarray(geo.get("eids", np.empty((0,), dtype=np.int32)), dtype=int).ravel()
            if eids.size == 0:
                pts = np.zeros((0, 2), dtype=float)
                spd = np.zeros((0,), dtype=float)
            else:
                gdofs_map = np.asarray(geo["gdofs_map"], dtype=int)
                gPhi = np.asarray(geo["g_Phi"], dtype=float)
                bPhi = np.asarray(geo["b_Phi"], dtype=float)
                normals = np.asarray(geo["normals"], dtype=float)
                qp_phys = np.asarray(geo["qp_phys"], dtype=float)

                u_loc = np.zeros((int(gdofs_map.shape[0]), int(gdofs_map.shape[1])), dtype=float)
                for i in range(int(gdofs_map.shape[0])):
                    u_loc[i, :] = np.asarray(Phi_k.get_nodal_values(gdofs_map[i, :]), dtype=float)

                gradPhi = np.einsum("ek,eqkd->eqd", u_loc, gPhi)  # (nE,nQ,2)

                # Add shifted-|phi| enrichment contribution N_i ∇|phi| with one-sided ∇|phi| = -∇phi in Ω_b (phi<0).
                eids_list = eids.tolist()
                grad_phi_e = np.zeros((int(eids.size), 2), dtype=float)
                for i, eid in enumerate(eids_list):
                    # Midpoint evaluation with eid hint ensures correct element choice.
                    try:
                        mid = np.mean(np.asarray(mesh.nodes_x_y_pos[np.asarray(mesh.corner_connectivity[eid], dtype=int), :], dtype=float), axis=0)
                        grad_phi_e[i, :] = np.asarray(ls_mesh.gradient(mid, eid=eid), dtype=float).reshape(2)
                    except Exception:
                        grad_phi_e[i, :] = np.asarray(ls_mesh.gradient(qp_phys[i, 0, :]), dtype=float).reshape(2)
                grad_abs_phi = -grad_phi_e[:, None, :]  # (nE,1,2)

                me_xfem = dh.xfem_mixed_element()
                sl_phi = me_xfem.component_dof_slices["Phi"]
                n_phi_loc = int(sl_phi.stop - sl_phi.start)
                n0 = int(n_phi_loc // 2)
                phi_enr = u_loc[:, sl_phi.start + n0 : sl_phi.start + 2 * n0]
                N_base = bPhi[:, :, sl_phi.start : sl_phi.start + n0]
                gradPhi += np.einsum("eqi,ei,eqd->eqd", N_base, phi_enr, grad_abs_phi)

                F_q = np.einsum("eqd,eqd->eq", normals, gradPhi)
                pts = qp_phys.reshape(-1, 2)
                spd = F_q.reshape(-1)
        else:
            raise ValueError(f"Unknown --speed-mode {speed_mode!r}.")

        F_ext = extend_speed_nearest_interface(grid, interface_points=pts, interface_speeds=spd)
        t_speed = time.perf_counter() - t_speed0

        maxF = float(np.max(np.abs(F_ext)))
        if not np.isfinite(maxF) or maxF <= 0.0:
            raise RuntimeError("Non-positive max speed; cannot advance level set.")

        # VTK snapshot for the current target time (exports the current interface + fields).
        if snap_idx >= 0:
            _write_vtk_snapshot(
                snap_idx=snap_idx,
                t_days=float(t),
                phi_grid=np.asarray(phi, float),
                F_ext=np.asarray(F_ext, float),
                S_base=np.asarray(S_base, float),
                Phi_base=np.asarray(Phi_base, float),
            )
            next_target += 1

        dt_max = float(min(grid.dx, grid.dy) / maxF)
        dt = max(float(args.dt_min), 0.8 * dt_max)
        if next_target < int(t_targets.size):
            t_next = float(t_targets[next_target])
            if t_next > t + 1.0e-12:
                dt = min(dt, t_next - t)
        if t + dt > float(args.t_final):
            dt = float(args.t_final) - t
        if dt <= 0.0:
            break

        # --- level set update ------------------------------------------------
        t_upd0 = time.perf_counter()
        phi = level_set_update(phi, F_ext=F_ext, dx=grid.dx, dy=grid.dy, dt=dt)
        if int(args.reinit_every) > 0 and (step + 1) % int(args.reinit_every) == 0:
            phi = reinitialize_signed_distance(phi, dx=grid.dx, dy=grid.dy)
        t_upd = time.perf_counter() - t_upd0

        t += float(dt)
        step += 1
        y_top_new = _biofilm_top_y(grid_y=grid.y, phi_grid=phi)
        y_D_new = float(min(float(args.H), float(y_top_new) + float(args.Ls)))
        rows.append(
            {
                "t_days": float(t),
                "step": int(step),
                "dt_days": float(dt),
                "y_top_mm": float(y_top_new),
                "y_D_mm": float(y_D_new),
                "maxF_mm_per_day": float(maxF),
            }
        )

        pe = max(1, int(getattr(args, "progress_every", 10)))
        if step == 1 or step % pe == 0:
            print(f"[step {step:04d}] t={t:.3f} d  dt={dt:.3e} d  maxF={maxF:.4g} mm/d  cut_elems={mesh.element_bitset('cut').cardinality()}")

        if bool(getattr(args, "timings", False)):
            t_step = time.perf_counter() - t_step0
            print(
                f"[timing] step={step:04d} total={t_step:.2f}s  ls={t_ls:.2f}s  classify={t_class:.2f}s  "
                f"newton={t_newton:.2f}s({newton_iters:d} it)  phi={t_phi:.2f}s  speed={t_speed:.2f}s  update={t_upd:.2f}s"
            )

    # Final solve on the final interface (so S,Phi correspond to the final contour)
    if abs(float(t) - float(args.t_final)) > 1.0e-8:
        raise RuntimeError(f"Did not reach t_final: t={t} vs t_final={args.t_final}")

    # Build level set + enrichment at final interface
    class _FinalSampler:
        def __init__(self, x: np.ndarray, y: np.ndarray, phi: np.ndarray):
            self.x = np.asarray(x, float)
            self.y = np.asarray(y, float)
            self.phi = np.asarray(phi, float)

        def __call__(self, x):
            xx = float(np.asarray(x, float).ravel()[0])
            yy = float(np.asarray(x, float).ravel()[1])
            return float(_sample_grid_bilinear(xg=self.x, yg=self.y, Z=self.phi, x=xx, y=yy))

    ls_mesh = PiecewiseLinearLevelSet.from_level_set(mesh, _FinalSampler(grid.x, grid.y, phi))
    dh0.classify_from_levelset(ls_mesh)
    dh.rebuild_enrichment(ls_mesh, enrich={"S": "abs", "Phi": "abs"})

    S_k = Function(name="S", field_name="S", dof_handler=dh)
    Phi_k = Function(name="Phi", field_name="Phi", dof_handler=dh)
    n_base_S = int(np.asarray(dh.base.get_field_slice("S"), dtype=int).size)
    n_base_Phi = int(np.asarray(dh.base.get_field_slice("Phi"), dtype=int).size)
    S_k.nodal_values[:] = 0.0
    Phi_k.nodal_values[:] = 0.0
    S_k.nodal_values[:n_base_S] = np.asarray(S_base, dtype=float)
    Phi_k.nodal_values[:n_base_Phi] = np.asarray(Phi_base, dtype=float)

    # Final moving substrate Dirichlet
    y_top = _biofilm_top_y(grid_y=grid.y, phi_grid=phi)
    y_D = float(min(float(args.H), y_top + float(args.Ls)))
    ls_Sd = AffineLevelSet(0.0, 1.0, -float(y_D))

    q = int(args.q)
    dx_pos = dx(level_set=ls_mesh, metadata={"side": "+", "q": q})
    dx_neg = dx(level_set=ls_mesh, metadata={"side": "-", "q": q})
    dGamma = dInterface(defined_on=mesh.element_bitset("cut"), level_set=ls_mesh, metadata={"q": q})
    dGammaS = dInterface(level_set=ls_Sd, metadata={"q": q, "linear_interface": True})

    vS = TestFunction("S", dof_handler=dh)
    dS = TrialFunction("S", dof_handler=dh)
    vPhi = TestFunction("Phi", dof_handler=dh)
    dPhi = TrialFunction("Phi", dof_handler=dh)

    Db = Constant(float(args.Db))
    Df = Constant(float(args.Df))
    pen = Constant(float(args.phi_penalty))
    h = CellDiameter()
    penS = Constant(float(args.S_penalty))
    Sbar = Constant(float(args.Sbar))

    R = kin.consumption(S_k)
    dR = kin.d_consumption(S_k)
    f_src = kin.divU(S_k)

    r_S = Db * inner(grad(S_k), grad(vS)) * dx_neg + Df * inner(grad(S_k), grad(vS)) * dx_pos + R * vS * dx_neg
    a_S = Db * inner(grad(dS), grad(vS)) * dx_neg + Df * inner(grad(dS), grad(vS)) * dx_pos + dR * dS * vS * dx_neg
    r_S += (penS / h) * (S_k - Sbar) * vS * dGammaS
    a_S += (penS / h) * dS * vS * dGammaS

    r_Phi = inner(grad(Phi_k), grad(vPhi)) * dx_neg + inner(grad(Phi_k), grad(vPhi)) * dx_pos
    r_Phi += f_src * vPhi * dx_neg + (pen / h) * Phi_k * vPhi * dGamma
    a_Phi = inner(grad(dPhi), grad(vPhi)) * dx_neg + inner(grad(dPhi), grad(vPhi)) * dx_pos
    a_Phi += (pen / h) * dPhi * vPhi * dGamma

    slS = np.asarray(dh.get_field_slice("S"), dtype=int)
    slPhi = np.asarray(dh.get_field_slice("Phi"), dtype=int)

    # Newton for final S
    for _it in range(int(args.max_it)):
        A_full, r_vec_full = assemble_form(Equation(a_S, r_S), dof_handler=dh, bcs=[], backend=str(args.backend))
        rS = np.asarray(r_vec_full[slS], dtype=float)
        if float(np.linalg.norm(rS)) <= float(args.newton_tol):
            break
        A_SS = A_full[slS, :][:, slS]
        dS_vec = _solve_linear_system(A_SS, -rS, linear_solver=str(args.linear_solver))
        S_k.nodal_values[:] = np.asarray(S_k.nodal_values, dtype=float) + np.asarray(dS_vec, dtype=float)
    else:
        raise RuntimeError("Final substrate Newton did not converge.")

    # Final Phi
    f_src = kin.divU(S_k)
    r_Phi = inner(grad(Phi_k), grad(vPhi)) * dx_neg + inner(grad(Phi_k), grad(vPhi)) * dx_pos
    r_Phi += f_src * vPhi * dx_neg + (pen / h) * Phi_k * vPhi * dGamma
    a_Phi = inner(grad(dPhi), grad(vPhi)) * dx_neg + inner(grad(dPhi), grad(vPhi)) * dx_pos
    a_Phi += (pen / h) * dPhi * vPhi * dGamma
    A_full, r_vec_full = assemble_form(Equation(a_Phi, r_Phi), dof_handler=dh, bcs=[], backend=str(args.backend))
    rP = np.asarray(r_vec_full[slPhi], dtype=float)
    A_PP = A_full[slPhi, :][:, slPhi]
    dPhi_vec = _solve_linear_system(A_PP, -rP, linear_solver=str(args.linear_solver))
    Phi_k.nodal_values[:] = np.asarray(Phi_k.nodal_values, dtype=float) + np.asarray(dPhi_vec, dtype=float)

    S_base_final = np.asarray(S_k.nodal_values[:n_base_S], dtype=float).copy()
    Phi_base_final = np.asarray(Phi_k.nodal_values[:n_base_Phi], dtype=float).copy()

    # Speed field at final time (for VTK export / inspection).
    speed_mode = str(getattr(args, "speed_mode", "qp")).strip().lower()
    if speed_mode == "duddu":
        segs = compute_interface_segment_speeds_duddu2007(dof_handler=dh, level_set=ls_mesh, Phi=Phi_k, field="Phi")
        pts, spd = sample_segment_speeds(segs, samples_per_segment=3)
    elif speed_mode == "qp":
        cut_eids = np.asarray(mesh.element_bitset("cut").to_indices(), dtype=int)
        geo = dh.precompute_interface_factors(
            cut_eids,
            qdeg=int(q),
            level_set=ls_mesh,
            linear_interface=True,
            reuse=False,
        )
        eids = np.asarray(geo.get("eids", np.empty((0,), dtype=np.int32)), dtype=int).ravel()
        if eids.size == 0:
            pts = np.zeros((0, 2), dtype=float)
            spd = np.zeros((0,), dtype=float)
        else:
            gdofs_map = np.asarray(geo["gdofs_map"], dtype=int)
            gPhi = np.asarray(geo["g_Phi"], dtype=float)
            bPhi = np.asarray(geo["b_Phi"], dtype=float)
            normals = np.asarray(geo["normals"], dtype=float)
            qp_phys = np.asarray(geo["qp_phys"], dtype=float)

            u_loc = np.zeros((int(gdofs_map.shape[0]), int(gdofs_map.shape[1])), dtype=float)
            for i in range(int(gdofs_map.shape[0])):
                u_loc[i, :] = np.asarray(Phi_k.get_nodal_values(gdofs_map[i, :]), dtype=float)

            gradPhi = np.einsum("ek,eqkd->eqd", u_loc, gPhi)  # (nE,nQ,2)

            eids_list = eids.tolist()
            grad_phi_e = np.zeros((int(eids.size), 2), dtype=float)
            for i, eid in enumerate(eids_list):
                try:
                    mid = np.mean(np.asarray(mesh.nodes_x_y_pos[np.asarray(mesh.corner_connectivity[eid], dtype=int), :], dtype=float), axis=0)
                    grad_phi_e[i, :] = np.asarray(ls_mesh.gradient(mid, eid=eid), dtype=float).reshape(2)
                except Exception:
                    grad_phi_e[i, :] = np.asarray(ls_mesh.gradient(qp_phys[i, 0, :]), dtype=float).reshape(2)
            grad_abs_phi = -grad_phi_e[:, None, :]

            me_xfem = dh.xfem_mixed_element()
            sl_phi = me_xfem.component_dof_slices["Phi"]
            n_phi_loc = int(sl_phi.stop - sl_phi.start)
            n0 = int(n_phi_loc // 2)
            phi_enr = u_loc[:, sl_phi.start + n0 : sl_phi.start + 2 * n0]
            N_base = bPhi[:, :, sl_phi.start : sl_phi.start + n0]
            gradPhi += np.einsum("eqi,ei,eqd->eqd", N_base, phi_enr, grad_abs_phi)

            F_q = np.einsum("eqd,eqd->eq", normals, gradPhi)
            pts = qp_phys.reshape(-1, 2)
            spd = F_q.reshape(-1)
    else:
        raise ValueError(f"Unknown --speed-mode {speed_mode!r}.")
    F_ext_final = extend_speed_nearest_interface(grid, interface_points=pts, interface_speeds=spd)

    # Make sure the final target time is represented in the stored snapshots.
    if (next_target < int(t_targets.size)) and abs(float(t_targets[next_target]) - float(t)) <= 1.0e-10:
        phi_snaps.append((float(t), np.asarray(phi, float).copy()))
        _write_vtk_snapshot(
            snap_idx=int(next_target),
            t_days=float(t),
            phi_grid=np.asarray(phi, float),
            F_ext=np.asarray(F_ext_final, float),
            S_base=S_base_final,
            Phi_base=Phi_base_final,
        )
        next_target += 1

    # --- outputs ------------------------------------------------------------
    np.savez_compressed(
        outdir / "snaps_phi.npz",
        t_days=np.asarray([t for t, _ in phi_snaps], dtype=np.float64),
        phi=np.asarray([np.asarray(p, dtype=np.float32) for _t, p in phi_snaps], dtype=np.float32),
    )

    _plot_level_set_contours_colored(
        phi_snaps,
        grid_x=grid.x,
        grid_y=grid.y,
        outpng=outdir / "fig5a_interface.png",
        title="Example 1: interface motion (colored by time)",
    )
    _plot_scalar_on_mesh(
        dh_base=dh_base,
        values_base=S_base_final,
        field="S",
        title=f"Example 1: substrate S at t={t:.1f} days",
        outpng=outdir / "fig5b_S.png",
        cmap=str(args.cmap_S),
        grid_x=grid.x,
        grid_y=grid.y,
        phi_grid=phi,
    )
    _plot_scalar_on_mesh(
        dh_base=dh_base,
        values_base=Phi_base_final,
        field="Phi",
        title=f"Example 1: velocity potential $\\Phi$ at t={t:.1f} days",
        outpng=outdir / "fig5c_Phi.png",
        cmap=str(args.cmap_Phi),
        grid_x=grid.x,
        grid_y=grid.y,
        phi_grid=phi,
    )

    _write_csv(outdir / "y_top_timeseries.csv", rows)

    summary = {
        "t_final_days": float(t),
        "steps": int(step),
        "mesh_nx": int(args.mesh_nx),
        "mesh_ny": int(args.mesh_ny),
        "grid_nx": int(args.grid_nx),
        "grid_ny": int(args.grid_ny),
        "center_x_mm": float(args.center_x),
        "radius_mm": float(args.radius),
        "substrate_bc": "moving",
        "Ls_mm": float(args.Ls),
        "S_penalty": float(args.S_penalty),
        "Sbar": float(args.Sbar),
        "y_top_final_mm": float(_biofilm_top_y(grid_y=grid.y, phi_grid=phi)),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if not bool(args.no_vtk) and vtk_datasets:
        _write_pvd(out_pvd=vtk_dir / "series.pvd", datasets=vtk_datasets)
        print(f"- Wrote {vtk_dir/'series.pvd'} ({len(vtk_datasets)} dataset(s))")

    print(f"- Wrote {outdir/'fig5a_interface.png'}")
    print(f"- Wrote {outdir/'fig5b_S.png'}")
    print(f"- Wrote {outdir/'fig5c_Phi.png'}")
    print(f"- Wrote {outdir/'y_top_timeseries.csv'}")
    print(f"- Wrote {outdir/'snaps_phi.npz'}")
    print(f"- Wrote {outdir/'summary.json'}")


if __name__ == "__main__":
    main()
