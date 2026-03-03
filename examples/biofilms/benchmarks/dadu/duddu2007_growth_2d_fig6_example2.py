"""
Duddu et al. (2007) growth-only model (XFEM + level set): 2D Example 2 (Fig. 6).

Paper reference
---------------
R. Duddu, S. Bordas, D. L. Chopp, B. Moran (2007)
"A combined extended finite element and level set method for biofilm growth"
Int. J. Numer. Meth. Engng.

We reproduce Example 2 (Fig. 6):
  - three semi-circular colonies attached to the bottom wall,
  - growth driven by quasi-steady substrate diffusion + velocity potential,
  - interface advanced by level set φ_t + F ||∇φ|| = 0 with adaptive dt.

Implementation notes
--------------------
- XFEM: shifted-|phi| ("abs") enrichment for S and Phi.
- Speed F: computed on Γ_int using Duddu (2007) Section 5.2 (Fig. 3) logic
  implemented in duddu2007_speed.py.
- Velocity extension: approximated by nearest-interface projection using an
  EDT-based closest-point extension (scipy.ndimage.distance_transform_edt).

All outputs are written under examples/biofilms/benchmarks/dadu/results/.
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

from pycutfem.core.levelset import PiecewiseLinearLevelSet
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.xfem import XFEMDofHandler
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    Function,
    Neg,
    TestFunction,
    TrialFunction,
    grad,
    inner,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dInterface, dx
from pycutfem.utils.meshgen import structured_triangles

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
    Y_wO: float = 0.215  # mg VS / mg O2 (aligned to Duddu 2007 Table I)
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
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
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


def _plot_level_set_contours(phi_snaps: list[tuple[float, np.ndarray]], *, grid_x: np.ndarray, grid_y: np.ndarray, outpng: Path) -> None:
    import matplotlib.pyplot as plt

    X, Y = np.meshgrid(grid_x, grid_y, indexing="xy")
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    for t, phi in phi_snaps:
        ax.contour(X, Y, phi, levels=[0.0], colors="k", linewidths=0.8, alpha=0.9)
    ax.set_xlim(float(grid_x[0]), float(grid_x[-1]))
    ax.set_ylim(float(grid_y[0]), float(grid_y[-1]))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Example 2: interface (every target time)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    fig.savefig(outpng, dpi=200)
    plt.close(fig)


def _plot_field_on_mesh(dh_base: DofHandler, values: np.ndarray, *, field: str, title: str, outpng: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    u_loc = np.asarray(values, dtype=float).ravel()
    gdofs = np.asarray(dh_base.get_field_slice(field), dtype=int).ravel()
    if gdofs.size != u_loc.size:
        raise ValueError(f"Field {field!r}: gdofs size {gdofs.size} != values size {u_loc.size}")
    gd_to_val = {int(gd): float(u_loc[i]) for i, gd in enumerate(gdofs.tolist())}

    mesh = dh_base.mixed_element.mesh
    n_nodes = int(np.asarray(mesh.nodes_x_y_pos, dtype=float).shape[0])
    node_to_gdof = (getattr(dh_base, "dof_map", {}) or {}).get(field) or {}
    if len(node_to_gdof) != n_nodes:
        raise RuntimeError(f"Field {field!r}: expected dof_map on all mesh nodes, got {len(node_to_gdof)} entries.")
    u_node = np.zeros(n_nodes, dtype=float)
    for nid in range(n_nodes):
        gd = int(node_to_gdof[int(nid)])
        u_node[int(nid)] = float(gd_to_val.get(gd, 0.0))

    # Triangulate from the underlying geometry mesh.
    tri_conn = np.asarray(mesh.corner_connectivity, dtype=int)
    tri = mtri.Triangulation(mesh.nodes_x_y_pos[:, 0], mesh.nodes_x_y_pos[:, 1], tri_conn)

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    cmap = "gray"
    if field == "S":
        # Duddu (2007) Fig. 6(b) caption: "darker areas are rich in substrate".
        # Matplotlib's "gray" maps low->black, high->white, so use the reversed map.
        cmap = "gray_r"
    tcf = ax.tricontourf(tri, u_node, levels=20, cmap=cmap)
    fig.colorbar(tcf, ax=ax, shrink=0.85)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.savefig(outpng, dpi=200)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    # Keep compiler logs quiet by default (they can dominate runtime on long runs).
    logging.getLogger("pycutfem").setLevel(logging.WARNING)
    logging.getLogger("pycutfem.ufl.compilers").setLevel(logging.WARNING)

    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="examples/biofilms/benchmarks/dadu/results/duddu2007_fig6_example2")
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

    # Example 2 initial colonies (centers on bottom wall)
    p.add_argument(
        "--centers-x",
        type=str,
        default="0.05,0.25,0.45",
        help="Comma-separated colony center x-positions (mm).",
    )
    p.add_argument("--radii", type=str, default="0.01,0.02,0.01", help="Comma-separated colony radii (mm).")

    # Model params
    p.add_argument("--Sbar", type=float, default=8.3e-6)
    p.add_argument("--Db", type=float, default=146.88)
    p.add_argument("--Df", type=float, default=183.6)
    p.add_argument("--Y-xO", type=float, default=Duddu2007Kinetics.Y_xO)
    p.add_argument("--Y-wO", type=float, default=Duddu2007Kinetics.Y_wO)

    # Time stepping (paper: adaptive dt=0.8*min(dx,dy)/max(F))
    p.add_argument("--t-final", type=float, default=28.6, help="Final time (days).")
    p.add_argument("--dt-min", type=float, default=1.0e-4, help="Lower bound on adaptive dt (days).")
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--reinit-every", type=int, default=1, help="Reinitialize signed distance every N steps (0=off).")

    # Substrate source boundary (paper: 0.1mm above top-most biofilm point)
    p.add_argument("--substrate-bc", choices=("moving", "top"), default="moving")
    p.add_argument("--Ls", type=float, default=0.1, help="Distance from top-most biofilm point to Γ_S^d (mm).")
    p.add_argument("--S-penalty", type=float, default=1.0e6, help="Penalty parameter for S Dirichlet on Γ_S^d.")

    # Newton / penalty
    p.add_argument("--newton-tol", type=float, default=1.0e-10)
    p.add_argument("--max-it", type=int, default=50)
    p.add_argument("--newton-verbose", action="store_true")
    p.add_argument("--newton-damping", type=float, default=1.0, help="Constant damping factor for Newton updates (0<d≤1).")
    # Note: enforcing S>=0 at each Newton update via clipping can stall convergence on coarse meshes.
    # Use --S-min only if needed to prevent large negative values (e.g. to stay away from S≈-K0).
    p.add_argument("--S-min", type=float, default=0.0)
    p.add_argument("--phi-penalty", type=float, default=1.0e6)
    p.add_argument(
        "--speed-mode",
        choices=("duddu", "qp"),
        default="qp",
        help="Interface speed evaluation: 'duddu' uses Fig.3 shaded-triangle logic; 'qp' samples n·∇Phi at interface quadrature points.",
    )
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--timings", action="store_true", help="Print per-step timing breakdown.")

    args = p.parse_args()

    def _biofilm_top_y(phi_grid: np.ndarray) -> float:
        """
        Approximate y_max on Γ_int from the FD level set grid using vertical
        sign-change detection and linear interpolation.
        """
        phi_grid = np.asarray(phi_grid, float)
        y = np.asarray(grid.y, float)
        top = 0.0
        # Scan each x-column independently.
        for i in range(int(phi_grid.shape[1])):
            col = np.asarray(phi_grid[:, i], float).ravel()
            # Find all crossings from inside (φ<=0) to outside (φ>0).
            for j in range(int(col.size) - 1):
                a = float(col[j])
                b = float(col[j + 1])
                if (a <= 0.0) and (b > 0.0) and (b != a):
                    t01 = a / (a - b)  # in [0,1]
                    y_int = float(y[j] + t01 * (y[j + 1] - y[j]))
                    top = max(top, y_int)
        return float(top)

    outdir = Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    # Paper times for Fig.6(a) (Example 2), every 20 time steps.
    t_targets = np.asarray(
        [0.0, 1.1, 2.7, 4.6, 6.6, 8.7, 10.7, 12.7, 14.7, 16.7, 18.6, 20.6, 22.5, 24.5, 26.5, 28.6],
        dtype=float,
    )

    centers_x = [float(s.strip()) for s in str(args.centers_x).split(",") if s.strip()]
    radii = [float(s.strip()) for s in str(args.radii).split(",") if s.strip()]
    if len(centers_x) != len(radii):
        raise ValueError("--centers-x and --radii must have the same length.")

    # --- FD level set grid -------------------------------------------------
    grid = make_uniform_grid(Lx=float(args.L), Ly=float(args.H), nx=int(args.grid_nx), ny=int(args.grid_ny))
    phi = phi_union_disks_on_wall(grid, centers_x=centers_x, radii=radii, wall_y=0.0)
    phi = reinitialize_signed_distance(phi, dx=grid.dx, dy=grid.dy)

    # --- background mesh ---------------------------------------------------
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

    me = MixedElement(mesh, {"S": 1, "Phi": 1})
    dh0 = DofHandler(me, method="cg")
    dh = XFEMDofHandler(dh0)

    kin = Duddu2007Kinetics(Y_xO=float(args.Y_xO), Y_wO=float(args.Y_wO))

    # Base solution carry-over (only base DOFs live here).
    S_base = None
    Phi_base = None

    # Snapshots for interface plot
    phi_snaps: list[tuple[float, np.ndarray]] = []
    next_snap = 0
    rows: list[dict[str, object]] = []

    t = 0.0
    step = 0
    # Store initial snapshot if requested
    if next_snap < int(t_targets.size) and abs(float(t_targets[next_snap]) - t) <= 1.0e-12:
        phi_snaps.append((float(t), np.asarray(phi, float).copy()))
        next_snap += 1
    y_top0 = _biofilm_top_y(phi)
    y_D0 = float(min(float(args.H), float(y_top0) + float(args.Ls)))
    rows.append({"t_days": float(t), "step": int(step), "dt_days": 0.0, "y_top_mm": float(y_top0), "y_D_mm": float(y_D0), "maxF_mm_per_day": None})

    while (t < float(args.t_final) - 1e-12) and (step < int(args.max_steps)):
        t_step0 = time.perf_counter()
        # --- build mesh-level P1 surrogate for XFEM (sample FD grid on mesh nodes) ---
        class _GridSampler:
            __slots__ = ("x", "y", "phi", "dx", "dy", "nx", "ny")

            def __init__(self, grid_x: np.ndarray, grid_y: np.ndarray, phi_grid: np.ndarray):
                self.x = np.asarray(grid_x, float)
                self.y = np.asarray(grid_y, float)
                self.phi = np.asarray(phi_grid, float)
                self.dx = float(self.x[1] - self.x[0])
                self.dy = float(self.y[1] - self.y[0])
                self.nx = int(self.x.size) - 1
                self.ny = int(self.y.size) - 1

            def __call__(self, x):
                xx = float(np.asarray(x, float).ravel()[0])
                yy = float(np.asarray(x, float).ravel()[1])
                i = int(math.floor(xx / self.dx))
                j = int(math.floor(yy / self.dy))
                i = 0 if i < 0 else (self.nx - 1 if i >= self.nx else i)
                j = 0 if j < 0 else (self.ny - 1 if j >= self.ny else j)
                x0 = float(self.x[i])
                y0 = float(self.y[j])
                tx = (xx - x0) / self.dx
                ty = (yy - y0) / self.dy
                p00 = float(self.phi[j, i])
                p10 = float(self.phi[j, i + 1])
                p01 = float(self.phi[j + 1, i])
                p11 = float(self.phi[j + 1, i + 1])
                return (1.0 - tx) * (1.0 - ty) * p00 + tx * (1.0 - ty) * p10 + (1.0 - tx) * ty * p01 + tx * ty * p11

        t_ls0 = time.perf_counter()
        ls_mesh = PiecewiseLinearLevelSet.from_level_set(mesh, _GridSampler(grid.x, grid.y, phi))
        t_ls = time.perf_counter() - t_ls0
        if bool(getattr(args, "timings", False)):
            print(f"[timing] step={step:04d} ls_mesh={t_ls:.2f}s")

        # Refresh cut-element segments/tags
        t_class0 = time.perf_counter()
        dh0.classify_from_levelset(ls_mesh)
        dh.rebuild_enrichment(ls_mesh, enrich={"S": "abs", "Phi": "abs"})
        t_class = time.perf_counter() - t_class0
        if bool(getattr(args, "timings", False)):
            print(f"[timing] step={step:04d} classify+enrich={t_class:.2f}s")

        # Create new unknowns on the current dof layout
        S_k = Function(name="S", field_name="S", dof_handler=dh)
        Phi_k = Function(name="Phi", field_name="Phi", dof_handler=dh)

        n_base_S = int(np.asarray(dh.base.get_field_slice("S"), dtype=int).size)
        n_base_Phi = int(np.asarray(dh.base.get_field_slice("Phi"), dtype=int).size)

        if S_base is None:
            S_base = np.full(n_base_S, float(args.Sbar), dtype=float)
            Phi_base = np.zeros(n_base_Phi, dtype=float)
        else:
            # Carry over base DOFs (mesh nodes) and reset any new enrichment DOFs to 0.
            if int(np.asarray(S_base).size) != int(n_base_S):
                # Mesh is fixed; base DOF count should not change.
                raise RuntimeError("Base DOF count changed unexpectedly.")
            if int(np.asarray(Phi_base).size) != int(n_base_Phi):
                raise RuntimeError("Base DOF count changed unexpectedly.")

        S_k.nodal_values[:] = 0.0
        Phi_k.nodal_values[:] = 0.0
        S_k.nodal_values[:n_base_S] = np.asarray(S_base, dtype=float)
        Phi_k.nodal_values[:n_base_Phi] = np.asarray(Phi_base, dtype=float)

        # Substrate boundary Γ_S^d
        if str(args.substrate_bc).lower() == "top":
            bc_top_S_val = BoundaryCondition("S", "dirichlet", "top", lambda x, y: float(args.Sbar))
            bc_top_S_homog = BoundaryCondition("S", "dirichlet", "top", lambda x, y: 0.0)

            # Enforce Dirichlet values in the iterate (as in 1D script)
            S_dir = dh.get_dirichlet_data([bc_top_S_val])
            for gd, vv in S_dir.items():
                li = S_k._g2l.get(int(gd))
                if li is not None:
                    S_k.nodal_values[int(li)] = float(vv)
        else:
            bc_top_S_val = None
            bc_top_S_homog = None
            S_dir = {}

            y_top = _biofilm_top_y(phi)
            y_D = float(min(float(args.H), y_top + float(args.Ls)))
            ls_Sd = AffineLevelSet(0.0, 1.0, -float(y_D))

        # Measures
        q = int(args.q)
        dx_pos = dx(level_set=ls_mesh, metadata={"side": "+", "q": q})
        dx_neg = dx(level_set=ls_mesh, metadata={"side": "-", "q": q})
        dGamma = dInterface(defined_on=mesh.element_bitset("cut"), level_set=ls_mesh, metadata={"q": q})
        if str(args.substrate_bc).lower() == "moving":
            dGammaS = dInterface(level_set=ls_Sd, metadata={"q": q, "linear_interface": True})
        else:
            dGammaS = None

        # Unknowns / forms
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
        if dGammaS is not None:
            # Penalty Dirichlet: S = Sbar on Γ_S^d (moving internal boundary).
            r_S += (penS / h) * (S_k - Sbar) * vS * dGammaS
            a_S += (penS / h) * dS * vS * dGammaS

        r_Phi = inner(grad(Phi_k), grad(vPhi)) * dx_neg + inner(grad(Phi_k), grad(vPhi)) * dx_pos
        r_Phi += f_src * vPhi * dx_neg + (pen / h) * Phi_k * vPhi * dGamma
        a_Phi = inner(grad(dPhi), grad(vPhi)) * dx_neg + inner(grad(dPhi), grad(vPhi)) * dx_pos
        a_Phi += (pen / h) * dPhi * vPhi * dGamma

        slS = np.asarray(dh.get_field_slice("S"), dtype=int)
        slPhi = np.asarray(dh.get_field_slice("Phi"), dtype=int)

        # Newton iterations (substrate only; Φ is linear once S is known)
        S_min = float(args.S_min)
        user_damp = float(args.newton_damping)
        if not (0.0 < user_damp <= 1.0):
            raise ValueError("--newton-damping must satisfy 0 < d <= 1.")
        t_newton0 = time.perf_counter()
        newton_iters = 0
        for it in range(int(args.max_it)):
            # keep Dirichlet values fixed
            if S_dir:
                for gd, vv in S_dir.items():
                    li = S_k._g2l.get(int(gd))
                    if li is not None:
                        S_k.nodal_values[int(li)] = float(vv)

            A_full, r_vec_full = assemble_form(
                Equation(a_S, r_S),
                dof_handler=dh,
                bcs=[bc_top_S_homog] if bc_top_S_homog is not None else [],
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
                    # Keep the *base* nodal dofs non-negative by damping, instead of clipping
                    # (clipping breaks Newton convergence for coarse meshes).
                    alpha = np.min((s_base[mask] - S_min) / (-ds_base[mask]))
                    if np.isfinite(alpha):
                        damp = float(min(damp, 0.99 * float(alpha)))
            S_k.nodal_values[:] = S_old + float(damp) * dS_vec
        else:
            raise RuntimeError("Substrate Newton did not converge.")
        if newton_iters == 0:
            newton_iters = int(args.max_it)
        t_newton = time.perf_counter() - t_newton0

        # Solve Φ once (linear) on the converged S.
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

        # Update base carry-over arrays
        S_base = np.asarray(S_k.nodal_values[:n_base_S], dtype=float).copy()
        Phi_base = np.asarray(Phi_k.nodal_values[:n_base_Phi], dtype=float).copy()

        # --- speed on interface (Duddu 5.2) ---------------------------------
        t_speed0 = time.perf_counter()
        speed_mode = str(getattr(args, "speed_mode", "qp")).strip().lower()
        if speed_mode == "duddu":
            segs = compute_interface_segment_speeds_duddu2007(dof_handler=dh, level_set=ls_mesh, Phi=Phi_k, field="Phi")
            if bool(getattr(args, "timings", False)) and segs:
                seg_lens = np.asarray([s.length for s in segs], dtype=float)
                print(
                    f"[speed] segments={len(segs)}  seg_len[min/med/max]={np.min(seg_lens):.3g}/{np.median(seg_lens):.3g}/{np.max(seg_lens):.3g} mm"
                )
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
                bPhi = np.asarray(geo["b_Phi"], dtype=float)
                gPhi = np.asarray(geo["g_Phi"], dtype=float)
                normals = np.asarray(geo["normals"], dtype=float)
                qp_phys = np.asarray(geo["qp_phys"], dtype=float)

                # Gather element-local Phi coefficients (zeros for non-Phi DOFs).
                u_loc = np.zeros((int(gdofs_map.shape[0]), int(gdofs_map.shape[1])), dtype=float)
                for i in range(int(gdofs_map.shape[0])):
                    u_loc[i, :] = np.asarray(Phi_k.get_nodal_values(gdofs_map[i, :]), dtype=float)

                # Base contribution + partial abs-enrichment contribution from precomputed tables.
                gradPhi = np.einsum("ek,eqkd->eqd", u_loc, gPhi)  # (nE,nQ,2)

                # For shifted-|phi| enrichment, the true gradient includes an extra term:
                #   ∇(N_i (|phi|-|phi_i|)) = (|phi|-|phi_i|) ∇N_i + N_i ∇|phi|.
                # XFEM interface precompute b_/g_ tables include only the first part. Add the
                # one-sided second part using ∇|phi| = sign(phi) ∇phi.
                # Compute per-element ∇phi (constant on each P1 triangle) and broadcast to QPs.
                grad_phi_e = np.zeros((int(eids.size), 2), dtype=float)
                for i, eid in enumerate(eids.tolist()):
                    # Use the piecewise-linear surrogate's constant gradient on the element.
                    try:
                        # Midpoint evaluation with eid hint ensures correct element choice.
                        grad_phi_e[i, :] = np.asarray(ls_mesh.gradient(qp_phys[i, 0, :], eid=int(eid)), dtype=float).reshape(2)
                    except Exception:
                        grad_phi_e[i, :] = np.asarray(ls_mesh.gradient(qp_phys[i, 0, :]), dtype=float).reshape(2)
                # One-sided (biofilm) sign: phi<0 => ∇|phi| = -∇phi.
                grad_abs_phi = -grad_phi_e[:, None, :]  # (nE,1,2)

                # Enriched coefficients are stored after the base block within the Phi slice. With only S and Phi
                # fields (both enriched), this is stable: [S_base,S_enr,Phi_base,Phi_enr].
                me_xfem = dh.xfem_mixed_element()
                sl_phi = me_xfem.component_dof_slices["Phi"]
                # base and enriched halves within the Phi slice
                n_phi_loc = int(sl_phi.stop - sl_phi.start)
                n0 = int(n_phi_loc // 2)
                phi_enr = u_loc[:, sl_phi.start + n0 : sl_phi.start + 2 * n0]
                # N_i on the interface (base basis values)
                N_base = bPhi[:, :, sl_phi.start : sl_phi.start + n0]
                gradPhi += np.einsum("eqi,ei,eqd->eqd", N_base, phi_enr, grad_abs_phi)

                F_q = np.einsum("eqd,eqd->eq", normals, gradPhi)  # (nE,nQ)
                pts = qp_phys.reshape(-1, 2)
                spd = F_q.reshape(-1)
        else:
            raise ValueError(f"Unknown --speed-mode {speed_mode!r}.")

        if bool(getattr(args, "timings", False)) and np.asarray(spd).size:
            spd_abs = np.abs(np.asarray(spd, float).ravel())
            spd_abs = spd_abs[np.isfinite(spd_abs)]
            if spd_abs.size:
                qv = np.quantile(spd_abs, [0.0, 0.5, 0.95, 0.99, 1.0])
                print(
                    f"[speed] |F|[min/med/p95/p99/max]={qv[0]:.3g}/{qv[1]:.3g}/{qv[2]:.3g}/{qv[3]:.3g}/{qv[4]:.3g} mm/d  n={spd_abs.size}"
                )
        F_ext = extend_speed_nearest_interface(grid, interface_points=pts, interface_speeds=spd)
        t_speed = time.perf_counter() - t_speed0

        maxF = float(np.max(np.abs(F_ext)))
        if not np.isfinite(maxF) or maxF <= 0.0:
            raise RuntimeError("Non-positive max speed; cannot advance level set.")

        dt_max = float(min(grid.dx, grid.dy) / maxF)
        dt = max(float(args.dt_min), 0.8 * dt_max)
        # Align dt to hit the next requested snapshot time exactly (paper lists).
        if next_snap < int(t_targets.size):
            t_next = float(t_targets[next_snap])
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
        y_top_new = _biofilm_top_y(phi)
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
        if next_snap < int(t_targets.size) and abs(float(t_targets[next_snap]) - t) <= 1.0e-10:
            phi_snaps.append((float(t), np.asarray(phi, float).copy()))
            next_snap += 1
        pe = max(1, int(getattr(args, "progress_every", 10)))
        if step == 1 or step % pe == 0:
            print(
                f"[step {step:04d}] t={t:.3f} d  dt={dt:.3e} d  maxF={maxF:.4g} mm/d  cut_elems={mesh.element_bitset('cut').cardinality()}"
            )
        if bool(getattr(args, "timings", False)):
            t_step = time.perf_counter() - t_step0
            print(
                f"[timing] step={step:04d} total={t_step:.2f}s  ls={t_ls:.2f}s  classify={t_class:.2f}s  "
                f"newton={t_newton:.2f}s({newton_iters:d} it)  phi={t_phi:.2f}s  speed={t_speed:.2f}s  update={t_upd:.2f}s"
            )

    # Ensure final snapshot present
    if (not phi_snaps) or abs(float(phi_snaps[-1][0]) - float(t)) > 1.0e-8:
        phi_snaps.append((float(t), np.asarray(phi, float).copy()))

    # --- outputs ------------------------------------------------------------
    _plot_level_set_contours(phi_snaps, grid_x=grid.x, grid_y=grid.y, outpng=outdir / "fig6a_interface.png")

    # Plot final fields (S and Phi) from the last solve
    dh_base = dh.base
    _plot_field_on_mesh(dh_base, np.asarray(S_base, float), field="S", title="Example 2: S at final time", outpng=outdir / "fig6b_S.png")
    _plot_field_on_mesh(dh_base, np.asarray(Phi_base, float), field="Phi", title="Example 2: Phi at final time", outpng=outdir / "fig6c_Phi.png")

    _write_csv(outdir / "y_top_timeseries.csv", rows)

    summary = {
        "t_final_days": float(t),
        "steps": int(step),
        "mesh_nx": int(args.mesh_nx),
        "mesh_ny": int(args.mesh_ny),
        "grid_nx": int(args.grid_nx),
        "grid_ny": int(args.grid_ny),
        "centers_x_mm": centers_x,
        "radii_mm": radii,
        "Y_wO": float(args.Y_wO),
        "substrate_bc": str(args.substrate_bc),
        "Ls_mm": float(args.Ls),
        "S_penalty": float(args.S_penalty),
        "Sbar": float(args.Sbar),
        "y_top_final_mm": float(_biofilm_top_y(phi)),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"- Wrote {outdir/'summary.json'}")
    print(f"- Wrote {outdir/'fig6a_interface.png'}")
    print(f"- Wrote {outdir/'fig6b_S.png'}")
    print(f"- Wrote {outdir/'fig6c_Phi.png'}")
    print(f"- Wrote {outdir/'y_top_timeseries.csv'}")


if __name__ == "__main__":
    main()
