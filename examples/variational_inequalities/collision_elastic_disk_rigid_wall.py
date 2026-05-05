from __future__ import annotations

from dataclasses import dataclass
import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.core.topology import Node
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.ufl.expressions import (
    Constant,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    div,
    dot,
    grad,
    inner,
)
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace

from .pdas import PDASOptions, solve_obstacle_pdas, solve_obstacle_semismooth_newton


def _disk_points(*, radius: float, center: tuple[float, float], h: float, n_boundary: int) -> np.ndarray:
    cx, cy = float(center[0]), float(center[1])
    r = float(radius)
    h = float(h)
    if h <= 0.0:
        raise ValueError("h must be > 0.")
    if r <= 0.0:
        raise ValueError("radius must be > 0.")
    if n_boundary < 16:
        raise ValueError("n_boundary must be >= 16.")

    # Boundary points (exact circle) + interior grid points.
    ang = np.linspace(0.0, 2.0 * np.pi, int(n_boundary), endpoint=False, dtype=float)
    boundary = np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)])

    x = np.arange(cx - r, cx + r + 0.5 * h, h, dtype=float)
    y = np.arange(cy - r, cy + r + 0.5 * h, h, dtype=float)
    X, Y = np.meshgrid(x, y, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    m = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2 <= r * r
    inside = pts[m]

    all_pts = np.vstack([inside, boundary])
    # Deduplicate by rounding to avoid near-duplicates from boundary+grid intersections.
    all_pts = np.unique(np.round(all_pts, decimals=12), axis=0)
    return all_pts


def make_disk_mesh(
    *,
    radius: float = 0.5,
    center: tuple[float, float] = (0.0, 0.55),
    h: float = 0.1,
    n_boundary: int = 96,
) -> Mesh:
    """
    Create a simple P1 triangular mesh of a disk using Delaunay + filtering.

    This lives in `examples/` because it's a benchmark utility, not a core feature.
    """
    try:
        from scipy.spatial import Delaunay  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("SciPy is required for this disk-mesh helper (scipy.spatial.Delaunay).") from exc

    pts = _disk_points(radius=radius, center=center, h=h, n_boundary=n_boundary)
    tri = Delaunay(pts)
    elems = np.asarray(tri.simplices, dtype=int).copy()

    cx, cy = float(center[0]), float(center[1])
    r = float(radius)
    cent = pts[elems].mean(axis=1)
    keep = (cent[:, 0] - cx) ** 2 + (cent[:, 1] - cy) ** 2 <= r * r
    elems = elems[keep]

    def signed_area(a, b, c) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    for t in elems:
        a, b, c = pts[t[0]], pts[t[1]], pts[t[2]]
        if signed_area(a, b, c) < 0.0:
            t[1], t[2] = t[2], t[1]

    nodes = [Node(id=i, x=float(pts[i, 0]), y=float(pts[i, 1])) for i in range(int(pts.shape[0]))]
    return Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=elems,
        element_type="tri",
        poly_order=1,
    )


@dataclass(frozen=True)
class DiskCollisionResult:
    mesh: Mesh
    dof_handler: DofHandler
    center: tuple[float, float]
    radius: float
    wall_y: float
    boundary_uy: np.ndarray  # uy DOFs on the outer boundary (potential contact)
    u_hist: np.ndarray  # shape (n_steps+1, n_dofs)
    lam_hist: np.ndarray  # shape (n_steps, n_dofs)
    active_hist: np.ndarray  # shape (n_steps, n_dofs), bool
    n_pdas_iter: np.ndarray  # shape (n_steps,), int
    contact_half_width: np.ndarray  # shape (n_steps,), float (nan if no contact)
    center_y: np.ndarray  # shape (n_steps+1,), float
    n_active: np.ndarray  # shape (n_steps,), int


def run_elastic_disk_collision_with_wall(
    *,
    radius: float = 0.5,
    gap: float = 0.05,
    h: float = 0.1,
    n_boundary: int = 96,
    E: float = 1.0e4,
    nu: float = 0.3,
    rho: float = 1.0,
    v0: float = -1.0,
    dt: float = 0.01,
    n_steps: int = 20,
    wall_y: float = 0.0,
    method: str = "pdas",
    backend: str = "python",
    pdas_opts: Optional[PDASOptions] = None,
    output_dir: Optional[str | Path] = None,
    output_stride: int = 5,
) -> DiskCollisionResult:
    """
    Dynamic collision benchmark: a free elastic disk impacts a rigid wall y=wall_y.

    Contact is enforced via a nodal (boundary-only) inequality:
        y_ref + u_y >= wall_y
    solved by PDAS / semismooth Newton on the Newmark effective system.
    """
    if dt <= 0.0:
        raise ValueError("dt must be > 0.")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")
    if radius <= 0.0:
        raise ValueError("radius must be > 0.")
    if gap < 0.0:
        raise ValueError("gap must be >= 0.")
    if output_stride < 1:
        raise ValueError("output_stride must be >= 1.")

    center = (0.0, float(wall_y) + float(radius) + float(gap))
    mesh = make_disk_mesh(radius=radius, center=center, h=h, n_boundary=n_boundary)

    me = MixedElement(mesh, field_specs={"ux": 1, "uy": 1})
    dh = DofHandler(me, method="cg")
    dh._ensure_node_maps()
    dh._ensure_dof_coords()

    # --- Linear elasticity (small strain, plane stress) ---
    V = FunctionSpace("displacement", ["ux", "uy"])
    u = VectorTrialFunction(V, dof_handler=dh)
    v = VectorTestFunction(V, dof_handler=dh)

    def eps(w):
        return 0.5 * (grad(w) + grad(w).T)

    lmbda = Constant(float(E) * float(nu) / (1.0 - float(nu) ** 2))
    mu = Constant(float(E) / (2.0 * (1.0 + float(nu))))
    rho_c = Constant(float(rho))

    aK = (lmbda * (div(u) * div(v)) + Constant(2.0) * mu * inner(eps(u), eps(v))) * dx(metadata={"q": 4})
    aM = (rho_c * dot(u, v)) * dx(metadata={"q": 4})

    K, _ = assemble_form(Equation(aK, None), dof_handler=dh, bcs=[], backend=backend)
    M, _ = assemble_form(Equation(aM, None), dof_handler=dh, bcs=[], backend=backend)

    # Ensure SciPy sparse matrices (PDAS sparse path expects scipy.sparse).
    import scipy.sparse as sp  # type: ignore

    if not sp.isspmatrix(K):
        K = sp.csr_matrix(K)
    else:
        K = K.tocsr()
    if not sp.isspmatrix(M):
        M = sp.csr_matrix(M)
    else:
        M = M.tocsr()

    # --- Contact constraint (boundary uy DOFs only) ---
    cx, cy = float(center[0]), float(center[1])
    r = float(radius)
    tol = max(5.0e-3, 0.75 * float(h))

    uy_dofs = np.array(sorted(dh.get_field_slice("uy")), dtype=int)
    uy_xy = dh._dof_coords[uy_dofs]
    uy_r = np.sqrt((uy_xy[:, 0] - cx) ** 2 + (uy_xy[:, 1] - cy) ** 2)
    boundary_uy = uy_dofs[np.abs(uy_r - r) <= tol]

    constrained = np.zeros(dh.total_dofs, dtype=bool)
    constrained[boundary_uy] = True

    psi = np.zeros(dh.total_dofs, dtype=float)
    psi[boundary_uy] = float(wall_y) - dh._dof_coords[boundary_uy, 1]

    # --- Newmark (average acceleration) ---
    beta = 0.25
    gamma = 0.5
    alpha = 1.0 / (beta * dt * dt)
    A_eff = (K + alpha * M).tocsr()

    # Initial conditions: rigid translation velocity.
    nd = int(dh.total_dofs)
    u_n = np.zeros(nd, dtype=float)
    v_n = np.zeros(nd, dtype=float)
    a_n = np.zeros(nd, dtype=float)
    v_n[dh.get_field_slice("uy")] = float(v0)

    lam_prev = np.zeros(nd, dtype=float)
    pdas_opts = pdas_opts or PDASOptions()

    out_dir = Path(output_dir) if output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Pick a "center" node for reporting y_center.
    node_xy = np.asarray(mesh.nodes_x_y_pos, dtype=float)
    center_nid = int(np.argmin((node_xy[:, 0] - cx) ** 2 + (node_xy[:, 1] - cy) ** 2))
    center_uy_dof = int(dh.dof_map["uy"][center_nid])

    u_hist = np.zeros((n_steps + 1, nd), dtype=float)
    lam_hist = np.zeros((n_steps, nd), dtype=float)
    active_hist = np.zeros((n_steps, nd), dtype=bool)
    n_pdas_iter = np.zeros((n_steps,), dtype=int)
    contact_half_width = np.full((n_steps,), np.nan, dtype=float)
    center_y = np.zeros((n_steps + 1,), dtype=float)
    n_active = np.zeros((n_steps,), dtype=int)

    u_hist[0] = u_n
    center_y[0] = float(node_xy[center_nid, 1] + u_n[center_uy_dof])

    method_key = str(method).strip().lower()

    use_examples = method_key in {
        "pdas",
        "examples-pdas",
        "primal-dual-active-set",
        "ssn",
        "examples-ssn",
        "semismooth",
        "semismooth-newton",
        "semi-smooth",
    }
    use_internal = method_key in {"internal-pdas", "internal", "pycutfem-pdas", "pdas-internal"}
    use_snesvi = method_key in {"snesvi", "petsc-snesvi", "snes"}

    if not (use_examples or use_internal or use_snesvi):
        raise ValueError(
            "method must be one of: "
            "'pdas' (examples), 'ssn' (examples), 'internal-pdas' (pycutfem), 'snesvi' (PETSc SNESVI)."
        )

    # Internal (pycutfem) VI solver setup: build an effective Newmark residual form once
    # and update the predictor field u_star each step.
    if use_internal or use_snesvi:
        from pycutfem.solvers.nonlinear_solver import (
            HAS_PETSC,
            LinearSolverParameters,
            NewtonParameters,
            PdasNewtonSolver,
            PetscSnesNewtonSolver,
            VIParameters,
        )

        # Unknown + snapshots (single mixed displacement field).
        u_fun = VectorFunction("u", ["ux", "uy"], dh)
        u_prev_fun = VectorFunction("u_prev", ["ux", "uy"], dh)
        u_star_fun = VectorFunction("u_star", ["ux", "uy"], dh)

        # Residual uses u_fun; Jacobian uses a trial increment du.
        du = VectorTrialFunction(V, dof_handler=dh)
        w = v  # test function already created above

        alpha_c = Constant(float(alpha))

        def eps(w_):
            return 0.5 * (grad(w_) + grad(w_).T)

        # Effective Newmark system: (K + alpha M) u = alpha M u_star
        residual_form_eff = (
            lmbda * (div(u_fun) * div(w))
            + Constant(2.0) * mu * inner(eps(u_fun), eps(w))
            + alpha_c * rho_c * (dot(u_fun, w) - dot(u_star_fun, w))
        ) * dx(metadata={"q": 4})

        jacobian_form_eff = (
            lmbda * (div(du) * div(w))
            + Constant(2.0) * mu * inner(eps(du), eps(w))
            + alpha_c * rho_c * dot(du, w)
        ) * dx(metadata={"q": 4})

        # Bounds: uy >= psi on potential contact nodes; all other DOFs unbounded.
        lo_full = np.full(nd, -np.inf, dtype=float)
        hi_full = np.full(nd, np.inf, dtype=float)
        lo_full[boundary_uy] = psi[boundary_uy]

        if use_internal:
            # PDAS scaling parameter: use a representative diagonal stiffness on the constrained DOFs.
            try:
                diag = np.asarray(A_eff.diagonal(), dtype=float)[boundary_uy]
                diag = diag[np.isfinite(diag) & (diag > 0.0)]
                c_vi = float(np.median(diag)) if diag.size else 1.0
            except Exception:
                c_vi = 1.0
            solver_vi = PdasNewtonSolver(
                residual_form_eff,
                jacobian_form_eff,
                dof_handler=dh,
                mixed_element=me,
                bcs=[],
                bcs_homog=[],
                vi_params=VIParameters(c=c_vi),
                newton_params=NewtonParameters(
                    newton_tol=1.0e-10,
                    max_newton_iter=50,
                    line_search=False,
                ),
                lin_params=LinearSolverParameters(backend=("petsc" if HAS_PETSC else "scipy")),
                quad_order=4,
                backend=backend,
            )
            solver_vi.set_box_bounds(lower=lo_full, upper=hi_full)
        else:
            petsc_opts = {
                "snes_type": "vinewtonrsls",
                "snes_linesearch_type": "bt",
                "snes_atol": 1.0e-10,
                "snes_rtol": 0.0,
                "snes_max_it": 50,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
            solver_vi = PetscSnesNewtonSolver(
                residual_form_eff,
                jacobian_form_eff,
                dof_handler=dh,
                mixed_element=me,
                bcs=[],
                bcs_homog=[],
                newton_params=NewtonParameters(
                    newton_tol=1.0e-10,
                    max_newton_iter=50,
                    line_search=True,
                    ls_mode="dealii",
                ),
                quad_order=4,
                backend=backend,
                petsc_options=petsc_opts,
            )
            solver_vi.set_box_bounds(lower=lo_full, upper=hi_full)

    def _solve(A, rhs, *, y0, lam0):
        if use_examples:
            if method_key in {"pdas", "examples-pdas", "primal-dual-active-set"}:
                return solve_obstacle_pdas(A, rhs, psi, y0=y0, lam0=lam0, constrained=constrained, opts=pdas_opts)
            return solve_obstacle_semismooth_newton(
                A, rhs, psi, y0=y0, lam0=lam0, constrained=constrained, opts=pdas_opts
            )

        # Internal (pycutfem) VI solvers: update coefficients and solve in-place.
        u_prev_fun.nodal_values = u_n[u_prev_fun._g_dofs]
        u_star_fun.nodal_values = y0[u_star_fun._g_dofs]
        u_fun.nodal_values = y0[u_fun._g_dofs]

        delta, converged, n_iters = solver_vi._newton_loop(
            funcs=[u_fun],
            prev_funcs=[u_prev_fun],
            aux_funcs={"u_star": u_star_fun},
            bcs_now=[],
        )
        if not converged:
            raise RuntimeError(f"VI solve did not converge (iters={int(n_iters)}).")

        y = np.zeros_like(y0)
        y[u_fun._g_dofs] = u_fun.nodal_values
        # Recover the dual variable: λ = A y - rhs (clipped to enforce λ>=0 on the constrained set).
        r = A @ y - rhs
        active = constrained & (y <= psi + 1.0e-10)
        lam = np.zeros_like(y0)
        lam[active] = np.maximum(r[active], 0.0)
        from .pdas import PDASResult  # lightweight container for downstream bookkeeping

        return PDASResult(
            y=y,
            lam=lam,
            active=active,
            n_iter=int(n_iters),
            converged=True,
            history={"n_active": [int(np.count_nonzero(active))]},
        )

    # --- Time loop ---
    for step in range(n_steps):
        u_star = u_n + dt * v_n + (0.5 - beta) * (dt * dt) * a_n
        v_star = v_n + (1.0 - gamma) * dt * a_n

        rhs = alpha * (M @ u_star)

        sol = _solve(A_eff, rhs, y0=u_star, lam0=lam_prev)
        u_np1 = sol.y
        lam_np1 = sol.lam

        # Newmark updates
        a_np1 = alpha * (u_np1 - u_star)
        v_np1 = v_star + gamma * dt * a_np1

        # Contact metrics (boundary nodes only).
        active = sol.active & constrained
        active_hist[step] = active
        lam_hist[step] = lam_np1
        n_pdas_iter[step] = int(sol.n_iter)
        n_active[step] = int(np.count_nonzero(active))

        if np.any(active):
            active_dofs = np.flatnonzero(active)
            active_nids = [dh._dof_to_node_map[int(gd)][1] for gd in active_dofs]
            active_nids = np.asarray([nid for nid in active_nids if nid is not None], dtype=int)
            x_act = node_xy[active_nids, 0] + u_np1[[int(dh.dof_map["ux"][int(nid)]) for nid in active_nids]]
            x0 = float(cx)  # expected symmetry axis
            contact_half_width[step] = float(np.max(np.abs(x_act - x0)))

        # VTK output (optional, for visual contour inspection).
        if out_dir is not None and (step % int(output_stride) == 0 or step == n_steps - 1):
            disp = VectorFunction("u", ["ux", "uy"], dh)
            disp.nodal_values = u_np1[disp._g_dofs]
            active_nodes = np.zeros(len(mesh.nodes_list), dtype=float)
            for gd in np.flatnonzero(active):
                _field, nid = dh._dof_to_node_map[int(gd)]
                if nid is not None:
                    active_nodes[int(nid)] = 1.0
            export_vtk(
                str(out_dir / f"disk_collision_step_{step:04d}.vtu"),
                mesh,
                dh,
                {
                    "u": disp,
                    "contact_active": active_nodes,
                },
            )

        u_hist[step + 1] = u_np1
        center_y[step + 1] = float(node_xy[center_nid, 1] + u_np1[center_uy_dof])

        u_n, v_n, a_n = u_np1, v_np1, a_np1
        lam_prev = lam_np1

    return DiskCollisionResult(
        mesh=mesh,
        dof_handler=dh,
        center=(float(center[0]), float(center[1])),
        radius=float(radius),
        wall_y=float(wall_y),
        boundary_uy=np.asarray(boundary_uy, dtype=int),
        u_hist=u_hist,
        lam_hist=lam_hist,
        active_hist=active_hist,
        n_pdas_iter=n_pdas_iter,
        contact_half_width=contact_half_width,
        center_y=center_y,
        n_active=n_active,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Dynamic elastic disk vs rigid wall contact benchmark (VI/PDAS/SSN).")
    ap.add_argument(
        "--method",
        type=str,
        default="internal-pdas",
        choices=["pdas", "ssn", "internal-pdas", "snesvi"],
        help="VI solver: examples PDAS/SSN, internal PDAS, or PETSc SNESVI.",
    )
    ap.add_argument("--backend", type=str, default="jit", choices=["python", "jit", "cpp"])
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--gap", type=float, default=0.05)
    ap.add_argument("--h", type=float, default=0.1)
    ap.add_argument("--n-boundary", type=int, default=96)
    ap.add_argument("--E", type=float, default=1.0e4)
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--v0", type=float, default=-1.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--n-steps", type=int, default=30)
    ap.add_argument("--wall-y", type=float, default=0.0)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--output-stride", type=int, default=5)
    args = ap.parse_args()

    res = run_elastic_disk_collision_with_wall(
        radius=float(args.radius),
        gap=float(args.gap),
        h=float(args.h),
        n_boundary=int(args.n_boundary),
        E=float(args.E),
        nu=float(args.nu),
        rho=float(args.rho),
        v0=float(args.v0),
        dt=float(args.dt),
        n_steps=int(args.n_steps),
        wall_y=float(args.wall_y),
        method=str(args.method),
        backend=str(args.backend),
        output_dir=(Path(args.output_dir) if args.output_dir else None),
        output_stride=int(args.output_stride),
    )
    print("max n_active:", int(np.max(res.n_active)))
    print("max contact half-width:", float(np.nanmax(res.contact_half_width)))
    print("min center_y:", float(np.min(res.center_y)))


if __name__ == "__main__":
    main()
