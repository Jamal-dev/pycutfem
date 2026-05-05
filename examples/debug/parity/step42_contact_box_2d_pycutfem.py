from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp  # type: ignore
import scipy.sparse.linalg as spla  # type: ignore

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.ufl.expressions import Constant, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction, div, grad, inner
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dS, dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def _obstacle_gap_lower_bound(
    x: np.ndarray | float,
    *,
    y_ref: np.ndarray | float,
    y_surface: float = 1.0,
) -> np.ndarray:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y_ref, dtype=float)
    rad2 = 0.36 - (xx - 0.5) ** 2
    out = np.full_like(xx, -1.0e30, dtype=float)
    mask = rad2 > 0.0
    out[mask] = -np.sqrt(rad2[mask]) + float(y_surface) + 0.59 - yy[mask]
    return out


def _as_csr(A) -> sp.csr_matrix:
    if sp.isspmatrix(A):
        return A.tocsr()
    return sp.csr_matrix(np.asarray(A, dtype=float))


def _solve_with_fixed_values(
    A: sp.csr_matrix,
    f: np.ndarray,
    *,
    fixed_mask: np.ndarray,
    fixed_values: np.ndarray,
) -> np.ndarray:
    n = int(f.shape[0])
    free = np.flatnonzero(~fixed_mask)
    fixed = np.flatnonzero(fixed_mask)

    u = np.asarray(fixed_values, dtype=float).copy()
    if free.size == 0:
        return u

    rhs = np.asarray(f[free], dtype=float)
    if fixed.size:
        rhs = rhs - A[free][:, fixed] @ u[fixed]
    A_ff = A[free][:, free].tocsr()
    u[free] = spla.spsolve(A_ff, rhs)
    return u


@dataclass(frozen=True)
class Step42BoxContactResult:
    mesh: Mesh
    dof_handler: DofHandler
    displacement: np.ndarray
    lambda_vec: np.ndarray
    active: np.ndarray
    diag_mass: np.ndarray
    iterations: int
    converged: bool
    top_uy: np.ndarray
    psi: np.ndarray


def solve_step42_contact_box_2d(
    *,
    degree: int = 1,
    refinements: int = 3,
    E: float = 200000.0,
    nu: float = 0.3,
    c_scale: float = 100.0,
    max_iter: int = 80,
    active_tol: float = 0.0,
    backend: str = "cpp",
    q: int | None = None,
) -> Step42BoxContactResult:
    if degree < 1:
        raise ValueError("degree must be >= 1")
    if refinements < 0:
        raise ValueError("refinements must be >= 0")

    nx = 2 ** int(refinements)
    ny = 2 ** int(refinements)
    qdeg = int(q) if q is not None else max(int(degree) + 2, 4)

    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=int(degree))
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=int(degree),
    )
    tol = 1.0e-12
    mesh.tag_boundary_edges(
        {
            "top": lambda x, y: abs(y - 1.0) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "sides": lambda x, y: abs(x - 0.0) <= tol or abs(x - 1.0) <= tol,
        }
    )

    me = MixedElement(mesh, field_specs={"ux": int(degree), "uy": int(degree)})
    dh = DofHandler(me, method="cg")
    dh._ensure_node_maps()
    dh._ensure_dof_coords()

    V = FunctionSpace("U", ["ux", "uy"])
    u = VectorTrialFunction(V, dof_handler=dh)
    v = VectorTestFunction(V, dof_handler=dh)
    uy = TrialFunction(name="trial_uy", field_name="uy", dof_handler=dh)
    wy = TestFunction(name="test_uy", field_name="uy", dof_handler=dh)

    def eps(w):
        return 0.5 * (grad(w) + grad(w).T)

    lam = Constant(float(E) * float(nu) / ((1.0 + float(nu)) * (1.0 - 2.0 * float(nu))))
    mu = Constant(float(E) / (2.0 * (1.0 + float(nu))))

    aK = (lam * (div(u) * div(v)) + Constant(2.0) * mu * inner(eps(u), eps(v))) * dx(metadata={"q": qdeg})
    aM = (uy * wy) * dS(mesh.edge_bitset("top"), metadata={"q": qdeg})

    K_full, _ = assemble_form(Equation(aK, None), dof_handler=dh, bcs=[], backend=backend)
    M_top, _ = assemble_form(Equation(aM, None), dof_handler=dh, bcs=[], backend=backend)

    K_full = _as_csr(K_full)
    M_top = _as_csr(M_top)
    diag_mass = np.asarray(M_top.sum(axis=1)).reshape((-1,))

    bcs = [
        BoundaryCondition("ux", "dirichlet", "bottom", lambda x, y: 0.0),
        BoundaryCondition("uy", "dirichlet", "bottom", lambda x, y: 0.0),
        BoundaryCondition("ux", "dirichlet", "sides", lambda x, y: 0.0),
    ]
    dirichlet = dh.get_dirichlet_data(bcs)

    nd = int(dh.total_dofs)
    rhs = np.zeros(nd, dtype=float)
    fixed_mask_dirichlet = np.zeros(nd, dtype=bool)
    fixed_values_dirichlet = np.zeros(nd, dtype=float)
    for gdof, value in dirichlet.items():
        fixed_mask_dirichlet[int(gdof)] = True
        fixed_values_dirichlet[int(gdof)] = float(value)

    top_uy_bc = BoundaryCondition("uy", "dirichlet", "top", lambda x, y: 0.0)
    top_uy = np.array(sorted(dh.get_dirichlet_data([top_uy_bc]).keys()), dtype=int)
    psi = np.full(nd, -1.0e30, dtype=float)
    x_top = dh._dof_coords[top_uy, 0]
    y_top = dh._dof_coords[top_uy, 1]
    psi[top_uy] = _obstacle_gap_lower_bound(x_top, y_ref=y_top, y_surface=1.0)

    c = float(c_scale) * float(E)
    u_k = fixed_values_dirichlet.copy()
    lam_k = np.zeros(nd, dtype=float)
    active = np.zeros(nd, dtype=bool)
    active[top_uy] = (lam_k[top_uy] + c * (u_k[top_uy] - psi[top_uy])) > float(active_tol)

    converged = False
    for it in range(1, int(max_iter) + 1):
        fixed_mask = fixed_mask_dirichlet.copy()
        fixed_values = fixed_values_dirichlet.copy()
        active_top = active[top_uy]
        fixed_mask[top_uy[active_top]] = True
        fixed_values[top_uy[active_top]] = psi[top_uy[active_top]]

        u_new = _solve_with_fixed_values(K_full, rhs, fixed_mask=fixed_mask, fixed_values=fixed_values)
        Ku = K_full @ u_new

        lam_new = np.zeros(nd, dtype=float)
        lam_new[top_uy[active_top]] = rhs[top_uy[active_top]] - Ku[top_uy[active_top]]

        indicator = lam_new[top_uy] + c * (u_new[top_uy] - psi[top_uy])
        active_new = np.zeros_like(active)
        active_new[top_uy] = indicator > float(active_tol)

        u_k = u_new
        lam_k = lam_new

        if np.array_equal(active_new, active):
            converged = True
            active = active_new
            break
        active = active_new

    return Step42BoxContactResult(
        mesh=mesh,
        dof_handler=dh,
        displacement=u_k,
        lambda_vec=lam_k,
        active=active,
        diag_mass=diag_mass,
        iterations=int(it),
        converged=bool(converged),
        top_uy=top_uy,
        psi=psi,
    )


def _make_vtk_fields(res: Step42BoxContactResult) -> tuple[VectorFunction, np.ndarray, np.ndarray]:
    dh = res.dof_handler
    disp = VectorFunction(name="displacement", field_names=["ux", "uy"], dof_handler=dh)
    disp.nodal_values[:] = res.displacement[disp._g_dofs]

    n_nodes = len(res.mesh.nodes_list)
    contact = np.zeros((n_nodes, 2), dtype=float)
    active = np.zeros((n_nodes, 2), dtype=float)

    for gdof in res.top_uy:
        node_id = dh._dof_to_node_map[int(gdof)][1]
        if node_id is None:
            continue
        node_id = int(node_id)
        if bool(res.active[int(gdof)]):
            active[node_id, 1] = 1.0
        mii = float(res.diag_mass[int(gdof)])
        if mii > 0.0:
            contact[node_id, 1] = float(res.lambda_vec[int(gdof)] / mii)

    return disp, contact, active


def main() -> None:
    ap = argparse.ArgumentParser(description="2D reduced step-42-style box contact reference in pycutfem.")
    ap.add_argument("--degree", type=int, default=1)
    ap.add_argument("--refinements", type=int, default=3)
    ap.add_argument("--backend", choices=("python", "jit", "cpp"), default="cpp")
    ap.add_argument("--q", type=int, default=None)
    ap.add_argument("--E", type=float, default=200000.0)
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--c-scale", type=float, default=100.0)
    ap.add_argument("--max-it", type=int, default=80)
    ap.add_argument("--active-tol", type=float, default=0.0)
    ap.add_argument("--out-dir", type=Path, default=Path("out/step42_contact_box_2d_pycutfem"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = solve_step42_contact_box_2d(
        degree=int(args.degree),
        refinements=int(args.refinements),
        E=float(args.E),
        nu=float(args.nu),
        c_scale=float(args.c_scale),
        max_iter=int(args.max_it),
        active_tol=float(args.active_tol),
        backend=str(args.backend),
        q=args.q,
    )

    disp, contact, active = _make_vtk_fields(res)
    vtk_path = out_dir / "solution.vtu"
    export_vtk(
        str(vtk_path),
        mesh=res.mesh,
        dof_handler=res.dof_handler,
        functions={
            "displacement": disp,
            "contact_force": contact,
            "active_set": active,
        },
    )

    top_mask = res.psi[res.top_uy] > -1.0e20
    top_gap = res.psi[res.top_uy[top_mask]] - res.displacement[res.top_uy[top_mask]]
    summary = {
        "degree": int(args.degree),
        "refinements": int(args.refinements),
        "backend": str(args.backend),
        "iterations": int(res.iterations),
        "converged": bool(res.converged),
        "n_active": int(np.count_nonzero(res.active[res.top_uy])),
        "contact_force_total_lumped": float(np.sum(res.lambda_vec[res.top_uy])),
        "uy_min_top": float(np.min(res.displacement[res.top_uy])) if res.top_uy.size else 0.0,
        "gap_min_top": float(np.min(top_gap)) if top_gap.size else 0.0,
        "gap_max_top": float(np.max(top_gap)) if top_gap.size else 0.0,
        "vtk": str(vtk_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
