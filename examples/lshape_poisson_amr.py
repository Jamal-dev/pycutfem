#!/usr/bin/env python
"""
Adaptive Poisson solve on an L-shaped domain with Kelly error indicators.

- Domain: square [-1,1]² with the lower-right quadrant (x>0, y<0) removed.
- Exact (polar) solution: u = r^{2/3} * sin(2θ/3)
  (singular at the re-entrant corner), f = 0, Dirichlet BC from exact.
- FE: Q2 continuous.
- Error indicator: Kelly flux jump on interior edges.
- Refinement: mark top fraction of cells by indicator, split them anisotropically
  (vertical on x<0, horizontal otherwise) to produce hanging nodes. Hanging-node
  constraints are built by DofHandler and condensed (Eᵀ K E, Eᵀ F).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import scipy.sparse.linalg as spla

from pycutfem.core.topology import Node
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem import transform
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.integration.quadrature import volume as vol_rule
from pycutfem.io.visualization import plot_mesh_2
from pycutfem.utils.gmsh_loader import mesh_from_gmsh
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    grad,
    inner,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad

# -----------------------------------------------------------------------------
# Exact solution and helpers
# -----------------------------------------------------------------------------


def exact_u(x: float, y: float) -> float:
    r = math.hypot(x, y)
    theta = math.atan2(y, x)
    return (r ** (2.0 / 3.0)) * math.sin((2.0 / 3.0) * theta) if r > 1e-14 else 0.0


def grad_exact_u(x: float, y: float) -> Tuple[float, float]:
    r = math.hypot(x, y)
    if r < 1e-14:
        return (0.0, 0.0)
    theta = math.atan2(y, x)
    pref = (2.0 / 3.0) * (r ** (-1.0 / 3.0))
    dtheta_dx = -y / (r * r)
    dtheta_dy = x / (r * r)
    s = math.sin((2.0 / 3.0) * theta)
    c = math.cos((2.0 / 3.0) * theta)
    du_dr = pref * s
    du_dtheta = (r ** (2.0 / 3.0)) * (2.0 / 3.0) * c
    du_dx = du_dr * (x / r) + du_dtheta * dtheta_dx
    du_dy = du_dr * (y / r) + du_dtheta * dtheta_dy
    return (du_dx, du_dy)


def tag_lshape_boundaries(mesh: Mesh, tol: float = 1e-12) -> None:
    xmin, ymin = mesh.nodes_x_y_pos.min(axis=0)
    xmax, ymax = mesh.nodes_x_y_pos.max(axis=0)
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - xmin) <= tol,
            "right": lambda x, y: abs(x - xmax) <= tol,
            "bottom": lambda x, y: (abs(y - ymin) <= tol) or (x > 0 and abs(y) <= tol),
            "top": lambda x, y: abs(y - ymax) <= tol,
            "reentrant": lambda x, y: (x >= 0) and (abs(y) <= tol) and (y < 0 + tol),
        }
    )


# -----------------------------------------------------------------------------
# Refinement utilities (1-to-2 splits)
# -----------------------------------------------------------------------------


def _quad_corner_indices(p: int) -> tuple[int, int, int, int]:
    n = p + 1
    bl = 0
    br = p
    tr = p * n + p
    tl = p * n
    return bl, br, tr, tl


def _refine_element_quads(mesh: Mesh, eid: int, nodes, node_lookup):
    """
    Split one quad element into 4 children (2x2). Returns (child_connectivity, child_corners).
    """
    p = mesh.poly_order
    t = np.linspace(-1.0, 1.0, p + 1)
    parent_conn = mesh.elements_connectivity[eid]

    def _parent_node(xi_p: float, eta_p: float) -> int | None:
        ix = np.where(np.isclose(t, xi_p, atol=1e-12))[0]
        iy = np.where(np.isclose(t, eta_p, atol=1e-12))[0]
        if ix.size and iy.size:
            idx = int(iy[0] * (p + 1) + ix[0])
            return int(parent_conn[idx])
        return None

    def _get_node(xi_p: float, eta_p: float) -> int:
        nid = _parent_node(xi_p, eta_p)
        if nid is not None:
            return nid
        x_phys = transform.x_mapping(mesh, eid, (float(xi_p), float(eta_p)))
        key = (float(round(x_phys[0], 14)), float(round(x_phys[1], 14)))
        nid = node_lookup.get(key)
        if nid is not None:
            return nid
        nid = len(nodes)
        node_lookup[key] = nid
        nodes.append(Node(nid, float(x_phys[0]), float(x_phys[1])))
        return nid

    def _child(offset_x: float, offset_y: float):
        conn = []
        xi_child = t
        eta_child = t
        for eta in eta_child:
            for xi in xi_child:
                xi_p = 0.5 * xi + offset_x
                eta_p = 0.5 * eta + offset_y
                conn.append(_get_node(xi_p, eta_p))
        bl, br, tr, tl = _quad_corner_indices(p)
        corners = [conn[bl], conn[br], conn[tr], conn[tl]]
        return conn, corners

    children = []
    child_corners = []
    for ox, oy in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]:
        c, cc = _child(ox, oy)
        children.append(c)
        child_corners.append(cc)
    return children, child_corners


def refine_marked(mesh: Mesh, marked: set[int]) -> Mesh:
    nodes = list(mesh.nodes_list)
    node_lookup = {(round(nd.x, 14), round(nd.y, 14)): int(nd.id) for nd in nodes}
    new_elems: List[List[int]] = []
    new_corners: List[List[int]] = []
    for eid, elem in enumerate(mesh.elements_list):
        if eid not in marked:
            new_elems.append(list(mesh.elements_connectivity[eid]))
            new_corners.append(list(mesh.corner_connectivity[eid]))
            continue
        conns, corners = _refine_element_quads(mesh, eid, nodes, node_lookup)
        new_elems.extend(conns)
        new_corners.extend(corners)
    new_mesh = Mesh(
        nodes=nodes,
        element_connectivity=np.asarray(new_elems, dtype=int),
        elements_corner_nodes=np.asarray(new_corners, dtype=int),
        element_type="quad",
        poly_order=mesh.poly_order,
    )
    tag_lshape_boundaries(new_mesh)
    return new_mesh


# -----------------------------------------------------------------------------
# Kelly indicator (flux jump)
# -----------------------------------------------------------------------------


def _grad_on_element(dh: DofHandler, u: Function, eid: int, xi: float, eta: float) -> np.ndarray:
    me = dh.mixed_element
    J = transform.jacobian(dh.mixed_element.mesh, eid, (float(xi), float(eta)))
    Jinv = np.linalg.inv(J)
    gd = dh.element_dofs("u", eid)
    coeffs = u.get_nodal_values(gd)
    G_ref_full = me.grad_basis("u", float(xi), float(eta))
    # Slice to field dofs
    sl = me.slice("u")
    G_ref = G_ref_full[sl, :]
    grad_ref = coeffs @ G_ref
    return np.asarray(grad_ref @ Jinv, float)


def kelly_indicators(mesh: Mesh, dh: DofHandler, u: Function) -> np.ndarray:
    indicators = np.zeros(len(mesh.elements_list), float)
    for edge in mesh.edges_list:
        if edge.left is None or edge.right is None:
            continue
        # midpoint of the edge in physical space
        pts = mesh.nodes_x_y_pos[list(edge.nodes)]
        mid = pts.mean(axis=0)
        n = edge.normal
        gl = _grad_on_element(dh, u, int(edge.left), 0.0, 0.0)  # will recompute below
        gr = _grad_on_element(dh, u, int(edge.right), 0.0, 0.0)
        # use inverse mapping to find reference coords for midpoint
        xi_l, eta_l = transform.inverse_mapping(mesh, int(edge.left), mid)
        xi_r, eta_r = transform.inverse_mapping(mesh, int(edge.right), mid)
        gl = _grad_on_element(dh, u, int(edge.left), float(xi_l), float(eta_l))
        gr = _grad_on_element(dh, u, int(edge.right), float(xi_r), float(eta_r))
        jump_n = abs(np.dot(gl - gr, n))
        h_e = np.linalg.norm(pts[1] - pts[0])
        contrib = h_e * (jump_n ** 2)
        indicators[int(edge.left)] += contrib
        indicators[int(edge.right)] += contrib
    return indicators


# -----------------------------------------------------------------------------
# Solve one step
# -----------------------------------------------------------------------------


@dataclass
class SolveResult:
    mesh: Mesh
    dof_handler: DofHandler
    solution: Function
    l2_error: float
    continuity_jump: float
    indicators: np.ndarray
    constraints_info: str


def solve_once(mesh: Mesh, poly_order: int) -> SolveResult:
    me = MixedElement(mesh, {"u": poly_order})
    dh = DofHandler(me, method="cg")
    tag_lshape_boundaries(mesh)

    # RHS is zero
    f_fun = Constant(0.0)

    u = Function(name="u_k", field_name="u", dof_handler=dh)
    du = TrialFunction("u", dof_handler=dh)
    v = TestFunction("u", dof_handler=dh)

    # Dirichlet BC from exact solution
    bcs = [
        BoundaryCondition("u", "dirichlet", "left", lambda x, y: exact_u(x, y)),
        BoundaryCondition("u", "dirichlet", "right", lambda x, y: exact_u(x, y)),
        BoundaryCondition("u", "dirichlet", "bottom", lambda x, y: exact_u(x, y)),
        BoundaryCondition("u", "dirichlet", "top", lambda x, y: exact_u(x, y)),
        BoundaryCondition("u", "dirichlet", "reentrant", lambda x, y: exact_u(x, y)),
    ]

    dx_all = dx(metadata={"q": 6})
    a = inner(grad(du), grad(v)) * dx_all
    r = (inner(grad(u), grad(v)) - f_fun * v) * dx_all
    eq = Equation(a, r)
    K, F = assemble_form(eq, dof_handler=dh, bcs=bcs, backend=os.getenv("BACKEND", "jit"))

    constraints = dh.build_hanging_node_constraints()
    if constraints is not None:
        print(f"[constraints] slaves={len(constraints.slave_to_master)} masters={constraints.n_master}")
        K = constraints.E_T @ (K @ constraints.E)
        F = constraints.E_T @ F
        sol = spla.spsolve(K.tocsc(), F)
        sol_full = constraints.prolong(sol)
    else:
        print("[constraints] none")
        sol_full = spla.spsolve(K.tocsc(), F)

    u.set_nodal_values(np.arange(len(sol_full), dtype=int), np.asarray(sol_full, float))

    # L2 error (elementwise)
    qp, qw = vol_rule(mesh.element_type, 6)
    elem_err = []
    for elem in mesh.elements_list:
        gd = dh.element_dofs("u", elem.id)
        coeffs = u.get_nodal_values(gd)
        loc = 0.0
        for (xi, eta), w in zip(qp, qw):
            J = transform.jacobian(mesh, elem.id, (float(xi), float(eta)))
            detJ = abs(float(np.linalg.det(J)))
            phi = dh.mixed_element.basis("u", float(xi), float(eta))[dh.mixed_element.slice("u")]
            uh = float(coeffs @ phi)
            xg = transform.x_mapping(mesh, elem.id, (float(xi), float(eta)))
            ue = exact_u(float(xg[0]), float(xg[1]))
            loc += w * detJ * (uh - ue) ** 2
        elem_err.append(loc)
    l2_err = math.sqrt(sum(elem_err))

    indicators = kelly_indicators(mesh, dh, u)

    # Continuity diagnostic at shared coordinates
    coord_to_vals = {}
    node_coords = mesh.nodes_x_y_pos
    for gdof, (fld, nid) in dh._dof_to_node_map.items():
        if fld != "u":
            continue
        key = (round(float(node_coords[int(nid), 0]), 12), round(float(node_coords[int(nid), 1]), 12))
        val = float(u.get_nodal_values(np.array([gdof], dtype=int))[0])
        coord_to_vals.setdefault(key, []).append(val)
    jumps = [abs(max(vs) - min(vs)) for vs in coord_to_vals.values() if len(vs) > 1]
    continuity_jump = max(jumps) if jumps else 0.0

    return SolveResult(
        mesh=mesh,
        dof_handler=dh,
        solution=u,
        l2_error=l2_err,
        continuity_jump=continuity_jump,
        indicators=indicators,
        constraints_info="[constraints] none" if constraints is None else f"[constraints] slaves={len(constraints.slave_to_master)}",
    )


# -----------------------------------------------------------------------------
# Mesh builder for L-shape
# -----------------------------------------------------------------------------


def _build_gmsh_lshape(path: Path, poly_order: int, size: float = 0.5) -> None:
    """
    Build a quadrilateral gmsh mesh for the L-shaped domain and write to *path*.
    """
    import gmsh  # type: ignore

    gmsh.initialize()
    try:
        gmsh.model.add("lshape")
        occ = gmsh.model.occ
        outer = occ.addRectangle(-1.0, -1.0, 0.0, 2.0, 2.0)
        cut = occ.addRectangle(0.0, -1.0, 0.0, 1.0, 1.0)  # remove lower-right quadrant
        ov, _ = occ.cut([(2, outer)], [(2, cut)], removeObject=True, removeTool=True)
        surf_tag = ov[0][1]
        occ.synchronize()

        gmsh.model.addPhysicalGroup(2, [surf_tag], tag=1)
        gmsh.model.setPhysicalName(2, 1, "domain")

        # Boundary curves classification
        boundary = gmsh.model.getBoundary([(2, surf_tag)], oriented=False, recursive=False)
        curves = [tag for (dim, tag) in boundary if dim == 1]
        left = []
        right = []
        top = []
        bottom = []
        reentrant = []
        tol = 1e-8
        for c in curves:
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, c)
            xmid = 0.5 * (xmin + xmax)
            ymid = 0.5 * (ymin + ymax)
            if abs(xmin + 1.0) < tol and abs(xmax + 1.0) < tol:
                left.append(c)
            elif abs(xmin - 1.0) < tol and abs(xmax - 1.0) < tol:
                right.append(c)
            elif abs(ymin - 1.0) < tol and abs(ymax - 1.0) < tol:
                top.append(c)
            elif abs(ymin + 1.0) < tol and abs(ymax + 1.0) < tol:
                bottom.append(c)
            elif abs(xmid) < tol or abs(ymid) < tol:
                # interface of the cut-out quadrant
                reentrant.append(c)

        def _phys(name: str, curve_list: list[int], tag_hint: int | None = None):
            if not curve_list:
                return
            pid = gmsh.model.addPhysicalGroup(1, curve_list, tag=tag_hint if tag_hint is not None else -1)
            gmsh.model.setPhysicalName(1, pid, name)

        _phys("left", left)
        _phys("right", right)
        _phys("top", top)
        _phys("bottom", bottom)
        _phys("reentrant", reentrant)

        gmsh.option.setNumber("Mesh.Algorithm", 8)  # frontal quad
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(int(poly_order))
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def initial_lshape_mesh(poly_order: int) -> Mesh:
    """
    Build an L-shaped mesh. Prefer gmsh (quad) if available; fall back to the
    structured generator with a cut-out quadrant otherwise.
    """
    use_gmsh = os.getenv("LSHAPE_USE_GMSH", "1") != "0"
    msh_path = Path(os.getenv("LSHAPE_MSH", "examples/meshes/lshape_quad.msh"))

    if use_gmsh:
        try:
            import gmsh  # type: ignore
        except Exception:
            gmsh = None

        if gmsh is not None:
            msh_path.parent.mkdir(parents=True, exist_ok=True)
            _build_gmsh_lshape(msh_path, poly_order, size=0.5)
            gmsh_mesh = mesh_from_gmsh(msh_path)
            # Ensure boundary tags exist even if physical groups are missing
            tag_lshape_boundaries(gmsh_mesh)
            return gmsh_mesh

    # --- fallback: structured quad with quadrant removed ---
    nodes, elems, edges, corners = structured_quad(Lx=2.0, Ly=2.0, nx=4, ny=4, poly_order=poly_order, offset=(-1.0, -1.0))
    keep = []
    keep_corners = []
    for econn, cconn in zip(elems, corners):
        pts = np.asarray([nodes[i].__dict__ for i in cconn])
        cx = float(np.mean([p["x"] for p in pts]))
        cy = float(np.mean([p["y"] for p in pts]))
        if (cx > 0.0) and (cy < 0.0):
            continue
        keep.append(econn)
        keep_corners.append(cconn)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=np.asarray(keep, dtype=int),
        edges_connectivity=None,
        elements_corner_nodes=np.asarray(keep_corners, dtype=int),
        element_type="quad",
        poly_order=poly_order,
    )
    tag_lshape_boundaries(mesh)
    return mesh


# -----------------------------------------------------------------------------
# Adaptive loop
# -----------------------------------------------------------------------------


def adaptive_solve(poly_order: int = 2, cycles: int = 3, mark_fraction: float = 0.25) -> None:
    """
    Run a few adaptive refinement cycles. By default only the final mesh is
    shown when ``PLOT_FINAL`` is set. To dump every cycle (without blocking),
    set ``PLOT_EACH=1`` and optional ``PLOT_DIR`` to control the output path.
    """
    plot_each = os.getenv("PLOT_EACH", "0") != "0"
    plot_dir = Path(os.getenv("PLOT_DIR", "."))
    plot_dir.mkdir(parents=True, exist_ok=True) if plot_each else None

    mesh = initial_lshape_mesh(poly_order)
    for cycle in range(cycles):
        print(f"\n=== Cycle {cycle} ===")
        result = solve_once(mesh, poly_order)
        print(f"L2 error: {result.l2_error:.3e}, continuity jump: {result.continuity_jump:.3e}")
        print(result.constraints_info)
        print(f"number of elements: {mesh.n_elements}")
        if plot_each:
            ax = plot_mesh_2(mesh, show=False, plot_nodes=False, elem_tags=True, edge_colors=True)
            ax.figure.tight_layout()
            fname = plot_dir / f"mesh_cycle_{cycle}.png"
            ax.figure.savefig(fname, dpi=200)
            plt.close(ax.figure)
            print(f"[plot] saved {fname}")
        if cycle == cycles - 1:
            break
        n_mark = max(1, int(mark_fraction * len(result.indicators)))
        marked = set(np.argsort(result.indicators)[-n_mark:])
        mesh = refine_marked(mesh, marked)

    if os.getenv("PLOT_FINAL", "0") != "0":
        plot_mesh_2(mesh, show=True, plot_nodes=False, elem_tags=False, edge_colors=True)


if __name__ == "__main__":
    adaptive_solve()
