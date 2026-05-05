#!/usr/bin/env python
"""
Moving-domain CutFEM verification against the ngsxfem `moving_domain.py` demo.

We solve a scalar convection–diffusion problem on a translating circular domain
described by a level set. The exact solution is known analytically; the goal is
to reproduce the ngsxfem error tables using the same geometry and time-stepping
parameters. The mesh is generated with Gmsh so it can be reused by both codes.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import scipy.sparse.linalg as sp_la

try:
    import gmsh  # type: ignore
except Exception:  # pragma: no cover - gmsh is an optional dependency
    gmsh = None

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import LevelSetFunction
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.fem import transform
from pycutfem.integration.quadrature import volume
from pycutfem.io.vtk import export_vtk
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    grad,
    inner,
    dot,
    jump,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dx, dGhost
from pycutfem.utils.bitset import BitSet
from pycutfem.utils.gmsh_loader import mesh_from_gmsh
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.helpers_geom import (
    clip_triangle_to_side,
    corner_tris,
    fan_triangulate,
    map_ref_tri_to_phys,
)
from dataclasses import dataclass, asdict

# -----------------------------------------------------------------------------
# Geometry and problem data (mirror ngsxfem demo)
# -----------------------------------------------------------------------------
LOWERLEFT = (-0.7, -0.7)
UPPERRIGHT = (0.9, 0.7)

r0 = 0.5
r1 = math.pi / (2.0 * r0)
nu = 1e-5
velmax = 2.0
c_gamma = 1.0


@dataclass
class MovingDomainResult:
    """Compact container for convergence logging and CSV export."""
    Lx: int
    Lt: int
    h_max: float
    dt: float
    n_steps: int
    err_l2l2: float
    err_l2h1: float
    linf_l2: float
    gamma_s: float
    delta: float
    mesh_file: str
    poly_order: int
    step_errors: list

    def as_dict(self):
        out = asdict(self)
        return out


# -----------------------------------------------------------------------------
# Level-set helper
# -----------------------------------------------------------------------------
class MovingCircleLevelSet(LevelSetFunction):
    """Signed distance to a translating circle."""

    def __init__(self, radius: float, *, shift: float = 0.0):
        self.radius = float(radius)
        self.shift = float(shift)
        self.time = 0.0

    def set_time(self, t: float) -> None:
        self.time = float(t)

    def with_shift(self, shift: float) -> "MovingCircleLevelSet":
        other = MovingCircleLevelSet(self.radius, shift=shift)
        other.time = self.time
        return other

    def rho(self) -> float:
        return (1.0 / math.pi) * math.sin(2.0 * math.pi * self.time)

    def __call__(self, x):
        arr = np.asarray(x, dtype=float)
        dx = arr[..., 0] - self.rho()
        dy = arr[..., 1]
        val = np.sqrt(dx * dx + dy * dy) - self.radius + self.shift
        return val

    def gradient(self, x):
        arr = np.asarray(x, dtype=float)
        dx = arr[..., 0] - self.rho()
        dy = arr[..., 1]
        nrm = np.sqrt(dx * dx + dy * dy)
        nrm_safe = np.where(nrm == 0.0, 1.0, nrm)
        gx = dx / nrm_safe
        gy = dy / nrm_safe
        return np.stack((gx, gy), axis=-1) if arr.ndim > 1 else np.array([gx, gy], float)


# -----------------------------------------------------------------------------
# Exact solution and forcing
# -----------------------------------------------------------------------------
def rho_fun(t: float) -> float:
    return (1.0 / math.pi) * math.sin(2.0 * math.pi * t)


def u_exact(x, y, t: float):
    rr = np.sqrt((x - rho_fun(t)) ** 2 + y**2)
    return np.cos(r1 * rr) ** 2


def grad_u_exact(x, y, t: float) -> Tuple[np.ndarray, np.ndarray]:
    rr = np.sqrt((x - rho_fun(t)) ** 2 + y**2)
    denom = np.where(rr == 0.0, 1.0, rr)
    factor = -math.pi * np.sin(math.pi / r0 * rr)
    gx = factor * (x - rho_fun(t)) / denom
    gy = factor * y / denom
    return gx, gy


def rhs_val(x, y, t: float) -> np.ndarray:
    rr = np.sqrt((x - rho_fun(t)) ** 2 + y**2)
    s = np.sin(r1 * rr)
    c = np.cos(r1 * rr)
    term = -(math.pi / r0) * r1 * (s * s - c * c) + (math.pi / r0) * c * s / np.where(rr == 0.0, 1.0, rr)
    return nu * term


# -----------------------------------------------------------------------------
# Mesh handling
# -----------------------------------------------------------------------------
def build_gmsh_mesh(path: Path, *, h: float, order: int = 1, element_type: str = "quad", visualize: bool = False) -> Path:
    """
    Generate a transfinite mesh for the rectangular background domain.
    element_type: "tri" or "quad"
    """
    if gmsh is None:
        raise RuntimeError("gmsh is required to build the background mesh.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.model.add("moving_domain")
        x0, y0 = LOWERLEFT
        x1, y1 = UPPERRIGHT
        p1 = gmsh.model.occ.addPoint(x0, y0, 0.0)
        p2 = gmsh.model.occ.addPoint(x1, y0, 0.0)
        p3 = gmsh.model.occ.addPoint(x1, y1, 0.0)
        p4 = gmsh.model.occ.addPoint(x0, y1, 0.0)
        lines = [
            gmsh.model.occ.addLine(p1, p2),
            gmsh.model.occ.addLine(p2, p3),
            gmsh.model.occ.addLine(p3, p4),
            gmsh.model.occ.addLine(p4, p1),
        ]
        loop = gmsh.model.occ.addCurveLoop(lines)
        surf = gmsh.model.occ.addPlaneSurface([loop])
        gmsh.model.occ.synchronize()

        Lx = x1 - x0
        Ly = y1 - y0
        nx = max(2, int(math.ceil(Lx / h)))
        ny = max(2, int(math.ceil(Ly / h)))

        gmsh.model.mesh.setTransfiniteCurve(lines[0], nx + 1)
        gmsh.model.mesh.setTransfiniteCurve(lines[2], nx + 1)
        gmsh.model.mesh.setTransfiniteCurve(lines[1], ny + 1)
        gmsh.model.mesh.setTransfiniteCurve(lines[3], ny + 1)
        gmsh.model.mesh.setTransfiniteSurface(surf)
        gmsh.option.setNumber("Mesh.Algorithm", 8)  # frontal
        if element_type == "quad":
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
        else:
            gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)

        gmsh.model.addPhysicalGroup(2, [surf], tag=1)
        gmsh.model.setPhysicalName(2, 1, "background")
        # Tag boundaries to help external readers (ngsxfem) keep edge info
        edge_tag = gmsh.model.addPhysicalGroup(1, lines, tag=11)
        gmsh.model.setPhysicalName(1, edge_tag, "boundary")

        if visualize:
            gmsh.fltk.initialize()
            gmsh.fltk.run()

        gmsh.write(str(path))
        return path
    finally:
        gmsh.finalize()


def element_masks(mesh, level_set: LevelSetFunction, tol: float = 1e-10):
    """Return (any_neg, any_pos, all_neg) boolean masks over elements."""
    phi_nodes = level_set.evaluate_on_nodes(mesh)
    elem_phi = phi_nodes[mesh.corner_connectivity]
    any_neg = (elem_phi < -tol).any(axis=1)
    any_pos = (elem_phi > tol).any(axis=1)
    all_neg = (elem_phi < -tol).all(axis=1)
    return any_neg, any_pos, all_neg


def ring_edge_bitset(mesh, outer_mask: np.ndarray, ring_mask: np.ndarray) -> BitSet:
    """Edges that touch the extension strip; includes ring–ring and ring–outer facets."""
    edge_mask = np.zeros(len(mesh.edges_list), dtype=bool)
    for e in mesh.edges_list:
        if e.right is None:
            continue  # boundary edge
        l = int(e.left)
        r = int(e.right)
        if (ring_mask[l] or ring_mask[r]) and (outer_mask[l] or outer_mask[r]):
            edge_mask[int(e.gid)] = True
    return BitSet(edge_mask)


def estimate_h_max(mesh) -> float:
    h_max = 0.0
    coords = mesh.nodes_x_y_pos
    for elem in mesh.elements_list:
        pts = coords[np.asarray(elem.corner_nodes, int)]
        rolled = np.roll(pts, -1, axis=0)
        h_max = max(h_max, float(np.max(np.linalg.norm(pts - rolled, axis=1))))
    return h_max


# -----------------------------------------------------------------------------
# Error computation
# -----------------------------------------------------------------------------
def compute_errors(mesh, element: MixedElement, dof_handler: DofHandler, sol_vec: np.ndarray,
                   level_set: LevelSetFunction, physical_mask: np.ndarray, t: float,
                   q_order: int) -> Tuple[float, float]:
    """Compute L2 and H1 errors over phi<0."""
    l2_sq = 0.0
    h1_sq = 0.0
    qp_ref, qw_ref = volume(mesh.element_type, q_order)
    qp_ref_tri, qw_ref_tri = volume("tri", q_order)

    for eid in np.nonzero(physical_mask)[0]:
        elem = mesh.elements_list[int(eid)]
        gdofs = dof_handler.get_elemental_dofs(int(eid))
        u_loc = sol_vec[gdofs]

        if elem.tag == "inside":
            for (xi, eta), w in zip(qp_ref, qw_ref):
                J = transform.jacobian(mesh, int(eid), (xi, eta))
                detJ = abs(np.linalg.det(J))
                x_phys, y_phys = transform.x_mapping(mesh, int(eid), (xi, eta))
                phi = element.basis("u", xi, eta)
                grad_ref = element.grad_basis("u", xi, eta)
                grad_phys = transform.map_grad_scalar(mesh, int(eid), grad_ref, (xi, eta))
                uh = float(phi @ u_loc)
                guh = (u_loc @ grad_phys).ravel()
                uex = float(u_exact(x_phys, y_phys, t))
                gux, guy = grad_u_exact(x_phys, y_phys, t)
                l2_sq += (uh - uex) ** 2 * w * detJ
                h1_sq += ((guh[0] - gux) ** 2 + (guh[1] - guy) ** 2) * w * detJ
            continue

        # Cut cell: triangulate and clip on the negative side.
        tri_local, corner_ids = corner_tris(mesh, elem)
        for loc_tri in tri_local:
            v_ids = [corner_ids[i] for i in loc_tri]
            v_coords = mesh.nodes_x_y_pos[v_ids]
            v_phi = np.array([level_set(np.asarray(xy)) for xy in v_coords])
            polygons = clip_triangle_to_side(v_coords, v_phi, side="-")
            for poly in polygons:
                for A, B, C in fan_triangulate(poly):
                    qp_phys, qw_phys = map_ref_tri_to_phys(A, B, C, qp_ref_tri, qw_ref_tri)
                    for x_phys, w_phys in zip(qp_phys, qw_phys):
                        xi, eta = transform.inverse_mapping(mesh, int(eid), x_phys)
                        phi = element.basis("u", xi, eta)
                        grad_ref = element.grad_basis("u", xi, eta)
                        grad_phys = transform.map_grad_scalar(mesh, int(eid), grad_ref, (xi, eta))
                        uh = float(phi @ u_loc)
                        guh = (u_loc @ grad_phys).ravel()
                        uex = float(u_exact(x_phys[0], x_phys[1], t))
                        gux, guy = grad_u_exact(x_phys[0], x_phys[1], t)
                        l2_sq += (uh - uex) ** 2 * w_phys
                        h1_sq += ((guh[0] - gux) ** 2 + (guh[1] - guy) ** 2) * w_phys

    return math.sqrt(l2_sq), math.sqrt(h1_sq)


# -----------------------------------------------------------------------------
# Solver driver
# -----------------------------------------------------------------------------
def run_moving_domain(args, *, return_data: bool | None = None):
    mesh_path = Path(args.mesh_file)
    if args.rebuild_mesh or not mesh_path.exists():
        target_h = args.h0 * (0.5 ** args.Lx)
        print(f"Building Gmsh mesh at {mesh_path} with h≈{target_h:.4f}")
        build_gmsh_mesh(mesh_path, h=target_h, order=args.mesh_order, element_type=args.gmsh_element, visualize=args.view_gmsh)
    else:
        print(f"Reusing existing mesh file {mesh_path}")

    mesh = mesh_from_gmsh(mesh_path)
    h_max = estimate_h_max(mesh)
    print(f"Loaded mesh with {len(mesh.elements_list)} elements, h_max = {h_max:.4f}")

    element = MixedElement(mesh, field_specs={"u": args.poly_order})
    dof_handler = DofHandler(element, method="cg")
    total_dofs = dof_handler.total_dofs
    field_dofs = np.asarray(dof_handler.get_field_slice("u"), dtype=int)
    print(f"Total DOFs: {total_dofs}")

    # Level set
    lset = MovingCircleLevelSet(radius=r0)

    # Initial condition at t=0
    coords_u = dof_handler.get_dof_coords("u")
    u_prev = Function(name="u_prev", field_name="u", dof_handler=dof_handler)
    u_prev_vals = u_exact(coords_u[:, 0], coords_u[:, 1], 0.0)
    u_prev.set_nodal_values(field_dofs, u_prev_vals)

    inactive_bc = BoundaryCondition("u", "dirichlet", "inactive", lambda x, y: 0.0)
    dt = args.t0 * (0.5 ** args.Lt)
    n_steps = int(round(args.T_end / dt))
    delta = dt * velmax
    K_tilde = int(math.ceil(delta / h_max))
    gamma_s = c_gamma * K_tilde
    print(f"dt = {dt:.5f}, steps = {n_steps}, delta = {delta:.4f}, gamma_s = {gamma_s:.2f}")

    outer_old = np.ones(len(mesh.elements_list), dtype=bool)  # history check
    errors_L2, errors_H1 = [], []
    h_expr = CellDiameter()

    step_errors = []
    for istep in range(1, n_steps + 1):
        t = istep * dt
        lset.set_time(t)

        # Bands for extension
        any_neg_curr, _, _ = element_masks(mesh, lset)
        lset_outer = lset.with_shift(-delta)
        lset_inner = lset.with_shift(delta)
        any_neg_outer, _, _ = element_masks(mesh, lset_outer)
        _, _, all_neg_inner = element_masks(mesh, lset_inner)
        ring_mask = any_neg_outer & ~all_neg_inner

        # History check: active domain must be subset of previous outer band
        missing_history = any_neg_curr & ~outer_old
        if missing_history.any():
            missing_ids = np.nonzero(missing_history)[0]
            raise RuntimeError(f"Active elements without history: {missing_ids}")
        outer_old = any_neg_outer.copy()

        # Update mesh classification on the actual level set
        mesh.classify_elements(lset)
        mesh.classify_edges(lset)
        mesh.build_interface_segments(lset)

        physical_bs = BitSet(any_neg_curr)
        outer_bs = BitSet(any_neg_outer)
        ring_bs = BitSet(ring_mask)
        ring_edges = ring_edge_bitset(mesh, any_neg_outer, ring_mask)

        # If the ghost strip vanished (coarse mesh vs delta), fall back to the physical band
        if ring_edges.cardinality() == 0:
            ring_bs = BitSet(np.zeros(len(any_neg_curr), dtype=bool))
            outer_bs = physical_bs

        # Update inactive DOFs (everything outside the *physical* domain).
        # Use strict=True so that DOFs with any support inside the physical band remain active.
        non_outer = ~any_neg_curr
        dof_handler.dof_tags["inactive"] = set()
        dof_handler.tag_dof_bitset("inactive", "u", non_outer, strict=True)

        # Measures
        quad_order = 2 * args.poly_order + 4
        dx_phys = dx(defined_on=physical_bs, level_set=lset, metadata={"side": "-", "q": quad_order})
        u = TrialFunction("u", dof_handler=dof_handler)
        v = TestFunction("u", dof_handler=dof_handler)
        ghost_term = None
        if ring_edges.cardinality() > 0:
            dG = dGhost(defined_on=ring_edges, level_set=lset, metadata={"q": quad_order})
            ghost_term = Constant(gamma_s) * (1.0 / (h_expr * h_expr)) * jump(u) * jump(v) * dG
        w_vec = Constant(np.array([2.0 * math.cos(2.0 * math.pi * t), 0.0], dtype=float))

        # Carry previous solution into the coefficient Function
        u_prev_coeff = Function("u_prev_coeff", field_name="u", dof_handler=dof_handler)
        u_prev_coeff.set_nodal_values(field_dofs, u_prev.nodal_values.copy())

        rhs_expr = Analytic(lambda x, y: rhs_val(x, y, t))
        a = (Constant(1.0 / dt) * u * v * dx_phys
             + Constant(nu) * inner(grad(u), grad(v)) * dx_phys
             + dot(w_vec, grad(u)) * v * dx_phys)
        if ghost_term is not None:
            a = a + ghost_term
        L = rhs_expr * v * dx_phys + Constant(1.0 / dt) * u_prev_coeff * v * dx_phys

        equation = Equation(a, L)
        K, F = assemble_form(equation, dof_handler=dof_handler, bcs=[inactive_bc], backend=args.backend)
        sol = sp_la.spsolve(K.tocsc(), F)

        # Update solution for next step
        u_prev.set_nodal_values(field_dofs, sol[field_dofs])

        # Errors on the physical domain
        l2, h1 = compute_errors(mesh, element, dof_handler, sol, lset, any_neg_curr, t, q_order=quad_order)
        errors_L2.append(l2)
        errors_H1.append(h1)
        step_errors.append({"step": istep, "t": t, "L2": l2, "H1": h1, "active": physical_bs.cardinality()})

        print(f"step {istep:02d}/{n_steps}  t={t:7.4f}  L2={l2:7.3e}  H1={h1:7.3e}  active={physical_bs.cardinality()}")

        if args.save_vtk and (istep % args.vtk_every == 0 or istep == n_steps):
            outdir = Path(args.output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            fname = outdir / f"moving_domain_{istep:04d}.vtu"
            export_vtk(str(fname), mesh=mesh, dof_handler=dof_handler, functions={"u": u_prev})

    err_l2l2 = math.sqrt(dt * sum(e * e for e in errors_L2))
    err_l2h1 = math.sqrt(dt * sum(e * e for e in errors_H1))
    print("\n--------------------------------------------------------")
    print(f"L2(0,T;L2)  = {err_l2l2:8.4e}")
    print(f"Linf(0,T;L2)= {max(errors_L2):8.4e}")
    print(f"L2(0,T;H1)  = {err_l2h1:8.4e}")

    if return_data is None:
        return_data = getattr(args, "return_data", False)
    if return_data:
        return MovingDomainResult(
            Lx=args.Lx,
            Lt=args.Lt,
            h_max=h_max,
            dt=dt,
            n_steps=n_steps,
            err_l2l2=err_l2l2,
            err_l2h1=err_l2h1,
            linf_l2=max(errors_L2),
            gamma_s=gamma_s,
            delta=delta,
            mesh_file=str(mesh_path),
            poly_order=args.poly_order,
            step_errors=step_errors,
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse():
    parser = argparse.ArgumentParser(description="Moving-domain CutFEM verification (pycutfem vs ngsxfem).")
    parser.add_argument("--mesh-file", type=Path, default=Path("examples/meshes/moving_domain_rect.msh"), help="Path to the shared Gmsh mesh.")
    parser.add_argument("--rebuild-mesh", action="store_true", help="Force regeneration of the Gmsh mesh.")
    parser.add_argument("--h0", type=float, default=0.2, help="Coarse mesh size before refinement (matches ngsxfem).")
    parser.add_argument("--Lx", type=int, default=3, help="Number of bisections of h0 (matches ngsxfem).")
    parser.add_argument("--mesh-order", type=int, default=1, choices=(1, 2), help="Geometric order for the Gmsh mesh.")
    parser.add_argument("--gmsh-element", type=str, choices=("tri", "quad"), default="quad", help="Element type for the Gmsh mesh.")
    parser.add_argument("--poly-order", type=int, default=1, help="FE polynomial order.")
    parser.add_argument("--t0", type=float, default=0.1, help="Base time step size before refinement (matches ngsxfem).")
    parser.add_argument("--Lt", type=int, default=3, help="Number of time-step halvings.")
    parser.add_argument("--T-end", type=float, default=0.4, dest="T_end", help="Final time.")
    parser.add_argument("--backend", type=str, default="jit", choices=("jit", "python"), help="Assembly backend.")
    parser.add_argument("--save-vtk", action="store_true", help="Write VTK output of the solution.")
    parser.add_argument("--vtk-every", type=int, default=10, help="VTK output frequency.")
    parser.add_argument("--output-dir", type=str, default="moving_domain_results", help="VTK output directory.")
    parser.add_argument("--view-gmsh", action="store_true", help="Open the Gmsh GUI before meshing.")
    parser.add_argument("--return-data", action="store_true", help="Return results object (for programmatic calls).")
    return parser.parse_args()


if __name__ == "__main__":
    run_moving_domain(_parse())
