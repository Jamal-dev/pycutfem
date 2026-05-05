"""
Turek–Hron FSI-2 benchmark using the biofilm one-domain model (diffuse interface).

Goal
----
Validate the FSI coupling (fluid ↔ solid) in the one-domain model in
`examples/utils/biofilm/one_domain.py` by reproducing the canonical Turek–Hron
beam benchmark observables:
  - drag/lift coefficients (C_D, C_L),
  - beam-tip displacement.

Modeling choices for this benchmark
-----------------------------------
* No growth/detachment/chemistry/damage (pure FSI).
* Treat the **beam** as the only diffuse solid indicator α(x,t).
  The cylinder is treated as a rigid obstacle by **removing it from the mesh**
  (channel-with-a-hole mesh) and applying no-slip on the cylinder boundary.
* Freeze porosity φ and tie it to α to represent an (almost) impermeable solid:
    φ(x,t) = 1 - (1-φ_solid) α(x,t).
* Evolve α with conservative advection + Cahn–Hilliard regularization to preserve
  the beam area (mass conservation of α).
* Stabilize the Eulerian reference-map kinematics by restricting (u,vS) DOFs to
  a fixed rectangle around the obstacle; outside the rectangle the DOFs are
  marked inactive (time-independent active set).

Notes
-----
* Drag/lift are computed as a sum of:
    - cylinder force: traction integral on the cylinder boundary,
        F_cyl = -∫ σ_f(v,p) n ds,
    - beam force: Brinkman/penalization volume reaction,
        F_beam ≈ ∫ β (v - vS) dx,
  where `n` is the outward normal of the *fluid* domain (points into the hole).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

import numpy as np

# Avoid extremely verbose Numba debug dumps if the environment enables them.
for _k in (
    "NUMBA_DEBUG",
    "NUMBA_DUMP_BYTECODE",
    "NUMBA_DUMP_IR",
    "NUMBA_DUMP_SSA",
    "NUMBA_DEBUG_ARRAY_OPT",
):
    os.environ[_k] = "0"

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.solvers.nonlinear_solver import (
    NewtonParameters,
    PdasNewtonSolver,
    PetscSnesNewtonSolver,
    TimeStepperParameters,
    VIParameters,
)
from pycutfem.ufl.expressions import (
    Constant,
    FacetNormal,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    Identity,
    dot,
    grad,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dS, dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.refinement import TensorRefiner

from examples.utils.fsi.turek_fsi2 import build_structured_channel_mesh, tag_channel_boundaries
from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms


# -----------------------------------------------------------------------------
# Benchmark geometry (Turek–Hron FSI-2)
# -----------------------------------------------------------------------------
H = 0.41
L = 2.2
RADIUS = 0.05
CENTER = (0.2, 0.2)
BEAM_LENGTH = 0.35
BEAM_HEIGHT = 0.02

TIP_REF = np.array([CENTER[0] + RADIUS + BEAM_LENGTH, CENTER[1]], dtype=float)  # point A
# Pressure-drop probes (DFG/Turek-style): upstream/downstream of the cylinder.
PROBE_A_REF = np.array([0.15, 0.2], dtype=float)
PROBE_B_REF = np.array([0.25, 0.2], dtype=float)
D_CYL = 2.0 * RADIUS


def _mark_inactive_fields(dh: DofHandler, *field_names: str) -> None:
    tags = getattr(dh, "dof_tags", None) or {}
    inactive = set(tags.get("inactive", set()))
    for fname in field_names:
        try:
            sl = np.asarray(dh.get_field_slice(fname), dtype=int).ravel()
        except Exception:
            continue
        inactive.update(int(i) for i in sl)
    tags["inactive"] = inactive
    dh.dof_tags = tags


def _mark_inactive_dofs(dh: DofHandler, dof_ids: np.ndarray) -> None:
    tags = getattr(dh, "dof_tags", None) or {}
    inactive = set(tags.get("inactive", set()))
    for i in np.asarray(dof_ids, dtype=int).ravel():
        inactive.add(int(i))
    tags["inactive"] = inactive
    dh.dof_tags = tags


def _restrict_skeleton_dofs_to_box(
    dh: DofHandler,
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> tuple[int, int]:
    """
    Restrict (u,vS) unknowns to a fixed axis-aligned rectangle in physical space.

    This provides a **time-independent** active set that avoids having to update
    inactive/active DOFs as alpha advects. Inside the box, the standard extension
    penalties (gamma_u, u_extension_mode) keep (u,vS) well-posed in the free
    fluid where alpha is small.
    """
    x0 = float(x0)
    x1 = float(x1)
    y0 = float(y0)
    y1 = float(y1)
    if not (np.isfinite(x0) and np.isfinite(x1) and np.isfinite(y0) and np.isfinite(y1)):
        raise ValueError("Box bounds must be finite.")
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid box: x0={x0}, x1={x1}, y0={y0}, y1={y1}.")

    u_xy = np.asarray(dh.get_dof_coords("u_x"), dtype=float)
    keep = (u_xy[:, 0] >= x0) & (u_xy[:, 0] <= x1) & (u_xy[:, 1] >= y0) & (u_xy[:, 1] <= y1)
    n_keep = int(np.sum(keep))
    n_tot = int(keep.size)
    if n_keep < max(10, int(0.01 * float(n_tot))):
        raise RuntimeError(f"restrict-skeleton box keeps too few DOFs: {n_keep}/{n_tot}. Increase the box.")

    for fname in ("u_x", "u_y", "vS_x", "vS_y"):
        sl = np.asarray(dh.get_field_slice(fname), dtype=int).ravel()
        if sl.size != keep.size:
            raise RuntimeError(f"Unexpected DOF count mismatch for {fname}: slice={sl.size}, coords={keep.size}")
        _mark_inactive_dofs(dh, sl[~keep])
    return n_keep, n_tot


def _smooth_step(z: np.ndarray) -> np.ndarray:
    # Robust sigmoid: 0.5*(1+tanh(z)).
    return 0.5 * (1.0 + np.tanh(np.asarray(z, dtype=float)))


def _tag_beam_root_alpha_dirichlet_dofs(
    dh: DofHandler,
    mesh: Mesh,
    *,
    tag: str,
    mesh_size: float,
    center: tuple[float, float],
    beam_height: float,
) -> int:
    """
    Tag alpha DOFs at the beam root (arc on the cylinder boundary) for Dirichlet α=1.

    The root is the right-hand arc of the cylinder boundary with y in the beam
    height band. DOFs are tagged through `dh.dof_tags[tag]` so the BC does not
    depend on exclusive mesh edge tags.
    """
    mesh_size = float(mesh_size)
    cx, cy = float(center[0]), float(center[1])
    hy = 0.5 * float(beam_height)
    y0 = cy - hy
    y1 = cy + hy
    pad = max(0.5 * mesh_size, 1.0e-6)

    dofs: set[int] = set()
    for edge in getattr(mesh, "edges_list", []):
        if (getattr(edge, "left", None) is not None) and (getattr(edge, "right", None) is not None):
            continue
        if getattr(edge, "tag", None) != "cylinder":
            continue
        xy = np.asarray(mesh.nodes_x_y_pos[list(edge.nodes)], dtype=float)
        ex_mid = float(np.mean(xy[:, 0]))
        ey_min = float(np.min(xy[:, 1]))
        ey_max = float(np.max(xy[:, 1]))
        if ex_mid < cx:
            continue
        if ey_max < y0 - pad or ey_min > y1 + pad:
            continue
        gid = int(getattr(edge, "gid", -1))
        if gid < 0:
            continue
        dofs.update(int(gd) for gd in dh.edge_dofs("alpha", gid))

    if not dofs:
        raise RuntimeError("No alpha DOFs found for the beam-root Dirichlet constraint.")

    tags = getattr(dh, "dof_tags", None) or {}
    tags[tag] = set(dofs)
    dh.dof_tags = tags
    return int(len(dofs))


def _pin_pressure_gauge(dh: DofHandler, *, tag: str = "p_anchor") -> int | None:
    """
    Pin a single pressure DOF to remove the constant-pressure nullspace.

    We prefer a DOF on the outlet boundary (x ≈ xmax) closest to mid-height.
    Returns the pinned global DOF id (or None if not found).
    """
    try:
        coords_p = np.asarray(dh.get_dof_coords("p"), dtype=float)
        p_dofs = np.asarray(dh.get_field_slice("p"), dtype=int).ravel()
    except Exception:
        return None
    if coords_p.size == 0 or p_dofs.size == 0 or coords_p.shape[0] != p_dofs.size:
        return None

    x_max = float(np.max(coords_p[:, 0]))
    outlet = np.where(np.isclose(coords_p[:, 0], x_max, atol=1.0e-10))[0]
    if outlet.size == 0:
        outlet = np.arange(coords_p.shape[0], dtype=int)

    y_mid = 0.5 * float(H)
    loc = int(outlet[np.argmin(np.abs(coords_p[outlet, 1] - y_mid))])
    gdof = int(p_dofs[loc])
    tags = getattr(dh, "dof_tags", None) or {}
    tags.setdefault(tag, set()).add(gdof)
    dh.dof_tags = tags
    return gdof


def _turek_obstacle_level_set(
    x: np.ndarray,
    y: np.ndarray,
    *,
    center: tuple[float, float],
    radius: float,
    beam_length: float,
    beam_height: float,
    root_inset: float = 0.0,
) -> np.ndarray:
    """
    Signed level-set-like function for the **beam** with a curved root following
    the cylinder arc.

    Returns phi(x,y) with:
      phi < 0 inside the beam,
      phi > 0 outside.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cx, cy = float(center[0]), float(center[1])
    r = float(radius)

    # Beam: rectangle with a curved root following the right half of the cylinder arc.
    hy = 0.5 * float(beam_height)
    x_tip = cx + r + float(beam_length)
    y0 = cy - hy
    y1 = cy + hy

    dy = y - cy
    inside = np.maximum(r * r - dy * dy, 0.0)
    x_arc = cx + np.sqrt(inside) - float(root_inset)

    phi_left = x_arc - x
    phi_right = x - x_tip
    phi_bottom = y0 - y
    phi_top = y - y1
    phi_beam = np.max(np.stack((phi_left, phi_right, phi_bottom, phi_top), axis=-1), axis=-1)

    return phi_beam


def _mark_bbox(mesh: Mesh, *, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    corners_all = np.asarray(mesh.nodes_x_y_pos[np.asarray(mesh.corner_connectivity, dtype=int)], dtype=float)
    ex_min = corners_all[..., 0].min(axis=1)
    ex_max = corners_all[..., 0].max(axis=1)
    ey_min = corners_all[..., 1].min(axis=1)
    ey_max = corners_all[..., 1].max(axis=1)
    marked = np.nonzero((ex_max >= float(x0)) & (ex_min <= float(x1)) & (ey_max >= float(y0)) & (ey_min <= float(y1)))[0]
    return np.asarray(marked, dtype=int)


def _refine_bbox(mesh: Mesh, *, x0: float, x1: float, y0: float, y1: float, levels: int) -> Mesh:
    levels = int(max(0, int(levels)))
    if levels <= 0:
        return mesh

    marked = _mark_bbox(mesh, x0=float(x0), x1=float(x1), y0=float(y0), y1=float(y1))
    if int(marked.size) <= 0:
        return mesh

    rx = np.zeros(len(mesh.elements_list), dtype=int)
    ry = np.zeros(len(mesh.elements_list), dtype=int)
    rx[marked] = int(levels)
    ry[marked] = int(levels)
    refiner = TensorRefiner(max_ratio=2.0, max_ref=int(levels))
    mesh1 = refiner.refine(mesh, rx, ry)
    return mesh1


def _interp_u_idw_factory(
    dh: DofHandler, u_k: VectorFunction, *, active_only: bool = True
):  # -> Callable[[np.ndarray], np.ndarray]
    u_coords_all = np.asarray(dh.get_dof_coords("u_x"), dtype=float)
    u_slice_all = np.asarray(dh.get_field_slice("u_x"), dtype=int).ravel()
    inactive = set((getattr(dh, "dof_tags", None) or {}).get("inactive", set()))
    if active_only:
        u_active_mask = np.asarray([int(g) not in inactive for g in u_slice_all.tolist()], dtype=bool)
        if int(np.sum(u_active_mask)) < 4:
            u_active_mask[:] = True
    else:
        u_active_mask = np.ones_like(u_slice_all, dtype=bool)
    u_coords = np.asarray(u_coords_all[u_active_mask, :], dtype=float)

    def _interp_u_idw(xq: np.ndarray, *, k: int = 12, power: float = 2.0) -> np.ndarray:
        xq = np.asarray(xq, dtype=float)
        if xq.ndim == 1:
            xq = xq.reshape(1, 2)
        if xq.ndim != 2 or xq.shape[1] != 2:
            raise ValueError("xq must have shape (N,2)")
        k = int(max(1, int(k)))

        ux = np.asarray(u_k.components[0].nodal_values, dtype=float).ravel()
        uy = np.asarray(u_k.components[1].nodal_values, dtype=float).ravel()
        if ux.size != uy.size or ux.size != u_slice_all.size:
            raise RuntimeError("Unexpected mismatch between u DOF coords and nodal values.")
        u_nodes = np.column_stack([ux[u_active_mask], uy[u_active_mask]])

        d2 = np.sum((u_coords[None, :, :] - xq[:, None, :]) ** 2, axis=2)  # (N,Ndof_active)
        kk = min(k, int(d2.shape[1]))
        idx = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        d2_k = np.take_along_axis(d2, idx, axis=1)

        j0 = np.argmin(d2_k, axis=1)
        d2_min = d2_k[np.arange(d2_k.shape[0]), j0]
        hit = d2_min <= 1.0e-28
        out = np.empty((xq.shape[0], 2), dtype=float)
        if np.any(hit):
            rows = np.where(hit)[0]
            cols = j0[rows]
            out[rows, :] = u_nodes[idx[rows, cols], :]

        miss = ~hit
        if np.any(miss):
            dist = np.sqrt(np.maximum(d2_k[miss, :], 1.0e-28))
            w = 1.0 / (dist**float(power))
            w_sum = np.sum(w, axis=1, keepdims=True)
            w = w / np.maximum(w_sum, 1.0e-30)
            out[miss, :] = np.sum(u_nodes[idx[miss, :], :] * w[:, :, None], axis=1)
        return np.asarray(out, dtype=float)

    return _interp_u_idw


def main() -> None:
    ap = argparse.ArgumentParser(description="Turek–Hron FSI-2 benchmark (one-domain diffuse-interface).")
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))

    # Benchmark presets (Turek–Hron).
    ap.add_argument(
        "--turek-case",
        choices=("fsi1", "fsi2", "fsi3"),
        default="fsi2",
        help="Turek–Hron preset: fsi1=steady (Re≈20), fsi2=periodic (Re≈100), fsi3=chaotic (Re≈200).",
    )

    # Mesh
    ap.add_argument("--mesh-size", type=float, default=0.01, help="Target mesh size (used if --nx/--ny are 0).")
    ap.add_argument("--nx", type=int, default=0, help="Cells in x (0=auto from --mesh-size).")
    ap.add_argument("--ny", type=int, default=0, help="Cells in y (0=auto from --mesh-size).")
    ap.add_argument("--q", type=int, default=6, help="Quadrature order (dx metadata + solver quad_order).")
    ap.add_argument("--refine-obstacle", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--refine-pad", type=float, default=0.05, help="Padding around obstacle bbox for refinement.")
    ap.add_argument("--refine-levels", type=int, default=1, help="Tensor refinement levels inside refinement bbox.")
    ap.add_argument(
        "--refine-beam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refine around the beam bbox (default: enabled). Keeps <=1 hanging node per edge by using a single refinement level.",
    )
    ap.add_argument("--refine-beam-pad", type=float, default=0.03, help="Padding around beam bbox for refinement.")

    # Time stepping
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--allow-dt-reduction", action="store_true", help="Reduce dt when Newton fails (robustness).")
    ap.add_argument("--dt-min", type=float, default=0.0, help="Minimum dt allowed with --allow-dt-reduction.")
    ap.add_argument("--dt-reduction-factor", type=float, default=0.5, help="dt <- factor*dt on step rejection.")
    ap.add_argument("--t-final", type=float, default=10.0)
    ap.add_argument("--theta", type=float, default=None)
    ap.add_argument("--t-ramp", type=float, default=2.0, help="Inflow ramp time.")

    # Nonlinear solve
    ap.add_argument("--newton-tol", type=float, default=5.0e-6)
    ap.add_argument("--max-it", type=int, default=20)
    ap.add_argument(
        "--snes-accept-factor",
        type=float,
        default=3.0,
        help="If PETSc SNES reports non-convergence but ‖F‖ <= factor*newton_tol, accept the best iterate (0 disables).",
    )
    ap.add_argument(
        "--petsc-monitor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable PETSc SNES/KSP monitors (prints Newton iterations for --newton-solver snes). "
        "Also enabled by env var PYCUTFEM_PETSC_MONITOR=1.",
    )
    ap.add_argument(
        "--newton-solver",
        type=str,
        default="snes",
        choices=("snes", "pdas"),
        help="Nonlinear solver: PETSc SNES (default) or PDAS (both support alpha box constraints).",
    )
    ap.add_argument(
        "--alpha-box-constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce 0<=alpha<=1 via box constraints (default: enabled). Use --no-alpha-box-constraints to disable.",
    )

    # Output
    ap.add_argument("--out-dir", type=str, default="out/turek_fsi2_one_domain")
    ap.add_argument("--vtk-every", type=int, default=10, help="Write VTK every N steps (0 disables).")

    # Diffuse interface + transport
    ap.add_argument("--eps", type=float, default=0.01, help="Diffuse interface half-thickness for alpha0.")
    ap.add_argument("--alpha-advection-form", type=str, default="conservative", choices=("advective", "conservative"))
    ap.add_argument("--alpha-ch-M", type=float, default=1.0e-6, help="Cahn–Hilliard mobility for alpha.")
    ap.add_argument("--alpha-ch-gamma", type=float, default=1.0e-2, help="Cahn–Hilliard energy weight for alpha.")
    ap.add_argument("--alpha-ch-eps", type=float, default=float("nan"), help="Cahn–Hilliard interface thickness (default: --eps).")

    # Solid / coupling parameters
    ap.add_argument("--rho-f", type=float, default=1.0e3)
    ap.add_argument("--mu-f", type=float, default=1.0)
    ap.add_argument("--u-mean", type=float, default=None)
    ap.add_argument("--kappa-inv", type=float, default=1.0e8, help="Inverse permeability scaling for Brinkman drag.")
    ap.add_argument("--phi-solid", type=float, default=0.1, help="Frozen porosity inside the solid (0<phi_solid<1).")
    ap.add_argument("--solid-model", type=str, default="svk", choices=("linear", "hencky", "svk", "neo_hookean"))
    ap.add_argument("--E-s", type=float, default=1.4e6, help="Solid Young's modulus (FSI-2 default ≈ 1.4e6).")
    ap.add_argument("--nu-s", type=float, default=0.4)
    ap.add_argument("--rho-s", type=float, default=None)
    ap.add_argument(
        "--skeleton-inertia-convection",
        type=str,
        default="lagged",
        choices=("lagged", "full"),
        help="How to treat div(rho_S vS⊗vS) in the Eulerian skeleton inertia term.",
    )

    # Extension / stabilization for (u,vS) in the free-fluid region.
    ap.add_argument("--gamma-u", type=float, default=1.0e-6, help="u extension penalty outside the solid.")
    ap.add_argument("--u-extension", type=str, default="grad", choices=("l2", "grad"))
    ap.add_argument("--gamma-u-pin", type=float, default=1.0e-10, help="Tiny L2 pin (only for u-extension=grad).")

    # Active-DOF rectangle (requested approach).
    ap.add_argument("--restrict-skeleton-dofs", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--restrict-box-x0", type=float, default=float("nan"))
    ap.add_argument("--restrict-box-x1", type=float, default=float("nan"))
    ap.add_argument("--restrict-box-y0", type=float, default=float("nan"))
    ap.add_argument("--restrict-box-y1", type=float, default=float("nan"))
    ap.add_argument("--restrict-box-pad", type=float, default=0.15, help="Auto-box padding around obstacle bbox.")

    # Pressure gauge / outlet treatment.
    ap.add_argument(
        "--pressure-bc",
        choices=("point", "outlet", "none"),
        default="point",
        help="Pressure handling: 'point' pins one pressure DOF (do-nothing outlet), "
        "'outlet' enforces p=0 on the whole outlet, 'none' applies no pressure Dirichlet (pure do-nothing outlet).",
    )

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    backend = str(args.backend)

    # ------------------------------------------------------------------
    # Turek–Hron case presets
    # ------------------------------------------------------------------
    case_defaults = {
        "fsi1": {"u_mean": 0.2, "rho_s": 1.0e3, "dt": 1.0, "theta": 1.0},
        "fsi2": {"u_mean": 1.0, "rho_s": 1.0e4, "dt": 0.005, "theta": 0.5},
        "fsi3": {"u_mean": 2.0, "rho_s": 1.0e4, "dt": 0.005, "theta": 0.5},
    }
    preset = case_defaults.get(str(args.turek_case), case_defaults["fsi2"])

    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------
    mesh_size = float(args.mesh_size)
    qdeg = int(args.q)

    mesh = build_structured_channel_mesh(mesh_size, poly_order=2)

    # Optional local refinement (benchmark: keep <=1 hanging node per edge).
    if bool(args.refine_obstacle) or bool(args.refine_beam):
        if bool(args.refine_obstacle) and int(args.refine_levels) != 1:
            raise ValueError("This benchmark assumes <=1 hanging node per edge; use --refine-levels 1.")
        rx = np.zeros(len(mesh.elements_list), dtype=int)
        ry = np.zeros(len(mesh.elements_list), dtype=int)

        if bool(args.refine_obstacle):
            pad = float(args.refine_pad)
            x0_ref = float(CENTER[0] - RADIUS) - pad
            x1_ref = float(CENTER[0] + RADIUS + BEAM_LENGTH) + pad
            y0_ref = float(CENTER[1] - max(RADIUS, 0.5 * BEAM_HEIGHT)) - pad
            y1_ref = float(CENTER[1] + max(RADIUS, 0.5 * BEAM_HEIGHT)) + pad
            marked = _mark_bbox(mesh, x0=x0_ref, x1=x1_ref, y0=y0_ref, y1=y1_ref)
            rx[marked] = 1
            ry[marked] = 1

        if bool(args.refine_beam):
            pad = float(args.refine_beam_pad)
            x0_ref = float(CENTER[0] + RADIUS) - pad
            x1_ref = float(CENTER[0] + RADIUS + BEAM_LENGTH) + pad
            y0_ref = float(CENTER[1] - 0.5 * BEAM_HEIGHT) - pad
            y1_ref = float(CENTER[1] + 0.5 * BEAM_HEIGHT) + pad
            marked = _mark_bbox(mesh, x0=x0_ref, x1=x1_ref, y0=y0_ref, y1=y1_ref)
            rx[marked] = 1
            ry[marked] = 1

        if np.any(rx) or np.any(ry):
            refiner = TensorRefiner(max_ratio=2.0, max_ref=1)
            mesh = refiner.refine(mesh, rx, ry)

    # (Re-)tag boundary edges (incl. cylinder) after optional refinement.
    tag_channel_boundaries(mesh, mesh_size)

    # ------------------------------------------------------------------
    # Mixed space
    # ------------------------------------------------------------------
    field_specs = {
        "v_x": 2,
        "v_y": 2,
        "p": 1,
        "vS_x": 2,
        "vS_y": 2,
        "u_x": 2,
        "u_y": 2,
        "phi": 1,
        "alpha": 1,
        "mu_alpha": 1,
        "S": 1,
    }

    me = MixedElement(mesh, field_specs=field_specs)
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
    VS = FunctionSpace("VS", ["vS_x", "vS_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    dv = VectorTrialFunction(space=V, dof_handler=dh)
    dvS = VectorTrialFunction(space=VS, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    dp = TrialFunction("p", dof_handler=dh)
    dphi = TrialFunction("phi", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    dmu_alpha = TrialFunction("mu_alpha", dof_handler=dh)
    dS_trial = TrialFunction("S", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    vS_test = VectorTestFunction(space=VS, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    mu_alpha_test = TestFunction("mu_alpha", dof_handler=dh)
    S_test = TestFunction("S", dof_handler=dh)

    # Unknowns (k) and previous (n)
    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    vS_k = VectorFunction("vS_k", ["vS_x", "vS_y"], dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    mu_alpha_k = Function("mu_alpha_k", "mu_alpha", dof_handler=dh)
    S_k = Function("S_k", "S", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    vS_n = VectorFunction("vS_n", ["vS_x", "vS_y"], dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    mu_alpha_n = Function("mu_alpha_n", "mu_alpha", dof_handler=dh)
    S_n = Function("S_n", "S", dof_handler=dh)

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------
    v_n.nodal_values[:] = 0.0
    p_n.nodal_values[:] = 0.0
    vS_n.nodal_values[:] = 0.0
    u_n.nodal_values[:] = 0.0
    mu_alpha_n.nodal_values[:] = 0.0
    # Substrate field is unused in this pure-FSI run (keep frozen at 0).
    S_n.nodal_values[:] = 0.0

    eps0 = float(args.eps)
    alpha_xy = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    phi_ls = _turek_obstacle_level_set(
        alpha_xy[:, 0],
        alpha_xy[:, 1],
        center=CENTER,
        radius=RADIUS,
        beam_length=BEAM_LENGTH,
        beam_height=BEAM_HEIGHT,
        root_inset=0.0,
    )
    alpha0 = _smooth_step((-phi_ls) / max(1.0e-12, eps0))
    alpha_n.nodal_values[:] = np.clip(np.asarray(alpha0, dtype=float), 0.0, 1.0)

    # Beam never detaches: pin alpha=1 at the beam root (arc on the cylinder boundary).
    beam_root_tag = "beam_root_alpha"
    n_root = _tag_beam_root_alpha_dirichlet_dofs(
        dh,
        mesh,
        tag=beam_root_tag,
        mesh_size=mesh_size,
        center=CENTER,
        beam_height=BEAM_HEIGHT,
    )
    root_dofs = np.asarray(sorted(dh.dof_tags[beam_root_tag]), dtype=int)
    alpha_n.set_nodal_values(root_dofs, np.ones(int(root_dofs.size), dtype=float))
    logging.info(f"[setup] pinned alpha=1 at beam root: {n_root} DOFs")

    phi_solid = float(args.phi_solid)
    if not (0.0 < phi_solid < 1.0):
        raise ValueError("--phi-solid must be in (0,1).")
    phi_n.nodal_values[:] = np.clip(1.0 - (1.0 - phi_solid) * np.asarray(alpha_n.nodal_values, dtype=float), 0.0, 1.0)

    # Freeze unused fields (no chemistry), and tie phi to alpha by manual updates.
    _mark_inactive_fields(dh, "phi", "S")

    # Pressure gauge: pin one pressure DOF instead of enforcing p=0 on the whole outlet.
    pressure_gauge_tag = "p_anchor"
    pressure_bc_key = str(args.pressure_bc).strip().lower()
    if pressure_bc_key == "point":
        pinned = _pin_pressure_gauge(dh, tag=pressure_gauge_tag)
        if pinned is None:
            raise RuntimeError("Failed to pin a pressure DOF for gauge fixing (--pressure-bc point).")
        logging.info(f"[setup] pinned pressure DOF for gauge fixing: {pinned}")

    # Restrict skeleton DOFs to a fixed rectangle around the obstacle.
    if bool(args.restrict_skeleton_dofs):
        pad = float(args.restrict_box_pad)
        x0_auto = float(CENTER[0] - RADIUS) - pad
        x1_auto = float(CENTER[0] + RADIUS + BEAM_LENGTH) + pad
        y0_auto = float(CENTER[1] - max(RADIUS, 0.5 * BEAM_HEIGHT)) - pad
        y1_auto = float(CENTER[1] + max(RADIUS, 0.5 * BEAM_HEIGHT)) + pad
        x0 = x0_auto if not np.isfinite(float(args.restrict_box_x0)) else float(args.restrict_box_x0)
        x1 = x1_auto if not np.isfinite(float(args.restrict_box_x1)) else float(args.restrict_box_x1)
        y0 = y0_auto if not np.isfinite(float(args.restrict_box_y0)) else float(args.restrict_box_y0)
        y1 = y1_auto if not np.isfinite(float(args.restrict_box_y1)) else float(args.restrict_box_y1)
        n_keep, n_tot = _restrict_skeleton_dofs_to_box(dh, x0=x0, x1=x1, y0=y0, y1=y1)
        logging.info(f"[setup] restricted (u,vS) DOFs to box: keep {n_keep}/{n_tot} Q2 nodes")

    # ------------------------------------------------------------------
    # Forms
    # ------------------------------------------------------------------
    u_mean = float(args.u_mean) if args.u_mean is not None else float(preset["u_mean"])
    rho_s = float(args.rho_s) if args.rho_s is not None else float(preset["rho_s"])
    dt_val = float(args.dt) if args.dt is not None else float(preset["dt"])
    theta = float(args.theta) if args.theta is not None else float(preset["theta"])
    dt_c = Constant(dt_val)

    rho_f = float(args.rho_f)
    mu_f = float(args.mu_f)
    u_max = 1.5 * u_mean
    kappa_inv = float(args.kappa_inv)

    E_s = float(args.E_s)
    nu_s = float(args.nu_s)
    if not (0.0 <= nu_s < 0.5):
        raise ValueError("--nu-s must be in [0,0.5).")
    mu_s = E_s / (2.0 * (1.0 + nu_s))
    lam_s = E_s * nu_s / ((1.0 + nu_s) * (1.0 - 2.0 * nu_s))

    # In this model, rho_S = rho_s0_tilde * alpha*(1-phi). With phi frozen and tied to alpha,
    # pick rho_s0_tilde so that rho_S ≈ rho_s inside the solid (alpha≈1, phi≈phi_solid).
    rho_s0_tilde = rho_s / max(1.0e-12, (1.0 - float(phi_solid)))

    alpha_ch_eps = float(args.alpha_ch_eps)
    if not np.isfinite(alpha_ch_eps):
        alpha_ch_eps = float(eps0)

    forms = build_biofilm_one_domain_forms(
        v_k=v_k,
        p_k=p_k,
        vS_k=vS_k,
        u_k=u_k,
        phi_k=phi_k,
        alpha_k=alpha_k,
        mu_alpha_k=mu_alpha_k,
        S_k=S_k,
        v_n=v_n,
        p_n=p_n,
        vS_n=vS_n,
        u_n=u_n,
        phi_n=phi_n,
        alpha_n=alpha_n,
        mu_alpha_n=mu_alpha_n,
        S_n=S_n,
        dv=dv,
        dp=dp,
        dvS=dvS,
        du=du,
        dphi=dphi,
        dalpha=dalpha,
        dmu_alpha=dmu_alpha,
        dS=dS_trial,
        v_test=v_test,
        q_test=q_test,
        vS_test=vS_test,
        u_test=u_test,
        phi_test=phi_test,
        alpha_test=alpha_test,
        mu_alpha_test=mu_alpha_test,
        S_test=S_test,
        dx=dx(metadata={"q": int(qdeg)}),
        dt=dt_c,
        theta=theta,
        rho_f=Constant(rho_f),
        mu_f=Constant(mu_f),
        kappa_inv=Constant(kappa_inv),
        mu_b_model="mu",
        solid_model=str(args.solid_model),
        mu_s=Constant(mu_s),
        lambda_s=Constant(lam_s),
        solid_visco_eta=0.0,
        rho_s0_tilde=Constant(float(rho_s0_tilde)),
        include_skeleton_acceleration=True,
        skeleton_inertia_convection=str(args.skeleton_inertia_convection),
        # Extension penalties (u,vS) in fluid region.
        gamma_u=float(args.gamma_u),
        u_extension_mode=str(args.u_extension),
        gamma_u_pin=float(args.gamma_u_pin),
        # Freeze transport fields (phi, S) externally; only solve alpha via CH.
        D_phi=0.0,
        gamma_phi=0.0,
        D_alpha=0.0,
        alpha_advection_form=str(args.alpha_advection_form),
        alpha_ch_M=float(args.alpha_ch_M),
        alpha_ch_gamma=float(args.alpha_ch_gamma),
        alpha_ch_eps=float(alpha_ch_eps),
        # Disable chemistry/growth/detachment/damage.
        D_S=0.0,
        mu_max=0.0,
        k_g=0.0,
        k_d=0.0,
        k_det=0.0,
        dim=2,
    )

    # ------------------------------------------------------------------
    # Boundary conditions (fluid)
    # ------------------------------------------------------------------
    def inflow_vx(_x, y, t):
        yv = float(y)
        v_base = float(u_max * 4.0 * yv * (float(H) - yv) / (float(H) ** 2))
        if t is None:
            return v_base
        tt = float(t)
        if tt < float(args.t_ramp):
            return v_base * 0.5 * (1.0 - math.cos(math.pi * tt / max(1.0e-12, float(args.t_ramp))))
        return v_base

    bcs = [
        BoundaryCondition("v_x", "dirichlet", "inlet", inflow_vx),
        BoundaryCondition("v_y", "dirichlet", "inlet", lambda x, y, t: 0.0),
        BoundaryCondition("v_x", "dirichlet", "walls", lambda x, y, t: 0.0),
        BoundaryCondition("v_y", "dirichlet", "walls", lambda x, y, t: 0.0),
        # No-slip on the rigid cylinder (hole boundary).
        BoundaryCondition("v_x", "dirichlet", "cylinder", lambda x, y, t: 0.0),
        BoundaryCondition("v_y", "dirichlet", "cylinder", lambda x, y, t: 0.0),
        # Clamp the beam root to the cylinder boundary.
        BoundaryCondition("u_x", "dirichlet", "cylinder", lambda x, y, t: 0.0),
        BoundaryCondition("u_y", "dirichlet", "cylinder", lambda x, y, t: 0.0),
        BoundaryCondition("vS_x", "dirichlet", "cylinder", lambda x, y, t: 0.0),
        BoundaryCondition("vS_y", "dirichlet", "cylinder", lambda x, y, t: 0.0),
        # Beam never detaches: keep alpha=1 at the root arc.
        BoundaryCondition("alpha", "dirichlet", beam_root_tag, lambda x, y, t: 1.0),
    ]
    if pressure_bc_key == "outlet":
        bcs.append(BoundaryCondition("p", "dirichlet", "outlet", lambda x, y, t: 0.0))
    elif pressure_bc_key == "point":
        bcs.append(BoundaryCondition("p", "dirichlet", pressure_gauge_tag, lambda x, y, t: 0.0))
    elif pressure_bc_key == "none":
        pass
    else:
        raise ValueError(f"Unknown --pressure-bc={args.pressure_bc!r}. Use 'point', 'outlet', or 'none'.")
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, (lambda x, y: 0.0)) for b in bcs]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    vtk_every = int(args.vtk_every)
    vtk_dir = out_dir / "vtk"
    if vtk_every > 0:
        vtk_dir.mkdir(parents=True, exist_ok=True)

    ts_path = out_dir / "timeseries.csv"
    with ts_path.open("w", encoding="utf-8") as f_ts:
        f_ts.write(
            "t,FD,FL,dp,CD,CL,FD_cyl,FL_cyl,CD_cyl,CL_cyl,FD_beam,FL_beam,CD_beam,CL_beam,tip_x,tip_y,tip_dx,tip_dy\n"
        )

    # Diagnostics helpers.
    u_interp = _interp_u_idw_factory(dh, u_k, active_only=True)
    fp_iters = 8
    ex = Constant(np.array([1.0, 0.0]), dim=1)
    ey = Constant(np.array([0.0, 1.0]), dim=1)
    beta_expr = Constant(float(mu_f)) * Constant(float(kappa_inv)) * alpha_k * (phi_k * phi_k)
    n = FacetNormal()
    sigma_f = Constant(float(mu_f)) * (grad(v_k) + grad(v_k).T) - p_k * Identity(2)
    traction_f = dot(sigma_f, n)
    d_gamma_cyl = dS(defined_on=mesh.edge_bitset("cylinder"), metadata={"q": int(qdeg)})

    def _post_step_update() -> None:
        # Tie frozen porosity to alpha after each accepted step.
        phi_k.nodal_values[:] = np.clip(1.0 - (1.0 - phi_solid) * np.asarray(alpha_k.nodal_values, dtype=float), 0.0, 1.0)
        # Keep frozen substrate non-negative.
        S_k.nodal_values[:] = np.maximum(np.asarray(S_k.nodal_values, dtype=float), 0.0)

    def _preassemble_cb(coeffs: dict[str, object]) -> None:
        """
        Keep frozen fields consistent with the current Newton iterate.

        This is intentionally a Picard-style update: we do not include d(phi)/d(alpha)
        contributions in the Jacobian because phi DOFs are marked inactive.
        """
        a = coeffs.get("alpha_k", None)
        p = coeffs.get("phi_k", None)
        if getattr(a, "nodal_values", None) is None or getattr(p, "nodal_values", None) is None:
            return
        p.nodal_values[:] = np.clip(1.0 - (1.0 - phi_solid) * np.asarray(a.nodal_values, dtype=float), 0.0, 1.0)

    def _compute_observables(t_now: float) -> None:
        # Tip position via fixed-point solve of x = X_ref + u(x).
        x_ref = np.asarray(TIP_REF, dtype=float)
        x_cur = np.asarray(x_ref, dtype=float).copy()
        for _ in range(int(fp_iters)):
            u_val = u_interp(x_cur, k=12, power=2.0).reshape(2)
            x_cur = x_ref + u_val
        tip = np.asarray(x_cur, dtype=float)
        tip_d = tip - x_ref

        # Hydrodynamic forces:
        #   - cylinder: traction integral (force on cylinder is minus force on fluid)
        #   - beam: Brinkman penalization reaction
        diff_v = v_k - vS_k

        drag_int_beam = dot(beta_expr * diff_v, ex) * dx(metadata={"q": int(qdeg)})
        lift_int_beam = dot(beta_expr * diff_v, ey) * dx(metadata={"q": int(qdeg)})
        drag_int_cyl = (-dot(traction_f, ex)) * d_gamma_cyl
        lift_int_cyl = (-dot(traction_f, ey)) * d_gamma_cyl

        hooks = {
            drag_int_cyl.integrand: {"name": "FD_cyl"},
            lift_int_cyl.integrand: {"name": "FL_cyl"},
            drag_int_beam.integrand: {"name": "FD_beam"},
            lift_int_beam.integrand: {"name": "FL_beam"},
        }
        res = assemble_form(
            Equation(None, drag_int_cyl + lift_int_cyl + drag_int_beam + lift_int_beam),
            dof_handler=dh,
            bcs=[],
            assembler_hooks=hooks,
            backend=backend,
        )
        F_D_cyl = float(np.asarray(res.get("FD_cyl", 0.0), dtype=float).reshape(()))
        F_L_cyl = float(np.asarray(res.get("FL_cyl", 0.0), dtype=float).reshape(()))
        F_D_beam = float(np.asarray(res.get("FD_beam", 0.0), dtype=float).reshape(()))
        F_L_beam = float(np.asarray(res.get("FL_beam", 0.0), dtype=float).reshape(()))
        F_D = float(F_D_cyl + F_D_beam)
        F_L = float(F_L_cyl + F_L_beam)

        # Pressure drop: dp = p(A) - p(B) at fixed probes.
        def _nearest_pressure(pt: np.ndarray) -> float:
            xy = np.asarray(dh.get_dof_coords("p"), dtype=float)
            vals = np.asarray(p_k.nodal_values, dtype=float).ravel()
            if xy.ndim != 2 or xy.shape[1] != 2 or vals.size != xy.shape[0]:
                raise RuntimeError("Unexpected pressure DOF coord/value mismatch.")
            d2 = np.sum((xy - np.asarray(pt, dtype=float).reshape(1, 2)) ** 2, axis=1)
            j = int(np.argmin(d2))
            return float(vals[j])

        pA = _nearest_pressure(PROBE_A_REF)
        pB = _nearest_pressure(PROBE_B_REF)
        dp = float(pA - pB)

        coeff = 2.0 / (float(rho_f) * float(u_mean) ** 2 * float(D_CYL))
        C_D = float(coeff) * F_D
        C_L = float(coeff) * F_L
        C_D_cyl = float(coeff) * F_D_cyl
        C_L_cyl = float(coeff) * F_L_cyl
        C_D_beam = float(coeff) * F_D_beam
        C_L_beam = float(coeff) * F_L_beam

        with ts_path.open("a", encoding="utf-8") as f_ts:
            f_ts.write(
                f"{float(t_now):.12e},{F_D:.12e},{F_L:.12e},{dp:.12e},{C_D:.12e},{C_L:.12e},"
                f"{F_D_cyl:.12e},{F_L_cyl:.12e},{C_D_cyl:.12e},{C_L_cyl:.12e},"
                f"{F_D_beam:.12e},{F_L_beam:.12e},{C_D_beam:.12e},{C_L_beam:.12e},"
                f"{tip[0]:.12e},{tip[1]:.12e},{tip_d[0]:.12e},{tip_d[1]:.12e}\n"
            )
        logging.info(
            f"[obs] t={t_now:.3f}  FD={F_D:.4e}  FL={F_L:.4e}  dp={dp:.4e}  CD={C_D:.4e}  CL={C_L:.4e}  "
            f"tip=({tip[0]:.5f},{tip[1]:.5f})  d=({tip_d[0]:.4e},{tip_d[1]:.4e})"
        )

    def post_step_refiner(step, bcs_now, functions, prev_functions):
        # Called after an accepted Newton step and **before** promotion.
        _post_step_update()
        step_no = int(getattr(solver, "_current_step_no", int(step)))
        t_now = float(getattr(solver, "_current_t", 0.0) + getattr(solver, "_current_dt", dt_val))
        _compute_observables(t_now)
        if vtk_every > 0 and (step_no % vtk_every == 0):
            export_vtk(
                str(vtk_dir / f"step={step_no:04d}.vtu"),
                mesh,
                dh,
                {"v": v_k, "p": p_k, "vS": vS_k, "u": u_k, "phi": phi_k, "alpha": alpha_k, "mu_alpha": mu_alpha_k},
            )

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    newton_solver_key = str(args.newton_solver).strip().lower()
    if newton_solver_key == "snes":
        petsc_opts = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_atol": float(args.newton_tol),
            "snes_rtol": 0.0,
            "snes_max_it": int(args.max_it),
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
        if bool(args.petsc_monitor) or os.getenv("PYCUTFEM_PETSC_MONITOR", "").strip().lower() in {"1", "true", "yes"}:
            petsc_opts.setdefault("snes_monitor", None)
            petsc_opts.setdefault("snes_converged_reason", None)
            petsc_opts.setdefault("ksp_converged_reason", None)
            petsc_opts.setdefault("ksp_monitor_short", None)
        if bool(args.alpha_box_constraints):
            # Bound constraints require a VI-capable SNES type.
            # `newtonls` does not support bounds (PETSc error 73).
            petsc_opts["snes_type"] = "vinewtonrsls"
        solver = PetscSnesNewtonSolver(
            forms.residual_form,
            forms.jacobian_form,
            dof_handler=dh,
            mixed_element=me,
            bcs=bcs,
            bcs_homog=bcs_homog,
            newton_params=NewtonParameters(
                newton_tol=float(args.newton_tol),
                max_newton_iter=int(args.max_it),
                accept_nonconverged_atol_factor=float(args.snes_accept_factor),
            ),
            quad_order=int(qdeg),
            backend=backend,
            petsc_options=petsc_opts,
            preassemble_cb=_preassemble_cb,
        )
    else:
        vi_params = VIParameters(project_initial_guess=True, project_each_iteration=True)
        solver = PdasNewtonSolver(
            forms.residual_form,
            forms.jacobian_form,
            dof_handler=dh,
            mixed_element=me,
            bcs=bcs,
            bcs_homog=bcs_homog,
            newton_params=NewtonParameters(newton_tol=float(args.newton_tol), max_newton_iter=int(args.max_it)),
            vi_params=vi_params,
            quad_order=int(qdeg),
            backend=backend,
            preassemble_cb=_preassemble_cb,
        )

    if bool(args.alpha_box_constraints):
        # Supported by both PDAS and PETSc SNES backends.
        solver.set_box_bounds(by_field={"alpha": (0.0, 1.0)})

    # Bootstrap k-state.
    for fk, fn in (
        (v_k, v_n),
        (p_k, p_n),
        (vS_k, vS_n),
        (u_k, u_n),
        (phi_k, phi_n),
        (alpha_k, alpha_n),
        (mu_alpha_k, mu_alpha_n),
        (S_k, S_n),
    ):
        fk.nodal_values[:] = np.asarray(fn.nodal_values, dtype=float)

    functions_k = [v_k, p_k, vS_k, u_k, phi_k, alpha_k, mu_alpha_k, S_k]
    functions_n = [v_n, p_n, vS_n, u_n, phi_n, alpha_n, mu_alpha_n, S_n]

    def _on_dt_change(new_dt: float) -> None:
        dt_c.value = float(new_dt)

    solver.solve_time_interval(
        functions=functions_k,
        prev_functions=functions_n,
        aux_functions={"dt": dt_c},
        time_params=TimeStepperParameters(
            dt=dt_val,
            final_time=float(args.t_final),
            max_steps=200_000,
            theta=theta,
            allow_dt_reduction=bool(args.allow_dt_reduction),
            dt_min=float(args.dt_min),
            dt_reduction_factor=float(args.dt_reduction_factor),
            on_dt_change=_on_dt_change,
        ),
        post_step_refiner=post_step_refiner,
    )


if __name__ == "__main__":
    main()
