#!/usr/bin/env python3
"""Sharp two-domain Stoter Section 5.3 reference benchmark.

This driver builds a boundary-fitted Stokes/Darcy reference for the 2D channel
flow intercepted by a porous medium from Stoter et al., CMAME 321 (2017),
Section 5.3. It uses:

  - Q1/Q1 Stokes with LSIC stabilization on the sharp Stokes subdomain,
  - Q1 Darcy head on the sharp Darcy subdomain,
  - explicit sharp interface coupling on the shared boundary:
      * mass conservation,
      * normal traction balance,
      * BJS tangential slip.

The geometry is the same as the diffuse benchmark driver in this folder so the
outputs can be compared pointwise on the same sampling grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import scipy.sparse.linalg as spla

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.nonmatching import build_composite_mesh
from pycutfem.nonmatching.interface import NonMatchingInterface
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver
from pycutfem.ufl.expressions import (
    Constant,
    FacetNormal,
    Function,
    Neg,
    Pos,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    div,
    dot,
    grad,
    inner,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dNonmatchingInterface, dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.bitset import BitSet
from pycutfem.utils.meshgen import structured_quad


def _eps(v):
    return Constant(0.5) * (grad(v) + grad(v).T)


@dataclass(frozen=True)
class Geometry:
    Lx: float = 80.0
    Ly: float = 100.0
    channel_x0: float = 25.0
    channel_x1: float = 55.0
    lower_interface_y: float = 30.0
    upper_interface_y: float = 70.0
    r_in: float = 8.5

    @property
    def center_x(self) -> float:
        return 0.5 * (self.channel_x0 + self.channel_x1)


def _segment_coords(a: float, b: float, h: float) -> list[float]:
    a_f = float(a)
    b_f = float(b)
    h_f = float(h)
    if b_f <= a_f:
        return [a_f]
    n = max(1, int(math.ceil((b_f - a_f) / h_f)))
    return [a_f + (b_f - a_f) * i / n for i in range(n)] + [b_f]


def _merge_coords(parts: list[list[float]], *, tol: float = 1.0e-12) -> np.ndarray:
    vals: list[float] = []
    for part in parts:
        for v in part:
            vv = float(v)
            if not vals or abs(vv - vals[-1]) > float(tol):
                vals.append(vv)
    return np.asarray(vals, dtype=float)


def _build_tensor_quad_mesh(x_coords: np.ndarray, y_coords: np.ndarray) -> tuple[list, np.ndarray, np.ndarray]:
    from pycutfem.core.topology import Node

    xs = np.asarray(x_coords, dtype=float)
    ys = np.asarray(y_coords, dtype=float)
    nx = int(xs.size) - 1
    ny = int(ys.size) - 1
    nodes = []
    node_id = 0
    for j in range(ys.size):
        for i in range(xs.size):
            nodes.append(Node(node_id, float(xs[i]), float(ys[j])))
            node_id += 1

    def nid(i: int, j: int) -> int:
        return int(j) * int(xs.size) + int(i)

    elems = []
    corners = []
    for j in range(ny):
        for i in range(nx):
            bl = nid(i, j)
            br = nid(i + 1, j)
            tl = nid(i, j + 1)
            tr = nid(i + 1, j + 1)
            elems.append([bl, br, tl, tr])
            corners.append([bl, br, tr, tl])
    return nodes, np.asarray(elems, dtype=int), np.asarray(corners, dtype=int)


def _is_stokes_region(geom: Geometry, x: float, y: float, *, tol: float = 1.0e-12) -> bool:
    xx = float(x)
    yy = float(y)
    in_x = float(geom.channel_x0) - tol <= xx <= float(geom.channel_x1) + tol
    in_lower = -tol <= yy <= float(geom.lower_interface_y) + tol
    in_upper = float(geom.upper_interface_y) - tol <= yy <= float(geom.Ly) + tol
    return bool(in_x and (in_lower or in_upper))


def _validate_alignment(geom: Geometry, *, hx: float, hy: float, tol: float = 1.0e-10) -> None:
    anchors_x = [0.0, float(geom.channel_x0), float(geom.channel_x1), float(geom.Lx)]
    anchors_y = [0.0, float(geom.lower_interface_y), float(geom.upper_interface_y), float(geom.Ly)]
    for val in anchors_x:
        idx = val / float(hx)
        if abs(idx - round(idx)) > float(tol):
            raise ValueError(
                f"Sharp Stoter reference requires x-geometry alignment to the grid. "
                f"Got x={val} with hx={hx}."
            )
    for val in anchors_y:
        idx = val / float(hy)
        if abs(idx - round(idx)) > float(tol):
            raise ValueError(
                f"Sharp Stoter reference requires y-geometry alignment to the grid. "
                f"Got y={val} with hy={hy}."
            )


def _extract_submesh(
    *,
    nodes,
    elems: np.ndarray,
    corners: np.ndarray,
    elem_mask: np.ndarray,
) -> Mesh:
    elems_sel = np.asarray(elems, dtype=int)[np.asarray(elem_mask, dtype=bool)]
    corners_sel = np.asarray(corners, dtype=int)[np.asarray(elem_mask, dtype=bool)]
    used = np.unique(elems_sel.reshape(-1))
    remap = -np.ones((len(nodes),), dtype=int)
    remap[used] = np.arange(int(used.size), dtype=int)
    nodes_sub = [nodes[int(i)] for i in used.tolist()]
    elems_sub = remap[elems_sel]
    corners_sub = remap[corners_sel]
    mesh = Mesh(
        nodes_sub,
        elems_sub,
        elements_corner_nodes=corners_sub,
        element_type="quad",
        poly_order=1,
    )
    if hasattr(mesh, "build_grid_search"):
        mesh.build_grid_search()
    return mesh


def _make_body_fitted_meshes(*, geom: Geometry, nx: int, ny: int, mesh_mode: str) -> tuple[Mesh, Mesh, float, float]:
    mesh_mode_key = str(mesh_mode).strip().lower()
    if mesh_mode_key == "uniform":
        nodes, elems, _, corners = structured_quad(
            float(geom.Lx),
            float(geom.Ly),
            nx=int(nx),
            ny=int(ny),
            poly_order=1,
            offset=(0.0, 0.0),
        )
        hx = float(geom.Lx) / float(nx)
        hy = float(geom.Ly) / float(ny)
    elif mesh_mode_key == "refined":
        hx_base = float(geom.Lx) / float(nx)
        hy_base = float(geom.Ly) / float(ny)
        hx_fine = 0.5 * hx_base
        hy_fine = 0.5 * hy_base
        xs = _merge_coords(
            [
                _segment_coords(0.0, float(geom.channel_x0), hx_base),
                _segment_coords(float(geom.channel_x0), float(geom.channel_x1), hx_fine),
                _segment_coords(float(geom.channel_x1), float(geom.Lx), hx_base),
            ]
        )
        ys = _merge_coords(
            [
                _segment_coords(0.0, float(geom.lower_interface_y), hy_fine),
                _segment_coords(float(geom.lower_interface_y), float(geom.upper_interface_y), hy_base),
                _segment_coords(float(geom.upper_interface_y), float(geom.Ly), hy_fine),
            ]
        )
        nodes, elems, corners = _build_tensor_quad_mesh(xs, ys)
        hx = float(np.min(np.diff(xs)))
        hy = float(np.min(np.diff(ys)))
    else:
        raise ValueError("mesh_mode must be 'uniform' or 'refined'.")

    elems = np.asarray(elems, dtype=int)
    corners = np.asarray(corners, dtype=int)
    mesh_full = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=1)
    _validate_alignment(geom, hx=hx, hy=hy)

    elem_centers = np.mean(np.asarray(mesh_full.nodes_x_y_pos, dtype=float)[corners], axis=1)
    stokes_mask = np.asarray(
        [_is_stokes_region(geom, float(xc), float(yc)) for xc, yc in elem_centers],
        dtype=bool,
    )
    darcy_mask = ~stokes_mask

    mesh_s = _extract_submesh(nodes=nodes, elems=elems, corners=corners, elem_mask=stokes_mask)
    mesh_d = _extract_submesh(nodes=nodes, elems=elems, corners=corners, elem_mask=darcy_mask)
    return mesh_s, mesh_d, hx, hy


def _tag_stokes_boundaries(mesh: Mesh, geom: Geometry, tol: float = 1.0e-12) -> None:
    cx = float(geom.center_x)
    rin = float(geom.r_in)

    def _inlet_active(x, y):
        return abs(y - 0.0) <= tol and abs(x - cx) <= rin + tol

    def _inlet_rest(x, y):
        return abs(y - 0.0) <= tol and not _inlet_active(x, y)

    def _outlet(x, y):
        return abs(y - float(geom.Ly)) <= tol

    def _interface(x, y):
        return not (_inlet_active(x, y) or _inlet_rest(x, y) or _outlet(x, y))

    mesh.tag_boundary_edges(
        {
            "bottom_active": _inlet_active,
            "bottom_rest": _inlet_rest,
            "outlet": _outlet,
            "interface": _interface,
        }
    )


def _tag_darcy_boundaries(mesh: Mesh, geom: Geometry, tol: float = 1.0e-12) -> None:
    def _outer(x, y):
        return (
            abs(x - 0.0) <= tol
            or abs(x - float(geom.Lx)) <= tol
            or abs(y - 0.0) <= tol
            or abs(y - float(geom.Ly)) <= tol
        )

    def _interface(x, y):
        return not _outer(x, y)

    mesh.tag_boundary_edges({"outer": _outer, "interface": _interface})


def _bitset_from_ids(mesh: Mesh, ids: np.ndarray) -> BitSet:
    mask = np.zeros((int(mesh.n_elements),), dtype=bool)
    mask[np.asarray(ids, dtype=int)] = True
    return BitSet(mask)


def _edge_key_from_points(p0: np.ndarray, p1: np.ndarray, *, ndigits: int = 12) -> tuple[tuple[float, float], tuple[float, float]]:
    a = (round(float(p0[0]), ndigits), round(float(p0[1]), ndigits))
    b = (round(float(p1[0]), ndigits), round(float(p1[1]), ndigits))
    return tuple(sorted((a, b)))


def _build_matching_interface_from_aligned_edges(
    *,
    mesh_neg: Mesh,
    mesh_pos: Mesh,
    neg_tag: str = "interface",
    pos_tag: str = "interface",
) -> NonMatchingInterface:
    pos_by_key: dict[tuple[tuple[float, float], tuple[float, float]], tuple[int, object]] = {}
    for gid in mesh_pos.edge_bitset(str(pos_tag)).to_indices().tolist():
        edge = mesh_pos.edge(int(gid))
        p0 = np.asarray(mesh_pos.nodes_x_y_pos[int(edge.nodes[0])], dtype=float)
        p1 = np.asarray(mesh_pos.nodes_x_y_pos[int(edge.nodes[1])], dtype=float)
        pos_by_key[_edge_key_from_points(p0, p1)] = (int(gid), edge)

    neg_edge_ids: list[int] = []
    pos_edge_ids: list[int] = []
    neg_elem_ids: list[int] = []
    pos_elem_ids: list[int] = []
    P0: list[np.ndarray] = []
    P1: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    h_neg: list[float] = []
    h_pos: list[float] = []

    for gid in mesh_neg.edge_bitset(str(neg_tag)).to_indices().tolist():
        edge_neg = mesh_neg.edge(int(gid))
        p0 = np.asarray(mesh_neg.nodes_x_y_pos[int(edge_neg.nodes[0])], dtype=float)
        p1 = np.asarray(mesh_neg.nodes_x_y_pos[int(edge_neg.nodes[1])], dtype=float)
        key = _edge_key_from_points(p0, p1)
        if key not in pos_by_key:
            raise ValueError(f"Failed to find matching Darcy interface edge for Stokes edge key={key!r}.")
        gid_pos, edge_pos = pos_by_key[key]
        if edge_neg.left is None or edge_pos.left is None:
            raise ValueError("Interface edges must have owning elements on both submeshes.")
        n_vec = np.asarray(getattr(edge_neg, "normal", None), dtype=float)
        if n_vec.shape != (2,):
            t = p1 - p0
            tnorm = float(np.linalg.norm(t))
            if tnorm <= 1.0e-14:
                raise ValueError("Degenerate interface edge encountered in sharp Stoter benchmark.")
            n_vec = np.array([t[1], -t[0]], dtype=float) / tnorm
        c_neg = np.asarray(mesh_neg.elements_list[int(edge_neg.left)].centroid(), dtype=float)
        c_pos = np.asarray(mesh_pos.elements_list[int(edge_pos.left)].centroid(), dtype=float)
        if float(np.dot(n_vec, c_pos - c_neg)) < 0.0:
            n_vec = -n_vec
        nn = float(np.linalg.norm(n_vec))
        if nn <= 1.0e-14:
            raise ValueError("Zero interface normal encountered in sharp Stoter benchmark.")
        n_vec = n_vec / nn

        neg_edge_ids.append(int(gid))
        pos_edge_ids.append(int(gid_pos))
        neg_elem_ids.append(int(edge_neg.left))
        pos_elem_ids.append(int(edge_pos.left))
        P0.append(p0)
        P1.append(p1)
        normals.append(n_vec)
        h_neg.append(float(mesh_neg.element_char_length(int(edge_neg.left)) or 0.0))
        h_pos.append(float(mesh_pos.element_char_length(int(edge_pos.left)) or 0.0))

    if not P0:
        raise ValueError("No aligned interface edges found for the sharp Stoter benchmark.")

    return NonMatchingInterface(
        mesh_neg=mesh_neg,
        mesh_pos=mesh_pos,
        neg_edge_ids=np.asarray(neg_edge_ids, dtype=int),
        pos_edge_ids=np.asarray(pos_edge_ids, dtype=int),
        neg_elem_ids=np.asarray(neg_elem_ids, dtype=int),
        pos_elem_ids=np.asarray(pos_elem_ids, dtype=int),
        P0=np.asarray(P0, dtype=float),
        P1=np.asarray(P1, dtype=float),
        n=np.asarray(normals, dtype=float),
        h_neg=np.asarray(h_neg, dtype=float),
        h_pos=np.asarray(h_pos, dtype=float),
    )


def _eval_scalar_at_point(dh: DofHandler, mesh: Mesh, field_name: str, func, point: tuple[float, float]) -> float:
    from pycutfem.fem import transform

    xy = np.asarray(point, dtype=float)
    for elem in mesh.elements_list:
        verts = mesh.nodes_x_y_pos[list(elem.nodes)]
        if not (
            verts[:, 0].min() - 1.0e-12 <= xy[0] <= verts[:, 0].max() + 1.0e-12
            and verts[:, 1].min() - 1.0e-12 <= xy[1] <= verts[:, 1].max() + 1.0e-12
        ):
            continue
        try:
            xi, eta = transform.inverse_mapping(mesh, elem.id, xy)
        except Exception:
            continue
        if not (-1.0001 <= xi <= 1.0001 and -1.0001 <= eta <= 1.0001):
            continue
        phi = dh.mixed_element.basis(field_name, float(xi), float(eta))[dh.mixed_element.slice(field_name)]
        gdofs = np.asarray(dh.element_maps[field_name][elem.id], dtype=int)
        vals = np.asarray(func.get_nodal_values(gdofs), dtype=float)
        return float(np.asarray(phi, dtype=float) @ vals)
    raise RuntimeError(f"Failed to locate point {point} in the sharp Stoter mesh.")


def _eval_scalar_grad_at_point(dh: DofHandler, mesh: Mesh, field_name: str, func, point: tuple[float, float]) -> np.ndarray:
    from pycutfem.fem import transform

    xy = np.asarray(point, dtype=float)
    for elem in mesh.elements_list:
        verts = mesh.nodes_x_y_pos[list(elem.nodes)]
        if not (
            verts[:, 0].min() - 1.0e-12 <= xy[0] <= verts[:, 0].max() + 1.0e-12
            and verts[:, 1].min() - 1.0e-12 <= xy[1] <= verts[:, 1].max() + 1.0e-12
        ):
            continue
        try:
            xi, eta = transform.inverse_mapping(mesh, elem.id, xy)
        except Exception:
            continue
        if not (-1.0001 <= xi <= 1.0001 and -1.0001 <= eta <= 1.0001):
            continue
        grad_ref = dh.mixed_element.grad_basis(field_name, float(xi), float(eta))[dh.mixed_element.slice(field_name), :]
        grad_phys = np.asarray(transform.map_grad_scalar(mesh, elem.id, grad_ref, (float(xi), float(eta))), dtype=float)
        gdofs = np.asarray(dh.element_maps[field_name][elem.id], dtype=int)
        coeffs = np.asarray(func.get_nodal_values(gdofs), dtype=float)
        return coeffs @ grad_phys
    raise RuntimeError(f"Failed to locate point {point} in the sharp Stoter mesh.")


def _write_centerline_samples(outdir: Path, dh: DofHandler, mesh: Mesh, geom: Geometry, ux_s, uy_s, phi_d, K: float) -> Path:
    xs = np.full(201, float(geom.center_x))
    ys = np.linspace(0.0, float(geom.Ly), 201)
    rows = ["y_mm,ux_stokes,uy_stokes,ux_combined,uy_combined,phi_D"]
    for x, y in zip(xs, ys):
        if _is_stokes_region(geom, float(x), float(y)):
            ux_h = _eval_scalar_at_point(dh, mesh, "ux_s", ux_s, (float(x), float(y)))
            uy_h = _eval_scalar_at_point(dh, mesh, "uy_s", uy_s, (float(x), float(y)))
            ux_c = ux_h
            uy_c = uy_h
            phi_h = 0.0
        else:
            grad_phi = _eval_scalar_grad_at_point(dh, mesh, "phi_d", phi_d, (float(x), float(y)))
            phi_h = _eval_scalar_at_point(dh, mesh, "phi_d", phi_d, (float(x), float(y)))
            ux_h = 0.0
            uy_h = 0.0
            ux_c = -float(K) * float(grad_phi[0])
            uy_c = -float(K) * float(grad_phi[1])
        rows.append(f"{float(y):.12e},{ux_h:.12e},{uy_h:.12e},{ux_c:.12e},{uy_c:.12e},{phi_h:.12e}")
    path = outdir / "centerline.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _write_velocity_grid_samples(
    outdir: Path,
    dh: DofHandler,
    mesh: Mesh,
    geom: Geometry,
    ux_s,
    uy_s,
    phi_d,
    K: float,
    nx: int = 41,
    ny: int = 51,
) -> Path:
    xs = np.linspace(0.0, float(geom.Lx), int(nx))
    ys = np.linspace(0.0, float(geom.Ly), int(ny))
    rows = ["x_mm,y_mm,is_stokes,ux_stokes,uy_stokes,ux_combined,uy_combined,phi_D"]
    for y in ys:
        for x in xs:
            if _is_stokes_region(geom, float(x), float(y)):
                ux_h = _eval_scalar_at_point(dh, mesh, "ux_s", ux_s, (float(x), float(y)))
                uy_h = _eval_scalar_at_point(dh, mesh, "uy_s", uy_s, (float(x), float(y)))
                ux_c = ux_h
                uy_c = uy_h
                phi_h = 0.0
                is_stokes = 1
            else:
                grad_phi = _eval_scalar_grad_at_point(dh, mesh, "phi_d", phi_d, (float(x), float(y)))
                phi_h = _eval_scalar_at_point(dh, mesh, "phi_d", phi_d, (float(x), float(y)))
                ux_h = 0.0
                uy_h = 0.0
                ux_c = -float(K) * float(grad_phi[0])
                uy_c = -float(K) * float(grad_phi[1])
                is_stokes = 0
            rows.append(
                f"{float(x):.12e},{float(y):.12e},{int(is_stokes)},{ux_h:.12e},{uy_h:.12e},{ux_c:.12e},{uy_c:.12e},{phi_h:.12e}"
            )
    path = outdir / "velocity_grid.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _compare_velocity_grids(reference_csv: Path, candidate_csv: Path) -> dict[str, float]:
    def _load(path: Path) -> dict[tuple[float, float], tuple[float, float]]:
        out: dict[tuple[float, float], tuple[float, float]] = {}
        with path.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                key = (round(float(row["x_mm"]), 12), round(float(row["y_mm"]), 12))
                out[key] = (float(row["ux_combined"]), float(row["uy_combined"]))
        return out

    ref = _load(reference_csv)
    cand = _load(candidate_csv)
    common = sorted(set(ref.keys()) & set(cand.keys()))
    if not common:
        raise ValueError("No common sample points between the one-domain and two-domain velocity grids.")
    diffs = []
    mags = []
    for key in common:
        ux_r, uy_r = ref[key]
        ux_c, uy_c = cand[key]
        diffs.append((ux_c - ux_r) ** 2 + (uy_c - uy_r) ** 2)
        mags.append(ux_r * ux_r + uy_r * uy_r)
    rmse = math.sqrt(float(np.mean(np.asarray(diffs, dtype=float))))
    ref_rms = math.sqrt(float(np.mean(np.asarray(mags, dtype=float))))
    return {
        "grid_common_points": float(len(common)),
        "grid_rmse_velocity": float(rmse),
        "grid_rel_rmse_velocity": float(rmse / max(ref_rms, 1.0e-14)),
    }


def _write_velocity_plots(outdir: Path, velocity_grid_csv: Path, geom: Geometry, *, umax: float) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(Path(velocity_grid_csv).open("r", encoding="utf-8")))
    xs = sorted({float(row["x_mm"]) for row in rows})
    ys = sorted({float(row["y_mm"]) for row in rows})
    nx = len(xs)
    ny = len(ys)
    x_idx = {x: i for i, x in enumerate(xs)}
    y_idx = {y: j for j, y in enumerate(ys)}
    X = np.asarray(xs, dtype=float)
    Y = np.asarray(ys, dtype=float)
    UX = np.zeros((ny, nx), dtype=float)
    UY = np.zeros((ny, nx), dtype=float)
    for row in rows:
        i = x_idx[float(row["x_mm"])]
        j = y_idx[float(row["y_mm"])]
        UX[j, i] = float(row["ux_combined"])
        UY[j, i] = float(row["uy_combined"])
    XX, YY = np.meshgrid(X, Y)
    SPEED = np.sqrt(UX * UX + UY * UY)

    def _draw_geometry(ax):
        x0 = float(geom.channel_x0)
        x1 = float(geom.channel_x1)
        y0 = float(geom.lower_interface_y)
        y1 = float(geom.upper_interface_y)
        ax.plot([x0, x1, x1, x0, x0], [0.0, 0.0, y0, y0, 0.0], color="k", lw=1.0)
        ax.plot([x0, x1, x1, x0, x0], [y1, y1, float(geom.Ly), float(geom.Ly), y1], color="k", lw=1.0)
        ax.plot([0.0, float(geom.Lx), float(geom.Lx), 0.0, 0.0], [0.0, 0.0, float(geom.Ly), float(geom.Ly), 0.0], color="0.3", lw=0.8)

    contour_path = outdir / "speed_contour.png"
    fig, ax = plt.subplots(figsize=(6.4, 8.0), constrained_layout=True)
    cf = ax.contourf(XX, YY, SPEED, levels=np.linspace(0.0, float(umax), 21), cmap="viridis", vmin=0.0, vmax=float(umax))
    _draw_geometry(ax)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, float(geom.Lx))
    ax.set_ylim(0.0, float(geom.Ly))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Sharp Two-Domain Speed")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("|u| [mm/s]")
    fig.savefig(contour_path, dpi=200)
    plt.close(fig)

    quiver_path = outdir / "quiver.png"
    stride_x = max(1, nx // 20)
    stride_y = max(1, ny // 25)
    fig, ax = plt.subplots(figsize=(6.4, 8.0), constrained_layout=True)
    bg = ax.contourf(XX, YY, SPEED, levels=np.linspace(0.0, float(umax), 21), cmap="viridis", vmin=0.0, vmax=float(umax), alpha=0.9)
    ax.quiver(
        XX[::stride_y, ::stride_x],
        YY[::stride_y, ::stride_x],
        UX[::stride_y, ::stride_x],
        UY[::stride_y, ::stride_x],
        color="white",
        angles="xy",
        scale_units="xy",
        scale=max(float(umax), 1.0) / 6.0,
        width=0.0035,
        headwidth=3.2,
        headlength=4.2,
    )
    _draw_geometry(ax)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, float(geom.Lx))
    ax.set_ylim(0.0, float(geom.Ly))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Sharp Two-Domain Velocity")
    cbar = fig.colorbar(bg, ax=ax)
    cbar.set_label("|u| [mm/s]")
    fig.savefig(quiver_path, dpi=200)
    plt.close(fig)

    return {
        "speed_contour_png": str(contour_path),
        "quiver_png": str(quiver_path),
    }


def solve_benchmark(
    *,
    nx: int,
    ny: int,
    Umax: float,
    K: float,
    nu: float,
    friction_alpha: float,
    g: float,
    lsic_scale: float,
    mesh_mode: str,
    backend: str,
    export: bool,
    outdir: Path,
    compare_one_domain_csv: Path | None,
) -> dict[str, object]:
    geom = Geometry()
    mesh_s, mesh_d, hx, hy = _make_body_fitted_meshes(geom=geom, nx=int(nx), ny=int(ny), mesh_mode=str(mesh_mode))
    _tag_stokes_boundaries(mesh_s, geom)
    _tag_darcy_boundaries(mesh_d, geom)

    iface = _build_matching_interface_from_aligned_edges(mesh_neg=mesh_s, mesh_pos=mesh_d, neg_tag="interface", pos_tag="interface")
    mapping = build_composite_mesh(mesh_pos=mesh_d, mesh_neg=mesh_s, order="pos_neg")
    mesh = mapping.mesh

    iface_c = NonMatchingInterface(
        mesh_neg=mesh,
        mesh_pos=mesh,
        neg_edge_ids=np.zeros((iface.n_segments(),), dtype=int),
        pos_edge_ids=np.zeros((iface.n_segments(),), dtype=int),
        neg_elem_ids=np.asarray(iface.neg_elem_ids, dtype=int) + int(mapping.neg_elem_offset),
        pos_elem_ids=np.asarray(iface.pos_elem_ids, dtype=int) + int(mapping.pos_elem_offset),
        P0=np.asarray(iface.P0, dtype=float),
        P1=np.asarray(iface.P1, dtype=float),
        n=np.asarray(iface.n, dtype=float),
        h_neg=np.asarray(iface.h_neg, dtype=float),
        h_pos=np.asarray(iface.h_pos, dtype=float),
    )

    bs_s = _bitset_from_ids(mesh, np.asarray(mapping.neg_elem_ids, dtype=int))
    bs_d = _bitset_from_ids(mesh, np.asarray(mapping.pos_elem_ids, dtype=int))
    qdeg = 6
    dx_s = dx(defined_on=bs_s, metadata={"q": int(qdeg)})
    dx_d = dx(defined_on=bs_d, metadata={"q": int(qdeg)})
    dGamma = dNonmatchingInterface(metadata={"q": int(qdeg) + 2, "interface": iface_c})

    me = MixedElement(
        mesh,
        field_specs={
            "ux_s": 1,
            "uy_s": 1,
            "p_s": 1,
            "phi_d": 1,
        },
    )
    dh = DofHandler(me, method="cg")
    dh.dof_tags["inactive"] = set()
    pos_mask = np.zeros((int(mesh.n_elements),), dtype=bool)
    pos_mask[np.asarray(mapping.pos_elem_ids, dtype=int)] = True
    neg_mask = np.zeros((int(mesh.n_elements),), dtype=bool)
    neg_mask[np.asarray(mapping.neg_elem_ids, dtype=int)] = True
    for fld in ("ux_s", "uy_s", "p_s"):
        dh.tag_dofs_from_element_bitset("inactive", fld, pos_mask, strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "phi_d", neg_mask, strict=True)

    h = min(float(hx), float(hy))
    tol_pin = 0.51 * h
    dh.tag_dofs_by_locator_map(
        {
            "bottom_active": lambda x, y: abs(y - 0.0) <= 1.0e-12 and abs(x - float(geom.center_x)) <= float(geom.r_in) + 1.0e-12,
            "bottom_rest": lambda x, y: abs(y - 0.0) <= 1.0e-12 and float(geom.channel_x0) - 1.0e-12 <= x <= float(geom.channel_x1) + 1.0e-12 and abs(x - float(geom.center_x)) > float(geom.r_in) + 1.0e-12,
            # The coupled normal-traction law fixes the Darcy head level relative to the
            # Stokes pressure, so only the Stokes pressure gauge is pinned here.
            "pressure_pin": lambda x, y: abs(x - float(geom.center_x)) <= tol_pin and abs(y - 0.0) <= tol_pin,
        },
        fields=["ux_s", "uy_s", "p_s"],
    )

    vel_space = FunctionSpace("velocity", ["ux_s", "uy_s"], dim=1)
    u = VectorTrialFunction(space=vel_space, dof_handler=dh)
    w = VectorTestFunction(space=vel_space, dof_handler=dh)
    p = TrialFunction("p_s", dof_handler=dh)
    q = TestFunction("p_s", dof_handler=dh)
    phi = TrialFunction("phi_d", dof_handler=dh)
    psi = TestFunction("phi_d", dof_handler=dh)
    u_k = VectorFunction("u_k", ["ux_s", "uy_s"], dof_handler=dh)
    p_k = Function("p_k", "p_s", dof_handler=dh)
    phi_k = Function("phi_k", "phi_d", dof_handler=dh)
    u_n = VectorFunction("u_n", ["ux_s", "uy_s"], dof_handler=dh)
    p_n = Function("p_n", "p_s", dof_handler=dh)
    phi_n = Function("phi_n", "phi_d", dof_handler=dh)
    for f in (u_k, p_k, phi_k, u_n, p_n, phi_n):
        f.nodal_values.fill(0.0)

    n = FacetNormal()
    tau_x = -n[1]
    tau_y = n[0]
    u_tang = Neg(u_k)[0] * tau_x + Neg(u_k)[1] * tau_y
    w_tang = Neg(w)[0] * tau_x + Neg(w)[1] * tau_y
    du_tang = Neg(u)[0] * tau_x + Neg(u)[1] * tau_y

    tau_lsic = Constant(float(lsic_scale) * float(Umax) * h / 2.0)
    beta_bjs = Constant(float(friction_alpha / math.sqrt(K)))
    conv_factor = Constant(0.0)

    residual_form = (
        conv_factor * dot(dot(grad(u_k), u_k), w) * dx_s
        + Constant(2.0 * float(nu)) * inner(_eps(u_k), _eps(w)) * dx_s
        - p_k * div(w) * dx_s
        + q * div(u_k) * dx_s
        + tau_lsic * div(u_k) * div(w) * dx_s
        + Constant(float(K)) * inner(grad(phi_k), grad(psi)) * dx_d
        - Pos(psi) * dot(Neg(u_k), n) * dGamma
        + Constant(float(g)) * Pos(phi_k) * dot(Neg(w), n) * dGamma
        + beta_bjs * u_tang * w_tang * dGamma
    )
    jacobian_form = (
        conv_factor * (dot(dot(grad(u), u_k), w) + dot(dot(grad(u_k), u), w)) * dx_s
        + Constant(2.0 * float(nu)) * inner(_eps(u), _eps(w)) * dx_s
        - p * div(w) * dx_s
        + q * div(u) * dx_s
        + tau_lsic * div(u) * div(w) * dx_s
        + Constant(float(K)) * inner(grad(phi), grad(psi)) * dx_d
        - Pos(psi) * dot(Neg(u), n) * dGamma
        + Constant(float(g)) * Pos(phi) * dot(Neg(w), n) * dGamma
        + beta_bjs * du_tang * w_tang * dGamma
    )

    zero = lambda x, y, t=0.0: 0.0

    def uy_inflow(x, y, t=0.0):
        xx = float(x)
        rr = (xx - float(geom.center_x)) / float(geom.r_in)
        if abs(xx - float(geom.center_x)) > float(geom.r_in):
            return 0.0
        return float(Umax) * (1.0 - rr * rr)

    bcs = [
        BoundaryCondition("ux_s", "dirichlet", "bottom_active", zero),
        BoundaryCondition("uy_s", "dirichlet", "bottom_active", uy_inflow),
        BoundaryCondition("ux_s", "dirichlet", "bottom_rest", zero),
        BoundaryCondition("uy_s", "dirichlet", "bottom_rest", zero),
        BoundaryCondition("p_s", "dirichlet", "pressure_pin", zero),
        BoundaryCondition("ux_s", "dirichlet", "inactive", zero),
        BoundaryCondition("uy_s", "dirichlet", "inactive", zero),
        BoundaryCondition("p_s", "dirichlet", "inactive", zero),
        BoundaryCondition("phi_d", "dirichlet", "inactive", zero),
    ]
    bcs_homog = [
        BoundaryCondition("ux_s", "dirichlet", "bottom_active", zero),
        BoundaryCondition("uy_s", "dirichlet", "bottom_active", zero),
        BoundaryCondition("ux_s", "dirichlet", "bottom_rest", zero),
        BoundaryCondition("uy_s", "dirichlet", "bottom_rest", zero),
        BoundaryCondition("p_s", "dirichlet", "pressure_pin", zero),
        BoundaryCondition("ux_s", "dirichlet", "inactive", zero),
        BoundaryCondition("uy_s", "dirichlet", "inactive", zero),
        BoundaryCondition("p_s", "dirichlet", "inactive", zero),
        BoundaryCondition("phi_d", "dirichlet", "inactive", zero),
    ]

    solver = NewtonSolver(
        residual_form=residual_form,
        jacobian_form=jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=1.0e-8, max_newton_iter=40, line_search=True),
        backend=str(backend),
    )
    bcs_now = NewtonSolver._freeze_bcs(bcs, 0.0)
    functions = [u_k, p_k, phi_k]
    prev_functions = [u_n, p_n, phi_n]
    dh.apply_bcs(bcs_now, *functions)
    dh.apply_bcs(bcs_now, *prev_functions)

    conv_factor.value = 0.0
    _, converged_stokes, iters_stokes = solver._newton_loop(functions, prev_functions, None, bcs_now)
    if not bool(converged_stokes):
        raise RuntimeError("Sharp Stoter Stokes initialization did not converge.")
    u_n.nodal_values[:] = u_k.nodal_values[:]
    p_n.nodal_values[:] = p_k.nodal_values[:]
    phi_n.nodal_values[:] = phi_k.nodal_values[:]

    conv_factor.value = 1.0
    _, converged_ns, iters_ns = solver._newton_loop(functions, prev_functions, None, bcs_now)
    if not bool(converged_ns):
        raise RuntimeError("Sharp Stoter Navier--Stokes/Darcy solve did not converge.")

    centerline_path = _write_centerline_samples(outdir, dh, mesh, geom, u_k[0], u_k[1], phi_k, float(K))
    velocity_grid_path = _write_velocity_grid_samples(outdir, dh, mesh, geom, u_k[0], u_k[1], phi_k, float(K))
    plot_paths = _write_velocity_plots(outdir, velocity_grid_path, geom, umax=float(Umax))

    probe = (float(geom.center_x), 50.0)
    grad_phi_probe = _eval_scalar_grad_at_point(dh, mesh, "phi_d", phi_k, probe)
    combined_probe = [
        float(-float(K) * float(grad_phi_probe[0])),
        float(-float(K) * float(grad_phi_probe[1])),
    ]

    if export:
        vtk_fields = {
            "u_stokes": u_k,
            "p_s": p_k,
            "phi_d": phi_k,
        }
        export_vtk(str(outdir / "final_state.vtu"), mesh=mesh, dof_handler=dh, functions=vtk_fields)

    summary: dict[str, object] = {
        "benchmark": "stoter_section_5_3_channel_porous_2d_sharp_two_domain",
        "nx": int(nx),
        "ny": int(ny),
        "h_x": float(hx),
        "h_y": float(hy),
        "Umax": float(Umax),
        "K": float(K),
        "nu": float(nu),
        "friction_alpha": float(friction_alpha),
        "g": float(g),
        "lsic_tau": float(float(lsic_scale) * float(Umax) * h / 2.0),
        "mesh_mode": str(mesh_mode),
        "interface_segments": int(iface.n_segments()),
        "combined_velocity_probe_mm_per_s": combined_probe,
        "stokes_init_iterations": int(iters_stokes),
        "navier_darcy_iterations": int(iters_ns),
        "centerline_csv": str(centerline_path),
        "velocity_grid_csv": str(velocity_grid_path),
        **plot_paths,
    }
    if compare_one_domain_csv is not None:
        summary["one_domain_comparison"] = _compare_velocity_grids(
            reference_csv=Path(compare_one_domain_csv),
            candidate_csv=velocity_grid_path,
        )

    with (outdir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=40)
    ap.add_argument("--Umax", type=float, default=10.0)
    ap.add_argument("--K", type=float, default=5000.0)
    ap.add_argument("--nu", type=float, default=2.927)
    ap.add_argument("--friction-alpha", type=float, default=1.0)
    ap.add_argument("--g", type=float, default=9.81)
    ap.add_argument("--lsic-scale", type=float, default=1.0)
    ap.add_argument("--mesh-mode", type=str, default="refined", choices=["uniform", "refined"])
    ap.add_argument("--backend", type=str, default="cpp", choices=["python", "jit", "cpp"])
    ap.add_argument("--outdir", type=str, default="out/stoter_channel_sharp_32x40")
    ap.add_argument("--compare-one-domain-csv", type=str, default="")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    outdir = Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    summary = solve_benchmark(
        nx=int(args.nx),
        ny=int(args.ny),
        Umax=float(args.Umax),
        K=float(args.K),
        nu=float(args.nu),
        friction_alpha=float(args.friction_alpha),
        g=float(args.g),
        lsic_scale=float(args.lsic_scale),
        mesh_mode=str(args.mesh_mode),
        backend=str(args.backend),
        export=not bool(args.no_export),
        outdir=outdir,
        compare_one_domain_csv=Path(str(args.compare_one_domain_csv)) if str(args.compare_one_domain_csv).strip() else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
