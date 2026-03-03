"""
Li et al. (2020) biofilm deformation benchmark (one-domain model).

Reference
---------
Li et al., "Predicting biofilm deformation with a viscoelastic phase-field model:
Modeling and experimental studies", Biotechnol Bioeng., DOI: 10.1002/bit.27491
(PDF copy in this folder: `Li-2020.pdf`).

This driver aligns (as closely as possible within the current pycutfem one-domain
Navier–Stokes–Brinkman–Biot model) with the paper's channel-flow deformation setup:
  - channel height H = 10 mm,
  - a rigid support block (1 mm wide × 3 mm tall) glued to the bottom at the channel center,
  - the biofilm is attached to the *top* of the support (clamped there),
  - fluid: rho = 1000 kg/m^3, mu = 1e-3 Pa·s,
  - average inflow speed u_avg = 6e-4 m/s -> Re ≈ rho*u_avg*H/mu ≈ 6,
  - velocity ramp over Tramp = 1 s,
  - simulate to T = 20 s.

Key modeling choices vs the paper
---------------------------------
The paper uses a two-fluid phase-field + Oldroyd-B viscoelastic constitutive law.
The one-domain model in `examples/utils/biofilm/one_domain.py` uses a poroelastic
skeleton coupled to a single mixture velocity via Brinkman drag.

To produce a comparable "deforming biofilm blob" test while keeping the run robust
and focused on deformation, this driver:
  - disables growth/detachment/chemistry,
  - evolves the indicator alpha either
      * by solving its conservative transport PDE with phase-field regularization
        (Cahn–Hilliard or conservative Allen–Cahn), or
      * by the Eulerian reference map alpha(x,t) = alpha0(x - u(x,t)) (debug option),
  - ties porosity to alpha via phi = 1 - (1-phi_b)*alpha,
  - maps (G_b, mu_b) from the paper to a Kelvin–Voigt skeleton:
        mu_s = G_b,  eta_s = mu_b,
    with a near-incompressible lambda_s computed from nu.

Outputs
-------
Writes to `out_dir`:
  - `timeseries.csv`: tracking-line displacements (alpha=0.5 contour) vs time,
  - `vtk/step=XXXX.vtu` (optional): VTK snapshots for visualization.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

import numpy as np

# Avoid extremely verbose Numba debug dumps if the environment enables them.
# These are read at import time by Numba, so set them before importing pycutfem.
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
from pycutfem.core.topology import Node
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.solvers.nonlinear_solver import (
    NewtonParameters,
    PdasNewtonSolver,
    PetscSnesNewtonSolver,
    TimeStepperParameters,
    VIParameters,
)
from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import ds, dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad

from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms


def _tag_channel_with_block_boundaries(
    mesh: Mesh,
    *,
    L: float,
    H: float,
    block_x0: float,
    block_x1: float,
    block_h: float,
    tol: float = 1.0e-12,
) -> None:
    L = float(L)
    H = float(H)
    block_x0 = float(block_x0)
    block_x1 = float(block_x1)
    block_h = float(block_h)

    def _in_block_x(x: float) -> bool:
        return (block_x0 - tol) <= float(x) <= (block_x1 + tol)

    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - L) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - H) <= tol,
            # Support block surfaces (these become boundary edges because the block is removed from the mesh).
            "block_top": lambda x, y: abs(y - block_h) <= tol and _in_block_x(x),
            "block_left": lambda x, y: abs(x - block_x0) <= tol and (y <= block_h + tol),
            "block_right": lambda x, y: abs(x - block_x1) <= tol and (y <= block_h + tol),
        }
    )


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


def _restrict_skeleton_dofs_to_alpha(
    dh: DofHandler,
    *,
    alpha0_kind: str,
    eps0: float,
    poly_m: np.ndarray | None,
    bio_x0: float,
    bio_y0: float,
    bio_w: float,
    bio_h: float,
    alpha_min: float,
) -> None:
    """
    Drop u/vS DOFs where the *initial* indicator α0 is small.

    Motivation
    ----------
    In a one-domain CG formulation, (u,vS) are defined over the entire channel and
    stabilized in the free-fluid region via extension penalties. For a small biofilm
    attached to a rigid support, those extension terms can dominate and suppress
    (u,vS) in the biofilm when the fluid region is orders of magnitude larger.

    By dropping (u,vS) DOFs far away from the biofilm (where α0≈0), we:
      - keep the monolithic solve well-posed without a large extension penalty,
      - avoid "fluid-region clamping" of the skeleton response,
      - retain the intended coupling because all physics terms involving (u,vS)
        are already localized by α (B=α(1-φ), β∝α, etc.).
    """
    alpha_min = float(alpha_min)
    if alpha_min <= 0.0:
        return

    alpha0_kind = str(alpha0_kind).strip().lower()
    eps0 = max(1.0e-12, float(eps0))

    # Use Q2 coordinates (same for u_x/u_y and vS_x/vS_y).
    u_xy = np.asarray(dh.get_dof_coords("u_x"), dtype=float)
    if alpha0_kind == "block":
        a = _alpha_rect_eval(
            u_xy[:, 0],
            u_xy[:, 1],
            x0=float(bio_x0),
            y0=float(bio_y0),
            w=float(bio_w),
            h=float(bio_h),
            eps=float(eps0),
        )
    elif alpha0_kind == "polygon":
        if poly_m is None:
            raise ValueError("poly_m must be provided when alpha0_kind='polygon'.")
        a = _smooth_step((-_signed_distance_polygon(u_xy[:, 0], u_xy[:, 1], poly_m)) / float(eps0))
    else:
        raise ValueError(f"Unknown alpha0-kind: {alpha0_kind}")

    keep = np.asarray(a, dtype=float).ravel() > float(alpha_min)
    # Keep everything if the threshold would remove essentially all DOFs.
    if int(np.sum(keep)) < max(10, int(0.01 * float(keep.size))):
        return

    for fname in ("u_x", "u_y", "vS_x", "vS_y"):
        sl = np.asarray(dh.get_field_slice(fname), dtype=int).ravel()
        if sl.size != keep.size:
            raise RuntimeError(f"Unexpected DOF count mismatch for {fname}: slice={sl.size}, coords={keep.size}")
        _mark_inactive_dofs(dh, sl[~keep])


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
    penalties (`gamma_u`, `u_extension_mode`) keep (u,vS) well-posed in the free
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
    keep = (
        (u_xy[:, 0] >= x0)
        & (u_xy[:, 0] <= x1)
        & (u_xy[:, 1] >= y0)
        & (u_xy[:, 1] <= y1)
    )
    n_keep = int(np.sum(keep))
    n_tot = int(keep.size)
    if n_keep < max(10, int(0.01 * float(n_tot))):
        raise RuntimeError(
            f"restrict-skeleton box keeps too few DOFs: {n_keep}/{n_tot}. "
            "Increase the box or disable restriction."
        )

    for fname in ("u_x", "u_y", "vS_x", "vS_y"):
        sl = np.asarray(dh.get_field_slice(fname), dtype=int).ravel()
        if sl.size != keep.size:
            raise RuntimeError(f"Unexpected DOF count mismatch for {fname}: slice={sl.size}, coords={keep.size}")
        _mark_inactive_dofs(dh, sl[~keep])
    return n_keep, n_tot


def _smooth_step(z: np.ndarray) -> np.ndarray:
    # Robust sigmoid: 0.5*(1+tanh(z)).
    return 0.5 * (1.0 + np.tanh(np.asarray(z, dtype=float)))


def _alpha_rect_eval(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x0: float,
    y0: float,
    w: float,
    h: float,
    eps: float,
) -> np.ndarray:
    """
    Smooth indicator for a rectangle:
      alpha ≈ 1 in [x0,x0+w]×[y0,y0+h], alpha ≈ 0 outside.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    eps = max(float(eps), 1.0e-12)
    sx0 = _smooth_step((x - float(x0)) / eps)
    sx1 = _smooth_step((float(x0 + w) - x) / eps)
    sy0 = _smooth_step((y - float(y0)) / eps)
    sy1 = _smooth_step((float(y0 + h) - y) / eps)
    return sx0 * sx1 * sy0 * sy1


def _read_polygon_csv(path: str) -> np.ndarray:
    arr = np.genfromtxt(str(path), delimiter=",", skip_header=1, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    arr = np.asarray(arr, dtype=float)
    if arr.shape[1] < 2:
        raise ValueError(f"Polygon CSV must have at least 2 columns; got shape={arr.shape}")
    pts = arr[:, :2]
    if pts.shape[0] < 3:
        raise ValueError(f"Polygon must have at least 3 points; got {pts.shape[0]}")
    # Ensure closed.
    if not np.allclose(pts[0], pts[-1], rtol=0.0, atol=1.0e-12):
        pts = np.vstack([pts, pts[0]])
    return pts


def _load_aligned_polygon_m(
    *,
    path: str,
    scale: float,
    align: str,
    block_x0: float,
    block_xc: float,
    block_h: float,
    tx: float,
    ty: float,
) -> np.ndarray:
    poly_mm = _read_polygon_csv(str(path))
    poly_m = poly_mm * float(scale)
    if str(align) == "block":
        poly_ymin = float(np.min(poly_m[:, 1]))
        # Prefer aligning the *base span* (y=ymin) to the rigid block top (x0..x1),
        # which keeps the biofilm footprint inside the 1mm support and avoids a small
        # upstream overhang on coarse meshes.
        tol_y = max(1.0e-12, 1.0e-6 * max(1.0, float(np.max(np.abs(poly_m[:, 1])))))
        base_pts = poly_m[np.abs(poly_m[:, 1] - poly_ymin) <= tol_y]
        if base_pts.shape[0] >= 2:
            base_xmin = float(np.min(base_pts[:, 0]))
            dx_align = float(block_x0 - base_xmin)
        else:
            poly_xc = 0.5 * (float(np.min(poly_m[:, 0])) + float(np.max(poly_m[:, 0])))
            dx_align = float(block_xc - poly_xc)
        poly_m = poly_m + np.array([dx_align, float(block_h - poly_ymin)], dtype=float)
    poly_m = poly_m + np.array([float(tx), float(ty)], dtype=float)
    return np.asarray(poly_m, dtype=float)


def _parse_float_list(s: str, *, n: int) -> list[float]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    out = [float(p) for p in parts]
    if len(out) != int(n):
        raise ValueError(f"Expected {int(n)} comma-separated values; got {len(out)} in {s!r}")
    return out


def _signed_distance_polygon(x: np.ndarray, y: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """
    Signed distance to a simple polygon (negative inside).

    Parameters
    ----------
    x, y
        Query point coordinates (broadcastable to same shape).
    poly
        Polygon vertices, shape (M,2), closed (poly[0]==poly[-1]).
    """
    xq = np.asarray(x, dtype=float).ravel()
    yq = np.asarray(y, dtype=float).ravel()
    pts = np.asarray(poly, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] < 4:
        raise ValueError("poly must be closed with shape (M,2), M>=4")

    # Point-in-polygon via ray casting (vectorized over points, loop over edges).
    inside = np.zeros_like(xq, dtype=bool)
    xi = pts[:-1, 0]
    yi = pts[:-1, 1]
    xj = pts[1:, 0]
    yj = pts[1:, 1]
    for a_x, a_y, b_x, b_y in zip(xi, yi, xj, yj):
        # Does the segment cross the horizontal ray to +inf from (xq,yq)?
        cond = ((a_y > yq) != (b_y > yq))
        # Avoid division by zero on horizontal segments.
        denom = (b_y - a_y) if abs(b_y - a_y) > 1.0e-30 else 1.0e-30
        x_at_y = (b_x - a_x) * (yq - a_y) / denom + a_x
        inside ^= cond & (xq < x_at_y)

    # Distance to segments (vectorized over points, loop over edges).
    d2 = np.full_like(xq, float("inf"), dtype=float)
    for a_x, a_y, b_x, b_y in zip(xi, yi, xj, yj):
        abx = float(b_x - a_x)
        aby = float(b_y - a_y)
        ab2 = abx * abx + aby * aby
        if ab2 <= 1.0e-30:
            dx = xq - float(a_x)
            dy = yq - float(a_y)
            d2 = np.minimum(d2, dx * dx + dy * dy)
            continue
        apx = xq - float(a_x)
        apy = yq - float(a_y)
        t = (apx * abx + apy * aby) / ab2
        t = np.clip(t, 0.0, 1.0)
        cx = float(a_x) + t * abx
        cy = float(a_y) + t * aby
        dx = xq - cx
        dy = yq - cy
        d2 = np.minimum(d2, dx * dx + dy * dy)
    d = np.sqrt(np.maximum(d2, 0.0))
    d[inside] *= -1.0
    return d.reshape(np.asarray(x, dtype=float).shape)


def _auto_split_counts(total: int, parts: tuple[float, ...], *, min_each: int = 1) -> tuple[int, ...]:
    total = int(total)
    if total <= 0:
        raise ValueError("total must be positive")
    w = np.asarray(parts, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("parts must be a non-empty 1D sequence")
    if np.any(w < 0.0):
        raise ValueError("parts must be non-negative")
    if float(np.sum(w)) <= 0.0:
        raise ValueError("parts sum must be positive")

    weights = w / float(np.sum(w))
    counts = np.floor(weights * float(total)).astype(int)
    for i in range(counts.size):
        if w[i] > 0.0 and counts[i] < int(min_each):
            counts[i] = int(min_each)

    # Distribute remainder to the largest-weight bins.
    while int(np.sum(counts)) < total:
        i = int(np.argmax(weights))
        counts[i] += 1
    while int(np.sum(counts)) > total:
        # Take from the largest count that still stays above min_each.
        cand = [i for i in range(counts.size) if counts[i] > int(min_each)]
        if not cand:
            break
        i = max(cand, key=lambda j: int(counts[j]))
        counts[i] -= 1
    return tuple(int(c) for c in counts.tolist())


def _merge_mesh_parts(parts: list[tuple[list[Node], np.ndarray, np.ndarray]]) -> tuple[list[Node], np.ndarray, np.ndarray]:
    """
    Merge multiple conforming structured-quad meshes into one, deduplicating nodes
    by coordinate (needed for the channel-minus-block decomposition).
    """
    lookup: dict[tuple[float, float], int] = {}
    nodes_out: list[Node] = []
    elems_out: list[np.ndarray] = []
    corners_out: list[np.ndarray] = []
    ndigits = 12

    for nodes, elems, corners in parts:
        if not nodes:
            continue
        local2global = np.empty((len(nodes),), dtype=int)
        for n in nodes:
            key = (round(float(n.x), ndigits), round(float(n.y), ndigits))
            gid = lookup.get(key)
            if gid is None:
                gid = len(nodes_out)
                lookup[key] = gid
                nodes_out.append(Node(gid, float(key[0]), float(key[1])))
            local2global[int(n.id)] = int(gid)

        elems_out.append(local2global[np.asarray(elems, dtype=int)])
        corners_out.append(local2global[np.asarray(corners, dtype=int)])

    if not elems_out:
        raise RuntimeError("No mesh parts were generated.")
    return nodes_out, np.vstack(elems_out), np.vstack(corners_out)


def _channel_minus_block_mesh(
    *,
    L: float,
    H: float,
    block_x0: float,
    block_x1: float,
    block_h: float,
    nx_left: int,
    nx_mid: int,
    nx_right: int,
    ny_bottom: int,
    ny_top: int,
    poly_order: int,
) -> tuple[list[Node], np.ndarray, np.ndarray]:
    """
    Structured mesh for (0,L)x(0,H) with a rectangular block (block_x0..block_x1)x(0..block_h) removed.
    """
    L = float(L)
    H = float(H)
    block_x0 = float(block_x0)
    block_x1 = float(block_x1)
    block_h = float(block_h)
    if not (0.0 < block_x0 < block_x1 < L):
        raise ValueError("Block must lie strictly inside the channel in x.")
    if not (0.0 < block_h < H):
        raise ValueError("Block height must satisfy 0 < block_h < H.")

    w_left = block_x0
    w_mid = block_x1 - block_x0
    w_right = L - block_x1
    h_bot = block_h
    h_top = H - block_h

    parts: list[tuple[list[Node], np.ndarray, np.ndarray]] = []

    def add_part(*, w: float, h: float, nx: int, ny: int, off: tuple[float, float]) -> None:
        if float(w) <= 0.0 or float(h) <= 0.0 or int(nx) <= 0 or int(ny) <= 0:
            return
        nodes, elems, _edges, corners = structured_quad(float(w), float(h), nx=int(nx), ny=int(ny), poly_order=int(poly_order), offset=off)
        parts.append((nodes, np.asarray(elems, dtype=int), np.asarray(corners, dtype=int)))

    add_part(w=w_left, h=h_bot, nx=nx_left, ny=ny_bottom, off=(0.0, 0.0))  # left-bottom
    add_part(w=w_left, h=h_top, nx=nx_left, ny=ny_top, off=(0.0, block_h))  # left-top
    add_part(w=w_mid, h=h_top, nx=nx_mid, ny=ny_top, off=(block_x0, block_h))  # mid-top
    add_part(w=w_right, h=h_bot, nx=nx_right, ny=ny_bottom, off=(block_x1, 0.0))  # right-bottom
    add_part(w=w_right, h=h_top, nx=nx_right, ny=ny_top, off=(block_x1, block_h))  # right-top

    return _merge_mesh_parts(parts)


def _lame_lambda_from_nu(mu: float, nu: float) -> float:
    nu = float(nu)
    if not (-1.0 < nu < 0.5):
        raise ValueError(f"nu must satisfy -1 < nu < 0.5; got nu={nu}")
    return float(2.0 * float(mu) * nu / max(1.0e-16, (1.0 - 2.0 * nu)))


def _build_coord_lookup(coords: np.ndarray, *, ndigits: int = 12) -> dict[tuple[float, float], int]:
    out: dict[tuple[float, float], int] = {}
    for i, xy in enumerate(np.asarray(coords, dtype=float)):
        out[(round(float(xy[0]), ndigits), round(float(xy[1]), ndigits))] = int(i)
    return out


def _nearest_y_levels(y_all: np.ndarray, targets: list[float]) -> list[float]:
    y = np.unique(np.asarray(y_all, dtype=float))
    if y.size == 0:
        return [float(t) for t in targets]
    y = np.sort(np.asarray(y, dtype=float))

    # Preserve the ordering of tracking lines (line 1 highest -> line N lowest).
    # The prior implementation chose distinct levels but could pick out-of-order
    # y-values on coarse meshes due to the "prefer higher y" tie-breaker.
    targets_f = [float(t) for t in targets]
    is_desc = all(targets_f[i] >= targets_f[i + 1] for i in range(len(targets_f) - 1))
    is_asc = all(targets_f[i] <= targets_f[i + 1] for i in range(len(targets_f) - 1))
    if not (is_desc or is_asc):
        # Fallback: treat as descending (this is how the Lie benchmark uses it).
        is_desc = True

    out: list[float] = []
    used: set[int] = set()
    tol = 1.0e-14
    if is_desc:
        max_allowed = float("inf")
        for t in targets_f:
            cand = np.array([i for i in range(y.size) if (i not in used) and (y[i] <= max_allowed + tol)], dtype=int)
            if cand.size == 0:
                cand = np.array([i for i in range(y.size) if y[i] <= max_allowed + tol], dtype=int)
            if cand.size == 0:
                cand = np.arange(0, y.size, dtype=int)
            diffs = np.abs(y[cand] - float(t))
            jmin = float(np.min(diffs))
            best = cand[diffs == jmin]
            # Tie-break: prefer the higher y but still within the allowed band.
            j = int(best[-1])
            out.append(float(y[j]))
            used.add(int(j))
            max_allowed = float(y[j]) - tol
    else:
        min_allowed = float("-inf")
        for t in targets_f:
            cand = np.array([i for i in range(y.size) if (i not in used) and (y[i] >= min_allowed - tol)], dtype=int)
            if cand.size == 0:
                cand = np.array([i for i in range(y.size) if y[i] >= min_allowed - tol], dtype=int)
            if cand.size == 0:
                cand = np.arange(0, y.size, dtype=int)
            diffs = np.abs(y[cand] - float(t))
            jmin = float(np.min(diffs))
            best = cand[diffs == jmin]
            # Tie-break: prefer the lower y while within the allowed band.
            j = int(best[0])
            out.append(float(y[j]))
            used.add(int(j))
            min_allowed = float(y[j]) + tol
    return out


def _x_alpha_half_on_y_line(alpha_xy: np.ndarray, alpha_vals: np.ndarray, *, y_line: float, alpha_half: float = 0.5) -> float:
    xy = np.asarray(alpha_xy, dtype=float)
    a = np.asarray(alpha_vals, dtype=float).ravel()
    if xy.ndim != 2 or xy.shape[1] < 2 or a.shape[0] != xy.shape[0]:
        return float("nan")

    # Structured meshes give exact y-levels; use exact match with a tiny tol.
    mask = np.abs(xy[:, 1] - float(y_line)) <= 1.0e-14
    if not np.any(mask):
        return float("nan")

    x = np.asarray(xy[mask, 0], dtype=float).ravel()
    av = np.asarray(a[mask], dtype=float).ravel()
    if x.size < 2:
        return float("nan")

    order = np.argsort(x)
    x = x[order]
    av = av[order]

    # Find the last index where alpha is still above the threshold, then
    # interpolate with the next node (assumes a single right-edge crossing).
    above = av >= float(alpha_half)
    if not np.any(above):
        return float("nan")
    i = int(np.max(np.nonzero(above)[0]))
    if i >= x.size - 1:
        return float(x[-1])
    a0 = float(av[i])
    a1 = float(av[i + 1])
    x0 = float(x[i])
    x1 = float(x[i + 1])
    da = a1 - a0
    if abs(da) <= 1.0e-16:
        return float(x0)
    t = (float(alpha_half) - a0) / da
    return float(x0 + t * (x1 - x0))


def _x_alpha_half_on_y_line_mode(
    alpha_xy: np.ndarray,
    alpha_vals: np.ndarray,
    *,
    y_line: float,
    mode: str,
    alpha_half: float = 0.5,
) -> float:
    """
    Intersection x of the alpha=alpha_half contour with the horizontal line y=y_line.

    mode:
      - "rightmost": downstream edge (max x)
      - "leftmost":  upstream edge (min x)
    """
    mode = str(mode).strip().lower()
    if mode not in {"rightmost", "leftmost"}:
        raise ValueError(f"Unknown dx intersection mode {mode!r}. Use 'rightmost' or 'leftmost'.")

    xy = np.asarray(alpha_xy, dtype=float)
    a = np.asarray(alpha_vals, dtype=float).ravel()
    if xy.ndim != 2 or xy.shape[1] < 2 or a.shape[0] != xy.shape[0]:
        return float("nan")

    mask = np.abs(xy[:, 1] - float(y_line)) <= 1.0e-14
    if not np.any(mask):
        return float("nan")

    x = np.asarray(xy[mask, 0], dtype=float).ravel()
    av = np.asarray(a[mask], dtype=float).ravel()
    if x.size < 2:
        return float("nan")

    order = np.argsort(x)
    x = x[order]
    av = av[order]

    above = av >= float(alpha_half)
    idx = np.nonzero(above)[0]
    if idx.size == 0:
        return float("nan")
    i0 = int(idx[0])
    i1 = int(idx[-1])

    if mode == "rightmost":
        if i1 >= x.size - 1:
            return float(x[-1])
        a0, a1 = float(av[i1]), float(av[i1 + 1])
        x0, x1 = float(x[i1]), float(x[i1 + 1])
    else:
        if i0 <= 0:
            return float(x[0])
        a0, a1 = float(av[i0 - 1]), float(av[i0])
        x0, x1 = float(x[i0 - 1]), float(x[i0])

    da = a1 - a0
    if abs(da) <= 1.0e-16:
        return float(x1 if mode == "leftmost" else x0)
    t = (float(alpha_half) - a0) / da
    return float(x0 + t * (x1 - x0))


def _x_alpha_half_quantile_on_y_line(
    alpha_xy: np.ndarray,
    alpha_vals: np.ndarray,
    *,
    y_line: float,
    mode: str,
    q: float,
    alpha_half: float = 0.5,
) -> float:
    """
    Track an interior x-position based on the alpha=alpha_half interval on y=y_line.

    Let (x_L, x_R) be the left/right intersections of the alpha=alpha_half contour.
    Then:
      - mode="leftmost":  x = x_L + q (x_R - x_L)
      - mode="rightmost": x = x_R - q (x_R - x_L)

    With q=0 this reduces to the boundary intersection used in the paper-style plots.
    With q>0 this approximates a DIC-like "point inside the body" at a fixed fraction
    of the cross-section width (more robust than the extreme boundary point).
    """
    mode = str(mode).strip().lower()
    if mode not in {"rightmost", "leftmost"}:
        raise ValueError(f"Unknown dx intersection mode {mode!r}. Use 'rightmost' or 'leftmost'.")
    q = float(np.clip(float(q), 0.0, 1.0))
    xL = _x_alpha_half_on_y_line_mode(alpha_xy, alpha_vals, y_line=float(y_line), mode="leftmost", alpha_half=float(alpha_half))
    xR = _x_alpha_half_on_y_line_mode(alpha_xy, alpha_vals, y_line=float(y_line), mode="rightmost", alpha_half=float(alpha_half))
    if not (math.isfinite(xL) and math.isfinite(xR)):
        return float("nan")
    if mode == "leftmost":
        return float(xL + q * (xR - xL))
    return float(xR - q * (xR - xL))


def main() -> None:
    ap = argparse.ArgumentParser(description="Li et al. (2008) synthetic biofilm deformation benchmark (one-domain).")
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--nx", type=int, default=200)
    ap.add_argument("--ny", type=int, default=80)
    ap.add_argument("--nx-left", type=int, default=0, help="Override left-region cells (0=auto).")
    ap.add_argument("--nx-mid", type=int, default=0, help="Override mid-region cells (0=auto).")
    ap.add_argument("--nx-right", type=int, default=0, help="Override right-region cells (0=auto).")
    ap.add_argument("--ny-bottom", type=int, default=0, help="Override bottom-region cells (0=auto).")
    ap.add_argument("--ny-top", type=int, default=0, help="Override top-region cells (0=auto).")
    ap.add_argument(
        "--refine-biofilm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refine the mesh around the *initial* biofilm bounding box (one refinement level; supports ≤1 hanging node per edge).",
    )
    ap.add_argument(
        "--refine-biofilm-pad",
        type=float,
        default=5.0e-4,
        help="Padding [m] added to the initial biofilm bounding box to define the refinement region (used with --refine-biofilm).",
    )
    ap.add_argument(
        "--refine-biofilm-levels",
        type=int,
        default=1,
        help=(
            "Tensor refinement levels inside the refinement region (1=split marked quads into 2x2, 2=4x4, ...). "
            "Levels are balanced to keep ≤1 hanging node per edge."
        ),
    )
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--allow-dt-reduction", action="store_true", help="Adaptively reduce dt when Newton fails (recommended for --transport-mode pde).")
    ap.add_argument("--dt-min", type=float, default=0.0, help="Minimum dt allowed when --allow-dt-reduction is enabled.")
    ap.add_argument("--dt-reduction-factor", type=float, default=0.5, help="dt <- factor*dt on step rejection (adaptive dt).")
    ap.add_argument(
        "--stop-on-steady",
        action="store_true",
        help=(
            "Early-exit time stepping when the Newton update is below --steady-tol (∞-norm), "
            "then extend timeseries.csv with a constant value at t_final. Useful to speed up "
            "calibration runs that reach steady state quickly."
        ),
    )
    ap.add_argument("--steady-tol", type=float, default=1.0e-12, help="Steady-state threshold on ||ΔU||_∞ (used with --stop-on-steady).")
    ap.add_argument("--t-final", type=float, default=20.0)
    ap.add_argument("--theta", type=float, default=1.0, help="Theta-scheme (1=BE, 0.5=CN).")
    ap.add_argument("--newton-tol", type=float, default=3.0e-6)
    ap.add_argument("--max-it", type=int, default=25)
    ap.add_argument(
        "--snes-accept-factor",
        type=float,
        default=3.0,
        help="If PETSc SNES reports non-convergence but ‖F‖ <= factor*newton_tol, accept the best iterate (0 disables).",
    )
    ap.add_argument(
        "--newton-solver",
        type=str,
        default="pdas",
        choices=("pdas", "snes"),
        help="Nonlinear solver: 'pdas' (default) supports alpha box constraints; 'snes' uses PETSc SNES.",
    )
    ap.add_argument(
        "--alpha-box-constraints",
        action="store_true",
        help="Enforce 0<=alpha<=1 via a box-constrained VI solve (PDAS). Recommended when alpha overshoots.",
    )
    ap.add_argument(
        "--no-alpha-box-constraints",
        action="store_true",
        help="Disable alpha box constraints (0<=alpha<=1) when using --newton-solver pdas.",
    )
    ap.add_argument(
        "--alpha-metrics",
        action="store_true",
        help="Write alpha integral/centroid/min/max to out_dir/alpha_metrics.csv each accepted step.",
    )
    ap.add_argument(
        "--alpha-clip-below-block",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "In PDE transport mode, keep the biofilm indicator out of the lower channel floor region by "
            "freezing alpha=0 (and phi=1) for all DOFs with y < block_h. This prevents CH regularization "
            "from 'wetting' down the block sides in the Lie benchmark (no-detachment assumption)."
        ),
    )
    ap.add_argument(
        "--alpha-pin-block-top",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "In PDE transport mode, enforce alpha=alpha0 on the rigid-support top boundary (block_top). "
            "This pins the contact line to the initial footprint (no detachment/spreading)."
        ),
    )
    ap.add_argument(
        "--alpha-pin-block-top-value",
        type=float,
        default=1.0,
        help=(
            "Optional constant value used with --alpha-pin-block-top. "
            "If not set, the pinned value is alpha0(x,y) on block_top. "
            "Set to 1.0 to enforce full attachment where alpha0 is high (may slightly change alpha mass)."
        ),
    )
    ap.add_argument(
        "--alpha-pin-block-top-alpha0-min",
        type=float,
        default=0.9,
        help=(
            "When using --alpha-pin-block-top-value, apply the constant value only where alpha0(x,block_h) >= this "
            "threshold; elsewhere the boundary value remains alpha0. This avoids forcing alpha=1 at the diffuse "
            "corner/edge nodes (which can reintroduce a 'tilted base' artifact)."
        ),
    )
    ap.add_argument(
        "--alpha-zero-block-sides",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "In PDE transport mode, enforce alpha=0 on the rigid-support side boundaries (block_left/block_right), "
            "except at the top corner node where alpha is set to the block-top value to match --alpha-pin-block-top."
        ),
    )
    ap.add_argument("--out-dir", type=str, default="out/lie_synthetic_one_domain")
    ap.add_argument("--vtk-every", type=int, default=10, help="Write VTK every N steps (0 disables).")
    ap.add_argument(
        "--vtk-stress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When writing VTU files, also export cell-centered stress magnitudes (Fig. 8-style diagnostics): "
            "sigma_newtonian_dev_norm_alpha, tau_skel_visc_dev_norm_alpha, sigma_skel_el_dev_norm_alpha."
        ),
    )

    # Geometry + interface
    ap.add_argument("--L", type=float, default=15.0e-3, help="Channel length [m].")
    ap.add_argument("--H", type=float, default=10.0e-3)
    ap.add_argument("--block-w", type=float, default=1.0e-3, help="Support block width (diameter) [m].")
    ap.add_argument("--block-h", type=float, default=3.0e-3, help="Support block height [m].")
    ap.add_argument("--block-xc", type=float, default=float("nan"), help="Support block center x [m]. Default: L/2.")

    ap.add_argument("--alpha0-kind", type=str, default="polygon", choices=("block", "polygon"))
    ap.add_argument("--bio-x0", type=float, default=float("nan"), help="Biofilm block start x0 [m] (only for --alpha0-kind block).")
    ap.add_argument("--bio-y0", type=float, default=float("nan"), help="Biofilm block start y0 [m] (only for --alpha0-kind block).")
    ap.add_argument("--bio-w", type=float, default=1.0e-3, help="Biofilm block width [m] (only for --alpha0-kind block).")
    ap.add_argument("--bio-h", type=float, default=1.7e-3, help="Biofilm block height [m] (only for --alpha0-kind block).")
    ap.add_argument(
        "--alpha0-file",
        type=str,
        default="examples/biofilms/benchmarks/lie/biofilm_exp_s1_frame0_polygon_mm_scalebar_smooth.csv",
    )
    ap.add_argument("--alpha0-scale", type=float, default=1.0e-3, help="Scale polygon coordinates to meters (mm->m: 1e-3).")
    ap.add_argument("--alpha0-tx", type=float, default=0.0, help="Extra polygon x-translation [m].")
    ap.add_argument("--alpha0-ty", type=float, default=0.0, help="Extra polygon y-translation [m].")
    ap.add_argument(
        "--alpha0-align",
        type=str,
        default="block",
        choices=("none", "block"),
        help="Polygon alignment: 'block' maps (x_center,y_min) to (block_xc, block_h).",
    )
    ap.add_argument("--eps", type=float, default=2.0e-4, help="Diffuse interface half-thickness used in alpha0 [m].")
    ap.add_argument("--phi-b", type=float, default=0.47, help="Porosity inside biofilm (phi_b).")
    ap.add_argument(
        "--transport-mode",
        type=str,
        default="pde",
        choices=("refmap", "pde"),
        help="How alpha/phi are evolved: 'refmap' updates alpha from u(x,t); 'pde' (default) solves the alpha/phi PDEs.",
    )

    # Flow: paper provides average u1; we convert to a parabolic profile.
    ap.add_argument("--u-avg", type=float, default=6.0e-4, help="Average inflow speed u_avg [m/s].")
    ap.add_argument("--t-ramp", type=float, default=1.0, help="Linear ramp time for inflow [s].")

    # Material parameters (paper Table 1 for synthetic biofilm)
    ap.add_argument("--rho-f", type=float, default=1000.0)
    ap.add_argument("--mu-f", type=float, default=1.0e-3)
    ap.add_argument(
        "--mu-b-fluid",
        type=float,
        default=30494.0,
        help=(
            "Effective viscosity assigned to the biofilm region in the *mixture* momentum equation [Pa*s]. "
            "For Li et al. this corresponds to Table-1 μ_b (synthetic biofilm)."
        ),
    )
    ap.add_argument(
        "--mu-b-model",
        type=str,
        default="alpha_mu",
        choices=("mu", "phi_mu", "alpha_mu", "alpha_phi_mu"),
        help=(
            "Mixture viscosity model μ(α,φ): "
            "'mu' keeps μ=μ_f everywhere; "
            "'alpha_mu' sets μ=(1-α)μ_f + α μ_b; "
            "'alpha_phi_mu' sets μ=(1-α)μ_f + α φ μ_b."
        ),
    )
    ap.add_argument("--G-b", type=float, default=69736.0, help="Shear modulus (mapped to mu_s) [Pa].")
    ap.add_argument("--mu-b", type=float, default=30494.0, help="Viscosity (mapped to solid_visco_eta) [Pa*s].")
    ap.add_argument("--nu", type=float, default=0.49, help="Poisson ratio for near-incompressible linear elastic skeleton.")
    ap.add_argument(
        "--solid-model",
        type=str,
        default="linear",
        choices=("linear", "hencky", "svk", "neo_hookean"),
        help=(
            "Skeleton constitutive model. 'linear' uses small-strain elasticity; "
            "'hencky'/'svk'/'neo_hookean' enable finite-strain hyperelasticity "
            "(Eulerian reference-map formulation in `examples/utils/biofilm/one_domain.py`)."
        ),
    )

    # Poroelastic coupling
    ap.add_argument("--kappa-inv", type=float, default=1.0e12, help="Inverse permeability kappa^{-1} [1/m^2].")
    ap.add_argument("--gamma-u", type=float, default=1.0e-6, help="u extension penalty outside biofilm.")
    ap.add_argument(
        "--u-extension",
        type=str,
        default="grad",
        choices=("l2", "grad"),
        help="Extension mode for u outside biofilm (matches build_biofilm_one_domain_forms).",
    )
    ap.add_argument(
        "--gamma-u-pin",
        type=float,
        default=1.0e-10,
        help="Tiny L2 pinning used only with --u-extension grad (removes translation nullspace).",
    )
    ap.add_argument("--gamma-phi", type=float, default=5.0, help="Penalty enforcing phi->1 in free fluid.")
    ap.add_argument(
        "--restrict-skeleton-dofs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop (u,vS) DOFs where the initial indicator alpha0 is small (recommended). "
            "This avoids extension penalties suppressing biofilm deformation when the fluid region is large."
        ),
    )
    ap.add_argument(
        "--restrict-skeleton-method",
        type=str,
        default="box",
        choices=("alpha", "box"),
        help=(
            "How to choose active (u,vS) DOFs when --restrict-skeleton-dofs is enabled: "
            "'alpha' keeps DOFs where alpha0 > --restrict-skeleton-alpha-min; "
            "'box' keeps DOFs inside a fixed rectangle around the initial biofilm."
        ),
    )
    ap.add_argument(
        "--restrict-skeleton-alpha-min",
        type=float,
        default=0.05,
        help="Alpha threshold for keeping skeleton DOFs when --restrict-skeleton-dofs is enabled.",
    )
    ap.add_argument(
        "--restrict-skeleton-box-pad",
        type=float,
        default=2.0e-3,
        help=(
            "Padding [m] added to the initial biofilm bounding box to form the skeleton-active rectangle "
            "(only used when --restrict-skeleton-method box and explicit box bounds are not given)."
        ),
    )
    ap.add_argument("--restrict-skeleton-box-x0", type=float, default=float("nan"), help="Skeleton-active box x0 [m].")
    ap.add_argument("--restrict-skeleton-box-x1", type=float, default=float("nan"), help="Skeleton-active box x1 [m].")
    ap.add_argument("--restrict-skeleton-box-y0", type=float, default=float("nan"), help="Skeleton-active box y0 [m].")
    ap.add_argument("--restrict-skeleton-box-y1", type=float, default=float("nan"), help="Skeleton-active box y1 [m].")
    ap.add_argument("--D-alpha", type=float, default=0.0, help="Diffusion coefficient for alpha PDE (only used with --transport-mode pde).")
    ap.add_argument("--alpha-supg", type=float, default=0.0, help="SUPG stabilization strength for alpha advection (pde mode).")
    ap.add_argument("--alpha-cip", type=float, default=0.0, help="CIP stabilization strength for alpha advection (pde mode).")
    ap.add_argument(
        "--alpha-advection-form",
        type=str,
        default="conservative",
        choices=("advective", "conservative"),
        help="Form of alpha advection by vS used in the alpha PDE (pde mode).",
    )
    ap.add_argument("--alpha-ch-M", type=float, default=1.0e-12, help="Cahn–Hilliard mobility M for alpha (0 disables CH).")
    ap.add_argument("--alpha-ch-gamma", type=float, default=2.0e-3, help="Cahn–Hilliard gamma for alpha (0 disables CH).")
    ap.add_argument(
        "--alpha-ch-eps",
        type=float,
        default=float("nan"),
        help="Cahn–Hilliard eps (interface thickness). Default: use --eps.",
    )
    ap.add_argument(
        "--alpha-ch-mobility",
        type=str,
        default="degenerate",
        choices=("constant", "degenerate"),
        help="Mobility model for Cahn–Hilliard alpha regularization.",
    )
    ap.add_argument("--alpha-cahn-M", type=float, default=0.0, help="Allen–Cahn mobility M for alpha (0 disables Allen–Cahn).")
    ap.add_argument("--alpha-cahn-gamma", type=float, default=0.0, help="Allen–Cahn gamma for alpha (0 disables Allen–Cahn).")
    ap.add_argument(
        "--alpha-cahn-eps",
        type=float,
        default=float("nan"),
        help="Allen–Cahn eps (interface thickness). Default: use --eps.",
    )
    ap.add_argument(
        "--alpha-cahn-mobility",
        type=str,
        default="constant",
        choices=("constant", "degenerate"),
        help="Mobility model for Allen–Cahn alpha regularization.",
    )
    ap.add_argument(
        "--alpha-cahn-conservative",
        action="store_true",
        help="Enable mass-conserving (global λ_α) conservative Allen–Cahn (requires --transport-mode pde).",
    )
    ap.add_argument(
        "--dx-intersection",
        type=str,
        default="leftmost",
        choices=("rightmost", "leftmost"),
        help="Which alpha=0.5 contour intersection to track for dx(t) on each horizontal line.",
    )
    ap.add_argument(
        "--dx-quantile",
        type=float,
        default=0.0,
        help=(
            "Optional interior quantile between the left/right alpha=0.5 intersections on each tracking line. "
            "q=0 uses the boundary intersection; q=0.1 tracks a point 10 percent into the body from the chosen side."
        ),
    )
    ap.add_argument(
        "--dx-tracking",
        type=str,
        default="alpha_half",
        choices=("alpha_half", "lagrangian_u"),
        help=(
            "How to compute dx(t) for the 3 tracked points. "
            "'alpha_half' tracks the alpha=0.5 contour intersection on a fixed y-line (Eulerian). "
            "'lagrangian_u' tracks the material point defined at t=0 by the alpha=0.5 intersection, "
            "using fixed-point iteration x = x_ref + u(x,t) (DIC-like)."
        ),
    )
    ap.add_argument(
        "--dx-fixed-point-iters",
        type=int,
        default=5,
        help="Fixed-point iterations for --dx-tracking lagrangian_u (x = x_ref + u(x,t)).",
    )
    ap.add_argument(
        "--y-fracs",
        type=str,
        default="",
        help="Override tracking-line y-fractions (top->bottom), e.g. '0.75,0.5,0.25'.",
    )
    args = ap.parse_args()

    logging.getLogger("pycutfem.ufl.compilers").setLevel(logging.WARNING)

    transport_mode = str(args.transport_mode).strip().lower()
    if transport_mode not in {"refmap", "pde"}:
        raise ValueError(f"Unknown transport mode: {transport_mode!r}")

    alpha_ch_M = float(args.alpha_ch_M)
    alpha_ch_gamma = float(args.alpha_ch_gamma)
    ch_requested = bool(alpha_ch_M != 0.0 and alpha_ch_gamma != 0.0)
    if ch_requested and transport_mode != "pde":
        logging.warning("Ignoring --alpha-ch-* because --transport-mode=%s.", transport_mode)
        alpha_ch_M = 0.0
        alpha_ch_gamma = 0.0
    ch_enabled = bool(alpha_ch_M != 0.0 and alpha_ch_gamma != 0.0)
    alpha_ch_eps = float(args.alpha_ch_eps) if np.isfinite(float(args.alpha_ch_eps)) else float(args.eps)
    alpha_cahn_M = float(args.alpha_cahn_M)
    alpha_cahn_gamma = float(args.alpha_cahn_gamma)
    ac_requested = bool(alpha_cahn_M != 0.0 and alpha_cahn_gamma != 0.0)
    if ac_requested and transport_mode != "pde":
        logging.warning("Ignoring --alpha-cahn-* because --transport-mode=%s.", transport_mode)
        alpha_cahn_M = 0.0
        alpha_cahn_gamma = 0.0
    ac_enabled = bool(alpha_cahn_M != 0.0 and alpha_cahn_gamma != 0.0)
    alpha_cahn_eps = float(args.alpha_cahn_eps) if np.isfinite(float(args.alpha_cahn_eps)) else float(args.eps)
    alpha_cahn_conservative = bool(args.alpha_cahn_conservative)
    if alpha_cahn_conservative and transport_mode != "pde":
        logging.warning("Ignoring --alpha-cahn-conservative because --transport-mode=%s.", transport_mode)
        alpha_cahn_conservative = False
    if alpha_cahn_conservative and (not ac_enabled):
        raise ValueError("--alpha-cahn-conservative requires --alpha-cahn-M and --alpha-cahn-gamma to be nonzero.")
    if ac_enabled and ch_enabled:
        raise ValueError("Allen–Cahn (--alpha-cahn-*) and Cahn–Hilliard (--alpha-ch-*) cannot both be enabled.")

    backend = str(args.backend)
    L = float(args.L)
    H = float(args.H)
    dt_val = float(args.dt)
    theta = float(args.theta)
    qdeg = int(args.q)

    u_avg = float(args.u_avg)
    Umax = 1.5 * u_avg  # plane Poiseuille: u_max = 1.5 u_avg
    t_ramp = max(1.0e-12, float(args.t_ramp))

    block_w = float(args.block_w)
    block_h = float(args.block_h)
    block_xc = float(args.block_xc)
    if not np.isfinite(block_xc):
        block_xc = 0.5 * L
    block_x0 = block_xc - 0.5 * block_w
    block_x1 = block_xc + 0.5 * block_w
    if not (0.0 < block_x0 < block_x1 < L):
        raise ValueError(f"Support block must satisfy 0 < x0 < x1 < L; got x0={block_x0}, x1={block_x1}, L={L}.")
    if not (0.0 < block_h < H):
        raise ValueError(f"Support block must satisfy 0 < block_h < H; got block_h={block_h}, H={H}.")

    alpha0_kind = str(args.alpha0_kind)
    poly_m: np.ndarray | None = None
    eps0 = float(args.eps)
    bio_x0 = float(args.bio_x0)
    if not np.isfinite(bio_x0):
        bio_x0 = float(block_x0)
    bio_y0 = float(args.bio_y0)
    if not np.isfinite(bio_y0):
        bio_y0 = float(block_h)
    if alpha0_kind == "polygon":
        poly_m = _load_aligned_polygon_m(
            path=str(args.alpha0_file),
            scale=float(args.alpha0_scale),
            align=str(args.alpha0_align),
            block_x0=float(block_x0),
            block_xc=float(block_xc),
            block_h=float(block_h),
            tx=float(args.alpha0_tx),
            ty=float(args.alpha0_ty),
        )

    # Derive sub-partition counts for the channel-minus-block decomposition.
    if int(args.nx_left) > 0 or int(args.nx_mid) > 0 or int(args.nx_right) > 0:
        nx_left = int(args.nx_left) if int(args.nx_left) > 0 else 0
        nx_mid = int(args.nx_mid) if int(args.nx_mid) > 0 else 0
        nx_right = int(args.nx_right) if int(args.nx_right) > 0 else 0
        if nx_left + nx_mid + nx_right != int(args.nx):
            raise ValueError("--nx-left + --nx-mid + --nx-right must equal --nx when any override is used.")
    else:
        nx_left, nx_mid, nx_right = _auto_split_counts(int(args.nx), (block_x0, block_w, L - block_x1), min_each=2)

    if int(args.ny_bottom) > 0 or int(args.ny_top) > 0:
        ny_bottom = int(args.ny_bottom) if int(args.ny_bottom) > 0 else 0
        ny_top = int(args.ny_top) if int(args.ny_top) > 0 else 0
        if ny_bottom + ny_top != int(args.ny):
            raise ValueError("--ny-bottom + --ny-top must equal --ny when any override is used.")
    else:
        ny_bottom, ny_top = _auto_split_counts(int(args.ny), (block_h, H - block_h), min_each=2)

    # ------------------------------------------------------------------
    # Mesh + tags
    # ------------------------------------------------------------------
    nodes, elems, corners = _channel_minus_block_mesh(
        L=L,
        H=H,
        block_x0=block_x0,
        block_x1=block_x1,
        block_h=block_h,
        nx_left=nx_left,
        nx_mid=nx_mid,
        nx_right=nx_right,
        ny_bottom=ny_bottom,
        ny_top=ny_top,
        poly_order=2,
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )

    if bool(getattr(args, "refine_biofilm", False)):
        pad = max(0.0, float(getattr(args, "refine_biofilm_pad", 0.0)))
        levels = int(max(1, int(getattr(args, "refine_biofilm_levels", 1))))
        if alpha0_kind == "block":
            x_min = float(bio_x0)
            x_max = float(bio_x0 + float(args.bio_w))
            y_min = float(bio_y0)
            y_max = float(bio_y0 + float(args.bio_h))
        else:
            if poly_m is None:
                raise ValueError("poly_m must be available when alpha0_kind='polygon'.")
            x_min = float(np.min(poly_m[:, 0]))
            x_max = float(np.max(poly_m[:, 0]))
            y_min = float(np.min(poly_m[:, 1]))
            y_max = float(np.max(poly_m[:, 1]))

        x0_ref = max(0.0, x_min - pad)
        x1_ref = min(float(L), x_max + pad)
        y0_ref = max(0.0, y_min - pad)
        y1_ref = min(float(H), y_max + pad)

        corners_all = np.asarray(mesh.nodes_x_y_pos[np.asarray(mesh.corner_connectivity, dtype=int)], dtype=float)  # (nE,4,2)
        ex_min = corners_all[..., 0].min(axis=1)
        ex_max = corners_all[..., 0].max(axis=1)
        ey_min = corners_all[..., 1].min(axis=1)
        ey_max = corners_all[..., 1].max(axis=1)
        marked = np.nonzero((ex_max >= x0_ref) & (ex_min <= x1_ref) & (ey_max >= y0_ref) & (ey_min <= y1_ref))[0]
        n_marked = int(marked.size)
        if n_marked > 0:
            from pycutfem.utils.refinement import TensorRefiner

            refiner = TensorRefiner(max_ratio=2.0, max_ref=int(levels))
            rx = np.zeros(len(mesh.elements_list), dtype=int)
            ry = np.zeros(len(mesh.elements_list), dtype=int)
            rx[marked] = int(levels)
            ry[marked] = int(levels)

            # Enforce 2:1 balance on planned split counts so each shared parent edge
            # has at most a single hanging node (required by the hanging-node solver).
            if int(levels) > 1:
                max_ref = int(levels)

                def _balance_splits_2to1(mesh_in: Mesh, rx_in: np.ndarray, ry_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                    rx_b = np.minimum(np.asarray(rx_in, dtype=int).copy(), max_ref)
                    ry_b = np.minimum(np.asarray(ry_in, dtype=int).copy(), max_ref)
                    changed = True
                    while changed:
                        changed = False
                        for eid, elem in enumerate(mesh_in.elements_list):
                            for lid, nb in elem.neighbors.items():
                                if nb is None:
                                    continue
                                nb = int(nb)
                                if lid in (0, 2):  # bottom/top share a horizontal edge → balance x-splits
                                    a = int(rx_b[eid])
                                    b = int(rx_b[nb])
                                    if abs(a - b) > 1:
                                        fine = max(a, b)
                                        if a < b:
                                            rx_b[eid] = min(max_ref, fine - 1)
                                        else:
                                            rx_b[nb] = min(max_ref, fine - 1)
                                        changed = True
                                else:  # left/right share a vertical edge → balance y-splits
                                    a = int(ry_b[eid])
                                    b = int(ry_b[nb])
                                    if abs(a - b) > 1:
                                        fine = max(a, b)
                                        if a < b:
                                            ry_b[eid] = min(max_ref, fine - 1)
                                        else:
                                            ry_b[nb] = min(max_ref, fine - 1)
                                        changed = True
                    return rx_b, ry_b

                rx, ry = _balance_splits_2to1(mesh, rx, ry)
            mesh0 = mesh
            mesh = refiner.refine(mesh0, rx, ry)
            n_hang = int(len(getattr(mesh, "hanging_nodes", []) or []))
            print(
                f"[setup] refined mesh near biofilm: levels={int(levels)} marked={n_marked}/{len(mesh0.elements_list)} elems in "
                f"[{x0_ref:.3e},{x1_ref:.3e}]x[{y0_ref:.3e},{y1_ref:.3e}] m -> "
                f"n_elem {len(mesh0.elements_list)}→{len(mesh.elements_list)}, n_hanging={n_hang}",
                flush=True,
            )
        else:
            print(
                f"[setup] refine_biofilm: no elements intersect refinement box "
                f"[{x0_ref:.3e},{x1_ref:.3e}]x[{y0_ref:.3e},{y1_ref:.3e}] m; skipping refinement.",
                flush=True,
            )

    _tag_channel_with_block_boundaries(mesh, L=L, H=H, block_x0=block_x0, block_x1=block_x1, block_h=block_h)

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
    }
    if alpha_cahn_conservative:
        field_specs["lambda_alpha"] = ":number:"
    if ch_enabled:
        field_specs["mu_alpha"] = 1
    field_specs["S"] = 1

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
    dlambda_alpha = TrialFunction("lambda_alpha", dof_handler=dh) if alpha_cahn_conservative else None
    dmu_alpha = TrialFunction("mu_alpha", dof_handler=dh) if ch_enabled else None
    dS = TrialFunction("S", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    vS_test = VectorTestFunction(space=VS, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    lambda_alpha_test = TestFunction("lambda_alpha", dof_handler=dh) if alpha_cahn_conservative else None
    mu_alpha_test = TestFunction("mu_alpha", dof_handler=dh) if ch_enabled else None
    S_test = TestFunction("S", dof_handler=dh)

    # Unknowns (k) and previous state (n)
    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    vS_k = VectorFunction("vS_k", ["vS_x", "vS_y"], dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    lambda_alpha_k = Function("lambda_alpha_k", "lambda_alpha", dof_handler=dh) if alpha_cahn_conservative else None
    mu_alpha_k = Function("mu_alpha_k", "mu_alpha", dof_handler=dh) if ch_enabled else None
    S_k = Function("S_k", "S", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    vS_n = VectorFunction("vS_n", ["vS_x", "vS_y"], dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    lambda_alpha_n = Function("lambda_alpha_n", "lambda_alpha", dof_handler=dh) if alpha_cahn_conservative else None
    mu_alpha_n = Function("mu_alpha_n", "mu_alpha", dof_handler=dh) if ch_enabled else None
    S_n = Function("S_n", "S", dof_handler=dh)

    # ------------------------------------------------------------------
    # Initial conditions (t=0)
    # ------------------------------------------------------------------
    v_n.nodal_values[:] = 0.0
    p_n.nodal_values[:] = 0.0
    vS_n.nodal_values[:] = 0.0
    u_n.nodal_values[:] = 0.0
    S_n.nodal_values[:] = 1.0

    alpha_xy = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    if alpha0_kind == "block":
        alpha0 = _alpha_rect_eval(
            alpha_xy[:, 0],
            alpha_xy[:, 1],
            x0=bio_x0,
            y0=bio_y0,
            w=float(args.bio_w),
            h=float(args.bio_h),
            eps=eps0,
        )
        y_base = float(bio_y0)
        y_top = float(bio_y0 + float(args.bio_h))
    elif alpha0_kind == "polygon":
        if poly_m is None:
            raise ValueError("poly_m must be available when alpha0_kind='polygon'.")
        alpha0 = _smooth_step((-_signed_distance_polygon(alpha_xy[:, 0], alpha_xy[:, 1], poly_m)) / max(1.0e-12, eps0))
        y_base = float(np.min(poly_m[:, 1]))
        y_top = float(np.max(poly_m[:, 1]))
    else:
        raise ValueError(f"Unknown alpha0-kind: {alpha0_kind}")
    alpha_n.nodal_values[:] = np.asarray(alpha0, dtype=float)
    phi_b = float(args.phi_b)
    phi_n.nodal_values[:] = 1.0 - (1.0 - phi_b) * np.asarray(alpha0, dtype=float)
    if alpha_cahn_conservative:
        lambda_alpha_n.nodal_values[:] = 0.0
        lambda_alpha_k.nodal_values[:] = lambda_alpha_n.nodal_values
    if ch_enabled:
        mu_alpha_n.nodal_values[:] = 0.0

    # We never solve substrate in this benchmark (no growth/chemistry).
    if transport_mode == "refmap":
        # Freeze transport fields: alpha/phi/S are updated (alpha from refmap, phi tied to alpha) but not solved.
        _mark_inactive_fields(dh, "alpha", "phi", "S")
    else:
        # Full solve of alpha/phi; keep substrate frozen at S=1.
        _mark_inactive_fields(dh, "S")
        if bool(getattr(args, "alpha_clip_below_block", False)):
            # For this step-channel geometry, the region y<block_h exists only to the left/right of the
            # rigid support. Numerically, CH regularization can pull alpha down the vertical block walls
            # (corner wetting). The Lie benchmark assumes the biofilm stays on the support top, so we
            # clip alpha out of this floor region by freezing alpha=0 (phi=1) below block_h.
            tol_y = 1.0e-12

            mask_alpha = np.asarray(alpha_xy[:, 1], dtype=float) < (float(block_h) - tol_y)
            if np.any(mask_alpha):
                alpha_n.nodal_values[mask_alpha] = 0.0
                alpha_sl = np.asarray(dh.get_field_slice("alpha"), dtype=int).ravel()
                if alpha_sl.size != mask_alpha.size:
                    raise RuntimeError("Unexpected alpha DOF count mismatch for --alpha-clip-below-block.")
                _mark_inactive_dofs(dh, alpha_sl[mask_alpha])

            phi_xy = np.asarray(dh.get_dof_coords("phi"), dtype=float)
            mask_phi = np.asarray(phi_xy[:, 1], dtype=float) < (float(block_h) - tol_y)
            if np.any(mask_phi):
                phi_n.nodal_values[mask_phi] = 1.0
                phi_sl = np.asarray(dh.get_field_slice("phi"), dtype=int).ravel()
                if phi_sl.size != mask_phi.size:
                    raise RuntimeError("Unexpected phi DOF count mismatch for --alpha-clip-below-block.")
                _mark_inactive_dofs(dh, phi_sl[mask_phi])

            if ch_enabled:
                mu_xy = np.asarray(dh.get_dof_coords("mu_alpha"), dtype=float)
                mask_mu = np.asarray(mu_xy[:, 1], dtype=float) < (float(block_h) - tol_y)
                if np.any(mask_mu):
                    mu_alpha_n.nodal_values[mask_mu] = 0.0
                    mu_sl = np.asarray(dh.get_field_slice("mu_alpha"), dtype=int).ravel()
                    if mu_sl.size != mask_mu.size:
                        raise RuntimeError("Unexpected mu_alpha DOF count mismatch for --alpha-clip-below-block.")
                    _mark_inactive_dofs(dh, mu_sl[mask_mu])

        if bool(getattr(args, "alpha_pin_block_top", False)):
            # Make the initial alpha satisfy the block-top pinning. This is important for
            # step=0000 overlays/diagnostics and also helps Newton robustness by starting
            # from a state consistent with the enforced no-detachment assumption.
            alpha_pin_val = float(getattr(args, "alpha_pin_block_top_value", float("nan")))
            alpha_pin_val_is_const = bool(np.isfinite(alpha_pin_val))
            alpha_pin_a0_min = float(getattr(args, "alpha_pin_block_top_alpha0_min", 0.9))
            if alpha_pin_val_is_const:
                if alpha0_kind == "block":
                    y_span = float(args.bio_h)
                else:
                    y_span = float(np.max(poly_m[:, 1]) - np.min(poly_m[:, 1]))
                y_probe = min(2.0 * float(eps0), 0.25 * y_span) if y_span > 0.0 else float(eps0)
                y_probe = float(max(1.0e-12, y_probe))

                tol_y = 1.0e-12
                mask_top = (
                    (np.abs(np.asarray(alpha_xy[:, 1], dtype=float) - float(block_h)) <= tol_y)
                    & (np.asarray(alpha_xy[:, 0], dtype=float) >= float(block_x0) - tol_y)
                    & (np.asarray(alpha_xy[:, 0], dtype=float) <= float(block_x1) + tol_y)
                )
                if np.any(mask_top):
                    y0 = float(block_h)
                    y_off = min(float(y_probe), max(0.0, float(H) - y0))
                    x_top_nodes = np.asarray(alpha_xy[mask_top, 0], dtype=float)
                    y_top_nodes = np.asarray(alpha_xy[mask_top, 1], dtype=float)
                    if alpha0_kind == "block":
                        a_probe = _alpha_rect_eval(
                            x_top_nodes,
                            y_top_nodes + float(y_off),
                            x0=bio_x0,
                            y0=bio_y0,
                            w=float(args.bio_w),
                            h=float(args.bio_h),
                            eps=eps0,
                        )
                    else:
                        a_probe = _smooth_step(
                            (-_signed_distance_polygon(x_top_nodes, y_top_nodes + float(y_off), poly_m)) / max(1.0e-12, eps0)
                        )
                    a_probe = np.asarray(a_probe, dtype=float).ravel()
                    apply = a_probe >= float(alpha_pin_a0_min)
                    if np.any(apply):
                        a_vals = np.asarray(alpha_n.nodal_values, dtype=float).ravel()
                        idx = np.where(mask_top)[0]
                        a_vals[idx[apply]] = float(np.clip(float(alpha_pin_val), 0.0, 1.0))
                        alpha_n.nodal_values[:] = a_vals

            # Keep phi consistent with the modified alpha.
            phi_n.nodal_values[:] = 1.0 - (1.0 - phi_b) * np.asarray(alpha_n.nodal_values, dtype=float)

    # ------------------------------------------------------------------
    # Alpha-from-refmap mapping: alpha(x,t) = alpha0(x - u(x,t))
    # Build alpha->u DOF lookup via coordinate matching (Q1 alpha is a subset of Q2 u).
    # ------------------------------------------------------------------
    if transport_mode == "refmap":
        ux_xy = np.asarray(dh.get_dof_coords("u_x"), dtype=float)
        ux_lut = _build_coord_lookup(ux_xy, ndigits=12)
        alpha_to_u = np.full(alpha_xy.shape[0], -1, dtype=int)
        for i, xy in enumerate(alpha_xy):
            key = (round(float(xy[0]), 12), round(float(xy[1]), 12))
            j = ux_lut.get(key, None)
            if j is not None:
                alpha_to_u[i] = int(j)
        if np.any(alpha_to_u < 0):
            missing = int(np.sum(alpha_to_u < 0))
            raise RuntimeError(f"Failed mapping {missing} alpha DOFs to u DOFs; cannot use refmap alpha transport.")

        def _post_step_update() -> None:
            ux = np.asarray(u_k.components[0].nodal_values, dtype=float).ravel()
            uy = np.asarray(u_k.components[1].nodal_values, dtype=float).ravel()
            u_at_alpha = np.column_stack([ux[alpha_to_u], uy[alpha_to_u]])
            chi = alpha_xy - u_at_alpha
            if alpha0_kind == "block":
                a = _alpha_rect_eval(
                    chi[:, 0],
                    chi[:, 1],
                    x0=bio_x0,
                    y0=bio_y0,
                    w=float(args.bio_w),
                    h=float(args.bio_h),
                    eps=eps0,
                )
            else:
                a = _smooth_step((-_signed_distance_polygon(chi[:, 0], chi[:, 1], poly_m)) / max(1.0e-12, eps0))
            alpha_k.nodal_values[:] = np.clip(np.asarray(a, dtype=float), 0.0, 1.0)
            phi_k.nodal_values[:] = np.clip(1.0 - (1.0 - phi_b) * np.asarray(alpha_k.nodal_values, dtype=float), 0.0, 1.0)
            # Keep frozen substrate non-negative.
            S_k.nodal_values[:] = np.maximum(np.asarray(S_k.nodal_values, dtype=float), 0.0)
    else:
        def _post_step_update() -> None:
            # Nothing to do in PDE mode; alpha/phi are solved and S is frozen.
            return

    # Optionally restrict skeleton DOFs (u,vS) to a fixed region to avoid extension
    # penalties dominating the solve when the fluid region is much larger.
    if bool(args.restrict_skeleton_dofs):
        method = str(args.restrict_skeleton_method).strip().lower()
        if method == "alpha":
            has_box_bounds = any(
                np.isfinite(float(getattr(args, k, float("nan"))))
                for k in (
                    "restrict_skeleton_box_x0",
                    "restrict_skeleton_box_x1",
                    "restrict_skeleton_box_y0",
                    "restrict_skeleton_box_y1",
                )
            )
            if has_box_bounds:
                print(
                    "    [warn] --restrict-skeleton-box-* bounds are ignored for --restrict-skeleton-method alpha.",
                    flush=True,
                )
            _restrict_skeleton_dofs_to_alpha(
                dh,
                alpha0_kind=alpha0_kind,
                eps0=float(args.eps),
                poly_m=poly_m if alpha0_kind == "polygon" else None,
                bio_x0=float(bio_x0),
                bio_y0=float(bio_y0),
                bio_w=float(args.bio_w),
                bio_h=float(args.bio_h),
                alpha_min=float(args.restrict_skeleton_alpha_min),
            )
        elif method == "box":
            if float(getattr(args, "restrict_skeleton_alpha_min", 0.05)) != 0.05:
                print(
                    "    [warn] --restrict-skeleton-alpha-min is ignored for --restrict-skeleton-method box. "
                    "Use --restrict-skeleton-method alpha to activate alpha-threshold restriction.",
                    flush=True,
                )
            x0_box = float(getattr(args, "restrict_skeleton_box_x0", float("nan")))
            x1_box = float(getattr(args, "restrict_skeleton_box_x1", float("nan")))
            y0_box = float(getattr(args, "restrict_skeleton_box_y0", float("nan")))
            y1_box = float(getattr(args, "restrict_skeleton_box_y1", float("nan")))
            if np.isfinite(x0_box) or np.isfinite(x1_box) or np.isfinite(y0_box) or np.isfinite(y1_box):
                if not (np.isfinite(x0_box) and np.isfinite(x1_box) and np.isfinite(y0_box) and np.isfinite(y1_box)):
                    raise ValueError(
                        "Explicit --restrict-skeleton-box-* requires all of x0,x1,y0,y1 to be set."
                    )
            else:
                pad = float(args.restrict_skeleton_box_pad)
                if alpha0_kind == "block":
                    x_min = float(bio_x0)
                    x_max = float(bio_x0 + float(args.bio_w))
                    y_min = float(bio_y0)
                    y_max = float(bio_y0 + float(args.bio_h))
                else:
                    if poly_m is None:
                        raise ValueError("poly_m must be available when alpha0_kind='polygon'.")
                    x_min = float(np.min(poly_m[:, 0]))
                    x_max = float(np.max(poly_m[:, 0]))
                    y_min = float(np.min(poly_m[:, 1]))
                    y_max = float(np.max(poly_m[:, 1]))
                x0_box = max(0.0, x_min - pad)
                x1_box = min(float(L), x_max + pad)
                # Don't activate skeleton DOFs below the rigid support top.
                y0_box = max(float(block_h), y_min - pad)
                y1_box = min(float(H), y_max + pad)

            n_keep, n_tot = _restrict_skeleton_dofs_to_box(dh, x0=x0_box, x1=x1_box, y0=y0_box, y1=y1_box)
            print(
                f"[setup] restrict_skeleton_method='box': active Q2 nodes {n_keep}/{n_tot} "
                f"in [{x0_box:.3e},{x1_box:.3e}]x[{y0_box:.3e},{y1_box:.3e}] m",
                flush=True,
            )
        else:
            raise ValueError(f"Unknown --restrict-skeleton-method {method!r}.")

    # ------------------------------------------------------------------
    # Tracking lines (match the paper's "lines 1-3" concept)
    # ------------------------------------------------------------------
    dx_q = float(np.clip(float(args.dx_quantile), 0.0, 1.0))
    a0_vals = np.asarray(alpha_n.nodal_values, dtype=float).ravel()
    y_min = float(y_base)
    y_max = float(y_top)
    hb = max(1.0e-12, y_max - y_min)
    if str(args.y_fracs).strip():
        y_fracs = _parse_float_list(str(args.y_fracs), n=3)
        for f in y_fracs:
            if not (0.0 < float(f) < 1.0):
                raise ValueError(f"--y-fracs must be fractions in (0,1); got {y_fracs}")
        y_fracs = sorted([float(v) for v in y_fracs], reverse=True)
    else:
        y_fracs = [0.75, 0.50, 0.25]
    # Follow the paper's convention (Fig. 7a): line 1 is the topmost line, line 3 is the lowest.
    y_targets = [y_min + float(f) * hb for f in y_fracs]
    # Choose tracking lines from available alpha DOF y-levels in the biofilm span,
    # excluding the clamped base y=y_base (avoid ambiguous/CH-smoothed contact line).
    y_src = alpha_xy[:, 1]
    y_src = y_src[(y_src > float(y_base) + 1.0e-14) & (y_src < float(y_top) - 1.0e-14)]
    if y_src.size == 0:
        y_src = alpha_xy[:, 1]
    y_lines = _nearest_y_levels(y_src, y_targets)
    x_ref = [_x_alpha_half_quantile_on_y_line(alpha_xy, a0_vals, y_line=y0, mode=str(args.dx_intersection), q=dx_q) for y0 in y_lines]
    lines_data = list(zip(y_targets, y_lines, x_ref))

    # ------------------------------------------------------------------
    # Forms
    # ------------------------------------------------------------------
    mu_s = float(args.G_b)
    lam_s = _lame_lambda_from_nu(mu_s, float(args.nu))
    rho_f = float(args.rho_f)
    mu_f = float(args.mu_f)

    # Paper check (informational): Re=rho*u_avg*H/mu
    Re = (rho_f * u_avg * H) / max(1.0e-16, mu_f)
    print(f"[setup] H={H:.3e} m, u_avg={u_avg:.3e} m/s -> Re={Re:.3g}", flush=True)

    dt_c = Constant(dt_val)

    forms = build_biofilm_one_domain_forms(
        v_k=v_k,
        p_k=p_k,
        vS_k=vS_k,
        u_k=u_k,
        phi_k=phi_k,
        alpha_k=alpha_k,
        mu_alpha_k=mu_alpha_k,
        S_k=S_k,
        lambda_alpha_k=lambda_alpha_k,
        v_n=v_n,
        p_n=p_n,
        vS_n=vS_n,
        u_n=u_n,
        phi_n=phi_n,
        alpha_n=alpha_n,
        mu_alpha_n=mu_alpha_n,
        S_n=S_n,
        lambda_alpha_n=lambda_alpha_n,
        dv=dv,
        dp=dp,
        dvS=dvS,
        du=du,
        dphi=dphi,
        dalpha=dalpha,
        dmu_alpha=dmu_alpha,
        dS=dS,
        dlambda_alpha=dlambda_alpha,
        v_test=v_test,
        q_test=q_test,
        vS_test=vS_test,
        u_test=u_test,
        phi_test=phi_test,
        alpha_test=alpha_test,
        mu_alpha_test=mu_alpha_test,
        S_test=S_test,
        lambda_alpha_test=lambda_alpha_test,
        dx=dx(metadata={"q": int(qdeg)}),
        ds_cip=ds(metadata={"q": int(qdeg)}),
        dt=dt_c,
        theta=theta,
        rho_f=Constant(rho_f),
        mu_f=Constant(mu_f),
        mu_b=Constant(float(args.mu_b_fluid)),
        kappa_inv=Constant(float(args.kappa_inv)),
        mu_b_model=str(args.mu_b_model),
        solid_model=str(args.solid_model),
        mu_s=Constant(mu_s),
        lambda_s=Constant(lam_s),
        solid_visco_eta=float(args.mu_b),
        # Extension penalties so (u,vS) remain well-posed in the free fluid when alpha is small.
        gamma_u=float(args.gamma_u),
        u_extension_mode=str(args.u_extension),
        gamma_u_pin=float(args.gamma_u_pin),
        # Transport/kinetics controls (we disable growth/chemistry, but may solve alpha/phi).
        D_phi=0.0,
        gamma_phi=float(args.gamma_phi),
        D_alpha=float(args.D_alpha) if transport_mode == "pde" else 0.0,
        alpha_advection_form=str(args.alpha_advection_form) if transport_mode == "pde" else "advective",
        alpha_ch_M=float(alpha_ch_M) if transport_mode == "pde" else 0.0,
        alpha_ch_gamma=float(alpha_ch_gamma) if transport_mode == "pde" else 0.0,
        alpha_ch_eps=float(alpha_ch_eps) if transport_mode == "pde" else 1.0,
        alpha_ch_mobility=str(args.alpha_ch_mobility),
        alpha_cahn_M=float(alpha_cahn_M) if transport_mode == "pde" else 0.0,
        alpha_cahn_gamma=float(alpha_cahn_gamma) if transport_mode == "pde" else 0.0,
        alpha_cahn_eps=float(alpha_cahn_eps) if transport_mode == "pde" else 1.0,
        alpha_cahn_conservative=bool(alpha_cahn_conservative) if transport_mode == "pde" else False,
        alpha_cahn_mobility=str(args.alpha_cahn_mobility),
        alpha_supg=float(args.alpha_supg) if transport_mode == "pde" else 0.0,
        alpha_cip=float(args.alpha_cip) if transport_mode == "pde" else 0.0,
        D_S=0.0,
        mu_max=0.0,
        k_g=0.0,
        k_d=0.0,
        k_det=0.0,
        dim=2,
    )

    # ------------------------------------------------------------------
    # Boundary conditions (channel)
    # ------------------------------------------------------------------
    def ramp(t: float) -> float:
        t = float(t)
        if t <= 0.0:
            return 0.0
        if t >= t_ramp:
            return 1.0
        return t / t_ramp

    def inflow_vx(_x, y, t):
        yy = float(y) / H
        return float(Umax * ramp(t) * 4.0 * yy * (1.0 - yy))

    bcs = []
    bcs.append(BoundaryCondition("v_x", "dirichlet", "left", inflow_vx))
    bcs.append(BoundaryCondition("v_y", "dirichlet", "left", lambda x, y, t: 0.0))
    for tag in ("bottom", "top"):
        bcs.append(BoundaryCondition("v_x", "dirichlet", tag, lambda x, y, t: 0.0))
        bcs.append(BoundaryCondition("v_y", "dirichlet", tag, lambda x, y, t: 0.0))
    for tag in ("block_top", "block_left", "block_right"):
        bcs.append(BoundaryCondition("v_x", "dirichlet", tag, lambda x, y, t: 0.0))
        bcs.append(BoundaryCondition("v_y", "dirichlet", tag, lambda x, y, t: 0.0))

    # Clamp the biofilm at the top of the rigid support.
    bcs.append(BoundaryCondition("u_x", "dirichlet", "block_top", lambda x, y, t: 0.0))
    bcs.append(BoundaryCondition("u_y", "dirichlet", "block_top", lambda x, y, t: 0.0))
    bcs.append(BoundaryCondition("vS_x", "dirichlet", "block_top", lambda x, y, t: 0.0))
    bcs.append(BoundaryCondition("vS_y", "dirichlet", "block_top", lambda x, y, t: 0.0))
    if transport_mode == "pde":
        alpha_pin = bool(getattr(args, "alpha_pin_block_top", False))
        alpha_zero_sides = bool(getattr(args, "alpha_zero_block_sides", False))
        if alpha_pin or alpha_zero_sides:
            if alpha0_kind == "polygon" and poly_m is None:
                raise ValueError("poly_m must be available when alpha0_kind='polygon'.")

            def _alpha0_at_xy(x, y) -> float:
                x0 = float(x)
                y0 = float(y)
                if alpha0_kind == "block":
                    a = _alpha_rect_eval(
                        np.asarray([x0], dtype=float),
                        np.asarray([y0], dtype=float),
                        x0=float(bio_x0),
                        y0=float(bio_y0),
                        w=float(args.bio_w),
                        h=float(args.bio_h),
                        eps=float(eps0),
                    )[0]
                else:
                    a = _smooth_step(
                        (-_signed_distance_polygon(np.asarray([x0], dtype=float), np.asarray([y0], dtype=float), poly_m))
                        / max(1.0e-12, float(eps0))
                    )[0]
                return float(np.clip(float(a), 0.0, 1.0))

            alpha_pin_val = float(getattr(args, "alpha_pin_block_top_value", float("nan")))
            alpha_pin_val_is_const = bool(np.isfinite(alpha_pin_val))
            alpha_pin_a0_min = float(getattr(args, "alpha_pin_block_top_alpha0_min", 0.9))

            # When alpha0 is built from a traced polygon, the *block_top* segment is part of the
            # traced boundary. With the smooth-step construction, boundary points have alpha0≈0.5.
            # Therefore, deciding whether a block-top node is "inside" the attached footprint by
            # looking at alpha0(x,block_h) will (almost) never exceed a high threshold like 0.9.
            #
            # To make `--alpha-pin-block-top-value` behave as intended (alpha=1 on the attached
            # footprint, alpha=alpha0 elsewhere), we evaluate alpha0 slightly *above* the block top
            # and apply the constant value when that probe point is inside (alpha0 high).
            if alpha0_kind == "block":
                y_span = float(args.bio_h)
            else:
                y_span = float(np.max(poly_m[:, 1]) - np.min(poly_m[:, 1]))
            y_probe = min(2.0 * float(eps0), 0.25 * y_span) if y_span > 0.0 else float(eps0)
            y_probe = float(max(1.0e-12, y_probe))

            def _alpha_top_bc_value(x, y) -> float:
                a0 = _alpha0_at_xy(x, y)
                if alpha_pin_val_is_const:
                    y0 = float(y)
                    # Clamp the probe coordinate inside the channel.
                    y_off = min(float(y_probe), max(0.0, float(H) - y0))
                    a0_probe = _alpha0_at_xy(x, y0 + y_off) if y_off > 0.0 else float(a0)
                    if float(a0_probe) >= float(alpha_pin_a0_min):
                        return float(np.clip(float(alpha_pin_val), 0.0, 1.0))
                return float(a0)

            if alpha_pin_val_is_const:
                # Emit a one-time warning if the constant value is effectively never applied.
                try:
                    xs = np.linspace(float(block_x0), float(block_x1), 25, dtype=float)
                    y0 = float(block_h)
                    y_off = min(float(y_probe), max(0.0, float(H) - y0))
                    a_probe = np.asarray([_alpha0_at_xy(x, y0 + y_off) for x in xs], dtype=float)
                    n_const = int(np.sum(a_probe >= float(alpha_pin_a0_min)))
                    if n_const == 0:
                        print(
                            f"    [warn] --alpha-pin-block-top-value requested, but alpha0 probe on block_top never reaches "
                            f"--alpha-pin-block-top-alpha0-min={alpha_pin_a0_min:g} (probe offset={y_off:.2e} m). "
                            "The constant pin value will never be applied; consider lowering --alpha-pin-block-top-alpha0-min.",
                            flush=True,
                        )
                except Exception:
                    pass

            if alpha_pin:
                bcs.append(BoundaryCondition("alpha", "dirichlet", "block_top", lambda x, y, t: _alpha_top_bc_value(x, y)))
            if alpha_zero_sides:
                tol_y = 1.0e-12

                def _alpha_side_bc(x, y, t):
                    # Avoid conflicting Dirichlet data at the top corner node:
                    # - on block_top we prescribe alpha=alpha0(x,y) (or a constant value),
                    # - on the vertical sides we prescribe alpha=0 except at y=block_h.
                    return _alpha_top_bc_value(x, y) if abs(float(y) - float(block_h)) <= tol_y else 0.0

                for tag in ("block_left", "block_right"):
                    bcs.append(BoundaryCondition("alpha", "dirichlet", tag, _alpha_side_bc))

    # Outlet: pin pressure.
    bcs.append(BoundaryCondition("p", "dirichlet", "right", lambda x, y, t: 0.0))
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, lambda x, y: 0.0) for b in bcs]

    # ------------------------------------------------------------------
    # Output setup
    # ------------------------------------------------------------------
    out_dir = Path(str(args.out_dir))
    vtk_dir = out_dir / "vtk"
    vtk_every = int(args.vtk_every)
    out_dir.mkdir(parents=True, exist_ok=True)
    if vtk_every > 0:
        vtk_dir.mkdir(parents=True, exist_ok=True)

    lines_path = out_dir / "lines.csv"
    with lines_path.open("w", encoding="utf-8") as f_lines:
        f_lines.write("line_id,y_target_m,y_line_m,x_ref_m\n")
        for i, (yt, yl, xr) in enumerate(lines_data, start=1):
            f_lines.write(f"{i},{float(yt):.12e},{float(yl):.12e},{float(xr):.12e}\n")

    vtk_stress_enabled = bool(args.vtk_stress) and vtk_every > 0
    stress_conn_all = None
    stress_corner_conn = None
    stress_hx = None
    stress_hy = None
    stress_num_nodes = int(mesh.nodes_x_y_pos.shape[0])
    stress_vx_node_ids = stress_vx_lidx = None
    stress_vy_node_ids = stress_vy_lidx = None
    stress_vSx_node_ids = stress_vSx_lidx = None
    stress_vSy_node_ids = stress_vSy_lidx = None
    stress_ux_node_ids = stress_ux_lidx = None
    stress_uy_node_ids = stress_uy_lidx = None
    stress_alpha_node_ids = stress_alpha_lidx = None
    stress_phi_node_ids = stress_phi_lidx = None

    if vtk_stress_enabled:
        try:
            dh._ensure_node_maps()
        except Exception:
            pass

        stress_conn_all = np.asarray(getattr(mesh, "elements_connectivity", None), dtype=int)
        stress_corner_conn = np.asarray(getattr(mesh, "corner_connectivity", None), dtype=int)
        if stress_conn_all.ndim != 2 or stress_conn_all.shape[1] != 9:
            raise RuntimeError("VTK stress export currently requires a Q2 quadrilateral mesh (elements_connectivity: n_cells×9).")
        if stress_corner_conn.ndim != 2 or stress_corner_conn.shape[1] != 4:
            raise RuntimeError("VTK stress export expects quadrilateral corner connectivity (n_cells×4).")
        if stress_conn_all.shape[0] != stress_corner_conn.shape[0]:
            raise RuntimeError("Mismatch between elements_connectivity and corner_connectivity cell counts.")

        xy_nodes = np.asarray(mesh.nodes_x_y_pos, dtype=float)
        x_nodes = xy_nodes[:, 0]
        y_nodes = xy_nodes[:, 1]
        bl = stress_conn_all[:, 0]
        br = stress_conn_all[:, 2]
        tl = stress_conn_all[:, 6]
        stress_hx = np.maximum(np.abs(x_nodes[br] - x_nodes[bl]), 1.0e-30)
        stress_hy = np.maximum(np.abs(y_nodes[tl] - y_nodes[bl]), 1.0e-30)

        def _inject_map_for_vector(vf: VectorFunction, field: str) -> tuple[np.ndarray, np.ndarray]:
            gdofs = np.asarray(dh.get_field_slice(field), dtype=int).ravel()
            node_ids: list[int] = []
            lidxs: list[int] = []
            for gd in gdofs.tolist():
                if gd not in vf._g2l:
                    continue
                try:
                    _fld, nid = dh._dof_to_node_map[int(gd)]
                except Exception:
                    nid = None
                if nid is None:
                    continue
                node_ids.append(int(nid))
                lidxs.append(int(vf._g2l[int(gd)]))
            return np.asarray(node_ids, dtype=int), np.asarray(lidxs, dtype=int)

        def _inject_map_for_scalar(f: Function, field: str) -> tuple[np.ndarray, np.ndarray]:
            gdofs = np.asarray(dh.get_field_slice(field), dtype=int).ravel()
            node_ids: list[int] = []
            lidxs: list[int] = []
            for gd in gdofs.tolist():
                if gd not in f._g2l:
                    continue
                try:
                    _fld, nid = dh._dof_to_node_map[int(gd)]
                except Exception:
                    nid = None
                if nid is None:
                    continue
                node_ids.append(int(nid))
                lidxs.append(int(f._g2l[int(gd)]))
            return np.asarray(node_ids, dtype=int), np.asarray(lidxs, dtype=int)

        stress_vx_node_ids, stress_vx_lidx = _inject_map_for_vector(v_k, "v_x")
        stress_vy_node_ids, stress_vy_lidx = _inject_map_for_vector(v_k, "v_y")
        stress_vSx_node_ids, stress_vSx_lidx = _inject_map_for_vector(vS_k, "vS_x")
        stress_vSy_node_ids, stress_vSy_lidx = _inject_map_for_vector(vS_k, "vS_y")
        stress_ux_node_ids, stress_ux_lidx = _inject_map_for_vector(u_k, "u_x")
        stress_uy_node_ids, stress_uy_lidx = _inject_map_for_vector(u_k, "u_y")
        stress_alpha_node_ids, stress_alpha_lidx = _inject_map_for_scalar(alpha_k, "alpha")
        stress_phi_node_ids, stress_phi_lidx = _inject_map_for_scalar(phi_k, "phi")

        mu_b_model_key = str(args.mu_b_model).strip().lower()
        if mu_b_model_key in {"mu", "const", "constant"}:
            mu_b_model_key = "mu"
        elif mu_b_model_key in {"phi_mu", "phi*mu", "phi"}:
            mu_b_model_key = "phi_mu"
        elif mu_b_model_key in {"alpha_mu", "alpha", "mu_b", "biofilm_mu", "biofilm"}:
            mu_b_model_key = "alpha_mu"
        elif mu_b_model_key in {"alpha_phi_mu", "alpha_phi", "alpha*phi_mu", "alpha*phi*mu"}:
            mu_b_model_key = "alpha_phi_mu"
        else:
            raise ValueError(f"Unknown mu_b_model {args.mu_b_model!r}.")

        mu_f_val = float(mu_f)
        mu_b_val = float(args.mu_b_fluid)
        mu_s_val = float(mu_s)
        lam_s_val = float(lam_s)
        eta_s_val = float(args.mu_b)

        def _dev_sym_norm(xx: np.ndarray, yy: np.ndarray, xy: np.ndarray) -> np.ndarray:
            tr = xx + yy
            dev_xx = xx - 0.5 * tr
            dev_yy = yy - 0.5 * tr
            return np.sqrt(dev_xx * dev_xx + dev_yy * dev_yy + 2.0 * xy * xy)

        def _cell_center_grad_q2(node_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            left_mid = stress_conn_all[:, 3]
            right_mid = stress_conn_all[:, 5]
            bot_mid = stress_conn_all[:, 1]
            top_mid = stress_conn_all[:, 7]
            df_dx = (node_vals[right_mid] - node_vals[left_mid]) / stress_hx
            df_dy = (node_vals[top_mid] - node_vals[bot_mid]) / stress_hy
            return np.asarray(df_dx, dtype=float), np.asarray(df_dy, dtype=float)

        def _stress_cell_data(vf_v: VectorFunction, vf_vS: VectorFunction, vf_u: VectorFunction, f_alpha: Function, f_phi: Function) -> dict[str, np.ndarray]:
            vx = np.zeros(stress_num_nodes, dtype=float)
            vy = np.zeros(stress_num_nodes, dtype=float)
            vx[stress_vx_node_ids] = np.asarray(vf_v.nodal_values, dtype=float).ravel()[stress_vx_lidx]
            vy[stress_vy_node_ids] = np.asarray(vf_v.nodal_values, dtype=float).ravel()[stress_vy_lidx]
            vSx = np.zeros(stress_num_nodes, dtype=float)
            vSy = np.zeros(stress_num_nodes, dtype=float)
            vSx[stress_vSx_node_ids] = np.asarray(vf_vS.nodal_values, dtype=float).ravel()[stress_vSx_lidx]
            vSy[stress_vSy_node_ids] = np.asarray(vf_vS.nodal_values, dtype=float).ravel()[stress_vSy_lidx]
            ux = np.zeros(stress_num_nodes, dtype=float)
            uy = np.zeros(stress_num_nodes, dtype=float)
            ux[stress_ux_node_ids] = np.asarray(vf_u.nodal_values, dtype=float).ravel()[stress_ux_lidx]
            uy[stress_uy_node_ids] = np.asarray(vf_u.nodal_values, dtype=float).ravel()[stress_uy_lidx]
            alpha_node = np.zeros(stress_num_nodes, dtype=float)
            phi_node = np.zeros(stress_num_nodes, dtype=float)
            alpha_node[stress_alpha_node_ids] = np.asarray(f_alpha.nodal_values, dtype=float).ravel()[stress_alpha_lidx]
            phi_node[stress_phi_node_ids] = np.asarray(f_phi.nodal_values, dtype=float).ravel()[stress_phi_lidx]

            alpha_cell = np.mean(alpha_node[stress_corner_conn], axis=1)
            phi_cell = np.mean(phi_node[stress_corner_conn], axis=1)
            if mu_b_model_key == "mu":
                mu_eff = mu_f_val * np.ones_like(alpha_cell)
            elif mu_b_model_key == "phi_mu":
                mu_eff = mu_f_val * ((1.0 - alpha_cell) + alpha_cell * phi_cell)
            elif mu_b_model_key == "alpha_mu":
                mu_eff = (1.0 - alpha_cell) * mu_f_val + alpha_cell * mu_b_val
            else:
                mu_eff = (1.0 - alpha_cell) * mu_f_val + alpha_cell * phi_cell * mu_b_val

            dvx_dx, dvx_dy = _cell_center_grad_q2(vx)
            dvy_dx, dvy_dy = _cell_center_grad_q2(vy)
            dxx = dvx_dx
            dyy = dvy_dy
            dxy = 0.5 * (dvx_dy + dvy_dx)
            sigma_xx = 2.0 * mu_eff * dxx
            sigma_yy = 2.0 * mu_eff * dyy
            sigma_xy = 2.0 * mu_eff * dxy
            sigma_newt_norm = _dev_sym_norm(sigma_xx, sigma_yy, sigma_xy)

            dvSx_dx, dvSx_dy = _cell_center_grad_q2(vSx)
            dvSy_dx, dvSy_dy = _cell_center_grad_q2(vSy)
            exx = dvSx_dx
            eyy = dvSy_dy
            exy = 0.5 * (dvSx_dy + dvSy_dx)
            tau_xx = 2.0 * eta_s_val * exx
            tau_yy = 2.0 * eta_s_val * eyy
            tau_xy = 2.0 * eta_s_val * exy
            tau_visc_norm = _dev_sym_norm(tau_xx, tau_yy, tau_xy)

            dux_dx, dux_dy = _cell_center_grad_q2(ux)
            duy_dx, duy_dy = _cell_center_grad_q2(uy)
            eps_xx = dux_dx
            eps_yy = duy_dy
            eps_xy = 0.5 * (dux_dy + duy_dx)
            tr_eps = eps_xx + eps_yy
            sig_el_xx = 2.0 * mu_s_val * eps_xx + lam_s_val * tr_eps
            sig_el_yy = 2.0 * mu_s_val * eps_yy + lam_s_val * tr_eps
            sig_el_xy = 2.0 * mu_s_val * eps_xy
            sig_el_norm = _dev_sym_norm(sig_el_xx, sig_el_yy, sig_el_xy)

            return {
                "alpha_cell": np.asarray(alpha_cell, dtype=float),
                "mu_eff_cell": np.asarray(mu_eff, dtype=float),
                "sigma_newtonian_dev_norm_alpha": np.asarray(alpha_cell * sigma_newt_norm, dtype=float),
                "tau_skel_visc_dev_norm_alpha": np.asarray(alpha_cell * tau_visc_norm, dtype=float),
                "sigma_skel_el_dev_norm_alpha": np.asarray(alpha_cell * sig_el_norm, dtype=float),
            }

    # Initial export + time series header
    if vtk_every > 0:
        vtk0 = {"v": v_n, "p": p_n, "vS": vS_n, "u": u_n, "phi": phi_n, "alpha": alpha_n, "S": S_n}
        if ch_enabled:
            vtk0["mu_alpha"] = mu_alpha_n
        export_vtk(
            str(vtk_dir / "step=0000.vtu"),
            mesh,
            dh,
            vtk0,
            cell_data=_stress_cell_data(v_n, vS_n, u_n, alpha_n, phi_n) if vtk_stress_enabled else None,
        )

    ts_path = out_dir / "timeseries.csv"
    with ts_path.open("w", encoding="utf-8") as f_ts:
        f_ts.write("t_s,dx_line1_m,dx_line2_m,dx_line3_m\n")
        f_ts.write("0.0,0.0,0.0,0.0\n")

    alpha_metrics_enabled = bool(args.alpha_metrics)
    alpha_metrics_path = out_dir / "alpha_metrics.csv"
    alpha_corner_conn = None
    alpha_geom_bl = None
    alpha_geom_dx = None
    alpha_geom_dy = None
    alpha_geom_area = None
    alpha_dof_node_ids = None
    alpha_dof_local_idx = None
    if alpha_metrics_enabled:
        try:
            dh._ensure_node_maps()
        except Exception:
            pass

        with alpha_metrics_path.open("w", encoding="utf-8") as f_am:
            f_am.write("t_s,alpha_mass_m2,alpha_x_cm_m,alpha_y_cm_m,alpha_min,alpha_max\n")

        alpha_corner_conn = np.asarray(mesh.corner_connectivity, dtype=int)
        if alpha_corner_conn.ndim != 2 or alpha_corner_conn.shape[1] != 4:
            raise RuntimeError("Expected quadrilateral corner connectivity (n_cells,4).")
        xy_nodes = np.asarray(mesh.nodes_x_y_pos, dtype=float)
        bl_xy = xy_nodes[alpha_corner_conn[:, 0], :]
        br_xy = xy_nodes[alpha_corner_conn[:, 1], :]
        tl_xy = xy_nodes[alpha_corner_conn[:, 3], :]
        dx_cell = np.abs(br_xy[:, 0] - bl_xy[:, 0])
        dy_cell = np.abs(tl_xy[:, 1] - bl_xy[:, 1])
        area_cell = dx_cell * dy_cell
        alpha_geom_bl = bl_xy
        alpha_geom_dx = dx_cell
        alpha_geom_dy = dy_cell
        alpha_geom_area = area_cell

        gdofs = np.asarray(getattr(alpha_k, "_g_dofs", []), dtype=int).ravel()
        node_ids = []
        local_idx = []
        for i, gd in enumerate(gdofs.tolist()):
            try:
                _fld, nid = dh._dof_to_node_map[int(gd)]
            except Exception:
                nid = None
            if nid is None:
                continue
            node_ids.append(int(nid))
            local_idx.append(int(i))
        alpha_dof_node_ids = np.asarray(node_ids, dtype=int)
        alpha_dof_local_idx = np.asarray(local_idx, dtype=int)

        def _alpha_metrics(alpha_f: Function) -> tuple[float, float, float, float, float]:
            vals = np.asarray(alpha_f.nodal_values, dtype=float).ravel()
            a_min = float(np.min(vals)) if vals.size else float("nan")
            a_max = float(np.max(vals)) if vals.size else float("nan")
            n_nodes = int(xy_nodes.shape[0])
            a_node = np.zeros(n_nodes, dtype=float)
            if alpha_dof_node_ids is not None and alpha_dof_local_idx is not None and vals.size:
                a_node[alpha_dof_node_ids] = vals[alpha_dof_local_idx]
            a_bl = a_node[alpha_corner_conn[:, 0]]
            a_br = a_node[alpha_corner_conn[:, 1]]
            a_tr = a_node[alpha_corner_conn[:, 2]]
            a_tl = a_node[alpha_corner_conn[:, 3]]

            a0 = a_bl
            b0 = a_br - a_bl
            c0 = a_tl - a_bl
            d0 = a_tr - a_br - a_tl + a_bl

            I_f = a0 + 0.5 * b0 + 0.5 * c0 + 0.25 * d0
            I_sf = 0.5 * a0 + (1.0 / 3.0) * b0 + 0.25 * c0 + (1.0 / 6.0) * d0
            I_tf = 0.5 * a0 + 0.25 * b0 + (1.0 / 3.0) * c0 + (1.0 / 6.0) * d0

            mass = float(np.sum(alpha_geom_area * I_f))
            if not np.isfinite(mass) or mass <= 1.0e-30:
                return mass, float("nan"), float("nan"), a_min, a_max

            x0 = alpha_geom_bl[:, 0]
            y0 = alpha_geom_bl[:, 1]
            mx = float(np.sum(alpha_geom_area * (x0 * I_f + alpha_geom_dx * I_sf)))
            my = float(np.sum(alpha_geom_area * (y0 * I_f + alpha_geom_dy * I_tf)))
            return mass, float(mx / mass), float(my / mass), a_min, a_max

        def _append_alpha_metrics(t_s: float, alpha_f: Function) -> None:
            mass, xcm, ycm, a_min, a_max = _alpha_metrics(alpha_f)
            with alpha_metrics_path.open("a", encoding="utf-8") as f_am:
                f_am.write(f"{float(t_s):.12e},{mass:.12e},{xcm:.12e},{ycm:.12e},{a_min:.12e},{a_max:.12e}\n")

        _append_alpha_metrics(0.0, alpha_n)

    # Precompute u DOF coordinates and active-DOF mask for tracking interpolation.
    u_coords_all = np.asarray(dh.get_dof_coords("u_x"), dtype=float)
    u_slice_all = np.asarray(dh.get_field_slice("u_x"), dtype=int).ravel()
    inactive = set((getattr(dh, "dof_tags", None) or {}).get("inactive", set()))
    u_active_mask = np.asarray([int(g) not in inactive for g in u_slice_all.tolist()], dtype=bool)
    if int(np.sum(u_active_mask)) < 4:
        u_active_mask[:] = True
    u_coords = np.asarray(u_coords_all[u_active_mask, :], dtype=float)

    def _interp_u_idw(xq: np.ndarray, *, k: int = 8, power: float = 2.0) -> np.ndarray:
        """
        Interpolate displacement u(x,t) at query points using simple IDW on Q2 DOF nodes.

        This avoids requiring a dedicated FE point-evaluation routine and is cheap for the
        3 tracking points used in this benchmark.
        """
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

        # Brute-force kNN for small query count (N=3).
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

    dx_tracking = str(args.dx_tracking).strip().lower()
    if dx_tracking not in {"alpha_half", "lagrangian_u"}:
        raise ValueError(f"Unknown --dx-tracking {args.dx_tracking!r}.")
    fp_iters = int(max(0, int(args.dx_fixed_point_iters)))
    track_ref_xy = np.column_stack([np.asarray(x_ref, dtype=float), np.asarray(y_lines, dtype=float)])  # (3,2)

    def _append_timeseries(t_s: float) -> None:
        if dx_tracking == "alpha_half":
            a = np.asarray(alpha_k.nodal_values, dtype=float)
            x_now = [
                _x_alpha_half_quantile_on_y_line(alpha_xy, a, y_line=y0, mode=str(args.dx_intersection), q=dx_q)
                for y0 in y_lines
            ]
            dxs = [
                float(xn - xr) if (math.isfinite(xn) and math.isfinite(xr)) else float("nan")
                for xn, xr in zip(x_now, x_ref)
            ]
        else:
            # Lagrangian tracking via the Eulerian displacement field u:
            #   x = x_ref + u(x,t)  (fixed-point)
            # with (x_ref,y_ref) defined at t=0 by the alpha=0.5 intersection.
            xy_ref = np.asarray(track_ref_xy, dtype=float)
            finite = np.all(np.isfinite(xy_ref), axis=1)
            xy_cur = np.asarray(xy_ref, dtype=float).copy()
            if np.any(finite) and fp_iters > 0:
                for _ in range(fp_iters):
                    u_val = _interp_u_idw(xy_cur[finite, :], k=8, power=2.0)
                    xy_cur[finite, :] = xy_ref[finite, :] + u_val
            dxs = [
                float(xy_cur[i, 0] - xy_ref[i, 0]) if bool(finite[i]) else float("nan")
                for i in range(int(xy_ref.shape[0]))
            ]
        with ts_path.open("a", encoding="utf-8") as f_ts:
            f_ts.write(f"{float(t_s):.12e},{dxs[0]:.12e},{dxs[1]:.12e},{dxs[2]:.12e}\n")

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
        if os.getenv("PYCUTFEM_PETSC_MONITOR", "").strip().lower() in {"1", "true", "yes"}:
            petsc_opts.setdefault("snes_monitor", None)
            petsc_opts.setdefault("snes_converged_reason", None)
            petsc_opts.setdefault("ksp_converged_reason", None)
            petsc_opts.setdefault("ksp_monitor_short", None)

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
        )
    else:
        # PDAS / semismooth Newton for box constraints (more control than SNES for bounded alpha).
        vi_params = VIParameters(project_initial_guess=True, project_each_iteration=True)
        solver = PdasNewtonSolver(
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
            vi_params=vi_params,
            quad_order=int(qdeg),
            backend=backend,
        )
        # Enable bounds in PDAS mode: enforce 0<=alpha<=1 unless explicitly disabled.
        enable_alpha_box = not bool(getattr(args, "no_alpha_box_constraints", False))
        if enable_alpha_box:
            solver.set_box_bounds(by_field={"alpha": (0.0, 1.0)})

    def _on_dt_change(new_dt: float) -> None:
        dt_c.value = float(new_dt)

    def post_step_refiner(step, bcs_now, functions, prev_functions):
        # Called after an accepted Newton step and **before** promotion.
        _post_step_update()
        step_no = int(getattr(solver, "_current_step_no", int(step)))
        t_now = float(getattr(solver, "_current_t", 0.0) + getattr(solver, "_current_dt", dt_val))
        _append_timeseries(t_now)
        if alpha_metrics_enabled:
            _append_alpha_metrics(t_now, alpha_k)
        if vtk_every > 0 and (step_no % vtk_every == 0):
            vtkf = {"v": v_k, "p": p_k, "vS": vS_k, "u": u_k, "phi": phi_k, "alpha": alpha_k, "S": S_k}
            if ch_enabled:
                vtkf["mu_alpha"] = mu_alpha_k
            export_vtk(
                str(vtk_dir / f"step={step_no:04d}.vtu"),
                mesh,
                dh,
                vtkf,
                cell_data=_stress_cell_data(v_k, vS_k, u_k, alpha_k, phi_k) if vtk_stress_enabled else None,
            )

    functions_k = [v_k, p_k, vS_k, u_k, phi_k, alpha_k, S_k]
    functions_n = [v_n, p_n, vS_n, u_n, phi_n, alpha_n, S_n]
    if alpha_cahn_conservative:
        functions_k.insert(-1, lambda_alpha_k)
        functions_n.insert(-1, lambda_alpha_n)
    if ch_enabled:
        functions_k.insert(-1, mu_alpha_k)
        functions_n.insert(-1, mu_alpha_n)

    solver.solve_time_interval(
        functions=functions_k,
        prev_functions=functions_n,
        aux_functions={"dt": dt_c},
        time_params=TimeStepperParameters(
            dt=dt_val,
            final_time=float(args.t_final),
            max_steps=10_000,
            theta=theta,
            stop_on_steady=bool(args.stop_on_steady),
            steady_tol=float(args.steady_tol),
            allow_dt_reduction=bool(args.allow_dt_reduction),
            dt_min=float(args.dt_min),
            dt_reduction_factor=float(args.dt_reduction_factor),
            on_dt_change=_on_dt_change,
        ),
        post_step_refiner=post_step_refiner,
    )
    if bool(args.stop_on_steady):
        # Ensure `timeseries.csv` reaches t_final so downstream comparison/optimization
        # scripts can treat a steady run as "complete".
        try:
            arr = np.genfromtxt(str(ts_path), delimiter=",", skip_header=1, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[0] >= 1 and arr.shape[1] >= 4:
                t_last = float(arr[-1, 0])
                t_final = float(args.t_final)
                if t_last < t_final - 1.0e-12:
                    dx_last = np.asarray(arr[-1, 1:4], dtype=float).ravel()
                    with ts_path.open("a", encoding="utf-8") as f_ts:
                        f_ts.write(f"{t_final:.12e},{dx_last[0]:.12e},{dx_last[1]:.12e},{dx_last[2]:.12e}\n")
        except Exception:
            pass
        if alpha_metrics_enabled:
            # Mirror the timeseries behavior: append a final row at t_final with the
            # last alpha metrics so plots can be generated without special-casing.
            try:
                arr = np.genfromtxt(str(alpha_metrics_path), delimiter=",", skip_header=1, dtype=float)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.shape[0] >= 1 and arr.shape[1] >= 6:
                    t_last = float(arr[-1, 0])
                    t_final = float(args.t_final)
                    if t_last < t_final - 1.0e-12:
                        last = np.asarray(arr[-1, 1:6], dtype=float).ravel()
                        with alpha_metrics_path.open("a", encoding="utf-8") as f_am:
                            f_am.write(
                                f"{t_final:.12e},{last[0]:.12e},{last[1]:.12e},{last[2]:.12e},{last[3]:.12e},{last[4]:.12e}\n"
                            )
            except Exception:
                pass


if __name__ == "__main__":
    main()
