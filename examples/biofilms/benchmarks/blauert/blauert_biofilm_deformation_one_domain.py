"""
Blauert et al. (2015) biofilm deformation benchmark (one-domain model).

Goal
----
Validate the pure FSI part of the one-domain biofilm model in
`examples/utils/biofilm/one_domain.py` (no growth/detachment/damage) against the
deformation observed in the Blauert et al. supplementary video (mmc1).

Key choices (default)
---------------------
* Indicator alpha0 is initialized from a polygon extracted from the video
  (`exp_frame0_polygon_mm.csv`) and evolved by a conservative transport PDE
  (`--transport-mode pde`) with optional Cahn–Hilliard regularization.
* Porosity phi is solved (no growth sources in this FSI-only benchmark) and
  weakly penalized to phi->1 in the free-fluid region (`--gamma-phi`).
* Skeleton (u,vS) DOFs are restricted to a fixed rectangle around the biofilm so
  the extension penalties do not suppress the deformation when the fluid region
  is large (time-independent active set).

Transport PDE mode
------------------
The driver also supports a refmap mode (`--transport-mode refmap`) where
`alpha(x,t)=alpha0(x-u(x,t))` and `phi` is tied to `alpha`. This can be useful
for pure-deformation debugging, but the default is the PDE transport mode.

Outputs
-------
Writes to `out_dir`:
* `timeseries.csv`: x_front(y,t) and dx_front(y,t) for requested y-levels.
* `vtk/step=XXXX.vtu` (optional): VTK snapshots.
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
from pycutfem.core.topology import Node
from pycutfem.fem import transform
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.solvers.nonlinear_solver import (
    NewtonParameters,
    PetscSnesNewtonSolver,
    TimeStepperParameters,
)
from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import ds, dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad

from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms


def _tag_channel_boundaries(mesh: Mesh, *, L: float, H: float, tol: float = 1.0e-12) -> None:
    L = float(L)
    H = float(H)
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - L) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - H) <= tol,
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

    Outside the box, the DOFs are marked inactive (time-independent active set).
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
        raise RuntimeError(
            f"restrict-skeleton box keeps too few DOFs: {n_keep}/{n_tot}. Increase the box or disable restriction."
        )

    for fname in ("u_x", "u_y", "vS_x", "vS_y"):
        sl = np.asarray(dh.get_field_slice(fname), dtype=int).ravel()
        if sl.size != keep.size:
            raise RuntimeError(f"Unexpected DOF count mismatch for {fname}: slice={sl.size}, coords={keep.size}")
        _mark_inactive_dofs(dh, sl[~keep])
    return n_keep, n_tot


def _smooth_step(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(np.asarray(z, dtype=float)))


def _read_polygon_mm_csv(path: str) -> np.ndarray:
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

    # Point-in-polygon via ray casting.
    inside = np.zeros_like(xq, dtype=bool)
    xi = pts[:-1, 0]
    yi = pts[:-1, 1]
    xj = pts[1:, 0]
    yj = pts[1:, 1]
    for a_x, a_y, b_x, b_y in zip(xi, yi, xj, yj):
        cond = ((a_y > yq) != (b_y > yq))
        denom = (b_y - a_y) if abs(b_y - a_y) > 1.0e-30 else 1.0e-30
        x_at_y = (b_x - a_x) * (yq - a_y) / denom + a_x
        inside ^= cond & (xq < x_at_y)

    # Distance to segments.
    d2 = np.full_like(xq, float("inf"), dtype=float)
    for a_x, a_y, b_x, b_y in zip(xi, yi, xj, yj):
        abx = float(b_x - a_x)
        aby = float(b_y - a_y)
        ab2 = abx * abx + aby * aby
        if ab2 <= 1.0e-30:
            dx0 = xq - float(a_x)
            dy0 = yq - float(a_y)
            d2 = np.minimum(d2, dx0 * dx0 + dy0 * dy0)
            continue
        apx = xq - float(a_x)
        apy = yq - float(a_y)
        t = (apx * abx + apy * aby) / ab2
        t = np.clip(t, 0.0, 1.0)
        cx = float(a_x) + t * abx
        cy = float(a_y) + t * aby
        dx0 = xq - cx
        dy0 = yq - cy
        d2 = np.minimum(d2, dx0 * dx0 + dy0 * dy0)
    d = np.sqrt(np.maximum(d2, 0.0))
    d[inside] *= -1.0
    return d.reshape(np.asarray(x, dtype=float).shape)


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
    out: list[float] = []
    used: set[int] = set()
    tol = 1.0e-14
    min_allowed = float("-inf")
    for t in [float(v) for v in targets]:
        cand = np.array([i for i in range(y.size) if (i not in used) and (y[i] >= min_allowed - tol)], dtype=int)
        if cand.size == 0:
            cand = np.arange(0, y.size, dtype=int)
        diffs = np.abs(y[cand] - float(t))
        jmin = float(np.min(diffs))
        best = cand[diffs == jmin]
        j = int(best[0])
        out.append(float(y[j]))
        used.add(int(j))
        min_allowed = float(y[j]) + tol
    return out


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
        raise ValueError(f"Unknown intersection mode {mode!r}. Use 'rightmost' or 'leftmost'.")

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


def _alpha_half_intervals_on_y_line(
    alpha_xy: np.ndarray,
    alpha_vals: np.ndarray,
    *,
    y_line: float,
    alpha_half: float = 0.5,
) -> list[tuple[float, float]]:
    """
    Return all (x_L, x_R) intervals where alpha >= alpha_half on the line y=y_line.

    This is robust to multiple disconnected segments (e.g. due to pores / detached blobs).
    """
    xy = np.asarray(alpha_xy, dtype=float)
    a = np.asarray(alpha_vals, dtype=float).ravel()
    if xy.ndim != 2 or xy.shape[1] < 2 or a.shape[0] != xy.shape[0]:
        return []

    mask = np.abs(xy[:, 1] - float(y_line)) <= 1.0e-14
    if not np.any(mask):
        return []

    x = np.asarray(xy[mask, 0], dtype=float).ravel()
    av = np.asarray(a[mask], dtype=float).ravel()
    if x.size < 2:
        return []

    order = np.argsort(x)
    x = x[order]
    av = av[order]

    above = av >= float(alpha_half)
    if not np.any(above):
        return []

    # Segment starts/ends in the boolean array.
    starts = np.nonzero(above & np.r_[True, ~above[:-1]])[0]
    ends = np.nonzero(above & np.r_[~above[1:], True])[0]

    out: list[tuple[float, float]] = []
    for s, e in zip(starts, ends):
        s = int(s)
        e = int(e)

        # Left intersection x_L.
        if s <= 0:
            xL = float(x[0])
        else:
            a0, a1 = float(av[s - 1]), float(av[s])
            x0, x1 = float(x[s - 1]), float(x[s])
            da = a1 - a0
            if abs(da) <= 1.0e-16:
                xL = float(x1)
            else:
                t = (float(alpha_half) - a0) / da
                xL = float(x0 + t * (x1 - x0))

        # Right intersection x_R.
        if e >= x.size - 1:
            xR = float(x[-1])
        else:
            a0, a1 = float(av[e]), float(av[e + 1])
            x0, x1 = float(x[e]), float(x[e + 1])
            da = a1 - a0
            if abs(da) <= 1.0e-16:
                xR = float(x0)
            else:
                t = (float(alpha_half) - a0) / da
                xR = float(x0 + t * (x1 - x0))

        if math.isfinite(xL) and math.isfinite(xR) and xR >= xL:
            out.append((float(xL), float(xR)))

    out.sort(key=lambda p: p[0])
    return out


def _x_from_interval(xL: float, xR: float, *, mode: str, q: float) -> float:
    mode = str(mode).strip().lower()
    if mode not in {"leftmost", "rightmost"}:
        raise ValueError(f"Unknown mode {mode!r}. Use 'leftmost' or 'rightmost'.")
    q = float(np.clip(float(q), 0.0, 1.0))
    if mode == "leftmost":
        return float(xL + q * (xR - xL))
    return float(xR - q * (xR - xL))


def _x_alpha_half_track_on_y_line(
    alpha_xy: np.ndarray,
    alpha_vals: np.ndarray,
    *,
    y_line: float,
    mode: str,
    q: float,
    prev_x: float | None,
    alpha_half: float = 0.5,
) -> float:
    intervals = _alpha_half_intervals_on_y_line(alpha_xy, alpha_vals, y_line=float(y_line), alpha_half=float(alpha_half))
    if not intervals:
        return float("nan")
    cands = np.asarray([_x_from_interval(xL, xR, mode=str(mode), q=float(q)) for xL, xR in intervals], dtype=float)
    if cands.size == 0 or not np.any(np.isfinite(cands)):
        return float("nan")
    if prev_x is not None and math.isfinite(float(prev_x)):
        j = int(np.nanargmin(np.abs(cands - float(prev_x))))
        return float(cands[j])
    # Fallback: pick the extreme consistent with the mode.
    return float(np.nanmin(cands) if str(mode).strip().lower() == "leftmost" else np.nanmax(cands))


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

    With q=0 this reduces to the boundary intersection.
    """
    mode = str(mode).strip().lower()
    q = float(np.clip(float(q), 0.0, 1.0))
    xL = _x_alpha_half_on_y_line_mode(alpha_xy, alpha_vals, y_line=float(y_line), mode="leftmost", alpha_half=float(alpha_half))
    xR = _x_alpha_half_on_y_line_mode(alpha_xy, alpha_vals, y_line=float(y_line), mode="rightmost", alpha_half=float(alpha_half))
    if not (math.isfinite(xL) and math.isfinite(xR)):
        return float("nan")
    if mode == "leftmost":
        return float(xL + q * (xR - xL))
    return float(xR - q * (xR - xL))


def _x_front_global_quantile(
    alpha_xy: np.ndarray,
    alpha_vals: np.ndarray,
    *,
    q: float,
    alpha_half: float = 0.5,
) -> float:
    """
    Global upstream front location based on a quantile of x where alpha >= alpha_half.

    This is intended to be comparable to the video extractor's global x_front
    measurement (quantile over segmented mask pixels).
    """
    xy = np.asarray(alpha_xy, dtype=float)
    a = np.asarray(alpha_vals, dtype=float).ravel()
    if xy.ndim != 2 or xy.shape[1] < 2 or a.shape[0] != xy.shape[0]:
        return float("nan")
    q = float(np.clip(float(q), 0.0, 1.0))
    inside = a >= float(alpha_half)
    if not np.any(inside):
        return float("nan")
    xs = np.asarray(xy[inside, 0], dtype=float).ravel()
    if xs.size == 0:
        return float("nan")
    return float(np.quantile(xs, q))


def _lame_from_E_nu(E: float, nu: float) -> tuple[float, float]:
    E = float(E)
    nu = float(nu)
    if not (E > 0.0):
        raise ValueError("--E must be positive.")
    if not (-1.0 < nu < 0.5):
        raise ValueError("--nu must satisfy -1 < nu < 0.5.")
    mu = E / (2.0 * (1.0 + nu))
    lam = (E * nu) / ((1.0 + nu) * max(1.0e-16, (1.0 - 2.0 * nu)))
    return float(mu), float(lam)


def _restart_peek(path: Path) -> dict[str, float | int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{str(path)}\n"
            "Hint: --restart-from must point to an existing checkpoint (.npz).\n"
            "  - For a fresh run, omit --restart-from and set --restart-write-every N to create checkpoints.\n"
            "  - Then restart from <out_dir>/restart/checkpoint_latest.npz or checkpoint_step=XXXXX.npz."
        )
    with np.load(str(path)) as data:
        fmt = int(np.asarray(data.get("format_version", [0]), dtype=int).ravel()[0])
        if fmt not in (1,):
            raise ValueError(f"Unsupported restart format_version={fmt} in {path}")
        t = float(np.asarray(data["t"], dtype=float).ravel()[0])
        step = int(np.asarray(data["step"], dtype=int).ravel()[0])
        dt = float(np.asarray(data["dt"], dtype=float).ravel()[0])
        theta = float(np.asarray(data.get("theta", [1.0]), dtype=float).ravel()[0])
    return {"t": t, "step": step, "dt": dt, "theta": theta}


def _write_restart_checkpoint(
    path: Path,
    *,
    t: float,
    step: int,
    dt: float,
    theta: float,
    y_lines: np.ndarray,
    x_ref_global: float,
    x_ref: np.ndarray,
    x_prev: np.ndarray,
    v: VectorFunction,
    p: Function,
    vS: VectorFunction,
    u: VectorFunction,
    alpha: Function,
    phi: Function,
    S: Function,
    mu_alpha: Function | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(
        format_version=np.asarray([1], dtype=int),
        t=float(t),
        step=int(step),
        dt=float(dt),
        theta=float(theta),
        y_lines=np.asarray(y_lines, dtype=float).ravel(),
        x_ref_global=float(x_ref_global),
        x_ref=np.asarray(x_ref, dtype=float).ravel(),
        x_prev=np.asarray(x_prev, dtype=float).ravel(),
        v=np.asarray(v.nodal_values, dtype=float),
        p=np.asarray(p.nodal_values, dtype=float),
        vS=np.asarray(vS.nodal_values, dtype=float),
        u=np.asarray(u.nodal_values, dtype=float),
        alpha=np.asarray(alpha.nodal_values, dtype=float),
        phi=np.asarray(phi.nodal_values, dtype=float),
        S=np.asarray(S.nodal_values, dtype=float),
    )
    if mu_alpha is not None:
        payload["mu_alpha"] = np.asarray(mu_alpha.nodal_values, dtype=float)

    np.savez_compressed(str(path), **payload)


def _load_restart_checkpoint(
    path: Path,
    *,
    v: VectorFunction,
    p: Function,
    vS: VectorFunction,
    u: VectorFunction,
    alpha: Function,
    phi: Function,
    S: Function,
    mu_alpha: Function | None = None,
) -> dict[str, object]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    with np.load(str(path)) as data:
        fmt = int(np.asarray(data.get("format_version", [0]), dtype=int).ravel()[0])
        if fmt not in (1,):
            raise ValueError(f"Unsupported restart format_version={fmt} in {path}")

        def _load_into(func, key: str) -> None:
            arr = np.asarray(data[key], dtype=float).ravel()
            if func.nodal_values.size != arr.size:
                raise ValueError(
                    f"Restart field {key!r} size mismatch: file has {int(arr.size)} values, "
                    f"function has {int(func.nodal_values.size)}."
                )
            func.nodal_values[:] = arr

        _load_into(v, "v")
        _load_into(p, "p")
        _load_into(vS, "vS")
        _load_into(u, "u")
        _load_into(alpha, "alpha")
        _load_into(phi, "phi")
        _load_into(S, "S")
        if mu_alpha is not None and "mu_alpha" in data:
            _load_into(mu_alpha, "mu_alpha")

        meta = {
            "t": float(np.asarray(data["t"], dtype=float).ravel()[0]),
            "step": int(np.asarray(data["step"], dtype=int).ravel()[0]),
            "dt": float(np.asarray(data["dt"], dtype=float).ravel()[0]),
            "theta": float(np.asarray(data.get("theta", [1.0]), dtype=float).ravel()[0]),
            "y_lines": np.asarray(data.get("y_lines", []), dtype=float).ravel(),
            "x_ref_global": float(np.asarray(data.get("x_ref_global", [float("nan")]), dtype=float).ravel()[0]),
            "x_ref": np.asarray(data.get("x_ref", []), dtype=float).ravel(),
            "x_prev": np.asarray(data.get("x_prev", []), dtype=float).ravel(),
        }
    return meta


def _quad_corner_indices(p: int) -> tuple[int, int, int, int]:
    """Return (bl, br, tr, tl) local indices in lattice order (eta outer, xi inner)."""
    n = p + 1
    bl = 0
    br = p
    tr = p * n + p
    tl = p * n
    return bl, br, tr, tl


def _refine_element_quad4(
    mesh: Mesh, eid: int, nodes: list[Node], node_lookup: dict[tuple[float, float], int]
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Split one quad element into 4 children (single-level isotropic refinement).

    Produces a 2:1 nonconforming mesh (hanging nodes) handled by the solver's
    constraint layer. Returns (child_connectivity, child_corners).
    """
    if mesh.element_type != "quad":
        raise ValueError("quad4 refinement only supports quad meshes.")
    p = int(mesh.poly_order)
    t = np.linspace(-1.0, 1.0, p + 1)

    parent_conn = np.asarray(mesh.elements_connectivity[int(eid)], dtype=int).ravel()
    if parent_conn.size != (p + 1) ** 2:
        raise ValueError(f"Unexpected parent connectivity size for p={p}: {int(parent_conn.size)}")

    def _parent_node(xi_p: float, eta_p: float) -> int | None:
        ix = np.where(np.isclose(t, float(xi_p), atol=1e-12))[0]
        iy = np.where(np.isclose(t, float(eta_p), atol=1e-12))[0]
        if ix.size and iy.size:
            idx = int(iy[0] * (p + 1) + ix[0])
            return int(parent_conn[idx])
        return None

    def _get_node(xi_p: float, eta_p: float) -> int:
        # Reuse parent nodes when possible; otherwise create/reuse global by coordinate.
        nid = _parent_node(xi_p, eta_p)
        if nid is not None:
            return nid
        x_phys = transform.x_mapping(mesh, int(eid), (float(xi_p), float(eta_p)))
        key = (float(round(float(x_phys[0]), 14)), float(round(float(x_phys[1]), 14)))
        nid = node_lookup.get(key)
        if nid is not None:
            return int(nid)
        nid = len(nodes)
        node_lookup[key] = int(nid)
        nodes.append(Node(int(nid), float(x_phys[0]), float(x_phys[1])))
        return int(nid)

    def _child(mode: str) -> tuple[list[int], list[int]]:
        # mode: "bl", "br", "tr", "tl"
        conn: list[int] = []
        for eta in t:
            for xi in t:
                if mode == "bl":
                    xi_p = 0.5 * (float(xi) - 1.0)
                    eta_p = 0.5 * (float(eta) - 1.0)
                elif mode == "br":
                    xi_p = 0.5 * (float(xi) + 1.0)
                    eta_p = 0.5 * (float(eta) - 1.0)
                elif mode == "tr":
                    xi_p = 0.5 * (float(xi) + 1.0)
                    eta_p = 0.5 * (float(eta) + 1.0)
                elif mode == "tl":
                    xi_p = 0.5 * (float(xi) - 1.0)
                    eta_p = 0.5 * (float(eta) + 1.0)
                else:
                    raise ValueError(mode)
                conn.append(_get_node(xi_p, eta_p))
        bl, br, tr, tl = _quad_corner_indices(p)
        corners = [conn[bl], conn[br], conn[tr], conn[tl]]
        return conn, corners

    c_bl, cr_bl = _child("bl")
    c_br, cr_br = _child("br")
    c_tr, cr_tr = _child("tr")
    c_tl, cr_tl = _child("tl")
    return [c_bl, c_br, c_tr, c_tl], [cr_bl, cr_br, cr_tr, cr_tl]


def refine_around_biofilm_bbox(
    mesh: Mesh,
    *,
    poly: np.ndarray,
    band: float,
    expand_layers: int = 0,
    L: float,
    H: float,
) -> Mesh:
    """
    Refine quads intersecting the biofilm bbox (expanded by band) in one pass.

    This produces a 2:1 nonconforming mesh (hanging nodes), handled by the
    solver's constraint layer. The refinement is applied only once (no nested
    refinement levels).
    """
    if mesh.element_type != "quad":
        return mesh

    pts = np.asarray(poly, dtype=float)
    x_min = float(np.min(pts[:, 0])) - float(band)
    x_max = float(np.max(pts[:, 0])) + float(band)
    y_min = float(np.min(pts[:, 1])) - float(band)
    y_max = float(np.max(pts[:, 1])) + float(band)
    x_min = max(0.0, x_min)
    x_max = min(float(L), x_max)
    y_min = max(0.0, y_min)
    y_max = min(float(H), y_max)

    marked: set[int] = set()
    for eid, elem in enumerate(mesh.elements_list):
        corners = mesh.nodes_x_y_pos[list(elem.corner_nodes)]
        ex_min, ey_min = corners.min(axis=0)
        ex_max, ey_max = corners.max(axis=0)
        hits_bbox = (ex_min <= x_max) and (ex_max >= x_min) and (ey_min <= y_max) and (ey_max >= y_min)
        if hits_bbox:
            marked.add(int(eid))

    for _ in range(max(0, int(expand_layers))):
        new: set[int] = set()
        for eid in marked:
            for nb in mesh._neighbors[int(eid)]:
                if nb is not None:
                    new.add(int(nb))
        marked |= new

    if not marked:
        return mesh

    corner_conn = getattr(mesh, "corner_connectivity", None)
    if corner_conn is None:
        corner_conn = getattr(mesh, "elements_corner_nodes", None)
    if corner_conn is None:
        raise RuntimeError("Mesh does not expose corner connectivity required for refinement.")

    nodes = list(mesh.nodes_list)
    node_lookup = {(round(float(nd.x), 14), round(float(nd.y), 14)): int(nd.id) for nd in nodes}
    new_elems: list[list[int]] = []
    new_corners: list[list[int]] = []

    for eid in range(len(mesh.elements_list)):
        if eid not in marked:
            new_elems.append(list(np.asarray(mesh.elements_connectivity[int(eid)], dtype=int).ravel()))
            new_corners.append(list(np.asarray(corner_conn[int(eid)], dtype=int).ravel()))
            continue
        conns, corners = _refine_element_quad4(mesh, int(eid), nodes, node_lookup)
        new_elems.extend(conns)
        new_corners.extend(corners)

    new_mesh = Mesh(
        nodes=nodes,
        element_connectivity=np.asarray(new_elems, dtype=int),
        elements_corner_nodes=np.asarray(new_corners, dtype=int),
        element_type="quad",
        poly_order=mesh.poly_order,
    )
    _tag_channel_boundaries(new_mesh, L=float(L), H=float(H))
    logging.info(
        f"[refine] bbox band={float(band):.3e}m, layers={int(expand_layers)}: "
        f"marked {len(marked)} elems -> {len(new_elems)} elements, {len(nodes)} nodes"
    )
    return new_mesh


def main() -> None:
    ap = argparse.ArgumentParser(description="Blauert et al. (2015) deformation benchmark (one-domain).")
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--nx", type=int, default=220)
    ap.add_argument("--ny", type=int, default=80)
    ap.add_argument("--q", type=int, default=8, help="Quadrature degree.")

    ap.add_argument("--L", type=float, default=5.5e-3, help="Channel length [m]. Default matches the Dian SPH setup (5.5 mm).")
    ap.add_argument("--H", type=float, default=1.0e-3, help="Channel height [m].")

    # Local refinement around the biofilm polygon (single pass; produces hanging nodes)
    ap.add_argument("--refine-biofilm", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--refine-band",
        type=float,
        default=float("nan"),
        help="Refinement band around the biofilm bbox [m]. Default is ~2*max(hx,hy).",
    )
    ap.add_argument("--refine-expand-layers", type=int, default=0, help="Expand marked region by N neighbor layers (still one refinement pass).")

    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--t-final", type=float, default=10.0)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--allow-dt-reduction", action="store_true")
    ap.add_argument("--dt-min", type=float, default=0.01)
    ap.add_argument("--dt-reduction-factor", type=float, default=0.5)

    ap.add_argument("--newton-tol", type=float, default=1.0e-6)
    ap.add_argument("--newton-rtol", type=float, default=0.0, help="Relative SNES tolerance (0 disables).")
    ap.add_argument("--max-it", type=int, default=15)
    ap.add_argument(
        "--accept-nonconverged-atol-factor",
        type=float,
        default=0.0,
        help="Accept SNES best iterate when ‖F‖ <= factor*atol even if SNES reports non-convergence (0 disables).",
    )

    # Flow
    ap.add_argument("--u-avg", type=float, default=6.84e-2, help="Average inflow velocity [m/s] (Dian paper default).")
    ap.add_argument("--t-ramp", type=float, default=0.5, help="Cosine ramp time for inflow [s].")

    # Material / coupling (FSI-only: disable growth/detachment/damage)
    ap.add_argument("--rho-f", type=float, default=1000.0)
    ap.add_argument("--mu-f", type=float, default=1.0e-3)
    ap.add_argument("--kappa-inv", type=float, default=1.0e12, help="Inverse permeability [1/m^2].")
    ap.add_argument("--phi-b", type=float, default=0.47, help="Initial porosity inside the biofilm (0<phi_b<1).")
    ap.add_argument("--E", type=float, default=200.0, help="Young's modulus of the solid phase [Pa] (Dian paper default).")
    ap.add_argument("--nu", type=float, default=0.4, help="Poisson ratio (Dian paper default).")
    ap.add_argument("--solid-visco-eta", type=float, default=0.0, help="Kelvin–Voigt viscosity eta_s [Pa*s] (0 disables).")

    ap.add_argument("--gamma-u", type=float, default=1.0e-6, help="u extension penalty outside biofilm.")
    ap.add_argument("--u-extension", type=str, default="grad", choices=("l2", "grad"))
    ap.add_argument("--gamma-u-pin", type=float, default=1.0e-10, help="Tiny pinning used only with --u-extension grad.")
    ap.add_argument(
        "--kinematics-scale",
        type=float,
        default=float("nan"),
        help="Scaling applied to the u-kinematics constraint residual (conditioning aid; does not change solution set).",
    )
    ap.add_argument(
        "--v-supg",
        type=float,
        default=0.0,
        help="SUPG-like streamline diffusion for fluid momentum convection (0 disables; typical 0.1–10).",
    )
    ap.add_argument(
        "--u-supg",
        type=float,
        default=0.0,
        help="SUPG-like streamline diffusion for kinematic u-transport (0 disables; typical 0.1–10).",
    )

    # Transport (alpha/phi)
    ap.add_argument(
        "--transport-mode",
        type=str,
        default="pde",
        choices=("refmap", "pde"),
        help=(
            "How alpha/phi are evolved: 'refmap' updates alpha(x,t)=alpha0(x-u(x,t)) and ties phi to alpha; "
            "'pde' (default) solves the alpha/phi transport PDEs."
        ),
    )
    ap.add_argument("--D-phi", type=float, default=0.0, help="Porosity diffusion coefficient D_phi (pde mode).")
    ap.add_argument("--gamma-phi", type=float, default=5.0, help="Penalty enforcing phi->1 in free fluid (pde mode).")
    ap.add_argument("--phi-supg", type=float, default=0.0, help="SUPG stabilization strength for phi advection (pde mode).")
    ap.add_argument("--phi-cip", type=float, default=0.0, help="CIP stabilization strength for phi advection (pde mode).")
    ap.add_argument("--D-alpha", type=float, default=0.0, help="Diffusion coefficient D_alpha for alpha PDE (pde mode).")
    ap.add_argument("--alpha-supg", type=float, default=0.0, help="SUPG stabilization strength for alpha advection (pde mode).")
    ap.add_argument("--alpha-cip", type=float, default=0.0, help="CIP stabilization strength for alpha advection (pde mode).")
    ap.add_argument(
        "--alpha-advect-with",
        type=str,
        default="vS",
        choices=("vS", "v", "mix", "mix_biofilm"),
        help="Which velocity advects alpha in the alpha PDE (pde mode).",
    )
    ap.add_argument(
        "--alpha-advection-form",
        type=str,
        default="conservative",
        choices=("advective", "conservative"),
        help="Form of alpha advection by the chosen velocity in the alpha PDE (pde mode).",
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

    # Alpha initialization
    ap.add_argument(
        "--alpha0-file",
        type=str,
        default="examples/biofilms/benchmarks/blauert/exp_frame0_polygon_mm.csv",
        help="Closed polygon CSV (x_mm,y_mm) for alpha0.",
    )
    ap.add_argument("--alpha0-scale", type=float, default=1.0e-3, help="Scale applied to polygon coordinates (mm->m default).")
    ap.add_argument("--alpha0-tx", type=float, default=5.0e-4, help="Translation in x applied to polygon [m].")
    ap.add_argument("--alpha0-ty", type=float, default=0.0, help="Translation in y applied to polygon [m].")
    ap.add_argument("--eps", type=float, default=2.0e-5, help="Diffuse interface thickness eps [m].")

    # Skeleton DOF restriction
    ap.add_argument("--restrict-skeleton-dofs", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--restrict-box-pad", type=float, default=5.0e-4, help="Padding [m] added to biofilm bbox for active (u,vS) box.")
    ap.add_argument("--restrict-box-x0", type=float, default=float("nan"))
    ap.add_argument("--restrict-box-x1", type=float, default=float("nan"))
    ap.add_argument("--restrict-box-y0", type=float, default=float("nan"))
    ap.add_argument("--restrict-box-y1", type=float, default=float("nan"))

    # Tracking output
    ap.add_argument(
        "--track-y-um",
        type=str,
        default="150,250,350",
        help="Comma-separated y-levels [um] at which to track x_front via alpha=0.5 contour.",
    )
    ap.add_argument(
        "--dx-intersection",
        type=str,
        default="leftmost",
        choices=("leftmost", "rightmost"),
        help="Which alpha=0.5 intersection to track on each y-line.",
    )
    ap.add_argument(
        "--dx-quantile",
        type=float,
        default=0.1,
        help="Interior quantile between left/right intersections (0=boundary, 0.1=slightly inside).",
    )
    ap.add_argument(
        "--global-front-quantile",
        type=float,
        default=0.005,
        help="Quantile for global x_front (over alpha>=0.5 DOFs). 0=min, 0.005 is robust to tiny blobs.",
    )
    ap.add_argument("--vtk-every", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="out/_blauert_one_domain")
    ap.add_argument(
        "--restart-from",
        type=str,
        default="",
        help="Path to a restart checkpoint (.npz) written by --restart-write-every. If set, resumes from that state.",
    )
    ap.add_argument(
        "--restart-write-every",
        type=int,
        default=0,
        help="Write restart checkpoint every N accepted time steps (0 disables).",
    )
    ap.add_argument(
        "--restart-dt",
        type=float,
        default=float("nan"),
        help="Override dt when resuming from --restart-from (default: use dt stored in the checkpoint).",
    )
    ap.add_argument(
        "--restart-dir",
        type=str,
        default="",
        help="Directory for restart checkpoints (default: <out_dir>/restart).",
    )
    ap.add_argument("--trace-residual-fields", action="store_true", help="Print per-field |R|_inf during Newton/SNES iterations.")
    ap.add_argument("--trace-residual-fields-n", type=int, default=8, help="How many fields to show with --trace-residual-fields.")
    ap.add_argument("--trace-residual-worst", action="store_true", help="Print worst-DOF residual diagnostics during Newton/SNES.")
    ap.add_argument("--trace-residual-coords", action="store_true", help="With --trace-residual-worst, also print coordinates of worst DOF when available.")

    args = ap.parse_args()

    if bool(args.trace_residual_fields):
        os.environ["PYCUTFEM_NEWTON_TRACE_RES_FIELDS"] = "1"
        os.environ["PYCUTFEM_NEWTON_TRACE_RES_FIELDS_N"] = str(max(1, int(args.trace_residual_fields_n)))
    if bool(args.trace_residual_worst):
        os.environ["PYCUTFEM_RESIDUAL_TRACE"] = "1"
        if bool(args.trace_residual_coords):
            os.environ["PYCUTFEM_RESIDUAL_TRACE_COORDS"] = "1"

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    transport_mode = str(getattr(args, "transport_mode", "refmap")).strip().lower()
    if transport_mode not in {"refmap", "pde"}:
        raise ValueError(f"Unknown --transport-mode {transport_mode!r}.")

    alpha_ch_M = float(getattr(args, "alpha_ch_M", 0.0))
    alpha_ch_gamma = float(getattr(args, "alpha_ch_gamma", 0.0))
    ch_requested = bool(alpha_ch_M != 0.0 and alpha_ch_gamma != 0.0)
    if ch_requested and transport_mode != "pde":
        logging.warning("Ignoring --alpha-ch-* because --transport-mode=%s.", transport_mode)
        alpha_ch_M = 0.0
        alpha_ch_gamma = 0.0
    ch_enabled = bool(alpha_ch_M != 0.0 and alpha_ch_gamma != 0.0)
    alpha_ch_eps = float(getattr(args, "alpha_ch_eps", float("nan")))
    if not np.isfinite(alpha_ch_eps):
        alpha_ch_eps = float(getattr(args, "eps", 1.0))

    if transport_mode == "pde":
        D_phi_val = float(getattr(args, "D_phi", 0.0))
        gamma_phi_val = float(getattr(args, "gamma_phi", 0.0))
        if abs(D_phi_val) < 1.0e-30 and abs(gamma_phi_val) < 1.0e-30:
            raise ValueError("--transport-mode pde requires --gamma-phi>0 (recommended) or --D-phi>0 so phi is well-posed in the fluid.")

    L = float(args.L)
    H = float(args.H)
    if not (L > 0.0 and H > 0.0):
        raise ValueError("--L and --H must be positive.")

    restart_from = Path(str(args.restart_from)) if str(args.restart_from).strip() else None
    restart_meta: dict[str, float | int] | None = None
    if restart_from is not None:
        restart_meta = _restart_peek(restart_from)

    # ------------------------------------------------------------------
    # Polygon (alpha0)
    # ------------------------------------------------------------------
    poly_mm = _read_polygon_mm_csv(str(args.alpha0_file))
    poly_m = poly_mm * float(args.alpha0_scale)
    poly_m = poly_m + np.array([float(args.alpha0_tx), float(args.alpha0_ty)], dtype=float)
    if float(np.min(poly_m[:, 1])) < -1.0e-12:
        raise ValueError("alpha0 polygon has negative y after translation; expected y>=0.")
    poly_min = np.min(poly_m[:, :2], axis=0)
    poly_max = np.max(poly_m[:, :2], axis=0)
    logging.info(
        "[setup] alpha0 bbox: "
        f"x=[{float(poly_min[0]) * 1.0e3:.3f},{float(poly_max[0]) * 1.0e3:.3f}]mm, "
        f"y=[{float(poly_min[1]) * 1.0e3:.3f},{float(poly_max[1]) * 1.0e3:.3f}]mm "
        f"(clearance_top={float(H - poly_max[1]) * 1.0e6:.1f}um)"
    )
    if float(poly_min[0]) < -1.0e-12 or float(poly_max[0]) > float(L) + 1.0e-12:
        raise ValueError(
            "alpha0 polygon is outside the channel in x after scaling/translation: "
            f"x=[{float(poly_min[0]):.6e},{float(poly_max[0]):.6e}] vs [0,L]=[0,{float(L):.6e}]. "
            "Adjust --alpha0-scale/--alpha0-tx or provide a polygon in the correct coordinate system."
        )
    if float(poly_max[1]) > float(H) + 1.0e-12:
        raise ValueError(
            "alpha0 polygon extends above the top wall after scaling/translation: "
            f"y_max={float(poly_max[1]):.6e} > H={float(H):.6e}. "
            "This typically indicates a wrong pixel-to-length scale or segmentation; "
            "fix the polygon or adjust --alpha0-scale/--alpha0-ty."
        )

    # ------------------------------------------------------------------
    # Mesh + tags
    # ------------------------------------------------------------------
    nodes, elems, _edges, corners = structured_quad(L, H, nx=int(args.nx), ny=int(args.ny), poly_order=2, offset=(0.0, 0.0))
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=np.asarray(elems, dtype=int),
        elements_corner_nodes=np.asarray(corners, dtype=int),
        element_type="quad",
        poly_order=2,
    )
    _tag_channel_boundaries(mesh, L=L, H=H)
    if bool(args.refine_biofilm):
        hx = float(L) / float(max(1, int(args.nx)))
        hy = float(H) / float(max(1, int(args.ny)))
        band = float(args.refine_band) if np.isfinite(float(args.refine_band)) else 2.0 * max(hx, hy)
        mesh = refine_around_biofilm_bbox(
            mesh,
            poly=poly_m,
            band=band,
            expand_layers=int(args.refine_expand_layers),
            L=L,
            H=H,
        )

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
    dS = TrialFunction("S", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    vS_test = VectorTestFunction(space=VS, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    mu_alpha_test = TestFunction("mu_alpha", dof_handler=dh)
    S_test = TestFunction("S", dof_handler=dh)

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
    S_n.nodal_values[:] = 0.0

    eps0 = float(args.eps)
    alpha_xy = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    alpha0 = _smooth_step((-_signed_distance_polygon(alpha_xy[:, 0], alpha_xy[:, 1], poly_m)) / max(1.0e-12, eps0))
    alpha_n.nodal_values[:] = np.clip(np.asarray(alpha0, dtype=float), 0.0, 1.0)

    phi_b = float(args.phi_b)
    if not (0.0 < phi_b < 1.0):
        raise ValueError("--phi-b must be in (0,1).")
    phi_n.nodal_values[:] = np.clip(1.0 - (1.0 - phi_b) * np.asarray(alpha_n.nodal_values, dtype=float), 0.0, 1.0)

    if transport_mode == "refmap":
        # Refmap mode: freeze transport fields and update alpha/phi after each accepted step.
        _mark_inactive_fields(dh, "alpha", "phi", "S", "mu_alpha")
    else:
        # PDE mode: solve alpha/phi. We never solve substrate in this benchmark (no growth/chemistry).
        _mark_inactive_fields(dh, "S")
        if not ch_enabled:
            # CH disabled -> mu_alpha is not part of the residual, so keep it inactive to avoid a singular block.
            _mark_inactive_fields(dh, "mu_alpha")
        else:
            # Initialize a consistent-ish μ guess (drop the -εΔα term to avoid extra projections).
            a0 = np.asarray(alpha_n.nodal_values, dtype=float)
            Wp = 2.0 * a0 * (1.0 - a0) * (1.0 - 2.0 * a0)
            mu_alpha_n.nodal_values[:] = float(alpha_ch_gamma / max(alpha_ch_eps, 1.0e-12)) * Wp

    # Restrict skeleton DOFs to a fixed rectangle around the initial biofilm.
    if bool(args.restrict_skeleton_dofs):
        pad = float(args.restrict_box_pad)
        x0_auto = float(np.min(poly_m[:, 0])) - pad
        x1_auto = float(np.max(poly_m[:, 0])) + pad
        y0_auto = float(np.min(poly_m[:, 1])) - pad
        y1_auto = float(np.max(poly_m[:, 1])) + pad
        x0 = x0_auto if not np.isfinite(float(args.restrict_box_x0)) else float(args.restrict_box_x0)
        x1 = x1_auto if not np.isfinite(float(args.restrict_box_x1)) else float(args.restrict_box_x1)
        y0 = y0_auto if not np.isfinite(float(args.restrict_box_y0)) else float(args.restrict_box_y0)
        y1 = y1_auto if not np.isfinite(float(args.restrict_box_y1)) else float(args.restrict_box_y1)
        # Clamp to domain.
        x0 = max(0.0, x0)
        x1 = min(L, x1)
        y0 = max(0.0, y0)
        y1 = min(H, y1)
        n_keep, n_tot = _restrict_skeleton_dofs_to_box(dh, x0=x0, x1=x1, y0=y0, y1=y1)
        logging.info(f"[setup] restricted (u,vS) DOFs to box: keep {n_keep}/{n_tot} Q2 nodes")

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
            a = _smooth_step((-_signed_distance_polygon(chi[:, 0], chi[:, 1], poly_m)) / max(1.0e-12, eps0))
            alpha_k.nodal_values[:] = np.clip(np.asarray(a, dtype=float), 0.0, 1.0)
            phi_k.nodal_values[:] = np.clip(1.0 - (1.0 - phi_b) * np.asarray(alpha_k.nodal_values, dtype=float), 0.0, 1.0)

    else:

        def _post_step_update() -> None:
            # Nothing to do in PDE mode; alpha/phi are solved.
            return

    # ------------------------------------------------------------------
    # Forms
    # ------------------------------------------------------------------
    if restart_meta is not None:
        dt_val = float(restart_meta["dt"])
        if np.isfinite(float(args.restart_dt)):
            dt_val = float(args.restart_dt)
            logging.info(f"[restart] overriding dt -> {dt_val:.3e} (checkpoint dt was {float(restart_meta['dt']):.3e})")
        t0 = float(restart_meta["t"])
        step0 = int(restart_meta["step"])
        theta0 = float(restart_meta.get("theta", float(args.theta)))
        if abs(theta0 - float(args.theta)) > 1.0e-12:
            logging.warning(f"[restart] checkpoint theta={theta0:.6g} differs from --theta={float(args.theta):.6g}; using --theta.")
    else:
        dt_val = float(args.dt)
        t0 = 0.0
        step0 = 0
    dt_c = Constant(dt_val)
    theta = float(args.theta)

    mu_s, lambda_s = _lame_from_E_nu(float(args.E), float(args.nu))

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
        dS=dS,
        v_test=v_test,
        q_test=q_test,
        vS_test=vS_test,
        u_test=u_test,
        phi_test=phi_test,
        alpha_test=alpha_test,
        mu_alpha_test=mu_alpha_test,
        S_test=S_test,
        dx=dx(metadata={"q": int(args.q)}),
        ds_cip=ds(metadata={"q": int(args.q)}),
        dt=dt_c,
        theta=theta,
        rho_f=Constant(float(args.rho_f)),
        mu_f=Constant(float(args.mu_f)),
        mu_b=Constant(float(args.mu_f)),
        kappa_inv=Constant(float(args.kappa_inv)),
        mu_s=Constant(float(mu_s)),
        lambda_s=Constant(float(lambda_s)),
        solid_visco_eta=float(args.solid_visco_eta),
        gamma_u=float(args.gamma_u),
        u_extension_mode=str(args.u_extension),
        gamma_u_pin=float(args.gamma_u_pin),
        kinematics_scale=float(args.kinematics_scale) if np.isfinite(float(args.kinematics_scale)) else None,
        v_supg=float(args.v_supg),
        u_supg=float(args.u_supg),
        # Transport/kinetics controls (FSI-only: disable growth/detachment/damage, but may solve alpha/phi in PDE mode).
        D_phi=float(args.D_phi) if transport_mode == "pde" else 0.0,
        gamma_phi=float(args.gamma_phi) if transport_mode == "pde" else 0.0,
        phi_supg=float(args.phi_supg) if transport_mode == "pde" else 0.0,
        phi_cip=float(args.phi_cip) if transport_mode == "pde" else 0.0,
        D_alpha=float(args.D_alpha) if transport_mode == "pde" else 0.0,
        alpha_advect_with=str(args.alpha_advect_with),
        alpha_advection_form=str(args.alpha_advection_form) if transport_mode == "pde" else "advective",
        alpha_ch_M=float(alpha_ch_M) if transport_mode == "pde" else 0.0,
        alpha_ch_gamma=float(alpha_ch_gamma) if transport_mode == "pde" else 0.0,
        alpha_ch_eps=float(alpha_ch_eps) if transport_mode == "pde" else 1.0,
        alpha_ch_mobility=str(args.alpha_ch_mobility),
        alpha_supg=float(args.alpha_supg) if transport_mode == "pde" else 0.0,
        alpha_cip=float(args.alpha_cip) if transport_mode == "pde" else 0.0,
        # (keep Allen–Cahn disabled in this benchmark)
        alpha_cahn_M=0.0,
        alpha_cahn_gamma=0.0,
        D_S=0.0,
        mu_max=0.0,
        k_g=0.0,
        k_d=0.0,
        k_det=0.0,
        dim=2,
    )

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------
    u_avg = float(args.u_avg)
    u_max = 1.5 * u_avg
    t_ramp = float(args.t_ramp)
    try:
        mu_f0 = float(args.mu_f)
        rho_f0 = float(args.rho_f)
        tau_w = 6.0 * mu_f0 * u_avg / float(H)  # plane Poiseuille between plates
        Re = rho_f0 * u_avg * float(H) / max(mu_f0, 1.0e-30)
        logging.info(f"[setup] inflow: u_avg={u_avg:.3e} m/s (u_max={u_max:.3e}), tau_w≈{tau_w:.3e} Pa, Re≈{Re:.1f}")
    except Exception:
        pass

    def inflow_vx(_x, y, t):
        yy = float(y) / float(H)
        base = float(u_max * 4.0 * yy * (1.0 - yy))
        if t is None:
            return base
        tt = float(t)
        if tt <= 0.0:
            return 0.0
        if tt >= t_ramp:
            return base
        return base * 0.5 * (1.0 - math.cos(math.pi * tt / max(1.0e-12, t_ramp)))

    bcs = [
        BoundaryCondition("v_x", "dirichlet", "left", inflow_vx),
        BoundaryCondition("v_y", "dirichlet", "left", lambda x, y, t: 0.0),
        BoundaryCondition("v_x", "dirichlet", "bottom", lambda x, y, t: 0.0),
        BoundaryCondition("v_y", "dirichlet", "bottom", lambda x, y, t: 0.0),
        BoundaryCondition("v_x", "dirichlet", "top", lambda x, y, t: 0.0),
        BoundaryCondition("v_y", "dirichlet", "top", lambda x, y, t: 0.0),
        # Clamp skeleton to the substratum (biofilm is attached; no detachment in this FSI-only run).
        BoundaryCondition("u_x", "dirichlet", "bottom", lambda x, y, t: 0.0),
        BoundaryCondition("u_y", "dirichlet", "bottom", lambda x, y, t: 0.0),
        BoundaryCondition("vS_x", "dirichlet", "bottom", lambda x, y, t: 0.0),
        BoundaryCondition("vS_y", "dirichlet", "bottom", lambda x, y, t: 0.0),
        # Pressure reference
        BoundaryCondition("p", "dirichlet", "right", lambda x, y, t: 0.0),
    ]
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

    restart_write_every = max(0, int(args.restart_write_every))
    restart_dir = Path(str(args.restart_dir)).expanduser() if str(args.restart_dir).strip() else (out_dir / "restart")
    if restart_write_every > 0 or restart_from is not None:
        restart_dir.mkdir(parents=True, exist_ok=True)

    restart_state: dict[str, object] | None = None
    if restart_from is not None:
        restart_state = _load_restart_checkpoint(
            restart_from,
            v=v_n,
            p=p_n,
            vS=vS_n,
            u=u_n,
            alpha=alpha_n,
            phi=phi_n,
            S=S_n,
            mu_alpha=mu_alpha_n,
        )
        # Start from the checkpoint state (copy n -> k).
        v_k.nodal_values[:] = v_n.nodal_values
        p_k.nodal_values[:] = p_n.nodal_values
        vS_k.nodal_values[:] = vS_n.nodal_values
        u_k.nodal_values[:] = u_n.nodal_values
        alpha_k.nodal_values[:] = alpha_n.nodal_values
        phi_k.nodal_values[:] = phi_n.nodal_values
        S_k.nodal_values[:] = S_n.nodal_values
        mu_alpha_k.nodal_values[:] = mu_alpha_n.nodal_values
        logging.info(f"[restart] loaded {restart_from} (t={float(t0):.6e}s, step={int(step0)}, dt={float(dt_val):.3e})")

    track_y_um = [float(v) for v in str(args.track_y_um).split(",") if str(v).strip()]
    y_targets = [1.0e-6 * float(v) for v in track_y_um]
    alpha_y = np.asarray(alpha_xy[:, 1], dtype=float)
    if restart_state is not None and np.asarray(restart_state.get("y_lines", []), dtype=float).size:
        y_lines = np.asarray(restart_state.get("y_lines", []), dtype=float).ravel()
    else:
        y_lines = np.asarray(_nearest_y_levels(alpha_y, y_targets), dtype=float).ravel()

    x_ref_arr = np.asarray(restart_state.get("x_ref", []), dtype=float).ravel() if restart_state is not None else np.empty((0,), dtype=float)
    if x_ref_arr.size == y_lines.size:
        x_ref = [float(v) for v in x_ref_arr]
    else:
        # Reference x positions at t=0 (use initial alpha_n).
        x_ref = [
            _x_alpha_half_track_on_y_line(
                alpha_xy,
                alpha_n.nodal_values,
                y_line=float(y0),
                mode=str(args.dx_intersection),
                q=float(args.dx_quantile),
                prev_x=None,
                alpha_half=0.5,
            )
            for y0 in y_lines
        ]
    x_ref_global_val = float(restart_state.get("x_ref_global", float("nan"))) if restart_state is not None else float("nan")
    if math.isfinite(x_ref_global_val):
        x_ref_global = float(x_ref_global_val)
    else:
        x_ref_global = _x_front_global_quantile(alpha_xy, alpha_n.nodal_values, q=float(args.global_front_quantile), alpha_half=0.5)

    ts_path = out_dir / "timeseries.csv"
    header = (
        ["t_s", "x_front_global", "dx_front_global"]
        + [f"x_front_y{int(round(1.0e6 * y))}um" for y in y_lines]
        + [f"dx_front_y{int(round(1.0e6 * y))}um" for y in y_lines]
    )
    header_line = ",".join(header)
    append_existing_ts = bool(restart_from is not None and ts_path.exists() and ts_path.stat().st_size > 0)
    if append_existing_ts:
        try:
            with ts_path.open("r", encoding="utf-8") as f:
                first = f.readline().strip()
            if first != header_line:
                raise RuntimeError(
                    f"timeseries.csv header mismatch for restart.\n"
                    f"  file: {first}\n"
                    f"  expected: {header_line}\n"
                    f"Use a fresh --out-dir or match --track-y-um and mesh settings."
                )
        except Exception as exc:
            raise RuntimeError(f"Failed validating existing timeseries.csv for restart: {exc}") from exc
    else:
        with ts_path.open("w", encoding="utf-8") as f:
            f.write(header_line + "\n")

    x_prev_arr = np.asarray(restart_state.get("x_prev", []), dtype=float).ravel() if restart_state is not None else np.empty((0,), dtype=float)
    if x_prev_arr.size == y_lines.size:
        x_prev = [float(v) for v in x_prev_arr]
    else:
        x_prev = [float(v) for v in x_ref]

    def _append_timeseries(t_s: float) -> None:
        xg = _x_front_global_quantile(alpha_xy, alpha_k.nodal_values, q=float(args.global_front_quantile), alpha_half=0.5)
        dxg = float(xg - x_ref_global) if (math.isfinite(xg) and math.isfinite(x_ref_global)) else float("nan")
        nonlocal x_prev
        xs = []
        x_new_prev: list[float] = []
        for i, y0 in enumerate(y_lines):
            xi = _x_alpha_half_track_on_y_line(
                alpha_xy,
                alpha_k.nodal_values,
                y_line=float(y0),
                mode=str(args.dx_intersection),
                q=float(args.dx_quantile),
                prev_x=float(x_prev[i]) if i < len(x_prev) else None,
                alpha_half=0.5,
            )
            xs.append(xi)
            if math.isfinite(xi):
                x_new_prev.append(float(xi))
            else:
                x_new_prev.append(float(x_prev[i]))
        x_prev = x_new_prev
        dxs = [float(xs[i] - x_ref[i]) if (math.isfinite(xs[i]) and math.isfinite(x_ref[i])) else float("nan") for i in range(len(xs))]
        with ts_path.open("a", encoding="utf-8") as f:
            f.write(
                ",".join(
                    [f"{float(t_s):.12e}", f"{xg:.12e}", f"{dxg:.12e}"]
                    + [f"{z:.12e}" for z in xs]
                    + [f"{z:.12e}" for z in dxs]
                )
                + "\n"
            )

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    petsc_opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_atol": float(args.newton_tol),
        "snes_rtol": float(args.newton_rtol),
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
            newton_rtol=float(args.newton_rtol),
            max_newton_iter=int(args.max_it),
            accept_nonconverged_atol_factor=float(args.accept_nonconverged_atol_factor),
        ),
        quad_order=int(args.q),
        backend=str(args.backend),
        petsc_options=petsc_opts,
    )

    def _on_dt_change(new_dt: float) -> None:
        dt_c.value = float(new_dt)

    def on_step_failure(**info):  # noqa: ANN001
        """
        Debug hook for failed/nonconverged time steps.

        Prints per-field residual norms at the best available iterate so we can
        identify which equation block is driving SNES/Newton failures.
        """
        try:
            step_no = int(info.get("step_no", info.get("global_step_no", -1)))
        except Exception:
            step_no = -1
        try:
            t_fail = float(info.get("t", 0.0))
        except Exception:
            t_fail = 0.0
        try:
            dt_fail = float(info.get("dt", float(dt_val)))
        except Exception:
            dt_fail = float(dt_val)
        reason = getattr(solver, "_last_nonlinear_reason", None)
        fnorm = getattr(solver, "_last_nonlinear_norm", None)
        print(f"    [fail] step={step_no} t={t_fail:.6e} dt={dt_fail:.3e} reason={reason} norm={fnorm}")

        try:
            funcs = list(info.get("functions", []) or [])
            prev_funcs = list(info.get("prev_functions", []) or [])
            bcs_now = info.get("bcs", None)
            if bcs_now is not None and funcs:
                dh.apply_bcs(bcs_now, *funcs)
            if getattr(solver, "constraints", None) is not None and funcs:
                solver._enforce_constraints_on_functions(funcs)

            coeffs = {f.name: f for f in funcs}
            coeffs.update({f.name: f for f in prev_funcs})
            aux = info.get("aux_functions", None)
            if isinstance(aux, dict):
                coeffs.update(aux)

            _, R_red = solver._assemble_system_reduced(coeffs, need_matrix=False)
            R_full = np.asarray(solver.restrictor.expand_vec(np.asarray(R_red, dtype=float)), dtype=float).ravel()

            field_norms: list[tuple[float, str]] = []
            for fld in getattr(dh, "field_names", []):
                try:
                    sl = np.asarray(dh.get_field_slice(fld), dtype=int).ravel()
                except Exception:
                    continue
                if sl.size == 0:
                    continue
                field_norms.append((float(np.linalg.norm(R_full[sl], ord=np.inf)), str(fld)))
            field_norms.sort(reverse=True, key=lambda t: t[0])
            n_show = min(12, len(field_norms))
            if n_show:
                items = ", ".join(f"{name}:{val:.2e}" for val, name in field_norms[:n_show])
                print(f"    [fail_res] {items}")
                try:
                    dt_min_val = float(getattr(args, "dt_min", 0.0))
                except Exception:
                    dt_min_val = 0.0
                if dt_min_val > 0.0 and dt_fail <= dt_min_val + 1.0e-15:
                    worst_field = str(field_norms[0][1])
                    if worst_field.startswith("v"):
                        print(
                            "    [hint] failure occurred at dt==--dt-min and is dominated by fluid momentum residuals; "
                            "try a smaller --dt-min (e.g. 5e-3 or 1e-3) or consider quasi-static flow (--rho-f 0) for this benchmark."
                        )
        except Exception as exc:
            print(f"    [fail_res] failed to compute residual breakdown: {exc}")
        return None

    def post_step_refiner(step, bcs_now, functions, prev_functions):
        _post_step_update()
        step_no = int(getattr(solver, "_current_step_no", int(step)))
        t_now = float(getattr(solver, "_current_t", 0.0) + getattr(solver, "_current_dt", dt_val))
        _append_timeseries(t_now)
        if vtk_every > 0 and (step_no % vtk_every == 0):
            vtkf = {"v": v_k, "p": p_k, "vS": vS_k, "u": u_k, "phi": phi_k, "alpha": alpha_k}
            if transport_mode == "pde" and ch_enabled:
                vtkf["mu_alpha"] = mu_alpha_k
            export_vtk(str(vtk_dir / f"step={step_no:04d}.vtu"), mesh, dh, vtkf)
        if restart_write_every > 0 and (step_no % restart_write_every == 0):
            dt_step = float(getattr(solver, "_current_dt", dt_val))
            ckpt = restart_dir / f"checkpoint_step={step_no:05d}.npz"
            _write_restart_checkpoint(
                ckpt,
                t=float(t_now),
                step=int(step_no),
                dt=float(dt_step),
                theta=float(theta),
                y_lines=np.asarray(y_lines, dtype=float),
                x_ref_global=float(x_ref_global),
                x_ref=np.asarray(x_ref, dtype=float),
                x_prev=np.asarray(x_prev, dtype=float),
                v=v_k,
                p=p_k,
                vS=vS_k,
                u=u_k,
                alpha=alpha_k,
                phi=phi_k,
                S=S_k,
                mu_alpha=mu_alpha_k,
            )
            _write_restart_checkpoint(
                restart_dir / "checkpoint_latest.npz",
                t=float(t_now),
                step=int(step_no),
                dt=float(dt_step),
                theta=float(theta),
                y_lines=np.asarray(y_lines, dtype=float),
                x_ref_global=float(x_ref_global),
                x_ref=np.asarray(x_ref, dtype=float),
                x_prev=np.asarray(x_prev, dtype=float),
                v=v_k,
                p=p_k,
                vS=vS_k,
                u=u_k,
                alpha=alpha_k,
                phi=phi_k,
                S=S_k,
                mu_alpha=mu_alpha_k,
            )

    # Prime alpha_k/phi_k and write the initial row (t=0 for fresh runs, or t=t0 for restart into a fresh out_dir).
    alpha_k.nodal_values[:] = alpha_n.nodal_values
    phi_k.nodal_values[:] = phi_n.nodal_values
    mu_alpha_k.nodal_values[:] = mu_alpha_n.nodal_values
    S_k.nodal_values[:] = S_n.nodal_values
    if restart_from is None:
        _append_timeseries(0.0)
        if restart_write_every > 0:
            _write_restart_checkpoint(
                restart_dir / "checkpoint_step=00000.npz",
                t=0.0,
                step=0,
                dt=float(dt_val),
                theta=float(theta),
                y_lines=np.asarray(y_lines, dtype=float),
                x_ref_global=float(x_ref_global),
                x_ref=np.asarray(x_ref, dtype=float),
                x_prev=np.asarray(x_prev, dtype=float),
                v=v_k,
                p=p_k,
                vS=vS_k,
                u=u_k,
                alpha=alpha_k,
                phi=phi_k,
                S=S_k,
                mu_alpha=mu_alpha_k,
            )
            _write_restart_checkpoint(
                restart_dir / "checkpoint_latest.npz",
                t=0.0,
                step=0,
                dt=float(dt_val),
                theta=float(theta),
                y_lines=np.asarray(y_lines, dtype=float),
                x_ref_global=float(x_ref_global),
                x_ref=np.asarray(x_ref, dtype=float),
                x_prev=np.asarray(x_prev, dtype=float),
                v=v_k,
                p=p_k,
                vS=vS_k,
                u=u_k,
                alpha=alpha_k,
                phi=phi_k,
                S=S_k,
                mu_alpha=mu_alpha_k,
            )
    elif not append_existing_ts:
        _append_timeseries(float(t0))
        if restart_write_every > 0:
            _write_restart_checkpoint(
                restart_dir / f"checkpoint_step={int(step0):05d}.npz",
                t=float(t0),
                step=int(step0),
                dt=float(dt_val),
                theta=float(theta),
                y_lines=np.asarray(y_lines, dtype=float),
                x_ref_global=float(x_ref_global),
                x_ref=np.asarray(x_ref, dtype=float),
                x_prev=np.asarray(x_prev, dtype=float),
                v=v_k,
                p=p_k,
                vS=vS_k,
                u=u_k,
                alpha=alpha_k,
                phi=phi_k,
                S=S_k,
                mu_alpha=mu_alpha_k,
            )
            _write_restart_checkpoint(
                restart_dir / "checkpoint_latest.npz",
                t=float(t0),
                step=int(step0),
                dt=float(dt_val),
                theta=float(theta),
                y_lines=np.asarray(y_lines, dtype=float),
                x_ref_global=float(x_ref_global),
                x_ref=np.asarray(x_ref, dtype=float),
                x_prev=np.asarray(x_prev, dtype=float),
                v=v_k,
                p=p_k,
                vS=vS_k,
                u=u_k,
                alpha=alpha_k,
                phi=phi_k,
                S=S_k,
                mu_alpha=mu_alpha_k,
            )

    functions_k = [v_k, p_k, vS_k, u_k, phi_k, alpha_k]
    functions_n = [v_n, p_n, vS_n, u_n, phi_n, alpha_n]
    if transport_mode == "pde" and ch_enabled:
        functions_k.append(mu_alpha_k)
        functions_n.append(mu_alpha_n)
    functions_k.append(S_k)
    functions_n.append(S_n)
    if float(t0) >= float(args.t_final) - 1.0e-14:
        logging.info(f"[done] restart t0={float(t0):.6e}s >= t_final={float(args.t_final):.6e}s; nothing to do.")
        return
    solver.solve_time_interval(
        functions=functions_k,
        prev_functions=functions_n,
        aux_functions={"dt": dt_c},
        time_params=TimeStepperParameters(
            dt=dt_val,
            final_time=float(args.t_final),
            max_steps=100_000,
            theta=theta,
            t0=float(t0),
            step0=int(step0),
            stop_on_steady=False,
            steady_tol=0.0,
            allow_dt_reduction=bool(args.allow_dt_reduction),
            dt_min=float(args.dt_min),
            dt_reduction_factor=float(args.dt_reduction_factor),
            on_dt_change=_on_dt_change,
            on_step_failure=on_step_failure,
        ),
        post_step_refiner=post_step_refiner,
    )


if __name__ == "__main__":
    main()
