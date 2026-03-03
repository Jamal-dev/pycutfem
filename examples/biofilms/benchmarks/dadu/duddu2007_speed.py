"""
Duddu et al. (2007) interface speed computation (Section 5.2, Eq. 31 + Fig. 3).

The paper computes the interface normal speed as:

    F = n_phi · ∇Phi  on Γ_int,   with  n_phi = ∇phi / ||∇phi||,

where Phi is the velocity potential and phi is the level set function.

Key detail: Phi is only C^0 across Γ_int, so ∇Phi must be computed *one-sided*
from the biofilm side (phi<0). Duddu (2007) computes ∇Phi on cut triangles by
restricting to a "shaded triangle" on the biofilm side and using a standard
P1 finite element gradient there (Fig. 3, Case 1 & Case 2).

This module implements that speed computation for P1 triangular meshes using
the interface segments already constructed by Mesh.build_interface_segments().
All code is kept inside the Duddu benchmark folder as requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class InterfaceSegmentSpeed:
    """Speed data on one interface segment inside a cut element."""

    eid: int
    p0: np.ndarray  # (2,)
    p1: np.ndarray  # (2,)
    F0: float
    F1: float
    n_phi: np.ndarray  # (2,), oriented by ∇phi/||∇phi|| (NEG->POS)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.asarray(self.p1, float) - np.asarray(self.p0, float)))


def _triangle_grad(coords: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Gradient of the unique P1 function with nodal values `values` on a
    physical triangle with vertex coordinates `coords`.

    Parameters
    ----------
    coords : (3,2) array
        Triangle vertex coordinates.
    values : (3,) array
        Scalar values at the triangle vertices.

    Returns
    -------
    grad : (2,) array
        Constant physical gradient [du/dx, du/dy] over the triangle.
    """
    X = np.asarray(coords, dtype=float).reshape(3, 2)
    u = np.asarray(values, dtype=float).reshape(3)
    x1, y1 = X[0]
    x2, y2 = X[1]
    x3, y3 = X[2]
    twoA = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if abs(float(twoA)) <= 1.0e-30:
        return np.zeros(2, dtype=float)
    dudx = (u[0] * (y2 - y3) + u[1] * (y3 - y1) + u[2] * (y1 - y2)) / twoA
    dudy = (u[0] * (x3 - x2) + u[1] * (x1 - x3) + u[2] * (x2 - x1)) / twoA
    return np.asarray([dudx, dudy], dtype=float)


def _twoA(coords: np.ndarray) -> float:
    """Twice the signed area of a physical triangle (used for degeneracy checks)."""
    X = np.asarray(coords, dtype=float).reshape(3, 2)
    x1, y1 = X[0]
    x2, y2 = X[1]
    x3, y3 = X[2]
    return float((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(2)
    nrm = float(np.linalg.norm(v))
    if nrm <= 1.0e-30:
        return np.zeros(2, dtype=float)
    return v / nrm


def compute_interface_segment_speeds_duddu2007(
    *,
    dof_handler,
    level_set,
    Phi,
    field: str = "Phi",
    phi_tol: float = 1.0e-12,
    include_aligned_edges: bool = True,
) -> list[InterfaceSegmentSpeed]:
    """
    Compute one-sided interface speeds on all cut elements using Duddu (2007)
    Section 5.2 (Fig. 3) logic.

    Assumptions / limitations
    -------------------------
    - Mesh must be a P1 triangular mesh (element_type='tri', poly_order=1).
    - `Phi` is a scalar Function defined on `dof_handler` (possibly XFEM).
    - `level_set` uses the CutFEM convention Ω⁻={phi<0} (biofilm), Ω⁺={phi>0}.
    """
    base = getattr(dof_handler, "base", dof_handler)
    mesh = base.mixed_element.mesh

    if str(getattr(mesh, "element_type", "")).lower() != "tri" or int(getattr(mesh, "poly_order", 1)) != 1:
        raise NotImplementedError("Duddu 2007 speed helper currently supports only P1 triangular meshes.")

    # Precompute nodal phi for sign classification.
    phi_nodes = np.asarray(level_set.evaluate_on_nodes(mesh), dtype=float).ravel()

    # Fast mapping node_id -> base gdof (CG P1), then to Phi value via Phi.get_nodal_values.
    dof_map = getattr(base, "dof_map", {}) or {}
    node_to_gdof = (dof_map.get(field) or {}).copy()
    if not node_to_gdof:
        raise RuntimeError(f"Missing dof_handler.dof_map for field {field!r}; cannot map triangle vertices to Phi DOFs.")

    cut_ids = getattr(mesh, "element_bitset", lambda _: None)("cut")
    eids: Iterable[int]
    if cut_ids is None:
        # Fallback: scan elements.
        eids = [i for i, e in enumerate(getattr(mesh, "elements_list", [])) if getattr(e, "tag", "") == "cut"]
    else:
        eids = (int(i) for i in cut_ids.to_indices())

    out: list[InterfaceSegmentSpeed] = []

    for eid in eids:
        elem = mesh.elements_list[int(eid)]
        corner_nodes = list(getattr(elem, "corner_nodes", []) or [])
        if len(corner_nodes) != 3:
            continue

        X = np.asarray(mesh.nodes_x_y_pos[corner_nodes], dtype=float).reshape(3, 2)
        twoA_parent = abs(_twoA(X))
        phi_v = np.asarray(phi_nodes[corner_nodes], dtype=float).reshape(3)

        # Identify biofilm-side vertices (phi<0). Treat |phi|<=phi_tol as "on interface".
        neg = np.where(phi_v < -float(phi_tol))[0].tolist()
        pos = np.where(phi_v > float(phi_tol))[0].tolist()
        if not neg or not pos:
            continue  # not a properly cut element

        # Interface segments already computed on this element.
        segs = getattr(elem, "interface_segments", None)
        segments: list[tuple[np.ndarray, np.ndarray]] = []
        if segs:
            for seg in segs:
                if not seg or len(seg) != 2:
                    continue
                p0, p1 = seg
                segments.append((np.asarray(p0, dtype=float).reshape(2), np.asarray(p1, dtype=float).reshape(2)))
        else:
            pts = getattr(elem, "interface_pts", None)
            if pts is not None and len(pts) == 2:
                p0, p1 = pts
                segments.append((np.asarray(p0, dtype=float).reshape(2), np.asarray(p1, dtype=float).reshape(2)))

        if not segments:
            continue

        # n_phi: Duddu uses standard FE gradient of phi in the *parent* triangle.
        grad_phi = _triangle_grad(X, phi_v)
        n_phi = _unit(grad_phi)
        if float(np.linalg.norm(n_phi)) <= 0.0 and hasattr(level_set, "gradient"):
            # Fallback: analytic gradient at segment midpoint.
            pm = 0.5 * (segments[0][0] + segments[0][1])
            n_phi = _unit(np.asarray(level_set.gradient(pm), dtype=float).reshape(2))

        # Phi values at the triangle vertices (mesh nodes). Enrichment is zero at nodes for shifted-|phi|.
        gdofs_v = np.asarray([int(node_to_gdof[int(nid)]) for nid in corner_nodes], dtype=int)
        Phi_v = np.asarray(Phi.get_nodal_values(gdofs_v), dtype=float).reshape(3)

        if len(neg) == 1:
            # Case 1: one vertex in Ω⁻. Shaded triangle: (vertex in Ω⁻) + (two interface cut points).
            i_in = int(neg[0])
            u_in = float(Phi_v[i_in])
            for p0, p1 in segments:
                Xs = np.vstack([X[i_in, :], p0, p1])
                us = np.asarray([u_in, 0.0, 0.0], dtype=float)
                grad_Phi = _triangle_grad(Xs, us)
                # Degenerate shaded triangle -> fall back to using the segment midpoint.
                if twoA_parent > 0.0 and abs(_twoA(Xs)) < 1.0e-4 * twoA_parent:
                    pm = 0.5 * (p0 + p1)
                    Xsm = np.vstack([X[i_in, :], pm, p1])
                    grad_Phi = _triangle_grad(Xsm, np.asarray([u_in, 0.0, 0.0], dtype=float))
                F = float(np.dot(n_phi, grad_Phi))
                out.append(
                    InterfaceSegmentSpeed(
                        eid=int(eid),
                        p0=p0,
                        p1=p1,
                        F0=F,
                        F1=F,
                        n_phi=n_phi,
                    )
                )
        elif len(neg) == 2:
            # Case 2: two vertices in Ω⁻. For each endpoint x0 of the interface segment,
            # use the shaded triangle (two Ω⁻ vertices + x0) and set Phi(x0)=0.
            i0, i1 = int(neg[0]), int(neg[1])
            u0, u1 = float(Phi_v[i0]), float(Phi_v[i1])
            for p0, p1 in segments:
                Xs0 = np.vstack([X[i0, :], X[i1, :], p0])
                us0 = np.asarray([u0, u1, 0.0], dtype=float)
                grad0 = _triangle_grad(Xs0, us0)
                F0 = float(np.dot(n_phi, grad0))

                Xs1 = np.vstack([X[i0, :], X[i1, :], p1])
                us1 = np.asarray([u0, u1, 0.0], dtype=float)
                grad1 = _triangle_grad(Xs1, us1)
                F1 = float(np.dot(n_phi, grad1))

                if twoA_parent > 0.0:
                    # If an endpoint triangle is nearly degenerate (interface grazes a vertex),
                    # use a midpoint evaluation to avoid huge spurious gradients.
                    pm = 0.5 * (p0 + p1)
                    Xsm = np.vstack([X[i0, :], X[i1, :], pm])
                    usm = np.asarray([u0, u1, 0.0], dtype=float)
                    Fm = float(np.dot(n_phi, _triangle_grad(Xsm, usm)))
                    if abs(_twoA(Xs0)) < 1.0e-4 * twoA_parent:
                        F0 = Fm
                    if abs(_twoA(Xs1)) < 1.0e-4 * twoA_parent:
                        F1 = Fm

                out.append(
                    InterfaceSegmentSpeed(
                        eid=int(eid),
                        p0=p0,
                        p1=p1,
                        F0=F0,
                        F1=F1,
                        n_phi=n_phi,
                    )
                )
        else:
            # Multiple interface/degenerate configurations (vertex on interface, etc.).
            # Keep a conservative fallback: use the first negative vertex with Case 1 triangle.
            i_in = int(neg[0])
            u_in = float(Phi_v[i_in])
            for p0, p1 in segments:
                Xs = np.vstack([X[i_in, :], p0, p1])
                us = np.asarray([u_in, 0.0, 0.0], dtype=float)
                grad_Phi = _triangle_grad(Xs, us)
                F = float(np.dot(n_phi, grad_Phi))
                out.append(
                    InterfaceSegmentSpeed(
                        eid=int(eid),
                        p0=p0,
                        p1=p1,
                        F0=F,
                        F1=F,
                        n_phi=n_phi,
                    )
                )

    if bool(include_aligned_edges):
        _add_aligned_edge_speeds(
            out,
            mesh=mesh,
            level_set=level_set,
            phi_nodes=phi_nodes,
            Phi=Phi,
            node_to_gdof=node_to_gdof,
            phi_tol=float(phi_tol),
        )

    return out


def _ls_value(level_set, x: np.ndarray, *, eid: int | None = None) -> float:
    """Safe helper to evaluate a (possibly element-aware) level set."""
    try:
        if eid is not None:
            return float(level_set(np.asarray(x, float).reshape(2), eid=int(eid)))
    except TypeError:
        pass
    return float(level_set(np.asarray(x, float).reshape(2)))


def _ls_grad(level_set, x: np.ndarray, *, eid: int | None = None) -> np.ndarray:
    """Safe helper to evaluate a (possibly element-aware) level set gradient."""
    try:
        if eid is not None:
            return np.asarray(level_set.gradient(np.asarray(x, float).reshape(2), eid=int(eid)), dtype=float).reshape(2)
    except TypeError:
        pass
    return np.asarray(level_set.gradient(np.asarray(x, float).reshape(2)), dtype=float).reshape(2)


def _add_aligned_edge_speeds(
    out: list[InterfaceSegmentSpeed],
    *,
    mesh,
    level_set,
    phi_nodes: np.ndarray,
    Phi,
    node_to_gdof: dict,
    phi_tol: float,
) -> None:
    """
    Add segment speeds for interface edges aligned with the mesh skeleton.

    When the interface coincides with mesh edges, ``mesh.element_bitset('cut')`` can
    become empty even though ``mesh.edge_bitset('interface')`` is non-empty. Duddu's
    speed definition still applies: use the one-sided (biofilm-side) ∇Φ and n_φ on Γ.
    """
    try:
        aligned = mesh.edge_bitset("interface")
    except Exception:
        return
    if aligned is None:
        return

    try:
        edge_ids = list(aligned.to_indices())
    except Exception:
        # Fall back: treat as iterable of indices.
        edge_ids = [int(i) for i in aligned]  # type: ignore[arg-type]

    for edge_id in edge_ids:
        e = mesh.edges_list[int(edge_id)]
        if getattr(e, "left", None) is None or getattr(e, "right", None) is None:
            continue  # boundary edge

        left = int(e.left)
        right = int(e.right)

        c_left = np.asarray(mesh.elements_list[left].centroid(), dtype=float).reshape(2)
        c_right = np.asarray(mesh.elements_list[right].centroid(), dtype=float).reshape(2)
        phiL = _ls_value(level_set, c_left, eid=left)
        phiR = _ls_value(level_set, c_right, eid=right)

        neg_eid = pos_eid = None
        if (phiL < -phi_tol) and (phiR > phi_tol):
            neg_eid, pos_eid = left, right
        elif (phiR < -phi_tol) and (phiL > phi_tol):
            neg_eid, pos_eid = right, left
        else:
            # Fallback: classify using the element vertex not on the edge.
            edge_nodes = set(int(n) for n in getattr(e, "nodes", ()))
            if len(edge_nodes) != 2:
                continue
            for cand, other in ((left, right), (right, left)):
                cn = mesh.elements_list[int(cand)].corner_nodes
                if not cn or len(cn) != 3:
                    continue
                extra = [int(n) for n in cn if int(n) not in edge_nodes]
                if len(extra) != 1:
                    continue
                phi_extra = float(phi_nodes[int(extra[0])])
                if phi_extra < -phi_tol:
                    neg_eid, pos_eid = int(cand), int(other)
                    break
            if neg_eid is None or pos_eid is None:
                continue

        # Biofilm-side element (Ω⁻) is neg_eid.
        elem = mesh.elements_list[int(neg_eid)]
        corner_nodes = list(getattr(elem, "corner_nodes", []) or [])
        if len(corner_nodes) != 3:
            continue
        edge_nodes = [int(n) for n in getattr(e, "nodes", ())]
        if len(edge_nodes) != 2:
            continue

        # Identify the interior (biofilm) vertex.
        in_nodes = [int(n) for n in corner_nodes if int(n) not in set(edge_nodes)]
        if len(in_nodes) != 1:
            continue
        nid_in = int(in_nodes[0])

        # Geometry
        p0 = np.asarray(mesh.nodes_x_y_pos[int(edge_nodes[0])], dtype=float).reshape(2)
        p1 = np.asarray(mesh.nodes_x_y_pos[int(edge_nodes[1])], dtype=float).reshape(2)
        Xin = np.asarray(mesh.nodes_x_y_pos[int(nid_in)], dtype=float).reshape(2)

        # n_phi: prefer ∇phi from the biofilm element; fallback to analytic grad; then to edge normal.
        X_parent = np.asarray(mesh.nodes_x_y_pos[corner_nodes], dtype=float).reshape(3, 2)
        phi_v = np.asarray(phi_nodes[corner_nodes], dtype=float).reshape(3)
        grad_phi = _triangle_grad(X_parent, phi_v)
        n_phi = _unit(grad_phi)
        if float(np.linalg.norm(n_phi)) <= 1.0e-15:
            mid = 0.5 * (p0 + p1)
            n_phi = _unit(_ls_grad(level_set, mid, eid=int(neg_eid)))
        if float(np.linalg.norm(n_phi)) <= 1.0e-15:
            t = p1 - p0
            n_phi = _unit(np.asarray([t[1], -t[0]], dtype=float))

        # Orient n_phi from Ω⁻ -> Ω⁺.
        c_neg = np.asarray(mesh.elements_list[int(neg_eid)].centroid(), float).reshape(2)
        c_pos = np.asarray(mesh.elements_list[int(pos_eid)].centroid(), float).reshape(2)
        if float(np.dot(n_phi, c_pos - c_neg)) < 0.0:
            n_phi = -n_phi

        # One-sided ∇Phi from the biofilm element, with Phi=0 on the interface edge.
        gd_in = int(node_to_gdof[int(nid_in)])
        u_in = float(Phi.get_nodal_values(np.asarray([gd_in], dtype=int))[0])
        grad_Phi = _triangle_grad(np.vstack([Xin, p0, p1]), np.asarray([u_in, 0.0, 0.0], dtype=float))
        F = float(np.dot(n_phi, grad_Phi))

        out.append(
            InterfaceSegmentSpeed(
                eid=int(neg_eid),
                p0=p0,
                p1=p1,
                F0=F,
                F1=F,
                n_phi=np.asarray(n_phi, dtype=float),
            )
        )


def average_interface_speed(segments: Sequence[InterfaceSegmentSpeed]) -> float:
    """
    Length-weighted average speed along Γ:
        F_avg = (1/|Γ|) ∫_Γ F ds,
    using piecewise-linear interpolation on each segment.
    """
    if not segments:
        return 0.0
    num = 0.0
    den = 0.0
    for s in segments:
        L = float(s.length)
        if not np.isfinite(L) or L <= 0.0:
            continue
        num += 0.5 * (float(s.F0) + float(s.F1)) * L
        den += L
    return float(num / den) if den > 0.0 else 0.0


def sample_segment_speeds(
    segments: Sequence[InterfaceSegmentSpeed], *, samples_per_segment: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample points and speeds along interface segments (linear interpolation).

    Returns
    -------
    points : (N,2) array
    speeds : (N,) array
    """
    m = int(samples_per_segment)
    if m < 2:
        raise ValueError("samples_per_segment must be >= 2")

    pts: list[np.ndarray] = []
    spd: list[float] = []
    for s in segments:
        p0 = np.asarray(s.p0, dtype=float).reshape(2)
        p1 = np.asarray(s.p1, dtype=float).reshape(2)
        for k in range(m):
            t = float(k) / float(m - 1)
            pts.append((1.0 - t) * p0 + t * p1)
            spd.append((1.0 - t) * float(s.F0) + t * float(s.F1))
    if not pts:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)
    return np.vstack(pts).astype(float), np.asarray(spd, dtype=float)
