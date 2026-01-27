"""CutFEM interface segment extraction for non-matching coupling.

This module provides a small bridge between:
  - CutFEM-resolved interfaces on a *background* mesh (fluid mesh), and
  - non-matching coupling/integration on a *separate* mesh (e.g. porous mesh).

It builds a segment-based interface description (P0-P1 line segments) together
with a **discrete normal** per segment (paper remark on using discrete normals).

No hand-written interface residual/Jacobian kernels live here. The intended use
is via UFL forms + `dNonmatchingInterface`, so the coupling works across all
project backends (python/jit/cpp).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycutfem.fem import transform


Array = np.ndarray


@dataclass(frozen=True, slots=True)
class CutFEMNonMatchingInterface:
    """Interface described by CutFEM segments on the background mesh.

    Conventions:
      - `n` is oriented from neg -> pos (e.g. poro -> fluid).
      - Fluid outward normal is `nF = -n`.
    """

    pos_elem_ids: Array  # (n_seg,)
    neg_elem_ids: Array  # (n_seg,)
    P0: Array  # (n_seg, 2)
    P1: Array  # (n_seg, 2)
    n: Array  # (n_seg, 2) unit normals (neg->pos)
    h_pos: Array  # (n_seg,)
    h_neg: Array  # (n_seg,)

    def n_segments(self) -> int:
        return int(self.P0.shape[0])


def _find_owner_element(mesh, x: Array, *, tol: float = 1.0e-12) -> int:
    """Return an element id whose reference image contains point `x`."""
    x = np.asarray(x, float).reshape(2,)
    cand = None
    if hasattr(mesh, "find_owner_element_fast"):
        try:
            cand = list(mesh.find_owner_element_fast(x, tol))
        except Exception:
            cand = None
    if not cand:
        cand = list(range(len(mesh.elements_list)))
    for eid in cand:
        try:
            xi, eta = transform.inverse_mapping(mesh, int(eid), x)
        except Exception:
            continue
        if mesh.element_type == "quad":
            if (-1.0 - tol <= float(xi) <= 1.0 + tol) and (-1.0 - tol <= float(eta) <= 1.0 + tol):
                return int(eid)
        else:
            if float(xi) >= -tol and float(eta) >= -tol and float(xi) + float(eta) <= 1.0 + tol:
                return int(eid)
    # centroid fallback
    d = [np.linalg.norm(np.asarray(e.centroid(), float) - x) for e in mesh.elements_list]
    return int(np.argmin(d))


def build_interface_from_cutfem_segments(
    *,
    mesh_f,
    fluid_ls,
    poro_ls,
    mesh_p,
    x0: float,
    tol_inlet: float = 5.0e-6,
    tol_owner: float = 1.0e-10,
) -> tuple[CutFEMNonMatchingInterface, dict[str, Array]]:
    """Build non-matching Γ^FP segments from the already-built CutFEM segments.

    Returns:
      (interface_fp, extra) where `extra` also contains inlet segments:
        extra["inlet_P0"], extra["inlet_P1"] for Γ^{F,N} at x=x0.
    """
    pos_ids: list[int] = []
    neg_ids: list[int] = []
    P0: list[Array] = []
    P1: list[Array] = []
    nn: list[Array] = []
    h_pos: list[float] = []
    h_neg: list[float] = []

    inlet_P0: list[Array] = []
    inlet_P1: list[Array] = []
    inlet_pos_ids: list[int] = []

    eps = 1.0e-8

    for elem in mesh_f.elements_list:
        segs = getattr(elem, "interface_segments", None)
        if not segs:
            continue
        for a, b in segs:
            p0 = np.asarray(a, float)
            p1 = np.asarray(b, float)
            mid = 0.5 * (p0 + p1)
            if abs(float(mid[0]) - float(x0)) <= float(tol_inlet):
                inlet_P0.append(p0)
                inlet_P1.append(p1)
                inlet_pos_ids.append(int(elem.id))
                continue

            # Poro owner element for this segment
            _ = poro_ls  # keep signature explicit (poro geometry may be used in future heuristics)
            neg_eid = _find_owner_element(mesh_p, mid, tol=tol_owner)

            # Discrete segment normal (paper remark): use the piecewise-linear
            # interface geometry rather than the smooth level-set gradient.
            t = p1 - p0
            tnorm = float(np.linalg.norm(t))
            if tnorm <= 1.0e-14:
                continue
            nvec = np.array([t[1], -t[0]], dtype=float) / tnorm
            probe = mid + eps * nvec
            try:
                inside_fluid = float(fluid_ls(probe)) > 0.0
            except Exception:
                inside_fluid = True
            if not inside_fluid:
                nvec = -nvec

            pos_ids.append(int(elem.id))
            neg_ids.append(int(neg_eid))
            P0.append(p0)
            P1.append(p1)
            nn.append(nvec)
            h_pos.append(float(mesh_f.element_char_length(int(elem.id)) or 0.0))
            h_neg.append(float(mesh_p.element_char_length(int(neg_eid)) or 0.0))

    if not P0:
        raise ValueError("No CutFEM interface segments found for Γ^FP (check mesh_f.build_interface_segments).")

    iface = CutFEMNonMatchingInterface(
        pos_elem_ids=np.asarray(pos_ids, dtype=int),
        neg_elem_ids=np.asarray(neg_ids, dtype=int),
        P0=np.asarray(P0, dtype=float),
        P1=np.asarray(P1, dtype=float),
        n=np.asarray(nn, dtype=float),
        h_pos=np.asarray(h_pos, dtype=float),
        h_neg=np.asarray(h_neg, dtype=float),
    )
    extra = dict(
        inlet_pos_elem_ids=np.asarray(inlet_pos_ids, dtype=int),
        inlet_P0=np.asarray(inlet_P0, dtype=float) if inlet_P0 else np.zeros((0, 2), dtype=float),
        inlet_P1=np.asarray(inlet_P1, dtype=float) if inlet_P1 else np.zeros((0, 2), dtype=float),
    )
    return iface, extra

