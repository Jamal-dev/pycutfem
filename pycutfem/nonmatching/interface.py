from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.utils.bitset import BitSet


@dataclass(frozen=True, slots=True)
class NonMatchingInterface:
    """Common-refinement description of a non-matching interface.

    The interface is represented as a set of *overlap* line segments. Each
    overlap segment stores the owning boundary edge (and owner element) from
    both meshes.

    Conventions
    -----------
    - ``neg`` and ``pos`` refer to the two sides of the interface.
    - ``n`` is oriented from the negative side to the positive side (neg -> pos).
    - Each overlap segment is straight (P1 geometry) with endpoints ``P0[i]``,
      ``P1[i]``.
    """

    mesh_neg: Mesh
    mesh_pos: Mesh
    neg_edge_ids: np.ndarray  # (n_overlap,)
    pos_edge_ids: np.ndarray  # (n_overlap,)
    neg_elem_ids: np.ndarray  # (n_overlap,)
    pos_elem_ids: np.ndarray  # (n_overlap,)
    P0: np.ndarray  # (n_overlap, 2)
    P1: np.ndarray  # (n_overlap, 2)
    n: np.ndarray  # (n_overlap, 2) oriented neg->pos
    h_neg: np.ndarray  # (n_overlap,)
    h_pos: np.ndarray  # (n_overlap,)

    def n_segments(self) -> int:
        return int(self.P0.shape[0])


def _to_edge_ids(mesh: Mesh, sel: str | BitSet | Iterable[int] | None) -> np.ndarray:
    if sel is None:
        raise ValueError("Edge selection is required for a non-matching interface.")
    if isinstance(sel, str):
        return np.asarray(mesh.edge_bitset(sel).to_indices(), dtype=int)
    if isinstance(sel, BitSet):
        return np.asarray(sel.to_indices(), dtype=int)
    return np.asarray(list(int(i) for i in sel), dtype=int)


def build_nonmatching_interface(
    *,
    mesh_neg: Mesh,
    mesh_pos: Mesh,
    neg_edges: str | BitSet | Iterable[int] = "interface",
    pos_edges: str | BitSet | Iterable[int] = "interface",
    tol: float = 1.0e-12,
) -> NonMatchingInterface:
    """Pair two non-matching interface edge sets via a common 1D refinement.

    This routine assumes both interface edge sets lie on the same (piecewise)
    straight curve and are ordered along a dominant tangent direction. It is
    robust for the MMS test-cases in this repository (straight interface).
    """

    neg_ids = _to_edge_ids(mesh_neg, neg_edges)
    pos_ids = _to_edge_ids(mesh_pos, pos_edges)
    if neg_ids.size == 0 or pos_ids.size == 0:
        raise ValueError("Empty interface edge set(s). Tag boundary edges first.")

    # Collect segments ---------------------------------------------------------
    def _collect(mesh: Mesh, eids: np.ndarray):
        out = []
        for gid in np.asarray(eids, dtype=int).ravel():
            e = mesh.edge(int(gid))
            if e.right is not None:
                # Interface is expected to be a boundary tag on each submesh.
                continue
            owner = e.left
            if owner is None:
                continue
            p0 = np.asarray(mesh.nodes_x_y_pos[int(e.nodes[0])], dtype=float)
            p1 = np.asarray(mesh.nodes_x_y_pos[int(e.nodes[1])], dtype=float)
            d = p1 - p0
            L = float(np.linalg.norm(d))
            if L <= float(tol):
                continue
            n = np.asarray(getattr(e, "normal", None), dtype=float)
            if n.shape != (2,):
                # Fallback: normal from tangent.
                tloc = d / L
                n = np.array([tloc[1], -tloc[0]], dtype=float)
            out.append(
                {
                    "gid": int(gid),
                    "eid": int(owner),
                    "p0": p0,
                    "p1": p1,
                    "n": n,
                }
            )
        if not out:
            raise ValueError("No valid boundary edges found for the interface tag.")
        return out

    neg = _collect(mesh_neg, neg_ids)
    pos = _collect(mesh_pos, pos_ids)

    # Global tangent direction -------------------------------------------------
    d0 = neg[0]["p1"] - neg[0]["p0"]
    t = d0 / max(float(np.linalg.norm(d0)), float(tol))
    # Pick a stable reference point on the supporting line.
    p_ref = np.asarray(neg[0]["p0"], dtype=float)
    s_ref = float(np.dot(p_ref, t))

    # Convert each edge segment to 1D intervals along t ------------------------
    def _as_intervals(items: list[dict]) -> list[dict]:
        out = []
        for it in items:
            p0 = np.asarray(it["p0"], float)
            p1 = np.asarray(it["p1"], float)
            s0 = float(np.dot(p0, t))
            s1 = float(np.dot(p1, t))
            if s1 < s0:
                s0, s1 = s1, s0
                p0, p1 = p1, p0
            out.append(
                {
                    "s0": s0,
                    "s1": s1,
                    "gid": int(it["gid"]),
                    "eid": int(it["eid"]),
                    "n": np.asarray(it["n"], float),
                }
            )
        out.sort(key=lambda d: d["s0"])
        return out

    neg_int = _as_intervals(neg)
    pos_int = _as_intervals(pos)

    # Sweep to compute overlap intervals --------------------------------------
    i = 0
    j = 0
    seg_neg_gid: list[int] = []
    seg_pos_gid: list[int] = []
    seg_neg_eid: list[int] = []
    seg_pos_eid: list[int] = []
    seg_P0: list[np.ndarray] = []
    seg_P1: list[np.ndarray] = []
    seg_n: list[np.ndarray] = []
    seg_hn: list[float] = []
    seg_hp: list[float] = []

    while i < len(neg_int) and j < len(pos_int):
        a = neg_int[i]
        b = pos_int[j]
        s0 = max(float(a["s0"]), float(b["s0"]))
        s1 = min(float(a["s1"]), float(b["s1"]))
        if s1 - s0 > float(tol):
            P0 = p_ref + (s0 - s_ref) * t
            P1 = p_ref + (s1 - s_ref) * t
            seg_neg_gid.append(int(a["gid"]))
            seg_pos_gid.append(int(b["gid"]))
            seg_neg_eid.append(int(a["eid"]))
            seg_pos_eid.append(int(b["eid"]))
            seg_P0.append(np.asarray(P0, float))
            seg_P1.append(np.asarray(P1, float))
            seg_n.append(np.asarray(a["n"], float))
            seg_hn.append(float(mesh_neg.element_char_length(int(a["eid"])) or 0.0))
            seg_hp.append(float(mesh_pos.element_char_length(int(b["eid"])) or 0.0))

        # Advance the interval that ends first.
        if float(a["s1"]) < float(b["s1"]) - float(tol):
            i += 1
        else:
            j += 1

    if not seg_P0:
        raise ValueError("Interface pairing produced no overlap segments.")

    P0_arr = np.asarray(seg_P0, dtype=float)
    P1_arr = np.asarray(seg_P1, dtype=float)
    n_arr = np.asarray(seg_n, dtype=float)
    # Robust normal orientation (neg -> pos) based on owner element centroids.
    cpos = np.asarray([mesh_pos.elements_list[int(eid)].centroid() for eid in seg_pos_eid], dtype=float)
    cneg = np.asarray([mesh_neg.elements_list[int(eid)].centroid() for eid in seg_neg_eid], dtype=float)
    orient_vec = cpos - cneg
    orient_norm = np.linalg.norm(orient_vec, axis=1)
    dotv = np.einsum("ij,ij->i", n_arr, orient_vec)
    flip = (orient_norm > float(tol)) & (dotv < 0.0)
    if np.any(flip):
        n_arr[flip, :] *= -1.0
    # Normalize normals defensively.
    nn = np.linalg.norm(n_arr, axis=1)
    nn = np.where(nn > 0.0, nn, 1.0)
    n_arr = n_arr / nn[:, None]

    return NonMatchingInterface(
        mesh_neg=mesh_neg,
        mesh_pos=mesh_pos,
        neg_edge_ids=np.asarray(seg_neg_gid, dtype=int),
        pos_edge_ids=np.asarray(seg_pos_gid, dtype=int),
        neg_elem_ids=np.asarray(seg_neg_eid, dtype=int),
        pos_elem_ids=np.asarray(seg_pos_eid, dtype=int),
        P0=P0_arr,
        P1=P1_arr,
        n=n_arr,
        h_neg=np.asarray(seg_hn, dtype=float),
        h_pos=np.asarray(seg_hp, dtype=float),
    )
