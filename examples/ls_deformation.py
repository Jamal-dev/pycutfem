"""Deformation-driven cut-integration demo (clean baseline).

This script compares P1-only and P1+deformation area and interface-length
results on both tri and quad meshes, printing outputs similar in style to the
user-provided reference. All geometry/integration logic is reused from
pycutfem modules to avoid duplication.
"""
from __future__ import annotations
import math
import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.utils.meshgen import structured_triangles, structured_quad
from pycutfem.integration.quadrature import volume as vol_rule, isoparam_interface_line_quadrature_batch
from pycutfem.integration.cut_integration import (
    CutIntegration,
)
from pycutfem.fem import transform
from pycutfem.core.levelset import CircleLevelSet, LevelSetMeshAdaptation
from pycutfem.core.sideconvention import SIDE


def make_meshes(L=1.5, H=1.5, nx=24, ny=24, geom_order: int = 3):
    """Return (tri_mesh, quad_mesh) centered at origin with given geometry order."""
    offset = [-L / 2, -H / 2]
    nT, eT, _, cT = structured_triangles(L, H, nx_quads=nx, ny_quads=ny, poly_order=geom_order, offset=offset)
    mesh_tri = Mesh(nT, eT, elements_corner_nodes=cT, element_type="tri", poly_order=geom_order)
    nQ, eQ, _, cQ = structured_quad(L, H, nx=nx, ny=ny, poly_order=geom_order, offset=offset)
    mesh_quad = Mesh(nQ, eQ, elements_corner_nodes=cQ, element_type="quad", poly_order=geom_order)
    return mesh_tri, mesh_quad


# ------------------------------- area integrals -----------------------------
# ------------------------------- area integrals -----------------------------
def area_on_cuts_reference_quad(mesh, cut_ids, lset_p1, deformation, *, side: str = '+', qvol: int = 6, tol: float = 1e-12) -> float:
    """Area contribution from cut quads using a straight‑cut rule in reference.

    For each reference quadrature point (ξ,η) on the kept subregion, multiply by
    |det(J_g + J_d)| to integrate in physical coordinates.
    """
    order_y = max(2, int(qvol) // 2)
    order_x = max(2, int(qvol) // 2)
    ref_geom = transform.get_reference(mesh.element_type, mesh.poly_order)
    A = 0.0
    for eid in cut_ids:
        qp_ref, qw_ref = CutIntegration.straight_cut_rule_quad_ref(mesh, int(eid), lset_p1,
                                                                   side=side, order_y=order_y, order_x=order_x, tol=tol)
        if qp_ref.size == 0:
            continue
        for (xi, eta), w in zip(qp_ref, qw_ref):
            det_tot = abs(float(np.linalg.det(transform.jacobian(mesh, int(eid), (float(xi), float(eta))))))
            if deformation is not None:
                dN = ref_geom.grad(float(xi), float(eta))
                conn = np.asarray(mesh.elements_connectivity[int(eid)], dtype=int)
                Uloc = np.asarray(deformation.node_displacements[conn], float)
                Jd = Uloc.T @ dN
                Jg = transform.jacobian(mesh, int(eid), (float(xi), float(eta)))
                det_tot = abs(float(np.linalg.det(Jg + Jd)))
            A += float(w) * float(det_tot)
    return float(A)


def area_on_cuts_reference_tri(mesh, cut_ids, lset_p1, deformation, ref_geom, *, side: str = '+', qvol: int = 8, tol: float = 1e-12) -> float:
    """Area contribution from cut triangles using reference‑space clipping and fan triangulation."""
    from pycutfem.ufl.helpers_geom import clip_triangle_to_side as _clip_tri
    from pycutfem.integration.quadrature import tri_rule
    R = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], float)
    qp_ref, qw_ref = tri_rule(int(qvol))

    def _acc_one(eid: int, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
        J = np.column_stack((B - A, C - A))
        area_ref = 0.5 * abs(float(np.linalg.det(J)))
        if area_ref <= 0.0:
            return 0.0
        acc = 0.0
        for (l1, l2), w in zip(qp_ref, qw_ref):
            l0 = 1.0 - l1 - l2
            xi_eta = l0 * A + l1 * B + l2 * C
            Jg = transform.jacobian(mesh, int(eid), (float(xi_eta[0]), float(xi_eta[1])))
            if deformation is None:
                det_tot = abs(float(np.linalg.det(Jg)))
            else:
                dN = ref_geom.grad(float(xi_eta[0]), float(xi_eta[1]))
                conn = np.asarray(mesh.elements_connectivity[int(eid)], dtype=int)
                Uloc = np.asarray(deformation.node_displacements[conn], float)
                Jd = Uloc.T @ dN
                det_tot = abs(float(np.linalg.det(Jg + Jd)))
            acc += float(w) * float(det_tot)
        return (2.0 * area_ref) * acc

    A_total = 0.0
    for eid0 in cut_ids:
        eid = int(eid0)
        phiV = np.array([lset_p1.value_on_element(eid, (float(R[i, 0]), float(R[i, 1]))) for i in range(3)], float)
        polys = _clip_tri(R, phiV, side=side, eps=tol)
        for poly in polys:
            if len(poly) < 3:
                continue
            p0 = np.asarray(poly[0], float)
            for i in range(1, len(poly) - 1):
                A_total += _acc_one(eid, p0, np.asarray(poly[i], float), np.asarray(poly[i + 1], float))
    return float(A_total)


def compute_A_side_with_deformation(mesh, lset_p1, deformation, *, side: str = '+', qvol: int = 6, tol: float = 1e-12, nseg_hint=None, lset_pn=None) -> float:
    """Integrate area of {Ihφ ▷ 0} after mesh deformation.

    Strategy:
    - Full elements: integrate with det(J_geom + J_def).
    - Cut elements: integrate a reference straight‑cut (quad) or clipped fan rule (tri),
      scaled by the deformed Jacobian.
    """
    from pycutfem.integration.quadrature import volume as volume_rule
    ref = transform.get_reference(mesh.element_type, mesh.poly_order)
    qp_ref, qw_ref = volume_rule(mesh.element_type, qvol)

    # 1) classify on Ihφ
    inside_ids, outside_ids, cut_ids = mesh.classify_elements(lset_p1, tol=tol)

    # 2) full elements with deformed Jacobian
    if side == '+':
        full_eids = outside_ids if SIDE.pos_is_phi_nonnegative else inside_ids
    else:
        full_eids = inside_ids if SIDE.pos_is_phi_nonnegative else outside_ids

    A = 0.0
    for eid0 in full_eids:
        eid = int(eid0)
        for (xi, eta), w in zip(qp_ref, qw_ref):
            Jg = transform.jacobian(mesh, eid, (float(xi), float(eta)))
            if deformation is None:
                det_tot = abs(float(np.linalg.det(Jg)))
            else:
                dN = ref.grad(float(xi), float(eta))
                conn = np.asarray(mesh.elements_connectivity[int(eid)], dtype=int)
                Uloc = np.asarray(deformation.node_displacements[conn], float)
                Jd = Uloc.T @ dN
                det_tot = abs(float(np.linalg.det(Jg + Jd)))
            A += float(w) * float(det_tot)

    # 3) cut elements via reference straight‑cut rules
    if mesh.element_type == 'quad':
        A += area_on_cuts_reference_quad(mesh, cut_ids, lset_p1, deformation, side=side, qvol=qvol, tol=tol)
    else:
        A += area_on_cuts_reference_tri(mesh, cut_ids, lset_p1, deformation, ref, side=side, qvol=qvol, tol=tol)
    return float(A)


def compute_interface_length(mesh, level_set, *, deformation=None, order: int = 12, tol: float = 1e-12) -> float:
    """Total |Γ| via Qp isoparametric interface quadrature + deformation stretch.

    - Builds per-cut element endpoints (P0,P1) using Mesh.build_interface_segments.
    - Uses isoparametric polyline + Gauss quadrature on the geometric interface.
    - If a deformation is supplied, each geometric weight is scaled by
      || (I + ∂u/∂x) · t̂ || at the quadrature point, where t̂ is the geometric
      unit tangent reconstructed from the level-set normal.
    """
    if not hasattr(level_set, "value_on_element"):
        raise NotImplementedError("Only FE-backed level set supported here.")

    # build endpoints on geometry
    mesh.build_interface_segments(level_set, tol=float(tol))
    _, _, cut_ids = mesh.classify_elements(level_set, tol=tol)
    P0_list, P1_list, eids_list = [], [], []
    for eid in cut_ids:
        pts = getattr(mesh.elements_list[int(eid)], 'interface_pts', [])
        if len(pts) >= 2:
            P0_list.append(np.asarray(pts[0], float))
            P1_list.append(np.asarray(pts[1], float))
            eids_list.append(int(eid))
    if not P0_list:
        return 0.0

    P0 = np.asarray(P0_list, float)
    P1 = np.asarray(P1_list, float)
    eids = np.asarray(eids_list, int)

    # Geometric line quadrature (points x and weights for |x'|)
    qpts, qw_geom = isoparam_interface_line_quadrature_batch(level_set, P0, P1,
                                                             p=int(getattr(mesh, 'poly_order', 1)),
                                                             order=int(order), project_steps=3, tol=float(tol),
                                                             mesh=mesh, eids=eids)

    if deformation is None:
        return float(np.sum(qw_geom))

    # Deformation-aware factor per point: || (I + ∂u/∂x) · t̂ ||
    ref_geom = transform.get_reference(mesh.element_type, mesh.poly_order)
    L = 0.0
    for i, eid in enumerate(eids):
        for q, x in enumerate(qpts[i]):
            # Inverse-map once per point
            xi, eta = transform.inverse_mapping(mesh, int(eid), np.asarray(x, float))
            # geometric Jacobian and its inverse
            Jg = transform.jacobian(mesh, int(eid), (float(xi), float(eta)))
            A = np.linalg.inv(Jg)
            # displacement gradient in physical coords: (U^T dN_ref) @ J^{-1}
            dN_ref = np.asarray(ref_geom.grad(float(xi), float(eta)), float)  # (nloc,2)
            conn = np.asarray(mesh.elements_connectivity[int(eid)], dtype=int)
            Uloc = np.asarray(deformation.node_displacements[conn], float)     # (nloc,2)
            G_ref = Uloc.T @ dN_ref                                            # (2,2)
            G_phys = G_ref @ A
            F = np.eye(2) + G_phys

            # geometric tangent via rotated LS normal (unit)
            if hasattr(level_set, 'gradient_on_element'):
                n = np.asarray(level_set.gradient_on_element(int(eid), (float(xi), float(eta))), float)
            else:
                n = np.asarray(level_set.gradient(np.asarray(x, float)), float)
            nn = np.linalg.norm(n)
            if nn <= 1e-30:
                # Fallback: finite-diff approx with a small step in xi
                h = 1e-8
                x_eps = transform.x_mapping(mesh, int(eid), (float(xi + h), float(eta)))
                n = np.asarray(level_set.gradient(np.asarray(x_eps, float)), float) - np.asarray(level_set.gradient(np.asarray(x, float)), float)
                nn = np.linalg.norm(n)
            t_unit = np.array([n[1], -n[0]], float)
            nt = np.linalg.norm(t_unit)
            if nt > 0.0:
                t_unit = t_unit / nt
            else:
                t_unit = np.array([1.0, 0.0], float)

            L += float(qw_geom[i, q]) * float(np.linalg.norm(F @ t_unit))

    return float(L)


def _compute_A_side_like_compiler(mesh: Mesh, level_set, *, side='+', qvol=4, nseg_hint=None, tol=1e-12):
    """Compiler-style area integration: full elements via volume rule, cut via curved triangles.

    This reuses the public helpers in the repository (corner-tris + curved rule).
    """
    from pycutfem.ufl.helpers_geom import corner_tris, curved_subcell_quadrature_for_cut_triangle

    inside_ids, outside_ids, cut_ids = mesh.classify_elements(level_set, tol=tol)
    qp_ref, qw_ref = vol_rule(mesh.element_type, qvol)
    full_eids = outside_ids if (side == '+') else inside_ids
    A = 0.0
    for eid in full_eids:
        for (xi, eta), w in zip(qp_ref, qw_ref):
            detJ = abs(np.linalg.det(transform.jacobian(mesh, int(eid), (float(xi), float(eta)))))
            A += w * detJ
    for eid in cut_ids:
        elem = mesh.elements_list[int(eid)]
        tri_list, corner_ids = corner_tris(mesh, elem)
        for loc_tri in tri_list:
            qx, qw = curved_subcell_quadrature_for_cut_triangle(
                mesh, int(eid), loc_tri, list(corner_ids), level_set,
                side=side, qvol=qvol, nseg_hint=nseg_hint, tol=tol
            )
            A += float(qw.sum())
    return float(A)


def _mapping_residual_report(mesh, lset_p1, phi, deformation, *, qvol=6, tol=1e-12):
    """Print simple residual stats on cut elements: r = Ihφ - φ(Θ_h)."""
    from pycutfem.integration.quadrature import volume as vol
    qp, qw = vol(mesh.element_type, qvol)
    _, _, cut_ids = mesh.classify_elements(lset_p1, tol=tol)
    vals = []
    for eid in cut_ids:
        for (xi, eta) in qp:
            x_def = deformation.mapped_point(int(eid), (float(xi), float(eta)))
            r = float(lset_p1.value_on_element(int(eid), (float(xi), float(eta))) - phi(x_def))
            g = np.asarray(phi.gradient(x_def), float)
            d = abs(r) / (np.linalg.norm(g) + 1e-30)
            vals.append((abs(r), d))
    if not vals:
        print("=== Mapping residual check on CUT elements ===\n(no cut elements)\n")
        return
    r_abs = np.array([v[0] for v in vals], float)
    dist = np.array([v[1] for v in vals], float)
    def _fmt(a): return f"max={a.max():.3e}, mean={a.mean():.3e}, p95={np.quantile(a,0.95):.3e}, p99={np.quantile(a,0.99):.3e}"
    print("=== Mapping residual check on CUT elements ===")
    print(f"|Ihφ - φ(Θ_h)|: {_fmt(r_abs)}")
    print(f"distance ≈ |r|/||∇φ||: {_fmt(dist)}\n")


def main():
    L, H = 2.0, 2.0
    maxh = 0.125
    R = 2.0 / 3.0
    geom_poly_order = 2
    nsegs_deformation = 13
    q_vol = 6
    max_newton_steps = 15
    print(f"nsg for deformation: {nsegs_deformation}, q_vol: {q_vol}")

    nx = int(L / maxh)
    ny = int(H / maxh)
    mesh_tri, mesh_quad = make_meshes(L=L, H=H, nx=nx, ny=ny, geom_order=geom_poly_order)
    phi = CircleLevelSet(center=(0.0, 0.0), radius=R)
    A_exact_inside = math.pi * R * R
    A_exact_outside = L * H - A_exact_inside
    rel = lambda A, Aex: (A - Aex) / Aex

    print("== TRI mesh (high-order φ reference) ==")
    adap = LevelSetMeshAdaptation(mesh_tri, order=3, threshold=10.5, max_steps=max_newton_steps)
    deformation_tri = adap.calc_deformation(phi, q_vol=q_vol + 2)
    lset_p1 = adap.lset_p1; assert lset_p1 is not None
    _mapping_residual_report(mesh_tri, lset_p1, phi, deformation_tri, qvol=q_vol)

    print("== TRI mesh: P1 vs P1 + deformation ==")
    nseg = 9
    A_p1_pos = _compute_A_side_like_compiler(mesh_tri, lset_p1, side='+', qvol=q_vol, nseg_hint=nseg)
    A_p1_neg = _compute_A_side_like_compiler(mesh_tri, lset_p1, side='-', qvol=q_vol, nseg_hint=nseg)
    A_def_pos_tri = compute_A_side_with_deformation(mesh_tri, lset_p1, deformation_tri, side='+', qvol=q_vol, nseg_hint=nsegs_deformation)
    A_def_neg_tri = compute_A_side_with_deformation(mesh_tri, lset_p1, deformation_tri, side='-', qvol=q_vol, nseg_hint=nsegs_deformation)
    print(f"P1 only         : A+={A_p1_pos:.8f} rel.err={rel(A_p1_pos, A_exact_outside):.3e}; A-={A_p1_neg:.8f} rel.err={rel(A_p1_neg, A_exact_inside):.3e}")

    # mismatch φ(Θ_h) vs Ihφ on cut tris
    xi = np.linspace(0.0, 1.0, 6)
    mismatches = []
    for eid in range(mesh_tri.num_elements()):
        elem = mesh_tri.elements_list[eid]
        if getattr(elem, 'tag', '') != 'cut':
            continue
        for r in xi:
            for s in xi:
                if r + s > 1.0 + 1e-12:
                    continue
                x_def = deformation_tri.mapped_point(eid, (float(r), float(s)))
                phi_true = float(phi(x_def))
                phi_lin = float(lset_p1.value_on_element(eid, (float(r), float(s))))
                mismatches.append(abs(phi_true - phi_lin))
    if mismatches:
        print(f"P1 + deformation: A+={A_def_pos_tri:.8f} rel.err={rel(A_def_pos_tri, A_exact_outside):.3e}; A-={A_def_neg_tri:.8f} rel.err={rel(A_def_neg_tri, A_exact_inside):.3e}")
        print(f"max |phi(mapped)-phi_p1| over cut tris: {max(mismatches):.3e}")

    print("\n== QUAD mesh (high-order φ reference) ==")
    adap_q = LevelSetMeshAdaptation(mesh_quad, order=2, threshold=10.5, max_steps=max_newton_steps)
    deformation_quad = adap_q.calc_deformation(phi, q_vol=q_vol + 2)
    lset_p1_q = adap_q.lset_p1; assert lset_p1_q is not None
    _mapping_residual_report(mesh_quad, lset_p1_q, phi, deformation_quad, qvol=q_vol)

    A_p1_pos_q = _compute_A_side_like_compiler(mesh_quad, lset_p1_q, side='+', qvol=q_vol, nseg_hint=nsegs_deformation)
    A_p1_neg_q = _compute_A_side_like_compiler(mesh_quad, lset_p1_q, side='-', qvol=q_vol, nseg_hint=nsegs_deformation)
    A_def_pos_q = compute_A_side_with_deformation(mesh_quad, lset_p1_q, deformation_quad, side='+', qvol=q_vol, nseg_hint=nsegs_deformation)
    A_def_neg_q = compute_A_side_with_deformation(mesh_quad, lset_p1_q, deformation_quad, side='-', qvol=q_vol, nseg_hint=nsegs_deformation)
    def exact_diff(A, Aex): return A - Aex
    print("== QUAD mesh: P1 vs P1 + deformation ==")
    print(f"P1 only         : A+={A_p1_pos_q:.8f} rel.err={rel(A_p1_pos_q, A_exact_outside):.3e}; A-={A_p1_neg_q:.8f} rel.err={rel(A_p1_neg_q, A_exact_inside):.3e}")
    print(f"P1 + deformation: A+={A_def_pos_q:.8f} rel.err={rel(A_def_pos_q, A_exact_outside):.3e}; A-={A_def_neg_q:.8f} rel.err={rel(A_def_neg_q, A_exact_inside):.3e}")

    A_pos_def_diff = exact_diff(A_def_pos_q, A_exact_outside)
    A_neg_def_diff = exact_diff(A_def_neg_q, A_exact_inside)
    A_pos_def_tri_diff = exact_diff(A_def_pos_tri, A_exact_outside)
    A_neg_def_tri_diff = exact_diff(A_def_neg_tri, A_exact_inside)
    print(f"Tri result with deformation: A+ :{A_pos_def_tri_diff:.3e}, A- :{A_neg_def_tri_diff:.3e}")
    print(f"Quad result with deformation: A+ :{A_pos_def_diff:.3e}, A- :{A_neg_def_diff:.3e}")
    print(f"NGSolve result with deformation: A+ :{-9.74e-06:.3e}, A- :{9.74e-06:.3e}")
    rd = lambda a, b: (a - b) / (b if b != 0.0 else 1.0)
    print(f"Relative error with NGSolve results: A+ diff: {rd(A_pos_def_diff, -9.74e-06):.3e}, A- diff: {rd(A_neg_def_diff, 9.74e-06):.3e}")

    print("\nInterface Linear Length Checks:")
    exact_circumference = 2.0 * math.pi * R
    L_p1 = compute_interface_length(mesh_quad, lset_p1_q, deformation=None, order=4)
    L_def = compute_interface_length(mesh_quad, lset_p1_q, deformation=deformation_quad, order=4)
    print(f"Exact circumference: {exact_circumference:.8f}")
    print(f"Level set φ_P1     : {L_p1:.8f}, rel.err={(L_p1-exact_circumference)/exact_circumference:.3e}")
    print(f"Deformed mesh      : {L_def:.8f}, rel.err={(L_def-exact_circumference)/exact_circumference:.3e}")


if __name__ == "__main__":
    main()
