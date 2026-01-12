"""pycutfem.integration.cut_integration

Straight‑cut reference rules and iso‑curve rules grouped in a single module.

- Primary public API is provided by the static methods in ``CutIntegration``.
- Higher‑level area integrators with deformation are provided as functions
  that reuse the static rules; this keeps a single source of truth for the
  reference constructions and avoids duplication.
"""
from __future__ import annotations
from typing import Tuple, Optional, List
import numpy as np

from pycutfem.fem import transform
from pycutfem.ufl.helpers_geom import phi_eval
from pycutfem.core.sideconvention import SIDE
from pycutfem.integration.quadrature import gauss_legendre, volume as volume_rule, tri_rule


class CutIntegration:
    """Static helpers for reference straight‑cut and iso‑curve rules.

    Notes:
    - Reuses transform and quadrature utilities exclusively.
    - Honors global SIDE convention for sign and zero ownership.
    """

    # ------------------------------- low‑level utils -----------------------
    @staticmethod
    def _brent01(f, a: float, b: float, fa: float, fb: float, tol: float = 1e-14, itmax: int = 60) -> float:
        """Small robust root finder on [a,b] with endpoint guards."""
        if fa == 0.0:
            return a
        if fb == 0.0:
            return b
        x0, x1, f0, f1 = a, b, fa, fb
        for _ in range(int(itmax)):
            c = x1 - f1 * (x1 - x0) / (f1 - f0 + 1e-300)
            if not (min(a, b) <= c <= max(a, b)):
                c = 0.5 * (a + b)
            fc = f(c)
            if abs(fc) < tol or abs(b - a) < tol:
                return c
            if (f0 > 0) != (fc > 0):
                b, fb, x1, f1 = c, fc, c, fc
            else:
                a, fa, x0, f0 = c, fc, c, fc
        return 0.5 * (a + b)

    @staticmethod
    def _segment_rule_1d(x0: float, x1: float, order: int):
        xi, w = gauss_legendre(int(order))
        xm = 0.5 * (x0 + x1)
        xr = 0.5 * (x1 - x0)
        x = xm + xr * xi
        wt = xr * w
        return x, wt

    # ---------------------------- QUAD straight‑cut ------------------------
    @staticmethod
    def _edge_phi_on_eta(mesh, eid: int, lset, x_fixed: float, eta: float) -> float:
        # Prefer element-aware evaluation for FE-backed level sets to avoid
        # a costly owner search / inverse mapping.
        if hasattr(lset, "value_on_element"):
            return float(lset.value_on_element(int(eid), (float(x_fixed), float(eta))))
        x_phys = transform.x_mapping(mesh, int(eid), (float(x_fixed), float(eta)))
        return float(phi_eval(lset, x_phys))

    @staticmethod
    def _edge_zero_eta(mesh, eid: int, lset_p1, x_fixed: float, tol: float) -> List[float]:
        f = lambda e: CutIntegration._edge_phi_on_eta(mesh, eid, lset_p1, x_fixed, e)
        e0, e1 = -1.0, +1.0
        f0, f1 = f(e0), f(e1)
        isz0 = abs(f0) <= tol
        isz1 = abs(f1) <= tol
        hascut = (f0 * f1 < 0.0) or (isz0 and f1 < -tol) or (isz1 and f0 < -tol)
        if not hascut:
            return []
        return [CutIntegration._brent01(f, e0, e1, f0, f1, tol=tol)]

    @staticmethod
    def _horizontal_cut_segment(mesh, eid, lset_p1, eta, side, order, tol):
        # Build Q1 REF coeffs once
        a00, a10, a01, a11 = CutIntegration._q1_coeffs_ref_from_corners(mesh, int(eid), lset_p1)
        # φ_ref(ξ,η)=a00 + a10 ξ + a01 η + a11 ξ η
        denom = (a10 + a11 * eta)
        num   = -(a00 + a01 * eta)
        hascut = abs(denom) > tol and (-1.0 - 1e-12) <= (num/denom) <= (1.0 + 1e-12)

        want_pos = (side == '+')
        fl = a00 + a10*(-1.0) + a01*eta + a11*(-1.0)*eta
        fr = a00 + a10*(+1.0) + a01*eta + a11*(+1.0)*eta
        keep_left  = SIDE.is_pos(fl, tol) if want_pos else SIDE.is_neg(fl, tol)
        keep_right = SIDE.is_pos(fr, tol) if want_pos else SIDE.is_neg(fr, tol)

        if not hascut:
            if keep_left and keep_right:
                return CutIntegration._segment_rule_1d(-1.0, +1.0, order)
            return np.empty((0,), float), np.empty((0,), float)

        # exact REF root at this η (clamp for safety)
        xi_cut = float(num / denom)
        xi_cut = -1.0 if xi_cut < -1.0 else (1.0 if xi_cut > 1.0 else xi_cut)

        # Robust selection: decide which sub-interval belongs to the requested side by
        # sampling φ strictly inside each candidate interval. This avoids degeneracies
        # when the cut hits an endpoint (φ=0 at ξ=±1), which can otherwise cause
        # + and - rules to overlap and double-count.
        def _phi_at(xi: float) -> float:
            return float(a00 + a10 * xi + a01 * eta + a11 * xi * eta)

        intervals: list[tuple[float, float, float]] = []
        left_len = xi_cut + 1.0
        right_len = 1.0 - xi_cut
        if left_len > 1e-14:
            xi_m = -1.0 + 0.5 * left_len
            intervals.append((-1.0, xi_cut, _phi_at(xi_m)))
        if right_len > 1e-14:
            xi_m = xi_cut + 0.5 * right_len
            intervals.append((xi_cut, 1.0, _phi_at(xi_m)))

        if want_pos:
            keep = [(a, b) for (a, b, phi_m) in intervals if SIDE.is_pos(phi_m, tol)]
        else:
            keep = [(a, b) for (a, b, phi_m) in intervals if SIDE.is_neg(phi_m, tol)]

        if not keep:
            return np.empty((0,), float), np.empty((0,), float)
        if len(keep) == 2:
            return CutIntegration._segment_rule_1d(-1.0, +1.0, order)
        a, b = keep[0]
        return CutIntegration._segment_rule_1d(float(a), float(b), order)

    @staticmethod
    def straight_cut_rule_quad_ref(mesh, eid: int, lset_p1, *, side: str = '+', order_y: int = 3, order_x: int = 3, tol: float = 1e-12):
        """Straight‑cut quadrature for QUAD elements in reference domain.

        Returns (qpts(N,2), qw(N,)) on the parent quad [-1,1]^2.
        """
        eta_breaks = [-1.0, +1.0]
        eta_breaks += CutIntegration._edge_zero_eta(mesh, eid, lset_p1, x_fixed=-1.0, tol=tol)
        eta_breaks += CutIntegration._edge_zero_eta(mesh, eid, lset_p1, x_fixed=+1.0, tol=tol)
        eta_breaks = sorted(set([float(e) for e in eta_breaks]))
        if len(eta_breaks) < 2:
            return np.empty((0, 2), float), np.empty((0,), float)

        qpts = []
        qwts = []
        for k in range(len(eta_breaks) - 1):
            e0, e1 = eta_breaks[k], eta_breaks[k + 1]
            if e1 - e0 <= 1e-16:
                continue
            eta_i, w_eta = CutIntegration._segment_rule_1d(e0, e1, order_y)
            for (eta, wy) in zip(eta_i, w_eta):
                xi_j, wx = CutIntegration._horizontal_cut_segment(mesh, eid, lset_p1, float(eta), side, order_x, tol)
                if xi_j.size == 0:
                    continue
                for (xi, wxj) in zip(xi_j, wx):
                    qpts.append([float(xi), float(eta)])
                    qwts.append(float(wy) * float(wxj))
        if not qpts:
            return np.empty((0, 2), float), np.empty((0,), float)
        return np.asarray(qpts, float), np.asarray(qwts, float)

    # ---------------------------- TRI straight‑cut -------------------------
    @staticmethod
    def straight_cut_rule_tri_ref(
        mesh,
        eid: int,
        lset_p1,
        *,
        side: str = "+",
        order: int = 3,
        tol: float = 1e-12,
    ):
        """Straight-cut quadrature for TRI elements in the parent reference triangle.

        Returns (qpts(N,2), qw(N,)) on the parent reference triangle:
          (0,0) - (1,0) - (0,1).

        Notes
        -----
        This is exact for piecewise-linear level sets (P1/Q1) because the cut
        region is a polygon in reference space that can be triangulated.
        """
        from pycutfem.ufl.helpers_geom import clip_triangle_to_side, fan_triangulate, map_ref_tri_to_phys

        R = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
        vals = np.empty(3, dtype=float)
        if hasattr(lset_p1, "value_on_element"):
            for i in range(3):
                vals[i] = float(lset_p1.value_on_element(int(eid), (float(R[i, 0]), float(R[i, 1]))))
        else:
            for i in range(3):
                x_phys = transform.x_mapping(mesh, int(eid), (float(R[i, 0]), float(R[i, 1])))
                vals[i] = float(phi_eval(lset_p1, x_phys))

        polys = clip_triangle_to_side(R, vals, side=str(side), eps=float(tol))
        if not polys:
            return np.empty((0, 2), float), np.empty((0,), float)

        qp_std, qw_std = tri_rule(int(order))
        qpts = []
        qwts = []
        for (A, B, C) in fan_triangulate(polys[0]):
            q_sub, w_sub = map_ref_tri_to_phys(A, B, C, qp_std, qw_std)
            qpts.append(np.asarray(q_sub, float))
            qwts.append(np.asarray(w_sub, float).ravel())

        if not qpts:
            return np.empty((0, 2), float), np.empty((0,), float)
        return np.vstack(qpts), np.concatenate(qwts)

    # ---------------------- Iso‑curve on Ihφ_ref = 0 -----------------------
    @staticmethod
    def _q1_coeffs_ref_from_corners(mesh, eid: int, lset) -> Tuple[float, float, float, float]:
        R = np.array([[-1.0, -1.0], [+1.0, -1.0], [+1.0, +1.0], [-1.0, +1.0]], dtype=float)
        # Evaluate φ at the four reference corners. For FE-backed level sets,
        # use the element-local fast path (no mapping needed).
        if hasattr(lset, "value_on_element"):
            v = np.array(
                [float(lset.value_on_element(int(eid), (float(R[i, 0]), float(R[i, 1])))) for i in range(4)],
                dtype=float,
            )
        else:
            vals = []
            for i in range(4):
                x_phys = transform.x_mapping(mesh, int(eid), (float(R[i, 0]), float(R[i, 1])))
                vals.append(float(phi_eval(lset, x_phys)))
            v = np.array(vals, dtype=float)
        a00 = 0.25 * (v[0] + v[1] + v[2] + v[3])
        a10 = 0.25 * (-v[0] + v[1] + v[2] - v[3])
        a01 = 0.25 * (-v[0] - v[1] + v[2] + v[3])
        a11 = 0.25 * (+v[0] - v[1] + v[2] - v[3])
        return a00, a10, a01, a11

    @staticmethod
    def _edge_intersections_from_Ihphi_ref(mesh, lset_p1, eid: int, tol: float = 1e-12):
        et = mesh.element_type
        if et == "tri":
            R = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], float)
            edges = [(0, 1), (1, 2), (2, 0)]
            vals = np.empty(3, dtype=float)
            if hasattr(lset_p1, "value_on_element"):
                for i in range(3):
                    vals[i] = float(lset_p1.value_on_element(int(eid), (float(R[i, 0]), float(R[i, 1]))))
            else:
                for i in range(3):
                    x_phys = transform.x_mapping(mesh, int(eid), (float(R[i, 0]), float(R[i, 1])))
                    vals[i] = float(phi_eval(lset_p1, x_phys))
            pts: List[np.ndarray] = []
            for i, j in edges:
                a, b = vals[i], vals[j]
                if (a == 0.0 and b == 0.0):
                    pts = [R[i], R[j]]
                    break
                if a == 0.0 and b != 0.0:
                    pts.append(R[i]); continue
                if b == 0.0 and a != 0.0:
                    pts.append(R[j]); continue
                if a * b < 0.0:
                    t = a / (a - b)
                    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                    pts.append((1.0 - t) * R[i] + t * R[j])
            if len(pts) > 2:
                pts = pts[:2]
            return (pts if len(pts) == 2 else []), (0.0, 0.0, 0.0, 0.0)
        # QUAD
        a00, a10, a01, a11 = CutIntegration._q1_coeffs_ref_from_corners(mesh, int(eid), lset_p1)
        pts: List[np.ndarray] = []
        for xi in (-1.0, +1.0):
            den = (a01 + a11 * xi); num = -(a00 + a10 * xi)
            if abs(den) > tol:
                eta = num / den
                if -1.0 - 1e-12 <= eta <= 1.0 + 1e-12:
                    pts.append(np.array([xi, eta], float))
        for eta in (-1.0, +1.0):
            den = (a10 + a11 * eta); num = -(a00 + a01 * eta)
            if abs(den) > tol:
                xi = num / den
                if -1.0 - 1e-12 <= xi <= 1.0 + 1e-12:
                    if all(np.linalg.norm(np.array([xi, eta]) - p) > 1e-12 for p in pts):
                        pts.append(np.array([xi, eta], float))
        if len(pts) > 2:
            D = np.array([[np.linalg.norm(pi - pj) for pj in pts] for pi in pts])
            i, j = np.unravel_index(np.argmax(D), D.shape)
            pts = [pts[i], pts[j]]
        return (pts if len(pts) == 2 else []), (a00, a10, a01, a11)

    @staticmethod
    def iso_interface_rule_ref(mesh, eid: int, lset_p1, *, order: int = 4, tol: float = 1e-12):
        """Reference iso‑curve rule for Ihφ_ref = 0 on a given element.

        Returns (qref(N,2), w_ref(N,), coeffs, tauhat_or_None)
        where tauhat is only returned for triangles (straight chord).
        """
        et = mesh.element_type
        P, coeffs = CutIntegration._edge_intersections_from_Ihphi_ref(mesh, lset_p1, int(eid), tol=tol)
        if not P:
            tauhat = np.array([1.0, 0.0], float) if et == 'tri' else None
            return (np.empty((0, 2), float), np.empty((0,), float), coeffs, tauhat)
        if et == 'tri':
            P0, P1 = np.asarray(P[0], float), np.asarray(P[1], float)
            d = P1 - P0
            Lr = float(np.linalg.norm(d))
            if Lr <= 1e-30:
                return (np.empty((0, 2), float), np.empty((0,), float), coeffs, np.array([1.0, 0.0]))
            tauhat = d / Lr
            xi, w = gauss_legendre(int(order))
            lam = 0.5 * (xi + 1.0)
            w01 = 0.5 * w * Lr
            qref = (1.0 - lam)[:, None] * P0[None, :] + lam[:, None] * P1[None, :]
            return qref, w01, coeffs, tauhat
        # QUAD
        a00, a10, a01, a11 = coeffs
        # Integrate only over the parameter interval covered by the interface segment.
        # The iso-curve in reference is generally a hyperbola; integrating over the full
        # [-1,1] and skipping out-of-range points introduces a systematic bias.
        P0 = np.asarray(P[0], float)
        P1 = np.asarray(P[1], float)
        dxi = abs(float(P1[0] - P0[0]))
        deta = abs(float(P1[1] - P0[1]))

        # Prefer parameterization along the dominant varying reference coordinate,
        # but avoid directions where the implicit-function denominator crosses 0.
        prefer_xi = dxi >= deta
        xi_min, xi_max = (float(min(P0[0], P1[0])), float(max(P0[0], P1[0])))
        eta_min, eta_max = (float(min(P0[1], P1[1])), float(max(P0[1], P1[1])))
        safe_xi = True
        safe_eta = True
        if abs(float(a11)) > 1e-14:
            root_xi = -float(a01) / float(a11)   # where ∂φ/∂η = 0
            root_eta = -float(a10) / float(a11)  # where ∂φ/∂ξ = 0
            safe_xi = not (xi_min - 1e-12 <= root_xi <= xi_max + 1e-12)
            safe_eta = not (eta_min - 1e-12 <= root_eta <= eta_max + 1e-12)

        if prefer_xi and safe_xi and dxi > 1e-14:
            slice_along = "xi"
        elif safe_eta and deta > 1e-14:
            slice_along = "eta"
        elif safe_xi and dxi > 1e-14:
            slice_along = "xi"
        else:
            # Degenerate / difficult case: fall back to the legacy global-range rule.
            slice_along = 'xi' if abs(a01) >= abs(a10) else 'eta'

        xi_hat, w_hat = gauss_legendre(int(order))
        qpts: List[List[float]] = []
        wref: List[float] = []
        if slice_along == 'xi':
            a = xi_min if dxi > 1e-14 else -1.0
            b = xi_max if dxi > 1e-14 else +1.0
            xm = 0.5 * (a + b)
            xr = 0.5 * (b - a)
            xi_nodes = xm + xr * xi_hat
            xi_w = xr * w_hat
            for xi, w in zip(xi_nodes, xi_w):
                den = (a01 + a11 * xi); num = -(a00 + a10 * xi)
                if abs(den) < 1e-14:
                    continue
                eta = num / den
                if not (-1.0 - 1e-12 <= eta <= 1.0 + 1e-12):
                    continue
                gx = a10 + a11 * eta; gy = a01 + a11 * xi
                iface_scale = float(np.hypot(gx, gy)) / abs(den)
                qpts.append([float(xi), float(eta)])
                wref.append(float(w) * iface_scale)
        else:
            a = eta_min if deta > 1e-14 else -1.0
            b = eta_max if deta > 1e-14 else +1.0
            xm = 0.5 * (a + b)
            xr = 0.5 * (b - a)
            eta_nodes = xm + xr * xi_hat
            eta_w = xr * w_hat
            for eta, w in zip(eta_nodes, eta_w):
                den = (a10 + a11 * eta); num = -(a00 + a01 * eta)
                if abs(den) < 1e-14:
                    continue
                xi = num / den
                if not (-1.0 - 1e-12 <= xi <= 1.0 + 1e-12):
                    continue
                gx = a10 + a11 * eta; gy = a01 + a11 * xi
                iface_scale = float(np.hypot(gx, gy)) / abs(den)
                qpts.append([float(xi), float(eta)])
                wref.append(float(w) * iface_scale)
        if len(qpts) == 0:
            return (np.empty((0, 2), float), np.empty((0,), float), coeffs, None)
        return np.asarray(qpts, float), np.asarray(wref, float), coeffs, None
    @staticmethod
    def _edge_phi_on_eta_ref(mesh, eid, lset_p1, x_fixed, eta):
        # Robust fallback: evaluate φ at the physical image of (ξ,η)
        x_phys = transform.x_mapping(mesh, int(eid), (float(x_fixed), float(eta)))
        return float(phi_eval(lset_p1, x_phys))
