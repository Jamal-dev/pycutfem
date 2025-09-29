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
    def _edge_phi_on_eta(mesh, eid: int, lset_p1, x_fixed: float, eta: float) -> float:
        return float(lset_p1.value_on_element(int(eid), (float(x_fixed), float(eta))))

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
    def _horizontal_cut_segment(mesh, eid: int, lset_p1, eta: float, side: str, order: int, tol: float):
        fl = float(lset_p1.value_on_element(int(eid), (-1.0, float(eta))))
        fr = float(lset_p1.value_on_element(int(eid), (+1.0, float(eta))))
        zl, zr = (abs(fl) <= tol), (abs(fr) <= tol)
        hascut = (fl * fr < 0.0) or (zl and fr < -tol) or (zr and fl < -tol)

        want_pos = (side == '+')
        keep_left = SIDE.is_pos(fl, tol) if want_pos else SIDE.is_neg(fl, tol)
        keep_right = SIDE.is_pos(fr, tol) if want_pos else SIDE.is_neg(fr, tol)

        if not hascut:
            if keep_left and keep_right:
                return CutIntegration._segment_rule_1d(-1.0, +1.0, order)
            return np.empty((0,), float), np.empty((0,), float)

        def fxi(xi):
            return float(lset_p1.value_on_element(int(eid), (float(xi), float(eta))))

        xi_cut = CutIntegration._brent01(fxi, -1.0, +1.0, fl, fr, tol=tol)

        if want_pos:
            if SIDE.is_pos(fl, tol) and not SIDE.is_pos(fr, tol):
                return CutIntegration._segment_rule_1d(-1.0, xi_cut, order)
            else:
                return CutIntegration._segment_rule_1d(xi_cut, +1.0, order)
        else:
            if SIDE.is_neg(fl, tol) and not SIDE.is_neg(fr, tol):
                return CutIntegration._segment_rule_1d(-1.0, xi_cut, order)
            else:
                return CutIntegration._segment_rule_1d(xi_cut, +1.0, order)

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

    # ---------------------- Iso‑curve on Ihφ_ref = 0 -----------------------
    @staticmethod
    def _q1_coeffs_ref_from_corners(mesh, eid: int, lset_p1) -> Tuple[float, float, float, float]:
        R = np.array([[-1.0, -1.0], [+1.0, -1.0], [+1.0, +1.0], [-1.0, +1.0]], dtype=float)
        v = np.array([float(lset_p1.value_on_element(int(eid), (float(R[i, 0]), float(R[i, 1])))) for i in range(4)], dtype=float)
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
            vals = np.array([float(lset_p1.value_on_element(int(eid), (float(R[i, 0]), float(R[i, 1])))) for i in range(3)], float)
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
        slice_along = 'xi' if abs(a01) >= abs(a10) else 'eta'
        xi_nodes, xi_w = gauss_legendre(int(order))
        qpts: List[List[float]] = []
        wref: List[float] = []
        if slice_along == 'xi':
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
                # GL weights are on [-1,1] already; no extra factor 2
                wref.append(float(w) * iface_scale)
        else:
            for eta, w in zip(xi_nodes, xi_w):
                den = (a10 + a11 * eta); num = -(a00 + a01 * eta)
                if abs(den) < 1e-14:
                    continue
                xi = num / den
                if not (-1.0 - 1e-12 <= xi <= 1.0 + 1e-12):
                    continue
                gx = a10 + a11 * eta; gy = a01 + a11 * xi
                iface_scale = float(np.hypot(gx, gy)) / abs(den)
                qpts.append([float(xi), float(eta)])
                # GL weights are on [-1,1] already; no extra factor 2
                wref.append(float(w) * iface_scale)
        if len(qpts) == 0:
            return (np.empty((0, 2), float), np.empty((0,), float), coeffs, None)
        return np.asarray(qpts, float), np.asarray(wref, float), coeffs, None

