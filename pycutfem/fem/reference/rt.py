from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

import numpy as np
from numpy.polynomial.legendre import legval

from numpy.polynomial.legendre import leggauss


def gauss_legendre(order: int):
    if order < 1:
        raise ValueError(order)
    return leggauss(int(order))  # (points, weights)


@lru_cache(maxsize=None)
def _quad_rule(order: int):
    xi, wi = gauss_legendre(int(order))
    pts = np.array([[x, y] for x in xi for y in xi], dtype=float)
    wts = np.array([wx * wy for wx in wi for wy in wi], dtype=float)
    return pts, wts


@lru_cache(maxsize=None)
def _tri_rule(order: int):
    """
    Square → reference-triangle mapping quadrature (degree ~ 2*order-1).
    Reference triangle: (0,0)-(1,0)-(0,1).
    """
    xi, wi = gauss_legendre(int(order))
    u = 0.5 * (xi + 1.0)  # [0,1]
    w_u = 0.5 * wi
    pts = []
    wts = []
    for i, ui in enumerate(u):
        for j, vj in enumerate(u):
            r = ui
            s = vj * (1.0 - ui)
            weight = w_u[i] * w_u[j] * (1.0 - ui)
            pts.append([r, s])
            wts.append(weight)
    return np.asarray(pts, dtype=float), np.asarray(wts, dtype=float)


def _volume_rule(element_type: str, order: int):
    if element_type == "tri":
        return _tri_rule(int(order))
    if element_type == "quad":
        return _quad_rule(int(order))
    raise KeyError(element_type)


def _legendre_all(k: int, s: np.ndarray) -> np.ndarray:
    """
    Values of Legendre polynomials P_0..P_k at points `s` (shape (nq,)).

    Returns
    -------
    np.ndarray, shape (k+1, nq)
    """
    s = np.asarray(s, dtype=float).ravel()
    out = np.empty((k + 1, s.size), dtype=float)
    for m in range(k + 1):
        coeff = np.zeros(m + 1, dtype=float)
        coeff[-1] = 1.0
        out[m, :] = legval(s, coeff)
    return out


class Poly2D:
    """
    Very small 2D polynomial helper in monomial basis.

    Represents p(x,y) = sum_{(i,j)} c_{ij} x^i y^j.
    """

    def __init__(self, terms: Dict[Tuple[int, int], float] | None = None) -> None:
        self.terms: Dict[Tuple[int, int], float] = dict(terms) if terms else {}

    @staticmethod
    def monomial(i: int, j: int, coeff: float = 1.0) -> "Poly2D":
        if coeff == 0.0:
            return Poly2D()
        return Poly2D({(int(i), int(j)): float(coeff)})

    def mul_x(self) -> "Poly2D":
        return self.mul_monomial(1, 0)

    def mul_y(self) -> "Poly2D":
        return self.mul_monomial(0, 1)

    def mul_monomial(self, di: int, dj: int, coeff: float = 1.0) -> "Poly2D":
        di = int(di)
        dj = int(dj)
        coeff = float(coeff)
        if coeff == 0.0 or not self.terms:
            return Poly2D()
        out: Dict[Tuple[int, int], float] = {}
        for (i, j), c in self.terms.items():
            key = (i + di, j + dj)
            out[key] = out.get(key, 0.0) + coeff * c
        return Poly2D(out)

    def deriv(self, dxi: int, deta: int) -> "Poly2D":
        dxi = int(dxi)
        deta = int(deta)
        if dxi < 0 or deta < 0:
            raise ValueError("Derivative orders must be non-negative.")
        if (dxi == 0 and deta == 0) or not self.terms:
            return Poly2D(self.terms)
        out = Poly2D(self.terms)
        for _ in range(dxi):
            out = out._deriv_x()
        for _ in range(deta):
            out = out._deriv_y()
        return out

    def _deriv_x(self) -> "Poly2D":
        out: Dict[Tuple[int, int], float] = {}
        for (i, j), c in self.terms.items():
            if i <= 0:
                continue
            key = (i - 1, j)
            out[key] = out.get(key, 0.0) + c * i
        return Poly2D(out)

    def _deriv_y(self) -> "Poly2D":
        out: Dict[Tuple[int, int], float] = {}
        for (i, j), c in self.terms.items():
            if j <= 0:
                continue
            key = (i, j - 1)
            out[key] = out.get(key, 0.0) + c * j
        return Poly2D(out)

    def eval(self, x: float, y: float) -> float:
        if not self.terms:
            return 0.0
        x = float(x)
        y = float(y)
        val = 0.0
        for (i, j), c in self.terms.items():
            val += float(c) * (x**i) * (y**j)
        return float(val)

    def eval_many(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        out = np.zeros_like(x, dtype=float)
        for (i, j), c in self.terms.items():
            out += float(c) * (x**i) * (y**j)
        return out


def _monomials_tri(deg: int) -> List[Tuple[int, int]]:
    """
    Exponents (i,j) for monomials x^i y^j with i+j <= deg.
    Ordering: eta-degree outer, then xi-degree inner (tri_pn-like).
    """
    if deg < 0:
        return []
    out: List[Tuple[int, int]] = []
    for j in range(deg + 1):
        for i in range(deg + 1 - j):
            out.append((i, j))
    return out


def _monomials_quad(ix: int, iy: int) -> List[Tuple[int, int]]:
    """Exponents (i,j) for tensor-product monomials 0<=i<=ix, 0<=j<=iy."""
    if ix < 0 or iy < 0:
        return []
    return [(i, j) for j in range(iy + 1) for i in range(ix + 1)]


def _edge_param(element_type: str, edge: int, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reference edge parameterization for RT DOFs.

    Parameters
    ----------
    element_type : {'tri','quad'}
    edge : local edge id
    s : np.ndarray
        1D points on [-1,1]

    Returns
    -------
    (xi, eta, w_scale) where:
    - (xi,eta) are reference coordinates on the edge
    - w_scale is |dx/ds| (reference arc-length Jacobian)
    """
    s = np.asarray(s, dtype=float).ravel()
    if element_type == "quad":
        if edge == 0:
            return s, -np.ones_like(s), np.ones_like(s)
        if edge == 1:
            return np.ones_like(s), s, np.ones_like(s)
        if edge == 2:
            return -s, np.ones_like(s), np.ones_like(s)
        if edge == 3:
            return -np.ones_like(s), -s, np.ones_like(s)
        raise IndexError(edge)
    if element_type == "tri":
        t = 0.5 * (s + 1.0)  # [0,1]
        if edge == 0:  # (0,0)->(1,0)
            xi = t
            eta = np.zeros_like(t)
            return xi, eta, 0.5 * np.ones_like(s)
        if edge == 1:  # (1,0)->(0,1)
            xi = 1.0 - t
            eta = t
            return xi, eta, 0.5 * np.sqrt(2.0) * np.ones_like(s)
        if edge == 2:  # (0,1)->(0,0)
            xi = np.zeros_like(t)
            eta = 1.0 - t
            return xi, eta, 0.5 * np.ones_like(s)
        raise IndexError(edge)
    raise KeyError(element_type)


def _edge_normal(element_type: str, edge: int) -> np.ndarray:
    """Unit outward normal on the reference element edge (local id)."""
    if element_type == "quad":
        normals = np.array(
            [
                [0.0, -1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=float,
        )
        return normals[int(edge)]
    if element_type == "tri":
        normals = np.array(
            [
                [0.0, -1.0],
                [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
                [-1.0, 0.0],
            ],
            dtype=float,
        )
        return normals[int(edge)]
    raise KeyError(element_type)


@dataclass(frozen=True, slots=True)
class DofDesc:
    kind: str  # 'edge' or 'cell'
    entity: int  # edge id or component id
    mode: int  # edge: Legendre mode; cell: monomial mode


class RaviartThomasRef:
    """
    Reference Raviart–Thomas RT_k element on triangles and quads (2D).

    - DOFs are moments of the normal component on each edge against Legendre P_m(s).
    - Interior DOFs (k>=1) are component-wise moments against lower-order polynomials.
    """

    def __init__(self, element_type: str, k: int) -> None:
        element_type = str(element_type)
        if element_type not in {"tri", "quad"}:
            raise KeyError(f"RT only implemented for 'tri'/'quad', got '{element_type}'")
        k = int(k)
        if k < 0:
            raise ValueError("RT order k must be >= 0")
        self.element_type = element_type
        self.k = k
        self.value_dim = 2

        self.n_edges = 3 if element_type == "tri" else 4
        self.n_edge_dofs = k + 1

        self._poly_basis: List[Tuple[Poly2D, Poly2D]] = self._build_poly_basis()
        self._div_poly_basis: List[Poly2D] = [
            px.deriv(1, 0)  # d/dx
            for px, _ in self._poly_basis
        ]
        for i, (_, py) in enumerate(self._poly_basis):
            self._div_poly_basis[i] = Poly2D(
                {
                    **self._div_poly_basis[i].terms,
                }
            )
            dy = py.deriv(0, 1)
            for key, c in dy.terms.items():
                self._div_poly_basis[i].terms[key] = self._div_poly_basis[i].terms.get(key, 0.0) + c

        self.dofs: List[DofDesc] = self._build_dofs()
        self.n_dofs = len(self.dofs)
        if self.n_dofs != len(self._poly_basis):
            raise RuntimeError(
                f"RT basis size mismatch: dofs={self.n_dofs} vs poly_basis={len(self._poly_basis)}"
            )

        A = self._build_dof_matrix()
        self.A = A
        try:
            self.C = np.linalg.inv(A)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"RT DOF matrix singular for {element_type} k={k}") from e

    @property
    def n_cell_dofs(self) -> int:
        return int(self.n_dofs - self.n_edges * self.n_edge_dofs)

    def _build_poly_basis(self) -> List[Tuple[Poly2D, Poly2D]]:
        k = self.k
        if self.element_type == "tri":
            mono_pk = _monomials_tri(k)
            basis: List[Tuple[Poly2D, Poly2D]] = []
            for (i, j) in mono_pk:
                p = Poly2D.monomial(i, j, 1.0)
                basis.append((p, Poly2D()))
            for (i, j) in mono_pk:
                p = Poly2D.monomial(i, j, 1.0)
                basis.append((Poly2D(), p))

            # homogeneous degree-k polynomials times (x,y)
            for i in range(k + 1):
                j = k - i
                q = Poly2D.monomial(i, j, 1.0)  # homogeneous
                basis.append((q.mul_x(), q.mul_y()))

            return basis

        # quad: ux ∈ Q_{k+1,k}, uy ∈ Q_{k,k+1}
        ux_mono = _monomials_quad(k + 1, k)
        uy_mono = _monomials_quad(k, k + 1)
        basis_q: List[Tuple[Poly2D, Poly2D]] = []
        for (i, j) in ux_mono:
            basis_q.append((Poly2D.monomial(i, j, 1.0), Poly2D()))
        for (i, j) in uy_mono:
            basis_q.append((Poly2D(), Poly2D.monomial(i, j, 1.0)))
        return basis_q

    def _build_dofs(self) -> List[DofDesc]:
        k = self.k
        dofs: List[DofDesc] = []
        for e in range(self.n_edges):
            for m in range(k + 1):
                dofs.append(DofDesc(kind="edge", entity=int(e), mode=int(m)))

        if k <= 0:
            return dofs

        if self.element_type == "tri":
            mono = _monomials_tri(k - 1)
            # component-wise moments against P_{k-1}
            for comp in (0, 1):
                for r, _ in enumerate(mono):
                    dofs.append(DofDesc(kind="cell", entity=int(comp), mode=int(r)))
            return dofs

        # quad: ux moments against Q_{k-1,k}, uy moments against Q_{k,k-1}
        mono_x = _monomials_quad(k - 1, k)
        mono_y = _monomials_quad(k, k - 1)
        for r in range(len(mono_x)):
            dofs.append(DofDesc(kind="cell", entity=0, mode=int(r)))
        for r in range(len(mono_y)):
            dofs.append(DofDesc(kind="cell", entity=1, mode=int(r)))
        return dofs

    def _eval_poly_basis_many(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=float).ravel()
        eta = np.asarray(eta, dtype=float).ravel()
        out = np.empty((len(self._poly_basis), xi.size, 2), dtype=float)
        for j, (px, py) in enumerate(self._poly_basis):
            out[j, :, 0] = px.eval_many(xi, eta)
            out[j, :, 1] = py.eval_many(xi, eta)
        return out

    def _eval_div_poly_basis_many(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=float).ravel()
        eta = np.asarray(eta, dtype=float).ravel()
        out = np.empty((len(self._div_poly_basis), xi.size), dtype=float)
        for j, p in enumerate(self._div_poly_basis):
            out[j, :] = p.eval_many(xi, eta)
        return out

    def _build_dof_matrix(self) -> np.ndarray:
        k = self.k
        n = len(self._poly_basis)
        A = np.zeros((n, n), dtype=float)

        # --- edge DOFs ---
        edge_q = max(k + 2, 2)
        s, w = gauss_legendre(edge_q)
        P = _legendre_all(k, s)  # (k+1, nq)

        for e in range(self.n_edges):
            xi, eta, w_scale = _edge_param(self.element_type, e, s)
            vals = self._eval_poly_basis_many(xi, eta)  # (n, nq, 2)
            nvec = _edge_normal(self.element_type, e)
            flux = vals[:, :, 0] * nvec[0] + vals[:, :, 1] * nvec[1]  # (n, nq)
            ww = (w * w_scale).reshape(1, -1)  # (1, nq)
            for m in range(k + 1):
                row = e * (k + 1) + m
                A[row, :] = np.sum(ww * flux * P[m][None, :], axis=1)

        # --- cell DOFs ---
        if k <= 0:
            return A

        qp, qw = _volume_rule(self.element_type, order=max(k + 2, 2))
        qp = np.asarray(qp, dtype=float)
        qw = np.asarray(qw, dtype=float)
        xi = qp[:, 0]
        eta = qp[:, 1]
        vals = self._eval_poly_basis_many(xi, eta)  # (n, nq, 2)

        row0 = self.n_edges * (k + 1)

        if self.element_type == "tri":
            mono = _monomials_tri(k - 1)
            mono_vals = np.vstack([Poly2D.monomial(i, j).eval_many(xi, eta) for (i, j) in mono])  # (nmono,nq)
            # component 0 then 1
            for comp in (0, 1):
                for r in range(len(mono)):
                    row = row0 + comp * len(mono) + r
                    integrand = vals[:, :, comp] * mono_vals[r][None, :]  # (n, nq)
                    A[row, :] = np.sum(integrand * qw.reshape(1, -1), axis=1)
            return A

        # quad
        mono_x = _monomials_quad(k - 1, k)
        mono_y = _monomials_quad(k, k - 1)
        mono_x_vals = np.vstack(
            [Poly2D.monomial(i, j).eval_many(xi, eta) for (i, j) in mono_x]
        )  # (nmono_x,nq)
        mono_y_vals = np.vstack(
            [Poly2D.monomial(i, j).eval_many(xi, eta) for (i, j) in mono_y]
        )  # (nmono_y,nq)

        for r in range(len(mono_x)):
            row = row0 + r
            integrand = vals[:, :, 0] * mono_x_vals[r][None, :]
            A[row, :] = np.sum(integrand * qw.reshape(1, -1), axis=1)
        base = row0 + len(mono_x)
        for r in range(len(mono_y)):
            row = base + r
            integrand = vals[:, :, 1] * mono_y_vals[r][None, :]
            A[row, :] = np.sum(integrand * qw.reshape(1, -1), axis=1)
        return A

    @lru_cache(maxsize=512)
    def tabulate_value(self, xi: float, eta: float) -> np.ndarray:
        """
        Reference basis values of the nodal RT basis.

        Returns
        -------
        np.ndarray, shape (n_dofs, 2)
        """
        xi = float(xi)
        eta = float(eta)
        phi = np.empty((self.n_dofs, 2), dtype=float)
        for j, (px, py) in enumerate(self._poly_basis):
            phi[j, 0] = px.eval(xi, eta)
            phi[j, 1] = py.eval(xi, eta)
        # Psi = C^T Phi
        return self.C.T @ phi

    @lru_cache(maxsize=512)
    def tabulate_div(self, xi: float, eta: float) -> np.ndarray:
        """
        Reference divergence of the nodal RT basis.

        Returns
        -------
        np.ndarray, shape (n_dofs,)
        """
        xi = float(xi)
        eta = float(eta)
        div_phi = np.empty((self.n_dofs,), dtype=float)
        for j, p in enumerate(self._div_poly_basis):
            div_phi[j] = p.eval(xi, eta)
        return self.C.T @ div_phi

    @lru_cache(maxsize=512)
    def tabulate_grad(self, xi: float, eta: float) -> np.ndarray:
        """
        Reference gradients of the nodal RT basis.

        Returns
        -------
        np.ndarray, shape (n_dofs, 2, 2)
            ``out[j, i, a] = d/dhat_x_a (phi_j)_i`` where ``i`` is the vector
            component and ``a`` is the reference derivative index.
        """
        xi = float(xi)
        eta = float(eta)
        grad_phi = np.empty((self.n_dofs, 2, 2), dtype=float)
        for j, (px, py) in enumerate(self._poly_basis):
            grad_phi[j, 0, 0] = px.deriv(1, 0).eval(xi, eta)
            grad_phi[j, 0, 1] = px.deriv(0, 1).eval(xi, eta)
            grad_phi[j, 1, 0] = py.deriv(1, 0).eval(xi, eta)
            grad_phi[j, 1, 1] = py.deriv(0, 1).eval(xi, eta)

        # Apply the nodal-basis transform Psi = C^T Phi component-wise.
        out = np.empty_like(grad_phi)
        for comp in range(2):
            for ax in range(2):
                out[:, comp, ax] = self.C.T @ grad_phi[:, comp, ax]
        return out

    @lru_cache(maxsize=512)
    def tabulate_hessian(self, xi: float, eta: float) -> np.ndarray:
        """
        Reference Hessians of the nodal RT basis components.

        Returns
        -------
        np.ndarray, shape (n_dofs, 2, 2, 2)
            ``out[j, comp, ax0, ax1] = d^2 / d hat_x_ax0 d hat_x_ax1 (phi_j)_comp``.
        """
        xi = float(xi)
        eta = float(eta)
        hess_phi = np.empty((self.n_dofs, 2, 2, 2), dtype=float)
        for j, (px, py) in enumerate(self._poly_basis):
            hess_phi[j, 0, 0, 0] = px.deriv(2, 0).eval(xi, eta)
            hess_phi[j, 0, 0, 1] = px.deriv(1, 1).eval(xi, eta)
            hess_phi[j, 0, 1, 0] = hess_phi[j, 0, 0, 1]
            hess_phi[j, 0, 1, 1] = px.deriv(0, 2).eval(xi, eta)
            hess_phi[j, 1, 0, 0] = py.deriv(2, 0).eval(xi, eta)
            hess_phi[j, 1, 0, 1] = py.deriv(1, 1).eval(xi, eta)
            hess_phi[j, 1, 1, 0] = hess_phi[j, 1, 0, 1]
            hess_phi[j, 1, 1, 1] = py.deriv(0, 2).eval(xi, eta)

        out = np.empty_like(hess_phi)
        for comp in range(2):
            for ax0 in range(2):
                for ax1 in range(2):
                    out[:, comp, ax0, ax1] = self.C.T @ hess_phi[:, comp, ax0, ax1]
        return out


@lru_cache(maxsize=None)
def get_rt_reference(element_type: str, k: int) -> RaviartThomasRef:
    return RaviartThomasRef(str(element_type), int(k))
