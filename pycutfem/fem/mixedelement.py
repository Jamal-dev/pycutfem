from __future__ import annotations
"""
new_fem_classes.py  – rev‑2 (heterogeneous orders)
=================================================
Mixed finite‑element infrastructure with **one authoritative class**
(:class:`MixedElement`) that encapsulates *all* element‑local information – DOF
count, ordering, basis values, gradients, Hessians.

Why rev‑2?
~~~~~~~~~~
* Supports *heterogeneous polynomial orders per field* (e.g. Q2‑Q2‑Q1 for
  Stokes ⇒ 22 local DOFs).
* Keeps the **zero‑padded** local vectors you requested so
  ``np.outer(φ,ψ)`` produces diagonal‑block mass matrices in one line.
* No hard‑coded assumptions: everything is derived from the reference element
  returned by :pyfunc:`pycutfem.fem.reference.get_reference`.
"""

from functools import lru_cache
from typing import Mapping, Sequence, Dict, Tuple, List

import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.fem.reference import get_reference

# -----------------------------------------------------------------------------
#  MixedElement
# -----------------------------------------------------------------------------

# How many nodes live on ONE geometric edge for a given Lagrange order?
def _edge_nodes(p: int, element_type: str) -> int:
    # quads and triangles both have p+1 equally‑spaced nodes on an edge
    return p + 1

class _NumberRef:
    def shape(self, xi, eta):
        # one basis value, identically 1
        return np.array([1.0], dtype=float)
    def grad(self, xi, eta):
        # gradient is zero in physical and reference coords
        return np.array([[0.0, 0.0]], dtype=float)
    
    def derivative(self, xi, eta, ox, oy):
        if ox == 0 and oy == 0:
            return np.array([1.0], dtype=float)
        else:
            return np.array([0.0], dtype=float)


class MixedElement:
    """Mixed finite element on a single mesh with *per‑field* polynomial orders.

    Parameters
    ----------
    mesh
        The common :class:`pycutfem.core.mesh.Mesh` instance.
    field_specs
        A mapping of field names to their polynomial orders, e.g. `{'ux': 2, 'uy': 2, 'p': 1}`.
        
    """

    # ..................................................................
    def __init__(
        self,
        mesh: Mesh,
        field_specs: Mapping[str, int] | None = None,
    ) -> None:
        if not isinstance(field_specs, dict):
            field_specs = dict(field_specs)
        if not isinstance(mesh, Mesh):
            raise TypeError("'mesh' must be a pycutfem Mesh instance.")
        self.q_orders: Dict[str, int] = {name:q for name,q in field_specs.items() if q != ':number:'}
        self.field_names: Tuple[str, ...] = tuple(field_specs.keys())
        if not self.field_names:
            raise ValueError("'field_names' cannot be empty.")

        self.mesh: Mesh = mesh
        # detect number fields
        self._number_fields = {name for name, p in field_specs.items()
                       if p == ':number:'}   # sentinel
        # keep _field_orders numeric so nothing downstream sees a string
        self._field_orders = {name: (0 if name in self._number_fields else int(p))
                            for name, p in field_specs.items()}


        # Build per‑field reference elements and basis counts -----------------
        self._ref = {}
        self._n_basis = {}
        for name in self.field_names:
            if name in self._number_fields:
                self._ref[name] = _NumberRef()
                self._n_basis[name] = 1
            else:
                ref = get_reference(mesh.element_type, self._field_orders[name])
                self._ref[name] = ref
                # or the formula you already use for quads/tris:
                self._n_basis[name] = ((self._field_orders[name] + 1)**2
                                    if mesh.element_type == 'quad'
                                    else (self._field_orders[name] + 1)*(self._field_orders[name] + 2)//2)
        # Build slices into the global local‑DOF vector -----------------------
        self.component_dof_slices: Dict[str, slice] = {}
        start = 0
        for name in self.field_names:
            n_b = self._n_basis[name]
            self.component_dof_slices[name] = slice(start, start + n_b)
            start += n_b
        self.n_dofs_local: int = start

        # Canonical reference‑node ordering for geometry ----------------------
        self.element_node_map: np.ndarray = self._reference_node_ordering()

         # global (element‑local) index → owning field
        self._field_of: List[str] = []
        for f in self.field_names:                       #   |ux|  |uy|  |p|
            self._field_of += [f]*self._n_basis[f]       # 0..8  9..17 18..21

        self.n_dofs_per_elem = len(self._field_of)
        # print(f"Dofs per element: {self.n_dofs_per_elem} (fields: {self.field_names})")
        # Debug
        # print(f"MixedElement created with {self.n_dofs_per_elem} local DOFs "
        #       f"({self.field_names}) for {mesh.element_type} elements of order "
        #       f"{self.mesh.poly_order} (p) and field orders {self._field_orders}.")
        # ------------------------------------------------------------------
        #  Per‑field edge node counts  (shared DOFs on one edge)
        self._n_edge: Dict[str, int] = {
            f: _edge_nodes(p, mesh.element_type) for f, p in self._field_orders.items()
        }
        #  Size of the union of two neighbouring elements on an interior edge
        #  =  2·(local DOFs)  –  (shared edge DOFs)   summed over all fields
        self.n_union_cg = sum(2*self._n_basis[f] - self._n_edge[f] for f in self.field_names)
        self.n_union_dg = sum(2*self._n_basis[f]                      for f in self.field_names)

    
    def _ensure_many_cache(self):
        if not hasattr(self, "_tab_many_cache"):
            # (field, kind, nqp, xi.tobytes(), eta.tobytes()) -> np.ndarray
            self._tab_many_cache = {}

    def union_dofs(self, method: str = "cg") -> int:
        """Return the ghost‑edge union size for 'cg' or 'dg' numbering."""
        return self.n_union_cg if method == "cg" else self.n_union_dg
    # ..................................................................
    #  Basis, gradient, Hessian
    # ..................................................................
    def _eval_scalar_basis(self,field:str, xi: float, eta: float) -> np.ndarray:
        return self._ref[field].shape(xi, eta)
    def _eval_scalar_grad(self, field: str, xi: float, eta: float) -> np.ndarray:
        """Return the gradient of the scalar basis functions for `field`."""
        return self._ref[field].grad(xi, eta)
    def _eval_scalar_basis_many(self, field: str, xi_arr, eta_arr):
        """
        Stack shape functions at many reference points.
        Returns (n_qp, n_loc_field).
        """
        self._ensure_many_cache()
        xi  = np.asarray(xi_arr, dtype=float).ravel()
        eta = np.asarray(eta_arr, dtype=float).ravel()
        if xi.shape != eta.shape:
            raise ValueError("xi/eta must have same shape")
        nqp = xi.size
        key = (field, "basis", nqp, xi.tobytes(), eta.tobytes())
        hit = self._tab_many_cache.get(key)
        if hit is not None:
            return hit

        # one local to get n_loc
        n_loc = len(self._eval_scalar_basis(field, float(xi[0]), float(eta[0])))
        out = np.empty((nqp, n_loc), dtype=float)
        # small nqp (e.g. 1..16) → tight Python loop is fine & cacheable
        for i in range(nqp):
            out[i, :] = self._eval_scalar_basis(field, float(xi[i]), float(eta[i]))
        self._tab_many_cache[key] = out
        return out

    # --- add: vectorized reference-grad over many (xi,eta) ---
    def _eval_scalar_grad_many(self, field: str, xi_arr, eta_arr):
        """
        Stack reference gradients at many points.
        Returns (n_qp, n_loc_field, 2) with columns (d/dξ, d/dη).
        """
        self._ensure_many_cache()
        xi  = np.asarray(xi_arr, dtype=float).ravel()
        eta = np.asarray(eta_arr, dtype=float).ravel()
        if xi.shape != eta.shape:
            raise ValueError("xi/eta must have same shape")
        nqp = xi.size
        key = (field, "grad", nqp, xi.tobytes(), eta.tobytes())
        hit = self._tab_many_cache.get(key)
        if hit is not None:
            return hit

        n_loc = len(self._eval_scalar_basis(field, float(xi[0]), float(eta[0])))
        out = np.empty((nqp, n_loc, 2), dtype=float)
        for i in range(nqp):
            out[i, :, :] = self._eval_scalar_grad(field, float(xi[i]), float(eta[i]))
        self._tab_many_cache[key] = out
        return out
    def _eval_scalar_deriv(self, field: str, xi: float, eta: float,order_x:int, order_y:int ):
        """Return the derivative of the scalar basis functions for `field`.
           order_x =1, order_y=1 would be ∂^2φ/∂eta∂xi
           order_x = 2, order_y=0 would be ∂^2φ/∂xi^2
        """
        return self._ref[field].derivative(xi, eta, order_x, order_y)
    def _cache_key(self, xi: float, eta: float) -> Tuple[float, float]:
        """Round coordinates so different IEEE‑754 spellings hit the same key."""
        return (round(xi, 14), round(eta, 14))

    def basis(self, field: str, xi: float, eta: float) -> np.ndarray:
        """22‑vector with non‑zeros only at the DOFs of ``field``."""
        phi = np.zeros(self.n_dofs_per_elem)
        local = self._eval_scalar_basis(field, xi, eta)
        idx = self.component_dof_slices[field]
        phi[idx] = local
        return phi

    def grad_basis(self, field: str, xi: float, eta: float) -> np.ndarray:
        """Shape (22,2); rows belonging to other fields are zero."""
        G = np.zeros((self.n_dofs_per_elem, 2))
        locG = self._eval_scalar_grad(field, xi, eta)  # (n,2)
        idx = self.component_dof_slices[field]
        G[idx, :] = locG
        return G
    def deriv_ref(self, field: str, xi, eta, order_x: int, order_y: int) -> np.ndarray:
        """
        Reference derivative for any order (union-sized vector).
        Fast path for quads P1/P2 and tris P1 when order_x+order_y <= 2.
        """
        import numpy as np
        from pycutfem.integration.pre_tabulates import (
            _eval_deriv_q1, _eval_deriv_q2, _eval_deriv_p1
        )
        phi = np.zeros(self.n_dofs_per_elem)
        sl = self.component_dof_slices[field]
        p  = self._field_orders[field]
        # ---- SPECIAL CASE: 0th order == basis value (not "derivative") ----
        if int(order_x) == 0 and int(order_y) == 0:
            phi[sl] = self._eval_scalar_basis(field, float(xi), float(eta))
            return phi
        if (order_x + order_y) <= 2:
            if self.mesh.element_type == "quad" and p == 1:
                phi[sl] = _eval_deriv_q1(float(xi), float(eta), int(order_x), int(order_y))
                return phi
            if self.mesh.element_type == "quad" and p == 2:
                phi[sl] = _eval_deriv_q2(float(xi), float(eta), int(order_x), int(order_y))
                return phi
            if self.mesh.element_type == "tri" and p == 1:
                phi[sl] = _eval_deriv_p1(float(xi), float(eta), int(order_x), int(order_y))
                return phi
        # fallback: per-field reference object
        loc = self._eval_scalar_deriv(field, float(xi), float(eta), int(order_x), int(order_y))
        phi[sl] = loc
        return phi



    @lru_cache(maxsize=256)
    def hess(self, xi: float, eta: float) -> np.ndarray:
        """Return Hessians for every local DOF (shape (n_dofs_local, 2, 2))."""
        out = np.zeros((self.n_dofs_local, 2, 2), dtype=float)
        for name in self.field_names:
            s = self.component_dof_slices[name]
            # reference provides 2×2 Hessian per basis fn (triangular index or full?)
            h = self._ref[name].hess(xi, eta)  # expects (n_scalar_basis, 2, 2)
            out[s, :, :] = h
        return out

    # ..................................................................
    #  Helpers
    # ..................................................................
    def slice(self, field: str) -> slice:
        return self.component_dof_slices[field]

    # ..................................................................
    def _reference_node_ordering(self) -> np.ndarray:
        """Return canonical node ordering for element geometry.

        For quads this is row‑by‑row over (eta, xi) to match the implementation
        in *quad_qn.py*.  For triangles a simple lexicographic loop over bary
        indices is used.  The ordering is *independent* of per‑field p‑orders –
        geometric nodes are controlled by the *mesh* order.
        """
        p = self.mesh.poly_order
        if self.mesh.element_type == "quad":
            coords = [(i, j) for j in range(p + 1) for i in range(p + 1)]
        elif self.mesh.element_type == "tri":
            coords = []
            for j in range(p + 1):
                for i in range(p + 1 - j):
                    coords.append((i, j))
        else:
            raise KeyError(f"Unsupported element_type '{self.mesh.element_type}'")
        return np.arange(len(coords), dtype=int)

    def signature(self) -> tuple:
        """
        A hashable identifier that changes as soon as *anything* relevant for the
        element-local layout changes (geometry type, mesh order, per-field order,
        or the resulting local DOF count).  Safe to use as part of a JIT cache key.
        """
        field_info = tuple(sorted(self._field_orders.items()))    # (('p',1),('ux',2),('uy',2))
        return (self.mesh.element_type,            # 'quad' / 'tri'
                self.mesh.poly_order,              # geometry order
                field_info,                        # heterogeneous field orders
                self.n_dofs_local)                 # 22, 18, 8, ...
    
    # ..................................................................
    def __repr__(self) -> str:  # pragma: no cover – debug aide
        orders = ", ".join(f"{k}:p{self._field_orders[k]}" for k in self.field_names)
        return (
            f"<MixedElement [{orders}], elem='{self.mesh.element_type}', geo‑p={self.mesh.poly_order}, "
            f"ndofs={self.n_dofs_local}>"
        )
