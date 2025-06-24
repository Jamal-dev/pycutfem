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
from typing import Mapping, Sequence, Dict, Tuple

import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.fem.reference import get_reference

# -----------------------------------------------------------------------------
#  MixedElement
# -----------------------------------------------------------------------------

class MixedElement:
    """Mixed finite element on a single mesh with *per‑field* polynomial orders.

    Parameters
    ----------
    mesh
        The common :class:`pycutfem.core.mesh.Mesh` instance.
    field_names
        Ordered iterable of field identifiers (e.g. ("ux", "uy", "p")).
    field_orders
        Optional mapping *field_name → polynomial order*.  Fields not listed
        inherit ``mesh.poly_order``.  Example::

            MixedElement(mesh, ("ux","uy","p"), field_orders={"p": 1})
    """

    # ..................................................................
    def __init__(
        self,
        mesh: Mesh,
        field_names: Sequence[str],
        *,
        field_orders: Mapping[str, int] | None = None,
    ) -> None:
        if not isinstance(mesh, Mesh):
            raise TypeError("'mesh' must be a pycutfem Mesh instance.")
        if not field_names:
            raise ValueError("'field_names' cannot be empty.")

        self.mesh: Mesh = mesh
        self.field_names: Tuple[str, ...] = tuple(field_names)
        self._field_orders: Dict[str, int] = {
            name: (field_orders[name] if field_orders and name in field_orders else mesh.poly_order)
            for name in self.field_names
        }

        # Build per‑field reference elements and basis counts -----------------
        self._ref: Dict[str, object] = {}
        self._n_basis: Dict[str, int] = {}
        for name in self.field_names:
            ref = get_reference(mesh.element_type, self._field_orders[name])
            self._ref[name] = ref
            # scalar basis fn count for *that* order
            self._n_basis[name] = int(ref.shape(0.0, 0.0).size)

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

    # ..................................................................
    #  Basis, gradient, Hessian
    # ..................................................................
    def _cache_key(self, xi: float, eta: float) -> Tuple[float, float]:
        """Round coordinates so different IEEE‑754 spellings hit the same key."""
        return (round(xi, 14), round(eta, 14))

    @lru_cache(maxsize=512)
    def basis(self, xi: float, eta: float) -> np.ndarray:  # noqa: D401
        """Return **zero‑padded** basis vector ϕ(xi,η) (shape ≡ (n_dofs_local,))."""
        # Use a single contiguous array → cache‑friendly in assemblers
        out = np.zeros(self.n_dofs_local, dtype=float)
        for name in self.field_names:
            s = self.component_dof_slices[name]
            out[s] = self._ref[name].shape(xi, eta).ravel()
        return out

    @lru_cache(maxsize=512)
    def grad(self, xi: float, eta: float) -> np.ndarray:
        """Return stacked gradient ∇ϕ (shape (n_dofs_local, 2))."""
        out = np.zeros((self.n_dofs_local, 2), dtype=float)
        for name in self.field_names:
            s = self.component_dof_slices[name]
            out[s, :] = self._ref[name].grad(xi, eta)
        return out

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

    # ..................................................................
    def __repr__(self) -> str:  # pragma: no cover – debug aide
        orders = ", ".join(f"{k}:p{self._field_orders[k]}" for k in self.field_names)
        return (
            f"<MixedElement [{orders}], elem='{self.mesh.element_type}', geo‑p={self.mesh.poly_order}, "
            f"ndofs={self.n_dofs_local}>"
        )
