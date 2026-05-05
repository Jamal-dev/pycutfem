from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from pycutfem.fem.mixedelement import MixedElement


class XFEMMixedElement:
    """
    MixedElement-like view with expanded local DOF counts for enriched fields.

    This object is intended for kernel code generation and static-table layout:
    - field names stay the same
    - component slices are widened for enriched fields (base+enriched contiguous)
    - total local DOF count increases accordingly

    Actual enrichment multipliers (alpha) are supplied through precomputed
    basis tables (e.g. via XFEMDofHandler.precompute_cut_volume_factors_xfem).
    """

    def __init__(self, base: MixedElement, *, enrich_kind_by_field: Mapping[str, str | None]):
        if not isinstance(base, MixedElement):
            raise TypeError("XFEMMixedElement expects a MixedElement instance.")
        self.base = base
        self.mesh = base.mesh
        self.field_names = base.field_names

        # Preserve base order and field polynomial orders.
        self._field_orders = dict(getattr(base, "_field_orders", {}))
        self.q_orders = dict(getattr(base, "q_orders", {}))

        # Expanded basis counts per field.
        self._n_basis_base: Dict[str, int] = dict(getattr(base, "_n_basis", {}))
        self._enrich_kind_by_field = {str(k): (None if v is None else str(v)) for k, v in dict(enrich_kind_by_field).items()}

        self._n_basis: Dict[str, int] = {}
        for f in self.field_names:
            n0 = int(self._n_basis_base[f])
            if self._enrich_kind_by_field.get(f) is None:
                self._n_basis[f] = n0
            else:
                self._n_basis[f] = 2 * n0

        # Build contiguous component slices in field order.
        self.component_dof_slices: Dict[str, slice] = {}
        start = 0
        for f in self.field_names:
            n = int(self._n_basis[f])
            self.component_dof_slices[f] = slice(start, start + n)
            start += n

        self.n_dofs_local = int(start)
        self.n_dofs_per_elem = int(start)
        self.spatial_dim = getattr(base, "spatial_dim", getattr(self.mesh, "spatial_dim", 2))

    # ------------------------------------------------------------------
    # Introspection / caching keys
    # ------------------------------------------------------------------
    def signature(self) -> tuple:
        base_sig = self.base.signature()
        enrich = tuple((f, self._enrich_kind_by_field.get(f)) for f in self.field_names if self._enrich_kind_by_field.get(f) is not None)
        return ("xfem", base_sig, enrich, int(self.n_dofs_local))

    def __repr__(self) -> str:  # pragma: no cover
        enr = ",".join(f"{f}:{self._enrich_kind_by_field.get(f)}" for f in self.field_names if self._enrich_kind_by_field.get(f) is not None) or "-"
        return f"<XFEMMixedElement base={self.base!r} enr=[{enr}] ndofs={self.n_dofs_local}>"

    # ------------------------------------------------------------------
    # Basis API: fallback to base (enrichment supplied by precomputed tables)
    # ------------------------------------------------------------------
    def basis(self, field: str, xi: float, eta: float) -> np.ndarray:
        """
        Union-sized basis with zeros outside the (expanded) field slice.

        Note: this fallback does *not* include enrichment multipliers. XFEM
        kernels should rely on prebuilt b_/g_ tables for cut/interface/ghost.
        """
        field = str(field)
        if field not in self.component_dof_slices:
            raise KeyError(field)
        out = np.zeros(self.n_dofs_local, dtype=float)
        sl = self.component_dof_slices[field]
        n0 = int(self._n_basis_base[field])
        # base contribution goes in the first n0 entries of the field slice
        out[sl.start : sl.start + n0] = self.base._eval_scalar_basis(field, float(xi), float(eta))
        return out

    def grad_basis(self, field: str, xi: float, eta: float) -> np.ndarray:
        out = np.zeros((self.n_dofs_local, 2), dtype=float)
        sl = self.component_dof_slices[str(field)]
        n0 = int(self._n_basis_base[str(field)])
        out[sl.start : sl.start + n0, :] = self.base._eval_scalar_grad(str(field), float(xi), float(eta))
        return out

    def deriv_ref(self, field: str, xi: float, eta: float, order_x: int, order_y: int) -> np.ndarray:
        out = np.zeros(self.n_dofs_local, dtype=float)
        sl = self.component_dof_slices[str(field)]
        n0 = int(self._n_basis_base[str(field)])
        out[sl.start : sl.start + n0] = self.base.deriv_ref(str(field), float(xi), float(eta), int(order_x), int(order_y))[
            self.base.component_dof_slices[str(field)]
        ]
        return out

    # ------------------------------------------------------------------
    # Vectorized reference tables (used by JIT kernel arg builders)
    # ------------------------------------------------------------------
    def _eval_scalar_basis_many(self, field: str, xi_arr, eta_arr) -> np.ndarray:
        """
        Match MixedElement._eval_scalar_basis_many but return a widened table for
        enriched fields (base values in the first block, zeros in the enriched block).
        """
        field = str(field)
        tab = self.base._eval_scalar_basis_many(field, xi_arr, eta_arr)  # (n_qp, n0)
        if self._enrich_kind_by_field.get(field) is None:
            return tab
        nqp, n0 = tab.shape
        out = np.zeros((nqp, 2 * n0), dtype=float)
        out[:, :n0] = tab
        return out

    def _eval_scalar_grad_many(self, field: str, xi_arr, eta_arr) -> np.ndarray:
        """
        Match MixedElement._eval_scalar_grad_many but return a widened table for
        enriched fields (base gradients in the first block, zeros in the enriched block).
        """
        field = str(field)
        tab = self.base._eval_scalar_grad_many(field, xi_arr, eta_arr)  # (n_qp, n0, 2)
        if self._enrich_kind_by_field.get(field) is None:
            return tab
        nqp, n0, dim = tab.shape
        out = np.zeros((nqp, 2 * n0, dim), dtype=float)
        out[:, :n0, :] = tab
        return out
