from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from pycutfem.core.dofhandler import DofHandler


@dataclass(frozen=True)
class EnrichmentSpec:
    """
    Per-field enrichment kind.

    kind:
      - None        : do not enrich
      - "heaviside" : strong discontinuity (shifted Heaviside)
      - "abs"       : weak discontinuity (shifted |phi|)
    """

    kind_by_field: Dict[str, str | None]

    @staticmethod
    def from_user(
        dh: DofHandler,
        spec: Mapping[str, str | None] | Sequence[str] | None,
        *,
        default_kind: str = "heaviside",
    ) -> "EnrichmentSpec":
        if spec is None:
            kind_by_field = {f: None for f in dh.field_names}
            return EnrichmentSpec(kind_by_field=kind_by_field)

        if isinstance(spec, (list, tuple, set)):
            fields = set(str(f) for f in spec)
            kind_by_field = {f: (default_kind if f in fields else None) for f in dh.field_names}
            return EnrichmentSpec(kind_by_field=kind_by_field)

        kind_by_field = {f: None for f in dh.field_names}
        for f, k in dict(spec).items():
            if f not in kind_by_field:
                raise KeyError(f"Unknown field '{f}' for enrichment.")
            if k is None:
                kind_by_field[f] = None
                continue
            kk = str(k).lower().strip()
            if kk not in {"heaviside", "abs"}:
                raise ValueError(f"Invalid enrichment kind '{k}' for field '{f}'.")
            kind_by_field[f] = kk
        return EnrichmentSpec(kind_by_field=kind_by_field)

    @property
    def enriched_fields(self) -> tuple[str, ...]:
        return tuple([f for f, k in self.kind_by_field.items() if k is not None])


class XFEMDofHandler:
    """
    Wrapper around a base DofHandler that manages dynamic XFEM enriched DOFs.

    - Base DOFs keep their numbering.
    - Enriched DOFs are appended after base DOFs (global ordering).
    - For cut elements, local elemental DOFs can be requested as base+enriched.
    """

    def __init__(self, base: DofHandler):
        if not isinstance(base, DofHandler):
            raise TypeError("XFEMDofHandler expects a pycutfem.core.dofhandler.DofHandler.")
        if base.mixed_element is None:
            raise RuntimeError("XFEMDofHandler requires a MixedElement-backed DofHandler.")

        self.base = base

        # Immutable base-space info
        self.base_total_dofs: int = int(getattr(base, "total_dofs"))
        self.field_names = list(base.field_names)
        self.mixed_element = base.mixed_element
        self.method = getattr(base, "method", "cg")

        # Dynamic enrichment state
        self.enrichment_spec: EnrichmentSpec = EnrichmentSpec.from_user(base, None)
        self.enrichment_map: Dict[tuple[str, int], int] = {}
        self.enriched_base_dofs_by_field: Dict[str, np.ndarray] = {f: np.empty((0,), dtype=int) for f in self.field_names}
        self.enriched_dofs_by_field: Dict[str, np.ndarray] = {f: np.empty((0,), dtype=int) for f in self.field_names}
        self.total_dofs: int = self.base_total_dofs

        # Extended coordinate table (allocated on rebuild)
        self._dof_coords_ext: np.ndarray | None = None
        # Enriched DOFs that are identically inactive (e.g. base node on φ=0)
        # and must be constrained to 0 to avoid singular systems.
        self._fixed_enriched_dofs: set[int] = set()
        # Base DOFs that must be fixed to 0 in *one-sided* Heaviside assemblies
        # (otherwise u_i and a_i appear only as a sum/difference and the system
        # becomes rank-deficient).
        self._fixed_base_dofs: set[int] = set()
        # Optional: remember which side is being assembled in one-sided problems.
        # When set, facet restriction masks remain CutFEM-style (side masks) for
        # the base block to avoid inadvertently changing a one-sided formulation
        # into a two-sided trace evaluation on facets.
        self._active_side: str | None = None

        # Extended field slices (base slices cached; enriched appended on rebuild)
        self._field_slices_base: Dict[str, np.ndarray] = {
            f: np.asarray(self.base.get_field_slice(f), dtype=int) for f in self.field_names
        }
        self._field_slices_ext: Dict[str, np.ndarray] = dict(self._field_slices_base)

    # ------------------------------------------------------------------
    # Delegation / compatibility
    # ------------------------------------------------------------------
    def __getattr__(self, name):
        # Delegate unknown attributes/methods to the base handler.
        return getattr(self.base, name)

    def get_field_slice(self, field: str) -> np.ndarray:
        """Return global DOFs for a field (base + enriched, if any)."""
        if field not in self._field_slices_ext:
            raise KeyError(field)
        return self._field_slices_ext[field]

    def get_all_dof_coords(self) -> np.ndarray:
        """Coordinates for every global DOF (total_dofs, 2), including enrichment."""
        self.base._ensure_dof_coords()
        if self._dof_coords_ext is None:
            return self.base.get_all_dof_coords()
        return self._dof_coords_ext.copy()

    def get_elemental_dofs(self, element_id: int) -> np.ndarray:
        """Base elemental DOFs (no enrichment)."""
        return self.base.get_elemental_dofs(int(element_id))

    def get_elemental_dofs_xfem(self, element_id: int) -> np.ndarray:
        """
        Elemental DOFs for XFEM contexts.

        Local ordering is per-field:
          [ base(field0), enr(field0), base(field1), enr(field1), ... ]
        for fields marked as enriched. Non-enriched fields contribute base only.
        """
        eid = int(element_id)
        parts: list[int] = []
        for fld in self.field_names:
            base_loc = self.base.element_maps[fld][eid]
            parts.extend(int(g) for g in base_loc)
            if self.enrichment_spec.kind_by_field.get(fld) is None:
                continue
            # append enriched dofs for those base dofs that are enriched
            for gd in base_loc:
                enr = self.enrichment_map.get((fld, int(gd)))
                if enr is not None:
                    parts.append(int(enr))
        return np.asarray(parts, dtype=int)

    def add_to_functions(self, delta, functions):
        """
        XFEM-aware version of DofHandler.add_to_functions.

        The base DofHandler implementation validates against `base.total_dofs`,
        which is incorrect once enriched DOFs are appended. This override keeps
        the same semantics but checks against `self.total_dofs`.
        """
        from pycutfem.ufl.expressions import Function, VectorFunction

        if isinstance(delta, np.ndarray):
            delta_vec = delta
        elif isinstance(delta, (Function, VectorFunction)):
            delta_vec = delta.nodal_values
        else:
            raise TypeError(
                f"Argument 'delta' must be a NumPy array or a Function object, not {type(delta)}"
            )

        if int(delta_vec.shape[0]) != int(self.total_dofs):
            raise ValueError(
                f"Shape of delta vector ({delta_vec.shape[0]}) does not match total DOFs in handler ({self.total_dofs})."
            )

        for func in functions:
            target_array = None
            g2l_map = None

            if isinstance(func, VectorFunction):
                target_array = func.nodal_values
                g2l_map = func._g2l
            elif isinstance(func, Function) and func._parent_vector is None:
                target_array = func._values
                g2l_map = func._g2l

            if target_array is not None and g2l_map is not None:
                for gdof, lidx in g2l_map.items():
                    if gdof < len(delta_vec):
                        target_array[lidx] += delta_vec[gdof]

    # ------------------------------------------------------------------
    # Enrichment build/rebuild
    # ------------------------------------------------------------------
    def rebuild_enrichment(
        self,
        level_set,
        *,
        enrich: Mapping[str, str | None] | Sequence[str] | None = None,
        active_side: str | None = None,
        tol: float = 1.0e-12,
    ) -> None:
        """
        Rebuild enriched DOF maps from the current level-set classification.

        Enriched nodes for a field are the union of that field's base DOFs that
        appear in any cut element.
        """
        if active_side is not None:
            active_side = str(active_side).strip()
            if active_side not in {"+", "-"}:
                raise ValueError("active_side must be '+', '-' or None.")
        # Remember one-sided intent for facet restriction masks.
        self._active_side = active_side

        self.enrichment_spec = EnrichmentSpec.from_user(self.base, enrich)
        enriched_fields = list(self.enrichment_spec.enriched_fields)

        # Reset to base if nothing is enriched.
        if not enriched_fields:
            self.enrichment_map.clear()
            self.enriched_base_dofs_by_field = {f: np.empty((0,), dtype=int) for f in self.field_names}
            self.enriched_dofs_by_field = {f: np.empty((0,), dtype=int) for f in self.field_names}
            self.total_dofs = self.base_total_dofs
            self._dof_coords_ext = None
            self._field_slices_ext = dict(self._field_slices_base)
            self._fixed_enriched_dofs = set()
            self._fixed_base_dofs = set()
            self._active_side = None
            return

        mesh = self.base.mixed_element.mesh

        # Re-classify elements for the given level set. This also refreshes the
        # mesh bitset caches used downstream by CutFEM/XFEM assembly.
        _, _, cut_ids = mesh.classify_elements(level_set, tol=tol)
        cut_ids = np.asarray(cut_ids, dtype=int)

        enriched_base_by_field: Dict[str, np.ndarray] = {f: np.empty((0,), dtype=int) for f in self.field_names}
        if cut_ids.size:
            for f in enriched_fields:
                s: set[int] = set()
                emap = self.base.element_maps[f]
                for eid in cut_ids:
                    for gd in emap[int(eid)]:
                        s.add(int(gd))
                enriched_base_by_field[f] = np.asarray(sorted(s), dtype=int)

        # Deterministic assignment: (field order in MixedElement) × (base_gdof asc)
        next_gid = int(self.base_total_dofs)
        enr_map: Dict[tuple[str, int], int] = {}
        enr_gdofs_by_field: Dict[str, np.ndarray] = {f: np.empty((0,), dtype=int) for f in self.field_names}
        assign_order: list[tuple[str, int]] = []
        for f in self.field_names:
            if f not in enriched_fields:
                continue
            base_list = enriched_base_by_field.get(f)
            if base_list is None or base_list.size == 0:
                continue
            gids = np.empty_like(base_list)
            for i, gd in enumerate(base_list):
                enr_map[(f, int(gd))] = int(next_gid)
                gids[i] = int(next_gid)
                assign_order.append((f, int(gd)))
                next_gid += 1
            enr_gdofs_by_field[f] = gids

        self.enrichment_map = enr_map
        self.enriched_base_dofs_by_field = enriched_base_by_field
        self.enriched_dofs_by_field = enr_gdofs_by_field
        self.total_dofs = int(next_gid)

        # Extend dof coords: enriched coords reuse the base dof coordinates.
        self.base._ensure_dof_coords()
        base_coords = np.asarray(self.base._dof_coords, dtype=float)

        # Mark enriched DOFs that are identically inactive and must be constrained
        # to 0 to avoid singular systems.
        #
        # 1) Nodes on the interface (|φ|<=tol): shifted-Heaviside α vanishes on both
        #    sides there.
        # 2) If the caller only assembles *one* side of the cut domain (e.g. a
        #    fictitious-domain obstacle problem assembling only Ω⁺), then half of
        #    the Heaviside enriched DOFs never appear in any integral (α=0 on that
        #    assembled side) and must also be fixed.
        fixed: set[int] = set()
        fixed_base: set[int] = set()
        if assign_order:
            for (f, gd) in assign_order:
                try:
                    phi = float(level_set(base_coords[int(gd), :]))
                except Exception:
                    try:
                        phi = float(level_set(np.asarray(base_coords[int(gd), :], dtype=float)))
                    except Exception:
                        continue
                enr = enr_map.get((str(f), int(gd)))
                if enr is None:
                    continue

                # (1) φ≈0: α=0 for shifted-Heaviside
                if abs(phi) <= float(tol):
                    fixed.add(int(enr))
                    continue

                # (2) One-sided assemblies: fix Heaviside enriched DOFs with α=0
                # on the assembled side.
                kind = (self.enrichment_spec.kind_by_field.get(str(f)) or "").lower().strip()
                if active_side is not None and kind == "heaviside":
                    # In a one-sided assembly (e.g. only Ω⁺), the trace on that side
                    # depends on u_i and a_i only through their sum/difference for
                    # nodes on the opposite sign side. Fixing the base DOF removes
                    # the redundancy without changing the represented trace.
                    if active_side == "+" and phi < -float(tol):
                        fixed_base.add(int(gd))
                    elif active_side == "-" and phi > float(tol):
                        fixed_base.add(int(gd))

                    # Heaviside convention: H_i = 1 on φ>0, 0 on φ<0.
                    # α⁺ = 1 - H_i  -> zero on φ>0
                    # α⁻ = 0 - H_i  -> zero on φ<0
                    if active_side == "+" and phi > float(tol):
                        fixed.add(int(enr))
                    elif active_side == "-" and phi < -float(tol):
                        fixed.add(int(enr))
        self._fixed_enriched_dofs = fixed
        self._fixed_base_dofs = fixed_base

        enr_coords = np.empty((len(assign_order), 2), dtype=float)
        base_node_map = getattr(self.base, "_dof_to_node_map", {}) or {}
        ext_node_map = dict(base_node_map)

        for i, (f, gd) in enumerate(assign_order):
            enr_coords[i, :] = base_coords[int(gd), :]
            enr_gid = int(self.enrichment_map[(f, int(gd))])
            ext_node_map[enr_gid] = (f, base_node_map.get(int(gd), (f, None))[1])

        self._dof_coords_ext = np.vstack([base_coords, enr_coords])

        # Extend field slices for enriched fields (base slice + enriched dofs)
        ext_slices: Dict[str, np.ndarray] = {}
        for f in self.field_names:
            base_sl = self._field_slices_base[f]
            enr_sl = enr_gdofs_by_field.get(f)
            if enr_sl is None or enr_sl.size == 0:
                ext_slices[f] = base_sl
            else:
                ext_slices[f] = np.concatenate([base_sl, np.asarray(enr_sl, dtype=int)])
        self._field_slices_ext = ext_slices

        # Keep an extended dof->node map for downstream utilities (plotting/vtk).
        self._dof_to_node_map = ext_node_map

    def get_dirichlet_data(self, bcs, locators=None):
        """
        Extend base Dirichlet elimination with internal XFEM constraints.

        Enriched DOFs whose base coordinate lies on the interface (|φ|<=tol at
        enrichment build time) do not contribute to any bilinear/linear forms
        under shifted-Heaviside α and would otherwise create a singular system.
        """
        data = self.base.get_dirichlet_data(bcs, locators=locators)
        if self._fixed_enriched_dofs:
            for gd in self._fixed_enriched_dofs:
                data[int(gd)] = 0.0
        if self._fixed_base_dofs:
            for gd in self._fixed_base_dofs:
                data[int(gd)] = 0.0
        return data

    # Convenience for tests / external callers
    def n_enriched(self) -> int:
        return int(self.total_dofs - self.base_total_dofs)

    def enriched_fields(self) -> tuple[str, ...]:
        return self.enrichment_spec.enriched_fields

    def enriched_dofs(self, field: str) -> np.ndarray:
        return np.asarray(self.enriched_dofs_by_field.get(field, np.empty((0,), dtype=int)), dtype=int)

    # ------------------------------------------------------------------
    # XFEM local layout helpers (for kernels / precompute)
    # ------------------------------------------------------------------
    def xfem_mixed_element(self):
        """
        Return an XFEMMixedElement reflecting the current enrichment spec.
        """
        from pycutfem.xfem.mixedelement import XFEMMixedElement

        return XFEMMixedElement(self.base.mixed_element, enrich_kind_by_field=self.enrichment_spec.kind_by_field)

    # ------------------------------------------------------------------
    # Precompute: cut volume (XFEM union expansion)
    # ------------------------------------------------------------------
    def precompute_cut_volume_factors(
        self,
        element_bitset,
        qdeg: int,
        derivs: set[tuple[int, int]],
        level_set,
        side: str = "+",
        reuse: bool = True,
        need_hess: bool = False,
        need_o3: bool = False,
        need_o4: bool = False,
        nseg_hint: int | None = None,
        deformation: any = None,
    ) -> dict:
        """
        XFEM-aware cut-volume precompute.

        Falls back to the base DofHandler if no fields are currently enriched.
        When enriched fields exist, expands union-sized tables by appending the
        enriched DOF block within each enriched field slice (base+enr contiguous).
        """
        enriched_fields = list(self.enrichment_spec.enriched_fields)
        if not enriched_fields:
            return self.base.precompute_cut_volume_factors(
                element_bitset,
                qdeg,
                derivs,
                level_set,
                side=side,
                reuse=reuse,
                need_hess=need_hess,
                need_o3=need_o3,
                need_o4=need_o4,
                nseg_hint=nseg_hint,
                deformation=deformation,
            )

        from pycutfem.ufl.helpers import HelpersFieldAware as _hfa
        from pycutfem.xfem.enrichment import alpha_from_side_masks
        from pycutfem.core.sideconvention import SIDE

        geo = self.base.precompute_cut_volume_factors(
            element_bitset,
            qdeg,
            derivs,
            level_set,
            side=side,
            reuse=reuse,
            need_hess=need_hess,
            need_o3=need_o3,
            need_o4=need_o4,
            nseg_hint=nseg_hint,
            deformation=deformation,
        )

        eids = np.asarray(geo.get("eids", []), dtype=int)
        if eids.size == 0:
            return geo

        base_me = self.base.mixed_element
        cut_me = self.xfem_mixed_element()
        n_union_cut = int(cut_me.n_dofs_local)

        # Alpha per element per enriched field (constant over QPs on a given side)
        alpha_by_field: Dict[str, np.ndarray] = {}
        for f in enriched_fields:
            n0 = int(base_me._n_basis[f])
            alpha_by_field[f] = np.zeros((eids.size, n0), dtype=float)

        for ie, eid in enumerate(eids.tolist()):
            pos_masks, neg_masks = _hfa.build_side_masks_by_field(
                self.base, enriched_fields, int(eid), level_set, tol=SIDE.tol
            )
            for f in enriched_fields:
                pm = pos_masks.get(f)
                nm = neg_masks.get(f)
                if pm is None or nm is None:
                    continue
                alpha_by_field[f][ie, :] = alpha_from_side_masks(pm, nm, side=side)

        # Replace gdofs_map with XFEM-local layout (per field base+enr)
        geo["gdofs_map"] = np.vstack([self.get_elemental_dofs_xfem(int(eid)) for eid in eids]).astype(np.int64)

        # Expand per-field union tables
        for f in base_me.field_names:
            base_sl = base_me.component_dof_slices[f]
            cut_sl = cut_me.component_dof_slices[f]
            n0 = int(base_sl.stop - base_sl.start)
            qtab = np.asarray(geo.get(f"b_{f}"))
            gtab = np.asarray(geo.get(f"g_{f}"))
            if qtab.size == 0:
                continue
            nE, nQ, _ = qtab.shape

            b_ext = np.zeros((nE, nQ, n_union_cut), dtype=qtab.dtype)
            g_ext = np.zeros((nE, nQ, n_union_cut, 2), dtype=gtab.dtype)

            # base part
            b_loc = qtab[:, :, base_sl]
            g_loc = gtab[:, :, base_sl, :]
            b_ext[:, :, cut_sl.start : cut_sl.start + n0] = b_loc
            g_ext[:, :, cut_sl.start : cut_sl.start + n0, :] = g_loc

            # enriched part (if requested)
            if f in alpha_by_field:
                a = alpha_by_field[f]  # (nE, n0)
                b_ext[:, :, cut_sl.start + n0 : cut_sl.start + 2 * n0] = b_loc * a[:, None, :]
                g_ext[:, :, cut_sl.start + n0 : cut_sl.start + 2 * n0, :] = g_loc * a[:, None, :, None]

            geo[f"b_{f}"] = b_ext
            geo[f"g_{f}"] = g_ext

            # Higher derivatives (if present)
            for (dx, dy) in sorted(set(tuple(map(int, t)) for t in derivs) | {(0, 0)}):
                if (dx, dy) == (0, 0):
                    continue
                key = f"d{dx}{dy}_{f}"
                if key not in geo:
                    continue
                dtab = np.asarray(geo[key])
                if dtab.size == 0:
                    geo[key] = np.zeros((0, 0, n_union_cut), dtype=float)
                    continue
                d_loc = dtab[:, :, base_sl]
                d_ext = np.zeros((nE, nQ, n_union_cut), dtype=dtab.dtype)
                d_ext[:, :, cut_sl.start : cut_sl.start + n0] = d_loc
                if f in alpha_by_field:
                    a = alpha_by_field[f]
                    d_ext[:, :, cut_sl.start + n0 : cut_sl.start + 2 * n0] = d_loc * a[:, None, :]
                geo[key] = d_ext

        return geo

    # ------------------------------------------------------------------
    # Precompute: interface + ghost (XFEM enrichment via masks + layout)
    # ------------------------------------------------------------------
    def precompute_interface_factors(
        self,
        cut_element_ids,
        qdeg: int,
        level_set,
        nseg: int | None = None,
        reuse: bool = True,
        need_hess: bool = False,
        need_o3: bool = False,
        need_o4: bool = False,
        deformation: any = None,
    ) -> dict:
        """
        XFEM-aware interface precompute.

        Strategy:
        - call the base handler to get interface quadrature + base tables
        - expand union-local tables to the XFEM cut-element layout (base+enr contiguous)
        - keep base DOFs masked as in the base implementation
        - encode the enrichment multipliers α into the per-field restrict masks for
          the enriched block (so both test/trial basis and coefficient gathering see it)

        Notes
        -----
        This implementation currently supports shifted-Heaviside enrichment only.
        """
        enriched_fields = list(self.enrichment_spec.enriched_fields)
        if not enriched_fields:
            return self.base.precompute_interface_factors(
                cut_element_ids,
                qdeg,
                level_set,
                nseg=nseg,
                reuse=reuse,
                need_hess=need_hess,
                need_o3=need_o3,
                need_o4=need_o4,
                deformation=deformation,
            )

        # Ensure we only claim support for what we actually implement.
        for f in enriched_fields:
            if (self.enrichment_spec.kind_by_field.get(f) or "").lower().strip() != "heaviside":
                raise NotImplementedError("Interface factors currently support only 'heaviside' XFEM enrichment.")

        from pycutfem.xfem.enrichment import alpha_from_side_masks

        geo_base = self.base.precompute_interface_factors(
            cut_element_ids,
            qdeg,
            level_set,
            nseg=nseg,
            reuse=reuse,
            need_hess=need_hess,
            need_o3=need_o3,
            need_o4=need_o4,
            deformation=deformation,
        )

        eids = np.asarray(geo_base.get("eids", []), dtype=int)
        if eids.size == 0:
            return geo_base

        base_me = self.base.mixed_element
        cut_me = self.xfem_mixed_element()
        n_union_base = int(base_me.n_dofs_per_elem)
        n_union_cut = int(cut_me.n_dofs_per_elem)

        # Start from a shallow copy of geometry; replace tables below.
        out = dict(geo_base)

        # Replace gdofs_map with XFEM-local layout (per field base+enr).
        out["gdofs_map"] = np.vstack([self.get_elemental_dofs_xfem(int(eid)) for eid in eids]).astype(np.int64)

        # pos_map/neg_map are identity for dInterface (same element on both sides).
        ident = np.tile(np.arange(n_union_cut, dtype=np.int32), (eids.size, 1))
        out["pos_map"] = ident
        out["neg_map"] = ident

        # Expand per-field union tables and restrict masks
        for f in base_me.field_names:
            base_sl = base_me.component_dof_slices[f]
            cut_sl = cut_me.component_dof_slices[f]
            n0 = int(base_sl.stop - base_sl.start)

            # --- basis value/grad tables (copy base into both base+enr blocks) ---
            b_key = f"b_{f}"
            g_key = f"g_{f}"
            btab = np.asarray(out.get(b_key))
            gtab = np.asarray(out.get(g_key))
            if btab.size:
                b_loc = btab[:, :, base_sl]  # (nE,nQ,n0)
                b_ext = np.zeros((b_loc.shape[0], b_loc.shape[1], n_union_cut), dtype=btab.dtype)
                b_ext[:, :, cut_sl.start : cut_sl.start + n0] = b_loc
                if f in enriched_fields:
                    b_ext[:, :, cut_sl.start + n0 : cut_sl.start + 2 * n0] = b_loc
                out[b_key] = b_ext
            if gtab.size:
                g_loc = gtab[:, :, base_sl, :]  # (nE,nQ,n0,2)
                g_ext = np.zeros((g_loc.shape[0], g_loc.shape[1], n_union_cut, 2), dtype=gtab.dtype)
                g_ext[:, :, cut_sl.start : cut_sl.start + n0, :] = g_loc
                if f in enriched_fields:
                    g_ext[:, :, cut_sl.start + n0 : cut_sl.start + 2 * n0, :] = g_loc
                out[g_key] = g_ext

            # --- higher derivative tables (if present) ---
            for (dx, dy) in ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4)):
                d_key = f"d{dx}{dy}_{f}"
                if d_key not in out:
                    continue
                dtab = np.asarray(out.get(d_key))
                if dtab.size == 0:
                    continue
                d_loc = dtab[:, :, base_sl]  # (nE,nQ,n0)
                d_ext = np.zeros((d_loc.shape[0], d_loc.shape[1], n_union_cut), dtype=dtab.dtype)
                d_ext[:, :, cut_sl.start : cut_sl.start + n0] = d_loc
                if f in enriched_fields:
                    d_ext[:, :, cut_sl.start + n0 : cut_sl.start + 2 * n0] = d_loc
                out[d_key] = d_ext

            # --- restriction masks (pos/neg) ---
            if level_set is not None:
                rm_pos_key = f"restrict_mask_{f}_pos"
                rm_neg_key = f"restrict_mask_{f}_neg"
                rm_pos = np.asarray(out.get(rm_pos_key))
                rm_neg = np.asarray(out.get(rm_neg_key))
                if rm_pos.size and rm_neg.size:
                    # Recover field-local side masks (0/1, interface nodes possibly 1 in both)
                    m_pos_loc = np.asarray(rm_pos[:, base_sl], dtype=float)
                    m_neg_loc = np.asarray(rm_neg[:, base_sl], dtype=float)

                    rm_pos_ext = np.zeros((m_pos_loc.shape[0], n_union_cut), dtype=float)
                    rm_neg_ext = np.zeros((m_neg_loc.shape[0], n_union_cut), dtype=float)

                    # Base block: keep original side masks (as in base CutFEM/XDG usage)
                    if f in enriched_fields:
                        if self._active_side is None:
                            # True XFEM (two-sided): base shape functions contribute on both sides.
                            rm_pos_ext[:, cut_sl.start : cut_sl.start + n0] = 1.0
                            rm_neg_ext[:, cut_sl.start : cut_sl.start + n0] = 1.0
                        else:
                            # One-sided use (e.g. fictitious domain): keep CutFEM-style
                            # side masks for the base block.
                            rm_pos_ext[:, cut_sl.start : cut_sl.start + n0] = m_pos_loc
                            rm_neg_ext[:, cut_sl.start : cut_sl.start + n0] = m_neg_loc
                    else:
                        rm_pos_ext[:, cut_sl.start : cut_sl.start + n0] = m_pos_loc
                        rm_neg_ext[:, cut_sl.start : cut_sl.start + n0] = m_neg_loc

                    # Enriched block: α multipliers per side
                    if f in enriched_fields:
                        a_pos = np.vstack([alpha_from_side_masks(m_pos_loc[i], m_neg_loc[i], side="+") for i in range(m_pos_loc.shape[0])])
                        a_neg = np.vstack([alpha_from_side_masks(m_pos_loc[i], m_neg_loc[i], side="-") for i in range(m_pos_loc.shape[0])])
                        rm_pos_ext[:, cut_sl.start + n0 : cut_sl.start + 2 * n0] = a_pos
                        rm_neg_ext[:, cut_sl.start + n0 : cut_sl.start + 2 * n0] = a_neg

                    out[rm_pos_key] = rm_pos_ext
                    out[rm_neg_key] = rm_neg_ext

                    # JIT expects sided reference tables r**_{field}_{pos|neg} on facet-like
                    # measures (including dInterface). For XFEM, the enriched block must be
                    # scaled by the trace multiplier α on each side; we encode this directly
                    # into r** tables so the JIT backend does not rely on implicit masking.
                    if f in enriched_fields:
                        # r00: basis values on Γ
                        if b_key in out:
                            b_tbl = np.asarray(out[b_key])
                            out[f"r00_{f}_pos"] = b_tbl * rm_pos_ext[:, None, :]
                            out[f"r00_{f}_neg"] = b_tbl * rm_neg_ext[:, None, :]

                        # r{dx}{dy}: reference derivatives on Γ
                        for (dx, dy) in (
                            (1, 0), (0, 1),
                            (2, 0), (1, 1), (0, 2),
                            (3, 0), (2, 1), (1, 2), (0, 3),
                            (4, 0), (3, 1), (2, 2), (1, 3), (0, 4),
                        ):
                            d_key = f"d{dx}{dy}_{f}"
                            if d_key not in out:
                                continue
                            d_tbl = np.asarray(out[d_key])
                            if d_tbl.size == 0:
                                continue
                            out[f"r{dx}{dy}_{f}_pos"] = d_tbl * rm_pos_ext[:, None, :]
                            out[f"r{dx}{dy}_{f}_neg"] = d_tbl * rm_neg_ext[:, None, :]

        return out

    def precompute_ghost_factors(
        self,
        ghost_edge_ids,
        qdeg: int,
        level_set,
        derivs: set[tuple[int, int]],
        allow_interface: bool = False,
        reuse: bool = True,
        need_hess: bool = False,
        need_o3: bool = False,
        need_o4: bool = False,
        deformation: any = None,
    ) -> dict:
        """
        XFEM-aware ghost-edge precompute.

        Expands:
        - per-side element-local union tables (r**_{field}_{pos|neg}) to XFEM layout
        - per-field side maps (pos_map_{field}, neg_map_{field}) to include enriched DOFs
        - union gdofs_map (per edge) to include enriched DOFs of cut elements
        - restrict_mask_{field}_{pos|neg} to include Heaviside α multipliers on the enriched block
        """
        enriched_fields = list(self.enrichment_spec.enriched_fields)
        if not enriched_fields:
            return self.base.precompute_ghost_factors(
                ghost_edge_ids=ghost_edge_ids,
                qdeg=qdeg,
                level_set=level_set,
                derivs=derivs,
                allow_interface=allow_interface,
                reuse=reuse,
                need_hess=need_hess,
                need_o3=need_o3,
                need_o4=need_o4,
                deformation=deformation,
            )

        for f in enriched_fields:
            if (self.enrichment_spec.kind_by_field.get(f) or "").lower().strip() != "heaviside":
                raise NotImplementedError("Ghost factors currently support only 'heaviside' XFEM enrichment.")

        from pycutfem.ufl.helpers import HelpersFieldAware as _hfa
        from pycutfem.core.sideconvention import SIDE
        from pycutfem.xfem.enrichment import alpha_from_side_masks

        geo_base = self.base.precompute_ghost_factors(
            ghost_edge_ids=ghost_edge_ids,
            qdeg=qdeg,
            level_set=level_set,
            derivs=derivs,
            allow_interface=allow_interface,
            reuse=reuse,
            need_hess=need_hess,
            need_o3=need_o3,
            need_o4=need_o4,
            deformation=deformation,
        )

        edge_ids = np.asarray(geo_base.get("eids", []), dtype=int)
        if edge_ids.size == 0:
            return geo_base

        mesh = self.base.mixed_element.mesh
        base_me = self.base.mixed_element
        cut_me = self.xfem_mixed_element()
        n_loc_base = int(base_me.n_dofs_per_elem)
        n_loc_cut = int(cut_me.n_dofs_per_elem)

        pos_ids = np.asarray(geo_base.get("owner_pos_id"), dtype=int)
        neg_ids = np.asarray(geo_base.get("owner_neg_id"), dtype=int)
        if pos_ids.size != edge_ids.size or neg_ids.size != edge_ids.size:
            return geo_base

        # Decide whether to include XFEM DOFs for an element on this edge.
        def _elem_dofs_for_edge(eid: int) -> np.ndarray:
            tag = str(getattr(mesh.elements_list[int(eid)], "tag", ""))
            if tag == "cut":
                return self.get_elemental_dofs_xfem(int(eid))
            return self.base.get_elemental_dofs(int(eid))

        # --- Union gdofs per edge (include enriched DOFs from cut elements) ---
        union_lists: list[np.ndarray] = []
        max_union = int(geo_base["gdofs_map"].shape[1]) if "gdofs_map" in geo_base else 0
        pos_len_max = 0
        neg_len_max = 0
        for i in range(edge_ids.size):
            pd = np.asarray(_elem_dofs_for_edge(int(pos_ids[i])), dtype=np.int64)
            nd = np.asarray(_elem_dofs_for_edge(int(neg_ids[i])), dtype=np.int64)
            union = np.unique(np.concatenate((pd, nd))).astype(np.int64)
            union_lists.append(union)
            max_union = max(max_union, int(union.size))
            pos_len_max = max(pos_len_max, int(pd.size))
            neg_len_max = max(neg_len_max, int(nd.size))

        gdofs_map = np.empty((edge_ids.size, max_union), dtype=np.int64)
        for i, union in enumerate(union_lists):
            pad_val = int(union[-1]) if union.size else 0
            gdofs_map[i, :] = pad_val
            gdofs_map[i, : union.size] = union

        # Side maps (element-local dofs -> edge union indices) for coefficient gathering.
        # These must be sized to the XFEM mixed-element local width so codegen can slice.
        pos_map = -np.ones((edge_ids.size, n_loc_cut), dtype=np.int32)
        neg_map = -np.ones((edge_ids.size, n_loc_cut), dtype=np.int32)

        # Per-field maps used by pad_basis_to_union: field-local dofs -> edge union indices
        maps_by_field: Dict[str, np.ndarray] = {}
        for fld in base_me.field_names:
            base_sl = base_me.component_dof_slices[fld]
            n0 = int(base_sl.stop - base_sl.start)
            nloc_f_cut = 2 * n0 if fld in enriched_fields else n0
            maps_by_field[f"pos_map_{fld}"] = -np.ones((edge_ids.size, nloc_f_cut), dtype=np.int32)
            maps_by_field[f"neg_map_{fld}"] = -np.ones((edge_ids.size, nloc_f_cut), dtype=np.int32)

        # Restriction masks in edge-union layout
        masks_by_field: Dict[str, np.ndarray] = {}
        if level_set is not None:
            for fld in base_me.field_names:
                masks_by_field[f"restrict_mask_{fld}_pos"] = np.zeros((edge_ids.size, max_union), dtype=float)
                masks_by_field[f"restrict_mask_{fld}_neg"] = np.zeros((edge_ids.size, max_union), dtype=float)

        # Cache per-element side masks to avoid repeated evaluations
        mask_cache: Dict[int, tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = {}

        def _elem_masks(eid: int):
            if eid not in mask_cache:
                mask_cache[eid] = _hfa.build_side_masks_by_field(
                    self.base, list(base_me.field_names), int(eid), level_set, tol=SIDE.tol
                )
            return mask_cache[eid]

        # Fill maps and masks per edge
        for i in range(edge_ids.size):
            union = union_lists[i]
            pos_eid = int(pos_ids[i])
            neg_eid = int(neg_ids[i])
            pos_is_cut = str(getattr(mesh.elements_list[pos_eid], "tag", "")) == "cut"
            neg_is_cut = str(getattr(mesh.elements_list[neg_eid], "tag", "")) == "cut"

            # For each field, map base/enriched dofs into edge-union indices
            for fld in base_me.field_names:
                base_sl = base_me.component_dof_slices[fld]
                cut_sl = cut_me.component_dof_slices[fld]
                n0 = int(base_sl.stop - base_sl.start)
                base_pos_loc = np.asarray(self.base.element_maps[fld][pos_eid], dtype=np.int64)
                base_neg_loc = np.asarray(self.base.element_maps[fld][neg_eid], dtype=np.int64)

                # pos_map/neg_map in element-union layout (mixed) for coefficient gathering
                for j, gd in enumerate(base_pos_loc.tolist()):
                    idx = int(np.searchsorted(union, int(gd)))
                    pos_map[i, cut_sl.start + j] = idx
                for j, gd in enumerate(base_neg_loc.tolist()):
                    idx = int(np.searchsorted(union, int(gd)))
                    neg_map[i, cut_sl.start + j] = idx

                # per-field maps (field-local)
                pm = maps_by_field[f"pos_map_{fld}"]
                nm = maps_by_field[f"neg_map_{fld}"]
                for j, gd in enumerate(base_pos_loc.tolist()):
                    pm[i, j] = int(np.searchsorted(union, int(gd)))
                for j, gd in enumerate(base_neg_loc.tolist()):
                    nm[i, j] = int(np.searchsorted(union, int(gd)))

                # Enriched block mapping (only if this field is enriched and the element is cut)
                if fld in enriched_fields:
                    if pos_is_cut:
                        for j, gd in enumerate(base_pos_loc.tolist()):
                            enr = self.enrichment_map.get((fld, int(gd)))
                            if enr is None:
                                continue
                            idx = int(np.searchsorted(union, int(enr)))
                            pos_map[i, cut_sl.start + n0 + j] = idx
                            pm[i, n0 + j] = idx
                    if neg_is_cut:
                        for j, gd in enumerate(base_neg_loc.tolist()):
                            enr = self.enrichment_map.get((fld, int(gd)))
                            if enr is None:
                                continue
                            idx = int(np.searchsorted(union, int(enr)))
                            neg_map[i, cut_sl.start + n0 + j] = idx
                            nm[i, n0 + j] = idx

                # Restriction masks (apply to coefficients on facet integrals)
                if level_set is not None:
                    pos_masks_elem_pos, neg_masks_elem_pos = _elem_masks(pos_eid)
                    pos_masks_elem_neg, neg_masks_elem_neg = _elem_masks(neg_eid)

                    mpos_loc = pos_masks_elem_pos.get(fld)
                    mneg_loc = neg_masks_elem_neg.get(fld)

                    if mpos_loc is not None and mpos_loc.shape[0] == n0:
                        cols = pm[i, :n0]
                        ok = (cols >= 0) & (cols < max_union)
                        if fld in enriched_fields:
                            if self._active_side is None:
                                masks_by_field[f"restrict_mask_{fld}_pos"][i, cols[ok]] = 1.0
                            else:
                                masks_by_field[f"restrict_mask_{fld}_pos"][i, cols[ok]] = np.asarray(mpos_loc, dtype=float)[ok]
                        else:
                            masks_by_field[f"restrict_mask_{fld}_pos"][i, cols[ok]] = np.asarray(mpos_loc, dtype=float)[ok]
                        if fld in enriched_fields and pos_is_cut:
                            mneg_local_same_elem = neg_masks_elem_pos.get(fld)
                            if mneg_local_same_elem is None:
                                mneg_local_same_elem = np.zeros_like(mpos_loc, dtype=float)
                            a_pos = alpha_from_side_masks(mpos_loc, mneg_local_same_elem, side="+")
                            cols_enr = pm[i, n0 : 2 * n0]
                            ok2 = (cols_enr >= 0) & (cols_enr < max_union)
                            masks_by_field[f"restrict_mask_{fld}_pos"][i, cols_enr[ok2]] = np.asarray(a_pos, dtype=float)[ok2]

                    if mneg_loc is not None and mneg_loc.shape[0] == n0:
                        cols = nm[i, :n0]
                        ok = (cols >= 0) & (cols < max_union)
                        if fld in enriched_fields:
                            if self._active_side is None:
                                masks_by_field[f"restrict_mask_{fld}_neg"][i, cols[ok]] = 1.0
                            else:
                                masks_by_field[f"restrict_mask_{fld}_neg"][i, cols[ok]] = np.asarray(mneg_loc, dtype=float)[ok]
                        else:
                            masks_by_field[f"restrict_mask_{fld}_neg"][i, cols[ok]] = np.asarray(mneg_loc, dtype=float)[ok]
                        if fld in enriched_fields and neg_is_cut:
                            # Need both side masks on the neg element to compute α(-)
                            pos_m_ne, neg_m_ne = _elem_masks(neg_eid)
                            pm_ne = pos_m_ne.get(fld)
                            nm_ne = neg_m_ne.get(fld)
                            if pm_ne is not None and nm_ne is not None:
                                a_neg = alpha_from_side_masks(pm_ne, nm_ne, side="-")
                                cols_enr = nm[i, n0 : 2 * n0]
                                ok2 = (cols_enr >= 0) & (cols_enr < max_union)
                                masks_by_field[f"restrict_mask_{fld}_neg"][i, cols_enr[ok2]] = np.asarray(a_neg, dtype=float)[ok2]

        # Expand r** basis/derivative tables to the XFEM element-union layout.
        basis_tables: Dict[str, np.ndarray] = {}
        for fld in base_me.field_names:
            base_sl = base_me.component_dof_slices[fld]
            cut_sl = cut_me.component_dof_slices[fld]
            n0 = int(base_sl.stop - base_sl.start)

            for side_tag in ("pos", "neg"):
                for key, arr in geo_base.items():
                    if not (isinstance(key, str) and key.endswith(f"_{fld}_{side_tag}") and key.startswith("r")):
                        continue
                    tab = np.asarray(arr)
                    if tab.size == 0:
                        continue
                    # tab shape: (nE,nQ,n_loc_base) or (nE,nQ,n_loc_base,k)
                    if tab.ndim == 3:
                        loc = tab[:, :, base_sl]
                        ext = np.zeros((loc.shape[0], loc.shape[1], n_loc_cut), dtype=tab.dtype)
                        ext[:, :, cut_sl.start : cut_sl.start + n0] = loc
                        if fld in enriched_fields:
                            ext[:, :, cut_sl.start + n0 : cut_sl.start + 2 * n0] = loc
                        basis_tables[key] = ext
                    elif tab.ndim == 4:
                        loc = tab[:, :, base_sl, :]
                        ext = np.zeros((loc.shape[0], loc.shape[1], n_loc_cut, loc.shape[3]), dtype=tab.dtype)
                        ext[:, :, cut_sl.start : cut_sl.start + n0, :] = loc
                        if fld in enriched_fields:
                            ext[:, :, cut_sl.start + n0 : cut_sl.start + 2 * n0, :] = loc
                        basis_tables[key] = ext

        # Assemble final output, reusing base geometry.
        out = dict(geo_base)
        out["gdofs_map"] = gdofs_map
        out["pos_map"] = pos_map
        out["neg_map"] = neg_map
        out.update(maps_by_field)
        out.update(masks_by_field)
        out.update(basis_tables)
        return out
