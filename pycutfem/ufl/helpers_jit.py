import numpy as np
from typing import Mapping, Tuple, Dict, Any, Sequence, Set
from pycutfem.integration import volume
import logging # Added for logging warnings
import os
import re
from hashlib import blake2b
import pycutfem.jit.symbols as symbols
from pycutfem.utils.bitset import BitSet, bitset_cache_token
from pycutfem.ufl.expressions import Restriction
from pycutfem.fem.transform import element_Hxi
from pycutfem.ufl.jit_parametrization import build_jit_parametrization



logger = logging.getLogger(__name__)

# Cache for reference-space tables to avoid rebuilding identical basis/grad
# stacks during JIT warm-up. Keys include element signature, quadrature order,
# target element count and derivative kind.
_REF_TABLE_CACHE: dict[tuple, np.ndarray] = {}


def _array_token(arr) -> str:
    carr = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    h = blake2b(digest_size=16)
    shape_info = np.asarray(carr.shape, dtype=np.int64)
    h.update(shape_info.tobytes())
    h.update(carr.tobytes())
    return h.hexdigest()


def _pad_coeffs(coeffs, phi, ctx):
    """Return coeffs padded to the length of `phi` on ghost edges."""
    if phi.shape[0] == coeffs.shape[0] or "global_dofs" not in ctx:
        return coeffs                       # interior element – nothing to do

    padded = np.zeros_like(phi)
    side   = '+' if ctx.get("phi_val", 0.0) >= 0 else '-'
    amap   = ctx["pos_map"] if side == '+' else ctx["neg_map"]
    padded[amap] = coeffs
    return padded

def _find_all_bitsets(expr):
    """
    Collect *all* distinct BitSet objects that occur anywhere in the
    expression graph – no matter how deeply nested.

    Uses a generic DFS over ``expr.__dict__`` instead of a fixed list of
    attribute names, so it automatically follows any new node types
    (Transpose, Derivative, Side, …) you might add in the future.
    """
    from pycutfem.ufl.expressions import Restriction, Expression
    bitsets   = set()
    seen      = set()          # guard against cycles
    stack     = [expr]

    while stack:
        node = stack.pop()
        nid  = id(node)
        if nid in seen:
            continue
        seen.add(nid)

        # Record BitSet carried by a Restriction node
        if isinstance(node, Restriction):
            bitsets.add(node.domain)

        # ---- generic child traversal ----------------------------------
        for child in node.__dict__.values():
            if isinstance(child, (list, tuple)):
                stack.extend(c for c in child if isinstance(c, Expression))
            elif isinstance(child, Expression):
                stack.append(child)

    return list(bitsets)

# Helper: select the first present (non-None) value by key, without triggering
def _first_present(d: dict, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _build_jit_kernel_args(       # ← signature unchanged
    ir,                           # linear IR produced by the visitor
    expression,                   # the UFL integrand
    mixed_element,                # MixedElement instance
    q_order: int,                 # quadrature order
    dof_handler,                  # DofHandler – *needed* for padding
    gdofs_map: np.ndarray = None, # global DOF map (mixed-space)
    param_order=None,             # order of parameters in the JIT kernel
    pre_built: dict | None = None
):
    """
    Return a **dict { name -> ndarray }** with *all* reference-space tables
    and coefficient arrays the kernel lists in `param_order`
    (or – if `param_order` is None – in the IR).

    Guarantees that names such as ``b_p``, ``g_ux``, ``d20_uy`` *always*
    appear when requested.

    Notes
    -----
    * The *mixed-space* element → global DOF map **(`gdofs_map`)** and
      `node_coords` are **NOT** created here because the surrounding
      assembler already provides them (and they do not depend on the IR).
    * Set ``PYCUTFEM_JIT_DEBUG=1`` to get a short print-out of every array
      built (name, shape, dtype) – useful for verification.
    """
    import os, re, numpy as np
    from typing import Dict, Any
    from pycutfem.integration.quadrature import volume
    from pycutfem.ufl.expressions import (
        Function, VectorFunction, Constant as UflConst, Grad, ElementWiseConstant
    )
    from pycutfem.jit.ir import LoadVariable, LoadConstantArray
    from pycutfem.ufl.analytic import Analytic
    from pycutfem.ufl.helpers import _find_all
    from pycutfem.fem import transform

  

    logger = __import__("logging").getLogger(__name__)
    dbg    = os.getenv("PYCUTFEM_JIT_DEBUG", "").lower() in {"1", "true", "yes"}
    param_map = build_jit_parametrization(expression)

    # ------------------------------------------------------------------
    # 0. Helpers
    # ------------------------------------------------------------------
    # Target element count for per-element tables (respect subsets when provided)
    n_elem = mixed_element.mesh.n_elements
    if gdofs_map is not None:
        try:
            n_elem = int(np.asarray(gdofs_map).shape[0])
        except Exception:
            pass
    elif pre_built is not None:
        for _key in ("qp_phys", "qw", "eids", "gdofs_map"):
            arr = pre_built.get(_key)
            if arr is None:
                continue
            try:
                n_elem = int(np.asarray(arr).shape[0])
                break
            except Exception:
                continue

    # full-length element size for CellDiameter() lookups via owner ids
    _h_global = np.asarray(
        [mixed_element.mesh.element_char_length(i) for i in range(mixed_element.mesh.n_elements)],
        dtype=np.float64
    )

    # Active fields & union mask for this kernel
    def _active_field_order():
        seen = set(); order = []
        for op in ir:
            if hasattr(op, "field_names"):
                for f in getattr(op, "field_names", []) or []:
                    if f in seen or f not in mixed_element.field_names:
                        continue
                    seen.add(f); order.append(f)
        if not order:
            order = list(mixed_element.field_names)
        return order

    _active_fields = _active_field_order()
    _active_cols = np.concatenate(
        [np.arange(mixed_element.component_dof_slices[f].start,
                   mixed_element.component_dof_slices[f].stop, dtype=np.int32)
         for f in _active_fields]
    ) if _active_fields else np.arange(mixed_element.n_dofs_local, dtype=np.int32)
    _active_n = int(len(_active_cols))
    def _cached_ref_table(kind: str, field: str, builder, deriv: tuple[int, int] | None = None):
        key = (mixed_element.signature(), q_order, n_elem, tuple(_active_cols.tolist()), kind, field, deriv)
        hit = _REF_TABLE_CACHE.get(key)
        if hit is not None:
            return hit
        arr = builder()
        _REF_TABLE_CACHE[key] = arr
        return arr

    def _expand_per_element(ref_tab: np.ndarray) -> np.ndarray:
        """
        Replicate a reference-space table so that shape[0] == n_elem.
        """
        return np.broadcast_to(ref_tab, (n_elem, *ref_tab.shape)).copy()

    # Reference-element quadrature (ξ-space) for table builders
    qp_ref, _ = volume(mixed_element.mesh.element_type, q_order)

    def _basis_table(field: str):
        def _build():
            ref = np.asarray(
                [mixed_element.basis(field, *xi_eta) for xi_eta in qp_ref],
                dtype=np.float64,
            )  # (n_q , n_loc)
            return _expand_per_element(ref)  # (n_elem , n_q , n_loc)
        return _cached_ref_table("basis", field, _build)

    def _grad_table(field: str):
        def _build():
            ref = np.asarray(
                [mixed_element.grad_basis(field, *xi_eta) for xi_eta in qp_ref],
                dtype=np.float64,
            )  # (n_q , n_loc , 2)
            return np.ascontiguousarray(_expand_per_element(ref))  # (n_elem , n_q , n_loc , 2)
        return _cached_ref_table("grad", field, _build)

    def _deriv_table(field: str, ax: int, ay: int):
        """∂^{ax+ay} φ_i / ∂ξ^{ax} ∂η^{ay}"""
        def _build():
            ref = np.asarray(
                [mixed_element.deriv_ref(field, *xi_eta, ax, ay) for xi_eta in qp_ref],
                dtype=np.float64,
            )
            return _expand_per_element(ref)
        return _cached_ref_table("deriv", field, _build, deriv=(ax, ay))
    # NEW: evaluate at interface physical quadrature points (qp_phys)
    def _n_union_for_eid(eid: int) -> int:
        return len(dof_handler.get_elemental_dofs(int(eid)))

    def _union_size_from_prebuilt() -> int:
        gmap = pre_built.get("gdofs_map") if pre_built is not None else None
        if isinstance(gmap, np.ndarray) and gmap.ndim == 2:
            return int(gmap.shape[1])
        return int(getattr(mixed_element, "n_dofs_per_elem", getattr(mixed_element, "n_dofs_local", 0)))

    def _basis_table_phys_union(field: str) -> np.ndarray:
        """Union-length basis at interface physical QPs."""
        key = f"b_{field}"
        if pre_built is not None and key in pre_built:
            return np.asarray(pre_built[key], dtype=np.float64)

        pts  = pre_built["qp_phys"]      # (nE, nQ, 2)
        eids = pre_built["eids"]         # (nE,)
        me   = mixed_element
        mesh = me.mesh
        nE, nQ, _ = pts.shape
        if nE == 0:
            n_union = _union_size_from_prebuilt()
            return np.empty((0, nQ, n_union), dtype=np.float64)
        n_union = _n_union_for_eid(eids[0])
        tab = np.empty((nE, nQ, n_union), dtype=np.float64)

        # Prefer cached reference coordinates if available to avoid inverse mapping
        qref_raw = pre_built.get("qref")
        if qref_raw is None:
            qref_raw = pre_built.get("qp_ref")
        qref_mode = None
        if qref_raw is not None:
            qref_arr = np.asarray(qref_raw, dtype=np.float64)
            if qref_arr.ndim == 3 and qref_arr.shape[2] == 2:
                qref_mode = ("per_element", qref_arr)
            elif qref_arr.ndim == 2 and qref_arr.shape[1] == 2:
                qref_mode = ("global", qref_arr)
            else:
                qref_mode = None
        else:
            qref_mode = None

        for i in range(nE):
            eid = int(eids[i])
            for q in range(nQ):
                if qref_mode is not None:
                    mode, arr = qref_mode
                    if mode == "per_element":
                        xi, eta = arr[i, q]
                    else:
                        xi, eta = arr[q]
                else:
                    x, y = pts[i, q]
                    xi, eta = transform.inverse_mapping(mesh, eid, np.array([x, y]))
                tab[i, q, :] = me.basis(field, float(xi), float(eta))
        return np.ascontiguousarray(tab)

    def _deriv_table_phys_union(field: str, ax: int, ay: int) -> np.ndarray:
        """Union-length derivative tables at interface physical QPs."""
        key = f"d{ax}{ay}_{field}"
        if pre_built is not None and key in pre_built:
            return np.asarray(pre_built[key], dtype=np.float64)

        pts  = pre_built["qp_phys"]      # (nE, nQ, 2)
        eids = pre_built["eids"]         # (nE,)
        me   = mixed_element
        mesh = me.mesh
        nE, nQ, _ = pts.shape
        if nE == 0:
            n_union = _union_size_from_prebuilt()
            return np.empty((0, nQ, n_union), dtype=np.float64)
        n_union = _n_union_for_eid(eids[0])
        tab = np.empty((nE, nQ, n_union), dtype=np.float64)

        qref_raw = pre_built.get("qref")
        if qref_raw is None:
            qref_raw = pre_built.get("qp_ref")
        qref_mode = None
        if qref_raw is not None:
            qref_arr = np.asarray(qref_raw, dtype=np.float64)
            if qref_arr.ndim == 3 and qref_arr.shape[2] == 2:
                qref_mode = ("per_element", qref_arr)
            elif qref_arr.ndim == 2 and qref_arr.shape[1] == 2:
                qref_mode = ("global", qref_arr)
            else:
                qref_mode = None

        for i in range(nE):
            eid = int(eids[i])
            for q in range(nQ):
                if qref_mode is not None:
                    mode, arr = qref_mode
                    if mode == "per_element":
                        xi, eta = arr[i, q]
                    else:
                        xi, eta = arr[q]
                else:
                    x, y = pts[i, q]
                    xi, eta = transform.inverse_mapping(mesh, eid, np.array([x, y]))
                tab[i, q, :] = me.deriv_ref(field, float(xi), float(eta), int(ax), int(ay))
        return tab

    def _grad_table_phys_union(field: str) -> np.ndarray:
        """Union-length gradient tables at interface physical QPs."""
        key = f"g_{field}"
        if pre_built is not None and key in pre_built:
            return np.ascontiguousarray(np.asarray(pre_built[key], dtype=np.float64))

        pts  = pre_built["qp_phys"]      # (nE, nQ, 2)
        eids = pre_built["eids"]         # (nE,)
        me   = mixed_element
        mesh = me.mesh
        nE, nQ, _ = pts.shape
        if nE == 0:
            n_union = _union_size_from_prebuilt()
            return np.empty((0, nQ, n_union, 2), dtype=np.float64)
        n_union = _n_union_for_eid(eids[0])
        tab = np.empty((nE, nQ, n_union, 2), dtype=np.float64)

        qref_raw = pre_built.get("qref")
        if qref_raw is None:
            qref_raw = pre_built.get("qp_ref")
        qref_mode = None
        if qref_raw is not None:
            qref_arr = np.asarray(qref_raw, dtype=np.float64)
            if qref_arr.ndim == 3 and qref_arr.shape[2] == 2:
                qref_mode = ("per_element", qref_arr)
            elif qref_arr.ndim == 2 and qref_arr.shape[1] == 2:
                qref_mode = ("global", qref_arr)
            else:
                qref_mode = None

        for i in range(nE):
            eid = int(eids[i])
            for q in range(nQ):
                if qref_mode is not None:
                    mode, arr = qref_mode
                    if mode == "per_element":
                        xi, eta = arr[i, q]
                    else:
                        xi, eta = arr[q]
                else:
                    x, y = pts[i, q]
                    xi, eta = transform.inverse_mapping(mesh, eid, np.array([x, y]))
                tab[i, q, :, :] = me.grad_basis(field, float(xi), float(eta))
        return np.ascontiguousarray(tab)
    def _decode_coeff_name(name: str):
        """
        Resolve u_<mid>_loc into (field_key, side).

        Prefer exact function name matches in func_map (e.g., 'velocity_neg').
        Only if no exact match exists, treat a trailing _pos/_neg or __pos/__neg
        as a side suffix and strip it, returning ('velocity', 'pos'|'neg').
        """
        assert name.startswith("u_") and name.endswith("_loc")
        mid = name[2:-4]  # strip leading 'u_' and trailing '_loc'

        # 1) Exact match first: supports distinct objects named 'velocity_pos'/'velocity_neg'
        if mid in func_map:
            return mid, None

        # 2) Recognize both single-underscore and symbols-based suffixes
        pos_suf = getattr(symbols, "POS_SUFFIX", "__pos")
        neg_suf = getattr(symbols, "NEG_SUFFIX", "__neg")

        for suf, side in ((pos_suf, "pos"), (neg_suf, "neg"), ("_pos", "pos"), ("_neg", "neg")):
            if mid.endswith(suf):
                base = mid[:-len(suf)]
                # prefer a real function/object if it exists
                if base in func_map:
                    return base, side
                # fallback: report stripped base even if not present (caller will error)
                return base, side

        # 3) No side suffix at all
        return mid, None



    # ------------------------------------------------------------------
    # 1. Map Function / VectorFunction names  →  objects
    # ------------------------------------------------------------------
    func_map: Dict[str, Any] = {}

    for f in _find_all(expression, Function):
        func_map[f.name] = f
        func_map.setdefault(f.field_name, f)

    for vf in _find_all(expression, VectorFunction):
        func_map[vf.name] = vf
        for comp, fld in zip(vf.components, vf.field_names):
            func_map.setdefault(fld, vf)

    # make sure every component points back to its parent VectorFunction
    for f in list(func_map.values()):
        pv = getattr(f, "_parent_vector", None)
        if pv is not None and hasattr(pv, "name"):
            func_map.setdefault(pv.name, pv)

    # ------------------------------------------------------------------
    # 2. Collect EVERY parameter the kernel will expect
    # ------------------------------------------------------------------
    required: set[str] = set(param_order) if param_order is not None else set()

    # If `param_order` was not given collect names from the IR too
    if param_order is None:
        for op in ir:
            if isinstance(op, LoadVariable):
                fnames = (op.field_names if op.is_vector and op.field_names
                          else [op.name])
                for fld in fnames:
                    if op.deriv_order == (0, 0):
                        required.add(f"b_{fld}")
                    else:
                        d0, d1 = op.deriv_order
                        required.add(f"d{d0}{d1}_{fld}")
                    if op.role == "function":
                        required.add(f"u_{op.name}_loc")

            elif isinstance(op, LoadConstantArray):
                required.add(op.name)

        for g in _find_all(expression, Grad):
            op = g.operand
            if hasattr(op, "field_names"):
                required.update(f"g_{f}" for f in op.field_names)
            else:
                required.add(f"g_{op.field_name}")

    # ------------------------------------------------------------------
    # 3. Start with pre_built arrays (geometry, facet maps, etc.)
    # ------------------------------------------------------------------
    if pre_built is None:
        pre_built = {}

    # Allow base keys, plus exactly what this kernel says it needs in `param_order`.
    base_allow = {
        # geometry / meta (unsided)
        "gdofs_map", "node_coords", "element_nodes",
        "qp_phys", "qref", "qp_ref", "qw", "detJ", "J_inv", "normals", "phis",
        "eids", "entity_kind",
        "owner_id", "owner_pos_id", "owner_neg_id", "pos_eids", "neg_eids",
        "pos_map", "neg_map",

        # Hessian-of-inverse-map tensors (volume / boundary / interface)
        "Hxi0", "Hxi1",
        "Txi0", "Txi1",           # 3rd order
        "Qxi0", "Qxi1",           # 4th order

        # sided Hessian tensors for ghost facets (and similar)
        "pos_Hxi0", "pos_Hxi1",
        "neg_Hxi0", "neg_Hxi1",
        "pos_Txi0", "pos_Txi1",  "neg_Txi0", "neg_Txi1",
        "pos_Qxi0", "pos_Qxi1",  "neg_Qxi0", "neg_Qxi1",
    }
    needed = set(param_order) if param_order is not None else set()

    # Side keys can appear under many names; keep only the ones actually present/needed.
    side_keys_present = {k for k in pre_built.keys() if k.startswith(("pos_", "neg_"))}

    pre_built = {
        k: v for k, v in pre_built.items()
        if (k in base_allow)
        or (k in needed)
        or (k in side_keys_present)
        or k.startswith(("b_", "g_", "d", "r", "restrict_mask_"))
    }
    args: Dict[str, Any] = dict(pre_built)

    # OPTIONAL alias: if a sided Hxi was requested but missing, fall back to unsided if available.
    for _side in ("pos", "neg"):
        for _t in ("Hxi0", "Hxi1", "Txi0", "Txi1", "Qxi0", "Qxi1"):
            _k = f"{_side}_{_t}"
            if (_k in needed) and (_k not in args) and (_t in pre_built):
                args[_k] = pre_built[_t]
    
    # ------------------------------------------------------------------
    # 3b. Build per-field side maps *exactly* as requested by the kernel.
    #     Requested names are of the form:  (pos|neg)_map_<fld>
    #     where <fld> MUST be in dof_handler.field_names.
    # ------------------------------------------------------------------
    if needed:
        _re_side_map = re.compile(r"^(pos|neg)_map_(.+)$")
        # Which fields were requested?
        requested_fields: Dict[str, set[str]] = {"pos": set(), "neg": set()}
        for name in needed:
            m = _re_side_map.match(name)
            if not m:
                continue
            side, fld = m.group(1), m.group(2)
            # Already available? skip
            if name in args:
                continue
            requested_fields[side].add(fld)

        # If none requested, nothing to do.
        if requested_fields["pos"] or requested_fields["neg"]:
            # Select gdofs_map explicitly: prefer function arg, else from args
            gmap = gdofs_map if gdofs_map is not None else args.get("gdofs_map")
            if gmap is None:
                raise KeyError("_build_jit_kernel_args: gdofs_map is required to synthesize per-field side maps.")
            gdofs = np.asarray(gmap)

            # Resolve side -> element ids (interface: owner_*_id; ghost: *_eids; fallbacks)
            pos_ids = _first_present(args, "owner_pos_id", "pos_eids", "owner_id", "eids")
            neg_ids = _first_present(args, "owner_neg_id", "neg_eids", "owner_id", "eids")
             
            # Validate requested field names strictly against DofHandler
            valid_fields = set(dof_handler.field_names)
            for side, fld_set in requested_fields.items():
                if not fld_set:
                    continue
                missing = sorted([f for f in fld_set if f not in valid_fields])
                if missing:
                    raise KeyError(
                        "_build_jit_kernel_args: kernel requested side maps for unknown fields: "
                        f"{missing}. Available fields: {sorted(valid_fields)}"
                    )
                side_ids = pos_ids if side == "pos" else neg_ids
                if side_ids is None:
                    raise KeyError(
                        f"_build_jit_kernel_args: cannot build '{side}_map_<fld>' — "
                        f"no side element-ids found (looked for owner_{side}_id / {side}_eids / owner_id / eids)."
                    )
                side_ids = np.asarray(side_ids, dtype=np.int32)
                nE, n_union = gdofs.shape[0], gdofs.shape[1]

                # Build each requested map for this side
                for fld in sorted(fld_set):
                    # Element-based kernels (e.g. dInterface) use the union-local
                    # layout of the element itself. In that case the per-field
                    # side maps are purely index maps into the union vector and
                    # must follow the MixedElement component slices (XFEM needs
                    # this to include enriched DOFs on the interface).
                    entity_kind = args.get("entity_kind", None)
                    if entity_kind == "element":
                        sl = getattr(mixed_element, "component_dof_slices", {}).get(fld, None)
                        if sl is None:
                            raise KeyError(
                                f"_build_jit_kernel_args: cannot build '{side}_map_{fld}': "
                                f"missing component_dof_slices for field '{fld}'."
                            )
                        if int(sl.stop) > int(n_union):
                            raise ValueError(
                                f"_build_jit_kernel_args: cannot build '{side}_map_{fld}': "
                                f"field slice {sl} exceeds union width n_union={n_union}."
                            )
                        row = np.arange(int(sl.start), int(sl.stop), dtype=np.int32)
                        args[f"{side}_map_{fld}"] = np.tile(row, (nE, 1))
                        continue

                    if nE == 0 or side_ids.size == 0:
                        # Empty entity set: still provide correctly-shaped placeholders so
                        # kernel signatures stay consistent, even though the kernel will not run.
                        sl = getattr(mixed_element, "component_dof_slices", {}).get(fld, None)
                        if sl is not None:
                            nloc_f = int(sl.stop - sl.start)
                        else:
                            nloc_f = int(getattr(getattr(dof_handler, "mixed_element", None), "_n_basis", {}).get(fld, 0))
                        args[f"{side}_map_{fld}"] = -np.ones((nE, nloc_f), dtype=np.int32)
                        continue
                    # element-local gdofs for this field on each owner element of the side
                    loc_lists = dof_handler.element_maps[fld]
                    # all elements share same nloc for that field
                    nloc_f = len(loc_lists[int(side_ids[0])])
                    m_arr = -np.ones((nE, nloc_f), dtype=np.int32)
                    for i in range(nE):
                        eid = int(side_ids[i])
                        union_row = gdofs[i, :n_union]
                        # map: global dof id -> union column
                        col_of = {int(d): j for j, d in enumerate(union_row)}
                        local_gdofs_f = loc_lists[eid]
                        for j, d in enumerate(local_gdofs_f):
                            try:
                                m_arr[i, j] = col_of[int(d)]
                            except KeyError:
                                # If a field was dropped from active_cols, mark as padding (-1)
                                m_arr[i, j] = -1
                    missing_mask = m_arr < 0
                    if missing_mask.any():
                        logger.debug(
                            "Side map for field '%s' missed %d entries (likely pruned by active_cols); padding with -1.",
                            fld, int(missing_mask.sum()),
                        )
                    args[f"{side}_map_{fld}"] = m_arr

    # ------------------------------------------------------------------
    # 4. BitSets present in the expression → full-length element masks
    # ------------------------------------------------------------------
    n_elems = mixed_element.mesh.n_elements
    eids = None
    if "eids" in args and args["eids"] is not None:
        eids = np.asarray(args["eids"], dtype=np.int32)

    def _expand_subset_to_full(arr: np.ndarray, what: str) -> np.ndarray:
        """
        Ensure per-element arrays (bitsets, EWC values, etc.) are FULL element-length.
        If 'arr' has shape[0] == len(eids), expand it into a length-n_elems array.
        Otherwise if shape[0] == n_elems, return as-is.
        Otherwise, raise with a helpful message.
        """
        a = np.asarray(arr)
        if a.ndim == 0:
            return a  # scalar
        if a.shape[0] == n_elems:
            return a
        if eids is not None and a.shape[0] == eids.shape[0]:
            out = np.zeros((n_elems,) + a.shape[1:], dtype=a.dtype)
            out[eids] = a
            return out
        raise ValueError(
            f"{what}: got length {a.shape[0]} but expected either n_elems={n_elems} "
            f"or len(eids)={0 if eids is None else int(eids.shape[0])}."
        )

    _re_flag = re.compile(r"^domain_flag_(.+?)(?:_(pos|neg))?$")
    requested_flags: dict[str, set[str]] = {}
    for name in required:
        m = _re_flag.match(name)
        if not m:
            continue
        token, side = m.group(1), m.group(2) or ""
        requested_flags.setdefault(token, set()).add(side)
    requested_bs = {n.split("domain_bs_", 1)[1] for n in required if n.startswith("domain_bs_")}

    for token, dom in param_map.domain_by_token.items():
        pname = f"domain_bs_{token}"
        need_flag = token in requested_flags
        need_bs = token in requested_bs
        if not (need_flag or need_bs):
            continue
        raw = getattr(dom, "array", dom)
        mask_full = _expand_subset_to_full(
            np.asarray(raw, dtype=np.bool_).ravel(), what=f"BitSet {pname}"
        )
        if need_bs:
            args[pname] = mask_full
        if need_flag:
            for side in sorted(requested_flags.get(token, {""})):
                if side == "pos":
                    owners = _first_present(args, "owner_pos_id", "pos_eids", "owner_id", "eids")
                elif side == "neg":
                    owners = _first_present(args, "owner_neg_id", "neg_eids", "owner_id", "eids")
                else:
                    owners = args.get("owner_id")
                    if owners is None:
                        owners = args.get("eids")
                if owners is None:
                    owners = np.arange(mask_full.shape[0], dtype=np.int32)
                owners = np.asarray(owners, dtype=np.int32)
                if mask_full.ndim:
                    flag = np.zeros(owners.shape, dtype=np.bool_)
                    valid = owners >= 0
                    if np.any(valid):
                        flag[valid] = mask_full[owners[valid]]
                else:
                    flag = np.full(owners.shape, bool(mask_full), dtype=np.bool_)
                n_q = 1
                qw_arr = args.get("qw", None)
                if isinstance(qw_arr, np.ndarray) and qw_arr.ndim >= 2:
                    n_q = qw_arr.shape[1]
                if flag.ndim == 1:
                    flag = flag.reshape(flag.shape[0], 1)
                if n_q > flag.shape[1]:
                    flag = np.broadcast_to(flag, (flag.shape[0], n_q)).copy()
                suffix = f"_{side}" if side else ""
                args[f"domain_flag_{token}{suffix}"] = flag

    # ------------------------------------------------------------------
    # 5. Constants / EWC / coefficient vectors / reference tables
    # ------------------------------------------------------------------
    const_arrays: dict[str, UflConst] = dict(param_map.const_by_name)
    ewc_arrays: dict[str, ElementWiseConstant] = dict(param_map.ewc_by_name)

    # cache gdofs_map for coefficient gathering
    if gdofs_map is None:
        gdofs_map = np.vstack([
            np.asarray(dof_handler.get_elemental_dofs(eid), dtype=np.int32)[_active_cols]
            for eid in range(mixed_element.mesh.n_elements)
        ]).astype(np.int32)

    total_dofs = dof_handler.total_dofs

    for name in sorted(required):
        # Always supply global element-length h_arr for CellDiameter
        if name == "h_arr":
            args["h_arr"] = _h_global
            continue

        # --- sided reference value/derivative tables: r{d0}{d1}_{field}_{pos|neg} ---
        m_r = re.match(r"^r(\d)(\d)_(.+)_(pos|neg)$", name)
        if m_r:
            d0_s, d1_s, fld, side = m_r.groups()
            d0, d1 = int(d0_s), int(d1_s)

            # tiny helpers
            def _is_ghost(pb):
                # Ghost precompute exports explicit side maps (pos_map/neg_map or per-field variants).
                # Boundary-edge precompute also stores owner ids for convenience, so *do not*
                # classify by owner_* keys alone.
                if pb is None:
                    return False
                return any(
                    k == "pos_map"
                    or k == "neg_map"
                    or k.startswith("pos_map_")
                    or k.startswith("neg_map_")
                    for k in pb.keys()
                )

            def _pad_owner_to_union(tab: np.ndarray, side: str) -> np.ndarray:
                """
                tab: (nE, nQ, L_owner_mixed). Returns (nE, nQ, n_union_ghost),
                using pos_map/neg_map that map owner-mixed dofs -> ghost-union dofs.
                """
                nE, nQ, L = tab.shape
                n_union = gdofs_map.shape[1]             # ghost union length
                out = np.zeros((nE, nQ, n_union), dtype=tab.dtype)
                side_map_key = "pos_map" if side == "pos" else "neg_map"
                owner2union = pre_built[side_map_key]    # shape (nE, L)
                # vectorized over q
                for e in range(nE):
                    m = owner2union[e]                   # (L,)
                    # keep only valid targets
                    valid = (m >= 0) & (m < n_union)
                    out[e, :, m[valid]] = tab[e, :, valid]
                return out

            # 1) If precompute already provided it, prefer that.
            if pre_built is not None and name in pre_built:
                tab = pre_built[name]
                # GHOST: owner-mixed (22) -> ghost-union (36) pre-pad once here
                if _is_ghost(pre_built):
                    n_union = gdofs_map.shape[1]
                    if tab.shape[2] != n_union:
                        tab = _pad_owner_to_union(tab, side)
                args[name] = tab
                continue

            # 2) INTERFACE fallback (entity_kind == 'element'):
            #    Build at interface physical QPs. Return *owner-mixed* length
            #    so the kernel’s fixed block slices [0:9],[9:18],[18:22] stay valid.
            if pre_built is not None and pre_built.get("entity_kind") == "element":
                # r00/rXY tables are used for sided operations (Pos/Neg/Jump) on dInterface.
                #
                # IMPORTANT: For element-based interface kernels (Γ inside a cut element),
                # the underlying CG basis is side-agnostic. The "side" here selects which
                # *trace* is being evaluated, but must NOT zero out a subset of DOFs.
                # Doing so breaks core identities such as grad(x^2) == grad(x^2-1) and
                # causes JIT/Python parity failures in Jump/Pos/Neg tests.
                #
                # If a restricted (one-sided) space is desired, it must be expressed
                # explicitly (e.g. via Restriction(...) or a dedicated restricted space),
                # not by implicitly masking r** tables here.
                if d0 == 0 and d1 == 0:
                    tab = _basis_table_phys_union(fld)
                else:
                    dkey = f"d{d0}{d1}_{fld}"
                    if dkey in args:
                        tab = args[dkey]
                    elif pre_built is not None and dkey in pre_built:
                        tab = pre_built[dkey]
                    else:
                        tab = _deriv_table_phys_union(fld, d0, d1)

                args[name] = tab
                continue

            # 3) Otherwise (GHOST with no precompute): must be provided via metadata['derivs']
            if _is_ghost(pre_built):
                raise KeyError(
                    f"_build_jit_kernel_args: kernel requests '{name}', but it wasn't provided. "
                    "For ghost integrals, include metadata['derivs'] (e.g. {(1,0),(0,1)}) so r10/r01 are emitted."
                )
            # If not ghost and not interface (shouldn’t happen), fail clearly
            raise KeyError(f"_build_jit_kernel_args: kernel requests '{name}', but it wasn't provided.")

        if name in args:
            continue




        # ---- basis tables ------------------------------------------------
        if name.startswith("b_"):
            fld = name[2:]
            if pre_built is not None and pre_built.get("entity_kind") == "element" and "qp_phys" in pre_built:
                args[name] = _basis_table_phys_union(fld)
            else:
                args[name] = _basis_table(fld)

        # ---- gradient tables ---------------------------------------------
        elif name.startswith("g_"):
            fld = name[2:]
            if pre_built is not None and pre_built.get("entity_kind") == "element" and "qp_phys" in pre_built:
                args[name] = _grad_table_phys_union(fld)
            else:
                args[name] = _grad_table(fld)

        # ---- higher-order ξ-derivatives ----------------------------------
        elif re.match(r"d\d\d_", name):
            tag, fld = name.split("_", 1)
            ax, ay = int(tag[1]), int(tag[2])
            if pre_built is not None and pre_built.get("entity_kind") == "element" and "qp_phys" in pre_built:
                args[name] = _deriv_table_phys_union(fld, ax, ay)
            else:
                args[name] = _deriv_table(fld, ax, ay)

        # ---- constant arrays ---------------------------------------------
        elif name in const_arrays:
            vals = np.asarray(const_arrays[name].value, dtype=np.float64)
            args[name] = vals

        # ---- element-wise constants --------------------------------------
        elif name in ewc_arrays:
            ewc = ewc_arrays[name]
            arr = _expand_subset_to_full(
                np.asarray(ewc.values, dtype=np.float64), what=f"EWC {name}"
            )
            args[name] = arr

        # ---- analytic expressions ----------------------------------------
        elif name.startswith("ana_"):
            func_id = int(name.split("_", 1)[1])
            try:
                ana = param_map.analytic_by_id[func_id]
            except KeyError as exc:
                raise KeyError(
                    f"_build_jit_kernel_args: kernel requested analytic '{name}', "
                    "but no matching Analytic node was found in the expression."
                ) from exc

            qp_phys = args["qp_phys"]                 # (n_elem, n_qp, 2)
            n_elem_, n_qp, _ = qp_phys.shape
            tshape = getattr(ana, "tensor_shape", ())

            if tshape == () or tshape is None:
                ana_vals = np.empty((n_elem_, n_qp), dtype=np.float64)
            else:
                ana_vals = np.empty((n_elem_, n_qp) + tuple(tshape), dtype=np.float64)

            ana_vals[...] = ana.eval(qp_phys)         # vectorised over (e,q)
            args[name] = ana_vals

        # ---- coefficient vectors (gather) --------------------------------
        elif (name.startswith("u_") and name.endswith("_loc")):
            # Live coefficient blocks are injected by KernelRunner; skip static gather.
            continue

    # ------------------------------------------------------------------
    # 6. Optional debug print
    # ------------------------------------------------------------------
    if dbg:
        print("[build_jit_kernel_args] built:")
        for k, v in args.items():
            if isinstance(v, np.ndarray):
                print(f"    {k:<20} shape={v.shape} dtype={v.dtype}")
            else:
                print(f"    {k:<20} type={type(v).__name__}")

    return args



#----------------------------------------------------------------------
# Generic scatter for element-based JIT kernels (e.g., dInterface)
#----------------------------------------------------------------------
def _scatter_element_contribs(
    K_elem: np.ndarray | None,
    F_elem: np.ndarray | None,
    J_elem: np.ndarray | None,
    element_ids: np.ndarray,
    gdofs_map: np.ndarray,
    matvec: np.ndarray,
    ctx: dict,
    integrand,
    hook,
):
    """
    Generic scatter for element-based JIT kernels (e.g., dInterface).
    """
    rhs = ctx.get("rhs", False)
    # hook = ctx.get("hooks", {}).get(type(integrand))

    # --- Matrix contributions ---
    if not rhs and K_elem is not None and K_elem.ndim == 3:
        for i, eid in enumerate(element_ids):
            gdofs = gdofs_map[i]
            r, c = np.meshgrid(gdofs, gdofs, indexing="ij")
            matvec[r, c] += K_elem[i]

    # --- Vector contributions ---
    if rhs and F_elem is not None:
        for i, eid in enumerate(element_ids):
            gdofs = gdofs_map[i]
            np.add.at(matvec, gdofs, F_elem[i])

    # --- Functional contributions ---
    if hook and J_elem is not None:
        # print(f"J_elem.shape: {J_elem.shape}---J_elem: {J_elem}")
        total = J_elem.sum(axis=0) if J_elem.ndim > 1 else J_elem.sum()
        # print(f"total.shape: {total.shape}---total: {total}")
        # This accumulator logic is correct.
        acc = ctx.setdefault("scalar_results", {}).setdefault(
            hook["name"], np.zeros_like(total)
        )
        acc += total


def _stack_ragged(chunks: Sequence[np.ndarray]) -> np.ndarray:
    """Stack 1‑D integer arrays of variable length → 2‑D, padded with ‑1."""
    n = len(chunks)
    m = max(len(c) for c in chunks)
    out = -np.ones((n, m), dtype=np.int32)
    for i, c in enumerate(chunks):
        out[i, : len(c)] = c
    return out
