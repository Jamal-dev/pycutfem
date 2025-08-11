import numpy as np
from typing import Mapping, Tuple, Dict, Any, Sequence
from pycutfem.integration import volume
import logging # Added for logging warnings
import os
import re
import pycutfem.jit.symbols as symbols
from pycutfem.utils.bitset import BitSet
from pycutfem.ufl.expressions import Restriction


logger = logging.getLogger(__name__)


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


def _build_jit_kernel_args(       # ← NEW signature (unchanged)
        ir,                       # linear IR produced by the visitor
        expression,               # the UFL integrand
        mixed_element,            # MixedElement instance
        q_order: int,             # quadrature order
        dof_handler,              # DofHandler – *needed* for padding
        gdofs_map: np.ndarray=None,  # global DOF map (mixed-space)
        param_order=None,          # order of parameters in the JIT kernel
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

    import logging

    from pycutfem.ufl.expressions import (
        Function, VectorFunction, Constant as UflConst, Grad, ElementWiseConstant
    )
    from pycutfem.jit.ir import LoadVariable, LoadConstantArray, LoadElementWiseConstant
    from pycutfem.ufl.analytic import Analytic
    from pycutfem.ufl.helpers import _find_all
    

    logger = logging.getLogger(__name__)
    dbg    = os.getenv("PYCUTFEM_JIT_DEBUG", "").lower() in {"1", "true", "yes"}

    # ------------------------------------------------------------------
    # 0. Helpers
    # ------------------------------------------------------------------
    n_elem = mixed_element.mesh.n_elements
    # full-length element size for CellDiameter() lookups via owner ids
    _h_global = np.asarray(
        [mixed_element.mesh.element_char_length(i) for i in range(n_elem)],
        dtype=np.float64
    )


    def _expand_per_element(ref_tab: np.ndarray) -> np.ndarray:
        """
        Replicate a reference-space table so that shape[0] == n_elem.

        We keep memory overhead low with ``np.broadcast_to``; the
        result is read-only, which is fine because kernels never write
        into these tables.
        """
        return np.broadcast_to(ref_tab, (n_elem, *ref_tab.shape)).copy()

    
    def _basis_table(field: str):
        ref = np.asarray([mixed_element.basis(field, *xi_eta)
                          for xi_eta in qp_ref], dtype=np.float64)      # (n_q , n_loc)
        return _expand_per_element(ref)                                 # (n_elem , n_q , n_loc)

    def _grad_table(field: str):
        ref = np.asarray([mixed_element.grad_basis(field, *xi_eta)
                          for xi_eta in qp_ref], dtype=np.float64)      # (n_q , n_loc , 2)
        return _expand_per_element(ref)  

    def _deriv_table(field: str, ax: int, ay: int):
        """∂^{ax+ay} φ_i / ∂ξ^{ax} ∂η^{ay}"""
        ref = np.asarray(
            [mixed_element.deriv_ref(field, *xi_eta, ax, ay)
             for xi_eta in qp_ref],
            dtype=np.float64
            )
        return _expand_per_element(ref)                                 # (n_elem, n_q, n_loc)

    # ------------------------------------------------------------------
    # 1. Reference-element quadrature (ξ-space)
    # ------------------------------------------------------------------
    qp_ref, _ = volume(mixed_element.mesh.element_type, q_order)

    # ------------------------------------------------------------------
    # 2. Map Function / VectorFunction names  →  objects
    # ------------------------------------------------------------------
    func_map: Dict[str, Any] = {}

    for f in _find_all(expression, Function):
        func_map[f.name] = f
        func_map.setdefault(f.field_name, f)

    for vf in _find_all(expression, VectorFunction):
        func_map[vf.name] = vf
        for comp, fld in zip(vf.components, vf.field_names):
            func_map.setdefault(fld, vf)

    # ------------------------------------------------------------------
    # make sure every component points back to its parent VectorFunction
    # ------------------------------------------------------------------
    for f in list(func_map.values()):
        pv = getattr(f, "_parent_vector", None)
        if pv is not None and hasattr(pv, "name"):
            func_map.setdefault(pv.name, pv)

    # ------------------------------------------------------------------
    # 3. Collect EVERY parameter the kernel will expect
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
    # 4. Build each requested array (skip ones already delivered elsewhere)
    # ------------------------------------------------------------------
    if pre_built is None:
        pre_built = {}

    # Allow base keys, plus exactly what this kernel says it needs in `param_order`.
    base_allow = {
        "gdofs_map", "node_coords", "element_nodes",
        "qp_phys", "qw", "detJ", "J_inv", "normals", "phis",
        "eids", "entity_kind",
        "owner_id", "owner_pos_id", "owner_neg_id", "pos_eids", "neg_eids",

    }
    needed = set(param_order) if param_order is not None else set()

    # Side keys can appear under many names; keep only the ones actually present/needed.
    side_keys_present = set(k for k in (pre_built or {}).keys() if k.startswith(("pos_", "neg_")))

    pre_built = {
        k: v for k, v in (pre_built or {}).items()
        if (k in base_allow) or (k in needed) or (k in side_keys_present)
    }
    args: Dict[str, Any] = dict(pre_built)

    # --- derived row info --------------------------------------------------
    # Number of rows the kernel will loop over
    _n_rows = None
    if gdofs_map is not None:
        _n_rows = int(gdofs_map.shape[0])
    elif "qp_phys" in args:
        _n_rows = int(np.asarray(args["qp_phys"]).shape[0])
    else:
        for _k in ("pos_qp_phys", "neg_qp_phys"):
            if _k in args:
                _n_rows = int(np.asarray(args[_k]).shape[0]); break

    # Row -> owner element ids if precompute provided them (never overwrite 'eids')
    _row_eids = None
    if "owner_id" in args:
        _row_eids = np.asarray(args["owner_id"], dtype=np.int32)
    elif "eids" in args and args.get("entity_kind", "element") == "element":
        _row_eids = np.asarray(args["eids"], dtype=np.int32)

    # Ghost/internal edges: prefer explicit owners when present
    _pos_owner = args.get("owner_pos_id", args.get("pos_eids", None))
    _neg_owner = args.get("owner_neg_id", args.get("neg_eids", None))
    if _pos_owner is not None:
        _pos_owner = np.asarray(_pos_owner, dtype=np.int32)
    if _neg_owner is not None:
        _neg_owner = np.asarray(_neg_owner, dtype=np.int32)


    # --- helper: expand a subset-length per-element array to full length using eids ---
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
            return a  # scalar, nothing to do

        # Already full-length?
        if a.shape[0] == n_elems:
            return a

        # Subset → expand to full using eids
        if eids is not None and a.shape[0] == eids.shape[0]:
            out = np.zeros((n_elems,) + a.shape[1:], dtype=a.dtype)
            out[eids] = a
            return out

        raise ValueError(
            f"{what}: got length {a.shape[0]} but expected either n_elems={n_elems} "
            f"or len(eids)={0 if eids is None else int(eids.shape[0])}."
        )

    # --- Add all domain BitSets used in the expression ---
    all_bitsets_in_form = _find_all_bitsets(expression)
    for bs in all_bitsets_in_form:
        pname = f"domain_bs_{id(bs)}"
        raw = getattr(bs, "array", bs)
        mask_full = _expand_subset_to_full(
            np.asarray(raw, dtype=np.bool_).ravel(), what=f"BitSet {pname}"
        )
        args[pname] = mask_full


    # --- Constant nd-arrays (unchanged)
    const_arrays = {
        f"const_arr_{id(c)}": c
        for c in _find_all(expression, UflConst) if c.dim != 0
    }
    # args.update(const_arrays)

    # cache gdofs_map for coefficient gathering
    if gdofs_map is None:
        gdofs_map = np.vstack([
            dof_handler.get_elemental_dofs(eid)
            for eid in range(mixed_element.mesh.n_elements)
        ]).astype(np.int32)

    total_dofs = dof_handler.total_dofs



    for name in sorted(required):
        
        # Always supply global element-length h_arr for CellDiameter
        # ---- cell diameter (row-aligned) -------------------------------------
        if name == "h_arr":
            ent = args.get("entity_kind", "element")

            if ent == "element":
                # Volume/interface: use per-row owner_ids if available; else full length
                if _row_eids is not None:
                    args["h_arr"] = _h_global[_row_eids]
                elif _n_rows is not None and _n_rows == n_elem:
                    args["h_arr"] = _h_global
                else:
                    # No row mapping available here. Provide global h; codegen can
                    # switch to owner_id indexing when you add it there.
                    args["h_arr"] = _h_global

            else:
                # Facet/ghost: compute face scale from **owners** (no 'eids' changes)
                if (_pos_owner is not None) and (_neg_owner is not None):
                    args["h_arr"] = 0.5 * (_h_global[_pos_owner] + _h_global[_neg_owner])
                elif _pos_owner is not None:
                    args["h_arr"] = _h_global[_pos_owner]
                elif _neg_owner is not None:
                    args["h_arr"] = _h_global[_neg_owner]
                else:
                    raise KeyError(
                        "helpers_jit: cannot build facet h_arr – provide "
                        "owner_pos_id/owner_neg_id or pos_eids/neg_eids in pre_built"
                    )
            continue

        if name in args:
            continue

        # ---- basis tables ------------------------------------------------
        if name.startswith("b_"):
            fld = name[2:]
            args[name] = _basis_table(fld)

        # ---- gradient tables ---------------------------------------------
        elif name.startswith("g_"):
            fld = name[2:]
            args[name] = _grad_table(fld)

        # ---- higher-order ξ-derivatives ----------------------------------
        elif re.match(r"d\d\d_", name):
            tag, fld = name.split("_", 1)
            ax, ay = int(tag[1]), int(tag[2])
            args[name] = _deriv_table(fld, ax, ay)

        
        
        
        
        # ---- constant arrays ---------------------------------------------
        elif name in const_arrays:
            vals = np.asarray(const_arrays[name].value, dtype=np.float64)
            # Element-indexed array? Align rows via owner/eids when available.
            if vals.ndim >= 1 and vals.shape[0] == n_elems and _row_eids is not None:
                args[name] = vals[_row_eids]
            elif vals.ndim >= 1 and _n_rows is not None and vals.shape[0] == _n_rows:
                # Already row-aligned
                args[name] = vals
            else:
                # Keep element-length (codegen can index by owner_id once you switch it)
                args[name] = vals


        # ---- element-wise constants ---------------------------------------------
        elif name.startswith("ewc_"):
            obj_id = int(name.split("_", 1)[1])
            ewc = next(c for c in _find_all(expression, ElementWiseConstant) if id(c) == obj_id)
            arr = np.asarray(ewc.values, dtype=np.float64)  # shape (n_elems, ...)
            if arr.ndim >= 1 and arr.shape[0] == n_elems and _row_eids is not None:
                args[name] = arr[_row_eids]
            elif arr.ndim >= 1 and _n_rows is not None and arr.shape[0] == _n_rows:
                args[name] = arr
            else:
                args[name] = arr   # element-length; codegen can index via owner_id later



        
        # ---- analytic expressions ------------------------------------------------
        elif name.startswith("ana_"):
            func_id = int(name.split("_", 1)[1])
            ana = next(a for a in _find_all(expression, Analytic)
                    if id(a) == func_id)

            # physical quadrature coordinates have already been built a few lines
            # earlier and live in args["qp_phys"]  (shape = n_elem × n_qp × 2)
            qp_phys = args["qp_phys"]
            n_elem, n_qp, _ = qp_phys.shape
            ana_vals = np.empty((n_elem, n_qp), dtype=np.float64)

            # tabulate once, in pure NumPy
            x = qp_phys[..., 0]
            y = qp_phys[..., 1]
            ana_vals[:] = ana.eval(np.stack((x, y), axis=-1))   # vectorised call

            args[name] = ana_vals
        
        # -------------------------------------------------------------------------
        #  Handle coefficient-vector kernel parameters
        #  •  u_<field>_loc                 – element-volume  (side=None)
        #  •  u_<field>_(pos|neg)_loc       – interior facet (side="pos"/"neg")
        #  •  u_u_…  /  u_<side>_loc / u_loc – special cases when the field itself
        #                                      is literally called “u”
        # -------------------------------------------------------------------------
        elif (name.startswith("u_") and name.endswith("_loc")) :
           

            # ---- 1. who is it? ------------------------------------------------
            field, side = symbols.decode_coeff(name)         # ("u", "neg"), ("velocity", None), …

            f = func_map.get(field)
            if f is None:
                raise NameError(
                    "Kernel requests coefficient array for unknown Function "
                    f"or VectorFunction '{field}'."
                    f" input name: {name}"
                    f" side: {side}"
                )

            # ---- 2. build the GLOBAL vector once ------------------------------
            full = np.zeros(total_dofs, dtype=np.float64)
            full[list(f._g2l.keys())] = f.nodal_values

            # ---- 3. gather what the kernel needs ------------------------------
            # ---- facet / ghost-edge geometry (robust to missing side) ---------
            facet_sides = [s for s in ("pos", "neg")
                        if any(f"{s}_{suffix}" in pre_built
                                for suffix in ("map", "qp_phys", "qw", "detJ", "J_inv", "normals", "phis", "eids"))]

            for side in facet_sides:
                # Copy over whatever this side actually provides
                for suffix in ("map", "qp_phys", "qw", "detJ", "J_inv", "normals", "phis", "eids"):
                    key = f"{side}_{suffix}"
                    if key not in pre_built:
                        continue
                    val = pre_built[key]
                    if suffix == "map":
                        # maps can be ragged (1D of objects) → densify
                        arr = np.asarray(val)
                        if arr.ndim == 1 and arr.dtype == object:
                            # use your existing helper if present; otherwise simple stack
                            try:
                                val = _stack_ragged(arr)  # preferred helper if you have it
                            except NameError:
                                val = np.vstack([np.asarray(row, dtype=np.int32) for row in arr]).astype(np.int32)
                    args[key] = val

            # For facet kernels, provide an owner-element eid mapping if available.
            # This lets the Restriction check index the domain mask by element.
            if args.get("entity_kind", None) != "element" and "eids" not in args:
                if "pos_eids" in pre_built:
                    args["eids"] = np.asarray(pre_built["pos_eids"], dtype=np.int32)
                elif "neg_eids" in pre_built:
                    args["eids"] = np.asarray(pre_built["neg_eids"], dtype=np.int32)
                # else: leave absent; the kernel-side guard will handle it

                continue                                    # done – next param

            # ---- volume tables -----------------------------------------------
            args[name] = full[gdofs_map]
            continue

    # ------------------------------------------------------------------
    # 5. Optional debug print
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