"""JIT kernel-argument preparation and scatter utilities.

This module hosts runtime/JIT-specific helpers that were previously in
`pycutfem.ufl.helpers_jit`. It is the canonical location for new code.
"""

import numpy as np
from pathlib import Path
from typing import Mapping, Tuple, Dict, Any, Sequence, Set
from pycutfem.integration import volume
import logging # Added for logging warnings
import os
import pickle
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
_REF_TABLE_CACHE_ABI = "2026-03-26-ref-tables-v2"
_REF_TABLE_CACHE_DIR: Path | None = None
_REF_TABLE_CACHE_DIR_TOKEN: tuple[str, ...] | None = None


def _array_token(arr) -> str:
    carr = np.ascontiguousarray(np.asarray(arr))
    h = blake2b(digest_size=16)
    shape_info = np.asarray(carr.shape, dtype=np.int64)
    h.update(shape_info.tobytes())
    h.update(str(carr.dtype).encode("ascii"))
    h.update(carr.tobytes())
    return h.hexdigest()


def _ref_table_cache_enabled() -> bool:
    return os.getenv("PYCUTFEM_REF_TABLE_CACHE", "1").lower() not in {"0", "false", "no"}


def _ref_table_cache_max_bytes() -> int:
    raw = os.getenv("PYCUTFEM_REF_TABLE_CACHE_MAX_MB", "128").strip()
    try:
        mb = max(1, int(raw))
    except Exception:
        mb = 128
    return mb * 1024 * 1024


def _resolve_ref_table_cache_dir() -> Path | None:
    global _REF_TABLE_CACHE_DIR, _REF_TABLE_CACHE_DIR_TOKEN
    if not _ref_table_cache_enabled():
        _REF_TABLE_CACHE_DIR = None
        _REF_TABLE_CACHE_DIR_TOKEN = None
        return None
    if _REF_TABLE_CACHE_DIR is not None:
        override = os.getenv("PYCUTFEM_REF_TABLE_CACHE_DIR", "").strip()
        if override:
            token = ("override", str(Path(override).expanduser()))
        else:
            cache_root = os.getenv("PYCUTFEM_CACHE_DIR", "").strip()
            xdg_root = os.getenv("XDG_CACHE_HOME", "").strip()
            token = ("root", cache_root, xdg_root)
        if token == _REF_TABLE_CACHE_DIR_TOKEN:
            return _REF_TABLE_CACHE_DIR
    override = os.getenv("PYCUTFEM_REF_TABLE_CACHE_DIR", "").strip()
    try:
        if override:
            base = Path(override).expanduser().resolve()
            token = ("override", str(base))
        else:
            from pycutfem.jit.cache import _resolve_cache_dir
            base = (_resolve_cache_dir() / "ref_tables").resolve()
            token = (
                "root",
                os.getenv("PYCUTFEM_CACHE_DIR", "").strip(),
                os.getenv("XDG_CACHE_HOME", "").strip(),
            )
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    _REF_TABLE_CACHE_DIR = base
    _REF_TABLE_CACHE_DIR_TOKEN = token
    return base


def _ref_table_cache_path(key: tuple) -> Path | None:
    root = _resolve_ref_table_cache_dir()
    if root is None:
        return None
    digest = blake2b(pickle.dumps((_REF_TABLE_CACHE_ABI, key), protocol=5), digest_size=16).hexdigest()
    return root / f"{digest}.npy"


def _cache_array_get(key: tuple) -> np.ndarray | None:
    hit = _REF_TABLE_CACHE.get(key)
    if hit is not None:
        return hit
    path = _ref_table_cache_path(key)
    if path is None or not path.exists():
        return None
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception:
        return None
    _REF_TABLE_CACHE[key] = arr
    return arr


def _cache_array_put(key: tuple, arr: np.ndarray) -> np.ndarray:
    _REF_TABLE_CACHE[key] = arr
    path = _ref_table_cache_path(key)
    if path is None:
        return arr
    if arr.nbytes > _ref_table_cache_max_bytes():
        return arr
    tmp_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return arr
        tmp_path = path.with_suffix(f".{os.getpid()}.tmp.npy")
        with open(tmp_path, "wb") as f:
            np.save(f, arr, allow_pickle=False)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return arr


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
    pre_built: dict | None = None,
    jit_param=None,
    active_cols: np.ndarray | None = None,
    materialize_tables: bool | None = None,
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
    from pycutfem.state.coefficient import QuadratureStateCoefficient
    from pycutfem.jit.ir import LoadVariable, LoadConstantArray
    from pycutfem.ufl.analytic import Analytic
    from pycutfem.ufl.helpers import _find_all
    from pycutfem.fem import transform

  

    logger = __import__("logging").getLogger(__name__)
    dbg    = os.getenv("PYCUTFEM_JIT_DEBUG", "").lower() in {"1", "true", "yes"}
    param_map = jit_param if jit_param is not None else build_jit_parametrization(expression)

    # ------------------------------------------------------------------
    # 0. Helpers
    # ------------------------------------------------------------------
    # Target element count for per-element tables (respect subsets when provided).
    n_elem = int(getattr(mixed_element.mesh, "n_elements", 0) or 0)
    if gdofs_map is not None:
        arr = np.asarray(gdofs_map)
        if arr.ndim < 1:
            raise ValueError(
                f"_build_jit_kernel_args: expected gdofs_map with ndim>=1 to infer n_elem, got shape={arr.shape}."
            )
        n_elem = int(arr.shape[0])
    elif pre_built is not None:
        for _key in ("qp_phys", "qw", "eids", "gdofs_map"):
            if _key not in pre_built:
                continue
            arr = pre_built.get(_key)
            if arr is None:
                continue
            arr_np = np.asarray(arr)
            if arr_np.ndim < 1:
                continue
            n_elem = int(arr_np.shape[0])
            break

    # full-length element size for CellDiameter() lookups via owner ids
    _h_global = np.asarray(
        [mixed_element.mesh.element_char_length(i) for i in range(mixed_element.mesh.n_elements)],
        dtype=np.float64
    )

    # Active fields & union mask for this kernel
    def _active_field_order():
        """
        Determine which fields are active for this kernel while preserving the
        MixedElement DOF ordering.

        IMPORTANT: The union-local DOF layout (and thus r** tables, slices, and
        gdofs_map) is order-dependent. Do *not* reorder fields based on first
        appearance in the IR; only *filter* the MixedElement order.
        """
        me_order = list(getattr(mixed_element, "field_names", ()))
        me_fields = set(me_order)
        active: set[str] = set()
        for op in ir:
            fns = getattr(op, "field_names", None)
            if fns:
                for f in fns or []:
                    if f in me_fields:
                        active.add(f)
                continue
            f = getattr(op, "field_name", None)
            if f in me_fields:
                active.add(f)
        if not active:
            return me_order
        return [f for f in me_order if f in active]

    _active_fields = _active_field_order()
    _active_cols_default = (
        np.concatenate(
            [
                np.arange(
                    mixed_element.component_dof_slices[f].start,
                    mixed_element.component_dof_slices[f].stop,
                    dtype=np.int32,
                )
                for f in _active_fields
            ]
        )
        if _active_fields
        else np.arange(mixed_element.n_dofs_local, dtype=np.int32)
    )
    _active_cols = np.asarray(active_cols, dtype=np.int32).ravel() if active_cols is not None else _active_cols_default
    _active_n = int(_active_cols.size)

    _materialize = bool(materialize_tables) if materialize_tables is not None else False
    _full_n = int(
        getattr(
            mixed_element,
            "n_dofs_local",
            getattr(mixed_element, "n_dofs_per_elem", int(_active_n)),
        )
    )
    if _active_cols.size:
        if int(_active_cols.min()) < 0 or int(_active_cols.max()) >= _full_n:
            raise ValueError(
                f"_build_jit_kernel_args: active_cols out of range [0,{_full_n}) "
                f"(min={int(_active_cols.min())}, max={int(_active_cols.max())})."
            )
        if int(np.unique(_active_cols).size) != int(_active_cols.size):
            raise ValueError("_build_jit_kernel_args: active_cols contains duplicates.")

    _col_map = -np.ones(_full_n, dtype=np.int32)
    if _active_cols.size:
        _col_map[_active_cols] = np.arange(_active_n, dtype=np.int32)

    def _field_new_slice(field: str):
        sl_full = mixed_element.component_dof_slices[field]
        new_idx = _col_map[int(sl_full.start) : int(sl_full.stop)]
        if new_idx.size == 0:
            return slice(0, 0)
        if np.any(new_idx < 0):
            raise ValueError(
                f"_build_jit_kernel_args: active_cols dropped DOFs required for field {field!r} "
                f"(slice={sl_full}, full_n={_full_n}, active_n={_active_n})."
            )
        start = int(new_idx[0])
        if np.array_equal(new_idx, np.arange(start, start + int(new_idx.size), dtype=np.int32)):
            return slice(start, start + int(new_idx.size))
        return new_idx

    _active_layout_token = _array_token(np.asarray(_active_cols, dtype=np.int32))

    def _cached_ref_table(kind: str, field: str, builder, deriv: tuple[int, int] | None = None):
        # Cache reference tables in the *current union layout*. These tables are
        # padded into the active mixed-space columns, so reusing them across a
        # different active_cols selection corrupts H(div) sign/div contractions.
        key = (
            mixed_element.signature(),
            q_order,
            kind,
            field,
            deriv,
            _active_layout_token,
        )
        hit = _cache_array_get(key)
        if hit is not None:
            return hit
        arr = builder()
        return _cache_array_put(key, arr)

    def _expand_per_element(ref_tab: np.ndarray) -> np.ndarray:
        """
        Replicate a reference-space table so that shape[0] == n_elem.
        """
        tab = np.broadcast_to(ref_tab, (n_elem, *ref_tab.shape))
        # Optional: materialize broadcast views once during setup (needed for
        # the C++ backend's pybind11 `c_style` argument conversion).
        return tab.copy() if _materialize else tab

    def _cached_phys_table(kind: str, field: str, token, builder):
        """
        Cache element-physical tables (basis/grad) keyed by a lightweight
        token that tracks the underlying qref/eids arrays for this kernel.
        """
        key = (
            mixed_element.signature(),
            q_order,
            n_elem,
            kind,
            field,
            token,
            _active_layout_token,
        )
        hit = _cache_array_get(key)
        if hit is not None:
            return hit
        arr = builder()
        return _cache_array_put(key, arr)

    def _pad_prebuilt_hdiv_table(field: str, arr: np.ndarray, *, dof_axis: int) -> np.ndarray:
        """
        Promote a field-local RT prebuilt table to the active mixed union layout.

        Prebuilt RT geometry tables are often emitted in the field-local layout
        `(n_loc_rt)` while mixed kernels expect every H(div) table to use the
        same active-union DOF axis as scalar/vector basis tables and sign maps.
        """
        arr = np.asarray(arr, dtype=np.float64)
        axis = int(dof_axis)
        if axis < 0:
            axis += arr.ndim
        if axis < 0 or axis >= arr.ndim:
            raise ValueError(
                f"_pad_prebuilt_hdiv_table: invalid dof axis {dof_axis} for shape {arr.shape}."
            )

        n_union = int(_active_n)
        axis_len = int(arr.shape[axis])
        if axis_len == n_union:
            return np.ascontiguousarray(arr)

        sl_full = mixed_element.component_dof_slices[field]
        nloc = int(sl_full.stop) - int(sl_full.start)
        if axis_len != nloc:
            return np.ascontiguousarray(arr)

        sl = _field_new_slice(field)
        out_shape = list(arr.shape)
        out_shape[axis] = n_union
        out = np.zeros(tuple(out_shape), dtype=np.float64)
        idx = [slice(None)] * arr.ndim
        idx[axis] = sl
        out[tuple(idx)] = arr
        return np.ascontiguousarray(out)

    # Reference-element quadrature (ξ-space) for table builders
    qp_ref, _ = volume(mixed_element.mesh.element_type, q_order)
    qp_ref = np.asarray(qp_ref, dtype=np.float64)
    xi_ref = qp_ref[:, 0]
    eta_ref = qp_ref[:, 1]

    def _basis_table(field: str):
        def _build_local():
            ref_loc = mixed_element._eval_scalar_basis_many(field, xi_ref, eta_ref)
            return np.ascontiguousarray(ref_loc, dtype=np.float64)

        ref_loc = _cached_ref_table("basis_local", field, _build_local)
        ref = np.zeros((int(qp_ref.shape[0]), _active_n), dtype=np.float64)
        if _active_n:
            ref[:, _field_new_slice(field)] = ref_loc
        return _expand_per_element(ref)  # (n_elem , n_q , active_n)

    def _grad_table(field: str):
        def _build_local():
            ref_loc = mixed_element._eval_scalar_grad_many(field, xi_ref, eta_ref)
            return np.ascontiguousarray(ref_loc, dtype=np.float64)

        ref_loc = _cached_ref_table("grad_local", field, _build_local)
        ref = np.zeros((int(qp_ref.shape[0]), _active_n, 2), dtype=np.float64)
        if _active_n:
            ref[:, _field_new_slice(field), :] = ref_loc
        return _expand_per_element(ref)  # (n_elem , n_q , active_n , 2)

    def _deriv_table(field: str, ax: int, ay: int):
        """∂^{ax+ay} φ_i / ∂ξ^{ax} ∂η^{ay} (union-padded to active layout)."""
        from pycutfem.integration.pre_tabulates import _eval_deriv_q1, _eval_deriv_q2, _eval_deriv_p1

        def _build_local():
            sl_full = mixed_element.component_dof_slices[field]
            nloc_f = int(sl_full.stop) - int(sl_full.start)
            if int(ax) == 0 and int(ay) == 0:
                ref_loc = mixed_element._eval_scalar_basis_many(field, xi_ref, eta_ref)
                return np.ascontiguousarray(ref_loc, dtype=np.float64)
            fam = getattr(mixed_element, "_field_families", {}).get(field, None)
            if fam == "RT":
                raise NotImplementedError("Reference derivatives are not defined for RT (H(div)) fields.")
            p = int(getattr(mixed_element, "_field_orders", {}).get(field, 0) or 0)
            elem_type = str(getattr(mixed_element.mesh, "element_type", ""))
            ref_loc = np.empty((int(qp_ref.shape[0]), nloc_f), dtype=np.float64)
            for q, (xi, eta) in enumerate(qp_ref):
                if (int(ax) + int(ay)) <= 2:
                    if elem_type == "quad" and p == 1:
                        ref_loc[q, :] = _eval_deriv_q1(float(xi), float(eta), int(ax), int(ay))
                        continue
                    if elem_type == "quad" and p == 2:
                        ref_loc[q, :] = _eval_deriv_q2(float(xi), float(eta), int(ax), int(ay))
                        continue
                    if elem_type == "tri" and p == 1:
                        ref_loc[q, :] = _eval_deriv_p1(float(xi), float(eta), int(ax), int(ay))
                        continue
                ref_loc[q, :] = np.asarray(
                    mixed_element._eval_scalar_deriv(field, float(xi), float(eta), int(ax), int(ay)),
                    dtype=np.float64,
                ).ravel()
            return np.ascontiguousarray(ref_loc, dtype=np.float64)

        ref_loc = _cached_ref_table("deriv_local", field, _build_local, deriv=(int(ax), int(ay)))
        ref = np.zeros((int(qp_ref.shape[0]), _active_n), dtype=np.float64)
        if _active_n:
            ref[:, _field_new_slice(field)] = ref_loc
        return _expand_per_element(ref)

    def _prebuilt_qref_mode():
        if pre_built is None:
            return None
        qref_raw = pre_built.get("qref")
        if qref_raw is None:
            qref_raw = pre_built.get("qp_ref")
        if qref_raw is None:
            return None
        qref_arr = np.asarray(qref_raw, dtype=np.float64)
        if qref_arr.ndim == 3 and qref_arr.shape[2] == 2:
            return ("per_element", qref_arr)
        if qref_arr.ndim == 2 and qref_arr.shape[1] == 2:
            return ("global", qref_arr)
        return None

    def _kernel_owner_ids(what: str) -> np.ndarray:
        """
        Resolve the owner element id for each kernel row.

        Volume kernels naturally use ``eids``; edge/interface kernels often carry
        non-element ``eids`` and must instead use ``owner_id`` for RT orientation
        and physical Piola mappings.
        """
        owners = None
        if pre_built is not None:
            owners = _first_present(pre_built, "owner_id", "eids")
        if owners is None:
            owners = np.arange(n_elem, dtype=np.int32)
        owners = np.asarray(owners, dtype=np.int64).ravel()
        if int(owners.shape[0]) != int(n_elem):
            raise ValueError(
                f"{what}: owner/eids length mismatch with kernel rows "
                f"(owners={int(owners.shape[0])}, n_elem={int(n_elem)})."
            )
        return owners

    # --- H(div) RT tables (reference-space, union-padded) ----------------------
    def _hdiv_bvec_table(field: str) -> np.ndarray:
        """
        Reference RT basis values, union-padded.

        Returns shape (n_elem, n_q, 2, n_union).
        """
        fam = getattr(mixed_element, "_field_families", {}).get(field, None)
        if fam != "RT":
            raise ValueError(f"Requested bvec_{field} for non-RT field family {fam!r}.")
        key = f"bvec_{field}"
        if pre_built is not None and key in pre_built:
            return _pad_prebuilt_hdiv_table(field, pre_built[key], dof_axis=-1)

        qref_mode = _prebuilt_qref_mode()
        if qref_mode is not None:
            mode, arr = qref_mode
            sl = _field_new_slice(field)
            ref_obj = mixed_element._ref[field]
            n_union = int(_active_n)
            if mode == "global":
                token = ("hdiv_bvec_global", _array_token(arr), int(arr.shape[0]), int(sl.start), int(sl.stop))

                def _build():
                    out = np.zeros((n_elem, int(arr.shape[0]), 2, n_union), dtype=np.float64)
                    for q, (xi, eta) in enumerate(arr):
                        Vhat = np.asarray(ref_obj.tabulate_value(float(xi), float(eta)), dtype=np.float64)
                        out[:, q, :, sl] = Vhat.T[None, :, :]
                    return np.ascontiguousarray(out)

                return _cached_phys_table("hdiv_bvec_union", field, token, _build)

            token = (
                "hdiv_bvec_elem",
                _array_token(arr),
                _array_token(np.asarray(eids, dtype=np.int64)),
                int(arr.shape[0]),
                int(arr.shape[1]),
                int(sl.start),
                int(sl.stop),
            )

            def _build():
                out = np.zeros((n_elem, int(arr.shape[1]), 2, n_union), dtype=np.float64)
                for i in range(n_elem):
                    for q in range(int(arr.shape[1])):
                        xi, eta = arr[i, q]
                        Vhat = np.asarray(ref_obj.tabulate_value(float(xi), float(eta)), dtype=np.float64)
                        out[i, q, :, sl] = Vhat.T
                return np.ascontiguousarray(out)

            return _cached_phys_table("hdiv_bvec_union", field, token, _build)

        def _build():
            n_union = int(_active_n)
            sl = _field_new_slice(field)
            out_ref = np.zeros((len(qp_ref), 2, n_union), dtype=np.float64)
            ref_obj = mixed_element._ref[field]
            for q, (xi, eta) in enumerate(qp_ref):
                Vhat = np.asarray(ref_obj.tabulate_value(float(xi), float(eta)), dtype=np.float64)  # (n_loc,2)
                out_ref[q, :, sl] = Vhat.T
            return np.ascontiguousarray(out_ref)

        ref = _cached_ref_table("hdiv_bvec", field, _build)
        return _expand_per_element(ref)  # (n_elem,n_q,2,n_union)

    def _hdiv_div_table(field: str) -> np.ndarray:
        """
        Reference RT divergences, union-padded.

        Returns shape (n_elem, n_q, n_union).
        """
        fam = getattr(mixed_element, "_field_families", {}).get(field, None)
        if fam != "RT":
            raise ValueError(f"Requested div_{field} for non-RT field family {fam!r}.")
        key = f"div_{field}"
        if pre_built is not None and key in pre_built:
            return _pad_prebuilt_hdiv_table(field, pre_built[key], dof_axis=-1)

        qref_mode = _prebuilt_qref_mode()
        if qref_mode is not None:
            mode, arr = qref_mode
            sl = _field_new_slice(field)
            ref_obj = mixed_element._ref[field]
            n_union = int(_active_n)
            if mode == "global":
                token = ("hdiv_div_global", _array_token(arr), int(arr.shape[0]), int(sl.start), int(sl.stop))

                def _build():
                    out = np.zeros((n_elem, int(arr.shape[0]), n_union), dtype=np.float64)
                    for q, (xi, eta) in enumerate(arr):
                        div_hat = np.asarray(ref_obj.tabulate_div(float(xi), float(eta)), dtype=np.float64).ravel()
                        out[:, q, sl] = div_hat[None, :]
                    return np.ascontiguousarray(out)

                return _cached_phys_table("hdiv_div_union", field, token, _build)

            token = (
                "hdiv_div_elem",
                _array_token(arr),
                _array_token(np.asarray(eids, dtype=np.int64)),
                int(arr.shape[0]),
                int(arr.shape[1]),
                int(sl.start),
                int(sl.stop),
            )

            def _build():
                out = np.zeros((n_elem, int(arr.shape[1]), n_union), dtype=np.float64)
                for i in range(n_elem):
                    for q in range(int(arr.shape[1])):
                        xi, eta = arr[i, q]
                        div_hat = np.asarray(ref_obj.tabulate_div(float(xi), float(eta)), dtype=np.float64).ravel()
                        out[i, q, sl] = div_hat
                return np.ascontiguousarray(out)

            return _cached_phys_table("hdiv_div_union", field, token, _build)

        def _build():
            n_union = int(_active_n)
            sl = _field_new_slice(field)
            out_ref = np.zeros((len(qp_ref), n_union), dtype=np.float64)
            ref_obj = mixed_element._ref[field]
            for q, (xi, eta) in enumerate(qp_ref):
                div_hat = np.asarray(ref_obj.tabulate_div(float(xi), float(eta)), dtype=np.float64).ravel()
                out_ref[q, sl] = div_hat
            return np.ascontiguousarray(out_ref)

        ref = _cached_ref_table("hdiv_div", field, _build)
        return _expand_per_element(ref)  # (n_elem,n_q,n_union)

    def _hdiv_grad_table(field: str) -> np.ndarray:
        """
        Reference RT gradients, union-padded.

        Returns shape ``(n_elem, n_q, 2, n_union, 2)`` with axes
        ``(elem, qp, component, dof, d/dhat_x)``.
        """
        fam = getattr(mixed_element, "_field_families", {}).get(field, None)
        if fam != "RT":
            raise ValueError(f"Requested gvec_{field} for non-RT field family {fam!r}.")
        key = f"gvec_{field}"
        if pre_built is not None and key in pre_built:
            return _pad_prebuilt_hdiv_table(field, pre_built[key], dof_axis=-2)

        qref_mode = _prebuilt_qref_mode()
        if qref_mode is not None:
            mode, arr = qref_mode
            sl = _field_new_slice(field)
            ref_obj = mixed_element._ref[field]
            n_union = int(_active_n)
            if mode == "global":
                token = ("hdiv_grad_global", _array_token(arr), int(arr.shape[0]), int(sl.start), int(sl.stop))

                def _build():
                    out = np.zeros((n_elem, int(arr.shape[0]), 2, n_union, 2), dtype=np.float64)
                    for q, (xi, eta) in enumerate(arr):
                        grad_hat = np.asarray(ref_obj.tabulate_grad(float(xi), float(eta)), dtype=np.float64)
                        out[:, q, :, sl, :] = np.transpose(grad_hat, (1, 0, 2))[None, :, :, :]
                    return np.ascontiguousarray(out)

                return _cached_phys_table("hdiv_grad_union", field, token, _build)

            token = (
                "hdiv_grad_elem",
                _array_token(arr),
                _array_token(np.asarray(eids, dtype=np.int64)),
                int(arr.shape[0]),
                int(arr.shape[1]),
                int(sl.start),
                int(sl.stop),
            )

            def _build():
                out = np.zeros((n_elem, int(arr.shape[1]), 2, n_union, 2), dtype=np.float64)
                for i in range(n_elem):
                    for q in range(int(arr.shape[1])):
                        xi, eta = arr[i, q]
                        grad_hat = np.asarray(ref_obj.tabulate_grad(float(xi), float(eta)), dtype=np.float64)
                        out[i, q, :, sl, :] = np.transpose(grad_hat, (1, 0, 2))
                return np.ascontiguousarray(out)

            return _cached_phys_table("hdiv_grad_union", field, token, _build)

        def _build():
            n_union = int(_active_n)
            sl = _field_new_slice(field)
            out_ref = np.zeros((len(qp_ref), 2, n_union, 2), dtype=np.float64)
            ref_obj = mixed_element._ref[field]
            for q, (xi, eta) in enumerate(qp_ref):
                grad_hat = np.asarray(ref_obj.tabulate_grad(float(xi), float(eta)), dtype=np.float64)  # (n_loc,2,2)
                out_ref[q, :, sl, :] = np.transpose(grad_hat, (1, 0, 2))
            return np.ascontiguousarray(out_ref)

        ref = _cached_ref_table("hdiv_grad", field, _build)
        return _expand_per_element(ref)  # (n_elem,n_q,2,n_union,2)

    def _hdiv_phys_component_table(field: str, kind: str) -> np.ndarray:
        """
        Physical RT component tables, union-padded.

        kind='val'   -> (n_elem, n_q, 2, n_union)
        kind='grad'  -> (n_elem, n_q, 2, n_union, 2)
        kind='hess'  -> (n_elem, n_q, 2, n_union, 2, 2)
        """
        fam = getattr(mixed_element, "_field_families", {}).get(field, None)
        if fam != "RT":
            raise ValueError(f"Requested physical H(div) component table for non-RT field {field!r}.")

        key_map = {"val": f"hval_{field}", "grad": f"hgrad_{field}", "hess": f"hhess_{field}"}
        key = key_map[str(kind)]
        if pre_built is not None and key in pre_built:
            axis_map = {"val": -1, "grad": -2, "hess": -3}
            return _pad_prebuilt_hdiv_table(field, pre_built[key], dof_axis=axis_map[str(kind)])

        qref_mode = _prebuilt_qref_mode()
        sl = _field_new_slice(field)
        n_union = int(_active_n)
        owners = _kernel_owner_ids(f"hdiv_phys_{kind}_{field}")
        try:
            signs_all = dof_handler.element_signs[field]
        except Exception as exc:
            raise RuntimeError(f"Missing dof_handler.element_signs for RT field '{field}'.") from exc
        if owners.size:
            valid_max = int(len(signs_all) - 1)
            max_owner = int(np.max(owners))
            min_owner = int(np.min(owners))
            if min_owner < 0 or max_owner > valid_max:
                entity_kind = None if pre_built is None else pre_built.get("entity_kind", None)
                raw_eids = None if pre_built is None else pre_built.get("eids", None)
                raise ValueError(
                    f"hdiv_phys_{kind}_{field}: owner ids out of range for element_signs "
                    f"(min={min_owner}, max={max_owner}, valid=[0,{valid_max}], "
                    f"entity_kind={entity_kind!r}, "
                    f"sample_eids={None if raw_eids is None else np.asarray(raw_eids, dtype=np.int64).ravel()[:8].tolist()})."
                )

        def _build_for_qref(qref_arr: np.ndarray):
            qref_arr = np.asarray(qref_arr, dtype=np.float64)
            if qref_arr.ndim == 2:
                qref_arr = np.broadcast_to(qref_arr[None, :, :], (n_elem, int(qref_arr.shape[0]), 2))
            out_shape = {
                "val": (n_elem, int(qref_arr.shape[1]), 2, n_union),
                "grad": (n_elem, int(qref_arr.shape[1]), 2, n_union, 2),
                "hess": (n_elem, int(qref_arr.shape[1]), 2, n_union, 2, 2),
            }[str(kind)]
            out = np.zeros(out_shape, dtype=np.float64)
            for i, eid in enumerate(owners):
                sgn = np.asarray(signs_all[int(eid)], dtype=np.float64).ravel()
                for q in range(int(qref_arr.shape[1])):
                    xi, eta = map(float, qref_arr[i, q])
                    if kind == "val":
                        tab = np.asarray(mixed_element.tabulate_value(field, xi, eta, element_id=int(eid)), dtype=np.float64)
                        tab = sgn[:, None] * tab
                        out[i, q, :, sl] = tab.T
                    elif kind == "grad":
                        tab = np.asarray(mixed_element.tabulate_grad(field, xi, eta, element_id=int(eid)), dtype=np.float64)
                        tab = sgn[:, None, None] * tab
                        out[i, q, :, sl, :] = np.transpose(tab, (1, 0, 2))
                    else:
                        tab = np.asarray(mixed_element.tabulate_hessian(field, xi, eta, element_id=int(eid)), dtype=np.float64)
                        tab = sgn[:, None, None, None] * tab
                        out[i, q, :, sl, :, :] = np.transpose(tab, (1, 0, 2, 3))
            return np.ascontiguousarray(out)

        if qref_mode is not None:
            mode, arr = qref_mode
            if mode == "global":
                token = (f"hdiv_phys_{kind}_global", _array_token(arr), int(arr.shape[0]), int(sl.start), int(sl.stop))
                return _cached_phys_table(f"hdiv_phys_{kind}_union", field, token, lambda: _build_for_qref(arr))
            token = (
                f"hdiv_phys_{kind}_elem",
                _array_token(arr),
                _array_token(owners),
                int(arr.shape[0]),
                int(arr.shape[1]),
                int(sl.start),
                int(sl.stop),
            )
            return _cached_phys_table(f"hdiv_phys_{kind}_union", field, token, lambda: _build_for_qref(arr))

        token = (
            f"hdiv_phys_{kind}_default",
            _array_token(qp_ref),
            _array_token(owners),
            int(qp_ref.shape[0]),
            int(sl.start),
            int(sl.stop),
        )
        return _cached_phys_table(f"hdiv_phys_{kind}_union", field, token, lambda: _build_for_qref(qp_ref))

    def _hdiv_sign_table(field: str) -> np.ndarray:
        """
        Per-element RT orientation signs, union-padded.

        Returns shape (n_elem, n_union).
        """
        fam = getattr(mixed_element, "_field_families", {}).get(field, None)
        if fam != "RT":
            raise ValueError(f"Requested sign_{field} for non-RT field family {fam!r}.")

        n_union = int(_active_n)
        sl = _field_new_slice(field)
        out = np.ones((n_elem, n_union), dtype=np.float64)

        owners = None
        if pre_built is not None:
            owners = _first_present(pre_built, "owner_id", "eids")
        if owners is None:
            owners = np.arange(n_elem, dtype=np.int32)
        owners = np.asarray(owners, dtype=np.int64).ravel()
        if owners.shape[0] != n_elem:
            raise ValueError("sign_<field>: owner_id/eids length mismatch with n_elem.")

        try:
            signs_all = dof_handler.element_signs[field]
        except Exception as exc:
            raise RuntimeError(f"Missing dof_handler.element_signs for RT field '{field}'.") from exc

        for e in range(n_elem):
            eid = int(owners[e])
            sgn_loc = np.asarray(signs_all[eid], dtype=np.float64).ravel()
            out[e, sl] = sgn_loc

        return out
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
        sl = me.component_dof_slices[field]
        n_union = int(getattr(me, "n_dofs_local", _n_union_for_eid(eids[0])))
        if n_union < int(sl.stop):
            n_union = int(sl.stop)
        n_loc = int(sl.stop - sl.start)

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

        if qref_mode is not None:
            mode, arr = qref_mode
            if mode == "global":
                token = ("b_phys_global", _array_token(arr), int(arr.shape[0]), int(sl.start), int(sl.stop))

                def _build():
                    xi = np.asarray(arr[:, 0], dtype=float)
                    eta = np.asarray(arr[:, 1], dtype=float)
                    loc = np.asarray(me._eval_scalar_basis_many(field, xi, eta), dtype=np.float64).reshape(int(arr.shape[0]), n_loc)
                    ref = np.zeros((int(arr.shape[0]), n_union), dtype=np.float64)
                    ref[:, sl] = loc
                    return np.ascontiguousarray(np.broadcast_to(ref[None, :, :], (nE, int(arr.shape[0]), n_union)).copy())

                return _cached_phys_table("basis_phys_union", field, token, _build)

            # per-element qref
            token = (
                "b_phys_elem",
                _array_token(arr),
                _array_token(np.asarray(eids, dtype=np.int64)),
                int(arr.shape[0]),
                int(arr.shape[1]),
                int(sl.start),
                int(sl.stop),
            )

            def _build():
                xi = np.asarray(arr[..., 0], dtype=float).ravel()
                eta = np.asarray(arr[..., 1], dtype=float).ravel()
                loc = np.asarray(me._eval_scalar_basis_many(field, xi, eta), dtype=np.float64).reshape(nE, int(arr.shape[1]), n_loc)
                out = np.zeros((nE, int(arr.shape[1]), n_union), dtype=np.float64)
                out[:, :, sl] = loc
                return np.ascontiguousarray(out)

            return _cached_phys_table("basis_phys_union", field, token, _build)

        # Fallback: inverse map physical points to reference and tabulate.
        tab = np.zeros((nE, nQ, n_union), dtype=np.float64)
        for i in range(nE):
            eid = int(eids[i])
            for q in range(nQ):
                x, y = pts[i, q]
                xi, eta = transform.inverse_mapping(mesh, eid, np.array([x, y]))
                tab[i, q, sl] = me._eval_scalar_basis(field, float(xi), float(eta))
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
        sl = me.component_dof_slices[field]
        n_union = int(getattr(me, "n_dofs_local", _n_union_for_eid(eids[0])))
        if n_union < int(sl.stop):
            n_union = int(sl.stop)
        n_loc = int(sl.stop - sl.start)

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

        if qref_mode is not None:
            mode, arr = qref_mode
            if mode == "global":
                token = ("d_phys_global", _array_token(arr), int(arr.shape[0]), int(ax), int(ay), int(sl.start), int(sl.stop))

                def _build():
                    out = np.zeros((nE, int(arr.shape[0]), n_union), dtype=np.float64)
                    for q in range(int(arr.shape[0])):
                        xi, eta = arr[q]
                        out[:, q, sl] = me._eval_scalar_deriv(field, float(xi), float(eta), int(ax), int(ay))[None, :]
                    return np.ascontiguousarray(out)

                return _cached_phys_table("deriv_phys_union", field, token, _build)

            token = (
                "d_phys_elem",
                _array_token(arr),
                _array_token(np.asarray(eids, dtype=np.int64)),
                int(arr.shape[0]),
                int(arr.shape[1]),
                int(ax),
                int(ay),
                int(sl.start),
                int(sl.stop),
            )

            def _build():
                out = np.zeros((nE, int(arr.shape[1]), n_union), dtype=np.float64)
                for i in range(nE):
                    for q in range(int(arr.shape[1])):
                        xi, eta = arr[i, q]
                        out[i, q, sl] = me._eval_scalar_deriv(field, float(xi), float(eta), int(ax), int(ay))
                return np.ascontiguousarray(out)

            return _cached_phys_table("deriv_phys_union", field, token, _build)

        tab = np.zeros((nE, nQ, n_union), dtype=np.float64)
        for i in range(nE):
            eid = int(eids[i])
            for q in range(nQ):
                x, y = pts[i, q]
                xi, eta = transform.inverse_mapping(mesh, eid, np.array([x, y]))
                tab[i, q, sl] = me._eval_scalar_deriv(field, float(xi), float(eta), int(ax), int(ay))
        return np.ascontiguousarray(tab)

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
        sl = me.component_dof_slices[field]
        n_union = int(getattr(me, "n_dofs_local", _n_union_for_eid(eids[0])))
        if n_union < int(sl.stop):
            n_union = int(sl.stop)
        n_loc = int(sl.stop - sl.start)

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

        if qref_mode is not None:
            mode, arr = qref_mode
            if mode == "global":
                token = ("g_phys_global", _array_token(arr), int(arr.shape[0]), int(sl.start), int(sl.stop))

                def _build():
                    xi = np.asarray(arr[:, 0], dtype=float)
                    eta = np.asarray(arr[:, 1], dtype=float)
                    loc = np.asarray(me._eval_scalar_grad_many(field, xi, eta), dtype=np.float64).reshape(int(arr.shape[0]), n_loc, 2)
                    ref = np.zeros((int(arr.shape[0]), n_union, 2), dtype=np.float64)
                    ref[:, sl, :] = loc
                    return np.ascontiguousarray(np.broadcast_to(ref[None, :, :, :], (nE, int(arr.shape[0]), n_union, 2)).copy())

                return _cached_phys_table("grad_phys_union", field, token, _build)

            token = (
                "g_phys_elem",
                _array_token(arr),
                _array_token(np.asarray(eids, dtype=np.int64)),
                int(arr.shape[0]),
                int(arr.shape[1]),
                int(sl.start),
                int(sl.stop),
            )

            def _build():
                xi = np.asarray(arr[..., 0], dtype=float).ravel()
                eta = np.asarray(arr[..., 1], dtype=float).ravel()
                loc = np.asarray(me._eval_scalar_grad_many(field, xi, eta), dtype=np.float64).reshape(nE, int(arr.shape[1]), n_loc, 2)
                out = np.zeros((nE, int(arr.shape[1]), n_union, 2), dtype=np.float64)
                out[:, :, sl, :] = loc
                return np.ascontiguousarray(out)

            return _cached_phys_table("grad_phys_union", field, token, _build)

        tab = np.zeros((nE, nQ, n_union, 2), dtype=np.float64)
        for i in range(nE):
            eid = int(eids[i])
            for q in range(nQ):
                x, y = pts[i, q]
                xi, eta = transform.inverse_mapping(mesh, eid, np.array([x, y]))
                tab[i, q, sl, :] = me._eval_scalar_grad(field, float(xi), float(eta))
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
                        # IMPORTANT:
                        # Element-based interface/cut kernels use a *single-element* union layout
                        # (MixedElement local ordering). Side maps must therefore be expressed in
                        # **full local indices** (slice.start:stop).
                        #
                        # Any active-field compression is handled *once* by the assembler via
                        # `_compress_static_for_active`, which remaps these full indices into the
                        # packed active layout consistently for all union-keyed arrays (b_/g_/r**,
                        # gdofs_map, restrict masks, and these maps).
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
                    if side_ids.size:
                        sample_eid = int(side_ids[0])
                    else:
                        # Empty entity set (e.g. no aligned interface edges). Still build
                        # a correctly-shaped empty map so kernel arg building succeeds.
                        sample_eid = 0
                    nloc_f = len(loc_lists[int(sample_eid)]) if loc_lists else 0
                    m_arr = -np.ones((nE, nloc_f), dtype=np.int32)
                    for i in range(nE):
                        eid = int(side_ids[i])
                        union_row = gdofs[i, :n_union]
                        # map: global dof id -> union column
                        col_of = {int(d): j for j, d in enumerate(union_row)}
                        local_gdofs_f = loc_lists[eid]
                        missing_dofs: list[int] = []
                        for j, d in enumerate(local_gdofs_f):
                            idx = col_of.get(int(d), None)
                            if idx is None:
                                missing_dofs.append(int(d))
                                continue
                            m_arr[i, j] = int(idx)
                        if missing_dofs:
                            sample = ", ".join(str(x) for x in missing_dofs[:6])
                            raise KeyError(
                                f"_build_jit_kernel_args: could not build '{side}_map_{fld}' on entity_kind={entity_kind!r}: "
                                f"{len(missing_dofs)} local DOFs were not found in gdofs_map union row for owner element {eid}. "
                                f"Sample missing global DOFs: [{sample}{'...' if len(missing_dofs) > 6 else ''}]. "
                                "This indicates a layout mismatch between gdofs_map and element_maps."
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
    qstate_arrays: dict[str, QuadratureStateCoefficient] = dict(param_map.qstate_by_name)

    def _validate_qstate_layout(qstate: QuadratureStateCoefficient, name: str) -> None:
        if pre_built is None:
            raise ValueError(
                f"QuadratureStateCoefficient '{name}' requires standard volume precomputed quadrature data."
            )
        entity_kind = str(pre_built.get("entity_kind", "") or "")
        if entity_kind and entity_kind != "element":
            raise NotImplementedError(
                f"QuadratureStateCoefficient '{name}' is only supported on standard volume elements in phase 1; "
                f"got entity_kind={entity_kind!r}."
            )
        qref_raw = _first_present(pre_built, "qp_ref", "qref")
        if qref_raw is None:
            raise ValueError(
                f"QuadratureStateCoefficient '{name}' requires 'qp_ref' or 'qref' in pre_built."
            )
        ref_points = np.asarray(qref_raw, dtype=np.float64)
        if ref_points.ndim == 3:
            raise NotImplementedError(
                f"QuadratureStateCoefficient '{name}' is not yet supported with per-element/ragged quadrature."
            )
        _, ref_weights = volume(mixed_element.mesh.element_type, q_order)
        qstate.layout.validate_against(
            reference_points=ref_points,
            reference_weights=np.asarray(ref_weights, dtype=np.float64),
            context=f"_build_jit_kernel_args[{name}]",
        )

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
            # Kernels index CellDiameter / ElementWiseConstant as h_arr[owner_id[e]].
            #
            # Contract:
            # - By default `h_arr` is a global per-element size array for `mixed_element.mesh`.
            # - For multi-mesh nonmatching interfaces, assemblers may inject a *custom*
            #   `owner_id` + `h_arr` pair (with `domain == "nonmatching_interface"`).
            dom = str(args.get("domain") or args.get("domain_type") or "")
            if dom == "nonmatching_interface" and args.get("h_arr") is not None:
                harr = np.asarray(args["h_arr"], dtype=np.float64)
                owners = args.get("owner_id", None)
                if owners is None:
                    raise KeyError("nonmatching_interface requires 'owner_id' when requesting 'h_arr'.")
                owners = np.asarray(owners, dtype=np.int64).ravel()
                valid = owners >= 0
                if np.any(valid) and int(np.max(owners[valid])) >= int(harr.shape[0]):
                    raise ValueError(
                        "Invalid nonmatching_interface h_arr/owner_id: "
                        f"max(owner_id)={int(np.max(owners[valid]))} >= len(h_arr)={int(harr.shape[0])}."
                    )
                args["h_arr"] = harr
            else:
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
                # Nonmatching-interface precompute also carries sided maps, but its r**
                # tables are already in the expected local layout. Do not treat it as
                # "ghost" for the owner->union pre-padding path.
                dom = str(pb.get("domain") or pb.get("domain_type") or "")
                if dom == "nonmatching_interface":
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
                owner2union = np.asarray(pre_built[side_map_key], dtype=np.int32)    # shape (nE, L)
                if owner2union.shape[0] != nE or owner2union.shape[1] != L:
                    raise ValueError(
                        f"Invalid {side_map_key} shape for ghost padding: got {owner2union.shape}, expected ({nE}, {L})."
                    )

                # Vectorized scatter: out[e,:, owner2union[e,i]] = tab[e,:, i]
                e_idx = np.repeat(np.arange(nE, dtype=np.int32), L)
                i_idx = np.tile(np.arange(L, dtype=np.int32), nE)
                u_idx = owner2union.reshape(-1)
                valid = (u_idx >= 0) & (u_idx < int(n_union))
                if np.any(valid):
                    e_v = e_idx[valid]
                    i_v = i_idx[valid]
                    u_v = u_idx[valid].astype(np.int32, copy=False)
                    out[e_v, :, u_v] = tab[e_v, :, i_v]
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

            if pre_built is not None:
                g_raw = gdofs_map if gdofs_map is not None else pre_built.get("gdofs_map", None)
                g_arr = None if g_raw is None else np.asarray(g_raw)
                nE_empty = 0
                n_union_empty = int(_union_size_from_prebuilt())
                if g_arr is not None and g_arr.ndim >= 1:
                    nE_empty = int(g_arr.shape[0])
                    if g_arr.ndim == 2:
                        n_union_empty = int(g_arr.shape[1])
                elif "eids" in pre_built:
                    nE_empty = int(np.asarray(pre_built.get("eids", np.empty((0,), dtype=np.int32))).shape[0])
                if nE_empty == 0:
                    args[name] = np.empty((0, 0, n_union_empty), dtype=np.float64)
                    continue

            # 2) INTERFACE fallback (entity_kind == 'element' or 'edge'):
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

            # 2b) Aligned-interface edge kernels:
            # compile_multi always registers an aligned-interface kernel so newly-aligned
            # edges can appear during refresh. Some precompute paths do not materialize
            # sided r** tables eagerly, so synthesize them here directly at the edge
            # quadrature points in the active union layout.
            if pre_built is not None and (
                pre_built.get("entity_kind") == "edge"
                or ("qp_phys" in pre_built and "eids" in pre_built)
            ):
                qp_phys_raw = pre_built.get("qp_phys", None)
                nQ = 0
                if qp_phys_raw is not None:
                    qp_phys = np.asarray(qp_phys_raw, dtype=np.float64)
                    if qp_phys.ndim == 3 and int(qp_phys.shape[2]) == 2:
                        nE = int(qp_phys.shape[0])
                        nQ = int(qp_phys.shape[1])
                    elif qp_phys.ndim == 0 and int(qp_phys.size) == 0:
                        nE = 0
                    else:
                        raise ValueError(
                            f"_build_jit_kernel_args: invalid edge 'qp_phys' shape {qp_phys.shape}; expected (nE,nQ,2)."
                        )
                else:
                    if gdofs_map is not None:
                        g = np.asarray(gdofs_map)
                        if g.ndim < 1:
                            raise ValueError(
                                f"_build_jit_kernel_args: invalid gdofs_map shape {g.shape} for edge placeholder."
                            )
                        nE = int(g.shape[0])
                    else:
                        nE = int(np.asarray(pre_built.get("eids", np.empty((0,), dtype=np.int32))).shape[0])

                if nE == 0:
                    if gdofs_map is not None:
                        g = np.asarray(gdofs_map)
                        if g.ndim != 2:
                            raise ValueError(
                                f"_build_jit_kernel_args: expected 2D gdofs_map for edge placeholder, got shape={g.shape}."
                            )
                        n_union = int(g.shape[1])
                    else:
                        n_union = int(_union_size_from_prebuilt())
                    args[name] = np.empty((0, nQ, n_union), dtype=np.float64)
                    continue

                if d0 == 0 and d1 == 0:
                    args[name] = _basis_table_phys_union(fld)
                else:
                    args[name] = _deriv_table_phys_union(fld, d0, d1)
                continue

            # 3) Otherwise (GHOST with no precompute): must be provided via metadata['derivs']
            if _is_ghost(pre_built):
                raise KeyError(
                    f"_build_jit_kernel_args: kernel requests '{name}', but it wasn't provided. "
                    "For ghost integrals, include metadata['derivs'] (e.g. {(1,0),(0,1)}) so r10/r01 are emitted."
                )
            # If not ghost and not interface (shouldn’t happen), fail clearly
            pb_keys = tuple(sorted(str(k) for k in (pre_built or {}).keys()))
            raise KeyError(
                "_build_jit_kernel_args: kernel requests "
                f"'{name}', but it wasn't provided. "
                f"entity_kind={pre_built.get('entity_kind') if isinstance(pre_built, dict) else None!r}, "
                f"has_qp_phys={bool(isinstance(pre_built, dict) and ('qp_phys' in pre_built))}, "
                f"has_eids={bool(isinstance(pre_built, dict) and ('eids' in pre_built))}, "
                f"pre_built_keys={pb_keys[:24]}"
            )

        if name in args:
            continue




        # ---- H(div) RT tables ---------------------------------------------
        if name.startswith("bvec_"):
            fld = name[5:]
            args[name] = _hdiv_bvec_table(fld)

        elif name.startswith("gvec_"):
            fld = name[5:]
            args[name] = _hdiv_grad_table(fld)

        elif name.startswith("hval_"):
            fld = name[5:]
            args[name] = _hdiv_phys_component_table(fld, "val")

        elif name.startswith("hgrad_"):
            fld = name[6:]
            args[name] = _hdiv_phys_component_table(fld, "grad")

        elif name.startswith("hhess_"):
            fld = name[6:]
            args[name] = _hdiv_phys_component_table(fld, "hess")

        elif name.startswith("div_"):
            fld = name[4:]
            fam = getattr(mixed_element, "_field_families", {}).get(fld, None)
            if fam == "RT":
                args[name] = _hdiv_div_table(fld)
            else:
                raise KeyError(f"_build_jit_kernel_args: kernel requested '{name}', but it is not an RT field.")

        elif name.startswith("sign_"):
            fld = name[5:]
            args[name] = _hdiv_sign_table(fld)

        # ---- basis tables ------------------------------------------------
        elif name.startswith("b_"):
            fld = name[2:]
            # Edge-based kernels (ghost facets / aligned interface) operate on an
            # edge-union `gdofs_map` (typically larger than `me.n_dofs_per_elem`).
            # The generated kernels may still request `b_<fld>` for unsided traces
            # (e.g. scalar functionals on aligned interface edges). In that case,
            # `b_<fld>` must be evaluated at the *edge* quadrature points and
            # scattered into the edge-union layout using pos/neg maps.
            if pre_built is not None and pre_built.get("entity_kind") == "edge":
                if gdofs_map is None:
                    raise KeyError("_build_jit_kernel_args: edge kernel requested basis but gdofs_map is missing.")
                gdofs_map_arr = np.asarray(gdofs_map)
                if gdofs_map_arr.ndim != 2:
                    raise ValueError(
                        f"_build_jit_kernel_args: expected 2D gdofs_map for edge kernels, got {gdofs_map_arr.shape}."
                    )
                nE = int(gdofs_map_arr.shape[0])
                n_union_edge = int(gdofs_map_arr.shape[1])

                # Empty aligned-interface placeholder: no edges ⇒ no r** tables.
                # Still provide a correctly-shaped empty basis array so the kernel
                # can be registered and later refreshed when edges become aligned.
                if nE == 0:
                    qp_phys = np.asarray(pre_built.get("qp_phys", np.empty((0, 0, 2))), dtype=np.float64)
                    nQ = int(qp_phys.shape[1]) if qp_phys.ndim >= 2 else 0
                    args[name] = np.empty((0, nQ, n_union_edge), dtype=np.float64)
                    continue

                # Prefer the (+) trace basis; this matches FormCompiler._active_side
                # default precedence (falls back to '+') for unsided evaluation.
                tab_key = f"r00_{fld}_pos"
                map_key = "pos_map"
                if tab_key not in pre_built:
                    # Fallback to (-) if (+) is unavailable.
                    tab_key = f"r00_{fld}_neg"
                    map_key = "neg_map"

                if tab_key not in pre_built or map_key not in pre_built:
                    raise KeyError(
                        f"_build_jit_kernel_args: edge kernel requested '{name}', but missing '{tab_key}'/'{map_key}' "
                        "in pre_built. Ensure metadata['derivs'] includes (0,0) and precompute_ghost_factors(..., allow_interface=True) "
                        "was used for aligned interface edges."
                    )

                tab = np.asarray(pre_built[tab_key], dtype=np.float64)          # (nE, nQ, n_loc_elem_union)
                mapper = np.asarray(pre_built[map_key], dtype=np.int32)         # (nE, n_loc_elem_union)
                if tab.ndim != 3 or mapper.ndim != 2:
                    raise ValueError(
                        f"_build_jit_kernel_args: invalid shapes for '{tab_key}'/'{map_key}': {tab.shape} / {mapper.shape}"
                    )
                if tab.shape[0] != mapper.shape[0] or tab.shape[2] != mapper.shape[1]:
                    raise ValueError(
                        f"_build_jit_kernel_args: shape mismatch for '{tab_key}'/'{map_key}': {tab.shape} / {mapper.shape}"
                    )

                nE, nQ, _ = tab.shape
                out = np.zeros((nE, nQ, n_union_edge), dtype=np.float64)
                ei = np.arange(nE, dtype=np.int32)[:, None, None]
                qi = np.arange(nQ, dtype=np.int32)[None, :, None]
                out[ei, qi, mapper[:, None, :]] += tab
                args[name] = out
            elif pre_built is not None and pre_built.get("entity_kind") == "element" and "qp_phys" in pre_built:
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

        # ---- quadrature-state coefficients ------------------------------
        elif name in qstate_arrays:
            qstate = qstate_arrays[name]
            _validate_qstate_layout(qstate, name)
            arr = _expand_subset_to_full(
                np.asarray(qstate.values, dtype=np.float64),
                what=f"QuadratureState {name}",
            )
            args[name] = np.ascontiguousarray(arr)

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
