# pycutfem/jit/__init__.py
import re
from pycutfem.ufl.helpers import required_multi_indices
from .visitor import IRGenerator
from .codegen import NumbaCodeGen
from .cache import KernelCache
from pycutfem.fem.mixedelement import MixedElement
import numpy as np
import os
from dataclasses import dataclass
from typing import Callable, Any

#  pycutfem/jit/__init__.py      

def _form_rank(expr):
    """Return 0 (functional), 1 (linear) or 2 (bilinear)."""
    from pycutfem.ufl.expressions import Function, VectorFunction
    from pycutfem.ufl.expressions import TrialFunction, VectorTrialFunction
    from pycutfem.ufl.expressions import TestFunction,  VectorTestFunction

    has_trial = expr.find_first(lambda n: isinstance(
        n, (TrialFunction, VectorTrialFunction))) is not None
    has_test  = expr.find_first(lambda n: isinstance(
        n, (TestFunction,  VectorTestFunction)))  is not None

    return 2 if (has_trial and has_test) else 1 if (has_test) else 0


# ----------------------------------------------------------------------
#  Active-field bookkeeping (per-kernel)
# ----------------------------------------------------------------------
def _active_field_order(ir_sequence, me) -> tuple[str, ...]:
    """Return the ordered list of fields actually referenced in the IR."""
    seen = set()
    order: list[str] = []
    for op in ir_sequence:
        if hasattr(op, "field_names"):
            for f in getattr(op, "field_names", []) or []:
                if f in seen:
                    continue
                if f in getattr(me, "field_names", ()):
                    seen.add(f)
                    order.append(f)
        elif hasattr(op, "field_name"):
            f = getattr(op, "field_name", None)
            if f is None:
                continue
            if f in seen:
                continue
            if f in getattr(me, "field_names", ()):
                seen.add(f)
                order.append(f)
    if not order:
        order = list(getattr(me, "field_names", ()))
    return tuple(order)


def _active_columns(me, active_fields: tuple[str, ...]) -> np.ndarray:
    """Concatenate component slices for the active fields."""
    cols: list[int] = []
    for f in active_fields:
        sl = getattr(me, "component_dof_slices")[f]
        cols.extend(range(sl.start, sl.stop))
    return np.asarray(cols, dtype=np.int32)


def _compress_static_for_active(static: dict[str, Any],
                                me,
                                active_cols: np.ndarray) -> dict[str, Any]:
    """
    Reduce union-sized arrays/maps to the active DOF columns only.
    Keeps geometry untouched; only reshapes arrays that carry union dimensions
    or union-mapping indices.
    """
    full_n = int(getattr(me, "n_dofs_local", active_cols.size))
    # If we are already using all columns in their natural order, nothing to do.
    if active_cols.size == full_n and np.array_equal(
        active_cols, np.arange(full_n, dtype=np.int32)
    ):
        return static

    col_map = -np.ones(full_n, dtype=np.int32)
    for i, old in enumerate(active_cols):
        col_map[int(old)] = int(i)

    def _remap_union(arr: np.ndarray) -> np.ndarray:
        # Prefer the *last* axis matching full_n to avoid slicing nQ when
        # nQ happens to equal the union size (common on cut cells).
        axes = [ax for ax in range(1, arr.ndim) if arr.shape[ax] == full_n]
        if not axes:
            return arr
        ax = axes[-1]
        idx = [slice(None)] * arr.ndim
        idx[ax] = active_cols
        return arr[tuple(idx)]

    def _remap_map(arr: np.ndarray) -> np.ndarray:
        out = -np.ones_like(arr)
        m = (arr >= 0) & (arr < full_n)
        out[m] = col_map[arr[m]]
        return out

    geom_keys = {
        "qp_phys", "qp_ref", "qref", "qw", "detJ",
        "J_inv", "J_inv_pos", "J_inv_neg",
        "normals", "phis", "h_arr",
        "node_coords", "element_nodes",
        "owner_id", "owner_pos_id", "owner_neg_id",
        "eids", "pos_eids", "neg_eids",
        "entity_kind", "is_interface", "is_ghost",
    }

    def _is_union_key(key: str) -> bool:
        return (
            key == "gdofs_map"
            or key.startswith(("b_", "g_", "d", "r", "pos_map", "neg_map", "restrict_mask_"))
        )

    compressed: dict[str, Any] = {}
    for k, v in static.items():
        if k == "gdofs_map" and isinstance(v, np.ndarray):
            if v.shape[1] == active_cols.size:
                compressed[k] = v
            else:
                compressed[k] = v[:, active_cols]
            continue
        if isinstance(v, np.ndarray):
            if k.startswith(("pos_map", "neg_map")) and v.ndim == 2 and v.dtype.kind in {"i", "u"}:
                compressed[k] = _remap_map(v)
                continue
            arr = v
            if (
                arr.ndim >= 2
                and arr.dtype.kind != "O"
                and k not in geom_keys
                and not k.startswith(("domain_bs_", "domain_flag_"))
                and _is_union_key(k)
            ):
                arr = _remap_union(arr)
            compressed[k] = arr
        else:
            compressed[k] = v
    return compressed


# New Newton: Create a class to handle data preparation and execution.
class KernelRunner:
    def __init__(self, kernel, param_order, ir_sequence, dof_handler):
        self.kernel = kernel
        self.param_order = param_order
        self._param_set = set(param_order or [])
        self.dof_handler = dof_handler
        
        # Identify which function coefficients are needed from the IR
        from pycutfem.jit.ir import LoadVariable
        self.func_names = {
            op.name for op in ir_sequence if isinstance(op, LoadVariable) and op.role == 'function'
        }
        # Also infer function names from PARAM_ORDER entries (u_<name>[_pos/_neg]_loc)
        for tag in param_order:
            if not (isinstance(tag, str) and tag.startswith("u_") and tag.endswith("_loc")):
                continue
            mid = tag[2:-4]  # strip leading 'u_' and trailing '_loc'
            # strip optional side suffix
            for suf in ("__pos", "__neg", "_pos", "_neg"):
                if mid.endswith(suf):
                    mid = mid[: -len(suf)]
                    break
            if mid:
                self.func_names.add(mid)


    def __call__(self, functions: dict, static_args: dict):
        """
        Build the positional argument list that the generated Numba kernel
        expects (see its PARAM_ORDER) and execute the kernel.

        Parameters
        ----------
        functions : dict
            Mapping ``name → Function/VectorFunction`` for all symbols
            that the kernel marked with role == 'function'.
        static_args : dict
            All element-wise, iteration-invariant arrays
            (geometry, quadrature weights, basis tables, …).
            Only *missing* items are added; the caller’s dict is **not**
            modified in-place.

        Environment
        -----------
        PYCUTFEM_JIT_DEBUG=1   Print shape & dtype of every kernel argument.
        """
        import os
        import numpy as np

        debug = os.getenv("PYCUTFEM_JIT_DEBUG", "").lower() in {"1", "true", "yes"}

        # ---------------------------------------------------------------
        # A)  start from a shallow copy of the caller-supplied dict
        # ---------------------------------------------------------------
        kernel_args = dict(static_args)

        # ---------------------------------------------------------------
        # B)  guarantee presence of 'gdofs_map'  and  'node_coords'
        # ---------------------------------------------------------------
        if "gdofs_map" not in kernel_args:
            mesh = self.dof_handler.mixed_element.mesh
            kernel_args["gdofs_map"] = np.vstack(
                [self.dof_handler.get_elemental_dofs(eid)
                 for eid in range(mesh.n_elements)]
            ).astype(np.int32)

        if "node_coords" not in kernel_args:          # only if kernel needs it
            kernel_args["node_coords"] = self.dof_handler.get_all_dof_coords()

        gdofs_map = kernel_args["gdofs_map"]          # ndarray, safe to use

        # ------------------------------------------------------------------
        # C)  inject coefficient blocks for every Function
        #      – volume ( ..._loc )
        #      – ghost  ( ..._pos_loc, ..._neg_loc )
        # ------------------------------------------------------------------
        pos_map = kernel_args.get("pos_map")
        neg_map = kernel_args.get("neg_map")

        # helper ------------------------------------------------------------
        n_union = gdofs_map.shape[1]
        total_dofs = self.dof_handler.total_dofs
        full_cache: dict[int, np.ndarray] = {}

        def _gather(full_vec: np.ndarray, side_map, tag, name):
            if side_map is None:
                return
            # side_map is (n_elem, n_side) with union indices (‑1 = padding)
            coeff = np.zeros((side_map.shape[0], n_union), dtype=full_vec.dtype)
            for e in range(side_map.shape[0]):
                idx = side_map[e]
                m = idx >= 0
                if np.any(m):
                    coeff[e, idx[m]] = full_vec[gdofs_map[e, idx[m]]]
            kernel_args[f"u_{name}__{tag}_loc"] = coeff   #  **double “__”**


        for name in self.func_names:                  # 'u_k', 'p'
            try:
                f = functions[name]                   # Function / VectorFunction
            except KeyError:
                # Fallback: match by field_name when the exact symbol name is absent
                candidates = [obj for obj in functions.values() if getattr(obj, "field_name", None) == name]
                if len(candidates) == 1:
                    f = candidates[0]
                else:
                    continue

            # 1) global vector with current nodal values --------------------
            base = getattr(f, "_parent_vector", None)
            source = base if base is not None else f
            cache_key = id(source)
            full_vec = full_cache.get(cache_key)
            if full_vec is None:
                full_vec = np.zeros(total_dofs, dtype=source.nodal_values.dtype)
                if hasattr(source, "_g_dofs") and source._g_dofs.size:
                    full_vec[source._g_dofs] = source.nodal_values
                else:
                    # Fallback: populate via mapping dict (rare path)
                    g2l = getattr(source, "_g2l", {})
                    if g2l:
                        local_vals = source.nodal_values
                        for gdof, lidx in g2l.items():
                            full_vec[gdof] = local_vals[lidx]
                full_cache[cache_key] = full_vec

            # 2a) volume coefficients  u_<name>_loc -------------------------
            if base is not None and base is not f:
                comp_key = (cache_key, f.field_name)
                comp_vec = full_cache.get(comp_key)
                if comp_vec is None:
                    comp_vec = np.zeros_like(full_vec)
                    fld_slice = self.dof_handler.get_field_slice(f.field_name)
                    comp_vec[fld_slice] = full_vec[fld_slice]
                    full_cache[comp_key] = comp_vec
                target_vec = comp_vec
            else:
                target_vec = full_vec
            kernel_args[f"u_{name}_loc"] = target_vec[gdofs_map]

            # 2b) ghost/interface  u_<name>_pos_loc / _neg_loc --------------
            _gather(target_vec, pos_map, "pos", name)
            _gather(target_vec, neg_map, "neg", name)

        # ---------------------------------------------------------------
        # D0) inject extra scalar/array parameters passed via `functions`
        #     (e.g. adaptive `dt` provided through aux_functions).
        # ---------------------------------------------------------------
        for tag in self.param_order:
            if tag in kernel_args or tag not in functions:
                continue
            val = functions[tag]
            try:
                if isinstance(val, np.ndarray):
                    kernel_args[tag] = val
                elif hasattr(val, "value") and not hasattr(val, "nodal_values"):
                    # UFL Constant-like object (scalar or array)
                    kernel_args[tag] = np.asarray(val.value, dtype=np.float64)
                else:
                    kernel_args[tag] = np.asarray(val, dtype=np.float64)
            except Exception:
                # Leave it missing; the check below will raise a clear error.
                pass
        # ---------------------------------------------------------------
        # D)  final sanity check – everything the kernel listed?
        # ---------------------------------------------------------------
        missing = [p for p in self.param_order if p not in kernel_args]
        if missing:
            raise KeyError(
                "KernelRunner: the following static arrays are still missing "
                f"after automatic completion: {missing}. "
                "Compute them once (e.g. with helpers_jit._build_jit_kernel_args) "
                "and pass them via 'static_args'."
            )

        # ---------------------------------------------------------------
        # E)  build positional list in required order  &  optional debug
        # ---------------------------------------------------------------
        final_args = [kernel_args[p] for p in self.param_order]

        if debug:
            print("[KernelRunner] launching kernel with:")
            import numpy as np
            for tag, arr in zip(self.param_order, final_args):
                if isinstance(arr, np.ndarray):
                    print(f"    {tag:<20} shape={arr.shape} dtype={arr.dtype}")
                else:
                    print(f"    {tag:<20} type={type(arr).__name__}")

        # ---------------------------------------------------------------
        # F)  fire the kernel and return its result tuple
        # ---------------------------------------------------------------
        try:
            return self.kernel(*final_args)
        except Exception as exc:
            # Debug aid: surface argument shapes/types when a compiled kernel fails.
            import sys
            print("[KernelRunner] kernel execution failed; argument dump:", file=sys.stderr)
            for tag, arr in zip(self.param_order, final_args):
                if isinstance(arr, np.ndarray):
                    print(f"    {tag:<20} shape={arr.shape} dtype={arr.dtype}", file=sys.stderr)
                else:
                    print(f"    {tag:<20} type={type(arr).__name__}", file=sys.stderr)
            raise

def compile_backend(integral_expression, dof_handler,mixed_element, *, on_facet: bool = False ): # New Newton: Pass dof_handler
    """
    Orchestrates the JIT compilation and returns a reusable runner.
    """
    backend = os.getenv("PYCUTFEM_JIT_BACKEND", "").lower()
    if backend in {"cpp", "c++"}:
        from pycutfem.jit.cpp_backend import compile_backend_cpp
        # Do not fall back silently; raise so we fix missing ops immediately.
        return compile_backend_cpp(
            integral_expression,
            dof_handler,
            mixed_element,
            on_facet=on_facet,
        )

    # Accept Form / Integral / plain Expression alike -----------------
    from pycutfem.ufl.measures import Integral as _Integral
    if hasattr(integral_expression, "integrals"):            # it is a Form
        if len(integral_expression.integrals) != 1:
            raise NotImplementedError("JIT expects a single-integral form.")
        integral_expression = integral_expression.integrals[0].integrand
    elif isinstance(integral_expression, _Integral):         # single Integral
        integral_expression = integral_expression.integrand
    ir_generator = IRGenerator()
    rank    = _form_rank(integral_expression)
    codegen = NumbaCodeGen(mixed_element=mixed_element,form_rank=rank, on_facet=on_facet) 
    cache = KernelCache()

    ir_sequence = ir_generator.generate(integral_expression)
    from pycutfem.jit.ir import strip_side_metadata
    ir_sequence = strip_side_metadata(ir_sequence, on_facet=on_facet)
    
    kernel, param_order = cache.get_kernel(ir_sequence, codegen, mixed_element.signature())
    
    if hasattr(kernel, "py_func"):
        kernel.python = kernel.py_func
        
    # New Newton: Return the runner, not the raw kernel
    runner = KernelRunner(kernel, param_order, ir_sequence, dof_handler)
    # Surface the active-field ordering from the codegen so callers can
    # mirror the static argument compression if needed.
    runner.active_fields = getattr(codegen, "active_fields", None)
    return runner, ir_sequence


# ----------------------------------------------------------------------
#  Convenience: accept a Form with N>1 integrals
# ----------------------------------------------------------------------
@dataclass
class _IntegralKernel:
    """Everything needed to evaluate one integral during Newton."""
    runner:        Callable              # compiled Numba function
    static_args:   dict[str, Any]        # geometry, basis tables, maps …
    domain:        str                   # "volume" | "interface" | "ghost"
    level_set:     Any | None = None     # moving level-set this kernel depends on
    side:          str | None = None     # '+' / '-' for sided volume terms
    builder:       Callable[[Any], dict[str, Any]] | None = None  # refresh hook
    eids:          np.ndarray | None = None  # cached element ids for quick diff

    def exec(self, current_funcs):
        """Execute the kernel and *return* (Kloc, Floc, Jloc)."""
        return self.runner(current_funcs, self.static_args)

    def refresh(self, level_set=None):
        """
        Rebuild static arguments for a (potentially) updated level set without
        re-JIT'ing the kernel. Returns True when the static args were replaced.
        """
        if self.builder is None:
            return False
        old_args = self.static_args if isinstance(self.static_args, dict) else None
        try:
            new_args = self.builder(level_set, reuse_static=self.static_args)
        except TypeError:
            new_args = self.builder(level_set)
        if new_args is None:
            return False
        if old_args is not None and isinstance(new_args, dict):
            for key, val in old_args.items():
                if key not in new_args:
                    new_args[key] = val
        self.static_args = new_args
        self.eids = np.asarray(new_args.get("eids", []), dtype=np.int32)
        return True


def _merge_static_arrays(target_eids: np.ndarray,
                         old_static: dict[str, Any] | None,
                         new_static: dict[str, Any] | None) -> dict[str, Any]:
    """
    Merge per-element static arrays from *old_static* (reused elements)
    and *new_static* (freshly computed subset) into a single dict whose
    first axis matches *target_eids* order. Non array entries are taken
    from new_static when present, else old_static.
    """
    target_eids = np.asarray(target_eids, dtype=np.int32)
    n_total = len(target_eids)
    old_eids = np.asarray(old_static.get("eids", []), dtype=np.int32) if old_static else np.zeros(0, dtype=np.int32)
    new_eids = np.asarray(new_static.get("eids", []), dtype=np.int32) if new_static else np.zeros(0, dtype=np.int32)
    old_index = {int(e): i for i, e in enumerate(old_eids)}
    new_index = {int(e): i for i, e in enumerate(new_eids)}

    def _is_elem_array(val, n_expected):
        return isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == n_expected

    merged: dict[str, Any] = {}
    keys: set[str] = set()
    if old_static:
        keys.update(old_static.keys())
    if new_static:
        keys.update(new_static.keys())

    def _alloc_elem_array(o_arr: np.ndarray | None, n_arr: np.ndarray | None) -> np.ndarray:
        """
        Allocate a merged per-element array, allowing the quadrature axis (axis=1)
        to differ between old/new by padding to max(nQ).
        """
        o_is = o_arr is not None and isinstance(o_arr, np.ndarray)
        n_is = n_arr is not None and isinstance(n_arr, np.ndarray)
        proto = n_arr if n_is else o_arr
        if proto is None:
            return np.zeros((n_total, 0), dtype=float)

        tail_shape = proto.shape[1:]
        if o_is and n_is:
            try:
                # Common case: (nE, nQ, ...) where only nQ differs.
                if o_arr.ndim == n_arr.ndim and o_arr.ndim >= 2 and o_arr.shape[2:] == n_arr.shape[2:]:
                    q = max(int(o_arr.shape[1]), int(n_arr.shape[1]))
                    tail_shape = (q, *tuple(int(s) for s in o_arr.shape[2:]))
            except Exception:
                tail_shape = proto.shape[1:]
        return np.zeros((n_total, *tail_shape), dtype=proto.dtype)

    def _copy_entry(dst: np.ndarray, src: np.ndarray) -> None:
        """
        Copy src -> dst, padding/truncating only along the leading axis when
        dst has a larger quadrature axis than src.
        """
        if dst.shape == src.shape:
            dst[...] = src
            return
        # Allow padding along the first axis (quadrature points).
        if dst.ndim == src.ndim and dst.ndim >= 1 and dst.shape[1:] == src.shape[1:] and dst.shape[0] >= src.shape[0]:
            n0 = int(src.shape[0])
            dst[:n0, ...] = src
            # Leave the tail padded with zeros.
            return
        raise ValueError(f"incompatible entry shapes: dst={dst.shape} src={src.shape}")

    def _assign_entry(arr: np.ndarray, dst_idx: int, src_entry: Any) -> None:
        """
        Assign a single per-element entry into the merged array.

        - For scalar-per-element arrays (ndim==1), assign directly.
        - For tensor-per-element arrays (ndim>=2), allow padding along the
          quadrature axis via _copy_entry.
        """
        if arr.ndim == 1:
            arr[dst_idx] = src_entry
            return
        _copy_entry(arr[dst_idx], np.asarray(src_entry))

    # allocate arrays
    for key in keys:
        o_val = old_static.get(key) if old_static else None
        n_val = new_static.get(key) if new_static else None
        if _is_elem_array(o_val, len(old_eids)) or _is_elem_array(n_val, len(new_eids)):
            merged[key] = _alloc_elem_array(
                np.asarray(o_val) if _is_elem_array(o_val, len(old_eids)) else None,
                np.asarray(n_val) if _is_elem_array(n_val, len(new_eids)) else None,
            )
        else:
            merged[key] = n_val if n_val is not None else o_val

    merged["eids"] = target_eids

    # fill reused entries
    for dst, eid in enumerate(target_eids):
        if eid in old_index:
            src = old_index[eid]
            for key, arr in merged.items():
                if key == "eids":
                    continue
                if isinstance(arr, np.ndarray) and arr.shape[0] == n_total:
                    o_val = old_static.get(key) if old_static else None
                    if _is_elem_array(o_val, len(old_eids)):
                        try:
                            _assign_entry(arr, dst, np.asarray(o_val)[src])
                        except ValueError as exc:
                            raise ValueError(
                                "_merge_static_arrays failed to reuse old entry: "
                                f"key={key!r} eid={int(eid)} dst={dst} src={src} "
                                f"merged_shape={arr.shape} old_shape={np.asarray(o_val).shape} "
                                f"old_entry_shape={np.asarray(o_val)[src].shape}"
                            ) from exc
        if eid in new_index:
            src = new_index[eid]
            for key, arr in merged.items():
                if key == "eids":
                    continue
                if isinstance(arr, np.ndarray) and arr.shape[0] == n_total:
                    n_val = new_static.get(key) if new_static else None
                    if _is_elem_array(n_val, len(new_eids)):
                        try:
                            _assign_entry(arr, dst, np.asarray(n_val)[src])
                        except ValueError as exc:
                            raise ValueError(
                                "_merge_static_arrays failed to insert new entry: "
                                f"key={key!r} eid={int(eid)} dst={dst} src={src} "
                                f"merged_shape={arr.shape} new_shape={np.asarray(n_val).shape} "
                                f"new_entry_shape={np.asarray(n_val)[src].shape}"
                            ) from exc

    return merged


def _phi_signature_from_static(static: dict[str, Any] | None) -> dict[int, float]:
    """
    Build a cheap per-element signature for phi to detect changed cuts.
    Uses stored '_phi_sig' when present; otherwise falls back to the
    first quadrature-point value in 'phis' when available.
    """
    sig: dict[int, float] = {}
    if static is None:
        return sig
    phi_sig_arr = static.get("_phi_sig")
    eids = static.get("eids")
    if eids is None:
        return sig
    arr_eids = np.asarray(eids, dtype=np.int32)
    if phi_sig_arr is not None:
        try:
            arr_sig = np.asarray(phi_sig_arr, dtype=float)
            if arr_sig.shape[0] == arr_eids.shape[0]:
                for eid, val in zip(arr_eids, arr_sig):
                    sig[int(eid)] = float(val)
                return sig
        except Exception:
            pass

    phis = static.get("phis")
    if phis is None:
        return sig
    try:
        arr_phi = np.asarray(phis)
        if arr_phi.ndim >= 2 and arr_phi.shape[0] == arr_eids.shape[0]:
            for eid, vals in zip(arr_eids, arr_phi):
                try:
                    sig[int(eid)] = float(vals.flat[0])
                except Exception:
                    continue
    except Exception:
        return {}
    return sig

def compile_multi(form, *, dof_handler, mixed_element,
                  quad_order: int | None = None, backend: str = "jit"):
    """
    Compile every integral contained in *form* and return a list of
    _IntegralKernel objects. Supports plain volume, cut volume (level set
    with side), interface, and ghost-edge integrals.
    """
    if backend not in {"jit", "cpp", "c++"}:
        raise ValueError(
            f"compile_multi supports backend='jit' or 'cpp'; got backend={backend!r}. "
            "Use FormCompiler(backend='python') for the pure-Python path."
        )
    from pycutfem.ufl.measures import Integral
    from pycutfem.ufl.forms    import Equation
    from pycutfem.ufl.helpers_jit import _build_jit_kernel_args
    from pycutfem.ufl.compilers import FormCompiler

    kernels : list[_IntegralKernel] = []
    fc = FormCompiler(dof_handler, quadrature_order=quad_order, backend=backend)

    def _append_kernel(kernel, integral):
        kernel.integral_id = id(integral)
        kernels.append(kernel)

    # Normalize to a list of Integrals
    if isinstance(form, Equation):   # (a, L)
        integrals = []
        if form.a is not None:
            integrals += form.a.integrals
        if form.L is not None:
            integrals += form.L.integrals
    elif isinstance(form, Integral):
        integrals = [form]
    else:
        integrals = form.integrals

    mesh = mixed_element.mesh
    p_geo  = int(getattr(mesh, "poly_order", 1))
    for intg in integrals:
        dom = intg.measure.domain_type           # "volume", "interface", "ghost_edge", ...
        qdeg = fc._find_q_order(intg)
        on_facet = intg.measure.on_facet
        md = intg.measure.metadata or {}
        qdeg  += 2 * max(0, p_geo - 1)
        nseg   = int(md.get("nseg", max(3, p_geo + qdeg//2)))
        n_loc  = mixed_element.n_dofs_per_elem

        # Compile the backend once; reuse for all subsets of this integral
        runner, ir = fc._compile_backend(intg.integrand, dof_handler, mixed_element, on_facet=on_facet)
        active_fields = _active_field_order(ir, mixed_element)

        # Prefer an explicit hint from the runner/codegen when present.
        runner_active = getattr(runner, "active_fields", None)
        if runner_active:
            active_fields = tuple(runner_active)
        else:
            # Fallback: infer active fields from param_order when IR lacks field annotations
            _param_fields: list[str] = []
            for name in getattr(runner, "param_order", []):
                has_deriv = name.startswith("d") and len(name) > 2 and name[1].isdigit() and "_" in name
                if name.startswith(("b_", "g_")) or has_deriv:
                    fld = name.split("_", 1)[1] if "_" in name else name
                    if fld in getattr(mixed_element, "field_names", ()):
                        _param_fields.append(fld)
            if _param_fields:
                active_fields = tuple(dict.fromkeys(_param_fields))  # preserve order, drop dups

        active_cols = _active_columns(mixed_element, active_fields)
        if os.getenv("PYCUTFEM_JIT_DEBUG_ACTIVE", "").lower() in {"1", "true", "yes"}:
            print(f"[jit] active fields for {intg.measure.domain_type}: {active_fields} (cols={len(active_cols)})")

        # Max derivative order we need in the geometry *inverse* jets (A, A2, A3, A4)
        def _max_required_order(ir_seq):
            from pycutfem.jit.ir import Hessian as IRHessian, Laplacian as IRLaplacian, LoadVariable
            k = 0
            for op in ir_seq:
                if isinstance(op, (IRHessian, IRLaplacian)):
                    k = max(k, 2)
                elif hasattr(op, "deriv_order"):
                    k = max(k, sum(getattr(op, "deriv_order", (0,0))))
            return k
        _kmax = _max_required_order(ir)
        need_hess = (_kmax >= 2)
        need_o3   = (_kmax >= 3)
        need_o4   = (_kmax >= 4)


        # ------------------------------------------------------------------
        # VOLUME (plain or cut)
        # ------------------------------------------------------------------
        if dom == "volume":
            level_set = intg.measure.level_set
            side      = intg.measure.metadata.get("side", "+")
            mesh      = mixed_element.mesh

            # ---- Plain volume (no level set) -----------------------------
            if level_set is None:
                # Respect measure.defined_on by slicing the element batch up-front.
                # Otherwise we'd execute the kernel on *all* elements and then
                # zero contributions post-hoc (wasted work; very expensive for
                # small subdomains like the solid in FSI).
                from pycutfem.ufl.helpers import normalize_elem_ids as _norm_eids

                sel = getattr(intg.measure, "defined_on", None)
                allowed = _norm_eids(mesh, sel)
                if allowed is None:
                    element_ids = np.arange(mesh.n_elements, dtype=np.int32)
                else:
                    element_ids = np.asarray(allowed, dtype=np.int32)

                geom_all = dof_handler.precompute_geometric_factors(
                    qdeg,
                    need_hess=need_hess,
                    need_o3=need_o3,
                    need_o4=need_o4,
                    deformation=getattr(intg.measure, "deformation", None),
                )

                # Slice only per-element arrays; keep global tables as-is.
                geom = {}
                n_total = int(np.asarray(geom_all["qp_phys"]).shape[0])
                for key, val in geom_all.items():
                    if isinstance(val, np.ndarray) and val.ndim >= 1 and int(val.shape[0]) == n_total:
                        geom[key] = val[element_ids]
                    else:
                        geom[key] = val

                geom["eids"] = element_ids
                geom["is_interface"] = False
                geom["is_ghost"] = False

                if element_ids.size:
                    gdofs_map = np.vstack(
                        [
                            np.asarray(dof_handler.get_elemental_dofs(int(e)), dtype=np.int32)[
                                active_cols
                            ]
                            for e in element_ids
                        ]
                    ).astype(np.int32)
                else:
                    gdofs_map = np.zeros((0, int(active_cols.size)), dtype=np.int32)
                geom["gdofs_map"] = gdofs_map

                static = {**geom, "gdofs_map": gdofs_map}

                static.update(
                    _build_jit_kernel_args(
                        ir,
                        intg.integrand,
                        mixed_element,
                        qdeg,
                        dof_handler=dof_handler,
                        gdofs_map=gdofs_map,
                        param_order=runner.param_order,
                        pre_built=geom,
                    )
                )
                static = _compress_static_for_active(static, mixed_element, active_cols)
                _append_kernel(
                    _IntegralKernel(
                        runner,
                        static,
                        "volume",
                        eids=np.asarray(element_ids, dtype=np.int32),
                    ),
                    intg,
                )
                continue  # done with this integral

            # ---- Cut volume (level set present) --------------------------
            bs = intg.measure.defined_on
            deformation = getattr(intg.measure, "deformation", None)

            # Geometry for full (uncut) elements does not depend on the moving LS
            geom_bg = dof_handler.precompute_geometric_factors(
                qdeg,
                level_set if "phis" in runner.param_order else None,
                need_hess=need_hess,
                need_o3=need_o3,
                need_o4=need_o4,
                deformation=deformation,
            )
            _full_cache = {"ids": None, "static": None}

            # NOTE: These helper/build functions are called later via kernel.refresh().
            # Bind loop-variant values as default args to avoid Python's late-binding
            # closure pitfall (which would mix up per-integral active_cols/qdeg/etc.).
            def _current_sets(ls_obj, _mesh=mesh):
                """Return (inside, outside, cut) ids based on current mesh tags or by reclassifying."""
                reclassify = False
                try:
                    inside_now = np.asarray(_mesh.element_bitset("inside").to_indices(), dtype=np.int32)
                    outside_now = np.asarray(_mesh.element_bitset("outside").to_indices(), dtype=np.int32)
                    cut_now = np.asarray(_mesh.element_bitset("cut").to_indices(), dtype=np.int32)
                    if (inside_now.size + outside_now.size + cut_now.size) == 0:
                        reclassify = True
                except Exception:
                    reclassify = True

                if reclassify:
                    inside_raw, outside_raw, cut_raw = _mesh.classify_elements(ls_obj)
                    inside_now = np.asarray(inside_raw, dtype=np.int32)
                    outside_now = np.asarray(outside_raw, dtype=np.int32)
                    cut_now = np.asarray(cut_raw, dtype=np.int32)
                return inside_now, outside_now, cut_now

            def _apply_defined_on(ids: np.ndarray, _bs=bs) -> np.ndarray:
                if _bs is None:
                    return ids
                try:
                    allowed = np.asarray(_bs.to_indices(), dtype=np.int32)
                except AttributeError:
                    arr = np.asarray(_bs)
                    allowed = (np.nonzero(arr)[0].astype(np.int32) if arr.dtype == bool else arr.astype(np.int32))
                return np.intersect1d(ids, allowed, assume_unique=False)

            def _full_static(
                ls_obj,
                reuse_static=None,
                _side=side,
                _geom_bg=geom_bg,
                _full_cache=_full_cache,
                _current_sets=_current_sets,
                _apply_defined_on=_apply_defined_on,
                _active_cols=np.asarray(active_cols, dtype=np.int32),
                _ir=ir,
                _integrand=intg.integrand,
                _qdeg=int(qdeg),
                _param_order=tuple(getattr(runner, "param_order", []) or []),
                _dof_handler=dof_handler,
                _me=mixed_element,
            ):
                if _side not in ("+", "-"):
                    raise ValueError(f"volume(side=...) must be '+' or '-', got {_side!r}")
                # Invalidate the full-element cache whenever the level set changes.
                ls_token = getattr(ls_obj, "cache_token", None)
                if ls_token is None:
                    ls_token = ("objid", int(id(ls_obj)))
                inside_now, outside_now, _ = _current_sets(ls_obj)
                side_full = inside_now if _side == "-" else outside_now
                full_ids = _apply_defined_on(side_full)

                phis_src = _geom_bg.get("phis")
                prev_ids = _full_cache.get("ids")
                prev_tok = _full_cache.get("ls_token")
                if (
                    prev_ids is not None
                    and full_ids.size
                    and np.array_equal(full_ids, prev_ids)
                    and prev_tok == ls_token
                    and "phis" not in _param_order
                ):
                    cached = _full_cache.get("static")
                    if cached is not None:
                        return cached

                qref_all = _geom_bg.get("qp_ref")
                if qref_all is not None:
                    qref_slice = qref_all[full_ids] if getattr(qref_all, "ndim", 0) == 3 else qref_all
                else:
                    qref_slice = None

                geom_full = {
                    "qp_phys": _geom_bg["qp_phys"][full_ids],
                    "qw": _geom_bg["qw"][full_ids],
                    "detJ": _geom_bg["detJ"][full_ids],
                    "J_inv": _geom_bg["J_inv"][full_ids],
                    "normals": _geom_bg["normals"][full_ids],
                    "phis": None if phis_src is None else phis_src[full_ids],
                    "h_arr": _geom_bg["h_arr"][full_ids],
                    "owner_id": _geom_bg.get("owner_id", _geom_bg.get("eids", np.arange(len(_geom_bg["qw"]))))[full_ids].astype(np.int32),
                    "entity_kind": "element",
                    "is_interface": False,
                    "is_ghost": False,
                    "eids": full_ids,
                }
                if qref_slice is not None:
                    geom_full["qref"] = qref_slice

                if full_ids.size:
                    gdofs_map_full = np.vstack([
                        np.asarray(_dof_handler.get_elemental_dofs(int(e)), dtype=np.int32)[_active_cols]
                        for e in full_ids
                    ]).astype(np.int32)
                else:
                    gdofs_map_full = np.zeros((0, int(_active_cols.size)), dtype=np.int32)
                geom_full["gdofs_map"] = gdofs_map_full

                static_full = dict(geom_full)
                static_full.update(
                    _build_jit_kernel_args(
                        _ir,
                        _integrand,
                        _me,
                        _qdeg,
                        dof_handler=_dof_handler,
                        gdofs_map=gdofs_map_full,
                        param_order=_param_order,
                        pre_built=geom_full,
                    )
                )
                static_full = _compress_static_for_active(static_full, _me, _active_cols)
                if full_ids.size:
                    _full_cache["ids"] = full_ids
                    _full_cache["static"] = static_full
                    _full_cache["ls_token"] = ls_token
                return static_full

            derivs_cut = required_multi_indices(intg.integrand) | {(0, 0)}

            def _cut_static(
                ls_obj,
                reuse_static=None,
                _side=side,
                _derivs_cut=derivs_cut,
                _current_sets=_current_sets,
                _apply_defined_on=_apply_defined_on,
                _active_cols=np.asarray(active_cols, dtype=np.int32),
                _ir=ir,
                _integrand=intg.integrand,
                _qdeg=int(qdeg),
                _param_order=tuple(getattr(runner, "param_order", []) or []),
                _dof_handler=dof_handler,
                _me=mixed_element,
                _nseg=nseg,
                _need_hess=need_hess,
                _need_o3=need_o3,
                _need_o4=need_o4,
                _deformation=deformation,
            ):
                _, _, cut_now = _current_sets(ls_obj)
                cut_ids = np.asarray(_apply_defined_on(cut_now), dtype=np.int32)
                cut_ids = np.unique(cut_ids)
                if cut_ids.size == 0:
                    geom_cut_src = dof_handler.precompute_cut_volume_factors(
                        cut_ids,
                        _qdeg,
                        _derivs_cut,
                        ls_obj,
                        side=_side,
                        need_hess=_need_hess,
                        need_o3=_need_o3,
                        need_o4=_need_o4,
                        nseg_hint=_nseg,
                        deformation=_deformation,
                    )
                    geom_cut = dict(geom_cut_src)
                    geom_cut["is_interface"] = False
                    geom_cut["is_ghost"] = False
                    geom_cut["entity_kind"] = "element"
                    if "owner_id" not in geom_cut:
                        geom_cut["owner_id"] = np.asarray(geom_cut.get("eids", cut_ids), dtype=np.int32)

                    gdofs_map_cut = np.zeros((0, int(_active_cols.size)), dtype=np.int32)

                    static_cut = {**geom_cut, "gdofs_map": gdofs_map_cut}
                    static_cut.update(
                        _build_jit_kernel_args(
                            _ir,
                            _integrand,
                            _me,
                            _qdeg,
                            dof_handler=_dof_handler,
                            gdofs_map=gdofs_map_cut,
                            param_order=_param_order,
                            pre_built=static_cut,
                        )
                    )
                    static_cut = _compress_static_for_active(static_cut, _me, _active_cols)
                    return static_cut

                old_static = reuse_static if isinstance(reuse_static, dict) else None
                old_phi_sig = _phi_signature_from_static(old_static)
                old_eids = np.asarray(old_static.get("eids", []), dtype=np.int32) if old_static else np.zeros(0, dtype=np.int32)
                # recompute if new eid or phi value changed
                need_recompute: list[int] = []
                for eid in cut_ids:
                    if eid not in old_phi_sig:
                        need_recompute.append(int(eid))
                        continue
                    try:
                        # evaluate phi at element centroid as a cheap signature
                        xc = np.asarray(mesh.elements_list[int(eid)].centroid(), float)
                        val = float(ls_obj(xc))
                    except Exception:
                        val = old_phi_sig.get(int(eid), None)
                    old_val = old_phi_sig.get(int(eid), None)
                    if old_val is None or not np.isclose(val, old_val, rtol=0, atol=1e-12):
                        need_recompute.append(int(eid))
                new_ids = np.asarray(sorted(set(need_recompute)), dtype=np.int32)

                static_new = None
                if new_ids.size:
                    geom_cut_src = _dof_handler.precompute_cut_volume_factors(
                        new_ids,
                        _qdeg,
                        _derivs_cut,
                        ls_obj,
                        side=_side,
                        need_hess=_need_hess,
                        need_o3=_need_o3,
                        need_o4=_need_o4,
                        nseg_hint=_nseg,
                        deformation=_deformation,
                    )
                    geom_cut = dict(geom_cut_src)
                    cut_eids_new = np.asarray(geom_cut.get("eids", new_ids), dtype=np.int32)
                    if "detJ" not in geom_cut and "qw" in geom_cut:
                        geom_cut["detJ"] = np.ones_like(geom_cut["qw"])

                    gdofs_map_cut = np.vstack([
                        np.asarray(_dof_handler.get_elemental_dofs(int(e)), dtype=np.int32)[_active_cols]
                        for e in cut_eids_new
                    ]).astype(np.int32)
                    static_new = {**geom_cut, "eids": cut_eids_new, "owner_id": cut_eids_new, "gdofs_map": gdofs_map_cut}
                    static_new.update(
                        _build_jit_kernel_args(
                            _ir,
                            _integrand,
                            _me,
                            _qdeg,
                            dof_handler=_dof_handler,
                            gdofs_map=gdofs_map_cut,
                            param_order=_param_order,
                            pre_built=static_new,
                        )
                    )
                    static_new = _compress_static_for_active(static_new, _me, _active_cols)

                # Merge reused rows with newly computed ones; arrays are built in target cut_ids order
                merged = _merge_static_arrays(cut_ids, old_static, static_new)
                merged["owner_id"] = merged.get("owner_id", cut_ids)
                merged["is_interface"] = False
                merged["is_ghost"] = False
                merged["entity_kind"] = "element"
                # store phi signature at centroids for future refresh decisions
                try:
                    merged["_phi_sig"] = np.asarray(
                        [float(ls_obj(mesh.elements_list[int(e)].centroid())) for e in cut_ids],
                        dtype=float,
                    )
                except Exception:
                    pass
                return merged

            static_full = _full_static(level_set)
            full_eids = np.asarray(static_full.get("eids", []), dtype=np.int32)
            _append_kernel(
                _IntegralKernel(
                    runner,
                    static_full,
                    "volume",
                    level_set=level_set,
                    side=side,
                    builder=_full_static,
                    eids=full_eids,
                ),
                intg,
            )

            static_cut = _cut_static(level_set)
            cut_eids = np.asarray(static_cut.get("eids", []), dtype=np.int32)
            _append_kernel(
                _IntegralKernel(
                    runner,
                    static_cut,
                    "volume",
                    level_set=level_set,
                    side=side,
                    builder=_cut_static,
                    eids=cut_eids,
                ),
                intg,
            )
            continue

            # finished handling this integral (even if one subset was empty)
            continue

        # ------------------------------------------------------------------
        # INTERFACE (cut edges/faces)
        # ------------------------------------------------------------------
        if dom == "interface":
            level_set = intg.measure.level_set
            bs_def = intg.measure.defined_on
            deformation = getattr(intg.measure, "deformation", None)

            def _interface_static(
                ls_obj,
                reuse_static=None,
                _bs_def=bs_def,
                _dof_handler=dof_handler,
                _me=mixed_element,
                _ir=ir,
                _integrand=intg.integrand,
                _qdeg=int(qdeg),
                _param_order=tuple(getattr(runner, "param_order", []) or []),
                _active_cols=np.asarray(active_cols, dtype=np.int32),
                _need_hess=need_hess,
                _need_o3=need_o3,
                _need_o4=need_o4,
                _nseg=nseg,
                _deformation=deformation,
            ):
                # IMPORTANT: mesh element/edge BitSets are rebuilt by classification,
                # so never capture them across refreshes; always fetch a fresh view.
                bs_cut_now = _me.mesh.element_bitset("cut")
                cut_eids_bitset = (_bs_def & bs_cut_now) if _bs_def is not None else bs_cut_now
                try:
                    new_eids_full = np.asarray(cut_eids_bitset.to_indices(), dtype=np.int32)
                except Exception:
                    new_eids_full = np.asarray(cut_eids_bitset, dtype=np.int32)
                new_eids_full = np.unique(new_eids_full)

                old_static = reuse_static if isinstance(reuse_static, dict) else None
                old_phi_sig = _phi_signature_from_static(old_static)
                old_eids = np.asarray(old_static.get("eids", []), dtype=np.int32) if old_static else np.zeros(0, dtype=np.int32)
                need_recompute: list[int] = []
                for eid in new_eids_full:
                    if eid not in old_phi_sig:
                        need_recompute.append(int(eid))
                        continue
                    try:
                        xc = np.asarray(mixed_element.mesh.elements_list[int(eid)].centroid(), float)
                        val = float(ls_obj(xc))
                    except Exception:
                        val = old_phi_sig.get(int(eid), None)
                    old_val = old_phi_sig.get(int(eid), None)
                    if old_val is None or not np.isclose(val, old_val, rtol=0, atol=1e-12):
                        need_recompute.append(int(eid))
                new_ids = np.asarray(sorted(set(need_recompute)), dtype=np.int32)

                static_new = None
                if new_ids.size:
                    geom_src = _dof_handler.precompute_interface_factors(
                        new_ids,
                        _qdeg,
                        ls_obj,
                        need_hess=_need_hess,
                        need_o3=_need_o3,
                        need_o4=_need_o4,
                        nseg=_nseg,
                        deformation=_deformation,
                    )
                    geom = dict(geom_src)
                    geom["is_interface"] = True
                    geom["is_ghost"] = False

                    eids_arr = np.asarray(geom.get("eids", new_ids), dtype=np.int32)
                    if "gdofs_map" in geom:
                        gdofs_map_raw = np.asarray(geom["gdofs_map"], dtype=np.int32)
                        ncols = gdofs_map_raw.shape[1] if gdofs_map_raw.ndim == 2 else 0
                        eff_cols = _active_cols[_active_cols < ncols] if ncols else _active_cols
                        if eff_cols.size == 0 and ncols:
                            eff_cols = np.arange(ncols, dtype=np.int32)
                        gdofs_map = gdofs_map_raw[:, eff_cols] if gdofs_map_raw.size else gdofs_map_raw
                    else:
                        eff_cols = _active_cols
                        gdofs_map = np.vstack([
                            np.asarray(_dof_handler.get_elemental_dofs(int(e)), dtype=np.int32)[eff_cols]
                            for e in eids_arr
                        ]).astype(np.int32)

                    static_new = {**geom, "gdofs_map": gdofs_map}
                    static_new.update(
                        _build_jit_kernel_args(
                            _ir,
                            _integrand,
                            _me,
                            _qdeg,
                            dof_handler=_dof_handler,
                            gdofs_map=gdofs_map,
                            param_order=_param_order,
                            pre_built=static_new,
                        )
                    )
                    static_new = _compress_static_for_active(static_new, _me, eff_cols)

                merged = _merge_static_arrays(new_eids_full, old_static, static_new)
                merged["is_interface"] = True
                merged["is_ghost"] = False
                merged["entity_kind"] = "element"
                try:
                    merged["_phi_sig"] = np.asarray(
                        [float(ls_obj(_me.mesh.elements_list[int(e)].centroid())) for e in new_eids_full],
                        dtype=float,
                    )
                except Exception:
                    pass
                return merged

            static = _interface_static(level_set)
            if "gdofs_map" not in static:
                static["gdofs_map"] = np.zeros((len(static.get("eids", [])), int(active_cols.size)), dtype=np.int32)
            eids_arr = np.asarray(static.get("eids", []), dtype=np.int32)
            _append_kernel(
                _IntegralKernel(
                    runner,
                    static,
                    "interface",
                    level_set=level_set,
                    builder=_interface_static,
                    eids=eids_arr,
                ),
                intg,
            )

            # Also support "aligned" interface facets (interface coincides with an element edge).
            # We always register this kernel so newly-aligned edges can appear during a refresh.
            derivs_ifc = required_multi_indices(intg.integrand)

            def _aligned_interface_static(
                ls_obj,
                reuse_static=None,
                _bs_def=bs_def,
                _derivs_ifc=derivs_ifc,
                _dof_handler=dof_handler,
                _me=mixed_element,
                _ir=ir,
                _integrand=intg.integrand,
                _qdeg=int(qdeg),
                _param_order=tuple(getattr(runner, "param_order", []) or []),
                _active_cols=np.asarray(active_cols, dtype=np.int32),
                _need_hess=need_hess,
                _need_o3=need_o3,
                _need_o4=need_o4,
                _deformation=deformation,
            ):
                mesh_obj = _me.mesh
                nodes_xy = getattr(mesh_obj, "nodes_x_y_pos", None)

                def _edge_midpoint(eid: int) -> np.ndarray:
                    edge = mesh_obj.edges_list[int(eid)]
                    idx = np.asarray(edge.all_nodes if edge.all_nodes else edge.nodes, dtype=int)
                    if nodes_xy is None:
                        raise ValueError("Mesh nodes are unavailable for midpoint computation.")
                    return np.asarray(nodes_xy[idx].mean(axis=0), dtype=float)

                old_static = reuse_static if isinstance(reuse_static, dict) else None
                old_phi_sig = _phi_signature_from_static(old_static)

                # Always fetch current interface edges (mesh edge BitSets are rebuilt on classify_edges).
                base_edges = mesh_obj.edge_bitset("interface")
                edge_sel = base_edges
                if _bs_def is not None:
                    # If defined_on is an edge mask (same length as edges), respect it.
                    try:
                        if len(_bs_def) == len(mesh_obj.edges_list):
                            edge_sel = _bs_def & base_edges
                    except Exception:
                        pass

                try:
                    all_eids = np.asarray(edge_sel.to_indices(), dtype=np.int32)
                except Exception:
                    all_eids = np.asarray(edge_sel, dtype=np.int32)
                all_eids = np.unique(all_eids)

                need_recompute: list[int] = []
                for eid in all_eids:
                    if eid not in old_phi_sig:
                        need_recompute.append(int(eid))
                        continue
                    try:
                        val = float(ls_obj(_edge_midpoint(eid)))
                    except Exception:
                        val = old_phi_sig.get(int(eid), None)
                    old_val = old_phi_sig.get(int(eid), None)
                    if old_val is None or not np.isclose(val, old_val, rtol=0, atol=1e-12):
                        need_recompute.append(int(eid))
                new_ids = np.asarray(sorted(set(need_recompute)), dtype=np.int32)

                static_new = None
                if new_ids.size:
                    geom_src = _dof_handler.precompute_ghost_factors(
                        ghost_edge_ids=new_ids,
                        qdeg=_qdeg,
                        level_set=ls_obj,
                        derivs=_derivs_ifc,
                        allow_interface=True,
                        need_hess=_need_hess,
                        need_o3=_need_o3,
                        need_o4=_need_o4,
                        deformation=_deformation,
                    )
                    geom = dict(geom_src)
                    gdofs_map_raw = np.asarray(
                        geom.get("gdofs_map", np.zeros((len(new_ids), n_loc), dtype=np.int32)),
                        dtype=np.int32,
                    )
                    ncols = gdofs_map_raw.shape[1] if gdofs_map_raw.ndim == 2 else 0
                    use_full_union = bool(ncols and ncols != _me.n_dofs_local)
                    if use_full_union:
                        eff_cols = np.arange(ncols, dtype=np.int32)
                    else:
                        eff_cols = _active_cols[_active_cols < ncols] if ncols else _active_cols
                        if eff_cols.size == 0 and ncols:
                            eff_cols = np.arange(ncols, dtype=np.int32)
                    gdofs_map = gdofs_map_raw[:, eff_cols] if gdofs_map_raw.size else gdofs_map_raw

                    static_new = {**geom, "gdofs_map": gdofs_map, "is_ghost": False, "is_interface": True}
                    static_new.update(
                        _build_jit_kernel_args(
                            _ir,
                            _integrand,
                            _me,
                            _qdeg,
                            dof_handler=_dof_handler,
                            gdofs_map=gdofs_map,
                            param_order=_param_order,
                            pre_built=static_new,
                        )
                    )
                    if not use_full_union:
                        static_new = _compress_static_for_active(static_new, _me, eff_cols)

                merged = _merge_static_arrays(all_eids, old_static, static_new)
                merged["is_ghost"] = False
                merged["is_interface"] = True
                merged["entity_kind"] = "edge"
                if "gdofs_map" not in merged:
                    merged["gdofs_map"] = np.zeros((len(all_eids), int(_active_cols.size)), dtype=np.int32)
                try:
                    merged["_phi_sig"] = np.asarray([float(ls_obj(_edge_midpoint(e))) for e in all_eids], dtype=float)
                except Exception:
                    pass
                return merged

            static_ifc_edge = _aligned_interface_static(level_set)
            eids_edge = np.asarray(static_ifc_edge.get("eids", []), dtype=np.int32)
            _append_kernel(
                _IntegralKernel(
                    runner,
                    static_ifc_edge,
                    "interface",
                    level_set=level_set,
                    builder=_aligned_interface_static,
                    eids=eids_edge,
                ),
                intg,
            )
            continue

        # ------------------------------------------------------------------
        # GHOST EDGE (stabilization across a cut)
        # ------------------------------------------------------------------
        if dom == "ghost_edge":
            level_set = intg.measure.level_set
            derivs    = required_multi_indices(intg.integrand)
            bs_def   = intg.measure.defined_on
            def _ghost_static(
                ls_obj,
                reuse_static=None,
                _bs_def=bs_def,
                _derivs=derivs,
                _dof_handler=dof_handler,
                _me=mixed_element,
                _ir=ir,
                _integrand=intg.integrand,
                _qdeg=int(qdeg),
                _param_order=tuple(getattr(runner, "param_order", []) or []),
                _active_cols=np.asarray(active_cols, dtype=np.int32),
                _need_hess=need_hess,
                _need_o3=need_o3,
                _need_o4=need_o4,
            ):
                mesh_obj = _me.mesh
                nodes_xy = getattr(mesh_obj, "nodes_x_y_pos", None)

                def _edge_midpoint(eid: int) -> np.ndarray:
                    edge = mesh_obj.edges_list[int(eid)]
                    idx = np.asarray(edge.all_nodes if edge.all_nodes else edge.nodes, dtype=int)
                    if nodes_xy is None:
                        raise ValueError("Mesh nodes are unavailable for midpoint computation.")
                    return np.asarray(nodes_xy[idx].mean(axis=0), dtype=float)

                old_static = reuse_static if isinstance(reuse_static, dict) else None
                old_phi_sig = _phi_signature_from_static(old_static)
                old_eids = np.asarray(old_static.get("eids", []), dtype=np.int32) if old_static else np.zeros(0, dtype=np.int32)

                # Always fetch current ghost edges (mesh edge BitSets are rebuilt on classify_edges).
                bs_ghost_now = mesh_obj.edge_bitset("ghost")
                edge_sel = (_bs_def & bs_ghost_now) if _bs_def is not None else bs_ghost_now
                try:
                    all_eids = np.asarray(edge_sel.to_indices(), dtype=np.int32)
                except Exception:
                    all_eids = np.asarray(edge_sel, dtype=np.int32)
                all_eids = np.unique(all_eids)

                need_recompute: list[int] = []
                for eid in all_eids:
                    if eid not in old_phi_sig:
                        need_recompute.append(int(eid))
                        continue
                    try:
                        val = float(ls_obj(_edge_midpoint(eid)))
                    except Exception:
                        val = old_phi_sig.get(int(eid), None)
                    old_val = old_phi_sig.get(int(eid), None)
                    if old_val is None or not np.isclose(val, old_val, rtol=0, atol=1e-12):
                        need_recompute.append(int(eid))
                new_ids = np.asarray(sorted(set(need_recompute)), dtype=np.int32)

                static_new = None
                if new_ids.size:
                    geom_src = _dof_handler.precompute_ghost_factors(
                        new_ids,
                        _qdeg,
                        ls_obj,
                        _derivs,
                        need_hess=_need_hess,
                        need_o3=_need_o3,
                        need_o4=_need_o4,
                    )
                    geom = dict(geom_src)
                    gdofs_map_raw = np.asarray(
                        geom.get("gdofs_map", np.zeros((len(new_ids), n_loc), dtype=np.int32)), dtype=np.int32
                    )
                    ncols = gdofs_map_raw.shape[1] if gdofs_map_raw.ndim == 2 else 0
                    use_full_union = bool(ncols and ncols != _me.n_dofs_local)
                    if use_full_union:
                        eff_cols = np.arange(ncols, dtype=np.int32)
                    else:
                        eff_cols = _active_cols[_active_cols < ncols] if ncols else _active_cols
                        if eff_cols.size == 0 and ncols:
                            eff_cols = np.arange(ncols, dtype=np.int32)
                    gdofs_map = gdofs_map_raw[:, eff_cols] if gdofs_map_raw.size else gdofs_map_raw

                    static_new = {**geom, "gdofs_map": gdofs_map, "is_ghost": True, "is_interface": False}
                    static_new.update(
                        _build_jit_kernel_args(
                            _ir,
                            _integrand,
                            _me,
                            _qdeg,
                            dof_handler=_dof_handler,
                            gdofs_map=gdofs_map,
                            param_order=_param_order,
                            pre_built=static_new,
                        )
                    )
                    if not use_full_union:
                        static_new = _compress_static_for_active(static_new, _me, eff_cols)

                merged = _merge_static_arrays(all_eids, old_static, static_new)
                merged["is_ghost"] = True
                merged["is_interface"] = False
                merged["entity_kind"] = "edge"
                try:
                    merged["_phi_sig"] = np.asarray([float(ls_obj(_edge_midpoint(e))) for e in all_eids], dtype=float)
                except Exception:
                    pass
                return merged

            static = _ghost_static(level_set)
            eids_arr = np.asarray(static.get("eids", []), dtype=np.int32)
            _append_kernel(
                _IntegralKernel(
                    runner,
                    static,
                    "ghost_edge",
                    level_set=level_set,
                    builder=_ghost_static,
                    eids=eids_arr,
                ),
                intg,
            )
            continue

        # ------------------------------------------------------------------
        # INTERIOR FACET (ds)
        # ------------------------------------------------------------------
        if dom == "interior_facet":
            mesh = mixed_element.mesh
            edge_set = intg.measure.defined_on
            if edge_set is None:
                edge_ids = [int(e.gid) for e in mesh.edges_list if e.right is not None]
            else:
                edge_ids = edge_set

            derivs = required_multi_indices(intg.integrand)

            def _interior_static():
                geom = dof_handler.precompute_interior_factors(
                    edge_ids=edge_ids,
                    qdeg=qdeg,
                    derivs=derivs,
                    need_hess=need_hess,
                    need_o3=need_o3,
                    need_o4=need_o4,
                )
                geom_local = dict(geom)
                geom_local["is_ghost"] = False
                geom_local["is_interface"] = False

                gdofs_map_raw = np.asarray(
                    geom_local.get("gdofs_map", np.zeros((len(geom_local.get("eids", [])), n_loc), dtype=np.int32)),
                    dtype=np.int32,
                )
                ncols = gdofs_map_raw.shape[1] if gdofs_map_raw.ndim == 2 else 0
                use_full_union = bool(ncols and ncols != mixed_element.n_dofs_local)
                if use_full_union:
                    eff_cols = np.arange(ncols, dtype=np.int32)
                else:
                    eff_cols = active_cols[active_cols < ncols] if ncols else active_cols
                    if eff_cols.size == 0 and ncols:
                        eff_cols = np.arange(ncols, dtype=np.int32)
                gdofs_map = gdofs_map_raw[:, eff_cols] if gdofs_map_raw.size else gdofs_map_raw
                geom_local["gdofs_map"] = gdofs_map

                static = {"gdofs_map": gdofs_map, **geom_local}
                static.update(
                    _build_jit_kernel_args(
                        ir,
                        intg.integrand,
                        mixed_element,
                        qdeg,
                        dof_handler=dof_handler,
                        gdofs_map=gdofs_map,
                        param_order=runner.param_order,
                        pre_built=static,
                    )
                )
                if not use_full_union:
                    static = _compress_static_for_active(static, mixed_element, eff_cols)
                return static

            static = _interior_static()
            eids_arr = np.asarray(static.get("eids", []), dtype=np.int32)
            _append_kernel(
                _IntegralKernel(
                    runner,
                    static,
                    "interior_facet",
                    eids=eids_arr,
                ),
                intg,
            )
            continue

        # ------------------------------------------------------------------
        # EXTERIOR FACET (boundary edges)
        # ------------------------------------------------------------------
        if dom == "exterior_facet":
            mesh = mixed_element.mesh
            edge_set = (
                intg.measure.defined_on
                if intg.measure.defined_on is not None
                else mesh.get_domain_bitset(intg.measure.tag, entity="edge")
            )
            if edge_set is None:
                raise ValueError(f"[jit] No edges defined for tag {intg.measure.tag!r}.")

            derivs = required_multi_indices(intg.integrand)
            geo = dof_handler.precompute_boundary_factors(
                edge_set,
                qdeg,
                derivs,
                need_hess=need_hess,
                need_o3=need_o3,
                need_o4=need_o4,
            )
            if geo.get("eids", np.zeros(0, dtype=np.int32)).size == 0:
                continue

            geo_local = dict(geo)
            gdofs_raw = np.asarray(geo_local.get("gdofs_map", np.zeros((0, n_loc), dtype=np.int32)), dtype=np.int32)
            ncols = gdofs_raw.shape[1] if gdofs_raw.ndim == 2 else 0
            eff_cols = active_cols[active_cols < ncols] if ncols else active_cols
            if eff_cols.size == 0 and ncols:
                eff_cols = np.arange(ncols, dtype=np.int32)
            gdofs_map = gdofs_raw[:, eff_cols] if gdofs_raw.size else gdofs_raw
            geo_local["is_interface"] = False
            geo_local["is_ghost"] = False
            geo_local["entity_kind"] = "edge"

            static = {"gdofs_map": gdofs_map, **geo_local}
            static.update(
                _build_jit_kernel_args(
                    ir,
                    intg.integrand,
                    mixed_element,
                    qdeg,
                    dof_handler=dof_handler,
                    gdofs_map=gdofs_map,
                    param_order=runner.param_order,
                    pre_built=static,
                )
            )
            static = _compress_static_for_active(static, mixed_element, eff_cols)
            eids_arr = np.asarray(static.get("eids", []), dtype=np.int32)
            _append_kernel(
                _IntegralKernel(
                    runner,
                    static,
                    "exterior_facet",
                    level_set=None,
                    builder=None,
                    eids=eids_arr,
                ),
                intg,
            )
            continue

        raise NotImplementedError(f"{dom!r} integrals are not supported by JIT.")

    return kernels
