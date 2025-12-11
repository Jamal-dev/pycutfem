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
        for ax in range(1, arr.ndim):
            if arr.shape[ax] == full_n:
                idx = [slice(None)] * arr.ndim
                idx[ax] = active_cols
                return arr[tuple(idx)]
        return arr

    def _remap_map(arr: np.ndarray) -> np.ndarray:
        out = -np.ones_like(arr)
        m = (arr >= 0) & (arr < full_n)
        out[m] = col_map[arr[m]]
        return out

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
            if arr.ndim >= 2 and arr.dtype.kind != "O":
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
        try:
            new_args = self.builder(level_set, reuse_static=self.static_args)
        except TypeError:
            new_args = self.builder(level_set)
        if new_args is None:
            return False
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

    # allocate arrays
    for key in keys:
        o_val = old_static.get(key) if old_static else None
        n_val = new_static.get(key) if new_static else None
        if _is_elem_array(o_val, len(old_eids)) or _is_elem_array(n_val, len(new_eids)):
            proto = n_val if _is_elem_array(n_val, len(new_eids)) else o_val
            if proto is None:
                continue
            tail_shape = proto.shape[1:]
            merged[key] = np.zeros((n_total, *tail_shape), dtype=proto.dtype)
        else:
            merged[key] = n_val if n_val is not None else o_val

    merged["eids"] = target_eids

    # fill reused entries
    for dst, eid in enumerate(target_eids):
        if eid in old_index:
            src = old_index[eid]
            for key, arr in merged.items():
                if isinstance(arr, np.ndarray) and arr.shape[0] == n_total:
                    o_val = old_static.get(key) if old_static else None
                    if _is_elem_array(o_val, len(old_eids)):
                        arr[dst] = o_val[src]
        if eid in new_index:
            src = new_index[eid]
            for key, arr in merged.items():
                if isinstance(arr, np.ndarray) and arr.shape[0] == n_total:
                    n_val = new_static.get(key) if new_static else None
                    if _is_elem_array(n_val, len(new_eids)):
                        arr[dst] = n_val[src]

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
    if backend != "jit":
        raise ValueError(
            f"compile_multi supports backend='jit'; got backend={backend!r}. "
            "Use FormCompiler(backend='python') for the pure-Python path."
        )
    from pycutfem.ufl.measures import Integral
    from pycutfem.ufl.forms    import Equation
    from pycutfem.ufl.helpers_jit import _build_jit_kernel_args
    from pycutfem.ufl.compilers import FormCompiler

    kernels : list[_IntegralKernel] = []
    fc = FormCompiler(dof_handler, quadrature_order=quad_order, backend=backend)

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
                geom = dof_handler.precompute_geometric_factors(qdeg, need_hess=need_hess, need_o3=need_o3, need_o4=need_o4,
                                                                deformation=getattr(intg.measure, 'deformation', None))
                geom["is_interface"] = False
                geom["is_ghost"] = False
                gdofs_map = np.vstack([
                    np.asarray(dof_handler.get_elemental_dofs(e), dtype=np.int32)[active_cols]
                    for e in range(mesh.n_elements)
                ]).astype(np.int32)
                geom["gdofs_map"] = gdofs_map

                if "eids" not in geom:
                    geom["eids"] = np.arange(mesh.n_elements, dtype=np.int32)
                static = {"gdofs_map": gdofs_map, **geom}

                static.update(_build_jit_kernel_args(
                    ir, intg.integrand, mixed_element, qdeg,
                    dof_handler=dof_handler,
                    gdofs_map   = gdofs_map,
                    param_order = runner.param_order,
                    pre_built   = geom,
                ))
                static = _compress_static_for_active(static, mixed_element, active_cols)
                kernels.append(_IntegralKernel(runner, static, "volume", eids=np.asarray(geom.get("eids", []), dtype=np.int32)))
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

            def _current_sets(ls_obj):
                """Return (inside, outside, cut) ids based on current mesh tags or by reclassifying."""
                reclassify = False
                try:
                    inside_now = np.asarray(mesh.element_bitset("inside").to_indices(), dtype=np.int32)
                    outside_now = np.asarray(mesh.element_bitset("outside").to_indices(), dtype=np.int32)
                    cut_now = np.asarray(mesh.element_bitset("cut").to_indices(), dtype=np.int32)
                    if (inside_now.size + outside_now.size + cut_now.size) == 0:
                        reclassify = True
                except Exception:
                    reclassify = True

                if reclassify:
                    inside_raw, outside_raw, cut_raw = mesh.classify_elements(ls_obj)
                    inside_now = np.asarray(inside_raw, dtype=np.int32)
                    outside_now = np.asarray(outside_raw, dtype=np.int32)
                    cut_now = np.asarray(cut_raw, dtype=np.int32)
                return inside_now, outside_now, cut_now

            def _apply_defined_on(ids: np.ndarray) -> np.ndarray:
                if bs is None:
                    return ids
                try:
                    allowed = np.asarray(bs.to_indices(), dtype=np.int32)
                except AttributeError:
                    arr = np.asarray(bs)
                    allowed = (np.nonzero(arr)[0].astype(np.int32) if arr.dtype == bool else arr.astype(np.int32))
                return np.intersect1d(ids, allowed, assume_unique=False)

            def _full_static(ls_obj, reuse_static=None):
                if side not in ("+", "-"):
                    raise ValueError(f"volume(side=...) must be '+' or '-', got {side!r}")
                inside_now, outside_now, _ = _current_sets(ls_obj)
                side_full = inside_now if side == "-" else outside_now
                full_ids = _apply_defined_on(side_full)
                prev_ids = _full_cache.get("ids")
                if (
                    prev_ids is not None
                    and full_ids.size
                    and np.array_equal(full_ids, prev_ids)
                    and "phis" not in runner.param_order
                ):
                    cached = _full_cache.get("static")
                    if cached is not None:
                        return cached
                if full_ids.size == 0:
                    empty_map = np.zeros((0, n_loc), dtype=np.int32)
                    return {
                        "gdofs_map": empty_map,
                        "eids": np.zeros(0, dtype=np.int32),
                        "entity_kind": "element",
                        "is_interface": False,
                        "is_ghost": False,
                    }

                qref_all = geom_bg.get("qp_ref")
                if qref_all is not None:
                    qref_slice = qref_all[full_ids] if getattr(qref_all, "ndim", 0) == 3 else qref_all
                else:
                    qref_slice = None

                geom_full = {
                    "qp_phys": geom_bg["qp_phys"][full_ids],
                    "qw": geom_bg["qw"][full_ids],
                    "detJ": geom_bg["detJ"][full_ids],
                    "J_inv": geom_bg["J_inv"][full_ids],
                    "normals": geom_bg["normals"][full_ids],
                    "phis": None if geom_bg.get("phis") is None else geom_bg["phis"][full_ids],
                    "h_arr": geom_bg["h_arr"][full_ids],
                    "owner_id": geom_bg.get("owner_id", geom_bg.get("eids", np.arange(len(geom_bg["qw"]))))[full_ids].astype(np.int32),
                    "entity_kind": "element",
                    "is_interface": False,
                    "is_ghost": False,
                    "eids": full_ids,
                }
                if qref_slice is not None:
                    geom_full["qref"] = qref_slice

                gdofs_map_full = np.vstack([
                    np.asarray(dof_handler.get_elemental_dofs(e), dtype=np.int32)[active_cols]
                    for e in full_ids
                ]).astype(np.int32)
                geom_full["gdofs_map"] = gdofs_map_full

                static_full = dict(geom_full)
                static_full.update(
                    _build_jit_kernel_args(
                        ir,
                        intg.integrand,
                        mixed_element,
                        qdeg,
                        dof_handler=dof_handler,
                        gdofs_map=gdofs_map_full,
                        param_order=runner.param_order,
                        pre_built=geom_full,
                    )
                )
                static_full = _compress_static_for_active(static_full, mixed_element, active_cols)
                _full_cache["ids"] = full_ids
                _full_cache["static"] = static_full
                return static_full

            derivs_cut = required_multi_indices(intg.integrand) | {(0, 0)}

            def _cut_static(ls_obj, reuse_static=None):
                _, _, cut_now = _current_sets(ls_obj)
                cut_ids = np.asarray(_apply_defined_on(cut_now), dtype=np.int32)
                cut_ids = np.unique(cut_ids)
                if cut_ids.size == 0:
                    empty_map = np.zeros((0, n_loc), dtype=np.int32)
                    return {
                        "gdofs_map": empty_map,
                        "eids": np.zeros(0, dtype=np.int32),
                        "owner_id": np.zeros(0, dtype=np.int32),
                        "entity_kind": "element",
                        "is_interface": False,
                        "is_ghost": False,
                        "detJ": np.ones((0, 0), dtype=float),
                        "qw": np.zeros((0, 0), dtype=float),
                    }

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
                    geom_cut = dof_handler.precompute_cut_volume_factors(
                        new_ids,
                        qdeg,
                        derivs_cut,
                        ls_obj,
                        side=side,
                        need_hess=need_hess,
                        need_o3=need_o3,
                        need_o4=need_o4,
                        nseg_hint=nseg,
                        deformation=deformation,
                    )
                    cut_eids_new = np.asarray(geom_cut.get("eids", new_ids), dtype=np.int32)
                    if "detJ" not in geom_cut:
                        geom_cut["detJ"] = np.ones_like(geom_cut["qw"])

                    gdofs_map_cut = np.vstack([
                        np.asarray(dof_handler.get_elemental_dofs(e), dtype=np.int32)[active_cols]
                        for e in cut_eids_new
                    ]).astype(np.int32)
                    geom_cut["gdofs_map"] = gdofs_map_cut
                    static_new = {"gdofs_map": gdofs_map_cut, "eids": cut_eids_new, "owner_id": cut_eids_new, **geom_cut}
                    static_new.update(
                        _build_jit_kernel_args(
                            ir,
                            intg.integrand,
                            mixed_element,
                            qdeg,
                            dof_handler=dof_handler,
                            gdofs_map=gdofs_map_cut,
                            param_order=runner.param_order,
                            pre_built=geom_cut,
                        )
                    )
                    static_new = _compress_static_for_active(static_new, mixed_element, active_cols)

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
            kernels.append(
                _IntegralKernel(
                    runner,
                    static_full,
                    "volume",
                    level_set=level_set,
                    side=side,
                    builder=_full_static,
                    eids=full_eids,
                )
            )

            static_cut = _cut_static(level_set)
            cut_eids = np.asarray(static_cut.get("eids", []), dtype=np.int32)
            kernels.append(
                _IntegralKernel(
                    runner,
                    static_cut,
                    "volume",
                    level_set=level_set,
                    side=side,
                    builder=_cut_static,
                    eids=cut_eids,
                )
            )
            continue

            # finished handling this integral (even if one subset was empty)
            continue

        # ------------------------------------------------------------------
        # INTERFACE (cut edges/faces)
        # ------------------------------------------------------------------
        if dom == "interface":
            level_set = intg.measure.level_set
            bs_cut = mixed_element.mesh.element_bitset("cut")
            bs_def = intg.measure.defined_on
            deformation = getattr(intg.measure, "deformation", None)

            def _interface_static(ls_obj, reuse_static=None):
                cut_eids_bitset = (bs_def & bs_cut) if bs_def is not None else bs_cut
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
                    geom = dof_handler.precompute_interface_factors(
                        new_ids,
                        qdeg,
                        ls_obj,
                        need_hess=need_hess,
                        need_o3=need_o3,
                        need_o4=need_o4,
                        nseg=nseg,
                        deformation=deformation,
                    )
                    geom["is_interface"] = True
                    geom["is_ghost"] = False

                    eids_arr = np.asarray(geom.get("eids", new_ids), dtype=np.int32)
                    if "gdofs_map" in geom:
                        gdofs_map = np.asarray(geom["gdofs_map"], dtype=np.int32)
                        gdofs_map = gdofs_map[:, active_cols] if gdofs_map.size else gdofs_map
                    else:
                        gdofs_map = np.vstack([
                            np.asarray(dof_handler.get_elemental_dofs(e), dtype=np.int32)[active_cols]
                            for e in eids_arr
                        ]).astype(np.int32)
                    geom["gdofs_map"] = gdofs_map

                    static_new = {"gdofs_map": gdofs_map, **geom}
                    static_new.update(
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
                    static_new = _compress_static_for_active(static_new, mixed_element, active_cols)

                merged = _merge_static_arrays(new_eids_full, old_static, static_new)
                merged["is_interface"] = True
                merged["is_ghost"] = False
                merged["entity_kind"] = "element"
                try:
                    merged["_phi_sig"] = np.asarray(
                        [float(ls_obj(mixed_element.mesh.elements_list[int(e)].centroid())) for e in new_eids_full],
                        dtype=float,
                    )
                except Exception:
                    pass
                return merged

            static = _interface_static(level_set)
            eids_arr = np.asarray(static.get("eids", []), dtype=np.int32)
            kernels.append(
                _IntegralKernel(
                    runner,
                    static,
                    "interface",
                    level_set=level_set,
                    builder=_interface_static,
                    eids=eids_arr,
                )
            )
            continue

        # ------------------------------------------------------------------
        # GHOST EDGE (stabilization across a cut)
        # ------------------------------------------------------------------
        if dom == "ghost_edge":
            level_set = intg.measure.level_set
            derivs    = required_multi_indices(intg.integrand)
            bs_ghost = mixed_element.mesh.edge_bitset("ghost")
            bs_def   = intg.measure.defined_on
            edges = (bs_def & bs_ghost) if bs_def is not None else bs_ghost
            def _ghost_static(ls_obj, reuse_static=None):
                mesh_obj = mixed_element.mesh
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

                try:
                    all_eids = np.asarray(edges.to_indices(), dtype=np.int32)
                except Exception:
                    all_eids = np.asarray(edges, dtype=np.int32)
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
                    geom = dof_handler.precompute_ghost_factors(
                        new_ids,
                        qdeg,
                        ls_obj,
                        derivs,
                        need_hess=need_hess,
                        need_o3=need_o3,
                        need_o4=need_o4,
                    )
                    geom["is_ghost"] = True
                    geom["is_interface"] = False
                    gdofs_map_raw = np.asarray(geom.get("gdofs_map", np.zeros((len(new_ids), n_loc), dtype=np.int32)), dtype=np.int32)
                    ncols = gdofs_map_raw.shape[1] if gdofs_map_raw.ndim == 2 else 0
                    eff_cols = active_cols[active_cols < ncols] if ncols else active_cols
                    if eff_cols.size == 0 and ncols:
                        eff_cols = np.arange(ncols, dtype=np.int32)
                    gdofs_map = gdofs_map_raw[:, eff_cols] if gdofs_map_raw.size else gdofs_map_raw
                    geom["gdofs_map"] = gdofs_map

                    static_new = {"gdofs_map": gdofs_map, **geom}
                    static_new.update(
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
                    static_new = _compress_static_for_active(static_new, mixed_element, eff_cols)

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
            kernels.append(
                _IntegralKernel(
                    runner,
                    static,
                    "ghost_edge",
                    level_set=level_set,
                    builder=_ghost_static,
                    eids=eids_arr,
                )
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

            gdofs_raw = np.asarray(geo.get("gdofs_map", np.zeros((0, n_loc), dtype=np.int32)), dtype=np.int32)
            ncols = gdofs_raw.shape[1] if gdofs_raw.ndim == 2 else 0
            eff_cols = active_cols[active_cols < ncols] if ncols else active_cols
            if eff_cols.size == 0 and ncols:
                eff_cols = np.arange(ncols, dtype=np.int32)
            gdofs_map = gdofs_raw[:, eff_cols] if gdofs_raw.size else gdofs_raw
            geo["gdofs_map"] = gdofs_map
            geo["is_interface"] = False
            geo["is_ghost"] = False
            geo["entity_kind"] = "edge"

            static = {"gdofs_map": gdofs_map, **geo}
            static.update(
                _build_jit_kernel_args(
                    ir,
                    intg.integrand,
                    mixed_element,
                    qdeg,
                    dof_handler=dof_handler,
                    gdofs_map=gdofs_map,
                    param_order=runner.param_order,
                    pre_built=geo,
                )
            )
            static = _compress_static_for_active(static, mixed_element, eff_cols)
            eids_arr = np.asarray(static.get("eids", []), dtype=np.int32)
            kernels.append(
                _IntegralKernel(
                    runner,
                    static,
                    "exterior_facet",
                    level_set=None,
                    builder=None,
                    eids=eids_arr,
                )
            )
            continue

        raise NotImplementedError(f"{dom!r} integrals are not supported by JIT.")

    return kernels
