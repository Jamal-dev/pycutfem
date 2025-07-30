# pycutfem/jit/__init__.py
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
        # C)  inject *fresh* coefficient blocks for every Function
        #      – volume ( ..._loc )
        #      – ghost  ( ..._pos_loc, ..._neg_loc )
        # ------------------------------------------------------------------
        pos_map = kernel_args.get("pos_map")
        neg_map = kernel_args.get("neg_map")

        # helper ------------------------------------------------------------
        n_union = gdofs_map.shape[1]
        # print(f"KernelRunner: gdofs_map.shape={gdofs_map.shape}, n_union={n_union}")
        def _gather(side_map, tag):
            if side_map is None:
                return
            # print(f"KernelRunner: side_map.shape={side_map.shape}, tag={tag}")
            # side_map is (n_elem, n_side) with union indices (‑1 = padding)
            coeff = np.zeros((side_map.shape[0], n_union), dtype=full_vec.dtype)
            for e in range(side_map.shape[0]):
                idx = side_map[e]
                m   = idx >= 0                         # ignore padding
                coeff[e, idx[m]] = full_vec[gdofs_map[e, idx[m]]]
            kernel_args[f"u_{name}__{tag}_loc"] = coeff   #  **double “__”**


        for name in self.func_names:                  # 'u_k', 'p'
            f = functions[name]                       # Function / VectorFunction

            # 1) global vector with current nodal values --------------------
            full_vec = np.zeros(self.dof_handler.total_dofs,
                                dtype=f.nodal_values.dtype)
            for gdof, lidx in f._g2l.items():
                full_vec[gdof] = f.nodal_values[lidx]

            # 2a) volume coefficients  u_<name>_loc -------------------------
            kernel_args[f"u_{name}_loc"] = full_vec[gdofs_map]

            # 2b) ghost/interface  u_<name>_pos_loc / _neg_loc --------------
            _gather(pos_map, "pos")
            _gather(neg_map, "neg")
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
        return self.kernel(*final_args)

def compile_backend(integral_expression, dof_handler,mixed_element ): # New Newton: Pass dof_handler
    """
    Orchestrates the JIT compilation and returns a reusable runner.
    """
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
    codegen = NumbaCodeGen(mixed_element=mixed_element,form_rank=rank) 
    cache = KernelCache()

    ir_sequence = ir_generator.generate(integral_expression)
    
    kernel, param_order = cache.get_kernel(ir_sequence, codegen,mixed_element.signature())
    
    if hasattr(kernel, "py_func"):
        kernel.python = kernel.py_func
        
    # New Newton: Return the runner, not the raw kernel
    runner = KernelRunner(kernel, param_order, ir_sequence, dof_handler)
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

    def exec(self, current_funcs):
        """Execute the kernel and *return* (Kloc, Floc, Jloc)."""
        return self.runner(current_funcs, self.static_args)

def compile_multi(form, *, dof_handler, mixed_element,
                  quad_order: int | None = None, backend: str = "jit"):
    """
    Compile every integral contained in *form* and return a list of
    _IntegralKernel objects. Supports plain volume, cut volume (level set
    with side), interface, and ghost-edge integrals.
    """
    from pycutfem.ufl.measures import Integral
    from pycutfem.ufl.forms    import Equation
    from pycutfem.ufl.helpers_jit import _build_jit_kernel_args
    from pycutfem.ufl.compilers import FormCompiler

    kernels : list[_IntegralKernel] = []
    fc = FormCompiler(dof_handler, quadrature_order=quad_order, backend=backend)

    # Normalize to a list of Integrals
    if isinstance(form, Equation):   # (a, L)
        integrals = form.a.integrals + form.L.integrals
    elif isinstance(form, Integral):
        integrals = [form]
    else:
        integrals = form.integrals

    for intg in integrals:
        dom = intg.measure.domain_type           # "volume", "interface", "ghost_edge", ...
        qdeg = fc._find_q_order(intg)

        # Compile the backend once; reuse for all subsets of this integral
        runner, ir = fc._compile_backend(intg.integrand, dof_handler, mixed_element)

        # ------------------------------------------------------------------
        # VOLUME (plain or cut)
        # ------------------------------------------------------------------
        if dom == "volume":
            level_set = intg.measure.level_set
            side      = intg.measure.metadata.get("side", "+")
            mesh      = mixed_element.mesh

            # ---- Plain volume (no level set) -----------------------------
            if level_set is None:
                geom = dof_handler.precompute_geometric_factors(qdeg)
                gdofs_map = np.vstack([
                    dof_handler.get_elemental_dofs(e)
                    for e in range(mesh.n_elements)
                ]).astype(np.int32)

                static = {"gdofs_map": gdofs_map, **geom}
                # Safety: ensure 'eids' exists (older helpers might omit it)
                if "eids" not in static:
                    static["eids"] = np.arange(mesh.n_elements, dtype=np.int32)

                static.update(_build_jit_kernel_args(
                    ir, intg.integrand, mixed_element, qdeg,
                    dof_handler=dof_handler,
                    gdofs_map   = gdofs_map,
                    param_order = runner.param_order,
                    pre_built   = geom,
                ))
                kernels.append(_IntegralKernel(runner, static, "volume"))
                continue  # done with this integral

            # ---- Cut volume (level set present) --------------------------
            inside_ids, outside_ids, cut_ids = mesh.classify_elements(level_set)
            # By convention in this codebase:
            #   inside_ids  ↔ elements with  φ < 0
            #   outside_ids ↔ elements with  φ > 0
            if side not in ("+", "-"):
                raise ValueError(f"volume(side=...) must be '+' or '-', got {side!r}")
            side_full = inside_ids if side == "-" else outside_ids

            # Start from full elements on the requested side
            full_ids = np.asarray(side_full, dtype=np.int32)

            # Respect 'defined_on' for the full-element subset too, if provided.
            # (The cut subset already uses defined_on via 'cut_bs' below.)
            bs = intg.measure.defined_on
            if bs is not None:
                # Try the common BitSet APIs first.
                try:
                    allowed = np.asarray(bs.to_indices(), dtype=np.int32)
                except AttributeError:
                    arr = np.asarray(bs)
                    allowed = (np.nonzero(arr)[0].astype(np.int32)
                               if arr.dtype == bool else arr.astype(np.int32))
                # Intersect: full elements from the requested side ∩ defined_on
                full_ids = np.intersect1d(full_ids, allowed, assume_unique=False)

            # 1) FULL elements on the requested side
            if full_ids.size:
                # include level_set so 'phis' is present (may still be None)
                geom_all = dof_handler.precompute_geometric_factors(qdeg, level_set)
                geom_full = {
                    "qp_phys": geom_all["qp_phys"][full_ids],
                    "qw":      geom_all["qw"][full_ids],
                    "detJ":    geom_all["detJ"][full_ids],
                    "J_inv":   geom_all["J_inv"][full_ids],
                    "normals": geom_all["normals"][full_ids],
                    "phis":    None if geom_all["phis"] is None else geom_all["phis"][full_ids],
                }
                gdofs_map_full = np.vstack([
                    dof_handler.get_elemental_dofs(e) for e in full_ids
                ]).astype(np.int32)

                static_full = {"gdofs_map": gdofs_map_full, **geom_full}
                # IMPORTANT: provide 'eids' for the scatter step
                static_full = {"gdofs_map": gdofs_map_full, "eids": full_ids, **geom_full}
                static_full.update(_build_jit_kernel_args(
                    ir, intg.integrand, mixed_element, qdeg,
                    dof_handler=dof_handler,
                    gdofs_map   = gdofs_map_full,
                    param_order = runner.param_order,
                    pre_built   = geom_full,
                ))
                kernels.append(_IntegralKernel(runner, static_full, "volume"))

            # 2) CUT elements (clipped physical quadrature & per-element basis)
            if len(cut_ids):
                derivs = required_multi_indices(intg.integrand) | {(0, 0)}
                cut_mask = mesh.element_bitset("cut")
                cut_bs = (bs & cut_mask) if bs is not None else cut_mask

                geom_cut = dof_handler.precompute_cut_volume_factors(
                    cut_bs, qdeg, derivs, level_set, side=side
                )
                cut_eids = np.asarray(geom_cut.get("eids", []), dtype=np.int32)
                if cut_eids.size:
                    # ensure detJ exists (weights already physical)
                    if "detJ" not in geom_cut:
                        geom_cut["detJ"] = np.ones_like(geom_cut["qw"])

                    gdofs_map_cut = np.vstack([
                        dof_handler.get_elemental_dofs(e) for e in cut_eids
                    ]).astype(np.int32)

                    # geom_cut already carries 'eids'; keep it and pass through
                    static_cut = {"gdofs_map": gdofs_map_cut, **geom_cut}
                    # _build_jit_kernel_args won't overwrite per-element basis we provide
                    static_cut.update(_build_jit_kernel_args(
                        ir, intg.integrand, mixed_element, qdeg,
                        dof_handler=dof_handler,
                        gdofs_map   = gdofs_map_cut,
                        param_order = runner.param_order,
                        pre_built   = geom_cut,
                    ))
                    kernels.append(_IntegralKernel(runner, static_cut, "volume"))

            # finished handling this integral (even if one subset was empty)
            continue

        # ------------------------------------------------------------------
        # INTERFACE (cut edges/faces)
        # ------------------------------------------------------------------
        if dom == "interface":
            level_set = intg.measure.level_set
            bs_cut = mixed_element.mesh.element_bitset("cut")
            bs_def = intg.measure.defined_on
            cut_eids = (bs_def & bs_cut) if bs_def is not None else bs_cut
            geom = dof_handler.precompute_interface_factors(cut_eids, qdeg, level_set)

            # interface assembly still uses element-local maps; safe to use all
            gdofs_map = np.vstack([
                dof_handler.get_elemental_dofs(e)
                for e in geom["eids"]
            ]).astype(np.int32)

            static = {"gdofs_map": gdofs_map, **geom}
            static.update(_build_jit_kernel_args(
                ir, intg.integrand, mixed_element, qdeg,
                dof_handler=dof_handler,
                gdofs_map   = gdofs_map,
                param_order = runner.param_order,
                pre_built   = geom,
            ))
            kernels.append(_IntegralKernel(runner, static, "interface"))
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
            geom = dof_handler.precompute_ghost_factors(edges, qdeg, level_set, derivs)

            # ghost precompute returns the union dof map for each edge
            gdofs_map = geom["gdofs_map"]

            static = {"gdofs_map": gdofs_map, **geom}
            static.update(_build_jit_kernel_args(
                ir, intg.integrand, mixed_element, qdeg,
                dof_handler=dof_handler,
                gdofs_map   = gdofs_map,
                param_order = runner.param_order,
                pre_built   = geom,
            ))
            kernels.append(_IntegralKernel(runner, static, "ghost_edge"))
            continue

        raise NotImplementedError(f"{dom!r} integrals are not supported by JIT.")

    return kernels
