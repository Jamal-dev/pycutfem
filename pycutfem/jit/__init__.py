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
    Compile **every** integral contained in *form* once and return a list of
    _IntegralKernel objects.

    Nothing else in pycutfem is modified, so the legacy “one integral at a
    time” path keeps working untouched.
    """
    from pycutfem.ufl.measures import Integral
    from pycutfem.ufl.forms    import Equation
    from pycutfem.ufl.helpers_jit import _build_jit_kernel_args
    from pycutfem.utils.domain_manager import get_domain_bitset
    from pycutfem.core.dofhandler import DofHandler
    from pycutfem.ufl.compilers import FormCompiler

    kernels : list[_IntegralKernel] = []
    fc = FormCompiler(dof_handler, quadrature_order=quad_order, backend=backend)

    # 0) normalise ----------------------------------------------------------
    if isinstance(form, Equation):   # we normally pass jac == -res
        integrals = form.a.integrals + form.L.integrals
    elif isinstance(form, Integral):
        integrals = [form]
    else:                            # Form
        integrals = form.integrals

    # 1) walk every integral exactly once ----------------------------------
    for intg in integrals:
        dom = intg.measure.domain_type        # "volume", "interface", …

        # ------------------------------ pre‑compute geometry & basis tables
        if dom == "volume":
            qdeg = fc._find_q_order(intg)
            geom = dof_handler.precompute_geometric_factors(qdeg)
            gdofs_map = np.vstack(
                [dof_handler.get_elemental_dofs(e)
                 for e in range(mixed_element.mesh.n_elements)]
            ).astype(np.int32)

        elif dom == "interface":
            qdeg = fc._find_q_order(intg)
            level_set = intg.measure.level_set
            cut_eids  = (intg.measure.defined_on
                         or mixed_element.mesh.element_bitset("cut"))
            geom = dof_handler.precompute_interface_factors(cut_eids,
                                                            qdeg, level_set)
            gdofs_map = np.vstack(
                [dof_handler.get_elemental_dofs(e)
                 for e in range(mixed_element.mesh.n_elements)]
            ).astype(np.int32)

        elif dom == "ghost_edge":
            qdeg = fc._find_q_order(intg)
            level_set = intg.measure.level_set
            edges     = intg.measure.defined_on \
                or mixed_element.mesh.edge_bitset("ghost")
            derivs = required_multi_indices(intg.integrand)
            geom = dof_handler.precompute_ghost_factors(edges, qdeg,
                                                        level_set, derivs)
            gdofs_map = geom["gdofs_map"]      # already the *union* map

        else:
            raise NotImplementedError(f"d{dom} + JIT not wired yet")

        # ------------------------------ compile kernel itself -------------
        runner, ir = fc._compile_backend(intg.integrand,
                                         dof_handler, mixed_element)

        static = {"gdofs_map": gdofs_map, **geom}
        # add only the *basis* tables this kernel really touches
        static.update(_build_jit_kernel_args(
            ir, intg.integrand, mixed_element, qdeg,
            dof_handler=dof_handler,
            gdofs_map   = gdofs_map,
            param_order = runner.param_order,
            pre_built   = geom,
        ))

        kernels.append(_IntegralKernel(runner, static, dom))

    return kernels