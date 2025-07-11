import numpy as np
from typing import Mapping, Tuple, Dict, Any
from pycutfem.integration import volume
import logging # Added for logging warnings
import os
import re

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


def _build_jit_kernel_args(       # ← NEW signature (unchanged)
        ir,                       # linear IR produced by the visitor
        expression,               # the UFL integrand
        mixed_element,            # MixedElement instance
        q_order: int,             # quadrature order
        dof_handler,              # DofHandler – *needed* for padding
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

    from pycutfem.integration import volume
    from pycutfem.ufl.expressions import (
        Function, VectorFunction, Constant as UflConst, Grad, ElementWiseConstant
    )
    from pycutfem.jit.ir import LoadVariable, LoadConstantArray, LoadElementWiseConstant
    from pycutfem.ufl.analytic import Analytic
    from pycutfem.ufl.compilers import _find_all

    logger = logging.getLogger(__name__)
    dbg    = os.getenv("PYCUTFEM_JIT_DEBUG", "").lower() in {"1", "true", "yes"}

    # ------------------------------------------------------------------
    # 0. Helpers
    # ------------------------------------------------------------------
    def _basis_table(field: str):
        """φ_i(ξ,η) for every quadrature point"""
        return np.asarray(
            [mixed_element.basis(field, *xi_eta) for xi_eta in qp_ref],
            dtype=np.float64
        )

    def _grad_table(field: str):
        """∇φ_i(ξ,η) in reference coords"""
        return np.asarray(
            [mixed_element.grad_basis(field, *xi_eta) for xi_eta in qp_ref],
            dtype=np.float64
        )

    def _deriv_table(field: str, ax: int, ay: int):
        """∂^{ax+ay} φ_i / ∂ξ^{ax} ∂η^{ay}"""
        return np.asarray(
            [mixed_element.deriv_ref(field, *xi_eta, ax, ay)
             for xi_eta in qp_ref],
            dtype=np.float64
        )

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
    # 3. Collect EVERY parameter the kernel will expect
    # ------------------------------------------------------------------
    required: set[str] = set(param_order) if param_order is not None else set()

    # If `param_order` was not given collect names from the IR too
    if param_order is None:
        for op in ir:
            if isinstance(op, LoadVariable):
                fnames = (
                    op.field_names if op.is_vector and op.field_names
                    else [op.name]
                )
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
    predeclared = {
         "gdofs_map", "node_coords", "element_nodes",
         "qp_phys", "qw", "detJ", "J_inv", "normals", "phis"
    }   
    pre_built = {**{k: v for k, v in pre_built.items()
                    if k in predeclared}, **pre_built}
    args: Dict[str, Any] = dict(pre_built)

    const_arrays = {
        f"const_arr_{id(c)}": c
        for c in _find_all(expression, UflConst) if c.dim != 0
    }

    # cache gdofs_map for coefficient gathering
    gdofs_map = np.vstack([
        dof_handler.get_elemental_dofs(eid)
        for eid in range(mixed_element.mesh.n_elements)
    ]).astype(np.int32)

    total_dofs = dof_handler.total_dofs

    for name in sorted(required):
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

        # ---- coefficient vectors / element-local blocks ------------------
        elif name.startswith("u_") and name.endswith("_loc"):
            func_name = name[2:-4]          # strip 'u_'   and '_loc'
            f = func_map.get(func_name)
            if f is None:
                raise NameError(
                    f"Kernel requests coefficient array for unknown Function "
                    f"or VectorFunction '{func_name}'."
                )

            # build once – padding to GLOBAL length ensures safe gather
            full_vec = np.zeros(total_dofs, dtype=np.float64)
            for gdof, lidx in f._g2l.items():
                full_vec[gdof] = f.nodal_values[lidx]

            args[name] = full_vec[gdofs_map]

        # ---- constant arrays ---------------------------------------------
        elif name in const_arrays:
            args[name] = const_arrays[name].value
        
        # ---- element-wise constants ---------------------------------------------
        elif name.startswith("ewc_"):
            obj_id = int(name.split("_", 1)[1])
            ewc = next(c for c in _find_all(expression, ElementWiseConstant)
                    if id(c) == obj_id)
            args[name] = np.asarray(ewc.values, dtype=np.float64)
        
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

        else:
            logger.warning(
                f"[build_jit_kernel_args] unrecognised kernel parameter '{name}' "
                "– skipped."
            )

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


