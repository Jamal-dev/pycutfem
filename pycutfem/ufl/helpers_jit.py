import numpy as np
from typing import Mapping, Tuple, Dict, Any
from pycutfem.integration import volume
from pycutfem.fem import transform
import logging # Added for logging warnings

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



def _build_jit_kernel_args(       # ← NEW signature
        ir,                       # linear IR produced by the visitor
        expression,               # the UFL integrand
        mixed_element,            # MixedElement instance
        q_order: int,             # quadrature order
        dof_handler,               # DofHandler – *needed* for padding
        param_order = None  # order of parameters in the JIT kernel
    ) -> Dict[str, Any]:
    """
    Allocate reference-space basis / gradient / derivative arrays **and**
    coefficient vectors for exactly the symbols that the JIT kernel requests.
    """
    # -- late imports to avoid circular dependencies -------------------
    from pycutfem.ufl.expressions import (
        Function, VectorFunction, Constant as UflConst, Grad
    )
    from pycutfem.jit.ir import LoadVariable, LoadConstantArray
    from pycutfem.ufl.compilers import _find_all

    # Symbols that the caller (FormCompiler) has already prepared
    prebuilt = {
        'gdofs_map', 'node_coords', 'element_nodes',
        'qp_phys', 'qw', 'detJ', 'J_inv', 'normals', 'phis'
    }
    args: Dict[str, Any] = {}

    # 1. reference-element quadrature (ξ-space)
    qp_ref, _ = volume(mixed_element.mesh.element_type, q_order)

    # 2. find all Functions / VectorFunctions appearing in *expression*
    func_map: Dict[str, Any] = {}
    for f in _find_all(expression, Function):
        func_map[f.name] = f
        func_map.setdefault(f.field_name, f)

    for vf in _find_all(expression, VectorFunction):
        func_map[vf.name] = vf
        for comp, fld in zip(vf.components, vf.field_names):
            func_map.setdefault(fld, vf)

    const_arrays = {
        f"const_arr_{id(c)}": c
        for c in _find_all(expression, UflConst) if c.dim != 0
    }

    # 3. collect the kernel’s required argument *names*
    if param_order is not None:
        required_args = set(param_order)
    else:
        required_args = set()

    for op in ir:
        if isinstance(op, LoadVariable):
            field_names = (
                op.field_names if op.is_vector and op.field_names else [op.name]
            )
            for f in field_names:
                if op.deriv_order == (0, 0):
                    required_args.add(f"b_{f}")
                else:
                    d0, d1 = op.deriv_order
                    required_args.add(f"d{d0}{d1}_{f}")

            if op.role == "function":
                # CHANGE: Default to requesting the new high-performance `_loc` arrays.
                required_args.add(f"u_{op.name}_loc")

        elif isinstance(op, LoadConstantArray):
            required_args.add(op.name)

    for g in _find_all(expression, Grad):
        op = g.operand
        if hasattr(op, "field_names"):
            required_args.update(f"g_{f}" for f in op.field_names)
        else:
            required_args.add(f"g_{op.field_name}")

    # 4. build the *numeric* arrays
    for name in sorted(required_args):
        if name in prebuilt: continue

        if name.startswith("b_"):
            fld = name[2:]
            vals = [mixed_element.basis(fld, *xi_eta) for xi_eta in qp_ref]
            args[name] = np.asarray(vals, dtype=np.float64)

        elif name.startswith("g_"):
            fld = name[2:]
            grads = [mixed_element.grad_basis(fld, *xi_eta) for xi_eta in qp_ref]
            args[name] = np.asarray(grads, dtype=np.float64)

        elif name.startswith("d"):
            dtag, fld = name.split("_", 1)
            ax, ay = int(dtag[1]), int(dtag[2])
            deriv = [
                mixed_element.deriv_ref(fld, *xi_eta, ax, ay)
                for xi_eta in qp_ref
            ]
            args[name] = np.asarray(deriv, dtype=np.float64)

        # CHANGE: This block now handles both _loc (new) and _coeffs (old) conventions.
        elif name.startswith("u_") and (name.endswith("_loc") or name.endswith("_coeffs")):
            is_local_request = name.endswith("_loc")

            if is_local_request:
                func_name = name[2:-4]  # e.g., 'u_u_k_loc' -> 'u_k'
            else:
                func_name = name[2:-7]  # e.g., 'u_u_k_coeffs' -> 'u_k'

            f = func_map.get(func_name)
            if f is None:
                raise NameError(
                    f"Coefficient array requested for unknown Function "
                    f"or VectorFunction '{func_name}'."
                )

            # NEW LOGIC: Always build the full global vector first.
            full_coeffs_vec = np.zeros(dof_handler.total_dofs, dtype=np.float64)
            for gdof, lidx in f._g2l.items():
                full_coeffs_vec[gdof] = f.nodal_values[lidx]

            if is_local_request:
                # If the kernel wants a local array, gather it now.
                gdofs_map = np.vstack([
                    dof_handler.get_elemental_dofs(eid)
                    for eid in range(mixed_element.mesh.n_elements)
                ]).astype(np.int32)
                args[name] = full_coeffs_vec[gdofs_map]
            else:
                # Otherwise, provide the full global vector for backward compatibility.
                args[name] = full_coeffs_vec

        else:
            if name in const_arrays:
                args[name] = const_arrays[name].value
            else:
                logger.warning(f"Argument builder skipped unrecognized kernel parameter: '{name}'")

    return args