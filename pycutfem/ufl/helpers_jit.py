import numpy as np
from typing import Mapping, Tuple, Dict, Any
from pycutfem.integration import volume
from pycutfem.fem import transform

def _pad_coeffs(coeffs, phi, ctx):
    """Return coeffs padded to the length of `phi` on ghost edges."""
    if phi.shape[0] == coeffs.shape[0] or "global_dofs" not in ctx:
        return coeffs                       # interior element – nothing to do

    padded = np.zeros_like(phi)
    side   = '+' if ctx.get("phi_val", 0.0) >= 0 else '-'
    amap   = ctx["pos_map"] if side == '+' else ctx["neg_map"]
    padded[amap] = coeffs
    return padded

def _precompute_physical_quadrature(mesh, q_order):
    """
    Return
        qp_phys : (nel, n_qp, dim)
        qw_phys : (nel, n_qp)                        *already* includes |detJ|
        detJ    : (nel, n_qp)
        J_inv   : (nel, n_qp, dim, dim)

    Geometry is gathered once per element, once per quadrature point,
    so the JIT kernel never touches the mesh object.
    """
    dim         = mesh.spatial_dim
    n_elements  = mesh.n_elements
    qp_ref, qw_ref = volume(mesh.element_type, q_order)     # reference rule
    n_qp        = len(qp_ref)

    # --- allocate output ------------------------------------------------
    qp_phys = np.empty((n_elements, n_qp, dim),  dtype=np.float64)
    detJ    = np.empty((n_elements, n_qp),       dtype=np.float64)
    J_inv   = np.empty((n_elements, n_qp, dim, dim), dtype=np.float64)

    # physical weights = w_q * |detJ|
    qw_phys = np.empty((n_elements, n_qp),       dtype=np.float64)

    
    # --- loop over elements & quad points ------------------------------
    # (vectorising across *all* elements costs memory, so a double loop
    #  is usually faster / leaner at this stage.)
    for eid in range(n_elements):
        for q, (xi, *rest) in enumerate(qp_ref):   # rest = eta,(zeta)
            # transform.jacobian() must accept both 2-D and 3-D ξ
            xi_tuple = (xi, *rest)

            J_q   = transform.jacobian(mesh, eid, xi_tuple)
            det   = np.linalg.det(J_q)
            invJ  = np.linalg.inv(J_q)

            detJ[eid, q]   = det
            J_inv[eid, q]  = invJ
            qp_phys[eid, q] = transform.x_mapping(mesh, eid, xi_tuple)
            qw_phys[eid, q] = qw_ref[q] * det          # already physical

    return qp_phys, qw_phys, detJ, J_inv

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
    zero-padded coefficient vectors for exactly the symbols that the JIT
    kernel requests.
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

    # -----------------------------------------------------------------
    # 1. reference-element quadrature (ξ-space)
    # -----------------------------------------------------------------
    qp_ref, _ = volume(mixed_element.mesh.element_type, q_order)

    # -----------------------------------------------------------------
    # 2. find all Functions / VectorFunctions appearing in *expression*
    # -----------------------------------------------------------------
    func_map: Dict[str, Any] = {}
    for f in _find_all(expression, Function):
        func_map[f.name] = f                     # p_k
        func_map.setdefault(f.field_name, f)     # p
    
    # VectorFunction → add alias for every component’s field
    for vf in _find_all(expression, VectorFunction):
        func_map[vf.name] = vf                   # u_k
        for comp, fld in zip(vf.components, vf.field_names):
            func_map.setdefault(fld, vf)         # ux, uy

    # constant tensors that need a NumPy array
    const_arrays = {
        f"const_arr_{id(c)}": c
        for c in _find_all(expression, UflConst) if c.dim != 0
    }

    # -----------------------------------------------------------------
    # 3. collect the kernel’s required argument *names*
    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    # 0.  Which symbols does the generated kernel require?
    # -----------------------------------------------------------------
    if param_order is not None:
        required_args = set(param_order)           # ← exactly as used later
    else:
        required_args = set()

    # 3.a – scan the IR for basis / derivative symbols and coeffs
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
                required_args.add(f"u_{op.name}_coeffs")

        elif isinstance(op, LoadConstantArray):
            required_args.add(op.name)

    # 3.b – gradients referenced directly in the expression tree
    for g in _find_all(expression, Grad):
        op = g.operand
        if hasattr(op, "field_names"):             # vector
            required_args.update(f"g_{f}" for f in op.field_names)
        else:                                      # scalar
            required_args.add(f"g_{op.field_name}")

    # -----------------------------------------------------------------
    # 4. build the *numeric* arrays
    # -----------------------------------------------------------------
    for name in sorted(required_args):
        if name in prebuilt: continue

        # ---- basis --------------------------------------------------
        if name.startswith("b_"):
            fld = name[2:]
            vals = [mixed_element.basis(fld, *xi_eta) for xi_eta in qp_ref]
            args[name] = np.asarray(vals, dtype=np.float64)            # (n_qp,n_loc)

        # ---- gradient ----------------------------------------------
        elif name.startswith("g_"):
            fld = name[2:]
            grads = [mixed_element.grad_basis(fld, *xi_eta) for xi_eta in qp_ref]
            args[name] = np.asarray(grads, dtype=np.float64)           # (n_qp,n_loc,dim)

        # ---- higher-order partials d<ax><ay>_field ------------------
        elif name.startswith("d"):
            dtag, fld = name.split("_", 1)
            ax, ay = int(dtag[1]), int(dtag[2])
            deriv = [
                mixed_element.deriv_ref(fld, *xi_eta, ax, ay)
                for xi_eta in qp_ref
            ]
            args[name] = np.asarray(deriv, dtype=np.float64)           # (n_qp,n_loc)

        # ---- zero-padded coefficient arrays ------------------------
        elif name.startswith("u_") and name.endswith("_coeffs"):
            # strip 'u_' prefix and '_coeffs' suffix
            func_name = name[2:-7]
            f = func_map.get(func_name)
            if f is None:
                raise NameError(
                    f"Coefficient array requested for unknown Function "
                    f"or VectorFunction '{func_name}'."
                )

            coeffs = np.zeros(dof_handler.total_dofs, dtype=np.float64)

            def _fill_slice(field_name, src_vals):
                sl = dof_handler.get_field_slice(field_name)
                coeffs[sl] = src_vals

            # scalar Function ----------------------------------------------
            if not hasattr(f, "field_names"):
                _fill_slice(f.field_name, f.nodal_values)

            # VectorFunction -----------------------------------------------
            else:
                for idx, fld in enumerate(f.field_names):
                    comp_vals = f.components[idx].nodal_values
                    _fill_slice(fld, comp_vals)

            args[name] = coeffs

        # ---- constant tensors --------------------------------------
        else:  # LoadConstantArray
            args[name] = const_arrays[name].value

    return args