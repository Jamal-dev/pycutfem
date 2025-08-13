# pycutfem/jit/codegen.py
import textwrap
from dataclasses import dataclass, field, replace


from pycutfem.jit.ir import (
    LoadVariable, LoadConstant, LoadConstantArray, LoadElementWiseConstant,
    LoadAnalytic, LoadFacetNormal, Grad, Div, PosOp, NegOp,
    BinaryOp, Inner, Dot, Store, Transpose, CellDiameter, LoadFacetNormalComponent, CheckDomain,
    Trace, Hessian as IRHessian, Laplacian as IRLaplacian
)
from pycutfem.jit.symbols import encode
import numpy as np
import re


# Numba is imported inside the generated kernel string
# import numba



@dataclass(frozen=True, slots=True)
class StackItem:
    """Holds metadata for an item on the code generation stack."""
    var_name: str
    # role can be 'test', 'trial', or 'value' (a computed quantity)
    role: str
    # shape of the data: e.g., () for scalar, (k,) for vector value,
    # (n,) for scalar basis, (k, n) for vector basis,
    # (n, d) for scalar grad basis, (k, n, d) for vector grad basis
    shape: tuple
    is_gradient: bool = field(default=False)
    is_vector: bool = field(default=False)
    # Stores the field name(s) to look up basis functions or coefficients
    field_names: list = field(default_factory=list)
    # Stores the name of the parent Function/VectorFunction
    parent_name: str = ""
    side: str = ""  # Side for ghost integrals, e.g., "+", "-", or ""
    # tiny shim so we can write  item = item._replace(var_name="tmp")
    is_transpose: bool = field(default=False)  # True if this item is a transposed version of another
    is_hessian: bool = field(default=False)  # True if this item is a Hessian matrix
    def _replace(self, **changes) -> "StackItem":
        
        return replace(self, **changes)


# ── scalar * Vector/Tensor  (or Vector/Tensor * scalar) ────────────
def _mul_scalar_vector(self,first_is_scalar,res_var,a, b, body_lines, stack):
    scalar = a if first_is_scalar else b
    vect   = b if first_is_scalar else a

    if scalar.shape != ():
        raise ValueError(f"Scalar operand must truly be scalar shapes: {getattr(scalar, 'shape', None)} / {getattr(vect, 'shape', None)},"
                         f" role: {getattr(scalar, 'role', None)}, {getattr(vect, 'role', None)}"
                         f", is_vector: {getattr(scalar, 'is_vector', None)}, {getattr(vect, 'is_vector', None)}"
                         f", is_gradient: {getattr(scalar, 'is_gradient', None)}, {getattr(vect, 'is_gradient', None)}"
                         f", Bilinear form" if self.form_rank == 2 else ", Linear form")

    res_shape = vect.shape
    # Collapse (1,n) only for rank-1 or rank-0 forms
    collapse = (self.form_rank < 2 and
                len(vect.shape) == 2 and vect.shape[0] == 1)
    lhs = f"{scalar.var_name}"
    rhs = f"{vect.var_name}[0]" if collapse else f"{vect.var_name}"
    body_lines.append(f"{res_var} = {lhs} * {rhs}")

    if collapse:
        res_shape = (vect.shape[1],)            # (n,)

    stack.append(StackItem(var_name   = res_var,
                           role       = vect.role,
                           shape      = res_shape,
                           is_vector  = vect.is_vector,
                           is_gradient= vect.is_gradient,
                           field_names= vect.field_names,
                           parent_name= vect.parent_name))

class NumbaCodeGen:
    """
    Translates a linear IR sequence into a Numba-Python kernel source string.
    """
    def __init__(self, nopython: bool = True, mixed_element=None,form_rank=0):
        """
        Initializes the code generator.
        
        Args:
            nopython (bool): Whether to use nopython mode for Numba.
            mixed_element: An INSTANCE of the MixedElement class, not the class type.
        """
        self.nopython = nopython
        self.form_rank = form_rank # rank of the form (0 for scalar, 1 for linearform, 2 for bilinear form etc.)
        if mixed_element is None:
            raise ValueError("NumbaCodeGen requires an instance of a MixedElement.")
        self.me = mixed_element
        self.n_dofs_local = self.me.n_dofs_local
        self.spatial_dim = self.me.mesh.spatial_dim
        self.last_side_for_store = "" # Track side for detJ
        self.dtype = "np.float64"  # Default data type for arrays

    
    
    def generate_source(self, ir_sequence: list, kernel_name: str):
        body_lines = []
        stack = []
        var_counter = 0
        required_args = set()
        functional_shape = None # () for scalar, (k,) for vector, 
        # Track names of Function/VectorFunction objects that provide coefficients
        solution_func_names = set()

        def new_var(prefix="tmp"):
            nonlocal var_counter
            var_counter += 1
            return f"{prefix}_{var_counter}"

        # --- Main IR processing loop ---
        for op in ir_sequence:

            if isinstance(op, LoadFacetNormal):
                required_args.add("normals")
                var_name = new_var("nrm")
                body_lines.append(f"{var_name} = normals[e, q]")
                stack.append(StackItem(var_name=var_name,
                                        role='const',
                                        shape=(self.spatial_dim,),
                                        is_vector=True))
            elif isinstance(op, LoadFacetNormalComponent):
                required_args.add("normals")
                var_name = new_var("nrm_c")
                body_lines.append(f"{var_name} = normals[e, q, {op.idx}]")
                stack.append(StackItem(var_name=var_name,
                                       role='const',
                                       shape=(),          # scalar
                                       is_vector=False))
            
            elif isinstance(op, CheckDomain):
                a = stack.pop()
                out = new_var("restricted")

                bs = f"domain_bs_{op.bitset_id}"
                required_args.add(bs)
                required_args.add("owner_id")          # <— NEW

                body_lines.append(f"# Restriction via {bs}")
                # Compute a safe global element id, no exceptions
                body_lines.append("eid = owner_id[e]  # global element id OR owner-element id for facets")
                body_lines.append(f"flag = (0 <= eid < {bs}.shape[0]) and {bs}[eid]")
                body_lines.append(f"{out} = {a.var_name} if flag else np.zeros_like({a.var_name}, dtype={self.dtype})")

                stack.append(a._replace(var_name=out))
            

            elif isinstance(op, IRHessian):
                a = stack.pop()  # carries .role, .field_names, .side

                # choose per-side symbols
                if   a.side == "+": jinv = "J_inv_pos"; H0 = "pos_Hxi0"; H1 = "pos_Hxi1"; suff = "_pos"
                elif a.side == "-": jinv = "J_inv_neg"; H0 = "neg_Hxi0"; H1 = "neg_Hxi1"; suff = "_neg"
                else:               jinv = "J_inv";     H0 = "Hxi0";     H1 = "Hxi1";     suff = ""

                required_args.update({jinv, H0, H1})
                required_args.add("gdofs_map")  # needed for union width on facets

                # need reference derivative tables (volume: d.._; facet: r.._pos/neg)
                d10, d01, d20, d11, d02 = [], [], [], [], []
                for fn in a.field_names:
                    name10 = (f"d10_{fn}" if suff=="" else f"r10_{fn}{suff}")
                    name01 = (f"d01_{fn}" if suff=="" else f"r01_{fn}{suff}")
                    name20 = (f"d20_{fn}" if suff=="" else f"r20_{fn}{suff}")
                    name11 = (f"d11_{fn}" if suff=="" else f"r11_{fn}{suff}")
                    name02 = (f"d02_{fn}" if suff=="" else f"r02_{fn}{suff}")
                    required_args.update({name10, name01, name20, name11, name02})
                    d10.append(name10); d01.append(name01); d20.append(name20); d11.append(name11); d02.append(name02)

                out = new_var("Hess")
                body_lines.append(f"{out} = []  # (k, n_union, 2, 2)")

                # Build local Hessians per field, then (if needed) scatter to union layout
                for i, fn in enumerate(a.field_names):
                    Hloc = new_var("Hloc")
                    body_lines += [
                        f"A  = {jinv}[e, q]",       # (2,2)
                        f"Hx = {H0}[e, q]",         # (2,2) = H_{xi0}
                        f"Hy = {H1}[e, q]",         # (2,2) = H_{xi1}"
                        f"d10_q = {d10[i]}[e, q]",  # (n_loc,)
                        f"d01_q = {d01[i]}[e, q]",
                        f"d20_q = {d20[i]}[e, q]",
                        f"d11_q = {d11[i]}[e, q]",
                        f"d02_q = {d02[i]}[e, q]",
                        "nloc = d20_q.shape[0]",
                        f"{Hloc} = np.empty((nloc, 2, 2), dtype=d20_q.dtype)",
                        "for j in range(nloc):",
                        "    Href = np.array([[d20_q[j], d11_q[j]],[d11_q[j], d02_q[j]]], dtype=d20_q.dtype)",
                        "    gref = np.array([d10_q[j], d01_q[j]], dtype=d20_q.dtype)",
                        "    core = A.T @ (Href @ A)",
                        "    Hphys = core + gref[0]*Hx + gref[1]*Hy",
                        f"    {Hloc}[j] = Hphys",
                    ]

                    if a.side:
                        map_arr = "pos_map" if a.side == "+" else "neg_map"
                        required_args.add(map_arr)
                        Hpad = new_var("Hpad"); me = new_var("map_e")
                        body_lines += [
                            f"{me} = {map_arr}[e]",                       # (n_loc,)
                            "n_union = gdofs_map[e].shape[0]",
                            f"{Hpad} = np.zeros((n_union, 2, 2), dtype=d20_q.dtype)",
                            "for j in range(nloc):",
                            f"    idx = {me}[j]",
                            "    if 0 <= idx < n_union:",
                            f"        {Hpad}[idx] = {Hloc}[j]",
                            f"{out}.append({Hpad})",
                        ]
                    else:
                        body_lines += [f"{out}.append({Hloc})"]

                # stack components, final shape = (k, n_union, 2, 2)
                body_lines.append(f"{out} = np.stack({out}, axis=0)")
                stack.append(StackItem(var_name={out}, role=a.role, shape=(-1, -1, 2, 2),
                                    is_vector=False, field_names=a.field_names, side=a.side, is_hessian=True))





            
            elif isinstance(op, IRLaplacian):
                a = stack.pop()

                if   a.side == "+": jinv = "J_inv_pos"; H0 = "pos_Hxi0"; H1 = "pos_Hxi1"; suff = "_pos"
                elif a.side == "-": jinv = "J_inv_neg"; H0 = "neg_Hxi0"; H1 = "neg_Hxi1"; suff = "_neg"
                else:               jinv = "J_inv";     H0 = "Hxi0";     H1 = "Hxi1";     suff = ""
                required_args.update({jinv, H0, H1})
                required_args.add("gdofs_map")

                # derivative tables (volume: d.._; facet: r.._pos/neg)
                d10, d01, d20, d11, d02 = [], [], [], [], []
                for fn in a.field_names:
                    n10 = (f"d10_{fn}" if suff=="" else f"r10_{fn}{suff}")
                    n01 = (f"d01_{fn}" if suff=="" else f"r01_{fn}{suff}")
                    n20 = (f"d20_{fn}" if suff=="" else f"r20_{fn}{suff}")
                    n11 = (f"d11_{fn}" if suff=="" else f"r11_{fn}{suff}")
                    n02 = (f"d02_{fn}" if suff=="" else f"r02_{fn}{suff}")
                    required_args.update({n10, n01, n20, n11, n02})
                    d10.append(n10); d01.append(n01); d20.append(n20); d11.append(n11); d02.append(n02)

                out = new_var("Lap")
                body_lines.append(f"{out} = []  # (k, n_union)")

                for i, fn in enumerate(a.field_names):
                    laploc = new_var("laploc")
                    body_lines += [
                        f"A  = {jinv}[e, q]",
                        f"Hx = {H0}[e, q]",
                        f"Hy = {H1}[e, q]",
                        f"d10_q = {d10[i]}[e, q]",
                        f"d01_q = {d01[i]}[e, q]",
                        f"d20_q = {d20[i]}[e, q]",
                        f"d11_q = {d11[i]}[e, q]",
                        f"d02_q = {d02[i]}[e, q]",
                        "nloc = d20_q.shape[0]",
                        f"{laploc} = np.empty((nloc,), dtype=d20_q.dtype)",
                        "tHx = Hx[0,0] + Hx[1,1]",
                        "tHy = Hy[0,0] + Hy[1,1]",
                        "for j in range(nloc):",
                        "    Href = np.array([[d20_q[j], d11_q[j]],[d11_q[j], d02_q[j]]], dtype=d20_q.dtype)",
                        "    gref = np.array([d10_q[j], d01_q[j]], dtype=d20_q.dtype)",
                        "    core = A.T @ (Href @ A)",
                        "    tr_core = core[0,0] + core[1,1]",
                        "    lap_j = tr_core + gref[0]*tHx + gref[1]*tHy",
                        f"    {laploc}[j] = lap_j",
                    ]

                    if a.side:
                        map_arr = "pos_map" if a.side == "+" else "neg_map"
                        required_args.add(map_arr)
                        lap_pad = new_var("lap_pad"); me = new_var("map_e")
                        body_lines += [
                            f"{me} = {map_arr}[e]",                      # (n_loc,)
                            "n_union = gdofs_map[e].shape[0]",
                            f"{lap_pad} = np.zeros((n_union,), dtype=d20_q.dtype)",
                            "for j in range(nloc):",
                            f"    idx = {me}[j]",
                            "    if 0 <= idx < n_union:",
                            f"        {lap_pad}[idx] = {laploc}[j]",
                            f"{out}.append({lap_pad})",
                        ]
                    else:
                        body_lines += [
                            f"{out}.append({laploc})",
                        ]

                body_lines.append(f"{out} = np.stack({out}, axis=0)")  # (k, n_union)
                stack.append(StackItem(var_name={out}, role=a.role, shape=(-1,-1), is_vector=True,
                                    field_names=a.field_names, side=a.side))


            
            elif isinstance(op, LoadElementWiseConstant):
                # the full (n_elem, …) array is passed as a kernel argument
                required_args.add(op.name)
                required_args.add("owner_id")          # <— NEW
                var_name = new_var("ewc")
                body_lines.append(f"{var_name} = {op.name}[owner_id[e]]")

                is_vec = len(op.tensor_shape) == 1
                stack.append(
                    StackItem(var_name=var_name,
                            role='value',
                            shape=op.tensor_shape,       # real shape, not ()
                            is_vector=is_vec)
                )
            # --- analytic (pre-tabulated) ---------------------------------------------
            elif isinstance(op, LoadAnalytic):
                param = f"ana_{op.func_id}"                # unique name in PARAM_ORDER
                required_args.add(param)
                var_name = new_var("ana")
                body_lines.append(f"{var_name} = {param}[e, q]")
                stack.append(StackItem(var_name=var_name,
                                    role='const',
                                    shape=(), is_vector=False))
            # --- LOAD OPERATIONS ---
            # ---------------------------------------------------------------------------
            # LOADVARIABLE –– basis tables and coefficient look-ups
            # ---------------------------------------------------------------------------
            elif isinstance(op, LoadVariable):
                # ------------------------------------------------------------------
                # 1. Common set‑up --------------------------------------------------
                # ------------------------------------------------------------------
                deriv_order = op.deriv_order
                field_names = op.field_names

                # *Reference* tables are side‑agnostic (no suffix)
                side_suffix_basis = ""

                # helper: "dxy_" or "b_" + field name
                def get_basis_arg_name(field_name, deriv):
                    d_str = f"d{deriv[0]}{deriv[1]}" if deriv != (0, 0) else "b"
                    return f"{d_str}_{field_name}{side_suffix_basis}"

                basis_arg_names = [get_basis_arg_name(fname, deriv_order) for fname in field_names]
                for arg_name in basis_arg_names:
                    required_args.add(arg_name)

                # Which J⁻¹ to use for push‑forward / derivatives
                if   op.side == "+":  jinv_sym = "J_inv_pos"; required_args.add("J_inv_pos")
                elif op.side == "-":  jinv_sym = "J_inv_neg"; required_args.add("J_inv_neg")
                else:                 jinv_sym = "J_inv";      required_args.add("J_inv")

                # reference value(s) at current quadrature point
                basis_vars_at_q = [f"{arg_name}[e, q]" for arg_name in basis_arg_names]

                # ------------------------------------------------------------------
                # 2. Test / Trial functions  ---------------------------------------
                # ------------------------------------------------------------------
                # print(f"Visiting LoadVariable: {op.name}, role={op.role}, side={op.side}, "
                #     f"deriv_order={deriv_order}, field_names={field_names}, "
                #     f"is_vector={getattr(op, 'is_vector', None)}, "
                #     f"is_gradient={getattr(op, 'is_gradient', None)}")

                if op.role in ("test", "trial"):

                    # ---------- facet integrals (+ / -) : pad to union DOFs ----------
                    if op.side:                                              # "+" or "-"
                        map_array_name = "pos_map" if op.side == "+" else "neg_map"
                        required_args.add(map_array_name)

                        map_e = new_var(f"{map_array_name}_e")
                        body_lines += [
                            f"{map_e} = {map_array_name}[e]",
                            f"n_union = gdofs_map[e].shape[0]",              # e.g. 36 for Stokes
                        ]

                        padded_vars = []                                     # one per component
                        for i, bq in enumerate(basis_vars_at_q):
                            loc = new_var(f"local_basis{i}")
                            pad = new_var(f"padded_basis{i}")
                            body_lines += [
                                f"{loc} = {bq}",
                                f"if n_union == {loc}.shape[0]:",          # fast path (interior facet)
                                f"    {pad} = {loc}.copy()",
                                f"else:",                                  # true ghost facet
                                f"    {pad} = np.zeros(n_union, dtype={self.dtype})",
                                f"    for j in range({loc}.shape[0]):",
                                f"        idx = {map_e}[j]",
                                f"        if 0 <= idx < n_union:",
                                f"            {pad}[idx] = {loc}[j]",
                            ]
                            padded_vars.append(pad)

                        basis_vars_at_q = padded_vars        # hand off the padded list
                        final_basis_var = padded_vars[0]     # for scalar reshape path
                        n_dofs = -1                          # run‑time on ghost facets

                    # ---------- volume / interface -----------------------------------
                    else:
                        final_basis_var = basis_vars_at_q[0]
                        n_dofs = self.me.n_dofs_local

                    # ---------- reshape (scalar) or stack (vector) -------------------
                    if not op.is_vector:                               # scalar basis
                        var_name = new_var("basis_reshaped")
                        body_lines.append(f"{var_name} = {final_basis_var}[np.newaxis, :].copy()")
                        shape = (1, n_dofs)
                    else:                                              # vector basis
                        var_name = new_var("basis_stack")
                        body_lines.append(f"{var_name} = np.stack(({', '.join(basis_vars_at_q)}))")
                        shape = (len(field_names), n_dofs)

                    # ---------- push‑forward for ∂‑ordered bases ---------------------
                    if deriv_order != (0, 0):
                        dx, dy = deriv_order
                        jinv_q = f"{jinv_sym}_q"                       # J_inv_pos_q, …
                        scale  = f"({jinv_q}[0,0]**{dx}) * ({jinv_q}[1,1]**{dy})"
                        body_lines.append(f"{var_name} *= {scale}")

                    # ---------- push onto stack --------------------------------------
                    stack.append(
                        StackItem(
                            var_name    = var_name,
                            role        = op.role,
                            shape       = shape,
                            is_vector   = op.is_vector,
                            field_names = field_names,
                            parent_name = op.name
                        )
                    )

                # ------------------------------------------------------------------
                # 3. Coefficient / Function values  –– scalar **and** vector
                # ------------------------------------------------------------------
                elif op.role == "function":
                    # --------------------------------------------------------------
                    # 3‑A  Which coefficient array do we need?  (single array)
                    # --------------------------------------------------------------
                    from pycutfem.jit.symbols import POS_SUFFIX, NEG_SUFFIX
                    coeff_side = POS_SUFFIX if op.side == "+" else NEG_SUFFIX if op.side == "-" else ""

                    if op.name.startswith("u_") and op.name.endswith("_loc"):
                        coeff_sym = op.name[:-4] + f"{coeff_side}_loc"
                    else:
                        coeff_sym = f"u_{op.name}{coeff_side}_loc"

                    required_args.add(coeff_sym)
                    solution_func_names.add(coeff_sym)

                    # --------------------------------------------------------------
                    # 3‑B  Pad reference bases to the DOF‑union on ghost facets
                    # --------------------------------------------------------------
                    if op.side:                                            # "+" or "-"
                        map_array_name = "pos_map" if op.side == "+" else "neg_map"
                        required_args.add(map_array_name)

                        map_e = new_var(f"{map_array_name}_e")
                        body_lines += [
                            f"{map_e} = {map_array_name}[e]",
                            f"n_union = gdofs_map[e].shape[0]",
                        ]

                        padded = []
                        for i, b_var in enumerate(basis_vars_at_q):
                            local = new_var(f"local_basis{i}")
                            pad   = new_var(f"padded_basis{i}")
                            body_lines += [
                                f"{local} = {b_var}",
                                # fast path: interior facet (no ghosts)
                                f"if n_union == {local}.shape[0]:",
                                f"    {pad} = {local}.copy()",
                                f"else:",                                     # genuine ghost facet
                                f"    {pad} = np.zeros(n_union, dtype={self.dtype})",
                                f"    for j in range({local}.shape[0]):",
                                f"        idx = {map_e}[j]",
                                f"        if 0 <= idx < n_union:",
                                f"            {pad}[idx] = {local}[j]",
                            ]
                            padded.append(pad)

                        basis_vars_at_q = padded             # hand padded list forward
                    # volume / interface: basis_vars_at_q already fine

                    # --------------------------------------------------------------
                    # 3‑C  Evaluate u_h(x_q)  (scalar or vector)
                    # --------------------------------------------------------------
                    val_var = new_var(f"{op.name}_val")

                    if op.is_vector:
                        # one dot product per component, same coefficient array
                        comps = [f"np.dot({b_var}, {coeff_sym})" for b_var in basis_vars_at_q]
                        body_lines.append(f"{val_var} = np.array([{', '.join(comps)}])")
                        shape = (len(field_names),)           # e.g. (2,)
                    else:
                        body_lines.append(f"{val_var} = np.dot({basis_vars_at_q[0]}, {coeff_sym})")
                        shape = ()

                    # --------------------------------------------------------------
                    # 3‑D  Optional ξ‑derivative rescaling
                    # --------------------------------------------------------------
                    if deriv_order != (0, 0):
                        dx, dy = deriv_order
                        jinv_q = f"{jinv_sym}_q"
                        body_lines.append(
                            f"{val_var} *= ({jinv_q}[0,0]**{dx}) * ({jinv_q}[1,1]**{dy})"
                        )

                    # --------------------------------------------------------------
                    # 3‑E  Push onto stack
                    # --------------------------------------------------------------
                    stack.append(
                        StackItem(
                            var_name    = val_var,
                            role        = "value",
                            shape       = shape,
                            is_vector   = op.is_vector,
                            field_names = field_names,
                            parent_name = coeff_sym
                        )
                    )

                else:
                    raise TypeError(f"Unknown role '{op.role}' for LoadVariable IR node.")

            elif isinstance(op, LoadConstant):
                stack.append(StackItem(var_name=str(op.value), role='const', shape=(), is_vector=False, field_names=[]))
            
            elif isinstance(op, LoadConstantArray):
                required_args.add(op.name)
                # The constant array is passed in as a list. Convert it to a
                # NumPy array inside the kernel for Numba compatibility.
                np_array_var = new_var("const_np_arr")
                is_vec = len(op.shape) == 1
                # body_lines.append(f"{np_array_var} = np.array({op.name}, dtype=np.float64)")
                body_lines.append(f"{np_array_var} = {op.name}")
                stack.append(StackItem(var_name=np_array_var, role='const', shape=op.shape, is_vector=is_vec, field_names=[]))
            
            elif isinstance(op, CellDiameter):
                required_args.add("owner_id")          # <— NEW
                required_args.add("h_arr")             # ensures builder injects global element-length sizes
                res = new_var("h")
                body_lines.append(f"{res} = h_arr[owner_id[e]]")

                stack.append(StackItem(var_name=res,
                                    shape=(),
                                    is_vector=False,
                                    role='const',
                                    is_gradient=False))
                if "h_arr" not in required_args:
                    required_args.add("h_arr")


            # ------------------------------------------------------------------
            # Trace operator --------
            # ------------------------------------------------------------------
            elif isinstance(op, Trace):
                a = stack.pop()
                res_var = new_var("trace_res")

                # --- Case 1: Trace of a computed value/constant (e.g., shape (2, 2)) ---
                if len(a.shape) == 2:
                    # First, validate that the matrix is square.
                    if a.shape[0] != a.shape[1]:
                        raise ValueError(f"Trace requires a square matrix, but got shape {a.shape}")
                    
                    # If valid, generate the execution code.
                    body_lines.append(f"# Trace of a computed matrix -> scalar value")
                    body_lines.append(f"{res_var} = np.trace({a.var_name})")
                    stack.append(StackItem(var_name=res_var,
                                            role='value',
                                            shape=(), # Result is a scalar
                                            is_vector=False,
                                            is_gradient=False,
                                            field_names=[]))

                # --- Case 2: Trace of a Test/Trial function tensor (e.g., shape (2, n, 2)) ---
                elif len(a.shape) == 3:
                    # First, validate that the tensor is square over its first and last dimensions.
                    if a.shape[0] != a.shape[2]:
                        raise ValueError(f"Trace requires a square tensor (k=d), but got shape {a.shape}")

                    # If valid, generate the execution code.
                    body_lines.append(f"# Trace of a symbolic tensor -> scalar basis of shape (1, n)")
                    k, n, _ = a.shape
                    body_lines += [
                        f"n_locs = {n}",
                        f"n_comps = {k}",
                        f"{res_var} = np.zeros((1, n_locs), dtype={self.dtype})",
                        f"for j in range(n_locs):",
                        f"    local_sum = 0.0",
                        f"    for i in range(n_comps):",
                        f"        local_sum += {a.var_name}[i, j, i]",
                        f"    {res_var}[0, j] = local_sum"
                    ]
                    stack.append(StackItem(var_name=res_var,
                                            role=a.role,
                                            shape=(1, n),
                                            is_vector=False,
                                            is_gradient=False,
                                            field_names=a.field_names,
                                            parent_name=a.parent_name))
                
                # --- Else: The shape is not a 2D or 3D tensor, so it's invalid. ---
                else:
                    raise TypeError(f"Cannot take trace of an operand with shape {a.shape}. Must be a 2D or 3D tensor.")
            
            # --- UNARY OPERATORS ---
            # ----------------------------------------------------------------------
            # ∇(·) operator
            # ----------------------------------------------------------------------
            elif isinstance(op, Grad):
                a = stack.pop()

                # --- which J⁻¹ array to use -------------------------------------------
                if   a.side == "+":  jinv_sym = "J_inv_pos"; required_args.add("J_inv_pos")
                elif a.side == "-":  jinv_sym = "J_inv_neg"; required_args.add("J_inv_neg")
                else:                jinv_sym = "J_inv";     required_args.add("J_inv")

                jinv_q = f"{jinv_sym}_q"          # ← element‑, qp‑local  (2×2)

                # --- reference‑gradient tables (no side suffix) -----------------------
                grad_q = [f"g_{f}" for f in a.field_names]
                for name in grad_q: required_args.add(name)

                # ======================================================================
                # (A)  grad(Test/Trial)           → (k , n_loc , 2)
                # ======================================================================
                if a.role in ("test", "trial"):
                    phys = []
                    for i in range(len(a.field_names)):
                        pg_loc = new_var("grad_loc")
                        body_lines.append(f"{pg_loc} = {grad_q[i]}[e, q] @ {jinv_q}.copy()")

                        if a.side:        # ---------- NEW ----------
                            map_arr = "pos_map" if a.side == "+" else "neg_map"
                            required_args.add(map_arr)
                            map_e   = new_var(f"{map_arr}_e")
                            pg_pad  = new_var("grad_pad")
                            body_lines += [
                                f"{map_e} = {map_arr}[e]",
                                f"n_union = gdofs_map[e].shape[0]",
                                f"{pg_pad} = np.zeros((n_union, {self.spatial_dim}), dtype={self.dtype})",
                                f"for j in range({pg_loc}.shape[0]):",
                                f"    idx = {map_e}[j]",
                                f"    if 0 <= idx < n_union:",
                                f"        {pg_pad}[idx] = {pg_loc}[j]",
                            ]
                            phys.append(pg_pad)
                        else:
                            phys.append(pg_loc)

                    n_dofs = self.n_dofs_local if not a.side else -1
                    if not a.is_vector:                                # scalar space
                        var = new_var("grad_scalar")
                        body_lines.append(f"{var} = {phys[0]}[None, :, :].copy()")
                        shape = (1, n_dofs, self.spatial_dim)
                        is_vector = False
                        is_gradient = True
                    else:                                              # vector space
                        var = new_var("grad_stack")
                        body_lines.append(f"{var} = np.stack(({', '.join(phys)}))")
                        shape = (len(a.field_names), n_dofs, self.spatial_dim)
                        is_vector = False
                        is_gradient = True

                    stack.append(a._replace(var_name=var,
                                            shape=shape,
                                            is_gradient=is_gradient,is_vector = is_vector))

                # ======================================================================
                # (B)  grad(Function/VectorFunction)  → (2,)  or  (k , 2)
                # ======================================================================
                elif a.role == "value":
                    coeff = (a.parent_name if a.parent_name.startswith("u_")
                            else f"u_{a.parent_name}_loc")

                    comps = []
                    for i in range(len(a.field_names)):
                        pg  = new_var("phys_grad_basis")
                        val = new_var("grad_val")
                        body_lines += [
                            f"{pg}  = {grad_q[i]}[e, q] @ {jinv_q}.copy()",   # (n_loc,2)
                            f"{val} = {pg}.T.copy() @ {coeff}",           # (2,)
                        ]
                        comps.append(val)

                    if not a.is_vector:               # scalar coefficient
                        var, shape = comps[0], (self.spatial_dim,)
                        is_vector = True
                        is_gradient = False
                    else:                             # k‑vector coefficient
                        var = new_var("grad_val_stack")
                        body_lines.append(f"{var} = np.stack(({', '.join(comps)}))")
                        shape = (len(a.field_names), self.spatial_dim)
                        is_vector = False
                        is_gradient = True

                    stack.append(
                        StackItem(var_name   = var,
                                role       = "value",
                                shape      = shape,
                                is_gradient= is_gradient,
                                is_vector  = is_vector,
                                field_names= a.field_names,
                                parent_name= coeff)
                    )
                else:
                    raise NotImplementedError(f"Grad not implemented for role {a.role}")

            
            elif isinstance(op, Div):
                a = stack.pop()

                if not a.is_gradient:
                    raise TypeError("Div can only be applied to a gradient quantity.")

                div_var = new_var("div")

                # ---------------------------------------------------------------
                # 1)  Divergence of basis gradients (Test / Trial)  →  scalar (1,n)
                #     a.var_name shape: (k , n_loc , d)
                # ---------------------------------------------------------------
                # print(f"Div: a.shape={a.shape}, a.role={a.role},a.is_vector={a.is_vector}, a.is_gradient={a.is_gradient}")
                if a.role in ("test", "trial") and a.shape[0] ==2 and a.is_gradient:
                    # print("hello "*10)
                    body_lines.append("# Div(basis) → scalar basis (1,n_loc)")

                    body_lines += [
                        f"n_loc  = {a.shape[1]}",  # number of local basis functions
                        f"n_vec  = {a.shape[0]}",     # components k
                        f"{div_var} = np.zeros((1, n_loc), dtype={self.dtype})",
                        f"for j in range(n_loc):",            # local basis index
                        f"    tmp = 0.0",
                        f"    for k in range(n_vec):",        # component index
                        f"        tmp += {a.var_name}[k, j, k]",   #  ∂φ_k / ∂x_k
                        f"    {div_var}[0, j] = tmp",
                    ]

                    stack.append(
                        StackItem(
                            var_name=div_var,
                            role=a.role,
                            shape=(1, a.shape[1]),            # (1 , n_loc)
                            is_vector=False,
                            field_names=a.field_names,
                            parent_name=a.parent_name,
                            is_gradient=False
                        )
                    )

                # ---------------------------------------------------------------
                # 2)  Divergence of a gradient VALUE  (Function / VectorFunction)
                #     a.var_name shape: (k , d)
                # ---------------------------------------------------------------
                elif a.role == "value":
                    k_comp, n_dim = a.shape        # (k , d)   both known at code-gen time

                    if k_comp == n_dim:            # vector field → scalar
                        body_lines.append("# Div(vector value) → scalar")
                        body_lines.append(f"{div_var} = 0.0")
                        for k in range(k_comp):
                            body_lines.append(f"{div_var} += {a.var_name}[{k}, {k}]")
                        stack.append(StackItem(var_name=div_var, role='value',
                                            shape=(), is_vector=False, is_gradient=False,
                                            field_names=a.field_names))
                    else:                          # tensor field → vector (k,)
                        body_lines.append("# Div(tensor value) → vector")
                        body_lines.append(f"{div_var} = np.zeros(({k_comp},), dtype={self.dtype})")
                        for k in range(k_comp):
                            body_lines.append(f"tmp = 0.0")
                            for d in range(n_dim):
                                body_lines.append(f"tmp += {a.var_name}[{k}, {d}]")
                            body_lines.append(f"{div_var}[{k}] = tmp")
                        stack.append(StackItem(var_name=div_var, role='value',
                                            shape=(k_comp,), is_vector=True, is_gradient=False,
                                            field_names=a.field_names))

                # ---------------------------------------------------------------
                # 3)  Anything else is not defined
                # ---------------------------------------------------------------
                else:
                    raise NotImplementedError(
                        f"Div not implemented for role '{a.role}' with gradient=True."
                    )

            elif isinstance(op, PosOp):
                a = stack.pop()
                res_var = new_var("pos")
                body_lines.append(f"{res_var} = {a.var_name} if phi_q >= 0.0 else np.zeros_like({a.var_name}, dtype={self.dtype})")
                stack.append(a._replace(var_name=res_var))

            elif isinstance(op, NegOp):
                a = stack.pop()
                res_var = new_var("neg")
                body_lines.append(f"{res_var} = {a.var_name} if phi_q < 0.0 else np.zeros_like({a.var_name}, dtype={self.dtype})")
                stack.append(a._replace(var_name=res_var))

            # --- Inner OPERATORS ---
            elif isinstance(op, Inner):
                b = stack.pop(); a = stack.pop()
                res_var = new_var("inner")
                # print(f"Inner operation: a.role={a.role}, b.role={b.role}, a.shape={a.shape}, b.shape={b.shape}"
                #       f", is_vector: {a.is_vector}/{b.is_vector}, is_gradient: {a.is_gradient}/{b.is_gradient}"
                #       f", a.is_transpose: {a.is_transpose}, b.is_transpose: {b.is_transpose}")

                # LHS Bilinear Form: always rows=test, cols=trial
                if a.role in ('test', 'trial') and b.role in ('test', 'trial'):
                    body_lines.append('# Inner(LHS): orient rows=test, cols=trial')

                    # pick the operands by role, regardless of stack order
                    test_var  = f"{a.var_name}" if a.role == "test"  else f"{b.var_name}"
                    trial_var = f"{a.var_name}" if a.role == "trial" else f"{b.var_name}"
                    n_test    = a.shape[1] if a.role == "test"  else b.shape[1]
                    n_trial   = a.shape[1] if a.role == "trial" else b.shape[1]

                    body_lines.append(f'{res_var} = np.zeros(({n_test}, {n_trial}), dtype={self.dtype})')

                    if a.is_gradient and b.is_gradient:
                        # grad-grad: sum over vector components
                        k_comps = a.shape[0]  # == b.shape[0]
                        body_lines.append(f'for k in range({k_comps}):')
                        # test[k] @ trial[k].T → (n_test, n_trial)
                        body_lines.append(f'    {res_var} += {test_var}[k] @ {trial_var}[k].T.copy()')
                    else:
                        # vector bases (or scalar lifted to (1,n)): test.T @ trial
                        body_lines.append(f'{res_var} = {test_var}.T.copy() @ {trial_var}')

                    # push with correct matrix shape
                    stack.append(StackItem(var_name=res_var, role="value",
                                        shape=(n_test, n_trial), is_vector=False,
                                        is_gradient=False))
                    continue

                # elif a.role == 'const' and b.role == 'const' and a.shape == b.shape:
                #     body_lines.append(f'# Inner(Const, Const): element-wise product')
                #     body_lines.append(f'{res_var} = {a.var_name} * {b.var_name}')
                elif a.role == 'value' and b.role == 'test': # RHS
                    body_lines.append(f'# RHS: Inner(Function, Test)')
                    # a is (k,d) , b is (k,n,d), 
                    if a.is_gradient and b.is_gradient:
                        # a is (k,d) and b is  (k,n,d)  --> (n)
                        body_lines.append(f'# RHS: Inner(Grad(Function), Grad(Test))')
                        body_lines += [
                            f'n_locs = {b.shape[1]}; n_vec_comps = {a.shape[0]};',
                            # f'print(f"a.shape: {{{a.var_name}.shape}}, b.shape: {{{b.var_name}.shape}}")',
                            f'{res_var} = np.zeros((n_locs), dtype={self.dtype})',
                            f'for n in range(n_locs):',
                            f"    {res_var}[n] = np.sum({a.var_name} * {b.var_name}[:,n,:].copy())"
                        ]
                    elif a.is_vector and b.is_vector:
                        body_lines.append(f'# RHS: Inner(Function, Test)')
                        # a is (k), b is (k,n) -> (n,)
                        # New Newton: Optimized manual dot product loop
                        body_lines += [
                            f'n_locs = {b.shape[1]}',
                            f'n_vec_comps = {b.shape[0]}',
                            f'{res_var} = np.zeros(n_locs, dtype={self.dtype})',
                            f'for n in range(n_locs):',
                            f'    local_sum = 0.0',
                            f'    for k in range(n_vec_comps):',
                            f'        local_sum += {a.var_name}[k] * {b.var_name}[k, n]',
                            f'    {res_var}[n] = local_sum',
                        ]
                    elif a.is_vector and b.is_gradient:
                        # Case: grad(scalarFunction)->(k,), grad(Test)->(k,n,d) --> (n,)
                        body_lines.append(f'# RHS: Inner(grad(scalarFunction), grad(Test)')
                        body_lines += [
                            f'n_locs = {b.shape[1]}; n_vec_comps = {a.shape[0]};',
                            f'{res_var} = np.zeros((n_locs), dtype={self.dtype})',
                            f'for n in range(n_locs):',
                            f"    {res_var}[n] = np.sum({a.var_name} * {b.var_name}[:,n,:].copy())"
                        ]
                    else:
                        raise NotImplementedError(f"Inner not implemented for roles {a.role}/{b.role}, is_vector: {a.is_vector}/{b.is_vector}, is_gradient: {a.is_gradient}/{b.is_gradient}, is_gradient: {b.is_gradient},shapes: {a.shape}/{b.shape}")
                        
                    
                elif a.role == 'value' and b.role == 'value':
                    if a.is_vector and b.is_vector:
                        body_lines.append(f'# Inner(Value, Value): dot product')
                        body_lines.append(f'{res_var} = np.dot({a.var_name}, {b.var_name})')
                    if a.is_gradient and b.is_gradient:
                        body_lines += [
                            f'# Inner(Grad(Value), Grad(Value)): stiffness matrix',
                            f'# (k,d) @ (k,d) -> (k,k)',
                            f'{res_var} = {a.var_name} @ {b.var_name}.T.copy()',]
                
                else:
                    raise NotImplementedError(f"JIT Inner not implemented for roles {a.role}/{b.role}, is_vector: {a.is_vector}/{b.is_vector}, is_gradient: {a.is_gradient}/{b.is_gradient}, is_gradient: {b.is_gradient}, shapes: {a.shape}/{b.shape}")
                stack.append(StackItem(var_name=res_var, role='value', shape=(), is_vector=False, field_names=[]))

            # ------------------------------------------------------------------
            # DOT   — special-cased branches for advection / mass terms --------
            # ------------------------------------------------------------------
            elif isinstance(op, Dot):
                b = stack.pop()
                a = stack.pop()
                res_var = new_var("dot")

                # print(f"Dot operation: a.role={a.role}, b.role={b.role}, a.shape={a.shape}, b.shape={b.shape}, is_vector: {a.is_vector}/{b.is_vector}, is_gradient: {a.is_gradient}/{b.is_gradient}")

                # Advection term: dot(grad(u_trial), u_k)
                if a.role == 'trial' and a.is_gradient and b.role == 'value' and b.is_vector:
                    body_lines.append(f"# Advection: dot(grad(Trial), Function)")
                    # body_lines.append(f"{res_var} = np.einsum('knd,d->kn', {a.var_name}, {b.var_name})")
                    body_lines += [
                        f"n_vec_comps = {a.shape[0]};n_locs = {a.shape[1]};n_spatial_dim = {a.shape[2]};",
                        f"{res_var} = np.zeros((n_vec_comps, n_locs), dtype={self.dtype})",
                        f"for k in range(n_vec_comps):",
                        f"    {res_var}[k] = {b.var_name} @ {a.var_name}[k].T.copy() ",
                        # f"assert {res_var}.shape == (2, 22), f'result shape mismatch {res_var}.shape with {{(n_vec_comps, n_locs)}}'"
                    ]
                    stack.append(StackItem(var_name=res_var, role='trial', shape=(a.shape[0], a.shape[1]), is_vector=True, is_gradient=False, field_names=a.field_names, parent_name=a.parent_name))
               
                # Final advection term: dot(advection_vector_trial, v_test)
                elif (a.role == 'trial' and (not a.is_gradient) and b.role == 'test' and (not b.is_gradient)):
                     body_lines.append(f"# Mass: dot(Trial, Test)")
                    #  body_lines.append(f"assert ({a.var_name}.shape == (2,22) and {b.var_name}.shape == (2,22)), 'Trial and Test to have the same shape'")
                     body_lines.append(f"{res_var} = {b.var_name}.T.copy() @ {a.var_name}")
                     stack.append(StackItem(var_name=res_var, 
                                            role='value', 
                                            shape=(b.shape[1],a.shape[1]), 
                                            is_vector=False, field_names=[]))
                elif (a.role == 'test' and (not a.is_gradient) and b.role == 'trial' and (not b.is_gradient)):
                    body_lines.append(f"# Mass: dot(Test, Trial)")
                    # body_lines.append(f"assert ({a.var_name}.shape == (2,22) and {b.var_name}.shape == (2,22)), 'Trial and Test to have the same shape'")
                    body_lines.append(f"{res_var} = {a.var_name}.T.copy() @ {b.var_name}")
                    stack.append(StackItem(var_name=res_var, 
                                            role='value', 
                                            shape=(a.shape[1],b.shape[1]), 
                                            is_vector=False, field_names=[]))
                
                # ---------------------------------------------------------------------
                # dot( grad(u_test) ,  const_vec )  ← symmetric term -> Test vec
                # ---------------------------------------------------------------------
                elif a.role == 'test' and a.is_gradient and b.role == 'const' and b.is_vector:
                    if a.shape[2] == b.shape[0]:
                        body_lines.append("# Symmetric term: dot(grad(Test), constant vector)")
                        body_lines += [
                                        # robust on ghost facets (pos/neg padding)
                                        f"n_vec_comps   = {a.var_name}.shape[0];",
                                        f"n_locs        = {a.var_name}.shape[1];",
                                        f"n_spatial_dim = {a.var_name}.shape[2];",
                                        f"{res_var} = np.zeros((n_vec_comps, n_locs), dtype={self.dtype})",
                                        f"for k in range(n_vec_comps):",
                                        f"    for d in range(n_spatial_dim):",
                                        f"        {res_var}[k] += {a.var_name}[k, :, d] * {b.var_name}[d]",

                        ]
                        stack.append(StackItem(var_name=res_var, role='test',
                                            shape=(a.shape[0], a.shape[1]), is_vector=True,
                                            is_gradient=False, field_names=a.field_names,
                                            parent_name=a.parent_name))

                # ---------------------------------------------------------------------
                # dot( grad(u_trial) ,  beta )  ← convection term (Function gradient · Trial)
                # ---------------------------------------------------------------------
                elif a.role == 'trial' and a.is_gradient and b.role == 'const' and b.is_vector:
                    if b.shape[0] == a.shape[2]:
                        body_lines.append("# Advection: dot(grad(Trial), constant beta vector)")
                        body_lines += [
                            f"n_vec_comps   = {a.var_name}.shape[0];",
                            f"n_locs        = {a.var_name}.shape[1];",
                            f"n_spatial_dim = {a.var_name}.shape[2];",
                            f"{res_var} = np.zeros((n_vec_comps, n_locs), dtype={self.dtype})",
                            f"for k in range(n_vec_comps):",
                            f"    for d in range(n_spatial_dim):",
                            f"        {res_var}[k] += {a.var_name}[k, :, d] * {b.var_name}[d]",
                        ]
                        stack.append(StackItem(var_name=res_var, role='trial',
                                            shape=(a.shape[0], a.shape[1]), is_vector=True,
                                            is_gradient=False, field_names=a.field_names,
                                            parent_name=a.parent_name))
                elif b.role == 'trial' and b.is_gradient and a.role == 'const' and a.is_vector:
                    if a.shape[0] == b.shape[2]:
                        body_lines.append("# Advection: dot(constant beta vector, grad(Trial))")
                        body_lines += [
                            f"n_vec_comps = {b.shape[0]}; n_locs = {b.shape[1]};n_spatial_dim = {b.shape[2]};",
                            f"{res_var} = np.zeros((n_vec_comps, n_locs), dtype={self.dtype})",
                            f"for k in range(n_vec_comps):",
                            f"    for d in range(n_spatial_dim):",
                            f"        {res_var}[k] += {b.var_name}[k, :, d] * {a.var_name}[d]",
                        ]
                        stack.append(StackItem(var_name=res_var, role='trial',
                                            shape=(b.shape[0], b.shape[1]), is_vector=True,
                                            is_gradient=False, field_names=b.field_names,
                                            parent_name=b.parent_name))
                
                
                # ---------------------------------------------------------------------
                # dot( grad(u_k) ,  u_trial )  ← convection term (Function gradient · Trial)
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_gradient and b.role == 'trial' and b.is_vector:
                    body_lines.append("# Advection: dot(grad(Function), Trial)")
                    body_lines += [
                        f"{res_var}   = {a.var_name} @ {b.var_name}",
                        
                    ]
                    stack.append(StackItem(var_name=res_var, role='trial',
                                        shape=(b.shape[0], b.shape[1]), is_vector=True,
                                        is_gradient=False, field_names=b.field_names,
                                        parent_name=b.parent_name))

                # ---------------------------------------------------------------------
                # dot( u_trial ,  grad(u_k) )   ← swap of the previous
                # ---------------------------------------------------------------------
                elif a.role == 'trial' and a.is_vector and b.role == 'value' and b.is_gradient:
                    body_lines.append("# Advection: dot(Trial, grad(Function))")
                    body_lines += [
                        f"{res_var} = {b.var_name} @ {a.var_name}",
                    ]
                    stack.append(StackItem(var_name=res_var, role='trial',
                                        shape=(a.shape[0], a.shape[1]), is_vector=True,
                                        is_gradient=False, field_names=a.field_names,
                                        parent_name=a.parent_name))

                # ---------------------------------------------------------------------
                # dot( u_k ,  u_k )             ← |u_k|², scalar
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_vector and b.role == 'value' and b.is_vector:
                    body_lines.append("# Non-linear term: dot(Function, Function)")
                    body_lines.append(f"{res_var} = np.dot({a.var_name}, {b.var_name})")
                    stack.append(StackItem(var_name=res_var, role='value',
                                        shape=(), is_vector=False))

                # ---------------------------------------------------------------------
                # dot( u_k ,  grad(u_trial) )   ← usually zero for skew-symm forms
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_vector and b.role == 'trial' and b.is_gradient:
                    body_lines.append("# dot(Function, grad(Trial))")
                    body_lines += [
                        f"n_vec_comps = {b.shape[0]};",
                        f"n_locs      = {b.shape[1]};",
                        f"{res_var}   = np.zeros((n_vec_comps, n_locs), dtype={self.dtype})",
                        # einsum: f"{res_var} = np.einsum('d,kld->kl', {a.var_name}, {b.var_name})",
                        f"for k in range(n_vec_comps):",
                        f"    {res_var}[k] = {a.var_name} @ {b.var_name}[k].T.copy()",
                    ]
                    stack.append(StackItem(var_name=res_var, role='trial',
                                        shape=(b.shape[0], b.shape[1]), is_vector=True,
                                        is_gradient=False, field_names=b.field_names,
                                        parent_name=b.parent_name))

                # ---------------------------------------------------------------------
                # dot( u_test ,  u_trial )      ← mass-matrix block
                # ---------------------------------------------------------------------
                elif a.role == 'test' and a.is_vector and b.role == 'trial' and b.is_vector:
                    body_lines.append("# Mass: dot(Test, Trial)")
                    body_lines.append(f"{res_var} = {a.var_name}.T.copy() @ {b.var_name}")
                    stack.append(StackItem(var_name=res_var, role='value',
                                        shape=(a.shape[1],b.shape[1]), is_vector=False))
                    

                # ---------------------new block--------------------------------
                # ---------------------------------------------------------------------
                # dot(grad(Trial), grad(Function)) and its transposed variants.
                # ---------------------------------------------------------------------
                elif (a.role == 'trial' and a.is_gradient and b.role == 'value' 
                    and b.is_gradient 
                    and a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1]): 
                    k = a.shape[0]; n_locs = a.shape[1]; d = a.shape[2]
                    
                    # a: grad(du) or grad(du).T -> Trial function basis, shape (k, n, d)
                    # b: grad(u_k)             -> Function value, shape (k, d)
                    

                    body_lines.append("# dot(grad(trial), grad(value)) -> (k,n,k) tensor basis")
                    body_lines += [
                        f"n_vec_comps = {k}; n_locs = {n_locs};",
                        f"{res_var} = np.zeros((n_vec_comps, n_locs, n_vec_comps), dtype={self.dtype})",
                        f"b_mat = np.ascontiguousarray({b.var_name})",
                        f"for n in range(n_locs):",
                        f"    a_slice = np.ascontiguousarray({a.var_name}[:, n, :])",
                        f"    {res_var}[:, n, :] = a_slice @ b_mat",
                    ]
                    
                    res_shape = (k, n_locs, d)
                    stack.append(StackItem(var_name=res_var, role='trial',
                                        shape=res_shape, is_vector=False, is_gradient=True,
                                        field_names=a.field_names, parent_name=a.parent_name))

                # ---------------------------------------------------------------------
                # dot(grad(Function), grad(Trial)) and its transposed variants.
                # ---------------------------------------------------------------------
                elif (a.role == 'value' and a.is_gradient and 
                    b.role == 'trial' and b.is_gradient and 
                    a.shape[0] == b.shape[0] and a.shape[1] == b.shape[2]):
                    k = b.shape[0]; n_locs = b.shape[1]; d = b.shape[2]

                    # a: grad(u_k) or grad(u_k).T -> Function value, shape (k, d)
                    # b: grad(du)             -> Trial function basis, shape (k, n, d)

                    body_lines.append("# dot(grad(value), grad(trial)) -> (k,n,k) tensor basis")
                    body_lines += [
                        f"n_vec_comps = {k}; n_locs = {n_locs};",
                        f"{res_var} = np.zeros((n_vec_comps, n_locs, {d}), dtype={self.dtype})",
                        f"a_contig = np.ascontiguousarray({a.var_name})",
                        f"for n in range(n_locs):",
                        f"    b_slice = np.ascontiguousarray({b.var_name}[:, n, :])",
                        f"    {res_var}[:, n, :] = a_contig @ b_slice",
                    ]
                    
                    res_shape = (k, n_locs, k)
                    stack.append(StackItem(var_name=res_var, role='trial',
                                        shape=res_shape, is_vector=False, is_gradient=True,
                                        field_names=b.field_names, parent_name=b.parent_name))

                # ---------------------------------------------------------------------
                # dot(grad(Function), grad(Function)) and its transposed variants.
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_gradient and b.role == 'value' and b.is_gradient:
                    # a: grad(u_k) or grad(u_k).T -> Function value, shape (k, d)
                    # b: grad(u_k) or grad(u_k).T -> Function value, shape (k, d)
                    # This block handles various combinations like dot(A, B), dot(A.T, B), etc.
                    # The generated code assumes k == d.
                    body_lines.append("# dot(grad(value), grad(value)) -> (k,k) tensor value")
                    body_lines.append(f"{res_var} = {a.var_name} @ {b.var_name}")
                    res_shape = (a.shape[0], a.shape[1])
                    stack.append(StackItem(var_name=res_var, role='value',
                                        shape=res_shape, is_vector=False, is_gradient=True,
                                        field_names=b.field_names, parent_name=b.parent_name))

                # ---------------------new block--------------------------------
                # ---------------------------------------------------------------------
                # dot( scalar ,  u_trial;u_test;u_k )     ← e.g. scalar constant time Function
                # ---------------------------------------------------------------------
                elif (a.role == 'const' or a.role == 'value') and not a.is_vector and not a.is_gradient:
                    # a is scalar, b is vector (trial/test/Function)
                    body_lines.append("# Scalar constant: dot(scalar, Function/Trial/Test)")
                    body_lines.append(f"{res_var} = float({a.var_name}) * {b.var_name}")
                    stack.append(StackItem(var_name=res_var, role=b.role,
                                        shape=b.shape, is_vector=b.is_vector,
                                        is_gradient=b.is_gradient, field_names=b.field_names,
                                        parent_name=b.parent_name))
                # ---------------------------------------------------------------------
                # dot( u_trial;u_test;u_k, scalar )     ← e.g. scalar constant time Function
                # ---------------------------------------------------------------------
                elif (b.role == 'const' or b.role == 'value') and not b.is_vector and not b.is_gradient:
                    # a is vector (trial/test/Function), b is scalar
                    body_lines.append("# Scalar constant: dot(Function/Trial/Test, scalar)")
                    body_lines.append(f"{res_var} = float({b.var_name}) * {a.var_name}")
                    stack.append(StackItem(var_name=res_var, role=a.role,
                                        shape=a.shape, is_vector=a.is_vector,
                                        is_gradient=a.is_gradient, field_names=a.field_names,
                                        parent_name=a.parent_name))
                
                # ---------------------------------------------------------------------
                # dot( u_k ,  grad(u_k) )     ← e.g. rhs advection term
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_vector and b.role == 'value' and b.is_gradient:
                    body_lines.append("# RHS: dot(Function, grad(Function)) (k).(k,d) ->k")
                    body_lines += [
                        f"{res_var}   = {b.var_name} @ {a.var_name}",
                    ]
                    stack.append(StackItem(var_name=res_var, role='const',
                                        shape=( a.shape[0],), is_vector=True,
                                        is_gradient=False, field_names=[]))
                # ---------------------------------------------------------------------
                # dot( grad(u_k) ,  u_k )     ← e.g. rhs advection term  -> (k,d).(k) -> k
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_gradient and b.role == 'value' and b.is_vector:
                    body_lines.append("# RHS: dot(grad(Function), Function) (k,d).(k) -> k")
                    body_lines += [
                        f"{res_var}   = {a.var_name} @ {b.var_name}",
                    ]
                    stack.append(StackItem(var_name=res_var, role='const',
                                        shape=(a.shape[0], ), is_vector=True,
                                        is_gradient=False, field_names=[]))
                # ---------------------------------------------------------------------
                # dot( np.array ,  u_test )     ← e.g. body-force · test -> (n,)
                # ---------------------------------------------------------------------
                elif a.role == 'const' and a.is_vector and b.role == 'test' and b.is_vector:
                    # a (k) and b (k,n)
                    body_lines.append("# Constant body-force: dot(const-vec, Test)")
                    # New Newton: Optimized manual dot product loop
                    body_lines += [
                        f"n_locs = {b.shape[1]}",
                        f"n_vec_comps = {a.shape[0]}",
                        f"{res_var} = np.zeros(n_locs, dtype={self.dtype})",
                        f"for n in range(n_locs):",
                        f"    local_sum = 0.0",
                        f"    for k in range(n_vec_comps):",
                        f"        local_sum += {a.var_name}[k] * {b.var_name}[k, n]",
                        f"    {res_var}[n] = local_sum",
                    ]
                    stack.append(StackItem(var_name=res_var, role='value',
                                        shape=(b.shape[1],), is_vector=False))
                
                # ---------------------------------------------------------------------
                # dot( u_k ,  u_test )          ← load-vector term -> (n,)
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_vector and b.role == 'test' and b.is_vector:
                    if self.form_rank == 2:
                        body_lines.append("# LHS: dot(Function, Test) (k,n)·(k) -> (1,n)")
                        shape = (1, b.shape[1])
                        role = 'test'
                        body_lines += [f"{res_var} = np.zeros((1,{b.shape[1]}), dtype={self.dtype})",
                                        f"for n in range({b.shape[1]}):",
                                        f"    {res_var}[0, n] = np.sum({a.var_name}[:, n] * {b.var_name})"]
                    elif self.form_rank == 1:
                        body_lines.append("# RHS: dot(Function, Test)")
                        body_lines.append(f"{res_var} = {a.var_name}.T.copy() @ {b.var_name}")
                        shape = (b.shape[1],)
                        role = 'value'
                    stack.append(StackItem(var_name=res_var, role=role,
                                        shape=shape, is_vector=False,is_gradient=False))

                # ---------------------------------------------------------------------
                # dot( u_test ,  u_k )          ← load-vector term -> (n,)
                # ---------------------------------------------------------------------
                elif a.role == 'test' and a.is_vector and b.role == 'value' and b.is_vector:
                    if self.form_rank == 2:
                        body_lines.append("# LHS: dot(Test, Function) (k,n)·(k) -> (1,n)")
                        shape = (1, a.shape[1])
                        role = 'test'
                        body_lines += [f"{res_var} = np.zeros((1,{a.shape[1]}), dtype={self.dtype})",
                                        f"for n in range({a.shape[1]}):",
                                        f"    {res_var}[0, n] = np.sum({a.var_name}[:, n] * {b.var_name})"]
                    elif self.form_rank == 1:
                        body_lines.append("# RHS: dot(Test, Function)")
                        body_lines.append(f"{res_var} = {b.var_name}.T.copy() @ {a.var_name}")
                        shape = (a.shape[1],)
                        role = 'value'
                    stack.append(StackItem(var_name=res_var, role=role,
                                        shape=shape, is_vector=False,is_gradient=False))
                # ---------------------------------------------------------------------
                # dot( u_test ,  const_vec )          ← load-vector term -> (n,)
                # ---------------------------------------------------------------------
                elif a.role == 'test' and a.is_vector and b.role == 'const' and b.is_vector:
                    if self.form_rank == 2:
                        body_lines.append("# LHS: dot(Test, Const) (k,n)·(k) -> (1,n)")
                        shape = (1, a.shape[1])
                        role = 'test'
                        body_lines += [f"{res_var} = np.zeros((1,{a.shape[1]}), dtype={self.dtype})",
                                        f"for n in range({a.shape[1]}):",
                                        f"    {res_var}[0, n] = np.sum({a.var_name}[:, n] * {b.var_name})"]
                        stack.append(StackItem(var_name=res_var, role=role,
                                            shape=shape, is_vector=False,is_gradient=False))
                    elif self.form_rank == 1:
                        body_lines.append("# RHS: dot(Test, Const)")
                        body_lines.append(f"{res_var} = {a.var_name}.T.copy() @ {b.var_name}")
                        shape = (a.shape[1],)
                        role = 'value'
                        stack.append(StackItem(var_name=res_var, role=role,
                                            shape=shape, is_vector=False,is_gradient=False))
                # ---------------------------------------------------------------------
                # dot( u_trial ,  const_vec )          ← load-vector term -> (1,n)
                # ---------------------------------------------------------------------
                elif a.role == 'trial' and a.is_vector and b.role == 'const' and b.is_vector:
                   body_lines.append("# LHS: dot(Trial, Const) (k,n)·(k) -> (1,n)")
                   shape = (1, a.shape[1])
                   role = 'trial'
                   body_lines += [f"{res_var} = np.zeros((1,{a.shape[1]}), dtype={self.dtype})",
                                   f"for n in range({a.shape[1]}):",
                                   f"    {res_var}[0, n] = np.sum({a.var_name}[:, n] * {b.var_name})"]
                   stack.append(StackItem(var_name=res_var, role=role,
                                        shape=shape, is_vector=False,is_gradient=False))
   

                # ---------------------------------------------------------------------
                # dot( value/const ,  value/const )          ← load-vector term -> (n,)
                # ---------------------------------------------------------------------
                elif (a.role in ('const', 'value') and   
                     b.role in ('const', 'value') ):
                    if a.is_gradient and b.is_vector:
                        body_lines.append("# Dot: grad(scalar) * const vector → const vector")
                        # print(f" a.shape: {a.shape}, b.shape: {b.shape}, a.is_vector: {a.is_vector}, b.is_vector: {b.is_vector}, a.is_gradient: {a.is_gradient}, b.is_gradient: {b.is_gradient}")
                        if a.shape == b.shape:
                            body_lines.append(f"{res_var} = np.dot({a.var_name} , {b.var_name})")
                            shape = ()
                            is_vector = False; is_grad = False
                        else:
                            body_lines.append(f"{res_var} = {a.var_name} @ {b.var_name}")
                            shape = (a.shape[0],)
                            is_vector = True; is_grad = False
                    elif a.is_vector and b.is_gradient:
                        body_lines.append("# Dot: const vector * grad(scalar) → const vector")
                        if a.shape == b.shape:
                            body_lines.append(f"{res_var} = np.dot({a.var_name} , {b.var_name})")
                            shape = ()
                            is_vector = False; is_grad = False
                        else:
                            body_lines.append(f"{res_var} = {a.var_name} @ {b.var_name}")
                            shape = (b.shape[0],)
                            is_vector = True; is_grad = False
                    elif a.is_vector and b.is_vector:
                        body_lines.append("# Dot: vector * vector → scalar")
                        body_lines.append(f"{res_var} = np.dot({a.var_name}, {b.var_name})")
                        shape = ()
                        is_vector = False; is_grad = False
                    elif a.is_gradient and  b.is_gradient:
                        body_lines.append("# Dot: grad(scalar) * grad(scalar) → scalar")
                        body_lines.append(f"{res_var} = np.dot({a.var_name}, {b.var_name})")
                        shape = ()
                        is_vector = False; is_grad = False
                    stack.append(StackItem(var_name=res_var, role='const',
                                        shape=shape, is_vector=is_vector, 
                                        is_gradient=is_grad,
                                        field_names=[]))
                
                else:
                    raise NotImplementedError(f"Dot not implemented for roles {a.role}/{b.role} with shapes {a.shape}/{b.shape}"
                                              f" with vectoors {a.is_vector}/{b.is_vector}"
                                              f" and gradients {a.is_gradient}/{b.is_gradient}"
                                              f", BilinearForm" if self.form_rank == 2 else ", LinearForm")

            elif isinstance(op, Transpose):
                a = stack.pop()                   # operand descriptor
                res = new_var("trp")

                # ---------------------------------------------------------------
                # 0) scalar  →  transpose is a no-op
                # ---------------------------------------------------------------
                if a.shape == ():
                    body_lines.append(f"{res} = {a.var_name}")     # just copy
                    res_shape = ()
                # -------- scalar grad: (1,n_qp,2)  -> (2,n_qp,1) ----------------
                # elif a.is_gradient and a.shape == (1, a.shape[1], 2):
                #     k, n, d = a.shape
                #     body_lines.append(
                #         f"{res} = {a.var_name}.swapaxes(0,2).copy()"   # (2,n,1)
                #     )
                #     res_shape = (2, n, 1)

                # -------- vector grad: (2,n_qp,2)  swap off-diagonals ------------
                elif a.is_gradient and len(a.shape) == 3:
                    n_locs = a.shape[1]
                    body_lines += [
                        f"{res} = np.zeros_like({a.var_name}, dtype={self.dtype});",
                        f"for n in range({n_locs}):",
                        f"    {res}[:, n, :] = ({a.var_name}[:, n, :].T).copy();"
                    ]
                    res_shape = a.shape  # still (2,n,2)

                # -------- plain 2×2 matrix --------------------------------------
                elif len(a.shape) == 2 :
                    body_lines.append(f"{res} = {a.var_name}.T.copy()")
                    res_shape = a.shape

                else:
                    raise NotImplementedError("Transpose not supported for shape "
                                            f"{a.shape}")

                stack.append(a._replace(var_name=res,
                                        shape=res_shape,
                                        is_vector=a.is_vector,
                                        is_gradient=a.is_gradient,
                                        is_transpose=True,
                                        field_names=a.field_names,
                                        parent_name=a.parent_name,
                                        role=a.role))

            
            elif isinstance(op, BinaryOp):
                 b = stack.pop(); a = stack.pop()
                 res_var = new_var("res")
                 # -------------------------------------
                 # ------------ PRODUCT ---------------
                 # -------------------------------------
                 if op.op_symbol == '*':
                    body_lines.append(f"# Product: {a.role} * {b.role}")
                    # -----------------------------------------------------------------
                    # 0. scalar:   scalar   *  scalar/np.ndarray    →  scalar/np.ndarray 
                    # -----------------------------------------------------------------
                    if ((a.role == 'const' or a.role=='value') and (b.role == 'const' or b.role=='value')  
                        and not a.is_vector and not b.is_vector 
                            and not a.is_gradient and not b.is_gradient):
                        body_lines.append("# Product: scalar * scalar/np.ndarry → scalar/np.ndarray")
                        if a.shape == () and b.shape == ():
                            # both are scalars
                            body_lines.append(f"{res_var} = {a.var_name} * {b.var_name}")
                            shape = ()
                        elif a.shape == () and b.shape != ():
                            # a is scalar, b is vector/tensor
                            body_lines.append(f"{res_var} = {a.var_name} * {b.var_name}")
                            shape = b.shape
                        elif b.shape == () and a.shape != ():
                            # b is scalar, a is vector/tensor
                            body_lines.append(f"{res_var} = {b.var_name} * {a.var_name}")
                            shape = a.shape
                        else:
                            # both are vectors/tensors, but not scalars
                            raise ValueError(f"Cannot multiply two non-scalar values: {a.var_name} (shape: {a.shape}) and {b.var_name} (shape: {b.shape})")

                        stack.append(StackItem(var_name=res_var, role='const', shape=shape, is_vector=False, field_names=[]))
                        
                    # -----------------------------------------------------------------
                    # 01. Vector, Tensor:   scalar   *  Vector/Tensor    →  Vector/Tensor 
                    # -----------------------------------------------------------------
                    
                    elif ((a.role == 'const' or a.role=='value')  
                         and 
                          (not a.is_vector and not a.is_gradient)) :
                        body_lines.append("# Product: scalar * Vector/Tensor → Vector/Tensor")
                        # a is scalar, b is vector/tensor
                        # if a.shape == ():
                        #     body_lines.append("# a is scalar, b is vector/tensor")
                        #     if b.role == 'test' and len(b.shape) == 2 and b.shape[0] == 1:
                        #         body_lines.append(f"{res_var} = {a.var_name} * {b.var_name}[0]")  # b is vector/tensor, a is scalar
                        #         shape = (b.shape[1],) # not true shape needs fix
                        #     else:
                        #         body_lines.append(f"{res_var} = {a.var_name} * {b.var_name}")
                        #         shape = b.shape
                        _mul_scalar_vector(self,first_is_scalar=True, a=a, b=b, res_var=res_var, body_lines=body_lines,stack=stack)
                        
                        
                    elif ((b.role == 'const' or b.role=='value')
                          and 
                        (not b.is_vector and not b.is_gradient)):
                        body_lines.append("# Product: Vector/Tensor * scalar → Vector/Tensor")
                        # b is scalar, a is vector/tensor
                        _mul_scalar_vector(self,first_is_scalar=False, a=a, b=b, res_var=res_var, body_lines=body_lines,stack=stack)
                        
                    
                    # -----------------------------------------------------------------
                    # 1. LHS block:   scalar test  *  scalar trial   →  outer product
                    # -----------------------------------------------------------------
                    elif (not a.is_gradient and not b.is_gradient and
                        ((a.role, b.role) == ("test", "trial") or
                        (a.role, b.role) == ("trial", "test"))):
                        n_locs = a.shape[1]
                        body_lines.append("# Product: scalar Test × scalar Trial → (n_loc,n_loc)")

                        # orient rows = test , columns = trial
                        test_var  = a if a.role == "test"  else b
                        trial_var = b if a.role == "test"  else a

                        body_lines += [
                            f"{res_var} = {test_var.var_name}.T.copy() @ {trial_var.var_name}",  # (n_loc, n_loc)
                        ]
                        stack.append(StackItem(var_name=res_var, role='value',
                                            shape=(n_locs,n_locs), is_vector=False))
                    # -----------------------------------------------------------------
                    # 2. RHS load:   scalar / vector Function  *  scalar Test
                    #                (u_k or c)                ·  φ_v
                    # -----------------------------------------------------------------
                    elif (b.role == "test" and not b.is_vector
                        and a.role == "value" and not a.is_vector
                        and not a.is_gradient and not b.is_gradient):
                        body_lines.append("# Load: scalar Function × scalar Test → (n_loc,)")

                        body_lines.append(f"{res_var} = {a.var_name} * {b.var_name}")   # (n_loc,)
                        stack.append(StackItem(var_name=res_var, role='value',
                                            shape=(b.shape[1],), is_vector=False))

                    # symmetric orientation
                    elif (a.role == "test" and not a.is_vector
                        and b.role == "value" and not b.is_vector):
                        body_lines.append("# Load: scalar Test × scalar Function → (n_loc,)")

                        body_lines.append(f"{res_var} = {b.var_name} * {a.var_name}")   # (n_loc,)
                        stack.append(StackItem(var_name=res_var, role='value',
                                            shape=(a.shape[1],), is_vector=False))
                    # -----------------------------------------------------------------
                    # 4. Anything else is ***not implemented yet*** – fail fast
                    # -----------------------------------------------------------------
                    else:
                        raise NotImplementedError(
                            f"Product not implemented for roles {a.role}/{b.role} "
                            f"with vector flags {a.is_vector}/{b.is_vector} "
                            f"and gradient flags {a.is_gradient}/{b.is_gradient}."
                            f" Shapes: {a.shape}/{b.shape}"
                        )
                 # ---------------------------------------------------------------------------
                 # Binary “+ / −” (element-wise) --------------------------------------------
                 # ---------------------------------------------------------------------------
                 elif op.op_symbol in ('+', '-'):
                    sym = op.op_symbol
                    body_lines.append(f"# {'Addition' if sym == '+' else 'Subtraction'}")

                    # --- utilities ---------------------------------------------------------
                    def _merge_role(ra, rb):
                        if 'trial' in (ra, rb):  return 'trial'
                        if 'test'  in (ra, rb):  return 'test'
                        if 'value' in (ra, rb):  return 'value'
                        return 'const'

                    def _broadcast_shape_with_minus1(sa, sb):
                        """
                        Like numpy.broadcast_shapes but treats -1 as a wildcard
                        (run-time size).  Compatible with NumPy’s rules otherwise.
                        """
                        from itertools import zip_longest
                        la, lb      = len(sa), len(sb)
                        max_len     = max(la, lb)
                        ra          = (1,)*(max_len-la) + sa
                        rb          = (1,)*(max_len-lb) + sb
                        out         = []
                        for da, db in zip_longest(ra, rb):
                            if da == db:               out.append(da)
                            elif da == 1:              out.append(db)
                            elif db == 1:              out.append(da)
                            elif da == -1 or db == -1: out.append(-1)      # NEW rule
                            else:
                                raise NotImplementedError(
                                    f"'{sym}' cannot broadcast shapes {sa} and {sb}"
                                )
                        # drop leading unit dims if NumPy would do so
                        while len(out) > 0 and out[0] == 1:
                            out.pop(0)
                        return tuple(out)

                    # ----------------------------------------------------------------------
                    # CASE A – broadcast with true scalar on exactly one side
                    # ----------------------------------------------------------------------
                    scalar_left  = (a.shape == () and not a.is_vector and not a.is_gradient)
                    scalar_right = (b.shape == () and not b.is_vector and not b.is_gradient)

                    # Special-case: subtracting two ghost-edge bases (both have -1 shape)
                    if (a.role in ('test', 'trial') and b.role in ('test', 'trial')
                        and -1 in a.shape and -1 in b.shape):
                        body_lines.append("# Addition / subtraction of two ghost-edge bases")
                        body_lines.append(f"{res_var} = {a.var_name} {sym} {b.var_name}")
                        stack.append(a._replace(
                            var_name = res_var,
                            role     = _merge_role(a.role, b.role)
                        ))

                    elif scalar_left ^ scalar_right:
                        # exactly one side is a scalar -> NumPy will broadcast
                        non_scal, scal = (b, a) if scalar_left else (a, b)
                        body_lines.append("# scalar broadcast with non-scalar")
                        body_lines.append(f"{res_var} = {non_scal.var_name} {sym} {scal.var_name}")
                        stack.append(StackItem(
                            var_name    = res_var,
                            role        = non_scal.role,
                            shape       = non_scal.shape,
                            is_vector   = non_scal.is_vector,
                            is_gradient = non_scal.is_gradient,
                            field_names = non_scal.field_names
                        ))

                    # ----------------------------------------------------------------------
                    # CASE B – both operands non-scalar; same vec/grad flags; broadcastable
                    # ----------------------------------------------------------------------
                    elif a.is_vector == b.is_vector and a.is_gradient == b.is_gradient:
                        try:
                            new_shape = np.broadcast_shapes(a.shape, b.shape)
                        except ValueError:
                            # NEW – allow “−1” wildcard broadcasting
                            if -1 in a.shape or -1 in b.shape:
                                new_shape = _broadcast_shape_with_minus1(a.shape, b.shape)
                            else:
                                raise

                        body_lines.append("# element-wise op with NumPy broadcasting")
                        body_lines.append(f"{res_var} = {a.var_name} {sym} {b.var_name}")
                        stack.append(StackItem(
                            var_name    = res_var,
                            role        = _merge_role(a.role, b.role),
                            shape       = new_shape,
                            is_vector   = a.is_vector,
                            is_gradient = a.is_gradient,
                            field_names = (a.field_names if a.role in ('trial', 'test')
                                        else b.field_names)
                        ))



                    # ----------------------------------------------------------------------
                    # CASE C – anything else is unsupported
                    # ----------------------------------------------------------------------
                    else:
                        raise NotImplementedError(
                            f"'{sym}' not implemented for roles {a.role}/{b.role} "
                            f"with shapes {a.shape}/{b.shape} "
                            f"or mismatched vector / gradient flags "
                            f"({a.is_vector},{b.is_vector}) – "
                            f"({a.is_gradient},{b.is_gradient}) "
                            f"names: {a.var_name}/{b.var_name}"
                        )

                 # -----------------------------------------------------------------
                 # ------------------  DIVISION  ( /  )  ---------------------------
                 # -----------------------------------------------------------------
                 elif op.op_symbol == '/':
                    body_lines.append("# Division")
                    # divide *anything* by a scalar constant (const in denominator)
                    if (b.role == 'const' or b.role == 'value') and not b.is_vector and b.shape == ():
                        body_lines.append(f"{res_var} = {a.var_name} / float({b.var_name})")
                        stack.append(StackItem(var_name=res_var, role=a.role,
                                            shape=a.shape, is_vector=a.is_vector,
                                            is_gradient=a.is_gradient, field_names=a.field_names,
                                            parent_name=a.parent_name))
                    elif (a.role == 'const' or a.role == 'value') and not a.is_vector and a.shape == ():
                        body_lines.append(f"{res_var} = float({a.var_name}) / {b.var_name}")
                        stack.append(StackItem(var_name=res_var, role=b.role,
                                            shape=b.shape, is_vector=b.is_vector,
                                            is_gradient=b.is_gradient, field_names=b.field_names,
                                            parent_name=b.parent_name))
                    else:
                        raise NotImplementedError(
                            f"Division not implemented for roles {a.role}/{b.role} "
                            f"with shapes {a.shape}/{b.shape}")
                # -----------------------------------------------------------------
                # ------------------  POWER  ( **  )  --------------------------------
                # -----------------------------------------------------------------
                 elif op.op_symbol == '**':
                    body_lines.append("# Power")
                    # power *anything* by a scalar constant (const in exponent)
                    if (b.role == 'const' or b.role == 'value') and not b.is_vector and b.shape == ():
                        body_lines.append(f"{res_var} = {a.var_name} ** float({b.var_name})")
                        stack.append(StackItem(var_name=res_var, role=a.role,
                                            shape=a.shape, is_vector=a.is_vector,
                                            is_gradient=a.is_gradient, field_names=a.field_names,
                                            parent_name=a.parent_name))
                    elif (a.role == 'const' or a.role == 'value') and not a.is_vector and a.shape == ():
                        body_lines.append(f"{res_var} = float({a.var_name}) ** {b.var_name}")
                        stack.append(StackItem(var_name=res_var, role=b.role,
                                            shape=b.shape, is_vector=b.is_vector,
                                            is_gradient=b.is_gradient, field_names=b.field_names,
                                            parent_name=b.parent_name))
                    else:
                        raise NotImplementedError(
                            f"Power not implemented for roles {a.role}/{b.role} "
                            f"with shapes {a.shape}/{b.shape}")

                    

            # --- STORE ---
            elif isinstance(op, Store):
                integrand = stack.pop()
                side = self.last_side_for_store

                if op.store_type == 'matrix':
                    body_lines.append(f"Ke += {integrand.var_name} * w_q")
                elif op.store_type == 'vector':
                    body_lines.append(f"Fe += {integrand.var_name} * w_q")
                elif op.store_type == 'functional':
                    body_lines.append(f"J += {integrand.var_name} * w_q")
                    if functional_shape is None:
                        functional_shape = integrand.shape
                else:
                    raise NotImplementedError(f"Store type '{op.store_type}' not implemented.")
            
            else:
                raise NotImplementedError(f"Opcode {type(op).__name__} not handled in JIT codegen.")

        source, param_order = self._build_kernel_string(
            kernel_name, body_lines, required_args, solution_func_names, functional_shape
        )
        return source, {}, param_order


    def _build_kernel_string(
            self, kernel_name: str,
            body_lines: list,
            required_args: set,
            solution_func_names: set,
            functional_shape: tuple = None,
            DEBUG: bool = False
        ):
        """
        Build complete kernel source code with parallel assembly.
        """
        # New Newton: Change parameter names to reflect they are pre-gathered.
        # print(f"DEBUG L:1016: solution_func_names: {list(solution_func_names)}")
        for name in solution_func_names:
            # We will pass u_{name}_loc directly, not the global coeffs.
            if name.startswith("u_") and name.endswith("_loc"):
                required_args.add(name)
            else:
                required_args.add(f"u_{name}_loc")

        # New Newton: Remove gdofs_map from parameters, it's used before the kernel call.
        # print(f"DEBUG L:1025: required_args: {sorted(list(required_args))}")
        param_order = [
            "gdofs_map",
            "node_coords",
            "qp_phys", "qw", "detJ", "J_inv", "normals", "phis",
            *sorted(list(required_args))
        ]
        if 'global_dofs' in param_order:
            param_order.append('union_sizes')
        param_order = list(dict.fromkeys(param_order))  # Remove duplicates while preserving order
        param_order_literal = ", ".join(f"'{arg}'" for arg in param_order)

        # New Newton: The unpacking block is now much simpler.
        # We just select the data for the current element `e`.
        # print(f"DEBUG L:1035: solution_func_names: {list(solution_func_names)}")
        coeffs_unpack_block = "\n".join(
            f"        {name}_e = {name}[e]" if name.startswith("u_") and name.endswith("_loc")
            else f"        u_{name}_e = u_{name}_loc[e]"
            for name in sorted(list(solution_func_names))
        )
        
        basis_unpack_block = "\n".join(
            f"            {arg}_q = {arg}[e,q]"
            for arg in sorted(required_args)
            if (
                arg.startswith(("b_", "g_"))       # reference tables
                or re.match(r"d\d\d?_.*", arg)     # d00_vx, d12_p,
                and not arg.startswith("domain_bs_")  # exclude bitsets
                or arg in {"J_inv", "J_inv_pos", "J_inv_neg",
                        "detJ", "detJ_pos", "detJ_neg"}
            )
        )


        body_code_block = "\n".join(
            f"            {line.replace('_loc', '_loc_e')}" for line in body_lines if line.strip()
        )

        decorator = ""
        if not DEBUG:
            decorator = "@numba.njit(parallel=True, fastmath=True)"
        # New Newton: The kernel signature and loop structure are updated.
        final_kernel_src = f"""
import numba
import numpy as np
PARAM_ORDER = [{param_order_literal}]
{decorator}
def {kernel_name}(
        {", ".join(param_order)}
    ):
    num_elements        = qp_phys.shape[0]
    # n_dofs_per_element  = {self.n_dofs_local}
    n_dofs_per_element  = gdofs_map.shape[1]            # 9 for volume, 15 on ghost edge
    # print(f"num_elements: {{num_elements}},n_dofs_per_element: {{n_dofs_per_element}}")
    # print(f"number of quadrature points: {{qw.shape[1]}}")

    K_values = np.zeros((num_elements, n_dofs_per_element, n_dofs_per_element), dtype={self.dtype})
    F_values = np.zeros((num_elements, n_dofs_per_element), dtype={self.dtype})
    # Shape of the integrand that lands in “J”.
    # -------- functional accumulator ------------------------------------
    k = {functional_shape[0] if functional_shape else 0}
    J_init = {'0.0' if not functional_shape else f'np.zeros((k,), dtype={self.dtype})'}
    J     = J_init
    J_values = np.zeros(({ 'num_elements' if not functional_shape else f'(num_elements, k)' }), dtype={self.dtype})

    for e in numba.prange(num_elements):
        Ke = np.zeros((n_dofs_per_element, n_dofs_per_element), dtype={self.dtype})
        Fe = np.zeros(n_dofs_per_element, dtype={self.dtype})
        J  = J_init.copy() if {bool(functional_shape)} else J_init

{coeffs_unpack_block}

        for q in range(qw.shape[1]):
            x_q, w_q, J_inv_q = qp_phys[e, q], qw[e, q], J_inv[e, q]
            normal_q = normals[e, q] if normals is not None else np.zeros(2, dtype={self.dtype})
            phi_q    = phis[e, q] if phis is not None else 0.0
{basis_unpack_block}
{body_code_block}

        K_values[e] = Ke
        F_values[e] = Fe
        J_values[e] = J
                
    return K_values, F_values, J_values
""".lstrip()

        return final_kernel_src, param_order
