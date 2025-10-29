# pycutfem/jit/codegen.py
from typing import Tuple, List, Any
from dataclasses import dataclass, field, replace


from pycutfem.jit.ir import (
    LoadVariable, LoadConstant, LoadConstantArray, LoadElementWiseConstant,
    LoadAnalytic, LoadFacetNormal, Grad, Div, PosOp, NegOp,
    BinaryOp, Inner, Dot, Store, Transpose, CellDiameter, LoadFacetNormalComponent, CheckDomain,
    Trace, Determinant, Inverse, Hessian as IRHessian, Laplacian as IRLaplacian
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
    field_sides: list = field(default_factory=list)   # NEW: 'pos'/'neg' per component (optional)
    # Stores the name of the parent Function/VectorFunction
    parent_name: str = ""
    side: str = ""  # Side for ghost integrals, e.g., "+", "-", or ""
    # tiny shim so we can write  item = item._replace(var_name="tmp")
    is_transpose: bool = field(default=False)  # True if this item is a transposed version of another
    is_hessian: bool = field(default=False)  # True if this item is a Hessian matrix
    def _replace(self, **changes) -> "StackItem":
        return replace(self, **changes)
    @staticmethod
    def _has_value(item: "StackItem", attr_name: str) -> bool:
        """Helper to check if an attribute has a non-default value."""
        if attr_name == "field_names":
            return bool(item.field_names)
        if attr_name == "parent_name":
            return bool(item.parent_name)
        if attr_name == "side":
            return bool(item.side)
        if attr_name == "field_sides":
            return bool(item.field_sides)
        return False

    @staticmethod
    def resolve_metadata(
        a: "StackItem",
        b: "StackItem",
        *,
        prefer: str | None = None,   # 'a' | 'b' | 'basis' | None
        strict: bool = False         # if True, raise on conflicts (except 'side')
    ) -> Tuple[List[str], str, str]:
        """
        Merge (field_names, parent_name, side) from a and b.
        - prefer='a' or 'b' chooses that operand on conflicts.
        - prefer='basis' chooses whichever operand has role in {'test','trial'}; if both are basis, prefer 'test'.
        - strict=True raises on conflicts for field_names/parent_name; 'side' never raises (degrades to "").
        - default (strict=False, prefer=None): degrade conflicts to default ([], "").
        """
        def default_for(attr: str):
            return [] if attr == "field_names" else ""

        def pick(attr: str, aval, bval, ahas, bhas):
            # preference
            if prefer == "a" and ahas: return aval
            if prefer == "b" and bhas: return bval
            if prefer == "basis":
                a_is_basis = a.role in {"test","trial"}
                b_is_basis = b.role in {"test","trial"}
                if a_is_basis and not b_is_basis and ahas: return aval
                if b_is_basis and not a_is_basis and bhas: return bval
                if a_is_basis and b_is_basis:
                    # prefer 'test' side if present, else 'trial'
                    if a.role == "test" and ahas: return aval
                    if b.role == "test" and bhas: return bval
                    # fall through
            # equality or single source
            if ahas and not bhas: return aval
            if bhas and not ahas: return bval
            if ahas and bhas:
                if aval == bval:
                    return aval
                # conflict
                if strict and attr in ("field_names","parent_name"):
                    raise ValueError(f"Metadata conflict for '{attr}': {aval} vs {bval}")
                # degrade on conflict
                return default_for(attr)
            # neither
            return default_for(attr)

        resolved = {}
        for attr in ("field_names","parent_name","side", "field_sides"):
            aval, bval = getattr(a, attr), getattr(b, attr)
            ahas, bhas = StackItem._has_value(a, attr), StackItem._has_value(b, attr)
            resolved[attr] = pick(attr, aval, bval, ahas, bhas)

        return resolved["field_names"], resolved["parent_name"], resolved["side"], resolved["field_sides"]
    


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
    rhs = f"{vect.var_name}[0]" if collapse else f"{vect.var_name}"
    body_lines.append(
        f"{res_var} = mul_scalar({scalar.var_name}, {rhs}, {self.dtype})"
    )

    if collapse:
        res_shape = (vect.shape[1],)            # (n,)

    stack.append(StackItem(var_name   = res_var,
                           role       = vect.role,
                           shape      = res_shape,
                           is_vector  = vect.is_vector,
                           is_gradient= vect.is_gradient,
                           is_hessian = vect.is_hessian,
                           field_names= vect.field_names,
                           parent_name= vect.parent_name,
                           side       = vect.side,
                           field_sides=vect.field_sides or [],))

class NumbaCodeGen:
    """
    Translates a linear IR sequence into a Numba-Python kernel source string.
    """
    def __init__(self, nopython: bool = True, mixed_element=None,form_rank=0, on_facet:bool = False):
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
        self.on_facet = on_facet

    # ------------------------------------------------------------------
    # Helpers for robust per-component side resolution / mapping
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_side_from_name(field_name: str) -> str | None:
        """
        Try to infer 'pos' / 'neg' from a component field name like 'u_pos_x' / 'p_neg'.
        Looks for whole-word '_pos_' / '_neg_' (also at start/end).
        """
        if re.search(r'(^|_)pos(_|$)', field_name):
            return "pos"
        if re.search(r'(^|_)neg(_|$)', field_name):
            return "neg"
        return None

    @classmethod
    def _component_side_tag(cls, default_side: str, field_sides: List[str] | None,
                            field_name: str, idx: int) -> str | None:
        """
        Resolve the effective side for a single component:
          1) explicit field_sides[idx] if present ('pos'/'neg')
          2) explicit side from Jump/Pos/Neg ('+' → 'pos', '-' → 'neg')
          3) infer from name ('*_pos_*' / '*_neg_*')
        Returns 'pos'/'neg' or None if nothing can be determined.
        """
        if field_sides and 0 <= idx < len(field_sides) and field_sides[idx] in ("pos","neg"):
            return field_sides[idx]
        # Prefer the explicit +/- from the IR (ghost/interface need this to distinguish owner/neighbor)
        if default_side in ("+","-"):
            return "pos" if default_side == "+" else "neg"
        # # Only as a final hint, glean from the component name
        # hint = cls._infer_side_from_name(field_name)
        # if hint:
        #     return hint
        return None

    
    
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
        for idx, op in enumerate(ir_sequence):
            next_op = ir_sequence[idx + 1] if (idx + 1) < len(ir_sequence) else None
            def _next_is_deriv():
                return isinstance(next_op, (Grad, IRHessian, IRLaplacian))

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
                if a.shape == () or a.shape == tuple():
                    zero_expr = "0.0"
                else:
                    zero_expr = f"np.zeros_like({a.var_name}, dtype={self.dtype})"
                body_lines.append(f"{out} = {a.var_name} if flag else {zero_expr}")

                stack.append(a._replace(var_name=out))
            

            elif isinstance(op, IRHessian):
                a = stack.pop()

                # choose per-side names
                if   a.side == "+": jinv = "J_inv_pos"; H0 = "pos_Hxi0"; H1 = "pos_Hxi1"; suff = "_pos"
                elif a.side == "-": jinv = "J_inv_neg"; H0 = "neg_Hxi0"; H1 = "neg_Hxi1"; suff = "_neg"
                else:               jinv = "J_inv";     H0 = "Hxi0";     H1 = "Hxi1";     suff = ""

                k_comps = len(a.field_names)  # 1 for scalar, >1 for vector fields

                required_args.update({jinv, H0, H1})
                required_args.add("gdofs_map")

                # reference derivative tables (volume: d.._; facet: r.._pos/neg)
                d10, d01, d20, d11, d02 = [], [], [], [], []
                for i, fn in enumerate(a.field_names):
                    if suff=="":
                        n10 = f"d10_{fn}"; n01 = f"d01_{fn}"; n20 = f"d20_{fn}"; n11 = f"d11_{fn}"; n02 = f"d02_{fn}"
                    else:
                        side_tag = self._component_side_tag(a.side, getattr(a, 'field_sides', None), fn, i)
                        n10 = f"r10_{fn}_{side_tag}"; n01 = f"r01_{fn}_{side_tag}"; n20 = f"r20_{fn}_{side_tag}"; n11 = f"r11_{fn}_{side_tag}"; n02 = f"r02_{fn}_{side_tag}"
                     
                    required_args.update({n10, n01, n20, n11, n02})
                    d10.append(n10); d01.append(n01); d20.append(n20); d11.append(n11); d02.append(n02)

                out = new_var("Hess")
                # print(f"Hessian operation: a.role={a.role}, is_vector={a.is_vector}, is_hessian={a.is_hessian}, shape={a.shape}, ")

                # ---------- TEST/TRIAL (keep basis tables, possibly padded) ----------
                if a.role in ("test", "trial"):
                    k_comps = len(a.field_names)
                    body_lines += [
                        "n_union = gdofs_map[e].shape[0]",
                        f"{out} = np.empty(({k_comps}, n_union, 2, 2), dtype={self.dtype})",
                    ]
                    for i, fn in enumerate(a.field_names):
                        s0 = self.me.component_dof_slices[fn].start; s1 = self.me.component_dof_slices[fn].stop
                        Hloc = new_var("Hloc")
                        body_lines += [
                            f"A  = {jinv}[e, q]",
                            f"Hx = {H0}[e, q]", f"Hy = {H1}[e, q]",
                            f"d10_q = {d10[i]}[e, q]", f"d01_q = {d01[i]}[e, q]",
                            f"d20_q = {d20[i]}[e, q]", f"d11_q = {d11[i]}[e, q]", f"d02_q = {d02[i]}[e, q]",
                            f"{Hloc} = compute_physical_hessian(d20_q, d11_q, d02_q, d10_q, d01_q, A, Hx, Hy, {self.dtype})",
                        ]
                        if a.side:  # pad to union using map
                            side_tag = self._component_side_tag(a.side, a.field_sides, fn, i)
                            map_arr = f"{side_tag}_map_{fn}"
                            required_args.add(map_arr)
                            Hpad, me = new_var("Hpad"), new_var("map_e")
                            Hsub = new_var("Hsub")
                            body_lines += [
                                f"{me} = {map_arr}[e]",
                                # slice down to this field first
                                f"{Hsub} = {Hloc}[{s0}:{s1}, :, :]",
                                f"{Hpad} = scatter_tensor_to_union({Hsub}, {me}, n_union, {self.dtype})",
                                f"{out}[{i}] = {Hpad}",
                            ]
                        else:
                            # volume: typically nloc == n_union; assign directly
                            body_lines += [f"{out}[{i}] = {Hloc}"]
                    # push
                    stack.append(StackItem(var_name=out, role=a.role,
                                        shape=(k_comps, -1, 2, 2),
                                        is_vector=False, is_hessian=True,
                                        field_names=a.field_names, 
                                        parent_name=a.parent_name, side=a.side,
                                        field_sides=a.field_sides or []))


                elif a.role == "value":
                    k_comps = len(a.field_names)
                    coeff = (a.parent_name if a.parent_name.startswith("u_")
                            else f"u_{a.parent_name}_loc")
                    required_args.add(coeff)

                    body_lines += [f"{out} = np.zeros(({k_comps}, 2, 2), dtype={self.dtype})"]
                    for i, fn in enumerate(a.field_names):
                        s0 = self.me.component_dof_slices[fn].start; s1 = self.me.component_dof_slices[fn].stop
                        Hloc = new_var("Hloc")
                        body_lines += [
                            f"A  = {jinv}[e, q]",
                            f"Hx = {H0}[e, q]", f"Hy = {H1}[e, q]",
                            f"d10_q = {d10[i]}[e, q]", f"d01_q = {d01[i]}[e, q]",
                            f"d20_q = {d20[i]}[e, q]", f"d11_q = {d11[i]}[e, q]", f"d02_q = {d02[i]}[e, q]",
                            f"{Hloc} = compute_physical_hessian(d20_q, d11_q, d02_q, d10_q, d01_q, A, Hx, Hy, {self.dtype})",
                        ]
                        if a.side:
                            side_tag = self._component_side_tag(a.side, a.field_sides, fn, i)
                            map_arr = f"{side_tag}_map_{fn}"
                            required_args.add(map_arr)
                            Hpad, me = new_var("Hpad"), new_var("map_e")
                            Hsub = new_var("Hsub")
                            body_lines += [
                                f"{me} = {map_arr}[e]",
                                "n_union = gdofs_map[e].shape[0]",
                                f"{Hsub} = {Hloc}[{s0}:{s1}, :, :]",
                                f"{Hpad} = scatter_tensor_to_union({Hsub}, {me}, n_union, {self.dtype})",
                                f"{out}[{i}] = hessian_qp({coeff}, {Hpad})",
                            ]
                        else:
                            # --- tensordot-free collapse: (n,2,2) -> (2,2)
                            body_lines += [
                                f"{out}[{i}] = hessian_qp({coeff}, {Hloc})",
                            ]

                    stack.append(StackItem(var_name=out, role="value",
                                        shape=(k_comps, 2, 2),
                                        is_vector=False, is_hessian=True,
                                        is_gradient=False,
                                        field_names=a.field_names, 
                                        parent_name=a.parent_name, side=a.side,
                                        field_sides=a.field_sides or []))



                else:
                    raise NotImplementedError(f"Hessian not implemented for role {a.role}")



            
            elif isinstance(op, IRLaplacian):
                a = stack.pop()

                # choose per-side arrays
                if   a.side == "+": jinv = "J_inv_pos"; H0 = "pos_Hxi0"; H1 = "pos_Hxi1"; suff = "_pos"
                elif a.side == "-": jinv = "J_inv_neg"; H0 = "neg_Hxi0"; H1 = "neg_Hxi1"; suff = "_neg"
                else:               jinv = "J_inv";     H0 = "Hxi0";     H1 = "Hxi1";     suff = ""

                required_args.update({jinv, H0, H1})
                required_args.add("gdofs_map")

                # derivative tables (volume: d.._; facet: r.._pos/neg)
                d10, d01, d20, d11, d02 = [], [], [], [], []
                for i, fn in enumerate(a.field_names):
                    if suff=="":
                        n10 = f"d10_{fn}"; n01 = f"d01_{fn}"; n20 = f"d20_{fn}"; n11 = f"d11_{fn}"; n02 = f"d02_{fn}"
                    else:
                        side_tag = self._component_side_tag(a.side, getattr(a, 'field_sides', None), fn, i)
                        n10 = f"r10_{fn}_{side_tag}"; n01 = f"r01_{fn}_{side_tag}"; n20 = f"r20_{fn}_{side_tag}"; n11 = f"r11_{fn}_{side_tag}"; n02 = f"r02_{fn}_{side_tag}"
                     
                    required_args.update({n10, n01, n20, n11, n02})
                    d10.append(n10); d01.append(n01); d20.append(n20); d11.append(n11); d02.append(n02)

                out = new_var("Lap")
                k_comps = len(a.field_names)


                # ---------------- TEST/TRIAL: keep basis tables ----------------
                if a.role in ("test", "trial"):
                    k_comps = len(a.field_names)
                    body_lines += [
                        "n_union = gdofs_map[e].shape[0]",
                        f"{out} = np.empty(({k_comps}, n_union), dtype={self.dtype})",
                    ]
                    for i, fn in enumerate(a.field_names):
                        s0 = self.me.component_dof_slices[fn].start; s1 = self.me.component_dof_slices[fn].stop
                        laploc = new_var("laploc")
                        body_lines += [
                            f"A  = {jinv}[e, q]",
                            f"Hx = {H0}[e, q]", f"Hy = {H1}[e, q]",
                            f"d10_q = {d10[i]}[e, q]", f"d01_q = {d01[i]}[e, q]",
                            f"d20_q = {d20[i]}[e, q]", f"d11_q = {d11[i]}[e, q]", f"d02_q = {d02[i]}[e, q]",
                            f"{laploc} = compute_physical_laplacian(d20_q, d11_q, d02_q, d10_q, d01_q, A, Hx, Hy, {self.dtype})",
                        ]
                        if a.side:
                            side_tag = self._component_side_tag(a.side, a.field_sides, fn, i)
                            map_arr = f"{side_tag}_map_{fn}"
                            required_args.add(map_arr)
                            lap_pad, me = new_var("lap_pad"), new_var("map_e")
                            lap_sub = new_var("lap_sub")
                            body_lines += [
                                f"{me} = {map_arr}[e]",
                                f"{lap_sub} = {laploc}[{s0}:{s1}]",
                                f"{lap_pad} = scatter_tensor_to_union({lap_sub}, {me}, n_union, {self.dtype})",
                                f"{out}[{i}] = {lap_pad}",
                            ]
                        else:
                            body_lines += [f"{out}[{i}] = {laploc}"]

                    stack.append(StackItem(var_name=out, role=a.role,
                                        shape=(k_comps, -1),
                                        is_vector=True, field_names=a.field_names, 
                                        parent_name=a.parent_name, side=a.side,
                                        field_sides=a.field_sides or []))


                # ---------------- VALUE: collapse with coeffs → (k,) ----------------
                elif a.role == "value":
                    coeff = (a.parent_name if a.parent_name.startswith("u_")
                            else f"u_{a.parent_name}_loc")
                    required_args.add(coeff)

                    body_lines += [f"{out} = np.zeros(({k_comps},), dtype={self.dtype})"]
                    for i, fn in enumerate(a.field_names):
                        s0 = self.me.component_dof_slices[fn].start; s1 = self.me.component_dof_slices[fn].stop
                        laploc = new_var("laploc")
                        body_lines += [
                            f"A  = {jinv}[e, q]",
                            f"Hx = {H0}[e, q]", f"Hy = {H1}[e, q]",
                            f"d10_q = {d10[i]}[e, q]", f"d01_q = {d01[i]}[e, q]",
                            f"d20_q = {d20[i]}[e, q]", f"d11_q = {d11[i]}[e, q]", f"d02_q = {d02[i]}[e, q]",
                            f"{laploc} = compute_physical_laplacian(d20_q, d11_q, d02_q, d10_q, d01_q, A, Hx, Hy, {self.dtype})",
                        ]
                        if a.side:
                            side_tag = self._component_side_tag(a.side, a.field_sides, fn, i)
                            map_arr = f"{side_tag}_map_{fn}"
                            required_args.add(map_arr)
                            lap_pad = new_var("lap_pad"); me = new_var("map_e"); lap_sub = new_var("lap_sub")
                            body_lines += [
                                f"{me} = {map_arr}[e]",
                                "n_union = gdofs_map[e].shape[0]",
                                f"{lap_sub} = {laploc}[{s0}:{s1}]",
                                f"{lap_pad} = scatter_tensor_to_union({lap_sub}, {me}, n_union, {self.dtype})",
                                f"{out}[{i}] = laplacian_qp({coeff}, {lap_pad})",
                            ]
                        else:
                            body_lines += [
                                f"{out}[{i}] = laplacian_qp({coeff}, {laploc})"
                            ]

        

                    stack.append(StackItem(var_name=out, role="value", shape=(k_comps,),
                                        is_vector=True, field_names=a.field_names, 
                                        parent_name=a.parent_name, side=a.side,
                                        field_sides=a.field_sides or []))

                else:
                    raise NotImplementedError(f"Laplacian not implemented for role {a.role}")



            
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
                param = f"ana_{op.func_id}"                  # unique name in PARAM_ORDER
                required_args.add(param)
                var_name = new_var("ana")
                body_lines.append(f"{var_name} = {param}[e, q]")
                tshape = tuple(getattr(op, "tensor_shape", ()))
                is_vec = (len(tshape) == 1)
                role = 'const' if not is_vec else 'value'
                stack.append(StackItem(var_name=var_name,
                                    role=role,
                                    shape=tshape if tshape else (),
                                    is_vector=is_vec))
            # --- LOAD OPERATIONS ---
            # ---------------------------------------------------------------------------
            # LOADVARIABLE –– basis tables and coefficient look-ups
            # ---------------------------------------------------------------------------
            elif isinstance(op, LoadVariable):
                # ------------------------------------------------------------------
                # 0. Peek ahead: if a differential operator will consume this
                #    test/trial next, don't materialize φ here.
                # ------------------------------------------------------------------
                next_op = ir_sequence[idx + 1] if idx + 1 < len(ir_sequence) else None
                followed_by_diff = isinstance(next_op, (Grad, IRHessian, IRLaplacian))

                # Fast path: symbolic hand-off to Grad/Hessian/Laplacian
                if (op.role in ("test", "trial")
                    and op.deriv_order == (0, 0)
                    and followed_by_diff):
                    stack.append(
                        StackItem(
                            var_name="__basis__",
                            role=op.role,
                            shape=(len(op.field_names), -1),
                            is_vector=op.is_vector,
                            field_names=op.field_names,
                            parent_name=op.name,
                            side=op.side,
                            field_sides=op.field_sides or []
                        )
                    )
                    continue

                # ------------------------------------------------------------------
                # 1. Common set-up --------------------------------------------------
                # ------------------------------------------------------------------
                deriv_order = op.deriv_order
                field_names = op.field_names
                is_sided = bool(self.on_facet and op.side in ("+", "-"))


             

                # *Reference* tables are side‑agnostic (no suffix)
                side_suffix_basis = ""

                # helper: "dxy_" or "b_" + field name
                def get_basis_arg_name(field_name: str, deriv: tuple[int, int], idx: int) -> str:
                    """
                    Return the correct table name for this component at the requested order.
                    • Facets (is_sided=True): rXY_<field>_{pos|neg}
                    • Volume/unsided        : b_<field> (00), dXY_<field> (>=10)
                    Always returns a non-empty string.
                    """
                    d0, d1 = deriv
                    if is_sided:
                        # per-component side, with safe fallback to op.side
                        side_tag = self._component_side_tag(op.side, getattr(op, "field_sides", None),
                                                            field_name, idx)
                        if side_tag not in ("pos", "neg"):
                            side_tag = "pos" if op.side == "+" else "neg"
                        if d0 == 0 and d1 == 0:
                            return f"r00_{field_name}_{side_tag}"
                        return f"r{d0}{d1}_{field_name}_{side_tag}"
                    else:
                        if d0 == 0 and d1 == 0:
                            return f"b_{field_name}"
                        return f"d{d0}{d1}_{field_name}"

         

                # IMPORTANT: pass the index so the helper can pick the correct side per component
                basis_arg_names = [get_basis_arg_name(fname, deriv_order, i)
                                for i, fname in enumerate(field_names)]


                # Only request b_* / d** tables *here* when they are actually needed
                # (Function followed immediately by derivative ops will assemble their
                # own derivative tables later; no b_* needed there).
                add_basis_now = not (op.role == "function" and _next_is_deriv())
                if add_basis_now:
                    for arg_name in basis_arg_names:
                        required_args.add(arg_name)

                # Which J⁻¹ to use for push‑forward / derivatives
                if   op.side == "+":  jinv_sym = "J_inv_pos"; required_args.add("J_inv_pos")
                elif op.side == "-":  jinv_sym = "J_inv_neg"; required_args.add("J_inv_neg")
                else:                 jinv_sym = "J_inv";      required_args.add("J_inv")
 

                # reference value(s) at current quadrature point (only if used below)
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
                    if op.side:
                        required_args.add("gdofs_map")  # used for union width                                              # "+" or "-"
                        body_lines += [
                            "n_union = gdofs_map[e].shape[0]",               # e.g. 36 for Stokes
                        ]
                        padded_vars = []                                     # one per component
                        for i, bq in enumerate(basis_vars_at_q):
                            fld_i = field_names[i]
                            # field slice in the element-union vector
                            s0 = self.me.component_dof_slices[fld_i].start
                            s1 = self.me.component_dof_slices[fld_i].stop
                            side_tag = self._component_side_tag(op.side, op.field_sides, fld_i, i)
                            map_array_name = f"{side_tag}_map_{fld_i}"
                            required_args.add(map_array_name)
                            map_e = new_var(f"{map_array_name}_e")
                            loc = new_var(f"local_basis{i}")
                            pad = new_var(f"padded_basis{i}")
                            body_lines += [
                                f"{map_e} = {map_array_name}[e]",
                                f"{loc}   = {bq}",
                                "n_union = gdofs_map[e].shape[0]",
                                # fast path: when local already matches the facet/ghost union
                                f"if n_union == {loc}.shape[0]:",
                                f"    {pad} = {loc}.copy()",
                                f"else:",
                                # robust path: map length is the field-local size
                                f"    m = {map_e}.shape[0]",
                                # align owner-mixed (e.g. 22) to field-local (e.g. 9) only if needed
                                f"    loc_vec = {loc} if {loc}.shape[0] == m else {loc}[{s0}:{s1}]",
                                f"    {pad} = np.zeros(n_union, dtype={self.dtype})",
                                f"    for j in range(m):",
                                f"        idx = {map_e}[j]",
                                f"        if 0 <= idx < n_union:",
                                f"            {pad}[idx] = loc_vec[j]",
                            ]
                            padded_vars.append(pad)

                        basis_vars_at_q = padded_vars        # hand off the padded list
                        final_basis_var = padded_vars[0]     # for scalar reshape path
                        n_dofs = -1                          # run-time on ghost facets

                    # ---------- volume / interface -----------------------------------
                    else:
                        final_basis_var = basis_vars_at_q[0]
                        n_dofs = self.me.n_dofs_local

                    ox, oy = deriv_order
                    tot = ox + oy

                    # ---------- (A) 0th order: keep your fast path -------------------
                    if tot == 0:
                        if not op.is_vector:
                            var_name = new_var("basis_reshaped")
                            body_lines.append(f"{var_name} = {final_basis_var}[np.newaxis, :].copy()")
                            shape = (1, n_dofs)
                        else:
                            var_name = new_var("basis_stack")
                            body_lines.append(f"{var_name} = np.stack(({', '.join(basis_vars_at_q)}))")
                            shape = (len(field_names), n_dofs)
                        stack.append(StackItem(var_name=var_name, role=op.role, shape=shape,
                                            is_vector=op.is_vector, field_names=field_names, 
                                            parent_name=op.name, side=op.side,
                                            field_sides=op.field_sides or []))
                        continue

                    # Decide which J^{-1} to use and bind a local for use below
                    if op.side == "+":
                        jinv_sym = "J_inv_pos"
                    elif op.side == "-":
                        jinv_sym = "J_inv_neg"
                    else:
                        jinv_sym = "J_inv"
                    required_args.add(jinv_sym)

                    # ---------- (B) order == 1: use grad tables @ J_inv --------------
                    if tot == 1:
                        comp = 0 if ox == 1 else 1
                        rows = []
                        # bind J_inv[e,q] once
                        Aq = new_var("Aq")
                        body_lines.append(f"{Aq} = {jinv_sym}[e, q]")
                        for i, fn in enumerate(field_names):
                            nm = f"g_{fn}"
                            required_args.add(nm)
                            gloc = new_var("g_loc"); prow = new_var("prow"); 
                            body_lines += [
                                f"{gloc} = {nm}[e, q] @ {Aq}",
                                f"{prow} = {gloc}[:, {comp}]",
                            ]
                            if op.side:
                                s0 = self.me.slice(fn).start; s1 = self.me.slice(fn).stop
                                side_tag = self._component_side_tag(op.side, op.field_sides, fn, i)
                                map_arr = f"{side_tag}_map_{fn}"
                                required_args.add(map_arr)
                                pad = new_var("pad"); me = new_var("me")
                                body_lines += [
                                    f"{me} = {map_arr}[e]",
                                    "n_union = gdofs_map[e].shape[0]",
                                    # use map length; slice prow only if needed
                                    f"m = {me}.shape[0]",
                                    f"prow_vec = {prow} if {prow}.shape[0] == m else {prow}[{s0}:{s1}]",
                                    f"{pad} = np.zeros(n_union, dtype={self.dtype})",
                                    f"for j in range(m):",
                                    f"    idx = {me}[j]",
                                    "    if 0 <= idx < n_union:",
                                    f"        {pad}[idx] = prow_vec[j]",
                                ]
                                rows.append(pad)
                            else:
                                rows.append(prow)
                        var_name = new_var("d1_stack")
                        if not op.is_vector:
                            body_lines.append(f"{var_name} = {rows[0]}[None, :].copy()")
                            shape = (1, -1 if op.side else self.n_dofs_local)
                        else:
                            body_lines.append(f"{var_name} = np.stack(({', '.join(rows)}))")
                            shape = (len(field_names), -1 if op.side else self.n_dofs_local)
                        stack.append(StackItem(var_name=var_name, role=op.role, shape=shape,
                                            is_vector=op.is_vector, field_names=field_names, 
                                            parent_name=op.name, side=op.side,
                                            field_sides=op.field_sides or []))
                        continue

                    # ---------- (C) order >= 2: exact chain-rule up to 4 -------------
                    # choose per-side jet names
                    if   op.side == "+": A_ = "J_inv_pos"; H0 = "pos_Hxi0"; H1 = "pos_Hxi1"; T0 = "pos_Txi0"; T1 = "pos_Txi1"; Q0 = "pos_Qxi0"; Q1 = "pos_Qxi1"
                    elif op.side == "-": A_ = "J_inv_neg"; H0 = "neg_Hxi0"; H1 = "neg_Hxi1"; T0 = "neg_Txi0"; T1 = "neg_Txi1"; Q0 = "neg_Qxi0"; Q1 = "neg_Qxi1"
                    else:                A_ = "J_inv";     H0 = "Hxi0";     H1 = "Hxi1";     T0 = "Txi0";     T1 = "Txi1";     Q0 = "Qxi0";     Q1 = "Qxi1"
                    required_args.update({A_})
                    if tot >= 2: required_args.update({H0, H1})
                    if tot >= 3: required_args.update({T0, T1})
                    if tot >= 4: required_args.update({Q0, Q1})

                    # bind A locally for use inside loops
                    A_loc = new_var("Aj")
                    body_lines.append(f"{A_loc} = {A_}[e, q]")

                    # derivative tables (volume: d.._; facet: r.._pos/neg)
                    def _tab(name, suff):
                        return (f"d{name}_{fn}" if not op.side else f"r{name}_{fn}_{'pos' if op.side=='+' else 'neg'}")
                    need = { "10":"d10", "01":"d01", "20":"d20","11":"d11","02":"d02",
                            "30":"d30","21":"d21","12":"d12","03":"d03",
                            "40":"d40","31":"d31","22":"d22","13":"d13","04":"d04" }

                    out_rows = []
                    for i, fn in enumerate(field_names):
                        names = {}
                        for key, tag in need.items():
                            if (tot >= int(key[0])+int(key[1])):     # only what we need
                                if not op.side:
                                    nm = f"{tag}_{fn}"
                                else:
                                    side_tag = self._component_side_tag(op.side, getattr(op, 'field_sides', None), fn, i)
                                    nm = f"r{key}_{fn}_{side_tag}"
                                required_args.add(nm); names[tag] = nm

                        row = new_var("drow")
                        # pull derivative arrays; slice ONLY on sided paths
                        if op.side:
                            s0 = self.me.slice(fn).start; s1 = self.me.slice(fn).stop
                            for tag, nm in names.items():
                                body_lines += [f"{tag}_q = {nm}[e, q]",
                                               f"{tag}_s = {tag}_q[{s0}:{s1}]"]
                            body_lines += [
                                "nloc = {}_s.shape[0]".format("d20" if tot>=2 else "d10"),
                                f"{row} = np.zeros((nloc,), dtype={self.dtype})",
                            ]
                        else:
                            for tag, nm in names.items():
                                body_lines.append(f"{tag}_q = {nm}[e, q]")
                            body_lines += [
                                "nloc = {}_q.shape[0]".format("d20" if tot>=2 else "d10"),
                                f"{row} = np.zeros((nloc,), dtype={self.dtype})",
                            ]

                        # prepare axes list (e.g. [0,0,1] for (2,1))
                        body_lines.append(f"axes = np.array([{', '.join(map(str, [0]*ox + [1]*oy))}], dtype=np.int32)")

                        # 2nd-order contribution (and extraction of xx/xy/yy)
                        if tot == 2:
                            body_lines += [
                                f"Hx = {H0}[e, q]; Hy = {H1}[e, q]",
                                f"Href = np.zeros((2,2), dtype={self.dtype})",
                                "for j in range(nloc):",
                                ("    Href[0,0] = d20_s[j]; Href[0,1] = d11_s[j]; Href[1,0] = d11_s[j]; Href[1,1] = d02_s[j]"
                                 if op.side else
                                 "    Href[0,0] = d20_q[j]; Href[0,1] = d11_q[j]; Href[1,0] = d11_q[j]; Href[1,1] = d02_q[j]"),
                                 
                                f"    core = {A_loc}.T @ (Href @ {A_loc})",
                                ("    Hphys = core + d10_s[j]*Hx + d01_s[j]*Hy"
                                 if op.side else
                                 "    Hphys = core + d10_q[j]*Hx + d01_q[j]*Hy"),
                                "    if   axes[0]==0 and axes[1]==0:  val = Hphys[0,0]",
                                "    elif axes[0]==1 and axes[1]==1:  val = Hphys[1,1]",
                                "    else:                              val = Hphys[0,1]",
                                f"    {row}[j] = val",
                            ]
                        elif tot == 3:
                            body_lines += [
                                f"Hx = {H0}[e, q]; Hy = {H1}[e, q]; Tx0={T0}[e,q]; Tx1={T1}[e,q]",
                                "for j in range(nloc):",
                                "    s = 0.0",
                                "    # g3 term",
                                "    for a in (0,1):",
                                "      for b in (0,1):",
                                "        for c in (0,1):",
                                "          ones = (a==1)+(b==1)+(c==1)",
                                ("          g3 = d30_s[j] if ones==0 else (d21_s[j] if ones==1 else (d12_s[j] if ones==2 else d03_s[j]))"
                                 if op.side else
                                 "          g3 = d30_q[j] if ones==0 else (d21_q[j] if ones==1 else (d12_q[j] if ones==2 else d03_q[j]) )"),
                                f"          s += g3 * {A_loc}[a, axes[0]] * {A_loc}[b, axes[1]] * {A_loc}[c, axes[2]]",
                                "    # g2 · A2 term (3 permutations)",
                                "    for a in (0,1):",
                                "      for b in (0,1):",
                                ("        g2 = d20_s[j] if (a==0 and b==0) else (d11_s[j] if a!=b else d02_s[j])"
                                 if op.side else
                                 "        g2 = d20_q[j] if (a==0 and b==0) else (d11_q[j] if a!=b else d02_q[j])"),
                                "        Hb = Hx if b==0 else Hy",
                                f"        s += g2 * ( {A_loc}[a,axes[0]]*Hb[axes[1],axes[2]] + {A_loc}[a,axes[1]]*Hb[axes[0],axes[2]] + {A_loc}[a,axes[2]]*Hb[axes[0],axes[1]] )",
                                "    # g1 · A3 term",
                                ("    s += d10_s[j] * Tx0[axes[0],axes[1],axes[2]] + d01_s[j] * Tx1[axes[0],axes[1],axes[2]]"
                                 if op.side else
                                 "    s += d10_q[j] * Tx0[axes[0],axes[1],axes[2]] + d01_q[j] * Tx1[axes[0],axes[1],axes[2]]"),

                                f"    {row}[j] = s",
                            ]
                        else:  # tot == 4
                            body_lines += [
                                f"Hx = {H0}[e, q]; Hy = {H1}[e, q]; Tx0={T0}[e,q]; Tx1={T1}[e,q]; Qx0={Q0}[e,q]; Qx1={Q1}[e,q]",
                                "for j in range(nloc):",
                                "    s = 0.0",
                                "    # g4 term",
                                "    for a in (0,1):",
                                "      for b in (0,1):",
                                "        for c in (0,1):",
                                "          for d in (0,1):",
                                "            ones = (a==1)+(b==1)+(c==1)+(d==1)",
                                ("            g4 = d40_s[j] if ones==0 else (d31_s[j] if ones==1 else (d22_s[j] if ones==2 else (d13_s[j] if ones==3 else d04_s[j])))"
                                 if op.side else
                                 "            g4 = d40_q[j] if ones==0 else (d31_q[j] if ones==1 else (d22_q[j] if ones==2 else (d13_q[j] if ones==3 else d04_q[j])))"),

                                f"            s += g4 * {A_loc}[a,axes[0]]*{A_loc}[b,axes[1]]*{A_loc}[c,axes[2]]*{A_loc}[d,axes[3]]",
                                "    # g3 · A2 term (choose the A2 holder, choose a pair)",
                                "    for a in (0,1):",
                                "      for b in (0,1):",
                                "        for c in (0,1):",
                                ("        g3v = d30_s[j] if (a+b+c)==0 else (d21_s[j] if (a+b+c)==1 else (d12_s[j] if (a+b+c)==2 else d03_s[j]))"
                                 if op.side else
                                 "        g3v = d30_q[j] if (a+b+c)==0 else (d21_q[j] if (a+b+c)==1 else (d12_q[j] if (a+b+c)==2 else d03_q[j]))"),

                                "          for holder in (0,1,2):",
                                "            hb = [a,b,c][holder]",
                                "            Hb = Hx if hb==0 else Hy",
                                "            others = [a,b,c][:holder]+[a,b,c][holder+1:]",
                                "            for p in range(4):",
                                "              for q in range(p+1,4):",
                                "                r = [0,1,2,3]",
                                "                r.remove(p); r.remove(q)",
                                f"                s += g3v * Hb[axes[p],axes[q]] * {A_loc}[others[0],axes[r[0]]] * {A_loc}[others[1],axes[r[1]]]",
                                "    # g2 · (A2·A2) term",
                                "    for a in (0,1):",
                                "      for b in (0,1):",
                                ("        g2v = d20_s[j] if (a==0 and b==0) else (d11_s[j] if a!=b else d02_s[j])"
                                 if op.side else
                                 "        g2v = d20_q[j] if (a==0 and b==0) else (d11_q[j] if a!=b else d02_q[j])"),
                                "        Ha = Hx if a==0 else Hy; Hb = Hx if b==0 else Hy",
                                "        s += g2v * ( Ha[axes[0],axes[1]]*Hb[axes[2],axes[3]] + Ha[axes[0],axes[2]]*Hb[axes[1],axes[3]] + Ha[axes[0],axes[3]]*Hb[axes[1],axes[2]] )",
                                "    # g2 · A3 term (pick the single-A index)",
                                "    for a in (0,1):",
                                "      for b in (0,1):",
                                ("        g2v = d20_s[j] if (a==0 and b==0) else (d11_s[j] if a!=b else d02_s[j])"
                                 if op.side else
                                 "        g2v = d20_q[j] if (a==0 and b==0) else (d11_q[j] if a!=b else d02_q[j])"),
                                "        Tb = Tx0 if b==0 else Tx1; Ta = Tx0 if a==0 else Tx1",
                                "        for p in range(4):",
                                "            rest = [axes[i] for i in range(4) if i != p]",
                                f"            s += g2v * ( {A_loc}[a,axes[p]] * Tb[rest[0],rest[1],rest[2]] + {A_loc}[b,axes[p]] * Ta[rest[0],rest[1],rest[2]] )",
                                "    # g1 · A4 term",
                                ("    s += d10_s[j] * Qx0[axes[0],axes[1],axes[2],axes[3]] + d01_s[j] * Qx1[axes[0],axes[1],axes[2],axes[3]]"
                                 if op.side else
                                 "    s += d10_q[j] * Qx0[axes[0],axes[1],axes[2],axes[3]] + d01_q[j] * Qx1[axes[0],axes[1],axes[2],axes[3]]"),
                                f"    {row}[j] = s",
                            ]

                        # Pad to union if side-restricted
                        if op.side:
                            side_tag = self._component_side_tag(op.side, op.field_sides, fn, i)
                            map_arr = f"{side_tag}_map_{fn}"
                            required_args.add(map_arr)
                            pad = new_var("pad"); me = new_var("me")
                            body_lines += [
                                f"{me} = {map_arr}[e]",
                                "n_union = gdofs_map[e].shape[0]",
                                f"{pad} = np.zeros(n_union, dtype={self.dtype})",
                                "for j in range(nloc):",
                                f"    idx = {me}[j]",
                                "    if 0 <= idx < n_union:",
                                f"        {pad}[idx] = {row}[j]",
                            ]
                            out_rows.append(pad)
                        else:
                            out_rows.append(row)

                    # stack per-component rows
                    var_name = new_var("d_stack")
                    if not op.is_vector:
                        body_lines.append(f"{var_name} = {out_rows[0]}[None, :].copy()")
                        shape = (1, -1 if op.side else self.n_dofs_local)
                    else:
                        body_lines.append(f"{var_name} = np.stack(({', '.join(out_rows)}))")
                        shape = (len(field_names), -1 if op.side else self.n_dofs_local)

                    stack.append(StackItem(var_name=var_name, role=op.role, shape=shape,
                                        is_vector=op.is_vector, field_names=field_names, 
                                        parent_name=op.name, side=op.side,
                                        field_sides=op.field_sides or []))
                    continue


                

                # ------------------------------------------------------------------
                # 3. Coefficient / Function values  –– scalar **and** vector
                # ------------------------------------------------------------------
                elif op.role == "function":
                    # --------------------------------------------------------------
                    # 3-A  Which coefficient array do we need?  (single array)
                    # --------------------------------------------------------------
                    # If the very next op is a derivative, don't build u(x_q) via b_*
                    # Just pass through the coefficient; derivative op will collapse
                    # r**/d** with it.  This avoids needing b_* on facets.
                    from pycutfem.jit.symbols import POS_SUFFIX, NEG_SUFFIX
                    coeff_side = POS_SUFFIX if op.side == "+" else NEG_SUFFIX if op.side == "-" else ""
                    if op.name.startswith("u_") and op.name.endswith("_loc"):
                        coeff_sym = op.name[:-4] + f"{coeff_side}_loc"
                    else:
                        coeff_sym = f"u_{op.name}{coeff_side}_loc"
                    required_args.add(coeff_sym)
                    solution_func_names.add(coeff_sym)

                    if _next_is_deriv():
                        # Push a lightweight placeholder – IRGrad/IRHessian/IRLaplacian
                        # will materialize the correct quantity using derivative tables.
                        stack.append(
                            StackItem(
                                var_name    = "coeff_placeholder",
                                role        = "value",
                                shape       = (len(field_names),),
                                is_vector   = op.is_vector,
                                field_names = field_names,
                                parent_name = coeff_sym,
                                side        = op.side,
                                field_sides = op.field_sides or []
                            )
                        )
                        continue

                    # --- Otherwise: we truly need either u(x_q) or D^{(ox,oy)}u(x_q) ---
                    # Pick side-specific J^{-1} symbol for mapping (used below)
                    if op.side == "+":
                        jinv_sym = "J_inv_pos"
                    elif op.side == "-":
                        jinv_sym = "J_inv_neg"
                    else:
                        jinv_sym = "J_inv"
                    required_args.add(jinv_sym)

                    # --------------------------------------------------------------
                    # 3-B  Pad reference bases to the DOF-union on ghost facets
                    #      (only used for tot==0 path that uses b_* tables)
                    # --------------------------------------------------------------
                    if op.side:                                            # "+" or "-"
                        body_lines += [
                            f"n_union = gdofs_map[e].shape[0]",
                        ]

                        padded = []
                        for i, b_var in enumerate(basis_vars_at_q):
                            fld_i = field_names[i]
                            # field slice bounds for conditional owner→field alignment
                            s0 = self.me.component_dof_slices[fld_i].start
                            s1 = self.me.component_dof_slices[fld_i].stop
                            side_tag = self._component_side_tag(op.side, op.field_sides, fld_i, i)
                            map_array_name = f"{side_tag}_map_{fld_i}"
                            required_args.add(map_array_name)
                            map_e = new_var(f"{map_array_name}_e")
                            local = new_var(f"local_basis{i}")
                            pad   = new_var(f"padded_basis{i}")
                            body_lines += [
                                f"{map_e} = {map_array_name}[e]",
                                f"{local} = {b_var}",
                                # fast path: interior/interface facet where local already equals union
                                f"if n_union == {local}.shape[0]:",
                                f"    {pad} = {local}.copy()",
                                f"else:",
                                # robust path for ghost: map length = n_loc(field); slice only if needed
                                f"    m = {map_e}.shape[0]",
                                f"    loc_vec = {local} if {local}.shape[0] == m else {local}[{s0}:{s1}]",
                                f"    {pad} = np.zeros(n_union, dtype={self.dtype})",
                                f"    for j in range(m):",
                                f"        idx = {map_e}[j]",
                                f"        if 0 <= idx < n_union:",
                                f"            {pad}[idx] = loc_vec[j]",
                            ]
                            padded.append(pad)

                        basis_vars_at_q = padded             # hand padded list forward
                    # volume/interface: basis_vars_at_q already fine

                    # --------------------------------------------------------------
                    # 3-C  Evaluate u_h or its derivative at x_q (scalar or vector)
                    # --------------------------------------------------------------
                    val_var = new_var(f"{op.name}_val")
                    ox, oy = deriv_order
                    tot = ox + oy

                    # --- Case 0: value u(x_q) via b_* --------------------------------
                    if tot == 0:
                        if op.is_vector:
                            body_lines.append(f"{val_var} = np.zeros(({len(field_names)},), dtype={self.dtype})")
                            for comp_idx, b_var in enumerate(basis_vars_at_q):
                                comp_val = new_var("val_comp")
                                body_lines.append(f"{comp_val} = load_variable_qp({coeff_sym}, {b_var})")
                                body_lines.append(f"{val_var}[{comp_idx}] = {comp_val}")
                            shape = (len(field_names),)
                        else:
                            body_lines.append(f"{val_var} = load_variable_qp({coeff_sym}, {basis_vars_at_q[0]})")
                            shape = ()
                    else:
                        # We need D^{(ox,oy)}u_h. Build per-component derivative rows,
                        # (pad to union if sided), then dot with the coefficient vector.

                        # Utility: reference derivative table name (sided vs unsided)
                        def tab_name(fn: str, tag: str, idx: int) -> str:
                            if op.side in ('+','-'):
                                side_tag = self._component_side_tag(op.side, getattr(op, 'field_sides', None), fn, idx)
                                return f"r{tag}_{fn}_{side_tag}"
                            else:
                                return f"d{tag}_{fn}"

                        rows_vec = []

                        if tot == 1:
                            # First order: grad_phys[:,comp] = [d10, d01] @ J_inv[:,comp]
                            comp = 0 if ox == 1 else 1
                            for i, fn in enumerate(field_names):
                                d10n = tab_name(fn, "10", i)
                                d01n = tab_name(fn, "01", i)
                                required_args.update({d10n, d01n})

                                # fetch reference first-derivative rows at (e,q)
                                d10q = new_var("d10_q"); d01q = new_var("d01_q"); row = new_var("row")
                                body_lines += [
                                    f"{d10q} = {d10n}[e,q]",
                                    f"{d01q} = {d01n}[e,q]",
                                    f"{row} = {d10q} * {jinv_sym}[e,q][0,{comp}] + {d01q} * {jinv_sym}[e,q][1,{comp}]",
                                ]

                                # pad to union if sided
                                rv = new_var("rv")
                                if op.side:
                                    s0 = self.me.slice(fn).start; s1 = self.me.slice(fn).stop
                                    side_tag = self._component_side_tag(op.side, op.field_sides, fn, i)
                                    map_arr = f"{side_tag}_map_{fn}"
                                    required_args.add(map_arr)
                                    pad = new_var("pad"); mep = new_var("mapp")
                                    body_lines += [
                                        f"{mep} = {map_arr}[e]",
                                        "n_union = gdofs_map[e].shape[0]",
                                        # use map length; slice row only if needed
                                        f"m = {mep}.shape[0]",
                                        f"row_vec = {row} if {row}.shape[0] == m else {row}[{s0}:{s1}]",
                                        f"{pad} = np.zeros(n_union, dtype={self.dtype})",
                                        f"for j in range(m):",
                                        f"    idx = {mep}[j]",
                                        "    if 0 <= idx < n_union:",
                                        f"        {pad}[idx] = row_vec[j]",
                                        f"{rv} = load_variable_qp({coeff_sym}, {pad})",
                                    ]
                                else:
                                    body_lines.append(f"{rv} = load_variable_qp({coeff_sym}, {row})")
                                rows_vec.append(rv)

                        else:
                            # Orders 2..4: exact chain rule with inverse-map jets (side-aware).
                            if   op.side == "+": A_="J_inv_pos"; H0="pos_Hxi0"; H1="pos_Hxi1"; T0="pos_Txi0"; T1="pos_Txi1"; Q0="pos_Qxi0"; Q1="pos_Qxi1"
                            elif op.side == "-": A_="J_inv_neg"; H0="neg_Hxi0"; H1="neg_Hxi1"; T0="neg_Txi0"; T1="neg_Txi1"; Q0="neg_Qxi0"; Q1="neg_Qxi1"
                            else:                A_="J_inv";     H0="Hxi0";     H1="Hxi1";     T0="Txi0";     T1="Txi1";     Q0="Qxi0";     Q1="Qxi1"
                            required_args.add(A_)
                            if tot >= 2: required_args.update({H0, H1})
                            if tot >= 3: required_args.update({T0, T1})
                            if tot >= 4: required_args.update({Q0, Q1})

                            # bind A locally in the generated kernel
                            A_loc = new_var("Aj")
                            body_lines.append(f"{A_loc} = {A_}[e,q]")

                            # which reference derivative tables we need
                            need_tags = ["10","01","20","11","02"]
                            if tot >= 3: need_tags += ["30","21","12","03"]
                            if tot >= 4: need_tags += ["40","31","22","13","04"]

                            for i, fn in enumerate(field_names):
                                # bring in reference tables for this component at (e,q)
                                for tg in need_tags:
                                    nm = tab_name(fn, tg, i)
                                    required_args.add(nm)
                                    body_lines.append(f"d{tg}_q = {nm}[e,q]")

                                if op.side:
                                    s0 = self.me.slice(fn).start; s1 = self.me.slice(fn).stop
                                    for tg in need_tags:
                                        body_lines.append(f"d{tg}_s = d{tg}_q[{s0}:{s1}]")
                                    body_lines.append("nloc = d20_s.shape[0]")
                                else:
                                    body_lines.append("nloc = d20_q.shape[0]")
                                row = new_var("row")
                                body_lines.append(f"{row} = np.zeros((nloc,), dtype={self.dtype})")


                                # axes list (e.g. [0,0] for (2,0); [0,1] for (1,1); ...)
                                axes_lit = ",".join(["0"]*ox + ["1"]*oy)
                                body_lines.append(f"axes = np.array([{axes_lit}], dtype=np.int32)")

                                if tot == 2:
                                    body_lines += [
                                        f"Hx = {H0}[e,q]; Hy = {H1}[e,q]",
                                        "Href = np.zeros((2,2), dtype={})".format(self.dtype),
                                        "for j in range(nloc):",
                                        ("    Href[0,0]=d20_s[j]; Href[0,1]=d11_s[j]; Href[1,0]=d11_s[j]; Href[1,1]=d02_s[j]"
                                         if op.side else
                                         "    Href[0,0]=d20_q[j]; Href[0,1]=d11_q[j]; Href[1,0]=d11_q[j]; Href[1,1]=d02_q[j]"),
                                        f"    core = {A_loc}.T @ (Href @ {A_loc})",
                                        ("    Hphys = core + d10_s[j]*Hx + d01_s[j]*Hy"
                                         if op.side else
                                         "    Hphys = core + d10_q[j]*Hx + d01_q[j]*Hy"),
                                        "    if   axes[0]==0 and axes[1]==0:  val = Hphys[0,0]",
                                        "    elif axes[0]==1 and axes[1]==1:  val = Hphys[1,1]",
                                        "    else:                              val = Hphys[0,1]",
                                        f"    {row}[j] = val",
                                    ]
                                elif tot == 3:
                                    body_lines += [
                                        f"Hx = {H0}[e,q]; Hy = {H1}[e,q]; Tx0={T0}[e,q]; Tx1={T1}[e,q]",
                                        "for j in range(nloc):",
                                        "    s = 0.0",
                                        "    # g3 term",
                                        "    for a in (0,1):",
                                        "      for b in (0,1):",
                                        "        for c in (0,1):",
                                        "          ones = (a==1)+(b==1)+(c==1)",
                                        ("          g3 = d30_s[j] if ones==0 else (d21_s[j] if ones==1 else (d12_s[j] if ones==2 else d03_s[j]))"
                                         if op.side else
                                         "          g3 = d30_q[j] if ones==0 else (d21_q[j] if ones==1 else (d12_q[j] if ones==2 else d03_q[j]))"),

                                        f"          s += g3 * {A_loc}[a,axes[0]]*{A_loc}[b,axes[1]]*{A_loc}[c,axes[2]]",
                                        "    # g2 · A2 (3 permutations)",
                                        "    for a in (0,1):",
                                        "      for b in (0,1):",
                                        ("        g2 = d20_s[j] if (a==0 and b==0) else (d11_s[j] if a!=b else d02_s[j])"
                                         if op.side else
                                         "        g2 = d20_q[j] if (a==0 and b==0) else (d11_q[j] if a!=b else d02_q[j])"),

                                        "        Hb = Hx if b==0 else Hy",
                                        f"        s += g2 * ( {A_loc}[a,axes[0]]*Hb[axes[1],axes[2]] + {A_loc}[a,axes[1]]*Hb[axes[0],axes[2]] + {A_loc}[a,axes[2]]*Hb[axes[0],axes[1]] )",
                                        "    # g1 · A3",
                                        ("    s += d10_s[j]*Tx0[axes[0],axes[1],axes[2]] + d01_s[j]*Tx1[axes[0],axes[1],axes[2]]"
                                     if op.side else
                                     "    s += d10_q[j]*Tx0[axes[0],axes[1],axes[2]] + d01_q[j]*Tx1[axes[0],axes[1],axes[2]]"),

                                        f"    {row}[j] = s",
                                    ]
                                else:  # tot == 4
                                    body_lines += [
                                        f"Hx = {H0}[e,q]; Hy = {H1}[e,q]; Tx0={T0}[e,q]; Tx1={T1}[e,q]; Qx0={Q0}[e,q]; Qx1={Q1}[e,q]",
                                        "for j in range(nloc):",
                                        "    s = 0.0",
                                        "    # g4 term",
                                        "    for a in (0,1):",
                                        "      for b in (0,1):",
                                        "        for c in (0,1):",
                                        "          for d in (0,1):",
                                        "            ones = (a==1)+(b==1)+(c==1)+(d==1)",
                                        ("            g4 = d40_s[j] if ones==0 else (d31_s[j] if ones==1 else (d22_s[j] if ones==2 else (d13_s[j] if ones==3 else d04_s[j]))"
                                         if op.side else
                                         "            g4 = d40_q[j] if ones==0 else (d31_q[j] if ones==1 else (d22_q[j] if ones==2 else (d13_q[j] if ones==3 else d04_q[j]))"),
                                        f"            s += g4 * {A_loc}[a,axes[0]]*{A_loc}[b,axes[1]]*{A_loc}[c,axes[2]]*{A_loc}[d,axes[3]]",
                                        "    # g3 · A2 (place A2 in one slot, A in the others)",
                                        "    for a in (0,1):",
                                        "      for b in (0,1):",
                                        ("        g3v = d30_s[j] if (a+b)==0 else (d21_s[j] if (a+b)==1 else (d12_s[j] if (a+b)==2 else d03_s[j]))"
                                         if op.side else 
                                         "        g3v = d30_q[j] if (a+b)==0 else (d21_q[j] if (a+b)==1 else (d12_q[j] if (a+b)==2 else d03_q[j]))"),
                                        "        for holder in (0,1,2):",
                                        "            hb = [a,b,0][holder]  # dummy pick; indices only drive ones count",
                                        "            Hb = Hx if hb==0 else Hy",
                                        "            for p in range(4):",
                                        "              for q2 in range(p+1,4):",
                                        "                r = [0,1,2,3]",
                                        "                r.remove(p); r.remove(q2)",
                                        f"                s += g3v * Hb[axes[p],axes[q2]] * {A_loc}[a,axes[r[0]]] * {A_loc}[b,axes[r[1]]]",
                                        "    # g2 · (A2·A2) + g2 · A3",
                                        "    for a in (0,1):",
                                        "      for b in (0,1):",
                                        ("        g2v = d20_s[j] if (a==0 and b==0) else (d11_s[j] if a!=b else d02_s[j])"
                                         if op.side else
                                         "        g2v = d20_q[j] if (a==0 and b==0) else (d11_q[j] if a!=b else d02_q[j])"),
                                        "        Ha = Hx if a==0 else Hy; Hb = Hx if b==0 else Hy",
                                        "        s += g2v * ( Ha[axes[0],axes[1]]*Hb[axes[2],axes[3]] + Ha[axes[0],axes[2]]*Hb[axes[1],axes[3]] + Ha[axes[0],axes[3]]*Hb[axes[1],axes[2]] )",
                                        "        Tb = Tx0 if b==0 else Tx1; Ta = Tx0 if a==0 else Tx1",
                                        "        for p in range(4):",
                                        "            rest = [axes[i] for i in range(4) if i != p]",
                                        f"            s += g2v * ( {A_loc}[a,axes[p]] * Tb[rest[0],rest[1],rest[2]] + {A_loc}[b,axes[p]] * Ta[rest[0],rest[1],rest[2]] )",
                                        "    # g1 · A4",
                                        ("    s += d10_s[j]*Qx0[axes[0],axes[1],axes[2],axes[3]] + d01_s[j]*Qx1[axes[0],axes[1],axes[2],axes[3]]"
                                         if op.side else
                                         "    s += d10_q[j]*Qx0[axes[0],axes[1],axes[2],axes[3]] + d01_q[j]*Qx1[axes[0],axes[1],axes[2],axes[3]]"),
                                        f"    {row}[j] = s",
                                    ]

                                # collapse with coefficients; pad to union if sided
                                rv = new_var("rv")
                                if op.side:
                                    side_tag = self._component_side_tag(op.side, op.field_sides, fn, 0)
                                    map_arr = f"{side_tag}_map_{fn}"
                                    required_args.add(map_arr)
                                    pad = new_var("pad"); mep = new_var("mapp")
                                    body_lines += [
                                        f"{mep} = {map_arr}[e]",
                                        "n_union = gdofs_map[e].shape[0]",
                                        f"{pad} = np.zeros(n_union, dtype={self.dtype})",
                                        "for j in range(nloc):",
                                        f"    idx = {mep}[j]",
                                        "    if 0 <= idx < n_union:",
                                        f"        {pad}[idx] = {row}[j]",
                                        f"{rv} = load_variable_qp({coeff_sym}, {pad})",
                                    ]
                                else:
                                    body_lines.append(f"{rv} = load_variable_qp({coeff_sym}, {row})")
                                rows_vec.append(rv)

                        # Final value (scalar or vector)
                        if op.is_vector:
                            body_lines.append(f"{val_var} = np.array([{', '.join(rows_vec)}], dtype={self.dtype})")
                            shape = (len(field_names),)
                        else:
                            body_lines.append(f"{val_var} = float({rows_vec[0]})")
                            shape = ()

                    # --------------------------------------------------------------
                    # 3-E  Push onto stack
                    # --------------------------------------------------------------
                    stack.append(
                        StackItem(
                            var_name    = val_var,
                            role        = "value",
                            shape       = shape,
                            is_vector   = op.is_vector,
                            field_names = field_names,
                            parent_name = coeff_sym,
                            side        = op.side,
                            field_sides = op.field_sides or []
                        )
                    )



                else:
                    raise TypeError(f"Unknown role '{op.role}' for LoadVariable IR node.")

            elif isinstance(op, LoadConstant):
                stack.append(StackItem(var_name=str(op.value), role='const', shape=(), is_vector=False, field_names=[]))
            
            elif isinstance(op, LoadConstantArray):
                required_args.add(op.name)
                np_array_var = new_var("const_np_arr")
                # retain metadata from IR (role/is_vector/is_gradient) to allow identity handling
                body_lines.append(f"{np_array_var} = {op.name}")
                stack.append(StackItem(
                    var_name   = np_array_var,
                    role       = getattr(op, 'role', 'const'),
                    shape      = op.shape,
                    is_vector  = getattr(op, 'is_vector', len(op.shape) == 1),
                    is_gradient= getattr(op, 'is_gradient', False),
                    field_names=[]
                ))
            
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
                    body_lines.append(
                        f"{res_var} = trace_matrix_value({a.var_name}, {self.dtype})"
                    )
                    stack.append(StackItem(var_name=res_var,
                                            role='value',
                                            shape=(), # Result is a scalar
                                            is_vector=False,
                                            is_gradient=False,
                                            field_names=[],
                                            parent_name=a.parent_name,
                                            side=a.side,
                                            field_sides=a.field_sides or []))

                # --- Case 2: Trace of a Test/Trial function tensor (e.g., shape (2, n, 2)) ---
                elif len(a.shape) == 3:
                    # First, validate that the tensor is square over its first and last dimensions.
                    if a.shape[0] != a.shape[2]:
                        raise ValueError(f"Trace requires a square tensor (k=d), but got shape {a.shape}")

                    # If valid, generate the execution code.
                    body_lines.append(f"# Trace of a symbolic tensor -> scalar basis of shape (1, n)")
                    body_lines.append(
                        f"{res_var} = trace_basis_tensor({a.var_name}, {self.dtype})"
                    )
                    stack.append(StackItem(var_name=res_var,
                                            role=a.role,
                                            shape=(1, a.shape[1]),
                                            is_vector=False,
                                            is_gradient=False,
                                            field_names=a.field_names,
                                            parent_name=a.parent_name,
                                            side=a.side,
                                            field_sides=a.field_sides or []))

                elif len(a.shape) == 4:
                    if a.shape[0] != a.shape[3]:
                        raise ValueError(f"Trace requires matching component/spatial dims, got shape {a.shape}")
                    body_lines.append(f"# Trace of a mixed tensor -> scalar mixed basis (1,n,m)")
                    body_lines.append(
                        f"{res_var} = trace_mixed_tensor({a.var_name}, {self.dtype})"
                    )
                    stack.append(StackItem(var_name=res_var,
                                            role=a.role,
                                            shape=(1, a.shape[1], a.shape[2]),
                                            is_vector=False,
                                            is_gradient=False,
                                            field_names=a.field_names,
                                            parent_name=a.parent_name,
                                            side=a.side,
                                            field_sides=a.field_sides or []))

                # --- Else: The shape is not a 2D or 3D tensor, so it's invalid. ---
                else:
                    raise TypeError(f"Cannot take trace of an operand with shape {a.shape}. Must be a 2D or 3D tensor.")
            
            elif isinstance(op, Determinant):
                a = stack.pop()
                if a.shape != (2, 2) or a.role not in ("value", "const"):
                    raise NotImplementedError(
                        "Determinant expects a 2x2 numeric tensor (role 'value' or 'const')."
                    )
                res_var = new_var("det2")
                body_lines += [
                    f"a00 = {a.var_name}[0, 0]",
                    f"a01 = {a.var_name}[0, 1]",
                    f"a10 = {a.var_name}[1, 0]",
                    f"a11 = {a.var_name}[1, 1]",
                    f"{res_var} = a00 * a11 - a01 * a10",
                ]
                stack.append(
                    a._replace(
                        var_name=res_var,
                        role="value",
                        shape=(),
                        is_vector=False,
                        is_gradient=False,
                        is_hessian=False,
                    )
                )

            elif isinstance(op, Inverse):
                a = stack.pop()
                if a.shape != (2, 2) or a.role not in ("value", "const"):
                    raise NotImplementedError(
                        "Inverse expects a 2x2 numeric tensor (role 'value' or 'const')."
                    )
                res_var = new_var("inv2")
                body_lines += [
                    f"a00 = {a.var_name}[0, 0]",
                    f"a01 = {a.var_name}[0, 1]",
                    f"a10 = {a.var_name}[1, 0]",
                    f"a11 = {a.var_name}[1, 1]",
                    "det = a00 * a11 - a01 * a10",
                    "inv_det = 1.0 / (det + 1e-300)",
                    f"{res_var} = np.empty((2, 2), dtype={self.dtype})",
                    f"{res_var}[0, 0] =  a11 * inv_det",
                    f"{res_var}[0, 1] = -a01 * inv_det",
                    f"{res_var}[1, 0] = -a10 * inv_det",
                    f"{res_var}[1, 1] =  a00 * inv_det",
                ]
                stack.append(
                    a._replace(
                        var_name=res_var,
                        role=a.role,
                        shape=(2, 2),
                    )
                )
            
            # --- UNARY OPERATORS ---
            # ----------------------------------------------------------------------
            # ∇(·) operator
            # ----------------------------------------------------------------------
            elif isinstance(op, Grad):
                a = stack.pop()

                # ------------------------------------------------------------------
                # Choose the correct J^{-1} symbol (volume vs sided ghost/interface)
                # ------------------------------------------------------------------
                if   a.side == "+":
                    jinv_sym = "J_inv_pos"; required_args.add("J_inv_pos")
                elif a.side == "-":
                    jinv_sym = "J_inv_neg"; required_args.add("J_inv_neg")
                else:
                    jinv_sym = "J_inv";     required_args.add("J_inv")
                jinv_q = f"{jinv_sym}_q"  # 2x2 at (e,q)
                body_lines.append(f"{jinv_q} = {jinv_sym}[e, q]")


                # ======================================================================
                # (A) grad(Test/Trial)  -> shape (k , n_loc_or_union , 2)
                # ======================================================================
                if a.role in ("test", "trial"):

                    phys = []

                    # Sided path (ghost/interface): use r10/r01 on the correct side,
                    # map with J_inv_{pos|neg}, then pad to union via {side}_map_<field>.
                    if a.side in ("+", "-") and self.on_facet:
                        required_args.add("gdofs_map")  # used for union width
                        for i, fld_i in enumerate(a.field_names):
                            # DOF slice for this component inside the union
                            s0 = self.me.component_dof_slices[fld_i].start
                            s1 = self.me.component_dof_slices[fld_i].stop

                            # decide per-component side tag ("pos" or "neg")
                            side_tag = self._component_side_tag(a.side, a.field_sides, fld_i, i)

                            n10 = f"r10_{fld_i}_{side_tag}"
                            n01 = f"r01_{fld_i}_{side_tag}"
                            required_args.update({n10, n01})

                            # per-component side map (local rows -> union rows)
                            map_arr = f"{side_tag}_map_{fld_i}"
                            required_args.add(map_arr)

                            pg_loc   = new_var("grad_loc")      # (n_loc, 2)
                            pg_loc_s = new_var("grad_loc_s")    # (n_comp_loc, 2)
                            map_e    = new_var(f"{map_arr}_e")  # (n_comp_loc,)
                            pg_pad   = new_var("grad_pad")      # (n_union, 2)

                            body_lines += [
                                f"d10_q = {n10}[e, q]",                               # (n_loc,)
                                f"d01_q = {n01}[e, q]",                               # (n_loc,)
                                f"{pg_loc} = np.stack((d10_q, d01_q), axis=1) @ {jinv_q}.copy()",  # (n_loc,2)
                                f"n_union = gdofs_map[e].shape[0]",
                                f"if {pg_loc}.shape[0] == n_union:",
                                f"    {pg_pad} = {pg_loc}",                      # already union-sized → no remap
                                f"else:",
                                f"    {pg_loc_s} = {pg_loc}[{s0}:{s1}, :]",      # local → slice
                                f"    {map_e} = {map_arr}[e]",
                                f"    {pg_pad} = scatter_tensor_to_union({pg_loc_s}, {map_e}, n_union, {self.dtype})",
                            ]
                            phys.append(pg_pad)

                        n_dofs = -1  # union-sized

                    else:
                        # Volume (unsided): use volume gradient tables g_<field> with J_inv

                        grad_tab_names = [f"g_{f}" for f in a.field_names]

                        for nm in grad_tab_names:
                            required_args.add(nm)

                        for i, fld_i in enumerate(a.field_names):
                            pg_loc = new_var("grad_loc")
                            body_lines.append(f"{pg_loc} = {grad_tab_names[i]}[e, q] @ {jinv_q}.copy()")
                            phys.append(pg_loc)

                        n_dofs = self.n_dofs_local

                    # Stack per-component physical gradients
                    if not a.is_vector:
                        var = new_var("grad_scalar")
                        body_lines.append(f"{var} = {phys[0]}[None, :, :].copy()")
                        shape, is_vector, is_gradient = (1, n_dofs, self.spatial_dim), False, True
                    else:
                        var = new_var("grad_stack")
                        body_lines.append(f"{var} = np.stack(({', '.join(phys)}))")
                        shape, is_vector, is_gradient = (len(a.field_names), n_dofs, self.spatial_dim), False, True

                    stack.append(a._replace(var_name=var, shape=shape,
                                            is_gradient=True, is_vector=False))
                    continue

                # ======================================================================
                # (B) grad(Function/VectorFunction)  (value role)
                # returns: scalar: (2,), vector: (k,2)
                # ======================================================================
                if a.role == "value":
                    # Base coeff alias (may already be '..._e')
                    coeff = (a.parent_name if a.parent_name.startswith("u_")
                            else f"u_{a.parent_name}_loc")
                    required_args.add(coeff)

                    comps = []
                    for i, fld in enumerate(a.field_names):

                        val = new_var("grad_val")

                        if a.side in ("+", "-"):
                            # Sided: use r10/r01 on the correct side and the side map
                            s0 = self.me.component_dof_slices[fld].start
                            s1 = self.me.component_dof_slices[fld].stop

                            side_tag = self._component_side_tag(a.side, getattr(a, 'field_sides', None), fld, i)
                            d10 = f"r10_{fld}_{side_tag}"
                            d01 = f"r01_{fld}_{side_tag}"
                            required_args.update({d10, d01})

                            map_sym = f"{side_tag}_map_{fld}"
                            required_args.add(map_sym)

                            coeff_e = coeff if coeff.endswith("_e") else f"{coeff}"

                            g2     = new_var("g_ref2")        # (n_loc,2) in (ξ,η)
                            phys   = new_var("g_phys2")       # (n_loc,2) in (x,y)
                            phys_s = new_var("g_phys2_s")     # (n_comp_loc,2)
                            u_sl   = new_var("u_side")        # (n_comp_loc,)

                            body_lines += [
                                f"d10_q = {d10}[e, q]",
                                f"d01_q = {d01}[e, q]",
                                f"{g2}   = np.stack((d10_q, d01_q), axis=1)",
                                f"{phys} = {g2} @ {jinv_q}.copy()",
                                f"if {phys}.shape[0] == {coeff_e}.shape[0]:",
                                f"    {val} = gradient_qp({coeff_e}, {phys})",
                                f"else:",
                                f"    {phys_s} = {phys}[{s0}:{s1}, :]",           # local vs local
                                f"    {u_sl}   = {coeff_e}[{map_sym}[e]]",
                                f"    {val}    = gradient_qp({u_sl}, {phys_s})",
                            ]

                        else:
                            # Volume (unsided): use g_<field> with J_inv
                            gnm = f"g_{fld}"
                            required_args.add(gnm)

                            coeff_e = coeff if coeff.endswith("_e") else f"{coeff}"
                            pg = new_var("phys_grad_basis")

                            body_lines += [
                                f"{pg}  = {gnm}[e, q] @ {jinv_q}.copy()",
                                f"{val} = gradient_qp({coeff_e}, {pg})",     # (2,)
                            ]

                        comps.append(val)

                    if not a.is_vector:
                        var, shape, is_vector, is_gradient = comps[0], (self.spatial_dim,), True, False
                    else:
                        var = new_var("grad_val_stack")
                        body_lines.append(f"{var} = np.stack(({', '.join(comps)}))")
                        shape, is_vector, is_gradient = (len(a.field_names), self.spatial_dim), False, True

                    stack.append(
                        StackItem(var_name=var, role="value", shape=shape,
                                is_gradient=is_gradient, is_vector=is_vector,
                                field_names=a.field_names, parent_name=coeff, side=a.side,
                                field_sides=a.field_sides or [])
                    )
                    continue

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
                            side=a.side,
                            is_gradient=False,
                            field_sides=a.field_sides or []
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
                                            field_names=a.field_names, parent_name=a.parent_name,
                                            side=a.side, field_sides=a.field_sides or []))
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
                                            field_names=a.field_names, parent_name=a.parent_name,
                                            side=a.side, field_sides=a.field_sides or []))

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
                # On the interface, keep both sides; otherwise gate by phi_q
                required_args.add("is_interface")
                body_lines.append(
                    f"# Pos Op"
                    f"{res_var} = {a.var_name} if (is_interface or (phi_q >= 0.0)) "
                    f"else np.zeros_like({a.var_name}, dtype={self.dtype})"
                )
                stack.append(a._replace(var_name=res_var))

            elif isinstance(op, NegOp):
                a = stack.pop()
                res_var = new_var("neg")
                required_args.add("is_interface")
                body_lines.append(
                    f"# Neg Op"
                    f"{res_var} = {a.var_name} if (is_interface or (phi_q <  0.0)) "
                    f"else np.zeros_like({a.var_name}, dtype={self.dtype})"
                )
                stack.append(a._replace(var_name=res_var))

            # --- Inner OPERATORS ---
            elif isinstance(op, Inner):
                b = stack.pop(); a = stack.pop()
                res_var = new_var("inner")
                # print(f"Inner operation: a.role={a.role}, b.role={b.role}, a.shape={a.shape}, b.shape={b.shape}"
                #       f", is_vector: {a.is_vector}/{b.is_vector}, is_gradient: {a.is_gradient}/{b.is_gradient}"
                #       f", a.is_transpose: {a.is_transpose}, b.is_transpose: {b.is_transpose}"
                #       f", a.is_hessian: {a.is_hessian}, b.is_hessian: {b.is_hessian}")

                # LHS Bilinear Form: always rows=test, cols=trial
                if a.role in ('test', 'trial') and b.role in ('test', 'trial'):
                    body_lines.append('# Inner(LHS): orient rows=test, cols=trial')

                    # pick operands by role (regardless of stack order)
                    test_var  = f"{a.var_name}" if a.role == "test"  else f"{b.var_name}"
                    trial_var = f"{a.var_name}" if a.role == "trial" else f"{b.var_name}"
                    test_item  = a if a.role == "test"  else b
                    trial_item = a if a.role == "trial" else b

                    if a.is_gradient and b.is_gradient:
                        # grad-grad: test/trial are lists over k, each item is (n, d)
                        body_lines.append(
                            f"{res_var} = inner_grad_grad({test_var}, {trial_var}, {self.dtype})"
                        )

                    elif a.is_hessian and b.is_hessian:
                        # Hessian-Hessian: items are (n, 2, 2)
                        body_lines.append(
                            f"{res_var} = inner_hessian_hessian({test_var}, {trial_var}, {self.dtype})"
                        )

                    else:
                        # vector bases: test.T @ trial
                        body_lines.append(f"# Inner(Vector, Vector): dot product, LHS mass matrix")
                        body_lines.append(
                            f"{res_var} = dot_mass_test_trial({test_var}, {trial_var}, {self.dtype})"
                        )

                    field_names,parent_name ,side, field_sides = StackItem.resolve_metadata(a, b, prefer=None, strict=False)
                    shape = (test_item.shape[1], trial_item.shape[1])  # (n_test, n_trial)
                    stack.append(StackItem(var_name=res_var, role='value',
                                        shape=shape, is_vector=False, is_gradient=False,
                                        field_names=field_names, parent_name=parent_name, side=side,
                                        field_sides=field_sides or []))
                    continue


                # elif a.role == 'const' and b.role == 'const' and a.shape == b.shape:
                #     body_lines.append(f'# Inner(Const, Const): element-wise product')
                #     body_lines.append(f'{res_var} = {a.var_name} * {b.var_name}')
                elif a.role in {'value', 'const'} and b.role == 'test':  # RHS
                    body_lines.append('# RHS: Inner(Function, Test)')
                    # body_lines.append(f"print(f'RHS inner: a.shape: {{{a.var_name}.shape}}, "
                    #                   f"b.shape: {{{b.var_name}.shape}}, ')")

                    if a.is_gradient and b.is_gradient:
                        # a: (k,d) collapsed grad(Function), b: (k,n,d)
                        body_lines.append("# RHS: Inner(Grad(Function), Grad(Test))")
                        body_lines.append(
                            f"{res_var} = inner_grad_function_grad_test({a.var_name}, {b.var_name}, {self.dtype})"
                        )

                    elif a.is_hessian and b.is_hessian:
                        # a: (k,2,2) collapsed Hess(Function), b: (k,n,2,2)
                        body_lines.append("# RHS: Inner(Hessian(Function), Hessian(Test))")
                        body_lines.append(
                            f"{res_var} = inner_hessian_function_hessian_test({a.var_name}, {b.var_name}, {self.dtype})"
                        )

                    elif a.is_vector and b.is_vector:
                        # a: (k,), b: (k,n)  → (n,)
                        body_lines.append("# RHS: Inner(Value(Vector), Test(Vector))")
                        body_lines.append(
                            f"{res_var} = const_vector_dot_basis_1d({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                    # elif not a.is_vector and b.is_vector and b.shape[0] == 1 and len(b.shape) == 2:
                    #     # a: (), b: (n,)  → (n,)
                    #     body_lines += [
                    #         "# RHS: Inner(Value(Scalar), Test(Vector))",
                    #         f"n_locs = {b.var_name}.shape[1]",
                    #         f"{res_var} = {b.var_name}[0] * {a.var_name}",        # (n,) * () -> (n,)
                    #     ]

                    elif a.is_vector and b.is_gradient:
                        # grad(scalarFunction): a: (d,), b: (1,n,d) → (n,)
                        body_lines.append("# RHS: Inner(grad(scalarFunction), grad(Test))")
                        body_lines.append(
                            f"{res_var} = contract_last_first({b.var_name}[0], {a.var_name}, {self.dtype})"
                        )
                    elif not a.is_vector and not b.is_vector and b.shape[0] == 1 and len(b.shape) == 2:
                        # a: (), b: (n,)  → (n,)
                        body_lines.append("# RHS: Inner(Value(Scalar), Test(Scalar))")
                        body_lines.append(
                            f"{res_var} = mul_scalar({a.var_name}, {b.var_name}[0], {self.dtype})"
                        )

                    else:
                        raise NotImplementedError(
                            f"Inner not implemented for roles {a.role}/{b.role}, "
                            f"is_vector: {a.is_vector}/{b.is_vector}, "
                            f"is_gradient: {a.is_gradient}/{b.is_gradient}, "
                            f"shapes: {a.shape}/{b.shape}, is_hessian: {a.is_hessian}/{b.is_hessian}"
                        )
                    # Push RHS vector (n,) with resolved metadata (prefer test’s side)
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer=b, strict=False)
                    stack.append(StackItem(var_name=res_var, role='value',
                                           shape=(b.shape[1],), is_vector=False, is_gradient=False,
                                           field_names=field_names, parent_name=parent_name, side=side,
                                           field_sides=field_sides))
                    continue
                elif a.role == 'mixed' and a.is_gradient and b.role in {'const','value'} and b.is_gradient:
                    body_lines.append("# Inner(mixed gradient, grad(const))")
                    body_lines.append(
                        f"{res_var} = inner_mixed_grad_const({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                    stack.append(StackItem(var_name=res_var, role='value',
                                           shape=(a.shape[1], a.shape[2]), is_vector=False, is_gradient=False,
                                           field_names=field_names, parent_name=parent_name, side=side,
                                           field_sides=field_sides))
                    continue
                elif a.role in {'const','value'} and a.is_gradient and b.role == 'mixed' and b.is_gradient:
                    body_lines.append("# Inner(grad(const), mixed gradient)")
                    body_lines.append(
                        f"{res_var} = inner_grad_const_mixed({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='b', strict=False)
                    stack.append(StackItem(var_name=res_var, role='value',
                                           shape=(b.shape[1], b.shape[2]), is_vector=False, is_gradient=False,
                                           field_names=field_names, parent_name=parent_name, side=side,
                                           field_sides=field_sides))
                    continue
                elif a.role in {'test','trial'} and a.is_gradient and b.role in {'const','value'} and b.is_gradient:
                    role = 'value'
                    body_lines.append("# Inner(grad(Test/Trial), grad(const))")
                    body_lines.append(
                        f"{res_var} = inner_grad_basis_grad_const({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                    stack.append(StackItem(var_name=res_var, role=role,
                                           shape=(a.shape[1],), is_vector=False, is_gradient=False,
                                           field_names=field_names, parent_name=parent_name, side=side,
                                           field_sides=field_sides))
                    continue

                        
                    
                elif a.role in {'value','const'} and b.role in {'value','const'}:
                    # body_lines.append(f"print(f'RHS Functional inner: a.shape: {{{a.var_name}.shape}}, "
                    #                   f"b.shape: {{{b.var_name}.shape}}, "
                    #                   f"a.role={a.role}, b.role={b.role}, ')")
                    if a.is_vector and b.is_vector:
                        body_lines.append(f'# Inner(Value, Value): dot product')
                        body_lines.append(
                            f"{res_var} = dot_vec_vec({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        shape = ()
                    elif a.is_gradient and b.is_gradient:
                        # if self.form_rank == 0:
                        body_lines += [f"{res_var} = float(np.sum({a.var_name} * {b.var_name}))"]
                        shape = ()
                    elif a.is_hessian and b.is_hessian:
                        if self.form_rank == 0:
                            body_lines += [
                                f'# Inner(Hessian(Value), Hessian(Value)): scalarvalue',
                                f'# (k,2,2) * (k,2,2) -> scalar',
                                f'{res_var} = float(np.sum({a.var_name} * {b.var_name}))',
                            ]
                            shape = ()
                        else:
                            body_lines += [
                                f'# Inner(Hessian(Value), Hessian(Value)): stiffness matrix',
                                f'# (k,2,2) @ (k,2,2) -> (k,k)',
                                f'{res_var} = {a.var_name} @ {b.var_name}.T.copy()',
                            ]
                            shape = (a.shape[0], b.shape[1])  # (k,k) for stiffness matrix
                    elif not ((a.is_vector and a.is_gradient and a.is_hessian) and (b.is_vector and b.is_gradient and b.is_hessian)) \
                         and a.shape == b.shape:
                        body_lines.append(f'# Inner(Scalar, Scalar): element-wise product')
                        body_lines.append(f'{res_var} = {a.var_name} * {b.var_name}')
                        shape = ()
                    else:
                        raise NotImplementedError(
                            f"Inner(Value/Const, Value/Const) not implemented for shapes {a.shape}/{b.shape} "
                            f"with flags a(v/g/h)={a.is_vector}/{a.is_gradient}/{a.is_hessian}, "
                            f"b(v/g/h)={b.is_vector}/{b.is_gradient}/{b.is_hessian}"
                        )

                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer=None, strict=False)
                    stack.append(StackItem(var_name=res_var, role='value',
                                           shape=shape, is_vector=False, is_gradient=False,
                                           field_names=field_names, parent_name=parent_name, side=side,
                                           field_sides=field_sides))
                    continue
                        
                
                else:
                    raise NotImplementedError(f"JIT Inner not implemented for roles {a.role}/{b.role}, "
                                              f" is_vector: {a.is_vector}/{b.is_vector}, " 
                                              f" is_gradient: {a.is_gradient}/{b.is_gradient}, " 
                                              f" shapes: {a.shape}/{b.shape}"
                                              f" is_hessian: {a.is_hessian}/{b.is_hessian}")



            # ------------------------------------------------------------------
            # DOT   — special-cased branches for advection / mass terms --------
            # ------------------------------------------------------------------
            elif isinstance(op, Dot):
                b = stack.pop()
                a = stack.pop()
                res_var = new_var("dot")

                # print(f"Dot operation: a.role={a.role}, b.role={b.role}, "
                #       f"a.shape={a.shape}, b.shape={b.shape}, "
                #       f"is_vector: {a.is_vector}/{b.is_vector}, "
                #       f"is_gradient: {a.is_gradient}/{b.is_gradient}, "
                #       f"is_hessian: {a.is_hessian}/{b.is_hessian}")

                # Advection term: dot(grad(u_trial), u_k)
                if a.role == 'trial' and a.is_gradient and b.role == 'value' and b.is_vector:
                    body_lines.append("# Advection: dot(grad(Trial), Function)")
                    body_lines.append(
                        f"{res_var} = dot_grad_basis_vector({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    stack.append(StackItem(var_name=res_var, role='trial', shape=(a.shape[0], a.shape[1]), 
                                           is_vector=True, is_gradient=False, field_names=a.field_names, parent_name=a.parent_name, side =a.side,
                                           field_sides=a.field_sides or []))
               
                # Final advection term: dot(advection_vector_trial, v_test)
                elif (a.role == 'trial' and (not a.is_gradient and not a.is_hessian) and b.role == 'test' and (not b.is_gradient and not b.is_hessian) ):
                    body_lines.append(f"# Mass: dot(Trial, Test)")
                    body_lines.append(
                        f"{res_var} = dot_mass_test_trial({b.var_name}, {a.var_name}, {self.dtype})"
                    )
                    # collapsing all
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer=None, strict=False)
                    stack.append(StackItem(var_name=res_var, 
                                            role='value', 
                                            shape=(b.shape[1],a.shape[1]), 
                                            is_vector=False, field_names=field_names, 
                                            parent_name=parent_name, side=side, field_sides=field_sides))
                elif (a.role == 'test' and (not a.is_gradient and not a.is_hessian) and b.role == 'trial' and (not b.is_gradient and not b.is_hessian) ):
                    body_lines.append(f"# Mass: dot(Test, Trial)")
                    body_lines.append(
                        f"{res_var} = dot_mass_test_trial({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    # Collapsing all
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer=None, strict=False)
                    stack.append(StackItem(var_name=res_var, 
                                            role='value', 
                                            shape=(a.shape[1],b.shape[1]), 
                                            is_vector=False, field_names=field_names, 
                                            parent_name=parent_name, side=side, field_sides=field_sides))

                # ---------------------------------------------------------------------
                # dot( grad(u_test) ,  const_vec )  ← symmetric term -> Test vec
                # ---------------------------------------------------------------------
                elif a.role == 'test' and a.is_gradient and b.role == 'const' and b.is_vector:
                    k_comps = a.shape[0]; n_locs = a.shape[1]; d = a.shape[2]
                    if k_comps > 1: is_vector = True
                    else: is_vector = False
                    if a.shape[2] == b.shape[0]:
                        body_lines.append("# Symmetric term: dot(grad(Test), constant vector)")
                        body_lines.append(
                            f"{res_var} = dot_grad_basis_vector({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        stack.append(StackItem(var_name=res_var, role='test',
                                            shape=(a.shape[0], a.shape[1]), is_vector=is_vector,
                                            is_gradient=False, field_names=a.field_names,
                                            parent_name=a.parent_name,
                                            side=a.side, field_sides=a.field_sides))

                # ---------------------------------------------------------------------
                # dot( grad(u_trial) ,  beta )  ← convection term (Function gradient · Trial)
                # ---------------------------------------------------------------------
                elif a.role == 'trial' and a.is_gradient and b.role in {'const','value'} and b.is_vector:
                    k_comps = a.shape[0]; n_locs = a.shape[1]; d = a.shape[2]
                    if k_comps > 1: is_vector = True
                    else: is_vector = False 
                    if b.shape[0] == a.shape[2]:
                        body_lines.append("# Advection: dot(grad(Trial), constant beta vector)")
                        body_lines.append(
                            f"{res_var} = dot_grad_basis_vector({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        stack.append(StackItem(var_name=res_var, role='trial',
                                            shape=(a.shape[0], a.shape[1]), is_vector=is_vector,
                                            is_gradient=False, field_names=a.field_names,
                                            parent_name=a.parent_name,
                                            side=a.side, field_sides=a.field_sides))
                # ---------------------------------------------------------------------
                # dot( beta, grad(u_trial)  )  ← beta_i * B_{inj} → 
                # ---------------------------------------------------------------------
                elif b.role == 'trial' and b.is_gradient and a.role in {'const','value'} and a.is_vector:
                    k_comps = b.shape[0]; n_locs = b.shape[1]
                    is_vector = k_comps > 1
                    res_shape = (1, n_locs) if k_comps == 1 else (b.shape[2], n_locs)
                    body_lines.append("# Advection: dot(constant beta vector, grad(Trial))")
                    body_lines.append(
                        f"{res_var} = vector_dot_grad_basis({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    stack.append(StackItem(var_name=res_var, role='trial',
                                        shape=res_shape, is_vector=is_vector,
                                        is_gradient=False, field_names=b.field_names,
                                        parent_name=b.parent_name,
                                        side=b.side, field_sides=b.field_sides))


                # ---------------------------------------------------------------------
                # dot( grad(u_k) ,  u_trial )  ← convection term (Function gradient · Trial)
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_gradient and b.role == 'trial' and b.is_vector:
                    body_lines.append("# Advection: dot(grad(Function), Trial)")
                    body_lines.append(
                        f"{res_var} = dot_grad_func_trial_vec({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    stack.append(StackItem(var_name=res_var, role='trial',
                                        shape=(b.shape[0], b.shape[1]), is_vector=True,
                                        is_gradient=False, field_names=b.field_names,
                                        parent_name=b.parent_name,
                                        side=b.side, field_sides=b.field_sides))

                # ---------------------------------------------------------------------
                # dot( u_trial ,  grad(u_k) )   ← swap of the previous
                # ---------------------------------------------------------------------
                elif a.role == 'trial' and a.is_vector and b.role == 'value' and b.is_gradient:
                    body_lines.append("# Advection: dot(Trial, grad(Function))")
                    body_lines.append(
                        f"{res_var} = dot_trial_vec_grad_func({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    stack.append(StackItem(var_name=res_var, role='trial',
                                        shape=(b.shape[1], a.shape[1]), is_vector=True,
                                        is_gradient=False, field_names=a.field_names,
                                        parent_name=a.parent_name,
                                        side=a.side, field_sides=a.field_sides or []))

                # ---------------------------------------------------------------------
                # dot( u_k ,  u_k )             ← |u_k|², scalar
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_vector and b.role == 'value' and b.is_vector:
                    body_lines.append("# Non-linear term: dot(Function, Function)")
                    body_lines.append(
                        f"{res_var} = dot_vec_vec({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    stack.append(StackItem(var_name=res_var, role='value',
                                        shape=(), is_vector=False, side = a.side,
                                        is_gradient=False, field_names=a.field_names,
                                        parent_name=a.parent_name, field_sides=a.field_sides or []))

                # ---------------------------------------------------------------------
                # dot( u_k ,  grad(u_trial) )   ← usually zero for skew-symm forms
                # ---------------------------------------------------------------------
                elif a.role in {'value','const'} and a.is_vector and b.role in {'trial','test'} and b.is_gradient:
                    k_comps = b.shape[0]; n_locs = b.shape[1]
                    role_b = "trial" if b.role == "trial" else "test"
                    is_vector = k_comps > 1
                    res_shape = (1, n_locs) if k_comps == 1 else (b.shape[2], n_locs)
                    body_lines.append(f"# dot(Function, grad({role_b.capitalize()}))")
                    body_lines.append(
                        f"{res_var} = vector_dot_grad_basis({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    stack.append(StackItem(var_name=res_var, role=role_b,
                                        shape=res_shape, is_vector=is_vector,
                                        is_gradient=False, field_names=b.field_names,
                                        parent_name=b.parent_name, side=b.side,
                                        field_sides=b.field_sides or []))

                # ---------------------------------------------------------------------
                # dot( u_test ,  u_trial )      ← mass-matrix block
                # ---------------------------------------------------------------------
                elif a.role == 'test' and a.is_vector and b.role == 'trial' and b.is_vector:
                    body_lines.append("# Mass: dot(Test, Trial)")
                    body_lines.append(
                        f"{res_var} = dot_mass_test_trial({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    # collapsing all
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer=None, strict=False)
                    stack.append(StackItem(var_name=res_var, role='value',
                                        shape=(a.shape[1],b.shape[1]), is_vector=False,
                                        field_names=field_names, parent_name=parent_name, side=side,
                                        field_sides=field_sides or []))

                # ---------------------new block--------------------------------
                # ---------------------------------------------------------------------
                # dot(grad(Trial/Test), grad(Function)) and its transposed variants.
                # ---------------------------------------------------------------------
                elif (a.role in {'trial', 'test'} and a.is_gradient and b.role == 'value' 
                    and b.is_gradient 
                    and a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1]): 
                    role = "trial" if a.role == "trial" else "test"
                    k = a.shape[0]; n_locs = a.shape[1]; d = a.shape[2]
                    
                    # a: grad(du) or grad(du).T -> Trial function basis, shape (k, n, d)
                    # b: grad(u_k)             -> Function value, shape (k, d)
                    

                    body_lines.append(f"# dot(grad({role}), grad(value)) -> (k,n,k) tensor basis")
                    body_lines.append(
                        f"{res_var} = dot_grad_basis_with_grad_value({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    
                    res_shape = (k, n_locs, k)
                    stack.append(StackItem(var_name=res_var, role=role,
                                        shape=res_shape, is_vector=False, is_gradient=True,
                                        field_names=a.field_names, parent_name=a.parent_name,
                                        side=a.side, field_sides=a.field_sides or []))
                # ---------------------------------------------------------------------
                # dot(grad(Trial/Test), grad(Trial/Test)) and its transposed variants.
                # ---------------------------------------------------------------------
                elif (a.role in {'trial', 'test'} and a.is_gradient and b.role in {'trial', 'test'} 
                    and b.is_gradient 
                    and a.shape==b.shape and a.shape[0] == a.shape[-1]): 
                    is_a_trial = a.role == "trial"
                    is_b_test = b.role == "test"
                    is_a_test = a.role == "test"
                    is_b_trial = b.role == "trial"
                    # a: grad(du) or grad(du).T -> Trial function basis, shape (k, n, d)
                    # b: grad(du) or grad(du).T -> Trial function basis, shape (k, n, d)
                    # c: mixed gradients (k,m,n,k) tensor basis
                    flag = 1
                    if is_a_trial and is_b_test:
                        flag = 1
                    elif is_a_test and is_b_trial:
                        flag = 2
                    else:
                        raise NotImplementedError("dot(grad(Trial/Test), grad(Trial/Test)) only implemented for Trial-Test or Test-Trial combinations.")

                    if flag == 1:
                        res_shape = (a.shape[0], b.shape[1], a.shape[1], a.shape[0])  # (k, m, n, k)
                    else:
                        res_shape = (a.shape[0], a.shape[1], b.shape[1], a.shape[0])  # (k, m, n, k)
                    body_lines.append(f"# dot(grad({a.role}), grad({b.role})) -> (k,m,n,k) tensor basis")

                    body_lines.append(f"# Call helper: dot_grad_grad_mixed")
                    body_lines.append(
                        f"{res_var} = dot_grad_grad_mixed({a.var_name}, {b.var_name}, {flag}, {self.dtype})"
                    )
                    
                    stack.append(StackItem(var_name=res_var, role="mixed",
                                        shape=res_shape, is_vector=False, is_gradient=True,
                                        field_names=a.field_names, parent_name=a.parent_name,
                                        side=a.side, field_sides=a.field_sides or []))

                # ---------------------------------------------------------------------
                # dot(grad(Function), grad(Trial/Test)) and its transposed variants.
                # ---------------------------------------------------------------------
                elif (a.role == 'value' and a.is_gradient and 
                    b.role in {'trial', 'test'} and b.is_gradient and 
                    a.shape[0] == b.shape[0] and a.shape[1] == b.shape[2]):
                    k = b.shape[0]; n_locs = b.shape[1]; d = b.shape[2]
                    role = "trial" if b.role == "trial" else "test"

                    # a: grad(u_k) or grad(u_k).T -> Function value, shape (k, d)
                    # b: grad(du)             -> Trial function basis, shape (k, n, d)

                    body_lines.append(f"# dot(grad(value), grad({role})) -> (k,n,d) tensor basis")
                    body_lines.append(
                        f"{res_var} = dot_grad_value_with_grad_basis({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    
                    res_shape = (k, n_locs, d)
                    stack.append(StackItem(var_name=res_var, role=role,
                                        shape=res_shape, is_vector=False, is_gradient=True,
                                        field_names=b.field_names, parent_name=b.parent_name,
                                        side=b.side, field_sides=b.field_sides or []))

                # ---------------------------------------------------------------------
                # dot(grad(Function), grad(Function)) and its transposed variants.
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_gradient and b.role == 'value' and b.is_gradient:
                    # a: grad(u_k) or grad(u_k).T -> Function value, shape (k, d)
                    # b: grad(u_k) or grad(u_k).T -> Function value, shape (k, d)
                    # This block handles various combinations like dot(A, B), dot(A.T, B), etc.
                    # The generated code assumes k == d.
                    body_lines.append("# dot(grad(value), grad(value)) -> (k,k) tensor value")
                    body_lines.append(
                        f"{res_var} = dot_grad_grad_value({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    res_shape = (a.shape[0], b.shape[1])
                    stack.append(StackItem(var_name=res_var, role='value',
                                        shape=res_shape, is_vector=False, is_gradient=True,
                                        field_names=b.field_names, parent_name=b.parent_name,
                                        side=b.side, field_sides=b.field_sides or []))

                # ---------------------new block--------------------------------
                # ---------------------------------------------------------------------
                # dot( scalar ,  u_trial;u_test;u_k )     ← e.g. scalar constant time Function
                # ---------------------------------------------------------------------
                elif (a.role == 'const' or a.role == 'value') and not a.is_vector and not a.is_gradient and not a.is_hessian and a.shape == ():
                    # a is scalar, b is vector (trial/test/Function)
                    body_lines.append("# Scalar constant: dot(scalar, Function/Trial/Test)")
                    body_lines.append(
                        f"{res_var} = mul_scalar({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    stack.append(StackItem(var_name=res_var, role=b.role,
                                        shape=b.shape, is_vector=b.is_vector,
                                        is_gradient=b.is_gradient, 
                                        is_hessian=b.is_hessian, 
                                        field_names=b.field_names,
                                        parent_name=b.parent_name,
                                        side=b.side, field_sides=b.field_sides or []))
                # ---------------------------------------------------------------------
                # dot( u_trial;u_test;u_k, scalar )     ← e.g. scalar constant time Function
                # ---------------------------------------------------------------------
                elif (b.role in {'const','value'} ) and not b.is_vector and not b.is_gradient and not b.is_hessian and b.shape == ():
                    # a is vector (trial/test/Function), b is scalar
                    body_lines.append("# Scalar constant: dot(Function/Trial/Test, scalar)")
                    body_lines.append(
                        f"{res_var} = mul_scalar({b.var_name}, {a.var_name}, {self.dtype})"
                    )
                    stack.append(StackItem(var_name=res_var, role=a.role,
                                        shape=a.shape, is_vector=a.is_vector,
                                        is_gradient=a.is_gradient, 
                                        is_hessian=a.is_hessian, 
                                        field_names=a.field_names,
                                        parent_name=a.parent_name,
                                        side=a.side, field_sides=a.field_sides or []))

                # ---------------------------------------------------------------------
                # dot( u_k ,  grad(u_k) )     ← e.g. rhs advection term
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_vector and b.role == 'value' and b.is_gradient:
                    body_lines.append("# RHS: dot(Function, grad(Function)) (k).(k,d) ->k")
                    body_lines.append(
                        f"{res_var} = dot_value_with_grad({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    stack.append(StackItem(var_name=res_var, role='const',
                                        shape=( b.shape[1],), is_vector=True,
                                        is_gradient=False, field_names=b.field_names,
                                        parent_name=b.parent_name,
                                        side=b.side, field_sides=b.field_sides or []))
                # ---------------------------------------------------------------------
                # dot( grad(u_k) ,  u_k )     ← e.g. rhs advection term  -> (k,d).(k) -> k
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_gradient and b.role == 'value' and b.is_vector:
                    body_lines.append("# RHS: dot(grad(Function), Function) (k,d).(k) -> k")
                    body_lines.append(
                        f"{res_var} = dot_grad_with_value({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    stack.append(StackItem(var_name=res_var, role='const',
                                        shape=(a.shape[0], ), is_vector=True,
                                        is_gradient=False, field_names=a.field_names,
                                        parent_name=a.parent_name,
                                        side=a.side, field_sides=a.field_sides or []))
                # ---------------------------------------------------------------------
                # dot( np.array ,  u_test )     ← e.g. body-force · test -> (n,)
                # ---------------------------------------------------------------------
                elif a.role == 'const' and a.is_vector and b.role == 'test' and b.is_vector:
                    # a (k) and b (k,n)
                    body_lines.append("# Constant body-force: dot(const-vec, Test)")
                    body_lines.append(
                        f"{res_var} = const_vector_dot_basis_1d({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    stack.append(StackItem(var_name=res_var, role='value',
                                        shape=(b.shape[1],), is_vector=False,
                                        is_gradient=False, field_names=b.field_names,
                                        parent_name=b.parent_name,
                                        side=b.side, field_sides=b.field_sides or []))
                elif a.role == 'trial' and a.is_vector and not a.is_gradient and b.role == 'test' and b.is_gradient:
                    body_lines.append("# Dot: vector Trial · Grad(Test) → mixed tensor")
                    body_lines.append(
                        f"{res_var} = dot_vector_trial_grad_test({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    res_shape = (b.shape[2], b.shape[1], a.shape[1])
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='b', strict=False)
                    stack.append(StackItem(var_name=res_var, role='mixed',
                                        shape=res_shape, is_vector=False, is_gradient=False,
                                        field_names=field_names, parent_name=parent_name,
                                        side=side, field_sides=field_sides))
                elif (a.role in {'trial', 'test'} and a.is_gradient and not a.is_hessian
                      and b.role in {'trial', 'test'} and b.is_vector and not b.is_gradient and not b.is_hessian):
                    body_lines.append("# Dot: Grad(basis) · vector basis → mixed tensor")
                    body_lines.append(
                        f"{res_var} = contract_last_first({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    res_shape = a.shape[:-1] + b.shape[1:]
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                    stack.append(StackItem(var_name=res_var, role='mixed',
                                        shape=res_shape, is_vector=False, is_gradient=False,
                                        field_names=field_names, parent_name=parent_name,
                                        side=side, field_sides=field_sides))
                elif (a.role == 'mixed' and not a.is_gradient and not a.is_hessian
                      and len(a.shape) == 3 and b.role in {'const','value'} and b.is_vector and not b.is_gradient and not b.is_hessian):
                    body_lines.append("# Dot: mixed basis · constant vector → matrix")
                    body_lines.append(
                        f"{res_var} = dot_mixed_const({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    res_shape = (a.shape[1], a.shape[2])
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                    stack.append(StackItem(var_name=res_var, role='value',
                                        shape=res_shape, is_vector=False, is_gradient=False,
                                        field_names=field_names, parent_name=parent_name,
                                        side=side, field_sides=field_sides))
                elif (a.role in {'const','value'} and a.is_vector and not a.is_gradient and not a.is_hessian
                      and len(a.shape) == 1 and b.role == 'mixed' and not b.is_gradient and not b.is_hessian and len(b.shape) == 3):
                    body_lines.append("# Dot: constant vector · mixed basis → matrix")
                    body_lines.append(
                        f"{res_var} = dot_const_mixed({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    res_shape = (b.shape[1], b.shape[2])
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='b', strict=False)
                    stack.append(StackItem(var_name=res_var, role='value',
                                        shape=res_shape, is_vector=False, is_gradient=False,
                                        field_names=field_names, parent_name=parent_name,
                                        side=side, field_sides=field_sides))

                # ---------------------------------------------------------------------
                # dot( u_k ,  u_test )          ← load-vector term -> (n,)
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_vector and b.role == 'test' and b.is_vector:
                    if self.form_rank == 2:
                        body_lines.append("# LHS: dot(Function, Test) (k,n)·(k) -> (1,n)")
                        shape = (1, b.shape[1])
                        role = 'test'
                        body_lines.append(
                            f"{res_var} = columnwise_dot({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                    elif self.form_rank == 1:
                        body_lines.append("# RHS: dot(Function, Test)")
                        body_lines.append(
                            f"{res_var} = dot_mass_trial_test({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        shape = (b.shape[1],)
                        role = 'value'
                    stack.append(StackItem(var_name=res_var, role=role,
                                        shape=shape, is_vector=False,is_gradient=False,
                                        field_names=b.field_names, parent_name=b.parent_name,
                                        side=b.side, field_sides=b.field_sides or []))

                # ---------------------------------------------------------------------
                # dot( u_test ,  u_k )          ← load-vector term -> (n,)
                # ---------------------------------------------------------------------
                elif a.role == 'test' and a.is_vector and b.role == 'value' and b.is_vector:
                    if self.form_rank == 2:
                        body_lines.append("# LHS: dot(Test, Function) (k,n)·(k) -> (1,n)")
                        shape = (1, a.shape[1])
                        role = 'test'
                        body_lines.append(
                            f"{res_var} = columnwise_dot({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                    elif self.form_rank == 1:
                        body_lines.append("# RHS: dot(Test, Function)")
                        body_lines.append(
                            f"{res_var} = dot_mass_trial_test({b.var_name}, {a.var_name}, {self.dtype})"
                        )
                        shape = (a.shape[1],)
                        role = 'value'
                    stack.append(StackItem(var_name=res_var, role=role,
                                        shape=shape, is_vector=False,is_gradient=False,
                                        field_names=a.field_names, parent_name=a.parent_name,
                                        side=a.side, field_sides=a.field_sides or []))
                # ---------------------------------------------------------------------
                # dot( u_test ,  const_vec )          ← load-vector term -> (n,)
                # ---------------------------------------------------------------------
                elif a.role == 'test' and a.is_vector and b.role == 'const' and b.is_vector:
                    if self.form_rank == 2:
                        body_lines.append("# LHS: dot(Test, Const) (k,n)·(k) -> (1,n)")
                        shape = (1, a.shape[1])
                        role = 'test'
                        body_lines.append(
                            f"{res_var} = basis_dot_const_vector({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        stack.append(StackItem(var_name=res_var, role=role,
                                            shape=shape, is_vector=False,is_gradient=False,
                                            field_names=a.field_names, parent_name=a.parent_name,
                                            side=a.side, field_sides=a.field_sides or []))
                    elif self.form_rank == 1:
                        body_lines.append("# RHS: dot(Test, Const)")
                        body_lines.append(
                            f"{res_var} = dot_mass_trial_test({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        shape = (a.shape[1],)
                        role = 'value'
                        stack.append(StackItem(var_name=res_var, role=role,
                                            shape=shape, is_vector=False,is_gradient=False,
                                            field_names=a.field_names, parent_name=a.parent_name,
                                            side=a.side, field_sides=a.field_sides or []))
                # ---------------------------------------------------------------------
                # dot( u_trial ,  const_vec )          ← load-vector term -> (1,n)
                # ---------------------------------------------------------------------
                elif a.role == 'trial' and a.is_vector and b.role == 'const' and b.is_vector:
                   body_lines.append("# LHS: dot(Trial, Const) (k,n)·(k) -> (1,n)")
                   shape = (1, a.shape[1])
                   role = 'trial'
                   body_lines.append(
                           f"{res_var} = basis_dot_const_vector({a.var_name}, {b.var_name}, {self.dtype})"
                   )
                   stack.append(StackItem(var_name=res_var, role=role,
                                        shape=shape, is_vector=False,is_gradient=False,
                                        field_names=a.field_names, parent_name=a.parent_name,
                                        side=a.side, field_sides=a.field_sides or []))


                # ---------------------------------------------------------------------
                # dot( Hessian ,  value/const )          ← Grad object
                # ---------------------------------------------------------------------
                # --- Hessian · vector (right) and vector · Hessian (left) ---
                # Contractions only with constant geometric vectors (e.g., facet normals)
                elif a.is_hessian and b.role in {'const','value'} and b.is_vector:
                    k_comps = a.shape[0]
                    if a.role in ('test', 'trial'):
                        body_lines.append("# Dot: Hessian(basis) · const spatial vec -> (k, n, d1)")
                        body_lines.append(
                            f"{res_var} = hessian_dot_vector({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        stack.append(StackItem(var_name=res_var, role=a.role,
                                            shape=(k_comps, a.shape[1], a.shape[2]),  # (k, n, d1)
                                            is_vector=False, is_gradient=True, is_hessian=False,
                                            field_names=a.field_names, parent_name=a.parent_name, side=a.side, field_sides=a.field_sides or []))
                    else:
                        body_lines.append("# Dot: Hessian(value) · const spatial vec -> (k, d1)")
                        body_lines.append(
                            f"{res_var} = hessian_dot_vector({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        stack.append(StackItem(var_name=res_var, role=a.role,
                                            shape=(k_comps, a.shape[1]),      # (k, d1)
                                            is_vector=False, is_gradient=True, is_hessian=False,
                                            field_names=a.field_names, parent_name=a.parent_name, side=a.side, field_sides=a.field_sides or []))


                elif b.is_hessian and a.role in {'const','value'} and a.is_vector:
                    if b.role in ('test', 'trial'):
                        body_lines.append("# Dot: vector · Hessian(basis)")
                        body_lines.append(
                            f"{res_var} = vector_dot_hessian_basis({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        if a.shape[0] == b.shape[0] and b.shape[0] > 1:
                            res_shape = (b.shape[2], b.shape[1], b.shape[3])
                        elif a.shape[0] == b.shape[2] and b.shape[0] == 1:
                            res_shape = (1, b.shape[1], b.shape[3])
                        else:
                            res_shape = (b.shape[2], b.shape[1], b.shape[3])
                        stack.append(StackItem(var_name=res_var, role=b.role,
                                            shape=res_shape,
                                            is_vector=False, is_gradient=True, is_hessian=False,
                                            field_names=b.field_names, parent_name=b.parent_name, side=b.side,
                                            field_sides=b.field_sides or []))
                    else:
                        body_lines.append("# Dot: vector · Hessian(value)")
                        body_lines.append(
                            f"{res_var} = vector_dot_hessian_value({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        if a.shape[0] == b.shape[0] and b.shape[0] > 1:
                            res_shape = (b.shape[1], b.shape[2])
                        elif a.shape[0] == b.shape[1] and b.shape[0] == 1:
                            res_shape = (1, b.shape[2])
                        else:
                            res_shape = (b.shape[1], b.shape[2])
                        stack.append(StackItem(var_name=res_var, role=b.role,
                                            shape=res_shape,
                                            is_vector=False, is_gradient=True, is_hessian=False,
                                            field_names=b.field_names, parent_name=b.parent_name, side=b.side,
                                            field_sides=b.field_sides or []))


                
                # ---------------------------------------------------------------------
                # dot( value/const ,  value/const )          ← load-vector term -> (n,)
                # ---------------------------------------------------------------------
                elif (a.role in ('const', 'value') and   
                     b.role in ('const', 'value') ):
                    if a.is_gradient and b.is_vector:
                        body_lines.append("# Dot: grad(scalar) * const vector → const vector")
                        # print(f" a.shape: {a.shape}, b.shape: {b.shape}, a.is_vector: {a.is_vector}, b.is_vector: {b.is_vector}, a.is_gradient: {a.is_gradient}, b.is_gradient: {b.is_gradient}")
                        if a.shape == b.shape:
                            body_lines.append(
                                f"{res_var} = dot_vec_vec({a.var_name}, {b.var_name}, {self.dtype})"
                            )
                            shape = ()
                            is_vector = False; is_grad = False
                        else:
                            body_lines.append(
                                f"{res_var} = contract_last_first({a.var_name}, {b.var_name}, {self.dtype})"
                            )
                            shape = (a.shape[0],)
                            is_vector = True; is_grad = False
                    elif a.is_vector and b.is_gradient:
                        body_lines.append("# Dot: const vector * grad(scalar) → const vector")
                        if a.shape == b.shape:
                            body_lines.append(f"# shape a and shape are equal result should be scalar")
                            body_lines.append(
                                f"{res_var} = dot_vec_vec({a.var_name}, {b.var_name}, {self.dtype})"
                            )
                            shape = ()
                            is_vector = False; is_grad = False
                        elif len(b.shape) == 2 and b.shape[0] ==1 and len(a.shape) == 1 and a.shape[0] == b.shape[1]:
                            body_lines.append(f"# shape a and shape are equal result should be scalar; special case for rhs")
                            body_lines.append(
                                f"{res_var} = dot_vec_vec({a.var_name}, {b.var_name}[0, :], {self.dtype})"
                            )
                            shape = ()
                            is_vector = False; is_grad = False
                        else:
                            body_lines.append(f"# shape a and shape are not equal result should be vector")
                            body_lines.append(
                                f"{res_var} = contract_last_first({a.var_name}, {b.var_name}, {self.dtype})"
                            )
                            shape = (b.shape[0],)
                            is_vector = True; is_grad = False
                    elif a.is_vector and b.is_vector:
                        body_lines.append("# Dot: vector * vector → scalar")
                        body_lines.append(
                            f"{res_var} = dot_vec_vec({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        shape = ()
                        is_vector = False; is_grad = False
                    elif a.is_gradient and  b.is_gradient:
                        body_lines.append("# Dot: grad(scalar) * grad(scalar) → scalar")
                        body_lines.append(
                            f"{res_var} = dot_vec_vec({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        shape = ()
                        is_vector = False; is_grad = False
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                    stack.append(StackItem(var_name=res_var, role='const',
                                        shape=shape, is_vector=is_vector, 
                                        is_gradient=is_grad,
                                        field_names=field_names,
                                        parent_name=parent_name, side=side,
                                        field_sides=field_sides))
                elif (len(a.shape) >= 1 and len(b.shape) >= 1
                      and not a.is_hessian and not b.is_hessian
                      and a.shape[-1] == b.shape[0]):
                    body_lines.append("# Dot: generic contraction (last axis of A · first axis of B)")
                    body_lines.append(
                        f"{res_var} = contract_last_first({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    res_shape = a.shape[:-1] + b.shape[1:]
                    if 'mixed' in (a.role, b.role):
                        res_role = 'mixed'
                    elif a.role in {'trial', 'test'} and b.role in {'trial', 'test'}:
                        res_role = 'mixed'
                    elif a.role in {'trial', 'test'}:
                        res_role = a.role
                    elif b.role in {'trial', 'test'}:
                        res_role = b.role
                    elif a.role in {'value', 'function', 'identity'} or b.role in {'value', 'function', 'identity'}:
                        res_role = 'value' if 'value' in (a.role, b.role) else ('function' if 'function' in (a.role, b.role) else 'identity')
                    elif a.role == 'const' and b.role != 'const':
                        res_role = b.role
                    else:
                        res_role = a.role
                    res_is_vector = False
                    if res_role in {'trial', 'test'} and len(res_shape) >= 1:
                        res_is_vector = True
                    elif res_role in {'value', 'function', 'identity'} and len(res_shape) == 1:
                        res_is_vector = True
                    res_is_gradient = a.is_gradient or b.is_gradient
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                    stack.append(StackItem(var_name=res_var, role=res_role,
                                        shape=res_shape, is_vector=res_is_vector, is_gradient=res_is_gradient,
                                        field_names=field_names, parent_name=parent_name,
                                        side=side, field_sides=field_sides))
                else:
                    raise NotImplementedError(f"Dot not implemented for roles {a.role}/{b.role} with shapes {a.shape}/{b.shape}"
                                              f" with vectoors {a.is_vector}/{b.is_vector}"
                                              f" and gradients {a.is_gradient}/{b.is_gradient}"
                                              f" also with hessians {a.is_hessian}/{b.is_hessian}"
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
                    body_lines.append(
                        f"{res} = transpose_grad_tensor({a.var_name}, {self.dtype})"
                    )
                    res_shape = a.shape  # still (2,n,2)
                elif a.role == 'mixed' and a.is_gradient and len(a.shape) == 4:
                    body_lines.append("# Transpose mixed gradient: swap component and spatial axes")
                    body_lines.append(
                        f"{res} = transpose_mixed_grad_tensor({a.var_name}, {self.dtype})"
                    )
                    res_shape = a.shape  # (k, n, m, d)
                
                # -------- Hessian tensor: (k,n,2,2) -> swap last two axes --------
                elif a.is_hessian and len(a.shape) == 4:
                    body_lines.append(
                        f"{res} = transpose_hessian_tensor({a.var_name}, {self.dtype})"
                    )
                    res_shape = a.shape  # unchanged (k,n,d,d)

                # -------- plain 2×2 matrix --------------------------------------
                elif len(a.shape) == 2 :
                    body_lines.append(
                        f"{res} = transpose_matrix({a.var_name}, {self.dtype})"
                    )
                    res_shape = (a.shape[1], a.shape[0])

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

                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer="a", strict=False)
                        stack.append(StackItem(var_name=res_var, role='const', shape=shape, is_vector=False, 
                                               field_names=field_names, parent_name=parent_name, side=side,
                                               field_sides=field_sides))

                    # -----------------------------------------------------------------
                    # 01. Vector, Tensor:   scalar   *  Vector/Tensor    →  Vector/Tensor 
                    # -----------------------------------------------------------------
                    
                    elif ((a.role == 'const' or a.role=='value')  
                         and 
                          (not a.is_vector and not a.is_gradient and not a.is_hessian)
                          and a.shape == ()) :
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
                        (not b.is_vector and not b.is_gradient and not b.is_hessian)
                          and b.shape == ()): 
                        body_lines.append("# Product: Vector/Tensor * scalar → Vector/Tensor")
                        # b is scalar, a is vector/tensor
                        _mul_scalar_vector(self,first_is_scalar=False, a=a, b=b, res_var=res_var, body_lines=body_lines,stack=stack)
                        
                    
                    # -----------------------------------------------------------------
                    # 1. LHS block:   scalar test  *  scalar trial   →  outer product
                    # -----------------------------------------------------------------
                    elif (not a.is_gradient and not b.is_gradient and
                          not a.is_hessian and not b.is_hessian and
                        ((a.role, b.role) == ("test", "trial") or
                        (a.role, b.role) == ("trial", "test"))):
                        n_locs = a.shape[1]
                        body_lines.append("# Product: scalar Test × scalar Trial → mixed (1,n,n)")

                        # orient rows = test , columns = trial
                        test_var  = a if a.role == "test"  else b
                        trial_var = b if a.role == "test"  else a

                        body_lines += [
                            f"{res_var} = dot_mass_test_trial({test_var.var_name}, {trial_var.var_name}, {self.dtype})",
                            f"{res_var} = {res_var}[np.newaxis, :, :]",
                        ]
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer=None, strict=False)
                        stack.append(StackItem(var_name=res_var, role='mixed',
                                            shape=(1, n_locs, n_locs), is_vector=False,
                                            field_names=field_names, parent_name=parent_name, side=side,
                                            field_sides=field_sides))
                    # -----------------------------------------------------------------
                    # 2. LHS block:   scalar trial/test  *  vector   →  vector trial/test
                    # -----------------------------------------------------------------
                    elif (a.role in {"trial", "test"} and not a.is_vector and not a.is_gradient and not a.is_hessian and a.shape[0] == 1
                          and b.role in {"value", "const"} and b.is_vector):
                        role = 'trial' if a.role == 'trial' else 'test'
                        body_lines.append("# Product: scalar Trial/Test × vector → vector Trial/Test")
                        body_lines.append(
                            f"{res_var} = scalar_basis_times_vector({a.var_name}, {b.var_name}, {self.dtype})"
                        )

                        stack.append(StackItem(var_name=res_var, role=role,
                                            shape=(b.shape[0],a.shape[1]), is_vector=True, field_names=a.field_names,
                                            parent_name=a.parent_name, side=a.side, field_sides=a.field_sides))
                    elif (b.role in {"trial", "test"} and not b.is_vector and not b.is_gradient and not b.is_hessian and b.shape[0] == 1
                          and a.role in {"value", "const"} and a.is_vector):
                        role = 'trial' if b.role == 'trial' else 'test'
                        body_lines.append("# Product: vector × scalar Trial/Test → vector Trial/Test")
                        body_lines.append(
                            f"{res_var} = scalar_basis_times_vector({b.var_name}, {a.var_name}, {self.dtype})"
                        )
                        stack.append(StackItem(var_name=res_var, role=role,
                                            shape=(a.shape[0],b.shape[1]), is_vector=True, field_names=b.field_names,
                                            parent_name=b.parent_name, side=b.side, field_sides=b.field_sides))
                    elif (a.role in {"value","const"} and not a.is_vector and not a.is_gradient and not a.is_hessian and len(a.shape) == 2
                          and b.role in {"trial","test"} and not b.is_vector and not b.is_gradient and not b.is_hessian and len(b.shape) == 2 and b.shape[0] == 1):
                        role = 'value'
                        body_lines.append("# Product: matrix Value × scalar Trial/Test → scalar Trial/Test contribution")
                        body_lines.append(
                            f"{res_var} = matrix_times_scalar_basis({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                        stack.append(StackItem(var_name=res_var, role=role,
                                            shape=(1, b.shape[1]), is_vector=False,
                                            field_names=field_names, parent_name=parent_name,
                                            side=side, field_sides=field_sides))
                    elif (b.role in {"value","const"} and not b.is_vector and not b.is_gradient and not b.is_hessian and len(b.shape) == 2
                          and a.role in {"trial","test"} and not a.is_vector and not a.is_gradient and not a.is_hessian and len(a.shape) == 2 and a.shape[0] == 1):
                        role = 'value'
                        body_lines.append("# Product: scalar Trial/Test × matrix Value → scalar Trial/Test contribution")
                        body_lines.append(
                            f"{res_var} = matrix_times_scalar_basis({b.var_name}, {a.var_name}, {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='b', strict=False)
                        stack.append(StackItem(var_name=res_var, role=role,
                                            shape=(1, a.shape[1]), is_vector=False,
                                            field_names=field_names, parent_name=parent_name,
                                            side=side, field_sides=field_sides))
                    # -----------------------------------------------------------------
                    # Scalar Trial/Test × Grad(Test/Trial) → mixed gradient tensor
                    # -----------------------------------------------------------------
                    elif (a.role == 'trial' and not a.is_vector and not a.is_gradient and not a.is_hessian and len(a.shape) == 2 and a.shape[0] == 1
                          and b.role == 'test' and b.is_gradient and len(b.shape) == 3):
                        body_lines.append("# Product: scalar Trial × Grad(Test) → mixed gradient")
                        body_lines.append(
                            f"{res_var} = scalar_trial_times_grad_test({b.var_name}, {a.var_name}[0], {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='b', strict=False)
                        stack.append(StackItem(var_name=res_var, role='mixed',
                                            shape=(b.shape[0], b.shape[1], a.shape[1], b.shape[2]),
                                            is_vector=False, is_gradient=True,
                                            field_names=field_names, parent_name=parent_name,
                                            side=side, field_sides=field_sides))
                    elif (a.role == 'test' and not a.is_vector and not a.is_gradient and not a.is_hessian and len(a.shape) == 2 and a.shape[0] == 1
                          and b.role == 'trial' and b.is_gradient and len(b.shape) == 3):
                        body_lines.append("# Product: scalar Test × Grad(Trial) → mixed gradient")
                        body_lines.append(
                            f"{res_var} = grad_trial_times_scalar_test({b.var_name}, {a.var_name}[0], {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                        stack.append(StackItem(var_name=res_var, role='mixed',
                                            shape=(b.shape[0], a.shape[1], b.shape[1], b.shape[2]),
                                            is_vector=False, is_gradient=True,
                                            field_names=field_names, parent_name=parent_name,
                                            side=side, field_sides=field_sides))
                    elif (a.role == 'test' and a.is_gradient and len(a.shape) == 3
                          and b.role == 'trial' and not b.is_vector and not b.is_gradient and not b.is_hessian and len(b.shape) == 2 and b.shape[0] == 1):
                        body_lines.append("# Product: Grad(Test) × scalar Trial → mixed gradient")
                        body_lines.append(
                            f"{res_var} = scalar_trial_times_grad_test({a.var_name}, {b.var_name}[0], {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                        stack.append(StackItem(var_name=res_var, role='mixed',
                                            shape=(a.shape[0], a.shape[1], b.shape[1], a.shape[2]),
                                            is_vector=False, is_gradient=True,
                                            field_names=field_names, parent_name=parent_name,
                                            side=side, field_sides=field_sides))
                    elif (a.role == 'trial' and a.is_gradient and len(a.shape) == 3
                          and b.role == 'test' and not b.is_vector and not b.is_gradient and not b.is_hessian and len(b.shape) == 2 and b.shape[0] == 1):
                        body_lines.append("# Product: Grad(Trial) × scalar Test → mixed gradient")
                        body_lines.append(
                            f"{res_var} = grad_trial_times_scalar_test({a.var_name}, {b.var_name}[0], {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='b', strict=False)
                        stack.append(StackItem(var_name=res_var, role='mixed',
                                            shape=(a.shape[0], b.shape[1], a.shape[1], a.shape[2]),
                                            is_vector=False, is_gradient=True,
                                            field_names=field_names, parent_name=parent_name,
                                            side=side, field_sides=field_sides))
                    elif (a.role in {"trial", "test"} and not a.is_vector and not a.is_gradient and not a.is_hessian and len(a.shape) == 1
                          and b.role in {"value", "const"} and b.is_vector):
                        role = 'trial' if a.role == 'trial' else 'test'
                        body_lines.append("# Product: scalar Trial/Test × vector → vector Trial/Test")
                        body_lines.append(
                            f"{res_var} = scalar_vector_outer_product({a.var_name}, {b.var_name}, {self.dtype})"
                        )

                        stack.append(StackItem(var_name=res_var, role=role,
                                            shape=(b.shape[0],a.shape[0]), is_vector=True, field_names=a.field_names,
                                            parent_name=a.parent_name, side=a.side, field_sides=a.field_sides))
                    elif (a.role == 'mixed' and not a.is_gradient and not a.is_hessian and len(a.shape) == 3
                          and b.role in {'value','const'} and b.is_gradient and len(b.shape) == 2):
                        body_lines.append("# Product: mixed basis × gradient matrix → mixed gradient")
                        body_lines.append(
                            f"{res_var} = scale_mixed_basis_with_coeffs({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                        stack.append(StackItem(var_name=res_var, role='mixed',
                                            shape=(b.shape[0], a.shape[1], a.shape[2], b.shape[1]),
                                            is_vector=False, is_gradient=True,
                                            field_names=field_names, parent_name=parent_name,
                                            side=side, field_sides=field_sides))
                    elif (a.role in {'value','const'} and a.is_gradient and not a.is_hessian and len(a.shape) == 2
                          and b.role == 'mixed' and not b.is_gradient and not b.is_hessian and len(b.shape) == 3):
                        body_lines.append("# Product: gradient matrix × mixed basis → mixed gradient")
                        body_lines.append(
                            f"{res_var} = scale_mixed_basis_with_coeffs({b.var_name}, {a.var_name}, {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='b', strict=False)
                        stack.append(StackItem(var_name=res_var, role='mixed',
                                            shape=(a.shape[0], b.shape[1], b.shape[2], a.shape[1]),
                                            is_vector=False, is_gradient=True,
                                            field_names=field_names, parent_name=parent_name,
                                            side=side, field_sides=field_sides))

                    # -----------------------------------------------------------------
                    # 1. RHS load:   scalar / vector Function  *  scalar Test
                    #                (u_k or c)                ·  φ_v
                    # -----------------------------------------------------------------
                    elif (b.role == "test" and not b.is_vector
                        and a.role == "value" and not a.is_vector
                        and not a.is_gradient and not b.is_gradient
                        and not a.is_hessian and not b.is_hessian
                        ):
                        body_lines.append("# Load: scalar Function × scalar Test → (n_loc,)")

                        body_lines.append(f"{res_var} = {a.var_name} * {b.var_name}")   # (n_loc,)
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer=b, strict=False)
                        stack.append(StackItem(var_name=res_var, role='value',
                                            shape=(b.shape[1],), is_vector=False,
                                            field_names=field_names, parent_name=parent_name, side=side,
                                            field_sides=field_sides))

                    # symmetric orientation
                    elif (a.role == "test" and not a.is_vector
                        and b.role == "value" and not b.is_vector
                        and not a.is_gradient and not b.is_gradient
                        and not a.is_hessian and not b.is_hessian
                        ):
                        body_lines.append("# Load: scalar Test × scalar Function → (n_loc,)")

                        body_lines.append(f"{res_var} = {b.var_name} * {a.var_name}")   # (n_loc,)
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer=a, strict=False)
                        stack.append(StackItem(var_name=res_var, role='value',
                                            shape=(a.shape[1],), is_vector=False,
                                            field_names=field_names, parent_name=parent_name, side=side,
                                            field_sides=field_sides))
                    # -----------------------------------------------------------------
                    # 1. RHS p * I:   trace(test)  * identity
                    #                (u_test)                ·  φ_v
                    # -----------------------------------------------------------------
                    elif (a.role in {"test", "trial"} and b.role in {"const", "value"} and b.is_gradient
                          and not a.is_vector and not a.is_gradient and not a.is_hessian
                          and len(a.shape) in (1, 2) and (len(a.shape) == 1 or a.shape[0] == 1)
                          and b.shape == (2, 2)):
                        role = 'test' if a.role == 'test' else 'trial'
                        body_lines.append("# rhs: trace(test) * Identity → ")
                        body_lines.append(
                            f"{res_var} = trace_times_identity({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer=a, strict=False)
                        stack.append(StackItem(var_name=res_var, role=role,
                                            shape=(b.shape[0], -1, b.shape[1]),
                                            is_vector =False,
                                            is_gradient=True, is_hessian=False,
                                            field_names=field_names, parent_name=parent_name, side=side,
                                            field_sides=field_sides))
                    elif (a.role in {"const", "value"} and b.role == "mixed" and b.is_gradient
                          and a.shape == (2, 2)):
                        role = 'mixed'
                        body_lines.append("# rhs: Identity × trace(mixed) → ")
                        body_lines.append(
                            f"{res_var} = identity_times_trace_matrix({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer=b, strict=False)
                        stack.append(StackItem(var_name=res_var, role=role,
                                            shape=(a.shape[0], -1, -1, a.shape[1]),
                                            is_vector=False,
                                            is_gradient=True, is_hessian=False, field_names=field_names,
                                            parent_name=parent_name, side=side, field_sides=field_sides))

                    # -----------------------------------------------------------------
                    # 4. Anything else is ***not implemented yet*** – fail fast
                    # -----------------------------------------------------------------
                    else:
                        raise NotImplementedError(
                            f"Product not implemented for roles {a.role}/{b.role} "
                            f"with vector flags {a.is_vector}/{b.is_vector} "
                            f"and gradient flags {a.is_gradient}/{b.is_gradient} "
                            f"and hessian flags {a.is_hessian}/{b.is_hessian}. "
                            f"Shapes: {a.shape}/{b.shape}; field_names={a.field_names}/{b.field_names}; "
                            f"parents={a.parent_name}/{b.parent_name}"
                        )
                 # ---------------------------------------------------------------------------
                 # Binary “+ / −” (element-wise) --------------------------------------------
                 # ---------------------------------------------------------------------------
 
                 elif op.op_symbol in ('+', '-'):
                     sym = op.op_symbol
                     body_lines.append(f"# {'Addition' if sym == '+' else 'Subtraction'}")

                     def _merge_role(ra, rb):
                         if 'mixed' in (ra, rb):
                             return 'mixed'
                         if 'trial' in (ra, rb):
                             return 'trial'
                         if 'test' in (ra, rb):
                             return 'test'
                         if 'value' in (ra, rb):
                             return 'value'
                         return 'const'

                     def _broadcast_shape_with_minus1(sa, sb):
                         from itertools import zip_longest
                         la, lb = len(sa), len(sb)
                         max_len = max(la, lb)
                         ra = (1,) * (max_len - la) + sa
                         rb = (1,) * (max_len - lb) + sb
                         out = []
                         for da, db in zip_longest(ra, rb):
                             if da == db:
                                 out.append(da)
                             elif da == 1:
                                 out.append(db)
                             elif db == 1:
                                 out.append(da)
                             elif da == -1 or db == -1:
                                 out.append(-1)
                             else:
                                 raise NotImplementedError(
                                     f"'{sym}' cannot broadcast shapes {sa} and {sb}"
                                 )
                         return tuple(out)

                     dim_a = len(a.shape)
                     dim_b = len(b.shape)
                     
                     if sym == '+':
                         if dim_a == 3 and dim_b == 4:
                             helper = 'binary_add_3_4'
                         elif dim_a == 4 and dim_b == 3:
                             helper = 'binary_add_4_3'
                         else:
                             helper = 'binary_add_generic'
                     else: # sym == '-'
                         if dim_a == 3 and dim_b == 4:
                             helper = 'binary_sub_3_4'
                         elif dim_a == 4 and dim_b == 3:
                             helper = 'binary_sub_4_3'
                         else:
                             helper = 'binary_sub_generic'
                     try:
                         new_shape = np.broadcast_shapes(a.shape, b.shape)
                     except ValueError:
                         new_shape = _broadcast_shape_with_minus1(a.shape, b.shape)

                     body_lines.append(
                         f"{res_var} = {helper}({a.var_name}, {b.var_name}, {self.dtype})"
                     )
                     field_names, parent_name, side, field_sides = StackItem.resolve_metadata(
                         a, b, prefer='basis', strict=False
                     )
                     stack.append(StackItem(
                         var_name    = res_var,
                         role        = _merge_role(a.role, b.role),
                         shape       = new_shape,
                         is_vector   = a.is_vector or b.is_vector,
                         is_gradient = a.is_gradient or b.is_gradient,
                         is_hessian  = a.is_hessian or b.is_hessian,
                         parent_name = parent_name,
                         field_names = field_names,
                         side        = side,
                         field_sides = field_sides
                     ))
                     continue
                 elif op.op_symbol == '/':
                    body_lines.append("# Division")
                    # divide *anything* by a scalar constant (const in denominator)
                    if (b.role == 'const' or b.role == 'value') and not b.is_vector and b.shape == ():
                        body_lines.append(f"{res_var} = {a.var_name} / float({b.var_name})")
                        stack.append(StackItem(var_name=res_var, role=a.role,
                                            shape=a.shape, is_vector=a.is_vector,
                                            is_gradient=a.is_gradient, is_hessian=a.is_hessian,
                                            field_names=a.field_names,
                                            parent_name=a.parent_name,
                                            side=a.side,
                                            field_sides=a.field_sides or []))
                    elif (a.role == 'const' or a.role == 'value') and not a.is_vector and a.shape == ():
                        body_lines.append(f"{res_var} = float({a.var_name}) / {b.var_name}")
                        stack.append(StackItem(var_name=res_var, role=b.role,
                                            shape=b.shape, is_vector=b.is_vector,
                                            is_gradient=b.is_gradient, is_hessian=b.is_hessian,
                                            field_names=b.field_names,
                                            parent_name=b.parent_name,
                                            side=b.side,
                                            field_sides=b.field_sides or []))
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
                                            is_gradient=a.is_gradient, is_hessian=a.is_hessian,
                                            field_names=a.field_names,
                                            parent_name=a.parent_name,
                                            side=a.side,
                                            field_sides=a.field_sides or [])
                    )
                    elif (a.role == 'const' or a.role == 'value') and not a.is_vector and a.shape == ():
                        body_lines.append(f"{res_var} = float({a.var_name}) ** {b.var_name}")
                        stack.append(StackItem(var_name=res_var, role=b.role,
                                            shape=b.shape, is_vector=b.is_vector,
                                            is_gradient=b.is_gradient, is_hessian=b.is_hessian,
                                            field_names=b.field_names,
                                            parent_name=b.parent_name,
                                            side=b.side,
                                            field_sides=b.field_sides or [])
                    )
                    else:
                        raise NotImplementedError(
                            f"Power not implemented for roles {a.role}/{b.role} "
                            f"with shapes {a.shape}/{b.shape}")

                    

            # --- STORE ---
            elif isinstance(op, Store):
                integrand = stack.pop()
                side = self.last_side_for_store

                if op.store_type == 'matrix':
                    if len(integrand.shape) >= 1 and integrand.shape[0] == 1:
                        body_lines.append(f"Ke += {integrand.var_name}[0] * w_q")
                    else:
                        body_lines.append(f"Ke += {integrand.var_name} * w_q")
                elif op.store_type == 'vector':
                    if len(integrand.shape) > 0 and integrand.shape[0] == 1:
                        body_lines.append(f"Fe += {integrand.var_name}[0] * w_q")
                    else:
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
        # sanitize None from required_args
        # required_args = {arg for arg in required_args if arg is not None} 
        param_order = [
            "gdofs_map",
            "node_coords",
            "qp_phys", "qw", "detJ", "J_inv", "normals", "phis",
            "is_interface",
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
            decorator = "@numba.njit(parallel=True, fastmath=True, cache=True)"
        # New Newton: The kernel signature and loop structure are updated.
        final_kernel_src = f"""
import numba
import numpy as np
from pycutfem.jit.numba_helpers import (
    dot_grad_grad_mixed,
    contract_last_first,
    dot_mixed_const,
    dot_const_mixed,
    dot_vector_trial_grad_test,
    dot_mass_test_trial,
    dot_mass_trial_test,
    dot_grad_func_trial_vec,
    dot_trial_vec_grad_func,
    dot_vec_vec,
    dot_grad_grad_value,
    mul_scalar,
    dot_grad_basis_vector,
    vector_dot_grad_basis,
    dot_grad_basis_with_grad_value,
    dot_grad_value_with_grad_basis,
    basis_dot_const_vector,
    const_vector_dot_basis,
    const_vector_dot_basis_1d,
    scalar_basis_times_vector,
    matrix_times_scalar_basis,
    scalar_vector_outer_product,
    scalar_trial_times_grad_test,
    grad_trial_times_scalar_test,
    scale_mixed_basis_with_coeffs,
    trace_times_identity,
    identity_times_trace_matrix,
    columnwise_dot,
    hessian_dot_vector,
    vector_dot_hessian_basis,
    vector_dot_hessian_value,
    inner_grad_grad,
    inner_hessian_hessian,
    inner_grad_function_grad_test,
    inner_hessian_function_hessian_test,
    inner_mixed_grad_const,
    inner_grad_const_mixed,
    inner_grad_basis_grad_const,
    trace_matrix_value,
    trace_basis_tensor,
    trace_mixed_tensor,
    transpose_grad_tensor,
    transpose_mixed_grad_tensor,
    transpose_hessian_tensor,
    transpose_matrix,
    dot_value_with_grad,
    dot_grad_with_value,
    compute_physical_hessian,
    compute_physical_laplacian,
    load_variable_qp,
    gradient_qp,
    laplacian_qp,
    hessian_qp,
    binary_add_generic,
    binary_add_3_4,
    binary_add_4_3,
    binary_sub_generic,
    binary_sub_3_4,
    binary_sub_4_3,
    scatter_tensor_to_union,
)
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
