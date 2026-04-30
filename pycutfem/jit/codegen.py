# pycutfem/jit/codegen.py
from typing import Tuple, List, Any
from dataclasses import dataclass, field, replace
from types import SimpleNamespace


from pycutfem.jit.ir import (
    LoadVariable, LoadConstant, LoadConstantArray, LoadElementWiseConstant, LoadQuadratureState,
    LoadAnalytic, LoadFacetNormal, Grad, PackGradient, Div, HdivDiv, PosOp, NegOp,
    PositivePartOp, HeavisideOp, LogOp, ExpOp, TanhOp, SinOp, CosOp, TanOp, AsinOp, AcosOp, AtanOp,
    SinhOp, CoshOp, AsinhOp, AcoshOp, AtanhOp,
    BinaryOp, Inner, Dot, Outer, Store, Transpose, CellDiameter, LoadFacetNormalComponent, CheckDomain,
    MeshSize,
    Trace, Determinant, Inverse, Cofactor, Hessian as IRHessian, Laplacian as IRLaplacian
)
from pycutfem.jit.symbols import encode
import numpy as np
import re
import os

from pycutfem.ufl.tensor_algebra import (
    DotKernelCase,
    MixedLayout,
    OperandTransform,
    ProductKernelCase,
    TensorRuleEngine,
)


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
    is_divergence: bool = field(default=False)  # True if this scalar came from div(grad(vector))
    layout_tag: str = field(default="")
    expression_meta: Any = field(default=None)
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
    

def _basis_col_dim(shape: tuple) -> int:
    """
    Return the basis column dimension, tolerating both (k, n) and (n,) / (1, n)
    layouts. This makes the mass-matrix dot cases robust to 1D arrays.
    """
    if len(shape) >= 2:
        return int(shape[1])
    if len(shape) == 1:
        return int(shape[0])
    raise ValueError(f"Unexpected basis shape {shape!r} for dot product")


def _shape_dim_merge(lhs_dim: int, rhs_dim: int, planned_dim: int | None = None) -> int:
    """Merge two runtime dimensions, tolerating broadcast axes and -1 wildcards."""
    if lhs_dim == rhs_dim:
        return int(lhs_dim)
    if lhs_dim == 1:
        return int(rhs_dim)
    if rhs_dim == 1:
        return int(lhs_dim)
    if lhs_dim == -1 and rhs_dim != -1:
        return int(rhs_dim)
    if rhs_dim == -1 and lhs_dim != -1:
        return int(lhs_dim)
    if planned_dim is not None:
        return int(planned_dim)
    if lhs_dim == -1 and rhs_dim == -1:
        return -1
    raise ValueError(f"Incompatible runtime dimensions {lhs_dim} and {rhs_dim}")


def _merge_runtime_shapes(lhs_shape: tuple, rhs_shape: tuple, planned_shape: tuple = ()) -> tuple:
    """
    Merge runtime shapes while preferring concrete emitted layouts over symbolic raw shapes.
    """
    from itertools import zip_longest

    if lhs_shape == rhs_shape:
        return lhs_shape
    la, lb = len(lhs_shape), len(rhs_shape)
    lp = len(planned_shape)
    max_len = max(la, lb, lp)
    lhs_pad = (1,) * (max_len - la) + tuple(lhs_shape)
    rhs_pad = (1,) * (max_len - lb) + tuple(rhs_shape)
    plan_pad = (None,) * (max_len - lp) + tuple(planned_shape) if planned_shape else (None,) * max_len
    merged = tuple(
        _shape_dim_merge(int(da), int(db), pd if pd is None else int(pd))
        for da, db, pd in zip_longest(lhs_pad, rhs_pad, plan_pad, fillvalue=None)
    )
    if not merged:
        return ()
    first_non_singleton = 0
    while first_non_singleton < len(merged) - 1 and merged[first_non_singleton] == 1:
        first_non_singleton += 1
    return merged[first_non_singleton:]


def _matches_shape_with_wildcards(shape: tuple, target_shape: tuple) -> bool:
    if len(shape) != len(target_shape):
        return False
    return all(sd == td or sd == -1 or td == -1 for sd, td in zip(shape, target_shape))


def _collapse_leading_singleton_ref(var_name: str, shape: tuple, target_shape: tuple) -> str:
    if len(shape) == len(target_shape) + 1 and shape[0] == 1:
        if _matches_shape_with_wildcards(shape[1:], target_shape):
            return f"{var_name}[0]"
    return var_name


def _is_basis_row_like(item: StackItem) -> bool:
    if item.role not in {"test", "trial"}:
        return False
    if item.is_gradient or item.is_hessian:
        return False
    return len(item.shape) == 1 or (len(item.shape) == 2 and item.shape[0] == 1)


def _try_dot_plan(lhs: Any, rhs: Any):
    try:
        return TensorRuleEngine.plan_dot(lhs, rhs)
    except Exception:
        return None


def _try_inner_plan(lhs: Any, rhs: Any):
    try:
        return TensorRuleEngine.plan_inner(lhs, rhs)
    except Exception:
        return None


def _try_inner_value_spec(lhs: Any, rhs: Any):
    try:
        return TensorRuleEngine.plan_inner_value_spec(lhs, rhs)
    except Exception:
        return None


def _try_dot_lowering(lhs: Any, rhs: Any):
    try:
        return TensorRuleEngine.plan_dot_lowering(lhs, rhs)
    except Exception:
        return None


def _try_dot_kernel(lhs: Any, rhs: Any):
    try:
        return TensorRuleEngine.plan_dot_kernel(lhs, rhs)
    except Exception:
        return None


def _try_sum_plan(lhs: Any, rhs: Any):
    try:
        return TensorRuleEngine.plan_sum(lhs, rhs)
    except Exception:
        return None


def _try_sum_lowering(lhs: Any, rhs: Any):
    try:
        return TensorRuleEngine.plan_sum_lowering(lhs, rhs)
    except Exception:
        return None


def _try_sum_value_spec(lhs: Any, rhs: Any):
    try:
        return TensorRuleEngine.plan_sum_value_spec(lhs, rhs)
    except Exception:
        return None


def _try_trace_meta(item: Any, *, spatial_dim: int = 2):
    try:
        return TensorRuleEngine.plan_trace_meta(item, spatial_dim=spatial_dim)
    except Exception:
        return None


def _try_determinant_meta(item: Any, *, spatial_dim: int = 2):
    try:
        return TensorRuleEngine.plan_determinant_meta(item, spatial_dim=spatial_dim)
    except Exception:
        return None


def _try_transpose_meta(item: Any, *, spatial_dim: int = 2):
    try:
        return TensorRuleEngine.plan_transpose_meta(item, spatial_dim=spatial_dim)
    except Exception:
        return None


def _try_product_lowering(lhs: Any, rhs: Any):
    try:
        return TensorRuleEngine.plan_product_lowering(lhs, rhs)
    except Exception:
        return None


def _try_product_kernel(lhs: Any, rhs: Any):
    try:
        return TensorRuleEngine.plan_product_kernel(lhs, rhs)
    except Exception:
        return None


def _try_dot_value_spec(lhs: Any, rhs: Any):
    try:
        return TensorRuleEngine.plan_dot_value_spec(lhs, rhs)
    except Exception:
        return None


def _try_product_value_spec(lhs: Any, rhs: Any):
    try:
        return TensorRuleEngine.plan_product_value_spec(lhs, rhs)
    except Exception:
        return None


def _try_division_value_spec(lhs: Any, rhs: Any):
    try:
        return TensorRuleEngine.plan_division_value_spec(lhs, rhs)
    except Exception:
        return None


def _try_signature(item: Any, *, spatial_dim: int = 2):
    try:
        return TensorRuleEngine.infer_signature(item, spatial_dim=spatial_dim)
    except Exception:
        return None


def _semantic_is_mixed_rank1(item: Any, *, spatial_dim: int = 2) -> bool:
    sig = _try_signature(item, spatial_dim=spatial_dim)
    return bool(sig is not None and sig.basis_rank == 2 and sig.tensor_rank == 1)


def _semantic_is_value_rank1(item: Any, *, spatial_dim: int = 2) -> bool:
    sig = _try_signature(item, spatial_dim=spatial_dim)
    return bool(sig is not None and sig.basis_rank == 0 and sig.tensor_rank == 1)


def _semantic_is_basis_rank1(item: Any, *, spatial_dim: int = 2) -> bool:
    sig = _try_signature(item, spatial_dim=spatial_dim)
    return bool(sig is not None and sig.basis_rank == 1 and sig.tensor_rank == 1)


def _semantic_is_scalar_basis(item: Any, *, spatial_dim: int = 2) -> bool:
    sig = _try_signature(item, spatial_dim=spatial_dim)
    return bool(sig is not None and sig.basis_rank == 1 and sig.tensor_rank == 0)


def _semantic_is_value_rank2(item: Any, *, spatial_dim: int = 2) -> bool:
    sig = _try_signature(item, spatial_dim=spatial_dim)
    return bool(sig is not None and sig.basis_rank == 0 and sig.tensor_rank == 2)


def _semantic_is_basis_rank2(item: Any, *, spatial_dim: int = 2) -> bool:
    sig = _try_signature(item, spatial_dim=spatial_dim)
    return bool(sig is not None and sig.basis_rank == 1 and sig.tensor_rank == 2)


def _mixed_rank1_layout(item: Any, *, spatial_dim: int = 2) -> str:
    layout = str(getattr(item, "layout_tag", "") or "")
    if layout:
        return layout
    sig = _try_signature(item, spatial_dim=spatial_dim)
    if sig is None or sig.basis_rank != 2 or sig.tensor_rank != 1:
        return ""
    shape = tuple(int(v) for v in getattr(item, "shape", ()) or ())
    if len(shape) == 3 and 0 < int(shape[0]) <= spatial_dim:
        return MixedLayout.COMPONENT_FIRST.value
    return MixedLayout.DEFAULT.value


def _try_tensor_rank(item: Any, *, spatial_dim: int = 2) -> int | None:
    try:
        return int(TensorRuleEngine.infer_signature(item, spatial_dim=spatial_dim).tensor_rank)
    except Exception:
        return None


def _apply_sum_operand_transform(expr: str, shape: tuple, transform: OperandTransform, dtype_expr: str) -> tuple[str, tuple]:
    if transform == OperandTransform.TRANSPOSE_2D:
        if len(shape) != 2:
            raise ValueError(f"transpose_2d sum transform expects rank-2 operand, got shape {shape!r}")
        return f"transpose_matrix({expr}, {dtype_expr})", (shape[1], shape[0])
    if transform == OperandTransform.SCALAR_GRAD_TO_VECTOR:
        if len(shape) != 3 or shape[0] != 1:
            raise ValueError(f"scalar_grad_to_vector sum transform expects shape (1,n,d), got {shape!r}")
        return f"transpose_matrix({expr}[0], {dtype_expr})", (shape[2], shape[1])
    return expr, shape


def _basis_result_shape_from_dot(dot_lowering: Any, basis_item: "StackItem", fallback_shape: tuple) -> tuple[tuple, bool, bool]:
    """
    Derive the basis-carrying runtime shape from the shared dot planner.

    This is the key guard against the old scalar-gradient leak:
    `dot(grad(a_trial), v)` must lower to a scalar basis row `(1,n)`, not a
    fake carried-vector basis `(1,n)` with `is_vector=True`.
    """
    if dot_lowering is None or basis_item.role not in {"trial", "test"}:
        return fallback_shape, False, False
    tensor = dot_lowering.algebra.result
    if tensor.basis_rank != 1:
        return fallback_shape, dot_lowering.result.is_vector, dot_lowering.result.is_gradient
    planned_shape = getattr(dot_lowering, "result_storage", None)
    if planned_shape is not None:
        return (
            tuple(int(v) for v in dot_lowering.result_storage.stored_shape),
            dot_lowering.result.is_vector,
            dot_lowering.result.is_gradient,
        )
    n_basis = _basis_col_dim(basis_item.shape)
    if tensor.tensor_rank == 0:
        return (1, n_basis), False, False
    if tensor.tensor_rank == 1:
        return (
            int(tensor.free_axes[0].size),
            n_basis,
        ), dot_lowering.result.is_vector, dot_lowering.result.is_gradient
    return fallback_shape, dot_lowering.result.is_vector, dot_lowering.result.is_gradient


def _is_scalar_grad_basis_shape(shape: tuple, spatial_dim: int) -> bool:
    return (
        (len(shape) == 3 and int(shape[0]) == 1 and int(shape[2]) == spatial_dim)
        or (len(shape) == 2 and int(shape[0]) == spatial_dim)
    )


def _scalar_grad_basis_ncols(shape: tuple) -> int:
    if len(shape) == 3:
        return int(shape[1])
    if len(shape) == 2:
        return int(shape[1])
    raise ValueError(f"Not a scalar gradient basis shape: {shape!r}")


def _planned_storage_shape(lowering: Any, fallback_shape: tuple) -> tuple:
    if lowering is None:
        return fallback_shape
    planned = tuple(int(v) for v in lowering.result_storage.stored_shape)
    return planned or fallback_shape


def _dot_result_flags_and_layout(lowering: Any, fallback_shape: tuple) -> tuple[tuple, bool, bool, bool, str]:
    shape = _planned_storage_shape(lowering, fallback_shape)
    if lowering is None:
        return shape, False, False, False, ""
    return (
        shape,
        bool(lowering.result.is_vector),
        bool(lowering.result.is_gradient),
        bool(lowering.result.is_hessian),
        lowering.result.layout.value,
    )


def _dot_result_stack_kwargs(dot_lowering: Any, dot_value_spec: Any, fallback_shape: tuple, default_role: str) -> dict:
    if dot_value_spec is not None:
        return {
            "role": dot_value_spec.role,
            "shape": tuple(int(v) for v in dot_value_spec.shape),
            "is_vector": bool(dot_value_spec.is_vector),
            "is_gradient": bool(dot_value_spec.is_gradient),
            "is_hessian": bool(dot_value_spec.is_hessian),
            "layout_tag": dot_value_spec.layout.value,
        }
    shape, is_vector, is_gradient, is_hessian, layout_tag = _dot_result_flags_and_layout(
        dot_lowering,
        fallback_shape,
    )
    return {
        "role": default_role,
        "shape": shape,
        "is_vector": is_vector,
        "is_gradient": is_gradient,
        "is_hessian": is_hessian,
        "layout_tag": layout_tag,
    }


def _push_product_result(
    stack: list["StackItem"],
    *,
    res_var: str,
    default_role: str,
    fallback_shape: tuple,
    field_names: list[str],
    parent_name: str,
    side: str,
    field_sides: list[str],
    product_lowering: Any = None,
    product_value_spec: Any = None,
    is_vector: bool = False,
    is_gradient: bool = False,
    is_hessian: bool = False,
    layout_tag: str = "",
    expression_meta: Any = None,
) -> None:
    def _planned_wraps_runtime(runtime_shape: tuple, candidate_shape: tuple) -> bool:
        runtime = tuple(int(v) for v in runtime_shape)
        candidate = tuple(int(v) for v in candidate_shape)
        if not runtime or len(candidate) <= len(runtime):
            return False
        lead = len(candidate) - len(runtime)
        return all(int(v) == 1 for v in candidate[:lead]) and candidate[lead:] == runtime

    role = default_role
    shape = tuple(int(v) for v in fallback_shape)
    vec = bool(is_vector)
    grad = bool(is_gradient)
    hess = bool(is_hessian)
    layout = str(layout_tag or "")
    meta = expression_meta

    if product_lowering is not None:
        planned_shape = tuple(int(v) for v in product_lowering.result_storage.stored_shape)
        if planned_shape and not _planned_wraps_runtime(shape, planned_shape):
            shape = planned_shape
        role = product_lowering.result.role or role
        vec = bool(product_lowering.result.is_vector)
        grad = bool(product_lowering.result.is_gradient)
        hess = bool(product_lowering.result.is_hessian)
        layout = product_lowering.result.layout.value
        if meta is None:
            meta = product_lowering.meta

    if product_value_spec is not None:
        planned_shape = tuple(int(v) for v in product_value_spec.shape)
        if planned_shape and not _planned_wraps_runtime(shape, planned_shape):
            shape = planned_shape
        role = product_value_spec.role or role
        vec = bool(product_value_spec.is_vector)
        grad = bool(product_value_spec.is_gradient)
        hess = bool(product_value_spec.is_hessian)
        layout = product_value_spec.layout.value
        meta = product_value_spec.meta if meta is None else meta

    stack.append(
        StackItem(
            var_name=res_var,
            role=role,
            shape=shape,
            is_vector=vec,
            is_gradient=grad,
            is_hessian=hess,
            field_names=field_names,
            parent_name=parent_name,
            side=side,
            field_sides=field_sides,
            layout_tag=layout,
            expression_meta=meta,
        )
    )


def _basis_dot_result_stack_kwargs(
    dot_lowering: Any,
    dot_value_spec: Any,
    basis_item: "StackItem",
    fallback_shape: tuple,
    default_role: str | None = None,
) -> dict:
    if dot_value_spec is not None:
        role = dot_value_spec.role if dot_value_spec.role not in {"", "none"} else (default_role or basis_item.role)
        return {
            "role": role,
            "shape": tuple(int(v) for v in dot_value_spec.shape),
            "is_vector": bool(dot_value_spec.is_vector),
            "is_gradient": bool(dot_value_spec.is_gradient),
            "is_hessian": bool(dot_value_spec.is_hessian),
            "layout_tag": dot_value_spec.layout.value,
        }
    shape, is_vector, is_gradient = _basis_result_shape_from_dot(dot_lowering, basis_item, fallback_shape)
    return {
        "role": default_role or basis_item.role,
        "shape": shape,
        "is_vector": is_vector,
        "is_gradient": is_gradient,
        "is_hessian": bool(getattr(getattr(dot_lowering, "result", None), "is_hessian", False)),
        "layout_tag": (
            dot_lowering.result.layout.value
            if dot_lowering is not None
            else ""
        ),
    }


def _rank1_basis_free_last_expr(var_name: str, shape: tuple) -> str:
    """
    Reorder rank-1 basis storage so the free physical axis is last.

    Stored scalar-gradient basis layouts are canonicalized as ``(d, n)`` in the
    refactored algebra layer, but ``contract_last_first`` expects the free axis
    to be the trailing dimension on the left operand. This helper produces that
    storage-only view without implying any semantic transpose of the rank-1
    tensor itself.
    """
    if len(shape) == 3 and int(shape[0]) == 1:
        return f"np.ascontiguousarray({var_name}[0])"
    if len(shape) == 2:
        return f"np.ascontiguousarray({var_name}.T)"
    raise ValueError(f"Expected canonical scalar-gradient basis shape, got {shape!r}.")


def _rank1_value_expr(var_name: str, shape: tuple) -> str:
    """Return a semantic rank-1 value view, collapsing legacy leading-singleton wrappers."""
    if len(shape) == 2 and (int(shape[0]) == 1 or int(shape[1]) == 1):
        return f"np.ascontiguousarray({var_name}).reshape({int(shape[0]) * int(shape[1])})"
    if len(shape) == 1:
        return var_name
    raise ValueError(f"Expected rank-1 value storage, got {shape!r}.")


def _scalar_basis_values_expr(var_name: str, shape: tuple) -> str:
    """Return a 1D scalar basis view for either (1, n) or (n,) storage."""
    if len(shape) == 2 and int(shape[0]) == 1:
        return f"{var_name}[0]"
    if len(shape) == 1:
        return var_name
    raise ValueError(f"Expected scalar basis storage, got {shape!r}.")


def _dot_value_operand_expr(var_name: str, shape: tuple, tensor_rank: int | None) -> str:
    """
    Return the storage view that matches the semantic tensor rank used by dot().

    Rank-1 values keep vector semantics even when carried through legacy
    leading-singleton wrappers such as ``(1, d)``.
    """
    if tensor_rank == 1:
        return _rank1_value_expr(var_name, shape)
    return var_name


def _emit_dot_storage_reorder(body_lines: list[str], var_name: str, raw_shape: tuple, planned_shape: tuple, dtype_expr: str) -> tuple:
    raw_shape = tuple(int(v) for v in raw_shape)
    planned_shape = tuple(int(v) for v in planned_shape)
    if not planned_shape or raw_shape == planned_shape:
        return raw_shape
    if len(raw_shape) == 2 and raw_shape[0] == 1 and len(planned_shape) == 1 and planned_shape == (raw_shape[1],):
        body_lines.append(f"{var_name} = np.ascontiguousarray({var_name}[0])")
        return planned_shape
    if len(raw_shape) == len(planned_shape) == 2 and planned_shape == (raw_shape[1], raw_shape[0]):
        body_lines.append(f"{var_name} = transpose_matrix({var_name}, {dtype_expr})")
        return planned_shape
    if len(raw_shape) == 3 and raw_shape[0] == 1 and len(planned_shape) == 2 and planned_shape == (raw_shape[2], raw_shape[1]):
        body_lines.append(f"{var_name} = transpose_matrix({var_name}[0], {dtype_expr})")
        return planned_shape
    if len(raw_shape) == len(planned_shape) == 3 and planned_shape == (raw_shape[1], raw_shape[0], raw_shape[2]):
        body_lines.append(f"{var_name} = swap_mixed_basis_tensor({var_name}, {dtype_expr})")
        return planned_shape
    if (
        len(raw_shape) == len(planned_shape)
        and len(raw_shape) >= 4
        and planned_shape == (raw_shape[0], raw_shape[2], raw_shape[1]) + raw_shape[3:]
    ):
        body_lines.append(f"{var_name} = swap_mixed_basis_tensor({var_name}, {dtype_expr})")
        return planned_shape
    return raw_shape


# ── scalar * Vector/Tensor  (or Vector/Tensor * scalar) ────────────
def _mul_scalar_vector(self,first_is_scalar,res_var,a, b, body_lines, stack):
    scalar = a if first_is_scalar else b
    vect   = b if first_is_scalar else a
    product_lowering = _try_product_lowering(a, b)
    product_value_spec = _try_product_value_spec(a, b)
    vect_is_scalar_basis = _semantic_is_scalar_basis(vect, spatial_dim=self.spatial_dim)

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

    role = vect.role
    is_vector = vect.is_vector
    is_gradient = vect.is_gradient
    is_hessian = vect.is_hessian
    layout_tag = getattr(vect, "layout_tag", "")
    expression_meta = (
        product_value_spec.meta
        if product_value_spec is not None
        else (product_lowering.meta if product_lowering is not None else getattr(vect, "expression_meta", None))
    )
    if product_value_spec is not None:
        planned_shape = tuple(int(v) for v in product_value_spec.shape)
        if planned_shape and not (
            collapse
            and len(planned_shape) == 2
            and planned_shape[0] == 1
            and len(res_shape) == 1
        ) and not (
            vect_is_scalar_basis
            and len(vect.shape) == 1
            and len(planned_shape) == 2
            and planned_shape[0] == 1
        ):
            res_shape = planned_shape
        role = product_value_spec.role or role
        is_vector = product_value_spec.is_vector
        is_gradient = product_value_spec.is_gradient
        is_hessian = product_value_spec.is_hessian
        layout_tag = product_value_spec.layout.value
    elif product_lowering is not None:
        planned_shape = tuple(int(v) for v in product_lowering.result_storage.stored_shape)
        if planned_shape and not (
            collapse
            and len(planned_shape) == 2
            and planned_shape[0] == 1
            and len(res_shape) == 1
        ):
            res_shape = planned_shape
        role = product_lowering.result.role or role
        is_vector = product_lowering.result.is_vector
        is_gradient = product_lowering.result.is_gradient
        is_hessian = product_lowering.result.is_hessian
        layout_tag = product_lowering.result.layout.value

    stack.append(StackItem(var_name   = res_var,
                           role       = role,
                           shape      = res_shape,
                           is_vector  = is_vector,
                           is_gradient= is_gradient,
                           is_hessian = is_hessian,
                           field_names= vect.field_names,
                           parent_name= vect.parent_name,
                           side       = vect.side,
                           field_sides=vect.field_sides or [],
                           layout_tag=layout_tag,
                           expression_meta=expression_meta))

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
        self.active_fields: tuple[str, ...] | None = None
        self.active_slices: dict[str, slice] = {}
        self.active_n_dofs: int = self.n_dofs_local
        self.spatial_dim = self.me.mesh.spatial_dim
        self.last_side_for_store = "" # Track side for detJ
        self.dtype = "np.float64"  # Default data type for arrays
        self.on_facet = on_facet

    def _field_slice(self, field: str) -> slice:
        """Return the contiguous slice for an active field."""
        if field in self.active_slices:
            return self.active_slices[field]
        return self.me.component_dof_slices[field]

    def _field_nloc(self, field: str) -> int:
        sl = self._field_slice(field)
        return int(sl.stop - sl.start)

    def _init_active_fields(self, ir_sequence):
        if self.active_fields is not None:
            return
        seen = set(); order = []
        for op in ir_sequence:
            fns = getattr(op, "field_names", None)
            if fns:
                for f in fns or []:
                    if f in seen or f not in getattr(self.me, "field_names", ()):
                        continue
                    seen.add(f)
                    order.append(f)
                continue
            f = getattr(op, "field_name", None)
            if f and f not in seen and f in getattr(self.me, "field_names", ()):
                seen.add(f)
                order.append(f)
        if not order:
            order = list(self.me.field_names)
        else:
            me_order = list(self.me.field_names)
            order = [f for f in me_order if f in order]
            if not order:
                order = list(self.me.field_names)
        self.active_fields = tuple(order)
        start = 0
        for f in self.active_fields:
            nloc = self._field_nloc(f)
            self.active_slices[f] = slice(start, start + nloc)
            start += nloc
        self.active_n_dofs = start

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
          1) explicit side from Jump/Pos/Neg ('+' → 'pos', '-' → 'neg')
          2) per-component hint field_sides[idx] if present ('pos'/'neg')
          3) infer from name ('*_pos_*' / '*_neg_*')
        Returns 'pos'/'neg' or None if nothing can be determined.
        """
        if default_side in ("+","-"):
            return "pos" if default_side == "+" else "neg"
        if field_sides and 0 <= idx < len(field_sides) and field_sides[idx] in ("pos","neg"):
            return field_sides[idx]
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
        self._init_active_fields(ir_sequence)
        if not self.on_facet:
            def _strip_side(op):
                if isinstance(op, LoadVariable):
                    if op.side or getattr(op, "field_sides", None):
                        return replace(op, side="", field_sides=None)
                    return op
                if isinstance(op, CheckDomain) and op.side:
                    return replace(op, side="")
                return op
            if any(
                (isinstance(op, LoadVariable) and (op.side or getattr(op, "field_sides", None)))
                or (isinstance(op, CheckDomain) and op.side)
                for op in ir_sequence
            ):
                ir_sequence = [_strip_side(op) for op in ir_sequence]

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

            elif isinstance(op, PackGradient):
                dy = stack.pop()
                dx = stack.pop()
                if dx.shape != dy.shape:
                    raise NotImplementedError(
                        f"PackGradient requires matching component shapes, got {dx.shape} and {dy.shape}."
                    )
                field_names, parent_name, side, field_sides = StackItem.resolve_metadata(
                    dx, dy, prefer="basis", strict=False
                )
                if dx.role in {"test", "trial"} or dy.role in {"test", "trial"}:
                    role = dx.role if dx.role in {"test", "trial"} else dy.role
                    var_name = new_var("grad_pack")
                    if len(dx.shape) == 2:
                        if dx.shape[0] == 1:
                            body_lines.append(
                                f"{var_name} = np.ascontiguousarray(np.stack(({dx.var_name}[0], {dy.var_name}[0]), axis=0))"
                            )
                            shape = (self.spatial_dim, dx.shape[1])
                        else:
                            body_lines.append(f"{var_name} = np.stack(({dx.var_name}, {dy.var_name}), axis=-1)")
                            shape = (dx.shape[0], dx.shape[1], self.spatial_dim)
                    elif len(dx.shape) == 1:
                        body_lines.append(
                            f"{var_name} = np.ascontiguousarray(np.stack(({dx.var_name}, {dy.var_name}), axis=0))"
                        )
                        shape = (self.spatial_dim, dx.shape[0])
                    else:
                        raise NotImplementedError(
                            f"PackGradient does not support basis component rank {len(dx.shape)}."
                        )
                    stack.append(
                        StackItem(
                            var_name=var_name,
                            role=role,
                            shape=shape,
                            is_vector=False,
                            is_gradient=True,
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                        )
                    )
                    continue

                role = "value" if (dx.role == "value" or dy.role == "value") else "const"
                var_name = new_var("grad_pack")
                if dx.shape == ():
                    body_lines.append(f"{var_name} = np.array([{dx.var_name}, {dy.var_name}], dtype={self.dtype})")
                    shape = (self.spatial_dim,)
                    is_vector = True
                    is_gradient = False
                elif len(dx.shape) == 2 and dx.shape == (1, 1):
                    body_lines.append(
                        f"{var_name} = np.array([{dx.var_name}[0, 0], {dy.var_name}[0, 0]], dtype={self.dtype})"
                    )
                    shape = (self.spatial_dim,)
                    is_vector = True
                    is_gradient = False
                else:
                    raise NotImplementedError(
                        f"PackGradient does not support value component shape {dx.shape}."
                    )
                stack.append(
                    StackItem(
                        var_name=var_name,
                        role=role,
                        shape=shape,
                        is_vector=is_vector,
                        is_gradient=is_gradient,
                        field_names=field_names,
                        parent_name=parent_name,
                        side=side,
                        field_sides=field_sides,
                    )
                )
                continue
            
            elif isinstance(op, CheckDomain):
                a = stack.pop()
                out = new_var("restricted")

                side = op.side or a.side
                if side == "+":
                    flag_arr = f"domain_flag_{op.bitset_id}_pos"
                elif side == "-":
                    flag_arr = f"domain_flag_{op.bitset_id}_neg"
                else:
                    flag_arr = f"domain_flag_{op.bitset_id}"
                required_args.add(flag_arr)

                body_lines.append(f"# Restriction via {flag_arr}")
                if a.shape == () or a.shape == tuple():
                    zero_expr = "0.0"
                else:
                    zero_expr = f"np.zeros_like({a.var_name}, dtype={self.dtype})"
                body_lines.append(f"{out} = {a.var_name} if {flag_arr}[e, q] else {zero_expr}")

                # Apply per-field restriction masks on facet integrals (ghost/interface)
                if (
                    self.on_facet
                    and side in ("+", "-")
                    and a.field_names
                    and any(dim == -1 for dim in a.shape)
                ):
                    # Some vector-valued fields (e.g. H(div) RT) use a single
                    # field name for multiple value components. Apply the same
                    # restriction mask to every component row.
                    try:
                        n_comp = int(a.shape[0]) if (len(a.shape) >= 1 and int(a.shape[0]) > 0) else len(a.field_names)
                    except Exception:
                        n_comp = len(a.field_names)

                    for i in range(n_comp):
                        fld = a.field_names[i] if i < len(a.field_names) else a.field_names[0]
                        side_tag = self._component_side_tag(
                            side, getattr(a, "field_sides", None), fld, i
                        )
                        if side_tag not in ("pos", "neg"):
                            side_tag = "pos" if side == "+" else "neg"
                        mask_name = f"restrict_mask_{fld}_{side_tag}"
                        required_args.add(mask_name)
                        if len(a.shape) == 1:
                            body_lines.append(f"for j in range(n_union): {out}[j] *= {mask_name}[e, j]")
                            break
                        if len(a.shape) == 2:
                            body_lines.append(f"for j in range(n_union): {out}[{i}, j] *= {mask_name}[e, j]")
                        elif len(a.shape) == 3:
                            body_lines.append(f"for j in range(n_union): {out}[{i}, j, :] *= {mask_name}[e, j]")
                        elif len(a.shape) == 4:
                            body_lines.append(f"for j in range(n_union): {out}[{i}, j, :, :] *= {mask_name}[e, j]")

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
                        f"{out} = np.empty(({k_comps}, n_union, 2, 2), dtype={self.dtype})",
                    ]
                    for i, fn in enumerate(a.field_names):
                        s0 = self._field_slice(fn).start; s1 = self._field_slice(fn).stop
                        Hloc = new_var("Hloc")
                        body_lines += [
                            f"A  = {jinv}[e, q]",
                            f"Hx = {H0}[e, q]", f"Hy = {H1}[e, q]",
                            f"d10_q = {d10[i]}[e, q]", f"d01_q = {d01[i]}[e, q]",
                            f"d20_q = {d20[i]}[e, q]", f"d11_q = {d11[i]}[e, q]", f"d02_q = {d02[i]}[e, q]",
                            f"{Hloc} = compute_physical_hessian(d20_q, d11_q, d02_q, d10_q, d01_q, A, Hx, Hy, {self.dtype})",
                        ]
                        if a.side:  # pad to union using map (unless already union-sized)
                            side_tag = self._component_side_tag(a.side, a.field_sides, fn, i)
                            map_arr = f"{side_tag}_map_{fn}"
                            required_args.add(map_arr)
                            Hpad, me = new_var("Hpad"), new_var("map_e")
                            Hsub = new_var("Hsub")
                            body_lines += [
                                f"if {Hloc}.shape[0] == n_union:",
                                f"    {out}[{i}] = {Hloc}",
                                f"else:",
                                f"    {me} = {map_arr}[e]",
                                # slice down to this field first
                                f"    {Hsub} = {Hloc}[{self._field_slice(fn).start}:{self._field_slice(fn).stop}, :, :]",
                                f"    {Hpad} = scatter_tensor_to_union({Hsub}, {me}, n_union, {self.dtype})",
                                f"    {out}[{i}] = {Hpad}",
                            ]
                        else:
                            # volume: typically nloc == n_union; assign directly
                            body_lines += [f"{out}[{i}] = {Hloc}"]
                    hess_basis_meta = TensorRuleEngine.infer_expression_meta(
                        SimpleNamespace(
                            role=a.role,
                            kind="hess",
                            shape=(k_comps, self.active_n_dofs, 2, 2),
                            field_names=a.field_names,
                            parent_name=a.parent_name,
                            side=a.side,
                            field_sides=a.field_sides or [],
                        ),
                        spatial_dim=self.spatial_dim,
                    )
                    stack.append(StackItem(var_name=out, role=a.role,
                                        shape=(k_comps, self.active_n_dofs, 2, 2),
                                        is_vector=False, is_hessian=True,
                                        field_names=a.field_names, 
                                        parent_name=a.parent_name, side=a.side,
                                        field_sides=a.field_sides or [],
                                        expression_meta=hess_basis_meta))


                elif a.role == "value":
                    k_comps = len(a.field_names)
                    coeff = (a.parent_name if a.parent_name.startswith("u_")
                            else f"u_{a.parent_name}_loc")
                    required_args.add(coeff)

                    body_lines += [f"{out} = np.zeros(({k_comps}, 2, 2), dtype={self.dtype})"]
                    for i, fn in enumerate(a.field_names):
                        s0 = self._field_slice(fn).start; s1 = self._field_slice(fn).stop
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
                                f"if {Hloc}.shape[0] == {coeff}.shape[0]:",
                                f"    {out}[{i}] = hessian_qp({coeff}, {Hloc})",
                                f"else:",
                                f"    {me} = {map_arr}[e]",
                                f"    {Hsub} = {Hloc}[{s0}:{s1}, :, :]",
                                f"    {Hpad} = scatter_tensor_to_union({Hsub}, {me}, n_union, {self.dtype})",
                                f"    {out}[{i}] = hessian_qp({coeff}, {Hpad})",
                            ]
                        else:
                            # --- tensordot-free collapse: (n,2,2) -> (2,2)
                            body_lines += [
                                f"{out}[{i}] = hessian_qp({coeff}, {Hloc})",
                            ]

                    hess_value_meta = TensorRuleEngine.infer_expression_meta(
                        SimpleNamespace(
                            role="value",
                            kind="hess",
                            shape=(k_comps, 2, 2),
                            field_names=a.field_names,
                            parent_name=a.parent_name,
                            side=a.side,
                            field_sides=a.field_sides or [],
                        ),
                        spatial_dim=self.spatial_dim,
                    )
                    stack.append(StackItem(var_name=out, role="value",
                                        shape=(k_comps, 2, 2),
                                        is_vector=False, is_hessian=True,
                                        is_gradient=False,
                                        field_names=a.field_names, 
                                        parent_name=a.parent_name, side=a.side,
                                        field_sides=a.field_sides or [],
                                        expression_meta=hess_value_meta))



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
                        f"{out} = np.empty(({k_comps}, n_union), dtype={self.dtype})",
                    ]
                    for i, fn in enumerate(a.field_names):
                        s0 = self._field_slice(fn).start; s1 = self._field_slice(fn).stop
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
                                f"if {laploc}.shape[0] == n_union:",
                                f"    {out}[{i}] = {laploc}",
                                f"else:",
                                f"    {me} = {map_arr}[e]",
                                f"    {lap_sub} = {laploc}[{s0}:{s1}]",
                                f"    {lap_pad} = scatter_tensor_to_union({lap_sub}, {me}, n_union, {self.dtype})",
                                f"    {out}[{i}] = {lap_pad}",
                            ]
                        else:
                            body_lines += [f"{out}[{i}] = {laploc}"]

                    lap_basis_is_vector = bool(a.is_vector)
                    stack.append(StackItem(var_name=out, role=a.role,
                                        shape=(k_comps, self.active_n_dofs),
                                        is_vector=lap_basis_is_vector, field_names=a.field_names, 
                                        parent_name=a.parent_name, side=a.side,
                                        field_sides=a.field_sides or []))


                # ---------------- VALUE: collapse with coeffs → (k,) ----------------
                elif a.role == "value":
                    coeff = (a.parent_name if a.parent_name.startswith("u_")
                            else f"u_{a.parent_name}_loc")
                    required_args.add(coeff)

                    body_lines += [f"{out} = np.zeros(({k_comps},), dtype={self.dtype})"]
                    for i, fn in enumerate(a.field_names):
                        s0 = self._field_slice(fn).start; s1 = self._field_slice(fn).stop
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
                                f"if {laploc}.shape[0] == {coeff}.shape[0]:",
                                f"    {out}[{i}] = laplacian_qp({coeff}, {laploc})",
                                f"else:",
                                f"    {me} = {map_arr}[e]",
                                f"    {lap_sub} = {laploc}[{s0}:{s1}]",
                                f"    {lap_pad} = scatter_tensor_to_union({lap_sub}, {me}, n_union, {self.dtype})",
                                f"    {out}[{i}] = laplacian_qp({coeff}, {lap_pad})",
                            ]
                        else:
                            body_lines += [
                                f"{out}[{i}] = laplacian_qp({coeff}, {laploc})"
                            ]

        

                    lap_value_var = out if a.is_vector else f"{out}[0]"
                    lap_value_shape = (k_comps,) if a.is_vector else ()
                    stack.append(StackItem(var_name=lap_value_var, role="value", shape=lap_value_shape,
                                        is_vector=bool(a.is_vector), field_names=a.field_names, 
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
            elif isinstance(op, LoadQuadratureState):
                required_args.add(op.name)
                required_args.add("qstate_owner_id")
                var_name = new_var("qstate")
                body_lines.append(f"{var_name} = {op.name}[qstate_owner_id[e], q]")

                is_vec = len(op.tensor_shape) == 1
                stack.append(
                    StackItem(
                        var_name=var_name,
                        role="value",
                        shape=op.tensor_shape,
                        is_vector=is_vec,
                    )
                )
            # --- analytic (pre-tabulated) ---------------------------------------------
            elif isinstance(op, LoadAnalytic):
                param = f"ana_{op.func_id}"                  # unique name in PARAM_ORDER
                required_args.add(param)
                # Avoid collisions with PARAM_ORDER names like "ana_0"/"ana_1" when
                # Analytic ids are small integers (we use stable indices for caching).
                var_name = new_var("ana_val")
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
                    and followed_by_diff
                    and getattr(op, "component_index", None) is None):
                    stack.append(
                        StackItem(
                            var_name="__basis__",
                            role=op.role,
                            shape=(len(op.field_names), self.active_n_dofs),
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

                hdiv_field = None
                try:
                    if (
                        field_names
                        and len(field_names) == 1
                        and getattr(self.me, "_field_families", {}).get(str(field_names[0])) == "RT"
                    ):
                        hdiv_field = str(field_names[0])
                except Exception:
                    hdiv_field = None

                hdiv_component = hdiv_field is not None and op.component_index is not None
                if hdiv_component:
                    if is_sided:
                        raise NotImplementedError("Direct H(div) component loads are currently implemented for volume integrals only in the JIT backend.")
                    comp_idx = int(op.component_index)
                    fld = str(hdiv_field)
                    s0 = self._field_slice(fld).start
                    s1 = self._field_slice(fld).stop
                    if op.role in ("test", "trial"):
                        if deriv_order == (0, 0):
                            tbl = f"hval_{fld}"
                            required_args.add(tbl)
                            var_name = new_var("hdiv_comp")
                            body_lines.append(f"{var_name} = {tbl}[e, q, {comp_idx}][None, :].copy()")
                        elif deriv_order in {(1, 0), (0, 1)}:
                            tbl = f"hgrad_{fld}"
                            ax = 0 if deriv_order == (1, 0) else 1
                            required_args.add(tbl)
                            var_name = new_var("hdiv_d1")
                            body_lines.append(f"{var_name} = {tbl}[e, q, {comp_idx}, :, {ax}][None, :].copy()")
                        elif deriv_order in {(2, 0), (1, 1), (0, 2)}:
                            tbl = f"hhess_{fld}"
                            a0, a1 = (0, 0) if deriv_order == (2, 0) else (0, 1) if deriv_order == (1, 1) else (1, 1)
                            required_args.add(tbl)
                            var_name = new_var("hdiv_d2")
                            body_lines.append(f"{var_name} = {tbl}[e, q, {comp_idx}, :, {a0}, {a1}][None, :].copy()")
                        else:
                            raise NotImplementedError(f"H(div) component derivative order {deriv_order} not implemented in JIT.")
                        stack.append(
                            StackItem(
                                var_name=var_name,
                                role=op.role,
                                shape=(1, self.active_n_dofs),
                                is_vector=False,
                                field_names=field_names,
                                parent_name=op.name,
                                side=op.side,
                                field_sides=op.field_sides or [],
                            )
                        )
                        continue

                    # H(div) coefficient component probes must use the already
                    # element-gathered coefficient block so the kernel reads the
                    # current element's local RT dofs, not one raw global vector.
                    coeff_sym = op.name if op.name.startswith("u_") and op.name.endswith("_loc") else f"u_{op.name}_loc"
                    required_args.add(coeff_sym)
                    solution_func_names.add(coeff_sym)
                    if deriv_order == (0, 0):
                        tbl = f"hval_{fld}"
                        row_expr = f"{tbl}[e, q, {comp_idx}]"
                    elif deriv_order in {(1, 0), (0, 1)}:
                        tbl = f"hgrad_{fld}"
                        ax = 0 if deriv_order == (1, 0) else 1
                        row_expr = f"{tbl}[e, q, {comp_idx}, :, {ax}]"
                    elif deriv_order in {(2, 0), (1, 1), (0, 2)}:
                        tbl = f"hhess_{fld}"
                        a0, a1 = (0, 0) if deriv_order == (2, 0) else (0, 1) if deriv_order == (1, 1) else (1, 1)
                        row_expr = f"{tbl}[e, q, {comp_idx}, :, {a0}, {a1}]"
                    else:
                        raise NotImplementedError(f"H(div) component derivative order {deriv_order} not implemented in JIT.")
                    required_args.add(tbl)
                    val_var = new_var(f"{op.name}_hdiv_comp")
                    row_var = new_var("hdiv_row")
                    coeff_loc = new_var("hdiv_coeff")
                    body_lines += [
                        f"{row_var} = {row_expr}",
                        f"{coeff_loc} = {coeff_sym} if {coeff_sym}.shape[0] == {row_var}.shape[0] else {coeff_sym}[{s0}:{s1}]",
                        f"{val_var} = load_variable_qp({coeff_loc}, {row_var})",
                    ]
                    stack.append(
                        StackItem(
                            var_name=val_var,
                            role="value",
                            shape=(),
                            is_vector=False,
                            field_names=field_names,
                            parent_name=coeff_sym,
                            side=op.side,
                            field_sides=op.field_sides or [],
                        )
                    )
                    continue

             

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

         

                if hdiv_field is not None:
                    if tuple(deriv_order) != (0, 0):
                        raise NotImplementedError("Derivatives of H(div) fields are not supported in JIT; use div().")
                    if is_sided:
                        side_tag = "pos" if op.side == "+" else "neg"
                        basis_arg_names = [f"bvec_{hdiv_field}_{side_tag}"]
                    else:
                        basis_arg_names = [f"bvec_{hdiv_field}"]
                else:
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
                    if op.side and hdiv_field is None:
                        required_args.add("gdofs_map")  # used for union width                                              # "+" or "-"
                        padded_vars = []                                     # one per component
                        for i, bq in enumerate(basis_vars_at_q):
                            fld_i = field_names[i]
                            # field slice in the element-union vector
                            s0 = self._field_slice(fld_i).start
                            s1 = self._field_slice(fld_i).stop
                            side_tag = self._component_side_tag(op.side, op.field_sides, fld_i, i)
                            map_array_name = f"{side_tag}_map_{fld_i}"
                            required_args.add(map_array_name)
                            pad = new_var(f"padded_basis{i}")
                            body_lines += [
                                f"{pad} = pad_basis_to_union({bq}, {map_array_name}[e], n_union, {s0}, {s1}, {self.dtype})",
                            ]
                            padded_vars.append(pad)

                        basis_vars_at_q = padded_vars        # hand off the padded list
                        final_basis_var = padded_vars[0]     # for scalar reshape path
                        n_dofs = self.active_n_dofs

                    # ---------- volume / interface -----------------------------------
                    elif not op.side:
                        final_basis_var = basis_vars_at_q[0]
                        n_dofs = self.active_n_dofs
                    else:
                        # H(div): facet tables are already in union layout (pos/neg).
                        final_basis_var = basis_vars_at_q[0]
                        n_dofs = self.active_n_dofs

                    ox, oy = deriv_order
                    tot = ox + oy

                    # ---------- (A) 0th order: keep your fast path -------------------
                    if tot == 0:
                        if hdiv_field is not None:
                            if not op.is_vector:
                                raise NotImplementedError(
                                    f"RT field '{hdiv_field}' must be used with HdivTestFunction/HdivTrialFunction."
                                )
                            s0 = self._field_slice(hdiv_field).start
                            s1 = self._field_slice(hdiv_field).stop
                            # contravariant Piola via adj(J_inv): u = adj(J_inv) @ uhat
                            if op.side == "+":
                                sign_tab = f"sign_{hdiv_field}_pos"
                            elif op.side == "-":
                                sign_tab = f"sign_{hdiv_field}_neg"
                            else:
                                sign_tab = f"sign_{hdiv_field}"
                            required_args.add(sign_tab)
                            if op.side == "+":
                                jinv_sym = "J_inv_pos"
                            elif op.side == "-":
                                jinv_sym = "J_inv_neg"
                            else:
                                jinv_sym = "J_inv"
                            required_args.add(jinv_sym)

                            bhat = new_var("bhat")
                            sgn_src = new_var("sgn_src")
                            sgn = new_var("sgn")
                            adj = new_var("adj")
                            bphys = new_var("bphys")
                            t0 = new_var("t0")
                            t1 = new_var("t1")
                            body_lines += [
                                f"{bhat} = {basis_arg_names[0]}[e, q]",
                                f"{sgn_src} = {sign_tab}[e]",
                                f"{sgn} = {sgn_src} if {sgn_src}.shape[0] == {bhat}.shape[1] else {sgn_src}[{s0}:{s1}]",
                                f"A = {jinv_sym}[e, q]",
                                f"{adj} = np.empty((2, 2), dtype={self.dtype})",
                                f"{adj}[0, 0] =  A[1, 1]",
                                f"{adj}[0, 1] = -A[0, 1]",
                                f"{adj}[1, 0] = -A[1, 0]",
                                f"{adj}[1, 1] =  A[0, 0]",
                                f"{bphys} = np.empty((2, {bhat}.shape[1]), dtype={self.dtype})",
                                f"for _j in range({bhat}.shape[1]):",
                                f"    {t0} = {adj}[0, 0] * {bhat}[0, _j] + {adj}[0, 1] * {bhat}[1, _j]",
                                f"    {t1} = {adj}[1, 0] * {bhat}[0, _j] + {adj}[1, 1] * {bhat}[1, _j]",
                                f"    {bphys}[0, _j] = {t0} * {sgn}[_j]",
                                f"    {bphys}[1, _j] = {t1} * {sgn}[_j]",
                            ]
                            stack.append(
                                StackItem(
                                    var_name=bphys,
                                    role=op.role,
                                    shape=(2, self.active_n_dofs),
                                    is_vector=True,
                                    field_names=field_names,
                                    parent_name=op.name,
                                    side=op.side,
                                    field_sides=op.field_sides or [],
                                )
                            )
                            continue

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
                            if op.side and self.on_facet:
                                side_tag = self._component_side_tag(op.side, op.field_sides, fn, i)
                                n10 = f"r10_{fn}_{side_tag}"
                                n01 = f"r01_{fn}_{side_tag}"
                                required_args.update({n10, n01})
                                d10_q = new_var("d10_q")
                                d01_q = new_var("d01_q")
                                row = new_var("row")
                                body_lines += [
                                    f"{d10_q} = {n10}[e, q]",
                                    f"{d01_q} = {n01}[e, q]",
                                    f"{row} = {d10_q} * {Aq}[0, {comp}] + {d01_q} * {Aq}[1, {comp}]",
                                ]
                                s0 = self._field_slice(fn).start; s1 = self._field_slice(fn).stop
                                map_arr = f"{side_tag}_map_{fn}"
                                required_args.add(map_arr)
                                pad = new_var("pad")
                                body_lines.append(
                                    f"{pad} = pad_basis_to_union({row}, {map_arr}[e], n_union, {s0}, {s1}, {self.dtype})"
                                )
                                rows.append(pad)
                            else:
                                nm = f"g_{fn}"
                                required_args.add(nm)
                                gloc = new_var("g_loc"); prow = new_var("prow")
                                body_lines += [
                                    f"{gloc} = np.ascontiguousarray({nm}[e, q]) @ np.ascontiguousarray({Aq}.copy())",
                                    f"{prow} = {gloc}[:, {comp}]",
                                ]
                                rows.append(prow)
                        var_name = new_var("d1_stack")
                        if not op.is_vector:
                            body_lines.append(f"{var_name} = {rows[0]}[None, :].copy()")
                            shape = (1, self.active_n_dofs)
                        else:
                            body_lines.append(f"{var_name} = np.stack(({', '.join(rows)}))")
                            shape = (len(field_names), self.active_n_dofs)
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
                            s0 = self._field_slice(fn).start; s1 = self._field_slice(fn).stop
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
                        d10v = "d10_s" if op.side else "d10_q"
                        d01v = "d01_s" if op.side else "d01_q"
                        d20v = "d20_s" if op.side else "d20_q"
                        d11v = "d11_s" if op.side else "d11_q"
                        d02v = "d02_s" if op.side else "d02_q"
                        if tot == 2:
                            body_lines += [
                                f"Hx = {H0}[e, q]; Hy = {H1}[e, q]",
                                f"Href = np.zeros((2,2), dtype={self.dtype})",
                                "for j in range(nloc):",
                                ("    Href[0,0] = d20_s[j]; Href[0,1] = d11_s[j]; Href[1,0] = d11_s[j]; Href[1,1] = d02_s[j]"
                                 if op.side else
                                 "    Href[0,0] = d20_q[j]; Href[0,1] = d11_q[j]; Href[1,0] = d11_q[j]; Href[1,1] = d02_q[j]"),
                                f"    AH00 = Href[0,0]*{A_loc}[0,0] + Href[0,1]*{A_loc}[1,0]",
                                f"    AH01 = Href[0,0]*{A_loc}[0,1] + Href[0,1]*{A_loc}[1,1]",
                                f"    AH10 = Href[1,0]*{A_loc}[0,0] + Href[1,1]*{A_loc}[1,0]",
                                f"    AH11 = Href[1,0]*{A_loc}[0,1] + Href[1,1]*{A_loc}[1,1]",
                                f"    core = np.empty((2,2), dtype={self.dtype})",
                                f"    core[0,0] = {A_loc}[0,0]*AH00 + {A_loc}[1,0]*AH10",
                                f"    core[0,1] = {A_loc}[0,0]*AH01 + {A_loc}[1,0]*AH11",
                                f"    core[1,0] = {A_loc}[0,1]*AH00 + {A_loc}[1,1]*AH10",
                                f"    core[1,1] = {A_loc}[0,1]*AH01 + {A_loc}[1,1]*AH11",
                                ("    Hphys = core + d10_s[j]*Hx + d01_s[j]*Hy"
                                 if op.side else
                                 "    Hphys = core + d10_q[j]*Hx + d01_q[j]*Hy"),
                                "    if   axes[0]==0 and axes[1]==0:  val = Hphys[0,0]",
                                "    elif axes[0]==1 and axes[1]==1:  val = Hphys[1,1]",
                                "    else:                              val = Hphys[0,1]",
                                f"    {row}[j] = val",
                            ]
                        elif tot == 3:
                            d30v = "d30_s" if op.side else "d30_q"
                            d21v = "d21_s" if op.side else "d21_q"
                            d12v = "d12_s" if op.side else "d12_q"
                            d03v = "d03_s" if op.side else "d03_q"
                            body_lines += [
                                f"Hx = {H0}[e, q]; Hy = {H1}[e, q]; Tx0={T0}[e,q]; Tx1={T1}[e,q]",
                                f"{row} = pushforward_d3({d10v}, {d01v}, {d20v}, {d11v}, {d02v}, {d30v}, {d21v}, {d12v}, {d03v}, {A_loc}, Hx, Hy, Tx0, Tx1, axes, {self.dtype})",
                            ]
                        else:  # tot == 4
                            d30v = "d30_s" if op.side else "d30_q"
                            d21v = "d21_s" if op.side else "d21_q"
                            d12v = "d12_s" if op.side else "d12_q"
                            d03v = "d03_s" if op.side else "d03_q"
                            d40v = "d40_s" if op.side else "d40_q"
                            d31v = "d31_s" if op.side else "d31_q"
                            d22v = "d22_s" if op.side else "d22_q"
                            d13v = "d13_s" if op.side else "d13_q"
                            d04v = "d04_s" if op.side else "d04_q"
                            body_lines += [
                                f"Hx = {H0}[e, q]; Hy = {H1}[e, q]; Tx0={T0}[e,q]; Tx1={T1}[e,q]; Qx0={Q0}[e,q]; Qx1={Q1}[e,q]",
                                f"{row} = pushforward_d4({d10v}, {d01v}, {d20v}, {d11v}, {d02v}, {d30v}, {d21v}, {d12v}, {d03v}, {d40v}, {d31v}, {d22v}, {d13v}, {d04v}, {A_loc}, Hx, Hy, Tx0, Tx1, Qx0, Qx1, axes, {self.dtype})",
                            ]

                        # Pad to union if side-restricted
                        if op.side:
                            side_tag = self._component_side_tag(op.side, op.field_sides, fn, i)
                            map_arr = f"{side_tag}_map_{fn}"
                            required_args.add(map_arr)
                            s0 = self._field_slice(fn).start; s1 = self._field_slice(fn).stop
                            pad = new_var("pad")
                            body_lines.append(
                                f"{pad} = pad_basis_to_union({row}, {map_arr}[e], n_union, {s0}, {s1}, {self.dtype})"
                            )
                            out_rows.append(pad)
                        else:
                            out_rows.append(row)

                    # stack per-component rows
                    var_name = new_var("d_stack")
                    if not op.is_vector:
                        body_lines.append(f"{var_name} = {out_rows[0]}[None, :].copy()")
                        shape = (1, self.active_n_dofs)
                    else:
                        body_lines.append(f"{var_name} = np.stack(({', '.join(out_rows)}))")
                        shape = (len(field_names), self.active_n_dofs)

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
                    apply_restrict_mask = isinstance(next_op, CheckDomain)

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
                    if op.side and hdiv_field is None:                      # "+" or "-"
                        padded = []
                        for i, b_var in enumerate(basis_vars_at_q):
                            fld_i = field_names[i]
                            # field slice bounds for conditional owner→field alignment
                            s0 = self._field_slice(fld_i).start
                            s1 = self._field_slice(fld_i).stop
                            side_tag = self._component_side_tag(op.side, op.field_sides, fld_i, i)
                            map_array_name = f"{side_tag}_map_{fld_i}"
                            required_args.add(map_array_name)
                            pad   = new_var(f"padded_basis{i}")
                            body_lines.append(
                                f"{pad} = pad_basis_to_union({b_var}, {map_array_name}[e], n_union, {s0}, {s1}, {self.dtype})"
                            )
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
                        if hdiv_field is not None:
                            s0 = self._field_slice(hdiv_field).start
                            s1 = self._field_slice(hdiv_field).stop
                            if op.side == "+":
                                sign_tab = f"sign_{hdiv_field}_pos"
                            elif op.side == "-":
                                sign_tab = f"sign_{hdiv_field}_neg"
                            else:
                                sign_tab = f"sign_{hdiv_field}"
                            required_args.add(sign_tab)
                            if op.side == "+":
                                jinv_sym = "J_inv_pos"
                            elif op.side == "-":
                                jinv_sym = "J_inv_neg"
                            else:
                                jinv_sym = "J_inv"
                            required_args.add(jinv_sym)

                            bhat = new_var("bhat")
                            sgn_src = new_var("sgn_src")
                            sgn = new_var("sgn")
                            adj = new_var("adj")
                            bphys = new_var("bphys")
                            coeff_loc = new_var("coeff_loc")
                            t0 = new_var("t0")
                            t1 = new_var("t1")
                            body_lines += [
                                f"{bhat} = {basis_arg_names[0]}[e, q]",
                                f"{sgn_src} = {sign_tab}[e]",
                                f"{sgn} = {sgn_src} if {sgn_src}.shape[0] == {bhat}.shape[1] else {sgn_src}[{s0}:{s1}]",
                                f"A = {jinv_sym}[e, q]",
                                f"{adj} = np.empty((2, 2), dtype={self.dtype})",
                                f"{adj}[0, 0] =  A[1, 1]",
                                f"{adj}[0, 1] = -A[0, 1]",
                                f"{adj}[1, 0] = -A[1, 0]",
                                f"{adj}[1, 1] =  A[0, 0]",
                                f"{bphys} = np.empty((2, {bhat}.shape[1]), dtype={self.dtype})",
                                f"for _j in range({bhat}.shape[1]):",
                                f"    {t0} = {adj}[0, 0] * {bhat}[0, _j] + {adj}[0, 1] * {bhat}[1, _j]",
                                f"    {t1} = {adj}[1, 0] * {bhat}[0, _j] + {adj}[1, 1] * {bhat}[1, _j]",
                                f"    {bphys}[0, _j] = {t0} * {sgn}[_j]",
                                f"    {bphys}[1, _j] = {t1} * {sgn}[_j]",
                                f"{coeff_loc} = {coeff_sym} if {coeff_sym}.shape[0] == {bphys}.shape[1] else {coeff_sym}[{s0}:{s1}]",
                                f"{val_var} = np.array([load_variable_qp({coeff_loc}, {bphys}[0]), load_variable_qp({coeff_loc}, {bphys}[1])], dtype={self.dtype})",
                            ]
                            shape = (2,)
                            stack.append(
                                StackItem(
                                    var_name=val_var,
                                    role="value",
                                    shape=shape,
                                    is_vector=True,
                                    field_names=field_names,
                                    parent_name=coeff_sym,
                                    side=op.side,
                                    field_sides=op.field_sides or [],
                                )
                            )
                            continue

                        if op.is_vector:
                            body_lines.append(f"{val_var} = np.zeros(({len(field_names)},), dtype={self.dtype})")
                            for comp_idx, b_var in enumerate(basis_vars_at_q):
                                comp_val = new_var("val_comp")
                                if apply_restrict_mask and op.side in ("+", "-"):
                                    fld_i = field_names[comp_idx]
                                    side_tag = self._component_side_tag(
                                        op.side, getattr(op, "field_sides", None), fld_i, comp_idx
                                    )
                                    if side_tag not in ("pos", "neg"):
                                        side_tag = "pos" if op.side == "+" else "neg"
                                    mask_sym = f"restrict_mask_{fld_i}_{side_tag}"
                                    required_args.add(mask_sym)
                                    coeff_masked = new_var("coeff_masked")
                                    body_lines.append(f"{coeff_masked} = {coeff_sym} * {mask_sym}[e]")
                                    body_lines.append(f"{comp_val} = load_variable_qp({coeff_masked}, {b_var})")
                                else:
                                    body_lines.append(f"{comp_val} = load_variable_qp({coeff_sym}, {b_var})")
                                body_lines.append(f"{val_var}[{comp_idx}] = {comp_val}")
                            shape = (len(field_names),)
                        else:
                            if apply_restrict_mask and op.side in ("+", "-"):
                                fld_i = field_names[0]
                                side_tag = self._component_side_tag(
                                    op.side, getattr(op, "field_sides", None), fld_i, 0
                                )
                                if side_tag not in ("pos", "neg"):
                                    side_tag = "pos" if op.side == "+" else "neg"
                                mask_sym = f"restrict_mask_{fld_i}_{side_tag}"
                                required_args.add(mask_sym)
                                coeff_masked = new_var("coeff_masked")
                                body_lines.append(f"{coeff_masked} = {coeff_sym} * {mask_sym}[e]")
                                body_lines.append(f"{val_var} = load_variable_qp({coeff_masked}, {basis_vars_at_q[0]})")
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
                                    s0 = self._field_slice(fn).start; s1 = self._field_slice(fn).stop
                                    side_tag = self._component_side_tag(op.side, op.field_sides, fn, i)
                                    map_arr = f"{side_tag}_map_{fn}"
                                    required_args.add(map_arr)
                                    pad = new_var("pad")
                                    body_lines.append(
                                        f"{pad} = pad_basis_to_union({row}, {map_arr}[e], n_union, {s0}, {s1}, {self.dtype})"
                                    )
                                    body_lines.append(f"{rv} = load_variable_qp({coeff_sym}, {pad})")
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
                                    s0 = self._field_slice(fn).start; s1 = self._field_slice(fn).stop
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

                                d10v = "d10_s" if op.side else "d10_q"
                                d01v = "d01_s" if op.side else "d01_q"
                                d20v = "d20_s" if op.side else "d20_q"
                                d11v = "d11_s" if op.side else "d11_q"
                                d02v = "d02_s" if op.side else "d02_q"
                                if tot == 2:
                                    body_lines += [
                                        f"Hx = {H0}[e,q]; Hy = {H1}[e,q]",
                                        "Href = np.zeros((2,2), dtype={})".format(self.dtype),
                                        "for j in range(nloc):",
                                        ("    Href[0,0]=d20_s[j]; Href[0,1]=d11_s[j]; Href[1,0]=d11_s[j]; Href[1,1]=d02_s[j]"
                                         if op.side else
                                         "    Href[0,0]=d20_q[j]; Href[0,1]=d11_q[j]; Href[1,0]=d11_q[j]; Href[1,1]=d02_q[j]"),
                                        f"    AH00 = Href[0,0]*{A_loc}[0,0] + Href[0,1]*{A_loc}[1,0]",
                                        f"    AH01 = Href[0,0]*{A_loc}[0,1] + Href[0,1]*{A_loc}[1,1]",
                                        f"    AH10 = Href[1,0]*{A_loc}[0,0] + Href[1,1]*{A_loc}[1,0]",
                                        f"    AH11 = Href[1,0]*{A_loc}[0,1] + Href[1,1]*{A_loc}[1,1]",
                                        f"    core = np.empty((2,2), dtype={self.dtype})",
                                        f"    core[0,0] = {A_loc}[0,0]*AH00 + {A_loc}[1,0]*AH10",
                                        f"    core[0,1] = {A_loc}[0,0]*AH01 + {A_loc}[1,0]*AH11",
                                        f"    core[1,0] = {A_loc}[0,1]*AH00 + {A_loc}[1,1]*AH10",
                                        f"    core[1,1] = {A_loc}[0,1]*AH01 + {A_loc}[1,1]*AH11",
                                        ("    Hphys = core + d10_s[j]*Hx + d01_s[j]*Hy"
                                         if op.side else
                                         "    Hphys = core + d10_q[j]*Hx + d01_q[j]*Hy"),
                                        "    if   axes[0]==0 and axes[1]==0:  val = Hphys[0,0]",
                                        "    elif axes[0]==1 and axes[1]==1:  val = Hphys[1,1]",
                                        "    else:                              val = Hphys[0,1]",
                                        f"    {row}[j] = val",
                                    ]
                                elif tot == 3:
                                    d30v = "d30_s" if op.side else "d30_q"
                                    d21v = "d21_s" if op.side else "d21_q"
                                    d12v = "d12_s" if op.side else "d12_q"
                                    d03v = "d03_s" if op.side else "d03_q"
                                    body_lines += [
                                        f"Hx = {H0}[e,q]; Hy = {H1}[e,q]; Tx0={T0}[e,q]; Tx1={T1}[e,q]",
                                        f"{row} = pushforward_d3({d10v}, {d01v}, {d20v}, {d11v}, {d02v}, {d30v}, {d21v}, {d12v}, {d03v}, {A_loc}, Hx, Hy, Tx0, Tx1, axes, {self.dtype})",
                                    ]
                                else:  # tot == 4
                                    d30v = "d30_s" if op.side else "d30_q"
                                    d21v = "d21_s" if op.side else "d21_q"
                                    d12v = "d12_s" if op.side else "d12_q"
                                    d03v = "d03_s" if op.side else "d03_q"
                                    d40v = "d40_s" if op.side else "d40_q"
                                    d31v = "d31_s" if op.side else "d31_q"
                                    d22v = "d22_s" if op.side else "d22_q"
                                    d13v = "d13_s" if op.side else "d13_q"
                                    d04v = "d04_s" if op.side else "d04_q"
                                    body_lines += [
                                        f"Hx = {H0}[e,q]; Hy = {H1}[e,q]; Tx0={T0}[e,q]; Tx1={T1}[e,q]; Qx0={Q0}[e,q]; Qx1={Q1}[e,q]",
                                        f"{row} = pushforward_d4({d10v}, {d01v}, {d20v}, {d11v}, {d02v}, {d30v}, {d21v}, {d12v}, {d03v}, {d40v}, {d31v}, {d22v}, {d13v}, {d04v}, {A_loc}, Hx, Hy, Tx0, Tx1, Qx0, Qx1, axes, {self.dtype})",
                                    ]

                                # collapse with coefficients; pad to union if sided
                                rv = new_var("rv")
                                if op.side:
                                    side_tag = self._component_side_tag(op.side, op.field_sides, fn, 0)
                                    map_arr = f"{side_tag}_map_{fn}"
                                    required_args.add(map_arr)
                                    s0 = self._field_slice(fn).start; s1 = self._field_slice(fn).stop
                                    pad = new_var("pad")
                                    body_lines.append(
                                        f"{pad} = pad_basis_to_union({row}, {map_arr}[e], n_union, {s0}, {s1}, {self.dtype})"
                                    )
                                    body_lines.append(f"{rv} = load_variable_qp({coeff_sym}, {pad})")
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
                # Scalars are passed as 0d NumPy arrays (for ABI compatibility with the C++ backend).
                # Convert them once to a true scalar to keep Numba typing happy (e.g. float(x) on
                # a 0d array is not supported).
                if getattr(op, "shape", ()) == ():
                    np_array_var = new_var("const_val")
                    body_lines.append(f"{np_array_var} = {op.name}.item()")
                else:
                    np_array_var = new_var("const_np_arr")
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

            elif isinstance(op, MeshSize):
                res = new_var("mesh_h")
                if self.on_facet:
                    required_args.add("owner_id")
                    required_args.add("h_arr")
                    body_lines.append(f"{res} = h_arr[owner_id[e]]")
                else:
                    # Pointwise mesh size from the (possibly deformed) Jacobian determinant.
                    required_args.add("detJ")
                    factor = 2.0 if getattr(self.me.mesh, "element_type", "tri") == "quad" else 1.0
                    body_lines.append(f"{res} = {factor} * np.sqrt(abs(detJ[e, q]))")
                stack.append(StackItem(var_name=res, shape=(), is_vector=False, role='const', is_gradient=False))


            # ------------------------------------------------------------------
            # Trace operator --------
            # ------------------------------------------------------------------
            elif isinstance(op, Trace):
                a = stack.pop()
                res_var = new_var("trace_res")
                trace_meta = _try_trace_meta(a, spatial_dim=self.spatial_dim)

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
                                            field_sides=a.field_sides or [],
                                            expression_meta=trace_meta))

                # --- Case 2: Trace of a grad(Test/Trial) function tensor (e.g., shape (2, n, 2)) ---
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
                                            field_sides=a.field_sides or [],
                                            expression_meta=trace_meta))

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
                                            field_sides=a.field_sides or [],
                                            expression_meta=trace_meta))

                # --- Else: The shape is not a 2D or 3D tensor, so it's invalid. ---
                else:
                    raise TypeError(f"Cannot take trace of an operand with shape {a.shape}. Must be a 2D or 3D tensor.")
            
            elif isinstance(op, Determinant):
                a = stack.pop()
                if a.shape != (2, 2) or a.role not in ("value", "const"):
                    raise NotImplementedError(
                        "Determinant expects a 2x2 numeric tensor (role 'value' or 'const')."
                    )
                det_meta = _try_determinant_meta(a, spatial_dim=self.spatial_dim)
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
                        expression_meta=det_meta,
                    )
                )

            elif isinstance(op, Cofactor):
                a = stack.pop()
                if a.role in ("trial", "test") and len(a.shape) == 3 and a.shape[0] == a.shape[-1] == 2:
                    res_var = new_var(f"cof_basis_{a.role}")
                    body_lines += [f"# Cofactor of a symbolic tensor -> tensor basis of shape (2, n, 2) ! only valid for 2d",
                                   f"a00 = {a.var_name}[0, :, 0]",
                                   f"a01 = {a.var_name}[0, :, 1]",
                                   f"a10 = {a.var_name}[1, :, 0]",
                                   f"a11 = {a.var_name}[1, :, 1]",
                                   f"{res_var} = np.zeros_like({a.var_name})",
                                   f"{res_var}[0, :, 0] = a11",
                                   f"{res_var}[0, :, 1] = -a10",
                                   f"{res_var}[1, :, 0] = -a01",
                                   f"{res_var}[1, :, 1] = a00",]
                    stack.append(
                        a._replace(
                            var_name=res_var,
                            shape=(2, a.shape[1], 2),
                        )
                    )
                else:
                    if a.shape != (2, 2) or a.role not in ("value", "const"):
                        raise NotImplementedError(
                            "Cofactor expects a 2x2 numeric tensor (role 'value' or 'const')."
                        )
                    res_var = new_var("cof2")
                    body_lines += [
                        f"a00 = {a.var_name}[0, 0]",
                        f"a01 = {a.var_name}[0, 1]",
                        f"a10 = {a.var_name}[1, 0]",
                        f"a11 = {a.var_name}[1, 1]",
                        f"{res_var} = np.array([[a11, -a10], [-a01, a00]], dtype={self.dtype})",
                    ]
                    stack.append(
                        a._replace(
                            var_name=res_var,
                            role=a.role,
                            shape=(2, 2),
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

                if a.is_divergence:
                    if self.spatial_dim != 2:
                        raise NotImplementedError("grad(div(.)) is currently implemented for 2D only.")
                    if len(a.field_names) < self.spatial_dim:
                        raise NotImplementedError(
                            "grad(div(.)) requires a vector field with one component per spatial direction."
                        )

                    if a.side == "+":
                        H0 = "pos_Hxi0"; H1 = "pos_Hxi1"; suff = "_pos"
                    elif a.side == "-":
                        H0 = "neg_Hxi0"; H1 = "neg_Hxi1"; suff = "_neg"
                    else:
                        H0 = "Hxi0"; H1 = "Hxi1"; suff = ""
                    required_args.update({H0, H1, "gdofs_map"})

                    d10, d01, d20, d11, d02 = [], [], [], [], []
                    for i, fn in enumerate(a.field_names[: self.spatial_dim]):
                        if suff == "":
                            n10 = f"d10_{fn}"; n01 = f"d01_{fn}"; n20 = f"d20_{fn}"; n11 = f"d11_{fn}"; n02 = f"d02_{fn}"
                        else:
                            side_tag = self._component_side_tag(a.side, getattr(a, "field_sides", None), fn, i)
                            n10 = f"r10_{fn}_{side_tag}"; n01 = f"r01_{fn}_{side_tag}"
                            n20 = f"r20_{fn}_{side_tag}"; n11 = f"r11_{fn}_{side_tag}"; n02 = f"r02_{fn}_{side_tag}"
                        required_args.update({n10, n01, n20, n11, n02})
                        d10.append(n10); d01.append(n01); d20.append(n20); d11.append(n11); d02.append(n02)

                    if a.role in ("test", "trial"):
                        out = new_var("graddiv")
                        body_lines.append(f"{out} = np.zeros(({self.spatial_dim}, n_union), dtype={self.dtype})")
                        for i, fn in enumerate(a.field_names[: self.spatial_dim]):
                            s0 = self._field_slice(fn).start
                            s1 = self._field_slice(fn).stop
                            Hloc = new_var("Hloc")
                            body_lines += [
                                f"Hx = {H0}[e, q]",
                                f"Hy = {H1}[e, q]",
                                f"d10_q = {d10[i]}[e, q]",
                                f"d01_q = {d01[i]}[e, q]",
                                f"d20_q = {d20[i]}[e, q]",
                                f"d11_q = {d11[i]}[e, q]",
                                f"d02_q = {d02[i]}[e, q]",
                                f"{Hloc} = compute_physical_hessian(d20_q, d11_q, d02_q, d10_q, d01_q, {jinv_q}, Hx, Hy, {self.dtype})",
                            ]
                            Huse = Hloc
                            if a.side:
                                side_tag = self._component_side_tag(a.side, a.field_sides, fn, i)
                                map_arr = f"{side_tag}_map_{fn}"
                                required_args.add(map_arr)
                                Hpad = new_var("Hpad")
                                Hsub = new_var("Hsub")
                                me = new_var("map_e")
                                body_lines += [
                                    f"if {Hloc}.shape[0] == n_union:",
                                    f"    {Hpad} = {Hloc}",
                                    f"else:",
                                    f"    {me} = {map_arr}[e]",
                                    f"    {Hsub} = {Hloc}[{s0}:{s1}, :, :]",
                                    f"    {Hpad} = scatter_tensor_to_union({Hsub}, {me}, n_union, {self.dtype})",
                                ]
                                Huse = Hpad
                            body_lines += [
                                f"{out}[0] += {Huse}[:, 0, {i}]",
                                f"{out}[1] += {Huse}[:, 1, {i}]",
                            ]

                        stack.append(
                            StackItem(
                                var_name=out,
                                role=a.role,
                                shape=(self.spatial_dim, self.active_n_dofs),
                                is_vector=True,
                                field_names=list(a.field_names[: self.spatial_dim]),
                                parent_name=a.parent_name,
                                side=a.side,
                                field_sides=a.field_sides or [],
                            )
                        )
                        continue

                    if a.role == "value":
                        coeff = a.parent_name if a.parent_name.startswith("u_") else f"u_{a.parent_name}_loc"
                        required_args.add(coeff)
                        out = new_var("graddivv")
                        body_lines.append(f"{out} = np.zeros(({self.spatial_dim},), dtype={self.dtype})")
                        for i, fn in enumerate(a.field_names[: self.spatial_dim]):
                            s0 = self._field_slice(fn).start
                            s1 = self._field_slice(fn).stop
                            Hloc = new_var("Hloc")
                            Hval = new_var("Hval")
                            body_lines += [
                                f"Hx = {H0}[e, q]",
                                f"Hy = {H1}[e, q]",
                                f"d10_q = {d10[i]}[e, q]",
                                f"d01_q = {d01[i]}[e, q]",
                                f"d20_q = {d20[i]}[e, q]",
                                f"d11_q = {d11[i]}[e, q]",
                                f"d02_q = {d02[i]}[e, q]",
                                f"{Hloc} = compute_physical_hessian(d20_q, d11_q, d02_q, d10_q, d01_q, {jinv_q}, Hx, Hy, {self.dtype})",
                            ]
                            if a.side:
                                side_tag = self._component_side_tag(a.side, a.field_sides, fn, i)
                                map_arr = f"{side_tag}_map_{fn}"
                                required_args.add(map_arr)
                                Hpad = new_var("Hpad")
                                Hsub = new_var("Hsub")
                                me = new_var("map_e")
                                body_lines += [
                                    f"if {Hloc}.shape[0] == {coeff}.shape[0]:",
                                    f"    {Hval} = hessian_qp({coeff}, {Hloc})",
                                    f"else:",
                                    f"    {me} = {map_arr}[e]",
                                    f"    {Hsub} = {Hloc}[{s0}:{s1}, :, :]",
                                    f"    {Hpad} = scatter_tensor_to_union({Hsub}, {me}, n_union, {self.dtype})",
                                    f"    {Hval} = hessian_qp({coeff}, {Hpad})",
                                ]
                            else:
                                body_lines.append(f"{Hval} = hessian_qp({coeff}, {Hloc})")
                            body_lines += [
                                f"{out}[0] += {Hval}[0, {i}]",
                                f"{out}[1] += {Hval}[1, {i}]",
                            ]

                        stack.append(
                            StackItem(
                                var_name=out,
                                role="value",
                                shape=(self.spatial_dim,),
                                is_vector=True,
                                field_names=list(a.field_names[: self.spatial_dim]),
                                parent_name=coeff,
                                side=a.side,
                                field_sides=a.field_sides or [],
                            )
                        )
                        continue

                    raise NotImplementedError(f"grad(div(.)) not implemented for role {a.role}")

                is_hdiv_grad = (
                    len(getattr(a, "field_names", []) or []) == 1
                    and getattr(self.me, "_field_families", {}).get(str(a.field_names[0]), None) == "RT"
                )
                if is_hdiv_grad:
                    fld = str(a.field_names[0])
                    if a.side in ("+", "-") and self.on_facet:
                        raise NotImplementedError("grad(H(div)) is currently implemented for volume integrals only in the JIT backend.")
                    s0 = self._field_slice(fld).start
                    s1 = self._field_slice(fld).stop

                    gvec_name = f"gvec_{fld}"
                    sign_name = f"sign_{fld}"
                    required_args.update({gvec_name, sign_name})

                    adj = new_var("adj")
                    ghat = new_var("ghat")
                    gphys = new_var("gphys")
                    sgn_src = new_var("sgn_src")
                    sgn = new_var("sgn")
                    t00 = new_var("t00")
                    t01 = new_var("t01")
                    t10 = new_var("t10")
                    t11 = new_var("t11")
                    body_lines += [
                        f"{ghat} = {gvec_name}[e, q]",
                        f"{sgn_src} = {sign_name}[e]",
                        f"{sgn} = {sgn_src} if {sgn_src}.shape[0] == {ghat}.shape[1] else {sgn_src}[{s0}:{s1}]",
                        f"{adj} = np.empty((2, 2), dtype={self.dtype})",
                        f"{adj}[0, 0] =  {jinv_q}[1, 1]",
                        f"{adj}[0, 1] = -{jinv_q}[0, 1]",
                        f"{adj}[1, 0] = -{jinv_q}[1, 0]",
                        f"{adj}[1, 1] =  {jinv_q}[0, 0]",
                        f"{gphys} = np.empty((2, {ghat}.shape[1], 2), dtype={self.dtype})",
                        f"for _j in range({ghat}.shape[1]):",
                        f"    {t00} = {adj}[0, 0] * {ghat}[0, _j, 0] + {adj}[0, 1] * {ghat}[1, _j, 0]",
                        f"    {t01} = {adj}[0, 0] * {ghat}[0, _j, 1] + {adj}[0, 1] * {ghat}[1, _j, 1]",
                        f"    {t10} = {adj}[1, 0] * {ghat}[0, _j, 0] + {adj}[1, 1] * {ghat}[1, _j, 0]",
                        f"    {t11} = {adj}[1, 0] * {ghat}[0, _j, 1] + {adj}[1, 1] * {ghat}[1, _j, 1]",
                        f"    {gphys}[0, _j, 0] = ({t00} * {jinv_q}[0, 0] + {t01} * {jinv_q}[1, 0]) * {sgn}[_j]",
                        f"    {gphys}[0, _j, 1] = ({t00} * {jinv_q}[0, 1] + {t01} * {jinv_q}[1, 1]) * {sgn}[_j]",
                        f"    {gphys}[1, _j, 0] = ({t10} * {jinv_q}[0, 0] + {t11} * {jinv_q}[1, 0]) * {sgn}[_j]",
                        f"    {gphys}[1, _j, 1] = ({t10} * {jinv_q}[0, 1] + {t11} * {jinv_q}[1, 1]) * {sgn}[_j]",
                    ]

                    if a.role in ("test", "trial"):
                        stack.append(
                            a._replace(
                                var_name=gphys,
                                shape=(2, self.active_n_dofs, self.spatial_dim),
                                is_gradient=True,
                                is_vector=False,
                                field_names=[fld, fld],
                            )
                        )
                        continue

                    if a.role == "value":
                        coeff = a.parent_name if a.parent_name.startswith("u_") else f"u_{a.parent_name}_loc"
                        required_args.add(coeff)
                        gval = new_var("hdiv_grad_val")
                        coeff_loc = new_var("hdiv_grad_coeff")
                        acc00 = new_var("acc00")
                        acc01 = new_var("acc01")
                        acc10 = new_var("acc10")
                        acc11 = new_var("acc11")
                        body_lines += [
                            f"{coeff_loc} = {coeff} if {coeff}.shape[0] == {gphys}.shape[1] else {coeff}[{s0}:{s1}]",
                            f"{acc00} = 0.0",
                            f"{acc01} = 0.0",
                            f"{acc10} = 0.0",
                            f"{acc11} = 0.0",
                            f"for _j in range({gphys}.shape[1]):",
                            f"    _c = {coeff_loc}[_j]",
                            f"    {acc00} += {gphys}[0, _j, 0] * _c",
                            f"    {acc01} += {gphys}[0, _j, 1] * _c",
                            f"    {acc10} += {gphys}[1, _j, 0] * _c",
                            f"    {acc11} += {gphys}[1, _j, 1] * _c",
                            f"{gval} = np.empty((2, {self.spatial_dim}), dtype={self.dtype})",
                            f"{gval}[0, 0] = {acc00}",
                            f"{gval}[0, 1] = {acc01}",
                            f"{gval}[1, 0] = {acc10}",
                            f"{gval}[1, 1] = {acc11}",
                        ]
                        stack.append(
                            StackItem(
                                var_name=gval,
                                role="value",
                                shape=(2, self.spatial_dim),
                                is_gradient=True,
                                is_vector=False,
                                field_names=[fld, fld],
                                parent_name=coeff,
                                side=a.side,
                                field_sides=a.field_sides or [],
                            )
                        )
                        continue

                    raise NotImplementedError(f"grad(H(div)) not implemented for role {a.role}")


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
                            s0 = self._field_slice(fld_i).start
                            s1 = self._field_slice(fld_i).stop

                            # decide per-component side tag ("pos" or "neg")
                            side_tag = self._component_side_tag(a.side, a.field_sides, fld_i, i)

                            n10 = f"r10_{fld_i}_{side_tag}"
                            n01 = f"r01_{fld_i}_{side_tag}"
                            required_args.update({n10, n01})

                            # per-component side map (local rows -> union rows)
                            map_arr = f"{side_tag}_map_{fld_i}"
                            required_args.add(map_arr)

                            pg_pad   = new_var("grad_pad")      # (n_union, 2)

                            body_lines.append(
                                f"{pg_pad} = pushforward_grad_to_union({n10}[e, q], {n01}[e, q], {jinv_q}, {map_arr}[e], n_union, {s0}, {s1}, {self.dtype})"
                            )
                            phys.append(pg_pad)

                        n_dofs = self.active_n_dofs

                    else:
                        # Volume (unsided): use volume gradient tables g_<field> with J_inv

                        grad_tab_names = [f"g_{f}" for f in a.field_names]

                        for nm in grad_tab_names:
                            required_args.add(nm)

                        for i, fld_i in enumerate(a.field_names):
                            pg_loc = new_var("grad_loc")
                            body_lines.append(f"{pg_loc} = np.ascontiguousarray({grad_tab_names[i]}[e, q]) @ np.ascontiguousarray({jinv_q}.copy())")
                            phys.append(pg_loc)

                        n_dofs = self.active_n_dofs

                    # Stack per-component physical gradients
                    if not a.is_vector:
                        var = new_var("grad_scalar")
                        body_lines.append(f"{var} = np.ascontiguousarray({phys[0]}.T.copy())")
                        shape, is_vector, is_gradient = (self.spatial_dim, n_dofs), False, True
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
                    apply_restrict_mask = isinstance(next_op, CheckDomain)
                    for i, fld in enumerate(a.field_names):

                        val = new_var("grad_val")

                        if a.side in ("+", "-"):
                            # Sided: use r10/r01 on the correct side and the side map
                            s0 = self._field_slice(fld).start
                            s1 = self._field_slice(fld).stop

                            side_tag = self._component_side_tag(a.side, getattr(a, 'field_sides', None), fld, i)
                            d10 = f"r10_{fld}_{side_tag}"
                            d01 = f"r01_{fld}_{side_tag}"
                            required_args.update({d10, d01})

                            map_sym = f"{side_tag}_map_{fld}"
                            required_args.add(map_sym)
                            mask_sym = None
                            if apply_restrict_mask and side_tag in ("pos", "neg"):
                                mask_sym = f"restrict_mask_{fld}_{side_tag}"
                                required_args.add(mask_sym)

                            coeff_e = coeff if coeff.endswith("_e") else f"{coeff}"

                            g2     = new_var("g_ref2")        # (n_loc,2) in (ξ,η)
                            phys   = new_var("g_phys2")       # (n_loc,2) in (x,y)
                            phys_s = new_var("g_phys2_s")     # (n_comp_loc,2)
                            u_sl   = new_var("u_side")        # (n_comp_loc,)
                            coeff_masked = new_var("coeff_masked")
                            mask_sl = new_var("mask_sl")

                            body_lines += [
                                f"d10_q = {d10}[e, q]",
                                f"d01_q = {d01}[e, q]",
                                f"{g2}   = np.stack((d10_q, d01_q), axis=1)",
                                f"{phys} = np.ascontiguousarray({g2}) @ np.ascontiguousarray({jinv_q}.copy())",
                                f"if {phys}.shape[0] == {coeff_e}.shape[0]:",
                                (
                                    f"    {coeff_masked} = {coeff_e} * {mask_sym}[e]"
                                    if mask_sym is not None
                                    else f"    {coeff_masked} = {coeff_e}"
                                ),
                                f"    {val} = gradient_qp({coeff_masked}, {phys})",
                                f"else:",
                                f"    {phys_s} = {phys}[{s0}:{s1}, :]",           # local vs local
                                f"    {u_sl}   = {coeff_e}[{map_sym}[e]]",
                                (
                                    f"    {mask_sl} = {mask_sym}[e][{map_sym}[e]]"
                                    if mask_sym is not None
                                    else f"    {mask_sl} = 1.0"
                                ),
                                f"    {u_sl}   = {u_sl} * {mask_sl}",
                                f"    {val}    = gradient_qp({u_sl}, {phys_s})",
                            ]

                        else:
                            # Volume (unsided): use g_<field> with J_inv
                            gnm = f"g_{fld}"
                            required_args.add(gnm)

                            coeff_e = coeff if coeff.endswith("_e") else f"{coeff}"
                            pg = new_var("phys_grad_basis")

                            body_lines += [
                                f"{pg}  = np.ascontiguousarray({gnm}[e, q]) @ np.ascontiguousarray({jinv_q}.copy())",
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
                        f"n_loc  = {a.var_name}.shape[1]",  # number of local basis functions (supports n_union=-1)
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
                            is_divergence=True,
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
                                            side=a.side, is_divergence=True,
                                            field_sides=a.field_sides or []))
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

            elif isinstance(op, HdivDiv):
                a = stack.pop()
                if not a.field_names:
                    raise ValueError("HdivDiv requires field metadata on the stack item.")
                fld = str(a.field_names[0])
                fam = getattr(self.me, "_field_families", {}).get(fld, None)
                if fam != "RT":
                    raise NotImplementedError(f"HdivDiv only supports RT fields (got {fam!r} for '{fld}').")

                # divergence tables are reference divs; map to physical via /detJ (affine-correct)
                if a.side == "+":
                    div_tab = f"div_{fld}_pos"
                    sign_tab = f"sign_{fld}_pos"
                    det_tab = "detJ_pos"
                elif a.side == "-":
                    div_tab = f"div_{fld}_neg"
                    sign_tab = f"sign_{fld}_neg"
                    det_tab = "detJ_neg"
                else:
                    div_tab = f"div_{fld}"
                    sign_tab = f"sign_{fld}"
                    det_tab = "detJ"
                required_args.update({div_tab, sign_tab, det_tab})
                s0 = self._field_slice(fld).start
                s1 = self._field_slice(fld).stop

                # signed physical divergence basis at this quadrature point
                div_hat = new_var("div_hat")
                sgn_src = new_var("sgn_src")
                sgn = new_var("sgn")
                div_phys = new_var("div_phys")
                body_lines += [
                    f"{div_hat} = {div_tab}[e, q]",
                    f"{sgn_src} = {sign_tab}[e]",
                    f"{sgn} = {sgn_src} if {sgn_src}.shape[0] == {div_hat}.shape[0] else {sgn_src}[{s0}:{s1}]",
                    f"{div_phys} = ({div_hat} * {sgn}) / {det_tab}[e, q]",
                ]

                if a.role in ("test", "trial"):
                    # scalar basis row (1, n)
                    div_var = new_var("hdiv_div_basis")
                    body_lines.append(f"{div_var} = {div_phys}[None, :].copy()")
                    stack.append(
                        StackItem(
                            var_name=div_var,
                            role=a.role,
                            shape=(1, self.active_n_dofs),
                            is_vector=False,
                            is_gradient=False,
                            field_names=a.field_names,
                            parent_name=a.parent_name,
                            side=a.side,
                            field_sides=a.field_sides or [],
                        )
                    )
                    continue

                if a.role == "value":
                    coeff_sym = getattr(a, "parent_name", "") or ""
                    if not coeff_sym:
                        raise ValueError("HdivDiv(value) requires parent_name=coefficient symbol.")
                    div_val = new_var("hdiv_div_val")
                    coeff_loc = new_var("hdiv_div_coeff")
                    body_lines += [
                        f"{coeff_loc} = {coeff_sym} if {coeff_sym}.shape[0] == {div_phys}.shape[0] else {coeff_sym}[{s0}:{s1}]",
                        f"{div_val} = load_variable_qp({coeff_loc}, {div_phys})",
                    ]
                    stack.append(
                        StackItem(
                            var_name=div_val,
                            role="value",
                            shape=(),
                            is_vector=False,
                            is_gradient=False,
                            field_names=a.field_names,
                            parent_name=coeff_sym,
                            side=a.side,
                            field_sides=a.field_sides or [],
                        )
                    )
                    continue

                raise NotImplementedError(f"HdivDiv not implemented for role '{a.role}'.")

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

            elif isinstance(op, PositivePartOp):
                a = stack.pop()
                res_var = new_var("pos_part")
                body_lines.append(f"# PositivePart: max(x,0)")
                body_lines.append(f"{res_var} = np.maximum({a.var_name}, 0.0)")
                stack.append(a._replace(var_name=res_var))

            elif isinstance(op, HeavisideOp):
                a = stack.pop()
                res_var = new_var("heaviside")
                body_lines.append(f"# Heaviside: 1 if x>0 else 0 (H(0)=0)")
                if a.shape == ():
                    body_lines.append(f"{res_var} = 1.0 if ({a.var_name} > 0.0) else 0.0")
                else:
                    body_lines.append(f"{res_var} = np.where({a.var_name} > 0.0, 1.0, 0.0)")
                stack.append(a._replace(var_name=res_var))

            elif isinstance(op, LogOp):
                a = stack.pop()
                res_var = new_var("log")
                body_lines.append("# Log: natural logarithm")
                body_lines.append(f"{res_var} = np.log({a.var_name})")
                stack.append(a._replace(var_name=res_var))

            elif isinstance(op, ExpOp):
                a = stack.pop()
                res_var = new_var("exp")
                body_lines.append("# Exp: natural exponential")
                body_lines.append(f"{res_var} = np.exp({a.var_name})")
                stack.append(a._replace(var_name=res_var))

            elif isinstance(op, TanhOp):
                a = stack.pop()
                res_var = new_var("tanh")
                body_lines.append("# Tanh: hyperbolic tangent")
                body_lines.append(f"{res_var} = np.tanh({a.var_name})")
                stack.append(a._replace(var_name=res_var))
            elif isinstance(op, SinOp):
                a = stack.pop()
                res_var = new_var("sin")
                body_lines.append("# Sin: trigonometric sine")
                body_lines.append(f"{res_var} = np.sin({a.var_name})")
                stack.append(a._replace(var_name=res_var))
            elif isinstance(op, CosOp):
                a = stack.pop()
                res_var = new_var("cos")
                body_lines.append("# Cos: trigonometric cosine")
                body_lines.append(f"{res_var} = np.cos({a.var_name})")
                stack.append(a._replace(var_name=res_var))
            elif isinstance(op, TanOp):
                a = stack.pop()
                res_var = new_var("tan")
                body_lines.append("# Tan: trigonometric tangent")
                body_lines.append(f"{res_var} = np.tan({a.var_name})")
                stack.append(a._replace(var_name=res_var))
            elif isinstance(op, AsinOp):
                a = stack.pop()
                res_var = new_var("asin")
                body_lines.append("# Asin: inverse sine")
                body_lines.append(f"{res_var} = np.arcsin({a.var_name})")
                stack.append(a._replace(var_name=res_var))
            elif isinstance(op, AcosOp):
                a = stack.pop()
                res_var = new_var("acos")
                body_lines.append("# Acos: inverse cosine")
                body_lines.append(f"{res_var} = np.arccos({a.var_name})")
                stack.append(a._replace(var_name=res_var))
            elif isinstance(op, AtanOp):
                a = stack.pop()
                res_var = new_var("atan")
                body_lines.append("# Atan: inverse tangent")
                body_lines.append(f"{res_var} = np.arctan({a.var_name})")
                stack.append(a._replace(var_name=res_var))
            elif isinstance(op, SinhOp):
                a = stack.pop()
                res_var = new_var("sinh")
                body_lines.append("# Sinh: hyperbolic sine")
                body_lines.append(f"{res_var} = np.sinh({a.var_name})")
                stack.append(a._replace(var_name=res_var))
            elif isinstance(op, CoshOp):
                a = stack.pop()
                res_var = new_var("cosh")
                body_lines.append("# Cosh: hyperbolic cosine")
                body_lines.append(f"{res_var} = np.cosh({a.var_name})")
                stack.append(a._replace(var_name=res_var))
            elif isinstance(op, AsinhOp):
                a = stack.pop()
                res_var = new_var("asinh")
                body_lines.append("# Asinh: inverse hyperbolic sine")
                body_lines.append(f"{res_var} = np.arcsinh({a.var_name})")
                stack.append(a._replace(var_name=res_var))
            elif isinstance(op, AcoshOp):
                a = stack.pop()
                res_var = new_var("acosh")
                body_lines.append("# Acosh: inverse hyperbolic cosine")
                body_lines.append(f"{res_var} = np.arccosh({a.var_name})")
                stack.append(a._replace(var_name=res_var))
            elif isinstance(op, AtanhOp):
                a = stack.pop()
                res_var = new_var("atanh")
                body_lines.append("# Atanh: inverse hyperbolic tangent")
                body_lines.append(f"{res_var} = np.arctanh({a.var_name})")
                stack.append(a._replace(var_name=res_var))

            # --- Inner OPERATORS ---
            elif isinstance(op, Inner):
                b = stack.pop(); a = stack.pop()
                res_var = new_var("inner")
                inner_plan = _try_inner_plan(a, b)
                inner_value_spec = _try_inner_value_spec(a, b)
                # print(f"Inner operation: a.role={a.role}, b.role={b.role}, a.shape={a.shape}, b.shape={b.shape}"
                #       f", is_vector: {a.is_vector}/{b.is_vector}, is_gradient: {a.is_gradient}/{b.is_gradient}"
                #       f", a.is_transpose: {a.is_transpose}, b.is_transpose: {b.is_transpose}"
                #       f", a.is_hessian: {a.is_hessian}, b.is_hessian: {b.is_hessian}")

                def _push_inner_result(
                    *,
                    default_role: str,
                    fallback_shape: tuple,
                    field_names,
                    parent_name: str,
                    side: str,
                    field_sides,
                ) -> None:
                    shape_out = tuple(fallback_shape)
                    role_out = default_role
                    layout_out = ""
                    expression_meta = inner_plan.result if inner_plan is not None else None
                    if inner_value_spec is not None:
                        planned_shape = tuple(int(v) for v in getattr(inner_value_spec, "shape", ()) or ())
                        if planned_shape:
                            shape_out = planned_shape
                        role_out = getattr(inner_value_spec, "role", role_out) or role_out
                        layout_obj = getattr(inner_value_spec, "layout", None)
                        layout_out = layout_obj.value if layout_obj is not None else layout_out
                        expression_meta = getattr(inner_value_spec, "meta", expression_meta)
                    stack.append(
                        StackItem(
                            var_name=res_var,
                            role=role_out,
                            shape=shape_out,
                            is_vector=False,
                            is_gradient=False,
                            is_hessian=False,
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides or [],
                            layout_tag=layout_out,
                            expression_meta=expression_meta,
                        )
                    )

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

                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(
                        a, b, prefer=None, strict=False
                    )
                    _push_inner_result(
                        default_role="value",
                        fallback_shape=(_basis_col_dim(test_item.shape), _basis_col_dim(trial_item.shape)),
                        field_names=field_names,
                        parent_name=parent_name,
                        side=side,
                        field_sides=field_sides,
                    )
                    continue

                elif (
                    a.role in {"test", "trial"}
                    and b.role in {"value", "const"}
                    and (
                        ((_semantic_is_basis_rank2(a, spatial_dim=self.spatial_dim)
                          and _semantic_is_value_rank2(b, spatial_dim=self.spatial_dim)
                          and not (a.is_hessian or b.is_hessian)))
                        or (_semantic_is_basis_rank1(a, spatial_dim=self.spatial_dim)
                            and _semantic_is_value_rank1(b, spatial_dim=self.spatial_dim))
                        or not (a.is_gradient or b.is_gradient or a.is_hessian or b.is_hessian)
                    )
                ):
                    # RHS: inner(Test, Function/Const) – swapped operand order.
                    body_lines.append("# RHS: Inner(Test, Function/Const)")
                    role = a.role

                    if (_semantic_is_basis_rank2(a, spatial_dim=self.spatial_dim)
                        and _semantic_is_value_rank2(b, spatial_dim=self.spatial_dim)
                        and not (a.is_hessian or b.is_hessian)):
                        body_lines.append("# RHS: Inner(Test rank-2, Value rank-2)")
                        body_lines.append(
                            f"{res_var} = inner_rank2_basis_rank2_value({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                    elif (_semantic_is_basis_rank1(a, spatial_dim=self.spatial_dim)
                        and _semantic_is_value_rank1(b, spatial_dim=self.spatial_dim)):
                        # a: basis rank-1 carrier, b: value rank-1 carrier -> (n,)
                        body_lines.append("# RHS: Inner(Test rank-1, Value rank-1)")
                        body_lines.append(
                            f"{res_var} = const_vector_dot_basis_1d({b.var_name}, {a.var_name}, {self.dtype})"
                        )
                    elif (
                        (not a.is_vector)
                        and (not b.is_vector)
                        and (
                            (len(a.shape) == 2 and a.shape[0] == 1)
                            or len(a.shape) == 1
                        )
                    ):
                        # a: (1,n), b: () -> (n,)
                        body_lines.append("# RHS: Inner(Test(Scalar), Value(Scalar))")
                        basis_expr = f"{a.var_name}[0]" if len(a.shape) == 2 else a.var_name
                        body_lines.append(
                            f"{res_var} = mul_scalar({b.var_name}, {basis_expr}, {self.dtype})"
                        )
                    else:
                        raise NotImplementedError(
                            f"Inner not implemented for roles {a.role}/{b.role}, "
                            f"is_vector: {a.is_vector}/{b.is_vector}, "
                            f"is_gradient: {a.is_gradient}/{b.is_gradient}, "
                            f"shapes: {a.shape}/{b.shape}, is_hessian: {a.is_hessian}/{b.is_hessian}"
                        )

                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(
                        a, b, prefer="a", strict=False
                    )
                    _push_inner_result(
                        default_role=role,
                        fallback_shape=(_basis_col_dim(a.shape),),
                        field_names=field_names,
                        parent_name=parent_name,
                        side=side,
                        field_sides=field_sides,
                    )
                    continue


                # elif a.role == 'const' and b.role == 'const' and a.shape == b.shape:
                #     body_lines.append(f'# Inner(Const, Const): element-wise product')
                #     body_lines.append(f'{res_var} = {a.var_name} * {b.var_name}')
                elif a.role in {'value', 'const'} and b.role in {'test', 'trial'}:
                    body_lines.append(f"# Inner(Function, {b.role.title()})")
                    # body_lines.append(f"print(f'RHS inner: a.shape: {{{a.var_name}.shape}}, "
                    #                   f"b.shape: {{{b.var_name}.shape}}, ')")

                    if (_semantic_is_value_rank2(a, spatial_dim=self.spatial_dim)
                          and _semantic_is_basis_rank2(b, spatial_dim=self.spatial_dim)
                          and not (a.is_hessian or b.is_hessian)):
                        body_lines.append(f"# Inner(Value rank-2, {b.role.title()} rank-2)")
                        body_lines.append(
                            f"{res_var} = inner_rank2_value_rank2_basis({a.var_name}, {b.var_name}, {self.dtype})"
                        )

                    elif a.is_gradient and b.is_gradient:
                        # a: (k,d) collapsed grad(Function), b: (k,n,d)
                        body_lines.append(f"# Inner(Grad(Function), Grad({b.role.title()}))")
                        body_lines.append(
                            f"{res_var} = inner_grad_function_grad_test({a.var_name}, {b.var_name}, {self.dtype})"
                        )

                    elif a.is_hessian and b.is_hessian:
                        # a: (k,2,2) collapsed Hess(Function), b: (k,n,2,2)
                        body_lines.append(f"# Inner(Hessian(Function), Hessian({b.role.title()}))")
                        body_lines.append(
                            f"{res_var} = inner_hessian_function_hessian_test({a.var_name}, {b.var_name}, {self.dtype})"
                        )

                    elif (_semantic_is_value_rank1(a, spatial_dim=self.spatial_dim)
                          and _semantic_is_basis_rank1(b, spatial_dim=self.spatial_dim)):
                        # a: value rank-1 carrier, b: basis rank-1 carrier → (n,)
                        body_lines.append(f"# Inner(Value rank-1, {b.role.title()} rank-1)")
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
                        # grad(scalarFunction): a: (d,), b: (d,n) or legacy (1,n,d) → (n,)
                        body_lines.append("# RHS: Inner(grad(scalarFunction), grad(Test))")
                        if len(b.shape) == 2:
                            body_lines.append(
                                f"{res_var} = contract_last_first(np.ascontiguousarray({b.var_name}.T), {a.var_name}, {self.dtype})"
                            )
                        else:
                            body_lines.append(
                                f"{res_var} = contract_last_first({b.var_name}[0], {a.var_name}, {self.dtype})"
                            )
                    elif (
                        not a.is_vector
                        and not b.is_vector
                        and (
                            (len(b.shape) == 2 and b.shape[0] == 1)
                            or len(b.shape) == 1
                        )
                    ):
                        # a: (), b: (n,)  → (n,)
                        body_lines.append(f"# Inner(Value(Scalar), {b.role.title()}(Scalar))")
                        basis_expr = f"{b.var_name}[0]" if len(b.shape) == 2 else b.var_name
                        body_lines.append(
                            f"{res_var} = mul_scalar({a.var_name}, {basis_expr}, {self.dtype})"
                        )

                    else:
                        raise NotImplementedError(
                            f"Inner not implemented for roles {a.role}/{b.role}, "
                            f"is_vector: {a.is_vector}/{b.is_vector}, "
                            f"is_gradient: {a.is_gradient}/{b.is_gradient}, "
                            f"shapes: {a.shape}/{b.shape}, is_hessian: {a.is_hessian}/{b.is_hessian}"
                        )
                    # Preserve the basis carrier role/metadata from the trial/test side.
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer=b, strict=False)
                    _push_inner_result(
                        default_role=b.role,
                        fallback_shape=(_basis_col_dim(b.shape),),
                        field_names=field_names,
                        parent_name=parent_name,
                        side=side,
                        field_sides=field_sides,
                    )
                    continue
                elif (a.role == 'mixed' and (a.is_gradient or len(a.shape) == 4)
                      and b.role in {'const','value'} and (b.is_gradient or len(b.shape) == 2)):
                    body_lines.append("# Inner(mixed gradient, grad(const))")
                    body_lines.append(
                        f"{res_var} = inner_mixed_grad_const({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                    _push_inner_result(
                        default_role="value",
                        fallback_shape=(a.shape[1], a.shape[2]),
                        field_names=field_names,
                        parent_name=parent_name,
                        side=side,
                        field_sides=field_sides,
                    )
                    continue
                elif (a.role in {'const','value'} and (a.is_gradient or len(a.shape) == 2)
                      and b.role == 'mixed' and (b.is_gradient or len(b.shape) == 4)):
                    body_lines.append("# Inner(grad(const), mixed gradient)")
                    body_lines.append(
                        f"{res_var} = inner_grad_const_mixed({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='b', strict=False)
                    _push_inner_result(
                        default_role="value",
                        fallback_shape=(b.shape[1], b.shape[2]),
                        field_names=field_names,
                        parent_name=parent_name,
                        side=side,
                        field_sides=field_sides,
                    )
                    continue
                elif a.role in {'test','trial'} and a.is_gradient and b.role in {'const','value'} and b.is_gradient:
                    role = a.role
                    body_lines.append("# Inner(grad(Test/Trial), grad(const))")
                    body_lines.append(
                        f"{res_var} = inner_grad_basis_grad_const({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                    _push_inner_result(
                        default_role=role,
                        fallback_shape=(a.shape[1],),
                        field_names=field_names,
                        parent_name=parent_name,
                        side=side,
                        field_sides=field_sides,
                    )
                    continue
                elif a.role in {'const','value'} and a.is_gradient and b.role in {'test','trial'} and b.is_gradient:
                    role = b.role
                    body_lines.append("# Inner(grad(const), grad(Test/Trial))")
                    body_lines.append(
                        f"{res_var} = inner_grad_function_grad_test({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='b', strict=False)
                    _push_inner_result(
                        default_role=role,
                        fallback_shape=(b.shape[1],),
                        field_names=field_names,
                        parent_name=parent_name,
                        side=side,
                        field_sides=field_sides,
                    )
                    continue

                        
                    
                elif a.role in {'value','const'} and b.role in {'value','const'}:
                    # body_lines.append(f"print(f'RHS Functional inner: a.shape: {{{a.var_name}.shape}}, "
                    #                   f"b.shape: {{{b.var_name}.shape}}, "
                    #                   f"a.role={a.role}, b.role={b.role}, ')")
                    tensor_rank_a = _try_tensor_rank(a, spatial_dim=self.spatial_dim)
                    tensor_rank_b = _try_tensor_rank(b, spatial_dim=self.spatial_dim)
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
                    elif tensor_rank_a == tensor_rank_b == 2 and not (a.is_hessian or b.is_hessian):
                        body_lines.append("# Inner(Value rank-2, Value rank-2): full contraction")
                        body_lines.append(
                            f"{res_var} = inner_full_contraction({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        shape = ()
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
                    _push_inner_result(
                        default_role="value",
                        fallback_shape=shape,
                        field_names=field_names,
                        parent_name=parent_name,
                        side=side,
                        field_sides=field_sides,
                    )
                    continue
                        
                
                else:
                    raise NotImplementedError(f"JIT Inner not implemented for roles {a.role}/{b.role}, "
                                              f" is_vector: {a.is_vector}/{b.is_vector}, " 
                                              f" is_gradient: {a.is_gradient}/{b.is_gradient}, " 
                                              f" shapes: {a.shape}/{b.shape}"
                                              f" is_hessian: {a.is_hessian}/{b.is_hessian}")



            # ------------------------------------------------------------------
            # OUTER (dyad) — vector ⊗ vector → matrix
            # ------------------------------------------------------------------
            elif isinstance(op, Outer):
                b = stack.pop()
                a = stack.pop()
                res_var = new_var("outer")
                product_lowering = _try_product_lowering(a, b)
                product_value_spec = _try_product_value_spec(a, b)
                product_meta = (
                    product_value_spec.meta
                    if product_value_spec is not None
                    else (product_lowering.meta if product_lowering is not None else None)
                )

                # scalar ⊗ anything -> scalar multiplication
                if a.shape == () and not a.is_vector and not a.is_gradient and not a.is_hessian:
                    body_lines.append("# Outer: scalar ⊗ tensor -> scalar multiplication")
                    body_lines.append(f"{res_var} = mul_scalar({a.var_name}, {b.var_name}, {self.dtype})")
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(
                        a, b, prefer="b", strict=False
                    )
                    stack.append(
                        StackItem(
                            var_name=res_var,
                            role=b.role,
                            shape=b.shape,
                            is_vector=b.is_vector,
                            is_gradient=b.is_gradient,
                            is_hessian=b.is_hessian,
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides or [],
                        )
                    )
                    continue

                if b.shape == () and not b.is_vector and not b.is_gradient and not b.is_hessian:
                    body_lines.append("# Outer: tensor ⊗ scalar -> scalar multiplication")
                    body_lines.append(f"{res_var} = mul_scalar({b.var_name}, {a.var_name}, {self.dtype})")
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(
                        a, b, prefer="a", strict=False
                    )
                    stack.append(
                        StackItem(
                            var_name=res_var,
                            role=a.role,
                            shape=a.shape,
                            is_vector=a.is_vector,
                            is_gradient=a.is_gradient,
                            is_hessian=a.is_hessian,
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides or [],
                        )
                    )
                    continue

                # Semantic rank-1 dyads
                if (
                    (_semantic_is_basis_rank1(a, spatial_dim=self.spatial_dim) or _semantic_is_value_rank1(a, spatial_dim=self.spatial_dim))
                    and (_semantic_is_basis_rank1(b, spatial_dim=self.spatial_dim) or _semantic_is_value_rank1(b, spatial_dim=self.spatial_dim))
                    and not (a.is_gradient or a.is_hessian or b.is_gradient or b.is_hessian)
                ):
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(
                        a, b, prefer="basis", strict=False
                    )
                    if (
                        _semantic_is_value_rank1(a, spatial_dim=self.spatial_dim)
                        and _semantic_is_value_rank1(b, spatial_dim=self.spatial_dim)
                    ):
                        body_lines.append("# Outer: value vector ⊗ value vector -> matrix")
                        body_lines.append(
                            f"{res_var} = vector_vector_outer_product({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role=("value" if (a.role == "value" or b.role == "value") else "const"),
                            fallback_shape=(a.shape[0], b.shape[0]),
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides or [],
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=False,
                            is_gradient=False,
                            is_hessian=False,
                            expression_meta=product_meta,
                        )
                        continue

                    if (
                        _semantic_is_value_rank1(a, spatial_dim=self.spatial_dim)
                        and _semantic_is_basis_rank1(b, spatial_dim=self.spatial_dim)
                        and b.role in {"trial", "test"}
                    ):
                        body_lines.append("# Outer: value vector ⊗ basis vector -> basis tensor")
                        body_lines.append(
                            f"{res_var} = value_vector_outer_basis_vector({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role=b.role,
                            fallback_shape=(a.shape[0], b.shape[1], b.shape[0]),
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides or [],
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=False,
                            is_gradient=True,
                            is_hessian=False,
                            expression_meta=product_meta,
                        )
                        continue

                    if (
                        _semantic_is_basis_rank1(a, spatial_dim=self.spatial_dim)
                        and a.role in {"trial", "test"}
                        and _semantic_is_value_rank1(b, spatial_dim=self.spatial_dim)
                    ):
                        body_lines.append("# Outer: basis vector ⊗ value vector -> basis tensor")
                        body_lines.append(
                            f"{res_var} = basis_vector_outer_value_vector({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role=a.role,
                            fallback_shape=(a.shape[0], a.shape[1], b.shape[0]),
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides or [],
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=False,
                            is_gradient=True,
                            is_hessian=False,
                            expression_meta=product_meta,
                        )
                        continue

                    if (
                        _semantic_is_basis_rank1(a, spatial_dim=self.spatial_dim)
                        and _semantic_is_basis_rank1(b, spatial_dim=self.spatial_dim)
                        and {a.role, b.role} == {"trial", "test"}
                    ):
                        body_lines.append("# Outer: basis vector ⊗ basis vector -> mixed tensor")
                        body_lines.append(
                            f"{res_var} = basis_vector_outer_basis_vector({a.var_name}, {b.var_name}, {str(a.role == 'test')}, {self.dtype})"
                        )
                        rows = a.shape[1] if a.role == "test" else b.shape[1]
                        cols = b.shape[1] if a.role == "test" else a.shape[1]
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role="mixed",
                            fallback_shape=(a.shape[0], rows, cols, b.shape[0]),
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides or [],
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=False,
                            is_gradient=True,
                            is_hessian=False,
                            expression_meta=product_meta,
                        )
                        continue

                raise NotImplementedError(
                    f"JIT Outer not implemented for roles {a.role}/{b.role}, "
                    f"flags a(v/g/h)={a.is_vector}/{a.is_gradient}/{a.is_hessian}, "
                    f"b(v/g/h)={b.is_vector}/{b.is_gradient}/{b.is_hessian}, "
                    f"shapes {a.shape}/{b.shape}."
                )

            # ------------------------------------------------------------------
            # DOT   — special-cased branches for advection / mass terms --------
            # ------------------------------------------------------------------
            elif isinstance(op, Dot):
                b = stack.pop()
                a = stack.pop()
                res_var = new_var("dot")
                dot_kernel = _try_dot_kernel(a, b)
                dot_lowering = dot_kernel.lowering if dot_kernel is not None else _try_dot_lowering(a, b)
                dot_meta = dot_lowering.meta if dot_lowering is not None else None
                dot_value_spec = _try_dot_value_spec(a, b)
                dot_case = dot_kernel.case if dot_kernel is not None else None

                # print(f"Dot operation: a.role={a.role}, b.role={b.role}, "
                #       f"a.shape={a.shape}, b.shape={b.shape}, "
                #       f"is_vector: {a.is_vector}/{b.is_vector}, "
                #       f"is_gradient: {a.is_gradient}/{b.is_gradient}, "
                #       f"is_hessian: {a.is_hessian}/{b.is_hessian}")

                if (
                    dot_case == DotKernelCase.BASIS_BASIS_MASS
                    and a.role in {"trial", "test"}
                    and b.role in {"trial", "test"}
                    and _semantic_is_basis_rank1(a, spatial_dim=self.spatial_dim)
                    and _semantic_is_basis_rank1(b, spatial_dim=self.spatial_dim)
                ):
                    body_lines.append("# Dot plan: basis-basis mass contraction")
                    if a.role == "trial" and b.role == "test":
                        body_lines.append(f"{res_var} = dot_mass_test_trial({b.var_name}, {a.var_name}, {self.dtype})")
                    elif a.role == "test" and b.role == "trial":
                        body_lines.append(f"{res_var} = dot_mass_test_trial({a.var_name}, {b.var_name}, {self.dtype})")
                    else:
                        raise NotImplementedError(f"Unsupported basis-basis dot orientation: {a.role}, {b.role}")
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer=None, strict=False)
                    stack.append(
                        StackItem(
                            var_name=res_var,
                            role=(dot_value_spec.role if dot_value_spec is not None else 'mixed'),
                            shape=tuple(int(v) for v in (
                                dot_value_spec.shape
                                if dot_value_spec is not None
                                else _planned_storage_shape(dot_lowering, (a.shape[-1], b.shape[-1]))
                            )),
                            is_vector=(dot_value_spec.is_vector if dot_value_spec is not None else False),
                            is_gradient=(dot_value_spec.is_gradient if dot_value_spec is not None else False),
                            is_hessian=(dot_value_spec.is_hessian if dot_value_spec is not None else False),
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            layout_tag=(dot_value_spec.layout.value if dot_value_spec is not None else ''),
                            expression_meta=dot_meta,
                        )
                    )
                    continue

                if dot_case == DotKernelCase.BASIS_GRAD_DOT_VALUE_VECTOR and a.role in {"trial", "test"}:
                    body_lines.append("# Dot plan: basis-gradient · value-vector")
                    lhs_is_rank1_grad = (
                        dot_lowering is not None and dot_lowering.algebra.lhs.tensor_rank == 1
                    )
                    if lhs_is_rank1_grad or _is_scalar_grad_basis_shape(a.shape, self.spatial_dim):
                        wants_1d = dot_value_spec is not None and len(dot_value_spec.shape) == 1
                        rank1_expr = (
                            f"contract_last_first({_rank1_basis_free_last_expr(a.var_name, a.shape)}, "
                            f"np.ravel({b.var_name}), {self.dtype})"
                        )
                        if wants_1d:
                            body_lines.append(f"{res_var} = {rank1_expr}")
                            fallback_shape = (_scalar_grad_basis_ncols(a.shape),)
                        else:
                            body_lines.append(f"{res_var} = {rank1_expr}[np.newaxis, :]")
                            fallback_shape = (1, _scalar_grad_basis_ncols(a.shape))
                    else:
                        body_lines.append(f"{res_var} = dot_grad_basis_vector({a.var_name}, {b.var_name}, {self.dtype})")
                        fallback_shape = a.shape[:2]
                    res_shape, is_vector, is_gradient = _basis_result_shape_from_dot(
                        dot_lowering,
                        a,
                        fallback_shape,
                    )
                    stack.append(
                        StackItem(
                            var_name=res_var,
                            role=a.role,
                            shape=res_shape,
                            is_vector=is_vector,
                            is_gradient=is_gradient,
                            field_names=a.field_names,
                            parent_name=a.parent_name,
                            side=a.side,
                            field_sides=a.field_sides or [],
                            expression_meta=dot_meta,
                        )
                    )
                    continue

                if dot_case == DotKernelCase.VALUE_VECTOR_DOT_BASIS_GRAD and b.role in {"trial", "test"}:
                    body_lines.append("# Dot plan: value-vector · basis-gradient")
                    rhs_is_rank1_grad = (
                        dot_lowering is not None and dot_lowering.algebra.rhs.tensor_rank == 1
                    )
                    if rhs_is_rank1_grad or _is_scalar_grad_basis_shape(b.shape, self.spatial_dim):
                        wants_1d = dot_value_spec is not None and len(dot_value_spec.shape) == 1
                        rank1_expr = (
                            f"contract_last_first({_rank1_basis_free_last_expr(b.var_name, b.shape)}, "
                            f"np.ravel({a.var_name}), {self.dtype})"
                        )
                        if wants_1d:
                            body_lines.append(f"{res_var} = {rank1_expr}")
                            fallback_shape = (_scalar_grad_basis_ncols(b.shape),)
                        else:
                            body_lines.append(f"{res_var} = {rank1_expr}[np.newaxis, :]")
                            fallback_shape = (1, _scalar_grad_basis_ncols(b.shape))
                    else:
                        body_lines.append(f"{res_var} = vector_dot_grad_basis({a.var_name}, {b.var_name}, {self.dtype})")
                        fallback_shape = (b.shape[-1], b.shape[1])
                    res_shape, is_vector, is_gradient = _basis_result_shape_from_dot(
                        dot_lowering,
                        b,
                        fallback_shape,
                    )
                    stack.append(
                        StackItem(
                            var_name=res_var,
                            role=b.role,
                            shape=res_shape,
                            is_vector=is_vector,
                            is_gradient=is_gradient,
                            field_names=b.field_names,
                            parent_name=b.parent_name,
                            side=b.side,
                            field_sides=b.field_sides or [],
                            expression_meta=dot_meta,
                        )
                    )
                    continue

                if dot_case == DotKernelCase.VALUE_GRAD_DOT_BASIS_VECTOR and b.role in {"trial", "test"}:
                    body_lines.append("# Dot plan: value-gradient · basis-vector")
                    lhs_is_rank1_grad = (
                        dot_lowering is not None and dot_lowering.algebra.lhs.tensor_rank == 1
                    )
                    if lhs_is_rank1_grad:
                        wants_1d = dot_value_spec is not None and len(dot_value_spec.shape) == 1
                        if wants_1d:
                            body_lines.append(
                                f"{res_var} = const_vector_dot_basis_1d(np.ravel({a.var_name}), {b.var_name}, {self.dtype})"
                            )
                            fallback_shape = (_basis_col_dim(b.shape),)
                        else:
                            body_lines.append(
                                f"{res_var} = basis_dot_const_vector({b.var_name}, np.ravel({a.var_name}), {self.dtype})"
                            )
                            fallback_shape = (1, _basis_col_dim(b.shape))
                    else:
                        body_lines.append(
                            f"{res_var} = contract_last_first({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        fallback_shape = a.shape[:-1] + b.shape[1:]
                    res_shape, is_vector, is_gradient = _basis_result_shape_from_dot(
                        dot_lowering,
                        b,
                        fallback_shape,
                    )
                    stack.append(
                        StackItem(
                            var_name=res_var,
                            role=b.role,
                            shape=res_shape,
                            is_vector=is_vector,
                            is_gradient=is_gradient,
                            field_names=b.field_names,
                            parent_name=b.parent_name,
                            side=b.side,
                            field_sides=b.field_sides or [],
                            expression_meta=dot_meta,
                        )
                    )
                    continue

                if dot_case == DotKernelCase.BASIS_VECTOR_DOT_VALUE_GRAD and a.role in {"trial", "test"}:
                    body_lines.append("# Dot plan: basis-vector · value-gradient")
                    rhs_is_rank1_grad = (
                        dot_lowering is not None and dot_lowering.algebra.rhs.tensor_rank == 1
                    )
                    if rhs_is_rank1_grad:
                        wants_1d = dot_value_spec is not None and len(dot_value_spec.shape) == 1
                        if wants_1d:
                            body_lines.append(
                                f"{res_var} = const_vector_dot_basis_1d(np.ravel({b.var_name}), {a.var_name}, {self.dtype})"
                            )
                            fallback_shape = (_basis_col_dim(a.shape),)
                        else:
                            body_lines.append(
                                f"{res_var} = basis_dot_const_vector({a.var_name}, np.ravel({b.var_name}), {self.dtype})"
                            )
                            fallback_shape = (1, _basis_col_dim(a.shape))
                    else:
                        body_lines.append(
                            f"{res_var} = contract_last_first(transpose_matrix({b.var_name}, {self.dtype}), {a.var_name}, {self.dtype})"
                        )
                        fallback_shape = (b.shape[1], a.shape[1])
                    res_shape, is_vector, is_gradient = _basis_result_shape_from_dot(
                        dot_lowering,
                        a,
                        fallback_shape,
                    )
                    stack.append(
                        StackItem(
                            var_name=res_var,
                            role=a.role,
                            shape=res_shape,
                            is_vector=is_vector,
                            is_gradient=is_gradient,
                            field_names=a.field_names,
                            parent_name=a.parent_name,
                            side=a.side,
                            field_sides=a.field_sides or [],
                            expression_meta=dot_meta,
                        )
                    )
                    continue

                if dot_case == DotKernelCase.BASIS_GRAD_DOT_BASIS_VECTOR and a.role in {"trial", "test"} and b.role in {"trial", "test"}:
                    body_lines.append("# Dot plan: basis-gradient · basis-vector")
                    lhs_is_rank1_grad = (
                        dot_lowering is not None and dot_lowering.algebra.lhs.tensor_rank == 1
                    )
                    if lhs_is_rank1_grad or _is_scalar_grad_basis_shape(a.shape, self.spatial_dim):
                        body_lines.append(
                            f"{res_var} = contract_last_first({_rank1_basis_free_last_expr(a.var_name, a.shape)}, {b.var_name}, {self.dtype})"
                        )
                        if dot_lowering is not None and getattr(dot_lowering, "swap_mixed_basis_axes", False):
                            body_lines.append(f"{res_var} = transpose_matrix({res_var}, {self.dtype})")
                        fallback_shape = (_scalar_grad_basis_ncols(a.shape), _basis_col_dim(b.shape))
                    else:
                        body_lines.append(
                            f"{res_var} = contract_last_first({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        if dot_lowering is not None and getattr(dot_lowering, "swap_mixed_basis_axes", False):
                            body_lines.append(f"{res_var} = swap_mixed_basis_tensor({res_var}, {self.dtype})")
                        fallback_shape = a.shape[:-1] + b.shape[1:]
                    res_shape, is_vector, is_gradient, is_hessian, layout_tag = _dot_result_flags_and_layout(
                        dot_lowering,
                        fallback_shape,
                    )
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                    stack.append(
                        StackItem(
                            var_name=res_var,
                            role='mixed',
                            shape=res_shape,
                            is_vector=is_vector,
                            is_gradient=is_gradient,
                            is_hessian=is_hessian,
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides or [],
                            layout_tag=layout_tag,
                            expression_meta=dot_meta,
                        )
                    )
                    continue

                if dot_case == DotKernelCase.BASIS_VECTOR_DOT_BASIS_GRAD and a.role in {"trial", "test"} and b.role in {"trial", "test"}:
                    body_lines.append("# Dot plan: basis-vector · basis-gradient")
                    test_var = a if a.role == "test" else b
                    trial_var = a if a.role == "trial" else b
                    if len(test_var.shape) == 2 and len(trial_var.shape) == 3:
                        body_lines.append(
                            f"{res_var} = dot_vec_grad_components({test_var.var_name}, {trial_var.var_name}, True, {self.dtype})"
                        )
                        fallback_shape = (
                            _scalar_grad_basis_ncols(trial_var.shape)
                            if _is_scalar_grad_basis_shape(trial_var.shape, self.spatial_dim)
                            else trial_var.shape[2],
                            _basis_col_dim(test_var.shape),
                            _basis_col_dim(trial_var.shape),
                        )
                        if _is_scalar_grad_basis_shape(trial_var.shape, self.spatial_dim):
                            fallback_shape = (
                                _basis_col_dim(test_var.shape),
                                _basis_col_dim(trial_var.shape),
                            )
                    else:
                        body_lines.append(
                            f"{res_var} = contract_component_first_basis({test_var.var_name}, {trial_var.var_name}, {self.dtype})"
                        )
                        lhs_tail = test_var.shape[2:] if len(test_var.shape) > 2 else ()
                        rhs_tail = trial_var.shape[2:] if len(trial_var.shape) > 2 else ()
                        if lhs_tail:
                            fallback_shape = lhs_tail + (_basis_col_dim(test_var.shape), _basis_col_dim(trial_var.shape)) + rhs_tail
                        elif rhs_tail:
                            fallback_shape = rhs_tail + (_basis_col_dim(test_var.shape), _basis_col_dim(trial_var.shape))
                        else:
                            fallback_shape = (_basis_col_dim(test_var.shape), _basis_col_dim(trial_var.shape))
                    res_shape, is_vector, is_gradient, is_hessian, layout_tag = _dot_result_flags_and_layout(
                        dot_lowering,
                        fallback_shape,
                    )
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='b', strict=False)
                    stack.append(
                        StackItem(
                            var_name=res_var,
                            role='mixed',
                            shape=res_shape,
                            is_vector=is_vector,
                            is_gradient=is_gradient,
                            is_hessian=is_hessian,
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides or [],
                            layout_tag=layout_tag,
                            expression_meta=dot_meta,
                        )
                    )
                    continue

                if (
                    a.role in {"trial", "test"}
                    and b.role in {"trial", "test"}
                    and _semantic_is_basis_rank1(a, spatial_dim=self.spatial_dim)
                    and _semantic_is_basis_rank1(b, spatial_dim=self.spatial_dim)
                    and a.is_gradient
                    and b.is_gradient
                    and not (a.is_hessian or b.is_hessian)
                ):
                    body_lines.append("# Dot: semantic scalar-gradient basis · scalar-gradient basis")
                    test_var = a if a.role == "test" else b
                    trial_var = a if a.role == "trial" else b
                    body_lines.append(
                        f"{res_var} = inner_grad_grad({test_var.var_name}, {trial_var.var_name}, {self.dtype})"
                    )
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(
                        a, b, prefer=None, strict=False
                    )
                    dot_kwargs = _dot_result_stack_kwargs(
                        dot_lowering,
                        dot_value_spec,
                        (_basis_col_dim(test_var.shape), _basis_col_dim(trial_var.shape)),
                        "mixed",
                    )
                    stack.append(
                        StackItem(
                            var_name=res_var,
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides or [],
                            expression_meta=dot_meta,
                            **dot_kwargs,
                        )
                    )
                    continue

                if (
                    a.role in {"trial", "test"}
                    and b.role in {"trial", "test"}
                    and _semantic_is_basis_rank1(a, spatial_dim=self.spatial_dim)
                    and _semantic_is_basis_rank1(b, spatial_dim=self.spatial_dim)
                    and not (a.is_gradient or b.is_gradient or a.is_hessian or b.is_hessian)
                ):
                    body_lines.append("# Dot fallback: semantic basis-vector · basis-vector mass contraction")
                    test_var = a if a.role == "test" else b
                    trial_var = a if a.role == "trial" else b
                    if len(test_var.shape) == 2 and len(trial_var.shape) == 2:
                        body_lines.append(
                            f"{res_var} = dot_mass_test_trial({test_var.var_name}, {trial_var.var_name}, {self.dtype})"
                        )
                        fallback_shape = (_basis_col_dim(test_var.shape), _basis_col_dim(trial_var.shape))
                    else:
                        body_lines.append(
                            f"{res_var} = contract_component_first_basis({test_var.var_name}, {trial_var.var_name}, {self.dtype})"
                        )
                        lhs_tail = test_var.shape[2:] if len(test_var.shape) > 2 else ()
                        rhs_tail = trial_var.shape[2:] if len(trial_var.shape) > 2 else ()
                        if lhs_tail:
                            fallback_shape = lhs_tail + (_basis_col_dim(test_var.shape), _basis_col_dim(trial_var.shape)) + rhs_tail
                        elif rhs_tail:
                            fallback_shape = rhs_tail + (_basis_col_dim(test_var.shape), _basis_col_dim(trial_var.shape))
                        else:
                            fallback_shape = (_basis_col_dim(test_var.shape), _basis_col_dim(trial_var.shape))
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer=None, strict=False)
                    stack.append(
                        StackItem(
                            var_name=res_var,
                            role=(dot_value_spec.role if dot_value_spec is not None else "mixed"),
                            shape=tuple(
                                int(v)
                                for v in (
                                    dot_value_spec.shape
                                    if dot_value_spec is not None
                                    else _planned_storage_shape(
                                        dot_lowering,
                                        fallback_shape,
                                    )
                                )
                            ),
                            is_vector=(dot_value_spec.is_vector if dot_value_spec is not None else False),
                            is_gradient=(dot_value_spec.is_gradient if dot_value_spec is not None else False),
                            is_hessian=(dot_value_spec.is_hessian if dot_value_spec is not None else False),
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides or [],
                            layout_tag=(dot_value_spec.layout.value if dot_value_spec is not None else ""),
                            expression_meta=dot_meta,
                        )
                    )
                    continue

                # Advection term: dot(grad(u_trial), u_k)
                if a.role == 'trial' and a.is_gradient and b.role == 'value' and b.is_vector:
                    body_lines.append("# Advection: dot(grad(Trial), Function)")
                    body_lines.append(
                        f"{res_var} = dot_grad_basis_vector({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    basis_dot_kwargs = _basis_dot_result_stack_kwargs(
                        dot_lowering,
                        dot_value_spec,
                        a,
                        (a.shape[0], a.shape[1]),
                        'trial',
                    )
                    stack.append(StackItem(var_name=res_var,
                                           field_names=a.field_names, parent_name=a.parent_name, side=a.side,
                                           field_sides=a.field_sides or [], expression_meta=dot_meta,
                                           **basis_dot_kwargs))
               
                # ---------------------------------------------------------------------
                # dot( grad(u_test) ,  const_vec )  ← symmetric term -> Test vec
                # ---------------------------------------------------------------------
                elif a.role == 'test' and a.is_gradient and b.role in {'const', 'value'} and b.is_vector:
                    basis_dot_kwargs = _basis_dot_result_stack_kwargs(
                        dot_lowering,
                        dot_value_spec,
                        a,
                        (a.shape[0], a.shape[1]),
                        'test',
                    )
                    if (
                        (len(a.shape) == 3 and a.shape[2] == b.shape[0])
                        or (len(a.shape) == 2 and a.shape[0] == b.shape[0])
                    ):
                        body_lines.append("# Symmetric/skew term: dot(grad(Test), (constant|value) vector)")
                        body_lines.append(
                            f"{res_var} = dot_grad_basis_vector({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                    stack.append(StackItem(var_name=res_var,
                                            field_names=a.field_names,
                                            parent_name=a.parent_name,
                                            side=a.side, field_sides=a.field_sides,
                                            expression_meta=dot_meta,
                                            **basis_dot_kwargs))

                # ---------------------------------------------------------------------
                # dot( grad(u_trial) ,  beta )  ← convection term (Function gradient · Trial)
                # ---------------------------------------------------------------------
                elif a.role == 'trial' and a.is_gradient and b.role in {'const','value'} and b.is_vector:
                    basis_dot_kwargs = _basis_dot_result_stack_kwargs(
                        dot_lowering,
                        dot_value_spec,
                        a,
                        (a.shape[0], a.shape[1]),
                        'trial',
                    )
                    if (
                        (len(a.shape) == 3 and b.shape[0] == a.shape[2])
                        or (len(a.shape) == 2 and b.shape[0] == a.shape[0])
                    ):
                        body_lines.append("# Advection: dot(grad(Trial), constant beta vector)")
                        body_lines.append(
                            f"{res_var} = dot_grad_basis_vector({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        stack.append(StackItem(var_name=res_var,
                                            field_names=a.field_names,
                                            parent_name=a.parent_name,
                                            side=a.side, field_sides=a.field_sides,
                                            expression_meta=dot_meta,
                                            **basis_dot_kwargs))
                # ---------------------------------------------------------------------
                # dot( beta, grad(u_trial)  )  ← beta_i * B_{inj} → 
                # ---------------------------------------------------------------------
                elif b.role == 'trial' and b.is_gradient and a.role in {'const','value'} and a.is_vector:
                    body_lines.append("# Advection: dot(constant beta vector, grad(Trial))")
                    body_lines.append(
                        f"{res_var} = vector_dot_grad_basis({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    fallback_shape = (
                        (1, _scalar_grad_basis_ncols(b.shape))
                        if _is_scalar_grad_basis_shape(b.shape, self.spatial_dim)
                        else ((1, b.shape[1]) if b.shape[0] == 1 else (b.shape[2], b.shape[1]))
                    )
                    basis_dot_kwargs = _basis_dot_result_stack_kwargs(
                        dot_lowering,
                        dot_value_spec,
                        b,
                        fallback_shape,
                        'trial',
                    )
                    stack.append(StackItem(var_name=res_var,
                                        field_names=b.field_names,
                                        parent_name=b.parent_name,
                                        side=b.side, field_sides=b.field_sides,
                                        expression_meta=dot_meta,
                                        **basis_dot_kwargs))
                elif (
                    a.role in {'trial', 'test'}
                    and a.is_gradient
                    and len(a.shape) == 2
                    and b.role in {'const', 'value'}
                    and len(b.shape) == 2
                    and not b.is_vector
                    and a.shape[0] == b.shape[0]
                ):
                    body_lines.append("# Dot: scalar Grad(basis) · matrix(value/const) → vector basis")
                    body_lines.append(
                        f"{res_var} = contract_last_first(transpose_matrix({b.var_name}, {self.dtype}), {a.var_name}, {self.dtype})"
                    )
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                    basis_dot_kwargs = _basis_dot_result_stack_kwargs(
                        dot_lowering,
                        dot_value_spec,
                        a,
                        (b.shape[1], a.shape[1]),
                        a.role,
                    )
                    stack.append(StackItem(
                        var_name=res_var,
                        field_names=field_names,
                        parent_name=parent_name,
                        side=side,
                        field_sides=field_sides or [],
                        expression_meta=dot_meta,
                        **basis_dot_kwargs,
                    ))


                # ---------------------------------------------------------------------
                # dot( grad(u_k) ,  u_trial )  ← convection term (Function gradient · Trial)
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_gradient and b.role == 'trial' and b.is_vector:
                    body_lines.append("# Advection: dot(grad(Function), Trial)")
                    body_lines.append(
                        f"{res_var} = dot_grad_func_trial_vec({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    basis_dot_kwargs = _basis_dot_result_stack_kwargs(
                        dot_lowering,
                        dot_value_spec,
                        b,
                        (b.shape[0], b.shape[1]),
                        'trial',
                    )
                    stack.append(StackItem(var_name=res_var,
                                        field_names=b.field_names,
                                        parent_name=b.parent_name,
                                        side=b.side, field_sides=b.field_sides,
                                        expression_meta=dot_meta,
                                        **basis_dot_kwargs))

                # ---------------------------------------------------------------------
                # dot( u_trial ,  grad(u_k) )   ← swap of the previous
                # ---------------------------------------------------------------------
                elif a.role == 'trial' and a.is_vector and b.role == 'value' and b.is_gradient:
                    dot_plan = dot_lowering.algebra if dot_lowering is not None else _try_dot_plan(a, b)
                    if dot_plan is not None and dot_plan.rhs.tensor_rank == 1:
                        body_lines.append("# Advection: dot(Trial, planner rank-1 grad(value))")
                        body_lines.append(
                            f"{res_var} = basis_dot_const_vector({a.var_name}, {_rank1_value_expr(b.var_name, b.shape)}, {self.dtype})"
                        )
                        fallback_shape = (1, a.shape[1])
                    else:
                        body_lines.append("# Advection: dot(Trial, grad(Function))")
                        body_lines.append(
                            f"{res_var} = dot_trial_vec_grad_func({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        fallback_shape = (b.shape[0], a.shape[1])
                    basis_dot_kwargs = _basis_dot_result_stack_kwargs(
                        dot_lowering,
                        dot_value_spec,
                        a,
                        fallback_shape,
                        'trial',
                    )
                    stack.append(StackItem(var_name=res_var,
                                        field_names=a.field_names,
                                        parent_name=a.parent_name,
                                        side=a.side, field_sides=a.field_sides or [],
                                        expression_meta=dot_meta,
                                        **basis_dot_kwargs))

                # ---------------------------------------------------------------------
                # dot( v_test ,  grad(u_k) )    ← test vector dotted with grad(value)
                # ---------------------------------------------------------------------
                elif a.role == 'test' and a.is_vector and b.role == 'value' and b.is_gradient and not b.is_vector:
                    dot_plan = dot_lowering.algebra if dot_lowering is not None else _try_dot_plan(a, b)
                    if dot_plan is not None and dot_plan.rhs.tensor_rank == 1:
                        body_lines.append("# Advection: dot(Test, planner rank-1 grad(value))")
                        body_lines.append(
                            f"{res_var} = basis_dot_const_vector({a.var_name}, {_rank1_value_expr(b.var_name, b.shape)}, {self.dtype})"
                        )
                        fallback_shape = (1, a.shape[1])
                    else:
                        body_lines.append("# Advection: dot(Test, grad(Function))")
                        body_lines.append(
                            f"{res_var} = contract_last_first(transpose_matrix({b.var_name}, {self.dtype}), {a.var_name}, {self.dtype})"
                        )
                        fallback_shape = (b.shape[1], a.shape[1])
                    basis_dot_kwargs = _basis_dot_result_stack_kwargs(
                        dot_lowering,
                        dot_value_spec,
                        a,
                        fallback_shape,
                        'test',
                    )
                    stack.append(StackItem(var_name=res_var,
                                        field_names=a.field_names,
                                        parent_name=a.parent_name,
                                        side=a.side, field_sides=a.field_sides or [],
                                        expression_meta=dot_meta,
                                        **basis_dot_kwargs))

                # ---------------------------------------------------------------------
                # dot( u_k ,  u_k )             ← |u_k|², scalar
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_vector and b.role == 'value' and b.is_vector:
                    body_lines.append("# Non-linear term: dot(Function, Function)")
                    body_lines.append(
                        f"{res_var} = dot_vec_vec({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    dot_kwargs = _dot_result_stack_kwargs(dot_lowering, dot_value_spec, (), 'value')
                    stack.append(StackItem(var_name=res_var,
                                        field_names=a.field_names,
                                        parent_name=a.parent_name,
                                        side=a.side,
                                        field_sides=a.field_sides or [],
                                        expression_meta=dot_meta,
                                        **dot_kwargs))

                # ---------------------------------------------------------------------
                # dot( u_k ,  grad(u_trial) )   ← usually zero for skew-symm forms
                # ---------------------------------------------------------------------
                elif a.role in {'value','const'} and a.is_vector and b.role in {'trial','test'} and b.is_gradient:
                    role_b = "trial" if b.role == "trial" else "test"
                    body_lines.append(f"# dot(Function, grad({role_b.capitalize()}))")
                    body_lines.append(
                        f"{res_var} = vector_dot_grad_basis({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    fallback_shape = (1, _scalar_grad_basis_ncols(b.shape)) if _is_scalar_grad_basis_shape(b.shape, self.spatial_dim) else (b.shape[2], b.shape[1])
                    basis_dot_kwargs = _basis_dot_result_stack_kwargs(
                        dot_lowering,
                        dot_value_spec,
                        b,
                        fallback_shape,
                        role_b,
                    )
                    stack.append(StackItem(var_name=res_var,
                                        field_names=b.field_names,
                                        parent_name=b.parent_name, side=b.side,
                                        field_sides=b.field_sides or [],
                                        expression_meta=dot_meta,
                                        **basis_dot_kwargs))

                

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
                    
                    fallback_shape = (k, n_locs, k)
                    res_shape, is_vector, is_gradient, is_hessian, layout_tag = _dot_result_flags_and_layout(
                        dot_lowering,
                        fallback_shape,
                    )
                    role_out = getattr(dot_lowering.result, "role", role) if dot_lowering is not None else role
                    stack.append(StackItem(var_name=res_var, role=role_out,
                                        shape=res_shape, is_vector=is_vector, is_gradient=is_gradient,
                                        is_hessian=is_hessian, layout_tag=layout_tag,
                                        field_names=a.field_names, parent_name=a.parent_name,
                                        side=a.side, field_sides=a.field_sides or [],
                                        expression_meta=dot_meta))
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

                    fallback_shape = (
                        (a.shape[0], b.shape[1], a.shape[1], a.shape[0])
                        if flag == 1
                        else (a.shape[0], a.shape[1], b.shape[1], a.shape[0])
                    )
                    body_lines.append(f"# dot(grad({a.role}), grad({b.role})) -> (k,m,n,k) tensor basis")

                    body_lines.append(f"# Call helper: dot_grad_grad_mixed")
                    body_lines.append(
                        f"{res_var} = dot_grad_grad_mixed({a.var_name}, {b.var_name}, {flag}, {self.dtype})"
                    )
                    res_shape, is_vector, is_gradient, is_hessian, layout_tag = _dot_result_flags_and_layout(
                        dot_lowering,
                        fallback_shape,
                    )
                    role_out = getattr(dot_lowering.result, "role", "mixed") if dot_lowering is not None else "mixed"
                    
                    stack.append(StackItem(var_name=res_var, role=role_out,
                                        shape=res_shape, is_vector=is_vector, is_gradient=is_gradient,
                                        is_hessian=is_hessian, layout_tag=layout_tag,
                                        field_names=a.field_names, parent_name=a.parent_name,
                                        side=a.side, field_sides=a.field_sides or [],
                                        expression_meta=dot_meta))

                # ---------------------------------------------------------------------
                # dot(grad(Function), grad(Trial/Test)) and its transposed variants.
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_gradient and b.role in {'trial', 'test'} and b.is_gradient:
                    role = "trial" if b.role == "trial" else "test"
                    if _is_scalar_grad_basis_shape(b.shape, self.spatial_dim):
                        body_lines.append("# Dot: matrix-like value gradient · scalar Grad(basis) -> vector basis")
                        body_lines.append(
                            f"{res_var} = contract_last_first({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        basis_dot_kwargs = _basis_dot_result_stack_kwargs(
                            dot_lowering,
                            dot_value_spec,
                            b,
                            (a.shape[0], _scalar_grad_basis_ncols(b.shape)),
                            role,
                        )
                        stack.append(StackItem(
                            var_name=res_var,
                            field_names=b.field_names,
                            parent_name=b.parent_name,
                            side=b.side,
                            field_sides=b.field_sides or [],
                            expression_meta=dot_meta,
                            **basis_dot_kwargs,
                        ))
                    elif (
                        len(b.shape) == 3
                        and len(a.shape) == 2
                        and a.shape[0] == b.shape[0]
                        and a.shape[1] == b.shape[2]
                    ):
                        k = b.shape[0]; n_locs = b.shape[1]; d = b.shape[2]

                        # a: grad(u_k) or grad(u_k).T -> Function value, shape (k, d)
                        # b: grad(du)                  -> Trial/Test basis, shape (k, n, d)
                        body_lines.append(f"# dot(grad(value), grad({role})) -> (k,n,d) tensor basis")
                        body_lines.append(
                            f"{res_var} = dot_grad_value_with_grad_basis({a.var_name}, {b.var_name}, {self.dtype})"
                        )

                        fallback_shape = (k, n_locs, d)
                        res_shape, is_vector, is_gradient, is_hessian, layout_tag = _dot_result_flags_and_layout(
                            dot_lowering,
                            fallback_shape,
                        )
                        role_out = getattr(dot_lowering.result, "role", role) if dot_lowering is not None else role
                        stack.append(StackItem(var_name=res_var, role=role_out,
                                            shape=res_shape, is_vector=is_vector, is_gradient=is_gradient,
                                            is_hessian=is_hessian, layout_tag=layout_tag,
                                            field_names=b.field_names, parent_name=b.parent_name,
                                            side=b.side, field_sides=b.field_sides or [],
                                            expression_meta=dot_meta))
                    else:
                        raise NotImplementedError(
                            f"dot(grad(value), grad({role})) unsupported for shapes {a.shape}/{b.shape}"
                        )

                # ---------------------------------------------------------------------
                # dot(grad(Function), grad(Function)) and its transposed variants.
                # ---------------------------------------------------------------------
                elif a.role in {'value', 'const'} and a.is_gradient and b.role in {'value', 'const'} and b.is_gradient:
                    dot_plan = dot_lowering.algebra if dot_lowering is not None else _try_dot_plan(a, b)
                    lhs_rank = dot_plan.lhs.tensor_rank if dot_plan is not None else _try_tensor_rank(a, spatial_dim=self.spatial_dim)
                    rhs_rank = dot_plan.rhs.tensor_rank if dot_plan is not None else _try_tensor_rank(b, spatial_dim=self.spatial_dim)
                    lhs_expr = _dot_value_operand_expr(a.var_name, a.shape, lhs_rank)
                    rhs_expr = _dot_value_operand_expr(b.var_name, b.shape, rhs_rank)
                    if dot_plan is not None and dot_plan.result.tensor_rank == 0:
                        body_lines.append("# Dot plan: value-gradient · value-gradient -> scalar contraction")
                        body_lines.append(
                            f"{res_var} = dot_vec_vec(np.ravel({lhs_expr}), np.ravel({rhs_expr}), {self.dtype})"
                        )
                        fallback_shape = ()
                    else:
                        body_lines.append("# Dot plan: value-gradient · value-gradient via semantic last-first contraction")
                        body_lines.append(
                            f"{res_var} = contract_last_first({lhs_expr}, {rhs_expr}, {self.dtype})"
                        )
                        if dot_plan is not None and dot_plan.result.tensor_rank > 0:
                            fallback_shape = tuple(int(axis.size) for axis in dot_plan.result.free_axes)
                        else:
                            fallback_shape = (a.shape[0], b.shape[1])
                    if dot_value_spec is not None:
                        res_shape = tuple(int(v) for v in dot_value_spec.shape)
                        is_vector = dot_value_spec.is_vector
                        is_gradient = dot_value_spec.is_gradient
                        is_hessian = dot_value_spec.is_hessian
                        layout_tag = dot_value_spec.layout.value
                        role_out = dot_value_spec.role
                    else:
                        res_shape, is_vector, is_gradient, is_hessian, layout_tag = _dot_result_flags_and_layout(
                            dot_lowering,
                            fallback_shape,
                        )
                        role_out = 'value'
                    stack.append(StackItem(var_name=res_var, role=role_out,
                                        shape=res_shape, is_vector=is_vector, is_gradient=is_gradient,
                                        is_hessian=is_hessian, layout_tag=layout_tag,
                                        field_names=b.field_names, parent_name=b.parent_name,
                                        side=b.side, field_sides=b.field_sides or [],
                                        expression_meta=dot_meta))

                # ---------------------new block--------------------------------
                # ---------------------------------------------------------------------
                # dot( scalar ,  u_trial;u_test;u_k )     ← e.g. scalar constant time Function
                # ---------------------------------------------------------------------
                elif (a.role == 'const' or a.role == 'value') and not a.is_vector and not a.is_gradient and not a.is_hessian and a.shape == ():
                    body_lines.append("# Dot fallback: scalar on lhs is scalar multiplication")
                    _mul_scalar_vector(self, first_is_scalar=True, a=a, b=b, res_var=res_var, body_lines=body_lines, stack=stack)
                # ---------------------------------------------------------------------
                # dot( u_trial;u_test;u_k, scalar )     ← e.g. scalar constant time Function
                # ---------------------------------------------------------------------
                elif (b.role in {'const','value'} ) and not b.is_vector and not b.is_gradient and not b.is_hessian and b.shape == ():
                    body_lines.append("# Dot fallback: scalar on rhs is scalar multiplication")
                    _mul_scalar_vector(self, first_is_scalar=False, a=a, b=b, res_var=res_var, body_lines=body_lines, stack=stack)

                # ---------------------------------------------------------------------
                # dot( u_k ,  grad(u_k) )     ← e.g. rhs advection term
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_vector and b.role == 'value' and b.is_gradient:
                    dot_plan = dot_lowering.algebra if dot_lowering is not None else _try_dot_plan(a, b)
                    if dot_plan is not None and dot_plan.result.tensor_rank == 0:
                        body_lines.append("# RHS: planner-driven scalar contraction for value vector · value gradient")
                        body_lines.append(
                            f"{res_var} = dot_vec_vec(np.ravel({a.var_name}), np.ravel({b.var_name}), {self.dtype})"
                        )
                        fallback_shape = ()
                    else:
                        body_lines.append("# RHS: planner-driven contraction for value vector · value gradient")
                        body_lines.append(
                            f"{res_var} = contract_last_first({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        if dot_plan is not None and dot_plan.result.tensor_rank > 0:
                            fallback_shape = tuple(int(axis.size) for axis in dot_plan.result.free_axes)
                        else:
                            fallback_shape = (b.shape[1],)
                    dot_kwargs = _dot_result_stack_kwargs(dot_lowering, dot_value_spec, fallback_shape, 'const')
                    stack.append(StackItem(var_name=res_var,
                                        field_names=b.field_names,
                                        parent_name=b.parent_name,
                                        side=b.side, field_sides=b.field_sides or [],
                                        expression_meta=dot_meta,
                                        **dot_kwargs))
                # ---------------------------------------------------------------------
                # dot( grad(u_k) ,  u_k )     ← e.g. rhs advection term  -> (k,d).(k) -> k
                # ---------------------------------------------------------------------
                elif a.role == 'value' and a.is_gradient and b.role == 'value' and b.is_vector:
                    dot_plan = dot_lowering.algebra if dot_lowering is not None else _try_dot_plan(a, b)
                    if dot_plan is not None and dot_plan.result.tensor_rank == 0:
                        body_lines.append("# RHS: planner-driven scalar contraction for value gradient · value vector")
                        body_lines.append(
                            f"{res_var} = dot_vec_vec(np.ravel({a.var_name}), np.ravel({b.var_name}), {self.dtype})"
                        )
                        fallback_shape = ()
                    else:
                        body_lines.append("# RHS: planner-driven contraction for value gradient · value vector")
                        body_lines.append(
                            f"{res_var} = contract_last_first({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        if dot_plan is not None and dot_plan.result.tensor_rank > 0:
                            fallback_shape = tuple(int(axis.size) for axis in dot_plan.result.free_axes)
                        else:
                            fallback_shape = (a.shape[0],)
                    dot_kwargs = _dot_result_stack_kwargs(dot_lowering, dot_value_spec, fallback_shape, 'const')
                    stack.append(StackItem(var_name=res_var,
                                        field_names=a.field_names,
                                        parent_name=a.parent_name,
                                        side=a.side, field_sides=a.field_sides or [],
                                        expression_meta=dot_meta,
                                        **dot_kwargs))
                # ---------------------------------------------------------------------
                # dot( np.array ,  u_test )     ← e.g. body-force · test -> (n,)
                # ---------------------------------------------------------------------
                elif a.role == 'const' and a.is_vector and b.role == 'test' and b.is_vector:
                    # a (k) and b (k,n)
                    body_lines.append("# Constant body-force: dot(const-vec, Test)")
                    body_lines.append(
                        f"{res_var} = const_vector_dot_basis_1d({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    stack.append(StackItem(var_name=res_var,
                                        role=b.role,
                                        shape=(b.shape[1],),
                                        is_vector=False,
                                        is_gradient=False,
                                        is_hessian=False,
                                        field_names=b.field_names,
                                        parent_name=b.parent_name,
                                        side=b.side, field_sides=b.field_sides or [],
                                        expression_meta=dot_meta,
                                        layout_tag=""))
                elif (a.role in {'trial', 'test'} and a.is_vector and not a.is_gradient and not a.is_hessian
                      and b.role in {'trial', 'test'} and b.is_gradient and not b.is_hessian):
                    test_var = a if a.role == 'test' else b
                    trial_var = a if a.role == 'trial' else b
                    if len(test_var.shape) == 2 and len(trial_var.shape) == 3:
                        if _is_scalar_grad_basis_shape(trial_var.shape, self.spatial_dim):
                            body_lines.append("# Dot: vector basis · scalar Grad(basis) → mixed scalar tensor")
                            body_lines.append(
                                f"{res_var} = dot_vec_grad_components({test_var.var_name}, {trial_var.var_name}, True, {self.dtype})"
                            )
                            res_shape = _planned_storage_shape(
                                dot_lowering,
                                (_basis_col_dim(test_var.shape), _basis_col_dim(trial_var.shape)),
                            )
                            res_layout_tag = ""
                        else:
                            body_lines.append("# Dot: vector basis · Grad(basis) → mixed tensor")
                            body_lines.append(
                                f"{res_var} = dot_vec_grad_components({test_var.var_name}, {trial_var.var_name}, True, {self.dtype})"
                            )
                            res_shape = _planned_storage_shape(
                                dot_lowering,
                                (trial_var.shape[2], _basis_col_dim(test_var.shape), _basis_col_dim(trial_var.shape)),
                            )
                            res_layout_tag = MixedLayout.COMPONENT_FIRST.value
                    else:
                        body_lines.append("# Dot: component-first basis carriers → mixed tensor")
                        body_lines.append(
                            f"{res_var} = contract_component_first_basis({test_var.var_name}, {trial_var.var_name}, {self.dtype})"
                        )
                        lhs_tail = test_var.shape[2:] if len(test_var.shape) > 2 else ()
                        rhs_tail = trial_var.shape[2:] if len(trial_var.shape) > 2 else ()
                        if lhs_tail:
                            raw_shape = lhs_tail + (_basis_col_dim(test_var.shape), _basis_col_dim(trial_var.shape)) + rhs_tail
                        elif rhs_tail:
                            raw_shape = rhs_tail + (_basis_col_dim(test_var.shape), _basis_col_dim(trial_var.shape))
                        else:
                            raw_shape = (_basis_col_dim(test_var.shape), _basis_col_dim(trial_var.shape))
                        res_shape = _planned_storage_shape(dot_lowering, raw_shape)
                        res_layout_tag = MixedLayout.COMPONENT_FIRST.value if len(res_shape) >= 3 else ""
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(
                        test_var, trial_var, prefer='a', strict=False
                    )
                    stack.append(StackItem(var_name=res_var, role='mixed',
                                        shape=res_shape, is_vector=False, is_gradient=False,
                                        field_names=field_names, parent_name=parent_name,
                                        side=side, field_sides=field_sides,
                                        layout_tag=res_layout_tag,
                                        expression_meta=dot_meta))
                # ---------------------------------------------------------------------
                # dot( grad(u_trial/test) ,  u_test/trial )          ← grad_u_mixed (k,m,n,d)
                # ---------------------------------------------------------------------
                elif (a.role in {'trial', 'test'} and a.is_gradient and not a.is_hessian
                      and b.role in {'trial', 'test'} and b.is_vector and not b.is_gradient and not b.is_hessian):
                    swap_mixed_axes = (
                        dot_lowering is not None and getattr(dot_lowering, "swap_mixed_basis_axes", False)
                    )
                    if _is_scalar_grad_basis_shape(a.shape, self.spatial_dim):
                        scalar_mixed = (
                            dot_lowering is not None
                            and dot_lowering.algebra.result.tensor_rank == 0
                        )
                        body_lines.append("# Dot: scalar Grad(basis) · vector basis → mixed scalar tensor")
                        if len(a.shape) == 3:
                            body_lines.append(
                                f"{res_var} = contract_last_first(np.ascontiguousarray({a.var_name}[0]), {b.var_name}, {self.dtype})"
                            )
                        else:
                            body_lines.append(
                                f"{res_var} = contract_last_first(np.ascontiguousarray({a.var_name}.T), {b.var_name}, {self.dtype})"
                            )
                        if swap_mixed_axes:
                            body_lines.append(
                                f"{res_var} = transpose_matrix({res_var}, {self.dtype})"
                            )
                            if scalar_mixed:
                                res_shape = _planned_storage_shape(
                                    dot_lowering,
                                    (b.shape[1], _scalar_grad_basis_ncols(a.shape)),
                                )
                            else:
                                res_shape = _planned_storage_shape(
                                    dot_lowering,
                                    (1, b.shape[1], _scalar_grad_basis_ncols(a.shape)),
                                )
                        else:
                            if scalar_mixed:
                                res_shape = _planned_storage_shape(
                                    dot_lowering,
                                    (_scalar_grad_basis_ncols(a.shape), b.shape[1]),
                                )
                            else:
                                res_shape = _planned_storage_shape(
                                    dot_lowering,
                                    (1, _scalar_grad_basis_ncols(a.shape), b.shape[1]),
                                )
                        if not scalar_mixed:
                            body_lines.append(f"{res_var} = {res_var}[np.newaxis, :, :]")
                    else:
                        body_lines.append("# Dot: Grad(basis) · vector basis → mixed tensor")
                        body_lines.append(
                            f"{res_var} = contract_last_first({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        if swap_mixed_axes:
                            body_lines.append(
                                f"{res_var} = swap_mixed_basis_tensor({res_var}, {self.dtype})"
                            )
                            res_shape = _planned_storage_shape(
                                dot_lowering,
                                (a.shape[0], b.shape[1], a.shape[1]),
                            )
                        else:
                            res_shape = _planned_storage_shape(
                                dot_lowering,
                                a.shape[:-1] + b.shape[1:],
                            )
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                    res_layout_tag = (
                        dot_lowering.result.layout.value
                        if dot_lowering is not None
                        else MixedLayout.COMPONENT_FIRST.value
                    )
                    stack.append(StackItem(var_name=res_var, role='mixed',
                                        shape=res_shape, is_vector=False, is_gradient=False,
                                        field_names=field_names, parent_name=parent_name,
                                        side=side, field_sides=field_sides,
                                        layout_tag=res_layout_tag,
                                        expression_meta=dot_meta))
                # ---------------------------------------------------------------------
                # dot( u_mixed ,  u_k )          ←  -> (m,n)
                # ---------------------------------------------------------------------
                elif (a.role == 'mixed'
                      and not a.is_gradient and not a.is_hessian
                      and len(a.shape) == 3 and b.role in {'const','value'} and b.is_vector and not b.is_gradient and not b.is_hessian):
                    body_lines.append("# Dot: mixed basis (k,m,n) · constant vector → matrix")
                    body_lines.append(
                        f"{res_var} = contract_first_first({a.var_name}, {_dot_value_operand_expr(b.var_name, b.shape, 1)}, {self.dtype})"
                    )
                    fallback_shape = (a.shape[1], a.shape[2])
                    if dot_value_spec is not None:
                        res_shape = tuple(int(v) for v in dot_value_spec.shape)
                        is_vector = dot_value_spec.is_vector
                        is_gradient = dot_value_spec.is_gradient
                        is_hessian = dot_value_spec.is_hessian
                        layout_tag = dot_value_spec.layout.value
                        role_out = dot_value_spec.role
                    else:
                        res_shape, is_vector, is_gradient, is_hessian, layout_tag = _dot_result_flags_and_layout(
                            dot_lowering,
                            fallback_shape,
                        )
                        role_out = 'value'
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                    stack.append(StackItem(var_name=res_var, role=role_out,
                                        shape=res_shape, is_vector=is_vector, is_gradient=is_gradient,
                                        is_hessian=is_hessian,
                                        field_names=field_names, parent_name=parent_name,
                                        side=side, field_sides=field_sides,
                                        layout_tag=layout_tag,
                                        expression_meta=dot_meta))
                elif (
                    _semantic_is_mixed_rank1(a, spatial_dim=self.spatial_dim)
                    and _semantic_is_value_rank1(b, spatial_dim=self.spatial_dim)
                    and b.role in {'const', 'value'}
                ):
                    body_lines.append("# Dot plan: semantic mixed rank-1 · value vector")
                    body_lines.append(
                        f"{res_var} = contract_first_first({a.var_name}, {_dot_value_operand_expr(b.var_name, b.shape, 1)}, {self.dtype})"
                    )
                    fallback_shape = (a.shape[1], a.shape[2]) if len(a.shape) == 3 else a.shape[1:]
                    if dot_value_spec is not None:
                        res_shape = tuple(int(v) for v in dot_value_spec.shape)
                        is_vector = dot_value_spec.is_vector
                        is_gradient = dot_value_spec.is_gradient
                        is_hessian = dot_value_spec.is_hessian
                        layout_tag = dot_value_spec.layout.value
                    else:
                        res_shape, is_vector, is_gradient, is_hessian, layout_tag = _dot_result_flags_and_layout(
                            dot_lowering,
                            fallback_shape,
                        )
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                    stack.append(
                        StackItem(
                            var_name=res_var,
                            role=(dot_value_spec.role if dot_value_spec is not None else 'value'),
                            shape=res_shape,
                            is_vector=is_vector,
                            is_gradient=is_gradient,
                            is_hessian=is_hessian,
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides or [],
                            layout_tag=layout_tag,
                            expression_meta=dot_meta,
                        )
                    )
                elif (
                    _semantic_is_value_rank1(a, spatial_dim=self.spatial_dim)
                    and b.role == 'mixed'
                    and len(b.shape) == 4
                    and a.role in {'const', 'value'}
                ):
                    body_lines.append("# Dot plan: value vector · mixed rank-2 tensor")
                    body_lines.append(
                        f"{res_var} = left_dot_mixed_tensor_with_vec({b.var_name}, {_dot_value_operand_expr(a.var_name, a.shape, 1)}, {self.dtype})"
                    )
                    fallback_shape = (b.shape[3], b.shape[1], b.shape[2])
                    if dot_value_spec is not None:
                        res_shape = tuple(int(v) for v in dot_value_spec.shape)
                        is_vector = dot_value_spec.is_vector
                        is_gradient = dot_value_spec.is_gradient
                        is_hessian = dot_value_spec.is_hessian
                    else:
                        res_shape, is_vector, is_gradient, is_hessian, _ = _dot_result_flags_and_layout(
                            dot_lowering,
                            fallback_shape,
                        )
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='b', strict=False)
                    stack.append(
                        StackItem(
                            var_name=res_var,
                            role=(dot_value_spec.role if dot_value_spec is not None else 'mixed'),
                            shape=res_shape,
                            is_vector=is_vector,
                            is_gradient=is_gradient,
                            is_hessian=is_hessian,
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides or [],
                            layout_tag=MixedLayout.COMPONENT_FIRST.value,
                            expression_meta=dot_meta,
                        )
                    )
                elif (
                    _semantic_is_value_rank1(a, spatial_dim=self.spatial_dim)
                    and _semantic_is_mixed_rank1(b, spatial_dim=self.spatial_dim)
                    and a.role in {'const', 'value'}
                ):
                    body_lines.append("# Dot plan: value vector · semantic mixed rank-1")
                    body_lines.append(
                        f"{res_var} = contract_first_first({b.var_name}, {_dot_value_operand_expr(a.var_name, a.shape, 1)}, {self.dtype})"
                    )
                    fallback_shape = (b.shape[1], b.shape[2]) if len(b.shape) == 3 else b.shape[1:]
                    if dot_value_spec is not None:
                        res_shape = tuple(int(v) for v in dot_value_spec.shape)
                        is_vector = dot_value_spec.is_vector
                        is_gradient = dot_value_spec.is_gradient
                        is_hessian = dot_value_spec.is_hessian
                        layout_tag = dot_value_spec.layout.value
                    else:
                        res_shape, is_vector, is_gradient, is_hessian, layout_tag = _dot_result_flags_and_layout(
                            dot_lowering,
                            fallback_shape,
                        )
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='b', strict=False)
                    stack.append(
                        StackItem(
                            var_name=res_var,
                            role=(dot_value_spec.role if dot_value_spec is not None else 'value'),
                            shape=res_shape,
                            is_vector=is_vector,
                            is_gradient=is_gradient,
                            is_hessian=is_hessian,
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides or [],
                            layout_tag=layout_tag,
                            expression_meta=dot_meta,
                        )
                    )
                elif (
                    _semantic_is_basis_rank1(a, spatial_dim=self.spatial_dim)
                    and _semantic_is_value_rank1(b, spatial_dim=self.spatial_dim)
                    and a.role in {'test', 'trial'}
                    and b.role in {'const', 'value'}
                    and len(a.shape) >= 2
                ):
                    body_lines.append("# Dot plan: semantic basis rank-1 · value vector")
                    b_dim = b.shape[0] if len(b.shape) == 1 else (b.shape[1] if len(b.shape) == 2 and b.shape[0] == 1 else (b.shape[0] if len(b.shape) == 2 and b.shape[1] == 1 else -1))
                    component_first = a.layout_tag == MixedLayout.COMPONENT_FIRST.value
                    if not component_first and b_dim not in (-1, 0):
                        component_first = a.shape[0] == b_dim
                    basis_len = a.shape[1] if component_first else a.shape[0]
                    wants_1d = dot_value_spec is not None and len(dot_value_spec.shape) == 1
                    if wants_1d:
                        body_lines.append(
                            f"{res_var} = const_vector_dot_basis_1d({b.var_name}, {a.var_name}, {self.dtype})"
                        )
                        fallback_shape = (basis_len,)
                    else:
                        body_lines.append(
                            f"{res_var} = basis_dot_const_vector({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        fallback_shape = (1, basis_len)
                    dot_kwargs = _dot_result_stack_kwargs(dot_lowering, dot_value_spec, fallback_shape, a.role)
                    stack.append(StackItem(var_name=res_var,
                                        field_names=a.field_names, parent_name=a.parent_name,
                                        side=a.side, field_sides=a.field_sides or [],
                                        expression_meta=dot_meta,
                                        **dot_kwargs))
                elif (
                    _semantic_is_value_rank1(a, spatial_dim=self.spatial_dim)
                    and _semantic_is_basis_rank1(b, spatial_dim=self.spatial_dim)
                    and a.role in {'const', 'value'}
                    and b.role in {'test', 'trial'}
                    and len(b.shape) >= 2
                ):
                    body_lines.append("# Dot plan: value vector · semantic basis rank-1")
                    a_dim = a.shape[0] if len(a.shape) == 1 else (a.shape[1] if len(a.shape) == 2 and a.shape[0] == 1 else (a.shape[0] if len(a.shape) == 2 and a.shape[1] == 1 else -1))
                    component_first = b.layout_tag == MixedLayout.COMPONENT_FIRST.value
                    if not component_first and a_dim not in (-1, 0):
                        component_first = b.shape[0] == a_dim
                    basis_len = b.shape[1] if component_first else b.shape[0]
                    wants_1d = dot_value_spec is not None and len(dot_value_spec.shape) == 1
                    if wants_1d:
                        body_lines.append(
                            f"{res_var} = const_vector_dot_basis_1d({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        fallback_shape = (basis_len,)
                    else:
                        body_lines.append(
                            f"{res_var} = basis_dot_const_vector({b.var_name}, {a.var_name}, {self.dtype})"
                        )
                        fallback_shape = (1, basis_len)
                    dot_kwargs = _dot_result_stack_kwargs(dot_lowering, dot_value_spec, fallback_shape, b.role)
                    stack.append(StackItem(var_name=res_var,
                                        field_names=b.field_names, parent_name=b.parent_name,
                                        side=b.side, field_sides=b.field_sides or [],
                                        expression_meta=dot_meta,
                                        **dot_kwargs))
                elif (
                    _is_basis_row_like(a)
                    and _is_basis_row_like(b)
                    and ((a.role, b.role) == ("test", "trial") or (a.role, b.role) == ("trial", "test"))
                ):
                    body_lines.append("# Dot plan: scalar test/trial basis rows")
                    test_var = a if a.role == "test" else b
                    trial_var = a if a.role == "trial" else b
                    fallback_shape = (_basis_col_dim(test_var.shape), _basis_col_dim(trial_var.shape))
                    body_lines.append(
                        f"{res_var} = dot_mass_test_trial({test_var.var_name}, {trial_var.var_name}, {self.dtype})"
                    )
                    dot_kwargs = _dot_result_stack_kwargs(dot_lowering, dot_value_spec, fallback_shape, "mixed")
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(
                        a, b, prefer=None, strict=False
                    )
                    stack.append(
                        StackItem(
                            var_name=res_var,
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides or [],
                            expression_meta=dot_meta,
                            **dot_kwargs,
                        )
                    )
                elif (
                    a.role in {"trial", "test"}
                    and b.role in {"trial", "test"}
                    and _semantic_is_basis_rank1(a, spatial_dim=self.spatial_dim)
                    and _semantic_is_basis_rank1(b, spatial_dim=self.spatial_dim)
                    and not (a.is_gradient or b.is_gradient or a.is_hessian or b.is_hessian)
                ):
                    body_lines.append("# Dot plan: semantic scalar basis · scalar basis")
                    test_var = a if a.role == "test" else b
                    trial_var = a if a.role == "trial" else b
                    if len(test_var.shape) == 2 and len(trial_var.shape) == 2:
                        body_lines.append(
                            f"{res_var} = dot_mass_test_trial({test_var.var_name}, {trial_var.var_name}, {self.dtype})"
                        )
                        fallback_shape = (_basis_col_dim(test_var.shape), _basis_col_dim(trial_var.shape))
                    else:
                        body_lines.append(
                            f"{res_var} = contract_component_first_basis({test_var.var_name}, {trial_var.var_name}, {self.dtype})"
                        )
                        lhs_tail = test_var.shape[2:] if len(test_var.shape) > 2 else ()
                        rhs_tail = trial_var.shape[2:] if len(trial_var.shape) > 2 else ()
                        if lhs_tail:
                            fallback_shape = lhs_tail + (_basis_col_dim(test_var.shape), _basis_col_dim(trial_var.shape)) + rhs_tail
                        elif rhs_tail:
                            fallback_shape = rhs_tail + (_basis_col_dim(test_var.shape), _basis_col_dim(trial_var.shape))
                        else:
                            fallback_shape = (_basis_col_dim(test_var.shape), _basis_col_dim(trial_var.shape))
                    dot_kwargs = _dot_result_stack_kwargs(dot_lowering, dot_value_spec, fallback_shape, "mixed")
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(
                        a, b, prefer=None, strict=False
                    )
                    stack.append(
                        StackItem(
                            var_name=res_var,
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides or [],
                            expression_meta=dot_meta,
                            **dot_kwargs,
                        )
                    )
                # ---------------------------------------------------------------------
                # dot( Hessian ,  value/const )          ← Grad object
                # ---------------------------------------------------------------------
                # --- Hessian · vector (right) and vector · Hessian (left) ---
                elif a.is_hessian and b.role in {'const','value'} and b.is_vector:
                    if a.role in ('test', 'trial'):
                        body_lines.append("# Dot: Hessian(basis) · const spatial vec -> rank-1 basis tensor")
                        body_lines.append(
                            f"{res_var} = hessian_dot_vector({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        raw_shape = (a.shape[0], a.shape[1], a.shape[2])
                    else:
                        body_lines.append("# Dot: Hessian(value) · const spatial vec -> rank-1 value tensor")
                        body_lines.append(
                            f"{res_var} = hessian_dot_vector({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        raw_shape = (a.shape[0], a.shape[1])
                    planned_shape = tuple(
                        int(v) for v in (dot_value_spec.shape if dot_value_spec is not None else _planned_storage_shape(dot_lowering, raw_shape))
                    )
                    res_shape = _emit_dot_storage_reorder(body_lines, res_var, raw_shape, planned_shape, self.dtype)
                    dot_kwargs = _dot_result_stack_kwargs(dot_lowering, dot_value_spec, res_shape, a.role)
                    stack.append(StackItem(var_name=res_var,
                                        field_names=a.field_names, parent_name=a.parent_name, side=a.side,
                                        field_sides=a.field_sides or [], expression_meta=dot_meta,
                                        **dot_kwargs))


                elif b.is_hessian and a.role in {'const','value'} and a.is_vector:
                    if b.role in ('test', 'trial'):
                        body_lines.append("# Dot: vector · Hessian(basis) -> rank-1 basis tensor")
                        body_lines.append(
                            f"{res_var} = vector_dot_hessian_basis({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        raw_shape = (b.shape[0], b.shape[1], b.shape[3])
                    else:
                        body_lines.append("# Dot: vector · Hessian(value) -> rank-1 value tensor")
                        body_lines.append(
                            f"{res_var} = vector_dot_hessian_value({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        raw_shape = (b.shape[0], b.shape[2])
                    planned_shape = tuple(
                        int(v) for v in (dot_value_spec.shape if dot_value_spec is not None else _planned_storage_shape(dot_lowering, raw_shape))
                    )
                    res_shape = _emit_dot_storage_reorder(body_lines, res_var, raw_shape, planned_shape, self.dtype)
                    dot_kwargs = _dot_result_stack_kwargs(dot_lowering, dot_value_spec, res_shape, b.role)
                    stack.append(StackItem(var_name=res_var,
                                        field_names=b.field_names, parent_name=b.parent_name, side=b.side,
                                        field_sides=b.field_sides or [], expression_meta=dot_meta,
                                        **dot_kwargs))


                
                # ---------------------------------------------------------------------
                # dot( value/const ,  value/const )          ← load-vector term -> (n,)
                # ---------------------------------------------------------------------
                elif (a.role in ('const', 'value') and   
                     b.role in ('const', 'value') ):
                    dot_plan = dot_lowering.algebra if dot_lowering is not None else _try_dot_plan(a, b)
                    if dot_plan is None:
                        raise NotImplementedError(
                            f"Dot(const/value, const/value) not implemented for shapes {a.shape}/{b.shape}"
                        )
                    if dot_plan.result.tensor_rank == 0:
                        body_lines.append("# Dot: planner-driven scalar const/value contraction")
                        body_lines.append(
                            f"{res_var} = dot_vec_vec(np.ravel({a.var_name}), np.ravel({b.var_name}), {self.dtype})"
                        )
                        fallback_shape = ()
                    else:
                        body_lines.append("# Dot: planner-driven const/value contraction")
                        body_lines.append(
                            f"{res_var} = contract_last_first({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        fallback_shape = tuple(int(axis.size) for axis in dot_plan.result.free_axes)
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                    dot_kwargs = _dot_result_stack_kwargs(dot_lowering, dot_value_spec, fallback_shape, 'const')
                    stack.append(StackItem(var_name=res_var,
                                        field_names=field_names,
                                        parent_name=parent_name, side=side,
                                        field_sides=field_sides,
                                        expression_meta=dot_meta,
                                        **dot_kwargs))
                elif (len(a.shape) >= 1 and len(b.shape) >= 1
                      and not a.is_hessian and not b.is_hessian
                      and a.shape[-1] == b.shape[0]):
                    
                    body_lines.append("# Dot: generic contraction (last axis of A · first axis of B)")
                    body_lines.append(
                        f"{res_var} = contract_last_first({a.var_name}, {b.var_name}, {self.dtype})"
                    )
                    dot_lowering = _try_dot_lowering(a, b)
                    raw_shape = a.shape[:-1] + b.shape[1:]
                    planned_shape = tuple(int(v) for v in (dot_value_spec.shape if dot_value_spec is not None else _planned_storage_shape(dot_lowering, raw_shape)))
                    res_shape = _emit_dot_storage_reorder(body_lines, res_var, raw_shape, planned_shape, self.dtype)
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                    default_role = 'mixed' if 'mixed' in (a.role, b.role) else (a.role if a.role in {'trial', 'test'} else (b.role if b.role in {'trial', 'test'} else 'value'))
                    dot_kwargs = _dot_result_stack_kwargs(dot_lowering, dot_value_spec, res_shape, default_role)
                    stack.append(StackItem(var_name=res_var,
                                        field_names=field_names, parent_name=parent_name,
                                        side=side, field_sides=field_sides,
                                        expression_meta=dot_meta,
                                        **dot_kwargs))
                elif (
                    a.role in {"const", "value"}
                    and a.is_gradient
                    and len(a.shape) == 2
                    and b.is_gradient
                    and len(b.shape) in {2, 3}
                    and b.shape[0] == 1
                    and a.shape[1] == b.shape[-1]
                ):
                    body_lines.append("# Dot: matrix(value/const) · scalar-gradient tensor")
                    body_lines.append(
                        f"{res_var} = contract_last_first({b.var_name}, np.ascontiguousarray({a.var_name}.T), {self.dtype})"
                    )
                    field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='b', strict=False)
                    if len(b.shape) == 3:
                        fallback_shape = (1, b.shape[1], a.shape[0])
                    else:
                        fallback_shape = (1, a.shape[0])
                    res_shape = _planned_storage_shape(dot_lowering, fallback_shape)
                    stack.append(StackItem(var_name=res_var, role=b.role,
                                        shape=res_shape,
                                        is_vector=False,
                                        is_gradient=True,
                                        is_hessian=False,
                                        field_names=field_names, parent_name=parent_name,
                                        side=side, field_sides=field_sides))
                else:
                    # (Keep your debug print here, it's very useful)
                    print("Dot failure debug:",
                          {"a_role": a.role, "a_shape": a.shape, "a_grad": a.is_gradient,
                           "a_vec": a.is_vector, "a_hess": a.is_hessian, "a_fields": getattr(a, 'field_names', None),
                           "b_role": b.role, "b_shape": b.shape, "b_grad": b.is_gradient,
                           "b_vec": b.is_vector, "b_hess": b.is_hessian, "b_fields": getattr(b, 'field_names', None)})
                    raise NotImplementedError(f"Dot not implemented for roles {a.role}/{b.role} with shapes {a.shape}/{b.shape}"
                                              f" with vectoors {a.is_vector}/{b.is_vector}"
                                              f" and gradients {a.is_gradient}/{b.is_gradient}"
                                              f" also with hessians {a.is_hessian}/{b.is_hessian}"
                                              f", BilinearForm" if self.form_rank == 2 else ", LinearForm")

            elif isinstance(op, Transpose):
                a = stack.pop()                   # operand descriptor
                res = new_var("trp")
                tensor_rank = _try_tensor_rank(a, spatial_dim=self.spatial_dim)
                transpose_meta = _try_transpose_meta(a, spatial_dim=self.spatial_dim)

                # ---------------------------------------------------------------
                # 0) scalar  →  transpose is a no-op
                # ---------------------------------------------------------------
                if a.shape == () and tensor_rank in {None, 0}:
                    body_lines.append(f"{res} = {a.var_name}")     # just copy
                    res_shape = ()

                elif tensor_rank not in {None, 2}:
                    raise NotImplementedError(
                        f"Transpose is only defined for rank-2 tensors; got shape {a.shape} "
                        f"with semantic tensor rank {tensor_rank}."
                    )

                # -------- semantic rank-2 basis carrier: (k,n,d)  swap free axes -------
                elif len(a.shape) == 3 and tensor_rank == 2:
                    body_lines.append(
                        f"{res} = transpose_grad_tensor({a.var_name}, {self.dtype})"
                    )
                    res_shape = a.shape  # still (2,n,2)
                elif a.role == 'mixed' and len(a.shape) == 4 and tensor_rank == 2:
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
                elif len(a.shape) == 2:
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
                                        role=a.role,
                                        expression_meta=transpose_meta))

            
            elif isinstance(op, BinaryOp):
                 b = stack.pop(); a = stack.pop()
                 res_var = new_var("res")
                 # -------------------------------------
                 # ------------ PRODUCT ---------------
                 # -------------------------------------
                 if op.op_symbol == '*':
                    product_lowering = _try_product_lowering(a, b)
                    product_kernel = _try_product_kernel(a, b)
                    product_meta = product_lowering.meta if product_lowering is not None else None
                    product_value_spec = _try_product_value_spec(a, b)
                    product_case = product_kernel.case if product_kernel is not None else None
                    # print(f"[*] shapes: ({a.shape}, {b.shape}) roles: ({a.role}, {b.role})")
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
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role="const",
                            fallback_shape=shape,
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=False,
                            is_gradient=False,
                            is_hessian=False,
                        )

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
                    elif (_is_basis_row_like(a) and _is_basis_row_like(b) and
                        ((a.role, b.role) == ("test", "trial") or
                        (a.role, b.role) == ("trial", "test"))):
                        body_lines.append("# Product: scalar Test × scalar Trial → mixed scalar basis tensor")

                        # orient rows = test , columns = trial. This is a true
                        # mixed scalar carrier, so do not keep any fake leading
                        # singleton tensor axis from earlier gradient-component
                        # metadata.
                        test_var  = a if a.role == "test"  else b
                        trial_var = b if a.role == "test"  else a
                        n_test = _basis_col_dim(test_var.shape)
                        n_trial = _basis_col_dim(trial_var.shape)
                        planned_shape = (n_test, n_trial)

                        body_lines.append(
                            f"{res_var} = dot_mass_test_trial({test_var.var_name}, {trial_var.var_name}, {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer=None, strict=False)
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role="mixed",
                            fallback_shape=planned_shape,
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=False,
                            is_gradient=False,
                            is_hessian=False,
                            expression_meta=product_meta,
                        )

                    # -----------------------------------------------------------------
                    # 2. LHS block:   scalar trial/test  *  vector   →  vector trial/test
                    # -----------------------------------------------------------------
                    elif (
                        (
                            product_case == ProductKernelCase.BASIS_SCALAR_TIMES_VALUE_VECTOR
                            and a.role in {"trial", "test"}
                            and (
                                len(a.shape) == 1
                                or (len(a.shape) >= 2 and a.shape[0] == 1)
                            )
                            and b.role in {"value", "const"}
                            and b.is_vector
                        )
                        or (
                            a.role in {"trial", "test"}
                            and not a.is_vector
                            and not a.is_gradient
                            and not a.is_hessian
                            and (
                                len(a.shape) == 1
                                or (len(a.shape) >= 2 and a.shape[0] == 1)
                            )
                            and b.role in {"value", "const"}
                            and b.is_vector
                        )
                    ):
                        role = product_lowering.result.role if product_lowering is not None else ('trial' if a.role == 'trial' else 'test')
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(
                            a, b, prefer="basis", strict=False
                        )
                        basis_expr = _scalar_basis_values_expr(a.var_name, a.shape)
                        if product_lowering is not None and product_lowering.result.is_gradient:
                            body_lines.append("# Product: scalar Trial/Test × vector → grad-tensor via shared lowering")
                            body_lines.append(
                                f"{res_var} = scalar_basis_times_vector_as_grad_tensor({basis_expr}, {b.var_name}, {self.dtype})"
                            )
                            res_shape = _planned_storage_shape(product_lowering, (b.shape[0], _basis_col_dim(a.shape)))
                            res_is_vector = False
                            res_is_gradient = True
                        else:
                            body_lines.append("# Product: scalar Trial/Test × vector → vector Trial/Test")
                            body_lines.append(
                                f"{res_var} = scalar_basis_times_vector({basis_expr}, {b.var_name}, {self.dtype})"
                            )
                            res_shape = _planned_storage_shape(product_lowering, (b.shape[0], _basis_col_dim(a.shape)))
                            res_is_vector = True
                            res_is_gradient = False

                        stack.append(StackItem(var_name=res_var, role=role,
                                            shape=res_shape, is_vector=res_is_vector, is_gradient=res_is_gradient,
                                            field_names=field_names, parent_name=parent_name,
                                            side=side, field_sides=field_sides or [],
                                            expression_meta=product_meta))
                    elif (
                        (
                            product_case == ProductKernelCase.VALUE_VECTOR_TIMES_BASIS_SCALAR
                            and b.role in {"trial", "test"}
                            and (
                                len(b.shape) == 1
                                or (len(b.shape) >= 2 and b.shape[0] == 1)
                            )
                            and a.role in {"value", "const"}
                            and a.is_vector
                        )
                        or (
                            b.role in {"trial", "test"}
                            and not b.is_vector
                            and not b.is_gradient
                            and not b.is_hessian
                            and (
                                len(b.shape) == 1
                                or (len(b.shape) >= 2 and b.shape[0] == 1)
                            )
                            and a.role in {"value", "const"}
                            and a.is_vector
                        )
                    ):
                        role = product_lowering.result.role if product_lowering is not None else ('trial' if b.role == 'trial' else 'test')
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(
                            a, b, prefer="basis", strict=False
                        )
                        basis_expr = _scalar_basis_values_expr(b.var_name, b.shape)
                        if product_lowering is not None and product_lowering.result.is_gradient:
                            body_lines.append("# Product: vector × scalar Trial/Test → grad-tensor via shared lowering")
                            body_lines.append(
                                f"{res_var} = scalar_basis_times_vector_as_grad_tensor({basis_expr}, {a.var_name}, {self.dtype})"
                            )
                            res_shape = _planned_storage_shape(product_lowering, (a.shape[0], _basis_col_dim(b.shape)))
                            res_is_vector = False
                            res_is_gradient = True
                        else:
                            body_lines.append("# Product: vector × scalar Trial/Test → vector Trial/Test")
                            body_lines.append(
                                f"{res_var} = scalar_basis_times_vector({basis_expr}, {a.var_name}, {self.dtype})"
                            )
                            res_shape = _planned_storage_shape(product_lowering, (a.shape[0], _basis_col_dim(b.shape)))
                            res_is_vector = True
                            res_is_gradient = False
                        stack.append(StackItem(var_name=res_var, role=role,
                                            shape=res_shape, is_vector=res_is_vector, is_gradient=res_is_gradient,
                                            field_names=field_names, parent_name=parent_name,
                                            side=side, field_sides=field_sides or [],
                                            expression_meta=product_meta))
                    elif (
                        (
                            product_case == ProductKernelCase.VALUE_MATRIX_TIMES_BASIS_SCALAR
                            and a.role in {"value", "const"}
                            and len(a.shape) == 2
                            and b.role in {"trial", "test"}
                            and (
                                (len(b.shape) == 2 and b.shape[0] == 1)
                                or len(b.shape) == 1
                            )
                        )
                        or (
                            a.role in {"value", "const"}
                            and not a.is_vector
                            and not a.is_gradient
                            and not a.is_hessian
                            and len(a.shape) == 2
                            and b.role in {"trial", "test"}
                            and not b.is_vector
                            and not b.is_gradient
                            and not b.is_hessian
                            and (
                                (len(b.shape) == 2 and b.shape[0] == 1)
                                or len(b.shape) == 1
                            )
                        )
                    ):
                        role = product_lowering.result.role if product_lowering is not None else b.role
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                        basis_expr = _scalar_basis_values_expr(b.var_name, b.shape)
                        planned_shape = tuple(int(v) for v in (
                            product_value_spec.shape
                            if product_value_spec is not None
                            else _planned_storage_shape(product_lowering, (a.shape[0], _basis_col_dim(b.shape), a.shape[1]))
                        ))
                        if len(planned_shape) == 2:
                            body_lines.append("# Product: matrix-like rank-1 Value × scalar Trial/Test → rank-1 basis matrix")
                            body_lines.append(
                                f"{res_var} = np.ascontiguousarray(np.reshape({a.var_name}, (-1, 1)) * {basis_expr}[np.newaxis, :])"
                            )
                            res_shape = planned_shape
                            res_is_gradient = bool(product_value_spec.is_gradient) if product_value_spec is not None else bool(product_lowering.result.is_gradient if product_lowering is not None else False)
                        else:
                            body_lines.append("# Product: matrix Value × scalar Trial/Test → tensor basis via shared lowering")
                            body_lines.append(
                                f"{res_var} = value_matrix_times_scalar_basis_tensor({a.var_name}, {basis_expr}, {self.dtype})"
                            )
                            res_shape = planned_shape
                            res_is_gradient = True
                        stack.append(StackItem(var_name=res_var, role=role,
                                            shape=res_shape, is_vector=False, is_gradient=res_is_gradient,
                                            field_names=field_names, parent_name=parent_name,
                                            side=side, field_sides=field_sides,
                                            expression_meta=product_meta))
                    elif (
                        (
                            product_case == ProductKernelCase.BASIS_SCALAR_TIMES_VALUE_MATRIX
                            and b.role in {"value", "const"}
                            and len(b.shape) == 2
                            and a.role in {"trial", "test"}
                            and (
                                (len(a.shape) == 2 and a.shape[0] == 1)
                                or len(a.shape) == 1
                            )
                        )
                        or (
                            b.role in {"value", "const"}
                            and not b.is_vector
                            and not b.is_gradient
                            and not b.is_hessian
                            and len(b.shape) == 2
                            and a.role in {"trial", "test"}
                            and not a.is_vector
                            and not a.is_gradient
                            and not a.is_hessian
                            and (
                                (len(a.shape) == 2 and a.shape[0] == 1)
                                or len(a.shape) == 1
                            )
                        )
                    ):
                        role = product_lowering.result.role if product_lowering is not None else a.role
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='b', strict=False)
                        basis_expr = _scalar_basis_values_expr(a.var_name, a.shape)
                        planned_shape = tuple(int(v) for v in (
                            product_value_spec.shape
                            if product_value_spec is not None
                            else _planned_storage_shape(product_lowering, (b.shape[0], _basis_col_dim(a.shape), b.shape[1]))
                        ))
                        if len(planned_shape) == 2:
                            body_lines.append("# Product: scalar Trial/Test × matrix-like rank-1 Value → rank-1 basis matrix")
                            body_lines.append(
                                f"{res_var} = np.ascontiguousarray(np.reshape({b.var_name}, (-1, 1)) * {basis_expr}[np.newaxis, :])"
                            )
                            res_shape = planned_shape
                            res_is_gradient = bool(product_value_spec.is_gradient) if product_value_spec is not None else bool(product_lowering.result.is_gradient if product_lowering is not None else False)
                        else:
                            body_lines.append("# Product: scalar Trial/Test × matrix Value → tensor basis via shared lowering")
                            body_lines.append(
                                f"{res_var} = value_matrix_times_scalar_basis_tensor({b.var_name}, {basis_expr}, {self.dtype})"
                            )
                            res_shape = planned_shape
                            res_is_gradient = True
                        stack.append(StackItem(var_name=res_var, role=role,
                                            shape=res_shape, is_vector=False, is_gradient=res_is_gradient,
                                            field_names=field_names, parent_name=parent_name,
                                            side=side, field_sides=field_sides,
                                            expression_meta=product_meta))
                    elif (
                        (
                            product_case == ProductKernelCase.MIXED_SCALAR_TIMES_VALUE_MATRIX
                            and a.role == 'mixed'
                            and not a.is_vector
                            and not a.is_gradient
                            and not a.is_hessian
                            and len(a.shape) in {2, 3}
                            and (
                                len(a.shape) == 2
                                or (len(a.shape) == 3 and a.shape[0] == 1)
                            )
                            and b.role in {'value', 'const'}
                            and len(b.shape) == 2
                        )
                        or (
                            a.role == 'mixed'
                            and not a.is_vector
                            and not a.is_gradient
                            and not a.is_hessian
                            and len(a.shape) in {2, 3}
                            and (
                                len(a.shape) == 2
                                or (len(a.shape) == 3 and a.shape[0] == 1)
                            )
                            and b.role in {'value', 'const'}
                            and len(b.shape) == 2
                            and product_lowering is not None
                            and product_lowering.algebra.kind.name in {'PRODUCT_SCALE', 'PRODUCT_PROMOTE'}
                            and product_lowering.algebra.result.basis_rank == 2
                            and product_lowering.algebra.result.tensor_rank == 2
                        )
                    ):
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                        role_out = product_value_spec.role if product_value_spec is not None else 'mixed'
                        layout_tag = product_value_spec.layout.value if product_value_spec is not None else ''
                        if len(a.shape) == 3:
                            body_lines.append("# Product: scalar mixed basis × matrix(value/const) → mixed tensor")
                            body_lines.append(
                                f"{res_var} = identity_times_trace_matrix({b.var_name}, {a.var_name}[0], {self.dtype})"
                            )
                            res_shape = _planned_storage_shape(product_lowering, (b.shape[0], a.shape[1], a.shape[2], b.shape[1]))
                        else:
                            body_lines.append("# Product: scalar mixed carrier × matrix(value/const) → mixed tensor")
                            body_lines.append(
                                f"{res_var} = identity_times_trace_matrix({b.var_name}, {a.var_name}, {self.dtype})"
                            )
                            res_shape = _planned_storage_shape(product_lowering, (b.shape[0], a.shape[0], a.shape[1], b.shape[1]))
                        stack.append(StackItem(var_name=res_var, role=role_out,
                                            shape=res_shape, is_vector=False,
                                            is_gradient=True, is_hessian=False,
                                            field_names=field_names, parent_name=parent_name,
                                            side=side, field_sides=field_sides,
                                            layout_tag=layout_tag,
                                            expression_meta=product_meta))
                    elif (
                        (
                            product_case == ProductKernelCase.VALUE_MATRIX_TIMES_MIXED_SCALAR
                            and b.role == 'mixed'
                            and not b.is_vector
                            and not b.is_gradient
                            and not b.is_hessian
                            and len(b.shape) in {2, 3}
                            and (
                                len(b.shape) == 2
                                or (len(b.shape) == 3 and b.shape[0] == 1)
                            )
                            and a.role in {'value', 'const'}
                            and len(a.shape) == 2
                        )
                        or (
                            b.role == 'mixed'
                            and not b.is_vector
                            and not b.is_gradient
                            and not b.is_hessian
                            and len(b.shape) in {2, 3}
                            and (
                                len(b.shape) == 2
                                or (len(b.shape) == 3 and b.shape[0] == 1)
                            )
                            and a.role in {'value', 'const'}
                            and len(a.shape) == 2
                            and product_lowering is not None
                            and product_lowering.algebra.kind.name in {'PRODUCT_SCALE', 'PRODUCT_PROMOTE'}
                            and product_lowering.algebra.result.basis_rank == 2
                            and product_lowering.algebra.result.tensor_rank == 2
                        )
                    ):
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='b', strict=False)
                        role_out = product_value_spec.role if product_value_spec is not None else 'mixed'
                        layout_tag = product_value_spec.layout.value if product_value_spec is not None else ''
                        if len(b.shape) == 3:
                            body_lines.append("# Product: matrix(value/const) × scalar mixed basis → mixed tensor")
                            body_lines.append(
                                f"{res_var} = identity_times_trace_matrix({a.var_name}, {b.var_name}[0], {self.dtype})"
                            )
                            res_shape = _planned_storage_shape(product_lowering, (a.shape[0], b.shape[1], b.shape[2], a.shape[1]))
                        else:
                            body_lines.append("# Product: matrix(value/const) × scalar mixed carrier → mixed tensor")
                            body_lines.append(
                                f"{res_var} = identity_times_trace_matrix({a.var_name}, {b.var_name}, {self.dtype})"
                            )
                            res_shape = _planned_storage_shape(product_lowering, (a.shape[0], b.shape[0], b.shape[1], a.shape[1]))
                        stack.append(StackItem(var_name=res_var, role=role_out,
                                            shape=res_shape, is_vector=False,
                                            is_gradient=True, is_hessian=False,
                                            field_names=field_names, parent_name=parent_name,
                                            side=side, field_sides=field_sides,
                                            layout_tag=layout_tag,
                                            expression_meta=product_meta))
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
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role="mixed",
                            fallback_shape=(b.shape[0], b.shape[1], a.shape[1], b.shape[2]),
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=False,
                            is_gradient=True,
                            is_hessian=False,
                            expression_meta=product_meta,
                        )
                    elif (a.role == 'test' and not a.is_vector and not a.is_gradient and not a.is_hessian and len(a.shape) == 2 and a.shape[0] == 1
                          and b.role == 'trial' and b.is_gradient and len(b.shape) == 3):
                        body_lines.append("# Product: scalar Test × Grad(Trial) → mixed gradient")
                        body_lines.append(
                            f"{res_var} = grad_trial_times_scalar_test({b.var_name}, {a.var_name}[0], {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role="mixed",
                            fallback_shape=(b.shape[0], a.shape[1], b.shape[1], b.shape[2]),
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=False,
                            is_gradient=True,
                            is_hessian=False,
                            expression_meta=product_meta,
                        )
                    elif (a.role == 'test' and a.is_gradient and len(a.shape) == 3
                          and b.role == 'trial' and not b.is_vector and not b.is_gradient and not b.is_hessian and len(b.shape) == 2 and b.shape[0] == 1):
                        body_lines.append("# Product: Grad(Test) × scalar Trial → mixed gradient")
                        body_lines.append(
                            f"{res_var} = scalar_trial_times_grad_test({a.var_name}, {b.var_name}[0], {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role="mixed",
                            fallback_shape=(a.shape[0], a.shape[1], b.shape[1], a.shape[2]),
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=False,
                            is_gradient=True,
                            is_hessian=False,
                            expression_meta=product_meta,
                        )
                    elif (a.role == 'trial' and a.is_gradient and len(a.shape) == 3
                          and b.role == 'test' and not b.is_vector and not b.is_gradient and not b.is_hessian and len(b.shape) == 2 and b.shape[0] == 1):
                        body_lines.append("# Product: Grad(Trial) × scalar Test → mixed gradient")
                        body_lines.append(
                            f"{res_var} = grad_trial_times_scalar_test({a.var_name}, {b.var_name}[0], {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='b', strict=False)
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role="mixed",
                            fallback_shape=(a.shape[0], b.shape[1], a.shape[1], a.shape[2]),
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=False,
                            is_gradient=True,
                            is_hessian=False,
                            expression_meta=product_meta,
                        )
                    elif (
                        a.role == "trial"
                        and _is_basis_row_like(a)
                        and not a.is_vector and not a.is_gradient and not a.is_hessian
                        and b.role == "test"
                        and _semantic_is_basis_rank2(b, spatial_dim=self.spatial_dim)
                        and not b.is_gradient and not b.is_hessian
                    ):
                        body_lines.append("# Product: scalar Trial × rank-2 Test basis → mixed rank-2 tensor")
                        scalar_expr = a.var_name if len(a.shape) == 1 else f"{a.var_name}[0]"
                        body_lines.append(
                            f"{res_var} = scalar_trial_times_basis_test({b.var_name}, {scalar_expr}, {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='b', strict=False)
                        stack.append(StackItem(
                            var_name=res_var,
                            role="mixed",
                            shape=(b.shape[0], b.shape[1], _basis_col_dim(a.shape), b.shape[2]),
                            is_vector=False,
                            is_gradient=False,
                            is_hessian=False,
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            layout_tag=MixedLayout.DEFAULT.value,
                            expression_meta=product_meta,
                        ))
                    elif (
                        a.role == "test"
                        and _is_basis_row_like(a)
                        and not a.is_vector and not a.is_gradient and not a.is_hessian
                        and b.role == "trial"
                        and _semantic_is_basis_rank2(b, spatial_dim=self.spatial_dim)
                        and not b.is_gradient and not b.is_hessian
                    ):
                        body_lines.append("# Product: scalar Test × rank-2 Trial basis → mixed rank-2 tensor")
                        scalar_expr = a.var_name if len(a.shape) == 1 else f"{a.var_name}[0]"
                        body_lines.append(
                            f"{res_var} = basis_trial_times_scalar_test({b.var_name}, {scalar_expr}, {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                        stack.append(StackItem(
                            var_name=res_var,
                            role="mixed",
                            shape=(b.shape[0], _basis_col_dim(a.shape), b.shape[1], b.shape[2]),
                            is_vector=False,
                            is_gradient=False,
                            is_hessian=False,
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            layout_tag=MixedLayout.DEFAULT.value,
                            expression_meta=product_meta,
                        ))
                    elif (
                        a.role == "test"
                        and _semantic_is_basis_rank2(a, spatial_dim=self.spatial_dim)
                        and not a.is_gradient and not a.is_hessian
                        and b.role == "trial"
                        and _is_basis_row_like(b)
                        and not b.is_vector and not b.is_gradient and not b.is_hessian
                    ):
                        body_lines.append("# Product: rank-2 Test basis × scalar Trial → mixed rank-2 tensor")
                        scalar_expr = b.var_name if len(b.shape) == 1 else f"{b.var_name}[0]"
                        body_lines.append(
                            f"{res_var} = scalar_trial_times_basis_test({a.var_name}, {scalar_expr}, {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                        stack.append(StackItem(
                            var_name=res_var,
                            role="mixed",
                            shape=(a.shape[0], a.shape[1], _basis_col_dim(b.shape), a.shape[2]),
                            is_vector=False,
                            is_gradient=False,
                            is_hessian=False,
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            layout_tag=MixedLayout.DEFAULT.value,
                            expression_meta=product_meta,
                        ))
                    elif (
                        a.role == "trial"
                        and _semantic_is_basis_rank2(a, spatial_dim=self.spatial_dim)
                        and not a.is_gradient and not a.is_hessian
                        and b.role == "test"
                        and _is_basis_row_like(b)
                        and not b.is_vector and not b.is_gradient and not b.is_hessian
                    ):
                        body_lines.append("# Product: rank-2 Trial basis × scalar Test → mixed rank-2 tensor")
                        scalar_expr = b.var_name if len(b.shape) == 1 else f"{b.var_name}[0]"
                        body_lines.append(
                            f"{res_var} = basis_trial_times_scalar_test({a.var_name}, {scalar_expr}, {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='b', strict=False)
                        stack.append(StackItem(
                            var_name=res_var,
                            role="mixed",
                            shape=(a.shape[0], _basis_col_dim(b.shape), a.shape[1], a.shape[2]),
                            is_vector=False,
                            is_gradient=False,
                            is_hessian=False,
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            layout_tag=MixedLayout.DEFAULT.value,
                            expression_meta=product_meta,
                        ))
                    elif (a.role in {"trial", "test"} and not a.is_vector and not a.is_gradient and not a.is_hessian and len(a.shape) == 1
                          and b.role in {"value", "const"} and b.is_vector):
                        role = 'trial' if a.role == 'trial' else 'test'
                        body_lines.append("# Product: scalar Trial/Test × vector → vector Trial/Test")
                        body_lines.append(
                            f"{res_var} = scalar_vector_outer_product({a.var_name}, {b.var_name}, {self.dtype})"
                        )

                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role=role,
                            fallback_shape=(b.shape[0], a.shape[0]),
                            field_names=a.field_names,
                            parent_name=a.parent_name,
                            side=a.side,
                            field_sides=a.field_sides,
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=True,
                            is_gradient=False,
                            is_hessian=False,
                            expression_meta=product_meta,
                        )
                    elif (
                        a.role == "trial"
                        and a.is_vector
                        and not a.is_gradient
                        and not a.is_hessian
                        and len(a.shape) == 2
                        and b.role == "test"
                        and not b.is_vector
                        and not b.is_gradient
                        and not b.is_hessian
                        and (len(b.shape) == 1 or (len(b.shape) == 2 and b.shape[0] == 1))
                    ):
                        body_lines.append("# Product: vector Trial × scalar Test → mixed vector basis")
                        scalar_expr = b.var_name if len(b.shape) == 1 else f"{b.var_name}"
                        body_lines.append(
                            f"{res_var} = vector_trial_times_scalar_test({a.var_name}, {scalar_expr}, {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='basis', strict=False)
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role="mixed",
                            fallback_shape=(a.shape[0], _basis_col_dim(b.shape), a.shape[1]),
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=False,
                            is_gradient=False,
                            is_hessian=False,
                            layout_tag=MixedLayout.COMPONENT_FIRST.value,
                            expression_meta=product_meta,
                        )
                    elif (
                        a.role == "test"
                        and a.is_vector
                        and not a.is_gradient
                        and not a.is_hessian
                        and len(a.shape) == 2
                        and b.role == "trial"
                        and not b.is_vector
                        and not b.is_gradient
                        and not b.is_hessian
                        and (len(b.shape) == 1 or (len(b.shape) == 2 and b.shape[0] == 1))
                    ):
                        body_lines.append("# Product: vector Test × scalar Trial → mixed vector basis")
                        scalar_expr = b.var_name if len(b.shape) == 1 else f"{b.var_name}"
                        body_lines.append(
                            f"{res_var} = vector_test_times_scalar_trial({a.var_name}, {scalar_expr}, {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='basis', strict=False)
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role="mixed",
                            fallback_shape=(a.shape[0], a.shape[1], _basis_col_dim(b.shape)),
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=False,
                            is_gradient=False,
                            is_hessian=False,
                            layout_tag=MixedLayout.COMPONENT_FIRST.value,
                            expression_meta=product_meta,
                        )
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
                                            side=side, field_sides=field_sides,
                                            expression_meta=product_meta))
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
                                            side=side, field_sides=field_sides,
                                            expression_meta=product_meta))
                    elif (
                        a.role in {"trial", "test"}
                        and a.is_gradient
                        and len(a.shape) == 3
                        and b.role in {"value", "const"}
                        and not b.is_vector
                        and len(b.shape) == 2
                        and a.shape[2] == b.shape[0]
                    ):
                        body_lines.append("# Product: Grad(basis) × matrix(value/const) → Grad(basis)")
                        body_lines.append(
                            f"{res_var} = contract_last_first({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                        stack.append(StackItem(var_name=res_var, role=a.role,
                                            shape=(a.shape[0], a.shape[1], b.shape[1]),
                                            is_vector=False, is_gradient=True,
                                            field_names=field_names, parent_name=parent_name,
                                            side=side, field_sides=field_sides,
                                            expression_meta=product_meta))
                    elif (
                        b.role in {"trial", "test"}
                        and b.is_gradient
                        and len(b.shape) == 3
                        and a.role in {"value", "const"}
                        and not a.is_vector
                        and len(a.shape) == 2
                        and (
                            (b.shape[0] == 1 and a.shape[1] == b.shape[2])
                            or a.shape[1] == b.shape[0]
                        )
                    ):
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='b', strict=False)
                        if b.shape[0] == 1 and a.shape[1] == b.shape[2]:
                            body_lines.append("# Product: matrix(value/const) × scalar Grad(basis) → Grad(basis)")
                            body_lines.append(
                                f"{res_var} = contract_last_first({b.var_name}, np.ascontiguousarray({a.var_name}.T), {self.dtype})"
                            )
                            out_shape = (1, b.shape[1], a.shape[0])
                        else:
                            body_lines.append("# Product: matrix(value/const) × vector Grad(basis) → Grad(basis)")
                            body_lines.append(
                                f"{res_var} = contract_last_first({a.var_name}, {b.var_name}, {self.dtype})"
                            )
                            out_shape = (a.shape[0], b.shape[1], b.shape[2])
                        stack.append(StackItem(var_name=res_var, role=b.role,
                                            shape=out_shape,
                                            is_vector=False, is_gradient=True,
                                            field_names=field_names, parent_name=parent_name,
                                            side=side, field_sides=field_sides,
                                            expression_meta=product_meta))
                    elif (
                        a.role in {"value", "const"}
                        and a.is_vector
                        and not a.is_gradient
                        and not a.is_hessian
                        and len(a.shape) == 1
                        and b.role in {"value", "const"}
                        and not b.is_vector
                        and len(b.shape) == 2
                        and a.shape[0] == b.shape[0]
                    ):
                        body_lines.append("# Product: vector(value/const) × matrix(value/const) → vector(value/const)")
                        body_lines.append(
                            f"{res_var} = np.ascontiguousarray({a.var_name}) @ np.ascontiguousarray({b.var_name})"
                        )
                        role = "value" if "value" in {a.role, b.role} else "const"
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role=role,
                            fallback_shape=(b.shape[1],),
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=True,
                            is_gradient=False,
                            is_hessian=False,
                            expression_meta=product_meta,
                        )
                    elif (
                        b.role in {"value", "const"}
                        and b.is_vector
                        and not b.is_gradient
                        and not b.is_hessian
                        and len(b.shape) == 1
                        and a.role in {"value", "const"}
                        and not a.is_vector
                        and len(a.shape) == 2
                        and a.shape[1] == b.shape[0]
                    ):
                        body_lines.append("# Product: matrix(value/const) × vector(value/const) → vector(value/const)")
                        body_lines.append(
                            f"{res_var} = np.ascontiguousarray({a.var_name}) @ np.ascontiguousarray({b.var_name})"
                        )
                        role = "value" if "value" in {a.role, b.role} else "const"
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role=role,
                            fallback_shape=(a.shape[0],),
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=True,
                            is_gradient=False,
                            is_hessian=False,
                            expression_meta=product_meta,
                        )

                    # -----------------------------------------------------------------
                    # 1. RHS load:   scalar / vector Function  *  scalar Test
                    #                (u_k or c)                ·  φ_v
                    # -----------------------------------------------------------------
                    elif (b.role == "test" and not b.is_vector
                        and a.role == "value"
                        and (
                            (not a.is_vector)
                            or (len(a.shape) == 1 and a.shape[0] == 1)
                        )
                        and not a.is_gradient and not b.is_gradient
                        and not a.is_hessian and not b.is_hessian
                        ):
                        body_lines.append("# Product fallback: scalar Function × scalar Test")
                        scalar_a = a
                        if a.is_vector and len(a.shape) == 1 and a.shape[0] == 1:
                            scalar_a = StackItem(
                                var_name=f"{a.var_name}[0]",
                                role=a.role,
                                shape=(),
                                is_vector=False,
                                is_gradient=False,
                                is_hessian=False,
                                field_names=a.field_names,
                                parent_name=a.parent_name,
                                side=a.side,
                                field_sides=a.field_sides or [],
                                layout_tag=getattr(a, 'layout_tag', ''),
                                expression_meta=getattr(a, 'expression_meta', None),
                            )
                        _mul_scalar_vector(self, first_is_scalar=True, a=scalar_a, b=b, res_var=res_var, body_lines=body_lines, stack=stack)

                    # symmetric orientation
                    elif (a.role == "test" and not a.is_vector
                        and b.role == "value"
                        and (
                            (not b.is_vector)
                            or (len(b.shape) == 1 and b.shape[0] == 1)
                        )
                        and not a.is_gradient and not b.is_gradient
                        and not a.is_hessian and not b.is_hessian
                        ):
                        body_lines.append("# Product fallback: scalar Test × scalar Function")
                        scalar_b = b
                        if b.is_vector and len(b.shape) == 1 and b.shape[0] == 1:
                            scalar_b = StackItem(
                                var_name=f"{b.var_name}[0]",
                                role=b.role,
                                shape=(),
                                is_vector=False,
                                is_gradient=False,
                                is_hessian=False,
                                field_names=b.field_names,
                                parent_name=b.parent_name,
                                side=b.side,
                                field_sides=b.field_sides or [],
                                layout_tag=getattr(b, 'layout_tag', ''),
                                expression_meta=getattr(b, 'expression_meta', None),
                            )
                        _mul_scalar_vector(self, first_is_scalar=False, a=a, b=scalar_b, res_var=res_var, body_lines=body_lines, stack=stack)
                    # -----------------------------------------------------------------
                    # scalar-like value/const coefficient (represented as a length-1
                    # vector) times a mixed basis block
                    # -----------------------------------------------------------------
                    elif (
                        a.role in {"const", "value"}
                        and a.is_vector
                        and len(a.shape) == 1
                        and a.shape[0] == 1
                        and not a.is_gradient and not a.is_hessian
                        and b.role == "mixed"
                        and not b.is_gradient and not b.is_hessian
                    ):
                        body_lines.append("# Product: scalar-like value * mixed basis → mixed")
                        body_lines.append(f"{res_var} = {a.var_name}[0] * {b.var_name}")
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='b', strict=False)
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role="mixed",
                            fallback_shape=b.shape,
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=False,
                            is_gradient=False,
                            is_hessian=False,
                            expression_meta=product_meta,
                        )
                    elif (
                        b.role in {"const", "value"}
                        and b.is_vector
                        and len(b.shape) == 1
                        and b.shape[0] == 1
                        and not b.is_gradient and not b.is_hessian
                        and a.role == "mixed"
                        and not a.is_gradient and not a.is_hessian
                    ):
                        body_lines.append("# Product: mixed basis * scalar-like value → mixed")
                        body_lines.append(f"{res_var} = {b.var_name}[0] * {a.var_name}")
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer='a', strict=False)
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role="mixed",
                            fallback_shape=a.shape,
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=False,
                            is_gradient=False,
                            is_hessian=False,
                            expression_meta=product_meta,
                        )
                    # -----------------------------------------------------------------
                    # 1. RHS p * I:   trace(test)  * identity
                    #                (u_test)                ·  φ_v
                    # -----------------------------------------------------------------
                    elif (a.role in {"test", "trial"} and b.role in {"const", "value"} and b.is_gradient
                          and not a.is_vector and not a.is_gradient and not a.is_hessian
                          and len(a.shape) in (1, 2) and (len(a.shape) == 1 or a.shape[0] == 1)
                          and b.shape == (2, 2)):
                        role = 'test' if a.role == 'test' else 'trial'
                        body_lines.append(f"# rhs: trace({role}) * Identity → ")
                        body_lines.append(
                            f"{res_var} = trace_times_identity({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer=a, strict=False)
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role=role,
                            fallback_shape=(b.shape[0], self.active_n_dofs, b.shape[1]),
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=False,
                            is_gradient=True,
                            is_hessian=False,
                            expression_meta=product_meta,
                        )
                    elif (b.role in {"test", "trial"} and a.role in {"const", "value"} and a.is_gradient
                          and not b.is_vector and not b.is_gradient and not b.is_hessian
                          and len(b.shape) in (1, 2) and (len(b.shape) == 1 or b.shape[0] == 1)
                          and a.shape == (2, 2)):
                        role = 'test' if b.role == 'test' else 'trial'
                        body_lines.append(f"# rhs: Identity * trace({role}) → ")
                        body_lines.append(
                            f"{res_var} = trace_times_identity({b.var_name}, {a.var_name}, {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer=b, strict=False)
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role=role,
                            fallback_shape=(a.shape[0], self.active_n_dofs, a.shape[1]),
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=False,
                            is_gradient=True,
                            is_hessian=False,
                            expression_meta=product_meta,
                        )
                    elif (a.role in {"const", "value"} and b.role in {"const", "value"} and b.is_gradient
                          and not a.is_vector and not a.is_gradient and not a.is_hessian
                           and (len(a.shape) == 1 )
                          and b.shape == (2, 2)) and a.shape[0] == 1:
                        # scale up to matrix
                        role = "value"
                        body_lines.append("# rhs: pk * Identity → ")
                        body_lines.append(
                            f"{res_var} = {a.var_name} * {b.var_name}"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer=a, strict=False)
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role=role,
                            fallback_shape=(b.shape[0], b.shape[1]),
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=False,
                            is_gradient=True,
                            is_hessian=False,
                            expression_meta=product_meta,
                        )
                    elif (b.role in {"const", "value"} and a.role in {"const", "value"} and a.is_gradient
                          and not b.is_vector and not b.is_gradient and not b.is_hessian
                           and (len(b.shape) == 1  and b.shape[0] == 1) and a.shape == (2, 2)):
                        # scale up to matrix
                        role = "value"
                        body_lines.append("# rhs: Identity * pk → ")
                        body_lines.append(
                            f"{res_var} = {b.var_name} * {a.var_name}"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer=b, strict=False)
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role=role,
                            fallback_shape=(a.shape[0], a.shape[1]),
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=False,
                            is_gradient=True,
                            is_hessian=False,
                            expression_meta=product_meta,
                        )
                    elif (a.role in {"const", "value"} and b.role == "mixed" and b.is_gradient
                          and a.shape == (2, 2)):
                        role = 'mixed'
                        body_lines.append("# rhs: Identity × trace(mixed) → ")
                        body_lines.append(
                            f"{res_var} = identity_times_trace_matrix({a.var_name}, {b.var_name}, {self.dtype})"
                        )
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer=b, strict=False)
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role=role,
                            fallback_shape=(a.shape[0], self.active_n_dofs, self.active_n_dofs, a.shape[1]),
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=False,
                            is_gradient=True,
                            is_hessian=False,
                            expression_meta=product_meta,
                        )

                    # -----------------------------------------------------------------
                    # 3. Value/const vector × value/const vector → elementwise product
                    # -----------------------------------------------------------------
                    elif (
                        a.role in {"const", "value"}
                        and b.role in {"const", "value"}
                        and a.is_vector
                        and b.is_vector
                        and (not a.is_gradient and not b.is_gradient)
                        and (not a.is_hessian and not b.is_hessian)
                        and len(a.shape) == 1
                        and a.shape == b.shape
                    ):
                        body_lines.append("# Product: vector(value) * vector(value) → vector(value) (elementwise)")
                        body_lines.append(f"{res_var} = {a.var_name} * {b.var_name}")
                        role = "value" if ("value" in {a.role, b.role}) else "const"
                        field_names, parent_name, side, field_sides = StackItem.resolve_metadata(a, b, prefer="a", strict=False)
                        _push_product_result(
                            stack,
                            res_var=res_var,
                            default_role=role,
                            fallback_shape=a.shape,
                            field_names=field_names,
                            parent_name=parent_name,
                            side=side,
                            field_sides=field_sides,
                            product_lowering=product_lowering,
                            product_value_spec=product_value_spec,
                            is_vector=True,
                            is_gradient=False,
                            is_hessian=False,
                            expression_meta=product_meta,
                        )

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
                     sum_plan = _try_sum_plan(a, b)
                     sum_lowering = _try_sum_lowering(a, b)
                     sum_value_spec = _try_sum_value_spec(a, b)
                     body_lines.append(f"# {'Addition' if sym == '+' else 'Subtraction'}")

                     def _merge_role(ra, rb):
                         if sum_value_spec is not None:
                             planned = getattr(sum_value_spec, "role", "") or ""
                             if planned not in {"", "none"}:
                                 return planned
                         if sum_plan is not None:
                             planned = getattr(sum_plan.result.tensor, "role", "") or ""
                             if planned not in {"", "none"}:
                                 return planned
                         if 'mixed' in (ra, rb):
                             return 'mixed'
                         if 'trial' in (ra, rb):
                             return 'trial'
                         if 'test' in (ra, rb):
                             return 'test'
                         if 'value' in (ra, rb):
                             return 'value'
                         return 'const'

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
                     planned_shape = ()
                     if sum_lowering is not None:
                         planned_shape = getattr(sum_lowering.result_storage, "stored_shape", ()) or ()
                     elif sum_plan is not None:
                         planned_shape = getattr(sum_plan.result.tensor, "raw_shape", ()) or ()
                     lhs_expr = a.var_name
                     rhs_expr = b.var_name
                     lhs_shape = a.shape
                     rhs_shape = b.shape
                     if sum_lowering is not None:
                         lhs_expr, lhs_shape = _apply_sum_operand_transform(
                             lhs_expr,
                             lhs_shape,
                             sum_lowering.lhs_transform,
                             self.dtype,
                         )
                         rhs_expr, rhs_shape = _apply_sum_operand_transform(
                             rhs_expr,
                             rhs_shape,
                             sum_lowering.rhs_transform,
                             self.dtype,
                         )
                     try:
                         new_shape = _merge_runtime_shapes(lhs_shape, rhs_shape, planned_shape)
                     except ValueError as exc:
                         raise NotImplementedError(
                             f"'{sym}' cannot broadcast runtime shapes {lhs_shape} and {rhs_shape}; "
                             f"roles={a.role}/{b.role}, "
                             f"vector={a.is_vector}/{b.is_vector}, "
                             f"gradient={a.is_gradient}/{b.is_gradient}, "
                             f"hessian={a.is_hessian}/{b.is_hessian}, "
                             f"layout={getattr(a, 'layout_tag', '')}/{getattr(b, 'layout_tag', '')}, "
                             f"fields={a.field_names}/{b.field_names}"
                         ) from exc

                     lhs_expr = _collapse_leading_singleton_ref(lhs_expr, lhs_shape, new_shape)
                     rhs_expr = _collapse_leading_singleton_ref(rhs_expr, rhs_shape, new_shape)
                     body_lines.append(
                         f"{res_var} = {helper}({lhs_expr}, {rhs_expr}, {self.dtype})"
                     )
                     field_names, parent_name, side, field_sides = StackItem.resolve_metadata(
                         a, b, prefer='basis', strict=False
                     )
                     role_out = _merge_role(a.role, b.role)
                     is_vector_out = a.is_vector or b.is_vector
                     is_gradient_out = a.is_gradient or b.is_gradient
                     is_hessian_out = a.is_hessian or b.is_hessian
                     layout_out = getattr(a, "layout_tag", "") or getattr(b, "layout_tag", "")
                     expression_meta = None
                     if sum_lowering is not None:
                         result_meta = getattr(sum_lowering.algebra, "result", None)
                         if result_meta is not None:
                             expression_meta = result_meta
                     if sum_value_spec is not None:
                         planned_shape = tuple(int(v) for v in getattr(sum_value_spec, "shape", ()) or ())
                         if planned_shape:
                             new_shape = _merge_runtime_shapes(new_shape, planned_shape, planned_shape)
                         role_out = getattr(sum_value_spec, "role", role_out) or role_out
                         is_vector_out = bool(getattr(sum_value_spec, "is_vector", is_vector_out))
                         is_gradient_out = bool(getattr(sum_value_spec, "is_gradient", is_gradient_out))
                         is_hessian_out = bool(getattr(sum_value_spec, "is_hessian", is_hessian_out))
                         layout_obj = getattr(sum_value_spec, "layout", None)
                         layout_out = layout_obj.value if layout_obj is not None else layout_out
                         expression_meta = getattr(sum_value_spec, "meta", expression_meta)
                     stack.append(StackItem(
                         var_name    = res_var,
                         role        = role_out,
                         shape       = new_shape,
                         is_vector   = is_vector_out,
                         is_gradient = is_gradient_out,
                         is_hessian  = is_hessian_out,
                         parent_name = parent_name,
                         field_names = field_names,
                         side        = side,
                         field_sides = field_sides,
                         layout_tag  = layout_out,
                         expression_meta=expression_meta,
                     ))
                     continue
                 elif op.op_symbol == '/':
                    division_value_spec = _try_division_value_spec(a, b)
                    body_lines.append("# Division")
                    def _push_division_result(default_item):
                        role_out = default_item.role
                        shape_out = default_item.shape
                        is_vector_out = default_item.is_vector
                        is_gradient_out = default_item.is_gradient
                        is_hessian_out = default_item.is_hessian
                        layout_out = getattr(default_item, "layout_tag", "")
                        expression_meta = getattr(default_item, "expression_meta", None)
                        if division_value_spec is not None:
                            planned_shape = tuple(int(v) for v in getattr(division_value_spec, "shape", ()) or ())
                            if planned_shape:
                                shape_out = planned_shape
                            role_out = getattr(division_value_spec, "role", role_out) or role_out
                            is_vector_out = bool(getattr(division_value_spec, "is_vector", is_vector_out))
                            is_gradient_out = bool(getattr(division_value_spec, "is_gradient", is_gradient_out))
                            is_hessian_out = bool(getattr(division_value_spec, "is_hessian", is_hessian_out))
                            layout_obj = getattr(division_value_spec, "layout", None)
                            layout_out = layout_obj.value if layout_obj is not None else layout_out
                            expression_meta = getattr(division_value_spec, "meta", expression_meta)
                        stack.append(StackItem(
                            var_name=res_var,
                            role=role_out,
                            shape=shape_out,
                            is_vector=is_vector_out,
                            is_gradient=is_gradient_out,
                            is_hessian=is_hessian_out,
                            field_names=default_item.field_names,
                            parent_name=default_item.parent_name,
                            side=default_item.side,
                            field_sides=default_item.field_sides or [],
                            layout_tag=layout_out,
                            expression_meta=expression_meta,
                        ))
                    # divide *anything* by a scalar constant (const in denominator)
                    if (b.role == 'const' or b.role == 'value') and not b.is_vector and b.shape == ():
                        body_lines.append(f"{res_var} = {a.var_name} / float({b.var_name})")
                        _push_division_result(a)
                    elif (a.role == 'const' or a.role == 'value') and not a.is_vector and a.shape == ():
                        body_lines.append(f"{res_var} = float({a.var_name}) / {b.var_name}")
                        _push_division_result(b)
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
                    # Only peel a leading singleton dimension when the runtime
                    # value is actually a rank-3 array (1,n,n). For 2D matrices
                    # (n,n) we must not index [0] (would broadcast a row).
                    if len(integrand.shape) >= 3 and integrand.shape[0] == 1:
                        body_lines.append(f"Ke += {integrand.var_name}[0] * w_q")
                    else:
                        body_lines.append(f"Ke += {integrand.var_name} * w_q")
                elif op.store_type == 'vector':
                    # Only peel a leading singleton dimension when the runtime
                    # value is actually rank-2 (1,n). If the value is 1D (n,),
                    # indexing [0] would turn it into a scalar and broadcast.
                    if len(integrand.shape) >= 2 and integrand.shape[0] == 1:
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

        needs_phis = any(isinstance(op, (PosOp, NegOp)) for op in ir_sequence)
        source, param_order = self._build_kernel_string(
            kernel_name,
            body_lines,
            required_args,
            solution_func_names,
            functional_shape,
            needs_phis=needs_phis,
        )
        return source, {}, param_order


    def _build_kernel_string(
            self, kernel_name: str,
            body_lines: list,
            required_args: set,
            solution_func_names: set,
            functional_shape: tuple = None,
            DEBUG: bool = False,
            *,
            needs_phis: bool = False,
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
        param_order = ["gdofs_map", "node_coords", "qp_phys", "qw", "detJ", "J_inv", "normals"]
        if needs_phis:
            param_order.append("phis")
        param_order += sorted(list(required_args))
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


        local_symbol_rewrites: list[tuple[re.Pattern[str], str]] = []
        for name in sorted(list(solution_func_names)):
            source_name = name if name.startswith("u_") and name.endswith("_loc") else f"u_{name}_loc"
            target_name = f"{source_name}_e"
            local_symbol_rewrites.append(
                (re.compile(rf"(?<![A-Za-z0-9_]){re.escape(source_name)}(?![A-Za-z0-9_])"), target_name)
            )

        rewritten_body_lines: list[str] = []
        for line in body_lines:
            if not line.strip():
                continue
            rewritten = line
            for pattern, replacement in local_symbol_rewrites:
                rewritten = pattern.sub(replacement, rewritten)
            rewritten_body_lines.append(f"            {rewritten}")

        body_code_block = "\n".join(rewritten_body_lines)

        phi_q_line = (
            "            phi_q    = phis[e, q] if phis is not None else 0.0"
            if needs_phis
            else "            phi_q    = 0.0"
        )

        decorator = ""
        if not DEBUG:
            # Numba's on-disk cache can be fragile for facet kernels (large signatures and
            # many small kernels). We already cache the generated Python source in
            # `~/.cache/pycutfem_jit`, so disable Numba caching on facet kernels by default.
            use_cache = not bool(getattr(self, "on_facet", False))
            # Parallel compilation can be very expensive for large generated kernels.
            # Allow opting out via an env var while keeping the historical default.
            use_parallel = os.getenv("PYCUTFEM_JIT_PARALLEL", "1").lower() not in {"0", "false", "no"}
            parallel_ir_limit_raw = os.getenv("PYCUTFEM_JIT_PARALLEL_IR_LIMIT", "800").strip().lower()
            try:
                parallel_ir_limit = int(parallel_ir_limit_raw)
            except ValueError:
                parallel_ir_limit = 800
            if parallel_ir_limit > 0 and len(body_lines) > parallel_ir_limit:
                use_parallel = False
            decorator = f"@numba.njit(parallel={str(use_parallel)}, fastmath=True, cache={str(use_cache)})"
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
    value_matrix_times_scalar_basis_tensor,
    dot_grad_basis_vector,
    vector_dot_grad_basis,
    vector_dot_grad_value,
    dot_grad_basis_with_grad_value,
    dot_grad_value_with_grad_basis,
    basis_dot_const_vector,
        const_vector_dot_basis,
        const_vector_dot_basis_1d,
        scalar_basis_times_vector,
        scalar_basis_times_vector_as_grad_tensor,
        matrix_times_scalar_basis,
        scalar_vector_outer_product,
        vector_trial_times_scalar_test,
        vector_test_times_scalar_trial,
        vector_vector_outer_product,
        value_vector_outer_basis_vector,
        basis_vector_outer_value_vector,
        basis_vector_outer_basis_vector,
        left_dot_mixed_tensor_with_vec,
        contract_component_first_basis,
    dot_vec_grad_components,
    scalar_trial_times_grad_test,
    scalar_trial_times_basis_test,
    grad_trial_times_scalar_test,
    basis_trial_times_scalar_test,
    scale_mixed_basis_with_coeffs,
    trace_times_identity,
    identity_times_trace_matrix,
    columnwise_dot,
    hessian_dot_vector,
    vector_dot_hessian_basis,
    vector_dot_hessian_value,
    inner_rank2_value_rank2_basis,
    inner_rank2_basis_rank2_value,
    inner_full_contraction,
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
    swap_mixed_basis_tensor,
    dot_value_with_grad,
    dot_grad_with_value,
    pushforward_d3,
    pushforward_d4,
    compute_physical_hessian,
    compute_physical_laplacian,
    load_variable_qp,
    gradient_qp,
    laplacian_qp,
    hessian_qp,
    ghost_grad_jump_penalty_scalar,
    ghost_grad_jump_penalty_vector,
    binary_add_generic,
    binary_add_3_4,
    binary_add_4_3,
    binary_sub_generic,
    binary_sub_3_4,
    binary_sub_4_3,
    scatter_tensor_to_union,
    pad_basis_to_union,
    pushforward_grad_to_union,
    contract_first_first
)
PARAM_ORDER = [{param_order_literal}]
{decorator}
def {kernel_name}(
        {", ".join(param_order)}
    ):
    num_elements        = qp_phys.shape[0]
    # n_dofs_per_element  = {self.active_n_dofs}
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

        n_union = gdofs_map.shape[1]

{coeffs_unpack_block}

        for q in range(qw.shape[1]):
            x_q, w_q, J_inv_q = qp_phys[e, q], qw[e, q], J_inv[e, q]
            normal_q = normals[e, q] if normals is not None else np.zeros(2, dtype={self.dtype})
{phi_q_line}
{basis_unpack_block}
{body_code_block}

        K_values[e] = Ke
        F_values[e] = Fe
        J_values[e] = J
                
    return K_values, F_values, J_values
""".lstrip()

        return final_kernel_src, param_order
