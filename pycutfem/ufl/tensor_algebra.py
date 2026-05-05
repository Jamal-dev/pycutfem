from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numbers
from typing import Any

import numpy as np


class AxisSpace(str, Enum):
    PHYSICAL = "physical"


class AxisLabel(str, Enum):
    COMPONENT = "component"
    DERIVATIVE = "derivative"
    ROW = "row"
    COL = "col"
    HESSIAN_ROW = "hessian_row"
    HESSIAN_COL = "hessian_col"


class BasisLabel(str, Enum):
    TEST = "test_basis"
    TRIAL = "trial_basis"


class OperationKind(str, Enum):
    DOT_VECTOR_VECTOR = "dot_vector_vector"
    DOT_TENSOR_TENSOR = "dot_tensor_tensor"
    DOT_HESSIAN_VECTOR = "dot_hessian_vector"
    DOT_VECTOR_HESSIAN = "dot_vector_hessian"
    INNER_FULL = "inner_full"
    SUM_GENERIC = "sum_generic"
    DIVIDE_GENERIC = "divide_generic"
    PRODUCT_SCALE = "product_scale"
    PRODUCT_OUTER = "product_outer"
    PRODUCT_PROMOTE = "product_promote"
    PRODUCT_TENSOR = "product_tensor"
    PRODUCT_GENERIC = "product_generic"


class MixedLayout(str, Enum):
    DEFAULT = "default"
    COMPONENT_FIRST = "component_first"
    COMPONENT_LAST = "component_last"


class OperandTransform(str, Enum):
    NONE = "none"
    TRANSPOSE_2D = "transpose_2d"
    SCALAR_GRAD_TO_VECTOR = "scalar_grad_to_vector"


class DotKernelCase(str, Enum):
    BASIS_BASIS_MASS = "basis_basis_mass"
    BASIS_GRAD_DOT_VALUE_VECTOR = "basis_grad_dot_value_vector"
    VALUE_VECTOR_DOT_BASIS_GRAD = "value_vector_dot_basis_grad"
    VALUE_GRAD_DOT_BASIS_VECTOR = "value_grad_dot_basis_vector"
    BASIS_VECTOR_DOT_VALUE_GRAD = "basis_vector_dot_value_grad"
    BASIS_GRAD_DOT_BASIS_VECTOR = "basis_grad_dot_basis_vector"
    BASIS_VECTOR_DOT_BASIS_GRAD = "basis_vector_dot_basis_grad"
    GENERIC_CONTRACT_LAST_FIRST = "generic_contract_last_first"


class ProductKernelCase(str, Enum):
    GENERIC_SCALE = "generic_scale"
    BASIS_SCALAR_TIMES_VALUE_VECTOR = "basis_scalar_times_value_vector"
    VALUE_VECTOR_TIMES_BASIS_SCALAR = "value_vector_times_basis_scalar"
    BASIS_SCALAR_TIMES_VALUE_MATRIX = "basis_scalar_times_value_matrix"
    VALUE_MATRIX_TIMES_BASIS_SCALAR = "value_matrix_times_basis_scalar"
    MIXED_SCALAR_TIMES_VALUE_MATRIX = "mixed_scalar_times_value_matrix"
    VALUE_MATRIX_TIMES_MIXED_SCALAR = "value_matrix_times_mixed_scalar"
    VALUE_VECTOR_OUTER_VALUE_VECTOR = "value_vector_outer_value_vector"
    GENERIC_TENSOR_PRODUCT = "generic_tensor_product"


@dataclass(frozen=True)
class TensorAxis:
    space: AxisSpace
    label: AxisLabel
    size: int

    def compatible_with(self, other: "TensorAxis") -> bool:
        if self.space != other.space:
            return False
        if self.size < 0 or other.size < 0:
            size_ok = True
        else:
            size_ok = int(self.size) == int(other.size)
        if not size_ok:
            return False
        vector_like = {
            AxisLabel.COMPONENT,
            AxisLabel.DERIVATIVE,
            AxisLabel.ROW,
            AxisLabel.COL,
            AxisLabel.HESSIAN_ROW,
            AxisLabel.HESSIAN_COL,
        }
        if self.label == other.label:
            return True
        return self.label in vector_like and other.label in vector_like


@dataclass(frozen=True)
class TensorSignature:
    free_axes: tuple[TensorAxis, ...]
    basis_axes: tuple[BasisLabel, ...]
    storage_kind: str
    raw_shape: tuple[int, ...]
    role: str
    source: str = ""

    @property
    def tensor_rank(self) -> int:
        return len(self.free_axes)

    @property
    def basis_rank(self) -> int:
        return len(self.basis_axes)

    @property
    def is_scalar(self) -> bool:
        return self.tensor_rank == 0


class ProvenanceKind(str, Enum):
    CONSTANT = "constant"
    FIELD_COMPONENTS = "field_components"
    DERIVATIVE_CHANNELS = "derivative_channels"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FieldSource:
    parent: str
    fields: tuple[str, ...]
    kind: ProvenanceKind
    derivative_depth: int = 0
    role: str = ""
    source: str = ""


@dataclass(frozen=True)
class ProvenanceSignature:
    sources: tuple[FieldSource, ...]


@dataclass(frozen=True)
class ExpressionMeta:
    tensor: TensorSignature
    provenance: ProvenanceSignature


@dataclass(frozen=True)
class StorageSpec:
    stored_shape: tuple[int, ...]
    free_axis_positions: tuple[int, ...]
    basis_axis_positions: tuple[int, ...]
    basis_sizes: tuple[int, ...]
    canonical_shape: tuple[int, ...]


@dataclass(frozen=True)
class DotPlan:
    kind: OperationKind
    lhs: TensorSignature
    rhs: TensorSignature
    result: TensorSignature


@dataclass(frozen=True)
class ProductPlan:
    kind: OperationKind
    lhs: TensorSignature
    rhs: TensorSignature
    result: TensorSignature


@dataclass(frozen=True)
class SumPlan:
    kind: OperationKind
    lhs: ExpressionMeta
    rhs: ExpressionMeta
    result: ExpressionMeta


@dataclass(frozen=True)
class SumLoweringPlan:
    algebra: SumPlan
    lhs_storage: StorageSpec
    rhs_storage: StorageSpec
    result_storage: StorageSpec
    lhs_transform: OperandTransform = OperandTransform.NONE
    rhs_transform: OperandTransform = OperandTransform.NONE


@dataclass(frozen=True)
class InnerPlan:
    kind: OperationKind
    lhs: ExpressionMeta
    rhs: ExpressionMeta
    result: ExpressionMeta


@dataclass(frozen=True)
class InnerLoweringPlan:
    algebra: InnerPlan
    result: LoweredResult
    lhs_storage: StorageSpec
    rhs_storage: StorageSpec
    result_storage: StorageSpec


@dataclass(frozen=True)
class DivisionPlan:
    kind: OperationKind
    lhs: ExpressionMeta
    rhs: ExpressionMeta
    result: ExpressionMeta


@dataclass(frozen=True)
class DivisionLoweringPlan:
    algebra: DivisionPlan
    result: LoweredResult
    lhs_storage: StorageSpec
    rhs_storage: StorageSpec
    result_storage: StorageSpec


@dataclass(frozen=True)
class LoweredResult:
    role: str
    layout: MixedLayout
    is_vector: bool
    is_gradient: bool
    is_hessian: bool


@dataclass(frozen=True)
class DotLoweringPlan:
    algebra: DotPlan
    meta: ExpressionMeta
    result: LoweredResult
    lhs_storage: StorageSpec
    rhs_storage: StorageSpec
    result_storage: StorageSpec
    swap_mixed_basis_axes: bool = False


@dataclass(frozen=True)
class ProductLoweringPlan:
    algebra: ProductPlan
    meta: ExpressionMeta
    result: LoweredResult
    lhs_storage: StorageSpec
    rhs_storage: StorageSpec
    result_storage: StorageSpec


@dataclass(frozen=True)
class DotKernelPlan:
    lowering: DotLoweringPlan
    case: DotKernelCase


@dataclass(frozen=True)
class ProductKernelPlan:
    lowering: ProductLoweringPlan
    case: ProductKernelCase


@dataclass(frozen=True)
class KernelValueSpec:
    kind: str
    role: str
    shape: tuple[int, ...]
    layout: MixedLayout
    is_vector: bool
    is_gradient: bool
    is_hessian: bool
    meta: ExpressionMeta


def _shape_of(obj: Any) -> tuple[int, ...]:
    shape = getattr(obj, "shape", None)
    if shape is None:
        arr = np.asarray(obj)
        return tuple(int(v) for v in arr.shape)
    return tuple(int(v) for v in shape)


def _infer_basis_vector_axis_size(shape: tuple[int, ...], spatial_dim: int) -> int | None:
    if len(shape) != 2:
        return None
    a = int(shape[0])
    b = int(shape[1])
    a_spatial = 0 < a <= spatial_dim
    b_spatial = 0 < b <= spatial_dim
    if a_spatial and (not b_spatial or b == -1):
        return a
    if b_spatial and (not a_spatial or a == -1):
        return b
    return None


def _basis_axes_for_role(role: str, *, mixed: bool = False) -> tuple[BasisLabel, ...]:
    if mixed or role == "mixed":
        return (BasisLabel.TEST, BasisLabel.TRIAL)
    if role in {"test", "test_n"}:
        return (BasisLabel.TEST,)
    if role in {"trial", "trial_n"}:
        return (BasisLabel.TRIAL,)
    return ()


def _role_from_basis_axes(
    basis_axes: tuple[BasisLabel, ...],
    lhs_role: str,
    rhs_role: str,
) -> str:
    if not basis_axes:
        if "const" in {lhs_role, rhs_role}:
            return "const" if lhs_role == rhs_role == "const" else "value"
        if "function" in {lhs_role, rhs_role}:
            return "function"
        if "value" in {lhs_role, rhs_role}:
            return "value"
        return lhs_role if lhs_role not in {"", "none"} else rhs_role
    if basis_axes == (BasisLabel.TEST,):
        return "test"
    if basis_axes == (BasisLabel.TRIAL,):
        return "trial"
    return "mixed"


def _canonical_basis_axes(lhs: TensorSignature, rhs: TensorSignature) -> tuple[BasisLabel, ...]:
    axes = list(lhs.basis_axes) + list(rhs.basis_axes)
    if not axes:
        return ()
    ordered: list[BasisLabel] = []
    if BasisLabel.TEST in axes:
        ordered.append(BasisLabel.TEST)
    if BasisLabel.TRIAL in axes:
        ordered.append(BasisLabel.TRIAL)
    for axis in axes:
        if axis not in ordered:
            ordered.append(axis)
    return tuple(ordered)


def _dedupe_sources(sources: list[FieldSource]) -> tuple[FieldSource, ...]:
    ordered: list[FieldSource] = []
    seen: set[tuple[str, tuple[str, ...], ProvenanceKind, int, str]] = set()
    for src in sources:
        key = (src.parent, src.fields, src.kind, int(src.derivative_depth), src.role)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(src)
    return tuple(ordered)


def _merge_provenance(lhs: ProvenanceSignature, rhs: ProvenanceSignature) -> ProvenanceSignature:
    return ProvenanceSignature(_dedupe_sources(list(lhs.sources) + list(rhs.sources)))


def _project_provenance_to_tensor(
    provenance: ProvenanceSignature,
    tensor: TensorSignature,
) -> ProvenanceSignature:
    derivative_rank = 0
    labels = {axis.label for axis in tensor.free_axes}
    if AxisLabel.HESSIAN_ROW in labels or AxisLabel.HESSIAN_COL in labels:
        derivative_rank = 2
    elif AxisLabel.DERIVATIVE in labels:
        derivative_rank = 1

    if derivative_rank == 0 and not provenance.sources:
        return provenance

    # Basis-backed intermediates frequently collapse or transform derivative
    # axes into carried component axes before a later contraction consumes them.
    # We still need that derivative origin in the provenance so chained dot()
    # planning can distinguish transported gradients from plain vector bases.
    #
    # For pure values we continue to project to the exposed tensor axes only;
    # this avoids reclassifying transformed value gradients such as
    # dot(Finv.T, grad(p)) as semantic gradients in later contractions.
    if tensor.basis_rank > 0:
        return ProvenanceSignature(_dedupe_sources(list(provenance.sources)))

    projected: list[FieldSource] = []
    for src in provenance.sources:
        new_depth = min(int(src.derivative_depth), derivative_rank)
        new_kind = src.kind
        if src.kind == ProvenanceKind.DERIVATIVE_CHANNELS and new_depth == 0:
            new_kind = ProvenanceKind.FIELD_COMPONENTS if src.fields else ProvenanceKind.UNKNOWN
        projected.append(
            FieldSource(
                parent=src.parent,
                fields=src.fields,
                kind=new_kind,
                derivative_depth=new_depth,
                role=src.role,
                source=src.source,
            )
        )
    return ProvenanceSignature(_dedupe_sources(projected))


def _storage_axis_positions(tensor_rank: int, basis_rank: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if basis_rank == 0:
        return tuple(range(tensor_rank)), ()
    if tensor_rank <= 1:
        free_positions = tuple(range(tensor_rank))
        basis_positions = tuple(range(tensor_rank, tensor_rank + basis_rank))
        return free_positions, basis_positions
    free_positions = tuple(range(tensor_rank - 1)) + (tensor_rank + basis_rank - 1,)
    basis_positions = tuple(range(tensor_rank - 1, tensor_rank + basis_rank - 1))
    return free_positions, basis_positions


def _storage_shape_from_free_and_basis(
    free_sizes: tuple[int, ...],
    basis_sizes: tuple[int, ...],
    *,
    scalar_basis_as_row: bool = False,
) -> tuple[int, ...]:
    tensor_rank = len(free_sizes)
    basis_rank = len(basis_sizes)
    if basis_rank == 0:
        return free_sizes
    if tensor_rank == 0 and basis_rank == 1 and scalar_basis_as_row:
        return (1, basis_sizes[0])
    if tensor_rank <= 1:
        return free_sizes + basis_sizes
    return free_sizes[:-1] + basis_sizes + free_sizes[-1:]


def _basis_size_map(sig: TensorSignature, spec: StorageSpec) -> dict[BasisLabel, int]:
    out: dict[BasisLabel, int] = {}
    for lbl, size in zip(sig.basis_axes, spec.basis_sizes, strict=True):
        out[lbl] = int(size)
    return out


def _canonical_shape_from_signature(
    sig: TensorSignature,
    free_sizes: tuple[int, ...],
    basis_sizes: tuple[int, ...],
) -> tuple[int, ...]:
    return free_sizes + basis_sizes


def _make_result_storage_spec(
    sig: TensorSignature,
    basis_sizes: tuple[int, ...],
) -> StorageSpec:
    free_sizes = tuple(int(axis.size) for axis in sig.free_axes)
    if sig.storage_kind == "hess" and sig.basis_rank == 1:
        if len(free_sizes) == 2:
            stored_shape = (1, int(basis_sizes[0]), int(free_sizes[0]), int(free_sizes[1]))
            free_axis_positions = (2, 3)
        else:
            stored_shape = (
                int(free_sizes[0]),
                int(basis_sizes[0]),
                *tuple(int(v) for v in free_sizes[1:]),
            )
            free_axis_positions = tuple(range(0, len(free_sizes)))
        basis_axis_positions = (1,)
        return StorageSpec(
            stored_shape=stored_shape,
            free_axis_positions=free_axis_positions,
            basis_axis_positions=basis_axis_positions,
            basis_sizes=basis_sizes,
            canonical_shape=_canonical_shape_from_signature(sig, free_sizes, basis_sizes),
        )
    if sig.storage_kind == "hess" and sig.basis_rank == 2:
        if len(free_sizes) == 2:
            stored_shape = (
                1,
                int(basis_sizes[0]),
                int(basis_sizes[1]),
                int(free_sizes[0]),
                int(free_sizes[1]),
            )
            free_axis_positions = (3, 4)
        else:
            stored_shape = (
                int(free_sizes[0]),
                int(basis_sizes[0]),
                int(basis_sizes[1]),
                *tuple(int(v) for v in free_sizes[1:]),
            )
            free_axis_positions = tuple(range(0, len(free_sizes)))
        basis_axis_positions = (1, 2)
        return StorageSpec(
            stored_shape=stored_shape,
            free_axis_positions=free_axis_positions,
            basis_axis_positions=basis_axis_positions,
            basis_sizes=basis_sizes,
            canonical_shape=_canonical_shape_from_signature(sig, free_sizes, basis_sizes),
        )
    stored_shape = _storage_shape_from_free_and_basis(
        free_sizes,
        basis_sizes,
        scalar_basis_as_row=(sig.tensor_rank == 0 and sig.basis_rank == 1),
    )
    if sig.tensor_rank == 0 and sig.basis_rank == 1:
        free_axis_positions: tuple[int, ...] = ()
        basis_axis_positions = (1,)
    else:
        free_axis_positions, basis_axis_positions = _storage_axis_positions(sig.tensor_rank, sig.basis_rank)
    return StorageSpec(
        stored_shape=stored_shape,
        free_axis_positions=free_axis_positions,
        basis_axis_positions=basis_axis_positions,
        basis_sizes=basis_sizes,
        canonical_shape=_canonical_shape_from_signature(sig, free_sizes, basis_sizes),
    )


def _infer_storage_spec(sig: TensorSignature, shape: tuple[int, ...]) -> StorageSpec:
    shape = tuple(int(v) for v in shape)
    if sig.basis_rank == 0:
        if sig.tensor_rank == 0:
            free_positions: tuple[int, ...] = ()
        elif sig.tensor_rank == 1 and len(shape) == 2 and 1 in shape and max(shape) > 1:
            free_positions = (0 if shape[0] > 1 else 1,)
        else:
            free_positions = tuple(range(sig.tensor_rank))
        free_sizes = tuple(
            int(axis.size if axis.size > 0 else shape[idx])
            for idx, axis in zip(free_positions, sig.free_axes, strict=True)
        )
        return StorageSpec(
            stored_shape=shape,
            free_axis_positions=free_positions,
            basis_axis_positions=(),
            basis_sizes=(),
            canonical_shape=_canonical_shape_from_signature(sig, free_sizes, ()),
        )

    if sig.tensor_rank == 0 and sig.basis_rank == 1:
        if len(shape) == 1:
            basis_sizes = (int(shape[0]),)
            return StorageSpec(
                stored_shape=shape,
                free_axis_positions=(),
                basis_axis_positions=(0,),
                basis_sizes=basis_sizes,
                canonical_shape=_canonical_shape_from_signature(sig, (), basis_sizes),
            )
        if len(shape) == 2 and int(shape[0]) == 1:
            basis_sizes = (int(shape[1]),)
            return StorageSpec(
                stored_shape=shape,
                free_axis_positions=(),
                basis_axis_positions=(1,),
                basis_sizes=basis_sizes,
                canonical_shape=_canonical_shape_from_signature(sig, (), basis_sizes),
            )

    if sig.tensor_rank == 0 and sig.basis_rank == 2:
        if len(shape) == 2:
            basis_sizes = (int(shape[0]), int(shape[1]))
            return StorageSpec(
                stored_shape=shape,
                free_axis_positions=(),
                basis_axis_positions=(0, 1),
                basis_sizes=basis_sizes,
                canonical_shape=_canonical_shape_from_signature(sig, (), basis_sizes),
            )
        if len(shape) == 3 and int(shape[0]) == 1:
            basis_sizes = (int(shape[1]), int(shape[2]))
            return StorageSpec(
                stored_shape=shape,
                free_axis_positions=(),
                basis_axis_positions=(1, 2),
                basis_sizes=basis_sizes,
                canonical_shape=_canonical_shape_from_signature(sig, (), basis_sizes),
            )

    # Legacy scalar-gradient basis layout: (1, n, d)
    if sig.tensor_rank == 1 and sig.basis_rank == 1 and len(shape) == 3 and int(shape[0]) == 1:
        free_sizes = (int(shape[2]),)
        basis_sizes = (int(shape[1]),)
        return StorageSpec(
            stored_shape=shape,
            free_axis_positions=(2,),
            basis_axis_positions=(1,),
            basis_sizes=basis_sizes,
            canonical_shape=_canonical_shape_from_signature(sig, free_sizes, basis_sizes),
        )

    # Scalar Hessian values/bases keep a legacy leading singleton component wrapper.
    # Algebraically that wrapper is not a free physical axis and must be ignored by
    # storage inference; otherwise planner lowering falls back to shape guessing for
    # dot(H(scalar), c), c·H(scalar), and the derived n^T H n paths.
    if sig.storage_kind == "hess" and sig.basis_rank == 0 and len(shape) == 3 and int(shape[0]) == 1:
        free_positions = (1, 2)
        basis_positions = ()
        free_sizes = tuple(
            int(axis.size if axis.size > 0 else shape[idx])
            for idx, axis in zip(free_positions, sig.free_axes, strict=True)
        )
        return StorageSpec(
            stored_shape=shape,
            free_axis_positions=free_positions,
            basis_axis_positions=basis_positions,
            basis_sizes=(),
            canonical_shape=_canonical_shape_from_signature(sig, free_sizes, ()),
        )

    # Current Hessian basis carriers are stored as (k, n, d, d) and mixed Hessian
    # carriers as (k, n_test, n_trial, d, d). Vector Hessians use the leading
    # component axis as a true free axis; scalar Hessians use it only as a legacy
    # singleton wrapper and therefore skip it.
    if sig.storage_kind == "hess" and sig.basis_rank == 1 and len(shape) == 4:
        free_positions = (2, 3) if int(shape[0]) == 1 and len(sig.free_axes) == 2 else (0, 2, 3)
        basis_positions = (1,)
        free_sizes = tuple(
            int(axis.size if axis.size > 0 else shape[idx])
            for idx, axis in zip(free_positions, sig.free_axes, strict=True)
        )
        basis_sizes = (int(shape[1]),)
        return StorageSpec(
            stored_shape=shape,
            free_axis_positions=free_positions,
            basis_axis_positions=basis_positions,
            basis_sizes=basis_sizes,
            canonical_shape=_canonical_shape_from_signature(sig, free_sizes, basis_sizes),
        )
    if sig.storage_kind == "hess" and sig.basis_rank == 2 and len(shape) == 5:
        free_positions = (3, 4) if int(shape[0]) == 1 and len(sig.free_axes) == 2 else (0, 3, 4)
        basis_positions = (1, 2)
        free_sizes = tuple(
            int(axis.size if axis.size > 0 else shape[idx])
            for idx, axis in zip(free_positions, sig.free_axes, strict=True)
        )
        basis_sizes = (int(shape[1]), int(shape[2]))
        return StorageSpec(
            stored_shape=shape,
            free_axis_positions=free_positions,
            basis_axis_positions=basis_positions,
            basis_sizes=basis_sizes,
            canonical_shape=_canonical_shape_from_signature(sig, free_sizes, basis_sizes),
        )

    free_positions, basis_positions = _storage_axis_positions(sig.tensor_rank, sig.basis_rank)
    if basis_positions and max(basis_positions) >= len(shape):
        raise TypeError(
            f"Stored shape {shape!r} is incompatible with tensor rank {sig.tensor_rank} "
            f"and basis rank {sig.basis_rank}."
        )
    free_sizes = tuple(
        int(axis.size if axis.size > 0 else shape[idx])
        for idx, axis in zip(free_positions, sig.free_axes, strict=True)
    )
    basis_sizes = tuple(int(shape[idx]) for idx in basis_positions)
    return StorageSpec(
        stored_shape=shape,
        free_axis_positions=free_positions,
        basis_axis_positions=basis_positions,
        basis_sizes=basis_sizes,
        canonical_shape=_canonical_shape_from_signature(sig, free_sizes, basis_sizes),
    )


def _canonical_rank1_basis_shape_and_transform(
    meta: ExpressionMeta,
    shape: tuple[int, ...],
) -> tuple[tuple[int, ...], OperandTransform]:
    tensor = meta.tensor
    if tensor.basis_rank != 1 or tensor.tensor_rank != 1:
        return shape, OperandTransform.NONE
    free_size = int(tensor.free_axes[0].size)
    if len(shape) == 3 and int(shape[0]) == 1:
        if free_size < 0 or int(shape[2]) == free_size:
            return (int(shape[2]), int(shape[1])), OperandTransform.SCALAR_GRAD_TO_VECTOR
    if len(shape) == 2:
        if free_size > 0 and int(shape[1]) == free_size and int(shape[0]) != free_size:
            return (int(shape[1]), int(shape[0])), OperandTransform.TRANSPOSE_2D
        if free_size < 0 and int(shape[0]) > int(shape[1]):
            return (int(shape[1]), int(shape[0])), OperandTransform.TRANSPOSE_2D
    return shape, OperandTransform.NONE


def _stored_expression_meta(obj: Any) -> ExpressionMeta | None:
    meta = getattr(obj, "expression_meta", None)
    if isinstance(meta, ExpressionMeta):
        return meta
    return None


def _broadcast_raw_shape(lhs_shape: tuple[int, ...], rhs_shape: tuple[int, ...]) -> tuple[int, ...]:
    from itertools import zip_longest

    la, lb = len(lhs_shape), len(rhs_shape)
    max_len = max(la, lb)
    lhs = (1,) * (max_len - la) + tuple(int(v) for v in lhs_shape)
    rhs = (1,) * (max_len - lb) + tuple(int(v) for v in rhs_shape)
    out: list[int] = []
    for lhs_dim, rhs_dim in zip_longest(lhs, rhs, fillvalue=1):
        if lhs_dim == rhs_dim:
            out.append(lhs_dim)
        elif lhs_dim == 1:
            out.append(rhs_dim)
        elif rhs_dim == 1:
            out.append(lhs_dim)
        elif lhs_dim < 0 or rhs_dim < 0:
            out.append(-1)
        else:
            raise TypeError(f"Cannot broadcast raw shapes {lhs_shape!r} and {rhs_shape!r}.")
    return tuple(out)


def _vector_axis(size: int, label: AxisLabel = AxisLabel.COMPONENT) -> tuple[TensorAxis, ...]:
    return (TensorAxis(AxisSpace.PHYSICAL, label, int(size)),)


def _matrix_axes(rows: int, cols: int) -> tuple[TensorAxis, ...]:
    return (
        TensorAxis(AxisSpace.PHYSICAL, AxisLabel.ROW, int(rows)),
        TensorAxis(AxisSpace.PHYSICAL, AxisLabel.COL, int(cols)),
    )


def _grad_axes(k: int, d: int) -> tuple[TensorAxis, ...]:
    if int(k) == 1:
        return (TensorAxis(AxisSpace.PHYSICAL, AxisLabel.DERIVATIVE, int(d)),)
    return (
        TensorAxis(AxisSpace.PHYSICAL, AxisLabel.COMPONENT, int(k)),
        TensorAxis(AxisSpace.PHYSICAL, AxisLabel.DERIVATIVE, int(d)),
    )


def _hess_axes(k: int, d1: int, d2: int) -> tuple[TensorAxis, ...]:
    if int(k) == 1:
        return (
            TensorAxis(AxisSpace.PHYSICAL, AxisLabel.HESSIAN_ROW, int(d1)),
            TensorAxis(AxisSpace.PHYSICAL, AxisLabel.HESSIAN_COL, int(d2)),
        )
    return (
        TensorAxis(AxisSpace.PHYSICAL, AxisLabel.COMPONENT, int(k)),
        TensorAxis(AxisSpace.PHYSICAL, AxisLabel.HESSIAN_ROW, int(d1)),
        TensorAxis(AxisSpace.PHYSICAL, AxisLabel.HESSIAN_COL, int(d2)),
    )


def _is_gradient_like(sig: TensorSignature) -> bool:
    return any(axis.label == AxisLabel.DERIVATIVE for axis in sig.free_axes) and not _is_hessian_like(sig)


def _is_hessian_like(sig: TensorSignature) -> bool:
    return any(axis.label in {AxisLabel.HESSIAN_ROW, AxisLabel.HESSIAN_COL} for axis in sig.free_axes)


def _is_grad_storage(sig: TensorSignature) -> bool:
    return sig.storage_kind == "grad" or _is_gradient_like(sig)


def _is_hess_storage(sig: TensorSignature) -> bool:
    return sig.storage_kind == "hess" or _is_hessian_like(sig)


def _is_plain_basis_tensor(sig: TensorSignature) -> bool:
    return sig.basis_rank == 1 and not _is_grad_storage(sig) and not _is_hess_storage(sig)


def _has_derivative_provenance(meta: ExpressionMeta) -> bool:
    return any(
        src.kind == ProvenanceKind.DERIVATIVE_CHANNELS and int(src.derivative_depth) > 0
        for src in meta.provenance.sources
    )


def _max_derivative_depth(meta: ExpressionMeta) -> int:
    if not meta.provenance.sources:
        return 0
    return max(int(src.derivative_depth) for src in meta.provenance.sources)


def _is_gradient_semantic(meta: ExpressionMeta) -> bool:
    sig = meta.tensor
    # Kernel selection must follow the *exposed tensor axes*, not merely the
    # fact that the entries depend on derivatives. Expressions like
    # ``dot(Finv.T, grad(p))`` or ``dot(grad(q), Finv)`` are transformed
    # gradients whose surviving free axis is a physical component/row/col axis,
    # not a derivative axis. Treating them as semantic gradients routes later
    # dot products through the grad-specialized backend kernels and breaks
    # transformed PSPG/ALE terms.
    if _is_grad_storage(sig):
        return True

    # Basis-backed rank-1 intermediates can lose their explicit derivative axis
    # after a transport or contraction, but they still need to route through
    # the basis-gradient kernels in subsequent dot() closures.
    return sig.basis_rank > 0 and sig.tensor_rank == 1 and _has_derivative_provenance(meta)


def _is_hessian_semantic(meta: ExpressionMeta) -> bool:
    sig = meta.tensor
    return _is_hess_storage(sig)


def _classify_product_result(
    plan: ProductPlan,
    lhs_meta: ExpressionMeta,
    rhs_meta: ExpressionMeta,
    result_meta: ExpressionMeta,
) -> LoweredResult:
    result = plan.result
    layout = MixedLayout.DEFAULT
    if result.basis_rank == 2 and result.tensor_rank == 1:
        layout = MixedLayout.COMPONENT_FIRST

    # Scalar scale/promotion keeps the algebraic tensor rank unchanged, but the
    # lowered carrier still needs derivative-aware tagging so promoted
    # scalar-gradient channels continue to lower as gradients instead of plain
    # vectors in backend codegen.
    if plan.lhs.is_scalar ^ plan.rhs.is_scalar:
        other = plan.rhs if plan.lhs.is_scalar else plan.lhs
        other_meta = rhs_meta if plan.lhs.is_scalar else lhs_meta
        is_hessian = False
        is_gradient = False
        if other.tensor_rank >= 2:
            is_hessian = _is_hessian_like(other)
            is_gradient = _is_gradient_like(other)
            if result.tensor_rank == 2 and result.basis_rank >= 1 and not is_hessian:
                is_gradient = True
        if not is_gradient and not is_hessian and _has_derivative_provenance(other_meta):
            if result.tensor_rank == 1:
                is_gradient = True
            elif other.tensor_rank == 2:
                is_hessian = True
            elif other.tensor_rank > 2:
                is_gradient = True
        is_vector = result.tensor_rank == 1 and not is_gradient and not is_hessian
        return LoweredResult(
            role=result.role,
            layout=layout,
            is_vector=is_vector,
            is_gradient=is_gradient,
            is_hessian=is_hessian,
        )

    is_hessian = False
    is_gradient = False
    if result.tensor_rank >= 2:
        if result.tensor_rank == 2 and result.basis_rank >= 1:
            is_gradient = True
        if not is_gradient and _has_derivative_provenance(result_meta):
            if result.tensor_rank == 2:
                is_hessian = True
            elif result.tensor_rank > 2:
                is_gradient = True
    is_vector = result.tensor_rank == 1 and not is_hessian
    return LoweredResult(
        role=result.role,
        layout=layout,
        is_vector=is_vector,
        is_gradient=is_gradient,
        is_hessian=is_hessian,
    )


def _classify_sum_result(plan: SumPlan) -> LoweredResult:
    result_meta = plan.result
    return _classify_expression_result(result_meta)


def _classify_expression_result(result_meta: ExpressionMeta) -> LoweredResult:
    result = result_meta.tensor
    layout = MixedLayout.DEFAULT
    if result.basis_rank == 2 and result.tensor_rank == 1 and result.free_axes:
        first_label = result.free_axes[0].label
        if first_label == AxisLabel.COMPONENT:
            layout = MixedLayout.COMPONENT_FIRST
        elif first_label in {AxisLabel.DERIVATIVE, AxisLabel.HESSIAN_ROW, AxisLabel.HESSIAN_COL}:
            layout = MixedLayout.COMPONENT_LAST

    is_hessian = _is_hessian_semantic(result_meta)
    is_gradient = (not is_hessian) and _is_gradient_semantic(result_meta)
    is_vector = result.tensor_rank == 1 and not is_gradient and not is_hessian
    return LoweredResult(
        role=result.role,
        layout=layout,
        is_vector=is_vector,
        is_gradient=is_gradient,
        is_hessian=is_hessian,
    )


def _kernel_value_shape_for_scalar_result(
    sig: TensorSignature,
    result_storage: StorageSpec,
) -> tuple[str, tuple[int, ...]]:
    if sig.tensor_rank != 0:
        shape = tuple(int(v) for v in result_storage.stored_shape)
        kind = "scalar" if not shape else _kernel_kind_from_lowering(sig, _classify_expression_result(ExpressionMeta(sig, ProvenanceSignature(()))), shape)
        return kind, shape
    if sig.basis_rank == 0:
        return "scalar", ()
    if sig.basis_rank == 1:
        return "vec", (int(result_storage.basis_sizes[0]),)
    if sig.basis_rank == 2:
        return "mat", tuple(int(v) for v in result_storage.basis_sizes)
    shape = tuple(int(v) for v in result_storage.stored_shape)
    kind = "scalar" if not shape else "mat"
    return kind, shape


def _classify_dot_result(plan: DotPlan, lhs: TensorSignature, rhs: TensorSignature) -> LoweredResult:
    result = plan.result
    layout = MixedLayout.DEFAULT

    if (
        lhs.basis_rank == 2
        and _is_gradient_like(lhs)
        and rhs.tensor_rank == 1
        and result.basis_rank == 2
        and result.tensor_rank == 1
        and result.free_axes[0].label == AxisLabel.COMPONENT
    ):
        layout = MixedLayout.COMPONENT_FIRST
    elif (
        lhs.tensor_rank == 1
        and rhs.basis_rank == 2
        and _is_gradient_like(rhs)
        and result.basis_rank == 2
        and result.tensor_rank == 1
        and result.free_axes[0].label == AxisLabel.DERIVATIVE
    ):
        layout = MixedLayout.COMPONENT_LAST
    elif (
        lhs.basis_rank == rhs.basis_rank == 1
        and result.basis_rank == 2
        and result.tensor_rank == 1
        and _is_gradient_like(lhs)
        and not _is_gradient_like(rhs)
        and result.free_axes[0].label == AxisLabel.COMPONENT
    ):
        layout = MixedLayout.COMPONENT_FIRST
    elif (
        lhs.basis_rank == rhs.basis_rank == 1
        and result.basis_rank == 2
        and result.tensor_rank == 1
        and not _is_gradient_like(lhs)
        and _is_gradient_like(rhs)
        and result.free_axes[0].label == AxisLabel.DERIVATIVE
    ):
        layout = MixedLayout.COMPONENT_LAST

    is_vector = False
    is_gradient = False
    is_hessian = False

    if plan.kind in {OperationKind.DOT_HESSIAN_VECTOR, OperationKind.DOT_VECTOR_HESSIAN}:
        # One contraction against a spatial vector destroys the "full Hessian"
        # lowering contract. Rank-1 results behave like vectors; rank-2 results
        # still carry one derivative-like physical axis and must lower through
        # the gradient/matrix paths, not plain matrix mass-dot paths.
        if result.tensor_rank == 1:
            is_vector = True
        elif result.tensor_rank >= 2:
            is_gradient = True
        return LoweredResult(
            role=result.role,
            layout=layout,
            is_vector=is_vector,
            is_gradient=is_gradient,
            is_hessian=False,
        )

    result_labels = {axis.label for axis in result.free_axes}
    has_hessian_row = AxisLabel.HESSIAN_ROW in result_labels
    has_hessian_col = AxisLabel.HESSIAN_COL in result_labels
    has_remaining_second_derivative_axis = has_hessian_row or has_hessian_col

    if result.tensor_rank >= 2:
        # A full Hessian requires two second-derivative axes to survive. After one
        # contraction, the remaining Hessian axis is semantically first-derivative-like
        # and should lower through the same paths as gradients/matrices, not Hessians.
        if has_hessian_row and has_hessian_col:
            is_hessian = True
        elif has_remaining_second_derivative_axis or _is_gradient_like(lhs) or _is_gradient_like(rhs):
            is_gradient = True
    elif result.tensor_rank == 1:
        is_vector = True

    return LoweredResult(
        role=result.role,
        layout=layout,
        is_vector=is_vector,
        is_gradient=is_gradient,
        is_hessian=is_hessian,
    )


def _dot_requires_mixed_basis_swap(plan: DotPlan) -> bool:
    if plan.lhs.basis_rank != 1 or plan.rhs.basis_rank != 1 or plan.result.basis_rank != 2:
        return False
    raw_basis_axes = plan.lhs.basis_axes + plan.rhs.basis_axes
    return raw_basis_axes != plan.result.basis_axes


def _kernel_kind_from_lowering(
    sig: TensorSignature,
    lowered: LoweredResult,
    stored_shape: tuple[int, ...],
) -> str:
    if sig.basis_rank == 2:
        if sig.tensor_rank == 0:
            if len(stored_shape) == 1:
                return "vec"
            if len(stored_shape) == 2:
                return "mat"
        return "mixed"
    if lowered.is_hessian:
        return "hess"
    # Basis-carrying rank-2+ tensors are stored in the JIT/C++ backends using
    # the same component-stack convention as gradients: one (n x d...) matrix
    # per leading physical component/free axis. Reporting them as plain "mat"
    # causes later sum/product/dot lowering to route through MatrixXd shortcuts
    # even though the runtime value is actually std::vector<Eigen::MatrixXd>.
    if sig.basis_rank == 1 and (lowered.is_gradient or sig.tensor_rank >= 2):
        return "grad"
    if sig.tensor_rank == 1 and len(stored_shape) == 2:
        return "mat"
    if lowered.is_gradient:
        return "grad"
    if sig.tensor_rank == 0:
        if sig.basis_rank == 0:
            return "scalar"
        return "vec" if len(stored_shape) == 1 else "mat"
    if sig.basis_rank == 0 and sig.tensor_rank == 1 and len(stored_shape) == 1:
        return "vec"
    return "mat"


def _result_basis_sizes(
    result_axes: tuple[BasisLabel, ...],
    lhs_sig: TensorSignature,
    lhs_storage: StorageSpec,
    rhs_sig: TensorSignature,
    rhs_storage: StorageSpec,
) -> tuple[int, ...]:
    lhs_map = _basis_size_map(lhs_sig, lhs_storage)
    rhs_map = _basis_size_map(rhs_sig, rhs_storage)
    sizes: list[int] = []
    for axis in result_axes:
        lhs_size = lhs_map.get(axis)
        rhs_size = rhs_map.get(axis)
        if lhs_size is None and rhs_size is None:
            raise TypeError(f"Missing basis extent for {axis.value}.")
        if lhs_size is not None and rhs_size is not None and lhs_size != rhs_size:
            raise TypeError(
                f"Incompatible basis extents for {axis.value}: {lhs_size} vs {rhs_size}."
            )
        sizes.append(int(lhs_size if lhs_size is not None else rhs_size))
    return tuple(sizes)


def _is_scalar_basis(sig: TensorSignature) -> bool:
    return sig.tensor_rank == 0 and sig.basis_rank == 1


def _is_mixed_scalar(sig: TensorSignature) -> bool:
    return sig.tensor_rank == 0 and sig.basis_rank == 2


def _is_value_rank1(sig: TensorSignature) -> bool:
    return sig.tensor_rank == 1 and sig.basis_rank == 0


def _is_value_rank2(sig: TensorSignature) -> bool:
    return sig.tensor_rank == 2 and sig.basis_rank == 0


class TensorRuleEngine:
    @staticmethod
    def infer_signature(obj: Any, *, spatial_dim: int = 2) -> TensorSignature:
        stored_meta = _stored_expression_meta(obj)
        if stored_meta is not None:
            return stored_meta.tensor
        if isinstance(obj, (numbers.Real, np.number)):
            return TensorSignature((), (), "scalar", (), "const", source="scalar")

        if hasattr(obj, "kind") or hasattr(obj, "is_gradient") or hasattr(obj, "is_vector") or hasattr(obj, "is_hessian"):
            return TensorRuleEngine._infer_from_codegen_item(obj, spatial_dim=spatial_dim)

        cls_name = obj.__class__.__name__
        if cls_name in {"VecOpInfo", "GradOpInfo", "HessOpInfo"}:
            return TensorRuleEngine._infer_from_runtime_opinfo(obj, spatial_dim=spatial_dim)

        arr = np.asarray(obj)
        if arr.ndim == 0:
            return TensorSignature((), (), "scalar", (), "const", source="ndarray")
        if arr.ndim == 1:
            return TensorSignature(_vector_axis(arr.shape[0]), (), "vec", tuple(int(v) for v in arr.shape), "const", source="ndarray")
        if arr.ndim == 2:
            return TensorSignature(_matrix_axes(arr.shape[0], arr.shape[1]), (), "mat", tuple(int(v) for v in arr.shape), "const", source="ndarray")
        raise TypeError(f"Cannot infer tensor signature for {type(obj).__name__} with shape {arr.shape}.")

    @staticmethod
    def infer_expression_meta(obj: Any, *, spatial_dim: int = 2) -> ExpressionMeta:
        stored_meta = _stored_expression_meta(obj)
        if stored_meta is not None:
            return stored_meta
        tensor = TensorRuleEngine.infer_signature(obj, spatial_dim=spatial_dim)
        provenance = TensorRuleEngine._infer_provenance(obj, tensor=tensor)
        return ExpressionMeta(tensor=tensor, provenance=provenance)

    @staticmethod
    def _merge_binary_expression_meta(
        lhs_obj: Any,
        rhs_obj: Any,
        result_tensor: TensorSignature,
        *,
        spatial_dim: int,
    ) -> ExpressionMeta:
        lhs_meta = TensorRuleEngine.infer_expression_meta(lhs_obj, spatial_dim=spatial_dim)
        rhs_meta = TensorRuleEngine.infer_expression_meta(rhs_obj, spatial_dim=spatial_dim)
        provenance = _project_provenance_to_tensor(
            _merge_provenance(lhs_meta.provenance, rhs_meta.provenance),
            result_tensor,
        )
        return ExpressionMeta(
            tensor=result_tensor,
            provenance=provenance,
        )

    @staticmethod
    def _infer_from_runtime_opinfo(obj: Any, *, spatial_dim: int) -> TensorSignature:
        cls_name = obj.__class__.__name__
        role = str(getattr(obj, "role", "") or "")
        shape = _shape_of(getattr(obj, "data", obj))
        basis_axes = _basis_axes_for_role(role, mixed=(role == "mixed"))

        if cls_name == "VecOpInfo":
            if role in {"scalar"} and len(shape) == 0:
                return TensorSignature((), (), "scalar", shape, role, source=cls_name)
            if role in {"scalar"} and len(shape) == 1 and shape[0] == 1:
                return TensorSignature((), (), "scalar", shape, role, source=cls_name)
            if role == "mixed":
                if len(shape) == 2:
                    # Mixed scalar carriers/basis blocks store only basis axes, even when
                    # one of them is a legacy singleton wrapper such as (1, n).
                    return TensorSignature((), basis_axes, "mixed", shape, role, source=cls_name)
                ncomp = shape[0] if len(shape) >= 1 else 1
                free_axes = () if int(ncomp) == 1 else _vector_axis(ncomp)
                return TensorSignature(free_axes, basis_axes, "mixed", shape, role, source=cls_name)
            if role in {"trial", "test", "trial_n", "test_n"}:
                if len(shape) == 1:
                    return TensorSignature((), basis_axes, "vec", shape, role, source=cls_name)
                if len(shape) == 2 and int(shape[0]) == 1:
                    return TensorSignature((), basis_axes, "vec", shape, role, source=cls_name)
                if len(shape) >= 2:
                    axis_size = _infer_basis_vector_axis_size(shape, spatial_dim)
                    if axis_size is not None:
                        return TensorSignature(_vector_axis(axis_size), basis_axes, "vec", shape, role, source=cls_name)
                    return TensorSignature(_vector_axis(shape[0]), basis_axes, "vec", shape, role, source=cls_name)
            if len(shape) == 0:
                return TensorSignature((), basis_axes, "scalar", shape, role, source=cls_name)
            if len(shape) == 1 and int(shape[0]) == 1:
                return TensorSignature((), basis_axes, "scalar", shape, role, source=cls_name)
            if len(shape) == 1:
                return TensorSignature(_vector_axis(shape[0]), basis_axes, "vec", shape, role, source=cls_name)
            if len(shape) == 2 and int(shape[0]) == 1:
                return TensorSignature(_vector_axis(shape[1]), basis_axes, "mat", shape, role, source=cls_name)
            if len(shape) == 2:
                return TensorSignature(_matrix_axes(shape[0], shape[1]), basis_axes, "mat", shape, role, source=cls_name)

        if cls_name == "GradOpInfo":
            if role == "mixed" and len(shape) == 4:
                free_axes = _grad_axes(shape[0], shape[3])
                return TensorSignature(free_axes, basis_axes, "grad", shape, role, source=cls_name)
            if len(shape) == 2:
                axis_size = _infer_basis_vector_axis_size(shape, spatial_dim)
                if axis_size is not None:
                    return TensorSignature(_vector_axis(axis_size), basis_axes, "grad", shape, role, source=cls_name)
                return TensorSignature(_grad_axes(shape[0], shape[1]), basis_axes, "grad", shape, role, source=cls_name)
            if len(shape) == 3:
                if role in {"trial", "test", "trial_n", "test_n"} and int(shape[0]) == 1:
                    return TensorSignature(_vector_axis(shape[2]), basis_axes, "grad", shape, role, source=cls_name)
                return TensorSignature(_grad_axes(shape[0], shape[2]), basis_axes, "grad", shape, role, source=cls_name)

        if cls_name == "HessOpInfo":
            if role == "mixed" and len(shape) == 5:
                free_axes = _hess_axes(shape[0], shape[3], shape[4])
                return TensorSignature(free_axes, basis_axes, "hess", shape, role, source=cls_name)
            if len(shape) == 3:
                return TensorSignature(_hess_axes(shape[0], shape[1], shape[2]), basis_axes, "hess", shape, role, source=cls_name)
            if len(shape) == 4:
                return TensorSignature(_hess_axes(shape[0], shape[2], shape[3]), basis_axes, "hess", shape, role, source=cls_name)

        raise TypeError(f"Cannot infer runtime tensor signature for {cls_name} with role={role!r} shape={shape!r}.")

    @staticmethod
    def _infer_from_codegen_item(item: Any, *, spatial_dim: int) -> TensorSignature:
        role = str(getattr(item, "role", "") or "")
        shape = _shape_of(item)
        kind = str(getattr(item, "kind", "") or "")
        if not kind:
            if getattr(item, "is_hessian", False):
                kind = "hess"
            elif role == "mixed" and len(shape) in {3, 4, 5}:
                kind = "mixed"
            elif getattr(item, "is_gradient", False):
                kind = "grad"
            elif getattr(item, "is_vector", False):
                kind = "vec"
            elif len(shape) == 0:
                kind = "scalar"
            else:
                kind = "mat"

        basis_axes = _basis_axes_for_role(role, mixed=(role == "mixed"))

        if role == "mixed" and len(shape) == 2:
            # Mixed scalar carriers/basis blocks such as (n_test, n_trial) or legacy
            # singleton-wrapped (1, n_trial) store only basis axes and must not invent a
            # fake free physical axis from the leading dimension.
            return TensorSignature((), basis_axes, "mixed", shape, role, source="stack")

        if kind == "scalar":
            return TensorSignature((), basis_axes, kind, shape, role, source="stack")

        if kind == "vec":
            if role in {"test", "trial"} and len(shape) >= 2 and int(shape[0]) == 1:
                return TensorSignature((), basis_axes, kind, shape, role, source="stack")
            if len(shape) == 1 and int(shape[0]) == 1:
                return TensorSignature((), basis_axes, kind, shape, role, source="stack")
            if len(shape) >= 1:
                size = shape[0] if len(shape) == 1 else shape[0]
                if len(shape) == 2:
                    axis_size = _infer_basis_vector_axis_size(shape, spatial_dim)
                    if axis_size is not None:
                        size = axis_size
                return TensorSignature(_vector_axis(size), basis_axes, kind, shape, role, source="stack")

        if kind == "mat":
            if role in {"test", "trial"} and len(shape) >= 2:
                if int(shape[0]) == 1:
                    return TensorSignature((), basis_axes, kind, shape, role, source="stack")
                axis_size = _infer_basis_vector_axis_size(shape, spatial_dim)
                if axis_size is not None:
                    return TensorSignature(_vector_axis(axis_size), basis_axes, kind, shape, role, source="stack")
                return TensorSignature(_vector_axis(shape[0]), basis_axes, kind, shape, role, source="stack")
            if len(shape) == 1:
                if int(shape[0]) == 1:
                    return TensorSignature((), basis_axes, kind, shape, role, source="stack")
                return TensorSignature(_vector_axis(shape[0]), basis_axes, kind, shape, role, source="stack")
            if len(shape) == 2:
                if 1 in shape and max(shape) > 1:
                    return TensorSignature(_vector_axis(max(shape)), basis_axes, kind, shape, role, source="stack")
                return TensorSignature(_matrix_axes(shape[0], shape[1]), basis_axes, kind, shape, role, source="stack")

        if kind == "grad":
            if len(shape) == 1 and role in {"test", "trial", "test_n", "trial_n"}:
                return TensorSignature((), basis_axes, kind, shape, role, source="stack")
            if len(shape) == 2:
                if role in {"test", "trial", "test_n", "trial_n"} and int(shape[0]) == 1:
                    return TensorSignature((), basis_axes, kind, shape, role, source="stack")
                axis_size = _infer_basis_vector_axis_size(shape, spatial_dim)
                if axis_size is not None:
                    return TensorSignature(_vector_axis(axis_size), basis_axes, kind, shape, role, source="stack")
                return TensorSignature(_grad_axes(shape[0], shape[1]), basis_axes, kind, shape, role, source="stack")
            if len(shape) == 3:
                if role in {"test", "trial"} and int(shape[0]) == 1:
                    return TensorSignature(_vector_axis(shape[2]), basis_axes, kind, shape, role, source="stack")
                return TensorSignature(_grad_axes(shape[0], shape[2]), basis_axes, kind, shape, role, source="stack")
            if len(shape) == 4:
                return TensorSignature(_grad_axes(shape[0], shape[3]), basis_axes, kind, shape, role, source="stack")

        if kind in {"mixed"}:
            if len(shape) == 3:
                free_axes = () if int(shape[0]) == 1 else _vector_axis(shape[0])
                return TensorSignature(free_axes, (BasisLabel.TEST, BasisLabel.TRIAL), kind, shape, role, source="stack")
            if len(shape) == 4:
                return TensorSignature(_grad_axes(shape[0], shape[3]), (BasisLabel.TEST, BasisLabel.TRIAL), kind, shape, role, source="stack")

        if kind == "hess":
            if len(shape) == 3 and not basis_axes:
                return TensorSignature(_hess_axes(shape[0], shape[1], shape[2]), basis_axes, kind, shape, role, source="stack")
            if len(shape) == 4:
                return TensorSignature(_hess_axes(shape[0], shape[2], shape[3]), basis_axes, kind, shape, role, source="stack")
            if len(shape) == 5:
                return TensorSignature(_hess_axes(shape[0], shape[3], shape[4]), (BasisLabel.TEST, BasisLabel.TRIAL), kind, shape, role, source="stack")

        raise TypeError(f"Cannot infer stack tensor signature for kind={kind!r} role={role!r} shape={shape!r}.")

    @staticmethod
    def _infer_provenance(obj: Any, *, tensor: TensorSignature) -> ProvenanceSignature:
        if isinstance(obj, (numbers.Real, np.number)):
            return ProvenanceSignature((FieldSource("", (), ProvenanceKind.CONSTANT, 0, "const", "scalar"),))

        arr = None
        try:
            arr = np.asarray(obj)
        except Exception:
            arr = None
        if arr is not None and arr.ndim <= 2 and not hasattr(obj, "field_names") and not hasattr(obj, "parent_name") and not hasattr(obj, "parent"):
            return ProvenanceSignature((FieldSource("", (), ProvenanceKind.CONSTANT, 0, "const", "ndarray"),))

        role = str(getattr(obj, "role", "") or "")
        fields = tuple(str(f) for f in (getattr(obj, "field_names", None) or ()))
        parent = str(
            getattr(obj, "parent_name", "")
            or getattr(obj, "parent", "")
            or getattr(obj, "name", "")
            or ""
        )
        derivative_depth = 0
        if tensor.storage_kind == "grad":
            derivative_depth = 1
        elif tensor.storage_kind == "hess":
            derivative_depth = 2

        if not fields and role in {"", "none", "const", "value", "scalar"}:
            return ProvenanceSignature((FieldSource(parent, (), ProvenanceKind.CONSTANT, derivative_depth, role or "const", tensor.source),))
        if not fields:
            return ProvenanceSignature((FieldSource(parent, (), ProvenanceKind.UNKNOWN, derivative_depth, role, tensor.source),))

        if derivative_depth > 0 and len(fields) == 1:
            kind = ProvenanceKind.DERIVATIVE_CHANNELS
        else:
            kind = ProvenanceKind.FIELD_COMPONENTS

        return ProvenanceSignature((FieldSource(parent, fields, kind, derivative_depth, role, tensor.source),))

    @staticmethod
    def plan_dot(lhs_obj: Any, rhs_obj: Any, *, spatial_dim: int = 2) -> DotPlan:
        lhs = TensorRuleEngine.infer_signature(lhs_obj, spatial_dim=spatial_dim)
        rhs = TensorRuleEngine.infer_signature(rhs_obj, spatial_dim=spatial_dim)

        if lhs.tensor_rank == 0 or rhs.tensor_rank == 0:
            # Match UFL/FEniCS tensor calculus semantics: dot with a scalar
            # degenerates to scalar multiplication, so only the non-scalar
            # operand contributes free physical axes. This is also the correct
            # lowering for scalar basis rows such as (1, n), where dot(q, p)
            # must form the mixed mass block instead of failing shape checks.
            basis_axes = _canonical_basis_axes(lhs, rhs)
            role = _role_from_basis_axes(basis_axes, lhs.role, rhs.role)
            if lhs.tensor_rank == rhs.tensor_rank == 0:
                storage_kind = "mixed" if basis_axes else "scalar"
                raw_shape: tuple[int, ...] = ()
                free_axes: tuple[TensorAxis, ...] = ()
            else:
                other = rhs if lhs.tensor_rank == 0 else lhs
                storage_kind = other.storage_kind
                raw_shape = other.raw_shape
                free_axes = other.free_axes
            result = TensorSignature(
                free_axes=free_axes,
                basis_axes=basis_axes,
                storage_kind=storage_kind,
                raw_shape=raw_shape,
                role=role,
                source="dot_scale",
            )
            return DotPlan(
                kind=OperationKind.DOT_TENSOR_TENSOR,
                lhs=lhs,
                rhs=rhs,
                result=result,
            )

        left_axis = lhs.free_axes[-1]
        right_axis = rhs.free_axes[0]
        if not left_axis.compatible_with(right_axis):
            raise TypeError(
                f"Incompatible dot axes: {left_axis.space}:{left_axis.label}:{left_axis.size} vs "
                f"{right_axis.space}:{right_axis.label}:{right_axis.size}."
            )

        free_axes = lhs.free_axes[:-1] + rhs.free_axes[1:]
        basis_axes = _canonical_basis_axes(lhs, rhs)
        role = _role_from_basis_axes(basis_axes, lhs.role, rhs.role)

        if lhs.storage_kind == "hess":
            kind = OperationKind.DOT_HESSIAN_VECTOR
        elif rhs.storage_kind == "hess":
            kind = OperationKind.DOT_VECTOR_HESSIAN
        elif lhs.tensor_rank == rhs.tensor_rank == 1:
            kind = OperationKind.DOT_VECTOR_VECTOR
        else:
            kind = OperationKind.DOT_TENSOR_TENSOR

        result = TensorSignature(
            free_axes=free_axes,
            basis_axes=basis_axes,
            storage_kind="dot_result",
            raw_shape=(),
            role=role,
            source="dot",
        )
        return DotPlan(kind=kind, lhs=lhs, rhs=rhs, result=result)

    @staticmethod
    def plan_dot_lowering(lhs_obj: Any, rhs_obj: Any, *, spatial_dim: int = 2) -> DotLoweringPlan:
        algebra = TensorRuleEngine.plan_dot(lhs_obj, rhs_obj, spatial_dim=spatial_dim)
        lhs_meta = TensorRuleEngine.infer_expression_meta(lhs_obj, spatial_dim=spatial_dim)
        rhs_meta = TensorRuleEngine.infer_expression_meta(rhs_obj, spatial_dim=spatial_dim)
        lhs_storage = _infer_storage_spec(algebra.lhs, _shape_of(lhs_obj))
        rhs_storage = _infer_storage_spec(algebra.rhs, _shape_of(rhs_obj))
        result_basis_sizes = _result_basis_sizes(
            algebra.result.basis_axes,
            algebra.lhs,
            lhs_storage,
            algebra.rhs,
            rhs_storage,
        )
        meta = TensorRuleEngine._merge_binary_expression_meta(
            lhs_obj,
            rhs_obj,
            algebra.result,
            spatial_dim=spatial_dim,
        )
        return DotLoweringPlan(
            algebra=algebra,
            meta=meta,
            result=_classify_dot_result(algebra, algebra.lhs, algebra.rhs),
            lhs_storage=lhs_storage,
            rhs_storage=rhs_storage,
            result_storage=_make_result_storage_spec(algebra.result, result_basis_sizes),
            swap_mixed_basis_axes=_dot_requires_mixed_basis_swap(algebra),
        )

    @staticmethod
    def plan_dot_kernel(lhs_obj: Any, rhs_obj: Any, *, spatial_dim: int = 2) -> DotKernelPlan:
        lowering = TensorRuleEngine.plan_dot_lowering(lhs_obj, rhs_obj, spatial_dim=spatial_dim)
        lhs = lowering.algebra.lhs
        rhs = lowering.algebra.rhs
        lhs_meta = TensorRuleEngine.infer_expression_meta(lhs_obj, spatial_dim=spatial_dim)
        rhs_meta = TensorRuleEngine.infer_expression_meta(rhs_obj, spatial_dim=spatial_dim)
        lhs_is_grad = _is_gradient_semantic(lhs_meta)
        rhs_is_grad = _is_gradient_semantic(rhs_meta)
        lhs_is_hess = _is_hessian_semantic(lhs_meta)
        rhs_is_hess = _is_hessian_semantic(rhs_meta)
        lhs_basis_tensor = lhs.basis_rank == 1 and lhs.tensor_rank > 1 and not lhs_is_hess
        rhs_basis_tensor = rhs.basis_rank == 1 and rhs.tensor_rank > 1 and not rhs_is_hess
        lhs_value_tensor = lhs.basis_rank == 0 and lhs.tensor_rank > 1 and not lhs_is_hess
        rhs_value_tensor = rhs.basis_rank == 0 and rhs.tensor_rank > 1 and not rhs_is_hess

        if (
            lhs.basis_rank == 1
            and rhs.basis_rank == 1
            and lhs.tensor_rank == 1
            and rhs.tensor_rank == 1
            and not lhs_is_grad
            and not rhs_is_grad
            and not lhs_is_hess
            and not rhs_is_hess
        ):
            return DotKernelPlan(lowering=lowering, case=DotKernelCase.BASIS_BASIS_MASS)

        if (
            lhs.basis_rank == 1
            and rhs.basis_rank == 0
            and rhs.tensor_rank == 1
            and (lhs_is_grad or lhs_basis_tensor)
        ):
            return DotKernelPlan(lowering=lowering, case=DotKernelCase.BASIS_GRAD_DOT_VALUE_VECTOR)

        if (
            lhs.basis_rank == 0
            and rhs.basis_rank == 1
            and lhs.tensor_rank == 1
            and (rhs_is_grad or rhs_basis_tensor)
        ):
            return DotKernelPlan(lowering=lowering, case=DotKernelCase.VALUE_VECTOR_DOT_BASIS_GRAD)

        if (
            lhs.basis_rank == 0
            and rhs.basis_rank == 1
            and rhs.tensor_rank == 1
            and (lhs_is_grad or lhs_value_tensor)
            and not rhs_is_grad
            and not rhs_is_hess
        ):
            return DotKernelPlan(lowering=lowering, case=DotKernelCase.VALUE_GRAD_DOT_BASIS_VECTOR)

        if (
            lhs.basis_rank == 1
            and rhs.basis_rank == 0
            and lhs.tensor_rank == 1
            and not lhs_is_grad
            and not lhs_is_hess
            and (rhs_is_grad or rhs_value_tensor)
        ):
            return DotKernelPlan(lowering=lowering, case=DotKernelCase.BASIS_VECTOR_DOT_VALUE_GRAD)

        if (
            lhs.basis_rank == 1
            and rhs.basis_rank == 1
            and rhs.tensor_rank == 1
            and (lhs_is_grad or lhs_basis_tensor)
            and not rhs_is_grad
            and not rhs_is_hess
        ):
            return DotKernelPlan(lowering=lowering, case=DotKernelCase.BASIS_GRAD_DOT_BASIS_VECTOR)

        if (
            lhs.basis_rank == 1
            and rhs.basis_rank == 1
            and lhs.tensor_rank == 1
            and not lhs_is_grad
            and not lhs_is_hess
            and (rhs_is_grad or rhs_basis_tensor)
        ):
            return DotKernelPlan(lowering=lowering, case=DotKernelCase.BASIS_VECTOR_DOT_BASIS_GRAD)

        return DotKernelPlan(lowering=lowering, case=DotKernelCase.GENERIC_CONTRACT_LAST_FIRST)

    @staticmethod
    def plan_dot_value_spec(lhs_obj: Any, rhs_obj: Any, *, spatial_dim: int = 2) -> KernelValueSpec:
        lowering = TensorRuleEngine.plan_dot_lowering(lhs_obj, rhs_obj, spatial_dim=spatial_dim)
        shape = tuple(int(v) for v in lowering.result_storage.stored_shape)
        return KernelValueSpec(
            kind=_kernel_kind_from_lowering(lowering.algebra.result, lowering.result, shape),
            role=lowering.result.role,
            shape=shape,
            layout=lowering.result.layout,
            is_vector=lowering.result.is_vector,
            is_gradient=lowering.result.is_gradient,
            is_hessian=lowering.result.is_hessian,
            meta=lowering.meta,
        )

    @staticmethod
    def plan_inner(lhs_obj: Any, rhs_obj: Any, *, spatial_dim: int = 2) -> InnerPlan:
        lhs = TensorRuleEngine.infer_expression_meta(lhs_obj, spatial_dim=spatial_dim)
        rhs = TensorRuleEngine.infer_expression_meta(rhs_obj, spatial_dim=spatial_dim)

        lhs_tensor = lhs.tensor
        rhs_tensor = rhs.tensor

        if lhs_tensor.tensor_rank != rhs_tensor.tensor_rank:
            raise TypeError(
                f"Cannot inner tensor ranks {lhs_tensor.tensor_rank} and {rhs_tensor.tensor_rank} "
                f"for shapes {lhs_tensor.raw_shape!r} and {rhs_tensor.raw_shape!r}."
            )
        if len(lhs_tensor.free_axes) != len(rhs_tensor.free_axes):
            raise TypeError("Cannot inner tensors with different free-axis counts.")
        for l_axis, r_axis in zip(lhs_tensor.free_axes, rhs_tensor.free_axes, strict=True):
            if not l_axis.compatible_with(r_axis):
                raise TypeError(
                    f"Incompatible inner axes: {l_axis.space}:{l_axis.label}:{l_axis.size} vs "
                    f"{r_axis.space}:{r_axis.label}:{r_axis.size}."
                )

        basis_axes = _canonical_basis_axes(lhs_tensor, rhs_tensor)
        role = _role_from_basis_axes(basis_axes, lhs_tensor.role, rhs_tensor.role)
        result_tensor = TensorSignature(
            free_axes=(),
            basis_axes=basis_axes,
            storage_kind="inner_result",
            raw_shape=(),
            role=role,
            source="inner",
        )
        return InnerPlan(
            kind=OperationKind.INNER_FULL,
            lhs=lhs,
            rhs=rhs,
            result=ExpressionMeta(
                tensor=result_tensor,
                provenance=_project_provenance_to_tensor(
                    _merge_provenance(lhs.provenance, rhs.provenance),
                    result_tensor,
                ),
            ),
        )

    @staticmethod
    def plan_inner_lowering(lhs_obj: Any, rhs_obj: Any, *, spatial_dim: int = 2) -> InnerLoweringPlan:
        algebra = TensorRuleEngine.plan_inner(lhs_obj, rhs_obj, spatial_dim=spatial_dim)
        lhs_storage = _infer_storage_spec(algebra.lhs.tensor, _shape_of(lhs_obj))
        rhs_storage = _infer_storage_spec(algebra.rhs.tensor, _shape_of(rhs_obj))
        result_basis_sizes = _result_basis_sizes(
            algebra.result.tensor.basis_axes,
            algebra.lhs.tensor,
            lhs_storage,
            algebra.rhs.tensor,
            rhs_storage,
        )
        return InnerLoweringPlan(
            algebra=algebra,
            result=_classify_expression_result(algebra.result),
            lhs_storage=lhs_storage,
            rhs_storage=rhs_storage,
            result_storage=_make_result_storage_spec(algebra.result.tensor, result_basis_sizes),
        )

    @staticmethod
    def plan_inner_value_spec(lhs_obj: Any, rhs_obj: Any, *, spatial_dim: int = 2) -> KernelValueSpec:
        lowering = TensorRuleEngine.plan_inner_lowering(lhs_obj, rhs_obj, spatial_dim=spatial_dim)
        kind, shape = _kernel_value_shape_for_scalar_result(
            lowering.algebra.result.tensor,
            lowering.result_storage,
        )
        return KernelValueSpec(
            kind=kind,
            role=lowering.result.role,
            shape=shape,
            layout=lowering.result.layout,
            is_vector=lowering.result.is_vector,
            is_gradient=lowering.result.is_gradient,
            is_hessian=lowering.result.is_hessian,
            meta=lowering.algebra.result,
        )

    @staticmethod
    def plan_division(lhs_obj: Any, rhs_obj: Any, *, spatial_dim: int = 2) -> DivisionPlan:
        lhs = TensorRuleEngine.infer_expression_meta(lhs_obj, spatial_dim=spatial_dim)
        rhs = TensorRuleEngine.infer_expression_meta(rhs_obj, spatial_dim=spatial_dim)

        lhs_tensor = lhs.tensor
        rhs_tensor = rhs.tensor
        basis_axes = _canonical_basis_axes(lhs_tensor, rhs_tensor)
        role = _role_from_basis_axes(basis_axes, lhs_tensor.role, rhs_tensor.role)

        if rhs_tensor.is_scalar:
            carrier = lhs_tensor
        elif lhs_tensor.is_scalar:
            carrier = rhs_tensor
        else:
            if lhs_tensor.tensor_rank != rhs_tensor.tensor_rank:
                raise TypeError(
                    f"Only scalar or elementwise-compatible division is supported, got ranks "
                    f"{lhs_tensor.tensor_rank} and {rhs_tensor.tensor_rank}."
                )
            if lhs_tensor.basis_axes != rhs_tensor.basis_axes:
                raise TypeError(
                    f"Elementwise division requires matching basis signatures, got "
                    f"{lhs_tensor.basis_axes!r} and {rhs_tensor.basis_axes!r}."
                )
            for l_axis, r_axis in zip(lhs_tensor.free_axes, rhs_tensor.free_axes, strict=True):
                if not l_axis.compatible_with(r_axis):
                    raise TypeError(
                        f"Incompatible division axes: {l_axis.space}:{l_axis.label}:{l_axis.size} vs "
                        f"{r_axis.space}:{r_axis.label}:{r_axis.size}."
                    )
            carrier = lhs_tensor

        result_tensor = TensorSignature(
            free_axes=carrier.free_axes,
            basis_axes=basis_axes,
            storage_kind=carrier.storage_kind,
            raw_shape=carrier.raw_shape,
            role=role,
            source="division",
        )
        return DivisionPlan(
            kind=OperationKind.DIVIDE_GENERIC,
            lhs=lhs,
            rhs=rhs,
            result=ExpressionMeta(
                tensor=result_tensor,
                provenance=_project_provenance_to_tensor(
                    _merge_provenance(lhs.provenance, rhs.provenance),
                    result_tensor,
                ),
            ),
        )

    @staticmethod
    def plan_division_lowering(lhs_obj: Any, rhs_obj: Any, *, spatial_dim: int = 2) -> DivisionLoweringPlan:
        algebra = TensorRuleEngine.plan_division(lhs_obj, rhs_obj, spatial_dim=spatial_dim)
        lhs_storage = _infer_storage_spec(algebra.lhs.tensor, _shape_of(lhs_obj))
        rhs_storage = _infer_storage_spec(algebra.rhs.tensor, _shape_of(rhs_obj))
        result_basis_sizes = _result_basis_sizes(
            algebra.result.tensor.basis_axes,
            algebra.lhs.tensor,
            lhs_storage,
            algebra.rhs.tensor,
            rhs_storage,
        )
        if algebra.rhs.tensor.is_scalar:
            carrier_storage = lhs_storage
        elif algebra.lhs.tensor.is_scalar:
            carrier_storage = rhs_storage
        else:
            carrier_storage = lhs_storage
        result_storage = StorageSpec(
            stored_shape=carrier_storage.stored_shape,
            free_axis_positions=carrier_storage.free_axis_positions,
            basis_axis_positions=carrier_storage.basis_axis_positions,
            basis_sizes=carrier_storage.basis_sizes,
            canonical_shape=_canonical_shape_from_signature(
                algebra.result.tensor,
                tuple(int(axis.size) for axis in algebra.result.tensor.free_axes),
                carrier_storage.basis_sizes,
            ),
        )
        return DivisionLoweringPlan(
            algebra=algebra,
            result=_classify_expression_result(algebra.result),
            lhs_storage=lhs_storage,
            rhs_storage=rhs_storage,
            result_storage=result_storage,
        )

    @staticmethod
    def plan_division_value_spec(lhs_obj: Any, rhs_obj: Any, *, spatial_dim: int = 2) -> KernelValueSpec:
        lowering = TensorRuleEngine.plan_division_lowering(lhs_obj, rhs_obj, spatial_dim=spatial_dim)
        shape = tuple(int(v) for v in lowering.result_storage.stored_shape)
        return KernelValueSpec(
            kind=_kernel_kind_from_lowering(lowering.algebra.result.tensor, lowering.result, shape),
            role=lowering.result.role,
            shape=shape,
            layout=lowering.result.layout,
            is_vector=lowering.result.is_vector,
            is_gradient=lowering.result.is_gradient,
            is_hessian=lowering.result.is_hessian,
            meta=lowering.algebra.result,
        )

    @staticmethod
    def plan_sum(lhs_obj: Any, rhs_obj: Any, *, spatial_dim: int = 2) -> SumPlan:
        lhs = TensorRuleEngine.infer_expression_meta(lhs_obj, spatial_dim=spatial_dim)
        rhs = TensorRuleEngine.infer_expression_meta(rhs_obj, spatial_dim=spatial_dim)

        lhs_tensor = lhs.tensor
        rhs_tensor = rhs.tensor

        if lhs_tensor.tensor_rank != rhs_tensor.tensor_rank:
            raise TypeError(
                f"Cannot add tensor ranks {lhs_tensor.tensor_rank} and {rhs_tensor.tensor_rank} "
                f"for shapes {lhs_tensor.raw_shape!r} and {rhs_tensor.raw_shape!r}."
            )
        if lhs_tensor.basis_axes != rhs_tensor.basis_axes:
            raise TypeError(
                f"Cannot add basis signatures {lhs_tensor.basis_axes!r} and {rhs_tensor.basis_axes!r}."
            )
        if len(lhs_tensor.free_axes) != len(rhs_tensor.free_axes):
            raise TypeError("Cannot add tensors with different free-axis counts.")
        for l_axis, r_axis in zip(lhs_tensor.free_axes, rhs_tensor.free_axes, strict=True):
            if not l_axis.compatible_with(r_axis):
                raise TypeError(
                    f"Incompatible sum axes: {l_axis.space}:{l_axis.label}:{l_axis.size} vs "
                    f"{r_axis.space}:{r_axis.label}:{r_axis.size}."
                )

        storage_kind = lhs_tensor.storage_kind if lhs_tensor.storage_kind == rhs_tensor.storage_kind else "sum_result"
        try:
            raw_shape = _broadcast_raw_shape(lhs_tensor.raw_shape, rhs_tensor.raw_shape)
        except TypeError:
            lhs_canonical_shape, _ = _canonical_rank1_basis_shape_and_transform(lhs, lhs_tensor.raw_shape)
            rhs_canonical_shape, _ = _canonical_rank1_basis_shape_and_transform(rhs, rhs_tensor.raw_shape)
            if lhs_canonical_shape == rhs_canonical_shape:
                raw_shape = lhs_canonical_shape
            elif lhs_tensor.storage_kind == rhs_tensor.storage_kind:
                raise
            else:
                raw_shape = lhs_tensor.raw_shape if lhs_tensor.raw_shape == rhs_tensor.raw_shape else ()
        role = _role_from_basis_axes(lhs_tensor.basis_axes, lhs_tensor.role, rhs_tensor.role)
        result_tensor = TensorSignature(
            free_axes=lhs_tensor.free_axes,
            basis_axes=lhs_tensor.basis_axes,
            storage_kind=storage_kind,
            raw_shape=raw_shape,
            role=role,
            source="sum",
        )
        result = ExpressionMeta(
            tensor=result_tensor,
            provenance=_project_provenance_to_tensor(
                _merge_provenance(lhs.provenance, rhs.provenance),
                result_tensor,
            ),
        )
        return SumPlan(kind=OperationKind.SUM_GENERIC, lhs=lhs, rhs=rhs, result=result)

    @staticmethod
    def plan_sum_lowering(lhs_obj: Any, rhs_obj: Any, *, spatial_dim: int = 2) -> SumLoweringPlan:
        algebra = TensorRuleEngine.plan_sum(lhs_obj, rhs_obj, spatial_dim=spatial_dim)
        lhs_storage = _infer_storage_spec(algebra.lhs.tensor, _shape_of(lhs_obj))
        rhs_storage = _infer_storage_spec(algebra.rhs.tensor, _shape_of(rhs_obj))
        result_basis_sizes = _result_basis_sizes(
            algebra.result.tensor.basis_axes,
            algebra.lhs.tensor,
            lhs_storage,
            algebra.rhs.tensor,
            rhs_storage,
        )
        result_storage = _make_result_storage_spec(algebra.result.tensor, result_basis_sizes)
        lhs_shape = _shape_of(lhs_obj)
        rhs_shape = _shape_of(rhs_obj)
        lhs_canonical_shape, lhs_transform = _canonical_rank1_basis_shape_and_transform(algebra.lhs, lhs_shape)
        rhs_canonical_shape, rhs_transform = _canonical_rank1_basis_shape_and_transform(algebra.rhs, rhs_shape)

        if lhs_canonical_shape != rhs_canonical_shape:
            lhs_transform = OperandTransform.NONE
            rhs_transform = OperandTransform.NONE

        return SumLoweringPlan(
            algebra=algebra,
            lhs_storage=lhs_storage,
            rhs_storage=rhs_storage,
            result_storage=result_storage,
            lhs_transform=lhs_transform,
            rhs_transform=rhs_transform,
        )

    @staticmethod
    def plan_sum_value_spec(lhs_obj: Any, rhs_obj: Any, *, spatial_dim: int = 2) -> KernelValueSpec:
        lowering = TensorRuleEngine.plan_sum_lowering(lhs_obj, rhs_obj, spatial_dim=spatial_dim)
        result = _classify_sum_result(lowering.algebra)
        shape = tuple(int(v) for v in lowering.result_storage.stored_shape)
        return KernelValueSpec(
            kind=_kernel_kind_from_lowering(lowering.algebra.result.tensor, result, shape),
            role=result.role,
            shape=shape,
            layout=result.layout,
            is_vector=result.is_vector,
            is_gradient=result.is_gradient,
            is_hessian=result.is_hessian,
            meta=lowering.algebra.result,
        )

    @staticmethod
    def plan_product(lhs_obj: Any, rhs_obj: Any, *, spatial_dim: int = 2) -> ProductPlan:
        lhs = TensorRuleEngine.infer_signature(lhs_obj, spatial_dim=spatial_dim)
        rhs = TensorRuleEngine.infer_signature(rhs_obj, spatial_dim=spatial_dim)

        if (
            lhs.tensor_rank == rhs.tensor_rank == 0
            and lhs.basis_rank == rhs.basis_rank == 1
            and lhs.basis_axes != rhs.basis_axes
        ):
            basis_axes = _canonical_basis_axes(lhs, rhs)
            role = _role_from_basis_axes(basis_axes, lhs.role, rhs.role)
            result = TensorSignature(
                free_axes=(),
                basis_axes=basis_axes,
                storage_kind="mixed",
                raw_shape=(),
                role=role,
                source="product_outer",
            )
            return ProductPlan(kind=OperationKind.PRODUCT_OUTER, lhs=lhs, rhs=rhs, result=result)

        if lhs.is_scalar or rhs.is_scalar:
            other = rhs if lhs.is_scalar else lhs
            basis_axes = _canonical_basis_axes(lhs, rhs)
            role = _role_from_basis_axes(basis_axes, lhs.role, rhs.role)
            result = TensorSignature(
                free_axes=other.free_axes,
                basis_axes=basis_axes,
                storage_kind=other.storage_kind,
                raw_shape=other.raw_shape,
                role=role,
                source="product_scale",
            )
            kind = OperationKind.PRODUCT_PROMOTE if basis_axes and other.basis_rank == 0 else OperationKind.PRODUCT_SCALE
            return ProductPlan(kind=kind, lhs=lhs, rhs=rhs, result=result)

        basis_axes = _canonical_basis_axes(lhs, rhs)
        role = _role_from_basis_axes(basis_axes, lhs.role, rhs.role)
        free_axes = lhs.free_axes + rhs.free_axes
        storage_kind = "tensor"
        if len(free_axes) == 1:
            storage_kind = "vec"
        elif len(free_axes) == 2:
            storage_kind = "mat"
        result = TensorSignature(
            free_axes=free_axes,
            basis_axes=basis_axes,
            storage_kind=storage_kind,
            raw_shape=tuple(axis.size for axis in free_axes),
            role=role,
            source="product_tensor",
        )
        kind = OperationKind.PRODUCT_OUTER if lhs.tensor_rank == rhs.tensor_rank == 1 else OperationKind.PRODUCT_TENSOR
        return ProductPlan(kind=kind, lhs=lhs, rhs=rhs, result=result)

    @staticmethod
    def plan_product_lowering(lhs_obj: Any, rhs_obj: Any, *, spatial_dim: int = 2) -> ProductLoweringPlan:
        algebra = TensorRuleEngine.plan_product(lhs_obj, rhs_obj, spatial_dim=spatial_dim)
        lhs_meta = TensorRuleEngine.infer_expression_meta(lhs_obj, spatial_dim=spatial_dim)
        rhs_meta = TensorRuleEngine.infer_expression_meta(rhs_obj, spatial_dim=spatial_dim)
        lhs_storage = _infer_storage_spec(algebra.lhs, _shape_of(lhs_obj))
        rhs_storage = _infer_storage_spec(algebra.rhs, _shape_of(rhs_obj))
        result_basis_sizes = _result_basis_sizes(
            algebra.result.basis_axes,
            algebra.lhs,
            lhs_storage,
            algebra.rhs,
            rhs_storage,
        )
        meta = ExpressionMeta(
            tensor=algebra.result,
            provenance=_project_provenance_to_tensor(
                _merge_provenance(lhs_meta.provenance, rhs_meta.provenance),
                algebra.result,
            ),
        )
        if algebra.kind == OperationKind.PRODUCT_SCALE:
            carrier_storage = rhs_storage if algebra.lhs.is_scalar else lhs_storage
            result_storage = StorageSpec(
                stored_shape=carrier_storage.stored_shape,
                free_axis_positions=carrier_storage.free_axis_positions,
                basis_axis_positions=carrier_storage.basis_axis_positions,
                basis_sizes=carrier_storage.basis_sizes,
                canonical_shape=_canonical_shape_from_signature(
                    algebra.result,
                    tuple(int(axis.size) for axis in algebra.result.free_axes),
                    carrier_storage.basis_sizes,
                ),
            )
        else:
            result_storage = _make_result_storage_spec(algebra.result, result_basis_sizes)
        return ProductLoweringPlan(
            algebra=algebra,
            meta=meta,
            result=_classify_product_result(algebra, lhs_meta, rhs_meta, meta),
            lhs_storage=lhs_storage,
            rhs_storage=rhs_storage,
            result_storage=result_storage,
        )

    @staticmethod
    def plan_product_kernel(lhs_obj: Any, rhs_obj: Any, *, spatial_dim: int = 2) -> ProductKernelPlan:
        lowering = TensorRuleEngine.plan_product_lowering(lhs_obj, rhs_obj, spatial_dim=spatial_dim)
        lhs = lowering.algebra.lhs
        rhs = lowering.algebra.rhs

        if lowering.algebra.kind in {OperationKind.PRODUCT_SCALE, OperationKind.PRODUCT_PROMOTE}:
            return ProductKernelPlan(lowering=lowering, case=ProductKernelCase.GENERIC_SCALE)

        if _is_scalar_basis(lhs) and _is_value_rank1(rhs):
            return ProductKernelPlan(
                lowering=lowering,
                case=ProductKernelCase.BASIS_SCALAR_TIMES_VALUE_VECTOR,
            )
        if _is_value_rank1(lhs) and _is_scalar_basis(rhs):
            return ProductKernelPlan(
                lowering=lowering,
                case=ProductKernelCase.VALUE_VECTOR_TIMES_BASIS_SCALAR,
            )
        if _is_scalar_basis(lhs) and _is_value_rank2(rhs):
            return ProductKernelPlan(
                lowering=lowering,
                case=ProductKernelCase.BASIS_SCALAR_TIMES_VALUE_MATRIX,
            )
        if _is_value_rank2(lhs) and _is_scalar_basis(rhs):
            return ProductKernelPlan(
                lowering=lowering,
                case=ProductKernelCase.VALUE_MATRIX_TIMES_BASIS_SCALAR,
            )
        if _is_mixed_scalar(lhs) and _is_value_rank2(rhs):
            return ProductKernelPlan(
                lowering=lowering,
                case=ProductKernelCase.MIXED_SCALAR_TIMES_VALUE_MATRIX,
            )
        if _is_value_rank2(lhs) and _is_mixed_scalar(rhs):
            return ProductKernelPlan(
                lowering=lowering,
                case=ProductKernelCase.VALUE_MATRIX_TIMES_MIXED_SCALAR,
            )
        if _is_value_rank1(lhs) and _is_value_rank1(rhs):
            return ProductKernelPlan(
                lowering=lowering,
                case=ProductKernelCase.VALUE_VECTOR_OUTER_VALUE_VECTOR,
            )
        return ProductKernelPlan(lowering=lowering, case=ProductKernelCase.GENERIC_TENSOR_PRODUCT)

    @staticmethod
    def plan_product_value_spec(lhs_obj: Any, rhs_obj: Any, *, spatial_dim: int = 2) -> KernelValueSpec:
        lowering = TensorRuleEngine.plan_product_lowering(lhs_obj, rhs_obj, spatial_dim=spatial_dim)
        shape = tuple(int(v) for v in lowering.result_storage.stored_shape)
        return KernelValueSpec(
            kind=_kernel_kind_from_lowering(lowering.algebra.result, lowering.result, shape),
            role=lowering.result.role,
            shape=shape,
            layout=lowering.result.layout,
            is_vector=lowering.result.is_vector,
            is_gradient=lowering.result.is_gradient,
            is_hessian=lowering.result.is_hessian,
            meta=lowering.meta,
        )

    @staticmethod
    def plan_determinant_meta(obj: Any, *, spatial_dim: int = 2) -> ExpressionMeta:
        meta = TensorRuleEngine.infer_expression_meta(obj, spatial_dim=spatial_dim)
        tensor = meta.tensor
        if tensor.tensor_rank != 2:
            raise TypeError(
                f"Determinant expects rank-2 tensor semantics, got rank {tensor.tensor_rank} for shape {tensor.raw_shape!r}."
            )
        left_axis, right_axis = tensor.free_axes
        if not left_axis.compatible_with(right_axis):
            raise TypeError(
                f"Determinant requires compatible axes, got {left_axis.label}:{left_axis.size} and {right_axis.label}:{right_axis.size}."
            )
        result_tensor = TensorSignature(
            free_axes=(),
            basis_axes=tensor.basis_axes,
            storage_kind="scalar",
            raw_shape=(),
            role=tensor.role,
            source="determinant",
        )
        return ExpressionMeta(tensor=result_tensor, provenance=meta.provenance)

    @staticmethod
    def plan_transpose_meta(obj: Any, *, spatial_dim: int = 2) -> ExpressionMeta:
        meta = TensorRuleEngine.infer_expression_meta(obj, spatial_dim=spatial_dim)
        tensor = meta.tensor
        if tensor.tensor_rank == 0:
            return meta
        if tensor.tensor_rank != 2:
            raise TypeError(
                f"Transpose expects rank-2 tensor semantics, got rank {tensor.tensor_rank} for shape {tensor.raw_shape!r}."
            )
        free_axes = tuple(reversed(tensor.free_axes))
        result_tensor = TensorSignature(
            free_axes=free_axes,
            basis_axes=tensor.basis_axes,
            storage_kind=tensor.storage_kind,
            raw_shape=tuple(int(axis.size) for axis in free_axes),
            role=tensor.role,
            source="transpose",
        )
        return ExpressionMeta(tensor=result_tensor, provenance=meta.provenance)

    @staticmethod
    def plan_trace_meta(obj: Any, *, spatial_dim: int = 2) -> ExpressionMeta:
        meta = TensorRuleEngine.infer_expression_meta(obj, spatial_dim=spatial_dim)
        tensor = meta.tensor
        if tensor.tensor_rank != 2:
            raise TypeError(
                f"Trace expects rank-2 tensor semantics, got rank {tensor.tensor_rank} for shape {tensor.raw_shape!r}."
            )
        left_axis, right_axis = tensor.free_axes
        if not left_axis.compatible_with(right_axis):
            raise TypeError(
                f"Trace requires compatible axes, got {left_axis.label}:{left_axis.size} and {right_axis.label}:{right_axis.size}."
            )
        result_tensor = TensorSignature(
            free_axes=(),
            basis_axes=tensor.basis_axes,
            storage_kind="scalar",
            raw_shape=(),
            role=tensor.role,
            source="trace",
        )
        return ExpressionMeta(tensor=result_tensor, provenance=meta.provenance)
