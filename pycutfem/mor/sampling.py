"""Problem-generic GNAT sampling utilities for mixed-field ROMs.

These helpers build the algebraic sampling data used by native reduced
nonlinear solves.  They deliberately work with only global rows, field slices,
and element-local DOF maps so the same machinery applies to fluids, FSI,
poromechanics, and multi-constituent systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import NativeGnatTargetSpec
from .decomposition import build_qdeim_interpolation_rule
from .mixed_reduction import build_block_row_weights


def _finite_matrix(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2-D array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _int_vector(value: Any, label: str, *, unique: bool = True) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64).reshape(-1)
    if np.any(arr < 0):
        raise ValueError(f"{label} must contain only nonnegative ids.")
    if unique:
        arr = np.unique(arr)
    return np.ascontiguousarray(arr, dtype=np.int64)


def _field_names(dof_handler: Any, fields: Sequence[str] | None = None) -> tuple[str, ...]:
    if fields is not None:
        names = tuple(str(field) for field in fields)
    else:
        names = tuple(str(field) for field in getattr(dof_handler, "field_names", ()))
        if not names:
            names = tuple(str(field) for field in getattr(dof_handler, "element_maps", {}).keys())
    if not names:
        raise ValueError("could not infer any field names from the dof handler.")
    return names


def _element_count(dof_handler: Any) -> int:
    mesh = getattr(getattr(dof_handler, "mixed_element", None), "mesh", None)
    if mesh is None:
        fe_map = getattr(dof_handler, "fe_map", {})
        if fe_map:
            mesh = next(iter(fe_map.values()))
    if mesh is not None:
        if hasattr(mesh, "n_elements"):
            return int(mesh.n_elements)
        if hasattr(mesh, "num_elements") and callable(mesh.num_elements):
            return int(mesh.num_elements())
        if hasattr(mesh, "elements_connectivity"):
            return int(np.asarray(mesh.elements_connectivity).shape[0])
    maps = getattr(dof_handler, "element_maps", {})
    if maps:
        return max(int(len(blocks)) for blocks in maps.values())
    raise ValueError("could not infer element count from the dof handler.")


def _global_row_lookup(free_rows: np.ndarray, *, n_total: int) -> np.ndarray:
    size = max(int(n_total), int(np.max(free_rows)) + 1 if free_rows.size else 0)
    lookup = np.full(size, -1, dtype=np.int64)
    lookup[free_rows] = np.arange(free_rows.size, dtype=np.int64)
    return lookup


def _coerce_block(block: Any) -> tuple[np.ndarray, str]:
    if isinstance(block, SamplingBlock):
        return np.asarray(block.rows, dtype=np.int64).reshape(-1), block.name
    if isinstance(block, Mapping):
        rows = np.asarray(block["rows"], dtype=np.int64).reshape(-1)
        return rows, str(block.get("name", ""))
    if isinstance(block, (tuple, list)) and len(block) in {1, 2}:
        rows = np.asarray(block[0], dtype=np.int64).reshape(-1)
        name = "" if len(block) == 1 else str(block[1])
        return rows, name
    return np.asarray(block, dtype=np.int64).reshape(-1), ""


def _localize_row_blocks(row_blocks: Sequence[Any], free_rows: np.ndarray, n_total: int) -> tuple[tuple[np.ndarray, str], ...]:
    lookup = _global_row_lookup(free_rows, n_total=n_total)
    localized: list[tuple[np.ndarray, str]] = []
    for block in row_blocks:
        rows, name = _coerce_block(block)
        rows = rows[(rows >= 0) & (rows < lookup.size)]
        local = lookup[rows]
        local = np.unique(local[local >= 0])
        localized.append((np.ascontiguousarray(local, dtype=np.int64), str(name)))
    return tuple(localized)


def _min_rows_for_block(min_rows: int | Mapping[str, int] | None, name: str) -> int:
    if min_rows is None:
        return 0
    if isinstance(min_rows, Mapping):
        return max(0, int(min_rows.get(name, min_rows.get("*", 0))))
    return max(0, int(min_rows))


def _element_local_rows(
    dof_handler: Any,
    element_ids: np.ndarray,
    fields: tuple[str, ...],
    row_lookup: np.ndarray,
) -> dict[int, np.ndarray]:
    maps = getattr(dof_handler, "element_maps", {})
    support: dict[int, np.ndarray] = {}
    for eid_raw in np.asarray(element_ids, dtype=np.int64).reshape(-1):
        eid = int(eid_raw)
        rows: list[np.ndarray] = []
        for field in fields:
            field_maps = maps.get(field)
            if field_maps is None or eid < 0 or eid >= len(field_maps):
                continue
            gdofs = np.asarray(field_maps[eid], dtype=np.int64).reshape(-1)
            gdofs = gdofs[(gdofs >= 0) & (gdofs < row_lookup.size)]
            if gdofs.size:
                local = row_lookup[gdofs]
                local = local[local >= 0]
                if local.size:
                    rows.append(local)
        if rows:
            support[eid] = np.unique(np.concatenate(rows))
        else:
            support[eid] = np.zeros(0, dtype=np.int64)
    return support


def _selected_supports_element(selected: np.ndarray, element_rows: np.ndarray) -> bool:
    if selected.size == 0 or element_rows.size == 0:
        return False
    return bool(np.intersect1d(selected, element_rows, assume_unique=False).size)


@dataclass(frozen=True)
class SamplingBlock:
    """Named residual row block used for scale balancing and coverage quotas."""

    rows: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", _int_vector(self.rows, "sampling block rows", unique=True))
        object.__setattr__(self, "name", str(self.name))


@dataclass(frozen=True)
class BlockBalancedGnatSampling:
    """Native GNAT target data produced by block-balanced row sampling."""

    row_dofs: np.ndarray
    element_ids: np.ndarray
    row_weights: np.ndarray
    selected_basis: np.ndarray
    residual_terms: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        row_dofs = np.asarray(self.row_dofs, dtype=np.int64).reshape(-1)
        if row_dofs.size == 0 or np.any(row_dofs < 0) or np.unique(row_dofs).size != row_dofs.size:
            raise ValueError("sampled row dofs must be nonempty, nonnegative, and unique.")
        element_ids = _int_vector(self.element_ids, "sampled element ids", unique=True)
        row_weights = np.asarray(self.row_weights, dtype=float).reshape(-1)
        if row_weights.size != row_dofs.size:
            raise ValueError("row_weights must have one entry per sampled row.")
        if not np.all(np.isfinite(row_weights)) or np.any(row_weights <= 0.0):
            raise ValueError("row_weights must contain finite positive values.")
        selected_basis = _finite_matrix(self.selected_basis, "selected_basis")
        residual_terms = _finite_matrix(self.residual_terms, "residual_terms")
        if selected_basis.shape != (row_dofs.size, row_dofs.size):
            raise ValueError("selected_basis must be square with one row per sampled row.")
        if residual_terms.shape != selected_basis.shape:
            raise ValueError("residual_terms must match selected_basis for identity sampled GNAT targets.")
        object.__setattr__(self, "row_dofs", row_dofs)
        object.__setattr__(self, "element_ids", element_ids)
        object.__setattr__(self, "row_weights", np.ascontiguousarray(row_weights, dtype=np.float64))
        object.__setattr__(self, "selected_basis", selected_basis)
        object.__setattr__(self, "residual_terms", residual_terms)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_native_target(self, *, objective: str = "sampled_gnat") -> NativeGnatTargetSpec:
        """Return a serializable native target specification."""

        return NativeGnatTargetSpec(
            row_dofs=self.row_dofs,
            element_ids=self.element_ids,
            row_weights=self.row_weights,
            selected_basis=self.selected_basis,
            residual_terms=self.residual_terms,
            objective=objective,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class ResidualNormEquivalenceCertificate:
    """Diagnostics comparing sampled and full residual norms.

    The certificate is purely algebraic: it needs residual snapshot columns,
    sampled global rows, optional row weights, and optional named row blocks.
    It is intended as an offline acceptance gate before a sampled GNAT target is
    trusted for native online solves.
    """

    passed: bool
    lower_constant: float
    upper_constant: float
    sample_count: int
    skipped_zero_columns: int
    block_constants: Mapping[str, Mapping[str, float | int | bool]]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "lower_constant": float(self.lower_constant),
            "upper_constant": float(self.upper_constant),
            "sample_count": int(self.sample_count),
            "skipped_zero_columns": int(self.skipped_zero_columns),
            "block_constants": dict(self.block_constants),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AugmentedNormEquivalenceResult:
    """Sample rows after adaptive norm-equivalence augmentation."""

    row_dofs: np.ndarray
    row_weights: np.ndarray
    certificate: ResidualNormEquivalenceCertificate
    added_rows: np.ndarray
    iterations: int
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        rows = _int_vector(self.row_dofs, "augmented row_dofs", unique=True)
        weights = np.asarray(self.row_weights, dtype=float).reshape(-1)
        if weights.size != rows.size:
            raise ValueError("augmented row_weights must have one entry per row.")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("augmented row_weights must contain finite positive values.")
        object.__setattr__(self, "row_dofs", rows)
        object.__setattr__(self, "row_weights", np.ascontiguousarray(weights, dtype=np.float64))
        object.__setattr__(self, "added_rows", _int_vector(self.added_rows, "added rows", unique=True) if np.asarray(self.added_rows).size else np.zeros(0, dtype=np.int64))
        object.__setattr__(self, "iterations", int(self.iterations))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_dofs": self.row_dofs,
            "row_weights": self.row_weights,
            "certificate": self.certificate.to_dict(),
            "added_rows": self.added_rows,
            "iterations": int(self.iterations),
            "metadata": dict(self.metadata),
        }


def _weighted_sample_norms(
    residuals: np.ndarray,
    row_dofs: np.ndarray,
    row_weights: np.ndarray,
) -> np.ndarray:
    if row_dofs.size == 0:
        return np.zeros(residuals.shape[1], dtype=np.float64)
    sampled = residuals[row_dofs, :]
    return np.linalg.norm(np.sqrt(row_weights)[:, None] * sampled, axis=0)


def _ratio_bounds(
    full_norm: np.ndarray,
    sampled_norm: np.ndarray,
    *,
    zero_tolerance: float,
) -> tuple[float, float, int, bool]:
    active = np.asarray(full_norm, dtype=float) > float(zero_tolerance)
    skipped = int(np.count_nonzero(~active))
    if not bool(np.any(active)):
        return 1.0, 1.0, skipped, True
    ratios = np.asarray(sampled_norm, dtype=float)[active] / np.asarray(full_norm, dtype=float)[active]
    if not np.all(np.isfinite(ratios)):
        return 0.0, float("inf"), skipped, False
    return float(np.min(ratios)), float(np.max(ratios)), skipped, True


def certify_sampled_residual_norm_equivalence(
    residual_matrix: Any,
    row_dofs: Any,
    *,
    row_weights: Any | None = None,
    row_blocks: Sequence[Any] | None = None,
    lower_bound: float = 1.0e-3,
    upper_bound: float = 1.0e3,
    zero_tolerance: float = 1.0e-14,
    metadata: Mapping[str, Any] | None = None,
) -> ResidualNormEquivalenceCertificate:
    """Certify that sampled residual norms see the full residual neighborhood.

    The returned constants estimate

    ``lower_constant * ||R|| <= ||S R||_W <= upper_constant * ||R||``

    over the supplied residual snapshot columns.  Optional ``row_blocks`` use
    global row ids and are evaluated independently, so a pressure, interface, or
    scalar block with zero sampled coverage fails even if the global ratio looks
    acceptable.
    """

    residuals = _finite_matrix(residual_matrix, "residual_matrix")
    rows = _int_vector(row_dofs, "row_dofs", unique=True)
    if rows.size == 0:
        raise ValueError("row_dofs must contain at least one sampled row.")
    if np.any(rows >= residuals.shape[0]):
        raise ValueError("row_dofs contains rows outside residual_matrix.")
    if row_weights is None:
        weights = np.ones(rows.size, dtype=np.float64)
    else:
        weights = np.asarray(row_weights, dtype=float).reshape(-1)
        if weights.size != rows.size:
            raise ValueError("row_weights must have one value per sampled row.")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("row_weights must contain finite positive values.")
        weights = np.ascontiguousarray(weights, dtype=np.float64)
    if not np.isfinite(lower_bound) or float(lower_bound) < 0.0:
        raise ValueError("lower_bound must be finite and nonnegative.")
    if not np.isfinite(upper_bound) or float(upper_bound) <= 0.0:
        raise ValueError("upper_bound must be finite and positive.")
    if not np.isfinite(zero_tolerance) or float(zero_tolerance) < 0.0:
        raise ValueError("zero_tolerance must be finite and nonnegative.")

    full_norms = np.linalg.norm(residuals, axis=0)
    sampled_norms = _weighted_sample_norms(residuals, rows, weights)
    lower, upper, skipped, global_finite = _ratio_bounds(
        full_norms,
        sampled_norms,
        zero_tolerance=float(zero_tolerance),
    )

    block_results: dict[str, dict[str, float | int | bool]] = {}
    block_passed = True
    if row_blocks is not None:
        row_position = {int(row): int(i) for i, row in enumerate(rows.tolist())}
        for block_idx, block in enumerate(row_blocks):
            block_rows, name = _coerce_block(block)
            block_rows = np.unique(block_rows[(block_rows >= 0) & (block_rows < residuals.shape[0])])
            block_name = str(name) if str(name) else f"block_{block_idx}"
            if block_rows.size == 0:
                block_results[block_name] = {
                    "passed": True,
                    "full_rows": 0,
                    "sampled_rows": 0,
                    "lower_constant": 1.0,
                    "upper_constant": 1.0,
                    "skipped_zero_columns": int(residuals.shape[1]),
                }
                continue
            sampled_positions = np.asarray(
                [row_position[int(row)] for row in block_rows.tolist() if int(row) in row_position],
                dtype=np.int64,
            )
            full_block_norms = np.linalg.norm(residuals[block_rows, :], axis=0)
            sampled_block_rows = rows[sampled_positions] if sampled_positions.size else np.zeros(0, dtype=np.int64)
            sampled_block_weights = weights[sampled_positions] if sampled_positions.size else np.zeros(0, dtype=np.float64)
            sampled_block_norms = _weighted_sample_norms(residuals, sampled_block_rows, sampled_block_weights)
            b_lower, b_upper, b_skipped, b_finite = _ratio_bounds(
                full_block_norms,
                sampled_block_norms,
                zero_tolerance=float(zero_tolerance),
            )
            b_passed = bool(
                b_finite
                and b_lower >= float(lower_bound)
                and b_upper <= float(upper_bound)
            )
            block_passed = bool(block_passed and b_passed)
            block_results[block_name] = {
                "passed": b_passed,
                "full_rows": int(block_rows.size),
                "sampled_rows": int(sampled_positions.size),
                "lower_constant": float(b_lower),
                "upper_constant": float(b_upper),
                "skipped_zero_columns": int(b_skipped),
            }

    passed = bool(
        global_finite
        and lower >= float(lower_bound)
        and upper <= float(upper_bound)
        and block_passed
    )
    return ResidualNormEquivalenceCertificate(
        passed=passed,
        lower_constant=float(lower),
        upper_constant=float(upper),
        sample_count=int(residuals.shape[1]),
        skipped_zero_columns=int(skipped),
        block_constants=block_results,
        metadata={
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "zero_tolerance": float(zero_tolerance),
            "sampled_rows": int(rows.size),
            "total_rows": int(residuals.shape[0]),
            **dict(metadata or {}),
        },
    )


def _worst_failed_block_rows(
    certificate: ResidualNormEquivalenceCertificate,
    row_blocks: Sequence[Any] | None,
    residuals: np.ndarray,
) -> np.ndarray:
    if row_blocks is None:
        return np.arange(residuals.shape[0], dtype=np.int64)
    failed: list[tuple[float, np.ndarray]] = []
    for block_idx, block in enumerate(row_blocks):
        rows, name = _coerce_block(block)
        block_name = str(name) if str(name) else f"block_{block_idx}"
        info = certificate.block_constants.get(block_name)
        if not info or bool(info.get("passed", False)):
            continue
        rows = np.unique(rows[(rows >= 0) & (rows < residuals.shape[0])])
        if rows.size == 0:
            continue
        lower = float(info.get("lower_constant", 0.0))
        sampled = int(info.get("sampled_rows", 0))
        score = (1.0 / max(lower, 1.0e-300)) + (1.0 if sampled == 0 else 0.0)
        failed.append((score, rows))
    if not failed:
        return np.arange(residuals.shape[0], dtype=np.int64)
    failed.sort(key=lambda item: item[0], reverse=True)
    return failed[0][1]


def augment_rows_for_residual_norm_equivalence(
    residual_matrix: Any,
    row_dofs: Any,
    *,
    row_weights: Any | None = None,
    full_row_weights: Any | None = None,
    row_blocks: Sequence[Any] | None = None,
    lower_bound: float = 1.0e-3,
    upper_bound: float = 1.0e3,
    zero_tolerance: float = 1.0e-14,
    mandatory_rows: Any | None = None,
    max_rows: int | None = None,
    max_iterations: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AugmentedNormEquivalenceResult:
    """Add high-deficit rows until the residual norm-equivalence gate passes.

    The augmentation is intentionally algebraic.  It can be applied to GNAT,
    DEIM/QDEIM, or any sampled residual target by passing residual snapshot
    columns from the validation neighborhood.  Rows are chosen from the failed
    block with largest residual energy deficit; mandatory rows are always kept.
    """

    residuals = _finite_matrix(residual_matrix, "residual_matrix")
    initial = _int_vector(row_dofs, "row_dofs", unique=True)
    if np.any(initial >= residuals.shape[0]):
        raise ValueError("row_dofs contains rows outside residual_matrix.")
    mandatory = (
        np.zeros(0, dtype=np.int64)
        if mandatory_rows is None
        else _int_vector(mandatory_rows, "mandatory_rows", unique=True)
    )
    if mandatory.size and np.any(mandatory >= residuals.shape[0]):
        raise ValueError("mandatory_rows contains rows outside residual_matrix.")
    rows = np.union1d(initial, mandatory).astype(np.int64, copy=False)
    if full_row_weights is not None:
        full_weights = np.asarray(full_row_weights, dtype=float).reshape(-1)
        if full_weights.size != residuals.shape[0]:
            raise ValueError("full_row_weights must have one entry per residual row.")
        if not np.all(np.isfinite(full_weights)) or np.any(full_weights <= 0.0):
            raise ValueError("full_row_weights must contain finite positive values.")
    else:
        full_weights = np.ones(residuals.shape[0], dtype=np.float64)
        if row_weights is not None:
            raw = np.asarray(row_weights, dtype=float).reshape(-1)
            if raw.size != initial.size:
                raise ValueError("row_weights must have one entry per initial row when full_row_weights is absent.")
            full_weights[initial] = raw
    limit = int(residuals.shape[0] if max_rows is None else max_rows)
    if limit < rows.size:
        raise ValueError("max_rows is smaller than the required initial/mandatory row set.")
    n_iter = int(residuals.shape[0] if max_iterations is None else max_iterations)
    if n_iter < 0:
        raise ValueError("max_iterations must be nonnegative.")

    added: list[int] = []
    cert = certify_sampled_residual_norm_equivalence(
        residuals,
        rows,
        row_weights=full_weights[rows],
        row_blocks=row_blocks,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        zero_tolerance=zero_tolerance,
        metadata=metadata,
    )
    iterations = 0
    while not cert.passed and rows.size < limit and iterations < n_iter:
        candidate_pool = _worst_failed_block_rows(cert, row_blocks, residuals)
        available = np.setdiff1d(candidate_pool, rows, assume_unique=False)
        if available.size == 0:
            available = np.setdiff1d(np.arange(residuals.shape[0], dtype=np.int64), rows, assume_unique=False)
        if available.size == 0:
            break
        row_energy = np.linalg.norm(residuals[available, :], axis=1)
        best = int(available[int(np.argmax(row_energy))])
        rows = np.union1d(rows, np.asarray([best], dtype=np.int64)).astype(np.int64, copy=False)
        added.append(best)
        iterations += 1
        cert = certify_sampled_residual_norm_equivalence(
            residuals,
            rows,
            row_weights=full_weights[rows],
            row_blocks=row_blocks,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            zero_tolerance=zero_tolerance,
            metadata=metadata,
        )
    return AugmentedNormEquivalenceResult(
        row_dofs=rows,
        row_weights=full_weights[rows],
        certificate=cert,
        added_rows=np.asarray(added, dtype=np.int64),
        iterations=iterations,
        metadata={
            "initial_rows": int(initial.size),
            "mandatory_rows": int(mandatory.size),
            "max_rows": int(limit),
            "passed": bool(cert.passed),
            **dict(metadata or {}),
        },
    )


def field_row_blocks(
    dof_handler: Any,
    fields: Sequence[str] | None = None,
    *,
    row_subset: Any | None = None,
) -> tuple[SamplingBlock, ...]:
    """Return one named global-row block per field.

    If ``row_subset`` is provided, each block is intersected with that set while
    preserving global row ids.  This keeps the function useful for both
    high-level metadata and samplers that localize rows internally.
    """

    names = _field_names(dof_handler, fields)
    subset: set[int] | None = None
    if row_subset is not None:
        subset = set(int(row) for row in np.asarray(row_subset, dtype=np.int64).reshape(-1))
    blocks: list[SamplingBlock] = []
    for field in names:
        rows = np.asarray(dof_handler.get_field_slice(field), dtype=np.int64).reshape(-1)
        if subset is not None:
            rows = np.asarray([int(row) for row in rows if int(row) in subset], dtype=np.int64)
        blocks.append(SamplingBlock(rows=rows, name=field))
    return tuple(blocks)


def support_element_ids_from_rows(
    dof_handler: Any,
    row_dofs: Any,
    *,
    fields: Sequence[str] | None = None,
) -> np.ndarray:
    """Return element ids whose local DOF map touches at least one sampled row."""

    rows = _int_vector(row_dofs, "row_dofs", unique=True)
    if rows.size == 0:
        return np.zeros(0, dtype=np.int64)
    n_total = max(int(getattr(dof_handler, "total_dofs", int(rows.max()) + 1)), int(rows.max()) + 1)
    flags = np.zeros(n_total, dtype=bool)
    flags[rows] = True
    names = _field_names(dof_handler, fields)
    maps = getattr(dof_handler, "element_maps", {})
    element_ids: list[int] = []
    for eid in range(_element_count(dof_handler)):
        touched = False
        for field in names:
            field_maps = maps.get(field)
            if field_maps is None or eid >= len(field_maps):
                continue
            gdofs = np.asarray(field_maps[eid], dtype=np.int64).reshape(-1)
            gdofs = gdofs[(gdofs >= 0) & (gdofs < flags.size)]
            if gdofs.size and bool(np.any(flags[gdofs])):
                touched = True
                break
        if touched:
            element_ids.append(eid)
    return np.ascontiguousarray(np.asarray(element_ids, dtype=np.int64))


def rows_supported_on_elements(
    dof_handler: Any,
    element_ids: Any,
    *,
    candidate_rows: Any | None = None,
    fields: Sequence[str] | None = None,
) -> np.ndarray:
    """Return candidate rows that appear in the local maps of selected elements."""

    eids = _int_vector(element_ids, "element_ids", unique=True)
    if eids.size == 0:
        return np.zeros(0, dtype=np.int64)
    if candidate_rows is None:
        n_total = int(getattr(dof_handler, "total_dofs"))
        rows = np.arange(n_total, dtype=np.int64)
    else:
        rows = _int_vector(candidate_rows, "candidate_rows", unique=True)
    lookup = _global_row_lookup(rows, n_total=max(int(getattr(dof_handler, "total_dofs", 0)), int(rows.max()) + 1 if rows.size else 0))
    local = _element_local_rows(dof_handler, eids, _field_names(dof_handler, fields), lookup)
    selected = [rows[block] for block in local.values() if block.size]
    if not selected:
        return np.zeros(0, dtype=np.int64)
    return np.ascontiguousarray(np.unique(np.concatenate(selected)), dtype=np.int64)


def select_coordinate_band_elements(
    mesh: Any,
    *,
    axis: int,
    center: float,
    half_width: float,
    use_corner_nodes: bool = True,
) -> np.ndarray:
    """Select elements intersecting a coordinate band.

    This is a generic way to mark diffuse-interface bands, material layers, or
    other axis-aligned regions that must be covered by a sampled native target.
    """

    coords = np.asarray(getattr(mesh, "nodes_x_y_pos"), dtype=float)
    if coords.ndim != 2:
        raise ValueError("mesh nodes_x_y_pos must be a 2-D coordinate array.")
    ax = int(axis)
    if ax < 0 or ax >= coords.shape[1]:
        raise ValueError("axis is outside the mesh coordinate dimension.")
    if not np.isfinite(center) or not np.isfinite(half_width) or half_width < 0.0:
        raise ValueError("center and half_width must be finite and half_width must be nonnegative.")
    conn_name = "corner_connectivity" if use_corner_nodes and getattr(mesh, "corner_connectivity", None) is not None else "elements_connectivity"
    conn = np.asarray(getattr(mesh, conn_name), dtype=np.int64)
    if conn.ndim != 2:
        raise ValueError(f"mesh {conn_name} must be a 2-D connectivity array.")
    values = coords[conn, ax]
    lo = float(center) - float(half_width)
    hi = float(center) + float(half_width)
    mask = (np.min(values, axis=1) <= hi) & (np.max(values, axis=1) >= lo)
    return np.ascontiguousarray(np.nonzero(mask)[0].astype(np.int64))


def build_block_balanced_gnat_sampling(
    dof_handler: Any,
    trial_basis: Any,
    *,
    snapshot_matrix: Any | None = None,
    reference_matrix: Any | None = None,
    free_rows: Any | None = None,
    row_blocks: Sequence[Any] | None = None,
    sample_rows: int,
    candidate_element_ids: Any | None = None,
    mandatory_element_ids: Any | None = None,
    min_rows_per_block: int | Mapping[str, int] | None = None,
    fields: Sequence[str] | None = None,
    row_weight_max: float = 1.0e8,
    rcond: float | None = None,
    require_mandatory_coverage: bool = True,
) -> BlockBalancedGnatSampling:
    """Build a block-balanced, mandatory-element-complete sampled GNAT target.

    The sampler starts from QDEIM rows on the row-weighted trial basis, augments
    the rows so every requested element has at least one sampled residual row,
    enforces per-block minimum coverage, and finally fills the requested row
    budget with high-energy rows.  Mandatory element coverage is treated as a
    correctness requirement and can exceed ``sample_rows`` when the requested
    budget is too small.
    """

    V = _finite_matrix(trial_basis, "trial_basis")
    n_total = int(getattr(dof_handler, "total_dofs", V.shape[0]))
    if V.shape[0] != n_total:
        raise ValueError("trial_basis row count must match dof_handler.total_dofs.")
    if V.shape[1] == 0:
        raise ValueError("trial_basis must contain at least one mode.")
    if int(sample_rows) <= 0:
        raise ValueError("sample_rows must be positive.")

    if free_rows is None:
        free = np.arange(n_total, dtype=np.int64)
    else:
        free = _int_vector(free_rows, "free_rows", unique=True)
        if np.any(free >= n_total):
            raise ValueError("free_rows contains rows outside trial_basis.")
    names = _field_names(dof_handler, fields)
    n_elements = _element_count(dof_handler)
    candidate_element_array = (
        np.zeros(0, dtype=np.int64)
        if candidate_element_ids is None
        else _int_vector(candidate_element_ids, "candidate_element_ids", unique=True)
    )
    if candidate_element_array.size and np.any(candidate_element_array >= n_elements):
        raise ValueError("candidate_element_ids contains ids outside the mesh element range.")
    if candidate_element_array.size:
        candidate_rows = rows_supported_on_elements(
            dof_handler,
            candidate_element_array,
            candidate_rows=free,
            fields=names,
        )
        if candidate_rows.size == 0:
            raise ValueError("candidate_element_ids do not support any free rows.")
        free = candidate_rows
    if free.size < V.shape[1]:
        raise ValueError("free_rows must contain at least as many rows as trial modes.")

    blocks = tuple(row_blocks) if row_blocks is not None else field_row_blocks(dof_handler, names)
    local_blocks = _localize_row_blocks(blocks, free, n_total)

    if reference_matrix is not None:
        reference = _finite_matrix(reference_matrix, "reference_matrix")
        if reference.shape[0] == n_total:
            reference = reference[free, :]
        elif reference.shape[0] != free.size:
            raise ValueError("reference_matrix row count must match total rows or free rows.")
    elif snapshot_matrix is not None:
        snapshots = _finite_matrix(snapshot_matrix, "snapshot_matrix")
        if snapshots.shape[0] != n_total:
            raise ValueError("snapshot_matrix row count must match dof_handler.total_dofs.")
        centered = snapshots - np.mean(snapshots, axis=1, keepdims=True)
        reference = np.ascontiguousarray(centered[free, :], dtype=np.float64)
    else:
        reference = np.ascontiguousarray(V[free, :], dtype=np.float64)

    if local_blocks:
        row_weights = build_block_row_weights(reference, local_blocks, max_weight=float(row_weight_max))
    else:
        row_weights = np.ones(free.size, dtype=np.float64)
    weighted_basis = np.ascontiguousarray(np.sqrt(row_weights)[:, None] * V[free, :], dtype=np.float64)
    rule = build_qdeim_interpolation_rule(weighted_basis, rcond=rcond)

    weighted_reference = np.sqrt(row_weights)[:, None] * reference
    basis_norm = np.linalg.norm(weighted_basis, axis=1)
    ref_norm = np.linalg.norm(weighted_reference, axis=1)
    basis_scale = basis_norm / max(float(np.max(basis_norm)), 1.0e-300)
    ref_scale = ref_norm / max(float(np.max(ref_norm)), 1.0e-300)
    tie = np.linspace(1.0e-12, 0.0, free.size, endpoint=False)
    scores = np.ascontiguousarray(basis_scale + ref_scale + tie, dtype=np.float64)

    selected: list[int] = []
    used = np.zeros(free.size, dtype=bool)

    def append_row(local_row: int) -> bool:
        row = int(local_row)
        if row < 0 or row >= free.size or used[row]:
            return False
        used[row] = True
        selected.append(row)
        return True

    for row in np.asarray(rule.rows, dtype=np.int64).reshape(-1):
        append_row(int(row))

    mandatory = (
        np.zeros(0, dtype=np.int64)
        if mandatory_element_ids is None
        else _int_vector(mandatory_element_ids, "mandatory_element_ids", unique=True)
    )
    if mandatory.size and np.any(mandatory >= n_elements):
        raise ValueError("mandatory_element_ids contains ids outside the mesh element range.")
    row_lookup = _global_row_lookup(free, n_total=n_total)
    mandatory_rows = _element_local_rows(dof_handler, mandatory, names, row_lookup)
    mandatory_added = 0

    while mandatory.size:
        selected_array = np.asarray(selected, dtype=np.int64)
        missing = [
            int(eid)
            for eid, rows in mandatory_rows.items()
            if not _selected_supports_element(selected_array, rows)
        ]
        if not missing:
            break
        additions = 0
        for eid in missing:
            row_candidates = mandatory_rows[eid]
            row_candidates = row_candidates[~used[row_candidates]]
            if row_candidates.size == 0:
                continue
            best = int(row_candidates[np.argmax(scores[row_candidates])])
            if append_row(best):
                mandatory_added += 1
                additions += 1
        if additions == 0:
            break

    block_added = 0
    block_counts: dict[str, int] = {}
    block_candidates: dict[str, int] = {}
    for local_rows, name in local_blocks:
        block_name = str(name)
        block_candidates[block_name] = int(local_rows.size)
        required = min(_min_rows_for_block(min_rows_per_block, block_name), int(local_rows.size))
        if required <= 0:
            block_counts[block_name] = int(np.count_nonzero(used[local_rows])) if local_rows.size else 0
            continue
        while int(np.count_nonzero(used[local_rows])) < required:
            row_candidates = local_rows[~used[local_rows]]
            if row_candidates.size == 0:
                break
            best = int(row_candidates[np.argmax(scores[row_candidates])])
            if append_row(best):
                block_added += 1
            else:
                break
        block_counts[block_name] = int(np.count_nonzero(used[local_rows])) if local_rows.size else 0

    target_rows = min(int(free.size), max(int(sample_rows), int(rule.rows.size)))
    while len(selected) < target_rows:
        row_candidates = np.nonzero(~used)[0]
        if row_candidates.size == 0:
            break
        best = int(row_candidates[np.argmax(scores[row_candidates])])
        append_row(best)

    local_rows = np.asarray(selected, dtype=np.int64)
    order = np.argsort(free[local_rows], kind="stable")
    local_rows = local_rows[order]
    row_dofs = np.ascontiguousarray(free[local_rows], dtype=np.int64)
    sampled_weights = np.ascontiguousarray(row_weights[local_rows], dtype=np.float64)
    support_elements = support_element_ids_from_rows(dof_handler, row_dofs, fields=names)
    element_ids = np.union1d(support_elements, mandatory).astype(np.int64, copy=False)

    selected_set = np.asarray(local_rows, dtype=np.int64)
    missing_mandatory = np.asarray(
        [
            int(eid)
            for eid, rows in mandatory_rows.items()
            if not _selected_supports_element(selected_set, rows)
        ],
        dtype=np.int64,
    )
    if missing_mandatory.size and require_mandatory_coverage:
        raise ValueError(
            "mandatory element coverage failed for "
            f"{missing_mandatory.size} elements; increase sample_rows or check free_rows."
        )

    selected_basis = np.ascontiguousarray(np.eye(row_dofs.size, dtype=np.float64))
    residual_terms = np.ascontiguousarray(np.eye(row_dofs.size, dtype=np.float64))
    metadata = {
        "sampler": "block_balanced_interface_complete_gnat",
        "target": "sampled_gnat",
        "free_rows": int(free.size),
        "selected_rows": int(row_dofs.size),
        "requested_rows": int(sample_rows),
        "qdeim_core_rows": int(rule.rows.size),
        "mandatory_element_count": int(mandatory.size),
        "candidate_element_count": int(candidate_element_array.size),
        "candidate_limited": bool(candidate_element_array.size > 0),
        "sampled_mandatory_element_count": int(mandatory.size - missing_mandatory.size),
        "missing_mandatory_element_count": int(missing_mandatory.size),
        "missing_mandatory_element_ids": tuple(int(v) for v in missing_mandatory.tolist()),
        "mandatory_rows_added": int(mandatory_added),
        "block_rows_added": int(block_added),
        "budget_exceeded_by_required_rows": bool(row_dofs.size > int(sample_rows)),
        "interface_complete": bool(missing_mandatory.size == 0),
        "sampled_element_count": int(element_ids.size),
        "support_element_count": int(support_elements.size),
        "total_element_count": int(n_elements),
        "row_weight_min": float(np.min(sampled_weights)),
        "row_weight_max": float(np.max(sampled_weights)),
        "block_selected_rows": block_counts,
        "block_candidate_rows": block_candidates,
        "selected_basis_condition": float(np.linalg.cond(selected_basis)),
    }
    return BlockBalancedGnatSampling(
        row_dofs=row_dofs,
        element_ids=element_ids,
        row_weights=sampled_weights,
        selected_basis=selected_basis,
        residual_terms=residual_terms,
        metadata=metadata,
    )


__all__ = [
    "BlockBalancedGnatSampling",
    "AugmentedNormEquivalenceResult",
    "ResidualNormEquivalenceCertificate",
    "SamplingBlock",
    "augment_rows_for_residual_norm_equivalence",
    "build_block_balanced_gnat_sampling",
    "certify_sampled_residual_norm_equivalence",
    "field_row_blocks",
    "rows_supported_on_elements",
    "select_coordinate_band_elements",
    "support_element_ids_from_rows",
]
