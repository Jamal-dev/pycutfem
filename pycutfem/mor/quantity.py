"""Gappy POD/DEIM quantity operators for certified ROMs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .decomposition import build_deim_interpolation_rule, build_qdeim_interpolation_rule, fit_collateral_basis


def _finite_matrix(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2-D array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _finite_vector(value: Any, label: str, size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values.")
    if size is not None and int(arr.size) != int(size):
        raise ValueError(f"{label} must have size {size}.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _index_vector(value: Any, label: str, upper: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64).reshape(-1)
    if arr.size == 0 or np.any(arr < 0) or np.any(arr >= int(upper)):
        raise ValueError(f"{label} must contain valid row ids.")
    if np.unique(arr).size != arr.size:
        raise ValueError(f"{label} must not contain duplicate row ids.")
    return np.ascontiguousarray(arr, dtype=np.int64)


def _oversample_rows(basis: np.ndarray, rows: np.ndarray, n_rows: int) -> np.ndarray:
    target = min(int(n_rows), int(basis.shape[0]))
    selected = list(int(v) for v in rows)
    if len(selected) >= target:
        return np.ascontiguousarray(np.asarray(selected[:target], dtype=np.int64))
    used = set(selected)
    scores = np.linalg.norm(np.asarray(basis, dtype=float), axis=1)
    for row in np.argsort(scores)[::-1]:
        rid = int(row)
        if rid in used:
            continue
        selected.append(rid)
        used.add(rid)
        if len(selected) >= target:
            break
    return np.ascontiguousarray(np.asarray(selected, dtype=np.int64))


@dataclass(frozen=True)
class GappyPODQuantityOperator:
    """Reconstruct a quantity from sampled rows using a collateral basis."""

    basis: np.ndarray
    sample_rows: np.ndarray
    offset: np.ndarray | None = None
    output_rows: np.ndarray | None = None
    sample_to_coefficients: np.ndarray | None = None
    rcond: float | None = None
    name: str = "quantity"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        basis = _finite_matrix(self.basis, "quantity basis")
        sample_rows = _index_vector(self.sample_rows, "quantity sample_rows", basis.shape[0])
        offset = np.zeros(basis.shape[0], dtype=np.float64) if self.offset is None else _finite_vector(
            self.offset,
            "quantity offset",
            basis.shape[0],
        )
        output_rows = None if self.output_rows is None else _index_vector(
            self.output_rows,
            "quantity output_rows",
            basis.shape[0],
        )
        rcond = None if self.rcond is None else float(self.rcond)
        if rcond is not None and (not np.isfinite(rcond) or rcond < 0.0):
            raise ValueError("quantity rcond must be finite and nonnegative.")
        selected = basis[sample_rows, :]
        if selected.shape[0] < selected.shape[1]:
            raise ValueError("quantity sample_rows must select at least as many rows as basis modes.")
        if self.sample_to_coefficients is None:
            sample_to_coefficients = np.linalg.pinv(selected, rcond=1.0e-12 if rcond is None else rcond)
        else:
            sample_to_coefficients = _finite_matrix(
                self.sample_to_coefficients,
                "quantity sample_to_coefficients",
            )
            expected_shape = (int(basis.shape[1]), int(sample_rows.size))
            if sample_to_coefficients.shape != expected_shape:
                raise ValueError(
                    "quantity sample_to_coefficients must have shape "
                    f"{expected_shape}, got {sample_to_coefficients.shape}."
                )
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "sample_rows", sample_rows)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "output_rows", output_rows)
        object.__setattr__(self, "sample_to_coefficients", np.ascontiguousarray(sample_to_coefficients))
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def n_features(self) -> int:
        return int(self.basis.shape[0])

    @property
    def n_modes(self) -> int:
        return int(self.basis.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.sample_rows.size)

    def sample(self, values: Any) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            return np.ascontiguousarray(arr.reshape(-1)[self.sample_rows])
        if arr.ndim == 2 and arr.shape[0] == self.n_features:
            return np.ascontiguousarray(arr[self.sample_rows, :])
        raise ValueError("quantity values must have shape (n_features,) or (n_features, n_snapshots).")

    def coefficients_from_samples(self, sample_values: Any) -> np.ndarray:
        samples = np.asarray(sample_values, dtype=float)
        one_column = samples.ndim == 1
        if one_column:
            samples = samples.reshape(-1, 1)
        if samples.ndim != 2 or samples.shape[0] != self.n_samples:
            raise ValueError("sample_values must have shape (n_samples,) or (n_samples, n_columns).")
        centered = samples - self.offset[self.sample_rows, None]
        coeffs = self.sample_to_coefficients @ centered
        return np.ascontiguousarray(coeffs[:, 0] if one_column else coeffs)

    def reconstruct_from_samples(self, sample_values: Any, *, output_rows: Any | None = None) -> np.ndarray:
        coeffs = self.coefficients_from_samples(sample_values)
        one_column = coeffs.ndim == 1
        if one_column:
            coeffs = coeffs.reshape(-1, 1)
        rows = self.output_rows if output_rows is None else _index_vector(output_rows, "quantity output_rows", self.n_features)
        basis = self.basis if rows is None else self.basis[rows, :]
        offset = self.offset if rows is None else self.offset[rows]
        values = offset[:, None] + basis @ coeffs
        return np.ascontiguousarray(values[:, 0] if one_column else values)

    def relative_error(self, reference_values: Any) -> float:
        reference = np.asarray(reference_values, dtype=float)
        reconstructed = self.reconstruct_from_samples(self.sample(reference))
        return float(np.linalg.norm(reconstructed - reference) / max(np.linalg.norm(reference), 1.0e-300))

    def to_native_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "basis": self.basis,
            "sample_rows": self.sample_rows,
            "offset": self.offset,
            "output_rows": np.zeros(0, dtype=np.int64) if self.output_rows is None else self.output_rows,
            "sample_to_coefficients": self.sample_to_coefficients,
            "rcond": -1.0 if self.rcond is None else float(self.rcond),
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_native_dict(cls, payload: Mapping[str, Any]) -> "GappyPODQuantityOperator":
        output_rows = payload.get("output_rows")
        if output_rows is not None and np.asarray(output_rows).size == 0:
            output_rows = None
        rcond = float(payload.get("rcond", -1.0))
        return cls(
            basis=payload["basis"],
            sample_rows=payload["sample_rows"],
            offset=payload.get("offset"),
            output_rows=output_rows,
            sample_to_coefficients=payload.get("sample_to_coefficients"),
            rcond=None if rcond < 0.0 else rcond,
            name=str(payload.get("name", "quantity")),
            metadata=payload.get("metadata", {}),
        )


def fit_gappy_pod_quantity_operator(
    *,
    snapshots: Any | None = None,
    basis: Any | None = None,
    n_modes: int | None = None,
    energy: float | None = None,
    center: bool = False,
    sample_rows: Any | None = None,
    n_sample_rows: int | None = None,
    method: str = "qdeim",
    offset: Any | None = None,
    output_rows: Any | None = None,
    sample_to_coefficients: Any | None = None,
    rcond: float | None = None,
    name: str = "quantity",
    metadata: Mapping[str, Any] | None = None,
) -> GappyPODQuantityOperator:
    """Fit a gappy POD quantity/reaction operator from snapshots or a basis."""

    if basis is None:
        if snapshots is None:
            raise ValueError("either snapshots or basis must be provided.")
        collateral = fit_collateral_basis(snapshots, n_modes=n_modes, energy=energy, center=center)
        basis_arr = collateral.basis
        if offset is None:
            offset_arr = collateral.snapshot_mean if collateral.snapshot_mean is not None else np.zeros(
                basis_arr.shape[0],
                dtype=float,
            )
        else:
            offset_arr = offset
    else:
        basis_arr = _finite_matrix(basis, "quantity basis")
        offset_arr = np.zeros(basis_arr.shape[0], dtype=float) if offset is None else offset

    method_name = str(method).strip().lower()
    if sample_rows is None:
        if method_name == "deim":
            rows = build_deim_interpolation_rule(basis_arr).rows
        elif method_name in {"qdeim", "qr"}:
            rows = build_qdeim_interpolation_rule(basis_arr).rows
        else:
            raise ValueError("method must be 'deim' or 'qdeim' when sample_rows is not provided.")
    else:
        rows = _index_vector(sample_rows, "quantity sample_rows", basis_arr.shape[0])
    if n_sample_rows is not None:
        rows = _oversample_rows(basis_arr, rows, int(n_sample_rows))
    return GappyPODQuantityOperator(
        basis=basis_arr,
        sample_rows=rows,
        offset=offset_arr,
        output_rows=output_rows,
        sample_to_coefficients=sample_to_coefficients,
        rcond=rcond,
        name=name,
        metadata={
            "method": method_name,
            "centered": bool(center),
            **dict(metadata or {}),
        },
    )


__all__ = [
    "GappyPODQuantityOperator",
    "fit_gappy_pod_quantity_operator",
]
