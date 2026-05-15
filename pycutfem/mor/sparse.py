"""Sparse MOR operator helpers for native hyper-reduction targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class NativeSparseMatrix:
    """Validated sparse matrix payload for native MOR backends.

    The runtime layout is CSR.  COO/CSC/SciPy inputs are accepted by constructors
    and normalized to CSR so the C++ online path has one compact ABI.
    """

    shape: tuple[int, int]
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    layout: str = "csr"

    def __post_init__(self) -> None:
        rows, cols = (int(self.shape[0]), int(self.shape[1]))
        if rows < 0 or cols < 0:
            raise ValueError("sparse matrix shape must be nonnegative.")
        indptr = np.asarray(self.indptr, dtype=np.int64).reshape(-1)
        indices = np.asarray(self.indices, dtype=np.int64).reshape(-1)
        data = np.asarray(self.data, dtype=float).reshape(-1)
        if str(self.layout).lower() != "csr":
            raise ValueError("NativeSparseMatrix runtime layout must be 'csr'.")
        if indptr.size != rows + 1:
            raise ValueError("CSR indptr length must be n_rows + 1.")
        if indices.size != data.size:
            raise ValueError("CSR indices and data must have the same length.")
        if indptr.size and int(indptr[0]) != 0:
            raise ValueError("CSR indptr must start at zero.")
        if np.any(indptr[1:] < indptr[:-1]):
            raise ValueError("CSR indptr must be nondecreasing.")
        if int(indptr[-1]) != int(indices.size):
            raise ValueError("CSR indptr[-1] must equal nnz.")
        if np.any(indices < 0) or np.any(indices >= cols):
            raise ValueError("CSR column indices are out of range.")
        if not np.all(np.isfinite(data)):
            raise ValueError("CSR data must be finite.")
        for row in range(rows):
            start, stop = int(indptr[row]), int(indptr[row + 1])
            if stop - start > 1:
                row_indices = indices[start:stop]
                if np.any(row_indices[1:] <= row_indices[:-1]):
                    raise ValueError("CSR row indices must be strictly increasing and duplicate-free.")
        object.__setattr__(self, "shape", (rows, cols))
        object.__setattr__(self, "indptr", np.ascontiguousarray(indptr, dtype=np.int64))
        object.__setattr__(self, "indices", np.ascontiguousarray(indices, dtype=np.int64))
        object.__setattr__(self, "data", np.ascontiguousarray(data, dtype=np.float64))
        object.__setattr__(self, "layout", "csr")

    @property
    def nnz(self) -> int:
        return int(self.data.size)

    @classmethod
    def from_dense(cls, matrix: Any, *, drop_tol: float = 0.0) -> "NativeSparseMatrix":
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim != 2:
            raise ValueError("dense sparse-source matrix must be rank-2.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("dense sparse-source matrix must be finite.")
        rows, cols = int(arr.shape[0]), int(arr.shape[1])
        indptr = np.zeros(rows + 1, dtype=np.int64)
        indices_parts: list[np.ndarray] = []
        data_parts: list[np.ndarray] = []
        threshold = abs(float(drop_tol))
        nnz = 0
        for row in range(rows):
            cols_i = np.flatnonzero(np.abs(arr[row]) > threshold).astype(np.int64, copy=False)
            vals_i = arr[row, cols_i].astype(float, copy=False)
            indices_parts.append(cols_i)
            data_parts.append(vals_i)
            nnz += int(cols_i.size)
            indptr[row + 1] = nnz
        indices = np.concatenate(indices_parts) if indices_parts else np.zeros(0, dtype=np.int64)
        data = np.concatenate(data_parts) if data_parts else np.zeros(0, dtype=float)
        return cls(shape=(rows, cols), indptr=indptr, indices=indices, data=data)

    @classmethod
    def from_coo(
        cls,
        row: Any,
        col: Any,
        data: Any,
        *,
        shape: tuple[int, int],
    ) -> "NativeSparseMatrix":
        rows, cols = int(shape[0]), int(shape[1])
        rr = np.asarray(row, dtype=np.int64).reshape(-1)
        cc = np.asarray(col, dtype=np.int64).reshape(-1)
        vv = np.asarray(data, dtype=float).reshape(-1)
        if rr.size != cc.size or rr.size != vv.size:
            raise ValueError("COO row, col, and data arrays must have matching sizes.")
        if np.any(rr < 0) or np.any(rr >= rows) or np.any(cc < 0) or np.any(cc >= cols):
            raise ValueError("COO indices are out of range.")
        if not np.all(np.isfinite(vv)):
            raise ValueError("COO data must be finite.")
        if rr.size == 0:
            return cls(shape=(rows, cols), indptr=np.zeros(rows + 1, dtype=np.int64), indices=[], data=[])
        order = np.lexsort((cc, rr))
        rr = rr[order]
        cc = cc[order]
        vv = vv[order]
        indptr = np.zeros(rows + 1, dtype=np.int64)
        indices: list[int] = []
        values: list[float] = []
        cursor = 0
        for r in range(rows):
            while cursor < rr.size and int(rr[cursor]) == r:
                c = int(cc[cursor])
                value = 0.0
                while cursor < rr.size and int(rr[cursor]) == r and int(cc[cursor]) == c:
                    value += float(vv[cursor])
                    cursor += 1
                if value != 0.0:
                    indices.append(c)
                    values.append(value)
            indptr[r + 1] = len(indices)
        return cls(shape=(rows, cols), indptr=indptr, indices=np.asarray(indices), data=np.asarray(values))

    @classmethod
    def from_csr(cls, indptr: Any, indices: Any, data: Any, *, shape: tuple[int, int]) -> "NativeSparseMatrix":
        return cls(shape=(int(shape[0]), int(shape[1])), indptr=indptr, indices=indices, data=data)

    @classmethod
    def from_csc(cls, indptr: Any, indices: Any, data: Any, *, shape: tuple[int, int]) -> "NativeSparseMatrix":
        n_rows, n_cols = int(shape[0]), int(shape[1])
        col_ptr = np.asarray(indptr, dtype=np.int64).reshape(-1)
        row_idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        vals = np.asarray(data, dtype=float).reshape(-1)
        if col_ptr.size != n_cols + 1:
            raise ValueError("CSC indptr length must be n_cols + 1.")
        cols = []
        rows = []
        out_vals = []
        for col in range(n_cols):
            start, stop = int(col_ptr[col]), int(col_ptr[col + 1])
            rows.extend(int(v) for v in row_idx[start:stop])
            cols.extend([col] * (stop - start))
            out_vals.extend(float(v) for v in vals[start:stop])
        return cls.from_coo(rows, cols, out_vals, shape=(n_rows, n_cols))

    @classmethod
    def from_scipy(cls, matrix: Any) -> "NativeSparseMatrix":
        if not hasattr(matrix, "tocsr"):
            raise TypeError("from_scipy expects an object with a tocsr() method.")
        csr = matrix.tocsr()
        csr.sum_duplicates()
        csr.sort_indices()
        return cls.from_csr(csr.indptr, csr.indices, csr.data, shape=tuple(csr.shape))

    @classmethod
    def from_native_dict(cls, payload: Mapping[str, Any]) -> "NativeSparseMatrix":
        layout = str(payload.get("layout", "csr")).lower()
        if layout != "csr":
            raise ValueError("native sparse payload must use CSR layout.")
        shape_raw = payload["shape"]
        return cls.from_csr(payload["indptr"], payload["indices"], payload["data"], shape=(int(shape_raw[0]), int(shape_raw[1])))

    @classmethod
    def coerce(cls, value: Any) -> "NativeSparseMatrix":
        if isinstance(value, NativeSparseMatrix):
            return value
        if isinstance(value, Mapping):
            return cls.from_native_dict(value)
        if hasattr(value, "tocsr"):
            return cls.from_scipy(value)
        return cls.from_dense(value)

    def to_native_dict(self) -> dict[str, Any]:
        return {
            "layout": "csr",
            "shape": np.asarray(self.shape, dtype=np.int64),
            "indptr": self.indptr,
            "indices": self.indices,
            "data": self.data,
            "nnz": int(self.nnz),
        }

    def to_dense(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=float)
        for row in range(self.shape[0]):
            start, stop = int(self.indptr[row]), int(self.indptr[row + 1])
            out[row, self.indices[start:stop]] = self.data[start:stop]
        return out

    def matvec(self, vector: Any) -> np.ndarray:
        vec = np.asarray(vector, dtype=float).reshape(-1)
        if vec.size != self.shape[1]:
            raise ValueError("vector size must match sparse matrix column count.")
        out = np.zeros(self.shape[0], dtype=float)
        for row in range(self.shape[0]):
            start, stop = int(self.indptr[row]), int(self.indptr[row + 1])
            out[row] = float(np.dot(self.data[start:stop], vec[self.indices[start:stop]]))
        return out

    def matmat(self, matrix: Any) -> np.ndarray:
        mat = np.asarray(matrix, dtype=float)
        if mat.ndim != 2 or mat.shape[0] != self.shape[1]:
            raise ValueError("matrix rows must match sparse matrix column count.")
        out = np.zeros((self.shape[0], mat.shape[1]), dtype=float)
        for row in range(self.shape[0]):
            start, stop = int(self.indptr[row]), int(self.indptr[row + 1])
            if start != stop:
                out[row, :] = self.data[start:stop] @ mat[self.indices[start:stop], :]
        return out


def is_sparse_matrix_like(value: Any) -> bool:
    return isinstance(value, NativeSparseMatrix) or isinstance(value, Mapping) or hasattr(value, "tocsr")


def apply_sparse_gnat_lift(
    sparse_lift: NativeSparseMatrix | Mapping[str, Any] | Any,
    sampled_residual: np.ndarray,
    sampled_trial_jacobian: np.ndarray,
    *,
    backend: str = "python",
) -> tuple[np.ndarray, np.ndarray]:
    lift = NativeSparseMatrix.coerce(sparse_lift)
    residual = np.asarray(sampled_residual, dtype=float).reshape(-1)
    trial = np.asarray(sampled_trial_jacobian, dtype=float)
    if lift.shape[1] != residual.size or trial.ndim != 2 or lift.shape[1] != trial.shape[0]:
        raise ValueError("GNAT sparse lift columns must match sampled residual/Jacobian rows.")
    if str(backend).strip().lower() in {"cpp", "c++"}:
        from .cpp_backend.sparse_gnat import module as _sparse_gnat_module

        out_residual, out_trial = _sparse_gnat_module().apply_sparse_gnat_lift(
            lift.to_native_dict(),
            residual,
            trial,
        )
        return np.asarray(out_residual, dtype=float).reshape(-1), np.asarray(out_trial, dtype=float)
    return lift.matvec(residual), lift.matmat(trial)


def sparse_gnat_normal_equations(
    sparse_lift: NativeSparseMatrix | Mapping[str, Any] | Any,
    sampled_residual: np.ndarray,
    sampled_trial_jacobian: np.ndarray,
    *,
    backend: str = "python",
) -> dict[str, Any]:
    lift = NativeSparseMatrix.coerce(sparse_lift)
    residual, trial = apply_sparse_gnat_lift(
        lift,
        sampled_residual,
        sampled_trial_jacobian,
        backend=backend,
    )
    if str(backend).strip().lower() in {"cpp", "c++"}:
        from .cpp_backend.sparse_gnat import module as _sparse_gnat_module

        raw = _sparse_gnat_module().sparse_gnat_normal_equations(
            lift.to_native_dict(),
            np.asarray(sampled_residual, dtype=float).reshape(-1),
            np.asarray(sampled_trial_jacobian, dtype=float),
        )
        return {
            "normal_matrix": np.asarray(raw["normal_matrix"], dtype=float),
            "normal_rhs": np.asarray(raw["normal_rhs"], dtype=float).reshape(-1),
            "lifted_residual_norm": float(raw["lifted_residual_norm"]),
            "nnz": int(raw["nnz"]),
            "path": str(raw["path"]),
        }
    return {
        "normal_matrix": np.asarray(trial.T @ trial, dtype=float),
        "normal_rhs": np.asarray(-(trial.T @ residual), dtype=float).reshape(-1),
        "lifted_residual_norm": float(np.linalg.norm(residual)),
        "nnz": int(lift.nnz),
        "path": "python_csr",
    }


__all__ = [
    "NativeSparseMatrix",
    "apply_sparse_gnat_lift",
    "is_sparse_matrix_like",
    "sparse_gnat_normal_equations",
]
