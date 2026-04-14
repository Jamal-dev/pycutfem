from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib

import numpy as np


def _normalize_persistence(persistence: str) -> str:
    value = str(persistence).strip().lower()
    if value not in {"step", "iteration"}:
        raise ValueError(
            f"Unsupported persistence {persistence!r}; use 'step' or 'iteration'."
        )
    return value


@dataclass(slots=True)
class QuadratureLayout:
    entity_kind: str
    cell_type: str
    quadrature_order: int
    reference_points: np.ndarray
    reference_weights: np.ndarray

    def __post_init__(self) -> None:
        self.entity_kind = str(self.entity_kind).strip().lower()
        self.cell_type = str(self.cell_type).strip().lower()
        self.quadrature_order = int(self.quadrature_order)
        self.reference_points = np.asarray(self.reference_points, dtype=float)
        self.reference_weights = np.asarray(self.reference_weights, dtype=float).reshape(-1)
        if self.entity_kind != "volume_cell":
            raise ValueError(f"Unsupported quadrature entity_kind {self.entity_kind!r}.")
        if self.reference_points.ndim != 2 or self.reference_points.shape[1] != 2:
            raise ValueError("reference_points must have shape (n_qp, 2).")
        if self.reference_points.shape[0] != self.reference_weights.shape[0]:
            raise ValueError(
                "reference_points/reference_weights length mismatch: "
                f"{self.reference_points.shape[0]} vs {self.reference_weights.shape[0]}"
            )

    @property
    def n_qp(self) -> int:
        return int(self.reference_points.shape[0])

    @property
    def signature(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(self.entity_kind.encode("utf-8"))
        h.update(b"|")
        h.update(self.cell_type.encode("utf-8"))
        h.update(b"|")
        h.update(str(self.quadrature_order).encode("ascii"))
        h.update(b"|")
        h.update(np.ascontiguousarray(self.reference_points, dtype=np.float64).tobytes())
        h.update(b"|")
        h.update(np.ascontiguousarray(self.reference_weights, dtype=np.float64).tobytes())
        return h.hexdigest()

    def validate_against(
        self,
        *,
        reference_points: np.ndarray,
        reference_weights: np.ndarray,
        context: str,
        atol: float = 1.0e-12,
        rtol: float = 1.0e-12,
    ) -> None:
        ref_pts = np.asarray(reference_points, dtype=float)
        ref_wts = np.asarray(reference_weights, dtype=float).reshape(-1)
        if ref_pts.shape != self.reference_points.shape:
            raise ValueError(
                f"{context}: quadrature-state points shape mismatch: "
                f"expected {self.reference_points.shape}, got {ref_pts.shape}."
            )
        if ref_wts.shape != self.reference_weights.shape:
            raise ValueError(
                f"{context}: quadrature-state weights shape mismatch: "
                f"expected {self.reference_weights.shape}, got {ref_wts.shape}."
            )
        if not np.allclose(ref_pts, self.reference_points, atol=atol, rtol=rtol):
            raise ValueError(
                f"{context}: quadrature-state reference points do not match the active quadrature rule."
            )
        if not np.allclose(ref_wts, self.reference_weights, atol=atol, rtol=rtol):
            raise ValueError(
                f"{context}: quadrature-state reference weights do not match the active quadrature rule."
            )


@dataclass(slots=True)
class CellStateField:
    name: str
    persistence: str
    _values: np.ndarray
    _staged: np.ndarray

    def __post_init__(self) -> None:
        self.persistence = _normalize_persistence(self.persistence)
        self._values = np.asarray(self._values, dtype=float)
        self._staged = np.asarray(self._staged, dtype=float)
        if self._values.ndim == 0:
            raise ValueError("CellStateField requires one value per cell.")
        if self._staged.shape != self._values.shape:
            raise ValueError(
                f"staged shape {self._staged.shape} does not match values shape {self._values.shape}"
            )

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def staged_values(self) -> np.ndarray:
        return self._staged

    @property
    def n_cells(self) -> int:
        return int(self._values.shape[0])

    @property
    def tensor_shape(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self._values.shape[1:])

    @property
    def shape(self) -> tuple[int, ...]:
        return self._values.shape

    def _validate(self, values) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.shape != self._values.shape:
            raise ValueError(
                f"{self.name!r} expected shape {self._values.shape}, got {arr.shape}"
            )
        return arr

    def assign(self, values) -> None:
        self._values[...] = self._validate(values)

    def stage(self, values) -> None:
        self._staged[...] = self._validate(values)

    def commit(self) -> None:
        self._values[...] = self._staged

    def rollback(self) -> None:
        self._staged[...] = self._values

    def coefficient(self, *, jit_name: str | None = None):
        from .coefficient import CellStateCoefficient

        return CellStateCoefficient(self, jit_name=jit_name)


@dataclass(slots=True)
class QuadratureStateField:
    name: str
    persistence: str
    layout: QuadratureLayout
    _values: np.ndarray
    _staged: np.ndarray

    def __post_init__(self) -> None:
        self.persistence = _normalize_persistence(self.persistence)
        self._values = np.asarray(self._values, dtype=float)
        self._staged = np.asarray(self._staged, dtype=float)
        if self._values.ndim < 2:
            raise ValueError("QuadratureStateField requires shape (n_entities, n_qp, ...).")
        if int(self._values.shape[1]) != int(self.layout.n_qp):
            raise ValueError(
                f"{self.name!r} expected n_qp={int(self.layout.n_qp)}, got {int(self._values.shape[1])}."
            )
        if self._staged.shape != self._values.shape:
            raise ValueError(
                f"staged shape {self._staged.shape} does not match values shape {self._values.shape}"
            )

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def staged_values(self) -> np.ndarray:
        return self._staged

    @property
    def n_entities(self) -> int:
        return int(self._values.shape[0])

    @property
    def tensor_shape(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self._values.shape[2:])

    @property
    def shape(self) -> tuple[int, ...]:
        return self._values.shape

    def _validate(self, values) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.shape != self._values.shape:
            raise ValueError(
                f"{self.name!r} expected shape {self._values.shape}, got {arr.shape}"
            )
        return arr

    def assign(self, values) -> None:
        self._values[...] = self._validate(values)

    def stage(self, values) -> None:
        self._staged[...] = self._validate(values)

    def commit(self) -> None:
        self._values[...] = self._staged

    def rollback(self) -> None:
        self._staged[...] = self._values

    def coefficient(self, *, jit_name: str | None = None):
        from .coefficient import QuadratureStateCoefficient

        return QuadratureStateCoefficient(self, jit_name=jit_name)


class StateRegistry(Mapping[str, CellStateField | QuadratureStateField]):
    def __init__(self) -> None:
        self._fields: dict[str, CellStateField | QuadratureStateField] = {}

    def __getitem__(self, key: str) -> CellStateField | QuadratureStateField:
        return self._fields[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def register_cell(
        self,
        name: str,
        *,
        values=None,
        n_cells: int | None = None,
        tensor_shape: tuple[int, ...] = (),
        dtype=float,
        persistence: str = "step",
        copy: bool = True,
    ) -> CellStateField:
        key = str(name)
        if key in self._fields:
            raise KeyError(f"State field {key!r} is already registered.")

        if values is None:
            if n_cells is None:
                raise ValueError("n_cells is required when values is not provided.")
            arr = np.zeros((int(n_cells), *tuple(int(v) for v in tensor_shape)), dtype=dtype)
        else:
            arr = np.asarray(values, dtype=dtype)
            if arr.ndim == 0:
                raise ValueError("Cell state values must include the cell axis.")
            expected_shape = tuple(int(v) for v in tensor_shape)
            if expected_shape == () and arr.ndim == 2 and arr.shape[1:] == (1,):
                # Accept legacy column-vector scalar storage when the caller
                # explicitly requests a scalar cell field.
                arr = arr.reshape(arr.shape[0])
            if n_cells is not None and int(arr.shape[0]) != int(n_cells):
                raise ValueError(
                    f"State field {key!r} expected {int(n_cells)} cells, got {int(arr.shape[0])}."
                )
            if expected_shape and tuple(arr.shape[1:]) != expected_shape:
                raise ValueError(
                    f"State field {key!r} expected tensor shape {expected_shape}, got {arr.shape[1:]}."
                )

        current = arr.copy() if copy else arr
        staged = np.asarray(current, dtype=float).copy()
        field = CellStateField(
            name=key,
            persistence=persistence,
            _values=current,
            _staged=staged,
        )
        self._fields[key] = field
        return field

    def register_quadrature(
        self,
        name: str,
        *,
        layout: QuadratureLayout,
        values=None,
        n_entities: int | None = None,
        tensor_shape: tuple[int, ...] = (),
        dtype=float,
        persistence: str = "step",
        copy: bool = True,
    ) -> QuadratureStateField:
        key = str(name)
        if key in self._fields:
            raise KeyError(f"State field {key!r} is already registered.")

        expected_shape = tuple(int(v) for v in tensor_shape)
        if values is None:
            if n_entities is None:
                raise ValueError("n_entities is required when values is not provided.")
            arr = np.zeros(
                (int(n_entities), int(layout.n_qp), *expected_shape),
                dtype=dtype,
            )
        else:
            arr = np.asarray(values, dtype=dtype)
            if arr.ndim < 2:
                raise ValueError("Quadrature state values must include entity and quadrature axes.")
            if expected_shape == () and arr.ndim == 3 and arr.shape[2:] == (1,):
                arr = arr.reshape(arr.shape[0], arr.shape[1])
            if n_entities is not None and int(arr.shape[0]) != int(n_entities):
                raise ValueError(
                    f"State field {key!r} expected {int(n_entities)} entities, got {int(arr.shape[0])}."
                )
            if int(arr.shape[1]) != int(layout.n_qp):
                raise ValueError(
                    f"State field {key!r} expected n_qp={int(layout.n_qp)}, got {int(arr.shape[1])}."
                )
            if expected_shape and tuple(arr.shape[2:]) != expected_shape:
                raise ValueError(
                    f"State field {key!r} expected tensor shape {expected_shape}, got {arr.shape[2:]}."
                )

        current = arr.copy() if copy else arr
        staged = np.asarray(current, dtype=float).copy()
        field = QuadratureStateField(
            name=key,
            persistence=persistence,
            layout=layout,
            _values=current,
            _staged=staged,
        )
        self._fields[key] = field
        return field

    def commit_step(self) -> None:
        for field in self._fields.values():
            if field.persistence == "step":
                field.commit()

    def rollback_step(self) -> None:
        for field in self._fields.values():
            if field.persistence == "step":
                field.rollback()

    def reset_iteration(self) -> None:
        for field in self._fields.values():
            if field.persistence == "iteration":
                field.rollback()
