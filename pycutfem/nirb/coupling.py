from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from pycutfem.mor.interface import InterfaceRestriction
from pycutfem.mor.io import load_model


@dataclass(frozen=True)
class CouplingIterationRecord:
    time: float
    subiteration: int
    converged: bool
    solid_time: float
    fluid_time: float
    reduced_displacement: np.ndarray


@dataclass
class CouplingTrace:
    records: list[CouplingIterationRecord] = field(default_factory=list)

    def append(self, record: CouplingIterationRecord) -> None:
        self.records.append(record)

    def accumulated_iterations(self) -> int:
        return len(self.records)

    def reduced_history(self) -> np.ndarray:
        if not self.records:
            return np.zeros((0, 0), dtype=float)
        return np.column_stack([record.reduced_displacement for record in self.records])


@dataclass(frozen=True)
class NIRBSolidPrediction:
    full_displacement: np.ndarray | None
    elapsed_s: float
    interface_displacement: np.ndarray | None = None
    reduced_displacement: np.ndarray | None = None


@dataclass
class NIRBSolidPredictor:
    """Adapter implementing the paper's Algorithm 2 solid ROM query."""

    model: Any
    full_shape: tuple[int, ...] | None = None
    interface_matrix: np.ndarray | None = None
    interface_shape: tuple[int, ...] | None = None
    zero_load_tolerance: float = 0.0
    _interface_decoder: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.interface_matrix is not None:
            restriction = InterfaceRestriction(matrix=np.asarray(self.interface_matrix, dtype=float))
            self._interface_decoder = restriction.restrict_decoder(self.model.decoder)
        elif getattr(self.model, "interface_restriction", None) is not None:
            self._interface_decoder = self.model.interface_restriction.restrict_decoder(self.model.decoder)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        full_shape: tuple[int, ...] | None = None,
        interface_matrix: np.ndarray | None = None,
        interface_shape: tuple[int, ...] | None = None,
        zero_load_tolerance: float = 0.0,
    ) -> "NIRBSolidPredictor":
        return cls(
            model=load_model(path),
            full_shape=full_shape,
            interface_matrix=interface_matrix,
            interface_shape=interface_shape,
            zero_load_tolerance=float(zero_load_tolerance),
        )

    def _zero_full_displacement(self) -> np.ndarray:
        if self.full_shape is None:
            raise RuntimeError("full_shape is required to return a zero full displacement")
        return np.zeros(int(np.prod(self.full_shape)), dtype=float)

    def _zero_interface_displacement(self) -> np.ndarray:
        if self.interface_shape is not None:
            return np.zeros(int(np.prod(self.interface_shape)), dtype=float)
        if self._interface_decoder is not None:
            return np.zeros(int(self._interface_decoder.linear_basis.shape[0]), dtype=float)
        raise RuntimeError("interface_shape or an interface decoder is required for interface prediction")

    def predict_reduced(self, interface_load: np.ndarray) -> NIRBSolidPrediction:
        force_vector = np.asarray(interface_load, dtype=float).reshape(-1)
        if np.linalg.norm(force_vector) <= float(self.zero_load_tolerance):
            return NIRBSolidPrediction(
                full_displacement=None,
                interface_displacement=None,
                reduced_displacement=np.zeros((int(self.model.decoder.n_linear_modes),), dtype=float),
                elapsed_s=0.0,
            )
        force = force_vector.reshape(-1, 1)
        started = perf_counter()
        reduced = np.asarray(self.model.predict_reduced(force), dtype=float)
        elapsed = perf_counter() - started
        if reduced.ndim != 2 or reduced.shape[1] != 1:
            raise ValueError("NIRB solid model must return a single feature-major reduced displacement column")
        return NIRBSolidPrediction(
            full_displacement=None,
            interface_displacement=None,
            reduced_displacement=reduced[:, 0],
            elapsed_s=float(elapsed),
        )

    def reconstruct_full(self, reduced_displacement: np.ndarray) -> np.ndarray:
        reduced = np.asarray(reduced_displacement, dtype=float).reshape(-1, 1)
        displacement = np.asarray(self.model.decoder.decode(reduced), dtype=float)
        if displacement.ndim != 2 or displacement.shape[1] != 1:
            raise ValueError("NIRB solid model must return a single feature-major displacement column")
        vector = displacement[:, 0]
        if self.full_shape is not None and vector.size != int(np.prod(self.full_shape)):
            raise ValueError(
                "NIRB solid displacement size does not match the target field: "
                f"{vector.size} != {int(np.prod(self.full_shape))}"
            )
        return vector

    def reconstruct_interface(self, reduced_displacement: np.ndarray) -> np.ndarray:
        reduced = np.asarray(reduced_displacement, dtype=float).reshape(-1, 1)
        if self._interface_decoder is None:
            if self.interface_shape is not None:
                raise RuntimeError(
                    "NIRB interface prediction requires an interface restriction matrix "
                    "when interface_shape is provided."
                )
            return self.reconstruct_full(reduced[:, 0])
        displacement = np.asarray(self._interface_decoder.decode(reduced), dtype=float)
        if displacement.ndim != 2 or displacement.shape[1] != 1:
            raise ValueError("NIRB interface decoder must return a single feature-major column")
        vector = displacement[:, 0]
        if self.interface_shape is not None and vector.size != int(np.prod(self.interface_shape)):
            raise ValueError(
                "NIRB interface displacement size does not match the target field: "
                f"{vector.size} != {int(np.prod(self.interface_shape))}"
            )
        return vector

    def predict_interface(self, interface_load: np.ndarray) -> NIRBSolidPrediction:
        force_vector = np.asarray(interface_load, dtype=float).reshape(-1)
        if np.linalg.norm(force_vector) <= float(self.zero_load_tolerance):
            return NIRBSolidPrediction(
                full_displacement=None,
                interface_displacement=self._zero_interface_displacement(),
                reduced_displacement=np.zeros((int(self.model.decoder.n_linear_modes),), dtype=float),
                elapsed_s=0.0,
            )
        prediction = self.predict_reduced(force_vector)
        started = perf_counter()
        interface_displacement = self.reconstruct_interface(prediction.reduced_displacement)
        elapsed = prediction.elapsed_s + (perf_counter() - started)
        return NIRBSolidPrediction(
            full_displacement=None,
            interface_displacement=interface_displacement,
            reduced_displacement=prediction.reduced_displacement,
            elapsed_s=float(elapsed),
        )

    def predict(self, interface_load: np.ndarray) -> NIRBSolidPrediction:
        force_vector = np.asarray(interface_load, dtype=float).reshape(-1)
        if self.full_shape is not None and np.linalg.norm(force_vector) <= float(self.zero_load_tolerance):
            return NIRBSolidPrediction(
                full_displacement=self._zero_full_displacement(),
                elapsed_s=0.0,
            )
        prediction = self.predict_reduced(force_vector)
        started = perf_counter()
        full_displacement = self.reconstruct_full(prediction.reduced_displacement)
        elapsed = prediction.elapsed_s + (perf_counter() - started)
        return NIRBSolidPrediction(
            full_displacement=full_displacement,
            interface_displacement=None,
            reduced_displacement=prediction.reduced_displacement,
            elapsed_s=float(elapsed),
        )
