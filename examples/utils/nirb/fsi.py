from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from pycutfem.mor.interface import InterfaceRestriction
from pycutfem.mor.io import load_model
from pycutfem.mor.snapshots import NamedSnapshotBatch
from pycutfem.mor.nirb.reduced_spaces import ReducedOutputDecoder


def _as_snapshot_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("expected a feature-major snapshot matrix")
    return matrix


def _array_path(co_sim_dir: Path, key: str) -> Path:
    name = key if key.endswith(".npy") else f"{key}.npy"
    return co_sim_dir / name


def _resolve_co_sim_dir(path: str | Path) -> tuple[Path, Path]:
    root = Path(path)
    if (root / "coSimData").is_dir():
        return root / "coSimData", root
    return root, root.parent


def _load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_snapshot_metadata(path: Path, n_snapshots: int) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if not path.exists():
        return None, None, None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != n_snapshots:
        return None, None, None
    times = np.asarray([float(row.get("time_s", 0.0) or 0.0) for row in rows], dtype=float)
    subiterations = np.asarray([int(row.get("coupling_iter", 0) or 0) for row in rows], dtype=int)
    converged = np.asarray(
        [str(row.get("converged", "")).strip().lower() in {"1", "true", "yes"} for row in rows],
        dtype=bool,
    )
    return times, subiterations, converged


def load_cosim_snapshot_batch(
    path: str | Path,
    *,
    force_key: str = "load_guess_data",
    displacement_key: str = "disp_data",
    input_field: str = "interface_load",
    output_field: str = "solid_displacement",
) -> NamedSnapshotBatch:
    """Load Example-2-style FSI ``coSimData`` arrays as a generic snapshot batch."""

    co_sim_dir, run_root = _resolve_co_sim_dir(path)
    force_path = _array_path(co_sim_dir, force_key)
    if not force_path.exists() and force_key == "load_guess_data":
        force_path = _array_path(co_sim_dir, "load_data")
        force_key = "load_data"
    displacement_path = _array_path(co_sim_dir, displacement_key)
    if not force_path.exists():
        raise FileNotFoundError(force_path)
    if not displacement_path.exists():
        raise FileNotFoundError(displacement_path)

    input_snapshots = _as_snapshot_matrix(np.load(force_path))
    output_snapshots = _as_snapshot_matrix(np.load(displacement_path))
    if input_snapshots.shape[1] != output_snapshots.shape[1]:
        raise ValueError(
            "coSimData input and output snapshot counts differ: "
            f"{input_snapshots.shape[1]} != {output_snapshots.shape[1]}"
        )

    n_snapshots = int(input_snapshots.shape[1])
    times, subiterations, converged = _load_snapshot_metadata(run_root / "snapshot_metadata.csv", n_snapshots)
    solid_times = None
    fluid_times = None
    solid_time_path = co_sim_dir / "structure_time.npy"
    fluid_time_path = co_sim_dir / "fluid_time.npy"
    if solid_time_path.exists():
        values = np.asarray(np.load(solid_time_path), dtype=float).reshape(-1)
        if values.size == n_snapshots:
            solid_times = values
    if fluid_time_path.exists():
        values = np.asarray(np.load(fluid_time_path), dtype=float).reshape(-1)
        if values.size == n_snapshots:
            fluid_times = values

    summary = _load_json_if_present(run_root / "summary.json")
    parameters = None
    if "reynolds" in summary:
        parameters = np.full((n_snapshots, 1), float(summary["reynolds"]), dtype=float)

    sample_metadata = []
    for index in range(n_snapshots):
        sample_metadata.append(
            {
                "subiteration": None if subiterations is None else int(subiterations[index]),
                "coupling_iter": None if subiterations is None else int(subiterations[index]),
                "solid_time": None if solid_times is None else float(solid_times[index]),
                "fluid_time": None if fluid_times is None else float(fluid_times[index]),
            }
        )

    return NamedSnapshotBatch(
        fields={
            str(input_field): input_snapshots,
            str(output_field): output_snapshots,
        },
        parameters=parameters,
        times=times,
        converged=converged,
        sample_metadata=tuple(sample_metadata),
        metadata={
            "source": "coSimData",
            "co_sim_dir": str(co_sim_dir),
            "run_root": str(run_root),
            "input_field": str(input_field),
            "output_field": str(output_field),
            "force_key": force_key,
            "displacement_key": displacement_key,
            "force_path": str(force_path),
            "displacement_path": str(displacement_path),
            "summary": summary,
        },
    )


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
    reduced_interface_displacement: np.ndarray | None = None


@dataclass(frozen=True)
class NIRBInterfaceTangentCorrector:
    """Secant interface-compliance correction for NIRB solid predictions.

    The tangent matrices map a flattened interface-load increment to either a
    flattened full solid displacement increment or a flattened interface
    displacement increment.  They are trained from consecutive FSI coupling
    states and are used online as a local impedance anchor:

        d_k ~= d_{k-1} + C_Gamma (f_k - f_{k-1}).
    """

    load_coords: np.ndarray
    interface_coords: np.ndarray
    full_tangent: np.ndarray | None = None
    interface_tangent: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_npz(cls, path: str | Path) -> "NIRBInterfaceTangentCorrector":
        with np.load(Path(path), allow_pickle=True) as data:
            load_coords = np.asarray(data["load_coords"], dtype=float)
            interface_coords = np.asarray(data["interface_coords"], dtype=float)
            full_tangent = np.asarray(data["full_tangent"], dtype=float) if "full_tangent" in data else None
            interface_tangent = (
                np.asarray(data["interface_tangent"], dtype=float) if "interface_tangent" in data else None
            )
            metadata: dict[str, Any] = {}
            if "metadata_json" in data:
                try:
                    import json

                    metadata = json.loads(str(np.asarray(data["metadata_json"]).reshape(())))
                except Exception:
                    metadata = {}
        return cls(
            load_coords=load_coords,
            interface_coords=interface_coords,
            full_tangent=full_tangent,
            interface_tangent=interface_tangent,
            metadata=metadata,
        )

    @property
    def n_load_dofs(self) -> int:
        return int(np.asarray(self.load_coords, dtype=float).shape[0] * 2)

    @property
    def n_interface_dofs(self) -> int:
        return int(np.asarray(self.interface_coords, dtype=float).shape[0] * 2)

    def validate(self) -> None:
        load_coords = np.asarray(self.load_coords, dtype=float)
        interface_coords = np.asarray(self.interface_coords, dtype=float)
        if load_coords.ndim != 2 or load_coords.shape[1] != 2:
            raise ValueError("NIRB tangent load_coords must have shape (n, 2)")
        if interface_coords.ndim != 2 or interface_coords.shape[1] != 2:
            raise ValueError("NIRB tangent interface_coords must have shape (n, 2)")
        n_load = int(load_coords.shape[0] * 2)
        n_interface = int(interface_coords.shape[0] * 2)
        if self.full_tangent is None and self.interface_tangent is None:
            raise ValueError("NIRB tangent artifact must contain full_tangent or interface_tangent")
        if self.full_tangent is not None:
            full = np.asarray(self.full_tangent, dtype=float)
            if full.ndim != 2 or full.shape[1] != n_load:
                raise ValueError(
                    "NIRB full_tangent must have shape (n_full_displacement_dofs, n_load_dofs); "
                    f"got {full.shape}, expected second dimension {n_load}."
                )
        if self.interface_tangent is not None:
            interface = np.asarray(self.interface_tangent, dtype=float)
            if interface.shape != (n_interface, n_load):
                raise ValueError(
                    "NIRB interface_tangent must have shape "
                    f"{(n_interface, n_load)}, got {interface.shape}."
                )

    def full_increment(self, load_increment: np.ndarray) -> np.ndarray | None:
        if self.full_tangent is None:
            return None
        delta = np.asarray(load_increment, dtype=float).reshape(-1)
        if delta.size != self.n_load_dofs:
            raise ValueError(f"expected {self.n_load_dofs} load increment dofs, got {delta.size}")
        return np.asarray(self.full_tangent, dtype=float) @ delta

    def interface_increment(self, load_increment: np.ndarray) -> np.ndarray | None:
        if self.interface_tangent is None:
            return None
        delta = np.asarray(load_increment, dtype=float).reshape(-1)
        if delta.size != self.n_load_dofs:
            raise ValueError(f"expected {self.n_load_dofs} load increment dofs, got {delta.size}")
        return np.asarray(self.interface_tangent, dtype=float) @ delta


@dataclass
class NIRBSolidPredictor:
    """Adapter implementing the paper's Algorithm 2 solid ROM query."""

    model: Any
    full_shape: tuple[int, ...] | None = None
    interface_matrix: np.ndarray | None = None
    interface_shape: tuple[int, ...] | None = None
    zero_load_tolerance: float = 0.0
    reduced_interface_decoder: ReducedOutputDecoder | None = None
    online_context: dict[str, float] = field(default_factory=dict)
    _interface_decoder: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.interface_matrix is not None:
            restriction = InterfaceRestriction(matrix=np.asarray(self.interface_matrix, dtype=float))
            self._interface_decoder = restriction.restrict_decoder(self.model.output_decoder)
        elif getattr(self.model, "output_restriction", None) is not None:
            self._interface_decoder = self.model.output_restriction.restrict_decoder(self.model.output_decoder)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        full_shape: tuple[int, ...] | None = None,
        interface_matrix: np.ndarray | None = None,
        interface_shape: tuple[int, ...] | None = None,
        zero_load_tolerance: float = 0.0,
        reduced_interface_decoder: ReducedOutputDecoder | None = None,
    ) -> "NIRBSolidPredictor":
        return cls(
            model=load_model(path),
            full_shape=full_shape,
            interface_matrix=interface_matrix,
            interface_shape=interface_shape,
            zero_load_tolerance=float(zero_load_tolerance),
            reduced_interface_decoder=reduced_interface_decoder,
        )

    def set_online_context(
        self,
        *,
        time: float | None = None,
        step: int | None = None,
        coupling_iter: int | None = None,
    ) -> None:
        context: dict[str, float] = {}
        if time is not None:
            context["time"] = float(time)
        if step is not None:
            context["step"] = float(step)
        if coupling_iter is not None:
            context["coupling_iter"] = float(coupling_iter)
            context["subiteration"] = float(coupling_iter)
        self.online_context = context

    def _model_context(self) -> dict[str, float] | None:
        if not tuple(getattr(self.model, "context_feature_names", ()) or ()):
            return None
        return dict(self.online_context)

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
                reduced_displacement=np.zeros((int(self.model.output_decoder.n_linear_modes),), dtype=float),
                elapsed_s=0.0,
            )
        force = force_vector.reshape(-1, 1)
        started = perf_counter()
        reduced = np.asarray(self.model.predict_reduced(force, context=self._model_context()), dtype=float)
        elapsed = perf_counter() - started
        if reduced.ndim != 2 or reduced.shape[1] != 1:
            raise ValueError("NIRB solid model must return a single feature-major reduced displacement column")
        return NIRBSolidPrediction(
            full_displacement=None,
            interface_displacement=None,
            reduced_displacement=reduced[:, 0],
            elapsed_s=float(elapsed),
        )

    def predict_reduced_from_force_coefficients(self, force_coefficients: np.ndarray) -> NIRBSolidPrediction:
        """Evaluate the solid ROM from reduced force coordinates.

        This is the fully reduced counterpart of :meth:`predict_reduced`.
        It assumes the supplied coefficients are already in the trained force
        POD coordinates, so it does not reconstruct or re-project a full
        interface load vector.
        """

        coeffs = np.asarray(force_coefficients, dtype=float).reshape(-1)
        n_force_modes = int(np.asarray(self.model.input_basis.basis).shape[1])
        if int(coeffs.size) != n_force_modes:
            raise ValueError(f"expected {n_force_modes} force coefficients, got {coeffs.size}.")
        if np.linalg.norm(coeffs) <= float(self.zero_load_tolerance):
            reduced_zero = np.zeros((int(self.model.output_decoder.n_linear_modes),), dtype=float)
            return NIRBSolidPrediction(
                full_displacement=None,
                interface_displacement=None,
                reduced_displacement=reduced_zero,
                reduced_interface_displacement=(
                    None
                    if self.reduced_interface_decoder is None
                    else self.reduced_interface_decoder.decode_coefficients(reduced_zero)
                ),
                elapsed_s=0.0,
            )

        started = perf_counter()
        if hasattr(self.model, "predict_reduced_from_input_coefficients"):
            reduced = np.asarray(
                self.model.predict_reduced_from_input_coefficients(
                    coeffs,
                    context=self._model_context(),
                ),
                dtype=float,
            )
        else:
            reduced = np.asarray(self.model.regressor.predict(coeffs.reshape(1, -1)), dtype=float).T
        elapsed = perf_counter() - started
        if reduced.ndim != 2 or reduced.shape[1] != 1:
            raise ValueError("NIRB solid model must return a single feature-major reduced displacement column")
        reduced_displacement = reduced[:, 0]
        reduced_interface = (
            None
            if self.reduced_interface_decoder is None
            else self.reduced_interface_decoder.decode_coefficients(reduced_displacement)
        )
        return NIRBSolidPrediction(
            full_displacement=None,
            interface_displacement=None,
            reduced_displacement=reduced_displacement,
            reduced_interface_displacement=reduced_interface,
            elapsed_s=float(elapsed),
        )

    def predict_reduced_interface_from_force_coefficients(self, force_coefficients: np.ndarray) -> NIRBSolidPrediction:
        if self.reduced_interface_decoder is None:
            raise RuntimeError("reduced_interface_decoder is required for reduced-interface solid prediction")
        prediction = self.predict_reduced_from_force_coefficients(force_coefficients)
        if prediction.reduced_interface_displacement is not None:
            return prediction
        if prediction.reduced_displacement is None:
            raise RuntimeError("NIRB prediction does not contain reduced displacement coordinates.")
        reduced_interface = self.reduced_interface_decoder.decode_coefficients(prediction.reduced_displacement)
        return NIRBSolidPrediction(
            full_displacement=None,
            interface_displacement=None,
            reduced_displacement=prediction.reduced_displacement,
            reduced_interface_displacement=reduced_interface,
            elapsed_s=prediction.elapsed_s,
        )

    def predict_interface_from_force_coefficients(self, force_coefficients: np.ndarray) -> NIRBSolidPrediction:
        """Evaluate an interface-displacement ROM from force POD coordinates."""

        prediction = self.predict_reduced_from_force_coefficients(force_coefficients)
        if prediction.reduced_displacement is None:
            raise RuntimeError("NIRB prediction does not contain reduced displacement coordinates.")
        started = perf_counter()
        interface_displacement = self.reconstruct_interface(prediction.reduced_displacement)
        elapsed = prediction.elapsed_s + (perf_counter() - started)
        return NIRBSolidPrediction(
            full_displacement=None,
            interface_displacement=interface_displacement,
            reduced_displacement=prediction.reduced_displacement,
            reduced_interface_displacement=prediction.reduced_interface_displacement,
            elapsed_s=float(elapsed),
        )

    def predict_from_force_coefficients(self, force_coefficients: np.ndarray) -> NIRBSolidPrediction:
        """Evaluate the full-displacement ROM from force POD coordinates."""

        prediction = self.predict_reduced_from_force_coefficients(force_coefficients)
        if prediction.reduced_displacement is None:
            raise RuntimeError("NIRB prediction does not contain reduced displacement coordinates.")
        started = perf_counter()
        full_displacement = self.reconstruct_full(prediction.reduced_displacement)
        elapsed = prediction.elapsed_s + (perf_counter() - started)
        return NIRBSolidPrediction(
            full_displacement=full_displacement,
            interface_displacement=None,
            reduced_displacement=prediction.reduced_displacement,
            reduced_interface_displacement=prediction.reduced_interface_displacement,
            elapsed_s=float(elapsed),
        )

    def reconstruct_full(self, reduced_displacement: np.ndarray) -> np.ndarray:
        reduced = np.asarray(reduced_displacement, dtype=float).reshape(-1, 1)
        displacement = np.asarray(self.model.output_decoder.decode(reduced), dtype=float)
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
            vector = self.reconstruct_full(reduced[:, 0])
            if self.interface_shape is not None and vector.size != int(np.prod(self.interface_shape)):
                raise RuntimeError(
                    "NIRB interface prediction requires an interface restriction matrix unless the "
                    "decoder output is already interface-sized."
                )
            return vector
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
                reduced_displacement=np.zeros((int(self.model.output_decoder.n_linear_modes),), dtype=float),
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
            reduced_interface_displacement=(
                None
                if self.reduced_interface_decoder is None or prediction.reduced_displacement is None
                else self.reduced_interface_decoder.decode_coefficients(prediction.reduced_displacement)
            ),
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
            reduced_interface_displacement=(
                None
                if self.reduced_interface_decoder is None or prediction.reduced_displacement is None
                else self.reduced_interface_decoder.decode_coefficients(prediction.reduced_displacement)
            ),
            elapsed_s=float(elapsed),
        )


__all__ = [
    "CouplingIterationRecord",
    "CouplingTrace",
    "NIRBInterfaceTangentCorrector",
    "NIRBSolidPrediction",
    "NIRBSolidPredictor",
    "load_cosim_snapshot_batch",
]
