from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from pycutfem.mor.pod import PODBasis, project_to_basis, reconstruct_from_basis


TRAJECTORY_REDUCED_SCHEMA_VERSION = 1


def _as_2d(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1-D or 2-D array, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _pod_arrays(prefix: str, basis: PODBasis) -> dict[str, np.ndarray]:
    return {
        f"{prefix}/basis": np.asarray(basis.basis, dtype=float),
        f"{prefix}/singular_values": np.asarray(basis.singular_values, dtype=float),
        f"{prefix}/energy_fraction": np.asarray(basis.energy_fraction, dtype=float),
        f"{prefix}/mean": (
            np.zeros((basis.basis.shape[0], 1), dtype=float)
            if basis.mean is None
            else np.asarray(basis.mean, dtype=float)
        ),
        f"{prefix}/centered": np.asarray(basis.mean is not None, dtype=bool),
    }


def _load_pod(data: dict[str, np.ndarray], prefix: str) -> PODBasis:
    centered = bool(np.asarray(data[f"{prefix}/centered"], dtype=bool).reshape(-1)[0])
    mean = np.asarray(data[f"{prefix}/mean"], dtype=float) if centered else None
    return PODBasis(
        basis=np.asarray(data[f"{prefix}/basis"], dtype=float),
        singular_values=np.asarray(data[f"{prefix}/singular_values"], dtype=float),
        energy_fraction=np.asarray(data[f"{prefix}/energy_fraction"], dtype=float),
        mean=mean,
    )


def _read_snapshot_metadata(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"snapshot metadata is empty: {path}")
    steps = np.asarray([int(row["step"]) for row in rows], dtype=int)
    times = np.asarray([float(row["time_s"]) for row in rows], dtype=float)
    coupling_iters = np.asarray([int(row["coupling_iter"]) for row in rows], dtype=int)
    converged = np.asarray([str(row["converged"]).strip().lower() == "true" for row in rows], dtype=bool)
    return steps, times, coupling_iters, converged


def _relative_change(values: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    val = np.asarray(values, dtype=float).reshape(-1)
    ref = np.asarray(reference, dtype=float).reshape(-1)
    if val.shape != ref.shape:
        raise ValueError("relative-change vectors must have matching shapes.")
    abs_change = float(np.linalg.norm(val - ref))
    denom = max(float(np.linalg.norm(ref)), 1.0e-30)
    return abs_change, abs_change / denom


def _pack_vector(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _load_step_history_matrix(
    step_history_dir: Path,
    *,
    key: str,
    max_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(step_history_dir.glob("step*.npz"))
    if max_steps is not None:
        files = files[: int(max_steps)]
    if not files:
        raise FileNotFoundError(f"no step history files found in {step_history_dir}")
    columns: list[np.ndarray] = []
    steps: list[int] = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                raise KeyError(f"{path} does not contain {key!r}")
            columns.append(_pack_vector(data[key]))
            steps.append(int(np.asarray(data["step"], dtype=int).reshape(-1)[0]))
    return np.column_stack(columns), np.asarray(steps, dtype=int)


def _select_rank(singular_values: np.ndarray, *, n_modes: int | None, energy: float | None) -> int:
    if n_modes is not None and energy is not None:
        raise ValueError("specify either n_modes or energy, not both")
    if n_modes is not None:
        if int(n_modes) < 1:
            raise ValueError("n_modes must be positive.")
        return min(int(n_modes), int(singular_values.size))
    if energy is None:
        return int(singular_values.size)
    if not 0.0 < float(energy) <= 1.0:
        raise ValueError("energy must lie in (0, 1].")
    squared = np.asarray(singular_values, dtype=float) ** 2
    total = float(np.sum(squared))
    if total <= 0.0:
        return 1
    cumulative = np.cumsum(squared) / total
    return int(np.searchsorted(cumulative, float(energy), side="left") + 1)


def _fit_basis(values: np.ndarray, *, n_modes: int | None, energy: float | None, center: bool) -> PODBasis:
    matrix = _as_2d(values, name="snapshot matrix")
    if n_modes is None and energy is None:
        n_modes = min(matrix.shape)
    mean = np.mean(matrix, axis=1, keepdims=True) if bool(center) else None
    work = matrix - mean if mean is not None else matrix.copy()
    if n_modes is not None and int(n_modes) >= int(work.shape[1]) and energy is None:
        rank = int(work.shape[1])
        basis, _ = np.linalg.qr(work, mode="reduced")
        gram = np.asarray(work.T @ work, dtype=float)
        eigvals = np.linalg.eigvalsh(gram)
        singular_values_full = np.sqrt(np.maximum(np.sort(eigvals)[::-1], 0.0))
        if int(singular_values_full.size) < rank:
            singular_values_full = np.pad(singular_values_full, (0, rank - int(singular_values_full.size)))
        squared = singular_values_full**2
        total = float(np.sum(squared))
        energy_fraction = np.cumsum(squared) / total if total > 0.0 else np.zeros(rank, dtype=float)
        return PODBasis(
            basis=np.asarray(basis[:, :rank], dtype=float),
            singular_values=np.asarray(singular_values_full[:rank], dtype=float),
            energy_fraction=np.asarray(energy_fraction[:rank], dtype=float),
            mean=mean,
        )
    # Method of snapshots: these trajectory matrices are tall and have at most
    # one column per accepted step, so the Gram eigenproblem is much cheaper
    # than a full dense SVD while producing the same left POD space.
    gram = np.asarray(work.T @ work, dtype=float)
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    keep = eigvals > max(float(eigvals[0]) if eigvals.size else 0.0, 1.0) * 1.0e-28
    if not np.any(keep):
        basis = np.zeros((matrix.shape[0], 1), dtype=float)
        singular_values = np.zeros(1, dtype=float)
        energy_fraction = np.zeros(1, dtype=float)
        return PODBasis(basis=basis, singular_values=singular_values, energy_fraction=energy_fraction, mean=mean)
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]
    singular_values_full = np.sqrt(eigvals)
    rank = _select_rank(singular_values_full, n_modes=n_modes, energy=energy)
    singular_values = singular_values_full[:rank]
    basis = work @ (eigvecs[:, :rank] / singular_values.reshape(1, -1))
    basis, _ = np.linalg.qr(basis, mode="reduced")
    squared = singular_values_full**2
    total = float(np.sum(squared))
    energy_fraction_full = np.cumsum(squared) / total if total > 0.0 else np.zeros_like(squared)
    return PODBasis(
        basis=np.asarray(basis[:, :rank], dtype=float),
        singular_values=np.asarray(singular_values, dtype=float),
        energy_fraction=np.asarray(energy_fraction_full[:rank], dtype=float),
        mean=mean,
    )


@dataclass(frozen=True)
class TrajectoryReducedArtifact:
    """Reduced-coordinate replay artifact for a known FSI trajectory.

    The online loop stores and exchanges only coefficient vectors.  Full fields
    are reconstructed only when `reconstruct_step` or validation routines are
    called.
    """

    load_basis: PODBasis
    displacement_basis: PODBasis
    velocity_basis: PODBasis
    pressure_basis: PODBasis
    mesh_displacement_basis: PODBasis
    mesh_velocity_basis: PODBasis
    structure_displacement_basis: PODBasis
    coupling_steps: np.ndarray
    coupling_times: np.ndarray
    coupling_iters: np.ndarray
    coupling_converged: np.ndarray
    load_guess_coefficients: np.ndarray
    load_return_coefficients: np.ndarray
    displacement_coefficients: np.ndarray
    accepted_steps: np.ndarray
    accepted_times: np.ndarray
    accepted_velocity_coefficients: np.ndarray
    accepted_pressure_coefficients: np.ndarray
    accepted_mesh_displacement_coefficients: np.ndarray
    accepted_mesh_velocity_coefficients: np.ndarray
    accepted_structure_displacement_coefficients: np.ndarray
    fluid_node_ids: np.ndarray
    fluid_coords_ref: np.ndarray
    structure_node_ids: np.ndarray
    structure_coords_ref: np.ndarray
    interface_coords_ref: np.ndarray
    metadata: dict[str, object]

    def arrays(self) -> dict[str, np.ndarray]:
        payload: dict[str, np.ndarray] = {
            "schema_version": np.asarray(TRAJECTORY_REDUCED_SCHEMA_VERSION, dtype=int),
            "coupling/steps": np.asarray(self.coupling_steps, dtype=int),
            "coupling/times": np.asarray(self.coupling_times, dtype=float),
            "coupling/iters": np.asarray(self.coupling_iters, dtype=int),
            "coupling/converged": np.asarray(self.coupling_converged, dtype=bool),
            "coupling/load_guess_coefficients": np.asarray(self.load_guess_coefficients, dtype=float),
            "coupling/load_return_coefficients": np.asarray(self.load_return_coefficients, dtype=float),
            "coupling/displacement_coefficients": np.asarray(self.displacement_coefficients, dtype=float),
            "accepted/steps": np.asarray(self.accepted_steps, dtype=int),
            "accepted/times": np.asarray(self.accepted_times, dtype=float),
            "accepted/velocity_coefficients": np.asarray(self.accepted_velocity_coefficients, dtype=float),
            "accepted/pressure_coefficients": np.asarray(self.accepted_pressure_coefficients, dtype=float),
            "accepted/mesh_displacement_coefficients": np.asarray(
                self.accepted_mesh_displacement_coefficients,
                dtype=float,
            ),
            "accepted/mesh_velocity_coefficients": np.asarray(self.accepted_mesh_velocity_coefficients, dtype=float),
            "accepted/structure_displacement_coefficients": np.asarray(
                self.accepted_structure_displacement_coefficients,
                dtype=float,
            ),
            "geometry/fluid_node_ids": np.asarray(self.fluid_node_ids, dtype=int),
            "geometry/fluid_coords_ref": np.asarray(self.fluid_coords_ref, dtype=float),
            "geometry/structure_node_ids": np.asarray(self.structure_node_ids, dtype=int),
            "geometry/structure_coords_ref": np.asarray(self.structure_coords_ref, dtype=float),
            "geometry/interface_coords_ref": np.asarray(self.interface_coords_ref, dtype=float),
            "metadata/json": np.asarray(json.dumps(self.metadata, sort_keys=True), dtype=np.str_),
        }
        payload.update(_pod_arrays("basis/load", self.load_basis))
        payload.update(_pod_arrays("basis/displacement", self.displacement_basis))
        payload.update(_pod_arrays("basis/velocity", self.velocity_basis))
        payload.update(_pod_arrays("basis/pressure", self.pressure_basis))
        payload.update(_pod_arrays("basis/mesh_displacement", self.mesh_displacement_basis))
        payload.update(_pod_arrays("basis/mesh_velocity", self.mesh_velocity_basis))
        payload.update(_pod_arrays("basis/structure_displacement", self.structure_displacement_basis))
        return payload

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **self.arrays())
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryReducedArtifact":
        source = Path(path)
        with np.load(source, allow_pickle=False) as raw:
            data = {key: np.asarray(raw[key]) for key in raw.files}
        version = int(np.asarray(data.get("schema_version", np.asarray(-1)), dtype=int).reshape(-1)[0])
        if version != TRAJECTORY_REDUCED_SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported trajectory-reduced schema_version={version}; "
                f"expected {TRAJECTORY_REDUCED_SCHEMA_VERSION}."
            )
        metadata = json.loads(str(np.asarray(data["metadata/json"]).reshape(-1)[0]))
        return cls(
            load_basis=_load_pod(data, "basis/load"),
            displacement_basis=_load_pod(data, "basis/displacement"),
            velocity_basis=_load_pod(data, "basis/velocity"),
            pressure_basis=_load_pod(data, "basis/pressure"),
            mesh_displacement_basis=_load_pod(data, "basis/mesh_displacement"),
            mesh_velocity_basis=_load_pod(data, "basis/mesh_velocity"),
            structure_displacement_basis=_load_pod(data, "basis/structure_displacement"),
            coupling_steps=np.asarray(data["coupling/steps"], dtype=int),
            coupling_times=np.asarray(data["coupling/times"], dtype=float),
            coupling_iters=np.asarray(data["coupling/iters"], dtype=int),
            coupling_converged=np.asarray(data["coupling/converged"], dtype=bool),
            load_guess_coefficients=np.asarray(data["coupling/load_guess_coefficients"], dtype=float),
            load_return_coefficients=np.asarray(data["coupling/load_return_coefficients"], dtype=float),
            displacement_coefficients=np.asarray(data["coupling/displacement_coefficients"], dtype=float),
            accepted_steps=np.asarray(data["accepted/steps"], dtype=int),
            accepted_times=np.asarray(data["accepted/times"], dtype=float),
            accepted_velocity_coefficients=np.asarray(data["accepted/velocity_coefficients"], dtype=float),
            accepted_pressure_coefficients=np.asarray(data["accepted/pressure_coefficients"], dtype=float),
            accepted_mesh_displacement_coefficients=np.asarray(
                data["accepted/mesh_displacement_coefficients"],
                dtype=float,
            ),
            accepted_mesh_velocity_coefficients=np.asarray(data["accepted/mesh_velocity_coefficients"], dtype=float),
            accepted_structure_displacement_coefficients=np.asarray(
                data["accepted/structure_displacement_coefficients"],
                dtype=float,
            ),
            fluid_node_ids=np.asarray(data["geometry/fluid_node_ids"], dtype=int),
            fluid_coords_ref=np.asarray(data["geometry/fluid_coords_ref"], dtype=float),
            structure_node_ids=np.asarray(data["geometry/structure_node_ids"], dtype=int),
            structure_coords_ref=np.asarray(data["geometry/structure_coords_ref"], dtype=float),
            interface_coords_ref=np.asarray(data["geometry/interface_coords_ref"], dtype=float),
            metadata=metadata,
        )

    def reconstruct_step(self, step: int) -> dict[str, np.ndarray]:
        matches = np.flatnonzero(np.asarray(self.accepted_steps, dtype=int) == int(step))
        if matches.size != 1:
            raise KeyError(f"accepted step {step} not present in reduced artifact.")
        idx = int(matches[0])
        velocity = reconstruct_from_basis(
            self.accepted_velocity_coefficients[:, idx],
            self.velocity_basis.basis,
            self.velocity_basis.mean,
        ).reshape(-1)
        pressure = reconstruct_from_basis(
            self.accepted_pressure_coefficients[:, idx],
            self.pressure_basis.basis,
            self.pressure_basis.mean,
        ).reshape(-1)
        mesh_displacement = reconstruct_from_basis(
            self.accepted_mesh_displacement_coefficients[:, idx],
            self.mesh_displacement_basis.basis,
            self.mesh_displacement_basis.mean,
        ).reshape(-1)
        mesh_velocity = reconstruct_from_basis(
            self.accepted_mesh_velocity_coefficients[:, idx],
            self.mesh_velocity_basis.basis,
            self.mesh_velocity_basis.mean,
        ).reshape(-1)
        structure_displacement = reconstruct_from_basis(
            self.accepted_structure_displacement_coefficients[:, idx],
            self.structure_displacement_basis.basis,
            self.structure_displacement_basis.mean,
        ).reshape(-1)
        return {
            "step": np.asarray(int(step), dtype=int),
            "time_s": np.asarray(float(self.accepted_times[idx]), dtype=float),
            "fluid_node_ids": np.asarray(self.fluid_node_ids, dtype=int),
            "fluid_coords_ref": np.asarray(self.fluid_coords_ref, dtype=float),
            "fluid_velocity_nodal_values": velocity.reshape(-1, 2),
            "fluid_pressure_nodal_values": pressure.reshape(-1, 1),
            "fluid_mesh_displacement_nodal_values": mesh_displacement.reshape(-1, 2),
            "fluid_mesh_velocity_nodal_values": mesh_velocity.reshape(-1, 2),
            "structure_node_ids": np.asarray(self.structure_node_ids, dtype=int),
            "structure_coords_ref": np.asarray(self.structure_coords_ref, dtype=float),
            "structure_displacement_nodal_values": structure_displacement.reshape(-1, 2),
        }


@dataclass(frozen=True)
class TrajectoryReducedRunResult:
    steps_requested: int
    steps_converged: int
    coupling_iters_per_step: tuple[int, ...]
    online_time_s: float
    timeseries: tuple[dict[str, object], ...]

    def summary(self) -> dict[str, object]:
        return {
            "steps_requested": int(self.steps_requested),
            "steps_converged": int(self.steps_converged),
            "coupling_iters_per_step": [int(v) for v in self.coupling_iters_per_step],
            "mean_coupling_iters": float(np.mean(self.coupling_iters_per_step))
            if self.coupling_iters_per_step
            else 0.0,
            "online_time_s": float(self.online_time_s),
            "snapshot_count": int(len(self.timeseries)),
        }


class TrajectoryReducedOnlineSolver:
    """Run the stored FSI trajectory using only reduced coupling coefficients."""

    def __init__(
        self,
        artifact: TrajectoryReducedArtifact,
        *,
        coupling_abs_tol: float = 5.0e-3,
        coupling_rel_tol: float = 5.0e-3,
    ) -> None:
        self.artifact = artifact
        self.coupling_abs_tol = float(coupling_abs_tol)
        self.coupling_rel_tol = float(coupling_rel_tol)

    def run(self, *, max_steps: int | None = None) -> TrajectoryReducedRunResult:
        artifact = self.artifact
        target_steps = np.asarray(artifact.accepted_steps, dtype=int)
        if max_steps is not None:
            target_steps = target_steps[target_steps <= int(max_steps)]
        rows: list[dict[str, object]] = []
        coupling_counts: list[int] = []
        t0 = perf_counter()

        previous_disp = np.zeros(int(artifact.displacement_coefficients.shape[0]), dtype=float)
        for step in target_steps:
            mask = np.asarray(artifact.coupling_steps, dtype=int) == int(step)
            indices = np.flatnonzero(mask)
            if indices.size == 0:
                raise RuntimeError(f"no reduced coupling rows found for step {int(step)}.")
            converged_this_step = False
            for local_count, idx in enumerate(indices, start=1):
                disp = artifact.displacement_coefficients[:, idx]
                load_guess = artifact.load_guess_coefficients[:, idx]
                load_return = artifact.load_return_coefficients[:, idx]
                disp_abs, disp_rel = _relative_change(disp, previous_disp)
                load_abs, load_rel = _relative_change(load_return, load_guess)
                reduced_converged = bool(
                    (disp_abs <= self.coupling_abs_tol or disp_rel <= self.coupling_rel_tol)
                    and (load_abs <= self.coupling_abs_tol or load_rel <= self.coupling_rel_tol)
                )
                artifact_converged = bool(artifact.coupling_converged[idx])
                rows.append(
                    {
                        "step": int(step),
                        "time_s": float(artifact.coupling_times[idx]),
                        "coupling_iter": int(artifact.coupling_iters[idx]),
                        "disp_abs_reduced": float(disp_abs),
                        "disp_rel_reduced": float(disp_rel),
                        "load_abs_reduced": float(load_abs),
                        "load_rel_reduced": float(load_rel),
                        "reduced_converged": bool(reduced_converged),
                        "artifact_converged": bool(artifact_converged),
                    }
                )
                previous_disp = disp.copy()
                if artifact_converged:
                    coupling_counts.append(int(local_count))
                    converged_this_step = True
                    break
            if not converged_this_step:
                break

        return TrajectoryReducedRunResult(
            steps_requested=int(target_steps[-1]) if target_steps.size else 0,
            steps_converged=int(len(coupling_counts)),
            coupling_iters_per_step=tuple(coupling_counts),
            online_time_s=float(perf_counter() - t0),
            timeseries=tuple(rows),
        )


def build_trajectory_reduced_artifact(
    trajectory_dir: str | Path,
    *,
    max_steps: int | None = None,
    interface_modes: int | None = None,
    state_modes: int | None = None,
    interface_energy: float | None = None,
    state_energy: float | None = None,
    center: bool = True,
) -> TrajectoryReducedArtifact:
    root = Path(trajectory_dir)
    co_sim = root / "coSimData"
    step_history = root / "step_history"
    metadata_path = root / "snapshot_metadata.csv"
    if not co_sim.exists() or not step_history.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"{root} does not look like an Example 2 trajectory output directory.")

    coupling_steps, coupling_times, coupling_iters, coupling_converged = _read_snapshot_metadata(metadata_path)
    if max_steps is not None:
        keep = coupling_steps <= int(max_steps)
        coupling_steps = coupling_steps[keep]
        coupling_times = coupling_times[keep]
        coupling_iters = coupling_iters[keep]
        coupling_converged = coupling_converged[keep]
    n_coupling = int(coupling_steps.size)

    load_guess = np.load(co_sim / "load_guess_data.npy", allow_pickle=False)
    load_return = np.load(co_sim / "load_return_data.npy", allow_pickle=False)
    displacement = np.load(co_sim / "interface_disp_data.npy", allow_pickle=False)
    if max_steps is not None:
        load_guess = load_guess[:, :n_coupling]
        load_return = load_return[:, :n_coupling]
        displacement = displacement[:, :n_coupling]
    if load_guess.shape[1] != n_coupling or load_return.shape[1] != n_coupling or displacement.shape[1] != n_coupling:
        raise ValueError("coupling snapshot matrix columns do not match snapshot_metadata rows.")

    load_basis = _fit_basis(
        np.column_stack([load_guess, load_return]),
        n_modes=interface_modes,
        energy=interface_energy,
        center=center,
    )
    displacement_basis = _fit_basis(
        displacement,
        n_modes=interface_modes,
        energy=interface_energy,
        center=center,
    )
    load_guess_coeffs = project_to_basis(load_guess, load_basis.basis, load_basis.mean)
    load_return_coeffs = project_to_basis(load_return, load_basis.basis, load_basis.mean)
    displacement_coeffs = project_to_basis(displacement, displacement_basis.basis, displacement_basis.mean)

    velocity, accepted_steps = _load_step_history_matrix(
        step_history,
        key="fluid_velocity_nodal_values",
        max_steps=max_steps,
    )
    pressure, accepted_steps_pressure = _load_step_history_matrix(
        step_history,
        key="fluid_pressure_nodal_values",
        max_steps=max_steps,
    )
    mesh_displacement, accepted_steps_mesh = _load_step_history_matrix(
        step_history,
        key="fluid_mesh_displacement_nodal_values",
        max_steps=max_steps,
    )
    mesh_velocity, accepted_steps_mesh_velocity = _load_step_history_matrix(
        step_history,
        key="fluid_mesh_velocity_nodal_values",
        max_steps=max_steps,
    )
    structure_displacement, accepted_steps_structure = _load_step_history_matrix(
        step_history,
        key="structure_displacement_nodal_values",
        max_steps=max_steps,
    )
    if not (
        np.array_equal(accepted_steps, accepted_steps_pressure)
        and np.array_equal(accepted_steps, accepted_steps_mesh)
        and np.array_equal(accepted_steps, accepted_steps_mesh_velocity)
        and np.array_equal(accepted_steps, accepted_steps_structure)
    ):
        raise ValueError("accepted step histories are not aligned.")

    velocity_basis = _fit_basis(velocity, n_modes=state_modes, energy=state_energy, center=center)
    pressure_basis = _fit_basis(pressure, n_modes=state_modes, energy=state_energy, center=center)
    mesh_displacement_basis = _fit_basis(mesh_displacement, n_modes=state_modes, energy=state_energy, center=center)
    mesh_velocity_basis = _fit_basis(mesh_velocity, n_modes=state_modes, energy=state_energy, center=center)
    structure_displacement_basis = _fit_basis(
        structure_displacement,
        n_modes=state_modes,
        energy=state_energy,
        center=center,
    )

    velocity_coeffs = project_to_basis(velocity, velocity_basis.basis, velocity_basis.mean)
    pressure_coeffs = project_to_basis(pressure, pressure_basis.basis, pressure_basis.mean)
    mesh_displacement_coeffs = project_to_basis(
        mesh_displacement,
        mesh_displacement_basis.basis,
        mesh_displacement_basis.mean,
    )
    mesh_velocity_coeffs = project_to_basis(mesh_velocity, mesh_velocity_basis.basis, mesh_velocity_basis.mean)
    structure_displacement_coeffs = project_to_basis(
        structure_displacement,
        structure_displacement_basis.basis,
        structure_displacement_basis.mean,
    )

    first_step = step_history / f"step{int(accepted_steps[0]):04d}.npz"
    with np.load(first_step, allow_pickle=False) as data:
        fluid_node_ids = np.asarray(data["fluid_node_ids"], dtype=int)
        fluid_coords_ref = np.asarray(data["fluid_coords_ref"], dtype=float)
        structure_node_ids = np.asarray(data["structure_node_ids"], dtype=int)
        structure_coords_ref = np.asarray(data["structure_coords_ref"], dtype=float)
        interface_coords_ref = np.asarray(data["interface_disp_coords_ref"], dtype=float)
    accepted_times = np.asarray(accepted_steps, dtype=float) * float(
        json.loads((root / "summary.json").read_text()).get("dt", 0.008)
        if (root / "summary.json").exists()
        else 0.008
    )
    if np.any(coupling_converged):
        accepted_times = np.asarray(
            [float(coupling_times[np.flatnonzero((coupling_steps == step) & coupling_converged)[-1]]) for step in accepted_steps],
            dtype=float,
        )

    metadata = {
        "source_trajectory_dir": str(root),
        "max_steps": None if max_steps is None else int(max_steps),
        "interface_modes": None if interface_modes is None else int(interface_modes),
        "state_modes": None if state_modes is None else int(state_modes),
        "interface_energy": None if interface_energy is None else float(interface_energy),
        "state_energy": None if state_energy is None else float(state_energy),
        "center": bool(center),
        "coupling_snapshot_count": int(n_coupling),
        "accepted_step_count": int(accepted_steps.size),
    }

    return TrajectoryReducedArtifact(
        load_basis=load_basis,
        displacement_basis=displacement_basis,
        velocity_basis=velocity_basis,
        pressure_basis=pressure_basis,
        mesh_displacement_basis=mesh_displacement_basis,
        mesh_velocity_basis=mesh_velocity_basis,
        structure_displacement_basis=structure_displacement_basis,
        coupling_steps=coupling_steps,
        coupling_times=coupling_times,
        coupling_iters=coupling_iters,
        coupling_converged=coupling_converged,
        load_guess_coefficients=load_guess_coeffs,
        load_return_coefficients=load_return_coeffs,
        displacement_coefficients=displacement_coeffs,
        accepted_steps=accepted_steps,
        accepted_times=accepted_times,
        accepted_velocity_coefficients=velocity_coeffs,
        accepted_pressure_coefficients=pressure_coeffs,
        accepted_mesh_displacement_coefficients=mesh_displacement_coeffs,
        accepted_mesh_velocity_coefficients=mesh_velocity_coeffs,
        accepted_structure_displacement_coefficients=structure_displacement_coeffs,
        fluid_node_ids=fluid_node_ids,
        fluid_coords_ref=fluid_coords_ref,
        structure_node_ids=structure_node_ids,
        structure_coords_ref=structure_coords_ref,
        interface_coords_ref=interface_coords_ref,
        metadata=metadata,
    )


def validate_reconstruction(
    artifact: TrajectoryReducedArtifact,
    trajectory_dir: str | Path,
    *,
    steps: Iterable[int] | None = None,
) -> dict[str, float | int]:
    root = Path(trajectory_dir)
    step_history = root / "step_history"
    if not step_history.exists():
        raise FileNotFoundError(f"step_history directory not found: {step_history}")
    step_values = list(int(v) for v in (artifact.accepted_steps if steps is None else steps))
    maxima: dict[str, float] = {
        "fluid_velocity_max_rel_error": 0.0,
        "fluid_pressure_max_rel_error": 0.0,
        "fluid_mesh_displacement_max_rel_error": 0.0,
        "fluid_mesh_velocity_max_rel_error": 0.0,
        "structure_displacement_max_rel_error": 0.0,
    }
    for step in step_values:
        reconstructed = artifact.reconstruct_step(step)
        source = step_history / f"step{int(step):04d}.npz"
        with np.load(source, allow_pickle=False) as data:
            pairs = [
                (
                    "fluid_velocity_max_rel_error",
                    reconstructed["fluid_velocity_nodal_values"],
                    data["fluid_velocity_nodal_values"],
                ),
                (
                    "fluid_pressure_max_rel_error",
                    reconstructed["fluid_pressure_nodal_values"],
                    data["fluid_pressure_nodal_values"],
                ),
                (
                    "fluid_mesh_displacement_max_rel_error",
                    reconstructed["fluid_mesh_displacement_nodal_values"],
                    data["fluid_mesh_displacement_nodal_values"],
                ),
                (
                    "fluid_mesh_velocity_max_rel_error",
                    reconstructed["fluid_mesh_velocity_nodal_values"],
                    data["fluid_mesh_velocity_nodal_values"],
                ),
                (
                    "structure_displacement_max_rel_error",
                    reconstructed["structure_displacement_nodal_values"],
                    data["structure_displacement_nodal_values"],
                ),
            ]
        for name, values, reference in pairs:
            _abs, rel = _relative_change(np.asarray(values, dtype=float), np.asarray(reference, dtype=float))
            maxima[name] = max(float(maxima[name]), float(rel))
    return {
        "validated_steps": int(len(step_values)),
        **maxima,
        "max_relative_error": float(max(maxima.values(), default=0.0)),
    }


def write_timeseries(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "time_s",
        "coupling_iter",
        "disp_abs_reduced",
        "disp_rel_reduced",
        "load_abs_reduced",
        "load_rel_reduced",
        "reduced_converged",
        "artifact_converged",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


__all__ = [
    "TRAJECTORY_REDUCED_SCHEMA_VERSION",
    "TrajectoryReducedArtifact",
    "TrajectoryReducedOnlineSolver",
    "TrajectoryReducedRunResult",
    "build_trajectory_reduced_artifact",
    "validate_reconstruction",
    "write_timeseries",
]
