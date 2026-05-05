from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycutfem.mor.pod import fit_pod

from examples.NIRB.fluid_fom_operator import FluidFOMOperator
from examples.NIRB.fluid_lspg import FluidTrialSpace, pack_fluid_state
from examples.NIRB.fluid_snapshots import FluidStageSnapshotBatch


def _unique_intersection(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.intersect1d(np.asarray(left, dtype=int).reshape(-1), np.asarray(right, dtype=int).reshape(-1)).astype(
        int,
        copy=False,
    )


def _field_dofs(operator: FluidFOMOperator, field_names: tuple[str, ...]) -> np.ndarray:
    ids: list[int] = []
    for field_name in field_names:
        ids.extend(np.asarray(operator.dh.get_field_slice(str(field_name)), dtype=int).reshape(-1).tolist())
    return np.asarray(sorted(set(ids)), dtype=int)


def _empty_block(n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    return (
        np.zeros((int(n_rows), 0), dtype=float),
        np.zeros(0, dtype=float),
        np.zeros(0, dtype=float),
        None,
    )


def _fit_block(
    values: np.ndarray,
    *,
    n_modes: int | None,
    energy: float | None,
    center: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("POD block values must be a 2D feature-major matrix.")
    if matrix.shape[0] == 0 or n_modes == 0:
        return _empty_block(matrix.shape[0])
    pod = fit_pod(matrix, n_modes=n_modes, energy=energy, center=bool(center))
    return (
        np.asarray(pod.basis, dtype=float),
        np.asarray(pod.singular_values, dtype=float),
        np.asarray(pod.energy_fraction, dtype=float),
        None if pod.mean is None else np.asarray(pod.mean, dtype=float).reshape(-1),
    )


@dataclass(frozen=True)
class FluidPODTrialBasis:
    """Mixed velocity-pressure POD basis embedded in the global fluid DOF layout."""

    basis: np.ndarray
    free_dofs: np.ndarray
    velocity_dofs: np.ndarray
    pressure_dofs: np.ndarray
    velocity_singular_values: np.ndarray
    pressure_singular_values: np.ndarray
    velocity_energy_fraction: np.ndarray
    pressure_energy_fraction: np.ndarray
    velocity_mean: np.ndarray | None = None
    pressure_mean: np.ndarray | None = None

    def __post_init__(self) -> None:
        basis = np.asarray(self.basis, dtype=float)
        free = np.asarray(self.free_dofs, dtype=int).reshape(-1)
        velocity = np.asarray(self.velocity_dofs, dtype=int).reshape(-1)
        pressure = np.asarray(self.pressure_dofs, dtype=int).reshape(-1)
        if basis.ndim != 2:
            raise ValueError("FluidPODTrialBasis.basis must be a 2D matrix.")
        if np.unique(free).size != free.size:
            raise ValueError("free_dofs must be unique.")
        if np.intersect1d(velocity, pressure).size:
            raise ValueError("velocity and pressure DOFs must be disjoint.")
        if np.setdiff1d(velocity, free).size or np.setdiff1d(pressure, free).size:
            raise ValueError("velocity_dofs and pressure_dofs must be subsets of free_dofs.")
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "free_dofs", free)
        object.__setattr__(self, "velocity_dofs", velocity)
        object.__setattr__(self, "pressure_dofs", pressure)
        object.__setattr__(self, "velocity_singular_values", np.asarray(self.velocity_singular_values, dtype=float))
        object.__setattr__(self, "pressure_singular_values", np.asarray(self.pressure_singular_values, dtype=float))
        object.__setattr__(
            self,
            "velocity_energy_fraction",
            np.asarray(self.velocity_energy_fraction, dtype=float),
        )
        object.__setattr__(
            self,
            "pressure_energy_fraction",
            np.asarray(self.pressure_energy_fraction, dtype=float),
        )
        if self.velocity_mean is not None:
            object.__setattr__(self, "velocity_mean", np.asarray(self.velocity_mean, dtype=float).reshape(-1))
        if self.pressure_mean is not None:
            object.__setattr__(self, "pressure_mean", np.asarray(self.pressure_mean, dtype=float).reshape(-1))

    @property
    def n_modes(self) -> int:
        return int(self.basis.shape[1])

    @property
    def n_velocity_modes(self) -> int:
        return int(self.velocity_singular_values.size)

    @property
    def n_pressure_modes(self) -> int:
        return int(self.pressure_singular_values.size)

    def make_trial_space(
        self,
        operator: FluidFOMOperator,
        *,
        offset: np.ndarray | None = None,
    ) -> FluidTrialSpace:
        base = pack_fluid_state(operator) if offset is None else np.asarray(offset, dtype=float).reshape(-1)
        return FluidTrialSpace(basis=self.basis, offset=base, free_dofs=self.free_dofs)

    def mean_state(self, *, reference: np.ndarray | None = None) -> np.ndarray:
        """Return a global offset using stored centered POD means where present."""

        if reference is None:
            state = np.zeros(int(self.basis.shape[0]), dtype=float)
        else:
            state = np.asarray(reference, dtype=float).reshape(-1).copy()
            if int(state.size) != int(self.basis.shape[0]):
                raise ValueError("reference size must match basis rows.")
        if self.velocity_mean is not None:
            if int(self.velocity_mean.size) != int(self.velocity_dofs.size):
                raise ValueError("Stored velocity mean size does not match velocity_dofs.")
            state[self.velocity_dofs] = self.velocity_mean
        if self.pressure_mean is not None:
            if int(self.pressure_mean.size) != int(self.pressure_dofs.size):
                raise ValueError("Stored pressure mean size does not match pressure_dofs.")
            state[self.pressure_dofs] = self.pressure_mean
        return state

    def project_state(self, values: np.ndarray, *, offset: np.ndarray | None = None) -> np.ndarray:
        state = np.asarray(values, dtype=float).reshape(-1)
        if int(state.size) != int(self.basis.shape[0]):
            raise ValueError(f"State has size {state.size}, expected {int(self.basis.shape[0])}.")
        base = np.zeros_like(state) if offset is None else np.asarray(offset, dtype=float).reshape(-1)
        if int(base.size) != int(state.size):
            raise ValueError("offset size must match state size.")
        reduced_basis = np.asarray(self.basis[self.free_dofs, :], dtype=float)
        rhs = np.asarray(state[self.free_dofs] - base[self.free_dofs], dtype=float)
        coefficients, *_ = np.linalg.lstsq(reduced_basis, rhs, rcond=None)
        return np.asarray(coefficients, dtype=float).reshape(-1)

    def reconstruct_state(self, coefficients: np.ndarray, *, offset: np.ndarray | None = None) -> np.ndarray:
        coeffs = np.asarray(coefficients, dtype=float).reshape(-1)
        if coeffs.size != self.n_modes:
            raise ValueError(f"Expected {self.n_modes} coefficients, got {coeffs.size}.")
        base = np.zeros(int(self.basis.shape[0]), dtype=float) if offset is None else np.asarray(offset, dtype=float).reshape(-1)
        if int(base.size) != int(self.basis.shape[0]):
            raise ValueError("offset size must match basis rows.")
        return base + self.basis @ coeffs


def fit_fluid_pod_trial_basis(
    operator: FluidFOMOperator,
    snapshots: FluidStageSnapshotBatch,
    *,
    velocity_modes: int | None = None,
    pressure_modes: int | None = None,
    velocity_energy: float | None = None,
    pressure_energy: float | None = None,
    center: bool = False,
) -> FluidPODTrialBasis:
    """Fit separate free-DOF velocity and pressure POD blocks for LSPG."""

    if int(snapshots.state.shape[0]) != int(operator.dh.total_dofs):
        raise ValueError("Snapshot state rows must match the operator DOF count.")
    return fit_fluid_pod_trial_basis_from_state_matrix(
        operator,
        snapshots.state,
        free_dofs=snapshots.free_dofs,
        velocity_modes=velocity_modes,
        pressure_modes=pressure_modes,
        velocity_energy=velocity_energy,
        pressure_energy=pressure_energy,
        center=center,
    )


def fit_fluid_pod_trial_basis_from_state_matrix(
    operator: FluidFOMOperator,
    state_matrix: np.ndarray,
    *,
    free_dofs: np.ndarray | None = None,
    velocity_modes: int | None = None,
    pressure_modes: int | None = None,
    velocity_energy: float | None = None,
    pressure_energy: float | None = None,
    center: bool = False,
) -> FluidPODTrialBasis:
    """Fit mixed POD blocks from a full-DOF state or increment matrix."""

    matrix = np.asarray(state_matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("state_matrix must be a feature-major 2D matrix.")
    if int(matrix.shape[0]) != int(operator.dh.total_dofs):
        raise ValueError("state_matrix rows must match the operator DOF count.")
    free_dofs = operator.free_fluid_dofs() if free_dofs is None else np.asarray(free_dofs, dtype=int).reshape(-1)
    velocity_dofs = _unique_intersection(_field_dofs(operator, ("ux", "uy")), free_dofs)
    pressure_dofs = _unique_intersection(_field_dofs(operator, ("p",)), free_dofs)
    velocity_basis, velocity_svals, velocity_energy_fraction, velocity_mean = _fit_block(
        matrix[velocity_dofs, :],
        n_modes=velocity_modes,
        energy=velocity_energy,
        center=bool(center),
    )
    pressure_basis, pressure_svals, pressure_energy_fraction, pressure_mean = _fit_block(
        matrix[pressure_dofs, :],
        n_modes=pressure_modes,
        energy=pressure_energy,
        center=bool(center),
    )
    full_basis = np.zeros(
        (
            int(operator.dh.total_dofs),
            int(velocity_basis.shape[1]) + int(pressure_basis.shape[1]),
        ),
        dtype=float,
    )
    cursor = 0
    if velocity_basis.shape[1]:
        full_basis[velocity_dofs, cursor : cursor + velocity_basis.shape[1]] = velocity_basis
        cursor += int(velocity_basis.shape[1])
    if pressure_basis.shape[1]:
        full_basis[pressure_dofs, cursor : cursor + pressure_basis.shape[1]] = pressure_basis
    return FluidPODTrialBasis(
        basis=full_basis,
        free_dofs=free_dofs,
        velocity_dofs=velocity_dofs,
        pressure_dofs=pressure_dofs,
        velocity_singular_values=velocity_svals,
        pressure_singular_values=pressure_svals,
        velocity_energy_fraction=velocity_energy_fraction,
        pressure_energy_fraction=pressure_energy_fraction,
        velocity_mean=velocity_mean,
        pressure_mean=pressure_mean,
    )


__all__ = [
    "FluidPODTrialBasis",
    "fit_fluid_pod_trial_basis",
    "fit_fluid_pod_trial_basis_from_state_matrix",
]
