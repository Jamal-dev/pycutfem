from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from examples.NIRB.fluid_basis import FluidPODTrialBasis, fit_fluid_pod_trial_basis
from examples.NIRB.fluid_fom_operator import FluidFOMOperator
from examples.NIRB.fluid_lspg import write_fluid_state
from examples.NIRB.fluid_snapshots import FluidStageSnapshotBatch, restore_fluid_stage


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float).reshape(-1)
    cand = np.asarray(candidate, dtype=float).reshape(-1)
    return float(np.linalg.norm(cand - ref) / max(float(np.linalg.norm(ref)), 1.0e-15))


def _field_rows(operator: FluidFOMOperator, field_names: tuple[str, ...], free_dofs: np.ndarray) -> np.ndarray:
    rows: list[np.ndarray] = []
    for field_name in field_names:
        rows.append(np.asarray(operator.dh.get_field_slice(str(field_name)), dtype=int).reshape(-1))
    return np.intersect1d(np.concatenate(rows), np.asarray(free_dofs, dtype=int).reshape(-1)).astype(int, copy=False)


def _split_indices(
    n_snapshots: int,
    *,
    test_fraction: float,
    random_state: int,
    test_indices: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if n_snapshots < 2:
        raise ValueError("At least two snapshots are required for held-out mode validation.")
    all_indices = np.arange(int(n_snapshots), dtype=int)
    if test_indices is None:
        if not 0.0 < float(test_fraction) < 1.0:
            raise ValueError("test_fraction must lie in (0, 1).")
        permutation = np.random.default_rng(int(random_state)).permutation(all_indices)
        n_test = max(1, int(round(float(test_fraction) * int(n_snapshots))))
        test = np.sort(np.asarray(permutation[:n_test], dtype=int))
    else:
        test = np.unique(np.asarray(test_indices, dtype=int).reshape(-1))
        if test.size == 0:
            raise ValueError("test_indices must not be empty.")
        if np.any(test < 0) or np.any(test >= int(n_snapshots)):
            raise IndexError("test_indices contains an out-of-range entry.")
    train = np.setdiff1d(all_indices, test, assume_unique=False).astype(int, copy=False)
    if train.size == 0:
        raise ValueError("Held-out split leaves no training snapshots.")
    return train, test


def _basis_offset(basis: FluidPODTrialBasis, train_snapshots: FluidStageSnapshotBatch) -> np.ndarray:
    reference = np.mean(np.asarray(train_snapshots.state, dtype=float), axis=1)
    return basis.mean_state(reference=reference)


def _reconstruct_preserving_fixed(
    *,
    basis: FluidPODTrialBasis,
    state: np.ndarray,
    fixed_dofs: np.ndarray,
    fixed_values: np.ndarray,
    offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = basis.project_state(state, offset=offset)
    reconstructed = basis.reconstruct_state(coefficients, offset=offset)
    fixed = np.asarray(fixed_dofs, dtype=int).reshape(-1)
    if fixed.size:
        reconstructed[fixed] = np.asarray(fixed_values, dtype=float).reshape(-1)
    return coefficients, reconstructed


@dataclass(frozen=True)
class FluidModeValidationEntry:
    velocity_modes: int
    pressure_modes: int
    score: float
    state_error: float
    velocity_error: float
    pressure_error: float
    reaction_error: float | None

    @property
    def total_modes(self) -> int:
        return int(self.velocity_modes) + int(self.pressure_modes)


@dataclass(frozen=True)
class FluidModeValidationResult:
    entries: list[FluidModeValidationEntry]
    train_indices: np.ndarray
    test_indices: np.ndarray
    centered: bool
    state_weight: float
    reaction_weight: float

    def best(self, *, plateau_rel_tol: float = 0.05) -> FluidModeValidationEntry:
        if not self.entries:
            raise RuntimeError("no fluid mode validation entries available")
        min_score = min(float(entry.score) for entry in self.entries)
        limit = min_score * (1.0 + max(0.0, float(plateau_rel_tol)))
        candidates = [entry for entry in self.entries if float(entry.score) <= limit]
        return min(
            candidates,
            key=lambda entry: (
                int(entry.total_modes),
                int(entry.velocity_modes),
                int(entry.pressure_modes),
                float(entry.score),
            ),
        )


def run_fluid_mode_cross_validation(
    operator: FluidFOMOperator,
    snapshots: FluidStageSnapshotBatch,
    *,
    velocity_modes: list[int] | range,
    pressure_modes: list[int] | range,
    test_fraction: float = 0.2,
    random_state: int = 0,
    test_indices: np.ndarray | list[int] | tuple[int, ...] | None = None,
    center: bool = True,
    include_reaction_error: bool = False,
    reaction_refresh_state: bool = True,
    state_weight: float | None = None,
    reaction_weight: float = 1.0,
) -> FluidModeValidationResult:
    """Run Tiba-style held-out validation for mixed fluid POD mode counts.

    The final coupled quantity for this fluid ROM is the interface reaction, so
    when reaction validation is enabled the score is reaction-dominated and the
    state reconstruction error is only a tie breaker.  This deliberately avoids
    accepting a mode count from singular-value energy alone.
    """

    if int(snapshots.state.shape[0]) != int(operator.dh.total_dofs):
        raise ValueError("Snapshot state rows must match the operator DOF count.")
    v_candidates = [int(value) for value in velocity_modes]
    p_candidates = [int(value) for value in pressure_modes]
    if not v_candidates or not p_candidates:
        raise ValueError("velocity_modes and pressure_modes must not be empty.")
    if any(value < 0 for value in v_candidates + p_candidates):
        raise ValueError("Mode counts must be nonnegative.")
    train_indices, heldout_indices = _split_indices(
        snapshots.n_snapshots,
        test_fraction=float(test_fraction),
        random_state=int(random_state),
        test_indices=None if test_indices is None else np.asarray(test_indices, dtype=int),
    )
    train_snapshots = snapshots.subset(train_indices)
    free = np.asarray(snapshots.free_dofs, dtype=int).reshape(-1)
    velocity_rows = _field_rows(operator, ("ux", "uy"), free)
    pressure_rows = _field_rows(operator, ("p",), free)
    if state_weight is None:
        state_weight_value = 0.1 if bool(include_reaction_error) else 1.0
    else:
        state_weight_value = float(state_weight)

    reference_reactions: dict[int, np.ndarray] = {}
    if bool(include_reaction_error):
        for idx in heldout_indices:
            record = snapshots.record(int(idx))
            if record.reaction_values is not None:
                reference_reactions[int(idx)] = np.asarray(record.reaction_values, dtype=float).reshape(-1)
            else:
                restore_fluid_stage(operator, record)
                reference_reactions[int(idx)] = np.asarray(
                    operator.reaction_loads(refresh_state=False).values,
                    dtype=float,
                ).reshape(-1)

    entries: list[FluidModeValidationEntry] = []
    for n_velocity in v_candidates:
        for n_pressure in p_candidates:
            basis = fit_fluid_pod_trial_basis(
                operator,
                train_snapshots,
                velocity_modes=int(n_velocity),
                pressure_modes=int(n_pressure),
                center=bool(center),
            )
            offset = _basis_offset(basis, train_snapshots) if bool(center) else np.asarray(train_snapshots.state[:, 0], dtype=float)
            state_errors: list[float] = []
            velocity_errors: list[float] = []
            pressure_errors: list[float] = []
            reaction_errors: list[float] = []
            for idx in heldout_indices:
                record = snapshots.record(int(idx))
                _coefficients, reconstructed = _reconstruct_preserving_fixed(
                    basis=basis,
                    state=record.state,
                    fixed_dofs=record.fixed_dofs,
                    fixed_values=record.fixed_values,
                    offset=offset,
                )
                state_errors.append(_relative_l2(record.state[free], reconstructed[free]))
                velocity_errors.append(_relative_l2(record.state[velocity_rows], reconstructed[velocity_rows]))
                pressure_errors.append(_relative_l2(record.state[pressure_rows], reconstructed[pressure_rows]))
                if bool(include_reaction_error):
                    restore_fluid_stage(operator, record)
                    write_fluid_state(operator, reconstructed)
                    reaction = np.asarray(
                        operator.reaction_loads(refresh_state=bool(reaction_refresh_state)).values,
                        dtype=float,
                    ).reshape(-1)
                    reaction_errors.append(_relative_l2(reference_reactions[int(idx)], reaction))
            mean_state = float(np.mean(state_errors))
            mean_reaction = None if not reaction_errors else float(np.mean(reaction_errors))
            score = float(state_weight_value * mean_state)
            if mean_reaction is not None:
                score += float(reaction_weight) * float(mean_reaction)
            entries.append(
                FluidModeValidationEntry(
                    velocity_modes=int(n_velocity),
                    pressure_modes=int(n_pressure),
                    score=score,
                    state_error=mean_state,
                    velocity_error=float(np.mean(velocity_errors)),
                    pressure_error=float(np.mean(pressure_errors)),
                    reaction_error=mean_reaction,
                )
            )
    return FluidModeValidationResult(
        entries=entries,
        train_indices=train_indices.copy(),
        test_indices=heldout_indices.copy(),
        centered=bool(center),
        state_weight=float(state_weight_value),
        reaction_weight=float(reaction_weight),
    )


__all__ = [
    "FluidModeValidationEntry",
    "FluidModeValidationResult",
    "run_fluid_mode_cross_validation",
]
