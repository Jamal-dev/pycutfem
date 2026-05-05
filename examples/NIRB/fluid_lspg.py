from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from examples.NIRB.fluid_fom_operator import FluidFOMOperator


def pack_fluid_state(operator: FluidFOMOperator) -> np.ndarray:
    """Pack Example 2 fluid unknowns `[ux, uy, p]` into global DOF layout."""

    dh = operator.dh
    prob = operator.prob
    values = np.zeros(int(dh.total_dofs), dtype=float)
    u_k = prob["u_k"]
    p_k = prob["p_k"]
    for component in u_k.components:
        ids = np.asarray(dh.get_field_slice(component.field_name), dtype=int)
        values[ids] = np.asarray(component.get_nodal_values(ids), dtype=float)
    p_ids = np.asarray(dh.get_field_slice(p_k.field_name), dtype=int)
    values[p_ids] = np.asarray(p_k.get_nodal_values(p_ids), dtype=float)
    return values


def write_fluid_state(operator: FluidFOMOperator, values: np.ndarray) -> None:
    """Write a global `[ux, uy, p]` state vector into the Example 2 fluid fields."""

    dh = operator.dh
    prob = operator.prob
    state = np.asarray(values, dtype=float).reshape(-1)
    if int(state.size) != int(dh.total_dofs):
        raise ValueError(f"Fluid state has size {state.size}, expected {int(dh.total_dofs)}.")
    u_k = prob["u_k"]
    p_k = prob["p_k"]
    for component in u_k.components:
        ids = np.asarray(dh.get_field_slice(component.field_name), dtype=int)
        component.set_nodal_values(ids, state[ids])
    p_ids = np.asarray(dh.get_field_slice(p_k.field_name), dtype=int)
    p_k.set_nodal_values(p_ids, state[p_ids])


@dataclass(frozen=True)
class FluidTrialSpace:
    """Affine fluid trial space `U = offset + basis y` in global DOF layout."""

    basis: np.ndarray
    offset: np.ndarray
    free_dofs: np.ndarray

    def __post_init__(self) -> None:
        basis = np.asarray(self.basis, dtype=float)
        offset = np.asarray(self.offset, dtype=float).reshape(-1)
        free = np.asarray(self.free_dofs, dtype=int).reshape(-1)
        if basis.ndim != 2:
            raise ValueError("FluidTrialSpace.basis must be a 2D matrix.")
        if int(basis.shape[0]) != int(offset.size):
            raise ValueError("FluidTrialSpace basis rows must match offset size.")
        if np.any(free < 0) or np.any(free >= int(offset.size)):
            raise ValueError("FluidTrialSpace free_dofs contains out-of-range entries.")
        if np.unique(free).size != free.size:
            raise ValueError("FluidTrialSpace free_dofs must be unique.")
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "free_dofs", free)

    @property
    def n_dofs(self) -> int:
        return int(self.basis.shape[0])

    @property
    def n_modes(self) -> int:
        return int(self.basis.shape[1])

    @classmethod
    def from_free_basis(
        cls,
        *,
        operator: FluidFOMOperator,
        free_basis: np.ndarray,
        free_dofs: np.ndarray | None = None,
        offset: np.ndarray | None = None,
    ) -> "FluidTrialSpace":
        free = operator.free_fluid_dofs() if free_dofs is None else np.asarray(free_dofs, dtype=int).reshape(-1)
        basis_free = np.asarray(free_basis, dtype=float)
        if basis_free.ndim != 2:
            raise ValueError("free_basis must be a 2D matrix.")
        if int(basis_free.shape[0]) != int(free.size):
            raise ValueError("free_basis row count must match free_dofs size.")
        if np.any(free < 0) or np.any(free >= int(operator.dh.total_dofs)):
            raise ValueError("free_dofs contains out-of-range entries.")
        if np.unique(free).size != free.size:
            raise ValueError("free_dofs must be unique.")
        base = pack_fluid_state(operator) if offset is None else np.asarray(offset, dtype=float).reshape(-1)
        full_basis = np.zeros((int(operator.dh.total_dofs), int(basis_free.shape[1])), dtype=float)
        full_basis[free, :] = basis_free
        return cls(basis=full_basis, offset=base, free_dofs=free)

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(coefficients, dtype=float).reshape(-1)
        if int(coeffs.size) != self.n_modes:
            raise ValueError(f"Expected {self.n_modes} coefficients, got {coeffs.size}.")
        return self.offset + self.basis @ coeffs

    def write(self, operator: FluidFOMOperator, coefficients: np.ndarray) -> np.ndarray:
        state = self.reconstruct(coefficients)
        write_fluid_state(operator, state)
        return state


@dataclass(frozen=True)
class FluidLSPGSystem:
    coefficients: np.ndarray
    residual: np.ndarray
    trial_jacobian: np.ndarray
    weighted_residual: np.ndarray
    weighted_trial_jacobian: np.ndarray
    normal_matrix: np.ndarray
    normal_rhs: np.ndarray
    residual_norm: float
    row_dofs: np.ndarray

    def gauss_newton_step(self, *, rcond: float | None = None) -> np.ndarray:
        step, *_ = np.linalg.lstsq(self.weighted_trial_jacobian, -self.weighted_residual, rcond=rcond)
        return np.asarray(step, dtype=float).reshape(-1)


@dataclass(frozen=True)
class FluidLSPGResult:
    coefficients: np.ndarray
    residual_norm: float
    iterations: int
    converged: bool
    trajectory: tuple[dict[str, float], ...] = ()


class FluidLSPGVerifier:
    """Full-operator LSPG algebra for validating fluid POD spaces.

    This class intentionally does not hyper-reduce anything.  It is the
    correctness layer that proves the affine trial space, Dirichlet lifting,
    residual convention, and projected Gauss-Newton algebra before sampled
    GNAT/empirical-quadrature assembly is introduced.
    """

    def __init__(
        self,
        *,
        operator: FluidFOMOperator,
        trial_space: FluidTrialSpace,
        row_dofs: np.ndarray | None = None,
        row_weights: np.ndarray | None = None,
        state_update_hook: Callable[[np.ndarray], None] | None = None,
        nonlinear_update_hook: Callable[[], None] | None = None,
    ) -> None:
        self.operator = operator
        self.trial_space = trial_space
        self.state_update_hook = state_update_hook
        self.nonlinear_update_hook = nonlinear_update_hook
        rows = trial_space.free_dofs if row_dofs is None else np.asarray(row_dofs, dtype=int).reshape(-1)
        if np.any(rows < 0) or np.any(rows >= int(operator.dh.total_dofs)):
            raise ValueError("row_dofs contains out-of-range entries.")
        if np.unique(rows).size != rows.size:
            raise ValueError("row_dofs must be unique.")
        self.row_dofs = rows
        if row_weights is None:
            self.row_weights = None
        else:
            weights = np.asarray(row_weights, dtype=float).reshape(-1)
            if int(weights.size) != int(rows.size):
                raise ValueError("row_weights size must match row_dofs size.")
            if np.any(weights < 0.0):
                raise ValueError("row_weights must be nonnegative.")
            self.row_weights = weights

    def assemble_system(
        self,
        coefficients: np.ndarray,
        *,
        element_ids: np.ndarray | None = None,
        refresh_predicted: bool | None = None,
    ) -> FluidLSPGSystem:
        coeffs = np.asarray(coefficients, dtype=float).reshape(-1)
        self.trial_space.write(self.operator, coeffs)
        if self.state_update_hook is not None:
            self.state_update_hook(coeffs)
        assembly = self.operator.assemble(
            need_matrix=True,
            element_ids=element_ids,
            convention="newton",
            refresh_predicted=refresh_predicted,
        )
        if assembly.matrix is None:
            raise RuntimeError("LSPG assembly requires a Jacobian matrix.")
        rows = np.asarray(self.row_dofs, dtype=int)
        residual = np.asarray(assembly.residual[rows], dtype=float).reshape(-1)
        trial_jacobian = np.asarray(assembly.matrix[rows, :] @ self.trial_space.basis, dtype=float)
        if self.row_weights is not None:
            scale = np.sqrt(np.asarray(self.row_weights, dtype=float)).reshape(-1)
            residual_for_normal = residual * scale
            trial_for_normal = trial_jacobian * scale[:, None]
        else:
            residual_for_normal = residual
            trial_for_normal = trial_jacobian
        normal_matrix = trial_for_normal.T @ trial_for_normal
        normal_rhs = -(trial_for_normal.T @ residual_for_normal)
        residual_norm = float(np.linalg.norm(residual_for_normal))
        return FluidLSPGSystem(
            coefficients=coeffs.copy(),
            residual=residual,
            trial_jacobian=trial_jacobian,
            weighted_residual=np.asarray(residual_for_normal, dtype=float).reshape(-1),
            weighted_trial_jacobian=np.asarray(trial_for_normal, dtype=float),
            normal_matrix=np.asarray(normal_matrix, dtype=float),
            normal_rhs=np.asarray(normal_rhs, dtype=float).reshape(-1),
            residual_norm=residual_norm,
            row_dofs=rows.copy(),
        )

    def solve(
        self,
        initial_coefficients: np.ndarray,
        *,
        max_iterations: int = 8,
        residual_tol: float = 1.0e-10,
        step_tol: float = 1.0e-12,
        element_ids: np.ndarray | None = None,
        line_search: bool = False,
        max_line_search: int = 6,
        sufficient_decrease: float = 1.0e-4,
    ) -> FluidLSPGResult:
        coeffs = np.asarray(initial_coefficients, dtype=float).reshape(-1).copy()
        if int(coeffs.size) != self.trial_space.n_modes:
            raise ValueError(f"Expected {self.trial_space.n_modes} coefficients, got {coeffs.size}.")
        last_norm = float("inf")
        trajectory: list[dict[str, float]] = []
        for iteration in range(1, max(1, int(max_iterations)) + 1):
            system = self.assemble_system(coeffs, element_ids=element_ids)
            last_norm = float(system.residual_norm)
            if last_norm <= float(residual_tol):
                trajectory.append(
                    {
                        "iteration": float(iteration),
                        "residual_norm": float(last_norm),
                        "step_norm": 0.0,
                        "line_search_alpha": 0.0,
                        "coefficients": np.asarray(coeffs, dtype=float).reshape(-1).tolist(),
                    }
                )
                return FluidLSPGResult(coeffs, last_norm, iteration, True, tuple(trajectory))
            step = system.gauss_newton_step()
            step_norm = float(np.linalg.norm(step))
            line_search_alpha = 1.0
            if bool(line_search):
                history = self.operator.snapshot_history()
                best_coeffs = coeffs + step
                best_norm = float("inf")
                accepted_coeffs = best_coeffs
                accepted_alpha = 1.0
                for search_iter in range(max(1, int(max_line_search))):
                    alpha = 0.5**search_iter
                    trial_coeffs = coeffs + float(alpha) * step
                    self.operator.restore_history(history)
                    trial_system = self.assemble_system(trial_coeffs, element_ids=element_ids)
                    trial_norm = float(trial_system.residual_norm)
                    if trial_norm < best_norm:
                        best_norm = trial_norm
                        best_coeffs = trial_coeffs
                        accepted_alpha = float(alpha)
                    if trial_norm <= (1.0 - float(sufficient_decrease) * float(alpha)) * last_norm:
                        accepted_coeffs = trial_coeffs
                        accepted_alpha = float(alpha)
                        break
                else:
                    accepted_coeffs = best_coeffs
                self.operator.restore_history(history)
                coeffs = np.asarray(accepted_coeffs, dtype=float).reshape(-1)
                line_search_alpha = float(accepted_alpha)
            else:
                coeffs = coeffs + step
            if self.nonlinear_update_hook is not None:
                self.trial_space.write(self.operator, coeffs)
                if self.state_update_hook is not None:
                    self.state_update_hook(coeffs)
                self.nonlinear_update_hook()
            trajectory.append(
                {
                    "iteration": float(iteration),
                    "residual_norm": float(last_norm),
                    "step_norm": float(step_norm),
                    "line_search_alpha": float(line_search_alpha),
                    "coefficients": np.asarray(coeffs, dtype=float).reshape(-1).tolist(),
                }
            )
            if step_norm <= float(step_tol) * max(1.0, float(np.linalg.norm(coeffs))):
                system = self.assemble_system(coeffs, element_ids=element_ids)
                return FluidLSPGResult(
                    coeffs,
                    float(system.residual_norm),
                    iteration,
                    float(system.residual_norm) <= float(residual_tol),
                    tuple(trajectory),
                )
        system = self.assemble_system(coeffs, element_ids=element_ids)
        last_norm = float(system.residual_norm)
        return FluidLSPGResult(
            coeffs,
            last_norm,
            max(1, int(max_iterations)),
            last_norm <= float(residual_tol),
            tuple(trajectory),
        )


__all__ = [
    "FluidLSPGResult",
    "FluidLSPGSystem",
    "FluidLSPGVerifier",
    "FluidTrialSpace",
    "pack_fluid_state",
    "write_fluid_state",
]
