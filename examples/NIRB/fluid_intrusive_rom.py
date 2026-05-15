from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Mapping

import numpy as np

from examples.NIRB.fluid_basis import FluidPODTrialBasis
from examples.NIRB.fluid_fom_operator import FluidFOMAssembly, FluidFOMOperator
from examples.NIRB.fluid_lspg import pack_fluid_state, write_fluid_state


def _as_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values.")
    return vector


def _safe_condition(matrix: np.ndarray) -> float:
    arr = np.asarray(matrix, dtype=float)
    if arr.size == 0:
        return 0.0
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return float("inf")
    if not np.all(np.isfinite(arr)):
        return float("inf")
    try:
        return float(np.linalg.cond(arr))
    except np.linalg.LinAlgError:
        return float("inf")


def _svd_values(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.size == 0:
        return np.zeros(0, dtype=float)
    try:
        return np.asarray(np.linalg.svd(arr, compute_uv=False), dtype=float)
    except np.linalg.LinAlgError:
        return np.zeros(0, dtype=float)


@dataclass(frozen=True)
class FluidReducedModeBlocks:
    """Column indices for the velocity-pressure coefficient blocks."""

    velocity: np.ndarray
    pressure: np.ndarray

    @classmethod
    def from_pod_basis(cls, basis: FluidPODTrialBasis) -> "FluidReducedModeBlocks":
        n_velocity = int(basis.n_velocity_modes)
        n_pressure = int(basis.n_pressure_modes)
        return cls(
            velocity=np.arange(n_velocity, dtype=int),
            pressure=np.arange(n_velocity, n_velocity + n_pressure, dtype=int),
        )

    @property
    def n_velocity(self) -> int:
        return int(self.velocity.size)

    @property
    def n_pressure(self) -> int:
        return int(self.pressure.size)

    @property
    def n_total(self) -> int:
        return int(self.n_velocity + self.n_pressure)


@dataclass(frozen=True)
class FluidReducedAssembly:
    """Projected residual/tangent for one ALE-DVMS contribution."""

    contribution_mode: str
    coefficients: np.ndarray
    residual: np.ndarray
    tangent: np.ndarray
    tangent_uu: np.ndarray
    tangent_up: np.ndarray
    tangent_pu: np.ndarray
    tangent_pp: np.ndarray
    residual_u: np.ndarray
    residual_p: np.ndarray
    full_residual_norm: float
    reduced_residual_norm: float

    def newton_step(self, *, rcond: float | None = None) -> np.ndarray:
        step, *_ = np.linalg.lstsq(self.tangent, -self.residual, rcond=rcond)
        return np.asarray(step, dtype=float).reshape(-1)


@dataclass(frozen=True)
class FluidReducedOperatorSplit:
    """Reduced projections of the operator pieces currently exposed by the FOM."""

    coefficients: np.ndarray
    components: Mapping[str, FluidReducedAssembly]

    def component(self, name: str) -> FluidReducedAssembly:
        return self.components[str(name)]

    @property
    def system(self) -> FluidReducedAssembly:
        return self.component("system")

    @property
    def mass(self) -> np.ndarray | None:
        lhs = self.components.get("mass_lhs")
        stab = self.components.get("mass_stabilization")
        if lhs is None and stab is None:
            return None
        if lhs is None:
            return np.asarray(stab.tangent, dtype=float)
        if stab is None:
            return np.asarray(lhs.tangent, dtype=float)
        return np.asarray(lhs.tangent, dtype=float) + np.asarray(stab.tangent, dtype=float)


@dataclass(frozen=True)
class FluidReducedStabilityReport:
    """Cheap algebraic checks for the reduced saddle-point system."""

    n_velocity_modes: int
    n_pressure_modes: int
    pressure_velocity_rank: int
    pressure_velocity_min_singular: float
    pressure_velocity_max_singular: float
    pressure_velocity_condition: float
    pressure_stabilization_norm: float
    velocity_block_condition: float
    reduced_tangent_condition: float
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.warnings


@dataclass(frozen=True)
class FluidReducedSolveResult:
    """Reduced Newton result for one full-mesh intrusive fluid stage."""

    coefficients: np.ndarray
    residual_norm: float
    iterations: int
    converged: bool
    trajectory: tuple[dict[str, float], ...] = ()


class FluidIntrusiveReducedOperator:
    """Full-mesh intrusive reduced ALE-DVMS operator.

    This is the first production-oriented fluid-ROM layer.  It does not
    hyper-reduce anything.  Instead it projects the already validated Example 2
    ALE-DVMS residual/Jacobian into separated velocity-pressure coefficient
    blocks, so stability of the mixed reduced system can be audited before any
    sampled cubature or online committed solve is attempted.
    """

    def __init__(
        self,
        *,
        operator: FluidFOMOperator,
        basis: FluidPODTrialBasis,
        test_basis: np.ndarray | None = None,
        state_update_hook: Callable[[np.ndarray], None] | None = None,
        nonlinear_update_hook: Callable[[], None] | None = None,
    ) -> None:
        if int(basis.basis.shape[0]) != int(operator.dh.total_dofs):
            raise ValueError("Fluid basis row count must match the FOM operator DOF count.")
        self.operator = operator
        self.basis = basis
        self.state_update_hook = state_update_hook
        self.nonlinear_update_hook = nonlinear_update_hook
        self.mode_blocks = FluidReducedModeBlocks.from_pod_basis(basis)
        trial = np.asarray(basis.basis, dtype=float)
        if int(trial.shape[1]) != int(self.mode_blocks.n_total):
            raise ValueError("Fluid basis column count does not match velocity/pressure mode metadata.")
        if test_basis is None:
            test = trial
        else:
            test = np.asarray(test_basis, dtype=float)
            if test.ndim != 2 or test.shape != trial.shape:
                raise ValueError("test_basis must have the same shape as the trial basis.")
        self.trial_basis = trial
        self.test_basis = test

    @property
    def n_modes(self) -> int:
        return int(self.trial_basis.shape[1])

    def reconstruct_state(self, coefficients: np.ndarray, *, offset: np.ndarray | None = None) -> np.ndarray:
        coeffs = _as_vector(coefficients, name="coefficients")
        if int(coeffs.size) != self.n_modes:
            raise ValueError(f"Expected {self.n_modes} reduced coefficients, got {coeffs.size}.")
        base = pack_fluid_state(self.operator) if offset is None else _as_vector(offset, name="offset")
        if int(base.size) != int(self.operator.dh.total_dofs):
            raise ValueError("offset size must match the fluid DOF count.")
        return base + self.trial_basis @ coeffs

    def write_state(self, coefficients: np.ndarray, *, offset: np.ndarray | None = None) -> np.ndarray:
        state = self.reconstruct_state(coefficients, offset=offset)
        write_fluid_state(self.operator, state)
        if self.state_update_hook is not None:
            self.state_update_hook(_as_vector(coefficients, name="coefficients"))
        return state

    def project_assembly(
        self,
        assembly: FluidFOMAssembly,
        *,
        coefficients: np.ndarray,
        contribution_mode: str,
    ) -> FluidReducedAssembly:
        coeffs = _as_vector(coefficients, name="coefficients")
        if int(coeffs.size) != self.n_modes:
            raise ValueError(f"Expected {self.n_modes} reduced coefficients, got {coeffs.size}.")
        if assembly.matrix is None:
            raise RuntimeError("Intrusive reduced assembly requires a FOM tangent matrix.")

        residual_full = np.asarray(assembly.residual, dtype=float).reshape(-1)
        tangent_full = assembly.matrix.tocsr()
        residual_r = np.asarray(self.test_basis.T @ residual_full, dtype=float).reshape(-1)
        tangent_r = np.asarray(self.test_basis.T @ (tangent_full @ self.trial_basis), dtype=float)

        v = self.mode_blocks.velocity
        p = self.mode_blocks.pressure
        return FluidReducedAssembly(
            contribution_mode=str(contribution_mode),
            coefficients=coeffs.copy(),
            residual=residual_r,
            tangent=tangent_r,
            tangent_uu=tangent_r[np.ix_(v, v)],
            tangent_up=tangent_r[np.ix_(v, p)],
            tangent_pu=tangent_r[np.ix_(p, v)],
            tangent_pp=tangent_r[np.ix_(p, p)],
            residual_u=residual_r[v],
            residual_p=residual_r[p],
            full_residual_norm=float(np.linalg.norm(residual_full)),
            reduced_residual_norm=float(np.linalg.norm(residual_r)),
        )

    def _operator_for_contribution(self, contribution_mode: str) -> FluidFOMOperator:
        mode = str(contribution_mode)
        if mode == str(self.operator.parameters.contribution_mode):
            return self.operator
        params = replace(self.operator.parameters, contribution_mode=mode)
        return FluidFOMOperator(
            prob=self.operator.prob,
            mesh=self.operator.mesh,
            parameters=params,
            boundary_tags=self.operator.boundary_tags,
        )

    def assemble_component(
        self,
        coefficients: np.ndarray,
        *,
        contribution_mode: str = "system",
        refresh_predicted: bool | None = None,
        offset: np.ndarray | None = None,
    ) -> FluidReducedAssembly:
        coeffs = _as_vector(coefficients, name="coefficients")
        self.write_state(coeffs, offset=offset)
        if self.nonlinear_update_hook is not None:
            self.nonlinear_update_hook()
        op = self._operator_for_contribution(str(contribution_mode))
        assembly = op.assemble(
            need_matrix=True,
            convention="newton",
            refresh_predicted=refresh_predicted,
        )
        return self.project_assembly(assembly, coefficients=coeffs, contribution_mode=str(contribution_mode))

    def solve(
        self,
        initial_coefficients: np.ndarray,
        *,
        offset: np.ndarray | None = None,
        contribution_mode: str = "system",
        max_iterations: int = 8,
        residual_tol: float = 1.0e-10,
        step_tol: float = 1.0e-12,
        line_search: bool = False,
        max_line_search: int = 6,
        sufficient_decrease: float = 1.0e-4,
    ) -> FluidReducedSolveResult:
        """Solve the reduced projected nonlinear system on the full mesh.

        This is a correctness layer, not the final speed layer.  It preserves
        the FOM state-update hooks while solving for velocity-pressure
        coefficients, then hyper-reduction can replace the expensive full
        assembly after this solve is stable.
        """

        coeffs = _as_vector(initial_coefficients, name="initial_coefficients").copy()
        if int(coeffs.size) != self.n_modes:
            raise ValueError(f"Expected {self.n_modes} reduced coefficients, got {coeffs.size}.")
        trajectory: list[dict[str, float]] = []
        last_norm = float("inf")
        for iteration in range(1, max(1, int(max_iterations)) + 1):
            assembly = self.assemble_component(
                coeffs,
                contribution_mode=str(contribution_mode),
                offset=offset,
            )
            last_norm = float(assembly.reduced_residual_norm)
            if last_norm <= float(residual_tol):
                trajectory.append(
                    {
                        "iteration": float(iteration),
                        "residual_norm": float(last_norm),
                        "step_norm": 0.0,
                        "line_search_alpha": 0.0,
                    }
                )
                return FluidReducedSolveResult(coeffs, last_norm, iteration, True, tuple(trajectory))

            step = assembly.newton_step()
            step_norm = float(np.linalg.norm(step))
            accepted = coeffs + step
            alpha_used = 1.0
            if bool(line_search):
                history = self.operator.snapshot_history()
                best_coeffs = accepted
                best_norm = float("inf")
                for search_iter in range(max(1, int(max_line_search))):
                    alpha = 0.5**search_iter
                    trial = coeffs + float(alpha) * step
                    self.operator.restore_history(history)
                    trial_assembly = self.assemble_component(
                        trial,
                        contribution_mode=str(contribution_mode),
                        offset=offset,
                    )
                    trial_norm = float(trial_assembly.reduced_residual_norm)
                    if trial_norm < best_norm:
                        best_norm = trial_norm
                        best_coeffs = trial
                        alpha_used = float(alpha)
                    if trial_norm <= (1.0 - float(sufficient_decrease) * float(alpha)) * last_norm:
                        best_coeffs = trial
                        alpha_used = float(alpha)
                        break
                self.operator.restore_history(history)
                accepted = np.asarray(best_coeffs, dtype=float).reshape(-1)

            coeffs = np.asarray(accepted, dtype=float).reshape(-1)
            trajectory.append(
                {
                    "iteration": float(iteration),
                    "residual_norm": float(last_norm),
                    "step_norm": float(step_norm),
                    "line_search_alpha": float(alpha_used),
                }
            )
            if step_norm <= float(step_tol) * max(1.0, float(np.linalg.norm(coeffs))):
                assembly = self.assemble_component(
                    coeffs,
                    contribution_mode=str(contribution_mode),
                    offset=offset,
                )
                final_norm = float(assembly.reduced_residual_norm)
                return FluidReducedSolveResult(
                    coeffs,
                    final_norm,
                    iteration,
                    final_norm <= float(residual_tol),
                    tuple(trajectory),
                )

        assembly = self.assemble_component(
            coeffs,
            contribution_mode=str(contribution_mode),
            offset=offset,
        )
        final_norm = float(assembly.reduced_residual_norm)
        return FluidReducedSolveResult(
            coeffs,
            final_norm,
            max(1, int(max_iterations)),
            final_norm <= float(residual_tol),
            tuple(trajectory),
        )

    def assemble_split(
        self,
        coefficients: np.ndarray,
        *,
        contribution_modes: Iterable[str] = ("system", "mass_lhs", "mass_stabilization", "velocity"),
        offset: np.ndarray | None = None,
    ) -> FluidReducedOperatorSplit:
        coeffs = _as_vector(coefficients, name="coefficients")
        self.write_state(coeffs, offset=offset)
        if self.nonlinear_update_hook is not None:
            self.nonlinear_update_hook()
        self.operator.refresh_predicted_subscale()

        components: dict[str, FluidReducedAssembly] = {}
        for mode in contribution_modes:
            name = str(mode)
            op = self._operator_for_contribution(name)
            assembly = op.assemble(need_matrix=True, convention="newton", refresh_predicted=False)
            components[name] = self.project_assembly(assembly, coefficients=coeffs, contribution_mode=name)
        return FluidReducedOperatorSplit(coefficients=coeffs.copy(), components=components)

    def stability_report(
        self,
        assembly: FluidReducedAssembly,
        *,
        rank_tol: float = 1.0e-10,
        condition_limit: float = 1.0e12,
    ) -> FluidReducedStabilityReport:
        pu = np.asarray(assembly.tangent_pu, dtype=float)
        pp = np.asarray(assembly.tangent_pp, dtype=float)
        svals = _svd_values(pu)
        max_sv = float(svals[0]) if svals.size else 0.0
        min_sv = float(svals[-1]) if svals.size else 0.0
        threshold = float(rank_tol) * max(1.0, max_sv)
        rank = int(np.count_nonzero(svals > threshold))
        pressure_condition = float(max_sv / max(min_sv, 1.0e-300)) if svals.size else float("inf")
        pp_norm = float(np.linalg.norm(pp))
        velocity_condition = _safe_condition(assembly.tangent_uu)
        reduced_condition = _safe_condition(assembly.tangent)

        warnings: list[str] = []
        if self.mode_blocks.n_velocity == 0:
            warnings.append("no velocity modes")
        if self.mode_blocks.n_pressure == 0:
            warnings.append("no pressure modes")
        if self.mode_blocks.n_pressure and rank < min(self.mode_blocks.n_pressure, self.mode_blocks.n_velocity):
            if pp_norm <= threshold:
                warnings.append("rank-deficient pressure/velocity coupling without pressure stabilization")
            else:
                warnings.append("rank-deficient pressure/velocity coupling; relying on pressure stabilization")
        if not np.all(np.isfinite(assembly.tangent)):
            warnings.append("non-finite reduced tangent")
        if float(reduced_condition) > float(condition_limit):
            warnings.append(f"ill-conditioned reduced tangent: cond={float(reduced_condition):.3e}")

        return FluidReducedStabilityReport(
            n_velocity_modes=self.mode_blocks.n_velocity,
            n_pressure_modes=self.mode_blocks.n_pressure,
            pressure_velocity_rank=rank,
            pressure_velocity_min_singular=min_sv,
            pressure_velocity_max_singular=max_sv,
            pressure_velocity_condition=pressure_condition,
            pressure_stabilization_norm=pp_norm,
            velocity_block_condition=velocity_condition,
            reduced_tangent_condition=reduced_condition,
            warnings=tuple(warnings),
        )


__all__ = [
    "FluidIntrusiveReducedOperator",
    "FluidReducedAssembly",
    "FluidReducedModeBlocks",
    "FluidReducedOperatorSplit",
    "FluidReducedSolveResult",
    "FluidReducedStabilityReport",
]
