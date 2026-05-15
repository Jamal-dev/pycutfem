from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pycutfem.mor.pod import fit_pod

from examples.NIRB.fluid_fom_operator import FluidFOMOperator
from examples.NIRB.fluid_lspg import FluidLSPGResult, FluidTrialSpace
from examples.NIRB.reduced_fluid import (
    ReducedFluidNativeOnlineSpec,
    reduced_fluid_gnat_step_backend,
    reduced_fluid_online_backend,
)
from examples.NIRB.fluid_snapshots import FluidStageSnapshotBatch, restore_fluid_stage
from examples.NIRB.run_example2_local import (
    _assemble_fluid_sampled_galerkin_reduced_system_raw,
    _assemble_fluid_sampled_lspg_rows_raw,
    _fluid_interface_reaction_element_ids,
)


def _unique_in_order(values: np.ndarray) -> np.ndarray:
    seen: set[int] = set()
    out: list[int] = []
    for value in np.asarray(values, dtype=int).reshape(-1):
        item = int(value)
        if item not in seen:
            seen.add(item)
            out.append(item)
    return np.asarray(out, dtype=int)


def fluid_element_dofs(
    operator: FluidFOMOperator,
    element_ids: np.ndarray,
    *,
    field_names: tuple[str, ...] = ("ux", "uy", "p"),
) -> np.ndarray:
    dh = operator.dh
    eids = np.asarray(element_ids, dtype=int).reshape(-1)
    if eids.size == 0:
        return np.zeros(0, dtype=int)
    rows: list[np.ndarray] = []
    for field_name in field_names:
        rows.append(np.asarray(dh.element_maps[str(field_name)], dtype=int)[eids].reshape(-1))
    values = np.concatenate(rows)
    return np.unique(values[values >= 0]).astype(int, copy=False)


def fluid_elements_touching_rows(
    operator: FluidFOMOperator,
    row_dofs: np.ndarray,
    *,
    field_names: tuple[str, ...] = ("ux", "uy", "p"),
) -> np.ndarray:
    rows = np.asarray(row_dofs, dtype=int).reshape(-1)
    if rows.size == 0:
        return np.zeros(0, dtype=int)
    mask = np.zeros(int(operator.mesh.n_elements), dtype=bool)
    for field_name in field_names:
        element_map = np.asarray(operator.dh.element_maps[str(field_name)], dtype=int)
        mask |= np.isin(element_map, rows).any(axis=1)
    return np.flatnonzero(mask).astype(int, copy=False)


def _fit_residual_basis(
    residual_snapshots: np.ndarray,
    *,
    residual_modes: int | None,
    residual_energy: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(residual_snapshots, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("residual_snapshots must be a feature-major matrix.")
    if matrix.shape[0] == 0:
        raise ValueError("Cannot fit a GNAT residual basis with zero residual rows.")
    pod = fit_pod(
        matrix,
        n_modes=residual_modes,
        energy=residual_energy,
        center=False,
    )
    return (
        np.asarray(pod.basis, dtype=float),
        np.asarray(pod.singular_values, dtype=float),
        np.asarray(pod.energy_fraction, dtype=float),
    )


def _qdeim_rows(
    basis_at_candidate_rows: np.ndarray,
    candidate_rows: np.ndarray,
    *,
    target_count: int,
) -> np.ndarray:
    basis = np.asarray(basis_at_candidate_rows, dtype=float)
    rows = np.asarray(candidate_rows, dtype=int).reshape(-1)
    if basis.ndim != 2:
        raise ValueError("basis_at_candidate_rows must be a matrix.")
    if basis.shape[0] != rows.size:
        raise ValueError("candidate row count must match basis rows.")
    if rows.size == 0 or target_count <= 0:
        return np.zeros(0, dtype=int)
    count = min(int(target_count), int(rows.size))
    try:
        from scipy.linalg import qr

        _q, _r, pivots = qr(basis.T, mode="economic", pivoting=True)
        selected = rows[np.asarray(pivots, dtype=int)[:count]]
    except Exception:
        norms = np.linalg.norm(basis, axis=1)
        selected = rows[np.argsort(norms)[::-1][:count]]
    if selected.size < count:
        missing = np.setdiff1d(rows, selected, assume_unique=False)
        selected = np.concatenate([selected, missing[: count - selected.size]])
    return _unique_in_order(selected)


def _local_positions(global_rows: np.ndarray, selected_rows: np.ndarray) -> np.ndarray:
    row_to_local = {int(row): idx for idx, row in enumerate(np.asarray(global_rows, dtype=int).reshape(-1))}
    try:
        return np.asarray([row_to_local[int(row)] for row in np.asarray(selected_rows, dtype=int).reshape(-1)], dtype=int)
    except KeyError as exc:
        raise ValueError("selected rows must be a subset of global_rows.") from exc


def _gram_fit_equations(matrix: np.ndarray, sample_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be 2D for sample-weight fitting.")
    positions = np.asarray(sample_positions, dtype=int).reshape(-1)
    if values.shape[1] == 0 or positions.size == 0:
        return np.zeros((0, positions.size), dtype=float), np.zeros(0, dtype=float)
    sample = values[positions, :]
    full_gram = values.T @ values
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for j in range(int(values.shape[1])):
        for k in range(j, int(values.shape[1])):
            coeff = np.asarray(sample[:, j] * sample[:, k], dtype=float)
            target = float(full_gram[j, k])
            if j != k:
                # Preserve Frobenius weighting when using only the upper triangle.
                coeff = np.sqrt(2.0) * coeff
                target = float(np.sqrt(2.0) * target)
            scale = max(float(np.linalg.norm(coeff)), abs(target), 1.0e-14)
            rows.append(coeff / scale)
            rhs.append(target / scale)
    return np.vstack(rows), np.asarray(rhs, dtype=float)


def _fit_nonnegative_sample_weights(
    *,
    matrix: np.ndarray,
    sample_positions: np.ndarray,
    method: str,
) -> tuple[np.ndarray, float]:
    """Fit row weights so sampled rows preserve a full Gram matrix."""

    kind = str(method).strip().lower().replace("_", "-")
    positions = np.asarray(sample_positions, dtype=int).reshape(-1)
    if kind in {"", "none", "identity"}:
        return np.ones(int(positions.size), dtype=float), 0.0
    bounded = kind.endswith("-bounded") or kind.endswith("-clipped")
    equations, targets = _gram_fit_equations(np.asarray(matrix, dtype=float), positions)
    if equations.size == 0:
        return np.ones(int(positions.size), dtype=float), 0.0
    if bounded:
        weights, *_ = np.linalg.lstsq(equations, targets, rcond=None)
        weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    else:
        try:
            from scipy.optimize import nnls

            weights, _resnorm = nnls(equations, targets, maxiter=max(3 * int(positions.size), 1))
        except Exception:
            weights, *_ = np.linalg.lstsq(equations, targets, rcond=None)
            weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size != positions.size:
        raise RuntimeError("sample-weight fit returned an invalid weight vector.")
    if not np.any(weights > 0.0):
        weights = np.ones(int(positions.size), dtype=float)
    if bounded:
        weights = np.maximum(weights, 0.0)
        total = float(np.sum(weights))
        if total > 0.0:
            weights *= float(positions.size) / total
        weights = np.clip(weights, 1.0e-4, 1.0e2)
        total = float(np.sum(weights))
        if total > 0.0:
            weights *= float(positions.size) / total
        weights = np.clip(weights, 1.0e-4, 1.0e2)
    residual = equations @ weights - targets
    relative_error = float(np.linalg.norm(residual) / max(float(np.linalg.norm(targets)), 1.0e-15))
    return weights, relative_error


def _weighted_sample_lift(sample_basis: np.ndarray, weights: np.ndarray) -> np.ndarray:
    basis = np.asarray(sample_basis, dtype=float)
    sample_weights = np.asarray(weights, dtype=float).reshape(-1)
    if basis.ndim != 2:
        raise ValueError("sample_basis must be 2D.")
    if sample_weights.size != basis.shape[0]:
        raise ValueError("sample weights must match sample_basis rows.")
    sqrt_weights = np.sqrt(np.maximum(sample_weights, 0.0))
    weighted_basis = basis * sqrt_weights[:, None]
    return np.linalg.pinv(weighted_basis) * sqrt_weights[None, :]


@dataclass(frozen=True)
class FluidGNATSampleSet:
    """GNAT residual basis and sample mesh for the Example 2 fluid operator."""

    residual_basis: np.ndarray
    residual_singular_values: np.ndarray
    residual_energy_fraction: np.ndarray
    basis_dofs: np.ndarray
    row_dofs: np.ndarray
    element_ids: np.ndarray
    sample_to_residual_coefficients: np.ndarray
    sampled_basis_rank: int
    sampled_basis_condition: float
    sample_weights: np.ndarray | None = None
    sample_weighting: str = "none"
    sample_weight_fit_relative_error: float = 0.0
    element_weights: np.ndarray | None = None
    element_weighting: str = "none"
    element_weight_fit_relative_error: float = 0.0

    def __post_init__(self) -> None:
        residual_basis = np.asarray(self.residual_basis, dtype=float)
        if residual_basis.ndim != 2:
            raise ValueError("residual_basis must be a matrix.")
        basis_dofs = np.asarray(self.basis_dofs, dtype=int).reshape(-1)
        row_dofs = np.asarray(self.row_dofs, dtype=int).reshape(-1)
        element_ids = np.asarray(self.element_ids, dtype=int).reshape(-1)
        if np.unique(basis_dofs).size != basis_dofs.size:
            raise ValueError("basis_dofs must be unique.")
        if np.unique(row_dofs).size != row_dofs.size:
            raise ValueError("row_dofs must be unique.")
        if np.unique(element_ids).size != element_ids.size:
            raise ValueError("element_ids must be unique.")
        if np.setdiff1d(row_dofs, basis_dofs).size:
            raise ValueError("GNAT sample rows must be a subset of residual-basis DOFs.")
        sample_to_coeffs = np.asarray(self.sample_to_residual_coefficients, dtype=float)
        if sample_to_coeffs.shape != (residual_basis.shape[1], row_dofs.size):
            raise ValueError("sample_to_residual_coefficients has incompatible shape.")
        if self.sample_weights is None:
            sample_weights = np.ones(int(row_dofs.size), dtype=float)
        else:
            sample_weights = np.asarray(self.sample_weights, dtype=float).reshape(-1)
            if sample_weights.size != row_dofs.size:
                raise ValueError("sample_weights size must match row_dofs.")
            if np.any(sample_weights < 0.0) or not np.all(np.isfinite(sample_weights)):
                raise ValueError("sample_weights must be finite and nonnegative.")
        if self.element_weights is None:
            element_weights = np.ones(int(element_ids.size), dtype=float)
        else:
            element_weights = np.asarray(self.element_weights, dtype=float).reshape(-1)
            if element_weights.size != element_ids.size:
                raise ValueError("element_weights size must match element_ids.")
            if np.any(element_weights < 0.0) or not np.all(np.isfinite(element_weights)):
                raise ValueError("element_weights must be finite and nonnegative.")
        object.__setattr__(self, "residual_basis", residual_basis)
        object.__setattr__(self, "residual_singular_values", np.asarray(self.residual_singular_values, dtype=float))
        object.__setattr__(self, "residual_energy_fraction", np.asarray(self.residual_energy_fraction, dtype=float))
        object.__setattr__(self, "basis_dofs", basis_dofs)
        object.__setattr__(self, "row_dofs", row_dofs)
        object.__setattr__(self, "element_ids", element_ids)
        object.__setattr__(self, "sample_to_residual_coefficients", sample_to_coeffs)
        object.__setattr__(self, "sample_weights", sample_weights)
        object.__setattr__(self, "sample_weighting", str(self.sample_weighting))
        object.__setattr__(self, "sample_weight_fit_relative_error", float(self.sample_weight_fit_relative_error))
        object.__setattr__(self, "element_weights", element_weights)
        object.__setattr__(self, "element_weighting", str(self.element_weighting))
        object.__setattr__(self, "element_weight_fit_relative_error", float(self.element_weight_fit_relative_error))

    @property
    def n_residual_modes(self) -> int:
        return int(self.residual_basis.shape[1])

    @property
    def n_sample_rows(self) -> int:
        return int(self.row_dofs.size)

    @property
    def n_sample_elements(self) -> int:
        return int(self.element_ids.size)


@dataclass(frozen=True)
class FluidGNATSystem:
    coefficients: np.ndarray
    sampled_residual: np.ndarray
    sampled_trial_jacobian: np.ndarray
    residual_coefficients: np.ndarray
    gnat_trial_jacobian: np.ndarray
    normal_matrix: np.ndarray
    normal_rhs: np.ndarray
    estimated_residual_norm: float
    row_dofs: np.ndarray
    element_ids: np.ndarray

    def gauss_newton_step(self, *, rcond: float | None = None, backend: str = "python") -> np.ndarray:
        backend_name = str(backend).strip().lower()
        if backend_name in {"cpp", "c++"}:
            from pycutfem.mor import gauss_newton_step

            return gauss_newton_step(
                self.gnat_trial_jacobian,
                self.residual_coefficients,
                backend="cpp",
                rcond=rcond,
            ).step
        if backend_name != "python":
            raise ValueError("FluidGNATSystem Gauss-Newton step backend must be 'python' or 'cpp'.")
        step, *_ = np.linalg.lstsq(self.gnat_trial_jacobian, -self.residual_coefficients, rcond=rcond)
        return np.asarray(step, dtype=float).reshape(-1)


class FluidGNATSolver:
    """Sample-mesh GNAT solver using the exact Example 2 local ALE-DVMS operator."""

    def __init__(
        self,
        *,
        operator: FluidFOMOperator,
        trial_space: FluidTrialSpace,
        sample_set: FluidGNATSampleSet,
        state_update_hook: Callable[[np.ndarray], None] | None = None,
        nonlinear_update_hook: Callable[[], None] | None = None,
        objective: str = "gnat",
        row_weights: np.ndarray | None = None,
        incompressibility_stabilization_scale: float = 1.0,
        native_online_spec: ReducedFluidNativeOnlineSpec | None = None,
        online_backend: str | None = None,
        gnat_step_backend: str | None = None,
    ) -> None:
        self.operator = operator
        self.trial_space = trial_space
        self.sample_set = sample_set
        self.state_update_hook = state_update_hook
        self.nonlinear_update_hook = nonlinear_update_hook
        objective_key = str(objective).strip().lower().replace("-", "_")
        if objective_key not in {"gnat", "sampled_lspg", "sampled_galerkin"}:
            raise ValueError("FluidGNATSolver objective must be 'gnat', 'sampled_lspg', or 'sampled_galerkin'.")
        self.objective = objective_key
        if row_weights is None:
            self.row_weights = None
        else:
            weights = np.asarray(row_weights, dtype=float).reshape(-1)
            if weights.size != sample_set.row_dofs.size:
                raise ValueError("FluidGNATSolver row_weights must match sample_set.row_dofs.")
            if np.any(weights < 0.0):
                raise ValueError("FluidGNATSolver row_weights must be nonnegative.")
            self.row_weights = weights
        self.incompressibility_stabilization_scale = float(incompressibility_stabilization_scale)
        if (
            not np.isfinite(self.incompressibility_stabilization_scale)
            or self.incompressibility_stabilization_scale <= 0.0
        ):
            raise ValueError("FluidGNATSolver incompressibility_stabilization_scale must be finite and positive.")
        online_backend_name = (
            reduced_fluid_online_backend() if online_backend is None else str(online_backend).strip().lower()
        )
        if online_backend_name not in {"python", "cpp"}:
            raise ValueError("FluidGNATSolver online_backend must be 'python' or 'cpp'.")
        step_backend_name = (
            reduced_fluid_gnat_step_backend() if gnat_step_backend is None else str(gnat_step_backend).strip().lower()
        )
        if step_backend_name in {"c++"}:
            step_backend_name = "cpp"
        if step_backend_name not in {"python", "cpp"}:
            raise ValueError("FluidGNATSolver gnat_step_backend must be 'python' or 'cpp'.")
        self.native_online_spec = native_online_spec
        self.online_backend = online_backend_name
        self.gnat_step_backend = step_backend_name

    def assemble_system(
        self,
        coefficients: np.ndarray,
        *,
        refresh_predicted: bool | None = None,
    ) -> FluidGNATSystem:
        coeffs = np.asarray(coefficients, dtype=float).reshape(-1)
        self.trial_space.write(self.operator, coeffs)
        if self.state_update_hook is not None:
            self.state_update_hook(coeffs)
        rows = np.asarray(self.sample_set.row_dofs, dtype=int)
        element_weights = np.asarray(self.sample_set.element_weights, dtype=float).reshape(-1)
        if self.objective == "sampled_galerkin":
            if refresh_predicted is None or bool(refresh_predicted):
                self.operator.refresh_predicted_subscale()
            p = self.operator.parameters
            reduced_residual, reduced_tangent = _assemble_fluid_sampled_galerkin_reduced_system_raw(
                prob=self.operator.prob,
                rho_f=float(p.rho_f),
                mu_f=float(p.mu_f),
                dt=float(p.dt),
                quad_order=int(p.quadrature_order),
                bossak_alpha=float(p.bossak_alpha),
                contribution_mode=str(p.contribution_mode),
                backend=str(p.backend),
                element_ids=self.sample_set.element_ids,
                basis=self.trial_space.basis,
                element_weights=element_weights,
                incompressibility_stabilization_scale=float(self.incompressibility_stabilization_scale),
            )
            residual_coefficients = np.asarray(reduced_residual, dtype=float).reshape(-1)
            gnat_trial_jacobian = np.asarray(reduced_tangent, dtype=float)
            normal_matrix = gnat_trial_jacobian.T @ gnat_trial_jacobian
            normal_rhs = -(gnat_trial_jacobian.T @ residual_coefficients)
            return FluidGNATSystem(
                coefficients=coeffs.copy(),
                sampled_residual=residual_coefficients.copy(),
                sampled_trial_jacobian=gnat_trial_jacobian.copy(),
                residual_coefficients=residual_coefficients,
                gnat_trial_jacobian=gnat_trial_jacobian,
                normal_matrix=np.asarray(normal_matrix, dtype=float),
                normal_rhs=np.asarray(normal_rhs, dtype=float).reshape(-1),
                estimated_residual_norm=float(np.linalg.norm(residual_coefficients)),
                row_dofs=rows.copy(),
                element_ids=np.asarray(self.sample_set.element_ids, dtype=int).copy(),
            )

        use_element_cubature = element_weights.size and not np.allclose(element_weights, 1.0)
        use_scaled_operator = abs(float(self.incompressibility_stabilization_scale) - 1.0) > 1.0e-14
        if bool(use_element_cubature) or bool(use_scaled_operator):
            if refresh_predicted is None or bool(refresh_predicted):
                self.operator.refresh_predicted_subscale()
            p = self.operator.parameters
            sampled_residual, sampled_trial_jacobian = _assemble_fluid_sampled_lspg_rows_raw(
                prob=self.operator.prob,
                rho_f=float(p.rho_f),
                mu_f=float(p.mu_f),
                dt=float(p.dt),
                quad_order=int(p.quadrature_order),
                bossak_alpha=float(p.bossak_alpha),
                contribution_mode=str(p.contribution_mode),
                backend=str(p.backend),
                element_ids=self.sample_set.element_ids,
                row_dofs=rows,
                basis=self.trial_space.basis,
                element_weights=element_weights,
                incompressibility_stabilization_scale=float(self.incompressibility_stabilization_scale),
            )
        else:
            assembly = self.operator.assemble(
                need_matrix=True,
                element_ids=self.sample_set.element_ids,
                convention="newton",
                refresh_predicted=refresh_predicted,
            )
            if assembly.matrix is None:
                raise RuntimeError("GNAT assembly requires a sampled Jacobian matrix.")
            sampled_residual = np.asarray(assembly.residual[rows], dtype=float).reshape(-1)
            sampled_trial_jacobian = np.asarray(assembly.matrix[rows, :] @ self.trial_space.basis, dtype=float)
        if self.objective == "sampled_lspg":
            combined_weights = np.maximum(np.asarray(self.sample_set.sample_weights, dtype=float).reshape(-1), 0.0)
            if self.row_weights is not None:
                combined_weights = combined_weights * np.maximum(np.asarray(self.row_weights, dtype=float), 0.0)
            scale = np.sqrt(combined_weights)
            residual_coefficients = np.asarray(scale * sampled_residual, dtype=float).reshape(-1)
            gnat_trial_jacobian = np.asarray(sampled_trial_jacobian * scale[:, None], dtype=float)
        else:
            lift = self.sample_set.sample_to_residual_coefficients
            residual_coefficients = np.asarray(lift @ sampled_residual, dtype=float).reshape(-1)
            gnat_trial_jacobian = np.asarray(lift @ sampled_trial_jacobian, dtype=float)
        normal_matrix = gnat_trial_jacobian.T @ gnat_trial_jacobian
        normal_rhs = -(gnat_trial_jacobian.T @ residual_coefficients)
        return FluidGNATSystem(
            coefficients=coeffs.copy(),
            sampled_residual=sampled_residual,
            sampled_trial_jacobian=sampled_trial_jacobian,
            residual_coefficients=residual_coefficients,
            gnat_trial_jacobian=gnat_trial_jacobian,
            normal_matrix=np.asarray(normal_matrix, dtype=float),
            normal_rhs=np.asarray(normal_rhs, dtype=float).reshape(-1),
            estimated_residual_norm=float(np.linalg.norm(residual_coefficients)),
            row_dofs=rows.copy(),
            element_ids=np.asarray(self.sample_set.element_ids, dtype=int).copy(),
        )

    def solve(
        self,
        initial_coefficients: np.ndarray,
        *,
        max_iterations: int = 8,
        residual_tol: float = 1.0e-10,
        step_tol: float = 1.0e-12,
        line_search: bool = False,
        max_line_search: int = 6,
        sufficient_decrease: float = 1.0e-4,
    ) -> FluidLSPGResult:
        if self.online_backend == "cpp":
            if self.native_online_spec is None:
                raise RuntimeError(
                    "FluidGNATSolver online_backend='cpp' requires a ReducedFluidNativeOnlineSpec."
                )
            native = self.native_online_spec.solve(
                initial_coefficients,
                max_iterations=max_iterations,
                residual_tol=residual_tol,
                step_tol=step_tol,
                line_search=line_search,
                max_line_search=max_line_search,
                sufficient_decrease=sufficient_decrease,
            )
            return FluidLSPGResult(
                coefficients=np.asarray(native.coefficients, dtype=float).reshape(-1),
                residual_norm=float(native.residual_norm),
                iterations=int(native.iterations),
                converged=bool(native.converged),
                trajectory=tuple(native.trajectory),
            )

        coeffs = np.asarray(initial_coefficients, dtype=float).reshape(-1).copy()
        if coeffs.size != self.trial_space.n_modes:
            raise ValueError(f"Expected {self.trial_space.n_modes} coefficients, got {coeffs.size}.")
        last_norm = float("inf")
        for iteration in range(1, max(1, int(max_iterations)) + 1):
            system = self.assemble_system(coeffs)
            last_norm = float(system.estimated_residual_norm)
            if last_norm <= float(residual_tol):
                return FluidLSPGResult(coeffs, last_norm, iteration, True)
            step = system.gauss_newton_step(backend=self.gnat_step_backend)
            if bool(line_search):
                history = self.operator.snapshot_history()
                best_coeffs = coeffs + step
                best_norm = float("inf")
                accepted_coeffs = best_coeffs
                for search_iter in range(max(1, int(max_line_search))):
                    alpha = 0.5**search_iter
                    trial_coeffs = coeffs + float(alpha) * step
                    self.operator.restore_history(history)
                    trial_system = self.assemble_system(trial_coeffs)
                    trial_norm = float(trial_system.estimated_residual_norm)
                    if trial_norm < best_norm:
                        best_norm = trial_norm
                        best_coeffs = trial_coeffs
                    if trial_norm <= (1.0 - float(sufficient_decrease) * float(alpha)) * last_norm:
                        accepted_coeffs = trial_coeffs
                        break
                else:
                    accepted_coeffs = best_coeffs
                self.operator.restore_history(history)
                coeffs = np.asarray(accepted_coeffs, dtype=float).reshape(-1)
                system = self.assemble_system(coeffs)
                last_norm = float(system.estimated_residual_norm)
            else:
                coeffs = coeffs + step
            if self.nonlinear_update_hook is not None:
                self.trial_space.write(self.operator, coeffs)
                if self.state_update_hook is not None:
                    self.state_update_hook(coeffs)
                self.nonlinear_update_hook()
            if float(np.linalg.norm(step)) <= float(step_tol) * max(1.0, float(np.linalg.norm(coeffs))):
                system = self.assemble_system(coeffs)
                return FluidLSPGResult(
                    coeffs,
                    float(system.estimated_residual_norm),
                    iteration,
                    float(system.estimated_residual_norm) <= float(residual_tol),
                )
        system = self.assemble_system(coeffs)
        return FluidLSPGResult(
            coeffs,
            float(system.estimated_residual_norm),
            max(1, int(max_iterations)),
            float(system.estimated_residual_norm) <= float(residual_tol),
        )


def collect_fluid_residual_snapshots(
    operator: FluidFOMOperator,
    snapshots: FluidStageSnapshotBatch,
    *,
    row_dofs: np.ndarray | None = None,
    element_ids: np.ndarray | None = None,
    refresh_predicted: bool | None = False,
) -> np.ndarray:
    rows = np.asarray(snapshots.free_dofs if row_dofs is None else row_dofs, dtype=int).reshape(-1)
    residuals: list[np.ndarray] = []
    for idx in range(snapshots.n_snapshots):
        restore_fluid_stage(operator, snapshots.record(idx))
        assembly = operator.assemble(
            need_matrix=False,
            element_ids=element_ids,
            convention="newton",
            refresh_predicted=refresh_predicted,
        )
        residuals.append(np.asarray(assembly.residual[rows], dtype=float).reshape(-1))
    return np.column_stack(residuals)


def fit_fluid_gnat_sample_set(
    operator: FluidFOMOperator,
    residual_snapshots: np.ndarray,
    *,
    basis_dofs: np.ndarray | None = None,
    residual_modes: int | None = None,
    residual_energy: float | None = 0.999,
    row_oversampling: float = 2.0,
    min_sample_rows: int | None = None,
    forced_row_dofs: np.ndarray | None = None,
    forced_element_ids: np.ndarray | None = None,
    include_interface_elements: bool = True,
    sample_weighting: str = "none",
) -> FluidGNATSampleSet:
    """Fit a GNAT residual basis and deterministic QDEIM sample mesh."""

    total_dofs = int(operator.dh.total_dofs)
    basis_rows = operator.free_fluid_dofs() if basis_dofs is None else np.asarray(basis_dofs, dtype=int).reshape(-1)
    residual_matrix = np.asarray(residual_snapshots, dtype=float)
    if residual_matrix.ndim == 1:
        residual_matrix = residual_matrix[:, None]
    if residual_matrix.shape[0] == total_dofs:
        residual_matrix = residual_matrix[basis_rows, :]
    if residual_matrix.shape[0] != basis_rows.size:
        raise ValueError("residual_snapshots rows must match basis_dofs or the full operator DOF count.")
    residual_basis_local, svals, energy_fraction = _fit_residual_basis(
        residual_matrix,
        residual_modes=residual_modes,
        residual_energy=None if residual_modes is not None else residual_energy,
    )
    residual_basis = np.zeros((total_dofs, int(residual_basis_local.shape[1])), dtype=float)
    residual_basis[basis_rows, :] = residual_basis_local
    n_modes = int(residual_basis_local.shape[1])
    target_rows = max(n_modes, int(np.ceil(float(row_oversampling) * max(n_modes, 1))))
    if min_sample_rows is not None:
        target_rows = max(target_rows, int(min_sample_rows))
    selected = _qdeim_rows(
        residual_basis_local,
        basis_rows,
        target_count=min(target_rows, int(basis_rows.size)),
    )
    if forced_row_dofs is not None:
        forced_rows = np.intersect1d(np.asarray(forced_row_dofs, dtype=int).reshape(-1), basis_rows)
        selected = _unique_in_order(np.concatenate([selected, forced_rows]))
    forced_elements: list[np.ndarray] = []
    if forced_element_ids is not None:
        forced_elements.append(np.asarray(forced_element_ids, dtype=int).reshape(-1))
    if bool(include_interface_elements):
        forced_elements.append(
            _fluid_interface_reaction_element_ids(
                operator.prob,
                interface_tag=operator.boundary_tags.interface_tag,
            )
        )
    supporting_elements = fluid_elements_touching_rows(operator, selected)
    if forced_elements:
        supporting_elements = np.unique(np.concatenate([supporting_elements, *forced_elements])).astype(int, copy=False)
    else:
        supporting_elements = np.unique(supporting_elements).astype(int, copy=False)
    sample_basis = np.asarray(residual_basis[selected, :], dtype=float)
    sample_positions = _local_positions(basis_rows, selected)
    weighting_key = str(sample_weighting).strip().lower().replace("_", "-")
    if weighting_key in {"basis-gram", "basis", "basis-gram-bounded", "basis-bounded", "basis-gram-clipped"}:
        weight_matrix = residual_basis_local
        weight_label = "basis-gram-bounded" if ("bounded" in weighting_key or "clipped" in weighting_key) else "basis-gram"
    elif weighting_key in {
        "snapshot-gram",
        "snapshots",
        "empirical-quadrature",
        "empirical-quadrature-rows",
        "snapshot-gram-bounded",
        "snapshot-gram-clipped",
        "snapshots-bounded",
        "empirical-quadrature-bounded",
        "empirical-quadrature-rows-bounded",
    }:
        weight_matrix = residual_matrix
        weight_label = "snapshot-gram-bounded" if ("bounded" in weighting_key or "clipped" in weighting_key) else "snapshot-gram"
    elif weighting_key in {"", "none", "identity"}:
        weight_matrix = residual_basis_local
        weight_label = "none"
    else:
        raise ValueError(f"Unsupported GNAT sample_weighting={sample_weighting!r}.")
    sample_weights, weight_fit_error = _fit_nonnegative_sample_weights(
        matrix=weight_matrix,
        sample_positions=sample_positions,
        method=weight_label,
    )
    singular_values = np.linalg.svd(sample_basis, compute_uv=False)
    rank = int(np.linalg.matrix_rank(sample_basis))
    if singular_values.size == 0 or float(singular_values[-1]) == 0.0:
        condition = float("inf")
    else:
        condition = float(singular_values[0] / singular_values[-1])
    return FluidGNATSampleSet(
        residual_basis=residual_basis,
        residual_singular_values=svals,
        residual_energy_fraction=energy_fraction,
        basis_dofs=basis_rows,
        row_dofs=selected,
        element_ids=supporting_elements,
        sample_to_residual_coefficients=_weighted_sample_lift(sample_basis, sample_weights),
        sampled_basis_rank=rank,
        sampled_basis_condition=condition,
        sample_weights=sample_weights,
        sample_weighting=weight_label,
        sample_weight_fit_relative_error=weight_fit_error,
    )


__all__ = [
    "FluidGNATSampleSet",
    "FluidGNATSolver",
    "FluidGNATSystem",
    "collect_fluid_residual_snapshots",
    "fit_fluid_gnat_sample_set",
    "fluid_element_dofs",
    "fluid_elements_touching_rows",
]
