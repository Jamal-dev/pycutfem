from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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
    velocity_pod_mode_count: int | None = None
    velocity_supremizer_mode_count: int = 0
    supremizer_singular_values: np.ndarray | None = None
    supremizer_riesz: str = "none"

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
        pod_count = (
            int(self.velocity_singular_values.size) - int(self.velocity_supremizer_mode_count)
            if self.velocity_pod_mode_count is None
            else int(self.velocity_pod_mode_count)
        )
        sup_count = int(self.velocity_supremizer_mode_count)
        if pod_count < 0 or sup_count < 0 or pod_count + sup_count != int(self.velocity_singular_values.size):
            raise ValueError("Velocity POD/supremizer mode counts must sum to n_velocity_modes.")
        object.__setattr__(self, "velocity_pod_mode_count", pod_count)
        object.__setattr__(self, "velocity_supremizer_mode_count", sup_count)
        if self.supremizer_singular_values is None:
            object.__setattr__(self, "supremizer_singular_values", np.zeros(sup_count, dtype=float))
        else:
            sup_svals = np.asarray(self.supremizer_singular_values, dtype=float).reshape(-1)
            if int(sup_svals.size) != sup_count:
                raise ValueError("supremizer_singular_values size must match velocity_supremizer_mode_count.")
            object.__setattr__(self, "supremizer_singular_values", sup_svals)
        object.__setattr__(self, "supremizer_riesz", str(self.supremizer_riesz))

    @property
    def n_modes(self) -> int:
        return int(self.basis.shape[1])

    @property
    def n_velocity_modes(self) -> int:
        return int(self.velocity_singular_values.size)

    @property
    def n_pressure_modes(self) -> int:
        return int(self.pressure_singular_values.size)

    @property
    def n_velocity_pod_modes(self) -> int:
        return int(self.velocity_pod_mode_count or 0)

    @property
    def n_velocity_supremizer_modes(self) -> int:
        return int(self.velocity_supremizer_mode_count)

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


def _orthonormalize_block_candidates(
    *,
    existing_full_basis: np.ndarray,
    candidate_full_basis: np.ndarray,
    block_dofs: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(block_dofs, dtype=int).reshape(-1)
    existing = np.asarray(existing_full_basis[rows, :], dtype=float)
    candidates = np.asarray(candidate_full_basis[rows, :], dtype=float)
    accepted: list[np.ndarray] = []
    raw_norms: list[float] = []
    if existing.size:
        q_existing, _ = np.linalg.qr(existing, mode="reduced")
    else:
        q_existing = np.zeros((int(rows.size), 0), dtype=float)
    for col in range(int(candidates.shape[1])):
        vector = np.asarray(candidates[:, col], dtype=float).reshape(-1)
        raw_norm = float(np.linalg.norm(vector))
        if raw_norm <= float(tolerance):
            continue
        if q_existing.size:
            vector = vector - q_existing @ (q_existing.T @ vector)
        for prior in accepted:
            vector = vector - prior * float(np.dot(prior, vector))
        norm = float(np.linalg.norm(vector))
        if norm <= float(tolerance):
            continue
        accepted.append(vector / norm)
        raw_norms.append(raw_norm)
    if not accepted:
        return (
            np.zeros((int(existing_full_basis.shape[0]), 0), dtype=float),
            np.zeros(0, dtype=float),
        )
    full = np.zeros((int(existing_full_basis.shape[0]), len(accepted)), dtype=float)
    for idx, vector in enumerate(accepted):
        full[rows, idx] = vector
    return full, np.asarray(raw_norms, dtype=float)


def _orthonormalize_velocity_candidates(
    *,
    existing_full_basis: np.ndarray,
    candidate_full_basis: np.ndarray,
    velocity_dofs: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    return _orthonormalize_block_candidates(
        existing_full_basis=existing_full_basis,
        candidate_full_basis=candidate_full_basis,
        block_dofs=velocity_dofs,
        tolerance=float(tolerance),
    )


def enrich_fluid_pod_trial_basis_with_interface_state_modes(
    operator: FluidFOMOperator,
    basis: FluidPODTrialBasis,
    state_matrix: np.ndarray,
    *,
    velocity_modes: int = 0,
    pressure_modes: int = 0,
    center: bool = False,
    orthogonalization_tol: float = 1.0e-12,
) -> FluidPODTrialBasis:
    """Append reaction-weighted interface state modes to a fluid trial basis.

    The input matrix is a full-DOF state or increment matrix.  The enrichment is
    still homogeneous: only free velocity/pressure rows are populated and all
    constrained rows remain zero.  This preserves the existing Dirichlet lifting
    policy while giving the reduced fluid state more directions that matter for
    the interface load map.
    """

    matrix = np.asarray(state_matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("state_matrix must be a full-DOF 2D matrix.")
    if int(matrix.shape[0]) != int(operator.dh.total_dofs):
        raise ValueError("state_matrix rows must match the operator DOF count.")
    if int(basis.basis.shape[0]) != int(operator.dh.total_dofs):
        raise ValueError("Basis row count must match the operator DOF count.")

    velocity_count = max(0, int(velocity_modes))
    pressure_count = max(0, int(pressure_modes))
    if velocity_count == 0 and pressure_count == 0:
        return basis

    velocity_free = _unique_intersection(basis.velocity_dofs, operator.free_fluid_dofs(("ux", "uy")))
    pressure_free = _unique_intersection(basis.pressure_dofs, operator.free_fluid_dofs(("p",)))

    accepted_velocity = np.zeros((int(operator.dh.total_dofs), 0), dtype=float)
    velocity_norms = np.zeros(0, dtype=float)
    if velocity_count > 0 and int(velocity_free.size) > 0:
        velocity_basis, _velocity_svals, _velocity_energy, _velocity_mean = _fit_block(
            matrix[velocity_free, :],
            n_modes=velocity_count,
            energy=None,
            center=bool(center),
        )
        velocity_candidates = np.zeros((int(operator.dh.total_dofs), int(velocity_basis.shape[1])), dtype=float)
        velocity_candidates[velocity_free, :] = velocity_basis
        accepted_velocity, velocity_norms = _orthonormalize_block_candidates(
            existing_full_basis=np.asarray(basis.basis[:, : int(basis.n_velocity_modes)], dtype=float),
            candidate_full_basis=velocity_candidates,
            block_dofs=velocity_free,
            tolerance=float(orthogonalization_tol),
        )

    accepted_pressure = np.zeros((int(operator.dh.total_dofs), 0), dtype=float)
    pressure_norms = np.zeros(0, dtype=float)
    if pressure_count > 0 and int(pressure_free.size) > 0:
        pressure_basis, _pressure_svals, _pressure_energy, _pressure_mean = _fit_block(
            matrix[pressure_free, :],
            n_modes=pressure_count,
            energy=None,
            center=bool(center),
        )
        pressure_candidates = np.zeros((int(operator.dh.total_dofs), int(pressure_basis.shape[1])), dtype=float)
        pressure_candidates[pressure_free, :] = pressure_basis
        accepted_pressure, pressure_norms = _orthonormalize_block_candidates(
            existing_full_basis=np.asarray(basis.basis[:, int(basis.n_velocity_modes) :], dtype=float),
            candidate_full_basis=pressure_candidates,
            block_dofs=pressure_free,
            tolerance=float(orthogonalization_tol),
        )

    if int(accepted_velocity.shape[1]) == 0 and int(accepted_pressure.shape[1]) == 0:
        return basis

    old_velocity = np.asarray(basis.basis[:, : int(basis.n_velocity_modes)], dtype=float)
    old_pressure = np.asarray(basis.basis[:, int(basis.n_velocity_modes) :], dtype=float)
    full_basis = np.column_stack([old_velocity, accepted_velocity, old_pressure, accepted_pressure])
    velocity_svals = np.concatenate(
        [
            np.asarray(basis.velocity_singular_values, dtype=float).reshape(-1),
            np.asarray(velocity_norms, dtype=float).reshape(-1),
        ]
    )
    velocity_energy = np.concatenate(
        [
            np.asarray(basis.velocity_energy_fraction, dtype=float).reshape(-1),
            np.ones(int(accepted_velocity.shape[1]), dtype=float),
        ]
    )
    pressure_svals = np.concatenate(
        [
            np.asarray(basis.pressure_singular_values, dtype=float).reshape(-1),
            np.asarray(pressure_norms, dtype=float).reshape(-1),
        ]
    )
    pressure_energy = np.concatenate(
        [
            np.asarray(basis.pressure_energy_fraction, dtype=float).reshape(-1),
            np.ones(int(accepted_pressure.shape[1]), dtype=float),
        ]
    )
    return FluidPODTrialBasis(
        basis=full_basis,
        free_dofs=basis.free_dofs,
        velocity_dofs=basis.velocity_dofs,
        pressure_dofs=basis.pressure_dofs,
        velocity_singular_values=velocity_svals,
        pressure_singular_values=pressure_svals,
        velocity_energy_fraction=velocity_energy,
        pressure_energy_fraction=pressure_energy,
        velocity_mean=basis.velocity_mean,
        pressure_mean=basis.pressure_mean,
        velocity_pod_mode_count=int(basis.n_velocity_pod_modes) + int(accepted_velocity.shape[1]),
        velocity_supremizer_mode_count=int(basis.n_velocity_supremizer_modes),
        supremizer_singular_values=basis.supremizer_singular_values,
        supremizer_riesz=basis.supremizer_riesz,
    )


def _assemble_h1_velocity_riesz(
    operator: FluidFOMOperator,
    *,
    quadrature_order: int,
    backend: str,
    mass_scale: float,
):
    from pycutfem.ufl.expressions import Constant, dot, grad, inner
    from pycutfem.ufl.forms import Equation, assemble_form
    from pycutfem.ufl.measures import dx

    du = operator.prob["du"]
    v = operator.prob["v"]
    q_measure = dx(metadata={"q": int(quadrature_order)})
    lhs = inner(grad(du), grad(v)) * q_measure
    if float(mass_scale) != 0.0:
        lhs += Constant(float(mass_scale)) * dot(du, v) * q_measure
    matrix, _rhs = assemble_form(
        Equation(lhs, None),
        dof_handler=operator.dh,
        bcs=[],
        quad_order=int(quadrature_order),
        backend=str(backend),
    )
    return matrix.tocsr() if hasattr(matrix, "tocsr") else sp.csr_matrix(np.asarray(matrix, dtype=float))


def _assemble_supremizer_rhs_matrix(
    operator: FluidFOMOperator,
    *,
    pressure_modes: np.ndarray,
    pressure_dofs: np.ndarray,
    velocity_dofs: np.ndarray,
    quadrature_order: int,
    backend: str,
) -> np.ndarray:
    from pycutfem.ufl.expressions import Function, div
    from pycutfem.ufl.forms import Equation, assemble_form
    from pycutfem.ufl.measures import dx

    p_sup = Function("p_supremizer", "p", dof_handler=operator.dh)
    v = operator.prob["v"]
    q_measure = dx(metadata={"q": int(quadrature_order)})
    rhs_form = (-p_sup * div(v)) * q_measure
    p_rows = np.asarray(pressure_dofs, dtype=int).reshape(-1)
    v_rows = np.asarray(velocity_dofs, dtype=int).reshape(-1)
    rhs = np.zeros((int(v_rows.size), int(pressure_modes.shape[1])), dtype=float)
    for mode_index in range(int(pressure_modes.shape[1])):
        p_sup.nodal_values.fill(0.0)
        p_sup.set_nodal_values(p_rows, np.asarray(pressure_modes[p_rows, mode_index], dtype=float))
        _matrix, vector = assemble_form(
            Equation(None, rhs_form),
            dof_handler=operator.dh,
            bcs=[],
            quad_order=int(quadrature_order),
            backend=str(backend),
        )
        rhs[:, mode_index] = np.asarray(vector, dtype=float).reshape(-1)[v_rows]
    return rhs


def _algebraic_pressure_velocity_rhs_matrix(
    operator: FluidFOMOperator,
    *,
    pressure_modes: np.ndarray,
    pressure_dofs: np.ndarray,
    velocity_dofs: np.ndarray,
) -> np.ndarray:
    from dataclasses import replace

    from examples.NIRB.fluid_fom_operator import FluidFOMOperator

    params = replace(operator.parameters, contribution_mode="velocity")
    velocity_operator = FluidFOMOperator(
        prob=operator.prob,
        mesh=operator.mesh,
        parameters=params,
        boundary_tags=operator.boundary_tags,
    )
    assembly = velocity_operator.assemble(need_matrix=True, convention="newton", refresh_predicted=True)
    if assembly.matrix is None:
        raise RuntimeError("Velocity contribution assembly did not return a matrix.")
    p_rows = np.asarray(pressure_dofs, dtype=int).reshape(-1)
    v_rows = np.asarray(velocity_dofs, dtype=int).reshape(-1)
    coupling = assembly.matrix.tocsr()[np.ix_(v_rows, p_rows)]
    return np.asarray(coupling @ pressure_modes[p_rows, :], dtype=float)


def _mass_velocity_riesz(operator: FluidFOMOperator, *, velocity_dofs: np.ndarray):
    from dataclasses import replace

    from examples.NIRB.fluid_fom_operator import FluidFOMOperator

    params = replace(operator.parameters, contribution_mode="mass_lhs")
    mass_operator = FluidFOMOperator(
        prob=operator.prob,
        mesh=operator.mesh,
        parameters=params,
        boundary_tags=operator.boundary_tags,
    )
    assembly = mass_operator.assemble(need_matrix=True, convention="newton", refresh_predicted=False)
    if assembly.matrix is None:
        raise RuntimeError("Mass contribution assembly did not return a matrix.")
    rows = np.asarray(velocity_dofs, dtype=int).reshape(-1)
    return assembly.matrix.tocsr()[np.ix_(rows, rows)]


def enrich_fluid_pod_trial_basis_with_supremizers(
    operator: FluidFOMOperator,
    basis: FluidPODTrialBasis,
    *,
    supremizer_modes: int | None = None,
    riesz: str = "h1",
    quadrature_order: int | None = None,
    backend: str | None = None,
    h1_mass_scale: float = 1.0,
    regularization: float = 1.0e-12,
    orthogonalization_tol: float = 1.0e-12,
) -> FluidPODTrialBasis:
    """Append homogeneous velocity supremizers generated from pressure modes.

    The returned basis keeps the same lifting policy as the input basis: all
    constrained rows remain zero, so online states are still
    ``Dirichlet_lift + V a``.
    """

    if int(basis.basis.shape[0]) != int(operator.dh.total_dofs):
        raise ValueError("Basis row count must match the operator DOF count.")
    n_pressure = int(basis.n_pressure_modes)
    if n_pressure == 0:
        return basis
    n_requested = n_pressure if supremizer_modes is None else max(0, int(supremizer_modes))
    if n_requested == 0:
        return basis
    n_requested = min(n_requested, n_pressure)
    velocity_free = _unique_intersection(basis.velocity_dofs, operator.free_fluid_dofs(("ux", "uy")))
    pressure_free = _unique_intersection(basis.pressure_dofs, operator.free_fluid_dofs(("p",)))
    if int(velocity_free.size) == 0 or int(pressure_free.size) == 0:
        raise ValueError("Supremizer enrichment requires free velocity and pressure DOFs.")

    pressure_start = int(basis.n_velocity_modes)
    pressure_columns = np.arange(pressure_start, pressure_start + n_requested, dtype=int)
    pressure_modes_full = np.asarray(basis.basis[:, pressure_columns], dtype=float)
    mode = str(riesz).strip().lower().replace("-", "_")
    if mode in {"h1", "h1_reference", "reference_h1"}:
        quad_order = int(operator.parameters.quadrature_order if quadrature_order is None else quadrature_order)
        backend_name = str(operator.parameters.backend if backend is None else backend)
        riesz_matrix = _assemble_h1_velocity_riesz(
            operator,
            quadrature_order=quad_order,
            backend=backend_name,
            mass_scale=float(h1_mass_scale),
        )[np.ix_(velocity_free, velocity_free)]
        rhs_matrix = _assemble_supremizer_rhs_matrix(
            operator,
            pressure_modes=pressure_modes_full,
            pressure_dofs=pressure_free,
            velocity_dofs=velocity_free,
            quadrature_order=quad_order,
            backend=backend_name,
        )
        mode_name = "h1"
    elif mode in {"mass", "mass_lhs", "l2"}:
        riesz_matrix = _mass_velocity_riesz(operator, velocity_dofs=velocity_free)
        rhs_matrix = _algebraic_pressure_velocity_rhs_matrix(
            operator,
            pressure_modes=pressure_modes_full,
            pressure_dofs=pressure_free,
            velocity_dofs=velocity_free,
        )
        mode_name = "mass_lhs"
    else:
        raise ValueError(f"Unsupported supremizer Riesz mode {riesz!r}.")

    riesz_csr = riesz_matrix.tocsr() if hasattr(riesz_matrix, "tocsr") else sp.csr_matrix(riesz_matrix)
    if float(regularization) > 0.0:
        diagonal_scale = float(np.mean(np.abs(riesz_csr.diagonal()))) if riesz_csr.shape[0] else 1.0
        diagonal_scale = max(diagonal_scale, 1.0)
        riesz_csr = riesz_csr + (float(regularization) * diagonal_scale) * sp.eye(
            int(riesz_csr.shape[0]),
            format="csr",
        )
    try:
        solutions = spla.spsolve(riesz_csr.tocsc(), rhs_matrix)
    except Exception:
        solutions = spla.lsmr(riesz_csr, rhs_matrix[:, 0])[0][:, None]
        for col in range(1, int(rhs_matrix.shape[1])):
            solutions = np.column_stack([solutions, spla.lsmr(riesz_csr, rhs_matrix[:, col])[0]])
    solutions = np.asarray(solutions, dtype=float)
    if solutions.ndim == 1:
        solutions = solutions[:, None]
    if not np.all(np.isfinite(solutions)):
        raise RuntimeError("Supremizer solve produced non-finite values.")

    candidate_full = np.zeros((int(basis.basis.shape[0]), int(solutions.shape[1])), dtype=float)
    candidate_full[velocity_free, :] = solutions
    velocity_columns = np.arange(int(basis.n_velocity_modes), dtype=int)
    accepted_full, supremizer_norms = _orthonormalize_velocity_candidates(
        existing_full_basis=basis.basis[:, velocity_columns],
        candidate_full_basis=candidate_full,
        velocity_dofs=velocity_free,
        tolerance=float(orthogonalization_tol),
    )
    if int(accepted_full.shape[1]) == 0:
        raise RuntimeError("Supremizer enrichment produced no independent velocity modes.")

    old_velocity = np.asarray(basis.basis[:, : int(basis.n_velocity_modes)], dtype=float)
    old_pressure = np.asarray(basis.basis[:, int(basis.n_velocity_modes) :], dtype=float)
    full_basis = np.column_stack([old_velocity, accepted_full, old_pressure])
    velocity_svals = np.concatenate(
        [
            np.asarray(basis.velocity_singular_values, dtype=float).reshape(-1),
            np.asarray(supremizer_norms, dtype=float).reshape(-1),
        ]
    )
    velocity_energy = np.concatenate(
        [
            np.asarray(basis.velocity_energy_fraction, dtype=float).reshape(-1),
            np.ones(int(accepted_full.shape[1]), dtype=float),
        ]
    )
    return FluidPODTrialBasis(
        basis=full_basis,
        free_dofs=basis.free_dofs,
        velocity_dofs=basis.velocity_dofs,
        pressure_dofs=basis.pressure_dofs,
        velocity_singular_values=velocity_svals,
        pressure_singular_values=basis.pressure_singular_values,
        velocity_energy_fraction=velocity_energy,
        pressure_energy_fraction=basis.pressure_energy_fraction,
        velocity_mean=basis.velocity_mean,
        pressure_mean=basis.pressure_mean,
        velocity_pod_mode_count=int(basis.n_velocity_modes),
        velocity_supremizer_mode_count=int(accepted_full.shape[1]),
        supremizer_singular_values=supremizer_norms,
        supremizer_riesz=mode_name,
    )


__all__ = [
    "FluidPODTrialBasis",
    "enrich_fluid_pod_trial_basis_with_interface_state_modes",
    "enrich_fluid_pod_trial_basis_with_supremizers",
    "fit_fluid_pod_trial_basis",
    "fit_fluid_pod_trial_basis_from_state_matrix",
]
