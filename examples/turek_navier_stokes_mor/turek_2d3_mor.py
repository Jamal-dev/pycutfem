#!/usr/bin/env python
"""Native-online MOR example for the Turek 2D-3 cylinder benchmark.

The script is split into explicit stages:

1. collect full-order snapshots;
2. train a lifted, supremizer-enriched, non-affine ROM;
3. run one native C++ online Gauss-Newton solve with generated UFL kernels.

The smoke command exercises the offline algebra without running the full
benchmark, which keeps tests cheap while preserving the production code path for
real runs.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from examples.turek_benchmark_volume_only import (
    CENTER,
    D,
    FE_ORDER,
    H,
    L,
    MU,
    RADIUS,
    RHO,
    U_MAX,
    build_structured_channel_mesh,
    check_positive_jacobians,
    prepare_mesh,
)
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.mor import (
    MixedBasisBlock,
    build_block_row_weights,
    build_dirichlet_lifting_vector,
    build_mixed_field_basis,
    build_nonaffine_reduced_decomposition,
    field_dof_indices,
    fit_pod,
    fit_lift_enriched_basis,
    native_kernel_metadata_from_runner,
    remove_lifting_from_snapshots,
    solve_coupled_lift_snapshots,
    solve_native_deim_online_gauss_newton,
)
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    Identity,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    div,
    dot,
    grad,
    inner,
    trace,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace


VELOCITY_FIELDS = ("ux", "uy")
PRESSURE_FIELD = "p"


@dataclass
class TurekMORConfig:
    """Configuration for the Turek 2D-3 MOR workflow."""

    mesh_backend: str = "structured"
    mesh_type: str = "quad"
    mesh_size: float = 0.04
    mesh_file: str | None = None
    rebuild_mesh: bool = False
    refine_hmin: float | None = None
    refine_band_radius: float | None = 0.2
    backend: str = "cpp"
    inflow: str = "dfg"
    dt: float = 0.05
    theta: float = 0.5
    max_steps: int = 8
    velocity_modes: int = 8
    pressure_modes: int = 4
    supremizer_modes: int = 4
    deim_modes: int = 32
    select_modes: bool = False
    velocity_mode_candidates: tuple[int, ...] = (2, 4, 6, 8)
    pressure_mode_candidates: tuple[int, ...] = (1, 2, 3, 4)
    supremizer_mode_candidates: tuple[int, ...] = (1, 2, 3, 4)
    deim_mode_candidates: tuple[int, ...] = (16, 24, 32, 48)
    cv_validation_fraction: float = 0.25
    residual_block_weighting: bool = True
    continuity_row_weight: float = 16.0
    supremizer_regularization: float = 1.0e-10
    center_snapshots: bool = False
    convection_form: str = "standard"
    residual_tol: float = 1.0e-9
    max_online_iterations: int = 10


@dataclass
class TurekFOMProblem:
    mesh: Any
    mixed_element: MixedElement
    dof_handler: DofHandler
    bcs: list[BoundaryCondition]
    bcs_homogeneous: list[BoundaryCondition]
    functions: list[Any]
    previous_functions: list[Any]
    residual_form: Any
    jacobian_form: Any
    constants: dict[str, Constant]
    config: TurekMORConfig

    @property
    def u_k(self) -> VectorFunction:
        return self.functions[0]

    @property
    def p_k(self) -> Function:
        return self.functions[1]

    @property
    def u_n(self) -> VectorFunction:
        return self.previous_functions[0]

    @property
    def p_n(self) -> Function:
        return self.previous_functions[1]


@dataclass
class TurekNativeOnlineRun:
    result: Any
    state: np.ndarray
    offset: np.ndarray
    wall_time_s: float


@dataclass
class TurekROMValidation:
    snapshot_index: int
    reference_norm: float
    online_relative_error: float
    velocity_relative_error: float
    pressure_relative_error: float
    pressure_shifted_relative_error: float
    projection_relative_error: float
    native_online_wall_time_s: float
    fom_mean_step_time_s: float | None
    speedup_vs_mean_fom_step: float | None
    converged: bool
    iterations: int
    residual_norm: float


def epsilon_2d(u):
    """Symmetric gradient."""

    return Constant(0.5) * (grad(u) + grad(u).T)


def deviatoric_strain_2d(u):
    """2D deviatoric strain: eps(u) - 0.5 tr(eps(u)) I."""

    eps_u = epsilon_2d(u)
    return eps_u - Constant(0.5) * trace(eps_u) * Identity(2)


def cauchy_stress_2d(u, p, mu):
    """Fluid stress sigma_f = -p I + 2 mu dev(eps(u)) in 2D."""

    return -p * Identity(2) + Constant(2.0) * mu * deviatoric_strain_2d(u)


def deviatoric_strain_matrix_2d(eps_matrix: np.ndarray) -> np.ndarray:
    """Numeric companion used by tests to lock the 2D 1/2 convention."""

    E = np.asarray(eps_matrix, dtype=float)
    if E.shape != (2, 2):
        raise ValueError("eps_matrix must have shape (2, 2).")
    return E - 0.5 * float(np.trace(E)) * np.eye(2)


def _advect_standard(w, u, v):
    return dot(dot(grad(u), w), v)


def _advect_skew(w, u, v):
    return Constant(0.5) * (dot(dot(grad(u), w), v) - dot(dot(grad(v), w), u))


def _linearized_convection(u_k, du, v, form_name: str):
    if form_name == "standard":
        return dot(dot(grad(u_k), du), v) + dot(dot(grad(du), u_k), v)
    if form_name == "skew":
        return Constant(0.5) * (
            dot(dot(grad(u_k), du), v)
            + dot(dot(grad(du), u_k), v)
            - dot(dot(grad(v), du), u_k)
            - dot(dot(grad(v), u_k), du)
        )
    raise ValueError("convection_form must be 'standard' or 'skew'.")


def build_navier_stokes_forms_2d(
    *,
    du,
    v,
    dp,
    q,
    u_k,
    u_n,
    p_k,
    rho,
    mu,
    dt,
    theta,
    dx_measure,
    convection_form: str,
):
    """Build residual and tangent with the 2D deviatoric stress convention."""

    convection_name = str(convection_form).lower()
    conv_lin = _linearized_convection(u_k, du, v, convection_name)
    conv_k = _advect_standard(u_k, u_k, v) if convection_name == "standard" else _advect_skew(u_k, u_k, v)
    conv_n = _advect_standard(u_n, u_n, v) if convection_name == "standard" else _advect_skew(u_n, u_n, v)

    jacobian = (
        rho * dot(du, v) / dt
        + theta * rho * conv_lin
        + theta * inner(cauchy_stress_2d(du, dp, mu), grad(v))
        + q * div(du)
    ) * dx_measure

    residual = (
        rho * dot(u_k - u_n, v) / dt
        + theta * rho * conv_k
        + (Constant(1.0) - theta) * rho * conv_n
        + theta * inner(cauchy_stress_2d(u_k, p_k, mu), grad(v))
        + (Constant(1.0) - theta)
        * inner(Constant(2.0) * mu * deviatoric_strain_2d(u_n), grad(v))
        + q * div(u_k)
    ) * dx_measure

    return jacobian, residual


def parabolic_inflow_value(y: float, t: float, config: TurekMORConfig) -> float:
    amp = 1.0
    if str(config.inflow).lower() == "dfg":
        amp = 0.2 + 0.8 * np.sin(np.pi * float(t) / 8.0)
    return float(4.0 * U_MAX * amp * y * (H - y) / (H * H))


def build_boundary_conditions(config: TurekMORConfig) -> list[BoundaryCondition]:
    def inlet_u(x, y, t=0.0):
        return parabolic_inflow_value(float(y), float(t), config)

    zero = lambda x, y: 0.0
    return [
        BoundaryCondition("ux", "dirichlet", "inlet", inlet_u),
        BoundaryCondition("uy", "dirichlet", "inlet", zero),
        BoundaryCondition("ux", "dirichlet", "walls", zero),
        BoundaryCondition("uy", "dirichlet", "walls", zero),
        BoundaryCondition("ux", "dirichlet", "cylinder", zero),
        BoundaryCondition("uy", "dirichlet", "cylinder", zero),
    ]


def load_or_build_turek_mesh(config: TurekMORConfig):
    if str(config.mesh_backend).lower() == "structured":
        mesh = build_structured_channel_mesh(float(config.mesh_size), poly_order=2)
        ok, min_det, failure = check_positive_jacobians(mesh)
        if not ok:
            raise RuntimeError(f"Structured Turek mesh has non-positive Jacobian: {failure}")
        print(f"structured Turek mesh: min det(J) = {min_det:.6e}")
        return mesh

    mesh_file = None if config.mesh_file is None else Path(config.mesh_file)
    mesh, persistent = prepare_mesh(
        mesh_file,
        float(config.mesh_size),
        bool(config.rebuild_mesh),
        str(config.mesh_type),
        False,
        refine_hmin=config.refine_hmin,
        refine_band_radius=config.refine_band_radius,
    )
    if persistent is not None:
        print(f"Turek mesh: {persistent}")
    return mesh


def build_fom_problem(config: TurekMORConfig) -> TurekFOMProblem:
    mesh = load_or_build_turek_mesh(config)
    mixed_element = MixedElement(mesh, field_specs={"ux": FE_ORDER, "uy": FE_ORDER, "p": FE_ORDER - 1})
    dh = DofHandler(mixed_element, method="cg")

    velocity_space = FunctionSpace(name="velocity", field_names=list(VELOCITY_FIELDS), dim=1, side="+")
    du = VectorTrialFunction(space=velocity_space, dof_handler=dh, side="+")
    v = VectorTestFunction(space=velocity_space, dof_handler=dh, side="+")
    dp = TrialFunction(name="trial_pressure", field_name=PRESSURE_FIELD, dof_handler=dh, side="+")
    q = TestFunction(name="test_pressure", field_name=PRESSURE_FIELD, dof_handler=dh, side="+")

    u_k = VectorFunction(name="u_k", field_names=list(VELOCITY_FIELDS), dof_handler=dh, side="+")
    u_n = VectorFunction(name="u_n", field_names=list(VELOCITY_FIELDS), dof_handler=dh, side="+")
    p_k = Function(name="p_k", field_name=PRESSURE_FIELD, dof_handler=dh, side="+")
    p_n = Function(name="p_n", field_name=PRESSURE_FIELD, dof_handler=dh, side="+")

    bcs = build_boundary_conditions(config)
    bcs_homogeneous = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]
    bcs_t0 = NewtonSolver._freeze_bcs(bcs, 0.0)
    dh.apply_bcs(bcs_t0, u_k, p_k)
    dh.apply_bcs(bcs_t0, u_n, p_n)

    constants = {
        "rho": Constant(RHO),
        "mu": Constant(MU),
        "dt": Constant(float(config.dt)),
        "theta": Constant(float(config.theta)),
    }
    dx_measure = dx(metadata={"q": 2 * FE_ORDER + 2})
    jacobian_form, residual_form = build_navier_stokes_forms_2d(
        du=du,
        v=v,
        dp=dp,
        q=q,
        u_k=u_k,
        u_n=u_n,
        p_k=p_k,
        rho=constants["rho"],
        mu=constants["mu"],
        dt=constants["dt"],
        theta=constants["theta"],
        dx_measure=dx_measure,
        convection_form=config.convection_form,
    )

    return TurekFOMProblem(
        mesh=mesh,
        mixed_element=mixed_element,
        dof_handler=dh,
        bcs=bcs,
        bcs_homogeneous=bcs_homogeneous,
        functions=[u_k, p_k],
        previous_functions=[u_n, p_n],
        residual_form=residual_form,
        jacobian_form=jacobian_form,
        constants=constants,
        config=config,
    )


def pack_mixed_state(problem: TurekFOMProblem) -> np.ndarray:
    dh = problem.dof_handler
    out = np.zeros(int(dh.total_dofs), dtype=float)
    for i, field in enumerate(VELOCITY_FIELDS):
        out[np.asarray(dh.get_field_slice(field), dtype=int)] = problem.u_k.nodal_values_component(i)
    out[np.asarray(dh.get_field_slice(PRESSURE_FIELD), dtype=int)] = problem.p_k.nodal_values
    return out


def initial_mixed_state(problem: TurekFOMProblem) -> np.ndarray:
    dh = problem.dof_handler
    out = np.zeros(int(dh.total_dofs), dtype=float)
    for i, field in enumerate(VELOCITY_FIELDS):
        out[np.asarray(dh.get_field_slice(field), dtype=int)] = problem.u_n.nodal_values_component(i)
    out[np.asarray(dh.get_field_slice(PRESSURE_FIELD), dtype=int)] = problem.p_n.nodal_values
    return out


def set_mixed_state(problem: TurekFOMProblem, state: np.ndarray, *, previous: bool = False) -> None:
    dh = problem.dof_handler
    values = np.asarray(state, dtype=float).reshape(-1)
    if values.size != int(dh.total_dofs):
        raise ValueError("state size does not match the problem DOF count.")
    velocity = problem.u_n if previous else problem.u_k
    pressure = problem.p_n if previous else problem.p_k
    for i, field in enumerate(VELOCITY_FIELDS):
        velocity.set_component_values(i, values[np.asarray(dh.get_field_slice(field), dtype=int)])
    pressure.nodal_values[:] = values[np.asarray(dh.get_field_slice(PRESSURE_FIELD), dtype=int)]


def collect_full_order_snapshots(config: TurekMORConfig, output_dir: Path) -> Path:
    """Run the FOM and store lifted snapshots for offline training."""

    problem = build_fom_problem(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    states: list[np.ndarray] = []
    lifts: list[np.ndarray] = []
    times: list[float] = []
    bc_times: list[float] = []
    end_times: list[float] = []

    solver = NewtonSolver(
        problem.residual_form,
        problem.jacobian_form,
        dof_handler=problem.dof_handler,
        mixed_element=problem.mixed_element,
        bcs=problem.bcs,
        bcs_homog=problem.bcs_homogeneous,
        newton_params=NewtonParameters(newton_tol=1.0e-7, line_search=True),
        backend=config.backend,
    )
    time_params = TimeStepperParameters(dt=float(config.dt), max_steps=int(config.max_steps), theta=float(config.theta))

    def record_step(step: int, bcs_now, funs, prev_funs):
        del funs, prev_funs
        states.append(pack_mixed_state(problem))
        lifts.append(build_dirichlet_lifting_vector(problem.dof_handler, bcs_now))
        start_time = float(step) * float(config.dt)
        times.append(start_time)
        bc_times.append(start_time + float(config.theta) * float(config.dt))
        end_times.append(start_time + float(config.dt))

    t0 = time.perf_counter()
    solver.solve_time_interval(
        functions=problem.functions,
        prev_functions=problem.previous_functions,
        time_params=time_params,
        post_step_refiner=record_step,
    )
    elapsed = time.perf_counter() - t0

    snapshot_file = output_dir / "turek_2d3_snapshots.npz"
    np.savez_compressed(
        snapshot_file,
        state_snapshots=np.column_stack(states) if states else np.zeros((int(problem.dof_handler.total_dofs), 0)),
        lifting_snapshots=np.column_stack(lifts) if lifts else np.zeros((int(problem.dof_handler.total_dofs), 0)),
        times=np.asarray(times, dtype=float),
        bc_times=np.asarray(bc_times, dtype=float),
        end_times=np.asarray(end_times, dtype=float),
        metadata=json.dumps(
            {
                "config": asdict(config),
                "geometry": {"L": L, "H": H, "D": D, "center": CENTER, "radius": RADIUS},
                "fom_wall_time_s": float(elapsed),
                "dofs": int(problem.dof_handler.total_dofs),
            }
        ),
    )
    print(f"wrote snapshots: {snapshot_file}")
    print(f"FOM snapshot run wall time: {elapsed:.6f} s")
    return snapshot_file


def assemble_supremizer_operators(problem: TurekFOMProblem):
    """Assemble velocity operator A_u and divergence coupling B."""

    dh = problem.dof_handler
    velocity_space = FunctionSpace(name="velocity_sup", field_names=list(VELOCITY_FIELDS), dim=1, side="+")
    du = VectorTrialFunction(space=velocity_space, dof_handler=dh, side="+")
    v = VectorTestFunction(space=velocity_space, dof_handler=dh, side="+")
    q = TestFunction(name="pressure_sup", field_name=PRESSURE_FIELD, dof_handler=dh, side="+")
    dx_measure = dx(metadata={"q": 2 * FE_ORDER + 2})

    mass_reg = Constant(float(problem.config.supremizer_regularization))
    a_u = (inner(deviatoric_strain_2d(du), deviatoric_strain_2d(v)) + mass_reg * dot(du, v)) * dx_measure
    b_form = q * div(du) * dx_measure
    K_u, _ = assemble_form(Equation(a_u, None), dof_handler=dh, bcs=[], backend=problem.config.backend)
    K_b, _ = assemble_form(Equation(b_form, None), dof_handler=dh, bcs=[], backend=problem.config.backend)

    velocity_rows = field_dof_indices(dh, VELOCITY_FIELDS)
    pressure_rows = field_dof_indices(dh, (PRESSURE_FIELD,))
    A_u = K_u[np.ix_(velocity_rows, velocity_rows)]
    B = K_b[np.ix_(pressure_rows, velocity_rows)]
    return A_u, B


def assemble_convective_residual_snapshots(
    problem: TurekFOMProblem,
    state_snapshots: np.ndarray,
) -> np.ndarray:
    """Assemble the non-affine convective residual snapshots offline."""

    dh = problem.dof_handler
    velocity_space = FunctionSpace(name="velocity_conv", field_names=list(VELOCITY_FIELDS), dim=1, side="+")
    v = VectorTestFunction(space=velocity_space, dof_handler=dh, side="+")
    dx_measure = dx(metadata={"q": 2 * FE_ORDER + 2})
    conv = Constant(RHO) * _advect_standard(problem.u_k, problem.u_k, v) * dx_measure

    residuals: list[np.ndarray] = []
    for col in range(state_snapshots.shape[1]):
        set_mixed_state(problem, state_snapshots[:, col])
        _, rhs = assemble_form(
            Equation(None, conv),
            dof_handler=dh,
            bcs=[],
            backend=problem.config.backend,
            need_matrix=False,
        )
        residuals.append(np.asarray(rhs, dtype=float).reshape(-1))
    return np.column_stack(residuals)


def assemble_full_residual_snapshots(
    problem: TurekFOMProblem,
    state_snapshots: np.ndarray,
) -> np.ndarray:
    """Assemble full residual snapshots for the native QDEIM online target."""

    residuals: list[np.ndarray] = []
    for col in range(state_snapshots.shape[1]):
        set_mixed_state(problem, state_snapshots[:, col], previous=False)
        prev_col = max(0, col - 1)
        set_mixed_state(problem, state_snapshots[:, prev_col], previous=True)
        _, rhs = assemble_form(
            Equation(None, problem.residual_form),
            dof_handler=problem.dof_handler,
            bcs=[],
            backend=problem.config.backend,
            need_matrix=False,
        )
        residuals.append(np.asarray(rhs, dtype=float).reshape(-1))
    return np.column_stack(residuals)


def assemble_reduced_residual_training_snapshots(
    problem: TurekFOMProblem,
    state_snapshots: np.ndarray,
    lifting_snapshots: np.ndarray,
    homogeneous_offset: np.ndarray,
    trial_basis: np.ndarray,
) -> np.ndarray:
    """Assemble residual snapshots on reduced states near the FOM trajectory.

    Converged FOM residuals are nearly zero and are not enough to train a useful
    QDEIM residual basis.  This routine samples the residual at the projected
    trajectory, at the lifted offset, and at small positive/negative perturbations
    along each retained reduced mode.
    """

    states = np.asarray(state_snapshots, dtype=float)
    lifts = np.asarray(lifting_snapshots, dtype=float)
    offset0 = np.asarray(homogeneous_offset, dtype=float).reshape(-1)
    basis = np.asarray(trial_basis, dtype=float)
    if lifts.shape != states.shape:
        raise ValueError("lifting_snapshots must have the same shape as state_snapshots.")
    if basis.shape[0] != states.shape[0] or offset0.size != states.shape[0]:
        raise ValueError("trial basis/offset are incompatible with state snapshots.")

    offsets = offset0[:, None] + lifts
    q_refs = np.zeros((basis.shape[1], states.shape[1]), dtype=float)
    for col in range(states.shape[1]):
        q_refs[:, col], *_ = np.linalg.lstsq(basis, states[:, col] - offsets[:, col], rcond=None)
    coeff_scale = np.maximum(np.std(q_refs, axis=1), 0.1 * np.maximum(np.max(np.abs(q_refs), axis=1), 1.0e-8))
    coeff_scale = np.maximum(coeff_scale, 1.0e-4)

    residuals: list[np.ndarray] = []
    for col in range(states.shape[1]):
        previous = initial_mixed_state(problem) if col == 0 else states[:, col - 1]
        set_mixed_state(problem, previous, previous=True)

        q_candidates: list[np.ndarray] = [
            np.zeros(basis.shape[1], dtype=float),
            0.5 * q_refs[:, col],
            q_refs[:, col].copy(),
        ]
        for mode in range(basis.shape[1]):
            dq = np.zeros(basis.shape[1], dtype=float)
            dq[mode] = 0.25 * coeff_scale[mode]
            q_candidates.append(q_refs[:, col] + dq)
            q_candidates.append(q_refs[:, col] - dq)

        for q_red in q_candidates:
            candidate = offsets[:, col] + basis @ q_red
            set_mixed_state(problem, candidate, previous=False)
            _, rhs = assemble_form(
                Equation(None, problem.residual_form),
                dof_handler=problem.dof_handler,
                bcs=[],
                backend=problem.config.backend,
                need_matrix=False,
            )
            residuals.append(np.asarray(rhs, dtype=float).reshape(-1))
    return np.column_stack(residuals)


def _parse_mode_candidates(raw: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        values = list(raw)
    parsed = sorted({int(value) for value in values if int(value) > 0})
    if not parsed:
        raise ValueError("mode candidate lists must contain at least one positive integer.")
    return tuple(parsed)


def _projection_cv_mode_selection(
    snapshots: np.ndarray,
    candidates: Sequence[int],
    *,
    center: bool,
    validation_fraction: float,
    min_modes: int = 1,
    label: str,
) -> tuple[int, dict[str, Any]]:
    matrix = np.asarray(snapshots, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(f"{label} snapshots must be a nonempty feature-major matrix.")
    n_snapshots = matrix.shape[1]
    if n_snapshots >= 3:
        validation_count = max(1, min(n_snapshots - 1, int(np.ceil(float(validation_fraction) * n_snapshots))))
        train = matrix[:, :-validation_count]
        validation = matrix[:, -validation_count:]
        validation_indices = tuple(range(n_snapshots - validation_count, n_snapshots))
    else:
        train = matrix
        validation = matrix
        validation_indices = tuple(range(n_snapshots))

    max_rank = max(1, min(train.shape))
    requested = _parse_mode_candidates(tuple(int(v) for v in candidates))
    evaluated_modes = sorted({max(int(min_modes), min(int(value), max_rank)) for value in requested})
    entries: list[dict[str, float | int]] = []
    for modes in evaluated_modes:
        pod = fit_pod(train, n_modes=int(modes), center=bool(center))
        centered = validation if pod.mean is None else validation - pod.mean
        reconstruction = pod.basis @ (pod.basis.T @ centered)
        if pod.mean is not None:
            reconstruction = reconstruction + pod.mean
        denom = float(np.linalg.norm(validation))
        if denom <= 1.0e-30:
            denom = 1.0
        error = float(np.linalg.norm(validation - reconstruction) / denom)
        entries.append({"modes": int(pod.n_modes), "relative_projection_error": error})

    best_error = min(float(entry["relative_projection_error"]) for entry in entries)
    tolerance = best_error * 1.02 + 1.0e-8
    selected = min(int(entry["modes"]) for entry in entries if float(entry["relative_projection_error"]) <= tolerance)
    return selected, {
        "label": label,
        "selected_modes": int(selected),
        "validation_indices": validation_indices,
        "evaluated": entries,
        "selection_rule": "smallest modes within 2 percent plus 1e-8 absolute tolerance of the best validation projection error",
    }


def train_turek_rom(snapshot_file: Path, output_dir: Path, config: TurekMORConfig) -> Path:
    """Train lifted POD, supremizers, and QDEIM non-affine data."""

    problem = build_fom_problem(config)
    data = np.load(snapshot_file, allow_pickle=True)
    state_snapshots = np.asarray(data["state_snapshots"], dtype=float)
    if "lifting_snapshots" in data.files:
        lifting_snapshots = np.asarray(data["lifting_snapshots"], dtype=float)
    else:
        lifting_snapshots = np.zeros_like(state_snapshots)
    if state_snapshots.shape[0] != int(problem.dof_handler.total_dofs):
        raise ValueError("snapshot DOF count does not match the configured Turek problem.")

    homogeneous = remove_lifting_from_snapshots(state_snapshots, lifting_snapshots)
    velocity_rows = field_dof_indices(problem.dof_handler, VELOCITY_FIELDS)
    pressure_rows = field_dof_indices(problem.dof_handler, (PRESSURE_FIELD,))
    velocity_snapshots = homogeneous[velocity_rows, :]
    pressure_snapshots = homogeneous[pressure_rows, :]

    mode_selection: dict[str, Any] = {}
    velocity_modes = int(config.velocity_modes)
    pressure_modes = int(config.pressure_modes)
    supremizer_modes = int(config.supremizer_modes)
    deim_modes_requested = int(config.deim_modes)
    if bool(config.select_modes):
        velocity_modes, mode_selection["velocity"] = _projection_cv_mode_selection(
            velocity_snapshots,
            config.velocity_mode_candidates,
            center=bool(config.center_snapshots),
            validation_fraction=float(config.cv_validation_fraction),
            label="velocity",
        )
        pressure_modes, mode_selection["pressure"] = _projection_cv_mode_selection(
            pressure_snapshots,
            config.pressure_mode_candidates,
            center=bool(config.center_snapshots),
            validation_fraction=float(config.cv_validation_fraction),
            label="pressure",
        )
        print(
            "cross-validation selected POD modes: "
            f"velocity={velocity_modes}, pressure={pressure_modes}"
        )

    velocity_pod = fit_pod(
        velocity_snapshots,
        n_modes=min(velocity_modes, min(velocity_snapshots.shape)),
        center=bool(config.center_snapshots),
    )
    pressure_pod = fit_pod(
        pressure_snapshots,
        n_modes=min(pressure_modes, min(pressure_snapshots.shape)),
        center=bool(config.center_snapshots),
    )

    fixed_rows = np.asarray(
        sorted(problem.dof_handler.get_dirichlet_data(problem.bcs_homogeneous).keys()),
        dtype=np.int64,
    )
    fixed_velocity = np.intersect1d(velocity_rows, fixed_rows, assume_unique=False)
    free_velocity_mask = ~np.isin(velocity_rows, fixed_velocity)

    A_u, B = assemble_supremizer_operators(problem)
    A_u_free = A_u[free_velocity_mask, :][:, free_velocity_mask]
    B_free = B[:, free_velocity_mask]
    supremizer_free = solve_coupled_lift_snapshots(
        A_u_free,
        B_free,
        pressure_pod.basis,
        regularization=float(config.supremizer_regularization),
    )
    supremizer_snapshots = np.zeros((velocity_rows.size, supremizer_free.shape[1]), dtype=float)
    supremizer_snapshots[free_velocity_mask, :] = supremizer_free
    if bool(config.select_modes):
        supremizer_modes, mode_selection["supremizer"] = _projection_cv_mode_selection(
            supremizer_snapshots,
            config.supremizer_mode_candidates,
            center=False,
            validation_fraction=float(config.cv_validation_fraction),
            label="supremizer",
        )
        print(f"cross-validation selected lift modes: supremizer={supremizer_modes}")

    enriched = fit_lift_enriched_basis(
        velocity_pod.basis,
        pressure_pod.basis,
        supremizer_snapshots,
        n_lift_modes=min(supremizer_modes, supremizer_snapshots.shape[1]),
    )

    trial_basis = build_mixed_field_basis(
        total_dofs=int(problem.dof_handler.total_dofs),
        field_blocks=(
            MixedBasisBlock(rows=velocity_rows, basis=enriched.enriched_primary_basis, name="velocity"),
            MixedBasisBlock(rows=pressure_rows, basis=pressure_pod.basis, name="pressure"),
        ),
    )

    homogeneous_offset = np.zeros(int(problem.dof_handler.total_dofs), dtype=float)
    if velocity_pod.mean is not None:
        homogeneous_offset[velocity_rows] = velocity_pod.mean.reshape(-1)
    if pressure_pod.mean is not None:
        homogeneous_offset[pressure_rows] = pressure_pod.mean.reshape(-1)

    if "nonaffine_residual_snapshots" in data:
        residual_snapshots = np.asarray(data["nonaffine_residual_snapshots"], dtype=float)
    else:
        residual_snapshots = assemble_reduced_residual_training_snapshots(
            problem,
            state_snapshots,
            lifting_snapshots,
            homogeneous_offset,
            trial_basis,
        )
    if fixed_rows.size:
        residual_snapshots[fixed_rows, :] = 0.0

    residual_row_weights = None
    if bool(config.residual_block_weighting):
        residual_row_weights = build_block_row_weights(
            residual_snapshots,
            (
                {"rows": velocity_rows, "name": "momentum"},
                {"rows": pressure_rows, "name": "continuity"},
            ),
        )
        residual_row_weights[pressure_rows] *= float(config.continuity_row_weight)
        if fixed_rows.size:
            residual_row_weights[fixed_rows] = 1.0
        print(
            "residual row weighting: "
            f"momentum_mean={float(np.mean(residual_row_weights[velocity_rows])):.3e}, "
            f"continuity_mean={float(np.mean(residual_row_weights[pressure_rows])):.3e}"
        )

    if bool(config.select_modes):
        deim_modes_requested, mode_selection["deim"] = _projection_cv_mode_selection(
            residual_snapshots if residual_row_weights is None else np.sqrt(residual_row_weights)[:, None] * residual_snapshots,
            config.deim_mode_candidates,
            center=False,
            validation_fraction=float(config.cv_validation_fraction),
            min_modes=int(trial_basis.shape[1]),
            label="qdeim_residual",
        )
        print(f"cross-validation selected QDEIM collateral modes: deim={deim_modes_requested}")

    requested_deim_modes = max(deim_modes_requested, int(trial_basis.shape[1]))
    deim_modes = min(requested_deim_modes, min(residual_snapshots.shape))
    if deim_modes < trial_basis.shape[1]:
        print(
            "[warn] QDEIM modes are fewer than reduced trial modes; "
            "increase snapshots or decrease velocity/pressure/supremizer modes."
        )
    nonaffine = build_nonaffine_reduced_decomposition(
        residual_snapshots,
        trial_basis,
        n_modes=deim_modes,
        method="qdeim",
        center=False,
        row_weights=residual_row_weights,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    rom_file = output_dir / "turek_2d3_rom.npz"
    np.savez_compressed(
        rom_file,
        trial_basis=trial_basis,
        homogeneous_offset=homogeneous_offset,
        velocity_rows=velocity_rows,
        pressure_rows=pressure_rows,
        velocity_basis=enriched.enriched_primary_basis,
        pressure_basis=pressure_pod.basis,
        supremizer_basis=enriched.lift_basis,
        qdeim_rows=nonaffine.interpolation_rule.rows,
        selected_basis=nonaffine.interpolation_rule.selected_basis,
        residual_terms=nonaffine.residual_terms,
        qdeim_row_weights=(
            np.asarray(nonaffine.sampled_row_weights, dtype=float)
            if nonaffine.sampled_row_weights is not None
            else np.ones(nonaffine.interpolation_rule.rows.size, dtype=float)
        ),
        collateral_basis=nonaffine.collateral_basis.basis,
        lifting_reference=lifting_snapshots[:, -1] if lifting_snapshots.shape[1] else np.zeros(int(problem.dof_handler.total_dofs)),
        metadata=json.dumps(
            {
                "config": asdict(config),
                "velocity_modes": int(velocity_pod.n_modes),
                "pressure_modes": int(pressure_pod.n_modes),
                "mode_selection": mode_selection,
                "supremizer": {
                    "velocity_modes": int(enriched.metadata["primary_modes"]),
                    "pressure_modes": int(enriched.metadata["coupled_modes"]),
                    "supremizer_modes": int(enriched.metadata["lift_modes"]),
                    "enriched_velocity_modes": int(enriched.metadata["enriched_primary_modes"]),
                    "generic_lift_metadata": dict(enriched.metadata),
                },
                "nonaffine": dict(nonaffine.metadata, target="full_residual_with_convective_nonaffine_term"),
                "residual_block_weighting": {
                    "enabled": bool(residual_row_weights is not None),
                    "continuity_row_weight": float(config.continuity_row_weight),
                    "sampled_row_weight_min": float(np.min(nonaffine.sampled_row_weights)) if nonaffine.sampled_row_weights is not None else 1.0,
                    "sampled_row_weight_max": float(np.max(nonaffine.sampled_row_weights)) if nonaffine.sampled_row_weights is not None else 1.0,
                },
                "native_online": "C++ Gauss-Newton can consume qdeim_rows, selected_basis, residual_terms",
            }
        ),
    )
    print(f"wrote ROM: {rom_file}")
    print(
        "ROM sizes: "
        f"trial={trial_basis.shape}, qdeim_rows={nonaffine.interpolation_rule.rows.size}, "
        f"residual_terms={nonaffine.residual_terms.shape}"
    )
    return rom_file


def _compile_native_pair(problem: TurekFOMProblem):
    element_ids = np.arange(int(problem.mesh.n_elements), dtype=np.int32)
    compiler = FormCompiler(problem.dof_handler, quadrature_order=2 * FE_ORDER + 2, backend="cpp")
    residual_runner, residual_funcs, residual_args, _ = compiler._prepare_volume_jit_kernel(
        problem.residual_form,
        element_ids=element_ids,
        full_local_layout=True,
    )
    tangent_runner, tangent_funcs, tangent_args, _ = compiler._prepare_volume_jit_kernel(
        problem.jacobian_form,
        element_ids=element_ids,
        full_local_layout=True,
    )
    residual_runner(residual_funcs, residual_args)
    tangent_runner(tangent_funcs, tangent_args)
    return residual_runner, residual_args, tangent_runner, tangent_args


def infer_current_state_arg_names(*param_orders) -> tuple[str, ...]:
    names: list[str] = []
    for order in param_orders:
        for raw in order:
            name = str(raw)
            if not name.endswith("_loc"):
                continue
            stem = name[: -len("_loc")]
            if stem.endswith(("_n", "_prev", "_previous", "_old")):
                continue
            if stem.endswith(("_k", "_current")) or name.startswith("u_"):
                names.append(name)
    return tuple(dict.fromkeys(names))


def run_native_online_step(rom_file: Path, config: TurekMORConfig, *, time_value: float | None = None):
    """Run one native reduced nonlinear solve from a trained ROM artifact."""

    problem = build_fom_problem(config)
    return _run_native_online_step_for_problem(rom_file, config, problem, time_value=time_value)


def _run_native_online_step_for_problem(
    rom_file: Path,
    config: TurekMORConfig,
    problem: TurekFOMProblem,
    *,
    time_value: float | None = None,
    initial_state: np.ndarray | None = None,
) -> TurekNativeOnlineRun:
    data = np.load(rom_file, allow_pickle=True)
    residual_runner, residual_args, tangent_runner, tangent_args = _compile_native_pair(problem)
    t = float(float(config.theta) * float(config.dt) if time_value is None else time_value)
    bcs_now = NewtonSolver._freeze_bcs(problem.bcs, t)
    lifting = build_dirichlet_lifting_vector(problem.dof_handler, bcs_now)
    offset = np.asarray(data["homogeneous_offset"], dtype=float).reshape(-1) + lifting
    trial_basis = np.asarray(data["trial_basis"], dtype=float)
    if initial_state is None:
        initial_state = initial_mixed_state(problem)
    initial, *_ = np.linalg.lstsq(trial_basis, np.asarray(initial_state, dtype=float).reshape(-1) - offset, rcond=None)
    initial = np.asarray(initial, dtype=float).reshape(-1)
    coefficient_arg_names = infer_current_state_arg_names(residual_runner.param_order, tangent_runner.param_order)
    if not coefficient_arg_names:
        raise RuntimeError(
            "Could not infer current-state coefficient arrays from native param_order. "
            f"Residual order: {residual_runner.param_order}"
        )

    t0 = time.perf_counter()
    result = solve_native_deim_online_gauss_newton(
        residual_metadata_capsule=native_kernel_metadata_from_runner(residual_runner),
        residual_param_order=residual_runner.param_order,
        residual_static_args=residual_args,
        tangent_metadata_capsule=native_kernel_metadata_from_runner(tangent_runner),
        tangent_param_order=tangent_runner.param_order,
        tangent_static_args=tangent_args,
        trial_basis=trial_basis,
        offset=offset,
        initial_coefficients=initial,
        row_dofs=np.asarray(data["qdeim_rows"], dtype=np.int64),
        selected_basis=np.asarray(data["selected_basis"], dtype=float),
        residual_terms=np.asarray(data["residual_terms"], dtype=float),
        row_weights=(
            np.asarray(data["qdeim_row_weights"], dtype=float)
            if "qdeim_row_weights" in data.files
            else None
        ),
        coefficient_arg_names=coefficient_arg_names,
        max_iterations=int(config.max_online_iterations),
        residual_tol=float(config.residual_tol),
        line_search=True,
        adaptive_damping=True,
    )
    elapsed = time.perf_counter() - t0
    state = np.ascontiguousarray(offset + trial_basis @ result.coefficients, dtype=float)
    print(
        "native online result: "
        f"converged={result.converged}, iterations={result.iterations}, "
        f"residual={result.residual_norm:.6e}, backend={result.backend}"
    )
    print(f"native online wall time: {elapsed:.6f} s")
    print(f"native timing counters: {result.timing_counters}")
    return TurekNativeOnlineRun(result=result, state=state, offset=offset, wall_time_s=float(elapsed))


def _relative_l2_error(reference: np.ndarray, prediction: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float).reshape(-1)
    pred = np.asarray(prediction, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(ref))
    if denom <= 1.0e-30:
        denom = 1.0
    return float(np.linalg.norm(pred - ref) / denom)


def _snapshot_metadata(snapshot_data) -> dict[str, Any]:
    if "metadata" not in snapshot_data.files:
        return {}
    raw = snapshot_data["metadata"]
    try:
        return json.loads(str(raw.item()))
    except Exception:
        return {}


def _snapshot_bc_time(snapshot_data, idx: int, config: TurekMORConfig) -> float:
    if "bc_times" in snapshot_data.files:
        values = np.asarray(snapshot_data["bc_times"], dtype=float).reshape(-1)
        if 0 <= int(idx) < values.size:
            return float(values[int(idx)])
    if "times" in snapshot_data.files:
        values = np.asarray(snapshot_data["times"], dtype=float).reshape(-1)
        if 0 <= int(idx) < values.size:
            return float(values[int(idx)]) + float(config.theta) * float(config.dt)
    return (float(idx) + float(config.theta)) * float(config.dt)


def validate_native_rom_against_snapshots(
    *,
    snapshot_file: Path,
    rom_file: Path,
    config: TurekMORConfig,
    snapshot_index: int = 0,
) -> TurekROMValidation:
    """Run native online ROM and compare it to a stored FOM snapshot."""

    snapshot_data = np.load(snapshot_file, allow_pickle=True)
    states = np.asarray(snapshot_data["state_snapshots"], dtype=float)
    if states.ndim != 2 or states.shape[1] == 0:
        raise ValueError("snapshot file contains no FOM states.")
    idx = int(snapshot_index)
    if idx < 0:
        idx = states.shape[1] + idx
    if idx < 0 or idx >= states.shape[1]:
        raise IndexError("snapshot_index is outside the stored snapshot range.")

    problem = build_fom_problem(config)
    previous_state = states[:, idx - 1] if idx > 0 else initial_mixed_state(problem)
    set_mixed_state(problem, previous_state, previous=True)

    time_value = _snapshot_bc_time(snapshot_data, idx, config)
    online = _run_native_online_step_for_problem(
        rom_file,
        config,
        problem,
        time_value=time_value,
        initial_state=previous_state,
    )
    reference = states[:, idx]
    online_error = _relative_l2_error(reference, online.state)
    velocity_rows = field_dof_indices(problem.dof_handler, VELOCITY_FIELDS)
    pressure_rows = field_dof_indices(problem.dof_handler, (PRESSURE_FIELD,))
    velocity_error = _relative_l2_error(reference[velocity_rows], online.state[velocity_rows])
    pressure_error = _relative_l2_error(reference[pressure_rows], online.state[pressure_rows])
    pressure_shift = float(np.mean(reference[pressure_rows] - online.state[pressure_rows]))
    pressure_shifted_error = _relative_l2_error(
        reference[pressure_rows],
        online.state[pressure_rows] + pressure_shift,
    )

    rom_data = np.load(rom_file, allow_pickle=True)
    trial_basis = np.asarray(rom_data["trial_basis"], dtype=float)
    coeffs, *_ = np.linalg.lstsq(trial_basis, reference - online.offset, rcond=None)
    best_state = online.offset + trial_basis @ coeffs
    projection_error = _relative_l2_error(reference, best_state)

    meta = _snapshot_metadata(snapshot_data)
    fom_time = meta.get("fom_wall_time_s")
    fom_mean_step = None
    speedup_value = None
    if fom_time is not None and states.shape[1] > 0:
        fom_mean_step = float(fom_time) / float(states.shape[1])
        if online.wall_time_s > 0.0:
            speedup_value = float(fom_mean_step / online.wall_time_s)

    validation = TurekROMValidation(
        snapshot_index=idx,
        reference_norm=float(np.linalg.norm(reference)),
        online_relative_error=online_error,
        velocity_relative_error=velocity_error,
        pressure_relative_error=pressure_error,
        pressure_shifted_relative_error=pressure_shifted_error,
        projection_relative_error=projection_error,
        native_online_wall_time_s=float(online.wall_time_s),
        fom_mean_step_time_s=fom_mean_step,
        speedup_vs_mean_fom_step=speedup_value,
        converged=bool(online.result.converged),
        iterations=int(online.result.iterations),
        residual_norm=float(online.result.residual_norm),
    )
    print(
        "ROM validation: "
        f"snapshot={validation.snapshot_index}, "
        f"online_rel_error={validation.online_relative_error:.6e}, "
        f"velocity_rel_error={validation.velocity_relative_error:.6e}, "
        f"pressure_shifted_rel_error={validation.pressure_shifted_relative_error:.6e}, "
        f"projection_rel_error={validation.projection_relative_error:.6e}, "
        f"native_wall={validation.native_online_wall_time_s:.6f}s"
    )
    if validation.speedup_vs_mean_fom_step is not None:
        print(
            "ROM timing: "
            f"fom_mean_step={validation.fom_mean_step_time_s:.6f}s, "
            f"speedup={validation.speedup_vs_mean_fom_step:.3f}x"
        )
    return validation


def run_synthetic_smoke(output_dir: Path) -> Path:
    """Cheap offline smoke for lifting, supremizers, and QDEIM artifacts."""

    rng = np.random.default_rng(7)
    total_dofs = 18
    velocity_rows = np.arange(0, 12, dtype=np.int64)
    pressure_rows = np.arange(12, 18, dtype=np.int64)
    true_v, _ = np.linalg.qr(rng.normal(size=(velocity_rows.size, 4)))
    true_p, _ = np.linalg.qr(rng.normal(size=(pressure_rows.size, 3)))
    coeffs = rng.normal(size=(4, 9))
    pcoeffs = rng.normal(size=(3, 9))
    lifting = np.zeros((total_dofs, 9), dtype=float)
    lifting[0, :] = np.linspace(0.2, 1.0, 9)
    lifting[5, :] = 0.1

    snapshots = lifting.copy()
    snapshots[velocity_rows, :] += true_v @ coeffs
    snapshots[pressure_rows, :] += true_p @ pcoeffs
    homogeneous = remove_lifting_from_snapshots(snapshots, lifting)

    velocity_pod = fit_pod(homogeneous[velocity_rows, :], n_modes=4, center=False)
    pressure_pod = fit_pod(homogeneous[pressure_rows, :], n_modes=3, center=False)
    A = np.eye(velocity_rows.size)
    B = rng.normal(size=(pressure_rows.size, velocity_rows.size))
    S = solve_coupled_lift_snapshots(A, B, pressure_pod.basis)
    enriched = fit_lift_enriched_basis(
        velocity_pod.basis,
        pressure_pod.basis,
        S,
        n_lift_modes=3,
    )
    trial_basis = build_mixed_field_basis(
        total_dofs=total_dofs,
        field_blocks=(
            MixedBasisBlock(rows=velocity_rows, basis=enriched.enriched_primary_basis, name="velocity"),
            MixedBasisBlock(rows=pressure_rows, basis=pressure_pod.basis, name="pressure"),
        ),
    )
    residual_snapshots = trial_basis @ rng.normal(size=(trial_basis.shape[1], 9))
    residual_snapshots += 0.05 * rng.normal(size=residual_snapshots.shape)
    nonaffine = build_nonaffine_reduced_decomposition(
        residual_snapshots,
        trial_basis,
        n_modes=min(6, min(residual_snapshots.shape)),
        method="qdeim",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    rom_file = output_dir / "turek_2d3_smoke_rom.npz"
    np.savez_compressed(
        rom_file,
        trial_basis=trial_basis,
        homogeneous_offset=np.zeros(total_dofs),
        velocity_rows=velocity_rows,
        pressure_rows=pressure_rows,
        velocity_basis=enriched.enriched_primary_basis,
        pressure_basis=pressure_pod.basis,
        supremizer_basis=enriched.lift_basis,
        qdeim_rows=nonaffine.interpolation_rule.rows,
        selected_basis=nonaffine.interpolation_rule.selected_basis,
        residual_terms=nonaffine.residual_terms,
        collateral_basis=nonaffine.collateral_basis.basis,
        lifting_reference=lifting[:, -1],
        metadata=json.dumps({"smoke": True, "lift_enrichment": dict(enriched.metadata), "nonaffine": dict(nonaffine.metadata)}),
    )
    print(f"wrote smoke ROM: {rom_file}")
    return rom_file


def _config_from_args(args: argparse.Namespace) -> TurekMORConfig:
    return TurekMORConfig(
        mesh_backend=str(args.mesh_backend),
        mesh_type=str(args.mesh_type),
        mesh_size=float(args.mesh_size),
        mesh_file=None if args.mesh_file is None else str(args.mesh_file),
        rebuild_mesh=bool(args.rebuild_mesh),
        refine_hmin=args.refine_hmin,
        refine_band_radius=args.refine_band_radius,
        backend=str(args.backend),
        inflow=str(args.inflow),
        dt=float(args.dt),
        theta=float(args.theta),
        max_steps=int(args.max_steps),
        velocity_modes=int(args.velocity_modes),
        pressure_modes=int(args.pressure_modes),
        supremizer_modes=int(args.supremizer_modes),
        deim_modes=int(args.deim_modes),
        select_modes=bool(args.select_modes),
        velocity_mode_candidates=_parse_mode_candidates(str(args.velocity_mode_candidates)),
        pressure_mode_candidates=_parse_mode_candidates(str(args.pressure_mode_candidates)),
        supremizer_mode_candidates=_parse_mode_candidates(str(args.supremizer_mode_candidates)),
        deim_mode_candidates=_parse_mode_candidates(str(args.deim_mode_candidates)),
        cv_validation_fraction=float(args.cv_validation_fraction),
        residual_block_weighting=not bool(args.no_residual_block_weighting),
        continuity_row_weight=float(args.continuity_row_weight),
        supremizer_regularization=float(args.supremizer_regularization),
        center_snapshots=bool(args.center_snapshots),
        convection_form=str(args.convection_form),
        residual_tol=float(args.residual_tol),
        max_online_iterations=int(args.max_online_iterations),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Turek 2D-3 Navier-Stokes native MOR workflow.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("offline-snapshots", "train", "online-native", "validate", "benchmark", "smoke"):
        p = sub.add_parser(name)
        p.add_argument("--output-dir", type=Path, default=Path("examples/turek_navier_stokes_mor/artifacts"))
        p.add_argument("--snapshot-file", type=Path, default=None)
        p.add_argument("--rom-file", type=Path, default=None)
        p.add_argument("--validation-index", type=int, default=0)
        p.add_argument("--backend", choices=("python", "jit", "cpp"), default="cpp")
        p.add_argument("--mesh-backend", choices=("structured", "gmsh"), default="structured")
        p.add_argument("--mesh-type", choices=("quad", "tri"), default="quad")
        p.add_argument("--mesh-size", type=float, default=0.04)
        p.add_argument("--mesh-file", type=Path, default=None)
        p.add_argument("--rebuild-mesh", action="store_true")
        p.add_argument("--refine-hmin", type=float, default=None)
        p.add_argument("--refine-band-radius", type=float, default=0.2)
        p.add_argument("--inflow", choices=("dfg", "constant"), default="dfg")
        p.add_argument("--dt", type=float, default=0.05)
        p.add_argument("--theta", type=float, default=0.5)
        p.add_argument("--max-steps", type=int, default=8)
        p.add_argument("--velocity-modes", type=int, default=8)
        p.add_argument("--pressure-modes", type=int, default=4)
        p.add_argument("--supremizer-modes", type=int, default=4)
        p.add_argument("--deim-modes", type=int, default=32)
        p.add_argument("--select-modes", action="store_true")
        p.add_argument("--velocity-mode-candidates", type=str, default="2,4,6,8")
        p.add_argument("--pressure-mode-candidates", type=str, default="1,2,3,4")
        p.add_argument("--supremizer-mode-candidates", type=str, default="1,2,3,4")
        p.add_argument("--deim-mode-candidates", type=str, default="16,24,32,48")
        p.add_argument("--cv-validation-fraction", type=float, default=0.25)
        p.add_argument("--no-residual-block-weighting", action="store_true")
        p.add_argument("--continuity-row-weight", type=float, default=16.0)
        p.add_argument("--supremizer-regularization", type=float, default=1.0e-10)
        p.add_argument("--residual-tol", type=float, default=1.0e-9)
        p.add_argument("--max-online-iterations", type=int, default=10)
        p.add_argument("--center-snapshots", action="store_true")
        p.add_argument("--convection-form", choices=("standard", "skew"), default="standard")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "smoke":
        run_synthetic_smoke(args.output_dir)
        return

    config = _config_from_args(args)
    if args.command == "offline-snapshots":
        collect_full_order_snapshots(config, args.output_dir)
        return
    if args.command == "train":
        snapshot_file = args.snapshot_file or (args.output_dir / "turek_2d3_snapshots.npz")
        train_turek_rom(snapshot_file, args.output_dir, config)
        return
    if args.command == "online-native":
        rom_file = args.rom_file or (args.output_dir / "turek_2d3_rom.npz")
        run_native_online_step(rom_file, config)
        return
    if args.command == "validate":
        snapshot_file = args.snapshot_file or (args.output_dir / "turek_2d3_snapshots.npz")
        rom_file = args.rom_file or (args.output_dir / "turek_2d3_rom.npz")
        validate_native_rom_against_snapshots(
            snapshot_file=snapshot_file,
            rom_file=rom_file,
            config=config,
            snapshot_index=int(args.validation_index),
        )
        return
    if args.command == "benchmark":
        snapshot_file = collect_full_order_snapshots(config, args.output_dir)
        rom_file = train_turek_rom(snapshot_file, args.output_dir, config)
        validate_native_rom_against_snapshots(
            snapshot_file=snapshot_file,
            rom_file=rom_file,
            config=config,
            snapshot_index=int(args.validation_index),
        )
        return
    raise ValueError(f"unsupported command {args.command!r}")


if __name__ == "__main__":
    main()
