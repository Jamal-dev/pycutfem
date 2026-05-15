from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np

from examples.NIRB.reduced_fluid import ReducedFluidSolveResult
from examples.NIRB.reduced_mesh import ReducedMeshMotionOperator, ReducedMeshMotionState
from pycutfem.nirb.reduced_interface import ReducedIQNILS, ReducedInterfaceSpace


SolidReducedSolve = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
FluidReducedSolve = Callable[[np.ndarray, object], ReducedFluidSolveResult]


@dataclass
class ReducedOnlineTimers:
    solid_reduced_s: float = 0.0
    interface_reduced_s: float = 0.0
    mesh_reduced_s: float = 0.0
    fluid_reduced_s: float = 0.0
    reaction_reduced_s: float = 0.0
    iqn_reduced_s: float = 0.0
    full_reconstruction_s: float = 0.0
    forbidden_full_call_count: int = 0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "solid_reduced_s": float(self.solid_reduced_s),
            "interface_reduced_s": float(self.interface_reduced_s),
            "mesh_reduced_s": float(self.mesh_reduced_s),
            "fluid_reduced_s": float(self.fluid_reduced_s),
            "reaction_reduced_s": float(self.reaction_reduced_s),
            "iqn_reduced_s": float(self.iqn_reduced_s),
            "full_reconstruction_s": float(self.full_reconstruction_s),
            "forbidden_full_call_count": int(self.forbidden_full_call_count),
        }

    def record_forbidden_full_call(self) -> None:
        self.forbidden_full_call_count += 1


@dataclass(frozen=True)
class ReducedFSIState:
    load: np.ndarray
    solid: np.ndarray
    interface_displacement: np.ndarray
    mesh: np.ndarray
    mesh_velocity: np.ndarray
    mesh_acceleration: np.ndarray
    fluid: np.ndarray
    reaction_load: np.ndarray
    mesh_history: ReducedMeshMotionState


@dataclass(frozen=True)
class ReducedCouplingIteration:
    iteration: int
    disp_abs: float
    disp_rel: float
    load_abs: float
    load_rel: float
    converged: bool
    fluid_iterations: int
    fluid_residual_norm: float


@dataclass(frozen=True)
class ReducedStepResult:
    state: ReducedFSIState
    iterations: tuple[ReducedCouplingIteration, ...]
    timers: dict[str, float | int]

    @property
    def converged(self) -> bool:
        return bool(self.iterations and self.iterations[-1].converged)


@dataclass
class ReducedOnlineFSISolver:
    """Reduced-coordinate partitioned FSI loop.

    This driver is intentionally independent from `run_example2_local.py`.
    Physics-specific solid/fluid callables are injected so the loop can be
    tested without full-order fields and then wired to the element-local
    ALE-DVMS implementation.
    """

    load_space: ReducedInterfaceSpace
    displacement_space: ReducedInterfaceSpace
    solid_solve: SolidReducedSolve
    mesh_operator: ReducedMeshMotionOperator
    fluid_solve: FluidReducedSolve
    iqn: ReducedIQNILS = field(default_factory=ReducedIQNILS)
    coupling_abs_tol: float = 5.0e-3
    coupling_rel_tol: float = 5.0e-3
    max_coupling_iterations: int = 50
    timers: ReducedOnlineTimers = field(default_factory=ReducedOnlineTimers)

    def _check_load(self, values: np.ndarray) -> np.ndarray:
        load = np.asarray(values, dtype=float).reshape(-1)
        if int(load.size) != self.load_space.n_modes:
            raise ValueError(f"expected {self.load_space.n_modes} reduced load modes, got {load.size}.")
        if not np.all(np.isfinite(load)):
            raise ValueError("reduced load contains non-finite values.")
        return load

    def run_step(self, state: ReducedFSIState) -> ReducedStepResult:
        current_load = self._check_load(state.load).copy()
        previous_interface_disp = np.asarray(state.interface_displacement, dtype=float).reshape(-1)
        if int(previous_interface_disp.size) != self.displacement_space.n_modes:
            raise ValueError("previous interface displacement size does not match displacement space.")

        mesh_history = state.mesh_history
        iterations: list[ReducedCouplingIteration] = []
        accepted_state = state

        for iteration in range(1, max(1, int(self.max_coupling_iterations)) + 1):
            t0 = perf_counter()
            solid_q, interface_disp = self.solid_solve(current_load.copy())
            self.timers.solid_reduced_s += perf_counter() - t0
            solid_q = np.asarray(solid_q, dtype=float).reshape(-1)
            interface_disp = np.asarray(interface_disp, dtype=float).reshape(-1)
            if int(interface_disp.size) != self.displacement_space.n_modes:
                raise ValueError("solid solver returned incompatible interface displacement coefficients.")

            t0 = perf_counter()
            mesh_result = self.mesh_operator.solve(interface_disp, mesh_history)
            self.timers.mesh_reduced_s += perf_counter() - t0

            t0 = perf_counter()
            fluid_result = self.fluid_solve(current_load.copy(), mesh_result)
            self.timers.fluid_reduced_s += perf_counter() - t0
            if fluid_result.reaction_coefficients is None:
                raise RuntimeError("reduced fluid solve did not return reaction/load coefficients.")
            returned_load = self._check_load(fluid_result.reaction_coefficients)

            t0 = perf_counter()
            disp_abs, disp_rel = self.displacement_space.relative_change(interface_disp, previous_interface_disp)
            load_abs, load_rel = self.load_space.relative_change(returned_load, current_load)
            converged = bool(
                (disp_abs <= float(self.coupling_abs_tol) or disp_rel <= float(self.coupling_rel_tol))
                and (load_abs <= float(self.coupling_abs_tol) or load_rel <= float(self.coupling_rel_tol))
            )
            self.timers.interface_reduced_s += perf_counter() - t0

            iterations.append(
                ReducedCouplingIteration(
                    iteration=int(iteration),
                    disp_abs=float(disp_abs),
                    disp_rel=float(disp_rel),
                    load_abs=float(load_abs),
                    load_rel=float(load_rel),
                    converged=bool(converged),
                    fluid_iterations=int(fluid_result.iterations),
                    fluid_residual_norm=float(fluid_result.residual_norm),
                )
            )
            accepted_state = ReducedFSIState(
                load=returned_load.copy(),
                solid=solid_q.copy(),
                interface_displacement=interface_disp.copy(),
                mesh=mesh_result.q.copy(),
                mesh_velocity=mesh_result.v.copy(),
                mesh_acceleration=mesh_result.a.copy(),
                fluid=np.asarray(fluid_result.coefficients, dtype=float).reshape(-1).copy(),
                reaction_load=returned_load.copy(),
                mesh_history=mesh_history.accept(mesh_result.q, mesh_result.v, mesh_result.a),
            )
            if converged:
                self.iqn.finalize_step()
                return ReducedStepResult(
                    state=accepted_state,
                    iterations=tuple(iterations),
                    timers=self.timers.as_dict(),
                )

            t0 = perf_counter()
            current_load = self.iqn.next(current_load, returned_load, converged=False)
            self.timers.iqn_reduced_s += perf_counter() - t0
            previous_interface_disp = interface_disp.copy()

        raise RuntimeError(
            "reduced FSI fixed-point did not converge after "
            f"{int(self.max_coupling_iterations)} iterations "
            f"(disp_rel={iterations[-1].disp_rel:.6e}, load_rel={iterations[-1].load_rel:.6e})."
        )


__all__ = [
    "ReducedCouplingIteration",
    "ReducedFSIState",
    "ReducedOnlineFSISolver",
    "ReducedOnlineTimers",
    "ReducedStepResult",
]
