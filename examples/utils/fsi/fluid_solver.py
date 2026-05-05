from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.operators import RuntimeOperator
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, NewtonSolver

from examples.NIRB.dvms import FluidDVMSState
from examples.NIRB.dvms.local_operator import FluidDVMSCondensedLocalSystemOperator
from examples.NIRB.run_example2_local import (
    CoordinateLookup,
    _FluidBossakAccelerationOperator,
    _FluidDVMSSolverOperator,
    _attach_runtime_operator_post_update_hook,
    _build_fluid_problem,
    _fluid_boundary_conditions,
    _fluid_interface_reaction_loads,
    _fluid_residual_and_jacobian,
    _fluid_zero_local_operator_forms,
)

ScalarProfile = Callable[[float, float], float]


@dataclass(frozen=True)
class FluidMaterial:
    """Fluid parameters for the Kratos-matched ALE-DVMS operator."""

    density: float
    dynamic_viscosity: float


@dataclass(frozen=True)
class FluidBoundaryTags:
    """Boundary tag names used by the reusable partitioned fluid solve."""

    inlet: str = "inlet"
    interface: str = "interface"
    outlet: str = "outlet"
    walls: str = "walls"
    cylinder: str = "cylinder"


@dataclass
class SanitizedDVMSFluidSolver:
    """
    Reusable wrapper for the sanitized Kratos-matched incompressible ALE-DVMS solve.

    The wrapper owns the mixed velocity-pressure-mesh problem dictionary used by
    the NIRB DoubleFlap driver, but exposes it through a stable example-utils API
    so other examples can reuse the same local operators without importing the
    full driver script at call sites.
    """

    problem: dict[str, object]
    material: FluidMaterial
    dt: float
    tags: FluidBoundaryTags = FluidBoundaryTags()
    bossak_alpha: float = -0.3
    dynamic_tau: float = 1.0
    pressure_gauge: float = 1.0e-5
    quad_order: int = 1

    @classmethod
    def from_mesh(
        cls,
        mesh: Mesh,
        *,
        density: float,
        dynamic_viscosity: float,
        dt: float,
        poly_order: int = 1,
        pressure_order: int = 1,
        mesh_order: int | None = None,
        quadrature_order: int | None = None,
        tags: FluidBoundaryTags | None = None,
        bossak_alpha: float = -0.3,
        dynamic_tau: float = 1.0,
        pressure_gauge: float = 1.0e-5,
    ) -> "SanitizedDVMSFluidSolver":
        problem = _build_fluid_problem(
            mesh,
            poly_order=int(poly_order),
            pressure_order=int(pressure_order),
            mesh_order=mesh_order,
            quadrature_order=quadrature_order,
        )
        quad_order = int(quadrature_order) if quadrature_order is not None else int(problem["dvms_state"].quadrature_order)
        return cls(
            problem=problem,
            material=FluidMaterial(float(density), float(dynamic_viscosity)),
            dt=float(dt),
            tags=FluidBoundaryTags() if tags is None else tags,
            bossak_alpha=float(bossak_alpha),
            dynamic_tau=float(dynamic_tau),
            pressure_gauge=float(pressure_gauge),
            quad_order=quad_order,
        )

    @property
    def dh(self) -> DofHandler:
        return self.problem["dh"]

    @property
    def mixed_element(self) -> MixedElement:
        return self.problem["me"]

    @property
    def mesh(self) -> Mesh:
        return self.dh.mixed_element.mesh

    @property
    def dvms_state(self) -> FluidDVMSState:
        return self.problem["dvms_state"]

    def boundary_conditions(
        self,
        *,
        iface_velocity: CoordinateLookup,
        inlet_lookup: ScalarProfile,
    ):
        return _fluid_boundary_conditions(
            iface_velocity=iface_velocity,
            inlet_lookup=inlet_lookup,
            inlet_tag=str(self.tags.inlet),
            interface_tag=str(self.tags.interface),
            outlet_tag=str(self.tags.outlet),
            walls_tag=str(self.tags.walls),
            cylinder_tag=str(self.tags.cylinder),
        )

    def residual_and_jacobian(
        self,
        *,
        iface_velocity: CoordinateLookup,
        inlet_lookup: ScalarProfile,
    ):
        return _fluid_residual_and_jacobian(
            prob=self.problem,
            rho_f=float(self.material.density),
            mu_f=float(self.material.dynamic_viscosity),
            dt=float(self.dt),
            bossak_alpha=float(self.bossak_alpha),
            dynamic_tau=float(self.dynamic_tau),
            pressure_gauge=float(self.pressure_gauge),
            iface_velocity=iface_velocity,
            inlet_lookup=inlet_lookup,
            inlet_tag=str(self.tags.inlet),
            interface_tag=str(self.tags.interface),
            outlet_tag=str(self.tags.outlet),
            walls_tag=str(self.tags.walls),
            cylinder_tag=str(self.tags.cylinder),
            quad_order=int(self.quad_order),
        )

    def exact_placeholder_forms(
        self,
        *,
        iface_velocity: CoordinateLookup,
        inlet_lookup: ScalarProfile,
    ):
        return _fluid_zero_local_operator_forms(
            prob=self.problem,
            iface_velocity=iface_velocity,
            inlet_lookup=inlet_lookup,
            inlet_tag=str(self.tags.inlet),
            interface_tag=str(self.tags.interface),
            outlet_tag=str(self.tags.outlet),
            walls_tag=str(self.tags.walls),
            cylinder_tag=str(self.tags.cylinder),
            quad_order=int(self.quad_order),
        )

    def condensed_local_operator(
        self,
        *,
        refresh_predicted_subscale: bool = False,
        apply_dirichlet_lift: bool = False,
        contribution_mode: str = "system",
        residualization: str = "kratos",
        element_ids: np.ndarray | None = None,
        body_force: np.ndarray | None = None,
        use_oss: bool = False,
    ) -> FluidDVMSCondensedLocalSystemOperator:
        return FluidDVMSCondensedLocalSystemOperator(
            mesh=self.mesh,
            dh=self.dh,
            u_k=self.problem["u_k"],
            u_prev=self.problem["u_prev"],
            a_prev=self.problem["a_prev"],
            a_curr=self.problem["a_k"],
            p_k=self.problem["p_k"],
            d_mesh=self.problem["d_mesh"],
            d_prev=self.problem["d_prev"],
            d_prev2=self.problem["d_prev2"],
            mesh_v=self.problem["w_mesh_k"],
            mesh_v_prev=self.problem["w_mesh_prev"],
            mesh_a_prev=self.problem["a_mesh_prev"],
            state=self.dvms_state,
            rho_f=float(self.material.density),
            mu_f=float(self.material.dynamic_viscosity),
            dt=float(self.dt),
            bossak_alpha=float(self.bossak_alpha),
            element_ids=element_ids,
            quadrature_order=int(self.quad_order),
            body_force=body_force,
            use_oss=bool(use_oss),
            apply_dirichlet_lift=bool(apply_dirichlet_lift),
            contribution_mode=str(contribution_mode),
            residualization=str(residualization),
            dynamic_tau=float(self.dynamic_tau),
            refresh_predicted_subscale=bool(refresh_predicted_subscale),
        )

    def dvms_predictor_operator(
        self,
        *,
        reset_predicted_to_old_on_step_begin: bool = False,
        refresh_on_initial_assembly: bool = True,
        max_iterations: int = 10,
        rel_tol: float = 1.0e-14,
        abs_tol: float = 1.0e-14,
        use_oss: bool = False,
    ) -> _FluidDVMSSolverOperator:
        return _FluidDVMSSolverOperator(
            state=self.dvms_state,
            dh=self.dh,
            mesh=self.mesh,
            u_k=self.problem["u_k"],
            u_prev=self.problem["u_prev"],
            a_prev=self.problem["a_prev"],
            a_curr=self.problem["a_k"],
            p_k=self.problem["p_k"],
            d_mesh=self.problem["d_mesh"],
            d_prev=self.problem["d_prev"],
            d_prev2=self.problem["d_prev2"],
            mesh_v=self.problem["w_mesh_k"],
            mesh_v_prev=self.problem["w_mesh_prev"],
            mesh_a_prev=self.problem["a_mesh_prev"],
            rho_f=float(self.material.density),
            mu_f=float(self.material.dynamic_viscosity),
            dt=float(self.dt),
            bossak_alpha=float(self.bossak_alpha),
            dynamic_tau=float(self.dynamic_tau),
            max_iterations=int(max_iterations),
            rel_tol=float(rel_tol),
            abs_tol=float(abs_tol),
            use_oss=bool(use_oss),
            refresh_on_initial_assembly=bool(refresh_on_initial_assembly),
            reset_predicted_to_old_on_step_begin=bool(reset_predicted_to_old_on_step_begin),
        )

    def bossak_acceleration_operator(self) -> _FluidBossakAccelerationOperator:
        return _FluidBossakAccelerationOperator(
            u_k=self.problem["u_k"],
            a_k=self.problem["a_k"],
            dt=float(self.dt),
            bossak_alpha=float(self.bossak_alpha),
        )

    def exact_runtime_operators(
        self,
        *,
        reset_predicted_to_old_on_step_begin: bool = False,
        refresh_predicted_subscale_in_local_operator: bool = False,
        apply_dirichlet_lift: bool = False,
        use_oss: bool = False,
    ) -> list[RuntimeOperator]:
        return [
            self.bossak_acceleration_operator(),
            self.dvms_predictor_operator(
                reset_predicted_to_old_on_step_begin=bool(reset_predicted_to_old_on_step_begin),
                use_oss=bool(use_oss),
            ),
            self.condensed_local_operator(
                refresh_predicted_subscale=bool(refresh_predicted_subscale_in_local_operator),
                apply_dirichlet_lift=bool(apply_dirichlet_lift),
                use_oss=bool(use_oss),
            ),
        ]

    def reaction_loads(
        self,
        *,
        backend: str = "python",
        contribution_mode: str = "system",
        apply_dirichlet_lift: bool = False,
        refresh_state: bool = True,
    ) -> CoordinateLookup:
        return _fluid_interface_reaction_loads(
            prob=self.problem,
            rho_f=float(self.material.density),
            mu_f=float(self.material.dynamic_viscosity),
            dt=float(self.dt),
            quad_order=int(self.quad_order),
            bossak_alpha=float(self.bossak_alpha),
            dynamic_tau=float(self.dynamic_tau),
            interface_tag=str(self.tags.interface),
            backend=str(backend),
            contribution_mode=str(contribution_mode),
            apply_dirichlet_lift=bool(apply_dirichlet_lift),
            refresh_state=bool(refresh_state),
        )

    def newton_solver(
        self,
        *,
        iface_velocity: CoordinateLookup,
        inlet_lookup: ScalarProfile,
        use_exact_local_operator: bool = True,
        backend: str = "cpp",
        linear_backend: str = "scipy",
        newton_params: NewtonParameters | None = None,
        lin_params: LinearSolverParameters | None = None,
        operators: Sequence[RuntimeOperator] = (),
        active_fields: Sequence[str] = ("ux", "uy", "p"),
    ) -> NewtonSolver:
        if bool(use_exact_local_operator):
            residual, jacobian, bcs, bcs_homog = self.exact_placeholder_forms(
                iface_velocity=iface_velocity,
                inlet_lookup=inlet_lookup,
            )
            runtime_ops = self.exact_runtime_operators()
        else:
            residual, jacobian, bcs, bcs_homog = self.residual_and_jacobian(
                iface_velocity=iface_velocity,
                inlet_lookup=inlet_lookup,
            )
            runtime_ops = [self.dvms_predictor_operator()]
        runtime_ops.extend(list(operators or ()))
        solver = NewtonSolver(
            residual_form=residual,
            jacobian_form=jacobian,
            dof_handler=self.dh,
            mixed_element=self.mixed_element,
            bcs=list(bcs),
            bcs_homog=list(bcs_homog),
            newton_params=newton_params if newton_params is not None else NewtonParameters(),
            lin_params=lin_params if lin_params is not None else LinearSolverParameters(backend=str(linear_backend)),
            quad_order=int(self.quad_order),
            backend=str(backend),
            operators=runtime_ops,
        )
        solver.set_active_fields(list(active_fields))
        _attach_runtime_operator_post_update_hook(solver=solver, operators=runtime_ops)
        return solver


__all__ = [
    "FluidBoundaryTags",
    "FluidMaterial",
    "SanitizedDVMSFluidSolver",
]
