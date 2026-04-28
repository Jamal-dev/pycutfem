from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.operators import RuntimeOperator
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, NewtonSolver

from examples.NIRB.run_example2_local import (
    CoordinateLookup,
    _ReducedResidualShiftOperator,
    _boundary_field_data,
    _boundary_point_load_vector,
    _build_solid_problem,
    _solid_residual_and_jacobian,
)


@dataclass(frozen=True)
class StructureMaterial:
    """Plane-strain compressible neo-Hookean material parameters."""

    shear_modulus: float
    lame_lambda: float


@dataclass(frozen=True)
class StructureBoundaryTags:
    """Boundary tag names used by the reusable partitioned structure solve."""

    interface: str = "interface"
    clamp: str = "clamp"


@dataclass
class SanitizedHyperelasticStructureSolver:
    """Reusable wrapper for the sanitized Kratos-matched structural solve."""

    problem: dict[str, object]
    material: StructureMaterial
    tags: StructureBoundaryTags = StructureBoundaryTags()
    quad_order: int = 2

    @classmethod
    def from_mesh(
        cls,
        mesh: Mesh,
        *,
        shear_modulus: float,
        lame_lambda: float,
        poly_order: int = 1,
        quad_order: int = 2,
        tags: StructureBoundaryTags | None = None,
    ) -> "SanitizedHyperelasticStructureSolver":
        return cls(
            problem=_build_solid_problem(mesh, poly_order=int(poly_order)),
            material=StructureMaterial(float(shear_modulus), float(lame_lambda)),
            tags=StructureBoundaryTags() if tags is None else tags,
            quad_order=int(quad_order),
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

    def interface_coordinates(self) -> np.ndarray:
        coords_x, _ = _boundary_field_data(self.dh, self.problem["d_k"].components[0].field_name, str(self.tags.interface))
        coords_y, _ = _boundary_field_data(self.dh, self.problem["d_k"].components[1].field_name, str(self.tags.interface))
        if coords_x.shape != coords_y.shape or (coords_x.size and not np.allclose(coords_x, coords_y)):
            raise RuntimeError(f"Mismatched structural interface coordinate ordering for tag {self.tags.interface!r}")
        return np.asarray(coords_x, dtype=float)

    def zero_traction_lookup(self) -> CoordinateLookup:
        coords = self.interface_coordinates()
        return CoordinateLookup(coords, np.zeros((coords.shape[0], 2), dtype=float), dim=2)

    def forms(self, *, traction_lookup: CoordinateLookup):
        return _solid_residual_and_jacobian(
            prob=self.problem,
            traction_lookup=traction_lookup,
            mu_s=float(self.material.shear_modulus),
            lambda_s=float(self.material.lame_lambda),
            interface_tag=str(self.tags.interface),
            clamp_tag=str(self.tags.clamp),
            quad_order=int(self.quad_order),
        )

    def point_load_vector(self, values: np.ndarray) -> np.ndarray:
        return _boundary_point_load_vector(
            self.dh,
            vector=self.problem["d_k"],
            tag=str(self.tags.interface),
            values=np.asarray(values, dtype=float),
        )

    def reduced_point_load_operator(
        self,
        values: np.ndarray,
        *,
        active_dofs: np.ndarray,
    ) -> _ReducedResidualShiftOperator:
        full_shift = self.point_load_vector(values)
        return _ReducedResidualShiftOperator(np.asarray(full_shift, dtype=float)[np.asarray(active_dofs, dtype=int)])

    def newton_solver(
        self,
        *,
        traction_lookup: CoordinateLookup | None = None,
        point_load_values: np.ndarray | None = None,
        backend: str = "cpp",
        linear_backend: str = "scipy",
        newton_params: NewtonParameters | None = None,
        lin_params: LinearSolverParameters | None = None,
        operators: Sequence[RuntimeOperator] = (),
        active_fields: Sequence[str] = ("dx", "dy"),
    ) -> NewtonSolver:
        residual, jacobian, bcs, bcs_homog = self.forms(
            traction_lookup=traction_lookup if traction_lookup is not None else self.zero_traction_lookup()
        )
        runtime_ops = list(operators or ())
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
        if point_load_values is not None:
            runtime_ops.append(self.reduced_point_load_operator(point_load_values, active_dofs=solver.active_dofs))
            solver.set_runtime_operators(runtime_ops)
        return solver


__all__ = [
    "SanitizedHyperelasticStructureSolver",
    "StructureBoundaryTags",
    "StructureMaterial",
]
