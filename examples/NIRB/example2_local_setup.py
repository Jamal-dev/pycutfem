from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .double_flap_reference import DoubleFlapReference, load_double_flap_reference, reynolds_to_mean_velocity
from .example2_problem import DoubleFlapGeometry, build_bcs, build_jac, build_residual, load_geometry


@dataclass(frozen=True)
class Example2Material:
    density: float
    kinematic_viscosity: float
    max_velocity: float
    young_modulus: float
    poisson_ratio: float

    @property
    def shear_modulus(self) -> float:
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def lame_lambda(self) -> float:
        nu = self.poisson_ratio
        return self.young_modulus * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


@dataclass(frozen=True)
class Example2BoundarySpec:
    fluid_tags: tuple[str, ...]
    solid_tags: tuple[str, ...]
    interface_tag: str
    clamp_tag: str
    inlet_ramp_end_time: float
    inlet_modulus_ramp: str
    inlet_modulus_steady: str
    time_step: float
    end_time: float


@dataclass(frozen=True)
class Example2LocalSetup:
    reference: DoubleFlapReference
    geometry: DoubleFlapGeometry
    material: Example2Material
    boundaries: Example2BoundarySpec
    mesh_builder_default: Path
    mesh_size_default: float
    mesh_order_default: int

    def u_mean_for_reynolds(self, reynolds: float) -> float:
        return reynolds_to_mean_velocity(
            reynolds,
            kinematic_viscosity=self.material.kinematic_viscosity,
            cylinder_diameter=0.1,
        )

    def local_baseline_run_config(self, reynolds: float) -> dict[str, Any]:
        del reynolds
        return {
            "mesh_builder": str(self.mesh_builder_default),
            "mesh_size": float(self.mesh_size_default),
            "mesh_order": int(self.mesh_order_default),
            "reference_velocity": float(self.material.max_velocity),
            "dt": float(self.boundaries.time_step),
            "theta": 0.5,
        }

    def to_dict(self, *, reynolds: float | None = None) -> dict[str, Any]:
        payload = {
            "reference_root": str(self.reference.root),
            "geometry": asdict(self.geometry),
            "material": asdict(self.material),
            "boundaries": asdict(self.boundaries),
            "mesh_builder_default": str(self.mesh_builder_default),
            "mesh_size_default": float(self.mesh_size_default),
            "mesh_order_default": int(self.mesh_order_default),
            "local_forms": {
                "residual_builder": f"{build_residual.__module__}.{build_residual.__name__}",
                "jacobian_builder": f"{build_jac.__module__}.{build_jac.__name__}",
                "bc_builder": f"{build_bcs.__module__}.{build_bcs.__name__}",
            },
            "paper_example2_notes": {
                "fluid_and_solid_meshes_in_reference_are_separate": True,
                "local_pycutfem_problem_uses_a_single_conforming_mesh": True,
                "local_solid_constitutive_law": "compressible Neo-Hookean",
                "current_target": "Use the local DoubleFlap geometry, boundary conditions, and weak forms to assemble the pycutfem FOM directly.",
            },
        }
        if reynolds is not None:
            payload["baseline_run_config"] = self.local_baseline_run_config(reynolds)
        return payload


def load_example2_local_setup(
    *,
    reference_root: str | Path | None = None,
    mesh_builder_default: str | Path = "examples/NIRB/example2_problem.py",
    mesh_size_default: float = 0.04,
    mesh_order_default: int = 1,
) -> Example2LocalSetup:
    reference = load_double_flap_reference(reference_root)
    geometry = load_geometry(reference_root)
    material = Example2Material(
        density=reference.density,
        kinematic_viscosity=reference.kinematic_viscosity,
        max_velocity=reference.max_velocity,
        young_modulus=10.0e6,
        poisson_ratio=0.3,
    )
    boundaries = Example2BoundarySpec(
        fluid_tags=("inlet", "outlet", "walls", "cylinder"),
        solid_tags=("structure_interface", "structure_clamp"),
        interface_tag=geometry.interface_tag,
        clamp_tag=geometry.clamp_tag,
        inlet_ramp_end_time=reference.inlet_ramp_end_time,
        inlet_modulus_ramp=reference.inlet_modulus_ramp,
        inlet_modulus_steady=reference.inlet_modulus_steady,
        time_step=reference.fluid_time_step,
        end_time=reference.end_time,
    )
    return Example2LocalSetup(
        reference=reference,
        geometry=geometry,
        material=material,
        boundaries=boundaries,
        mesh_builder_default=Path(mesh_builder_default),
        mesh_size_default=float(mesh_size_default),
        mesh_order_default=int(mesh_order_default),
    )
