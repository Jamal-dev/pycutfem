"""Validation-case registry for Kratos Poromechanics example parity."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PoromechanicsValidationCase:
    name: str
    kratos_path: str
    status: str
    implemented_gate: str | None
    missing_capabilities: tuple[str, ...] = ()

    @property
    def is_supported(self) -> bool:
        return self.status == "exact"


VALIDATION_CASES: dict[str, PoromechanicsValidationCase] = {
    "undrained_soil_column_2d": PoromechanicsValidationCase(
        name="Undrained soil column 2D test",
        kratos_path="applications/PoromechanicsApplication/tests/element_tests/undrained_soil_column_2D",
        status="exact",
        implemented_gate=(
            "tests/test_examples_poromechanics_utils.py::"
            "test_kratos_undrained_soil_column_2d_pressure_parity_cpp_backend"
        ),
    ),
    "four_point_shear": PoromechanicsValidationCase(
        name="Four point shear test",
        kratos_path="poromechanics/validation/arc_length_test",
        status="blocked",
        implemented_gate=None,
        missing_capabilities=(
            "arc-length nonlinear strategy",
            "nonlocal damage constitutive update",
            "triangular UPwSmallStrainElement2D3N parity",
        ),
    ),
    "vertical_fault_consolidation": PoromechanicsValidationCase(
        name="Consolidation test with a vertical fault",
        kratos_path="poromechanics/validation/consolidation_interface_2D",
        status="partial",
        implemented_gate=(
            "tests/test_examples_poromechanics_utils.py::"
            "test_kratos_consolidation_interface_2d_pressure_parity_all_backends"
        ),
        missing_capabilities=("full long validation-case mdpa import and output comparison",),
    ),
    "consolidation_interface_2d": PoromechanicsValidationCase(
        name="Consolidation interface 2D",
        kratos_path="applications/PoromechanicsApplication/tests/element_tests/consolidation_interface_2D",
        status="exact",
        implemented_gate=(
            "tests/test_examples_poromechanics_utils.py::"
            "test_kratos_consolidation_interface_2d_pressure_parity_all_backends"
        ),
    ),
    "fracture_network_flow": PoromechanicsValidationCase(
        name="Fluid flow in pre-existing fractures network",
        kratos_path="poromechanics/use_cases/fluid_pumping_2D",
        status="partial",
        implemented_gate=(
            "tests/test_examples_poromechanics_utils.py::"
            "test_kratos_fic_triangle_bulk_pressure_gradient_stabilization_all_backends; "
            "tests/test_examples_poromechanics_utils.py::"
            "test_upl_link_interface_ufl_uses_kratos_link_permeability_law; "
            "tests/test_examples_poromechanics_utils.py::"
            "test_kratos_normal_liquid_flux_rhs_all_backends; "
            "tests/test_examples_poromechanics_utils.py::"
            "test_kratos_fluid_pumping_2d_reference_runner_first_step; "
            "tests/test_examples_poromechanics_utils.py::"
            "test_kratos_fluid_pumping_2d_pycutfem_cpp_smoke; "
            "tests/test_examples_poromechanics_utils.py::"
            "test_kratos_fluid_pumping_2d_first_newton_increment_parity_cpp_backend"
        ),
        missing_capabilities=(
            "full Kratos Poromechanics nonlinear scheme-state sequencing for the imported fluid_pumping_2D network",
            "strict converged first-step pressure/displacement parity at the pump and far-field sample nodes",
            "strict first-step and full fluid_pumping_2D 100-step output parity",
        ),
    ),
    "fluid_driven_fracture_propagation": PoromechanicsValidationCase(
        name="Fluid driven fracture propagation test",
        kratos_path="poromechanics/use_cases/fluid_pumping_2D_fracture",
        status="blocked",
        implemented_gate=None,
        missing_capabilities=(
            "fracture propagation/insertion utility",
            "FIC normal-flux boundary stabilization",
            "full fracture example mdpa pairing and output comparison",
        ),
    ),
}


def unsupported_validation_cases() -> dict[str, PoromechanicsValidationCase]:
    return {key: case for key, case in VALIDATION_CASES.items() if not case.is_supported}


def require_validation_case_supported(key: str) -> PoromechanicsValidationCase:
    case = VALIDATION_CASES[str(key)]
    if not case.is_supported:
        missing = ", ".join(case.missing_capabilities) or "no implementation gate"
        raise NotImplementedError(f"{case.name} is not supported yet: {missing}.")
    return case
