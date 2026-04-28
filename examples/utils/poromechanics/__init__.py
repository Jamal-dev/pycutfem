"""Example-level 2D poromechanics helpers.

These utilities intentionally live under ``examples/utils``. They are shared
building blocks for examples and Kratos-parity experiments, not pycutfem core
APIs.
"""

from .materials import UPlMaterial2D
from .interface import (
    PairedUPlInterface2D,
    PairedUPlInterfaceLocalSystem2D,
    PairedUPlInterfaceNewtonBatch2D,
    UPlInterfaceUFLSystem2D,
    UPlInterfaceMaterial2D,
    assemble_paired_upl_interface_2d,
    build_paired_upl_interface_kratos_newton_batch_2d,
    build_paired_upl_interface_local_batch_2d,
    build_paired_upl_interface_local_system_2d,
    build_paired_upl_interface_operator_2d,
    build_upl_interface_ufl_system_2d,
    build_upl_link_interface_ufl_system_2d,
    paired_upl_interface_to_nonmatching_interface_2d,
)
from .kratos_parity import (
    FLUID_PUMPING_2D_SAMPLE_NODE_IDS,
    KratosConsolidation2DResult,
    KratosFluidPumping2DResult,
    build_kratos_consolidation_2d_mesh,
    build_kratos_consolidation_interface_2d_mesh,
    solve_kratos_consolidation_2d_pycutfem,
    solve_kratos_consolidation_interface_2d_pycutfem,
    solve_kratos_fluid_pumping_2d_pycutfem,
    solve_kratos_fluid_pumping_2d_reference,
    solve_kratos_undrained_soil_column_2d_pycutfem,
)
from .upl import (
    UPlKratosQuasistaticSystem2D,
    UPlKratosFICQuasistaticSystem2D,
    UPlThetaSystem2D,
    build_kratos_fic_triangle_upl_system_2d,
    build_kratos_quasistatic_upl_system_2d,
    build_upl_theta_system_2d,
    displacement_neumann_rhs,
    effective_stress_linear_2d,
    epsilon_2d,
    hydraulic_conductivity_form_2d,
    kratos_fic_triangle_element_length_squared,
    normal_liquid_flux_rhs_2d,
)
from .validation_cases import (
    VALIDATION_CASES,
    PoromechanicsValidationCase,
    require_validation_case_supported,
    unsupported_validation_cases,
)

__all__ = [
    "UPlMaterial2D",
    "UPlInterfaceMaterial2D",
    "UPlInterfaceUFLSystem2D",
    "PairedUPlInterface2D",
    "PairedUPlInterfaceLocalSystem2D",
    "PairedUPlInterfaceNewtonBatch2D",
    "KratosConsolidation2DResult",
    "KratosFluidPumping2DResult",
    "UPlKratosQuasistaticSystem2D",
    "UPlKratosFICQuasistaticSystem2D",
    "UPlThetaSystem2D",
    "FLUID_PUMPING_2D_SAMPLE_NODE_IDS",
    "assemble_paired_upl_interface_2d",
    "build_paired_upl_interface_kratos_newton_batch_2d",
    "build_paired_upl_interface_local_batch_2d",
    "build_paired_upl_interface_local_system_2d",
    "build_paired_upl_interface_operator_2d",
    "build_upl_interface_ufl_system_2d",
    "build_upl_link_interface_ufl_system_2d",
    "paired_upl_interface_to_nonmatching_interface_2d",
    "build_kratos_consolidation_2d_mesh",
    "build_kratos_fic_triangle_upl_system_2d",
    "build_kratos_consolidation_interface_2d_mesh",
    "build_kratos_quasistatic_upl_system_2d",
    "build_upl_theta_system_2d",
    "displacement_neumann_rhs",
    "effective_stress_linear_2d",
    "epsilon_2d",
    "hydraulic_conductivity_form_2d",
    "kratos_fic_triangle_element_length_squared",
    "normal_liquid_flux_rhs_2d",
    "solve_kratos_consolidation_2d_pycutfem",
    "solve_kratos_consolidation_interface_2d_pycutfem",
    "solve_kratos_fluid_pumping_2d_pycutfem",
    "solve_kratos_fluid_pumping_2d_reference",
    "solve_kratos_undrained_soil_column_2d_pycutfem",
    "PoromechanicsValidationCase",
    "VALIDATION_CASES",
    "require_validation_case_supported",
    "unsupported_validation_cases",
]
