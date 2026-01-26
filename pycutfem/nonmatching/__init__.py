"""Non-matching mesh interface coupling utilities.

This subpackage provides geometry pairing (common-refinement of two interface
edge sets) plus Nitsche/mortar coupling assemblers implemented for the
project's backends.
"""

from .interface import NonMatchingInterface, build_nonmatching_interface
from .diagnostics import (
    poisson_flux_mismatch_L2,
    scalar_jump_L2,
    stokes_traction_mismatch_L2,
    stokes_velocity_jump_L2,
)
from .mortar import MortarCoupling, assemble_mortar_saddle_matrix, assemble_poisson_mortar_coupling
from .nitsche import assemble_poisson_nitsche_interface_matrix, assemble_stokes_nitsche_interface_matrix
from .norms import scalar_H1_semi_error, scalar_L2_error
from .system import apply_dirichlet_data, coupled_dirichlet_data

__all__ = [
    "NonMatchingInterface",
    "build_nonmatching_interface",
    "scalar_jump_L2",
    "poisson_flux_mismatch_L2",
    "stokes_velocity_jump_L2",
    "stokes_traction_mismatch_L2",
    "MortarCoupling",
    "assemble_poisson_mortar_coupling",
    "assemble_mortar_saddle_matrix",
    "assemble_poisson_nitsche_interface_matrix",
    "assemble_stokes_nitsche_interface_matrix",
    "scalar_L2_error",
    "scalar_H1_semi_error",
    "apply_dirichlet_data",
    "coupled_dirichlet_data",
]
