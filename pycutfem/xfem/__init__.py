"""
XFEM / AgFEM utilities.

Phase 1 delivers:
- XFEMDofHandler: dynamic enriched-DOF bookkeeping (no mesh topology changes)
- AgFEMMapper:   ghost→root aggregation mapping and constraint construction
"""

from .dofhandler import XFEMDofHandler
from .agfem import AgFEMMapper
from .enrichment import alpha_from_side_masks, build_alpha_by_field
from .mixedelement import XFEMMixedElement
from .projection import l2_project_moving_interface

__all__ = [
    "XFEMDofHandler",
    "XFEMMixedElement",
    "AgFEMMapper",
    "alpha_from_side_masks",
    "build_alpha_by_field",
    "l2_project_moving_interface",
]
