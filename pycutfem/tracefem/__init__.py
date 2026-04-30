"""Trace-FEM integration domains.

This package contains first-class integration objects for discrete trace/link
entities.  These are element-like line domains whose FE traces are supplied by
explicit station or quadrature tables, rather than by intersecting a mesh facet
or building a non-matching common refinement.
"""

from .fracture import (
    TraceFractureExtensionPlan2D,
    TraceFractureNetwork2D,
    TraceFracturePropagationSettings2D,
    TraceStationEntity2D,
    plan_fracture_extensions_from_damage,
    transfer_trace_quadrature_state,
)
from .interface import TraceLinkInterface

__all__ = [
    "TraceFractureExtensionPlan2D",
    "TraceFractureNetwork2D",
    "TraceFracturePropagationSettings2D",
    "TraceLinkInterface",
    "TraceStationEntity2D",
    "plan_fracture_extensions_from_damage",
    "transfer_trace_quadrature_state",
]
