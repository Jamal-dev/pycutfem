"""
Small variational-inequality (VI) utilities and benchmarks.

These live in `examples/` on purpose so we can validate algorithms (PDAS /
semismooth Newton) in isolation before integrating them into contact mechanics.
"""

from .pdas import PDASOptions, PDASResult, solve_obstacle_pdas, solve_obstacle_semismooth_newton

__all__ = [
    "PDASOptions",
    "PDASResult",
    "solve_obstacle_pdas",
    "solve_obstacle_semismooth_newton",
]
