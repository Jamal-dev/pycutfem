from __future__ import annotations

from typing import Any


class RuntimeOperator:
    """
    Solver-time hook interface for runtime-local operator logic.

    Operators are backend-neutral: they update runtime data before assembly,
    optionally post-process the reduced matrix/residual after assembly, and can
    observe accepted or rejected pseudo-time steps.
    """

    def bind(self, solver: Any) -> None:
        """Called once after the operator is attached to a solver."""

    def on_step_begin(
        self,
        *,
        solver: Any,
        functions,
        prev_functions,
        aux_functions,
        step: int,
        step_no: int,
        t: float,
        dt: float,
        bcs,
    ) -> None:
        """Called once before the Newton loop of a time step begins."""

    def before_assembly(self, *, solver: Any, coeffs, need_matrix: bool) -> None:
        """Called before each reduced residual/Jacobian assembly."""

    def after_assembly(self, *, solver: Any, coeffs, A_red, R_red, need_matrix: bool):
        """Called after each reduced residual/Jacobian assembly."""
        return A_red, R_red

    def on_step_accept(
        self,
        *,
        solver: Any,
        functions,
        prev_functions,
        aux_functions,
        step: int,
        step_no: int,
        t: float,
        dt: float,
        bcs,
    ) -> None:
        """Called after a pseudo-time step has been accepted."""

    def on_step_reject(
        self,
        *,
        solver: Any,
        functions,
        prev_functions,
        aux_functions,
        step: int,
        step_no: int,
        t: float,
        dt: float,
        bcs,
        exception,
        reason: str | None,
    ) -> None:
        """Called whenever a pseudo-time step attempt is rejected."""
