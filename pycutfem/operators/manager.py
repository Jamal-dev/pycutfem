from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import RuntimeOperator


class OperatorManager:
    """Compose runtime operators and run their hooks in a stable order."""

    def __init__(self, operators: Sequence[RuntimeOperator] | None = None) -> None:
        self.operators = tuple(operators or ())

    def bind(self, solver: Any) -> None:
        for operator in self.operators:
            operator.bind(solver)

    def on_step_begin(self, **kwargs) -> None:
        for operator in self.operators:
            operator.on_step_begin(**kwargs)

    def before_assembly(self, **kwargs) -> None:
        for operator in self.operators:
            operator.before_assembly(**kwargs)

    def after_assembly(self, **kwargs):
        A_red = kwargs.get("A_red")
        R_red = kwargs.get("R_red")
        base_kwargs = {key: value for key, value in kwargs.items() if key not in {"A_red", "R_red"}}
        for operator in self.operators:
            result = operator.after_assembly(**base_kwargs, A_red=A_red, R_red=R_red)
            if result is not None:
                A_red, R_red = result
        return A_red, R_red

    def on_step_accept(self, **kwargs) -> None:
        for operator in self.operators:
            operator.on_step_accept(**kwargs)

    def on_step_reject(self, **kwargs) -> None:
        for operator in self.operators:
            operator.on_step_reject(**kwargs)
