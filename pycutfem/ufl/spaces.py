"""Symbolic function-space declarations for UFL-style APIs."""

from __future__ import annotations

from typing import Iterable


class FunctionSpace:
    """Lightweight symbolic function-space descriptor.

    Parameters
    ----------
    name:
        User-facing space name (e.g. ``"velocity"``).
    field_names:
        Ordered scalar field components (e.g. ``["u_x", "u_y"]``).
    dim:
        Symbolic rank marker used by expression constructors.
    side:
        Optional side tag (``"+"`` / ``"-"``) for interface-restricted spaces.
    """

    def __init__(self, name: str, field_names: Iterable[str], dim: int = 0, side: str = ""):
        fields = tuple(str(f) for f in field_names)
        if not fields:
            raise ValueError("FunctionSpace.field_names must not be empty.")
        self.name = str(name)
        self.field_names = list(fields)
        self.dim = int(dim)
        self.side = str(side)

    def __repr__(self) -> str:
        return (
            f"FunctionSpace(name={self.name!r}, field_names={self.field_names!r}, "
            f"dim={self.dim}, side={self.side!r})"
        )
