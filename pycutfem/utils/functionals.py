"""Scalar functional evaluation helpers.

These utilities are intentionally lightweight: they compile rank-0 (functional)
forms once and then allow re-evaluation for different coefficient values,
optionally refreshing precomputed geometry for moving level sets.

Typical use cases:
  - energies (∫_Ω ... dx) for diagnostics / quantities of interest
  - interface forces (∫_Γ ... ds) in moving-interface problems
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

import numpy as np


def _as_backend(backend: str) -> str:
    b = str(backend or "").strip().lower()
    if b in {"c++"}:
        return "cpp"
    return b or "jit"


def _iter_integrals(form) -> Iterable:
    from pycutfem.ufl.forms import Equation, Form
    from pycutfem.ufl.measures import Integral

    if isinstance(form, Equation):
        ints = []
        if form.a is not None:
            ints += list(getattr(form.a, "integrals", []))
        if form.L is not None:
            ints += list(getattr(form.L, "integrals", []))
        return ints
    if isinstance(form, Integral):
        return [form]
    if isinstance(form, Form):
        return list(getattr(form, "integrals", []))
    # Fall back: allow "Expression with .integrals" like Form
    return list(getattr(form, "integrals", []))


@dataclass
class ScalarFunctionalEvaluator:
    """Compile and evaluate a scalar functional on a chosen backend."""

    form: object
    dof_handler: object
    mixed_element: object
    backend: str = "jit"
    quad_order: int | None = None

    def __post_init__(self) -> None:
        self.backend = _as_backend(self.backend)
        self._kernels = None
        self._eq = None
        self._hooks = None

        if self.backend in {"jit", "cpp"}:
            from pycutfem.jit import compile_multi

            self._kernels = compile_multi(
                self.form,
                dof_handler=self.dof_handler,
                mixed_element=self.mixed_element,
                quad_order=self.quad_order,
                backend=self.backend,
            )
            return

        if self.backend == "python":
            from pycutfem.ufl.forms import Equation

            self._eq = Equation(None, self.form)
            # Python backend requires assembler hooks for pure functionals.
            self._hooks = {intg.integrand: {"name": "J"} for intg in _iter_integrals(self.form)}
            return

        raise ValueError(f"Unsupported backend '{self.backend}'. Expected python|jit|cpp.")

    def refresh_levelset(self, level_set: object) -> None:
        """Refresh precomputed geometry for kernels depending on `level_set`."""
        if self._kernels is None:
            return
        for ker in self._kernels:
            if getattr(ker, "level_set", None) is level_set:
                ker.refresh(level_set)

    def evaluate(self, coeffs: Dict[str, Any]) -> float:
        """Evaluate the functional for the given coefficient mapping."""
        if self._kernels is not None:
            total = 0.0
            for ker in self._kernels:
                _, _, J_loc = ker.exec(coeffs)
                if J_loc is None:
                    continue
                total += float(np.asarray(J_loc, dtype=float).sum())
            return float(total)

        assert self._eq is not None
        assert self._hooks is not None
        from pycutfem.ufl.forms import assemble_form

        res = assemble_form(
            self._eq,
            dof_handler=self.dof_handler,
            bcs=[],
            quad_order=self.quad_order,
            backend="python",
            assembler_hooks=self._hooks,
        )
        return float(np.asarray(res.get("J", 0.0), dtype=float).reshape(-1)[0])


@dataclass
class NamedFunctionalEvaluator:
    """
    Evaluate multiple named scalar functionals that share the same DOFHandler.

    Parameters
    ----------
    forms:
        Mapping name -> (rank-0) Form/Integral.
    """

    forms: Dict[str, object]
    dof_handler: object
    mixed_element: object
    backend: str = "jit"
    quad_order: int | None = None

    def __post_init__(self) -> None:
        self.backend = _as_backend(self.backend)
        self._kernels = None
        self._eq = None
        self._hooks = None
        self._id_to_name: Dict[int, str] = {}

        if not self.forms:
            raise ValueError("NamedFunctionalEvaluator requires a non-empty 'forms' mapping.")

        combined = None
        for f in self.forms.values():
            combined = f if combined is None else (combined + f)
        self._combined = combined

        if self.backend in {"jit", "cpp"}:
            from pycutfem.jit import compile_multi

            self._kernels = compile_multi(
                self._combined,
                dof_handler=self.dof_handler,
                mixed_element=self.mixed_element,
                quad_order=self.quad_order,
                backend=self.backend,
            )
            for name, f in self.forms.items():
                for intg in _iter_integrals(f):
                    self._id_to_name[int(id(intg))] = str(name)
            return

        if self.backend == "python":
            from pycutfem.ufl.forms import Equation

            self._eq = Equation(None, self._combined)
            hooks = {}
            for name, f in self.forms.items():
                for intg in _iter_integrals(f):
                    hooks[intg.integrand] = {"name": str(name)}
            self._hooks = hooks
            return

        raise ValueError(f"Unsupported backend '{self.backend}'. Expected python|jit|cpp.")

    def refresh_levelset(self, level_set: object) -> None:
        if self._kernels is None:
            return
        for ker in self._kernels:
            if getattr(ker, "level_set", None) is level_set:
                ker.refresh(level_set)

    def evaluate(self, coeffs: Dict[str, Any]) -> Dict[str, float]:
        out = {str(k): 0.0 for k in self.forms.keys()}
        if self._kernels is not None:
            for ker in self._kernels:
                _, _, J_loc = ker.exec(coeffs)
                if J_loc is None:
                    continue
                name = self._id_to_name.get(int(getattr(ker, "integral_id", 0)), None)
                if name is None:
                    continue
                out[name] += float(np.asarray(J_loc, dtype=float).sum())
            return out

        assert self._eq is not None
        assert self._hooks is not None
        from pycutfem.ufl.forms import assemble_form

        res = assemble_form(
            self._eq,
            dof_handler=self.dof_handler,
            bcs=[],
            quad_order=self.quad_order,
            backend="python",
            assembler_hooks=self._hooks,
        )
        for k in out:
            out[k] = float(np.asarray(res.get(k, 0.0), dtype=float).reshape(-1)[0])
        return out
