"""
C++ backend scaffolding for pycutfem JIT kernels.

This module mirrors the public surface of the Numba-based backend so we can
swap backends via an environment switch without touching the assemblers.
"""

from __future__ import annotations

import os
from typing import Any, Tuple


def _backend_requested() -> bool:
    """Return True when the user asked for the experimental C++ backend."""
    return os.getenv("PYCUTFEM_JIT_BACKEND", "").lower() in {"cpp", "c++"}


def compile_backend_cpp(
    integral_expression,
    dof_handler,
    mixed_element,
    *,
    on_facet: bool = False,
) -> Tuple[Any, list]:
    """
    Entry point that mirrors :func:`pycutfem.jit.compile_backend` but routes to
    the experimental C++ code generator and cache.

    The return value matches the Numba backend: (runner, ir_sequence).
    """
    from pycutfem.jit.visitor import IRGenerator
    from pycutfem.jit.codegen import NumbaCodeGen
    from .codegen import CppCodeGen
    from .cache import CppKernelCache
    from .runner import KernelRunnerCpp

    # Accept Form / Integral / plain Expression alike -----------------
    from pycutfem.ufl.measures import Integral as _Integral
    if hasattr(integral_expression, "integrals"):  # it is a Form
        if len(integral_expression.integrals) != 1:
            raise NotImplementedError("JIT expects a single-integral form.")
        integral_expression = integral_expression.integrals[0].integrand
    elif isinstance(integral_expression, _Integral):  # single Integral
        integral_expression = integral_expression.integrand

    ir_generator = IRGenerator()
    rank = _form_rank(integral_expression)

    # Reuse the Python codegen to extract param_order/active fields for now.
    numba_codegen = NumbaCodeGen(
        mixed_element=mixed_element,
        form_rank=rank,
        on_facet=on_facet,
    )
    cpp_codegen = CppCodeGen(
        mixed_element=mixed_element,
        form_rank=rank,
        on_facet=on_facet,
        mirror=numba_codegen,
    )
    cache = CppKernelCache()

    ir_sequence = ir_generator.generate(integral_expression)
    param = getattr(ir_generator, "_param", None)
    from pycutfem.jit.ir import strip_side_metadata
    ir_sequence = strip_side_metadata(ir_sequence, on_facet=on_facet)

    try:
        cache_sig = (mixed_element.signature(), bool(on_facet), int(rank))
        kernel, param_order, active_fields = cache.get_kernel(
            ir_sequence, cpp_codegen, cache_sig
        )
    except Exception as exc:
        # Fail hard so missing op coverage is fixed immediately.
        raise

    runner = KernelRunnerCpp(kernel, param_order, ir_sequence, dof_handler, fallback_runner=None)
    try:
        runner._delegate._jit_param = param
    except Exception:
        pass
    # Expose the field ordering used by the codegen so static arg compression
    # in the solver can mirror it (avoids misaligned gdofs_map columns).
    runner.active_fields = active_fields or getattr(cpp_codegen, "active_fields", None)
    return runner, ir_sequence


def _form_rank(expr):
    """Return 0 (functional), 1 (linear) or 2 (bilinear)."""
    from pycutfem.ufl.expressions import Function, VectorFunction
    from pycutfem.ufl.expressions import TrialFunction, VectorTrialFunction
    from pycutfem.ufl.expressions import TestFunction, VectorTestFunction

    has_trial = expr.find_first(lambda n: isinstance(
        n, (TrialFunction, VectorTrialFunction))) is not None
    has_test = expr.find_first(lambda n: isinstance(
        n, (TestFunction, VectorTestFunction))) is not None

    return 2 if (has_trial and has_test) else 1 if (has_test) else 0


__all__ = [
    "compile_backend_cpp",
    "_backend_requested",
]
