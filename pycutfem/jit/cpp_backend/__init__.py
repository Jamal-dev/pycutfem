"""
C++ backend scaffolding for pycutfem JIT kernels.

This module mirrors the public surface of the Numba-based backend so we can
swap backends via an environment switch without touching the assemblers.
"""

from __future__ import annotations

import os
from typing import Any, Sequence, Tuple


def _backend_requested() -> bool:
    """Return True when the user asked for the experimental C++ backend."""
    return os.getenv("PYCUTFEM_JIT_BACKEND", "").lower() in {"cpp", "c++"}


_CPP_KERNEL_CACHE_SINGLETON = None
_CPP_KERNEL_CACHE_DIR_TOKEN = None


def _get_cpp_kernel_cache():
    """
    Return a process-wide `CppKernelCache` instance, recreating it when the
    cache directory changes (tests often patch `CppKernelCache._CACHE_DIR` or
    `PYCUTFEM_CACHE_DIR`).
    """
    global _CPP_KERNEL_CACHE_SINGLETON, _CPP_KERNEL_CACHE_DIR_TOKEN
    from .cache import CppKernelCache

    cache_dir = CppKernelCache._resolve_cache_dir()
    if _CPP_KERNEL_CACHE_SINGLETON is None or cache_dir != _CPP_KERNEL_CACHE_DIR_TOKEN:
        _CPP_KERNEL_CACHE_SINGLETON = CppKernelCache()
        _CPP_KERNEL_CACHE_DIR_TOKEN = cache_dir
    return _CPP_KERNEL_CACHE_SINGLETON


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
    from .codegen import CppCodeGen
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

    cpp_codegen = CppCodeGen(
        mixed_element=mixed_element,
        form_rank=rank,
        on_facet=on_facet,
    )
    cache = _get_cpp_kernel_cache()

    ir_sequence = ir_generator.generate(integral_expression)
    param = getattr(ir_generator, "_param", None)
    from pycutfem.jit.ir import strip_side_metadata
    ir_sequence = strip_side_metadata(ir_sequence, on_facet=on_facet)
    from pycutfem.jit import _kernel_layout_signature

    cache_sig = _kernel_layout_signature(
        mixed_element,
        ir_sequence,
        on_facet=on_facet,
        rank=rank,
    )
    kernel, param_order, active_fields = cache.get_kernel(ir_sequence, cpp_codegen, cache_sig)

    runner = KernelRunnerCpp(kernel, param_order, ir_sequence, dof_handler, fallback_runner=None)
    try:
        runner._delegate._jit_param = param
    except Exception:
        pass
    # Expose the field ordering used by the codegen so static arg compression
    # in the solver can mirror it (avoids misaligned gdofs_map columns).
    runner.active_fields = active_fields or getattr(cpp_codegen, "active_fields", None)
    return runner, ir_sequence


def compile_backend_cpp_group(
    integral_expressions: Sequence[Any],
    dof_handler,
    mixed_element,
    *,
    on_facet: bool = False,
) -> Tuple[Any, list]:
    """
    Compile multiple integrands into one C++ kernel with a shared element/QP loop.

    The generated kernel keeps the original per-integral IR programs and merely
    concatenates them, so accumulation order matches the unfused path while loop
    overhead is paid only once.
    """
    from pycutfem.jit.visitor import IRGenerator
    from pycutfem.ufl.jit_parametrization import build_jit_parametrization
    from .codegen import CppCodeGen
    from .runner import KernelRunnerCpp

    exprs = [expr for expr in integral_expressions if expr is not None]
    if not exprs:
        raise ValueError("compile_backend_cpp_group requires at least one integrand.")

    rank = _form_rank(exprs[0])
    for expr in exprs[1:]:
        expr_rank = _form_rank(expr)
        if expr_rank != rank:
            raise ValueError(
                "compile_backend_cpp_group requires integrands with the same form rank; "
                f"got {rank} and {expr_rank}."
            )

    combined_expr = exprs[0]
    for expr in exprs[1:]:
        combined_expr = combined_expr + expr
    param = build_jit_parametrization(combined_expr)

    cpp_codegen = CppCodeGen(
        mixed_element=mixed_element,
        form_rank=rank,
        on_facet=on_facet,
    )
    cache = _get_cpp_kernel_cache()

    from pycutfem.jit.ir import strip_side_metadata

    flat_ir: list = []
    for expr in exprs:
        ir_generator = IRGenerator()
        ir_sequence = ir_generator.generate(expr, jit_param=param)
        flat_ir.extend(strip_side_metadata(ir_sequence, on_facet=on_facet))

    from pycutfem.jit import _kernel_layout_signature

    cache_sig = _kernel_layout_signature(
        mixed_element,
        flat_ir,
        on_facet=on_facet,
        rank=rank,
    )
    kernel, param_order, active_fields = cache.get_kernel(flat_ir, cpp_codegen, cache_sig)

    runner = KernelRunnerCpp(kernel, param_order, flat_ir, dof_handler, fallback_runner=None)
    try:
        runner._delegate._jit_param = param
    except Exception:
        pass
    runner.active_fields = active_fields or getattr(cpp_codegen, "active_fields", None)
    return runner, flat_ir


def _form_rank(expr):
    """Return 0 (functional), 1 (linear) or 2 (bilinear)."""
    from pycutfem.ufl.expressions import Function, VectorFunction
    from pycutfem.ufl.expressions import TrialFunction, VectorTrialFunction
    from pycutfem.ufl.expressions import TestFunction, VectorTestFunction
    from pycutfem.ufl.expressions import HdivTrialFunction, HdivTestFunction

    has_trial = expr.find_first(lambda n: isinstance(
        n, (TrialFunction, VectorTrialFunction, HdivTrialFunction))) is not None
    has_test = expr.find_first(lambda n: isinstance(
        n, (TestFunction, VectorTestFunction, HdivTestFunction))) is not None

    return 2 if (has_trial and has_test) else 1 if (has_test) else 0


__all__ = [
    "compile_backend_cpp",
    "compile_backend_cpp_group",
    "_backend_requested",
]
