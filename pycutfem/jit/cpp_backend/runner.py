"""Thin wrapper around the Python KernelRunner to support the C++ backend."""

from __future__ import annotations


class KernelRunnerCpp:
    """
    Wraps a compiled C++ kernel and falls back to the Python/Numba runner when
    the C++ path is incomplete. The interface mirrors KernelRunner.
    """

    def __init__(self, kernel, param_order, ir_sequence, dof_handler, fallback_runner=None):
        from pycutfem.jit import KernelRunner as _KernelRunner

        self.kernel = kernel
        self.param_order = param_order
        self._delegate = _KernelRunner(kernel, param_order, ir_sequence, dof_handler)
        self._fallback = fallback_runner

    def __call__(self, functions, static_args):
        import os
        use_native = os.getenv("PYCUTFEM_CPP_NATIVE", "").lower() in {"1", "true", "yes"}
        if use_native:
            try:
                return self._delegate(functions, static_args)
            except Exception:
                if self._fallback is None:
                    raise
        if self._fallback is not None:
            return self._fallback(functions, static_args)
        return self._delegate(functions, static_args)

    def __getattr__(self, name):
        # Provide transparent access to delegate attributes (e.g., param_order).
        if hasattr(self._delegate, name):
            return getattr(self._delegate, name)
        raise AttributeError(name)
