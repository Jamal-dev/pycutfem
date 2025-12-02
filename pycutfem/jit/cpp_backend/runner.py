"""Thin wrapper around the Python KernelRunner to support the C++ backend."""

from __future__ import annotations


class KernelRunnerCpp:
    """
    Wraps a compiled C++ kernel. No silent fallback to Python/Numba to ensure
    failures are visible. The interface mirrors KernelRunner.
    """

    def __init__(self, kernel, param_order, ir_sequence, dof_handler, fallback_runner=None):
        from pycutfem.jit import KernelRunner as _KernelRunner

        self.kernel = kernel
        self.param_order = param_order
        self._delegate = _KernelRunner(kernel, param_order, ir_sequence, dof_handler)
        self._fallback = fallback_runner  # kept for API compatibility; unused

    def __call__(self, functions, static_args):
        import os, sys
        from pathlib import Path

        try:
            return self._delegate(functions, static_args)
        except Exception as exc:
            # Persist failing kernel info to aid targeted cleanup.
            mod_name = getattr(self.kernel, "__module__", "")
            mod_file = ""
            if mod_name in sys.modules:
                mod_file = getattr(sys.modules[mod_name], "__file__", "") or ""
            cache_dir = Path(
                os.environ.get("PYCUTFEM_CACHE_DIR", Path.home() / ".cache" / "pycutfem_jit")
            ).expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)
            fail_log = cache_dir / "failed_kernels.txt"
            try:
                with fail_log.open("a", encoding="utf-8") as fh:
                    fh.write(f"{mod_name}|{mod_file}|{exc}\n")
            except OSError:
                pass
            raise

    def __getattr__(self, name):
        # Provide transparent access to delegate attributes (e.g., param_order).
        if hasattr(self._delegate, name):
            return getattr(self._delegate, name)
        raise AttributeError(name)
