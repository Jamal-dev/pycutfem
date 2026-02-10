"""Compatibility shim for JIT kernel-argument helpers.

Canonical location: ``pycutfem.jit.kernel_args``.
"""

import pycutfem.jit.kernel_args as _kernel_args

# Re-export everything (including leading-underscore helper names) so legacy
# imports keep working unchanged.
for _name in dir(_kernel_args):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_kernel_args, _name)

__all__ = [n for n in dir(_kernel_args) if not n.startswith("__")]
