import os


def _setdefault_env(name: str, value: str) -> None:
    if name not in os.environ:
        os.environ[name] = value


# Keep pytest runs responsive and avoid oversubscription: most unit tests are
# small and do not benefit from multi-threaded BLAS/OpenMP overhead.
#
# Users can override any of these by setting them explicitly in their environment
# before invoking pytest.
_setdefault_env("OMP_NUM_THREADS", "1")
_setdefault_env("OPENBLAS_NUM_THREADS", "1")
_setdefault_env("MKL_NUM_THREADS", "1")
_setdefault_env("NUMEXPR_NUM_THREADS", "1")
_setdefault_env("VECLIB_MAXIMUM_THREADS", "1")

# C++ kernel compilation can dominate the test suite. Default to fast-compile
# mode under pytest (can be overridden with PYCUTFEM_CPP_FAST_COMPILE=0).
_setdefault_env("PYCUTFEM_CPP_FAST_COMPILE", "1")

