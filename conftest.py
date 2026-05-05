import pytest

try:
    import matplotlib
except ModuleNotFoundError:  # pragma: no cover
    matplotlib = None

@pytest.fixture(autouse=True)
def mpl_test_backend():
    """Switch to a non-interactive backend for all tests."""
    if matplotlib is not None:
        matplotlib.use("Agg")
