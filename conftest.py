# conftest.py
import matplotlib
import pytest

@pytest.fixture(autouse=True)
def mpl_test_backend():
    """Switch to a non-interactive backend for all tests."""
    matplotlib.use('Agg')
