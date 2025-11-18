from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def test_sbdf_notebook_exec():
    """
    Execute the SBDF CutFEM notebook with a reduced configuration to ensure
    the IMEX scheme converges without raising errors.
    """
    project_root = Path(__file__).resolve().parents[1]
    nb_path = project_root / "examples/schafer/turek_benchmark_ngsolve_2.ipynb"
    assert nb_path.exists(), f"Notebook not found at {nb_path}"

    # Configure a short run for testing purposes.
    os.environ.setdefault("PYCUTFEM_SBDF_TEST", "1")
    os.environ.setdefault("PYCUTFEM_SBDF_TEST_STEPS", "2")
    os.environ.setdefault("PYCUTFEM_SBDF_TEST_SOLVER", "sbdf")
    os.environ.setdefault("PYCUTFEM_SBDF_TEST_DT", "0.02")
    os.environ.setdefault("PYCUTFEM_SBDF_TEST_CFL", "100.0")

    notebook = nbformat.read(nb_path, as_version=4)
    executor = ExecutePreprocessor(timeout=None, kernel_name="python3", allow_errors=False)
    executor.preprocess(notebook, {"metadata": {"path": str(nb_path.parent)}})


if __name__ == "__main__":
    test_sbdf_notebook_exec()
