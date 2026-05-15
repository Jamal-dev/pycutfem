from __future__ import annotations

import numpy as np
import pytest

from pycutfem.mor import (
    NativeSparseMatrix,
    apply_gnat_lift,
    apply_sparse_gnat_lift,
    sparse_gnat_normal_equations,
)


def _have_cpp_backend() -> bool:
    try:
        import pybind11  # noqa: F401

        return True
    except Exception:
        return False


def test_native_sparse_matrix_constructors_validate_and_preserve_values() -> None:
    dense = np.array(
        [
            [1.0, 0.0, -2.0],
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
        ],
        dtype=float,
    )

    csr = NativeSparseMatrix.from_dense(dense)
    np.testing.assert_allclose(csr.to_dense(), dense)
    assert csr.nnz == 4

    coo = NativeSparseMatrix.from_coo(
        row=np.array([0, 0, 2, 2, 2]),
        col=np.array([0, 2, 0, 1, 1]),
        data=np.array([1.0, -2.0, 3.0, 1.5, 2.5]),
        shape=dense.shape,
    )
    np.testing.assert_allclose(coo.to_dense(), dense)

    csc = NativeSparseMatrix.from_csc(
        indptr=np.array([0, 2, 3, 4], dtype=np.int64),
        indices=np.array([0, 2, 2, 0], dtype=np.int64),
        data=np.array([1.0, 3.0, 4.0, -2.0], dtype=float),
        shape=dense.shape,
    )
    np.testing.assert_allclose(csc.to_dense(), dense)
    np.testing.assert_allclose(NativeSparseMatrix.from_native_dict(csr.to_native_dict()).to_dense(), dense)

    with pytest.raises(ValueError, match="strictly increasing"):
        NativeSparseMatrix.from_csr(
            np.array([0, 2], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            np.array([1.0, 2.0]),
            shape=(1, 3),
        )


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_sparse_gnat_lift_matches_dense(backend: str, tmp_path, monkeypatch) -> None:
    if backend == "cpp" and not _have_cpp_backend():
        pytest.skip("cpp backend requires pybind11")
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"sparse_lift_{backend}"))
    lift_dense = np.array(
        [
            [1.0, 0.0, -0.5, 0.0],
            [0.0, 2.0, 0.0, 0.25],
            [0.0, 0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    lift = NativeSparseMatrix.from_dense(lift_dense)
    residual = np.array([1.0, -2.0, 0.5, 3.0], dtype=float)
    trial = np.array(
        [
            [1.0, 2.0],
            [0.5, -1.0],
            [3.0, 0.25],
            [-2.0, 4.0],
        ],
        dtype=float,
    )

    lifted_residual, lifted_trial = apply_sparse_gnat_lift(lift, residual, trial, backend=backend)

    np.testing.assert_allclose(lifted_residual, lift_dense @ residual, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(lifted_trial, lift_dense @ trial, rtol=1.0e-13, atol=1.0e-13)


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_sparse_gnat_normal_equations_cpp_matches_dense(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "sparse_normal"))
    lift_dense = np.array(
        [
            [1.0, 0.0, -0.5, 0.0],
            [0.0, 2.0, 0.0, 0.25],
            [0.0, 0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    lift = NativeSparseMatrix.from_dense(lift_dense)
    residual = np.array([1.0, -2.0, 0.5, 3.0], dtype=float)
    trial = np.array(
        [
            [1.0, 2.0],
            [0.5, -1.0],
            [3.0, 0.25],
            [-2.0, 4.0],
        ],
        dtype=float,
    )
    lifted_residual = lift_dense @ residual
    lifted_trial = lift_dense @ trial

    out = sparse_gnat_normal_equations(lift, residual, trial, backend="cpp")

    np.testing.assert_allclose(out["normal_matrix"], lifted_trial.T @ lifted_trial, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(out["normal_rhs"], -(lifted_trial.T @ lifted_residual), rtol=1.0e-13, atol=1.0e-13)
    assert out["nnz"] == lift.nnz
    assert out["path"] == "csr_direct_normal"


@pytest.mark.skipif(not _have_cpp_backend(), reason="cpp backend requires pybind11")
def test_apply_gnat_lift_dispatches_sparse_cpp_backend(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "sparse_reduced_assembly_dispatch"))
    lift_dense = np.array([[1.0, 0.0, 2.0], [0.0, -1.0, 0.5]], dtype=float)
    residual = np.array([2.0, -3.0, 4.0], dtype=float)
    trial = np.array([[1.0, 0.0], [2.0, -1.0], [0.5, 3.0]], dtype=float)

    lifted_residual, lifted_trial = apply_gnat_lift(
        sample_to_residual_coefficients=NativeSparseMatrix.from_dense(lift_dense),
        sampled_residual=residual,
        sampled_trial_jacobian=trial,
        backend="cpp",
    )

    np.testing.assert_allclose(lifted_residual, lift_dense @ residual, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(lifted_trial, lift_dense @ trial, rtol=1.0e-13, atol=1.0e-13)
