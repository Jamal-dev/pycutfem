import numpy as np
import pytest

from pycutfem.ufl.helpers import GradOpInfo, VecOpInfo


def _make_function_grad(matrix):
    """
    Construct a GradOpInfo that represents a 2×2 tensor-valued function
    gradient.  The data layout (k, d) matches the internal storage used once
    the basis has been evaluated at a quadrature point and collapsed.
    """
    mat = np.asarray(matrix, dtype=float)
    if mat.shape != (2, 2):
        raise ValueError("Only 2x2 tensors are supported in this helper.")
    return GradOpInfo(mat, role="function", is_rhs=False)


def _make_basis_grad(matrix, role):
    """
    Construct a GradOpInfo for a test or trial basis gradient.  The data layout
    for LHS assembly is (k, n, d); we encode a single local dof (n = 1) so that
    the result is easy to reason about.
    """
    mat = np.asarray(matrix, dtype=float)
    if mat.shape != (2, 2):
        raise ValueError("Only 2x2 tensors are supported in this helper.")
    data = mat[:, None, :]  # (k, 1, d)
    return GradOpInfo(data, role=role, is_rhs=False)


def _extract_matrix_from_basis(grad_op):
    """
    Convert a GradOpInfo with shape (k, 1, d) back into a 2×2 matrix.
    """
    data = np.asarray(grad_op.data)
    if data.shape != (2, 1, 2):
        raise ValueError(f"Unexpected gradient shape {data.shape}")
    return np.stack([data[0, 0, :], data[1, 0, :]], axis=0)


@pytest.mark.parametrize("trial_mat,test_mat", [
    ([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]),
    ([[2.5, -1.0], [0.0, 4.0]], [[-3.0, 0.5], [1.2, -2.2]])
])
def test_inner_trial_test_matches_frobenius(trial_mat, test_mat):
    trial = _make_basis_grad(trial_mat, role="trial")
    test = _make_basis_grad(test_mat, role="test")

    result = test.inner(trial)  # shape (n_test, n_trial) => (1, 1)

    expected = np.sum(np.asarray(trial_mat) * np.asarray(test_mat))
    assert result.shape == (1, 1)
    np.testing.assert_allclose(result[0, 0], expected)


def test_inner_function_test_matches_componentwise_dot():
    finv = _make_function_grad([[1.0, 2.0], [3.0, 4.0]])
    grad_test = _make_basis_grad([[5.0, 6.0], [7.0, 8.0]], role="test")

    result = finv.inner(grad_test)  # returns length-1 vector for n=1

    expected = (np.asarray(finv.data) * np.asarray(grad_test.data[:, 0, :])).sum()
    assert result.shape == (1,)
    np.testing.assert_allclose(result[0], expected)


def test_function_test_dot_preserves_matrix_structure():
    finv = _make_function_grad([[1.0, 2.0], [3.0, 4.0]])
    grad_test = _make_basis_grad([[5.0, 6.0], [7.0, 8.0]], role="test")

    result = finv.dot(grad_test)

    assert isinstance(result, GradOpInfo)
    assert result.role == "test"

    result_matrix = _extract_matrix_from_basis(result)
    expected = np.asarray(finv.data) @ np.asarray(grad_test.data[:, 0, :])
    np.testing.assert_allclose(result_matrix, expected)


def test_scalar_trial_times_function_gradient_stays_gradient_valued():
    trial = VecOpInfo(np.array([[2.0, -1.0, 0.5]], dtype=float), role="trial", is_rhs=False)
    grad_fun = GradOpInfo(np.array([[3.0, -4.0]], dtype=float), role="function", is_rhs=False)

    result = trial * grad_fun

    assert isinstance(result, GradOpInfo)
    assert result.role == "trial"
    expected = np.array(
        [[[6.0, -8.0], [-3.0, 4.0], [1.5, -2.0]]],
        dtype=float,
    )
    np.testing.assert_allclose(result.data, expected)


def test_scalar_function_times_function_gradient_stays_gradient_valued():
    coeff = VecOpInfo(np.array([2.5], dtype=float), role="function", is_rhs=False)
    grad_fun = GradOpInfo(np.array([[3.0, -4.0]], dtype=float), role="function", is_rhs=False)

    result = coeff * grad_fun

    assert isinstance(result, GradOpInfo)
    assert result.role == "function"
    np.testing.assert_allclose(result.data, np.array([[7.5, -10.0]], dtype=float))
