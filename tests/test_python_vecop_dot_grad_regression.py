import numpy as np

from pycutfem.ufl.helpers import GradOpInfo, VecOpInfo


def test_vecop_dot_grad_basis_matches_grad_left_dot():
    vec_trial = VecOpInfo(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        role="trial",
        field_names=["ux", "uy"],
        parent_name="u",
        is_rhs=False,
    )
    grad_test = GradOpInfo(
        np.array(
            [
                [[10.0, 11.0], [12.0, 13.0]],
                [[20.0, 21.0], [22.0, 23.0]],
            ],
            dtype=float,
        ),
        role="test",
        field_names=["vx", "vy"],
        parent_name="v",
        is_rhs=False,
    )

    via_vec = vec_trial.dot_grad(grad_test)
    via_grad = grad_test.left_dot(vec_trial)

    np.testing.assert_allclose(via_vec.data, via_grad.data)
    assert via_vec.role == "mixed"
    assert via_vec.data.shape == (2, 2, 2)
