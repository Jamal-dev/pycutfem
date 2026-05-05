import numpy as np


def test_pre_tabulates_q1_values_at_center():
    from pycutfem.integration import pre_tabulates as pt

    xi = np.asarray([[0.0]])
    eta = np.asarray([[0.0]])
    N = np.zeros((1, 1, 4), dtype=float)
    dN = np.zeros((1, 1, 4, 2), dtype=float)
    pt._tabulate_q1(xi, eta, N, dN)

    assert np.allclose(N[0, 0, :], 0.25)
    # Sum of bilinear shape functions equals 1.
    assert abs(float(np.sum(N[0, 0, :])) - 1.0) < 1.0e-14

    # Derivative tabulation is required even without numba.
    out = np.zeros((1, 1, 4), dtype=float)
    pt._tabulate_deriv_q1(xi, eta, 1, 0, out)
    assert np.allclose(out[0, 0, :], np.asarray([-0.25, 0.25, -0.25, 0.25]))


def test_pre_tabulates_q2_values_at_center():
    from pycutfem.integration import pre_tabulates as pt

    xi = np.asarray([[0.0]])
    eta = np.asarray([[0.0]])
    N = np.zeros((1, 1, 9), dtype=float)
    dN = np.zeros((1, 1, 9, 2), dtype=float)
    pt._tabulate_q2(xi, eta, N, dN)

    assert abs(float(N[0, 0, 4]) - 1.0) < 1.0e-14
    assert abs(float(np.sum(N[0, 0, :])) - 1.0) < 1.0e-14


def test_pre_tabulates_p1_values_and_derivatives():
    from pycutfem.integration import pre_tabulates as pt

    xi = np.asarray([[0.2]])
    eta = np.asarray([[0.3]])
    N = np.zeros((1, 1, 3), dtype=float)
    dN = np.zeros((1, 1, 3, 2), dtype=float)
    pt._tabulate_p1(xi, eta, N, dN)

    assert np.allclose(N[0, 0, :], np.asarray([0.5, 0.2, 0.3]))
    assert np.allclose(dN[0, 0, :, :], np.asarray([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]))

    out = np.zeros((1, 1, 3), dtype=float)
    pt._tabulate_deriv_p1(xi, eta, 1, 0, out)
    assert np.allclose(out[0, 0, :], np.asarray([-1.0, 1.0, 0.0]))
