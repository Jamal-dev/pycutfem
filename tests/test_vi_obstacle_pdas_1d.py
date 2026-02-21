import numpy as np

from examples.variational_inequalities import PDASOptions
from examples.variational_inequalities.obstacle_1d import solve_obstacle_1d_benchmark


def test_obstacle_1d_pdas_zero_penetration_and_dual_feasibility():
    res = solve_obstacle_1d_benchmark(
        n=250,
        f=-1.0,
        g=0.05,
        method="pdas",
        opts=PDASOptions(c=1.0, max_iter=200),
    )
    assert res.status == "converged"

    # Marker 1: zero penetration on the active set (enforced exactly by the restricted solve).
    assert res.max_active_gap <= 1.0e-12

    # Global feasibility (up to floating point): no node should go below the obstacle.
    assert res.min_gap >= -1.0e-10

    # Dual feasibility: contact forces should not be negative.
    assert res.min_lam >= -1.0e-10

    # Equilibrium should be satisfied to (near) machine precision.
    assert res.equilibrium_residual <= 1.0e-10

    # Marker 2: once the active set stabilizes, another Newton step is zero (superlinear/quadratic).
    assert res.post_stable_delta_y <= 1.0e-12

    # Basic sanity: we should be reasonably close to the analytical free boundary location.
    # (The estimate is nodal, so first-order accuracy is expected.)
    assert res.contact_a_error < 2.5e-2

    # Solution accuracy should improve with n; keep the threshold loose and robust.
    assert np.isfinite(res.l2_error)
    assert res.l2_error < 2.0e-2


def test_obstacle_1d_semismooth_newton_matches_pdas():
    res_pdas = solve_obstacle_1d_benchmark(
        n=200, f=-1.0, g=0.05, method="pdas", opts=PDASOptions(c=1.0, max_iter=200)
    )
    res_ssn = solve_obstacle_1d_benchmark(
        n=200, f=-1.0, g=0.05, method="ssn", opts=PDASOptions(c=1.0, max_iter=200)
    )
    assert res_pdas.status == "converged"
    assert res_ssn.status == "converged"

    # For the linear obstacle problem, PDAS == one semismooth Newton update.
    assert abs(res_pdas.l2_error - res_ssn.l2_error) < 1.0e-12
    assert abs(res_pdas.linf_error - res_ssn.linf_error) < 1.0e-12
    assert abs(res_pdas.contact_a_est - res_ssn.contact_a_est) < 1.0e-12
