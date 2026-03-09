from __future__ import annotations

from examples.biofilms.benchmarks.FSI.paper1_benchmark5_jonas_shear import _solve_one


def test_benchmark5_jonas_shear_smoke_solve() -> None:
    result = _solve_one(
        nx=4,
        qdeg=8,
        qerr=10,
        backend="cpp",
        error_backend="cpp",
        newton_tol=1.0e-10,
        max_it=20,
        profile_samples=41,
        vtk_dir=None,
    )
    row = result.row

    assert row["newton_iters"] <= 6
    assert row["v_l2"] < 5.0e-2
    assert row["p_l2"] < 2.5e-1
    assert row["u_l2"] < 8.0e-2
    assert row["alpha_l2"] < 1.2e-1
    assert row["u_interface_error"] < 8.0e-2
