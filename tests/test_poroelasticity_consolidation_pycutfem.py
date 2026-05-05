import numpy as np

from examples.poroelasticity.consolidation_pycutfem import solve_consolidation_pycutfem


def test_poroelasticity_consolidation_pycutfem_regression():
    res = solve_consolidation_pycutfem(
        output_dir=None,
        nx=9,
        ny=5,
        num_time_steps=2,
        final_time=0.6,
        t_1=3.6,
        backend="jit",
        write_csv=False,
        print_progress=False,
    )

    assert res["time"] == [0.3, 0.6]
    assert np.isfinite(res["theta_val"])

    got = np.asarray(res["p_w_max"], dtype=float)
    expected = np.asarray([4.3974624356372386e08, 4.9292445162721235e08], dtype=float)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1.0e-6, atol=0.0)
