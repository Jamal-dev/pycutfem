import numpy as np

from pycutfem.mor.state_updates import apply_affine_state_updates, build_dirichlet_lift_state_updates


def test_build_dirichlet_lift_state_updates_overwrites_fixed_rows_only() -> None:
    gdofs = np.asarray([[0, 1, 2], [2, 3, -1]], dtype=np.int32)
    static_args = {
        "gdofs_map": gdofs,
        "u_state_loc": np.zeros(gdofs.shape, dtype=np.float64),
    }
    trial_basis = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=np.float64,
    )
    offset = np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    lift = np.asarray([100.0, 200.0, 300.0, 400.0], dtype=np.float64)

    updates = build_dirichlet_lift_state_updates(
        static_args,
        ("u_state_loc",),
        trial_basis=trial_basis,
        offset=offset,
        fixed_rows=np.asarray([1, 3], dtype=np.int64),
        lift_values=lift,
    )

    assert len(updates) == 1
    values = apply_affine_state_updates(updates, np.asarray([2.0, 5.0], dtype=np.float64))["u_state_loc"]
    np.testing.assert_allclose(
        values.reshape(gdofs.shape),
        np.asarray(
            [
                [12.0, 200.0, 34.0],
                [34.0, 400.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )
