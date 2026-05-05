from __future__ import annotations

import numpy as np

from examples.NIRB.dvms.state import FluidDVMSState
from examples.NIRB.debug.build_local_checkpoint_from_step_history import _rebuild_iqn_history_from_monitor
from examples.NIRB.run_example2_local import (
    _fluid_dvms_restart_snapshot_from_payload,
    _restore_fluid_dvms_state,
)


def _single_point_dvms_state() -> FluidDVMSState:
    return FluidDVMSState(
        sample_coords=np.asarray([[0.25, 0.25]], dtype=float),
        sample_element_ids=np.asarray([0], dtype=int),
        sample_ref_coords=np.asarray([[1.0 / 3.0, 1.0 / 3.0]], dtype=float),
        sample_ref_weights=np.asarray([0.5], dtype=float),
        quadrature_order=1,
        cell_type="triangle",
        old_subscale_velocity=np.zeros((1, 2), dtype=float),
        predicted_subscale_velocity=np.zeros((1, 2), dtype=float),
        momentum_projection=np.zeros((1, 2), dtype=float),
        mass_projection=np.zeros((1,), dtype=float),
        old_mass_residual=np.zeros((1,), dtype=float),
    )


def test_dvms_restart_snapshot_restores_nodal_projection_buffers() -> None:
    state = _single_point_dvms_state()
    payload = {
        "dvms_old_subscale_velocity": np.asarray([[1.0, 2.0]], dtype=float),
        "dvms_predicted_subscale_velocity": np.asarray([[3.0, 4.0]], dtype=float),
        "dvms_momentum_projection": np.asarray([[5.0, 6.0]], dtype=float),
        "dvms_mass_projection": np.asarray([7.0], dtype=float),
        "dvms_old_mass_residual": np.asarray([8.0], dtype=float),
        "dvms_nodal_momentum_projection": np.asarray([[9.0, 10.0], [11.0, 12.0]], dtype=float),
        "dvms_nodal_div_projection": np.asarray([13.0, 14.0], dtype=float),
        "dvms_prev_nodal_div_projection": np.asarray([15.0, 16.0], dtype=float),
    }

    snapshot = _fluid_dvms_restart_snapshot_from_payload(state, payload)
    _restore_fluid_dvms_state(state, snapshot)

    np.testing.assert_allclose(state.old_subscale_velocity, [[1.0, 2.0]])
    np.testing.assert_allclose(state.predicted_subscale_velocity, [[3.0, 4.0]])
    np.testing.assert_allclose(state.momentum_projection, [[5.0, 6.0]])
    np.testing.assert_allclose(state.mass_projection, [7.0])
    np.testing.assert_allclose(state.old_mass_residual, [8.0])
    np.testing.assert_allclose(state._nodal_momentum_projection, [[9.0, 10.0], [11.0, 12.0]])
    np.testing.assert_allclose(state._nodal_div_projection, [13.0, 14.0])
    np.testing.assert_allclose(state._prev_nodal_div_projection, [15.0, 16.0])


def test_iqn_restart_history_ignores_converged_after_iteration_record(tmp_path) -> None:
    coords = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    for iteration in range(1, 4):
        previous = np.full((4,), float(iteration), dtype=float)
        current = previous + 0.25
        np.savez(
            tmp_path / f"step0001_iter{iteration:04d}_pre_update.npz",
            fluid_load_coords_ref=coords,
            crit_0_fluid_load_previous=previous,
            crit_0_fluid_load_current=current,
        )

    np.savez(
        tmp_path / "step0001_iter0004_after_iteration.npz",
        fluid_load_coords_ref=coords,
        crit_0_fluid_load_previous=np.full((4,), 4.0, dtype=float),
        crit_0_fluid_load_current=np.full((4,), 4.25, dtype=float),
    )

    dr_hist, dg_hist = _rebuild_iqn_history_from_monitor(
        monitor_dir=tmp_path,
        target_step=1,
        force_history=3,
        iteration_horizon=50,
        fluid_iface_coords_local=coords,
    )

    assert len(dr_hist) == 1
    assert len(dg_hist) == 1
    assert dr_hist[0].shape == (4, 2)
    assert dg_hist[0].shape == (4, 2)
