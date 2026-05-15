from __future__ import annotations

import json

import numpy as np

from examples.turek_navier_stokes_mor.turek_2d3_mor import (
    _projection_cv_mode_selection,
    deviatoric_strain_matrix_2d,
    infer_current_state_arg_names,
    run_synthetic_smoke,
)


def test_turek_stress_uses_two_dimensional_deviatoric_factor() -> None:
    eps = np.array([[2.0, 0.25], [0.25, 4.0]], dtype=float)

    dev = deviatoric_strain_matrix_2d(eps)

    np.testing.assert_allclose(dev, np.array([[-1.0, 0.25], [0.25, 1.0]]))
    assert abs(float(np.trace(dev))) <= 1.0e-14


def test_turek_mor_smoke_writes_native_ready_nonaffine_rom(tmp_path) -> None:
    rom_file = run_synthetic_smoke(tmp_path)
    data = np.load(rom_file, allow_pickle=True)
    metadata = json.loads(str(data["metadata"].item()))

    trial_basis = np.asarray(data["trial_basis"], dtype=float)
    selected_basis = np.asarray(data["selected_basis"], dtype=float)
    residual_terms = np.asarray(data["residual_terms"], dtype=float)
    qdeim_rows = np.asarray(data["qdeim_rows"], dtype=np.int64)

    assert metadata["smoke"] is True
    assert trial_basis.ndim == 2
    assert qdeim_rows.size == selected_basis.shape[0]
    assert selected_basis.shape[1] == residual_terms.shape[0]
    assert residual_terms.shape[1] == trial_basis.shape[1]
    assert np.unique(qdeim_rows).size == qdeim_rows.size


def test_projection_cross_validation_selects_compact_pod_rank() -> None:
    rng = np.random.default_rng(19)
    basis, _ = np.linalg.qr(rng.normal(size=(12, 2)))
    coeffs = rng.normal(size=(2, 8))
    snapshots = basis @ coeffs + 1.0e-8 * rng.normal(size=(12, 8))

    selected, report = _projection_cv_mode_selection(
        snapshots,
        candidates=(1, 2, 4, 6),
        center=False,
        validation_fraction=0.25,
        label="synthetic",
    )

    assert selected == 2
    assert report["selected_modes"] == 2
    assert len(report["evaluated"]) >= 3


def test_native_current_state_inference_includes_pressure_field() -> None:
    names = infer_current_state_arg_names(
        ("gdofs_map", "u_k_loc", "p_k_loc", "u_n_loc"),
        ("p_k_loc", "u_k_loc", "p_n_loc"),
    )

    assert names == ("u_k_loc", "p_k_loc")
