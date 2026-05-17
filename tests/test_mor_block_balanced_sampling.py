from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from pycutfem.mor import (
    NativeKernelReference,
    NativeReducedArtifact,
    augment_rows_for_residual_norm_equivalence,
    build_block_balanced_gnat_sampling,
    certify_sampled_residual_norm_equivalence,
    field_row_blocks,
    load_native_reduced_artifact,
    rows_supported_on_elements,
    select_coordinate_band_elements,
    support_element_ids_from_rows,
)


@dataclass
class _FakeMixedElement:
    mesh: object


class _FakeDofHandler:
    def __init__(self) -> None:
        self.total_dofs = 18
        self.field_names = ["a", "b", "c"]
        self._field_slices = {
            "a": np.arange(0, 6, dtype=np.int64),
            "b": np.arange(6, 12, dtype=np.int64),
            "c": np.arange(12, 18, dtype=np.int64),
        }
        self.element_maps = {
            "a": [
                np.array([0, 1]),
                np.array([1, 2]),
                np.array([2, 3]),
                np.array([3, 4]),
            ],
            "b": [
                np.array([6, 7]),
                np.array([7, 8]),
                np.array([8, 9]),
                np.array([9, 10]),
            ],
            "c": [
                np.array([12, 13]),
                np.array([13, 14]),
                np.array([14, 15]),
                np.array([15, 16]),
            ],
        }
        self.mixed_element = _FakeMixedElement(mesh=type("Mesh", (), {"n_elements": 4})())

    def get_field_slice(self, field: str) -> list[int]:
        return self._field_slices[field].tolist()


class _FakeMesh:
    def __init__(self) -> None:
        self.nodes_x_y_pos = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.5],
                [1.0, 0.5],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 1.5],
                [1.0, 1.5],
            ],
            dtype=float,
        )
        self.corner_connectivity = np.array(
            [
                [0, 1, 3, 2],
                [2, 3, 5, 4],
                [4, 5, 7, 6],
            ],
            dtype=np.int64,
        )
        self.elements_connectivity = self.corner_connectivity
        self.n_elements = int(self.corner_connectivity.shape[0])


def _trial_basis() -> np.ndarray:
    rng = np.random.default_rng(1234)
    q, _ = np.linalg.qr(rng.normal(size=(18, 5)))
    return np.ascontiguousarray(q[:, :5], dtype=np.float64)


def test_block_balanced_sampling_covers_mandatory_elements_and_blocks() -> None:
    dh = _FakeDofHandler()
    snapshots = np.column_stack(
        [
            np.linspace(0.0, 1.0, dh.total_dofs),
            np.linspace(1.0, -0.25, dh.total_dofs),
            np.sin(np.linspace(0.0, 2.0, dh.total_dofs)),
        ]
    )

    sampling = build_block_balanced_gnat_sampling(
        dh,
        _trial_basis(),
        snapshot_matrix=snapshots,
        row_blocks=field_row_blocks(dh),
        sample_rows=5,
        candidate_element_ids=np.array([1, 2, 3], dtype=np.int64),
        mandatory_element_ids=np.array([2, 3], dtype=np.int64),
        min_rows_per_block={"a": 2, "b": 2, "c": 2},
        row_weight_max=10.0,
        rcond=1.0e-12,
    )

    support = set(int(eid) for eid in support_element_ids_from_rows(dh, sampling.row_dofs))
    assert {2, 3}.issubset(support)
    assert sampling.metadata["interface_complete"] is True
    assert sampling.metadata["candidate_limited"] is True
    assert sampling.metadata["candidate_element_count"] == 3
    assert sampling.metadata["missing_mandatory_element_count"] == 0
    assert sampling.metadata["budget_exceeded_by_required_rows"] is True
    assert sampling.metadata["block_selected_rows"]["a"] >= 2
    assert sampling.metadata["block_selected_rows"]["b"] >= 2
    assert sampling.metadata["block_selected_rows"]["c"] >= 2
    np.testing.assert_allclose(sampling.selected_basis, np.eye(sampling.row_dofs.size))
    np.testing.assert_allclose(sampling.residual_terms, np.eye(sampling.row_dofs.size))


def test_block_balanced_sampling_rejects_uncoverable_mandatory_element() -> None:
    dh = _FakeDofHandler()
    free_rows = np.setdiff1d(
        np.arange(dh.total_dofs, dtype=np.int64),
        rows_supported_on_elements(dh, np.array([3], dtype=np.int64)),
    )
    with pytest.raises(ValueError, match="mandatory element coverage failed"):
        build_block_balanced_gnat_sampling(
            dh,
            _trial_basis(),
            free_rows=free_rows,
            row_blocks=field_row_blocks(dh),
            sample_rows=8,
            mandatory_element_ids=np.array([3], dtype=np.int64),
            min_rows_per_block=1,
        )


def test_coordinate_band_element_selection_uses_element_intersection() -> None:
    mesh = _FakeMesh()
    selected = select_coordinate_band_elements(mesh, axis=1, center=1.0, half_width=0.05)
    np.testing.assert_array_equal(selected, np.array([1, 2], dtype=np.int64))


def test_sampling_metadata_roundtrips_through_native_artifact(tmp_path) -> None:
    dh = _FakeDofHandler()
    sampling = build_block_balanced_gnat_sampling(
        dh,
        _trial_basis(),
        row_blocks=field_row_blocks(dh),
        sample_rows=8,
        mandatory_element_ids=np.array([1, 2], dtype=np.int64),
        min_rows_per_block=1,
    )
    artifact = NativeReducedArtifact(
        problem_id="block_balanced_sampling",
        trial_basis=_trial_basis(),
        offset=np.zeros(dh.total_dofs, dtype=np.float64),
        residual_kernel=NativeKernelReference(
            kernel_id="residual",
            abi="native-kernel-v1",
            param_order=("gdofs_map",),
        ),
        tangent_kernel=NativeKernelReference(
            kernel_id="tangent",
            abi="native-kernel-v1",
            param_order=("gdofs_map",),
        ),
        target=sampling.to_native_target(),
        metadata={"sampling": sampling.metadata},
    )
    path = tmp_path / "sampling_artifact.npz"
    artifact.save(path)
    loaded = load_native_reduced_artifact(path)

    assert loaded.target is not None
    np.testing.assert_array_equal(loaded.target.row_dofs, sampling.row_dofs)
    np.testing.assert_array_equal(loaded.target.element_ids, sampling.element_ids)
    assert loaded.target.metadata["sampler"] == "block_balanced_interface_complete_gnat"
    assert loaded.target.metadata["missing_mandatory_element_count"] == 0
    assert loaded.metadata["sampling"]["interface_complete"] is True


def test_residual_norm_equivalence_certificate_fails_missing_block() -> None:
    residuals = np.array(
        [
            [1.0, 0.5],
            [0.2, -0.4],
            [10.0, -5.0],
            [8.0, 3.0],
        ],
        dtype=float,
    )
    cert = certify_sampled_residual_norm_equivalence(
        residuals,
        row_dofs=np.array([0, 1], dtype=np.int64),
        row_blocks=[
            {"name": "a", "rows": np.array([0, 1], dtype=np.int64)},
            {"name": "b", "rows": np.array([2, 3], dtype=np.int64)},
        ],
        lower_bound=1.0e-6,
        upper_bound=10.0,
    )

    assert cert.passed is False
    assert cert.block_constants["a"]["passed"] is True
    assert cert.block_constants["b"]["passed"] is False
    assert cert.block_constants["b"]["sampled_rows"] == 0


def test_residual_norm_equivalence_certificate_accepts_weighted_full_coverage() -> None:
    residuals = np.array(
        [
            [1.0, 2.0, -1.0],
            [2.0, -1.0, 0.5],
            [0.5, 1.0, 3.0],
            [1.5, -0.5, 1.0],
        ],
        dtype=float,
    )
    cert = certify_sampled_residual_norm_equivalence(
        residuals,
        row_dofs=np.array([0, 1, 2, 3], dtype=np.int64),
        row_weights=np.ones(4, dtype=float),
        row_blocks=[
            {"name": "all", "rows": np.arange(4, dtype=np.int64)},
        ],
        lower_bound=0.99,
        upper_bound=1.01,
    )

    assert cert.passed is True
    assert cert.lower_constant == pytest.approx(1.0)
    assert cert.upper_constant == pytest.approx(1.0)
    assert cert.to_dict()["metadata"]["sampled_rows"] == 4


def test_norm_equivalence_augmentation_adds_missing_high_energy_block_rows() -> None:
    residuals = np.array(
        [
            [1.0, 0.5, -0.25],
            [0.5, -0.5, 0.25],
            [20.0, 10.0, -5.0],
            [-10.0, 4.0, 2.0],
        ],
        dtype=float,
    )
    result = augment_rows_for_residual_norm_equivalence(
        residuals,
        row_dofs=np.array([0, 1], dtype=np.int64),
        row_blocks=[
            {"name": "bulk", "rows": np.array([0, 1], dtype=np.int64)},
            {"name": "interface", "rows": np.array([2, 3], dtype=np.int64)},
        ],
        lower_bound=1.0e-8,
        upper_bound=2.0,
        max_rows=4,
    )

    assert result.certificate.passed
    assert {2, 3}.intersection(set(int(v) for v in result.added_rows))
    assert result.certificate.block_constants["interface"]["sampled_rows"] > 0
