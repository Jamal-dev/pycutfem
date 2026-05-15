from __future__ import annotations

import numpy as np

from examples.NIRB.reduced_fluid import SampledFluidStateDecoder
from examples.NIRB.run_example2_local import CoordinateLookup, _SampledFluidHROMStateWriter, _load_sampled_lspg_hybrid_model


class _FakeDofHandler:
    def __init__(self) -> None:
        self.total_dofs = 9
        self._slices = {
            "ux": np.asarray([0, 1, 2], dtype=int),
            "uy": np.asarray([3, 4, 5], dtype=int),
            "p": np.asarray([6, 7, 8], dtype=int),
        }
        self.element_maps = {
            "ux": np.asarray([[0, 1, 2], [2, 1, 0]], dtype=int),
            "uy": np.asarray([[3, 4, 5], [5, 4, 3]], dtype=int),
            "p": np.asarray([[6, 7, 8], [8, 7, 6]], dtype=int),
        }

    def get_field_slice(self, field_name: str) -> np.ndarray:
        return self._slices[str(field_name)]


class _FakeScalar:
    def __init__(self, field_name: str, gdofs: np.ndarray) -> None:
        self.field_name = str(field_name)
        self.nodal_values = np.zeros(int(np.asarray(gdofs).size), dtype=float)
        self._g2l = {int(g): i for i, g in enumerate(np.asarray(gdofs, dtype=int).reshape(-1))}

    def set_nodal_values(self, global_dofs: np.ndarray, values: np.ndarray) -> None:
        for gdof, value in zip(np.asarray(global_dofs, dtype=int), np.asarray(values, dtype=float), strict=False):
            self.nodal_values[self._g2l[int(gdof)]] = float(value)


class _FakeVector:
    def __init__(self, fields: tuple[str, str], dh: _FakeDofHandler) -> None:
        self.components = [_FakeScalar(field, dh.get_field_slice(field)) for field in fields]


def test_sampled_fluid_state_decoder_returns_unique_and_element_local_values() -> None:
    dh = _FakeDofHandler()
    basis = np.arange(dh.total_dofs * 2, dtype=float).reshape(dh.total_dofs, 2) / 10.0
    offset = np.arange(dh.total_dofs, dtype=float)
    coeffs = np.asarray([0.5, -1.25], dtype=float)
    decoder = SampledFluidStateDecoder.from_sample_elements(
        dh=dh,
        basis=basis,
        offset=offset,
        element_ids=np.asarray([1], dtype=int),
    )

    expected = offset + basis @ coeffs
    np.testing.assert_allclose(decoder.field_values("ux", coeffs), expected[[0, 1, 2]])
    np.testing.assert_allclose(decoder.element_field_values("ux", coeffs), expected[[[2, 1, 0]]])
    np.testing.assert_allclose(decoder.element_field_values("p", coeffs), expected[[[8, 7, 6]]])
    local_values = decoder.element_local_values(
        coeffs,
        fluid_prev_step_u=np.arange(6, dtype=float),
        fluid_a_prev_stage=np.ones(6, dtype=float),
        bossak={"ma0": 2.0, "ma2": 3.0},
    )
    np.testing.assert_allclose(local_values["ux"], expected[[[2, 1, 0]]])
    np.testing.assert_allclose(local_values["uy"], expected[[[5, 4, 3]]])
    np.testing.assert_allclose(local_values["p"], expected[[[8, 7, 6]]])
    np.testing.assert_allclose(local_values["ax_curr"], 2.0 * (expected[[2, 1, 0]][None, :] - np.array([[2, 1, 0]])) + 3.0)
    np.testing.assert_allclose(local_values["ay_curr"], 2.0 * (expected[[5, 4, 3]][None, :] - np.array([[5, 4, 3]])) + 3.0)
    np.testing.assert_allclose(
        decoder.bossak_acceleration_field_values(
            "uy",
            coeffs,
            fluid_prev_step_u=np.arange(6, dtype=float),
            fluid_a_prev_stage=np.ones(6, dtype=float),
            ma0=2.0,
            ma2=3.0,
            velocity_field_offsets={"ux": 0, "uy": 3},
        ),
        2.0 * (expected[[3, 4, 5]] - np.arange(3, 6)) + 3.0,
    )


def test_sampled_hrom_state_writer_updates_only_sampled_fields_and_acceleration() -> None:
    dh = _FakeDofHandler()
    basis = np.arange(dh.total_dofs * 2, dtype=float).reshape(dh.total_dofs, 2) / 10.0
    offset = np.arange(dh.total_dofs, dtype=float)
    coeffs = np.asarray([0.5, -1.25], dtype=float)
    writer = _SampledFluidHROMStateWriter.from_sample_elements(
        dh=dh,
        basis=basis,
        offset=offset,
        element_ids=np.asarray([0], dtype=int),
    )
    u_k = _FakeVector(("ux", "uy"), dh)
    a_k = _FakeVector(("ux", "uy"), dh)
    p_k = _FakeScalar("p", dh.get_field_slice("p"))
    for component in a_k.components:
        component.nodal_values[:] = 99.0

    writer.write(
        u_k=u_k,
        p_k=p_k,
        a_k=a_k,
        coefficients=coeffs,
        fluid_prev_step_u=np.arange(6, dtype=float),
        fluid_a_prev_stage=np.ones(6, dtype=float),
        bossak={"ma0": 2.0, "ma2": 3.0},
        preserve_acceleration_seed=True,
    )

    expected = offset + basis @ coeffs
    np.testing.assert_allclose(u_k.components[0].nodal_values, expected[[0, 1, 2]])
    np.testing.assert_allclose(u_k.components[1].nodal_values, expected[[3, 4, 5]])
    np.testing.assert_allclose(p_k.nodal_values, expected[[6, 7, 8]])
    np.testing.assert_allclose(a_k.components[0].nodal_values, 99.0)
    np.testing.assert_allclose(a_k.components[1].nodal_values, 99.0)

    writer.write(
        u_k=u_k,
        p_k=p_k,
        a_k=a_k,
        coefficients=coeffs,
        fluid_prev_step_u=np.arange(6, dtype=float),
        fluid_a_prev_stage=np.ones(6, dtype=float),
        bossak={"ma0": 2.0, "ma2": 3.0},
        preserve_acceleration_seed=False,
    )

    np.testing.assert_allclose(a_k.components[0].nodal_values, 2.0 * (expected[[0, 1, 2]] - np.arange(3)) + 3.0)
    np.testing.assert_allclose(a_k.components[1].nodal_values, 2.0 * (expected[[3, 4, 5]] - np.arange(3, 6)) + 3.0)


def test_sampled_hrom_model_loads_and_evaluates_reduced_reaction(tmp_path) -> None:
    path = tmp_path / "model_with_reaction.npz"
    basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, -0.25],
            [-1.0, 2.0],
        ],
        dtype=float,
    )
    reaction_matrix = np.array(
        [
            [2.0, 0.0],
            [0.0, 3.0],
            [-1.0, 0.5],
            [0.25, -2.0],
        ],
        dtype=float,
    )
    reaction_bias = np.array([0.5, -1.0, 2.0, 0.25], dtype=float)
    reaction_coords = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=int),
        basis=basis,
        free_dofs=np.array([0, 1, 2, 3], dtype=int),
        sample_row_dofs=np.array([0, 2], dtype=int),
        sample_element_ids=np.array([1], dtype=int),
        sample_weights=np.ones(2, dtype=float),
        sample_element_weights=np.ones(1, dtype=float),
        objective=np.asarray("sampled_lspg"),
        reaction_matrix=reaction_matrix,
        reaction_bias=reaction_bias,
        reaction_coords=reaction_coords,
        reaction_kind=np.asarray("point"),
    )

    model = _load_sampled_lspg_hybrid_model(path, total_dofs=4, n_elements=3)
    coeffs = np.array([0.25, -0.5], dtype=float)
    lookup = model.reduced_reaction_lookup(coeffs)

    assert model.has_reduced_reaction
    np.testing.assert_allclose(lookup.coords, reaction_coords)
    np.testing.assert_allclose(lookup.values.reshape(-1), reaction_bias + reaction_matrix @ coeffs)


def test_incremental_reduced_reaction_adds_base_lookup(tmp_path) -> None:
    path = tmp_path / "model_with_incremental_reaction.npz"
    basis = np.eye(4, 2)
    reaction_matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, -1.0],
            [-0.5, 0.25],
        ],
        dtype=float,
    )
    reaction_bias = np.array([0.1, -0.2, 0.3, -0.4], dtype=float)
    reaction_coords = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=int),
        basis=basis,
        free_dofs=np.array([0, 1, 2, 3], dtype=int),
        sample_row_dofs=np.array([0, 2], dtype=int),
        sample_element_ids=np.array([1], dtype=int),
        sample_weights=np.ones(2, dtype=float),
        sample_element_weights=np.ones(1, dtype=float),
        objective=np.asarray("sampled_lspg"),
        reaction_matrix=reaction_matrix,
        reaction_bias=reaction_bias,
        reaction_coords=reaction_coords,
        reaction_kind=np.asarray("incremental_point"),
    )

    model = _load_sampled_lspg_hybrid_model(path, total_dofs=4, n_elements=3)
    coeffs = np.array([0.5, -2.0], dtype=float)
    base = CoordinateLookup(reaction_coords, np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float), dim=2)
    lookup = model.reduced_reaction_lookup(coeffs, base_lookup=base)

    assert model.reaction_is_incremental
    np.testing.assert_allclose(
        lookup.values.reshape(-1),
        base.values.reshape(-1) + reaction_bias + reaction_matrix @ coeffs,
    )


def test_sampled_nonlinear_reaction_reconstructs_from_gappy_rows(tmp_path) -> None:
    path = tmp_path / "model_with_sampled_reaction.npz"
    basis = np.eye(4, 2)
    reaction_coords = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    reaction_basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    reaction_mean = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=int),
        basis=basis,
        free_dofs=np.array([0, 1, 2, 3], dtype=int),
        sample_row_dofs=np.array([0, 2], dtype=int),
        sample_element_ids=np.array([1], dtype=int),
        sample_weights=np.ones(2, dtype=float),
        sample_element_weights=np.ones(1, dtype=float),
        objective=np.asarray("sampled_lspg"),
        reaction_coords=reaction_coords,
        reaction_nonlinear_kind=np.asarray("gappy_pod_point"),
        reaction_basis=reaction_basis,
        reaction_mean=reaction_mean,
        reaction_sample_row_dofs=np.array([0, 2], dtype=int),
        reaction_sample_element_ids=np.array([1], dtype=int),
        reaction_sample_to_coefficients=np.eye(2, dtype=float),
        reaction_sample_output_positions=np.array([0, 2], dtype=int),
    )

    model = _load_sampled_lspg_hybrid_model(path, total_dofs=4, n_elements=3)
    lookup = model.sampled_reaction_lookup(np.array([12.0, 27.0], dtype=float))

    assert model.has_sampled_reaction
    np.testing.assert_allclose(lookup.coords, reaction_coords)
    np.testing.assert_allclose(lookup.values.reshape(-1), np.array([12.0, 20.0, 27.0, 40.0]))


def test_interface_impedance_reaction_lookup_adds_secant_delta(tmp_path) -> None:
    path = tmp_path / "model_with_impedance.npz"
    basis = np.eye(4, 2)
    coords = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    feature_basis = np.eye(8, 2)
    impedance_matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [-1.0, 0.5],
            [0.25, -0.5],
        ],
        dtype=float,
    )
    impedance_bias = np.array([0.1, -0.2, 0.3, -0.4], dtype=float)
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=int),
        basis=basis,
        free_dofs=np.array([0, 1, 2, 3], dtype=int),
        sample_row_dofs=np.array([0, 2], dtype=int),
        sample_element_ids=np.array([1], dtype=int),
        sample_weights=np.ones(2, dtype=float),
        sample_element_weights=np.ones(1, dtype=float),
        objective=np.asarray("sampled_lspg"),
        impedance_matrix=impedance_matrix,
        impedance_bias=impedance_bias,
        impedance_coords=coords,
        impedance_feature_basis=feature_basis,
        impedance_feature_mean=np.zeros(8, dtype=float),
        impedance_velocity_scale=np.asarray(0.5, dtype=float),
        impedance_kind=np.asarray("secant_point"),
    )

    model = _load_sampled_lspg_hybrid_model(path, total_dofs=4, n_elements=3)
    current_disp = CoordinateLookup(coords, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float), dim=2)
    previous_disp = CoordinateLookup(coords, np.array([[0.5, 1.5], [2.5, 3.5]], dtype=float), dim=2)
    current_vel = CoordinateLookup(coords, np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float), dim=2)
    previous_vel = CoordinateLookup(coords, np.array([[8.0, 18.0], [29.0, 39.0]], dtype=float), dim=2)
    previous_reaction = CoordinateLookup(coords, np.array([[100.0, 200.0], [300.0, 400.0]], dtype=float), dim=2)

    lookup = model.interface_impedance_reaction_lookup(
        interface_disp_lookup=current_disp,
        interface_velocity_lookup=current_vel,
        previous_interface_disp_lookup=previous_disp,
        previous_interface_velocity_lookup=previous_vel,
        previous_reaction_lookup=previous_reaction,
    )

    feature = np.concatenate(
        [
            (current_disp.values - previous_disp.values).reshape(-1),
            0.5 * (current_vel.values - previous_vel.values).reshape(-1),
        ]
    )
    expected_delta = impedance_bias + impedance_matrix @ (feature @ feature_basis)
    assert model.has_interface_impedance
    np.testing.assert_allclose(lookup.values.reshape(-1), previous_reaction.values.reshape(-1) + expected_delta)
