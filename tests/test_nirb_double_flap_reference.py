from __future__ import annotations

import math
from pathlib import Path

import pytest

from examples.NIRB.double_flap_reference import _parse_mdpa, load_double_flap_reference, reynolds_to_mean_velocity
from examples.NIRB.example2_local_setup import load_example2_local_setup


def test_parse_mdpa_synthetic(tmp_path: Path) -> None:
    mdpa = tmp_path / "synthetic.mdpa"
    mdpa.write_text(
        "\n".join(
            [
                "Begin ModelPartData",
                "End ModelPartData",
                "Begin Properties 0",
                "End Properties",
                "Begin Nodes",
                "1 0.0 0.0 0.0",
                "2 1.0 0.0 0.0",
                "3 1.0 1.0 0.0",
                "4 0.0 1.0 0.0",
                "End Nodes",
                "Begin Elements Element2D3N",
                "1 0 1 2 3",
                "End Elements",
                "Begin Conditions LineCondition2D2N",
                "10 0 1 2",
                "11 0 2 3",
                "End Conditions",
                "Begin SubModelPart Outer",
                "Begin SubModelPartNodes",
                "1",
                "2",
                "3",
                "End SubModelPartNodes",
                "Begin SubModelPartElements",
                "1",
                "End SubModelPartElements",
                "Begin SubModelPartConditions",
                "10",
                "11",
                "End SubModelPartConditions",
                "End SubModelPart",
            ]
        ),
        encoding="utf-8",
    )

    mesh = _parse_mdpa(mdpa)
    assert mesh.element_block == "Element2D3N"
    assert mesh.condition_blocks == ("LineCondition2D2N",)
    assert mesh.elements[1] == (1, 2, 3)
    assert mesh.conditions[10] == (1, 2)
    assert mesh.submodelparts["Outer"].node_ids == (1, 2, 3)
    assert mesh.submodelparts["Outer"].condition_ids == (10, 11)


def test_reynolds_to_mean_velocity() -> None:
    value = reynolds_to_mean_velocity(250.0, kinematic_viscosity=1.0e-3, cylinder_diameter=0.1)
    assert math.isclose(value, 2.5)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / ".tmp" / "nirb_benchmarks" / "DoubleFlap").exists(),
    reason="DoubleFlap reference files are not available in this checkout.",
)
def test_double_flap_reference_summary_matches_download() -> None:
    reference = load_double_flap_reference()
    setup = load_example2_local_setup()

    assert reference.interface_node_count == 265
    assert reference.clamp_node_count == 65
    assert reference.interface_max_mismatch < 5.0e-4
    assert math.isclose(reference.channel_length, 2.5)
    assert math.isclose(reference.channel_height, 0.492)
    assert math.isclose(reference.fluid_time_step, 0.008)
    assert math.isclose(reference.solid_time_step, 0.008)
    assert math.isclose(reference.cylinder_radius, 0.05, rel_tol=2.0e-2)
    assert math.isclose(reference.density, 1000.0, rel_tol=1.0e-12)
    assert math.isclose(reference.kinematic_viscosity, 7.0e-4, rel_tol=1.0e-12)
    assert math.isclose(
        setup.u_mean_for_reynolds(250.0),
        reynolds_to_mean_velocity(250.0, kinematic_viscosity=reference.kinematic_viscosity, cylinder_diameter=0.1),
        rel_tol=1.0e-12,
    )
    assert math.isclose(setup.geometry.base_height, 0.06, rel_tol=5.0e-2)
    assert math.isclose(setup.geometry.arm_width, 0.06, rel_tol=5.0e-2)
