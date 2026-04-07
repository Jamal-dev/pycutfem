from __future__ import annotations

import math
from pathlib import Path

import pytest

from pycutfem.utils.gmsh_loader import mesh_from_gmsh

from examples.NIRB.example2_problem import DoubleFlapGeometry, build_bcs, build_conforming_mesh, classify_fluid_solid, retag_boundaries, tag_interface_edges


def _geometry() -> DoubleFlapGeometry:
    return DoubleFlapGeometry(
        channel_length=2.5,
        channel_height=0.492,
        cylinder_center=(0.2, 0.2),
        cylinder_radius=0.05,
        solid_x0=1.2,
        solid_x1=1.52,
        solid_y0=0.0,
        solid_y1=0.28,
        base_height=0.06,
        arm_width=0.06,
        inlet_ramp_end_time=1.0,
    )


def test_double_flap_geometry_contains_expected_regions() -> None:
    geometry = _geometry()

    assert geometry.contains_solid_point(1.22, 0.20)
    assert geometry.contains_solid_point(1.48, 0.20)
    assert geometry.contains_solid_point(1.36, 0.03)
    assert not geometry.contains_solid_point(1.36, 0.20)
    assert math.isclose(geometry.gap_width, 0.20, rel_tol=1.0e-12)


def test_double_flap_boundary_conditions_use_ramped_parabola() -> None:
    geometry = _geometry()
    bcs, bcs_homog = build_bcs(geometry=geometry, reference_velocity=2.5)

    inlet_x = next(bc for bc in bcs if bc.field == "ux" and bc.domain_tag == geometry.inlet_tag)
    inlet_y = next(bc for bc in bcs if bc.field == "uy" and bc.domain_tag == geometry.inlet_tag)
    center_y = 0.5 * geometry.channel_height

    assert inlet_y.value(0.0, center_y, 0.25) == 0.0
    assert inlet_x.value(0.0, center_y, 0.0) == 0.0
    assert inlet_x.value(0.0, center_y, geometry.inlet_ramp_end_time) > 2.4
    assert len(bcs) == len(bcs_homog)


def test_double_flap_conforming_mesh_smoke(tmp_path: Path) -> None:
    pytest.importorskip("gmsh")

    geometry = _geometry()
    mesh_path = tmp_path / "double_flap_smoke.msh"
    build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=0.12, order=1)

    mesh = mesh_from_gmsh(mesh_path, apply_boundary_tags=True)
    fluid_bs, solid_bs = classify_fluid_solid(mesh, geometry)
    retag_boundaries(mesh, geometry, overwrite=True, tol=5.0e-3)
    tag_interface_edges(mesh, geometry)

    assert fluid_bs.cardinality() > 0
    assert solid_bs.cardinality() > 0
    assert mesh.edge_bitset(geometry.inlet_tag).cardinality() > 0
    assert mesh.edge_bitset(geometry.outlet_tag).cardinality() > 0
    assert mesh.edge_bitset(geometry.walls_tag).cardinality() > 0
    assert mesh.edge_bitset(geometry.cylinder_tag).cardinality() > 0
    assert mesh.edge_bitset(geometry.clamp_tag).cardinality() > 0
    assert mesh.edge_bitset(geometry.interface_tag).cardinality() > 0
