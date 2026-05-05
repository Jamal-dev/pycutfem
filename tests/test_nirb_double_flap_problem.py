from __future__ import annotations

import math
from pathlib import Path

import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction
from pycutfem.ufl.jit_parametrization import build_jit_parametrization
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.gmsh_loader import mesh_from_gmsh

from examples.NIRB.example2_problem import (
    DoubleFlapGeometry,
    _named_constant,
    build_bcs,
    build_conforming_mesh,
    build_jac,
    build_residual,
    classify_fluid_solid,
    retag_boundaries,
    tag_interface_edges,
)


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


def test_double_flap_monolithic_forms_use_named_jit_constants(tmp_path: Path) -> None:
    pytest.importorskip("gmsh")

    geometry = _geometry()
    mesh_path = tmp_path / "double_flap_named_constants.msh"
    build_conforming_mesh(mesh_path, geometry=geometry, mesh_size=0.20, order=1)

    mesh = mesh_from_gmsh(mesh_path, apply_boundary_tags=True)
    fluid_bs, solid_bs = classify_fluid_solid(mesh, geometry)
    retag_boundaries(mesh, geometry, overwrite=True, tol=5.0e-3)
    tag_interface_edges(mesh, geometry)
    outlet_bs = mesh.edge_bitset(geometry.outlet_tag)

    element = MixedElement(
        mesh,
        field_specs={"ux": 1, "uy": 1, "dx": 1, "dy": 1, "p": 1},
    )
    dh = DofHandler(element, method="cg", field_methods={"p": "dg"})

    vel_space = FunctionSpace(name="vel", field_names=["ux", "uy"], dim=1)
    disp_space = FunctionSpace(name="disp", field_names=["dx", "dy"], dim=1)

    du = VectorTrialFunction(space=vel_space, dof_handler=dh)
    dd = VectorTrialFunction(space=disp_space, dof_handler=dh)
    dp = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)
    v = VectorTestFunction(space=vel_space, dof_handler=dh)
    w = VectorTestFunction(space=disp_space, dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)

    uk = VectorFunction(name="u", field_names=["ux", "uy"], dof_handler=dh)
    u_prev = VectorFunction(name="u_prev", field_names=["ux", "uy"], dof_handler=dh)
    dk = VectorFunction(name="d", field_names=["dx", "dy"], dof_handler=dh)
    d_prev = VectorFunction(name="d_prev", field_names=["dx", "dy"], dof_handler=dh)
    pk = Function(name="p", field_name="p", dof_handler=dh)
    p_prev = Function(name="p_prev", field_name="p", dof_handler=dh)
    for func in (uk, u_prev, dk, d_prev, pk, p_prev):
        func.nodal_values.fill(0.0)

    rho_f = _named_constant("example2_rho_f", 1000.0)
    nu_f = _named_constant("example2_nu_f", 1.0e-3)
    mu_f = rho_f * nu_f
    rho_s = _named_constant("example2_rho_s", 1000.0)
    mu_s = _named_constant("example2_mu_s", 5.0e5)
    lambda_s = _named_constant("example2_lambda_s", 5.0e5)
    alpha_u = _named_constant("example2_alpha_u", 1.0e-8)
    stab_eps = _named_constant("example2_stab_eps", 1.0e-8)
    dt_const = _named_constant("example2_dt", 0.008)
    theta_const = _named_constant("example2_theta", 0.5)
    p_gauge = _named_constant("example2_p_gauge", 1.0e-6)

    residual = build_residual(
        uk=uk,
        u_prev=u_prev,
        dk=dk,
        d_prev=d_prev,
        pk=pk,
        p_prev=p_prev,
        v_test=v,
        w_test=w,
        q_test=q,
        dt=dt_const,
        theta=theta_const,
        rho_f=rho_f,
        mu_f=mu_f,
        rho_s=rho_s,
        lambda_s=lambda_s,
        mu_s=mu_s,
        alpha_u=alpha_u,
        stab_eps=stab_eps,
        p_gauge=p_gauge,
        fluid_bs=fluid_bs,
        solid_bs=solid_bs,
        outlet_bs=outlet_bs,
        quad_order=4,
    )
    jacobian = build_jac(
        uk=uk,
        u_prev=u_prev,
        dk=dk,
        d_prev=d_prev,
        pk=pk,
        p_prev=p_prev,
        du=du,
        dd=dd,
        dp=dp,
        test_v=v,
        test_w=w,
        test_q=q,
        timestep=dt_const,
        theta=theta_const,
        rho_f=rho_f,
        mu_f=mu_f,
        rho_s=rho_s,
        lambda_s=lambda_s,
        mu_s=mu_s,
        alpha_u=alpha_u,
        stab_eps=stab_eps,
        p_gauge=p_gauge,
        fluid_bs=fluid_bs,
        solid_bs=solid_bs,
        outlet_bs=outlet_bs,
        quad_order=4,
    )

    names = set(build_jit_parametrization(residual).const_by_name)
    names.update(build_jit_parametrization(jacobian).const_by_name)

    assert {
        "example2_dt",
        "example2_theta",
        "example2_rho_f",
        "example2_nu_f",
        "example2_rho_s",
        "example2_mu_s",
        "example2_lambda_s",
        "example2_alpha_u",
        "example2_p_gauge",
        "example2_half",
        "example2_one",
        "example2_two",
        "example2_two_thirds",
    }.issubset(names)
