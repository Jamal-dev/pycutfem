from __future__ import annotations

import argparse
from pathlib import Path

from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.gmsh_loader import mesh_from_gmsh

from examples.NIRB.example2_local_setup import load_example2_local_setup
from examples.NIRB.example2_problem import (
    _named_constant,
    build_bcs,
    build_conforming_mesh,
    build_jac,
    build_residual,
    classify_fluid_solid,
    retag_boundaries,
    tag_interface_edges,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble the local Example 2 DoubleFlap pycutfem problem once.")
    parser.add_argument("--reference-root", type=Path, default=None, help="Downloaded DoubleFlap benchmark directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("examples/NIRB/artifacts/example2_local_smoke"))
    parser.add_argument("--mesh-size", type=float, default=0.08)
    parser.add_argument("--mesh-order", type=int, default=1)
    parser.add_argument("--field-order", type=int, default=1)
    parser.add_argument("--reference-velocity", type=float, default=2.5)
    parser.add_argument("--backend", choices=("python", "jit"), default="python")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup = load_example2_local_setup(reference_root=args.reference_root, mesh_size_default=args.mesh_size, mesh_order_default=args.mesh_order)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = args.output_dir / "double_flap_conforming.msh"
    build_conforming_mesh(
        mesh_path,
        geometry=setup.geometry,
        mesh_size=float(args.mesh_size),
        order=int(args.mesh_order),
    )

    mesh = mesh_from_gmsh(mesh_path, apply_boundary_tags=True)
    fluid_bs, solid_bs = classify_fluid_solid(mesh, setup.geometry)
    retag_boundaries(mesh, setup.geometry, overwrite=True, tol=max(1.0e-6, 2.5e-2 * float(args.mesh_size)))
    tag_interface_edges(mesh, setup.geometry)
    outlet_bs = mesh.edge_bitset(setup.geometry.outlet_tag)

    element = MixedElement(
        mesh,
        field_specs={
            "ux": int(args.field_order),
            "uy": int(args.field_order),
            "dx": int(args.field_order),
            "dy": int(args.field_order),
            "p": max(1, int(args.field_order)),
        },
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

    rho_f = _named_constant("example2_rho_f", setup.material.density)
    nu_f = _named_constant("example2_nu_f", setup.material.kinematic_viscosity)
    mu_f = rho_f * nu_f
    rho_s = _named_constant("example2_rho_s", setup.material.density)
    mu_s = _named_constant("example2_mu_s", setup.material.shear_modulus)
    lambda_s = _named_constant("example2_lambda_s", setup.material.lame_lambda)
    alpha_u = _named_constant("example2_alpha_u", 1.0e-8)
    stab_eps = _named_constant("example2_stab_eps", 1.0e-8)
    dt_const = _named_constant("example2_dt", setup.boundaries.time_step)
    theta_const = _named_constant("example2_theta", 0.5)
    quad_order = 2 * int(args.field_order) + 2
    p_gauge = _named_constant("example2_p_gauge", 1.0e-6)

    res_form = build_residual(
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
        quad_order=quad_order,
    )
    jac_form = build_jac(
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
        quad_order=quad_order,
    )

    bcs, _ = build_bcs(geometry=setup.geometry, reference_velocity=float(args.reference_velocity))
    dh.apply_bcs(bcs, uk, u_prev, dk, d_prev, pk, p_prev)

    _, residual_vec = assemble_form(Equation(None, res_form), dof_handler=dh, bcs=bcs, backend=args.backend, quad_degree=quad_order)
    jacobian_mat, _ = assemble_form(Equation(jac_form, None), dof_handler=dh, bcs=bcs, backend=args.backend, quad_degree=quad_order)

    print(f"mesh: {mesh_path}")
    print(f"elements: fluid={fluid_bs.cardinality()} solid={solid_bs.cardinality()}")
    print(f"edge tags: inlet={mesh.edge_bitset(setup.geometry.inlet_tag).cardinality()} outlet={outlet_bs.cardinality()} walls={mesh.edge_bitset(setup.geometry.walls_tag).cardinality()} cylinder={mesh.edge_bitset(setup.geometry.cylinder_tag).cardinality()} interface={mesh.edge_bitset(setup.geometry.interface_tag).cardinality()} clamp={mesh.edge_bitset(setup.geometry.clamp_tag).cardinality()}")
    print(f"dofs: total={dh.total_dofs}")
    print(f"residual_inf: {abs(residual_vec).max():.6e}")
    print(f"jacobian_shape: {jacobian_mat.shape}")


if __name__ == "__main__":
    main()
