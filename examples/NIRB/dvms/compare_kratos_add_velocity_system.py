from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.NIRB.example2_local_setup import load_example2_local_setup
from examples.NIRB.run_example2_local import _build_fluid_problem, _load_reference_partitioned_meshes
from examples.NIRB.dvms import assemble_dvms_calculate_local_velocity_contribution_p1_tri

_GROUPED_TO_INTERLEAVED_TRI3 = np.asarray([0, 3, 6, 1, 4, 7, 2, 5, 8], dtype=int)


def _find_pycutfem_element_by_old_node_ids(mesh, node_ids: np.ndarray) -> int:
    target = frozenset(int(i) for i in np.asarray(node_ids, dtype=int).reshape(-1))
    new_to_old = getattr(mesh, "_mdpa_new_to_old_node", None)
    if new_to_old is None:
        raise RuntimeError("Reference pycutfem mesh does not carry MDPA node-id mapping.")
    if isinstance(new_to_old, dict):
        old_id = lambda nid: int(new_to_old[int(nid)])
    else:
        arr = np.asarray(new_to_old, dtype=int).reshape(-1)
        old_id = lambda nid: int(arr[int(nid)])
    for eid, conn in enumerate(np.asarray(mesh.elements_connectivity, dtype=int)):
        old_ids = frozenset(old_id(nid) for nid in conn)
        if old_ids == target:
            return int(eid)
    raise RuntimeError(f"Could not match Kratos node ids {sorted(target)} to a pycutfem fluid element.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the local pycutfem AddVelocitySystem block to Kratos.")
    parser.add_argument(
        "--kratos-dump",
        type=Path,
        required=True,
        help="Path to the .npz dump produced by dump_kratos_local_operator.py --mode velocity.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/NIRB/artifacts/compare_kratos_add_velocity_system.json"),
    )
    parser.add_argument("--quadrature-order", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.008)
    parser.add_argument("--bossak-alpha", type=float, default=-0.3)
    parser.add_argument("--rho", type=float, default=1000.0)
    parser.add_argument("--mu", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dump = np.load(Path(args.kratos_dump).resolve(), allow_pickle=False)
    node_ids = np.asarray(dump["node_ids"], dtype=int)

    setup = load_example2_local_setup()
    mesh_f, _ = _load_reference_partitioned_meshes(setup=setup)
    fluid = _build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=int(args.quadrature_order))
    eid = _find_pycutfem_element_by_old_node_ids(mesh_f, node_ids)
    mu_value = (
        float(args.mu)
        if args.mu is not None
        else float(setup.material.density * setup.material.kinematic_viscosity)
    )

    ux_g = np.asarray(fluid["dh"].element_maps["ux"][eid], dtype=int)
    uy_g = np.asarray(fluid["dh"].element_maps["uy"][eid], dtype=int)
    p_g = np.asarray(fluid["dh"].element_maps["p"][eid], dtype=int)
    mesh_dh = fluid.get("mesh_dh", fluid["dh"])
    mx_g = np.asarray(mesh_dh.element_maps["mx"][eid], dtype=int)
    my_g = np.asarray(mesh_dh.element_maps["my"][eid], dtype=int)

    velocity = np.asarray(dump["velocity"], dtype=float).reshape(3, 2)
    pressure = np.asarray(dump["pressure"], dtype=float).reshape(3)
    coords_ref = np.asarray(dump["coords_ref"], dtype=float).reshape(3, 2)
    coords_cur = np.asarray(dump["coords_cur"], dtype=float).reshape(3, 2)
    mesh_velocity = np.asarray(dump["mesh_velocity"], dtype=float).reshape(3, 2)
    subscale_velocity = np.asarray(dump["subscale_velocity"], dtype=float)

    fluid["u_k"].components[0].set_nodal_values(ux_g, velocity[:, 0])
    fluid["u_k"].components[1].set_nodal_values(uy_g, velocity[:, 1])
    fluid["p_k"].set_nodal_values(p_g, pressure)

    disp = coords_cur - coords_ref
    disp_prev = disp - float(args.dt) * mesh_velocity
    fluid["d_mesh"].components[0].set_nodal_values(mx_g, disp[:, 0])
    fluid["d_mesh"].components[1].set_nodal_values(my_g, disp[:, 1])
    fluid["d_prev"].components[0].set_nodal_values(mx_g, disp_prev[:, 0])
    fluid["d_prev"].components[1].set_nodal_values(my_g, disp_prev[:, 1])

    state = fluid["dvms_state"]
    state.predicted_subscale_velocity.fill(0.0)
    state.old_subscale_velocity.fill(0.0)
    state.momentum_projection.fill(0.0)
    state.mass_projection.fill(0.0)
    state.old_mass_residual.fill(0.0)
    start = int(eid) * int(state.n_qp_per_element)
    stop = start + int(state.n_qp_per_element)
    if subscale_velocity.shape[0] == int(state.n_qp_per_element):
        state.predicted_subscale_velocity[start:stop, :] = np.asarray(subscale_velocity[:, :2], dtype=float)
    state.sync_coefficients_from_samples()

    lhs_local, rhs_local, gdofs = assemble_dvms_calculate_local_velocity_contribution_p1_tri(
        mesh=mesh_f,
        dh=fluid["dh"],
        eid=eid,
        u_k=fluid["u_k"],
        p_k=fluid["p_k"],
        d_mesh=fluid["d_mesh"],
        d_prev=fluid["d_prev"],
        state=state,
        rho_f=float(args.rho),
        mu_f=float(mu_value),
        dt=float(args.dt),
        bossak_alpha=float(args.bossak_alpha),
        quadrature_order=int(args.quadrature_order),
    )

    perm = _GROUPED_TO_INTERLEAVED_TRI3
    lhs_kratos = np.asarray(dump["lhs"], dtype=float)[np.ix_(perm, perm)]
    rhs_kratos = np.asarray(dump["rhs"], dtype=float)[perm]

    lhs_diff = lhs_local - lhs_kratos
    rhs_diff = rhs_local - rhs_kratos
    report = {
        "element_id_pycutfem": int(eid),
        "element_id_kratos": int(np.asarray(dump["element_id"]).reshape(())),
        "gdofs": np.asarray(gdofs, dtype=int).tolist(),
        "mu_f": float(mu_value),
        "lhs_inf_norm_local": float(np.max(np.abs(lhs_local))) if lhs_local.size else 0.0,
        "lhs_inf_norm_kratos": float(np.max(np.abs(lhs_kratos))) if lhs_kratos.size else 0.0,
        "lhs_diff_inf_norm": float(np.max(np.abs(lhs_diff))) if lhs_diff.size else 0.0,
        "rhs_inf_norm_local": float(np.max(np.abs(rhs_local))) if rhs_local.size else 0.0,
        "rhs_inf_norm_kratos": float(np.max(np.abs(rhs_kratos))) if rhs_kratos.size else 0.0,
        "rhs_diff_inf_norm": float(np.max(np.abs(rhs_diff))) if rhs_diff.size else 0.0,
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
