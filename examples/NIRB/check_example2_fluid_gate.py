from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

from examples.NIRB.common import dump_json
from examples.NIRB.dvms import (
    _update_fluid_dvms_predicted_subscale,
    _update_fluid_dvms_state_from_previous_step,
)
from examples.NIRB.example2_local_setup import load_example2_local_setup
from examples.NIRB.run_example2_local import (
    CoordinateLookup,
    _assemble_fluid_local_velocity_contribution_raw,
    _boundary_vector_from_global_values,
    _build_fluid_problem,
    _fluid_interface_constrained_reaction_vector,
    _fluid_boundary_conditions,
    _load_reference_partitioned_meshes,
)


def _coord_key(x: float, y: float, ndigits: int = 12) -> tuple[float, float]:
    return (round(float(x), ndigits), round(float(y), ndigits))


class PointLookup:
    def __init__(self, coords: np.ndarray, values: np.ndarray):
        self.coords = np.asarray(coords, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self._exact = {
            _coord_key(x, y): np.asarray(self.values[i], dtype=float).copy()
            for i, (x, y) in enumerate(self.coords)
        }

    def sample(self, target_coords: np.ndarray) -> np.ndarray:
        target = np.asarray(target_coords, dtype=float)
        out = np.empty((target.shape[0], self.values.shape[1]), dtype=float)
        for i, (x, y) in enumerate(target):
            hit = self._exact.get(_coord_key(x, y))
            if hit is not None:
                out[i, :] = hit
            else:
                dist2 = np.sum((self.coords - target[i][None, :]) ** 2, axis=1)
                out[i, :] = self.values[int(np.argmin(dist2)), :]
        return out


def _compare_point_fields(
    *,
    ref_coords: np.ndarray,
    ref_values: np.ndarray,
    local_coords: np.ndarray,
    local_values: np.ndarray,
) -> dict[str, float]:
    ref = np.asarray(ref_values, dtype=float)
    loc = PointLookup(local_coords, local_values).sample(ref_coords)
    diff = loc - ref
    ref_flat = ref.reshape(-1)
    loc_flat = loc.reshape(-1)
    denom = max(float(np.linalg.norm(ref_flat)), 1.0e-15)
    loc_norm = float(np.linalg.norm(loc_flat))
    ref_norm = float(np.linalg.norm(ref_flat))
    cosine = float(np.dot(loc_flat, ref_flat) / max(loc_norm * ref_norm, 1.0e-15))
    return {
        "rel_l2": float(np.linalg.norm(diff.reshape(-1)) / denom),
        "abs_rms": float(np.linalg.norm(diff.reshape(-1)) / np.sqrt(max(diff.size, 1))),
        "cosine": cosine,
        "reference_max_norm": float(np.max(np.linalg.norm(ref, axis=1))),
        "local_max_norm": float(np.max(np.linalg.norm(loc, axis=1))),
    }


def _load_fluid_step1_reference(vtk_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = pv.read(vtk_path)
    coords = np.asarray(mesh.points[:, :2], dtype=float)
    velocity = np.asarray(mesh.point_data["VELOCITY"], dtype=float)[:, :2]
    pressure = np.asarray(mesh.point_data["PRESSURE"], dtype=float).reshape(-1, 1)
    return coords, velocity, pressure


def _assign_vector_field_from_lookup(field, dh, field_names: tuple[str, str], lookup: PointLookup) -> None:
    dh._ensure_dof_coords()
    for comp_idx, field_name in enumerate(field_names):
        ids = np.asarray(dh.get_field_slice(field_name), dtype=int)
        coords = np.asarray(dh._dof_coords[ids], dtype=float)
        values = lookup.sample(coords)
        field.components[comp_idx].set_nodal_values(ids, values[:, comp_idx])


def _assign_scalar_field_from_lookup(field, dh, field_name: str, lookup: PointLookup) -> None:
    dh._ensure_dof_coords()
    ids = np.asarray(dh.get_field_slice(field_name), dtype=int)
    coords = np.asarray(dh._dof_coords[ids], dtype=float)
    values = lookup.sample(coords)
    field.set_nodal_values(ids, values[:, 0])


def _vector_field_matrix(dh, vector) -> tuple[np.ndarray, np.ndarray]:
    dh._ensure_dof_coords()
    x_ids = np.asarray(dh.get_field_slice(vector.components[0].field_name), dtype=int)
    y_ids = np.asarray(dh.get_field_slice(vector.components[1].field_name), dtype=int)
    coords = np.asarray(dh._dof_coords[x_ids], dtype=float)
    values = np.column_stack(
        [
            np.asarray(vector.components[0].get_nodal_values(x_ids), dtype=float),
            np.asarray(vector.components[1].get_nodal_values(y_ids), dtype=float),
        ]
    )
    return coords, values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the local Example 2 fluid DVMS operator against the Kratos step-1 field.")
    parser.add_argument("--output", type=Path, default=Path("examples/NIRB/artifacts/fluid_gate_check.json"))
    parser.add_argument(
        "--kratos-fluid-vtk",
        type=Path,
        default=Path(".tmp/kratos_runs/DoubleFlap_fom_step1_scripted/vtk_output_fsi_cfd/FluidParts_FluidPart_0_1.vtk"),
    )
    parser.add_argument(
        "--kratos-nodal-state",
        type=Path,
        default=None,
        help="Optional NPZ with full-precision Kratos nodal fields from dump_kratos_fluid_step1_state.py. When provided, it overrides --kratos-fluid-vtk for velocity/pressure injection.",
    )
    parser.add_argument(
        "--kratos-monitor",
        type=Path,
        default=Path("examples/NIRB/artifacts/kratos_example2_monitor_20260407/coupling_monitor/step0001_iter0001_after_sync_output_fluid.npz"),
    )
    parser.add_argument("--backend", choices=("python", "jit", "cpp"), default="cpp")
    parser.add_argument("--quad-order", type=int, default=6)
    parser.add_argument("--bossak-alpha", type=float, default=-0.3)
    parser.add_argument("--dynamic-tau", type=float, default=1.0)
    parser.add_argument("--pressure-gauge", type=float, default=1.0e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup = load_example2_local_setup()
    mesh_f, _ = _load_reference_partitioned_meshes(setup=setup)
    fluid = _build_fluid_problem(
        mesh_f,
        poly_order=1,
        pressure_order=1,
        quadrature_order=int(args.quad_order),
    )
    mu_f = float(setup.material.density * setup.material.kinematic_viscosity)
    dt = float(setup.boundaries.time_step)

    if args.kratos_nodal_state is not None:
        with np.load(Path(args.kratos_nodal_state).resolve()) as nodal:
            ref_coords = np.asarray(nodal["node_coords_ref"], dtype=float)
            ref_velocity = np.asarray(nodal["velocity"], dtype=float)
            ref_pressure = np.asarray(nodal["pressure"], dtype=float).reshape(-1, 1)
    else:
        ref_coords, ref_velocity, ref_pressure = _load_fluid_step1_reference(args.kratos_fluid_vtk)
    vel_lookup = PointLookup(ref_coords, ref_velocity)
    p_shifted = ref_pressure - np.mean(ref_pressure, axis=0, keepdims=True)
    p_lookup = PointLookup(ref_coords, ref_pressure)

    _assign_vector_field_from_lookup(
        fluid["u_k"],
        fluid["dh"],
        (fluid["u_k"].components[0].field_name, fluid["u_k"].components[1].field_name),
        vel_lookup,
    )
    _assign_scalar_field_from_lookup(fluid["p_k"], fluid["dh"], fluid["p_k"].field_name, p_lookup)

    zero_iface = CoordinateLookup(np.asarray([[0.0, 0.0]], dtype=float), np.zeros((1, 2), dtype=float), dim=2)

    def inlet_profile(x: float, y: float) -> float:
        del x
        return setup.geometry.inlet_velocity(y, dt, reference_velocity=float(setup.material.max_velocity))

    _update_fluid_dvms_state_from_previous_step(
        state=fluid["dvms_state"],
        dh=fluid["dh"],
        mesh=mesh_f,
        u_prev=fluid["u_prev"],
        d_prev=fluid["d_prev"],
        backend=str(args.backend),
    )
    _update_fluid_dvms_predicted_subscale(
        state=fluid["dvms_state"],
        dh=fluid["dh"],
        mesh=mesh_f,
        u_k=fluid["u_k"],
        u_prev=fluid["u_prev"],
        a_prev=fluid["a_prev"],
        p_k=fluid["p_k"],
        d_mesh=fluid["d_mesh"],
        d_prev=fluid["d_prev"],
        rho_f=float(setup.material.density),
        mu_f=mu_f,
        dt=dt,
        bossak_alpha=float(args.bossak_alpha),
        dynamic_tau=float(args.dynamic_tau),
        backend=str(args.backend),
    )

    fluid_bcs, _ = _fluid_boundary_conditions(
        iface_velocity=zero_iface,
        inlet_lookup=inlet_profile,
        interface_tag=setup.geometry.interface_tag,
        outlet_tag=setup.geometry.outlet_tag,
        walls_tag=setup.geometry.walls_tag,
        cylinder_tag=setup.geometry.cylinder_tag,
    )
    fluid["_current_bcs"] = fluid_bcs
    _, raw_residual = _assemble_fluid_local_velocity_contribution_raw(
        prob=fluid,
        rho_f=float(setup.material.density),
        mu_f=mu_f,
        dt=dt,
        quad_order=int(args.quad_order),
        bossak_alpha=float(args.bossak_alpha),
        need_matrix=False,
        contribution_mode="system",
        backend=str(args.backend),
    )
    raw_residual = np.asarray(raw_residual, dtype=float)
    bc_map = fluid["dh"].get_dirichlet_data(fluid_bcs) or {}
    reduced_residual = raw_residual.copy()
    if bc_map:
        reduced_residual[np.fromiter((int(d) for d in bc_map.keys()), dtype=int)] = 0.0
    reaction_residual = _fluid_interface_constrained_reaction_vector(
        prob=fluid,
        system_rhs=raw_residual,
        interface_tag=setup.geometry.interface_tag,
    )
    reaction_coords, reaction_values = _boundary_vector_from_global_values(
        fluid["dh"],
        vector=fluid["u_k"],
        tag=setup.geometry.interface_tag,
        global_values=reaction_residual,
    )

    with np.load(args.kratos_monitor) as data:
        ref_load_coords = np.asarray(data["fluid_load_coords_ref"], dtype=float)
        ref_load_values = np.asarray(data["fluid_load_values"], dtype=float)

    vel_local_coords, vel_local_values = _vector_field_matrix(fluid["dh"], fluid["u_k"])
    p_ids = np.asarray(fluid["dh"].get_field_slice(fluid["p_k"].field_name), dtype=int)
    fluid["dh"]._ensure_dof_coords()
    p_local_coords = np.asarray(fluid["dh"]._dof_coords[p_ids], dtype=float)
    p_local_values = np.asarray(fluid["p_k"].get_nodal_values(p_ids), dtype=float).reshape(-1, 1)
    p_local_values_shifted = p_local_values - np.mean(p_local_values, axis=0, keepdims=True)

    velocity_cmp = _compare_point_fields(
        ref_coords=ref_coords,
        ref_values=ref_velocity,
        local_coords=vel_local_coords,
        local_values=vel_local_values,
    )
    pressure_cmp = _compare_point_fields(
        ref_coords=ref_coords,
        ref_values=ref_pressure,
        local_coords=p_local_coords,
        local_values=p_local_values,
    )
    pressure_shifted_cmp = _compare_point_fields(
        ref_coords=ref_coords,
        ref_values=p_shifted,
        local_coords=p_local_coords,
        local_values=p_local_values_shifted,
    )
    reaction_cmp = _compare_point_fields(
        ref_coords=ref_load_coords,
        ref_values=ref_load_values,
        local_coords=np.asarray(reaction_coords, dtype=float),
        local_values=np.asarray(reaction_values, dtype=float),
    )
    criteria = {
        "velocity_rel_l2_le_1e-10": bool(float(velocity_cmp["rel_l2"]) <= 1.0e-10),
        "pressure_rel_l2_le_1e-10": bool(float(pressure_cmp["rel_l2"]) <= 1.0e-10),
        "reaction_rel_l2_le_1e-10": bool(float(reaction_cmp["rel_l2"]) <= 1.0e-10),
        "reaction_cosine_ge_1-1e-10": bool(float(reaction_cmp["cosine"]) >= 1.0 - 1.0e-10),
        "reduced_residual_inf_le_1e-8": bool(float(np.max(np.abs(np.asarray(reduced_residual, dtype=float)))) <= 1.0e-8),
    }

    report = {
        "backend": str(args.backend),
        "velocity_field": velocity_cmp,
        "pressure_field_raw": pressure_cmp,
        "pressure_field_mean_shifted": pressure_shifted_cmp,
        "interface_reaction_vs_kratos_raw": reaction_cmp,
        "reduced_residual_inf": float(np.max(np.abs(np.asarray(reduced_residual, dtype=float)))),
        "reduced_residual_l2": float(np.linalg.norm(np.asarray(reduced_residual, dtype=float))),
        "dvms_state_summary": fluid["dvms_state"].summary(),
        "criteria": criteria,
        "criteria_all_pass": bool(all(criteria.values())),
    }
    dump_json(report, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
