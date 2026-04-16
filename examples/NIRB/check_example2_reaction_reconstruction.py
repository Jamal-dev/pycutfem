from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from examples.NIRB.common import dump_json
from examples.NIRB.example2_local_setup import load_example2_local_setup
from examples.NIRB.run_example2_local import (
    CoordinateLookup,
    _build_fluid_problem,
    _fluid_boundary_conditions,
    _fluid_interface_reaction_loads,
    _load_reference_partitioned_meshes,
    _transfer_vector_field,
    _update_fluid_dvms_state_from_previous_step,
)
from examples.NIRB.dvms import _update_fluid_dvms_predicted_subscale


def _coord_key(x: float, y: float, ndigits: int = 12) -> tuple[float, float]:
    return (round(float(x), ndigits), round(float(y), ndigits))


class PointLookup:
    def __init__(self, coords: np.ndarray, values: np.ndarray) -> None:
        self.coords = np.asarray(coords, dtype=float)
        self.values = np.asarray(values, dtype=float)
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError("coords must have shape (n, 2)")
        if self.values.ndim != 2 or self.values.shape[0] != self.coords.shape[0]:
            raise ValueError("values must have shape (n, m)")
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
                continue
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
    cosine = float(np.dot(loc_flat, ref_flat) / max(float(np.linalg.norm(loc_flat) * np.linalg.norm(ref_flat)), 1.0e-15))
    return {
        "rel_l2": float(np.linalg.norm(diff.reshape(-1)) / denom),
        "abs_rms": float(np.linalg.norm(diff.reshape(-1)) / np.sqrt(max(diff.size, 1))),
        "abs_max": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "cosine": cosine,
        "reference_max_norm": float(np.max(np.linalg.norm(ref, axis=1))) if ref.size else 0.0,
        "local_max_norm": float(np.max(np.linalg.norm(loc, axis=1))) if loc.size else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Example 2 reaction reconstruction on an exact Kratos coupled-stage state.")
    parser.add_argument("--kratos-stage-state", type=Path, required=True, help="Full nodal stage-state NPZ dumped from Kratos.")
    parser.add_argument("--kratos-monitor", type=Path, required=True, help="Monitored coupling-stage NPZ with exact interface velocity/load.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("python", "jit", "cpp"), default="cpp")
    parser.add_argument("--quad-order", type=int, default=1)
    parser.add_argument("--bossak-alpha", type=float, default=-0.3)
    parser.add_argument("--dynamic-tau", type=float, default=1.0)
    parser.add_argument("--reference-velocity", type=float, default=None)
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

    with np.load(Path(args.kratos_stage_state).resolve()) as stage:
        stage_coords = np.asarray(stage["node_coords_ref"], dtype=float)
        stage_velocity = np.asarray(stage["velocity"], dtype=float)
        stage_pressure = np.asarray(stage["pressure"], dtype=float).reshape(-1, 1)
        stage_mesh_displacement = np.asarray(stage["mesh_displacement"], dtype=float)
        stage_reaction = np.asarray(stage["reaction"], dtype=float)

    with np.load(Path(args.kratos_monitor).resolve()) as monitor:
        iface_velocity_coords = np.asarray(monitor["fluid_velocity_coords_ref"], dtype=float)
        iface_velocity_values = np.asarray(monitor["fluid_velocity_values"], dtype=float)
        iface_load_coords = np.asarray(monitor["fluid_load_coords_ref"], dtype=float)
        iface_load_values = np.asarray(monitor["fluid_load_values"], dtype=float)

    _transfer_vector_field(
        target_dh=fluid["dh"],
        target_vec=fluid["u_k"],
        source_lookup=CoordinateLookup(stage_coords, stage_velocity, dim=2),
    )
    _transfer_vector_field(
        target_dh=fluid["dh"],
        target_vec=fluid["d_mesh"],
        source_lookup=CoordinateLookup(stage_coords, stage_mesh_displacement, dim=2),
    )

    fluid["dh"]._ensure_dof_coords()
    p_ids = np.asarray(fluid["dh"].get_field_slice(fluid["p_k"].field_name), dtype=int)
    p_coords = np.asarray(fluid["dh"]._dof_coords[p_ids], dtype=float)
    p_values = PointLookup(stage_coords, stage_pressure).sample(p_coords)[:, 0]
    fluid["p_k"].set_nodal_values(p_ids, np.asarray(p_values, dtype=float))

    iface_velocity = CoordinateLookup(iface_velocity_coords, iface_velocity_values, dim=2)
    reference_velocity = (
        float(args.reference_velocity)
        if args.reference_velocity is not None
        else float(setup.material.max_velocity)
    )

    def inlet_profile(x: float, y: float) -> float:
        del x
        return setup.geometry.inlet_velocity(
            y,
            float(setup.boundaries.time_step),
            reference_velocity=reference_velocity,
        )

    fluid_bcs, fluid_bcs_homog = _fluid_boundary_conditions(
        iface_velocity=iface_velocity,
        inlet_lookup=inlet_profile,
        interface_tag=setup.geometry.interface_tag,
        outlet_tag=setup.geometry.outlet_tag,
        walls_tag=setup.geometry.walls_tag,
        cylinder_tag=setup.geometry.cylinder_tag,
    )
    fluid["_current_bcs"] = fluid_bcs
    fluid["_current_bcs_homog"] = fluid_bcs_homog

    _update_fluid_dvms_state_from_previous_step(
        state=fluid["dvms_state"],
        dh=fluid["dh"],
        mesh=mesh_f,
        u_prev=fluid["u_prev"],
        d_prev=fluid["d_prev"],
        d_geo=fluid["d_mesh"],
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
        mu_f=float(setup.material.density * setup.material.kinematic_viscosity),
        dt=float(setup.boundaries.time_step),
        bossak_alpha=float(args.bossak_alpha),
        dynamic_tau=float(args.dynamic_tau),
        backend=str(args.backend),
    )

    reaction_nolift = _fluid_interface_reaction_loads(
        prob=fluid,
        rho_f=float(setup.material.density),
        mu_f=float(setup.material.density * setup.material.kinematic_viscosity),
        dt=float(setup.boundaries.time_step),
        quad_order=int(args.quad_order),
        bossak_alpha=float(args.bossak_alpha),
        dynamic_tau=float(args.dynamic_tau),
        interface_tag=setup.geometry.interface_tag,
        backend=str(args.backend),
        contribution_mode="system",
        apply_dirichlet_lift=False,
    )
    reaction_lift = _fluid_interface_reaction_loads(
        prob=fluid,
        rho_f=float(setup.material.density),
        mu_f=float(setup.material.density * setup.material.kinematic_viscosity),
        dt=float(setup.boundaries.time_step),
        quad_order=int(args.quad_order),
        bossak_alpha=float(args.bossak_alpha),
        dynamic_tau=float(args.dynamic_tau),
        interface_tag=setup.geometry.interface_tag,
        backend=str(args.backend),
        contribution_mode="system",
        apply_dirichlet_lift=True,
    )

    exact_stage_reaction_cmp = _compare_point_fields(
        ref_coords=iface_load_coords,
        ref_values=iface_load_values,
        local_coords=stage_coords,
        local_values=stage_reaction,
    )
    reaction_nolift_cmp = _compare_point_fields(
        ref_coords=iface_load_coords,
        ref_values=iface_load_values,
        local_coords=np.asarray(reaction_nolift.coords, dtype=float),
        local_values=np.asarray(reaction_nolift.values, dtype=float),
    )
    reaction_lift_cmp = _compare_point_fields(
        ref_coords=iface_load_coords,
        ref_values=iface_load_values,
        local_coords=np.asarray(reaction_lift.coords, dtype=float),
        local_values=np.asarray(reaction_lift.values, dtype=float),
    )

    summary = {
        "kratos_stage_state": str(Path(args.kratos_stage_state).resolve()),
        "kratos_monitor": str(Path(args.kratos_monitor).resolve()),
        "backend": str(args.backend),
        "quad_order": int(args.quad_order),
        "target_quantity_cmp": exact_stage_reaction_cmp,
        "reaction_nolift_cmp": reaction_nolift_cmp,
        "reaction_lift_cmp": reaction_lift_cmp,
        "criteria": {
            "target_quantity_rel_l2_le_1e-12": bool(float(exact_stage_reaction_cmp["rel_l2"]) <= 1.0e-12),
            "reaction_nolift_rel_l2_le_1e-6": bool(float(reaction_nolift_cmp["rel_l2"]) <= 1.0e-6),
            "reaction_lift_worse_than_nolift": bool(float(reaction_lift_cmp["rel_l2"]) > float(reaction_nolift_cmp["rel_l2"])),
        },
    }
    summary["criteria_all_pass"] = bool(all(summary["criteria"].values()))
    dump_json(summary, Path(args.output))
    print(f"reaction reconstruction check: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
