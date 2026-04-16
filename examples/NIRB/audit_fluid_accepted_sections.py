from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import examples.NIRB.run_example2_local as ex
from examples.NIRB.build_local_checkpoint_from_step_history import _assign_scalar_function, _assign_vector_function
from examples.NIRB.compare_example2_step_history import _compare_fields
from examples.NIRB.dvms import (
    _advance_fluid_dvms_history_after_step,
    _update_fluid_dvms_predicted_subscale,
    _update_fluid_dvms_state_from_previous_step,
)
from examples.NIRB.example2_local_setup import load_example2_local_setup


def _dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported JSON type: {type(value).__name__}")

    path.write_text(json.dumps(data, indent=2, default=default), encoding="utf-8")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_step(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {str(key): np.asarray(data[key]) for key in data.files}


def _resolve_step_history_dir(path: Path) -> Path:
    path = Path(path).resolve()
    if path.is_dir() and (path / "step_history").is_dir():
        return path / "step_history"
    return path


def _step_file(step_dir: Path, step: int) -> Path:
    path = step_dir / f"step{int(step):04d}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit accepted-step exact fluid reaction compatibility against Kratos step history.")
    parser.add_argument(
        "--kratos-step-dir",
        type=Path,
        default=Path("examples/NIRB/artifacts/kratos_step_history_coupled_qstate_5steps_20260415/step_history"),
    )
    parser.add_argument("--step-start", type=int, default=1)
    parser.add_argument("--step-end", type=int, default=5)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("examples/NIRB/artifacts/fluid_accepted_section_audit_20260415.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("examples/NIRB/artifacts/fluid_accepted_section_audit_20260415.csv"),
    )
    parser.add_argument("--backend", choices=("python", "jit", "cpp"), default="cpp")
    parser.add_argument("--quad-order", type=int, default=1)
    parser.add_argument("--bossak-alpha", type=float, default=-0.3)
    parser.add_argument("--dynamic-tau", type=float, default=1.0)
    parser.add_argument(
        "--contribution-mode",
        choices=("velocity", "system", "system_condensed"),
        default="system",
    )
    parser.add_argument(
        "--apply-dirichlet-lift",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--old-mass-geometry",
        choices=("current", "previous"),
        default="current",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = main()

    setup = load_example2_local_setup()
    step_dir = _resolve_step_history_dir(Path(args.kratos_step_dir))
    mesh_f, _ = ex._load_reference_partitioned_meshes(setup=setup)
    fluid = ex._build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=int(args.quad_order))

    mu_f = float(setup.material.density * setup.material.kinematic_viscosity)
    dt = float(setup.boundaries.time_step)
    bossak = ex._bossak_coefficients(alpha=float(args.bossak_alpha), dt=dt)

    fluid_coords = np.asarray(mesh_f.nodes_x_y_pos, dtype=float)
    zero_vec = np.zeros((fluid_coords.shape[0], 2), dtype=float)
    prev_prev_mesh_disp = np.zeros_like(zero_vec)
    prev_mesh_disp = np.zeros_like(zero_vec)
    prev_mesh_vel = np.zeros_like(zero_vec)
    prev_mesh_acc = np.zeros_like(zero_vec)
    prev_fluid_vel = np.zeros_like(zero_vec)
    prev_fluid_acc = np.zeros_like(zero_vec)
    prev_pressure = np.zeros((fluid_coords.shape[0], 1), dtype=float)

    rows: list[dict[str, Any]] = []
    first_bad_step: int | None = None

    warm_start = max(1, int(args.step_start))
    for step in range(1, int(args.step_end) + 1):
        data = _load_step(_step_file(step_dir, step))
        t_now = float(np.asarray(data["time_s"], dtype=float).reshape(-1)[0])

        fluid_vel = np.asarray(data["fluid_velocity_nodal_values"], dtype=float)
        fluid_p = np.asarray(data["fluid_pressure_nodal_values"], dtype=float).reshape(-1, 1)
        mesh_disp = np.asarray(data["fluid_mesh_displacement_nodal_values"], dtype=float)
        iface_vel_coords = np.asarray(data["interface_velocity_coords_ref"], dtype=float)
        iface_vel_vals = np.asarray(data["interface_velocity_values"], dtype=float)
        load_ref_coords = np.asarray(data["fluid_load_coords_ref"], dtype=float)
        load_ref_vals = np.asarray(data["fluid_load_values"], dtype=float)

        if "fluid_acceleration_nodal_values" in data:
            fluid_acc = np.asarray(data["fluid_acceleration_nodal_values"], dtype=float)
        elif step == 1:
            fluid_acc = float(bossak["ma0"]) * fluid_vel
        else:
            fluid_acc = float(bossak["ma0"]) * (fluid_vel - prev_fluid_vel) + float(bossak["ma2"]) * prev_fluid_acc
        # The production driver advances mesh history with Bossak kinematics from
        # displacement history. Reusing the dumped mesh velocity is fine, but
        # reconstructing mesh acceleration by a plain time difference is not:
        # it drifts away from the scheme-consistent `a_mesh_prev` that enters the
        # DVMS predictor and reaction helper.
        mesh_vel, mesh_acc = ex._bossak_displacement_kinematics_values(
            d_curr=mesh_disp,
            d_prev=prev_mesh_disp,
            v_prev=prev_mesh_vel,
            a_prev=prev_mesh_acc,
            dt=dt,
            alpha=float(args.bossak_alpha),
        )

        _assign_vector_function(fluid["u_prev"], source_coords=fluid_coords, point_values=prev_fluid_vel)
        _assign_vector_function(fluid["u_k"], source_coords=fluid_coords, point_values=fluid_vel)
        _assign_scalar_function(fluid["p_prev"], source_coords=fluid_coords, point_values=prev_pressure)
        _assign_scalar_function(fluid["p_k"], source_coords=fluid_coords, point_values=fluid_p)
        _assign_vector_function(fluid["a_prev"], source_coords=fluid_coords, point_values=prev_fluid_acc)
        _assign_vector_function(fluid["d_prev2"], source_coords=fluid_coords, point_values=prev_prev_mesh_disp)
        _assign_vector_function(fluid["d_prev"], source_coords=fluid_coords, point_values=prev_mesh_disp)
        _assign_vector_function(fluid["d_mesh"], source_coords=fluid_coords, point_values=mesh_disp)
        _assign_vector_function(fluid["w_mesh_prev"], source_coords=fluid_coords, point_values=prev_mesh_vel)
        _assign_vector_function(fluid["a_mesh_prev"], source_coords=fluid_coords, point_values=prev_mesh_acc)

        _update_fluid_dvms_state_from_previous_step(
            state=fluid["dvms_state"],
            dh=fluid["dh"],
            mesh=mesh_f,
            u_prev=fluid["u_prev"],
            d_prev=fluid["d_prev"],
            d_geo=fluid["d_mesh"] if str(args.old_mass_geometry) == "current" else fluid["d_prev"],
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
            d_prev2=fluid["d_prev2"],
            mesh_v_prev=fluid["w_mesh_prev"],
            mesh_a_prev=fluid["a_mesh_prev"],
            rho_f=float(setup.material.density),
            mu_f=mu_f,
            dt=dt,
            bossak_alpha=float(args.bossak_alpha),
            dynamic_tau=float(args.dynamic_tau),
            backend=str(args.backend),
        )

        iface_velocity_lookup = ex.CoordinateLookup(iface_vel_coords, iface_vel_vals, dim=2)

        def inlet_profile(x: float, y: float) -> float:
            del x
            return setup.geometry.inlet_velocity(y, t_now, reference_velocity=float(setup.material.max_velocity))

        fluid_bcs, fluid_bcs_homog = ex._fluid_boundary_conditions(
            iface_velocity=iface_velocity_lookup,
            inlet_lookup=inlet_profile,
            interface_tag=setup.geometry.interface_tag,
            outlet_tag=setup.geometry.outlet_tag,
            walls_tag=setup.geometry.walls_tag,
            cylinder_tag=setup.geometry.cylinder_tag,
        )
        fluid["_current_bcs"] = fluid_bcs
        fluid["_current_bcs_homog"] = fluid_bcs_homog

        reaction_lookup = ex._fluid_interface_reaction_loads(
            prob=fluid,
            rho_f=float(setup.material.density),
            mu_f=mu_f,
            dt=dt,
            quad_order=int(args.quad_order),
            bossak_alpha=float(args.bossak_alpha),
            dynamic_tau=float(args.dynamic_tau),
            interface_tag=setup.geometry.interface_tag,
            backend=str(args.backend),
            contribution_mode=str(args.contribution_mode),
            apply_dirichlet_lift=bool(args.apply_dirichlet_lift),
        )

        velocity_cmp = _compare_fields(
            np.asarray(data["fluid_coords_ref"], dtype=float),
            np.asarray(data["fluid_velocity_nodal_values"], dtype=float),
            *ex._vector_field_matrix(fluid["dh"], fluid["u_k"]),
        )
        pressure_ids = np.asarray(fluid["dh"].get_field_slice(fluid["p_k"].field_name), dtype=int)
        fluid["dh"]._ensure_dof_coords()
        p_local_coords = np.asarray(fluid["dh"]._dof_coords[pressure_ids], dtype=float)
        p_local_vals = np.asarray(fluid["p_k"].get_nodal_values(pressure_ids), dtype=float).reshape(-1, 1)
        pressure_cmp = _compare_fields(
            np.asarray(data["fluid_coords_ref"], dtype=float),
            np.asarray(data["fluid_pressure_nodal_values"], dtype=float),
            p_local_coords,
            p_local_vals,
        )
        reaction_cmp = _compare_fields(
            load_ref_coords,
            load_ref_vals,
            np.asarray(reaction_lookup.coords, dtype=float),
            np.asarray(reaction_lookup.values, dtype=float),
        )

        if step >= warm_start:
            row = {
                "step": int(step),
                "time_s": float(t_now),
                "fluid_velocity_rel_l2": float(velocity_cmp["rel_l2"]),
                "fluid_pressure_rel_l2": float(pressure_cmp["rel_l2"]),
                "fluid_reaction_rel_l2": float(reaction_cmp["rel_l2"]),
                "fluid_reaction_abs_max": float(reaction_cmp["abs_max"]),
                "fluid_reaction_cosine": float(reaction_cmp["cosine"]),
                "fluid_reaction_reference_max_norm": float(reaction_cmp["reference_max_norm"]),
                "fluid_reaction_local_max_norm": float(reaction_cmp["local_max_norm"]),
            }
            rows.append(row)
            if first_bad_step is None and float(reaction_cmp["rel_l2"]) > 5.0e-3:
                first_bad_step = int(step)

        _advance_fluid_dvms_history_after_step(fluid["dvms_state"])
        prev_prev_mesh_disp = np.asarray(prev_mesh_disp, dtype=float).copy()
        prev_mesh_disp = np.asarray(mesh_disp, dtype=float).copy()
        prev_mesh_vel = np.asarray(mesh_vel, dtype=float).copy()
        prev_mesh_acc = np.asarray(mesh_acc, dtype=float).copy()
        prev_fluid_vel = np.asarray(fluid_vel, dtype=float).copy()
        prev_fluid_acc = np.asarray(fluid_acc, dtype=float).copy()
        prev_pressure = np.asarray(fluid_p, dtype=float).copy()

    payload = {
        "kratos_step_dir": str(step_dir),
        "contribution_mode": str(args.contribution_mode),
        "apply_dirichlet_lift": bool(args.apply_dirichlet_lift),
        "old_mass_geometry": str(args.old_mass_geometry),
        "steps_compared": [int(r["step"]) for r in rows],
        "first_bad_reaction_step_gt_5e-3": first_bad_step,
        "rows": rows,
    }
    _dump_json(payload, Path(args.output_json).resolve())
    _write_csv(rows, Path(args.output_csv).resolve())
    print(Path(args.output_json).resolve())
