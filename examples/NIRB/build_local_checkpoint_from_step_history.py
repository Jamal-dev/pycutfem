from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np

from examples.NIRB.run_example2_local import (
    CoordinateLookup,
    _boundary_field_data,
    _bossak_coefficients,
    _bossak_displacement_kinematics_values,
    _checkpoint_payload,
    _coord_key,
    _load_reference_partitioned_meshes,
    _build_fluid_problem,
    _build_solid_problem,
)
from examples.NIRB.dvms import (
    _advance_fluid_dvms_history_after_step,
    _update_fluid_dvms_predicted_subscale,
    _update_fluid_dvms_state_from_previous_step,
)
from examples.NIRB.example2_local_setup import load_example2_local_setup


def _resolve_step_history_dir(path: Path) -> Path:
    path = Path(path).resolve()
    if path.is_dir() and (path / "step_history").is_dir():
        return path / "step_history"
    return path


def _step_file(step_history_dir: Path, step: int) -> Path:
    path = step_history_dir / f"step{int(step):04d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Step-history file not found: {path}")
    return path


def _lookup_sample(source_coords: np.ndarray, source_values: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
    source_coords = np.asarray(source_coords, dtype=float)
    source_values = np.asarray(source_values, dtype=float)
    target_coords = np.asarray(target_coords, dtype=float)
    if source_values.ndim == 1:
        source_values = source_values.reshape(-1, 1)
    exact = {_coord_key(x, y): source_values[i].copy() for i, (x, y) in enumerate(source_coords)}
    out = np.empty((target_coords.shape[0], source_values.shape[1]), dtype=float)
    for i, (x, y) in enumerate(target_coords):
        hit = exact.get(_coord_key(x, y))
        if hit is not None:
            out[i, :] = hit
            continue
        dist2 = np.sum((source_coords - target_coords[i][None, :]) ** 2, axis=1)
        out[i, :] = source_values[int(np.argmin(dist2)), :]
    return out


def _assign_vector_function(
    vector_fn,
    *,
    source_coords: np.ndarray,
    point_values: np.ndarray,
) -> np.ndarray:
    point_values = np.asarray(point_values, dtype=float)
    if point_values.ndim != 2 or point_values.shape[1] != 2:
        raise ValueError("point_values must have shape (n, 2)")
    dh = vector_fn._dof_handler
    x_field = vector_fn.components[0].field_name
    y_field = vector_fn.components[1].field_name
    x_coords = np.asarray(dh.get_dof_coords(x_field), dtype=float)
    y_coords = np.asarray(dh.get_dof_coords(y_field), dtype=float)
    x_vals = _lookup_sample(source_coords, point_values[:, 0], x_coords).reshape(-1)
    y_vals = _lookup_sample(source_coords, point_values[:, 1], y_coords).reshape(-1)
    vector_fn.components[0].nodal_values[:] = x_vals
    vector_fn.components[1].nodal_values[:] = y_vals
    x_slice = np.asarray(dh.get_field_slice(x_field), dtype=int)
    y_slice = np.asarray(dh.get_field_slice(y_field), dtype=int)
    vector_fn.set_nodal_values(x_slice, x_vals)
    vector_fn.set_nodal_values(y_slice, y_vals)
    return np.asarray(vector_fn.nodal_values, dtype=float).copy()


def _assign_scalar_function(
    function,
    *,
    source_coords: np.ndarray,
    point_values: np.ndarray,
) -> np.ndarray:
    dh = function._dof_handler
    field_coords = np.asarray(dh.get_dof_coords(function.field_name), dtype=float)
    nodal_values = _lookup_sample(source_coords, point_values, field_coords).reshape(-1)
    function.nodal_values[:] = nodal_values
    return np.asarray(function.nodal_values, dtype=float).copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local Example 2 restart checkpoint from accepted-step history files.")
    parser.add_argument("--step-history-dir", type=Path, required=True, help="Kratos/local step_history dir or its parent run dir.")
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory that will receive checkpoints/checkpoint_step_XXXX.npz")
    parser.add_argument("--mesh-size", type=float, default=0.04)
    parser.add_argument("--mesh-order", type=int, default=1)
    parser.add_argument("--poly-order", type=int, default=1)
    parser.add_argument("--pressure-order", type=int, default=1)
    parser.add_argument(
        "--quad-order",
        type=int,
        default=None,
        help=(
            "Fluid DVMS quadrature order to reconstruct in the checkpoint. "
            "Default: 1 (Kratos-matched 3-point triangle rule)."
        ),
    )
    parser.add_argument("--dt", type=float, default=0.008)
    parser.add_argument("--bossak-alpha", type=float, default=-0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    step_history_dir = _resolve_step_history_dir(Path(args.step_history_dir))
    target_step = int(args.target_step)
    if target_step < 1:
        raise ValueError("target-step must be >= 1")

    setup = load_example2_local_setup(reference_root=None, mesh_size_default=float(args.mesh_size), mesh_order_default=int(args.mesh_order))
    mesh_f, mesh_s = _load_reference_partitioned_meshes(setup=setup)
    quad_order = int(args.quad_order) if args.quad_order is not None else 1
    fluid = _build_fluid_problem(mesh_f, poly_order=int(args.poly_order), pressure_order=int(args.pressure_order), quadrature_order=quad_order)
    solid = _build_solid_problem(mesh_s, poly_order=int(args.poly_order))
    solid_iface_coords_local, _ = _boundary_field_data(solid["dh"], "dx", setup.geometry.interface_tag)
    if solid_iface_coords_local.size == 0:
        raise RuntimeError("Failed to extract local solid interface coordinates for restart conversion.")

    bossak = _bossak_coefficients(alpha=float(args.bossak_alpha), dt=float(args.dt))

    fluid_coords = np.asarray(mesh_f.nodes_x_y_pos, dtype=float)
    solid_coords = np.asarray(mesh_s.nodes_x_y_pos, dtype=float)
    fluid_u_prev = np.zeros((fluid_coords.shape[0], 2), dtype=float)
    fluid_a_prev = np.zeros_like(fluid_u_prev)
    mesh_d_prev = np.zeros((fluid_coords.shape[0], 2), dtype=float)
    mesh_v_prev = np.zeros_like(mesh_d_prev)
    mesh_a_prev = np.zeros_like(mesh_d_prev)
    solid_d_prev = np.zeros((solid_coords.shape[0], 2), dtype=float)
    interface_load = None
    interface_load_coords = None
    time_value = 0.0
    prev_mesh_d = np.zeros_like(mesh_d_prev)

    for step in range(1, target_step + 1):
        with np.load(_step_file(step_history_dir, step)) as data:
            fluid_u = _lookup_sample(
                np.asarray(data["fluid_coords_ref"], dtype=float),
                np.asarray(data["fluid_velocity_nodal_values"], dtype=float),
                fluid_coords,
            )
            fluid_p = _lookup_sample(
                np.asarray(data["fluid_coords_ref"], dtype=float),
                np.asarray(data["fluid_pressure_nodal_values"], dtype=float),
                fluid_coords,
            ).reshape(-1)
            mesh_d = _lookup_sample(
                np.asarray(data["fluid_coords_ref"], dtype=float),
                np.asarray(data["fluid_mesh_displacement_nodal_values"], dtype=float),
                fluid_coords,
            )
            solid_d = _lookup_sample(
                np.asarray(data["structure_coords_ref"], dtype=float),
                np.asarray(data["structure_displacement_nodal_values"], dtype=float),
                solid_coords,
            )
            interface_coords = np.asarray(data["interface_load_coords_ref"], dtype=float)
            if "structure_point_load_nodal_values" in data and "structure_coords_ref" in data:
                interface_load = _lookup_sample(
                    np.asarray(data["structure_coords_ref"], dtype=float),
                    np.asarray(data["structure_point_load_nodal_values"], dtype=float),
                    solid_iface_coords_local,
                )
            else:
                interface_load = _lookup_sample(
                    interface_coords,
                    -np.asarray(data["interface_load_values"], dtype=float),
                    solid_iface_coords_local,
                )
            interface_load_coords = solid_iface_coords_local
            time_value = float(np.asarray(data["time_s"], dtype=float).reshape(-1)[0])

        if step == 1:
            fluid_a_prev = float(bossak["ma0"]) * fluid_u
        else:
            fluid_a_prev = float(bossak["ma0"]) * (fluid_u - fluid_u_prev) + float(bossak["ma2"]) * fluid_a_prev
        mesh_v, mesh_a = _bossak_displacement_kinematics_values(
            d_curr=mesh_d,
            d_prev=mesh_d_prev,
            v_prev=mesh_v_prev,
            a_prev=mesh_a_prev,
            dt=float(args.dt),
            alpha=float(args.bossak_alpha),
        )
        fluid_u_prev = fluid_u
        mesh_v_prev = mesh_v
        mesh_a_prev = mesh_a
        prev_mesh_d = mesh_d_prev
        mesh_d_prev = mesh_d
        solid_d_prev = solid_d

    if interface_load is None:
        raise RuntimeError("No accepted-step interface load found in step history.")

    _assign_vector_function(fluid["u_prev"], source_coords=fluid_coords, point_values=fluid_u_prev)
    _assign_vector_function(fluid["u_k"], source_coords=fluid_coords, point_values=fluid_u_prev)
    _assign_scalar_function(fluid["p_prev"], source_coords=fluid_coords, point_values=fluid_p)
    _assign_scalar_function(fluid["p_k"], source_coords=fluid_coords, point_values=fluid_p)
    _assign_vector_function(fluid["d_prev"], source_coords=fluid_coords, point_values=mesh_d_prev)
    _assign_vector_function(fluid["d_mesh"], source_coords=fluid_coords, point_values=mesh_d_prev)
    _assign_vector_function(fluid["d_prev2"], source_coords=fluid_coords, point_values=prev_mesh_d)
    _assign_vector_function(fluid["w_mesh_prev"], source_coords=fluid_coords, point_values=mesh_v_prev)
    _assign_vector_function(fluid["a_mesh_prev"], source_coords=fluid_coords, point_values=mesh_a_prev)
    _assign_vector_function(fluid["a_prev"], source_coords=fluid_coords, point_values=fluid_a_prev)
    _assign_vector_function(solid["d_prev"], source_coords=solid_coords, point_values=solid_d_prev)
    _assign_vector_function(solid["d_k"], source_coords=solid_coords, point_values=solid_d_prev)
    _update_fluid_dvms_state_from_previous_step(
        state=fluid["dvms_state"],
        dh=fluid["dh"],
        mesh=mesh_f,
        u_prev=fluid["u_prev"],
        d_prev=fluid["d_prev"],
        d_geo=fluid["d_mesh"],
        backend="cpp",
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
        mu_f=float(setup.material.density * setup.material.kinematic_viscosity),
        dt=float(args.dt),
        bossak_alpha=float(args.bossak_alpha),
        dynamic_tau=1.0,
        backend="cpp",
    )
    _advance_fluid_dvms_history_after_step(fluid["dvms_state"])

    if interface_load_coords is None:
        raise RuntimeError("No accepted-step interface load coordinates found in step history.")
    solid_iface_coords = np.asarray(interface_load_coords, dtype=float)
    current_load_lookup = CoordinateLookup(solid_iface_coords, np.asarray(interface_load, dtype=float), dim=2)
    payload = _checkpoint_payload(
        step=target_step,
        time_s=time_value,
        solid=solid,
        fluid=fluid,
        current_load_lookup=current_load_lookup,
        iqn_old_dr_mats=deque(maxlen=0),
        iqn_old_dg_mats=deque(maxlen=0),
    )

    checkpoint_dir = Path(args.output_dir).resolve() / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{target_step:04d}.npz"
    np.savez_compressed(checkpoint_path, **payload)
    (checkpoint_dir / "latest_checkpoint.txt").write_text(checkpoint_path.name, encoding="utf-8")
    print(f"checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
