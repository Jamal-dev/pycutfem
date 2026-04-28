from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np

import examples.NIRB.run_example2_local as ex
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
    _update_fluid_dvms_oss_projections,
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


def _accepted_step_old_subscale_qstate(
    step_data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    if "fluid_old_subscale_velocity_flat" in step_data and "fluid_q_coords_ref_flat" in step_data:
        q_coords = np.asarray(step_data["fluid_q_coords_ref_flat"], dtype=float).reshape(-1, 2)
        q_subscale = np.asarray(step_data["fluid_old_subscale_velocity_flat"], dtype=float).reshape(-1, 2)
        return q_coords, q_subscale
    if "fluid_old_subscale_velocity" in step_data and "fluid_q_coords_ref" in step_data:
        q_coords = np.asarray(step_data["fluid_q_coords_ref"], dtype=float).reshape(-1, 2)
        q_subscale = np.asarray(step_data["fluid_old_subscale_velocity"], dtype=float).reshape(-1, 2)
        return q_coords, q_subscale
    return None


def _accepted_step_reported_subscale_qstate(
    step_data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    if "fluid_q_coords_ref_flat" in step_data and "fluid_subscale_velocity_flat" in step_data:
        q_coords = np.asarray(step_data["fluid_q_coords_ref_flat"], dtype=float).reshape(-1, 2)
        q_subscale = np.asarray(step_data["fluid_subscale_velocity_flat"], dtype=float).reshape(-1, 2)
        return q_coords, q_subscale
    if "fluid_q_coords_ref" in step_data and "fluid_subscale_velocity" in step_data:
        q_coords = np.asarray(step_data["fluid_q_coords_ref"], dtype=float).reshape(-1, 2)
        q_subscale = np.asarray(step_data["fluid_subscale_velocity"], dtype=float).reshape(-1, 2)
        return q_coords, q_subscale
    return None


def _accepted_step_projection_state(
    step_data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if (
        "fluid_coords_ref" in step_data
        and "fluid_advproj_nodal_values" in step_data
        and "fluid_divproj_nodal_values" in step_data
    ):
        coords = np.asarray(step_data["fluid_coords_ref"], dtype=float).reshape(-1, 2)
        advproj = np.asarray(step_data["fluid_advproj_nodal_values"], dtype=float).reshape(-1, 2)
        divproj = np.asarray(step_data["fluid_divproj_nodal_values"], dtype=float).reshape(-1)
        return coords, advproj, divproj
    return None


def _resolve_monitor_dir(path: Path | None) -> Path | None:
    if path is None:
        return None
    path = Path(path).resolve()
    if path.is_dir() and (path / "coupling_monitor").is_dir():
        return path / "coupling_monitor"
    return path


def _monitor_iqn_file_candidates(monitor_dir: Path, step: int, iteration: int) -> tuple[Path, ...]:
    stem = f"step{int(step):04d}_iter{int(iteration):04d}"
    # Kratos only forms a new IQN block for iterations that call
    # ComputeAndApplyUpdate. The final converged iteration has an
    # after_iteration monitor dump but no pre_update dump, so using only
    # pre_update records avoids adding a non-existent final IQN column.
    return (monitor_dir / f"{stem}_pre_update.npz",)


def _reshape_monitor_interface(values: np.ndarray, n_pts: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != int(n_pts) * 2:
        raise ValueError(
            f"Expected flattened interface vector of length {int(n_pts) * 2}, got {arr.size}."
        )
    return arr.reshape(int(n_pts), 2)


def _monitor_iqn_iteration_pair(
    monitor_dir: Path,
    step: int,
    iteration: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    for path in _monitor_iqn_file_candidates(monitor_dir, step, iteration):
        if not path.exists():
            continue
        with np.load(path) as data:
            coords = np.asarray(data["fluid_load_coords_ref"], dtype=float)
            prev_key = None
            curr_key = None
            for prefix in ("crit_0_fluid_load", "acc_0_fluid_load"):
                cand_prev = f"{prefix}_previous"
                cand_curr = f"{prefix}_current"
                if cand_prev in data and cand_curr in data:
                    prev_key = cand_prev
                    curr_key = cand_curr
                    break
            if prev_key is None or curr_key is None:
                continue
            x_curr = _reshape_monitor_interface(np.asarray(data[prev_key], dtype=float), int(coords.shape[0]))
            g_curr = _reshape_monitor_interface(np.asarray(data[curr_key], dtype=float), int(coords.shape[0]))
            return coords, x_curr, g_curr
    return None


def _rebuild_iqn_history_from_monitor(
    *,
    monitor_dir: Path | None,
    target_step: int,
    force_history: int,
    iteration_horizon: int,
    fluid_iface_coords_local: np.ndarray | None = None,
) -> tuple[deque[np.ndarray], deque[np.ndarray]]:
    maxlen = max(int(force_history) - 1, 0)
    dr_hist: deque[np.ndarray] = deque(maxlen=maxlen)
    dg_hist: deque[np.ndarray] = deque(maxlen=maxlen)
    if monitor_dir is None or maxlen <= 0:
        return dr_hist, dg_hist

    target_coords = None if fluid_iface_coords_local is None else np.asarray(fluid_iface_coords_local, dtype=float)

    for step in range(1, int(target_step) + 1):
        x_history: list[np.ndarray] = []
        g_history: list[np.ndarray] = []
        iteration = 1
        while True:
            pair = _monitor_iqn_iteration_pair(monitor_dir, step, iteration)
            if pair is None:
                break
            coords, x_curr, g_curr = pair
            if target_coords is not None:
                x_curr = _lookup_sample(coords, x_curr, target_coords)
                g_curr = _lookup_sample(coords, g_curr, target_coords)
            x_history.append(x_curr)
            g_history.append(g_curr)
            iteration += 1

        v_new, w_new = ex._iqnils_iteration_matrices(
            x_history=x_history,
            g_history=g_history,
            iteration_horizon=int(iteration_horizon),
        )
        if v_new is not None and w_new is not None:
            dr_hist.appendleft(np.asarray(v_new, dtype=float).copy())
            dg_hist.appendleft(np.asarray(w_new, dtype=float).copy())

    return dr_hist, dg_hist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local Example 2 restart checkpoint from accepted-step history files.")
    parser.add_argument("--step-history-dir", type=Path, required=True, help="Kratos/local step_history dir or its parent run dir.")
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory that will receive checkpoints/checkpoint_step_XXXX.npz")
    parser.add_argument(
        "--coupling-monitor-dir",
        type=Path,
        default=None,
        help="Optional Kratos coupling_monitor dir (or its parent run dir) used to rebuild IQNILS memory.",
    )
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
    parser.add_argument("--force-history", type=int, default=3)
    parser.add_argument("--force-iteration-horizon", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    step_history_dir = _resolve_step_history_dir(Path(args.step_history_dir))
    monitor_dir = _resolve_monitor_dir(args.coupling_monitor_dir)
    target_step = int(args.target_step)
    if target_step < 1:
        raise ValueError("target-step must be >= 1")

    setup = load_example2_local_setup(reference_root=None, mesh_size_default=float(args.mesh_size), mesh_order_default=int(args.mesh_order))
    mesh_f, mesh_s = _load_reference_partitioned_meshes(setup=setup)
    quad_order = int(args.quad_order) if args.quad_order is not None else 1
    fluid = _build_fluid_problem(mesh_f, poly_order=int(args.poly_order), pressure_order=int(args.pressure_order), quadrature_order=quad_order)
    solid = _build_solid_problem(mesh_s, poly_order=int(args.poly_order))
    fluid_iface_coords_local, _ = _boundary_field_data(fluid["dh"], "ux", setup.geometry.interface_tag)
    solid_iface_coords_local, _ = _boundary_field_data(solid["dh"], "dx", setup.geometry.interface_tag)
    if fluid_iface_coords_local.size == 0:
        raise RuntimeError("Failed to extract local fluid interface coordinates for restart conversion.")
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
    current_load_values = None
    target_step_data: dict[str, np.ndarray] | None = None
    time_value = 0.0
    prev_mesh_d = np.zeros_like(mesh_d_prev)

    for step in range(1, target_step + 1):
        with np.load(_step_file(step_history_dir, step)) as data:
            step_data = {str(key): np.asarray(data[key]) for key in data.files}
            fluid_u = _lookup_sample(
                np.asarray(step_data["fluid_coords_ref"], dtype=float),
                np.asarray(step_data["fluid_velocity_nodal_values"], dtype=float),
                fluid_coords,
            )
            fluid_p = _lookup_sample(
                np.asarray(step_data["fluid_coords_ref"], dtype=float),
                np.asarray(step_data["fluid_pressure_nodal_values"], dtype=float),
                fluid_coords,
            ).reshape(-1)
            mesh_d = _lookup_sample(
                np.asarray(step_data["fluid_coords_ref"], dtype=float),
                np.asarray(step_data["fluid_mesh_displacement_nodal_values"], dtype=float),
                fluid_coords,
            )
            solid_d = _lookup_sample(
                np.asarray(step_data["structure_coords_ref"], dtype=float),
                np.asarray(step_data["structure_displacement_nodal_values"], dtype=float),
                solid_coords,
            )
            if "fluid_load_values" in step_data and "fluid_load_coords_ref" in step_data:
                current_load_values = _lookup_sample(
                    np.asarray(step_data["fluid_load_coords_ref"], dtype=float),
                    np.asarray(step_data["fluid_load_values"], dtype=float),
                    fluid_iface_coords_local,
                )
            elif "structure_point_load_nodal_values" in step_data and "structure_coords_ref" in step_data:
                current_load_values = _lookup_sample(
                    np.asarray(step_data["structure_coords_ref"], dtype=float),
                    -np.asarray(step_data["structure_point_load_nodal_values"], dtype=float),
                    fluid_iface_coords_local,
                )
            else:
                current_load_values = _lookup_sample(
                    np.asarray(step_data["interface_load_coords_ref"], dtype=float),
                    -np.asarray(step_data["interface_load_values"], dtype=float),
                    fluid_iface_coords_local,
                )
            time_value = float(np.asarray(step_data["time_s"], dtype=float).reshape(-1)[0])
            if step == target_step:
                target_step_data = step_data

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

    if current_load_values is None:
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
    _assign_vector_function(fluid["a_k"], source_coords=fluid_coords, point_values=fluid_a_prev)
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
    qstate = _accepted_step_old_subscale_qstate(target_step_data or {})
    reported_qstate = _accepted_step_reported_subscale_qstate(target_step_data or {})
    projection_state = _accepted_step_projection_state(target_step_data or {})
    if qstate is not None:
        q_coords_ref, q_subscale = qstate
        local_q_coords = np.asarray(fluid["dvms_state"].sample_coords, dtype=float)
        sampled = _lookup_sample(
            np.asarray(q_coords_ref, dtype=float),
            np.asarray(q_subscale, dtype=float),
            local_q_coords,
        )
        fluid["dvms_state"].old_subscale_velocity[:, :] = np.asarray(sampled, dtype=float)
        fluid["dvms_state"].predicted_subscale_velocity[:, :] = np.asarray(sampled, dtype=float)
        if projection_state is not None:
            proj_coords, advproj_vals, divproj_vals = projection_state
            local_node_coords = np.asarray(mesh_f.nodes_x_y_pos, dtype=float)
            fluid["dvms_state"]._nodal_momentum_projection = _lookup_sample(
                np.asarray(proj_coords, dtype=float),
                np.asarray(advproj_vals, dtype=float),
                local_node_coords,
            ).reshape(-1, 2)
            div_local = _lookup_sample(
                np.asarray(proj_coords, dtype=float),
                np.asarray(divproj_vals, dtype=float).reshape(-1, 1),
                local_node_coords,
            ).reshape(-1)
            fluid["dvms_state"]._nodal_div_projection = np.asarray(div_local, dtype=float).copy()
            fluid["dvms_state"]._prev_nodal_div_projection = np.asarray(div_local, dtype=float).copy()
        _update_fluid_dvms_oss_projections(
            state=fluid["dvms_state"],
            dh=fluid["dh"],
            mesh=mesh_f,
            u_k=fluid["u_k"],
            p_k=fluid["p_k"],
            d_mesh=fluid["d_mesh"],
            d_prev=fluid["d_prev"],
            d_prev2=fluid["d_prev2"],
            mesh_v=fluid["w_mesh_prev"],
            mesh_v_prev=fluid["w_mesh_prev"],
            mesh_a_prev=fluid["a_mesh_prev"],
            rho_f=float(setup.material.density),
            dt=float(args.dt),
            bossak_alpha=float(args.bossak_alpha),
        )
        fluid["dvms_state"].sync_coefficients_from_samples()
    elif reported_qstate is not None:
        q_coords_ref, q_subscale = reported_qstate
        local_q_coords = np.asarray(fluid["dvms_state"].sample_coords, dtype=float)
        sampled = _lookup_sample(
            np.asarray(q_coords_ref, dtype=float),
            np.asarray(q_subscale, dtype=float),
            local_q_coords,
        )
        fluid["dvms_state"].old_subscale_velocity[:, :] = np.asarray(sampled, dtype=float)
        fluid["dvms_state"].predicted_subscale_velocity[:, :] = np.asarray(sampled, dtype=float)
        if projection_state is not None:
            proj_coords, advproj_vals, divproj_vals = projection_state
            local_node_coords = np.asarray(mesh_f.nodes_x_y_pos, dtype=float)
            fluid["dvms_state"]._nodal_momentum_projection = _lookup_sample(
                np.asarray(proj_coords, dtype=float),
                np.asarray(advproj_vals, dtype=float),
                local_node_coords,
            ).reshape(-1, 2)
            div_local = _lookup_sample(
                np.asarray(proj_coords, dtype=float),
                np.asarray(divproj_vals, dtype=float).reshape(-1, 1),
                local_node_coords,
            ).reshape(-1)
            fluid["dvms_state"]._nodal_div_projection = np.asarray(div_local, dtype=float).copy()
            fluid["dvms_state"]._prev_nodal_div_projection = np.asarray(div_local, dtype=float).copy()
        _update_fluid_dvms_oss_projections(
            state=fluid["dvms_state"],
            dh=fluid["dh"],
            mesh=mesh_f,
            u_k=fluid["u_k"],
            p_k=fluid["p_k"],
            d_mesh=fluid["d_mesh"],
            d_prev=fluid["d_prev"],
            d_prev2=fluid["d_prev2"],
            mesh_v=fluid["w_mesh_prev"],
            mesh_v_prev=fluid["w_mesh_prev"],
            mesh_a_prev=fluid["a_mesh_prev"],
            rho_f=float(setup.material.density),
            dt=float(args.dt),
            bossak_alpha=float(args.bossak_alpha),
        )
        fluid["dvms_state"].sync_coefficients_from_samples()
    else:
        # Accepted-step Kratos step_history stores SUBSCALE_VELOCITY after
        # OutputSolutionStep(). That is a post-finalize report, not the hidden
        # old_subscale history carried into the next step, so do not seed the
        # restart from those fields unless an explicit old-subscale dump exists.
        _ = _accepted_step_reported_subscale_qstate(target_step_data or {})
        _update_fluid_dvms_predicted_subscale(
            state=fluid["dvms_state"],
            dh=fluid["dh"],
            mesh=mesh_f,
            u_k=fluid["u_k"],
            u_prev=fluid["u_prev"],
            a_prev=fluid["a_prev"],
            a_curr=fluid["a_k"],
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
        mesh_v_curr = ex.VectorFunction("mesh_v_curr_finalize", ["mx", "my"], dof_handler=fluid["dh"])
        _assign_vector_function(
            mesh_v_curr,
            source_coords=fluid_coords,
            point_values=np.asarray(mesh_v_prev, dtype=float),
        )
        _advance_fluid_dvms_history_after_step(
            fluid["dvms_state"],
            dh=fluid["dh"],
            mesh=mesh_f,
            u_curr=fluid["u_k"],
            a_curr=fluid["a_k"],
            p_curr=fluid["p_k"],
            d_curr=fluid["d_mesh"],
            mesh_v_curr=mesh_v_curr,
            rho_f=float(setup.material.density),
            mu_f=float(setup.material.density * setup.material.kinematic_viscosity),
            dt=float(args.dt),
            dynamic_tau=1.0,
            backend="cpp",
        )

    current_load_lookup = CoordinateLookup(
        np.asarray(fluid_iface_coords_local, dtype=float),
        np.asarray(current_load_values, dtype=float),
        dim=2,
    )
    iqn_old_dr_mats, iqn_old_dg_mats = _rebuild_iqn_history_from_monitor(
        monitor_dir=monitor_dir,
        target_step=int(target_step),
        force_history=int(args.force_history),
        iteration_horizon=int(args.force_iteration_horizon),
        fluid_iface_coords_local=np.asarray(fluid_iface_coords_local, dtype=float),
    )
    payload = _checkpoint_payload(
        step=target_step,
        time_s=time_value,
        solid=solid,
        fluid=fluid,
        current_load_lookup=current_load_lookup,
        iqn_old_dr_mats=iqn_old_dr_mats,
        iqn_old_dg_mats=iqn_old_dg_mats,
    )

    checkpoint_dir = Path(args.output_dir).resolve() / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{target_step:04d}.npz"
    np.savez_compressed(checkpoint_path, **payload)
    (checkpoint_dir / "latest_checkpoint.txt").write_text(checkpoint_path.name, encoding="utf-8")
    print(f"checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
