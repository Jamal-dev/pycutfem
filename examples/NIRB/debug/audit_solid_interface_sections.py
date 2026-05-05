from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import examples.NIRB.run_example2_local as ex
from examples.NIRB.debug.build_local_checkpoint_from_step_history import _assign_vector_function, _lookup_sample
from examples.NIRB.debug.compare_example2_step_history import _compare_fields
from examples.NIRB.example2_local_setup import load_example2_local_setup
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, NewtonSolver, TimeStepperParameters


def _dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported JSON type: {type(value).__name__}")

    path.write_text(json.dumps(data, indent=2, default=default), encoding="utf-8")


def _load_step(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {str(key): np.asarray(data[key]) for key in data.files}


def _kratos_iqn_update(
    *,
    x_curr: np.ndarray,
    g_curr: np.ndarray,
    x_history: list[np.ndarray],
    g_history: list[np.ndarray],
    dr_old_mats: list[np.ndarray] | None,
    dg_old_mats: list[np.ndarray] | None,
    alpha: float,
    iteration_horizon: int,
) -> np.ndarray:
    x_curr_vec = np.asarray(x_curr, dtype=float).reshape(-1)
    g_curr_vec = np.asarray(g_curr, dtype=float).reshape(-1)
    r_curr = g_curr_vec - x_curr_vec

    # Kratos stores at most ``iteration_horizon`` residual/prediction snapshots.
    keep_count = min(max(int(iteration_horizon), 1), len(x_history), len(g_history))
    if keep_count <= 0:
        return (x_curr_vec + float(alpha) * r_curr).reshape(np.asarray(x_curr, dtype=float).shape)

    x_seq = [np.asarray(values, dtype=float).reshape(-1) for values in x_history[-keep_count:]]
    g_seq = [np.asarray(values, dtype=float).reshape(-1) for values in g_history[-keep_count:]]
    r_seq = [g_vec - x_vec for x_vec, g_vec in zip(x_seq, g_seq)]
    R = list(reversed(r_seq))
    X = list(reversed(g_seq))
    row = len(r_curr)
    col = len(R) - 1
    k = col

    V_old_blocks = [np.asarray(block, dtype=float) for block in (dr_old_mats or []) if np.asarray(block).size]
    W_old_blocks = [np.asarray(block, dtype=float) for block in (dg_old_mats or []) if np.asarray(block).size]
    has_old = bool(V_old_blocks and W_old_blocks)

    if not has_old:
        if k == 0:
            return (x_curr_vec + float(alpha) * r_curr).reshape(np.asarray(x_curr, dtype=float).shape)
        V_new = np.empty((col, row), dtype=float)
        W_new = np.empty((col, row), dtype=float)
        for i in range(col):
            V_new[i] = R[i] - R[i + 1]
            W_new[i] = X[i] - X[i + 1]
        V = V_new.T
        W = W_new.T
        delta_r = -R[0]
        c = np.linalg.lstsq(V, delta_r, rcond=None)[0]
        delta_x = W @ c - delta_r
        return (x_curr_vec + delta_x).reshape(np.asarray(x_curr, dtype=float).shape)

    V_old = np.hstack(V_old_blocks)
    W_old = np.hstack(W_old_blocks)
    if k == 0:
        delta_r = -R[0]
        c = np.linalg.lstsq(V_old, delta_r, rcond=None)[0]
        delta_x = W_old @ c - delta_r
        return (x_curr_vec + delta_x).reshape(np.asarray(x_curr, dtype=float).shape)

    V_new = np.empty((col, row), dtype=float)
    W_new = np.empty((col, row), dtype=float)
    for i in range(col):
        V_new[i] = R[i] - R[i + 1]
        W_new[i] = X[i] - X[i + 1]
    V = np.hstack((V_new.T, V_old))
    W = np.hstack((W_new.T, W_old))
    delta_r = -R[0]
    c = np.linalg.lstsq(V, delta_r, rcond=None)[0]
    delta_x = W @ c - delta_r
    return (x_curr_vec + delta_x).reshape(np.asarray(x_curr, dtype=float).shape)


def _isolated_solid_response_metrics(*, step1: dict[str, np.ndarray]) -> dict[str, float]:
    setup = load_example2_local_setup()
    _, mesh_s = ex._load_reference_partitioned_meshes(setup=setup)
    solid = ex._build_solid_problem(mesh_s, poly_order=1)
    kratos_local_solid_backend = ex._maybe_build_kratos_local_solid_backend(
        benchmark_root=Path(setup.reference.root),
        prob=solid,
    )

    solid_coords = np.asarray(step1["structure_coords_ref"], dtype=float)
    solid_disp = np.asarray(step1["structure_displacement_nodal_values"], dtype=float)
    solid_load = np.asarray(step1["structure_point_load_nodal_values"], dtype=float)

    _assign_vector_function(solid["d_prev"], source_coords=solid_coords, point_values=np.zeros_like(solid_disp))
    _assign_vector_function(solid["d_k"], source_coords=solid_coords, point_values=np.zeros_like(solid_disp))

    iface_coords, _ = ex._boundary_vector_snapshot(solid["dh"], solid["d_k"], setup.geometry.interface_tag)
    load_lookup = ex.CoordinateLookup(
        np.asarray(iface_coords, dtype=float),
        _lookup_sample(solid_coords, solid_load, iface_coords),
        dim=2,
    )
    solid_res, solid_jac, solid_bcs, solid_bcs_homog = ex._solid_residual_and_jacobian(
        prob=solid,
        traction_lookup=ex.CoordinateLookup(np.asarray(iface_coords, dtype=float), np.zeros((iface_coords.shape[0], 2), dtype=float), dim=2),
        mu_s=float(setup.material.shear_modulus),
        lambda_s=float(setup.material.lame_lambda),
        interface_tag=setup.geometry.interface_tag,
        clamp_tag=setup.geometry.clamp_tag,
        quad_order=2,
    )
    solver = NewtonSolver(
        residual_form=solid_res,
        jacobian_form=solid_jac,
        dof_handler=solid["dh"],
        mixed_element=solid["me"],
        bcs=solid_bcs,
        bcs_homog=solid_bcs_homog,
        newton_params=NewtonParameters(
            newton_tol=1.0e-10,
            max_newton_iter=20,
            line_search=False,
            globalization="none",
        ),
        lin_params=LinearSolverParameters(backend="petsc"),
        quad_order=2,
        backend="cpp",
    )
    point_load_full = ex._boundary_point_load_vector(
        solid["dh"],
        vector=solid["d_k"],
        tag=setup.geometry.interface_tag,
        values=np.asarray(load_lookup.values, dtype=float),
    )
    point_load_red = np.asarray(point_load_full[np.asarray(solver.active_dofs, dtype=int)], dtype=float)
    runtime_ops = []
    if kratos_local_solid_backend is not None:
        runtime_ops.append(
            ex._KratosLocalSolidSystemOperator(
                backend=kratos_local_solid_backend,
                d_k=solid["d_k"],
            )
        )
    runtime_ops.append(ex._ReducedResidualShiftOperator(point_load_red))
    solver.set_runtime_operators(runtime_ops)
    solver.solve_time_interval(
        functions=[solid["d_k"]],
        prev_functions=[solid["d_prev"]],
        time_params=TimeStepperParameters(dt=1.0, max_steps=1, final_time=1.0, stop_on_steady=False),
    )

    local_coords, local_values = ex._vector_field_matrix(solid["dh"], solid["d_k"])
    return _compare_fields(solid_coords, solid_disp, local_coords, local_values)


def _interface_snapshot_metrics(*, step1: dict[str, np.ndarray], step2: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    setup = load_example2_local_setup()
    mesh_f, mesh_s = ex._load_reference_partitioned_meshes(setup=setup)
    solid = ex._build_solid_problem(mesh_s, poly_order=1)

    solid_coords = np.asarray(step2["structure_coords_ref"], dtype=float)
    d_curr = np.asarray(step2["structure_displacement_nodal_values"], dtype=float)
    d_prev = np.asarray(step1["structure_displacement_nodal_values"], dtype=float)
    _assign_vector_function(solid["d_k"], source_coords=solid_coords, point_values=d_curr)
    _assign_vector_function(solid["d_prev"], source_coords=solid_coords, point_values=d_prev)

    iface_coords = np.asarray(step2["interface_disp_coords_ref"], dtype=float)
    iface_prev_coords = np.asarray(step1["interface_disp_coords_ref"], dtype=float)
    iface_prev_disp = np.asarray(step1["interface_disp_values"], dtype=float)
    iface_prev_vel, iface_prev_acc = ex._bossak_displacement_kinematics_values(
        d_curr=iface_prev_disp,
        d_prev=np.zeros_like(iface_prev_disp),
        v_prev=np.zeros_like(iface_prev_disp),
        a_prev=np.zeros_like(iface_prev_disp),
        dt=0.008,
        alpha=-0.3,
    )
    prev_vel_lookup = ex.CoordinateLookup(iface_prev_coords, iface_prev_vel, dim=2)
    prev_acc_lookup = ex.CoordinateLookup(iface_prev_coords, iface_prev_acc, dim=2)
    disp_lookup, _vel_lookup = ex._solid_interface_disp_velocity(
        dh=solid["dh"],
        mesh=mesh_s,
        d_curr=solid["d_k"],
        d_prev=solid["d_prev"],
        iface_coords=iface_coords,
        dt=0.008,
        v_prev_lookup=prev_vel_lookup,
        a_prev_lookup=prev_acc_lookup,
        bossak_alpha=-0.3,
    )
    return {
        "disp": _compare_fields(
            iface_coords,
            np.asarray(step2["interface_disp_values"], dtype=float),
            np.asarray(disp_lookup.coords, dtype=float),
            np.asarray(disp_lookup.values, dtype=float),
        ),
        "vel": _compare_fields(
            np.asarray(step2["interface_velocity_coords_ref"], dtype=float),
            np.asarray(step2["interface_velocity_values"], dtype=float),
            np.asarray(_vel_lookup.coords, dtype=float),
            np.asarray(_vel_lookup.values, dtype=float),
        ),
    }


def _iqn_parity_metrics() -> dict[str, float]:
    x_hist = [
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float),
        np.array([[0.6, 0.2], [0.1, 0.7]], dtype=float),
        np.array([[1.1, 0.4], [0.2, 1.1]], dtype=float),
        np.array([[1.4, 0.7], [0.4, 1.4]], dtype=float),
    ]
    g_hist = [
        np.array([[0.2, 0.1], [0.0, 0.3]], dtype=float),
        np.array([[0.9, 0.5], [0.3, 1.0]], dtype=float),
        np.array([[1.5, 0.9], [0.6, 1.5]], dtype=float),
        np.array([[1.8, 1.1], [0.9, 1.8]], dtype=float),
    ]
    dr_old = [np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5]], dtype=float)]
    dg_old = [2.0 * dr_old[0]]

    local = ex._iqnils_next_iterate(
        x_curr=x_hist[-1],
        g_curr=g_hist[-1],
        x_history=x_hist,
        g_history=g_hist,
        dr_old_mats=dr_old,
        dg_old_mats=dg_old,
        omega=0.5,
        horizon=3,
        regularization=0.0,
    )
    kratos = _kratos_iqn_update(
        x_curr=x_hist[-1],
        g_curr=g_hist[-1],
        x_history=x_hist,
        g_history=g_hist,
        dr_old_mats=dr_old,
        dg_old_mats=dg_old,
        alpha=0.5,
        iteration_horizon=3,
    )
    return _compare_fields(
        np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        np.asarray(kratos, dtype=float),
        np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        np.asarray(local, dtype=float),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Example 2 solid/interface sections against Kratos source behavior.")
    parser.add_argument(
        "--kratos-step-dir",
        type=Path,
        default=Path("examples/NIRB/artifacts/kratos_step_history_0140_0145/step_history"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("examples/NIRB/artifacts/solid_interface_section_audit_20260414.json"),
    )
    args = parser.parse_args()

    step_dir = Path(args.kratos_step_dir).resolve()
    step1 = _load_step(step_dir / "step0001.npz")
    step2 = _load_step(step_dir / "step0002.npz")
    payload = {
        "isolated_solid_response": _isolated_solid_response_metrics(step1=step1),
        "interface_snapshots": _interface_snapshot_metrics(step1=step1, step2=step2),
        "iqn_parity": _iqn_parity_metrics(),
    }
    _dump_json(payload, Path(args.output_json).resolve())
    print(Path(args.output_json).resolve())


if __name__ == "__main__":
    main()
