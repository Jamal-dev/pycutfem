from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import examples.NIRB.run_example2_local as ex
from examples.NIRB.debug.build_local_checkpoint_from_step_history import _assign_scalar_function, _assign_vector_function
from examples.NIRB.dvms import (
    _advance_fluid_dvms_history_after_step,
    _report_fluid_dvms_subscale_after_step,
    _update_fluid_dvms_predicted_subscale,
    _update_fluid_dvms_state_from_previous_step,
)
from examples.NIRB.example2_local_setup import load_example2_local_setup


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


def _compare_fields(local_coords: np.ndarray, local_values: np.ndarray, ref_coords: np.ndarray, ref_values: np.ndarray) -> dict[str, float]:
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
        "reference_inf_norm": float(np.max(np.abs(ref))) if ref.size else 0.0,
        "local_inf_norm": float(np.max(np.abs(loc))) if loc.size else 0.0,
    }


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


def _load_step(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {str(key): np.asarray(data[key]) for key in data.files}


def _load_kratos_subscale_samples(step_data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if "fluid_q_coords_ref" in step_data and "fluid_subscale_velocity" in step_data:
        q_coords = np.asarray(step_data["fluid_q_coords_ref"], dtype=float).reshape(-1, 2)
        subscale = np.asarray(step_data["fluid_subscale_velocity"], dtype=float).reshape(-1, 2)
        return q_coords, subscale
    if "fluid_q_coords_ref_flat" in step_data and "fluid_subscale_velocity_flat" in step_data:
        q_coords = np.asarray(step_data["fluid_q_coords_ref_flat"], dtype=float).reshape(-1, 2)
        subscale = np.asarray(step_data["fluid_subscale_velocity_flat"], dtype=float).reshape(-1, 2)
        return q_coords, subscale
    raise KeyError(
        "The provided Kratos step-history file does not contain fluid quadrature subscale samples. "
        "Regenerate it with the enhanced run_kratos_example2_reference.py dumper."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the pycutfem DVMS accepted-step reported subscale against the Kratos "
            "accepted-step quadrature report written by OutputSolutionStep."
        )
    )
    parser.add_argument(
        "--kratos-step-dir",
        type=Path,
        default=Path("examples/NIRB/artifacts/kratos_step_history_coupled_qstate_5steps_20260415/step_history"),
    )
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--backend", choices=("python", "jit", "cpp"), default="cpp")
    parser.add_argument("--quad-order", type=int, default=1)
    parser.add_argument("--bossak-alpha", type=float, default=-0.3)
    parser.add_argument("--dynamic-tau", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/NIRB/artifacts/compare_kratos_accepted_subscale.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    step_dir = _resolve_step_history_dir(Path(args.kratos_step_dir))
    target_step = int(args.step)

    setup = load_example2_local_setup()
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
    payload = None

    for step in range(1, target_step + 1):
        data = _load_step(_step_file(step_dir, step))
        fluid_vel = np.asarray(data["fluid_velocity_nodal_values"], dtype=float)
        fluid_p = np.asarray(data["fluid_pressure_nodal_values"], dtype=float).reshape(-1, 1)
        mesh_disp = np.asarray(data["fluid_mesh_displacement_nodal_values"], dtype=float)
        if "fluid_acceleration_nodal_values" in data:
            fluid_acc = np.asarray(data["fluid_acceleration_nodal_values"], dtype=float)
        elif step == 1:
            fluid_acc = float(bossak["ma0"]) * fluid_vel
        else:
            fluid_acc = float(bossak["ma0"]) * (fluid_vel - prev_fluid_vel) + float(bossak["ma2"]) * prev_fluid_acc
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

        mesh_v_curr = ex.VectorFunction("mesh_v_curr_finalize", ["mx", "my"], dof_handler=fluid["dh"])
        _assign_vector_function(mesh_v_curr, source_coords=fluid_coords, point_values=mesh_vel)
        a_curr = ex.VectorFunction("a_curr_finalize", ["ux", "uy"], dof_handler=fluid["dh"])
        _assign_vector_function(a_curr, source_coords=fluid_coords, point_values=fluid_acc)
        _advance_fluid_dvms_history_after_step(
            fluid["dvms_state"],
            dh=fluid["dh"],
            mesh=mesh_f,
            u_curr=fluid["u_k"],
            a_curr=a_curr,
            p_curr=fluid["p_k"],
            d_curr=fluid["d_mesh"],
            mesh_v_curr=mesh_v_curr,
            rho_f=float(setup.material.density),
            mu_f=mu_f,
            dt=dt,
            dynamic_tau=float(args.dynamic_tau),
            backend=str(args.backend),
        )
        if step == target_step:
            kratos_q_coords, kratos_subscale = _load_kratos_subscale_samples(data)
            local_q_coords = np.asarray(fluid["dvms_state"].sample_coords, dtype=float)
            local_reported_subscale = _report_fluid_dvms_subscale_after_step(
                state=fluid["dvms_state"],
                dh=fluid["dh"],
                mesh=mesh_f,
                u_curr=fluid["u_k"],
                a_curr=a_curr,
                p_curr=fluid["p_k"],
                d_curr=fluid["d_mesh"],
                mesh_v_curr=mesh_v_curr,
                rho_f=float(setup.material.density),
                mu_f=mu_f,
                dt=dt,
                dynamic_tau=float(args.dynamic_tau),
                backend=str(args.backend),
            )
            payload = {
                "step": int(step),
                "backend": str(args.backend),
                "quadrature_order": int(args.quad_order),
                "kratos_q_count": int(kratos_q_coords.shape[0]),
                "local_q_count": int(local_q_coords.shape[0]),
                "reported_subscale_vs_kratos": _compare_fields(
                    local_coords=local_q_coords,
                    local_values=local_reported_subscale,
                    ref_coords=kratos_q_coords,
                    ref_values=kratos_subscale,
                ),
                "local_state_summary": fluid["dvms_state"].summary(),
                "note": (
                    "Kratos accepted-step step_history stores SUBSCALE_VELOCITY after "
                    "FinalizeSolutionStep(), so this compare mirrors the post-finalize "
                    "reported field rather than the hidden old_subscale history."
                ),
            }
        prev_prev_mesh_disp = np.asarray(prev_mesh_disp, dtype=float).copy()
        prev_mesh_disp = np.asarray(mesh_disp, dtype=float).copy()
        prev_mesh_vel = np.asarray(mesh_vel, dtype=float).copy()
        prev_mesh_acc = np.asarray(mesh_acc, dtype=float).copy()
        prev_fluid_vel = np.asarray(fluid_vel, dtype=float).copy()
        prev_fluid_acc = np.asarray(fluid_acc, dtype=float).copy()
        prev_pressure = np.asarray(fluid_p, dtype=float).copy()

    if payload is None:
        raise RuntimeError(f"Failed to build comparison payload for accepted step {target_step}.")
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
