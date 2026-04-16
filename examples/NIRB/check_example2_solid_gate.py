from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from examples.NIRB.common import dump_json
from examples.NIRB.example2_local_setup import load_example2_local_setup
from examples.NIRB.run_example2_local import (
    CoordinateLookup,
    _KratosLocalSolidSystemOperator,
    _ReducedResidualShiftOperator,
    _boundary_point_load_vector,
    _boundary_vector_snapshot,
    _build_solid_problem,
    _load_reference_partitioned_meshes,
    _maybe_build_kratos_local_solid_backend,
    _solid_residual_and_jacobian,
)
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, NewtonSolver, TimeStepperParameters


def _coord_key(x: float, y: float, ndigits: int = 12) -> tuple[float, float]:
    return (round(float(x), ndigits), round(float(y), ndigits))


class PointLookup:
    def __init__(self, coords: np.ndarray, values: np.ndarray):
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
        "abs_max": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "cosine": cosine,
        "reference_max_norm": float(np.max(np.linalg.norm(ref, axis=1))) if ref.size else 0.0,
        "local_max_norm": float(np.max(np.linalg.norm(loc, axis=1))) if loc.size else 0.0,
    }


def _zero_lookup(coords: np.ndarray) -> CoordinateLookup:
    return CoordinateLookup(np.asarray(coords, dtype=float), np.zeros((int(coords.shape[0]), 2), dtype=float), dim=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the local Example 2 solid block against the Kratos monitored structure step.")
    parser.add_argument("--output", type=Path, default=Path("examples/NIRB/artifacts/solid_gate_check.json"))
    parser.add_argument(
        "--kratos-monitor",
        type=Path,
        default=Path("examples/NIRB/artifacts/kratos_example2_monitor_20260407/coupling_monitor/step0001_iter0001_after_sync_output_structure.npz"),
    )
    parser.add_argument("--backend", choices=("python", "jit", "cpp"), default="cpp")
    parser.add_argument("--linear-backend", choices=("scipy", "petsc"), default="petsc")
    parser.add_argument("--quad-order", type=int, default=1)
    parser.add_argument("--newton-tol", type=float, default=1.0e-8)
    parser.add_argument("--max-newton-iter", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup = load_example2_local_setup()
    _, mesh_s = _load_reference_partitioned_meshes(setup=setup)
    solid = _build_solid_problem(mesh_s, poly_order=1)
    kratos_local_solid_backend = _maybe_build_kratos_local_solid_backend(
        benchmark_root=Path(setup.reference.root),
        prob=solid,
    )

    with np.load(Path(args.kratos_monitor).resolve()) as data:
        ref_load_coords = np.asarray(data["structure_load_coords_ref"], dtype=float)
        ref_load_values = np.asarray(data["structure_load_values"], dtype=float)
        ref_disp_coords = np.asarray(data["structure_disp_coords_ref"], dtype=float)
        ref_disp_values = np.asarray(data["structure_disp_values"], dtype=float)

    iface_coords, _ = _boundary_vector_snapshot(solid["dh"], solid["d_k"], setup.geometry.interface_tag)
    load_lookup = PointLookup(ref_load_coords, ref_load_values)
    local_load_values = load_lookup.sample(iface_coords)

    solid_res, solid_jac, solid_bcs, solid_bcs_homog = _solid_residual_and_jacobian(
        prob=solid,
        traction_lookup=_zero_lookup(iface_coords),
        mu_s=float(setup.material.shear_modulus),
        lambda_s=float(setup.material.lame_lambda),
        interface_tag=setup.geometry.interface_tag,
        clamp_tag=setup.geometry.clamp_tag,
        quad_order=int(args.quad_order),
    )

    solver = NewtonSolver(
        residual_form=solid_res,
        jacobian_form=solid_jac,
        dof_handler=solid["dh"],
        mixed_element=solid["me"],
        bcs=solid_bcs,
        bcs_homog=solid_bcs_homog,
        newton_params=NewtonParameters(
            newton_tol=float(args.newton_tol),
            max_newton_iter=int(args.max_newton_iter),
            line_search=False,
            globalization="none",
        ),
        lin_params=LinearSolverParameters(backend=str(args.linear_backend)),
        quad_order=int(args.quad_order),
        backend=str(args.backend),
    )

    point_load_full = _boundary_point_load_vector(
        solid["dh"],
        vector=solid["d_k"],
        tag=setup.geometry.interface_tag,
        values=local_load_values,
    )
    point_load_red = np.asarray(point_load_full[np.asarray(solver.active_dofs, dtype=int)], dtype=float)
    runtime_ops = []
    if kratos_local_solid_backend is not None:
        runtime_ops.append(
            _KratosLocalSolidSystemOperator(
                backend=kratos_local_solid_backend,
                d_k=solid["d_k"],
            )
        )
    runtime_ops.append(_ReducedResidualShiftOperator(point_load_red))
    solver.set_runtime_operators(runtime_ops)
    solver.solve_time_interval(
        functions=[solid["d_k"]],
        prev_functions=[solid["d_prev"]],
        time_params=TimeStepperParameters(
            dt=1.0,
            max_steps=1,
            final_time=1.0,
            stop_on_steady=False,
        ),
    )

    local_disp_coords, local_disp_values = _boundary_vector_snapshot(solid["dh"], solid["d_k"], setup.geometry.interface_tag)
    load_cmp = _compare_point_fields(
        ref_coords=ref_load_coords,
        ref_values=ref_load_values,
        local_coords=iface_coords,
        local_values=local_load_values,
    )
    disp_cmp = _compare_point_fields(
        ref_coords=ref_disp_coords,
        ref_values=ref_disp_values,
        local_coords=local_disp_coords,
        local_values=local_disp_values,
    )

    summary = {
        "kratos_monitor": str(Path(args.kratos_monitor).resolve()),
        "backend": str(args.backend),
        "linear_backend": str(args.linear_backend),
        "quad_order": int(args.quad_order),
        "structure_load_vs_local_applied": load_cmp,
        "structure_disp_vs_local_solve": disp_cmp,
        "local_interface_points": int(local_disp_coords.shape[0]),
        "reference_interface_points": int(ref_disp_coords.shape[0]),
        "criteria": {
            "load_rel_l2_le_1e-12": bool(float(load_cmp["rel_l2"]) <= 1.0e-12),
            "disp_rel_l2_le_1e-6": bool(float(disp_cmp["rel_l2"]) <= 1.0e-6),
            "disp_cosine_ge_0.999999": bool(float(disp_cmp["cosine"]) >= 0.999999),
        },
    }
    summary["criteria_all_pass"] = bool(all(summary["criteria"].values()))
    dump_json(summary, Path(args.output))
    print(f"solid gate: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
