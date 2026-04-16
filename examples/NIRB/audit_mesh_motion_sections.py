from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

import examples.NIRB.run_example2_local as ex
from examples.NIRB.compare_example2_step_history import _compare_fields
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


def _isolated_structural_similarity_metrics(*, kratos_step: dict[str, np.ndarray]) -> dict[str, float]:
    setup = load_example2_local_setup()
    mesh_f, _ = ex._load_reference_partitioned_meshes(setup=setup)
    mesh_ext = ex._build_mesh_extension_problem(mesh_f, poly_order=1)
    iface_lookup = ex.CoordinateLookup(
        np.asarray(kratos_step["interface_disp_coords_ref"], dtype=float),
        np.asarray(kratos_step["interface_disp_values"], dtype=float),
        dim=2,
    )
    geometry = setup.geometry
    fixed_mesh_tags = (
        geometry.inlet_tag,
        geometry.outlet_tag,
        geometry.walls_tag,
        geometry.cylinder_tag,
    )
    mesh_eq, mesh_bcs = ex._mesh_extension_equation(
        prob=mesh_ext,
        interface_disp=iface_lookup,
        interface_tag=geometry.interface_tag,
        fixed_tags=fixed_mesh_tags,
        quad_order=6,
    )
    ex._solve_linear(
        eq=mesh_eq,
        dh=mesh_ext["dh"],
        bcs=mesh_bcs,
        quad_order=6,
        backend="cpp",
        linear_backend="petsc",
        functions=[mesh_ext["m_k"]],
    )
    local_coords, local_values = ex._vector_field_matrix(mesh_ext["dh"], mesh_ext["m_k"])
    return _compare_fields(
        np.asarray(kratos_step["fluid_coords_ref"], dtype=float),
        np.asarray(kratos_step["fluid_mesh_displacement_nodal_values"], dtype=float),
        local_coords,
        local_values,
    )


def _bossak_mesh_velocity_metrics(
    *,
    kratos_step1: dict[str, np.ndarray],
    kratos_step2: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    d1 = np.asarray(kratos_step1["fluid_mesh_displacement_nodal_values"], dtype=float)
    v1_ref = np.asarray(kratos_step1["fluid_mesh_velocity_nodal_values"], dtype=float)
    d2 = np.asarray(kratos_step2["fluid_mesh_displacement_nodal_values"], dtype=float)
    v2_ref = np.asarray(kratos_step2["fluid_mesh_velocity_nodal_values"], dtype=float)
    zero = np.zeros_like(d1)
    v1, a1 = ex._bossak_displacement_kinematics_values(
        d_curr=d1,
        d_prev=zero,
        v_prev=zero,
        a_prev=zero,
        dt=0.008,
        alpha=-0.3,
    )
    v2, _a2 = ex._bossak_displacement_kinematics_values(
        d_curr=d2,
        d_prev=d1,
        v_prev=v1,
        a_prev=a1,
        dt=0.008,
        alpha=-0.3,
    )
    return {
        "step1": _compare_fields(
            np.asarray(kratos_step1["fluid_coords_ref"], dtype=float),
            v1_ref,
            np.asarray(kratos_step1["fluid_coords_ref"], dtype=float),
            v1,
        ),
        "step2": _compare_fields(
            np.asarray(kratos_step2["fluid_coords_ref"], dtype=float),
            v2_ref,
            np.asarray(kratos_step2["fluid_coords_ref"], dtype=float),
            v2,
        ),
    }


def _ale_boundary_copy_metrics(*, kratos_step: dict[str, np.ndarray]) -> dict[str, float]:
    full_coords = np.asarray(kratos_step["fluid_coords_ref"], dtype=float)
    full_values = np.asarray(kratos_step["fluid_mesh_velocity_nodal_values"], dtype=float)
    iface_coords = np.asarray(kratos_step["interface_velocity_coords_ref"], dtype=float)
    iface_values = np.asarray(kratos_step["interface_velocity_values"], dtype=float)
    _, idx = cKDTree(full_coords).query(iface_coords)
    return _compare_fields(iface_coords, iface_values, iface_coords, full_values[idx])


def _local_checkpoint_promotion_metrics(
    *,
    checkpoint_path: Path,
    local_step_path: Path,
) -> dict[str, float]:
    with np.load(checkpoint_path) as chk, np.load(local_step_path) as step:
        setup = load_example2_local_setup()
        mesh_f, _ = ex._load_reference_partitioned_meshes(setup=setup)
        fluid = ex._build_fluid_problem(mesh_f, poly_order=1, pressure_order=1, quadrature_order=6)
        full_coords = np.asarray(step["fluid_coords_ref"], dtype=float)
        full_disp = np.asarray(step["fluid_mesh_displacement_nodal_values"], dtype=float)
        full_vel = np.asarray(step["fluid_mesh_velocity_nodal_values"], dtype=float)
        fluid["d_mesh"].nodal_values[:] = np.asarray(chk["fluid_d_mesh"], dtype=float)
        fluid["d_prev"].nodal_values[:] = np.asarray(chk["fluid_d_prev"], dtype=float)
        fluid["w_mesh_prev"].nodal_values[:] = np.asarray(chk["fluid_w_mesh_prev"], dtype=float)
        chk_disp = ex._vector_point_data_from_function(fluid["dh"], fluid["d_mesh"])
        chk_prev = ex._vector_point_data_from_function(fluid["dh"], fluid["d_prev"])
        chk_vel = ex._vector_point_data_from_function(fluid["dh"], fluid["w_mesh_prev"])
        return {
            "checkpoint_d_mesh": _compare_fields(full_coords, full_disp, full_coords, chk_disp),
            "checkpoint_d_prev": _compare_fields(full_coords, full_disp, full_coords, chk_prev),
            "checkpoint_w_mesh_prev": _compare_fields(full_coords, full_vel, full_coords, chk_vel),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Example 2 ALE mesh-motion sections against Kratos.")
    parser.add_argument(
        "--kratos-step-dir",
        type=Path,
        default=Path("examples/NIRB/artifacts/kratos_step_history_0140_0145/step_history"),
    )
    parser.add_argument(
        "--local-output-dir",
        type=Path,
        default=Path("examples/NIRB/artifacts/local_iqn_step_history_0001_0005_meshfix_20260413"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("examples/NIRB/artifacts/mesh_motion_section_audit_20260413.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kratos_step1 = dict(np.load(Path(args.kratos_step_dir) / "step0001.npz"))
    kratos_step2 = dict(np.load(Path(args.kratos_step_dir) / "step0002.npz"))
    local_step1_path = Path(args.local_output_dir) / "step_history" / "step0001.npz"
    checkpoint1_path = Path(args.local_output_dir) / "checkpoints" / "checkpoint_step_0001.npz"

    report = {
        "isolated_structural_similarity": _isolated_structural_similarity_metrics(kratos_step=kratos_step1),
        "bossak_mesh_velocity": _bossak_mesh_velocity_metrics(
            kratos_step1=kratos_step1,
            kratos_step2=kratos_step2,
        ),
        "ale_boundary_copy": _ale_boundary_copy_metrics(kratos_step=kratos_step1),
    }
    if checkpoint1_path.exists() and local_step1_path.exists():
        report["local_checkpoint_promotion"] = _local_checkpoint_promotion_metrics(
            checkpoint_path=checkpoint1_path,
            local_step_path=local_step1_path,
        )
    _dump_json(report, Path(args.output_json))
    print(f"audit_json: {Path(args.output_json).resolve()}")


if __name__ == "__main__":
    main()
