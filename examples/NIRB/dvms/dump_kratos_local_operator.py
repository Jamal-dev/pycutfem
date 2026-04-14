from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from examples.NIRB.double_flap_reference import _load_json, default_double_flap_root
from examples.NIRB.run_kratos_example2_reference import _copy_inputs, _prepare_fluid_json, _write_json


def _select_element(root_model_part, volume_model_part, *, element_id: int | None, node_id: int | None, interface_part_name: str):
    if element_id is not None:
        return volume_model_part.Elements[int(element_id)]
    if node_id is not None:
        node_value = int(node_id)
        for elem in volume_model_part.Elements:
            if any(int(node.Id) == node_value for node in elem.GetGeometry()):
                return elem

    interface_ids = {int(node.Id) for node in root_model_part.GetSubModelPart(interface_part_name).Nodes}
    best = None
    best_overlap = -1
    for elem in volume_model_part.Elements:
        overlap = sum(1 for node in elem.GetGeometry() if int(node.Id) in interface_ids)
        if overlap > best_overlap:
            best = elem
            best_overlap = overlap
    if best is None:
        raise RuntimeError("No fluid element found in Kratos model part.")
    return best


def _read_kratos_attr(obj, name: str):
    value = getattr(obj, name)
    return value() if callable(value) else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Kratos fluid-only step-1 case and dump one local fluid element operator/state."
    )
    parser.add_argument("--benchmark-root", type=Path, default=default_double_flap_root())
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("examples/NIRB/artifacts/kratos_fluid_local_operator"),
    )
    parser.add_argument("--end-time", type=float, default=0.008)
    parser.add_argument("--element-id", type=int, default=None, help="Kratos element id to dump. Default: pick an interface-adjacent element.")
    parser.add_argument("--node-id", type=int, default=None, help="Kratos node id to target. The first adjacent element is dumped.")
    parser.add_argument("--interface-part", type=str, default="NoSlip2D_Interface")
    parser.add_argument("--output-stem", type=str, default="fluid_local_element")
    parser.add_argument(
        "--mode",
        choices=("system", "velocity", "all"),
        default="all",
        help="Kratos local contribution to dump. 'all' stores both CalculateLocalVelocityContribution and CalculateLocalSystem.",
    )
    parser.add_argument(
        "--stage",
        choices=("predicted", "solved"),
        default="solved",
        help="Dump the local operator before SolveSolutionStep() ('predicted') or after the first solved step ('solved').",
    )
    return parser.parse_args()


def _dump_local_operator(
    *,
    KM,
    KFD,
    root_model_part,
    model_part,
    output_stem: str,
    run_dir: Path,
    element_id: int | None,
    node_id: int | None,
    interface_part_name: str,
    mode: str,
    stage: str,
) -> tuple[Path, Path]:
    elem = _select_element(
        root_model_part,
        model_part,
        element_id=element_id,
        node_id=node_id,
        interface_part_name=interface_part_name,
    )
    process_info = model_part.ProcessInfo

    lhs = KM.Matrix()
    rhs = KM.Vector()
    velocity_lhs = KM.Matrix()
    velocity_rhs = KM.Vector()
    system_lhs = KM.Matrix()
    system_rhs = KM.Vector()
    elem.CalculateLocalVelocityContribution(velocity_lhs, velocity_rhs, process_info)
    elem.CalculateLocalSystem(system_lhs, system_rhs, process_info)
    if str(mode) == "velocity":
        lhs, rhs = velocity_lhs, velocity_rhs
    elif str(mode) == "system":
        lhs, rhs = system_lhs, system_rhs
    else:
        lhs, rhs = velocity_lhs, velocity_rhs

    dofs = list(elem.GetDofList(process_info))
    dof_info: list[dict[str, object]] = []
    for dof in dofs:
        try:
            variable = dof.GetVariable()
            variable_name = variable.Name() if hasattr(variable, "Name") else str(variable)
        except Exception:
            variable_name = "<unknown>"
        try:
            value = float(dof.GetSolutionStepValue())
        except Exception:
            value = 0.0
        dof_info.append(
            {
                "id": int(_read_kratos_attr(dof, "Id")),
                "equation_id": int(_read_kratos_attr(dof, "EquationId")),
                "variable": str(variable_name),
                "value": float(value),
            }
        )

    extra_matrices: dict[str, np.ndarray] = {}
    for method_name in ("CalculateMassMatrix", "CalculateDampingMatrix"):
        if not hasattr(elem, method_name):
            continue
        try:
            matrix = KM.Matrix()
            getattr(elem, method_name)(matrix, process_info)
            extra_matrices[method_name] = np.asarray(matrix, dtype=float)
        except Exception:
            continue
    extra_vectors: dict[str, np.ndarray] = {}
    if hasattr(elem, "CalculateLumpedMassVector"):
        try:
            lumped = KM.Vector()
            elem.CalculateLumpedMassVector(lumped, process_info)
            extra_vectors["CalculateLumpedMassVector"] = np.asarray(lumped, dtype=float)
        except Exception:
            pass

    geom = elem.GetGeometry()
    node_ids = [int(node.Id) for node in geom]
    coords_ref = np.asarray([[float(node.X0), float(node.Y0)] for node in geom], dtype=float)
    coords_cur = np.asarray([[float(node.X), float(node.Y)] for node in geom], dtype=float)
    velocity = np.asarray(
        [[float(node.GetSolutionStepValue(KM.VELOCITY)[0]), float(node.GetSolutionStepValue(KM.VELOCITY)[1])] for node in geom],
        dtype=float,
    )
    velocity_prev = np.asarray(
        [[float(node.GetSolutionStepValue(KM.VELOCITY, 1)[0]), float(node.GetSolutionStepValue(KM.VELOCITY, 1)[1])] for node in geom],
        dtype=float,
    )
    acceleration = np.asarray(
        [[float(node.GetSolutionStepValue(KM.ACCELERATION)[0]), float(node.GetSolutionStepValue(KM.ACCELERATION)[1])] for node in geom],
        dtype=float,
    )
    mesh_velocity = np.asarray(
        [[float(node.GetSolutionStepValue(KM.MESH_VELOCITY)[0]), float(node.GetSolutionStepValue(KM.MESH_VELOCITY)[1])] for node in geom],
        dtype=float,
    )
    reaction = np.asarray(
        [[float(node.GetSolutionStepValue(KM.REACTION)[0]), float(node.GetSolutionStepValue(KM.REACTION)[1])] for node in geom],
        dtype=float,
    )
    pressure = np.asarray([float(node.GetSolutionStepValue(KM.PRESSURE)) for node in geom], dtype=float)
    advproj = np.asarray(
        [[float(node.GetSolutionStepValue(KM.ADVPROJ)[0]), float(node.GetSolutionStepValue(KM.ADVPROJ)[1])] for node in geom],
        dtype=float,
    )
    divproj = np.asarray([float(node.GetSolutionStepValue(KM.DIVPROJ)) for node in geom], dtype=float)
    subscale_velocity = np.asarray(
        elem.CalculateOnIntegrationPoints(KFD.SUBSCALE_VELOCITY, process_info),
        dtype=float,
    )
    subscale_pressure = np.asarray(
        elem.CalculateOnIntegrationPoints(KFD.SUBSCALE_PRESSURE, process_info),
        dtype=float,
    )

    npz_path = run_dir / f"{output_stem}.npz"
    np.savez(
        npz_path,
        stage=str(stage),
        element_id=int(elem.Id),
        node_ids=np.asarray(node_ids, dtype=int),
        coords_ref=coords_ref,
        coords_cur=coords_cur,
        velocity=velocity,
        velocity_prev=velocity_prev,
        acceleration=acceleration,
        mesh_velocity=mesh_velocity,
        reaction=reaction,
        pressure=pressure,
        advproj=advproj,
        divproj=divproj,
        subscale_velocity=subscale_velocity,
        subscale_pressure=subscale_pressure,
        lhs=np.asarray(lhs, dtype=float),
        rhs=np.asarray(rhs, dtype=float),
        velocity_lhs=np.asarray(velocity_lhs, dtype=float),
        velocity_rhs=np.asarray(velocity_rhs, dtype=float),
        system_lhs=np.asarray(system_lhs, dtype=float),
        system_rhs=np.asarray(system_rhs, dtype=float),
        dof_equation_ids=np.asarray([int(item["equation_id"]) for item in dof_info], dtype=int),
    )
    if extra_matrices:
        np.savez(
            run_dir / f"{output_stem}_extra_matrices.npz",
            **extra_matrices,
            **extra_vectors,
        )

    json_path = run_dir / f"{output_stem}.json"
    json_path.write_text(
        json.dumps(
            {
                "stage": str(stage),
                "element_id": int(elem.Id),
                "node_ids": node_ids,
                "coords_ref": coords_ref.tolist(),
                "coords_cur": coords_cur.tolist(),
                "velocity": velocity.tolist(),
                "velocity_prev": velocity_prev.tolist(),
                "acceleration": acceleration.tolist(),
                "mesh_velocity": mesh_velocity.tolist(),
                "reaction": reaction.tolist(),
                "pressure": pressure.tolist(),
                "advproj": advproj.tolist(),
                "divproj": divproj.tolist(),
                "subscale_velocity": subscale_velocity.tolist(),
                "subscale_pressure": subscale_pressure.tolist(),
                "mode": str(mode),
                "velocity_lhs_shape": [int(velocity_lhs.Size1()), int(velocity_lhs.Size2())],
                "velocity_rhs_size": int(velocity_rhs.Size()),
                "velocity_lhs_inf_norm": float(np.max(np.abs(np.asarray(velocity_lhs, dtype=float)))),
                "velocity_rhs_inf_norm": float(np.max(np.abs(np.asarray(velocity_rhs, dtype=float)))) if velocity_rhs.Size() else 0.0,
                "system_lhs_shape": [int(system_lhs.Size1()), int(system_lhs.Size2())],
                "system_rhs_size": int(system_rhs.Size()),
                "system_lhs_inf_norm": float(np.max(np.abs(np.asarray(system_lhs, dtype=float)))),
                "system_rhs_inf_norm": float(np.max(np.abs(np.asarray(system_rhs, dtype=float)))) if system_rhs.Size() else 0.0,
                "dofs": dof_info,
                "lhs_shape": [int(lhs.Size1()), int(lhs.Size2())],
                "rhs_size": int(rhs.Size()),
                "lhs_inf_norm": float(np.max(np.abs(np.asarray(lhs, dtype=float)))),
                "rhs_inf_norm": float(np.max(np.abs(np.asarray(rhs, dtype=float)))) if rhs.Size() else 0.0,
                "extra_matrices": {
                    key: {
                        "shape": list(np.asarray(value, dtype=float).shape),
                        "inf_norm": float(np.max(np.abs(np.asarray(value, dtype=float)))) if np.asarray(value).size else 0.0,
                    }
                    for key, value in extra_matrices.items()
                },
                "extra_vectors": {
                    key: {
                        "shape": list(np.asarray(value, dtype=float).shape),
                        "inf_norm": float(np.max(np.abs(np.asarray(value, dtype=float)))) if np.asarray(value).size else 0.0,
                    }
                    for key, value in extra_vectors.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return npz_path, json_path


def main() -> None:
    args = parse_args()

    try:
        import KratosMultiphysics as KM
        import KratosMultiphysics.FluidDynamicsApplication as KFD
        from KratosMultiphysics.FluidDynamicsApplication.fluid_dynamics_analysis import FluidDynamicsAnalysis
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "KratosMultiphysics is not importable in the current Python environment."
        ) from exc

    benchmark_root = Path(args.benchmark_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists():
        shutil.rmtree(run_dir)
    _copy_inputs(benchmark_root, run_dir)

    fluid_json = _prepare_fluid_json(
        _load_json(benchmark_root / "ProjectParametersCFD.json"),
        end_time=float(args.end_time),
        echo_level=0,
        output_path="vtk_output_fsi_cfd",
    )
    _write_json(run_dir / "ProjectParametersCFD.json", fluid_json)

    cwd = Path.cwd()
    try:
        os.chdir(run_dir)
        with (run_dir / "ProjectParametersCFD.json").open("r", encoding="utf-8") as f:
            params = KM.Parameters(f.read())
        model = KM.Model()
        analysis = FluidDynamicsAnalysis(model, params)
        analysis.Initialize()
        solver = analysis._GetSolver()
        current_time = float(getattr(analysis, "time", 0.0))
        new_time = float(solver.AdvanceInTime(current_time))
        if hasattr(analysis, "time"):
            analysis.time = new_time
        analysis.InitializeSolutionStep()
        solver.Predict()

        if str(args.stage) == "solved":
            solver.SolveSolutionStep()

        root_model_part = model["FluidModelPart"]
        model_part = model["FluidModelPart.FluidParts_FluidPart"]
        npz_path, json_path = _dump_local_operator(
            KM=KM,
            KFD=KFD,
            root_model_part=root_model_part,
            model_part=model_part,
            output_stem=str(args.output_stem),
            run_dir=run_dir,
            element_id=args.element_id,
            node_id=args.node_id,
            interface_part_name=str(args.interface_part),
            mode=str(args.mode),
            stage=str(args.stage),
        )
        analysis.FinalizeSolutionStep()
        analysis.OutputSolutionStep()
        analysis.Finalize()
        print(npz_path)
        print(json_path)
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
