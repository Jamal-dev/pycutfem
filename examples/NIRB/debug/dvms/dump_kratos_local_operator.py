from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from examples.NIRB.double_flap_reference import _load_json, default_double_flap_root
from examples.NIRB.debug.run_kratos_example2_reference import _copy_inputs, _prepare_fluid_json, _write_json


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
    parser.add_argument(
        "--state-npz",
        type=Path,
        default=None,
        help=(
            "Optional saved fluid state npz. When provided, the model is initialized and "
            "the nodal/current geometry state is injected from this file instead of rerunning "
            "the nonlinear solve."
        ),
    )
    return parser.parse_args()


def _safe_np_array(state: dict[str, np.ndarray], key: str, *, default=None):
    value = state.get(str(key), default)
    if value is None:
        return None
    return np.asarray(value)


def _safe_np_array_first(state: dict[str, np.ndarray], *keys: str, default=None):
    for key in keys:
        value = _safe_np_array(state, str(key), default=None)
        if value is not None:
            return value
    return None if default is None else np.asarray(default)


def _load_state_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path).resolve(), allow_pickle=False) as data:
        payload: dict[str, np.ndarray] = {}
        for key in data.files:
            payload[str(key)] = np.asarray(data[key])
    return payload


def _set_step_scalar(node, variable, value: float, *, step: int = 0) -> None:
    try:
        node.SetSolutionStepValue(variable, int(step), float(value))
    except Exception:
        node.SetValue(variable, float(value))


def _set_step_vec2(node, var_x, var_y, values, *, step: int = 0) -> None:
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    x_val = float(values_arr[0]) if values_arr.size >= 1 else 0.0
    y_val = float(values_arr[1]) if values_arr.size >= 2 else 0.0
    _set_step_scalar(node, var_x, x_val, step=int(step))
    _set_step_scalar(node, var_y, y_val, step=int(step))


def _array_inf_norm(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.max(np.abs(arr))) if arr.size else 0.0


def _apply_state_npz_to_model(*, KM, model_part, state: dict[str, np.ndarray], dt: float) -> None:
    node_ids = _safe_np_array(state, "node_ids")
    if node_ids is None:
        raise KeyError("State npz does not contain 'node_ids'.")
    coords_ref = _safe_np_array_first(state, "node_coords_ref", "coords_ref")
    coords_cur = _safe_np_array_first(state, "node_coords_cur", "coords_cur", default=coords_ref)
    velocity = _safe_np_array_first(state, "velocity", "velocity_nodal_values")
    velocity_prev = _safe_np_array_first(state, "velocity_prev", "velocity_prev_nodal_values")
    acceleration = _safe_np_array_first(state, "acceleration", "acceleration_nodal_values")
    acceleration_prev = _safe_np_array_first(
        state,
        "acceleration_prev",
        "acceleration_prev_nodal_values",
        default=np.zeros_like(acceleration, dtype=float) if acceleration is not None else None,
    )
    mesh_velocity = _safe_np_array_first(state, "mesh_velocity", "mesh_velocity_nodal_values")
    pressure = _safe_np_array_first(state, "pressure", "pressure_nodal_values")
    reaction = _safe_np_array_first(state, "reaction", "reaction_nodal_values")
    advproj = _safe_np_array_first(state, "advproj", "advproj_nodal_values")
    divproj = _safe_np_array_first(state, "divproj", "divproj_nodal_values")
    mesh_displacement = _safe_np_array_first(state, "mesh_displacement", "mesh_displacement_nodal_values")
    if mesh_displacement is None and coords_ref is not None and coords_cur is not None:
        mesh_displacement = np.asarray(coords_cur, dtype=float) - np.asarray(coords_ref, dtype=float)
    mesh_displacement_prev = _safe_np_array_first(
        state,
        "mesh_displacement_prev",
        "mesh_displacement_prev_nodal_values",
    )
    if mesh_displacement_prev is None and mesh_displacement is not None and mesh_velocity is not None:
        mesh_displacement_prev = np.asarray(mesh_displacement, dtype=float) - float(dt) * np.asarray(mesh_velocity, dtype=float)

    index_by_node = {int(nid): idx for idx, nid in enumerate(np.asarray(node_ids, dtype=int).reshape(-1).tolist())}
    for node in model_part.Nodes:
        node_id = int(node.Id)
        idx = index_by_node.get(node_id)
        if idx is None:
            continue
        if coords_ref is not None:
            node.X0 = float(coords_ref[idx, 0])
            node.Y0 = float(coords_ref[idx, 1])
        if coords_cur is not None:
            node.X = float(coords_cur[idx, 0])
            node.Y = float(coords_cur[idx, 1])
        if velocity is not None:
            _set_step_vec2(node, KM.VELOCITY_X, KM.VELOCITY_Y, velocity[idx], step=0)
        if velocity_prev is not None:
            _set_step_vec2(node, KM.VELOCITY_X, KM.VELOCITY_Y, velocity_prev[idx], step=1)
        if acceleration is not None:
            _set_step_vec2(node, KM.ACCELERATION_X, KM.ACCELERATION_Y, acceleration[idx], step=0)
        if acceleration_prev is not None:
            _set_step_vec2(node, KM.ACCELERATION_X, KM.ACCELERATION_Y, acceleration_prev[idx], step=1)
        if mesh_velocity is not None:
            _set_step_vec2(node, KM.MESH_VELOCITY_X, KM.MESH_VELOCITY_Y, mesh_velocity[idx], step=0)
            mesh_velocity_prev = _safe_np_array_first(
                state,
                "mesh_velocity_prev",
                "mesh_velocity_prev_nodal_values",
                default=mesh_velocity,
            )
            _set_step_vec2(node, KM.MESH_VELOCITY_X, KM.MESH_VELOCITY_Y, mesh_velocity_prev[idx], step=1)
        if mesh_displacement is not None:
            _set_step_vec2(node, KM.MESH_DISPLACEMENT_X, KM.MESH_DISPLACEMENT_Y, mesh_displacement[idx], step=0)
        if mesh_displacement_prev is not None:
            _set_step_vec2(node, KM.MESH_DISPLACEMENT_X, KM.MESH_DISPLACEMENT_Y, mesh_displacement_prev[idx], step=1)
        if pressure is not None:
            _set_step_scalar(node, KM.PRESSURE, float(pressure[idx]), step=0)
        if reaction is not None:
            _set_step_vec2(node, KM.REACTION_X, KM.REACTION_Y, reaction[idx], step=0)
        if advproj is not None:
            _set_step_vec2(node, KM.ADVPROJ_X, KM.ADVPROJ_Y, advproj[idx], step=0)
        if divproj is not None:
            _set_step_scalar(node, KM.DIVPROJ, float(divproj[idx]), step=0)


def _dump_local_operator(
    *,
    KM,
    KFD,
    root_model_part,
    model_part,
    scheme,
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
    scheme_lhs = KM.Matrix()
    scheme_rhs = KM.Vector()
    scheme_eq_ids = KM.Vector()
    elem.CalculateLocalVelocityContribution(velocity_lhs, velocity_rhs, process_info)
    elem.CalculateLocalSystem(system_lhs, system_rhs, process_info)
    scheme_contrib_available = hasattr(scheme, "CalculateSystemContributions")
    if bool(scheme_contrib_available):
        scheme.CalculateSystemContributions(elem, scheme_lhs, scheme_rhs, scheme_eq_ids, process_info)
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
        scheme_lhs=np.asarray(scheme_lhs, dtype=float),
        scheme_rhs=np.asarray(scheme_rhs, dtype=float),
        scheme_equation_ids=np.asarray(scheme_eq_ids, dtype=int),
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
                "velocity_lhs_inf_norm": _array_inf_norm(velocity_lhs),
                "velocity_rhs_inf_norm": float(np.max(np.abs(np.asarray(velocity_rhs, dtype=float)))) if velocity_rhs.Size() else 0.0,
                "system_lhs_shape": [int(system_lhs.Size1()), int(system_lhs.Size2())],
                "system_rhs_size": int(system_rhs.Size()),
                "system_lhs_inf_norm": _array_inf_norm(system_lhs),
                "system_rhs_inf_norm": float(np.max(np.abs(np.asarray(system_rhs, dtype=float)))) if system_rhs.Size() else 0.0,
                "scheme_lhs_shape": [int(scheme_lhs.Size1()), int(scheme_lhs.Size2())],
                "scheme_rhs_size": int(scheme_rhs.Size()),
                "scheme_lhs_inf_norm": _array_inf_norm(scheme_lhs),
                "scheme_rhs_inf_norm": float(np.max(np.abs(np.asarray(scheme_rhs, dtype=float)))) if scheme_rhs.Size() else 0.0,
                "scheme_contrib_available": bool(scheme_contrib_available),
                "scheme_equation_ids": np.asarray(scheme_eq_ids, dtype=int).tolist(),
                "dofs": dof_info,
                "lhs_shape": [int(lhs.Size1()), int(lhs.Size2())],
                "rhs_size": int(rhs.Size()),
                "lhs_inf_norm": _array_inf_norm(lhs),
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
    if args.state_npz is not None:
        args.state_npz = Path(args.state_npz).resolve()

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
        if args.state_npz is not None:
            root_process_info = model["FluidModelPart"].ProcessInfo
            dt_value = float(root_process_info[KM.DELTA_TIME]) if root_process_info.Has(KM.DELTA_TIME) else float(args.end_time)
            _apply_state_npz_to_model(
                KM=KM,
                model_part=model["FluidModelPart.FluidParts_FluidPart"],
                state=_load_state_npz(Path(args.state_npz)),
                dt=float(dt_value),
            )
        elif str(args.stage) == "solved":
            solver.SolveSolutionStep()

        root_model_part = model["FluidModelPart"]
        model_part = model["FluidModelPart.FluidParts_FluidPart"]
        scheme = solver.fluid_solver._GetScheme()
        npz_path, json_path = _dump_local_operator(
            KM=KM,
            KFD=KFD,
            root_model_part=root_model_part,
            model_part=model_part,
            scheme=scheme,
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
