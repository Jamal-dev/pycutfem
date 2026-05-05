from __future__ import annotations

import argparse
import contextlib
import json
import shutil
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np

from examples.NIRB.double_flap_reference import _load_json, default_double_flap_root


def _copy_inputs(source_root: Path, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for rel_path in (
        Path("Double_Flap_Mesh"),
        Path("FluidMaterials.json"),
        Path("StructuralMaterials.json"),
    ):
        src = source_root / rel_path
        dst = run_dir / rel_path
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _prepare_coupling_json(data: dict[str, Any], *, end_time: float, echo_level: int) -> dict[str, Any]:
    problem_data = dict(data["problem_data"])
    problem_data["end_time"] = float(end_time)

    solver_settings = dict(data["solver_settings"])
    solver_settings["echo_level"] = int(echo_level)
    solver_settings["num_coupling_iterations"] = int(solver_settings.get("num_coupling_iterations", 50))
    solver_settings.pop("initial_guess", None)
    solver_settings.pop("initial_guess_launch_time", None)
    solver_settings.pop("save_tr_data", None)
    solver_settings.pop("training_launch_time", None)
    solver_settings.pop("training_end_time", None)

    accelerators = []
    for accelerator in solver_settings.get("convergence_accelerators", []):
        clean = dict(accelerator)
        if clean.get("type") == "iqnilsM":
            clean["type"] = "iqnils"
        clean.pop("save_tr_data", None)
        clean.pop("training_launch_time", None)
        clean.pop("training_end_time", None)
        clean.pop("prediction_launch_time", None)
        clean.pop("prediction_end_time", None)
        clean.pop("orthogonal_w", None)
        accelerators.append(clean)
    solver_settings["convergence_accelerators"] = accelerators

    solvers = dict(solver_settings["solvers"])
    structure = dict(solvers["structure"])
    structure["type"] = "solver_wrappers.kratos.structural_mechanics_wrapper"
    for key in (
        "launch_time",
        "start_collecting_time",
        "input_data",
        "output_data",
        "interface_only",
        "imported_model",
        "save_model",
        "save_training_data",
        "use_map",
        "file",
    ):
        structure.pop(key, None)
    solvers["structure"] = structure
    solver_settings["solvers"] = solvers

    return {
        "problem_data": problem_data,
        "solver_settings": solver_settings,
    }


def _prepare_fluid_json(data: dict[str, Any], *, end_time: float, echo_level: int, output_path: str) -> dict[str, Any]:
    prepared = json.loads(json.dumps(data))
    prepared["problem_data"]["end_time"] = float(end_time)
    prepared["solver_settings"]["fluid_solver_settings"]["echo_level"] = int(echo_level)
    prepared["output_processes"]["vtk_output"][0]["Parameters"]["output_path"] = str(output_path)
    return prepared


def _prepare_solid_json(data: dict[str, Any], *, end_time: float, echo_level: int, output_path: str) -> dict[str, Any]:
    prepared = json.loads(json.dumps(data))
    prepared["problem_data"]["end_time"] = float(end_time)
    prepared["solver_settings"]["echo_level"] = int(echo_level)
    prepared["output_processes"]["vtk_output"][0]["Parameters"]["output_path"] = str(output_path)
    return prepared


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _reshape_interface_values(interface_data) -> np.ndarray:
    raw = np.asarray(interface_data.GetData(), dtype=float)
    dim = int(getattr(interface_data, "dimension", 1))
    if dim <= 1:
        return raw.reshape(-1, 1)
    return raw.reshape(-1, dim)


def _interface_node_arrays(interface_data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = list(interface_data.GetModelPart().GetCommunicator().LocalMesh().Nodes)
    ids = np.asarray([int(node.Id) for node in nodes], dtype=int)
    coords_ref = np.asarray([[float(node.X0), float(node.Y0)] for node in nodes], dtype=float)
    coords_cur = np.asarray([[float(node.X), float(node.Y)] for node in nodes], dtype=float)
    return ids, coords_ref, coords_cur


def _try_get_solver_model_part(solver_wrapper, solver_name: str) -> Any | None:
    candidates: list[Any] = []
    for attr in ("_analysis_stage", "analysis_stage"):
        obj = getattr(solver_wrapper, attr, None)
        if obj is not None:
            candidates.append(obj)
    for obj in list(candidates):
        for getter in ("_GetSolver", "GetSolver"):
            fn = getattr(obj, getter, None)
            if callable(fn):
                try:
                    candidates.append(fn())
                except Exception:
                    pass
    for obj in list(candidates):
        for getter in ("GetComputingModelPart", "GetFluidComputingModelPart", "GetStructureComputingModelPart"):
            fn = getattr(obj, getter, None)
            if callable(fn):
                try:
                    mp = fn()
                    if mp is not None:
                        return mp
                except Exception:
                    pass
        for attr in ("main_model_part", "computing_model_part"):
            mp = getattr(obj, attr, None)
            if mp is not None:
                return mp
    if str(solver_name) == "fluid":
        for obj in candidates:
            mp = getattr(obj, "fluid_solver", None)
            if mp is not None and hasattr(mp, "GetComputingModelPart"):
                try:
                    return mp.GetComputingModelPart()
                except Exception:
                    pass
    return None


def _process_info_step_time(KM, process_info) -> tuple[int | None, float | None]:
    if process_info is None:
        return None, None

    step_value: int | None = None
    time_value: float | None = None
    dt_value: float | None = None
    start_time = 0.0

    try:
        if process_info.Has(KM.STEP):
            step_raw = int(process_info[KM.STEP])
            if step_raw > 0:
                step_value = step_raw
    except Exception:
        step_value = None

    try:
        if process_info.Has(KM.TIME):
            time_value = float(process_info[KM.TIME])
    except Exception:
        time_value = None

    try:
        if process_info.Has(KM.DELTA_TIME):
            dt_raw = float(process_info[KM.DELTA_TIME])
            if abs(dt_raw) > 1.0e-15:
                dt_value = dt_raw
    except Exception:
        dt_value = None

    start_time_var = getattr(KM, "START_TIME", None)
    if start_time_var is not None:
        try:
            if process_info.Has(start_time_var):
                start_time = float(process_info[start_time_var])
        except Exception:
            start_time = 0.0

    if time_value is not None and dt_value is not None:
        step_from_time = int(round((float(time_value) - float(start_time)) / float(dt_value)))
        if step_from_time > 0 and (step_value is None or step_value <= 1 < step_from_time or step_value != step_from_time):
            step_value = step_from_time

    return step_value, time_value


def _solver_wrapper_step_time(KM, solver_wrapper, solver_name: str) -> tuple[int | None, float | None]:
    model_part = _try_get_solver_model_part(solver_wrapper, solver_name)
    if model_part is not None:
        step_value, time_value = _process_info_step_time(KM, getattr(model_part, "ProcessInfo", None))
        if step_value is not None or time_value is not None:
            return step_value, time_value

    for attr in ("_analysis_stage", "analysis_stage"):
        stage = getattr(solver_wrapper, attr, None)
        if stage is None:
            continue
        step_value = getattr(stage, "step", None)
        time_value = getattr(stage, "time", None)
        if step_value is not None or time_value is not None:
            step_out = None if step_value is None else int(step_value)
            time_out = None if time_value is None else float(time_value)
            return step_out, time_out

    return None, None


def _safe_scalar_step_values(nodes, variable, step: int = 0) -> np.ndarray:
    values = np.zeros((len(nodes), 1), dtype=float)
    if variable is None:
        return values
    for i, node in enumerate(nodes):
        try:
            values[i, 0] = float(node.GetSolutionStepValue(variable, int(step)))
        except Exception:
            values[i, 0] = 0.0
    return values


def _safe_vector_step_values(nodes, variable, step: int = 0) -> np.ndarray:
    values = np.zeros((len(nodes), 2), dtype=float)
    if variable is None:
        return values
    for i, node in enumerate(nodes):
        try:
            raw = node.GetSolutionStepValue(variable, int(step))
            arr = np.asarray(raw, dtype=float).reshape(-1)
            if arr.size >= 2:
                values[i, :] = arr[:2]
            elif arr.size == 1:
                values[i, 0] = arr[0]
        except Exception:
            values[i, :] = 0.0
    return values


def _safe_condition_vector_values(conditions, variable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.zeros((len(conditions),), dtype=int)
    node_ids = np.zeros((len(conditions), 1), dtype=int)
    values = np.zeros((len(conditions), 2), dtype=float)
    if variable is None:
        return ids, node_ids, values
    for i, condition in enumerate(conditions):
        ids[i] = int(condition.Id)
        try:
            geom = condition.GetGeometry()
            if len(geom) >= 1:
                node_ids[i, 0] = int(geom[0].Id)
        except Exception:
            node_ids[i, 0] = -1
        try:
            raw = condition.GetValue(variable)
            arr = np.asarray(raw, dtype=float).reshape(-1)
            if arr.size >= 2:
                values[i, :] = arr[:2]
            elif arr.size == 1:
                values[i, 0] = arr[0]
        except Exception:
            values[i, :] = 0.0
    return ids, node_ids, values


def _model_part_state_payload(model_part, *, solver_name: str) -> dict[str, np.ndarray]:
    import KratosMultiphysics as KM

    try:
        import KratosMultiphysics.StructuralMechanicsApplication as KSM  # type: ignore
    except Exception:  # pragma: no cover - depends on env
        KSM = None

    nodes = list(model_part.Nodes)
    ids = np.asarray([int(node.Id) for node in nodes], dtype=int)
    coords_ref = np.asarray([[float(node.X0), float(node.Y0)] for node in nodes], dtype=float)
    coords_cur = np.asarray([[float(node.X), float(node.Y)] for node in nodes], dtype=float)
    payload: dict[str, np.ndarray] = {
        f"{solver_name}_node_ids": ids,
        f"{solver_name}_coords_ref": coords_ref,
        f"{solver_name}_coords_cur": coords_cur,
    }

    if str(solver_name) == "fluid":
        import KratosMultiphysics.FluidDynamicsApplication as KFD  # type: ignore

        from examples.NIRB.debug.dvms.dump_kratos_fluid_step1_state import _integration_point_coords, _resolve_integration_method

        payload[f"{solver_name}_velocity_nodal_values"] = _safe_vector_step_values(nodes, KM.VELOCITY)
        payload[f"{solver_name}_velocity_prev_nodal_values"] = _safe_vector_step_values(nodes, KM.VELOCITY, step=1)
        payload[f"{solver_name}_acceleration_nodal_values"] = _safe_vector_step_values(nodes, KM.ACCELERATION)
        payload[f"{solver_name}_acceleration_prev_nodal_values"] = _safe_vector_step_values(
            nodes,
            KM.ACCELERATION,
            step=1,
        )
        payload[f"{solver_name}_pressure_nodal_values"] = _safe_scalar_step_values(nodes, KM.PRESSURE)
        payload[f"{solver_name}_pressure_prev_nodal_values"] = _safe_scalar_step_values(
            nodes,
            KM.PRESSURE,
            step=1,
        )
        payload[f"{solver_name}_reaction_nodal_values"] = _safe_vector_step_values(nodes, KM.REACTION)
        payload[f"{solver_name}_mesh_displacement_nodal_values"] = _safe_vector_step_values(nodes, KM.MESH_DISPLACEMENT)
        payload[f"{solver_name}_mesh_displacement_prev_nodal_values"] = _safe_vector_step_values(
            nodes,
            KM.MESH_DISPLACEMENT,
            step=1,
        )
        payload[f"{solver_name}_mesh_velocity_nodal_values"] = _safe_vector_step_values(nodes, KM.MESH_VELOCITY)
        payload[f"{solver_name}_mesh_velocity_prev_nodal_values"] = _safe_vector_step_values(
            nodes,
            KM.MESH_VELOCITY,
            step=1,
        )
        payload[f"{solver_name}_advproj_nodal_values"] = _safe_vector_step_values(nodes, KM.ADVPROJ)
        payload[f"{solver_name}_divproj_nodal_values"] = _safe_scalar_step_values(nodes, KM.DIVPROJ)

        elements = list(model_part.Elements)
        n_elem = len(elements)
        node_ids_by_elem = np.zeros((n_elem, 3), dtype=int)
        q_coords_ref_list: list[np.ndarray] = []
        q_coords_cur_list: list[np.ndarray] = []
        q_weights_list: list[np.ndarray] = []
        subscale_velocity_list: list[np.ndarray] = []
        subscale_pressure_list: list[np.ndarray] = []
        for eidx, elem in enumerate(elements):
            geom = elem.GetGeometry()
            node_ids_by_elem[eidx, :] = np.asarray([int(node.Id) for node in geom], dtype=int)
            elem_subscale_velocity = np.asarray(
                elem.CalculateOnIntegrationPoints(KFD.SUBSCALE_VELOCITY, model_part.ProcessInfo),
                dtype=float,
            )[:, :2]
            elem_subscale_pressure = np.asarray(
                elem.CalculateOnIntegrationPoints(KFD.SUBSCALE_PRESSURE, model_part.ProcessInfo),
                dtype=float,
            ).reshape(-1)
            integration_method = _resolve_integration_method(
                KM,
                elem,
                geom,
                n_qp=int(elem_subscale_velocity.shape[0]),
            )
            elem_q_coords_ref, elem_q_coords_cur, elem_q_weights = _integration_point_coords(
                geom,
                integration_method,
                n_qp_hint=int(elem_subscale_velocity.shape[0]),
            )
            q_coords_ref_list.append(np.asarray(elem_q_coords_ref, dtype=float))
            q_coords_cur_list.append(np.asarray(elem_q_coords_cur, dtype=float))
            q_weights_list.append(np.asarray(elem_q_weights, dtype=float))
            subscale_velocity_list.append(np.asarray(elem_subscale_velocity, dtype=float))
            subscale_pressure_list.append(np.asarray(elem_subscale_pressure, dtype=float))
        q_counts = np.asarray([arr.shape[0] for arr in q_coords_ref_list], dtype=int)
        q_offsets = np.zeros((n_elem + 1,), dtype=int)
        if q_counts.size:
            q_offsets[1:] = np.cumsum(q_counts, dtype=int)
        q_coords_ref_flat = (
            np.concatenate(q_coords_ref_list, axis=0)
            if q_coords_ref_list
            else np.zeros((0, 2), dtype=float)
        )
        q_coords_cur_flat = (
            np.concatenate(q_coords_cur_list, axis=0)
            if q_coords_cur_list
            else np.zeros((0, 2), dtype=float)
        )
        q_weights_flat = (
            np.concatenate(q_weights_list, axis=0)
            if q_weights_list
            else np.zeros((0,), dtype=float)
        )
        subscale_velocity_flat = (
            np.concatenate(subscale_velocity_list, axis=0)
            if subscale_velocity_list
            else np.zeros((0, 2), dtype=float)
        )
        subscale_pressure_flat = (
            np.concatenate(subscale_pressure_list, axis=0)
            if subscale_pressure_list
            else np.zeros((0,), dtype=float)
        )
        payload[f"{solver_name}_element_node_ids"] = node_ids_by_elem
        payload[f"{solver_name}_q_point_offsets"] = q_offsets
        payload[f"{solver_name}_q_point_counts"] = q_counts
        payload[f"{solver_name}_q_coords_ref_flat"] = q_coords_ref_flat
        payload[f"{solver_name}_q_coords_cur_flat"] = q_coords_cur_flat
        payload[f"{solver_name}_q_weights_flat"] = q_weights_flat
        payload[f"{solver_name}_subscale_velocity_flat"] = subscale_velocity_flat
        payload[f"{solver_name}_subscale_pressure_flat"] = subscale_pressure_flat
        if q_counts.size and int(np.min(q_counts)) == int(np.max(q_counts)):
            n_qp = int(q_counts[0])
            payload[f"{solver_name}_q_coords_ref"] = q_coords_ref_flat.reshape(n_elem, n_qp, 2)
            payload[f"{solver_name}_q_coords_cur"] = q_coords_cur_flat.reshape(n_elem, n_qp, 2)
            payload[f"{solver_name}_q_weights"] = q_weights_flat.reshape(n_elem, n_qp)
            payload[f"{solver_name}_subscale_velocity"] = subscale_velocity_flat.reshape(n_elem, n_qp, 2)
            payload[f"{solver_name}_subscale_pressure"] = subscale_pressure_flat.reshape(n_elem, n_qp)
    elif str(solver_name) == "structure":
        point_load_var = getattr(KSM, "POINT_LOAD", None) if KSM is not None else None
        payload[f"{solver_name}_displacement_nodal_values"] = _safe_vector_step_values(nodes, KM.DISPLACEMENT)
        payload[f"{solver_name}_velocity_nodal_values"] = _safe_vector_step_values(nodes, KM.VELOCITY)
        payload[f"{solver_name}_reaction_nodal_values"] = _safe_vector_step_values(nodes, KM.REACTION)
        payload[f"{solver_name}_point_load_nodal_values"] = _safe_vector_step_values(nodes, point_load_var)
        conditions = [cond for cond in model_part.Conditions if cond.GetGeometry().PointsNumber() == 1]
        cond_ids, cond_node_ids, cond_values = _safe_condition_vector_values(conditions, point_load_var)
        payload[f"{solver_name}_point_load_condition_ids"] = cond_ids
        payload[f"{solver_name}_point_load_condition_node_ids"] = cond_node_ids
        payload[f"{solver_name}_point_load_condition_values"] = cond_values

    return payload


def _residual_norms(residual: np.ndarray, current: np.ndarray) -> dict[str, float]:
    residual_arr = np.asarray(residual, dtype=float).reshape(-1)
    current_arr = np.asarray(current, dtype=float).reshape(-1)
    res_norm = float(np.linalg.norm(residual_arr))
    current_norm = float(np.linalg.norm(current_arr))
    if current_norm < 1.0e-15:
        current_norm = 1.0
    abs_norm = res_norm / np.sqrt(max(residual_arr.size, 1))
    rel_norm = res_norm / current_norm
    return {
        "res_norm": res_norm,
        "abs_norm": float(abs_norm),
        "rel_norm": float(rel_norm),
    }


class KratosCouplingMonitor:
    def __init__(
        self,
        *,
        monitor_dir: Path,
        solver_names: tuple[str, ...] = ("fluid", "structure"),
        field_names: tuple[str, ...] = ("load", "disp", "velocity"),
        stage_filters: tuple[str, ...] | None = None,
        include_model_part_state: bool = False,
    ) -> None:
        self.monitor_dir = Path(monitor_dir)
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        self.solver_names = tuple(str(name) for name in solver_names)
        self.field_names = tuple(str(name) for name in field_names)
        self.stage_filters = tuple(str(name) for name in stage_filters) if stage_filters else None
        self.include_model_part_state = bool(include_model_part_state)
        self.records: list[dict[str, Any]] = []

    def _want_stage(self, stage: str) -> bool:
        if self.stage_filters is None:
            return True
        stage_value = str(stage)
        return any(
            token == stage_value or token in stage_value
            for token in self.stage_filters
        )

    def dump(self, coupled_solver, *, stage: str, iteration: int, extra: dict[str, Any] | None = None) -> None:
        import KratosMultiphysics as KM

        if not self._want_stage(stage):
            return
        step_value = None
        time_value = None
        for solver_name in self.solver_names:
            solver = coupled_solver.solver_wrappers.get(solver_name)
            if solver is None:
                continue
            step_candidate, time_candidate = _solver_wrapper_step_time(KM, solver, str(solver_name))
            if step_value is None and step_candidate is not None:
                step_value = int(step_candidate)
            if time_value is None and time_candidate is not None:
                time_value = float(time_candidate)

        if step_value is None or time_value is None:
            fallback_step, fallback_time = _process_info_step_time(KM, coupled_solver.process_info)
            if step_value is None:
                step_value = fallback_step
            if time_value is None:
                time_value = fallback_time

        payload: dict[str, Any] = {
            "stage": str(stage),
            "iteration": int(iteration),
            "time": 0.0 if time_value is None else float(time_value),
            "step": 0 if step_value is None else int(step_value),
        }
        if extra:
            for key, value in extra.items():
                if isinstance(value, np.ndarray):
                    continue
                if isinstance(value, (np.generic,)):
                    payload[key] = value.item()
                else:
                    payload[key] = value

        npz_payload: dict[str, np.ndarray] = {}
        npz_payload["step"] = np.asarray(int(payload["step"]), dtype=int)
        npz_payload["time_s"] = np.asarray(float(payload["time"]), dtype=float)
        npz_payload["iteration"] = np.asarray(int(iteration), dtype=int)
        npz_payload["stage"] = np.asarray(str(stage))
        for solver_name in self.solver_names:
            solver = coupled_solver.solver_wrappers.get(solver_name)
            if solver is None or not getattr(solver, "IsDefinedOnThisRank", lambda: False)():
                continue
            if self.include_model_part_state:
                model_part = _try_get_solver_model_part(solver, str(solver_name))
                if model_part is not None:
                    npz_payload.update(_model_part_state_payload(model_part, solver_name=str(solver_name)))
            for field_name in self.field_names:
                try:
                    interface_data = solver.GetInterfaceData(field_name)
                except Exception:
                    continue
                if not interface_data.IsDefinedOnThisRank():
                    continue
                ids, coords_ref, coords_cur = _interface_node_arrays(interface_data)
                values = _reshape_interface_values(interface_data)
                prefix = f"{solver_name}_{field_name}"
                npz_payload[f"{prefix}_node_ids"] = ids
                npz_payload[f"{prefix}_coords_ref"] = coords_ref
                npz_payload[f"{prefix}_coords_cur"] = coords_cur
                npz_payload[f"{prefix}_values"] = values
                payload[f"{prefix}_size"] = int(values.shape[0])
                payload[f"{prefix}_dim"] = int(values.shape[1])

        if extra:
            for key, value in extra.items():
                if isinstance(value, np.ndarray):
                    npz_payload[key] = np.asarray(value)

        stem = f"step{payload['step']:04d}_iter{int(iteration):04d}_{stage}"
        np.savez(self.monitor_dir / f"{stem}.npz", **npz_payload)
        self.records.append(payload)

    def finalize(self) -> None:
        _write_json(self.monitor_dir / "manifest.json", {"records": self.records})


class KratosStepHistoryDumper:
    def __init__(
        self,
        *,
        step_history_dir: Path,
        solver_names: tuple[str, ...] = ("fluid", "structure"),
        field_names: tuple[str, ...] = ("load", "disp", "velocity"),
        every: int = 1,
        target_step: int | None = None,
    ) -> None:
        self.step_history_dir = Path(step_history_dir)
        self.step_history_dir.mkdir(parents=True, exist_ok=True)
        self.solver_names = tuple(str(name) for name in solver_names)
        self.field_names = tuple(str(name) for name in field_names)
        self.every = max(int(every), 1)
        self.target_step = None if target_step is None else int(target_step)
        self.records: list[dict[str, Any]] = []

    def dump(self, analysis) -> None:
        step = int(getattr(analysis, "step", 0))
        time_value = float(getattr(analysis, "time", 0.0))
        if step <= 0 or (step % self.every) != 0:
            return
        if self.target_step is not None and step != self.target_step:
            return
        coupled_solver = analysis._GetSolver()
        payload: dict[str, np.ndarray] = {
            "step": np.asarray(step, dtype=int),
            "time_s": np.asarray(time_value, dtype=float),
        }
        record: dict[str, Any] = {
            "step": step,
            "time_s": time_value,
        }

        normalized_interface_names = {
            ("fluid", "load"): "interface_load",
            ("structure", "disp"): "interface_disp",
            ("fluid", "velocity"): "interface_velocity",
        }

        for solver_name in self.solver_names:
            solver = coupled_solver.solver_wrappers.get(solver_name)
            if solver is None or not getattr(solver, "IsDefinedOnThisRank", lambda: False)():
                continue
            model_part = _try_get_solver_model_part(solver, solver_name)
            if model_part is not None:
                payload.update(_model_part_state_payload(model_part, solver_name=str(solver_name)))
            for field_name in self.field_names:
                try:
                    interface_data = solver.GetInterfaceData(field_name)
                except Exception:
                    continue
                if not interface_data.IsDefinedOnThisRank():
                    continue
                ids, coords_ref, coords_cur = _interface_node_arrays(interface_data)
                values = _reshape_interface_values(interface_data)
                prefix = f"{solver_name}_{field_name}"
                payload[f"{prefix}_node_ids"] = ids
                payload[f"{prefix}_coords_ref"] = coords_ref
                payload[f"{prefix}_coords_cur"] = coords_cur
                payload[f"{prefix}_values"] = values
                alias = normalized_interface_names.get((str(solver_name), str(field_name)))
                if alias is not None:
                    payload[f"{alias}_coords_ref"] = coords_ref
                    payload[f"{alias}_coords_cur"] = coords_cur
                    payload[f"{alias}_values"] = values
                record[f"{prefix}_size"] = int(values.shape[0])
                record[f"{prefix}_dim"] = int(values.shape[1])

        step_path = self.step_history_dir / f"step{step:04d}.npz"
        np.savez_compressed(step_path, **payload)
        record["npz_path"] = str(step_path)
        self.records.append(record)

    def finalize(self) -> None:
        _write_json(self.step_history_dir / "manifest.json", {"records": self.records})


def _install_kratos_coupling_monitor(*, monitor: KratosCouplingMonitor) -> None:
    import KratosMultiphysics as KM
    import KratosMultiphysics.CoSimulationApplication as KratosCoSim
    import KratosMultiphysics.CoSimulationApplication.co_simulation_tools as cs_tools
    import KratosMultiphysics.CoSimulationApplication.colors as colors
    from KratosMultiphysics.CoSimulationApplication.coupled_solvers import gauss_seidel_strong

    cls = gauss_seidel_strong.GaussSeidelStrongCoupledSolver
    if getattr(cls, "_pycutfem_monitor_installed", False):
        return

    @wraps(cls.SolveSolutionStep)
    def _monitored_solve_solution_step(self):
        for k in range(self.num_coupling_iterations):
            self.process_info[KratosCoSim.COUPLING_ITERATION_NUMBER] += 1
            iteration = int(k + 1)

            if self.echo_level > 0:
                cs_tools.cs_print_info(
                    self._ClassName(),
                    colors.cyan("Coupling iteration:"),
                    colors.bold(str(iteration) + " / " + str(self.num_coupling_iterations)),
                )

            for coupling_op in self.coupling_operations_dict.values():
                coupling_op.InitializeCouplingIteration()

            for conv_acc in self.convergence_accelerators_list:
                conv_acc.InitializeNonLinearIteration()

            for conv_crit in self.convergence_criteria_list:
                conv_crit.InitializeNonLinearIteration()

            monitor.dump(self, stage="iter_start", iteration=iteration)

            for solver_name, solver in self.solver_wrappers.items():
                self._SynchronizeInputData(solver_name)
                monitor.dump(self, stage=f"after_sync_input_{solver_name}", iteration=iteration)
                solver.SolveSolutionStep()
                monitor.dump(self, stage=f"after_solve_{solver_name}", iteration=iteration)
                self._SynchronizeOutputData(solver_name)
                monitor.dump(self, stage=f"after_sync_output_{solver_name}", iteration=iteration)

            for coupling_op in self.coupling_operations_dict.values():
                coupling_op.FinalizeCouplingIteration()

            for conv_acc in self.convergence_accelerators_list:
                conv_acc.FinalizeNonLinearIteration()

            for conv_crit in self.convergence_criteria_list:
                conv_crit.FinalizeNonLinearIteration()

            crit_summary: dict[str, Any] = {}
            crit_states: list[bool] = []
            for idx, conv_crit in enumerate(self.convergence_criteria_list):
                if not hasattr(conv_crit, "interface_data") or not conv_crit.interface_data.IsDefinedOnThisRank():
                    continue
                current = np.asarray(conv_crit.interface_data.GetData(), dtype=float)
                previous = np.asarray(getattr(conv_crit, "input_data", np.zeros_like(current)), dtype=float)
                residual = current - previous
                norms = _residual_norms(residual, current)
                label = f"crit_{idx}_{conv_crit.interface_data.solver_name}_{conv_crit.interface_data.name}"
                for key, value in norms.items():
                    crit_summary[f"{label}_{key}"] = float(value)
                crit_summary[f"{label}_current"] = np.asarray(current, dtype=float)
                crit_summary[f"{label}_previous"] = np.asarray(previous, dtype=float)
                crit_summary[f"{label}_residual"] = np.asarray(residual, dtype=float)
                is_converged_i = bool(conv_crit.IsConverged())
                crit_summary[f"{label}_is_converged"] = is_converged_i
                crit_states.append(is_converged_i)

            is_converged = all(crit_states) if crit_states else False
            crit_summary["is_converged"] = bool(is_converged)
            monitor.dump(self, stage="after_iteration", iteration=iteration, extra=crit_summary)

            if is_converged:
                if self.echo_level > 0:
                    cs_tools.cs_print_info(self._ClassName(), colors.green("### CONVERGENCE WAS ACHIEVED ###"))
                self._GaussSeidelStrongCoupledSolver__CommunicateIfTimeStepNeedsToBeRepeated(False)
                return True

            if iteration >= self.num_coupling_iterations:
                if self.echo_level > 0:
                    cs_tools.cs_print_info(self._ClassName(), colors.red("XXX CONVERGENCE WAS NOT ACHIEVED XXX"))
                self._GaussSeidelStrongCoupledSolver__CommunicateIfTimeStepNeedsToBeRepeated(False)
                return False

            self._GaussSeidelStrongCoupledSolver__CommunicateIfTimeStepNeedsToBeRepeated(True)
            monitor.dump(self, stage="pre_update", iteration=iteration, extra=crit_summary)
            for conv_acc in self.convergence_accelerators_list:
                conv_acc.ComputeAndApplyUpdate()
            acc_summary: dict[str, Any] = {}
            for idx, conv_acc in enumerate(self.convergence_accelerators_list):
                if not hasattr(conv_acc, "interface_data") or not conv_acc.interface_data.IsDefinedOnThisRank():
                    continue
                current = np.asarray(conv_acc.interface_data.GetData(), dtype=float)
                previous = np.asarray(getattr(conv_acc, "input_data", np.zeros_like(current)), dtype=float)
                residual = current - previous
                norms = _residual_norms(residual, current)
                label = f"acc_{idx}_{conv_acc.interface_data.solver_name}_{conv_acc.interface_data.name}"
                for key, value in norms.items():
                    acc_summary[f"{label}_{key}"] = float(value)
                acc_summary[f"{label}_current"] = np.asarray(current, dtype=float)
                acc_summary[f"{label}_previous"] = np.asarray(previous, dtype=float)
                acc_summary[f"{label}_residual"] = np.asarray(residual, dtype=float)
                if hasattr(conv_acc, "R"):
                    acc_summary[f"{label}_buffer_R_len"] = int(len(conv_acc.R))
                    for ridx, values in enumerate(list(conv_acc.R)):
                        acc_summary[f"{label}_R_{ridx}"] = np.asarray(values, dtype=float)
                if hasattr(conv_acc, "X"):
                    acc_summary[f"{label}_buffer_X_len"] = int(len(conv_acc.X))
                    for xidx, values in enumerate(list(conv_acc.X)):
                        acc_summary[f"{label}_X_{xidx}"] = np.asarray(values, dtype=float)
                if hasattr(conv_acc, "V_new") and len(getattr(conv_acc, "V_new", [])) != 0:
                    acc_summary[f"{label}_V_new"] = np.asarray(conv_acc.V_new, dtype=float)
                if hasattr(conv_acc, "W_new") and len(getattr(conv_acc, "W_new", [])) != 0:
                    acc_summary[f"{label}_W_new"] = np.asarray(conv_acc.W_new, dtype=float)
                if hasattr(conv_acc, "V_old") and len(getattr(conv_acc, "V_old", [])) != 0:
                    acc_summary[f"{label}_V_old"] = np.asarray(conv_acc.V_old, dtype=float)
                if hasattr(conv_acc, "W_old") and len(getattr(conv_acc, "W_old", [])) != 0:
                    acc_summary[f"{label}_W_old"] = np.asarray(conv_acc.W_old, dtype=float)
            monitor.dump(self, stage="post_update", iteration=iteration, extra=acc_summary)

    cls.SolveSolutionStep = _monitored_solve_solution_step
    cls._pycutfem_monitor_installed = True


def _install_kratos_step_history_dumper(*, dumper: KratosStepHistoryDumper) -> None:
    from KratosMultiphysics.CoSimulationApplication import co_simulation_analysis

    cls = co_simulation_analysis.CoSimulationAnalysis
    if getattr(cls, "_pycutfem_step_history_installed", False):
        return

    original = cls.OutputSolutionStep

    @wraps(original)
    def _monitored_output_solution_step(self):
        result = original(self)
        dumper.dump(self)
        return result

    cls.OutputSolutionStep = _monitored_output_solution_step
    cls._pycutfem_step_history_installed = True


def _split_csv_values(text: str | None) -> tuple[str, ...] | None:
    if text is None:
        return None
    values = tuple(part.strip() for part in str(text).split(",") if part.strip())
    return values or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the adapted Kratos DoubleFlap step-1 FOM reference case.")
    parser.add_argument("--benchmark-root", type=Path, default=default_double_flap_root())
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("examples/NIRB/artifacts/kratos_example2_reference_step1"),
    )
    parser.add_argument("--end-time", type=float, default=0.008)
    parser.add_argument("--echo-level", type=int, default=2)
    parser.add_argument("--fluid-echo-level", type=int, default=1)
    parser.add_argument("--solid-echo-level", type=int, default=1)
    parser.add_argument("--fluid-output-path", type=str, default="vtk_output_fsi_cfd")
    parser.add_argument("--solid-output-path", type=str, default="vtk_output_fsi_csm")
    parser.add_argument("--monitor-coupling", action="store_true", help="Dump interface data around each strong-coupling substep.")
    parser.add_argument("--monitor-dir", type=Path, default=None, help="Directory for monitored coupling npz files.")
    parser.add_argument("--monitor-solvers", type=str, default="fluid,structure", help="Comma-separated solver names to monitor.")
    parser.add_argument("--monitor-fields", type=str, default="load,disp,velocity", help="Comma-separated interface data names to monitor.")
    parser.add_argument(
        "--monitor-model-state",
        action="store_true",
        help="Include full solver model-part state in each monitored coupling-stage dump.",
    )
    parser.add_argument(
        "--monitor-stages",
        type=str,
        default=None,
        help="Optional comma-separated stage filter. Example: after_solve_fluid,after_sync_output_fluid,post_update",
    )
    parser.add_argument("--dump-step-history", action="store_true", help="Dump accepted-step full-state npz files for later comparison.")
    parser.add_argument("--step-history-dir", type=Path, default=None, help="Directory for accepted-step state dumps.")
    parser.add_argument("--step-history-solvers", type=str, default="fluid,structure", help="Comma-separated solver names to include in step-history dumps.")
    parser.add_argument("--step-history-fields", type=str, default="load,disp,velocity", help="Comma-separated interface data names to include in step-history dumps.")
    parser.add_argument("--step-history-every", type=int, default=1, help="Write one accepted-step dump every N global steps when step-history dumping is enabled.")
    parser.add_argument("--step-history-target-step", type=int, default=None, help="Optional exact accepted step to dump when step-history dumping is enabled.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_root = Path(args.benchmark_root).resolve()
    run_dir = Path(args.run_dir).resolve()

    if not benchmark_root.exists():
        raise FileNotFoundError(f"Benchmark root not found: {benchmark_root}")

    _copy_inputs(benchmark_root, run_dir)

    coupling_json = _prepare_coupling_json(
        _load_json(benchmark_root / "DoubleFlap_fsi_parameters_ROM.json"),
        end_time=float(args.end_time),
        echo_level=int(args.echo_level),
    )
    fluid_json = _prepare_fluid_json(
        _load_json(benchmark_root / "ProjectParametersCFD.json"),
        end_time=float(args.end_time),
        echo_level=int(args.fluid_echo_level),
        output_path=str(args.fluid_output_path),
    )
    solid_json = _prepare_solid_json(
        _load_json(benchmark_root / "ProjectParametersCSM.json"),
        end_time=float(args.end_time),
        echo_level=int(args.solid_echo_level),
        output_path=str(args.solid_output_path),
    )

    _write_json(run_dir / "DoubleFlap_fsi_parameters_ROM.json", coupling_json)
    _write_json(run_dir / "ProjectParametersCFD.json", fluid_json)
    _write_json(run_dir / "ProjectParametersCSM.json", solid_json)

    try:
        import KratosMultiphysics as KM
        import KratosMultiphysics.ConstitutiveLawsApplication  # noqa: F401
        from KratosMultiphysics.CoSimulationApplication.co_simulation_analysis import CoSimulationAnalysis
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "KratosMultiphysics is not importable in the current Python environment. "
            "Run this script from a Kratos-enabled Python when launching the monitored reference case."
        ) from exc

    monitor: KratosCouplingMonitor | None = None
    step_history: KratosStepHistoryDumper | None = None
    if bool(args.monitor_coupling):
        monitor_dir = Path(args.monitor_dir) if args.monitor_dir is not None else run_dir / "coupling_monitor"
        monitor = KratosCouplingMonitor(
            monitor_dir=monitor_dir,
            solver_names=_split_csv_values(args.monitor_solvers) or ("fluid", "structure"),
            field_names=_split_csv_values(args.monitor_fields) or ("load", "disp", "velocity"),
            stage_filters=_split_csv_values(args.monitor_stages),
            include_model_part_state=bool(args.monitor_model_state),
        )
        _install_kratos_coupling_monitor(monitor=monitor)
    if bool(args.dump_step_history):
        step_history_dir = Path(args.step_history_dir) if args.step_history_dir is not None else run_dir / "step_history"
        step_history = KratosStepHistoryDumper(
            step_history_dir=step_history_dir,
            solver_names=_split_csv_values(args.step_history_solvers) or ("fluid", "structure"),
            field_names=_split_csv_values(args.step_history_fields) or ("load", "disp", "velocity"),
            every=int(args.step_history_every),
            target_step=None if args.step_history_target_step is None else int(args.step_history_target_step),
        )
        _install_kratos_step_history_dumper(dumper=step_history)

    with (run_dir / "DoubleFlap_fsi_parameters_ROM.json").open("r", encoding="utf-8") as f:
        parameters = KM.Parameters(f.read())

    with contextlib.chdir(run_dir):
        simulation = CoSimulationAnalysis(parameters)
        try:
            simulation.Run()
        finally:
            if monitor is not None:
                monitor.finalize()
            if step_history is not None:
                step_history.finalize()

    summary = {
        "benchmark_root": str(benchmark_root),
        "run_dir": str(run_dir),
        "end_time": float(args.end_time),
        "fluid_vtk_dir": str(run_dir / str(args.fluid_output_path)),
        "solid_vtk_dir": str(run_dir / str(args.solid_output_path)),
        "coupling_parameters_path": str(run_dir / "DoubleFlap_fsi_parameters_ROM.json"),
        "fluid_parameters_path": str(run_dir / "ProjectParametersCFD.json"),
        "solid_parameters_path": str(run_dir / "ProjectParametersCSM.json"),
        "monitor_enabled": bool(args.monitor_coupling),
        "monitor_dir": str((Path(args.monitor_dir) if args.monitor_dir is not None else run_dir / "coupling_monitor").resolve())
        if bool(args.monitor_coupling)
        else None,
        "monitor_solvers": list(_split_csv_values(args.monitor_solvers) or ("fluid", "structure")),
        "monitor_fields": list(_split_csv_values(args.monitor_fields) or ("load", "disp", "velocity")),
        "monitor_model_state": bool(args.monitor_model_state),
        "monitor_stages": list(_split_csv_values(args.monitor_stages) or []),
        "step_history_enabled": bool(args.dump_step_history),
        "step_history_dir": str((Path(args.step_history_dir) if args.step_history_dir is not None else run_dir / "step_history").resolve())
        if bool(args.dump_step_history)
        else None,
        "step_history_solvers": list(_split_csv_values(args.step_history_solvers) or ("fluid", "structure")),
        "step_history_fields": list(_split_csv_values(args.step_history_fields) or ("load", "disp", "velocity")),
        "step_history_every": int(args.step_history_every),
        "step_history_target_step": None if args.step_history_target_step is None else int(args.step_history_target_step),
    }
    _write_json(run_dir / "summary.json", summary)
    print(f"summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
