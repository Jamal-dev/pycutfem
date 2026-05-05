from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from examples.NIRB.double_flap_reference import _load_json, default_double_flap_root
from examples.NIRB.debug.run_kratos_example2_reference import (
    _copy_inputs,
    _prepare_coupling_json,
    _prepare_fluid_json,
    _prepare_solid_json,
    _write_json,
)


def _try_get_solver(solver_wrapper) -> Any | None:
    for attr in ("_analysis_stage", "analysis_stage"):
        stage = getattr(solver_wrapper, attr, None)
        if stage is None:
            continue
        for getter in ("_GetSolver", "GetSolver"):
            fn = getattr(stage, getter, None)
            if callable(fn):
                try:
                    solver = fn()
                except Exception:
                    continue
                if solver is not None:
                    return solver
    return None


def _pack_csr(matrix_csr) -> dict[str, np.ndarray]:
    return {
        "data": np.asarray(matrix_csr.data, dtype=float),
        "indices": np.asarray(matrix_csr.indices, dtype=np.int32),
        "indptr": np.asarray(matrix_csr.indptr, dtype=np.int32),
        "shape": np.asarray(matrix_csr.shape, dtype=np.int32),
    }


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
        if step_from_time > 0 and (
            step_value is None or step_value <= 1 < step_from_time or step_value != step_from_time
        ):
            step_value = step_from_time

    return step_value, time_value


def _ordered_dof_metadata(KM, model_part, ndof: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_ids = np.full((ndof,), -1, dtype=np.int32)
    is_fixed = np.zeros((ndof,), dtype=bool)
    var_names = np.full((ndof,), "", dtype=object)
    values = np.zeros((ndof,), dtype=float)
    vars_to_dump = (
        (KM.DISPLACEMENT_X, "DISPLACEMENT_X"),
        (KM.DISPLACEMENT_Y, "DISPLACEMENT_Y"),
    )
    for node in model_part.Nodes:
        node_id = int(node.Id)
        for variable, label in vars_to_dump:
            if not node.HasDofFor(variable):
                continue
            dof = node.GetDof(variable)
            eq_id = int(dof.EquationId)
            if eq_id < 0 or eq_id >= ndof:
                continue
            node_ids[eq_id] = node_id
            is_fixed[eq_id] = bool(dof.IsFixed())
            var_names[eq_id] = str(label)
            try:
                values[eq_id] = float(dof.GetSolutionStepValue())
            except Exception:
                values[eq_id] = 0.0
    return node_ids, var_names, is_fixed, values


def _safe_vector_step_values(nodes, variable) -> np.ndarray:
    values = np.zeros((len(nodes), 2), dtype=float)
    if variable is None:
        return values
    for i, node in enumerate(nodes):
        try:
            raw = node.GetSolutionStepValue(variable)
            arr = np.asarray(raw, dtype=float).reshape(-1)
            if arr.size >= 2:
                values[i, :] = arr[:2]
            elif arr.size == 1:
                values[i, 0] = arr[0]
        except Exception:
            values[i, :] = 0.0
    return values


def _dump_structural_system(
    *,
    KM,
    KSM,
    scipy_conversion_tools,
    structure_solver,
    output_npz: Path,
    stage: str,
    step: int | None,
    iteration: int,
    time_s: float | None,
) -> None:
    strategy = structure_solver._GetSolutionStrategy()
    builder = structure_solver._GetBuilderAndSolver()
    scheme = structure_solver._GetScheme()
    computing_model_part = structure_solver.GetComputingModelPart()
    space = KM.UblasSparseSpace()

    A = strategy.GetSystemMatrix()
    b = strategy.GetSystemVector()
    x = strategy.GetSolutionVector()
    space.SetToZeroMatrix(A)
    space.SetToZeroVector(b)
    builder.Build(scheme, computing_model_part, A, b)
    A_raw = scipy_conversion_tools.to_csr(A).copy()
    b_raw = np.asarray(b, dtype=float).copy()
    builder.ApplyDirichletConditions(scheme, computing_model_part, A, x, b)
    A_constrained = scipy_conversion_tools.to_csr(A).copy()
    b_constrained = np.asarray(b, dtype=float).copy()

    root_model_part = structure_solver.main_model_part
    nodes = list(root_model_part.Nodes)
    node_ids = np.asarray([int(node.Id) for node in nodes], dtype=int)
    node_coords_ref = np.asarray([[float(node.X0), float(node.Y0)] for node in nodes], dtype=float)
    node_coords_cur = np.asarray([[float(node.X), float(node.Y)] for node in nodes], dtype=float)
    displacement = _safe_vector_step_values(nodes, KM.DISPLACEMENT)
    velocity = _safe_vector_step_values(nodes, getattr(KM, "VELOCITY", None))
    reaction = _safe_vector_step_values(nodes, getattr(KM, "REACTION", None))
    point_load = _safe_vector_step_values(nodes, getattr(KSM, "POINT_LOAD", None))

    point_conditions = [cond for cond in root_model_part.Conditions if cond.GetGeometry().PointsNumber() == 1]
    point_load_condition_ids = np.asarray([int(cond.Id) for cond in point_conditions], dtype=int)
    point_load_condition_node_ids = np.asarray([[int(cond.GetGeometry()[0].Id)] for cond in point_conditions], dtype=int)
    point_load_condition_values = np.zeros((len(point_conditions), 2), dtype=float)
    for i, cond in enumerate(point_conditions):
        try:
            raw = np.asarray(cond.GetValue(KSM.POINT_LOAD), dtype=float).reshape(-1)
            if raw.size >= 2:
                point_load_condition_values[i, :] = raw[:2]
            elif raw.size == 1:
                point_load_condition_values[i, 0] = raw[0]
        except Exception:
            pass

    eq_node_ids, eq_var_names, eq_is_fixed, eq_values = _ordered_dof_metadata(KM, computing_model_part, int(A_raw.shape[0]))

    np.savez(
        output_npz,
        stage=str(stage),
        step=np.asarray(-1 if step is None else int(step), dtype=int),
        iteration=int(iteration),
        time_s=np.asarray(np.nan if time_s is None else float(time_s), dtype=float),
        node_ids=node_ids,
        node_coords_ref=node_coords_ref,
        node_coords_cur=node_coords_cur,
        displacement=displacement,
        velocity=velocity,
        reaction=reaction,
        point_load=point_load,
        point_load_condition_ids=point_load_condition_ids,
        point_load_condition_node_ids=point_load_condition_node_ids,
        point_load_condition_values=point_load_condition_values,
        equation_node_ids=eq_node_ids,
        equation_var_names=np.asarray(eq_var_names, dtype=str),
        equation_is_fixed=np.asarray(eq_is_fixed, dtype=bool),
        equation_values=np.asarray(eq_values, dtype=float),
        A_raw_data=_pack_csr(A_raw)["data"],
        A_raw_indices=_pack_csr(A_raw)["indices"],
        A_raw_indptr=_pack_csr(A_raw)["indptr"],
        A_raw_shape=_pack_csr(A_raw)["shape"],
        b_raw=b_raw,
        A_constrained_data=_pack_csr(A_constrained)["data"],
        A_constrained_indices=_pack_csr(A_constrained)["indices"],
        A_constrained_indptr=_pack_csr(A_constrained)["indptr"],
        A_constrained_shape=_pack_csr(A_constrained)["shape"],
        b_constrained=b_constrained,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run coupled Kratos DoubleFlap and dump the assembled structural global system at a chosen coupling stage."
    )
    parser.add_argument("--benchmark-root", type=Path, default=default_double_flap_root())
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("examples/NIRB/artifacts/kratos_structure_global_system"),
    )
    parser.add_argument("--end-time", type=float, default=0.008)
    parser.add_argument("--echo-level", type=int, default=0)
    parser.add_argument("--fluid-echo-level", type=int, default=0)
    parser.add_argument("--solid-echo-level", type=int, default=0)
    parser.add_argument(
        "--target-stage",
        choices=("after_sync_input_structure", "after_solve_structure", "after_sync_output_structure"),
        default="after_sync_input_structure",
    )
    parser.add_argument("--target-step", type=int, default=1)
    parser.add_argument("--target-iteration", type=int, default=5)
    parser.add_argument("--output-stem", type=str, default="structure_global_system")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import KratosMultiphysics as KM
        import KratosMultiphysics.CoSimulationApplication as KratosCoSim
        import KratosMultiphysics.CoSimulationApplication.co_simulation_analysis as analysis
        import KratosMultiphysics.StructuralMechanicsApplication as KSM
        from KratosMultiphysics import scipy_conversion_tools
        from KratosMultiphysics.CoSimulationApplication.coupled_solvers import gauss_seidel_strong
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("KratosMultiphysics is not importable in the current environment.") from exc

    benchmark_root = Path(args.benchmark_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists():
        shutil.rmtree(run_dir)
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
        output_path="vtk_output_fsi_cfd",
    )
    solid_json = _prepare_solid_json(
        _load_json(benchmark_root / "ProjectParametersCSM.json"),
        end_time=float(args.end_time),
        echo_level=int(args.solid_echo_level),
        output_path="vtk_output_fsi_csm",
    )
    _write_json(run_dir / "DoubleFlap_fsi_parameters_ROM.json", coupling_json)
    _write_json(run_dir / "ProjectParametersCFD.json", fluid_json)
    _write_json(run_dir / "ProjectParametersCSM.json", solid_json)

    target_stage = str(args.target_stage)
    target_step = int(args.target_step)
    target_iteration = int(args.target_iteration)
    output_stem = str(args.output_stem)

    cls = gauss_seidel_strong.GaussSeidelStrongCoupledSolver
    if getattr(cls, "_pycutfem_structure_global_system_dump_installed", False):
        raise RuntimeError("Structural global-system dumper already installed in this Python process.")
    original = cls.SolveSolutionStep

    def _wrapped(self):
        for k in range(self.num_coupling_iterations):
            self.process_info[KratosCoSim.COUPLING_ITERATION_NUMBER] += 1
            iteration = int(k + 1)
            current_step, current_time = _process_info_step_time(KM, self.process_info)

            for coupling_op in self.coupling_operations_dict.values():
                coupling_op.InitializeCouplingIteration()
            for conv_acc in self.convergence_accelerators_list:
                conv_acc.InitializeNonLinearIteration()
            for conv_crit in self.convergence_criteria_list:
                conv_crit.InitializeNonLinearIteration()

            for solver_name, solver_wrapper in self.solver_wrappers.items():
                self._SynchronizeInputData(solver_name)
                if (
                    str(solver_name) == "structure"
                    and str(target_stage) == "after_sync_input_structure"
                    and current_step == target_step
                    and iteration == target_iteration
                ):
                    structure_solver = _try_get_solver(solver_wrapper)
                    if structure_solver is None:
                        raise RuntimeError("Could not access Kratos structural solver for system dump.")
                    _dump_structural_system(
                        KM=KM,
                        KSM=KSM,
                        scipy_conversion_tools=scipy_conversion_tools,
                        structure_solver=structure_solver,
                        output_npz=run_dir / f"{output_stem}.npz",
                        stage=str(target_stage),
                        step=current_step,
                        iteration=int(iteration),
                        time_s=current_time,
                    )
                solver_wrapper.SolveSolutionStep()
                if (
                    str(solver_name) == "structure"
                    and str(target_stage) == "after_solve_structure"
                    and current_step == target_step
                    and iteration == target_iteration
                ):
                    structure_solver = _try_get_solver(solver_wrapper)
                    if structure_solver is None:
                        raise RuntimeError("Could not access Kratos structural solver for system dump.")
                    _dump_structural_system(
                        KM=KM,
                        KSM=KSM,
                        scipy_conversion_tools=scipy_conversion_tools,
                        structure_solver=structure_solver,
                        output_npz=run_dir / f"{output_stem}.npz",
                        stage=str(target_stage),
                        step=current_step,
                        iteration=int(iteration),
                        time_s=current_time,
                    )
                self._SynchronizeOutputData(solver_name)
                if (
                    str(solver_name) == "structure"
                    and str(target_stage) == "after_sync_output_structure"
                    and current_step == target_step
                    and iteration == target_iteration
                ):
                    structure_solver = _try_get_solver(solver_wrapper)
                    if structure_solver is None:
                        raise RuntimeError("Could not access Kratos structural solver for system dump.")
                    _dump_structural_system(
                        KM=KM,
                        KSM=KSM,
                        scipy_conversion_tools=scipy_conversion_tools,
                        structure_solver=structure_solver,
                        output_npz=run_dir / f"{output_stem}.npz",
                        stage=str(target_stage),
                        step=current_step,
                        iteration=int(iteration),
                        time_s=current_time,
                    )

            for coupling_op in self.coupling_operations_dict.values():
                coupling_op.FinalizeCouplingIteration()
            for conv_acc in self.convergence_accelerators_list:
                conv_acc.FinalizeNonLinearIteration()
            for conv_crit in self.convergence_criteria_list:
                conv_crit.FinalizeNonLinearIteration()

            is_converged = all(conv_crit.IsConverged() for conv_crit in self.convergence_criteria_list)
            if is_converged:
                self._GaussSeidelStrongCoupledSolver__CommunicateIfTimeStepNeedsToBeRepeated(False)
                return True
            if iteration >= self.num_coupling_iterations:
                self._GaussSeidelStrongCoupledSolver__CommunicateIfTimeStepNeedsToBeRepeated(False)
                return False
            self._GaussSeidelStrongCoupledSolver__CommunicateIfTimeStepNeedsToBeRepeated(True)
            for conv_acc in self.convergence_accelerators_list:
                conv_acc.ComputeAndApplyUpdate()
        return False

    cls.SolveSolutionStep = _wrapped
    cls._pycutfem_structure_global_system_dump_installed = True

    cwd = Path.cwd()
    try:
        os.chdir(run_dir)
        with (run_dir / "DoubleFlap_fsi_parameters_ROM.json").open("r", encoding="utf-8") as f:
            params = KM.Parameters(f.read())
        params["problem_data"]["echo_level"].SetInt(int(args.echo_level))
        co_sim = analysis.CoSimulationAnalysis(params)
        co_sim.Run()
    finally:
        os.chdir(cwd)
        cls.SolveSolutionStep = original
        cls._pycutfem_structure_global_system_dump_installed = False

    npz_path = run_dir / f"{output_stem}.npz"
    if not npz_path.exists():
        raise RuntimeError(f"Requested structural system dump was not produced: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        A_raw_shape = tuple(np.asarray(data["A_raw_shape"], dtype=int).tolist())
        A_constrained_shape = tuple(np.asarray(data["A_constrained_shape"], dtype=int).tolist())
        b_raw = np.asarray(data["b_raw"], dtype=float)
        b_constrained = np.asarray(data["b_constrained"], dtype=float)
    summary = {
        "run_dir": str(run_dir),
        "target_stage": str(target_stage),
        "target_step": int(target_step),
        "target_iteration": int(target_iteration),
        "npz_path": str(npz_path),
        "system_size": int(A_raw_shape[0]),
        "raw_rhs_inf_norm": float(np.max(np.abs(b_raw))) if b_raw.size else 0.0,
        "constrained_rhs_inf_norm": float(np.max(np.abs(b_constrained))) if b_constrained.size else 0.0,
        "raw_matrix_shape": list(A_raw_shape),
        "constrained_matrix_shape": list(A_constrained_shape),
    }
    json_path = run_dir / f"{output_stem}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
