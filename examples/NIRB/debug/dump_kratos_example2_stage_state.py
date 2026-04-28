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
    _model_part_state_payload,
    _prepare_coupling_json,
    _prepare_fluid_json,
    _prepare_solid_json,
    _write_json,
)


class _StageDumpComplete(RuntimeError):
    """Internal sentinel used to stop the coupled run once the target dump is written."""


def _try_get_model_part(solver_wrapper) -> Any | None:
    candidates = []
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
        for getter in ("GetComputingModelPart", "GetFluidComputingModelPart"):
            fn = getattr(obj, getter, None)
            if callable(fn):
                try:
                    mp = fn()
                    if mp is not None:
                        return mp
                except Exception:
                    pass
        mp = getattr(obj, "main_model_part", None)
        if mp is not None:
            return mp
        mp = getattr(obj, "computing_model_part", None)
        if mp is not None:
            return mp
    return None


def _dump_model_part_state(
    model_part,
    output_path: Path,
    *,
    solver_name: str,
    step: int | None = None,
    time_s: float | None = None,
    stage: str | None = None,
    iteration: int | None = None,
) -> None:
    payload = _model_part_state_payload(model_part, solver_name=str(solver_name))
    prefix = f"{solver_name}_"
    slim_payload = {
        str(key)[len(prefix) :]: np.asarray(value)
        for key, value in payload.items()
        if str(key).startswith(prefix)
    }
    if step is not None:
        slim_payload["step"] = np.asarray(int(step), dtype=int)
    if time_s is not None:
        slim_payload["time_s"] = np.asarray(float(time_s), dtype=float)
    if stage is not None:
        slim_payload["stage"] = np.asarray(str(stage))
    if iteration is not None:
        slim_payload["iteration"] = np.asarray(int(iteration), dtype=int)
    np.savez(output_path, **slim_payload)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run coupled Kratos DoubleFlap and dump full fluid nodal state at a chosen coupling stage.")
    parser.add_argument("--benchmark-root", type=Path, default=default_double_flap_root())
    parser.add_argument("--run-dir", type=Path, default=Path("examples/NIRB/artifacts/kratos_example2_stage_state"))
    parser.add_argument("--end-time", type=float, default=0.008)
    parser.add_argument("--echo-level", type=int, default=1)
    parser.add_argument("--fluid-echo-level", type=int, default=0)
    parser.add_argument("--solid-echo-level", type=int, default=0)
    parser.add_argument("--solver-name", choices=("fluid", "structure"), default="fluid")
    parser.add_argument("--target-stage", type=str, default="after_sync_output_fluid")
    parser.add_argument("--target-step", type=int, default=1)
    parser.add_argument("--target-iteration", type=int, default=2)
    parser.add_argument("--output-stem", type=str, default="fluid_stage_state")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import KratosMultiphysics as KM
        import KratosMultiphysics.CoSimulationApplication as KratosCoSim
        import KratosMultiphysics.CoSimulationApplication.co_simulation_analysis as analysis
        from KratosMultiphysics.CoSimulationApplication.coupled_solvers import gauss_seidel_strong
    except ModuleNotFoundError as exc:
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

    # Keep the coupled settings otherwise unchanged.
    solver_name = str(args.solver_name)
    target_stage = str(args.target_stage)
    target_step = int(args.target_step)
    target_iteration = int(args.target_iteration)
    output_stem = str(args.output_stem)

    cls = gauss_seidel_strong.GaussSeidelStrongCoupledSolver
    if getattr(cls, "_pycutfem_stage_state_dump_installed", False):
        raise RuntimeError("Stage-state dumper already installed in this Python process.")
    original = cls.SolveSolutionStep
    stage_dump_context: dict[str, object] = {
        "iteration": -1,
        "dumped": False,
        "step": None,
        "time_s": None,
        "coupled_step": None,
        "coupled_time_s": None,
    }
    original_ale = None
    if solver_name == "fluid":
        try:
            from KratosMultiphysics.MeshMovingApplication.ale_fluid_solver import AleFluidSolver

            original_ale = AleFluidSolver.SolveSolutionStep

            def _wrapped_ale(self):
                is_converged = True
                for mesh_solver in self.mesh_motion_solvers:
                    is_converged &= mesh_solver.SolveSolutionStep()

                if target_stage == "after_mesh_motion_before_fluid":
                    mp = self.GetComputingModelPart()
                    current_step, current_time = _process_info_step_time(KM, getattr(mp, "ProcessInfo", None))
                    if current_step is None:
                        current_step = (
                            None
                            if stage_dump_context.get("coupled_step") is None
                            else int(stage_dump_context["coupled_step"])
                        )
                    if current_time is None:
                        current_time = (
                            None
                            if stage_dump_context.get("coupled_time_s") is None
                            else float(stage_dump_context["coupled_time_s"])
                        )
                    coupling_iteration = int(stage_dump_context.get("iteration", -1))
                    if (
                        coupling_iteration == target_iteration
                        and current_step == target_step
                        and not bool(stage_dump_context.get("dumped", False))
                    ):
                        _dump_model_part_state(
                            mp,
                            run_dir / f"{output_stem}.npz",
                            solver_name="fluid",
                            step=current_step,
                            time_s=current_time,
                            stage=target_stage,
                            iteration=coupling_iteration,
                        )
                        stage_dump_context["dumped"] = True
                        stage_dump_context["step"] = current_step
                        stage_dump_context["time_s"] = current_time
                        raise _StageDumpComplete()

                if self.fluid_solver.GetComputingModelPart().ProcessInfo[KM.TIME] >= self.start_fluid_solution_time:
                    self._AleFluidSolver__ApplyALEBoundaryCondition()
                    if target_stage == "after_ale_boundary_before_fluid":
                        mp = self.GetComputingModelPart()
                        current_step, current_time = _process_info_step_time(KM, getattr(mp, "ProcessInfo", None))
                        if current_step is None:
                            current_step = (
                                None
                                if stage_dump_context.get("coupled_step") is None
                                else int(stage_dump_context["coupled_step"])
                            )
                        if current_time is None:
                            current_time = (
                                None
                                if stage_dump_context.get("coupled_time_s") is None
                                else float(stage_dump_context["coupled_time_s"])
                            )
                        coupling_iteration = int(stage_dump_context.get("iteration", -1))
                        if (
                            coupling_iteration == target_iteration
                            and current_step == target_step
                            and not bool(stage_dump_context.get("dumped", False))
                        ):
                            _dump_model_part_state(
                                mp,
                                run_dir / f"{output_stem}.npz",
                                solver_name="fluid",
                                step=current_step,
                                time_s=current_time,
                                stage=target_stage,
                                iteration=coupling_iteration,
                            )
                            stage_dump_context["dumped"] = True
                            stage_dump_context["step"] = current_step
                            stage_dump_context["time_s"] = current_time
                            raise _StageDumpComplete()
                    is_converged &= self.fluid_solver.SolveSolutionStep()

                return is_converged

            AleFluidSolver.SolveSolutionStep = _wrapped_ale
        except Exception:
            original_ale = None

    def _wrapped(self):
        for k in range(self.num_coupling_iterations):
            self.process_info[KratosCoSim.COUPLING_ITERATION_NUMBER] += 1
            iteration = int(k + 1)
            stage_dump_context["iteration"] = iteration
            coupled_step, coupled_time = _process_info_step_time(KM, self.process_info)
            stage_dump_context["coupled_step"] = coupled_step
            stage_dump_context["coupled_time_s"] = coupled_time

            for coupling_op in self.coupling_operations_dict.values():
                coupling_op.InitializeCouplingIteration()
            for conv_acc in self.convergence_accelerators_list:
                conv_acc.InitializeNonLinearIteration()
            for conv_crit in self.convergence_criteria_list:
                conv_crit.InitializeNonLinearIteration()

            for solver_name, solver in self.solver_wrappers.items():
                self._SynchronizeInputData(solver_name)
                if target_stage == f"after_sync_input_{solver_name}" and iteration == target_iteration and solver_name == str(args.solver_name):
                    mp = _try_get_model_part(solver)
                    if mp is None:
                        raise RuntimeError(f"Could not access Kratos {args.solver_name} model part for stage dump.")
                    current_step, current_time = _process_info_step_time(KM, getattr(mp, "ProcessInfo", None))
                    if current_step == target_step and not bool(stage_dump_context.get("dumped", False)):
                        _dump_model_part_state(
                            mp,
                            run_dir / f"{output_stem}.npz",
                            solver_name=str(args.solver_name),
                            step=current_step,
                            time_s=current_time,
                            stage=target_stage,
                            iteration=iteration,
                        )
                        stage_dump_context["dumped"] = True
                        stage_dump_context["step"] = current_step
                        stage_dump_context["time_s"] = current_time
                        raise _StageDumpComplete()
                solver.SolveSolutionStep()
                if target_stage == f"after_solve_{solver_name}" and iteration == target_iteration and solver_name == str(args.solver_name):
                    mp = _try_get_model_part(solver)
                    if mp is None:
                        raise RuntimeError(f"Could not access Kratos {args.solver_name} model part for stage dump.")
                    current_step, current_time = _process_info_step_time(KM, getattr(mp, "ProcessInfo", None))
                    if current_step == target_step and not bool(stage_dump_context.get("dumped", False)):
                        _dump_model_part_state(
                            mp,
                            run_dir / f"{output_stem}.npz",
                            solver_name=str(args.solver_name),
                            step=current_step,
                            time_s=current_time,
                            stage=target_stage,
                            iteration=iteration,
                        )
                        stage_dump_context["dumped"] = True
                        stage_dump_context["step"] = current_step
                        stage_dump_context["time_s"] = current_time
                        raise _StageDumpComplete()
                self._SynchronizeOutputData(solver_name)
                if target_stage == f"after_sync_output_{solver_name}" and iteration == target_iteration and solver_name == str(args.solver_name):
                    mp = _try_get_model_part(solver)
                    if mp is None:
                        raise RuntimeError(f"Could not access Kratos {args.solver_name} model part for stage dump.")
                    current_step, current_time = _process_info_step_time(KM, getattr(mp, "ProcessInfo", None))
                    if current_step == target_step and not bool(stage_dump_context.get("dumped", False)):
                        _dump_model_part_state(
                            mp,
                            run_dir / f"{output_stem}.npz",
                            solver_name=str(args.solver_name),
                            step=current_step,
                            time_s=current_time,
                            stage=target_stage,
                            iteration=iteration,
                        )
                        stage_dump_context["dumped"] = True
                        stage_dump_context["step"] = current_step
                        stage_dump_context["time_s"] = current_time
                        raise _StageDumpComplete()

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
    cls._pycutfem_stage_state_dump_installed = True

    cwd = Path.cwd()
    try:
        os.chdir(run_dir)
        with (run_dir / "DoubleFlap_fsi_parameters_ROM.json").open("r", encoding="utf-8") as f:
            params = KM.Parameters(f.read())
        # reduce noise
        params["problem_data"]["echo_level"].SetInt(int(args.echo_level))
        co_sim = analysis.CoSimulationAnalysis(params)
        try:
            co_sim.Run()
        except _StageDumpComplete:
            pass
    finally:
        os.chdir(cwd)
        cls.SolveSolutionStep = original
        cls._pycutfem_stage_state_dump_installed = False
        if original_ale is not None:
            from KratosMultiphysics.MeshMovingApplication.ale_fluid_solver import AleFluidSolver

            AleFluidSolver.SolveSolutionStep = original_ale

    npz_path = run_dir / f"{output_stem}.npz"
    if not npz_path.exists():
        raise RuntimeError(f"Requested stage dump was not produced: {npz_path}")
    summary = {
        "run_dir": str(run_dir),
        "solver_name": str(args.solver_name),
        "stage": target_stage,
        "target_step": target_step,
        "iteration": target_iteration,
        "actual_step": None if stage_dump_context.get("step") is None else int(stage_dump_context["step"]),
        "time_s": None if stage_dump_context.get("time_s") is None else float(stage_dump_context["time_s"]),
        "npz_path": str(npz_path),
    }
    (run_dir / f"{output_stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
