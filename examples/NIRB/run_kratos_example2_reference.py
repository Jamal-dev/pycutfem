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
    ) -> None:
        self.monitor_dir = Path(monitor_dir)
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        self.solver_names = tuple(str(name) for name in solver_names)
        self.field_names = tuple(str(name) for name in field_names)
        self.stage_filters = tuple(str(name) for name in stage_filters) if stage_filters else None
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
        payload: dict[str, Any] = {
            "stage": str(stage),
            "iteration": int(iteration),
            "time": float(coupled_solver.process_info[KM.TIME]) if coupled_solver.process_info.Has(KM.TIME) else 0.0,
            "step": int(coupled_solver.process_info[KM.STEP]) if coupled_solver.process_info.Has(KM.STEP) else 0,
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
        for solver_name in self.solver_names:
            solver = coupled_solver.solver_wrappers.get(solver_name)
            if solver is None or not getattr(solver, "IsDefinedOnThisRank", lambda: False)():
                continue
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
                if hasattr(conv_acc, "X"):
                    acc_summary[f"{label}_buffer_X_len"] = int(len(conv_acc.X))
            monitor.dump(self, stage="post_update", iteration=iteration, extra=acc_summary)

    cls.SolveSolutionStep = _monitored_solve_solution_step
    cls._pycutfem_monitor_installed = True


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
        "--monitor-stages",
        type=str,
        default=None,
        help="Optional comma-separated stage filter. Example: after_solve_fluid,after_sync_output_fluid,post_update",
    )
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
    if bool(args.monitor_coupling):
        monitor_dir = Path(args.monitor_dir) if args.monitor_dir is not None else run_dir / "coupling_monitor"
        monitor = KratosCouplingMonitor(
            monitor_dir=monitor_dir,
            solver_names=_split_csv_values(args.monitor_solvers) or ("fluid", "structure"),
            field_names=_split_csv_values(args.monitor_fields) or ("load", "disp", "velocity"),
            stage_filters=_split_csv_values(args.monitor_stages),
        )
        _install_kratos_coupling_monitor(monitor=monitor)

    with (run_dir / "DoubleFlap_fsi_parameters_ROM.json").open("r", encoding="utf-8") as f:
        parameters = KM.Parameters(f.read())

    with contextlib.chdir(run_dir):
        simulation = CoSimulationAnalysis(parameters)
        try:
            simulation.Run()
        finally:
            if monitor is not None:
                monitor.finalize()

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
        "monitor_stages": list(_split_csv_values(args.monitor_stages) or []),
    }
    _write_json(run_dir / "summary.json", summary)
    print(f"summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
