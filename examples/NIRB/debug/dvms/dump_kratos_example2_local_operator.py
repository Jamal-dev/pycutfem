from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from examples.NIRB.double_flap_reference import _load_json, default_double_flap_root
from examples.NIRB.debug.dump_kratos_example2_stage_state import _try_get_model_part
from examples.NIRB.debug.dvms.dump_kratos_local_operator import _dump_local_operator
from examples.NIRB.debug.run_kratos_example2_reference import (
    _copy_inputs,
    _prepare_coupling_json,
    _prepare_fluid_json,
    _prepare_solid_json,
    _write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run coupled Kratos DoubleFlap and dump one fluid local operator/state at a chosen coupling stage."
    )
    parser.add_argument("--benchmark-root", type=Path, default=default_double_flap_root())
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("examples/NIRB/artifacts/kratos_example2_local_operator_stage"),
    )
    parser.add_argument("--end-time", type=float, default=0.008)
    parser.add_argument("--echo-level", type=int, default=1)
    parser.add_argument("--fluid-echo-level", type=int, default=0)
    parser.add_argument("--solid-echo-level", type=int, default=0)
    parser.add_argument(
        "--target-stage",
        choices=("after_ale_boundary_before_fluid", "after_solve_fluid", "after_sync_output_fluid"),
        default="after_sync_output_fluid",
    )
    parser.add_argument("--target-iteration", type=int, default=6)
    parser.add_argument("--element-id", type=int, default=None)
    parser.add_argument("--node-id", type=int, default=None)
    parser.add_argument("--interface-part", type=str, default="NoSlip2D_Interface")
    parser.add_argument("--output-stem", type=str, default="fluid_local_element_stage")
    parser.add_argument(
        "--mode",
        choices=("system", "velocity", "all"),
        default="all",
        help="Kratos local contribution to dump. 'all' stores both CalculateLocalVelocityContribution and CalculateLocalSystem.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import KratosMultiphysics as KM
        import KratosMultiphysics.CoSimulationApplication as KratosCoSim
        import KratosMultiphysics.CoSimulationApplication.co_simulation_analysis as analysis
        import KratosMultiphysics.FluidDynamicsApplication as KFD
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
    target_iteration = int(args.target_iteration)
    output_stem = str(args.output_stem)

    cls = gauss_seidel_strong.GaussSeidelStrongCoupledSolver
    if getattr(cls, "_pycutfem_local_operator_dump_installed", False):
        raise RuntimeError("Local-operator dumper already installed in this Python process.")
    original = cls.SolveSolutionStep
    original_ale = None
    stage_dump_context: dict[str, int] = {"iteration": -1}

    def _dump_target_local_operator(*, solver, stage: str, iteration: int) -> None:
        model_part = _try_get_model_part(solver)
        if model_part is None:
            main_model_part = getattr(solver, "main_model_part", None)
            if main_model_part is not None:
                try:
                    model_part = main_model_part.GetSubModelPart("FluidParts_FluidPart")
                except Exception:
                    model_part = None
        if model_part is None:
            getter = getattr(solver, "GetComputingModelPart", None)
            if callable(getter):
                try:
                    model_part = getter()
                except Exception:
                    model_part = None
        if model_part is None:
            raise RuntimeError("Could not access Kratos fluid model part for local-operator dump.")
        root_model_part = model_part.GetRootModelPart()
        _dump_local_operator(
            KM=KM,
            KFD=KFD,
            root_model_part=root_model_part,
            model_part=model_part,
            scheme=None,
            output_stem=output_stem,
            run_dir=run_dir,
            element_id=args.element_id,
            node_id=args.node_id,
            interface_part_name=str(args.interface_part),
            mode=str(args.mode),
            stage=str(stage),
        )

    try:
        from KratosMultiphysics.MeshMovingApplication.ale_fluid_solver import AleFluidSolver

        original_ale = AleFluidSolver.SolveSolutionStep

        def _wrapped_ale(self):
            is_converged = True
            for mesh_solver in self.mesh_motion_solvers:
                is_converged &= mesh_solver.SolveSolutionStep()

            if self.fluid_solver.GetComputingModelPart().ProcessInfo[KM.TIME] >= self.start_fluid_solution_time:
                self._AleFluidSolver__ApplyALEBoundaryCondition()
                coupling_iteration = int(stage_dump_context.get("iteration", -1))
                if target_stage == "after_ale_boundary_before_fluid" and coupling_iteration == target_iteration:
                    _dump_target_local_operator(
                        solver=self.fluid_solver,
                        stage=target_stage,
                        iteration=coupling_iteration,
                    )
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

            for coupling_op in self.coupling_operations_dict.values():
                coupling_op.InitializeCouplingIteration()
            for conv_acc in self.convergence_accelerators_list:
                conv_acc.InitializeNonLinearIteration()
            for conv_crit in self.convergence_criteria_list:
                conv_crit.InitializeNonLinearIteration()

            for solver_name, solver in self.solver_wrappers.items():
                self._SynchronizeInputData(solver_name)
                solver.SolveSolutionStep()
                if (
                    solver_name == "fluid"
                    and iteration == target_iteration
                    and target_stage == f"after_solve_{solver_name}"
                ):
                    _dump_target_local_operator(
                        solver=solver,
                        stage=target_stage,
                        iteration=iteration,
                    )
                self._SynchronizeOutputData(solver_name)
                if (
                    solver_name == "fluid"
                    and iteration == target_iteration
                    and target_stage == f"after_sync_output_{solver_name}"
                ):
                    _dump_target_local_operator(
                        solver=solver,
                        stage=target_stage,
                        iteration=iteration,
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
    cls._pycutfem_local_operator_dump_installed = True

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
        cls._pycutfem_local_operator_dump_installed = False
        if original_ale is not None:
            from KratosMultiphysics.MeshMovingApplication.ale_fluid_solver import AleFluidSolver

            AleFluidSolver.SolveSolutionStep = original_ale

    npz_path = run_dir / f"{output_stem}.npz"
    if not npz_path.exists():
        raise RuntimeError(f"Requested local-operator dump was not produced: {npz_path}")
    summary = {
        "run_dir": str(run_dir),
        "stage": target_stage,
        "iteration": target_iteration,
        "npz_path": str(npz_path),
    }
    (run_dir / f"{output_stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
