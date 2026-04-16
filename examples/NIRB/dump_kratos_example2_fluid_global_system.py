from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from examples.NIRB.double_flap_reference import _load_json, default_double_flap_root
from examples.NIRB.dvms.dump_kratos_fluid_global_system import _ordered_dof_metadata
from examples.NIRB.run_kratos_example2_reference import (
    _copy_inputs,
    _model_part_state_payload,
    _prepare_coupling_json,
    _prepare_fluid_json,
    _prepare_solid_json,
    _write_json,
)


def _pack_csr(matrix_csr) -> dict[str, np.ndarray]:
    return {
        "data": np.asarray(matrix_csr.data, dtype=float),
        "indices": np.asarray(matrix_csr.indices, dtype=np.int32),
        "indptr": np.asarray(matrix_csr.indptr, dtype=np.int32),
        "shape": np.asarray(matrix_csr.shape, dtype=np.int32),
    }


def _dump_fluid_global_system(
    *,
    KM,
    scipy_conversion_tools,
    fluid_solver,
    output_path: Path,
    stage: str,
    iteration: int,
) -> None:
    strategy = fluid_solver._GetSolutionStrategy()
    builder = fluid_solver._GetBuilderAndSolver()
    scheme = fluid_solver._GetScheme()
    computing_model_part = fluid_solver.GetComputingModelPart()
    space = KM.UblasSparseSpace()

    A = strategy.GetSystemMatrix()
    b = strategy.GetSystemVector()
    x = strategy.GetSolutionVector()
    ndof = int(A.Size1()) if hasattr(A, "Size1") else int(len(x))
    for node in computing_model_part.Nodes:
        for variable in (KM.VELOCITY_X, KM.VELOCITY_Y, KM.PRESSURE):
            if not node.HasDofFor(variable):
                continue
            dof = node.GetDof(variable)
            eq_id = int(dof.EquationId)
            if eq_id < 0 or eq_id >= ndof:
                continue
            try:
                x[eq_id] = float(dof.GetSolutionStepValue())
            except Exception:
                x[eq_id] = 0.0

    space.SetToZeroMatrix(A)
    space.SetToZeroVector(b)
    builder.Build(scheme, computing_model_part, A, b)
    A_raw = scipy_conversion_tools.to_csr(A).copy()
    b_raw = np.asarray(b, dtype=float).copy()

    builder.ApplyDirichletConditions(scheme, computing_model_part, A, x, b)
    A_constrained = scipy_conversion_tools.to_csr(A).copy()
    b_constrained = np.asarray(b, dtype=float).copy()

    eq_node_ids, eq_var_names, eq_is_fixed, eq_values = _ordered_dof_metadata(
        KM,
        computing_model_part,
        int(A_raw.shape[0]),
    )

    main_model_part = getattr(fluid_solver, "main_model_part", None)
    volume_model_part = None
    if main_model_part is not None:
        try:
            volume_model_part = main_model_part.GetSubModelPart("FluidParts_FluidPart")
        except Exception:
            try:
                volume_model_part = main_model_part.GetModel().GetModelPart("FluidModelPart.FluidParts_FluidPart")
            except Exception:
                volume_model_part = None
    if volume_model_part is None:
        raise RuntimeError("Could not resolve the Kratos fluid volume model part for the global-system dump.")

    payload = _model_part_state_payload(volume_model_part, solver_name="fluid")
    slim_payload = {
        str(key)[len("fluid_") :]: np.asarray(value)
        for key, value in payload.items()
        if str(key).startswith("fluid_")
    }
    slim_payload.update(
        {
            "stage": np.asarray(str(stage)),
            "iteration": np.asarray(int(iteration), dtype=int),
            "equation_node_ids": np.asarray(eq_node_ids, dtype=np.int32),
            "equation_var_names": np.asarray(eq_var_names, dtype=str),
            "equation_is_fixed": np.asarray(eq_is_fixed, dtype=bool),
            "equation_values": np.asarray(eq_values, dtype=float),
            "A_raw_data": _pack_csr(A_raw)["data"],
            "A_raw_indices": _pack_csr(A_raw)["indices"],
            "A_raw_indptr": _pack_csr(A_raw)["indptr"],
            "A_raw_shape": _pack_csr(A_raw)["shape"],
            "b_raw": np.asarray(b_raw, dtype=float),
            "A_constrained_data": _pack_csr(A_constrained)["data"],
            "A_constrained_indices": _pack_csr(A_constrained)["indices"],
            "A_constrained_indptr": _pack_csr(A_constrained)["indptr"],
            "A_constrained_shape": _pack_csr(A_constrained)["shape"],
            "b_constrained": np.asarray(b_constrained, dtype=float),
        }
    )
    np.savez(output_path, **slim_payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run coupled Kratos DoubleFlap and dump the fluid global system at a chosen coupling stage."
    )
    parser.add_argument("--benchmark-root", type=Path, default=default_double_flap_root())
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("examples/NIRB/artifacts/kratos_example2_fluid_global_system"),
    )
    parser.add_argument("--end-time", type=float, default=0.008)
    parser.add_argument("--echo-level", type=int, default=1)
    parser.add_argument("--fluid-echo-level", type=int, default=0)
    parser.add_argument("--solid-echo-level", type=int, default=0)
    parser.add_argument(
        "--target-stage",
        choices=("after_ale_boundary_before_fluid", "after_solve_fluid", "after_sync_output_fluid"),
        default="after_ale_boundary_before_fluid",
    )
    parser.add_argument("--target-iteration", type=int, default=5)
    parser.add_argument("--output-stem", type=str, default="fluid_global_system_stage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import KratosMultiphysics as KM
        from KratosMultiphysics import scipy_conversion_tools
        import KratosMultiphysics.CoSimulationApplication as KratosCoSim
        import KratosMultiphysics.CoSimulationApplication.co_simulation_analysis as analysis
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
    output_path = run_dir / f"{str(args.output_stem)}.npz"

    cls = gauss_seidel_strong.GaussSeidelStrongCoupledSolver
    if getattr(cls, "_pycutfem_fluid_global_dump_installed", False):
        raise RuntimeError("Fluid global-system dumper already installed in this Python process.")
    original = cls.SolveSolutionStep
    original_ale = None
    stage_dump_context: dict[str, int] = {"iteration": -1}

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
                    _dump_fluid_global_system(
                        KM=KM,
                        scipy_conversion_tools=scipy_conversion_tools,
                        fluid_solver=self.fluid_solver,
                        output_path=output_path,
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
                if solver_name == "fluid" and target_stage == "after_solve_fluid" and iteration == target_iteration:
                    fluid_solver = solver._analysis_stage._GetSolver().fluid_solver  # noqa: SLF001
                    _dump_fluid_global_system(
                        KM=KM,
                        scipy_conversion_tools=scipy_conversion_tools,
                        fluid_solver=fluid_solver,
                        output_path=output_path,
                        stage=target_stage,
                        iteration=iteration,
                    )
                self._SynchronizeOutputData(solver_name)
                if solver_name == "fluid" and target_stage == "after_sync_output_fluid" and iteration == target_iteration:
                    fluid_solver = solver._analysis_stage._GetSolver().fluid_solver  # noqa: SLF001
                    _dump_fluid_global_system(
                        KM=KM,
                        scipy_conversion_tools=scipy_conversion_tools,
                        fluid_solver=fluid_solver,
                        output_path=output_path,
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
    cls._pycutfem_fluid_global_dump_installed = True

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
        cls._pycutfem_fluid_global_dump_installed = False
        if original_ale is not None:
            from KratosMultiphysics.MeshMovingApplication.ale_fluid_solver import AleFluidSolver

            AleFluidSolver.SolveSolutionStep = original_ale

    if not output_path.exists():
        raise RuntimeError(f"Requested fluid global-system dump was not produced: {output_path}")
    summary = {
        "run_dir": str(run_dir),
        "stage": target_stage,
        "iteration": target_iteration,
        "npz_path": str(output_path),
    }
    (run_dir / f"{str(args.output_stem)}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
