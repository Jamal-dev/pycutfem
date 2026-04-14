from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from examples.NIRB.double_flap_reference import _load_json, default_double_flap_root
from examples.NIRB.run_kratos_example2_reference import (
    _copy_inputs,
    _prepare_coupling_json,
    _prepare_fluid_json,
    _prepare_solid_json,
    _write_json,
)


def _try_get_fluid_model_part(solver_wrapper) -> Any | None:
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


def _dump_model_part_state(model_part, output_path: Path) -> None:
    import KratosMultiphysics as KM

    nodes = list(model_part.Nodes)
    node_ids = np.asarray([int(node.Id) for node in nodes], dtype=int)
    node_coords_ref = np.asarray([[float(node.X0), float(node.Y0)] for node in nodes], dtype=float)
    node_coords_cur = np.asarray([[float(node.X), float(node.Y)] for node in nodes], dtype=float)
    velocity = np.asarray(
        [[float(node.GetSolutionStepValue(KM.VELOCITY)[0]), float(node.GetSolutionStepValue(KM.VELOCITY)[1])] for node in nodes],
        dtype=float,
    )
    pressure = np.asarray([float(node.GetSolutionStepValue(KM.PRESSURE)) for node in nodes], dtype=float)
    reaction = np.asarray(
        [[float(node.GetSolutionStepValue(KM.REACTION)[0]), float(node.GetSolutionStepValue(KM.REACTION)[1])] for node in nodes],
        dtype=float,
    )
    mesh_displacement = np.asarray(
        [[float(node.GetSolutionStepValue(KM.MESH_DISPLACEMENT)[0]), float(node.GetSolutionStepValue(KM.MESH_DISPLACEMENT)[1])] for node in nodes],
        dtype=float,
    )
    mesh_velocity = np.asarray(
        [[float(node.GetSolutionStepValue(KM.MESH_VELOCITY)[0]), float(node.GetSolutionStepValue(KM.MESH_VELOCITY)[1])] for node in nodes],
        dtype=float,
    )
    np.savez(
        output_path,
        node_ids=node_ids,
        node_coords_ref=node_coords_ref,
        node_coords_cur=node_coords_cur,
        velocity=velocity,
        pressure=pressure,
        reaction=reaction,
        mesh_displacement=mesh_displacement,
        mesh_velocity=mesh_velocity,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run coupled Kratos DoubleFlap and dump full fluid nodal state at a chosen coupling stage.")
    parser.add_argument("--benchmark-root", type=Path, default=default_double_flap_root())
    parser.add_argument("--run-dir", type=Path, default=Path("examples/NIRB/artifacts/kratos_example2_stage_state"))
    parser.add_argument("--end-time", type=float, default=0.008)
    parser.add_argument("--echo-level", type=int, default=1)
    parser.add_argument("--fluid-echo-level", type=int, default=0)
    parser.add_argument("--solid-echo-level", type=int, default=0)
    parser.add_argument("--target-stage", type=str, default="after_sync_output_fluid")
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
    target_stage = str(args.target_stage)
    target_iteration = int(args.target_iteration)
    output_stem = str(args.output_stem)

    cls = gauss_seidel_strong.GaussSeidelStrongCoupledSolver
    if getattr(cls, "_pycutfem_stage_state_dump_installed", False):
        raise RuntimeError("Stage-state dumper already installed in this Python process.")
    original = cls.SolveSolutionStep

    def _wrapped(self):
        for k in range(self.num_coupling_iterations):
            self.process_info[KratosCoSim.COUPLING_ITERATION_NUMBER] += 1
            iteration = int(k + 1)

            for coupling_op in self.coupling_operations_dict.values():
                coupling_op.InitializeCouplingIteration()
            for conv_acc in self.convergence_accelerators_list:
                conv_acc.InitializeNonLinearIteration()
            for conv_crit in self.convergence_criteria_list:
                conv_crit.InitializeNonLinearIteration()

            for solver_name, solver in self.solver_wrappers.items():
                self._SynchronizeInputData(solver_name)
                if target_stage == f"after_sync_input_{solver_name}" and iteration == target_iteration and solver_name == "fluid":
                    mp = _try_get_fluid_model_part(solver)
                    if mp is None:
                        raise RuntimeError("Could not access Kratos fluid model part for stage dump.")
                    _dump_model_part_state(mp, run_dir / f"{output_stem}.npz")
                solver.SolveSolutionStep()
                if target_stage == f"after_solve_{solver_name}" and iteration == target_iteration and solver_name == "fluid":
                    mp = _try_get_fluid_model_part(solver)
                    if mp is None:
                        raise RuntimeError("Could not access Kratos fluid model part for stage dump.")
                    _dump_model_part_state(mp, run_dir / f"{output_stem}.npz")
                self._SynchronizeOutputData(solver_name)
                if target_stage == f"after_sync_output_{solver_name}" and iteration == target_iteration and solver_name == "fluid":
                    mp = _try_get_fluid_model_part(solver)
                    if mp is None:
                        raise RuntimeError("Could not access Kratos fluid model part for stage dump.")
                    _dump_model_part_state(mp, run_dir / f"{output_stem}.npz")

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
        co_sim.Run()
    finally:
        os.chdir(cwd)
        cls.SolveSolutionStep = original
        cls._pycutfem_stage_state_dump_installed = False

    npz_path = run_dir / f"{output_stem}.npz"
    if not npz_path.exists():
        raise RuntimeError(f"Requested stage dump was not produced: {npz_path}")
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
