from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from examples.NIRB.double_flap_reference import _load_json, default_double_flap_root
from examples.NIRB.run_kratos_example2_reference import _copy_inputs, _prepare_fluid_json, _write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Kratos fluid-only DoubleFlap step-1 case and dump the assembled global fluid system."
    )
    parser.add_argument("--benchmark-root", type=Path, default=default_double_flap_root())
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("examples/NIRB/artifacts/kratos_fluid_global_system"),
    )
    parser.add_argument("--end-time", type=float, default=0.008)
    parser.add_argument("--echo-level", type=int, default=0)
    parser.add_argument(
        "--stage",
        choices=("predicted", "solved"),
        default="predicted",
        help="Assemble the global system before SolveSolutionStep() ('predicted') or after the solved first step ('solved').",
    )
    parser.add_argument("--output-stem", type=str, default="fluid_global_system")
    return parser.parse_args()


def _pack_csr(matrix_csr) -> dict[str, np.ndarray]:
    return {
        "data": np.asarray(matrix_csr.data, dtype=float),
        "indices": np.asarray(matrix_csr.indices, dtype=np.int32),
        "indptr": np.asarray(matrix_csr.indptr, dtype=np.int32),
        "shape": np.asarray(matrix_csr.shape, dtype=np.int32),
    }


def _ordered_dof_metadata(KM, model_part, ndof: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_ids = np.full((ndof,), -1, dtype=np.int32)
    is_fixed = np.zeros((ndof,), dtype=bool)
    var_names = np.full((ndof,), "", dtype=object)
    values = np.zeros((ndof,), dtype=float)
    vars_to_dump = (
        (KM.VELOCITY_X, "VELOCITY_X"),
        (KM.VELOCITY_Y, "VELOCITY_Y"),
        (KM.PRESSURE, "PRESSURE"),
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


def main() -> None:
    args = parse_args()

    try:
        import KratosMultiphysics as KM
        from KratosMultiphysics import scipy_conversion_tools
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
        echo_level=int(args.echo_level),
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

        fluid_solver = solver.fluid_solver
        strategy = fluid_solver._GetSolutionStrategy()
        builder = fluid_solver._GetBuilderAndSolver()
        scheme = fluid_solver._GetScheme()
        computing_model_part = fluid_solver.GetComputingModelPart()
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

        root_model_part = model["FluidModelPart.FluidParts_FluidPart"]
        nodes = list(root_model_part.Nodes)
        node_ids = np.asarray([int(node.Id) for node in nodes], dtype=int)
        node_coords_ref = np.asarray([[float(node.X0), float(node.Y0)] for node in nodes], dtype=float)
        node_coords_cur = np.asarray([[float(node.X), float(node.Y)] for node in nodes], dtype=float)
        velocity = np.asarray(
            [[float(node.GetSolutionStepValue(KM.VELOCITY)[0]), float(node.GetSolutionStepValue(KM.VELOCITY)[1])] for node in nodes],
            dtype=float,
        )
        velocity_prev = np.asarray(
            [[float(node.GetSolutionStepValue(KM.VELOCITY, 1)[0]), float(node.GetSolutionStepValue(KM.VELOCITY, 1)[1])] for node in nodes],
            dtype=float,
        )
        acceleration = np.asarray(
            [[float(node.GetSolutionStepValue(KM.ACCELERATION)[0]), float(node.GetSolutionStepValue(KM.ACCELERATION)[1])] for node in nodes],
            dtype=float,
        )
        mesh_velocity = np.asarray(
            [[float(node.GetSolutionStepValue(KM.MESH_VELOCITY)[0]), float(node.GetSolutionStepValue(KM.MESH_VELOCITY)[1])] for node in nodes],
            dtype=float,
        )
        pressure = np.asarray([float(node.GetSolutionStepValue(KM.PRESSURE)) for node in nodes], dtype=float)
        reaction = np.asarray(
            [[float(node.GetSolutionStepValue(KM.REACTION)[0]), float(node.GetSolutionStepValue(KM.REACTION)[1])] for node in nodes],
            dtype=float,
        )
        advproj = np.asarray(
            [[float(node.GetSolutionStepValue(KM.ADVPROJ)[0]), float(node.GetSolutionStepValue(KM.ADVPROJ)[1])] for node in nodes],
            dtype=float,
        )
        divproj = np.asarray([float(node.GetSolutionStepValue(KM.DIVPROJ)) for node in nodes], dtype=float)

        eq_node_ids, eq_var_names, eq_is_fixed, eq_values = _ordered_dof_metadata(KM, computing_model_part, int(A_raw.shape[0]))

        npz_path = run_dir / f"{args.output_stem}_{args.stage}.npz"
        np.savez(
            npz_path,
            stage=str(args.stage),
            node_ids=node_ids,
            node_coords_ref=node_coords_ref,
            node_coords_cur=node_coords_cur,
            velocity=velocity,
            velocity_prev=velocity_prev,
            acceleration=acceleration,
            mesh_velocity=mesh_velocity,
            pressure=pressure,
            reaction=reaction,
            advproj=advproj,
            divproj=divproj,
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

        summary = {
            "run_dir": str(run_dir),
            "npz_path": str(npz_path),
            "stage": str(args.stage),
            "system_size": int(A_raw.shape[0]),
            "raw_matrix_inf_norm": float(np.max(np.abs(A_raw.data))) if A_raw.nnz else 0.0,
            "raw_rhs_inf_norm": float(np.max(np.abs(b_raw))) if b_raw.size else 0.0,
            "constrained_matrix_inf_norm": float(np.max(np.abs(A_constrained.data))) if A_constrained.nnz else 0.0,
            "constrained_rhs_inf_norm": float(np.max(np.abs(b_constrained))) if b_constrained.size else 0.0,
        }
        json_path = run_dir / f"{args.output_stem}_{args.stage}.json"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        analysis.FinalizeSolutionStep()
        analysis.OutputSolutionStep()
        analysis.Finalize()
        print(npz_path)
        print(json_path)
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
