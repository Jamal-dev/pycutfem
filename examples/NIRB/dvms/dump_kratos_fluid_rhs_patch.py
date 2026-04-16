from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from examples.NIRB.dvms.dump_kratos_fluid_global_system import _ordered_dof_metadata
from examples.NIRB.dvms.dump_kratos_local_operator import _apply_state_npz_to_model, _load_state_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a saved Kratos fluid state, activate only the element/condition patch "
            "touching a target node, and dump the builder-level RHS/LHS contribution."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--state-npz", type=Path, required=True)
    parser.add_argument("--node-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.008)
    parser.add_argument("--echo-level", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import KratosMultiphysics as KM
        from KratosMultiphysics import scipy_conversion_tools
        from KratosMultiphysics.FluidDynamicsApplication.fluid_dynamics_analysis import FluidDynamicsAnalysis
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("KratosMultiphysics is not importable in the current Python environment.") from exc

    run_dir = Path(args.run_dir).resolve()
    state_npz = Path(args.state_npz).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cwd = Path.cwd()
    os.chdir(run_dir)
    try:
        with Path("ProjectParametersCFD.json").open("r", encoding="utf-8") as f:
            params = KM.Parameters(f.read())
        if params.Has("problem_data") and params["problem_data"].Has("echo_level"):
            params["problem_data"]["echo_level"].SetInt(int(args.echo_level))

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
        fluid_solver = solver.fluid_solver
        strategy = fluid_solver._GetSolutionStrategy()
        builder = fluid_solver._GetBuilderAndSolver()
        scheme = fluid_solver._GetScheme()
        computing_model_part = fluid_solver.GetComputingModelPart()
        root_model_part = model["FluidModelPart.FluidParts_FluidPart"]

        state = _load_state_npz(state_npz)
        _apply_state_npz_to_model(KM=KM, model_part=root_model_part, state=state, dt=float(args.dt))

        target_node_id = int(args.node_id)
        target_node = root_model_part.Nodes[target_node_id]
        target_eq_id = int(target_node.GetDof(KM.VELOCITY_X).EquationId)

        elem_ids = sorted(
            int(elem.Id)
            for elem in computing_model_part.Elements
            if any(int(node.Id) == target_node_id for node in elem.GetGeometry())
        )
        cond_ids = sorted(
            int(cond.Id)
            for cond in computing_model_part.Conditions
            if any(int(node.Id) == target_node_id for node in cond.GetGeometry())
        )
        node_ids = sorted(
            {
                int(node.Id)
                for eid in elem_ids
                for node in computing_model_part.Elements[eid].GetGeometry()
            }
            | {
                int(node.Id)
                for cid in cond_ids
                for node in computing_model_part.Conditions[cid].GetGeometry()
            }
        )

        for elem in computing_model_part.Elements:
            elem.Set(KM.ACTIVE, int(elem.Id) in elem_ids)
        for cond in computing_model_part.Conditions:
            cond.Set(KM.ACTIVE, int(cond.Id) in cond_ids)

        space = KM.UblasSparseSpace()
        A = strategy.GetSystemMatrix()
        b = strategy.GetSystemVector()
        space.SetToZeroMatrix(A)
        space.SetToZeroVector(b)
        builder.BuildLHS(scheme, computing_model_part, A)
        builder.BuildRHS(scheme, computing_model_part, b)

        A_patch = scipy_conversion_tools.to_csr(A).copy()
        b_patch = np.asarray(b, dtype=float).copy()
        eq_node_ids, eq_var_names, eq_is_fixed, _ = _ordered_dof_metadata(KM, computing_model_part, int(b_patch.shape[0]))

        with np.load(state_npz, allow_pickle=False) as data:
            b_global = np.asarray(data["b_raw"], dtype=float)

        report = {
            "run_dir": str(run_dir),
            "state_npz": str(state_npz),
            "target_node_id": target_node_id,
            "target_equation_id": target_eq_id,
            "patch_element_ids": elem_ids,
            "patch_condition_ids": cond_ids,
            "patch_node_ids": node_ids,
            "builder_methods": [name for name in dir(builder) if "Build" in name or "Apply" in name],
            "target_row": {
                "node_id": int(eq_node_ids[target_eq_id]),
                "variable": str(eq_var_names[target_eq_id]),
                "is_fixed": bool(eq_is_fixed[target_eq_id]),
                "rhs_patch": float(b_patch[target_eq_id]),
                "rhs_global": float(b_global[target_eq_id]),
                "rhs_diff": float(b_patch[target_eq_id] - b_global[target_eq_id]),
                "matrix_row_patch_inf": float(np.max(np.abs(A_patch.getrow(target_eq_id).data))) if A_patch.getrow(target_eq_id).nnz else 0.0,
            },
        }
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(output_path)
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
