from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {str(key): np.asarray(data[key]) for key in data.files}


def _safe_vec2(values: np.ndarray) -> Any:
    import KratosMultiphysics as KM

    vec = KM.Array3()
    vec[0] = float(values[0])
    vec[1] = float(values[1])
    vec[2] = 0.0
    return vec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump one Kratos structural local operator from a saved Example 2 structure stage state."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("examples/NIRB/artifacts/kratos_example2_structure_stage_state_iter5_20260415"),
        help="Directory containing ProjectParametersCSM.json, StructuralMaterials.json, and Double_Flap_Mesh/.",
    )
    parser.add_argument(
        "--stage-state",
        type=Path,
        default=Path("examples/NIRB/artifacts/kratos_example2_structure_stage_state_iter5_20260415/structure_stage_state_iter5.npz"),
    )
    parser.add_argument("--element-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    stage_state = _load_npz(Path(args.stage_state).resolve())

    import KratosMultiphysics as KM
    import KratosMultiphysics.StructuralMechanicsApplication as KSM
    from KratosMultiphysics.StructuralMechanicsApplication import python_solvers_wrapper_structural as struct_wrapper

    settings_data = json.loads((run_dir / "ProjectParametersCSM.json").read_text(encoding="utf-8"))
    solver_settings = KM.Parameters(json.dumps(settings_data["solver_settings"]))
    solver_settings["echo_level"].SetInt(0)

    model = KM.Model()
    solver = struct_wrapper.CreateSolverByParameters(model, solver_settings, "OpenMP")
    cwd = Path.cwd()
    try:
        os.chdir(run_dir)
        solver.AddVariables()
        solver.ImportModelPart()
        solver.PrepareModelPart()
        solver.AddDofs()
        solver.Initialize()
    finally:
        os.chdir(cwd)

    try:
        model_part = solver.GetComputingModelPart()
    except Exception:
        model_part = getattr(solver, "main_model_part", None)
    if model_part is None:
        raise RuntimeError("Could not access the Kratos structural computing model part.")

    ids = np.asarray(stage_state["node_ids"], dtype=int).reshape(-1)
    disp = np.asarray(stage_state["displacement_nodal_values"], dtype=float)
    vel = np.asarray(stage_state.get("velocity_nodal_values", np.zeros_like(disp)), dtype=float)
    point_load = np.asarray(stage_state.get("point_load_nodal_values", np.zeros_like(disp)), dtype=float)
    reaction = np.asarray(stage_state.get("reaction_nodal_values", np.zeros_like(disp)), dtype=float)
    coords_ref = np.asarray(stage_state["coords_ref"], dtype=float)
    coords_cur = coords_ref + disp
    by_id = {int(node_id): idx for idx, node_id in enumerate(ids)}

    for node in model_part.Nodes:
        idx = by_id.get(int(node.Id))
        if idx is None:
            continue
        node.X = float(coords_cur[idx, 0])
        node.Y = float(coords_cur[idx, 1])
        node.Z = 0.0
        node.SetSolutionStepValue(KM.DISPLACEMENT, 0, _safe_vec2(disp[idx]))
        try:
            node.SetSolutionStepValue(KM.VELOCITY, 0, _safe_vec2(vel[idx]))
        except Exception:
            pass
        try:
            node.SetSolutionStepValue(KM.REACTION, 0, _safe_vec2(reaction[idx]))
        except Exception:
            pass
        try:
            node.SetSolutionStepValue(KSM.POINT_LOAD, 0, _safe_vec2(point_load[idx]))
        except Exception:
            pass

    elem = model_part.GetElement(int(args.element_id))
    geom = elem.GetGeometry()
    lhs = KM.Matrix()
    rhs = KM.Vector()
    elem.CalculateLocalSystem(lhs, rhs, model_part.ProcessInfo)

    gp_green_lagrange = None
    gp_pk2_stress = None
    gp_cauchy_stress = None
    try:
        gp_green_lagrange = np.asarray(
            [np.asarray(v, dtype=float) for v in elem.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, model_part.ProcessInfo)],
            dtype=float,
        )
    except Exception:
        pass
    try:
        gp_pk2_stress = np.asarray(
            [np.asarray(v, dtype=float) for v in elem.CalculateOnIntegrationPoints(KM.PK2_STRESS_VECTOR, model_part.ProcessInfo)],
            dtype=float,
        )
    except Exception:
        pass
    try:
        gp_cauchy_stress = np.asarray(
            [np.asarray(v, dtype=float) for v in elem.CalculateOnIntegrationPoints(KM.CAUCHY_STRESS_VECTOR, model_part.ProcessInfo)],
            dtype=float,
        )
    except Exception:
        pass

    geom_node_ids = np.asarray([int(node.Id) for node in geom], dtype=int)
    geom_ref = np.asarray([[float(node.X0), float(node.Y0)] for node in geom], dtype=float)
    geom_cur = np.asarray([coords_cur[by_id[int(node.Id)]] for node in geom], dtype=float)
    geom_disp = np.asarray([disp[by_id[int(node.Id)]] for node in geom], dtype=float)
    geom_vel = np.asarray([vel[by_id[int(node.Id)]] for node in geom], dtype=float)
    geom_point_load = np.asarray([point_load[by_id[int(node.Id)]] for node in geom], dtype=float)

    payload = {
        "run_dir": str(run_dir),
        "stage_state": str(Path(args.stage_state).resolve()),
        "element_id": int(args.element_id),
        "node_ids": geom_node_ids,
        "coords_ref": geom_ref,
        "coords_cur": geom_cur,
        "displacement": geom_disp,
        "velocity": geom_vel,
        "point_load": geom_point_load,
        "local_lhs": np.asarray(lhs, dtype=float),
        "local_rhs": np.asarray(rhs, dtype=float),
    }
    if gp_green_lagrange is not None:
        payload["gp_green_lagrange_strain"] = gp_green_lagrange
    if gp_pk2_stress is not None:
        payload["gp_pk2_stress"] = gp_pk2_stress
    if gp_cauchy_stress is not None:
        payload["gp_cauchy_stress"] = gp_cauchy_stress

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **payload)
    print(output)


if __name__ == "__main__":
    main()
