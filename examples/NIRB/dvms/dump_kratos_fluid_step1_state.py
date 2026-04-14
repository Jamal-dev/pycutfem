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
        description="Run the Kratos fluid-only DoubleFlap step-1 case and dump full nodal and DVMS quadrature state."
    )
    parser.add_argument("--benchmark-root", type=Path, default=default_double_flap_root())
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("examples/NIRB/artifacts/kratos_fluid_step1_state"),
    )
    parser.add_argument("--end-time", type=float, default=0.008)
    parser.add_argument("--echo-level", type=int, default=0)
    parser.add_argument("--output-stem", type=str, default="fluid_step1_state")
    parser.add_argument("--nodal-only", action="store_true", help="Dump only nodal fields and skip per-element quadrature data.")
    parser.add_argument("--interface-only", action="store_true", help="Dump only fluid elements touching the FSI interface.")
    parser.add_argument("--interface-part", type=str, default="NoSlip2D_Interface")
    return parser.parse_args()


def _integration_point_coords(geom, integration_method) -> tuple[np.ndarray, np.ndarray]:
    points = list(geom.IntegrationPoints(integration_method))
    n_qp = len(points)
    dim = 2
    q_coords_ref = np.zeros((n_qp, dim), dtype=float)
    q_coords_cur = np.zeros((n_qp, dim), dtype=float)
    q_weights = np.zeros((n_qp,), dtype=float)
    shape_values = np.asarray(geom.ShapeFunctionsValues(integration_method), dtype=float)
    x0 = np.asarray([[float(node.X0), float(node.Y0)] for node in geom], dtype=float)
    x = np.asarray([[float(node.X), float(node.Y)] for node in geom], dtype=float)
    for q, gp in enumerate(points):
        q_weights[q] = float(gp.Weight())
        N = np.asarray(shape_values[q], dtype=float).reshape(-1)
        q_coords_ref[q, :] = N @ x0
        q_coords_cur[q, :] = N @ x
    return q_coords_ref, q_coords_cur, q_weights


def main() -> None:
    args = parse_args()

    try:
        import KratosMultiphysics as KM
        import KratosMultiphysics.FluidDynamicsApplication as KFD
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
        analysis.Run()
        model_part = model["FluidModelPart.FluidParts_FluidPart"]

        nodes = list(model_part.Nodes)
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
        reaction = np.asarray(
            [[float(node.GetSolutionStepValue(KM.REACTION)[0]), float(node.GetSolutionStepValue(KM.REACTION)[1])] for node in nodes],
            dtype=float,
        )
        pressure = np.asarray([float(node.GetSolutionStepValue(KM.PRESSURE)) for node in nodes], dtype=float)
        advproj = np.asarray(
            [[float(node.GetSolutionStepValue(KM.ADVPROJ)[0]), float(node.GetSolutionStepValue(KM.ADVPROJ)[1])] for node in nodes],
            dtype=float,
        )
        divproj = np.asarray([float(node.GetSolutionStepValue(KM.DIVPROJ)) for node in nodes], dtype=float)

        npz_payload = {
            "node_ids": node_ids,
            "node_coords_ref": node_coords_ref,
            "node_coords_cur": node_coords_cur,
            "velocity": velocity,
            "velocity_prev": velocity_prev,
            "acceleration": acceleration,
            "mesh_velocity": mesh_velocity,
            "reaction": reaction,
            "pressure": pressure,
            "advproj": advproj,
            "divproj": divproj,
        }

        q_coords_ref = None
        q_coords_cur = None
        q_weights = None
        subscale_velocity = None
        subscale_pressure = None
        node_ids_by_elem = np.zeros((0, 3), dtype=int)
        if not bool(args.nodal_only):
            interface_node_ids: set[int] = set()
            if bool(args.interface_only):
                root_model_part = model["FluidModelPart"]
                interface_node_ids = {
                    int(node.Id) for node in root_model_part.GetSubModelPart(str(args.interface_part)).Nodes
                }

            elements_all = list(model_part.Elements)
            if interface_node_ids:
                elements = [
                    elem
                    for elem in elements_all
                    if any(int(node.Id) in interface_node_ids for node in elem.GetGeometry())
                ]
            else:
                elements = elements_all
            n_elem = len(elements)
            node_ids_by_elem = np.zeros((n_elem, 3), dtype=int)

            for eidx, elem in enumerate(elements):
                geom = elem.GetGeometry()
                node_ids_by_elem[eidx, :] = np.asarray([int(node.Id) for node in geom], dtype=int)
                integration_method = elem.GetIntegrationMethod()
                elem_q_coords_ref, elem_q_coords_cur, elem_q_weights = _integration_point_coords(geom, integration_method)
                elem_subscale_velocity = np.asarray(
                    elem.CalculateOnIntegrationPoints(KFD.SUBSCALE_VELOCITY, model_part.ProcessInfo),
                    dtype=float,
                )[:, :2]
                elem_subscale_pressure = np.asarray(
                    elem.CalculateOnIntegrationPoints(KFD.SUBSCALE_PRESSURE, model_part.ProcessInfo),
                    dtype=float,
                ).reshape(-1)
                if q_coords_ref is None:
                    n_qp = int(elem_q_coords_ref.shape[0])
                    q_coords_ref = np.zeros((n_elem, n_qp, 2), dtype=float)
                    q_coords_cur = np.zeros((n_elem, n_qp, 2), dtype=float)
                    q_weights = np.zeros((n_elem, n_qp), dtype=float)
                    subscale_velocity = np.zeros((n_elem, n_qp, 2), dtype=float)
                    subscale_pressure = np.zeros((n_elem, n_qp), dtype=float)
                q_coords_ref[eidx, :, :] = elem_q_coords_ref
                q_coords_cur[eidx, :, :] = elem_q_coords_cur
                q_weights[eidx, :] = elem_q_weights
                subscale_velocity[eidx, :, :] = elem_subscale_velocity
                subscale_pressure[eidx, :] = elem_subscale_pressure

            npz_payload.update(
                {
                    "element_node_ids": node_ids_by_elem,
                    "q_coords_ref": q_coords_ref,
                    "q_coords_cur": q_coords_cur,
                    "q_weights": q_weights,
                    "subscale_velocity": subscale_velocity,
                    "subscale_pressure": subscale_pressure,
                }
            )

        npz_path = run_dir / f"{args.output_stem}.npz"
        np.savez(npz_path, **npz_payload)
        summary = {
            "run_dir": str(run_dir),
            "npz_path": str(npz_path),
            "num_nodes": int(node_ids.shape[0]),
            "num_elements": int(node_ids_by_elem.shape[0]),
            "num_qp_per_element": int(q_coords_ref.shape[1]) if q_coords_ref is not None else 0,
            "nodal_only": bool(args.nodal_only),
            "interface_only": bool(args.interface_only),
            "interface_part": str(args.interface_part),
            "subscale_velocity_inf_norm": float(np.max(np.abs(subscale_velocity))) if subscale_velocity is not None else 0.0,
            "reaction_inf_norm": float(np.max(np.abs(reaction))) if reaction.size else 0.0,
        }
        (run_dir / f"{args.output_stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
