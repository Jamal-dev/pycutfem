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
    parser.add_argument(
        "--stage",
        choices=("predicted", "solved", "finalized"),
        default="finalized",
        help="Dump the state before SolveSolutionStep(), after SolveSolutionStep(), or after FinalizeSolutionStep().",
    )
    parser.add_argument("--nodal-only", action="store_true", help="Dump only nodal fields and skip per-element quadrature data.")
    parser.add_argument("--interface-only", action="store_true", help="Dump only fluid elements touching the FSI interface.")
    parser.add_argument("--interface-part", type=str, default="NoSlip2D_Interface")
    return parser.parse_args()


def _integration_point_coords(geom, integration_method, n_qp_hint: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = int(len(geom))
    dim = 2
    x0 = np.asarray([[float(node.X0), float(node.Y0)] for node in geom], dtype=float)
    x = np.asarray([[float(node.X), float(node.Y)] for node in geom], dtype=float)
    n_qp = int(n_qp_hint) if n_qp_hint is not None else None
    integration_points = getattr(geom, "IntegrationPoints", None)
    if n_qp is None and callable(integration_points):
        try:
            points = list(integration_points(integration_method))
        except TypeError:
            points = list(integration_points())
        if points:
            n_qp = int(len(points))
    if n_qp is None:
        count_getter = getattr(geom, "IntegrationPointsNumber", None)
        if callable(count_getter):
            try:
                n_qp = int(count_getter(integration_method))
            except TypeError:
                n_qp = int(count_getter())
    if n_qp is None:
        raise RuntimeError("Could not determine the Kratos quadrature count for the requested integration method.")
    q_coords_ref = np.zeros((n_qp, dim), dtype=float)
    q_coords_cur = np.zeros((n_qp, dim), dtype=float)
    q_weights = np.full((n_qp,), np.nan, dtype=float)

    if n_nodes == 3 and n_qp in {1, 3}:
        if n_qp == 1:
            local_points = np.asarray([[1.0 / 3.0, 1.0 / 3.0]], dtype=float)
            q_weights[:] = 0.5
        else:
            local_points = np.asarray(
                [
                    [1.0 / 6.0, 1.0 / 6.0],
                    [2.0 / 3.0, 1.0 / 6.0],
                    [1.0 / 6.0, 2.0 / 3.0],
                ],
                dtype=float,
            )
            q_weights[:] = 1.0 / 6.0
        shape_values = np.column_stack(
            [
                1.0 - local_points[:, 0] - local_points[:, 1],
                local_points[:, 0],
                local_points[:, 1],
            ]
        )
    else:
        shape_values_getter = getattr(geom, "ShapeFunctionsValues", None)
        if not callable(shape_values_getter):
            raise RuntimeError("Kratos geometry does not expose ShapeFunctionsValues().")
        try:
            shape_values = np.asarray(shape_values_getter(integration_method), dtype=float)
        except TypeError:
            shape_values = np.asarray(shape_values_getter(), dtype=float)
        if callable(integration_points):
            try:
                points = list(integration_points(integration_method))
            except TypeError:
                points = list(integration_points())
            if len(points) == n_qp:
                q_weights = np.asarray([float(gp.Weight()) for gp in points], dtype=float)

    for q in range(n_qp):
        N = np.asarray(shape_values[q], dtype=float).reshape(-1)
        q_coords_ref[q, :] = N @ x0
        q_coords_cur[q, :] = N @ x
    return q_coords_ref, q_coords_cur, q_weights


def _resolve_integration_method(KM, elem, geom, n_qp: int):
    getter = getattr(elem, "GetIntegrationMethod", None)
    if callable(getter):
        try:
            method = getter()
            if len(list(geom.IntegrationPoints(method))) == int(n_qp):
                return method
        except Exception:
            pass
    enum_type = getattr(getattr(KM, "GeometryData", None), "IntegrationMethod", None)
    if enum_type is None:
        raise RuntimeError("Kratos GeometryData.IntegrationMethod enum is not available.")
    for name in dir(enum_type):
        if not name.startswith("GI_GAUSS_"):
            continue
        method = getattr(enum_type, name)
        try:
            if len(list(geom.IntegrationPoints(method))) == int(n_qp):
                return method
        except Exception:
            continue
    fallback_names = {
        1: ("GI_GAUSS_1",),
        3: ("GI_GAUSS_2",),
        6: ("GI_GAUSS_3",),
        12: ("GI_GAUSS_4",),
        16: ("GI_GAUSS_5",),
    }
    for name in fallback_names.get(int(n_qp), ()):
        if hasattr(enum_type, name):
            return getattr(enum_type, name)
    raise RuntimeError(f"Could not resolve a Kratos integration method with {int(n_qp)} quadrature points.")


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
        analysis.Initialize()
        solver = analysis._GetSolver()
        current_time = float(getattr(analysis, "time", 0.0))
        new_time = float(solver.AdvanceInTime(current_time))
        if hasattr(analysis, "time"):
            analysis.time = new_time
        analysis.InitializeSolutionStep()
        solver.Predict()
        if str(args.stage) in {"solved", "finalized"}:
            solver.SolveSolutionStep()
        if str(args.stage) == "finalized":
            analysis.FinalizeSolutionStep()
            analysis.OutputSolutionStep()
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
            "stage": np.asarray(str(args.stage)),
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

        node_ids_by_elem = np.zeros((0, 3), dtype=int)
        q_counts = np.zeros((0,), dtype=int)
        q_offsets = np.zeros((1,), dtype=int)
        subscale_velocity_flat = np.zeros((0, 2), dtype=float)
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
            q_coords_ref_list: list[np.ndarray] = []
            q_coords_cur_list: list[np.ndarray] = []
            q_weights_list: list[np.ndarray] = []
            subscale_velocity_list: list[np.ndarray] = []
            subscale_pressure_list: list[np.ndarray] = []

            for eidx, elem in enumerate(elements):
                geom = elem.GetGeometry()
                node_ids_by_elem[eidx, :] = np.asarray([int(node.Id) for node in geom], dtype=int)
                elem_subscale_velocity = np.asarray(
                    elem.CalculateOnIntegrationPoints(KFD.SUBSCALE_VELOCITY, model_part.ProcessInfo),
                    dtype=float,
                )[:, :2]
                elem_subscale_pressure = np.asarray(
                    elem.CalculateOnIntegrationPoints(KFD.SUBSCALE_PRESSURE, model_part.ProcessInfo),
                    dtype=float,
                ).reshape(-1)
                integration_method = _resolve_integration_method(
                    KM,
                    elem,
                    geom,
                    n_qp=int(elem_subscale_velocity.shape[0]),
                )
                elem_q_coords_ref, elem_q_coords_cur, elem_q_weights = _integration_point_coords(
                    geom,
                    integration_method,
                    n_qp_hint=int(elem_subscale_velocity.shape[0]),
                )
                q_coords_ref_list.append(np.asarray(elem_q_coords_ref, dtype=float))
                q_coords_cur_list.append(np.asarray(elem_q_coords_cur, dtype=float))
                q_weights_list.append(np.asarray(elem_q_weights, dtype=float))
                subscale_velocity_list.append(np.asarray(elem_subscale_velocity, dtype=float))
                subscale_pressure_list.append(np.asarray(elem_subscale_pressure, dtype=float))

            q_counts = np.asarray([arr.shape[0] for arr in q_coords_ref_list], dtype=int)
            q_offsets = np.zeros((n_elem + 1,), dtype=int)
            if q_counts.size:
                q_offsets[1:] = np.cumsum(q_counts, dtype=int)
            q_coords_ref_flat = (
                np.concatenate(q_coords_ref_list, axis=0)
                if q_coords_ref_list
                else np.zeros((0, 2), dtype=float)
            )
            q_coords_cur_flat = (
                np.concatenate(q_coords_cur_list, axis=0)
                if q_coords_cur_list
                else np.zeros((0, 2), dtype=float)
            )
            q_weights_flat = (
                np.concatenate(q_weights_list, axis=0)
                if q_weights_list
                else np.zeros((0,), dtype=float)
            )
            subscale_velocity_flat = (
                np.concatenate(subscale_velocity_list, axis=0)
                if subscale_velocity_list
                else np.zeros((0, 2), dtype=float)
            )
            subscale_pressure_flat = (
                np.concatenate(subscale_pressure_list, axis=0)
                if subscale_pressure_list
                else np.zeros((0,), dtype=float)
            )

            npz_payload.update(
                {
                    "element_node_ids": node_ids_by_elem,
                    "q_point_offsets": q_offsets,
                    "q_point_counts": q_counts,
                    "q_coords_ref_flat": q_coords_ref_flat,
                    "q_coords_cur_flat": q_coords_cur_flat,
                    "q_weights_flat": q_weights_flat,
                    "subscale_velocity_flat": subscale_velocity_flat,
                    "subscale_pressure_flat": subscale_pressure_flat,
                }
            )
            if q_counts.size and int(np.min(q_counts)) == int(np.max(q_counts)):
                n_qp = int(q_counts[0])
                npz_payload.update(
                    {
                        "q_coords_ref": q_coords_ref_flat.reshape(n_elem, n_qp, 2),
                        "q_coords_cur": q_coords_cur_flat.reshape(n_elem, n_qp, 2),
                        "q_weights": q_weights_flat.reshape(n_elem, n_qp),
                        "subscale_velocity": subscale_velocity_flat.reshape(n_elem, n_qp, 2),
                        "subscale_pressure": subscale_pressure_flat.reshape(n_elem, n_qp),
                    }
                )

        npz_path = run_dir / f"{args.output_stem}.npz"
        np.savez(npz_path, **npz_payload)
        summary = {
            "run_dir": str(run_dir),
            "npz_path": str(npz_path),
            "stage": str(args.stage),
            "num_nodes": int(node_ids.shape[0]),
            "num_elements": int(node_ids_by_elem.shape[0]),
            "num_qp_total": int(q_offsets[-1]) if not bool(args.nodal_only) and q_offsets.size else 0,
            "num_qp_per_element_min": int(np.min(q_counts)) if not bool(args.nodal_only) and q_counts.size else 0,
            "num_qp_per_element_max": int(np.max(q_counts)) if not bool(args.nodal_only) and q_counts.size else 0,
            "nodal_only": bool(args.nodal_only),
            "interface_only": bool(args.interface_only),
            "interface_part": str(args.interface_part),
            "subscale_velocity_inf_norm": float(np.max(np.abs(subscale_velocity_flat))) if not bool(args.nodal_only) and subscale_velocity_flat.size else 0.0,
            "reaction_inf_norm": float(np.max(np.abs(reaction))) if reaction.size else 0.0,
        }
        (run_dir / f"{args.output_stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if str(args.stage) != "finalized":
            analysis.FinalizeSolutionStep()
            analysis.OutputSolutionStep()
        analysis.Finalize()
        print(json.dumps(summary, indent=2))
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
