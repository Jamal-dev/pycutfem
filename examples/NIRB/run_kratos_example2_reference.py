from __future__ import annotations

import argparse
import contextlib
import json
import shutil
from pathlib import Path
from typing import Any

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

    import KratosMultiphysics as KM
    import KratosMultiphysics.ConstitutiveLawsApplication  # noqa: F401
    from KratosMultiphysics.CoSimulationApplication.co_simulation_analysis import CoSimulationAnalysis

    with (run_dir / "DoubleFlap_fsi_parameters_ROM.json").open("r", encoding="utf-8") as f:
        parameters = KM.Parameters(f.read())

    with contextlib.chdir(run_dir):
        simulation = CoSimulationAnalysis(parameters)
        simulation.Run()

    summary = {
        "benchmark_root": str(benchmark_root),
        "run_dir": str(run_dir),
        "end_time": float(args.end_time),
        "fluid_vtk_dir": str(run_dir / str(args.fluid_output_path)),
        "solid_vtk_dir": str(run_dir / str(args.solid_output_path)),
        "coupling_parameters_path": str(run_dir / "DoubleFlap_fsi_parameters_ROM.json"),
        "fluid_parameters_path": str(run_dir / "ProjectParametersCFD.json"),
        "solid_parameters_path": str(run_dir / "ProjectParametersCSM.json"),
    }
    _write_json(run_dir / "summary.json", summary)
    print(f"summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
