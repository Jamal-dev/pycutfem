#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def _command(*parts: str) -> list[str]:
    return [str(part) for part in parts]


def _build_clean_subprocess_env() -> tuple[dict[str, str], dict[str, str]]:
    env = dict(os.environ)
    updates: dict[str, str] = {}
    conda_prefix = env.get("CONDA_PREFIX", "").strip()
    lib_entries: list[str] = []
    if conda_prefix:
        conda_root = str(Path(conda_prefix).resolve())
        conda_lib = str((Path(conda_prefix) / "lib").resolve())
        lib_entries.append(conda_lib)
        for raw_entry in env.get("LD_LIBRARY_PATH", "").split(":"):
            entry = raw_entry.strip()
            if not entry:
                continue
            resolved = str(Path(entry).resolve())
            if resolved.startswith(conda_root) and resolved not in lib_entries:
                lib_entries.append(resolved)
    if lib_entries:
        env["LD_LIBRARY_PATH"] = ":".join(lib_entries)
        updates["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH"]
    else:
        env.pop("LD_LIBRARY_PATH", None)
        updates["LD_LIBRARY_PATH"] = ""
    return env, updates


def _should_filter_line(line: str) -> bool:
    return "PETSc Error --- Application was linked against both OpenMPI and MPICH based MPI libraries" in line


def _relay_stream(label: str, stream, log_handle, sink) -> None:
    try:
        for line in iter(stream.readline, ""):
            if _should_filter_line(line):
                continue
            log_handle.write(line)
            log_handle.flush()
            sink.write(f"[{label}] {line}")
            sink.flush()
    finally:
        stream.close()


def _write_top_center_plot(csv_path: Path, outdir: Path) -> list[Path]:
    if not csv_path.exists():
        return []
    rows: list[tuple[float, float]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                rows.append((float(row["time"]), float(row["uy_top_center"])))
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        return []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    rows.sort(key=lambda item: item[0])
    times = [item[0] for item in rows]
    uy_vals = [item[1] for item in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(times, uy_vals, color="#0f766e", linewidth=2.0)
    ax.set_xlabel("time")
    ax.set_ylabel("u_y at top edge x=0.5")
    ax.set_title("Seboldt Example 2 preCICE history")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    outputs = [
        outdir / "uy_top_center_vs_time.png",
        outdir / "uy_top_center_vs_time.svg",
    ]
    fig.savefig(outputs[0], dpi=160)
    fig.savefig(outputs[1])
    plt.close(fig)
    return outputs


def _write_precice_config(
    path: Path,
    *,
    dt: float,
    num_steps: int,
    max_iterations: int,
    convergence_limit: float,
    initial_relaxation: float,
    used_iterations: int,
    reused_windows: int,
) -> None:
    path.write_text(
        f"""<?xml version="1.0"?>
<precice-configuration>
  <log enabled="true">
    <sink enabled="true" filter="%Severity% >= info" format="[%Time%] %Message%" />
  </log>

  <data:scalar name="FluidTraction" />
  <data:scalar name="PorePressure" />
  <data:scalar name="RobinL" />
  <data:vector name="FluidVelocity" />
  <data:vector name="Displacement" />
  <data:vector name="SkeletonVelocity" />
  <data:vector name="DarcyFlux" />

  <mesh name="Fluid-Interface-Mesh" dimensions="2">
    <use-data name="FluidTraction" />
    <use-data name="PorePressure" />
    <use-data name="RobinL" />
    <use-data name="FluidVelocity" />
    <use-data name="Displacement" />
    <use-data name="SkeletonVelocity" />
    <use-data name="DarcyFlux" />
  </mesh>

  <mesh name="Porous-Interface-Mesh" dimensions="2">
    <use-data name="FluidTraction" />
    <use-data name="PorePressure" />
    <use-data name="RobinL" />
    <use-data name="FluidVelocity" />
    <use-data name="Displacement" />
    <use-data name="SkeletonVelocity" />
    <use-data name="DarcyFlux" />
  </mesh>

  <participant name="PorousSolver">
    <provide-mesh name="Porous-Interface-Mesh" />
    <receive-mesh name="Fluid-Interface-Mesh" from="FluidSolver" />
    <read-data name="FluidTraction" mesh="Porous-Interface-Mesh" />
    <read-data name="FluidVelocity" mesh="Porous-Interface-Mesh" />
    <read-data name="RobinL" mesh="Porous-Interface-Mesh" />
    <write-data name="Displacement" mesh="Porous-Interface-Mesh" />
    <write-data name="SkeletonVelocity" mesh="Porous-Interface-Mesh" />
    <write-data name="DarcyFlux" mesh="Porous-Interface-Mesh" />
    <write-data name="PorePressure" mesh="Porous-Interface-Mesh" />
    <mapping:nearest-neighbor direction="write" from="Porous-Interface-Mesh" to="Fluid-Interface-Mesh" constraint="consistent" />
    <mapping:nearest-neighbor direction="read" from="Fluid-Interface-Mesh" to="Porous-Interface-Mesh" constraint="consistent" />
    <watch-point mesh="Porous-Interface-Mesh" name="PorousCenter" coordinate="0.5; 1.0" />
  </participant>

  <participant name="FluidSolver">
    <provide-mesh name="Fluid-Interface-Mesh" />
    <receive-mesh name="Porous-Interface-Mesh" from="PorousSolver" />
    <read-data name="Displacement" mesh="Fluid-Interface-Mesh" />
    <read-data name="SkeletonVelocity" mesh="Fluid-Interface-Mesh" />
    <read-data name="DarcyFlux" mesh="Fluid-Interface-Mesh" />
    <read-data name="PorePressure" mesh="Fluid-Interface-Mesh" />
    <write-data name="FluidTraction" mesh="Fluid-Interface-Mesh" />
    <write-data name="FluidVelocity" mesh="Fluid-Interface-Mesh" />
    <write-data name="RobinL" mesh="Fluid-Interface-Mesh" />
    <mapping:nearest-neighbor direction="write" from="Fluid-Interface-Mesh" to="Porous-Interface-Mesh" constraint="consistent" />
    <mapping:nearest-neighbor direction="read" from="Porous-Interface-Mesh" to="Fluid-Interface-Mesh" constraint="consistent" />
    <watch-point mesh="Fluid-Interface-Mesh" name="FluidCenter" coordinate="0.5; 1.0" />
  </participant>

  <m2n:sockets acceptor="PorousSolver" connector="FluidSolver" />

  <coupling-scheme:serial-implicit>
    <participants first="PorousSolver" second="FluidSolver" />
    <max-time-windows value="{int(num_steps)}" />
    <time-window-size value="{float(dt):.16e}" />
    <exchange data="FluidTraction" mesh="Fluid-Interface-Mesh" from="FluidSolver" to="PorousSolver" />
    <exchange data="FluidVelocity" mesh="Fluid-Interface-Mesh" from="FluidSolver" to="PorousSolver" />
    <exchange data="RobinL" mesh="Fluid-Interface-Mesh" from="FluidSolver" to="PorousSolver" />
    <exchange data="Displacement" mesh="Porous-Interface-Mesh" from="PorousSolver" to="FluidSolver" />
    <exchange data="SkeletonVelocity" mesh="Porous-Interface-Mesh" from="PorousSolver" to="FluidSolver" />
    <exchange data="DarcyFlux" mesh="Porous-Interface-Mesh" from="PorousSolver" to="FluidSolver" />
    <exchange data="PorePressure" mesh="Porous-Interface-Mesh" from="PorousSolver" to="FluidSolver" />
    <max-iterations value="{int(max_iterations)}" />
    <relative-convergence-measure limit="{float(convergence_limit):.16e}" data="Displacement" mesh="Porous-Interface-Mesh" />
    <relative-convergence-measure limit="{float(convergence_limit):.16e}" data="SkeletonVelocity" mesh="Porous-Interface-Mesh" />
    <relative-convergence-measure limit="{float(convergence_limit):.16e}" data="DarcyFlux" mesh="Porous-Interface-Mesh" />
    <relative-convergence-measure limit="{float(convergence_limit):.16e}" data="PorePressure" mesh="Porous-Interface-Mesh" />
    <relative-convergence-measure limit="{float(convergence_limit):.16e}" data="FluidVelocity" mesh="Fluid-Interface-Mesh" />
    <relative-convergence-measure limit="{float(convergence_limit):.16e}" data="FluidTraction" mesh="Fluid-Interface-Mesh" />
    <acceleration:IQN-ILS>
      <data name="FluidVelocity" mesh="Fluid-Interface-Mesh" />
      <data name="FluidTraction" mesh="Fluid-Interface-Mesh" />
      <initial-relaxation value="{float(initial_relaxation):.16e}" />
      <max-used-iterations value="{int(used_iterations)}" />
      <time-windows-reused value="{int(reused_windows)}" />
    </acceleration:IQN-ILS>
  </coupling-scheme:serial-implicit>
</precice-configuration>
""",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conda-env", default="fenicsx")
    ap.add_argument("--kappa", type=float, default=1.0e-3)
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--dt", type=float, default=1.0e-3)
    ap.add_argument("--t-final", type=float, default=3.0)
    ap.add_argument("--L0", type=float, default=2000.0)
    ap.add_argument("--interface-samples", type=int, default=201)
    ap.add_argument("--profile-samples", type=int, default=401)
    ap.add_argument("--factor-solver", default="mumps")
    ap.add_argument("--report-every", type=int, default=100)
    ap.add_argument("--mms-sample-every", type=int, default=0)
    ap.add_argument("--mms-fluid-nx", type=int, default=21)
    ap.add_argument("--mms-fluid-ny", type=int, default=21)
    ap.add_argument("--mms-solid-nx", type=int, default=21)
    ap.add_argument("--mms-solid-ny", type=int, default=11)
    ap.add_argument("--max-iterations", type=int, default=40)
    ap.add_argument("--convergence-limit", type=float, default=1.0e-5)
    ap.add_argument("--initial-relaxation", type=float, default=0.1)
    ap.add_argument("--max-used-iterations", type=int, default=20)
    ap.add_argument("--reused-windows", type=int, default=1)
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument(
        "--outdir",
        default="out/benchmark7_seboldt_precice",
        help="Top-level output directory for the coupled preCICE run.",
    )
    args = ap.parse_args()

    num_steps = int(round(float(args.t_final) / float(args.dt)))
    if abs(num_steps * float(args.dt) - float(args.t_final)) > 1.0e-12:
        raise SystemExit("t_final must be an integer multiple of dt.")

    root = Path(__file__).resolve().parent
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    porous_log = outdir / "porous.log"
    fluid_log = outdir / "fluid.log"
    config_file = outdir / "precice-config.xml"
    porous_out = outdir / "porous"
    fluid_out = outdir / "fluid"
    porous_out.mkdir(parents=True, exist_ok=True)
    fluid_out.mkdir(parents=True, exist_ok=True)

    _write_precice_config(
        config_file,
        dt=float(args.dt),
        num_steps=int(num_steps),
        max_iterations=int(args.max_iterations),
        convergence_limit=float(args.convergence_limit),
        initial_relaxation=float(args.initial_relaxation),
        used_iterations=int(args.max_used_iterations),
        reused_windows=int(args.reused_windows),
    )

    clean_env, env_updates = _build_clean_subprocess_env()
    print(
        "Runtime environment: "
        + " ".join(f"{k}={v}" for k, v in env_updates.items()),
        flush=True,
    )

    if not bool(args.skip_validate):
        validate_cmd = _command(
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            str(args.conda_env),
            "precice-config-validate",
            str(config_file),
        )
        subprocess.run(validate_cmd, cwd=str(root), check=True, env=clean_env)
    common_args = [
        "--precice-config",
        str(config_file),
        "--kappa",
        str(args.kappa),
        "--nx",
        str(args.nx),
        "--dt",
        str(args.dt),
        "--t-final",
        str(args.t_final),
        "--L0",
        str(args.L0),
        "--interface-samples",
        str(args.interface_samples),
        "--profile-samples",
        str(args.profile_samples),
        "--factor-solver",
        str(args.factor_solver),
        "--report-every",
        str(args.report_every),
        "--mms-sample-every",
        str(args.mms_sample_every),
        "--mms-fluid-nx",
        str(args.mms_fluid_nx),
        "--mms-fluid-ny",
        str(args.mms_fluid_ny),
        "--mms-solid-nx",
        str(args.mms_solid_nx),
        "--mms-solid-ny",
        str(args.mms_solid_ny),
    ]

    porous_cmd = _command(
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        str(args.conda_env),
        "python",
        str(root / "paper1_benchmark7_seboldt_precice_porous.py"),
        *common_args,
        "--outdir",
        str(porous_out),
    )
    fluid_cmd = _command(
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        str(args.conda_env),
        "python",
        str(root / "paper1_benchmark7_seboldt_precice_fluid.py"),
        *common_args,
        "--outdir",
        str(fluid_out),
    )

    porous = None
    fluid = None
    fluid_rc = -999
    porous_rc = -999
    with porous_log.open("w", encoding="utf-8") as porous_handle, fluid_log.open("w", encoding="utf-8") as fluid_handle:
        try:
            print(f"Launching PorousSolver: {' '.join(porous_cmd)}", flush=True)
            porous = subprocess.Popen(
                porous_cmd,
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=clean_env,
                text=True,
                bufsize=1,
            )
            time.sleep(1.0)
            print(f"Launching FluidSolver: {' '.join(fluid_cmd)}", flush=True)
            fluid = subprocess.Popen(
                fluid_cmd,
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=clean_env,
                text=True,
                bufsize=1,
            )
            porous_thread = threading.Thread(
                target=_relay_stream,
                args=("Porous", porous.stdout, porous_handle, sys.stdout),
                daemon=True,
            )
            fluid_thread = threading.Thread(
                target=_relay_stream,
                args=("Fluid", fluid.stdout, fluid_handle, sys.stdout),
                daemon=True,
            )
            porous_thread.start()
            fluid_thread.start()
            fluid_rc = fluid.wait()
            porous_rc = porous.wait()
            porous_thread.join()
            fluid_thread.join()
            fluid_rc = int(fluid_rc)
            porous_rc = int(porous_rc)
            print(f"Participant exit codes: porous={porous_rc} fluid={fluid_rc}", flush=True)
        finally:
            if fluid is not None and fluid.poll() is None:
                fluid.terminate()
            if porous is not None and porous.poll() is None:
                porous.terminate()

    plot_outputs = _write_top_center_plot(porous_out / "top_edge_center_history.csv", outdir)
    if plot_outputs:
        print(
            "Top-edge history plot: " + " ".join(str(path) for path in plot_outputs),
            flush=True,
        )

    if int(fluid_rc) != 0 or int(porous_rc) != 0:
        raise SystemExit(
            f"Coupled run failed (porous rc={int(porous_rc)}, fluid rc={int(fluid_rc)}). "
            f"Inspect {porous_log} and {fluid_log}."
        )
    print(
        f"Coupled Seboldt preCICE run completed. Logs: {porous_log} {fluid_log} "
        f"Outputs: {porous_out} {fluid_out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
