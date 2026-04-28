#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def _command(*parts: str) -> list[str]:
    return [str(part) for part in parts]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conda-env", default="fenicsx")
    ap.add_argument("--outdir", default="out/precice_channel_wall")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    fluid_log = outdir / "fluid.log"
    solid_log = outdir / "solid.log"
    fluid_out = outdir / "fluid"
    solid_out = outdir / "solid"
    fluid_out.mkdir(parents=True, exist_ok=True)
    solid_out.mkdir(parents=True, exist_ok=True)

    fluid_cmd = _command(
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        str(args.conda_env),
        "python",
        str(root / "pycutfem_fluid_participant.py"),
        "--precice-config",
        str(root / "precice-config.xml"),
        "--outdir",
        str(fluid_out),
    )
    solid_cmd = _command(
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        str(args.conda_env),
        "python",
        str(root / "fenicsx_solid_participant.py"),
        "--adapter-config",
        str(root / "solid-precice-adapter-config.json"),
        "--outdir",
        str(solid_out),
    )

    fluid = None
    solid = None
    with fluid_log.open("w") as fluid_handle, solid_log.open("w") as solid_handle:
        try:
            fluid = subprocess.Popen(fluid_cmd, cwd=str(root), stdout=fluid_handle, stderr=subprocess.STDOUT)
            time.sleep(1.0)
            solid = subprocess.Popen(solid_cmd, cwd=str(root), stdout=solid_handle, stderr=subprocess.STDOUT)
            solid_rc = solid.wait()
            fluid_rc = fluid.wait()
        finally:
            if solid is not None and solid.poll() is None:
                solid.terminate()
            if fluid is not None and fluid.poll() is None:
                fluid.terminate()

    if solid_rc != 0 or fluid_rc != 0:
        raise SystemExit(
            f"Coupled run failed (fluid rc={int(fluid_rc)}, solid rc={int(solid_rc)}). "
            f"Inspect {fluid_log} and {solid_log}."
        )
    print(f"Coupled run completed. Logs: {fluid_log} {solid_log}", flush=True)


if __name__ == "__main__":
    main()
