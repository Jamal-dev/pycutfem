from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_benchmark6_observation_campaign_cli_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples/biofilms/benchmarks/blauert/benchmark6_observation_campaign.py"
    assert script.exists()

    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert "Benchmark 6" in proc.stdout or "benchmark 6" in proc.stdout.lower()
    assert "--run-root" in proc.stdout
    assert "--cases" in proc.stdout
    assert "--stream" in proc.stdout
    assert "--continue-on-error" in proc.stdout
    assert "--restart-from" in proc.stdout
    assert "--restart-write-every" in proc.stdout
    assert "--restart-dt" in proc.stdout
    assert "--gamma-div" in proc.stdout
    assert "--adaptive-gamma-div" in proc.stdout
    assert "--gamma-div-max" in proc.stdout
    assert "--alpha-ch-eps" in proc.stdout
    assert "--scale-alpha-ch-eps-with-zeta" in proc.stdout
    assert "--diffuse-shear-scale-ref" in proc.stdout
    assert "--vtk-every" in proc.stdout
    assert "--dry-run" in proc.stdout
