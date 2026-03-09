from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_benchmark6_blauert_channel_cli_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples/biofilms/benchmarks/blauert/paper1_benchmark6_blauert_channel.py"
    assert script.exists()

    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert "Benchmark 6" in proc.stdout
    assert "--profile" in proc.stdout
    assert "--diffuse-shear-scale-list" in proc.stdout
    assert "--diffuse-shear-time-scheme" in proc.stdout
    assert "--nonlinear-solver" in proc.stdout
    assert "--ls-mode" in proc.stdout
    assert "--gamma-u" in proc.stdout
    assert "--u-extension" in proc.stdout
    assert "--dt" in proc.stdout
    assert "--dt-min" in proc.stdout
    assert "--gamma-div" in proc.stdout
    assert "--max-it" in proc.stdout
    assert "--newton-tol" in proc.stdout
    assert "--restart-from" in proc.stdout
    assert "--restart-write-every" in proc.stdout
    assert "--restart-dt" in proc.stdout
    assert "--continue-on-candidate-failure" in proc.stdout
    assert "--stream-subprocess" in proc.stdout
    assert "--calibration-only" in proc.stdout
    assert "--observation-scenarios" in proc.stdout
    assert "--phi-b" in proc.stdout
    assert "--refine-biofilm" in proc.stdout
    assert "--q" in proc.stdout
