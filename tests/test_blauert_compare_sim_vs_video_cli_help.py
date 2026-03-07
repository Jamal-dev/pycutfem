from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_blauert_compare_sim_vs_video_cli_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples/biofilms/benchmarks/blauert/compare_sim_vs_video.py"

    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"rc={proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n"
    assert "--exp-csv" in proc.stdout
    assert "--out-dir" in proc.stdout
    assert "--compare" in proc.stdout

