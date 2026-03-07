from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_blauert_biofilm_deformation_one_domain_cli_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples/biofilms/benchmarks/blauert/blauert_biofilm_deformation_one_domain.py"

    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"rc={proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n"
    assert "Blauert" in proc.stdout
    assert "--refine-biofilm" in proc.stdout
    assert "--refine-band" in proc.stdout
    assert "--restart-from" in proc.stdout
    assert "--restart-write-every" in proc.stdout
    assert "--restart-dt" in proc.stdout
    assert "--restart-dir" in proc.stdout
    assert "--trace-residual-fields" in proc.stdout
    assert "--trace-residual-worst" in proc.stdout
    assert "--v-supg" in proc.stdout
    assert "--u-supg" in proc.stdout
    assert "--newton-rtol" in proc.stdout
    assert "--accept-nonconverged-atol-factor" in proc.stdout
    assert "--kinematics-scale" in proc.stdout
    assert "--transport-mode" in proc.stdout
    assert "--gamma-phi" in proc.stdout
    assert "--D-phi" in proc.stdout
    assert "--alpha-advection-form" in proc.stdout
    assert "--alpha-ch-M" in proc.stdout
    assert "--alpha-ch-gamma" in proc.stdout
