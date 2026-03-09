from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_paper1_paper_ready_suite_dry_run_lists_expected_commands() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples/biofilms/deformation_only_paper_ready_suite.py"
    assert script.exists(), "paper_ready suite script not found under examples/biofilms."

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--profile",
            "smoke",
            "--cases",
            "static,transport_translation,benchmark3_wang2014_layered,benchmark3_wang2014_staircase,benchmark4_terzaghi,benchmark5_jonas_shear,benchmark6_blauert_channel",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, (
        f"rc={proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n"
    )
    assert "deformation_only_mms_static_convergence.py" in proc.stdout
    assert "deformation_only_interface_transport_translation.py" in proc.stdout
    assert "paper1_benchmark3_wang2014_layered.py" in proc.stdout
    assert "paper1_benchmark3_wang2014_staircase.py" in proc.stdout
    assert "paper1_benchmark4_terzaghi_consolidation.py" in proc.stdout
    assert "paper1_benchmark5_jonas_shear.py" in proc.stdout
    assert "paper1_benchmark6_blauert_channel.py" in proc.stdout
    assert "examples/biofilms" in proc.stdout
