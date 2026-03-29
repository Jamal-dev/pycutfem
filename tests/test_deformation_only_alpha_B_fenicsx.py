import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def _fenicsx_env_available() -> bool:
    if shutil.which("conda") is None:
        return False
    probe = subprocess.run(
        ["conda", "run", "-n", "fenicsx", "python", "-c", "import dolfinx, basix, ufl"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return probe.returncode == 0


@pytest.mark.skipif(not _fenicsx_env_available(), reason="fenicsx conda environment is not available")
def test_deformation_only_alpha_B_matches_fenicsx_autodiff(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        {
            "PYCUTFEM_CACHE_DIR": str(tmp_path / "jit_cache"),
            "PYTHONPATH": str(repo_root) + os.pathsep + env.get("PYTHONPATH", ""),
        }
    )

    proc = subprocess.run(
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "fenicsx",
            "python",
            "examples/debug/compare_deformation_only_alpha_B_fenicsx.py",
            "--pycutfem-backend",
            "python",
        ],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=900,
        check=False,
    )
    output = proc.stdout

    assert proc.returncode == 0, output
    data = json.loads(output)
    assert float(data["full_res_rel"]) < 1.0e-10, output
    assert float(data["full_jac_rel"]) < 1.0e-9, output
    assert float(data["row_blocks"]["alpha"]["res_rel"]) < 1.0e-12, output
    assert float(data["row_blocks"]["alpha"]["jac_rel"]) < 1.0e-12, output
    assert float(data["row_blocks"]["B"]["res_rel"]) < 1.0e-12, output
    assert float(data["row_blocks"]["B"]["jac_rel"]) < 1.0e-12, output
