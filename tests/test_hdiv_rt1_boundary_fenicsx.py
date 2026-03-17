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
def test_hdiv_rt1_boundary_python_matches_fenicsx(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        {
            "PYCUTFEM_CACHE_DIR": str(tmp_path / "jit_cache"),
            "PYTHONPATH": str(repo_root) + os.pathsep + env.get("PYTHONPATH", ""),
        }
    )

    proc = subprocess.run(
        ["conda", "run", "-n", "fenicsx", "python", "examples/debug/compare_hdiv_rt1_boundary_fenicsx.py"],
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
    assert "RT1 boundary H(div) python vs FEniCSx: residual and Jacobian match." in output, output
