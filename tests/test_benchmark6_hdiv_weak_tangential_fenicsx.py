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
def test_benchmark6_hdiv_weak_tangential_nitsche_supg_matches_fenicsx(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        {
            "COMP_FENICS_PROBLEM": "biofilm",
            "COMP_FENICS_NX": "2",
            "COMP_FENICS_NY": "2",
            "COMP_FENICS_FLUID_SPACE": "hdiv",
            "COMP_FENICS_FLUID_HDIV_ORDER": "0",
            "COMP_FENICS_HDIV_TANGENTIAL_DIRICHLET": "1",
            "COMP_FENICS_HDIV_TANGENTIAL_GAMMA": "20.0",
            "COMP_FENICS_HDIV_TANGENTIAL_METHOD": "nitsche",
            "COMP_FENICS_V_SUPG": "0.5",
            # FEniCSx cannot assemble the RT strong-residual form used by the
            # residual SUPG variant here because it requires unsupported
            # higher-order reference-gradient wrapping on this H(div) path.
            # Keep this cross-check on the supported streamline SUPG form.
            "COMP_FENICS_V_SUPG_MODE": "streamline",
            "COMP_FENICS_TERMS": "Biofilm total residual,Biofilm total jacobian",
            "COMP_FENICS_SPARSE_COMPARE": "1",
            "COMP_FENICS_WRITE_XLSX": "0",
            "BACKEND": "python",
            "PYCUTFEM_CACHE_DIR": str(tmp_path / "jit_cache"),
            "PYTHONPATH": str(repo_root) + os.pathsep + env.get("PYTHONPATH", ""),
        }
    )

    proc = subprocess.run(
        ["conda", "run", "-n", "fenicsx", "python", "examples/debug/comparison_with_fenics.py"],
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
    assert "Biofilm total residual [backend=python]" in output, output
    assert "Biofilm total jacobian [backend=python]" in output, output
    assert "Residual vector for 'Biofilm total residual [backend=python]' is numerically equivalent." in output, output
    assert "Jacobian matrix for 'Biofilm total jacobian [backend=python]' is numerically equivalent." in output, output
    assert "❌ OVERALL Failed tests:     0" in output, output
