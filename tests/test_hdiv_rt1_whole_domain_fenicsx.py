import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import (
    HdivTestFunction,
    HdivTrialFunction,
    TestFunction as UFLTestFunction,
    TrialFunction as UFLTrialFunction,
    div,
    inner,
)
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad


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


def _assemble_rt1_whole_domain_operator(backend: str):
    nodes, elem_conn, edges, corner = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes,
        elem_conn,
        edges_connectivity=edges,
        elements_corner_nodes=corner,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, {"u": ("RT", 1), "p": 1})
    dh = DofHandler(me, method="cg")

    du = HdivTrialFunction("u")
    v = HdivTestFunction("u")
    dp = UFLTrialFunction("p", dof_handler=dh)
    q = UFLTestFunction("p", dof_handler=dh)
    a = (inner(du, v) - dp * div(v) + q * div(du)) * dx(metadata={"q": 6})

    A, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend=backend)
    A = A.toarray()

    x = np.zeros((dh.total_dofs,), dtype=float)
    u_slice = np.asarray(dh.get_field_slice("u"), dtype=int)
    p_slice = np.asarray(dh.get_field_slice("p"), dtype=int)
    x[u_slice] = np.linspace(-0.3, 0.4, u_slice.size)
    p_xy = np.asarray(dh.get_dof_coords("p"), dtype=float)
    x[p_slice] = 0.2 + 0.1 * p_xy[:, 0] - 0.05 * p_xy[:, 1]
    r = np.asarray(A @ x, dtype=float)
    return A, r


@pytest.mark.parametrize("backend", ["jit", "cpp"])
def test_hdiv_rt1_whole_domain_backend_matches_python(monkeypatch, tmp_path, backend):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"jit_cache_{backend}"))
    if backend == "cpp":
        monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")
    else:
        monkeypatch.delenv("PYCUTFEM_JIT_BACKEND", raising=False)

    A_py, r_py = _assemble_rt1_whole_domain_operator("python")
    A_backend, r_backend = _assemble_rt1_whole_domain_operator(backend)

    np.testing.assert_allclose(A_backend, A_py, rtol=1.0e-10, atol=1.0e-12)
    np.testing.assert_allclose(r_backend, r_py, rtol=1.0e-10, atol=1.0e-12)


@pytest.mark.skipif(not _fenicsx_env_available(), reason="fenicsx conda environment is not available")
def test_hdiv_rt1_whole_domain_python_matches_fenicsx(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        {
            "PYCUTFEM_CACHE_DIR": str(tmp_path / "jit_cache"),
            "PYTHONPATH": str(repo_root) + os.pathsep + env.get("PYTHONPATH", ""),
        }
    )

    proc = subprocess.run(
        ["conda", "run", "-n", "fenicsx", "python", "examples/debug/compare_hdiv_rt1_whole_domain_fenicsx.py"],
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
    assert "RT1 whole-domain H(div) volume mixed operator python vs FEniCSx: residual and Jacobian match." in output, output
