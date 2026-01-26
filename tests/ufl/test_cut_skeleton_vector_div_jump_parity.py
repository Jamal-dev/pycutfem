import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import VectorTestFunction, VectorTrialFunction, div, jump
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dCutSkeleton
from pycutfem.utils.meshgen import structured_quad


@pytest.mark.parametrize("backend", ("jit", "cpp"))
def test_cut_skeleton_vector_div_jump_matrix_matches_python(backend: str, monkeypatch, tmp_path):
    """
    Regression:
    Numba JIT must not hard-code unknown union sizes for vector-basis divergence on
    cut-skeleton (cut_interior_facet) measures (previously could generate n_loc=-1).
    """
    monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "numba")
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"jit_cache_{backend}_cutsk_div"))

    nodes, elems, edges, corners = structured_quad(
        2.0, 2.0, nx=2, ny=2, poly_order=1, offset=(-1.0, -1.0)
    )
    mesh = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=1)

    level_set = AffineLevelSet(a=1.0, b=0.0, c=-0.17)
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)

    me = MixedElement(mesh, {"ux": 1, "uy": 1})
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["ux", "uy"])
    u = VectorTrialFunction(space=V, dof_handler=dh)
    v = VectorTestFunction(space=V, dof_handler=dh)

    qdeg = 6
    dsk = dCutSkeleton(level_set=level_set, metadata={"side": "-", "q": qdeg})
    a = jump(div(u)) * jump(div(v)) * dsk

    K_py, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend="python")
    K_b, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend=backend)

    A_py = np.asarray(K_py.toarray(), dtype=float)
    A_b = np.asarray(K_b.toarray(), dtype=float)

    diff = float(np.max(np.abs(A_py - A_b)))
    scale = max(float(np.max(np.abs(A_py))), 1e-14)
    rel = diff / scale

    assert np.isfinite(A_b).all()
    assert diff < 1.0e-8
    assert rel < 1.0e-8

