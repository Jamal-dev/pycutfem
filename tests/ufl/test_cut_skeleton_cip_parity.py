import numpy as np
import pytest

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.core.levelset import AffineLevelSet, LevelSetDeformation
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.expressions import FacetNormal, TestFunction, TrialFunction, dot, grad, jump
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dCutSkeleton


@pytest.mark.parametrize("backend", ("jit", "cpp"))
def test_cut_skeleton_cip_matrix_matches_python(backend: str):
    """
    Regression:
    The cut-skeleton (CIP) measure must clip interior facets by the level set and
    remain consistent across backends, including with an isoparametric deformation.
    """
    nodes, elems, edges, corners = structured_quad(
        2.0, 2.0, nx=2, ny=2, poly_order=2, offset=(-1.0, -1.0)
    )
    mesh = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=2)

    level_set = AffineLevelSet(a=1.0, b=0.0, c=-0.17)
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)

    coords = mesh.nodes_x_y_pos
    node_disp = np.zeros_like(coords)
    node_disp[:, 0] = 0.08 * np.sin(np.pi * coords[:, 1])
    node_disp[:, 1] = 0.05 * np.cos(np.pi * coords[:, 0])
    deformation = LevelSetDeformation(mesh, node_disp)

    me = MixedElement(mesh, {"u": 2})
    dh = DofHandler(me, method="cg")

    u = TrialFunction(field_name="u", name="u", dof_handler=dh)
    v = TestFunction(field_name="u", name="v", dof_handler=dh)
    n = FacetNormal()

    qdeg = 6
    dsk = dCutSkeleton(level_set=level_set, deformation=deformation, metadata={"side": "-", "q": qdeg})
    a = jump(dot(grad(u), n)) * jump(dot(grad(v), n)) * dsk

    K_py, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend="python")
    K_b, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend=backend)

    A_py = np.asarray(K_py.toarray(), dtype=float)
    A_b = np.asarray(K_b.toarray(), dtype=float)

    diff = float(np.max(np.abs(A_py - A_b)))
    scale = max(float(np.max(np.abs(A_py))), 1e-14)
    rel = diff / scale

    assert diff < 1e-8
    assert rel < 1e-8

