import numpy as np
import pytest

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.core.levelset import AffineLevelSet, LevelSetDeformation
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.expressions import Laplacian, TestFunction, TrialFunction
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx


@pytest.mark.parametrize("backend", ("jit", "cpp"))
def test_cut_volume_hessian_deformation_matches_python(backend: str):
    """
    Regression:
    When an isoparametric deformation is active, cut-volume assembly must use
    deformation-aware inverse-map jets (A2/A3/A4). Otherwise Hessian/Laplacian
    terms disagree across backends.
    """
    nodes, elems, edges, corners = structured_quad(
        2.0, 2.0, nx=4, ny=4, poly_order=2, offset=(-1.0, -1.0)
    )
    mesh = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=2)

    level_set = AffineLevelSet(a=1.0, b=0.0, c=-0.1)  # cut not aligned with mesh
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)

    coords = mesh.nodes_x_y_pos
    node_disp = np.zeros_like(coords)
    node_disp[:, 0] = 0.15 * np.sin(np.pi * coords[:, 0]) * np.sin(0.5 * np.pi * coords[:, 1])
    node_disp[:, 1] = -0.10 * np.cos(0.5 * np.pi * coords[:, 0]) * np.sin(np.pi * coords[:, 1])
    deformation = LevelSetDeformation(mesh, node_disp)

    me = MixedElement(mesh, {"u": 2})
    dh = DofHandler(me, method="cg")

    u = TrialFunction(field_name="u", name="u", dof_handler=dh)
    v = TestFunction(field_name="u", name="v", dof_handler=dh)

    qdeg = 6
    dV = dx(level_set=level_set, deformation=deformation, metadata={"side": "-", "q": qdeg})
    a = Laplacian(u) * Laplacian(v) * dV

    K_py, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend="python")
    K_b, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend=backend)

    A_py = np.asarray(K_py.toarray(), dtype=float)
    A_b = np.asarray(K_b.toarray(), dtype=float)

    diff = float(np.max(np.abs(A_py - A_b)))
    scale = max(float(np.max(np.abs(A_py))), 1e-14)
    rel = diff / scale

    assert diff < 1e-3
    assert rel < 1e-6

