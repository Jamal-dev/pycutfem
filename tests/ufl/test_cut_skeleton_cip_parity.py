import numpy as np
import pytest

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.core.levelset import AffineLevelSet, LevelSetDeformation
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.expressions import FacetNormal, Function, TestFunction, TrialFunction, dot, grad, jump, restrict
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dCutSkeleton
from examples.utils.fsi.fully_eulerian import make_domain_sets


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
    # Jump of normal derivatives (CIP): `jump(grad(u), n)` uses sided outward normals.
    a = jump(grad(u), n) * jump(grad(v), n) * dsk

    K_py, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend="python")
    K_b, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend=backend)

    A_py = np.asarray(K_py.toarray(), dtype=float)
    A_b = np.asarray(K_b.toarray(), dtype=float)

    diff = float(np.max(np.abs(A_py - A_b)))
    scale = max(float(np.max(np.abs(A_py))), 1e-14)
    rel = diff / scale

    assert diff < 1e-8
    assert rel < 1e-8


def test_cut_skeleton_cip_restricted_value_residual_matches_python_cpp():
    """
    Regression:
    On cut skeleton facets, the C++ residual path for `grad(Restricted(Function))`
    must agree with Python. This specifically guards the residual-only CIP bug where
    sided coefficient blocks were clipped twice in the facet value-gradient path.
    """
    nodes, elems, edges, corners = structured_quad(
        2.0, 2.0, nx=4, ny=4, poly_order=1, offset=(-1.0, -1.0)
    )
    mesh = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=1)

    level_set = AffineLevelSet(a=1.0, b=0.0, c=0.13)
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)

    coords = mesh.nodes_x_y_pos
    node_disp = np.zeros_like(coords)
    node_disp[:, 0] = 0.04 * np.sin(np.pi * coords[:, 1])
    node_disp[:, 1] = 0.03 * np.cos(np.pi * coords[:, 0])
    deformation = LevelSetDeformation(mesh, node_disp)

    domains = make_domain_sets(mesh, use_aligned_interface=False)

    me = MixedElement(mesh, {"p_pos_": 1})
    dh = DofHandler(me, method="cg")

    p = Function(field_name="p_pos_", name="p_k", dof_handler=dh, side="+")
    q = TestFunction(field_name="p_pos_", name="q", dof_handler=dh, side="+")

    p_r = restrict(p, domains["has_pos"])
    q_r = restrict(q, domains["has_pos"])
    n = FacetNormal()
    dsk = dCutSkeleton(
        level_set=level_set,
        deformation=deformation,
        metadata={"side": "+", "q": 4, "derivs": {(1, 0), (0, 1)}},
    )

    xy = dh.get_all_dof_coords()
    p.nodal_values[:] = np.sin(0.7 * xy[:, 0]) + 0.3 * np.cos(1.1 * xy[:, 1])

    r = dot(jump(grad(p_r)), n) * dot(jump(grad(q_r)), n) * dsk

    _, F_py = assemble_form(Equation(None, r), dof_handler=dh, bcs=[], backend="python")
    _, F_cpp = assemble_form(Equation(None, r), dof_handler=dh, bcs=[], backend="cpp")

    vec_py = np.asarray(F_py, dtype=float).ravel()
    vec_cpp = np.asarray(F_cpp, dtype=float).ravel()

    diff = float(np.max(np.abs(vec_py - vec_cpp)))
    scale = max(float(np.max(np.abs(vec_py))), 1e-14)
    rel = diff / scale

    assert diff < 1e-9
    assert rel < 1e-9
