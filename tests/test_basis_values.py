import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.core.topology import Node
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler

from pycutfem.xfem import build_alpha_by_field


def _unit_square_two_tris():
    n0 = Node(0, 0.0, 0.0)
    n1 = Node(1, 1.0, 0.0)
    n2 = Node(2, 1.0, 1.0)
    n3 = Node(3, 0.0, 1.0)
    nodes = [n0, n1, n2, n3]
    elements = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    edges = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [3, 2]], dtype=int)
    corners = elements.copy()
    return Mesh(nodes, elements, edges, corners, element_type="tri", poly_order=1)


def test_alpha_values_match_node_sides():
    mesh = _unit_square_two_tris()
    me = MixedElement(mesh, {"u": 1})
    dh = DofHandler(me, method="cg")

    ls = AffineLevelSet(1.0, 0.0, -0.5)  # x - 0.5
    dh.classify_from_levelset(ls)

    # Element 0 has one node on '-' and two on '+'
    eid = 0
    gidx = np.asarray(dh.element_maps["u"][eid], dtype=int)
    coords = dh.get_all_dof_coords()[gidx]
    phi = np.asarray([float(ls(xy)) for xy in coords], dtype=float)

    alpha_pos = build_alpha_by_field(dh, ["u"], eid, ls, side="+")["u"]
    alpha_neg = build_alpha_by_field(dh, ["u"], eid, ls, side="-")["u"]

    # '+' integration: alpha=0 on '+' nodes, alpha=1 on '-' nodes
    exp_pos = np.where(phi > 0.0, 0.0, 1.0)
    # '-' integration: alpha=0 on '-' nodes, alpha=-1 on '+' nodes
    exp_neg = np.where(phi < 0.0, 0.0, -1.0)

    assert np.allclose(alpha_pos, exp_pos)
    assert np.allclose(alpha_neg, exp_neg)

