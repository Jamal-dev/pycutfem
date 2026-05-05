import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.core.topology import Node
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad

from pycutfem.xfem import XFEMDofHandler, AgFEMMapper


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


def test_xfem_dof_count_and_map():
    mesh = _unit_square_two_tris()
    me = MixedElement(mesh, {"u": 1})
    dh = DofHandler(me, method="cg")

    ls = AffineLevelSet(1.0, 0.0, -0.5)  # x - 0.5
    dh.classify_from_levelset(ls)

    xfem = XFEMDofHandler(dh)
    xfem.rebuild_enrichment(ls, enrich={"u": "heaviside"})

    cut_ids = mesh.element_bitset("cut").to_indices()
    enriched_base = set()
    for eid in cut_ids:
        enriched_base.update(int(g) for g in dh.element_maps["u"][int(eid)])

    assert xfem.base_total_dofs == dh.total_dofs
    assert xfem.total_dofs == dh.total_dofs + len(enriched_base)

    for gd in enriched_base:
        enr = xfem.enrichment_map.get(("u", int(gd)))
        assert enr is not None
        assert int(enr) >= xfem.base_total_dofs


def test_agfem_mapper_ghost_to_root_valid():
    # Structured quads: 4x1 elements on [0,1]x[0,1]
    nodes, quads, edges, corners = structured_quad(1.0, 1.0, nx=4, ny=1, poly_order=1)
    mesh = Mesh(nodes, quads, edges, corners, element_type="quad", poly_order=1)
    me = MixedElement(mesh, {"u": 1})
    dh = DofHandler(me, method="cg")

    # Vertical interface extremely close to x=0.25 so that the first column is a tiny '+' sliver
    ls = AffineLevelSet(1.0, 0.0, -0.249999)  # x - 0.249999

    mapper = AgFEMMapper(dh)
    ag = mapper.build_aggregation_map(ls, side="+", theta_min=0.05)

    assert ag.ghost_eids.size > 0
    ghost_set = set(int(e) for e in ag.ghost_eids.tolist())

    for ge, re in ag.ghost_to_root.items():
        assert int(ge) in ghost_set
        assert 0 <= int(re) < mesh.n_elements
        assert int(re) not in ghost_set

