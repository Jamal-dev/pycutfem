import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import CircleLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.meshgen import structured_triangles
from pycutfem.xfem import AgFEMMapper


def test_agfem_maps_cut_elements_to_multiple_roots() -> None:
    """
    Regression test: AgFEM aggregation must map *each* cut (ghost) element to a
    nearby root element, not collapse an entire connected ghost band onto a
    single root.
    """
    nodes, elems, edges, corners = structured_triangles(
        2.0, 2.0, nx_quads=10, ny_quads=10, poly_order=1, offset=(-1.0, -1.0)
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="tri",
        poly_order=1,
    )

    ls = CircleLevelSet(radius=2.0 / 3.0, center=(0.0, 0.0))

    me = MixedElement(mesh, {"u": 1})
    dh = DofHandler(me, method="cg")
    dh.classify_from_levelset(ls)

    physical = mesh.element_bitset("inside") | mesh.element_bitset("cut")

    mapper = AgFEMMapper(dh)
    ag = mapper.build_aggregation_map(ls, side="-", theta_min=1.0, defined_on=physical)

    assert int(ag.ghost_eids.size) > 1
    assert len(set(int(r) for r in ag.root_eids.tolist())) > 1

