import numpy as np

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import LevelSetGridFunction
from pycutfem.utils.fsi_fully_eulerian import nudge_levelset_zeros


def test_nudge_levelset_zeros_breaks_aligned_interface():
    nodes, elems, edges, corners = structured_quad(
        2.0,
        1.0,
        nx=2,
        ny=1,
        poly_order=1,
        offset=(-1.0, -0.5),
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, field_specs={"phi": 1})
    dh = DofHandler(me, method="cg")
    ls = LevelSetGridFunction(dh, field="phi")
    ls.interpolate(lambda x, y: x)
    ls.commit()

    assert mesh.element_bitset("cut").cardinality() == 0
    assert mesh.edge_bitset("interface").cardinality() > 0

    nudged = nudge_levelset_zeros(ls, 1.0e-6, prefer_negative=True, commit=True)
    assert nudged > 0
    assert mesh.element_bitset("cut").cardinality() > 0
    assert mesh.edge_bitset("interface").cardinality() == 0
