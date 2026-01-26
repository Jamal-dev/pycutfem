import math

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.meshgen import structured_quad


def _paper_mesh(*, nx: int, poly_order: int) -> Mesh:
    nodes, elems, edges, corners = structured_quad(
        1.0,
        1.0,
        nx=nx,
        ny=nx,
        poly_order=poly_order,
        offset=(-0.5, -0.5),
        rotation=math.pi / 4.0,
        rotation_center=(0.0, 0.0),
    )
    return Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )


def test_example41_paper_outer_dirichlet_dofs_p1():
    x0 = -0.45
    r = math.sqrt(2.0) / 2.0
    tol = 5.0e-10

    def on_outer_boundary(x: float, y: float) -> bool:
        return abs(abs(x) + abs(y) - r) <= tol

    def in_physical_domain(x: float, y: float) -> bool:
        return x >= x0 - 1.0e-12

    def outer_dof_locator(x: float, y: float) -> bool:
        return in_physical_domain(x, y) and on_outer_boundary(x, y)

    for nx in (4, 8, 16):
        mesh = _paper_mesh(nx=nx, poly_order=1)
        cut_ls = AffineLevelSet(-1.0, 0.0, x0)
        mesh.tag_boundary_edges({"outer_dirichlet": lambda x, y: float(cut_ls(np.array([x, y], dtype=float))) <= 1.0e-12})

        me = MixedElement(mesh, field_specs={"v_pos_x": 1, "v_pos_y": 1})
        dh = DofHandler(me, method="cg")
        dh.tag_dofs_by_locator_map({"outer_dofs": outer_dof_locator}, fields=["v_pos_x", "v_pos_y"])

        tag = set(int(i) for i in dh.dof_tags.get("outer_dofs", set()))
        assert tag, "Expected non-empty outer_dofs tag for the paper layout."

        for field in ("v_pos_x", "v_pos_y"):
            ids = np.asarray(dh.get_field_slice(field), dtype=int)
            coords = dh.get_dof_coords(field)
            mask = np.array([outer_dof_locator(float(x), float(y)) for x, y in coords], dtype=bool)
            expected = set(int(i) for i in ids[mask].tolist())
            got = set(int(i) for i in ids[np.isin(ids, np.fromiter(tag, dtype=int, count=len(tag)))].tolist())
            assert got == expected

        # Midpoint-based edge tag is not exact on coarse meshes, but should be non-empty.
        assert int(mesh.edge_bitset("outer_dirichlet").cardinality()) > 0

