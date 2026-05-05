import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.meshgen import structured_quad


def _unit_square_mesh_with_all_boundary_tag(nx: int = 2, ny: int = 1) -> Mesh:
    nodes, elem_conn, edges, corner = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=1)
    mesh = Mesh(
        nodes,
        elem_conn,
        edges_connectivity=edges,
        elements_corner_nodes=corner,
        element_type="quad",
        poly_order=1,
    )
    mesh.tag_boundary_edges({"all": lambda _x, _y: True})
    return mesh


def test_project_hdiv_boundary_flux_rt0_constant_normal_flux():
    mesh = _unit_square_mesh_with_all_boundary_tag(nx=2, ny=1)
    me = MixedElement(mesh, {"u": ("RT", 0)})
    dh = DofHandler(me, method="cg")

    vals = dh.project_hdiv_boundary_flux("u", "all", 1.0, qdeg=10)
    boundary_edges = [e for e in mesh.edges_list if e.right is None]
    assert len(vals) == len(boundary_edges) * 1

    for e in boundary_edges:
        dofs = dh.edge_trace_dofs("u", int(e.gid))
        assert len(dofs) == 1
        p0, p1 = mesh.nodes_x_y_pos[list(e.nodes)]
        L = float(np.linalg.norm(p1 - p0))
        assert abs(float(vals[int(dofs[0])]) - L) < 1e-12


def test_project_hdiv_boundary_flux_rt1_constant_normal_flux_legendre_mode_1_is_zero():
    mesh = _unit_square_mesh_with_all_boundary_tag(nx=2, ny=1)
    me = MixedElement(mesh, {"u": ("RT", 1)})
    dh = DofHandler(me, method="cg")

    vals = dh.project_hdiv_boundary_flux("u", "all", 1.0, qdeg=12)
    boundary_edges = [e for e in mesh.edges_list if e.right is None]
    assert len(vals) == len(boundary_edges) * 2

    for e in boundary_edges:
        dofs = dh.edge_trace_dofs("u", int(e.gid))
        assert len(dofs) == 2
        p0, p1 = mesh.nodes_x_y_pos[list(e.nodes)]
        L = float(np.linalg.norm(p1 - p0))
        assert abs(float(vals[int(dofs[0])]) - L) < 1e-12
        assert abs(float(vals[int(dofs[1])])) < 1e-12

