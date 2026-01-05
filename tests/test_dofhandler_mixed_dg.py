import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.utils.meshgen import structured_quad


def _make_quad_mesh(nx=2, ny=1, poly_order=1):
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=poly_order)
    return Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )


def test_mixed_cg_dg_edge_sharing():
    mesh = _make_quad_mesh(nx=2, ny=1, poly_order=1)
    me = MixedElement(mesh, field_specs={"u": 1, "p": 1})
    dh = DofHandler(me, method="cg", field_methods={"p": "dg"})

    interior_edge = next(e for e in mesh.edges_list if e.right is not None)

    u_left = dh.edge_dofs("u", interior_edge.gid, owner="left")
    u_right = dh.edge_dofs("u", interior_edge.gid, owner="right")
    assert set(u_left) == set(u_right)
    assert len(u_left) == me._field_orders["u"] + 1

    p_left = dh.edge_dofs("p", interior_edge.gid, owner="left")
    p_right = dh.edge_dofs("p", interior_edge.gid, owner="right")
    assert set(p_left).isdisjoint(set(p_right))

    nid = int(interior_edge.nodes[0])
    assert nid in dh.dof_map["p"]
    assert isinstance(dh.dof_map["p"][nid], dict)
    assert {interior_edge.left, interior_edge.right}.issubset(set(dh.dof_map["p"][nid].keys()))

    p_dofs = dh.get_field_slice("p")
    n_basis = (me._field_orders["p"] + 1) ** 2
    assert len(p_dofs) == mesh.n_elements * n_basis


def test_dg_field_slice_and_coords():
    mesh = _make_quad_mesh(nx=2, ny=1, poly_order=1)
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="dg")

    dofs = dh.get_field_slice("u")
    n_basis = (me._field_orders["u"] + 1) ** 2
    assert len(dofs) == mesh.n_elements * n_basis

    coords = dh.get_field_dof_coords("u")
    assert coords.shape == (len(dofs), 2)
    assert np.isfinite(coords).all()


def test_dirichlet_skipped_for_dg_fields():
    mesh = _make_quad_mesh(nx=1, ny=1, poly_order=1)
    mesh.tag_boundary_edges({"all": lambda x, y: True})

    me = MixedElement(mesh, field_specs={"u": 1, "p": 1})
    dh = DofHandler(me, method="cg", field_methods={"p": "dg"})
    bcs = [
        BoundaryCondition("u", "dirichlet", "all", lambda x, y: 0.0),
        BoundaryCondition("p", "dirichlet", "all", lambda x, y: 0.0),
    ]
    data = dh.get_dirichlet_data(bcs)
    assert data
    u_set = set(dh.get_field_slice("u"))
    p_set = set(dh.get_field_slice("p"))
    assert set(data).issubset(u_set)
    assert set(data).isdisjoint(p_set)

    dh_dg = DofHandler(MixedElement(mesh, field_specs={"u": 1}), method="dg")
    data_dg = dh_dg.get_dirichlet_data([BoundaryCondition("u", "dirichlet", "all", lambda x, y: 0.0)])
    assert data_dg == {}


def test_dg_edge_pairs_ordering():
    mesh = _make_quad_mesh(nx=2, ny=1, poly_order=1)
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="dg")

    interior_edge = next(e for e in mesh.edges_list if e.right is not None)
    left, right = dh.get_dof_pairs_for_edge("u", interior_edge.gid)

    assert len(left) == len(right) == me._field_orders["u"] + 1
    assert set(left).isdisjoint(set(right))

    coords = dh.get_field_dof_coords("u")
    for ldof, rdof in zip(left, right):
        assert np.allclose(coords[ldof], coords[rdof])
