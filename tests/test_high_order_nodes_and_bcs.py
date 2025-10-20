# tests/test_high_order_nodes_and_bcs.py

import numpy as np
import pytest

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.forms import BoundaryCondition


# --- helper: same grouping idea as in your geometry-independent tests ----------
def _group_dirichlet_by_field_and_node(dh: DofHandler, dirichlet_data: dict[int, float]):
    """
    Map constrained DOFs back to (field, mesh-node-id) for easy assertions.
    Mirrors the pattern used elsewhere in your tests.
    """
    results = {f: {} for f in dh.field_names}
    for gd, val in dirichlet_data.items():
        f, nid = dh._dof_to_node_map.get(int(gd), (None, None))
        if f is not None and nid is not None:
            results[f][int(nid)] = val
    return results


# =============================================================================
# A) EDGE NODES EXISTENCE via Edge.all_nodes (Qp geometry)
# =============================================================================
@pytest.mark.parametrize("p_geom", [2, 3, 4])
@pytest.mark.parametrize("nx,ny", [(2, 2), (3, 1), (1, 3)])
def test_boundary_edge_nodes_all_present(p_geom, nx, ny):
    """
    Ensure every tagged boundary edge contributes exactly (p_geom+1) nodes
    via Edge.all_nodes, and the union across edges has (ny*p+1) or (nx*p+1).
    """
    L = H = 1.0
    nodes, elems, _, corners = structured_quad(L, H, nx=nx, ny=ny, poly_order=p_geom)
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=p_geom)

    mesh.tag_boundary_edges({
        'left':   lambda x, y: np.isclose(x, 0.0),
        'right':  lambda x, y: np.isclose(x, 1.0),
        'bottom': lambda x, y: np.isclose(y, 0.0),
        'top':    lambda x, y: np.isclose(y, 1.0),
    })

    # Check all tagged 'left' edges
    left_bitset = mesh.edge_bitset('left')
    left_edges = np.nonzero(left_bitset.to_indices())[0]

    # Each 'left' edge should contribute (p_geom+1) nodes
    for eid in left_edges:
        e = mesh.edge(int(eid))
        assert len(e.all_nodes) == p_geom + 1, "Edge did not expose all high-order nodes"

    # Union along the full boundary should be (ny*p + 1) unique nodes
    left_node_union = set()
    for eid in left_edges:
        e = mesh.edge(int(eid))
        left_node_union.update(int(n) for n in e.all_nodes)
    assert len(left_node_union) == ny * p_geom + 1


# =============================================================================
# B) CG DOF COUNTS and SHARED EDGE INTERSECTIONS
# =============================================================================
@pytest.mark.parametrize("nx,ny", [(2, 1), (3, 1)])
def test_shared_edge_cg_intersection_q3q1(nx, ny):
    """
    For Q3/Q1, two elements that share an interior edge should share
    (p+1) velocity DOFs on that edge, and (1+1)=2 pressure DOFs.
    """
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=3)
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=3)
    me = MixedElement(mesh, field_specs={'ux': 3, 'p': 1})
    dh = DofHandler(me, method='cg')

    # pick a pair of neighbors (row-major)
    e0, e1 = 0, 1
    ux_shared = set(dh.element_maps['ux'][e0]).intersection(dh.element_maps['ux'][e1])
    p_shared  = set(dh.element_maps['p' ][e0]).intersection(dh.element_maps['p' ][e1])

    assert len(ux_shared) == 3 + 1
    assert len(p_shared)  == 1 + 1


# =============================================================================
# C) DIRICHLET COLLECTION: exact counts & values for Q2, Q3, Q4 fields
# =============================================================================
@pytest.mark.parametrize("p_geom, p_field", [(2, 2), (3, 3), (4, 4)])
def test_dirichlet_counts_and_values_velocity_on_top(p_geom, p_field):
    """
    On a Qp geometry with a field of the same order, tagging 'top' must select
    (nx*p + 1) nodes with correct coordinate-based values.
    """
    nx, ny = 2, 2
    L = H = 1.0
    nodes, elems, _, corners = structured_quad(L, H, nx=nx, ny=ny, poly_order=p_geom)
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=p_geom)

    me = MixedElement(mesh, field_specs={'u': p_field})
    dh = DofHandler(me, method='cg')

    mesh.tag_boundary_edges({'top': lambda x, y: np.isclose(y, 1.0)})

    # Value sensitive to x so we catch any missing mid-edge nodes
    bcs = [BoundaryCondition('u', 'dirichlet', 'top', lambda x, y: x**2)]
    data = dh.get_dirichlet_data(bcs)
    grouped = _group_dirichlet_by_field_and_node(dh, data)

    want = nx * p_geom + 1
    assert len(grouped['u']) == want

    # Check each constrained node lies on y=1 and value matches x**2
    for nid, val in grouped['u'].items():
        x, y = mesh.nodes_x_y_pos[int(nid)]
        assert np.isclose(y, 1.0)
        assert np.isclose(val, x**2)


@pytest.mark.parametrize("p_geom, p_field", [(3, 1), (4, 2)])
def test_dirichlet_counts_mixed_orders_on_left(p_geom, p_field):
    """
    Mixed-order check: on Q3/Q1 and Q4/Q2, a left-edge tag must select
    (ny*p_geom + 1) DOFs for the *field* (even if field order < geom order).
    """
    nx, ny = 2, 2
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=p_geom)
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=p_geom)

    me = MixedElement(mesh, field_specs={'w': p_field})
    dh = DofHandler(me, method='cg')

    mesh.tag_boundary_edges({'left': lambda x, y: np.isclose(x, 0.0)})

    bcs = [BoundaryCondition('w', 'dirichlet', 'left', lambda x, y: -y)]
    data = dh.get_dirichlet_data(bcs)
    grouped = _group_dirichlet_by_field_and_node(dh, data)

    want = ny * p_field + 1    # <-- RIGHT: counts follow the FIELD order
    assert len(grouped['w']) == want

    # Values check
    for nid, val in grouped['w'].items():
        x, y = mesh.nodes_x_y_pos[int(nid)]
        assert np.isclose(x, 0.0)
        assert np.isclose(val, -y)


# =============================================================================
# D) FIELD-SAFE: velocity BC must not constrain pressure DOFs and vice versa
# =============================================================================
def test_field_safe_bc_application_q3q1():
    nx = ny = 2
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=3)
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=3)
    me = MixedElement(mesh, field_specs={'ux': 3, 'uy': 3, 'p': 1})
    dh = DofHandler(me, method='cg')

    mesh.tag_boundary_edges({'top': lambda x, y: np.isclose(y, 1.0)})

    bcs = [
        BoundaryCondition('ux', 'dirichlet', 'top', lambda x, y: x + y),
        # No p-BCs here on purpose
    ]
    data = dh.get_dirichlet_data(bcs)
    grouped = _group_dirichlet_by_field_and_node(dh, data)

    assert len(grouped['ux']) == nx * 3 + 1
    assert len(grouped['uy']) == 0
    assert len(grouped['p'])  == 0


# =============================================================================
# E) LOCATOR: single-node pin at center for the lower-order field on Q3/Q1
# =============================================================================
def test_locator_single_node_pin_q3q1():
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=3)
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=3)
    me = MixedElement(mesh, field_specs={'ux': 3, 'uy': 3, 'p': 1})
    dh = DofHandler(me, method='cg')

    bcs = [BoundaryCondition('p', 'dirichlet', 'pin', 99.0)]
    locs = {'pin': lambda x, y: np.isclose(x, 0.5) and np.isclose(y, 0.5)}

    data = dh.get_dirichlet_data(bcs, locators=locs)
    grouped = _group_dirichlet_by_field_and_node(dh, data)

    assert len(data) == 1
    assert len(grouped['p']) == 1
    nid = next(iter(grouped['p'].keys()))
    xy = mesh.nodes_x_y_pos[int(nid)]
    assert np.allclose(xy, [0.5, 0.5])
    assert grouped['p'][nid] == 99.0

def test_q3q1_dof_map_contains_coarse_nodes():
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=3)
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=3)
    me   = MixedElement(mesh, field_specs={'p': 1})  # Q1 field
    dh   = DofHandler(me, method='cg')

    # the coarse grid nodes for Q1 on [0,1]^2
    grid = {0.0, 0.5, 1.0}
    expected = {
        nid for nid,(x,y) in enumerate(mesh.nodes_x_y_pos)
        if any(np.isclose(x,g) for g in grid) and any(np.isclose(y,g) for g in grid)
    }
    got = set(dh.dof_map['p'].keys())
    assert got == expected, f"Q1 field on Q3 geometry should map exactly the coarse 3×3 nodes; got {len(got)}"

def test_field_safe_bc_q3q1_velocity_only():
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=3)
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=3)
    me   = MixedElement(mesh, field_specs={'ux':3,'uy':3,'p':1})
    dh   = DofHandler(me, method='cg')
    mesh.tag_boundary_edges({'top': lambda x,y: np.isclose(y,1.0)})

    data = dh.get_dirichlet_data([BoundaryCondition('ux','dirichlet','top', lambda x,y: x+y)])
    # group back by field
    grouped = {f:{} for f in dh.field_names}
    for gd,val in data.items():
        f,nid = dh._dof_to_node_map[gd]
        grouped[f][nid] = val

    assert len(grouped['ux']) == 2*3 + 1  # ny=2, p_field=3
    assert len(grouped['uy']) == 0
    assert len(grouped['p'])  == 0


