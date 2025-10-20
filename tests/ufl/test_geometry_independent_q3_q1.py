import numpy as np
import pytest

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.forms import BoundaryCondition

# ------------- local helper (same idea as in test_dofhandler.py) -------------
def analyze_dirichlet_result(dh, dirichlet_data):
    """
    Group constrained DOFs by field and report their *mesh node ids* (when a DOF
    coincides with a mesh node), mirroring the pattern used in your other tests.
    """
    dof_to_node = {g: n for g, (f, n) in dh._dof_to_node_map.items() if n is not None}
    results = {f: {} for f in dh.field_names}
    for gdof, val in dirichlet_data.items():
        # figure out which field this global dof belongs to
        f, _ = dh._dof_to_node_map.get(gdof, (None, None))
        if f is None:
            # shouldn't happen for CG in these tests
            continue
        nid = dof_to_node.get(gdof, None)
        if nid is not None:
            results[f][nid] = val
        else:
            # if a field DOF does not coincide with any mesh node,
            # still record it using a synthetic negative id (rare here for Q3/Q1)
            results[f][-(gdof + 1)] = val
    return results

# ======================================================================
# 1) Construction & DOF counts on geometry-independent Q3/Q1
# ======================================================================
@pytest.mark.parametrize("nx,ny", [(2,2), (3,1)])
def test_q3q1_construction_and_counts(nx, ny):
    # Q3 geometry mesh
    L = H = 1.0
    poly_geom = 3
    nodes, elems, _, corners = structured_quad(L, H, nx=nx, ny=ny, poly_order=poly_geom)
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=poly_geom)

    # MixedElement: Q3 for ux,uy; Q1 for p
    me = MixedElement(mesh, field_specs={'ux': 3, 'uy': 3, 'p': 1})
    dh = DofHandler(me, method='cg')  # should NOT raise

    # Expected CG DOF counts per field: ((nx*p)+1) * ((ny*p)+1)
    def expected_ndofs(p):
        return (nx * p + 1) * (ny * p + 1)

    ux_n = len(dh.get_field_slice('ux'))
    uy_n = len(dh.get_field_slice('uy'))
    p_n  = len(dh.get_field_slice('p'))

    assert ux_n == expected_ndofs(3)
    assert uy_n == expected_ndofs(3)
    assert p_n  == expected_ndofs(1)

# ======================================================================
# 2) CG continuity across a shared edge (intersection size check)
# ======================================================================
def test_q3q1_shared_edge_dofs():
    # Two elements side-by-side (share a vertical interior edge)
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=3)
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=3)
    me = MixedElement(mesh, field_specs={'ux': 3, 'p': 1})
    dh = DofHandler(me, method='cg')

    e0, e1 = 0, 1
    ux_shared = set(dh.element_maps['ux'][e0]).intersection(dh.element_maps['ux'][e1])
    p_shared  = set(dh.element_maps['p' ][e0]).intersection(dh.element_maps['p' ][e1])

    # Along the common edge we expect (p+1) shared DOFs
    assert len(ux_shared) == 3 + 1  # Q3 → 4 edge nodes
    assert len(p_shared)  == 1 + 1  # Q1 → 2 edge nodes

# ======================================================================
# 3) Boundary DOF collection by tag (left edge) and value correctness
# ======================================================================
def test_q3q1_dirichlet_collection_left_edge():
    nx, ny = 2, 2
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=3)
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=3)
    me = MixedElement(mesh, field_specs={'ux': 3, 'uy': 3, 'p': 1})
    dh = DofHandler(me, method='cg')

    # Tag x=0 boundary
    mesh.tag_boundary_edges({"left": lambda x, y: np.isclose(x, 0.0)})

    # Use values that depend on coordinates to be sensitive to any mismatch
    bcs = [
        BoundaryCondition(field='ux', method='dirichlet', domain_tag='left', value=lambda x,y: y),
        BoundaryCondition(field='uy', method='dirichlet', domain_tag='left', value=lambda x,y: 2*y),
        BoundaryCondition(field='p',  method='dirichlet', domain_tag='left', value=lambda x,y: -y),
    ]
    data = dh.get_dirichlet_data(bcs)
    grouped = analyze_dirichlet_result(dh, data)

    # Counts: Q3 has (ny*p + 1) nodes along the edge; Q1 has (ny*1 + 1)
    assert len(grouped['ux']) == ny*3 + 1  # 7 for ny=2
    assert len(grouped['uy']) == ny*3 + 1
    assert len(grouped['p'])  == ny*1 + 1  # 3 for ny=2

    # Values match the coordinate-based functions at each constrained node
    for nid, val in grouped['ux'].items():
        x, y = mesh.nodes_x_y_pos[nid]
        assert np.isclose(x, 0.0)
        assert np.isclose(val, y)
    for nid, val in grouped['uy'].items():
        x, y = mesh.nodes_x_y_pos[nid]
        assert np.isclose(x, 0.0)
        assert np.isclose(val, 2*y)
    for nid, val in grouped['p'].items():
        x, y = mesh.nodes_x_y_pos[nid]
        assert np.isclose(x, 0.0)
        assert np.isclose(val, -y)

# ======================================================================
# 4) Locator-based BC picks a single Q1 node at (0.5, 0.5)
# ======================================================================
def test_q3q1_single_node_bc_with_locator_on_p_center():
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=3)
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=3)
    me = MixedElement(mesh, field_specs={'ux': 3, 'uy': 3, 'p': 1})
    dh = DofHandler(me, method='cg')

    bcs = [BoundaryCondition(field='p', domain_tag='pin', method='dirichlet', value=42.0)]
    locators = {'pin': lambda x,y: np.isclose(x, 0.5) and np.isclose(y, 0.5)}

    data = dh.get_dirichlet_data(bcs, locators=locators)
    grouped = analyze_dirichlet_result(dh, data)

    # Exactly one constrained DOF, and it belongs to 'p'
    assert len(data) == 1
    assert len(grouped['p']) == 1
    assert len(grouped['ux']) == 0
    assert len(grouped['uy']) == 0

    nid = next(iter(grouped['p'].keys()))
    xy = mesh.nodes_x_y_pos[nid]
    assert np.allclose(xy, [0.5, 0.5])
    assert grouped['p'][nid] == 42.0

# ======================================================================
# 5) Node↔DOF mapping subset for Q1 on a Q3 geometry
#     (only coarse 3x3 grid nodes should be present for 'p')
# ======================================================================
def test_q3q1_dof_map_subset_for_p():
    nx, ny = 2, 2
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=3)
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=3)
    me = MixedElement(mesh, field_specs={'ux': 3, 'uy': 3, 'p': 1})
    dh = DofHandler(me, method='cg')

    # expected Q1 grid nodes on [0,1]x[0,1]: x,y ∈ {0, 0.5, 1}
    grid = {0.0, 0.5, 1.0}
    expected_node_ids = {
        nid for nid, (x, y) in enumerate(mesh.nodes_x_y_pos)
        if any(np.isclose(x, g) for g in grid) and any(np.isclose(y, g) for g in grid)
    }
    # the mapping for 'p' should exist exactly for those nodes
    mapped_node_ids = set(dh.dof_map['p'].keys())

    assert mapped_node_ids == expected_node_ids
    assert len(mapped_node_ids) == 9          # 3x3
    assert len(dh.get_field_slice('p')) == 9  # and the field really has 9 DOFs

def test_tagging_q3_geometry_q2_field_strict_inside():
    # 2x2 Q3 geometry
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=3)
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=3)
    # Single field: Q2 (e.g., pressure)
    me = MixedElement(mesh, field_specs={'p': 2})
    dh = DofHandler(me, method='cg')

    # Select "inside" as the left half elements (ids {0,2} in row-major)
    inside_eids = {0, 2}
    # Tag strictly-inside DOFs
    tagged = dh.tag_dofs_from_element_bitset("inactive_inside_p", "p", inside_eids, strict=True)

    # Verify: every tagged DOF's adjacent elements are subset of inside_eids
    # and DOFs on the internal vertical interface (shared with right cells) are NOT tagged.
    # Build adjacency to check:
    adj = {}
    for eid, gds in enumerate(dh.element_maps['p']):
        for g in gds:
            adj.setdefault(int(g), set()).add(int(eid))

    assert len(tagged) > 0
    for gd in tagged:
        assert adj[gd].issubset(inside_eids)
    # And at least one DOF on the interior vertical edge exists but is excluded:
    # Find DOFs that touch both a left and a right element:
    right_eids = {1, 3}
    interface_dofs = {gd for gd, es in adj.items() if (es & inside_eids) and (es & right_eids)}
    assert interface_dofs - tagged == interface_dofs  # all interface DOFs untagged under strict=True