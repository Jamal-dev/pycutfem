import pytest
import numpy as np


from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.utils.meshgen import structured_quad


# --- Helper Function for Test Analysis ---

def analyze_dirichlet_result(dof_handler, dirichlet_data):
    """
    Analyzes the output of get_dirichlet_data, sorting constrained DOFs
    by field and providing their corresponding node IDs.
    """
    # Create inverted maps from DOF index to node ID for each field
    dof_to_node_maps = {
        field: {dof: nid for nid, dof in dof_map.items()}
        for field, dof_map in dof_handler.dof_map.items()
    }

    # Group the results by field
    results_by_field = {field: {} for field in dof_handler.field_names}

    for dof, value in dirichlet_data.items():
        found = False
        for field, d_to_n_map in dof_to_node_maps.items():
            if dof in d_to_n_map:
                node_id = d_to_n_map[dof]
                results_by_field[field][node_id] = value
                found = True
                break
        if not found:
            raise AssertionError(f"DOF {dof} was constrained but does not belong to any field.")
            
    return results_by_field


# --- Test Cases for DofHandler ---

class TestDofHandlerCG:
    """
    Test suite for the DofHandler in Continuous Galerkin (CG) mode.
    """

    def test_mixed_element_bc_collection(self):
        """
        Tests Dirichlet data collection on a mixed-element space (Q2-Q2-Q1)
        to ensure the correct number of DOFs for each field are constrained
        on a boundary.
        """
        # 1. ARRANGE: Build a 2x2 mesh of Q2 elements and a mixed FE space
        nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=2)
        mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=2)
        
        # Taylor-Hood-like element: Q2 for velocity (u,v), Q1 for pressure (p)
        me = MixedElement(mesh, field_specs={'ux': 2, 'uy': 2, 'p': 1})
        dof_handler = DofHandler(me, method='cg')

        # 2. DEFINE BOUNDARY CONDITIONS
        # Tag the 'left' boundary (x=0) and apply BCs to all fields
        mesh.tag_boundary_edges({"left": lambda x, y: np.isclose(x, 0.0)})
        bcs = [
            BoundaryCondition(field='ux', domain_tag="left", method="dirichlet", value=0.0),
            BoundaryCondition(field='uy', domain_tag="left", method="dirichlet", value=0.0),
            BoundaryCondition(field='p',  domain_tag="left", method="dirichlet", value=lambda x, y: y * 5)
        ]

        # 3. ACT: Get the Dirichlet data
        dirichlet_data = dof_handler.get_dirichlet_data(bcs)
        results = analyze_dirichlet_result(dof_handler, dirichlet_data)

        # 4. ASSERT
        # A 2x2 grid of Q2 elements results in a 5x5 grid of nodes.
        # The left edge (x=0) contains 5 nodes.
        # ux (Q2) and uy (Q2) should use all 5 nodes on this edge.
        # p (Q1) uses a lower-order space. On a Q2 mesh, a P1 field only uses
        # the corner nodes of the Q2 elements. The left edge has 2 Q2 elements
        # stacked, so it has 3 unique corner nodes (bottom, middle, top).
        assert len(results['ux']) == 5, "Expected 5 constrained DOFs for 'ux' field."
        assert len(results['uy']) == 5, "Expected 5 constrained DOFs for 'uy' field."
        assert len(results['p']) == 3, "Expected 3 constrained DOFs for 'p' field."

        # Verify that all constrained nodes are physically on the boundary
        for field, field_results in results.items():
            for node_id, value in field_results.items():
                node_x = mesh.nodes_x_y_pos[node_id][0]
                assert np.isclose(node_x, 0.0), \
                    f"Node {node_id} for field '{field}' is not on the left boundary (x={node_x})."

    def test_single_node_bc_with_locator(self):
        """
        Tests applying a BC to a single node for a specific field using a
        runtime geometric locator, not a pre-defined tag.
        """
        # 1. ARRANGE
        nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=2)
        mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=2)
        me = MixedElement(mesh, field_specs={'u': 2, 'v': 2})
        dof_handler = DofHandler(me, method='cg')

        # 2. DEFINE BC for a single point (the mesh center) for field 'u' only.
        # We give it a domain_tag 'pinned' which is NOT on the mesh.
        bcs = [
            BoundaryCondition(field='u', domain_tag='pinned', method="dirichlet", value=99.0)
        ]
        # We provide a locator function for 'pinned' at runtime.
        locators = {'pinned': lambda x, y: np.isclose(x, 0.5) and np.isclose(y, 0.5)}

        # 3. ACT
        dirichlet_data = dof_handler.get_dirichlet_data(bcs, locators=locators)
        results = analyze_dirichlet_result(dof_handler, dirichlet_data)
        
        # 4. ASSERT
        # We expect exactly one DOF to be constrained in total.
        assert len(dirichlet_data) == 1, "Expected exactly one constrained DOF."
        
        # The constrained DOF must belong to field 'u'.
        assert len(results['u']) == 1, "Expected one constrained DOF for field 'u'."
        assert len(results['v']) == 0, "Field 'v' should have no constrained DOFs."

        # Verify the node location and value.
        pinned_node_id = list(results['u'].keys())[0]
        pinned_node_coords = mesh.nodes_x_y_pos[pinned_node_id]
        assert np.allclose(pinned_node_coords, [0.5, 0.5]), \
            f"Pinned node has wrong coordinates: {pinned_node_coords}"
        assert results['u'][pinned_node_id] == 99.0, "Pinned node has wrong value."
        
    def test_cg_shared_dofs(self):
        """
        Verifies that adjacent elements in a CG space share DOFs on their
        common edge.
        """
        nodes, elems, _, corners = structured_quad(2.0, 1.0, nx=2, ny=1, poly_order=1)
        mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=1)
        me = MixedElement(mesh, field_specs={'u': 1})
        dof_handler = DofHandler(me, method='cg')

        dofs_e0 = set(dof_handler.element_maps['u'][0])
        dofs_e1 = set(dof_handler.element_maps['u'][1])

        # The intersection should contain exactly 2 DOFs for the shared edge.
        shared_dofs = dofs_e0.intersection(dofs_e1)
        assert len(shared_dofs) == 2, \
            f"Expected 2 shared DOFs, found {len(shared_dofs)}"

