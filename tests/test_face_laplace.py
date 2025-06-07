import pytest
import numpy as np
from numpy.testing import assert_allclose

# --- Imports from your project structure ---
# Assuming these are in the same directory or accessible via python path
from pycutfem.assembly.dg_local import face_laplace
from pycutfem.core import Mesh
from pycutfem.utils.meshgen import structured_quad # Placeholder

# --- Test Setup: Helper function to find specific edges ---

def pick_edges(mesh: Mesh) -> tuple[int, int]:
    """
    Finds the edge ID of the first interior edge and the first boundary edge.
    """
    interior_edge_id = None
    boundary_edge_id = None
    for edge in mesh.edges_list:
        if edge.right is not None and interior_edge_id is None:
            interior_edge_id = edge.gid
        if edge.right is None and boundary_edge_id is None:
            boundary_edge_id = edge.gid
        if interior_edge_id is not None and boundary_edge_id is not None:
            break
            
    if interior_edge_id is None:
        raise RuntimeError("No interior edge found in the mesh for testing.")
    if boundary_edge_id is None:
        raise RuntimeError("No boundary edge found in the mesh for testing.")
        
    return interior_edge_id, boundary_edge_id

# --- Test Setup: A robust two-element mesh using the mesh generator ---

@pytest.fixture
def two_quad_q1_mesh() -> Mesh:
    """Creates a simple, consistent 2-element mesh of Q1 quads."""
    # structured_quad is a placeholder for your actual mesh generator
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=1)
    return Mesh(nodes, element_connectivity=elems, edges_connectivity=None, elements_corner_nodes=corners, element_type="quad", poly_order=1)

# --- Test Cases ---

def test_face_zero_for_constant_solution(two_quad_q1_mesh):
    """
    Checks that for a constant solution u=1, the face integral is zero.
    """
    mesh = two_quad_q1_mesh
    int_eid, _ = pick_edges(mesh)

    Ke_face, Fe_face = face_laplace(mesh, int_eid, alpha=10.0, quad_order=3)

    n_loc = 4
    assert Ke_face.shape == (2 * n_loc, 2 * n_loc)
    
    vec_of_ones = np.ones(2 * n_loc)
    
    assert_allclose(Ke_face @ vec_of_ones, 0.0, atol=1e-12, err_msg="K_face @ [1] should be 0")
    assert_allclose(Fe_face, 0.0, atol=1e-12, err_msg="Fe_face should be 0 for constant solution")

def test_face_zero_for_linear_solution(two_quad_q1_mesh):
    """
    Checks that for a linear solution u(x,y), the interior face integral is zero.
    """
    mesh = two_quad_q1_mesh
    int_eid, _ = pick_edges(mesh)
    edge_obj = mesh.edge(int_eid)
    
    Ke_face, _ = face_laplace(mesh, int_eid, alpha=10.0, quad_order=3)
    
    u_linear = lambda x, y: 2*x + 3*y
    
    nodes_L = mesh.nodes[mesh.elements_connectivity[edge_obj.left]]
    nodes_R = mesh.nodes[mesh.elements_connectivity[edge_obj.right]]
    
    uh_coeffs_L = u_linear(nodes_L[:, 0], nodes_L[:, 1])
    uh_coeffs_R = u_linear(nodes_R[:, 0], nodes_R[:, 1])
    uh_coeffs_combined = np.concatenate([uh_coeffs_L, uh_coeffs_R])
    
    assert_allclose(Ke_face @ uh_coeffs_combined, 0.0, atol=1e-12, err_msg="K_face @ [u_linear] should be 0")

def test_face_symmetry_interior(two_quad_q1_mesh):
    """Checks if the interior face matrix contribution is symmetric."""
    mesh = two_quad_q1_mesh
    int_eid, _ = pick_edges(mesh)
    Ke_face, _ = face_laplace(mesh, int_eid, alpha=10.0, quad_order=3, symmetry=1)
    assert_allclose(Ke_face, Ke_face.T, atol=1e-12, err_msg="Ke_face for interior edge should be symmetric")

def test_boundary_face_zero_dirichlet(two_quad_q1_mesh):
    """Checks that for a boundary face with u_D=0, the RHS vector Fe is zero."""
    mesh = two_quad_q1_mesh
    _, bdy_eid = pick_edges(mesh)

    Ke_face, Fe_face = face_laplace(mesh, bdy_eid, alpha=10.0, quad_order=3, dirichlet=lambda x, y: 0.0)

    assert_allclose(Fe_face, 0.0, atol=1e-12, err_msg="Fe_face on boundary should be 0 for u_D=0")
    assert_allclose(Ke_face, Ke_face.T, atol=1e-12, err_msg="Ke_face for boundary should be symmetric")

def test_boundary_face_nonzero_dirichlet(two_quad_q1_mesh):
    """Checks the RHS vector Fe for a non-zero Dirichlet condition u_D = x."""
    mesh = two_quad_q1_mesh
    
    right_boundary_edge = None
    for edge in mesh.edges_list:
        if edge.right is None: # Is a boundary edge
            p1_coords = mesh.nodes[edge.nodes[0]]
            if np.allclose(p1_coords[0], 1.0):
                right_boundary_edge = edge
                break
    
    assert right_boundary_edge is not None, "Could not find right boundary edge (x=1.0)"

    _, Fe_face_nonzero = face_laplace(
        mesh, right_boundary_edge.gid,
        alpha=10.0, quad_order=3,
        dirichlet=lambda x, y: x
    )
    assert np.linalg.norm(Fe_face_nonzero) > 1e-9, "Fe_face should be non-zero for u_D=x on the x=1 boundary"
